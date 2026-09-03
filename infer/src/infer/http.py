import asyncio
from collections.abc import AsyncIterator, Awaitable
from concurrent.futures import Future
from threading import BoundedSemaphore, Lock
from time import time
from uuid import uuid4

import orjson
from starlette.applications import Starlette
from starlette.requests import ClientDisconnect, Request
from starlette.responses import Response, StreamingResponse
from starlette.routing import Route

from infer.api import (
    ChatRequest,
    InvalidRequestError,
    create_chat_completion,
    parse_chat_request,
    submit_chat_request,
)
from infer.codec import CachedCodec, Codec
from infer.service import Completion, Service, ServiceOverloadedError, Submission

_STREAM_BUFFER_TOKENS = 4096
_STREAM_INTERVAL_TOKENS = 3
_TOKENIZER_CAPACITY = 128


def create_app(*, codec: Codec, service: Service, model_name: str) -> Starlette:
    codec = CachedCodec(codec)
    tokenizer_slots = BoundedSemaphore(_TOKENIZER_CAPACITY)

    async def chat_completions(request: Request) -> Response:
        return await _chat_completions(
            request,
            codec=codec,
            service=service,
            model_name=model_name,
            tokenizer_slots=tokenizer_slots,
        )

    return Starlette(
        routes=[Route("/v1/chat/completions", chat_completions, methods=["POST"])]
    )


async def _chat_completions(
    request: Request,
    *,
    codec: Codec,
    service: Service,
    model_name: str,
    tokenizer_slots: BoundedSemaphore,
) -> Response:
    try:
        payload = orjson.loads(await request.body())
        chat_request = parse_chat_request(payload, model_name)
        if chat_request.stream:
            return await _stream_response(
                chat_request,
                codec=codec,
                service=service,
                model_name=model_name,
                tokenizer_slots=tokenizer_slots,
            )
        completion = await _complete_or_disconnect(
            request,
            create_chat_completion(
                chat_request,
                codec=codec,
                service=service,
                model_name=model_name,
                tokenizer_slots=tokenizer_slots,
            ),
        )
    except ClientDisconnect:
        raise asyncio.CancelledError from None
    except orjson.JSONDecodeError:
        return _error_response(
            400,
            "request body must be valid JSON",
            "invalid_request_error",
        )
    except ServiceOverloadedError as error:
        return _error_response(429, str(error), "server_overloaded")
    except InvalidRequestError as error:
        return _error_response(400, str(error), "invalid_request_error")

    return _json_response(completion)


async def _complete_or_disconnect(
    request: Request,
    completion: Awaitable[dict[str, object]],
) -> dict[str, object]:
    completion_task = asyncio.create_task(completion)
    disconnect_task = asyncio.create_task(_wait_for_disconnect(request))
    try:
        done, _ = await asyncio.wait(
            (completion_task, disconnect_task),
            return_when=asyncio.FIRST_COMPLETED,
        )
        if completion_task in done:
            return await completion_task
        await disconnect_task
        completion_task.cancel()
        raise asyncio.CancelledError
    finally:
        for task in (completion_task, disconnect_task):
            if not task.done():
                task.cancel()
        await asyncio.gather(
            completion_task,
            disconnect_task,
            return_exceptions=True,
        )


async def _stream_response(
    request: ChatRequest,
    *,
    codec: Codec,
    service: Service,
    model_name: str,
    tokenizer_slots: BoundedSemaphore,
) -> StreamingResponse:
    loop = asyncio.get_running_loop()
    buffer = _StreamBuffer(loop)
    submission = await submit_chat_request(
        request,
        codec=codec,
        service=service,
        tokenizer_slots=tokenizer_slots,
        token_sink=buffer.enqueue,
    )
    submission.completion.add_done_callback(buffer.complete)
    return StreamingResponse(
        _stream_events(
            buffer,
            submission,
            request.include_usage,
            codec,
            model_name,
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _stream_events(
    buffer: "_StreamBuffer",
    submission: Submission,
    include_usage: bool,
    codec: Codec,
    model_name: str,
) -> AsyncIterator[bytes]:
    completion_id = f"chatcmpl-{uuid4().hex}"
    created = int(time())
    yield _stream_chunk(
        completion_id,
        created,
        model_name,
        {"role": "assistant", "content": ""},
    )
    decoder = codec.incremental_decoder()
    try:
        while True:
            item = await buffer.get()
            if isinstance(item, Future):
                completion = item.result()
                break
            content = decoder.decode(item)
            if content:
                yield _stream_chunk(
                    completion_id,
                    created,
                    model_name,
                    {"content": content},
                )

        if completion.finish_reason == "cancelled":
            return
        yield _stream_chunk(
            completion_id,
            created,
            model_name,
            {},
            finish_reason=("stop" if completion.finish_reason == "eos" else "length"),
        )
        if include_usage:
            yield _stream_chunk(
                completion_id,
                created,
                model_name,
                {},
                usage=(
                    submission.prompt_tokens,
                    len(completion.output_token_ids),
                    completion.cached_tokens,
                ),
            )
        yield b"data: [DONE]\n\n"
    finally:
        if not submission.completion.done():
            submission.cancel()


class _StreamBuffer:
    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self._queue: asyncio.Queue[tuple[int, ...] | Future[Completion]] = (
            asyncio.Queue(1)
        )
        self._pending: list[int] = []
        self._buffered_tokens = 0
        self._completion: Future[Completion] | None = None
        self._notified = False
        self._published_tokens = False
        self._lock = Lock()

    def enqueue(self, token_ids: tuple[int, ...]) -> bool:
        with self._lock:
            if self._buffered_tokens + len(token_ids) > _STREAM_BUFFER_TOKENS:
                return False
            self._pending.extend(token_ids)
            self._buffered_tokens += len(token_ids)
            self._notify()
        return True

    def complete(self, completion: Future[Completion]) -> None:
        with self._lock:
            self._completion = completion
            self._notify()

    async def get(self) -> tuple[int, ...] | Future[Completion]:
        item = await self._queue.get()
        with self._lock:
            if isinstance(item, tuple):
                self._buffered_tokens -= len(item)
            self._notified = False
            self._notify()
        return item

    def _notify(self) -> None:
        ready = (
            self._completion is not None
            or bool(self._pending)
            and (
                not self._published_tokens
                or len(self._pending) >= _STREAM_INTERVAL_TOKENS
            )
        )
        if ready and not self._notified:
            self._notified = True
            self._loop.call_soon_threadsafe(self._publish)

    def _publish(self) -> None:
        with self._lock:
            if self._pending:
                item: tuple[int, ...] | Future[Completion] = tuple(self._pending)
                self._pending.clear()
                self._published_tokens = True
            elif self._completion is not None:
                item = self._completion
                self._completion = None
            else:
                self._notified = False
                return
        self._queue.put_nowait(item)


def _stream_chunk(
    completion_id: str,
    created: int,
    model_name: str,
    delta: dict[str, str],
    *,
    finish_reason: str | None = None,
    usage: tuple[int, int, int] | None = None,
) -> bytes:
    payload: dict[str, object] = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_name,
    }
    if usage is None:
        payload["choices"] = [
            {
                "index": 0,
                "delta": delta,
                "logprobs": None,
                "finish_reason": finish_reason,
            }
        ]
    else:
        prompt_tokens, completion_tokens, cached_tokens = usage
        payload["choices"] = []
        payload["usage"] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "prompt_tokens_details": {"cached_tokens": cached_tokens},
        }
    return b"data: " + orjson.dumps(payload) + b"\n\n"


async def _wait_for_disconnect(request: Request) -> None:
    while True:
        if (await request.receive())["type"] == "http.disconnect":
            return


def _json_response(payload: object, status_code: int = 200) -> Response:
    return Response(
        orjson.dumps(payload),
        status_code=status_code,
        media_type="application/json",
    )


def _error_response(status_code: int, message: str, error_type: str) -> Response:
    return _json_response(
        {"error": {"message": message, "type": error_type}},
        status_code,
    )

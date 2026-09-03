import asyncio
from dataclasses import dataclass
from threading import BoundedSemaphore
from time import time
from uuid import uuid4

from infer.codec import Codec
from infer.service import Service, ServiceOverloadedError, Submission, TokenSink

_FIELDS = frozenset(
    (
        "model",
        "messages",
        "max_tokens",
        "temperature",
        "stream",
        "stream_options",
        "ignore_eos",
    )
)
_REQUIRED_FIELDS = frozenset(("model", "messages", "max_tokens"))
_MESSAGE_FIELDS = frozenset(("role", "content"))
_ROLES = frozenset(("system", "user", "assistant"))


class InvalidRequestError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ChatRequest:
    messages: tuple[dict[str, str], ...]
    max_tokens: int
    stream: bool
    ignore_eos: bool = False
    include_usage: bool = False


def parse_chat_request(payload: object, model_name: str) -> ChatRequest:
    if type(payload) is not dict:
        raise InvalidRequestError("request body must be a JSON object")
    if any(type(name) is not str for name in payload):
        raise InvalidRequestError("request field names must be strings")

    unknown = set(payload) - _FIELDS
    if unknown:
        raise InvalidRequestError(f"unsupported fields: {', '.join(sorted(unknown))}")
    missing = _REQUIRED_FIELDS - set(payload)
    if missing:
        raise InvalidRequestError(
            f"required fields missing: {', '.join(sorted(missing))}"
        )
    if payload["model"] != model_name:
        raise InvalidRequestError(f"model must be {model_name}")

    max_tokens = payload["max_tokens"]
    if type(max_tokens) is not int or max_tokens <= 0:
        raise InvalidRequestError("max_tokens must be a positive integer")

    temperature = payload.get("temperature", 0)
    if type(temperature) not in (int, float) or temperature != 0:
        raise InvalidRequestError("only temperature=0 is supported")
    stream = payload.get("stream", False)
    if type(stream) is not bool:
        raise InvalidRequestError("stream must be a boolean")
    include_usage = False
    if "stream_options" in payload:
        stream_options = payload["stream_options"]
        if type(stream_options) is not dict or set(stream_options) != {"include_usage"}:
            raise InvalidRequestError(
                "stream_options must be an object containing only include_usage"
            )
        include_usage = stream_options["include_usage"]
        if type(include_usage) is not bool:
            raise InvalidRequestError("stream_options.include_usage must be a boolean")
        if not stream:
            raise InvalidRequestError("stream_options requires stream=true")
    ignore_eos = payload.get("ignore_eos", False)
    if type(ignore_eos) is not bool:
        raise InvalidRequestError("ignore_eos must be a boolean")

    messages = payload["messages"]
    if type(messages) is not list or not messages:
        raise InvalidRequestError("messages must be a non-empty list")
    return ChatRequest(
        tuple(_parse_message(index, message) for index, message in enumerate(messages)),
        max_tokens,
        stream,
        ignore_eos,
        include_usage,
    )


async def submit_chat_request(
    request: ChatRequest,
    *,
    codec: Codec,
    service: Service,
    tokenizer_slots: BoundedSemaphore | None = None,
    token_sink: TokenSink | None = None,
) -> Submission:
    if tokenizer_slots is not None and not tokenizer_slots.acquire(blocking=False):
        raise ServiceOverloadedError("tokenizer capacity is exhausted")
    encoding = asyncio.create_task(
        asyncio.to_thread(codec.encode_messages, request.messages)
    )
    try:
        try:
            prompt_token_ids = tuple(await asyncio.shield(encoding))
        except asyncio.CancelledError:
            await asyncio.gather(encoding, return_exceptions=True)
            raise
    finally:
        if tokenizer_slots is not None:
            tokenizer_slots.release()

    try:
        if token_sink is None:
            return service.submit(
                prompt_token_ids,
                request.max_tokens,
                ignore_eos=request.ignore_eos,
            )
        return service.submit(
            prompt_token_ids,
            request.max_tokens,
            ignore_eos=request.ignore_eos,
            token_sink=token_sink,
        )
    except ValueError as error:
        raise InvalidRequestError(str(error)) from error


async def create_chat_completion(
    request: ChatRequest,
    *,
    codec: Codec,
    service: Service,
    model_name: str,
    tokenizer_slots: BoundedSemaphore | None = None,
) -> dict[str, object]:
    submission = await submit_chat_request(
        request,
        codec=codec,
        service=service,
        tokenizer_slots=tokenizer_slots,
    )
    try:
        completion = await asyncio.wrap_future(submission.completion)
    except asyncio.CancelledError:
        submission.cancel()
        raise

    if completion.finish_reason == "cancelled":
        raise asyncio.CancelledError

    return {
        "id": f"chatcmpl-{uuid4().hex}",
        "object": "chat.completion",
        "created": int(time()),
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": codec.decode(completion.output_token_ids),
                    "refusal": None,
                },
                "logprobs": None,
                "finish_reason": (
                    "stop" if completion.finish_reason == "eos" else "length"
                ),
            }
        ],
        "usage": {
            "prompt_tokens": submission.prompt_tokens,
            "completion_tokens": len(completion.output_token_ids),
            "total_tokens": submission.prompt_tokens + len(completion.output_token_ids),
            "prompt_tokens_details": {
                "cached_tokens": completion.cached_tokens,
            },
        },
    }


def _parse_message(index: int, message: object) -> dict[str, str]:
    if type(message) is not dict:
        raise InvalidRequestError(f"message {index} must be an object")
    if any(type(name) is not str for name in message):
        raise InvalidRequestError(f"message {index} field names must be strings")

    unknown = set(message) - _MESSAGE_FIELDS
    if unknown:
        raise InvalidRequestError(
            f"message {index} has unsupported fields: {', '.join(sorted(unknown))}"
        )
    if set(message) != _MESSAGE_FIELDS:
        raise InvalidRequestError(f"message {index} requires role and content")

    role = message["role"]
    content = message["content"]
    if type(role) is not str or role not in _ROLES:
        raise InvalidRequestError(f"message {index} has invalid role")
    if type(content) is not str:
        raise InvalidRequestError(f"message {index} content must be a string")
    return {"role": role, "content": content}

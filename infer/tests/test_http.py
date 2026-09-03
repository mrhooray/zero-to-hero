import asyncio
import unittest
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future
from threading import Event, Thread
from unittest.mock import Mock, patch

import orjson
from starlette.applications import Starlette

from infer.engine import DistributedEngine
from infer.http import _STREAM_BUFFER_TOKENS, _StreamBuffer, create_app
from infer.models.glm53_flash import MODEL_ID
from infer.models.glm53_flash.codec import EOS_TOKEN_IDS, GLM53Codec
from infer.models.glm53_flash.model import TOKENIZER_VOCAB_SIZE
from infer.service import Completion, Service, ServiceOverloadedError, Submission


def valid_request() -> dict[str, object]:
    return {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 2,
    }


def completed_submission(
    prompt_tokens: int = 2, cached_tokens: int = 0
) -> Submission:
    completion: Future[Completion] = Future()
    completion.set_result(Completion((20, 21), "length", cached_tokens))
    return Submission(completion, prompt_tokens, Event())


class Receiver:
    def __init__(self, *messages: dict[str, object]) -> None:
        self.messages = deque(messages)
        self.waiting = 0
        self.wait_started = asyncio.Event()

    async def __call__(self) -> dict[str, object]:
        if self.messages:
            return self.messages.popleft()

        self.waiting += 1
        self.wait_started.set()
        blocker: asyncio.Future[dict[str, object]] = (
            asyncio.get_running_loop().create_future()
        )
        try:
            return await blocker
        finally:
            self.waiting -= 1


async def call_app(
    app: Starlette,
    body: bytes,
    *later_messages: dict[str, object],
    more_body: bool = False,
    sent: list[dict[str, object]] | None = None,
) -> tuple[list[dict[str, object]], Receiver]:
    receiver = Receiver(
        {"type": "http.request", "body": body, "more_body": more_body},
        *later_messages,
    )
    sent = [] if sent is None else sent

    async def send(message: dict[str, object]) -> None:
        sent.append(message)

    await app(
        {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/v1/chat/completions",
            "raw_path": b"/v1/chat/completions",
            "query_string": b"",
            "root_path": "",
            "headers": [(b"content-type", b"application/json")],
            "client": ("test", 1),
            "server": ("test", 80),
        },
        receiver,
        send,
    )
    return sent, receiver


def response(sent: list[dict[str, object]]) -> tuple[int, object]:
    return sent[0]["status"], orjson.loads(sent[1]["body"])


def stream_events(sent: list[dict[str, object]]) -> list[object]:
    body = b"".join(
        message.get("body", b"")
        for message in sent
        if message["type"] == "http.response.body"
    )
    return [
        orjson.loads(line.removeprefix(b"data: "))
        if line != b"data: [DONE]"
        else "[DONE]"
        for line in body.splitlines()
        if line
    ]


class HTTPTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.codec = Mock(spec=GLM53Codec)
        self.codec.encode_messages.return_value = (10, 11)
        self.codec.decode.return_value = "hello back"
        self.service = Mock(spec=Service)
        self.app = create_app(
            codec=self.codec,
            service=self.service,
            model_name=MODEL_ID,
        )

    async def test_exposes_only_the_chat_completion_route(self) -> None:
        self.assertEqual(len(self.app.routes), 1)
        self.assertEqual(self.app.routes[0].path, "/v1/chat/completions")
        self.assertEqual(self.app.routes[0].methods, {"POST"})

    async def test_returns_orjson_chat_completion_and_stops_disconnect_watch(
        self,
    ) -> None:
        self.service.submit.return_value = completed_submission()

        sent, receiver = await call_app(self.app, orjson.dumps(valid_request()))

        status, payload = response(sent)
        self.assertEqual(status, 200)
        self.assertEqual(payload["model"], MODEL_ID)
        self.assertEqual(payload["choices"][0]["message"]["content"], "hello back")
        self.assertTrue(receiver.wait_started.is_set())
        self.assertEqual(receiver.waiting, 0)
        self.assertEqual(
            dict(sent[0]["headers"])[b"content-type"],
            b"application/json",
        )

    async def test_maps_invalid_json_and_request_validation_to_400(self) -> None:
        cases = (
            (b"{", "request body must be valid JSON"),
            (
                orjson.dumps({**valid_request(), "stream": 0}),
                "stream must be a boolean",
            ),
        )
        for body, message in cases:
            with self.subTest(message=message):
                sent, receiver = await call_app(self.app, body)

                self.assertEqual(
                    response(sent),
                    (
                        400,
                        {
                            "error": {
                                "message": message,
                                "type": "invalid_request_error",
                            }
                        },
                    ),
                )
                self.assertEqual(receiver.waiting, 0)

    async def test_maps_service_capacity_to_429(self) -> None:
        self.service.submit.side_effect = ServiceOverloadedError(
            "GLM service capacity is 64 requests"
        )

        sent, receiver = await call_app(self.app, orjson.dumps(valid_request()))

        self.assertEqual(
            response(sent),
            (
                429,
                {
                    "error": {
                        "message": "GLM service capacity is 64 requests",
                        "type": "server_overloaded",
                    }
                },
            ),
        )
        self.assertEqual(receiver.waiting, 0)

    async def test_bounds_tokenization_without_blocking_the_event_loop(self) -> None:
        started = Event()
        release = Event()

        def encode(_messages: object) -> tuple[int, ...]:
            started.set()
            release.wait()
            return (10, 11)

        self.codec.encode_messages.side_effect = encode
        with patch("infer.http._TOKENIZER_CAPACITY", 1):
            app = create_app(
                codec=self.codec,
                service=self.service,
                model_name=MODEL_ID,
            )

        first = asyncio.create_task(call_app(app, orjson.dumps(valid_request())))
        try:
            self.assertTrue(await asyncio.to_thread(started.wait, 1))
            sent, _ = await call_app(app, orjson.dumps(valid_request()))
            self.assertEqual(response(sent)[0], 429)
            self.service.submit.return_value = completed_submission()
        finally:
            release.set()
        await first

    async def test_maps_the_service_step_limit_to_400(self) -> None:
        engine = Mock(spec=DistributedEngine)
        service = Service(
            engine,
            capacity=1,
            vocab_size=TOKENIZER_VOCAB_SIZE,
            max_context_tokens=128,
            eos_token_ids=EOS_TOKEN_IDS,
        )
        self.codec.encode_messages.return_value = tuple(range(128))
        app = create_app(codec=self.codec, service=service, model_name=MODEL_ID)

        sent, _ = await call_app(app, orjson.dumps(valid_request()))

        self.assertEqual(
            response(sent),
            (
                400,
                {
                    "error": {
                        "message": (
                            "prompt plus decode steps exceeds the model context "
                            "of 128 tokens"
                        ),
                        "type": "invalid_request_error",
                    }
                },
            ),
        )
        engine.submit.assert_not_called()

    async def test_streams_each_accepted_token_batch_and_terminal_chunk(self) -> None:
        decoder = Mock()
        decoder.decode.return_value = "hello back"
        self.codec.incremental_decoder.return_value = decoder

        def submit(
            _prompt_token_ids: tuple[int, ...],
            _max_output_tokens: int,
            *,
            ignore_eos: bool,
            token_sink: Callable[[tuple[int, ...]], bool],
        ) -> Submission:
            self.assertFalse(ignore_eos)
            completion: Future[Completion] = Future()
            submission = Submission(completion, 2, Event())
            self.assertTrue(token_sink((20, 21)))
            completion.set_result(Completion((20, 21), "length"))
            return submission

        self.service.submit.side_effect = submit

        sent, receiver = await call_app(
            self.app,
            orjson.dumps({**valid_request(), "stream": True}),
        )

        events = stream_events(sent)
        self.assertEqual(sent[0]["status"], 200)
        self.assertEqual(events[0]["choices"][0]["delta"]["role"], "assistant")
        self.assertEqual(events[1]["choices"][0]["delta"]["content"], "hello back")
        self.assertEqual(events[2]["choices"][0]["finish_reason"], "length")
        self.assertEqual(events[3], "[DONE]")
        decoder.decode.assert_called_once_with((20, 21))
        self.assertEqual(receiver.waiting, 0)

    async def test_stream_buffer_coalesces_pending_tokens(self) -> None:
        decoder = Mock()
        decoder.decode.side_effect = lambda token_ids: "".join(
            {20: "a", 21: "b", 22: "c"}[token_id] for token_id in token_ids
        )
        self.codec.incremental_decoder.return_value = decoder
        accepted: list[bool] = []

        def submit(
            _prompt_token_ids: tuple[int, ...],
            _max_output_tokens: int,
            *,
            ignore_eos: bool,
            token_sink: Callable[[tuple[int, ...]], bool],
        ) -> Submission:
            completion: Future[Completion] = Future()
            submission = Submission(completion, 2, Event())
            for token_id in (20, 21, 22):
                accepted.append(token_sink((token_id,)))
            completion.set_result(
                Completion(
                    (20, 21, 22),
                    "length" if all(accepted) else "cancelled",
                )
            )
            return submission

        self.service.submit.side_effect = submit

        sent, _ = await call_app(
            self.app,
            orjson.dumps(
                {
                    **valid_request(),
                    "max_tokens": 3,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                }
            ),
        )

        events = stream_events(sent)
        content = "".join(
            event["choices"][0]["delta"].get("content", "")
            for event in events[:-1]
            if event["choices"]
        )
        self.assertEqual(accepted, [True, True, True])
        self.assertEqual(content, "abc")
        decoder.decode.assert_called_once_with((20, 21, 22))
        self.assertEqual(events[-3]["choices"][0]["finish_reason"], "length")
        self.assertEqual(events[-2]["usage"]["completion_tokens"], 3)
        self.assertEqual(events[-1], "[DONE]")

    async def test_stream_buffer_preserves_threaded_backlog_before_completion(
        self,
    ) -> None:
        buffer = _StreamBuffer(asyncio.get_running_loop())
        token_ids = tuple(range(64))
        completion: Future[Completion] = Future()
        completion.set_result(Completion(token_ids, "length"))

        def produce() -> None:
            for token_id in token_ids:
                self.assertTrue(buffer.enqueue((token_id,)))
            buffer.complete(completion)

        producer = Thread(target=produce)
        producer.start()
        producer.join()

        self.assertEqual(await buffer.get(), token_ids)
        self.assertIs(await buffer.get(), completion)

    async def test_stream_buffer_emits_first_then_groups_three_and_flushes(
        self,
    ) -> None:
        buffer = _StreamBuffer(asyncio.get_running_loop())
        completion: Future[Completion] = Future()
        completion.set_result(Completion(tuple(range(6)), "length"))

        self.assertTrue(buffer.enqueue((0,)))
        self.assertEqual(await buffer.get(), (0,))
        self.assertTrue(buffer.enqueue((1,)))
        self.assertTrue(buffer.enqueue((2,)))
        await asyncio.sleep(0)
        self.assertTrue(buffer._queue.empty())
        self.assertTrue(buffer.enqueue((3,)))
        self.assertEqual(await buffer.get(), (1, 2, 3))
        self.assertTrue(buffer.enqueue((4,)))
        self.assertTrue(buffer.enqueue((5,)))
        buffer.complete(completion)
        self.assertEqual(await buffer.get(), (4, 5))
        self.assertIs(await buffer.get(), completion)

    async def test_stream_buffer_rejects_excess_backlog(self) -> None:
        buffer = _StreamBuffer(asyncio.get_running_loop())

        self.assertTrue(buffer.enqueue(tuple(range(_STREAM_BUFFER_TOKENS))))
        await asyncio.sleep(0)
        self.assertFalse(buffer.enqueue((0,)))
        self.assertEqual(
            await buffer.get(),
            tuple(range(_STREAM_BUFFER_TOKENS)),
        )

    async def test_streams_requested_usage_after_the_finish_chunk(self) -> None:
        self.service.submit.return_value = completed_submission(
            prompt_tokens=3, cached_tokens=2
        )
        self.codec.incremental_decoder.return_value.decode.return_value = ""

        sent, _ = await call_app(
            self.app,
            orjson.dumps(
                {
                    **valid_request(),
                    "stream": True,
                    "stream_options": {"include_usage": True},
                }
            ),
        )

        events = stream_events(sent)
        self.assertEqual(events[-3]["choices"][0]["finish_reason"], "length")
        self.assertEqual(
            events[-2],
            {
                "id": events[0]["id"],
                "object": "chat.completion.chunk",
                "created": events[0]["created"],
                "model": MODEL_ID,
                "choices": [],
                "usage": {
                    "prompt_tokens": 3,
                    "completion_tokens": 2,
                    "total_tokens": 5,
                    "prompt_tokens_details": {"cached_tokens": 2},
                },
            },
        )
        self.assertEqual(events[-1], "[DONE]")

    async def test_stream_disconnect_cancels_the_submission(self) -> None:
        completion: Future[Completion] = Future()
        completion.set_running_or_notify_cancel()
        submission = Mock(spec=Submission)
        submission.completion = completion
        self.service.submit.return_value = submission

        sent, receiver = await call_app(
            self.app,
            orjson.dumps({**valid_request(), "stream": True}),
            {"type": "http.disconnect"},
        )

        submission.cancel.assert_called_once_with()
        self.assertEqual(sent[0]["status"], 200)
        self.assertEqual(receiver.waiting, 0)

    async def test_disconnect_while_reading_body_terminates_without_inference(
        self,
    ) -> None:
        sent: list[dict[str, object]] = []

        with self.assertRaises(asyncio.CancelledError):
            await call_app(
                self.app,
                b"{",
                {"type": "http.disconnect"},
                more_body=True,
                sent=sent,
            )

        self.assertEqual(sent, [])
        self.codec.encode_messages.assert_not_called()
        self.service.submit.assert_not_called()

    async def test_disconnect_cancels_inference_without_a_response_or_task_leak(
        self,
    ) -> None:
        completion: Future[Completion] = Future()
        completion.set_running_or_notify_cancel()
        submission = Mock(spec=Submission)
        submission.completion = completion
        submitted = asyncio.Event()

        def submit(*_args: object, **_kwargs: object) -> object:
            submitted.set()
            return submission

        async def disconnect(_request: object) -> None:
            await submitted.wait()

        self.service.submit.side_effect = submit
        tasks_before = asyncio.all_tasks()

        with (
            patch("infer.http._wait_for_disconnect", new=disconnect),
            self.assertRaises(asyncio.CancelledError),
        ):
            await call_app(self.app, orjson.dumps(valid_request()))

        submission.cancel.assert_called_once_with()
        self.assertEqual(asyncio.all_tasks(), tasks_before)

    async def test_fatal_inference_errors_propagate_to_the_lifecycle_owner(
        self,
    ) -> None:
        completion: Future[Completion] = Future()
        completion.set_exception(ValueError("worker produced an invalid result"))
        self.service.submit.return_value = Submission(completion, 2, Event())

        with self.assertRaisesRegex(ValueError, "worker produced an invalid result"):
            await call_app(self.app, orjson.dumps(valid_request()))

    async def test_fatal_completion_wins_a_simultaneous_disconnect(self) -> None:
        ready = asyncio.Event()
        ready.set()

        async def fail_completion(*args: object, **kwargs: object) -> object:
            await ready.wait()
            raise ValueError("worker produced an invalid result")

        async def disconnect(*args: object, **kwargs: object) -> None:
            await ready.wait()

        with (
            patch("infer.http.create_chat_completion", new=fail_completion),
            patch("infer.http._wait_for_disconnect", new=disconnect),
            self.assertRaisesRegex(ValueError, "worker produced an invalid result"),
        ):
            await call_app(self.app, orjson.dumps(valid_request()))

    async def test_successful_completion_wins_a_simultaneous_disconnect(self) -> None:
        ready = asyncio.Event()
        ready.set()
        completion = {"result": "complete"}

        async def complete(*args: object, **kwargs: object) -> object:
            await ready.wait()
            return completion

        async def disconnect(*args: object, **kwargs: object) -> None:
            await ready.wait()

        with (
            patch("infer.http.create_chat_completion", new=complete),
            patch("infer.http._wait_for_disconnect", new=disconnect),
        ):
            sent, _ = await call_app(self.app, orjson.dumps(valid_request()))

        self.assertEqual(response(sent), (200, completion))


if __name__ == "__main__":
    unittest.main()

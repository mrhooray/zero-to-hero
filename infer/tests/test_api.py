import asyncio
import unittest
from concurrent.futures import Future
from threading import Event
from unittest.mock import Mock, patch
from uuid import UUID

from infer.api import InvalidRequestError, create_chat_completion, parse_chat_request
from infer.models.glm53_flash import MODEL_ID
from infer.models.glm53_flash.codec import GLM53Codec
from infer.service import Completion, Service, Submission


def completed_submission(
    output_token_ids: tuple[int, ...],
    finish_reason: str,
) -> Submission:
    completion: Future[Completion] = Future()
    completion.set_result(Completion(output_token_ids, finish_reason))
    return Submission(completion, 2, Event())


def valid_request(**updates: object) -> dict[str, object]:
    request: dict[str, object] = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 2,
        "temperature": 0,
        "stream": False,
    }
    request.update(updates)
    return request


class ChatCompletionAPITest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.codec = Mock(spec=GLM53Codec)
        self.codec.encode_messages.return_value = [10, 11]
        self.codec.decode.return_value = "hello back"
        self.service = Mock(spec=Service)

    async def test_maps_eos_to_an_openai_chat_completion(self) -> None:
        self.service.submit.return_value = completed_submission((20, 154_820), "eos")
        completion_id = UUID("12345678-1234-5678-1234-567812345678")

        with (
            patch("infer.api.time", return_value=1_725_000_000.9),
            patch("infer.api.uuid4", return_value=completion_id),
        ):
            response = await create_chat_completion(
                parse_chat_request(valid_request(), MODEL_ID),
                codec=self.codec,
                service=self.service,
                model_name=MODEL_ID,
            )

        self.codec.encode_messages.assert_called_once_with(
            ({"role": "user", "content": "hello"},)
        )
        self.service.submit.assert_called_once_with((10, 11), 2, ignore_eos=False)
        self.codec.decode.assert_called_once_with((20, 154_820))
        self.assertEqual(
            response,
            {
                "id": "chatcmpl-12345678123456781234567812345678",
                "object": "chat.completion",
                "created": 1_725_000_000,
                "model": MODEL_ID,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "hello back",
                            "refusal": None,
                        },
                        "logprobs": None,
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 2,
                    "completion_tokens": 2,
                    "total_tokens": 4,
                    "prompt_tokens_details": {"cached_tokens": 0},
                },
            },
        )
        self.assertNotIn("token_ids", repr(response).lower())

    async def test_maps_length_and_defaults_greedy_nonstreaming_fields(self) -> None:
        self.service.submit.return_value = completed_submission((20, 21), "length")
        request = valid_request()
        del request["temperature"]
        del request["stream"]

        response = await create_chat_completion(
            parse_chat_request(request, MODEL_ID),
            codec=self.codec,
            service=self.service,
            model_name=MODEL_ID,
        )

        self.assertEqual(response["choices"][0]["finish_reason"], "length")

    async def test_forwards_ignore_eos_to_the_service(self) -> None:
        self.service.submit.return_value = completed_submission((20, 21), "length")

        await create_chat_completion(
            parse_chat_request(valid_request(ignore_eos=True), MODEL_ID),
            codec=self.codec,
            service=self.service,
            model_name=MODEL_ID,
        )

        self.service.submit.assert_called_once_with((10, 11), 2, ignore_eos=True)

    async def test_parses_stream_usage_option(self) -> None:
        enabled = parse_chat_request(
            valid_request(stream=True, stream_options={"include_usage": True}),
            MODEL_ID,
        )
        disabled = parse_chat_request(
            valid_request(stream=True, stream_options={"include_usage": False}),
            MODEL_ID,
        )

        self.assertTrue(enabled.include_usage)
        self.assertFalse(disabled.include_usage)

    async def test_rejects_invalid_or_unsupported_top_level_fields(self) -> None:
        cases = (
            (None, "JSON object"),
            ({}, "required fields"),
            (valid_request(model="other"), "model"),
            (valid_request(max_tokens=0), "positive integer"),
            (valid_request(max_tokens=True), "positive integer"),
            (valid_request(temperature=0.1), "temperature=0"),
            (valid_request(temperature=False), "temperature=0"),
            (valid_request(stream=0), "stream must be a boolean"),
            (
                valid_request(stream=True, stream_options={}),
                "stream_options must be an object",
            ),
            (
                valid_request(
                    stream=True,
                    stream_options={"include_usage": 1},
                ),
                "stream_options.include_usage must be a boolean",
            ),
            (
                valid_request(stream_options={"include_usage": False}),
                "stream_options requires stream=true",
            ),
            (valid_request(ignore_eos=1), "ignore_eos must be a boolean"),
            (valid_request(tools=[]), "unsupported fields: tools"),
            (valid_request(logprobs=True), "unsupported fields: logprobs"),
            ({**valid_request(), 1: "invalid"}, "field names must be strings"),
        )

        for request, error in cases:
            with (
                self.subTest(error=error),
                self.assertRaisesRegex(InvalidRequestError, error),
            ):
                parse_chat_request(request, MODEL_ID)

        self.codec.encode_messages.assert_not_called()
        self.service.submit.assert_not_called()

    async def test_rejects_messages_outside_the_minimal_text_contract(self) -> None:
        cases = (
            ([], "non-empty list"),
            ("hello", "non-empty list"),
            (["hello"], "message 0 must be an object"),
            ([{"role": "user"}], "message 0 requires role and content"),
            (
                [{"role": "user", "content": "hello", "name": "alice"}],
                "message 0 has unsupported fields: name",
            ),
            ([{1: "user", "content": "hello"}], "field names must be strings"),
            ([{"role": "tool", "content": "hello"}], "message 0 has invalid role"),
            ([{"role": [], "content": "hello"}], "message 0 has invalid role"),
            ([{"role": "user", "content": ["hello"]}], "message 0 content"),
        )

        for messages, error in cases:
            with (
                self.subTest(error=error),
                self.assertRaisesRegex(InvalidRequestError, error),
            ):
                parse_chat_request(valid_request(messages=messages), MODEL_ID)

        self.codec.encode_messages.assert_not_called()
        self.service.submit.assert_not_called()

    async def test_caller_cancellation_cancels_the_service_submission(self) -> None:
        completion: Future[Completion] = Future()
        completion.set_running_or_notify_cancel()
        submission = Mock(spec=Submission)
        submission.completion = completion
        self.service.submit.return_value = submission
        task = asyncio.create_task(
            create_chat_completion(
                parse_chat_request(valid_request(), MODEL_ID),
                codec=self.codec,
                service=self.service,
                model_name=MODEL_ID,
            )
        )
        while not self.service.submit.called:
            await asyncio.sleep(0)

        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task

        submission.cancel.assert_called_once_with()

    async def test_service_cancellation_never_returns_partial_text(self) -> None:
        self.service.submit.return_value = completed_submission((20,), "cancelled")

        with self.assertRaises(asyncio.CancelledError):
            await create_chat_completion(
                parse_chat_request(valid_request(), MODEL_ID),
                codec=self.codec,
                service=self.service,
                model_name=MODEL_ID,
            )

        self.codec.decode.assert_not_called()


if __name__ == "__main__":
    unittest.main()

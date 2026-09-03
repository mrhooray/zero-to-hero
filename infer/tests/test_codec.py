import unittest
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from unittest.mock import Mock, call, patch

from tokenizers import Tokenizer

from infer.codec import CachedCodec, IncrementalDecoder


class IncrementalDecoderTest(unittest.TestCase):
    def test_decodes_each_accepted_token_in_order(self) -> None:
        tokenizer = Mock(spec=Tokenizer)
        stream = Mock()
        stream.step.side_effect = (None, "hel", "lo")

        with patch("infer.codec.DecodeStream", return_value=stream) as constructor:
            decoder = IncrementalDecoder(tokenizer)
            self.assertEqual(decoder.decode((10, 11, 12)), "hello")

        constructor.assert_called_once_with(skip_special_tokens=True)
        self.assertEqual(
            stream.step.call_args_list,
            [call(tokenizer, 10), call(tokenizer, 11), call(tokenizer, 12)],
        )


class CachedCodecTest(unittest.TestCase):
    def test_reuses_exact_messages_and_forwards_decode(self) -> None:
        codec = Mock()
        codec.encode_messages.return_value = [1, 2]
        cached = CachedCodec(codec)
        messages = ({"role": "user", "content": "hello"},)

        self.assertEqual(cached.encode_messages(messages), (1, 2))
        self.assertEqual(cached.encode_messages(messages), (1, 2))
        self.assertEqual(cached.decode((1,)), codec.decode.return_value)
        codec.encode_messages.assert_called_once_with(messages)

    @patch("infer.codec._TOKEN_CACHE_ENTRIES", 1)
    @patch("infer.codec._TOKEN_CACHE_ENTRY_TOKENS", 1)
    def test_bounds_entries_and_tokens(self) -> None:
        codec = Mock()
        codec.encode_messages.side_effect = ([1], [2], [1], [3, 4], [3, 4])
        cached = CachedCodec(codec)
        a = ({"role": "user", "content": "a"},)
        b = ({"role": "user", "content": "b"},)
        oversized = ({"role": "user", "content": "oversized"},)

        for messages in (a, b, a, oversized, oversized):
            cached.encode_messages(messages)
        self.assertEqual(codec.encode_messages.call_count, 5)

    @patch("infer.codec._TOKEN_CACHE_ENTRIES", 2)
    def test_duplicate_miss_refreshes_lru_recency(self) -> None:
        first_started = Event()
        release_first = Event()
        calls: list[str] = []

        def encode(messages: tuple[dict[str, str], ...]) -> list[int]:
            content = messages[0]["content"]
            calls.append(content)
            if content == "a" and calls.count("a") == 1:
                first_started.set()
                release_first.wait()
            return [ord(content)]

        codec = Mock()
        codec.encode_messages.side_effect = encode
        cached = CachedCodec(codec)
        a = ({"role": "user", "content": "a"},)
        b = ({"role": "user", "content": "b"},)
        c = ({"role": "user", "content": "c"},)

        with ThreadPoolExecutor(max_workers=1) as executor:
            first = executor.submit(cached.encode_messages, a)
            try:
                self.assertTrue(first_started.wait(1))
                self.assertEqual(cached.encode_messages(a), (97,))
                self.assertEqual(cached.encode_messages(b), (98,))
            finally:
                release_first.set()
            self.assertEqual(first.result(1), (97,))

        self.assertEqual(cached.encode_messages(c), (99,))
        self.assertEqual(cached.encode_messages(a), (97,))
        self.assertEqual(calls.count("a"), 2)


if __name__ == "__main__":
    unittest.main()

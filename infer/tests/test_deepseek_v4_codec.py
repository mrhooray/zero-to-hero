import hashlib
import tempfile
import unittest
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from infer.models.deepseek_v4_flash import codec as codec_module
from infer.models.deepseek_v4_flash import model as deepseek_v4_flash

REFERENCE_PROMPTS = (
    (
        ({"role": "user", "content": "Hello, 世界"},),
        "<｜begin▁of▁sentence｜><｜User｜>Hello, 世界<｜Assistant｜></think>",
        "bd39d85db7e9ab1017d904e0cfd93abf86921ea1802eb5e90be234caddc65521",
        (0, 128_803, 19_923, 14, 223, 3_427, 128_804, 128_822),
    ),
    (
        (
            {"role": "system", "content": "You are concise."},
            {"role": "user", "content": "Answer with one word: sky color?"},
        ),
        (
            "<｜begin▁of▁sentence｜>You are concise.<｜User｜>"
            "Answer with one word: sky color?<｜Assistant｜></think>"
        ),
        "e782393c4e2320f7e71ff0a2a5f0ef2608c33e8ac83cf6c52a548cc8d4c24fdf",
        (
            0,
            3_476,
            477,
            47_468,
            16,
            128_803,
            7_805,
            418,
            834,
            2_004,
            28,
            12_709,
            3_605,
            33,
            128_804,
            128_822,
        ),
    ),
    (
        (
            {"role": "system", "content": "S"},
            {"role": "user", "content": "U1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "U2"},
        ),
        (
            "<｜begin▁of▁sentence｜>S<｜User｜>U1<｜Assistant｜></think>"
            "A1<｜end▁of▁sentence｜><｜User｜>U2<｜Assistant｜></think>"
        ),
        "acfef7e6accf8d51ca4196b9540c7994ef53d09d91c4fcf60552132b66ccbba3",
        (
            0,
            53,
            128_803,
            55,
            19,
            128_804,
            128_822,
            35,
            19,
            1,
            128_803,
            55,
            20,
            128_804,
            128_822,
        ),
    ),
)

_SPECIAL_TOKEN_IDS = {
    "<｜begin▁of▁sentence｜>": 0,
    "<｜end▁of▁sentence｜>": 1,
    "<｜User｜>": 128_803,
    "<｜Assistant｜>": 128_804,
    "</think>": 128_822,
}


class FakeTokenizer:
    def __init__(self) -> None:
        self.encoded = []
        self.decoded = []

    def get_vocab_size(self, *, with_added_tokens: bool) -> int:
        if not with_added_tokens:
            raise AssertionError("added tokens must be included")
        return deepseek_v4_flash.VOCAB_SIZE

    def token_to_id(self, token: str) -> int | None:
        return _SPECIAL_TOKEN_IDS.get(token)

    def encode(self, text: str, *, add_special_tokens: bool) -> SimpleNamespace:
        self.encoded.append((text, add_special_tokens))
        token_ids = next(
            ids for _, prompt, _, ids in REFERENCE_PROMPTS if prompt == text
        )
        return SimpleNamespace(ids=list(token_ids))

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str:
        self.decoded.append((token_ids, skip_special_tokens))
        return "decoded"


class DeepSeekV4CodecTest(unittest.TestCase):
    def test_exact_pinned_reference_prompts_and_token_ids(self) -> None:
        tokenizer = FakeTokenizer()
        with codec_fixture(tokenizer) as codec:
            for messages, expected, expected_hash, expected_ids in REFERENCE_PROMPTS:
                with self.subTest(messages=messages):
                    rendered = codec.render(messages)
                    self.assertEqual(rendered, expected)
                    self.assertEqual(
                        hashlib.sha256(rendered.encode()).hexdigest(), expected_hash
                    )
                    self.assertEqual(
                        codec.encode_messages(messages), list(expected_ids)
                    )
        self.assertTrue(all(not special for _, special in tokenizer.encoded))

    def test_decode_preserves_the_requested_special_token_policy(self) -> None:
        tokenizer = FakeTokenizer()
        with codec_fixture(tokenizer) as codec:
            self.assertEqual(codec.decode([1], skip_special_tokens=False), "decoded")
        self.assertEqual(tokenizer.decoded, [([1], False)])

    def test_codec_artifacts_fail_closed_on_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            hashes = write_fixture(root)
            for relative in hashes:
                with self.subTest(relative=relative):
                    path = root / relative
                    original = path.read_bytes()
                    path.write_bytes(b"wrong")
                    with (
                        patch_hashes(hashes),
                        patch.object(codec_module.Tokenizer, "from_str") as from_str,
                        self.assertRaisesRegex(RuntimeError, path.name),
                    ):
                        codec_module.DeepSeekV4Codec(root)
                    from_str.assert_not_called()
                    path.write_bytes(original)

    def test_tokenizer_contract_fails_closed(self) -> None:
        tokenizer = FakeTokenizer()
        tokenizer.get_vocab_size = Mock(
            return_value=deepseek_v4_flash.VOCAB_SIZE - 1
        )
        with (
            self.assertRaisesRegex(RuntimeError, "vocabulary size"),
            codec_fixture(tokenizer),
        ):
            pass

        tokenizer = FakeTokenizer()
        tokenizer.token_to_id = Mock(return_value=None)
        with (
            self.assertRaisesRegex(RuntimeError, "special tokens"),
            codec_fixture(tokenizer),
        ):
            pass

    def test_invalid_messages_and_token_ids_are_rejected(self) -> None:
        with codec_fixture(FakeTokenizer()) as codec:
            with self.assertRaisesRegex(ValueError, "invalid role"):
                codec.render(({"role": "tool", "content": "result"},))
            with self.assertRaisesRegex(ValueError, "content must be a string"):
                codec.render(({"role": "user", "content": None},))
            for token_ids in ([True], [-1], [deepseek_v4_flash.VOCAB_SIZE]):
                with (
                    self.subTest(token_ids=token_ids),
                    self.assertRaisesRegex(ValueError, "within the vocabulary"),
                ):
                    codec.decode(token_ids)

    def test_exact_eos_set(self) -> None:
        self.assertEqual(codec_module.EOS_TOKEN_IDS, frozenset((1,)))


@contextmanager
def codec_fixture(tokenizer: FakeTokenizer) -> Iterator[codec_module.DeepSeekV4Codec]:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        hashes = write_fixture(root)
        with (
            patch_hashes(hashes),
            patch.object(codec_module.Tokenizer, "from_str", return_value=tokenizer),
        ):
            yield codec_module.DeepSeekV4Codec(root)


def write_fixture(root: Path) -> dict[Path, str]:
    files = {
        Path("tokenizer.json"): b'{"fixture": true}',
        Path("tokenizer_config.json"): b'{"model_max_length": 1048576}',
        Path("generation_config.json"): b'{"bos_token_id": 0, "eos_token_id": 1}',
        Path("encoding/encoding_dsv4.py"): b"# pinned fixture\n",
    }
    hashes = {}
    for relative, payload in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        hashes[relative] = hashlib.sha256(payload).hexdigest()
    return hashes


@contextmanager
def patch_hashes(hashes: dict[Path, str]) -> Iterator[None]:
    with (
        patch.object(
            codec_module, "TOKENIZER_JSON_SHA256", hashes[Path("tokenizer.json")]
        ),
        patch.object(
            codec_module,
            "TOKENIZER_CONFIG_SHA256",
            hashes[Path("tokenizer_config.json")],
        ),
        patch.object(
            codec_module,
            "GENERATION_CONFIG_SHA256",
            hashes[Path("generation_config.json")],
        ),
        patch.object(
            codec_module,
            "ENCODING_REFERENCE_SHA256",
            hashes[Path("encoding/encoding_dsv4.py")],
        ),
    ):
        yield


if __name__ == "__main__":
    unittest.main()

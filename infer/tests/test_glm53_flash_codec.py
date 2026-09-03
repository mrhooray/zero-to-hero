import hashlib
import json
import tempfile
import unittest
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import jinja2

from infer.models.glm53_flash import codec as codec_module

PROMPT = "Answer with one word: What is two plus two?"
RENDERED_PROMPT = (
    "[gMASK]<sop><|system|>Reasoning Effort: Max<|user|>"
    "Answer with one word: What is two plus two?<|assistant|><think>"
)
RENDERED_PROMPT_SHA256 = (
    "84844f2d6e77285c9b215a7e9363cf254178c8cd9032c84c159b585d52e62903"
)
PROMPT_TOKEN_IDS = [
    154_822,
    154_824,
    154_826,
    25_062,
    287,
    29_905,
    371,
    25,
    7_487,
    154_827,
    16_127,
    448,
    825,
    3_409,
    25,
    3_555,
    374,
    1_378,
    5_519,
    1_378,
    30,
    154_828,
    154_841,
]
FIXTURE_TEMPLATE = """[gMASK]<sop>
{%- set effort = reasoning_effort if reasoning_effort in ['low', 'high'] else 'max' -%}
<|system|>Reasoning Effort: {{ effort | capitalize }}
{%- for message in messages -%}
<|{{ message.role }}|>{{ message.content }}
{%- endfor -%}
{%- if add_generation_prompt -%}<|assistant|>{{- '<think>' -}}{%- endif -%}
"""


class FakeTokenizer:
    def __init__(self) -> None:
        self.encoded = []
        self.decoded = []

    def get_vocab_size(self, *, with_added_tokens: bool) -> int:
        if not with_added_tokens:
            raise AssertionError("added tokens must be included")
        return codec_module.TOKENIZER_VOCAB_SIZE

    def encode(self, text: str, *, add_special_tokens: bool) -> SimpleNamespace:
        self.encoded.append((text, add_special_tokens))
        return SimpleNamespace(ids=list(PROMPT_TOKEN_IDS))

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str:
        self.decoded.append((token_ids, skip_special_tokens))
        return "decoded"


class GLM53CodecTest(unittest.TestCase):
    def test_exact_prompt_fixture(self) -> None:
        fake_tokenizer = FakeTokenizer()
        with codec_fixture(fake_tokenizer) as codec:
            messages = [{"role": "user", "content": PROMPT}]

            rendered = codec.render(messages)

            self.assertEqual(rendered, RENDERED_PROMPT)
            self.assertEqual(
                hashlib.sha256(rendered.encode()).hexdigest(),
                RENDERED_PROMPT_SHA256,
            )
            self.assertEqual(codec.encode_messages(messages), PROMPT_TOKEN_IDS)
            self.assertEqual(fake_tokenizer.encoded, [(RENDERED_PROMPT, False)])

    def test_decode_preserves_the_requested_special_token_policy(self) -> None:
        fake_tokenizer = FakeTokenizer()
        with codec_fixture(fake_tokenizer) as codec:
            self.assertEqual(
                codec.decode([154_841], skip_special_tokens=False), "decoded"
            )
            self.assertEqual(
                fake_tokenizer.decoded,
                [([154_841], False)],
            )

    def test_codec_artifacts_fail_closed_on_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "tokenizer.json").write_text("wrong")
            (root / "chat_template.jinja").write_text(FIXTURE_TEMPLATE)

            with (
                self.assertRaisesRegex(RuntimeError, "tokenizer.json"),
                patch.object(codec_module.Tokenizer, "from_str") as from_str,
            ):
                codec_module.GLM53Codec(root)

            from_str.assert_not_called()

            tokenizer_bytes = (root / "tokenizer.json").read_bytes()
            with (
                self.assertRaisesRegex(RuntimeError, "chat_template.jinja"),
                patch.object(
                    codec_module,
                    "TOKENIZER_JSON_SHA256",
                    hashlib.sha256(tokenizer_bytes).hexdigest(),
                ),
                patch.object(codec_module.Tokenizer, "from_str") as from_str,
            ):
                codec_module.GLM53Codec(root)

            from_str.assert_not_called()

    def test_tokenizer_vocabulary_size_fails_closed(self) -> None:
        fake_tokenizer = FakeTokenizer()
        fake_tokenizer.get_vocab_size = Mock(
            return_value=codec_module.TOKENIZER_VOCAB_SIZE - 1
        )

        with (
            self.assertRaisesRegex(RuntimeError, "vocabulary size"),
            codec_fixture(fake_tokenizer),
        ):
            pass

    def test_invalid_token_ids_are_rejected(self) -> None:
        with codec_fixture(FakeTokenizer()) as codec:
            for token_ids in (
                [True],
                [-1],
                [codec_module.TOKENIZER_VOCAB_SIZE],
            ):
                with (
                    self.subTest(token_ids=token_ids),
                    self.assertRaisesRegex(ValueError, "within the vocabulary"),
                ):
                    codec.decode(token_ids)

    def test_exact_eos_set(self) -> None:
        self.assertEqual(
            codec_module.EOS_TOKEN_IDS,
            frozenset((154_820, 154_827, 154_829)),
        )


class ChatTemplateEnvironmentTest(unittest.TestCase):
    def test_tojson_is_unicode_preserving_and_does_not_html_escape(self) -> None:
        template = codec_module._compile_chat_template(
            "{% for value in values %}{{ value | tojson }}{% break %}{% endfor %}"
        )

        self.assertEqual(template.render(values=["<工具>", "unused"]), '"<工具>"')

    def test_raise_exception_raises_a_template_error(self) -> None:
        template = codec_module._compile_chat_template(
            "{{ raise_exception('invalid message') }}"
        )

        with self.assertRaisesRegex(jinja2.TemplateError, "invalid message"):
            template.render()


@contextmanager
def codec_fixture(tokenizer: FakeTokenizer) -> Iterator[codec_module.GLM53Codec]:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        tokenizer_bytes = json.dumps({"fixture": True}).encode()
        template_bytes = FIXTURE_TEMPLATE.encode()
        (root / "tokenizer.json").write_bytes(tokenizer_bytes)
        (root / "chat_template.jinja").write_bytes(template_bytes)
        with (
            patch.object(
                codec_module,
                "TOKENIZER_JSON_SHA256",
                hashlib.sha256(tokenizer_bytes).hexdigest(),
            ),
            patch.object(
                codec_module,
                "CHAT_TEMPLATE_SHA256",
                hashlib.sha256(template_bytes).hexdigest(),
            ),
            patch.object(
                codec_module.Tokenizer,
                "from_str",
                return_value=tokenizer,
            ),
        ):
            yield codec_module.GLM53Codec(root)


if __name__ == "__main__":
    unittest.main()

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path

from tokenizers import Tokenizer

from infer.codec import IncrementalDecoder
from infer.models.deepseek_v4_flash import model as deepseek_v4_flash

TOKENIZER_JSON_SHA256 = (
    "8f9f37ca37fdc4f5fd36d5cf4d3b0e8392edb4e894fd10cc0d70b4957c8633cf"
)
TOKENIZER_CONFIG_SHA256 = (
    "6ac8c8dc065ed118161d02dd532749ae3f52c243deac27872134fae2f50d8547"
)
GENERATION_CONFIG_SHA256 = (
    "5fccff80f55a4d455bbe516bdd552edf3e9623df95e99fbf2a3c3389fdf91af0"
)
ENCODING_REFERENCE_SHA256 = (
    "abc0d26120250dda0ae077dc64aa28836026e61e970854aaeb792445e6a0dde6"
)
EOS_TOKEN_IDS = frozenset((1,))

_BOS = "<｜begin▁of▁sentence｜>"
_EOS = "<｜end▁of▁sentence｜>"
_USER = "<｜User｜>"
_ASSISTANT = "<｜Assistant｜>"
_THINK_END = "</think>"
_SPECIAL_TOKEN_IDS = (
    (_BOS, 0),
    (_EOS, 1),
    (_USER, 128_803),
    (_ASSISTANT, 128_804),
    (_THINK_END, 128_822),
)


class DeepSeekV4Codec:
    def __init__(self, checkpoint_root: str | Path) -> None:
        root = Path(checkpoint_root)
        tokenizer_bytes = _read_pinned(root / "tokenizer.json", TOKENIZER_JSON_SHA256)
        _read_pinned(root / "tokenizer_config.json", TOKENIZER_CONFIG_SHA256)
        _read_pinned(root / "generation_config.json", GENERATION_CONFIG_SHA256)
        _read_pinned(root / "encoding" / "encoding_dsv4.py", ENCODING_REFERENCE_SHA256)

        self._tokenizer = Tokenizer.from_str(tokenizer_bytes.decode("utf-8"))
        if (
            self._tokenizer.get_vocab_size(with_added_tokens=True)
            != deepseek_v4_flash.VOCAB_SIZE
        ):
            raise RuntimeError("DeepSeek V4 tokenizer has the wrong vocabulary size")
        if any(
            self._tokenizer.token_to_id(token) != token_id
            for token, token_id in _SPECIAL_TOKEN_IDS
        ):
            raise RuntimeError("DeepSeek V4 tokenizer has the wrong special tokens")

    def encode_messages(self, messages: Sequence[Mapping[str, object]]) -> list[int]:
        token_ids = self._tokenizer.encode(
            self.render(messages), add_special_tokens=False
        ).ids
        _validate_token_ids(token_ids, allow_empty=False)
        return token_ids

    def render(self, messages: Sequence[Mapping[str, object]]) -> str:
        rendered = [_BOS]
        for index, message in enumerate(messages):
            role = message.get("role")
            content = message.get("content")
            if type(content) is not str:
                raise ValueError(
                    f"DeepSeek V4 message {index} content must be a string"
                )
            if role == "system":
                rendered.append(content)
            elif role == "user":
                rendered.extend((_USER, content))
            elif role == "assistant":
                rendered.extend((content, _EOS))
            else:
                raise ValueError(f"DeepSeek V4 message {index} has an invalid role")

            next_role = (
                messages[index + 1].get("role") if index + 1 < len(messages) else None
            )
            if role == "user" and (next_role == "assistant" or next_role is None):
                rendered.extend((_ASSISTANT, _THINK_END))
        return "".join(rendered)

    def decode(
        self,
        token_ids: Sequence[int],
        *,
        skip_special_tokens: bool = True,
    ) -> str:
        _validate_token_ids(token_ids, allow_empty=True)
        return self._tokenizer.decode(
            list(token_ids), skip_special_tokens=skip_special_tokens
        )

    def incremental_decoder(self) -> IncrementalDecoder:
        return IncrementalDecoder(self._tokenizer)


def _read_pinned(path: Path, expected_sha256: str) -> bytes:
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise RuntimeError(f"DeepSeek V4 codec artifact SHA-256 mismatch: {path.name}")
    return payload


def _validate_token_ids(token_ids: Sequence[int], *, allow_empty: bool) -> None:
    if not allow_empty and not token_ids:
        raise RuntimeError("DeepSeek V4 tokenizer returned no token IDs")
    if any(
        type(token_id) is not int or not 0 <= token_id < deepseek_v4_flash.VOCAB_SIZE
        for token_id in token_ids
    ):
        raise ValueError("DeepSeek V4 token IDs must be integers within the vocabulary")

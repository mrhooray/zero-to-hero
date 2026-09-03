import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import jinja2
from jinja2.ext import LoopControlExtension
from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer

from infer.codec import IncrementalDecoder
from infer.models.glm53_flash.model import TOKENIZER_VOCAB_SIZE

TOKENIZER_JSON_SHA256 = (
    "19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d"
)
CHAT_TEMPLATE_SHA256 = (
    "34d5ee66b12fa6446cdae131c352b8f68cd85369e0e6fda115583805fada3891"
)
EOS_TOKEN_IDS = frozenset((154_820, 154_827, 154_829))


class GLM53Codec:
    def __init__(self, checkpoint_root: str | Path) -> None:
        root = Path(checkpoint_root)
        tokenizer_bytes = _read_pinned(root / "tokenizer.json", TOKENIZER_JSON_SHA256)
        template_bytes = _read_pinned(
            root / "chat_template.jinja", CHAT_TEMPLATE_SHA256
        )

        self._tokenizer = Tokenizer.from_str(tokenizer_bytes.decode("utf-8"))
        if (
            self._tokenizer.get_vocab_size(with_added_tokens=True)
            != TOKENIZER_VOCAB_SIZE
        ):
            raise RuntimeError("GLM tokenizer has the wrong vocabulary size")
        self._template = _compile_chat_template(template_bytes.decode("utf-8"))

    def encode_messages(
        self,
        messages: Sequence[Mapping[str, object]],
        *,
        tools: Sequence[Mapping[str, object]] | None = None,
        reasoning_effort: str | None = None,
    ) -> list[int]:
        rendered = self.render(
            messages,
            tools=tools,
            reasoning_effort=reasoning_effort,
        )
        token_ids = self._tokenizer.encode(rendered, add_special_tokens=False).ids
        _validate_token_ids(token_ids, allow_empty=False)
        return token_ids

    def render(
        self,
        messages: Sequence[Mapping[str, object]],
        *,
        tools: Sequence[Mapping[str, object]] | None = None,
        reasoning_effort: str | None = None,
    ) -> str:
        rendered = self._template.render(
            messages=messages,
            tools=tools,
            add_generation_prompt=True,
            reasoning_effort=reasoning_effort,
        )
        if not rendered:
            raise RuntimeError("GLM chat template rendered an empty prompt")
        return rendered

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


def _compile_chat_template(source: str) -> jinja2.Template:
    environment = ImmutableSandboxedEnvironment(
        trim_blocks=True,
        lstrip_blocks=True,
        extensions=[LoopControlExtension],
    )
    environment.filters["tojson"] = _tojson
    environment.globals["raise_exception"] = _raise_exception
    return environment.from_string(source)


def _tojson(
    value: Any,
    ensure_ascii: bool = False,
    indent: int | None = None,
    separators: tuple[str, str] | None = None,
    sort_keys: bool = False,
) -> str:
    return json.dumps(
        value,
        ensure_ascii=ensure_ascii,
        indent=indent,
        separators=separators,
        sort_keys=sort_keys,
    )


def _raise_exception(message: str) -> None:
    raise jinja2.TemplateError(message)


def _read_pinned(path: Path, expected_sha256: str) -> bytes:
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise RuntimeError(f"GLM codec artifact SHA-256 mismatch: {path.name}")
    return payload


def _validate_token_ids(token_ids: Sequence[int], *, allow_empty: bool) -> None:
    if not allow_empty and not token_ids:
        raise RuntimeError("GLM tokenizer returned no token IDs")
    if any(
        type(token_id) is not int or not 0 <= token_id < TOKENIZER_VOCAB_SIZE
        for token_id in token_ids
    ):
        raise ValueError("GLM token IDs must be integers within the vocabulary")

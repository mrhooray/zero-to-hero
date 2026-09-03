from collections.abc import Sequence
from hashlib import sha256
from threading import Lock
from typing import Protocol

import orjson
from tokenizers import Tokenizer
from tokenizers.decoders import DecodeStream

_TOKEN_CACHE_ENTRIES, _TOKEN_CACHE_ENTRY_TOKENS = 64, 1 << 14


class IncrementalDecoder:
    def __init__(self, tokenizer: Tokenizer) -> None:
        self._tokenizer = tokenizer
        self._stream = DecodeStream(skip_special_tokens=True)

    def decode(self, token_ids: Sequence[int]) -> str:
        return "".join(
            self._stream.step(self._tokenizer, token_id) or "" for token_id in token_ids
        )


class Codec(Protocol):
    def encode_messages(self, messages: Sequence[dict[str, str]]) -> Sequence[int]: ...

    def decode(self, token_ids: Sequence[int]) -> str: ...

    def incremental_decoder(self) -> IncrementalDecoder: ...


class CachedCodec:
    def __init__(self, codec: Codec) -> None:
        self._codec = codec
        self._entries: dict[bytes, tuple[int, ...]] = {}
        self._lock = Lock()

    def encode_messages(self, messages: Sequence[dict[str, str]]) -> tuple[int, ...]:
        key = sha256(orjson.dumps(messages)).digest()
        with self._lock:
            cached = self._entries.pop(key, None)
            if cached is not None:
                self._entries[key] = cached
        if cached is not None:
            return cached
        token_ids = tuple(self._codec.encode_messages(messages))
        if not token_ids or len(token_ids) > _TOKEN_CACHE_ENTRY_TOKENS:
            return token_ids
        with self._lock:
            token_ids = self._entries.pop(key, token_ids)
            if len(self._entries) == _TOKEN_CACHE_ENTRIES:
                self._entries.pop(next(iter(self._entries)))
            self._entries[key] = token_ids
        return token_ids

    def __getattr__(self, name: str):
        return getattr(self._codec, name)

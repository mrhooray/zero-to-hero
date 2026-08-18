from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from huggingface_hub import snapshot_download

MANIFEST_NAME = "infer-checkpoint-manifest.json"
RESULT_PREFIX = "INFER_CHECKPOINT_RESULT="
HASH_CHUNK_BYTES = 8 * 1024 * 1024
DOWNLOAD_PATTERNS = (
    "config.json",
    "model.safetensors.index.json",
    "*.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "generation_config.json",
    "chat_template.jinja",
    "encoding/*",
)


@dataclass(frozen=True, slots=True)
class ModelSpec:
    model_id: str
    revision: str
    required_files: tuple[str, ...]


MODEL_SPECS = {
    spec.model_id: spec
    for spec in (
        ModelSpec(
            "deepseek-ai/DeepSeek-V4-Flash-0731",
            "7872f01b1d1fe23eabc4c98b48bffcef5a386062",
            (
                "config.json",
                "model.safetensors.index.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "generation_config.json",
                "encoding/encoding_dsv4.py",
            ),
        ),
        ModelSpec(
            "zai-org/GLM-5.3-Flash",
            "04c4e9e95c5da8862dced7e5056455116f83a7e0",
            (
                "config.json",
                "model.safetensors.index.json",
                "tokenizer.json",
                "chat_template.jinja",
            ),
        ),
    )
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Stage one pinned model checkpoint into a local directory."
    )
    parser.add_argument("--model", choices=tuple(MODEL_SPECS), required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        result = stage_checkpoint(args.model, args.root)
    except Exception as error:  # noqa: BLE001
        print(
            RESULT_PREFIX
            + json.dumps(
                {
                    "error": str(error),
                    "model_id": args.model,
                    "revision": MODEL_SPECS[args.model].revision,
                    "status": "failed",
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 1

    print(RESULT_PREFIX + json.dumps(result, sort_keys=True), flush=True)
    return 0


def stage_checkpoint(
    model_id: str,
    root: Path,
    downloader: Callable[..., str] = snapshot_download,
) -> dict[str, object]:
    spec = _model_spec(model_id)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / MANIFEST_NAME
    try:
        existing = _read_json(manifest_path)
    except FileNotFoundError:
        existing = None
    if existing is not None and validate_manifest(root, existing, spec):
        return _receipt(spec, manifest_path, existing, cache_hit=True)

    downloader(
        repo_id=spec.model_id,
        revision=spec.revision,
        local_dir=str(root),
        allow_patterns=DOWNLOAD_PATTERNS,
        token=False,
    )
    names = checkpoint_files(root, spec)
    missing = [name for name in names if not (root / name).is_file()]
    if missing:
        raise ValueError("checkpoint download omitted: " + ", ".join(missing))

    manifest = make_manifest(root, spec, names)
    _write_json(manifest_path, manifest)
    if not validate_manifest(root, manifest, spec):
        raise RuntimeError("published checkpoint manifest failed validation")
    return _receipt(spec, manifest_path, manifest, cache_hit=False)


def checkpoint_files(root: Path, spec: ModelSpec) -> tuple[str, ...]:
    index = _read_json(root / "model.safetensors.index.json")
    weight_map = index.get("weight_map") if isinstance(index, Mapping) else None
    if not isinstance(weight_map, Mapping) or any(
        not isinstance(name, str) or not isinstance(shard, str)
        for name, shard in weight_map.items()
    ):
        raise ValueError("checkpoint index does not contain a string weight_map")
    shards = set(weight_map.values())
    if any(not _safe_relative_name(shard) for shard in shards):
        raise ValueError("checkpoint index contains an unsafe shard path")
    return tuple(sorted((*spec.required_files, *shards)))


def make_manifest(
    root: Path,
    spec: ModelSpec,
    names: tuple[str, ...] | None = None,
) -> dict[str, object]:
    names = checkpoint_files(root, spec) if names is None else names
    return {
        "files": [file_manifest(root / name, name) for name in names],
        "model_id": spec.model_id,
        "revision": spec.revision,
        "schema": 1,
    }


def validate_manifest(root: Path, manifest: object, spec: ModelSpec) -> bool:
    if not isinstance(manifest, Mapping):
        return False
    if (
        manifest.get("schema") != 1
        or manifest.get("model_id") != spec.model_id
        or manifest.get("revision") != spec.revision
    ):
        return False
    entries = manifest.get("files")
    if not isinstance(entries, list) or any(
        not isinstance(entry, Mapping) for entry in entries
    ):
        return False
    try:
        expected = checkpoint_files(root, spec)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False
    if tuple(entry.get("name") for entry in entries) != expected:
        return False
    for entry, name in zip(entries, expected, strict=True):
        if (
            entry.get("name") != name
            or type(entry.get("size")) is not int
            or type(entry.get("sha256")) is not str
            or not validate_file(root / name, entry["size"], entry["sha256"])
        ):
            return False
    return True


def validate_file(path: Path, expected_size: int, expected_sha256: str) -> bool:
    try:
        if path.stat().st_size != expected_size:
            return False
        digest = _sha256(path)
    except OSError:
        return False
    return digest == expected_sha256


def file_manifest(path: Path, name: str) -> dict[str, object]:
    return {
        "name": name,
        "sha256": _sha256(path),
        "size": path.stat().st_size,
    }


def _receipt(
    spec: ModelSpec,
    manifest_path: Path,
    manifest: Mapping[str, object],
    *,
    cache_hit: bool,
) -> dict[str, object]:
    entries = manifest["files"]
    assert isinstance(entries, list)
    return {
        "cache_hit": cache_hit,
        "files": len(entries),
        "manifest": str(manifest_path),
        "model_id": spec.model_id,
        "revision": spec.revision,
        "status": "passed",
        "total_bytes": sum(entry["size"] for entry in entries),
    }


def _model_spec(model_id: str) -> ModelSpec:
    try:
        return MODEL_SPECS[model_id]
    except KeyError as error:
        raise ValueError(f"unsupported model: {model_id}") from error


def _read_json(path: Path) -> object:
    return json.loads(path.read_bytes())


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _safe_relative_name(name: str) -> bool:
    path = PurePosixPath(name)
    return bool(name) and not path.is_absolute() and ".." not in path.parts


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())

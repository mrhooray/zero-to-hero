"""Shared Modal and checkpoint mechanics for benchmark profiles.

Profiles own model pins, the checkpoint volume, the server command, and the
workload codec. Everything here is model-agnostic plumbing used verbatim by
every profile.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import modal

REMOTE_ROOT = Path("/opt/infer")
CHECKPOINT_MOUNT = Path("/checkpoint")
GPU = "B200:4"
CPU = 16
MEMORY_MIB = 32_768
FUNCTION_TIMEOUT_SECONDS = 6 * 60 * 60
ENVIRONMENT = {
    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
    "NCCL_SOCKET_IFNAME": "=lo",
    "PYTHONPATH": f"{REMOTE_ROOT}/src:{REMOTE_ROOT}",
    "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
}

BASE_IMAGE = (
    "nvidia/cuda:12.9.1-devel-ubuntu22.04@"
    "sha256:38804006c937a83f28f63a959abcee688042072319c8614ad57b350958a30bd3"
)
TORCH_INDEX = "https://download.pytorch.org/whl/cu129"
TORCH_WHEEL = (
    "https://download-r2.pytorch.org/whl/cu129/"
    "torch-2.13.0%2Bcu129-cp312-cp312-manylinux_2_28_x86_64.whl#"
    "sha256=df28741fcd89e3da7cce2d48cbe5299d6732d510ac20f5d422d0b85edf18c327"
)
UV_VERSION = "0.12.5"

FLASHINFER_VERSION = "0.6.18.dev20260819"
FLASHINFER_COMMIT = "61a6c651872a7d3f2f6dcc1ced61633d8f8ba3dd"
FLASHINFER_PYTHON_WHEEL = (
    "https://github.com/flashinfer-ai/whl/releases/download/"
    "nightly-v0.6.18-20260819/flashinfer_python-0.6.18.dev20260819-py3-none-any.whl"
)
FLASHINFER_CUBIN_WHEEL = (
    "https://github.com/flashinfer-ai/whl/releases/download/"
    "nightly-v0.6.18-20260819/flashinfer_cubin-0.6.18.dev20260819-py3-none-any.whl"
)
FLASHINFER_JIT_WHEEL = (
    "https://github.com/flashinfer-ai/whl/releases/download/"
    "nightly-v0.6.18-20260819/flashinfer_jit_cache-0.6.18.dev20260819%2Bcu129-"
    "cp39-abi3-manylinux_2_28_x86_64.whl"
)


def torch_base_image():
    return (
        modal.Image.from_registry(BASE_IMAGE, add_python="3.12")
        .entrypoint([])
        .apt_install("git")
        .uv_pip_install(
            TORCH_WHEEL,
            "triton==3.7.1",
            "ninja==1.13.0",
            "numpy==2.5.2",
            "nvidia-ml-py==13.580.82",
            "apache-tvm-ffi==0.1.11",
            "build==1.3.0",
            "cuda-bindings==12.9.7",
            "cuda-pathfinder==1.8.1",
            "packaging==26.3",
            "safetensors==0.8.0",
            "tokenizers==0.23.1",
            "huggingface-hub==1.29.0",
            "httptools==0.8.0",
            "jinja2==3.1.6",
            "orjson==3.12.0",
            "requests==2.32.5",
            "starlette==1.6.0",
            "uvicorn==0.41.0",
            "uvloop==0.22.1",
            "setuptools==84.0.0",
            "wheel==0.48.0",
            index_url="https://pypi.org/simple",
            extra_index_url=TORCH_INDEX,
            extra_options="--index-strategy unsafe-best-match",
            uv_version=UV_VERSION,
        )
    )


def with_flashinfer(image):
    return image.uv_pip_install(
        FLASHINFER_PYTHON_WHEEL,
        FLASHINFER_CUBIN_WHEEL,
        FLASHINFER_JIT_WHEEL,
        extra_options="--no-deps",
        uv_version=UV_VERSION,
    )


def with_pinned_wheels(image, *wheels: str):
    return image.uv_pip_install(
        *wheels,
        extra_options="--no-deps",
        uv_version=UV_VERSION,
    )


def with_source_tree(image, root: Path):
    return image.add_local_dir(
        root / "src", str(REMOTE_ROOT / "src"), copy=True
    ).add_local_dir(root / "tools", str(REMOTE_ROOT / "tools"), copy=True)


def launcher_metadata(
    root: Path,
    *,
    volume: str,
    concurrencies: list[int],
    job_name: str,
    model_id: str,
    model_revision: str,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "base_image": BASE_IMAGE,
        "checkpoint_volume": volume,
        "concurrencies": concurrencies,
        "job_name": job_name,
        "model_id": model_id,
        "model_revision": model_revision,
        "modal_version": modal.__version__,
        "source_sha256": source_hashes(root, (root / "src", root / "tools")),
        "torch_wheel": TORCH_WHEEL,
        "transport": "modal_function",
        **(extra or {}),
    }


def checkpoint_manifest(
    mount: Path, model_id: str, model_revision: str
) -> dict[str, object]:
    path = mount / "infer-checkpoint-manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema") != 1
        or manifest.get("model_id") != model_id
        or manifest.get("revision") != model_revision
        or not isinstance(manifest.get("files"), list)
    ):
        raise RuntimeError(f"invalid checkpoint manifest: {path}")
    return {
        "files": manifest["files"],
        "model_id": manifest["model_id"],
        "revision": manifest["revision"],
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def gpu_topology() -> list[dict[str, object]]:
    import torch

    topology = [
        {
            "capability": list(torch.cuda.get_device_capability(index)),
            "index": index,
            "name": torch.cuda.get_device_name(index),
        }
        for index in range(torch.cuda.device_count())
    ]
    if len(topology) != 4 or any(
        item["name"] != "NVIDIA B200" or item["capability"] != [10, 0]
        for item in topology
    ):
        raise RuntimeError(f"expected four B200 GPUs, found {topology}")
    return topology


def source_hashes(root: Path, paths: tuple[Path, ...]) -> dict[str, str]:
    files = []
    for source in paths:
        if source.is_dir():
            files.extend(
                candidate
                for candidate in source.rglob("*")
                if candidate.is_file() and "__pycache__" not in candidate.parts
            )
        else:
            files.append(source)
    return {
        str(source.relative_to(root)): hashlib.sha256(source.read_bytes()).hexdigest()
        for source in sorted(files)
    }

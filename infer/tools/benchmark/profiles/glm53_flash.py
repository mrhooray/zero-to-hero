from __future__ import annotations

import sys
from pathlib import Path

from tools.benchmark import decode
from tools.benchmark.profiles import common
from tools.kernels.install_flash_kda import (
    CUTLASS_COMMIT,
    FLASH_KDA_COMMIT,
    FLASH_KDA_VERSION,
)

NAME = "glm-5-3-flash"
MODEL_KEY = "glm53_flash"
MODEL_ID = "zai-org/GLM-5.3-Flash"
MODEL_REVISION = "04c4e9e95c5da8862dced7e5056455116f83a7e0"

CHECKPOINT_VOLUME = "infer-glm53-04c4e9e-20260903"
CHECKPOINT_MOUNT = common.CHECKPOINT_MOUNT
REMOTE_ROOT = common.REMOTE_ROOT
GPU = common.GPU
CPU = common.CPU
MEMORY_MIB = common.MEMORY_MIB
FUNCTION_TIMEOUT_SECONDS = common.FUNCTION_TIMEOUT_SECONDS
ENVIRONMENT = common.ENVIRONMENT


def build_image(root: Path):
    image = common.with_flashinfer(common.torch_base_image())
    image = common.with_pinned_wheels(
        image,
        "sgl-deep-gemm==0.1.5.post3",
        "nvidia-cutlass-dsl==4.5.0",
        "nvidia-cutlass-dsl-libs-base==4.5.0",
    )
    image = image.add_local_file(
        root / "tools/kernels/install_flash_kda.py",
        f"{REMOTE_ROOT}/tools/install_flash_kda.py",
        copy=True,
    )
    image = image.run_commands(f"python {REMOTE_ROOT}/tools/install_flash_kda.py")
    image = common.with_source_tree(image, root)
    return image


def launcher_metadata(
    root: Path,
    *,
    volume: str,
    concurrencies: list[int],
    job_name: str,
) -> dict[str, object]:
    return common.launcher_metadata(
        root,
        volume=volume,
        concurrencies=concurrencies,
        job_name=job_name,
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        extra={
            "cutlass_commit": CUTLASS_COMMIT,
            "flash_kda_commit": FLASH_KDA_COMMIT,
            "flash_kda_version": FLASH_KDA_VERSION,
            "flashinfer_commit": common.FLASHINFER_COMMIT,
            "flashinfer_version": common.FLASHINFER_VERSION,
        },
    )


def benchmark_inputs() -> tuple[
    dict[str, object], dict[str, object], dict[str, object]
]:
    from infer.models.glm53_flash.codec import GLM53Codec

    checkpoint = checkpoint_manifest()
    workload = decode.build_workload(
        GLM53Codec(CHECKPOINT_MOUNT),
        model_key=MODEL_KEY,
        checkpoint_revision=MODEL_REVISION,
    )
    return workload, server_config(checkpoint), {
        "checkpoint_manifest": checkpoint,
        "gpu_topology": gpu_topology(),
    }


def checkpoint_manifest() -> dict[str, object]:
    return common.checkpoint_manifest(CHECKPOINT_MOUNT, MODEL_ID, MODEL_REVISION)


def server_config(checkpoint: dict[str, object]) -> dict[str, object]:
    host = "127.0.0.1"
    port = 18181
    config = {
        "schema": decode.SERVER_CONFIG_SCHEMA,
        "engine": "infer",
        "model_key": MODEL_KEY,
        "model_id": MODEL_ID,
        "endpoint": f"http://{host}:{port}",
        "launch_command": [
            sys.executable,
            "-m",
            "infer",
            "serve",
            MODEL_ID,
            "--checkpoint-dir",
            str(CHECKPOINT_MOUNT),
            "--parallelism",
            "dep4",
            "--speculation",
            "none",
            "--host",
            host,
            "--port",
            str(port),
        ],
        "resolved_server_config": {
            "speculation": "none",
            "prefix_cache": "enabled",
            "prefix_routing": "engine_global_cache",
            "gpu_count": 4,
            "data_parallel": 4,
            "expert_parallel": 4,
        },
        "checkpoint": {
            "revision": checkpoint["revision"],
            "sha256": checkpoint["sha256"],
        },
    }
    return config


def gpu_topology() -> list[dict[str, object]]:
    return common.gpu_topology()

from __future__ import annotations

import sys
from pathlib import Path

from tools.benchmark import decode
from tools.benchmark.profiles import common, external
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


def build_image(root: Path, engine: str = "infer"):
    if engine != "infer":
        return external.build_image(root, MODEL_KEY, engine)
    image = common.with_flashinfer(common.torch_base_image())
    image = common.with_pinned_wheels(
        image,
        "nvidia-cutlass-dsl==4.5.0",
        "nvidia-cutlass-dsl-libs-base==4.5.0",
    )
    image = image.add_local_dir(
        root / "tools/kernels/deepgemm_paged_mqa_out",
        "/opt/deepgemm-paged-mqa-out",
        copy=True,
    )
    image = image.run_commands(
        f"INFER_BASE_IMAGE='{common.BASE_IMAGE}' "
        "python /opt/deepgemm-paged-mqa-out/build_patched_wheel.py "
        "--work-dir /tmp/deepgemm-build --install"
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
    engine: str = "infer",
) -> dict[str, object]:
    return common.launcher_metadata(
        root,
        volume=volume,
        concurrencies=concurrencies,
        job_name=job_name,
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        extra={
            "engine": engine,
            "cutlass_commit": CUTLASS_COMMIT,
            "flash_kda_commit": FLASH_KDA_COMMIT,
            "flash_kda_version": FLASH_KDA_VERSION,
            "flashinfer_commit": common.FLASHINFER_COMMIT,
            "flashinfer_version": common.FLASHINFER_VERSION,
        }
        if engine == "infer"
        else None,
        runtime=external.metadata(MODEL_KEY, engine) if engine != "infer" else None,
    )


def benchmark_inputs(
    speculation: str = "none", engine: str = "infer"
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    from infer.models.glm53_flash.codec import GLM53Codec

    checkpoint = checkpoint_manifest()
    workload = decode.build_workload(
        GLM53Codec(CHECKPOINT_MOUNT),
        model_key=MODEL_KEY,
        checkpoint_revision=MODEL_REVISION,
    )
    config = server_config(checkpoint, speculation, engine)
    return (
        workload,
        config,
        {
            "checkpoint_manifest": checkpoint,
            "gpu_topology": gpu_topology(),
        },
    )


def checkpoint_manifest() -> dict[str, object]:
    return common.checkpoint_manifest(CHECKPOINT_MOUNT, MODEL_ID, MODEL_REVISION)


def server_config(
    checkpoint: dict[str, object], speculation: str = "none", engine: str = "infer"
) -> dict[str, object]:
    if engine != "infer":
        if speculation != "none":
            raise ValueError(f"{engine} benchmark profile is target-only")
        return external.server_config(MODEL_KEY, engine, checkpoint)
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
            speculation,
            "--host",
            host,
            "--port",
            str(port),
        ],
        "resolved_server_config": {
            "speculation": speculation,
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

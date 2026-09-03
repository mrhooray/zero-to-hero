from __future__ import annotations

import sys
from pathlib import Path

from tools.benchmark import decode
from tools.benchmark.profiles import common

NAME = "deepseek-v4-flash"
MODEL_KEY = "deepseek_v4_flash"
MODEL_ID = "deepseek-ai/DeepSeek-V4-Flash-0731"
MODEL_REVISION = "7872f01b1d1fe23eabc4c98b48bffcef5a386062"

CHECKPOINT_VOLUME = "infer-dsv4-0731-20260902"
CHECKPOINT_MOUNT = common.CHECKPOINT_MOUNT
REMOTE_ROOT = common.REMOTE_ROOT
GPU = common.GPU
CPU = common.CPU
MEMORY_MIB = common.MEMORY_MIB
FUNCTION_TIMEOUT_SECONDS = common.FUNCTION_TIMEOUT_SECONDS
ENVIRONMENT = common.ENVIRONMENT

FLASH_MLA_COMMIT = "15f13e5030374295491c5ce31b02d7e63a7772c6"


def build_image(root: Path):
    image = common.with_flashinfer(common.torch_base_image())
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
    image = image.run_commands(
        "git clone --filter=blob:none https://github.com/deepseek-ai/FlashMLA.git "
        "/opt/flashmla && "
        f"git -C /opt/flashmla checkout {FLASH_MLA_COMMIT} && "
        "CC=/usr/bin/gcc CXX=/usr/bin/g++ CUDAHOSTCXX=/usr/bin/g++ "
        "python -m pip install --no-build-isolation /opt/flashmla && "
        "rm -rf /opt/flashmla"
    )
    image = common.with_source_tree(image, root)
    return image.run_commands(
        f"python {REMOTE_ROOT}/tools/kernels/build_deepseek_v4_flash_writer.py"
    )


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
            "flashinfer_commit": common.FLASHINFER_COMMIT,
            "flashinfer_version": common.FLASHINFER_VERSION,
            "flash_mla_commit": FLASH_MLA_COMMIT,
        },
    )


def benchmark_inputs() -> tuple[
    dict[str, object], dict[str, object], dict[str, object]
]:
    from infer.models.deepseek_v4_flash.codec import DeepSeekV4Codec

    checkpoint = checkpoint_manifest()
    workload = decode.build_workload(
        DeepSeekV4Codec(CHECKPOINT_MOUNT),
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

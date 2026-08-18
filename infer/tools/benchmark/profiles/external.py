from __future__ import annotations

from pathlib import Path

from tools.benchmark import decode
from tools.benchmark.profiles import common

ENGINES = ("infer", "tokenspeed", "sglang")

_PROFILES = {
    ("deepseek_v4_flash", "tokenspeed"): {
        "image": {"kind": "id", "ref": "im-hWqO27iSHUj6J1xVSVgm5v"},
        "source_revision": "c312bcf961a0ce70eb110af2f0d6aee3632d7692",
        "model_id": "/checkpoint",
        "endpoint": "http://127.0.0.1:8000",
        "launch_command": [
            "tokenspeed",
            "serve",
            "/checkpoint",
            "--trust-remote-code",
            "--data-parallel-size",
            "4",
            "--dist-init-addr",
            "127.0.0.1:4000",
            "--enable-expert-parallel",
            "--kv-cache-dtype",
            "fp8_e4m3",
            "--moe-backend",
            "mega_moe",
            "--attention-use-fp4-indexer-cache",
            "--max-model-len",
            "16384",
            "--max-num-seqs",
            "128",
            "--max-total-tokens",
            "327680",
            "--max-prefill-tokens",
            "32768",
            "--chunked-prefill-size",
            "32768",
            "--enable-mixed-batch",
            "--gpu-memory-utilization",
            "0.90",
            "--disable-kvstore",
            "--enable-prefix-caching",
            "--max-cudagraph-capture-size",
            "32",
            "--cudagraph-capture-sizes",
            "1",
            "2",
            "4",
            "8",
            "16",
            "32",
            "--prefill-graph-max-tokens",
            "2048",
            "--enable-metrics",
            "--enable-cache-report",
            "--enable-log-request-stats",
            "--policy",
            "cache_aware",
            "--dp-aware",
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
        ],
        "resolved": {
            "max_num_seqs_global": 128,
            "max_total_tokens": 327680,
            "cuda_graph_local_batch_max": 32,
            "prefix_routing": "gateway_cache_aware_dp",
        },
    },
    ("deepseek_v4_flash", "sglang"): {
        "image": {
            "kind": "registry",
            "ref": "lmsysorg/sglang@sha256:bde16a8447b19e89056b9eea06c72be6c02801dc89d528c9ea90c53368fd74bf",
        },
        "source_revision": "71de97b264b04dcd514cf904003028aefe9775c8",
        "model_id": "/checkpoint",
        "endpoint": "http://127.0.0.1:30000",
        "launch_command": [
            "env",
            "SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=4096",
            "python3",
            "-m",
            "sglang.launch_server",
            "--trust-remote-code",
            "--model-path",
            "/checkpoint",
            "--tp",
            "4",
            "--dp",
            "4",
            "--enable-dp-attention",
            "--enable-dp-lm-head",
            "--enable-dp-attention-local-control-broadcast",
            "--moe-a2a-backend",
            "megamoe",
            "--max-running-requests",
            "128",
            "--chunked-prefill-size",
            "8192",
            "--cuda-graph-max-bs-decode",
            "32",
            "--swa-full-tokens-ratio",
            "0.1",
            "--stream-interval",
            "3",
            "--enable-metrics",
            "--enable-cache-report",
            "--host",
            "127.0.0.1",
            "--port",
            "30000",
        ],
        "resolved": {
            "max_running_requests_global": 128,
            "cuda_graph_local_batch_max": 32,
            "mega_moe_max_tokens_per_rank": 4096,
            "stream_interval_tokens": 3,
            "prefix_routing": "client_group_to_dp_rank",
        },
    },
    ("glm53_flash", "tokenspeed"): {
        "image": {"kind": "id", "ref": "im-Gp4V0GapElHD2FR7B3N9n0"},
        "source_revision": "95dca6caf9a6d821c7d9e056d57480b88edb24dd",
        "model_id": "glm-5.3-flash",
        "endpoint": "http://127.0.0.1:8000",
        "launch_command": [
            "ts",
            "serve",
            "--model",
            "/checkpoint",
            "--served-model-name",
            "glm-5.3-flash",
            "--data-parallel-size",
            "4",
            "--dist-init-addr",
            "127.0.0.1:4000",
            "--enable-expert-parallel",
            "--language-model-only",
            "--kv-cache-dtype",
            "fp8_e4m3",
            "--moe-backend",
            "flashinfer_trtllm",
            "--max-model-len",
            "32768",
            "--max-num-seqs",
            "128",
            "--max-prefill-tokens",
            "8192",
            "--chunked-prefill-size",
            "4096",
            "--max-cudagraph-capture-size",
            "32",
            "--cudagraph-capture-sizes",
            "1",
            "2",
            "4",
            "8",
            "16",
            "32",
            "--gpu-memory-utilization",
            "0.85",
            "--trust-remote-code",
            "--sampling-backend",
            "flashinfer",
            "--disable-kvstore",
            "--kvstore-ratio",
            "0.0",
            "--enable-prefix-caching",
            "--enable-cache-report",
            "--enable-log-request-stats",
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
            "--policy",
            "cache_aware",
            "--dp-aware",
            "--disable-health-check",
            "--engine-startup-timeout",
            "1800",
            "--gateway-startup-timeout",
            "600",
        ],
        "resolved": {
            "max_num_seqs_global": 128,
            "cuda_graph_local_batch_max": 32,
            "prefix_routing": "gateway_cache_aware_dp",
        },
    },
    ("glm53_flash", "sglang"): {
        "image": {
            "kind": "registry-source-overlay",
            "ref": "lmsysorg/sglang@sha256:0836f0160fa785e424e68d13ef88ddd548f87e6e11ad9f0e4de982e4f9188aaf",
            "source_url": "https://codeload.github.com/sgl-project/sglang/tar.gz/7fa1924c04487feffc3f5f111a424ec1f0b22362",
            "source_sha256": "449dd60c59181ffa60e9659557ad0274829265fd57415351ee01ac2141489bca",
        },
        "source_revision": "7fa1924c04487feffc3f5f111a424ec1f0b22362",
        "model_id": "/checkpoint",
        "endpoint": "http://127.0.0.1:8000",
        "launch_command": [
            "/opt/sglang/bin/python",
            "-m",
            "sglang.launch_server",
            "--model-path",
            "/checkpoint",
            "--tp-size",
            "4",
            "--dp-size",
            "4",
            "--enable-dp-attention",
            "--enable-dp-lm-head",
            "--enable-dp-attention-local-control-broadcast",
            "--ep-size",
            "4",
            "--dsa-prefill-backend",
            "trtllm",
            "--dsa-decode-backend",
            "trtllm",
            "--kv-cache-dtype",
            "fp8_e4m3",
            "--moe-runner-backend",
            "deep_gemm",
            "--context-length",
            "16384",
            "--max-running-requests",
            "128",
            "--max-mamba-cache-size",
            "640",
            "--cuda-graph-backend-decode",
            "full",
            "--cuda-graph-max-bs-decode",
            "32",
            "--max-prefill-tokens",
            "8192",
            "--chunked-prefill-size",
            "8192",
            "--mem-fraction-static",
            "0.85",
            "--language-only",
            "--enable-cache-report",
            "--random-seed",
            "0",
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
        ],
        "resolved": {
            "max_running_requests_global": 128,
            "max_mamba_cache_size_global": 640,
            "cuda_graph_local_batch_max": 32,
            "prefix_routing": "client_group_to_dp_rank",
        },
    },
}


def build_image(root: Path, model_key: str, engine: str):
    import modal

    profile = _profile(model_key, engine)
    image_config = profile["image"]
    image = (
        modal.Image.from_id(str(image_config["ref"]))
        if image_config["kind"] == "id"
        else modal.Image.from_registry(str(image_config["ref"]))
    ).entrypoint([])
    if image_config["kind"] == "registry-source-overlay":
        source = "/opt/sglang-prefix-source"
        archive = f"{source}.tar.gz"
        image = image.run_commands(
            f"curl --proto '=https' --tlsv1.2 --retry 3 --retry-delay 2 -fsSL -o {archive} {image_config['source_url']}",
            f"echo '{image_config['source_sha256']}  {archive}' | sha256sum -c -",
            f"install -d {source} && tar -xzf {archive} --strip-components=1 --no-same-owner -C {source}",
            f"SGLANG_BUILD_RUST_EXTS=none /opt/sglang/bin/python -m pip install --disable-pip-version-check --no-build-isolation --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu130 -e {source}/python",
        )
    return common.with_source_tree(image, root)


def metadata(model_key: str, engine: str) -> dict[str, object]:
    profile = _profile(model_key, engine)
    return {
        "engine": engine,
        "engine_image": profile["image"],
        "engine_source_revision": profile["source_revision"],
    }


def server_config(
    model_key: str,
    engine: str,
    checkpoint: dict[str, object],
) -> dict[str, object]:
    profile = _profile(model_key, engine)
    resolved = {
        "speculation": "none",
        "prefix_cache": "enabled",
        "gpu_count": 4,
        "data_parallel": 4,
        "expert_parallel": 4,
        **profile["resolved"],
    }
    return {
        "schema": decode.SERVER_CONFIG_SCHEMA,
        "engine": engine,
        "model_key": model_key,
        "model_id": profile["model_id"],
        "endpoint": profile["endpoint"],
        "launch_command": profile["launch_command"],
        "resolved_server_config": resolved,
        "checkpoint": {
            "revision": checkpoint["revision"],
            "sha256": checkpoint["sha256"],
        },
    }


def _profile(model_key: str, engine: str) -> dict[str, object]:
    if engine == "infer":
        raise ValueError("infer is owned by the model profile")
    try:
        return _PROFILES[(model_key, engine)]
    except KeyError as error:
        raise ValueError(
            f"unsupported benchmark profile: {model_key}/{engine}"
        ) from error

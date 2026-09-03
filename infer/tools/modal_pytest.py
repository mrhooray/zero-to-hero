"""Run tests/gpu on a Modal GPU as plain pytest.

The tests themselves never import modal: this harness mounts the repo onto
a CUDA image and executes `pytest` there, exactly as a local CUDA machine
would. Use the cheapest GPU that satisfies the selection (pure-torch tests
run anywhere; Triton/SM100 kernels need B200):

    modal run tools/modal_pytest.py --selector tests/gpu/test_glm53_distributed_argmax.py
    INFER_MODAL_GPU=B200 modal run tools/modal_pytest.py --selector tests/gpu
"""

import os
import subprocess
import sys
from pathlib import Path

import modal

app = modal.App("infer-gpu-tests")

# INFER_MODAL_GPU=T4 (default, pure-torch tests) or B200 (Triton/SM100 kernels).

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.13.0",
        index_url="https://download.pytorch.org/whl/cu129",
    )
    .pip_install(
        "huggingface-hub==1.29.0",
        "httptools==0.8.0",
        "jinja2==3.1.6",
        "orjson==3.12.0",
        "starlette==1.6.0",
        "tokenizers==0.23.1",
        "uvicorn==0.41.0",
        "uvloop==0.22.1",
        "pytest>=9.1.1",
    )
    # NOTE: no standalone triton pin — it collides with torch's bundled
    # pytorch-triton on the `triton` package dir. torch alone provides it.
    .add_local_dir(
        "/Users/rh/Projects/zero-to-hero/infer",
        remote_path="/repo",
        # Volatile/local-only dirs must not enter the upload: .pytest_cache
        # mutating mid-upload aborts the build (and races local runs).
        ignore=[".pytest_cache", ".venv", "__pycache__", ".ruff_cache", ".git"],
    )
)


@app.function(
    image=image,
    gpu=os.environ.get("INFER_MODAL_GPU", "T4"),
    timeout=1800,
)
def _run(selector: str) -> int:
    import hashlib

    root = Path("/repo/tests/gpu")
    digest = hashlib.sha256(
        "".join(
            sorted(
                f"{path.name}:{path.stat().st_size}"
                for path in root.glob("test_*.py")
            )
        ).encode()
    ).hexdigest()[:12]
    print(f"gpu-suite: {digest}", flush=True)
    subprocess.run(["nvidia-smi", "-L"], cwd="/repo")
    import importlib.metadata

    for dist in ("torch", "triton", "pytorch-triton"):
        try:
            print(f"{dist}=={importlib.metadata.version(dist)}", flush=True)
        except importlib.metadata.PackageNotFoundError:
            print(f"{dist} MISSING", flush=True)
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            *selector.split(),
            "-q",
            "-p",
            "no:cacheprovider",
        ],
        cwd="/repo",
        env={
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "PYTHONPATH": "/repo/src:/repo",
            "INFER_RUN_GPU_TESTS": "1",
            "HF_HUB_OFFLINE": "1",
        },
    )
    return completed.returncode


@app.local_entrypoint()
def main(selector: str = "tests/gpu") -> None:
    code = _run.remote(selector)
    raise SystemExit(code)


if __name__ == "__main__":
    # Allows `python tools/modal_pytest.py` to show modal's own help.
    print(__doc__)

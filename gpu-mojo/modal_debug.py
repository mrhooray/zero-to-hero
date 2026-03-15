"""GPU debug shell for mojo-gpu-puzzles on Modal.

Usage:
    modal shell modal_debug.py::debug
"""

import os

import modal

app = modal.App("mojo-gpu-puzzles-debug")

_gpu = os.environ.get("GPU", "A10")

# nvidia/cuda:devel provides system cuda-gdb/sanitizer that setup-cuda-gdb links into the pixi env.
image = (
    modal.Image.from_registry("nvidia/cuda:12.6.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install(
        "curl",
        "bash",
        "ca-certificates",
        "cuda-gdb-12-6",
        "cuda-sanitizer-12-6",
        "libbsd0",
    )
    .run_commands("curl -fsSL https://pixi.sh/install.sh | PIXI_HOME=/usr/local bash")
    .workdir("/app")
    .add_local_file("pixi.toml", "/app/pixi.toml", copy=True)
    .add_local_file("pixi.lock", "/app/pixi.lock", copy=True)
    .add_local_dir("scripts", remote_path="/app/scripts", copy=True)
    .run_commands("pixi install -e nvidia")
    .run_commands("pixi run setup-cuda-gdb")
    .add_local_dir("problems", remote_path="/app/problems", copy=False)
)


@app.function(image=image, cpu=1, memory=1024, gpu=_gpu, region="us-west")
def debug():
    pass

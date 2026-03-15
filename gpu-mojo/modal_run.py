"""Run mojo-gpu-puzzles on Modal with NVIDIA GPU.

Usage:
    PUZZLE=p01 modal run modal_run.py
    PUZZLE=p15 GPU=A10G modal run modal_run.py
"""

import os

import modal

app = modal.App("mojo-gpu-puzzles")

# GPU must be set at module load time — Modal reads the decorator on import.
_gpu = os.environ.get("GPU", "A10")
_puzzle = os.environ.get("PUZZLE", "p01")

image = (
    modal.Image.debian_slim()
    .apt_install("curl", "bash", "ca-certificates")
    .run_commands("curl -fsSL https://pixi.sh/install.sh | PIXI_HOME=/usr/local bash")
    .workdir("/app")
    .add_local_file("pixi.toml", "/app/pixi.toml", copy=True)
    .add_local_file("pixi.lock", "/app/pixi.lock", copy=True)
    .add_local_dir("scripts", remote_path="/app/scripts", copy=True)
    .run_commands("pixi install -e nvidia")
    .add_local_dir("problems", remote_path="/app/problems", copy=False)
)


@app.function(image=image, cpu=1, memory=1024, gpu=_gpu, timeout=600)
def run(puzzle: str):
    import subprocess

    subprocess.run(["pixi", "run", puzzle], cwd="/app", check=True)


@app.local_entrypoint()
def main():
    run.remote(_puzzle)

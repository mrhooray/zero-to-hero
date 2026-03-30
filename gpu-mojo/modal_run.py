"""Run mojo-gpu-puzzles on Modal with NVIDIA GPU.

Usage:
    modal run modal_run.py --puzzle p01
    modal run modal_run.py --puzzle p01 --gpu A100-80GB
    modal run modal_run.py --puzzle p09 --args "--first-case"
"""

import modal

app = modal.App("mojo-gpu-puzzles")

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


@app.cls(image=image, cpu=1, memory=1024, gpu="A10", timeout=600)
class Runner:
    @modal.method()
    def run(self, puzzle: str, extra_args: list[str] = []):
        import subprocess

        subprocess.run(["pixi", "run", puzzle] + extra_args, cwd="/app", check=True)


@app.local_entrypoint()
def main(puzzle: str = "p01", gpu: str = "A10", args: str = ""):
    Runner.with_options(gpu=gpu)().run.remote(puzzle, args.split() if args else [])

"""Run mojo-gpu-puzzles on Modal with NVIDIA GPU.

Usage:
    modal run modal_run.py --puzzle p01
    modal run modal_run.py --puzzle p01 --gpu A100-80GB
    modal run modal_run.py --puzzle p09 --args "--first-case"
    modal run modal_run.py --puzzle p10 --sanitizer --tool memcheck --args "--memory-bug"
    modal run modal_run.py --puzzle p10 --sanitizer --tool racecheck --args "--race-condition"
"""

import modal

app = modal.App("mojo-gpu-puzzles")

image = (
    modal.Image.debian_slim()
    .apt_install("curl", "bash", "ca-certificates")
    .run_commands("curl -fsSL https://pixi.sh/install.sh | PIXI_HOME=/usr/local bash")
    .run_commands(
        "pixi self-update --version 0.69.0 --no-release-note",
    )
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

    @modal.method()
    def run_sanitizer(self, puzzle: str, tool: str, extra_args: list[str] = []):
        import os
        import subprocess

        env = {**os.environ, "MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT": "0"}
        cmd = [
            "pixi", "run", "compute-sanitizer",
            "--tool", tool,
            "mojo", f"problems/{puzzle}/{puzzle}.mojo",
        ] + extra_args
        subprocess.run(cmd, cwd="/app", check=True, env=env)


@app.local_entrypoint()
def main(puzzle: str = "p01", gpu: str = "A10", args: str = "", sanitizer: bool = False, tool: str = ""):
    extra_args = args.split() if args else []
    runner = Runner.with_options(gpu=gpu)()
    if sanitizer:
        runner.run_sanitizer.remote(puzzle, tool, extra_args)
    else:
        runner.run.remote(puzzle, extra_args)

from pathlib import Path

import modal

app = modal.App("gpt2-train")

image = (
    modal.Image.from_registry(
        "pytorch/pytorch:2.10.0-cuda12.6-cudnn9-runtime",
    )
    .uv_pip_install("tiktoken", "datasets")
    .add_local_dir(
        local_path=Path(__file__).parent,
        remote_path="/root/gpt2",
    )
)


@app.function(
    image=image,
    gpu="a100-80gb",
    timeout=60 * 60 * 24,  # modal max value
)
def train():
    import runpy
    import sys

    sys.path.insert(0, "/root/gpt2")
    runpy.run_path("/root/gpt2/train.py", run_name="__main__")


@app.function(
    image=image,
    gpu="a100-80gb:8",
    timeout=60 * 60 * 12,
)
def train_ddp():
    import subprocess
    import sys

    sys.path.insert(0, "/root/gpt2")
    subprocess.run(
        [
            "torchrun",
            "--nproc_per_node=8",
            "/root/gpt2/train_ddp.py",
        ],
        check=True,
    )

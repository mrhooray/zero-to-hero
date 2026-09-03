from __future__ import annotations

import argparse
import hashlib
import json
import secrets
from pathlib import Path

import modal

if __package__:
    from .stage import MODEL_SPECS, RESULT_PREFIX
else:
    from stage import MODEL_SPECS, RESULT_PREFIX

CHECKPOINT_MOUNT = "/checkpoint"
REMOTE_SCRIPT = "/opt/infer/stage_checkpoint.py"
SANDBOX_TIMEOUT_SECONDS = 3 * 60 * 60
HUGGINGFACE_HUB_VERSION = "1.29.0"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Transfer one pinned checkpoint into a Modal Volume."
    )
    parser.add_argument("--model", choices=tuple(MODEL_SPECS), required=True)
    parser.add_argument("--volume", required=True)
    parser.add_argument("--create-volume", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    source_root = Path(__file__).resolve().parents[2]
    helper = source_root / "tools/checkpoint/stage.py"
    spec = MODEL_SPECS[args.model]
    job_name = secrets.token_hex(8)
    output = args.output or Path(
        f".artifacts/{_model_slug(args.model)}-checkpoint-stage.json"
    )
    launcher = {
        "helper_sha256": hashlib.sha256(helper.read_bytes()).hexdigest(),
        "job_name": job_name,
        "model_id": spec.model_id,
        "revision": spec.revision,
        "volume": args.volume,
    }
    result: dict[str, object] = {"launcher": launcher, "status": "failed"}
    sandbox = None
    stdout = ""
    stderr = ""

    try:
        app = modal.App.lookup(
            job_name,
            create_if_missing=True,
        )
        image = build_image(helper)
        volume = modal.Volume.from_name(
            args.volume,
            create_if_missing=args.create_volume,
            version=2,
        )
        with modal.enable_output():
            image = image.build(app)
            sandbox = create_stage_sandbox(app, image, volume, args.model, job_name)
            sandbox.wait(raise_on_termination=False)
            stdout = sandbox.stdout.read()
            stderr = sandbox.stderr.read()

        remote_result = parse_remote_result(stdout)
        result = {**remote_result, "launcher": launcher}
        if stderr:
            result["sandbox_stderr"] = stderr
        if sandbox.returncode != 0 and result.get("status") == "passed":
            result["status"] = "failed"
            result["error"] = f"stager exited with status {sandbox.returncode}"
    except (
        json.JSONDecodeError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        modal.exception.Error,
    ) as error:
        result["error"] = str(error)
        if stdout:
            result["sandbox_stdout"] = stdout
        if stderr:
            result["sandbox_stderr"] = stderr
    finally:
        if sandbox is not None:
            try:
                if sandbox.returncode is None:
                    sandbox.terminate(wait=True)
            except modal.exception.Error as error:
                result["cleanup_error"] = str(error)
            try:
                sandbox.detach()
            except modal.exception.Error as error:
                result["cleanup_error"] = str(error)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    print(output)
    return 0 if result.get("status") == "passed" else 1


def build_image(helper: Path):
    return (
        modal.Image.debian_slim(python_version="3.12")
        .entrypoint([])
        .pip_install(f"huggingface-hub=={HUGGINGFACE_HUB_VERSION}")
        .add_local_file(helper, REMOTE_SCRIPT, copy=True)
    )


def create_stage_sandbox(app, image, volume, model_id: str, job_name: str):
    return modal.Sandbox.create(
        "python",
        REMOTE_SCRIPT,
        "--model",
        model_id,
        "--root",
        CHECKPOINT_MOUNT,
        app=app,
        image=image,
        volumes={CHECKPOINT_MOUNT: volume},
        cpu=8,
        memory=8192,
        timeout=SANDBOX_TIMEOUT_SECONDS,
        tags={"job": job_name, "revision": MODEL_SPECS[model_id].revision},
    )


def parse_remote_result(stdout: str) -> dict[str, object]:
    payloads = [
        line.removeprefix(RESULT_PREFIX)
        for line in stdout.splitlines()
        if line.startswith(RESULT_PREFIX)
    ]
    if len(payloads) != 1:
        raise ValueError(f"expected one {RESULT_PREFIX} payload, found {len(payloads)}")
    result = json.loads(payloads[0])
    if not isinstance(result, dict):
        raise TypeError("remote result is not an object")
    return result


def _model_slug(model_id: str) -> str:
    return model_id.rsplit("/", 1)[-1].lower().replace(".", "-")


if __name__ == "__main__":
    raise SystemExit(main())

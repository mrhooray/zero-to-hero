from __future__ import annotations

import argparse
import importlib
import json
import os
import secrets
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import modal

if __package__:
    from . import benchmark_decode, decode, profiles
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    benchmark_decode = importlib.import_module("tools.benchmark.benchmark_decode")
    decode = importlib.import_module("tools.benchmark.decode")
    profiles = importlib.import_module("tools.benchmark.profiles")

RESULT_PREFIX = benchmark_decode.RESULT_PREFIX
SANDBOX_RESULT = "/tmp/infer-decode-result.json"
SANDBOX_RESULT_ACK = "/tmp/infer-decode-result.ack"
SANDBOX_RESULT_STATUS = "/tmp/infer-decode-result.status"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a decode benchmark in a model-specific Modal environment."
    )
    parser.add_argument("--profile", choices=profiles.NAMES, default=profiles.NAMES[0])
    parser.add_argument(
        "--engine", choices=("infer", "tokenspeed", "sglang"), default="infer"
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        choices=decode.CONCURRENCIES,
        nargs="+",
        default=decode.CONCURRENCIES,
    )
    parser.add_argument("--volume")
    parser.add_argument("--volume-subpath")
    parser.add_argument("--speculation", choices=("native", "none"), default="none")
    parser.add_argument("--output", type=Path)
    parser.add_argument("-e", "--environment")
    args = parser.parse_args(argv)
    if args.engine != "infer" and args.speculation != "none":
        parser.error(f"{args.engine} benchmark profile is target-only")

    profile = profiles.load(args.profile)
    volume_name = args.volume or profile.CHECKPOINT_VOLUME
    root = Path(__file__).resolve().parents[2]
    job_name = secrets.token_hex(8)
    output = args.output or Path(
        f".artifacts/{profile.NAME}-{args.engine}-decode-"
        + "-".join(f"c{value}" for value in args.concurrency)
        + ".json"
    )
    launcher = profile.launcher_metadata(
        root,
        volume=volume_name,
        concurrencies=args.concurrency,
        job_name=job_name,
        engine=args.engine,
    )
    launcher["modal_environment"] = args.environment
    launcher["checkpoint_volume_subpath"] = args.volume_subpath
    launcher["speculation"] = args.speculation
    result: dict[str, object] = {
        "launcher": launcher,
        "started_at": datetime.now(UTC).isoformat(),
        "status": "failed",
    }
    app_id = None
    remote = None
    remote_result_received = False
    started = datetime.now(UTC)
    try:
        app = modal.App(job_name)
        app_id = getattr(app, "app_id", None)
        image = profile.build_image(root, args.engine)
        volume = modal.Volume.from_name(
            volume_name,
            environment_name=args.environment,
            version=2,
        ).with_mount_options(read_only=True, sub_path=args.volume_subpath)
        with modal.enable_output():
            if args.engine == "infer":
                benchmark = create_benchmark_function(app, image, volume, profile)
                with app.run(environment_name=args.environment):
                    remote = benchmark.remote(tuple(args.concurrency), args.speculation)
                    app_id = getattr(app, "app_id", None) or app_id
            else:
                with app.run(environment_name=args.environment):
                    app_id = getattr(app, "app_id", None) or app_id
                    remote = run_sandbox(
                        app,
                        image.build(app),
                        volume,
                        profile,
                        args.profile,
                        args.engine,
                        args.concurrency,
                        args.speculation,
                        job_name,
                    )
                app_id = getattr(app, "app_id", None) or app_id
            if remote is None:
                raise RuntimeError("benchmark ended before returning a remote result")
            remote_returncode = int(remote["returncode"])
            remote_result = remote.get("result")
            if not isinstance(remote_result, dict):
                detail = str(
                    remote.get("stdout_tail") or remote.get("stderr_tail") or ""
                )
                raise TypeError(f"remote benchmark returned no result: {detail}")
            remote_result_received = True
            result = {
                **remote_result,
                "launcher": launcher,
                "started_at": started.isoformat(),
            }
            result["finished_at"] = datetime.now(UTC).isoformat()
            result["app_id"] = app_id
            stderr = str(remote.get("stderr_tail") or "")
            if stderr:
                result["launcher_remote_stderr"] = stderr
            if remote_returncode != 0 and result.get("status") == "passed":
                result["status"] = "failed"
                result["error"] = f"benchmark exited with status {remote_returncode}"
    except Exception as error:  # noqa: BLE001
        if remote_result_received:
            result["launcher_context_error"] = str(error)
        else:
            result["error"] = str(error)
        result["app_id"] = app_id
    finally:
        if app_id:
            stop_app(app_id, result, args.environment)
        result.setdefault("finished_at", datetime.now(UTC).isoformat())
        result["launcher_duration_s"] = (
            datetime.fromisoformat(result["finished_at"]) - started
        ).total_seconds()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    print(output)
    return 0 if result.get("status") == "passed" else 1


def create_benchmark_function(app, image, volume, profile):
    remote_runner = profile.REMOTE_ROOT / "tools/benchmark/benchmark_decode.py"

    @app.function(
        image=image,
        gpu=profile.GPU,
        cpu=profile.CPU,
        memory=profile.MEMORY_MIB,
        timeout=profile.FUNCTION_TIMEOUT_SECONDS,
        block_network=True,
        volumes={str(profile.CHECKPOINT_MOUNT): volume},
        env=profile.ENVIRONMENT,
        serialized=True,
    )
    def benchmark(concurrencies: tuple[int, ...], speculation: str):
        workload, config, metadata = profile.benchmark_inputs(speculation)
        workload_path = Path("/tmp/infer-decode-workload.json")
        config_path = Path("/tmp/infer-benchmark-server.json")
        metadata_path = Path("/tmp/infer-benchmark-metadata.json")
        write_json(workload_path, workload)
        write_json(config_path, config)
        write_json(metadata_path, metadata)
        completed = subprocess.run(
            [
                sys.executable,
                str(remote_runner),
                "--workload",
                str(workload_path),
                "--server-config",
                str(config_path),
                "--metadata",
                str(metadata_path),
                "--concurrency",
                *(str(concurrency) for concurrency in concurrencies),
            ],
            cwd=profile.REMOTE_ROOT,
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            check=False,
        )
        payloads = [
            line.removeprefix(RESULT_PREFIX)
            for line in completed.stdout.splitlines()
            if line.startswith(RESULT_PREFIX)
        ]
        parsed_result = json.loads(payloads[0]) if len(payloads) == 1 else None
        return {
            "returncode": completed.returncode,
            "result": parsed_result,
            "stdout_tail": completed.stdout[-8192:] if parsed_result is None else "",
            "stderr_tail": completed.stderr[-8192:],
        }

    return benchmark


def run_sandbox(
    app,
    image,
    volume,
    profile,
    profile_name: str,
    engine: str,
    concurrencies: list[int],
    speculation: str,
    job_name: str,
) -> dict[str, object]:
    runner = profile.REMOTE_ROOT / "tools/benchmark/run_modal.py"
    sandbox = modal.Sandbox.create(
        "python3",
        str(runner),
        "--profile",
        profile_name,
        "--engine",
        engine,
        "--speculation",
        speculation,
        "--concurrency",
        *(str(concurrency) for concurrency in concurrencies),
        app=app,
        name=job_name,
        image=image,
        gpu=profile.GPU,
        cpu=64,
        memory=256 * 1024,
        timeout=profile.FUNCTION_TIMEOUT_SECONDS,
        block_network=True,
        volumes={str(profile.CHECKPOINT_MOUNT): volume},
        env=profile.ENVIRONMENT,
        workdir=str(profile.REMOTE_ROOT),
    )
    try:
        receipt = sandbox.exec(
            "bash",
            "-lc",
            f"while [ ! -f {SANDBOX_RESULT_STATUS} ]; do sleep 1; done",
        )
        receipt.wait()
        try:
            parsed_result = json.loads(sandbox.filesystem.read_text(SANDBOX_RESULT))
        except Exception:  # noqa: BLE001
            parsed_result = None
        finally:
            sandbox.filesystem.write_text("received\n", SANDBOX_RESULT_ACK)
        sandbox.wait(raise_on_termination=False)
        stdout = sandbox.stdout.read()
        stderr = sandbox.stderr.read()
    finally:
        if sandbox.returncode is None:
            sandbox.terminate(wait=True)
        sandbox.detach()
    return {
        "returncode": sandbox.returncode,
        "result": parsed_result,
        "stdout_tail": stdout[-8192:] if parsed_result is None else "",
        "stderr_tail": stderr[-8192:],
    }


def write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def stop_app(
    app_id: str,
    result: dict[str, object],
    environment: str | None = None,
) -> None:
    command = [sys.executable, "-m", "modal", "app", "stop", "-y"]
    if environment is not None:
        command.extend(("-e", environment))
    command.append(app_id)
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    detail = completed.stderr or completed.stdout
    if completed.returncode == 0 or "App is already stopped." in detail:
        result["app_stopped"] = True
    else:
        result["cleanup_error"] = detail


if __name__ == "__main__":
    raise SystemExit(main())

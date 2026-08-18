from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import signal
import socket
import subprocess
from pathlib import Path
from time import perf_counter, sleep

if __package__:
    from . import decode
else:
    import decode

SERVER_LOG = Path("/tmp/infer-decode-server.log")
RESULT_PREFIX = "INFER_DECODE_BENCHMARK_RESULT="
SERVER_START_TIMEOUT_SECONDS = 30 * 60


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the decode benchmark against a configured server."
    )
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--server-config", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--working-directory", type=Path, default=Path("/opt/infer"))
    parser.add_argument(
        "--concurrency",
        type=int,
        choices=decode.CONCURRENCIES,
        nargs="+",
        default=decode.CONCURRENCIES,
    )
    args = parser.parse_args(argv)
    if len(set(args.concurrency)) != len(args.concurrency):
        parser.error("--concurrency values must be unique")

    started = perf_counter()
    server = None
    log = None
    result: dict[str, object] = {
        "concurrencies": args.concurrency,
        "status": "failed",
    }
    try:
        workload = _read_json(args.workload)
        config = _read_json(args.server_config)
        workload_validation = decode.validate_workload(workload)
        server_validation = decode.validate_server_config(config)
        _validate_model_identity(workload, config)
        result.update(
            schema=decode.RECEIPT_SCHEMA,
            contract=workload["contract"],
            model=workload["model"],
            server_config=config,
            server_validation=server_validation,
            workload_validation=workload_validation,
        )
        if args.metadata is not None:
            result.update(_read_json(args.metadata))

        log = SERVER_LOG.open("w", encoding="utf-8")
        server = subprocess.Popen(
            config["launch_command"],
            cwd=args.working_directory,
            env=os.environ.copy(),
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        wait_for_server(server, str(config["endpoint"]))
        if config.get("engine") == "tokenspeed":
            result["startup_validation"] = wait_for_tokenizer(
                server,
                str(config["endpoint"]),
                str(config["model_id"]),
            )
        decode.run_decode_benchmark(
            result,
            workload,
            config,
            tuple(args.concurrency),
        )
        result["status"] = "passed"
    except Exception as error:  # noqa: BLE001
        result["error"] = str(error)
    finally:
        if server is not None:
            stop_server(server)
            result["server_returncode"] = server.returncode
        if log is not None:
            log.close()
        if SERVER_LOG.exists():
            result["server_log_tail"] = SERVER_LOG.read_text(encoding="utf-8")[-20_000:]
        result["duration_s"] = perf_counter() - started

    compact_success_receipt(result)
    if args.output is None:
        print(RESULT_PREFIX + json.dumps(result, sort_keys=True), flush=True)
    else:
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return 0 if result.get("status") == "passed" else 1


def compact_success_receipt(result: dict[str, object]) -> None:
    if result.get("status") != "passed":
        return
    for primer in result.get("priming", []):
        primer.pop("requests", None)
    for point in result.get("benchmarks", []):
        content_hashes = point.pop("content_sha256", [])
        point.pop("requests", None)
        point["distinct_measured_content_count"] = len(content_hashes)
        point["measured_content_set_sha256"] = hashlib.sha256(
            "".join(sorted(content_hashes)).encode()
        ).hexdigest()


def wait_for_server(
    server: subprocess.Popen[bytes],
    endpoint: str,
    timeout: int = SERVER_START_TIMEOUT_SECONDS,
) -> None:
    origin = decode.parse_endpoint(endpoint)
    assert origin.hostname is not None and origin.port is not None
    deadline = perf_counter() + timeout
    while perf_counter() < deadline:
        if server.poll() is not None:
            raise RuntimeError(f"server exited before listening: {server.returncode}")
        try:
            with socket.create_connection((origin.hostname, origin.port), timeout=1):
                return
        except OSError:
            sleep(1)
    raise TimeoutError("server did not start listening")


def stop_server(server: subprocess.Popen[bytes]) -> None:
    if server.poll() is None:
        os.killpg(server.pid, signal.SIGTERM)
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(server.pid, signal.SIGKILL)
            server.wait()


def wait_for_tokenizer(
    server: subprocess.Popen[bytes],
    endpoint: str,
    model_id: str,
    timeout: int = SERVER_START_TIMEOUT_SECONDS,
) -> dict[str, str]:
    origin = decode.parse_endpoint(endpoint)
    body = json.dumps(
        {
            "model": model_id,
            "messages": [{"role": "user", "content": "Startup readiness probe."}],
            "max_tokens": 1,
            "temperature": 0,
        }
    )
    deadline = perf_counter() + timeout
    last_error = "tokenizer not ready"
    while perf_counter() < deadline:
        if server.poll() is not None:
            raise RuntimeError(
                f"server exited before tokenizer was ready: {server.returncode}"
            )
        connection = http.client.HTTPConnection(
            origin.hostname, origin.port, timeout=10
        )
        try:
            connection.request(
                "POST",
                "/v1/chat/completions",
                body,
                {"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            detail = response.read(4000)
            if response.status == 200:
                return {"chat_completion": "ready"}
            last_error = f"HTTP status {response.status}: {detail!r}"
        except OSError as error:
            last_error = f"{type(error).__name__}: {error}"
        finally:
            connection.close()
        sleep(1)
    raise TimeoutError(f"tokenizer did not become ready: {last_error}")


def _validate_model_identity(
    workload: dict[str, object], config: dict[str, object]
) -> None:
    model = workload["model"]
    checkpoint = config["checkpoint"]
    assert isinstance(model, dict) and isinstance(checkpoint, dict)
    if model["key"] != config["model_key"]:
        raise ValueError("workload and server model keys differ")
    if model["checkpoint_revision"] != checkpoint["revision"]:
        raise ValueError("workload and server checkpoint revisions differ")


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


if __name__ == "__main__":
    raise SystemExit(main())

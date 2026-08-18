from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import monotonic, sleep

from tools.benchmark import benchmark_decode, decode, profiles

RESULT = Path("/tmp/infer-decode-result.json")
RESULT_ACK = Path("/tmp/infer-decode-result.ack")
RESULT_STATUS = Path("/tmp/infer-decode-result.status")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=profiles.NAMES, required=True)
    parser.add_argument("--engine", choices=("tokenspeed", "sglang"), required=True)
    parser.add_argument("--speculation", choices=("none",), default="none")
    parser.add_argument(
        "--concurrency",
        type=int,
        choices=decode.CONCURRENCIES,
        nargs="+",
        required=True,
    )
    args = parser.parse_args()

    profile = profiles.load(args.profile)
    workload, config, metadata = profile.benchmark_inputs(args.speculation, args.engine)
    paths = {
        "--workload": Path("/tmp/infer-decode-workload.json"),
        "--server-config": Path("/tmp/infer-benchmark-server.json"),
        "--metadata": Path("/tmp/infer-benchmark-metadata.json"),
    }
    for path, value in zip(paths.values(), (workload, config, metadata), strict=True):
        path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    returncode = benchmark_decode.main(
        [
            *(value for item in paths.items() for value in (item[0], str(item[1]))),
            "--output",
            str(RESULT),
            "--concurrency",
            *(str(concurrency) for concurrency in args.concurrency),
        ]
    )
    RESULT_STATUS.write_text(f"{returncode}\n", encoding="utf-8")
    deadline = monotonic() + 60
    while not RESULT_ACK.exists() and monotonic() < deadline:
        sleep(0.1)
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())

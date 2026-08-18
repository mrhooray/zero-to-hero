from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

from tokenizers import Tokenizer

CASES = (
    ("arithmetic", "137 + 248 ="),
    ("identifier", "Repeat this identifier exactly: nova-x7."),
    ("unicode", "Return only the UTF-8 text: café 東京."),
    ("code", "Write a Python function named add that returns the sum of two integers."),
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare deterministic decoded output from two chat engines."
    )
    parser.add_argument("--candidate-endpoint", required=True)
    parser.add_argument("--reference-endpoint", required=True)
    parser.add_argument("--candidate-model", required=True)
    parser.add_argument("--reference-model", required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--minimum-token-agreement", type=float, default=0.90)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.max_tokens <= 0:
        parser.error("--max-tokens must be positive")
    if not 0 <= args.minimum_token_agreement <= 1:
        parser.error("--minimum-token-agreement must be between zero and one")

    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    rows = []
    for case_id, prompt in CASES:
        messages = [{"role": "user", "content": prompt}]
        candidate = request_completion(
            args.candidate_endpoint,
            args.candidate_model,
            messages,
            args.max_tokens,
        )
        reference = request_completion(
            args.reference_endpoint,
            args.reference_model,
            messages,
            args.max_tokens,
        )
        candidate_ids = tokenizer.encode(candidate, add_special_tokens=False).ids
        reference_ids = tokenizer.encode(reference, add_special_tokens=False).ids
        rows.append(compare_case(case_id, candidate_ids, reference_ids))

    passed = all(row["token_agreement"] >= args.minimum_token_agreement for row in rows)
    result = {
        "schema": "infer-token-parity/v1",
        "status": "passed" if passed else "failed",
        "method": "greedy decoded text re-tokenized with the checkpoint tokenizer",
        "candidate": {
            "endpoint": args.candidate_endpoint,
            "model": args.candidate_model,
        },
        "reference": {
            "endpoint": args.reference_endpoint,
            "model": args.reference_model,
        },
        "max_tokens": args.max_tokens,
        "minimum_token_agreement": args.minimum_token_agreement,
        "cases": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(args.output)
    return 0 if passed else 1


def request_completion(
    endpoint: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
) -> str:
    payload = json.dumps(
        {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0,
        }
    ).encode()
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        result = json.load(response)
    content = result["choices"][0]["message"]["content"]
    if not isinstance(content, str):
        raise TypeError("chat completion content must be a string")
    return content


def compare_case(
    case_id: str, candidate: list[int], reference: list[int]
) -> dict[str, object]:
    distance = edit_distance(candidate, reference)
    width = max(len(candidate), len(reference))
    common_prefix = 0
    for left, right in zip(candidate, reference, strict=False):
        if left != right:
            break
        common_prefix += 1
    return {
        "id": case_id,
        "candidate_tokens": len(candidate),
        "reference_tokens": len(reference),
        "common_prefix_tokens": common_prefix,
        "edit_distance": distance,
        "token_agreement": 1.0 if not width else 1.0 - distance / width,
        "exact": candidate == reference,
    }


def edit_distance(left: list[int], right: list[int]) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_token in enumerate(left, 1):
        current = [left_index]
        for right_index, right_token in enumerate(right, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_token != right_token),
                )
            )
        previous = current
    return previous[-1]


if __name__ == "__main__":
    raise SystemExit(main())

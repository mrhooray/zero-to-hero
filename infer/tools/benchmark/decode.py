from __future__ import annotations

import hashlib
import http.client
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import pairwise
from threading import Barrier
from urllib.parse import SplitResult, urlsplit

WORKLOAD_SCHEMA = "infer-decode-workload/v1"
SERVER_CONFIG_SCHEMA = "infer-benchmark-server-config/v1"
RECEIPT_SCHEMA = "infer-decode-benchmark-result/v1"
PROMPT_TOKENS = 8_193
SHARED_PREFIX_TOKENS = 8_192
SUFFIX_TOKENS = 1
OUTPUT_TOKENS = 1_024
GROUP_COUNT = 4
CONCURRENCIES = (4, 16, 64, 128)
MEASURED_PER_SLOT = 20
FILLER = " context"
HTTP_TIMEOUT_SECONDS = 2 * 60 * 60


def build_workload(
    codec,
    *,
    model_key: str,
    checkpoint_revision: str,
) -> dict[str, object]:
    groups = [build_group(codec, model_key, group) for group in range(GROUP_COUNT)]
    validate_group_prefixes(groups)
    precomputed = {
        int(group["group"]): _request_digests(group) for group in groups
    }
    requests = [
        build_request(
            spec, groups[int(spec["group"])], precomputed[int(spec["group"])]
        )
        for spec in request_schedule()
    ]

    workload: dict[str, object] = {
        "schema": WORKLOAD_SCHEMA,
        "contract": contract(),
        "model": {
            "key": model_key,
            "checkpoint_revision": checkpoint_revision,
            "renderer": type(codec).__name__,
        },
        "groups": groups,
        "requests": requests,
    }
    validate_workload(workload, codec)
    return workload


def build_group(codec, model_key: str, group: int) -> dict[str, object]:
    identity = hashlib.sha256(
        f"prefix-rolling-v1:{model_key}:group:{group}".encode()
    ).hexdigest()
    header = (
        f"Repeated-prefix benchmark group {group}; immutable identity {identity}. "
        + (f"Group-{group} identity segment {identity}. " * 4)
        + "The shared context follows."
    )
    filler_count = PROMPT_TOKENS
    attempted: set[int] = set()
    for _ in range(12):
        if filler_count in attempted or filler_count < 0:
            break
        attempted.add(filler_count)
        content = header + FILLER * filler_count
        token_ids = encode(codec, content)
        if len(token_ids) == PROMPT_TOKENS:
            return {
                "group": group,
                "shared_content": content,
                "rendered_prefix_token_ids": token_ids[:SHARED_PREFIX_TOKENS],
                "rendered_prefix_sha256": digest_json(
                    token_ids[:SHARED_PREFIX_TOKENS]
                ),
                "terminal_token_id": token_ids[-1],
            }
        filler_count += PROMPT_TOKENS - len(token_ids)

    for count in range(max(0, filler_count - 32), filler_count + 33):
        if count in attempted:
            continue
        content = header + FILLER * count
        token_ids = encode(codec, content)
        if len(token_ids) == PROMPT_TOKENS:
            return {
                "group": group,
                "shared_content": content,
                "rendered_prefix_token_ids": token_ids[:SHARED_PREFIX_TOKENS],
                "rendered_prefix_sha256": digest_json(
                    token_ids[:SHARED_PREFIX_TOKENS]
                ),
                "terminal_token_id": token_ids[-1],
            }
    raise RuntimeError(f"could not build exact 8193-token group {group}")


def build_request(
    spec: dict[str, object],
    group: dict[str, object],
    precomputed: tuple[str, str, str] | None = None,
) -> dict[str, object]:
    content = str(group["shared_content"])
    suffix_content = ""
    suffix_ids = [int(group["terminal_token_id"])]
    token_ids = group["rendered_prefix_token_ids"] + suffix_ids
    messages = [{"role": "user", "content": content}]
    if precomputed is None:
        messages_sha256 = digest_json(messages)
        prompt_token_sha256 = digest_json(token_ids)
        suffix_token_sha256 = digest_json(suffix_ids)
    else:
        messages_sha256, prompt_token_sha256, suffix_token_sha256 = precomputed
    return {
        **spec,
        "messages_sha256": messages_sha256,
        "prompt_token_sha256": prompt_token_sha256,
        "suffix_content": suffix_content,
        "suffix_first_token": suffix_ids[0],
        "suffix_token_ids": suffix_ids,
        "suffix_token_sha256": suffix_token_sha256,
    }


def _request_digests(group: dict[str, object]) -> tuple[str, str, str]:
    suffix_ids = [int(group["terminal_token_id"])]
    token_ids = group["rendered_prefix_token_ids"] + suffix_ids
    messages = [{"role": "user", "content": str(group["shared_content"])}]
    return (
        digest_json(messages),
        digest_json(token_ids),
        digest_json(suffix_ids),
    )


def encode(codec, content: str) -> list[int]:
    return list(codec.encode_messages([{"role": "user", "content": content}]))


def request_schedule() -> list[dict[str, object]]:
    specs = []
    ordinal = 0
    for wave in (1, 2):
        for slot in range(GROUP_COUNT):
            specs.append(
                {
                    "ordinal": ordinal,
                    "phase": "prime",
                    "prime_wave": wave,
                    "concurrency": GROUP_COUNT,
                    "slot": slot,
                    "iteration": wave - 1,
                    "group": slot,
                }
            )
            ordinal += 1
    for concurrency in CONCURRENCIES:
        for slot in range(concurrency):
            specs.append(
                {
                    "ordinal": ordinal,
                    "phase": "warmup",
                    "prime_wave": None,
                    "concurrency": concurrency,
                    "slot": slot,
                    "iteration": 0,
                    "group": slot % GROUP_COUNT,
                }
            )
            ordinal += 1
            for iteration in range(MEASURED_PER_SLOT):
                specs.append(
                    {
                        "ordinal": ordinal,
                        "phase": "measured",
                        "prime_wave": None,
                        "concurrency": concurrency,
                        "slot": slot,
                        "iteration": iteration,
                        "group": slot % GROUP_COUNT,
                    }
                )
                ordinal += 1
    return specs


# Workload and server validation


def validate_workload(workload: dict[str, object], codec=None) -> dict[str, object]:
    if workload.get("schema") != WORKLOAD_SCHEMA:
        raise ValueError("unsupported workload schema")
    if workload.get("contract") != contract():
        raise ValueError("workload contract changed")
    model = workload.get("model")
    if not isinstance(model, dict) or not isinstance(model.get("key"), str):
        raise TypeError("workload model is invalid")
    groups = workload.get("groups")
    requests = workload.get("requests")
    if not isinstance(groups, list) or not isinstance(requests, list):
        raise TypeError("workload groups and requests must be lists")
    validate_group_prefixes(groups)
    expected_schedule = request_schedule()
    if len(requests) != len(expected_schedule):
        raise ValueError("workload request count changed")

    by_group = {int(group["group"]): group for group in groups}
    suffix_by_group: dict[int, int] = {}
    digest_cache: dict[tuple, str] = {}
    render_cache: dict[tuple, list] = {}
    schedule_keys = (
        "ordinal",
        "phase",
        "prime_wave",
        "concurrency",
        "slot",
        "iteration",
        "group",
    )
    for request, expected in zip(requests, expected_schedule, strict=True):
        if not isinstance(request, dict):
            raise TypeError("workload request must be an object")
        if {key: request.get(key) for key in schedule_keys} != expected:
            raise ValueError(
                f"request schedule changed at ordinal {expected['ordinal']}"
            )
        group = by_group[int(request["group"])]
        suffix_ids = request.get("suffix_token_ids")
        if (
            not isinstance(suffix_ids, list)
            or len(suffix_ids) != SUFFIX_TOKENS
            or any(type(token_id) is not int or token_id < 0 for token_id in suffix_ids)
        ):
            raise ValueError("request suffix token vector is invalid")
        first_token = suffix_ids[0]
        if request.get("suffix_first_token") != first_token:
            raise ValueError("suffix-first token receipt changed")
        if first_token != group["terminal_token_id"]:
            raise ValueError("request terminal token differs from its prefix group")
        group_id = int(request["group"])
        if group_id in suffix_by_group and suffix_by_group[group_id] != first_token:
            raise ValueError("a prefix group used multiple suffix tokens")
        suffix_by_group[group_id] = first_token
        if request.get("suffix_token_sha256") != digest_json(suffix_ids):
            raise ValueError("suffix token digest changed")
        token_ids = group["rendered_prefix_token_ids"] + suffix_ids
        if len(token_ids) != PROMPT_TOKENS:
            raise ValueError("rendered prompt length changed")
        prompt_key = ("prompt", group_id, tuple(suffix_ids))
        expected_prompt = _memo(
            digest_cache, prompt_key, lambda: digest_json(token_ids)
        )
        if request.get("prompt_token_sha256") != expected_prompt:
            raise ValueError("prompt token digest changed")
        suffix_content = str(request.get("suffix_content"))
        content = str(group["shared_content"]) + suffix_content
        messages = [{"role": "user", "content": content}]
        message_key = ("messages", group_id, suffix_content)
        expected_messages = _memo(
            digest_cache, message_key, lambda: digest_json(messages)
        )
        if request.get("messages_sha256") != expected_messages:
            raise ValueError("request message digest changed")
        if codec is not None:
            render_key = ("render", group_id, suffix_content)
            rendered = _memo(
                render_cache, render_key, lambda: encode(codec, content)
            )
            if rendered != token_ids:
                raise ValueError(
                    f"model renderer disagrees with ordinal {request['ordinal']}"
                )

    group_counts = {
        group: sum(int(request["group"]) == group for request in requests)
        for group in range(GROUP_COUNT)
    }
    if set(group_counts.values()) != {len(expected_schedule) // GROUP_COUNT}:
        raise ValueError("four-group request balance changed")
    if len(suffix_by_group) != GROUP_COUNT:
        raise ValueError("a prefix group is missing its terminal token")
    return {
        "status": "validated",
        "renderer_rechecked": codec is not None,
        "request_count": len(requests),
        "group_counts": group_counts,
        "rendered_prompt_tokens": PROMPT_TOKENS,
        "within_group_lcp_tokens": SHARED_PREFIX_TOKENS,
        "suffix_tokens": SUFFIX_TOKENS,
        "unique_suffix_first_tokens": len(set(suffix_by_group.values())),
        "measured_completion_counts": {
            str(concurrency): concurrency * MEASURED_PER_SLOT
            for concurrency in CONCURRENCIES
        },
    }


def validate_group_prefixes(groups: list[dict[str, object]]) -> None:
    if len(groups) != GROUP_COUNT or [group.get("group") for group in groups] != list(
        range(GROUP_COUNT)
    ):
        raise ValueError("workload must contain ordered groups 0..3")
    prefixes = []
    for group in groups:
        prefix = group.get("rendered_prefix_token_ids")
        if (
            not isinstance(prefix, list)
            or len(prefix) != SHARED_PREFIX_TOKENS
            or any(type(token_id) is not int or token_id < 0 for token_id in prefix)
        ):
            raise ValueError("group rendered prefix vector is invalid")
        if group.get("rendered_prefix_sha256") != digest_json(prefix):
            raise ValueError("group prefix digest changed")
        if not isinstance(group.get("shared_content"), str):
            raise TypeError("group shared content must be text")
        if type(group.get("terminal_token_id")) is not int:
            raise TypeError("group terminal token must be an integer")
        prefixes.append(prefix)
    for left in range(GROUP_COUNT):
        for right in range(left + 1, GROUP_COUNT):
            if longest_common_prefix((prefixes[left], prefixes[right])) >= 128:
                raise ValueError("prefix groups do not differ within 128 tokens")


def validate_server_config(config: dict[str, object]) -> dict[str, object]:
    if config.get("schema") != SERVER_CONFIG_SCHEMA:
        raise ValueError("unsupported server config schema")
    engine = config.get("engine")
    if not isinstance(engine, str) or not engine:
        raise ValueError("server engine is required")
    if not isinstance(config.get("model_key"), str) or not config["model_key"]:
        raise ValueError("server model key is invalid")
    if not isinstance(config.get("model_id"), str) or not config["model_id"]:
        raise ValueError("server model ID is required")
    endpoint = parse_endpoint(str(config.get("endpoint", "")))
    if endpoint.hostname not in {"127.0.0.1", "localhost"}:
        raise ValueError("benchmark endpoint must use loopback")
    command = config.get("launch_command")
    if (
        not isinstance(command, list)
        or not command
        or not all(isinstance(value, str) for value in command)
    ):
        raise ValueError("launch command must be a nonempty string array")
    if not isinstance(config.get("resolved_server_config"), dict):
        raise TypeError("resolved server configuration is required")
    checkpoint = config.get("checkpoint")
    if (
        not isinstance(checkpoint, dict)
        or not isinstance(checkpoint.get("revision"), str)
        or not checkpoint["revision"]
    ):
        raise ValueError("checkpoint revision is required")
    return {
        "status": "validated",
        "engine": engine,
        "endpoint": endpoint.geturl(),
    }


# Rolling client


def run_decode_benchmark(
    receipt: dict[str, object],
    workload: dict[str, object],
    config: dict[str, object],
    concurrencies: tuple[int, ...] = CONCURRENCIES,
) -> None:
    endpoint = parse_endpoint(str(config["endpoint"]))
    groups = {int(group["group"]): group for group in workload["groups"]}
    requests = workload["requests"]
    origin = time.perf_counter()
    primers = []
    for primer in priming_waves(requests):
        rows = run_priming_wave(
            primer["specs"],
            groups,
            config,
            endpoint,
            origin,
            expected_cached=primer["expected_cached"],
        )
        primers.append(
            {
                "wave": primer["wave"],
                "stage": primer["stage"],
                "requests": rows,
                "overlap": validate_initial_overlap(rows, GROUP_COUNT),
                "cached_tokens": [row["cached_tokens"] for row in rows],
            }
        )
        receipt["priming"] = primers

    priming_qualification = qualify_priming(primers)
    receipt["qualification"] = priming_qualification
    if not all(priming_qualification.values()):
        raise RuntimeError(f"priming qualification failed: {priming_qualification}")

    benchmarks = []
    receipt["executed_concurrencies"] = list(concurrencies)
    for concurrency in concurrencies:
        point = run_concurrency(
            requests,
            groups,
            config,
            endpoint,
            origin,
            concurrency,
        )
        benchmarks.append(point)
        receipt["benchmarks"] = benchmarks
    receipt["qualification"] = priming_qualification | {
        "warmup_and_measured_cached_tokens": SHARED_PREFIX_TOKENS,
        "rolling_constant_occupancy": all(
            point["rolling_occupancy"]["valid"] for point in benchmarks
        ),
        "speculation": "none",
        "prefix_cache": "enabled",
    }


def qualify_priming(primers: list[dict[str, object]]) -> dict[str, bool]:
    seeds = [primer for primer in primers if primer["stage"] == "seed"]
    promotions = [primer for primer in primers if primer["stage"] == "promotion"]
    complete = (
        len(primers) == 2
        and len(seeds) == 1
        and len(promotions) == 1
        and all(len(primer["requests"]) == GROUP_COUNT for primer in primers)
        and all(primer["overlap"].get("valid") is True for primer in primers)
        and all(
            {row["group"] for row in primer["requests"]} == set(range(GROUP_COUNT))
            for primer in primers
        )
    )
    return {
        "four_way_seed_wave_cold": complete
        and all(row["cached_tokens"] == 0 for row in seeds[0]["requests"]),
        "four_way_promotion_wave_completed": complete,
        "four_way_promotion_wave_cached_8192": complete
        and all(
            row["cached_tokens"] == SHARED_PREFIX_TOKENS
            for row in promotions[0]["requests"]
        ),
        "four_distinct_prefix_groups_seeded_and_promoted": complete,
    }


def priming_waves(
    requests: list[dict[str, object]],
) -> list[dict[str, object]]:
    sources = [request for request in requests if request["phase"] == "prime"]
    schedule_keys = (
        "ordinal",
        "phase",
        "prime_wave",
        "concurrency",
        "slot",
        "iteration",
        "group",
    )
    source_schedule = [
        {key: source[key] for key in schedule_keys} for source in sources
    ]
    if source_schedule != request_schedule()[: 2 * GROUP_COUNT]:
        raise ValueError("priming source schedule changed")
    return [
        {
            "wave": wave,
            "stage": stage,
            "expected_cached": expected_cached,
            "specs": [source for source in sources if source["prime_wave"] == wave],
        }
        for wave, stage, expected_cached in (
            (1, "seed", 0),
            (2, "promotion", SHARED_PREFIX_TOKENS),
        )
    ]


def run_priming_wave(
    specs: list[dict[str, object]],
    groups: dict[int, dict[str, object]],
    config: dict[str, object],
    endpoint: SplitResult,
    origin: float,
    *,
    expected_cached: int | None,
) -> list[dict[str, object]]:
    if len(specs) != GROUP_COUNT:
        raise ValueError("a priming wave must contain four requests")
    barrier = Barrier(GROUP_COUNT)

    def worker(spec: dict[str, object]) -> dict[str, object]:
        barrier.wait()
        return execute_request(
            spec,
            groups,
            config,
            endpoint,
            origin,
            expected_cached=expected_cached,
        )

    with ThreadPoolExecutor(max_workers=GROUP_COUNT) as pool:
        futures = [pool.submit(worker, spec) for spec in specs]
        try:
            rows = [future.result() for future in as_completed(futures)]
        except BaseException:
            barrier.abort()
            raise
    return sorted(rows, key=lambda row: int(row["ordinal"]))


def run_concurrency(
    requests: list[dict[str, object]],
    groups: dict[int, dict[str, object]],
    config: dict[str, object],
    endpoint: SplitResult,
    origin: float,
    concurrency: int,
) -> dict[str, object]:
    selected = [
        request
        for request in requests
        if request["concurrency"] == concurrency and request["phase"] != "prime"
    ]
    by_slot = {
        slot: [request for request in selected if request["slot"] == slot]
        for slot in range(concurrency)
    }
    barrier_offset = [None]

    def release_measurement() -> None:
        barrier_offset[0] = time.perf_counter() - origin

    barrier = Barrier(concurrency, action=release_measurement)

    def worker(slot: int) -> list[dict[str, object]]:
        specs = by_slot[slot]
        if [spec["phase"] for spec in specs] != ["warmup"] + [
            "measured"
        ] * MEASURED_PER_SLOT:
            raise ValueError(f"slot {slot} schedule changed")
        rows = [
            execute_request(
                specs[0],
                groups,
                config,
                endpoint,
                origin,
                expected_cached=SHARED_PREFIX_TOKENS,
            )
        ]
        barrier.wait()
        for spec in specs[1:]:
            rows.append(
                execute_request(
                    spec,
                    groups,
                    config,
                    endpoint,
                    origin,
                    expected_cached=SHARED_PREFIX_TOKENS,
                )
            )
        return rows

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(worker, slot) for slot in range(concurrency)]
        try:
            rows = [row for future in as_completed(futures) for row in future.result()]
        except BaseException:
            barrier.abort()
            raise
    rows.sort(key=lambda row: int(row["ordinal"]))
    if barrier_offset[0] is None:
        raise RuntimeError("warmup barrier did not release")
    warmups = [row for row in rows if row["phase"] == "warmup"]
    measured = [row for row in rows if row["phase"] == "measured"]
    if max(float(row["end_offset_seconds"]) for row in warmups) > barrier_offset[0]:
        raise RuntimeError("warmup overlapped the measurement window")
    if min(float(row["start_offset_seconds"]) for row in measured) < barrier_offset[0]:
        raise RuntimeError("measurement started before all warmups completed")
    return aggregate_point(concurrency, warmups, measured)


def execute_request(
    spec: dict[str, object],
    groups: dict[int, dict[str, object]],
    config: dict[str, object],
    endpoint: SplitResult,
    origin: float,
    *,
    expected_cached: int | None,
) -> dict[str, object]:
    group = groups[int(spec["group"])]
    content = str(group["shared_content"]) + str(spec["suffix_content"])
    response = request_stream(
        endpoint,
        model_id=str(config["model_id"]),
        messages=[{"role": "user", "content": content}],
        origin=origin,
        expected_cached=expected_cached,
    )
    keys = (
        "ordinal",
        "phase",
        "prime_wave",
        "concurrency",
        "slot",
        "iteration",
        "group",
        "messages_sha256",
        "prompt_token_sha256",
        "suffix_first_token",
    )
    return {key: spec[key] for key in keys} | response


def request_stream(
    endpoint: SplitResult,
    *,
    model_id: str,
    messages: list[dict[str, str]],
    origin: float,
    expected_cached: int | None,
) -> dict[str, object]:
    body = request_body(model_id=model_id, messages=messages)
    connection = http.client.HTTPConnection(
        endpoint.hostname,
        endpoint.port,
        timeout=HTTP_TIMEOUT_SECONDS,
    )
    started = time.perf_counter()
    try:
        connection.request(
            "POST",
            "/v1/chat/completions",
            body,
            {"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        if response.status != 200:
            raise RuntimeError(
                f"HTTP status {response.status}: {response.read(4000)!r}"
            )
        content_type = str(response.getheader("content-type"))
        if not content_type.startswith("text/event-stream"):
            raise RuntimeError(f"unexpected content type: {content_type!r}")
        parsed = parse_stream(response)
    finally:
        connection.close()
    ended = time.perf_counter()
    usage = parsed.pop("usage")
    cached_tokens = validate_usage(usage, expected_cached)
    first = float(parsed.pop("first_content_at"))
    last = float(parsed.pop("last_content_at"))
    return {
        **parsed,
        "cached_tokens": cached_tokens,
        "prompt_tokens": PROMPT_TOKENS,
        "completion_tokens": OUTPUT_TOKENS,
        "total_tokens": PROMPT_TOKENS + OUTPUT_TOKENS,
        "start_offset_seconds": started - origin,
        "first_token_offset_seconds": first - origin,
        "end_offset_seconds": ended - origin,
        "ttft_seconds": first - started,
        "tpot_seconds": (last - first) / (OUTPUT_TOKENS - 1),
        "e2e_seconds": ended - started,
    }


def request_body(
    *,
    model_id: str,
    messages: list[dict[str, str]],
) -> str:
    payload = {
        "model": model_id,
        "messages": messages,
        "max_tokens": OUTPUT_TOKENS,
        "temperature": 0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    return json.dumps(payload, separators=(",", ":"))


def parse_stream(response: http.client.HTTPResponse) -> dict[str, object]:
    content = []
    completion_ids: set[str] = set()
    finish_reasons = []
    event_count = 0
    done_count = 0
    usage = None
    finished = False
    first_content_at = None
    last_content_at = None
    while line := response.readline():
        line = line.strip()
        if not line:
            continue
        if not line.startswith(b"data: "):
            raise RuntimeError(f"unexpected SSE line: {line[:200]!r}")
        payload = line.removeprefix(b"data: ")
        if payload == b"[DONE]":
            if not finished or usage is None:
                raise RuntimeError("received [DONE] before finish or usage")
            done_count += 1
            continue
        if done_count:
            raise RuntimeError("received an event after [DONE]")
        event_count += 1
        chunk = json.loads(payload)
        if isinstance(chunk.get("id"), str):
            completion_ids.add(chunk["id"])
        choices = chunk.get("choices")
        if choices == [] and "usage" in chunk:
            if (
                not finished
                or usage is not None
                or not isinstance(chunk["usage"], dict)
            ):
                raise RuntimeError("stream returned invalid or duplicate usage")
            usage = chunk["usage"]
            continue
        if finished or usage is not None:
            raise RuntimeError("received content after finish or usage")
        if not isinstance(choices, list) or len(choices) != 1:
            raise RuntimeError(f"stream returned invalid choices: {chunk}")
        choice = choices[0]
        delta = choice.get("delta")
        if not isinstance(delta, dict):
            raise TypeError("stream delta is not an object")
        fields = (
            choice.get("text"),
            delta.get("reasoning_content"),
            delta.get("content"),
        )
        if any(value is not None and not isinstance(value, str) for value in fields):
            raise TypeError("stream content is not text")
        parts = [value for value in fields if value]
        if len(parts) > 1:
            raise RuntimeError("stream populated multiple content fields")
        text = "".join(parts)
        if text:
            last_content_at = time.perf_counter()
            if first_content_at is None:
                first_content_at = last_content_at
        content.append(text)
        if choice.get("finish_reason") is not None:
            finish_reasons.append(choice["finish_reason"])
            finished = True
    if (
        done_count != 1
        or usage is None
        or first_content_at is None
        or last_content_at is None
    ):
        raise RuntimeError("stream omitted content, usage, or exactly one [DONE]")
    if finish_reasons != ["length"]:
        raise RuntimeError(f"wrong finish reason sequence: {finish_reasons}")
    if len(completion_ids) != 1:
        raise RuntimeError(f"stream returned {len(completion_ids)} completion IDs")
    completion = "".join(content)
    return {
        "completion_id": next(iter(completion_ids)),
        "content_sha256": hashlib.sha256(completion.encode()).hexdigest(),
        "finish_reason": "length",
        "http_status": 200,
        "sse_done": True,
        "sse_event_count": event_count,
        "first_content_at": first_content_at,
        "last_content_at": last_content_at,
        "usage": usage,
    }


def validate_usage(
    usage: object,
    expected_cached: int | None,
) -> int:
    if not isinstance(usage, dict):
        raise TypeError("usage must be an object")
    expected = {
        "prompt_tokens": PROMPT_TOKENS,
        "completion_tokens": OUTPUT_TOKENS,
        "total_tokens": PROMPT_TOKENS + OUTPUT_TOKENS,
    }
    if any(usage.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"wrong authoritative usage counts: {usage}")
    details = usage.get("prompt_tokens_details")
    if details is None and expected_cached == 0:
        cached_tokens = 0
    elif not isinstance(details, dict) or type(details.get("cached_tokens")) is not int:
        raise RuntimeError(f"response omitted exact cached_tokens: {usage}")
    else:
        cached_tokens = int(details["cached_tokens"])
    if not 0 <= cached_tokens <= PROMPT_TOKENS:
        raise RuntimeError(f"cached token count is out of range: {cached_tokens}")
    if expected_cached is not None and cached_tokens != expected_cached:
        raise RuntimeError(
            f"expected {expected_cached} cached tokens, received {cached_tokens}"
        )
    return cached_tokens


def aggregate_point(
    concurrency: int,
    warmups: list[dict[str, object]],
    measured: list[dict[str, object]],
) -> dict[str, object]:
    if len(warmups) != concurrency or len(measured) != concurrency * MEASURED_PER_SLOT:
        raise RuntimeError("rolling completion count mismatch")
    occupancy = validate_rolling_occupancy(measured, concurrency)
    started = min(float(row["start_offset_seconds"]) for row in measured)
    ended = max(float(row["end_offset_seconds"]) for row in measured)
    wall_seconds = ended - started
    output_tokens = sum(int(row["completion_tokens"]) for row in measured)
    mean_tpot = sum(float(row["tpot_seconds"]) for row in measured) / len(measured)
    return {
        "concurrency": concurrency,
        "warmup_completion_count": len(warmups),
        "measured_completion_count": len(measured),
        "wall_seconds": wall_seconds,
        "aggregate_output_tokens_per_second": output_tokens / wall_seconds,
        "requests_per_second": len(measured) / wall_seconds,
        "mean_output_tokens_per_second_per_user": 1 / mean_tpot,
        "mean_tpot_seconds": mean_tpot,
        "percentiles_seconds": {
            metric: {
                "p50": percentile([float(row[metric]) for row in measured], 0.50),
                "p95": percentile([float(row[metric]) for row in measured], 0.95),
                "p99": percentile([float(row[metric]) for row in measured], 0.99),
            }
            for metric in ("ttft_seconds", "tpot_seconds", "e2e_seconds")
        },
        "content_sha256": sorted({str(row["content_sha256"]) for row in measured}),
        "rolling_occupancy": occupancy,
        "requests": sorted(warmups + measured, key=lambda row: int(row["ordinal"])),
    }


def validate_initial_overlap(
    rows: list[dict[str, object]], expected_count: int
) -> dict[str, object]:
    if len(rows) != expected_count:
        raise RuntimeError("parallel wave completion count changed")
    latest_start = max(float(row["start_offset_seconds"]) for row in rows)
    earliest_end = min(float(row["end_offset_seconds"]) for row in rows)
    if latest_start >= earliest_end:
        raise RuntimeError("C4 priming requests did not overlap")
    return {
        "valid": True,
        "request_count": expected_count,
        "fully_overlapped_seconds": earliest_end - latest_start,
    }


def validate_rolling_occupancy(
    rows: list[dict[str, object]], concurrency: int
) -> dict[str, object]:
    gaps = []
    first_rows = []
    last_rows = []
    for slot in range(concurrency):
        slot_rows = sorted(
            (row for row in rows if row["slot"] == slot),
            key=lambda row: int(row["iteration"]),
        )
        if [row["iteration"] for row in slot_rows] != list(range(MEASURED_PER_SLOT)):
            raise RuntimeError(f"slot {slot} did not complete all rolling requests")
        first_rows.append(slot_rows[0])
        last_rows.append(slot_rows[-1])
        for previous, current in pairwise(slot_rows):
            gap = float(current["start_offset_seconds"]) - float(
                previous["end_offset_seconds"]
            )
            if gap < 0:
                raise RuntimeError(f"slot {slot} overlapped its own requests")
            gaps.append(gap)
    initial = validate_initial_overlap(first_rows, concurrency)
    steady_start = max(float(row["start_offset_seconds"]) for row in first_rows)
    drain_start = min(float(row["end_offset_seconds"]) for row in last_rows)
    if steady_start >= drain_start:
        raise RuntimeError("a persistent slot retired before full occupancy began")
    return {
        "valid": True,
        "definition": (
            "all persistent slot lifetimes remain occupied until the first final EOF; "
            "EOF-to-successor dispatch gaps are recorded and the final drain is excluded"
        ),
        "persistent_slots": concurrency,
        "logical_min_occupancy_before_drain": concurrency,
        "successor_dispatch_count": concurrency * (MEASURED_PER_SLOT - 1),
        "measurement_barrier_count": 1,
        "per_iteration_barrier_count": 0,
        "early_retired_slots": [],
        "initial_overlap": initial,
        "steady_window_seconds": drain_start - steady_start,
        "max_replacement_dispatch_gap_seconds": max(gaps, default=0.0),
    }


# Shared utilities


def contract() -> dict[str, object]:
    return {
        "prompt_tokens": PROMPT_TOKENS,
        "shared_prefix_tokens": SHARED_PREFIX_TOKENS,
        "unique_suffix_tokens": SUFFIX_TOKENS,
        "output_tokens": OUTPUT_TOKENS,
        "group_count": GROUP_COUNT,
        "priming": {
            "waves": 2,
            "concurrency": 4,
            "distinct_prefix_groups_per_wave": 4,
            "barrier_between_waves": True,
            "qualification_before_measurement": True,
        },
        "concurrencies": list(CONCURRENCIES),
        "warmups_per_slot": 1,
        "measured_completions_per_slot": MEASURED_PER_SLOT,
        "slot_group_mapping": "slot % 4",
        "scheduler": "rolling closed-loop persistent workers",
        "request": {
            "temperature": 0,
            "stream": True,
            "include_usage": True,
            "ignore_eos": True,
        },
        "full_hit_proxy": (
            "8192 shared cached tokens plus one terminal uncached token required to "
            "produce first-token logits"
        ),
    }


def parse_endpoint(value: str) -> SplitResult:
    endpoint = urlsplit(value)
    if (
        endpoint.scheme != "http"
        or not endpoint.hostname
        or endpoint.port is None
        or endpoint.path not in {"", "/"}
        or endpoint.query
        or endpoint.fragment
    ):
        raise ValueError("endpoint must be an explicit HTTP origin")
    return endpoint


def longest_common_prefix(token_sets) -> int:
    token_sets = list(token_sets)
    for index, tokens in enumerate(zip(*token_sets, strict=False)):
        if len(set(tokens)) != 1:
            return index
    return min(map(len, token_sets), default=0)


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _memo(cache: dict, key: tuple, thunk):
    if key not in cache:
        cache[key] = thunk()
    return cache[key]


def digest_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).hexdigest()

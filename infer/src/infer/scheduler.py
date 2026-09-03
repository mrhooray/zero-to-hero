from collections import deque
from dataclasses import dataclass
from itertools import pairwise

from infer.protocol import (
    MAX_TOKEN_ID,
    BatchPlan,
    BatchPlanRow,
    Request,
    StepResult,
    StepResultRow,
)
from infer.state import AppendIntent, PlanLease, PrefixLease, StateManager


@dataclass(frozen=True, slots=True)
class _Candidate:
    request: Request
    lane: int
    token_start: int
    token_ids: tuple[int, ...]
    max_accept_tokens: int
    reserved_tokens: int
    decode: bool
    capture_tokens: int


@dataclass(frozen=True, slots=True)
class _Inflight:
    plan: BatchPlan
    state_plan: PlanLease
    candidates: tuple[_Candidate, ...]

    @property
    def decode_only(self) -> bool:
        return all(candidate.decode for candidate in self.candidates)


class Scheduler:
    def __init__(
        self,
        state_manager: StateManager,
        token_budget: int,
        prefill_chunk_size: int,
        decode_width: int,
        max_batch_size: int,
        max_prefill_rows: int,
        max_decode_ticks_between_prefills: int,
        graph_buckets: tuple[int, ...],
        max_queued_requests: int,
        single_resident_prefill_chunk_size: int | None = None,
    ) -> None:
        if token_budget <= 0:
            raise ValueError("token_budget must be positive")
        if prefill_chunk_size <= 0:
            raise ValueError("prefill_chunk_size must be positive")
        if single_resident_prefill_chunk_size is not None and not (
            0 < single_resident_prefill_chunk_size <= prefill_chunk_size
        ):
            raise ValueError(
                "single_resident_prefill_chunk_size must be in [1, prefill_chunk_size]"
            )
        if decode_width <= 0:
            raise ValueError("decode_width must be positive")
        if decode_width > token_budget:
            raise ValueError("decode_width cannot exceed token_budget")
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        if not 0 < max_prefill_rows <= max_batch_size:
            raise ValueError("max_prefill_rows must be in [1, max_batch_size]")
        if max_decode_ticks_between_prefills < 0:
            raise ValueError("max_decode_ticks_between_prefills must be non-negative")
        if (
            not graph_buckets
            or any(type(bucket) is not int or bucket <= 0 for bucket in graph_buckets)
            or any(left >= right for left, right in pairwise(graph_buckets))
        ):
            raise ValueError("graph_buckets must be positive and strictly increasing")
        if max_batch_size > graph_buckets[-1]:
            raise ValueError("largest graph bucket must cover max_batch_size")
        if max_queued_requests <= 0:
            raise ValueError("max_queued_requests must be positive")

        self._state = state_manager
        self._token_budget = token_budget
        self._prefill_chunk_size = prefill_chunk_size
        self._single_resident_prefill_chunk_size = single_resident_prefill_chunk_size
        self._decode_width = decode_width
        self._max_batch_size = max_batch_size
        self._max_prefill_rows = max_prefill_rows
        self._max_decode_ticks_between_prefills = max_decode_ticks_between_prefills
        self._graph_buckets = graph_buckets
        self._max_queued_requests = max_queued_requests
        self._requests: dict[int, Request] = {}
        self._footprints: dict[int, int] = {}
        self._waiting: deque[int] = deque()
        self._order: deque[int] = deque()
        self._admitted: set[int] = set()
        self._terminal: set[int] = set()
        self._prefixes: dict[int, PrefixLease] = {}
        self._cache_boundaries: dict[int, deque[int]] = {}
        lane_count = len(state_manager.slots)
        self._admitted_requests = [0] * lane_count
        self._admitted_tokens = [0] * lane_count
        self._inflight: deque[_Inflight] = deque()
        self._next_step_id = 0
        self._decode_ticks_with_waiting_prefill = 0

    def submit(self, request: Request) -> None:
        if type(request) is not Request:
            raise TypeError("submit requires a Request")
        if request.request_id in self._requests:
            raise ValueError(f"request {request.request_id} already exists")
        if not request.prompt_token_ids:
            raise ValueError("prompt_token_ids must not be empty")
        if any(
            type(token_id) is not int or not 0 <= token_id <= MAX_TOKEN_ID
            for token_id in request.prompt_token_ids
        ):
            raise ValueError("prompt_token_ids must contain unsigned 32-bit integers")
        if type(request.max_output_tokens) is not int or request.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be a positive integer")
        if (
            request.slot is not None
            or request.committed_tokens
            or request.cached_tokens
            or request.output_token_ids
            or request.cancelled
        ):
            raise ValueError("request already has runtime state")
        if request.lane is not None and (
            type(request.lane) is not int
            or not 0 <= request.lane < len(self._state.slots)
        ):
            raise ValueError("request lane is out of range")

        logical_tokens = len(request.prompt_token_ids) + request.max_output_tokens - 1
        logical_blocks = self._state.blocks_for_tokens(logical_tokens)
        speculation_pad = self._decode_width - 1 if request.max_output_tokens > 1 else 0
        lanes = (
            (request.lane,)
            if request.lane is not None
            else tuple(range(len(self._state.slots)))
        )
        capacities = tuple(
            self._state.histories[lane].capacity
            for lane in lanes
            if self._state.slots[lane].live_capacity
            and logical_blocks <= self._state.histories[lane].capacity
        )
        if not capacities:
            raise ValueError(
                f"request {request.request_id} requires {logical_blocks} history blocks"
            )
        footprint = min(
            self._state.blocks_for_tokens(logical_tokens + speculation_pad),
            max(capacities),
        )

        self._requests[request.request_id] = request
        self._footprints[request.request_id] = footprint
        self._waiting.append(request.request_id)
        self._admit_waiting()
        if (
            request.request_id not in self._admitted
            and len(self._waiting) > self._max_queued_requests
        ):
            self._waiting.remove(request.request_id)
            del self._footprints[request.request_id]
            del self._requests[request.request_id]
            raise RuntimeError("request queue is full")

    def cancel(self, request_id: int) -> None:
        request = self._requests.get(request_id)
        if request is None or request.cancelled:
            return
        request.cancelled = True
        self._state.cancel(request_id)
        if not self._has_inflight(request_id):
            self._retire(request)

    def schedule(self) -> BatchPlan | None:
        if len(self._inflight) == 2:
            raise RuntimeError("two plans are already outstanding")

        phases = [self._phase(self._requests[request_id]) for request_id in self._order]
        has_prefill = any(phase == "prefill" for phase in phases)
        has_decode = any(phase == "decode" for phase in phases)
        decode_rows = [0] * len(self._state.slots)
        for request_id, phase in zip(self._order, phases, strict=True):
            if phase == "decode":
                lane = self._requests[request_id].lane
                assert lane is not None
                decode_rows[lane] += 1
        prefill_cadence = min(
            self._max_decode_ticks_between_prefills,
            3 * max(1, max(decode_rows)),
        )
        prefill_due = (
            has_prefill
            and has_decode
            and self._decode_ticks_with_waiting_prefill >= prefill_cadence
        )
        if self._inflight and prefill_due:
            return None

        candidates = self._select_candidates(
            phases,
            allow_prefill=not self._inflight and (not has_decode or prefill_due),
        )
        if not candidates:
            return None
        candidates.sort(key=lambda candidate: candidate.lane)
        state_plan = self._state.reserve(
            self._next_step_id,
            tuple(self._intent(candidate) for candidate in candidates),
        )
        if state_plan is None:
            raise RuntimeError("admitted state capacity is inconsistent")

        lanes: list[list[BatchPlanRow]] = [[] for _ in self._state.slots]
        for candidate, lease in zip(candidates, state_plan.rows, strict=True):
            request = candidate.request
            if request.slot is None:
                self._prefixes.pop(request.request_id, None)
            if candidate.capture_tokens:
                boundaries = self._cache_boundaries[request.request_id]
                if boundaries.popleft() != candidate.capture_tokens:
                    raise RuntimeError("prefix capture boundary changed")
                if not boundaries:
                    del self._cache_boundaries[request.request_id]
            request.slot = lease.sequence.slot
            request.lane = lease.sequence.lane
            lanes[candidate.lane].append(
                BatchPlanRow(
                    slot=lease.sequence.slot,
                    token_start=candidate.token_start,
                    token_ids=candidate.token_ids,
                    max_accept_tokens=candidate.max_accept_tokens,
                    sample=candidate.decode
                    or candidate.token_start + len(candidate.token_ids)
                    == len(request.prompt_token_ids),
                    table_delta=lease.table_delta,
                    ignore_eos=request.ignore_eos,
                    restore_slot=lease.restore_slot,
                    capture_slot=lease.capture_slot,
                )
            )
        decode_only = all(candidate.decode for candidate in candidates)
        graph_bucket = None
        if decode_only:
            row_count = max(map(len, lanes))
            graph_bucket = next(
                bucket for bucket in self._graph_buckets if bucket >= row_count
            )
        plan = BatchPlan(
            self._next_step_id,
            tuple(tuple(lane) for lane in lanes),
            graph_bucket,
        )
        self._inflight.append(_Inflight(plan, state_plan, tuple(candidates)))
        self._next_step_id += 1
        self._record_cadence(decode_only, has_prefill)
        self._rotate(candidate.request.request_id for candidate in candidates)
        return plan

    def commit(self, result: StepResult) -> StepResult:
        if not self._inflight:
            raise ValueError("no plan is inflight")
        inflight = self._inflight[0]
        observations = self._validate_result(result, inflight)
        leases = self._state.commit(
            inflight.state_plan,
            tuple(observation.accepted_count for observation in observations),
        )

        committed: list[list[StepResultRow]] = [[] for _ in self._state.slots]
        touched: dict[int, Request] = {}
        for candidate, state_row, observation, lease in zip(
            inflight.candidates,
            inflight.state_plan.rows,
            observations,
            leases,
            strict=True,
        ):
            request = candidate.request
            touched[request.request_id] = request
            if state_row.restore_slot is not None:
                request.cached_tokens = state_row.sequence.committed_tokens
                self._admitted_tokens[candidate.lane] -= request.cached_tokens
            if lease is not None:
                request.committed_tokens = lease.committed_tokens
            terminal = request.request_id in self._terminal
            output_token_ids = (
                () if request.cancelled or terminal else observation.output_token_ids
            )
            request.output_token_ids.extend(output_token_ids)
            finished = (
                request.cancelled
                or terminal
                or observation.finished
                or len(request.output_token_ids) >= request.max_output_tokens
            )
            committed[candidate.lane].append(
                StepResultRow(
                    request.slot,
                    observation.accepted_count,
                    output_token_ids,
                    finished,
                )
            )
            if finished and not request.cancelled and not terminal:
                self._terminal.add(request.request_id)
                self._state.cancel(request.request_id)

        self._inflight.popleft()
        for request in touched.values():
            if (
                request.cancelled or request.request_id in self._terminal
            ) and not self._has_inflight(request.request_id):
                self._retire(request)
        return StepResult(
            result.step_id,
            tuple(tuple(lane) for lane in committed),
            result.elapsed_ns,
        )

    def _select_candidates(
        self, phases: list[str | None], *, allow_prefill: bool
    ) -> list[_Candidate]:
        single_resident_chunk = self._single_resident_prefill_chunk_size
        prefill_chunk_size = (
            single_resident_chunk
            if single_resident_chunk is not None and max(self._admitted_requests) <= 1
            else self._prefill_chunk_size
        )
        prefill_budget = min(self._token_budget, prefill_chunk_size)
        lane_count = len(self._state.slots)
        prefill_tokens = [0] * lane_count
        prefill_rows = [0] * lane_count
        prefill_candidates = []
        if allow_prefill:
            for request_id, phase in zip(self._order, phases, strict=True):
                if phase != "prefill":
                    continue
                request = self._requests[request_id]
                assert request.lane is not None
                lane = request.lane
                if (
                    prefill_tokens[lane] >= prefill_budget
                    or prefill_rows[lane] >= self._max_prefill_rows
                ):
                    continue
                start, _ = self._projected_progress(request)
                size = self._prefill_size(
                    request,
                    start,
                    min(
                        prefill_budget - prefill_tokens[lane],
                        len(request.prompt_token_ids) - start,
                    ),
                )
                end = start + size
                boundaries = self._cache_boundaries.get(request.request_id)
                prefill_candidates.append(
                    _Candidate(
                        request,
                        lane,
                        start,
                        request.prompt_token_ids[start:end],
                        size,
                        size,
                        False,
                        end if boundaries and end == boundaries[0] else 0,
                    )
                )
                prefill_tokens[lane] += size
                prefill_rows[lane] += 1
        remaining = [self._token_budget - tokens for tokens in prefill_tokens]
        decode_rows = [0] * lane_count
        selected = []
        for request_id, phase in zip(self._order, phases, strict=True):
            if phase != "decode":
                continue
            request = self._requests[request_id]
            assert request.lane is not None
            lane = request.lane
            if remaining[lane] < self._decode_width:
                continue
            if decode_rows[lane] + prefill_rows[lane] >= self._max_batch_size:
                continue
            start, output_count = self._projected_progress(request)
            max_accept_tokens = min(
                self._decode_width, request.max_output_tokens - output_count
            )
            reserved_tokens = min(
                self._decode_width,
                self._state.histories[lane].capacity
                * self._state.history_block_tokens
                - start,
            )
            selected.append(
                _Candidate(
                    request,
                    lane,
                    start,
                    (),
                    max_accept_tokens,
                    reserved_tokens,
                    True,
                    0,
                )
            )
            decode_rows[lane] += 1
            remaining[lane] -= self._decode_width

        selected.extend(prefill_candidates)
        return selected

    def _phase(self, request: Request) -> str | None:
        if (
            request.cancelled
            or request.request_id in self._terminal
            or request.request_id not in self._admitted
        ):
            return None
        token_start, output_count = self._projected_progress(request)
        if token_start < len(request.prompt_token_ids):
            return "prefill"
        if output_count < request.max_output_tokens:
            return "decode"
        return None

    def _projected_progress(self, request: Request) -> tuple[int, int]:
        prefix = self._prefixes.get(request.request_id)
        token_start = (
            len(prefix.token_ids)
            if request.slot is None and prefix is not None
            else request.committed_tokens
        )
        output_count = len(request.output_token_ids)
        for inflight in self._inflight:
            for candidate, lease in zip(
                inflight.candidates, inflight.state_plan.rows, strict=True
            ):
                if candidate.request is request:
                    token_start = max(token_start, lease.sequence.committed_tokens)
                    token_start += candidate.reserved_tokens
                    if candidate.decode:
                        output_count += candidate.max_accept_tokens
                    elif candidate.token_start + len(candidate.token_ids) == len(
                        request.prompt_token_ids
                    ):
                        output_count += 1
        return token_start, output_count

    def _intent(self, candidate: _Candidate) -> AppendIntent:
        request = candidate.request
        return AppendIntent(
            request.request_id,
            candidate.reserved_tokens,
            lane=candidate.lane if request.slot is None else None,
            prefix=(
                self._prefixes.get(request.request_id) if request.slot is None else None
            ),
            cache_token_ids=(
                request.prompt_token_ids[: candidate.capture_tokens]
                if candidate.capture_tokens
                else ()
            ),
        )

    def _prefill_size(self, request: Request, start: int, limit: int) -> int:
        boundaries = self._cache_boundaries.get(request.request_id)
        if not boundaries:
            return limit
        boundary = boundaries[0]
        if boundary <= start:
            raise RuntimeError("prefix capture boundary was skipped")
        return min(limit, boundary - start)

    def _validate_result(
        self, result: StepResult, inflight: _Inflight
    ) -> tuple[StepResultRow, ...]:
        if result.step_id != inflight.plan.step_id:
            raise ValueError(
                f"result step {result.step_id} does not match "
                f"inflight step {inflight.plan.step_id}"
            )
        if type(result.elapsed_ns) is not int or result.elapsed_ns < 0:
            raise ValueError("elapsed_ns must be a non-negative integer")
        if len(result.lanes) != len(inflight.plan.lanes):
            raise ValueError("result lanes must match plan lanes")

        observations = []
        candidates = iter(inflight.candidates)
        for lane, (plan_rows, result_rows) in enumerate(
            zip(inflight.plan.lanes, result.lanes, strict=True)
        ):
            if len(plan_rows) != len(result_rows):
                raise ValueError("result rows must match plan rows")
            for plan_row, observation in zip(plan_rows, result_rows, strict=True):
                candidate = next(candidates)
                assert candidate.lane == lane
                request_id = candidate.request.request_id
                if observation.slot != plan_row.slot:
                    raise ValueError("result rows must match plan slots")
                if (
                    type(observation.accepted_count) is not int
                    or not 0 <= observation.accepted_count <= plan_row.max_accept_tokens
                ):
                    raise ValueError(
                        f"request {request_id} accepted {observation.accepted_count} "
                        f"tokens, planned {plan_row.max_accept_tokens}"
                    )
                if not candidate.decode and observation.accepted_count != len(
                    plan_row.token_ids
                ):
                    raise ValueError(
                        f"request {request_id} prefill accepted "
                        f"{observation.accepted_count} tokens, planned "
                        f"{len(plan_row.token_ids)}"
                    )
                expected_outputs = (
                    observation.accepted_count
                    if candidate.decode
                    else int(
                        plan_row.sample
                        and observation.accepted_count == len(plan_row.token_ids)
                    )
                )
                if len(observation.output_token_ids) != expected_outputs:
                    raise ValueError(
                        f"request {request_id} must return "
                        f"{expected_outputs} output tokens"
                    )
                if (
                    observation.finished
                    and expected_outputs == 0
                    and not candidate.request.cancelled
                ):
                    raise ValueError(
                        f"request {request_id} finished without an output token"
                    )
                observations.append(observation)
        return tuple(observations)

    def _record_cadence(self, decode_only: bool, waiting_prefill: bool) -> None:
        if decode_only and waiting_prefill:
            self._decode_ticks_with_waiting_prefill += 1
        else:
            self._decode_ticks_with_waiting_prefill = 0

    def _admit_waiting(self) -> None:
        still_waiting = deque()
        blocked_lanes: set[int] = set()
        while self._waiting:
            request_id = self._waiting.popleft()
            request = self._requests[request_id]
            footprint = self._footprints[request_id]
            lanes = (
                (request.lane,)
                if request.lane is not None
                else tuple(range(len(self._state.slots)))
            )
            candidate_lanes = [
                lane
                for lane in lanes
                if lane not in blocked_lanes
                and self._admitted_requests[lane]
                < self._state.slots[lane].live_capacity
            ]
            if not candidate_lanes:
                still_waiting.append(request_id)
                if request.lane is not None:
                    blocked_lanes.add(request.lane)
                continue
            probed = self._state.acquire_prefix_candidates(
                request.prompt_token_ids, tuple(candidate_lanes)
            )
            available = []
            for candidate in probed:
                reserved = self._reserved_history_blocks(candidate.lane)
                shared = (
                    len(candidate.lease.shared_history_blocks)
                    if candidate.lease is not None
                    else 0
                )
                if (
                    reserved + footprint - shared
                    <= self._state.reservable_history_blocks(candidate.lane)
                ):
                    available.append((candidate, reserved))
            if not available:
                for candidate in probed:
                    if candidate.lease is not None:
                        self._state.release_prefix(candidate.lease)
                still_waiting.append(request_id)
                if request.lane is not None:
                    blocked_lanes.add(request.lane)
                continue
            selected, _ = min(
                available,
                key=lambda item: (
                    -(len(item[0].lease.token_ids) if item[0].lease is not None else 0),
                    -item[0].promotion_tokens,
                    self._admitted_tokens[item[0].lane],
                    item[1],
                    self._admitted_requests[item[0].lane],
                    item[0].lane,
                ),
            )
            for candidate in probed:
                if candidate.lane != selected.lane and candidate.lease is not None:
                    self._state.release_prefix(candidate.lease)
            lane = selected.lane
            prefix = selected.lease
            request.lane = lane
            if prefix is not None:
                self._prefixes[request_id] = prefix
            if selected.cache_boundaries:
                self._cache_boundaries[request_id] = deque(selected.cache_boundaries)
            self._admitted.add(request_id)
            self._admitted_requests[lane] += 1
            self._admitted_tokens[lane] += (
                len(request.prompt_token_ids) + request.max_output_tokens
            )
            self._order.append(request_id)
        self._waiting = still_waiting

    def _retire(self, request: Request) -> None:
        request_id = request.request_id
        prefix = self._prefixes.pop(request_id, None)
        if prefix is not None:
            self._state.release_prefix(prefix)
        self._cache_boundaries.pop(request_id, None)
        try:
            self._waiting.remove(request_id)
        except ValueError:
            pass
        if request_id in self._admitted:
            assert request.lane is not None
            self._admitted.remove(request_id)
            self._admitted_requests[request.lane] -= 1
            self._admitted_tokens[request.lane] -= (
                len(request.prompt_token_ids)
                + request.max_output_tokens
                - request.cached_tokens
            )
            self._order.remove(request_id)
        del self._footprints[request_id]
        del self._requests[request_id]
        self._terminal.discard(request_id)
        self._admit_waiting()

    def _reserved_history_blocks(self, lane: int) -> int:
        reserved = 0
        for request_id in self._admitted:
            request = self._requests[request_id]
            if request.lane != lane:
                continue
            if request.slot is None:
                prefix = self._prefixes.get(request_id)
                allocated = (
                    len(prefix.shared_history_blocks) if prefix is not None else 0
                )
            else:
                allocated = self._state.blocks_for_tokens(
                    self._projected_progress(request)[0]
                )
            reserved += max(0, self._footprints[request_id] - allocated)
        return reserved

    def _has_inflight(self, request_id: int) -> bool:
        return any(
            candidate.request.request_id == request_id
            for inflight in self._inflight
            for candidate in inflight.candidates
        )

    def _rotate(self, request_ids) -> None:
        for request_id in request_ids:
            self._order.remove(request_id)
            self._order.append(request_id)

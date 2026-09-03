from collections import OrderedDict
from dataclasses import dataclass, field
from hashlib import blake2b
from struct import Struct

from infer.protocol import TableDelta

_INDEX_BITS = 32
_INDEX_MASK = (1 << _INDEX_BITS) - 1
_LANE_BITS = 8
_LANE_MASK = (1 << _LANE_BITS) - 1
_GENERATION_SHIFT = _INDEX_BITS + _LANE_BITS


@dataclass(frozen=True, slots=True)
class SequenceLease:
    request_id: int
    slot: int
    lane: int
    committed_tokens: int
    history_blocks: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class PrefixLease:
    lane: int
    token_ids: tuple[int, ...]
    shared_history_blocks: tuple[int, ...]
    snapshot_slot: int
    digest: bytes = field(repr=False)
    ticket: int = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class PrefixCandidate:
    lane: int
    lease: PrefixLease | None
    promotion_tokens: int
    cache_boundaries: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class AppendIntent:
    request_id: int
    max_tokens: int
    lane: int | None = None
    prefix: PrefixLease | None = None
    cache_token_ids: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class AppendLease:
    sequence: SequenceLease
    max_tokens: int
    table_delta: TableDelta
    restore_slot: int | None = None
    capture_slot: int | None = None


@dataclass(frozen=True, slots=True)
class PlanLease:
    step_id: int
    rows: tuple[AppendLease, ...]


class HistoryPool:
    def __init__(self, block_count: int, lane: int = 0) -> None:
        if not 0 <= block_count <= _INDEX_MASK:
            raise ValueError("history block count is out of range")
        if not 0 <= lane <= _LANE_MASK:
            raise ValueError("history lane is out of range")
        self.capacity = block_count
        self.lane = lane
        self._free = list(range(block_count - 1, -1, -1))
        self._generations = [0] * block_count
        self._references = [0] * block_count

    @property
    def available(self) -> int:
        return len(self._free)

    def allocate(self, count: int) -> tuple[int, ...] | None:
        if type(count) is not int or count < 0:
            raise ValueError("history allocation count must be a non-negative integer")
        if count > self.available:
            return None
        handles = []
        for _ in range(count):
            index = self._free.pop()
            self._generations[index] += 1
            self._references[index] = 1
            handles.append(
                self._generations[index] << _GENERATION_SHIFT
                | self.lane << _INDEX_BITS
                | index
            )
        return tuple(handles)

    def retain(self, handles: tuple[int, ...]) -> None:
        for index in self._indices(handles):
            self._references[index] += 1

    def release(self, handles: tuple[int, ...]) -> None:
        indices = self._indices(handles)
        for index in indices:
            self._references[index] -= 1
            if self._references[index] == 0:
                self._free.append(index)

    def physical(self, handles: tuple[int, ...]) -> tuple[int, ...]:
        return self._indices(handles)

    def reclaimable(self, groups: tuple[tuple[int, ...], ...]) -> int:
        releases: dict[int, int] = {}
        for handles in groups:
            for index in self._indices(handles):
                releases[index] = releases.get(index, 0) + 1
        return sum(releases[index] == self._references[index] for index in releases)

    def reclaim_prefix_count(
        self, groups: tuple[tuple[int, ...], ...], needed: int
    ) -> int | None:
        if type(needed) is not int or needed < 0:
            raise ValueError("needed block count must be a non-negative integer")
        if needed == 0:
            return 0
        releases: dict[int, int] = {}
        reclaimed = 0
        for group_count, handles in enumerate(groups, 1):
            for index in self._indices(handles):
                planned = releases.get(index, 0) + 1
                if planned > self._references[index]:
                    raise ValueError("planned releases exceed history references")
                releases[index] = planned
                if planned == self._references[index]:
                    reclaimed += 1
            if reclaimed >= needed:
                return group_count
        return None

    def _indices(self, handles: tuple[int, ...]) -> tuple[int, ...]:
        if len(set(handles)) != len(handles):
            raise ValueError("a history block appears more than once")
        indices = []
        for handle in handles:
            index = handle & _INDEX_MASK
            lane = handle >> _INDEX_BITS & _LANE_MASK
            generation = handle >> _GENERATION_SHIFT
            if (
                index >= self.capacity
                or lane != self.lane
                or generation != self._generations[index]
                or self._references[index] == 0
            ):
                raise ValueError(f"stale history block {handle}")
            indices.append(index)
        return tuple(indices)


class StateSlotPool:
    def __init__(self, live_count: int, snapshot_count: int) -> None:
        if live_count < 0 or snapshot_count < 0:
            raise ValueError("state slot counts must be non-negative")
        self.live_capacity = live_count
        self.snapshot_capacity = snapshot_count
        self._free_live = list(range(live_count - 1, -1, -1))
        self._free_snapshots = list(
            range(live_count + snapshot_count - 1, live_count - 1, -1)
        )
        self._allocated: set[int] = set()

    @property
    def available_live(self) -> int:
        return len(self._free_live)

    @property
    def available_snapshots(self) -> int:
        return len(self._free_snapshots)

    def acquire_live(self) -> int | None:
        return self._acquire(self._free_live)

    def acquire_snapshot(self) -> int | None:
        return self._acquire(self._free_snapshots)

    def release(self, slot: int) -> None:
        if slot not in self._allocated:
            raise ValueError(f"stale state slot {slot}")
        self._allocated.remove(slot)
        free = self._free_live if slot < self.live_capacity else self._free_snapshots
        free.append(slot)

    def _acquire(self, free: list[int]) -> int | None:
        if not free:
            return None
        slot = free.pop()
        self._allocated.add(slot)
        return slot


@dataclass(slots=True)
class _Pending:
    lease: AppendLease
    table: tuple[int, ...]
    prefix: PrefixLease | None
    cache_token_ids: tuple[int, ...]


@dataclass(slots=True)
class _Sequence:
    lease: SequenceLease
    pending: list[_Pending] = field(default_factory=list)
    cancelled: bool = False


@dataclass(slots=True)
class _PrefixEndpoint:
    lease: PrefixLease
    ancestor_keys: tuple[tuple[int, bytes], ...]
    acquired: set[int] = field(default_factory=set)
    inflight: set[int] = field(default_factory=set)

    @property
    def pins(self) -> int:
        return len(self.acquired) + len(self.inflight)


class StateManager:
    def __init__(
        self,
        history_block_count: int,
        live_slot_count: int,
        snapshot_slot_count: int,
        lane_count: int = 1,
        *,
        history_block_tokens: int,
    ) -> None:
        if not 1 <= lane_count <= _LANE_MASK + 1:
            raise ValueError("lane count is out of range")
        if type(history_block_tokens) is not int or history_block_tokens <= 0:
            raise ValueError("history_block_tokens must be a positive integer")
        self.history_block_tokens = history_block_tokens
        self._token_block = Struct(f"<{history_block_tokens}I")
        self.histories = tuple(
            HistoryPool(history_block_count, lane) for lane in range(lane_count)
        )
        self.slots = tuple(
            StateSlotPool(live_slot_count, snapshot_slot_count)
            for _ in range(lane_count)
        )
        self._sequences: dict[int, _Sequence] = {}
        self._plans: list[tuple[PlanLease, tuple[_Pending, ...]]] = []
        self._prefixes: OrderedDict[tuple[int, bytes], _PrefixEndpoint] = OrderedDict()
        self._prefix_ancestors: dict[tuple[int, bytes], set[tuple[int, bytes]]] = {}
        self._next_prefix_ticket = 1

    def acquire_prefix_candidates(
        self, token_ids: tuple[int, ...], lanes: tuple[int, ...]
    ) -> tuple[PrefixCandidate, ...]:
        if len(set(lanes)) != len(lanes):
            raise ValueError("prefix candidate lanes must be unique")
        for lane in lanes:
            self._validate_lane(lane)
        digests = tuple(self._prefix_digests(token_ids))
        final = digests[-1][0] if digests else 0
        candidates = []
        for lane in lanes:
            lease = self._acquire_prefix(self._find_prefix(token_ids, lane, digests))
            promotion = self._promotion_length(token_ids, lane, digests)
            after_tokens = len(lease.token_ids) if lease is not None else 0
            boundaries = ()
            if self.slots[lane].snapshot_capacity and final > after_tokens:
                boundaries = tuple(
                    boundary
                    for boundary in sorted({promotion, final})
                    if after_tokens < boundary <= final
                )
            candidates.append(PrefixCandidate(lane, lease, promotion, boundaries))
        return tuple(candidates)

    def _acquire_prefix(self, endpoint: _PrefixEndpoint | None) -> PrefixLease | None:
        if endpoint is None:
            return None
        ticket = self._next_prefix_ticket
        self._next_prefix_ticket += 1
        endpoint.acquired.add(ticket)
        lease = endpoint.lease
        return PrefixLease(
            lease.lane,
            lease.token_ids,
            lease.shared_history_blocks,
            lease.snapshot_slot,
            lease.digest,
            ticket,
        )

    def release_prefix(self, lease: PrefixLease) -> None:
        endpoint = self._prefixes.get((lease.lane, lease.digest))
        if (
            endpoint is None
            or endpoint.lease != lease
            or lease.ticket not in endpoint.acquired
        ):
            raise ValueError("stale prefix lease")
        endpoint.acquired.remove(lease.ticket)

    def reservable_history_blocks(self, lane: int) -> int:
        self._validate_lane(lane)
        groups = tuple(
            endpoint.lease.shared_history_blocks
            for endpoint in self._prefixes.values()
            if endpoint.lease.lane == lane
            and endpoint.pins == 0
            and endpoint.lease.shared_history_blocks
        )
        history = self.histories[lane]
        return history.available + history.reclaimable(groups)

    def reserve(
        self, step_id: int, intents: tuple[AppendIntent, ...]
    ) -> PlanLease | None:
        if len(self._plans) == 2:
            raise RuntimeError("two plans are already outstanding")
        if any(plan.step_id == step_id for plan, _ in self._plans):
            raise ValueError(f"step {step_id} already has a reservation")
        prepared, required_slots, required_blocks = self._requirements(intents)
        if any(
            required_slots[lane] > pool.available_live
            for lane, pool in enumerate(self.slots)
        ):
            return None
        if not self._evict_for_blocks(required_blocks):
            return None
        for intent in intents:
            if intent.prefix is not None:
                self._transfer_prefix(intent.prefix)

        rows = []
        pending_rows = []
        for intent, sequence, base_table, new_block_count in prepared:
            lane = intent.lane if sequence is None else sequence.lease.lane
            assert lane is not None
            history = self.histories[lane]
            new_sequence = sequence is None
            if sequence is None:
                slot = self.slots[lane].acquire_live()
                assert slot is not None
                history.retain(base_table)
                lease = SequenceLease(
                    intent.request_id,
                    slot,
                    intent.lane,
                    len(intent.prefix.token_ids) if intent.prefix else 0,
                    base_table,
                )
                sequence = _Sequence(lease)
                self._sequences[intent.request_id] = sequence

            history.retain(base_table[len(sequence.lease.history_blocks) :])
            new_blocks = history.allocate(new_block_count)
            assert new_blocks is not None
            table = base_table + new_blocks
            capture_slot = self._capture_slot(intent.cache_token_ids, lane)
            delta_start = 0 if new_sequence else len(base_table)
            delta_blocks = table if new_sequence else new_blocks
            if capture_slot is not None and not delta_blocks:
                delta_start = (
                    len(intent.cache_token_ids) // self.history_block_tokens - 1
                )
                delta_blocks = table[delta_start : delta_start + 1]
            row = AppendLease(
                sequence.lease,
                intent.max_tokens,
                TableDelta(
                    delta_start,
                    history.physical(delta_blocks),
                ),
                intent.prefix.snapshot_slot if new_sequence and intent.prefix else None,
                capture_slot,
            )
            pending = _Pending(
                row,
                table,
                intent.prefix if new_sequence else None,
                intent.cache_token_ids if capture_slot is not None else (),
            )
            sequence.pending.append(pending)
            rows.append(row)
            pending_rows.append(pending)

        plan = PlanLease(step_id, tuple(rows))
        self._plans.append((plan, tuple(pending_rows)))
        return plan

    def commit(
        self, plan: PlanLease, accepted_counts: tuple[int, ...]
    ) -> tuple[SequenceLease | None, ...]:
        if not self._plans or self._plans[0][0] is not plan:
            raise ValueError(f"step {plan.step_id} has a stale or out-of-order lease")
        pending_rows = self._plans[0][1]
        if len(accepted_counts) != len(pending_rows):
            raise ValueError("accepted counts must match plan rows")
        for pending, accepted in zip(pending_rows, accepted_counts, strict=True):
            sequence = self._sequences[pending.lease.sequence.request_id]
            if sequence.pending[0] is not pending:
                raise ValueError("request reservation order changed")
            if (
                type(accepted) is not int
                or not 0 <= accepted <= pending.lease.max_tokens
            ):
                raise ValueError("accepted count exceeds its reservation")

        results = []
        for pending, accepted in zip(pending_rows, accepted_counts, strict=True):
            sequence = self._sequences[pending.lease.sequence.request_id]
            committed_tokens = sequence.lease.committed_tokens + accepted
            block_count = self.blocks_for_tokens(committed_tokens)
            committed_blocks = pending.table[:block_count]
            new_blocks = committed_blocks[len(sequence.lease.history_blocks) :]
            history = self.histories[sequence.lease.lane]
            history.retain(new_blocks)
            history.release(pending.table[len(pending.lease.sequence.history_blocks) :])
            sequence.pending.pop(0)
            sequence.lease = SequenceLease(
                sequence.lease.request_id,
                sequence.lease.slot,
                sequence.lease.lane,
                committed_tokens,
                committed_blocks,
            )
            if pending.prefix is not None:
                self._release_inflight_prefix(pending.prefix)
            capture_slot = pending.lease.capture_slot
            if capture_slot is not None:
                if (
                    accepted == pending.lease.max_tokens
                    and not sequence.cancelled
                    and committed_tokens == len(pending.cache_token_ids)
                ):
                    self._publish_prefix(
                        pending.cache_token_ids,
                        sequence.lease,
                        capture_slot,
                    )
                else:
                    self.slots[sequence.lease.lane].release(capture_slot)
            if sequence.cancelled and not sequence.pending:
                self._release_sequence(sequence)
                results.append(None)
            else:
                results.append(sequence.lease)

        self._plans.pop(0)
        return tuple(results)

    def cancel(self, request_id: int) -> None:
        sequence = self._sequences.get(request_id)
        if sequence is None:
            return
        sequence.cancelled = True
        if not sequence.pending:
            self._release_sequence(sequence)

    def release(self, lease: SequenceLease) -> None:
        sequence = self._sequences.get(lease.request_id)
        if sequence is None:
            return
        if sequence.lease is not lease:
            raise ValueError(f"request {lease.request_id} has a stale lease")
        if sequence.pending:
            raise RuntimeError(f"request {lease.request_id} has outstanding plans")
        self._release_sequence(sequence)

    def _requirements(
        self, intents: tuple[AppendIntent, ...]
    ) -> tuple[
        list[tuple[AppendIntent, _Sequence | None, tuple[int, ...], int]],
        list[int],
        list[int],
    ]:
        if not intents:
            raise ValueError("a plan must contain at least one append")
        if len({intent.request_id for intent in intents}) != len(intents):
            raise ValueError("a request appears twice in one plan")

        prepared = []
        required_slots = [0] * len(self.slots)
        required_blocks = [0] * len(self.histories)
        prefix_tickets: set[int] = set()
        for intent in intents:
            if type(intent.max_tokens) is not int or intent.max_tokens <= 0:
                raise ValueError("max_tokens must be a positive integer")
            sequence = self._sequences.get(intent.request_id)
            if sequence is None:
                if intent.lane is None or not 0 <= intent.lane < len(self.slots):
                    raise ValueError("a new request requires a lane")
                if intent.prefix is not None:
                    if intent.prefix.lane != intent.lane:
                        raise ValueError("a shared prefix belongs to another lane")
                    endpoint = self._prefixes.get(
                        (intent.prefix.lane, intent.prefix.digest)
                    )
                    if (
                        endpoint is None
                        or endpoint.lease != intent.prefix
                        or intent.prefix.ticket not in endpoint.acquired
                        or intent.prefix.ticket in prefix_tickets
                    ):
                        raise ValueError("stale prefix lease")
                    prefix_tickets.add(intent.prefix.ticket)
                    base_tokens = len(intent.prefix.token_ids)
                    base_table = intent.prefix.shared_history_blocks
                else:
                    base_tokens = 0
                    base_table = ()
                self.histories[intent.lane].physical(base_table)
                required_slots[intent.lane] += 1
                lane = intent.lane
            else:
                if intent.lane is not None or intent.prefix is not None:
                    raise ValueError(
                        "an admitted request cannot replace its base state"
                    )
                if sequence.cancelled:
                    raise RuntimeError("a cancelled request cannot reserve more work")
                if len(sequence.pending) == 2:
                    raise RuntimeError("a request already has two reservations")
                base_tokens = sequence.lease.committed_tokens + sum(
                    pending.lease.max_tokens for pending in sequence.pending
                )
                base_table = (
                    sequence.pending[-1].table
                    if sequence.pending
                    else sequence.lease.history_blocks
                )
                lane = sequence.lease.lane
            if intent.cache_token_ids and (
                len(intent.cache_token_ids) != base_tokens + intent.max_tokens
                or len(intent.cache_token_ids) % self.history_block_tokens
            ):
                raise ValueError(
                    "cached prefixes must end on the reserved block boundary"
                )
            if any(
                type(token_id) is not int or not 0 <= token_id <= _INDEX_MASK
                for token_id in intent.cache_token_ids
            ):
                raise ValueError("cached token IDs must be unsigned 32-bit integers")
            if (
                intent.prefix is not None
                and intent.cache_token_ids
                and intent.cache_token_ids[:base_tokens] != intent.prefix.token_ids
            ):
                raise ValueError("a cached prefix must extend the restored prefix")
            new_block_count = max(
                0,
                self.blocks_for_tokens(base_tokens + intent.max_tokens)
                - len(base_table),
            )
            prepared.append((intent, sequence, base_table, new_block_count))
            required_blocks[lane] += new_block_count
        return prepared, required_slots, required_blocks

    def _find_prefix(
        self,
        token_ids: tuple[int, ...],
        lane: int,
        digests: tuple[tuple[int, bytes], ...],
    ) -> _PrefixEndpoint | None:
        for boundary, digest in reversed(digests):
            endpoint = self._prefixes.get((lane, digest))
            if (
                endpoint is not None
                and endpoint.lease.token_ids == token_ids[:boundary]
            ):
                return endpoint
        return None

    def _promotion_length(
        self,
        token_ids: tuple[int, ...],
        lane: int,
        digests: tuple[tuple[int, bytes], ...],
    ) -> int:
        for boundary, digest in reversed(digests):
            for endpoint_key in self._prefix_ancestors.get((lane, digest), ()):
                endpoint = self._prefixes[endpoint_key]
                if endpoint.lease.token_ids[:boundary] == token_ids[:boundary]:
                    return boundary
        return 0

    def _capture_slot(self, token_ids: tuple[int, ...], lane: int) -> int | None:
        if not token_ids:
            return None
        key = (lane, self._prefix_digest(token_ids))
        existing = self._prefixes.get(key)
        if existing is not None and existing.lease.token_ids == token_ids:
            self._prefixes.move_to_end(key)
            return None
        if existing is not None:
            return None
        pool = self.slots[lane]
        slot = pool.acquire_snapshot()
        while slot is None:
            key = next(
                (
                    key
                    for key, endpoint in self._prefixes.items()
                    if endpoint.lease.lane == lane and endpoint.pins == 0
                ),
                None,
            )
            if key is None:
                return None
            self._evict(key)
            slot = pool.acquire_snapshot()
        return slot

    def _publish_prefix(
        self, token_ids: tuple[int, ...], sequence: SequenceLease, snapshot_slot: int
    ) -> None:
        block_count = len(token_ids) // self.history_block_tokens
        blocks = sequence.history_blocks[: block_count - 1]
        if len(blocks) + 1 != block_count:
            raise RuntimeError("cached prefix has no private snapshot tail")
        digests = tuple(self._complete_block_digests(token_ids))
        digest = digests[-1][1]
        key = (sequence.lane, digest)
        existing = self._prefixes.get(key)
        if existing is not None:
            self.slots[sequence.lane].release(snapshot_slot)
            if existing.lease.token_ids == token_ids:
                self._prefixes.move_to_end(key)
            return
        ancestor_keys = tuple(
            dict.fromkeys((sequence.lane, digest) for _, digest in digests)
        )
        self.histories[sequence.lane].retain(blocks)
        self._prefixes[key] = _PrefixEndpoint(
            PrefixLease(sequence.lane, token_ids, blocks, snapshot_slot, digest, 0),
            ancestor_keys,
        )
        for ancestor_key in ancestor_keys:
            self._prefix_ancestors.setdefault(ancestor_key, set()).add(key)

    def _evict_for_blocks(self, required: list[int]) -> bool:
        evictions = []
        for lane, count in enumerate(required):
            needed = count - self.histories[lane].available
            if needed <= 0:
                continue
            candidates = []
            for key, endpoint in self._prefixes.items():
                if (
                    endpoint.lease.lane != lane
                    or endpoint.pins
                    or not endpoint.lease.shared_history_blocks
                ):
                    continue
                candidates.append((key, endpoint.lease.shared_history_blocks))
            group_count = self.histories[lane].reclaim_prefix_count(
                tuple(blocks for _, blocks in candidates), needed
            )
            if group_count is None:
                return False
            evictions.extend(key for key, _ in candidates[:group_count])
        for key in evictions:
            self._evict(key)
        return True

    def _evict(self, key: tuple[int, bytes]) -> None:
        endpoint = self._prefixes[key]
        if endpoint.pins:
            raise RuntimeError("a pinned prefix cannot be evicted")
        del self._prefixes[key]
        lease = endpoint.lease
        for ancestor_key in endpoint.ancestor_keys:
            endpoints = self._prefix_ancestors[ancestor_key]
            endpoints.remove(key)
            if not endpoints:
                del self._prefix_ancestors[ancestor_key]
        self.histories[lease.lane].release(lease.shared_history_blocks)
        self.slots[lease.lane].release(lease.snapshot_slot)

    def _transfer_prefix(self, lease: PrefixLease) -> None:
        key = (lease.lane, lease.digest)
        endpoint = self._prefixes[key]
        endpoint.acquired.remove(lease.ticket)
        endpoint.inflight.add(lease.ticket)
        self._prefixes.move_to_end(key)

    def _release_inflight_prefix(self, lease: PrefixLease) -> None:
        endpoint = self._prefixes.get((lease.lane, lease.digest))
        if (
            endpoint is None
            or endpoint.lease != lease
            or lease.ticket not in endpoint.inflight
        ):
            raise ValueError("stale in-flight prefix lease")
        endpoint.inflight.remove(lease.ticket)

    def _validate_lane(self, lane: int) -> None:
        if type(lane) is not int or not 0 <= lane < len(self.slots):
            raise ValueError("prefix lane is out of range")

    def _release_sequence(self, sequence: _Sequence) -> None:
        del self._sequences[sequence.lease.request_id]
        self.histories[sequence.lease.lane].release(sequence.lease.history_blocks)
        self.slots[sequence.lease.lane].release(sequence.lease.slot)

    def blocks_for_tokens(self, tokens: int) -> int:
        return (tokens + self.history_block_tokens - 1) // self.history_block_tokens

    def _prefix_digests(self, token_ids: tuple[int, ...]):
        for boundary, digest in self._complete_block_digests(token_ids):
            if boundary == len(token_ids):
                break
            yield boundary, digest

    def _complete_block_digests(self, token_ids: tuple[int, ...]):
        digest = bytes(16)
        for boundary in range(
            self.history_block_tokens,
            len(token_ids) + 1,
            self.history_block_tokens,
        ):
            block = token_ids[boundary - self.history_block_tokens : boundary]
            digest = blake2b(
                digest + self._token_block.pack(*block), digest_size=len(digest)
            ).digest()
            yield boundary, digest

    def _prefix_digest(self, token_ids: tuple[int, ...]) -> bytes:
        if not token_ids or len(token_ids) % self.history_block_tokens:
            raise ValueError("a cached prefix must contain complete token blocks")
        for _, digest in self._complete_block_digests(token_ids):
            pass
        return digest

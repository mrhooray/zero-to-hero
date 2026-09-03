from dataclasses import dataclass, field
from struct import Struct, error

MAX_BATCH_PLAN_BYTES = 4 * 1024 * 1024
MAX_TOKEN_ID = (1 << 32) - 1
_NONE_GRAPH_BUCKET = (1 << 32) - 1
_NONE_STATE_SLOT = (1 << 32) - 1
_PLAN = Struct("<QII")
_COUNT = Struct("<I")
_ROW = Struct("<IIIIIIIIII")


@dataclass(slots=True)
class Request:
    request_id: int
    prompt_token_ids: tuple[int, ...]
    max_output_tokens: int
    temperature: float
    top_p: float
    seed: int
    ignore_eos: bool = False
    slot: int | None = None
    lane: int | None = None
    committed_tokens: int = 0
    cached_tokens: int = 0
    output_token_ids: list[int] = field(default_factory=list)
    cancelled: bool = False


@dataclass(frozen=True, slots=True)
class TableDelta:
    start_block: int
    physical_blocks: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class BatchPlanRow:
    slot: int
    token_start: int
    token_ids: tuple[int, ...]
    max_accept_tokens: int
    sample: bool
    table_delta: TableDelta
    ignore_eos: bool = False
    restore_slot: int | None = None
    capture_slot: int | None = None


@dataclass(frozen=True, slots=True)
class BatchPlan:
    step_id: int
    lanes: tuple[tuple[BatchPlanRow, ...], ...]
    graph_bucket: int | None

    @property
    def staging_index(self) -> int:
        return self.step_id & 1


@dataclass(frozen=True, slots=True)
class StepResultRow:
    slot: int
    accepted_count: int
    output_token_ids: tuple[int, ...]
    finished: bool


@dataclass(frozen=True, slots=True)
class StepResult:
    step_id: int
    lanes: tuple[tuple[StepResultRow, ...], ...]
    elapsed_ns: int


def encode_batch_plan(plan: BatchPlan | None, buffer: bytearray) -> int:
    if plan is None:
        return 0
    if type(plan) is not BatchPlan:
        raise TypeError("plan must be a BatchPlan or None")

    try:
        graph_bucket = (
            _NONE_GRAPH_BUCKET if plan.graph_bucket is None else plan.graph_bucket
        )
        _PLAN.pack_into(buffer, 0, plan.step_id, graph_bucket, len(plan.lanes))
        offset = _PLAN.size
        for lane in plan.lanes:
            _COUNT.pack_into(buffer, offset, len(lane))
            offset += _COUNT.size
            for row in lane:
                token_count = len(row.token_ids)
                block_count = len(row.table_delta.physical_blocks)
                _ROW.pack_into(
                    buffer,
                    offset,
                    row.slot,
                    row.token_start,
                    token_count,
                    row.max_accept_tokens,
                    int(row.sample),
                    row.table_delta.start_block,
                    block_count,
                    int(row.ignore_eos),
                    _NONE_STATE_SLOT if row.restore_slot is None else row.restore_slot,
                    _NONE_STATE_SLOT if row.capture_slot is None else row.capture_slot,
                )
                offset += _ROW.size
                offset = _pack_uints(buffer, offset, row.token_ids)
                offset = _pack_uints(buffer, offset, row.table_delta.physical_blocks)
    except (error, OverflowError) as exception:
        raise ValueError("batch plan does not fit its binary schema") from exception
    if offset > MAX_BATCH_PLAN_BYTES:
        raise ValueError("batch plan exceeds its transport envelope")
    return offset


def decode_batch_plan(payload: bytes) -> BatchPlan | None:
    if not payload:
        return None
    if len(payload) > MAX_BATCH_PLAN_BYTES:
        raise ValueError("batch plan exceeds its transport envelope")

    offset = 0

    def take(schema: Struct) -> tuple[int, ...]:
        nonlocal offset
        try:
            values = schema.unpack_from(payload, offset)
        except error as exception:
            raise ValueError("truncated batch plan") from exception
        offset += schema.size
        return values

    step_id, graph_bucket, lane_count = take(_PLAN)
    lanes = []
    for _ in range(lane_count):
        (row_count,) = take(_COUNT)
        rows = []
        for _ in range(row_count):
            (
                slot,
                token_start,
                token_count,
                max_accept_tokens,
                sample,
                start_block,
                block_count,
                ignore_eos,
                restore_slot,
                capture_slot,
            ) = take(_ROW)
            if sample not in (0, 1):
                raise ValueError("invalid batch plan sample flag")
            if ignore_eos not in (0, 1):
                raise ValueError("invalid batch plan ignore_eos flag")
            token_ids, offset = _unpack_uints(payload, offset, token_count)
            physical_blocks, offset = _unpack_uints(payload, offset, block_count)
            rows.append(
                BatchPlanRow(
                    slot,
                    token_start,
                    token_ids,
                    max_accept_tokens,
                    bool(sample),
                    TableDelta(start_block, physical_blocks),
                    bool(ignore_eos),
                    None if restore_slot == _NONE_STATE_SLOT else restore_slot,
                    None if capture_slot == _NONE_STATE_SLOT else capture_slot,
                )
            )
        lanes.append(tuple(rows))
    if offset != len(payload):
        raise ValueError("batch plan has trailing bytes")
    return BatchPlan(
        step_id,
        tuple(lanes),
        None if graph_bucket == _NONE_GRAPH_BUCKET else graph_bucket,
    )


def _pack_uints(buffer: bytearray, offset: int, values: tuple[int, ...]) -> int:
    if values:
        Struct(f"<{len(values)}I").pack_into(buffer, offset, *values)
    return offset + 4 * len(values)


def _unpack_uints(
    payload: bytes, offset: int, count: int
) -> tuple[tuple[int, ...], int]:
    size = 4 * count
    if offset + size > len(payload):
        raise ValueError("truncated batch plan")
    return Struct(f"<{count}I").unpack_from(payload, offset), offset + size

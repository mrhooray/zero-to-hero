import triton
import triton.language as tl

from infer.models.deepseek_v4_flash import model as deepseek_v4_flash

_BLOCK = 128
_TABLE_WIDTH = 8_192


def stage_deepseek_v4_verify(verify) -> None:
    _stage_target_rows(
        verify.execution,
        verify.staging.device.state_slots,
        verify.staging.device.table_deltas,
        verify.candidates,
        verify.committed_lengths,
        verify.staging.device.active,
        verify.candidates.shape[1],
        False,
        verify.live_slots,
    )


def stage_deepseek_v4_target(target) -> None:
    _stage_target_rows(
        target.execution,
        target.staging.device.state_slots,
        target.staging.device.table_deltas,
        target.sampled_tokens,
        target.committed_lengths,
        target.staging.device.active,
        1,
        True,
        target.live_slots,
    )


def _stage_target_rows(
    runtime,
    state_slots,
    table_deltas,
    input_tokens,
    committed_lengths,
    active,
    width,
    input_by_slot,
    dummy_base,
) -> None:
    if not runtime.token_ids.numel():
        return
    descriptors = runtime._descriptors
    _stage_verify[(runtime.token_ids.shape[0], triton.cdiv(_TABLE_WIDTH, _BLOCK))](
        state_slots,
        table_deltas,
        runtime.execution_tables,
        runtime.token_ids,
        descriptors.positions,
        descriptors.request_indices,
        descriptors.raw_slots,
        descriptors.raw_indices,
        descriptors.raw_lengths,
        descriptors.c4_slots,
        descriptors.c4_outputs,
        descriptors.c4_table,
        descriptors.c4_base,
        descriptors.c128_slots,
        descriptors.c128_outputs,
        descriptors.c128_table,
        descriptors.c128_base,
        descriptors.block_table,
        descriptors.c4_lengths,
        descriptors.c128_lengths,
        input_tokens,
        committed_lengths,
        active,
        VERIFY_WIDTH=width,
        INPUT_BY_SLOT=input_by_slot,
        DUMMY_BASE=dummy_base,
        RAW_RING=deepseek_v4_flash.RAW_STATE_RING_TOKENS,
        C4_RING=deepseek_v4_flash.C4_STATE_RING_TOKENS,
        C128_RING=deepseek_v4_flash.C128_STATE_RING_TOKENS,
        MAX_CONTEXT_TOKENS=_TABLE_WIDTH * _BLOCK,
        TABLE_WIDTH=_TABLE_WIDTH,
        BLOCK=_BLOCK,
    )


def publish_deepseek_v4_prefill_lengths(
    committed_lengths, state_slots, end_lengths
) -> None:
    _publish_lengths[(state_slots.numel(),)](
        committed_lengths,
        state_slots,
        end_lengths,
    )


def stage_deepseek_v4_prefill_outputs(
    positions, request_indices, block_table, c4_outputs, c128_outputs
) -> None:
    _stage_prefill_outputs[(positions.numel(),)](
        positions,
        request_indices,
        block_table,
        c4_outputs,
        c128_outputs,
        block_table.shape[1],
    )


def stage_deepseek_v4_dspark(
    batch, controls, committed_lengths, sampled_tokens, candidates, dummy_base
) -> None:
    groups = batch.anchor_token_ids.shape[0]
    if not groups:
        return
    _stage_dspark[(groups,)](
        controls.state_slots,
        controls.active,
        committed_lengths,
        sampled_tokens,
        candidates,
        batch.anchor_token_ids,
        batch.anchor_positions,
        batch.state_slots,
        batch.input_token_ids,
        batch.positions,
        batch.persistent_slots,
        batch.block_slots,
        batch.context_indices,
        batch.context_lengths,
        batch.block_indices,
        batch.block_lengths,
        DUMMY_BASE=dummy_base,
        MAX_CONTEXT_TOKENS=deepseek_v4_flash.MAX_CONTEXT_TOKENS,
        WINDOW=deepseek_v4_flash.DSPARK_WINDOW_TOKENS,
        PAGE=deepseek_v4_flash.DSPARK_PAGE_TOKENS,
        BLOCK=deepseek_v4_flash.DSPARK_BLOCK_SIZE,
        VERIFY_WIDTH=deepseek_v4_flash.DSPARK_VERIFY_WIDTH,
        NOISE_TOKEN_ID=deepseek_v4_flash.DSPARK_NOISE_TOKEN_ID,
    )


def commit_deepseek_v4_dspark(
    captured_hidden,
    accepted,
    controls,
    committed_lengths,
    anchor_hidden,
    commit_hidden,
    positions,
    window_slots,
) -> None:
    groups = accepted.numel()
    if not groups:
        return
    width = captured_hidden.shape[-1]
    _commit_dspark[(groups * deepseek_v4_flash.DSPARK_BLOCK_SIZE, triton.cdiv(width, 256))](
        captured_hidden,
        accepted,
        controls.state_slots,
        controls.active,
        committed_lengths,
        anchor_hidden,
        commit_hidden,
        positions,
        window_slots,
        CAPTURE_WIDTH=width,
        VERIFY_WIDTH=deepseek_v4_flash.DSPARK_VERIFY_WIDTH,
        BLOCK=deepseek_v4_flash.DSPARK_BLOCK_SIZE,
        WINDOW=deepseek_v4_flash.DSPARK_WINDOW_TOKENS,
    )


def seed_deepseek_v4_dspark_anchors(
    captured_hidden, source_indices, state_slots, anchor_hidden
) -> None:
    requests = source_indices.numel()
    if not requests:
        return
    width = captured_hidden.shape[1]
    _seed_dspark_anchors[(requests, triton.cdiv(width, 256))](
        captured_hidden,
        source_indices,
        state_slots,
        anchor_hidden,
        CAPTURE_WIDTH=width,
    )


@triton.jit
def _stage_verify(
    state_slots,
    table_deltas,
    execution_tables,
    token_ids,
    positions,
    request_indices,
    raw_slots,
    raw_indices,
    raw_lengths,
    c4_slots,
    c4_outputs,
    c4_table,
    c4_base,
    c128_slots,
    c128_outputs,
    c128_table,
    c128_base,
    block_table,
    c4_lengths,
    c128_lengths,
    candidate_token_ids,
    committed_lengths,
    active_ptr,
    VERIFY_WIDTH: tl.constexpr,
    INPUT_BY_SLOT: tl.constexpr,
    DUMMY_BASE: tl.constexpr,
    RAW_RING: tl.constexpr,
    C4_RING: tl.constexpr,
    C128_RING: tl.constexpr,
    MAX_CONTEXT_TOKENS: tl.constexpr,
    TABLE_WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    block = tl.program_id(1)
    group = row // VERIFY_WIDTH
    step = row % VERIFY_WIDTH
    control_slot = tl.load(state_slots + group)
    committed = tl.load(committed_lengths + control_slot)
    live = (tl.load(active_ptr + group) != 0) & (committed + step < MAX_CONTEXT_TOKENS)
    slot = tl.where(live, control_slot, DUMMY_BASE + group)
    position = tl.where(live, committed + step, step)
    delta_column = tl.load(table_deltas + group * 2)
    delta_block = tl.load(table_deltas + group * 2 + 1)

    offsets = tl.arange(0, BLOCK)
    columns = block * BLOCK + offsets
    table_offsets = slot * TABLE_WIDTH + columns
    values = tl.load(
        execution_tables + table_offsets,
        mask=columns < TABLE_WIDTH,
        other=-1,
    )
    changed = (columns == delta_column) & (delta_block >= 0)
    control_offsets = control_slot * TABLE_WIDTH + columns
    tl.store(
        execution_tables + control_offsets,
        delta_block,
        mask=changed & (step == 0),
    )
    values = tl.where(live & changed, delta_block, values)
    tl.store(
        block_table + row * TABLE_WIDTH + columns,
        values,
        mask=columns < TABLE_WIDTH,
    )

    first = block == 0
    length = tl.minimum(position + 1, 128)
    logical = position - length + 1 + offsets
    tl.store(
        raw_indices + row * 128 + offsets,
        slot * RAW_RING + logical % RAW_RING,
        mask=first & (offsets < length),
    )
    history_column = position // 128
    history_block = tl.load(execution_tables + slot * TABLE_WIDTH + history_column)
    history_block = tl.where(
        live & (history_column == delta_column) & (delta_block >= 0),
        delta_block,
        history_block,
    )
    c4_page = tl.maximum(position // 4 - 2, 0)
    c128_page = tl.maximum(position // 8 - 16, 0)
    if INPUT_BY_SLOT:
        token = tl.load(candidate_token_ids + control_slot)
    else:
        token = tl.load(candidate_token_ids + row)
    tl.store(token_ids + row, token, mask=first)
    tl.store(positions + row, position, mask=first)
    tl.store(request_indices + row, row, mask=first)
    tl.store(raw_slots + row, slot * RAW_RING + position % RAW_RING, mask=first)
    tl.store(raw_lengths + row, length, mask=first)
    tl.store(c4_slots + row, slot * C4_RING + position % C4_RING, mask=first)
    tl.store(
        c4_outputs + row,
        tl.where(
            (position + 1) % 4 == 0,
            history_block * 32 + (position // 4) % 32,
            -1,
        ),
        mask=first,
    )
    tl.store(c4_base + row, c4_page, mask=first)
    tl.store(c4_lengths + row, tl.where(live, (position + 1) // 4, 0), mask=first)
    tl.store(
        c4_table + row * 3 + offsets,
        slot * (C4_RING // 4) + (c4_page + offsets) % (C4_RING // 4),
        mask=first & (offsets < 3),
    )
    tl.store(c128_slots + row, slot * C128_RING + position % C128_RING, mask=first)
    tl.store(
        c128_outputs + row,
        tl.where((position + 1) % 128 == 0, history_block, -1),
        mask=first,
    )
    tl.store(c128_base + row, c128_page, mask=first)
    tl.store(
        c128_lengths + row,
        tl.where(live, (position + 1) // 128, 0),
        mask=first,
    )
    tl.store(
        c128_table + row * 17 + offsets,
        slot * (C128_RING // 8) + (c128_page + offsets) % (C128_RING // 8),
        mask=first & (offsets < 17),
    )


@triton.jit(do_not_specialize=["table_width"])
def _stage_prefill_outputs(
    positions,
    request_indices,
    block_table,
    c4_outputs,
    c128_outputs,
    table_width,
):
    token = tl.program_id(0)
    position = tl.load(positions + token)
    request = tl.load(request_indices + token)
    block = tl.load(block_table + request * table_width + position // 128)
    tl.store(
        c4_outputs + token,
        tl.where((position + 1) % 4 == 0, block * 32 + (position // 4) % 32, -1),
    )
    tl.store(
        c128_outputs + token,
        tl.where((position + 1) % 128 == 0, block, -1),
    )


@triton.jit
def _publish_lengths(committed_lengths, state_slots, end_lengths):
    row = tl.program_id(0)
    tl.store(
        committed_lengths + tl.load(state_slots + row),
        tl.load(end_lengths + row),
    )


@triton.jit
def _stage_dspark(
    control_slots,
    active_ptr,
    committed_lengths,
    sampled_tokens,
    candidates,
    anchor_token_ids,
    anchor_positions,
    state_slots,
    input_token_ids,
    positions,
    persistent_slots,
    block_slots,
    context_indices,
    context_lengths,
    block_indices,
    block_lengths,
    DUMMY_BASE: tl.constexpr,
    MAX_CONTEXT_TOKENS: tl.constexpr,
    WINDOW: tl.constexpr,
    PAGE: tl.constexpr,
    BLOCK: tl.constexpr,
    VERIFY_WIDTH: tl.constexpr,
    NOISE_TOKEN_ID: tl.constexpr,
):
    group = tl.program_id(0)
    control_slot = tl.load(control_slots + group)
    length = tl.load(committed_lengths + control_slot)
    active = tl.load(active_ptr + group) != 0
    live = active & (length > 0) & (length <= MAX_CONTEXT_TOKENS - BLOCK)
    slot = tl.where(live, control_slot, DUMMY_BASE + group)
    anchor_position = tl.where(live, length - 1, 0)
    sampled = tl.load(sampled_tokens + control_slot)
    token = tl.where(live, sampled, 0)
    context_length = tl.minimum(anchor_position + 1, WINDOW)
    logical_start = anchor_position - context_length + 1
    offsets = tl.arange(0, WINDOW)

    tl.store(anchor_token_ids + group, token)
    tl.store(anchor_positions + group, anchor_position)
    tl.store(state_slots + group, slot)
    tl.store(
        persistent_slots + group,
        slot * WINDOW + anchor_position % WINDOW,
    )
    tl.store(
        candidates + group * VERIFY_WIDTH,
        tl.where(active, sampled, 0),
    )
    for step in range(BLOCK):
        row = group * BLOCK + step
        tl.store(
            input_token_ids + row,
            tl.where(step == 0, token, NOISE_TOKEN_ID),
        )
        tl.store(positions + row, anchor_position + step + 1)
        tl.store(block_slots + row, group * PAGE + step)
        tl.store(context_lengths + row, context_length)
        tl.store(block_lengths + row, BLOCK)
        tl.store(
            context_indices + row * WINDOW + offsets,
            slot * WINDOW + (logical_start + offsets) % WINDOW,
            mask=offsets < context_length,
        )
        tl.store(
            block_indices + row * PAGE + offsets,
            group * PAGE + offsets,
            mask=offsets < BLOCK,
        )


@triton.jit
def _commit_dspark(
    captured_hidden,
    accepted_ptr,
    state_slots,
    active_ptr,
    committed_lengths,
    anchor_hidden,
    commit_hidden,
    positions,
    window_slots,
    CAPTURE_WIDTH: tl.constexpr,
    VERIFY_WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
    WINDOW: tl.constexpr,
):
    row = tl.program_id(0)
    group = row // BLOCK
    offset = row % BLOCK
    dimensions = tl.program_id(1) * 256 + tl.arange(0, 256)
    dimension_valid = dimensions < CAPTURE_WIDTH
    values = tl.load(
        captured_hidden + (group * VERIFY_WIDTH + offset) * CAPTURE_WIDTH + dimensions,
        mask=dimension_valid,
    )
    tl.store(
        commit_hidden + row * CAPTURE_WIDTH + dimensions,
        values,
        mask=dimension_valid,
    )

    accepted = tl.load(accepted_ptr + group)
    active = tl.load(active_ptr + group) != 0
    state_slot = tl.load(state_slots + group)
    selected = tl.maximum(accepted - 1, 0)
    selected_values = tl.load(
        captured_hidden
        + (group * VERIFY_WIDTH + selected) * CAPTURE_WIDTH
        + dimensions,
        mask=dimension_valid & active & (offset == 0),
    )
    tl.store(
        anchor_hidden + state_slot * CAPTURE_WIDTH + dimensions,
        selected_values,
        mask=dimension_valid & active & (offset == 0),
    )

    first = tl.program_id(1) == 0
    position = tl.load(committed_lengths + state_slot) + offset
    valid = active & (offset < accepted - 1)
    tl.store(positions + row, position, mask=first)
    tl.store(
        window_slots + row,
        tl.where(valid, state_slot * WINDOW + position % WINDOW, -1),
        mask=first,
    )


@triton.jit
def _seed_dspark_anchors(
    captured_hidden,
    source_indices,
    state_slots,
    anchor_hidden,
    CAPTURE_WIDTH: tl.constexpr,
):
    request = tl.program_id(0)
    dimensions = tl.program_id(1) * 256 + tl.arange(0, 256)
    valid = dimensions < CAPTURE_WIDTH
    source = tl.load(source_indices + request)
    slot = tl.load(state_slots + request)
    values = tl.load(
        captured_hidden + source * CAPTURE_WIDTH + dimensions,
        mask=valid,
    )
    tl.store(
        anchor_hidden + slot * CAPTURE_WIDTH + dimensions,
        values,
        mask=valid,
    )

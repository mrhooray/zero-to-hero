"""Compact packed target output into shifted GLM NextN input pairs."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from infer.models.glm53_flash.model import (
    HIDDEN_SIZE,
    KDA_CHUNK_SIZE,
)

_BLOCK_SIZE = 256


def glm53_stage_nextn_prefill(
    token_ids: torch.Tensor,
    target_hidden: torch.Tensor,
    pending_hidden: torch.Tensor,
    cu_seqlens: torch.Tensor,
    start_tokens: torch.Tensor,
    state_slots: torch.Tensor,
    source_rows: torch.Tensor,
    has_pending: torch.Tensor,
    last_indices: torch.Tensor,
    last_state_slots: torch.Tensor,
    output_token_ids: torch.Tensor,
    output_hidden: torch.Tensor,
    sequence_ids: torch.Tensor,
    token_state_slots: torch.Tensor,
    active: torch.Tensor,
    raw_lengths: torch.Tensor,
    total_tokens: int,
    sequence_count: int,
) -> None:
    source_tokens = token_ids.shape[0] if token_ids.ndim == 1 else 0
    live_slots = pending_hidden.shape[0] if pending_hidden.ndim == 2 else 0
    max_sequences = cu_seqlens.shape[0] - 1 if cu_seqlens.ndim == 1 else 0
    device = token_ids.device
    specifications = (
        (token_ids, torch.int64, (source_tokens,)),
        (target_hidden, torch.bfloat16, (source_tokens, HIDDEN_SIZE)),
        (pending_hidden, torch.bfloat16, (live_slots, HIDDEN_SIZE)),
        (cu_seqlens, torch.int32, (max_sequences + 1,)),
        (start_tokens, torch.int32, (max_sequences,)),
        (state_slots, torch.int64, (max_sequences,)),
        (source_rows, torch.int32, (max_sequences,)),
        (has_pending, torch.bool, (max_sequences,)),
        (last_indices, torch.int64, (max_sequences,)),
        (last_state_slots, torch.int64, (max_sequences,)),
        (output_token_ids, torch.int64, (KDA_CHUNK_SIZE,)),
        (output_hidden, torch.bfloat16, (total_tokens, HIDDEN_SIZE)),
        (sequence_ids, torch.int32, (KDA_CHUNK_SIZE,)),
        (token_state_slots, torch.int64, (KDA_CHUNK_SIZE,)),
        (active, torch.uint8, (KDA_CHUNK_SIZE,)),
        (raw_lengths, torch.int32, (KDA_CHUNK_SIZE,)),
    )
    if (
        not 0 <= total_tokens <= source_tokens <= KDA_CHUNK_SIZE
        or not 1 <= sequence_count <= max_sequences
        or not live_slots
    ):
        raise ValueError("invalid GLM NextN packed prefill size")
    for tensor, dtype, shape in specifications:
        if tuple(tensor.shape) != shape or tensor.dtype != dtype:
            raise ValueError("invalid GLM NextN packed prefill tensor")
        if tensor.device != device or not tensor.is_cuda or not tensor.is_contiguous():
            raise ValueError("GLM NextN packed prefill tensors must share CUDA layout")

    _glm53_stage_nextn_descriptors[(1,)](
        cu_seqlens,
        start_tokens,
        state_slots,
        source_rows,
        has_pending,
        last_indices,
        last_state_slots,
        sequence_count,
        MAX_SEQUENCES=max_sequences,
    )
    if not total_tokens:
        return
    _glm53_stage_nextn_prefill_kernel[
        (total_tokens, triton.cdiv(HIDDEN_SIZE, _BLOCK_SIZE))
    ](
        token_ids,
        target_hidden,
        pending_hidden,
        cu_seqlens,
        source_rows,
        start_tokens,
        state_slots,
        has_pending,
        output_token_ids,
        output_hidden,
        sequence_ids,
        token_state_slots,
        active,
        raw_lengths,
        MAX_SEQUENCES=max_sequences,
        HIDDEN=HIDDEN_SIZE,
        BLOCK_SIZE=_BLOCK_SIZE,
    )


@triton.jit
def _glm53_stage_nextn_descriptors(
    cu_seqlens,
    start_tokens,
    state_slots,
    source_rows,
    has_pending,
    last_indices,
    last_state_slots,
    sequence_count,
    MAX_SEQUENCES: tl.constexpr,
):
    indices = tl.arange(0, MAX_SEQUENCES)
    valid = indices < sequence_count
    starts = tl.load(start_tokens + indices, mask=valid, other=0)
    slots = tl.load(state_slots + indices, mask=valid, other=0)
    begins = tl.load(cu_seqlens + indices, mask=valid, other=0)
    ends = tl.load(cu_seqlens + indices + 1, mask=valid, other=0)
    fresh = (starts == 0).to(tl.int32)
    pair_counts = tl.where(valid, ends - begins - fresh, 0)
    present = pair_counts > 0
    compact_indices = tl.cumsum(present.to(tl.int32), axis=0) - 1
    pair_ends = tl.cumsum(pair_counts, axis=0)

    tl.store(source_rows + compact_indices, begins + fresh, mask=present)
    tl.store(start_tokens + compact_indices, tl.maximum(starts - 1, 0), mask=present)
    tl.store(state_slots + compact_indices, slots, mask=present)
    tl.store(has_pending + compact_indices, starts > 0, mask=present)
    tl.store(cu_seqlens + compact_indices + 1, pair_ends, mask=present)
    tl.store(last_indices + indices, ends - 1, mask=valid)
    tl.store(last_state_slots + indices, slots, mask=valid)

    active_count = tl.sum(present.to(tl.int32), axis=0)
    total_tokens = tl.sum(pair_counts, axis=0)
    tl.store(cu_seqlens, 0)
    tl.store(
        cu_seqlens + indices + 1,
        total_tokens,
        mask=indices >= active_count,
    )


@triton.jit
def _glm53_stage_nextn_prefill_kernel(
    token_ids,
    target_hidden,
    pending_hidden,
    cu_seqlens,
    source_rows,
    start_tokens,
    state_slots,
    has_pending,
    output_token_ids,
    output_hidden,
    sequence_ids,
    token_state_slots,
    active,
    raw_lengths,
    MAX_SEQUENCES: tl.constexpr,
    HIDDEN: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    block = tl.program_id(1)
    sequence = 0
    for index in tl.static_range(MAX_SEQUENCES):
        sequence += (row >= tl.load(cu_seqlens + index + 1)).to(tl.int32)
    offset = row - tl.load(cu_seqlens + sequence)
    source = tl.load(source_rows + sequence) + offset
    slot = tl.load(state_slots + sequence)

    columns = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    column_mask = columns < HIDDEN
    previous = tl.maximum(source - 1, 0)
    target_value = tl.load(
        target_hidden + previous * HIDDEN + columns,
        mask=column_mask,
    )
    pending_value = tl.load(
        pending_hidden + slot * HIDDEN + columns,
        mask=column_mask,
    )
    use_pending = (tl.load(has_pending + sequence) != 0) & (offset == 0)
    tl.store(
        output_hidden + row * HIDDEN + columns,
        tl.where(use_pending, pending_value, target_value),
        mask=column_mask,
    )

    first_block = block == 0
    tl.store(output_token_ids + row, tl.load(token_ids + source), mask=first_block)
    tl.store(sequence_ids + row, sequence, mask=first_block)
    tl.store(token_state_slots + row, slot, mask=first_block)
    tl.store(active + row, 1, mask=first_block)
    tl.store(
        raw_lengths + row,
        tl.load(start_tokens + sequence) + offset + 1,
        mask=first_block,
    )

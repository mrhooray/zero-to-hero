from __future__ import annotations

import torch
import triton
import triton.language as tl

from infer.models.glm53_flash.codec import EOS_TOKEN_IDS
from infer.models.glm53_flash.model import GLM53_TARGET_VERIFY_WIDTH

_WIDTH = GLM53_TARGET_VERIFY_WIDTH
_BLOCK_SIZE = 8
_COPY_BLOCK_SIZE = 256
_TABLE_COPY_BLOCK_WORDS = 1024
_EOS_0, _EOS_1, _EOS_2 = sorted(EOS_TOKEN_IDS)


def glm53_greedy_accept(
    candidate_token_ids: torch.Tensor,
    target_token_ids: torch.Tensor,
    remaining_tokens: torch.Tensor,
    ignore_eos: torch.Tensor,
    active: torch.Tensor,
    accepted_count: torch.Tensor,
    output_token_ids: torch.Tensor,
    continuing: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Write each active row's accepted target prefix."""

    device = candidate_token_ids.device
    if device.type != "cuda":
        raise ValueError(f"candidate_token_ids must be on CUDA, got {device}")
    batch_size = candidate_token_ids.shape[0] if candidate_token_ids.ndim == 2 else 0
    for name, tensor, shape, dtype in (
        (
            "candidate_token_ids",
            candidate_token_ids,
            (batch_size, _WIDTH),
            torch.int64,
        ),
        ("target_token_ids", target_token_ids, (batch_size, _WIDTH), torch.int64),
        ("remaining_tokens", remaining_tokens, (batch_size,), torch.int32),
        ("ignore_eos", ignore_eos, (batch_size,), torch.uint8),
        ("active", active, (batch_size,), torch.uint8),
        ("accepted_count", accepted_count, (batch_size,), torch.int32),
        (
            "output_token_ids",
            output_token_ids,
            (batch_size, _WIDTH),
            torch.int64,
        ),
        ("continuing", continuing, (batch_size,), torch.uint8),
    ):
        _require_tensor(name, tensor, shape, dtype, device)
    if not batch_size:
        raise ValueError("candidate_token_ids must contain at least one row")

    _glm53_greedy_accept_kernel[(batch_size,)](
        candidate_token_ids,
        target_token_ids,
        remaining_tokens,
        ignore_eos,
        active,
        accepted_count,
        output_token_ids,
        continuing,
        EOS_0=_EOS_0,
        EOS_1=_EOS_1,
        EOS_2=_EOS_2,
        WIDTH=_WIDTH,
        BLOCK_SIZE=_BLOCK_SIZE,
        num_warps=1,
    )
    return accepted_count, output_token_ids, continuing


def glm53_copy_verified_rows(
    source: torch.Tensor,
    destination: torch.Tensor,
    accepted_count: torch.Tensor,
    state_slots: torch.Tensor,
    active: torch.Tensor,
) -> None:
    """Commit one device-selected transaction row per active request."""

    batch_size = accepted_count.shape[0] if accepted_count.ndim == 1 else 0
    device = source.device
    if (
        not batch_size
        or source.ndim < 2
        or source.shape[0] != batch_size * _WIDTH
        or destination.ndim != source.ndim
        or destination.shape[1:] != source.shape[1:]
        or source.dtype != destination.dtype
        or source.device != destination.device
        or not source.is_cuda
    ):
        raise ValueError("source and destination must be compatible CUDA row tensors")
    for name, tensor, dtype in (
        ("accepted_count", accepted_count, torch.int32),
        ("state_slots", state_slots, torch.int32),
        ("active", active, torch.uint8),
    ):
        _require_tensor(name, tensor, (batch_size,), dtype, device)
    row_size = _require_row_layout(source)
    if _require_row_layout(destination) != row_size:
        raise ValueError("source and destination rows must have the same layout")

    _glm53_copy_verified_rows_kernel[
        (batch_size, triton.cdiv(row_size, _COPY_BLOCK_SIZE))
    ](
        source,
        destination,
        accepted_count,
        state_slots,
        active,
        source.stride(0),
        destination.stride(0),
        ROW_SIZE=row_size,
        WIDTH=_WIDTH,
        BLOCK_SIZE=_COPY_BLOCK_SIZE,
    )


def glm53_copy_verified_row_table(
    source_addresses: torch.Tensor,
    destination_addresses: torch.Tensor,
    source_strides: torch.Tensor,
    destination_strides: torch.Tensor,
    accepted_count: torch.Tensor,
    state_slots: torch.Tensor,
    active: torch.Tensor,
    *,
    row_words: int,
) -> None:
    """Commit one family of device-selected transaction rows."""

    layers = source_addresses.shape[0] if source_addresses.ndim == 1 else 0
    batch_size = accepted_count.shape[0] if accepted_count.ndim == 1 else 0
    device = source_addresses.device
    if not layers or not batch_size or type(row_words) is not int or row_words < 1:
        raise ValueError("copy table, request rows, and row width must be non-empty")
    for name, tensor, dtype in (
        ("source_addresses", source_addresses, torch.uint64),
        ("destination_addresses", destination_addresses, torch.uint64),
        ("source_strides", source_strides, torch.int64),
        ("destination_strides", destination_strides, torch.int64),
    ):
        _require_tensor(name, tensor, (layers,), dtype, device)
    for name, tensor, dtype in (
        ("accepted_count", accepted_count, torch.int32),
        ("state_slots", state_slots, torch.int32),
        ("active", active, torch.uint8),
    ):
        _require_tensor(name, tensor, (batch_size,), dtype, device)

    _glm53_copy_verified_row_table_kernel[
        (
            layers * batch_size,
            triton.cdiv(row_words, _TABLE_COPY_BLOCK_WORDS),
        )
    ](
        source_addresses,
        destination_addresses,
        source_strides,
        destination_strides,
        accepted_count,
        state_slots,
        active,
        GROUPS=batch_size,
        WIDTH=_WIDTH,
        ROW_WORDS=row_words,
        BLOCK_WORDS=_TABLE_COPY_BLOCK_WORDS,
    )


def glm53_publish_accepted(
    output_token_ids: torch.Tensor,
    accepted_count: torch.Tensor,
    state_slots: torch.Tensor,
    active: torch.Tensor,
    sampled_tokens: torch.Tensor,
    committed_lengths: torch.Tensor,
) -> None:
    """Publish each active request's accepted tail and committed length."""

    batch_size = output_token_ids.shape[0] if output_token_ids.ndim == 2 else 0
    device = output_token_ids.device
    for name, tensor, shape, dtype in (
        ("output_token_ids", output_token_ids, (batch_size, _WIDTH), torch.int64),
        ("accepted_count", accepted_count, (batch_size,), torch.int32),
        ("state_slots", state_slots, (batch_size,), torch.int32),
        ("active", active, (batch_size,), torch.uint8),
    ):
        _require_tensor(name, tensor, shape, dtype, device)
    sampled_slots = sampled_tokens.shape[0] if sampled_tokens.ndim == 1 else 0
    live_slots = committed_lengths.shape[0] if committed_lengths.ndim == 1 else 0
    _require_tensor(
        "sampled_tokens", sampled_tokens, (sampled_slots,), torch.int64, device
    )
    _require_tensor(
        "committed_lengths", committed_lengths, (live_slots,), torch.int32, device
    )
    if not batch_size or not live_slots or sampled_slots < live_slots:
        raise ValueError("publish buffers must contain at least one row")

    _glm53_publish_accepted_kernel[(batch_size,)](
        output_token_ids,
        accepted_count,
        state_slots,
        active,
        sampled_tokens,
        committed_lengths,
        WIDTH=_WIDTH,
    )


def _require_tensor(name, tensor, shape, dtype, device) -> None:
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")
    if tensor.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {tensor.dtype}")
    if tensor.device != device or not tensor.is_cuda or not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous on {device}")


def _require_row_layout(tensor: torch.Tensor) -> int:
    row_size = 1
    for dimension in range(tensor.ndim - 1, 0, -1):
        if tensor.stride(dimension) != row_size:
            raise ValueError("tensor rows must be contiguous")
        row_size *= tensor.shape[dimension]
    return row_size


@triton.jit
def _glm53_greedy_accept_kernel(
    candidate_token_ids,
    target_token_ids,
    remaining_tokens,
    ignore_eos_ptr,
    active_ptr,
    accepted_count_ptr,
    output_token_ids,
    continuing_ptr,
    EOS_0: tl.constexpr,
    EOS_1: tl.constexpr,
    EOS_2: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    row_offset = row * WIDTH
    target_0 = tl.load(target_token_ids + row_offset)
    target_1 = tl.load(target_token_ids + row_offset + 1)
    target_2 = tl.load(target_token_ids + row_offset + 2)
    target_3 = tl.load(target_token_ids + row_offset + 3)
    match_1 = tl.load(candidate_token_ids + row_offset + 1) == target_0
    match_2 = match_1 & (tl.load(candidate_token_ids + row_offset + 2) == target_1)
    match_3 = match_2 & (tl.load(candidate_token_ids + row_offset + 3) == target_2)
    accepted = 1 + match_1.to(tl.int32) + match_2.to(tl.int32) + match_3.to(tl.int32)

    remaining = tl.maximum(0, tl.minimum(WIDTH, tl.load(remaining_tokens + row)))
    accepted = tl.where(tl.load(active_ptr + row), tl.minimum(accepted, remaining), 0)
    ignore_eos = tl.load(ignore_eos_ptr + row) != 0
    eos_0 = (target_0 == EOS_0) | (target_0 == EOS_1) | (target_0 == EOS_2)
    eos_1 = (target_1 == EOS_0) | (target_1 == EOS_1) | (target_1 == EOS_2)
    eos_2 = (target_2 == EOS_0) | (target_2 == EOS_1) | (target_2 == EOS_2)
    eos_3 = (target_3 == EOS_0) | (target_3 == EOS_1) | (target_3 == EOS_2)
    accepted = tl.where((accepted > 0) & eos_0 & ~ignore_eos, 1, accepted)
    accepted = tl.where((accepted > 1) & eos_1 & ~ignore_eos, 2, accepted)
    accepted = tl.where((accepted > 2) & eos_2 & ~ignore_eos, 3, accepted)
    accepted = tl.where((accepted > 3) & eos_3 & ~ignore_eos, 4, accepted)

    tl.store(accepted_count_ptr + row, accepted)
    last_target = tl.load(
        target_token_ids + row_offset + accepted - 1,
        mask=accepted > 0,
        other=EOS_0,
    )
    continuing = (
        (accepted > 0)
        & (
            ignore_eos
            | ((last_target != EOS_0) & (last_target != EOS_1) & (last_target != EOS_2))
        )
        & ((remaining == WIDTH) | (accepted < remaining))
    )
    tl.store(continuing_ptr + row, continuing)
    offsets = tl.arange(0, BLOCK_SIZE)
    target = tl.load(
        target_token_ids + row_offset + offsets,
        mask=offsets < WIDTH,
        other=0,
    )
    tl.store(
        output_token_ids + row_offset + offsets,
        tl.where(offsets < accepted, target, 0),
        mask=offsets < WIDTH,
    )


@triton.jit
def _glm53_copy_verified_rows_kernel(
    source,
    destination,
    accepted_count,
    state_slots,
    active_ptr,
    source_row_stride,
    destination_row_stride,
    ROW_SIZE: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    active = tl.load(active_ptr + row) != 0
    selected = row * WIDTH + tl.load(accepted_count + row) - 1
    state_slot = tl.load(state_slots + row)
    mask = active & (tl.load(accepted_count + row) > 0) & (offsets < ROW_SIZE)
    values = tl.load(
        source + selected * source_row_stride + offsets,
        mask=mask,
        other=0,
    )
    tl.store(
        destination + state_slot * destination_row_stride + offsets,
        values,
        mask=mask,
    )


@triton.jit
def _glm53_copy_verified_row_table_kernel(
    source_addresses,
    destination_addresses,
    source_strides,
    destination_strides,
    accepted_count,
    state_slots,
    active_ptr,
    GROUPS: tl.constexpr,
    WIDTH: tl.constexpr,
    ROW_WORDS: tl.constexpr,
    BLOCK_WORDS: tl.constexpr,
):
    work = tl.program_id(0)
    chunk = tl.program_id(1)
    layer = work // GROUPS
    row = work - layer * GROUPS
    source = tl.cast(tl.load(source_addresses + layer), tl.pointer_type(tl.int32))
    destination = tl.cast(
        tl.load(destination_addresses + layer), tl.pointer_type(tl.int32)
    )
    accepted = tl.load(accepted_count + row)
    selected = row * WIDTH + accepted - 1
    state_slot = tl.load(state_slots + row)
    offsets = chunk * BLOCK_WORDS + tl.arange(0, BLOCK_WORDS)
    mask = (tl.load(active_ptr + row) != 0) & (accepted > 0) & (offsets < ROW_WORDS)
    values = tl.load(
        source + selected * tl.load(source_strides + layer) + offsets,
        mask=mask,
        other=0,
    )
    tl.store(
        destination + state_slot * tl.load(destination_strides + layer) + offsets,
        values,
        mask=mask,
    )


@triton.jit
def _glm53_publish_accepted_kernel(
    output_token_ids,
    accepted_count,
    state_slots,
    active_ptr,
    sampled_tokens,
    committed_lengths,
    WIDTH: tl.constexpr,
):
    row = tl.program_id(0)
    active = tl.load(active_ptr + row) != 0
    accepted = tl.load(accepted_count + row)
    active = active & (accepted > 0)
    state_slot = tl.load(state_slots + row)
    token = tl.load(
        output_token_ids + row * WIDTH + accepted - 1,
        mask=active,
        other=0,
    )
    length = tl.load(committed_lengths + state_slot, mask=active, other=0)
    tl.store(sampled_tokens + state_slot, token, mask=active)
    tl.store(committed_lengths + state_slot, length + accepted, mask=active)

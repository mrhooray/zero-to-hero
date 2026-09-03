"""Fixed GLM-5.3-Flash packed variable-length causal convolution.

The Triton kernel is derived from TokenSpeed commit
f17b03efc1728875c586d848f49da5905032e87c, file
python/tokenspeed/runtime/layers/attention/linear/causal_conv1d.py. The upstream
continuous-batching kernel is lines 35-397; its launch and stride setup are
lines 482-511 and 596-650. This derivative fixes the GLM TP4 geometry, combines
the upstream sequence and chunk-ordinal arrays into ``(sequence, token_offset)``
rows, and writes directly to the FLA input layout. See
segmented_conv.LICENSE.txt for exact hashes, attribution, and terms.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise

import torch
import triton
import triton.language as tl

from infer.models.glm53_flash.model import (
    CONV_KERNEL_SIZE,
    HEAD_DIM,
)

TOKENSPEED_COMMIT = "f17b03efc1728875c586d848f49da5905032e87c"
TOKENSPEED_SOURCE_PATH = (
    "python/tokenspeed/runtime/layers/attention/linear/causal_conv1d.py"
)
TOKENSPEED_GIT_BLOB_SHA = "f586a0979dee382926a098dc356a41c4487e7de4"
TOKENSPEED_SOURCE_SHA256 = (
    "1f7afb81136fdd29aeb01407e4eb112fff20122d04b47330c57220e275d37036"
)
TOKENSPEED_KERNEL_SLICE_SHA256 = (
    "cd7b82369ee357287b2b7e6c38a39f92361f8c7fac1a3bc350d2e7de553fe41c"
)
TOKENSPEED_LAUNCH_SLICE_SHA256 = (
    "0bdda672e43db4477e35010536f2082c6f9e5e25cf313bf16679ef13c10e042c"
)

BLOCK_M = 8
BLOCK_N = 256


def validate_glm53_segmented_conv_plan(
    cu_seqlens: Sequence[int],
    state_indices: Sequence[int],
    has_initial: Sequence[bool],
    segments: Sequence[Sequence[int]],
    *,
    total_tokens: int,
    state_pages: int,
) -> int:
    """Validate a CPU prefill descriptor and return its program count.

    Each segment is ``(sequence, token_offset)``. Segments cover every positive-
    length sequence exactly once in canonical sequence-major order, with token
    offsets ``0, 8, ...``. State pages are unique because each sequence updates
    its three-token convolution history in place.
    """
    if type(total_tokens) is not int:
        raise TypeError("total_tokens must be an int")
    if total_tokens < 1:
        raise ValueError("total_tokens must be positive")
    if type(state_pages) is not int:
        raise TypeError("state_pages must be an int")
    if state_pages < 1:
        raise ValueError("state_pages must be positive")

    lengths = _plain_int_tuple("cu_seqlens", cu_seqlens)
    indices = _plain_int_tuple("state_indices", state_indices)
    initial = tuple(has_initial)
    batch = len(indices)
    if batch < 1:
        raise ValueError("prefill batch must be positive")
    if len(lengths) != batch + 1:
        raise ValueError("cu_seqlens must have one more element than state_indices")
    if lengths[0] != 0:
        raise ValueError("cu_seqlens must start at zero")
    if lengths[-1] != total_tokens:
        raise ValueError("cu_seqlens must end at total_tokens")
    if len(initial) != batch:
        raise ValueError("has_initial must have one element per sequence")
    for sequence, flag in enumerate(initial):
        if type(flag) is not bool:
            raise TypeError(f"has_initial[{sequence}] must be a bool")

    program_count = 0
    for sequence, (start, end) in enumerate(pairwise(lengths)):
        sequence_tokens = end - start
        if sequence_tokens < 1:
            raise ValueError(f"sequence {sequence} must contain at least one token")
        program_count += (sequence_tokens + BLOCK_M - 1) // BLOCK_M

    for sequence, index in enumerate(indices):
        if not 0 <= index < state_pages:
            raise ValueError(
                f"state_indices[{sequence}] must be in [0, {state_pages}), got {index}"
            )
    if len(set(indices)) != batch:
        raise ValueError("state_indices must be unique")

    rows = tuple(segments)
    if len(rows) != program_count:
        raise ValueError(f"segments must contain {program_count} rows, got {len(rows)}")
    program = 0
    for sequence, (start, end) in enumerate(pairwise(lengths)):
        for offset in range(0, end - start, BLOCK_M):
            row = tuple(rows[program])
            if len(row) != 2:
                raise ValueError(f"segments[{program}] must contain two integers")
            if type(row[0]) is not int or type(row[1]) is not int:
                raise TypeError(f"segments[{program}] must contain two integers")
            expected = (sequence, offset)
            if row != expected:
                raise ValueError(f"segments[{program}] must be {expected}, got {row}")
            program += 1
    return program_count


def glm53_segmented_conv_prefill(
    qkv_raw: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    state_indices: torch.Tensor,
    has_initial: torch.Tensor,
    segments: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    """Launch packed causal convolution into caller-owned FLA input storage.

    Call ``validate_glm53_segmented_conv_plan`` before asynchronously copying
    the descriptor values. This hot wrapper checks tensor metadata but does not
    synchronize to inspect descriptor contents. Warm this fixed specialization
    before its first timed execution.
    """
    programs = _validate_prefill_tensors(
        qkv_raw,
        conv_weight,
        conv_state,
        cu_seqlens,
        state_indices,
        has_initial,
        segments,
        out,
    )
    local_heads = out.shape[3]
    local_projection_size = local_heads * HEAD_DIM
    grid = (programs, 3 * local_projection_size // BLOCK_N)
    _glm53_segmented_causal_conv_kernel[grid](
        qkv_raw=qkv_raw,
        conv_weight=conv_weight,
        conv_state=conv_state,
        cu_seqlens=cu_seqlens,
        state_indices=state_indices,
        has_initial=has_initial,
        segments=segments,
        out=out,
        stride_raw_token=qkv_raw.stride(0),
        stride_state_page=conv_state.stride(0),
        stride_out_section=out.stride(0),
        stride_out_token=out.stride(2),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        CONV_KERNEL_SIZE=CONV_KERNEL_SIZE,
        LOCAL_PROJECTION_SIZE=local_projection_size,
        num_warps=4,
        num_stages=2,
    )
    return out


def _plain_int_tuple(name: str, values: Sequence[int]) -> tuple[int, ...]:
    result = tuple(values)
    for position, value in enumerate(result):
        if type(value) is not int:
            raise TypeError(f"{name}[{position}] must be an int")
    return result


def _validate_prefill_tensors(
    qkv_raw: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    state_indices: torch.Tensor,
    has_initial: torch.Tensor,
    segments: torch.Tensor,
    out: torch.Tensor,
) -> int:
    if qkv_raw.ndim != 2:
        raise ValueError(f"qkv_raw must have rank 2, got {qkv_raw.ndim}")
    tokens = qkv_raw.shape[0]
    if tokens < 1:
        raise ValueError("prefill token count must be positive")
    if conv_state.ndim != 3:
        raise ValueError(f"conv_state must have rank 3, got {conv_state.ndim}")
    pages = conv_state.shape[0]
    if pages < 1:
        raise ValueError("conv_state must contain at least one page")
    if state_indices.ndim != 1:
        raise ValueError(f"state_indices must have rank 1, got {state_indices.ndim}")
    batch = state_indices.shape[0]
    if batch < 1:
        raise ValueError("prefill batch must be positive")
    if segments.ndim != 2:
        raise ValueError(f"segments must have rank 2, got {segments.ndim}")
    programs = segments.shape[0]
    if programs < 1:
        raise ValueError("segments must contain at least one row")

    device = qkv_raw.device
    if device.type != "cuda":
        raise ValueError(f"qkv_raw must be on CUDA, got {device}")
    if out.ndim != 5:
        raise ValueError(f"out must have rank 5, got {out.ndim}")
    local_heads = out.shape[3]
    local_projection_size = local_heads * HEAD_DIM
    packed_qkv_size = 3 * local_projection_size
    _require_row_tensor(
        "qkv_raw", qkv_raw, (tokens, packed_qkv_size), torch.bfloat16, device
    )
    _require_tensor(
        "conv_weight",
        conv_weight,
        (packed_qkv_size, CONV_KERNEL_SIZE),
        torch.float32,
        device,
    )
    _require_tensor(
        "conv_state",
        conv_state,
        (pages, packed_qkv_size, CONV_KERNEL_SIZE - 1),
        torch.bfloat16,
        device,
    )
    _require_tensor("cu_seqlens", cu_seqlens, (batch + 1,), torch.int32, device)
    _require_tensor("state_indices", state_indices, (batch,), torch.int32, device)
    _require_tensor("has_initial", has_initial, (batch,), torch.bool, device)
    _require_tensor("segments", segments, (programs, 2), torch.int32, device)
    _require_tensor(
        "out",
        out,
        (3, 1, tokens, local_heads, HEAD_DIM),
        torch.bfloat16,
        device,
        contiguous=False,
    )
    if (
        out.stride(4) != 1
        or out.stride(3) != HEAD_DIM
        or out.stride(2) < local_projection_size
        or out.stride(0) < tokens * out.stride(2)
    ):
        raise ValueError("out must have contiguous tokens and non-overlapping sections")
    return programs


def _require_row_tensor(
    name: str,
    tensor: torch.Tensor,
    shape: tuple[int, int],
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    _require_tensor(name, tensor, shape, dtype, device, contiguous=False)
    if tensor.stride(1) != 1 or tensor.stride(0) < shape[1]:
        raise ValueError(
            f"{name} must be non-overlapping and contiguous within each row, "
            f"got stride {tuple(tensor.stride())}"
        )


def _require_tensor(
    name: str,
    tensor: torch.Tensor,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    *,
    contiguous: bool = True,
) -> None:
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")
    if tensor.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {tensor.dtype}")
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")
    if contiguous and not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


@triton.jit(do_not_specialize=["stride_out_section"])
def _glm53_segmented_causal_conv_kernel(
    qkv_raw,
    conv_weight,
    conv_state,
    cu_seqlens,
    state_indices,
    has_initial,
    segments,
    out,
    stride_raw_token: tl.constexpr,
    stride_state_page: tl.constexpr,
    stride_out_section,
    stride_out_token: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CONV_KERNEL_SIZE: tl.constexpr,
    LOCAL_PROJECTION_SIZE: tl.constexpr,
):
    program = tl.program_id(0)
    feature_block = tl.program_id(1)
    sequence = tl.load(segments + program * 2)
    token_offset = tl.load(segments + program * 2 + 1)
    sequence_start = tl.load(cu_seqlens + sequence)
    sequence_end = tl.load(cu_seqlens + sequence + 1)
    sequence_tokens = sequence_end - sequence_start
    segment_tokens = tl.minimum(BLOCK_M, sequence_tokens - token_offset)

    lanes = tl.arange(0, BLOCK_N)
    features = feature_block * BLOCK_N + lanes
    state_page = tl.load(state_indices + sequence).to(tl.int64)
    state_base = conv_state + state_page * stride_state_page + features * 3
    raw_base = qkv_raw + sequence_start * stride_raw_token + features

    if token_offset == 0:
        # A final-segment writer could race this warm-history read.
        load_initial = tl.load(has_initial + sequence).to(tl.int1)
        col0 = tl.load(state_base, mask=load_initial, other=0.0).to(tl.float32)
        col1 = tl.load(state_base + 1, mask=load_initial, other=0.0).to(tl.float32)
        col2 = tl.load(state_base + 2, mask=load_initial, other=0.0).to(tl.float32)
        for state_column in tl.static_range(CONV_KERNEL_SIZE - 1):
            source_token = sequence_tokens - (CONV_KERNEL_SIZE - 1) + state_column
            from_sequence = source_token >= 0
            from_initial = source_token < 0
            sequence_value = tl.load(
                raw_base + source_token * stride_raw_token,
                mask=from_sequence,
                other=0.0,
            )
            initial_value = tl.load(
                state_base + source_token + (CONV_KERNEL_SIZE - 1),
                mask=from_initial & load_initial,
                other=0.0,
            )
            value = tl.where(from_sequence, sequence_value, initial_value)
            tl.store(state_base + state_column, value)
    else:
        prior = raw_base + (token_offset - 3) * stride_raw_token
        col0 = tl.load(prior).to(tl.float32)
        col1 = tl.load(prior + stride_raw_token).to(tl.float32)
        col2 = tl.load(prior + 2 * stride_raw_token).to(tl.float32)

    weight_base = conv_weight + features * CONV_KERNEL_SIZE
    weight0 = tl.load(weight_base).to(tl.float32)
    weight1 = tl.load(weight_base + 1).to(tl.float32)
    weight2 = tl.load(weight_base + 2).to(tl.float32)
    weight3 = tl.load(weight_base + 3).to(tl.float32)
    section = feature_block // (LOCAL_PROJECTION_SIZE // BLOCK_N)
    section_features = (
        feature_block % (LOCAL_PROJECTION_SIZE // BLOCK_N) * BLOCK_N + lanes
    )
    out_base = out + section * stride_out_section + section_features

    for token in tl.static_range(BLOCK_M):
        active = token < segment_tokens
        value = tl.load(
            raw_base + (token_offset + token) * stride_raw_token,
            mask=active,
            other=0.0,
        ).to(tl.float32)
        activation = col0 * weight0 + col1 * weight1 + col2 * weight2 + value * weight3
        activation = activation / (1 + tl.exp(-activation))
        tl.store(
            out_base + (sequence_start + token_offset + token) * stride_out_token,
            activation,
            mask=active,
        )
        col0 = tl.where(active, col1, col0)
        col1 = tl.where(active, col2, col1)
        col2 = tl.where(active, value, col2)

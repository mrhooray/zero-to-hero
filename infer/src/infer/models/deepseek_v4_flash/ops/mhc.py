"""Fixed B1/B4/B16/B32 DeepSeek V4 post-to-pre mHC transition."""

import triton
import triton.language as tl

from infer.models.deepseek_v4_flash.model import (
    HIDDEN_SIZE,
    MHC_EPS,
    MHC_NORMALIZATION_ITERS,
    RMS_NORM_EPS,
)

_MIX_WIDTH = 24


def deepseek_v4_mhc_transition(
    hidden,
    residual,
    post,
    comb,
    fn,
    scale,
    base,
    norm_weight,
    residual_out,
    normalized_out,
    scratch,
) -> None:
    rows = residual.shape[0]
    if rows in (1, 4):
        splits, block_h, mix_tile = 8, 512, 2
    elif rows in (16, 32):
        splits, block_h, mix_tile = 4, 1_024, 4
    else:
        raise ValueError("fused DeepSeek V4 mHC requires exactly 1, 4, 16, or 32 rows")

    sq_offset = splits * rows * _MIX_WIDTH
    _post_pre_partials[(rows, _MIX_WIDTH // mix_tile, splits)](
        hidden,
        residual,
        post,
        comb,
        fn,
        residual_out,
        scratch,
        ROWS=rows,
        BLOCK_H=block_h,
        MIX_TILE=mix_tile,
        HIDDEN_SIZE=HIDDEN_SIZE,
        MIX_WIDTH=_MIX_WIDTH,
        SQ_OFFSET=sq_offset,
        num_warps=8,
    )
    _finish_pre_norm[(rows, 2)](
        residual_out,
        scale,
        base,
        norm_weight,
        post,
        comb,
        normalized_out,
        scratch,
        ROWS=rows,
        SPLITS=splits,
        HIDDEN_SIZE=HIDDEN_SIZE,
        MIX_WIDTH=_MIX_WIDTH,
        SQ_OFFSET=sq_offset,
        RMS_EPS=RMS_NORM_EPS,
        MHC_EPS=MHC_EPS,
        SINKHORN_ITERS=MHC_NORMALIZATION_ITERS,
        num_warps=8,
        num_stages=1,
    )


@triton.jit
def _post_pre_partials(
    hidden,
    residual,
    post,
    comb,
    fn,
    residual_out,
    scratch,
    ROWS: tl.constexpr,
    BLOCK_H: tl.constexpr,
    MIX_TILE: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    MIX_WIDTH: tl.constexpr,
    SQ_OFFSET: tl.constexpr,
):
    row = tl.program_id(0)
    mix_block = tl.program_id(1)
    split = tl.program_id(2)
    h = split * BLOCK_H + tl.arange(0, BLOCK_H)
    mixes = mix_block * MIX_TILE + tl.arange(0, MIX_TILE)

    hidden_values = tl.load(hidden + row * HIDDEN_SIZE + h).to(tl.float32)
    residual0 = tl.load(residual + row * 4 * HIDDEN_SIZE + h).to(tl.float32)
    residual1 = tl.load(residual + row * 4 * HIDDEN_SIZE + HIDDEN_SIZE + h).to(
        tl.float32
    )
    residual2 = tl.load(residual + row * 4 * HIDDEN_SIZE + 2 * HIDDEN_SIZE + h).to(
        tl.float32
    )
    residual3 = tl.load(residual + row * 4 * HIDDEN_SIZE + 3 * HIDDEN_SIZE + h).to(
        tl.float32
    )
    dot = tl.zeros([MIX_TILE], dtype=tl.float32)
    sqrsum = 0.0

    for route in range(4):
        values = hidden_values * tl.load(post + row * 4 + route)
        values = tl.fma(residual0, tl.load(comb + row * 16 + route), values)
        values = tl.fma(residual1, tl.load(comb + row * 16 + 4 + route), values)
        values = tl.fma(residual2, tl.load(comb + row * 16 + 8 + route), values)
        values = tl.fma(residual3, tl.load(comb + row * 16 + 12 + route), values)
        rounded = values.to(tl.bfloat16)
        values = rounded.to(tl.float32)
        tl.store(
            residual_out + row * 4 * HIDDEN_SIZE + route * HIDDEN_SIZE + h,
            rounded,
            mask=mix_block == 0,
        )
        weights = tl.load(
            fn + mixes[:, None] * (4 * HIDDEN_SIZE) + route * HIDDEN_SIZE + h[None, :]
        ).to(tl.float32)
        dot += tl.sum(weights * values[None, :], axis=1)
        sqrsum += tl.sum(values * values)

    tl.store(
        scratch + (split * ROWS + row) * MIX_WIDTH + mixes,
        dot,
    )
    tl.store(
        scratch + SQ_OFFSET + split * ROWS + row,
        sqrsum,
        mask=mix_block == 0,
    )


@triton.jit
def _finish_pre_norm(
    residual,
    scale,
    base,
    norm_weight,
    post,
    comb,
    normalized_out,
    scratch,
    ROWS: tl.constexpr,
    SPLITS: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    MIX_WIDTH: tl.constexpr,
    SQ_OFFSET: tl.constexpr,
    RMS_EPS: tl.constexpr,
    MHC_EPS: tl.constexpr,
    SINKHORN_ITERS: tl.constexpr,
):
    row = tl.program_id(0)
    task = tl.program_id(1)
    control_offsets = tl.arange(0, 4)
    comb_offsets = tl.arange(0, 16)
    pre_dot = tl.zeros([4], dtype=tl.float32)
    post_dot = tl.zeros([4], dtype=tl.float32)
    comb_dot = tl.zeros([16], dtype=tl.float32)
    sqrsum = 0.0

    for split in range(SPLITS):
        mix_base = scratch + (split * ROWS + row) * MIX_WIDTH
        pre_dot += tl.load(mix_base + control_offsets)
        post_dot += tl.load(mix_base + 4 + control_offsets)
        comb_dot += tl.load(mix_base + 8 + comb_offsets)
        sqrsum += tl.load(scratch + SQ_OFFSET + split * ROWS + row)

    rstd = tl.rsqrt(sqrsum / (4 * HIDDEN_SIZE) + RMS_EPS)
    pre_logits = pre_dot * rstd * tl.load(scale) + tl.load(base + control_offsets)
    pre = tl.sigmoid(pre_logits) + MHC_EPS

    if task == 0:
        h = tl.arange(0, HIDDEN_SIZE)
        collapsed = tl.zeros([HIDDEN_SIZE], dtype=tl.float32)
        for route in range(4):
            values = tl.load(
                residual + row * 4 * HIDDEN_SIZE + route * HIDDEN_SIZE + h
            ).to(tl.float32)
            pre_value = tl.sum(tl.where(control_offsets == route, pre, 0.0))
            collapsed = tl.fma(values, pre_value, collapsed)
        collapsed = collapsed.to(tl.bfloat16)
        collapsed = collapsed.to(tl.float32)
        norm_rstd = tl.rsqrt(tl.sum(collapsed * collapsed) / HIDDEN_SIZE + RMS_EPS)
        weight = tl.load(norm_weight + h).to(tl.float32)
        tl.store(
            normalized_out + row * HIDDEN_SIZE + h,
            (collapsed * norm_rstd * weight).to(tl.bfloat16),
        )
    else:
        post_logits = post_dot * rstd * tl.load(scale + 1) + tl.load(
            base + 4 + control_offsets
        )
        tl.store(post + row * 4 + control_offsets, 2.0 * tl.sigmoid(post_logits))
        comb_logits = comb_dot * rstd * tl.load(scale + 2) + tl.load(
            base + 8 + comb_offsets
        )
        normalized_comb = _sinkhorn4(comb_logits, MHC_EPS, SINKHORN_ITERS)
        tl.store(comb + row * 16 + comb_offsets, normalized_comb)


@triton.jit
def _sinkhorn4(values, eps: tl.constexpr, iterations: tl.constexpr):
    offsets = tl.arange(0, 16)
    rows = offsets // 4
    columns = offsets % 4
    row_max = _group_max4(values, rows)
    values = tl.exp(values - row_max)
    values = values / _group_sum4(values, rows) + eps
    values = values / (_group_sum4(values, columns) + eps)
    for _ in tl.range(1, iterations, loop_unroll_factor=1):
        values = values / (_group_sum4(values, rows) + eps)
        values = values / (_group_sum4(values, columns) + eps)
    return values


@triton.jit
def _group_sum4(values, groups):
    sum0 = tl.sum(tl.where(groups == 0, values, 0.0))
    sum1 = tl.sum(tl.where(groups == 1, values, 0.0))
    sum2 = tl.sum(tl.where(groups == 2, values, 0.0))
    sum3 = tl.sum(tl.where(groups == 3, values, 0.0))
    return tl.where(
        groups == 0,
        sum0,
        tl.where(groups == 1, sum1, tl.where(groups == 2, sum2, sum3)),
    )


@triton.jit
def _group_max4(values, groups):
    lowest = float("-inf")
    max0 = tl.max(tl.where(groups == 0, values, lowest))
    max1 = tl.max(tl.where(groups == 1, values, lowest))
    max2 = tl.max(tl.where(groups == 2, values, lowest))
    max3 = tl.max(tl.where(groups == 3, values, lowest))
    return tl.where(
        groups == 0,
        max0,
        tl.where(groups == 1, max1, tl.where(groups == 2, max2, max3)),
    )

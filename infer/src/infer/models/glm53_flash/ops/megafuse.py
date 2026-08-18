"""Fixed-shape GLM-5.3-Flash TP4 decode and target verify kernels.

The decode kernel is copied verbatim and the verify kernel is adapted from
TokenSpeed. See megafuse.LICENSE.txt for exact source identity,
attribution, and terms.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise

import torch
import triton
import triton.language as tl

from infer.models.glm53_flash.model import (
    CONV_KERNEL_SIZE,
    GATE_LOWER_BOUND,
    GLM53_TARGET_VERIFY_WIDTH,
    HEAD_DIM,
    RMS_NORM_EPS,
)

TOKENSPEED_COMMIT = "24ff5ab3dfd42aaea8f8f7476d4293f6c2717ed3"
TOKENSPEED_SOURCE_PATH = (
    "tokenspeed-kernel/python/tokenspeed_kernel/thirdparty/triton/fla_kda_recurrent.py"
)
TOKENSPEED_GIT_BLOB_SHA = "10e461ab01013ce6837804b4eea8d57b2b8ae7b1"
TOKENSPEED_SOURCE_SHA256 = (
    "e4ac7c46c534dc4403d3f7d204c05dd3ca0675ce6a835dfadb1a99d35d080f7f"
)


def validate_glm53_megafuse_plan(
    state_indices: Sequence[int],
    cu_seqlens: Sequence[int],
    *,
    state_pages: int,
) -> int:
    """Validate a CPU decode descriptor and return its active lane count.

    Active lanes must precede graph-padding lanes. Active cumulative-length
    deltas are one, padding deltas are zero, and padded indices are ``-1``.
    Active state indices are valid and unique because decode updates each page
    in place.
    """
    if type(state_pages) is not int:
        raise TypeError("state_pages must be an int")
    if state_pages < 1:
        raise ValueError("state_pages must be positive")

    indices = _plain_int_tuple("state_indices", state_indices)
    lengths = _plain_int_tuple("cu_seqlens", cu_seqlens)
    bucket = len(indices)
    if len(lengths) != bucket + 1:
        raise ValueError("cu_seqlens must have one more element than the indices")
    if lengths[0] != 0:
        raise ValueError("cu_seqlens must start at zero")

    active = 0
    saw_padding = False
    for lane, (start, end) in enumerate(pairwise(lengths)):
        delta = end - start
        if delta == 1 and not saw_padding:
            active += 1
        elif delta == 0:
            saw_padding = True
        elif delta == 1:
            raise ValueError(f"active lane {lane} follows graph padding")
        else:
            raise ValueError(f"cu_seqlens delta at lane {lane} must be zero or one")

    active_indices = indices[:active]
    for lane, index in enumerate(active_indices):
        if not 0 <= index < state_pages:
            raise ValueError(
                f"state_indices[{lane}] must be in [0, {state_pages}), got {index}"
            )
    if len(set(active_indices)) != active:
        raise ValueError("active state_indices must be unique")
    if any(index != -1 for index in indices[active:]):
        raise ValueError("padded state_indices must be -1")
    return active


def glm53_megafuse_decode(
    qkv_raw: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_state: torch.Tensor,
    f_a: torch.Tensor,
    f_b_weight: torch.Tensor,
    beta_logits: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    output_gate: torch.Tensor,
    norm_weight: torch.Tensor,
    recurrent_state: torch.Tensor,
    state_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    """Launch the fixed GLM TP4 decode into caller-owned storage.

    Call ``validate_glm53_megafuse_plan`` before copying CPU descriptors to
    these fixed GPU tensors. Warm the Triton specialization before CUDA graph
    capture. Only the active prefix of ``out`` is written; graph-padding rows
    retain their previous contents.
    """
    bucket = _validate_decode_tensors(
        qkv_raw,
        conv_weight,
        conv_state,
        f_a,
        f_b_weight,
        beta_logits,
        a_log,
        dt_bias,
        output_gate,
        norm_weight,
        recurrent_state,
        state_indices,
        cu_seqlens,
        out,
    )
    local_heads = a_log.shape[0]
    local_projection_size = local_heads * HEAD_DIM
    grid = (bucket * local_heads,)
    fused_recurrent_kda_megafuse_fwd_kernel[grid](
        qkv_raw=qkv_raw,
        conv_w=conv_weight,
        conv_pool=conv_state,
        f_a=f_a,
        w_fb=f_b_weight,
        beta=beta_logits,
        A_log=a_log,
        dt_bias=dt_bias,
        o=out,
        gate=output_gate,
        norm_w=norm_weight,
        norm_eps=RMS_NORM_EPS,
        h_pool=recurrent_state,
        read_indices=state_indices,
        write_indices=state_indices,
        cu_seqlens=cu_seqlens,
        lower_bound=GATE_LOWER_BOUND,
        stride_raw_tok=qkv_raw.stride(0),
        stride_fa_tok=f_a.stride(0),
        stride_beta_tok=beta_logits.stride(0),
        stride_gate_tok=output_gate.stride(0),
        scale=HEAD_DIM**-0.5,
        N=bucket,
        T=1,
        H=local_heads,
        HV=local_heads,
        K=HEAD_DIM,
        V=HEAD_DIM,
        P=local_projection_size,
        D_FA=HEAD_DIM,
        BK=HEAD_DIM,
        BV=HEAD_DIM,
        stride_state_page=recurrent_state.stride(0),
        stride_conv_page=conv_state.stride(0),
        num_warps=4,
        num_stages=2,
    )
    return out


def glm53_megafuse_verify(
    qkv_raw: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_state: torch.Tensor,
    conv_tape: torch.Tensor,
    f_a: torch.Tensor,
    f_b_weight: torch.Tensor,
    gate_raw: torch.Tensor,
    beta_logits: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    recurrent_state: torch.Tensor,
    recurrent_tape: torch.Tensor,
    state_indices: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    """Verify request-major groups of four rows into caller-owned tapes."""
    groups = _validate_verify_tensors(
        qkv_raw,
        conv_weight,
        conv_state,
        conv_tape,
        f_a,
        f_b_weight,
        gate_raw,
        beta_logits,
        a_log,
        dt_bias,
        recurrent_state,
        recurrent_tape,
        state_indices,
        out,
    )
    head_block = 32
    head_blocks = triton.cdiv(HEAD_DIM, head_block)
    local_heads = a_log.shape[0]
    local_projection_size = local_heads * HEAD_DIM
    gate_grid = (groups * local_heads * head_blocks,)
    glm53_verify_gate_precompute_kernel[gate_grid](
        f_a=f_a,
        w_fb=f_b_weight,
        gate_raw=gate_raw,
        stride_fa_tok=f_a.stride(0),
        stride_gate_tok=gate_raw.stride(0),
        T=GLM53_TARGET_VERIFY_WIDTH,
        H=local_heads,
        D_FA=HEAD_DIM,
        K=HEAD_DIM,
        BK=head_block,
        num_warps=4,
    )
    grid = (groups * head_blocks * local_heads,)
    fused_recurrent_kda_verify_megafuse_fwd_kernel[grid](
        qkv_raw=qkv_raw,
        conv_w=conv_weight,
        conv_pool=conv_state,
        conv_out=conv_tape,
        gate_raw=gate_raw,
        beta=beta_logits,
        A_log=a_log,
        dt_bias=dt_bias,
        o=out,
        h_pool=recurrent_state,
        h_pool_out=recurrent_tape,
        read_indices=state_indices,
        lower_bound=GATE_LOWER_BOUND,
        stride_raw_tok=qkv_raw.stride(0),
        stride_gate_tok=gate_raw.stride(0),
        stride_beta_tok=beta_logits.stride(0),
        scale=HEAD_DIM**-0.5,
        T=GLM53_TARGET_VERIFY_WIDTH,
        H=local_heads,
        HV=local_heads,
        K=HEAD_DIM,
        V=HEAD_DIM,
        P=local_projection_size,
        BK=HEAD_DIM,
        BV=head_block,
        stride_state_page=recurrent_state.stride(0),
        stride_state_out_page=recurrent_tape.stride(0),
        stride_conv_page=conv_state.stride(0),
        stride_conv_out_page=conv_tape.stride(0),
        num_warps=4,
        num_stages=2,
    )
    return out


def _plain_int_tuple(name: str, values: Sequence[int]) -> tuple[int, ...]:
    result = tuple(values)
    for lane, value in enumerate(result):
        if type(value) is not int:
            raise TypeError(f"{name}[{lane}] must be an int")
    return result


def _validate_decode_tensors(
    qkv_raw: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_state: torch.Tensor,
    f_a: torch.Tensor,
    f_b_weight: torch.Tensor,
    beta_logits: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    output_gate: torch.Tensor,
    norm_weight: torch.Tensor,
    recurrent_state: torch.Tensor,
    state_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    out: torch.Tensor,
) -> int:
    if qkv_raw.ndim != 2:
        raise ValueError(f"qkv_raw must have rank 2, got {qkv_raw.ndim}")
    bucket = qkv_raw.shape[0]
    if bucket < 1:
        raise ValueError("decode bucket must be positive")
    device = qkv_raw.device
    if device.type != "cuda":
        raise ValueError(f"qkv_raw must be on CUDA, got {device}")
    if recurrent_state.ndim != 4:
        raise ValueError(
            f"recurrent_state must have rank 4, got {recurrent_state.ndim}"
        )
    pages = recurrent_state.shape[0]
    if pages < 1:
        raise ValueError("state pools must contain at least one page")

    local_heads = a_log.shape[0]
    local_projection_size = local_heads * HEAD_DIM
    packed_qkv_size = 3 * local_projection_size
    _require_row_tensor(
        "qkv_raw", qkv_raw, (bucket, packed_qkv_size), torch.bfloat16, device
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
    _require_row_tensor("f_a", f_a, (bucket, HEAD_DIM), torch.bfloat16, device)
    _require_tensor(
        "f_b_weight",
        f_b_weight,
        (local_projection_size, HEAD_DIM),
        torch.bfloat16,
        device,
    )
    _require_row_tensor(
        "beta_logits", beta_logits, (bucket, local_heads), torch.bfloat16, device
    )
    _require_tensor("a_log", a_log, (local_heads,), torch.float32, device)
    _require_tensor("dt_bias", dt_bias, (local_projection_size,), torch.float32, device)
    _require_row_tensor(
        "output_gate",
        output_gate,
        (bucket, local_projection_size),
        torch.bfloat16,
        device,
    )
    _require_tensor("norm_weight", norm_weight, (HEAD_DIM,), torch.bfloat16, device)
    _require_tensor(
        "recurrent_state",
        recurrent_state,
        (pages, local_heads, HEAD_DIM, HEAD_DIM),
        torch.float32,
        device,
    )
    _require_tensor("state_indices", state_indices, (bucket,), torch.int32, device)
    _require_tensor("cu_seqlens", cu_seqlens, (bucket + 1,), torch.int32, device)
    _require_tensor("out", out, (bucket, local_heads, HEAD_DIM), torch.bfloat16, device)
    return bucket


def _validate_verify_tensors(
    qkv_raw: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_state: torch.Tensor,
    conv_tape: torch.Tensor,
    f_a: torch.Tensor,
    f_b_weight: torch.Tensor,
    gate_raw: torch.Tensor,
    beta_logits: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    recurrent_state: torch.Tensor,
    recurrent_tape: torch.Tensor,
    state_indices: torch.Tensor,
    out: torch.Tensor,
) -> int:
    width = GLM53_TARGET_VERIFY_WIDTH
    device = qkv_raw.device
    if device.type != "cuda":
        raise ValueError(f"qkv_raw must be on CUDA, got {device}")
    if state_indices.ndim != 1:
        raise ValueError(f"state_indices must have rank 1, got {state_indices.ndim}")
    groups = state_indices.shape[0]
    if groups < 1:
        raise ValueError("verify groups must be positive")
    rows = groups * width
    if recurrent_state.ndim != 4 or recurrent_state.shape[0] < 1:
        raise ValueError("recurrent_state must contain at least one state page")
    pages = recurrent_state.shape[0]
    local_heads = a_log.shape[0]
    local_projection_size = local_heads * HEAD_DIM
    packed_qkv_size = 3 * local_projection_size
    _require_row_tensor(
        "qkv_raw", qkv_raw, (rows, packed_qkv_size), torch.bfloat16, device
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
    _require_tensor(
        "conv_tape",
        conv_tape,
        (rows, packed_qkv_size, CONV_KERNEL_SIZE - 1),
        torch.bfloat16,
        device,
    )
    _require_row_tensor("f_a", f_a, (rows, HEAD_DIM), torch.bfloat16, device)
    _require_tensor(
        "f_b_weight",
        f_b_weight,
        (local_projection_size, HEAD_DIM),
        torch.bfloat16,
        device,
    )
    _require_row_tensor(
        "gate_raw", gate_raw, (rows, local_projection_size), torch.float32, device
    )
    _require_row_tensor(
        "beta_logits", beta_logits, (rows, local_heads), torch.bfloat16, device
    )
    _require_tensor("a_log", a_log, (local_heads,), torch.float32, device)
    _require_tensor("dt_bias", dt_bias, (local_projection_size,), torch.float32, device)
    _require_tensor(
        "recurrent_state",
        recurrent_state,
        (pages, local_heads, HEAD_DIM, HEAD_DIM),
        torch.float32,
        device,
    )
    _require_tensor(
        "recurrent_tape",
        recurrent_tape,
        (rows, local_heads, HEAD_DIM, HEAD_DIM),
        torch.float32,
        device,
    )
    _require_tensor("state_indices", state_indices, (groups,), torch.int32, device)
    _require_tensor("out", out, (rows, local_heads, HEAD_DIM), torch.bfloat16, device)
    return groups


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


@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        "HAS_DT_BIAS": lambda args: args["dt_bias"] is not None,
        "USE_LOWER_BOUND": lambda args: args["lower_bound"] is not None,
        "FUSE_OUTPUT_NORM": lambda args: args["gate"] is not None,
    }
)
@triton.jit(do_not_specialize=["N", "T"])
def fused_recurrent_kda_megafuse_fwd_kernel(
    qkv_raw,  # [T, 3*P] pre-conv packed projections (token-strided)
    conv_w,  # [3*P, 4] fused conv bank
    conv_pool,  # [pages, 3*P, 3] conv state
    f_a,  # [T, D_fa] low-rank gate input
    w_fb,  # [P, D_fa] f_b weight
    beta,
    A_log,
    dt_bias,
    o,
    gate,  # tokenspeed extension: [T, HV*V] raw output-gate logits, or None
    norm_w,  # tokenspeed extension: [V] gated-RMSNorm weight, or None
    norm_eps,
    h_pool,
    read_indices,
    write_indices,
    cu_seqlens,
    lower_bound,
    stride_raw_tok: tl.constexpr,
    stride_fa_tok: tl.constexpr,
    stride_beta_tok: tl.constexpr,
    stride_gate_tok: tl.constexpr,
    scale: tl.constexpr,
    N: tl.int64,
    T: tl.int64,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    P: tl.constexpr,  # proj_local = HV * K
    D_FA: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    stride_state_page: tl.constexpr,
    stride_conv_page: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    HAS_DT_BIAS: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
    FUSE_OUTPUT_NORM: tl.constexpr,
):
    pid = tl.program_id(0)
    NV = tl.cdiv(V, BV)
    i_v = pid % NV
    i_nh = pid // NV
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)
    if IS_VARLEN:
        bos = tl.load(cu_seqlens + i_n).to(tl.int64)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos = i_n * T
    if T == 0:
        return

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_v[:, None] & mask_k[None, :]
    b_read = tl.load(read_indices + i_n).to(tl.int64)
    b_write = tl.load(write_indices + i_n).to(tl.int64)
    read_ok = b_read >= 0

    # --- fused conv (4-tap depthwise + silu) over this program's features ---
    qf = i_h * K + o_k  # q features in section 0
    kf = P + i_h * K + o_k  # k features in section 1
    vf = 2 * P + i_hv * V + o_v  # v features in section 2

    x_q = tl.load(qkv_raw + bos * stride_raw_tok + qf, mask=mask_k, other=0.0).to(
        tl.float32
    )
    x_k = tl.load(qkv_raw + bos * stride_raw_tok + kf, mask=mask_k, other=0.0).to(
        tl.float32
    )
    x_v = tl.load(qkv_raw + bos * stride_raw_tok + vf, mask=mask_v, other=0.0).to(
        tl.float32
    )

    acc_q = x_q * tl.load(conv_w + qf * 4 + 3, mask=mask_k, other=0.0).to(tl.float32)
    acc_k = x_k * tl.load(conv_w + kf * 4 + 3, mask=mask_k, other=0.0).to(tl.float32)
    acc_v = x_v * tl.load(conv_w + vf * 4 + 3, mask=mask_v, other=0.0).to(tl.float32)
    s_q0 = tl.zeros([BK], dtype=tl.float32)
    s_k0 = tl.zeros([BK], dtype=tl.float32)
    s_v0 = tl.zeros([BV], dtype=tl.float32)
    s_q1 = tl.zeros([BK], dtype=tl.float32)
    s_k1 = tl.zeros([BK], dtype=tl.float32)
    s_v1 = tl.zeros([BV], dtype=tl.float32)
    s_q2 = tl.zeros([BK], dtype=tl.float32)
    s_k2 = tl.zeros([BK], dtype=tl.float32)
    s_v2 = tl.zeros([BV], dtype=tl.float32)
    if read_ok:
        cp = conv_pool + b_read * stride_conv_page
        s_q0 = tl.load(cp + qf * 3 + 0, mask=mask_k, other=0.0).to(tl.float32)
        s_q1 = tl.load(cp + qf * 3 + 1, mask=mask_k, other=0.0).to(tl.float32)
        s_q2 = tl.load(cp + qf * 3 + 2, mask=mask_k, other=0.0).to(tl.float32)
        s_k0 = tl.load(cp + kf * 3 + 0, mask=mask_k, other=0.0).to(tl.float32)
        s_k1 = tl.load(cp + kf * 3 + 1, mask=mask_k, other=0.0).to(tl.float32)
        s_k2 = tl.load(cp + kf * 3 + 2, mask=mask_k, other=0.0).to(tl.float32)
        s_v0 = tl.load(cp + vf * 3 + 0, mask=mask_v, other=0.0).to(tl.float32)
        s_v1 = tl.load(cp + vf * 3 + 1, mask=mask_v, other=0.0).to(tl.float32)
        s_v2 = tl.load(cp + vf * 3 + 2, mask=mask_v, other=0.0).to(tl.float32)
    acc_q += (
        s_q0 * tl.load(conv_w + qf * 4 + 0, mask=mask_k, other=0.0).to(tl.float32)
        + s_q1 * tl.load(conv_w + qf * 4 + 1, mask=mask_k, other=0.0).to(tl.float32)
        + s_q2 * tl.load(conv_w + qf * 4 + 2, mask=mask_k, other=0.0).to(tl.float32)
    )
    acc_k += (
        s_k0 * tl.load(conv_w + kf * 4 + 0, mask=mask_k, other=0.0).to(tl.float32)
        + s_k1 * tl.load(conv_w + kf * 4 + 1, mask=mask_k, other=0.0).to(tl.float32)
        + s_k2 * tl.load(conv_w + kf * 4 + 2, mask=mask_k, other=0.0).to(tl.float32)
    )
    acc_v += (
        s_v0 * tl.load(conv_w + vf * 4 + 0, mask=mask_v, other=0.0).to(tl.float32)
        + s_v1 * tl.load(conv_w + vf * 4 + 1, mask=mask_v, other=0.0).to(tl.float32)
        + s_v2 * tl.load(conv_w + vf * 4 + 2, mask=mask_v, other=0.0).to(tl.float32)
    )
    b_q = acc_q * tl.sigmoid(acc_q)  # silu
    b_k = acc_k * tl.sigmoid(acc_k)
    b_v = acc_v * tl.sigmoid(acc_v)

    # conv state update (shift window); q/k dupes across NV write same values.
    if b_write >= 0:
        cw = conv_pool + b_write * stride_conv_page
        tl.store(cw + qf * 3 + 0, s_q1.to(cw.dtype.element_ty), mask=mask_k)
        tl.store(cw + qf * 3 + 1, s_q2.to(cw.dtype.element_ty), mask=mask_k)
        tl.store(cw + qf * 3 + 2, x_q.to(cw.dtype.element_ty), mask=mask_k)
        tl.store(cw + kf * 3 + 0, s_k1.to(cw.dtype.element_ty), mask=mask_k)
        tl.store(cw + kf * 3 + 1, s_k2.to(cw.dtype.element_ty), mask=mask_k)
        tl.store(cw + kf * 3 + 2, x_k.to(cw.dtype.element_ty), mask=mask_k)
        tl.store(cw + vf * 3 + 0, s_v1.to(cw.dtype.element_ty), mask=mask_v)
        tl.store(cw + vf * 3 + 1, s_v2.to(cw.dtype.element_ty), mask=mask_v)
        tl.store(cw + vf * 3 + 2, x_v.to(cw.dtype.element_ty), mask=mask_v)

    # --- fused f_b: g_raw[c] = w_fb[c, :] . f_a for this head's K features ---
    o_fa = tl.arange(0, D_FA)
    fa = tl.load(f_a + bos * stride_fa_tok + o_fa).to(tl.float32)
    gc = i_hv * K + o_k  # gate feature = same head slice of P
    wfb = tl.load(
        w_fb + gc[:, None] * D_FA + o_fa[None, :],
        mask=mask_k[:, None],
        other=0.0,
    ).to(tl.float32)
    b_g = tl.sum(wfb * fa[None, :], axis=1)

    # --- recurrence (T=1 decode) ---
    b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
    b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
    b_q = b_q * scale

    b_A = tl.load(A_log + i_hv).to(tl.float32)
    if HAS_DT_BIAS:
        b_bias = tl.load(dt_bias + i_hv * K + o_k, mask=mask_k, other=0.0).to(
            tl.float32
        )
        b_g = b_g + b_bias
    if USE_LOWER_BOUND:
        b_gk = lower_bound * tl.sigmoid(tl.exp(b_A) * b_g)
    else:
        b_gk = -tl.exp(b_A) * tl.where(
            b_g < 20.0, tl.math.log(1 + tl.math.exp(b_g)), b_g
        )

    p_h0 = h_pool + b_read * stride_state_page + i_hv * K * V
    b_h = tl.zeros([BV, BK], dtype=tl.float32)
    p_h0 += o_v[:, None] * K + o_k[None, :]
    b_h += tl.load(p_h0, mask=mask_h & read_ok, other=0.0).to(tl.float32)

    b_h *= tl.exp(b_gk)[None, :]
    b_v = b_v - tl.sum(b_h * b_k[None, :], axis=1)
    b_beta = tl.load(beta + bos * stride_beta_tok + i_hv).to(tl.float32)
    b_beta = tl.sigmoid(b_beta)
    b_v *= b_beta
    b_h += b_v[:, None] * b_k[None, :]
    b_o = tl.sum(b_h * b_q[None, :], axis=1)
    # tokenspeed extension: NV == 1, so b_o is the whole row the norm would reload.
    if FUSE_OUTPUT_NORM:
        b_rsig = tl.math.rsqrt(tl.sum(b_o * b_o) / V + norm_eps)
        b_nw = tl.load(norm_w + o_v, mask=mask_v, other=0.0).to(tl.float32)
        b_gate = tl.load(
            gate + bos * stride_gate_tok + i_hv * V + o_v, mask=mask_v, other=0.0
        ).to(tl.float32)
        b_o = b_o * b_rsig * b_nw * tl.sigmoid(b_gate)
    tl.store(
        o + (bos * HV + i_hv) * V + o_v,
        b_o.to(o.dtype.element_ty),
        mask=mask_v,
    )
    p_ht = h_pool + b_write * stride_state_page + i_hv * K * V
    p_ht += o_v[:, None] * K + o_k[None, :]
    tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h & (b_write >= 0))


@triton.jit
def glm53_verify_gate_precompute_kernel(
    f_a,
    w_fb,
    gate_raw,
    stride_fa_tok: tl.constexpr,
    stride_gate_tok: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    D_FA: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
):
    pid = tl.program_id(0)
    NK = tl.cdiv(K, BK)
    i_k = pid % NK
    i_nh = pid // NK
    i_n, i_hv = i_nh // H, i_nh % H
    o_k = i_k * BK + tl.arange(0, BK)
    o_fa = tl.arange(0, D_FA)
    mask_k = o_k < K
    gc = i_hv * K + o_k
    wfb = tl.load(
        w_fb + gc[:, None] * D_FA + o_fa[None, :],
        mask=mask_k[:, None],
        other=0.0,
    ).to(tl.float32)
    for i_t in range(T):
        row = i_n * T + i_t
        fa = tl.load(f_a + row * stride_fa_tok + o_fa).to(tl.float32)
        b_g = tl.sum(wfb * fa[None, :], axis=1)
        tl.store(gate_raw + row * stride_gate_tok + gc, b_g, mask=mask_k)


@triton.jit
def fused_recurrent_kda_verify_megafuse_fwd_kernel(
    qkv_raw,
    conv_w,
    conv_pool,
    conv_out,
    gate_raw,
    beta,
    A_log,
    dt_bias,
    o,
    h_pool,
    h_pool_out,
    read_indices,
    lower_bound,
    stride_raw_tok: tl.constexpr,
    stride_gate_tok: tl.constexpr,
    stride_beta_tok: tl.constexpr,
    scale: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    P: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    stride_state_page: tl.constexpr,
    stride_state_out_page: tl.constexpr,
    stride_conv_page: tl.constexpr,
    stride_conv_out_page: tl.constexpr,
):
    pid = tl.program_id(0)
    NV = tl.cdiv(V, BV)
    i_v = pid % NV
    i_nh = pid // NV
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)
    b_read = tl.load(read_indices + i_n).to(tl.int64)
    if b_read < 0:
        return
    bos = i_n * T

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_v[:, None] & mask_k[None, :]

    qf = i_h * K + o_k
    kf = P + i_h * K + o_k
    vf = 2 * P + i_hv * V + o_v
    w_q0 = tl.load(conv_w + qf * 4 + 0, mask=mask_k, other=0.0).to(tl.float32)
    w_q1 = tl.load(conv_w + qf * 4 + 1, mask=mask_k, other=0.0).to(tl.float32)
    w_q2 = tl.load(conv_w + qf * 4 + 2, mask=mask_k, other=0.0).to(tl.float32)
    w_q3 = tl.load(conv_w + qf * 4 + 3, mask=mask_k, other=0.0).to(tl.float32)
    w_k0 = tl.load(conv_w + kf * 4 + 0, mask=mask_k, other=0.0).to(tl.float32)
    w_k1 = tl.load(conv_w + kf * 4 + 1, mask=mask_k, other=0.0).to(tl.float32)
    w_k2 = tl.load(conv_w + kf * 4 + 2, mask=mask_k, other=0.0).to(tl.float32)
    w_k3 = tl.load(conv_w + kf * 4 + 3, mask=mask_k, other=0.0).to(tl.float32)
    w_v0 = tl.load(conv_w + vf * 4 + 0, mask=mask_v, other=0.0).to(tl.float32)
    w_v1 = tl.load(conv_w + vf * 4 + 1, mask=mask_v, other=0.0).to(tl.float32)
    w_v2 = tl.load(conv_w + vf * 4 + 2, mask=mask_v, other=0.0).to(tl.float32)
    w_v3 = tl.load(conv_w + vf * 4 + 3, mask=mask_v, other=0.0).to(tl.float32)
    s_q0 = tl.zeros([BK], dtype=tl.float32)
    s_q1 = tl.zeros([BK], dtype=tl.float32)
    s_q2 = tl.zeros([BK], dtype=tl.float32)
    s_k0 = tl.zeros([BK], dtype=tl.float32)
    s_k1 = tl.zeros([BK], dtype=tl.float32)
    s_k2 = tl.zeros([BK], dtype=tl.float32)
    s_v0 = tl.zeros([BV], dtype=tl.float32)
    s_v1 = tl.zeros([BV], dtype=tl.float32)
    s_v2 = tl.zeros([BV], dtype=tl.float32)
    cp = conv_pool + b_read * stride_conv_page
    s_q0 = tl.load(cp + qf * 3 + 0, mask=mask_k, other=0.0).to(tl.float32)
    s_q1 = tl.load(cp + qf * 3 + 1, mask=mask_k, other=0.0).to(tl.float32)
    s_q2 = tl.load(cp + qf * 3 + 2, mask=mask_k, other=0.0).to(tl.float32)
    s_k0 = tl.load(cp + kf * 3 + 0, mask=mask_k, other=0.0).to(tl.float32)
    s_k1 = tl.load(cp + kf * 3 + 1, mask=mask_k, other=0.0).to(tl.float32)
    s_k2 = tl.load(cp + kf * 3 + 2, mask=mask_k, other=0.0).to(tl.float32)
    s_v0 = tl.load(cp + vf * 3 + 0, mask=mask_v, other=0.0).to(tl.float32)
    s_v1 = tl.load(cp + vf * 3 + 1, mask=mask_v, other=0.0).to(tl.float32)
    s_v2 = tl.load(cp + vf * 3 + 2, mask=mask_v, other=0.0).to(tl.float32)

    p_h0 = h_pool + b_read * stride_state_page + i_hv * K * V
    p_h0 += o_v[:, None] * K + o_k[None, :]
    b_h = tl.load(p_h0, mask=mask_h, other=0.0).to(tl.float32)
    b_A = tl.load(A_log + i_hv).to(tl.float32)
    gc = i_hv * K + o_k
    b_bias = tl.load(dt_bias + i_hv * K + o_k, mask=mask_k, other=0.0).to(tl.float32)

    for i_t in range(T):
        row = bos + i_t
        raw = qkv_raw + row * stride_raw_tok
        x_q = tl.load(raw + qf, mask=mask_k, other=0.0).to(tl.float32)
        x_k = tl.load(raw + kf, mask=mask_k, other=0.0).to(tl.float32)
        x_v = tl.load(raw + vf, mask=mask_v, other=0.0).to(tl.float32)
        acc_q = x_q * w_q3 + s_q0 * w_q0 + s_q1 * w_q1 + s_q2 * w_q2
        acc_k = x_k * w_k3 + s_k0 * w_k0 + s_k1 * w_k1 + s_k2 * w_k2
        acc_v = x_v * w_v3 + s_v0 * w_v0 + s_v1 * w_v1 + s_v2 * w_v2
        b_q = acc_q * tl.sigmoid(acc_q)
        b_k = acc_k * tl.sigmoid(acc_k)
        b_v = acc_v * tl.sigmoid(acc_v)
        s_q0, s_q1, s_q2 = s_q1, s_q2, x_q
        s_k0, s_k1, s_k2 = s_k1, s_k2, x_k
        s_v0, s_v1, s_v2 = s_v1, s_v2, x_v

        cw = conv_out + row * stride_conv_out_page
        tl.store(cw + qf * 3 + 0, s_q0.to(cw.dtype.element_ty), mask=mask_k)
        tl.store(cw + qf * 3 + 1, s_q1.to(cw.dtype.element_ty), mask=mask_k)
        tl.store(cw + qf * 3 + 2, s_q2.to(cw.dtype.element_ty), mask=mask_k)
        tl.store(cw + kf * 3 + 0, s_k0.to(cw.dtype.element_ty), mask=mask_k)
        tl.store(cw + kf * 3 + 1, s_k1.to(cw.dtype.element_ty), mask=mask_k)
        tl.store(cw + kf * 3 + 2, s_k2.to(cw.dtype.element_ty), mask=mask_k)
        tl.store(cw + vf * 3 + 0, s_v0.to(cw.dtype.element_ty), mask=mask_v)
        tl.store(cw + vf * 3 + 1, s_v1.to(cw.dtype.element_ty), mask=mask_v)
        tl.store(cw + vf * 3 + 2, s_v2.to(cw.dtype.element_ty), mask=mask_v)

        b_g = (
            tl.load(
                gate_raw + row * stride_gate_tok + gc, mask=mask_k, other=0.0
            ).to(tl.float32)
            + b_bias
        )
        b_gk = lower_bound * tl.sigmoid(tl.exp(b_A) * b_g)
        b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
        b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
        b_q = b_q * scale

        b_h *= tl.exp(b_gk)[None, :]
        b_v = b_v - tl.sum(b_h * b_k[None, :], axis=1)
        b_beta = tl.sigmoid(tl.load(beta + row * stride_beta_tok + i_hv).to(tl.float32))
        b_v *= b_beta
        b_h += b_v[:, None] * b_k[None, :]
        b_o = tl.sum(b_h * b_q[None, :], axis=1)
        tl.store(
            o + (row * HV + i_hv) * V + o_v,
            b_o.to(o.dtype.element_ty),
            mask=mask_v,
        )

        p_ht = h_pool_out + row * stride_state_out_page + i_hv * K * V
        p_ht += o_v[:, None] * K + o_k[None, :]
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)

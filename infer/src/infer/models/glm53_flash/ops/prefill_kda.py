"""Pinned FlashKDA packed-prefill path for GLM 5.3 TP4."""

from __future__ import annotations

from collections.abc import Sequence
from importlib.metadata import version
from itertools import pairwise

import torch
from flash_kda_C import fwd as _flash_kda_fwd
from flash_kda_C import get_workspace_size as _get_workspace_size

from infer.models.glm53_flash.model import (
    FLASH_KDA_WORKSPACE_ALIGNMENT,
    GATE_LOWER_BOUND,
    HEAD_DIM,
    NUM_HEADS,
    TP_SIZE,
    glm53_local_heads,
    kda_prefill_kernel_workspace_bytes,
)

FLASH_KDA_VERSION = "0.0.1+1ce47ea.infer1"

if version("flash_kda") != FLASH_KDA_VERSION:
    raise RuntimeError(f"flash_kda must be {FLASH_KDA_VERSION}")


class GLM53PrefillMetadata:
    """Immutable, content-validated descriptors for one scheduler plan.

    Cumulative lengths are validated on the CPU before either GPU descriptor
    is created. The int32 descriptor serves the segmented convolution and the
    int64 descriptor serves the pinned FlashKDA raw ABI.
    """

    __slots__ = (
        "_batch_size",
        "_cpu_version",
        "_cu_seqlens",
        "_cu_seqlens_cpu",
        "_cu_seqlens_int64",
        "_cuda_int32_version",
        "_cuda_int64_version",
        "_total_tokens",
    )

    def __init__(
        self,
        cu_seqlens: Sequence[int],
        device: torch.device | str,
    ) -> None:
        values = _validate_cumulative_lengths(cu_seqlens)
        cuda_device = torch.device(device)
        if cuda_device.type != "cuda":
            raise ValueError(f"device must be CUDA, got {cuda_device}")

        with torch.inference_mode(False):
            cu_seqlens_cpu = torch.tensor(values, dtype=torch.int64, device="cpu")
            cu_seqlens_int32 = cu_seqlens_cpu.to(
                device=cuda_device,
                dtype=torch.int32,
            )
            cu_seqlens_int64 = cu_seqlens_cpu.to(
                device=cuda_device,
                dtype=torch.int64,
            )

        self._bind(cu_seqlens_cpu, cu_seqlens_int32, cu_seqlens_int64, values[-1])

    @classmethod
    def stage(
        cls,
        cu_seqlens: Sequence[int],
        cpu: torch.Tensor,
        cuda_int32: torch.Tensor,
        cuda_int64: torch.Tensor,
    ) -> GLM53PrefillMetadata:
        """Stage one receipt into caller-owned maximum-capacity descriptors."""

        values = _validate_cumulative_lengths(cu_seqlens)
        size = len(values)
        tensors = (
            ("cpu", cpu, torch.int64, "cpu"),
            ("cuda_int32", cuda_int32, torch.int32, "cuda"),
            ("cuda_int64", cuda_int64, torch.int64, "cuda"),
        )
        for name, tensor, dtype, device_type in tensors:
            if tensor.ndim != 1 or tensor.shape[0] < size or tensor.dtype != dtype:
                raise ValueError(f"{name} cannot hold {size} cumulative lengths")
            if tensor.device.type != device_type or not tensor.is_contiguous():
                raise ValueError(f"{name} has the wrong device or layout")
        if not cpu.is_pinned():
            raise ValueError("cpu cumulative lengths must be pinned")
        if cuda_int32.device != cuda_int64.device:
            raise ValueError("CUDA cumulative lengths must share a device")

        cpu = cpu[:size]
        cuda_int32 = cuda_int32[:size]
        cuda_int64 = cuda_int64[:size]
        for index, value in enumerate(values):
            cpu[index] = value
        cuda_int32.copy_(cpu, non_blocking=True)
        cuda_int64.copy_(cpu, non_blocking=True)
        metadata = cls.__new__(cls)
        metadata._bind(cpu, cuda_int32, cuda_int64, values[-1])
        return metadata

    def _bind(
        self,
        cpu: torch.Tensor,
        cuda_int32: torch.Tensor,
        cuda_int64: torch.Tensor,
        total_tokens: int,
    ) -> None:
        self._cu_seqlens = cuda_int32
        self._cu_seqlens_int64 = cuda_int64
        self._cu_seqlens_cpu = cpu
        self._batch_size = cpu.shape[0] - 1
        self._total_tokens = total_tokens
        self._cuda_int32_version = cuda_int32._version
        self._cuda_int64_version = cuda_int64._version
        self._cpu_version = cpu._version

    @property
    def cu_seqlens(self) -> torch.Tensor:
        return self._cu_seqlens

    @property
    def cu_seqlens_int64(self) -> torch.Tensor:
        return self._cu_seqlens_int64

    @property
    def cu_seqlens_cpu(self) -> torch.Tensor:
        return self._cu_seqlens_cpu

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def total_tokens(self) -> int:
        return self._total_tokens

    def validate_unchanged(self) -> None:
        if (
            self._cu_seqlens._version != self._cuda_int32_version
            or self._cu_seqlens_int64._version != self._cuda_int64_version
            or self._cu_seqlens_cpu._version != self._cpu_version
        ):
            raise RuntimeError(
                "GLM53PrefillMetadata descriptors were modified; create a new "
                "instance for each changed plan"
            )


def glm53_prefill_kda(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta_logits: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    initial_state: torch.Tensor,
    metadata: GLM53PrefillMetadata,
    out: torch.Tensor,
    final_state: torch.Tensor,
    workspace: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the direct raw ABI from the pinned patched FlashKDA build.

    The caller owns output, final state, and opaque workspace, including the
    transposed BF16 beta tail used by the patched C++ wrapper.
    """
    metadata.validate_unchanged()
    _validate_inputs(
        q,
        k,
        v,
        gate,
        beta_logits,
        a_log,
        dt_bias,
        initial_state,
        out,
        final_state,
        metadata,
    )
    required_workspace = kda_prefill_kernel_workspace_bytes(
        metadata.total_tokens,
        metadata.batch_size,
        NUM_HEADS // a_log.shape[0],
    )
    _require_workspace(workspace, required_workspace, q.device)

    _flash_kda_fwd(
        q,
        k,
        v,
        gate,
        beta_logits,
        HEAD_DIM**-0.5,
        out,
        workspace,
        a_log,
        dt_bias.view(a_log.shape[0], HEAD_DIM),
        GATE_LOWER_BOUND,
        initial_state=initial_state,
        final_state=final_state,
        cu_seqlens=metadata.cu_seqlens_int64,
    )
    return out, final_state


def make_glm53_prefill_kda_workspace(
    storage: torch.Tensor,
    total_tokens: int,
    batch_size: int,
    attention_tp_size: int = TP_SIZE,
) -> torch.Tensor:
    """Check the pinned ABI once and return a 128-byte-aligned arena view."""
    required_workspace = kda_prefill_kernel_workspace_bytes(
        total_tokens,
        batch_size,
        attention_tp_size,
    )
    local_heads = glm53_local_heads(attention_tp_size)
    extension_workspace = _get_workspace_size(
        total_tokens,
        local_heads,
        batch_size,
    )
    if extension_workspace != required_workspace:
        raise RuntimeError(
            "FlashKDA workspace ABI differs from the pinned GLM shape contract: "
            f"{extension_workspace} != {required_workspace}"
        )
    if storage.ndim != 1:
        raise ValueError(f"workspace storage must have rank 1, got {storage.ndim}")
    if storage.dtype != torch.uint8:
        raise TypeError(
            f"workspace storage must have dtype {torch.uint8}, got {storage.dtype}"
        )
    if storage.device.type != "cuda":
        raise ValueError(f"workspace storage must be on CUDA, got {storage.device}")
    if not storage.is_contiguous():
        raise ValueError("workspace storage must be contiguous")
    alignment_slack = FLASH_KDA_WORKSPACE_ALIGNMENT - 1
    if storage.numel() < required_workspace + alignment_slack:
        raise ValueError(
            "workspace storage must contain the required bytes plus 127-byte "
            f"alignment slack, got {storage.numel()}"
        )

    offset = (-storage.data_ptr()) % FLASH_KDA_WORKSPACE_ALIGNMENT
    workspace = storage[offset : offset + required_workspace]
    _require_workspace(workspace, required_workspace, storage.device)
    return workspace


def _validate_cumulative_lengths(cu_seqlens: Sequence[int]) -> tuple[int, ...]:
    try:
        values = tuple(cu_seqlens)
    except TypeError as error:
        raise TypeError("cu_seqlens must be a sequence of integers") from error

    if len(values) < 2:
        raise ValueError("cu_seqlens must describe at least one sequence")
    for index, value in enumerate(values):
        if type(value) is not int:
            raise TypeError(f"cu_seqlens[{index}] must be an int")
    if values[0] != 0:
        raise ValueError("cu_seqlens must start at zero")
    if any(end <= start for start, end in pairwise(values)):
        raise ValueError("each packed sequence must contain at least one token")
    if values[-1] > torch.iinfo(torch.int32).max:
        raise ValueError("total packed tokens exceed int32 range")
    return values


def _validate_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta_logits: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    initial_state: torch.Tensor,
    out: torch.Tensor,
    final_state: torch.Tensor,
    metadata: GLM53PrefillMetadata,
) -> None:
    device = q.device
    local_heads = a_log.shape[0]
    local_projection_size = local_heads * HEAD_DIM
    token_shape = (1, metadata.total_tokens, local_heads, HEAD_DIM)
    state_shape = (metadata.batch_size, local_heads, HEAD_DIM, HEAD_DIM)
    tensors = (
        ("q", q, token_shape, torch.bfloat16),
        ("k", k, token_shape, torch.bfloat16),
        ("v", v, token_shape, torch.bfloat16),
        ("gate", gate, token_shape, torch.bfloat16),
        (
            "beta_logits",
            beta_logits,
            (1, metadata.total_tokens, local_heads),
            torch.bfloat16,
        ),
        ("a_log", a_log, (local_heads,), torch.float32),
        ("dt_bias", dt_bias, (local_projection_size,), torch.float32),
        ("initial_state", initial_state, state_shape, torch.float32),
        (
            "cu_seqlens",
            metadata.cu_seqlens,
            (metadata.batch_size + 1,),
            torch.int32,
        ),
        (
            "cu_seqlens_int64",
            metadata.cu_seqlens_int64,
            (metadata.batch_size + 1,),
            torch.int64,
        ),
        ("out", out, token_shape, torch.bfloat16),
        ("final_state", final_state, state_shape, torch.float32),
    )
    for name, tensor, shape, dtype in tensors:
        _require_tensor(name, tensor, shape, dtype, device)

    if device.type != "cuda":
        raise ValueError(f"q must be on CUDA, got {device}")
    if metadata.cu_seqlens_int64.device != device:
        raise ValueError("metadata and KDA inputs must be on the same CUDA device")


def _require_tensor(
    name: str,
    tensor: torch.Tensor,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")
    if tensor.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {tensor.dtype}")
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _require_workspace(
    workspace: torch.Tensor,
    required_bytes: int,
    device: torch.device,
) -> None:
    if workspace.ndim != 1:
        raise ValueError(f"workspace must have rank 1, got {workspace.ndim}")
    if workspace.dtype != torch.uint8:
        raise TypeError(
            f"workspace must have dtype {torch.uint8}, got {workspace.dtype}"
        )
    if workspace.device != device:
        raise ValueError(f"workspace must be on {device}, got {workspace.device}")
    if not workspace.is_contiguous():
        raise ValueError("workspace must be contiguous")
    if workspace.data_ptr() % FLASH_KDA_WORKSPACE_ALIGNMENT:
        raise ValueError("workspace must be 128-byte aligned")
    if workspace.numel() < required_bytes:
        raise ValueError(
            f"workspace must contain at least {required_bytes} bytes, "
            f"got {workspace.numel()}"
        )

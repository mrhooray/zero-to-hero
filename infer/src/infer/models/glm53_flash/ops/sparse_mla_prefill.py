"""GLM-5.3-Flash sparse-MLA history append for packed prefill."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from infer.models.glm53_flash.model import (
    KDA_CHUNK_SIZE,
    SPARSE_MLA_COMPOUND_BLOCKS,
    SPARSE_MLA_HISTORY_BLOCK_TOKENS,
    SPARSE_MLA_INDEX_PAGE_BYTES,
    SPARSE_MLA_INDEX_POOL_TOKENS,
    SPARSE_MLA_INDEXER_HEAD_DIM,
    SPARSE_MLA_KV_LORA_RANK,
    SPARSE_MLA_MAX_CONTEXT_TOKENS,
    SparseMLAHistory,
)


@dataclass(frozen=True, slots=True)
class GLM53SparseMLAPrefillPlan:
    start_token: int
    total_tokens: int
    state_slot: int
    history_blocks: int
    live_slots: int
    has_initial: bool


@dataclass(eq=False, frozen=True, slots=True)
class GLM53StagedSparseMLAPrefillBatch:
    """Packed prefill receipt over one fixed-capacity descriptor arena."""

    total_tokens: int
    history_blocks: int
    live_slots: int
    cu_seqlens: torch.Tensor
    start_tokens: torch.Tensor
    state_slots: torch.Tensor
    sequence_ids: torch.Tensor
    active: torch.Tensor
    raw_lengths: torch.Tensor
    block_table: torch.Tensor
    null_token: torch.Tensor
    _versions: tuple[int, ...]

    @property
    def batch_size(self) -> int:
        return self.state_slots.shape[0]

    def validate_unchanged(self) -> None:
        tensors = (
            self.cu_seqlens,
            self.start_tokens,
            self.state_slots,
            self.sequence_ids,
            self.active,
            self.raw_lengths,
            self.block_table,
            self.null_token,
        )
        if any(
            tensor._version != version
            for tensor, version in zip(tensors, self._versions, strict=True)
        ):
            raise RuntimeError(
                "GLM packed sparse MLA descriptors changed after staging"
            )


def validate_glm53_sparse_mla_prefill_plan(
    start_token: int,
    total_tokens: int,
    state_slot: int,
    block_table: Sequence[int] | None,
    *,
    history_blocks: int,
    live_slots: int,
    has_initial: bool,
) -> GLM53SparseMLAPrefillPlan:
    """Validate and freeze one B=1 sparse-MLA append plan."""

    _require_plain_int("start_token", start_token)
    _require_plain_int("total_tokens", total_tokens)
    _require_plain_int("state_slot", state_slot)
    _require_plain_int("history_blocks", history_blocks)
    _require_plain_int("live_slots", live_slots)
    if start_token < 0:
        raise ValueError("start_token must be non-negative")
    if total_tokens < 1:
        raise ValueError("total_tokens must be positive")
    if start_token + total_tokens > SPARSE_MLA_MAX_CONTEXT_TOKENS:
        raise ValueError("sparse MLA prefill exceeds the model context")
    if history_blocks < 2:
        raise ValueError("history_blocks must include null and live blocks")
    if live_slots < 1:
        raise ValueError("live_slots must be positive")
    if not 0 <= state_slot < live_slots:
        raise ValueError("state_slot is outside the live state pool")
    if type(has_initial) is not bool:
        raise TypeError("has_initial must be a bool")
    if has_initial != (start_token > 0):
        raise ValueError("only resumed prefill may consume initial sparse MLA state")

    required_blocks = (
        start_token + total_tokens + SPARSE_MLA_HISTORY_BLOCK_TOKENS - 1
    ) // SPARSE_MLA_HISTORY_BLOCK_TOKENS
    if block_table is None:
        if required_blocks >= history_blocks:
            raise ValueError("sparse MLA prefill exceeds the physical history pool")
        blocks = None
    else:
        blocks = tuple(block_table)
        if len(blocks) != required_blocks:
            raise ValueError(
                f"block_table must contain {required_blocks} live mappings, "
                f"got {len(blocks)}"
            )
        for index, block in enumerate(blocks):
            if type(block) is not int:
                raise TypeError(f"block_table[{index}] must be an int")
            if not 0 <= block < history_blocks - 1:
                raise ValueError(
                    f"block_table[{index}] must be in [0, {history_blocks - 1}), "
                    f"got {block}"
                )
        if len(set(blocks)) != len(blocks):
            raise ValueError("live sparse MLA blocks must be unique")

    return GLM53SparseMLAPrefillPlan(
        start_token,
        total_tokens,
        state_slot,
        history_blocks,
        live_slots,
        has_initial,
    )


def stage_glm53_sparse_mla_prefill_batch(
    *,
    plans: Sequence[GLM53SparseMLAPrefillPlan],
    cu_seqlens: torch.Tensor,
    start_tokens: torch.Tensor,
    state_slots: torch.Tensor,
    sequence_ids: torch.Tensor,
    token_state_slots: torch.Tensor,
    active: torch.Tensor,
    raw_lengths: torch.Tensor,
    execution_tables: torch.Tensor,
    block_table: torch.Tensor,
    null_token: torch.Tensor,
) -> GLM53StagedSparseMLAPrefillBatch:
    plans = tuple(plans)
    if not plans:
        raise ValueError("packed sparse MLA prefill requires at least one sequence")
    total_tokens = sum(plan.total_tokens for plan in plans)
    batch_size = len(plans)
    history_blocks = plans[0].history_blocks
    live_slots = plans[0].live_slots
    capacity = active.shape[0] if active.ndim == 1 else 0
    if not 1 <= capacity <= KDA_CHUNK_SIZE or total_tokens > capacity:
        raise ValueError("packed sparse MLA prefill exceeds its token arena")
    if any(
        plan.history_blocks != history_blocks or plan.live_slots != live_slots
        for plan in plans
    ):
        raise ValueError("packed sparse MLA plans must share one state pool")

    device = execution_tables.device
    specifications = (
        (cu_seqlens, torch.int32, (batch_size + 1,)),
        (start_tokens, torch.int32, (batch_size,)),
        (state_slots, torch.int64, (batch_size,)),
        (sequence_ids, torch.int32, (capacity,)),
        (token_state_slots, torch.int64, (capacity,)),
        (active, torch.uint8, (capacity,)),
        (raw_lengths, torch.int32, (capacity,)),
        (execution_tables, torch.int32, (live_slots, SPARSE_MLA_COMPOUND_BLOCKS)),
        (
            block_table,
            torch.int32,
            (capacity, SPARSE_MLA_COMPOUND_BLOCKS),
        ),
        (null_token, torch.int32, (1,)),
    )
    for tensor, dtype, shape in specifications:
        if tensor.shape != shape or tensor.dtype != dtype:
            raise ValueError("invalid packed sparse MLA descriptor shape or dtype")
        if tensor.device != device or not tensor.is_cuda or not tensor.is_contiguous():
            raise ValueError("invalid packed sparse MLA descriptor device or layout")
    torch.index_select(execution_tables, 0, token_state_slots, out=block_table)
    null_token.zero_()
    tensors = (
        cu_seqlens,
        start_tokens,
        state_slots,
        sequence_ids,
        active,
        raw_lengths,
        block_table,
        null_token,
    )
    return GLM53StagedSparseMLAPrefillBatch(
        total_tokens,
        history_blocks,
        live_slots,
        cu_seqlens,
        start_tokens,
        state_slots,
        sequence_ids,
        active,
        raw_lengths,
        block_table,
        null_token,
        tuple(tensor._version for tensor in tensors),
    )


def glm53_sparse_mla_packed_prefill_append(
    *,
    batch: GLM53StagedSparseMLAPrefillBatch,
    latent: torch.Tensor,
    key: torch.Tensor,
    gate: torch.Tensor,
    pool_ape: torch.Tensor,
    history: SparseMLAHistory[torch.Tensor],
) -> SparseMLAHistory[torch.Tensor]:
    """Append every sequence in one packed target-prefill batch."""

    if type(batch) is not GLM53StagedSparseMLAPrefillBatch:
        raise TypeError("batch must be a GLM53StagedSparseMLAPrefillBatch")
    batch.validate_unchanged()
    total_tokens = batch.total_tokens
    if any(tensor.shape[0] != total_tokens for tensor in (latent, key, gate)):
        raise ValueError("packed sparse MLA tensors disagree with the token count")
    if (
        latent.shape[1:] != (SPARSE_MLA_KV_LORA_RANK,)
        or key.shape[1:] != (SPARSE_MLA_INDEXER_HEAD_DIM,)
        or gate.shape[1:] != (SPARSE_MLA_INDEXER_HEAD_DIM,)
    ):
        raise ValueError("invalid packed sparse MLA token shape")
    device = latent.device
    if any(
        tensor.device != device or not tensor.is_cuda or not tensor.is_contiguous()
        for tensor in (latent, key, gate, pool_ape)
    ):
        raise ValueError("invalid packed sparse MLA token device or layout")
    if history.index_cache.shape != (
        batch.history_blocks,
        SPARSE_MLA_INDEX_PAGE_BYTES,
    ) or history.tail_key.shape != (
        2,
        batch.live_slots,
        SPARSE_MLA_INDEX_POOL_TOKENS,
        SPARSE_MLA_INDEXER_HEAD_DIM,
    ):
        raise ValueError("packed sparse MLA history disagrees with the receipt")

    _append_packed_sparse_mla_latent[(total_tokens,)](
        batch.block_table,
        batch.raw_lengths,
        latent,
        history.latent,
        TABLE_BLOCKS=SPARSE_MLA_COMPOUND_BLOCKS,
    )
    _append_packed_sparse_mla_index[(total_tokens,)](
        batch.block_table,
        batch.raw_lengths,
        batch.sequence_ids,
        batch.cu_seqlens,
        batch.start_tokens,
        batch.state_slots,
        key,
        gate,
        pool_ape,
        history.index_cache,
        history.tail_key,
        history.tail_gate,
        TABLE_BLOCKS=SPARSE_MLA_COMPOUND_BLOCKS,
    )
    _store_packed_sparse_mla_tail[(batch.batch_size * SPARSE_MLA_INDEX_POOL_TOKENS,)](
        key,
        gate,
        batch.cu_seqlens,
        batch.start_tokens,
        batch.state_slots,
        history.tail_key,
        history.tail_gate,
        LIVE_SLOTS=batch.live_slots,
    )
    return history


def _require_plain_int(name: str, value: int) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be an int")


@triton.jit
def _append_packed_sparse_mla_latent(
    table_ptr,
    raw_lengths_ptr,
    latent_ptr,
    latent_cache_ptr,
    TABLE_BLOCKS: tl.constexpr,
):
    token = tl.program_id(0)
    position = tl.load(raw_lengths_ptr + token) - 1
    block = tl.load(table_ptr + token * TABLE_BLOCKS + position // 128)
    within = position % 128
    offsets = tl.arange(0, 512)
    latent_page = 2 * block + within // 64
    output = latent_cache_ptr + (latent_page * 64 + within % 64) * 512
    tl.store(output + offsets, tl.load(latent_ptr + token * 512 + offsets))


@triton.jit
def _append_packed_sparse_mla_index(
    table_ptr,
    raw_lengths_ptr,
    sequence_ids_ptr,
    cu_seqlens_ptr,
    start_tokens_ptr,
    state_slots_ptr,
    key_ptr,
    gate_ptr,
    ape_ptr,
    index_cache_ptr,
    tail_key_ptr,
    tail_gate_ptr,
    TABLE_BLOCKS: tl.constexpr,
):
    token = tl.program_id(0)
    end_token = tl.load(raw_lengths_ptr + token)
    complete = end_token % 4 == 0
    sequence = tl.load(sequence_ids_ptr + token)
    sequence_start = tl.load(cu_seqlens_ptr + sequence)
    chunk_start = tl.load(start_tokens_ptr + sequence)
    state_slot = tl.load(state_slots_ptr + sequence)
    pool = end_token // 4 - 1
    offsets = tl.arange(0, 512)
    lane = offsets // 128
    dim = offsets % 128
    position = pool * 4 + lane
    input_token = sequence_start + position - chunk_start
    from_chunk = position >= chunk_start
    old_tail = state_slot * 512 + lane * 128 + dim
    key = tl.where(
        from_chunk,
        tl.load(
            key_ptr + input_token * 128 + dim, mask=complete & from_chunk, other=0.0
        ),
        tl.load(tail_key_ptr + old_tail, mask=complete & ~from_chunk, other=0.0),
    )
    gate = tl.where(
        from_chunk,
        tl.load(
            gate_ptr + input_token * 128 + dim, mask=complete & from_chunk, other=0.0
        ),
        tl.load(tail_gate_ptr + old_tail, mask=complete & ~from_chunk, other=0.0),
    )
    keys = tl.reshape(key, (4, 128))
    logits = tl.reshape(
        gate.to(tl.float32) + tl.load(ape_ptr + offsets).to(tl.float32),
        (4, 128),
    )
    probability = tl.softmax(logits, dim=0).to(tl.bfloat16)
    product = (probability * keys).to(tl.bfloat16).to(tl.float32)
    product_02, product_13 = tl.split(tl.reshape(tl.trans(product), (128, 2, 2)))
    product_0, product_2 = tl.split(product_02)
    product_1, product_3 = tl.split(product_13)
    pooled = product_0 + product_1
    pooled = pooled + product_2
    pooled = pooled + product_3
    pooled = pooled.to(tl.bfloat16).to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(pooled), axis=0), 1.0e-4) / 448.0
    fp8 = tl.clamp(pooled / scale, -448.0, 448.0).to(tl.float8e4nv)
    block = tl.load(
        table_ptr + token * TABLE_BLOCKS + (pool * 4) // 128,
        mask=complete,
        other=0,
    )
    page = index_cache_ptr + block * 4224
    index_row = pool % 32
    output_dim = tl.arange(0, 128)
    tl.store(
        page + index_row * 128 + output_dim,
        fp8.to(tl.uint8, bitcast=True),
        mask=complete,
    )
    byte = tl.arange(0, 4)
    scale_bits = scale.to(tl.int32, bitcast=True)
    scale_bytes = ((scale_bits >> (8 * byte)) & 255).to(tl.uint8)
    tl.store(page + 4096 + index_row * 4 + byte, scale_bytes, mask=complete)


@triton.jit
def _store_packed_sparse_mla_tail(
    key_ptr,
    gate_ptr,
    cu_seqlens_ptr,
    start_tokens_ptr,
    state_slots_ptr,
    tail_key_ptr,
    tail_gate_ptr,
    LIVE_SLOTS: tl.constexpr,
):
    program = tl.program_id(0)
    sequence = program // 4
    lane = program % 4
    sequence_start = tl.load(cu_seqlens_ptr + sequence)
    sequence_end = tl.load(cu_seqlens_ptr + sequence + 1)
    chunk_start = tl.load(start_tokens_ptr + sequence)
    end_token = chunk_start + sequence_end - sequence_start
    last_position = end_token - 1
    position = last_position - (last_position + 4 - lane) % 4
    appended = position >= chunk_start
    offsets = tl.arange(0, 128)
    input_token = sequence_start + position - chunk_start
    key = tl.load(key_ptr + input_token * 128 + offsets, mask=appended)
    gate = tl.load(gate_ptr + input_token * 128 + offsets, mask=appended)
    state_slot = tl.load(state_slots_ptr + sequence)
    for parity in tl.static_range(2):
        output = (parity * LIVE_SLOTS + state_slot) * 512 + lane * 128 + offsets
        tl.store(tail_key_ptr + output, key, mask=appended)
        tl.store(tail_gate_ptr + output, gate, mask=appended)

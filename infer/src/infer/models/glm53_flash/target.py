from __future__ import annotations

from dataclasses import dataclass
from math import prod
from pathlib import Path
from typing import TYPE_CHECKING

from infer.models.glm53_flash import checkpoint
from infer.models.glm53_flash import model as glm53_flash

if TYPE_CHECKING:
    from infer.models.glm53_flash.ops.core import GLM53PrefillBatch
    from infer.models.glm53_flash.ops.sparse_mla_prefill import (
        GLM53StagedSparseMLAPrefillBatch,
    )

GLM53_TARGET_LAYER_IDS = tuple(range(45))
GLM53_TARGET_KDA_LAYER_IDS = (*glm53_flash.DENSE_LAYER_IDS, *glm53_flash.SPARSE_KDA_LAYER_IDS)
GLM53_TARGET_VERIFY_WIDTH = glm53_flash.GLM53_TARGET_VERIFY_WIDTH
GLM53_TARGET_HISTORY_BLOCKS = glm53_flash.SPARSE_MLA_COMPOUND_BLOCKS
GLM53_TARGET_LIVE_SLOTS = 64
GLM53_TARGET_DECODE_DUMMY_SLOTS = max(glm53_flash.GLM53_TARGET_DECODE_BATCH_SIZES)


@dataclass(frozen=True, slots=True)
class GLM53TargetOutput[TensorT]:
    """Endpoint-workspace views valid until that workspace is reused."""

    normalized: TensorT
    token: TensorT | None


@dataclass(frozen=True, slots=True)
class GLM53TargetWeights[TensorT]:
    """Target-only payload; each layer tuple follows its exported ID tuple."""

    tp_rank: int
    endpoint: glm53_flash.GLM53EndpointWeights[TensorT]
    dense_layers: tuple[glm53_flash.GLM53LayerWeights[TensorT], ...]
    sparse_kda_layers: tuple[glm53_flash.GLM53SparseKDALayerWeights[TensorT], ...]
    sparse_mla_layers: tuple[glm53_flash.GLM53SparseMLALayerWeights[TensorT], ...]
    attention_tp_size: int = glm53_flash.TP_SIZE


@dataclass(frozen=True, slots=True)
class GLM53TargetState[TensorT]:
    """Persistent KDA states and MLA histories in ascending layer order."""

    kda_layers: tuple[glm53_flash.KDAState[TensorT], ...]
    sparse_mla_layers: tuple[glm53_flash.SparseMLAHistory[TensorT], ...]


@dataclass(frozen=True, slots=True)
class GLM53TargetPrefixSnapshots[TensorT]:
    kda_layers: tuple[glm53_flash.KDAState[TensorT], ...]
    sparse_mla_layers: tuple[glm53_flash.SparseMLAPrefixSnapshots[TensorT], ...]
    committed_lengths: TensorT


@dataclass(frozen=True, slots=True)
class GLM53TargetSparseMLATransaction[TensorT]:
    tail_key: TensorT
    tail_gate: TensorT


@dataclass(frozen=True, slots=True)
class GLM53TargetTransaction[TensorT]:
    kda_layers: tuple[glm53_flash.KDAState[TensorT], ...]
    sparse_mla_layers: tuple[GLM53TargetSparseMLATransaction[TensorT], ...]


@dataclass(frozen=True, slots=True)
class GLM53TargetCommitTable[TensorT]:
    source_addresses: TensorT
    destination_addresses: TensorT
    source_strides: TensorT
    destination_strides: TensorT
    row_words: int


@dataclass(frozen=True, slots=True)
class GLM53TargetDecodeWorkspace[TensorT]:
    """One request-major Bx4 verify view."""

    endpoint: glm53_flash.GLM53EndpointWorkspace[TensorT]
    layer: glm53_flash.GLM53DenseWorkspace[TensorT]
    kda: glm53_flash.KDADecodeWorkspace[TensorT]
    sparse_ffn: glm53_flash.SparseFFNDecodeWorkspace[TensorT]
    sparse_mla_projection: glm53_flash.SparseMLAProjectionWorkspace[TensorT]
    sparse_mla_decode: glm53_flash.SparseMLADecodeWorkspace[TensorT]
    sparse_mla_output: glm53_flash.SparseMLAOutputWorkspace[TensorT]


@dataclass(frozen=True, slots=True)
class GLM53TargetPrefillWorkspace[TensorT]:
    """One fixed prefill arena set reused serially by all 45 layers."""

    endpoint: glm53_flash.GLM53EndpointWorkspace[TensorT]
    layer: glm53_flash.GLM53DenseWorkspace[TensorT]
    kda: glm53_flash.KDAPrefillWorkspace[TensorT]
    sparse_ffn: glm53_flash.SparseFFNDecodeWorkspace[TensorT]
    sparse_mla_projection: glm53_flash.SparseMLAProjectionWorkspace[TensorT]
    sparse_mla_decode: glm53_flash.SparseMLADecodeWorkspace[TensorT]
    sparse_mla_output: glm53_flash.SparseMLAOutputWorkspace[TensorT]


@dataclass(frozen=True, slots=True)
class GLM53TargetDecodeBatch[TensorT]:
    """Fixed GPU descriptors shared by draft and Bx4 target verify."""

    token_ids: TensorT
    state_indices: TensorT
    state_indices_int64: TensorT
    cu_seqlens: TensorT
    sparse_mla: glm53_flash.SparseMLADecodeBatch[TensorT]


@dataclass(frozen=True, slots=True)
class GLM53TargetDecodeRuntime[TensorT]:
    head: glm53_flash.GLM53EndpointWorkspace[TensorT]
    batches: tuple[GLM53TargetDecodeBatch[TensorT], GLM53TargetDecodeBatch[TensorT]]


@dataclass(frozen=True, slots=True)
class GLM53TargetVerifyRuntime[TensorT]:
    workspace: GLM53TargetDecodeWorkspace[TensorT]
    batch: GLM53TargetDecodeBatch[TensorT]
    transaction: GLM53TargetTransaction[TensorT]


@dataclass(frozen=True, slots=True)
class GLM53TargetPrefillBatch[TensorT]:
    token_ids: TensorT
    kda: GLM53PrefillBatch
    sparse_mla: GLM53StagedSparseMLAPrefillBatch
    sample_indices: TensorT | None
    sample_state_indices: TensorT | None
    sample_count: int


@dataclass(frozen=True, slots=True)
class _GLM53TargetPrefillStaging[TensorT]:
    token_ids_cpu: TensorT
    token_ids: TensorT
    cu_seqlens_cpu: TensorT
    cu_seqlens: TensorT
    cu_seqlens_int64: TensorT
    state_indices_cpu: TensorT
    state_indices: TensorT
    state_indices_int64: TensorT
    has_initial_cpu: TensorT
    has_initial: TensorT
    segments_cpu: TensorT
    segments: TensorT
    start_tokens_cpu: TensorT
    start_tokens: TensorT
    sequence_ids_cpu: TensorT
    sequence_ids: TensorT
    token_state_slots_cpu: TensorT
    token_state_slots: TensorT
    active_cpu: TensorT
    active: TensorT
    raw_lengths_cpu: TensorT
    raw_lengths: TensorT
    end_tokens_cpu: TensorT
    end_tokens: TensorT
    block_table: TensorT
    null_token: TensorT
    sample_indices_cpu: TensorT
    sample_indices: TensorT
    sample_state_indices_cpu: TensorT
    sample_state_indices: TensorT


@dataclass(frozen=True, slots=True)
class _GLM53TargetDecodeStaging[TensorT]:
    control_cpu: TensorT
    control: TensorT
    table_delta_cpu: TensorT


@dataclass(slots=True)
class GLM53TargetRuntime[TensorT]:
    state: GLM53TargetState[TensorT]
    sampled_tokens: TensorT
    committed_lengths: TensorT
    execution_tables: TensorT
    decode: dict[int, GLM53TargetDecodeRuntime[TensorT]]
    target_decode: dict[int, GLM53TargetDecodeWorkspace[TensorT]]
    decode_staging: tuple[
        _GLM53TargetDecodeStaging[TensorT], _GLM53TargetDecodeStaging[TensorT]
    ]
    prefill_workspace: GLM53TargetPrefillWorkspace[TensorT]
    verify: dict[tuple[int, int], GLM53TargetVerifyRuntime[TensorT]]
    verify_length_offsets: TensorT
    prefill_staging: _GLM53TargetPrefillStaging[TensorT]
    prefix_snapshots: GLM53TargetPrefixSnapshots[TensorT]
    commit_tables: tuple[GLM53TargetCommitTable[TensorT], ...]
    attention_tp_size: int = glm53_flash.TP_SIZE

    def stage_prefill(
        self,
        token_ids: tuple[tuple[int, ...], ...],
        start_tokens: tuple[int, ...],
        state_slots: tuple[int, ...],
        sample_rows: tuple[int, ...],
    ) -> GLM53TargetPrefillBatch[TensorT]:
        batch_size = len(token_ids)
        if (
            not batch_size
            or len(start_tokens) != batch_size
            or len(state_slots) != batch_size
        ):
            raise ValueError("GLM target prefill descriptors must have equal lengths")
        total_tokens = sum(map(len, token_ids))
        if not 1 <= total_tokens <= glm53_flash.KDA_CHUNK_SIZE:
            raise ValueError(
                f"GLM target prefill requires 1 to {glm53_flash.KDA_CHUNK_SIZE} packed tokens"
            )
        if batch_size > max(self.decode):
            raise ValueError("GLM target prefill has too many sequences")
        if len(set(state_slots)) != batch_size:
            raise ValueError("GLM target prefill state slots must be unique")
        if any(type(start) is not int or start < 0 for start in start_tokens):
            raise ValueError("start_tokens must be non-negative integers")
        if any(not sequence for sequence in token_ids):
            raise ValueError("every GLM target prefill sequence must contain tokens")
        if any(not 0 <= row < batch_size for row in sample_rows) or len(
            set(sample_rows)
        ) != len(sample_rows):
            raise ValueError("invalid GLM target prefill sample rows")
        for sequence in token_ids:
            for token_id in sequence:
                _validate_target_token_id(token_id)

        from infer.models.glm53_flash.ops.core import GLM53PrefillBatch
        from infer.models.glm53_flash.ops.prefill_kda import GLM53PrefillMetadata
        from infer.models.glm53_flash.ops.segmented_conv import (
            BLOCK_M,
            validate_glm53_segmented_conv_plan,
        )
        from infer.models.glm53_flash.ops.sparse_mla_prefill import (
            stage_glm53_sparse_mla_prefill_batch,
            validate_glm53_sparse_mla_prefill_plan,
        )

        history = self.state.sparse_mla_layers[0]
        plans = tuple(
            validate_glm53_sparse_mla_prefill_plan(
                start,
                len(sequence),
                slot,
                None,
                history_blocks=history.index_cache.shape[0],
                live_slots=history.tail_key.shape[1],
                has_initial=start > 0,
            )
            for sequence, start, slot in zip(
                token_ids, start_tokens, state_slots, strict=True
            )
        )
        import torch

        staging = self.prefill_staging
        cu_seqlens = [0]
        segments = []
        token = 0
        for sequence_index, (sequence, start, slot) in enumerate(
            zip(token_ids, start_tokens, state_slots, strict=True)
        ):
            staging.state_indices_cpu[sequence_index] = slot
            staging.has_initial_cpu[sequence_index] = start > 0
            staging.start_tokens_cpu[sequence_index] = start
            staging.end_tokens_cpu[sequence_index] = start + len(sequence)
            end = token + len(sequence)
            rows = slice(token, end)
            staging.token_ids_cpu[rows].copy_(
                torch.as_tensor(sequence, dtype=torch.int64)
            )
            staging.sequence_ids_cpu[rows].fill_(sequence_index)
            staging.token_state_slots_cpu[rows].fill_(slot)
            staging.active_cpu[rows].fill_(1)
            raw_lengths = staging.raw_lengths_cpu[rows]
            raw_lengths.copy_(
                torch.arange(
                    start + 1,
                    start + len(sequence) + 1,
                    dtype=raw_lengths.dtype,
                    device=raw_lengths.device,
                )
            )
            token = end
            segments.extend(
                (sequence_index, offset) for offset in range(0, len(sequence), BLOCK_M)
            )
            cu_seqlens.append(token)
        if segments:
            staging.segments_cpu[: len(segments)].copy_(
                torch.as_tensor(segments, dtype=torch.int32)
            )
        validate_glm53_segmented_conv_plan(
            cu_seqlens,
            state_slots,
            tuple(start > 0 for start in start_tokens),
            segments,
            total_tokens=total_tokens,
            state_pages=history.tail_key.shape[1],
        )
        metadata = GLM53PrefillMetadata.stage(
            cu_seqlens,
            staging.cu_seqlens_cpu,
            staging.cu_seqlens,
            staging.cu_seqlens_int64,
        )
        device_copies = (
            (staging.state_indices, staging.state_indices_cpu, batch_size),
            (staging.state_indices_int64, staging.state_indices_cpu, batch_size),
            (staging.has_initial, staging.has_initial_cpu, batch_size),
            (staging.segments, staging.segments_cpu, len(segments)),
            (staging.start_tokens, staging.start_tokens_cpu, batch_size),
            (staging.sequence_ids, staging.sequence_ids_cpu, total_tokens),
            (
                staging.token_state_slots,
                staging.token_state_slots_cpu,
                total_tokens,
            ),
            (
                staging.active,
                staging.active_cpu,
                total_tokens,
            ),
            (staging.raw_lengths, staging.raw_lengths_cpu, total_tokens),
            (staging.end_tokens, staging.end_tokens_cpu, batch_size),
        )
        for target, source, size in device_copies:
            target[:size].copy_(source[:size], non_blocking=target.is_cuda)
        staging.token_ids[:total_tokens].copy_(
            staging.token_ids_cpu[:total_tokens],
            non_blocking=staging.token_ids.is_cuda,
        )
        sparse_mla = stage_glm53_sparse_mla_prefill_batch(
            plans=plans,
            cu_seqlens=metadata.cu_seqlens,
            start_tokens=staging.start_tokens[:batch_size],
            state_slots=staging.state_indices_int64[:batch_size],
            sequence_ids=staging.sequence_ids[:total_tokens],
            token_state_slots=staging.token_state_slots[:total_tokens],
            active=staging.active[:total_tokens],
            raw_lengths=staging.raw_lengths[:total_tokens],
            execution_tables=self.execution_tables[: history.tail_key.shape[1]],
            block_table=staging.block_table[:total_tokens],
            null_token=staging.null_token,
        )
        self.committed_lengths.index_copy_(
            0,
            staging.state_indices_int64[:batch_size],
            staging.end_tokens[:batch_size],
        )
        sample_count = len(sample_rows)
        sample_indices = sample_state_indices = None
        if sample_count:
            bucket = next(size for size in self.decode if size >= sample_count)
            for index in range(bucket):
                if index < sample_count:
                    row = sample_rows[index]
                    staging.sample_indices_cpu[index] = cu_seqlens[row + 1] - 1
                    staging.sample_state_indices_cpu[index] = state_slots[row]
                else:
                    staging.sample_indices_cpu[index] = (
                        cu_seqlens[sample_rows[0] + 1] - 1
                    )
                    staging.sample_state_indices_cpu[index] = (
                        history.tail_key.shape[1] + index - sample_count
                    )
            staging.sample_indices[:bucket].copy_(
                staging.sample_indices_cpu[:bucket],
                non_blocking=staging.sample_indices.is_cuda,
            )
            staging.sample_state_indices[:bucket].copy_(
                staging.sample_state_indices_cpu[:bucket],
                non_blocking=staging.sample_state_indices.is_cuda,
            )
            sample_indices = staging.sample_indices[:bucket]
            sample_state_indices = staging.sample_state_indices[:bucket]
        return GLM53TargetPrefillBatch(
            token_ids=staging.token_ids[:total_tokens],
            kda=GLM53PrefillBatch(
                metadata,
                staging.state_indices[:batch_size],
                staging.state_indices_int64[:batch_size],
                staging.has_initial[:batch_size],
                staging.segments[: len(segments)],
            ),
            sparse_mla=sparse_mla,
            sample_indices=sample_indices,
            sample_state_indices=sample_state_indices,
            sample_count=sample_count,
        )

    def apply_table_delta(
        self,
        state_slot: int,
        start_block: int,
        physical_blocks: tuple[int, ...],
        staging_index: int,
        row: int,
    ) -> None:
        staging = self.decode_staging[staging_index]
        if len(physical_blocks) > staging.table_delta_cpu.shape[1]:
            raise ValueError("a GLM plan exceeds the prefill history-block budget")
        if len(physical_blocks) == 1:
            staging.table_delta_cpu[row, 0] = physical_blocks[0] + 1
        elif physical_blocks:
            staging.table_delta_cpu[row, : len(physical_blocks)].copy_(
                staging.table_delta_cpu.new_tensor(physical_blocks).add_(1)
            )
        if physical_blocks:
            self.execution_tables[
                state_slot,
                start_block : start_block + len(physical_blocks),
            ].copy_(
                staging.table_delta_cpu[row, : len(physical_blocks)],
                non_blocking=self.execution_tables.is_cuda,
            )

    def capture_prefix(
        self, live_slot: int, snapshot_slot: int, tail_physical_block: int
    ) -> None:
        snapshot = self._prefix_snapshot_index(live_slot, snapshot_slot)
        self._validate_prefix_block(tail_physical_block)
        for state, cached in zip(
            self.state.kda_layers, self.prefix_snapshots.kda_layers, strict=True
        ):
            cached.recurrent[snapshot].copy_(state.recurrent[live_slot])
            cached.conv[snapshot].copy_(state.conv[live_slot])
        for history, cached in zip(
            self.state.sparse_mla_layers,
            self.prefix_snapshots.sparse_mla_layers,
            strict=True,
        ):
            cached.capture(snapshot, history, live_slot, tail_physical_block)
        self.prefix_snapshots.committed_lengths[snapshot].copy_(
            self.committed_lengths[live_slot]
        )

    def restore_prefix(
        self, snapshot_slot: int, live_slot: int, tail_physical_block: int
    ) -> None:
        snapshot = self._prefix_snapshot_index(live_slot, snapshot_slot)
        self._validate_prefix_block(tail_physical_block)
        for cached, state in zip(
            self.prefix_snapshots.kda_layers, self.state.kda_layers, strict=True
        ):
            state.recurrent[live_slot].copy_(cached.recurrent[snapshot])
            state.conv[live_slot].copy_(cached.conv[snapshot])
        for cached, history in zip(
            self.prefix_snapshots.sparse_mla_layers,
            self.state.sparse_mla_layers,
            strict=True,
        ):
            cached.restore(snapshot, history, live_slot, tail_physical_block)
        self.committed_lengths[live_slot].copy_(
            self.prefix_snapshots.committed_lengths[snapshot]
        )

    def _prefix_snapshot_index(self, live_slot: int, snapshot_slot: int) -> int:
        live_slots = self.state.kda_layers[0].recurrent.shape[0]
        snapshots = self.prefix_snapshots.committed_lengths.shape[0]
        if type(live_slot) is not int or not 0 <= live_slot < live_slots:
            raise ValueError("live_slot is outside the GLM live state pool")
        if type(snapshot_slot) is not int:
            raise ValueError("snapshot_slot is outside the GLM snapshot pool")
        snapshot = snapshot_slot - live_slots
        if not 0 <= snapshot < snapshots:
            raise ValueError("snapshot_slot is outside the GLM snapshot pool")
        return snapshot

    def _validate_prefix_block(self, physical_block: int) -> None:
        blocks = self.state.sparse_mla_layers[0].index_cache.shape[0] - 1
        if type(physical_block) is not int or not 0 <= physical_block < blocks:
            raise ValueError("prefix history block is outside the GLM history pool")

    def reset_state(self, state_slot: int) -> None:
        live_slots = self.state.kda_layers[0].recurrent.shape[0]
        if type(state_slot) is not int or not 0 <= state_slot < live_slots:
            raise ValueError("state_slot is outside the live state pool")
        for layer in self.state.kda_layers:
            layer.recurrent[state_slot].zero_()
            layer.conv[state_slot].zero_()
        for history in self.state.sparse_mla_layers:
            history.tail_key[:, state_slot].zero_()
            history.tail_gate[:, state_slot].zero_()
        self.committed_lengths[state_slot].zero_()

    def stage_decode(
        self,
        state_slots: tuple[int, ...],
        staging_index: int,
        bucket: int | None = None,
    ) -> GLM53TargetDecodeBatch[TensorT]:
        batch_size = len(state_slots)
        requested_bucket = bucket
        if bucket is None:
            try:
                bucket = next(size for size in self.decode if size >= batch_size)
            except StopIteration as error:
                raise ValueError(
                    "GLM target decode exceeds the largest bucket"
                ) from error
        if bucket not in self.decode or batch_size > bucket:
            raise ValueError("GLM target decode has an invalid bucket")
        if batch_size == 0 and requested_bucket is None:
            raise ValueError("GLM target decode requires at least one row")
        if type(staging_index) is not int or not 0 <= staging_index < len(
            self.decode_staging
        ):
            raise ValueError("invalid GLM target staging index")

        from infer.models.glm53_flash.ops.megafuse import validate_glm53_megafuse_plan

        live_slots = self.state.kda_layers[0].recurrent.shape[0]
        validate_glm53_megafuse_plan(
            (*state_slots, *((-1,) * (bucket - batch_size))),
            (*range(batch_size + 1), *((batch_size,) * (bucket - batch_size))),
            state_pages=live_slots,
        )
        staging = self.decode_staging[staging_index]
        for row in range(bucket):
            if row < batch_size:
                state_slot = state_slots[row]
                control = (
                    state_slot,
                    state_slot,
                    row + 1,
                    1,
                    state_slot,
                )
            else:
                control = (
                    live_slots + row - batch_size,
                    -1,
                    batch_size,
                    0,
                    0,
                )
            for column, value in enumerate(control):
                staging.control_cpu[row, column] = value
        staging.control[:bucket].copy_(
            staging.control_cpu[:bucket], non_blocking=staging.control.is_cuda
        )
        return self.decode[bucket].batches[staging_index]

    def prepare_decode(
        self,
        batch: GLM53TargetDecodeBatch[TensorT],
        staging: _GLM53TargetDecodeStaging[TensorT],
    ) -> None:
        import torch

        control = staging.control[: batch.token_ids.shape[0]]
        batch.state_indices_int64.copy_(control[:, 0])
        batch.state_indices.copy_(control[:, 1])
        batch.cu_seqlens[1:].copy_(control[:, 2])
        batch.sparse_mla.active.copy_(control[:, 3])
        batch.sparse_mla.state_slots.copy_(control[:, 4])
        torch.index_select(
            self.committed_lengths,
            0,
            batch.sparse_mla.state_slots,
            out=batch.sparse_mla.raw_lengths,
        )
        batch.sparse_mla.raw_lengths.add_(1).mul_(batch.sparse_mla.active)
        torch.index_select(
            self.sampled_tokens,
            0,
            batch.state_indices_int64,
            out=batch.token_ids,
        )
        torch.index_select(
            self.execution_tables,
            0,
            batch.state_indices_int64,
            out=batch.sparse_mla.block_table,
        )

    def publish_prefill(
        self,
        batch: GLM53TargetPrefillBatch[TensorT],
        token: TensorT,
    ) -> None:
        if batch.sample_state_indices is None or batch.sample_count < 1:
            raise ValueError("cannot publish an unsampled GLM prefill")
        self.sampled_tokens.index_copy_(0, batch.sample_state_indices, token)

    def publish_decoded(
        self,
        batch: GLM53TargetDecodeBatch[TensorT],
        token: TensorT,
        accepted: TensorT,
    ) -> None:
        if token.shape != batch.token_ids.shape or token.dtype != batch.token_ids.dtype:
            raise ValueError("decoded tokens must match the GLM target batch")
        if accepted.shape != batch.sparse_mla.active.shape:
            raise ValueError("accepted rows must match the GLM target batch")
        self.sampled_tokens.index_copy_(0, batch.state_indices_int64, token)
        self.committed_lengths.index_add_(
            0,
            batch.sparse_mla.state_slots,
            accepted,
        )

    def stage_verify(
        self,
        token_ids: TensorT,
        state_slots: TensorT,
        active: TensorT,
        raw_lengths: TensorT,
        staging_index: int,
    ) -> GLM53TargetVerifyRuntime[TensorT]:
        groups = token_ids.shape[0] if token_ids.ndim == 2 else 0
        key = (groups, staging_index)
        if key not in self.verify:
            raise ValueError("verify group count must match a target decode bucket")
        runtime = self.verify[key]
        batch = runtime.batch
        device = batch.token_ids.device
        for name, tensor, shape, dtype in (
            (
                "token_ids",
                token_ids,
                (groups, GLM53_TARGET_VERIFY_WIDTH),
                batch.token_ids.dtype,
            ),
            (
                "raw_lengths",
                raw_lengths,
                (groups,),
                batch.sparse_mla.raw_lengths.dtype,
            ),
            ("state_slots", state_slots, (groups,), batch.state_indices.dtype),
            ("active", active, (groups,), batch.sparse_mla.active.dtype),
        ):
            if tensor.shape != shape or tensor.dtype != dtype:
                raise ValueError(f"{name} must have shape {shape} and dtype {dtype}")
            if tensor.device != device or not tensor.is_contiguous():
                raise ValueError(f"{name} must be contiguous on {device}")

        batch.token_ids.copy_(token_ids.view(-1))
        batch.state_indices.copy_(state_slots)
        repeated_slots = batch.sparse_mla.state_slots.view(
            groups, GLM53_TARGET_VERIFY_WIDTH
        )
        repeated_slots.copy_(state_slots[:, None]).clamp_min_(0)
        batch.sparse_mla.active.view(groups, GLM53_TARGET_VERIFY_WIDTH).copy_(
            active[:, None]
        )
        request_indices = batch.state_indices_int64.view(
            groups, GLM53_TARGET_VERIFY_WIDTH
        )
        safe_state_slots = batch.state_indices_int64[:groups]
        safe_state_slots.copy_(repeated_slots[:, 0])
        _index_select_rows(
            self.committed_lengths,
            safe_state_slots,
            raw_lengths,
        )
        request_indices.copy_(repeated_slots)
        lengths = batch.sparse_mla.raw_lengths.view(groups, GLM53_TARGET_VERIFY_WIDTH)
        lengths.copy_(raw_lengths[:, None]).add_(self.verify_length_offsets)
        _index_select_rows(
            self.execution_tables,
            batch.state_indices_int64,
            batch.sparse_mla.block_table,
        )
        return runtime

    def publish_verified(
        self,
        runtime: GLM53TargetVerifyRuntime[TensorT],
        accepted: TensorT,
        active: TensorT,
        output_token_ids: TensorT,
    ) -> None:
        groups = accepted.shape[0] if accepted.ndim == 1 else 0
        batch = runtime.batch
        if groups != batch.state_indices.shape[0]:
            raise ValueError("accepted must match the target verify groups")

        from infer.models.glm53_flash.ops.speculate import (
            glm53_copy_verified_row_table,
            glm53_publish_accepted,
        )

        state_slots = batch.state_indices
        for table in self.commit_tables:
            glm53_copy_verified_row_table(
                table.source_addresses,
                table.destination_addresses,
                table.source_strides,
                table.destination_strides,
                accepted,
                state_slots,
                active,
                row_words=table.row_words,
            )
        glm53_publish_accepted(
            output_token_ids,
            accepted,
            state_slots,
            active,
            self.sampled_tokens,
            self.committed_lengths,
        )


def _index_select_rows(source, indices, out) -> None:
    import torch

    torch.index_select(source, 0, indices, out=out)


def allocate_glm53_target_runtime(
    token_id: int,
    device: str | int,
    *,
    history_block_count: int = GLM53_TARGET_HISTORY_BLOCKS,
    live_slot_count: int = GLM53_TARGET_LIVE_SLOTS,
    snapshot_slot_count: int = 0,
    attention_tp_size: int = glm53_flash.TP_SIZE,
    speculative: bool = True,
) -> GLM53TargetRuntime[object]:
    _validate_target_token_id(token_id)
    if type(history_block_count) is not int or history_block_count < 1:
        raise ValueError("history_block_count must be positive")
    if type(live_slot_count) is not int or live_slot_count < 2:
        raise ValueError("live_slot_count must be at least two")
    if type(snapshot_slot_count) is not int or snapshot_slot_count < 0:
        raise ValueError("snapshot_slot_count must be non-negative")
    if type(speculative) is not bool:
        raise TypeError("speculative must be a bool")

    from infer.models.glm53_flash.ops.megafuse import validate_glm53_megafuse_plan

    decode_batch_sizes = glm53_flash.glm53_decode_batch_sizes(
        attention_tp_size, speculative=speculative
    )
    for bucket in decode_batch_sizes:
        state_indices = (-1,) * bucket
        cu_seqlens = (0,) * (bucket + 1)
        active = validate_glm53_megafuse_plan(
            state_indices,
            cu_seqlens,
            state_pages=live_slot_count,
        )
        if active:
            raise RuntimeError("invalid GLM target decode descriptor plan")

    import torch

    request_slot_count = live_slot_count + GLM53_TARGET_DECODE_DUMMY_SLOTS
    sampled_tokens = torch.full(
        (request_slot_count,), token_id, dtype=torch.int64, device=device
    )
    execution_tables = torch.zeros(
        (request_slot_count, glm53_flash.SPARSE_MLA_COMPOUND_BLOCKS),
        dtype=torch.int32,
        device=device,
    )
    execution_tables[0, 0] = 1
    execution_tables[1, 0] = min(2, history_block_count)
    prefill_workspace = _allocate_target_prefill_workspace(
        torch,
        device,
        attention_tp_size,
        batch_capacity=max(decode_batch_sizes),
    )
    decode = {}
    for bucket in decode_batch_sizes:
        request_indices = tuple(range(live_slot_count, live_slot_count + bucket))
        batches = tuple(
            _allocate_target_batch(
                torch,
                device,
                (token_id,) * bucket,
                (0,) * bucket,
                (-1,) * bucket,
                (0,) * (bucket + 1),
                (0,) * bucket,
                (0,) * bucket,
                request_indices=request_indices,
            )
            for _ in range(2)
        )
        decode[bucket] = GLM53TargetDecodeRuntime(
            _allocate_endpoint_workspace(torch, device, bucket, attention_tp_size),
            batches,
        )
    state = _allocate_target_state(
        torch,
        device,
        history_block_count,
        live_slot_count,
        attention_tp_size,
    )
    max_groups = max(decode_batch_sizes)
    target_decode = {}
    verify = {}
    commit_tables = ()
    if speculative:
        max_verify_rows = max_groups * GLM53_TARGET_VERIFY_WIDTH
        verify_workspace = _target_decode_workspace(
            prefill_workspace, torch, device, max_verify_rows, attention_tp_size
        )
        transaction = _allocate_target_transaction(
            torch, device, max_verify_rows, attention_tp_size
        )
        verify_batches = tuple(
            _allocate_target_batch(
                torch,
                device,
                (0,) * max_verify_rows,
                tuple(range(1, GLM53_TARGET_VERIFY_WIDTH + 1)) * max_groups,
                (0,) * max_groups,
                tuple(range(0, max_verify_rows + 1, GLM53_TARGET_VERIFY_WIDTH)),
                (0,) * max_verify_rows,
                (1,) * max_verify_rows,
                request_indices=(0,) * max_verify_rows,
            )
            for _ in range(2)
        )
        for groups in decode_batch_sizes:
            rows = groups * GLM53_TARGET_VERIFY_WIDTH
            workspace = _target_decode_workspace_view(
                verify_workspace, rows, attention_tp_size
            )
            transaction_view = _target_transaction_view(transaction, rows)
            for staging_index in range(2):
                verify[groups, staging_index] = GLM53TargetVerifyRuntime(
                    workspace=workspace,
                    batch=_target_batch_view(verify_batches[staging_index], groups),
                    transaction=transaction_view,
                )
        commit_tables = _allocate_target_commit_tables(torch, transaction, state)
    else:
        decode_workspace = _target_decode_workspace(
            prefill_workspace, torch, device, max_groups, attention_tp_size
        )
        target_decode = {
            groups: _target_decode_workspace_view(
                decode_workspace, groups, attention_tp_size
            )
            for groups in decode_batch_sizes
        }
    return GLM53TargetRuntime(
        state=state,
        sampled_tokens=sampled_tokens,
        committed_lengths=torch.zeros(
            live_slot_count, dtype=torch.int32, device=device
        ),
        execution_tables=execution_tables,
        decode=decode,
        target_decode=target_decode,
        decode_staging=(
            _allocate_decode_staging(
                torch,
                device,
                live_slot_count,
                history_block_count,
                max_groups,
            ),
            _allocate_decode_staging(
                torch,
                device,
                live_slot_count,
                history_block_count,
                max_groups,
            ),
        ),
        prefill_workspace=prefill_workspace,
        verify=verify,
        verify_length_offsets=torch.arange(
            1, GLM53_TARGET_VERIFY_WIDTH + 1, dtype=torch.int32, device=device
        ),
        prefill_staging=_allocate_prefill_staging(
            torch,
            device,
            glm53_flash.KDA_CHUNK_SIZE,
            max(decode_batch_sizes),
        ),
        prefix_snapshots=_allocate_target_prefix_snapshots(
            torch, device, snapshot_slot_count, attention_tp_size
        ),
        commit_tables=commit_tables,
        attention_tp_size=attention_tp_size,
    )


def load_glm53_target_weights(
    checkpoint_dir: str | Path,
    tp_rank: int,
    device: str | int,
    attention_tp_size: int = glm53_flash.TP_SIZE,
) -> GLM53TargetWeights[object]:
    if not 0 <= tp_rank < glm53_flash.TP_SIZE:
        raise ValueError(f"tp_rank must be in [0, {glm53_flash.TP_SIZE}), got {tp_rank}")

    root = Path(checkpoint_dir)
    weight_map = checkpoint._read_checkpoint_weight_map(root)
    checkpoint.validate_glm53_checkpoint_inventory(weight_map)

    endpoint = checkpoint.load_glm53_endpoint_weights(
        root, tp_rank, device, attention_tp_size
    )
    dense_layers = []
    sparse_kda_layers = []
    sparse_mla_layers = []
    for layer_id in GLM53_TARGET_LAYER_IDS:
        if layer_id in glm53_flash.DENSE_LAYER_IDS:
            dense_layers.append(
                checkpoint.load_glm53_dense_layer_weights(
                    root, layer_id, tp_rank, device, attention_tp_size
                )
            )
        elif layer_id in glm53_flash.MAIN_SPARSE_MLA_LAYER_IDS:
            sparse_mla_layers.append(
                checkpoint.load_glm53_sparse_mla_layer_weights(
                    root, layer_id, tp_rank, device, attention_tp_size
                )
            )
        else:
            sparse_kda_layers.append(
                checkpoint.load_glm53_sparse_kda_layer_weights(
                    root, layer_id, tp_rank, device, attention_tp_size
                )
            )

    return GLM53TargetWeights(
        tp_rank=tp_rank,
        endpoint=endpoint,
        dense_layers=tuple(dense_layers),
        sparse_kda_layers=tuple(sparse_kda_layers),
        sparse_mla_layers=tuple(sparse_mla_layers),
        attention_tp_size=attention_tp_size,
    )


class GLM53TargetModel:
    def __init__(
        self,
        weights: GLM53TargetWeights[object],
        ops: object,
    ) -> None:
        _validate_target_weights(weights)
        self.weights = weights
        self.ops = ops

    def prefill_tokens(
        self,
        batch: GLM53TargetPrefillBatch[object],
        state: GLM53TargetState[object],
        workspace: GLM53TargetPrefillWorkspace[object],
        head_workspace: glm53_flash.GLM53EndpointWorkspace[object] | None,
        process_group: object,
        moe_token_counts: tuple[int, ...] | None = None,
    ) -> GLM53TargetOutput[object]:
        """Run one packed prompt chunk."""

        rank = _validate_tep4_world(process_group)
        if rank != self.weights.tp_rank:
            raise ValueError(
                f"loaded TP rank {self.weights.tp_rank} does not match process rank {rank}"
            )
        _validate_target_state(state)
        total_tokens = _validate_target_prefill_batch(batch)
        workspace = _target_prefill_workspace_view(
            workspace,
            total_tokens,
            self.weights.attention_tp_size,
            max(moe_token_counts) if moe_token_counts is not None else total_tokens,
        )
        if self.weights.attention_tp_size == 1:
            self.ops.set_moe_token_counts(total_tokens, moe_token_counts)
        all_reduce = _nccl_all_reduce

        streams = self.ops.prefill_embedding(
            batch.token_ids,
            self.weights.endpoint,
            workspace.endpoint,
            process_group,
        )
        kda_index = 0
        for weights in self.weights.dense_layers:
            streams = self.ops.prefill_dense_layer(
                streams,
                weights,
                state.kda_layers[kda_index],
                batch.kda,
                workspace.kda,
                workspace.layer,
                all_reduce,
            )
            kda_index += 1

        streams = workspace.layer.streams_out
        sparse_kda_index = 0
        sparse_mla_index = 0
        for layer_id in GLM53_TARGET_LAYER_IDS[len(glm53_flash.DENSE_LAYER_IDS) :]:
            if layer_id in glm53_flash.MAIN_SPARSE_MLA_LAYER_IDS:
                streams = self.ops.prefill_sparse_mla_layer(
                    streams,
                    self.weights.sparse_mla_layers[sparse_mla_index],
                    batch.sparse_mla,
                    state.sparse_mla_layers[sparse_mla_index],
                    workspace.sparse_mla_projection,
                    workspace.sparse_mla_decode,
                    workspace.sparse_mla_output,
                    workspace.sparse_ffn,
                    workspace.layer,
                    process_group,
                    all_reduce,
                )
                sparse_mla_index += 1
            else:
                streams = self.ops.prefill_sparse_kda_layer(
                    streams,
                    self.weights.sparse_kda_layers[sparse_kda_index],
                    state.kda_layers[kda_index],
                    batch.kda,
                    workspace.kda,
                    workspace.sparse_ffn,
                    workspace.layer,
                    process_group,
                    all_reduce,
                )
                sparse_kda_index += 1
                kda_index += 1

        normalized = self.ops.normalize_head(
            streams,
            self.weights.endpoint,
            workspace.endpoint,
            process_group,
        )
        if not batch.sample_count:
            if head_workspace is not None:
                raise ValueError(
                    "unsampled GLM prefill must not provide a head workspace"
                )
            return GLM53TargetOutput(normalized, None)
        if head_workspace is None or batch.sample_indices is None:
            raise ValueError("sampled GLM prefill requires head descriptors")
        import torch

        torch.index_select(
            streams,
            0,
            batch.sample_indices,
            out=head_workspace.streams,
        )
        token = self.ops.decode_head(
            head_workspace.streams,
            self.weights.endpoint,
            head_workspace,
            process_group,
        )
        return GLM53TargetOutput(normalized, token)

    def prefill_empty(
        self,
        workspace: GLM53TargetPrefillWorkspace[object],
        process_group: object,
        moe_token_counts: tuple[int, ...],
    ) -> None:
        rank = _validate_tep4_world(process_group)
        self.ops.set_moe_token_counts(0, moe_token_counts)
        local_capacity = max(moe_token_counts)
        sparse_ffn = _target_prefill_workspace_view(
            workspace,
            1,
            self.weights.attention_tp_size,
            local_capacity,
        ).sparse_ffn
        hidden_states = workspace.layer.normalized[:0]
        sparse_kda_index = 0
        sparse_mla_index = 0
        for layer_id in GLM53_TARGET_LAYER_IDS[len(glm53_flash.DENSE_LAYER_IDS) :]:
            if layer_id in glm53_flash.MAIN_SPARSE_MLA_LAYER_IDS:
                weights = self.weights.sparse_mla_layers[sparse_mla_index].ffn
                sparse_mla_index += 1
            else:
                weights = self.weights.sparse_kda_layers[sparse_kda_index].ffn
                sparse_kda_index += 1
            self.ops.decode_sparse_ffn(
                hidden_states,
                weights,
                sparse_ffn,
                rank,
                lambda value: value,
            )

    def decode_tokens(
        self,
        batch: GLM53TargetDecodeBatch[object],
        state: GLM53TargetState[object],
        workspace: GLM53TargetDecodeWorkspace[object],
        process_group: object,
        all_reduce: object,
    ) -> GLM53TargetOutput[object]:
        rank = _validate_tep4_world(process_group)
        if rank != self.weights.tp_rank:
            raise ValueError(
                f"loaded TP rank {self.weights.tp_rank} does not match process rank {rank}"
            )
        _validate_target_state(state)
        if self.weights.attention_tp_size == 1:
            self.ops.set_moe_token_counts(batch.token_ids.shape[0])

        streams = self.ops.decode_embedding(
            batch.token_ids,
            self.weights.endpoint,
            workspace.endpoint,
            process_group,
            all_reduce,
        )
        kda_index = 0
        for weights in self.weights.dense_layers:
            streams = self.ops.decode_dense_layer(
                streams,
                weights,
                state.kda_layers[kda_index],
                batch.state_indices,
                batch.cu_seqlens,
                workspace.kda,
                workspace.layer,
                all_reduce,
            )
            kda_index += 1

        sparse_kda_index = 0
        sparse_mla_index = 0
        for layer_id in GLM53_TARGET_LAYER_IDS[len(glm53_flash.DENSE_LAYER_IDS) :]:
            if layer_id in glm53_flash.MAIN_SPARSE_MLA_LAYER_IDS:
                streams = self.ops.decode_sparse_mla_layer(
                    streams,
                    self.weights.sparse_mla_layers[sparse_mla_index],
                    batch.sparse_mla,
                    state.sparse_mla_layers[sparse_mla_index],
                    workspace.sparse_mla_projection,
                    workspace.sparse_mla_decode,
                    workspace.sparse_mla_output,
                    workspace.sparse_ffn,
                    workspace.layer,
                    process_group,
                    all_reduce,
                )
                sparse_mla_index += 1
            else:
                streams = self.ops.decode_sparse_kda_layer(
                    streams,
                    self.weights.sparse_kda_layers[sparse_kda_index],
                    state.kda_layers[kda_index],
                    batch.state_indices,
                    batch.cu_seqlens,
                    workspace.kda,
                    workspace.sparse_ffn,
                    workspace.layer,
                    process_group,
                    all_reduce,
                )
                sparse_kda_index += 1
                kda_index += 1

        tokens = self.ops.decode_head(
            streams,
            self.weights.endpoint,
            workspace.endpoint,
            process_group,
        )
        return GLM53TargetOutput(workspace.endpoint.normalized, tokens)

    def verify_tokens(
        self,
        batch: GLM53TargetDecodeBatch[object],
        state: GLM53TargetState[object],
        transaction: GLM53TargetTransaction[object],
        workspace: GLM53TargetDecodeWorkspace[object],
        process_group: object,
        all_reduce: object,
    ) -> GLM53TargetOutput[object]:
        rank = _validate_tep4_world(process_group)
        if rank != self.weights.tp_rank:
            raise ValueError(
                f"loaded TP rank {self.weights.tp_rank} does not match process rank {rank}"
            )
        _validate_target_state(state)
        if self.weights.attention_tp_size == 1:
            self.ops.set_moe_token_counts(batch.token_ids.shape[0])

        streams = self.ops.decode_embedding(
            batch.token_ids,
            self.weights.endpoint,
            workspace.endpoint,
            process_group,
            all_reduce,
        )
        kda_index = 0
        for weights in self.weights.dense_layers:
            streams = self.ops.verify_dense_layer(
                streams,
                weights,
                state.kda_layers[kda_index],
                transaction.kda_layers[kda_index],
                batch.state_indices,
                workspace.kda,
                workspace.layer,
                all_reduce,
            )
            kda_index += 1

        sparse_kda_index = 0
        sparse_mla_index = 0
        for layer_id in GLM53_TARGET_LAYER_IDS[len(glm53_flash.DENSE_LAYER_IDS) :]:
            if layer_id in glm53_flash.MAIN_SPARSE_MLA_LAYER_IDS:
                streams = self.ops.verify_sparse_mla_layer(
                    streams,
                    self.weights.sparse_mla_layers[sparse_mla_index],
                    batch.sparse_mla,
                    state.sparse_mla_layers[sparse_mla_index],
                    transaction.sparse_mla_layers[sparse_mla_index].tail_key,
                    transaction.sparse_mla_layers[sparse_mla_index].tail_gate,
                    workspace.sparse_mla_projection,
                    workspace.sparse_mla_decode,
                    workspace.sparse_mla_output,
                    workspace.sparse_ffn,
                    workspace.layer,
                    process_group,
                    all_reduce,
                )
                sparse_mla_index += 1
            else:
                streams = self.ops.verify_sparse_kda_layer(
                    streams,
                    self.weights.sparse_kda_layers[sparse_kda_index],
                    state.kda_layers[kda_index],
                    transaction.kda_layers[kda_index],
                    batch.state_indices,
                    workspace.kda,
                    workspace.sparse_ffn,
                    workspace.layer,
                    process_group,
                    all_reduce,
                )
                sparse_kda_index += 1
                kda_index += 1

        tokens = self.ops.decode_head(
            streams,
            self.weights.endpoint,
            workspace.endpoint,
            process_group,
        )
        return GLM53TargetOutput(
            workspace.endpoint.normalized,
            tokens,
        )


def _allocate_target_state(
    torch: object,
    device: str | int,
    history_block_count: int,
    live_slot_count: int,
    attention_tp_size: int = glm53_flash.TP_SIZE,
) -> GLM53TargetState:
    zeros = lambda shape, dtype: torch.zeros(shape, dtype=dtype, device=device)
    kda_shapes = glm53_flash.kda_state_shapes(live_slot_count, attention_tp_size)
    kda_layers = tuple(
        glm53_flash.KDAState(
            recurrent=zeros(kda_shapes.recurrent, torch.float32),
            conv=zeros(kda_shapes.conv, torch.bfloat16),
        )
        for _ in GLM53_TARGET_KDA_LAYER_IDS
    )

    physical_blocks = history_block_count + 1
    latent_shape = (
        physical_blocks * len(glm53_flash.SPARSE_MLA_BF16_LATENT_PAGES),
        *glm53_flash.SPARSE_MLA_BF16_LATENT_PAGES[0],
    )
    index_shape = (physical_blocks, glm53_flash.SPARSE_MLA_INDEX_PAGE_BYTES)
    tail_shape = (
        2,
        live_slot_count,
        *glm53_flash.SPARSE_MLA_INDEX_TAIL_K_BF16_SHAPE,
    )
    sparse_mla_layers = tuple(
        glm53_flash.SparseMLAHistory(
            latent=zeros(latent_shape, torch.bfloat16),
            index_cache=zeros(index_shape, torch.uint8),
            tail_key=zeros(tail_shape, torch.bfloat16),
            tail_gate=zeros(tail_shape, torch.bfloat16),
        )
        for _ in glm53_flash.MAIN_SPARSE_MLA_LAYER_IDS
    )
    return GLM53TargetState(
        kda_layers=kda_layers,
        sparse_mla_layers=sparse_mla_layers,
    )


def _allocate_target_prefix_snapshots(
    torch: object,
    device: str | int,
    snapshot_slot_count: int,
    attention_tp_size: int = glm53_flash.TP_SIZE,
) -> GLM53TargetPrefixSnapshots:
    empty = lambda shape, dtype: torch.empty(shape, dtype=dtype, device=device)
    kda = glm53_flash.kda_state_shapes(1, attention_tp_size)
    kda_layers = tuple(
        glm53_flash.KDAState(
            recurrent=empty((snapshot_slot_count, *kda.recurrent[1:]), torch.float32),
            conv=empty((snapshot_slot_count, *kda.conv[1:]), torch.bfloat16),
        )
        for _ in GLM53_TARGET_KDA_LAYER_IDS
    )
    sparse_mla_layers = tuple(
        glm53_flash.allocate_sparse_mla_prefix_snapshots(torch, device, snapshot_slot_count)
        for _ in glm53_flash.MAIN_SPARSE_MLA_LAYER_IDS
    )
    return GLM53TargetPrefixSnapshots(
        kda_layers,
        sparse_mla_layers,
        empty((snapshot_slot_count,), torch.int32),
    )


def _target_decode_workspace(
    prefill, torch, device, capacity, attention_tp_size=glm53_flash.TP_SIZE
):
    return GLM53TargetDecodeWorkspace(
        endpoint=prefill.endpoint,
        layer=prefill.layer,
        kda=_allocate_kda_decode_workspace(torch, device, capacity, attention_tp_size),
        sparse_ffn=prefill.sparse_ffn,
        sparse_mla_projection=prefill.sparse_mla_projection,
        sparse_mla_decode=prefill.sparse_mla_decode,
        sparse_mla_output=prefill.sparse_mla_output,
    )


def _target_decode_workspace_view(workspace, rows, attention_tp_size=glm53_flash.TP_SIZE):
    moe_world_size = glm53_flash.TP_SIZE if attention_tp_size == 1 else 1
    return _workspace_view(
        workspace,
        GLM53TargetDecodeWorkspace(
            endpoint=glm53_flash.glm53_endpoint_workspace_shapes(rows, attention_tp_size),
            layer=glm53_flash.dense_workspace_shapes(rows, attention_tp_size),
            kda=glm53_flash.kda_decode_workspace_shapes(rows, attention_tp_size),
            sparse_ffn=glm53_flash.sparse_ffn_decode_workspace_shapes(rows, moe_world_size),
            sparse_mla_projection=glm53_flash.sparse_mla_projection_workspace_shapes(
                rows, attention_tp_size
            ),
            sparse_mla_decode=glm53_flash.sparse_mla_decode_workspace_shapes(
                rows, attention_tp_size
            ),
            sparse_mla_output=glm53_flash.sparse_mla_output_workspace_shapes(
                rows, attention_tp_size
            ),
        ),
    )


def _target_batch_view(batch, groups):
    rows = groups * GLM53_TARGET_VERIFY_WIDTH
    sparse_mla = batch.sparse_mla
    return GLM53TargetDecodeBatch(
        token_ids=batch.token_ids[:rows],
        state_indices=batch.state_indices[:groups],
        state_indices_int64=batch.state_indices_int64[:rows],
        cu_seqlens=batch.cu_seqlens[: groups + 1],
        sparse_mla=glm53_flash.SparseMLADecodeBatch(
            active=sparse_mla.active[:rows],
            raw_lengths=sparse_mla.raw_lengths[:rows],
            state_slots=sparse_mla.state_slots[:rows],
            block_table=sparse_mla.block_table[:rows],
            null_token=sparse_mla.null_token,
        ),
    )


def _allocate_target_prefill_workspace(
    torch: object,
    device: str | int,
    attention_tp_size: int = glm53_flash.TP_SIZE,
    *,
    batch_capacity: int,
) -> GLM53TargetPrefillWorkspace:
    capacity = glm53_flash.KDA_CHUNK_SIZE
    moe_world_size = glm53_flash.TP_SIZE if attention_tp_size == 1 else 1
    return GLM53TargetPrefillWorkspace(
        endpoint=_allocate_endpoint_workspace(
            torch, device, capacity, attention_tp_size
        ),
        layer=_allocate_layer_workspace(torch, device, capacity, attention_tp_size),
        kda=_allocate_kda_prefill_workspace(
            torch,
            device,
            batch_capacity,
            attention_tp_size,
        ),
        sparse_ffn=_allocate_sparse_ffn_workspace(
            torch, device, capacity, moe_world_size
        ),
        sparse_mla_projection=_allocate_sparse_mla_projection_workspace(
            torch, device, capacity, attention_tp_size
        ),
        sparse_mla_decode=_allocate_sparse_mla_decode_workspace(
            torch, device, capacity, attention_tp_size
        ),
        sparse_mla_output=_allocate_sparse_mla_output_workspace(
            torch, device, capacity, attention_tp_size
        ),
    )


def _target_prefill_workspace_view(
    workspace,
    total_tokens,
    attention_tp_size=glm53_flash.TP_SIZE,
    moe_local_capacity: int | None = None,
):
    moe_world_size = glm53_flash.TP_SIZE if attention_tp_size == 1 else 1
    if moe_local_capacity is None:
        moe_local_capacity = total_tokens
    return _workspace_view(
        workspace,
        GLM53TargetPrefillWorkspace(
            endpoint=glm53_flash.glm53_endpoint_workspace_shapes(
                total_tokens, attention_tp_size
            ),
            layer=glm53_flash.dense_workspace_shapes(total_tokens, attention_tp_size),
            kda=glm53_flash.kda_prefill_workspace_shapes(
                total_tokens,
                workspace.kda.initial_state.shape[0],
                attention_tp_size,
            ),
            sparse_ffn=glm53_flash.sparse_ffn_decode_workspace_shapes(
                moe_local_capacity, moe_world_size
            ),
            sparse_mla_projection=glm53_flash.sparse_mla_projection_workspace_shapes(
                total_tokens, attention_tp_size
            ),
            sparse_mla_decode=glm53_flash.sparse_mla_decode_workspace_shapes(
                total_tokens, attention_tp_size
            ),
            sparse_mla_output=glm53_flash.sparse_mla_output_workspace_shapes(
                total_tokens, attention_tp_size
            ),
        ),
    )


def _workspace_view(workspace, shapes):
    return type(workspace)(
        *(
            _workspace_view(getattr(workspace, name), getattr(shapes, name))
            if getattr(type(getattr(workspace, name)), "__dataclass_fields__", None)
            else getattr(workspace, name)
            .view(-1)[: prod(getattr(shapes, name))]
            .view(getattr(shapes, name))
            for name in workspace.__dataclass_fields__
        )
    )


def _allocate_endpoint_workspace(
    torch, device, batch_size, attention_tp_size=glm53_flash.TP_SIZE
):
    from infer.models.glm53_flash.ops.distributed_argmax import (
        allocate_glm53_distributed_argmax_workspace,
    )

    zeros = lambda shape, dtype: torch.zeros(shape, dtype=dtype, device=device)
    shapes = glm53_flash.glm53_endpoint_workspace_shapes(batch_size, attention_tp_size)
    return glm53_flash.GLM53EndpointWorkspace(
        local_token=zeros(shapes.local_token, torch.int64),
        local_active=zeros(shapes.local_active, torch.bool),
        embedding=zeros(shapes.embedding, torch.bfloat16),
        streams=zeros(shapes.streams, torch.bfloat16),
        mean_f32=zeros(shapes.mean_f32, torch.float32),
        collapsed=zeros(shapes.collapsed, torch.bfloat16),
        normalized=zeros(shapes.normalized, torch.bfloat16),
        local_logits=zeros(shapes.local_logits, torch.bfloat16),
        argmax=allocate_glm53_distributed_argmax_workspace(device, batch_size),
        token=zeros(shapes.token, torch.int64),
    )


def _allocate_layer_workspace(
    torch, device, token_capacity, attention_tp_size=glm53_flash.TP_SIZE
):
    zeros = lambda shape, dtype: torch.zeros(shape, dtype=dtype, device=device)
    shapes = glm53_flash.dense_workspace_shapes(token_capacity, attention_tp_size)
    return glm53_flash.GLM53DenseWorkspace(
        mhc_sqrsum=zeros(shapes.mhc_sqrsum, torch.float32),
        mhc_dot=zeros(shapes.mhc_dot, torch.float32),
        post=zeros(shapes.post, torch.float32),
        comb=zeros(shapes.comb, torch.float32),
        collapsed=zeros(shapes.collapsed, torch.bfloat16),
        normalized=zeros(shapes.normalized, torch.bfloat16),
        streams_mid=zeros(shapes.streams_mid, torch.bfloat16),
        hidden_fp8=zeros(shapes.hidden_fp8, torch.float8_e4m3fn),
        hidden_scale=zeros(shapes.hidden_scale, torch.float32),
        gate_up=zeros(shapes.gate_up, torch.bfloat16),
        activated=zeros(shapes.activated, torch.bfloat16),
        activated_fp8=zeros(shapes.activated_fp8, torch.float8_e4m3fn),
        activated_scale=zeros(shapes.activated_scale, torch.float32),
        ffn_output=zeros(shapes.ffn_output, torch.bfloat16),
        streams_out=zeros(shapes.streams_out, torch.bfloat16),
    )


def _allocate_kda_decode_workspace(
    torch, device, batch_size=1, attention_tp_size=glm53_flash.TP_SIZE
):
    zeros = lambda shape, dtype: torch.zeros(shape, dtype=dtype, device=device)
    shapes = glm53_flash.kda_decode_workspace_shapes(batch_size, attention_tp_size)
    return glm53_flash.KDADecodeWorkspace(
        projection=zeros(shapes.projection, torch.bfloat16),
        output_gate=zeros(shapes.output_gate, torch.bfloat16),
        gate_raw=zeros(shapes.gate_raw, torch.float32),
        local_output=zeros(shapes.local_output, torch.bfloat16),
        output=zeros(shapes.output, torch.bfloat16),
    )


def _allocate_target_transaction(
    torch,
    device,
    rows=GLM53_TARGET_VERIFY_WIDTH,
    attention_tp_size=glm53_flash.TP_SIZE,
) -> GLM53TargetTransaction:
    zeros = lambda shape, dtype: torch.zeros(shape, dtype=dtype, device=device)
    kda = glm53_flash.kda_state_shapes(rows, attention_tp_size)
    kda_layers = tuple(
        glm53_flash.KDAState(
            recurrent=zeros(kda.recurrent, torch.float32),
            conv=zeros(kda.conv, torch.bfloat16),
        )
        for _ in GLM53_TARGET_KDA_LAYER_IDS
    )
    tail = (
        rows,
        2,
        1,
        *glm53_flash.SPARSE_MLA_INDEX_TAIL_K_BF16_SHAPE,
    )
    sparse_mla_layers = tuple(
        GLM53TargetSparseMLATransaction(
            tail_key=zeros(tail, torch.bfloat16),
            tail_gate=zeros(tail, torch.bfloat16),
        )
        for _ in glm53_flash.MAIN_SPARSE_MLA_LAYER_IDS
    )
    return GLM53TargetTransaction(kda_layers, sparse_mla_layers)


def _allocate_target_commit_tables(
    torch, transaction: GLM53TargetTransaction, state: GLM53TargetState
) -> tuple[GLM53TargetCommitTable, GLM53TargetCommitTable, GLM53TargetCommitTable]:
    recurrent = tuple(
        (source.recurrent, destination.recurrent)
        for source, destination in zip(
            transaction.kda_layers, state.kda_layers, strict=True
        )
    )
    conv = tuple(
        (source.conv, destination.conv)
        for source, destination in zip(
            transaction.kda_layers, state.kda_layers, strict=True
        )
    )
    tails = []
    for source, destination in zip(
        transaction.sparse_mla_layers, state.sparse_mla_layers, strict=True
    ):
        for parity in range(destination.tail_key.shape[0]):
            tails.append((source.tail_key[:, parity, 0], destination.tail_key[parity]))
            tails.append(
                (source.tail_gate[:, parity, 0], destination.tail_gate[parity])
            )
    return tuple(
        _target_commit_table(torch, pairs) for pairs in (recurrent, conv, tails)
    )


def _target_commit_table(torch, pairs) -> GLM53TargetCommitTable:
    source, _ = pairs[0]
    row_bytes = source[0].numel() * source.element_size()
    for current_source, current_destination in pairs:
        current_bytes = current_source[0].numel() * current_source.element_size()
        if (
            current_source.dtype != current_destination.dtype
            or current_source.device != current_destination.device
            or current_source.shape[1:] != current_destination.shape[1:]
            or not current_source[0].is_contiguous()
            or not current_destination[0].is_contiguous()
            or current_bytes != row_bytes
            or current_bytes % 4
        ):
            raise ValueError("target commit rows must share an int32-aligned layout")
    device = source.device
    return GLM53TargetCommitTable(
        source_addresses=torch.tensor(
            [current.data_ptr() for current, _ in pairs],
            dtype=torch.uint64,
            device=device,
        ),
        destination_addresses=torch.tensor(
            [current.data_ptr() for _, current in pairs],
            dtype=torch.uint64,
            device=device,
        ),
        source_strides=torch.tensor(
            [current.stride(0) * current.element_size() // 4 for current, _ in pairs],
            dtype=torch.int64,
            device=device,
        ),
        destination_strides=torch.tensor(
            [current.stride(0) * current.element_size() // 4 for _, current in pairs],
            dtype=torch.int64,
            device=device,
        ),
        row_words=row_bytes // 4,
    )


def _target_transaction_view(transaction, rows):
    return GLM53TargetTransaction(
        tuple(
            glm53_flash.KDAState(layer.recurrent[:rows], layer.conv[:rows])
            for layer in transaction.kda_layers
        ),
        tuple(
            GLM53TargetSparseMLATransaction(
                layer.tail_key[:rows], layer.tail_gate[:rows]
            )
            for layer in transaction.sparse_mla_layers
        ),
    )


def _allocate_kda_prefill_workspace(
    torch, device, batch_size=1, attention_tp_size=glm53_flash.TP_SIZE
):
    from infer.models.glm53_flash.ops.prefill_kda import (
        make_glm53_prefill_kda_workspace,
    )

    zeros = lambda shape, dtype: torch.zeros(shape, dtype=dtype, device=device)
    shapes = glm53_flash.kda_prefill_workspace_shapes(
        glm53_flash.KDA_CHUNK_SIZE, batch_size, attention_tp_size
    )
    storage = zeros(shapes.kda_workspace_storage, torch.uint8)
    return glm53_flash.KDAPrefillWorkspace(
        projection=zeros(shapes.projection, torch.bfloat16),
        gates=zeros(shapes.gates, torch.bfloat16),
        beta=zeros(shapes.beta, torch.bfloat16),
        qkv=zeros(shapes.qkv, torch.bfloat16),
        initial_state=zeros(shapes.initial_state, torch.float32),
        kda_output=zeros(shapes.kda_output, torch.bfloat16),
        final_state=zeros(shapes.final_state, torch.float32),
        kda_workspace_storage=storage,
        kda_workspace=make_glm53_prefill_kda_workspace(
            storage, glm53_flash.KDA_CHUNK_SIZE, batch_size, attention_tp_size
        ),
        output=zeros(shapes.output, torch.bfloat16),
    )


def _allocate_sparse_ffn_workspace(torch, device, batch_size, moe_world_size=1):
    zeros = lambda shape, dtype: torch.zeros(shape, dtype=dtype, device=device)
    shapes = glm53_flash.sparse_ffn_decode_workspace_shapes(batch_size, moe_world_size)
    return glm53_flash.SparseFFNDecodeWorkspace(
        gathered=zeros(shapes.gathered, torch.bfloat16),
        scattered=zeros(shapes.scattered, torch.bfloat16),
        scores=zeros(shapes.scores, torch.float32),
        selection=zeros(shapes.selection, torch.float32),
        topk_values=zeros(shapes.topk_values, torch.float32),
        topk_ids64=zeros(shapes.topk_ids64, torch.int64),
        topk_ids=zeros(shapes.topk_ids, torch.int32),
        hidden_fp8=zeros(shapes.hidden_fp8, torch.float8_e4m3fn),
        hidden_scale_mn=zeros(shapes.hidden_scale_mn, torch.float32),
        routed=zeros(shapes.routed, torch.bfloat16),
        shared_gate_up=zeros(shapes.shared_gate_up, torch.bfloat16),
        shared_fp8=zeros(shapes.shared_fp8, torch.float8_e4m3fn),
        shared_scale=zeros(shapes.shared_scale, torch.float32),
        output=zeros(shapes.output, torch.bfloat16),
    )


def _allocate_sparse_mla_projection_workspace(
    torch, device, batch_size, attention_tp_size=glm53_flash.TP_SIZE
):
    zeros = lambda shape, dtype: torch.zeros(shape, dtype=dtype, device=device)
    shapes = glm53_flash.sparse_mla_projection_workspace_shapes(batch_size, attention_tp_size)
    return glm53_flash.SparseMLAProjectionWorkspace(
        hidden_fp8=zeros(shapes.hidden_fp8, torch.float8_e4m3fn),
        hidden_scale=zeros(shapes.hidden_scale, torch.float32),
        low_rank=zeros(shapes.low_rank, torch.bfloat16),
        q_resid=zeros(shapes.q_resid, torch.bfloat16),
        latent=zeros(shapes.latent, torch.bfloat16),
        q_resid_fp8=zeros(shapes.q_resid_fp8, torch.float8_e4m3fn),
        q_resid_scale=zeros(shapes.q_resid_scale, torch.float32),
        main_q=zeros(shapes.main_q, torch.bfloat16),
        index_q=zeros(shapes.index_q, torch.bfloat16),
        index_prep=zeros(shapes.index_prep, torch.bfloat16),
        key=zeros(shapes.key, torch.bfloat16),
        pool_gate=zeros(shapes.pool_gate, torch.bfloat16),
        score_weights=zeros(shapes.score_weights, torch.float32),
    )


def _allocate_sparse_mla_decode_workspace(
    torch, device, batch_size, attention_tp_size=glm53_flash.TP_SIZE
):
    zeros = lambda shape, dtype: torch.zeros(shape, dtype=dtype, device=device)
    shapes = glm53_flash.sparse_mla_decode_workspace_shapes(batch_size, attention_tp_size)
    return glm53_flash.SparseMLADecodeWorkspace(
        context_lengths=zeros(shapes.context_lengths, torch.int32),
        sequence_lengths=zeros(shapes.sequence_lengths, torch.int32),
        index_q=zeros(shapes.index_q, torch.float8_e4m3fn),
        score_weights=zeros(shapes.score_weights, torch.float32),
        main_q_hbd=zeros(shapes.main_q_hbd, torch.bfloat16),
        absorbed_hbd=zeros(shapes.absorbed_hbd, torch.bfloat16),
        attention_q=zeros(shapes.attention_q, torch.bfloat16),
        schedule=zeros(shapes.schedule, torch.int32),
        logits=zeros(shapes.logits, torch.float32),
        selected=zeros(shapes.selected, torch.int32),
        topk_offsets=zeros(shapes.topk_offsets, torch.int32),
        topk_rows=zeros(shapes.topk_rows, torch.uint8),
        sparse_ids=zeros(shapes.sparse_ids, torch.int32),
        sparse_lengths=zeros(shapes.sparse_lengths, torch.int32),
        flashinfer_workspace=zeros(shapes.flashinfer_workspace, torch.uint8),
        counter=zeros(shapes.counter, torch.uint8),
        output=zeros(shapes.output, torch.bfloat16),
    )


def _allocate_sparse_mla_output_workspace(
    torch, device, batch_size, attention_tp_size=glm53_flash.TP_SIZE
):
    zeros = lambda shape, dtype: torch.zeros(shape, dtype=dtype, device=device)
    shapes = glm53_flash.sparse_mla_output_workspace_shapes(batch_size, attention_tp_size)
    return glm53_flash.SparseMLAOutputWorkspace(
        value_hbd=zeros(shapes.value_hbd, torch.bfloat16),
        projected=zeros(shapes.projected, torch.bfloat16),
        projected_fp8=zeros(shapes.projected_fp8, torch.float8_e4m3fn),
        projected_scale=zeros(shapes.projected_scale, torch.float32),
        output=zeros(shapes.output, torch.bfloat16),
    )


def _allocate_target_batch(
    torch: object,
    device: str | int,
    token_ids: tuple[int, ...],
    raw_lengths: tuple[int, ...],
    state_indices: tuple[int, ...],
    cu_seqlens: tuple[int, ...],
    state_slots: tuple[int, ...] | None = None,
    physical_blocks: tuple[int, ...] | None = None,
    request_indices: tuple[int, ...] | None = None,
) -> GLM53TargetDecodeBatch:
    bucket = len(token_ids)
    if state_slots is None:
        state_slots = (state_indices[0],) * bucket
    if physical_blocks is None:
        physical_blocks = (1,) * bucket
    if request_indices is None:
        request_indices = state_indices
    block_table = torch.zeros(
        (bucket, glm53_flash.SPARSE_MLA_COMPOUND_BLOCKS),
        dtype=torch.int32,
        device=device,
    )
    for row, block in enumerate(physical_blocks):
        block_table[row, 0] = block
    sparse_mla = glm53_flash.SparseMLADecodeBatch(
        active=torch.tensor(raw_lengths, dtype=torch.uint8, device=device).clamp_(
            max=1
        ),
        raw_lengths=torch.tensor(raw_lengths, dtype=torch.int32, device=device),
        state_slots=torch.tensor(state_slots, dtype=torch.int32, device=device),
        block_table=block_table,
        null_token=torch.tensor((0,), dtype=torch.int32, device=device),
    )
    return GLM53TargetDecodeBatch(
        token_ids=torch.tensor(token_ids, dtype=torch.int64, device=device),
        state_indices=torch.tensor(state_indices, dtype=torch.int32, device=device),
        state_indices_int64=torch.tensor(
            request_indices, dtype=torch.int64, device=device
        ),
        cu_seqlens=torch.tensor(cu_seqlens, dtype=torch.int32, device=device),
        sparse_mla=sparse_mla,
    )


def _allocate_decode_staging(
    torch,
    device,
    dummy_start,
    history_block_count,
    batch_size=glm53_flash.GLM53_TARGET_DECODE_BATCH_SIZES[-1],
):
    empty = lambda shape, dtype: torch.empty(
        shape,
        dtype=dtype,
        device="cpu",
        pin_memory=True,
    )
    control_cpu = empty((batch_size, 5), torch.int32)
    control_cpu.zero_()
    control_cpu[:, 0] = torch.arange(
        dummy_start, dummy_start + batch_size, dtype=torch.int32
    )
    control_cpu[:, 1] = -1
    control = torch.empty((batch_size, 5), dtype=torch.int32, device=device)
    control.copy_(control_cpu)
    return _GLM53TargetDecodeStaging(
        control_cpu=control_cpu,
        control=control,
        table_delta_cpu=empty(
            (batch_size, history_block_count),
            torch.int32,
        ),
    )


def _allocate_prefill_staging(
    torch, device, capacity=glm53_flash.KDA_CHUNK_SIZE, batch_size=1
):
    segment_capacity = (capacity + 7) // 8 + batch_size - 1
    cpu = lambda shape, dtype: torch.zeros(
        shape, dtype=dtype, device="cpu", pin_memory=True
    )
    gpu = lambda shape, dtype: torch.empty(shape, dtype=dtype, device=device)
    return _GLM53TargetPrefillStaging(
        token_ids_cpu=cpu(capacity, torch.int64),
        token_ids=gpu(capacity, torch.int64),
        cu_seqlens_cpu=cpu(batch_size + 1, torch.int64),
        cu_seqlens=gpu(batch_size + 1, torch.int32),
        cu_seqlens_int64=gpu(batch_size + 1, torch.int64),
        state_indices_cpu=cpu(batch_size, torch.int64),
        state_indices=gpu(batch_size, torch.int32),
        state_indices_int64=gpu(batch_size, torch.int64),
        has_initial_cpu=cpu(batch_size, torch.bool),
        has_initial=gpu(batch_size, torch.bool),
        segments_cpu=cpu((segment_capacity, 2), torch.int32),
        segments=gpu((segment_capacity, 2), torch.int32),
        start_tokens_cpu=cpu(batch_size, torch.int32),
        start_tokens=gpu(batch_size, torch.int32),
        sequence_ids_cpu=cpu(capacity, torch.int32),
        sequence_ids=gpu(capacity, torch.int32),
        token_state_slots_cpu=cpu(capacity, torch.int64),
        token_state_slots=gpu(capacity, torch.int64),
        active_cpu=cpu(capacity, torch.uint8),
        active=gpu(capacity, torch.uint8),
        raw_lengths_cpu=cpu(capacity, torch.int32),
        raw_lengths=gpu(capacity, torch.int32),
        end_tokens_cpu=cpu(batch_size, torch.int32),
        end_tokens=gpu(batch_size, torch.int32),
        block_table=gpu((capacity, glm53_flash.SPARSE_MLA_COMPOUND_BLOCKS), torch.int32),
        null_token=gpu(1, torch.int32),
        sample_indices_cpu=cpu(batch_size, torch.int64),
        sample_indices=gpu(batch_size, torch.int64),
        sample_state_indices_cpu=cpu(batch_size, torch.int64),
        sample_state_indices=gpu(batch_size, torch.int64),
    )


def _nccl_all_reduce(input_):
    import torch.distributed as dist

    dist.all_reduce(input_)
    return input_


def _validate_target_weights(weights: GLM53TargetWeights[object]) -> None:
    _require_count("dense weights", weights.dense_layers, len(glm53_flash.DENSE_LAYER_IDS))
    _require_count(
        "sparse KDA weights",
        weights.sparse_kda_layers,
        len(glm53_flash.SPARSE_KDA_LAYER_IDS),
    )
    _require_count(
        "sparse MLA weights",
        weights.sparse_mla_layers,
        len(glm53_flash.MAIN_SPARSE_MLA_LAYER_IDS),
    )
    if not 0 <= weights.tp_rank < glm53_flash.TP_SIZE:
        raise ValueError(f"invalid loaded TP rank {weights.tp_rank}")


def _validate_target_state(state: GLM53TargetState[object]) -> None:
    _require_count("KDA states", state.kda_layers, len(GLM53_TARGET_KDA_LAYER_IDS))
    _require_count(
        "sparse MLA histories",
        state.sparse_mla_layers,
        len(glm53_flash.MAIN_SPARSE_MLA_LAYER_IDS),
    )


def _validate_target_prefill_batch(batch: GLM53TargetPrefillBatch[object]) -> int:
    if type(batch) is not GLM53TargetPrefillBatch:
        raise TypeError("batch must be a GLM53TargetPrefillBatch")
    batch.kda.metadata.validate_unchanged()
    batch.sparse_mla.validate_unchanged()
    total_tokens = batch.kda.metadata.total_tokens
    if (
        batch.kda.metadata.batch_size != batch.sparse_mla.batch_size
        or not 1 <= total_tokens <= glm53_flash.KDA_CHUNK_SIZE
        or batch.token_ids.shape != (total_tokens,)
        or batch.sparse_mla.total_tokens != total_tokens
        or not 0 <= batch.sample_count <= batch.kda.metadata.batch_size
    ):
        raise ValueError("GLM target prefill descriptors disagree")
    if batch.sample_count:
        if (
            batch.sample_indices is None
            or batch.sample_state_indices is None
            or batch.sample_indices.shape != batch.sample_state_indices.shape
            or batch.sample_indices.shape[0]
            not in glm53_flash.GLM53_TARGET_DECODE_BATCH_SIZES
        ):
            raise ValueError("GLM target prefill sample descriptors disagree")
    elif batch.sample_indices is not None or batch.sample_state_indices is not None:
        raise ValueError("unsampled GLM target prefill has sample descriptors")
    return total_tokens


def _require_count(name: str, values: tuple[object, ...], expected: int) -> None:
    if len(values) != expected:
        raise ValueError(f"{name} must contain {expected} entries, got {len(values)}")


def _validate_target_token_id(token_id: int) -> None:
    if type(token_id) is not int:
        raise TypeError("token_id must be an int")
    if not 0 <= token_id < glm53_flash.TOKENIZER_VOCAB_SIZE:
        raise ValueError(
            f"token_id must be in [0, {glm53_flash.TOKENIZER_VOCAB_SIZE}), got {token_id}"
        )


def _validate_tep4_world(process_group: object) -> int:
    return glm53_flash.validate_glm53_tep4_world(process_group)

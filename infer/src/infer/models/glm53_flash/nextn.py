from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from infer.models.glm53_flash import checkpoint
from infer.models.glm53_flash import model as glm53_flash

_NEXTN_PREFIX = checkpoint.DENSE_LAYER_PREFIX.format(layer_id=glm53_flash.NEXTN_LAYER_ID)
_NEXTN_ATTENTION_PREFIX = f"{_NEXTN_PREFIX}self_attn."


@dataclass(frozen=True, slots=True)
class GLM53NextNWeights[TensorT]:
    tp_rank: int
    embedding_norm: TensorT
    hidden_norm: TensorT
    fusion: TensorT
    input_norm: TensorT
    mla_projection: glm53_flash.SparseMLAProjectionWeights[TensorT]
    mla_decode: glm53_flash.SparseMLADecodeWeights[TensorT]
    mla_output: glm53_flash.SparseMLAOutputWeights[TensorT]
    post_attention_norm: TensorT
    ffn: glm53_flash.SparseFFNWeights[TensorT]
    output_norm: TensorT
    attention_tp_size: int = glm53_flash.TP_SIZE


@dataclass(frozen=True, slots=True)
class GLM53NextNState[TensorT]:
    history: glm53_flash.SparseMLAHistory[TensorT]


@dataclass(frozen=True, slots=True)
class GLM53NextNPrefixSnapshots[TensorT]:
    sparse_mla: glm53_flash.SparseMLAPrefixSnapshots[TensorT]
    draft_hidden: TensorT


@dataclass(frozen=True, slots=True)
class GLM53NextNWorkspace[TensorT]:
    endpoint: glm53_flash.GLM53EndpointWorkspace[TensorT]
    fusion: TensorT
    residual: TensorT
    normalized: TensorT
    sparse_mla_projection: glm53_flash.SparseMLAProjectionWorkspace[TensorT]
    sparse_mla_decode: glm53_flash.SparseMLADecodeWorkspace[TensorT]
    sparse_mla_output: glm53_flash.SparseMLAOutputWorkspace[TensorT]
    sparse_ffn: glm53_flash.SparseFFNDecodeWorkspace[TensorT]


@dataclass(frozen=True, slots=True)
class GLM53NextNTransaction[TensorT]:
    tail_key: TensorT
    tail_gate: TensorT


@dataclass(frozen=True, slots=True)
class GLM53NextNBatch[TensorT]:
    token_ids: TensorT
    state_indices: TensorT
    state_indices_int64: TensorT
    sparse_mla: glm53_flash.SparseMLADecodeBatch[TensorT]


@dataclass(frozen=True, slots=True)
class GLM53NextNPrefillBatch[TensorT]:
    token_ids: TensorT
    hidden: TensorT
    sparse_mla: object | None
    target_normalized: TensorT
    last_indices: TensorT
    state_indices: TensorT


@dataclass(frozen=True, slots=True)
class GLM53NextNRuntime[TensorT]:
    state: GLM53NextNState[TensorT]
    working_state: GLM53NextNState[TensorT]
    transactions: dict[int, GLM53NextNTransaction[TensorT]]
    workspaces: dict[int, GLM53NextNWorkspace[TensorT]]
    draft_batches: dict[int, tuple[GLM53NextNBatch[TensorT], ...]]
    verify_batches: dict[int, tuple[GLM53NextNBatch[TensorT], ...]]
    execution_tables: TensorT
    committed_lengths: TensorT
    prefill_staging: object
    prefill_token_ids: TensorT
    draft_hidden: TensorT
    draft_pool_ids: TensorT
    candidate_token_ids: TensorT
    prefix_snapshots: GLM53NextNPrefixSnapshots[TensorT] | None = None
    attention_tp_size: int = glm53_flash.TP_SIZE

    def capture_prefix(
        self, live_slot: int, snapshot_slot: int, tail_physical_block: int
    ) -> None:
        snapshot = self._prefix_snapshot_index(live_slot, snapshot_slot)
        self._validate_prefix_block(tail_physical_block)
        assert self.prefix_snapshots is not None
        self.prefix_snapshots.sparse_mla.capture(
            snapshot,
            self.state.history,
            live_slot,
            tail_physical_block,
        )
        self.prefix_snapshots.draft_hidden[snapshot].copy_(self.draft_hidden[live_slot])

    def restore_prefix(
        self, snapshot_slot: int, live_slot: int, tail_physical_block: int
    ) -> None:
        snapshot = self._prefix_snapshot_index(live_slot, snapshot_slot)
        self._validate_prefix_block(tail_physical_block)
        assert self.prefix_snapshots is not None
        self.prefix_snapshots.sparse_mla.restore(
            snapshot,
            self.state.history,
            live_slot,
            tail_physical_block,
        )
        self.draft_hidden[live_slot].copy_(self.prefix_snapshots.draft_hidden[snapshot])

    def _prefix_snapshot_index(self, live_slot: int, snapshot_slot: int) -> int:
        live_slots = self.draft_hidden.shape[0]
        if type(live_slot) is not int or not 0 <= live_slot < live_slots:
            raise ValueError("live_slot is outside the NextN live state pool")
        if self.prefix_snapshots is None:
            raise ValueError("the NextN snapshot pool is not allocated")
        snapshots = self.prefix_snapshots.draft_hidden.shape[0]
        if type(snapshot_slot) is not int:
            raise ValueError("snapshot_slot is outside the NextN snapshot pool")
        snapshot = snapshot_slot - live_slots
        if not 0 <= snapshot < snapshots:
            raise ValueError("snapshot_slot is outside the NextN snapshot pool")
        return snapshot

    def _validate_prefix_block(self, physical_block: int) -> None:
        blocks = self.state.history.index_cache.shape[0] - 1
        if type(physical_block) is not int or not 0 <= physical_block < blocks:
            raise ValueError("prefix history block is outside the NextN history pool")

    def reset_slot(self, state_slot: int) -> None:
        live_slots = self.state.history.tail_key.shape[1]
        if type(state_slot) is not int or not 0 <= state_slot < live_slots:
            raise ValueError("state_slot is outside the NextN live state pool")
        self.state.history.tail_key[:, state_slot].zero_()
        self.state.history.tail_gate[:, state_slot].zero_()
        self.draft_hidden[state_slot].zero_()
        self.draft_pool_ids[state_slot].zero_()
        self.candidate_token_ids[state_slot].zero_()

    def stage_prefill(
        self,
        token_ids: TensorT,
        target_normalized: TensorT,
        sequence_lengths: tuple[int, ...],
        start_tokens: tuple[int, ...],
        state_slots: tuple[int, ...],
    ) -> GLM53NextNPrefillBatch[TensorT]:
        """Reuse target staging after its packed forward and publication."""

        sequence_count = len(sequence_lengths)
        if not sequence_count or not (
            len(start_tokens) == len(state_slots) == sequence_count
        ):
            raise ValueError("GLM NextN prefill descriptors must have equal lengths")
        total_tokens = sum(sequence_lengths)
        if token_ids.shape != (total_tokens,) or target_normalized.shape != (
            total_tokens,
            glm53_flash.HIDDEN_SIZE,
        ):
            raise ValueError("GLM NextN prefill tensors disagree with the plan")

        from infer.models.glm53_flash.ops.nextn_stage import glm53_stage_nextn_prefill
        from infer.models.glm53_flash.ops.sparse_mla_prefill import (
            stage_glm53_sparse_mla_prefill_batch,
            validate_glm53_sparse_mla_prefill_plan,
        )

        staging = self.prefill_staging
        history = self.state.history
        pair_row = 0
        plans = []
        for length, start, slot in zip(
            sequence_lengths, start_tokens, state_slots, strict=True
        ):
            pair_count = length - (start == 0)
            if pair_count:
                cache_start = max(start - 1, 0)
                pair_row += pair_count
                plans.append(
                    validate_glm53_sparse_mla_prefill_plan(
                        cache_start,
                        pair_count,
                        slot,
                        None,
                        history_blocks=history.index_cache.shape[0],
                        live_slots=history.tail_key.shape[1],
                        has_initial=cache_start > 0,
                    )
                )

        active_count = len(plans)
        compact_hidden = self.workspaces[glm53_flash.KDA_CHUNK_SIZE].residual[:pair_row]
        glm53_stage_nextn_prefill(
            token_ids,
            target_normalized,
            self.draft_hidden,
            staging.cu_seqlens,
            staging.start_tokens,
            staging.state_indices_int64,
            staging.end_tokens,
            staging.has_initial,
            staging.sample_indices,
            staging.sample_state_indices,
            self.prefill_token_ids,
            compact_hidden,
            staging.sequence_ids,
            staging.token_state_slots,
            staging.active,
            staging.raw_lengths,
            pair_row,
            sequence_count,
        )
        sparse_mla = None
        if pair_row:
            sparse_mla = stage_glm53_sparse_mla_prefill_batch(
                plans=plans,
                cu_seqlens=staging.cu_seqlens[: active_count + 1],
                start_tokens=staging.start_tokens[:active_count],
                state_slots=staging.state_indices_int64[:active_count],
                sequence_ids=staging.sequence_ids[:pair_row],
                token_state_slots=staging.token_state_slots[:pair_row],
                active=staging.active[:pair_row],
                raw_lengths=staging.raw_lengths[:pair_row],
                execution_tables=self.execution_tables[: history.tail_key.shape[1]],
                block_table=staging.block_table[:pair_row],
                null_token=staging.null_token,
            )
        return GLM53NextNPrefillBatch(
            self.prefill_token_ids[:pair_row],
            compact_hidden,
            sparse_mla,
            target_normalized,
            staging.sample_indices[:sequence_count],
            staging.sample_state_indices[:sequence_count],
        )

    def publish_prefill(self, batch: GLM53NextNPrefillBatch[TensorT]) -> None:
        """Publish target tails after the staged NextN forward."""

        import torch

        rows = batch.last_indices.shape[0]
        hidden = self.workspaces[glm53_flash.KDA_CHUNK_SIZE].normalized[:rows]
        torch.index_select(
            batch.target_normalized,
            0,
            batch.last_indices,
            out=hidden,
        )
        self.draft_hidden.index_copy_(0, batch.state_indices, hidden)

    def seed_candidates(
        self,
        model: GLM53NextNModel,
        root_token_ids: TensorT,
        state_indices: TensorT,
        active_count: int,
        staging_index: int,
        endpoint: object,
        process_group: object,
        all_reduce: object,
    ) -> None:
        """Consume sampled prefill roots and seed each live draft chain."""

        import torch

        groups = root_token_ids.shape[0] if root_token_ids.ndim == 1 else 0
        if (
            groups not in self.draft_batches
            or state_indices.shape != (groups,)
            or not 0 <= active_count <= groups
            or staging_index not in (0, 1)
        ):
            raise ValueError("invalid GLM NextN seed batch")
        batch = self.draft_batches[groups][staging_index]
        workspace = self.workspaces[groups]
        sparse = batch.sparse_mla
        batch.token_ids.copy_(root_token_ids)
        if active_count < groups:
            batch.token_ids[active_count:].copy_(root_token_ids[:1])
        safe_states = batch.state_indices_int64
        safe_states.copy_(state_indices).clamp_(0, self.draft_hidden.shape[0] - 1)
        sparse.state_slots.copy_(safe_states)
        sparse.active.zero_()
        sparse.active[:active_count].fill_(1)
        torch.index_select(
            self.committed_lengths, 0, safe_states, out=sparse.raw_lengths
        )
        torch.index_select(
            self.execution_tables, 0, safe_states, out=sparse.block_table
        )
        torch.index_select(
            self.draft_hidden,
            0,
            safe_states,
            out=workspace.endpoint.normalized,
        )
        outcome = model.forward(
            batch.token_ids,
            workspace.endpoint.normalized,
            sparse,
            self.state.history,
            workspace,
            workspace.endpoint,
            endpoint,
            process_group,
            all_reduce,
            head_rows=slice(None),
        )
        states = state_indices[:active_count]
        self.candidate_token_ids[:, 0].index_copy_(
            0, states, root_token_ids[:active_count]
        )
        self.candidate_token_ids[:, 1].index_copy_(0, states, outcome[0][:active_count])
        self.draft_hidden.index_copy_(0, states, outcome[1][:active_count])
        self.draft_pool_ids.index_copy_(
            0,
            states,
            workspace.sparse_mla_decode.selected[:active_count],
        )

    def draft_candidates(
        self,
        model: GLM53NextNModel,
        groups: int,
        staging_index: int,
        endpoint: object,
        process_group: object,
        all_reduce: object,
    ) -> TensorT:
        """Generate two recurrent drafts after each resident first draft."""

        import torch

        batch = self.draft_batches[groups][staging_index]
        candidates = self.verify_batches[groups][staging_index].token_ids.view(
            groups, glm53_flash.GLM53_TARGET_VERIFY_WIDTH
        )
        workspace = self.workspaces[groups]
        safe_states = batch.state_indices_int64
        safe_states.copy_(batch.sparse_mla.state_slots)
        torch.index_select(self.candidate_token_ids, 0, safe_states, out=candidates)
        torch.index_select(
            self.draft_hidden,
            0,
            safe_states,
            out=workspace.endpoint.normalized,
        )
        torch.index_select(
            self.draft_pool_ids,
            0,
            safe_states,
            out=workspace.sparse_mla_decode.selected,
        )
        self.working_state.history.tail_key.copy_(self.state.history.tail_key)
        self.working_state.history.tail_gate.copy_(self.state.history.tail_gate)

        for column in range(1, glm53_flash.GLM53_TARGET_VERIFY_WIDTH - 1):
            batch.token_ids.copy_(candidates[:, column])
            outcome = model.forward(
                batch.token_ids,
                workspace.endpoint.normalized,
                batch.sparse_mla,
                self.working_state.history,
                workspace,
                workspace.endpoint,
                endpoint,
                process_group,
                all_reduce,
                head_rows=slice(None),
                shared_selected=workspace.sparse_mla_decode.selected,
            )
            candidates[:, column + 1].copy_(outcome[0])
            if column + 1 < glm53_flash.GLM53_TARGET_VERIFY_WIDTH - 1:
                batch.sparse_mla.raw_lengths.add_(1)
        return candidates

    def verify_candidates(
        self,
        model: GLM53NextNModel,
        target_token_ids: TensorT,
        target_normalized: TensorT,
        groups: int,
        staging_index: int,
        endpoint: object,
        process_group: object,
        all_reduce: object,
    ) -> None:
        """Advance all target outcomes transactionally for device selection."""

        width = glm53_flash.GLM53_TARGET_VERIFY_WIDTH
        batch = self.verify_batches[groups][staging_index]
        transaction = self.transactions[groups]
        workspace = self.workspaces[groups * width]
        self.working_state.history.tail_key.copy_(self.state.history.tail_key)
        self.working_state.history.tail_gate.copy_(self.state.history.tail_gate)
        model.forward(
            target_token_ids,
            target_normalized,
            batch.sparse_mla,
            self.working_state.history,
            workspace,
            workspace.endpoint,
            endpoint,
            process_group,
            all_reduce,
            head_rows=slice(None),
            transaction=transaction,
        )

    def commit_candidates(
        self,
        groups: int,
        staging_index: int,
        output_token_ids: TensorT,
        accepted: TensorT,
        active: TensorT,
        continuing: TensorT,
    ) -> None:
        """Commit selected NextN tails and seed only continuing requests."""

        from infer.models.glm53_flash.ops.speculate import glm53_copy_verified_rows

        batch = self.verify_batches[groups][staging_index]
        transaction = self.transactions[groups]
        state_slots = batch.state_indices
        for parity in range(self.state.history.tail_key.shape[0]):
            glm53_copy_verified_rows(
                transaction.tail_key[:, parity, 0],
                self.state.history.tail_key[parity],
                accepted,
                state_slots,
                active,
            )
            glm53_copy_verified_rows(
                transaction.tail_gate[:, parity, 0],
                self.state.history.tail_gate[parity],
                accepted,
                state_slots,
                active,
            )

        width = glm53_flash.GLM53_TARGET_VERIFY_WIDTH
        workspace = self.workspaces[groups * width]
        sources = (
            (output_token_ids.view(-1, 1), self.candidate_token_ids[:, :1]),
            (workspace.endpoint.token.view(-1, 1), self.candidate_token_ids[:, 1:2]),
            (workspace.endpoint.normalized, self.draft_hidden),
            (workspace.sparse_mla_decode.selected, self.draft_pool_ids),
        )
        for source, destination in sources:
            glm53_copy_verified_rows(
                source,
                destination,
                accepted,
                state_slots,
                continuing,
            )


def load_glm53_nextn_weights(
    checkpoint_dir: str | Path,
    tp_rank: int,
    device: str | int,
    attention_tp_size: int = glm53_flash.TP_SIZE,
) -> GLM53NextNWeights[object]:
    import torch

    root = Path(checkpoint_dir)
    specs = _nextn_checkpoint_specs()
    keyed = checkpoint._load_indexed_weights(
        root,
        tp_rank,
        device,
        specs,
        _nextn_checkpoint_shards(),
        checkpoint.NEXTN_SHARD_PINS,
        (
            f"{_NEXTN_PREFIX}eh_proj.",
            f"{_NEXTN_PREFIX}enorm.",
            f"{_NEXTN_PREFIX}hnorm.",
            f"{_NEXTN_PREFIX}input_layernorm.",
            f"{_NEXTN_PREFIX}post_attention_layernorm.",
            f"{_NEXTN_PREFIX}shared_head.norm.",
            _NEXTN_ATTENTION_PREFIX,
        ),
        "GLM NextN layer",
        torch,
        attention_tp_size,
    )
    raw = {
        spec.name: keyed[key]
        for key, spec in specs.items()
        if key.startswith(_NEXTN_ATTENTION_PREFIX)
    }
    return GLM53NextNWeights(
        tp_rank=tp_rank,
        embedding_norm=keyed[f"{_NEXTN_PREFIX}enorm.weight"],
        hidden_norm=keyed[f"{_NEXTN_PREFIX}hnorm.weight"],
        fusion=keyed[f"{_NEXTN_PREFIX}eh_proj.weight"],
        input_norm=keyed[f"{_NEXTN_PREFIX}input_layernorm.weight"],
        mla_projection=checkpoint.pack_sparse_mla_projection_weights(raw, torch),
        mla_decode=checkpoint.pack_sparse_mla_decode_weights(raw, torch),
        mla_output=checkpoint.pack_sparse_mla_output_weights(raw),
        post_attention_norm=keyed[f"{_NEXTN_PREFIX}post_attention_layernorm.weight"],
        ffn=checkpoint.load_glm53_sparse_ffn_weights(
            root, glm53_flash.NEXTN_LAYER_ID, tp_rank, device
        ),
        output_norm=keyed[f"{_NEXTN_PREFIX}shared_head.norm.weight"],
        attention_tp_size=attention_tp_size,
    )


def allocate_glm53_nextn_runtime(target: object) -> GLM53NextNRuntime[object]:
    import torch

    return _allocate_glm53_nextn_runtime(torch, target)


class GLM53NextNModel:
    def __init__(
        self,
        weights: GLM53NextNWeights[object],
        ops: object,
    ) -> None:
        if not 0 <= weights.tp_rank < glm53_flash.TP_SIZE:
            raise ValueError(f"invalid loaded TP rank {weights.tp_rank}")
        self.weights = weights
        self.ops = ops

    def forward(
        self,
        token_ids: object,
        target_normalized: object,
        batch: object,
        history: glm53_flash.SparseMLAHistory[object],
        workspace: GLM53NextNWorkspace[object],
        head_workspace: glm53_flash.GLM53EndpointWorkspace[object],
        endpoint: glm53_flash.GLM53EndpointWeights[object],
        process_group: object,
        all_reduce: object,
        *,
        head_rows: slice | None,
        transaction: GLM53NextNTransaction[object] | None = None,
        shared_selected: object | None = None,
        moe_token_counts: tuple[int, ...] | None = None,
    ) -> tuple[object, object] | None:
        rank = glm53_flash.validate_glm53_tep4_world(process_group)
        if rank != self.weights.tp_rank:
            raise ValueError(
                f"loaded TP rank {self.weights.tp_rank} does not match process rank {rank}"
            )
        if self.weights.attention_tp_size == 1:
            self.ops.set_moe_token_counts(token_ids.shape[0], moe_token_counts)
        streams = self.ops.decode_embedding(
            token_ids,
            endpoint,
            workspace.endpoint,
            process_group,
            all_reduce,
        )
        embedding = workspace.endpoint.collapsed
        embedding.copy_(streams[:, 0])
        hidden_states = self.ops.nextn_layer(
            embedding,
            target_normalized,
            self.weights,
            batch,
            history,
            workspace,
            process_group,
            all_reduce,
            None if transaction is None else transaction.tail_key,
            None if transaction is None else transaction.tail_gate,
            shared_selected,
        )
        if head_rows is None:
            return None
        return self.ops.decode_nextn_head(
            hidden_states[head_rows],
            self.weights.output_norm,
            endpoint,
            head_workspace,
            process_group,
        )

    def prefill_empty(
        self,
        workspace: GLM53NextNWorkspace[object],
        process_group: object,
        moe_token_counts: tuple[int, ...],
    ) -> None:
        rank = glm53_flash.validate_glm53_tep4_world(process_group)
        self.ops.set_moe_token_counts(0, moe_token_counts)
        local_capacity = max(moe_token_counts)
        view = _nextn_workspace_view(
            workspace,
            local_capacity,
            self.weights.attention_tp_size,
        )
        self.ops.decode_sparse_ffn(
            view.normalized[:0],
            self.weights.ffn,
            view.sparse_ffn,
            rank,
            lambda value: value,
        )


def _nextn_checkpoint_specs() -> dict[str, checkpoint.WeightSpec]:
    direct = (
        checkpoint.WeightSpec("eh_proj.weight", (4096, 8192), "BF16", None, _NEXTN_PREFIX),
        checkpoint.WeightSpec("enorm.weight", (4096,), "BF16", None, _NEXTN_PREFIX),
        checkpoint.WeightSpec("hnorm.weight", (4096,), "BF16", None, _NEXTN_PREFIX),
        checkpoint.WeightSpec(
            "input_layernorm.weight", (4096,), "BF16", None, _NEXTN_PREFIX
        ),
        checkpoint.WeightSpec(
            "post_attention_layernorm.weight",
            (4096,),
            "BF16",
            None,
            _NEXTN_PREFIX,
        ),
        checkpoint.WeightSpec(
            "shared_head.norm.weight", (4096,), "BF16", None, _NEXTN_PREFIX
        ),
    )
    attention = tuple(
        checkpoint.WeightSpec(
            spec.name,
            spec.shape,
            spec.dtype,
            spec.shard_axis,
            _NEXTN_ATTENTION_PREFIX,
        )
        for spec in checkpoint.SPARSE_MLA_WEIGHTS
    )
    return {spec.checkpoint_key: spec for spec in (*direct, *attention)}


def _nextn_checkpoint_shards() -> dict[str, str]:
    primary, secondary = (shard[0] for shard in checkpoint.NEXTN_LAYER_SHARDS)
    return {
        key: (
            secondary
            if key.startswith(_NEXTN_ATTENTION_PREFIX)
            or key.endswith(
                ("post_attention_layernorm.weight", "shared_head.norm.weight")
            )
            else primary
        )
        for key in _nextn_checkpoint_specs()
    }


def _allocate_glm53_nextn_runtime(torch, target) -> GLM53NextNRuntime:
    reference = target.state.sparse_mla_layers[0]
    device = target.execution_tables.device
    history = glm53_flash.SparseMLAHistory(
        latent=torch.zeros_like(reference.latent),
        index_cache=torch.zeros_like(reference.index_cache),
        tail_key=torch.zeros_like(reference.tail_key),
        tail_gate=torch.zeros_like(reference.tail_gate),
    )
    state = GLM53NextNState(history)
    working_state = GLM53NextNState(
        glm53_flash.SparseMLAHistory(
            latent=history.latent,
            index_cache=history.index_cache,
            tail_key=torch.zeros_like(reference.tail_key),
            tail_gate=torch.zeros_like(reference.tail_gate),
        )
    )
    width = glm53_flash.GLM53_TARGET_VERIFY_WIDTH
    arena = _nextn_workspace_arena(target)
    decode_batch_sizes = tuple(target.decode)
    attention_tp_size = target.attention_tp_size
    capacities = {
        *decode_batch_sizes,
        *(groups * width for groups in decode_batch_sizes),
        glm53_flash.KDA_CHUNK_SIZE,
    }
    snapshot_slot_count = target.prefix_snapshots.committed_lengths.shape[0]
    return GLM53NextNRuntime(
        state=state,
        working_state=working_state,
        transactions={
            groups: GLM53NextNTransaction(
                target.verify[groups, 0].transaction.sparse_mla_layers[0].tail_key,
                target.verify[groups, 0].transaction.sparse_mla_layers[0].tail_gate,
            )
            for groups in decode_batch_sizes
        },
        workspaces={
            capacity: _nextn_workspace_view(arena, capacity, attention_tp_size)
            for capacity in capacities
        },
        draft_batches={
            groups: tuple(
                _nextn_batch_view(batch, groups, 1)
                for batch in target.decode[groups].batches
            )
            for groups in decode_batch_sizes
        },
        verify_batches={
            groups: tuple(
                _nextn_batch_view(target.verify[groups, parity].batch, groups, width)
                for parity in range(2)
            )
            for groups in decode_batch_sizes
        },
        execution_tables=target.execution_tables,
        committed_lengths=target.committed_lengths,
        prefill_staging=target.prefill_staging,
        prefill_token_ids=target.prefill_workspace.endpoint.token,
        draft_hidden=torch.zeros(
            (reference.tail_key.shape[1], glm53_flash.HIDDEN_SIZE),
            dtype=torch.bfloat16,
            device=device,
        ),
        draft_pool_ids=torch.zeros(
            (reference.tail_key.shape[1], glm53_flash.SPARSE_MLA_INDEX_TOP_K),
            dtype=torch.int32,
            device=device,
        ),
        candidate_token_ids=torch.zeros(
            (reference.tail_key.shape[1], width),
            dtype=torch.int64,
            device=device,
        ),
        prefix_snapshots=_allocate_nextn_prefix_snapshots(
            torch, device, snapshot_slot_count
        ),
        attention_tp_size=attention_tp_size,
    )


def _allocate_nextn_prefix_snapshots(
    torch, device, snapshot_slot_count
) -> GLM53NextNPrefixSnapshots:
    empty = lambda shape, dtype: torch.empty(shape, dtype=dtype, device=device)
    return GLM53NextNPrefixSnapshots(
        sparse_mla=glm53_flash.allocate_sparse_mla_prefix_snapshots(
            torch, device, snapshot_slot_count
        ),
        draft_hidden=empty((snapshot_slot_count, glm53_flash.HIDDEN_SIZE), torch.bfloat16),
    )


def _nextn_workspace_arena(target) -> GLM53NextNWorkspace:
    """Alias scratch after the stream-ordered target forward has completed."""

    prefill = target.prefill_workspace
    capacity = glm53_flash.KDA_CHUNK_SIZE
    return GLM53NextNWorkspace(
        endpoint=prefill.endpoint,
        fusion=prefill.layer.streams_out.view(-1)[
            : capacity * 2 * glm53_flash.HIDDEN_SIZE
        ].view(capacity, 2 * glm53_flash.HIDDEN_SIZE),
        residual=prefill.layer.collapsed,
        normalized=prefill.layer.normalized,
        sparse_mla_projection=prefill.sparse_mla_projection,
        sparse_mla_decode=prefill.sparse_mla_decode,
        sparse_mla_output=prefill.sparse_mla_output,
        sparse_ffn=prefill.sparse_ffn,
    )


def _nextn_workspace_view(
    workspace, rows, attention_tp_size=glm53_flash.TP_SIZE
) -> GLM53NextNWorkspace:
    from infer.models.glm53_flash import target

    return target._workspace_view(
        workspace,
        GLM53NextNWorkspace(
            endpoint=glm53_flash.glm53_endpoint_workspace_shapes(rows, attention_tp_size),
            fusion=(rows, 2 * glm53_flash.HIDDEN_SIZE),
            residual=(rows, glm53_flash.HIDDEN_SIZE),
            normalized=(rows, glm53_flash.HIDDEN_SIZE),
            sparse_mla_projection=glm53_flash.sparse_mla_projection_workspace_shapes(
                rows, attention_tp_size
            ),
            sparse_mla_decode=glm53_flash.sparse_mla_decode_workspace_shapes(
                rows, attention_tp_size
            ),
            sparse_mla_output=glm53_flash.sparse_mla_output_workspace_shapes(
                rows, attention_tp_size
            ),
            sparse_ffn=glm53_flash.sparse_ffn_decode_workspace_shapes(
                rows, glm53_flash.TP_SIZE if attention_tp_size == 1 else 1
            ),
        ),
    )


def _nextn_batch_view(batch, groups, rows_per_group) -> GLM53NextNBatch:
    rows = groups * rows_per_group
    sparse = batch.sparse_mla
    return GLM53NextNBatch(
        token_ids=batch.token_ids[:rows],
        state_indices=batch.state_indices[:groups],
        state_indices_int64=batch.state_indices_int64[:rows],
        sparse_mla=glm53_flash.SparseMLADecodeBatch(
            active=sparse.active[:rows],
            raw_lengths=sparse.raw_lengths[:rows],
            state_slots=sparse.state_slots[:rows],
            block_table=sparse.block_table[:rows],
            null_token=sparse.null_token,
        ),
    )

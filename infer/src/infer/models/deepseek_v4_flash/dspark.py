from __future__ import annotations

from dataclasses import dataclass, fields

from infer.models.deepseek_v4_flash import attention, checkpoint, megamoe
from infer.models.deepseek_v4_flash import model as deepseek_v4_flash
from infer.models.deepseek_v4_flash.target import _allocate_shape_record


@dataclass(frozen=True, slots=True)
class DeepSeekV4DSparkStageWeights[TensorT]:
    stage_id: int
    attention_mhc: attention.DeepSeekV4MHCWeights[TensorT]
    attention: attention.DeepSeekV4ProjectionWeights[TensorT]
    ffn_mhc: attention.DeepSeekV4MHCWeights[TensorT]
    ffn_norm: TensorT
    ffn: megamoe.DeepSeekV4MoEWeights[TensorT]


@dataclass(frozen=True, slots=True)
class DeepSeekV4DSparkWeights[TensorT]:
    stages: tuple[DeepSeekV4DSparkStageWeights[TensorT], ...]
    main_norm: TensorT
    main_projection: TensorT
    main_projection_scale: TensorT
    head_mhc: attention.DeepSeekV4MHCWeights[TensorT]
    final_norm: TensorT
    markov_embedding: TensorT
    markov_projection: TensorT
    confidence: TensorT


@dataclass(frozen=True, slots=True)
class DeepSeekV4DSparkBatch[TensorT]:
    anchor_token_ids: TensorT
    anchor_positions: TensorT
    state_slots: TensorT
    input_token_ids: TensorT
    positions: TensorT
    persistent_slots: TensorT
    block_slots: TensorT
    context_indices: TensorT
    context_lengths: TensorT
    block_indices: TensorT
    block_lengths: TensorT
    output_token_ids: TensorT
    active_count: int
    metadata: object
    cos_sin: TensorT

    def reset_attention_metadata(self) -> None:
        deepseek_v4_flash._reset_flash_mla_metadata(self.metadata)


@dataclass(frozen=True, slots=True)
class DeepSeekV4DSparkWorkspace[TensorT]:
    layer: attention.DeepSeekV4DecodeLayerWorkspace[TensorT]
    main_input_fp8: TensorT
    main_input_scale: TensorT
    main_projected: TensorT
    main_hidden: TensorT
    main_kv_projected: TensorT
    main_kv: TensorT
    query: TensorT
    block_cache: TensorT
    head_hidden: TensorT
    base_logits: TensorT
    markov_hidden: TensorT
    markov_logits: TensorT
    confidence_features: TensorT
    confidence_previous: TensorT
    confidence: TensorT
    anchor_hidden: TensorT
    commit_hidden: TensorT


@dataclass(frozen=True, slots=True)
class DeepSeekV4DSparkOutput[TensorT]:
    token_ids: TensorT
    confidence: TensorT
    hidden: TensorT


@dataclass(frozen=True, slots=True)
class DeepSeekV4DSparkPrefixSnapshots[TensorT]:
    windows: tuple[TensorT, ...]
    anchor_hidden: TensorT


@dataclass(frozen=True, slots=True)
class DeepSeekV4DSparkRuntime[TensorT]:
    windows: tuple[TensorT, ...]
    anchor_hidden: TensorT
    batches: dict[tuple[int, int], DeepSeekV4DSparkBatch[TensorT]]
    workspaces: dict[int, DeepSeekV4DSparkWorkspace[TensorT]]
    mega_buffer: object
    live_slots: int
    prefix_snapshots: DeepSeekV4DSparkPrefixSnapshots[TensorT] | None = None

    def capture_prefix(self, live_slot: int, snapshot_slot: int) -> None:
        snapshot = self._prefix_snapshot_index(live_slot, snapshot_slot)
        assert self.prefix_snapshots is not None
        for source, destination in zip(
            self.windows, self.prefix_snapshots.windows, strict=True
        ):
            destination[snapshot].copy_(source[live_slot])
        self.prefix_snapshots.anchor_hidden[snapshot].copy_(
            self.anchor_hidden[live_slot]
        )

    def restore_prefix(self, snapshot_slot: int, live_slot: int) -> None:
        snapshot = self._prefix_snapshot_index(live_slot, snapshot_slot)
        assert self.prefix_snapshots is not None
        for source, destination in zip(
            self.prefix_snapshots.windows, self.windows, strict=True
        ):
            destination[live_slot].copy_(source[snapshot])
        self.anchor_hidden[live_slot].copy_(
            self.prefix_snapshots.anchor_hidden[snapshot]
        )

    def _prefix_snapshot_index(self, live_slot: int, snapshot_slot: int) -> int:
        if type(live_slot) is not int or not 0 <= live_slot < self.live_slots:
            raise ValueError("live_slot is outside the DSpark live state pool")
        if self.prefix_snapshots is None or type(snapshot_slot) is not int:
            raise ValueError("snapshot_slot is outside the DSpark snapshot pool")
        snapshot = snapshot_slot - self.live_slots
        if not 0 <= snapshot < self.prefix_snapshots.anchor_hidden.shape[0]:
            raise ValueError("snapshot_slot is outside the DSpark snapshot pool")
        return snapshot

    def reset_state(self, state_slot: int) -> None:
        if type(state_slot) is not int or not 0 <= state_slot < self.live_slots:
            raise ValueError("state_slot is outside the DSpark live state pool")
        for window in self.windows:
            window[state_slot].zero_()
        self.anchor_hidden[state_slot].zero_()

    @property
    def state_slot_bytes(self) -> int:
        return (
            sum(window[0].numel() * window.element_size() for window in self.windows)
            + self.anchor_hidden[0].numel() * self.anchor_hidden.element_size()
        )

    def batch(self, groups: int, parity: int) -> DeepSeekV4DSparkBatch[TensorT]:
        try:
            return self.batches[groups, parity]
        except KeyError as error:
            raise ValueError("DSpark groups/parity is not configured") from error


class DeepSeekV4DSparkModel:
    def __init__(
        self,
        weights: DeepSeekV4DSparkWeights[object],
        target_embedding: object,
        target_head: object,
        ops: object,
    ) -> None:
        if [stage.stage_id for stage in weights.stages] != list(
            range(deepseek_v4_flash.DSPARK_STAGE_COUNT)
        ):
            raise ValueError("DSpark weights must contain every stage")
        self.weights = weights
        self.target_embedding = target_embedding
        self.target_head = target_head
        self.ops = ops

    def forward(
        self,
        captured_hidden: object,
        batch: DeepSeekV4DSparkBatch[object],
        windows: tuple[object, ...],
        workspace: DeepSeekV4DSparkWorkspace[object],
        mega_buffer: object,
    ) -> DeepSeekV4DSparkOutput[object]:
        main_hidden = self.ops.prepare(
            captured_hidden,
            batch,
            self.weights,
            self.target_embedding,
            workspace,
        )
        streams = workspace.layer.streams_out
        for stage in self.weights.stages:
            streams = self.ops.stage(
                streams,
                stage,
                main_hidden,
                batch,
                windows[stage.stage_id],
                workspace,
                mega_buffer,
            )
        return self.ops.head(
            streams,
            batch,
            self.weights,
            self.target_head,
            workspace,
        )

    def draft(
        self,
        runtime: DeepSeekV4DSparkRuntime[object],
        batch: DeepSeekV4DSparkBatch[object],
        verify: object,
    ) -> None:
        import torch

        self.ops.prepare_batch(
            batch,
            verify.staging.device,
            verify.committed_lengths,
            verify.sampled_tokens,
            verify.candidates,
            runtime.live_slots,
        )
        workspace = runtime.workspaces[batch.anchor_token_ids.shape[0]]
        torch.index_select(
            runtime.anchor_hidden,
            0,
            batch.state_slots,
            out=workspace.anchor_hidden,
        )
        output = self.forward(
            workspace.anchor_hidden,
            batch,
            runtime.windows,
            workspace,
            runtime.mega_buffer,
        )
        verify.candidates[:, 1:].copy_(output.token_ids)

    def commit(
        self,
        runtime: DeepSeekV4DSparkRuntime[object],
        batch: DeepSeekV4DSparkBatch[object],
        verify: object,
    ) -> None:
        workspace = runtime.workspaces[batch.anchor_token_ids.shape[0]]
        self.ops.commit_state(
            verify.captured_hidden,
            verify.accepted,
            verify.staging.device,
            verify.committed_lengths,
            runtime.anchor_hidden,
            workspace.commit_hidden,
            batch.positions,
            batch.block_slots,
        )
        self.seed_windows(
            workspace.commit_hidden,
            batch.positions,
            batch.block_slots,
            batch.cos_sin,
            runtime.windows,
            workspace,
        )

    def seed_prefill(
        self,
        runtime: DeepSeekV4DSparkRuntime[object],
        target_runtime: object,
        batch: object,
    ) -> None:
        if not batch.dspark_seed_count:
            return
        descriptors = target_runtime.staging[batch.staging_index].device
        captured = target_runtime.resources.dspark_hidden[: batch.dspark_seed_count]
        workspace = runtime.workspaces[max(deepseek_v4_flash.DECODE_BATCH_SIZES)]
        self.seed_windows(
            captured,
            descriptors.dspark_positions[: batch.dspark_seed_count],
            descriptors.dspark_window_slots[: batch.dspark_seed_count],
            target_runtime.cos_sin[0],
            runtime.windows,
            workspace,
        )
        self.ops.seed_anchors(
            captured,
            descriptors.dspark_anchor_indices[: batch.request_count],
            descriptors.state_slots[: batch.request_count],
            runtime.anchor_hidden,
        )

    def seed_windows(
        self,
        captured_hidden: object,
        positions: object,
        window_slots: object,
        cos_sin: object,
        windows: tuple[object, ...],
        workspace: DeepSeekV4DSparkWorkspace[object],
    ) -> None:
        if not captured_hidden.shape[0]:
            return
        capacity = workspace.main_hidden.shape[0]
        for start in range(0, captured_hidden.shape[0], capacity):
            stop = min(start + capacity, captured_hidden.shape[0])
            self.ops.seed_windows(
                captured_hidden[start:stop],
                positions[start:stop],
                window_slots[start:stop],
                cos_sin,
                windows,
                self.weights,
                workspace,
            )


def load_dep4_dspark_weights(
    view: checkpoint.DeepSeekV4CheckpointView,
    ep_rank: int,
    device: str | int,
) -> DeepSeekV4DSparkWeights[object]:
    import torch

    def load(stage_id: int, name: str) -> object:
        return view.load_dspark_tensor(f"mtp.{stage_id}.{name}", ep_rank, device)

    def load_mhc(stage_id: int, prefix: str) -> attention.DeepSeekV4MHCWeights[object]:
        return attention.DeepSeekV4MHCWeights(
            base=load(stage_id, f"{prefix}_base"),
            fn=load(stage_id, f"{prefix}_fn"),
            scale=load(stage_id, f"{prefix}_scale"),
        )

    stages = []
    for stage_id in range(deepseek_v4_flash.DSPARK_STAGE_COUNT):
        raw = {
            spec.name: load(stage_id, spec.name)
            for spec in deepseek_v4_flash.COMPRESSED_ATTENTION_COMMON_WEIGHTS
        }
        stages.append(
            DeepSeekV4DSparkStageWeights(
                stage_id,
                load_mhc(stage_id, "hc_attn"),
                attention._pack_projection(raw, torch, pack_index=False),
                load_mhc(stage_id, "hc_ffn"),
                load(stage_id, "ffn_norm.weight"),
                megamoe.load_dep4_dspark_moe_weights(view, stage_id, ep_rank, device),
            )
        )

    last = deepseek_v4_flash.DSPARK_STAGE_COUNT - 1
    return DeepSeekV4DSparkWeights(
        tuple(stages),
        load(0, "main_norm.weight"),
        load(0, "main_proj.weight"),
        load(0, "main_proj.scale").to(torch.float32),
        load_mhc(last, "hc_head"),
        load(last, "norm.weight"),
        load(last, "markov_head.markov_w1.weight"),
        load(last, "markov_head.markov_w2.weight"),
        load(last, "confidence_head.proj.weight"),
    )


def allocate_dep4_dspark_runtime(
    target_runtime: object,
    device: str | int,
    *,
    metadata_allocator: object | None = None,
) -> DeepSeekV4DSparkRuntime[object]:
    import torch

    if metadata_allocator is None:
        import flash_mla

        metadata_allocator = lambda: flash_mla.get_mla_metadata()[0]
    live_slots = target_runtime.live_slots
    total_slots = live_slots + max(deepseek_v4_flash.DECODE_BATCH_SIZES)
    raw_page_bytes = deepseek_v4_flash.TARGET_STATE_SLOT_SHAPES.raw_window[-1]
    window_pages = deepseek_v4_flash.DSPARK_WINDOW_TOKENS // deepseek_v4_flash.DSPARK_PAGE_TOKENS
    windows = tuple(
        torch.zeros(
            (total_slots, window_pages, raw_page_bytes),
            dtype=torch.uint8,
            device=device,
        )
        for _ in range(deepseek_v4_flash.DSPARK_STAGE_COUNT)
    )
    anchor_hidden = torch.zeros(
        (
            total_slots,
            len(deepseek_v4_flash.DSPARK_TARGET_LAYER_IDS) * deepseek_v4_flash.HIDDEN_SIZE,
        ),
        dtype=torch.bfloat16,
        device=device,
    )
    target_snapshots = getattr(target_runtime, "prefix_snapshots", None)
    snapshot_slot_count = (
        0 if target_snapshots is None else target_snapshots.committed_lengths.shape[0]
    )
    workspaces = {
        groups: _allocate_workspace(torch, device, groups)
        for groups in (0, *deepseek_v4_flash.DECODE_BATCH_SIZES)
    }
    batches = {
        (groups, parity): _allocate_batch(
            torch,
            device,
            groups,
            target_runtime.cos_sin[0],
            metadata_allocator(),
        )
        for groups in (0, *deepseek_v4_flash.DECODE_BATCH_SIZES)
        for parity in range(deepseek_v4_flash.DECODE_DESCRIPTOR_PARITIES)
    }
    return DeepSeekV4DSparkRuntime(
        windows,
        anchor_hidden,
        batches,
        workspaces,
        target_runtime.resources.mega_buffer,
        live_slots,
        _allocate_prefix_snapshots(torch, device, snapshot_slot_count),
    )


def _allocate_prefix_snapshots(torch, device, snapshot_slot_count):
    if not snapshot_slot_count:
        return None
    raw_page_bytes = deepseek_v4_flash.TARGET_STATE_SLOT_SHAPES.raw_window[-1]
    window_pages = deepseek_v4_flash.DSPARK_WINDOW_TOKENS // deepseek_v4_flash.DSPARK_PAGE_TOKENS
    return DeepSeekV4DSparkPrefixSnapshots(
        tuple(
            torch.empty(
                (snapshot_slot_count, window_pages, raw_page_bytes),
                dtype=torch.uint8,
                device=device,
            )
            for _ in range(deepseek_v4_flash.DSPARK_STAGE_COUNT)
        ),
        torch.empty(
            (
                snapshot_slot_count,
                len(deepseek_v4_flash.DSPARK_TARGET_LAYER_IDS) * deepseek_v4_flash.HIDDEN_SIZE,
            ),
            dtype=torch.bfloat16,
            device=device,
        ),
    )


def _allocate_batch(torch, device, groups, cos_sin, metadata):
    block = deepseek_v4_flash.DSPARK_BLOCK_SIZE
    rows = groups * block

    def empty(shape, dtype=torch.int32):
        return torch.empty(shape, dtype=dtype, device=device)

    return DeepSeekV4DSparkBatch(
        empty(groups, torch.int64),
        empty(groups),
        empty(groups),
        empty(rows, torch.int64),
        empty(rows),
        empty(groups),
        empty(rows),
        empty((rows, 1, deepseek_v4_flash.DSPARK_WINDOW_TOKENS)),
        empty(rows),
        torch.full(
            (rows, 1, deepseek_v4_flash.DSPARK_PAGE_TOKENS),
            -1,
            dtype=torch.int32,
            device=device,
        ),
        empty(rows),
        empty((groups, block), torch.int64),
        groups,
        metadata,
        cos_sin,
    )


def _allocate_workspace(torch, device, groups):
    rows = groups * deepseek_v4_flash.DSPARK_BLOCK_SIZE

    def empty(shape, dtype=torch.bfloat16):
        return torch.empty(shape, dtype=dtype, device=device)

    shapes = attention.prefill_layer_workspace_shapes(0, rows)
    attention_workspace = _allocate_shape_record(
        shapes.attention,
        torch,
        device,
        fp8={"hidden_fp8", "q_residual_fp8", "output_fp8"},
        fp32={"hidden_fp32", "hidden_scale", "q_residual_scale", "output_scale"},
    )
    attention_workspace.projected_q.zero_()
    ffn = _allocate_shape_record(
        shapes.ffn,
        torch,
        device,
        fp8={"hidden_fp8", "shared_activated_fp8"},
        fp32={
            "router_logits",
            "topk_weights",
            "hidden_scale",
            "shared_activated_scale",
        },
        int32={"topk_ids"},
    )
    fp32 = {"post", "comb"}
    shared = {
        field.name: empty(
            getattr(shapes, field.name),
            torch.float32 if field.name in fp32 else torch.bfloat16,
        )
        for field in fields(shapes)
        if field.name not in {"attention", "ffn"}
    }
    layer = attention.DeepSeekV4DecodeLayerWorkspace(attention_workspace, ffn, **shared)
    return DeepSeekV4DSparkWorkspace(
        layer,
        empty((rows, 3 * deepseek_v4_flash.HIDDEN_SIZE), torch.float8_e4m3fn),
        empty(
            (rows, 3 * deepseek_v4_flash.HIDDEN_SIZE // deepseek_v4_flash.FP8_BLOCK_SIZE),
            torch.float32,
        ),
        empty((rows, deepseek_v4_flash.HIDDEN_SIZE)),
        empty((rows, deepseek_v4_flash.HIDDEN_SIZE)),
        empty((rows, deepseek_v4_flash.HEAD_DIM)),
        empty((rows, deepseek_v4_flash.HEAD_DIM)),
        empty((rows, 1, deepseek_v4_flash.NUM_QUERY_HEADS, deepseek_v4_flash.HEAD_DIM)),
        empty(
            (groups, deepseek_v4_flash.TARGET_STATE_SLOT_SHAPES.raw_window[-1]), torch.uint8
        ),
        empty((rows, deepseek_v4_flash.HIDDEN_SIZE)),
        empty((rows, deepseek_v4_flash.VOCAB_SIZE)),
        empty((groups, deepseek_v4_flash.DSPARK_MARKOV_RANK)),
        empty((groups, deepseek_v4_flash.VOCAB_SIZE)),
        empty((rows, deepseek_v4_flash.HIDDEN_SIZE + deepseek_v4_flash.DSPARK_MARKOV_RANK)),
        empty((groups, deepseek_v4_flash.DSPARK_BLOCK_SIZE), torch.int64),
        empty((groups, deepseek_v4_flash.DSPARK_BLOCK_SIZE)),
        empty(
            (
                groups,
                len(deepseek_v4_flash.DSPARK_TARGET_LAYER_IDS) * deepseek_v4_flash.HIDDEN_SIZE,
            )
        ),
        empty(
            (
                rows,
                len(deepseek_v4_flash.DSPARK_TARGET_LAYER_IDS) * deepseek_v4_flash.HIDDEN_SIZE,
            )
        ),
    )

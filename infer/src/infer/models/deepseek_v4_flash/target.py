from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, fields
from math import prod

from infer.models.deepseek_v4_flash import attention, checkpoint, megamoe
from infer.models.deepseek_v4_flash import model as deepseek_v4_flash

_ROPE_BASES = (10_000.0, 160_000.0)
_YARN_CORRECTION_RANGE = (15, 25)
DECODE_DUMMY_SLOTS = max(deepseek_v4_flash.DECODE_BATCH_SIZES)
_PREFILL_STATE_PAGE = {4: 4, 128: 8}
_PREFILL_RETAINED_PAGES = {4: 2, 128: 16}
_RAW_RING_PAGES = deepseek_v4_flash.RAW_STATE_RING_TOKENS // deepseek_v4_flash.DSPARK_PAGE_TOKENS
_C4_RING_PAGES = deepseek_v4_flash.C4_STATE_RING_TOKENS // _PREFILL_STATE_PAGE[4]
_C128_RING_PAGES = deepseek_v4_flash.C128_STATE_RING_TOKENS // _PREFILL_STATE_PAGE[128]
_PREFILL_STATE_RING_PAGES = {4: _C4_RING_PAGES, 128: _C128_RING_PAGES}
_C4_STATE_TABLE_WIDTH = deepseek_v4_flash.PREFILL_CHUNK_TOKENS // 4 + 3
_C128_STATE_TABLE_WIDTH = deepseek_v4_flash.PREFILL_CHUNK_TOKENS // 8 + 17
_C4_STATE_PAGES = _C4_STATE_TABLE_WIDTH + 3 * deepseek_v4_flash.MAX_PREFILL_REQUESTS
_C128_STATE_PAGES = _C128_STATE_TABLE_WIDTH + 16 * deepseek_v4_flash.MAX_PREFILL_REQUESTS
_C128_SELECTED_WIDTH = (
    (deepseek_v4_flash.MAX_CONTEXT_TOKENS // 128 + deepseek_v4_flash.SWA_WINDOW_TOKENS + 127)
    // 128
    * 128
)
_DSPARK_CAPTURE_WIDTH = (
    len(deepseek_v4_flash.DSPARK_TARGET_LAYER_IDS) * deepseek_v4_flash.HIDDEN_SIZE
)
_DSPARK_PREFILL_ROWS = (
    deepseek_v4_flash.MAX_PREFILL_REQUESTS * deepseek_v4_flash.DSPARK_WINDOW_TOKENS
)


@dataclass(frozen=True, slots=True)
class DeepSeekV4TargetWeights[TensorT]:
    embedding: TensorT
    layers: tuple[attention.DeepSeekV4DecodeLayerWeights[TensorT], ...]
    head_mhc: attention.DeepSeekV4MHCWeights[TensorT]
    final_norm: TensorT
    head: TensorT


@dataclass(frozen=True, slots=True)
class DeepSeekV4DecodeLayerInputs[TensorT]:
    batch: deepseek_v4_flash.DeepSeekV4CompressionBatch[TensorT]
    raw_slots: TensorT
    main: deepseek_v4_flash.DeepSeekV4Compression[TensorT] | None
    raw: deepseek_v4_flash.DeepSeekV4RawAttention[TensorT]
    attention_state: (
        attention.DeepSeekV4C4DecodeState[TensorT]
        | attention.DeepSeekV4C128DecodeState[TensorT]
        | None
    )
    workspace: attention.DeepSeekV4DecodeLayerWorkspace[TensorT]
    metadata: object
    mega_buffer: object


@dataclass(frozen=True, slots=True)
class _DeepSeekV4PrefillLayerBinding[TensorT]:
    main: deepseek_v4_flash.DeepSeekV4Compression[TensorT] | None
    index: deepseek_v4_flash.DeepSeekV4Compression[TensorT] | None
    persistent_main: TensorT | None
    persistent_index: TensorT | None
    raw_cache: TensorT
    sink: TensorT


@dataclass(frozen=True, slots=True)
class _DeepSeekV4PrefillDescriptors[TensorT]:
    token_ids: TensorT
    positions: TensorT
    request_indices: TensorT
    query_start: TensorT
    prefix_lengths: TensorT
    state_slots: TensorT
    block_table: TensorT
    block_table_base: TensorT
    c4_slots: TensorT
    c4_outputs: TensorT
    c4_table: TensorT
    c4_base: TensorT
    c128_slots: TensorT
    c128_outputs: TensorT
    c128_table: TensorT
    c128_base: TensorT
    c4_seed_source: TensorT
    c4_seed_destination: TensorT
    c4_retain_source: TensorT
    c4_retain_destination: TensorT
    c128_seed_source: TensorT
    c128_seed_destination: TensorT
    c128_retain_source: TensorT
    c128_retain_destination: TensorT
    sample_indices: TensorT
    sample_slots: TensorT
    end_lengths: TensorT
    dspark_indices: TensorT
    dspark_positions: TensorT
    dspark_window_slots: TensorT
    dspark_anchor_indices: TensorT


@dataclass(frozen=True, slots=True)
class _DeepSeekV4PrefillStaging[TensorT]:
    device: _DeepSeekV4PrefillDescriptors[TensorT]
    host: _DeepSeekV4PrefillDescriptors[TensorT]


@dataclass(frozen=True, slots=True)
class _DeepSeekV4PrefillResources[TensorT]:
    workspace_a: attention.DeepSeekV4DecodeLayerWorkspace[TensorT]
    workspace_b: attention.DeepSeekV4DecodeLayerWorkspace[TensorT]
    workspace_c4: attention.DeepSeekV4DecodeLayerWorkspace[TensorT]
    c4_main: _DeepSeekV4CompressionScratch[TensorT]
    c4_index: _DeepSeekV4CompressionScratch[TensorT]
    c128_main: _DeepSeekV4CompressionScratch[TensorT]
    c4_main_state: TensorT
    c4_index_state: TensorT
    c128_state: TensorT
    c4_main_transfer: TensorT
    c4_index_transfer: TensorT
    c128_transfer: TensorT
    raw_indices: TensorT
    raw_lengths: TensorT
    raw_query: TensorT | None
    attention: deepseek_v4_flash.DeepSeekV4PrefillWorkspace[TensorT]
    index: deepseek_v4_flash.DeepSeekV4PrefillIndexWorkspace[TensorT]
    head: (
        DeepSeekV4EndpointWorkspace[TensorT] | DeepSeekV4TEP4EndpointWorkspace[TensorT]
    )
    head_tokens: TensorT
    embedding_tokens: TensorT | None
    embedding_select: TensorT | None
    dspark_streams: TensorT
    dspark_hidden: TensorT
    mega_buffer: object


@dataclass(frozen=True, slots=True)
class DeepSeekV4PrefillBatch:
    staging_index: int
    token_count: int
    request_count: int
    table_width: int
    c4_state_width: int
    c128_state_width: int
    c4_seed_count: int
    c4_retain_count: int
    c128_seed_count: int
    c128_retain_count: int
    sample_count: int
    dspark_seed_count: int


@dataclass(frozen=True, slots=True)
class _DeepSeekV4VerifyControls[TensorT]:
    state_slots: TensorT
    table_deltas: TensorT
    remaining: TensorT
    ignore_eos: TensorT
    active: TensorT


@dataclass(frozen=True, slots=True)
class _DeepSeekV4VerifyStaging[TensorT]:
    device: _DeepSeekV4VerifyControls[TensorT]
    host: _DeepSeekV4VerifyControls[TensorT]


@dataclass(frozen=True, slots=True)
class DeepSeekV4VerifyRuntime[TensorT]:
    execution: _DeepSeekV4DecodeRuntime[TensorT]
    candidates: TensorT
    target_tokens: TensorT
    captured_hidden: TensorT
    matches: TensorT
    non_eos: TensorT
    accepted: TensorT
    records: TensorT
    tail_indices: TensorT
    tail_tokens: TensorT
    lengths: TensorT
    staging: _DeepSeekV4VerifyStaging[TensorT]
    committed_lengths: TensorT
    sampled_tokens: TensorT
    result_records: TensorT
    live_slots: int

    def stage(
        self,
        controls: tuple[tuple[int, int, int, int, bool], ...],
    ) -> None:
        groups = self.candidates.shape[0]
        if len(controls) > groups:
            raise ValueError("DeepSeek verify rows exceed the selected bucket")
        host = self.staging.host
        for row in range(groups):
            if row < len(controls):
                slot, table_column, table_block, remaining, ignore_eos = controls[row]
                if not 0 <= slot < self.live_slots:
                    raise ValueError("DeepSeek verify slot is outside the live pool")
                if not 1 <= remaining <= deepseek_v4_flash.DSPARK_VERIFY_WIDTH:
                    raise ValueError("DeepSeek verify acceptance is out of range")
                active = 1
            else:
                slot = self.live_slots + row
                table_column = 0
                table_block = -1
                remaining = ignore_eos = active = 0
            host.state_slots[row] = slot
            host.table_deltas[row, 0] = table_column
            host.table_deltas[row, 1] = table_block
            host.remaining[row] = remaining
            host.ignore_eos[row] = ignore_eos
            host.active[row] = active
        for field in fields(host):
            destination = getattr(self.staging.device, field.name)
            destination.copy_(
                getattr(host, field.name), non_blocking=destination.is_cuda
            )

    def prepare(self) -> None:
        from infer.models.deepseek_v4_flash.ops.stage import stage_deepseek_v4_verify

        stage_deepseek_v4_verify(self)

    def reset_attention_metadata(self) -> None:
        for inputs in self.execution.layer_inputs:
            deepseek_v4_flash._reset_flash_mla_metadata(inputs.metadata)


@dataclass(frozen=True, slots=True)
class _DeepSeekV4TargetDecodeControls[TensorT]:
    state_slots: TensorT
    table_deltas: TensorT
    active: TensorT


@dataclass(frozen=True, slots=True)
class _DeepSeekV4TargetDecodeStaging[TensorT]:
    device: _DeepSeekV4TargetDecodeControls[TensorT]
    host: _DeepSeekV4TargetDecodeControls[TensorT]


@dataclass(frozen=True, slots=True)
class DeepSeekV4TargetDecodeRuntime[TensorT]:
    execution: _DeepSeekV4DecodeRuntime[TensorT]
    target_tokens: TensorT
    records: TensorT
    lengths: TensorT
    staging: _DeepSeekV4TargetDecodeStaging[TensorT]
    committed_lengths: TensorT
    sampled_tokens: TensorT
    result_records: TensorT
    live_slots: int

    def stage(self, controls: tuple[tuple[int, int, int], ...]) -> None:
        rows = self.target_tokens.shape[0]
        if len(controls) > rows:
            raise ValueError("DeepSeek target rows exceed the selected bucket")
        host = self.staging.host
        for row in range(rows):
            if row < len(controls):
                slot, table_column, table_block = controls[row]
                if not 0 <= slot < self.live_slots:
                    raise ValueError("DeepSeek target slot is outside the live pool")
                active = 1
            else:
                slot = self.live_slots + row
                table_column = 0
                table_block = -1
                active = 0
            host.state_slots[row] = slot
            host.table_deltas[row, 0] = table_column
            host.table_deltas[row, 1] = table_block
            host.active[row] = active
        for field in fields(host):
            destination = getattr(self.staging.device, field.name)
            destination.copy_(
                getattr(host, field.name), non_blocking=destination.is_cuda
            )

    def prepare(self) -> None:
        from infer.models.deepseek_v4_flash.ops.stage import stage_deepseek_v4_target

        stage_deepseek_v4_target(self)

    def publish(self) -> None:
        import torch

        state_slots = self.staging.device.state_slots
        active = self.staging.device.active
        self.records[:, 0].copy_(active)
        self.records[:, 1].copy_(self.target_tokens)
        self.result_records.index_copy_(0, state_slots, self.records)
        self.sampled_tokens.index_copy_(0, state_slots, self.target_tokens)
        torch.index_select(self.committed_lengths, 0, state_slots, out=self.lengths)
        self.lengths.add_(active)
        self.committed_lengths.index_copy_(0, state_slots, self.lengths)

    def reset_attention_metadata(self) -> None:
        for inputs in self.execution.layer_inputs:
            deepseek_v4_flash._reset_flash_mla_metadata(inputs.metadata)


@dataclass(frozen=True, slots=True)
class DeepSeekV4TargetPrefixSnapshots[TensorT]:
    state: deepseek_v4_flash.DeepSeekV4TargetState[TensorT]
    c4_main_tail: TensorT
    c4_index_tail: tuple[TensorT, ...]
    c128_main_tail: TensorT
    committed_lengths: TensorT


@dataclass(frozen=True, slots=True)
class DeepSeekV4TargetRuntime[TensorT]:
    state: deepseek_v4_flash.DeepSeekV4TargetState[TensorT]
    sampled_tokens: TensorT
    committed_lengths: TensorT
    result_records: TensorT
    execution_tables: TensorT
    verify: dict[tuple[int, int], DeepSeekV4VerifyRuntime[TensorT]]
    decode: dict[tuple[int, int], DeepSeekV4TargetDecodeRuntime[TensorT]]
    bindings: tuple[_DeepSeekV4PrefillLayerBinding[TensorT], ...]
    resources: _DeepSeekV4PrefillResources[TensorT]
    staging: tuple[_DeepSeekV4PrefillStaging[TensorT], ...]
    cos_sin: tuple[TensorT, TensorT]
    live_slots: int
    history_blocks: int
    history: tuple[object, ...]
    prefix_snapshots: DeepSeekV4TargetPrefixSnapshots[TensorT] | None = None

    def capture_prefix(
        self, live_slot: int, snapshot_slot: int, tail_physical_block: int
    ) -> None:
        snapshot = self._prefix_snapshot_index(live_slot, snapshot_slot)
        self._validate_prefix_block(tail_physical_block)
        assert self.prefix_snapshots is not None
        for state_field in fields(self.state):
            getattr(self.prefix_snapshots.state, state_field.name)[:, snapshot].copy_(
                getattr(self.state, state_field.name)[:, live_slot]
            )
        c4_main, c4_index, c128_main = self.history
        self.prefix_snapshots.c4_main_tail[:, snapshot].copy_(
            c4_main[:, tail_physical_block]
        )
        for destination, source in zip(
            self.prefix_snapshots.c4_index_tail, c4_index, strict=True
        ):
            destination[snapshot].copy_(source[tail_physical_block])
        self.prefix_snapshots.c128_main_tail[:, snapshot].copy_(
            c128_main[:, tail_physical_block]
        )
        self.prefix_snapshots.committed_lengths[snapshot].copy_(
            self.committed_lengths[live_slot]
        )

    def restore_prefix(
        self, snapshot_slot: int, live_slot: int, tail_physical_block: int
    ) -> None:
        snapshot = self._prefix_snapshot_index(live_slot, snapshot_slot)
        self._validate_prefix_block(tail_physical_block)
        assert self.prefix_snapshots is not None
        for state_field in fields(self.state):
            getattr(self.state, state_field.name)[:, live_slot].copy_(
                getattr(self.prefix_snapshots.state, state_field.name)[:, snapshot]
            )
        c4_main, c4_index, c128_main = self.history
        c4_main[:, tail_physical_block].copy_(
            self.prefix_snapshots.c4_main_tail[:, snapshot]
        )
        for destination, source in zip(
            c4_index, self.prefix_snapshots.c4_index_tail, strict=True
        ):
            destination[tail_physical_block].copy_(source[snapshot])
        c128_main[:, tail_physical_block].copy_(
            self.prefix_snapshots.c128_main_tail[:, snapshot]
        )
        self.committed_lengths[live_slot].copy_(
            self.prefix_snapshots.committed_lengths[snapshot]
        )

    def _prefix_snapshot_index(self, live_slot: int, snapshot_slot: int) -> int:
        if type(live_slot) is not int or not 0 <= live_slot < self.live_slots:
            raise ValueError("live_slot is outside the DeepSeek live state pool")
        if self.prefix_snapshots is None or type(snapshot_slot) is not int:
            raise ValueError("snapshot_slot is outside the DeepSeek snapshot pool")
        snapshot = snapshot_slot - self.live_slots
        if not 0 <= snapshot < self.prefix_snapshots.committed_lengths.shape[0]:
            raise ValueError("snapshot_slot is outside the DeepSeek snapshot pool")
        return snapshot

    def _validate_prefix_block(self, physical_block: int) -> None:
        if (
            type(physical_block) is not int
            or not 0 <= physical_block < self.history_blocks
        ):
            raise ValueError(
                "prefix history block is outside the DeepSeek history pool"
            )

    def reset_state(self, state_slot: int) -> None:
        if type(state_slot) is not int or not 0 <= state_slot < self.live_slots:
            raise ValueError("state_slot is outside the live state pool")
        self.state.raw_window[:, state_slot].zero_()
        for values in (self.state.c4_main, self.state.c4_index, self.state.c128_main):
            midpoint = values.shape[-1] // 2
            values[:, state_slot, :, :midpoint].fill_(
                deepseek_v4_flash.COMPRESS_STATE_INITIAL_VALUES[0]
            )
            values[:, state_slot, :, midpoint:].fill_(
                deepseek_v4_flash.COMPRESS_STATE_INITIAL_VALUES[1]
            )
        self.committed_lengths[state_slot].zero_()

    def stage_prefill(
        self,
        staging_index: int,
        token_ids: tuple[tuple[int, ...], ...],
        start_tokens: tuple[int, ...],
        state_slots: tuple[int, ...],
        table_deltas: tuple[tuple[int, tuple[int, ...]], ...],
        sample_rows: tuple[int, ...],
    ) -> DeepSeekV4PrefillBatch:
        if not 0 <= staging_index < len(self.staging):
            raise ValueError("DeepSeek prefill staging index is out of range")
        requests = len(token_ids)
        if not requests:
            if start_tokens or state_slots or table_deltas or sample_rows:
                raise ValueError("empty DeepSeek prefill descriptors must agree")
            return DeepSeekV4PrefillBatch(
                staging_index, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            )
        if not 1 <= requests <= deepseek_v4_flash.MAX_PREFILL_REQUESTS or not (
            len(start_tokens) == len(state_slots) == len(table_deltas) == requests
        ):
            raise ValueError("DeepSeek prefill requires one to four equal descriptors")
        lengths = tuple(map(len, token_ids))
        tokens = sum(lengths)
        if not 1 <= tokens <= deepseek_v4_flash.PREFILL_CHUNK_TOKENS or any(
            not length for length in lengths
        ):
            raise ValueError("DeepSeek prefill requires 1 to 4096 packed tokens")
        if len(set(state_slots)) != requests or any(
            not 0 <= slot < self.live_slots for slot in state_slots
        ):
            raise ValueError("DeepSeek prefill state slots must be unique and live")
        if any(
            start < 0 or start + length > deepseek_v4_flash.MAX_CONTEXT_TOKENS
            for start, length in zip(start_tokens, lengths, strict=True)
        ):
            raise ValueError("DeepSeek prefill positions exceed the model context")
        if len(set(sample_rows)) != len(sample_rows) or any(
            not 0 <= row < requests for row in sample_rows
        ):
            raise ValueError("DeepSeek prefill sample rows are invalid")

        staging = self.staging[staging_index]
        host = staging.host
        query = 0
        dspark_seed = 0
        host.query_start[0] = 0
        for request, (sequence, start, slot, delta) in enumerate(
            zip(token_ids, start_tokens, state_slots, table_deltas, strict=True)
        ):
            delta_start, blocks = delta
            if (
                delta_start < 0
                or delta_start + len(blocks) > self.execution_tables.shape[1]
                or any(not 0 <= block < self.history_blocks for block in blocks)
            ):
                raise ValueError("DeepSeek prefill table delta is out of range")
            request_start = query
            host.prefix_lengths[request] = start
            host.state_slots[request] = slot
            host.end_lengths[request] = start + len(sequence)
            host.block_table_base[request] = 0
            for offset, token_id in enumerate(sequence):
                if not 0 <= token_id < deepseek_v4_flash.VOCAB_SIZE:
                    raise ValueError("DeepSeek prefill token ID is out of range")
                host.token_ids[query] = token_id
                host.positions[query] = start + offset
                host.request_indices[query] = request
                query += 1
            host.query_start[request + 1] = query
            keep = (
                min(len(sequence), deepseek_v4_flash.DSPARK_WINDOW_TOKENS)
                if self.verify
                else 0
            )
            for offset in range(len(sequence) - keep, len(sequence)):
                position = start + offset
                host.dspark_indices[dspark_seed] = request_start + offset
                host.dspark_positions[dspark_seed] = position
                host.dspark_window_slots[dspark_seed] = (
                    slot * deepseek_v4_flash.DSPARK_WINDOW_TOKENS
                    + position % deepseek_v4_flash.DSPARK_WINDOW_TOKENS
                )
                dspark_seed += 1
            host.dspark_anchor_indices[request] = dspark_seed - 1
            if len(blocks) == 1:
                host.block_table[request, 0] = blocks[0]
            elif blocks:
                host.block_table[request, : len(blocks)].copy_(
                    host.block_table.new_tensor(blocks)
                )
            if blocks:
                self.execution_tables[
                    slot, delta_start : delta_start + len(blocks)
                ].copy_(
                    host.block_table[request, : len(blocks)],
                    non_blocking=self.execution_tables.is_cuda,
                )

        c4 = _stage_prefill_state(
            start_tokens,
            lengths,
            state_slots,
            4,
            host.c4_slots,
            host.c4_table,
            host.c4_base,
            host.c4_seed_source,
            host.c4_seed_destination,
            host.c4_retain_source,
            host.c4_retain_destination,
        )
        c128 = _stage_prefill_state(
            start_tokens,
            lengths,
            state_slots,
            128,
            host.c128_slots,
            host.c128_table,
            host.c128_base,
            host.c128_seed_source,
            host.c128_seed_destination,
            host.c128_retain_source,
            host.c128_retain_destination,
        )
        if c4[0] > _C4_STATE_PAGES or c128[0] > _C128_STATE_PAGES:
            raise ValueError("DeepSeek prefill state scratch capacity exceeded")

        table_width = max(
            (start + length + 127) // 128
            for start, length in zip(start_tokens, lengths, strict=True)
        )
        sample_count = len(sample_rows)
        for index, row in enumerate(sample_rows):
            host.sample_indices[index] = host.query_start[row + 1] - 1
            host.sample_slots[index] = state_slots[row]

        device = staging.device
        copies = (
            (device.token_ids, host.token_ids, tokens),
            (device.positions, host.positions, tokens),
            (device.request_indices, host.request_indices, tokens),
            (device.query_start, host.query_start, requests + 1),
            (device.prefix_lengths, host.prefix_lengths, requests),
            (device.state_slots, host.state_slots, requests),
            (device.block_table_base, host.block_table_base, requests),
            (device.c4_slots, host.c4_slots, tokens),
            (device.c4_base, host.c4_base, requests),
            (device.c128_slots, host.c128_slots, tokens),
            (device.c128_base, host.c128_base, requests),
            (device.c4_seed_source, host.c4_seed_source, c4[2]),
            (device.c4_seed_destination, host.c4_seed_destination, c4[2]),
            (device.c4_retain_source, host.c4_retain_source, c4[3]),
            (device.c4_retain_destination, host.c4_retain_destination, c4[3]),
            (device.c128_seed_source, host.c128_seed_source, c128[2]),
            (device.c128_seed_destination, host.c128_seed_destination, c128[2]),
            (device.c128_retain_source, host.c128_retain_source, c128[3]),
            (device.c128_retain_destination, host.c128_retain_destination, c128[3]),
            (device.sample_indices, host.sample_indices, sample_count),
            (device.sample_slots, host.sample_slots, sample_count),
            (device.end_lengths, host.end_lengths, requests),
            (device.dspark_indices, host.dspark_indices, dspark_seed),
            (device.dspark_positions, host.dspark_positions, dspark_seed),
            (
                device.dspark_window_slots,
                host.dspark_window_slots,
                dspark_seed,
            ),
            (
                device.dspark_anchor_indices,
                host.dspark_anchor_indices,
                requests,
            ),
        )
        for destination, source, count in copies:
            destination[:count].copy_(source[:count], non_blocking=destination.is_cuda)
        _copy_packed_rows(device.c4_table, host.c4_table, requests, c4[1])
        _copy_packed_rows(device.c128_table, host.c128_table, requests, c128[1])
        block_table = device.block_table.view(-1)[: requests * table_width].view(
            requests, table_width
        )
        for request, slot in enumerate(state_slots):
            block_table[request].copy_(
                self.execution_tables[slot, :table_width],
                non_blocking=block_table.is_cuda,
            )
        from infer.models.deepseek_v4_flash.ops.stage import (
            stage_deepseek_v4_prefill_outputs,
        )

        stage_deepseek_v4_prefill_outputs(
            device.positions[:tokens],
            device.request_indices[:tokens],
            block_table,
            device.c4_outputs[:tokens],
            device.c128_outputs[:tokens],
        )
        return DeepSeekV4PrefillBatch(
            staging_index,
            tokens,
            requests,
            table_width,
            c4[1],
            c128[1],
            c4[2],
            c4[3],
            c128[2],
            c128[3],
            sample_count,
            dspark_seed,
        )

    def publish_prefill(self, batch: DeepSeekV4PrefillBatch) -> None:
        if not batch.request_count:
            return
        staging = self.staging[batch.staging_index]
        from infer.models.deepseek_v4_flash.ops.stage import (
            publish_deepseek_v4_prefill_lengths,
        )

        publish_deepseek_v4_prefill_lengths(
            self.committed_lengths,
            staging.device.state_slots[: batch.request_count],
            staging.device.end_lengths[: batch.request_count],
        )
        if not batch.sample_count:
            return
        tokens = self.resources.head_tokens[: batch.sample_count]
        slots = staging.device.sample_slots[: batch.sample_count]
        self.sampled_tokens.index_copy_(0, slots, tokens)
        self.result_records[:, 1].index_copy_(0, slots, tokens)


@dataclass(frozen=True, slots=True)
class DeepSeekV4EndpointWorkspace[TensorT]:
    streams: TensorT
    collapsed: TensorT
    normalized: TensorT
    head_output: TensorT
    logits: TensorT


@dataclass(frozen=True, slots=True)
class DeepSeekV4TEP4EndpointWorkspace[TensorT]:
    streams: TensorT
    collapsed: TensorT
    normalized: TensorT
    logits: TensorT
    local_values: TensorT
    local_indices: TensorT
    local_candidates: TensorT
    gathered_candidates: TensorT
    best_values: TensorT
    select: TensorT
    tokens: TensorT


@dataclass(frozen=True, slots=True)
class _DeepSeekV4Descriptors[TensorT]:
    storage: TensorT
    positions: TensorT
    request_indices: TensorT
    raw_slots: TensorT
    raw_indices: TensorT
    raw_lengths: TensorT
    c4_slots: TensorT
    c4_outputs: TensorT
    c4_table: TensorT
    c4_base: TensorT
    c128_slots: TensorT
    c128_outputs: TensorT
    c128_table: TensorT
    c128_base: TensorT
    block_table: TensorT
    c4_lengths: TensorT
    c128_lengths: TensorT


@dataclass(frozen=True, slots=True)
class _DeepSeekV4CompressionScratch[TensorT]:
    kv_score: TensorT


@dataclass(frozen=True, slots=True)
class _DeepSeekV4IndexScratch[TensorT]:
    query: TensorT
    scale: TensorT
    weights: TensorT


@dataclass(frozen=True, slots=True)
class _DeepSeekV4ExecutionResources[TensorT]:
    workspace_a: attention.DeepSeekV4DecodeLayerWorkspace[TensorT]
    workspace_b: attention.DeepSeekV4DecodeLayerWorkspace[TensorT]
    workspace_c4: attention.DeepSeekV4DecodeLayerWorkspace[TensorT]
    index: _DeepSeekV4IndexScratch[TensorT]
    core: deepseek_v4_flash.DeepSeekV4DecodeWorkspace[TensorT]
    c4_main: _DeepSeekV4CompressionScratch[TensorT]
    c4_index: _DeepSeekV4CompressionScratch[TensorT]
    c128_main: _DeepSeekV4CompressionScratch[TensorT]
    raw_query: TensorT
    endpoint: (
        DeepSeekV4EndpointWorkspace[TensorT] | DeepSeekV4TEP4EndpointWorkspace[TensorT]
    )


@dataclass(frozen=True, slots=True)
class _DeepSeekV4DecodeRuntime[TensorT]:
    execution_tables: TensorT
    token_ids: TensorT
    layer_inputs: tuple[DeepSeekV4DecodeLayerInputs[TensorT], ...]
    endpoint: (
        DeepSeekV4EndpointWorkspace[TensorT] | DeepSeekV4TEP4EndpointWorkspace[TensorT]
    )
    _descriptors: _DeepSeekV4Descriptors[TensorT]


def endpoint_workspace_shapes(
    batch_size: int,
) -> DeepSeekV4EndpointWorkspace[deepseek_v4_flash.Shape]:
    verify_sizes = {
        groups * deepseek_v4_flash.DSPARK_VERIFY_WIDTH
        for groups in deepseek_v4_flash.DECODE_BATCH_SIZES
    }
    if not (
        0 <= batch_size <= deepseek_v4_flash.MAX_PREFILL_REQUESTS
        or batch_size in deepseek_v4_flash.DECODE_BATCH_SIZES
        or batch_size in verify_sizes
    ):
        raise ValueError("batch_size must be a configured execution bucket")
    return DeepSeekV4EndpointWorkspace(
        streams=(batch_size, deepseek_v4_flash.MHC_STREAMS, deepseek_v4_flash.HIDDEN_SIZE),
        collapsed=(batch_size, deepseek_v4_flash.HIDDEN_SIZE),
        normalized=(batch_size, deepseek_v4_flash.HIDDEN_SIZE),
        head_output=(batch_size, deepseek_v4_flash.VOCAB_SIZE),
        logits=(batch_size, deepseek_v4_flash.VOCAB_SIZE),
    )


def tep4_endpoint_workspace_shapes(
    batch_size: int,
) -> DeepSeekV4TEP4EndpointWorkspace[deepseek_v4_flash.Shape]:
    endpoint_workspace_shapes(batch_size)
    local_vocab = deepseek_v4_flash.VOCAB_SIZE // deepseek_v4_flash.TEP_SIZE
    return DeepSeekV4TEP4EndpointWorkspace(
        streams=(batch_size, deepseek_v4_flash.MHC_STREAMS, deepseek_v4_flash.HIDDEN_SIZE),
        collapsed=(batch_size, deepseek_v4_flash.HIDDEN_SIZE),
        normalized=(batch_size, deepseek_v4_flash.HIDDEN_SIZE),
        logits=(batch_size, local_vocab),
        local_values=(batch_size,),
        local_indices=(batch_size,),
        local_candidates=(batch_size, 2),
        gathered_candidates=(deepseek_v4_flash.TEP_SIZE * batch_size, 2),
        best_values=(batch_size,),
        select=(batch_size,),
        tokens=(batch_size,),
    )


def allocate_target_runtime(
    weights: DeepSeekV4TargetWeights[object],
    live_slots: int,
    history_blocks: int,
    device: str | int,
    *,
    snapshot_slot_count: int = 0,
    speculative: bool = True,
    tensor_parallel: bool = False,
    metadata_allocator: Callable[[], object] | None = None,
    mega_buffer_allocator: Callable[[], object] | None = None,
) -> DeepSeekV4TargetRuntime[object]:
    import deep_gemm
    import torch

    if live_slots <= 0 or history_blocks <= 0:
        raise ValueError("DeepSeek runtime capacities must be positive")
    if type(snapshot_slot_count) is not int or snapshot_slot_count < 0:
        raise ValueError("DeepSeek snapshot slot count must be non-negative")
    if type(speculative) is not bool:
        raise TypeError("DeepSeek speculative mode must be a bool")
    if [layer.layer_id for layer in weights.layers] != list(
        range(deepseek_v4_flash.NUM_LAYERS)
    ):
        raise ValueError("DeepSeek V4 target weights must contain every layer")
    if metadata_allocator is None:
        import flash_mla

        metadata_allocator = lambda: flash_mla.get_mla_metadata()[0]
    if mega_buffer_allocator is None:
        mega_buffer_allocator = lambda: deep_gemm.get_symm_buffer_for_mega_moe(
            torch.distributed.group.WORLD,
            deepseek_v4_flash.NUM_ROUTED_EXPERTS,
            deepseek_v4_flash.PREFILL_CHUNK_TOKENS,
            deepseek_v4_flash.TOP_K,
            deepseek_v4_flash.HIDDEN_SIZE,
            deepseek_v4_flash.EXPERT_INTERMEDIATE_SIZE,
            mma_type="fp8xfp4",
        )

    shapes = deepseek_v4_flash.TARGET_STATE_SLOT_SHAPES
    state_slots = live_slots + DECODE_DUMMY_SLOTS

    def state_arena(name, dtype):
        shape = getattr(shapes, name)
        return torch.empty(
            (shape[0], state_slots, *shape[1:]), dtype=dtype, device=device
        )

    state = deepseek_v4_flash.DeepSeekV4TargetState(
        state_arena("raw_window", torch.uint8),
        state_arena("c4_main", torch.float32),
        state_arena("c4_index", torch.float32),
        state_arena("c128_main", torch.float32),
    )
    history = tuple(
        tuple(
            torch.empty(
                (history_blocks, layout.physical_storage_bytes),
                dtype=torch.uint8,
                device=device,
            )
            for _ in layer_ids
        )
        if layout is deepseek_v4_flash.C4_FP4_INDEX_HISTORY_CANDIDATE
        else torch.empty(
            (len(layer_ids), history_blocks, layout.physical_storage_bytes),
            dtype=torch.uint8,
            device=device,
        )
        for layer_ids, layout in (
            (deepseek_v4_flash.CSA_LAYER_IDS, deepseek_v4_flash.C4_FP8_MAIN_HISTORY_CANDIDATE),
            (deepseek_v4_flash.CSA_LAYER_IDS, deepseek_v4_flash.C4_FP4_INDEX_HISTORY_CANDIDATE),
            (deepseek_v4_flash.HCA_LAYER_IDS, deepseek_v4_flash.C128_FP8_MAIN_HISTORY_CANDIDATE),
        )
    )
    cos_sin = (
        _rope_table(torch, device, compressed=False),
        _rope_table(torch, device, compressed=True),
    )
    table_width = deepseek_v4_flash.MAX_CONTEXT_TOKENS // 128
    sampled_tokens = torch.zeros(state_slots, dtype=torch.int64, device=device)
    committed_lengths = torch.zeros(state_slots, dtype=torch.int32, device=device)
    result_records = torch.zeros(
        (state_slots, 1 + (deepseek_v4_flash.DSPARK_VERIFY_WIDTH if speculative else 1)),
        dtype=torch.int64,
        device=device,
    )
    execution_tables = torch.full(
        (state_slots, table_width), -1, dtype=torch.int32, device=device
    )
    parity_count = deepseek_v4_flash.DECODE_DESCRIPTOR_PARITIES
    mega_buffer = mega_buffer_allocator()
    num_sms = deep_gemm.get_num_sms()
    prefill_resources = _allocate_prefill_resources(
        num_sms, device, torch, mega_buffer, tensor_parallel
    )
    prefill_bindings = _bind_prefill_layers(
        weights, state, history, prefill_resources, state_slots
    )
    prefill_staging = tuple(
        _allocate_prefill_staging(torch, device) for _ in range(parity_count)
    )
    verify = {}
    decode = {}
    for groups in (0, *deepseek_v4_flash.DECODE_BATCH_SIZES):
        batch_size = groups * deepseek_v4_flash.DSPARK_VERIFY_WIDTH if speculative else groups
        execution = _allocate_execution_resources(
            batch_size, num_sms, device, torch, tensor_parallel
        )
        for parity in range(parity_count):
            metadata = {kind: metadata_allocator() for kind in ("swa", "csa", "hca")}
            target = _allocate_decode_runtime(
                weights,
                state,
                history,
                cos_sin,
                state_slots,
                device,
                torch,
                batch_size,
                execution,
                metadata,
                mega_buffer,
                execution_tables,
            )
            if speculative:
                verify[groups, parity] = _allocate_verify_runtime(
                    target,
                    groups,
                    committed_lengths,
                    sampled_tokens,
                    result_records,
                    live_slots,
                    device,
                    torch,
                )
            else:
                decode[groups, parity] = _allocate_target_decode_runtime(
                    target,
                    groups,
                    committed_lengths,
                    sampled_tokens,
                    result_records,
                    live_slots,
                    device,
                    torch,
                )
    return DeepSeekV4TargetRuntime(
        state,
        sampled_tokens,
        committed_lengths,
        result_records,
        execution_tables,
        verify,
        decode,
        prefill_bindings,
        prefill_resources,
        prefill_staging,
        cos_sin,
        live_slots,
        history_blocks,
        history,
        _allocate_target_prefix_snapshots(torch, device, snapshot_slot_count),
    )


def _allocate_target_prefix_snapshots(torch, device, snapshot_slot_count):
    if not snapshot_slot_count:
        return None
    shapes = deepseek_v4_flash.TARGET_STATE_SLOT_SHAPES

    def state_arena(name, dtype):
        shape = getattr(shapes, name)
        return torch.empty(
            (shape[0], snapshot_slot_count, *shape[1:]),
            dtype=dtype,
            device=device,
        )

    state = deepseek_v4_flash.DeepSeekV4TargetState(
        state_arena("raw_window", torch.uint8),
        state_arena("c4_main", torch.float32),
        state_arena("c4_index", torch.float32),
        state_arena("c128_main", torch.float32),
    )
    c4_main = deepseek_v4_flash.C4_FP8_MAIN_HISTORY_CANDIDATE
    c4_index = deepseek_v4_flash.C4_FP4_INDEX_HISTORY_CANDIDATE
    c128_main = deepseek_v4_flash.C128_FP8_MAIN_HISTORY_CANDIDATE
    return DeepSeekV4TargetPrefixSnapshots(
        state,
        torch.empty(
            (
                len(deepseek_v4_flash.CSA_LAYER_IDS),
                snapshot_slot_count,
                c4_main.physical_storage_bytes,
            ),
            dtype=torch.uint8,
            device=device,
        ),
        tuple(
            torch.empty(
                (snapshot_slot_count, c4_index.physical_storage_bytes),
                dtype=torch.uint8,
                device=device,
            )
            for _ in deepseek_v4_flash.CSA_LAYER_IDS
        ),
        torch.empty(
            (
                len(deepseek_v4_flash.HCA_LAYER_IDS),
                snapshot_slot_count,
                c128_main.physical_storage_bytes,
            ),
            dtype=torch.uint8,
            device=device,
        ),
        torch.empty(snapshot_slot_count, dtype=torch.int32, device=device),
    )


def _allocate_verify_runtime(
    execution,
    groups,
    committed_lengths,
    sampled_tokens,
    result_records,
    live_slots,
    device,
    torch,
):
    width = deepseek_v4_flash.DSPARK_VERIFY_WIDTH

    def empty(shape, dtype=torch.int64):
        return torch.empty(shape, dtype=dtype, device=device)

    def controls(where, pin_memory=False):
        return _DeepSeekV4VerifyControls(
            torch.empty(
                groups,
                dtype=torch.int64,
                device=where,
                pin_memory=pin_memory,
            ),
            torch.empty(
                (groups, 2),
                dtype=torch.int32,
                device=where,
                pin_memory=pin_memory,
            ),
            torch.empty(
                groups,
                dtype=torch.int32,
                device=where,
                pin_memory=pin_memory,
            ),
            torch.empty(
                groups,
                dtype=torch.uint8,
                device=where,
                pin_memory=pin_memory,
            ),
            torch.empty(
                groups,
                dtype=torch.uint8,
                device=where,
                pin_memory=pin_memory,
            ),
        )

    pin_memory = device != "meta" and torch.cuda.is_available()
    return DeepSeekV4VerifyRuntime(
        execution,
        empty((groups, width)),
        empty((groups, width)),
        empty((groups, width, _DSPARK_CAPTURE_WIDTH), torch.bfloat16),
        empty((groups, width - 1), torch.bool),
        empty(groups, torch.bool),
        empty(groups, torch.int32),
        empty((groups, width + 1)),
        empty((groups, 1)),
        empty((groups, 1)),
        empty(groups, torch.int32),
        _DeepSeekV4VerifyStaging(
            controls(device),
            controls("cpu", pin_memory),
        ),
        committed_lengths,
        sampled_tokens,
        result_records,
        live_slots,
    )


def _allocate_target_decode_runtime(
    execution,
    rows,
    committed_lengths,
    sampled_tokens,
    result_records,
    live_slots,
    device,
    torch,
):
    def controls(where, pin_memory=False):
        return _DeepSeekV4TargetDecodeControls(
            torch.empty(rows, dtype=torch.int64, device=where, pin_memory=pin_memory),
            torch.empty(
                (rows, 2), dtype=torch.int32, device=where, pin_memory=pin_memory
            ),
            torch.empty(rows, dtype=torch.uint8, device=where, pin_memory=pin_memory),
        )

    pin_memory = device != "meta" and torch.cuda.is_available()
    return DeepSeekV4TargetDecodeRuntime(
        execution,
        torch.empty(rows, dtype=torch.int64, device=device),
        torch.empty((rows, 2), dtype=torch.int64, device=device),
        torch.empty(rows, dtype=torch.int32, device=device),
        _DeepSeekV4TargetDecodeStaging(
            controls(device),
            controls("cpu", pin_memory),
        ),
        committed_lengths,
        sampled_tokens,
        result_records,
        live_slots,
    )


def _allocate_execution_resources(
    batch_size, num_sms, device, torch, tensor_parallel=False
):
    def empty(shape, dtype=torch.bfloat16):
        return torch.empty(shape, dtype=dtype, device=device)

    query_heads = (
        deepseek_v4_flash.LOCAL_QUERY_HEADS
        if tensor_parallel
        else deepseek_v4_flash.NUM_QUERY_HEADS
    )
    layer_shapes = attention.decode_layer_workspace_shapes(
        2, batch_size, query_heads=query_heads
    )
    workspace_a, workspace_b, workspace_c4 = _allocate_layer_workspaces(
        layer_shapes, device, torch
    )

    core_shapes = deepseek_v4_flash.DeepSeekV4DecodeWorkspace(
        (num_sms + 1, 2),
        (batch_size, deepseek_v4_flash.INDEX_CONTEXT_CAPACITY),
        (deepseek_v4_flash.TOPK_ROW_STATES_BYTES,),
        (batch_size, 1, deepseek_v4_flash.INDEX_TOP_K),
        (batch_size,),
    )
    core = _allocate_shape_record(
        core_shapes,
        torch,
        device,
        fp32={"logits"},
        int32=set(core_shapes.__dataclass_fields__) - {"logits", "topk_rows"},
        uint8={"topk_rows"},
    )
    core.schedule.zero_()
    core.topk_rows.zero_()

    def compression(width):
        return _DeepSeekV4CompressionScratch(
            empty((batch_size, 2 * width), torch.float32)
        )

    raw_query = workspace_a.attention.projected_q.view(
        batch_size, 1, query_heads, deepseek_v4_flash.HEAD_DIM
    )
    if tensor_parallel:
        raw_query = empty(
            (batch_size, 1, deepseek_v4_flash.NUM_QUERY_HEADS, deepseek_v4_flash.HEAD_DIM)
        ).zero_()
    return _DeepSeekV4ExecutionResources(
        workspace_a,
        workspace_b,
        workspace_c4,
        _DeepSeekV4IndexScratch(
            empty((batch_size, 1, deepseek_v4_flash.NUM_QUERY_HEADS, 64), torch.int8),
            empty((batch_size, 1, deepseek_v4_flash.NUM_QUERY_HEADS), torch.int32),
            empty((batch_size, deepseek_v4_flash.NUM_QUERY_HEADS), torch.float32),
        ),
        core,
        compression(1_024),
        compression(256),
        compression(512),
        raw_query,
        _allocate_endpoint_workspace(batch_size, tensor_parallel, torch, device),
    )


def _allocate_layer_workspaces(layer_shapes, device, torch):
    def empty(shape, dtype=torch.bfloat16):
        return torch.empty(shape, dtype=dtype, device=device)

    common_shapes = layer_shapes.attention.common
    packed_q = empty(layer_shapes.attention.packed_q)
    common = _allocate_shape_record(
        common_shapes,
        torch,
        device,
        fp8={"hidden_fp8", "q_residual_fp8", "output_fp8"},
        fp32={"hidden_fp32", "hidden_scale", "q_residual_scale", "output_scale"},
    )
    c4_attention = attention.DeepSeekV4C4AttentionWorkspace(
        common, packed_q, empty(layer_shapes.attention.index_weights)
    )
    ffn = _allocate_shape_record(
        layer_shapes.ffn,
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
            getattr(layer_shapes, field.name),
            torch.float32 if field.name in fp32 else torch.bfloat16,
        )
        for field in fields(layer_shapes)
        if field.name not in {"attention", "ffn", "streams_out"}
    }
    streams_a = empty(layer_shapes.streams_out)
    streams_b = torch.empty_like(streams_a)
    workspace_a = attention.DeepSeekV4DecodeLayerWorkspace(
        common, ffn, **shared, streams_out=streams_a
    )
    workspace_b = attention.DeepSeekV4DecodeLayerWorkspace(
        common, ffn, **shared, streams_out=streams_b
    )
    workspace_c4 = attention.DeepSeekV4DecodeLayerWorkspace(
        c4_attention, ffn, **shared, streams_out=streams_a
    )
    return workspace_a, workspace_b, workspace_c4


def _allocate_prefill_resources(
    num_sms, device, torch, mega_buffer, tensor_parallel=False
):
    def empty(shape, dtype=torch.bfloat16):
        return torch.empty(shape, dtype=dtype, device=device)

    tokens = deepseek_v4_flash.PREFILL_CHUNK_TOKENS
    query_heads = (
        deepseek_v4_flash.LOCAL_QUERY_HEADS
        if tensor_parallel
        else deepseek_v4_flash.NUM_QUERY_HEADS
    )
    layer_shapes = attention.prefill_layer_workspace_shapes(
        2, tokens, query_heads=query_heads
    )
    workspace_a, workspace_b, workspace_c4 = _allocate_layer_workspaces(
        layer_shapes, device, torch
    )

    def compression(width):
        return _DeepSeekV4CompressionScratch(empty((tokens, 2 * width), torch.float32))

    index_rows = 512
    table_width = deepseek_v4_flash.MAX_CONTEXT_TOKENS // 128
    staged = deepseek_v4_flash.DeepSeekV4IndexQuery(
        empty((index_rows, 1, deepseek_v4_flash.NUM_QUERY_HEADS, 64), torch.int8),
        empty((index_rows, 1, deepseek_v4_flash.NUM_QUERY_HEADS), torch.int32),
        empty((index_rows, deepseek_v4_flash.NUM_QUERY_HEADS), torch.float32),
        empty((index_rows, table_width), torch.int32),
        empty((index_rows, 1), torch.int32).zero_(),
    )
    index = deepseek_v4_flash.DeepSeekV4PrefillIndexWorkspace(
        staged,
        empty(index_rows, torch.int32).zero_(),
        empty((num_sms + 1, 2), torch.int32).zero_(),
        empty((index_rows, deepseek_v4_flash.INDEX_CONTEXT_CAPACITY), torch.float32),
        empty((tokens, deepseek_v4_flash.INDEX_TOP_K), torch.int32),
        empty(index_rows, torch.int32).zero_(),
        empty(deepseek_v4_flash.TOPK_ROW_STATES_BYTES, torch.uint8).zero_(),
    )
    attention_rows = (
        deepseek_v4_flash.MAX_PREFILL_REQUESTS
        * (deepseek_v4_flash.INDEX_CONTEXT_CAPACITY + deepseek_v4_flash.SWA_WINDOW_TOKENS)
        + tokens
    )
    return _DeepSeekV4PrefillResources(
        workspace_a,
        workspace_b,
        workspace_c4,
        compression(1_024),
        compression(256),
        compression(512),
        empty((_C4_STATE_PAGES, 4, 2_048), torch.float32),
        empty((_C4_STATE_PAGES, 4, 512), torch.float32),
        empty((_C128_STATE_PAGES, 8, 1_024), torch.float32),
        empty((8, 4, 2_048), torch.float32),
        empty((8, 4, 512), torch.float32),
        empty((64, 8, 1_024), torch.float32),
        empty((tokens, 1, _C128_SELECTED_WIDTH), torch.int32),
        empty(tokens, torch.int32),
        (
            empty(
                (tokens, 1, deepseek_v4_flash.NUM_QUERY_HEADS, deepseek_v4_flash.HEAD_DIM)
            ).zero_()
            if tensor_parallel
            else None
        ),
        deepseek_v4_flash.DeepSeekV4PrefillWorkspace(
            empty((attention_rows, 1, deepseek_v4_flash.HEAD_DIM)),
            empty(tokens, torch.int32),
        ),
        index,
        _allocate_endpoint_workspace(
            deepseek_v4_flash.MAX_PREFILL_REQUESTS, tensor_parallel, torch, device
        ),
        empty(deepseek_v4_flash.MAX_PREFILL_REQUESTS, torch.int64),
        empty(tokens, torch.int64) if tensor_parallel else None,
        empty(tokens, torch.bool) if tensor_parallel else None,
        empty(
            (
                _DSPARK_PREFILL_ROWS,
                deepseek_v4_flash.MHC_STREAMS,
                deepseek_v4_flash.HIDDEN_SIZE,
            )
        ),
        empty((_DSPARK_PREFILL_ROWS, _DSPARK_CAPTURE_WIDTH)),
        mega_buffer,
    )


def _bind_prefill_layers(weights, state, history, resources, state_slots):
    bindings = []
    for layer_weights in weights.layers:
        layer_id = layer_weights.layer_id
        kind = deepseek_v4_flash.attention_kind(layer_id)
        projection = (
            layer_weights.attention
            if kind == "swa"
            else layer_weights.attention.projection
        )
        main = index = persistent_main = persistent_index = None
        if kind == "csa":
            compressed = layer_id // 2 - 1
            main = deepseek_v4_flash.DeepSeekV4Compression(
                resources.c4_main.kv_score,
                layer_weights.attention.main_ape,
                resources.c4_main_state,
                layer_weights.attention.main_norm,
                history[0][compressed],
            )
            index = deepseek_v4_flash.DeepSeekV4Compression(
                resources.c4_index.kv_score,
                layer_weights.attention.index_ape,
                resources.c4_index_state,
                layer_weights.attention.index_norm,
                history[1][compressed],
            )
            persistent_main = state.c4_main[compressed].reshape(
                state_slots * _C4_RING_PAGES, 4, -1
            )
            persistent_index = state.c4_index[compressed].reshape(
                state_slots * _C4_RING_PAGES, 4, -1
            )
        elif kind == "hca":
            compressed = (layer_id - 3) // 2
            main = deepseek_v4_flash.DeepSeekV4Compression(
                resources.c128_main.kv_score,
                layer_weights.attention.main_ape,
                resources.c128_state,
                layer_weights.attention.main_norm,
                history[2][compressed],
            )
            persistent_main = state.c128_main[compressed].reshape(
                state_slots * _C128_RING_PAGES, 8, -1
            )
        bindings.append(
            _DeepSeekV4PrefillLayerBinding(
                main,
                index,
                persistent_main,
                persistent_index,
                state.raw_window[layer_id].reshape(state_slots * _RAW_RING_PAGES, -1),
                projection.sink,
            )
        )
    return tuple(bindings)


def _prefill_layer_views(runtime, batch, layer_id):
    tokens = batch.token_count
    kind = deepseek_v4_flash.attention_kind(layer_id)
    resources = runtime.resources
    workspace_source = (
        resources.workspace_c4
        if kind == "csa"
        else resources.workspace_a
        if layer_id % 2 == 0
        else resources.workspace_b
    )
    workspace = _view_shape_record(
        workspace_source,
        attention.prefill_layer_workspace_shapes(
            layer_id,
            tokens,
            query_heads=(
                deepseek_v4_flash.LOCAL_QUERY_HEADS
                if resources.raw_query is not None
                else deepseek_v4_flash.NUM_QUERY_HEADS
            ),
        ),
    )
    if not tokens:
        return (None,) * 7 + (workspace, None, None, resources.mega_buffer)

    requests = batch.request_count
    descriptors = runtime.staging[batch.staging_index].device
    block_table = _view_tensor(descriptors.block_table, (requests, batch.table_width))
    metadata = deepseek_v4_flash.DeepSeekV4PrefillMetadata(
        descriptors.query_start[: requests + 1],
        descriptors.prefix_lengths[:requests],
        descriptors.state_slots[:requests],
        block_table,
        descriptors.block_table_base[:requests],
    )

    ratio = 1 if kind == "swa" else 4 if kind == "csa" else 128
    prefix = "c4" if ratio in (1, 4) else "c128"
    state_width = batch.c4_state_width if prefix == "c4" else batch.c128_state_width
    compression_batch = deepseek_v4_flash.DeepSeekV4CompressionBatch(
        descriptors.positions[:tokens],
        descriptors.request_indices[:tokens],
        getattr(descriptors, f"{prefix}_slots")[:tokens],
        getattr(descriptors, f"{prefix}_outputs")[:tokens],
        _view_tensor(
            getattr(descriptors, f"{prefix}_table"),
            (requests, state_width),
        ),
        getattr(descriptors, f"{prefix}_base")[:requests],
        runtime.cos_sin[ratio != 1],
    )

    common = workspace.attention
    if isinstance(common, attention.DeepSeekV4C4AttentionWorkspace):
        common = common.common
    selected_width = (
        deepseek_v4_flash.INDEX_TOP_K + deepseek_v4_flash.SWA_WINDOW_TOKENS
        if ratio == 4
        else (batch.table_width if ratio == 128 else 0) + deepseek_v4_flash.SWA_WINDOW_TOKENS
    )
    selected_width = (selected_width + 127) // 128 * 128
    raw = deepseek_v4_flash.DeepSeekV4RawAttention(
        (
            resources.raw_query[:tokens]
            if resources.raw_query is not None
            else common.projected_q.view(
                tokens, 1, deepseek_v4_flash.NUM_QUERY_HEADS, deepseek_v4_flash.HEAD_DIM
            )
        ),
        runtime.bindings[layer_id].raw_cache,
        _view_tensor(resources.raw_indices, (tokens, 1, selected_width)),
        resources.raw_lengths[:tokens],
        runtime.bindings[layer_id].sink,
    )
    compressed_capacity = (
        0 if ratio == 1 else batch.table_width * (32 if ratio == 4 else 1)
    )
    rows = requests * (compressed_capacity + deepseek_v4_flash.SWA_WINDOW_TOKENS) + tokens
    attention_workspace = deepseek_v4_flash.DeepSeekV4PrefillWorkspace(
        resources.attention.kv[:rows], resources.attention.raw_slots[:tokens]
    )

    binding = runtime.bindings[layer_id]
    main = _view_compression(binding.main, tokens)
    index = _view_compression(binding.index, tokens)
    main_state = index_state = None
    index_workspace = None
    if ratio == 4:
        main_state = _prefill_state_view(
            binding.persistent_main,
            descriptors,
            resources.c4_main_transfer,
            batch,
            "c4",
        )
        index_state = _prefill_state_view(
            binding.persistent_index,
            descriptors,
            resources.c4_index_transfer,
            batch,
            "c4",
        )
        index_workspace = _prefill_index_view(
            resources.index, tokens, batch.table_width
        )
    elif ratio == 128:
        main_state = _prefill_state_view(
            binding.persistent_main,
            descriptors,
            resources.c128_transfer,
            batch,
            "c128",
        )
    return (
        compression_batch,
        main,
        index,
        raw,
        metadata,
        index_workspace,
        attention_workspace,
        workspace,
        main_state,
        index_state,
        resources.mega_buffer,
    )


def _view_compression(compression, tokens):
    if compression is None:
        return None
    return deepseek_v4_flash.DeepSeekV4Compression(
        compression.kv_score[:tokens],
        compression.ape,
        compression.state,
        compression.norm_weight,
        compression.cache,
    )


def _prefill_state_view(persistent, descriptors, transfer, batch, prefix):
    seed_count = getattr(batch, f"{prefix}_seed_count")
    retain_count = getattr(batch, f"{prefix}_retain_count")
    transfer_count = max(seed_count, retain_count)
    return deepseek_v4_flash.DeepSeekV4PrefillState(
        persistent,
        getattr(descriptors, f"{prefix}_seed_source")[:seed_count],
        getattr(descriptors, f"{prefix}_seed_destination")[:seed_count],
        getattr(descriptors, f"{prefix}_retain_source")[:retain_count],
        getattr(descriptors, f"{prefix}_retain_destination")[:retain_count],
        transfer[:transfer_count],
    )


def _prefill_index_view(workspace, tokens, table_width):
    capacity = (table_width * 32 + 255) // 256 * 256
    staged = workspace.staged
    return deepseek_v4_flash.DeepSeekV4PrefillIndexWorkspace(
        deepseek_v4_flash.DeepSeekV4IndexQuery(
            staged.query,
            staged.scale,
            staged.weights,
            _view_tensor(staged.block_table, (512, table_width)),
            staged.lengths,
        ),
        workspace.request_indices,
        workspace.schedule,
        _view_tensor(workspace.logits, (512, capacity)),
        workspace.candidates[:tokens],
        workspace.topk_offsets,
        workspace.topk_rows,
    )


def _view_shape_record(values, shapes):
    return type(values)(
        **{
            field.name: (
                _view_tensor(getattr(values, field.name), expected)
                if isinstance(expected := getattr(shapes, field.name), tuple)
                else _view_shape_record(getattr(values, field.name), expected)
            )
            for field in fields(shapes)
        }
    )


def _view_tensor(tensor, shape):
    return tensor.view(-1)[: prod(shape)].view(shape)


def _allocate_prefill_staging(torch, device):
    tokens = deepseek_v4_flash.PREFILL_CHUNK_TOKENS
    requests = deepseek_v4_flash.MAX_PREFILL_REQUESTS
    table_width = deepseek_v4_flash.MAX_CONTEXT_TOKENS // 128

    def allocate(target, pin_memory=False):
        def empty(shape, dtype=torch.int32):
            return torch.empty(
                shape,
                dtype=dtype,
                device=target,
                pin_memory=pin_memory,
            )

        return _DeepSeekV4PrefillDescriptors(
            empty(tokens, torch.int64),
            empty(tokens),
            empty(tokens),
            empty(requests + 1),
            empty(requests),
            empty(requests),
            empty((requests, table_width)),
            empty(requests),
            empty(tokens),
            empty(tokens),
            empty((requests, _C4_STATE_TABLE_WIDTH)),
            empty(requests),
            empty(tokens),
            empty(tokens),
            empty((requests, _C128_STATE_TABLE_WIDTH)),
            empty(requests),
            empty(8, torch.int64),
            empty(8, torch.int64),
            empty(8, torch.int64),
            empty(8, torch.int64),
            empty(64, torch.int64),
            empty(64, torch.int64),
            empty(64, torch.int64),
            empty(64, torch.int64),
            empty(requests, torch.int64),
            empty(requests, torch.int64),
            empty(requests),
            empty(_DSPARK_PREFILL_ROWS),
            empty(_DSPARK_PREFILL_ROWS),
            empty(_DSPARK_PREFILL_ROWS),
            empty(requests),
        )

    pin_memory = device != "meta" and torch.cuda.is_available()
    return _DeepSeekV4PrefillStaging(
        allocate(device),
        allocate("cpu", pin_memory),
    )


def _allocate_decode_runtime(
    weights,
    state,
    history,
    cos_sin,
    state_slots,
    device,
    torch,
    batch_size,
    execution,
    metadata,
    mega_buffer,
    execution_tables,
):
    def empty(shape, dtype=torch.bfloat16):
        return torch.empty(shape, dtype=dtype, device=device)

    table_width = deepseek_v4_flash.MAX_CONTEXT_TOKENS // 128
    descriptor_shapes = {
        "positions": (batch_size,),
        "request_indices": (batch_size,),
        "raw_slots": (batch_size,),
        "raw_indices": (batch_size, 1, 128),
        "raw_lengths": (batch_size,),
        "c4_slots": (batch_size,),
        "c4_outputs": (batch_size,),
        "c4_table": (batch_size, 3),
        "c4_base": (batch_size,),
        "c128_slots": (batch_size,),
        "c128_outputs": (batch_size,),
        "c128_table": (batch_size, 17),
        "c128_base": (batch_size,),
        "block_table": (batch_size, table_width),
        "c4_lengths": (batch_size, 1),
        "c128_lengths": (batch_size,),
    }
    descriptor_size = sum(
        _align_descriptor_offset(prod(shape)) for shape in descriptor_shapes.values()
    )
    descriptors = _bind_descriptors(
        empty(descriptor_size, torch.int32),
        descriptor_shapes,
    )
    descriptors.storage.fill_(-1)
    for tensor in (
        descriptors.positions,
        descriptors.raw_lengths,
        descriptors.c4_base,
        descriptors.c4_lengths,
        descriptors.c128_base,
        descriptors.c128_lengths,
    ):
        tensor.zero_()
    token_ids = empty(batch_size, torch.int64).zero_()

    def batch(prefix, rope):
        return deepseek_v4_flash.DeepSeekV4CompressionBatch(
            descriptors.positions,
            descriptors.request_indices,
            getattr(descriptors, f"{prefix}_slots"),
            getattr(descriptors, f"{prefix}_outputs"),
            getattr(descriptors, f"{prefix}_table"),
            getattr(descriptors, f"{prefix}_base"),
            rope,
        )

    batch_swa = batch("c4", cos_sin[0])
    batch_c4 = batch("c4", cos_sin[1])
    batch_c128 = batch("c128", cos_sin[1])
    query = deepseek_v4_flash.DeepSeekV4IndexQuery(
        execution.index.query,
        execution.index.scale,
        execution.index.weights,
        descriptors.block_table,
        descriptors.c4_lengths,
    )
    pool = deepseek_v4_flash.DeepSeekV4AttentionPool(
        descriptors.block_table.unsqueeze(1), descriptors.c128_lengths
    )
    raw_query = execution.raw_query

    def compression(scratch, ape, state, norm, cache):
        return deepseek_v4_flash.DeepSeekV4Compression(
            scratch.kv_score, ape, state, norm, cache
        )

    layer_inputs = []
    for layer_weights in weights.layers:
        layer_id = layer_weights.layer_id
        kind = deepseek_v4_flash.attention_kind(layer_id)
        projection = (
            layer_weights.attention
            if kind == "swa"
            else layer_weights.attention.projection
        )
        raw = deepseek_v4_flash.DeepSeekV4RawAttention(
            raw_query,
            state.raw_window[layer_id].reshape(state_slots * _RAW_RING_PAGES, -1),
            descriptors.raw_indices,
            descriptors.raw_lengths,
            projection.sink,
        )
        main = None
        attention_state = None
        batch = batch_swa
        workspace = (
            execution.workspace_a if layer_id % 2 == 0 else execution.workspace_b
        )
        if kind == "csa":
            compressed = layer_id // 2 - 1
            main = compression(
                execution.c4_main,
                layer_weights.attention.main_ape,
                state.c4_main[compressed].reshape(state_slots * _C4_RING_PAGES, 4, -1),
                layer_weights.attention.main_norm,
                history[0][compressed],
            )
            index = compression(
                execution.c4_index,
                layer_weights.attention.index_ape,
                state.c4_index[compressed].reshape(state_slots * _C4_RING_PAGES, 4, -1),
                layer_weights.attention.index_norm,
                history[1][compressed],
            )
            attention_state = attention.DeepSeekV4C4DecodeState(
                index,
                query,
                execution.core,
                layer_id == deepseek_v4_flash.CSA_LAYER_IDS[0],
            )
            batch = batch_c4
            workspace = execution.workspace_c4
        elif kind == "hca":
            compressed = (layer_id - 3) // 2
            main = compression(
                execution.c128_main,
                layer_weights.attention.main_ape,
                state.c128_main[compressed].reshape(
                    state_slots * _C128_RING_PAGES, 8, -1
                ),
                layer_weights.attention.main_norm,
                history[2][compressed],
            )
            attention_state = attention.DeepSeekV4C128DecodeState(pool)
            batch = batch_c128
        layer_inputs.append(
            DeepSeekV4DecodeLayerInputs(
                batch,
                descriptors.raw_slots,
                main,
                raw,
                attention_state,
                workspace,
                metadata[kind],
                mega_buffer,
            )
        )

    return _DeepSeekV4DecodeRuntime(
        execution_tables,
        token_ids,
        tuple(layer_inputs),
        execution.endpoint,
        descriptors,
    )


def _bind_descriptors(storage, shapes):
    views = {"storage": storage}
    offset = 0
    for name, shape in shapes.items():
        offset = _align_descriptor_offset(offset)
        size = prod(shape)
        views[name] = storage[offset : offset + size].view(shape)
        offset += size
    return _DeepSeekV4Descriptors(**views)


def _align_descriptor_offset(offset):
    return (offset + 3) // 4 * 4


def _copy_packed_rows(destination, source, rows, width) -> None:
    packed = destination.view(-1)[: rows * width].view(rows, width)
    for row in range(rows):
        packed[row].copy_(source[row, :width], non_blocking=destination.is_cuda)


def _stage_prefill_state(
    starts,
    lengths,
    slots,
    ratio,
    token_slots,
    state_table,
    table_base,
    seed_source,
    seed_destination,
    retain_source,
    retain_destination,
):
    page_size = _PREFILL_STATE_PAGE[ratio]
    retained_pages = _PREFILL_RETAINED_PAGES[ratio]
    ring_pages = _PREFILL_STATE_RING_PAGES[ratio]
    scratch_page = token = seed_count = retain_count = table_width = 0
    for request, (start, length, slot) in enumerate(
        zip(starts, lengths, slots, strict=True)
    ):
        end = start + length
        first_output = start + (ratio - 1 - start) % ratio
        first_state = (
            max(first_output - (2 * ratio if ratio == 4 else ratio) + 1, 0)
            if first_output < end
            else start
        )
        logical_base = first_state // page_size
        logical_end = (end - 1) // page_size
        pages = logical_end - logical_base + 1
        table_base[request] = logical_base
        for column in range(pages):
            state_table[request, column] = scratch_page + column
        table_width = max(table_width, pages)

        for position in range(start, end):
            token_slots[token] = (
                scratch_page + position // page_size - logical_base
            ) * page_size + position % page_size
            token += 1

        prefix_end = min((start - 1) // page_size, logical_end)
        for logical_page in range(logical_base, prefix_end + 1):
            seed_source[seed_count] = slot * ring_pages + logical_page % ring_pages
            seed_destination[seed_count] = scratch_page + logical_page - logical_base
            seed_count += 1

        retained_start = max(logical_base, logical_end - retained_pages + 1)
        for logical_page in range(retained_start, logical_end + 1):
            retain_source[retain_count] = scratch_page + logical_page - logical_base
            retain_destination[retain_count] = (
                slot * ring_pages + logical_page % ring_pages
            )
            retain_count += 1
        scratch_page += pages
    return scratch_page, table_width, seed_count, retain_count


def _allocate_endpoint_workspace(batch_size, tensor_parallel, torch, device):
    shapes = (
        tep4_endpoint_workspace_shapes if tensor_parallel else endpoint_workspace_shapes
    )(batch_size)
    fp32 = (
        {"local_candidates", "gathered_candidates", "best_values"}
        if tensor_parallel
        else {"logits"}
    )
    return _allocate_shape_record(
        shapes,
        torch,
        device,
        fp32=fp32,
        int64={"local_indices", "tokens"},
        bool_={"select"},
    )


def _allocate_shape_record(
    shapes, torch, device, *, fp8=(), fp32=(), int32=(), int64=(), uint8=(), bool_=()
):
    values = {}
    for field in fields(shapes):
        dtype = torch.bfloat16
        if field.name in fp8:
            dtype = torch.float8_e4m3fn
        elif field.name in fp32:
            dtype = torch.float32
        elif field.name in int32:
            dtype = torch.int32
        elif field.name in int64:
            dtype = torch.int64
        elif field.name in uint8:
            dtype = torch.uint8
        elif field.name in bool_:
            dtype = torch.bool
        values[field.name] = torch.empty(
            getattr(shapes, field.name), dtype=dtype, device=device
        )
    return type(shapes)(**values)


def _rope_table(torch, device, *, compressed, positions=None):
    base = _ROPE_BASES[compressed]
    dimensions = torch.arange(
        0, deepseek_v4_flash.ROPE_HEAD_DIM, 2, dtype=torch.float32, device=device
    )
    frequencies = (base ** (dimensions / deepseek_v4_flash.ROPE_HEAD_DIM)).reciprocal_()
    if compressed:
        low, high = _YARN_CORRECTION_RANGE
        ramp = dimensions.clone().div_(2).sub_(low).div_(high - low).clamp_(0, 1)
        frequencies = frequencies / 16 * ramp + frequencies * (1 - ramp)
    if positions is None:
        positions = torch.arange(
            deepseek_v4_flash.MAX_CONTEXT_TOKENS, dtype=torch.float32, device=device
        )
    else:
        positions = positions.to(dtype=torch.float32, device=device)
    angles = positions.unsqueeze(1) * frequencies.unsqueeze(0)
    return torch.cat((torch.cos(angles), torch.sin(angles)), dim=1).contiguous()


def load_dep4_target_weights(
    view: checkpoint.DeepSeekV4CheckpointView,
    ep_rank: int,
    device: str | int,
) -> DeepSeekV4TargetWeights[object]:
    return _load_target_weights(view, ep_rank, device, tensor_parallel=False)


def load_tep4_target_weights(
    view: checkpoint.DeepSeekV4CheckpointView,
    rank: int,
    device: str | int,
) -> DeepSeekV4TargetWeights[object]:
    return _load_target_weights(view, rank, device, tensor_parallel=True)


def _load_target_weights(
    view: checkpoint.DeepSeekV4CheckpointView,
    rank: int,
    device: str | int,
    *,
    tensor_parallel: bool,
) -> DeepSeekV4TargetWeights[object]:
    def load(name: str) -> object:
        return view.load_target_tensor(name, rank, device, sharded=tensor_parallel)

    def load_mhc(prefix: str) -> attention.DeepSeekV4MHCWeights[object]:
        return attention.DeepSeekV4MHCWeights(
            base=load(f"{prefix}_base"),
            fn=load(f"{prefix}_fn"),
            scale=load(f"{prefix}_scale"),
        )

    load_moe = (
        megamoe.load_tep4_moe_weights
        if tensor_parallel
        else megamoe.load_dep4_moe_weights
    )
    embedding = load("embed.weight")
    layers = []
    for layer_id in range(deepseek_v4_flash.NUM_LAYERS):
        prefix = f"layers.{layer_id}."
        layers.append(
            attention.DeepSeekV4DecodeLayerWeights(
                layer_id=layer_id,
                attention_mhc=load_mhc(f"{prefix}hc_attn"),
                attention=attention.load_attention_weights(
                    view,
                    layer_id,
                    rank,
                    device,
                    tensor_parallel=tensor_parallel,
                ),
                ffn_mhc=load_mhc(f"{prefix}hc_ffn"),
                ffn_norm=load(f"{prefix}ffn_norm.weight"),
                ffn=load_moe(view, layer_id, rank, device),
            )
        )

    return DeepSeekV4TargetWeights(
        embedding=embedding,
        layers=tuple(layers),
        head_mhc=load_mhc("hc_head"),
        final_norm=load("norm.weight"),
        head=load("head.weight"),
    )


class DeepSeekV4TargetModel:
    def __init__(
        self,
        weights: DeepSeekV4TargetWeights[object],
        ops: object,
    ) -> None:
        if [layer.layer_id for layer in weights.layers] != list(
            range(deepseek_v4_flash.NUM_LAYERS)
        ):
            raise ValueError("DeepSeek V4 target weights must contain every layer")
        self.weights = weights
        self.ops = ops

    def decode(
        self,
        token_ids: object,
        layer_inputs: tuple[DeepSeekV4DecodeLayerInputs[object], ...],
        endpoint: DeepSeekV4EndpointWorkspace[object]
        | DeepSeekV4TEP4EndpointWorkspace[object],
    ) -> object:
        return self._decode(token_ids, layer_inputs, endpoint, None)

    def decode_target(self, runtime: DeepSeekV4TargetDecodeRuntime[object]) -> None:
        runtime.prepare()
        execution = runtime.execution
        logits = self.decode(
            execution.token_ids,
            execution.layer_inputs,
            execution.endpoint,
        )
        self.ops.select_tokens(logits, execution.endpoint, runtime.target_tokens)
        runtime.publish()

    def verify(self, runtime: DeepSeekV4VerifyRuntime[object]) -> None:
        runtime.prepare()
        execution = runtime.execution
        logits = self._decode(
            execution.token_ids,
            execution.layer_inputs,
            execution.endpoint,
            runtime.captured_hidden.view(
                -1, len(deepseek_v4_flash.DSPARK_TARGET_LAYER_IDS) * deepseek_v4_flash.HIDDEN_SIZE
            ),
        )
        self.ops.select_tokens(
            logits, execution.endpoint, runtime.target_tokens.view(-1)
        )

    def accept_verified(self, runtime: DeepSeekV4VerifyRuntime[object]) -> None:
        self.ops.accept_greedy(runtime)

    def publish_verified(self, runtime: DeepSeekV4VerifyRuntime[object]) -> None:
        self.ops.publish_verified(runtime)

    def _decode(
        self,
        token_ids: object,
        layer_inputs: tuple[DeepSeekV4DecodeLayerInputs[object], ...],
        endpoint: DeepSeekV4EndpointWorkspace[object]
        | DeepSeekV4TEP4EndpointWorkspace[object],
        captured_hidden: object | None,
    ) -> object:
        if len(layer_inputs) != deepseek_v4_flash.NUM_LAYERS:
            raise ValueError("DeepSeek V4 decode requires inputs for every layer")
        streams = self.ops.decode_embedding(token_ids, self.weights, endpoint)
        capture = 0
        cross_layer_mhc = len(token_ids) in (1, 4, 16, 32)
        for layer_index, (weights, inputs) in enumerate(
            zip(self.weights.layers, layer_inputs, strict=True)
        ):
            next_weights = (
                self.weights.layers[layer_index + 1]
                if cross_layer_mhc and layer_index + 1 < len(self.weights.layers)
                else None
            )
            streams = self.ops.layer.decode_layer(
                streams,
                token_ids,
                weights,
                inputs.batch,
                inputs.raw_slots,
                inputs.main,
                inputs.raw,
                inputs.attention_state,
                inputs.workspace,
                inputs.metadata,
                inputs.mega_buffer,
                attention_prepared=cross_layer_mhc and layer_index > 0,
                next_weights=next_weights,
            )
            if (
                captured_hidden is not None
                and weights.layer_id in deepseek_v4_flash.DSPARK_TARGET_LAYER_IDS
            ):
                start = capture * deepseek_v4_flash.HIDDEN_SIZE
                self.ops.capture_hidden(
                    streams,
                    captured_hidden[:, start : start + deepseek_v4_flash.HIDDEN_SIZE],
                )
                capture += 1
        return self.ops.decode_head(streams, self.weights, endpoint)

    def prefill(
        self,
        runtime: DeepSeekV4TargetRuntime[object],
        batch: DeepSeekV4PrefillBatch,
    ) -> object | None:
        import torch

        staging = runtime.staging[batch.staging_index].device
        token_ids = staging.token_ids[: batch.token_count]
        resources = runtime.resources
        streams = self.ops.prefill_embedding(
            token_ids,
            self.weights,
            resources.workspace_b.collapsed[: batch.token_count],
            resources.workspace_b.streams_out[: batch.token_count],
            (
                None
                if resources.embedding_tokens is None
                else resources.embedding_tokens[: batch.token_count]
            ),
            (
                None
                if resources.embedding_select is None
                else resources.embedding_select[: batch.token_count]
            ),
        )
        capture = 0
        capture_streams = runtime.resources.dspark_streams[: batch.dspark_seed_count]
        captured_hidden = runtime.resources.dspark_hidden[: batch.dspark_seed_count]
        for weights in self.weights.layers:
            streams = self.ops.layer.prefill_layer(
                streams,
                token_ids,
                weights,
                *_prefill_layer_views(runtime, batch, weights.layer_id),
            )
            if (
                batch.dspark_seed_count
                and weights.layer_id in deepseek_v4_flash.DSPARK_TARGET_LAYER_IDS
            ):
                torch.index_select(
                    streams,
                    0,
                    staging.dspark_indices[: batch.dspark_seed_count],
                    out=capture_streams,
                )
                start = capture * deepseek_v4_flash.HIDDEN_SIZE
                self.ops.capture_hidden(
                    capture_streams,
                    captured_hidden[:, start : start + deepseek_v4_flash.HIDDEN_SIZE],
                )
                capture += 1
        if not batch.sample_count:
            return None
        endpoint = _view_shape_record(
            runtime.resources.head,
            (
                tep4_endpoint_workspace_shapes(batch.sample_count)
                if isinstance(runtime.resources.head, DeepSeekV4TEP4EndpointWorkspace)
                else endpoint_workspace_shapes(batch.sample_count)
            ),
        )
        torch.index_select(
            streams,
            0,
            staging.sample_indices[: batch.sample_count],
            out=endpoint.streams,
        )
        logits = self.ops.decode_head(endpoint.streams, self.weights, endpoint)
        self.ops.select_tokens(
            logits,
            endpoint,
            runtime.resources.head_tokens[: batch.sample_count],
        )
        return logits

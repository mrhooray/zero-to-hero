from dataclasses import dataclass
from time import perf_counter_ns

from infer.models.deepseek_v4_flash import model as deepseek_v4_flash
from infer.models.deepseek_v4_flash.dspark import (
    DeepSeekV4DSparkModel,
    DeepSeekV4DSparkRuntime,
)
from infer.models.deepseek_v4_flash.target import (
    DeepSeekV4TargetModel,
    DeepSeekV4TargetRuntime,
)
from infer.protocol import BatchPlan, BatchPlanRow, StepResult, StepResultRow

_EOS_TOKEN_ID = 1


@dataclass(frozen=True, slots=True)
class _Pending:
    plan: BatchPlan
    event: object
    started_ns: int


class DeepSeekV4TargetWorker:
    def __init__(
        self,
        model: DeepSeekV4TargetModel,
        runtime: DeepSeekV4TargetRuntime[object],
        dspark_model: DeepSeekV4DSparkModel | None,
        dspark_runtime: DeepSeekV4DSparkRuntime[object] | None,
        graphs: dict[tuple[int, int], object],
        rank: int,
        process_group: object,
        collectives: object,
        completion_events: tuple[object, object],
        parallelism: str,
    ) -> None:
        import torch

        keys = {
            (bucket, parity)
            for bucket in (0, *deepseek_v4_flash.DECODE_BATCH_SIZES)
            for parity in range(deepseek_v4_flash.DECODE_DESCRIPTOR_PARITIES)
        }
        if parallelism not in {"dep4", "tep4"}:
            raise ValueError("DeepSeek parallelism must be dep4 or tep4")
        if not 0 <= rank < deepseek_v4_flash.DEP_SIZE:
            raise ValueError("DeepSeek rank is out of range")
        if (dspark_model is None) != (dspark_runtime is None):
            raise ValueError("DeepSeek DSpark model and runtime must agree")
        target_only = dspark_model is None
        decode = runtime.decode if target_only else runtime.verify
        if set(decode) != keys or set(graphs) != keys:
            raise ValueError(
                "DeepSeek runtimes and graphs must cover every bucket parity"
            )
        if dspark_runtime is not None and set(dspark_runtime.batches) != keys:
            raise ValueError(
                "DeepSeek runtimes and graphs must cover every bucket parity"
            )
        self.model = model
        self.runtime = runtime
        self.dspark_model = dspark_model
        self.dspark_runtime = dspark_runtime
        self._decode = decode
        self._decode_width = 1 if target_only else deepseek_v4_flash.DSPARK_VERIFY_WIDTH
        self._graphs = graphs
        self._rank = rank
        self._process_group = process_group
        self._collectives = collectives
        self._completion_events = completion_events
        self._lane_count = deepseek_v4_flash.DEP_SIZE if parallelism == "dep4" else 1
        result_width = runtime.result_records.shape[1]
        self._result_device = (
            None
            if self._lane_count == 1
            else runtime.sampled_tokens.new_empty(
                (
                    deepseek_v4_flash.DECODE_DESCRIPTOR_PARITIES,
                    self._lane_count * runtime.live_slots,
                    result_width,
                )
            )
        )
        self._result_cpu = torch.empty(
            (
                deepseek_v4_flash.DECODE_DESCRIPTOR_PARITIES,
                self._lane_count,
                runtime.live_slots,
                result_width,
            ),
            dtype=runtime.sampled_tokens.dtype,
            pin_memory=runtime.sampled_tokens.is_cuda,
        )

    def dispatch(self, plan: BatchPlan) -> object:
        started = perf_counter_ns()
        rows = self._validate(plan)
        staging_index = plan.staging_index
        decode_rows = tuple(row for row in rows if not row.token_ids)
        prefill_rows = tuple(row for row in rows if row.token_ids)
        restores = tuple(
            (row, _prefix_tail_physical(row, row.token_start))
            for row in prefill_rows
            if row.restore_slot is not None
        )
        captures = tuple(
            (row, _prefix_tail_physical(row, row.token_start + len(row.token_ids)))
            for row in prefill_rows
            if row.capture_slot is not None
        )
        if any(not row.token_ids for lane in plan.lanes for row in lane):
            key = _bucket(len(decode_rows)), staging_index
            decode = self._decode[key]
            controls = []
            for row in decode_rows:
                block = (
                    row.table_delta.physical_blocks[0]
                    if row.table_delta.physical_blocks
                    else -1
                )
                control = (row.slot, row.table_delta.start_block, block)
                if self.dspark_model is not None:
                    control += (row.max_accept_tokens, row.ignore_eos)
                controls.append(control)
            decode.stage(tuple(controls))
            self._graphs[key].replay()

        if any(row.token_ids for lane in plan.lanes for row in lane):
            sample_rows = tuple(
                index for index, row in enumerate(prefill_rows) if row.sample
            )
            batch = self.runtime.stage_prefill(
                staging_index,
                tuple(row.token_ids for row in prefill_rows),
                tuple(row.token_start for row in prefill_rows),
                tuple(row.slot for row in prefill_rows),
                tuple(
                    (row.table_delta.start_block, row.table_delta.physical_blocks)
                    for row in prefill_rows
                ),
                sample_rows,
            )
            for row, tail in restores:
                self.runtime.restore_prefix(row.restore_slot, row.slot, tail)
                if self.dspark_runtime is not None:
                    self.dspark_runtime.restore_prefix(row.restore_slot, row.slot)
            for row in prefill_rows:
                if not row.token_start:
                    self.runtime.reset_state(row.slot)
                    if self.dspark_runtime is not None:
                        self.dspark_runtime.reset_state(row.slot)
            self.model.prefill(self.runtime, batch)
            if self.dspark_model is not None:
                assert self.dspark_runtime is not None
                self.dspark_model.seed_prefill(self.dspark_runtime, self.runtime, batch)
            self.runtime.publish_prefill(batch)
            for row, tail in captures:
                self.runtime.capture_prefix(row.slot, row.capture_slot, tail)
                if self.dspark_runtime is not None:
                    self.dspark_runtime.capture_prefix(row.slot, row.capture_slot)

        if self._lane_count == 1:
            result_device = self.runtime.result_records[: self.runtime.live_slots]
        else:
            result_device = self._result_device[staging_index]
            self._collectives.all_gather_into_tensor(
                result_device,
                self.runtime.result_records[: self.runtime.live_slots],
                group=self._process_group,
            )
        self._result_cpu[staging_index].copy_(
            result_device.view(
                self._lane_count,
                self.runtime.live_slots,
                self.runtime.result_records.shape[1],
            ),
            non_blocking=result_device.is_cuda,
        )
        event = self._completion_events[staging_index]
        event.record()
        return _Pending(plan, event, started)

    def collect(self, pending: object) -> StepResult:
        if type(pending) is not _Pending:
            raise TypeError("DeepSeek target worker requires its pending dispatch")
        pending.event.synchronize()
        records = self._result_cpu[pending.plan.staging_index]
        lanes = []
        for rows, lane_records in zip(pending.plan.lanes, records, strict=True):
            results = []
            for row in rows:
                record = lane_records[row.slot]
                if row.token_ids:
                    accepted = len(row.token_ids)
                    output = (int(record[1]),) if row.sample else ()
                else:
                    accepted = int(record[0])
                    output = tuple(int(token) for token in record[1 : accepted + 1])
                results.append(
                    StepResultRow(
                        row.slot,
                        accepted,
                        output,
                        not row.ignore_eos and _EOS_TOKEN_ID in output,
                    )
                )
            lanes.append(tuple(results))
        return StepResult(
            pending.plan.step_id,
            tuple(lanes),
            perf_counter_ns() - pending.started_ns,
        )

    def _validate(self, plan: BatchPlan) -> tuple[object, ...]:
        if len(plan.lanes) != self._lane_count:
            raise ValueError("DeepSeek plan has the wrong lane count")
        if not any(plan.lanes):
            raise ValueError("DeepSeek plan must contain at least one row")
        has_prefill = any(row.token_ids for lane in plan.lanes for row in lane)
        decode_count = max(
            sum(not row.token_ids for row in lane) for lane in plan.lanes
        )
        if decode_count > max(deepseek_v4_flash.DECODE_BATCH_SIZES):
            raise ValueError("DeepSeek target decode exceeds its largest bucket")
        if (has_prefill and plan.graph_bucket is not None) or (
            not has_prefill and plan.graph_bucket != _bucket(decode_count)
        ):
            raise ValueError("DeepSeek plan has an invalid global graph bucket")
        rows = plan.lanes[self._rank if self._lane_count > 1 else 0]
        if len({row.slot for row in rows}) != len(rows):
            raise ValueError("a DeepSeek target slot appears twice in one plan")
        prefill_requests = sum(bool(row.token_ids) for row in rows)
        if (
            prefill_requests > deepseek_v4_flash.MAX_PREFILL_REQUESTS
            or sum(len(row.token_ids) for row in rows)
            > deepseek_v4_flash.PREFILL_CHUNK_TOKENS
        ):
            raise ValueError("DeepSeek target prefill exceeds its packed budget")
        for row in rows:
            if not 0 <= row.slot < self.runtime.live_slots:
                raise ValueError("DeepSeek target slot is out of range")
            delta = row.table_delta
            if (
                delta.start_block < 0
                or delta.start_block + len(delta.physical_blocks)
                > deepseek_v4_flash.MAX_CONTEXT_TOKENS // 128
                or any(
                    not 0 <= block < self.runtime.history_blocks
                    for block in delta.physical_blocks
                )
            ):
                raise ValueError("DeepSeek target table delta is out of range")
            if row.token_ids:
                if (
                    row.token_start < 0
                    or row.max_accept_tokens != len(row.token_ids)
                    or row.token_start + len(row.token_ids)
                    > deepseek_v4_flash.MAX_CONTEXT_TOKENS
                    or any(
                        not 0 <= token_id < deepseek_v4_flash.VOCAB_SIZE
                        for token_id in row.token_ids
                    )
                ):
                    raise ValueError("DeepSeek target prefill row is invalid")
                continue
            if row.restore_slot is not None or row.capture_slot is not None:
                raise ValueError("DeepSeek prefix snapshots require a prefill row")
            if not 0 < row.token_start < deepseek_v4_flash.MAX_CONTEXT_TOKENS:
                raise ValueError("DeepSeek decode slot or position is out of range")
            if not 1 <= row.max_accept_tokens <= self._decode_width:
                raise ValueError("DeepSeek decode acceptance exceeds its width")
            if not row.sample:
                raise ValueError("DeepSeek decode must sample output tokens")
            if len(row.table_delta.physical_blocks) > 1:
                raise ValueError("DeepSeek decode can append at most one history block")
        return rows


def _bucket(rows: int) -> int:
    if rows == 0:
        return 0
    return next(bucket for bucket in deepseek_v4_flash.DECODE_BATCH_SIZES if bucket >= rows)


def _prefix_tail_physical(row: BatchPlanRow, boundary: int) -> int:
    block_tokens = deepseek_v4_flash.COMPRESSED_HISTORY_CANDIDATE_BLOCK_TOKENS
    if boundary <= 0 or boundary % block_tokens:
        raise ValueError("a DeepSeek prefix snapshot requires a complete history block")
    logical_block = boundary // block_tokens - 1
    delta_index = logical_block - row.table_delta.start_block
    blocks = row.table_delta.physical_blocks
    if not 0 <= delta_index < len(blocks):
        raise ValueError("the DeepSeek prefix tail is absent from its table delta")
    return blocks[delta_index]


def _run_decode(model, runtime, dspark_model, dspark_runtime, key) -> None:
    verify = runtime.verify[key]
    batch = dspark_runtime.batch(*key)
    dspark_model.draft(dspark_runtime, batch, verify)
    model.verify(verify)
    model.accept_verified(verify)
    dspark_model.commit(dspark_runtime, batch, verify)
    model.publish_verified(verify)


def _run_target_decode(model, runtime, key) -> None:
    model.decode_target(runtime.decode[key])

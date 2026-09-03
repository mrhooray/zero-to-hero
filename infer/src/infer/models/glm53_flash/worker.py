from dataclasses import dataclass
from time import perf_counter_ns

from infer.models.glm53_flash import model as glm53_flash
from infer.models.glm53_flash.codec import EOS_TOKEN_IDS
from infer.models.glm53_flash.nextn import (
    GLM53NextNModel,
    GLM53NextNRuntime,
    _nextn_workspace_view,
)
from infer.models.glm53_flash.target import (
    GLM53TargetModel,
    GLM53TargetRuntime,
    _nccl_all_reduce,
)
from infer.protocol import BatchPlan, BatchPlanRow, StepResult, StepResultRow


@dataclass(frozen=True, slots=True)
class _Pending:
    plan: BatchPlan
    event: object
    started_ns: int


@dataclass(frozen=True, slots=True)
class GLM53WorkerStaging[TensorT]:
    remaining_cpu: TensorT
    remaining: TensorT
    ignore_eos_cpu: TensorT
    ignore_eos: TensorT
    accepted: TensorT
    continuing: TensorT
    seed_states_cpu: TensorT
    seed_states: TensorT
    seed_tokens: TensorT
    result_records: TensorT
    gathered_records: TensorT
    result_cpu: TensorT


def allocate_glm53_worker_staging(
    device: object,
    live_slot_count: int = max(glm53_flash.GLM53_TARGET_DECODE_BATCH_SIZES),
    lane_count: int = 1,
    *,
    speculative: bool = True,
) -> GLM53WorkerStaging[object]:
    import torch

    if lane_count not in {1, glm53_flash.TP_SIZE}:
        raise ValueError("GLM worker lane count must be one or four")
    decode_capacity = max(
        glm53_flash.glm53_decode_batch_sizes(
            1 if lane_count == glm53_flash.TP_SIZE else glm53_flash.TP_SIZE,
            speculative=speculative,
        )
    )
    if type(live_slot_count) is not int or live_slot_count < decode_capacity:
        raise ValueError("GLM worker live slots do not cover its decode capacity")
    shape = (2, decode_capacity)
    record_shape = (2, live_slot_count, 1 + glm53_flash.GLM53_TARGET_VERIFY_WIDTH)
    cpu = lambda shape, dtype: torch.zeros(
        shape, dtype=dtype, device="cpu", pin_memory=True
    )
    gpu = lambda shape, dtype: torch.zeros(shape, dtype=dtype, device=device)
    return GLM53WorkerStaging(
        remaining_cpu=cpu(shape, torch.int32),
        remaining=gpu(shape, torch.int32),
        ignore_eos_cpu=cpu(shape, torch.uint8),
        ignore_eos=gpu(shape, torch.uint8),
        accepted=gpu(shape, torch.int32),
        continuing=gpu(shape, torch.uint8),
        seed_states_cpu=cpu(shape, torch.int64),
        seed_states=gpu(shape, torch.int64),
        seed_tokens=gpu(shape, torch.int64),
        result_records=gpu(record_shape, torch.int64),
        gathered_records=gpu(
            (2, lane_count * live_slot_count, 1 + glm53_flash.GLM53_TARGET_VERIFY_WIDTH),
            torch.int64,
        ),
        result_cpu=cpu(
            (
                2,
                lane_count,
                live_slot_count,
                1 + glm53_flash.GLM53_TARGET_VERIFY_WIDTH,
            ),
            torch.int64,
        ),
    )


class GLM53Worker:
    """Asynchronous bridge from one scheduler plan to the GLM model."""

    def __init__(
        self,
        model: GLM53TargetModel,
        runtime: GLM53TargetRuntime[object],
        nextn_model: GLM53NextNModel | None,
        nextn_runtime: GLM53NextNRuntime[object] | None,
        process_group: object,
        all_reduce: object,
        staging: GLM53WorkerStaging[object],
        completion_events: tuple[object, object],
        *,
        rank: int = 0,
        collectives: object | None = None,
        parallelism: str = "tep4",
    ) -> None:
        if len(completion_events) != 2:
            raise ValueError("GLM worker requires two completion events")
        if parallelism not in {"dep4", "tep4"}:
            raise ValueError("GLM parallelism must be dep4 or tep4")
        if not 0 <= rank < glm53_flash.TP_SIZE:
            raise ValueError("GLM worker rank is out of range")
        if (nextn_model is None) != (nextn_runtime is None):
            raise ValueError("GLM NextN model and runtime must be provided together")
        self.model = model
        self.runtime = runtime
        self.nextn_model = nextn_model
        self.nextn_runtime = nextn_runtime
        self._process_group = process_group
        self._all_reduce = all_reduce
        self._staging = staging
        self._completion_events = completion_events
        self._rank = rank
        self._collectives = collectives
        self._lane_count = glm53_flash.TP_SIZE if parallelism == "dep4" else 1
        self._live_slots = runtime.committed_lengths.shape[0]
        self._decode_width = (
            glm53_flash.GLM53_TARGET_VERIFY_WIDTH if nextn_runtime is not None else 1
        )
        if self._lane_count > 1 and collectives is None:
            raise ValueError("GLM DEP4 worker requires distributed collectives")
        self._decode_graphs: dict[tuple[int, int], object] | None = None

    def install_decode_graphs(
        self, decode_graphs: dict[tuple[int, int], object]
    ) -> None:
        self._decode_graphs = decode_graphs

    def capture_decode_graphs(self, plans, stream, torch) -> None:
        graphs = {}
        pool = torch.cuda.graph_pool_handle()
        for plan in plans:
            with torch.cuda.stream(stream):
                lane = plan.lanes[self._rank if self._lane_count > 1 else 0]
                batch, groups = self._stage_decode(
                    lane,
                    plan.staging_index,
                    plan.graph_bucket,
                )
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=pool, stream=stream):
                self._run_decode(batch, groups, plan.staging_index)
            graphs[groups, plan.staging_index] = graph
        torch.cuda.synchronize()
        self.install_decode_graphs(graphs)

    def dispatch(self, plan: BatchPlan) -> object:
        started = perf_counter_ns()
        rows = self._validate(plan)
        staging_index = plan.staging_index
        for index, row in enumerate(rows):
            self.runtime.apply_table_delta(
                row.slot,
                row.table_delta.start_block,
                row.table_delta.physical_blocks,
                staging_index,
                index,
            )

        decode_rows = tuple(row for row in rows if not row.token_ids)
        prefill_rows = tuple(row for row in rows if row.token_ids)
        has_decode = any(not row.token_ids for lane in plan.lanes for row in lane)
        has_prefill = any(row.token_ids for lane in plan.lanes for row in lane)
        decode_bucket = 0
        if has_decode:
            decode_bucket = _decode_bucket(
                max(sum(not row.token_ids for row in lane) for lane in plan.lanes),
                tuple(self.runtime.decode),
            )
            self._dispatch_decode(
                decode_rows,
                staging_index,
                decode_bucket,
            )

        if has_prefill:
            sample_rows = tuple(
                index for index, row in enumerate(prefill_rows) if row.sample
            )
            target_counts = tuple(
                sum(len(row.token_ids) for row in lane if row.token_ids)
                for lane in plan.lanes
            )
            nextn_counts = (
                tuple(
                    sum(
                        len(row.token_ids) - (row.token_start == 0)
                        for row in lane
                        if row.token_ids
                    )
                    for lane in plan.lanes
                )
                if self.nextn_runtime is not None
                else (0,) * len(plan.lanes)
            )
            sample_bucket = (
                _decode_bucket(
                    max(sum(row.sample for row in lane) for lane in plan.lanes),
                    tuple(self.runtime.decode),
                )
                if self.nextn_runtime is not None
                else 0
            )
            self._dispatch_prefill(
                prefill_rows,
                sample_rows,
                staging_index,
                target_counts,
                nextn_counts,
                sample_bucket,
            )

        self._stage_results(rows, decode_rows, staging_index, decode_bucket)

        event = self._completion_events[staging_index]
        event.record()
        return _Pending(plan, event, started)

    def collect(self, pending: object) -> StepResult:
        if type(pending) is not _Pending:
            raise TypeError("GLM worker requires its pending dispatch")
        pending.event.synchronize()
        plan = pending.plan
        records = self._staging.result_cpu[plan.staging_index]
        lanes = []
        for rows, lane_records in zip(plan.lanes, records, strict=True):
            results = []
            for row in rows:
                record = lane_records[row.slot]
                accepted = int(record[0])
                output = (
                    (int(record[1]),)
                    if row.token_ids and row.sample
                    else ()
                    if row.token_ids
                    else tuple(int(token) for token in record[1 : accepted + 1])
                )
                results.append(_make_result(row, accepted, output))
            lanes.append(tuple(results))
        return StepResult(
            plan.step_id,
            tuple(lanes),
            perf_counter_ns() - pending.started_ns,
        )

    def _dispatch_decode(
        self,
        rows: tuple[BatchPlanRow, ...],
        staging_index: int,
        bucket: int,
    ) -> None:
        batch, groups = self._stage_decode(rows, staging_index, bucket)
        if self._decode_graphs is None:
            self._run_decode(batch, groups, staging_index)
        else:
            self._decode_graphs[groups, staging_index].replay()

    def _stage_decode(
        self,
        rows: tuple[BatchPlanRow, ...],
        staging_index: int,
        bucket: int | None = None,
    ) -> tuple[object, int]:
        state_slots = tuple(row.slot for row in rows)
        batch = (
            self.runtime.stage_decode(state_slots, staging_index)
            if self._lane_count == 1
            else self.runtime.stage_decode(state_slots, staging_index, bucket)
        )
        groups = batch.token_ids.shape[0]
        if self.nextn_runtime is None:
            return batch, groups
        remaining_cpu = self._staging.remaining_cpu[staging_index, :groups]
        remaining_cpu.zero_()
        ignore_eos_cpu = self._staging.ignore_eos_cpu[staging_index, :groups]
        ignore_eos_cpu.zero_()
        for row, plan_row in enumerate(rows):
            remaining_cpu[row] = plan_row.max_accept_tokens
            ignore_eos_cpu[row] = plan_row.ignore_eos
        remaining = self._staging.remaining[staging_index, :groups]
        remaining.copy_(remaining_cpu, non_blocking=remaining.is_cuda)
        ignore_eos = self._staging.ignore_eos[staging_index, :groups]
        ignore_eos.copy_(ignore_eos_cpu, non_blocking=ignore_eos.is_cuda)
        return batch, groups

    def _run_decode(self, batch: object, groups: int, staging_index: int) -> None:
        self.runtime.prepare_decode(
            batch,
            self.runtime.decode_staging[staging_index],
        )
        if self.nextn_runtime is None:
            outcome = self.model.decode_tokens(
                batch,
                self.runtime.state,
                self.runtime.target_decode[groups],
                self._process_group,
                self._all_reduce.decode,
            )
            if outcome.token is None:
                raise RuntimeError("GLM target decode produced no tokens")
            accepted = self._staging.accepted[staging_index, :groups]
            accepted.copy_(batch.sparse_mla.active)
            self.runtime.publish_decoded(batch, outcome.token, accepted)
            return

        assert self.nextn_model is not None
        remaining = self._staging.remaining[staging_index, :groups]
        candidates = self.nextn_runtime.draft_candidates(
            self.nextn_model,
            groups,
            staging_index,
            self.model.weights.endpoint,
            self._process_group,
            self._all_reduce.decode,
        )
        verify = self.runtime.stage_verify(
            candidates,
            batch.state_indices,
            batch.sparse_mla.active,
            batch.sparse_mla.raw_lengths,
            staging_index,
        )
        outcome = self.model.verify_tokens(
            verify.batch,
            self.runtime.state,
            verify.transaction,
            verify.workspace,
            self._process_group,
            self._all_reduce.verify,
        )
        if outcome.token is None:
            raise RuntimeError("GLM target verification produced no tokens")

        from infer.models.glm53_flash.ops.speculate import glm53_greedy_accept

        active = batch.sparse_mla.active
        accepted = self._staging.accepted[staging_index, :groups]
        continuing = self._staging.continuing[staging_index, :groups]
        glm53_greedy_accept(
            candidates,
            outcome.token.view(groups, glm53_flash.GLM53_TARGET_VERIFY_WIDTH),
            remaining,
            self._staging.ignore_eos[staging_index, :groups],
            active,
            accepted,
            candidates,
            continuing,
        )
        self.runtime.publish_verified(verify, accepted, active, candidates)
        self.nextn_runtime.verify_candidates(
            self.nextn_model,
            outcome.token,
            outcome.normalized,
            groups,
            staging_index,
            self.model.weights.endpoint,
            self._process_group,
            self._all_reduce.verify,
        )
        self.nextn_runtime.commit_candidates(
            groups,
            staging_index,
            candidates,
            accepted,
            active,
            continuing,
        )

    def _dispatch_prefill(
        self,
        rows: tuple[BatchPlanRow, ...],
        sample_rows: tuple[int, ...],
        staging_index: int,
        target_counts: tuple[int, ...],
        nextn_counts: tuple[int, ...],
        sample_bucket: int,
    ) -> None:
        if not rows:
            self.model.prefill_empty(
                self.runtime.prefill_workspace,
                self._process_group,
                target_counts,
            )
            if self.nextn_runtime is None:
                return
            if any(nextn_counts):
                assert self.nextn_model is not None
                self.nextn_model.prefill_empty(
                    self.nextn_runtime.workspaces[glm53_flash.KDA_CHUNK_SIZE],
                    self._process_group,
                    nextn_counts,
                )
            self._seed_prefill_candidates((), staging_index, sample_bucket)
            return

        start_tokens = tuple(row.token_start for row in rows)
        state_slots = tuple(row.slot for row in rows)
        for row in rows:
            if not row.token_start:
                self.runtime.reset_state(row.slot)
                if self.nextn_runtime is not None:
                    self.nextn_runtime.reset_slot(row.slot)
        for row in rows:
            if row.restore_slot is None:
                continue
            tail = _prefix_tail_physical(row, row.token_start)
            self.runtime.restore_prefix(row.restore_slot, row.slot, tail)
            if self.nextn_runtime is not None:
                self.nextn_runtime.restore_prefix(row.restore_slot, row.slot, tail)
        batch = self.runtime.stage_prefill(
            tuple(row.token_ids for row in rows),
            start_tokens,
            state_slots,
            sample_rows,
        )
        head_workspace = (
            self.runtime.decode[batch.sample_indices.shape[0]].head
            if batch.sample_indices is not None
            else None
        )
        prefill_args = (
            batch,
            self.runtime.state,
            self.runtime.prefill_workspace,
            head_workspace,
            self._process_group,
        )
        outcome = (
            self.model.prefill_tokens(*prefill_args, target_counts)
            if self._lane_count > 1
            else self.model.prefill_tokens(*prefill_args)
        )
        if sample_rows:
            if outcome.token is None:
                raise RuntimeError("sampled GLM target step returned no token")
            self.runtime.publish_prefill(batch, outcome.token)

        if self.nextn_runtime is None:
            self._capture_prefixes(rows)
            return

        sequence_lengths = tuple(len(row.token_ids) for row in rows)
        nextn_batch = self.nextn_runtime.stage_prefill(
            batch.token_ids,
            outcome.normalized,
            sequence_lengths,
            start_tokens,
            state_slots,
        )
        if any(nextn_counts) and nextn_batch.sparse_mla is None:
            assert self.nextn_model is not None
            self.nextn_model.prefill_empty(
                self.nextn_runtime.workspaces[glm53_flash.KDA_CHUNK_SIZE],
                self._process_group,
                nextn_counts,
            )
        elif nextn_batch.sparse_mla is not None:
            assert self.nextn_model is not None
            workspace = _nextn_workspace_view(
                self.nextn_runtime.workspaces[glm53_flash.KDA_CHUNK_SIZE],
                nextn_batch.token_ids.shape[0],
                self.nextn_runtime.attention_tp_size,
            )
            forward_args = (
                nextn_batch.token_ids,
                nextn_batch.hidden,
                nextn_batch.sparse_mla,
                self.nextn_runtime.state.history,
                workspace,
                workspace.endpoint,
                self.model.weights.endpoint,
                self._process_group,
                _nccl_all_reduce,
            )
            if self._lane_count > 1:
                self.nextn_model.forward(
                    *forward_args,
                    head_rows=None,
                    moe_token_counts=nextn_counts,
                )
            else:
                self.nextn_model.forward(*forward_args, head_rows=None)
        self.nextn_runtime.publish_prefill(nextn_batch)
        self._capture_prefixes(rows)
        if self._lane_count > 1:
            self._seed_prefill_candidates(rows, staging_index, sample_bucket)
        elif sample_rows:
            if batch.sample_state_indices is None or outcome.token is None:
                raise RuntimeError("sampled GLM prefill has no NextN seed")
            self.nextn_runtime.seed_candidates(
                self.nextn_model,
                outcome.token,
                batch.sample_state_indices,
                batch.sample_count,
                staging_index,
                self.model.weights.endpoint,
                self._process_group,
                self._all_reduce.decode,
            )

    def _seed_prefill_candidates(
        self,
        rows: tuple[BatchPlanRow, ...],
        staging_index: int,
        sample_bucket: int,
    ) -> None:
        if not sample_bucket:
            return
        if self.nextn_model is None or self.nextn_runtime is None:
            raise RuntimeError("target-only GLM must not seed NextN candidates")
        staging = self._staging
        sampled_slots = tuple(row.slot for row in rows if row.sample)
        states_cpu = staging.seed_states_cpu[staging_index, :sample_bucket]
        states_cpu.zero_()
        for index, slot in enumerate(sampled_slots):
            states_cpu[index] = slot
        states = staging.seed_states[staging_index, :sample_bucket]
        states.copy_(states_cpu, non_blocking=states.is_cuda)

        import torch

        tokens = staging.seed_tokens[staging_index, :sample_bucket]
        torch.index_select(self.runtime.sampled_tokens, 0, states, out=tokens)
        self.nextn_runtime.seed_candidates(
            self.nextn_model,
            tokens,
            states,
            len(sampled_slots),
            staging_index,
            self.model.weights.endpoint,
            self._process_group,
            self._all_reduce.decode,
        )

    def _capture_prefixes(self, rows: tuple[BatchPlanRow, ...]) -> None:
        for row in rows:
            if row.capture_slot is None:
                continue
            tail = _prefix_tail_physical(
                row,
                row.token_start + len(row.token_ids),
            )
            self.runtime.capture_prefix(row.slot, row.capture_slot, tail)
            if self.nextn_runtime is not None:
                self.nextn_runtime.capture_prefix(row.slot, row.capture_slot, tail)

    def _stage_results(
        self,
        rows: tuple[BatchPlanRow, ...],
        decode_rows: tuple[BatchPlanRow, ...],
        staging_index: int,
        decode_bucket: int,
    ) -> None:
        staging = self._staging
        records = staging.result_records[staging_index]
        for row in rows:
            records[row.slot].zero_()
            records[row.slot, 0].fill_(len(row.token_ids) if row.token_ids else 0)
            if row.token_ids and row.sample:
                records[row.slot, 1].copy_(self.runtime.sampled_tokens[row.slot])

        if decode_rows:
            groups = decode_bucket
            accepted = staging.accepted[staging_index, :groups]
            output = (
                self.runtime.target_decode[groups].endpoint.token.view(groups, 1)
                if self.nextn_runtime is None
                else self.nextn_runtime.verify_batches[groups][
                    staging_index
                ].token_ids.view(groups, glm53_flash.GLM53_TARGET_VERIFY_WIDTH)
            )
            for index, row in enumerate(decode_rows):
                records[row.slot, 0].copy_(accepted[index])
                records[row.slot, 1 : 1 + self._decode_width].copy_(output[index])

        if self._lane_count == 1:
            gathered = records
        else:
            gathered = staging.gathered_records[staging_index]
            assert self._collectives is not None
            self._collectives.all_gather_into_tensor(
                gathered,
                records,
                group=self._process_group,
            )
        staging.result_cpu[staging_index].copy_(
            gathered.view(
                self._lane_count,
                self._live_slots,
                1 + glm53_flash.GLM53_TARGET_VERIFY_WIDTH,
            ),
            non_blocking=gathered.is_cuda,
        )

    def _validate(self, plan: BatchPlan) -> tuple[BatchPlanRow, ...]:
        if len(plan.lanes) != self._lane_count:
            raise ValueError("GLM worker plan has the wrong lane count")
        if not any(plan.lanes):
            raise ValueError("GLM worker plan must contain at least one row")
        max_batch_size = max(self.runtime.decode)
        if any(len(lane) > max_batch_size for lane in plan.lanes):
            raise ValueError("GLM target worker batch exceeds the largest bucket")
        rows = plan.lanes[self._rank if self._lane_count > 1 else 0]
        if len({row.slot for row in rows}) != len(rows):
            raise ValueError("a GLM target slot appears twice in one plan")
        for row in rows:
            _validate_row(row, self._decode_width)
        if any(
            sum(len(row.token_ids) for row in lane) > glm53_flash.KDA_CHUNK_SIZE
            for lane in plan.lanes
        ):
            raise ValueError("a GLM prefill lane exceeds the token budget")
        has_prefill = any(row.token_ids for lane in plan.lanes for row in lane)
        decode_count = max(
            sum(not row.token_ids for row in lane) for lane in plan.lanes
        )
        expected_bucket = (
            None
            if has_prefill
            else _decode_bucket(decode_count, tuple(self.runtime.decode))
        )
        if plan.graph_bucket != expected_bucket:
            raise ValueError("GLM plan has an invalid global graph bucket")
        return rows


def _decode_bucket(
    rows: int,
    batch_sizes: tuple[int, ...] = glm53_flash.GLM53_TARGET_DECODE_BATCH_SIZES,
) -> int:
    if rows == 0:
        return 0
    return next(bucket for bucket in batch_sizes if bucket >= rows)


def _prefix_tail_physical(row: BatchPlanRow, boundary: int) -> int:
    block_tokens = glm53_flash.SPARSE_MLA_HISTORY_BLOCK_TOKENS
    if boundary <= 0 or boundary % block_tokens:
        raise ValueError("a GLM prefix snapshot requires a complete history block")
    logical_block = boundary // block_tokens - 1
    delta_index = logical_block - row.table_delta.start_block
    blocks = row.table_delta.physical_blocks
    if 0 <= delta_index < len(blocks):
        return blocks[delta_index]
    raise ValueError("the GLM prefix tail is absent from its table delta")


def _make_result(
    row: BatchPlanRow,
    accepted_count: int,
    output_token_ids: tuple[int, ...],
) -> StepResultRow:
    if row.sample and not output_token_ids:
        raise RuntimeError("sampled GLM step returned no token")
    return StepResultRow(
        row.slot,
        accepted_count,
        output_token_ids,
        bool(
            not row.ignore_eos
            and output_token_ids
            and output_token_ids[-1] in EOS_TOKEN_IDS
        ),
    )


def _validate_row(
    row: BatchPlanRow,
    decode_width: int = glm53_flash.GLM53_TARGET_VERIFY_WIDTH,
) -> None:
    if not row.token_ids:
        if row.restore_slot is not None or row.capture_slot is not None:
            raise ValueError("GLM prefix snapshots require a prefill row")
        if row.token_start == 0:
            raise ValueError("a GLM target request must start with prefill")
        if not 1 <= row.max_accept_tokens <= decode_width:
            kind = (
                "verify"
                if decode_width == glm53_flash.GLM53_TARGET_VERIFY_WIDTH
                else "decode"
            )
            raise ValueError(f"GLM decode acceptance exceeds its {kind} width")
        if not row.sample:
            raise ValueError("a GLM target decode must sample output")
        return
    if len(row.token_ids) > glm53_flash.KDA_CHUNK_SIZE:
        raise ValueError(
            f"GLM target prefill requires at most {glm53_flash.KDA_CHUNK_SIZE} tokens"
        )
    if row.max_accept_tokens != len(row.token_ids):
        raise ValueError("GLM target prefill must reserve its input length")
    if row.restore_slot is not None:
        _prefix_tail_physical(row, row.token_start)
    if row.capture_slot is not None:
        _prefix_tail_physical(row, row.token_start + len(row.token_ids))

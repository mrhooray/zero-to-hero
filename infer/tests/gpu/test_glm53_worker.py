import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

pytestmark = pytest.mark.gpu

import torch

from infer.models.glm53_flash import model as glm53
from infer.models.glm53_flash.codec import EOS_TOKEN_IDS
from infer.models.glm53_flash.target import GLM53TargetOutput
from infer.models.glm53_flash.worker import GLM53Worker, GLM53WorkerStaging
from infer.protocol import BatchPlan, BatchPlanRow, StepResultRow, TableDelta


def accept_glm53_greedy_chain(
    candidate_token_ids: tuple[int, ...],
    target_token_ids: tuple[int, ...],
    max_append_tokens: int,
) -> tuple[int, ...]:
    """Return the accepted target prefix for one root and three draft tokens."""

    width = glm53.GLM53_TARGET_VERIFY_WIDTH
    if len(candidate_token_ids) != width or len(target_token_ids) != width:
        raise ValueError(f"GLM greedy acceptance requires two {width}-token chains")
    if type(max_append_tokens) is not int or not 1 <= max_append_tokens <= width:
        raise ValueError(f"max_append_tokens must be in [1, {width}]")

    accepted = 1
    for candidate, target in zip(
        candidate_token_ids[1:], target_token_ids[:-1], strict=True
    ):
        if candidate != target:
            break
        accepted += 1

    output_token_ids = target_token_ids[: min(accepted, max_append_tokens)]
    for index, token_id in enumerate(output_token_ids):
        if token_id in EOS_TOKEN_IDS:
            return output_token_ids[: index + 1]
    return output_token_ids


class FakeRuntime:
    def __init__(self, events: list[tuple], sampled: list[int]) -> None:
        self.events = events
        self.sampled = sampled
        self.sampled_tokens = torch.zeros(32, dtype=torch.int64)
        self.state = object()
        self.prefill_workspace = object()
        self.decode = {
            bucket: SimpleNamespace(
                head=SimpleNamespace(token=torch.zeros(bucket, dtype=torch.int64))
            )
            for bucket in glm53.GLM53_TARGET_DECODE_BATCH_SIZES
        }
        self.target_decode = {
            bucket: SimpleNamespace(
                endpoint=SimpleNamespace(
                    normalized=torch.zeros((bucket, 1)),
                    token=torch.zeros(bucket, dtype=torch.int64),
                )
            )
            for bucket in glm53.GLM53_TARGET_DECODE_BATCH_SIZES
        }
        self.decode_staging = (object(), object())
        self.stage_error: Exception | None = None
        self.last_candidates = None
        self.committed_lengths = torch.zeros(32, dtype=torch.int32)

    def apply_table_delta(
        self, state_slot, start_block, physical_blocks, staging_index, row
    ) -> None:
        self.events.append(
            (
                "table_delta",
                state_slot,
                start_block,
                physical_blocks,
                staging_index,
                row,
            )
        )

    def capture_prefix(self, live_slot, snapshot_slot, tail_physical_block) -> None:
        self.events.append(
            (
                "target_capture",
                live_slot,
                snapshot_slot,
                tail_physical_block,
                int(self.committed_lengths[live_slot]),
            )
        )

    def restore_prefix(self, snapshot_slot, live_slot, tail_physical_block) -> None:
        self.events.append(
            ("target_restore", snapshot_slot, live_slot, tail_physical_block)
        )

    def stage_prefill(self, token_ids, start_tokens, state_slots, sample_rows):
        self.events.append(
            (
                "stage_prefill",
                token_ids,
                start_tokens,
                state_slots,
                sample_rows,
            )
        )
        if self.stage_error is not None:
            raise self.stage_error
        for sequence, start, slot in zip(
            token_ids, start_tokens, state_slots, strict=True
        ):
            self.committed_lengths[slot] = start + len(sequence)
        sample_count = len(sample_rows)
        bucket = (
            next(
                size
                for size in glm53.GLM53_TARGET_DECODE_BATCH_SIZES
                if size >= sample_count
            )
            if sample_count
            else 0
        )
        return SimpleNamespace(
            token_ids=torch.tensor(
                tuple(token for sequence in token_ids for token in sequence),
                dtype=torch.int64,
            ),
            sample_indices=torch.empty(bucket, dtype=torch.int64) if bucket else None,
            sample_state_indices=(
                torch.tensor(
                    tuple(state_slots[row] for row in sample_rows)
                    + (state_slots[sample_rows[0]],) * (bucket - sample_count),
                    dtype=torch.int64,
                )
                if bucket
                else None
            ),
            sample_count=sample_count,
        )

    def stage_decode(self, state_slots, staging_index, bucket=None):
        raw_lengths = tuple(
            int(self.committed_lengths[slot]) + 1 for slot in state_slots
        )
        self.events.append(("stage_decode", raw_lengths, state_slots, staging_index))
        if bucket is None:
            bucket = next(
                size
                for size in glm53.GLM53_TARGET_DECODE_BATCH_SIZES
                if size >= len(raw_lengths)
            )
        padded = bucket - len(state_slots)
        return SimpleNamespace(
            token_ids=torch.zeros(bucket, dtype=torch.int64),
            state_indices=torch.tensor(
                (*state_slots, *((-1,) * padded)), dtype=torch.int32
            ),
            state_indices_int64=torch.tensor(
                (*state_slots, *((0,) * padded)), dtype=torch.int64
            ),
            sparse_mla=SimpleNamespace(
                active=torch.tensor(
                    (*((1,) * len(state_slots)), *((0,) * padded)),
                    dtype=torch.uint8,
                ),
                raw_lengths=torch.tensor(
                    (*raw_lengths, *((0,) * padded)), dtype=torch.int32
                ),
                state_slots=torch.tensor(
                    (*state_slots, *((0,) * padded)), dtype=torch.int32
                ),
            ),
        )

    def prepare_decode(self, batch, staging) -> None:
        self.events.append(("prepare_decode", batch.token_ids.shape[0], staging))

    def stage_verify(self, candidates, state_slots, active, raw_lengths, staging_index):
        self.events.append(("stage_verify", candidates.shape[0], staging_index))
        self.last_candidates = candidates
        return SimpleNamespace(
            batch="verify_batch",
            transaction="target_transaction",
            workspace="target_workspace",
            state_slots=state_slots,
        )

    def publish_verified(self, verify, accepted, active, output) -> None:
        for row in range(accepted.shape[0]):
            if active[row]:
                self.committed_lengths[verify.state_slots[row]] += accepted[row]
        self.events.append(
            (
                "target_publish",
                accepted.clone(),
                active.clone(),
                output.clone(),
            )
        )

    def publish_decoded(self, batch, token, accepted) -> None:
        for row in range(accepted.shape[0]):
            if accepted[row]:
                self.committed_lengths[batch.sparse_mla.state_slots[row]] += accepted[
                    row
                ]
        self.events.append(("target_publish_decoded", token.clone()))

    def publish_prefill(self, batch, token) -> None:
        self.sampled_tokens.index_copy_(0, batch.sample_state_indices, token)
        self.events.append(("publish_prefill", batch.sample_count))

    def reset_state(self, state_slot) -> None:
        self.committed_lengths[state_slot] = 0
        self.events.append(("reset_state", state_slot))


class FakeModel:
    def __init__(self, runtime: FakeRuntime, nextn_runtime) -> None:
        self.runtime = runtime
        self.nextn_runtime = nextn_runtime
        self.weights = SimpleNamespace(endpoint="endpoint")

    def prefill_tokens(
        self,
        batch,
        state,
        workspace,
        head_workspace,
        process_group,
        token_counts=None,
    ):
        self.runtime.events.append(
            (
                "prefill",
                batch.token_ids,
                state,
                workspace,
                head_workspace,
                process_group,
                batch.sample_count,
            )
        )
        token = (
            torch.tensor(
                [self.runtime.sampled.pop(0) for _ in range(batch.sample_count)],
                dtype=torch.int64,
            )
            if batch.sample_count
            else None
        )
        return GLM53TargetOutput(
            torch.zeros((batch.token_ids.shape[0], 1)),
            token,
        )

    def prefill_empty(self, _workspace, _process_group, token_counts) -> None:
        self.runtime.events.append(("target_prefill_empty", token_counts))

    def verify_tokens(
        self,
        batch,
        state,
        transaction,
        workspace,
        process_group,
        all_reduce,
    ):
        candidates = self.runtime.last_candidates
        targets = candidates.clone()
        targets[:, :-1].copy_(candidates[:, 1:])
        targets[:, -1].add_(1)
        for row, accepted in enumerate(self.nextn_runtime.accept_counts):
            if row >= targets.shape[0]:
                break
            if accepted < glm53.GLM53_TARGET_VERIFY_WIDTH:
                targets[row, accepted - 1].add_(1000)
        for row, position in self.nextn_runtime.eos_positions.items():
            targets[row, position] = min(EOS_TOKEN_IDS)
            if position + 1 < targets.shape[1]:
                candidates[row, position + 1] = min(EOS_TOKEN_IDS)
        self.runtime.events.append(("target_verify", targets.clone()))
        return GLM53TargetOutput(
            torch.zeros((targets.numel(), 1)),
            targets.view(-1),
        )

    def decode_tokens(
        self,
        batch,
        state,
        workspace,
        process_group,
        all_reduce,
    ):
        tokens = torch.arange(batch.token_ids.shape[0], dtype=torch.int64).add_(100)
        workspace.endpoint.token.copy_(tokens)
        self.runtime.events.append(
            ("target_decode", batch.token_ids.shape[0], process_group, all_reduce)
        )
        return GLM53TargetOutput(workspace.endpoint.normalized, tokens)


class FakeNextNRuntime:
    def __init__(
        self,
        events: list[tuple],
        accept_counts: tuple[int, ...],
        eos_positions: dict[int, int],
    ) -> None:
        self.events = events
        self.accept_counts = accept_counts
        self.eos_positions = eos_positions
        self.state = SimpleNamespace(history="nextn_history")
        self.workspaces = {glm53.KDA_CHUNK_SIZE: "nextn_arena"}
        self.attention_tp_size = glm53.TP_SIZE
        self.verify_batches = {
            bucket: tuple(
                SimpleNamespace(
                    token_ids=torch.empty(
                        (bucket, glm53.GLM53_TARGET_VERIFY_WIDTH),
                        dtype=torch.int64,
                    )
                )
                for _ in range(2)
            )
            for bucket in glm53.GLM53_TARGET_DECODE_BATCH_SIZES
        }

    def reset_slot(self, state_slot: int) -> None:
        self.events.append(("nextn_reset_slot", state_slot))

    def capture_prefix(self, live_slot, snapshot_slot, tail_physical_block) -> None:
        self.events.append(
            ("nextn_capture", live_slot, snapshot_slot, tail_physical_block)
        )

    def restore_prefix(self, snapshot_slot, live_slot, tail_physical_block) -> None:
        self.events.append(
            ("nextn_restore", snapshot_slot, live_slot, tail_physical_block)
        )

    def draft_candidates(
        self, model, groups, staging_index, endpoint, process_group, all_reduce
    ):
        self.events.append(("nextn_draft", groups, staging_index))
        candidates = self.verify_batches[groups][staging_index].token_ids
        candidates.copy_(
            torch.arange(groups * glm53.GLM53_TARGET_VERIFY_WIDTH).view_as(candidates)
        )
        return candidates

    def verify_candidates(self, *args) -> None:
        self.events.append(("nextn_verify", args[3], args[4]))

    def commit_candidates(self, *args) -> None:
        self.events.append(("nextn_commit", args[0], args[1]))

    def stage_prefill(
        self, token_ids, target_normalized, sequence_lengths, start_tokens, state_slots
    ):
        pairs = sum(
            length - (start == 0)
            for length, start in zip(sequence_lengths, start_tokens, strict=True)
        )
        self.events.append(
            ("nextn_stage_prefill", sequence_lengths, start_tokens, state_slots)
        )
        return SimpleNamespace(
            token_ids=torch.zeros(pairs, dtype=torch.int64),
            hidden="nextn_hidden",
            sparse_mla="nextn_sparse" if pairs else None,
        )

    def publish_prefill(self, batch) -> None:
        self.events.append(("nextn_publish_prefill", batch.token_ids.shape[0]))

    def seed_candidates(self, *args) -> None:
        self.events.append(("nextn_seed", args[3], args[4]))


class FakeNextNModel:
    def __init__(self, events: list[tuple]) -> None:
        self.events = events

    def forward(self, *args, **kwargs) -> None:
        self.events.append(("nextn_forward", args[0].shape[0]))

    def prefill_empty(self, *args) -> None:
        self.events.append(("nextn_prefill_empty", args[2]))


class FakeCollectives:
    def __init__(self, rank: int) -> None:
        self.rank = rank

    def all_gather_into_tensor(self, output, input_, group) -> None:
        output.zero_()
        output.view(glm53.TP_SIZE, *input_.shape)[self.rank].copy_(input_)


class FakeEvent:
    def __init__(self, events: list[tuple], index: int) -> None:
        self.events = events
        self.index = index

    def record(self) -> None:
        self.events.append(("record", self.index))

    def synchronize(self) -> None:
        self.events.append(("synchronize", self.index))


def row(
    slot: int,
    token_start: int,
    token_ids: tuple[int, ...],
    *,
    sample: bool = False,
    start_block: int = 0,
    physical_blocks: tuple[int, ...] = (0,),
    max_accept_tokens: int | None = None,
    ignore_eos: bool = False,
    restore_slot: int | None = None,
    capture_slot: int | None = None,
) -> BatchPlanRow:
    return BatchPlanRow(
        slot,
        token_start,
        token_ids,
        len(token_ids) if max_accept_tokens is None else max_accept_tokens,
        sample,
        TableDelta(start_block, physical_blocks),
        ignore_eos,
        restore_slot,
        capture_slot,
    )


def plan(
    step_id: int,
    *rows: BatchPlanRow,
    graph_bucket: int | None = None,
) -> BatchPlan:
    return BatchPlan(step_id, (rows,), graph_bucket)


class GLM53WorkerTest(unittest.TestCase):
    def make_worker(
        self,
        sampled_values: tuple[int, ...] = (),
        *,
        accept_counts: tuple[int, ...] = (),
        eos_positions: dict[int, int] | None = None,
        rank: int = 0,
        parallelism: str = "tep4",
        speculative: bool = True,
    ):
        events: list[tuple] = []
        runtime = FakeRuntime(events, list(sampled_values))
        nextn_runtime = (
            FakeNextNRuntime(events, accept_counts, eos_positions or {})
            if speculative
            else None
        )
        if nextn_runtime is not None:
            nextn_runtime.attention_tp_size = (
                1 if parallelism == "dep4" else glm53.TP_SIZE
            )
        shape = (2, max(glm53.GLM53_TARGET_DECODE_BATCH_SIZES))
        lane_count = glm53.TP_SIZE if parallelism == "dep4" else 1
        live_slots = runtime.committed_lengths.shape[0]
        result_width = 1 + glm53.GLM53_TARGET_VERIFY_WIDTH
        staging = GLM53WorkerStaging(
            remaining_cpu=torch.zeros(shape, dtype=torch.int32),
            remaining=torch.zeros(shape, dtype=torch.int32),
            ignore_eos_cpu=torch.zeros(shape, dtype=torch.uint8),
            ignore_eos=torch.zeros(shape, dtype=torch.uint8),
            accepted=torch.zeros(shape, dtype=torch.int32),
            continuing=torch.zeros(shape, dtype=torch.uint8),
            seed_states_cpu=torch.zeros(shape, dtype=torch.int64),
            seed_states=torch.zeros(shape, dtype=torch.int64),
            seed_tokens=torch.zeros(shape, dtype=torch.int64),
            result_records=torch.zeros(
                (2, live_slots, result_width), dtype=torch.int64
            ),
            gathered_records=torch.zeros(
                (2, lane_count * live_slots, result_width), dtype=torch.int64
            ),
            result_cpu=torch.zeros(
                (2, lane_count, live_slots, result_width), dtype=torch.int64
            ),
        )
        worker = GLM53Worker(
            FakeModel(runtime, nextn_runtime),
            runtime,
            FakeNextNModel(events) if speculative else None,
            nextn_runtime,
            "world",
            SimpleNamespace(decode="decode_reduce", verify="verify_reduce"),
            staging,
            (FakeEvent(events, 0), FakeEvent(events, 1)),
            rank=rank,
            collectives=FakeCollectives(rank) if parallelism == "dep4" else None,
            parallelism=parallelism,
        )
        return worker, runtime, nextn_runtime, staging, events

    def execute(self, worker, batch_plan, *, collect: bool = True):
        events = worker.runtime.events

        def greedy_accept(
            candidates,
            targets,
            remaining,
            ignore_eos,
            active,
            accepted,
            output,
            continuing,
        ):
            events.append(("accept",))
            candidate_rows = candidates.clone()
            target_rows = targets.clone()
            accepted.zero_()
            output.zero_()
            continuing.zero_()
            for index in range(candidates.shape[0]):
                limit = int(remaining[index])
                if not active[index] or not limit:
                    continue
                candidates = tuple(int(token) for token in candidate_rows[index])
                targets = tuple(int(token) for token in target_rows[index])
                if ignore_eos[index]:
                    accepted_count = 1
                    while (
                        accepted_count < min(limit, glm53.GLM53_TARGET_VERIFY_WIDTH)
                        and candidates[accepted_count] == targets[accepted_count - 1]
                    ):
                        accepted_count += 1
                    tokens = targets[:accepted_count]
                else:
                    tokens = accept_glm53_greedy_chain(candidates, targets, limit)
                accepted[index] = len(tokens)
                output[index, : len(tokens)] = torch.tensor(tokens)
                continuing[index] = int(
                    tokens
                    and (ignore_eos[index] or tokens[-1] not in EOS_TOKEN_IDS)
                    and (
                        limit == glm53.GLM53_TARGET_VERIFY_WIDTH or len(tokens) < limit
                    )
                )

        with (
            patch(
                "infer.models.glm53_flash.ops.speculate.glm53_greedy_accept",
                greedy_accept,
            ),
            patch(
                "infer.models.glm53_flash.worker._nextn_workspace_view",
                return_value=SimpleNamespace(endpoint="nextn_endpoint"),
            ),
        ):
            pending = worker.dispatch(batch_plan)
        return worker.collect(pending) if collect else pending

    def test_cold_prefill_resets_both_states_before_staging_and_seeds_nextn(
        self,
    ) -> None:
        worker, runtime, _, _, events = self.make_worker((20,))
        rows = (
            row(3, 0, tuple(range(64)), physical_blocks=(7,)),
            row(4, 0, (10,), sample=True, physical_blocks=(8,)),
        )

        result = self.execute(worker, plan(0, *rows))

        self.assertEqual(
            result.lanes,
            ((StepResultRow(3, 64, (), False), StepResultRow(4, 1, (20,), False)),),
        )
        self.assertEqual(runtime.committed_lengths[[3, 4]].tolist(), [64, 1])
        self.assertEqual(
            [event[0] for event in events if event[0] != "table_delta"],
            [
                "reset_state",
                "nextn_reset_slot",
                "reset_state",
                "nextn_reset_slot",
                "stage_prefill",
                "prefill",
                "publish_prefill",
                "nextn_stage_prefill",
                "nextn_forward",
                "nextn_publish_prefill",
                "nextn_seed",
                "record",
                "synchronize",
            ],
        )

    def test_cold_slot_reuse_resets_nextn_even_for_one_token_prompt(self) -> None:
        worker, _, _, _, events = self.make_worker((20, 21))
        cold = row(3, 0, (10,), sample=True, physical_blocks=(7,))

        self.execute(worker, plan(0, cold))
        self.execute(worker, plan(1, cold))

        self.assertEqual(
            [event for event in events if event == ("nextn_reset_slot", 3)],
            [("nextn_reset_slot", 3), ("nextn_reset_slot", 3)],
        )

    def test_resumed_prefill_preserves_state_at_start_one_and_later(self) -> None:
        worker, runtime, _, _, events = self.make_worker()
        self.execute(worker, plan(0, row(3, 0, (1,), physical_blocks=(7,))))
        events.clear()

        self.execute(
            worker,
            plan(
                1,
                row(3, 1, (10,), start_block=1, physical_blocks=(8,)),
                row(4, 128, (11, 12), start_block=1, physical_blocks=(9,)),
            ),
        )

        self.assertFalse(any(event[0].endswith("reset_slot") for event in events))
        self.assertFalse(any(event[0] == "reset_state" for event in events))
        self.assertIn(("nextn_stage_prefill", (1, 2), (1, 128), (3, 4)), events)
        self.assertEqual(runtime.committed_lengths[[3, 4]].tolist(), [2, 130])

    def test_decode_precedes_prefix_restore_and_capture_wraps_prefill(self) -> None:
        worker, _, _, _, events = self.make_worker((20,))
        decoded = row(2, 129, (), sample=True, max_accept_tokens=1)
        promoted = row(
            3,
            256,
            tuple(range(128)),
            start_block=0,
            physical_blocks=(3, 7, 9),
            restore_slot=64,
            capture_slot=65,
        )
        sampled = row(
            4,
            1,
            (10,),
            sample=True,
            start_block=1,
            physical_blocks=(11,),
        )

        self.execute(worker, plan(0, decoded, promoted, sampled))

        names = [event[0] for event in events]
        self.assertLess(names.index("table_delta"), names.index("target_restore"))
        self.assertLess(names.index("nextn_commit"), names.index("target_restore"))
        self.assertLess(names.index("target_restore"), names.index("nextn_restore"))
        self.assertLess(names.index("nextn_restore"), names.index("stage_prefill"))
        self.assertLess(
            names.index("nextn_publish_prefill"), names.index("target_capture")
        )
        self.assertLess(names.index("target_capture"), names.index("nextn_capture"))
        self.assertLess(names.index("nextn_capture"), names.index("nextn_seed"))
        self.assertIn(("target_restore", 64, 3, 7), events)
        self.assertIn(("target_capture", 3, 65, 9, 384), events)
        self.assertIn(("nextn_capture", 3, 65, 9), events)
        self.assertFalse(any(event[0].endswith("reset_slot") for event in events))
        self.assertFalse(any(event[0] == "reset_state" for event in events))

    def test_prefix_capture_repeats_tail_allocated_by_an_earlier_chunk(self) -> None:
        worker, _, _, _, events = self.make_worker()
        self.execute(
            worker,
            plan(0, row(3, 0, tuple(range(64)), physical_blocks=(7,))),
        )
        events.clear()

        self.execute(
            worker,
            plan(
                1,
                row(
                    3,
                    64,
                    tuple(range(64)),
                    start_block=0,
                    physical_blocks=(7,),
                    capture_slot=64,
                ),
            ),
        )

        self.assertIn(("target_capture", 3, 64, 7, 128), events)
        self.assertIn(("nextn_capture", 3, 64, 7), events)

    def test_prefix_capture_rejects_a_missing_tail_delta(self) -> None:
        worker, _, _, _, _ = self.make_worker()

        with self.assertRaisesRegex(ValueError, "tail is absent"):
            self.execute(
                worker,
                plan(
                    0,
                    row(
                        3,
                        64,
                        tuple(range(64)),
                        start_block=1,
                        physical_blocks=(),
                        capture_slot=64,
                    ),
                ),
            )

    def test_decode_runs_target_then_nextn_transaction_in_required_order(self) -> None:
        worker, _, _, _, events = self.make_worker(accept_counts=(4, 2))
        rows = (
            row(2, 129, (), sample=True, max_accept_tokens=4),
            row(5, 17, (), sample=True, max_accept_tokens=4),
        )

        result = self.execute(worker, plan(1, *rows, graph_bucket=2))

        self.assertEqual(
            result.lanes[0],
            (
                StepResultRow(2, 4, (1, 2, 3, 4), False),
                StepResultRow(5, 2, (5, 1006), False),
            ),
        )
        self.assertEqual(
            [event[0] for event in events if event[0] != "table_delta"],
            [
                "stage_decode",
                "prepare_decode",
                "nextn_draft",
                "stage_verify",
                "target_verify",
                "accept",
                "target_publish",
                "nextn_verify",
                "nextn_commit",
                "record",
                "synchronize",
            ],
        )

    def test_target_only_decode_runs_one_target_token_without_nextn(self) -> None:
        worker, runtime, nextn_runtime, _, events = self.make_worker(speculative=False)
        runtime.committed_lengths[[2, 5]] = torch.tensor((129, 17), dtype=torch.int32)
        rows = (
            row(2, 129, (), sample=True, max_accept_tokens=1),
            row(5, 17, (), sample=True, max_accept_tokens=1),
        )

        result = self.execute(worker, plan(1, *rows, graph_bucket=2))

        self.assertIsNone(nextn_runtime)
        self.assertEqual(
            result.lanes[0],
            (
                StepResultRow(2, 1, (100,), False),
                StepResultRow(5, 1, (101,), False),
            ),
        )
        self.assertEqual(runtime.committed_lengths[[2, 5]].tolist(), [130, 18])
        self.assertEqual(
            [event[0] for event in events if event[0] != "table_delta"],
            [
                "stage_decode",
                "prepare_decode",
                "target_decode",
                "target_publish_decoded",
                "record",
                "synchronize",
            ],
        )

    def test_target_only_prefix_state_does_not_touch_nextn(self) -> None:
        worker, _, _, _, events = self.make_worker((20,), speculative=False)
        prefilling = row(
            3,
            0,
            tuple(range(128)),
            sample=True,
            physical_blocks=(7,),
            capture_slot=64,
        )

        result = self.execute(worker, plan(0, prefilling))

        self.assertEqual(result.lanes[0], (StepResultRow(3, 128, (20,), False),))
        self.assertIn(("target_capture", 3, 64, 7, 128), events)
        self.assertFalse(any(event[0].startswith("nextn_") for event in events))

    def test_target_only_dep4_empty_lane_runs_only_target_collectives(self) -> None:
        worker, _, _, _, events = self.make_worker(
            rank=1,
            parallelism="dep4",
            speculative=False,
        )
        lanes = (
            (row(0, 0, (1, 2), physical_blocks=(0,)),),
            (),
            (),
            (),
        )

        self.execute(worker, BatchPlan(0, lanes, None), collect=False)

        self.assertIn(("target_prefill_empty", (2, 0, 0, 0)), events)
        self.assertFalse(any(event[0].startswith("nextn_") for event in events))

    def test_target_only_rejects_multi_token_decode_acceptance(self) -> None:
        worker, _, _, _, events = self.make_worker(speculative=False)

        with self.assertRaisesRegex(ValueError, "decode width"):
            worker.dispatch(
                plan(
                    0,
                    row(1, 1, (), sample=True, max_accept_tokens=2),
                    graph_bucket=1,
                )
            )

        self.assertEqual(events, [])

    def test_installed_decode_graph_replaces_only_the_fixed_device_step(self) -> None:
        worker, _, _, staging, events = self.make_worker()
        graphs = {
            (bucket, parity): Mock()
            for bucket in glm53.GLM53_TARGET_DECODE_BATCH_SIZES
            for parity in range(2)
        }
        worker.install_decode_graphs(graphs)

        worker.dispatch(
            plan(
                1,
                row(2, 1, (), sample=True, max_accept_tokens=3),
                row(
                    5,
                    1,
                    (),
                    sample=True,
                    max_accept_tokens=2,
                    ignore_eos=True,
                ),
                graph_bucket=2,
            )
        )

        graphs[2, 1].replay.assert_called_once_with()
        self.assertTrue(
            all(
                graph.replay.call_count == int(key == (2, 1))
                for key, graph in graphs.items()
            )
        )
        self.assertEqual(staging.remaining[1, :2].tolist(), [3, 2])
        self.assertEqual(staging.ignore_eos[1, :2].tolist(), [0, 1])
        self.assertEqual(
            [event[0] for event in events if event[0] != "table_delta"],
            ["stage_decode", "record"],
        )

    def test_overlapped_successor_reads_device_committed_length(self) -> None:
        worker, runtime, _, _, events = self.make_worker(accept_counts=(2,))
        runtime.committed_lengths[3] = 1
        first_row = row(3, 1, (), sample=True, max_accept_tokens=4)
        projected_successor = row(3, 5, (), sample=True, max_accept_tokens=4)

        first = self.execute(worker, plan(1, first_row, graph_bucket=1), collect=False)
        second = self.execute(
            worker, plan(2, projected_successor, graph_bucket=1), collect=False
        )

        staged = [event for event in events if event[0] == "stage_decode"]
        self.assertEqual(
            staged,
            [
                ("stage_decode", (2,), (3,), 1),
                ("stage_decode", (4,), (3,), 0),
            ],
        )
        self.assertEqual(runtime.committed_lengths[3], 5)
        worker.collect(first)
        worker.collect(second)

    def test_decode_covers_acceptance_width_budget_eos_and_dummy_rows(self) -> None:
        eos = min(EOS_TOKEN_IDS)
        worker, _, _, _, _ = self.make_worker(
            accept_counts=(1, 2, 3, 3), eos_positions={2: 1, 3: 1}
        )
        rows = tuple(
            row(
                slot,
                1,
                (),
                sample=True,
                max_accept_tokens=(slot + 1 if slot < 2 else 4),
                ignore_eos=slot == 3,
            )
            for slot in range(4)
        )

        result = self.execute(worker, plan(0, *rows, graph_bucket=4))

        self.assertEqual(
            result.lanes[0],
            (
                StepResultRow(0, 1, (1001,), False),
                StepResultRow(1, 2, (5, 1006), False),
                StepResultRow(2, 2, (9, eos), True),
                StepResultRow(3, 3, (13, eos, 1015), False),
            ),
        )

    def test_mixed_plan_decodes_before_prefill(self) -> None:
        worker, _, _, _, events = self.make_worker((31,), accept_counts=(1,))
        mixed = (
            row(3, 1, (), sample=True, max_accept_tokens=4),
            row(4, 0, (11,), sample=True, physical_blocks=(1,)),
        )

        result = self.execute(worker, plan(1, *mixed))

        self.assertEqual(
            result.lanes[0],
            (
                StepResultRow(3, 1, (1001,), False),
                StepResultRow(4, 1, (31,), False),
            ),
        )
        names = [event[0] for event in events]
        self.assertLess(names.index("nextn_commit"), names.index("stage_prefill"))

    def test_mixed_plan_replays_installed_decode_graph(self) -> None:
        worker, _, _, _, events = self.make_worker((31,), accept_counts=(1,))
        graphs = {
            (bucket, parity): Mock()
            for bucket in glm53.GLM53_TARGET_DECODE_BATCH_SIZES
            for parity in range(2)
        }
        worker.install_decode_graphs(graphs)
        mixed = (
            row(3, 1, (), sample=True, max_accept_tokens=4),
            row(4, 0, (11,), sample=True, physical_blocks=(1,)),
        )

        worker.dispatch(plan(1, *mixed))

        graphs[1, 1].replay.assert_called_once_with()
        self.assertIn("stage_prefill", [event[0] for event in events])

    def test_dep4_selects_its_lane_and_uses_the_global_decode_bucket(self) -> None:
        worker, _, _, _, _ = self.make_worker(rank=2, parallelism="dep4")
        lanes = tuple(
            tuple(
                row(slot, 1, (), sample=True, max_accept_tokens=1)
                for slot in range(lane + 1)
            )
            for lane in range(glm53.TP_SIZE)
        )
        batch_plan = BatchPlan(0, lanes, graph_bucket=4)

        self.assertEqual(worker._validate(batch_plan), lanes[2])

    def test_dep4_empty_lane_aligns_nextn_prefill_collectives(self) -> None:
        worker, _, _, _, events = self.make_worker(rank=1, parallelism="dep4")
        lanes = (
            (row(0, 0, (1, 2), physical_blocks=(0,)),),
            (),
            (),
            (),
        )

        self.execute(worker, BatchPlan(0, lanes, None), collect=False)

        # Empty lanes still enter the NextN prefill collective when any lane
        # has NextN work, mirroring the unconditional target path. Skipping it
        # would desync the per-rank kernel sequence.
        self.assertEqual(
            [event for event in events if event[0].endswith("prefill_empty")],
            [
                ("target_prefill_empty", (2, 0, 0, 0)),
                ("nextn_prefill_empty", (1, 0, 0, 0)),
            ],
        )

    def test_dep4_one_token_lane_skips_nextn_history_but_seeds_candidates(
        self,
    ) -> None:
        worker, _, _, _, events = self.make_worker((20,), rank=1, parallelism="dep4")
        lanes = (
            (row(0, 0, (1, 2), sample=True, physical_blocks=(0,)),),
            (row(1, 0, (3,), sample=True, physical_blocks=(1,)),),
            (),
            (),
        )

        self.execute(worker, BatchPlan(0, lanes, None), collect=False)

        self.assertNotIn("nextn_prefill_history", [event[0] for event in events])
        self.assertIn("nextn_seed", [event[0] for event in events])

    def test_dep4_empty_lane_still_runs_the_global_decode_bucket(self) -> None:
        worker, _, _, _, events = self.make_worker(
            accept_counts=(1,), rank=1, parallelism="dep4"
        )
        lanes = (
            (row(0, 1, (), sample=True, max_accept_tokens=1),),
            (),
            (),
            (),
        )

        self.execute(worker, BatchPlan(0, lanes, 1), collect=False)

        self.assertIn(("stage_decode", (), (), 0), events)
        self.assertIn("target_verify", [event[0] for event in events])

    def test_validation_precedes_model_execution(self) -> None:
        worker, _, _, _, events = self.make_worker()
        invalid = (
            (BatchPlan(0, (), None), "wrong lane count"),
            (plan(0), "at least one row"),
            (plan(0, row(1, 0, (1,)), row(1, 0, (2,))), "appears twice"),
            (plan(0, row(1, 0, ())), "start with prefill"),
            (
                plan(
                    0, row(1, 1, (), sample=True, max_accept_tokens=5), graph_bucket=1
                ),
                "verify width",
            ),
            (plan(0, row(1, 0, (1,)), graph_bucket=1), "global graph bucket"),
        )
        for invalid_plan, error in invalid:
            with self.subTest(error=error), self.assertRaisesRegex(ValueError, error):
                worker.dispatch(invalid_plan)
        self.assertEqual(events, [])

    def test_staging_failure_stops_before_model_execution(self) -> None:
        worker, runtime, _, _, events = self.make_worker()
        runtime.stage_error = ValueError("staging failed")

        with self.assertRaisesRegex(ValueError, "staging failed"):
            worker.dispatch(plan(0, row(7, 1, (10,))))

        self.assertEqual(
            [event[0] for event in events], ["table_delta", "stage_prefill"]
        )


if __name__ == "__main__":
    unittest.main()

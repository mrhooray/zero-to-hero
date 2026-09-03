import unittest
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.gpu

import torch

from infer.models.deepseek_v4_flash.worker import DeepSeekV4TargetWorker
from infer.models.deepseek_v4_flash import model as deepseek_v4
from infer.protocol import BatchPlan, BatchPlanRow, StepResultRow, TableDelta


class FakeVerifyRuntime:
    def __init__(self, bucket: int, events: list[tuple], result_records) -> None:
        self.events = events
        self.outputs = [(41, 42, 43, 44), (1,), *((7,),) * max(0, bucket - 2)]
        self.result_records = result_records

    def stage(self, controls) -> None:
        self.events.append(("stage_verify", controls))
        for control, output in zip(controls, self.outputs):
            slot, _, _, remaining, _ = control
            accepted = min(len(output), remaining)
            record = self.result_records[slot]
            record.zero_()
            record[0] = accepted
            record[1 : accepted + 1] = torch.tensor(output[:accepted])


class FakeTargetDecodeRuntime:
    def __init__(self, events: list[tuple], result_records) -> None:
        self.events = events
        self.result_records = result_records

    def stage(self, controls) -> None:
        self.events.append(("stage_target", controls))
        for slot, _, _ in controls:
            self.result_records[slot, 0] = 1
            self.result_records[slot, 1] = 61 + slot


class FakeRuntime:
    def __init__(self, events: list[tuple], *, target_only=False, live_slots=8) -> None:
        self.live_slots = live_slots
        self.history_blocks = 16
        self.events = events
        self.sampled_tokens = torch.zeros(self.live_slots, dtype=torch.int64)
        self.result_records = torch.zeros(
            (
                self.live_slots,
                2 if target_only else 1 + deepseek_v4.DSPARK_VERIFY_WIDTH,
            ),
            dtype=torch.int64,
        )
        keys = tuple(
            (bucket, parity)
            for bucket in (0, *deepseek_v4.DECODE_BATCH_SIZES)
            for parity in (0, 1)
        )
        self.verify = (
            {}
            if target_only
            else {
                key: FakeVerifyRuntime(key[0], events, self.result_records)
                for key in keys
            }
        )
        self.decode = (
            {key: FakeTargetDecodeRuntime(events, self.result_records) for key in keys}
            if target_only
            else {}
        )
        self.prefill_outputs = (51, 52, 53, 54)
        self.prefill_rows = ()

    def stage_prefill(
        self,
        staging_index,
        token_ids,
        start_tokens,
        state_slots,
        table_deltas,
        sample_rows,
    ):
        self.prefill_rows = state_slots, sample_rows
        self.events.append(
            (
                "stage_prefill",
                staging_index,
                token_ids,
                start_tokens,
                state_slots,
                table_deltas,
                sample_rows,
            )
        )
        return SimpleNamespace(
            token_count=sum(map(len, token_ids)),
            sample_count=len(sample_rows),
            dspark_seed_count=0,
        )

    def reset_state(self, slot) -> None:
        self.events.append(("reset", slot))

    def capture_prefix(self, live_slot, snapshot_slot, tail_physical_block) -> None:
        self.events.append(
            ("target_capture", live_slot, snapshot_slot, tail_physical_block)
        )

    def restore_prefix(self, snapshot_slot, live_slot, tail_physical_block) -> None:
        self.events.append(
            ("target_restore", snapshot_slot, live_slot, tail_physical_block)
        )

    def publish_prefill(self, batch) -> None:
        self.events.append(("publish_prefill", batch.sample_count))
        state_slots, sample_rows = self.prefill_rows
        for output, row_index in zip(self.prefill_outputs, sample_rows):
            slot = state_slots[row_index]
            self.sampled_tokens[slot] = output
            self.result_records[slot].zero_()
            self.result_records[slot, 1] = output


class FakeModel:
    def __init__(self, events: list[tuple]) -> None:
        self.events = events

    def prefill(self, runtime, batch) -> None:
        self.events.append(("prefill", batch.token_count))


class FakeDSparkRuntime:
    def __init__(self, events: list[tuple]) -> None:
        self.events = events
        self.batches = {
            (bucket, parity): object()
            for bucket in (0, *deepseek_v4.DECODE_BATCH_SIZES)
            for parity in (0, 1)
        }

    def batch(self, *key):
        self.events.append(("batch", key))
        return self.batches[key]

    def reset_state(self, slot) -> None:
        self.events.append(("dspark_reset", slot))

    def capture_prefix(self, live_slot, snapshot_slot) -> None:
        self.events.append(("dspark_capture", live_slot, snapshot_slot))

    def restore_prefix(self, snapshot_slot, live_slot) -> None:
        self.events.append(("dspark_restore", snapshot_slot, live_slot))


class FakeDSparkModel:
    def __init__(self, events: list[tuple]) -> None:
        self.events = events

    def seed_prefill(self, _dspark_runtime, _target_runtime, batch) -> None:
        self.events.append(("dspark_seed", batch.dspark_seed_count))


class FakeGraph:
    def __init__(self, key, events: list[tuple]) -> None:
        self.key = key
        self.events = events

    def replay(self) -> None:
        self.events.append(("replay", self.key))


class FakeEvent:
    def __init__(self, parity: int, events: list[tuple]) -> None:
        self.parity = parity
        self.events = events

    def record(self) -> None:
        self.events.append(("record", self.parity))

    def synchronize(self) -> None:
        self.events.append(("sync", self.parity))


class FakeCollectives:
    def __init__(self, rank: int, events: list[tuple], remote_lanes=()) -> None:
        self.rank = rank
        self.events = events
        self.remote_lanes = remote_lanes
        self.calls = []

    def all_gather_into_tensor(self, output, local, *, group) -> None:
        self.events.append(("gather",))
        self.calls.append((tuple(output.shape), tuple(local.shape), group))
        gathered = torch.zeros((4, *local.shape), dtype=local.dtype)
        for lane, rows in enumerate(self.remote_lanes):
            for result in rows:
                gathered[lane, result.slot, 0] = result.accepted_count
                for index, token in enumerate(result.output_token_ids, 1):
                    gathered[lane, result.slot, index] = token
        gathered[self.rank].copy_(local)
        output.copy_(gathered.view_as(output))

    def all_gather_object(self, *_args, **_kwargs) -> None:
        raise AssertionError("Python-object collectives are not allowed")


def row(
    slot: int,
    position: int,
    *,
    tokens=(),
    block=(),
    sample=True,
    max_accept=None,
    ignore_eos=False,
    start_block=None,
    restore_slot=None,
    capture_slot=None,
) -> BatchPlanRow:
    return BatchPlanRow(
        slot,
        position,
        tokens,
        len(tokens)
        if tokens
        else deepseek_v4.DSPARK_VERIFY_WIDTH
        if max_accept is None
        else max_accept,
        sample,
        TableDelta(position // 128 if start_block is None else start_block, block),
        ignore_eos,
        restore_slot,
        capture_slot,
    )


def make_worker(
    rank: int,
    remote_lanes=((), (), (), ()),
    parallelism="dep4",
    *,
    target_only=False,
    live_slots=8,
):
    events = []
    runtime = FakeRuntime(events, target_only=target_only, live_slots=live_slots)
    dspark_runtime = None if target_only else FakeDSparkRuntime(events)
    decode = runtime.decode if target_only else runtime.verify
    graphs = {key: FakeGraph(key, events) for key in decode}
    collectives = FakeCollectives(rank, events, remote_lanes)
    worker = DeepSeekV4TargetWorker(
        FakeModel(events),
        runtime,
        None if target_only else FakeDSparkModel(events),
        dspark_runtime,
        graphs,
        rank,
        "world",
        collectives,
        (FakeEvent(0, events), FakeEvent(1, events)),
        parallelism,
    )
    return worker, runtime, collectives, events


class DeepSeekV4TargetWorkerTest(unittest.TestCase):
    def test_local_bucket_stages_compact_rows_and_gathers_lane_results(self) -> None:
        remote = (
            (),
            (),
            tuple(StepResultRow(index, 1, (9,), False) for index in range(7)),
            (StepResultRow(0, 1, (8,), False),),
        )
        worker, _, collectives, events = make_worker(1, remote)
        lanes = (
            (),
            (row(2, 127, block=(11,)), row(5, 128)),
            tuple(row(index, 9) for index in range(7)),
            (row(0, 9),),
        )
        plan = BatchPlan(3, lanes, 8)

        result = worker.collect(worker.dispatch(plan))

        self.assertEqual(
            events,
            [
                ("stage_verify", ((2, 0, 11, 6, False), (5, 1, -1, 6, False))),
                ("replay", (2, 1)),
                ("gather",),
                ("record", 1),
                ("sync", 1),
            ],
        )
        self.assertEqual(
            result.lanes[1],
            (
                StepResultRow(2, 4, (41, 42, 43, 44), False),
                StepResultRow(5, 1, (1,), True),
            ),
        )
        self.assertEqual(result.lanes[2:], remote[2:])
        self.assertEqual(collectives.calls, [((32, 7), (8, 7), "world")])

    def test_tep4_runs_one_shared_lane_without_result_gather(self) -> None:
        worker, _, collectives, events = make_worker(2, parallelism="tep4")

        result = worker.collect(
            worker.dispatch(BatchPlan(3, ((row(2, 127, block=(11,)),),), 1))
        )

        self.assertEqual(
            result.lanes,
            ((StepResultRow(2, 4, (41, 42, 43, 44), False),),),
        )
        self.assertEqual(
            events,
            [
                ("stage_verify", ((2, 0, 11, 6, False),)),
                ("replay", (1, 1)),
                ("record", 1),
                ("sync", 1),
            ],
        )
        self.assertFalse(collectives.calls)
        with self.assertRaisesRegex(ValueError, "lane count"):
            worker.dispatch(BatchPlan(4, ((), (), (), ()), 0))

    def test_target_only_uses_one_token_graph_and_skips_dspark_prefill(self) -> None:
        worker, _, collectives, events = make_worker(0, target_only=True)
        plan = BatchPlan(
            2,
            (
                (
                    row(2, 127, block=(11,), max_accept=1),
                    row(5, 0, tokens=(3, 4), block=(12,)),
                ),
                (),
                (),
                (),
            ),
            None,
        )

        result = worker.collect(worker.dispatch(plan))

        self.assertEqual(
            result.lanes[0],
            (
                StepResultRow(2, 1, (63,), False),
                StepResultRow(5, 2, (51,), False),
            ),
        )
        self.assertIn(("stage_target", ((2, 0, 11),)), events)
        self.assertIn(("replay", (1, 0)), events)
        self.assertFalse(any(event[0].startswith("dspark") for event in events))
        self.assertEqual(collectives.calls, [((32, 2), (8, 2), "world")])

        with self.assertRaisesRegex(ValueError, "decode acceptance exceeds its width"):
            worker.dispatch(BatchPlan(3, ((row(2, 128, max_accept=2),), (), (), ()), 1))

    def test_dep4_global_c128_replays_local_b32_target_graph(self) -> None:
        worker, _, _, events = make_worker(0, target_only=True, live_slots=64)
        lane = tuple(row(slot, 128, max_accept=1) for slot in range(32))

        worker.dispatch(BatchPlan(2, (lane, lane, lane, lane), 32))

        stage = next(event for event in events if event[0] == "stage_target")
        self.assertEqual(len(stage[1]), 32)
        self.assertIn(("replay", (32, 0)), events)

    def test_empty_local_lane_replays_b0_for_collective_order(self) -> None:
        remote = ((), (StepResultRow(0, 1, (9,), False),), (), ())
        worker, _, _, events = make_worker(0, remote)
        plan = BatchPlan(2, ((), (row(0, 9),), (), ()), 1)

        result = worker.collect(worker.dispatch(plan))

        self.assertEqual(result.lanes, remote)
        self.assertEqual(
            events,
            [
                ("stage_verify", ()),
                ("replay", (0, 0)),
                ("gather",),
                ("record", 0),
                ("sync", 0),
            ],
        )

    def test_mixed_plan_runs_global_decode_then_local_packed_prefill(self) -> None:
        worker, _, _, events = make_worker(0)
        lanes = (
            (
                row(2, 127, block=(11,)),
                row(5, 0, tokens=(3, 4), block=(12,)),
                row(6, 128, tokens=(5,), block=(13,), sample=False),
            ),
            (row(0, 9),),
            tuple(row(index, 9) for index in range(7)),
            (),
        )

        result = worker.collect(worker.dispatch(BatchPlan(4, lanes, None)))

        self.assertEqual(
            events,
            [
                ("stage_verify", ((2, 0, 11, 6, False),)),
                ("replay", (1, 0)),
                (
                    "stage_prefill",
                    0,
                    ((3, 4), (5,)),
                    (0, 128),
                    (5, 6),
                    ((0, (12,)), (1, (13,))),
                    (0,),
                ),
                ("reset", 5),
                ("dspark_reset", 5),
                ("prefill", 3),
                ("dspark_seed", 0),
                ("publish_prefill", 1),
                ("gather",),
                ("record", 0),
                ("sync", 0),
            ],
        )
        self.assertEqual(
            result.lanes[0],
            (
                StepResultRow(2, 4, (41, 42, 43, 44), False),
                StepResultRow(5, 2, (51,), False),
                StepResultRow(6, 1, (), False),
            ),
        )

    def test_pure_prefill_skips_the_decode_graph(self) -> None:
        worker, _, _, events = make_worker(0)
        plan = BatchPlan(
            5,
            ((row(3, 0, tokens=(8, 9), block=(4,)),), (), (), ()),
            None,
        )

        result = worker.collect(worker.dispatch(plan))

        self.assertEqual(
            events,
            [
                (
                    "stage_prefill",
                    1,
                    ((8, 9),),
                    (0,),
                    (3,),
                    ((0, (4,)),),
                    (0,),
                ),
                ("reset", 3),
                ("dspark_reset", 3),
                ("prefill", 2),
                ("dspark_seed", 0),
                ("publish_prefill", 1),
                ("gather",),
                ("record", 1),
                ("sync", 1),
            ],
        )
        self.assertEqual(result.lanes[0], (StepResultRow(3, 2, (51,), False),))

    def test_prefix_restore_and_capture_wrap_prefill_publication(self) -> None:
        worker, _, _, events = make_worker(0)
        promoted = row(
            3,
            256,
            tokens=tuple(range(128)),
            block=(3, 7, 9),
            sample=False,
            start_block=0,
            restore_slot=8,
            capture_slot=9,
        )

        worker.collect(worker.dispatch(BatchPlan(6, ((promoted,), (), (), ()), None)))

        names = [event[0] for event in events]
        self.assertLess(names.index("stage_prefill"), names.index("target_restore"))
        self.assertLess(names.index("target_restore"), names.index("dspark_restore"))
        self.assertLess(names.index("dspark_restore"), names.index("prefill"))
        self.assertLess(names.index("dspark_seed"), names.index("target_capture"))
        self.assertLess(names.index("publish_prefill"), names.index("target_capture"))
        self.assertLess(names.index("target_capture"), names.index("dspark_capture"))
        self.assertLess(names.index("dspark_capture"), names.index("record"))
        self.assertIn(("target_restore", 8, 3, 7), events)
        self.assertIn(("target_capture", 3, 9, 9), events)
        self.assertFalse(any(event[0] in {"reset", "dspark_reset"} for event in events))

    def test_prefix_capture_resolves_a_repeated_existing_tail_block(self) -> None:
        worker, _, _, events = make_worker(0)
        capture = row(
            3,
            257,
            tokens=tuple(range(127)),
            block=(9,),
            sample=False,
            start_block=2,
            capture_slot=8,
        )

        worker.dispatch(BatchPlan(6, ((capture,), (), (), ()), None))

        self.assertIn(("target_capture", 3, 8, 9), events)
        self.assertIn(("dspark_capture", 3, 8), events)

    def test_long_prefix_restore_accepts_its_complete_initial_table(self) -> None:
        for start, block_count in ((8_192, 65), (32_768, 257), (69_632, 545)):
            with self.subTest(start=start):
                worker, runtime, _, events = make_worker(0)
                runtime.history_blocks = 600
                restored = row(
                    3,
                    start,
                    tokens=(7,),
                    block=tuple(range(block_count)),
                    sample=False,
                    start_block=0,
                    restore_slot=8,
                )

                worker.dispatch(BatchPlan(6, ((restored,), (), (), ()), None))

                self.assertIn(("target_restore", 8, 3, start // 128 - 1), events)

    def test_prefix_snapshot_descriptors_require_a_resolvable_prefill_tail(
        self,
    ) -> None:
        worker, _, _, events = make_worker(0)
        invalid = (
            (row(3, 128, restore_slot=8), 1, "prefill row"),
            (
                row(
                    3,
                    256,
                    tokens=(7,),
                    block=(3, 7, 9),
                    start_block=0,
                    capture_slot=9,
                ),
                None,
                "complete history block",
            ),
            (
                row(
                    3,
                    256,
                    tokens=(7,),
                    block=(3,),
                    start_block=0,
                    restore_slot=8,
                ),
                None,
                "absent",
            ),
        )
        for step, (candidate, bucket, error) in enumerate(invalid):
            with self.subTest(error=error), self.assertRaisesRegex(ValueError, error):
                worker.dispatch(BatchPlan(step, ((candidate,), (), (), ()), bucket))
        self.assertFalse(events)

    def test_ignore_eos_keeps_a_decode_token_nonterminal(self) -> None:
        worker, runtime, _, _ = make_worker(0)
        runtime.verify[1, 0].outputs[0] = (1,)
        plan = BatchPlan(
            0,
            ((row(2, 9, ignore_eos=True),), (), (), ()),
            1,
        )

        result = worker.collect(worker.dispatch(plan))

        self.assertEqual(result.lanes[0], (StepResultRow(2, 1, (1,), False),))

    def test_two_inflight_steps_keep_parity_local_results(self) -> None:
        worker, runtime, _, _ = make_worker(0)
        first = worker.dispatch(BatchPlan(0, ((row(2, 9),), (), (), ()), 1))
        runtime.verify[1, 1].outputs[0] = (77, 78)
        second = worker.dispatch(BatchPlan(1, ((row(2, 10),), (), (), ()), 1))

        first_result = worker.collect(first)
        second_result = worker.collect(second)

        self.assertEqual(first_result.lanes[0][0].output_token_ids, (41, 42, 43, 44))
        self.assertEqual(second_result.lanes[0][0].output_token_ids, (77, 78))

    def test_empty_local_lane_runs_b0_inputs_for_both_global_phases(self) -> None:
        remote = (
            (StepResultRow(0, 1, (), False),),
            (),
            (StepResultRow(0, 1, (9,), False),),
            (),
        )
        worker, _, _, events = make_worker(3, remote)
        lanes = (
            (row(0, 0, tokens=(3,), sample=False),),
            (),
            (row(0, 9),),
            (),
        )

        result = worker.collect(worker.dispatch(BatchPlan(6, lanes, None)))

        self.assertEqual(result.lanes, remote)
        self.assertEqual(
            events,
            [
                ("stage_verify", ()),
                ("replay", (0, 0)),
                ("stage_prefill", 0, (), (), (), (), ()),
                ("prefill", 0),
                ("dspark_seed", 0),
                ("publish_prefill", 0),
                ("gather",),
                ("record", 0),
                ("sync", 0),
            ],
        )

    def test_rejects_wrong_phase_bucket_and_packed_budget(self) -> None:
        worker, _, _, _ = make_worker(0)
        with self.assertRaisesRegex(ValueError, "graph bucket"):
            worker.dispatch(BatchPlan(0, ((row(0, 0, tokens=(3,)),), (), (), ()), 1))
        with self.assertRaisesRegex(ValueError, "graph bucket"):
            worker.dispatch(BatchPlan(0, ((row(0, 9),), (), (), ()), 7))
        with self.assertRaisesRegex(ValueError, "packed budget"):
            worker.dispatch(
                BatchPlan(
                    0,
                    (
                        tuple(
                            row(index, 0, tokens=(3,), sample=False)
                            for index in range(deepseek_v4.MAX_PREFILL_REQUESTS + 1)
                        ),
                        (),
                        (),
                        (),
                    ),
                    None,
                )
            )

        with self.assertRaisesRegex(ValueError, "rank"):
            make_worker(4)


if __name__ == "__main__":
    unittest.main()

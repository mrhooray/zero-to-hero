import unittest

from infer.protocol import Request, StepResult, StepResultRow
from infer.scheduler import Scheduler
from infer.state import StateManager


def make_request(
    request_id: int,
    prompt_token_ids: tuple[int, ...],
    max_output_tokens: int,
    *,
    lane: int | None = None,
    ignore_eos: bool = False,
) -> Request:
    return Request(
        request_id,
        prompt_token_ids,
        max_output_tokens,
        temperature=0.7,
        top_p=0.9,
        seed=request_id + 100,
        ignore_eos=ignore_eos,
        lane=lane,
    )


def make_scheduler(
    *,
    state_manager: StateManager | None = None,
    lane_count: int = 1,
    history_blocks: int = 8,
    live_slots: int = 4,
    snapshot_slots: int = 0,
    token_budget: int = 4,
    prefill_chunk_size: int = 4,
    decode_width: int = 1,
    max_batch_size: int = 4,
    max_prefill_rows: int | None = None,
    max_decode_ticks_between_prefills: int = 4,
    graph_buckets: tuple[int, ...] = (1, 2, 4),
    max_queued_requests: int = 8,
    single_resident_prefill_chunk_size: int | None = None,
) -> Scheduler:
    return Scheduler(
        state_manager
        or StateManager(
            history_block_count=history_blocks,
            live_slot_count=live_slots,
            snapshot_slot_count=snapshot_slots,
            lane_count=lane_count,
            history_block_tokens=128,
        ),
        token_budget=token_budget,
        prefill_chunk_size=prefill_chunk_size,
        decode_width=decode_width,
        max_batch_size=max_batch_size,
        max_prefill_rows=(
            max_batch_size if max_prefill_rows is None else max_prefill_rows
        ),
        max_decode_ticks_between_prefills=max_decode_ticks_between_prefills,
        graph_buckets=graph_buckets,
        max_queued_requests=max_queued_requests,
        single_resident_prefill_chunk_size=single_resident_prefill_chunk_size,
    )


def step_result(
    plan,
    *,
    accepts: dict[tuple[int, int], int] | None = None,
    finished: set[tuple[int, int]] | None = None,
) -> StepResult:
    accepts = accepts or {}
    finished = finished or set()
    lanes = []
    for lane, rows in enumerate(plan.lanes):
        results = []
        for index, row in enumerate(rows):
            key = (lane, index)
            accepted = accepts.get(
                key,
                len(row.token_ids) if row.token_ids else min(1, row.max_accept_tokens),
            )
            output_count = (
                accepted
                if not row.token_ids
                else int(row.sample and accepted == len(row.token_ids))
            )
            results.append(
                StepResultRow(
                    row.slot,
                    accepted,
                    tuple(
                        1000 + lane * 100 + index * 10 + i for i in range(output_count)
                    ),
                    key in finished,
                )
            )
        lanes.append(tuple(results))
    return StepResult(plan.step_id, tuple(lanes), elapsed_ns=10)


class SchedulerTest(unittest.TestCase):
    def test_dep_priming_waves_restore_group_prefixes_on_their_lanes(self) -> None:
        scheduler = make_scheduler(
            lane_count=4,
            history_blocks=128,
            live_slots=1,
            snapshot_slots=2,
            token_budget=8_192,
            prefill_chunk_size=8_192,
            max_batch_size=1,
            graph_buckets=(1,),
        )
        group_prefixes = tuple(
            tuple(range(group * 10_000, group * 10_000 + 7_936)) for group in range(4)
        )
        waves = tuple(
            tuple(
                make_request(
                    wave * 4 + group + 1,
                    group_prefixes[group]
                    + tuple(
                        range(
                            100_000 + wave * 2_000 + group * 256,
                            100_000 + wave * 2_000 + (group + 1) * 256,
                        )
                    ),
                    1,
                )
                for group in range(4)
            )
            for wave in range(3)
        )
        self.assertTrue(
            all(
                len(request.prompt_token_ids) == 8_192
                for wave in waves
                for request in wave
            )
        )

        first = waves[0]
        for request in first:
            scheduler.submit(request)
        first_lanes = tuple(request.lane for request in first)
        self.assertEqual(set(first_lanes), set(range(4)))
        while (plan := scheduler.schedule()) is not None:
            self.assertEqual(tuple(map(len, plan.lanes)), (1, 1, 1, 1))
            scheduler.commit(step_result(plan))

        second = waves[1]
        for group in (2, 0, 3, 1):
            scheduler.submit(second[group])
        self.assertEqual(
            tuple(request.lane for request in second),
            first_lanes,
        )
        promoted = scheduler.schedule()
        assert promoted is not None
        self.assertEqual(tuple(map(len, promoted.lanes)), (1, 1, 1, 1))
        self.assertTrue(
            all(
                row.token_start == 0
                and len(row.token_ids) == 7_936
                and row.capture_slot is not None
                for lane in promoted.lanes
                for row in lane
            )
        )
        scheduler.commit(step_result(promoted))
        while (plan := scheduler.schedule()) is not None:
            self.assertEqual(tuple(map(len, plan.lanes)), (1, 1, 1, 1))
            scheduler.commit(step_result(plan))
        self.assertEqual(tuple(request.cached_tokens for request in second), (0,) * 4)

        hits = waves[2]
        for group in (1, 3, 0, 2):
            scheduler.submit(hits[group])
        self.assertEqual(
            tuple(request.lane for request in hits),
            first_lanes,
        )
        restored = scheduler.schedule()
        assert restored is not None
        self.assertEqual(tuple(map(len, restored.lanes)), (1, 1, 1, 1))
        self.assertTrue(
            all(
                row.token_start == 7_936 and row.restore_slot is not None
                for lane in restored.lanes
                for row in lane
            )
        )
        scheduler.commit(step_result(restored))
        self.assertEqual(tuple(request.cached_tokens for request in hits), (7_936,) * 4)

    def test_prefix_hit_is_lane_affine_and_restores_only_the_uncached_tail(
        self,
    ) -> None:
        scheduler = make_scheduler(
            lane_count=2,
            history_blocks=8,
            live_slots=2,
            snapshot_slots=2,
            token_budget=128,
            prefill_chunk_size=128,
            max_batch_size=2,
            graph_buckets=(1, 2),
        )
        prompt = tuple(range(257))
        first = make_request(1, prompt, 1, lane=1)
        scheduler.submit(first)

        plans = []
        while (plan := scheduler.schedule()) is not None:
            plans.append(plan)
            scheduler.commit(step_result(plan))

        self.assertEqual(
            [len(plan.lanes[1][0].token_ids) for plan in plans], [128, 128, 1]
        )
        self.assertIsNotNone(plans[1].lanes[1][0].capture_slot)
        continuation = make_request(2, prompt[:256] + (999,), 1)
        scheduler.submit(continuation)
        self.assertEqual(continuation.lane, 1)
        self.assertEqual(continuation.committed_tokens, 0)

        restored = scheduler.schedule()

        assert restored is not None
        row = restored.lanes[1][0]
        self.assertEqual(continuation.committed_tokens, 0)
        self.assertEqual(continuation.cached_tokens, 0)
        self.assertEqual((row.token_start, row.token_ids), (256, (999,)))
        self.assertIsNotNone(row.restore_slot)
        self.assertIsNone(row.capture_slot)
        self.assertEqual(len(row.table_delta.physical_blocks), 3)
        scheduler.commit(step_result(restored))
        self.assertEqual(continuation.committed_tokens, 257)
        self.assertEqual(continuation.cached_tokens, 256)

    def test_terminal_prefix_suffix_pipelines_one_decode_without_second_prefill(
        self,
    ) -> None:
        scheduler = make_scheduler(
            history_blocks=16,
            live_slots=2,
            snapshot_slots=2,
            token_budget=256,
            prefill_chunk_size=256,
            max_batch_size=1,
            max_prefill_rows=1,
            max_decode_ticks_between_prefills=4,
            graph_buckets=(1,),
        )
        prefix = tuple(range(256))
        scheduler.submit(make_request(1, prefix + (256,), 1))
        while (plan := scheduler.schedule()) is not None:
            scheduler.commit(step_result(plan))

        hit = make_request(2, prefix + tuple(range(1_000, 1_256)), 3)
        scheduler.submit(hit)

        restored = scheduler.schedule()
        assert restored is not None
        restored_row = restored.lanes[0][0]
        self.assertEqual(
            (restored_row.token_start, len(restored_row.token_ids)), (256, 128)
        )
        self.assertIsNotNone(restored_row.restore_slot)
        self.assertIsNone(scheduler.schedule())
        scheduler.commit(step_result(restored))

        waiting_prefill = make_request(3, (2_000,), 1)
        scheduler.submit(waiting_prefill)

        suffix = scheduler.schedule()
        assert suffix is not None
        suffix_row = suffix.lanes[0][0]
        self.assertEqual(
            (suffix_row.token_start, len(suffix_row.token_ids)), (384, 128)
        )
        self.assertIsNone(suffix_row.restore_slot)
        self.assertTrue(suffix_row.sample)

        successor = scheduler.schedule()

        assert successor is not None
        self.assertEqual(successor.graph_bucket, 1)
        self.assertEqual(len(successor.lanes[0]), 1)
        successor_row = successor.lanes[0][0]
        self.assertEqual(successor_row.slot, suffix_row.slot)
        self.assertEqual(
            (successor_row.token_start, successor_row.token_ids), (512, ())
        )
        self.assertIsNone(waiting_prefill.slot)
        with self.assertRaisesRegex(RuntimeError, "two plans"):
            scheduler.schedule()

        scheduler.commit(step_result(suffix))
        scheduler.commit(step_result(successor))
        self.assertEqual(hit.cached_tokens, 256)
        self.assertEqual(hit.committed_tokens, 513)
        self.assertEqual(len(hit.output_token_ids), 2)
        self.assertEqual(waiting_prefill.committed_tokens, 0)

    def test_admission_reserves_only_uncached_history_for_prefix_hits(self) -> None:
        scheduler = make_scheduler(
            history_blocks=5,
            live_slots=2,
            snapshot_slots=2,
            token_budget=128,
            prefill_chunk_size=128,
            max_batch_size=2,
            graph_buckets=(1, 2),
        )
        prompt = tuple(range(257))
        scheduler.submit(make_request(1, prompt, 1))
        while (plan := scheduler.schedule()) is not None:
            scheduler.commit(step_result(plan))

        first = make_request(2, prompt[:256] + (998,), 1)
        second = make_request(3, prompt[:256] + (999,), 1)
        scheduler.submit(first)
        scheduler.submit(second)
        restored = scheduler.schedule()

        assert restored is not None
        self.assertEqual(len(restored.lanes[0]), 2)
        self.assertEqual((first.cached_tokens, second.cached_tokens), (0, 0))
        scheduler.commit(step_result(restored))
        self.assertEqual((first.cached_tokens, second.cached_tokens), (256, 256))

    def test_prefix_restore_adjusts_and_retires_lane_token_charge(self) -> None:
        scheduler = make_scheduler(
            lane_count=2,
            history_blocks=16,
            live_slots=4,
            snapshot_slots=2,
            token_budget=128,
            prefill_chunk_size=128,
            max_batch_size=4,
            graph_buckets=(1, 2, 4),
        )
        prompt = tuple(range(257))
        scheduler.submit(make_request(1, prompt, 1, lane=0))
        while (plan := scheduler.schedule()) is not None:
            scheduler.commit(step_result(plan))

        hit = make_request(2, prompt[:256] + (999,), 4)
        scheduler.submit(hit)
        restored = scheduler.schedule()
        assert restored is not None
        self.assertEqual(hit.cached_tokens, 0)
        scheduler.commit(step_result(restored))
        self.assertEqual(hit.cached_tokens, 256)

        scheduler.submit(make_request(3, tuple(range(1_000, 1_050)), 4, lane=1))
        first = make_request(4, tuple(range(2_000, 2_096)), 4)
        scheduler.submit(first)
        self.assertEqual(first.lane, 0)

        scheduler.cancel(hit.request_id)
        second = make_request(5, (3_000,), 1)
        scheduler.submit(second)
        self.assertEqual(second.lane, 1)

    def test_divergent_suffix_promotes_the_shared_prefix(self) -> None:
        scheduler = make_scheduler(
            history_blocks=8,
            live_slots=2,
            snapshot_slots=3,
            token_budget=256,
            prefill_chunk_size=256,
            max_batch_size=2,
            graph_buckets=(1, 2),
        )
        shared = tuple(range(128))
        first = make_request(1, shared + tuple(range(1_000, 1_129)), 1)
        scheduler.submit(first)
        while (plan := scheduler.schedule()) is not None:
            scheduler.commit(step_result(plan))

        second = make_request(2, shared + tuple(range(2_000, 2_129)), 1)
        scheduler.submit(second)
        promoted = scheduler.schedule()
        assert promoted is not None
        self.assertEqual(len(promoted.lanes[0][0].token_ids), 128)
        self.assertIsNotNone(promoted.lanes[0][0].capture_slot)
        scheduler.commit(step_result(promoted))
        while (plan := scheduler.schedule()) is not None:
            scheduler.commit(step_result(plan))

        third = make_request(3, shared + tuple(range(3_000, 3_129)), 1)
        scheduler.submit(third)
        self.assertEqual(third.committed_tokens, 0)
        hit = scheduler.schedule()
        assert hit is not None
        self.assertEqual(third.committed_tokens, 0)
        self.assertEqual(third.cached_tokens, 0)
        self.assertEqual(hit.lanes[0][0].token_start, 128)
        self.assertIsNotNone(hit.lanes[0][0].restore_slot)
        scheduler.commit(step_result(hit))
        self.assertEqual(third.committed_tokens, 256)
        self.assertEqual(third.cached_tokens, 128)

    def test_partial_prefill_cannot_discard_a_capture_boundary(self) -> None:
        scheduler = make_scheduler(
            history_blocks=4,
            live_slots=1,
            snapshot_slots=1,
            token_budget=128,
            prefill_chunk_size=128,
            max_batch_size=1,
            graph_buckets=(1,),
        )
        scheduler.submit(make_request(1, tuple(range(129)), 1))
        capture = scheduler.schedule()
        assert capture is not None
        self.assertIsNotNone(capture.lanes[0][0].capture_slot)

        with self.assertRaisesRegex(
            ValueError, "prefill accepted 64 tokens, planned 128"
        ):
            scheduler.commit(step_result(capture, accepts={(0, 0): 64}))
        self.assertIsNone(scheduler.schedule())
        scheduler.commit(step_result(capture))

        tail = scheduler.schedule()
        assert tail is not None
        self.assertEqual(tail.lanes[0][0].token_ids, (128,))

    def test_cancelling_an_unscheduled_hit_releases_its_cache_pin(self) -> None:
        manager = StateManager(8, 2, 2, history_block_tokens=128)
        scheduler = make_scheduler(
            state_manager=manager,
            token_budget=128,
            prefill_chunk_size=128,
            max_batch_size=2,
            graph_buckets=(1, 2),
        )
        prompt = tuple(range(257))
        scheduler.submit(make_request(1, prompt, 1))
        while (plan := scheduler.schedule()) is not None:
            scheduler.commit(step_result(plan))

        hit = make_request(2, prompt[:256] + (999,), 1)
        scheduler.submit(hit)
        self.assertEqual(hit.committed_tokens, 0)
        scheduler.cancel(hit.request_id)

        self.assertEqual(manager.histories[0].available, 7)
        self.assertEqual(manager.reservable_history_blocks(0), 8)

    def test_carries_ignore_eos_from_request_to_each_plan_row(self) -> None:
        scheduler = make_scheduler(live_slots=1, max_batch_size=1, graph_buckets=(1,))
        scheduler.submit(make_request(1, (10,), 2, ignore_eos=True))

        prefill = scheduler.schedule()
        assert prefill is not None
        self.assertTrue(prefill.lanes[0][0].ignore_eos)
        scheduler.commit(step_result(prefill))

        decode = scheduler.schedule()
        assert decode is not None
        self.assertTrue(decode.lanes[0][0].ignore_eos)
        committed = scheduler.commit(step_result(decode))
        self.assertTrue(committed.lanes[0][0].finished)

    def test_prefill_rows_have_a_model_specific_cap(self) -> None:
        scheduler = make_scheduler(
            token_budget=4,
            max_batch_size=4,
            max_prefill_rows=1,
        )
        for request_id in range(3):
            scheduler.submit(make_request(request_id, (10 + request_id,), 1))

        plan = scheduler.schedule()

        assert plan is not None
        self.assertEqual(len(plan.lanes[0]), 1)
        self.assertTrue(plan.lanes[0][0].token_ids)

    def test_prefills_share_one_aggregate_chunk_budget(self) -> None:
        scheduler = make_scheduler(
            token_budget=6,
            prefill_chunk_size=4,
            max_batch_size=3,
        )
        first = make_request(1, (10, 11, 12), 1)
        second = make_request(2, (20, 21, 22, 23), 1)
        third = make_request(3, (30, 31, 32, 33), 1)
        for request in (first, second, third):
            scheduler.submit(request)

        plan = scheduler.schedule()

        assert plan is not None
        self.assertEqual([len(row.token_ids) for row in plan.lanes[0]], [3, 1])
        self.assertEqual([row.sample for row in plan.lanes[0]], [True, False])
        self.assertEqual(sum(len(row.token_ids) for row in plan.lanes[0]), 4)
        self.assertIsNone(plan.graph_bucket)
        self.assertTrue(all(row.table_delta.start_block == 0 for row in plan.lanes[0]))
        self.assertIsNone(scheduler.schedule())

        scheduler.commit(step_result(plan))
        self.assertEqual(first.output_token_ids, [1000])
        self.assertFalse(first.cancelled)
        self.assertEqual(second.output_token_ids, [])

    def test_dep_lanes_have_independent_token_budgets(self) -> None:
        scheduler = make_scheduler(
            lane_count=2,
            live_slots=3,
            token_budget=2,
            prefill_chunk_size=2,
            max_batch_size=2,
            max_decode_ticks_between_prefills=0,
            graph_buckets=(1, 2),
        )
        for request_id, lane in ((1, 0), (2, 1), (3, 0), (4, 1)):
            scheduler.submit(make_request(request_id, (request_id,), 2, lane=lane))

        initial = scheduler.schedule()
        assert initial is not None
        self.assertEqual([len(lane) for lane in initial.lanes], [2, 2])
        scheduler.commit(step_result(initial))
        scheduler.submit(make_request(5, (50, 51), 1, lane=0))

        mixed = scheduler.schedule()

        assert mixed is not None
        self.assertEqual(
            [[len(row.token_ids) for row in lane] for lane in mixed.lanes],
            [[2], [0, 0]],
        )

    def test_dep_lanes_balance_and_decode_uses_the_smallest_graph_bucket(self) -> None:
        scheduler = make_scheduler(
            lane_count=2,
            live_slots=5,
            token_budget=4,
            prefill_chunk_size=4,
            max_batch_size=4,
        )
        requests = [
            make_request(1, (10,), 4, lane=0),
            make_request(2, (20,), 4, lane=0),
            make_request(3, (30,), 4, lane=0),
            make_request(4, (40,), 4, lane=1),
        ]
        for request in requests:
            scheduler.submit(request)
        prefill = scheduler.schedule()
        assert prefill is not None
        self.assertEqual([len(lane) for lane in prefill.lanes], [3, 1])
        self.assertEqual(prefill.lanes[0][0].slot, prefill.lanes[1][0].slot)
        scheduler.commit(step_result(prefill))

        decode = scheduler.schedule()

        assert decode is not None
        self.assertEqual([len(lane) for lane in decode.lanes], [3, 1])
        self.assertEqual(decode.graph_bucket, 4)
        self.assertTrue(all(not row.token_ids for lane in decode.lanes for row in lane))
        self.assertTrue(all(row.sample for lane in decode.lanes for row in lane))
        self.assertTrue(
            all(
                not row.table_delta.physical_blocks
                for lane in decode.lanes
                for row in lane
            )
        )
        self.assertEqual((requests[0].temperature, requests[0].top_p), (0.7, 0.9))

    def test_decode_precedes_prefill_in_a_mixed_eager_step(self) -> None:
        scheduler = make_scheduler(
            token_budget=3,
            prefill_chunk_size=2,
            max_batch_size=2,
            max_decode_ticks_between_prefills=0,
        )
        decoding = make_request(1, (10,), 4)
        scheduler.submit(decoding)
        initial = scheduler.schedule()
        assert initial is not None
        scheduler.commit(step_result(initial))
        scheduler.submit(make_request(2, (20, 21), 1))

        mixed = scheduler.schedule()

        assert mixed is not None
        rows = mixed.lanes[0]
        self.assertEqual([row.slot for row in rows], [decoding.slot, 1])
        self.assertEqual([row.token_ids for row in rows], [(), (20, 21)])
        self.assertIsNone(mixed.graph_bucket)

    def test_prefill_cadence_reserves_a_complete_chunk(self) -> None:
        scheduler = make_scheduler(
            live_slots=2,
            token_budget=4,
            prefill_chunk_size=4,
            decode_width=1,
            max_batch_size=2,
            max_decode_ticks_between_prefills=0,
            graph_buckets=(1, 2),
        )
        scheduler.submit(make_request(1, (10, 11, 12, 13), 4))
        scheduler.submit(make_request(2, (20, 21, 22, 23), 1))
        first = scheduler.schedule()
        assert first is not None
        scheduler.commit(step_result(first))

        second = scheduler.schedule()

        assert second is not None
        self.assertEqual(
            [row.token_ids for row in second.lanes[0]],
            [(20, 21, 22, 23)],
        )
        self.assertTrue(second.lanes[0][0].sample)

    def test_single_resident_prefill_chunk_is_per_lane(self) -> None:
        scheduler = make_scheduler(
            lane_count=4,
            live_slots=2,
            token_budget=8,
            prefill_chunk_size=8,
            single_resident_prefill_chunk_size=2,
            max_batch_size=2,
            graph_buckets=(1, 2),
        )
        for request_id in range(4):
            scheduler.submit(make_request(request_id, tuple(range(16)), 1))

        single = scheduler.schedule()

        assert single is not None
        self.assertEqual(
            [[len(row.token_ids) for row in lane] for lane in single.lanes],
            [[2], [2], [2], [2]],
        )
        scheduler.commit(step_result(single))
        scheduler.submit(make_request(4, tuple(range(16)), 1))

        multiple = scheduler.schedule()

        assert multiple is not None
        self.assertEqual(
            [[len(row.token_ids) for row in lane] for lane in multiple.lanes],
            [[8], [8], [8], [8]],
        )

    def test_prefill_cadence_reserves_aggregate_chunk_rows_before_decode(
        self,
    ) -> None:
        scheduler = make_scheduler(
            history_blocks=8,
            live_slots=5,
            token_budget=8,
            prefill_chunk_size=4,
            max_batch_size=4,
            max_decode_ticks_between_prefills=0,
        )
        for request_id in range(1, 4):
            scheduler.submit(make_request(request_id, (request_id,), 4))
        initial = scheduler.schedule()
        assert initial is not None
        scheduler.commit(step_result(initial))
        scheduler.submit(make_request(4, (40,), 1))
        scheduler.submit(make_request(5, (50, 51, 52, 53), 1))

        mixed = scheduler.schedule()

        assert mixed is not None
        self.assertEqual(
            [row.token_ids for row in mixed.lanes[0]],
            [(), (), (40,), (50, 51, 52)],
        )
        self.assertIsNone(mixed.graph_bucket)

    def test_prefill_cadence_drains_decode_overlap_before_eager_work(self) -> None:
        scheduler = make_scheduler(
            live_slots=2,
            token_budget=1,
            prefill_chunk_size=1,
            max_batch_size=1,
            max_decode_ticks_between_prefills=1,
            graph_buckets=(1,),
        )
        scheduler.submit(make_request(1, (10,), 4))
        initial = scheduler.schedule()
        assert initial is not None
        scheduler.commit(step_result(initial))
        scheduler.submit(make_request(2, (20, 21), 1))

        decode = scheduler.schedule()
        assert decode is not None
        self.assertEqual(decode.graph_bucket, 1)
        self.assertIsNone(scheduler.schedule())
        scheduler.commit(step_result(decode))

        prefill = scheduler.schedule()

        assert prefill is not None
        self.assertEqual(prefill.lanes[0][0].token_ids, (20,))
        self.assertIsNone(prefill.graph_bucket)

    def test_prefill_cadence_scales_with_decode_rows_per_lane(self) -> None:
        scheduler = make_scheduler(
            lane_count=2,
            live_slots=3,
            token_budget=2,
            prefill_chunk_size=2,
            max_batch_size=2,
            max_decode_ticks_between_prefills=6,
            graph_buckets=(1, 2),
        )
        for request_id, lane in ((1, 0), (2, 0), (3, 1), (4, 1)):
            scheduler.submit(make_request(request_id, (request_id,), 8, lane=lane))
        initial = scheduler.schedule()
        assert initial is not None
        scheduler.commit(step_result(initial))
        scheduler.submit(make_request(5, (50, 51), 1, lane=0))

        for _ in range(6):
            decode = scheduler.schedule()
            assert decode is not None
            self.assertIsNotNone(decode.graph_bucket)
            scheduler.commit(step_result(decode))

        mixed = scheduler.schedule()

        assert mixed is not None
        self.assertEqual(mixed.lanes[0][-1].token_ids, (50, 51))
        self.assertIsNone(mixed.graph_bucket)

    def test_prefill_does_not_pipeline_behind_a_final_decode(self) -> None:
        scheduler = make_scheduler(
            live_slots=2,
            token_budget=1,
            prefill_chunk_size=1,
            max_batch_size=1,
            graph_buckets=(1,),
        )
        scheduler.submit(make_request(1, (10,), 2))
        initial = scheduler.schedule()
        assert initial is not None
        scheduler.commit(step_result(initial))
        scheduler.submit(make_request(2, (20,), 1))

        final_decode = scheduler.schedule()

        assert final_decode is not None
        self.assertEqual(final_decode.graph_bucket, 1)
        self.assertIsNone(scheduler.schedule())
        scheduler.commit(step_result(final_decode))
        self.assertIsNotNone(scheduler.schedule())

    def test_same_request_decode_reservations_use_device_resident_progress(
        self,
    ) -> None:
        scheduler = make_scheduler(
            history_blocks=2,
            live_slots=1,
            token_budget=2,
            prefill_chunk_size=2,
            decode_width=2,
            max_batch_size=1,
            graph_buckets=(1,),
        )
        request = make_request(1, (10,), 10)
        scheduler.submit(request)
        prefill = scheduler.schedule()
        assert prefill is not None
        scheduler.commit(step_result(prefill))

        first = scheduler.schedule()
        second = scheduler.schedule()

        assert first is not None
        assert second is not None
        self.assertEqual((first.step_id, first.staging_index), (1, 1))
        self.assertEqual((second.step_id, second.staging_index), (2, 0))
        self.assertEqual(first.lanes[0][0].token_ids, ())
        self.assertEqual(second.lanes[0][0].token_ids, ())
        self.assertEqual(first.lanes[0][0].token_start, 1)
        self.assertEqual(second.lanes[0][0].token_start, 3)
        self.assertEqual(first.lanes[0][0].slot, second.lanes[0][0].slot)
        with self.assertRaisesRegex(RuntimeError, "two plans"):
            scheduler.schedule()
        with self.assertRaisesRegex(ValueError, "inflight step 1"):
            scheduler.commit(step_result(second))

        scheduler.commit(step_result(first))
        third = scheduler.schedule()
        assert third is not None
        self.assertEqual(third.lanes[0][0].token_start, 4)
        scheduler.commit(step_result(second))
        scheduler.commit(step_result(third))
        self.assertEqual(request.committed_tokens, 4)
        self.assertEqual(len(request.output_token_ids), 4)

    def test_tail_decode_reserves_the_full_physical_window(self) -> None:
        bounded = make_scheduler(
            history_blocks=1,
            live_slots=1,
            token_budget=128,
            prefill_chunk_size=128,
            decode_width=4,
            max_batch_size=1,
            graph_buckets=(1,),
        )
        bounded.submit(make_request(1, tuple(range(126)), 3))
        bounded_prefill = bounded.schedule()
        assert bounded_prefill is not None
        bounded.commit(step_result(bounded_prefill))
        bounded_decode = bounded.schedule()
        assert bounded_decode is not None
        self.assertEqual(bounded_decode.lanes[0][0].table_delta.physical_blocks, ())

        scheduler = make_scheduler(
            history_blocks=2,
            live_slots=2,
            token_budget=128,
            prefill_chunk_size=128,
            decode_width=4,
            max_batch_size=1,
            graph_buckets=(1,),
        )
        request = make_request(1, tuple(range(126)), 3)
        scheduler.submit(request)
        waiting = make_request(2, (10,), 1)
        scheduler.submit(waiting)
        self.assertIsNone(waiting.lane)
        prefill = scheduler.schedule()
        assert prefill is not None
        scheduler.commit(step_result(prefill))

        decode = scheduler.schedule()

        assert decode is not None
        self.assertEqual(decode.lanes[0][0].max_accept_tokens, 2)
        self.assertEqual(decode.lanes[0][0].table_delta.physical_blocks, (1,))
        self.assertIsNone(scheduler.schedule())
        scheduler.commit(step_result(decode, accepts={(0, 0): 2}))
        self.assertEqual(request.committed_tokens, 128)
        self.assertEqual(len(request.output_token_ids), 3)

    def test_tail_decode_keeps_the_full_compute_budget(self) -> None:
        scheduler = make_scheduler(
            live_slots=2,
            token_budget=4,
            prefill_chunk_size=4,
            decode_width=4,
            max_batch_size=2,
            graph_buckets=(1, 2),
        )
        for request_id in (1, 2):
            scheduler.submit(make_request(request_id, (request_id,), 2))
        prefill = scheduler.schedule()
        assert prefill is not None
        scheduler.commit(step_result(prefill))

        decode = scheduler.schedule()

        assert decode is not None
        self.assertEqual(len(decode.lanes[0]), 1)
        self.assertEqual(decode.lanes[0][0].max_accept_tokens, 1)

    def test_backpressure_and_cancelled_inflight_work_release_capacity(self) -> None:
        scheduler = make_scheduler(
            history_blocks=1,
            live_slots=1,
            token_budget=1,
            prefill_chunk_size=1,
            max_batch_size=1,
            graph_buckets=(1,),
            max_queued_requests=1,
        )
        active = make_request(1, (10,), 2)
        waiting = make_request(2, (20,), 1)
        scheduler.submit(active)
        scheduler.submit(waiting)
        with self.assertRaisesRegex(RuntimeError, "queue is full"):
            scheduler.submit(make_request(3, (30,), 1))
        plan = scheduler.schedule()
        assert plan is not None
        scheduler.cancel(active.request_id)

        committed = scheduler.commit(step_result(plan))

        self.assertEqual(committed.lanes[0][0].output_token_ids, ())
        self.assertTrue(committed.lanes[0][0].finished)
        self.assertEqual(active.output_token_ids, [])
        replacement = scheduler.schedule()
        assert replacement is not None
        self.assertEqual(replacement.lanes[0][0].slot, 0)
        self.assertEqual(replacement.lanes[0][0].table_delta.physical_blocks, (0,))
        self.assertEqual(waiting.lane, 0)

    def test_a_blocked_affinity_lane_does_not_idle_another_lane(self) -> None:
        scheduler = make_scheduler(
            lane_count=2,
            history_blocks=1,
            live_slots=1,
            token_budget=2,
            prefill_chunk_size=1,
            max_batch_size=1,
            graph_buckets=(1,),
            max_queued_requests=2,
        )
        first = make_request(1, (10,), 1, lane=0)
        blocked = make_request(2, (20,), 1, lane=0)
        other_lane = make_request(3, (30,), 1, lane=1)
        for request in (first, blocked, other_lane):
            scheduler.submit(request)

        plan = scheduler.schedule()

        assert plan is not None
        self.assertEqual([len(lane) for lane in plan.lanes], [1, 1])
        scheduler.commit(step_result(plan))
        next_plan = scheduler.schedule()
        assert next_plan is not None
        self.assertEqual([len(lane) for lane in next_plan.lanes], [1, 0])
        self.assertEqual(blocked.slot, 0)

    def test_result_validation_is_lane_local_atomic_and_retryable(self) -> None:
        scheduler = make_scheduler(
            lane_count=2,
            live_slots=1,
            token_budget=2,
            prefill_chunk_size=1,
            max_batch_size=1,
            graph_buckets=(1,),
        )
        scheduler.submit(make_request(1, (10,), 1, lane=0))
        scheduler.submit(make_request(2, (20,), 1, lane=1))
        plan = scheduler.schedule()
        assert plan is not None
        valid = step_result(plan)
        wrong_lane = StepResult(
            plan.step_id,
            (
                valid.lanes[0],
                (StepResultRow(7, 1, (1100,), False),),
            ),
            elapsed_ns=10,
        )

        with self.assertRaisesRegex(ValueError, "plan slots"):
            scheduler.commit(wrong_lane)
        self.assertIsNone(scheduler.schedule())
        committed = scheduler.commit(valid)

        self.assertEqual([len(lane) for lane in committed.lanes], [1, 1])
        self.assertTrue(all(row.finished for lane in committed.lanes for row in lane))

    def test_constructor_and_request_contracts_are_strict(self) -> None:
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            make_scheduler(graph_buckets=(1, 1))
        with self.assertRaisesRegex(ValueError, "cover max_batch_size"):
            make_scheduler(max_batch_size=3, graph_buckets=(1, 2))
        with self.assertRaisesRegex(ValueError, "max_prefill_rows"):
            make_scheduler(max_prefill_rows=5)
        with self.assertRaisesRegex(ValueError, "max_queued_requests"):
            make_scheduler(max_queued_requests=0)
        for chunk_size in (0, 5):
            with self.assertRaisesRegex(ValueError, "single_resident"):
                make_scheduler(single_resident_prefill_chunk_size=chunk_size)

        scheduler = make_scheduler()
        with self.assertRaisesRegex(TypeError, "requires a Request"):
            scheduler.submit(1)  # type: ignore[arg-type]
        dirty = make_request(1, (10,), 1)
        dirty.output_token_ids.append(20)
        with self.assertRaisesRegex(ValueError, "runtime state"):
            scheduler.submit(dirty)
        with self.assertRaisesRegex(ValueError, "requires 2 history blocks"):
            make_scheduler(history_blocks=1).submit(
                make_request(2, tuple(range(129)), 1)
            )
        for invalid in (
            make_request(3, (10,), 1, lane=True),
            make_request(3, (-1,), 1),
        ):
            with self.subTest(invalid=invalid):
                isolated = make_scheduler()
                with self.assertRaises(ValueError):
                    isolated.submit(invalid)
                isolated.submit(make_request(3, (10,), 1))


if __name__ == "__main__":
    unittest.main()

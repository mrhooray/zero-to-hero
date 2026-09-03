import unittest
from collections import deque
from unittest.mock import Mock

from infer.engine import DistributedEngine, Worker
from infer.protocol import (
    MAX_BATCH_PLAN_BYTES,
    BatchPlan,
    BatchPlanRow,
    Request,
    StepResult,
    StepResultRow,
    TableDelta,
    encode_batch_plan,
)
from infer.scheduler import Scheduler
from infer.state import StateManager


class FakePlanConnection:
    def __init__(self, incoming: tuple[bytes, ...] = ()) -> None:
        self.incoming = deque(incoming)
        self.sent: list[bytes] = []
        self.receive_limits: list[int | None] = []

    def send_bytes(
        self, buffer: object, offset: int = 0, size: int | None = None
    ) -> None:
        view = memoryview(buffer)
        stop = len(view) if size is None else offset + size
        self.sent.append(bytes(view[offset:stop]))

    def recv_bytes(self, maxlength: int | None = None) -> bytes:
        self.receive_limits.append(maxlength)
        payload = self.incoming.popleft()
        if maxlength is not None and len(payload) > maxlength:
            raise OSError("bad message length")
        return payload


def plan_payload(plan: BatchPlan | None) -> bytes:
    buffer = bytearray(MAX_BATCH_PLAN_BYTES)
    size = encode_batch_plan(plan, buffer)
    return bytes(buffer[:size])


def plan_connections(
    rank: int, incoming: tuple[BatchPlan | None, ...] = ()
) -> tuple[FakePlanConnection, ...]:
    if rank == 0:
        return tuple(FakePlanConnection() for _ in range(3))
    return (FakePlanConnection(tuple(plan_payload(plan) for plan in incoming)),)


def request(request_id: int = 7, max_output_tokens: int = 1) -> Request:
    return Request(request_id, (10,), max_output_tokens, 0.0, 1.0, 0)


def make_scheduler() -> Scheduler:
    return Scheduler(
        StateManager(
            history_block_count=1,
            live_slot_count=1,
            snapshot_slot_count=0,
            history_block_tokens=128,
        ),
        token_budget=1,
        prefill_chunk_size=1,
        decode_width=1,
        max_batch_size=1,
        max_prefill_rows=1,
        max_decode_ticks_between_prefills=0,
        graph_buckets=(1,),
        max_queued_requests=1,
    )


def make_plan(step_id: int = 0) -> BatchPlan:
    return BatchPlan(
        step_id,
        ((BatchPlanRow(0, 0, (10,), 1, True, TableDelta(0, (0,))),),),
        None,
    )


def make_result(step_id: int = 0, *, finished: bool = False) -> StepResult:
    return StepResult(
        step_id, ((StepResultRow(0, 1, (20,), finished),),), elapsed_ns=10
    )


def worker(result: object | None = None) -> Mock:
    instance = Mock(spec=Worker)
    instance.dispatch.side_effect = lambda plan: plan.step_id
    instance.collect.return_value = make_result() if result is None else result
    return instance


class DistributedEngineTest(unittest.TestCase):
    def test_request_control_forwards_the_same_object_only_on_rank_zero(self) -> None:
        scheduler = Mock(spec=Scheduler)
        engine = DistributedEngine(
            0,
            scheduler,
            worker(),
            plan_connections(0),
        )
        submitted = request()

        engine.submit(submitted)
        engine.cancel(7)

        scheduler.submit.assert_called_once_with(submitted)
        scheduler.cancel.assert_called_once_with(7)

        engine = DistributedEngine(
            1,
            None,
            worker(),
            plan_connections(1),
        )
        with self.assertRaisesRegex(RuntimeError, "submission.*rank 0"):
            engine.submit(submitted)
        with self.assertRaisesRegex(RuntimeError, "cancellation.*rank 0"):
            engine.cancel(7)

    def test_rank_zero_broadcasts_plan_and_commits_local_result(self) -> None:
        scheduler = make_scheduler()
        submitted = request()
        scheduler.submit(submitted)
        target = worker()
        connections = plan_connections(0)
        engine = DistributedEngine(0, scheduler, target, connections)

        committed = engine.tick()

        self.assertEqual(committed, make_result(finished=True))
        self.assertEqual(submitted.output_token_ids, [20])
        plan = target.dispatch.call_args.args[0]
        target.collect.assert_called_once_with(0)
        self.assertTrue(
            all(
                connection.sent == [plan_payload(plan), b""]
                for connection in connections
            )
        )
        self.assertIsNone(scheduler.schedule())

    def test_nonzero_rank_returns_local_lane_result_after_plan(self) -> None:
        plan = make_plan()
        observed = make_result()
        connections = plan_connections(2, (plan, None))
        target = worker(observed)
        engine = DistributedEngine(2, None, target, connections)

        result = engine.tick()

        self.assertIs(result, observed)
        target.dispatch.assert_called_once_with(plan)
        target.collect.assert_called_once_with(0)
        self.assertEqual(
            connections[0].receive_limits,
            [MAX_BATCH_PLAN_BYTES, MAX_BATCH_PLAN_BYTES],
        )

    def test_dispatches_successor_before_collecting_predecessor(self) -> None:
        scheduler = Mock(spec=Scheduler)
        scheduler.schedule.side_effect = (make_plan(0), make_plan(1), make_plan(2))
        scheduler.commit.side_effect = lambda result: result
        events = []
        target = worker()
        target.dispatch.side_effect = lambda plan: (
            events.append(("dispatch", plan.step_id)) or plan.step_id
        )
        target.collect.side_effect = lambda pending: (
            events.append(("collect", pending)) or make_result(pending)
        )
        engine = DistributedEngine(0, scheduler, target, plan_connections(0))

        self.assertEqual(engine.tick(), make_result(0))

        self.assertEqual(events, [("dispatch", 0), ("dispatch", 1), ("collect", 0)])
        self.assertTrue(engine.has_pending)

        self.assertEqual(engine.tick(), make_result(1))
        self.assertEqual(
            events,
            [
                ("dispatch", 0),
                ("dispatch", 1),
                ("collect", 0),
                ("dispatch", 2),
                ("collect", 1),
            ],
        )

    def test_dispatches_decode_behind_terminal_prefix_suffix(self) -> None:
        scheduler = Scheduler(
            StateManager(
                history_block_count=16,
                live_slot_count=2,
                snapshot_slot_count=2,
                history_block_tokens=128,
            ),
            token_budget=256,
            prefill_chunk_size=256,
            decode_width=1,
            max_batch_size=1,
            max_prefill_rows=1,
            max_decode_ticks_between_prefills=4,
            graph_buckets=(1,),
            max_queued_requests=2,
        )
        events = []
        dispatched = []
        target = Mock(spec=Worker)

        def dispatch(plan: BatchPlan) -> BatchPlan:
            events.append(("dispatch", plan.step_id))
            dispatched.append(plan)
            return plan

        def collect(plan: BatchPlan) -> StepResult:
            events.append(("collect", plan.step_id))
            lanes = []
            for rows in plan.lanes:
                results = []
                for row in rows:
                    accepted = len(row.token_ids) if row.token_ids else 1
                    output_count = accepted if not row.token_ids else int(row.sample)
                    results.append(
                        StepResultRow(
                            row.slot,
                            accepted,
                            tuple(range(20, 20 + output_count)),
                            False,
                        )
                    )
                lanes.append(tuple(results))
            return StepResult(plan.step_id, tuple(lanes), elapsed_ns=10)

        target.dispatch.side_effect = dispatch
        target.collect.side_effect = collect
        engine = DistributedEngine(0, scheduler, target, plan_connections(0))
        prefix = tuple(range(256))
        seed = Request(1, prefix + (256,), 1, 0.0, 1.0, 1)
        engine.submit(seed)
        while not seed.output_token_ids:
            self.assertIsNotNone(engine.tick())

        events.clear()
        dispatched.clear()
        hit = Request(
            2,
            prefix + tuple(range(1_000, 1_256)),
            3,
            0.0,
            1.0,
            2,
        )
        engine.submit(hit)

        self.assertIsNotNone(engine.tick())
        self.assertEqual(hit.cached_tokens, 256)
        self.assertEqual(hit.committed_tokens, 384)

        events.clear()
        dispatched.clear()
        waiting_prefill = Request(3, (2_000,), 1, 0.0, 1.0, 3)
        engine.submit(waiting_prefill)

        result = engine.tick()

        self.assertIsNotNone(result)
        self.assertEqual(len(dispatched), 2)
        self.assertEqual(
            events,
            [
                ("dispatch", dispatched[0].step_id),
                ("dispatch", dispatched[1].step_id),
                ("collect", dispatched[0].step_id),
            ],
        )
        self.assertTrue(dispatched[0].lanes[0][0].token_ids)
        self.assertEqual(dispatched[1].graph_bucket, 1)
        self.assertEqual(dispatched[1].lanes[0][0].token_ids, ())
        self.assertEqual(
            dispatched[1].lanes[0][0].slot,
            dispatched[0].lanes[0][0].slot,
        )
        self.assertIsNone(waiting_prefill.slot)
        self.assertTrue(engine.has_pending)

    def test_cancelled_commit_is_returned_by_rank_zero(self) -> None:
        scheduler = make_scheduler()
        submitted = request(max_output_tokens=2)
        scheduler.submit(submitted)
        target = worker()

        def cancel_after_plan(_pending: object) -> StepResult:
            scheduler.cancel(7)
            return make_result()

        target.collect.side_effect = cancel_after_plan
        engine = DistributedEngine(
            0,
            scheduler,
            target,
            plan_connections(0),
        )

        committed = engine.tick()

        self.assertEqual(
            committed,
            StepResult(0, ((StepResultRow(0, 1, (), True),),), elapsed_ns=10),
        )
        self.assertEqual(submitted.output_token_ids, [])

    def test_idle_plan_broadcast_returns_none_without_execution(self) -> None:
        for rank, scheduler, incoming in (
            (0, make_scheduler(), ()),
            (3, None, (None,)),
        ):
            with self.subTest(rank=rank):
                target = worker()
                connections = plan_connections(rank, incoming)
                engine = DistributedEngine(rank, scheduler, target, connections)

                self.assertIsNone(engine.tick())

                target.dispatch.assert_not_called()
                target.collect.assert_not_called()
                if rank == 0:
                    self.assertTrue(
                        all(connection.sent == [b""] for connection in connections)
                    )
                else:
                    self.assertEqual(
                        connections[0].receive_limits, [MAX_BATCH_PLAN_BYTES]
                    )

    def test_requires_tep4_and_a_scheduler_only_on_rank_zero(self) -> None:
        target = worker()
        cases = (
            (-1, None, (), r"rank must be in \[0, 4\)"),
            (0, None, plan_connections(0), "only on rank 0"),
            (1, make_scheduler(), plan_connections(1), "only on rank 0"),
            (0, make_scheduler(), plan_connections(0)[:2], "3 plan connections"),
            (2, None, (), "1 plan connections"),
        )
        for rank, scheduler, connections, error in cases:
            with self.subTest(error=error), self.assertRaisesRegex(ValueError, error):
                DistributedEngine(rank, scheduler, target, connections)

    def test_worker_and_commit_failures_are_fatal(self) -> None:
        plan = make_plan()
        result = make_result()
        scheduler = Mock(spec=Scheduler)
        scheduler.schedule.return_value = plan
        target = worker()
        target.dispatch.side_effect = RuntimeError("worker failed")
        engine = DistributedEngine(0, scheduler, target, plan_connections(0))
        with self.assertRaisesRegex(RuntimeError, "worker failed"):
            engine.tick()
        scheduler.commit.assert_not_called()

        scheduler = Mock(spec=Scheduler)
        scheduler.schedule.return_value = plan
        target = worker(object())
        engine = DistributedEngine(0, scheduler, target, plan_connections(0))
        with self.assertRaisesRegex(TypeError, "worker.*StepResult"):
            engine.tick()
        scheduler.commit.assert_not_called()

        scheduler = Mock(spec=Scheduler)
        scheduler.schedule.return_value = plan
        scheduler.commit.side_effect = RuntimeError("commit failed")
        target = worker(result)
        engine = DistributedEngine(0, scheduler, target, plan_connections(0))
        with self.assertRaisesRegex(RuntimeError, "commit failed"):
            engine.tick()

    def test_invalid_pipe_payload_is_fatal(self) -> None:
        connection = FakePlanConnection((b"invalid",))
        target = worker()
        with self.assertRaisesRegex(ValueError, "truncated batch plan"):
            DistributedEngine(1, None, target, (connection,)).tick()


if __name__ == "__main__":
    unittest.main()

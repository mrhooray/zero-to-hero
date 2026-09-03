import threading
import unittest
from unittest.mock import Mock, call

from infer.engine import DistributedEngine
from infer.models.glm53_flash.codec import EOS_TOKEN_IDS
from infer.models.glm53_flash.model import (
    SPARSE_MLA_MAX_CONTEXT_TOKENS,
    TOKENIZER_VOCAB_SIZE,
)
from infer.protocol import Request, StepResult, StepResultRow
from infer.service import Completion, Service, ServiceOverloadedError


def row(
    slot: int,
    *output_token_ids: int,
    finished: bool,
    accepted_count: int = 1,
) -> StepResultRow:
    return StepResultRow(slot, accepted_count, output_token_ids, finished)


def committed_step(
    requests: dict[tuple[int, int], Request],
    step_id: int,
    *lanes: tuple[StepResultRow, ...],
) -> StepResult:
    for lane, rows in enumerate(lanes):
        for result in rows:
            requests[(lane, result.slot)].output_token_ids.extend(
                result.output_token_ids
            )
    return StepResult(step_id, lanes, elapsed_ns=10)


def make_service(
    capacity: int = 4,
) -> tuple[Service, Mock, dict[tuple[int, int], Request]]:
    engine = Mock(spec=DistributedEngine)
    engine.has_pending = False
    requests = {}

    def submit(request: Request) -> None:
        request.lane = 0
        request.slot = request.request_id
        requests[(0, request.slot)] = request

    engine.submit.side_effect = submit
    return (
        Service(
            engine,
            capacity=capacity,
            vocab_size=TOKENIZER_VOCAB_SIZE,
            max_context_tokens=SPARSE_MLA_MAX_CONTEXT_TOKENS,
            eos_token_ids=EOS_TOKEN_IDS,
        ),
        engine,
        requests,
    )


def start_service(
    service: Service,
) -> tuple[threading.Thread, list[BaseException]]:
    errors: list[BaseException] = []

    def serve() -> None:
        try:
            service.serve()
        except BaseException as error:  # noqa: BLE001
            errors.append(error)

    thread = threading.Thread(target=serve)
    thread.start()
    return thread, errors


class ServiceTest(unittest.TestCase):
    def test_submits_protocol_requests_and_routes_lane_local_slots(self) -> None:
        service, engine, requests = make_service()
        eos = min(EOS_TOKEN_IDS)

        def assign_lane(request: Request) -> None:
            request.lane = request.request_id
            request.slot = 0
            request.cached_tokens = request.request_id + 3
            requests[(request.lane, 0)] = request

        engine.submit.side_effect = assign_lane
        engine.tick.side_effect = lambda: committed_step(
            requests,
            0,
            (row(0, eos, finished=True),),
            (row(0, eos, finished=True),),
        )
        first = service.submit((10,), 1)
        second = service.submit((11,), 1, ignore_eos=True)

        self.assertEqual(first.prompt_tokens, 1)

        thread, errors = start_service(service)

        self.assertEqual(
            first.completion.result(timeout=1),
            Completion((eos,), "eos", 3),
        )
        self.assertEqual(
            second.completion.result(timeout=1),
            Completion((eos,), "length", 4),
        )
        service.close()
        thread.join(timeout=1)
        self.assertEqual(errors, [])
        submitted = [entry.args[0] for entry in engine.submit.call_args_list]
        self.assertTrue(all(type(request) is Request for request in submitted))
        self.assertEqual(
            [
                (
                    request.request_id,
                    request.prompt_token_ids,
                    request.max_output_tokens,
                    request.temperature,
                    request.top_p,
                    request.seed,
                    request.ignore_eos,
                )
                for request in submitted
            ],
            [
                (0, (10,), 1, 0.0, 1.0, 0, False),
                (1, (11,), 1, 0.0, 1.0, 0, True),
            ],
        )

    def test_routes_partial_rows_until_each_request_finishes(self) -> None:
        service, engine, requests = make_service()
        first_chunks: list[tuple[int, ...]] = []
        second_chunks: list[tuple[int, ...]] = []

        def publish_first(tokens: tuple[int, ...]) -> bool:
            first_chunks.append(tokens)
            return True

        def publish_second(tokens: tuple[int, ...]) -> bool:
            second_chunks.append(tokens)
            return True

        first = service.submit((10,), 1, token_sink=publish_first)
        second = service.submit((11,), 2, token_sink=publish_second)
        ticks = iter(
            (
                lambda: committed_step(
                    requests,
                    0,
                    (row(1, 30, finished=False), row(0, 20, finished=True)),
                ),
                lambda: committed_step(requests, 1, (row(1, 31, finished=True),)),
            )
        )
        engine.tick.side_effect = lambda: next(ticks)()

        thread, errors = start_service(service)

        self.assertEqual(
            first.completion.result(timeout=1),
            Completion((20,), "length"),
        )
        self.assertEqual(
            second.completion.result(timeout=1),
            Completion((30, 31), "length"),
        )
        service.close()
        thread.join(timeout=1)
        self.assertEqual(errors, [])
        self.assertEqual(engine.tick.call_count, 2)
        self.assertEqual(first_chunks, [(20,)])
        self.assertEqual(second_chunks, [(30,), (31,)])

    def test_stream_backpressure_cancels_instead_of_dropping_output(self) -> None:
        service, engine, requests = make_service()
        submission = service.submit((10,), 1, token_sink=lambda _tokens: False)
        engine.tick.side_effect = lambda: committed_step(
            requests, 0, (row(0, 20, finished=True),)
        )

        thread, errors = start_service(service)

        self.assertEqual(
            submission.completion.result(timeout=1),
            Completion((20,), "cancelled"),
        )
        service.close()
        thread.join(timeout=1)
        self.assertEqual(errors, [])
        engine.cancel.assert_called_once_with(0)

    def test_reused_slot_clears_its_retired_marker(self) -> None:
        service, engine, requests = make_service()

        def assign_slot_zero(request: Request) -> None:
            request.lane = 0
            request.slot = 0
            requests[(0, 0)] = request

        engine.submit.side_effect = assign_slot_zero
        service.submit((10,), 1)
        first = service._requests[0]
        engine.submit(first.request)
        first.submitted = True
        service._complete(first, "cancelled")
        self.assertEqual(service._retired_slots, {(0, 0)})
        service._apply(committed_step(requests, 0, (row(0, finished=True),)))
        service._apply(committed_step(requests, 1, (row(0, finished=True),)))
        self.assertEqual(service._retired_slots, {(0, 0)})

        service.submit((11,), 2)
        second = service._requests[1]
        engine.submit(second.request)
        second.submitted = True
        service._apply(committed_step(requests, 2, (row(0, 21, finished=False),)))

        self.assertEqual(service._retired_slots, set())

    def test_drains_overlapped_work_after_last_request_finishes(self) -> None:
        service, engine, requests = make_service()
        submission = service.submit((10,), 1)
        eos = min(EOS_TOKEN_IDS)

        def finish() -> StepResult:
            engine.has_pending = True
            return committed_step(requests, 0, (row(0, eos, finished=True),))

        def drain() -> StepResult:
            engine.has_pending = False
            service.close()
            return committed_step(
                requests, 1, (row(0, finished=True, accepted_count=0),)
            )

        ticks = iter((finish, drain))
        engine.tick.side_effect = lambda: next(ticks)()

        service.serve()

        self.assertEqual(
            submission.completion.result(),
            Completion((eos,), "eos"),
        )
        self.assertEqual(engine.tick.call_count, 2)
        self.assertEqual(service._retired_slots, {(0, 0)})

    def test_cancellation_preserves_rows_committed_during_the_tick(self) -> None:
        service, engine, requests = make_service()
        cancelled = service.submit((10,), 3)
        remaining = service.submit((11,), 2)

        def first_tick() -> StepResult:
            cancelled.cancel()
            return committed_step(
                requests,
                0,
                (row(0, 20, finished=False), row(1, 30, finished=False)),
            )

        ticks = iter(
            (
                first_tick,
                lambda: committed_step(
                    requests,
                    1,
                    (
                        row(0, finished=True, accepted_count=0),
                        row(1, 31, finished=True),
                    ),
                ),
            )
        )
        engine.tick.side_effect = lambda: next(ticks)()
        thread, errors = start_service(service)

        self.assertEqual(
            cancelled.completion.result(timeout=1),
            Completion((20,), "cancelled"),
        )
        self.assertEqual(
            remaining.completion.result(timeout=1),
            Completion((30, 31), "length"),
        )
        service.close()
        thread.join(timeout=1)
        self.assertEqual(errors, [])
        engine.cancel.assert_called_once_with(0)

    def test_cancellation_before_the_first_tick_still_reaches_the_engine(self) -> None:
        service, engine, _ = make_service()
        submission = service.submit((10,), 3)
        submission.cancel()

        thread, errors = start_service(service)

        self.assertEqual(
            submission.completion.result(timeout=1),
            Completion((), "cancelled"),
        )
        service.close()
        thread.join(timeout=1)
        self.assertEqual(errors, [])
        submitted = engine.submit.call_args.args[0]
        self.assertEqual((submitted.request_id, submitted.prompt_token_ids), (0, (10,)))
        engine.cancel.assert_called_once_with(0)
        engine.tick.assert_not_called()

    def test_completed_row_wins_cancellation_during_its_tick(self) -> None:
        service, engine, requests = make_service()
        submission = service.submit((10,), 1)

        def tick() -> StepResult:
            submission.cancel()
            return committed_step(requests, 0, (row(0, 20, finished=True),))

        engine.tick.side_effect = tick
        thread, errors = start_service(service)

        self.assertEqual(
            submission.completion.result(timeout=1),
            Completion((20,), "length"),
        )
        service.close()
        thread.join(timeout=1)
        self.assertEqual(errors, [])
        engine.cancel.assert_not_called()

    def test_capacity_counts_all_pending_and_active_requests(self) -> None:
        service, engine, _ = make_service(capacity=2)
        first = service.submit((10,), 1)
        second = service.submit((11,), 1)

        with self.assertRaisesRegex(
            ServiceOverloadedError, "capacity is 2 requests"
        ):
            service.submit((12,), 1)

        service.close()
        service.serve()
        self.assertEqual(first.completion.result().finish_reason, "cancelled")
        self.assertEqual(second.completion.result().finish_reason, "cancelled")
        self.assertEqual(engine.submit.call_count, 2)
        self.assertEqual(engine.cancel.call_count, 2)
        engine.tick.assert_not_called()

    def test_close_cancels_active_requests_then_stops_the_loop(self) -> None:
        service, engine, requests = make_service()
        first = service.submit((10,), 2)
        second = service.submit((11,), 2)
        tick_started = threading.Event()
        finish_tick = threading.Event()

        def tick() -> StepResult:
            tick_started.set()
            self.assertTrue(finish_tick.wait(timeout=1))
            return committed_step(
                requests,
                0,
                (row(0, 20, finished=False), row(1, 21, finished=False)),
            )

        engine.tick.side_effect = tick
        thread, errors = start_service(service)
        self.assertTrue(tick_started.wait(timeout=1))

        service.close()
        finish_tick.set()
        thread.join(timeout=1)

        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(
            first.completion.result(timeout=0),
            Completion((20,), "cancelled"),
        )
        self.assertEqual(
            second.completion.result(timeout=0),
            Completion((21,), "cancelled"),
        )
        self.assertEqual(engine.cancel.call_args_list, [call(0), call(1)])

    def test_close_wakes_an_idle_service_and_rejects_new_work(self) -> None:
        service, engine, _ = make_service()
        thread, errors = start_service(service)

        service.close()
        service.close()
        thread.join(timeout=1)

        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        with self.assertRaisesRegex(RuntimeError, "service is closed"):
            service.submit((10,), 1)
        engine.submit.assert_not_called()
        engine.tick.assert_not_called()

    def test_fatal_engine_errors_fail_every_outstanding_request(self) -> None:
        for failure in ("submit", "tick"):
            with self.subTest(failure=failure):
                service, engine, _ = make_service()
                first = service.submit((10,), 1)
                second = service.submit((11,), 1)
                error = RuntimeError(f"{failure} failed")
                if failure == "submit":
                    engine.submit.side_effect = (None, error)
                else:
                    engine.tick.side_effect = error

                with self.assertRaises(RuntimeError) as raised:
                    service.serve()

                self.assertIs(raised.exception, error)
                for submission in (first, second):
                    with self.assertRaises(RuntimeError) as completed:
                        submission.completion.result(timeout=0)
                    self.assertIs(completed.exception, error)
                with self.assertRaisesRegex(RuntimeError, "service is closed"):
                    service.submit((12,), 1)

    def test_idle_tick_is_fatal(self) -> None:
        service, engine, _ = make_service()
        submission = service.submit((10,), 1)
        engine.tick.return_value = None

        with self.assertRaisesRegex(RuntimeError, "produced no distributed plan"):
            service.serve()

        self.assertTrue(submission.completion.done())
        engine.tick.assert_called_once_with()

    def test_enforces_request_limits_before_reserving_capacity(self) -> None:
        service, _, _ = make_service(capacity=1)
        token_range_error = rf"\[0, {TOKENIZER_VOCAB_SIZE}\)"
        cases = (
            ((), 1, "must not be empty"),
            ((10,), 0, "positive integer"),
            ((10,), True, "positive integer"),
            ((True,), 1, "only integers"),
            ((1.0,), 1, "only integers"),
            ((-1,), 1, token_range_error),
            ((TOKENIZER_VOCAB_SIZE,), 1, token_range_error),
            (
                (10,) * SPARSE_MLA_MAX_CONTEXT_TOKENS,
                2,
                f"model context of {SPARSE_MLA_MAX_CONTEXT_TOKENS} tokens",
            ),
        )
        for prompt_token_ids, max_output_tokens, error in cases:
            with self.subTest(error=error), self.assertRaisesRegex(ValueError, error):
                service.submit(prompt_token_ids, max_output_tokens)

        self.assertFalse(service.submit((10,), 1).completion.done())

    def test_capacity_must_be_a_positive_integer(self) -> None:
        engine = Mock(spec=DistributedEngine)
        for capacity in (0, -1, True, 1.0):
            with (
                self.subTest(capacity=capacity),
                self.assertRaisesRegex(
                    ValueError, "capacity must be a positive integer"
                ),
            ):
                Service(
                    engine,
                    capacity=capacity,
                    vocab_size=TOKENIZER_VOCAB_SIZE,
                    max_context_tokens=SPARSE_MLA_MAX_CONTEXT_TOKENS,
                    eos_token_ids=EOS_TOKEN_IDS,
                )


if __name__ == "__main__":
    unittest.main()

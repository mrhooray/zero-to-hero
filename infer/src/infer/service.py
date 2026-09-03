from collections import deque
from collections.abc import Callable, Sequence
from concurrent.futures import Future
from dataclasses import dataclass, field
from threading import Condition, Event
from typing import Literal, Protocol

from infer.protocol import Request, StepResult, StepResultRow

type FinishReason = Literal["eos", "length", "cancelled"]
type TokenSink = Callable[[tuple[int, ...]], bool]


class Engine(Protocol):
    @property
    def has_pending(self) -> bool: ...

    def submit(self, request: Request) -> None: ...

    def cancel(self, request_id: int) -> None: ...

    def tick(self) -> StepResult | None: ...


@dataclass(frozen=True, slots=True)
class Completion:
    output_token_ids: tuple[int, ...]
    finish_reason: FinishReason
    cached_tokens: int = 0


@dataclass(frozen=True, slots=True)
class Submission:
    completion: Future[Completion]
    prompt_tokens: int
    _cancelled: Event = field(repr=False)

    def cancel(self) -> None:
        self._cancelled.set()


@dataclass(slots=True)
class _ServiceRequest:
    request: Request
    submission: Submission
    token_sink: TokenSink | None
    submitted: bool = False


class ServiceOverloadedError(RuntimeError):
    pass


class Service:
    def __init__(
        self,
        engine: Engine,
        *,
        capacity: int,
        vocab_size: int,
        max_context_tokens: int,
        eos_token_ids: frozenset[int],
    ) -> None:
        if type(capacity) is not int or capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        if type(vocab_size) is not int or vocab_size <= 0:
            raise ValueError("vocab_size must be a positive integer")
        if type(max_context_tokens) is not int or max_context_tokens <= 0:
            raise ValueError("max_context_tokens must be a positive integer")
        if not eos_token_ids:
            raise ValueError("service requires at least one EOS token")

        self._engine = engine
        self._capacity = capacity
        self._vocab_size = vocab_size
        self._max_context_tokens = max_context_tokens
        self._eos_token_ids = eos_token_ids
        self._condition = Condition()
        self._requests: dict[int, _ServiceRequest] = {}
        self._pending: deque[_ServiceRequest] = deque()
        self._retired_slots: set[tuple[int, int]] = set()
        self._next_request_id = 0
        self._closed = False

    def close(self) -> None:
        with self._condition:
            if self._closed:
                return
            self._closed = True
            for request in self._requests.values():
                request.submission.cancel()
            self._condition.notify_all()

    def submit(
        self,
        prompt_token_ids: Sequence[int],
        max_output_tokens: int,
        *,
        ignore_eos: bool = False,
        token_sink: TokenSink | None = None,
    ) -> Submission:
        prompt_token_ids = tuple(prompt_token_ids)
        self._validate_request(prompt_token_ids, max_output_tokens, ignore_eos)

        with self._condition:
            if self._closed:
                raise RuntimeError("service is closed")
            if len(self._requests) == self._capacity:
                raise ServiceOverloadedError(
                    f"service capacity is {self._capacity} requests"
                )

            completion: Future[Completion] = Future()
            completion.set_running_or_notify_cancel()
            submission = Submission(completion, len(prompt_token_ids), Event())
            request = _ServiceRequest(
                request=Request(
                    self._next_request_id,
                    prompt_token_ids,
                    max_output_tokens,
                    temperature=0.0,
                    top_p=1.0,
                    seed=0,
                    ignore_eos=ignore_eos,
                ),
                submission=submission,
                token_sink=token_sink,
            )
            self._requests[request.request.request_id] = request
            self._pending.append(request)
            self._next_request_id += 1
            self._condition.notify()
            return submission

    def serve(self) -> None:
        try:
            while True:
                pending = self._await_work()
                if pending is None:
                    return
                for request in pending:
                    self._engine.submit(request.request)
                    request.submitted = True

                self._cancel_requested()
                with self._condition:
                    has_submitted = any(
                        request.submitted for request in self._requests.values()
                    )
                if not has_submitted and not self._engine.has_pending:
                    continue

                result = self._engine.tick()
                if result is None:
                    raise RuntimeError("active requests produced no distributed plan")
                self._apply(result)
                self._cancel_requested()
        except BaseException as error:
            self._fail(error)
            raise

    def _await_work(self) -> tuple[_ServiceRequest, ...] | None:
        with self._condition:
            while (
                not self._pending
                and not any(request.submitted for request in self._requests.values())
                and not self._engine.has_pending
            ):
                if self._closed:
                    return None
                self._condition.wait()
            pending = tuple(self._pending)
            self._pending.clear()
            return pending

    def _cancel_requested(self) -> None:
        with self._condition:
            cancelled = tuple(
                request
                for request in self._requests.values()
                if request.submitted and request.submission._cancelled.is_set()
            )
        for request in cancelled:
            self._engine.cancel(request.request.request_id)
            self._complete(request, "cancelled")

    def _apply(self, result: StepResult) -> None:
        with self._condition:
            resident = {
                (request.request.lane, request.request.slot): request
                for request in self._requests.values()
                if request.submitted and request.request.slot is not None
            }
            self._retired_slots.difference_update(resident)
            retired_slots = self._retired_slots.copy()
        for lane, rows in enumerate(result.lanes):
            for row in rows:
                request = resident.get((lane, row.slot))
                if request is None:
                    if (lane, row.slot) in retired_slots:
                        continue
                    raise RuntimeError(
                        f"distributed result contains unknown slot ({lane}, {row.slot})"
                    )
                self._finish_row(request, row)

    def _finish_row(self, request: _ServiceRequest, row: StepResultRow) -> None:
        if (
            row.output_token_ids
            and request.token_sink is not None
            and not request.token_sink(row.output_token_ids)
        ):
            self._engine.cancel(request.request.request_id)
            self._complete(request, "cancelled")
            return
        if not row.finished:
            return
        output_token_ids = request.request.output_token_ids
        finish_reason: Literal["eos", "length"] = (
            "eos"
            if (
                not request.request.ignore_eos
                and output_token_ids
                and output_token_ids[-1] in self._eos_token_ids
            )
            else "length"
        )
        self._complete(request, finish_reason)

    def _complete(self, request: _ServiceRequest, finish_reason: FinishReason) -> None:
        request_id = request.request.request_id
        with self._condition:
            if self._requests.pop(request_id, None) is not request:
                return
            if request.request.lane is not None and request.request.slot is not None:
                self._retired_slots.add((request.request.lane, request.request.slot))
            self._condition.notify_all()
        request.submission.completion.set_result(
            Completion(
                tuple(request.request.output_token_ids),
                finish_reason,
                request.request.cached_tokens,
            )
        )

    def _fail(self, error: BaseException) -> None:
        with self._condition:
            self._closed = True
            requests = tuple(self._requests.values())
            self._requests.clear()
            self._pending.clear()
            self._condition.notify_all()
        for request in requests:
            request.submission.completion.set_exception(error)

    def _validate_request(
        self,
        prompt_token_ids: tuple[int, ...],
        max_output_tokens: int,
        ignore_eos: bool,
    ) -> None:
        if type(max_output_tokens) is not int or max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be a positive integer")
        if type(ignore_eos) is not bool:
            raise ValueError("ignore_eos must be a boolean")
        if not prompt_token_ids:
            raise ValueError("prompt_token_ids must not be empty")
        for token_id in prompt_token_ids:
            if type(token_id) is not int:
                raise ValueError("prompt_token_ids must contain only integers")
            if not 0 <= token_id < self._vocab_size:
                raise ValueError(f"prompt token IDs must be in [0, {self._vocab_size})")
        if len(prompt_token_ids) + max_output_tokens - 1 > self._max_context_tokens:
            raise ValueError(
                "prompt plus decode steps exceeds the model context "
                f"of {self._max_context_tokens} tokens"
            )

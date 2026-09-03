from collections import deque
from typing import Protocol

from infer.protocol import (
    MAX_BATCH_PLAN_BYTES,
    BatchPlan,
    Request,
    StepResult,
    decode_batch_plan,
    encode_batch_plan,
)
from infer.scheduler import Scheduler

WORLD_SIZE = 4


class PlanConnection(Protocol):
    def send_bytes(
        self, buffer: object, offset: int = 0, size: int | None = None
    ) -> None: ...

    def recv_bytes(self, maxlength: int | None = None) -> bytes: ...


class Worker(Protocol):
    def dispatch(self, plan: BatchPlan) -> object: ...

    def collect(self, pending: object) -> StepResult: ...


class DistributedEngine:
    """Four-rank engine with rank-zero scheduling and ordered commit."""

    def __init__(
        self,
        rank: int,
        scheduler: Scheduler | None,
        worker: Worker,
        plan_connections: tuple[PlanConnection, ...],
    ) -> None:
        if not 0 <= rank < WORLD_SIZE:
            raise ValueError(f"rank must be in [0, {WORLD_SIZE}), got {rank}")
        if (rank == 0) != (scheduler is not None):
            raise ValueError("scheduler must be supplied only on rank 0")
        expected_connections = WORLD_SIZE - 1 if rank == 0 else 1
        if len(plan_connections) != expected_connections:
            raise ValueError(
                f"rank {rank} requires {expected_connections} plan connections"
            )

        self._scheduler = scheduler
        self._worker = worker
        self._plan_connections = plan_connections
        self._plan_buffer = bytearray(MAX_BATCH_PLAN_BYTES)
        self._rank = rank
        self._pending: deque[object] = deque()

    @property
    def has_pending(self) -> bool:
        return bool(self._pending)

    def submit(self, request: Request) -> None:
        if self._rank != 0:
            raise RuntimeError("request submission is only available on rank 0")
        assert self._scheduler is not None
        self._scheduler.submit(request)

    def cancel(self, request_id: int) -> None:
        if self._rank != 0:
            raise RuntimeError("request cancellation is only available on rank 0")
        assert self._scheduler is not None
        self._scheduler.cancel(request_id)

    def tick(self) -> StepResult | None:
        while len(self._pending) < 2:
            scheduled = self._scheduler.schedule() if self._rank == 0 else None
            plan = self._exchange_plan(scheduled)
            if plan is None:
                break
            self._pending.append(self._worker.dispatch(plan))

        if not self._pending:
            return None
        local_result = self._worker.collect(self._pending.popleft())
        if type(local_result) is not StepResult:
            raise TypeError("worker must return a StepResult")
        if self._rank != 0:
            return local_result

        assert self._scheduler is not None
        return self._scheduler.commit(local_result)

    def _exchange_plan(self, plan: BatchPlan | None) -> BatchPlan | None:
        if self._rank != 0:
            payload = self._plan_connections[0].recv_bytes(MAX_BATCH_PLAN_BYTES)
            return decode_batch_plan(payload)

        size = encode_batch_plan(plan, self._plan_buffer)
        for connection in self._plan_connections:
            connection.send_bytes(self._plan_buffer, 0, size)
        return plan

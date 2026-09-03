import argparse
import os
import signal
from collections.abc import Callable
from concurrent.futures import Future
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Thread

from infer.engine import WORLD_SIZE

_MAX_PORT = 65535
_MEMORY_RESERVE_DENOMINATOR = 10


def launch(
    entrypoint: Callable[..., None],
    checkpoint: Path,
    host: str,
    port: int,
    temporary_prefix: str,
    supervisor: Callable[[object], None],
) -> None:
    from torch.multiprocessing import get_context, spawn

    with TemporaryDirectory(prefix=temporary_prefix) as directory:
        rendezvous = (Path(directory) / "rendezvous").as_uri()
        pipe_context = get_context("spawn")
        pipes = tuple(pipe_context.Pipe(duplex=False) for _ in range(WORLD_SIZE - 1))
        try:
            context = spawn(
                entrypoint,
                args=(str(checkpoint), host, port, rendezvous, pipes),
                nprocs=WORLD_SIZE,
                join=False,
                daemon=False,
            )
        finally:
            for pipe in pipes:
                for connection in pipe:
                    connection.close()
        supervisor(context)


def supervise(context: object) -> None:
    received_signal = None

    def terminate_children(number: int, _frame: object) -> None:
        nonlocal received_signal
        received_signal = number
        for process in context.processes:
            if process.is_alive():
                process.terminate()

    previous = {
        number: signal.signal(number, terminate_children)
        for number in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        while not context.join():
            pass
    except BaseException:
        if received_signal is not None:
            raise SystemExit(128 + received_signal) from None
        raise
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)
    if received_signal is not None:
        raise SystemExit(128 + received_signal)


def take_plan_connections(
    rank: int, plan_pipes: tuple[tuple[object, object], ...]
) -> tuple[object, ...]:
    connections = []
    for worker_rank, (receiver, sender) in enumerate(plan_pipes, start=1):
        if rank == 0:
            receiver.close()
            connections.append(sender)
        else:
            sender.close()
            if rank == worker_rank:
                connections.append(receiver)
            else:
                receiver.close()
    return tuple(connections)


def wait_until_ready(dist: object, rank: int, model_name: str) -> None:
    ready: list[object | None] = [None] * WORLD_SIZE
    dist.all_gather_object(ready, rank, group=dist.group.WORLD)
    if ready != list(range(WORLD_SIZE)):
        raise RuntimeError(f"{model_name} serving ranks did not become ready: {ready}")


def drive_service(
    service: object,
    app: object,
    host: str,
    port: int,
    thread_name: str,
) -> None:
    import uvicorn

    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host=host,
            port=port,
            loop="uvloop",
            http="httptools",
            lifespan="off",
        )
    )
    outcome: Future[None] = Future()

    def serve_http() -> None:
        try:
            server.run()
        except BaseException as error:  # noqa: BLE001
            outcome.set_exception(error)
        else:
            outcome.set_result(None)
        finally:
            service.close()

    thread = Thread(target=serve_http, name=thread_name, daemon=True)
    thread.start()
    try:
        service.serve()
    finally:
        server.should_exit = True
        service.close()
        thread.join()
    outcome.result()


def drive_worker(engine: object) -> None:
    while engine.tick() is not None:
        pass


def bind_b200(torch: object, rank: int, model_name: str) -> object:
    if torch.cuda.device_count() != WORLD_SIZE:
        raise RuntimeError(
            f"{model_name} serving requires exactly {WORLD_SIZE} GPUs"
        )
    torch.cuda.set_device(rank)
    identity = (
        torch.cuda.get_device_name(rank),
        torch.cuda.get_device_capability(rank),
    )
    if identity != ("NVIDIA B200", (10, 0)):
        raise RuntimeError(
            f"{model_name} serving requires NVIDIA B200 SM100, found {identity}"
        )

    torch.set_grad_enabled(False)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    return torch.device("cuda", rank)


def require_memory_reserve(torch: object, device: object, rank: int) -> None:
    torch.cuda.empty_cache()
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    required_free_bytes = (
        total_bytes + _MEMORY_RESERVE_DENOMINATOR - 1
    ) // _MEMORY_RESERVE_DENOMINATOR
    if free_bytes < required_free_bytes:
        raise RuntimeError(
            f"rank={rank} has insufficient free HBM after warmup: "
            f"free={free_bytes} required={required_free_bytes} total={total_bytes}"
        )


def set_rank_env(rank: int) -> None:
    os.environ.update(
        DG_JIT_CACHE_DIR=f"/tmp/infer-deep-gemm-rank{rank}",
        TRITON_CACHE_DIR=f"/tmp/infer-triton-cache-rank{rank}",
        TRITON_HOME=f"/tmp/infer-triton-home-rank{rank}",
    )


def port(value: str) -> int:
    number = int(value)
    if not 1 <= number <= _MAX_PORT:
        raise argparse.ArgumentTypeError(f"port must be in [1, {_MAX_PORT}]")
    return number

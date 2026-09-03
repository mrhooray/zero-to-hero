from __future__ import annotations

from datetime import timedelta
from pathlib import Path

from infer.engine import DistributedEngine
from infer.http import create_app
from infer.models.glm53_flash import MODEL_ID
from infer.models.glm53_flash import model as glm53_flash
from infer.models.glm53_flash.codec import EOS_TOKEN_IDS, GLM53Codec
from infer.models.glm53_flash.nextn import (
    GLM53NextNModel,
    allocate_glm53_nextn_runtime,
    load_glm53_nextn_weights,
)
from infer.models.glm53_flash.target import (
    GLM53_TARGET_HISTORY_BLOCKS,
    GLM53_TARGET_LIVE_SLOTS,
    GLM53TargetModel,
    allocate_glm53_target_runtime,
    load_glm53_target_weights,
)
from infer.models.glm53_flash.worker import GLM53Worker, allocate_glm53_worker_staging
from infer.protocol import BatchPlan, BatchPlanRow, TableDelta
from infer.runtime import (
    bind_b200,
    drive_service,
    drive_worker,
    require_memory_reserve,
    set_rank_env,
    take_plan_connections,
    wait_until_ready,
)
from infer.scheduler import Scheduler
from infer.service import Service
from infer.state import StateManager

_DEP4_NATIVE_LIVE_SLOTS = 16
_DEP4_TARGET_ONLY_LIVE_SLOTS = 32
_DEP4_SINGLE_RESIDENT_PREFILL_CHUNK_SIZE = 2_048


def serve_rank(
    rank: int,
    checkpoint: str,
    host: str,
    port: int,
    rendezvous: str,
    plan_pipes: tuple[tuple[object, object], ...],
    *,
    parallelism: str,
    speculation: str = "native",
) -> None:
    plan_connections = take_plan_connections(rank, plan_pipes)
    set_rank_env(rank)
    import torch
    import torch.distributed as dist

    device = _bind_b200(torch, rank)
    dist.init_process_group(
        "nccl",
        init_method=rendezvous,
        rank=rank,
        world_size=glm53_flash.TP_SIZE,
        timeout=timedelta(hours=2),
        device_id=device,
    )
    all_reduce = _create_all_reduce(rank, device, dist)
    try:
        engine = _build_engine(
            rank,
            Path(checkpoint),
            device,
            dist,
            all_reduce,
            plan_connections,
            parallelism,
            speculation,
        )
        codec = GLM53Codec(checkpoint) if rank == 0 else None

        wait_until_ready(dist, rank, "GLM")

        if rank == 0:
            assert codec is not None
            _drive_rank_zero(engine, codec, host, port, parallelism, speculation)
        else:
            drive_worker(engine)
        dist.barrier()
    finally:
        all_reduce.destroy()
        dist.destroy_process_group()
        for connection in plan_connections:
            connection.close()


def _create_all_reduce(rank: int, device: object, dist: object):
    from flashinfer import comm

    from infer.models.glm53_flash.allreduce import GLM53AllReduce
    from infer.models.glm53_flash.ops.core import _sparse_mla_runtime

    _sparse_mla_runtime()
    return GLM53AllReduce(comm, rank, dist.group.WORLD, device)


def _build_engine(
    rank: int,
    checkpoint: Path,
    device: object,
    dist: object,
    all_reduce: object,
    plan_connections: tuple[object, ...],
    parallelism: str,
    speculation: str = "native",
) -> DistributedEngine:
    import torch

    if speculation not in {"native", "none"}:
        raise ValueError("GLM speculation must be native or none")
    speculative = speculation == "native"
    attention_tp_size = glm53_flash.glm53_attention_tp_size(parallelism)
    lane_count = glm53_flash.TP_SIZE if parallelism == "dep4" else 1
    live_slot_count = _live_slot_count(parallelism, speculation)
    snapshot_slot_count = 2 * live_slot_count
    decode_batch_sizes = glm53_flash.glm53_decode_batch_sizes(
        attention_tp_size, speculative=speculative
    )
    from infer.models.glm53_flash.ops.core import GLM53Ops

    ops = GLM53Ops(attention_tp_size, dist.group.WORLD)
    model = GLM53TargetModel(
        load_glm53_target_weights(checkpoint, rank, device, attention_tp_size),
        ops,
    )
    runtime = allocate_glm53_target_runtime(
        0,
        device,
        live_slot_count=live_slot_count,
        snapshot_slot_count=snapshot_slot_count,
        attention_tp_size=attention_tp_size,
        speculative=speculative,
    )
    nextn_model = None
    nextn_runtime = None
    if speculative:
        nextn_model = GLM53NextNModel(
            load_glm53_nextn_weights(checkpoint, rank, device, attention_tp_size),
            ops,
        )
        nextn_runtime = allocate_glm53_nextn_runtime(runtime)
    staging = allocate_glm53_worker_staging(
        device, live_slot_count, lane_count, speculative=speculative
    )
    completion_events = (torch.cuda.Event(), torch.cuda.Event())
    worker = GLM53Worker(
        model,
        runtime,
        nextn_model,
        nextn_runtime,
        dist.group.WORLD,
        all_reduce,
        staging,
        completion_events,
        rank=rank,
        collectives=dist,
        parallelism=parallelism,
    )
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    decode_width = glm53_flash.GLM53_TARGET_VERIFY_WIDTH if speculative else 1
    decode_plans = []
    with torch.cuda.stream(stream):
        for bucket in decode_batch_sizes:
            rows = tuple(
                BatchPlanRow(
                    slot,
                    1,
                    (),
                    decode_width,
                    True,
                    TableDelta(0, ()),
                )
                for slot in range(bucket)
            )
            for staging_index in range(2):
                batch = runtime.stage_decode(tuple(range(bucket)), staging_index)
                runtime.prepare_decode(batch, runtime.decode_staging[staging_index])
                if nextn_runtime is not None:
                    assert nextn_model is not None
                    nextn_runtime.seed_candidates(
                        nextn_model,
                        batch.token_ids,
                        batch.state_indices_int64,
                        bucket,
                        staging_index,
                        model.weights.endpoint,
                        dist.group.WORLD,
                        all_reduce.decode,
                    )
                plan = BatchPlan(
                    len(decode_plans),
                    (rows,) * lane_count,
                    bucket,
                )
                worker.dispatch(plan)
                decode_plans.append(plan)
        worker.dispatch(
            BatchPlan(
                len(decode_plans),
                (
                    (
                        BatchPlanRow(
                            0,
                            0,
                            (0,) * glm53_flash.KDA_CHUNK_SIZE,
                            glm53_flash.KDA_CHUNK_SIZE,
                            False,
                            TableDelta(
                                0,
                                tuple(
                                    range(
                                        glm53_flash.KDA_CHUNK_SIZE
                                        // glm53_flash.SPARSE_MLA_HISTORY_BLOCK_TOKENS
                                    )
                                ),
                            ),
                        ),
                    ),
                )
                * lane_count,
                None,
            )
        )
    stream.synchronize()
    dist.barrier(group=dist.group.WORLD)
    worker.capture_decode_graphs(decode_plans, stream, torch)
    with torch.cuda.stream(stream):
        for slot in range(max(decode_batch_sizes)):
            runtime.reset_state(slot)
            if nextn_runtime is not None:
                nextn_runtime.reset_slot(slot)
    stream.synchronize()
    dist.barrier(group=dist.group.WORLD)
    require_memory_reserve(torch, device, rank)
    scheduler = None
    if rank == 0:
        scheduler = Scheduler(
            StateManager(
                history_block_count=GLM53_TARGET_HISTORY_BLOCKS,
                live_slot_count=live_slot_count,
                snapshot_slot_count=snapshot_slot_count,
                lane_count=lane_count,
                history_block_tokens=glm53_flash.SPARSE_MLA_HISTORY_BLOCK_TOKENS,
            ),
            token_budget=(
                glm53_flash.KDA_CHUNK_SIZE + max(decode_batch_sizes) * decode_width
            ),
            prefill_chunk_size=glm53_flash.KDA_CHUNK_SIZE,
            decode_width=decode_width,
            max_batch_size=max(decode_batch_sizes),
            max_prefill_rows=max(decode_batch_sizes),
            max_decode_ticks_between_prefills=8,
            graph_buckets=decode_batch_sizes,
            max_queued_requests=lane_count * live_slot_count,
            single_resident_prefill_chunk_size=(
                _DEP4_SINGLE_RESIDENT_PREFILL_CHUNK_SIZE
                if parallelism == "dep4"
                else None
            ),
        )
    return DistributedEngine(
        rank,
        scheduler,
        worker,
        plan_connections,
    )


def _drive_rank_zero(
    engine: DistributedEngine,
    codec: GLM53Codec,
    host: str,
    port: int,
    parallelism: str,
    speculation: str = "native",
) -> None:
    lane_count = glm53_flash.TP_SIZE if parallelism == "dep4" else 1
    live_slot_count = _live_slot_count(parallelism, speculation)
    service = Service(
        engine,
        capacity=lane_count * live_slot_count,
        vocab_size=glm53_flash.TOKENIZER_VOCAB_SIZE,
        max_context_tokens=glm53_flash.SPARSE_MLA_MAX_CONTEXT_TOKENS,
        eos_token_ids=EOS_TOKEN_IDS,
    )
    drive_service(
        service,
        create_app(codec=codec, service=service, model_name=MODEL_ID),
        host,
        port,
        "glm53_flash-http",
    )
    if engine.tick() is not None:
        raise RuntimeError("closed GLM service retained scheduled work")


def _live_slot_count(parallelism: str, speculation: str) -> int:
    if parallelism == "dep4":
        return (
            _DEP4_NATIVE_LIVE_SLOTS
            if speculation == "native"
            else _DEP4_TARGET_ONLY_LIVE_SLOTS
        )
    return GLM53_TARGET_LIVE_SLOTS


def _bind_b200(torch: object, rank: int) -> object:
    return bind_b200(torch, rank, "GLM")

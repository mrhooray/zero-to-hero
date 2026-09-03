from datetime import timedelta
from pathlib import Path

from infer.engine import DistributedEngine
from infer.http import create_app
from infer.models.deepseek_v4_flash import MODEL_ID
from infer.models.deepseek_v4_flash import model as deepseek_v4_flash
from infer.models.deepseek_v4_flash.checkpoint import open_deepseek_v4_checkpoint
from infer.models.deepseek_v4_flash.codec import EOS_TOKEN_IDS, DeepSeekV4Codec
from infer.models.deepseek_v4_flash.dspark import (
    DeepSeekV4DSparkModel,
    allocate_dep4_dspark_runtime,
    load_dep4_dspark_weights,
)
from infer.models.deepseek_v4_flash.target import (
    DeepSeekV4TargetModel,
    allocate_target_runtime,
    load_dep4_target_weights,
    load_tep4_target_weights,
)
from infer.models.deepseek_v4_flash.worker import (
    DeepSeekV4TargetWorker,
    _run_decode,
    _run_target_decode,
)
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

_PROVISIONAL_LIVE_SLOTS_PER_RANK = 64
_PREFIX_SNAPSHOT_SLOTS_PER_RANK = 2 * _PROVISIONAL_LIVE_SLOTS_PER_RANK
_PROVISIONAL_HISTORY_BLOCKS_PER_RANK = (
    deepseek_v4_flash.MAX_CONTEXT_TOKENS
    // deepseek_v4_flash.COMPRESSED_HISTORY_CANDIDATE_BLOCK_TOKENS
)


def serve_rank(
    rank: int,
    checkpoint: str,
    host: str,
    port: int,
    rendezvous: str,
    plan_pipes: tuple[tuple[object, object], ...],
    *,
    parallelism: str,
    speculation: str,
) -> None:
    plan_connections = take_plan_connections(rank, plan_pipes)
    set_rank_env(rank)
    import torch
    import torch.distributed as dist

    device = bind_b200(torch, rank, "DeepSeek V4")
    dist.init_process_group(
        "nccl",
        init_method=rendezvous,
        rank=rank,
        world_size=deepseek_v4_flash.DEP_SIZE,
        timeout=timedelta(hours=2),
        device_id=device,
    )
    try:
        engine = _build_engine(
            rank,
            Path(checkpoint),
            device,
            dist,
            plan_connections,
            parallelism,
            speculation,
        )
        codec = DeepSeekV4Codec(checkpoint) if rank == 0 else None
        wait_until_ready(dist, rank, "DeepSeek V4")
        if rank == 0:
            assert codec is not None
            _drive_rank_zero(engine, codec, host, port, parallelism)
        else:
            drive_worker(engine)
        dist.barrier()
    finally:
        dist.destroy_process_group()
        for connection in plan_connections:
            connection.close()


def _build_engine(
    rank: int,
    checkpoint: Path,
    device: object,
    dist: object,
    plan_connections: tuple[object, ...],
    parallelism: str,
    speculation: str = "native",
) -> DistributedEngine:
    import torch

    if parallelism not in {"dep4", "tep4"}:
        raise ValueError("DeepSeek parallelism must be dep4 or tep4")
    if speculation not in {"native", "none"}:
        raise ValueError("DeepSeek speculation must be native or none")
    tensor_parallel = parallelism == "tep4"
    speculative = speculation == "native"
    decode_width = deepseek_v4_flash.DSPARK_VERIFY_WIDTH if speculative else 1
    _record_memory(torch, device, rank, "start")
    view = open_deepseek_v4_checkpoint(checkpoint)
    weights = (
        load_tep4_target_weights if tensor_parallel else load_dep4_target_weights
    )(view, rank, device)
    from infer.models.deepseek_v4_flash.ops.target import (
        DeepSeekV4TargetOps,
        DeepSeekV4TEP4TargetOps,
    )

    if tensor_parallel:
        ops = DeepSeekV4TEP4TargetOps(dist.group.WORLD)
    else:
        ops = DeepSeekV4TargetOps()
    model = DeepSeekV4TargetModel(weights, ops)
    dspark_model = None
    if speculative:
        from infer.models.deepseek_v4_flash.ops.dspark import DeepSeekV4DSparkOps

        target_embedding, target_head = weights.embedding, weights.head
        if tensor_parallel:
            target_embedding = view.load_target_tensor(
                "embed.weight", rank, device, sharded=False
            )
            target_head = view.load_target_tensor(
                "head.weight", rank, device, sharded=False
            )
        dspark_model = DeepSeekV4DSparkModel(
            load_dep4_dspark_weights(view, rank, device),
            target_embedding,
            target_head,
            DeepSeekV4DSparkOps(),
        )
    _record_memory(torch, device, rank, "weights")

    runtime = allocate_target_runtime(
        weights,
        _PROVISIONAL_LIVE_SLOTS_PER_RANK,
        _PROVISIONAL_HISTORY_BLOCKS_PER_RANK,
        device,
        speculative=speculative,
        tensor_parallel=tensor_parallel,
        snapshot_slot_count=_PREFIX_SNAPSHOT_SLOTS_PER_RANK,
    )
    dspark_runtime = (
        allocate_dep4_dspark_runtime(runtime, device) if speculative else None
    )
    _record_memory(torch, device, rank, "runtime")
    graphs = _warm_and_capture(
        rank,
        model,
        runtime,
        dspark_model,
        dspark_runtime,
        dist,
        torch,
        parallelism,
    )
    require_memory_reserve(torch, device, rank)
    _record_memory(torch, device, rank, "graphs")

    worker = DeepSeekV4TargetWorker(
        model,
        runtime,
        dspark_model,
        dspark_runtime,
        graphs,
        rank,
        dist.group.WORLD,
        dist,
        (torch.cuda.Event(), torch.cuda.Event()),
        parallelism,
    )
    scheduler = None
    if rank == 0:
        scheduler = Scheduler(
            StateManager(
                history_block_count=_PROVISIONAL_HISTORY_BLOCKS_PER_RANK,
                live_slot_count=_PROVISIONAL_LIVE_SLOTS_PER_RANK,
                snapshot_slot_count=_PREFIX_SNAPSHOT_SLOTS_PER_RANK,
                lane_count=deepseek_v4_flash.DEP_SIZE if not tensor_parallel else 1,
                history_block_tokens=(
                    deepseek_v4_flash.COMPRESSED_HISTORY_CANDIDATE_BLOCK_TOKENS
                ),
            ),
            token_budget=(
                deepseek_v4_flash.PREFILL_CHUNK_TOKENS
                + max(deepseek_v4_flash.DECODE_BATCH_SIZES) * decode_width
            ),
            prefill_chunk_size=deepseek_v4_flash.PREFILL_CHUNK_TOKENS,
            decode_width=decode_width,
            max_batch_size=max(deepseek_v4_flash.DECODE_BATCH_SIZES),
            max_prefill_rows=deepseek_v4_flash.MAX_PREFILL_REQUESTS,
            max_decode_ticks_between_prefills=16,
            graph_buckets=deepseek_v4_flash.DECODE_BATCH_SIZES,
            max_queued_requests=(
                (deepseek_v4_flash.DEP_SIZE if not tensor_parallel else 1)
                * _PROVISIONAL_LIVE_SLOTS_PER_RANK
            ),
        )
    return DistributedEngine(rank, scheduler, worker, plan_connections)


def _warm_and_capture(
    rank, model, runtime, dspark_model, dspark_runtime, dist, torch, parallelism
):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    speculative = dspark_model is not None
    decode = runtime.verify if speculative else runtime.decode
    keys = tuple(sorted(decode))
    tokens_per_request = (
        deepseek_v4_flash.PREFILL_CHUNK_TOKENS // deepseek_v4_flash.MAX_PREFILL_REQUESTS
    )
    blocks_per_request = (
        tokens_per_request // deepseek_v4_flash.COMPRESSED_HISTORY_CANDIDATE_BLOCK_TOKENS
    )
    max_prefill = (
        ((0,) * tokens_per_request,) * deepseek_v4_flash.MAX_PREFILL_REQUESTS,
        (0,) * deepseek_v4_flash.MAX_PREFILL_REQUESTS,
        tuple(range(deepseek_v4_flash.MAX_PREFILL_REQUESTS)),
        tuple(
            (
                0,
                tuple(
                    range(
                        request * blocks_per_request,
                        (request + 1) * blocks_per_request,
                    )
                ),
            )
            for request in range(deepseek_v4_flash.MAX_PREFILL_REQUESTS)
        ),
        tuple(range(deepseek_v4_flash.MAX_PREFILL_REQUESTS)),
    )
    with torch.cuda.stream(stream):
        for _ in range(4):
            for key in keys:
                decode[key].stage(())
                decode[key].reset_attention_metadata()
                if speculative:
                    dspark_runtime.batch(*key).reset_attention_metadata()
                    _run_decode(model, runtime, dspark_model, dspark_runtime, key)
                else:
                    _run_target_decode(model, runtime, key)

        warm_lanes = deepseek_v4_flash.DEP_SIZE if parallelism == "dep4" else 1
        for active_rank in range(warm_lanes):
            if parallelism == "tep4" or rank == active_rank:
                for slot in range(deepseek_v4_flash.MAX_PREFILL_REQUESTS):
                    runtime.reset_state(slot)
                batch = runtime.stage_prefill(0, *max_prefill)
            else:
                batch = runtime.stage_prefill(0, (), (), (), (), ())
            model.prefill(runtime, batch)
            if speculative:
                dspark_model.seed_prefill(dspark_runtime, runtime, batch)
            runtime.publish_prefill(batch)
    stream.synchronize()
    dist.barrier(group=dist.group.WORLD)

    graphs = {}
    pool = torch.cuda.graph_pool_handle()
    for key in keys:
        decode[key].reset_attention_metadata()
        if speculative:
            dspark_runtime.batch(*key).reset_attention_metadata()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=pool, stream=stream):
            if speculative:
                _run_decode(model, runtime, dspark_model, dspark_runtime, key)
            else:
                _run_target_decode(model, runtime, key)
        graphs[key] = graph
    torch.cuda.synchronize()
    dist.barrier(group=dist.group.WORLD)
    return graphs


def _record_memory(
    torch: object,
    device: object,
    rank: int,
    stage: str,
) -> None:
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    allocated_bytes = torch.cuda.memory_allocated(device)
    reserved_bytes = torch.cuda.memory_reserved(device)
    print(
        f"rank={rank} stage={stage} allocated={allocated_bytes} "
        f"reserved={reserved_bytes} free={free_bytes} total={total_bytes}",
        flush=True,
    )


def _drive_rank_zero(
    engine: DistributedEngine,
    codec: DeepSeekV4Codec,
    host: str,
    port: int,
    parallelism: str,
) -> None:
    service = Service(
        engine,
        capacity=(deepseek_v4_flash.DEP_SIZE if parallelism == "dep4" else 1)
        * _PROVISIONAL_LIVE_SLOTS_PER_RANK,
        vocab_size=deepseek_v4_flash.VOCAB_SIZE,
        max_context_tokens=deepseek_v4_flash.MAX_CONTEXT_TOKENS,
        eos_token_ids=EOS_TOKEN_IDS,
    )
    drive_service(
        service,
        create_app(codec=codec, service=service, model_name=MODEL_ID),
        host,
        port,
        "deepseek-v4-http",
    )
    if engine.tick() is not None:
        raise RuntimeError("closed DeepSeek V4 service retained scheduled work")

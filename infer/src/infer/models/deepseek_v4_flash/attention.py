from collections.abc import Mapping
from dataclasses import dataclass

from infer.models.deepseek_v4_flash import checkpoint, megamoe
from infer.models.deepseek_v4_flash import model as deepseek_v4_flash
from infer.models.deepseek_v4_flash.model import (
    DECODE_BATCH_SIZES,
    FP8_BLOCK_SIZE,
    HEAD_DIM,
    HIDDEN_SIZE,
    LOCAL_QUERY_HEADS,
    NUM_QUERY_HEADS,
    DeepSeekV4AttentionPool,
    DeepSeekV4Compression,
    DeepSeekV4DecodeWorkspace,
    DeepSeekV4IndexQuery,
)

Q_LORA_RANK = 1_024
INDEX_HEAD_DIM = 128
Shape = tuple[int, ...]


@dataclass(frozen=True, slots=True)
class DeepSeekV4ProjectionWeights[TensorT]:
    input_norm: TensorT
    sink: TensorT
    qkv_a: TensorT
    qkv_a_scale: TensorT
    q_norm: TensorT
    q_b: TensorT
    q_b_scale: TensorT
    kv_norm: TensorT
    output_a: TensorT
    output_a_scale: TensorT
    output_b: TensorT
    output_b_scale: TensorT


@dataclass(frozen=True, slots=True)
class DeepSeekV4C4AttentionWeights[TensorT]:
    projection: DeepSeekV4ProjectionWeights[TensorT]
    main_kv_score_t: TensorT
    main_ape: TensorT
    main_norm: TensorT
    index_weights_t: TensorT
    index_kv_score_t: TensorT
    index_ape: TensorT
    index_norm: TensorT


@dataclass(frozen=True, slots=True)
class DeepSeekV4C128AttentionWeights[TensorT]:
    projection: DeepSeekV4ProjectionWeights[TensorT]
    main_kv_score_t: TensorT
    main_ape: TensorT
    main_norm: TensorT


@dataclass(frozen=True, slots=True)
class DeepSeekV4AttentionWorkspace[TensorT]:
    normalized: TensorT
    hidden_fp32: TensorT
    hidden_fp8: TensorT
    hidden_scale: TensorT
    qkv_low_rank: TensorT
    q_residual: TensorT
    q_residual_fp8: TensorT
    q_residual_scale: TensorT
    projected_q: TensorT
    normalized_kv: TensorT
    output_a: TensorT
    output_fp8: TensorT
    output_scale: TensorT
    output: TensorT


@dataclass(frozen=True, slots=True)
class DeepSeekV4C4AttentionWorkspace[TensorT]:
    common: DeepSeekV4AttentionWorkspace[TensorT]
    packed_q: TensorT
    index_weights: TensorT


@dataclass(frozen=True, slots=True)
class DeepSeekV4MHCWeights[TensorT]:
    base: TensorT
    fn: TensorT
    scale: TensorT


@dataclass(frozen=True, slots=True)
class DeepSeekV4DecodeLayerWeights[TensorT]:
    layer_id: int
    attention_mhc: DeepSeekV4MHCWeights[TensorT]
    attention: (
        DeepSeekV4ProjectionWeights[TensorT]
        | DeepSeekV4C4AttentionWeights[TensorT]
        | DeepSeekV4C128AttentionWeights[TensorT]
    )
    ffn_mhc: DeepSeekV4MHCWeights[TensorT]
    ffn_norm: TensorT
    ffn: megamoe.DeepSeekV4MoEWeights[TensorT]


@dataclass(frozen=True, slots=True)
class DeepSeekV4C4DecodeState[TensorT]:
    index: DeepSeekV4Compression[TensorT]
    query: DeepSeekV4IndexQuery[TensorT]
    workspace: DeepSeekV4DecodeWorkspace[TensorT]
    prepare_selection: bool


@dataclass(frozen=True, slots=True)
class DeepSeekV4C128DecodeState[TensorT]:
    pool: DeepSeekV4AttentionPool[TensorT]


@dataclass(frozen=True, slots=True)
class DeepSeekV4DecodeLayerWorkspace[TensorT]:
    attention: (
        DeepSeekV4C4AttentionWorkspace[TensorT] | DeepSeekV4AttentionWorkspace[TensorT]
    )
    ffn: megamoe.DeepSeekV4MoEWorkspace[TensorT]
    post: TensorT
    comb: TensorT
    collapsed: TensorT
    streams_mid: TensorT
    streams_out: TensorT


def decode_layer_workspace_shapes(
    layer_id: int,
    token_count: int,
    *,
    query_heads: int = NUM_QUERY_HEADS,
) -> DeepSeekV4DecodeLayerWorkspace[Shape]:
    verify_sizes = {
        batch_size * deepseek_v4_flash.DSPARK_VERIFY_WIDTH
        for batch_size in DECODE_BATCH_SIZES
    }
    if token_count not in (0, *DECODE_BATCH_SIZES) and token_count not in verify_sizes:
        raise ValueError("token_count must be zero or a configured decode bucket")
    return _layer_workspace_shapes(
        layer_id, token_count, query_heads=query_heads, prefill=False
    )


def prefill_layer_workspace_shapes(
    layer_id: int,
    token_count: int,
    *,
    query_heads: int = NUM_QUERY_HEADS,
) -> DeepSeekV4DecodeLayerWorkspace[Shape]:
    if not 0 <= token_count <= deepseek_v4_flash.PREFILL_CHUNK_TOKENS:
        raise ValueError("prefill token_count must be between zero and 4096")
    return _layer_workspace_shapes(
        layer_id, token_count, query_heads=query_heads, prefill=True
    )


def _layer_workspace_shapes(layer_id, token_count, *, query_heads, prefill):
    kind = deepseek_v4_flash.attention_kind(layer_id)
    attention = (
        _c4_workspace_shapes(
            token_count, query_heads, allow_empty=not prefill, prefill=prefill
        )
        if kind == "csa"
        else _attention_workspace_shapes(
            token_count, query_heads, allow_empty=not prefill, prefill=prefill
        )
    )
    streams = deepseek_v4_flash.MHC_STREAMS
    return DeepSeekV4DecodeLayerWorkspace(
        attention=attention,
        ffn=(
            megamoe.tep4_moe_workspace_shapes(token_count)
            if query_heads == LOCAL_QUERY_HEADS
            else megamoe.moe_workspace_shapes(token_count)
        ),
        post=(token_count, streams),
        comb=(token_count, streams, streams),
        collapsed=(token_count, HIDDEN_SIZE),
        streams_mid=(token_count, streams, HIDDEN_SIZE),
        streams_out=(token_count, streams, HIDDEN_SIZE),
    )


def load_attention_weights(
    view: checkpoint.DeepSeekV4CheckpointView,
    layer_id: int,
    rank: int,
    device: str | int,
    *,
    tensor_parallel: bool,
) -> (
    DeepSeekV4ProjectionWeights[object]
    | DeepSeekV4C4AttentionWeights[object]
    | DeepSeekV4C128AttentionWeights[object]
):
    import torch

    kind = deepseek_v4_flash.attention_kind(layer_id)
    specs = (
        deepseek_v4_flash.COMPRESSED_ATTENTION_COMMON_WEIGHTS
        if kind == "swa"
        else deepseek_v4_flash.compressed_attention_weights(layer_id)
    )
    raw = {
        spec.name: view.load_target_tensor(
            spec.checkpoint_key_for_layer(layer_id),
            rank,
            device,
            sharded=tensor_parallel,
        )
        for spec in specs
    }
    if kind == "swa":
        return _pack_projection(raw, torch, pack_index=False)
    if kind == "csa":
        return _pack_c4_attention(raw, torch)
    return _pack_c128_attention(raw, torch)


def _pack_c4_attention(
    raw: Mapping[str, object], torch: object
) -> DeepSeekV4C4AttentionWeights[object]:
    return DeepSeekV4C4AttentionWeights(
        projection=_pack_projection(raw, torch, pack_index=True),
        main_kv_score_t=_pack_compressor_weight(
            raw["attn.compressor.wkv.weight"],
            raw["attn.compressor.wgate.weight"],
            torch,
        ),
        main_ape=_reorder_c4_ape(raw["attn.compressor.ape"], torch),
        main_norm=raw["attn.compressor.norm.weight"],
        index_weights_t=raw["attn.indexer.weights_proj.weight"].T.contiguous(),
        index_kv_score_t=_pack_compressor_weight(
            raw["attn.indexer.compressor.wkv.weight"],
            raw["attn.indexer.compressor.wgate.weight"],
            torch,
        ),
        index_ape=_reorder_c4_ape(raw["attn.indexer.compressor.ape"], torch),
        index_norm=raw["attn.indexer.compressor.norm.weight"],
    )


def _pack_c128_attention(
    raw: Mapping[str, object], torch: object
) -> DeepSeekV4C128AttentionWeights[object]:
    return DeepSeekV4C128AttentionWeights(
        projection=_pack_projection(raw, torch, pack_index=False),
        main_kv_score_t=_pack_compressor_weight(
            raw["attn.compressor.wkv.weight"],
            raw["attn.compressor.wgate.weight"],
            torch,
        ),
        main_ape=raw["attn.compressor.ape"],
        main_norm=raw["attn.compressor.norm.weight"],
    )


def _attention_workspace_shapes(
    batch_size: int,
    query_heads: int,
    *,
    allow_empty: bool = False,
    prefill: bool = False,
) -> DeepSeekV4AttentionWorkspace[Shape]:
    valid = (
        0 <= batch_size <= deepseek_v4_flash.PREFILL_CHUNK_TOKENS
        if prefill
        else batch_size in DECODE_BATCH_SIZES
        or batch_size
        in {size * deepseek_v4_flash.DSPARK_VERIFY_WIDTH for size in DECODE_BATCH_SIZES}
        or (allow_empty and batch_size == 0)
    )
    if not valid:
        raise ValueError(
            f"batch_size must be one of {DECODE_BATCH_SIZES}, got {batch_size}"
        )
    output_groups = query_heads * HEAD_DIM // HIDDEN_SIZE
    return DeepSeekV4AttentionWorkspace(
        normalized=(batch_size, HIDDEN_SIZE),
        hidden_fp32=(batch_size, HIDDEN_SIZE),
        hidden_fp8=(batch_size, HIDDEN_SIZE),
        hidden_scale=(batch_size, HIDDEN_SIZE // FP8_BLOCK_SIZE),
        qkv_low_rank=(batch_size, 1_536),
        q_residual=(batch_size, Q_LORA_RANK),
        q_residual_fp8=(batch_size, Q_LORA_RANK),
        q_residual_scale=(batch_size, Q_LORA_RANK // FP8_BLOCK_SIZE),
        projected_q=(batch_size, query_heads, HEAD_DIM),
        normalized_kv=(batch_size, HEAD_DIM),
        output_a=(batch_size, output_groups, 1_024),
        output_fp8=(batch_size, output_groups * 1_024),
        output_scale=((batch_size + 3) // 4 * 4, output_groups * 8),
        output=(batch_size, HIDDEN_SIZE),
    )


def _c4_workspace_shapes(
    batch_size: int,
    query_heads: int,
    *,
    allow_empty: bool = False,
    prefill: bool = False,
) -> DeepSeekV4C4AttentionWorkspace[Shape]:
    return DeepSeekV4C4AttentionWorkspace(
        common=_attention_workspace_shapes(
            batch_size, query_heads, allow_empty=allow_empty, prefill=prefill
        ),
        packed_q=(
            batch_size,
            query_heads * HEAD_DIM + NUM_QUERY_HEADS * INDEX_HEAD_DIM,
        ),
        index_weights=(batch_size, NUM_QUERY_HEADS),
    )


def _pack_projection(
    raw: Mapping[str, object], torch: object, *, pack_index: bool
) -> DeepSeekV4ProjectionWeights[object]:
    local_sink = raw["attn.attn_sink"]
    if local_sink.shape == (NUM_QUERY_HEADS,):
        sink = local_sink
    else:
        sink = torch.full(
            (NUM_QUERY_HEADS,),
            -float("inf"),
            dtype=torch.float32,
            device=local_sink.device,
        )
        sink[:LOCAL_QUERY_HEADS].copy_(local_sink)
    q_b = raw["attn.wq_b.weight"]
    q_b_scale = raw["attn.wq_b.scale"].to(torch.float32)
    if pack_index:
        q_b = torch.cat((q_b, raw["attn.indexer.wq_b.weight"]))
        q_b_scale = torch.cat(
            (q_b_scale, raw["attn.indexer.wq_b.scale"].to(torch.float32))
        )
    output_a = raw["attn.wo_a.weight"].reshape(-1, 1_024, 4_096)
    output_a_scale = raw["attn.wo_a.scale"].to(torch.float32).reshape(-1, 8, 32)
    if output_a_scale.device.type == "meta":
        output_a_scale = torch.empty(
            (output_a.shape[0], 1_024, 8), dtype=torch.int32, device="meta"
        )
    else:
        import deep_gemm

        output_a_scale = deep_gemm.transform_sf_into_required_layout(
            deep_gemm.ceil_to_ue8m0(output_a_scale),
            mn=1_024,
            k=4_096,
            recipe=(1, 128, 128),
            num_groups=output_a.shape[0],
            is_sfa=False,
        )
    return DeepSeekV4ProjectionWeights(
        input_norm=raw["attn_norm.weight"],
        sink=sink,
        qkv_a=torch.cat((raw["attn.wq_a.weight"], raw["attn.wkv.weight"])),
        qkv_a_scale=torch.cat(
            (
                raw["attn.wq_a.scale"].to(torch.float32),
                raw["attn.wkv.scale"].to(torch.float32),
            )
        ),
        q_norm=raw["attn.q_norm.weight"],
        q_b=q_b,
        q_b_scale=q_b_scale,
        kv_norm=raw["attn.kv_norm.weight"],
        output_a=output_a,
        output_a_scale=output_a_scale,
        output_b=raw["attn.wo_b.weight"],
        output_b_scale=raw["attn.wo_b.scale"].to(torch.float32),
    )


def _pack_compressor_weight(kv: object, score: object, torch: object) -> object:
    return torch.cat((kv, score)).T.contiguous()


def _reorder_c4_ape(checkpoint_ape: object, torch: object) -> object:
    older, newer = checkpoint_ape.chunk(2, dim=-1)
    return torch.cat((older, newer), dim=0).reshape_as(checkpoint_ape).contiguous()

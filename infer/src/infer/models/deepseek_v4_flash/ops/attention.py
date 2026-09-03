"""DeepSeek V4 decode-layer attention and mHC execution."""

import hashlib
from dataclasses import fields
from functools import cache
from importlib.metadata import version
from pathlib import Path

import deep_gemm
import infer_deepseek_v4_writer as deepseek_writer
import torch
import triton
import triton.language as tl

from infer.models.deepseek_v4_flash.attention import (
    FP8_BLOCK_SIZE,
    INDEX_HEAD_DIM,
    DeepSeekV4AttentionWorkspace,
    DeepSeekV4C4AttentionWeights,
    DeepSeekV4C4AttentionWorkspace,
    DeepSeekV4C4DecodeState,
    DeepSeekV4C128AttentionWeights,
    DeepSeekV4C128DecodeState,
    DeepSeekV4DecodeLayerWeights,
    DeepSeekV4DecodeLayerWorkspace,
    DeepSeekV4ProjectionWeights,
    _attention_workspace_shapes,
    _c4_workspace_shapes,
    decode_layer_workspace_shapes,
    prefill_layer_workspace_shapes,
)
from infer.models.deepseek_v4_flash.model import (
    DECODE_BATCH_SIZES,
    DSPARK_VERIFY_WIDTH,
    HEAD_DIM,
    HIDDEN_SIZE,
    LOCAL_QUERY_HEADS,
    MHC_EPS,
    MHC_NORMALIZATION_ITERS,
    MHC_STREAMS,
    NUM_QUERY_HEADS,
    RMS_NORM_EPS,
    TEP_SIZE,
    DeepSeekV4AttentionPool,
    DeepSeekV4Compression,
    DeepSeekV4CompressionBatch,
    DeepSeekV4PrefillIndexWorkspace,
    DeepSeekV4PrefillMetadata,
    DeepSeekV4PrefillState,
    DeepSeekV4PrefillWorkspace,
    DeepSeekV4RawAttention,
    attention_kind,
    uses_hash_routing,
)
from infer.models.deepseek_v4_flash.ops.core import (
    DeepSeekV4Ops,
    _c4_decode_query,
    _copy_c4_raw_query,
    _fp8_gemm,
    _quantize,
    _require_tensor,
)
from infer.models.deepseek_v4_flash.ops.mhc import deepseek_v4_mhc_transition

WRITER_PACKAGE = "infer-deepseek-v4-writer"
WRITER_VERSION = "0.3.0+f17b03ef"
WRITER_SHA256 = "f49a2af7d66847f4a4ddb622cf6aa2cf6cbb945b098170ff2241abaab54f985f"
WRITER_SIZE = 279_432
INDEX_WEIGHT_SCALE = (INDEX_HEAD_DIM * NUM_QUERY_HEADS) ** -0.5


class DeepSeekV4AttentionOps:
    def __init__(self) -> None:
        if version(WRITER_PACKAGE) != WRITER_VERSION or (
            _writer_identity() != (WRITER_SHA256, WRITER_SIZE)
        ):
            raise RuntimeError("DeepSeek V4 writer identity mismatch")
        writer = getattr(deepseek_writer, "fused_qnorm_rope_kv_insert", None)
        if not callable(writer):
            raise TypeError("DeepSeek V4 writer-v3 ABI is unavailable")
        torch.set_float32_matmul_precision("highest")
        self._writer = writer
        self._core = DeepSeekV4Ops()
        self._query_heads = NUM_QUERY_HEADS
        self._process_group = None
        self._compressor_stream = torch.cuda.Stream()
        self._index_stream = torch.cuda.Stream()
        self._input_ready = torch.cuda.Event()
        self._query_ready = torch.cuda.Event()
        self._compressor_done = torch.cuda.Event()
        self._index_done = torch.cuda.Event()

    def prefill_c4_attention(
        self,
        projection: DeepSeekV4C4AttentionWorkspace[torch.Tensor],
        batch: DeepSeekV4CompressionBatch[torch.Tensor],
        metadata: DeepSeekV4PrefillMetadata[torch.Tensor],
        main: DeepSeekV4Compression[torch.Tensor],
        index: DeepSeekV4Compression[torch.Tensor],
        raw: DeepSeekV4RawAttention[torch.Tensor],
        index_workspace: DeepSeekV4PrefillIndexWorkspace[torch.Tensor],
        workspace: DeepSeekV4PrefillWorkspace[torch.Tensor],
        main_state: DeepSeekV4PrefillState[torch.Tensor],
        index_state: DeepSeekV4PrefillState[torch.Tensor],
    ) -> torch.Tensor:
        self._core.prefill_compress_c4(batch, main, index, main_state, index_state)
        candidates = self._core.prefill_c4_candidates(
            metadata,
            batch,
            index.cache,
            projection.packed_q,
            projection.index_weights,
            index_workspace,
        )
        return self._prefill_attention(
            projection.common,
            batch,
            metadata,
            main,
            raw,
            candidates,
            workspace,
            4,
        )

    def _prefill_attention(
        self,
        projection: DeepSeekV4AttentionWorkspace[torch.Tensor],
        batch: DeepSeekV4CompressionBatch[torch.Tensor],
        metadata: DeepSeekV4PrefillMetadata[torch.Tensor],
        main: DeepSeekV4Compression[torch.Tensor] | None,
        raw: DeepSeekV4RawAttention[torch.Tensor],
        local_candidates: torch.Tensor | None,
        workspace: DeepSeekV4PrefillWorkspace[torch.Tensor],
        ratio: int,
    ) -> torch.Tensor:
        self._core.stage_prefill_attention(
            metadata,
            batch,
            projection.normalized_kv,
            raw,
            main.cache if main is not None else raw.cache,
            local_candidates,
            workspace,
            ratio,
        )
        self._writer(
            projection.projected_q,
            projection.normalized_kv,
            raw.query,
            raw.cache,
            workspace.raw_slots,
            batch.positions,
            batch.cos_sin,
            RMS_NORM_EPS,
        )
        return self._core.prefill_selected_attention(
            raw.query[:, 0],
            workspace.kv,
            raw.indices,
            raw.lengths,
            raw.sink,
        )[:, : self._query_heads]

    def prefill_layer(
        self,
        streams: torch.Tensor,
        token_ids: torch.Tensor,
        weights: DeepSeekV4DecodeLayerWeights[torch.Tensor],
        batch: DeepSeekV4CompressionBatch[torch.Tensor],
        main: DeepSeekV4Compression[torch.Tensor] | None,
        index: DeepSeekV4Compression[torch.Tensor] | None,
        raw: DeepSeekV4RawAttention[torch.Tensor],
        metadata: DeepSeekV4PrefillMetadata[torch.Tensor],
        index_workspace: DeepSeekV4PrefillIndexWorkspace[torch.Tensor] | None,
        attention_workspace: DeepSeekV4PrefillWorkspace[torch.Tensor],
        workspace: DeepSeekV4DecodeLayerWorkspace[torch.Tensor],
        main_state: DeepSeekV4PrefillState[torch.Tensor] | None,
        index_state: DeepSeekV4PrefillState[torch.Tensor] | None,
        mega_buffer: object,
    ) -> torch.Tensor:
        _validate_prefill_layer(
            streams, token_ids, weights, workspace, self._query_heads
        )
        common = workspace.attention
        if isinstance(common, DeepSeekV4C4AttentionWorkspace):
            common = common.common
        if streams.shape[0]:
            _mhc_pre(streams, weights.attention_mhc, workspace)
            kind = attention_kind(weights.layer_id)
            if kind == "swa":
                projection = weights.attention
                _project_common(workspace.collapsed, projection, common)
                _project_query(
                    projection, common, common.projected_q.view(token_ids.shape[0], -1)
                )
                attention = self._prefill_attention(
                    common,
                    batch,
                    metadata,
                    None,
                    raw,
                    None,
                    attention_workspace,
                    1,
                )
            elif kind == "csa":
                projection = weights.attention.projection
                _project_common(workspace.collapsed, projection, common)
                _project_compressor(
                    common.normalized,
                    weights.attention.main_kv_score_t,
                    main.kv_score,
                )
                _project_compressor(
                    common.normalized,
                    weights.attention.index_kv_score_t,
                    index.kv_score,
                )
                _project_query(projection, common, workspace.attention.packed_q)
                _copy_c4_raw_query(workspace.attention.packed_q, common.projected_q)
                torch.mm(
                    common.normalized,
                    weights.attention.index_weights_t,
                    out=workspace.attention.index_weights,
                )
                attention = self.prefill_c4_attention(
                    workspace.attention,
                    batch,
                    metadata,
                    main,
                    index,
                    raw,
                    index_workspace,
                    attention_workspace,
                    main_state,
                    index_state,
                )
            else:
                projection = weights.attention.projection
                _project_common(workspace.collapsed, projection, common)
                _project_query(
                    projection, common, common.projected_q.view(token_ids.shape[0], -1)
                )
                _project_compressor(
                    common.normalized,
                    weights.attention.main_kv_score_t,
                    main.kv_score,
                )
                self._core.prefill_compress_c128(batch, main, main_state)
                attention = self._prefill_attention(
                    common,
                    batch,
                    metadata,
                    main,
                    raw,
                    None,
                    attention_workspace,
                    128,
                )
            attention = _project_output(
                attention, projection, common, batch, self._process_group
            )
            _mhc_post(attention, streams, workspace, workspace.streams_mid)
            _mhc_pre(workspace.streams_mid, weights.ffn_mhc, workspace)
            _rmsnorm(workspace.collapsed, weights.ffn_norm, common.normalized)

        ffn = self._core._decode_moe(
            common.normalized,
            token_ids,
            weights.ffn,
            workspace.ffn,
            mega_buffer,
            uses_hash_routing(weights.layer_id),
            self._process_group,
        )
        if streams.shape[0]:
            _mhc_post(ffn, workspace.streams_mid, workspace, workspace.streams_out)
        return workspace.streams_out

    def decode_layer(
        self,
        streams: torch.Tensor,
        token_ids: torch.Tensor,
        weights: DeepSeekV4DecodeLayerWeights[torch.Tensor],
        batch: DeepSeekV4CompressionBatch[torch.Tensor],
        raw_slots: torch.Tensor,
        main: DeepSeekV4Compression[torch.Tensor] | None,
        raw: DeepSeekV4RawAttention[torch.Tensor],
        attention_state: (
            DeepSeekV4C4DecodeState[torch.Tensor]
            | DeepSeekV4C128DecodeState[torch.Tensor]
            | None
        ),
        workspace: DeepSeekV4DecodeLayerWorkspace[torch.Tensor],
        metadata: object,
        mega_buffer: object,
        *,
        attention_prepared: bool = False,
        next_weights: DeepSeekV4DecodeLayerWeights[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        _validate_decode_layer(
            streams,
            token_ids,
            weights,
            attention_state,
            workspace,
            self._query_heads,
        )
        common = workspace.attention
        if isinstance(common, DeepSeekV4C4AttentionWorkspace):
            common = common.common
        tokens = streams.shape[0]
        if tokens:
            if not attention_prepared:
                _mhc_pre(streams, weights.attention_mhc, workspace)
                _rmsnorm(
                    workspace.collapsed,
                    _projection_weights(weights.attention).input_norm,
                    common.normalized,
                )
            kind = attention_kind(weights.layer_id)
            if kind == "swa":
                attention = self._decode_swa_attention(
                    common.normalized,
                    weights.attention,
                    batch,
                    raw_slots,
                    raw,
                    workspace.attention,
                    metadata,
                    self._query_heads,
                    self._process_group,
                )
            elif isinstance(attention_state, DeepSeekV4C4DecodeState):
                attention = self._decode_c4_attention(
                    common.normalized,
                    weights.attention,
                    batch,
                    raw_slots,
                    main,
                    attention_state.index,
                    raw,
                    attention_state.query,
                    workspace.attention,
                    attention_state.workspace,
                    attention_state.prepare_selection,
                    metadata,
                    self._query_heads,
                    self._process_group,
                )
            else:
                attention = self._decode_c128_attention(
                    common.normalized,
                    weights.attention,
                    batch,
                    raw_slots,
                    main,
                    raw,
                    attention_state.pool,
                    workspace.attention,
                    metadata,
                    self._query_heads,
                    self._process_group,
                )
            if tokens in (1, 4, 16, 32):
                deepseek_v4_mhc_transition(
                    attention,
                    streams,
                    workspace.post,
                    workspace.comb,
                    weights.ffn_mhc.fn,
                    weights.ffn_mhc.scale,
                    weights.ffn_mhc.base,
                    weights.ffn_norm,
                    workspace.streams_mid,
                    common.normalized,
                    common.hidden_fp32,
                )
            else:
                _mhc_post(attention, streams, workspace, workspace.streams_mid)
                _mhc_pre(workspace.streams_mid, weights.ffn_mhc, workspace)
                _rmsnorm(
                    workspace.collapsed,
                    weights.ffn_norm,
                    common.normalized,
                )

        ffn = self._core._decode_moe(
            common.normalized,
            token_ids,
            weights.ffn,
            workspace.ffn,
            mega_buffer,
            uses_hash_routing(weights.layer_id),
            self._process_group,
        )
        if tokens:
            if next_weights is None:
                _mhc_post(ffn, workspace.streams_mid, workspace, workspace.streams_out)
            else:
                deepseek_v4_mhc_transition(
                    ffn,
                    workspace.streams_mid,
                    workspace.post,
                    workspace.comb,
                    next_weights.attention_mhc.fn,
                    next_weights.attention_mhc.scale,
                    next_weights.attention_mhc.base,
                    _projection_weights(next_weights.attention).input_norm,
                    workspace.streams_out,
                    common.normalized,
                    common.hidden_fp32,
                )
        return workspace.streams_out

    def _decode_swa_attention(
        self,
        hidden: torch.Tensor,
        weights: DeepSeekV4ProjectionWeights[torch.Tensor],
        batch: DeepSeekV4CompressionBatch[torch.Tensor],
        raw_slots: torch.Tensor,
        raw: DeepSeekV4RawAttention[torch.Tensor],
        workspace: DeepSeekV4AttentionWorkspace[torch.Tensor],
        metadata: object,
        query_heads: int,
        process_group: object,
    ) -> torch.Tensor:
        _validate_common(
            hidden,
            weights,
            batch,
            raw_slots,
            raw,
            workspace,
            query_heads,
            process_group,
        )
        _project_qkv(weights, workspace)
        _project_query(
            weights, workspace, workspace.projected_q.view(hidden.shape[0], -1)
        )
        self._write_raw(workspace, batch, raw_slots, raw)
        output, _ = self._core.decode_swa(batch, raw, metadata, query_heads)
        return _project_output(output, weights, workspace, batch, process_group)

    def _decode_c4_attention(
        self,
        hidden,
        weights,
        batch,
        raw_slots,
        main,
        index,
        raw,
        query,
        workspace,
        core_workspace,
        prepare_selection,
        metadata,
        query_heads,
        process_group,
    ) -> torch.Tensor:
        _validate_common(
            hidden,
            weights.projection,
            batch,
            raw_slots,
            raw,
            workspace.common,
            query_heads,
            process_group,
        )
        _validate_c4_bindings(weights, main, index, query, workspace, query_heads)
        self._core.validate_c4(
            batch, main, index, raw, query, core_workspace, query_heads
        )
        if prepare_selection:
            self._core.prepare_c4(query, core_workspace)
        common = workspace.common
        current = torch.cuda.current_stream()
        self._input_ready.record(current)

        with torch.cuda.stream(self._compressor_stream):
            self._compressor_stream.wait_event(self._input_ready)
            _project_compressor(
                common.normalized, weights.main_kv_score_t, main.kv_score
            )
            self._core.compress_c4_main(batch, main)
            self._compressor_done.record(self._compressor_stream)

        with torch.cuda.stream(self._index_stream):
            self._index_stream.wait_event(self._input_ready)
            _project_compressor(
                common.normalized, weights.index_kv_score_t, index.kv_score
            )
            torch.mm(
                common.normalized,
                weights.index_weights_t,
                out=workspace.index_weights,
            )
            self._core.compress_c4_index(batch, index)

        _project_qkv(weights.projection, common)
        _project_query(weights.projection, common, workspace.packed_q)
        self._query_ready.record(current)

        with torch.cuda.stream(self._index_stream):
            self._index_stream.wait_event(self._query_ready)
            _c4_decode_query[(hidden.shape[0] * NUM_QUERY_HEADS,)](
                workspace.packed_q,
                workspace.index_weights,
                batch.positions,
                batch.cos_sin,
                query.query,
                query.scale,
                query.weights,
                HEADS=NUM_QUERY_HEADS,
                LOCAL_HEADS=query_heads,
                HEAD_WIDTH=INDEX_HEAD_DIM,
                PACKED_WIDTH=workspace.packed_q.shape[1],
                ROPE_WIDTH=64,
                WEIGHT_SCALE=INDEX_WEIGHT_SCALE,
                num_warps=4,
            )
            self._core.select_c4(index, query, core_workspace)
            self._index_done.record(self._index_stream)

        _copy_c4_raw_query(workspace.packed_q, common.projected_q)
        self._write_raw(common, batch, raw_slots, raw)
        current.wait_event(self._compressor_done)
        current.wait_event(self._index_done)
        output, _ = self._core.attend_c4(
            raw, main, core_workspace, metadata, query_heads
        )
        return _project_output(output, weights.projection, common, batch, process_group)

    def _decode_c128_attention(
        self,
        hidden: torch.Tensor,
        weights: DeepSeekV4C128AttentionWeights[torch.Tensor],
        batch: DeepSeekV4CompressionBatch[torch.Tensor],
        raw_slots: torch.Tensor,
        main: DeepSeekV4Compression[torch.Tensor],
        raw: DeepSeekV4RawAttention[torch.Tensor],
        pool: DeepSeekV4AttentionPool[torch.Tensor],
        workspace: DeepSeekV4AttentionWorkspace[torch.Tensor],
        metadata: object,
        query_heads: int,
        process_group: object,
    ) -> torch.Tensor:
        _validate_common(
            hidden,
            weights.projection,
            batch,
            raw_slots,
            raw,
            workspace,
            query_heads,
            process_group,
        )
        if main.ape.data_ptr() != weights.main_ape.data_ptr() or (
            main.norm_weight.data_ptr() != weights.main_norm.data_ptr()
        ):
            raise ValueError("C128 compressor must use the loaded APE/norm tensors")
        self._project_c128(hidden, weights, workspace, main)
        self._write_raw(workspace, batch, raw_slots, raw)
        output, _ = self._core.decode_c128(
            batch, main, raw, pool, metadata, query_heads
        )
        return _project_output(
            output, weights.projection, workspace, batch, process_group
        )

    def _project_c128(self, hidden, weights, workspace, main) -> None:
        current = torch.cuda.current_stream()
        self._input_ready.record(current)
        with torch.cuda.stream(self._compressor_stream):
            self._compressor_stream.wait_event(self._input_ready)
            _project_compressor(
                workspace.normalized, weights.main_kv_score_t, main.kv_score
            )
            self._compressor_done.record(self._compressor_stream)
        _project_qkv(weights.projection, workspace)
        _project_query(
            weights.projection,
            workspace,
            workspace.projected_q.view(hidden.shape[0], -1),
        )
        current.wait_event(self._compressor_done)

    def _write_raw(self, workspace, batch, raw_slots, raw) -> None:
        self._writer(
            workspace.projected_q,
            workspace.normalized_kv,
            raw.query,
            raw.cache,
            raw_slots,
            batch.positions,
            batch.cos_sin,
            RMS_NORM_EPS,
        )


class DeepSeekV4TEP4AttentionOps(DeepSeekV4AttentionOps):
    def __init__(self, process_group: object) -> None:
        super().__init__()
        if torch.distributed.get_world_size(process_group) != TEP_SIZE:
            raise ValueError("DeepSeek V4 attention requires a TP4 process group")
        self._query_heads = LOCAL_QUERY_HEADS
        self._process_group = process_group


@cache
def _writer_identity() -> tuple[str, int]:
    path = Path(deepseek_writer.__file__).resolve()
    return hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_size


def _project_common(hidden, weights, workspace) -> None:
    _project_input(hidden, weights, workspace)
    _project_qkv(weights, workspace)


def _projection_weights(weights):
    return getattr(weights, "projection", weights)


def _project_input(hidden, weights, workspace) -> None:
    _rmsnorm(hidden, weights.input_norm, workspace.normalized)


def _project_compressor(input_, weight, out) -> None:
    torch.mm(input_, weight, out=out, out_dtype=torch.float32)


def _project_qkv(weights, workspace) -> None:
    _quantize(workspace.normalized, workspace.hidden_fp8, workspace.hidden_scale)
    _fp8_gemm(
        workspace.hidden_fp8,
        weights.qkv_a,
        workspace.hidden_scale,
        weights.qkv_a_scale,
        workspace.qkv_low_rank,
    )
    _rmsnorm(workspace.qkv_low_rank[:, :1024], weights.q_norm, workspace.q_residual)
    _quantize(
        workspace.q_residual,
        workspace.q_residual_fp8,
        workspace.q_residual_scale,
    )
    _rmsnorm(workspace.qkv_low_rank[:, 1024:], weights.kv_norm, workspace.normalized_kv)


def _project_query(weights, workspace, out) -> None:
    _fp8_gemm(
        workspace.q_residual_fp8,
        weights.q_b,
        workspace.q_residual_scale,
        weights.q_b_scale,
        out,
    )


def _project_output(attention, weights, workspace, batch, process_group):
    _inverse_rope[(attention.shape[0] * attention.shape[-2],)](
        attention,
        batch.positions,
        batch.cos_sin,
        BATCH_STRIDE=attention.stride(0),
        HEAD_STRIDE=attention.stride(-2),
        HEADS=attention.shape[-2],
        HEAD_WIDTH=HEAD_DIM,
        ROPE_WIDTH=64,
        num_warps=1,
    )
    output_groups = weights.output_a.shape[0]
    grouped = attention.view(attention.shape[0], output_groups, -1)
    tokens = attention.shape[0]
    packs = HIDDEN_SIZE // FP8_BLOCK_SIZE // 4
    aligned_tokens = workspace.output_scale.shape[0]
    projected_q = workspace.projected_q.view(torch.float8_e4m3fn).flatten()
    projected_q = projected_q[: tokens * output_groups * HIDDEN_SIZE]
    projected_q = projected_q.view(output_groups, tokens, HIDDEN_SIZE).transpose(0, 1)
    packed_scale = workspace.output_scale.view(torch.int32).view(
        output_groups, packs, aligned_tokens
    )
    _power2_quantize_packed_groups[(tokens * output_groups * packs,)](
        grouped,
        projected_q,
        packed_scale,
        TOKENS=tokens,
        ALIGNED_TOKENS=aligned_tokens,
        GROUP_COUNT=output_groups,
        NUM_PACKS=packs,
        WIDTH=HIDDEN_SIZE,
        BLOCK_SIZE=FP8_BLOCK_SIZE,
        INPUT_ROW_STRIDE=grouped.stride(0),
        INPUT_GROUP_STRIDE=grouped.stride(1),
        num_warps=8,
    )
    deep_gemm.fp8_einsum(
        "bhr,hdr->bhd",
        (projected_q, packed_scale.permute(2, 0, 1)[:tokens]),
        (weights.output_a, weights.output_a_scale),
        workspace.output_a,
        recipe=(1, 1, FP8_BLOCK_SIZE),
    )
    output_scale = workspace.output_scale[:tokens]
    _quantize(workspace.output_a.view(tokens, -1), workspace.output_fp8, output_scale)
    _fp8_gemm(
        workspace.output_fp8,
        weights.output_b,
        output_scale,
        weights.output_b_scale,
        workspace.output,
    )
    if process_group is not None:
        workspace.hidden_fp32.copy_(workspace.output)
        torch.distributed.all_reduce(workspace.hidden_fp32, group=process_group)
        workspace.output.copy_(workspace.hidden_fp32)
    return workspace.output


def _rmsnorm(input_: torch.Tensor, weight: torch.Tensor, out: torch.Tensor) -> None:
    from flashinfer.norm import rmsnorm

    _require_tensor(weight, input_.dtype, (input_.shape[1],), input_.device)
    rmsnorm(input_, weight, eps=RMS_NORM_EPS, out=out, enable_pdl=True)


def _mhc_pre(streams, weights, workspace) -> None:
    from flashinfer.mhc import get_mhc_module

    rows = streams.shape[0]
    splits = _mhc_pre_splits(rows, streams.device)
    mix = MHC_STREAMS * (MHC_STREAMS + 2)
    common = workspace.attention
    if isinstance(common, DeepSeekV4C4AttentionWorkspace):
        common = common.common
    scratch = common.hidden_fp32.view(-1)
    dot = scratch[: splits * rows * mix].view(splits, rows, mix)
    sqrsum = scratch[splits * rows * mix : splits * rows * (mix + 1)].view(splits, rows)
    deep_gemm.tf32_hc_prenorm_gemm(
        streams.view(rows, -1),
        weights.fn,
        dot,
        sqrsum,
        num_splits=splits,
    )
    get_mhc_module().mhc_pre_big_fuse(
        workspace.post,
        workspace.comb,
        workspace.collapsed,
        dot if splits > 1 else dot[0],
        sqrsum if splits > 1 else sqrsum[0],
        streams,
        weights.scale,
        weights.base,
        MHC_STREAMS * HIDDEN_SIZE,
        RMS_NORM_EPS,
        MHC_EPS,
        MHC_EPS,
        2.0,
        MHC_NORMALIZATION_ITERS,
        splits,
        0,
    )


def _mhc_pre_splits(rows: int, device: torch.device) -> int:
    grid = max((rows + 63) // 64, 1)
    properties = torch.cuda.get_device_properties(device)
    available = min(properties.multi_processor_count // grid, 64)
    return max(split for split in (1, 2, 4, 8, 16) if split <= max(available, 1))


def _mhc_post(hidden, residual, workspace, out) -> None:
    from flashinfer.mhc import get_mhc_module

    get_mhc_module().mhc_post(out, hidden, residual, workspace.post, workspace.comb)


def _validate_decode_layer(
    streams, token_ids, weights, attention_state, workspace, query_heads=NUM_QUERY_HEADS
) -> None:
    if streams.ndim != 3:
        raise ValueError("streams must match a configured decode bucket")
    expected = decode_layer_workspace_shapes(
        weights.layer_id, streams.shape[0], query_heads=query_heads
    )
    kind = attention_kind(weights.layer_id)
    if kind == "swa":
        expected_types = (
            DeepSeekV4ProjectionWeights,
            DeepSeekV4AttentionWorkspace,
        )
        values = (weights.attention, workspace.attention)
        valid_state = attention_state is None
    elif kind == "csa":
        expected_types = (
            DeepSeekV4C4AttentionWeights,
            DeepSeekV4C4AttentionWorkspace,
            DeepSeekV4C4DecodeState,
        )
        values = (weights.attention, workspace.attention, attention_state)
        valid_state = True
    else:
        expected_types = (
            DeepSeekV4C128AttentionWeights,
            DeepSeekV4AttentionWorkspace,
            DeepSeekV4C128DecodeState,
        )
        values = (weights.attention, workspace.attention, attention_state)
        valid_state = True
    if not valid_state or not all(
        isinstance(value, type_)
        for value, type_ in zip(values, expected_types, strict=True)
    ):
        raise TypeError("decode attention weights, state, and workspace disagree")
    device = streams.device
    _require_tensor(streams, torch.bfloat16, expected.streams_out, device)
    _require_tensor(token_ids, torch.int64, (streams.shape[0],), device)
    _require_tensor(weights.ffn_norm, torch.bfloat16, (expected.collapsed[1],), device)
    float32 = {"post", "comb"}
    for field in fields(expected):
        if field.name not in {"attention", "ffn"}:
            dtype = torch.float32 if field.name in float32 else torch.bfloat16
            _require_tensor(
                getattr(workspace, field.name),
                dtype,
                getattr(expected, field.name),
                device,
            )
    mix = MHC_STREAMS * (MHC_STREAMS + 2)
    for mhc in (weights.attention_mhc, weights.ffn_mhc):
        _require_tensor(mhc.base, torch.float32, (mix,), device)
        _require_tensor(
            mhc.fn,
            torch.float32,
            (mix, expected.streams_out[1] * expected.streams_out[2]),
            device,
        )
        _require_tensor(mhc.scale, torch.float32, (MHC_STREAMS - 1,), device)


def _validate_prefill_layer(
    streams, token_ids, weights, workspace, query_heads=NUM_QUERY_HEADS
) -> None:
    tokens = streams.shape[0] if streams.ndim == 3 else -1
    expected = prefill_layer_workspace_shapes(
        weights.layer_id, tokens, query_heads=query_heads
    )
    kind = attention_kind(weights.layer_id)
    expected_types = (
        (DeepSeekV4ProjectionWeights, DeepSeekV4AttentionWorkspace)
        if kind == "swa"
        else (
            (DeepSeekV4C4AttentionWeights, DeepSeekV4C4AttentionWorkspace)
            if kind == "csa"
            else (DeepSeekV4C128AttentionWeights, DeepSeekV4AttentionWorkspace)
        )
    )
    if not isinstance(weights.attention, expected_types[0]) or not isinstance(
        workspace.attention, expected_types[1]
    ):
        raise TypeError("prefill attention weights and workspace disagree")
    device = streams.device
    _require_tensor(streams, torch.bfloat16, expected.streams_out, device)
    _require_tensor(token_ids, torch.int64, (tokens,), device)
    float32 = {"post", "comb"}
    for field in fields(expected):
        if field.name not in {"attention", "ffn"}:
            dtype = torch.float32 if field.name in float32 else torch.bfloat16
            _require_tensor(
                getattr(workspace, field.name),
                dtype,
                getattr(expected, field.name),
                device,
            )


def _validate_common(
    hidden, weights, batch, raw_slots, raw, workspace, query_heads, group
) -> None:
    size = hidden.shape[0]
    if size not in DECODE_BATCH_SIZES and size not in {
        batch_size * DSPARK_VERIFY_WIDTH for batch_size in DECODE_BATCH_SIZES
    }:
        raise ValueError(f"unsupported DeepSeek V4 hidden shape: {hidden.shape}")
    if raw.sink.data_ptr() != weights.sink.data_ptr():
        raise ValueError("raw attention must use the loaded sink")
    if query_heads == LOCAL_QUERY_HEADS and (
        group is None or torch.distributed.get_world_size(group) != TEP_SIZE
    ):
        raise ValueError("DeepSeek V4 attention requires a TP4 process group")
    if query_heads == NUM_QUERY_HEADS and group is not None:
        raise ValueError("replicated DeepSeek V4 attention must be rank-local")
    _require_tensor(hidden, torch.bfloat16, (size, HIDDEN_SIZE))
    _require_tensor(raw_slots, torch.int32, (size,))
    _require_tensor(batch.positions, torch.int32, (size,))
    _require_tensor(batch.cos_sin, torch.float32)
    expected = _attention_workspace_shapes(size, query_heads)
    fp8 = {"hidden_fp8", "q_residual_fp8", "output_fp8"}
    fp32 = {"hidden_fp32", "hidden_scale", "q_residual_scale", "output_scale"}
    for field in fields(expected):
        dtype = torch.float8_e4m3fn if field.name in fp8 else torch.bfloat16
        dtype = torch.float32 if field.name in fp32 else dtype
        _require_tensor(
            getattr(workspace, field.name),
            dtype,
            getattr(expected, field.name),
            hidden.device,
        )
    for tensor in (raw_slots, batch.positions, batch.cos_sin, raw.query):
        if tensor.device != hidden.device:
            raise ValueError("DeepSeek V4 attention tensors must share one CUDA device")
    if batch.cos_sin.ndim != 2 or batch.cos_sin.shape[1] != 64:
        raise ValueError("DeepSeek V4 cos/sin table has the wrong layout")


def _validate_c4_bindings(weights, main, index, query, workspace, query_heads) -> None:
    size = workspace.common.normalized.shape[0]
    if (
        main.ape.data_ptr() != weights.main_ape.data_ptr()
        or main.norm_weight.data_ptr() != weights.main_norm.data_ptr()
        or index.ape.data_ptr() != weights.index_ape.data_ptr()
        or index.norm_weight.data_ptr() != weights.index_norm.data_ptr()
    ):
        raise ValueError("C4 compressor must use the loaded APE/norm tensors")
    device = workspace.common.normalized.device
    expected = _c4_workspace_shapes(size, query_heads)
    _require_tensor(workspace.packed_q, torch.bfloat16, expected.packed_q, device)
    _require_tensor(
        workspace.index_weights, torch.bfloat16, (size, NUM_QUERY_HEADS), device
    )
    _require_tensor(weights.index_weights_t, torch.bfloat16, (4096, 64), device)
    _require_tensor(query.query, torch.int8, (size, 1, NUM_QUERY_HEADS, 64), device)
    _require_tensor(query.scale, torch.int32, (size, 1, NUM_QUERY_HEADS), device)
    _require_tensor(query.weights, torch.float32, (size, NUM_QUERY_HEADS), device)


@triton.jit
def _power2_quantize_packed_groups(
    input_ptr,
    output_ptr,
    scale_ptr,
    TOKENS: tl.constexpr,
    ALIGNED_TOKENS: tl.constexpr,
    GROUP_COUNT: tl.constexpr,
    NUM_PACKS: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    INPUT_ROW_STRIDE: tl.constexpr,
    INPUT_GROUP_STRIDE: tl.constexpr,
):
    program = tl.program_id(0)
    token = program // (GROUP_COUNT * NUM_PACKS)
    remainder = program - token * GROUP_COUNT * NUM_PACKS
    group = remainder // NUM_PACKS
    pack = remainder - group * NUM_PACKS
    offsets = tl.arange(0, 4 * BLOCK_SIZE)
    source = token * INPUT_ROW_STRIDE + group * INPUT_GROUP_STRIDE
    source += pack * 4 * BLOCK_SIZE + offsets
    values = tl.load(input_ptr + source).to(tl.float32)
    segment = offsets // BLOCK_SIZE
    maximum0 = tl.maximum(
        tl.max(tl.where(segment == 0, tl.abs(values), 0.0), axis=0), 1.0e-10
    )
    maximum1 = tl.maximum(
        tl.max(tl.where(segment == 1, tl.abs(values), 0.0), axis=0), 1.0e-10
    )
    maximum2 = tl.maximum(
        tl.max(tl.where(segment == 2, tl.abs(values), 0.0), axis=0), 1.0e-10
    )
    maximum3 = tl.maximum(
        tl.max(tl.where(segment == 3, tl.abs(values), 0.0), axis=0), 1.0e-10
    )
    exponent0 = tl.ceil(tl.log2(maximum0 / 448.0))
    exponent1 = tl.ceil(tl.log2(maximum1 / 448.0))
    exponent2 = tl.ceil(tl.log2(maximum2 / 448.0))
    exponent3 = tl.ceil(tl.log2(maximum3 / 448.0))
    scale = tl.where(
        segment == 0,
        tl.exp2(exponent0),
        tl.where(
            segment == 1,
            tl.exp2(exponent1),
            tl.where(segment == 2, tl.exp2(exponent2), tl.exp2(exponent3)),
        ),
    )
    quantized = tl.maximum(tl.minimum(values / scale, 448.0), -448.0)
    destination = (group * TOKENS + token) * WIDTH
    destination += pack * 4 * BLOCK_SIZE + offsets
    tl.store(output_ptr + destination, quantized)
    packed = (exponent0.to(tl.int32) + 127) | ((exponent1.to(tl.int32) + 127) << 8)
    packed |= (exponent2.to(tl.int32) + 127) << 16
    packed |= (exponent3.to(tl.int32) + 127) << 24
    scale_destination = (group * NUM_PACKS + pack) * ALIGNED_TOKENS + token
    tl.store(scale_ptr + scale_destination, packed)


@triton.jit
def _inverse_rope(
    output,
    positions,
    cos_sin,
    BATCH_STRIDE: tl.constexpr,
    HEAD_STRIDE: tl.constexpr,
    HEADS: tl.constexpr,
    HEAD_WIDTH: tl.constexpr,
    ROPE_WIDTH: tl.constexpr,
):
    row = tl.program_id(0)
    token = row // HEADS
    head = row - token * HEADS
    pair = tl.arange(0, ROPE_WIDTH // 2)
    base = token * BATCH_STRIDE + head * HEAD_STRIDE
    base += HEAD_WIDTH - ROPE_WIDTH + 2 * pair
    even = tl.load(output + base).to(tl.float32)
    odd = tl.load(output + base + 1).to(tl.float32)
    cs_base = cos_sin + tl.load(positions + token) * ROPE_WIDTH
    cosine = tl.load(cs_base + pair)
    sine = tl.load(cs_base + ROPE_WIDTH // 2 + pair)
    tl.store(output + base, even * cosine + odd * sine)
    tl.store(output + base + 1, odd * cosine - even * sine)

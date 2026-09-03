# SPDX-License-Identifier: MIT AND Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 LightSeek Foundation
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4 compressed-attention and routed-FFN core.

The compressor kernels are reduced from TokenSpeed commit
f17b03efc1728875c586d848f49da5905032e87c. See the adjacent license notice.

The attention seam excludes input projections, inverse output RoPE, output
projection, full prefill, and an end-to-end one-million-token capacity claim.
"""

import hashlib
import inspect
from dataclasses import fields
from functools import cache
from importlib.metadata import version
from pathlib import Path

import deep_gemm
import flash_mla
import flash_mla.cuda as flash_mla_cuda
import flashinfer
import torch
import triton
import triton.language as tl
from flashinfer.jit.topk import gen_topk_module
from flashinfer.topk import TopKTieBreak, get_topk_module
from triton.language.extra import libdevice

from infer.models.deepseek_v4_flash.megamoe import (
    DeepSeekV4MoEWeights,
    DeepSeekV4MoEWorkspace,
    moe_workspace_shapes,
    tep4_moe_workspace_shapes,
)
from infer.models.deepseek_v4_flash.model import (
    C4_FP4_INDEX_HISTORY_CANDIDATE,
    C4_FP8_MAIN_HISTORY_CANDIDATE,
    C128_FP8_MAIN_HISTORY_CANDIDATE,
    DECODE_BATCH_SIZES,
    DSPARK_VERIFY_WIDTH,
    FP8_BLOCK_SIZE,
    HEAD_DIM,
    HIDDEN_SIZE,
    INDEX_CONTEXT_CAPACITY,
    INDEX_TOP_K,
    LOCAL_QUERY_HEADS,
    MAX_CONTEXT_TOKENS,
    MAX_PREFILL_REQUESTS,
    NUM_QUERY_HEADS,
    NUM_ROUTED_EXPERTS,
    PREFILL_CHUNK_TOKENS,
    RAW_STATE_RING_TOKENS,
    ROUTER_DENOMINATOR_MIN,
    ROUTER_SCALE,
    SWA_CANDIDATE_ROW_BYTES,
    SWA_WINDOW_TOKENS,
    TEP_SIZE,
    TOP_K,
    TOPK_ROW_STATES_BYTES,
    VOCAB_SIZE,
    DeepSeekV4AttentionPool,
    DeepSeekV4Compression,
    DeepSeekV4CompressionBatch,
    DeepSeekV4DecodeWorkspace,
    DeepSeekV4IndexQuery,
    DeepSeekV4PrefillIndexWorkspace,
    DeepSeekV4PrefillMetadata,
    DeepSeekV4PrefillState,
    DeepSeekV4PrefillWorkspace,
    DeepSeekV4RawAttention,
)

FLASHINFER_PACKAGES = {
    "flashinfer-cubin": "0.6.18.dev20260819",
    "flashinfer-jit-cache": "0.6.18.dev20260819+cu129",
    "flashinfer-python": "0.6.18.dev20260819",
}
FLASHINFER_COMMIT = "61a6c651872a7d3f2f6dcc1ced61633d8f8ba3dd"
DEEP_GEMM_VERSION = "0.1.5.post3"
FLASH_MLA_COMMIT = "15f13e5030374295491c5ce31b02d7e63a7772c6"
TRITON_VERSION = "3.7.1"
TORCH_VERSION = "2.13.0+cu129"
CUDA_VERSION = "12.9"
NATIVE_BINARY_SHA256 = {
    "deep_gemm": "e882d88dacb62acc2973b35abb24e88915973e096b4725d68b4d0397a330a0d8",
    "flash_mla": "6fa8a4d34c6461a17725d972411434ac32d3248847ef63d5ce4507b59e3d94b2",
    "flashinfer_topk": "ac9d6901e5579539cf70725b183030e33188d7305f91daebd49d04093ba57c18",
}

_C4_PAGE = C4_FP8_MAIN_HISTORY_CANDIDATE.logical_view_shape[0]
_C128_PAGE = C128_FP8_MAIN_HISTORY_CANDIDATE.logical_view_shape[0]
_RAW_PAGE = 64
_C4_STATE_PAGE = 4
_C128_STATE_PAGE = 8
_MAIN_ROW_BYTES = C4_FP8_MAIN_HISTORY_CANDIDATE.logical_view_shape[-1]
_INDEX_ROW_BYTES = C4_FP4_INDEX_HISTORY_CANDIDATE.logical_view_shape[-1]
_RAW_PAGE_STRIDE = (_RAW_PAGE * SWA_CANDIDATE_ROW_BYTES + 575) // 576 * 576
_C4_MAIN_PAGE_STRIDE = C4_FP8_MAIN_HISTORY_CANDIDATE.physical_storage_bytes
_C4_INDEX_PAGE_STRIDE = C4_FP4_INDEX_HISTORY_CANDIDATE.physical_storage_bytes
_C128_MAIN_PAGE_STRIDE = C128_FP8_MAIN_HISTORY_CANDIDATE.physical_storage_bytes
_PREFILL_INDEX_ROWS = 512


class DeepSeekV4Ops:
    """Direct pinned DeepGEMM, FlashInfer, and FlashMLA implementation.

    A graph bucket owns two independent descriptor/workspace/metadata sets.
    FlashMLA allocates its output, so returned views belong to the graph pool.
    """

    def __init__(self) -> None:
        packages = {name: version(name) for name in FLASHINFER_PACKAGES}
        if packages != FLASHINFER_PACKAGES or (
            flashinfer.__version__ != FLASHINFER_PACKAGES["flashinfer-python"]
            or flashinfer.__git_commit__ != FLASHINFER_COMMIT
            or version("triton") != TRITON_VERSION
            or torch.__version__ != TORCH_VERSION
            or torch.version.cuda != CUDA_VERSION
        ):
            raise RuntimeError("DeepSeek V4 dependency identity mismatch")
        if version("sgl-deep-gemm").split("+", 1)[0] != DEEP_GEMM_VERSION:
            raise RuntimeError("DeepGEMM version mismatch")
        topk_spec = gen_topk_module()
        if not topk_spec.is_aot:
            raise RuntimeError("FlashInfer top-k AOT artifact is unavailable")
        native_hashes = _native_binary_hashes(str(topk_spec.get_library_path()))
        if native_hashes != NATIVE_BINARY_SHA256:
            raise RuntimeError(
                "DeepSeek V4 native binary identity mismatch: "
                f"expected={NATIVE_BINARY_SHA256}, actual={native_hashes}"
            )
        topk = get_topk_module()
        if not callable(getattr(topk, "radix_topk_page_table_transform", None)):
            raise TypeError("FlashInfer fused top-k page-table ABI is unavailable")
        for name in (
            "get_paged_mqa_logits_metadata_out",
            "fp8_fp4_paged_mqa_logits_out",
            "mega_moe_pre_dispatch",
            "fp8_fp4_mega_moe",
            "set_pdl",
        ):
            if not callable(getattr(deep_gemm, name, None)):
                raise TypeError(f"patched DeepGEMM lacks {name}")
        parameters = inspect.signature(flash_mla.flash_mla_with_kvcache).parameters
        if not {
            "extra_k_cache",
            "extra_indices_in_kvcache",
            "extra_topk_length",
        }.issubset(parameters):
            raise TypeError("FlashMLA two-pool ABI is unavailable")
        if not callable(getattr(flash_mla, "flash_mla_sparse_fwd", None)):
            raise TypeError("FlashMLA sparse-prefill ABI is unavailable")
        self._topk = topk
        self._tie_break = int(TopKTieBreak.SMALL)
        self._num_sms = deep_gemm.get_num_sms()
        self._moe_overlap: tuple[object, object, object] | None = None
        deep_gemm.set_pdl(True)

    def prefill_compress_c4(
        self,
        batch: DeepSeekV4CompressionBatch[torch.Tensor],
        main: DeepSeekV4Compression[torch.Tensor],
        index: DeepSeekV4Compression[torch.Tensor],
        main_state: DeepSeekV4PrefillState[torch.Tensor],
        index_state: DeepSeekV4PrefillState[torch.Tensor],
    ) -> None:
        """Compress a chunk from caller-owned prefix-tail plus chunk state."""
        size = _validate_prefill_batch(batch)
        _validate_compression(main, size, 4, 1_024, 512, _C4_MAIN_PAGE_STRIDE)
        _validate_compression(index, size, 4, 256, 128, _C4_INDEX_PAGE_STRIDE)
        _seed_prefill_state(main, main_state)
        _seed_prefill_state(index, index_state)
        _run_c4_compression(batch, main, index)
        _retain_prefill_state(main, main_state)
        _retain_prefill_state(index, index_state)

    def prefill_compress_c128(
        self,
        batch: DeepSeekV4CompressionBatch[torch.Tensor],
        main: DeepSeekV4Compression[torch.Tensor],
        state: DeepSeekV4PrefillState[torch.Tensor],
    ) -> None:
        """Compress a chunk from caller-owned prefix-tail plus chunk state."""
        size = _validate_prefill_batch(batch)
        _validate_compression(main, size, 128, 512, 512, _C128_MAIN_PAGE_STRIDE)
        _seed_prefill_state(main, state)
        _run_c128_compression(batch, main)
        _retain_prefill_state(main, state)

    def prefill_c4_candidates(
        self,
        metadata: DeepSeekV4PrefillMetadata[torch.Tensor],
        batch: DeepSeekV4CompressionBatch[torch.Tensor],
        index_cache: torch.Tensor,
        projected: torch.Tensor,
        score_weights: torch.Tensor,
        workspace: DeepSeekV4PrefillIndexWorkspace[torch.Tensor],
    ) -> torch.Tensor:
        size, capacity, local_heads = _validate_prefill_c4_candidates(
            metadata,
            batch,
            index_cache,
            projected,
            score_weights,
            workspace,
        )
        staged = workspace.staged
        cache = index_cache.view(-1, _C4_PAGE, 1, _INDEX_ROW_BYTES)
        for start in range(0, size, _PREFILL_INDEX_ROWS):
            count = min(size - start, _PREFILL_INDEX_ROWS)
            _c4_query[(count * NUM_QUERY_HEADS, 4)](
                projected[start:],
                score_weights[start:],
                batch.positions[start:],
                batch.cos_sin,
                staged.query,
                staged.scale,
                staged.weights,
                HEADS=NUM_QUERY_HEADS,
                LOCAL_HEADS=local_heads,
                HEAD_WIDTH=128,
                PACKED_WIDTH=projected.shape[1],
                ROPE_WIDTH=64,
                WEIGHT_SCALE=(128 * NUM_QUERY_HEADS) ** -0.5,
                num_warps=4,
            )
            _stage_prefill_index_kernel[(_PREFILL_INDEX_ROWS, 4)](
                batch.positions,
                batch.request_indices,
                metadata.block_table,
                metadata.block_table_base,
                staged.block_table,
                staged.lengths,
                workspace.request_indices,
                start,
                count,
                metadata.block_table.shape[1],
                num_warps=4,
            )
            deep_gemm.get_paged_mqa_logits_metadata_out(
                staged.lengths,
                workspace.schedule,
                _C4_PAGE,
                self._num_sms,
                indices=workspace.request_indices,
            )
            deep_gemm.fp8_fp4_paged_mqa_logits_out(
                (staged.query, staged.scale),
                cache,
                staged.weights,
                staged.lengths,
                staged.block_table,
                workspace.schedule,
                workspace.logits,
                capacity,
                clean_logits=False,
                indices=workspace.request_indices,
            )
            self._topk.radix_topk_ragged_transform(
                workspace.logits,
                workspace.candidates[start : start + _PREFILL_INDEX_ROWS],
                workspace.topk_offsets,
                staged.lengths.view(-1),
                workspace.topk_rows,
                INDEX_TOP_K,
                True,
                self._tie_break,
                dsa_graph_safe=True,
                row_starts=None,
            )
        return workspace.candidates[:size]

    def stage_prefill_attention(
        self,
        metadata: DeepSeekV4PrefillMetadata[torch.Tensor],
        batch: DeepSeekV4CompressionBatch[torch.Tensor],
        current_kv: torch.Tensor,
        raw: DeepSeekV4RawAttention[torch.Tensor],
        compressed_cache: torch.Tensor,
        local_candidates: torch.Tensor | None,
        workspace: DeepSeekV4PrefillWorkspace[torch.Tensor],
        ratio: int,
    ) -> None:
        compressed_capacity = _validate_prefill_staging(
            metadata,
            batch,
            current_kv,
            raw,
            compressed_cache,
            local_candidates,
            workspace,
            ratio,
        )
        rows = workspace.kv.shape[0]
        _stage_prefill_attention_kernel[(min(rows, 2_048), 4)](
            current_kv,
            batch.positions,
            batch.request_indices,
            batch.cos_sin,
            metadata.query_start,
            metadata.prefix_lengths,
            metadata.state_slots,
            metadata.block_table,
            metadata.block_table_base,
            raw.cache,
            compressed_cache,
            local_candidates if local_candidates is not None else raw.indices,
            workspace.kv,
            raw.indices,
            raw.lengths,
            workspace.raw_slots,
            current_kv.shape[0],
            metadata.prefix_lengths.shape[0],
            compressed_capacity,
            raw.indices.stride(0),
            metadata.block_table.shape[1],
            rows,
            compressed_cache.stride(0),
            raw.cache.stride(0),
            RATIO=ratio,
            COMPRESSED_PAGE=_C4_PAGE if ratio == 4 else _C128_PAGE,
            RAW_RING_PAGES=RAW_STATE_RING_TOKENS // _RAW_PAGE,
            num_warps=4,
        )

    def prefill_selected_attention(
        self,
        query: torch.Tensor,
        kv: torch.Tensor,
        indices: torch.Tensor,
        lengths: torch.Tensor,
        sink: torch.Tensor,
    ) -> torch.Tensor:
        _validate_prefill_attention(query, kv, indices, lengths, sink)
        output, _, _ = flash_mla.flash_mla_sparse_fwd(
            q=query,
            kv=kv,
            indices=indices,
            sm_scale=HEAD_DIM**-0.5,
            d_v=HEAD_DIM,
            attn_sink=sink,
            topk_length=lengths,
        )
        return output

    def _decode_moe(
        self,
        hidden: torch.Tensor,
        token_ids: torch.Tensor,
        weights: DeepSeekV4MoEWeights[torch.Tensor],
        workspace: DeepSeekV4MoEWorkspace[torch.Tensor],
        mega_buffer: object,
        hash_routing: bool,
        process_group: object | None = None,
    ) -> torch.Tensor:
        _validate_moe(
            hidden,
            token_ids,
            weights.routing,
            workspace,
            mega_buffer,
            hash_routing,
            process_group,
        )
        w = workspace
        torch.mm(
            hidden,
            weights.router_t,
            out=w.router_logits,
            out_dtype=torch.float32,
        )
        if hash_routing:
            _route_hash(w.router_logits, token_ids, weights.routing, w)
        else:
            _route_learned(w.router_logits, weights.routing, w)

        overlap = self._moe_overlap
        if overlap is None:
            overlap = (torch.cuda.Stream(), torch.cuda.Event(), torch.cuda.Event())
            self._moe_overlap = overlap
        current_stream = torch.cuda.current_stream()
        shared_stream, fork_event, join_event = overlap
        fork_event.record(current_stream)
        shared_stream.wait_event(fork_event)

        _run_shared_expert(hidden, weights, w, process_group)

        with torch.cuda.stream(shared_stream):
            predispatch_result = deep_gemm.mega_moe_pre_dispatch(
                hidden,
                w.topk_ids,
                w.topk_weights,
                mega_buffer.x.view(torch.float8_e4m3fn),
                mega_buffer.x_sf,
                mega_buffer.topk_idx,
                mega_buffer.topk_weights,
                num_tokens=hidden.shape[0],
                group_size=32,
                use_fp4_acts=False,
            )
            moe_result = deep_gemm.fp8_fp4_mega_moe(
                w.routed,
                weights.routed.l1,
                weights.routed.l2,
                mega_buffer,
                recipe=(1, 1, 32),
                activation="swiglu",
                activation_clamp=10.0,
                fast_math=True,
            )
            join_event.record(shared_stream)
        if predispatch_result is not None or moe_result is not None:
            raise RuntimeError("caller-owned MegaMoE operations must return None")

        current_stream.wait_event(join_event)
        w.output.add_(w.routed)
        return w.output

    def validate_c4(
        self,
        batch: DeepSeekV4CompressionBatch[torch.Tensor],
        main: DeepSeekV4Compression[torch.Tensor],
        index: DeepSeekV4Compression[torch.Tensor],
        raw: DeepSeekV4RawAttention[torch.Tensor],
        query: DeepSeekV4IndexQuery[torch.Tensor],
        workspace: DeepSeekV4DecodeWorkspace[torch.Tensor],
        output_heads: int,
    ) -> None:
        if output_heads not in (LOCAL_QUERY_HEADS, NUM_QUERY_HEADS):
            raise ValueError("C4 output heads must be local or replicated")
        batch_size = _validate_batch(batch, raw)
        _validate_compression(main, batch_size, 4, 1_024, 512, _C4_MAIN_PAGE_STRIDE)
        _validate_compression(index, batch_size, 4, 256, 128, _C4_INDEX_PAGE_STRIDE)
        _validate_c4(query, workspace, batch_size, self._num_sms)

    def compress_c4_main(
        self,
        batch: DeepSeekV4CompressionBatch[torch.Tensor],
        main: DeepSeekV4Compression[torch.Tensor],
    ) -> None:
        _run_c4_main(batch, main)

    def compress_c4_index(
        self,
        batch: DeepSeekV4CompressionBatch[torch.Tensor],
        index: DeepSeekV4Compression[torch.Tensor],
    ) -> None:
        _run_c4_index(batch, index)

    def prepare_c4(
        self,
        query: DeepSeekV4IndexQuery[torch.Tensor],
        workspace: DeepSeekV4DecodeWorkspace[torch.Tensor],
    ) -> None:
        if (
            deep_gemm.get_paged_mqa_logits_metadata_out(
                query.lengths, workspace.schedule, _C4_PAGE, self._num_sms, indices=None
            )
            is not None
        ):
            raise RuntimeError("DeepGEMM metadata out-ABI returned a value")
        torch.clamp(
            query.lengths.view(-1),
            min=0,
            max=INDEX_TOP_K,
            out=workspace.selected_lengths,
        )

    def select_c4(
        self,
        index: DeepSeekV4Compression[torch.Tensor],
        query: DeepSeekV4IndexQuery[torch.Tensor],
        workspace: DeepSeekV4DecodeWorkspace[torch.Tensor],
    ) -> None:
        if (
            deep_gemm.fp8_fp4_paged_mqa_logits_out(
                (query.query, query.scale),
                index.cache.view(-1, _C4_PAGE, 1, _INDEX_ROW_BYTES),
                query.weights,
                query.lengths,
                query.block_table,
                workspace.schedule,
                workspace.logits,
                INDEX_CONTEXT_CAPACITY,
                clean_logits=False,
                indices=None,
            )
            is not None
        ):
            raise RuntimeError("DeepGEMM logits out-ABI returned a value")
        batch_size = query.query.shape[0]
        self._topk.radix_topk_page_table_transform(
            workspace.logits,
            workspace.mapped_c4.view(batch_size, -1),
            query.block_table,
            None,
            query.lengths.view(-1),
            workspace.topk_rows,
            INDEX_TOP_K,
            True,
            self._tie_break,
            page_size=_C4_PAGE,
            dsa_graph_safe=True,
            row_starts=None,
            page_table_row_starts=None,
            output_raw_indices=None,
        )

    def attend_c4(
        self,
        raw: DeepSeekV4RawAttention[torch.Tensor],
        main: DeepSeekV4Compression[torch.Tensor],
        workspace: DeepSeekV4DecodeWorkspace[torch.Tensor],
        metadata: object,
        output_heads: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _attention(
            raw,
            main.cache,
            workspace.mapped_c4,
            workspace.selected_lengths,
            metadata,
            _C4_PAGE,
            output_heads,
        )

    def decode_swa(
        self,
        batch: DeepSeekV4CompressionBatch[torch.Tensor],
        raw: DeepSeekV4RawAttention[torch.Tensor],
        metadata: object,
        output_heads: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if output_heads not in (LOCAL_QUERY_HEADS, NUM_QUERY_HEADS):
            raise ValueError("SWA output heads must be local or replicated")
        _validate_batch(batch, raw)
        return _attention(raw, None, None, None, metadata, _RAW_PAGE, output_heads)

    def decode_c128(
        self,
        batch: DeepSeekV4CompressionBatch[torch.Tensor],
        main: DeepSeekV4Compression[torch.Tensor],
        raw: DeepSeekV4RawAttention[torch.Tensor],
        pool: DeepSeekV4AttentionPool[torch.Tensor],
        metadata: object,
        output_heads: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if output_heads not in (LOCAL_QUERY_HEADS, NUM_QUERY_HEADS):
            raise ValueError("C128 output heads must be local or replicated")
        batch_size = _validate_batch(batch, raw)
        _validate_compression(main, batch_size, 128, 512, 512, _C128_MAIN_PAGE_STRIDE)
        if (
            pool.indices.ndim != 3
            or pool.indices.shape[:2]
            != (
                batch_size,
                1,
            )
            or pool.lengths.shape != (batch_size,)
        ):
            raise ValueError("C128 pool does not match the decode bucket")
        _require_cuda_types(
            (pool.indices, torch.int32),
            (pool.lengths, torch.int32),
        )
        _run_c128_compression(batch, main)
        return _attention(
            raw,
            main.cache,
            pool.indices,
            pool.lengths,
            metadata,
            _C128_PAGE,
            output_heads,
        )


@cache
def _native_binary_hashes(topk_library: str) -> dict[str, str]:
    paths = {
        "deep_gemm": Path(deep_gemm.__file__).resolve().parent / "_C.so",
        "flash_mla": Path(flash_mla_cuda.__file__).resolve(),
        "flashinfer_topk": Path(topk_library).resolve(),
    }
    hashes = {}
    for name, path in paths.items():
        with path.open("rb") as binary:
            hashes[name] = hashlib.file_digest(binary, "sha256").hexdigest()
    return hashes


def _route_hash(router_logits, token_ids, hash_routes, workspace) -> None:
    if router_logits.shape[0] == 0:
        return
    _hash_route[(router_logits.shape[0],)](
        router_logits,
        token_ids,
        hash_routes,
        workspace.topk_ids,
        workspace.topk_weights,
        NUM_EXPERTS=NUM_ROUTED_EXPERTS,
        TOP_K=TOP_K,
        ROUTE_SCALE=ROUTER_SCALE,
        DENOMINATOR_MIN=ROUTER_DENOMINATOR_MIN,
    )


def _route_learned(router_logits, route_bias, workspace) -> None:
    if router_logits.shape[0] == 0:
        return
    _learned_route[(router_logits.shape[0],)](
        router_logits,
        route_bias,
        workspace.topk_ids,
        workspace.topk_weights,
        NUM_EXPERTS=NUM_ROUTED_EXPERTS,
        TOP_K=TOP_K,
        BLOCK_K=8,
        ROUTE_SCALE=ROUTER_SCALE,
        DENOMINATOR_MIN=ROUTER_DENOMINATOR_MIN,
    )


def _run_shared_expert(hidden, weights, workspace, process_group) -> None:
    if hidden.shape[0] == 0:
        return
    _quantize(hidden, workspace.hidden_fp8, workspace.hidden_scale)
    _fp8_gemm(
        workspace.hidden_fp8,
        weights.shared_gate_up,
        workspace.hidden_scale,
        weights.shared_gate_up_scale,
        workspace.shared_gate_up,
    )
    shared = weights.shared_down.shape[1]
    blocks = shared // FP8_BLOCK_SIZE
    _shared_swiglu_quantize[(hidden.shape[0] * blocks,)](
        workspace.shared_gate_up,
        workspace.shared_activated_fp8,
        workspace.shared_activated_scale,
        ROWS=hidden.shape[0],
        WIDTH=shared,
        BLOCKS=blocks,
        BLOCK_SIZE=FP8_BLOCK_SIZE,
        LIMIT=10.0,
        FP8_LIMIT=448.0,
    )
    _fp8_gemm(
        workspace.shared_activated_fp8,
        weights.shared_down,
        workspace.shared_activated_scale,
        weights.shared_down_scale,
        workspace.output,
    )
    if process_group is not None:
        torch.distributed.all_reduce(workspace.output, group=process_group)


def _quantize(input_: torch.Tensor, output: torch.Tensor, scale: torch.Tensor) -> None:
    blocks = input_.shape[1] // FP8_BLOCK_SIZE
    _power2_quantize[(input_.shape[0] * blocks,)](
        input_,
        output,
        scale,
        INPUT_ROW_STRIDE=input_.stride(0),
        ROWS=input_.shape[0],
        BLOCKS=blocks,
        BLOCK_SIZE=FP8_BLOCK_SIZE,
        FP8_LIMIT=448.0,
    )


def _copy_c4_raw_query(packed: torch.Tensor, dense: torch.Tensor) -> None:
    dense = dense.view(dense.shape[0], -1)
    dense.copy_(packed[:, : dense.shape[1]])


def _fp8_gemm(input_, weight, input_scale, weight_scale, out) -> None:
    from flashinfer.gemm import gemm_fp8_nt_groupwise

    _require_tensor(
        weight, torch.float8_e4m3fn, (out.shape[1], input_.shape[1]), input_.device
    )
    _require_tensor(
        weight_scale,
        torch.float32,
        (out.shape[1] // FP8_BLOCK_SIZE, input_.shape[1] // FP8_BLOCK_SIZE),
        input_.device,
    )
    gemm_fp8_nt_groupwise(
        input_,
        weight,
        input_scale.view(input_scale.shape[1], input_scale.shape[0]).T,
        weight_scale.T,
        scale_major_mode=None,
        scale_granularity_mnk=(1, FP8_BLOCK_SIZE, FP8_BLOCK_SIZE),
        out=out,
        backend="trtllm",
    )


def _validate_moe(
    hidden, token_ids, routing, workspace, mega_buffer, hash_routing, process_group
):
    tokens = hidden.shape[0]
    device = hidden.device
    _require_tensor(hidden, torch.bfloat16, (tokens, HIDDEN_SIZE))
    _require_tensor(token_ids, torch.int64, (tokens,), device)

    routing_dtype = torch.int32 if hash_routing else torch.float32
    routing_shape = (VOCAB_SIZE, TOP_K) if hash_routing else (NUM_ROUTED_EXPERTS,)
    _require_tensor(routing, routing_dtype, routing_shape, device)

    if (
        process_group is not None
        and torch.distributed.get_world_size(process_group) != TEP_SIZE
    ):
        raise ValueError("DeepSeek V4 shared expert requires a TP4 process group")
    expected = (
        tep4_moe_workspace_shapes(tokens)
        if process_group is not None
        else moe_workspace_shapes(tokens)
    )
    fp8 = {"hidden_fp8", "shared_activated_fp8"}
    bf16 = {"shared_gate_up", "routed", "output"}
    int32 = {"topk_ids"}
    for field in fields(expected):
        dtype = (
            torch.float8_e4m3fn
            if field.name in fp8
            else torch.bfloat16
            if field.name in bf16
            else torch.int32
            if field.name in int32
            else torch.float32
        )
        _require_tensor(
            getattr(workspace, field.name),
            dtype,
            getattr(expected, field.name),
            device,
        )

    if mega_buffer.num_max_tokens_per_rank < tokens:
        raise ValueError("MegaMoE buffer is smaller than the decode lane")


def _require_tensor(tensor, dtype, shape=None, device=None) -> None:
    if not tensor.is_contiguous() or (shape is not None and tensor.shape != shape):
        raise ValueError("DeepSeek V4 tensor has the wrong layout")
    if (
        not tensor.is_cuda
        or tensor.dtype != dtype
        or (device is not None and tensor.device != device)
    ):
        raise TypeError(f"expected CUDA {dtype}, got {tensor.device} {tensor.dtype}")


def _attention(
    raw,
    extra_cache,
    extra_indices,
    extra_lengths,
    metadata,
    page_size,
    output_heads,
):
    def cache_view(cache, page):
        used = page * _MAIN_ROW_BYTES
        if cache.ndim != 2 or used > cache.shape[1]:
            raise ValueError("attention cache is smaller than its logical page")
        return torch.as_strided(
            cache,
            (cache.shape[0], page, 1, _MAIN_ROW_BYTES),
            (cache.stride(0), _MAIN_ROW_BYTES, _MAIN_ROW_BYTES, 1),
        )

    output, lse = flash_mla.flash_mla_with_kvcache(
        q=raw.query,
        k_cache=cache_view(raw.cache, _RAW_PAGE),
        block_table=None,
        cache_seqlens=None,
        head_dim_v=HEAD_DIM,
        tile_scheduler_metadata=metadata,
        num_splits=None,
        softmax_scale=HEAD_DIM**-0.5,
        causal=False,
        is_fp8_kvcache=True,
        indices=raw.indices,
        attn_sink=raw.sink,
        extra_k_cache=(
            None if extra_cache is None else cache_view(extra_cache, page_size)
        ),
        extra_indices_in_kvcache=extra_indices,
        topk_length=raw.lengths,
        extra_topk_length=extra_lengths,
    )
    return output[:, :, :output_heads], lse[:, :output_heads]


def _validate_batch(batch, raw) -> int:
    size = raw.query.shape[0]
    verify_sizes = {
        batch_size * DSPARK_VERIFY_WIDTH for batch_size in DECODE_BATCH_SIZES
    }
    if (
        size not in DECODE_BATCH_SIZES and size not in verify_sizes
    ) or raw.query.shape != (
        size,
        1,
        NUM_QUERY_HEADS,
        HEAD_DIM,
    ):
        raise ValueError(f"unsupported decode query shape: {raw.query.shape}")
    if (
        batch.positions.shape != (size,)
        or batch.request_indices.shape != (size,)
        or batch.state_slots.shape != (size,)
        or batch.output_slots.shape != (size,)
        or batch.state_table.ndim != 2
        or batch.state_table.shape[0] != size
        or batch.state_table_base.shape != (size,)
        or batch.cos_sin.ndim != 2
        or batch.cos_sin.shape[1] != 64
        or raw.lengths.shape != (size,)
        or raw.indices.shape != (size, 1, 128)
        or raw.sink.shape != (NUM_QUERY_HEADS,)
        or raw.cache.shape[1:] != (_RAW_PAGE_STRIDE,)
    ):
        raise ValueError("decode descriptors have the wrong layout")
    _require_cuda_types(
        (batch.positions, torch.int32),
        (batch.request_indices, torch.int32),
        (batch.state_slots, torch.int32),
        (batch.output_slots, torch.int32),
        (batch.state_table, torch.int32),
        (batch.state_table_base, torch.int32),
        (batch.cos_sin, torch.float32),
        (raw.query, torch.bfloat16),
        (raw.cache, torch.uint8),
        (raw.indices, torch.int32),
        (raw.lengths, torch.int32),
        (raw.sink, torch.float32),
    )
    return size


def _validate_prefill_batch(batch) -> int:
    size = batch.positions.numel()
    table_rows = batch.state_table.shape[0] if batch.state_table.ndim == 2 else 0
    if (
        size <= 0
        or size > PREFILL_CHUNK_TOKENS
        or batch.positions.shape != (size,)
        or batch.request_indices.shape != (size,)
        or batch.state_slots.shape != (size,)
        or batch.output_slots.shape != (size,)
        or table_rows <= 0
        or table_rows > size
        or batch.state_table_base.shape != (table_rows,)
        or batch.cos_sin.ndim != 2
        or batch.cos_sin.shape[1] != 64
    ):
        raise ValueError("prefill compressor descriptors have the wrong layout")
    _require_cuda_types(
        (batch.positions, torch.int32),
        (batch.request_indices, torch.int32),
        (batch.state_slots, torch.int32),
        (batch.output_slots, torch.int32),
        (batch.state_table, torch.int32),
        (batch.state_table_base, torch.int32),
        (batch.cos_sin, torch.float32),
    )
    return size


def _validate_prefill_attention(query, kv, indices, lengths, sink) -> None:
    size = query.shape[0]
    selected_width = indices.shape[2] if indices.ndim == 3 else 0
    if (
        size <= 0
        or size > PREFILL_CHUNK_TOKENS
        or query.shape != (size, NUM_QUERY_HEADS, HEAD_DIM)
        or kv.ndim != 3
        or kv.shape[0] <= 0
        or kv.shape[1:] != (1, HEAD_DIM)
        or indices.shape != (size, 1, selected_width)
        or selected_width <= 0
        or selected_width % 128
        or lengths.shape != (size,)
        or sink.shape != (NUM_QUERY_HEADS,)
    ):
        raise ValueError("sparse prefill attention tensors have the wrong layout")
    _require_cuda_types(
        (query, torch.bfloat16),
        (kv, torch.bfloat16),
        (indices, torch.int32),
        (lengths, torch.int32),
        (sink, torch.float32),
    )


def _validate_prefill_metadata(metadata) -> int:
    requests = metadata.prefix_lengths.numel()
    if (
        requests <= 0
        or requests > MAX_PREFILL_REQUESTS
        or metadata.query_start.shape != (requests + 1,)
        or metadata.state_slots.shape != (requests,)
        or metadata.block_table.ndim != 2
        or metadata.block_table.shape[0] != requests
        or metadata.block_table.shape[1] <= 0
        or metadata.block_table_base.shape != (requests,)
    ):
        raise ValueError("prefill metadata has the wrong layout")
    _require_cuda_types(
        (metadata.query_start, torch.int32),
        (metadata.prefix_lengths, torch.int32),
        (metadata.state_slots, torch.int32),
        (metadata.block_table, torch.int32),
        (metadata.block_table_base, torch.int32),
    )
    return requests


def _validate_prefill_c4_candidates(
    metadata,
    batch,
    index_cache,
    projected,
    score_weights,
    workspace,
) -> tuple[int, int, int]:
    size = _validate_prefill_batch(batch)
    _validate_prefill_metadata(metadata)
    capacity = metadata.block_table.shape[1] * _C4_PAGE
    projected_width = projected.shape[1] if projected.ndim == 2 else 0
    local_heads = (projected_width - NUM_QUERY_HEADS * 128) // HEAD_DIM
    staged = workspace.staged
    logit_capacity = (capacity + 255) // 256 * 256
    if (
        capacity > INDEX_CONTEXT_CAPACITY
        or local_heads not in (LOCAL_QUERY_HEADS, NUM_QUERY_HEADS)
        or index_cache.ndim != 2
        or index_cache.shape[0] <= 0
        or workspace.logits.shape != (_PREFILL_INDEX_ROWS, logit_capacity)
        or not workspace.logits.is_contiguous()
    ):
        raise ValueError("C4 prefill indexer tensors have the wrong layout")
    specs = (
        (index_cache, torch.uint8, (index_cache.shape[0], _C4_INDEX_PAGE_STRIDE)),
        (
            projected,
            torch.bfloat16,
            (size, local_heads * HEAD_DIM + NUM_QUERY_HEADS * 128),
        ),
        (score_weights, torch.bfloat16, (size, NUM_QUERY_HEADS)),
        (staged.query, torch.int8, (_PREFILL_INDEX_ROWS, 1, NUM_QUERY_HEADS, 64)),
        (staged.scale, torch.int32, (_PREFILL_INDEX_ROWS, 1, NUM_QUERY_HEADS)),
        (staged.weights, torch.float32, (_PREFILL_INDEX_ROWS, NUM_QUERY_HEADS)),
        (
            staged.block_table,
            torch.int32,
            (_PREFILL_INDEX_ROWS, metadata.block_table.shape[1]),
        ),
        (staged.lengths, torch.int32, (_PREFILL_INDEX_ROWS, 1)),
        (workspace.request_indices, torch.int32, (_PREFILL_INDEX_ROWS,)),
    )
    for tensor, dtype, shape in specs:
        _require_tensor(tensor, dtype, shape, projected.device)
    return size, logit_capacity, local_heads


def _validate_prefill_staging(
    metadata,
    batch,
    current_kv,
    raw,
    compressed_cache,
    local_candidates,
    workspace,
    ratio,
) -> int:
    size = _validate_prefill_batch(batch)
    requests = _validate_prefill_metadata(metadata)
    raw_rows = requests * SWA_WINDOW_TOKENS + size
    remaining = workspace.kv.shape[0] - raw_rows
    compressed_capacity = remaining // requests if remaining >= 0 else -1
    selected_width = (
        INDEX_TOP_K + SWA_WINDOW_TOKENS
        if ratio == 4
        else (compressed_capacity + SWA_WINDOW_TOKENS)
    )
    selected_width = (selected_width + 127) // 128 * 128
    compressed_page = _C4_PAGE if ratio == 4 else _C128_PAGE
    capacity_limit = 0 if ratio == 1 else MAX_CONTEXT_TOKENS // ratio
    expected_capacity = (
        0 if ratio == 1 else metadata.block_table.shape[1] * compressed_page
    )
    expected_stride = (
        _C4_MAIN_PAGE_STRIDE
        if ratio == 4
        else _C128_MAIN_PAGE_STRIDE
        if ratio == 128
        else _RAW_PAGE_STRIDE
    )
    if (
        ratio not in (1, 4, 128)
        or current_kv.shape != (size, HEAD_DIM)
        or raw.cache.ndim != 2
        or raw.cache.shape[1] != _RAW_PAGE_STRIDE
        or compressed_cache.ndim != 2
        or compressed_cache.shape[1] != expected_stride
        or remaining < 0
        or remaining % requests
        or compressed_capacity != expected_capacity
        or compressed_capacity > capacity_limit
        or workspace.kv.shape[1:] != (1, HEAD_DIM)
        or workspace.raw_slots.shape != (size,)
        or raw.query.shape != (size, 1, NUM_QUERY_HEADS, HEAD_DIM)
        or raw.indices.shape != (size, 1, selected_width)
        or (
            ratio == 4
            and (
                local_candidates is None
                or local_candidates.shape != (size, INDEX_TOP_K)
            )
        )
        or (ratio != 4 and local_candidates is not None)
    ):
        raise ValueError("prefill staging tensors have the wrong layout")
    _validate_prefill_attention(
        raw.query[:, 0], workspace.kv, raw.indices, raw.lengths, raw.sink
    )
    _require_cuda_types(
        (current_kv, torch.bfloat16),
        (raw.cache, torch.uint8),
        (compressed_cache, torch.uint8),
        (workspace.raw_slots, torch.int32),
    )
    if local_candidates is not None:
        _require_cuda_types((local_candidates, torch.int32))
    return compressed_capacity


def _validate_compression(
    compression, size, ratio, width, norm_width, cache_stride
) -> None:
    state_page = _C4_STATE_PAGE if ratio == 4 else _C128_STATE_PAGE
    shapes = (
        (compression.kv_score.shape, (size, 2 * width)),
        (compression.ape.shape, (ratio, width)),
        (compression.state.shape[1:], (state_page, 2 * width)),
        (compression.norm_weight.shape, (norm_width,)),
        (compression.cache.shape[1:], (cache_stride,)),
    )
    if any(actual != expected for actual, expected in shapes):
        raise ValueError("compressor tensors have the wrong fixed layout")
    _require_cuda_types(
        (compression.kv_score, torch.float32),
        (compression.ape, torch.float32),
        (compression.state, torch.float32),
        (compression.norm_weight, torch.bfloat16),
        (compression.cache, torch.uint8),
    )


def _validate_c4(query, workspace, size: int, num_sms: int) -> None:
    shapes = (
        (query.query.shape, (size, 1, NUM_QUERY_HEADS, 64)),
        (query.scale.shape, (size, 1, NUM_QUERY_HEADS)),
        (query.weights.shape, (size, NUM_QUERY_HEADS)),
        (query.block_table.ndim, 2),
        (query.block_table.shape[:1], (size,)),
        (query.lengths.shape, (size, 1)),
        (workspace.schedule.shape, (num_sms + 1, 2)),
        (workspace.logits.shape, (size, INDEX_CONTEXT_CAPACITY)),
        (workspace.mapped_c4.shape, (size, 1, INDEX_TOP_K)),
        (workspace.selected_lengths.shape, (size,)),
        (workspace.topk_rows.numel(), TOPK_ROW_STATES_BYTES),
    )
    if any(actual != expected for actual, expected in shapes):
        raise ValueError("C4 tensors do not match the fixed decode layout")
    if workspace.schedule.storage_offset() or workspace.schedule.data_ptr() % 8:
        raise ValueError("DeepGEMM schedule must be an aligned base allocation")
    _require_cuda_types(
        (query.query, torch.int8),
        (query.scale, torch.int32),
        (query.weights, torch.float32),
        (query.block_table, torch.int32),
        (query.lengths, torch.int32),
        (workspace.schedule, torch.int32),
        (workspace.logits, torch.float32),
        (workspace.topk_rows, torch.uint8),
        (workspace.mapped_c4, torch.int32),
        (workspace.selected_lengths, torch.int32),
    )


def _require_cuda_types(*specifications) -> None:
    for tensor, dtype in specifications:
        if not tensor.is_contiguous():
            raise ValueError("DeepSeek V4 decode tensors must be contiguous")
        if not tensor.is_cuda or tensor.dtype != dtype:
            raise TypeError(
                f"expected CUDA {dtype}, got {tensor.device} {tensor.dtype}"
            )


def _seed_prefill_state(compression, state) -> None:
    _validate_prefill_state(compression, state)
    _copy_prefill_state(
        state.persistent,
        compression.state,
        state.seed_source,
        state.seed_destination,
        state.transfer,
    )


def _retain_prefill_state(compression, state) -> None:
    _copy_prefill_state(
        compression.state,
        state.persistent,
        state.retain_source,
        state.retain_destination,
        state.transfer,
    )


def _copy_prefill_state(source, destination, source_rows, destination_rows, transfer):
    if not source_rows.numel():
        return
    rows = source_rows.shape[0]
    selected = transfer[:rows]
    torch.index_select(source, 0, source_rows, out=selected)
    destination.index_copy_(0, destination_rows, selected)


def _validate_prefill_state(compression, state) -> None:
    scratch = compression.state
    rows = max(state.seed_source.numel(), state.retain_source.numel())
    if (
        scratch.ndim != 3
        or state.persistent.ndim != 3
        or scratch.shape[1:] != state.persistent.shape[1:]
        or state.transfer.shape != (rows, *scratch.shape[1:])
        or state.seed_source.shape != state.seed_destination.shape
        or state.retain_source.shape != state.retain_destination.shape
    ):
        raise ValueError("prefill compressor state tensors have the wrong layout")
    _require_cuda_types(
        (scratch, torch.float32),
        (state.persistent, torch.float32),
        (state.seed_source, torch.int64),
        (state.seed_destination, torch.int64),
        (state.retain_source, torch.int64),
        (state.retain_destination, torch.int64),
        (state.transfer, torch.float32),
    )


def _save(batch, compression, ratio, overlap, state_page) -> None:
    width = compression.kv_score.shape[1] // 2
    _save_state[(compression.kv_score.shape[0],)](
        compression.kv_score,
        compression.ape,
        batch.positions,
        compression.state,
        batch.state_slots,
        STATE_WIDTH=width,
        BLOCK=triton.next_power_of_2(width),
        RATIO=ratio,
        OVERLAP=overlap,
        STATE_PAGE=state_page,
        num_warps=4,
    )


def _run_c4_compression(batch, main, index) -> None:
    _run_c4_main(batch, main)
    _run_c4_index(batch, index)


def _run_c4_main(batch, main) -> None:
    _save(batch, main, 4, True, _C4_STATE_PAGE)
    _compress_main(batch, main, 4, True, _C4_PAGE, 4)


def _run_c4_index(batch, index) -> None:
    _save(batch, index, 4, True, _C4_STATE_PAGE)
    _compress_index(batch, index)


def _run_c128_compression(batch, main) -> None:
    _save(batch, main, 128, False, _C128_STATE_PAGE)
    _compress_main(batch, main, 128, False, _C128_PAGE, 16)


def _compress_main(batch, compression, ratio, overlap, cache_page, num_warps):
    _main_compress[(batch.positions.numel(),)](
        compression.state,
        batch.request_indices,
        batch.positions,
        batch.state_table,
        batch.state_table_base,
        compression.norm_weight,
        batch.cos_sin,
        compression.cache,
        batch.state_slots,
        batch.output_slots,
        STATE_WIDTH=compression.kv_score.shape[1] // 2,
        RATIO=ratio,
        OVERLAP=overlap,
        STATE_PAGE=_C4_STATE_PAGE if ratio == 4 else _C128_STATE_PAGE,
        CACHE_PAGE=cache_page,
        CACHE_STRIDE=compression.cache.stride(0),
        TABLE_WIDTH=batch.state_table.shape[1],
        HEAD_DIM=HEAD_DIM,
        num_warps=num_warps,
    )


def _compress_index(batch, compression) -> None:
    _index_compress[(batch.positions.numel(), 4)](
        compression.state,
        batch.request_indices,
        batch.positions,
        batch.state_table,
        batch.state_table_base,
        compression.norm_weight,
        batch.cos_sin,
        compression.cache,
        batch.state_slots,
        batch.output_slots,
        CACHE_STRIDE=compression.cache.stride(0),
        TABLE_WIDTH=batch.state_table.shape[1],
        num_warps=4,
    )


@triton.jit
def _c128_add_rn(left, right):
    return tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;",
        "=f,f,f",
        [left, right],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _c128_mul_rn(left, right):
    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;",
        "=f,f,f",
        [left, right],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _c128_div_rn(left, right):
    return tl.inline_asm_elementwise(
        "div.rn.f32 $0, $1, $2;",
        "=f,f,f",
        [left, right],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _c128_sub_rn(left, right):
    return tl.inline_asm_elementwise(
        "sub.rn.f32 $0, $1, $2;",
        "=f,f,f",
        [left, right],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _c128_torch_sum_dim0_rn(product):
    by_k = tl.trans(tl.reshape(product, (8, 4, 2, 2, 512)), (4, 1, 2, 3, 0))
    k_even, k_odd = tl.split(tl.reshape(by_k, (512, 4, 2, 2, 4, 2)))
    k_04, k_26 = tl.split(tl.reshape(k_even, (512, 4, 2, 2, 2, 2)))
    k_15, k_37 = tl.split(tl.reshape(k_odd, (512, 4, 2, 2, 2, 2)))
    k_0, k_4 = tl.split(k_04)
    k_2, k_6 = tl.split(k_26)
    k_1, k_5 = tl.split(k_15)
    k_3, k_7 = tl.split(k_37)
    partial = _c128_add_rn(k_0, k_1)
    partial = _c128_add_rn(partial, k_2)
    partial = _c128_add_rn(partial, k_3)
    partial = _c128_add_rn(partial, k_4)
    partial = _c128_add_rn(partial, k_5)
    partial = _c128_add_rn(partial, k_6)
    partial = _c128_add_rn(partial, k_7)

    by_a = tl.trans(partial, (0, 2, 3, 1))
    a_02, a_13 = tl.split(tl.reshape(by_a, (512, 2, 2, 2, 2)))
    a_0, a_2 = tl.split(a_02)
    a_1, a_3 = tl.split(a_13)
    partial = _c128_add_rn(a_0, a_1)
    partial = _c128_add_rn(partial, a_2)
    partial = _c128_add_rn(partial, a_3)

    y_02, y_13 = tl.split(partial)
    y_0, y_2 = tl.split(y_02)
    y_1, y_3 = tl.split(y_13)
    return _c128_add_rn(
        _c128_add_rn(y_0, y_2),
        _c128_add_rn(y_1, y_3),
    )


@triton.jit
def _c128_reduce_halves_rn(values, WIDTH: tl.constexpr):
    left, right = tl.split(tl.trans(tl.reshape(values, (2, WIDTH // 2))))
    return _c128_add_rn(left, right)


@triton.jit
def _c128_torch_variance(pooled):
    squared = _c128_mul_rn(pooled, pooled)
    columns_02, columns_13 = tl.split(tl.reshape(squared, (128, 2, 2)))
    column_0, column_2 = tl.split(columns_02)
    column_1, column_3 = tl.split(columns_13)
    partial = _c128_add_rn(column_0, column_1)
    partial = _c128_add_rn(partial, column_2)
    partial = _c128_add_rn(partial, column_3)
    partial = _c128_reduce_halves_rn(partial, WIDTH=128)
    partial = _c128_reduce_halves_rn(partial, WIDTH=64)
    partial = _c128_reduce_halves_rn(partial, WIDTH=32)
    partial = _c128_reduce_halves_rn(partial, WIDTH=16)
    partial = _c128_reduce_halves_rn(partial, WIDTH=8)
    partial = _c128_reduce_halves_rn(partial, WIDTH=4)
    partial = _c128_reduce_halves_rn(partial, WIDTH=2)
    return _c128_mul_rn(partial, 1.0 / 512.0)


@triton.jit
def _power2_quantize(
    input_ptr,
    output_ptr,
    scale_ptr,
    INPUT_ROW_STRIDE: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    FP8_LIMIT: tl.constexpr,
):
    program = tl.program_id(0)
    row = program // BLOCKS
    block = program - row * BLOCKS
    offsets = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    values = tl.load(input_ptr + row * INPUT_ROW_STRIDE + offsets).to(tl.float32)
    maximum = tl.maximum(tl.max(tl.abs(values), axis=0), 1.0e-4)
    exponent = tl.ceil(tl.log2(maximum / FP8_LIMIT))
    scale = tl.exp2(exponent)
    quantized = tl.maximum(tl.minimum(values / scale, FP8_LIMIT), -FP8_LIMIT)
    tl.store(output_ptr + row * BLOCKS * BLOCK_SIZE + offsets, quantized)
    tl.store(scale_ptr + block * ROWS + row, scale)


@triton.jit
def _normalize_hash_route_weight(
    weight,
    denominator,
    route_scale: tl.constexpr,
    denominator_min: tl.constexpr,
):
    return weight * route_scale / tl.maximum(denominator, denominator_min)


@triton.jit
def _ordered_float_key(value):
    bits = value.to(tl.uint32, bitcast=True)
    sign = 0x80000000
    return bits ^ tl.where((bits & sign) != 0, 0xFFFFFFFF, sign)


@triton.jit
def _learned_route(
    router_logits,
    route_bias,
    topk_ids,
    topk_weights,
    NUM_EXPERTS: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    ROUTE_SCALE: tl.constexpr,
    DENOMINATOR_MIN: tl.constexpr,
):
    row = tl.program_id(0)
    expert = tl.arange(0, NUM_EXPERTS)
    logits = tl.load(router_logits + row * NUM_EXPERTS + expert)
    scores = tl.sqrt(tl.where(logits > 20.0, logits, tl.log(1.0 + tl.exp(logits))))
    choice = scores + tl.load(route_bias + expert)
    packed = (_ordered_float_key(choice).to(tl.uint64) << 32) | (
        NUM_EXPERTS - expert
    ).to(tl.uint64)
    selected = tl.topk(packed, BLOCK_K, dim=0)
    slot = tl.arange(0, BLOCK_K)
    active = slot < TOP_K
    selected_ids = NUM_EXPERTS - (selected & 0xFFFFFFFF).to(tl.int32)
    selected_logits = tl.load(router_logits + row * NUM_EXPERTS + selected_ids)
    selected_weights = tl.sqrt(
        tl.where(
            selected_logits > 20.0,
            selected_logits,
            tl.log(1.0 + tl.exp(selected_logits)),
        )
    )
    selected_weights = tl.where(active, selected_weights, 0.0)
    selected_weights = _normalize_hash_route_weight(
        selected_weights,
        tl.sum(selected_weights, axis=0),
        ROUTE_SCALE,
        DENOMINATOR_MIN,
    )
    tl.store(topk_ids + row * TOP_K + slot, selected_ids, mask=active)
    tl.store(topk_weights + row * TOP_K + slot, selected_weights, mask=active)


@triton.jit
def _hash_route(
    router_logits,
    token_ids,
    hash_routes,
    topk_ids,
    topk_weights,
    NUM_EXPERTS: tl.constexpr,
    TOP_K: tl.constexpr,
    ROUTE_SCALE: tl.constexpr,
    DENOMINATOR_MIN: tl.constexpr,
):
    row = tl.program_id(0)
    slot = tl.arange(0, 8)
    active = slot < TOP_K
    token_id = tl.load(token_ids + row)
    expert_id = tl.load(hash_routes + token_id * TOP_K + slot, mask=active, other=0)
    score = tl.load(
        router_logits + row * NUM_EXPERTS + expert_id,
        mask=active,
        other=0.0,
    )
    softplus = tl.maximum(score, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(score)))
    weight = tl.where(active, tl.sqrt(softplus), 0.0)
    weight = _normalize_hash_route_weight(
        weight,
        tl.sum(weight, axis=0),
        ROUTE_SCALE,
        DENOMINATOR_MIN,
    )
    tl.store(topk_ids + row * TOP_K + slot, expert_id, mask=active)
    tl.store(topk_weights + row * TOP_K + slot, weight, mask=active)


@triton.jit
def _shared_swiglu_quantize(
    gate_up,
    output,
    output_scale,
    ROWS: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    LIMIT: tl.constexpr,
    FP8_LIMIT: tl.constexpr,
):
    program = tl.program_id(0)
    row = program // BLOCKS
    block = program - row * BLOCKS
    offsets = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    source = row * 2 * WIDTH + offsets
    gate = tl.load(gate_up + source).to(tl.float32)
    gate = tl.minimum(gate, LIMIT)
    up = tl.load(gate_up + source + WIDTH).to(tl.float32)
    up = tl.maximum(tl.minimum(up, LIMIT), -LIMIT)
    activated = (gate * tl.sigmoid(gate) * up).to(tl.bfloat16).to(tl.float32)
    maximum = tl.maximum(tl.max(tl.abs(activated), axis=0), 1.0e-4)
    exponent = tl.ceil(tl.log2(maximum / FP8_LIMIT))
    scale = tl.exp2(exponent)
    quantized = tl.maximum(tl.minimum(activated / scale, FP8_LIMIT), -FP8_LIMIT)
    tl.store(output + row * WIDTH + offsets, quantized)
    tl.store(output_scale + block * ROWS + row, scale)


@triton.jit
def _mxfp4_nibble(x):
    a = tl.minimum(tl.abs(x), 6.0)
    code = tl.where(
        a <= 0.25,
        0.0,
        tl.where(
            a <= 0.75,
            1.0,
            tl.where(
                a <= 1.25,
                2.0,
                tl.where(
                    a <= 1.75,
                    3.0,
                    tl.where(
                        a <= 2.5,
                        4.0,
                        tl.where(a <= 3.5, 5.0, tl.where(a <= 5.0, 6.0, 7.0)),
                    ),
                ),
            ),
        ),
    ).to(tl.uint8)
    return code | (((x < 0) & (code != 0)).to(tl.uint8) << 3)


@triton.jit
def _c4_query(
    projected,
    score_weights,
    positions,
    cos_sin,
    packed_query,
    packed_scale,
    output_weights,
    HEADS: tl.constexpr,
    LOCAL_HEADS: tl.constexpr,
    HEAD_WIDTH: tl.constexpr,
    PACKED_WIDTH: tl.constexpr,
    ROPE_WIDTH: tl.constexpr,
    WEIGHT_SCALE: tl.constexpr,
):
    row = tl.program_id(0)
    block = tl.program_id(1)
    token = row // HEADS
    head = row - token * HEADS
    dim = tl.arange(0, HEAD_WIDTH)
    values = tl.load(
        projected
        + token * PACKED_WIDTH
        + LOCAL_HEADS * 4 * HEAD_WIDTH
        + head * HEAD_WIDTH
        + dim
    ).to(tl.float32)
    pairs = tl.reshape(values, (HEAD_WIDTH // 2, 2))
    even, odd = tl.split(pairs)
    rope_pair = tl.arange(0, HEAD_WIDTH // 2) - (HEAD_WIDTH - ROPE_WIDTH) // 2
    is_rope = rope_pair >= 0
    cs = tl.maximum(rope_pair, 0)
    cs_base = cos_sin + tl.load(positions + token) * ROPE_WIDTH
    cosine = tl.load(cs_base + cs, mask=is_rope, other=1.0)
    sine = tl.load(cs_base + ROPE_WIDTH // 2 + cs, mask=is_rope, other=0.0)
    rotated = tl.interleave(even * cosine - odd * sine, odd * cosine + even * sine)
    rotated = rotated.to(tl.bfloat16).to(tl.float32)

    output = block * 32 + tl.arange(0, 32)
    parity = dim[:, None] & output[None, :]
    parity ^= parity >> 4
    parity ^= parity >> 2
    parity ^= parity >> 1
    signs = tl.where((parity & 1) == 0, 1.0, -1.0)
    hadamard = tl.sum(rotated[:, None] * signs, axis=0) * 128**-0.5
    hadamard = hadamard.to(tl.bfloat16).to(tl.float32)
    lo, hi = tl.split(tl.reshape(hadamard, (16, 2)))
    maximum = tl.maximum(tl.maximum(tl.max(tl.abs(lo)), tl.max(tl.abs(hi))), 1.0e-4)
    exponent = tl.ceil(tl.log2(maximum / 6.0))
    exponent = tl.minimum(tl.maximum(exponent, -127.0), 127.0)
    packed = _mxfp4_nibble(lo * tl.exp2(-exponent))
    packed |= _mxfp4_nibble(hi * tl.exp2(-exponent)) << 4
    tl.store(packed_query + row * 64 + block * 16 + tl.arange(0, 16), packed)
    byte_scale = (packed_scale + row).to(tl.pointer_type(tl.uint8))
    tl.store(byte_scale + block, (exponent + 127.0).to(tl.uint8))
    if block == 0:
        weight = (tl.load(score_weights + row).to(tl.float32) * WEIGHT_SCALE).to(
            tl.bfloat16
        )
        tl.store(output_weights + row, weight.to(tl.float32))


@triton.jit
def _fht_stage(values, dim, stride: tl.constexpr):
    peer = tl.gather(values, dim ^ stride, axis=0)
    return tl.where((dim & stride) == 0, values + peer, peer - values)


@triton.jit
def _c4_decode_query(
    projected,
    score_weights,
    positions,
    cos_sin,
    packed_query,
    packed_scale,
    output_weights,
    HEADS: tl.constexpr,
    LOCAL_HEADS: tl.constexpr,
    HEAD_WIDTH: tl.constexpr,
    PACKED_WIDTH: tl.constexpr,
    ROPE_WIDTH: tl.constexpr,
    WEIGHT_SCALE: tl.constexpr,
):
    row = tl.program_id(0)
    token = row // HEADS
    head = row - token * HEADS
    dim = tl.arange(0, HEAD_WIDTH)
    values = tl.load(
        projected
        + token * PACKED_WIDTH
        + LOCAL_HEADS * 4 * HEAD_WIDTH
        + head * HEAD_WIDTH
        + dim
    ).to(tl.float32)
    pairs = tl.reshape(values, (HEAD_WIDTH // 2, 2))
    even, odd = tl.split(pairs)
    rope_pair = tl.arange(0, HEAD_WIDTH // 2) - (HEAD_WIDTH - ROPE_WIDTH) // 2
    is_rope = rope_pair >= 0
    cs = tl.maximum(rope_pair, 0)
    cs_base = cos_sin + tl.load(positions + token) * ROPE_WIDTH
    cosine = tl.load(cs_base + cs, mask=is_rope, other=1.0)
    sine = tl.load(cs_base + ROPE_WIDTH // 2 + cs, mask=is_rope, other=0.0)
    values = tl.interleave(even * cosine - odd * sine, odd * cosine + even * sine)
    values = values.to(tl.bfloat16).to(tl.float32)
    values = _fht_stage(values, dim, 1)
    values = _fht_stage(values, dim, 2)
    values = _fht_stage(values, dim, 4)
    values = _fht_stage(values, dim, 8)
    values = _fht_stage(values, dim, 16)
    values = _fht_stage(values, dim, 32)
    values = _fht_stage(values, dim, 64)
    hadamard = (values * 128**-0.5).to(tl.bfloat16).to(tl.float32)
    groups = tl.reshape(hadamard, (4, 32))
    maximum = tl.maximum(tl.max(tl.abs(groups), axis=1), 1.0e-4)
    exponent = tl.ceil(tl.log2(maximum / 6.0))
    exponent = tl.minimum(tl.maximum(exponent, -127.0), 127.0)
    normalized = groups * tl.exp2(-exponent[:, None])
    lo, hi = tl.split(tl.reshape(normalized, (4, 16, 2)))
    packed = _mxfp4_nibble(lo) | (_mxfp4_nibble(hi) << 4)
    byte = tl.arange(0, 4)[:, None] * 16 + tl.arange(0, 16)[None, :]
    tl.store(packed_query + row * 64 + byte, packed)
    scale = (packed_scale + row).to(tl.pointer_type(tl.uint8))
    tl.store(scale + tl.arange(0, 4), (exponent + 127.0).to(tl.uint8))
    weight = (tl.load(score_weights + row).to(tl.float32) * WEIGHT_SCALE).to(
        tl.bfloat16
    )
    tl.store(output_weights + row, weight.to(tl.float32))


@triton.jit(do_not_specialize=["token_start", "token_count", "table_width"])
def _stage_prefill_index_kernel(
    positions,
    requests,
    block_table,
    block_table_base,
    staged_table,
    staged_lengths,
    staged_requests,
    token_start,
    token_count,
    table_width,
):
    row = tl.program_id(0)
    block = tl.program_id(1)
    token = token_start + row
    valid = row < token_count
    request = tl.load(requests + token, mask=valid, other=-1)
    previous = tl.load(requests + token - 1, mask=valid & (row > 0), other=-2)
    position = tl.load(positions + token, mask=valid, other=-1)
    base = tl.load(block_table_base + request, mask=valid, other=0)
    length = tl.maximum((position + 1) // 4 - base * 32, 0)
    length = tl.minimum(length, table_width * 32)

    if block == 0:
        tl.store(staged_lengths + row, length)
        tl.store(staged_requests + row, request)

    run_start = valid & ((row == 0) | (request != previous))
    for column in range(block * 256, table_width, 1024):
        columns = column + tl.arange(0, 256)
        mask = run_start & (columns < table_width)
        tl.store(
            staged_table + row * table_width + columns,
            tl.load(
                block_table + request * table_width + columns,
                mask=mask,
                other=0,
            ),
            mask=mask,
        )


@triton.jit
def _load_prefill_cache_row(
    cache,
    page,
    row,
    dimensions,
    valid,
    PAGE: tl.constexpr,
    stride,
):
    token = cache + page.to(tl.int64) * stride + row * 576
    fp8 = tl.load(token + dimensions, mask=valid & (dimensions < 448), other=0)
    exponent = tl.load(
        cache + page.to(tl.int64) * stride + PAGE * 576 + row * 8 + dimensions // 64,
        mask=valid & (dimensions < 448),
        other=127,
    ).to(tl.float32)
    dequantized = fp8.to(tl.float8e4nv, bitcast=True).to(tl.float32)
    dequantized *= tl.exp2(exponent - 127.0)
    rope = tl.load(
        (token + 448).to(tl.pointer_type(tl.bfloat16)) + dimensions - 448,
        mask=valid & (dimensions >= 448),
        other=0.0,
    ).to(tl.float32)
    return tl.where(dimensions < 448, dequantized, rope)


@triton.jit(
    do_not_specialize=[
        "token_count",
        "request_count",
        "compressed_capacity",
        "index_stride",
        "table_width",
        "row_count",
        "compressed_stride",
        "raw_stride",
    ]
)
def _stage_prefill_attention_kernel(
    current_kv,
    positions,
    requests,
    cos_sin,
    query_start,
    prefix_lengths,
    state_slots,
    block_table,
    block_table_base,
    raw_cache,
    compressed_cache,
    candidates,
    output_kv,
    output_indices,
    output_lengths,
    raw_slots,
    token_count,
    request_count,
    compressed_capacity,
    index_stride,
    table_width,
    row_count,
    compressed_stride,
    raw_stride,
    RATIO: tl.constexpr,
    COMPRESSED_PAGE: tl.constexpr,
    RAW_RING_PAGES: tl.constexpr,
):
    worker = tl.program_id(0)
    block = tl.program_id(1)
    dimensions = block * 128 + tl.arange(0, 128)
    compressed_rows = request_count * compressed_capacity
    raw_offset = compressed_rows
    current_offset = compressed_rows + request_count * 128
    safe_capacity = tl.maximum(compressed_capacity, 1)

    for row in range(worker, row_count, tl.num_programs(0)):
        is_compressed = row < compressed_rows
        compressed_request = tl.minimum(row // safe_capacity, request_count - 1)
        compressed_local = row % safe_capacity
        compressed_base = (
            tl.load(block_table_base + compressed_request) * COMPRESSED_PAGE
        )
        query_tokens = tl.load(query_start + compressed_request + 1)
        query_tokens -= tl.load(query_start + compressed_request)
        compressed_end = (
            tl.load(prefix_lengths + compressed_request) + query_tokens
        ) // RATIO
        compressed_valid = is_compressed & (
            compressed_local < compressed_end - compressed_base
        )
        compressed_position = compressed_base + compressed_local
        table_column = compressed_position // COMPRESSED_PAGE
        table_column -= tl.load(block_table_base + compressed_request)
        physical_page = tl.load(
            block_table + compressed_request * table_width + table_column,
            mask=compressed_valid & (table_column >= 0) & (table_column < table_width),
            other=-1,
        )
        values = _load_prefill_cache_row(
            compressed_cache,
            physical_page,
            compressed_position % COMPRESSED_PAGE,
            dimensions,
            compressed_valid & (physical_page >= 0),
            COMPRESSED_PAGE,
            compressed_stride,
        )
        tl.store(
            output_kv + (row * 512 + dimensions),
            values.to(tl.bfloat16),
            mask=is_compressed & (dimensions < 512),
        )

        is_prefix = (row >= raw_offset) & (row < current_offset)
        prefix_row = row - raw_offset
        prefix_request = tl.maximum(0, tl.minimum(prefix_row // 128, request_count - 1))
        prefix_position = (
            tl.load(prefix_lengths + prefix_request) - 128 + prefix_row % 128
        )
        prefix_valid = is_prefix & (prefix_position >= 0)
        raw_page = tl.load(state_slots + prefix_request) * RAW_RING_PAGES
        raw_page += (prefix_position // 64) % RAW_RING_PAGES
        values = _load_prefill_cache_row(
            raw_cache,
            raw_page,
            prefix_position % 64,
            dimensions,
            prefix_valid,
            64,
            raw_stride,
        )
        tl.store(
            output_kv + (row * 512 + dimensions),
            values.to(tl.bfloat16),
            mask=is_prefix & (dimensions < 512),
        )

        token = row - current_offset
        is_current = (row >= current_offset) & (token < token_count)
        position = tl.load(positions + token, mask=is_current, other=0)
        values = tl.load(
            current_kv + token * 512 + dimensions,
            mask=is_current & (dimensions < 512),
            other=0.0,
        ).to(tl.float32)
        rope_dimension = dimensions - 448
        pair = tl.maximum(rope_dimension // 2, 0)
        pair_base = 448 + pair * 2
        even = tl.load(
            current_kv + token * 512 + pair_base,
            mask=is_current & (dimensions >= 448),
            other=0.0,
        ).to(tl.float32)
        odd = tl.load(
            current_kv + token * 512 + pair_base + 1,
            mask=is_current & (dimensions >= 448),
            other=0.0,
        ).to(tl.float32)
        cosine = tl.load(
            cos_sin + position * 64 + pair,
            mask=is_current & (dimensions >= 448),
            other=1.0,
        )
        sine = tl.load(
            cos_sin + position * 64 + 32 + pair,
            mask=is_current & (dimensions >= 448),
            other=0.0,
        )
        rotated = tl.where(
            rope_dimension % 2 == 0,
            even * cosine - odd * sine,
            odd * cosine + even * sine,
        )
        values = tl.where(dimensions >= 448, rotated, values)
        tl.store(
            output_kv + row * 512 + dimensions,
            values.to(tl.bfloat16),
            mask=is_current & (dimensions < 512),
        )

        if block == 0:
            token = row
            token_valid = token < token_count
            request = tl.load(requests + token, mask=token_valid, other=0)
            position = tl.load(positions + token, mask=token_valid, other=0)
            base = tl.load(block_table_base + request) * COMPRESSED_PAGE
            compressed_limit = tl.minimum(
                (position + 1) // RATIO - base, compressed_capacity
            )
            compressed_limit = tl.maximum(compressed_limit, 0)
            if RATIO == 4:
                candidate_offsets = tl.arange(0, 512)
                candidate = tl.load(
                    candidates + token * 512 + candidate_offsets,
                    mask=token_valid,
                    other=-1,
                )
                candidate_valid = (candidate >= 0) & (candidate < compressed_limit)
                candidate_rank = tl.cumsum(candidate_valid.to(tl.int32), axis=0) - 1
                tl.store(
                    output_indices + token * index_stride + candidate_rank,
                    request * compressed_capacity + candidate,
                    mask=token_valid & candidate_valid,
                )
                compressed_length = tl.sum(candidate_valid.to(tl.int32), axis=0)
            else:
                compressed_length = compressed_limit
                for start in range(0, compressed_capacity, 128):
                    offsets = start + tl.arange(0, 128)
                    tl.store(
                        output_indices + token * index_stride + offsets,
                        request * compressed_capacity + offsets,
                        mask=token_valid & (offsets < compressed_length),
                    )

            query_offset = tl.load(query_start + request)
            local = token - query_offset
            prefix = tl.load(prefix_lengths + request)
            current_length = tl.minimum(local + 1, 128)
            prefix_length = tl.minimum(prefix, 128 - current_length)
            raw_length = prefix_length + current_length
            raw_indices = tl.arange(0, 128)
            from_prefix = raw_indices < prefix_length
            prefix_index = (
                raw_offset + request * 128 + 128 - prefix_length + raw_indices
            )
            current_start = local + 1 - current_length
            current_index = current_offset + query_offset + current_start
            current_index += raw_indices - prefix_length
            tl.store(
                output_indices + token * index_stride + compressed_length + raw_indices,
                tl.where(from_prefix, prefix_index, current_index),
                mask=token_valid & (raw_indices < raw_length),
            )
            tl.store(
                output_lengths + token,
                compressed_length + raw_length,
                mask=token_valid,
            )
            query_tokens = tl.load(query_start + request + 1) - query_offset
            keep = local >= tl.maximum(query_tokens - 128, 0)
            raw_ring = RAW_RING_PAGES * 64
            slot = tl.load(state_slots + request) * raw_ring + position % raw_ring
            tl.store(raw_slots + token, tl.where(keep, slot, -1), mask=token_valid)


@triton.jit
def _save_state(
    kv_score,
    ape,
    positions,
    state,
    slots,
    STATE_WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
    RATIO: tl.constexpr,
    OVERLAP: tl.constexpr,
    STATE_PAGE: tl.constexpr,
):
    token = tl.program_id(0)
    slot = tl.load(slots + token)
    if slot < 0:
        return
    base = state + (slot // STATE_PAGE).to(tl.int64) * STATE_PAGE * 2 * STATE_WIDTH
    base += (slot % STATE_PAGE) * 2 * STATE_WIDTH
    offsets = tl.arange(0, BLOCK)
    mask = offsets < STATE_WIDTH
    row = tl.load(positions + token) % RATIO
    if OVERLAP:
        half: tl.constexpr = STATE_WIDTH // 2
        ape_offsets = tl.where(
            offsets < half,
            row * half + offsets,
            (row + RATIO) * half + offsets - half,
        )
    else:
        ape_offsets = row * STATE_WIDTH + offsets
    projected = kv_score + token * 2 * STATE_WIDTH
    values = tl.load(projected + offsets, mask=mask)
    score = tl.load(projected + STATE_WIDTH + offsets, mask=mask)
    tl.store(base + offsets, values, mask=mask)
    tl.store(
        base + STATE_WIDTH + offsets,
        score + tl.load(ape + ape_offsets, mask=mask),
        mask=mask,
    )


@triton.jit
def _main_compress(
    state,
    requests,
    positions,
    table,
    table_base,
    norm_weight,
    cos_sin,
    cache,
    slots,
    output_slots,
    STATE_WIDTH: tl.constexpr,
    RATIO: tl.constexpr,
    OVERLAP: tl.constexpr,
    STATE_PAGE: tl.constexpr,
    CACHE_PAGE: tl.constexpr,
    CACHE_STRIDE: tl.constexpr,
    TABLE_WIDTH: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    token = tl.program_id(0)
    slot = tl.load(slots + token)
    if slot < 0:
        return
    position = tl.load(positions + token)
    if (position + 1) % RATIO != 0:
        return
    output_slot = tl.load(output_slots + token)
    if output_slot < 0:
        return
    request = tl.load(requests + token)
    window: tl.constexpr = (1 + OVERLAP) * RATIO
    window_offsets = tl.arange(0, window)
    source_positions = position - window + 1 + window_offsets
    valid = source_positions >= 0
    columns = source_positions // STATE_PAGE - tl.load(table_base + request)
    valid &= (columns >= 0) & (columns < TABLE_WIDTH)
    pages = tl.load(table + request * TABLE_WIDTH + columns, mask=valid, other=-1)
    head = (window_offsets >= RATIO).to(tl.int32) * HEAD_DIM
    source = state + pages.to(tl.int64)[:, None] * STATE_PAGE * 2 * STATE_WIDTH
    source += (source_positions % STATE_PAGE)[:, None] * 2 * STATE_WIDTH + head[:, None]
    dim = tl.arange(0, HEAD_DIM)
    valid = valid[:, None] & (pages[:, None] >= 0)
    score = tl.load(
        source + STATE_WIDTH + dim[None, :], mask=valid, other=-float("inf")
    )
    if RATIO == 128:
        values = tl.load(source + dim[None, :], mask=valid, other=0.0)
        maximum = tl.full((HEAD_DIM,), -3.4028234663852886e38, tl.float32)
        for row_index in tl.range(0, 128, loop_unroll_factor=1):
            row_position = position - 127 + row_index
            row_valid = row_position >= 0
            column = row_position // STATE_PAGE - tl.load(table_base + request)
            row_valid &= (column >= 0) & (column < TABLE_WIDTH)
            source_page = tl.load(
                table + request * TABLE_WIDTH + column,
                mask=row_valid,
                other=-1,
            )
            row_source = state + source_page.to(tl.int64) * STATE_PAGE * 2 * STATE_WIDTH
            row_source += (row_position % STATE_PAGE) * 2 * STATE_WIDTH
            row_score = tl.load(
                row_source + STATE_WIDTH + dim,
                mask=row_valid & (source_page >= 0),
                other=-float("inf"),
            )
            maximum = tl.maximum(maximum, row_score)
        denominator = tl.zeros((HEAD_DIM,), tl.float32)
        for row_index in tl.range(0, 128, loop_unroll_factor=1):
            row_position = position - 127 + row_index
            row_valid = row_position >= 0
            column = row_position // STATE_PAGE - tl.load(table_base + request)
            row_valid &= (column >= 0) & (column < TABLE_WIDTH)
            source_page = tl.load(
                table + request * TABLE_WIDTH + column,
                mask=row_valid,
                other=-1,
            )
            row_source = state + source_page.to(tl.int64) * STATE_PAGE * 2 * STATE_WIDTH
            row_source += (row_position % STATE_PAGE) * 2 * STATE_WIDTH
            row_score = tl.load(
                row_source + STATE_WIDTH + dim,
                mask=row_valid & (source_page >= 0),
                other=-float("inf"),
            )
            denominator = _c128_add_rn(
                denominator, libdevice.exp(_c128_sub_rn(row_score, maximum))
            )
        numerator = libdevice.exp(_c128_sub_rn(score, maximum[None, :]))
        weights = _c128_div_rn(numerator, denominator[None, :])
        product = _c128_mul_rn(values, weights)
        pooled = _c128_torch_sum_dim0_rn(product)
        pooled = pooled.to(tl.bfloat16).to(tl.float32)
    else:
        weights = tl.softmax(score, dim=0)
        values = tl.load(source + dim[None, :], mask=valid, other=0.0)
        pooled = tl.sum(values * weights, axis=0).to(tl.bfloat16).to(tl.float32)
    rms = tl.load(norm_weight + dim)
    if RATIO == 128:
        variance = _c128_torch_variance(pooled)
    else:
        variance = tl.sum(pooled * pooled, axis=0) / HEAD_DIM
    normed = (pooled * tl.rsqrt(variance + 1.0e-6) * rms).to(tl.bfloat16).to(tl.float32)
    page = cache + (output_slot // CACHE_PAGE).to(tl.int64) * CACHE_STRIDE
    row = output_slot % CACHE_PAGE
    value_out = page + row * 576
    scale_out = page + CACHE_PAGE * 576 + row * 8
    grouped = tl.reshape(normed, (8, 64))
    amax = tl.maximum(tl.max(tl.abs(grouped), axis=1), 1.0e-4)
    exponents = tl.ceil(tl.log2(amax / 448.0))
    fp8 = tl.clamp(grouped * tl.reshape(tl.exp2(-exponents), (8, 1)), -448.0, 448.0)
    fp8 = tl.reshape(fp8.to(tl.float8e4nv).to(tl.uint8, bitcast=True), (HEAD_DIM,))
    tl.store(value_out + dim, fp8, mask=dim < 448)
    scale = tl.arange(0, 8)
    encoded = tl.maximum(tl.minimum(exponents + 127.0, 255.0), 0.0)
    tl.store(scale_out + scale, encoded.to(tl.uint8), mask=scale < 7)
    tl.store(scale_out + 7, tl.zeros((), dtype=tl.uint8))
    pairs = tl.reshape(normed, (HEAD_DIM // 2, 2))
    even, odd = tl.split(pairs)
    rope_pair = tl.arange(0, HEAD_DIM // 2) - 224
    is_rope = rope_pair >= 0
    cs = tl.maximum(rope_pair, 0)
    cs_base = cos_sin + (position // RATIO * RATIO) * 64
    cosine = tl.load(cs_base + cs, mask=is_rope, other=1.0)
    sine = tl.load(cs_base + 32 + cs, mask=is_rope, other=0.0)
    if RATIO == 128:
        even_cosine = _c128_mul_rn(even, cosine)
        odd_sine = _c128_mul_rn(odd, sine)
        odd_cosine = _c128_mul_rn(odd, cosine)
        even_sine = _c128_mul_rn(even, sine)
        real = _c128_sub_rn(even_cosine, odd_sine)
        imaginary = _c128_add_rn(odd_cosine, even_sine)
        rotated = tl.interleave(real, imaginary)
    else:
        rotated = tl.interleave(even * cosine - odd * sine, odd * cosine + even * sine)
    rope_out = (value_out + 448).to(tl.pointer_type(tl.bfloat16))
    tl.store(rope_out + dim - 448, rotated.to(tl.bfloat16), mask=dim >= 448)


@triton.jit
def _index_compress(
    state,
    requests,
    positions,
    table,
    table_base,
    norm_weight,
    cos_sin,
    cache,
    slots,
    output_slots,
    CACHE_STRIDE: tl.constexpr,
    TABLE_WIDTH: tl.constexpr,
):
    token = tl.program_id(0)
    block = tl.program_id(1)
    slot = tl.load(slots + token)
    if slot < 0:
        return
    position = tl.load(positions + token)
    if (position + 1) % 4 != 0:
        return
    output_slot = tl.load(output_slots + token)
    if output_slot < 0:
        return
    request = tl.load(requests + token)
    offsets = tl.arange(0, 8)
    source_positions = position - 7 + offsets
    valid = source_positions >= 0
    columns = source_positions // 4 - tl.load(table_base + request)
    valid &= (columns >= 0) & (columns < TABLE_WIDTH)
    pages = tl.load(table + request * TABLE_WIDTH + columns, mask=valid, other=-1)
    head = (offsets >= 4).to(tl.int32) * 128
    source = (
        state
        + pages.to(tl.int64)[:, None] * 2048
        + (source_positions % 4)[:, None] * 512
    )
    source += head[:, None]
    dim = tl.arange(0, 128)
    valid = valid[:, None] & (pages[:, None] >= 0)
    score = tl.load(source + 256 + dim[None, :], mask=valid, other=-float("inf"))
    weights = tl.softmax(score, dim=0)
    values = tl.load(source + dim[None, :], mask=valid, other=0.0)
    pooled = tl.sum(values * weights, axis=0).to(tl.bfloat16).to(tl.float32)
    variance = tl.sum(pooled * pooled, axis=0) / 128
    normed = pooled * tl.rsqrt(variance + 1.0e-6) * tl.load(norm_weight + dim)
    normed = normed.to(tl.bfloat16).to(tl.float32)
    pairs = tl.reshape(normed, (64, 2))
    even, odd = tl.split(pairs)
    rope_pair = tl.arange(0, 64) - 32
    is_rope = rope_pair >= 0
    cs = tl.maximum(rope_pair, 0)
    cs_base = cos_sin + (position // 4 * 4) * 64
    cosine = tl.load(cs_base + cs, mask=is_rope, other=1.0)
    sine = tl.load(cs_base + 32 + cs, mask=is_rope, other=0.0)
    rotated = tl.interleave(even * cosine - odd * sine, odd * cosine + even * sine)
    rotated = rotated.to(tl.bfloat16).to(tl.float32)
    output = block * 32 + tl.arange(0, 32)
    inputs = tl.arange(0, 128)
    parity = inputs[:, None] & output[None, :]
    parity ^= parity >> 4
    parity ^= parity >> 2
    parity ^= parity >> 1
    signs = tl.where((parity & 1) == 0, 1.0, -1.0)
    hadamard = tl.sum(rotated[:, None] * signs, axis=0) * 128**-0.5
    hadamard = hadamard.to(tl.bfloat16).to(tl.float32)
    lo, hi = tl.split(tl.reshape(hadamard, (16, 2)))
    amax = tl.maximum(tl.max(tl.abs(lo)), tl.max(tl.abs(hi)))
    exponent = tl.ceil(tl.log2(tl.maximum(amax, 1.0e-4) / 6.0))
    exponent = tl.minimum(tl.maximum(exponent, -127.0), 127.0)
    scale = tl.exp2(-exponent)
    packed = _mxfp4_nibble(lo * scale) | (_mxfp4_nibble(hi * scale) << 4)
    page = cache + (output_slot // 32).to(tl.int64) * CACHE_STRIDE
    row = output_slot % 32
    tl.store(page + row * 64 + block * 16 + tl.arange(0, 16), packed)
    tl.store(page + 32 * 64 + row * 4 + block, (exponent + 127.0).to(tl.uint8))

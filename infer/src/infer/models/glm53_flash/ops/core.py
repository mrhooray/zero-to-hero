import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from importlib.metadata import version
from pathlib import Path

import torch
import torch.distributed as dist
import triton
import triton.language as tl

from infer.models.glm53_flash.model import (
    CONV_KERNEL_SIZE,
    FP8_BLOCK_SIZE,
    GLM53_TARGET_VERIFY_WIDTH,
    HEAD_DIM,
    HIDDEN_SIZE,
    KDA_CHUNK_SIZE,
    LOCAL_ROUTED_EXPERTS,
    LOCAL_VOCAB_SIZE,
    MHC_EPS,
    MHC_NORMALIZATION_ITERS,
    MHC_PRE_MAX_SPLITS,
    MHC_STREAMS,
    MOE_INTERMEDIATE_SIZE,
    NUM_HEADS,
    NUM_ROUTED_EXPERTS,
    RMS_NORM_EPS,
    SPARSE_FFN_CLAMP,
    SPARSE_FFN_DECODE_BATCH_SIZES,
    SPARSE_FFN_ROUTE_SCALE,
    SPARSE_FFN_TEP4_SHAPES,
    SPARSE_MLA_B200_SMS,
    SPARSE_MLA_COMPOUND_BLOCKS,
    SPARSE_MLA_HISTORY_BLOCK_TOKENS,
    SPARSE_MLA_INDEX_CONTEXT_CAPACITY,
    SPARSE_MLA_INDEX_PAGE_BYTES,
    SPARSE_MLA_INDEX_POOL_TOKENS,
    SPARSE_MLA_INDEX_SCORE_SCALE,
    SPARSE_MLA_INDEX_TOP_K,
    SPARSE_MLA_INDEX_WEIGHT_SCALE,
    SPARSE_MLA_INDEXER_HEAD_DIM,
    SPARSE_MLA_INDEXER_HEADS,
    SPARSE_MLA_KV_LORA_RANK,
    SPARSE_MLA_LATENT_PAGE_TOKENS,
    SPARSE_MLA_LAYER_NORM_EPS,
    SPARSE_MLA_MAX_CONTEXT_TOKENS,
    SPARSE_MLA_Q_LORA_RANK,
    SPARSE_MLA_QK_NOPE_HEAD_DIM,
    SPARSE_MLA_SPARSE_CAPACITY,
    SPARSE_MLA_VALUE_HEAD_DIM,
    TOKENIZER_VOCAB_SIZE,
    TOP_K,
    TP_SIZE,
    VOCAB_SIZE,
    GLM53DenseWorkspace,
    GLM53EndpointWeights,
    GLM53EndpointWorkspace,
    GLM53LayerWeights,
    GLM53SparseKDALayerWeights,
    GLM53SparseMLALayerWeights,
    KDADecodeWorkspace,
    KDAPrefillWorkspace,
    KDAState,
    KDAWeights,
    MHCWeights,
    SparseFFNDecodeWorkspace,
    SparseFFNWeights,
    SparseMLADecodeBatch,
    SparseMLADecodeWeights,
    SparseMLADecodeWorkspace,
    SparseMLAHistory,
    SparseMLAOutputWeights,
    SparseMLAOutputWorkspace,
    SparseMLAProjectionWeights,
    SparseMLAProjectionWorkspace,
    dense_workspace_shapes,
    glm53_endpoint_workspace_shapes,
    glm53_local_heads,
    kda_decode_workspace_shapes,
    kda_prefill_workspace_shapes,
    sparse_ffn_decode_workspace_shapes,
    sparse_mla_decode_workspace_shapes,
    sparse_mla_output_workspace_shapes,
    sparse_mla_projection_workspace_shapes,
)
from infer.models.glm53_flash.ops.distributed_argmax import glm53_distributed_argmax
from infer.models.glm53_flash.ops.megafuse import (
    glm53_megafuse_decode,
    glm53_megafuse_verify,
)
from infer.models.glm53_flash.ops.prefill_kda import (
    GLM53PrefillMetadata,
    glm53_prefill_kda,
)
from infer.models.glm53_flash.ops.segmented_conv import glm53_segmented_conv_prefill
from infer.models.glm53_flash.ops.sparse_mla_prefill import (
    GLM53StagedSparseMLAPrefillBatch,
    glm53_sparse_mla_packed_prefill_append,
)

_SPARSE_MLA_FLASHINFER_PACKAGES = {
    "apache-tvm-ffi": "0.1.11",
    "flashinfer-cubin": "0.6.18.dev20260819",
    "flashinfer-jit-cache": "0.6.18.dev20260819+cu129",
    "flashinfer-python": "0.6.18.dev20260819",
    "nvidia-cutlass-dsl": "4.5.0",
    "nvidia-cutlass-dsl-libs-base": "4.5.0",
    "sgl-deep-gemm": "0.1.5.post3",
    "triton": "3.7.1",
}
_SPARSE_MLA_FLASHINFER_COMMIT = "61a6c651872a7d3f2f6dcc1ced61633d8f8ba3dd"
_SPARSE_MLA_DEEP_GEMM_SHA256 = (
    "fafde24e6413a6be95ed0e094cf5746311ae8c87c3fe46466839f16b17e1896e"
)

type _AllReduce = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True, slots=True)
class GLM53PrefillBatch:
    """GPU descriptors for one immutable, CPU-validated scheduler plan.

    Construct this only after ``validate_glm53_segmented_conv_plan`` succeeds.
    Both state-index tensors must come from the same unique host slot list; the
    int32 form serves Triton and the int64 form serves PyTorch gather/scatter.
    """

    metadata: GLM53PrefillMetadata
    state_indices: torch.Tensor
    state_indices_int64: torch.Tensor
    has_initial: torch.Tensor
    segments: torch.Tensor


class GLM53Ops:
    def __init__(
        self,
        attention_tp_size: int = TP_SIZE,
        process_group: object | None = None,
    ) -> None:
        glm53_local_heads(attention_tp_size)
        if attention_tp_size == 1 and process_group is None:
            raise ValueError("GLM DEP4 ops require a process group")
        self._attention_tp_size = attention_tp_size
        self._process_group = process_group
        self._moe_token_counts: tuple[int, ...] | None = None
        self._sparse_ffn_overlap: tuple[object, object, object] | None = None

    def set_moe_token_counts(
        self,
        local_tokens: int,
        token_counts: tuple[int, ...] | None = None,
    ) -> None:
        if self._attention_tp_size != 1:
            return
        if type(local_tokens) is not int or local_tokens < 0:
            raise ValueError("local MoE token count must be a nonnegative int")
        counts = (local_tokens,) * TP_SIZE if token_counts is None else token_counts
        if (
            len(counts) != TP_SIZE
            or any(type(count) is not int or count < 0 for count in counts)
            or not any(counts)
        ):
            raise ValueError("GLM DEP4 MoE token counts must be four nonnegative ints")
        assert self._process_group is not None
        if counts[dist.get_rank(self._process_group)] != local_tokens:
            raise ValueError("local MoE token count does not match the process rank")
        self._moe_token_counts = counts

    def decode_embedding(
        self,
        token_ids: torch.Tensor,
        weights: GLM53EndpointWeights[torch.Tensor],
        workspace: GLM53EndpointWorkspace[torch.Tensor],
        process_group: object,
        all_reduce: _AllReduce,
    ) -> torch.Tensor:
        batch_size = token_ids.shape[0] if token_ids.ndim == 1 else 0
        _validate_glm53_endpoint(weights, workspace, process_group, batch_size)
        _require_glm53_endpoint_tensors(
            weights.embedding.device,
            (token_ids, torch.int64, (batch_size,)),
        )
        w = workspace
        if self._attention_tp_size == 1:
            w.local_token.copy_(token_ids)
            torch.index_select(weights.embedding, 0, token_ids, out=w.embedding)
            w.streams.copy_(w.embedding.unsqueeze(1))
            return w.streams

        vocab_start = dist.get_rank(process_group) * LOCAL_VOCAB_SIZE

        torch.sub(token_ids, vocab_start, out=w.local_token)
        torch.clamp(w.local_token, 0, LOCAL_VOCAB_SIZE - 1, out=w.local_token)
        torch.index_select(weights.embedding, 0, w.local_token, out=w.embedding)
        torch.ge(token_ids, vocab_start, out=w.local_active)
        w.embedding.mul_(w.local_active.view(batch_size, 1))
        torch.lt(token_ids, vocab_start + LOCAL_VOCAB_SIZE, out=w.local_active)
        w.embedding.mul_(w.local_active.view(batch_size, 1))
        embedding = all_reduce(w.embedding)
        w.streams.copy_(embedding.unsqueeze(1))
        return w.streams

    def prefill_embedding(
        self,
        token_ids: torch.Tensor,
        weights: GLM53EndpointWeights[torch.Tensor],
        workspace: GLM53EndpointWorkspace[torch.Tensor],
        process_group: object,
    ) -> torch.Tensor:
        if process_group is not dist.group.WORLD:
            raise ValueError("GLM prefill embedding requires the WORLD process group")
        total_tokens = token_ids.shape[0] if token_ids.ndim == 1 else 0
        if not 1 <= total_tokens <= KDA_CHUNK_SIZE:
            raise ValueError(
                f"prefill token_ids must have shape [T] for 1 <= T <= {KDA_CHUNK_SIZE}"
            )
        _validate_glm53_endpoint(weights, workspace, process_group, total_tokens)
        _require_glm53_endpoint_tensors(
            weights.embedding.device,
            (token_ids, torch.int64, (total_tokens,)),
        )
        _reject_sparse_layer_aliases(
            _workspace_tensors("workspace", workspace),
            (("token_ids", token_ids), *_workspace_tensors("weights", weights)),
        )
        w = workspace
        rows = slice(total_tokens)
        if self._attention_tp_size == 1:
            w.local_token[rows].copy_(token_ids)
            torch.index_select(weights.embedding, 0, token_ids, out=w.embedding[rows])
            w.streams.copy_(w.embedding.unsqueeze(1))
            return w.streams

        vocab_start = dist.get_rank(process_group) * LOCAL_VOCAB_SIZE

        torch.sub(token_ids, vocab_start, out=w.local_token[rows])
        torch.clamp(
            w.local_token[rows], 0, LOCAL_VOCAB_SIZE - 1, out=w.local_token[rows]
        )
        torch.index_select(
            weights.embedding, 0, w.local_token[rows], out=w.embedding[rows]
        )
        torch.ge(token_ids, vocab_start, out=w.local_active[rows])
        w.embedding[rows].mul_(w.local_active[rows].view(total_tokens, 1))
        torch.lt(
            token_ids,
            vocab_start + LOCAL_VOCAB_SIZE,
            out=w.local_active[rows],
        )
        w.embedding[rows].mul_(w.local_active[rows].view(total_tokens, 1))
        dist.all_reduce(w.embedding, op=dist.ReduceOp.SUM, group=process_group)
        w.streams.copy_(w.embedding.unsqueeze(1))
        return w.streams

    def normalize_head(
        self,
        streams: torch.Tensor,
        weights: GLM53EndpointWeights[torch.Tensor],
        workspace: GLM53EndpointWorkspace[torch.Tensor],
        process_group: object,
    ) -> torch.Tensor:
        batch_size = streams.shape[0] if streams.ndim == 3 else 0
        _validate_glm53_endpoint(weights, workspace, process_group, batch_size)
        _require_glm53_endpoint_tensors(
            weights.embedding.device,
            (streams, torch.bfloat16, (batch_size, MHC_STREAMS, HIDDEN_SIZE)),
        )
        w = workspace

        w.mean_f32.copy_(streams[:, 0])
        w.mean_f32.add_(streams[:, 1])
        w.mean_f32.add_(streams[:, 2])
        w.mean_f32.add_(streams[:, 3])
        w.mean_f32.mul_(0.25)
        w.collapsed.copy_(w.mean_f32)
        _rmsnorm(w.collapsed, weights.final_norm, w.normalized)
        return w.normalized

    def decode_head(
        self,
        streams: torch.Tensor,
        weights: GLM53EndpointWeights[torch.Tensor],
        workspace: GLM53EndpointWorkspace[torch.Tensor],
        process_group: object,
    ) -> torch.Tensor:
        normalized = self.normalize_head(streams, weights, workspace, process_group)
        return _project_glm53_head(normalized, weights, workspace, process_group)

    def nextn_layer(
        self,
        embedding: torch.Tensor,
        target_normalized: torch.Tensor,
        weights: object,
        batch: SparseMLADecodeBatch[torch.Tensor] | GLM53StagedSparseMLAPrefillBatch,
        history: SparseMLAHistory[torch.Tensor],
        workspace: object,
        process_group: object,
        all_reduce: _AllReduce,
        transaction_tail_key: torch.Tensor | None = None,
        transaction_tail_gate: torch.Tensor | None = None,
        shared_selected: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (transaction_tail_key is None) != (transaction_tail_gate is None):
            raise ValueError("GLM NextN transaction tails must be provided together")
        _validate_nextn_layer(embedding, target_normalized, weights, workspace)
        w = workspace
        _rmsnorm(embedding, weights.embedding_norm, w.fusion[:, :HIDDEN_SIZE])
        _rmsnorm(target_normalized, weights.hidden_norm, w.fusion[:, HIDDEN_SIZE:])
        torch.mm(w.fusion, weights.fusion.T, out=w.residual)
        _rmsnorm(w.residual, weights.input_norm, w.normalized)
        projection = self.decode_sparse_mla_projection(
            w.normalized, weights.mla_projection, w.sparse_mla_projection
        )
        if isinstance(batch, GLM53StagedSparseMLAPrefillBatch):
            self.prefill_sparse_mla_history(
                projection, weights.mla_decode, batch, history
            )
            latent = self.prefill_sparse_mla_query(
                projection, weights.mla_decode, batch, history, w.sparse_mla_decode
            )
        elif transaction_tail_key is not None and transaction_tail_gate is not None:
            latent = self.verify_sparse_mla(
                projection,
                weights.mla_decode,
                batch,
                history,
                transaction_tail_key,
                transaction_tail_gate,
                w.sparse_mla_decode,
            )
        else:
            latent = self.decode_sparse_mla(
                projection,
                weights.mla_decode,
                batch,
                history,
                w.sparse_mla_decode,
                shared_selected,
            )
        attention = self.decode_sparse_mla_output(
            latent,
            weights.mla_output,
            w.sparse_mla_output,
            process_group,
            all_reduce,
        )
        w.residual.add_(attention)

        _rmsnorm(w.residual, weights.post_attention_norm, w.normalized)
        ffn = self.decode_sparse_ffn(
            w.normalized, weights.ffn, w.sparse_ffn, weights.tp_rank, all_reduce
        )
        w.residual.add_(ffn)
        return w.residual

    def decode_nextn_head(
        self,
        hidden_states: torch.Tensor,
        output_norm: torch.Tensor,
        weights: GLM53EndpointWeights[torch.Tensor],
        workspace: GLM53EndpointWorkspace[torch.Tensor],
        process_group: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = hidden_states.shape[0] if hidden_states.ndim == 2 else 0
        _validate_glm53_endpoint(weights, workspace, process_group, batch_size)
        _require_glm53_endpoint_tensors(
            weights.embedding.device,
            (hidden_states, torch.bfloat16, (batch_size, HIDDEN_SIZE)),
            (output_norm, torch.bfloat16, (HIDDEN_SIZE,)),
        )
        token = _decode_glm53_head(
            hidden_states, output_norm, weights, workspace, process_group
        )
        return token, workspace.normalized

    def prefill_dense_layer(
        self,
        streams: torch.Tensor,
        weights: GLM53LayerWeights[torch.Tensor],
        state: KDAState[torch.Tensor],
        batch: GLM53PrefillBatch,
        kda_workspace: KDAPrefillWorkspace[torch.Tensor],
        dense_workspace: GLM53DenseWorkspace[torch.Tensor],
        all_reduce: _AllReduce,
    ) -> torch.Tensor:
        w = dense_workspace
        rows = _prepare_dense_layer(streams, weights, w)
        attention_output = self.prefill_attention(
            w.normalized[rows],
            weights.attention,
            state,
            batch,
            kda_workspace,
        )
        return _finish_dense_layer(
            attention_output, streams, weights, w, rows, all_reduce
        )

    def decode_dense_layer(
        self,
        streams: torch.Tensor,
        weights: GLM53LayerWeights[torch.Tensor],
        state: KDAState[torch.Tensor],
        state_indices: torch.Tensor,
        cu_seqlens: torch.Tensor,
        kda_workspace: KDADecodeWorkspace[torch.Tensor],
        dense_workspace: GLM53DenseWorkspace[torch.Tensor],
        all_reduce: _AllReduce,
    ) -> torch.Tensor:
        w = dense_workspace
        rows = _prepare_dense_layer(streams, weights, w)
        attention_output = self.decode_attention(
            w.normalized[rows],
            weights.attention,
            state,
            state_indices,
            cu_seqlens,
            kda_workspace,
            all_reduce,
        )
        return _finish_dense_layer(
            attention_output, streams, weights, w, rows, all_reduce
        )

    def verify_dense_layer(
        self,
        streams: torch.Tensor,
        weights: GLM53LayerWeights[torch.Tensor],
        state: KDAState[torch.Tensor],
        transaction: KDAState[torch.Tensor],
        state_indices: torch.Tensor,
        kda_workspace: KDADecodeWorkspace[torch.Tensor],
        dense_workspace: GLM53DenseWorkspace[torch.Tensor],
        all_reduce: _AllReduce,
    ) -> torch.Tensor:
        rows = _prepare_dense_layer(streams, weights, dense_workspace)
        attention_output = _verify_kda_attention(
            dense_workspace.normalized[rows],
            weights.attention,
            state,
            transaction,
            state_indices,
            kda_workspace,
            all_reduce,
        )
        return _finish_dense_layer(
            attention_output, streams, weights, dense_workspace, rows, all_reduce
        )

    def prefill_sparse_kda_layer(
        self,
        streams: torch.Tensor,
        weights: GLM53SparseKDALayerWeights[torch.Tensor],
        state: KDAState[torch.Tensor],
        batch: GLM53PrefillBatch,
        kda_workspace: KDAPrefillWorkspace[torch.Tensor],
        ffn_workspace: SparseFFNDecodeWorkspace[torch.Tensor],
        layer_workspace: GLM53DenseWorkspace[torch.Tensor],
        process_group: object,
        all_reduce: _AllReduce,
    ) -> torch.Tensor:
        tep_rank = _sparse_tep_rank(process_group)

        total_tokens = _validate_sparse_kda_prefill_layer(
            streams,
            weights,
            state,
            batch,
            kda_workspace,
            ffn_workspace,
            layer_workspace,
            tep_rank,
        )
        w = layer_workspace
        rows = slice(total_tokens)
        _mhc_pre(streams, weights.attention_mhc, w, rows)
        _rmsnorm(w.collapsed[rows], weights.input_norm, w.normalized[rows])
        attention_output = self.prefill_attention(
            w.normalized[rows],
            weights.attention,
            state,
            batch,
            kda_workspace,
        )
        _mhc_post(
            attention_output,
            streams,
            w,
            w.streams_mid,
            rows,
        )
        _mhc_pre(w.streams_mid, weights.ffn_mhc, w, rows)
        _rmsnorm(w.collapsed[rows], weights.post_attention_norm, w.normalized[rows])
        ffn_output = self.decode_sparse_ffn(
            w.normalized[rows], weights.ffn, ffn_workspace, tep_rank, all_reduce
        )
        _mhc_post(ffn_output, w.streams_mid, w, w.streams_out, rows)
        return w.streams_out

    def decode_sparse_kda_layer(
        self,
        streams: torch.Tensor,
        weights: GLM53SparseKDALayerWeights[torch.Tensor],
        state: KDAState[torch.Tensor],
        state_indices: torch.Tensor,
        cu_seqlens: torch.Tensor,
        kda_workspace: KDADecodeWorkspace[torch.Tensor],
        ffn_workspace: SparseFFNDecodeWorkspace[torch.Tensor],
        layer_workspace: GLM53DenseWorkspace[torch.Tensor],
        process_group: object,
        all_reduce: _AllReduce,
    ) -> torch.Tensor:
        tep_rank = _sparse_tep_rank(process_group)

        rows = _validate_sparse_kda_layer(
            streams,
            weights,
            state,
            state_indices,
            cu_seqlens,
            kda_workspace,
            ffn_workspace,
            layer_workspace,
            tep_rank,
        )
        w = layer_workspace
        _mhc_pre(streams, weights.attention_mhc, w, rows)
        _rmsnorm(w.collapsed[rows], weights.input_norm, w.normalized[rows])
        attention_output = self.decode_attention(
            w.normalized[rows],
            weights.attention,
            state,
            state_indices,
            cu_seqlens,
            kda_workspace,
            all_reduce,
        )
        _mhc_post(attention_output, streams, w, w.streams_mid[rows], rows)
        _mhc_pre(w.streams_mid[rows], weights.ffn_mhc, w, rows)
        _rmsnorm(
            w.collapsed[rows],
            weights.post_attention_norm,
            w.normalized[rows],
        )
        ffn_output = self.decode_sparse_ffn(
            w.normalized[rows], weights.ffn, ffn_workspace, tep_rank, all_reduce
        )
        _mhc_post(ffn_output, w.streams_mid[rows], w, w.streams_out[rows], rows)
        return w.streams_out[rows]

    def verify_sparse_kda_layer(
        self,
        streams: torch.Tensor,
        weights: GLM53SparseKDALayerWeights[torch.Tensor],
        state: KDAState[torch.Tensor],
        transaction: KDAState[torch.Tensor],
        state_indices: torch.Tensor,
        kda_workspace: KDADecodeWorkspace[torch.Tensor],
        ffn_workspace: SparseFFNDecodeWorkspace[torch.Tensor],
        layer_workspace: GLM53DenseWorkspace[torch.Tensor],
        process_group: object,
        all_reduce: _AllReduce,
    ) -> torch.Tensor:
        tep_rank = _sparse_tep_rank(process_group)
        rows = _validate_sparse_layer_buffers(streams, layer_workspace)
        _validate_sparse_layer_shell(weights, streams.device)
        _validate_sparse_ffn(
            layer_workspace.normalized[rows], weights.ffn, ffn_workspace, tep_rank
        )
        w = layer_workspace
        _mhc_pre(streams, weights.attention_mhc, w, rows)
        _rmsnorm(w.collapsed[rows], weights.input_norm, w.normalized[rows])
        attention_output = _verify_kda_attention(
            w.normalized[rows],
            weights.attention,
            state,
            transaction,
            state_indices,
            kda_workspace,
            all_reduce,
        )
        _mhc_post(attention_output, streams, w, w.streams_mid[rows], rows)
        _mhc_pre(w.streams_mid[rows], weights.ffn_mhc, w, rows)
        _rmsnorm(w.collapsed[rows], weights.post_attention_norm, w.normalized[rows])
        ffn_output = self.decode_sparse_ffn(
            w.normalized[rows], weights.ffn, ffn_workspace, tep_rank, all_reduce
        )
        _mhc_post(ffn_output, w.streams_mid[rows], w, w.streams_out[rows], rows)
        return w.streams_out[rows]

    def prefill_sparse_mla_layer(
        self,
        streams: torch.Tensor,
        weights: GLM53SparseMLALayerWeights[torch.Tensor],
        staged_query: GLM53StagedSparseMLAPrefillBatch,
        history: SparseMLAHistory[torch.Tensor],
        projection_workspace: SparseMLAProjectionWorkspace[torch.Tensor],
        decode_workspace: SparseMLADecodeWorkspace[torch.Tensor],
        output_workspace: SparseMLAOutputWorkspace[torch.Tensor],
        ffn_workspace: SparseFFNDecodeWorkspace[torch.Tensor],
        layer_workspace: GLM53DenseWorkspace[torch.Tensor],
        process_group: object,
        all_reduce: _AllReduce,
    ) -> torch.Tensor:
        tep_rank = _sparse_tep_rank(process_group)

        total_tokens = _validate_sparse_mla_prefill_layer(
            streams,
            weights,
            staged_query,
            history,
            projection_workspace,
            decode_workspace,
            output_workspace,
            ffn_workspace,
            layer_workspace,
            process_group,
            tep_rank,
        )
        w = layer_workspace
        rows = slice(total_tokens)
        _mhc_pre(streams, weights.attention_mhc, w, rows)
        _rmsnorm(w.collapsed[rows], weights.input_norm, w.normalized[rows])
        projection = self.decode_sparse_mla_projection(
            w.normalized[rows], weights.mla_projection, projection_workspace
        )
        self.prefill_sparse_mla_history(
            projection,
            weights.mla_decode,
            staged_query,
            history,
        )
        latent = self.prefill_sparse_mla_query(
            projection,
            weights.mla_decode,
            staged_query,
            history,
            decode_workspace,
        )
        attention_output = self.decode_sparse_mla_output(
            latent,
            weights.mla_output,
            output_workspace,
            process_group,
            all_reduce,
        )
        _mhc_post(attention_output, streams, w, w.streams_mid, rows)
        _mhc_pre(w.streams_mid, weights.ffn_mhc, w, rows)
        _rmsnorm(w.collapsed[rows], weights.post_attention_norm, w.normalized[rows])
        ffn_output = self.decode_sparse_ffn(
            w.normalized[rows], weights.ffn, ffn_workspace, tep_rank, all_reduce
        )
        _mhc_post(ffn_output, w.streams_mid, w, w.streams_out, rows)
        return w.streams_out

    def decode_sparse_mla_layer(
        self,
        streams: torch.Tensor,
        weights: GLM53SparseMLALayerWeights[torch.Tensor],
        batch: SparseMLADecodeBatch[torch.Tensor],
        history: SparseMLAHistory[torch.Tensor],
        projection_workspace: SparseMLAProjectionWorkspace[torch.Tensor],
        decode_workspace: SparseMLADecodeWorkspace[torch.Tensor],
        output_workspace: SparseMLAOutputWorkspace[torch.Tensor],
        ffn_workspace: SparseFFNDecodeWorkspace[torch.Tensor],
        layer_workspace: GLM53DenseWorkspace[torch.Tensor],
        process_group: object,
        all_reduce: _AllReduce,
    ) -> torch.Tensor:
        tep_rank = _sparse_tep_rank(process_group)

        rows = _validate_sparse_mla_layer(
            streams,
            weights,
            batch,
            history,
            projection_workspace,
            decode_workspace,
            output_workspace,
            ffn_workspace,
            layer_workspace,
            process_group,
            tep_rank,
        )
        w = layer_workspace
        _mhc_pre(streams, weights.attention_mhc, w, rows)
        _rmsnorm(w.collapsed[rows], weights.input_norm, w.normalized[rows])
        projection = self.decode_sparse_mla_projection(
            w.normalized[rows], weights.mla_projection, projection_workspace
        )
        latent = self.decode_sparse_mla(
            projection,
            weights.mla_decode,
            batch,
            history,
            decode_workspace,
        )
        attention_output = self.decode_sparse_mla_output(
            latent,
            weights.mla_output,
            output_workspace,
            process_group,
            all_reduce,
        )
        _mhc_post(attention_output, streams, w, w.streams_mid[rows], rows)
        _mhc_pre(w.streams_mid[rows], weights.ffn_mhc, w, rows)
        _rmsnorm(
            w.collapsed[rows],
            weights.post_attention_norm,
            w.normalized[rows],
        )
        ffn_output = self.decode_sparse_ffn(
            w.normalized[rows], weights.ffn, ffn_workspace, tep_rank, all_reduce
        )
        _mhc_post(ffn_output, w.streams_mid[rows], w, w.streams_out[rows], rows)
        return w.streams_out[rows]

    def verify_sparse_mla_layer(
        self,
        streams: torch.Tensor,
        weights: GLM53SparseMLALayerWeights[torch.Tensor],
        batch: SparseMLADecodeBatch[torch.Tensor],
        history: SparseMLAHistory[torch.Tensor],
        transaction_tail_key: torch.Tensor,
        transaction_tail_gate: torch.Tensor,
        projection_workspace: SparseMLAProjectionWorkspace[torch.Tensor],
        decode_workspace: SparseMLADecodeWorkspace[torch.Tensor],
        output_workspace: SparseMLAOutputWorkspace[torch.Tensor],
        ffn_workspace: SparseFFNDecodeWorkspace[torch.Tensor],
        layer_workspace: GLM53DenseWorkspace[torch.Tensor],
        process_group: object,
        all_reduce: _AllReduce,
    ) -> torch.Tensor:
        tep_rank = _sparse_tep_rank(process_group)
        rows = _validate_sparse_mla_layer(
            streams,
            weights,
            batch,
            history,
            projection_workspace,
            decode_workspace,
            output_workspace,
            ffn_workspace,
            layer_workspace,
            process_group,
            tep_rank,
        )
        w = layer_workspace
        _mhc_pre(streams, weights.attention_mhc, w, rows)
        _rmsnorm(w.collapsed[rows], weights.input_norm, w.normalized[rows])
        projection = self.decode_sparse_mla_projection(
            w.normalized[rows], weights.mla_projection, projection_workspace
        )
        latent = self.verify_sparse_mla(
            projection,
            weights.mla_decode,
            batch,
            history,
            transaction_tail_key,
            transaction_tail_gate,
            decode_workspace,
        )
        attention_output = self.decode_sparse_mla_output(
            latent, weights.mla_output, output_workspace, process_group, all_reduce
        )
        _mhc_post(attention_output, streams, w, w.streams_mid[rows], rows)
        _mhc_pre(w.streams_mid[rows], weights.ffn_mhc, w, rows)
        _rmsnorm(w.collapsed[rows], weights.post_attention_norm, w.normalized[rows])
        ffn_output = self.decode_sparse_ffn(
            w.normalized[rows], weights.ffn, ffn_workspace, tep_rank, all_reduce
        )
        _mhc_post(ffn_output, w.streams_mid[rows], w, w.streams_out[rows], rows)
        return w.streams_out[rows]

    def decode_sparse_ffn(
        self,
        hidden_states: torch.Tensor,
        weights: SparseFFNWeights[torch.Tensor],
        workspace: SparseFFNDecodeWorkspace[torch.Tensor],
        tep_rank: int,
        all_reduce: _AllReduce,
    ) -> torch.Tensor:
        from flashinfer import tllm_enums
        from flashinfer.fused_moe import trtllm_fp8_block_scale_routed_moe

        w = workspace
        local_batch_size = hidden_states.shape[0] if hidden_states.ndim == 2 else 0
        moe_world_size = 1
        if self._attention_tp_size == 1:
            if self._moe_token_counts is None:
                raise RuntimeError("GLM DEP4 MoE token counts were not staged")
            local_capacity = max(self._moe_token_counts)
            if local_batch_size != self._moe_token_counts[tep_rank]:
                raise ValueError("hidden states disagree with staged MoE token counts")
            w.scattered.zero_()
            if local_batch_size:
                w.scattered[:local_batch_size].copy_(hidden_states)
            assert self._process_group is not None
            dist.all_gather_into_tensor(
                w.gathered,
                w.scattered,
                group=self._process_group,
            )
            hidden_states = w.gathered
            moe_world_size = TP_SIZE
        else:
            local_capacity = local_batch_size
        batch_size = _validate_sparse_ffn(
            hidden_states,
            weights,
            workspace,
            tep_rank,
            moe_world_size,
            local_capacity,
        )

        torch.mm(
            hidden_states,
            weights.router_t,
            out=w.scores,
            out_dtype=torch.float32,
        )
        _route_sparse_ffn_pre_topk[(batch_size,)](
            w.scores,
            weights.correction_bias,
            w.selection,
            NUM_EXPERTS=NUM_ROUTED_EXPERTS,
            BLOCK=triton.next_power_of_2(NUM_ROUTED_EXPERTS),
            num_warps=4,
        )
        torch.topk(
            w.selection,
            TOP_K,
            dim=1,
            sorted=False,
            out=(w.topk_values, w.topk_ids64),
        )
        _route_sparse_ffn_post_topk[(batch_size,)](
            w.scores,
            w.topk_ids64,
            w.topk_values,
            w.topk_ids,
            NUM_EXPERTS=NUM_ROUTED_EXPERTS,
            TOPK=TOP_K,
            ROUTE_SCALE=SPARSE_FFN_ROUTE_SCALE,
            num_warps=1,
        )

        hidden_scale = w.hidden_scale_mn.T
        _quantize(hidden_states, w.hidden_fp8, hidden_scale, zero_scale=1e-12)
        overlap = self._sparse_ffn_overlap
        if overlap is None:
            overlap = (torch.cuda.Stream(), torch.cuda.Event(), torch.cuda.Event())
            self._sparse_ffn_overlap = overlap
        current_stream = torch.cuda.current_stream()
        shared_stream, fork_event, join_event = overlap
        fork_event.record(current_stream)
        trtllm_fp8_block_scale_routed_moe(
            topk_ids=(w.topk_ids, w.topk_values),
            routing_bias=None,
            hidden_states=w.hidden_fp8,
            hidden_states_scale=w.hidden_scale_mn,
            gemm1_weights=weights.routed_up_gate,
            gemm1_weights_scale=weights.routed_up_gate_scale_inv,
            gemm2_weights=weights.routed_down,
            gemm2_weights_scale=weights.routed_down_scale_inv,
            num_experts=NUM_ROUTED_EXPERTS,
            top_k=TOP_K,
            n_group=1,
            topk_group=1,
            intermediate_size=MOE_INTERMEDIATE_SIZE,
            local_expert_offset=tep_rank * LOCAL_ROUTED_EXPERTS,
            local_num_experts=LOCAL_ROUTED_EXPERTS,
            routed_scaling_factor=None,
            routing_method_type=tllm_enums.RoutingMethodType.DeepSeekV3.value,
            use_shuffled_weight=False,
            weight_layout=tllm_enums.WeightLayout.MajorK.value,
            do_finalize=True,
            enable_pdl=True,
            output=w.routed,
            tune_max_num_tokens=max(SPARSE_FFN_DECODE_BATCH_SIZES) * moe_world_size,
            fp8_quantization_type=tllm_enums.Fp8QuantizationType.DeepSeekFp8,
            activation_type=tllm_enums.ActivationType.Swiglu.value,
            gemm1_clamp_limit=weights.routed_clamp,
        )
        with torch.cuda.stream(shared_stream):
            shared_stream.wait_event(fork_event)
            _fp8_gemm(
                w.hidden_fp8,
                weights.shared_gate_up,
                hidden_scale,
                weights.shared_gate_up_scale_inv,
                w.shared_gate_up,
            )
            shared_intermediate = MOE_INTERMEDIATE_SIZE // TP_SIZE
            shared_scale = _mn_scale_view(w.shared_scale)
            _swiglu_kernel[(batch_size, shared_intermediate // FP8_BLOCK_SIZE)](
                w.shared_gate_up,
                w.shared_fp8,
                shared_scale,
                WIDTH=shared_intermediate,
                BLOCK_SIZE=FP8_BLOCK_SIZE,
                FP8_LIMIT=448.0,
                LIMIT=SPARSE_FFN_CLAMP,
                QUANTIZE=True,
            )
            _fp8_gemm(
                w.shared_fp8,
                weights.shared_down,
                shared_scale,
                weights.shared_down_scale_inv,
                w.output,
            )
            join_event.record(shared_stream)
        current_stream.wait_event(join_event)
        w.output.add_(w.routed)
        if moe_world_size > 1:
            dist.reduce_scatter_tensor(
                w.scattered,
                w.output,
                group=self._process_group,
            )
            return _sparse_ffn_local_output(w, local_batch_size)
        return all_reduce(w.output)

    def decode_sparse_mla_projection(
        self,
        hidden_states: torch.Tensor,
        weights: SparseMLAProjectionWeights[torch.Tensor],
        workspace: SparseMLAProjectionWorkspace[torch.Tensor],
    ) -> SparseMLAProjectionWorkspace[torch.Tensor]:
        batch_size = hidden_states.shape[0]
        if not 1 <= batch_size <= KDA_CHUNK_SIZE or hidden_states.shape != (
            batch_size,
            HIDDEN_SIZE,
        ):
            raise ValueError(
                f"hidden_states must have shape [N, {HIDDEN_SIZE}] for "
                f"1 <= N <= {KDA_CHUNK_SIZE}, "
                f"got {tuple(hidden_states.shape)}"
            )
        if hidden_states.dtype != torch.bfloat16:
            raise TypeError(
                f"hidden_states must be bfloat16, got {hidden_states.dtype}"
            )
        if not hidden_states.is_contiguous():
            raise ValueError("hidden_states must be contiguous")

        w = workspace
        local_heads = weights.q_b.shape[0] // SPARSE_MLA_QK_NOPE_HEAD_DIM
        attention_tp_size = _attention_tp_size_from_heads(local_heads)
        expected = sparse_mla_projection_workspace_shapes(batch_size, attention_tp_size)
        workspace_tensors = (
            ("hidden_fp8", w.hidden_fp8, torch.float8_e4m3fn),
            ("hidden_scale", w.hidden_scale, torch.float32),
            ("low_rank", w.low_rank, torch.bfloat16),
            ("q_resid", w.q_resid, torch.bfloat16),
            ("latent", w.latent, torch.bfloat16),
            ("q_resid_fp8", w.q_resid_fp8, torch.float8_e4m3fn),
            ("q_resid_scale", w.q_resid_scale, torch.float32),
            ("main_q", w.main_q, torch.bfloat16),
            ("index_q", w.index_q, torch.bfloat16),
            ("index_prep", w.index_prep, torch.bfloat16),
            ("key", w.key, torch.bfloat16),
            ("pool_gate", w.pool_gate, torch.bfloat16),
            ("score_weights", w.score_weights, torch.float32),
        )
        for name, tensor, dtype in workspace_tensors:
            if tensor.shape != getattr(expected, name):
                raise ValueError(f"invalid sparse MLA {name} workspace shape")
            if tensor.dtype != dtype:
                raise TypeError(f"invalid sparse MLA {name} workspace dtype")
            if tensor.device != hidden_states.device or not tensor.is_contiguous():
                raise ValueError(f"invalid sparse MLA {name} workspace layout")

        hidden_scale = _mn_scale_view(w.hidden_scale)
        _quantize(hidden_states, w.hidden_fp8, hidden_scale)
        _fp8_gemm(
            w.hidden_fp8,
            weights.low_rank,
            hidden_scale,
            weights.low_rank_scale_inv,
            w.low_rank,
        )
        _rmsnorm(w.low_rank[:, :SPARSE_MLA_Q_LORA_RANK], weights.q_norm, w.q_resid)
        _rmsnorm(w.low_rank[:, SPARSE_MLA_Q_LORA_RANK:], weights.kv_norm, w.latent)
        torch.mm(hidden_states, weights.index_prep.t(), out=w.index_prep)
        _layernorm_bf16[(batch_size,)](
            w.index_prep,
            weights.k_norm,
            weights.k_bias,
            w.key,
            ROW_WIDTH=2 * SPARSE_MLA_INDEXER_HEAD_DIM + SPARSE_MLA_INDEXER_HEADS,
            WIDTH=SPARSE_MLA_INDEXER_HEAD_DIM,
            EPS=SPARSE_MLA_LAYER_NORM_EPS,
        )
        w.pool_gate.copy_(
            w.index_prep[
                :,
                SPARSE_MLA_INDEXER_HEAD_DIM : 2 * SPARSE_MLA_INDEXER_HEAD_DIM,
            ]
        )
        w.score_weights.copy_(w.index_prep[:, 2 * SPARSE_MLA_INDEXER_HEAD_DIM :]).mul_(
            SPARSE_MLA_INDEX_WEIGHT_SCALE
        )
        q_resid_scale = _mn_scale_view(w.q_resid_scale)
        _quantize(w.q_resid, w.q_resid_fp8, q_resid_scale)
        _fp8_gemm(
            w.q_resid_fp8,
            weights.q_b,
            q_resid_scale,
            weights.q_b_scale_inv,
            w.main_q,
        )
        torch.mm(w.q_resid, weights.wq_b.t(), out=w.index_q)
        return w

    def prefill_sparse_mla_history(
        self,
        projection: SparseMLAProjectionWorkspace[torch.Tensor],
        weights: SparseMLADecodeWeights[torch.Tensor],
        staged: GLM53StagedSparseMLAPrefillBatch,
        history: SparseMLAHistory[torch.Tensor],
    ) -> SparseMLAHistory[torch.Tensor]:
        rows = slice(staged.total_tokens)
        return glm53_sparse_mla_packed_prefill_append(
            batch=staged,
            latent=projection.latent[rows],
            key=projection.key[rows],
            gate=projection.pool_gate[rows],
            pool_ape=weights.pool_ape,
            history=history,
        )

    def prefill_sparse_mla_query(
        self,
        projection: SparseMLAProjectionWorkspace[torch.Tensor],
        weights: SparseMLADecodeWeights[torch.Tensor],
        staged_query: GLM53StagedSparseMLAPrefillBatch,
        history: SparseMLAHistory[torch.Tensor],
        workspace: SparseMLADecodeWorkspace[torch.Tensor],
    ) -> torch.Tensor:
        batch_size = _validate_sparse_mla_prefill_query(
            projection, weights, staged_query, history, workspace
        )
        runtime = _sparse_mla_runtime()
        _prepare_sparse_mla_query_lengths[(batch_size,)](
            staged_query.active,
            staged_query.raw_lengths,
            workspace.context_lengths,
            workspace.sequence_lengths,
        )
        return _run_sparse_mla_query(
            projection,
            weights,
            staged_query,
            history,
            workspace,
            batch_size,
            runtime,
        )

    def decode_sparse_mla(
        self,
        projection: SparseMLAProjectionWorkspace[torch.Tensor],
        weights: SparseMLADecodeWeights[torch.Tensor],
        batch: SparseMLADecodeBatch[torch.Tensor],
        history: SparseMLAHistory[torch.Tensor],
        workspace: SparseMLADecodeWorkspace[torch.Tensor],
        shared_selected: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = _validate_sparse_mla_decode(
            projection, weights, batch, history, workspace
        )
        if shared_selected is not None:
            _require_sparse_mla_tensors(
                projection.latent.device,
                (
                    shared_selected,
                    torch.int32,
                    (batch_size, SPARSE_MLA_INDEX_TOP_K),
                ),
            )
        deep_gemm, flashinfer, topk, tie_break = _sparse_mla_runtime()
        runtime = deep_gemm, flashinfer, topk, tie_break
        w = workspace

        _write_sparse_mla_history[(batch_size,)](
            batch.active,
            batch.raw_lengths,
            batch.state_slots,
            batch.block_table,
            projection.latent,
            projection.key,
            projection.pool_gate,
            weights.pool_ape,
            history.latent,
            history.index_cache,
            history.tail_key,
            history.tail_gate,
            w.context_lengths,
            w.sequence_lengths,
            history.tail_key,
            history.tail_gate,
            LIVE_SLOTS=history.tail_key.shape[1],
            ROW_START=0,
            ROW_STRIDE=1,
            WRITE_TRANSACTION=False,
        )
        return _run_sparse_mla_query(
            projection,
            weights,
            batch,
            history,
            workspace,
            batch_size,
            runtime,
            shared_selected,
        )

    def verify_sparse_mla(
        self,
        projection: SparseMLAProjectionWorkspace[torch.Tensor],
        weights: SparseMLADecodeWeights[torch.Tensor],
        batch: SparseMLADecodeBatch[torch.Tensor],
        history: SparseMLAHistory[torch.Tensor],
        transaction_tail_key: torch.Tensor,
        transaction_tail_gate: torch.Tensor,
        workspace: SparseMLADecodeWorkspace[torch.Tensor],
    ) -> torch.Tensor:
        batch_size = _validate_sparse_mla_decode(
            projection, weights, batch, history, workspace
        )
        if batch_size % GLM53_TARGET_VERIFY_WIDTH:
            raise ValueError("GLM sparse MLA verification rows must form Bx4 groups")
        tail_shape = (
            batch_size,
            2,
            1,
            SPARSE_MLA_INDEX_POOL_TOKENS,
            SPARSE_MLA_INDEXER_HEAD_DIM,
        )
        _require_sparse_mla_tensors(
            projection.latent.device,
            (transaction_tail_key, torch.bfloat16, tail_shape),
            (transaction_tail_gate, torch.bfloat16, tail_shape),
        )
        groups = batch_size // GLM53_TARGET_VERIFY_WIDTH
        for row in range(GLM53_TARGET_VERIFY_WIDTH):
            _write_sparse_mla_history[(groups,)](
                batch.active,
                batch.raw_lengths,
                batch.state_slots,
                batch.block_table,
                projection.latent,
                projection.key,
                projection.pool_gate,
                weights.pool_ape,
                history.latent,
                history.index_cache,
                history.tail_key,
                history.tail_gate,
                workspace.context_lengths,
                workspace.sequence_lengths,
                transaction_tail_key,
                transaction_tail_gate,
                LIVE_SLOTS=history.tail_key.shape[1],
                ROW_START=row,
                ROW_STRIDE=GLM53_TARGET_VERIFY_WIDTH,
                WRITE_TRANSACTION=True,
            )
        return _run_sparse_mla_query(
            projection,
            weights,
            batch,
            history,
            workspace,
            batch_size,
            _sparse_mla_runtime(),
        )

    def decode_sparse_mla_output(
        self,
        latent: torch.Tensor,
        weights: SparseMLAOutputWeights[torch.Tensor],
        workspace: SparseMLAOutputWorkspace[torch.Tensor],
        process_group: object,
        all_reduce: _AllReduce,
    ) -> torch.Tensor:
        batch_size = _validate_sparse_mla_output(
            latent, weights, workspace, process_group
        )
        w = workspace

        torch.bmm(
            latent.transpose(0, 1),
            weights.w_vc,
            out=w.value_hbd,
        )
        local_heads = weights.w_vc.shape[0]
        w.projected.view(batch_size, local_heads, SPARSE_MLA_VALUE_HEAD_DIM).copy_(
            w.value_hbd.transpose(0, 1)
        )
        projected_scale = _mn_scale_view(w.projected_scale)
        _quantize(w.projected, w.projected_fp8, projected_scale)
        _fp8_gemm(
            w.projected_fp8,
            weights.o_proj,
            projected_scale,
            weights.o_proj_scale_inv,
            w.output,
        )
        if local_heads == NUM_HEADS:
            return w.output
        return all_reduce(w.output)

    def prefill_attention(
        self,
        hidden_states: torch.Tensor,
        weights: KDAWeights[torch.Tensor],
        state: KDAState[torch.Tensor],
        batch: GLM53PrefillBatch,
        workspace: KDAPrefillWorkspace[torch.Tensor],
    ) -> torch.Tensor:
        total_tokens = _validate_kda_prefill(
            hidden_states, weights, state, batch, workspace
        )
        batch_size = batch.metadata.batch_size
        rows = slice(total_tokens)
        state_rows = slice(batch_size)
        projection = workspace.projection[rows]
        gates = workspace.gates[:, rows]
        beta = workspace.beta[:, rows]
        qkv = workspace.qkv[:, :, rows]
        initial_state = workspace.initial_state[state_rows]
        kda_output = workspace.kda_output[:, rows]
        final_state = workspace.final_state[state_rows]
        output = workspace.output[rows]
        local_heads = weights.a_log.shape[0]
        local_projection_size = local_heads * HEAD_DIM
        packed_qkv_size = 3 * local_projection_size

        torch.mm(hidden_states, weights.projection.t(), out=projection)
        torch.bmm(
            projection[:, packed_qkv_size : packed_qkv_size + 2 * HEAD_DIM]
            .view(total_tokens, 2, HEAD_DIM)
            .transpose(0, 1),
            weights.gate_projections.transpose(1, 2),
            out=gates,
        )
        beta[0].copy_(projection[:, -local_heads:])
        glm53_segmented_conv_prefill(
            qkv_raw=projection[:, :packed_qkv_size],
            conv_weight=weights.conv,
            conv_state=state.conv,
            cu_seqlens=batch.metadata.cu_seqlens,
            state_indices=batch.state_indices,
            has_initial=batch.has_initial,
            segments=batch.segments,
            out=qkv,
        )

        torch.index_select(
            state.recurrent,
            0,
            batch.state_indices_int64,
            out=initial_state,
        )
        final_state.zero_()
        torch.where(
            batch.has_initial.view(batch_size, 1, 1, 1),
            initial_state,
            final_state,
            out=initial_state,
        )
        core_output, final_state = glm53_prefill_kda(
            q=qkv[0],
            k=qkv[1],
            v=qkv[2],
            gate=gates[0].view(1, total_tokens, local_heads, HEAD_DIM),
            beta_logits=beta,
            a_log=weights.a_log,
            dt_bias=weights.dt_bias,
            initial_state=initial_state,
            metadata=batch.metadata,
            out=kda_output,
            final_state=final_state,
            workspace=workspace.kda_workspace,
        )
        state.recurrent.index_copy_(0, batch.state_indices_int64, final_state)

        _rmsnorm_sigmoid_gate[(total_tokens * local_heads,)](
            core_output,
            gates[1],
            weights.o_norm,
            core_output,
            WIDTH=HEAD_DIM,
            EPS=RMS_NORM_EPS,
        )
        torch.mm(
            core_output.view(total_tokens, local_projection_size),
            weights.o_projection.t(),
            out=output,
        )
        if local_heads != NUM_HEADS:
            dist.all_reduce(output)
        return output

    def decode_attention(
        self,
        hidden_states: torch.Tensor,
        weights: KDAWeights[torch.Tensor],
        state: KDAState[torch.Tensor],
        state_indices: torch.Tensor,
        cu_seqlens: torch.Tensor,
        workspace: KDADecodeWorkspace[torch.Tensor],
        all_reduce: _AllReduce,
    ) -> torch.Tensor:
        _validate_kda_decode(
            hidden_states,
            weights,
            state,
            state_indices,
            cu_seqlens,
            workspace,
        )

        local_heads = weights.a_log.shape[0]
        local_projection_size = local_heads * HEAD_DIM
        packed_qkv_size = 3 * local_projection_size
        torch.mm(hidden_states, weights.projection.t(), out=workspace.projection)
        torch.mm(
            workspace.projection[
                :, packed_qkv_size + HEAD_DIM : packed_qkv_size + 2 * HEAD_DIM
            ],
            weights.gate_projections[1].t(),
            out=workspace.output_gate,
        )
        workspace.local_output.zero_()
        glm53_megafuse_decode(
            qkv_raw=workspace.projection[:, :packed_qkv_size],
            conv_weight=weights.conv,
            conv_state=state.conv,
            f_a=workspace.projection[:, packed_qkv_size : packed_qkv_size + HEAD_DIM],
            f_b_weight=weights.gate_projections[0],
            beta_logits=workspace.projection[:, -local_heads:],
            a_log=weights.a_log,
            dt_bias=weights.dt_bias,
            output_gate=workspace.output_gate,
            norm_weight=weights.o_norm,
            recurrent_state=state.recurrent,
            state_indices=state_indices,
            cu_seqlens=cu_seqlens,
            out=workspace.local_output,
        )
        return _project_kda_output(
            workspace.local_output, weights, workspace, all_reduce
        )


def _verify_kda_attention(
    hidden_states,
    weights,
    state,
    transaction,
    state_indices,
    workspace,
    all_reduce,
):
    batch_size = _validate_kda_inputs(hidden_states, weights, state)
    local_heads = weights.a_log.shape[0]
    local_projection_size = local_heads * HEAD_DIM
    packed_qkv_size = 3 * local_projection_size
    if (
        batch_size % GLM53_TARGET_VERIFY_WIDTH
        or state_indices.shape != (batch_size // GLM53_TARGET_VERIFY_WIDTH,)
        or transaction.recurrent.shape[0] != batch_size
    ):
        raise ValueError("KDA target verify requires request-major Bx4 transactions")

    torch.mm(hidden_states, weights.projection.t(), out=workspace.projection)
    torch.mm(
        workspace.projection[
            :, packed_qkv_size + HEAD_DIM : packed_qkv_size + 2 * HEAD_DIM
        ],
        weights.gate_projections[1].t(),
        out=workspace.output_gate,
    )
    workspace.local_output.zero_()
    active = slice(batch_size)
    glm53_megafuse_verify(
        qkv_raw=workspace.projection[active, :packed_qkv_size],
        conv_weight=weights.conv,
        conv_state=state.conv,
        conv_tape=transaction.conv,
        f_a=workspace.projection[active, packed_qkv_size : packed_qkv_size + HEAD_DIM],
        f_b_weight=weights.gate_projections[0],
        gate_raw=workspace.gate_raw[active],
        beta_logits=workspace.projection[active, -local_heads:],
        a_log=weights.a_log,
        dt_bias=weights.dt_bias,
        recurrent_state=state.recurrent,
        recurrent_tape=transaction.recurrent,
        state_indices=state_indices,
        out=workspace.local_output[active],
    )
    _rmsnorm_sigmoid_gate[(batch_size * local_heads,)](
        workspace.local_output[active],
        workspace.output_gate[active],
        weights.o_norm,
        workspace.local_output[active],
        WIDTH=HEAD_DIM,
        EPS=RMS_NORM_EPS,
    )
    return _project_kda_output(workspace.local_output, weights, workspace, all_reduce)


def _project_kda_output(local_output, weights, workspace, all_reduce):
    local_projection_size = local_output.shape[1] * HEAD_DIM
    torch.mm(
        local_output.view(local_output.shape[0], local_projection_size),
        weights.o_projection.t(),
        out=workspace.output,
    )
    if local_output.shape[1] == NUM_HEADS:
        return workspace.output
    return all_reduce(workspace.output)


def _run_sparse_mla_query(
    projection,
    weights,
    batch,
    history,
    workspace,
    batch_size,
    runtime,
    shared_selected=None,
):
    deep_gemm, flashinfer, topk, tie_break = runtime
    w = workspace
    local_heads = weights.w_kc.shape[0]
    w.index_q.copy_(
        projection.index_q.view(
            batch_size,
            1,
            SPARSE_MLA_INDEXER_HEADS,
            SPARSE_MLA_INDEXER_HEAD_DIM,
        )
    )
    w.score_weights.copy_(projection.score_weights).mul_(SPARSE_MLA_INDEX_SCORE_SCALE)
    w.main_q_hbd.copy_(
        projection.main_q.view(
            batch_size, local_heads, SPARSE_MLA_QK_NOPE_HEAD_DIM
        ).transpose(0, 1)
    )
    torch.bmm(w.main_q_hbd, weights.w_kc, out=w.absorbed_hbd)
    w.attention_q[:, 0].copy_(w.absorbed_hbd.transpose(0, 1))

    if shared_selected is None:
        if (
            deep_gemm.get_paged_mqa_logits_metadata_out(
                w.context_lengths,
                w.schedule,
                SPARSE_MLA_HISTORY_BLOCK_TOKENS // SPARSE_MLA_INDEX_POOL_TOKENS,
                SPARSE_MLA_B200_SMS,
                indices=None,
            )
            is not None
        ):
            raise RuntimeError("DeepGEMM metadata out-ABI returned a value")
        if (
            deep_gemm.fp8_paged_mqa_logits_out(
                w.index_q,
                history.index_cache.view(
                    -1,
                    SPARSE_MLA_HISTORY_BLOCK_TOKENS // SPARSE_MLA_INDEX_POOL_TOKENS,
                    1,
                    SPARSE_MLA_INDEXER_HEAD_DIM + 4,
                ),
                w.score_weights,
                w.context_lengths,
                batch.block_table,
                w.schedule,
                w.logits,
                SPARSE_MLA_INDEX_CONTEXT_CAPACITY,
                clean_logits=False,
                indices=None,
            )
            is not None
        ):
            raise RuntimeError("DeepGEMM logits out-ABI returned a value")
        topk.radix_topk_ragged_transform(
            w.logits,
            w.selected,
            w.topk_offsets,
            w.context_lengths.view(-1),
            w.topk_rows,
            SPARSE_MLA_INDEX_TOP_K,
            True,
            tie_break,
            dsa_graph_safe=True,
            row_starts=None,
        )
    else:
        w.selected.copy_(shared_selected)
    _map_sparse_mla_ids[(batch_size, triton.cdiv(SPARSE_MLA_SPARSE_CAPACITY, 256))](
        batch.active,
        batch.raw_lengths,
        batch.block_table,
        batch.null_token,
        w.selected,
        w.sparse_ids,
        w.sparse_lengths,
    )
    flashinfer.mla.trtllm_batch_decode_with_kv_cache_mla(
        query=w.attention_q,
        kv_cache=history.latent,
        workspace_buffer=w.flashinfer_workspace,
        qk_nope_head_dim=SPARSE_MLA_QK_NOPE_HEAD_DIM,
        kv_lora_rank=SPARSE_MLA_KV_LORA_RANK,
        qk_rope_head_dim=0,
        block_tables=w.sparse_ids,
        seq_lens=w.sequence_lengths,
        max_seq_len=SPARSE_MLA_MAX_CONTEXT_TOKENS,
        sparse_mla_top_k=SPARSE_MLA_SPARSE_CAPACITY,
        sparse_mla_top_k_lens=w.sparse_lengths,
        out=w.output,
        bmm1_scale=1 / local_heads,
        bmm2_scale=1,
        backend="trtllm-gen",
        enable_pdl=True,
        is_var_seq=True,
        uses_shared_paged_kv_idx=True,
        multi_ctas_kv_counter_buffer=w.counter,
    )
    torch.mul(
        w.output,
        batch.active.view(batch_size, 1, 1, 1),
        out=w.output,
    )
    return w.output[:, 0]


@cache
def _sparse_mla_runtime():
    import cutlass
    import deep_gemm
    import flashinfer
    from flashinfer.jit import gen_trtllm_gen_fmha_module
    from flashinfer.jit.topk import gen_topk_module
    from flashinfer.topk import TopKTieBreak, get_topk_module

    packages = {name: version(name) for name in _SPARSE_MLA_FLASHINFER_PACKAGES}
    cutlass_cuda = (cutlass.CUDA_VERSION.major, cutlass.CUDA_VERSION.minor)
    if (
        packages != _SPARSE_MLA_FLASHINFER_PACKAGES
        or cutlass_cuda != (12, 9)
        or (
            flashinfer.__git_commit__,
            torch.__version__,
            torch.version.cuda,
        )
        != (
            _SPARSE_MLA_FLASHINFER_COMMIT,
            "2.13.0+cu129",
            "12.9",
        )
    ):
        raise RuntimeError("GLM sparse MLA dependency identity mismatch")
    extension = Path(deep_gemm.__file__).resolve().parent / "_C.so"
    with extension.open("rb") as binary:
        digest = hashlib.file_digest(binary, "sha256").hexdigest()
    if digest != _SPARSE_MLA_DEEP_GEMM_SHA256:
        raise RuntimeError("GLM sparse MLA DeepGEMM binary identity mismatch")
    symbols = (
        "get_num_sms",
        "get_paged_mqa_logits_metadata_out",
        "fp8_paged_mqa_logits_out",
        "tf32_hc_prenorm_gemm",
        "set_pdl",
    )
    if not all(callable(getattr(deep_gemm, name, None)) for name in symbols):
        raise TypeError("GLM sparse MLA native ABI is unavailable")
    if (
        not gen_topk_module().is_aot
        or not gen_trtllm_gen_fmha_module().is_aot
        or deep_gemm.get_num_sms() != SPARSE_MLA_B200_SMS
    ):
        raise RuntimeError("GLM sparse MLA AOT/B200 runtime is unavailable")
    deep_gemm.set_pdl(True)
    return deep_gemm, flashinfer, get_topk_module(), int(TopKTieBreak.SMALL)


def _validate_glm53_endpoint(weights, workspace, group, batch_size: int = 1) -> None:
    if dist.get_world_size(group) != TP_SIZE:
        raise ValueError("GLM endpoint requires a four-rank process group")
    local_vocab_size = weights.embedding.shape[0]
    if local_vocab_size not in {LOCAL_VOCAB_SIZE, VOCAB_SIZE}:
        raise ValueError("GLM endpoint has an unsupported vocabulary placement")
    attention_tp_size = VOCAB_SIZE // local_vocab_size
    expected = glm53_endpoint_workspace_shapes(batch_size, attention_tp_size)
    device = weights.embedding.device
    _require_glm53_endpoint_tensors(
        device,
        (weights.embedding, torch.bfloat16, (local_vocab_size, HIDDEN_SIZE)),
        (weights.final_norm, torch.bfloat16, (HIDDEN_SIZE,)),
        (weights.lm_head, torch.bfloat16, (local_vocab_size, HIDDEN_SIZE)),
        (workspace.local_token, torch.int64, expected.local_token),
        (workspace.local_active, torch.bool, expected.local_active),
        (workspace.embedding, torch.bfloat16, expected.embedding),
        (workspace.streams, torch.bfloat16, expected.streams),
        (workspace.mean_f32, torch.float32, expected.mean_f32),
        (workspace.collapsed, torch.bfloat16, expected.collapsed),
        (workspace.normalized, torch.bfloat16, expected.normalized),
        (workspace.local_logits, torch.bfloat16, expected.local_logits),
        (workspace.token, torch.int64, expected.token),
    )
    if weights.embedding.data_ptr() == weights.lm_head.data_ptr():
        raise ValueError("GLM embedding and LM head weights must be distinct")


def _validate_nextn_layer(embedding, target_normalized, weights, workspace) -> None:
    if not 0 <= weights.tp_rank < TP_SIZE:
        raise ValueError(f"invalid NextN TP rank {weights.tp_rank}")
    batch_size = embedding.shape[0] if embedding.ndim == 2 else 0
    if not 1 <= batch_size <= KDA_CHUNK_SIZE:
        raise ValueError("unsupported GLM NextN batch size")
    device = embedding.device
    _require_sparse_mla_tensors(
        device,
        (embedding, torch.bfloat16, (batch_size, HIDDEN_SIZE)),
        (target_normalized, torch.bfloat16, (batch_size, HIDDEN_SIZE)),
        (weights.embedding_norm, torch.bfloat16, (HIDDEN_SIZE,)),
        (weights.hidden_norm, torch.bfloat16, (HIDDEN_SIZE,)),
        (weights.fusion, torch.bfloat16, (HIDDEN_SIZE, 2 * HIDDEN_SIZE)),
        (weights.input_norm, torch.bfloat16, (HIDDEN_SIZE,)),
        (weights.post_attention_norm, torch.bfloat16, (HIDDEN_SIZE,)),
        (workspace.fusion, torch.bfloat16, (batch_size, 2 * HIDDEN_SIZE)),
        (workspace.residual, torch.bfloat16, (batch_size, HIDDEN_SIZE)),
        (workspace.normalized, torch.bfloat16, (batch_size, HIDDEN_SIZE)),
    )


def _require_glm53_endpoint_tensors(device, *specifications) -> None:
    for tensor, dtype, shape in specifications:
        if tensor.shape != shape or tensor.dtype != dtype:
            raise ValueError("GLM endpoint tensor has the wrong shape or dtype")
        if tensor.device != device or not tensor.is_cuda or not tensor.is_contiguous():
            raise ValueError(
                "GLM endpoint tensors must be contiguous on one CUDA device"
            )


def _decode_glm53_head(hidden_states, norm, weights, workspace, process_group):
    _rmsnorm(hidden_states, norm, workspace.normalized)
    return _project_glm53_head(workspace.normalized, weights, workspace, process_group)


def _project_glm53_head(normalized, weights, workspace, process_group):
    torch.mm(normalized, weights.lm_head.t(), out=workspace.local_logits)
    if weights.lm_head.shape[0] == VOCAB_SIZE:
        torch.argmax(
            workspace.local_logits[:, :TOKENIZER_VOCAB_SIZE],
            dim=1,
            out=workspace.token,
        )
        return workspace.token
    glm53_distributed_argmax(
        workspace.local_logits,
        workspace.token,
        workspace.argmax,
        process_group,
    )
    return workspace.token


def _attention_tp_size_from_heads(local_heads: int) -> int:
    if local_heads not in {glm53_local_heads(1), glm53_local_heads(TP_SIZE)}:
        raise ValueError("GLM attention weights have an unsupported head placement")
    return NUM_HEADS // local_heads


def _validate_kda_inputs(hidden_states, weights, state) -> int:
    hidden_shape = tuple(hidden_states.shape)
    if len(hidden_shape) != 2 or hidden_shape[0] < 1 or hidden_shape[1] != HIDDEN_SIZE:
        raise ValueError(f"hidden_states must have shape [N, {HIDDEN_SIZE}]")
    recurrent_shape = tuple(state.recurrent.shape)
    if len(recurrent_shape) != 4 or recurrent_shape[0] < 1:
        raise ValueError("KDA state must contain at least one page")
    local_heads = weights.a_log.shape[0]
    _attention_tp_size_from_heads(local_heads)
    local_projection_size = local_heads * HEAD_DIM
    packed_qkv_size = 3 * local_projection_size
    packed_projection_size = packed_qkv_size + 2 * HEAD_DIM + local_heads
    weight_specs = (
        ("projection", torch.bfloat16, (packed_projection_size, HIDDEN_SIZE)),
        (
            "gate_projections",
            torch.bfloat16,
            (2, local_projection_size, HEAD_DIM),
        ),
        ("conv", torch.float32, (packed_qkv_size, CONV_KERNEL_SIZE)),
        ("a_log", torch.float32, (local_heads,)),
        ("dt_bias", torch.float32, (local_projection_size,)),
        ("o_norm", torch.bfloat16, (HEAD_DIM,)),
        ("o_projection", torch.bfloat16, (HIDDEN_SIZE, local_projection_size)),
    )
    _require_kda_decode_tensors(
        hidden_states.device,
        ("hidden_states", hidden_states, torch.bfloat16, hidden_shape),
        *(
            (f"weights.{name}", getattr(weights, name), dtype, shape)
            for name, dtype, shape in weight_specs
        ),
        (
            "state.recurrent",
            state.recurrent,
            torch.float32,
            (recurrent_shape[0], local_heads, HEAD_DIM, HEAD_DIM),
        ),
        (
            "state.conv",
            state.conv,
            torch.bfloat16,
            (recurrent_shape[0], packed_qkv_size, CONV_KERNEL_SIZE - 1),
        ),
    )
    return hidden_shape[0]


def _validate_kda_prefill(hidden_states, weights, state, batch, workspace) -> int:
    if type(batch) is not GLM53PrefillBatch:
        raise TypeError("batch must be a GLM53PrefillBatch")
    batch.metadata.validate_unchanged()
    total_tokens = _validate_kda_inputs(hidden_states, weights, state)
    batch_size = batch.metadata.batch_size
    if total_tokens != batch.metadata.total_tokens:
        raise ValueError("hidden_states and KDA prefill metadata disagree")
    state_capacity = workspace.initial_state.shape[0]
    if total_tokens > workspace.projection.shape[0] or state_capacity < batch_size:
        raise ValueError("KDA prefill workspace has insufficient capacity")
    attention_tp_size = _attention_tp_size_from_heads(weights.a_log.shape[0])
    expected = kda_prefill_workspace_shapes(
        workspace.projection.shape[0], state_capacity, attention_tp_size
    )
    _require_kda_decode_tensors(
        hidden_states.device,
        ("batch.cu_seqlens", batch.metadata.cu_seqlens, torch.int32, (batch_size + 1,)),
        (
            "batch.cu_seqlens_int64",
            batch.metadata.cu_seqlens_int64,
            torch.int64,
            (batch_size + 1,),
        ),
        ("batch.state_indices", batch.state_indices, torch.int32, (batch_size,)),
        (
            "batch.state_indices_int64",
            batch.state_indices_int64,
            torch.int64,
            (batch_size,),
        ),
        ("batch.has_initial", batch.has_initial, torch.bool, (batch_size,)),
        ("batch.segments", batch.segments, torch.int32, tuple(batch.segments.shape)),
        *(
            (
                f"workspace.{name}",
                getattr(workspace, name),
                torch.uint8
                if name in ("kda_workspace_storage", "kda_workspace")
                else torch.float32
                if name in ("initial_state", "final_state")
                else torch.bfloat16,
                getattr(expected, name),
            )
            for name in workspace.__dataclass_fields__
        ),
    )
    if batch.segments.ndim != 2 or batch.segments.shape[1] != 2:
        raise ValueError("batch.segments must have shape [programs, 2]")
    if workspace.kda_workspace.data_ptr() % 128:
        raise ValueError("KDA prefill workspace must be 128-byte aligned")
    return total_tokens


def _validate_sparse_mla_prefill_query(
    projection, weights, staged_query, history, workspace
) -> int:
    if type(staged_query) is not GLM53StagedSparseMLAPrefillBatch:
        raise TypeError("staged_query must be a GLM sparse MLA prefill receipt")
    staged_query.validate_unchanged()
    batch_size = staged_query.active.shape[0]
    device = projection.main_q.device
    local_heads = weights.w_kc.shape[0]
    attention_tp_size = _attention_tp_size_from_heads(local_heads)
    projection_shapes = sparse_mla_projection_workspace_shapes(
        batch_size, attention_tp_size
    )
    history_blocks = staged_query.history_blocks
    live_slots = staged_query.live_slots
    tail_shape = (2, live_slots, SPARSE_MLA_INDEX_POOL_TOKENS, 128)
    _require_sparse_mla_tensors(
        device,
        (projection.main_q, torch.bfloat16, projection_shapes.main_q),
        (projection.index_q, torch.bfloat16, projection_shapes.index_q),
        (projection.score_weights, torch.float32, projection_shapes.score_weights),
        (weights.w_kc, torch.bfloat16, (local_heads, 256, 512)),
        (staged_query.active, torch.uint8, (batch_size,)),
        (staged_query.raw_lengths, torch.int32, (batch_size,)),
        (
            staged_query.block_table,
            torch.int32,
            (batch_size, SPARSE_MLA_COMPOUND_BLOCKS),
        ),
        (staged_query.null_token, torch.int32, (1,)),
        (
            history.latent,
            torch.bfloat16,
            (
                2 * history_blocks,
                1,
                SPARSE_MLA_LATENT_PAGE_TOKENS,
                SPARSE_MLA_KV_LORA_RANK,
            ),
        ),
        (
            history.index_cache,
            torch.uint8,
            (history_blocks, SPARSE_MLA_INDEX_PAGE_BYTES),
        ),
        (history.tail_key, torch.bfloat16, tail_shape),
        (history.tail_gate, torch.bfloat16, tail_shape),
    )
    _validate_sparse_mla_query_workspace(
        device, batch_size, workspace, attention_tp_size
    )
    _validate_sparse_mla_query_aliases(
        projection, weights, staged_query, history, workspace
    )
    return batch_size


def _validate_sparse_mla_decode(projection, weights, batch, history, workspace) -> int:
    batch_size = projection.latent.shape[0]
    if not 1 <= batch_size <= KDA_CHUNK_SIZE:
        raise ValueError("unsupported sparse MLA decode batch")
    device = projection.latent.device
    local_heads = weights.w_kc.shape[0]
    attention_tp_size = _attention_tp_size_from_heads(local_heads)
    projection_shapes = sparse_mla_projection_workspace_shapes(
        batch_size, attention_tp_size
    )
    _require_sparse_mla_tensors(
        device,
        (projection.latent, torch.bfloat16, projection_shapes.latent),
        (projection.main_q, torch.bfloat16, projection_shapes.main_q),
        (projection.index_q, torch.bfloat16, projection_shapes.index_q),
        (projection.key, torch.bfloat16, projection_shapes.key),
        (projection.pool_gate, torch.bfloat16, projection_shapes.pool_gate),
        (projection.score_weights, torch.float32, projection_shapes.score_weights),
        (weights.w_kc, torch.bfloat16, (local_heads, 256, 512)),
        (weights.pool_ape, torch.bfloat16, (4, 128)),
        (batch.active, torch.uint8, (batch_size,)),
        (batch.raw_lengths, torch.int32, (batch_size,)),
        (batch.state_slots, torch.int32, (batch_size,)),
        (
            batch.block_table,
            torch.int32,
            (batch_size, SPARSE_MLA_COMPOUND_BLOCKS),
        ),
        (batch.null_token, torch.int32, (1,)),
    )

    block_count = history.index_cache.shape[0]
    live_slots = history.tail_key.shape[1]
    if block_count < 1 or live_slots < 1:
        raise ValueError("sparse MLA history must not be empty")
    tail_shape = (2, live_slots, SPARSE_MLA_INDEX_POOL_TOKENS, 128)
    _require_sparse_mla_tensors(
        device,
        (
            history.latent,
            torch.bfloat16,
            (2 * block_count, 1, SPARSE_MLA_LATENT_PAGE_TOKENS, 512),
        ),
        (history.index_cache, torch.uint8, (block_count, 4224)),
        (history.tail_key, torch.bfloat16, tail_shape),
        (history.tail_gate, torch.bfloat16, tail_shape),
    )
    _validate_sparse_mla_query_workspace(
        device, batch_size, workspace, attention_tp_size
    )
    return batch_size


def _validate_sparse_mla_query_workspace(
    device, batch_size, workspace, attention_tp_size
) -> None:
    expected = sparse_mla_decode_workspace_shapes(batch_size, attention_tp_size)
    for name in workspace.__dataclass_fields__:
        if name == "index_q":
            dtype = torch.float8_e4m3fn
        elif name in ("score_weights", "logits"):
            dtype = torch.float32
        elif name in ("main_q_hbd", "absorbed_hbd", "attention_q", "output"):
            dtype = torch.bfloat16
        elif name in ("topk_rows", "flashinfer_workspace", "counter"):
            dtype = torch.uint8
        else:
            dtype = torch.int32
        _require_sparse_mla_tensors(
            device, (getattr(workspace, name), dtype, getattr(expected, name))
        )
    if workspace.schedule.storage_offset() or workspace.schedule.data_ptr() % 8:
        raise ValueError("DeepGEMM schedule must be an aligned base allocation")


def _validate_sparse_mla_query_aliases(
    projection, weights, staged_query, history, workspace
) -> None:
    writable = _workspace_tensors("workspace", workspace)
    for index, (left_name, left) in enumerate(writable):
        for right_name, right in writable[index + 1 :]:
            if _contiguous_tensors_overlap(left, right):
                raise ValueError(
                    f"sparse MLA query buffers {left_name} and {right_name} overlap"
                )

    read_only = (
        ("projection.main_q", projection.main_q),
        ("projection.index_q", projection.index_q),
        ("projection.score_weights", projection.score_weights),
        ("weights.w_kc", weights.w_kc),
        ("staged_query.active", staged_query.active),
        ("staged_query.raw_lengths", staged_query.raw_lengths),
        ("staged_query.block_table", staged_query.block_table),
        ("staged_query.null_token", staged_query.null_token),
        *_workspace_tensors("history", history),
    )
    for writable_name, writable_tensor in writable:
        for read_only_name, read_only_tensor in read_only:
            if _contiguous_tensors_overlap(writable_tensor, read_only_tensor):
                raise ValueError(
                    "sparse MLA query buffers "
                    f"{writable_name} and {read_only_name} overlap"
                )


def _validate_sparse_mla_output(latent, weights, workspace, group) -> int:
    batch_size = latent.shape[0]
    if not 1 <= batch_size <= KDA_CHUNK_SIZE:
        raise ValueError("unsupported sparse MLA output batch")
    if dist.get_world_size(group) != TP_SIZE:
        raise ValueError("GLM sparse MLA output requires a four-rank process group")

    device = latent.device
    local_heads = weights.w_vc.shape[0]
    attention_tp_size = _attention_tp_size_from_heads(local_heads)
    expected = sparse_mla_output_workspace_shapes(batch_size, attention_tp_size)
    _require_sparse_mla_tensors(
        device,
        (
            latent,
            torch.bfloat16,
            (batch_size, local_heads, SPARSE_MLA_KV_LORA_RANK),
        ),
        (
            weights.w_vc,
            torch.bfloat16,
            (local_heads, SPARSE_MLA_KV_LORA_RANK, SPARSE_MLA_VALUE_HEAD_DIM),
        ),
        (
            weights.o_proj,
            torch.float8_e4m3fn,
            (HIDDEN_SIZE, local_heads * SPARSE_MLA_VALUE_HEAD_DIM),
        ),
        (
            weights.o_proj_scale_inv,
            torch.float32,
            (
                HIDDEN_SIZE // FP8_BLOCK_SIZE,
                local_heads * SPARSE_MLA_VALUE_HEAD_DIM // FP8_BLOCK_SIZE,
            ),
        ),
        (workspace.value_hbd, torch.bfloat16, expected.value_hbd),
        (workspace.projected, torch.bfloat16, expected.projected),
        (workspace.projected_fp8, torch.float8_e4m3fn, expected.projected_fp8),
        (workspace.projected_scale, torch.float32, expected.projected_scale),
        (workspace.output, torch.bfloat16, expected.output),
    )
    return batch_size


def _require_sparse_mla_tensors(device, *specifications) -> None:
    for tensor, dtype, shape in specifications:
        if tensor.shape != shape or tensor.dtype != dtype:
            raise ValueError("sparse MLA tensor has the wrong shape or dtype")
        if tensor.device != device or not tensor.is_cuda or not tensor.is_contiguous():
            raise ValueError("sparse MLA tensors must be contiguous on one CUDA device")


def _validate_sparse_layer_buffers(streams, layer_workspace, fields=None) -> slice:
    streams_shape = tuple(streams.shape)
    if len(streams_shape) != 3 or not 1 <= streams_shape[0] <= KDA_CHUNK_SIZE:
        raise ValueError("streams must match a supported sparse layer batch")
    batch_size = streams_shape[0]
    expected_streams = (batch_size, MHC_STREAMS, HIDDEN_SIZE)
    if streams_shape != expected_streams:
        raise ValueError("streams must match a supported sparse layer batch")
    rows = slice(batch_size)
    expected = dense_workspace_shapes(batch_size)
    require_cuda = fields is None
    if fields is None:
        fields = (
            "mhc_sqrsum",
            "mhc_dot",
            "post",
            "comb",
            "collapsed",
            "normalized",
            "streams_mid",
            "streams_out",
        )
    float32 = {"mhc_sqrsum", "mhc_dot", "post", "comb"}
    _require_kda_decode_tensors(
        streams.device,
        ("streams", streams, torch.bfloat16, expected_streams),
        *(
            (
                f"layer.{name}",
                getattr(layer_workspace, name)[rows],
                torch.float32 if name in float32 else torch.bfloat16,
                getattr(expected, name),
            )
            for name in fields
        ),
        require_cuda=require_cuda,
    )
    return rows


def _sparse_tep_rank(process_group: object) -> int:
    world = process_group is dist.group.WORLD
    if not world or dist.get_world_size(process_group) != TP_SIZE:
        raise ValueError("GLM sparse layers require a WORLD TEP4 process group")
    return dist.get_rank(process_group)


def _validate_sparse_kda_prefill_layer(
    streams,
    weights,
    state,
    batch,
    kda_workspace,
    ffn_workspace,
    layer_workspace,
    tep_rank,
) -> int:
    rows = _validate_sparse_layer_buffers(streams, layer_workspace)
    _validate_sparse_layer_shell(weights, streams.device)
    total_tokens = _validate_kda_prefill(
        layer_workspace.normalized[: batch.metadata.total_tokens],
        weights.attention,
        state,
        batch,
        kda_workspace,
    )
    if total_tokens != rows.stop or (
        batch.metadata.batch_size > kda_workspace.initial_state.shape[0]
        or kda_workspace.projection.shape[0] != rows.stop
    ):
        raise ValueError("sparse KDA prefill workspace must match its token count")
    _validate_sparse_ffn(
        layer_workspace.normalized[rows], weights.ffn, ffn_workspace, tep_rank
    )
    _validate_sparse_kda_layer_aliases(
        streams,
        weights,
        state,
        (
            ("cu_seqlens", batch.metadata.cu_seqlens),
            ("cu_seqlens_int64", batch.metadata.cu_seqlens_int64),
            *_workspace_tensors("batch", batch),
        ),
        kda_workspace,
        ffn_workspace,
        layer_workspace,
        rows,
    )
    return total_tokens


def _validate_sparse_mla_prefill_layer(
    streams,
    weights,
    staged_query,
    history,
    projection_workspace,
    decode_workspace,
    output_workspace,
    ffn_workspace,
    layer_workspace,
    process_group,
    tep_rank,
) -> int:
    rows = _validate_sparse_layer_buffers(streams, layer_workspace)
    _validate_sparse_layer_shell(weights, streams.device)
    query_tokens = _validate_sparse_mla_prefill_query(
        projection_workspace,
        weights.mla_decode,
        staged_query,
        history,
        decode_workspace,
    )
    if query_tokens != rows.stop or staged_query.total_tokens != rows.stop:
        raise ValueError("sparse MLA descriptors must match their token count")
    total_tokens = staged_query.total_tokens
    active_rows = slice(total_tokens)
    _require_sparse_mla_tensors(
        streams.device,
        (
            projection_workspace.latent[active_rows],
            torch.bfloat16,
            (total_tokens, SPARSE_MLA_KV_LORA_RANK),
        ),
        (
            projection_workspace.key[active_rows],
            torch.bfloat16,
            (total_tokens, SPARSE_MLA_INDEXER_HEAD_DIM),
        ),
        (
            projection_workspace.pool_gate[active_rows],
            torch.bfloat16,
            (total_tokens, SPARSE_MLA_INDEXER_HEAD_DIM),
        ),
        (
            weights.mla_decode.pool_ape,
            torch.bfloat16,
            (SPARSE_MLA_INDEX_POOL_TOKENS, SPARSE_MLA_INDEXER_HEAD_DIM),
        ),
    )
    _validate_sparse_mla_output(
        decode_workspace.output[:, 0],
        weights.mla_output,
        output_workspace,
        process_group,
    )
    _validate_sparse_ffn(
        layer_workspace.normalized[rows], weights.ffn, ffn_workspace, tep_rank
    )
    mutable = (
        ("streams", streams),
        *_workspace_tensors("layer", layer_workspace),
        *_workspace_tensors("projection", projection_workspace),
        *_workspace_tensors("decode", decode_workspace),
        *_workspace_tensors("output", output_workspace),
        *_workspace_tensors("ffn", ffn_workspace),
        *_workspace_tensors("history", history),
    )
    read_only = (
        ("query.active", staged_query.active),
        ("query.raw_lengths", staged_query.raw_lengths),
        ("query.block_table", staged_query.block_table),
        ("query.null_token", staged_query.null_token),
        *_workspace_tensors("weights", weights),
    )
    _reject_sparse_layer_aliases(mutable, read_only)
    _sparse_mla_runtime()
    return total_tokens


def _validate_sparse_kda_layer(
    streams: torch.Tensor,
    weights: GLM53SparseKDALayerWeights[torch.Tensor],
    state: KDAState[torch.Tensor],
    state_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    kda_workspace: KDADecodeWorkspace[torch.Tensor],
    ffn_workspace: SparseFFNDecodeWorkspace[torch.Tensor],
    layer_workspace: GLM53DenseWorkspace[torch.Tensor],
    tep_rank: int,
) -> slice:
    rows = _validate_sparse_layer_buffers(streams, layer_workspace)
    _validate_sparse_layer_shell(weights, streams.device)
    _validate_kda_decode(
        layer_workspace.normalized[rows],
        weights.attention,
        state,
        state_indices,
        cu_seqlens,
        kda_workspace,
    )
    _validate_sparse_ffn(
        layer_workspace.normalized[rows], weights.ffn, ffn_workspace, tep_rank
    )
    _validate_sparse_kda_layer_aliases(
        streams,
        weights,
        state,
        (("state_indices", state_indices), ("cu_seqlens", cu_seqlens)),
        kda_workspace,
        ffn_workspace,
        layer_workspace,
        rows,
    )
    return rows


def _validate_kda_decode(
    hidden_states: torch.Tensor,
    weights: KDAWeights[torch.Tensor],
    state: KDAState[torch.Tensor],
    state_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    workspace: KDADecodeWorkspace[torch.Tensor],
) -> int:
    batch_size = _validate_kda_inputs(hidden_states, weights, state)
    local_heads = weights.a_log.shape[0]
    attention_tp_size = _attention_tp_size_from_heads(local_heads)
    expected = kda_decode_workspace_shapes(batch_size, attention_tp_size)
    workspace_specs = (
        ("projection", expected.projection, torch.bfloat16),
        ("output_gate", expected.output_gate, torch.bfloat16),
        ("gate_raw", expected.gate_raw, torch.float32),
        ("local_output", expected.local_output, torch.bfloat16),
        ("output", expected.output, torch.bfloat16),
    )
    _require_kda_decode_tensors(
        hidden_states.device,
        ("state_indices", state_indices, torch.int32, (batch_size,)),
        ("cu_seqlens", cu_seqlens, torch.int32, (batch_size + 1,)),
        *(
            (
                f"workspace.{name}",
                getattr(workspace, name),
                dtype,
                shape,
            )
            for name, shape, dtype in workspace_specs
        ),
    )
    return batch_size


def _require_kda_decode_tensors(device, *specifications, require_cuda=True) -> None:
    for name, tensor, dtype, shape in specifications:
        if tuple(tensor.shape) != shape or tensor.dtype != dtype:
            raise ValueError(f"invalid KDA {name} shape or dtype")
        valid_device = tensor.device == device and (tensor.is_cuda or not require_cuda)
        if not valid_device or not tensor.is_contiguous():
            raise ValueError(f"invalid KDA {name} device or layout")


def _validate_sparse_kda_layer_aliases(
    streams: torch.Tensor,
    weights: GLM53SparseKDALayerWeights[torch.Tensor],
    state: KDAState[torch.Tensor],
    descriptors,
    kda_workspace: KDADecodeWorkspace[torch.Tensor],
    ffn_workspace: SparseFFNDecodeWorkspace[torch.Tensor],
    layer_workspace: GLM53DenseWorkspace[torch.Tensor],
    rows: slice,
) -> None:
    layer_fields = (
        "mhc_sqrsum",
        "mhc_dot",
        "post",
        "comb",
        "collapsed",
        "normalized",
        "streams_mid",
        "streams_out",
    )
    mutable_buffers = (
        ("streams", streams),
        ("state.recurrent", state.recurrent),
        ("state.conv", state.conv),
        *tuple(
            (f"layer.{name}", getattr(layer_workspace, name)[rows])
            for name in layer_fields
        ),
        *_workspace_tensors("kda", kda_workspace),
        *_workspace_tensors("ffn", ffn_workspace),
    )
    read_only_buffers = (*descriptors, *_workspace_tensors("weights", weights))
    _reject_sparse_layer_aliases(mutable_buffers, read_only_buffers)


def _validate_sparse_mla_layer(
    streams: torch.Tensor,
    weights: GLM53SparseMLALayerWeights[torch.Tensor],
    batch: SparseMLADecodeBatch[torch.Tensor],
    history: SparseMLAHistory[torch.Tensor],
    projection_workspace: SparseMLAProjectionWorkspace[torch.Tensor],
    decode_workspace: SparseMLADecodeWorkspace[torch.Tensor],
    output_workspace: SparseMLAOutputWorkspace[torch.Tensor],
    ffn_workspace: SparseFFNDecodeWorkspace[torch.Tensor],
    layer_workspace: GLM53DenseWorkspace[torch.Tensor],
    process_group: object,
    tep_rank: int,
) -> slice:
    rows = _validate_sparse_layer_buffers(
        streams,
        layer_workspace,
        ("streams_mid", "streams_out"),
    )
    batch_size = rows.stop
    streams_mid = layer_workspace.streams_mid[rows]
    streams_out = layer_workspace.streams_out[rows]
    ffn_output = _sparse_ffn_local_output(ffn_workspace, batch_size)
    for name, tensor in (
        ("attention_output", output_workspace.output),
        ("ffn_output", ffn_output),
    ):
        if (
            tensor.shape != (batch_size, HIDDEN_SIZE)
            or tensor.dtype != torch.bfloat16
            or tensor.device != streams.device
            or not tensor.is_contiguous()
        ):
            raise ValueError(f"invalid sparse MLA layer {name} buffer")

    buffers = (
        ("streams", streams),
        ("streams_mid", streams_mid),
        ("streams_out", streams_out),
        ("attention_output", output_workspace.output),
        ("ffn_output", ffn_output),
    )
    _reject_sparse_layer_aliases(buffers, ())

    for workspace_name, workspace in (
        ("projection", projection_workspace),
        ("decode", decode_workspace),
        ("output", output_workspace),
    ):
        for name, tensor in _workspace_tensors(workspace_name, workspace):
            if _contiguous_tensors_overlap(streams, tensor):
                raise ValueError(f"sparse MLA layer streams and {name} scratch overlap")
    for name, tensor in _workspace_tensors("ffn", ffn_workspace):
        if _contiguous_tensors_overlap(streams_mid, tensor):
            raise ValueError(f"sparse MLA layer streams_mid and {name} scratch overlap")

    _validate_sparse_layer_shell(weights, streams.device)
    _validate_sparse_mla_decode(
        projection_workspace,
        weights.mla_decode,
        batch,
        history,
        decode_workspace,
    )
    _validate_sparse_mla_output(
        decode_workspace.output[:, 0],
        weights.mla_output,
        output_workspace,
        process_group,
    )
    _validate_sparse_ffn(
        layer_workspace.normalized[rows], weights.ffn, ffn_workspace, tep_rank
    )
    return rows


def _contiguous_tensors_overlap(left: torch.Tensor, right: torch.Tensor) -> bool:
    if left.device != right.device:
        return False
    left_start = left.data_ptr()
    right_start = right.data_ptr()
    left_end = left_start + left.numel() * left.element_size()
    right_end = right_start + right.numel() * right.element_size()
    return left_start < right_end and right_start < left_end


def _workspace_tensors(
    label: str, workspace: object
) -> tuple[tuple[str, torch.Tensor], ...]:
    names = getattr(type(workspace), "__dataclass_fields__", None) or vars(workspace)
    tensors = []
    for name in names:
        value = getattr(workspace, name)
        nested_label = f"{label}.{name}"
        if isinstance(value, torch.Tensor):
            tensors.append((nested_label, value))
        elif getattr(type(value), "__dataclass_fields__", None) is not None:
            tensors.extend(_workspace_tensors(nested_label, value))
    return tuple(tensors)


def _reject_sparse_layer_aliases(mutable, read_only) -> None:
    for index, (left_name, left) in enumerate(mutable):
        for right_name, right in mutable[index + 1 :]:
            names = {left_name, right_name}
            exact_stream_output = names in (
                {"streams", "streams_out"},
                {"streams", "layer.streams_out"},
            ) and (
                left.data_ptr() == right.data_ptr()
                and left.numel() * left.element_size()
                == right.numel() * right.element_size()
            )
            kda_arena = names == {
                "kda.kda_workspace_storage",
                "kda.kda_workspace",
            }
            if (
                not exact_stream_output
                and not kda_arena
                and _contiguous_tensors_overlap(left, right)
            ):
                raise ValueError(
                    f"sparse layer buffers {left_name} and {right_name} overlap"
                )
        for right_name, right in read_only:
            if _contiguous_tensors_overlap(left, right):
                raise ValueError(
                    f"sparse layer buffers {left_name} and {right_name} overlap"
                )


def _validate_sparse_layer_shell(
    weights: GLM53SparseKDALayerWeights[torch.Tensor]
    | GLM53SparseMLALayerWeights[torch.Tensor],
    device: torch.device,
) -> None:
    mhc_width = MHC_STREAMS * (MHC_STREAMS + 2)
    _require_sparse_mla_tensors(
        device,
        (weights.attention_mhc.base, torch.float32, (mhc_width,)),
        (
            weights.attention_mhc.fn,
            torch.float32,
            (mhc_width, MHC_STREAMS * HIDDEN_SIZE),
        ),
        (weights.attention_mhc.scale, torch.float32, (MHC_STREAMS - 1,)),
        (weights.input_norm, torch.bfloat16, (HIDDEN_SIZE,)),
        (weights.ffn_mhc.base, torch.float32, (mhc_width,)),
        (
            weights.ffn_mhc.fn,
            torch.float32,
            (mhc_width, MHC_STREAMS * HIDDEN_SIZE),
        ),
        (weights.ffn_mhc.scale, torch.float32, (MHC_STREAMS - 1,)),
        (weights.post_attention_norm, torch.bfloat16, (HIDDEN_SIZE,)),
    )


def _validate_sparse_ffn(
    hidden_states: torch.Tensor,
    weights: SparseFFNWeights[torch.Tensor],
    workspace: SparseFFNDecodeWorkspace[torch.Tensor],
    tep_rank: int,
    moe_world_size: int | None = None,
    local_batch_size: int | None = None,
) -> int:
    batch_size = hidden_states.shape[0]
    inferred_world_size = _sparse_ffn_world_size(workspace)
    validating_local_input = moe_world_size is None
    if moe_world_size is None:
        moe_world_size = inferred_world_size
    if local_batch_size is None:
        local_batch_size = (
            workspace.scattered.shape[0] if inferred_world_size > 1 else batch_size
        )
    compute_batch_size = local_batch_size * moe_world_size
    max_input_tokens = (
        KDA_CHUNK_SIZE if validating_local_input else KDA_CHUNK_SIZE * moe_world_size
    )
    if not 1 <= batch_size <= max_input_tokens or hidden_states.shape != (
        batch_size,
        HIDDEN_SIZE,
    ):
        raise ValueError(
            f"hidden_states must have shape [N, {HIDDEN_SIZE}] for "
            f"1 <= N <= {max_input_tokens}, "
            f"got {tuple(hidden_states.shape)}"
        )
    if hidden_states.dtype != torch.bfloat16:
        raise TypeError(f"hidden_states must be bfloat16, got {hidden_states.dtype}")
    if not 0 <= tep_rank < TP_SIZE:
        raise ValueError(f"tep_rank must be in [0, {TP_SIZE}), got {tep_rank}")
    if validating_local_input and batch_size > local_batch_size:
        raise ValueError("sparse FFN local batch exceeds its workspace capacity")
    if not validating_local_input and batch_size != compute_batch_size:
        raise ValueError("sparse FFN local and global batch sizes disagree")
    if weights.router.dtype != torch.bfloat16:
        raise TypeError("sparse FFN router must be bfloat16")

    device = hidden_states.device
    if not hidden_states.is_cuda or not hidden_states.is_contiguous():
        raise ValueError("hidden_states must be contiguous on a CUDA device")

    weight_fp8 = {
        "routed_up_gate",
        "routed_down",
        "shared_gate_up",
        "shared_down",
    }
    for name in weights.__dataclass_fields__:
        tensor = getattr(weights, name)
        dtype = (
            torch.float8_e4m3fn
            if name in weight_fp8
            else torch.bfloat16
            if name in {"router", "router_t"}
            else torch.float32
        )
        if (
            tensor.shape != getattr(SPARSE_FFN_TEP4_SHAPES, name)
            or tensor.dtype != dtype
        ):
            raise ValueError(f"invalid sparse FFN {name} weight shape or dtype")
        if tensor.device != device or not tensor.is_cuda:
            raise ValueError(f"invalid sparse FFN {name} weight device")
        if name != "router_t" and not tensor.is_contiguous():
            raise ValueError(f"invalid sparse FFN {name} weight layout")

    expected = sparse_ffn_decode_workspace_shapes(local_batch_size, moe_world_size)
    workspace_fp8 = {"hidden_fp8", "shared_fp8"}
    workspace_bf16 = {
        "gathered",
        "scattered",
        "routed",
        "shared_gate_up",
        "output",
    }
    workspace_int64 = {"topk_ids64"}
    workspace_int32 = {"topk_ids"}
    for name in workspace.__dataclass_fields__:
        tensor = getattr(workspace, name)
        dtype = (
            torch.float8_e4m3fn
            if name in workspace_fp8
            else torch.bfloat16
            if name in workspace_bf16
            else torch.int64
            if name in workspace_int64
            else torch.int32
            if name in workspace_int32
            else torch.float32
        )
        if tensor.shape != getattr(expected, name) or tensor.dtype != dtype:
            raise ValueError(f"invalid sparse FFN {name} workspace shape or dtype")
        if tensor.device != device or not tensor.is_cuda or not tensor.is_contiguous():
            raise ValueError(f"invalid sparse FFN {name} workspace layout")
    return batch_size


def _sparse_ffn_world_size(
    workspace: SparseFFNDecodeWorkspace[torch.Tensor],
) -> int:
    return TP_SIZE if workspace.gathered.shape[0] else 1


def _sparse_ffn_local_output(
    workspace: SparseFFNDecodeWorkspace[torch.Tensor], rows: int
) -> torch.Tensor:
    output = (
        workspace.scattered
        if _sparse_ffn_world_size(workspace) > 1
        else workspace.output
    )
    return output[:rows]


def _prepare_dense_layer(
    streams: torch.Tensor,
    weights: GLM53LayerWeights[torch.Tensor],
    workspace: GLM53DenseWorkspace[torch.Tensor],
) -> slice:
    rows = slice(streams.shape[0])
    w = workspace
    _mhc_pre(streams, weights.attention_mhc, w, rows)
    _rmsnorm(w.collapsed[rows], weights.input_norm, w.normalized[rows])
    return rows


def _finish_dense_layer(
    attention_output: torch.Tensor,
    streams: torch.Tensor,
    weights: GLM53LayerWeights[torch.Tensor],
    workspace: GLM53DenseWorkspace[torch.Tensor],
    rows: slice,
    all_reduce: _AllReduce,
) -> torch.Tensor:
    w = workspace
    _mhc_post(attention_output, streams, w, w.streams_mid[rows], rows)
    _mhc_pre(w.streams_mid[rows], weights.ffn_mhc, w, rows)
    _rmsnorm(w.collapsed[rows], weights.post_attention_norm, w.normalized[rows])
    hidden_scale = _mn_scale_view(w.hidden_scale[rows])
    _quantize(w.normalized[rows], w.hidden_fp8[rows], hidden_scale)
    _fp8_gemm(
        w.hidden_fp8[rows],
        weights.ffn.gate_up,
        hidden_scale,
        weights.ffn.gate_up_scale_inv,
        w.gate_up[rows],
    )
    _swiglu_kernel[(w.gate_up[rows].shape[0], triton.cdiv(w.activated.shape[1], 256))](
        w.gate_up[rows],
        w.activated[rows],
        w.activated_scale[rows],
        WIDTH=w.activated.shape[1],
        BLOCK_SIZE=256,
        FP8_LIMIT=448.0,
        LIMIT=10.0,
        QUANTIZE=False,
    )
    activated_scale = _mn_scale_view(w.activated_scale[rows])
    _quantize(
        w.activated[rows],
        w.activated_fp8[rows],
        activated_scale,
    )
    _fp8_gemm(
        w.activated_fp8[rows],
        weights.ffn.down,
        activated_scale,
        weights.ffn.down_scale_inv,
        w.ffn_output[rows],
    )
    ffn_output = w.ffn_output[rows]
    if weights.ffn.gate_up.shape[0] != 2 * 12_288:
        ffn_output = all_reduce(ffn_output)
    _mhc_post(ffn_output, w.streams_mid[rows], w, w.streams_out[rows], rows)
    return w.streams_out[rows]


def _mn_scale_view(scale: torch.Tensor) -> torch.Tensor:
    return scale.view(scale.shape[1], scale.shape[0]).T


def _mhc_pre(
    streams: torch.Tensor,
    weights: MHCWeights[torch.Tensor],
    workspace: GLM53DenseWorkspace[torch.Tensor],
    rows: slice,
) -> None:
    from flashinfer.mhc import get_mhc_module

    row_count = streams.shape[0]
    if row_count <= 1024:
        num_splits = MHC_PRE_MAX_SPLITS
    else:
        num_splits = 4
    split_rows = num_splits * row_count
    mix_count = MHC_STREAMS * (MHC_STREAMS + 2)
    dot = workspace.mhc_dot.view(-1, mix_count)[:split_rows].view(
        num_splits, row_count, mix_count
    )
    sqrsum = workspace.mhc_sqrsum.view(-1)[:split_rows].view(num_splits, row_count)
    deep_gemm = _sparse_mla_runtime()[0]
    deep_gemm.tf32_hc_prenorm_gemm(
        streams.view(row_count, MHC_STREAMS * HIDDEN_SIZE),
        weights.fn,
        dot,
        sqrsum,
        num_splits=num_splits,
    )
    get_mhc_module().mhc_pre_big_fuse(
        workspace.post[rows],
        workspace.comb[rows],
        workspace.collapsed[rows],
        dot,
        sqrsum,
        streams,
        weights.scale,
        weights.base,
        MHC_STREAMS * HIDDEN_SIZE,
        RMS_NORM_EPS,
        MHC_EPS,
        MHC_EPS,
        2.0,
        MHC_NORMALIZATION_ITERS,
        num_splits,
        0,
    )


def _mhc_post(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    workspace: GLM53DenseWorkspace[torch.Tensor],
    out: torch.Tensor,
    rows: slice,
) -> None:
    from flashinfer.mhc import get_mhc_module

    get_mhc_module().mhc_post(
        out, hidden_states, residual, workspace.post[rows], workspace.comb[rows]
    )


def _rmsnorm(input_: torch.Tensor, weight: torch.Tensor, out: torch.Tensor) -> None:
    from flashinfer.norm import rmsnorm

    rmsnorm(input_, weight, eps=RMS_NORM_EPS, out=out, enable_pdl=True)


def _quantize(
    input_: torch.Tensor,
    output: torch.Tensor,
    scale: torch.Tensor,
    zero_scale: float = 1.0,
) -> None:
    blocks = input_.shape[1] // FP8_BLOCK_SIZE
    _quantize_fp8[(input_.shape[0] * blocks,)](
        input_,
        output,
        scale,
        ROWS=input_.shape[0],
        BLOCKS=blocks,
        BLOCK_SIZE=FP8_BLOCK_SIZE,
        FP8_LIMIT=448.0,
        ZERO_SCALE=zero_scale,
    )


@triton.jit
def _route_sparse_ffn_pre_topk(
    scores_ptr,
    correction_bias_ptr,
    selection_ptr,
    NUM_EXPERTS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < NUM_EXPERTS
    scores = tl.sigmoid(
        tl.load(scores_ptr + row * NUM_EXPERTS + offsets, mask=mask).to(tl.float32)
    )
    selection = scores + tl.load(correction_bias_ptr + offsets, mask=mask)
    tl.store(scores_ptr + row * NUM_EXPERTS + offsets, scores, mask=mask)
    tl.store(selection_ptr + row * NUM_EXPERTS + offsets, selection, mask=mask)


@triton.jit
def _route_sparse_ffn_post_topk(
    scores_ptr,
    ids64_ptr,
    weights_ptr,
    ids32_ptr,
    NUM_EXPERTS: tl.constexpr,
    TOPK: tl.constexpr,
    ROUTE_SCALE: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, TOPK)
    ids = tl.load(ids64_ptr + row * TOPK + offsets)
    weights = tl.load(scores_ptr + row * NUM_EXPERTS + ids).to(tl.float32)
    denominator = tl.sum(weights, axis=0) + 1e-20
    tl.store(weights_ptr + row * TOPK + offsets, weights / denominator * ROUTE_SCALE)
    tl.store(ids32_ptr + row * TOPK + offsets, ids)


def _fp8_gemm(
    input_: torch.Tensor,
    weight: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    out: torch.Tensor,
) -> None:
    from flashinfer.gemm import gemm_fp8_nt_groupwise

    gemm_fp8_nt_groupwise(
        input_,
        weight,
        input_scale,
        weight_scale.T,
        scale_major_mode=None,
        mma_sm=2 if input_.shape[0] >= 256 else 1,
        scale_granularity_mnk=(1, FP8_BLOCK_SIZE, FP8_BLOCK_SIZE),
        out=out,
        backend="trtllm",
    )


@triton.jit
def _layernorm_bf16(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    ROW_WIDTH: tl.constexpr,
    WIDTH: tl.constexpr,
    EPS: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, WIDTH)
    values = tl.load(input_ptr + row * ROW_WIDTH + offsets).to(tl.float32)
    mean = tl.sum(values, axis=0) / WIDTH
    centered = values - mean
    variance = tl.sum(centered * centered, axis=0) / WIDTH
    weight = tl.load(weight_ptr + offsets).to(tl.float32)
    bias = tl.load(bias_ptr + offsets).to(tl.float32)
    output = centered * tl.rsqrt(variance + EPS) * weight + bias
    tl.store(output_ptr + row * WIDTH + offsets, output)


@triton.jit
def _prepare_sparse_mla_query_lengths(
    active_ptr,
    length_ptr,
    context_ptr,
    sequence_ptr,
):
    row = tl.program_id(0)
    active = tl.load(active_ptr + row) != 0
    raw_length = tl.load(length_ptr + row)
    tl.store(context_ptr + row, tl.where(active, raw_length // 4, 0))
    tl.store(sequence_ptr + row, tl.where(active, raw_length, 1))


@triton.jit
def _write_sparse_mla_history(
    active_ptr,
    length_ptr,
    slot_ptr,
    table_ptr,
    latent_ptr,
    key_ptr,
    gate_ptr,
    ape_ptr,
    latent_cache_ptr,
    index_cache_ptr,
    tail_key_ptr,
    tail_gate_ptr,
    context_ptr,
    sequence_ptr,
    transaction_key_ptr,
    transaction_gate_ptr,
    LIVE_SLOTS: tl.constexpr,
    ROW_START: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
    WRITE_TRANSACTION: tl.constexpr,
):
    row = ROW_START + tl.program_id(0) * ROW_STRIDE
    active = tl.load(active_ptr + row) != 0
    raw_length = tl.load(length_ptr + row)
    position = tl.maximum(raw_length - 1, 0)
    table_offset = row * 8192 + position // 128
    block = tl.load(table_ptr + table_offset, mask=active, other=0)
    within = position % 128
    latent_dim = tl.arange(0, 512)
    latent_page = 2 * block + within // 64
    latent_out = latent_cache_ptr + (latent_page * 64 + within % 64) * 512
    latent = tl.load(latent_ptr + row * 512 + latent_dim)
    tl.store(latent_out + latent_dim, latent, mask=active)

    dim = tl.arange(0, 128)
    slot = tl.load(slot_ptr + row)
    tail_row = position % 4
    key = tl.load(key_ptr + row * 128 + dim)
    gate = tl.load(gate_ptr + row * 128 + dim)
    for parity in tl.static_range(2):
        tail = (parity * LIVE_SLOTS + slot) * 512 + tail_row * 128
        tl.store(tail_key_ptr + tail + dim, key, mask=active)
        tl.store(tail_gate_ptr + tail + dim, gate, mask=active)

    if WRITE_TRANSACTION:
        transaction_offsets = tl.arange(0, 512)
        transaction_lane = transaction_offsets // 128
        transaction_dim = transaction_offsets % 128
        current_key = tl.load(key_ptr + row * 128 + transaction_dim)
        current_gate = tl.load(gate_ptr + row * 128 + transaction_dim)
        for parity in tl.static_range(2):
            tail = (parity * LIVE_SLOTS + slot) * 512
            prior_key = tl.load(tail_key_ptr + tail + transaction_offsets)
            prior_gate = tl.load(tail_gate_ptr + tail + transaction_offsets)
            key_snapshot = tl.where(
                transaction_lane == tail_row, current_key, prior_key
            )
            gate_snapshot = tl.where(
                transaction_lane == tail_row, current_gate, prior_gate
            )
            transaction = (row * 2 + parity) * 512 + transaction_offsets
            tl.store(transaction_key_ptr + transaction, key_snapshot, mask=active)
            tl.store(transaction_gate_ptr + transaction, gate_snapshot, mask=active)

    complete = active & (tail_row == 3)
    pooled_offsets = tl.arange(0, 512)
    lane = pooled_offsets // 128
    pooled_dim = pooled_offsets % 128
    tail = slot * 512 + lane * 128 + pooled_dim
    prior_key = tl.load(tail_key_ptr + tail, mask=complete & (lane < 3), other=0.0)
    prior_gate = tl.load(tail_gate_ptr + tail, mask=complete & (lane < 3), other=0.0)
    current_key = tl.load(key_ptr + row * 128 + pooled_dim)
    current_gate = tl.load(gate_ptr + row * 128 + pooled_dim)
    keys = tl.reshape(tl.where(lane == 3, current_key, prior_key), (4, 128))
    logits = tl.reshape(
        tl.where(lane == 3, current_gate, prior_gate).to(tl.float32)
        + tl.load(ape_ptr + pooled_offsets).to(tl.float32),
        (4, 128),
    )
    probability = tl.softmax(logits, dim=0).to(tl.bfloat16)
    product = (probability * keys).to(tl.bfloat16).to(tl.float32)
    product_02, product_13 = tl.split(tl.reshape(tl.trans(product), (128, 2, 2)))
    product_0, product_2 = tl.split(product_02)
    product_1, product_3 = tl.split(product_13)
    pooled = product_0 + product_1
    pooled = pooled + product_2
    pooled = pooled + product_3
    pooled = pooled.to(tl.bfloat16).to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(pooled), axis=0), 1.0e-4) / 448.0
    fp8 = tl.clamp(pooled / scale, -448.0, 448.0).to(tl.float8e4nv)
    index_row = (position // 4) % 32
    page = index_cache_ptr + block * 4224
    fp8_bytes = fp8.to(tl.uint8, bitcast=True)
    tl.store(page + index_row * 128 + dim, fp8_bytes, mask=complete)
    byte = tl.arange(0, 4)
    scale_bits = scale.to(tl.int32, bitcast=True)
    scale_bytes = ((scale_bits >> (8 * byte)) & 255).to(tl.uint8)
    tl.store(page + 4096 + index_row * 4 + byte, scale_bytes, mask=complete)
    tl.store(context_ptr + row, tl.where(active, raw_length // 4, 0))
    tl.store(sequence_ptr + row, tl.where(active, raw_length, 1))


@triton.jit
def _map_sparse_mla_ids(
    active_ptr,
    length_ptr,
    table_ptr,
    null_ptr,
    selected_ptr,
    output_ptr,
    sparse_length_ptr,
):
    row = tl.program_id(0)
    offsets = tl.program_id(1) * 256 + tl.arange(0, 256)
    active = tl.load(active_ptr + row) != 0
    raw_length = tl.load(length_ptr + row)
    pools = tl.minimum(raw_length // 4, 512)
    selected_tokens = 4 * pools
    tail = raw_length % 4
    active_length = selected_tokens + tail
    selected_offset = row * 512 + offsets // 4
    selected_mask = active & (offsets < selected_tokens)
    selected = tl.load(selected_ptr + selected_offset, mask=selected_mask, other=0)
    raw = tl.where(
        offsets < selected_tokens,
        4 * selected + offsets % 4,
        raw_length - tail + offsets - selected_tokens,
    )
    valid = active & (offsets < active_length)
    block = tl.load(table_ptr + row * 8192 + raw // 128, mask=valid, other=0)
    physical = block * 128 + raw % 128
    value = tl.where(valid, physical, -1)
    value = tl.where((~active) & (offsets == 0), tl.load(null_ptr), value)
    tl.store(output_ptr + row * 2052 + offsets, value, mask=offsets < 2052)
    sparse_length = tl.where(active, active_length, 1)
    tl.store(sparse_length_ptr + row, sparse_length, mask=tl.program_id(1) == 0)


@triton.jit
def _rmsnorm_sigmoid_gate(
    input_ptr,
    gate_ptr,
    weight_ptr,
    output_ptr,
    WIDTH: tl.constexpr,
    EPS: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, WIDTH)
    input_ = tl.load(input_ptr + row * WIDTH + offsets).to(tl.float32)
    gate = tl.load(gate_ptr + row * WIDTH + offsets).to(tl.float32)
    weight = tl.load(weight_ptr + offsets).to(tl.float32)
    inverse_rms = 1.0 / tl.sqrt(tl.sum(input_ * input_, axis=0) / WIDTH + EPS)
    output = input_ * inverse_rms * weight * tl.sigmoid(gate)
    tl.store(output_ptr + row * WIDTH + offsets, output)


@triton.jit(do_not_specialize=["ROWS"])
def _quantize_fp8(
    input_ptr,
    output_ptr,
    scale_ptr,
    ROWS,
    BLOCKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    FP8_LIMIT: tl.constexpr,
    ZERO_SCALE: tl.constexpr,
):
    program = tl.program_id(0)
    row = program // BLOCKS
    block = program - row * BLOCKS
    offsets = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    values = tl.load(input_ptr + row * BLOCKS * BLOCK_SIZE + offsets).to(tl.float32)
    maximum = tl.max(tl.abs(values), axis=0)
    scale = tl.where(maximum > 0.0, maximum / FP8_LIMIT, ZERO_SCALE)
    quantized = tl.maximum(tl.minimum(values / scale, FP8_LIMIT), -FP8_LIMIT)
    tl.store(output_ptr + row * BLOCKS * BLOCK_SIZE + offsets, quantized)
    tl.store(scale_ptr + block * ROWS + row, scale)


@triton.jit
def _swiglu_kernel(
    gate_up_ptr,
    output_ptr,
    scale_ptr,
    WIDTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    FP8_LIMIT: tl.constexpr,
    LIMIT: tl.constexpr,
    QUANTIZE: tl.constexpr,
):
    row = tl.program_id(0)
    block = tl.program_id(1)
    offsets = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < WIDTH
    gate = tl.load(gate_up_ptr + row * 2 * WIDTH + offsets, mask=mask)
    up = tl.load(gate_up_ptr + row * 2 * WIDTH + WIDTH + offsets, mask=mask)
    gate = tl.minimum(gate.to(tl.float32), LIMIT)
    up = tl.maximum(tl.minimum(up.to(tl.float32), LIMIT), -LIMIT)
    values = gate * tl.sigmoid(gate) * up
    if QUANTIZE:
        scale = tl.maximum(tl.max(tl.abs(values), axis=0) / FP8_LIMIT, 1e-12)
        values = (values / scale).to(tl.float8e4nv)
        tl.store(scale_ptr + block * tl.num_programs(0) + row, scale)
    tl.store(output_ptr + row * WIDTH + offsets, values, mask=mask)

import torch

from infer.models.deepseek_v4_flash import model as deepseek_v4_flash
from infer.models.deepseek_v4_flash.dspark import DeepSeekV4DSparkOutput
from infer.models.deepseek_v4_flash.model import DeepSeekV4RawAttention
from infer.models.deepseek_v4_flash.ops.attention import (
    DeepSeekV4AttentionOps,
    _mhc_post,
    _mhc_pre,
    _project_common,
    _project_output,
    _project_query,
    _rmsnorm,
)
from infer.models.deepseek_v4_flash.ops.core import _attention, _fp8_gemm, _quantize
from infer.models.deepseek_v4_flash.ops.target import _hc_head


class DeepSeekV4DSparkOps:
    def __init__(self) -> None:
        layer = DeepSeekV4AttentionOps()
        self._core = layer._core
        self._writer = layer._writer

    def prepare_batch(
        self,
        batch,
        controls,
        committed_lengths,
        sampled_tokens,
        candidates,
        live_slots,
    ):
        from infer.models.deepseek_v4_flash.ops.stage import stage_deepseek_v4_dspark

        stage_deepseek_v4_dspark(
            batch,
            controls,
            committed_lengths,
            sampled_tokens,
            candidates,
            live_slots,
        )

    def commit_state(
        self,
        captured_hidden,
        accepted,
        controls,
        committed_lengths,
        anchor_hidden,
        commit_hidden,
        positions,
        window_slots,
    ):
        from infer.models.deepseek_v4_flash.ops.stage import commit_deepseek_v4_dspark

        commit_deepseek_v4_dspark(
            captured_hidden,
            accepted,
            controls,
            committed_lengths,
            anchor_hidden,
            commit_hidden,
            positions,
            window_slots,
        )

    def seed_anchors(self, captured_hidden, source_indices, state_slots, anchor_hidden):
        from infer.models.deepseek_v4_flash.ops.stage import (
            seed_deepseek_v4_dspark_anchors,
        )

        seed_deepseek_v4_dspark_anchors(
            captured_hidden,
            source_indices,
            state_slots,
            anchor_hidden,
        )

    def prepare(
        self,
        captured_hidden,
        batch,
        weights,
        target_embedding,
        workspace,
    ):
        main_hidden = self._project_captured(captured_hidden, weights, workspace)
        torch.index_select(
            target_embedding,
            0,
            batch.input_token_ids,
            out=workspace.layer.collapsed,
        )
        workspace.layer.streams_out.copy_(workspace.layer.collapsed.unsqueeze(1))
        return main_hidden

    def seed_windows(
        self,
        captured_hidden,
        positions,
        window_slots,
        cos_sin,
        windows,
        weights,
        workspace,
    ):
        main_hidden = self._project_captured(captured_hidden, weights, workspace)
        common = workspace.layer.attention
        rows = captured_hidden.shape[0]
        hidden_fp8 = common.hidden_fp8[:rows]
        hidden_scale = common.hidden_scale[:rows]
        projected = workspace.main_kv_projected[:rows]
        normalized = workspace.main_kv[:rows]
        for stage, window in zip(weights.stages, windows, strict=True):
            _quantize(main_hidden, hidden_fp8, hidden_scale)
            _fp8_gemm(
                hidden_fp8,
                stage.attention.qkv_a[1024:],
                hidden_scale,
                stage.attention.qkv_a_scale[8:],
                projected,
            )
            _rmsnorm(projected, stage.attention.kv_norm, normalized)
            self._writer(
                common.projected_q[:rows],
                normalized,
                workspace.query[:rows],
                window.view(-1, window.shape[-1]),
                window_slots,
                positions,
                cos_sin,
                deepseek_v4_flash.RMS_NORM_EPS,
            )

    def stage(
        self,
        streams,
        weights,
        main_hidden,
        batch,
        persistent_window,
        workspace,
        mega_buffer,
    ):
        layer = workspace.layer
        common = layer.attention
        groups = batch.anchor_token_ids.shape[0]
        rows = batch.input_token_ids.shape[0]
        if not rows:
            self._core._decode_moe(
                common.normalized,
                batch.input_token_ids,
                weights.ffn,
                layer.ffn,
                mega_buffer,
                False,
            )
            return layer.streams_out

        _mhc_pre(streams, weights.attention_mhc, layer)
        _project_common(layer.collapsed, weights.attention, common)
        _project_query(
            weights.attention,
            common,
            common.projected_q.view(rows, -1),
        )

        main_hidden_fp8 = common.hidden_fp8[:groups]
        main_hidden_scale = common.hidden_scale[:groups]
        _quantize(main_hidden, main_hidden_fp8, main_hidden_scale)
        main_kv_projected = workspace.main_kv_projected[:groups]
        main_kv = workspace.main_kv[:groups]
        _fp8_gemm(
            main_hidden_fp8,
            weights.attention.qkv_a[1024:],
            main_hidden_scale,
            weights.attention.qkv_a_scale[8:],
            main_kv_projected,
        )
        _rmsnorm(
            main_kv_projected,
            weights.attention.kv_norm,
            main_kv,
        )
        self._writer(
            common.projected_q[:groups],
            main_kv,
            workspace.query[:groups],
            persistent_window.view(-1, persistent_window.shape[-1]),
            batch.persistent_slots,
            batch.anchor_positions,
            batch.cos_sin,
            deepseek_v4_flash.RMS_NORM_EPS,
        )
        self._writer(
            common.projected_q,
            common.normalized_kv,
            workspace.query,
            workspace.block_cache,
            batch.block_slots,
            batch.positions,
            batch.cos_sin,
            deepseek_v4_flash.RMS_NORM_EPS,
        )

        raw = DeepSeekV4RawAttention(
            workspace.query,
            persistent_window.view(-1, persistent_window.shape[-1]),
            batch.context_indices,
            batch.context_lengths,
            weights.attention.sink,
        )
        attended, _ = _attention(
            raw,
            workspace.block_cache,
            batch.block_indices,
            batch.block_lengths,
            batch.metadata,
            deepseek_v4_flash.DSPARK_PAGE_TOKENS,
            deepseek_v4_flash.NUM_QUERY_HEADS,
        )
        projected = _project_output(attended, weights.attention, common, batch, None)
        _mhc_post(projected, streams, layer, layer.streams_mid)
        _mhc_pre(layer.streams_mid, weights.ffn_mhc, layer)
        _rmsnorm(layer.collapsed, weights.ffn_norm, common.normalized)
        ffn = self._core._decode_moe(
            common.normalized,
            batch.input_token_ids,
            weights.ffn,
            layer.ffn,
            mega_buffer,
            False,
        )
        _mhc_post(ffn, layer.streams_mid, layer, layer.streams_out)
        return layer.streams_out

    def head(self, streams, batch, weights, target_head, workspace):
        groups = batch.anchor_token_ids.shape[0]
        block = deepseek_v4_flash.DSPARK_BLOCK_SIZE
        rows = groups * block
        if not groups:
            return DeepSeekV4DSparkOutput(
                batch.output_token_ids,
                workspace.confidence,
                workspace.head_hidden.view(groups, block, deepseek_v4_flash.HIDDEN_SIZE),
            )
        workspace.head_hidden.copy_(
            _hc_head(
                streams,
                weights.head_mhc.fn,
                weights.head_mhc.scale,
                weights.head_mhc.base,
            )
        )
        _rmsnorm(workspace.head_hidden, weights.final_norm, workspace.layer.collapsed)
        torch.mm(
            workspace.layer.collapsed,
            target_head.T,
            out=workspace.base_logits,
        )

        base_logits = workspace.base_logits.view(groups, block, -1)
        previous = batch.anchor_token_ids
        for step in range(block):
            torch.index_select(
                weights.markov_embedding,
                0,
                previous,
                out=workspace.markov_hidden,
            )
            torch.mm(
                workspace.markov_hidden,
                weights.markov_projection.T,
                out=workspace.markov_logits,
            )
            workspace.markov_logits.add_(base_logits[:, step])
            torch.argmax(
                workspace.markov_logits,
                dim=1,
                out=batch.output_token_ids[:, step],
            )
            previous = batch.output_token_ids[:, step]

        workspace.confidence_previous[:, 0].copy_(batch.anchor_token_ids)
        workspace.confidence_previous[:, 1:].copy_(batch.output_token_ids[:, :-1])
        workspace.confidence_features[:, : deepseek_v4_flash.HIDDEN_SIZE].copy_(
            workspace.head_hidden
        )
        torch.index_select(
            weights.markov_embedding,
            0,
            workspace.confidence_previous.view(-1),
            out=workspace.confidence_features[:, deepseek_v4_flash.HIDDEN_SIZE :],
        )
        torch.mm(
            workspace.confidence_features,
            weights.confidence.T,
            out=workspace.confidence.view(rows, 1),
        )
        torch.sigmoid(workspace.confidence, out=workspace.confidence)
        active = batch.active_count
        return DeepSeekV4DSparkOutput(
            batch.output_token_ids[:active],
            workspace.confidence[:active],
            workspace.head_hidden.view(groups, block, -1)[:active],
        )

    def _project_captured(self, captured_hidden, weights, workspace):
        rows = captured_hidden.shape[0]
        expected = (
            rows,
            len(deepseek_v4_flash.DSPARK_TARGET_LAYER_IDS) * deepseek_v4_flash.HIDDEN_SIZE,
        )
        if captured_hidden.shape != expected or rows > workspace.main_hidden.shape[0]:
            raise ValueError(f"DSpark captured hidden must fit shape {expected}")
        if not rows:
            return workspace.main_hidden[:0]
        input_fp8 = workspace.main_input_fp8[:rows]
        input_scale = workspace.main_input_scale[:rows]
        projected = workspace.main_projected[:rows]
        hidden = workspace.main_hidden[:rows]
        _quantize(captured_hidden, input_fp8, input_scale)
        _fp8_gemm(
            input_fp8,
            weights.main_projection,
            input_scale,
            weights.main_projection_scale,
            projected,
        )
        _rmsnorm(projected, weights.main_norm, hidden)
        return hidden

from dataclasses import fields

import torch
import torch.nn.functional as F
from flashinfer.norm import rmsnorm

from infer.models.deepseek_v4_flash.model import (
    DSPARK_TARGET_LAYER_IDS,
    HIDDEN_SIZE,
    MHC_EPS,
    MHC_STREAMS,
    RMS_NORM_EPS,
    TEP_SIZE,
    VOCAB_SIZE,
)
from infer.models.deepseek_v4_flash.ops.attention import (
    DeepSeekV4AttentionOps,
    DeepSeekV4TEP4AttentionOps,
)
from infer.models.deepseek_v4_flash.ops.core import _require_tensor
from infer.models.deepseek_v4_flash.target import (
    DeepSeekV4EndpointWorkspace,
    DeepSeekV4TargetWeights,
    endpoint_workspace_shapes,
    tep4_endpoint_workspace_shapes,
)


class DeepSeekV4TargetOps:
    def __init__(self) -> None:
        self.layer = DeepSeekV4AttentionOps()
        self._process_group = None

    def prefill_embedding(
        self,
        token_ids: torch.Tensor,
        weights: DeepSeekV4TargetWeights[torch.Tensor],
        collapsed: torch.Tensor,
        streams: torch.Tensor,
        local_tokens: torch.Tensor | None = None,
        select: torch.Tensor | None = None,
    ) -> torch.Tensor:
        tokens = token_ids.shape[0]
        device = weights.embedding.device
        vocab = VOCAB_SIZE // TEP_SIZE if local_tokens is not None else VOCAB_SIZE
        _require_tensor(weights.embedding, torch.bfloat16, (vocab, HIDDEN_SIZE), device)
        _require_tensor(token_ids, torch.int64, (tokens,), device)
        _require_tensor(collapsed, torch.bfloat16, (tokens, HIDDEN_SIZE), device)
        _require_tensor(
            streams, torch.bfloat16, (tokens, MHC_STREAMS, HIDDEN_SIZE), device
        )
        if local_tokens is not None:
            _require_tensor(local_tokens, torch.int64, (tokens,), device)
            _require_tensor(select, torch.bool, (tokens,), device)
        return self._embedding(
            token_ids, weights, collapsed, streams, local_tokens, select
        )

    def select_tokens(self, logits, _workspace, output) -> None:
        torch.argmax(logits, dim=1, out=output)

    def decode_embedding(
        self,
        token_ids: torch.Tensor,
        weights: DeepSeekV4TargetWeights[torch.Tensor],
        workspace: DeepSeekV4EndpointWorkspace[torch.Tensor],
    ) -> torch.Tensor:
        batch_size = token_ids.shape[0] if token_ids.ndim == 1 else 0
        _validate_endpoint(weights, workspace, batch_size)
        _require_tensor(token_ids, torch.int64, (batch_size,), weights.embedding.device)
        return self._embedding(
            token_ids, weights, workspace.collapsed, workspace.streams, None, None
        )

    def _embedding(self, token_ids, weights, collapsed, streams, local_tokens, select):
        if local_tokens is None:
            torch.index_select(weights.embedding, 0, token_ids, out=collapsed)
        else:
            local_vocab = VOCAB_SIZE // TEP_SIZE
            torch.div(token_ids, local_vocab, rounding_mode="floor", out=local_tokens)
            torch.eq(local_tokens, self._rank, out=select)
            torch.remainder(token_ids, local_vocab, out=local_tokens)
            torch.index_select(weights.embedding, 0, local_tokens, out=collapsed)
            collapsed.mul_(select.unsqueeze(1))
            torch.distributed.all_reduce(collapsed, group=self._process_group)
        streams.copy_(collapsed.unsqueeze(1))
        return streams

    def decode_head(
        self,
        streams: torch.Tensor,
        weights: DeepSeekV4TargetWeights[torch.Tensor],
        workspace: DeepSeekV4EndpointWorkspace[torch.Tensor],
    ) -> torch.Tensor:
        batch_size = streams.shape[0] if streams.ndim == 3 else 0
        _validate_endpoint(weights, workspace, batch_size)
        head_output = _decode_head(
            streams, weights, workspace.normalized, workspace.head_output
        )
        workspace.logits.copy_(head_output)
        return workspace.logits

    def capture_hidden(
        self,
        streams: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        if output.shape != (streams.shape[0], HIDDEN_SIZE) or output.stride() != (
            len(DSPARK_TARGET_LAYER_IDS) * HIDDEN_SIZE,
            1,
        ):
            raise ValueError("capture output has the wrong layout")
        if output.dtype != torch.bfloat16 or output.device != streams.device:
            raise TypeError("capture output has the wrong type")
        torch.mean(streams, dim=1, out=output)

    def accept_greedy(self, verify: object) -> None:
        verify.accepted.fill_(1)
        for step in range(verify.matches.shape[1]):
            match = verify.matches[:, step]
            torch.eq(
                verify.candidates[:, step + 1],
                verify.target_tokens[:, step],
                out=match,
            )
            if step:
                match.logical_and_(verify.matches[:, step - 1])
            torch.ne(verify.target_tokens[:, step], 1, out=verify.non_eos)
            verify.non_eos.logical_or_(verify.staging.device.ignore_eos)
            match.logical_and_(verify.non_eos)
            verify.accepted.add_(match)
        torch.minimum(
            verify.accepted,
            verify.staging.device.remaining,
            out=verify.accepted,
        )
        verify.accepted.mul_(verify.staging.device.active)

    def publish_verified(self, verify: object) -> None:
        state_slots = verify.staging.device.state_slots
        verify.records[:, 0].copy_(verify.accepted)
        verify.records[:, 1:].copy_(verify.target_tokens)
        verify.result_records.index_copy_(0, state_slots, verify.records)

        verify.tail_indices[:, 0].copy_(verify.accepted)
        verify.tail_indices.sub_(1).clamp_min_(0)
        torch.gather(
            verify.target_tokens,
            1,
            verify.tail_indices,
            out=verify.tail_tokens,
        )
        verify.sampled_tokens.index_copy_(0, state_slots, verify.tail_tokens[:, 0])
        torch.index_select(
            verify.committed_lengths,
            0,
            state_slots,
            out=verify.lengths,
        )
        verify.lengths.add_(verify.accepted)
        verify.committed_lengths.index_copy_(0, state_slots, verify.lengths)


class DeepSeekV4TEP4TargetOps(DeepSeekV4TargetOps):
    def __init__(self, process_group: object) -> None:
        self.layer = DeepSeekV4TEP4AttentionOps(process_group)
        self._process_group = process_group
        self._rank = torch.distributed.get_rank(process_group)

    def decode_embedding(self, token_ids, weights, workspace) -> torch.Tensor:
        batch_size = token_ids.shape[0] if token_ids.ndim == 1 else 0
        _validate_endpoint(weights, workspace, batch_size, tensor_parallel=True)
        _require_tensor(token_ids, torch.int64, (batch_size,), weights.embedding.device)
        return self._embedding(
            token_ids,
            weights,
            workspace.collapsed,
            workspace.streams,
            workspace.tokens,
            workspace.select,
        )

    def decode_head(self, streams, weights, workspace) -> torch.Tensor:
        batch_size = streams.shape[0] if streams.ndim == 3 else 0
        _validate_endpoint(weights, workspace, batch_size, tensor_parallel=True)
        return _decode_head(streams, weights, workspace.normalized, workspace.logits)

    def distributed_argmax(self, logits, workspace) -> torch.Tensor:
        torch.max(
            logits,
            dim=1,
            out=(workspace.local_values, workspace.local_indices),
        )
        workspace.local_indices.add_(self._rank * (VOCAB_SIZE // TEP_SIZE))
        workspace.local_candidates[:, 0].copy_(workspace.local_values)
        workspace.local_candidates[:, 1].copy_(workspace.local_indices)
        torch.distributed.all_gather_single(
            workspace.gathered_candidates,
            workspace.local_candidates,
            group=self._process_group,
        )
        candidates = workspace.gathered_candidates.view(TEP_SIZE, logits.shape[0], 2)
        workspace.best_values.copy_(candidates[0, :, 0])
        workspace.tokens.copy_(candidates[0, :, 1])
        for rank in range(1, TEP_SIZE):
            candidate = candidates[rank]
            torch.gt(candidate[:, 0], workspace.best_values, out=workspace.select)
            workspace.local_indices.copy_(candidate[:, 1])
            torch.where(
                workspace.select,
                candidate[:, 0],
                workspace.best_values,
                out=workspace.best_values,
            )
            torch.where(
                workspace.select,
                workspace.local_indices,
                workspace.tokens,
                out=workspace.tokens,
            )
        return workspace.tokens

    def select_tokens(self, logits, workspace, output) -> None:
        output.copy_(self.distributed_argmax(logits, workspace))


def _decode_head(streams, weights, normalized, output):
    batch_size = streams.shape[0]
    _require_tensor(
        streams,
        torch.bfloat16,
        (batch_size, MHC_STREAMS, HIDDEN_SIZE),
        weights.embedding.device,
    )
    if batch_size:
        collapsed = _hc_head(
            streams,
            weights.head_mhc.fn,
            weights.head_mhc.scale,
            weights.head_mhc.base,
        )
        rmsnorm(
            collapsed,
            weights.final_norm,
            eps=RMS_NORM_EPS,
            out=normalized,
            enable_pdl=True,
        )
        torch.mm(normalized, weights.head.T, out=output)
    return output


def _hc_head(hidden_states, hc_fn, hc_scale, hc_base):
    shape, dtype = hidden_states.size(), hidden_states.dtype
    x = hidden_states.flatten(1).float()
    rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + RMS_NORM_EPS)
    mixes = F.linear(x, hc_fn.float()) * rsqrt
    pre = torch.sigmoid(mixes * hc_scale.float() + hc_base.float()) + MHC_EPS
    return torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=1).to(dtype)


def _validate_endpoint(
    weights, workspace, batch_size, *, tensor_parallel=False
) -> None:
    expected = (
        tep4_endpoint_workspace_shapes(batch_size)
        if tensor_parallel
        else endpoint_workspace_shapes(batch_size)
    )
    device = weights.embedding.device
    vocab = VOCAB_SIZE // TEP_SIZE if tensor_parallel else VOCAB_SIZE
    _require_tensor(weights.embedding, torch.bfloat16, (vocab, HIDDEN_SIZE), device)
    _require_tensor(weights.head_mhc.base, torch.float32, (MHC_STREAMS,), device)
    _require_tensor(
        weights.head_mhc.fn,
        torch.float32,
        (MHC_STREAMS, MHC_STREAMS * HIDDEN_SIZE),
        device,
    )
    _require_tensor(weights.head_mhc.scale, torch.float32, (1,), device)
    _require_tensor(weights.final_norm, torch.bfloat16, (HIDDEN_SIZE,), device)
    _require_tensor(weights.head, torch.bfloat16, (vocab, HIDDEN_SIZE), device)
    fp32 = (
        {"local_candidates", "gathered_candidates", "best_values"}
        if tensor_parallel
        else {"logits"}
    )
    int64 = {"local_indices", "tokens"}
    bool_ = {"select"}
    for field in fields(expected):
        dtype = torch.float32 if field.name in fp32 else torch.bfloat16
        dtype = torch.int64 if field.name in int64 else dtype
        dtype = torch.bool if field.name in bool_ else dtype
        _require_tensor(
            getattr(workspace, field.name),
            dtype,
            getattr(expected, field.name),
            device,
        )

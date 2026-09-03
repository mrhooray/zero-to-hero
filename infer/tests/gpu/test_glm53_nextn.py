import ast
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pytest

import torch

from infer.models.glm53_flash import checkpoint as glm53_checkpoint
from infer.models.glm53_flash import model as glm53_flash
from infer.models.glm53_flash import nextn
from infer.models.glm53_flash import target as glm53_target
from infer.models.glm53_flash.codec import EOS_TOKEN_IDS

pytestmark = pytest.mark.gpu

ROOT = Path(__file__).resolve().parents[2]
OPS_SOURCE = (ROOT / "src/infer/models/glm53_flash/ops/core.py").read_text()
VERIFY_WIDTH = glm53_flash.GLM53_TARGET_VERIFY_WIDTH


def _accept_glm53_greedy_chain(
    candidate_token_ids: tuple[int, ...],
    target_token_ids: tuple[int, ...],
    max_append_tokens: int,
) -> tuple[int, ...]:
    """Return the accepted target prefix for one root and three draft tokens."""

    width = glm53_flash.GLM53_TARGET_VERIFY_WIDTH
    if len(candidate_token_ids) != width or len(target_token_ids) != width:
        raise ValueError(f"GLM greedy acceptance requires two {width}-token chains")
    if type(max_append_tokens) is not int or not 1 <= max_append_tokens <= width:
        raise ValueError(f"max_append_tokens must be in [1, {width}]")

    accepted = 1
    for candidate, target in zip(
        candidate_token_ids[1:], target_token_ids[:-1], strict=True
    ):
        if candidate != target:
            break
        accepted += 1

    output_token_ids = target_token_ids[: min(accepted, max_append_tokens)]
    for index, token_id in enumerate(output_token_ids):
        if token_id in EOS_TOKEN_IDS:
            return output_token_ids[: index + 1]
    return output_token_ids


def _make_prefill_staging():
    capacity = glm53_flash.KDA_CHUNK_SIZE
    sequences = max(glm53_flash.GLM53_TARGET_DECODE_BATCH_SIZES)
    return SimpleNamespace(
        cu_seqlens=torch.zeros(sequences + 1, dtype=torch.int32),
        state_indices_int64=torch.zeros(sequences, dtype=torch.int64),
        has_initial=torch.zeros(sequences, dtype=torch.bool),
        start_tokens=torch.zeros(sequences, dtype=torch.int32),
        end_tokens=torch.zeros(sequences, dtype=torch.int32),
        sequence_ids=torch.empty(capacity, dtype=torch.int32),
        token_state_slots=torch.empty(capacity, dtype=torch.int64),
        active=torch.empty(capacity, dtype=torch.uint8),
        raw_lengths=torch.empty(capacity, dtype=torch.int32),
        block_table=torch.empty((capacity, 1), dtype=torch.int32),
        null_token="null_token",
        sample_indices=torch.zeros(sequences, dtype=torch.int64),
        sample_state_indices=torch.zeros(sequences, dtype=torch.int64),
    )


def make_target(live_slots: int = 3, physical_blocks: int = 3):
    def batch(groups: int, rows: int):
        return SimpleNamespace(
            token_ids=torch.zeros(rows, dtype=torch.int64),
            state_indices=torch.zeros(groups, dtype=torch.int32),
            state_indices_int64=torch.zeros(rows, dtype=torch.int64),
            sparse_mla=glm53_flash.SparseMLADecodeBatch(
                active=torch.zeros(rows, dtype=torch.uint8),
                raw_lengths=torch.zeros(rows, dtype=torch.int32),
                state_slots=torch.zeros(rows, dtype=torch.int32),
                block_table=torch.zeros(
                    (rows, glm53_flash.SPARSE_MLA_COMPOUND_BLOCKS), dtype=torch.int32
                ),
                null_token=torch.zeros(1, dtype=torch.int32),
            ),
        )

    history = glm53_flash.SparseMLAHistory(
        latent=torch.zeros(
            (
                physical_blocks * len(glm53_flash.SPARSE_MLA_BF16_LATENT_PAGES),
                *glm53_flash.SPARSE_MLA_BF16_LATENT_PAGES[0],
            ),
            dtype=torch.bfloat16,
        ),
        index_cache=torch.zeros(
            (physical_blocks, glm53_flash.SPARSE_MLA_INDEX_PAGE_BYTES),
            dtype=torch.uint8,
        ),
        tail_key=torch.zeros(
            (2, live_slots, *glm53_flash.SPARSE_MLA_INDEX_TAIL_K_BF16_SHAPE),
            dtype=torch.bfloat16,
        ),
        tail_gate=torch.zeros(
            (2, live_slots, *glm53_flash.SPARSE_MLA_INDEX_TAIL_K_BF16_SHAPE),
            dtype=torch.bfloat16,
        ),
    )
    target = SimpleNamespace(
        attention_tp_size=glm53_flash.TP_SIZE,
        state=SimpleNamespace(kda_layers=(), sparse_mla_layers=(history,)),
        sampled_tokens=torch.zeros(live_slots + 32, dtype=torch.int64),
        execution_tables=torch.zeros(
            (live_slots + 32, glm53_flash.SPARSE_MLA_COMPOUND_BLOCKS),
            dtype=torch.int32,
        ),
        committed_lengths=torch.zeros(live_slots, dtype=torch.int32),
        prefill_staging="prefill_staging",
        prefill_workspace=SimpleNamespace(
            endpoint=SimpleNamespace(
                token=torch.empty(glm53_flash.KDA_CHUNK_SIZE, dtype=torch.int64)
            )
        ),
        prefix_snapshots=SimpleNamespace(
            committed_lengths=torch.empty(0, dtype=torch.int32)
        ),
    )
    target.decode = {
        groups: SimpleNamespace(batches=(batch(groups, groups), batch(groups, groups)))
        for groups in glm53_flash.GLM53_TARGET_DECODE_BATCH_SIZES
    }
    max_rows = max(glm53_flash.GLM53_TARGET_DECODE_BATCH_SIZES) * VERIFY_WIDTH
    transaction_key = torch.zeros(
        (
            max_rows,
            2,
            1,
            *glm53_flash.SPARSE_MLA_INDEX_TAIL_K_BF16_SHAPE,
        ),
        dtype=torch.bfloat16,
    )
    transaction_gate = torch.zeros_like(transaction_key)
    target.verify = {
        (groups, parity): SimpleNamespace(
            batch=batch(groups, groups * VERIFY_WIDTH),
            transaction=SimpleNamespace(
                kda_layers=(),
                sparse_mla_layers=(
                    SimpleNamespace(
                        tail_key=transaction_key[: groups * VERIFY_WIDTH],
                        tail_gate=transaction_gate[: groups * VERIFY_WIDTH],
                    ),
                ),
            ),
        )
        for groups in glm53_flash.GLM53_TARGET_DECODE_BATCH_SIZES
        for parity in range(2)
    }
    return target


def make_speculative_runtime(groups: int):
    target = make_target(live_slots=groups + 1)
    workspaces = {}

    def view(_arena, rows, attention_tp_size):
        assert attention_tp_size == glm53_flash.TP_SIZE
        workspace = SimpleNamespace(
            endpoint=SimpleNamespace(
                token=torch.zeros(rows, dtype=torch.int64),
                normalized=torch.zeros(
                    (rows, glm53_flash.HIDDEN_SIZE), dtype=torch.bfloat16
                ),
            ),
            sparse_mla_decode=SimpleNamespace(
                selected=torch.zeros(
                    (rows, glm53_flash.SPARSE_MLA_INDEX_TOP_K), dtype=torch.int32
                )
            ),
        )
        workspaces[rows] = workspace
        return workspace

    with (
        patch.object(nextn, "_nextn_workspace_arena", return_value="arena"),
        patch.object(nextn, "_nextn_workspace_view", side_effect=view),
    ):
        runtime = nextn._allocate_glm53_nextn_runtime(torch, target)
    for (bucket, _parity), verify in target.verify.items():
        verify.workspace = workspaces[bucket * VERIFY_WIDTH]
    return target, runtime, workspaces


class FakeNextNModel:
    def __init__(self) -> None:
        self.calls = []

    def forward(
        self,
        token_ids,
        target_normalized,
        batch,
        history,
        workspace,
        head_workspace,
        endpoint,
        process_group,
        all_reduce,
        **options,
    ):
        self.calls.append(
            SimpleNamespace(
                token_ids=token_ids.clone(),
                target_normalized=target_normalized.clone(),
                raw_lengths=batch.raw_lengths.clone(),
                history=history,
                history_tail_key=history.tail_key.clone(),
                workspace=workspace,
                options=options,
            )
        )
        rows = token_ids.shape[0]
        if options.get("shared_selected") is None:
            workspace.sparse_mla_decode.selected.copy_(
                torch.arange(rows, dtype=torch.int32)[:, None]
            )
        transaction = options.get("transaction")
        if transaction is not None:
            values = torch.arange(rows, dtype=torch.bfloat16).view(rows, 1, 1, 1, 1)
            transaction.tail_key.copy_(values + 100)
            transaction.tail_gate.copy_(values + 200)
        workspace.endpoint.token.copy_(token_ids).add_(10)
        workspace.endpoint.normalized.copy_(target_normalized).add_(1)
        return workspace.endpoint.token, workspace.endpoint.normalized


class GLM53NextNResidentRuntimeTest(unittest.TestCase):
    def test_full_history_buckets_parities(self) -> None:
        target = make_target()

        def view(arena, rows, attention_tp_size):
            self.assertEqual(attention_tp_size, glm53_flash.TP_SIZE)
            return arena, rows

        with (
            patch.object(nextn, "_nextn_workspace_arena", return_value="arena"),
            patch.object(
                nextn,
                "_nextn_workspace_view",
                side_effect=view,
            ),
        ):
            runtime = nextn._allocate_glm53_nextn_runtime(torch, target)

        reference = target.state.sparse_mla_layers[0]
        history = runtime.state.history
        working = runtime.working_state.history
        self.assertEqual(history.latent.shape, reference.latent.shape)
        self.assertEqual(history.index_cache.shape, reference.index_cache.shape)
        self.assertEqual(history.tail_key.shape, reference.tail_key.shape)
        self.assertNotEqual(history.latent.data_ptr(), reference.latent.data_ptr())
        self.assertIs(working.latent, history.latent)
        self.assertIs(working.index_cache, history.index_cache)
        self.assertNotEqual(working.tail_key.data_ptr(), history.tail_key.data_ptr())
        self.assertIs(runtime.execution_tables, target.execution_tables)
        self.assertIs(runtime.committed_lengths, target.committed_lengths)
        self.assertIs(runtime.prefill_staging, target.prefill_staging)
        self.assertIs(
            runtime.prefill_token_ids,
            target.prefill_workspace.endpoint.token,
        )

        buckets = glm53_flash.GLM53_TARGET_DECODE_BATCH_SIZES
        self.assertEqual(set(runtime.draft_batches), set(buckets))
        self.assertEqual(set(runtime.verify_batches), set(buckets))
        for groups in buckets:
            for parity in range(2):
                draft = runtime.draft_batches[groups][parity]
                verify = runtime.verify_batches[groups][parity]
                self.assertEqual(draft.token_ids.shape, (groups,))
                self.assertEqual(draft.state_indices.shape, (groups,))
                self.assertEqual(draft.sparse_mla.raw_lengths.shape, (groups,))
                self.assertEqual(verify.token_ids.shape, (groups * VERIFY_WIDTH,))
                self.assertEqual(
                    verify.sparse_mla.raw_lengths.shape,
                    (groups * VERIFY_WIDTH,),
                )
                self.assertEqual(
                    verify.sparse_mla.block_table.shape,
                    (groups * VERIFY_WIDTH, glm53_flash.SPARSE_MLA_COMPOUND_BLOCKS),
                )
                self.assertEqual(
                    draft.token_ids.data_ptr(),
                    target.decode[groups].batches[parity].token_ids.data_ptr(),
                )
                self.assertEqual(
                    draft.state_indices.data_ptr(),
                    target.decode[groups].batches[parity].state_indices.data_ptr(),
                )
                self.assertEqual(
                    draft.sparse_mla.block_table.data_ptr(),
                    target.decode[groups]
                    .batches[parity]
                    .sparse_mla.block_table.data_ptr(),
                )
                self.assertEqual(
                    verify.token_ids.data_ptr(),
                    target.verify[groups, parity].batch.token_ids.data_ptr(),
                )
                self.assertEqual(
                    verify.sparse_mla.block_table.data_ptr(),
                    target.verify[
                        groups, parity
                    ].batch.sparse_mla.block_table.data_ptr(),
                )
        self.assertNotEqual(
            runtime.draft_batches[32][0].state_indices.data_ptr(),
            runtime.draft_batches[32][1].state_indices.data_ptr(),
        )

        self.assertEqual(set(runtime.transactions), set(buckets))
        for groups in buckets:
            transaction = runtime.transactions[groups]
            self.assertEqual(
                transaction.tail_key.shape,
                (
                    groups * VERIFY_WIDTH,
                    2,
                    1,
                    *glm53_flash.SPARSE_MLA_INDEX_TAIL_K_BF16_SHAPE,
                ),
            )
            self.assertEqual(
                transaction.tail_key.data_ptr(),
                target.verify[groups, 0]
                .transaction.sparse_mla_layers[0]
                .tail_key.data_ptr(),
            )
        self.assertEqual(runtime.draft_hidden.shape, (3, glm53_flash.HIDDEN_SIZE))
        self.assertEqual(
            runtime.draft_pool_ids.shape, (3, glm53_flash.SPARSE_MLA_INDEX_TOP_K)
        )
        self.assertEqual(runtime.candidate_token_ids.shape, (3, VERIFY_WIDTH))
        self.assertIsNotNone(runtime.prefix_snapshots)
        assert runtime.prefix_snapshots is not None
        self.assertEqual(
            runtime.prefix_snapshots.draft_hidden.shape, (0, glm53_flash.HIDDEN_SIZE)
        )
        self.assertEqual(
            set(runtime.workspaces),
            {
                *buckets,
                *(groups * VERIFY_WIDTH for groups in buckets),
                glm53_flash.KDA_CHUNK_SIZE,
            },
        )

    def test_prefix_snapshot_round_trip_copies_nextn_state_and_private_tail(
        self,
    ) -> None:
        target = make_target(live_slots=2, physical_blocks=3)
        target.prefix_snapshots = SimpleNamespace(
            committed_lengths=torch.zeros(1, dtype=torch.int32)
        )

        def view(arena, rows, attention_tp_size):
            self.assertEqual(attention_tp_size, glm53_flash.TP_SIZE)
            return arena, rows

        with (
            patch.object(nextn, "_nextn_workspace_arena", return_value="arena"),
            patch.object(
                nextn,
                "_nextn_workspace_view",
                side_effect=view,
            ),
        ):
            runtime = nextn._allocate_glm53_nextn_runtime(torch, target)

        source_slot, destination_slot, snapshot_slot = 0, 1, 2
        source_block, destination_block = 0, 1
        history = runtime.state.history
        source_page = 2 * (source_block + 1)
        history.latent[source_page].fill_(11)
        history.latent[source_page + 1].fill_(12)
        history.index_cache[source_block + 1].fill_(13)
        history.tail_key[:, source_slot].fill_(14)
        history.tail_gate[:, source_slot].fill_(15)
        runtime.draft_hidden[source_slot].fill_(16)

        runtime.capture_prefix(source_slot, snapshot_slot, source_block)

        history.latent[source_page : source_page + 2].zero_()
        history.index_cache[source_block + 1].zero_()
        history.tail_key[:, source_slot].zero_()
        history.tail_gate[:, source_slot].zero_()
        runtime.draft_hidden[source_slot].zero_()
        runtime.draft_pool_ids[destination_slot].fill_(17)
        runtime.candidate_token_ids[destination_slot].fill_(18)

        runtime.restore_prefix(snapshot_slot, destination_slot, destination_block)

        destination_page = 2 * (destination_block + 1)
        self.assertTrue(torch.all(history.latent[destination_page] == 11))
        self.assertTrue(torch.all(history.latent[destination_page + 1] == 12))
        self.assertTrue(torch.all(history.index_cache[destination_block + 1] == 13))
        self.assertTrue(torch.all(history.tail_key[:, destination_slot] == 14))
        self.assertTrue(torch.all(history.tail_gate[:, destination_slot] == 15))
        self.assertTrue(torch.all(runtime.draft_hidden[destination_slot] == 16))
        self.assertTrue(torch.all(runtime.draft_pool_ids[destination_slot] == 17))
        self.assertTrue(torch.all(runtime.candidate_token_ids[destination_slot] == 18))

        history.latent[destination_page : destination_page + 2].zero_()
        runtime.draft_hidden[destination_slot].zero_()
        runtime.restore_prefix(snapshot_slot, destination_slot, destination_block)
        self.assertTrue(torch.all(history.latent[destination_page] == 11))
        self.assertTrue(torch.all(runtime.draft_hidden[destination_slot] == 16))

    def test_packed_catchup_shifts_fresh_and_resumed_sequences(self) -> None:
        target = make_target(live_slots=4)
        staging = _make_prefill_staging()
        history = target.state.sparse_mla_layers[0]
        workspace = SimpleNamespace(
            residual=torch.empty((5, glm53_flash.HIDDEN_SIZE), dtype=torch.bfloat16),
            normalized=torch.empty((4, glm53_flash.HIDDEN_SIZE), dtype=torch.bfloat16),
        )
        runtime = nextn.GLM53NextNRuntime(
            state=nextn.GLM53NextNState(history),
            working_state=nextn.GLM53NextNState(history),
            transactions={},
            workspaces={glm53_flash.KDA_CHUNK_SIZE: workspace},
            draft_batches={},
            verify_batches={},
            execution_tables=target.execution_tables,
            committed_lengths=target.committed_lengths,
            prefill_staging=staging,
            prefill_token_ids=torch.empty(glm53_flash.KDA_CHUNK_SIZE, dtype=torch.int64),
            draft_hidden=torch.zeros((4, glm53_flash.HIDDEN_SIZE), dtype=torch.bfloat16),
            draft_pool_ids=torch.empty(0),
            candidate_token_ids=torch.empty(0),
        )
        runtime.draft_hidden[0].fill_(99)
        runtime.draft_hidden[3].fill_(77)
        token_ids = torch.tensor((10, 11, 12, 20, 21, 30, 40), dtype=torch.int64)
        target_hidden = (
            torch.arange(7, dtype=torch.bfloat16)[:, None]
            .expand(-1, glm53_flash.HIDDEN_SIZE)
            .contiguous()
        )
        staging.cu_seqlens[:5].copy_(torch.tensor((0, 3, 5, 6, 7)))
        staging.start_tokens[:4].copy_(torch.tensor((0, 1, 0, 5)))
        staging.state_indices_int64[:4].copy_(torch.tensor((2, 0, 1, 3)))

        def compact(*arguments):
            (
                source_tokens,
                hidden,
                pending,
                cu_seqlens,
                starts,
                slots,
                end_tokens,
                has_pending,
                sample_indices,
                sample_slots,
                output_tokens,
                output_hidden,
                sequence_ids,
                token_slots,
                active,
                raw_lengths,
                total_tokens,
                sequence_count,
            ) = arguments
            self.assertIs(source_tokens, token_ids)
            self.assertIs(hidden, target_hidden)
            self.assertIs(pending, runtime.draft_hidden)
            self.assertEqual(total_tokens, 5)
            self.assertEqual(sequence_count, 4)
            self.assertEqual(cu_seqlens[:5].tolist(), [0, 3, 5, 6, 7])
            self.assertEqual(starts[:4].tolist(), [0, 1, 0, 5])
            self.assertEqual(slots[:4].tolist(), [2, 0, 1, 3])
            cu_seqlens.fill_(5)
            cu_seqlens[:4].copy_(torch.tensor((0, 2, 4, 5)))
            end_tokens[:3].copy_(torch.tensor((1, 3, 6)))
            starts[:3].copy_(torch.tensor((0, 0, 4)))
            slots[:3].copy_(torch.tensor((2, 0, 3)))
            has_pending[:3].copy_(torch.tensor((False, True, True)))
            sample_indices[:4].copy_(torch.tensor((2, 4, 5, 6)))
            sample_slots[:4].copy_(torch.tensor((2, 0, 1, 3)))
            output_tokens[:5].copy_(token_ids[[1, 2, 3, 4, 6]])
            output_hidden.copy_(
                torch.stack(
                    (
                        target_hidden[0],
                        target_hidden[1],
                        pending[0],
                        target_hidden[3],
                        pending[3],
                    )
                )
            )
            sequence_ids[:5].copy_(torch.tensor((0, 0, 1, 1, 2)))
            token_slots[:5].copy_(torch.tensor((2, 2, 0, 0, 3)))
            active[:5].fill_(1)
            raw_lengths[:5].copy_(torch.tensor((1, 2, 1, 2, 5)))

        compactor = ModuleType("infer.models.glm53_flash.ops.nextn_stage")
        compactor.glm53_stage_nextn_prefill = compact
        sparse = ModuleType("infer.models.glm53_flash.ops.sparse_mla_prefill")
        sparse.validate_glm53_sparse_mla_prefill_plan = (
            lambda start, total, slot, _table, **options: SimpleNamespace(
                start_token=start,
                total_tokens=total,
                state_slot=slot,
                **options,
            )
        )
        stage_sparse = Mock(return_value="sparse")
        sparse.stage_glm53_sparse_mla_prefill_batch = stage_sparse
        with patch.dict(
            sys.modules,
            {
                compactor.__name__: compactor,
                sparse.__name__: sparse,
            },
        ):
            batch = runtime.stage_prefill(
                token_ids,
                target_hidden,
                (3, 2, 1, 1),
                (0, 1, 0, 5),
                (2, 0, 1, 3),
            )

        self.assertEqual(batch.token_ids.tolist(), [11, 12, 20, 21, 40])
        self.assertEqual(batch.hidden[:, 0].tolist(), [0, 1, 99, 3, 77])
        self.assertEqual(batch.sparse_mla, "sparse")
        for name in (
            "sequence_ids",
            "token_state_slots",
            "active",
            "raw_lengths",
            "block_table",
        ):
            self.assertEqual(stage_sparse.call_args.kwargs[name].shape[0], 5)
        plans = stage_sparse.call_args.kwargs["plans"]
        self.assertEqual(
            tuple(
                (plan.start_token, plan.total_tokens, plan.state_slot, plan.has_initial)
                for plan in plans
            ),
            ((0, 2, 2, False), (0, 2, 0, False), (4, 1, 3, True)),
        )

        runtime.publish_prefill(batch)
        self.assertEqual(runtime.draft_hidden[:, 0].tolist(), [4, 5, 2, 6])

    def test_workspace_arena_reuses_target_scratch(self) -> None:
        flattened = MagicMock()
        selected = Mock()
        streams_out = MagicMock()
        streams_out.view.return_value = flattened
        flattened.__getitem__.return_value = selected
        selected.view.return_value = "fusion"
        prefill = SimpleNamespace(
            endpoint="endpoint",
            layer=SimpleNamespace(
                collapsed="residual",
                normalized="normalized",
                streams_out=streams_out,
            ),
            sparse_mla_projection="projection",
            sparse_mla_decode="decode",
            sparse_mla_output="output",
            sparse_ffn="ffn",
        )
        target = SimpleNamespace(
            prefill_workspace=prefill,
            execution_tables=SimpleNamespace(device="cuda:2"),
        )

        workspace = nextn._nextn_workspace_arena(target)

        streams_out.view.assert_called_once_with(-1)
        flattened.__getitem__.assert_called_once_with(
            slice(None, glm53_flash.KDA_CHUNK_SIZE * 2 * glm53_flash.HIDDEN_SIZE)
        )
        selected.view.assert_called_once_with(
            glm53_flash.KDA_CHUNK_SIZE, 2 * glm53_flash.HIDDEN_SIZE
        )
        self.assertEqual(workspace.endpoint, "endpoint")
        self.assertEqual(workspace.fusion, "fusion")
        self.assertEqual(workspace.residual, "residual")
        self.assertEqual(workspace.normalized, "normalized")
        self.assertEqual(workspace.sparse_mla_projection, "projection")
        self.assertEqual(workspace.sparse_mla_decode, "decode")
        self.assertEqual(workspace.sparse_mla_output, "output")
        self.assertEqual(workspace.sparse_ffn, "ffn")

    def test_batched_seed_and_two_recurrent_drafts(self) -> None:
        for groups in (1, 2, 16, 32):
            with self.subTest(groups=groups):
                _target, runtime, workspaces = make_speculative_runtime(groups)
                model = FakeNextNModel()
                active_count = groups if groups == 1 else groups - 1
                state_indices = torch.arange(groups, dtype=torch.int64)
                state_indices[active_count:] = groups + 7
                root_token_ids = torch.arange(groups, dtype=torch.int64) + 1000
                root_token_ids[active_count:] = -1
                runtime.committed_lengths.copy_(
                    torch.arange(groups + 1, dtype=torch.int32) * 128 + 1
                )
                runtime.execution_tables[:, 0].copy_(
                    torch.arange(runtime.execution_tables.shape[0])
                )
                runtime.draft_hidden.copy_(
                    torch.arange(groups + 1, dtype=torch.bfloat16)[:, None]
                )

                runtime.seed_candidates(
                    model,
                    root_token_ids,
                    state_indices,
                    active_count,
                    1,
                    "endpoint",
                    "group",
                    "all_reduce",
                )

                batch = runtime.draft_batches[groups][1]
                self.assertEqual(
                    batch.sparse_mla.active.tolist(),
                    [*((1,) * active_count), *((0,) * (groups - active_count))],
                )
                self.assertEqual(
                    model.calls[0].token_ids[active_count:].tolist(),
                    [root_token_ids[0].item()] * (groups - active_count),
                )
                safe_states = state_indices.clamp(0, groups)
                self.assertEqual(
                    model.calls[0].raw_lengths.tolist(),
                    runtime.committed_lengths[safe_states].tolist(),
                )
                self.assertEqual(
                    runtime.candidate_token_ids[:active_count, :2].tolist(),
                    torch.stack(
                        (
                            root_token_ids[:active_count],
                            root_token_ids[:active_count] + 10,
                        ),
                        dim=1,
                    ).tolist(),
                )
                self.assertEqual(
                    runtime.draft_hidden[:active_count, 0].tolist(),
                    (torch.arange(active_count) + 1).tolist(),
                )

                batch.sparse_mla.state_slots.copy_(
                    torch.arange(groups, dtype=torch.int32)
                )
                batch.sparse_mla.state_slots[active_count:].zero_()
                batch.sparse_mla.active.zero_()
                batch.sparse_mla.active[:active_count].fill_(1)
                batch.sparse_mla.raw_lengths.copy_(
                    torch.arange(groups, dtype=torch.int32) + 30
                )
                runtime.state.history.tail_key.fill_(3)
                runtime.state.history.tail_gate.fill_(4)
                candidates = runtime.draft_candidates(
                    model,
                    groups,
                    1,
                    "endpoint",
                    "group",
                    "all_reduce",
                )

                self.assertEqual(
                    candidates[:active_count].tolist(),
                    torch.stack(
                        (
                            root_token_ids[:active_count],
                            root_token_ids[:active_count] + 10,
                            root_token_ids[:active_count] + 20,
                            root_token_ids[:active_count] + 30,
                        ),
                        dim=1,
                    ).tolist(),
                )
                self.assertEqual(model.calls[-2].raw_lengths[0].item(), 30)
                self.assertEqual(model.calls[-1].raw_lengths[0].item(), 31)
                self.assertIs(
                    model.calls[-1].options["shared_selected"],
                    workspaces[groups].sparse_mla_decode.selected,
                )
                self.assertTrue(
                    torch.equal(
                        model.calls[-2].history_tail_key,
                        runtime.state.history.tail_key,
                    )
                )

    def test_reset_slot_clears_only_reused_resident_state(self) -> None:
        _target, runtime, _workspaces = make_speculative_runtime(2)
        runtime.state.history.tail_key.fill_(1)
        runtime.state.history.tail_gate.fill_(2)
        runtime.draft_hidden.fill_(3)
        runtime.draft_pool_ids.fill_(4)
        runtime.candidate_token_ids.fill_(5)

        runtime.reset_slot(1)

        for tensor in (
            runtime.state.history.tail_key[:, 1],
            runtime.state.history.tail_gate[:, 1],
            runtime.draft_hidden[1],
            runtime.draft_pool_ids[1],
            runtime.candidate_token_ids[1],
        ):
            self.assertEqual(torch.count_nonzero(tensor), 0)
        self.assertTrue(torch.all(runtime.state.history.tail_key[:, 0] == 1))
        self.assertTrue(torch.all(runtime.state.history.tail_gate[:, 0] == 2))
        self.assertTrue(torch.all(runtime.draft_hidden[0] == 3))
        self.assertTrue(torch.all(runtime.draft_pool_ids[0] == 4))
        self.assertTrue(torch.all(runtime.candidate_token_ids[0] == 5))

    def test_batched_verify_commit_selects_target_and_nextn_rows(self) -> None:
        def copy_rows(source, destination, accepted, state_slots, active):
            for row in range(accepted.shape[0]):
                if active[row]:
                    destination[state_slots[row]].copy_(
                        source[row * VERIFY_WIDTH + accepted[row] - 1]
                    )

        for groups in (1, 2, 16, 32):
            with self.subTest(groups=groups):
                target, runtime, workspaces = make_speculative_runtime(groups)
                model = FakeNextNModel()
                batch = runtime.verify_batches[groups][0]
                state_slots = torch.arange(groups, dtype=torch.int32)
                batch.state_indices.copy_(state_slots)
                shared = target.verify[groups, 0].workspace.endpoint
                target_token_ids = shared.token
                target_token_ids.copy_(
                    torch.arange(groups * VERIFY_WIDTH, dtype=torch.int64) + 1000
                )
                target_normalized = shared.normalized
                target_normalized.copy_(
                    torch.arange(groups * VERIFY_WIDTH, dtype=torch.bfloat16)[:, None]
                )
                target_hidden = target_normalized.clone()
                accepted = torch.arange(groups, dtype=torch.int32) % VERIFY_WIDTH + 1
                active = torch.ones(groups, dtype=torch.uint8)
                continuing = torch.ones(groups, dtype=torch.uint8)
                if groups > 1:
                    active[-1] = 0
                    continuing[-1] = 0
                if groups > 2:
                    continuing[-2] = 0
                output_token_ids = batch.token_ids.view(groups, VERIFY_WIDTH)
                output_token_ids.copy_(target_token_ids.view(groups, VERIFY_WIDTH))
                self.assertEqual(
                    target_token_ids.data_ptr(),
                    workspaces[groups * VERIFY_WIDTH].endpoint.token.data_ptr(),
                )
                self.assertEqual(
                    target_normalized.data_ptr(),
                    workspaces[groups * VERIFY_WIDTH].endpoint.normalized.data_ptr(),
                )
                self.assertNotEqual(
                    output_token_ids.data_ptr(), target_token_ids.data_ptr()
                )
                transaction = runtime.transactions[groups]
                rows = torch.arange(groups * VERIFY_WIDTH, dtype=torch.bfloat16).view(
                    -1, 1, 1, 1, 1
                )
                transaction.tail_key.copy_(rows + 10)
                transaction.tail_gate.copy_(rows + 20)
                runtime.state.history.tail_key.fill_(3)
                runtime.state.history.tail_gate.fill_(4)
                canonical_tail = runtime.state.history.tail_key.clone()
                target.commit_tables = tuple(
                    SimpleNamespace(
                        source_addresses=source,
                        destination_addresses=destination,
                        source_strides=None,
                        destination_strides=None,
                        row_words=source[0].numel() * source.element_size() // 4,
                    )
                    for parity in range(2)
                    for source, destination in (
                        (
                            target.verify[groups, 0]
                            .transaction.sparse_mla_layers[0]
                            .tail_key[:, parity, 0],
                            target.state.sparse_mla_layers[0].tail_key[parity],
                        ),
                        (
                            target.verify[groups, 0]
                            .transaction.sparse_mla_layers[0]
                            .tail_gate[:, parity, 0],
                            target.state.sparse_mla_layers[0].tail_gate[parity],
                        ),
                    )
                )

                def copy_table(source, destination, *args, row_words):
                    del row_words
                    copy_rows(source, destination, *args[-3:])

                speculate = ModuleType("infer.models.glm53_flash.ops.speculate")
                speculate.glm53_copy_verified_rows = copy_rows
                speculate.glm53_copy_verified_row_table = copy_table
                speculate.glm53_publish_accepted = Mock()
                with patch.dict(sys.modules, {speculate.__name__: speculate}):
                    glm53_target.GLM53TargetRuntime.publish_verified(
                        target,
                        target.verify[groups, 0],
                        accepted,
                        active,
                        output_token_ids,
                    )
                    target_tail = target.state.sparse_mla_layers[0].tail_key.clone()
                    runtime.verify_candidates(
                        model,
                        target_token_ids,
                        target_normalized,
                        groups,
                        0,
                        "endpoint",
                        "group",
                        "all_reduce",
                    )
                    runtime.candidate_token_ids.fill_(-1)
                    runtime.draft_hidden.fill_(-1)
                    runtime.draft_pool_ids.fill_(-1)
                    runtime.commit_candidates(
                        groups,
                        0,
                        output_token_ids,
                        accepted,
                        active,
                        continuing,
                    )

                self.assertIs(model.calls[-1].options["transaction"], transaction)
                self.assertTrue(
                    torch.equal(target_token_ids, output_token_ids.view(-1) + 10)
                )
                self.assertTrue(torch.equal(target_normalized, target_hidden + 1))
                self.assertTrue(
                    torch.equal(
                        model.calls[-1].history_tail_key,
                        canonical_tail,
                    )
                )
                self.assertTrue(
                    torch.equal(
                        target.state.sparse_mla_layers[0].tail_key,
                        target_tail,
                    )
                )
                workspace = workspaces[groups * VERIFY_WIDTH]
                for row in range(groups):
                    selected = row * VERIFY_WIDTH + accepted[row].item() - 1
                    slot = state_slots[row].item()
                    if active[row]:
                        self.assertEqual(
                            target.state.sparse_mla_layers[0]
                            .tail_key[0, slot, 0, 0]
                            .item(),
                            10 + selected,
                        )
                        self.assertEqual(
                            runtime.state.history.tail_key[0, slot, 0, 0].item(),
                            100 + selected,
                        )
                    if continuing[row]:
                        self.assertEqual(
                            runtime.candidate_token_ids[slot, 0].item(),
                            output_token_ids[row, accepted[row] - 1].item(),
                        )
                        self.assertEqual(
                            runtime.candidate_token_ids[slot, 1].item(),
                            workspace.endpoint.token[selected].item(),
                        )
                        self.assertEqual(
                            runtime.draft_hidden[slot, 0].item(),
                            workspace.endpoint.normalized[selected, 0].item(),
                        )
                    else:
                        self.assertEqual(
                            runtime.candidate_token_ids[slot].tolist(),
                            [-1] * VERIFY_WIDTH,
                        )


class GLM53NextNIndexShareTest(unittest.TestCase):
    def test_shared_topk_still_appends_and_remaps_each_row(self) -> None:
        tree = ast.parse(OPS_SOURCE)
        shared = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "decode_sparse_mla"
        )
        query = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == "_run_sparse_mla_query"
        )
        shared_source = ast.unparse(shared)
        query_source = ast.unparse(query)

        self.assertLess(
            shared_source.index("_write_sparse_mla_history"),
            shared_source.index("_run_sparse_mla_query"),
        )
        self.assertIn("if shared_selected is None", query_source)
        self.assertIn("w.selected.copy_(shared_selected)", query_source)
        self.assertNotIn("batch_size != 1", shared_source)
        self.assertIn("(batch_size, SPARSE_MLA_INDEX_TOP_K)", shared_source)
        self.assertGreater(
            query_source.index("_map_sparse_mla_ids"),
            query_source.index("w.selected.copy_(shared_selected)"),
        )


class GLM53NextNAcceptanceTest(unittest.TestCase):
    def test_mismatch_bonus_eos_and_budget(self) -> None:
        targets = tuple(range(11, 11 + VERIFY_WIDTH))
        matching = (100, *targets[:-1])
        self.assertEqual(
            _accept_glm53_greedy_chain(matching, targets, VERIFY_WIDTH),
            targets,
        )
        for accepted in range(1, VERIFY_WIDTH):
            candidates = list(matching)
            candidates[accepted] += 100
            with self.subTest(accepted=accepted):
                self.assertEqual(
                    _accept_glm53_greedy_chain(tuple(candidates), targets, VERIFY_WIDTH),
                    targets[:accepted],
                )
        for remaining in range(1, VERIFY_WIDTH + 1):
            with self.subTest(remaining=remaining):
                self.assertEqual(
                    _accept_glm53_greedy_chain(matching, targets, remaining),
                    targets[:remaining],
                )
        eos = min(EOS_TOKEN_IDS)
        for index in range(VERIFY_WIDTH):
            output = list(targets)
            output[index] = eos
            with self.subTest(eos_index=index):
                self.assertEqual(
                    _accept_glm53_greedy_chain(
                        (100, *output[:-1]), tuple(output), VERIFY_WIDTH
                    ),
                    tuple(output[: index + 1]),
                )


class GLM53NextNCheckpointTest(unittest.TestCase):
    def test_layer45_contract_uses_two_pinned_shards(self) -> None:
        specs = nextn._nextn_checkpoint_specs()
        shards = nextn._nextn_checkpoint_shards()
        primary, secondary = (
            shard[0] for shard in glm53_checkpoint.NEXTN_LAYER_SHARDS
        )

        self.assertEqual(len(specs), 24)
        self.assertEqual(set(shards.values()), {primary, secondary})
        self.assertEqual(shards[f"{nextn._NEXTN_PREFIX}eh_proj.weight"], primary)
        self.assertEqual(
            shards[f"{nextn._NEXTN_PREFIX}shared_head.norm.weight"], secondary
        )
        self.assertTrue(
            all(
                shards[key] == secondary
                for key in specs
                if key.startswith(nextn._NEXTN_ATTENTION_PREFIX)
            )
        )

    def test_loader_composes_layer45_and_target_head_weights(self) -> None:
        specs = nextn._nextn_checkpoint_specs()
        keyed = {key: f"tensor:{key}" for key in specs}
        with (
            patch.object(
                glm53_checkpoint, "_load_indexed_weights", return_value=keyed
            ) as load,
            patch.object(
                glm53_checkpoint,
                "pack_sparse_mla_projection_weights",
                return_value="projection",
            ),
            patch.object(
                glm53_checkpoint,
                "pack_sparse_mla_decode_weights",
                return_value="decode",
            ),
            patch.object(
                glm53_checkpoint,
                "pack_sparse_mla_output_weights",
                return_value="output",
            ),
            patch.object(
                glm53_checkpoint, "load_glm53_sparse_ffn_weights", return_value="ffn"
            ) as load_ffn,
        ):
            weights = nextn.load_glm53_nextn_weights("/checkpoint", 2, "cuda:2")

        self.assertEqual(
            (
                weights.mla_projection,
                weights.mla_decode,
                weights.mla_output,
                weights.ffn,
            ),
            ("projection", "decode", "output", "ffn"),
        )
        self.assertEqual(
            load.call_args.args[:5],
            (
                Path("/checkpoint"),
                2,
                "cuda:2",
                specs,
                nextn._nextn_checkpoint_shards(),
            ),
        )
        load_ffn.assert_called_once_with(
            Path("/checkpoint"), glm53_flash.NEXTN_LAYER_ID, 2, "cuda:2"
        )


if __name__ == "__main__":
    unittest.main()

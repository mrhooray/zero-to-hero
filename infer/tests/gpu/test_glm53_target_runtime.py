import inspect
import sys
import types
import unittest
from dataclasses import dataclass, fields, is_dataclass, replace
from itertools import pairwise
from unittest.mock import Mock, patch

import pytest

pytestmark = pytest.mark.gpu

from infer.models.glm53_flash import model as glm53
from infer.models.glm53_flash import target as target


def tensors(value):
    import torch

    if isinstance(value, torch.Tensor):
        yield value
    elif is_dataclass(value):
        for field in fields(value):
            yield from tensors(getattr(value, field.name))
    elif isinstance(value, tuple):
        for item in value:
            yield from tensors(item)


@dataclass(frozen=True, slots=True)
class FakePrefillMetadata:
    cu_seqlens_cpu: object
    cu_seqlens: object
    cu_seqlens_int64: object
    batch_size: int
    total_tokens: int

    def __init__(self, cu_seqlens, device) -> None:
        import torch

        values = tuple(cu_seqlens)
        object.__setattr__(
            self,
            "cu_seqlens_cpu",
            torch.tensor(values, dtype=torch.int64, device="cpu"),
        )
        object.__setattr__(
            self,
            "cu_seqlens",
            torch.tensor(values, dtype=torch.int32, device=device),
        )
        object.__setattr__(
            self,
            "cu_seqlens_int64",
            torch.tensor(values, dtype=torch.int64, device=device),
        )
        object.__setattr__(self, "batch_size", len(values) - 1)
        object.__setattr__(self, "total_tokens", values[-1])

    @classmethod
    def stage(cls, cu_seqlens, cpu, cuda_int32, cuda_int64):
        values = tuple(cu_seqlens)
        for index, value in enumerate(values):
            cpu[index] = value
        size = len(values)
        cuda_int32[:size].copy_(cpu[:size])
        cuda_int64[:size].copy_(cpu[:size])
        metadata = cls.__new__(cls)
        object.__setattr__(metadata, "cu_seqlens_cpu", cpu[:size])
        object.__setattr__(metadata, "cu_seqlens", cuda_int32[:size])
        object.__setattr__(metadata, "cu_seqlens_int64", cuda_int64[:size])
        object.__setattr__(metadata, "batch_size", size - 1)
        object.__setattr__(metadata, "total_tokens", values[-1])
        return metadata

    def validate_unchanged(self) -> None:
        pass


@dataclass(frozen=True, slots=True)
class FakePrefillBatch:
    metadata: object
    state_indices: object
    state_indices_int64: object
    has_initial: object
    segments: object


def validator_module(validator):
    module = types.ModuleType("infer.models.glm53_flash.ops.megafuse")
    module.validate_glm53_megafuse_plan = validator
    return module


def prefill_modules(conv_validator):
    def make_workspace(
        storage, total_tokens, batch_size, attention_tp_size=glm53.TP_SIZE
    ):
        required = glm53.kda_prefill_workspace_shapes(
            total_tokens, batch_size, attention_tp_size
        ).kda_workspace[0]
        offset = (-storage.data_ptr()) % glm53.FLASH_KDA_WORKSPACE_ALIGNMENT
        return storage[offset : offset + required]

    prefill_kda = types.ModuleType("infer.models.glm53_flash.ops.prefill_kda")
    prefill_kda.GLM53PrefillMetadata = FakePrefillMetadata
    prefill_kda.make_glm53_prefill_kda_workspace = make_workspace

    segmented_conv = types.ModuleType("infer.models.glm53_flash.ops.segmented_conv")
    segmented_conv.BLOCK_M = 8
    segmented_conv.validate_glm53_segmented_conv_plan = conv_validator

    ops = types.ModuleType("infer.models.glm53_flash.ops.core")
    ops.GLM53PrefillBatch = FakePrefillBatch
    return {
        prefill_kda.__name__: prefill_kda,
        segmented_conv.__name__: segmented_conv,
        ops.__name__: ops,
    }


class GLM53TargetRuntimeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import torch

        cls.torch = torch
        cls.validator = Mock(
            side_effect=lambda _indices, lengths, **_kwargs: sum(
                right > left for left, right in pairwise(lengths)
            )
        )
        cls.conv_validator = Mock(
            side_effect=lambda _cu, _indices, _initial, segments, **_kwargs: len(
                segments
            )
        )
        with patch.dict(
            sys.modules,
            {
                "infer.models.glm53_flash.ops.megafuse": validator_module(cls.validator),
                **prefill_modules(cls.conv_validator),
            },
        ):
            cls.runtime = target.allocate_glm53_target_runtime(
                7,
                "cpu",
                history_block_count=2,
                live_slot_count=2,
            )

    def test_dep4_prefill_capacity_tracks_speculation(self) -> None:
        class StopAllocation(Exception):
            pass

        for speculative, expected in ((True, 32), (False, 32)):
            with (
                self.subTest(speculative=speculative),
                patch.dict(
                    sys.modules,
                    {
                        "infer.models.glm53_flash.ops.megafuse": validator_module(
                            Mock(return_value=0)
                        )
                    },
                ),
                patch.object(target, "_allocate_target_state", return_value="state"),
                patch.object(
                    target,
                    "_allocate_target_prefill_workspace",
                    side_effect=StopAllocation,
                ) as allocate_prefill,
                self.assertRaises(StopAllocation),
            ):
                target.allocate_glm53_target_runtime(
                    5,
                    "cpu",
                    history_block_count=1,
                    live_slot_count=expected,
                    attention_tp_size=1,
                    speculative=speculative,
                )

            self.assertEqual(allocate_prefill.call_args.args[1:], ("cpu", 1))
            self.assertEqual(
                allocate_prefill.call_args.kwargs, {"batch_capacity": expected}
            )

    def test_prefill_workspace_view_uses_allocated_kda_capacity(self) -> None:
        for capacity in (16, 32):
            workspace = types.SimpleNamespace(
                kda=types.SimpleNamespace(
                    initial_state=types.SimpleNamespace(shape=(capacity,))
                )
            )
            with (
                self.subTest(capacity=capacity),
                patch.object(
                    target, "_workspace_view", side_effect=lambda _, shapes: shapes
                ),
            ):
                shapes = target._target_prefill_workspace_view(workspace, 1, 1)

            self.assertEqual(shapes.kda.initial_state[0], capacity)

    def test_allocator_builds_decode_parities_and_verify_views(self) -> None:
        events = []

        def validate(indices, lengths, *, state_pages):
            events.append(("validate", indices, lengths, state_pages))
            return 0

        def workspace(*args):
            bucket = args[2] if len(args) >= 3 else 1
            events.append(("workspace", bucket))
            return f"workspace-{bucket}"

        with (
            patch.dict(
                sys.modules,
                {"infer.models.glm53_flash.ops.megafuse": validator_module(validate)},
            ),
            patch.object(target, "_allocate_target_state", return_value="state"),
            patch.object(target, "_allocate_endpoint_workspace", side_effect=workspace),
            patch.object(
                target,
                "_allocate_target_prefill_workspace",
                return_value="prefill-workspace",
            ),
            patch.object(target, "_allocate_target_batch") as allocate_batch,
            patch.object(
                target,
                "_allocate_decode_staging",
                return_value="decode-staging",
            ),
            patch.object(
                target,
                "_allocate_target_transaction",
                return_value="transaction",
            ),
            patch.object(
                target,
                "_allocate_target_commit_tables",
                return_value="commit-tables",
            ),
            patch.object(
                target,
                "_target_decode_workspace",
                return_value="verify-workspace",
            ),
            patch.object(
                target,
                "_target_decode_workspace_view",
                side_effect=lambda _workspace, rows, _tp: f"verify-workspace-{rows}",
            ),
            patch.object(
                target,
                "_target_transaction_view",
                side_effect=lambda _transaction, rows: f"transaction-{rows}",
            ),
            patch.object(
                target,
                "_target_batch_view",
                return_value=allocate_batch.return_value,
            ),
            patch.object(
                target,
                "_allocate_prefill_staging",
                return_value="prefill-staging",
            ),
            patch.object(
                target,
                "_allocate_target_prefix_snapshots",
                return_value="prefix-snapshots",
            ),
        ):
            runtime = target.allocate_glm53_target_runtime(
                5,
                "cpu",
                history_block_count=1,
                live_slot_count=2,
                snapshot_slot_count=3,
            )

        buckets = glm53.GLM53_TARGET_DECODE_BATCH_SIZES
        self.assertEqual(
            events[: len(buckets)],
            [
                ("validate", (-1,) * bucket, (0,) * (bucket + 1), 2)
                for bucket in buckets
            ],
        )
        self.assertEqual(
            events[len(buckets) :],
            [("workspace", bucket) for bucket in buckets],
        )
        self.assertEqual(tuple(runtime.decode), buckets)
        self.assertTrue(
            all(
                decode.head == f"workspace-{bucket}"
                and decode.batches == (allocate_batch.return_value,) * 2
                for bucket, decode in runtime.decode.items()
            )
        )
        self.assertEqual(runtime.decode_staging, ("decode-staging",) * 2)
        self.assertEqual(runtime.target_decode, {})
        self.assertEqual(runtime.prefill_workspace, "prefill-workspace")
        self.assertEqual(runtime.prefill_staging, "prefill-staging")
        self.assertEqual(runtime.prefix_snapshots, "prefix-snapshots")
        self.assertEqual(runtime.commit_tables, "commit-tables")
        self.assertEqual(runtime.sampled_tokens.shape, (66,))
        self.assertEqual(runtime.execution_tables.shape[0], 66)
        self.assertEqual(
            allocate_batch.call_args_list[0].kwargs["request_indices"], (2,)
        )
        self.assertEqual(
            tuple(runtime.verify),
            tuple((bucket, parity) for bucket in buckets for parity in range(2)),
        )
        for (groups, _parity), verify in runtime.verify.items():
            rows = groups * glm53.GLM53_TARGET_VERIFY_WIDTH
            self.assertEqual(verify.workspace, f"verify-workspace-{rows}")
            self.assertIs(verify.batch, allocate_batch.return_value)
            self.assertEqual(verify.transaction, f"transaction-{rows}")
        self.assertEqual(allocate_batch.call_count, 2 * len(buckets) + 2)
        self.assertEqual(tuple(runtime.committed_lengths.shape), (2,))

    def test_target_only_allocator_skips_verify_transactions(self) -> None:
        with (
            patch.dict(
                sys.modules,
                {"infer.models.glm53_flash.ops.megafuse": validator_module(Mock(return_value=0))},
            ),
            patch.object(target, "_allocate_target_state", return_value="state"),
            patch.object(
                target,
                "_allocate_endpoint_workspace",
                return_value="head-workspace",
            ),
            patch.object(
                target,
                "_allocate_target_prefill_workspace",
                return_value="prefill-workspace",
            ),
            patch.object(target, "_allocate_target_batch") as allocate_batch,
            patch.object(
                target,
                "_allocate_decode_staging",
                return_value="decode-staging",
            ),
            patch.object(target, "_allocate_target_transaction") as transaction,
            patch.object(target, "_allocate_target_commit_tables") as commit_tables,
            patch.object(
                target,
                "_target_decode_workspace",
                return_value="decode-workspace",
            ) as allocate_workspace,
            patch.object(
                target,
                "_target_decode_workspace_view",
                side_effect=lambda _workspace, rows, _tp: f"decode-workspace-{rows}",
            ),
            patch.object(
                target,
                "_allocate_prefill_staging",
                return_value="prefill-staging",
            ),
            patch.object(
                target,
                "_allocate_target_prefix_snapshots",
                return_value="prefix-snapshots",
            ),
        ):
            runtime = target.allocate_glm53_target_runtime(
                5,
                "cpu",
                history_block_count=1,
                live_slot_count=2,
                speculative=False,
            )

        buckets = glm53.GLM53_TARGET_DECODE_BATCH_SIZES
        self.assertEqual(runtime.verify, {})
        self.assertEqual(runtime.commit_tables, ())
        self.assertEqual(
            runtime.target_decode,
            {bucket: f"decode-workspace-{bucket}" for bucket in buckets},
        )
        transaction.assert_not_called()
        commit_tables.assert_not_called()
        allocate_workspace.assert_called_once_with(
            "prefill-workspace",
            self.torch,
            "cpu",
            max(buckets),
            glm53.TP_SIZE,
        )
        self.assertEqual(allocate_batch.call_count, 2 * len(buckets))

    def test_pool_and_descriptor_geometry(self) -> None:
        torch = self.torch
        state = self.runtime.state
        self.assertEqual(target.GLM53_TARGET_HISTORY_BLOCKS, 8192)
        self.assertEqual(target.GLM53_TARGET_LIVE_SLOTS, 64)
        self.assertEqual(len(state.kda_layers), 34)
        self.assertEqual(len(state.sparse_mla_layers), 11)
        self.assertEqual(
            tuple(state.kda_layers[0].recurrent.shape),
            glm53.kda_state_shapes(2).recurrent,
        )
        history = state.sparse_mla_layers[0]
        self.assertEqual(tuple(history.latent.shape), (6, 1, 64, 512))
        self.assertEqual(tuple(history.index_cache.shape), (3, 4_224))
        self.assertEqual(tuple(history.tail_key.shape), (2, 2, 4, 128))

        b1 = self.runtime.decode[1].batches[0]
        self.assertTrue(torch.equal(b1.token_ids, torch.tensor((7,))))
        self.assertTrue(
            torch.equal(b1.state_indices, torch.tensor((-1,), dtype=torch.int32))
        )
        self.assertTrue(
            torch.equal(b1.cu_seqlens, torch.tensor((0, 0), dtype=torch.int32))
        )
        self.assertEqual(b1.sparse_mla.active.tolist(), [0])
        self.assertEqual(b1.state_indices_int64.tolist(), [2])
        self.assertEqual(tuple(self.runtime.sampled_tokens.shape), (66,))
        self.assertEqual(
            tuple(self.runtime.execution_tables.shape),
            (66, glm53.SPARSE_MLA_COMPOUND_BLOCKS),
        )
        self.assertEqual(self.runtime.prefix_snapshots.committed_lengths.shape, (0,))
        self.assertEqual(
            [table.source_addresses.shape[0] for table in self.runtime.commit_tables],
            [34, 34, 44],
        )
        self.assertEqual(
            [table.row_words for table in self.runtime.commit_tables],
            [262_144, 9_216, 256],
        )
        self.assertEqual(
            self.runtime.commit_tables[2].source_strides.unique().tolist(), [512]
        )
        self.assertEqual(
            self.runtime.commit_tables[2].destination_strides.unique().tolist(), [256]
        )

        b2 = self.runtime.decode[2].batches[0]
        self.assertTrue(torch.equal(b2.token_ids, torch.tensor((7, 7))))
        self.assertEqual(b2.state_indices.tolist(), [-1, -1])
        self.assertEqual(b2.cu_seqlens.tolist(), [0, 0, 0])
        self.assertEqual(b2.state_indices_int64.tolist(), [2, 3])

    def test_prefix_snapshot_round_trip_copies_fixed_state_and_private_tail(
        self,
    ) -> None:
        torch = self.torch
        runtime = replace(
            self.runtime,
            prefix_snapshots=target._allocate_target_prefix_snapshots(torch, "cpu", 1),
        )
        source_slot, destination_slot, snapshot_slot = 0, 1, 2
        source_block, destination_block = 0, 1

        for index, state in enumerate(runtime.state.kda_layers, 1):
            state.recurrent[source_slot].fill_(index)
            state.conv[source_slot].fill_(index + 40)
        for index, history in enumerate(runtime.state.sparse_mla_layers, 1):
            source_page = 2 * (source_block + 1)
            history.latent[source_page : source_page + 2].fill_(index)
            history.index_cache[source_block + 1].fill_(index + 80)
            history.tail_key[:, source_slot].fill_(index + 100)
            history.tail_gate[:, source_slot].fill_(index + 120)
        runtime.committed_lengths[source_slot] = 256

        runtime.capture_prefix(source_slot, snapshot_slot, source_block)

        for state in runtime.state.kda_layers:
            state.recurrent[source_slot].zero_()
            state.conv[source_slot].zero_()
        for history in runtime.state.sparse_mla_layers:
            source_page = 2 * (source_block + 1)
            history.latent[source_page : source_page + 2].zero_()
            history.index_cache[source_block + 1].zero_()
            history.tail_key[:, source_slot].zero_()
            history.tail_gate[:, source_slot].zero_()
        runtime.committed_lengths[source_slot] = 0

        runtime.restore_prefix(snapshot_slot, destination_slot, destination_block)

        for index, state in enumerate(runtime.state.kda_layers, 1):
            self.assertTrue(torch.all(state.recurrent[destination_slot] == index))
            self.assertTrue(torch.all(state.conv[destination_slot] == index + 40))
        for index, history in enumerate(runtime.state.sparse_mla_layers, 1):
            destination_page = 2 * (destination_block + 1)
            self.assertTrue(
                torch.all(
                    history.latent[destination_page : destination_page + 2] == index
                )
            )
            self.assertTrue(
                torch.all(history.index_cache[destination_block + 1] == index + 80)
            )
            self.assertTrue(
                torch.all(history.tail_key[:, destination_slot] == index + 100)
            )
            self.assertTrue(
                torch.all(history.tail_gate[:, destination_slot] == index + 120)
            )
        self.assertEqual(runtime.committed_lengths[destination_slot], 256)

        runtime.state.kda_layers[0].recurrent[destination_slot].zero_()
        runtime.restore_prefix(snapshot_slot, destination_slot, destination_block)
        self.assertTrue(
            torch.all(runtime.state.kda_layers[0].recurrent[destination_slot] == 1)
        )

        for state in runtime.state.kda_layers:
            state.recurrent[destination_slot].zero_()
            state.conv[destination_slot].zero_()
        for history in runtime.state.sparse_mla_layers:
            destination_page = 2 * (destination_block + 1)
            history.latent[destination_page : destination_page + 2].zero_()
            history.index_cache[destination_block + 1].zero_()
            history.tail_key[:, destination_slot].zero_()
            history.tail_gate[:, destination_slot].zero_()
        runtime.committed_lengths[destination_slot] = 0

    def test_bucket_head_workspaces_are_separate_exact_shapes(self) -> None:
        for batch_size, decode in self.runtime.decode.items():
            with self.subTest(batch_size=batch_size):
                self.assertEqual(
                    tuple(decode.head.token.shape),
                    (batch_size,),
                )
                self.assertEqual(
                    tuple(decode.head.streams.shape),
                    (batch_size, glm53.MHC_STREAMS, glm53.HIDDEN_SIZE),
                )
        pointers = [
            {tensor.data_ptr() for tensor in tensors(decode.head)}
            for decode in self.runtime.decode.values()
        ]
        self.assertTrue(
            all(left.isdisjoint(right) for left, right in pairwise(pointers))
        )

    def test_workspace_view_reinterprets_one_contiguous_backing_prefix(self) -> None:
        torch = self.torch
        sources = tuple(torch.arange(24).view(4, 6) + offset for offset in range(5))
        workspace = glm53.KDADecodeWorkspace(*sources)
        shapes = glm53.KDADecodeWorkspace(*((3, 4),) * 5)

        viewed = target._workspace_view(workspace, shapes)

        for source, view in zip(sources, tensors(viewed), strict=True):
            self.assertEqual(view.shape, (3, 4))
            self.assertTrue(view.is_contiguous())
            self.assertEqual(view.data_ptr(), source.data_ptr())
        viewed.output[0, 0] = -1
        self.assertEqual(workspace.output.view(-1)[0], -1)

    def test_two_decode_staging_parities_hold_only_compact_io(self) -> None:
        torch = self.torch
        for staging in self.runtime.decode_staging:
            self.assertEqual(tuple(staging.control_cpu.shape), (64, 5))
            self.assertEqual(tuple(staging.control.shape), (64, 5))
            self.assertEqual(tuple(staging.table_delta_cpu.shape), (64, 2))
            self.assertEqual(staging.control_cpu.dtype, torch.int32)
            self.assertEqual(staging.control.dtype, torch.int32)
            self.assertEqual(staging.table_delta_cpu.dtype, torch.int32)
        self.assertEqual(
            inspect.getsource(target._allocate_decode_staging).count("pin_memory=True"),
            1,
        )

    def test_capture_control_starts_all_inactive_on_request_dummy_rows(self) -> None:
        staging = target._allocate_decode_staging(self.torch, "cpu", 5, 65)

        self.assertEqual(staging.control[:, 0].tolist(), list(range(5, 69)))
        self.assertEqual(staging.control[:, 1].tolist(), [-1] * 64)
        self.assertFalse(staging.control[:, 2:].any())
        self.assertEqual(tuple(staging.table_delta_cpu.shape), (64, 65))

    def test_successor_delta_uses_another_parity_before_collect(self) -> None:
        self.runtime.apply_table_delta(0, 0, (0,), 0, 0)
        self.runtime.apply_table_delta(0, 1, (1,), 1, 0)

        self.assertNotEqual(
            self.runtime.decode_staging[0].table_delta_cpu.data_ptr(),
            self.runtime.decode_staging[1].table_delta_cpu.data_ptr(),
        )
        self.assertEqual(self.runtime.execution_tables[0, :2].tolist(), [1, 2])

    def test_one_restored_sequence_can_stage_more_than_one_chunk_of_history(
        self,
    ) -> None:
        for block_count in (65, 257, 545):
            with self.subTest(block_count=block_count):
                staging = target._allocate_decode_staging(
                    self.torch, "cpu", 2, block_count
                )
                runtime = replace(self.runtime, decode_staging=(staging, staging))
                previous = self.runtime.execution_tables[0, :block_count].clone()
                try:
                    runtime.apply_table_delta(0, 0, tuple(range(block_count)), 0, 0)
                    self.assertEqual(
                        runtime.execution_tables[0, :block_count].tolist(),
                        list(range(1, block_count + 1)),
                    )
                finally:
                    self.runtime.execution_tables[0, :block_count].copy_(previous)

    def test_stage_compact_deltas_and_device_resident_decode_descriptors(self) -> None:
        torch = self.torch
        validator = Mock(return_value=3)
        self.runtime.committed_lengths.copy_(torch.tensor((17, 129)))
        with (
            patch.dict(
                sys.modules,
                {"infer.models.glm53_flash.ops.megafuse": validator_module(validator)},
            ),
            patch.object(torch, "empty", side_effect=AssertionError("allocation")),
            patch.object(torch, "zeros", side_effect=AssertionError("allocation")),
            patch.object(torch, "tensor", side_effect=AssertionError("allocation")),
            patch.object(
                torch.Tensor, "new_tensor", side_effect=AssertionError("allocation")
            ),
        ):
            self.runtime.apply_table_delta(1, 0, (0,), 0, 0)
            self.runtime.apply_table_delta(0, 0, (1,), 0, 1)
            batch = self.runtime.stage_decode((1, 0, 1), 0)
            self.runtime.prepare_decode(batch, self.runtime.decode_staging[0])

        self.assertIs(batch, self.runtime.decode[4].batches[0])
        self.assertEqual(batch.token_ids.tolist(), [7, 7, 7, 7])
        self.assertEqual(batch.state_indices.tolist(), [1, 0, 1, -1])
        self.assertEqual(batch.state_indices_int64.tolist(), [1, 0, 1, 2])
        self.assertEqual(batch.cu_seqlens.tolist(), [0, 1, 2, 3, 3])
        self.assertEqual(batch.sparse_mla.active.tolist(), [1, 1, 1, 0])
        self.assertEqual(batch.sparse_mla.state_slots.tolist(), [1, 0, 1, 0])
        self.assertEqual(batch.sparse_mla.raw_lengths.tolist(), [130, 18, 130, 0])
        self.assertEqual(batch.sparse_mla.block_table[0, :2].tolist(), [1, 0])
        self.assertEqual(batch.sparse_mla.block_table[1, :2].tolist(), [2, 0])
        self.assertEqual(
            self.runtime.decode_staging[0].table_delta_cpu[:2, 0].tolist(),
            [1, 2],
        )
        validator.assert_called_once_with((1, 0, 1, -1), (0, 1, 2, 3, 3), state_pages=2)

        with patch.dict(
            sys.modules,
            {"infer.models.glm53_flash.ops.megafuse": validator_module(validator)},
        ):
            batch = self.runtime.stage_decode((0,), 1)
        self.assertIs(batch, self.runtime.decode[1].batches[1])

    def test_snapshot_slots_do_not_shift_padded_decode_request_ids(self) -> None:
        torch = self.torch
        runtime = replace(
            self.runtime,
            sampled_tokens=torch.full((66,), 7, dtype=torch.int64),
            execution_tables=torch.zeros(
                (66, glm53.SPARSE_MLA_COMPOUND_BLOCKS), dtype=torch.int32
            ),
        )
        with patch.dict(
            sys.modules,
            {"infer.models.glm53_flash.ops.megafuse": validator_module(Mock(return_value=3))},
        ):
            batch = runtime.stage_decode((1, 0, 1), 0)
            runtime.prepare_decode(batch, runtime.decode_staging[0])

        self.assertEqual(batch.state_indices_int64.tolist(), [1, 0, 1, 2])

    def test_target_decode_publish_updates_only_active_sequence_lengths(self) -> None:
        torch = self.torch
        sampled = self.runtime.sampled_tokens.clone()
        lengths = self.runtime.committed_lengths.clone()
        try:
            self.runtime.committed_lengths.copy_(torch.tensor((7, 11)))
            with patch.dict(
                sys.modules,
                {"infer.models.glm53_flash.ops.megafuse": validator_module(Mock(return_value=1))},
            ):
                batch = self.runtime.stage_decode((1,), 0, bucket=2)
                self.runtime.prepare_decode(batch, self.runtime.decode_staging[0])
            token = torch.tensor((29, 99), dtype=torch.int64)
            accepted = batch.sparse_mla.active.to(torch.int32)

            self.runtime.publish_decoded(batch, token, accepted)

            self.assertEqual(self.runtime.sampled_tokens[1], 29)
            self.assertEqual(self.runtime.sampled_tokens[2], 99)
            self.assertEqual(self.runtime.committed_lengths.tolist(), [7, 12])
        finally:
            self.runtime.sampled_tokens.copy_(sampled)
            self.runtime.committed_lengths.copy_(lengths)

    def test_prefill_packs_fresh_and_resumed_rows_into_one_arena(self) -> None:
        validate = Mock(
            side_effect=lambda start, total, slot, blocks, **kwargs: (
                types.SimpleNamespace(
                    start_token=start,
                    total_tokens=total,
                    state_slot=slot,
                    logical_block_table=blocks,
                    **kwargs,
                )
            )
        )
        stage_batch = Mock(return_value="packed-query")
        module = types.ModuleType("infer.models.glm53_flash.ops.sparse_mla_prefill")
        module.validate_glm53_sparse_mla_prefill_plan = validate
        module.stage_glm53_sparse_mla_prefill_batch = stage_batch

        with patch.dict(
            sys.modules,
            {module.__name__: module, **prefill_modules(self.conv_validator)},
        ):
            batch = self.runtime.stage_prefill(
                ((10, 11), (12,)),
                (0, 64),
                (1, 0),
                (0, 1),
            )

        self.assertEqual(batch.token_ids.tolist(), [10, 11, 12])
        self.assertEqual(batch.sparse_mla, "packed-query")
        self.assertEqual(batch.kda.metadata.cu_seqlens.tolist(), [0, 2, 3])
        self.assertEqual(batch.kda.state_indices.tolist(), [1, 0])
        self.assertEqual(batch.kda.state_indices_int64.tolist(), [1, 0])
        self.assertEqual(batch.kda.has_initial.tolist(), [False, True])
        self.assertEqual(batch.sample_indices.tolist(), [1, 2])
        self.assertEqual(batch.sample_state_indices.tolist(), [1, 0])
        self.assertEqual(batch.sample_count, 2)
        self.assertEqual(self.runtime.committed_lengths[:2].tolist(), [65, 2])
        self.assertEqual(validate.call_args_list[0].args, (0, 2, 1, None))
        self.assertEqual(validate.call_args_list[0].kwargs["history_blocks"], 3)
        self.assertEqual(validate.call_args_list[0].kwargs["live_slots"], 2)
        self.assertFalse(validate.call_args_list[0].kwargs["has_initial"])
        self.assertTrue(validate.call_args_list[1].kwargs["has_initial"])
        self.assertEqual(stage_batch.call_args.kwargs["active"].shape, (3,))
        self.assertEqual(stage_batch.call_args.kwargs["block_table"].shape, (3, 8192))

    def test_reset_clears_only_the_selected_state_slot(self) -> None:
        torch = self.torch
        state = self.runtime.state
        try:
            self.runtime.committed_lengths[:2].copy_(
                torch.tensor((1, 2), dtype=torch.int32)
            )
            for layer in state.kda_layers:
                layer.recurrent[0].fill_(1)
                layer.recurrent[1].fill_(2)
                layer.conv[0].fill_(1)
                layer.conv[1].fill_(2)
            for history in state.sparse_mla_layers:
                history.latent.fill_(7)
                history.index_cache.fill_(7)
                history.tail_key[:, 0].fill_(1)
                history.tail_key[:, 1].fill_(2)
                history.tail_gate[:, 0].fill_(1)
                history.tail_gate[:, 1].fill_(2)

            self.runtime.reset_state(1)

            for layer in state.kda_layers:
                self.assertTrue(torch.all(layer.recurrent[0] == 1))
                self.assertEqual(torch.count_nonzero(layer.recurrent[1]), 0)
                self.assertTrue(torch.all(layer.conv[0] == 1))
                self.assertEqual(torch.count_nonzero(layer.conv[1]), 0)
            for history in state.sparse_mla_layers:
                self.assertTrue(torch.all(history.latent == 7))
                self.assertTrue(torch.all(history.index_cache == 7))
                self.assertTrue(torch.all(history.tail_key[:, 0] == 1))
                self.assertEqual(torch.count_nonzero(history.tail_key[:, 1]), 0)
                self.assertTrue(torch.all(history.tail_gate[:, 0] == 1))
                self.assertEqual(torch.count_nonzero(history.tail_gate[:, 1]), 0)
            self.assertEqual(self.runtime.committed_lengths[:2].tolist(), [1, 0])
        finally:
            for tensor in tensors(state):
                tensor.zero_()
            self.runtime.committed_lengths.zero_()

    def test_verify_stages_request_major_bucket_views(self) -> None:
        torch = self.torch
        width = glm53.GLM53_TARGET_VERIFY_WIDTH
        for groups in (1, 2, 16, 32):
            with self.subTest(groups=groups):
                tokens = torch.arange(groups * width, dtype=torch.int64).view(
                    groups, width
                )
                raw_lengths = torch.empty(groups, dtype=torch.int32)
                state_slots = torch.zeros(groups, dtype=torch.int32)
                active = torch.ones(groups, dtype=torch.uint8)
                self.runtime.committed_lengths[0] = 2
                verify = self.runtime.stage_verify(
                    tokens, state_slots, active, raw_lengths, 1
                )
                batch = verify.batch
                self.assertIs(verify, self.runtime.verify[groups, 1])
                self.assertEqual(batch.token_ids.tolist(), tokens.view(-1).tolist())
                self.assertEqual(batch.state_indices.tolist(), [0] * groups)
                self.assertEqual(
                    batch.sparse_mla.active.tolist(), [1] * (groups * width)
                )
                self.assertEqual(
                    batch.sparse_mla.state_slots.tolist(), [0] * (groups * width)
                )
                self.assertEqual(
                    batch.sparse_mla.raw_lengths.view(groups, width).tolist(),
                    [
                        list(range(raw_length + 1, raw_length + width + 1))
                        for raw_length in raw_lengths.tolist()
                    ],
                )
                self.assertEqual(
                    batch.sparse_mla.block_table[:, 0].tolist(),
                    [1] * (groups * width),
                )
                self.assertEqual(
                    verify.transaction.kda_layers[0].recurrent.shape[0],
                    groups * width,
                )
                self.assertEqual(
                    verify.workspace.endpoint.token.shape[0], groups * width
                )
                other = self.runtime.verify[groups, 0]
                self.assertNotEqual(
                    verify.batch.token_ids.data_ptr(), other.batch.token_ids.data_ptr()
                )
                self.assertEqual(
                    verify.workspace.endpoint.token.data_ptr(),
                    other.workspace.endpoint.token.data_ptr(),
                )
                self.assertEqual(
                    verify.transaction.kda_layers[0].recurrent.data_ptr(),
                    other.transaction.kda_layers[0].recurrent.data_ptr(),
                )

        self.assertEqual(
            self.runtime.verify[1, 0].workspace.endpoint.token.data_ptr(),
            self.runtime.prefill_workspace.endpoint.token.data_ptr(),
        )
        self.assertEqual(
            self.runtime.verify[1, 0].transaction.kda_layers[0].recurrent.data_ptr(),
            self.runtime.verify[32, 1].transaction.kda_layers[0].recurrent.data_ptr(),
        )
        self.assertEqual(
            self.runtime.verify[1, 0].batch.token_ids.data_ptr(),
            self.runtime.verify[32, 0].batch.token_ids.data_ptr(),
        )

    def test_publish_verified_commits_each_device_selected_row(self) -> None:
        torch = self.torch
        width = glm53.GLM53_TARGET_VERIFY_WIDTH
        self.runtime.committed_lengths[:2].copy_(
            torch.tensor((2, 7), dtype=torch.int32)
        )
        active = torch.tensor((1, 0), dtype=torch.uint8)
        verify = self.runtime.stage_verify(
            torch.arange(2 * width, dtype=torch.int64).view(2, width),
            torch.tensor((0, -1), dtype=torch.int32),
            active,
            torch.empty(2, dtype=torch.int32),
            0,
        )
        accepted = torch.tensor((3, 0), dtype=torch.int32)
        output = torch.arange(2 * width, dtype=torch.int64).view(2, width)
        copy_table = Mock()
        publish = Mock()
        speculate = types.ModuleType("infer.models.glm53_flash.ops.speculate")
        speculate.glm53_copy_verified_row_table = copy_table
        speculate.glm53_publish_accepted = publish
        with patch.dict(sys.modules, {speculate.__name__: speculate}):
            self.runtime.publish_verified(verify, accepted, active, output)

        self.assertEqual(copy_table.call_count, 3)
        for call, table in zip(
            copy_table.call_args_list, self.runtime.commit_tables, strict=True
        ):
            self.assertEqual(
                call.args,
                (
                    table.source_addresses,
                    table.destination_addresses,
                    table.source_strides,
                    table.destination_strides,
                    accepted,
                    verify.batch.state_indices,
                    active,
                ),
            )
            self.assertEqual(call.kwargs, {"row_words": table.row_words})
        publish.assert_called_once_with(
            output,
            accepted,
            verify.batch.state_indices,
            active,
            self.runtime.sampled_tokens,
            self.runtime.committed_lengths,
        )
        self.assertEqual(verify.batch.state_indices.tolist(), [0, -1])
        self.assertEqual(
            verify.batch.sparse_mla.state_slots.tolist(), [0] * (2 * width)
        )

    def test_prefill_owns_one_max_capacity_descriptor_arena(self) -> None:
        staging = self.runtime.prefill_staging
        self.assertEqual(tuple(staging.token_ids.shape), (glm53.KDA_CHUNK_SIZE,))
        self.assertEqual(tuple(staging.cu_seqlens.shape), (65,))
        self.assertEqual(tuple(staging.state_indices.shape), (64,))
        self.assertEqual(tuple(staging.active.shape), (glm53.KDA_CHUNK_SIZE,))
        self.assertEqual(
            tuple(staging.block_table.shape),
            (glm53.KDA_CHUNK_SIZE, glm53.SPARSE_MLA_COMPOUND_BLOCKS),
        )
        self.assertEqual(tuple(staging.sample_indices.shape), (64,))

    def test_decode_hot_path_methods_do_not_allocate_or_synchronize(self) -> None:
        for method in (
            target.GLM53TargetRuntime.stage_decode,
            target.GLM53TargetRuntime.capture_prefix,
            target.GLM53TargetRuntime.restore_prefix,
            target.GLM53TargetRuntime.reset_state,
            target.GLM53TargetRuntime.publish_prefill,
            target.GLM53TargetRuntime.stage_verify,
            target.GLM53TargetRuntime.publish_verified,
        ):
            source = inspect.getsource(method)
            for forbidden in (
                "torch.",
                ".new_",
                ".clone(",
                ".contiguous(",
                ".cpu(",
                ".item(",
                ".tolist(",
                ".to(",
                "empty(",
                "synchronize(",
                "tensor(",
                "zeros(",
            ):
                self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()

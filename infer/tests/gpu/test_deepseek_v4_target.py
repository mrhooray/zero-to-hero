import importlib
import math
import sys
import types
import unittest
from dataclasses import fields, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.gpu

import torch

from infer.models.deepseek_v4_flash import model as deepseek_v4
from infer.models.deepseek_v4_flash import attention
from infer.models.deepseek_v4_flash import checkpoint
from infer.models.deepseek_v4_flash import megamoe
from infer.models.deepseek_v4_flash import target

OPS_SOURCE = (
    Path(__file__).resolve().parents[2] / "src/infer/models/deepseek_v4_flash/ops/target.py"
).read_text()


def load_target_ops(class_name="DeepSeekV4TargetOps"):
    flashinfer = types.ModuleType("flashinfer")
    flashinfer.__path__ = []
    norm = types.ModuleType("flashinfer.norm")
    norm.rmsnorm = lambda *_args, **_kwargs: None
    core = types.ModuleType("infer.models.deepseek_v4_flash.ops.core")
    core._require_tensor = lambda *_args: None
    attention_ops = types.ModuleType("infer.models.deepseek_v4_flash.ops.attention")
    attention_ops.DeepSeekV4AttentionOps = object
    attention_ops.DeepSeekV4TEP4AttentionOps = object
    name = "infer.models.deepseek_v4_flash.ops.target"
    with patch.dict(
        sys.modules,
        {
            "flashinfer": flashinfer,
            "flashinfer.norm": norm,
            "infer.models.deepseek_v4_flash.ops.core": core,
            "infer.models.deepseek_v4_flash.ops.attention": attention_ops,
        },
    ):
        sys.modules.pop(name, None)
        module = importlib.import_module(name)
        sys.modules.pop(name, None)
    return getattr(module, class_name)


class FakeDeepGemm:
    def transform_weights_for_mega_moe(self, l1, l2):
        return l1, l2


class FakeCheckpointView:
    def __init__(self) -> None:
        self.calls = []
        self.loaded_shapes = {}

    def load_target_tensor(self, name, rank, device, *, sharded):
        self.calls.append((name, rank, device, sharded))
        spec = checkpoint._target_spec(name)
        dtype = {
            "BF16": torch.bfloat16,
            "F32": torch.float32,
            "F8_E4M3": torch.float8_e4m3fn,
            "F8_E8M0": torch.float8_e8m0fnu,
            "I8": torch.int8,
            "I64": torch.int64,
        }[spec.dtype]
        shape = spec.local_shape() if sharded else spec.shape
        self.loaded_shapes[name] = shape
        return torch.empty(shape, dtype=dtype, device=device)


class FakeTargetOps:
    def __init__(self) -> None:
        self.layer = self
        self.calls = []
        self.decode_kwargs = []

    def decode_embedding(self, token_ids, weights, workspace):
        self.calls.append(("embedding", token_ids, weights, workspace))
        return ("streams", -1)

    def prefill_embedding(
        self, token_ids, weights, collapsed, streams, local_tokens, select
    ):
        self.calls.append(
            (
                "prefill_embedding",
                token_ids,
                weights,
                collapsed,
                streams,
                local_tokens,
                select,
            )
        )
        return streams

    def decode_layer(self, streams, token_ids, weights, *inputs, **kwargs):
        self.calls.append(("layer", weights.layer_id, streams, token_ids, *inputs))
        self.decode_kwargs.append(kwargs)
        return ("streams", weights.layer_id)

    def prefill_layer(self, streams, token_ids, weights, *inputs):
        self.calls.append(
            (
                "prefill_layer",
                weights.layer_id,
                deepseek_v4.attention_kind(weights.layer_id),
                streams,
                token_ids,
                *inputs,
            )
        )
        return inputs[7].streams_out

    def decode_head(self, streams, weights, workspace):
        self.calls.append(("head", streams, weights, workspace))
        return "logits"

    def select_tokens(self, logits, workspace, output):
        self.calls.append(("select", logits, workspace, output))

    def capture_hidden(self, streams, output):
        self.calls.append(("capture", streams, output))


def fake_runtime_weights():
    layers = []
    for layer_id in range(deepseek_v4.NUM_LAYERS):
        projection = SimpleNamespace(sink=torch.empty(64, device="meta"))
        kind = deepseek_v4.attention_kind(layer_id)
        if kind == "swa":
            layer_attention = projection
        elif kind == "csa":
            layer_attention = SimpleNamespace(
                projection=projection,
                main_ape=torch.empty((4, 1024), device="meta"),
                main_norm=torch.empty(512, device="meta"),
                index_ape=torch.empty((4, 256), device="meta"),
                index_norm=torch.empty(128, device="meta"),
            )
        else:
            layer_attention = SimpleNamespace(
                projection=projection,
                main_ape=torch.empty((128, 512), device="meta"),
                main_norm=torch.empty(512, device="meta"),
            )
        layers.append(SimpleNamespace(layer_id=layer_id, attention=layer_attention))
    return SimpleNamespace(layers=tuple(layers))


def make_runtimes(
    live_slots=6,
    history_blocks=8,
    *,
    snapshot_slots=0,
    speculative=True,
    tensor_parallel=False,
):
    metadata = object()
    mega_buffer = object()
    deep_gemm = SimpleNamespace(get_num_sms=lambda: 148)
    rope_tables = (
        torch.empty((deepseek_v4.MAX_CONTEXT_TOKENS, 64), device="meta"),
        torch.empty((deepseek_v4.MAX_CONTEXT_TOKENS, 64), device="meta"),
    )
    with (
        patch.dict(sys.modules, {"deep_gemm": deep_gemm}),
        patch.object(target, "_rope_table", side_effect=rope_tables),
    ):
        runtimes = target.allocate_target_runtime(
            fake_runtime_weights(),
            live_slots,
            history_blocks,
            "meta",
            snapshot_slot_count=snapshot_slots,
            speculative=speculative,
            tensor_parallel=tensor_parallel,
            metadata_allocator=lambda: metadata,
            mega_buffer_allocator=lambda: mega_buffer,
        )
    return runtimes, metadata, mega_buffer


def make_prefill_runtime(live_slots=6, history_blocks=16, state=None):
    staging = tuple(target._allocate_prefill_staging(torch, "cpu") for _ in range(2))
    resources = SimpleNamespace(
        head=SimpleNamespace(logits=torch.zeros(4, 3)),
        head_tokens=torch.empty(4, dtype=torch.int64),
    )
    return target.DeepSeekV4TargetRuntime(
        state=object() if state is None else state,
        sampled_tokens=torch.zeros(live_slots, dtype=torch.int64),
        committed_lengths=torch.zeros(live_slots, dtype=torch.int32),
        result_records=torch.zeros(
            (live_slots, 1 + deepseek_v4.DSPARK_VERIFY_WIDTH), dtype=torch.int64
        ),
        execution_tables=torch.full((live_slots, 8192), -1, dtype=torch.int32),
        verify={},
        decode={},
        bindings=(),
        resources=resources,
        staging=staging,
        cos_sin=(object(), object()),
        live_slots=live_slots,
        history_blocks=history_blocks,
        history=(object(), (), object()),
    )


class DeepSeekV4TargetWeightsTest(unittest.TestCase):
    def test_reset_initializes_only_the_selected_live_state(self) -> None:
        state = deepseek_v4.DeepSeekV4TargetState(
            torch.full((2, 3, 2, 4), 7, dtype=torch.uint8),
            torch.full((1, 3, 2, 8), 7.0),
            torch.full((1, 3, 2, 4), 7.0),
            torch.full((1, 3, 2, 6), 7.0),
        )
        runtime = make_prefill_runtime(live_slots=3, state=state)

        runtime.reset_state(1)

        self.assertFalse(state.raw_window[:, 1].any())
        self.assertTrue((state.raw_window[:, 0] == 7).all())
        for values in (state.c4_main, state.c4_index, state.c128_main):
            midpoint = values.shape[-1] // 2
            self.assertFalse(values[:, 1, :, :midpoint].any())
            self.assertTrue(torch.isneginf(values[:, 1, :, midpoint:]).all())
            self.assertTrue((values[:, 0] == 7).all())
        with self.assertRaisesRegex(ValueError, "live state pool"):
            runtime.reset_state(3)

    def test_prefix_snapshot_round_trip_copies_fixed_state_and_private_tail(
        self,
    ) -> None:
        state = deepseek_v4.DeepSeekV4TargetState(
            torch.zeros((2, 2, 2, 4), dtype=torch.uint8),
            torch.zeros((1, 2, 2, 8)),
            torch.zeros((1, 2, 2, 4)),
            torch.zeros((1, 2, 2, 6)),
        )
        snapshot_state = deepseek_v4.DeepSeekV4TargetState(
            torch.zeros((2, 1, 2, 4), dtype=torch.uint8),
            torch.zeros((1, 1, 2, 8)),
            torch.zeros((1, 1, 2, 4)),
            torch.zeros((1, 1, 2, 6)),
        )
        history = (
            torch.zeros((2, 3, 5), dtype=torch.uint8),
            tuple(torch.zeros((3, 3), dtype=torch.uint8) for _ in range(2)),
            torch.zeros((1, 3, 7), dtype=torch.uint8),
        )
        snapshots = target.DeepSeekV4TargetPrefixSnapshots(
            snapshot_state,
            torch.zeros((2, 1, 5), dtype=torch.uint8),
            tuple(torch.zeros((1, 3), dtype=torch.uint8) for _ in range(2)),
            torch.zeros((1, 1, 7), dtype=torch.uint8),
            torch.zeros(1, dtype=torch.int32),
        )
        runtime = replace(
            make_prefill_runtime(live_slots=2, history_blocks=3, state=state),
            history=history,
            prefix_snapshots=snapshots,
        )
        source_slot, destination_slot, snapshot_slot = 0, 1, 2
        source_block, destination_block = 0, 2
        for index, state_field in enumerate(fields(state), 1):
            getattr(state, state_field.name)[:, source_slot].fill_(index)
        history[0][:, source_block].fill_(11)
        for index, cache in enumerate(history[1], 12):
            cache[source_block].fill_(index)
        history[2][:, source_block].fill_(14)
        runtime.committed_lengths[source_slot] = 256

        runtime.capture_prefix(source_slot, snapshot_slot, source_block)
        for state_field in fields(state):
            getattr(state, state_field.name)[:, source_slot].zero_()
        history[0][:, source_block].zero_()
        for cache in history[1]:
            cache[source_block].zero_()
        history[2][:, source_block].zero_()

        runtime.restore_prefix(snapshot_slot, destination_slot, destination_block)

        for index, state_field in enumerate(fields(state), 1):
            self.assertTrue(
                torch.all(
                    getattr(state, state_field.name)[:, destination_slot] == index
                )
            )
        self.assertTrue(torch.all(history[0][:, destination_block] == 11))
        for index, cache in enumerate(history[1], 12):
            self.assertTrue(torch.all(cache[destination_block] == index))
        self.assertTrue(torch.all(history[2][:, destination_block] == 14))
        self.assertEqual(runtime.committed_lengths[destination_slot], 256)

    def test_prefix_snapshot_validates_live_snapshot_and_history_slots(self) -> None:
        runtime = make_prefill_runtime(live_slots=2, history_blocks=3)
        with self.assertRaisesRegex(ValueError, "snapshot pool"):
            runtime.capture_prefix(0, 2, 0)

        snapshots = target.DeepSeekV4TargetPrefixSnapshots(
            deepseek_v4.DeepSeekV4TargetState(
                torch.empty((0, 1)),
                torch.empty((0, 1)),
                torch.empty((0, 1)),
                torch.empty((0, 1)),
            ),
            torch.empty((0, 1)),
            (),
            torch.empty((0, 1)),
            torch.empty(1, dtype=torch.int32),
        )
        runtime = replace(runtime, prefix_snapshots=snapshots)
        with self.assertRaisesRegex(ValueError, "live state pool"):
            runtime.capture_prefix(2, 2, 0)
        with self.assertRaisesRegex(ValueError, "snapshot pool"):
            runtime.capture_prefix(0, 1, 0)
        with self.assertRaisesRegex(ValueError, "history pool"):
            runtime.capture_prefix(0, 2, 3)

    def test_prefill_state_uses_paged_chunk_scratch_and_retains_each_request(
        self,
    ) -> None:
        def stage(ratio, starts, lengths, slots):
            tokens = sum(lengths)
            page = target._PREFILL_STATE_PAGE[ratio]
            retained = target._PREFILL_RETAINED_PAGES[ratio]
            token_slots = torch.empty(tokens, dtype=torch.int64)
            table = torch.full((len(starts), tokens + retained), -1, dtype=torch.int64)
            base = torch.empty(len(starts), dtype=torch.int64)
            mappings = [
                torch.empty(len(starts) * retained, dtype=torch.int64) for _ in range(4)
            ]
            counts = target._stage_prefill_state(
                starts, lengths, slots, ratio, token_slots, table, base, *mappings
            )
            return token_slots, table, base, mappings, counts, page

        token_slots, table, base, mappings, counts, _ = stage(4, (0, 9), (9, 8), (2, 5))
        self.assertEqual(counts, (7, 4, 2, 4))
        self.assertEqual(base.tolist(), [0, 1])
        self.assertEqual(table[0, :3].tolist(), [0, 1, 2])
        self.assertEqual(table[1, :4].tolist(), [3, 4, 5, 6])
        self.assertEqual(token_slots.tolist(), list(range(9)) + list(range(17, 25)))
        self.assertEqual(mappings[0][:2].tolist(), [21, 22])
        self.assertEqual(mappings[1][:2].tolist(), [3, 4])
        self.assertEqual(mappings[2][:4].tolist(), [1, 2, 5, 6])
        self.assertEqual(mappings[3][:4].tolist(), [9, 10, 23, 20])

        for ratio, retained_tokens, ring_tokens in (
            (4, 8, deepseek_v4.C4_STATE_RING_TOKENS),
            (128, 128, deepseek_v4.C128_STATE_RING_TOKENS),
        ):
            for tokens in (
                retained_tokens - 1,
                retained_tokens,
                retained_tokens + 1,
                4_096,
            ):
                with self.subTest(ratio=ratio, tokens=tokens):
                    values = stage(ratio, (257,), (tokens,), (3,))
                    token_slots, table, _, mappings, counts, page = values
                    pages, width, seed_count, retain_count = counts
                    self.assertEqual(len(set(token_slots.tolist())), tokens)
                    self.assertEqual(len(set(table[0, :width].tolist())), width)
                    self.assertEqual(pages, width)
                    self.assertLessEqual(seed_count, retained_tokens // page)
                    self.assertLessEqual(retain_count, retained_tokens // page)
                    self.assertTrue(
                        all(
                            destination // (ring_tokens // page) == 3
                            for destination in mappings[3][:retain_count].tolist()
                        )
                    )

    def test_prefill_stages_packed_rows_in_both_descriptor_parities(self) -> None:
        runtime = make_prefill_runtime(
            live_slots=deepseek_v4.MAX_PREFILL_REQUESTS,
            history_blocks=deepseek_v4.MAX_PREFILL_REQUESTS + 1,
        )
        calls = []
        stage = SimpleNamespace(
            stage_deepseek_v4_prefill_outputs=lambda *args: calls.append(
                tuple(value.clone() for value in args)
            ),
            publish_deepseek_v4_prefill_lengths=lambda lengths, slots, ends: (
                lengths.index_copy_(0, slots.to(torch.int64), ends)
            ),
        )
        with patch.dict(sys.modules, {"infer.models.deepseek_v4_flash.ops.stage": stage}):
            for parity in (0, 1):
                for requests in (1, 2, 4):
                    calls.clear()
                    slots = tuple(range(requests))
                    batch = runtime.stage_prefill(
                        parity,
                        tuple((10 + row,) for row in slots),
                        (0,) * requests,
                        slots,
                        tuple((0, (row + 1,)) for row in slots),
                        tuple(range(requests)),
                    )

                    self.assertEqual(
                        (batch.token_count, batch.request_count, batch.sample_count),
                        (requests, requests, requests),
                    )
                    self.assertEqual(calls[0][2].tolist(), [[row + 1] for row in slots])
                    device = runtime.staging[parity].device
                    self.assertEqual(
                        device.sample_indices[:requests].tolist(), list(range(requests))
                    )
                    self.assertEqual(
                        device.sample_slots[:requests].tolist(), list(slots)
                    )

        self.assertIsNot(runtime.staging[0].device, runtime.staging[1].device)

    def test_prefill_updates_full_tables_and_publishes_sample_rows(self) -> None:
        runtime = make_prefill_runtime()
        runtime.execution_tables[2, :3] = torch.tensor((5, 6, 7))
        calls = []
        stage = SimpleNamespace(
            stage_deepseek_v4_prefill_outputs=lambda *args: calls.append(
                tuple(value.clone() for value in args)
            ),
            publish_deepseek_v4_prefill_lengths=lambda lengths, slots, ends: (
                lengths.index_copy_(0, slots.to(torch.int64), ends)
            ),
        )
        with patch.dict(sys.modules, {"infer.models.deepseek_v4_flash.ops.stage": stage}):
            batch = runtime.stage_prefill(
                1,
                ((20, 21),),
                (129,),
                (2,),
                ((1, (9,)),),
                (0,),
            )

        self.assertEqual(calls[0][2].tolist(), [[5, 9]])
        self.assertEqual(runtime.execution_tables[2, :3].tolist(), [5, 9, 7])
        device = runtime.staging[1].device
        self.assertEqual((batch.c4_seed_count, batch.c128_seed_count), (1, 1))
        self.assertEqual(device.c4_seed_source[:1].tolist(), [8])
        self.assertEqual(device.c128_seed_source[:1].tolist(), [80])

        runtime.resources.head_tokens[0] = 1
        with patch.dict(sys.modules, {"infer.models.deepseek_v4_flash.ops.stage": stage}):
            runtime.publish_prefill(batch)
        self.assertEqual(int(runtime.sampled_tokens[2]), 1)

        empty = runtime.stage_prefill(0, (), (), (), (), ())
        self.assertEqual((empty.token_count, empty.request_count), (0, 0))

    def test_long_prefix_restore_stages_its_complete_initial_table(self) -> None:
        for start, block_count in ((8_192, 65), (32_768, 257), (69_632, 545)):
            with self.subTest(start=start):
                runtime = make_prefill_runtime(history_blocks=600)
                calls = []
                stage = SimpleNamespace(
                    stage_deepseek_v4_prefill_outputs=(
                        lambda *args, calls=calls: calls.append(
                            tuple(value.clone() for value in args)
                        )
                    )
                )
                blocks = tuple(range(block_count))

                with patch.dict(sys.modules, {"infer.models.deepseek_v4_flash.ops.stage": stage}):
                    batch = runtime.stage_prefill(
                        0,
                        ((7,),),
                        (start,),
                        (0,),
                        ((0, blocks),),
                        (),
                    )

                self.assertEqual(batch.table_width, block_count)
                self.assertEqual(
                    runtime.execution_tables[0, :block_count].tolist(), list(blocks)
                )
                self.assertEqual(calls[0][2][0].tolist(), list(blocks))

    def test_dep4_assembles_every_local_target_tensor_once(self) -> None:
        for rank in (0, 3):
            with self.subTest(rank=rank):
                view = FakeCheckpointView()
                with (
                    patch.dict(sys.modules, {"deep_gemm": FakeDeepGemm()}),
                    patch.object(megamoe, "_validate_hash_routes_tensor"),
                ):
                    weights = target.load_dep4_target_weights(view, rank, "meta")

                self.assertEqual(
                    tuple(field.name for field in fields(weights)),
                    ("embedding", "layers", "head_mhc", "final_norm", "head"),
                )
                self.assertEqual(
                    tuple(layer.layer_id for layer in weights.layers),
                    tuple(range(43)),
                )
                self.assertIsInstance(
                    weights.layers[0].attention,
                    attention.DeepSeekV4ProjectionWeights,
                )
                self.assertIsInstance(
                    weights.layers[2].attention,
                    attention.DeepSeekV4C4AttentionWeights,
                )
                self.assertIsInstance(
                    weights.layers[3].attention,
                    attention.DeepSeekV4C128AttentionWeights,
                )
                self.assertIsInstance(
                    weights.layers[42].attention,
                    attention.DeepSeekV4C4AttentionWeights,
                )

                local_experts = set(megamoe.local_routed_expert_ids(rank))
                expected = {
                    name
                    for name in checkpoint._expected_target_weight_map()
                    if ".ffn.experts." not in name
                    or int(name.split(".")[4]) in local_experts
                }
                self.assertEqual(len(view.calls), len(expected))
                self.assertEqual({call[0] for call in view.calls}, expected)
                self.assertTrue(
                    all(call[1:] == (rank, "meta", False) for call in view.calls)
                )

    def test_tep4_shards_every_marked_target_tensor_once(self) -> None:
        for rank in (0, 3):
            with self.subTest(rank=rank):
                view = FakeCheckpointView()
                with (
                    patch.dict(sys.modules, {"deep_gemm": FakeDeepGemm()}),
                    patch.object(megamoe, "_validate_hash_routes_tensor"),
                ):
                    weights = target.load_tep4_target_weights(view, rank, "meta")

                local_experts = set(megamoe.local_routed_expert_ids(rank))
                expected = {
                    name
                    for name in checkpoint._expected_target_weight_map()
                    if ".ffn.experts." not in name
                    or int(name.split(".")[4]) in local_experts
                }
                self.assertEqual(len(view.calls), len(expected))
                self.assertEqual({call[0] for call in view.calls}, expected)
                self.assertTrue(
                    all(call[1:] == (rank, "meta", True) for call in view.calls)
                )
                self.assertEqual(
                    view.loaded_shapes,
                    {
                        name: checkpoint._target_spec(name).local_shape()
                        for name in expected
                    },
                )
                self.assertEqual(weights.embedding.shape, (32320, 4096))
                self.assertEqual(weights.head.shape, (32320, 4096))
                projection = weights.layers[0].attention
                self.assertEqual(projection.sink.shape, (64,))
                self.assertEqual(projection.q_b.shape, (8192, 1024))
                self.assertEqual(projection.output_a.shape, (2, 1024, 4096))
                self.assertEqual(projection.output_b.shape, (4096, 2048))
                ffn = weights.layers[0].ffn
                self.assertEqual(ffn.shared_gate_up.shape, (1024, 4096))
                self.assertEqual(ffn.shared_gate_up_scale.shape, (8, 32))
                self.assertEqual(ffn.shared_down.shape, (4096, 512))
                self.assertEqual(ffn.shared_down_scale.shape, (32, 4))
                self.assertEqual(ffn.routed.l1[0].shape, (64, 4096, 2048))

    def test_eager_decode_preserves_all_43_layer_bindings_and_endpoint_order(
        self,
    ) -> None:
        layers = tuple(SimpleNamespace(layer_id=layer_id) for layer_id in range(43))
        weights = target.DeepSeekV4TargetWeights(
            "embedding", layers, "head_mhc", "final_norm", "head"
        )
        inputs = tuple(
            target.DeepSeekV4DecodeLayerInputs(
                *(
                    f"{field.name}-{layer_id}"
                    for field in fields(target.DeepSeekV4DecodeLayerInputs)
                )
            )
            for layer_id in range(43)
        )
        ops = FakeTargetOps()
        model = target.DeepSeekV4TargetModel(weights, ops)

        token_ids = tuple(range(8))
        output = model.decode(token_ids, inputs, "endpoint")

        self.assertEqual(output, "logits")
        self.assertEqual(
            [call[0] for call in ops.calls],
            ["embedding", *("layer" for _ in range(43)), "head"],
        )
        for layer_id, (call, bindings) in enumerate(
            zip(ops.calls[1:-1], inputs, strict=True)
        ):
            self.assertEqual(
                call[1:4], (layer_id, ("streams", layer_id - 1), token_ids)
            )
            self.assertEqual(
                call[4:],
                tuple(getattr(bindings, field.name) for field in fields(bindings)),
            )
        self.assertEqual(ops.calls[-1][1], ("streams", 42))
        self.assertEqual(
            ops.decode_kwargs,
            [{"attention_prepared": False, "next_weights": None}] * 43,
        )

        for rows in (1, 4, 16, 32):
            ops.calls.clear()
            ops.decode_kwargs.clear()
            model.decode(tuple(range(rows)), inputs, "endpoint")
            self.assertEqual(
                sum(kwargs["next_weights"] is not None for kwargs in ops.decode_kwargs),
                42,
            )
            self.assertEqual(
                [kwargs["attention_prepared"] for kwargs in ops.decode_kwargs],
                [False, *([True] * 42)],
            )
            for layer_id, kwargs in enumerate(ops.decode_kwargs[:-1]):
                self.assertIs(kwargs["next_weights"], layers[layer_id + 1])
            self.assertIsNone(ops.decode_kwargs[-1]["next_weights"])

        with self.assertRaisesRegex(ValueError, "every layer"):
            model.decode(token_ids, inputs[:-1], "endpoint")

    def test_eager_prefill_runs_all_layer_kinds_and_sampled_head_rows(self) -> None:
        allocation, _, _ = make_runtimes()
        ops = FakeTargetOps()
        model = target.DeepSeekV4TargetModel(fake_runtime_weights(), ops)
        batch = target.DeepSeekV4PrefillBatch(0, 2, 1, 1, 1, 1, 0, 0, 0, 0, 1, 2)

        output = model.prefill(allocation, batch)

        self.assertEqual(output, "logits")
        calls = [call for call in ops.calls if call[0] == "prefill_layer"]
        self.assertEqual(len(calls), deepseek_v4.NUM_LAYERS)
        self.assertEqual(
            [call[2] for call in calls].count("swa"), sum(1 for layer_id in range(deepseek_v4.NUM_LAYERS) if deepseek_v4.attention_kind(layer_id) == "swa")
        )
        self.assertEqual(
            [call[2] for call in calls].count("csa"), len(deepseek_v4.CSA_LAYER_IDS)
        )
        self.assertEqual(
            [call[2] for call in calls].count("hca"), len(deepseek_v4.HCA_LAYER_IDS)
        )
        for call in calls:
            kind = call[2]
            self.assertEqual(call[12].streams_out.shape, (2, 4, 4096))
            self.assertEqual(call[6] is not None, kind != "swa")
            self.assertEqual(call[7] is not None, kind == "csa")
            self.assertEqual(call[10] is not None, kind == "csa")
            self.assertEqual(call[13] is not None, kind != "swa")
            self.assertEqual(call[14] is not None, kind == "csa")
        self.assertEqual([call[0] for call in ops.calls[-2:]], ["head", "select"])

        ops.calls.clear()
        empty = target.DeepSeekV4PrefillBatch(1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        self.assertIsNone(model.prefill(allocation, empty))
        self.assertEqual(
            sum(call[0] == "prefill_layer" for call in ops.calls),
            deepseek_v4.NUM_LAYERS,
        )
        self.assertNotIn("head", (call[0] for call in ops.calls))

    def test_tep4_partial_prefill_slices_embedding_scratch_to_live_tokens(self) -> None:
        runtime, _, _ = make_runtimes(tensor_parallel=True)
        ops = FakeTargetOps()
        model = target.DeepSeekV4TargetModel(fake_runtime_weights(), ops)
        batch = target.DeepSeekV4PrefillBatch(0, 2, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0)

        model.prefill(runtime, batch)

        embedding = next(call for call in ops.calls if call[0] == "prefill_embedding")
        self.assertEqual(embedding[5].shape, (2,))
        self.assertEqual(embedding[6].shape, (2,))

    def test_endpoint_workspace_and_graph_captured_head_expression(self) -> None:
        workspace = target.endpoint_workspace_shapes(8)
        self.assertEqual(
            tuple(field.name for field in fields(workspace)),
            ("streams", "collapsed", "normalized", "head_output", "logits"),
        )
        self.assertEqual(workspace.streams, (8, 4, 4096))
        self.assertEqual(workspace.head_output, (8, 129280))
        self.assertEqual(workspace.logits, (8, 129280))
        self.assertEqual(target.endpoint_workspace_shapes(0).logits, (0, 129280))
        self.assertEqual(target.endpoint_workspace_shapes(384).logits, (384, 129280))
        with self.assertRaisesRegex(ValueError, "execution bucket"):
            target.endpoint_workspace_shapes(17)

        tep4 = target.tep4_endpoint_workspace_shapes(8)
        self.assertEqual(tep4.streams, (8, 4, 4096))
        self.assertEqual(tep4.logits, (8, 32320))
        self.assertEqual(tep4.gathered_candidates, (32, 2))
        self.assertEqual(tep4.tokens, (8,))
        self.assertNotIn("head_output", tep4.__dataclass_fields__)
        self.assertEqual(target.tep4_endpoint_workspace_shapes(3).logits, (3, 32320))
        self.assertEqual(target.tep4_endpoint_workspace_shapes(6).logits, (6, 32320))
        with self.assertRaisesRegex(ValueError, "execution bucket"):
            target.tep4_endpoint_workspace_shapes(17)

        entrypoints = OPS_SOURCE[
            OPS_SOURCE.index("class DeepSeekV4TargetOps:") : OPS_SOURCE.index(
                "class DeepSeekV4TEP4TargetOps"
            )
        ]
        self.assertLess(
            entrypoints.index("torch.index_select"),
            entrypoints.index("head_output = _decode_head("),
        )
        head = OPS_SOURCE[
            OPS_SOURCE.index("def _decode_head") : OPS_SOURCE.index("def _hc_head")
        ]
        ordered = (
            "if batch_size:",
            "collapsed = _hc_head(",
            "rmsnorm(",
            "torch.mm(",
            "out=output",
        )
        offsets = [head.index(operation) for operation in ordered]
        self.assertEqual(offsets, sorted(offsets))
        expression = OPS_SOURCE[OPS_SOURCE.index("def _hc_head") :]
        exact = (
            "shape, dtype = hidden_states.size(), hidden_states.dtype",
            "x = hidden_states.flatten(1).float()",
            "torch.rsqrt(x.square().mean(-1, keepdim=True) + RMS_NORM_EPS)",
            "F.linear(x, hc_fn.float()) * rsqrt",
            "torch.sigmoid(mixes * hc_scale.float() + hc_base.float()) + MHC_EPS",
            "torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=1).to(dtype)",
        )
        offsets = [expression.index(operation) for operation in exact]
        self.assertEqual(offsets, sorted(offsets))
        base_ops = OPS_SOURCE[
            OPS_SOURCE.index("class DeepSeekV4TargetOps:") : OPS_SOURCE.index(
                "class DeepSeekV4TEP4TargetOps"
            )
        ]
        for forbidden in (
            "import deep_gemm",
            "tf32_hc_prenorm_gemm",
            "head_dot",
            "head_sqrsum",
            "collapsed_fp32",
        ):
            self.assertNotIn(forbidden, base_ops)

    def test_runtime_binds_fixed_buckets_shared_arenas_and_true_layer_parity(
        self,
    ) -> None:
        live_slots = 6
        allocation, metadata, mega_buffer = make_runtimes(live_slots)
        verifies = allocation.verify
        runtimes = {key: verify.execution for key, verify in verifies.items()}

        self.assertEqual(
            set(runtimes),
            {
                (batch, parity)
                for batch in (0, *deepseek_v4.DECODE_BATCH_SIZES)
                for parity in (0, 1)
            },
        )
        self.assertIs(allocation.execution_tables, runtimes[64, 1].execution_tables)
        self.assertIsNone(allocation.prefix_snapshots)
        self.assertEqual(len(allocation.bindings), deepseek_v4.NUM_LAYERS)
        self.assertEqual(len(allocation.staging), 2)
        attention_rows = (
            deepseek_v4.MAX_PREFILL_REQUESTS
            * (deepseek_v4.INDEX_CONTEXT_CAPACITY + deepseek_v4.SWA_WINDOW_TOKENS)
            + deepseek_v4.PREFILL_CHUNK_TOKENS
        )
        self.assertEqual(
            allocation.resources.attention.kv.shape,
            (attention_rows, 1, deepseek_v4.HEAD_DIM),
        )
        for (batch, parity), runtime in runtimes.items():
            execution_rows = batch * deepseek_v4.DSPARK_VERIFY_WIDTH
            self.assertEqual(runtime.execution_tables.shape, (live_slots + 64, 8192))
            self.assertEqual(runtime.token_ids.shape, (execution_rows,))
            self.assertEqual(runtime._descriptors.positions.shape, (execution_rows,))
            for field in fields(runtime._descriptors):
                if field.name != "storage":
                    self.assertEqual(
                        getattr(runtime._descriptors, field.name).storage_offset() % 4,
                        0,
                    )
            self.assertEqual(len(runtime.layer_inputs), 43)
            self.assertEqual(runtime.endpoint.logits.shape, (execution_rows, 129280))
            self.assertEqual(
                runtime.layer_inputs[0].raw.indices.shape,
                (execution_rows, 1, 128),
            )
            self.assertEqual(
                runtime.layer_inputs[2].batch.state_table.shape, (execution_rows, 3)
            )
            self.assertEqual(
                runtime.layer_inputs[3].batch.state_table.shape, (execution_rows, 17)
            )
            self.assertEqual(
                runtime.layer_inputs[2].attention_state.query.block_table.shape,
                (execution_rows, 8192),
            )
            c4_states = [
                item.attention_state
                for item in runtime.layer_inputs
                if isinstance(item.attention_state, attention.DeepSeekV4C4DecodeState)
            ]
            self.assertEqual(
                [state.prepare_selection for state in c4_states],
                [True, *([False] * (len(c4_states) - 1))],
            )
            self.assertTrue(
                all(item.metadata is metadata for item in runtime.layer_inputs)
            )
            self.assertTrue(
                all(item.mega_buffer is mega_buffer for item in runtime.layer_inputs)
            )
            outputs = [item.workspace.streams_out for item in runtime.layer_inputs]
            self.assertTrue(
                all(outputs[index] is outputs[index % 2] for index in range(43))
            )
            self.assertIsNot(outputs[0], outputs[1])
            self.assertTrue(
                all(
                    item.workspace.ffn is runtime.layer_inputs[0].workspace.ffn
                    for item in runtime.layer_inputs
                )
            )
            workspaces = [runtime.layer_inputs[index].workspace for index in range(3)]
            for field in ("post", "comb", "streams_mid"):
                self.assertTrue(
                    all(
                        getattr(workspace, field) is getattr(workspaces[0], field)
                        for workspace in workspaces[1:]
                    )
                )
            self.assertIs(
                runtime.layer_inputs[2].workspace.attention.common,
                runtime.layer_inputs[0].workspace.attention,
            )
            common = runtime.layer_inputs[2].workspace.attention.common
            self.assertIs(runtime.layer_inputs[1].workspace.attention, common)
            for field in ("hidden_fp32", "normalized"):
                self.assertIs(
                    getattr(runtime.layer_inputs[1].workspace.attention, field),
                    getattr(common, field),
                )
            packed = runtime.layer_inputs[2].workspace.attention.packed_q
            self.assertNotEqual(
                common.projected_q.untyped_storage()._cdata,
                packed.untyped_storage()._cdata,
            )
            self.assertTrue(common.projected_q.is_contiguous())
            verify = verifies[batch, parity]
            self.assertEqual(verify.candidates.shape, (batch, 6))
            self.assertEqual(verify.target_tokens.shape, (batch, 6))
            self.assertEqual(verify.captured_hidden.shape, (batch, 6, 12288))

        raw_storage = runtimes[1, 0].layer_inputs[0].raw.cache.untyped_storage()._cdata
        self.assertEqual(
            runtimes[64, 1].layer_inputs[42].raw.cache.untyped_storage()._cdata,
            raw_storage,
        )
        self.assertEqual(
            runtimes[1, 0].layer_inputs[2].main.cache.untyped_storage()._cdata,
            runtimes[64, 1].layer_inputs[42].main.cache.untyped_storage()._cdata,
        )
        index_caches = [
            inputs.attention_state.index.cache
            for inputs in runtimes[1, 0].layer_inputs
            if isinstance(inputs.attention_state, attention.DeepSeekV4C4DecodeState)
        ]
        self.assertEqual(len(index_caches), len(deepseek_v4.CSA_LAYER_IDS))
        self.assertTrue(
            all(
                cache.shape == (8, 2_176) and cache.stride() == (2_176, 1)
                for cache in index_caches
            )
        )
        self.assertTrue(all(cache.storage_offset() == 0 for cache in index_caches))
        self.assertEqual(
            len({cache.untyped_storage()._cdata for cache in index_caches}),
            len(index_caches),
        )
        self.assertIs(
            runtimes[1, 0].layer_inputs[0].workspace,
            runtimes[1, 1].layer_inputs[0].workspace,
        )
        self.assertIs(
            runtimes[1, 0].layer_inputs[2].main.kv_score,
            runtimes[1, 1].layer_inputs[2].main.kv_score,
        )
        self.assertIs(
            runtimes[1, 0].layer_inputs[2].attention_state.workspace,
            runtimes[1, 1].layer_inputs[2].attention_state.workspace,
        )
        self.assertIs(runtimes[1, 0].endpoint, runtimes[1, 1].endpoint)
        self.assertIsNot(runtimes[1, 0].endpoint, runtimes[8, 0].endpoint)
        self.assertIsNot(runtimes[1, 0].token_ids, runtimes[1, 1].token_ids)
        self.assertIsNot(runtimes[1, 0]._descriptors, runtimes[1, 1]._descriptors)
        self.assertIsNot(
            runtimes[1, 0].layer_inputs[0].raw.indices,
            runtimes[1, 1].layer_inputs[0].raw.indices,
        )
        self.assertIs(runtimes[1, 0].execution_tables, runtimes[64, 1].execution_tables)

    def test_snapshot_allocation_is_separate_from_live_and_dummy_state(self) -> None:
        live_slots = 6
        allocation, _, _ = make_runtimes(live_slots, snapshot_slots=2)
        snapshots = allocation.prefix_snapshots
        assert snapshots is not None

        self.assertEqual(allocation.state.raw_window.shape[1], live_slots + 64)
        self.assertEqual(snapshots.state.raw_window.shape[1], 2)
        self.assertEqual(snapshots.c4_main_tail.shape[1], 2)
        self.assertEqual(len(snapshots.c4_index_tail), len(deepseek_v4.CSA_LAYER_IDS))
        self.assertEqual(snapshots.c128_main_tail.shape[1], 2)
        self.assertEqual(snapshots.committed_lengths.shape, (2,))

    def test_tep4_runtime_uses_local_heads_with_native_six_row_verification(
        self,
    ) -> None:
        runtime, _, _ = make_runtimes(tensor_parallel=True)
        execution = runtime.verify[1, 0].execution

        self.assertEqual(execution.token_ids.shape, (6,))
        self.assertEqual(execution.endpoint.logits.shape, (6, 32320))
        self.assertEqual(
            execution.layer_inputs[0].workspace.attention.projected_q.shape,
            (6, deepseek_v4.LOCAL_QUERY_HEADS, deepseek_v4.HEAD_DIM),
        )
        self.assertEqual(
            execution.layer_inputs[0].raw.query.shape,
            (6, 1, deepseek_v4.NUM_QUERY_HEADS, deepseek_v4.HEAD_DIM),
        )
        self.assertEqual(
            runtime.resources.workspace_a.attention.projected_q.shape,
            (
                deepseek_v4.PREFILL_CHUNK_TOKENS,
                deepseek_v4.LOCAL_QUERY_HEADS,
                deepseek_v4.HEAD_DIM,
            ),
        )
        self.assertEqual(runtime.resources.embedding_tokens.dtype, torch.int64)
        self.assertEqual(runtime.resources.embedding_select.dtype, torch.bool)

    def test_target_only_runtime_allocates_one_row_per_decode_slot(self) -> None:
        runtime, _, _ = make_runtimes(speculative=False)
        keys = {
            (bucket, parity)
            for bucket in (0, *deepseek_v4.DECODE_BATCH_SIZES)
            for parity in range(deepseek_v4.DECODE_DESCRIPTOR_PARITIES)
        }

        self.assertFalse(runtime.verify)
        self.assertEqual(set(runtime.decode), keys)
        self.assertEqual(runtime.result_records.shape[1], 2)
        self.assertEqual(runtime.decode[1, 0].execution.token_ids.shape, (1,))
        self.assertEqual(runtime.decode[64, 0].execution.token_ids.shape, (64,))

    def test_target_only_decode_stages_samples_and_publishes_one_token(self) -> None:
        allocation, _, _ = make_runtimes(live_slots=6, speculative=False)
        runtime = allocation.decode[4, 1]
        staged = []
        stage = SimpleNamespace(stage_deepseek_v4_target=staged.append)

        with patch.dict(sys.modules, {"infer.models.deepseek_v4_flash.ops.stage": stage}):
            runtime.stage(((2, 0, 4), (5, 1, -1)))
            runtime.prepare()

        self.assertEqual(staged, [runtime])
        self.assertEqual(runtime.staging.host.state_slots.tolist(), [2, 5, 8, 9])
        self.assertEqual(
            runtime.staging.host.table_deltas.tolist(),
            [[0, 4], [1, -1], [0, -1], [0, -1]],
        )
        self.assertEqual(runtime.staging.host.active.tolist(), [1, 1, 0, 0])

        committed = torch.tensor([0, 0, 7, 0, 0, 11, 0, 0], dtype=torch.int32)
        sampled = torch.zeros(8, dtype=torch.int64)
        results = torch.zeros((8, 2), dtype=torch.int64)
        controls = target._DeepSeekV4TargetDecodeControls(
            torch.tensor([2, 5]),
            torch.zeros((2, 2), dtype=torch.int32),
            torch.tensor([1, 1], dtype=torch.uint8),
        )
        cpu_runtime = target.DeepSeekV4TargetDecodeRuntime(
            execution=object(),
            target_tokens=torch.tensor([71, 72]),
            records=torch.zeros((2, 2), dtype=torch.int64),
            lengths=torch.empty(2, dtype=torch.int32),
            staging=target._DeepSeekV4TargetDecodeStaging(controls, controls),
            committed_lengths=committed,
            sampled_tokens=sampled,
            result_records=results,
            live_slots=6,
        )

        cpu_runtime.publish()

        self.assertEqual(results[2].tolist(), [1, 71])
        self.assertEqual(results[5].tolist(), [1, 72])
        self.assertEqual(sampled[[2, 5]].tolist(), [71, 72])
        self.assertEqual(committed[[2, 5]].tolist(), [8, 12])

    def test_target_only_decode_runs_one_target_row_without_hidden_capture(
        self,
    ) -> None:
        allocation, _, _ = make_runtimes(speculative=False)
        decode = allocation.decode[1, 0]
        ops = FakeTargetOps()
        model = target.DeepSeekV4TargetModel(fake_runtime_weights(), ops)
        staged = []
        stage = SimpleNamespace(stage_deepseek_v4_target=staged.append)

        with patch.dict(sys.modules, {"infer.models.deepseek_v4_flash.ops.stage": stage}):
            model.decode_target(decode)

        self.assertEqual(staged, [decode])
        self.assertEqual(
            sum(call[0] == "layer" for call in ops.calls),
            deepseek_v4.NUM_LAYERS,
        )
        self.assertFalse(any(call[0] == "capture" for call in ops.calls))
        select = [call for call in ops.calls if call[0] == "select"]
        self.assertEqual(len(select), 1)
        self.assertEqual(select[0][1:3], ("logits", decode.execution.endpoint))
        self.assertIs(select[0][3], decode.target_tokens)

    def test_verify_flattens_six_candidates_into_one_target_traversal(self) -> None:
        allocation, _, _ = make_runtimes()
        verify = allocation.verify[1, 0]
        ops = FakeTargetOps()
        model = target.DeepSeekV4TargetModel(fake_runtime_weights(), ops)
        staged = []
        stage = SimpleNamespace(stage_deepseek_v4_verify=staged.append)

        with patch.dict(sys.modules, {"infer.models.deepseek_v4_flash.ops.stage": stage}):
            model.verify(verify)

        self.assertEqual(staged, [verify])
        self.assertEqual(
            sum(call[0] == "layer" for call in ops.calls),
            deepseek_v4.NUM_LAYERS,
        )
        self.assertEqual(
            ops.decode_kwargs,
            [{"attention_prepared": False, "next_weights": None}]
            * deepseek_v4.NUM_LAYERS,
        )
        captures = [call for call in ops.calls if call[0] == "capture"]
        self.assertEqual(
            [call[1] for call in captures],
            [("streams", layer) for layer in deepseek_v4.DSPARK_TARGET_LAYER_IDS],
        )
        self.assertTrue(all(call[2].shape == (6, 4096) for call in captures))
        select = [call for call in ops.calls if call[0] == "select"]
        self.assertEqual(len(select), 1)
        self.assertEqual(select[0][1:3], ("logits", verify.execution.endpoint))
        self.assertEqual(
            select[0][3].untyped_storage()._cdata,
            verify.target_tokens.untyped_storage()._cdata,
        )

    def test_decode_control_uses_parity_local_rows_and_unique_dummy_slots(self) -> None:
        allocation, _, _ = make_runtimes(live_slots=6)
        runtime = allocation.verify[8, 1]
        called = []
        stage = SimpleNamespace(stage_deepseek_v4_verify=called.append)

        with patch.dict(sys.modules, {"infer.models.deepseek_v4_flash.ops.stage": stage}):
            runtime.stage(((2, 0, 4, 6, False), (5, 1, -1, 3, True)))
            self.assertEqual(called, [])
            runtime.prepare()

        self.assertEqual(called, [runtime])
        self.assertEqual(
            runtime.staging.host.state_slots.tolist(),
            [2, 5, 8, 9, 10, 11, 12, 13],
        )
        self.assertEqual(
            runtime.staging.host.table_deltas.tolist(),
            [
                [0, 4],
                [1, -1],
                [0, -1],
                [0, -1],
                [0, -1],
                [0, -1],
                [0, -1],
                [0, -1],
            ],
        )
        self.assertEqual(
            runtime.staging.host.ignore_eos.tolist(),
            [0, 1, 0, 0, 0, 0, 0, 0],
        )
        largest = allocation.verify[64, 0]
        largest.stage(())
        self.assertEqual(
            largest.staging.host.state_slots.tolist(),
            list(range(6, 70)),
        )
        self.assertLess(
            max(largest.staging.host.state_slots),
            allocation.sampled_tokens.shape[0],
        )

    def test_default_megamoe_buffer_covers_the_prefill_chunk(self) -> None:
        calls = []
        metadata_calls = []
        mega_buffer = object()

        def allocate_metadata():
            metadata = object()
            metadata_calls.append(metadata)
            return metadata, None

        def allocate_mega_buffer(*args, **kwargs):
            calls.append((args, kwargs))
            return mega_buffer

        deep_gemm = SimpleNamespace(
            get_num_sms=lambda: 148,
            get_symm_buffer_for_mega_moe=allocate_mega_buffer,
        )
        flash_mla = SimpleNamespace(get_mla_metadata=allocate_metadata)
        rope_tables = (
            torch.empty((deepseek_v4.MAX_CONTEXT_TOKENS, 64), device="meta"),
            torch.empty((deepseek_v4.MAX_CONTEXT_TOKENS, 64), device="meta"),
        )
        with (
            patch.dict(sys.modules, {"deep_gemm": deep_gemm, "flash_mla": flash_mla}),
            patch.object(target, "_rope_table", side_effect=rope_tables),
        ):
            runtimes = target.allocate_target_runtime(
                fake_runtime_weights(), 6, 8, "meta"
            )

        expected_metadata = 3 * 2 * (len(deepseek_v4.DECODE_BATCH_SIZES) + 1)
        self.assertEqual(len(metadata_calls), expected_metadata)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0][2], deepseek_v4.PREFILL_CHUNK_TOKENS)
        self.assertEqual(calls[0][1], {"mma_type": "fp8xfp4"})
        assigned_metadata = {
            id(inputs.metadata)
            for verify in runtimes.verify.values()
            for inputs in verify.execution.layer_inputs
        }
        self.assertEqual(assigned_metadata, set(map(id, metadata_calls)))
        runtime = runtimes.verify[1, 0].execution
        self.assertIsNot(
            runtime.layer_inputs[0].metadata, runtime.layer_inputs[2].metadata
        )
        self.assertIsNot(
            runtime.layer_inputs[2].metadata, runtime.layer_inputs[3].metadata
        )
        self.assertIsNot(
            runtime.layer_inputs[0].metadata,
            runtimes.verify[1, 1].execution.layer_inputs[0].metadata,
        )
        self.assertIs(
            runtimes.verify[64, 1].execution.layer_inputs[42].mega_buffer,
            mega_buffer,
        )

    def test_attention_metadata_is_reset_once_per_kind(self) -> None:
        def metadata():
            return SimpleNamespace(
                have_initialized=True,
                config=object(),
                tile_scheduler_metadata=object(),
                num_splits=object(),
            )

        swa, csa, hca = metadata(), metadata(), metadata()
        verify = SimpleNamespace(
            execution=SimpleNamespace(
                layer_inputs=tuple(
                    SimpleNamespace(metadata=value)
                    for value in (swa, swa, csa, hca, csa)
                )
            )
        )

        target.DeepSeekV4VerifyRuntime.reset_attention_metadata(verify)

        for value in (swa, csa, hca):
            self.assertFalse(value.have_initialized)
            self.assertIsNone(value.config)
            self.assertIsNone(value.tile_scheduler_metadata)
            self.assertIsNone(value.num_splits)

    def test_rope_tables_match_official_swa_and_compressed_formulas(self) -> None:
        positions = torch.tensor([0, 1, 65_535, 65_536, 1_048_575])

        def official(base, yarn):
            frequencies = 1 / (base ** (torch.arange(0, 64, 2) / 64))
            if yarn:
                correction = lambda rotations: (
                    64
                    * math.log(65_536 / (rotations * 2 * math.pi))
                    / (2 * math.log(base))
                )
                low, high = math.floor(correction(32)), math.ceil(correction(1))
                ramp = ((torch.arange(32) - low) / (high - low)).clamp(0, 1)
                frequencies = frequencies / 16 * ramp + frequencies * (1 - ramp)
            angles = positions.to(torch.float32)[:, None] * frequencies
            return torch.cat((angles.cos(), angles.sin()), dim=1)

        self.assertTrue(
            torch.allclose(
                target._rope_table(torch, "cpu", compressed=False, positions=positions),
                official(10_000.0, False),
            )
        )
        self.assertTrue(
            torch.allclose(
                target._rope_table(torch, "cpu", compressed=True, positions=positions),
                official(160_000.0, True),
            )
        )


class DeepSeekV4TargetOpsTest(unittest.TestCase):
    def test_tep4_prefill_embedding_uses_local_ids_and_one_collective(self) -> None:
        ops = object.__new__(load_target_ops("DeepSeekV4TEP4TargetOps"))
        ops._rank = 1
        ops._process_group = object()
        weights = SimpleNamespace(embedding=torch.tensor(((1.0, 2.0), (3.0, 4.0))))
        collapsed = torch.empty((2, 2))
        streams = torch.empty((2, deepseek_v4.MHC_STREAMS, 2))
        local_tokens = torch.empty(2, dtype=torch.int64)
        select = torch.empty(2, dtype=torch.bool)

        with patch.object(torch.distributed, "all_reduce") as all_reduce:
            output = ops.prefill_embedding(
                torch.tensor((0, deepseek_v4.VOCAB_SIZE // 4 + 1)),
                weights,
                collapsed,
                streams,
                local_tokens,
                select,
            )

        self.assertIs(output, streams)
        self.assertEqual(local_tokens.tolist(), [0, 1])
        self.assertEqual(select.tolist(), [False, True])
        self.assertEqual(collapsed.tolist(), [[0.0, 0.0], [3.0, 4.0]])
        self.assertTrue(torch.equal(streams, collapsed.unsqueeze(1).expand_as(streams)))
        all_reduce.assert_called_once_with(collapsed, group=ops._process_group)

    def test_capture_hidden_accepts_only_the_row_major_capture_slice(self) -> None:
        capture_width = (
            len(deepseek_v4.DSPARK_TARGET_LAYER_IDS) * deepseek_v4.HIDDEN_SIZE
        )
        streams = torch.ones(
            (2, deepseek_v4.MHC_STREAMS, deepseek_v4.HIDDEN_SIZE),
            dtype=torch.bfloat16,
        )
        captured = torch.full((2, capture_width), -1, dtype=torch.bfloat16)
        output = captured[:, deepseek_v4.HIDDEN_SIZE : 2 * deepseek_v4.HIDDEN_SIZE]
        ops = object.__new__(load_target_ops())

        ops.capture_hidden(streams, output)

        self.assertEqual(output.stride(), (capture_width, 1))
        self.assertTrue(torch.all(output == 1))
        self.assertTrue(torch.all(captured[:, : deepseek_v4.HIDDEN_SIZE] == -1))
        self.assertTrue(torch.all(captured[:, 2 * deepseek_v4.HIDDEN_SIZE :] == -1))
        with self.assertRaisesRegex(ValueError, "capture output has the wrong layout"):
            ops.capture_hidden(streams, torch.empty_like(output).contiguous())

    def test_greedy_accepts_every_prefix_length_and_publishes_selected_tail(
        self,
    ) -> None:
        groups = deepseek_v4.DSPARK_VERIFY_WIDTH
        target_tokens = torch.arange(groups * groups, dtype=torch.int64).view(
            groups, groups
        )
        candidates = torch.full((groups, groups), -1, dtype=torch.int64)
        for accepted in range(1, groups + 1):
            row = accepted - 1
            candidates[row, 1:accepted] = target_tokens[row, : accepted - 1]
        state_slots = torch.arange(groups, dtype=torch.int32)
        committed = torch.arange(10, 10 + groups, dtype=torch.int32)
        verify = SimpleNamespace(
            candidates=candidates,
            target_tokens=target_tokens,
            matches=torch.empty((groups, groups - 1), dtype=torch.bool),
            non_eos=torch.empty(groups, dtype=torch.bool),
            accepted=torch.empty(groups, dtype=torch.int32),
            records=torch.empty((groups, groups + 1), dtype=torch.int64),
            tail_indices=torch.empty((groups, 1), dtype=torch.int64),
            tail_tokens=torch.empty((groups, 1), dtype=torch.int64),
            lengths=torch.empty(groups, dtype=torch.int32),
            staging=SimpleNamespace(
                device=SimpleNamespace(
                    state_slots=state_slots.to(torch.int64),
                    remaining=torch.full((groups,), groups, dtype=torch.int32),
                    ignore_eos=torch.zeros(groups, dtype=torch.uint8),
                    active=torch.ones(groups, dtype=torch.uint8),
                )
            ),
            committed_lengths=committed.clone(),
            sampled_tokens=torch.zeros(groups, dtype=torch.int64),
            result_records=torch.zeros((groups, groups + 1), dtype=torch.int64),
        )
        ops = object.__new__(load_target_ops())

        ops.accept_greedy(verify)
        ops.publish_verified(verify)

        accepted = torch.arange(1, groups + 1, dtype=torch.int32)
        self.assertTrue(torch.equal(verify.accepted, accepted))
        self.assertTrue(torch.equal(verify.committed_lengths, committed + accepted))
        self.assertTrue(
            torch.equal(verify.result_records[:, 0], accepted.to(torch.int64))
        )
        self.assertTrue(torch.equal(verify.result_records[:, 1:], target_tokens))
        self.assertEqual(
            verify.sampled_tokens.tolist(),
            [int(target_tokens[row, row]) for row in range(groups)],
        )

    def test_acceptance_honors_eos_policy_and_remaining_budget(self) -> None:
        verify = SimpleNamespace(
            candidates=torch.tensor(((0, 10, 1, 12, 13, 14),) * 3),
            target_tokens=torch.tensor(((10, 1, 12, 13, 14, 15),) * 3),
            matches=torch.empty((3, 5), dtype=torch.bool),
            non_eos=torch.empty(3, dtype=torch.bool),
            accepted=torch.empty(3, dtype=torch.int32),
            staging=SimpleNamespace(
                device=SimpleNamespace(
                    remaining=torch.tensor((6, 6, 1), dtype=torch.int32),
                    ignore_eos=torch.tensor((0, 1, 1), dtype=torch.uint8),
                    active=torch.ones(3, dtype=torch.uint8),
                )
            ),
        )

        object.__new__(load_target_ops()).accept_greedy(verify)

        self.assertEqual(verify.accepted.tolist(), [2, 6, 1])


if __name__ == "__main__":
    unittest.main()

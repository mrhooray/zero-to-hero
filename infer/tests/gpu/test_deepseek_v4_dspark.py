import importlib
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.gpu

import torch

from infer.models.deepseek_v4_flash import model as deepseek_v4
from infer.models.deepseek_v4_flash import checkpoint
from infer.models.deepseek_v4_flash import dspark


class _CheckpointView:
    def __init__(self) -> None:
        self.names = []

    def load_dspark_tensor(self, name, _rank, device):
        self.names.append(name)
        spec = checkpoint._dspark_spec(name)
        dtype = {
            "BF16": torch.bfloat16,
            "F32": torch.float32,
            "F8_E4M3": torch.float8_e4m3fn,
            "F8_E8M0": torch.float8_e8m0fnu,
            "I8": torch.int8,
            "I64": torch.int64,
        }[spec.dtype]
        return torch.empty(spec.shape, dtype=dtype, device=device)


class _ModelOps:
    def __init__(self) -> None:
        self.calls = []

    def prepare(self, captured, batch, weights, target_weights, workspace):
        self.calls.append(("prepare", captured, batch, weights, target_weights))
        return "main"

    def stage(
        self,
        streams,
        weights,
        main,
        batch,
        window,
        workspace,
        mega_buffer,
    ):
        self.calls.append(("stage", weights.stage_id, main, batch, window, mega_buffer))
        return f"streams-{weights.stage_id}"

    def head(self, streams, batch, weights, target_weights, workspace):
        self.calls.append(("head", streams, batch, weights, target_weights))
        return "output"


def _load_ops_module(calls, writer_calls):
    core = types.ModuleType("infer.models.deepseek_v4_flash.ops.core")
    attention = types.ModuleType("infer.models.deepseek_v4_flash.ops.attention")
    target = types.ModuleType("infer.models.deepseek_v4_flash.ops.target")

    def quantize(_source, output, scale):
        calls.append("quantize")
        output.zero_()
        scale.fill_(1)

    def fp8_gemm(_a, _b, _a_scale, _b_scale, output):
        calls.append("fp8_gemm")
        output.zero_()

    def run_attention(raw, extra, indices, lengths, metadata, page, heads):
        calls.append("attention")
        self = raw.query
        return self.new_zeros((self.shape[0], 1, heads, self.shape[-1])), None

    core._quantize = quantize
    core._fp8_gemm = fp8_gemm
    core._attention = run_attention

    class AttentionOps:
        pass

    def mhc_pre(streams, _weights, workspace):
        calls.append("mhc_pre")
        workspace.collapsed.copy_(streams[:, 0])

    def mhc_post(_hidden, _residual, _workspace, output):
        calls.append("mhc_post")
        output.zero_()

    def project_common(_hidden, _weights, workspace):
        calls.append("project_common")
        workspace.projected_q.zero_()
        workspace.normalized_kv.zero_()

    def project_query(_weights, _workspace, output):
        calls.append("project_query")
        output.zero_()

    def rmsnorm(source, _weight, output):
        calls.append("rmsnorm")
        output.copy_(source)

    def project_output(attended, _weights, _workspace, _batch, _group):
        calls.append("project_output")
        return attended.new_zeros((attended.shape[0], deepseek_v4.HIDDEN_SIZE))

    attention.DeepSeekV4AttentionOps = AttentionOps
    attention._mhc_pre = mhc_pre
    attention._mhc_post = mhc_post
    attention._project_common = project_common
    attention._project_query = project_query
    attention._project_output = project_output
    attention._rmsnorm = rmsnorm
    target._hc_head = lambda *_args: None

    modules = {
        "infer.models.deepseek_v4_flash.ops.core": core,
        "infer.models.deepseek_v4_flash.ops.attention": attention,
        "infer.models.deepseek_v4_flash.ops.target": target,
    }
    name = "infer.models.deepseek_v4_flash.ops.dspark"
    with patch.dict(sys.modules, modules):
        sys.modules.pop(name, None)
        module = importlib.import_module(name)
        sys.modules.pop(name, None)

    def writer(*args):
        calls.append("writer")
        writer_calls.append(args)
        args[2].zero_()

    module._test_writer = writer
    return module


def _projection():
    return SimpleNamespace(
        qkv_a=torch.empty((1536, 1)),
        qkv_a_scale=torch.empty((12, 1)),
        kv_norm=torch.empty(512),
        sink=torch.empty(64),
    )


class DeepSeekV4DSparkModelTest(unittest.TestCase):
    def test_pinned_recipe_has_one_source_of_shape_truth(self) -> None:
        self.assertEqual(deepseek_v4.DSPARK_STAGE_COUNT, 3)
        self.assertEqual(deepseek_v4.DSPARK_BLOCK_SIZE, 5)
        self.assertEqual(deepseek_v4.DSPARK_VERIFY_WIDTH, 6)
        self.assertEqual(deepseek_v4.DSPARK_WINDOW_TOKENS, 128)
        self.assertEqual(deepseek_v4.DSPARK_PAGE_TOKENS, 64)
        self.assertEqual(deepseek_v4.DSPARK_TARGET_LAYER_IDS, (40, 41, 42))
        self.assertEqual(deepseek_v4.DSPARK_NOISE_TOKEN_ID, 128_799)
        self.assertEqual(deepseek_v4.DSPARK_MARKOV_RANK, 256)

    def test_loads_three_exact_stages_and_checkpoint_endpoints(self) -> None:
        view = _CheckpointView()
        routed = tuple(object() for _ in range(deepseek_v4.DSPARK_STAGE_COUNT))
        with patch.object(
            dspark.megamoe,
            "load_dep4_dspark_moe_weights",
            side_effect=routed,
        ) as load_moe:
            weights = dspark.load_dep4_dspark_weights(view, 2, "meta")

        expected = set()
        for stage in range(deepseek_v4.DSPARK_STAGE_COUNT):
            expected.update(
                f"mtp.{stage}.{spec.name}"
                for spec in deepseek_v4.COMPRESSED_ATTENTION_COMMON_WEIGHTS
            )
            expected.update(
                f"mtp.{stage}.{name}"
                for name in (
                    "hc_attn_base",
                    "hc_attn_fn",
                    "hc_attn_scale",
                    "hc_ffn_base",
                    "hc_ffn_fn",
                    "hc_ffn_scale",
                    "ffn_norm.weight",
                )
            )
        expected.update(
            f"mtp.0.{spec.name}" for spec in deepseek_v4.DSPARK_STAGE0_WEIGHTS
        )
        expected.update(
            f"mtp.2.{spec.name}" for spec in deepseek_v4.DSPARK_ENDPOINT_WEIGHTS
        )
        self.assertEqual(set(view.names), expected)
        self.assertEqual(tuple(stage.stage_id for stage in weights.stages), (0, 1, 2))
        self.assertEqual(tuple(stage.ffn for stage in weights.stages), routed)
        self.assertEqual(weights.main_projection.shape, (4096, 12288))
        self.assertEqual(weights.main_projection_scale.dtype, torch.float32)
        self.assertEqual(weights.markov_embedding.shape, (129280, 256))
        self.assertEqual(weights.confidence.shape, (1, 4352))
        self.assertEqual(load_moe.call_count, deepseek_v4.DSPARK_STAGE_COUNT)

    def test_model_runs_all_stages_in_checkpoint_order(self) -> None:
        stages = tuple(SimpleNamespace(stage_id=stage) for stage in range(3))
        weights = SimpleNamespace(stages=stages)
        target_embedding = object()
        target_head = object()
        ops = _ModelOps()
        model = dspark.DeepSeekV4DSparkModel(
            weights, target_embedding, target_head, ops
        )
        batch = object()
        windows = tuple(object() for _ in stages)
        workspace = SimpleNamespace(layer=SimpleNamespace(streams_out="embedding"))

        result = model.forward("captured", batch, windows, workspace, "mega")

        self.assertEqual(result, "output")
        self.assertEqual(
            [call[0] for call in ops.calls],
            ["prepare", "stage", "stage", "stage", "head"],
        )
        self.assertEqual([call[1] for call in ops.calls[1:4]], [0, 1, 2])
        self.assertEqual([call[4] for call in ops.calls[1:4]], list(windows))
        self.assertIs(ops.calls[0][4], target_embedding)
        self.assertIs(ops.calls[-1][4], target_head)


class DeepSeekV4DSparkRuntimeTest(unittest.TestCase):
    def test_allocates_model_local_windows_and_reuses_target_resources(self) -> None:
        cos_sin = object()
        mega_buffer = object()
        target = SimpleNamespace(
            live_slots=64,
            cos_sin=(cos_sin, object()),
            resources=SimpleNamespace(mega_buffer=mega_buffer),
            prefix_snapshots=SimpleNamespace(
                committed_lengths=torch.empty(2, device="meta")
            ),
        )
        metadata = []

        runtime = dspark.allocate_dep4_dspark_runtime(
            target,
            "meta",
            metadata_allocator=lambda: metadata.append(object()) or metadata[-1],
        )

        self.assertEqual(len(runtime.windows), deepseek_v4.DSPARK_STAGE_COUNT)
        self.assertEqual(
            {tuple(window.shape) for window in runtime.windows},
            {(128, 2, 37_440)},
        )
        self.assertEqual(runtime.anchor_hidden.shape, (128, 12_288))
        self.assertEqual(
            {tuple(window.shape) for window in runtime.prefix_snapshots.windows},
            {(2, 2, 37_440)},
        )
        self.assertEqual(runtime.prefix_snapshots.anchor_hidden.shape, (2, 12_288))
        self.assertEqual(runtime.state_slot_bytes, 249_216)
        self.assertEqual(set(runtime.workspaces), {0, *deepseek_v4.DECODE_BATCH_SIZES})
        self.assertEqual(
            set(runtime.batches),
            {
                (size, parity)
                for size in (0, *deepseek_v4.DECODE_BATCH_SIZES)
                for parity in range(2)
            },
        )
        self.assertEqual(len(metadata), 16)
        self.assertIs(runtime.mega_buffer, mega_buffer)
        self.assertIs(runtime.batches[1, 0].cos_sin, cos_sin)
        self.assertFalse(hasattr(runtime, "execution_tables"))
        self.assertFalse(hasattr(runtime, "committed_lengths"))
        workspace = runtime.workspaces[64]
        self.assertEqual(workspace.main_input_fp8.shape, (320, 12288))
        self.assertEqual(workspace.query.shape, (320, 1, 64, 512))
        self.assertEqual(workspace.block_cache.shape, (64, 37440))
        self.assertEqual(workspace.anchor_hidden.shape, (64, 12_288))
        self.assertEqual(workspace.commit_hidden.shape, (320, 12_288))

    def test_owns_parity_local_device_batches_and_resets_full_slot(self) -> None:
        batches = {
            (8, parity): dspark._allocate_batch(
                torch, "cpu", 8, torch.empty((136, 64)), object()
            )
            for parity in range(2)
        }
        windows = tuple(torch.ones((71, 2, 4)) for _ in range(3))
        anchors = torch.ones((71, 12_288), dtype=torch.bfloat16)
        runtime = dspark.DeepSeekV4DSparkRuntime(
            windows, anchors, batches, {}, object(), live_slots=64
        )

        self.assertEqual(runtime.batch(8, 0).active_count, 8)
        self.assertNotEqual(
            batches[8, 0].anchor_token_ids.data_ptr(),
            batches[8, 1].anchor_token_ids.data_ptr(),
        )
        self.assertEqual(batches[8, 0].block_indices.shape, (40, 1, 64))
        self.assertTrue(batches[8, 0].block_indices.eq(-1).all())

        runtime.reset_state(5)
        self.assertTrue(all(not window[5].any() for window in windows))
        self.assertTrue(all(window[4].all() for window in windows))
        self.assertFalse(anchors[5].any())
        self.assertTrue(anchors[4].all())
        with self.assertRaisesRegex(ValueError, "configured"):
            runtime.batch(7, 0)

    def test_prefix_snapshot_round_trip_copies_windows_and_anchor(self) -> None:
        windows = tuple(torch.zeros((3, 2, 4), dtype=torch.uint8) for _ in range(3))
        anchors = torch.zeros((3, 6), dtype=torch.bfloat16)
        snapshots = dspark.DeepSeekV4DSparkPrefixSnapshots(
            tuple(torch.zeros((1, 2, 4), dtype=torch.uint8) for _ in range(3)),
            torch.zeros((1, 6), dtype=torch.bfloat16),
        )
        runtime = dspark.DeepSeekV4DSparkRuntime(
            windows,
            anchors,
            {},
            {},
            object(),
            live_slots=2,
            prefix_snapshots=snapshots,
        )
        for index, window in enumerate(windows, 1):
            window[0].fill_(index)
        anchors[0].fill_(4)

        runtime.capture_prefix(0, 2)
        for window in windows:
            window[0].zero_()
        anchors[0].zero_()
        runtime.restore_prefix(2, 1)

        for index, window in enumerate(windows, 1):
            self.assertTrue(torch.all(window[1] == index))
        self.assertTrue(torch.all(anchors[1] == 4))
        with self.assertRaisesRegex(ValueError, "live state pool"):
            runtime.restore_prefix(2, 2)
        with self.assertRaisesRegex(ValueError, "snapshot pool"):
            runtime.restore_prefix(1, 0)


class DeepSeekV4DSparkOpsTest(unittest.TestCase):
    def test_seed_writes_only_accepted_rows_into_the_same_stage_windows(self) -> None:
        calls = []
        writer_calls = []
        ops_module = _load_ops_module(calls, writer_calls)
        ops = object.__new__(ops_module.DeepSeekV4DSparkOps)
        writes = []

        def writer(_q, _kv, output, cache, slots, *_args):
            writes.append(cache)
            output.zero_()
            for slot in slots.tolist():
                if slot >= 0:
                    cache[slot // deepseek_v4.DSPARK_PAGE_TOKENS, 0] = len(writes)

        ops._writer = writer
        stages = tuple(
            SimpleNamespace(stage_id=stage, attention=_projection())
            for stage in range(deepseek_v4.DSPARK_STAGE_COUNT)
        )
        weights = SimpleNamespace(
            stages=stages,
            main_projection=torch.empty((1, 1)),
            main_projection_scale=torch.empty((1, 1)),
            main_norm=torch.empty(4096),
        )
        model = dspark.DeepSeekV4DSparkModel(weights, object(), object(), ops)
        workspace = dspark._allocate_workspace(torch, "cpu", 1)
        windows = tuple(torch.zeros((1, 2, 37440), dtype=torch.uint8) for _ in stages)

        model.seed_windows(
            torch.empty((0, 12288)),
            torch.empty(0, dtype=torch.int32),
            torch.empty(0, dtype=torch.int32),
            torch.empty((2, 64)),
            windows,
            workspace,
        )
        self.assertFalse(writes)

        model.seed_windows(
            torch.empty((7, 12288), dtype=torch.bfloat16),
            torch.arange(7, dtype=torch.int32),
            torch.tensor([0, -1, -1, -1, -1, -1, -1], dtype=torch.int32),
            torch.empty((7, 64)),
            windows,
            workspace,
        )

        self.assertEqual(len(writes), 2 * deepseek_v4.DSPARK_STAGE_COUNT)
        for stage, window in enumerate(windows, 1):
            self.assertEqual(writes[stage - 1].data_ptr(), window.data_ptr())
            self.assertEqual(writes[stage + 2].data_ptr(), window.data_ptr())
            self.assertEqual(window[0, 0, 0].item(), stage)
            self.assertFalse(window[0, 1].any())

    def test_stage_executes_reused_projection_attention_and_moe_seam(self) -> None:
        calls = []
        writer_calls = []
        ops_module = _load_ops_module(calls, writer_calls)
        ops = object.__new__(ops_module.DeepSeekV4DSparkOps)
        ops._writer = ops_module._test_writer

        def decode_moe(hidden, _ids, _weights, _workspace, _buffer, _hash):
            calls.append("decode_moe")
            return torch.zeros_like(hidden)

        ops._core = SimpleNamespace(_decode_moe=decode_moe)
        workspace = dspark._allocate_workspace(torch, "cpu", 1)
        batch = dspark._allocate_batch(torch, "cpu", 1, torch.empty((6, 64)), object())
        runtime = dspark.DeepSeekV4DSparkRuntime(
            tuple(torch.zeros((1, 2, 37440), dtype=torch.uint8) for _ in range(3)),
            torch.zeros((1, 12_288), dtype=torch.bfloat16),
            {(1, 0): batch},
            {1: workspace},
            object(),
            1,
        )
        batch.anchor_token_ids.fill_(17)
        batch.anchor_positions.zero_()
        weights = SimpleNamespace(
            attention_mhc=object(),
            attention=_projection(),
            ffn_mhc=object(),
            ffn_norm=torch.empty(4096),
            ffn=object(),
        )
        streams = torch.empty((5, 4, 4096), dtype=torch.bfloat16)

        output = ops.stage(
            streams,
            weights,
            torch.empty((1, 4096), dtype=torch.bfloat16),
            batch,
            runtime.windows[0],
            workspace,
            runtime.mega_buffer,
        )

        self.assertIs(output, workspace.layer.streams_out)
        self.assertEqual(
            calls,
            [
                "mhc_pre",
                "project_common",
                "project_query",
                "quantize",
                "fp8_gemm",
                "rmsnorm",
                "writer",
                "writer",
                "attention",
                "project_output",
                "mhc_post",
                "mhc_pre",
                "rmsnorm",
                "decode_moe",
                "mhc_post",
            ],
        )
        self.assertEqual(
            writer_calls[0][3].untyped_storage()._cdata,
            runtime.windows[0].untyped_storage()._cdata,
        )
        self.assertIs(writer_calls[1][3], workspace.block_cache)
        self.assertEqual(
            writer_calls[0][2].untyped_storage()._cdata,
            workspace.query.untyped_storage()._cdata,
        )


if __name__ == "__main__":
    unittest.main()

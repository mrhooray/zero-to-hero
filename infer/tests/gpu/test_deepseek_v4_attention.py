import ast
import inspect
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

pytestmark = pytest.mark.gpu

import torch

from infer.models.deepseek_v4_flash import model as base
from infer.models.deepseek_v4_flash import attention as model
from infer.models.deepseek_v4_flash import checkpoint
from tools.kernels import build_deepseek_v4_flash_writer as writer_installer

OPS_PATH = Path(__file__).resolve().parents[2] / "src/infer/models/deepseek_v4_flash/ops/attention.py"
OPS_SOURCE = OPS_PATH.read_text()
OPS_TREE = ast.parse(OPS_SOURCE)
OPS_FUNCTIONS = {
    node.name: node
    for node in ast.walk(OPS_TREE)
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
}
CORE_PATH = Path(__file__).resolve().parents[2] / "src/infer/models/deepseek_v4_flash/ops/core.py"
CORE_FUNCTIONS = {
    node.name: node
    for node in ast.walk(ast.parse(CORE_PATH.read_text()))
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
}


def calls(function: ast.AST) -> list[ast.Call]:
    return [node for node in ast.walk(function) if isinstance(node, ast.Call)]


def call_name(call: ast.Call) -> str:
    def name(expression: ast.expr) -> str:
        if isinstance(expression, ast.Name):
            return expression.id
        if isinstance(expression, ast.Attribute):
            prefix = name(expression.value)
            return f"{prefix}.{expression.attr}" if prefix else expression.attr
        return ""

    return name(call.func)


class DeepSeekV4AttentionModelTest(unittest.TestCase):
    def test_head_aware_workspace_shapes_cover_all_graph_buckets(self) -> None:
        batch_sizes = (
            *base.DECODE_BATCH_SIZES,
            *(size * 6 for size in base.DECODE_BATCH_SIZES),
        )
        for batch_size in batch_sizes:
            aligned_batch_size = (batch_size + 3) // 4 * 4
            common = model._attention_workspace_shapes(
                batch_size, base.LOCAL_QUERY_HEADS
            )
            c4 = model._c4_workspace_shapes(batch_size, base.LOCAL_QUERY_HEADS)
            replicated = model._c4_workspace_shapes(batch_size, base.NUM_QUERY_HEADS)
            self.assertEqual(common.normalized, (batch_size, 4096))
            self.assertEqual(common.hidden_scale, (batch_size, 32))
            self.assertEqual(common.qkv_low_rank, (batch_size, 1536))
            self.assertEqual(common.q_residual_scale, (batch_size, 8))
            self.assertEqual(common.projected_q, (batch_size, 16, 512))
            self.assertEqual(common.output_a, (batch_size, 2, 1024))
            self.assertEqual(common.output_scale, (aligned_batch_size, 16))
            self.assertEqual(c4.packed_q, (batch_size, 16384))
            self.assertEqual(c4.index_weights, (batch_size, 64))
            self.assertEqual(replicated.common.projected_q, (batch_size, 64, 512))
            self.assertEqual(replicated.common.output_a, (batch_size, 8, 1024))
            self.assertEqual(replicated.common.output_fp8, (batch_size, 8192))
            self.assertEqual(
                replicated.common.output_scale,
                (aligned_batch_size, 64),
            )
            self.assertEqual(replicated.packed_q, (batch_size, 40960))
        with self.assertRaisesRegex(ValueError, "batch_size must be"):
            model._attention_workspace_shapes(7, base.LOCAL_QUERY_HEADS)

    def test_decode_layer_workspace_supports_all_layers(self) -> None:
        token_counts = (
            0,
            *base.DECODE_BATCH_SIZES,
            *(size * 6 for size in base.DECODE_BATCH_SIZES),
        )
        for layer_id in (0, 1, 2, 3, 4, 41, 42):
            for tokens in token_counts:
                workspace = model.decode_layer_workspace_shapes(layer_id, tokens)
                self.assertEqual(workspace.ffn.output, (tokens, 4096))
                self.assertEqual(workspace.comb, (tokens, 4, 4))
                self.assertEqual(workspace.streams_out, (tokens, 4, 4096))
        with self.assertRaisesRegex(ValueError, "decode bucket"):
            model.decode_layer_workspace_shapes(2, 7)

    def test_layer_workspaces_support_tensor_parallel_heads(self) -> None:
        for layer_id in (0, 2, 3):
            decode = model.decode_layer_workspace_shapes(
                layer_id, 6, query_heads=base.LOCAL_QUERY_HEADS
            )
            prefill = model.prefill_layer_workspace_shapes(
                layer_id, 4096, query_heads=base.LOCAL_QUERY_HEADS
            )
            for workspace, tokens in ((decode, 6), (prefill, 4096)):
                attention = workspace.attention
                common = attention.common if layer_id == 2 else attention
                self.assertEqual(
                    common.projected_q,
                    (tokens, base.LOCAL_QUERY_HEADS, base.HEAD_DIM),
                )
                self.assertEqual(common.output_a, (tokens, 2, 1024))
                self.assertEqual(workspace.ffn.shared_gate_up, (tokens, 1024))
                self.assertEqual(workspace.ffn.shared_activated_fp8, (tokens, 512))

    def test_mmap_loader_slices_both_tp_axes_exactly(self) -> None:
        values = tuple(float(value) for value in range(64))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.bin"
            path.write_bytes(struct.pack("64f", *values))
            storage = torch.from_file(
                str(path), shared=False, size=256, dtype=torch.uint8
            )
            for axis, rank, expected in (
                (0, 2, torch.arange(32, 48).view(2, 8)),
                (1, 1, torch.arange(64).view(8, 8)[:, 2:4]),
            ):
                spec = base.WeightSpec("weight", (8, 8), "F32", axis)
                actual = checkpoint._load_local_tensor(
                    storage, 0, spec, rank, "cpu", torch
                )
                torch.testing.assert_close(actual, expected.float())
                self.assertTrue(actual.is_contiguous())

    def test_dep4_projection_keeps_all_heads_and_output_groups(self) -> None:
        dtypes = {
            "BF16": torch.bfloat16,
            "F32": torch.float32,
            "F8_E4M3": torch.float8_e4m3fn,
            "F8_E8M0": torch.float8_e8m0fnu,
        }
        raw = {
            spec.name: torch.empty(spec.shape, dtype=dtypes[spec.dtype], device="meta")
            for spec in base.C4_ATTENTION_WEIGHTS
        }

        packed = model._pack_projection(raw, torch, pack_index=True)

        self.assertIs(packed.sink, raw["attn.attn_sink"])
        self.assertEqual(tuple(packed.q_b.shape), (40960, 1024))
        self.assertEqual(tuple(packed.q_b_scale.shape), (320, 8))
        self.assertEqual(tuple(packed.output_a.shape), (8, 1024, 4096))
        self.assertEqual(tuple(packed.output_a_scale.shape), (8, 1024, 8))
        self.assertEqual(packed.output_a_scale.dtype, torch.int32)
        self.assertEqual(tuple(packed.output_b.shape), (4096, 8192))

    def test_one_loader_covers_swa_csa_and_hca_placements(self) -> None:
        cases = (
            (0, base.COMPRESSED_ATTENTION_COMMON_WEIGHTS, "_pack_projection"),
            (2, base.C4_ATTENTION_WEIGHTS, "_pack_c4_attention"),
            (3, base.C128_ATTENTION_WEIGHTS, "_pack_c128_attention"),
        )
        for layer_id, specs, packer_name in cases:
            with self.subTest(layer_id=layer_id):
                view = Mock()
                view.load_target_tensor.side_effect = lambda name, *_args, **_kwargs: (
                    name
                )
                with patch.object(model, packer_name, return_value=layer_id) as packer:
                    actual = model.load_attention_weights(
                        view,
                        layer_id,
                        3,
                        "cuda:3",
                        tensor_parallel=layer_id != 0,
                    )

                self.assertEqual(actual, layer_id)
                raw = packer.call_args.args[0]
                self.assertEqual(tuple(raw), tuple(spec.name for spec in specs))
                self.assertEqual(
                    [call.args[:3] for call in view.load_target_tensor.call_args_list],
                    [
                        (spec.checkpoint_key_for_layer(layer_id), 3, "cuda:3")
                        for spec in specs
                    ],
                )
                self.assertTrue(
                    all(
                        call.kwargs["sharded"] == (layer_id != 0)
                        for call in view.load_target_tensor.call_args_list
                    )
                )

    def test_packing_keeps_exact_precision_boundaries(self) -> None:
        layer2 = inspect.getsource(model._pack_c4_attention)
        layer3 = inspect.getsource(model._pack_c128_attention)
        compressor = inspect.getsource(model._pack_compressor_weight)
        projection = inspect.getsource(model._pack_projection)
        loader = inspect.getsource(model.load_attention_weights)
        shard_reader = inspect.getsource(checkpoint._open_checkpoint_shard)
        local = inspect.getsource(checkpoint._load_local_tensor)
        self.assertEqual(layer2.count("_pack_compressor_weight("), 2)
        self.assertEqual(layer3.count("_pack_compressor_weight("), 1)
        self.assertIn("torch.cat((kv, score)).T.contiguous()", compressor)
        self.assertNotIn("float32", compressor)
        self.assertIn("pack_index=True", layer2)
        self.assertEqual(projection.count("torch.cat("), 4)
        self.assertIn(
            'torch.cat((raw["attn.wq_a.weight"], raw["attn.wkv.weight"]))',
            projection,
        )
        self.assertIn('torch.cat((q_b, raw["attn.indexer.wq_b.weight"]))', projection)
        self.assertNotIn(".float()", projection)
        self.assertIn('raw["attn.wo_a.weight"].reshape(-1, 1_024, 4_096)', projection)
        self.assertIn(".reshape(-1, 8, 32)", projection)
        self.assertIn("sink[:LOCAL_QUERY_HEADS].copy_(local_sink)", projection)
        self.assertIn("view.load_target_tensor(", loader)
        self.assertIn("sharded=tensor_parallel", loader)
        self.assertIn("torch.from_file(", shard_reader)
        self.assertNotIn("hashlib", shard_reader)
        self.assertNotIn("file_digest", shard_reader)
        self.assertIn(
            "tensor.chunk(deepseek_v4_flash.TEP_SIZE, dim=spec.shard_axis)[tep_rank]",
            local,
        )

        kv = torch.arange(24, dtype=torch.bfloat16).reshape(3, 8)
        score = kv + 32
        packed = model._pack_compressor_weight(kv, score, torch)
        self.assertEqual(packed.dtype, torch.bfloat16)
        self.assertTrue(packed.is_contiguous())
        self.assertTrue(torch.equal(packed, torch.cat((kv, score)).T.contiguous()))

        checkpoint_ape = torch.arange(32).view(4, 8)
        self.assertTrue(
            torch.equal(
                model._reorder_c4_ape(checkpoint_ape, torch),
                torch.cat(checkpoint_ape.chunk(2, dim=-1), dim=0).reshape(4, 8),
            )
        )


class DeepSeekV4AttentionOpsContractTest(unittest.TestCase):
    def test_decode_block_order_is_direct_and_empty_lanes_enter_megamoe(self) -> None:
        function = OPS_FUNCTIONS["decode_layer"]
        source = ast.unparse(function)
        ordered = (
            (
                "_validate_decode_layer(streams, token_ids, weights, "
                "attention_state, workspace, self._query_heads)"
            ),
            "if not attention_prepared:",
            "_mhc_pre(streams, weights.attention_mhc",
            "_rmsnorm(workspace.collapsed, _projection_weights",
            "attention = self._decode_",
            "if tokens in (1, 4, 16, 32):",
            "_mhc_post(attention, streams",
            "_mhc_pre(workspace.streams_mid, weights.ffn_mhc",
            "self._core._decode_moe",
            "if next_weights is None:",
            "_mhc_post(ffn, workspace.streams_mid",
        )
        offsets = [source.index(fragment) for fragment in ordered]
        self.assertEqual(offsets, sorted(offsets))
        self.assertIn("isinstance(attention_state, DeepSeekV4C4DecodeState)", source)
        self.assertIn("self._decode_c128_attention", source)
        self.assertIn("common = common.common", source)
        self.assertIn("self._process_group", source)
        self.assertNotIn("all_reduce", source)
        top_level_calls = [
            node.value
            for node in function.body
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
        ]
        self.assertTrue(
            any(
                isinstance(call.func, ast.Attribute) and call.func.attr == "_decode_moe"
                for call in top_level_calls
            )
        )
        validation = ast.unparse(OPS_FUNCTIONS["_validate_decode_layer"])
        self.assertIn("decode_layer_workspace_shapes", validation)
        self.assertIn("DeepSeekV4C4AttentionWeights", validation)
        self.assertIn("DeepSeekV4C128AttentionWeights", validation)
        self.assertIn("DeepSeekV4ProjectionWeights", validation)
        self.assertIn("weights.attention_mhc, weights.ffn_mhc", validation)

    def test_mhc_uses_caller_owned_deepgemm_and_flashinfer_outputs(self) -> None:
        source = ast.unparse(OPS_FUNCTIONS["_mhc_pre"])
        self.assertIn("deep_gemm.tf32_hc_prenorm_gemm", source)
        self.assertIn("splits = _mhc_pre_splits(rows, streams.device)", source)
        self.assertIn("num_splits=splits", source)
        self.assertIn("get_mhc_module().mhc_pre_big_fuse", source)
        self.assertIn("dot if splits > 1 else dot[0]", source)
        self.assertNotIn("torch.empty", source)

    def test_decode_fuses_b1_b4_b16_b32_mhc_across_both_layer_boundaries(
        self,
    ) -> None:
        decode = ast.unparse(OPS_FUNCTIONS["decode_layer"])
        prefill = ast.unparse(OPS_FUNCTIONS["prefill_layer"])

        self.assertIn("if tokens in (1, 4, 16, 32):", decode)
        self.assertEqual(decode.count("deepseek_v4_mhc_transition("), 2)
        self.assertIn("if not attention_prepared:", decode)
        self.assertIn("if next_weights is None:", decode)
        self.assertIn("else:\n            _mhc_post(attention, streams", decode)
        self.assertEqual(decode.count("(common.normalized, weights.attention"), 3)
        self.assertNotIn("deepseek_v4_mhc_transition", prefill)

        for path in (
            "_decode_swa_attention",
            "_decode_c4_attention",
            "_project_c128",
        ):
            self.assertNotIn("_project_input(", ast.unparse(OPS_FUNCTIONS[path]))

    def test_decode_paths_are_direct_and_keep_writer_v2_padding_invariant(self) -> None:
        swa = ast.unparse(OPS_FUNCTIONS["_decode_swa_attention"])
        c4 = ast.unparse(OPS_FUNCTIONS["_decode_c4_attention"])
        c128 = ast.unparse(OPS_FUNCTIONS["_decode_c128_attention"])
        c128_projection = ast.unparse(OPS_FUNCTIONS["_project_c128"])
        prefill_layer = ast.unparse(OPS_FUNCTIONS["prefill_layer"])
        c4_phases = (
            "validate_c4",
            "compress_c4_main",
            "compress_c4_index",
            "prepare_c4",
            "select_c4",
            "attend_c4",
        )
        ops_class = next(
            node
            for node in OPS_TREE.body
            if isinstance(node, ast.ClassDef) and node.name == "DeepSeekV4AttentionOps"
        )
        init = ast.unparse(
            next(
                node
                for node in ops_class.body
                if isinstance(node, ast.FunctionDef) and node.name == "__init__"
            )
        )
        self.assertEqual(
            [
                node.name
                for node in ops_class.body
                if isinstance(node, ast.FunctionDef) and not node.name.startswith("_")
            ],
            [
                "prefill_c4_attention",
                "prefill_layer",
                "decode_layer",
            ],
        )
        for phase in c4_phases:
            self.assertEqual(CORE_FUNCTIONS[phase].args.defaults, [])
        self.assertEqual(CORE_FUNCTIONS["decode_c128"].args.defaults, [])
        self.assertEqual(CORE_FUNCTIONS["_attention"].args.defaults, [])
        self.assertIn("self._core.decode_swa", swa)
        for phase in c4_phases:
            self.assertIn(f"self._core.{phase}", c4)
        self.assertIn("self._core.decode_c128", c128)
        self.assertEqual(swa.count("torch.mm("), 0)
        self.assertEqual(c4.count("torch.mm("), 1)
        self.assertEqual(c4.count("_project_compressor("), 2)
        self.assertEqual(c128_projection.count("_project_compressor("), 1)
        self.assertEqual(prefill_layer.count("_project_compressor("), 3)
        compressor = ast.unparse(OPS_FUNCTIONS["_project_compressor"])
        self.assertIn("out=out", compressor)
        self.assertIn("out_dtype=torch.float32", compressor)
        self.assertEqual(swa.count("_project_query("), 1)
        self.assertEqual(c128.count("self._project_c128("), 1)
        self.assertEqual(c4.count("_project_query("), 1)
        self.assertEqual(c4.count("_copy_c4_raw_query("), 1)
        self.assertEqual(c128_projection.count("_project_query("), 1)
        self.assertIn("current.wait_event(self._compressor_done)", c4)
        self.assertIn("current.wait_event(self._index_done)", c4)
        self.assertIn("current.wait_event(self._compressor_done)", c128_projection)
        self.assertEqual(init.count("torch.cuda.Stream()"), 2)
        self.assertEqual(init.count("torch.cuda.Event()"), 4)
        self.assertEqual(prefill_layer.count("_project_common("), 3)
        self.assertEqual(prefill_layer.count("self.prefill_c4_attention("), 1)
        self.assertNotIn("self._prefill_c128_attention(", prefill_layer)
        self.assertNotIn("_projection_stream", prefill_layer)
        self.assertNotIn("zero_", swa + c4 + c128)
        self.assertNotIn("raw.query.copy_", swa + c4 + c128)
        writer = OPS_FUNCTIONS["_write_raw"]
        writer_call = next(
            call for call in calls(writer) if call_name(call) == "self._writer"
        )
        self.assertEqual(len(writer_call.args), 8)
        self.assertEqual(ast.unparse(writer_call.args[0]), "workspace.projected_q")
        self.assertEqual(ast.unparse(writer_call.args[2]), "raw.query")
        self.assertEqual(ast.unparse(writer_call.args[4]), "raw_slots")

        c4_prefill = OPS_FUNCTIONS["prefill_c4_attention"]
        c4_prefill_source = ast.unparse(c4_prefill)
        self.assertLess(
            c4_prefill_source.index("self._core.prefill_compress_c4"),
            c4_prefill_source.index("self._core.prefill_c4_candidates"),
        )
        self.assertLess(
            c4_prefill_source.index("self._core.prefill_c4_candidates"),
            c4_prefill_source.index("self._prefill_attention"),
        )
        prefill = OPS_FUNCTIONS["_prefill_attention"]
        prefill_calls = [call_name(call) for call in calls(prefill)]
        self.assertLess(
            prefill_calls.index("self._core.stage_prefill_attention"),
            prefill_calls.index("self._writer"),
        )
        self.assertLess(
            prefill_calls.index("self._writer"),
            prefill_calls.index("self._core.prefill_selected_attention"),
        )
        prefill_writer = next(
            call for call in calls(prefill) if call_name(call) == "self._writer"
        )
        self.assertEqual(ast.unparse(prefill_writer.args[4]), "workspace.raw_slots")
        self.assertIn(
            ")[:, :self._query_heads]",
            ast.unparse(OPS_FUNCTIONS["_prefill_attention"]),
        )
        prefill_layer = ast.unparse(OPS_FUNCTIONS["prefill_layer"])
        self.assertEqual(prefill_layer.count("self._process_group"), 2)

    def test_output_uses_layout_invariant_inverse_rope_fp8_gemms_and_one_fp32_reduce(
        self,
    ) -> None:
        output = OPS_FUNCTIONS["_project_output"]
        source = ast.unparse(output)
        self.assertIn("BATCH_STRIDE=attention.stride(0)", source)
        self.assertIn("HEAD_STRIDE=attention.stride(-2)", source)
        loops = [node for node in ast.walk(output) if isinstance(node, ast.For)]
        self.assertEqual(loops, [])
        self.assertNotIn("torch.mm(", source)
        self.assertEqual(source.count("_fp8_gemm("), 1)
        self.assertIn("attention.shape[0] * attention.shape[-2]", source)
        self.assertIn("_power2_quantize_packed_groups", source)
        self.assertIn("deep_gemm.fp8_einsum", source)
        self.assertIn("_quantize(workspace.output_a.view(tokens, -1)", source)
        self.assertIn(
            "INPUT_ROW_STRIDE=input_.stride(0)",
            ast.unparse(CORE_FUNCTIONS["_quantize"]),
        )
        self.assertIn("HEADS=attention.shape[-2]", source)
        self.assertIn("workspace.hidden_fp32.copy_(workspace.output)", source)
        self.assertIn("torch.distributed.all_reduce(workspace.hidden_fp32", source)
        self.assertIn("workspace.output.copy_(workspace.hidden_fp32)", source)
        self.assertIn("if process_group is not None", source)
        self.assertEqual(OPS_SOURCE.count(".all_reduce("), 1)

    def test_flashinfer_gemm_uses_trtllm_mn_scale_layout(self) -> None:
        gemm = next(
            call
            for call in calls(CORE_FUNCTIONS["_fp8_gemm"])
            if call_name(call) == "gemm_fp8_nt_groupwise"
        )
        keywords = {
            keyword.arg: ast.unparse(keyword.value) for keyword in gemm.keywords
        }
        self.assertEqual(
            ast.unparse(gemm.args[2]),
            "input_scale.view(input_scale.shape[1], input_scale.shape[0]).T",
        )
        self.assertEqual(ast.unparse(gemm.args[3]), "weight_scale.T")
        self.assertEqual(keywords["scale_major_mode"], "None")
        self.assertNotIn("mma_sm", keywords)
        self.assertEqual(
            keywords["scale_granularity_mnk"],
            "(1, FP8_BLOCK_SIZE, FP8_BLOCK_SIZE)",
        )
        self.assertEqual(keywords["backend"], "'trtllm'")
        for function, destination in (
            (CORE_FUNCTIONS["_power2_quantize"], "scale_ptr"),
            (CORE_FUNCTIONS["_shared_swiglu_quantize"], "output_scale"),
        ):
            self.assertIn(f"{destination} + block * ROWS + row", ast.unparse(function))
        grouped_quantize = ast.unparse(OPS_FUNCTIONS["_power2_quantize_packed_groups"])
        self.assertIn(
            "(group * NUM_PACKS + pack) * ALIGNED_TOKENS + token",
            grouped_quantize,
        )
        gemm_validation = ast.unparse(CORE_FUNCTIONS["_fp8_gemm"])
        self.assertIn(
            "_require_tensor(weight, torch.float8_e4m3fn, "
            "(out.shape[1], input_.shape[1]), input_.device)",
            gemm_validation,
        )
        self.assertIn("_require_tensor(weight_scale, torch.float32", gemm_validation)
        validation = ast.unparse(OPS_FUNCTIONS["_validate_common"])
        self.assertIn("batch_size * DSPARK_VERIFY_WIDTH", validation)
        self.assertIn("get_world_size(group) != TEP_SIZE", validation)
        self.assertIn(
            "_require_tensor(weights.index_weights_t, torch.bfloat16, "
            "(4096, 64), device)",
            ast.unparse(OPS_FUNCTIONS["_validate_c4_bindings"]),
        )

    def test_c4_prep_preserves_official_rope_hadamard_and_mxfp4_boundaries(
        self,
    ) -> None:
        source = ast.unparse(CORE_FUNCTIONS["_c4_query"])
        self.assertIn("rotated.to(tl.bfloat16).to(tl.float32)", source)
        self.assertIn("* 128 ** (-0.5)", source)
        self.assertIn("hadamard.to(tl.bfloat16).to(tl.float32)", source)
        self.assertIn("tl.log2(maximum / 6.0)", source)
        self.assertIn("_mxfp4_nibble", source)
        self.assertIn("tl.pointer_type(tl.uint8)", source)
        self.assertIn("token * PACKED_WIDTH", source)
        self.assertIn(
            "tl.load(score_weights + row).to(tl.float32) * WEIGHT_SCALE", source
        )

    def test_c4_decode_prep_uses_the_qualified_butterfly_kernel(self) -> None:
        decode = ast.unparse(OPS_FUNCTIONS["_decode_c4_attention"])
        self.assertIn("_c4_decode_query[hidden.shape[0] * NUM_QUERY_HEADS,]", decode)
        self.assertNotIn("_c4_query", decode)

        source = ast.unparse(CORE_FUNCTIONS["_c4_decode_query"])
        self.assertEqual(source.count("_fht_stage"), 7)
        for stride in (1, 2, 4, 8, 16, 32, 64):
            self.assertIn(f"_fht_stage(values, dim, {stride})", source)
        self.assertIn("values.to(tl.bfloat16).to(tl.float32)", source)
        self.assertIn("values * 128 ** (-0.5)", source)
        self.assertIn("tl.reshape(hadamard, (4, 32))", source)
        self.assertIn("tl.log2(maximum / 6.0)", source)
        self.assertIn("_mxfp4_nibble", source)
        self.assertIn("tl.pointer_type(tl.uint8)", source)
        self.assertIn(
            "tl.load(score_weights + row).to(tl.float32) * WEIGHT_SCALE", source
        )

    def test_triton_kernels_do_not_capture_plain_integer_globals(self) -> None:
        kernels = {
            node.name: node
            for node in OPS_TREE.body
            if isinstance(node, ast.FunctionDef)
            and any(
                ast.unparse(decorator) == "triton.jit"
                for decorator in node.decorator_list
            )
        }
        self.assertEqual(
            set(kernels),
            {"_power2_quantize_packed_groups", "_inverse_rope"},
        )
        for name, kernel in kernels.items():
            arguments = {argument.arg for argument in kernel.args.args}
            captured = {
                node.id
                for node in ast.walk(kernel)
                if isinstance(node, ast.Name)
                and isinstance(node.ctx, ast.Load)
                and node.id.isupper()
                and node.id not in arguments
            }
            self.assertEqual(captured, set(), name)

    def test_writer_binary_identity_is_exact(self) -> None:
        constants = {
            node.targets[0].id: ast.literal_eval(node.value)
            for node in OPS_TREE.body
            if isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id
            in {"WRITER_PACKAGE", "WRITER_VERSION", "WRITER_SHA256", "WRITER_SIZE"}
        }
        self.assertEqual(constants["WRITER_PACKAGE"], "infer-deepseek-v4-writer")
        self.assertEqual(constants["WRITER_VERSION"], "0.3.0+f17b03ef")
        self.assertEqual(
            constants["WRITER_SIZE"], writer_installer.QUALIFIED_AOT_EXTENSION_SIZE
        )
        self.assertEqual(
            constants["WRITER_SHA256"],
            writer_installer.QUALIFIED_AOT_EXTENSION_SHA256,
        )


if __name__ == "__main__":
    unittest.main()

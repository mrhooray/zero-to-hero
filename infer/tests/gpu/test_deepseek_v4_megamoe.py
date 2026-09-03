import ast
import sys
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.gpu

import torch

from infer.models.deepseek_v4_flash import model as base
from infer.models.deepseek_v4_flash import checkpoint
from infer.models.deepseek_v4_flash import megamoe as model

OPS_PATH = Path(__file__).resolve().parents[2] / "src/infer/models/deepseek_v4_flash/ops/core.py"
OPS_TREE = ast.parse(OPS_PATH.read_text())
OPS_FUNCTIONS = {
    node.name: node
    for node in ast.walk(OPS_TREE)
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
}


def call_name(call):
    if isinstance(call, ast.Name):
        return call.id
    if isinstance(call, ast.Attribute):
        prefix = call_name(call.value)
        return f"{prefix}.{call.attr}" if prefix else call.attr
    return ""


class FakeDeepGemm:
    def __init__(self) -> None:
        self.calls = []

    def transform_weights_for_mega_moe(self, l1, l2):
        self.calls.append((l1, l2))
        return l1, l2


class FakeCheckpointView:
    def __init__(self) -> None:
        self.calls = []

    def load_target_tensor(self, name, rank, device, *, sharded):
        self.calls.append((name, rank, device, sharded))
        local_name = name.split(".", 2)[2]
        if local_name.startswith("ffn.experts."):
            spec_name = local_name.split(".", 3)[3]
            specs = {spec.name: spec for spec in base.ROUTED_EXPERT_TEMPLATE}
            spec = specs[spec_name]
        else:
            specs = {
                spec.name: spec
                for spec in (
                    *model._MOE_NON_EXPERT_TEMPLATE,
                    model._HASH_ROUTING_SPEC,
                    model._LEARNED_ROUTING_SPEC,
                )
            }
            spec = specs[local_name]
        dtype = {
            "BF16": torch.bfloat16,
            "F32": torch.float32,
            "F8_E4M3": torch.float8_e4m3fn,
            "F8_E8M0": torch.float8_e8m0fnu,
            "I8": torch.int8,
            "I64": torch.int64,
        }[spec.dtype]
        shape = spec.local_shape() if sharded else spec.shape
        return torch.empty(shape, dtype=dtype, device=device)


class FakeDSparkCheckpointView:
    def __init__(self) -> None:
        self.calls = []

    def load_dspark_tensor(self, name, rank, device):
        self.calls.append((name, rank, device))
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


class DeepSeekV4MegaMoEModelTest(unittest.TestCase):
    def test_loads_dspark_learned_router_and_local_experts(self) -> None:
        view = FakeDSparkCheckpointView()
        with patch.dict(sys.modules, {"deep_gemm": FakeDeepGemm()}):
            weights = model.load_dep4_dspark_moe_weights(view, 1, 2, "meta")

        expected = {f"mtp.1.{spec.name}" for spec in model._MOE_NON_EXPERT_TEMPLATE}
        expected.add("mtp.1.ffn.gate.bias")
        expected.update(
            f"mtp.1.ffn.experts.{expert_id}.{spec.name}"
            for expert_id in model.local_routed_expert_ids(2)
            for spec in base.ROUTED_EXPERT_TEMPLATE
        )
        self.assertEqual({name for name, _, _ in view.calls}, expected)
        self.assertTrue(all(call[1:] == (2, "meta") for call in view.calls))
        self.assertEqual(
            (weights.routing.shape, weights.routing.dtype), ((256,), torch.float32)
        )

    def test_loads_hash_and_learned_moe_with_exact_rank_ownership(
        self,
    ) -> None:
        cases = (
            (0, 0, "ffn.gate.tid2eid"),
            (2, 3, "ffn.gate.tid2eid"),
            (3, 0, "ffn.gate.bias"),
            (42, 3, "ffn.gate.bias"),
        )
        for layer_id, rank, routing_name in cases:
            with self.subTest(layer_id=layer_id, rank=rank):
                view = FakeCheckpointView()
                deep_gemm = FakeDeepGemm()
                with (
                    patch.dict(sys.modules, {"deep_gemm": deep_gemm}),
                    patch.object(
                        model, "_validate_hash_routes_tensor"
                    ) as validate_routes,
                ):
                    weights = model.load_dep4_moe_weights(view, layer_id, rank, "meta")

                expert_ids = model.local_routed_expert_ids(rank)
                expected = {
                    f"layers.{layer_id}.{spec.name}"
                    for spec in model._MOE_NON_EXPERT_TEMPLATE
                }
                expected.add(f"layers.{layer_id}.{routing_name}")
                expected.update(
                    f"layers.{layer_id}.ffn.experts.{expert_id}.{spec.name}"
                    for expert_id in expert_ids
                    for spec in base.ROUTED_EXPERT_TEMPLATE
                )
                self.assertEqual(len(view.calls), len(expected))
                self.assertEqual({call[0] for call in view.calls}, expected)
                self.assertTrue(
                    all(call[1:] == (rank, "meta", False) for call in view.calls)
                )
                self.assertEqual(validate_routes.call_count, int(layer_id < 3))
                routing_shape = (129280, 6) if layer_id < 3 else (256,)
                routing_dtype = torch.int32 if layer_id < 3 else torch.float32
                self.assertEqual(
                    (tuple(weights.routing.shape), weights.routing.dtype),
                    (routing_shape, routing_dtype),
                )
                self.assertEqual(
                    (tuple(weights.router_t.shape), weights.router_t.dtype),
                    ((4096, 256), torch.bfloat16),
                )
                self.assertEqual(
                    (tuple(weights.shared_gate_up.shape), weights.shared_gate_up.dtype),
                    ((4096, 4096), torch.float8_e4m3fn),
                )
                self.assertEqual(tuple(weights.shared_gate_up_scale.shape), (32, 32))
                self.assertEqual(tuple(weights.shared_down.shape), (4096, 2048))
                self.assertEqual(tuple(weights.shared_down_scale.shape), (32, 16))

    def test_moe_packer_keeps_shared_gate_before_up(self) -> None:
        specs = (
            base.WeightSpec("ffn.gate.tid2eid", (8, 2), "I64"),
            base.WeightSpec("ffn.gate.weight", (4, 8), "BF16"),
            base.WeightSpec("ffn.shared_experts.w1.scale", (2, 1), "F8_E8M0"),
            base.WeightSpec("ffn.shared_experts.w2.scale", (1, 2), "F8_E8M0"),
            base.WeightSpec("ffn.shared_experts.w3.scale", (2, 1), "F8_E8M0"),
            base.WeightSpec("ffn.shared_experts.w1.weight", (2, 8), "F8_E4M3"),
            base.WeightSpec("ffn.shared_experts.w2.weight", (8, 2), "F8_E4M3"),
            base.WeightSpec("ffn.shared_experts.w3.weight", (2, 8), "F8_E4M3"),
        )
        markers = {
            "ffn.gate.tid2eid": 0,
            "ffn.gate.weight": 4,
            "ffn.shared_experts.w1.scale": 1,
            "ffn.shared_experts.w2.scale": 2,
            "ffn.shared_experts.w3.scale": 4,
            "ffn.shared_experts.w1.weight": 11,
            "ffn.shared_experts.w2.weight": 22,
            "ffn.shared_experts.w3.weight": 33,
        }

        def load(spec, _expert_id):
            dtype = {
                "BF16": torch.bfloat16,
                "F8_E4M3": torch.float8_e4m3fn,
                "F8_E8M0": torch.float8_e8m0fnu,
                "I64": torch.int64,
            }[spec.dtype]
            return torch.full(spec.shape, markers[spec.name], dtype=dtype)

        routed = model.DeepSeekV4MegaMoEWeights(l1=(1, 2), l2=(3, 4))
        with (
            patch.object(model, "_MOE_NON_EXPERT_TEMPLATE", specs[1:]),
            patch.object(model, "_HASH_ROUTING_SPEC", specs[0]),
            patch.object(base, "LOCAL_EXPERTS", 0),
            patch.object(model, "_pack_megamoe_weights", return_value=routed),
            patch.object(model, "_validate_hash_routes_tensor"),
        ):
            weights = model._pack_moe_weights(
                load,
                0,
                "cpu",
                torch,
                FakeDeepGemm(),
                hash_routing=True,
                tensor_parallel=False,
            )

        self.assertTrue(torch.all(weights.shared_gate_up[:2] == 11))
        self.assertTrue(torch.all(weights.shared_gate_up[2:] == 33))
        self.assertTrue(torch.all(weights.shared_gate_up_scale[:2] == 1))
        self.assertTrue(torch.all(weights.shared_gate_up_scale[2:] == 4))
        self.assertTrue(torch.all(weights.shared_down == 22))
        self.assertIs(weights.routed, routed)

    def test_moe_workspaces_cover_zero_and_uneven_lanes(self) -> None:
        counts = (0, 1, 3, 7)
        shapes = tuple(model.moe_workspace_shapes(m) for m in counts)
        self.assertEqual(tuple(shape.output[0] for shape in shapes), counts)
        shapes = shapes[-1]
        self.assertEqual(shapes.router_logits, (7, 256))
        self.assertEqual(shapes.topk_ids, (7, 6))
        self.assertEqual(shapes.topk_weights, (7, 6))
        self.assertEqual(shapes.hidden_fp8, (7, 4096))
        self.assertEqual(shapes.hidden_scale, (7, 32))
        self.assertEqual(shapes.shared_gate_up, (7, 4096))
        self.assertEqual(shapes.shared_activated_fp8, (7, 2048))
        self.assertEqual(shapes.shared_activated_scale, (7, 16))
        self.assertEqual(shapes.routed, (7, 4096))
        self.assertEqual(shapes.output, (7, 4096))
        self.assertFalse(hasattr(shapes, "expert_recv_stats"))
        with self.assertRaisesRegex(ValueError, "token_count"):
            model.moe_workspace_shapes(-1)

        tep4 = model.tep4_moe_workspace_shapes(7)
        self.assertEqual(tep4.shared_gate_up, (7, 1024))
        self.assertEqual(tep4.shared_activated_fp8, (7, 512))
        self.assertEqual(tep4.shared_activated_scale, (7, 4))

    def test_validates_ep_rank_ownership(self) -> None:
        self.assertEqual((base.DEP_SIZE, base.EP_SIZE), (4, 4))
        ownership = tuple(model.local_routed_expert_ids(rank) for rank in range(4))
        self.assertEqual(ownership[0], tuple(range(64)))
        self.assertEqual(ownership[3], tuple(range(192, 256)))
        self.assertEqual(
            tuple(expert_id for rank in ownership for expert_id in rank),
            tuple(range(256)),
        )
        with self.assertRaisesRegex(ValueError, "ep_rank"):
            model.local_routed_expert_ids(4)

    def test_packs_native_bytes_once_in_megamoe_layout(self) -> None:
        calls = Counter()

        def load(spec, expert_id):
            calls[(expert_id, spec.name)] += 1
            dtype = {
                "F8_E8M0": torch.float8_e8m0fnu,
                "I8": torch.int8,
            }[spec.dtype]
            return torch.empty(spec.shape, dtype=dtype, device="meta")

        deep_gemm = FakeDeepGemm()
        weights = model._pack_megamoe_weights(load, 2, "meta", torch, deep_gemm)

        self.assertEqual(len(calls), 64 * 6)
        self.assertEqual(set(calls.values()), {1})
        self.assertEqual({expert_id for expert_id, _ in calls}, set(range(128, 192)))
        self.assertEqual(len(deep_gemm.calls), 1)
        l1, l2 = deep_gemm.calls[0]
        self.assertEqual(
            (tuple(l1[0].shape), l1[0].dtype),
            ((64, 4096, 2048), torch.int8),
        )
        self.assertEqual(
            (tuple(l1[1].shape), l1[1].dtype, tuple(l1[1].stride())),
            ((64, 4096, 32), torch.int32, (131072, 1, 4096)),
        )
        self.assertEqual(
            (tuple(l2[0].shape), l2[0].dtype),
            ((64, 4096, 1024), torch.int8),
        )
        self.assertEqual(
            (tuple(l2[1].shape), l2[1].dtype, tuple(l2[1].stride())),
            ((64, 4096, 16), torch.int32, (65536, 1, 4096)),
        )
        self.assertIs(weights.l1, l1)
        self.assertIs(weights.l2, l2)

    def test_scale_packing_is_a_lossless_four_byte_view(self) -> None:
        raw_bytes = torch.tensor(
            [[120, 121, 122, 123, 124, 125, 126, 127]], dtype=torch.uint8
        )
        raw = raw_bytes.view(torch.float8_e8m0fnu)
        target = torch.empty_strided((1, 2), (1, 1), dtype=torch.int32)

        model._copy_packed_scales(target, raw, torch)

        self.assertEqual(
            target.view(torch.uint8).reshape_as(raw_bytes).tolist(),
            raw_bytes.tolist(),
        )

    def test_packing_keeps_w1_gate_before_w3_up(self) -> None:
        template = (
            base.WeightSpec("w1.scale", (128, 4), "F8_E8M0"),
            base.WeightSpec("w2.scale", (128, 4), "F8_E8M0"),
            base.WeightSpec("w3.scale", (128, 4), "F8_E8M0"),
            base.WeightSpec("w1.weight", (128, 64), "I8"),
            base.WeightSpec("w2.weight", (128, 64), "I8"),
            base.WeightSpec("w3.weight", (128, 64), "I8"),
        )
        markers = {
            "w1.weight": 11,
            "w2.weight": 22,
            "w3.weight": 33,
            "w1.scale": 120,
            "w2.scale": 121,
            "w3.scale": 122,
        }

        def load(spec, _expert_id):
            value = markers[spec.name]
            tensor = torch.full(spec.shape, value, dtype=torch.int8)
            if spec.dtype == "F8_E8M0":
                tensor = tensor.view(torch.uint8).view(torch.float8_e8m0fnu)
            return tensor

        with (
            patch.object(base, "HIDDEN_SIZE", 128),
            patch.object(base, "EXPERT_INTERMEDIATE_SIZE", 128),
            patch.object(base, "LOCAL_EXPERTS", 2),
            patch.object(base, "NUM_ROUTED_EXPERTS", 8),
            patch.object(base, "ROUTED_EXPERT_TEMPLATE", template),
        ):
            weights = model._pack_megamoe_weights(load, 1, "cpu", torch, FakeDeepGemm())

        self.assertTrue(torch.all(weights.l1[0][:, :128] == 11))
        self.assertTrue(torch.all(weights.l1[0][:, 128:] == 33))
        self.assertTrue(torch.all(weights.l2[0] == 22))

        def packed(value):
            return int.from_bytes(bytes([value] * 4), "little", signed=True)

        self.assertTrue(torch.all(weights.l1[1][:, :128] == packed(120)))
        self.assertTrue(torch.all(weights.l1[1][:, 128:] == packed(122)))
        self.assertTrue(torch.all(weights.l2[1] == packed(121)))


class DeepSeekV4MegaMoEOpsContractTest(unittest.TestCase):
    def test_moe_limits_tp_collective_to_the_shared_expert(self) -> None:
        function = OPS_FUNCTIONS["_decode_moe"]
        names = [
            call_name(node.func)
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
        ]
        for name in (
            "_route_hash",
            "_route_learned",
            "_run_shared_expert",
            "deep_gemm.mega_moe_pre_dispatch",
            "deep_gemm.fp8_fp4_mega_moe",
        ):
            self.assertIn(name, names)
        source = ast.unparse(function)
        self.assertNotIn("all_reduce", source)
        self.assertIn("process_group", source)
        self.assertIn("w.output.add_(w.routed)", source)
        self.assertIn("activation_clamp=10.0", source)
        self.assertIn("out_dtype=torch.float32", source)
        self.assertNotIn("fp32_scratch", source)
        self.assertNotIn("cumulative_local_expert_recv_stats", source)

    def test_moe_overlap_is_private_persistent_and_ordered(self) -> None:
        function = OPS_FUNCTIONS["_decode_moe"]
        source = ast.unparse(function)
        module_source = OPS_PATH.read_text()

        self.assertNotIn("decode_moe", OPS_FUNCTIONS)
        self.assertIn(
            "self._moe_overlap: tuple[object, object, object] | None = None",
            module_source,
        )
        self.assertIn("if overlap is None:", source)
        self.assertIn("self._moe_overlap = overlap", source)
        self.assertEqual(source.count("torch.cuda.Stream()"), 1)
        self.assertEqual(source.count("torch.cuda.Event()"), 2)
        ordering = (
            "fork_event.record(current_stream)",
            "shared_stream.wait_event(fork_event)",
            "_run_shared_expert(hidden, weights, w, process_group)",
            "deep_gemm.mega_moe_pre_dispatch(",
            "deep_gemm.fp8_fp4_mega_moe(",
            "join_event.record(shared_stream)",
            "current_stream.wait_event(join_event)",
            "w.output.add_(w.routed)",
        )
        positions = [source.index(item) for item in ordering]
        self.assertEqual(positions, sorted(positions))

    def test_hash_route_kernel_preserves_table_order_and_normalizes_top6(self) -> None:
        source = ast.unparse(OPS_FUNCTIONS["_hash_route"])
        self.assertIn("hash_routes + token_id * TOP_K + slot", source)
        self.assertIn("router_logits + row * NUM_EXPERTS + expert_id", source)
        self.assertIn("tl.sqrt", source)
        self.assertIn("tl.sum", source)
        self.assertIn("ROUTE_SCALE", source)
        self.assertIn("tl.store(topk_ids", source)
        self.assertIn("tl.store(topk_weights", source)

    def test_learned_route_selects_with_bias_then_normalizes_unbiased_scores(
        self,
    ) -> None:
        wrapper = ast.unparse(OPS_FUNCTIONS["_route_learned"])
        self.assertIn("_learned_route", wrapper)
        self.assertIn("BLOCK_K=8", wrapper)
        source = ast.unparse(OPS_FUNCTIONS["_learned_route"])
        self.assertIn("scores + tl.load(route_bias + expert)", source)
        self.assertIn("tl.topk(packed, BLOCK_K", source)
        self.assertIn("slot < TOP_K", source)
        self.assertIn("tl.where(active, selected_weights, 0.0)", source)
        self.assertIn("NUM_EXPERTS - expert", source)
        self.assertIn("selected_logits = tl.load(router_logits", source)
        self.assertIn("tl.sum(selected_weights", source)
        self.assertIn("ROUTE_SCALE", source)
        self.assertIn("tl.store(topk_ids", source)
        self.assertIn("tl.store(topk_weights", source)

    def test_shared_expert_keeps_fp8_bf16_clamp_and_tp_boundaries(self) -> None:
        shared = ast.unparse(OPS_FUNCTIONS["_run_shared_expert"])
        self.assertEqual(shared.count("_fp8_gemm("), 2)
        self.assertIn("weights.shared_gate_up", shared)
        self.assertIn("weights.shared_down", shared)
        self.assertIn("torch.distributed.all_reduce", shared)
        kernel = ast.unparse(OPS_FUNCTIONS["_shared_swiglu_quantize"])
        self.assertIn("tl.minimum(gate, LIMIT)", kernel)
        self.assertIn("tl.maximum(tl.minimum(up, LIMIT), -LIMIT)", kernel)
        self.assertIn("to(tl.bfloat16).to(tl.float32)", kernel)
        self.assertIn("tl.store(output_scale", kernel)


if __name__ == "__main__":
    unittest.main()

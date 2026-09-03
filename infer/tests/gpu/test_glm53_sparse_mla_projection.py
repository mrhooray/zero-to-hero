import ast
import unittest
from pathlib import Path

import pytest

import torch

from infer.models.glm53_flash import checkpoint as glm53_checkpoint
from infer.models.glm53_flash import model as glm53_flash

pytestmark = pytest.mark.gpu

ROOT = Path(__file__).resolve().parents[2]
OPS_SOURCE = (ROOT / "src/infer/models/glm53_flash/ops/core.py").read_text()


class SparseMLAProjectionModelTest(unittest.TestCase):
    def test_workspace_and_score_contract(self) -> None:
        width = glm53_flash.GLM53_TARGET_VERIFY_WIDTH
        self.assertEqual(
            glm53_flash.SPARSE_MLA_PROJECTION_BATCH_SIZES,
            (
                *glm53_flash.GLM53_TARGET_DECODE_BATCH_SIZES,
                glm53_flash.GLM53_TARGET_VERIFY_WIDTH,
                glm53_flash.SPARSE_MLA_HISTORY_BLOCK_TOKENS,
                glm53_flash.KDA_CHUNK_SIZE,
            ),
        )
        self.assertEqual(glm53_flash.SPARSE_MLA_INDEX_SCORE_SCALE, 128**-0.5)
        self.assertEqual(glm53_flash.SPARSE_MLA_INDEX_WEIGHT_SCALE, 32**-0.5)
        self.assertEqual(glm53_flash.SPARSE_MLA_LAYER_NORM_EPS, 1e-6)
        self.assertIn(
            "one independent instance per parity",
            glm53_flash.SparseMLAProjectionWorkspace.__doc__,
        )
        self.assertEqual(
            glm53_flash.sparse_mla_projection_workspace_shapes(2).latent,
            (2, 512),
        )
        self.assertEqual(
            glm53_flash.sparse_mla_projection_workspace_shapes(width),
            glm53_flash.SparseMLAProjectionWorkspace(
                hidden_fp8=(width, 4096),
                hidden_scale=(width, 32),
                low_rank=(width, 2048),
                q_resid=(width, 1536),
                latent=(width, 512),
                q_resid_fp8=(width, 1536),
                q_resid_scale=(width, 12),
                main_q=(width, 4096),
                index_q=(width, 4096),
                index_prep=(width, 288),
                key=(width, 128),
                pool_gate=(width, 128),
                score_weights=(width, 32),
            ),
        )
        self.assertEqual(
            glm53_flash.sparse_mla_projection_workspace_shapes(1455).latent,
            (1455, 512),
        )
        for batch_size in (0, glm53_flash.KDA_CHUNK_SIZE + 1):
            with self.assertRaisesRegex(ValueError, rf"\[1, {glm53_flash.KDA_CHUNK_SIZE}\]"):
                glm53_flash.sparse_mla_projection_workspace_shapes(batch_size)

    def test_cpu_packer_retains_four_gemm_weights(self) -> None:
        raw = {
            "q_a_proj.weight": torch.tensor(((1.0, 2.0),)),
            "q_a_proj.weight_scale_inv": torch.tensor(((3.0, 4.0),)),
            "q_a_layernorm.weight": torch.tensor((5.0,)),
            "kv_a_proj_with_mqa.weight": torch.tensor(((6.0, 7.0),)),
            "kv_a_proj_with_mqa.weight_scale_inv": torch.tensor(((8.0, 9.0),)),
            "kv_a_layernorm.weight": torch.tensor((10.0,)),
            "q_b_proj.weight": torch.tensor(((11.0,),)),
            "q_b_proj.weight_scale_inv": torch.tensor(((12.0,),)),
            "indexer.wq_b.weight": torch.tensor(((13.0,),)),
            "indexer.wk.weight": torch.tensor(((14.0, 15.0),)),
            "indexer.index_kpool_compress_gate": torch.tensor(((16.0, 17.0),)),
            "indexer.weights_proj.weight": torch.tensor(((18.0, 19.0),)),
            "indexer.k_norm.weight": torch.tensor((20.0,)),
            "indexer.k_norm.bias": torch.tensor((21.0,)),
        }

        weights = glm53_checkpoint.pack_sparse_mla_projection_weights(raw, torch)

        self.assertTrue(
            torch.equal(
                weights.low_rank,
                torch.tensor(((1.0, 2.0), (6.0, 7.0))),
            )
        )
        self.assertTrue(
            torch.equal(
                weights.low_rank_scale_inv,
                torch.tensor(((3.0, 4.0), (8.0, 9.0))),
            )
        )
        self.assertTrue(
            torch.equal(
                weights.index_prep,
                torch.tensor(((14.0, 15.0), (16.0, 17.0), (18.0, 19.0))),
            )
        )
        self.assertIs(weights.wq_b, raw["indexer.wq_b.weight"])
        self.assertIs(weights.q_b, raw["q_b_proj.weight"])


class SparseMLAProjectionOpsTest(unittest.TestCase):
    def test_mn_scale_view_is_allocation_free_column_major_storage(self) -> None:
        function = functions_by_name(OPS_SOURCE)["_mn_scale_view"]
        namespace = {"torch": torch}
        exec(  # noqa: S102
            compile(
                ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
                "glm53.py",
                "exec",
            ),
            namespace,
        )

        storage = torch.empty((6, 32))
        scale = namespace["_mn_scale_view"](storage)

        self.assertEqual(scale.shape, (6, 32))
        self.assertEqual(scale.stride(), (1, 6))
        self.assertEqual(scale.data_ptr(), storage.data_ptr())

    def test_prelude_is_exact_four_gemms_without_state_writes(self) -> None:
        method = functions_by_name(OPS_SOURCE)["decode_sparse_mla_projection"]
        calls = list(ast_calls(method))
        source = ast.unparse(method)

        for name, count in (
            ("_mn_scale_view", 2),
            ("_quantize", 2),
            ("_fp8_gemm", 2),
            ("mm", 2),
            ("_rmsnorm", 2),
            ("_layernorm_bf16", 1),
        ):
            self.assertEqual(count_calls(calls, name), count)
        self.assertIn("weights.wq_b.t()", source)
        self.assertIn("SPARSE_MLA_INDEX_WEIGHT_SCALE", source)
        self.assertIn("hidden_scale = _mn_scale_view(w.hidden_scale)", source)
        self.assertIn("q_resid_scale = _mn_scale_view(w.q_resid_scale)", source)
        self.assertIn(
            "attention_tp_size = _attention_tp_size_from_heads(local_heads)", source
        )
        self.assertIn(
            "sparse_mla_projection_workspace_shapes(batch_size, attention_tp_size)",
            source,
        )
        attention_calls = [
            call_name(call)
            for call in calls
            if "attention" in (call_name(call) or "").lower()
        ]
        self.assertEqual(attention_calls, ["_attention_tp_size_from_heads"])
        for excluded in ("cache", "scorer", "topk"):
            self.assertNotIn(excluded, source.lower())
        for excluded in ("_run_sparse_mla_query", "_write_sparse_mla_history"):
            self.assertNotIn(excluded, source)
        for allocation in ("empty", "zeros", "clone"):
            self.assertEqual(count_calls(calls, allocation), 0)

        for field in glm53_flash.SparseMLAProjectionWorkspace.__dataclass_fields__:
            self.assertIn(f"('{field}', w.{field},", source)
        self.assertIn("not hidden_states.is_contiguous()", source)
        self.assertIn("not tensor.is_contiguous()", source)
        self.assertLess(
            source.index("for name, tensor, dtype"), source.index("_quantize")
        )

    def test_layernorm_writes_exact_bf16_key_without_allocation(self) -> None:
        kernel = functions_by_name(OPS_SOURCE)["_layernorm_bf16"]
        source = ast.unparse(kernel)

        self.assertIn(".to(tl.float32)", source)
        self.assertIn("tl.sum", source)
        self.assertIn("tl.rsqrt", source)
        self.assertIn("SPARSE_MLA_LAYER_NORM_EPS", OPS_SOURCE)
        self.assertNotIn("torch.nn.functional.layer_norm", OPS_SOURCE)


def functions_by_name(source: str) -> dict[str, ast.FunctionDef]:
    return {
        node.name: node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.FunctionDef)
    }


def ast_calls(node: ast.AST):
    return (child for child in ast.walk(node) if isinstance(child, ast.Call))


def count_calls(calls: list[ast.Call], name: str) -> int:
    return sum(call_name(call) == name for call in calls)


def call_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    if isinstance(call.func, ast.Subscript) and isinstance(call.func.value, ast.Name):
        return call.func.value.id
    return None


if __name__ == "__main__":
    unittest.main()

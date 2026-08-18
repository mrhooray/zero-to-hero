import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OPS = ROOT / "src/infer/models/glm53_flash/ops/core.py"


class SparseMLAPrefillOpsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = OPS.read_text()
        tree = ast.parse(cls.source)
        cls.methods = {
            node.name: node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
        }

    def test_history_entrypoint_slices_only_the_staged_token_prefix(self) -> None:
        method = self.methods["prefill_sparse_mla_history"]
        self.assertEqual(
            [argument.arg for argument in method.args.args],
            ["self", "projection", "weights", "staged", "history"],
        )
        calls = [node for node in ast.walk(method) if isinstance(node, ast.Call)]
        append = next(
            call
            for call in calls
            if ast.unparse(call.func) == "glm53_sparse_mla_packed_prefill_append"
        )
        self.assertEqual(
            [(keyword.arg, ast.unparse(keyword.value)) for keyword in append.keywords],
            [
                ("batch", "staged"),
                ("latent", "projection.latent[rows]"),
                ("key", "projection.key[rows]"),
                ("gate", "projection.pool_gate[rows]"),
                ("pool_ape", "weights.pool_ape"),
                ("history", "history"),
            ],
        )
        self.assertEqual(
            ast.unparse(method.body[0].value), "slice(staged.total_tokens)"
        )

    def test_query_entrypoint_prepares_lengths_then_uses_shared_core(self) -> None:
        method = self.methods["prefill_sparse_mla_query"]
        source = ast.unparse(method)
        ordered = (
            "_validate_sparse_mla_prefill_query",
            "_sparse_mla_runtime",
            "_prepare_sparse_mla_query_lengths",
            "_run_sparse_mla_query",
        )
        positions = [source.index(name) for name in ordered]
        self.assertEqual(positions, sorted(positions))
        self.assertNotIn("_write_sparse_mla_history", source)
        for allocation in ("empty", "zeros", "tensor", "arange", "clone", "item"):
            self.assertNotIn(f".{allocation}(", source)


if __name__ == "__main__":
    unittest.main()

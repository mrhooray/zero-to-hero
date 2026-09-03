import ast
import importlib
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

MODULE = "infer.models.deepseek_v4_flash.ops.mhc"
SOURCE = (Path(__file__).resolve().parents[1] / "src/infer/models/deepseek_v4_flash/ops/mhc.py").read_text()
FUNCTIONS = {
    node.name: node
    for node in ast.walk(ast.parse(SOURCE))
    if isinstance(node, ast.FunctionDef)
}


class _Tensor:
    def __init__(self, shape=()):
        self.shape = shape


class _Kernel:
    def __init__(self, name, calls):
        self.name = name
        self.calls = calls

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.calls.append((self.name, grid, args, kwargs))

        return launch


def _load_module():
    triton = types.ModuleType("triton")
    language = types.ModuleType("triton.language")
    language.constexpr = object()
    triton.jit = lambda function: function
    with patch.dict(
        sys.modules,
        {"triton": triton, "triton.language": language},
    ):
        sys.modules.pop(MODULE, None)
        module = importlib.import_module(MODULE)
        sys.modules.pop(MODULE, None)
    return module


class DeepSeekV4MHCTransitionTest(unittest.TestCase):
    def test_fixed_buckets_launch_two_ordered_caller_owned_kernels(self) -> None:
        for rows, expected in (
            (1, (12, 8, 512, 2)),
            (4, (12, 8, 512, 2)),
            (16, (6, 4, 1024, 4)),
            (32, (6, 4, 1024, 4)),
        ):
            with self.subTest(rows=rows):
                module = _load_module()
                calls = []
                module._post_pre_partials = _Kernel("partials", calls)
                module._finish_pre_norm = _Kernel("finish", calls)
                tensors = [_Tensor() for _ in range(11)]
                tensors[1] = _Tensor((rows, 4, 4096))

                module.deepseek_v4_mhc_transition(*tensors)

                mix_blocks, splits, block_h, mix_tile = expected
                self.assertEqual([call[0] for call in calls], ["partials", "finish"])
                self.assertEqual(calls[0][1], (rows, mix_blocks, splits))
                self.assertEqual(calls[1][1], (rows, 2))
                self.assertEqual(calls[0][3]["BLOCK_H"], block_h)
                self.assertEqual(calls[0][3]["MIX_TILE"], mix_tile)
                self.assertIs(calls[0][2][5], tensors[8])
                self.assertIs(calls[1][2][0], tensors[8])
                self.assertIs(calls[0][2][6], tensors[10])
                self.assertIs(calls[1][2][7], tensors[10])
                self.assertEqual(calls[0][3]["SQ_OFFSET"], splits * rows * 24)
                self.assertEqual(calls[1][3]["SQ_OFFSET"], splits * rows * 24)

    def test_other_row_counts_are_rejected_before_launch(self) -> None:
        module = _load_module()
        module._post_pre_partials = _Kernel("partials", [])
        module._finish_pre_norm = _Kernel("finish", [])
        tensors = [_Tensor() for _ in range(11)]
        tensors[1] = _Tensor((8, 4, 4096))

        with self.assertRaisesRegex(
            ValueError, "DeepSeek V4.*exactly 1, 4, 16, or 32 rows"
        ):
            module.deepseek_v4_mhc_transition(*tensors)

    def test_kernels_preserve_bf16_boundaries_and_flashinfer_sinkhorn_order(
        self,
    ) -> None:
        partials = ast.unparse(FUNCTIONS["_post_pre_partials"])
        finish = ast.unparse(FUNCTIONS["_finish_pre_norm"])
        sinkhorn = ast.unparse(FUNCTIONS["_sinkhorn4"])

        self.assertIn("rounded = values.to(tl.bfloat16)", partials)
        self.assertIn("values = rounded.to(tl.float32)", partials)
        self.assertEqual(partials.count("tl.fma("), 4)
        self.assertIn("collapsed = collapsed.to(tl.bfloat16)", finish)
        self.assertIn("collapsed = collapsed.to(tl.float32)", finish)
        self.assertIn("task = tl.program_id(1)", finish)
        self.assertIn("if task == 0:", finish)
        self.assertIn("2.0 * tl.sigmoid(post_logits)", finish)
        self.assertIn("values / _group_sum4(values, rows) + eps", sinkhorn)
        self.assertIn("tl.range(1, iterations, loop_unroll_factor=1)", sinkhorn)
        self.assertNotIn("empty(", SOURCE)
        self.assertNotIn("autotune", SOURCE)


if __name__ == "__main__":
    unittest.main()

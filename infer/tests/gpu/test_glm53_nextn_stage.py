import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import torch

from infer.models.glm53_flash import model as glm53_flash

pytestmark = pytest.mark.gpu

ROOT = Path(__file__).resolve().parents[2]
IMPLEMENTATION = ROOT / "src/infer/models/glm53_flash/ops/nextn_stage.py"
HAS_GPU_RUNTIME = (
    importlib.util.find_spec("triton") is not None and torch.cuda.is_available()
)


def load_implementation():
    triton = types.ModuleType("triton")
    language = types.ModuleType("triton.language")
    triton.jit = lambda function: function
    triton.cdiv = lambda left, right: (left + right - 1) // right
    triton.language = language
    language.constexpr = object()
    spec = importlib.util.spec_from_file_location(
        "infer.models.glm53_flash.ops.nextn_stage", IMPLEMENTATION
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load GLM NextN staging module")
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"triton": triton, "triton.language": language}):
        spec.loader.exec_module(module)
    return module


class FakeTensor:
    def __init__(self, shape, dtype, device, *, stride=None):
        self.shape = shape
        self.ndim = len(shape)
        self.dtype = dtype
        self.device = device
        self.is_cuda = True
        self._stride = stride or tuple(torch.empty(shape).stride() if shape else ())

    def is_contiguous(self):
        return True

    def stride(self, dimension):
        return self._stride[dimension]


class FakeKernel:
    def __init__(self):
        self.grid = None
        self.arguments = None
        self.keywords = None

    def __getitem__(self, grid):
        self.grid = grid

        def launch(*arguments, **keywords):
            self.arguments = arguments
            self.keywords = keywords

        return launch


class GLM53NextNStageContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.implementation = load_implementation()

    def make_tensors(self, sequences=64):
        cuda = SimpleNamespace(type="cuda")
        capacity = glm53_flash.KDA_CHUNK_SIZE
        return (
            FakeTensor((6,), torch.int64, cuda),
            FakeTensor((6, glm53_flash.HIDDEN_SIZE), torch.bfloat16, cuda),
            FakeTensor((3, glm53_flash.HIDDEN_SIZE), torch.bfloat16, cuda),
            FakeTensor((sequences + 1,), torch.int32, cuda),
            FakeTensor((sequences,), torch.int32, cuda),
            FakeTensor((sequences,), torch.int64, cuda),
            FakeTensor((sequences,), torch.int32, cuda),
            FakeTensor((sequences,), torch.bool, cuda),
            FakeTensor((sequences,), torch.int64, cuda),
            FakeTensor((sequences,), torch.int64, cuda),
            FakeTensor((capacity,), torch.int64, cuda),
            FakeTensor((4, glm53_flash.HIDDEN_SIZE), torch.bfloat16, cuda),
            FakeTensor((capacity,), torch.int32, cuda),
            FakeTensor((capacity,), torch.int64, cuda),
            FakeTensor((capacity,), torch.uint8, cuda),
            FakeTensor((capacity,), torch.int32, cuda),
        )

    def test_launches_one_fixed_compactor_into_caller_outputs(self):
        tensors = self.make_tensors()
        descriptors = FakeKernel()
        compactor = FakeKernel()
        self.implementation._glm53_stage_nextn_descriptors = descriptors
        self.implementation._glm53_stage_nextn_prefill_kernel = compactor

        self.implementation.glm53_stage_nextn_prefill(*tensors, 4, 3)

        self.assertEqual(descriptors.grid, (1,))
        self.assertEqual(descriptors.arguments, (*tensors[3:10], 3))
        self.assertEqual(descriptors.keywords["MAX_SEQUENCES"], 64)
        self.assertEqual(compactor.grid, (4, 16))
        self.assertEqual(
            compactor.arguments,
            (
                *tensors[:4],
                tensors[6],
                tensors[4],
                tensors[5],
                tensors[7],
                *tensors[10:],
            ),
        )
        self.assertEqual(compactor.keywords["MAX_SEQUENCES"], 64)
        self.assertEqual(compactor.keywords["HIDDEN"], glm53_flash.HIDDEN_SIZE)
        self.assertEqual(compactor.keywords["BLOCK_SIZE"], 256)

    def test_dep_uses_local_sequence_capacity(self):
        tensors = self.make_tensors(16)
        descriptors = FakeKernel()
        compactor = FakeKernel()
        self.implementation._glm53_stage_nextn_descriptors = descriptors
        self.implementation._glm53_stage_nextn_prefill_kernel = compactor

        self.implementation.glm53_stage_nextn_prefill(*tensors, 4, 3)

        self.assertEqual(descriptors.keywords["MAX_SEQUENCES"], 16)
        self.assertEqual(compactor.keywords["MAX_SEQUENCES"], 16)

    def test_request_varying_axes_are_runtime_values(self):
        source = IMPLEMENTATION.read_text()

        self.assertIn("tl.static_range(MAX_SEQUENCES)", source)
        self.assertIn("tl.cumsum(pair_counts, axis=0)", source)
        self.assertIn("tl.maximum(source - 1, 0)", source)
        self.assertNotIn("total_tokens: tl.constexpr", source)
        self.assertNotIn("sequence_count: tl.constexpr", source)


@unittest.skipUnless(HAS_GPU_RUNTIME, "requires CUDA and Triton")
class GLM53NextNStageGPUTest(unittest.TestCase):
    def test_mixed_fresh_and_resumed_pairs(self):
        from infer.models.glm53_flash.ops.nextn_stage import glm53_stage_nextn_prefill

        device = "cuda"
        capacity = glm53_flash.KDA_CHUNK_SIZE
        tokens = torch.tensor((10, 11, 12, 20, 21, 30), device=device)
        target = (
            torch.arange(6, dtype=torch.bfloat16, device=device)[:, None]
            .expand(-1, glm53_flash.HIDDEN_SIZE)
            .contiguous()
        )
        pending = torch.full(
            (3, glm53_flash.HIDDEN_SIZE), 99, dtype=torch.bfloat16, device=device
        )
        cu = torch.zeros(33, dtype=torch.int32, device=device)
        cu[:4] = torch.tensor((0, 3, 5, 6), dtype=torch.int32, device=device)
        sources = torch.zeros(32, dtype=torch.int32, device=device)
        starts = torch.zeros(32, dtype=torch.int32, device=device)
        starts[:3] = torch.tensor((0, 1, 0), dtype=torch.int32, device=device)
        slots = torch.zeros(32, dtype=torch.int64, device=device)
        slots[:3] = torch.tensor((2, 0, 1), dtype=torch.int64, device=device)
        has_pending = torch.zeros(32, dtype=torch.bool, device=device)
        last_indices = torch.empty(32, dtype=torch.int64, device=device)
        last_slots = torch.empty(32, dtype=torch.int64, device=device)
        output_tokens = torch.empty(capacity, dtype=torch.int64, device=device)
        output_hidden = torch.empty(
            (4, glm53_flash.HIDDEN_SIZE), dtype=torch.bfloat16, device=device
        )
        sequence_ids = torch.empty(capacity, dtype=torch.int32, device=device)
        token_slots = torch.empty(capacity, dtype=torch.int64, device=device)
        active = torch.empty(capacity, dtype=torch.uint8, device=device)
        raw_lengths = torch.empty(capacity, dtype=torch.int32, device=device)

        glm53_stage_nextn_prefill(
            tokens,
            target,
            pending,
            cu,
            starts,
            slots,
            sources,
            has_pending,
            last_indices,
            last_slots,
            output_tokens,
            output_hidden,
            sequence_ids,
            token_slots,
            active,
            raw_lengths,
            4,
            3,
        )

        self.assertEqual(cu[:4].tolist(), [0, 2, 4, 4])
        self.assertEqual(sources[:2].tolist(), [1, 3])
        self.assertEqual(starts[:2].tolist(), [0, 0])
        self.assertEqual(slots[:2].tolist(), [2, 0])
        self.assertEqual(has_pending[:2].tolist(), [False, True])
        self.assertEqual(last_indices[:3].tolist(), [2, 4, 5])
        self.assertEqual(last_slots[:3].tolist(), [2, 0, 1])
        self.assertEqual(output_tokens[:4].tolist(), [11, 12, 20, 21])
        self.assertEqual(output_hidden[:, 0].tolist(), [0, 1, 99, 3])
        self.assertEqual(sequence_ids[:4].tolist(), [0, 0, 1, 1])
        self.assertEqual(token_slots[:4].tolist(), [2, 2, 0, 0])
        self.assertEqual(active[:4].tolist(), [1, 1, 1, 1])
        self.assertEqual(raw_lengths[:4].tolist(), [1, 2, 1, 2])


if __name__ == "__main__":
    unittest.main()

import ast
import importlib.util
import inspect
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import torch

from infer.models.glm53_flash import model as glm53_flash
from infer.models.glm53_flash.codec import EOS_TOKEN_IDS

pytestmark = pytest.mark.gpu

ROOT = Path(__file__).resolve().parents[2]
IMPLEMENTATION = ROOT / "src/infer/models/glm53_flash/ops/speculate.py"
HAS_GPU_RUNTIME = (
    importlib.util.find_spec("triton") is not None and torch.cuda.is_available()
)
WIDTH = glm53_flash.GLM53_TARGET_VERIFY_WIDTH


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


def load_implementation():
    triton = types.ModuleType("triton")
    language = types.ModuleType("triton.language")
    triton.jit = lambda function: function
    triton.cdiv = lambda left, right: (left + right - 1) // right
    triton.language = language
    language.constexpr = object()
    spec = importlib.util.spec_from_file_location(
        "infer.models.glm53_flash.ops.speculate", IMPLEMENTATION
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load GLM speculation module")
    module = importlib.util.module_from_spec(spec)
    with patch.dict(
        sys.modules,
        {"triton": triton, "triton.language": language},
    ):
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


class GLM53SpeculateContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.implementation = load_implementation()

    def make_tensors(self):
        device = SimpleNamespace(type="cuda")
        return (
            FakeTensor((3, WIDTH), torch.int64, device),
            FakeTensor((3, WIDTH), torch.int64, device),
            FakeTensor((3,), torch.int32, device),
            FakeTensor((3,), torch.uint8, device),
            FakeTensor((3,), torch.uint8, device),
            FakeTensor((3,), torch.int32, device),
            FakeTensor((3, WIDTH), torch.int64, device),
            FakeTensor((3,), torch.uint8, device),
        )

    def test_launches_one_fixed_specialization_into_caller_outputs(self):
        tensors = self.make_tensors()
        kernel = FakeKernel()
        self.implementation._glm53_greedy_accept_kernel = kernel

        result = self.implementation.glm53_greedy_accept(*tensors)

        self.assertEqual(result, tensors[-3:])
        self.assertEqual(kernel.grid, (3,))
        self.assertEqual(kernel.arguments, tensors)
        self.assertEqual(kernel.keywords["WIDTH"], WIDTH)
        self.assertEqual(kernel.keywords["BLOCK_SIZE"], 8)
        self.assertEqual(
            {kernel.keywords[f"EOS_{index}"] for index in range(3)},
            EOS_TOKEN_IDS,
        )
        self.assertEqual(kernel.keywords["num_warps"], 1)

    def test_rejects_wrong_shape_dtype_device_and_layout(self):
        cases = (
            (0, "shape", (3, 5), ValueError),
            (2, "dtype", torch.int64, TypeError),
            (3, "dtype", torch.bool, TypeError),
            (6, "is_cuda", False, ValueError),
            (1, "is_contiguous", lambda: False, ValueError),
        )
        for position, attribute, value, error in cases:
            tensors = list(self.make_tensors())
            setattr(tensors[position], attribute, value)
            with (
                self.subTest(position=position, attribute=attribute),
                self.assertRaises(error),
            ):
                self.implementation.glm53_greedy_accept(*tensors)

    def test_selected_row_copy_uses_runtime_row_strides(self):
        device = SimpleNamespace(type="cuda")
        tensors = (
            FakeTensor((3 * WIDTH, 2, 4), torch.bfloat16, device),
            FakeTensor((8, 2, 4), torch.bfloat16, device),
            FakeTensor((3,), torch.int32, device),
            FakeTensor((3,), torch.int32, device),
            FakeTensor((3,), torch.uint8, device),
        )
        kernel = FakeKernel()
        self.implementation._glm53_copy_verified_rows_kernel = kernel

        self.implementation.glm53_copy_verified_rows(*tensors)

        self.assertEqual(kernel.grid, (3, 1))
        self.assertEqual(kernel.arguments[:5], tensors)
        self.assertEqual(kernel.arguments[5:], (8, 8))
        self.assertEqual(kernel.keywords["ROW_SIZE"], 8)
        self.assertEqual(kernel.keywords["WIDTH"], WIDTH)

    def test_selected_row_table_batches_one_uniform_state_family(self):
        device = SimpleNamespace(type="cuda")
        tensors = (
            FakeTensor((34,), torch.uint64, device),
            FakeTensor((34,), torch.uint64, device),
            FakeTensor((34,), torch.int64, device),
            FakeTensor((34,), torch.int64, device),
            FakeTensor((3,), torch.int32, device),
            FakeTensor((3,), torch.int32, device),
            FakeTensor((3,), torch.uint8, device),
        )
        kernel = FakeKernel()
        self.implementation._glm53_copy_verified_row_table_kernel = kernel

        self.implementation.glm53_copy_verified_row_table(*tensors, row_words=262_144)

        self.assertEqual(kernel.grid, (102, 256))
        self.assertEqual(kernel.arguments, tensors)
        self.assertEqual(kernel.keywords["GROUPS"], 3)
        self.assertEqual(kernel.keywords["WIDTH"], WIDTH)
        self.assertEqual(kernel.keywords["ROW_WORDS"], 262_144)
        self.assertEqual(kernel.keywords["BLOCK_WORDS"], 1024)

    def test_publish_launches_into_resident_slot_state(self):
        device = SimpleNamespace(type="cuda")
        tensors = (
            FakeTensor((3, WIDTH), torch.int64, device),
            FakeTensor((3,), torch.int32, device),
            FakeTensor((3,), torch.int32, device),
            FakeTensor((3,), torch.uint8, device),
            FakeTensor((8,), torch.int64, device),
            FakeTensor((8,), torch.int32, device),
        )
        kernel = FakeKernel()
        self.implementation._glm53_publish_accepted_kernel = kernel

        self.implementation.glm53_publish_accepted(*tensors)

        self.assertEqual(kernel.grid, (3,))
        self.assertEqual(kernel.arguments, tensors)
        self.assertEqual(kernel.keywords["WIDTH"], WIDTH)

    def test_acceptance_accumulator_stays_scalar(self):
        source = inspect.getsource(self.implementation._glm53_greedy_accept_kernel)

        self.assertNotIn("tl.full", source)
        self.assertNotIn("tl.static_range", source)
        self.assertIn("match_3", source)

    def test_commit_masks_are_boolean(self):
        for kernel in (
            self.implementation._glm53_copy_verified_rows_kernel,
            self.implementation._glm53_copy_verified_row_table_kernel,
            self.implementation._glm53_publish_accepted_kernel,
        ):
            self.assertIn(
                "tl.load(active_ptr + row) != 0",
                inspect.getsource(kernel),
            )

    def test_hot_path_has_no_host_scalar_extraction_or_torch_allocation(self):
        tree = ast.parse(IMPLEMENTATION.read_text())
        forbidden_attributes = {"item", "tolist", "cpu", "numpy"}
        self.assertFalse(
            any(
                isinstance(node, ast.Attribute) and node.attr in forbidden_attributes
                for node in ast.walk(tree)
            )
        )
        self.assertFalse(
            any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "torch"
                for node in ast.walk(tree)
            )
        )


@unittest.skipUnless(HAS_GPU_RUNTIME, "requires CUDA and Triton")
class GLM53SpeculateGPUTest(unittest.TestCase):
    def setUp(self):
        from infer.models.glm53_flash.ops.speculate import (
            glm53_copy_verified_row_table,
            glm53_copy_verified_rows,
            glm53_greedy_accept,
            glm53_publish_accepted,
        )

        self.accept = glm53_greedy_accept
        self.copy_table = glm53_copy_verified_row_table
        self.copy_rows = glm53_copy_verified_rows
        self.publish = glm53_publish_accepted
        self.candidates = torch.empty((3, WIDTH), dtype=torch.int64, device="cuda")
        self.targets = torch.empty((3, WIDTH), dtype=torch.int64, device="cuda")
        self.remaining = torch.empty(3, dtype=torch.int32, device="cuda")
        self.ignore_eos = torch.zeros(3, dtype=torch.uint8, device="cuda")
        self.active = torch.empty(3, dtype=torch.uint8, device="cuda")
        self.accepted = torch.empty(3, dtype=torch.int32, device="cuda")
        self.output = torch.empty((3, WIDTH), dtype=torch.int64, device="cuda")
        self.continuing = torch.empty(3, dtype=torch.uint8, device="cuda")

    def assert_acceptance(self, candidates, targets, remaining):
        inactive_candidates = tuple(reversed(candidates))
        inactive_targets = tuple(reversed(targets))
        self.candidates.copy_(
            torch.tensor((candidates, inactive_candidates, candidates), device="cuda")
        )
        self.targets.copy_(
            torch.tensor((targets, inactive_targets, targets), device="cuda")
        )
        self.remaining.copy_(torch.tensor((remaining, WIDTH, 1), device="cuda"))
        self.active.copy_(torch.tensor((1, 0, 1), dtype=torch.uint8, device="cuda"))
        self.output.fill_(-1)

        self.accept(
            self.candidates,
            self.targets,
            self.remaining,
            self.ignore_eos,
            self.active,
            self.accepted,
            self.output,
            self.continuing,
        )

        expected = _accept_glm53_greedy_chain(candidates, targets, remaining)
        accepted = self.accepted.cpu().tolist()
        output = self.output.cpu().tolist()
        continuing = self.continuing.cpu().tolist()
        self.assertEqual(accepted, [len(expected), 0, 1])
        self.assertEqual(output[0], [*expected, *(0,) * (WIDTH - len(expected))])
        self.assertEqual(output[1], [0] * WIDTH)
        self.assertEqual(output[2], [targets[0], *(0,) * (WIDTH - 1)])
        self.assertEqual(
            continuing,
            [
                int(
                    expected[-1] not in EOS_TOKEN_IDS
                    and (remaining == WIDTH or len(expected) < remaining)
                ),
                0,
                0,
            ],
        )

    def test_zero_remaining_has_no_commit_or_continuation(self):
        self.candidates.fill_(1)
        self.targets.fill_(2)
        self.remaining.zero_()
        self.active.fill_(1)

        self.accept(
            self.candidates,
            self.targets,
            self.remaining,
            self.ignore_eos,
            self.active,
            self.accepted,
            self.output,
            self.continuing,
        )

        self.assertEqual(self.accepted.cpu().tolist(), [0, 0, 0])
        self.assertEqual(self.continuing.cpu().tolist(), [0, 0, 0])

    def test_device_commit_masks_zero_and_inactive_rows(self):
        source = torch.arange(3 * WIDTH, dtype=torch.bfloat16, device="cuda")
        source = source[:, None].expand(-1, 6).contiguous().view(3 * WIDTH, 2, 3)
        destination = torch.full((5, 2, 3), -1, dtype=torch.bfloat16, device="cuda")
        accepted = torch.tensor((0, WIDTH + 1, 2), dtype=torch.int32, device="cuda")
        slots = torch.tensor((2, -1, 4), dtype=torch.int32, device="cuda")
        active = torch.tensor((1, 0, 1), dtype=torch.uint8, device="cuda")

        self.copy_rows(source, destination, accepted, slots, active)

        self.assertTrue(torch.all(destination[4] == 9))
        self.assertTrue(torch.all(destination[:4] == -1))

        output = torch.arange(3 * WIDTH, dtype=torch.int64, device="cuda").view(
            3, WIDTH
        )
        sampled = torch.full((5,), -1, dtype=torch.int64, device="cuda")
        lengths = torch.tensor((10, 11, 12, 13, 14), dtype=torch.int32, device="cuda")
        self.publish(output, accepted, slots, active, sampled, lengths)

        self.assertEqual(sampled.cpu().tolist(), [-1, -1, -1, -1, 9])
        self.assertEqual(lengths.cpu().tolist(), [10, 11, 12, 13, 16])

    def test_table_copy_replays_changed_controls_in_cuda_graph(self):
        layers, groups, row_words, slots = 3, 5, 8, 7
        source_backing = torch.arange(
            layers * groups * WIDTH * row_words * 2,
            dtype=torch.int32,
            device="cuda",
        ).view(layers, groups * WIDTH, row_words * 2)
        source = source_backing[:, :, :row_words]
        destination = torch.full(
            (layers, slots, row_words), -1, dtype=torch.int32, device="cuda"
        )
        source_addresses = torch.tensor(
            [source[layer].data_ptr() for layer in range(layers)],
            dtype=torch.uint64,
            device="cuda",
        )
        destination_addresses = torch.tensor(
            [destination[layer].data_ptr() for layer in range(layers)],
            dtype=torch.uint64,
            device="cuda",
        )
        source_strides = torch.tensor(
            [source[layer].stride(0) for layer in range(layers)],
            dtype=torch.int64,
            device="cuda",
        )
        destination_strides = torch.tensor(
            [destination[layer].stride(0) for layer in range(layers)],
            dtype=torch.int64,
            device="cuda",
        )
        accepted = torch.tensor((0, 1, 2, 3, 4), dtype=torch.int32, device="cuda")
        state_slots = torch.tensor((1, -1, 3, 4, 5), dtype=torch.int32, device="cuda")
        active = torch.tensor((1, 0, 1, 1, 1), dtype=torch.uint8, device="cuda")
        source_before = source_backing.clone()

        def copy():
            self.copy_table(
                source_addresses,
                destination_addresses,
                source_strides,
                destination_strides,
                accepted,
                state_slots,
                active,
                row_words=row_words,
            )

        copy()
        self.assertTrue(torch.equal(destination[:, 3], source[:, 9]))
        self.assertTrue(torch.equal(destination[:, 4], source[:, 14]))
        self.assertTrue(torch.equal(destination[:, 5], source[:, 19]))
        self.assertTrue(torch.all(destination[:, (0, 1, 2, 6)] == -1))

        destination.fill_(-1)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            copy()
        destination.fill_(-1)
        accepted.copy_(torch.tensor((4, 3, 2, 1, 0), device="cuda"))
        state_slots.copy_(torch.tensor((6, 5, -1, 2, 1), device="cuda"))
        active.copy_(torch.tensor((1, 1, 0, 1, 1), dtype=torch.uint8, device="cuda"))
        graph.replay()

        self.assertTrue(torch.equal(destination[:, 6], source[:, 3]))
        self.assertTrue(torch.equal(destination[:, 5], source[:, 6]))
        self.assertTrue(torch.equal(destination[:, 2], source[:, 12]))
        self.assertTrue(torch.all(destination[:, (0, 1, 3, 4)] == -1))
        self.assertTrue(torch.equal(source_backing, source_before))

    def test_continuation_distinguishes_budget_eos_and_full_width(self):
        targets = tuple(range(11, 11 + WIDTH))
        matching = (100, *targets[:-1])
        cases = (
            (matching, targets, 0, 0),
            ((100, targets[0] + 1, *targets[1:-1]), targets, 3, 1),
            (matching, targets, 3, 0),
            (matching, (targets[0], min(EOS_TOKEN_IDS), *targets[2:]), WIDTH, 0),
            (matching, targets, WIDTH, 1),
        )
        for candidates, target, remaining, expected in cases:
            self.candidates.copy_(
                torch.tensor((candidates,) * 3, dtype=torch.int64, device="cuda")
            )
            self.targets.copy_(
                torch.tensor((target,) * 3, dtype=torch.int64, device="cuda")
            )
            self.remaining.fill_(remaining)
            self.active.fill_(1)
            self.accept(
                self.candidates,
                self.targets,
                self.remaining,
                self.ignore_eos,
                self.active,
                self.accepted,
                self.output,
                self.continuing,
            )
            with self.subTest(remaining=remaining, target=target):
                self.assertEqual(self.continuing.cpu().tolist(), [expected] * 3)

    def test_matches_cpu_oracle_for_mismatch_budget_and_each_eos_position(self):
        targets = tuple(range(11, 11 + WIDTH))
        matching = (100, *targets[:-1])
        for mismatch in range(1, WIDTH):
            candidates = list(matching)
            candidates[mismatch] += 100
            for remaining in range(1, WIDTH + 1):
                with self.subTest(mismatch=mismatch, remaining=remaining):
                    self.assert_acceptance(tuple(candidates), targets, remaining)
        for eos_token_id in EOS_TOKEN_IDS:
            for eos_position in range(WIDTH):
                eos_targets = list(targets)
                eos_targets[eos_position] = eos_token_id
                with self.subTest(eos_token_id=eos_token_id, position=eos_position):
                    self.assert_acceptance(matching, tuple(eos_targets), WIDTH)

    def test_ignore_eos_accepts_the_matching_suffix(self):
        eos = min(EOS_TOKEN_IDS)
        targets = (11, eos, 13, 14)
        candidates = (100, *targets[:-1])
        self.candidates.copy_(torch.tensor((candidates,) * 3, device="cuda"))
        self.targets.copy_(torch.tensor((targets,) * 3, device="cuda"))
        self.remaining.fill_(WIDTH)
        self.ignore_eos.copy_(torch.tensor((1, 0, 1), device="cuda"))
        self.active.fill_(1)

        self.accept(
            self.candidates,
            self.targets,
            self.remaining,
            self.ignore_eos,
            self.active,
            self.accepted,
            self.output,
            self.continuing,
        )

        self.assertEqual(self.accepted.cpu().tolist(), [WIDTH, 2, WIDTH])
        self.assertEqual(self.continuing.cpu().tolist(), [1, 0, 1])

    def test_replays_in_a_cuda_graph(self):
        targets = tuple(range(11, 11 + WIDTH))
        matching = (100, *targets[:-1])
        self.assert_acceptance(matching, targets, WIDTH)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self.accept(
                self.candidates,
                self.targets,
                self.remaining,
                self.ignore_eos,
                self.active,
                self.accepted,
                self.output,
                self.continuing,
            )

        self.remaining.copy_(torch.tensor((3, WIDTH, 1), device="cuda"))
        graph.replay()

        self.assertEqual(self.accepted.cpu().tolist(), [3, 0, 1])
        self.assertEqual(
            self.output.cpu()[0].tolist(), [11, 12, 13, *(0,) * (WIDTH - 3)]
        )


if __name__ == "__main__":
    unittest.main()

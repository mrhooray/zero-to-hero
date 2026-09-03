import ast
import hashlib
import importlib.util
import inspect
import sys
import types
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

from infer.models.glm53_flash import model as glm53

ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATION = ROOT / "src/infer/models/glm53_flash/ops/segmented_conv.py"
LICENSE = ROOT / "src/infer/models/glm53_flash/ops/segmented_conv.LICENSE.txt"
KERNEL_SHA256 = "5a6a1779e9e37b046c0ec7741303a697d91a7aca62f254b96bc82e902f3b0203"


def load_implementation() -> types.ModuleType:
    torch = types.ModuleType("torch")
    torch.bfloat16 = object()
    torch.float32 = object()
    torch.int32 = object()
    torch.bool = object()
    triton = types.ModuleType("triton")
    language = types.ModuleType("triton.language")

    def decorator(*args: object, **kwargs: object) -> object:
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return lambda function: function

    triton.jit = decorator
    triton.language = language
    modules = {"torch": torch, "triton": triton, "triton.language": language}
    spec = importlib.util.spec_from_file_location(
        "segmented_conv", IMPLEMENTATION
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load segmented-conv module")
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module


class SegmentedConvPlanTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.implementation = load_implementation()

    def test_accepts_canonical_segments_across_state_boundary_lengths(self) -> None:
        lengths = (1, 2, 3, 8, 9, 15, 16, 17, 65)
        cu_seqlens = [0]
        segments = []
        total = 0
        for sequence, length in enumerate(lengths):
            segments.extend(
                (sequence, offset)
                for offset in range(0, length, self.implementation.BLOCK_M)
            )
            total += length
            cu_seqlens.append(total)

        programs = self.implementation.validate_glm53_segmented_conv_plan(
            cu_seqlens,
            list(range(len(lengths))),
            [bool(sequence % 2) for sequence in range(len(lengths))],
            segments,
            total_tokens=total,
            state_pages=len(lengths),
        )

        self.assertEqual(programs, 22)

    def test_rejects_invalid_packed_lengths(self) -> None:
        cases = (
            ([1, 2], 2, "start at zero"),
            ([0, 2], 3, "end at total_tokens"),
            ([0, 0], 0, "total_tokens must be positive"),
            ([0, 1, 1], 1, "must contain at least one token"),
        )
        for cu_seqlens, total_tokens, message in cases:
            batch = len(cu_seqlens) - 1
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(ValueError, message),
            ):
                self.implementation.validate_glm53_segmented_conv_plan(
                    cu_seqlens,
                    list(range(batch)),
                    [False] * batch,
                    [(0, 0)] * batch,
                    total_tokens=total_tokens,
                    state_pages=max(batch, 1),
                )

    def test_rejects_unsafe_state_ownership(self) -> None:
        for indices, message in (([2], "state_indices"), ([0, 0], "must be unique")):
            batch = len(indices)
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(ValueError, message),
            ):
                self.implementation.validate_glm53_segmented_conv_plan(
                    list(range(batch + 1)),
                    indices,
                    [False] * batch,
                    [(sequence, 0) for sequence in range(batch)],
                    total_tokens=batch,
                    state_pages=2,
                )

    def test_rejects_noncanonical_or_malformed_segments(self) -> None:
        cases = (
            ([(0, 0)], "must contain 2 rows"),
            ([(0, 0), (0, 1)], r"must be \(0, 8\)"),
            ([(0, 0), (0, "8")], "must contain two integers"),
            ([(0, 0), (0, 8, 16)], "must contain two integers"),
        )
        for segments, message in cases:
            with (
                self.subTest(message=message),
                self.assertRaisesRegex((ValueError, TypeError), message),
            ):
                self.implementation.validate_glm53_segmented_conv_plan(
                    [0, 9],
                    [0],
                    [False],
                    segments,
                    total_tokens=9,
                    state_pages=1,
                )

    def test_requires_plain_bool_initial_flags(self) -> None:
        with self.assertRaisesRegex(TypeError, r"has_initial\[0\] must be a bool"):
            self.implementation.validate_glm53_segmented_conv_plan(
                [0, 1],
                [0],
                [1],
                [(0, 0)],
                total_tokens=1,
                state_pages=1,
            )


@dataclass
class FakeDevice:
    type: str = "cuda"


@dataclass
class FakeTensor:
    shape: tuple[int, ...]
    dtype: object
    device: FakeDevice
    strides: tuple[int, ...]
    contiguous: bool = True

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def stride(self, dimension: int | None = None) -> tuple[int, ...] | int:
        if dimension is None:
            return self.strides
        return self.strides[dimension]

    def is_contiguous(self) -> bool:
        return self.contiguous


class SegmentedConvTensorContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.implementation = load_implementation()

    def test_accepts_padded_projection_rows_and_caller_output(self) -> None:
        for attention_tp_size in (1, glm53.TP_SIZE):
            with self.subTest(attention_tp_size=attention_tp_size):
                tensors = self._valid_tensors(attention_tp_size)

                programs = self.implementation._validate_prefill_tensors(*tensors)

                self.assertEqual(programs, 3)

    def test_accepts_active_prefix_of_b64_output_without_accepting_overlap(
        self,
    ) -> None:
        tensors = self._valid_tensors()
        padded = FakeTensor(
            (3, 1, 10, 16, 128),
            self.implementation.torch.bfloat16,
            tensors[0].device,
            (131072, 131072, 2048, 128, 1),
            contiguous=False,
        )
        self.assertEqual(
            self.implementation._validate_prefill_tensors(*tensors[:-1], padded),
            3,
        )

        overlapping = FakeTensor(
            padded.shape,
            padded.dtype,
            padded.device,
            (18432, 18432, 2048, 128, 1),
            contiguous=False,
        )
        with self.assertRaisesRegex(ValueError, "non-overlapping sections"):
            self.implementation._validate_prefill_tensors(*tensors[:-1], overlapping)

    def test_rejects_non_row_contiguous_projection(self) -> None:
        tensors = self._valid_tensors()
        qkv_raw = FakeTensor(
            shape=(10, 6144),
            dtype=self.implementation.torch.bfloat16,
            device=tensors[0].device,
            strides=(6416, 2),
            contiguous=False,
        )

        with self.assertRaisesRegex(ValueError, "contiguous within each row"):
            self.implementation._validate_prefill_tensors(qkv_raw, *tensors[1:])

    def _valid_tensors(
        self, attention_tp_size: int = glm53.TP_SIZE
    ) -> tuple[FakeTensor, ...]:
        torch = self.implementation.torch
        device = FakeDevice()
        local_heads = glm53.glm53_local_heads(attention_tp_size)
        local_projection_size = local_heads * self.implementation.HEAD_DIM
        packed_qkv_size = 3 * local_projection_size
        section_stride = 10 * local_projection_size
        return (
            FakeTensor(
                (10, packed_qkv_size),
                torch.bfloat16,
                device,
                (packed_qkv_size + 272, 1),
                False,
            ),
            FakeTensor((packed_qkv_size, 4), torch.float32, device, (4, 1)),
            FakeTensor(
                (4, packed_qkv_size, 3),
                torch.bfloat16,
                device,
                (packed_qkv_size * 3, 3, 1),
            ),
            FakeTensor((3,), torch.int32, device, (1,)),
            FakeTensor((2,), torch.int32, device, (1,)),
            FakeTensor((2,), torch.bool, device, (1,)),
            FakeTensor((3, 2), torch.int32, device, (2, 1)),
            FakeTensor(
                (3, 1, 10, local_heads, self.implementation.HEAD_DIM),
                torch.bfloat16,
                device,
                (
                    section_stride,
                    section_stride,
                    local_projection_size,
                    self.implementation.HEAD_DIM,
                    1,
                ),
            ),
        )


class SegmentedConvSourceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = IMPLEMENTATION.read_text()
        cls.tree = ast.parse(cls.source)
        cls.implementation = load_implementation()

    def test_pins_origin_and_local_derivative(self) -> None:
        self.assertEqual(
            self.implementation.TOKENSPEED_COMMIT,
            "f17b03efc1728875c586d848f49da5905032e87c",
        )
        self.assertEqual(
            self.implementation.TOKENSPEED_SOURCE_PATH,
            "python/tokenspeed/runtime/layers/attention/linear/causal_conv1d.py",
        )
        self.assertEqual(
            self.implementation.TOKENSPEED_GIT_BLOB_SHA,
            "f586a0979dee382926a098dc356a41c4487e7de4",
        )
        self.assertEqual(
            self.implementation.TOKENSPEED_SOURCE_SHA256,
            "1f7afb81136fdd29aeb01407e4eb112fff20122d04b47330c57220e275d37036",
        )
        self.assertEqual(
            self.implementation.TOKENSPEED_KERNEL_SLICE_SHA256,
            "cd7b82369ee357287b2b7e6c38a39f92361f8c7fac1a3bc350d2e7de553fe41c",
        )
        self.assertEqual(
            self.implementation.TOKENSPEED_LAUNCH_SLICE_SHA256,
            "0bdda672e43db4477e35010536f2082c6f9e5e25cf313bf16679ef13c10e042c",
        )
        kernel = function(self.tree, "_glm53_segmented_causal_conv_kernel")
        start = min(decorator.lineno for decorator in kernel.decorator_list)
        block = "".join(
            self.source.splitlines(keepends=True)[start - 1 : kernel.end_lineno]
        )

        self.assertEqual(hashlib.sha256(block.encode()).hexdigest(), KERNEL_SHA256)

    def test_public_api_and_geometry_are_topology_aware(self) -> None:
        signature = inspect.signature(self.implementation.glm53_segmented_conv_prefill)

        self.assertEqual(
            list(signature.parameters),
            [
                "qkv_raw",
                "conv_weight",
                "conv_state",
                "cu_seqlens",
                "state_indices",
                "has_initial",
                "segments",
                "out",
            ],
        )
        self.assertEqual(self.implementation.CONV_KERNEL_SIZE, 4)
        self.assertEqual(self.implementation.HEAD_DIM, 128)
        self.assertEqual(self.implementation.BLOCK_M, 8)
        self.assertEqual(self.implementation.BLOCK_N, 256)
        wrapper = ast.unparse(function(self.tree, "glm53_segmented_conv_prefill"))
        validation = ast.unparse(function(self.tree, "_validate_prefill_tensors"))
        for source in (wrapper, validation):
            self.assertIn("local_heads = out.shape[3]", source)
            self.assertIn("local_projection_size = local_heads * HEAD_DIM", source)
        self.assertIn("packed_qkv_size = 3 * local_projection_size", validation)

    def test_wrapper_uses_caller_storage_and_topology_grid(self) -> None:
        wrapper = function(self.tree, "glm53_segmented_conv_prefill")
        calls = [node for node in ast.walk(wrapper) if isinstance(node, ast.Call)]
        torch_calls = [
            call
            for call in calls
            if isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "torch"
        ]
        launch = next(
            call
            for call in calls
            if isinstance(call.func, ast.Subscript)
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "_glm53_segmented_causal_conv_kernel"
        )
        keywords = {keyword.arg: keyword.value for keyword in launch.keywords}

        self.assertEqual(torch_calls, [])
        self.assertEqual(ast.unparse(wrapper.body[-1].value), "out")
        self.assertEqual(ast.unparse(keywords["out"]), "out")
        self.assertEqual(ast.unparse(keywords["stride_raw_token"]), "qkv_raw.stride(0)")
        self.assertEqual(ast.unparse(keywords["stride_out_section"]), "out.stride(0)")
        self.assertEqual(ast.unparse(keywords["stride_out_token"]), "out.stride(2)")
        kernel = ast.unparse(function(self.tree, "_glm53_segmented_causal_conv_kernel"))
        self.assertIn("do_not_specialize=['stride_out_section']", kernel)
        self.assertNotIn("stride_out_section: tl.constexpr", kernel)
        for name in ("BLOCK_M", "BLOCK_N", "CONV_KERNEL_SIZE"):
            self.assertEqual(ast.unparse(keywords[name]), name)
        self.assertEqual(
            ast.unparse(keywords["LOCAL_PROJECTION_SIZE"]),
            "local_projection_size",
        )
        source = ast.unparse(wrapper)
        self.assertIn("local_heads = out.shape[3]", source)
        self.assertIn("local_projection_size = local_heads * HEAD_DIM", source)
        self.assertIn("grid = (programs, 3 * local_projection_size // BLOCK_N)", source)
        self.assertIn("validate_glm53_segmented_conv_plan", ast.get_docstring(wrapper))

    def test_only_offset_zero_segment_writes_shift_history(self) -> None:
        kernel = function(self.tree, "_glm53_segmented_causal_conv_kernel")
        source = ast.get_source_segment(self.source, kernel)
        if source is None:
            raise RuntimeError("kernel source was unavailable")

        self.assertIn("if token_offset == 0:", source)
        self.assertIn("A final-segment writer could race", source)
        self.assertIn("for token in tl.static_range(BLOCK_M):", source)
        self.assertIn("mask=active", source)
        self.assertIn("col2 = tl.where(active, value, col2)", source)
        self.assertIn("tl.store(state_base + state_column, value)", source)
        self.assertNotIn("token_offset + segment_tokens == sequence_tokens", source)
        self.assertNotIn("torch.", source)

    def test_has_no_tokenspeed_numpy_or_fla_runtime_import(self) -> None:
        imports = set()
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module)

        self.assertEqual(
            imports,
            {
                "__future__",
                "collections.abc",
                "itertools",
                "infer.models.glm53_flash.model",
                "torch",
                "triton",
                "triton.language",
            },
        )

    def test_notice_preserves_mit_and_apache_attribution(self) -> None:
        notice = LICENSE.read_text()

        self.assertIn("SPDX-License-Identifier: MIT AND Apache-2.0", notice)
        self.assertIn("Copyright (c) 2026 LightSeek Foundation", notice)
        self.assertIn("Copyright contributors to the vLLM project", notice)
        self.assertIn("Copyright (c) 2024, Tri Dao", notice)
        self.assertIn(self.implementation.TOKENSPEED_COMMIT, notice)
        self.assertIn(self.implementation.TOKENSPEED_GIT_BLOB_SHA, notice)
        self.assertIn(self.implementation.TOKENSPEED_SOURCE_SHA256, notice)
        self.assertIn(self.implementation.TOKENSPEED_KERNEL_SLICE_SHA256, notice)
        self.assertIn(self.implementation.TOKENSPEED_LAUNCH_SLICE_SHA256, notice)
        self.assertIn(KERNEL_SHA256, notice)
        self.assertIn("Permission is hereby granted, free of charge", notice)
        self.assertIn("Apache License", notice)


def function(tree: ast.AST, name: str) -> ast.FunctionDef:
    return next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


if __name__ == "__main__":
    unittest.main()

import ast
import importlib.metadata
import importlib.util
import math
import sys
import types
import unittest
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATION = ROOT / "src/infer/models/glm53_flash/ops/prefill_kda.py"
LICENSE = ROOT / "src/infer/models/glm53_flash/ops/prefill_kda.LICENSE.txt"


@dataclass(frozen=True)
class FakeDevice:
    type: str
    index: int | None = None

    def __str__(self) -> str:
        return self.type if self.index is None else f"{self.type}:{self.index}"


@dataclass
class FakeTensor:
    shape: tuple[int, ...]
    dtype: object
    device: FakeDevice
    contiguous: bool = True
    pointer: int = 256
    values: tuple[int, ...] = ()
    _version: int = 0

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def is_contiguous(self) -> bool:
        return self.contiguous

    def numel(self) -> int:
        return math.prod(self.shape)

    def data_ptr(self) -> int:
        return self.pointer

    def __getitem__(self, index: slice) -> "FakeTensor":
        if not isinstance(index, slice):
            raise TypeError("fake tensor only supports slices")
        start = 0 if index.start is None else index.start
        stop = self.shape[0] if index.stop is None else index.stop
        return FakeTensor(
            (stop - start,),
            self.dtype,
            self.device,
            self.contiguous,
            self.pointer + start,
        )

    def to(self, *, device: FakeDevice, dtype: object) -> "FakeTensor":
        return FakeTensor(self.shape, dtype, device, values=self.values)

    def view(self, *shape: int) -> "FakeTensor":
        if math.prod(shape) != self.numel():
            raise ValueError("invalid fake view")
        return FakeTensor(tuple(shape), self.dtype, self.device, self.contiguous)


class RecordingExtension(types.ModuleType):
    def __init__(self) -> None:
        super().__init__("flash_kda_C")
        self.fwd_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
        self.workspace_calls: list[tuple[int, int, int]] = []
        self.fwd = self._fwd
        self.get_workspace_size = self._get_workspace_size

    def _fwd(self, *args: object, **kwargs: object) -> None:
        self.fwd_calls.append((args, kwargs))

    def _get_workspace_size(self, tokens: int, heads: int, batch: int) -> int:
        self.workspace_calls.append((tokens, heads, batch))
        tiles = (tokens + 15) // 16 + batch
        tile_bytes = 3 * 16 * 128 * 2 + 128 * 4 + 2 * 16 * 16 * 2
        prefix_bytes = ((batch + 1) * 4 + 127) // 128 * 128
        return heads * tiles * tile_bytes + prefix_bytes + tokens * heads * 2


def load_implementation(
    installed_version: str = "0.0.1+1ce47ea.infer1",
) -> tuple[types.ModuleType, RecordingExtension]:
    torch = types.ModuleType("torch")
    torch.Tensor = FakeTensor
    torch.dtype = object
    torch.device = FakeDevice
    torch.bfloat16 = object()
    torch.float32 = object()
    torch.int32 = object()
    torch.int64 = object()
    torch.uint8 = object()

    def as_device(value: FakeDevice | str) -> FakeDevice:
        if isinstance(value, FakeDevice):
            return value
        kind, separator, index = value.partition(":")
        return FakeDevice(kind, int(index) if separator else None)

    def tensor(
        values: tuple[int, ...],
        *,
        dtype: object,
        device: str,
    ) -> FakeTensor:
        return FakeTensor(
            (len(values),),
            dtype,
            as_device(device),
            values=tuple(values),
        )

    torch.device = as_device
    torch.tensor = tensor
    torch.inference_mode = lambda enabled=True: nullcontext()
    torch.iinfo = lambda dtype: types.SimpleNamespace(max=2**31 - 1)

    extension = RecordingExtension()
    spec = importlib.util.spec_from_file_location(
        "prefill_kda",
        IMPLEMENTATION,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load GLM prefill KDA adapter")
    module = importlib.util.module_from_spec(spec)
    with (
        patch.dict(sys.modules, {"torch": torch, "flash_kda_C": extension}),
        patch.object(
            importlib.metadata,
            "version",
            return_value=installed_version,
        ),
    ):
        spec.loader.exec_module(module)
    return module, extension


class GLM53PrefillMetadataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.implementation, _ = load_implementation()

    def test_owns_fresh_content_validated_descriptors_per_plan(self) -> None:
        module = self.implementation
        first = module.GLM53PrefillMetadata((0, 3, 8), "cuda:2")
        second = module.GLM53PrefillMetadata((0, 3, 8), "cuda:2")

        self.assertIsNot(first.cu_seqlens, second.cu_seqlens)
        self.assertIsNot(first.cu_seqlens_int64, second.cu_seqlens_int64)
        self.assertEqual(first.cu_seqlens.dtype, module.torch.int32)
        self.assertEqual(first.cu_seqlens_int64.dtype, module.torch.int64)
        self.assertEqual(first.cu_seqlens.device, FakeDevice("cuda", 2))
        self.assertEqual(first.cu_seqlens_int64.device, FakeDevice("cuda", 2))
        self.assertEqual(first.cu_seqlens_cpu.values, (0, 3, 8))
        self.assertEqual(first.batch_size, 2)
        self.assertEqual(first.total_tokens, 8)

    def test_rejects_invalid_cumulative_lengths_before_gpu_descriptors(self) -> None:
        cases = (
            ((), "at least one sequence"),
            ((1, 2), "start at zero"),
            ((0, 2, 2), "at least one token"),
            ((0, True), "must be an int"),
        )
        for values, message in cases:
            with (
                self.subTest(values=values),
                self.assertRaisesRegex((TypeError, ValueError), message),
            ):
                self.implementation.GLM53PrefillMetadata(values, "cuda")

    def test_detects_each_descriptor_mutation(self) -> None:
        for name in ("cu_seqlens", "cu_seqlens_int64", "cu_seqlens_cpu"):
            metadata = self.implementation.GLM53PrefillMetadata((0, 1), "cuda")
            getattr(metadata, name)._version += 1
            with (
                self.subTest(name=name),
                self.assertRaisesRegex(RuntimeError, "create a new instance"),
            ):
                metadata.validate_unchanged()


class GLM53PrefillKDATest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.implementation, cls.extension = load_implementation()

    def setUp(self) -> None:
        self.extension.fwd_calls.clear()
        self.extension.workspace_calls.clear()

    def test_uses_caller_owned_raw_extension_contract(self) -> None:
        inputs = self._valid_inputs()

        output, final_state = self.implementation.glm53_prefill_kda(**inputs)

        self.assertIs(output, inputs["out"])
        self.assertIs(final_state, inputs["final_state"])
        self.assertEqual(self.extension.workspace_calls, [])
        self.assertEqual(len(self.extension.fwd_calls), 1)
        args, kwargs = self.extension.fwd_calls[0]
        self.assertEqual(len(args), 11)
        self.assertIs(args[0], inputs["q"])
        self.assertIs(args[4], inputs["beta_logits"])
        self.assertEqual(args[5], 128**-0.5)
        self.assertIs(args[6], inputs["out"])
        self.assertIs(args[7], inputs["workspace"])
        self.assertIs(args[8], inputs["a_log"])
        self.assertEqual(args[9].shape, (16, 128))
        self.assertEqual(args[10], -5.0)
        self.assertIs(kwargs["initial_state"], inputs["initial_state"])
        self.assertIs(kwargs["final_state"], inputs["final_state"])
        self.assertIs(kwargs["cu_seqlens"], inputs["metadata"].cu_seqlens_int64)

    def test_constructs_aligned_workspace_after_one_raw_abi_check(self) -> None:
        module = self.implementation
        required = 16 * (((5 + 15) // 16) + 2) * 13_824 + 128 + 5 * 16 * 2
        storage = FakeTensor(
            (required + 127,),
            module.torch.uint8,
            FakeDevice("cuda", 0),
            pointer=257,
        )

        workspace = module.make_glm53_prefill_kda_workspace(storage, 5, 2)

        self.assertEqual(self.extension.workspace_calls, [(5, 16, 2)])
        self.assertEqual(workspace.shape, (required,))
        self.assertEqual(workspace.data_ptr() % 128, 0)

    def test_rejects_unpinned_extension_build(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "0.0.1\\+1ce47ea.infer1"):
            load_implementation("0.0.1+different")

    def test_rejects_contract_changes_before_raw_launch(self) -> None:
        cases = (
            ("beta_logits", "contiguous", False, "contiguous"),
            (
                "final_state",
                "dtype",
                self.implementation.torch.bfloat16,
                "dtype",
            ),
            ("workspace", "shape", (1,), "at least"),
            ("workspace", "pointer", 257, "128-byte aligned"),
        )
        for name, attribute, value, message in cases:
            inputs = self._valid_inputs()
            setattr(inputs[name], attribute, value)
            with (
                self.subTest(name=name),
                self.assertRaisesRegex((TypeError, ValueError), message),
            ):
                self.implementation.glm53_prefill_kda(**inputs)

        inputs = self._valid_inputs()
        inputs["metadata"].cu_seqlens_int64.dtype = self.implementation.torch.uint8
        with self.assertRaisesRegex(TypeError, "cu_seqlens_int64 must have dtype"):
            self.implementation.glm53_prefill_kda(**inputs)

        self.assertEqual(self.extension.fwd_calls, [])

    def test_rejects_workspace_abi_drift(self) -> None:
        module = self.implementation
        required = 16 * (((5 + 15) // 16) + 2) * 13_824 + 128 + 5 * 16 * 2
        storage = FakeTensor(
            (required + 127,),
            module.torch.uint8,
            FakeDevice("cuda", 0),
        )
        with (
            patch.object(module, "_get_workspace_size", return_value=1),
            self.assertRaisesRegex(RuntimeError, "workspace ABI differs"),
        ):
            module.make_glm53_prefill_kda_workspace(storage, 5, 2)

    def _valid_inputs(self) -> dict[str, FakeTensor]:
        module = self.implementation
        device = FakeDevice("cuda", 0)
        token_shape = (1, 5, 16, 128)
        state_shape = (2, 16, 128, 128)
        metadata = module.GLM53PrefillMetadata((0, 2, 5), device)
        workspace_bytes = self.extension._get_workspace_size(5, 16, 2)
        self.extension.workspace_calls.clear()
        return {
            "q": FakeTensor(token_shape, module.torch.bfloat16, device),
            "k": FakeTensor(token_shape, module.torch.bfloat16, device),
            "v": FakeTensor(token_shape, module.torch.bfloat16, device),
            "gate": FakeTensor(token_shape, module.torch.bfloat16, device),
            "beta_logits": FakeTensor(
                (1, 5, 16),
                module.torch.bfloat16,
                device,
            ),
            "a_log": FakeTensor((16,), module.torch.float32, device),
            "dt_bias": FakeTensor((2048,), module.torch.float32, device),
            "initial_state": FakeTensor(state_shape, module.torch.float32, device),
            "metadata": metadata,
            "out": FakeTensor(token_shape, module.torch.bfloat16, device),
            "final_state": FakeTensor(state_shape, module.torch.float32, device),
            "workspace": FakeTensor(
                (workspace_bytes,),
                module.torch.uint8,
                device,
            ),
        }


class GLM53PrefillKDASourceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = IMPLEMENTATION.read_text()
        cls.tree = ast.parse(cls.source)

    def test_pins_direct_extension_without_production_dispatch(self) -> None:
        imports = {
            (node.module, alias.name)
            for node in ast.walk(self.tree)
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
        }

        self.assertIn(("flash_kda_C", "fwd"), imports)
        self.assertIn(("flash_kda_C", "get_workspace_size"), imports)
        self.assertNotIn(("flash_kda", "fwd"), imports)
        self.assertNotIn("BackendRegistry", self.source)
        self.assertNotIn("torch.empty", self.source)
        production = (ROOT / "src/infer/models/glm53_flash/ops/core.py").read_text()
        self.assertIn("glm53_prefill_kda", production)
        self.assertNotIn("ChunkKDAFunction", production)

    def test_hot_path_does_not_query_the_native_workspace_size(self) -> None:
        hot_path = next(
            node
            for node in self.tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "glm53_prefill_kda"
        )
        constructor = next(
            node
            for node in self.tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "make_glm53_prefill_kda_workspace"
        )

        self.assertNotIn("_get_workspace_size", ast.unparse(hot_path))
        self.assertIn("_get_workspace_size", ast.unparse(constructor))

    def test_documents_caller_owned_beta_transpose(self) -> None:
        function = next(
            node
            for node in self.tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "glm53_prefill_kda"
        )
        docstring = ast.get_docstring(function)

        self.assertIn("caller owns", docstring)
        self.assertIn("transposed BF16 beta tail", docstring)

    def test_records_exact_dependency_and_license_provenance(self) -> None:
        notice = LICENSE.read_text()

        self.assertIn("1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b", notice)
        self.assertIn(
            "f736ab87c7c2e47fe416f8d04d62502a257a6ff32a0b23452c07e1bc65c87c04",
            notice,
        )
        self.assertIn(
            "27c80df97cfa24d8b4a5bebc53e6e8cc906c16afc1ed7509cf9da9af9424e464",
            notice,
        )
        self.assertIn(
            "93382056dd78a5f1990f47656e3baaebe6b52b66d490345882d0e50a04b88c24",
            notice,
        )
        self.assertIn("5c149f52a436782210263fb2f19b354443a61c6a", notice)
        self.assertIn("Copyright (c) 2026 MoonshotAI", notice)
        self.assertIn("NVIDIA CORPORATION & AFFILIATES", notice)
        self.assertIn("SPDX-License-Identifier: MIT AND BSD-3-Clause", notice)


if __name__ == "__main__":
    unittest.main()

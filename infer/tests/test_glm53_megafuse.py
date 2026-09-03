import ast
import hashlib
import importlib.util
import inspect
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from infer.models.glm53_flash import model as glm53

ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATION = ROOT / "src/infer/models/glm53_flash/ops/megafuse.py"
LICENSE = ROOT / "src/infer/models/glm53_flash/ops/megafuse.LICENSE.txt"
KERNEL_SHA256 = "8cba6db8936eb80336154794271122928fae9d898bb94ec44b7501f2536d3f40"
VERIFY_SOURCE_SHA256 = (
    "a4f3425b1df8b50fbcccd14fe42df0edd60fc610e15c9274232bd5ebee5ca2ae"
)


def load_implementation() -> types.ModuleType:
    torch = types.ModuleType("torch")
    torch.bfloat16 = object()
    torch.float32 = object()
    torch.int32 = object()
    triton = types.ModuleType("triton")
    language = types.ModuleType("triton.language")

    def decorator(*args: object, **kwargs: object) -> object:
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return lambda function: function

    triton.heuristics = decorator
    triton.jit = decorator
    triton.cdiv = lambda value, divisor: (value + divisor - 1) // divisor
    triton.language = language
    modules = {"torch": torch, "triton": triton, "triton.language": language}
    spec = importlib.util.spec_from_file_location("megafuse", IMPLEMENTATION)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load megafuse module")
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module


class FakeTensor:
    def __init__(
        self,
        shape: tuple[int, ...],
        dtype: object,
        device: object,
        *,
        values: tuple[int, ...] = (),
    ) -> None:
        self.shape = shape
        self.ndim = len(shape)
        self.dtype = dtype
        self.device = device
        self.values = values
        strides = []
        stride = 1
        for size in reversed(shape):
            strides.append(stride)
            stride *= size
        self._strides = tuple(reversed(strides))

    def is_contiguous(self) -> bool:
        return True

    def stride(self, dimension: int | None = None) -> int | tuple[int, ...]:
        if dimension is None:
            return self._strides
        return self._strides[dimension]


class FakeKernel:
    def __init__(self) -> None:
        self.grid: tuple[int, ...] | None = None
        self.keywords: dict[str, object] | None = None

    def __getitem__(self, grid: tuple[int, ...]):
        self.grid = grid

        def launch(**keywords: object) -> None:
            self.keywords = keywords

        return launch


def verify_tensors(
    implementation: types.ModuleType,
    state_indices: tuple[int, ...],
    attention_tp_size: int = glm53.TP_SIZE,
) -> dict[str, FakeTensor]:
    groups = len(state_indices)
    rows = groups * implementation.GLM53_TARGET_VERIFY_WIDTH
    local_heads = glm53.glm53_local_heads(attention_tp_size)
    local_projection_size = local_heads * implementation.HEAD_DIM
    packed_qkv_size = 3 * local_projection_size
    pages = max((index for index in state_indices if index >= 0), default=0) + 1
    device = types.SimpleNamespace(type="cuda")
    bf16 = implementation.torch.bfloat16
    float32 = implementation.torch.float32
    int32 = implementation.torch.int32
    return {
        "qkv_raw": FakeTensor((rows, packed_qkv_size), bf16, device),
        "conv_weight": FakeTensor(
            (packed_qkv_size, implementation.CONV_KERNEL_SIZE),
            float32,
            device,
        ),
        "conv_state": FakeTensor(
            (
                pages,
                packed_qkv_size,
                implementation.CONV_KERNEL_SIZE - 1,
            ),
            bf16,
            device,
        ),
        "conv_tape": FakeTensor(
            (
                rows,
                packed_qkv_size,
                implementation.CONV_KERNEL_SIZE - 1,
            ),
            bf16,
            device,
        ),
        "f_a": FakeTensor((rows, implementation.HEAD_DIM), bf16, device),
        "f_b_weight": FakeTensor(
            (local_projection_size, implementation.HEAD_DIM),
            bf16,
            device,
        ),
        "gate_raw": FakeTensor((rows, local_projection_size), float32, device),
        "beta_logits": FakeTensor((rows, local_heads), bf16, device),
        "a_log": FakeTensor((local_heads,), float32, device),
        "dt_bias": FakeTensor((local_projection_size,), float32, device),
        "recurrent_state": FakeTensor(
            (
                pages,
                local_heads,
                implementation.HEAD_DIM,
                implementation.HEAD_DIM,
            ),
            float32,
            device,
        ),
        "recurrent_tape": FakeTensor(
            (
                rows,
                local_heads,
                implementation.HEAD_DIM,
                implementation.HEAD_DIM,
            ),
            float32,
            device,
        ),
        "state_indices": FakeTensor((groups,), int32, device, values=state_indices),
        "out": FakeTensor(
            (rows, local_heads, implementation.HEAD_DIM),
            bf16,
            device,
        ),
    }


class MegafusePlanTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.implementation = load_implementation()

    def test_accepts_active_prefix_and_graph_padding(self) -> None:
        active = self.implementation.validate_glm53_megafuse_plan(
            [4, 5, -1, -1],
            [0, 1, 2, 2, 2],
            state_pages=6,
        )

        self.assertEqual(active, 2)

    def test_accepts_active_only(self) -> None:
        active = self.implementation.validate_glm53_megafuse_plan(
            [2, 0], [0, 1, 2], state_pages=3
        )

        self.assertEqual(active, 2)

    def test_rejects_non_prefix_padding_and_non_unit_delta(self) -> None:
        with self.assertRaisesRegex(ValueError, "follows graph padding"):
            self.implementation.validate_glm53_megafuse_plan(
                [0, -1, 1], [0, 1, 1, 2], state_pages=5
            )
        with self.assertRaisesRegex(ValueError, "must be zero or one"):
            self.implementation.validate_glm53_megafuse_plan([0], [0, 2], state_pages=3)

    def test_rejects_unsafe_active_state_indices(self) -> None:
        cases = (
            ([-1], "state_indices"),
            ([2], "state_indices"),
            ([0, 0], "must be unique"),
        )
        for indices, message in cases:
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(ValueError, message),
            ):
                self.implementation.validate_glm53_megafuse_plan(
                    indices,
                    list(range(len(indices) + 1)),
                    state_pages=2,
                )

    def test_rejects_non_sentinel_padding_indices(self) -> None:
        with self.assertRaisesRegex(ValueError, "padded state_indices must be -1"):
            self.implementation.validate_glm53_megafuse_plan(
                [0, 1], [0, 1, 1], state_pages=3
            )


class MegafuseVerifyBatchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.implementation = load_implementation()

    def test_launches_g1_g2_and_g4_request_major_groups(self) -> None:
        for attention_tp_size in (glm53.TP_SIZE, 1):
            for state_indices in ((4,), (4, 1), (7, 3, -1, -1)):
                with self.subTest(
                    attention_tp_size=attention_tp_size,
                    state_indices=state_indices,
                ):
                    tensors = verify_tensors(
                        self.implementation, state_indices, attention_tp_size
                    )
                    local_heads = tensors["a_log"].shape[0]
                    gate_kernel = FakeKernel()
                    kernel = FakeKernel()
                    self.implementation.glm53_verify_gate_precompute_kernel = (
                        gate_kernel
                    )
                    self.implementation.fused_recurrent_kda_verify_megafuse_fwd_kernel = kernel

                    result = self.implementation.glm53_megafuse_verify(**tensors)

                    groups = len(state_indices)
                    rows = groups * self.implementation.GLM53_TARGET_VERIFY_WIDTH
                    self.assertIs(result, tensors["out"])
                    self.assertEqual(tensors["qkv_raw"].shape[0], rows)
                    self.assertEqual(tensors["conv_tape"].shape[0], rows)
                    self.assertEqual(tensors["recurrent_tape"].shape[0], rows)
                    self.assertEqual(tensors["out"].shape[0], rows)
                    expected_grid = (
                        groups * local_heads * self.implementation.HEAD_DIM // 32,
                    )
                    self.assertEqual(gate_kernel.grid, expected_grid)
                    if gate_kernel.keywords is None:
                        self.fail("verify gate kernel was not launched")
                    self.assertIs(gate_kernel.keywords["gate_raw"], tensors["gate_raw"])
                    self.assertEqual(gate_kernel.keywords["H"], local_heads)
                    self.assertEqual(kernel.grid, expected_grid)
                    if kernel.keywords is None:
                        self.fail("verify kernel was not launched")
                    self.assertIs(kernel.keywords["gate_raw"], tensors["gate_raw"])
                    self.assertNotIn("f_a", kernel.keywords)
                    self.assertNotIn("w_fb", kernel.keywords)
                    self.assertIs(
                        kernel.keywords["read_indices"], tensors["state_indices"]
                    )
                    self.assertEqual(
                        kernel.keywords["T"],
                        self.implementation.GLM53_TARGET_VERIFY_WIDTH,
                    )
                    self.assertEqual(kernel.keywords["H"], local_heads)
                    self.assertEqual(kernel.keywords["HV"], local_heads)
                    self.assertEqual(
                        kernel.keywords["P"],
                        local_heads * self.implementation.HEAD_DIM,
                    )

    def test_rejects_ambiguous_or_inconsistent_group_shapes(self) -> None:
        width = self.implementation.GLM53_TARGET_VERIFY_WIDTH
        rows = 2 * width
        local_heads = glm53.glm53_local_heads(glm53.TP_SIZE)
        local_projection_size = local_heads * self.implementation.HEAD_DIM
        packed_qkv_size = 3 * local_projection_size
        cases = (
            (
                "state_indices",
                (2, 1),
                "state_indices must have rank 1",
            ),
            ("state_indices", (0,), "verify groups must be positive"),
            (
                "qkv_raw",
                (rows - 1, packed_qkv_size),
                rf"qkv_raw must have shape \({rows}, {packed_qkv_size}\)",
            ),
            (
                "conv_tape",
                (2, width, packed_qkv_size, 3),
                rf"conv_tape must have shape \({rows}, {packed_qkv_size}, 3\)",
            ),
            (
                "recurrent_tape",
                (rows - 1, local_heads, 128, 128),
                rf"recurrent_tape must have shape \({rows}, {local_heads}, 128, 128\)",
            ),
            (
                "gate_raw",
                (rows - 1, local_projection_size),
                rf"gate_raw must have shape \({rows}, {local_projection_size}\)",
            ),
            (
                "out",
                (rows - 1, local_heads, 128),
                rf"out must have shape \({rows}, {local_heads}, 128\)",
            ),
        )
        for name, shape, message in cases:
            with self.subTest(name=name, shape=shape):
                tensors = verify_tensors(self.implementation, (5, 2))
                original = tensors[name]
                tensors[name] = FakeTensor(
                    shape,
                    original.dtype,
                    original.device,
                    values=original.values,
                )

                with self.assertRaisesRegex(ValueError, message):
                    self.implementation._validate_verify_tensors(**tensors)

    def test_requires_fp32_gate_scratch(self) -> None:
        tensors = verify_tensors(self.implementation, (5, 2))
        gate_raw = tensors["gate_raw"]
        tensors["gate_raw"] = FakeTensor(
            gate_raw.shape,
            self.implementation.torch.bfloat16,
            gate_raw.device,
        )

        with self.assertRaisesRegex(TypeError, "gate_raw must have dtype"):
            self.implementation._validate_verify_tensors(**tensors)


class MegafuseSourceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = IMPLEMENTATION.read_text()
        cls.tree = ast.parse(cls.source)
        cls.implementation = load_implementation()

    def test_pins_origin_and_exact_upstream_kernel_slice(self) -> None:
        self.assertEqual(
            self.implementation.TOKENSPEED_COMMIT,
            "24ff5ab3dfd42aaea8f8f7476d4293f6c2717ed3",
        )
        self.assertEqual(
            self.implementation.TOKENSPEED_GIT_BLOB_SHA,
            "10e461ab01013ce6837804b4eea8d57b2b8ae7b1",
        )
        self.assertEqual(
            self.implementation.TOKENSPEED_SOURCE_PATH,
            "tokenspeed-kernel/python/tokenspeed_kernel/thirdparty/triton/"
            "fla_kda_recurrent.py",
        )
        self.assertEqual(
            self.implementation.TOKENSPEED_SOURCE_SHA256,
            "e4ac7c46c534dc4403d3f7d204c05dd3ca0675ce6a835dfadb1a99d35d080f7f",
        )
        kernel = function(self.tree, "fused_recurrent_kda_megafuse_fwd_kernel")
        start = min(decorator.lineno for decorator in kernel.decorator_list)
        block = "".join(
            self.source.splitlines(keepends=True)[start - 1 : kernel.end_lineno]
        )

        self.assertEqual(hashlib.sha256(block.encode()).hexdigest(), KERNEL_SHA256)
        verify = function(
            self.tree,
            "fused_recurrent_kda_verify_megafuse_fwd_kernel",
        )
        verify_start = min(decorator.lineno for decorator in verify.decorator_list)
        verify_block = "".join(
            self.source.splitlines(keepends=True)[verify_start - 1 : verify.end_lineno]
        )
        self.assertEqual(
            hashlib.sha256(verify_block.encode()).hexdigest(),
            VERIFY_SOURCE_SHA256,
        )

    def test_public_api_and_geometry_are_topology_aware(self) -> None:
        signature = inspect.signature(self.implementation.glm53_megafuse_decode)

        self.assertEqual(
            list(signature.parameters),
            [
                "qkv_raw",
                "conv_weight",
                "conv_state",
                "f_a",
                "f_b_weight",
                "beta_logits",
                "a_log",
                "dt_bias",
                "output_gate",
                "norm_weight",
                "recurrent_state",
                "state_indices",
                "cu_seqlens",
                "out",
            ],
        )
        self.assertEqual(self.implementation.HEAD_DIM, 128)
        self.assertEqual(self.implementation.CONV_KERNEL_SIZE, 4)
        self.assertEqual(self.implementation.GATE_LOWER_BOUND, -5.0)
        self.assertEqual(self.implementation.RMS_NORM_EPS, 1e-5)
        for name in ("glm53_megafuse_decode", "glm53_megafuse_verify"):
            source = ast.unparse(function(self.tree, name))
            self.assertIn("local_heads = a_log.shape[0]", source)
            self.assertIn("local_projection_size = local_heads * HEAD_DIM", source)
            self.assertIn("H=local_heads", source)
            self.assertIn("HV=local_heads", source)
            self.assertIn("P=local_projection_size", source)
        verify = inspect.signature(self.implementation.glm53_megafuse_verify)
        self.assertIn("gate_raw", verify.parameters)
        self.assertEqual(
            list(verify.parameters)[-3:],
            ["recurrent_tape", "state_indices", "out"],
        )

    def test_wrapper_uses_caller_output_and_fixed_fused_features(self) -> None:
        wrapper = function(self.tree, "glm53_megafuse_decode")
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
            and call.func.value.id == "fused_recurrent_kda_megafuse_fwd_kernel"
        )
        keywords = {keyword.arg: keyword.value for keyword in launch.keywords}

        self.assertEqual(torch_calls, [])
        self.assertIsInstance(wrapper.body[-1], ast.Return)
        self.assertEqual(ast.unparse(wrapper.body[-1].value), "out")
        self.assertEqual(ast.unparse(keywords["o"]), "out")
        self.assertEqual(ast.unparse(keywords["gate"]), "output_gate")
        self.assertEqual(ast.unparse(keywords["norm_w"]), "norm_weight")
        self.assertEqual(ast.unparse(keywords["w_fb"]), "f_b_weight")
        self.assertEqual(ast.unparse(keywords["read_indices"]), "state_indices")
        self.assertEqual(ast.unparse(keywords["write_indices"]), "state_indices")
        self.assertEqual(ast.unparse(keywords["lower_bound"]), "GATE_LOWER_BOUND")
        self.assertEqual(ast.literal_eval(keywords["T"]), 1)
        self.assertEqual(ast.unparse(keywords["H"]), "local_heads")
        self.assertEqual(ast.unparse(keywords["HV"]), "local_heads")
        self.assertEqual(ast.unparse(keywords["P"]), "local_projection_size")

    def test_padding_returns_before_index_or_state_access(self) -> None:
        kernel = function(self.tree, "fused_recurrent_kda_megafuse_fwd_kernel")
        source = ast.get_source_segment(self.source, kernel)
        if source is None:
            raise RuntimeError("kernel source was unavailable")

        self.assertLess(
            source.index("if T == 0:\n        return"),
            source.index("b_read = tl.load(read_indices"),
        )

    def test_verify_wrapper_uses_fixed_width_and_caller_output(self) -> None:
        wrapper = function(self.tree, "glm53_megafuse_verify")
        source = ast.get_source_segment(self.source, wrapper)
        calls = [node for node in ast.walk(wrapper) if isinstance(node, ast.Call)]

        self.assertFalse(
            any(
                isinstance(call.func, ast.Attribute)
                and isinstance(call.func.value, ast.Name)
                and call.func.value.id == "torch"
                for call in calls
            )
        )
        self.assertIn("T=GLM53_TARGET_VERIFY_WIDTH", source)
        self.assertIn("local_heads = a_log.shape[0]", source)
        self.assertIn("local_projection_size = local_heads * HEAD_DIM", source)
        self.assertIn("H=local_heads", source)
        self.assertIn("HV=local_heads", source)
        self.assertIn("P=local_projection_size", source)
        self.assertLess(
            source.index("glm53_verify_gate_precompute_kernel"),
            source.index("fused_recurrent_kda_verify_megafuse_fwd_kernel"),
        )
        self.assertEqual(ast.unparse(wrapper.body[-1].value), "out")

    def test_verify_gate_precompute_preserves_fp32_reduction(self) -> None:
        kernel = function(self.tree, "glm53_verify_gate_precompute_kernel")
        source = ast.get_source_segment(self.source, kernel)
        if source is None:
            raise RuntimeError("verify gate kernel source was unavailable")

        self.assertIn(").to(tl.float32)", source)
        self.assertIn("i_k = pid % NK", source)
        self.assertIn("i_n, i_hv = i_nh // H, i_nh % H", source)
        self.assertNotIn("LOCAL_HEADS", source)
        self.assertIn("o_k = i_k * BK + tl.arange(0, BK)", source)
        self.assertIn("b_g = tl.sum(wfb * fa[None, :], axis=1)", source)
        self.assertIn("tl.store(gate_raw", source)

    def test_verify_kernel_routes_each_request_to_its_rows_and_state(self) -> None:
        kernel = function(self.tree, "fused_recurrent_kda_verify_megafuse_fwd_kernel")
        source = ast.get_source_segment(self.source, kernel)
        if source is None:
            raise RuntimeError("verify kernel source was unavailable")

        self.assertIn("i_n, i_hv = i_nh // HV, i_nh % HV", source)
        self.assertIn("b_read = tl.load(read_indices + i_n)", source)
        self.assertIn("bos = i_n * T", source)
        self.assertIn("row = bos + i_t", source)
        for access in (
            "qkv_raw + row * stride_raw_tok",
            "conv_out + row * stride_conv_out_page",
            "gate_raw + row * stride_gate_tok",
            "beta + row * stride_beta_tok",
            "o + (row * HV + i_hv) * V",
            "h_pool_out + row * stride_state_out_page",
        ):
            with self.subTest(access=access):
                self.assertIn(access, source)

    def test_verify_padding_returns_before_request_data_access(self) -> None:
        kernel = function(self.tree, "fused_recurrent_kda_verify_megafuse_fwd_kernel")
        source = ast.get_source_segment(self.source, kernel)
        if source is None:
            raise RuntimeError("verify kernel source was unavailable")

        state_index = source.index("b_read = tl.load(read_indices + i_n)")
        padding_return = source.index("if b_read < 0:\n        return")
        self.assertLess(state_index, padding_return)
        for access in (
            "qkv_raw + row * stride_raw_tok",
            "conv_pool + b_read * stride_conv_page",
            "conv_out + row * stride_conv_out_page",
            "gate_raw + row * stride_gate_tok",
            "beta + row * stride_beta_tok",
            "o + (row * HV + i_hv) * V",
            "h_pool + b_read * stride_state_page",
            "h_pool_out + row * stride_state_out_page",
        ):
            with self.subTest(access=access):
                self.assertLess(padding_return, source.index(access))

    def test_has_no_tokenspeed_or_fla_runtime_import(self) -> None:
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

    def test_license_preserves_both_mit_attributions(self) -> None:
        notice = LICENSE.read_text()

        self.assertIn("Copyright (c) 2026 LightSeek Foundation", notice)
        self.assertIn(
            "Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li",
            notice,
        )
        self.assertIn(self.implementation.TOKENSPEED_COMMIT, notice)
        self.assertIn(self.implementation.TOKENSPEED_GIT_BLOB_SHA, notice)
        self.assertIn(self.implementation.TOKENSPEED_SOURCE_SHA256, notice)
        self.assertIn(KERNEL_SHA256, notice)
        self.assertIn(VERIFY_SOURCE_SHA256, notice)
        self.assertIn("2e38c1fab332174d056928feaf29f8c5fd5ac550", notice)
        self.assertIn("Permission is hereby granted, free of charge", notice)


def function(tree: ast.AST, name: str) -> ast.FunctionDef:
    return next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


if __name__ == "__main__":
    unittest.main()

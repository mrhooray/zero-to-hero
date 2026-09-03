import ast
import importlib.util
import inspect
import sys
import types
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATION = ROOT / "src/infer/models/glm53_flash/ops/sparse_mla_prefill.py"


def load_implementation() -> types.ModuleType:
    torch = types.ModuleType("torch")
    for name in ("bfloat16", "int32", "int64", "uint8"):
        setattr(torch, name, object())
    torch.Tensor = object
    torch.index_select = _fake_index_select

    triton = types.ModuleType("triton")
    language = types.ModuleType("triton.language")
    triton.jit = lambda function: function
    triton.language = language

    model = types.ModuleType("infer.models.glm53_flash.model")
    constants = {
        "KDA_CHUNK_SIZE": 4096,
        "SPARSE_MLA_COMPOUND_BLOCKS": 8192,
        "SPARSE_MLA_HISTORY_BLOCK_TOKENS": 128,
        "SPARSE_MLA_INDEX_PAGE_BYTES": 4224,
        "SPARSE_MLA_INDEX_POOL_TOKENS": 4,
        "SPARSE_MLA_INDEXER_HEAD_DIM": 128,
        "SPARSE_MLA_KV_LORA_RANK": 512,
        "SPARSE_MLA_MAX_CONTEXT_TOKENS": 1_048_576,
        "SparseMLAHistory": object,
    }
    for name, value in constants.items():
        setattr(model, name, value)

    spec = importlib.util.spec_from_file_location(
        "sparse_mla_prefill", IMPLEMENTATION
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load sparse MLA prefill module")
    module = importlib.util.module_from_spec(spec)
    modules = {
        "torch": torch,
        "triton": triton,
        "triton.language": language,
        "infer.models.glm53_flash.model": model,
        spec.name: module,
    }
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module


class SparseMLAPrefillPlanTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.implementation = load_implementation()

    def validate(self, start: int, tokens: int, blocks: tuple[int, ...]):
        return self.implementation.validate_glm53_sparse_mla_prefill_plan(
            start,
            tokens,
            0,
            blocks,
            history_blocks=max(blocks, default=-1) + 2,
            live_slots=1,
            has_initial=start > 0,
        )

    def test_accepts_fresh_resumed_and_block_boundary_plans(self) -> None:
        cases = (
            (0, 1, (0,)),
            (0, 4, (0,)),
            (1, 3, (0,)),
            (127, 2, (0, 1)),
        )
        for start, tokens, blocks in cases:
            with self.subTest(start=start, tokens=tokens):
                plan = self.validate(start, tokens, blocks)
                self.assertEqual((plan.start_token, plan.total_tokens), (start, tokens))

    def test_rejects_invalid_ownership_and_lengths(self) -> None:
        cases = (
            (
                (0, 1, 0, (1,)),
                {"history_blocks": 3, "live_slots": 1, "has_initial": True},
                ValueError,
                "only resumed",
            ),
            (
                (0, 129, 0, (0,)),
                {"history_blocks": 3, "live_slots": 1, "has_initial": False},
                ValueError,
                "2 live mappings",
            ),
            (
                (0, 129, 0, (0, 0)),
                {"history_blocks": 3, "live_slots": 1, "has_initial": False},
                ValueError,
                "must be unique",
            ),
            (
                (0, 1, 1, (0,)),
                {"history_blocks": 2, "live_slots": 1, "has_initial": False},
                ValueError,
                "state_slot",
            ),
            (
                (True, 1, 0, (0,)),
                {"history_blocks": 2, "live_slots": 1, "has_initial": False},
                TypeError,
                "start_token",
            ),
        )
        for args, kwargs, error, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(error, message):
                self.implementation.validate_glm53_sparse_mla_prefill_plan(
                    *args, **kwargs
                )


@dataclass
class FakeDevice:
    type: str = "cuda"


@dataclass
class FakeTensor:
    shape: tuple[int, ...]
    dtype: object
    device: FakeDevice
    contiguous: bool = True
    version: int = 0
    selected: tuple[object, object] | None = None

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def is_cuda(self) -> bool:
        return self.device.type == "cuda"

    @property
    def _version(self) -> int:
        return self.version

    def is_contiguous(self) -> bool:
        return self.contiguous

    def zero_(self):
        self.version += 1
        return self


def _fake_index_select(source, dimension, indices, *, out) -> None:
    if dimension != 0:
        raise AssertionError("packed tables select rows")
    out.selected = source, indices
    out.version += 1


class SparseMLAPackedStagingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.implementation = load_implementation()

    def tensor(self, shape, dtype) -> FakeTensor:
        return FakeTensor(shape, dtype, FakeDevice())

    def stage(self, plans=None, **changes):
        implementation = self.implementation
        torch = implementation.torch
        if plans is None:
            plans = (
                implementation.validate_glm53_sparse_mla_prefill_plan(
                    0,
                    2,
                    0,
                    None,
                    history_blocks=4,
                    live_slots=2,
                    has_initial=False,
                ),
                implementation.validate_glm53_sparse_mla_prefill_plan(
                    17,
                    3,
                    1,
                    None,
                    history_blocks=4,
                    live_slots=2,
                    has_initial=True,
                ),
            )
        tensors = {
            "cu_seqlens": self.tensor((len(plans) + 1,), torch.int32),
            "start_tokens": self.tensor((len(plans),), torch.int32),
            "state_slots": self.tensor((len(plans),), torch.int64),
            "sequence_ids": self.tensor((4096,), torch.int32),
            "token_state_slots": self.tensor((4096,), torch.int64),
            "active": self.tensor((4096,), torch.uint8),
            "raw_lengths": self.tensor((4096,), torch.int32),
            "execution_tables": self.tensor((2, 8192), torch.int32),
            "block_table": self.tensor((4096, 8192), torch.int32),
            "null_token": self.tensor((1,), torch.int32),
        }
        tensors.update(changes)
        receipt = implementation.stage_glm53_sparse_mla_prefill_batch(
            plans=plans, **tensors
        )
        return receipt, tensors

    def test_stages_one_fixed_arena_and_binds_active_views(self) -> None:
        receipt, tensors = self.stage()

        self.assertEqual(receipt.total_tokens, 5)
        self.assertEqual(receipt.batch_size, 2)
        self.assertEqual(receipt.history_blocks, 4)
        self.assertEqual(receipt.live_slots, 2)
        self.assertEqual(
            receipt.block_table.selected,
            (tensors["execution_tables"], tensors["token_state_slots"]),
        )
        receipt.validate_unchanged()
        receipt.active.zero_()
        with self.assertRaisesRegex(RuntimeError, "changed after staging"):
            receipt.validate_unchanged()

    def test_rejects_over_capacity_mixed_pools_and_bad_abi_shape(self) -> None:
        implementation = self.implementation
        oversized = (
            implementation.validate_glm53_sparse_mla_prefill_plan(
                0,
                4097,
                0,
                None,
                history_blocks=34,
                live_slots=1,
                has_initial=False,
            ),
        )
        with self.assertRaisesRegex(ValueError, "token arena"):
            self.stage(oversized)

        first = implementation.validate_glm53_sparse_mla_prefill_plan(
            0,
            1,
            0,
            None,
            history_blocks=4,
            live_slots=2,
            has_initial=False,
        )
        second = implementation.validate_glm53_sparse_mla_prefill_plan(
            0,
            1,
            0,
            None,
            history_blocks=5,
            live_slots=2,
            has_initial=False,
        )
        with self.assertRaisesRegex(ValueError, "share one state pool"):
            self.stage((first, second))

        bad = self.tensor((4095, 8192), implementation.torch.int32)
        with self.assertRaisesRegex(ValueError, "shape or dtype"):
            self.stage(block_table=bad)


class SparseMLAPrefillSourceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = IMPLEMENTATION.read_text()
        cls.tree = ast.parse(cls.source)

    def test_only_packed_production_contract_remains(self) -> None:
        self.assertNotIn("GLM53StagedSparseMLAPrefillQuery", self.source)
        self.assertNotIn("GLM53StagedSparseMLAPrefillPlan", self.source)
        self.assertNotIn("_append_sparse_mla_latent", self.source)
        signature = inspect.signature(
            load_implementation().glm53_sparse_mla_packed_prefill_append
        )
        self.assertEqual(
            list(signature.parameters),
            ["batch", "latent", "key", "gate", "pool_ape", "history"],
        )

    def test_append_orders_latent_index_and_tail_without_allocating(self) -> None:
        wrapper = function(self.tree, "glm53_sparse_mla_packed_prefill_append")
        source = ast.unparse(wrapper)
        launches = (
            "_append_packed_sparse_mla_latent",
            "_append_packed_sparse_mla_index",
            "_store_packed_sparse_mla_tail",
        )
        positions = [source.index(name) for name in launches]
        self.assertEqual(positions, sorted(positions))
        self.assertLess(source.index("validate_unchanged"), positions[0])
        self.assertEqual(ast.unparse(wrapper.body[-1].value), "history")
        for allocation in ("empty", "zeros", "tensor", "arange", "clone", "item"):
            self.assertNotIn(f".{allocation}(", source)

    def test_pooling_matches_decode_writer_precision_order(self) -> None:
        kernel = function(self.tree, "_append_packed_sparse_mla_index")
        source = ast.get_source_segment(self.source, kernel)
        if source is None:
            raise RuntimeError("kernel source was unavailable")
        ordered = (
            "tl.softmax(logits, dim=0).to(tl.bfloat16)",
            "(probability * keys).to(tl.bfloat16).to(tl.float32)",
            "product_0 + product_1",
            "pooled = pooled + product_2",
            "pooled = pooled + product_3",
            "tl.max(tl.abs(pooled), axis=0)",
            ".to(tl.float8e4nv)",
        )
        positions = [source.index(fragment) for fragment in ordered]
        self.assertEqual(positions, sorted(positions))

    def test_resumed_pool_reads_tail_and_each_sequence_stores_its_suffix(self) -> None:
        index = ast.get_source_segment(
            self.source, function(self.tree, "_append_packed_sparse_mla_index")
        )
        tail = ast.get_source_segment(
            self.source, function(self.tree, "_store_packed_sparse_mla_tail")
        )
        if index is None or tail is None:
            raise RuntimeError("kernel source was unavailable")
        self.assertIn("position >= chunk_start", index)
        self.assertIn("tail_key_ptr + old_tail", index)
        self.assertIn("sequence = program // 4", tail)
        self.assertIn("last_position - (last_position + 4 - lane) % 4", tail)
        self.assertIn("tl.static_range(2)", tail)


def function(tree: ast.AST, name: str) -> ast.FunctionDef:
    return next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


if __name__ == "__main__":
    unittest.main()

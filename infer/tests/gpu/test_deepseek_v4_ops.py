import ast
import importlib
import inspect
import symtable
import sys
import types
import unittest
from dataclasses import replace
from enum import IntEnum
from pathlib import Path
from unittest.mock import call, patch

import pytest

pytestmark = pytest.mark.gpu

import torch

from infer.models.deepseek_v4_flash import model


class _TieBreak(IntEnum):
    SMALL = 3


class _TopK:
    def __init__(self) -> None:
        self.calls = []

    def radix_topk_ragged_transform(self, *args, **kwargs) -> None:
        self.calls.append((args, kwargs))
        selected = args[1]
        selected.fill_(-1)
        for row, length in enumerate(args[3]):
            count = min(int(length), model.INDEX_TOP_K)
            selected[row, :count].copy_(torch.arange(count, dtype=torch.int32))

    def radix_topk_page_table_transform(self, *args, **kwargs) -> None:
        self.calls.append((args, kwargs))
        output, table, lengths = args[1], args[2], args[4]
        page_size = kwargs["page_size"]
        output.fill_(-1)
        for row, length in enumerate(lengths):
            count = min(int(length), model.INDEX_TOP_K)
            selected = torch.arange(count, dtype=torch.int32)
            pages = table[row, selected.div(page_size, rounding_mode="floor")]
            output[row, :count].copy_(pages * page_size + selected.remainder(page_size))


class _KernelLaunch:
    def __init__(self) -> None:
        self.grid = None
        self.args = None
        self.kwargs = None
        self.calls = []

    def __getitem__(self, grid):
        self.grid = grid
        return self

    def __call__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs
        self.calls.append((self.grid, args, kwargs))


def load_ops_module():
    deep_gemm = types.ModuleType("deep_gemm")
    deep_gemm.__file__ = "/fake/deep_gemm/__init__.py"
    deep_gemm.metadata_calls = []
    deep_gemm.logit_calls = []
    deep_gemm.pdl = None
    deep_gemm.get_num_sms = lambda: 8
    deep_gemm.set_pdl = lambda enabled: setattr(deep_gemm, "pdl", enabled)

    def metadata_out(*args, **kwargs):
        deep_gemm.metadata_calls.append((args, kwargs))

    def logits_out(*args, **kwargs):
        deep_gemm.logit_calls.append((args, kwargs))

    deep_gemm.get_paged_mqa_logits_metadata_out = metadata_out
    deep_gemm.fp8_fp4_paged_mqa_logits_out = logits_out
    deep_gemm.mega_moe_pre_dispatch = lambda *args, **kwargs: None
    deep_gemm.fp8_fp4_mega_moe = lambda *args, **kwargs: None

    flash_mla = types.ModuleType("flash_mla")
    flash_mla.__path__ = []
    flash_mla.calls = []
    flash_mla.sparse_calls = []
    flash_mla_cuda = types.ModuleType("flash_mla.cuda")
    flash_mla_cuda.__file__ = "/fake/flash_mla/cuda.so"
    flash_mla.cuda = flash_mla_cuda

    def attention(
        *,
        q,
        k_cache,
        block_table,
        cache_seqlens,
        head_dim_v,
        tile_scheduler_metadata,
        num_splits,
        softmax_scale,
        causal,
        is_fp8_kvcache,
        indices,
        attn_sink,
        extra_k_cache,
        extra_indices_in_kvcache,
        topk_length,
        extra_topk_length,
    ):
        flash_mla.calls.append(locals())
        batch_size = q.shape[0]
        return (
            torch.empty(batch_size, 1, 64, 512),
            torch.empty(batch_size, 64, 1),
        )

    flash_mla.flash_mla_with_kvcache = attention

    def sparse_attention(**kwargs):
        flash_mla.sparse_calls.append(kwargs)
        size = kwargs["q"].shape[0]
        return (
            torch.empty(size, 64, 512, dtype=torch.bfloat16),
            torch.empty(size, 64),
            torch.empty(size, 64),
        )

    flash_mla.flash_mla_sparse_fwd = sparse_attention

    topk = _TopK()
    flashinfer = types.ModuleType("flashinfer")
    flashinfer.__path__ = []
    flashinfer.__version__ = "0.6.18.dev20260819"
    flashinfer.__git_commit__ = "61a6c651872a7d3f2f6dcc1ced61633d8f8ba3dd"
    flashinfer_jit = types.ModuleType("flashinfer.jit")
    flashinfer_jit.__path__ = []
    flashinfer_jit_topk = types.ModuleType("flashinfer.jit.topk")
    flashinfer_jit_topk.gen_topk_module = lambda: types.SimpleNamespace(
        is_aot=True, get_library_path=lambda: "/fake/flashinfer/topk.so"
    )
    flashinfer_topk = types.ModuleType("flashinfer.topk")
    flashinfer_topk.TopKTieBreak = _TieBreak
    flashinfer_topk.get_topk_module = lambda: topk
    triton = types.ModuleType("triton")
    triton.__path__ = []
    triton.jit = lambda function=None, **_: (
        (lambda decorated: decorated) if function is None else function
    )
    triton.next_power_of_2 = lambda value: 1 << (value - 1).bit_length()
    triton_language = types.ModuleType("triton.language")
    triton_language.__path__ = []
    triton_language.constexpr = object()
    triton_language.maximum = lambda value, floor: value.clamp_min(floor)
    triton_language_extra = types.ModuleType("triton.language.extra")
    triton_language_extra.libdevice = types.SimpleNamespace()
    modules = {
        "deep_gemm": deep_gemm,
        "flash_mla": flash_mla,
        "flash_mla.cuda": flash_mla_cuda,
        "flashinfer": flashinfer,
        "flashinfer.jit": flashinfer_jit,
        "flashinfer.jit.topk": flashinfer_jit_topk,
        "flashinfer.topk": flashinfer_topk,
        "triton": triton,
        "triton.language": triton_language,
        "triton.language.extra": triton_language_extra,
    }
    with patch.dict(sys.modules, modules):
        sys.modules.pop("infer.models.deepseek_v4_flash.ops.core", None)
        ops = importlib.import_module("infer.models.deepseek_v4_flash.ops.core")
    return ops, deep_gemm, topk


def compression(batch_size, width, ratio, cache_stride):
    state_page = 4 if ratio == 4 else 8
    norm_width = 128 if width == 256 else 512
    return model.DeepSeekV4Compression(
        kv_score=torch.empty(batch_size, 2 * width),
        ape=torch.empty(ratio, width),
        state=torch.empty(1, state_page, 2 * width),
        norm_weight=torch.empty(norm_width, dtype=torch.bfloat16),
        cache=torch.empty(1, cache_stride, dtype=torch.uint8),
    )


def prefill_state(compressed, persistent_rows=2):
    trailing = compressed.state.shape[1:]
    indices = torch.empty(0, dtype=torch.int64)
    return model.DeepSeekV4PrefillState(
        persistent=torch.empty(persistent_rows, *trailing),
        seed_source=indices,
        seed_destination=indices,
        retain_source=indices,
        retain_destination=indices,
        transfer=torch.empty(0, *trailing),
    )


def common_inputs(batch_size=1):
    batch = model.DeepSeekV4CompressionBatch(
        positions=torch.zeros(batch_size, dtype=torch.int32),
        request_indices=torch.arange(batch_size, dtype=torch.int32),
        state_slots=torch.zeros(batch_size, dtype=torch.int32),
        output_slots=torch.zeros(batch_size, dtype=torch.int32),
        state_table=torch.zeros(batch_size, 1, dtype=torch.int32),
        state_table_base=torch.zeros(batch_size, dtype=torch.int32),
        cos_sin=torch.empty(1, 64),
    )
    raw = model.DeepSeekV4RawAttention(
        query=torch.empty(batch_size, 1, 64, 512, dtype=torch.bfloat16),
        cache=torch.empty(1, 37_440, dtype=torch.uint8),
        indices=torch.zeros(batch_size, 1, 128, dtype=torch.int32),
        lengths=torch.full((batch_size,), 128, dtype=torch.int32),
        sink=torch.empty(64),
    )
    return batch, raw


def prefill_batch(token_count=5, request_count=2):
    request_indices = torch.arange(1, token_count + 1, dtype=torch.int32)
    request_indices.mul_(request_count).sub_(1)
    request_indices.div_(max(token_count, 1), rounding_mode="floor")
    return model.DeepSeekV4CompressionBatch(
        positions=torch.arange(token_count, dtype=torch.int32),
        request_indices=request_indices,
        state_slots=torch.arange(token_count, dtype=torch.int32),
        output_slots=torch.arange(token_count, dtype=torch.int32),
        state_table=torch.zeros(request_count, 2, dtype=torch.int32),
        state_table_base=torch.zeros(request_count, dtype=torch.int32),
        cos_sin=torch.empty(max(token_count, 1), 64),
    )


def prefill_metadata(token_count=5, prefixes=(3, 129)):
    requests = len(prefixes)
    query_start = torch.arange(requests + 1, dtype=torch.int32).mul_(token_count)
    query_start.div_(requests, rounding_mode="floor")
    return model.DeepSeekV4PrefillMetadata(
        query_start=query_start,
        prefix_lengths=torch.tensor(prefixes, dtype=torch.int32),
        state_slots=torch.arange(requests, dtype=torch.int32),
        block_table=torch.zeros(requests, 4, dtype=torch.int32),
        block_table_base=torch.zeros(requests, dtype=torch.int32),
    )


def sparse_prefill_inputs(token_count=1, selected_width=128, device=None):
    return (
        torch.empty(token_count, 64, 512, dtype=torch.bfloat16, device=device),
        torch.empty(1, 1, 512, dtype=torch.bfloat16, device=device),
        torch.empty(token_count, 1, selected_width, dtype=torch.int32, device=device),
        torch.empty(token_count, dtype=torch.int32, device=device),
        torch.empty(64, dtype=torch.float32, device=device),
    )


def prefill_workspace(token_count, request_count, compressed_capacity):
    rows = request_count * (compressed_capacity + model.SWA_WINDOW_TOKENS)
    rows += token_count
    return model.DeepSeekV4PrefillWorkspace(
        kv=torch.empty(rows, 1, 512, dtype=torch.bfloat16),
        raw_slots=torch.empty(token_count, dtype=torch.int32),
    )


def prefill_index_workspace(table_width=4, device=None):
    rows = 512
    capacity = table_width * 32
    aligned_capacity = (capacity + 255) // 256 * 256
    staged = model.DeepSeekV4IndexQuery(
        query=torch.empty(rows, 1, 64, 64, dtype=torch.int8, device=device),
        scale=torch.empty(rows, 1, 64, dtype=torch.int32, device=device),
        weights=torch.empty(rows, 64, device=device),
        block_table=torch.empty(rows, table_width, dtype=torch.int32, device=device),
        lengths=torch.zeros(rows, 1, dtype=torch.int32, device=device),
    )
    logits = torch.empty(rows, aligned_capacity, device=device)
    return model.DeepSeekV4PrefillIndexWorkspace(
        staged=staged,
        request_indices=torch.zeros(rows, dtype=torch.int32, device=device),
        schedule=torch.empty(9, 2, dtype=torch.int32, device=device),
        logits=logits,
        candidates=torch.empty(
            model.PREFILL_CHUNK_TOKENS,
            model.INDEX_TOP_K,
            dtype=torch.int32,
            device=device,
        ),
        topk_offsets=torch.zeros(rows, dtype=torch.int32, device=device),
        topk_rows=torch.zeros(
            model.TOPK_ROW_STATES_BYTES, dtype=torch.uint8, device=device
        ),
    )


def c4_inputs(block_table, lengths=None):
    batch_size = block_table.shape[0]
    if lengths is None:
        lengths = torch.full((batch_size, 1), 512, dtype=torch.int32)
    query = model.DeepSeekV4IndexQuery(
        query=torch.empty(batch_size, 1, 64, 64, dtype=torch.int8),
        scale=torch.empty(batch_size, 1, 64, dtype=torch.int32),
        weights=torch.empty(batch_size, 64),
        block_table=block_table,
        lengths=lengths,
    )
    workspace = model.DeepSeekV4DecodeWorkspace(
        schedule=torch.empty(9, 2, dtype=torch.int32),
        logits=torch.empty(batch_size, model.INDEX_CONTEXT_CAPACITY),
        topk_rows=torch.zeros(model.TOPK_ROW_STATES_BYTES, dtype=torch.uint8),
        mapped_c4=torch.empty(batch_size, 1, model.INDEX_TOP_K, dtype=torch.int32),
        selected_lengths=torch.empty(batch_size, dtype=torch.int32),
    )
    return query, workspace


class DeepSeekV4OpsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module, cls.deep_gemm, cls.topk = load_ops_module()

    def new_ops(self):
        versions = {
            **self.module.FLASHINFER_PACKAGES,
            "sgl-deep-gemm": "0.1.5.post3",
            "triton": "3.7.1",
        }
        with (
            patch.object(self.module, "version", side_effect=versions.__getitem__),
            patch.object(self.module.torch, "__version__", "2.13.0+cu129"),
            patch.object(self.module.torch.version, "cuda", "12.9"),
            patch.object(
                self.module,
                "_native_binary_hashes",
                return_value=self.module.NATIVE_BINARY_SHA256,
            ),
        ):
            return self.module.DeepSeekV4Ops()

    def test_pins_direct_native_abis_and_requires_aot_topk(self) -> None:
        ops = self.new_ops()

        self.assertEqual(ops._num_sms, 8)
        self.assertIs(self.deep_gemm.pdl, True)
        self.assertEqual(
            self.module.FLASH_MLA_COMMIT, "15f13e5030374295491c5ce31b02d7e63a7772c6"
        )
        self.assertEqual(
            self.module.NATIVE_BINARY_SHA256["flash_mla"],
            "6fa8a4d34c6461a17725d972411434ac32d3248847ef63d5ce4507b59e3d94b2",
        )
        with (
            patch.object(self.module, "version", return_value="changed"),
            self.assertRaisesRegex(RuntimeError, "identity mismatch"),
        ):
            self.module.DeepSeekV4Ops()

    def test_hash_route_normalization_clamps_an_underflowed_sum(self) -> None:
        scores = torch.zeros(6, dtype=torch.float32)
        normalized = self.module._normalize_hash_route_weight(
            scores, scores.sum(), 1.5, model.ROUTER_DENOMINATOR_MIN
        )
        self.assertTrue(torch.isfinite(normalized).all())
        self.assertTrue(torch.equal(normalized, scores))

        scores.fill_(1.0e-30)
        normalized = self.module._normalize_hash_route_weight(
            scores, scores.sum(), 1.5, model.ROUTER_DENOMINATOR_MIN
        )
        self.assertTrue(torch.equal(normalized, scores * (1.5 / 1.0e-20)))

    def test_c4_packed_projection_copies_dense_raw_query(self) -> None:
        packed = torch.arange(2 * 40_960, dtype=torch.float32).view(2, 40_960)
        dense = torch.empty(2, model.NUM_QUERY_HEADS, model.HEAD_DIM)

        self.module._copy_c4_raw_query(packed, dense)

        self.assertTrue(torch.equal(dense.view(2, -1), packed[:, :32_768]))
        self.assertTrue(dense.is_contiguous())

    def test_compressor_save_reads_one_packed_projection_buffer(self) -> None:
        save = inspect.getsource(self.module._save_state)
        self.assertIn("projected = kv_score + token * 2 * STATE_WIDTH", save)
        self.assertIn("values = tl.load(projected + offsets", save)
        self.assertIn("score = tl.load(projected + STATE_WIDTH + offsets", save)
        self.assertNotIn("scores,", save)

    def test_c4_components_compose_scorer_topk_mapping_and_attention(self) -> None:
        ops = self.new_ops()
        self.deep_gemm.metadata_calls.clear()
        self.deep_gemm.logit_calls.clear()
        self.topk.calls.clear()
        batch, raw = common_inputs()
        main = compression(1, 1_024, 4, 19_008)
        index = compression(1, 256, 4, 2_176)
        table = torch.arange(16, dtype=torch.int32).flip(0).view(1, -1).contiguous()
        query, workspace = c4_inputs(table)
        output = object()
        lse = object()

        with (
            patch.object(self.module, "_save") as save,
            patch.object(self.module, "_compress_main") as compress_main,
            patch.object(self.module, "_compress_index") as compress_index,
            patch.object(
                self.module, "_attention", return_value=(output, lse)
            ) as attention,
            patch.object(self.module, "_require_cuda_types"),
        ):
            ops.validate_c4(
                batch, main, index, raw, query, workspace, model.LOCAL_QUERY_HEADS
            )
            ops.compress_c4_main(batch, main)
            ops.compress_c4_index(batch, index)
            ops.prepare_c4(query, workspace)
            ops.select_c4(index, query, workspace)
            actual = ops.attend_c4(
                raw, main, workspace, "even", model.LOCAL_QUERY_HEADS
            )

        self.assertEqual(actual, (output, lse))
        self.assertEqual(save.call_count, 2)
        compress_main.assert_called_once_with(batch, main, 4, True, 32, 4)
        compress_index.assert_called_once_with(batch, index)
        self.assertEqual(len(self.deep_gemm.metadata_calls), 1)
        self.assertEqual(len(self.deep_gemm.logit_calls), 1)
        self.assertEqual(len(self.topk.calls), 1)
        topk_args, topk_kwargs = self.topk.calls[0]
        self.assertIs(topk_args[1]._base, workspace.mapped_c4)
        self.assertIs(topk_args[2], query.block_table)
        self.assertEqual(topk_args[6:9], (512, True, int(_TieBreak.SMALL)))
        self.assertEqual(
            topk_kwargs,
            {
                "page_size": 32,
                "dsa_graph_safe": True,
                "row_starts": None,
                "page_table_row_starts": None,
                "output_raw_indices": None,
            },
        )
        self.assertEqual(self.deep_gemm.logit_calls[0][0][7], 262_144)
        selected = torch.arange(512, dtype=torch.int32)
        expected = (15 - selected.div(32, rounding_mode="floor")) * 32
        expected.add_(selected.remainder(32))
        self.assertTrue(torch.equal(workspace.mapped_c4.view(-1), expected))
        attention.assert_called_once_with(
            raw,
            main.cache,
            workspace.mapped_c4,
            workspace.selected_lengths,
            "even",
            32,
            model.LOCAL_QUERY_HEADS,
        )

    def test_prefill_compresses_one_prefix_tail_plus_chunk_batch(self) -> None:
        ops = self.new_ops()
        batch = prefill_batch()
        c4_main = compression(5, 1_024, 4, 19_008)
        c4_index = compression(5, 256, 4, 2_176)
        c128_main = compression(5, 512, 128, 1_152)
        c4_main_state = prefill_state(c4_main)
        c4_index_state = prefill_state(c4_index)
        c128_state = prefill_state(c128_main)

        with (
            patch.object(self.module, "_save") as save,
            patch.object(self.module, "_compress_main") as compress_main,
            patch.object(self.module, "_compress_index") as compress_index,
            patch.object(self.module, "_require_cuda_types"),
        ):
            ops.prefill_compress_c4(
                batch, c4_main, c4_index, c4_main_state, c4_index_state
            )
            self.assertEqual(
                save.call_args_list,
                [
                    call(batch, c4_main, 4, True, 4),
                    call(batch, c4_index, 4, True, 4),
                ],
            )
            compress_main.assert_called_once_with(batch, c4_main, 4, True, 32, 4)
            compress_index.assert_called_once_with(batch, c4_index)

            save.reset_mock()
            compress_main.reset_mock()
            ops.prefill_compress_c128(batch, c128_main, c128_state)
            save.assert_called_once_with(batch, c128_main, 128, False, 8)
            compress_main.assert_called_once_with(batch, c128_main, 128, False, 1, 16)

    def test_prefill_compressor_rejects_empty_or_oversized_chunks(self) -> None:
        with patch.object(self.module, "_require_cuda_types"):
            self.assertEqual(self.module._validate_prefill_batch(prefill_batch()), 5)
            self.assertEqual(
                self.module._validate_prefill_batch(
                    prefill_batch(model.PREFILL_CHUNK_TOKENS)
                ),
                model.PREFILL_CHUNK_TOKENS,
            )
            for token_count in (0, model.PREFILL_CHUNK_TOKENS + 1):
                with (
                    self.subTest(token_count=token_count),
                    self.assertRaisesRegex(ValueError, "prefill compressor"),
                ):
                    self.module._validate_prefill_batch(
                        prefill_batch(token_count, request_count=1)
                    )

    def test_prefill_compressor_seeds_scratch_and_retains_final_pages(self) -> None:
        compressed = compression(1, 512, 128, 1_152)
        compressed = replace(
            compressed, state=torch.zeros(2, 8, 1_024, dtype=torch.float32)
        )
        persistent = torch.zeros(2, 8, 1_024, dtype=torch.float32)
        persistent[1].fill_(7)
        state = model.DeepSeekV4PrefillState(
            persistent,
            torch.tensor((1,), dtype=torch.int64),
            torch.tensor((0,), dtype=torch.int64),
            torch.tensor((1,), dtype=torch.int64),
            torch.tensor((0,), dtype=torch.int64),
            torch.empty(1, 8, 1_024, dtype=torch.float32),
        )

        def run(_batch, value):
            self.assertTrue(torch.equal(value.state[0], persistent[1]))
            value.state[1].fill_(9)

        with (
            patch.object(self.module, "_require_cuda_types"),
            patch.object(self.module, "_run_c128_compression", side_effect=run),
        ):
            self.new_ops().prefill_compress_c128(prefill_batch(1, 1), compressed, state)

        self.assertTrue(torch.equal(persistent[0], torch.full_like(persistent[0], 9)))

    def test_prefill_request_group_is_bounded(self) -> None:
        prefixes = (0,) * (model.MAX_PREFILL_REQUESTS + 1)
        with self.assertRaisesRegex(ValueError, "prefill metadata"):
            self.module._validate_prefill_metadata(prefill_metadata(5, prefixes))

    def test_prefill_c4_candidates_use_one_fixed_shape_across_packed_chunks(
        self,
    ) -> None:
        ops = self.new_ops()
        self.deep_gemm.metadata_calls.clear()
        self.deep_gemm.logit_calls.clear()
        self.topk.calls.clear()
        batch = prefill_batch(515, 2)
        batch = replace(
            batch,
            positions=torch.cat(
                (torch.arange(512, dtype=torch.int32), torch.arange(509, 512))
            ),
            request_indices=torch.cat(
                (torch.zeros(512, dtype=torch.int32), torch.ones(3, dtype=torch.int32))
            ),
        )
        metadata = replace(
            prefill_metadata(515, (0, 509)),
            query_start=torch.tensor((0, 512, 515), dtype=torch.int32),
            block_table_base=torch.tensor((0, 2), dtype=torch.int32),
        )
        projected = torch.empty(515, 40_960, dtype=torch.bfloat16)
        weights = torch.empty(515, 64, dtype=torch.bfloat16)
        workspace = prefill_index_workspace()
        index_cache = torch.empty(1, 2_176, dtype=torch.uint8)
        query_launch = _KernelLaunch()
        metadata_launch = _KernelLaunch()

        with (
            patch.object(self.module, "_require_cuda_types"),
            patch.object(self.module, "_require_tensor"),
            patch.object(self.module, "_c4_query", query_launch),
            patch.object(self.module, "_stage_prefill_index_kernel", metadata_launch),
        ):
            result = ops.prefill_c4_candidates(
                metadata, batch, index_cache, projected, weights, workspace
            )

        self.assertEqual(result.shape, (515, model.INDEX_TOP_K))
        self.assertEqual(result.data_ptr(), workspace.candidates.data_ptr())
        self.assertEqual(
            [(grid, args[7:10]) for grid, args, _ in metadata_launch.calls],
            [((512, 4), (0, 512, 4)), ((512, 4), (512, 3, 4))],
        )
        self.assertEqual(
            [grid for grid, _, _ in query_launch.calls],
            [(512 * 64, 4), (3 * 64, 4)],
        )
        self.assertEqual(len(self.deep_gemm.metadata_calls), 2)
        self.assertEqual(len(self.deep_gemm.logit_calls), 2)
        self.assertEqual(len(self.topk.calls), 2)
        for args, kwargs in self.deep_gemm.metadata_calls:
            self.assertEqual(args[2:], (32, 8))
            self.assertIs(args[0], workspace.staged.lengths)
            self.assertEqual(kwargs, {"indices": workspace.request_indices})
        for args, kwargs in self.deep_gemm.logit_calls:
            self.assertEqual(args[1].stride(), (2_176, 68, 68, 1))
            self.assertEqual(args[7], 256)
            self.assertEqual(
                kwargs,
                {"clean_logits": False, "indices": workspace.request_indices},
            )
        outputs = [args[1] for args, _ in self.topk.calls]
        self.assertEqual([output.shape for output in outputs], [(512, 512)] * 2)
        self.assertEqual(outputs[1].data_ptr() - outputs[0].data_ptr(), 512 * 512 * 4)

        producer = inspect.getsource(self.module.DeepSeekV4Ops.prefill_c4_candidates)
        self.assertNotIn("torch.empty", producer)
        staging = inspect.getsource(self.module._stage_prefill_index_kernel)
        self.assertIn(
            'do_not_specialize=["token_start", "token_count", "table_width"]', staging
        )

    def test_prefill_staging_accepts_cold_and_resumed_requests(self) -> None:
        query_start = torch.tensor((0, 512, 515), dtype=torch.int32)
        metadata = model.DeepSeekV4PrefillMetadata(
            query_start=query_start,
            prefix_lengths=torch.tensor((0, 509), dtype=torch.int32),
            state_slots=torch.tensor((3, 7), dtype=torch.int32),
            block_table=torch.tensor(
                ((11, 12, 13, 14), (21, 22, 23, 24)), dtype=torch.int32
            ),
            block_table_base=torch.tensor((0, 2), dtype=torch.int32),
        )
        positions = torch.cat(
            (
                torch.arange(512, dtype=torch.int32),
                torch.arange(509, 512, dtype=torch.int32),
            )
        )
        batch = model.DeepSeekV4CompressionBatch(
            positions=positions,
            request_indices=torch.cat(
                (torch.zeros(512, dtype=torch.int32), torch.ones(3, dtype=torch.int32))
            ),
            state_slots=torch.empty(515, dtype=torch.int32),
            output_slots=torch.empty(515, dtype=torch.int32),
            state_table=torch.empty(2, 132, dtype=torch.int32),
            state_table_base=torch.empty(2, dtype=torch.int32),
            cos_sin=torch.empty(512, 64),
        )
        current_kv = torch.empty(515, 512, dtype=torch.bfloat16)
        raw_cache = torch.empty(16, 37_440, dtype=torch.uint8)
        raw = model.DeepSeekV4RawAttention(
            query=torch.empty(515, 1, 64, 512, dtype=torch.bfloat16),
            cache=raw_cache,
            indices=torch.empty(515, 1, 640, dtype=torch.int32),
            lengths=torch.empty(515, dtype=torch.int32),
            sink=torch.empty(64),
        )
        compressed_cache = torch.empty(32, 19_008, dtype=torch.uint8)
        local_candidates = torch.arange(512, dtype=torch.int32).expand(515, -1)
        workspace = prefill_workspace(515, 2, 128)
        launch = _KernelLaunch()

        with (
            patch.object(self.module, "_require_cuda_types"),
            patch.object(self.module, "_stage_prefill_attention_kernel", launch),
        ):
            self.new_ops().stage_prefill_attention(
                metadata,
                batch,
                current_kv,
                raw,
                compressed_cache,
                local_candidates,
                workspace,
                4,
            )

        self.assertEqual(launch.grid, (1_027, 4))
        self.assertIs(launch.args[11], local_candidates)
        self.assertEqual(
            launch.args[16:],
            (515, 2, 128, 640, 4, 1_027, 19_008, 37_440),
        )
        self.assertEqual(
            launch.kwargs,
            {
                "RATIO": 4,
                "COMPRESSED_PAGE": 32,
                "RAW_RING_PAGES": 3,
                "num_warps": 4,
            },
        )
        staging = inspect.getsource(self.module._stage_prefill_attention_kernel)
        self.assertIn("raw_ring = RAW_RING_PAGES * 64", staging)
        self.assertIn(
            "tl.load(state_slots + request) * raw_ring + position % raw_ring",
            staging,
        )
        self.assertNotIn("* 128 + position % 128", staging)

        with (
            patch.object(self.module, "_require_cuda_types"),
            self.assertRaisesRegex(ValueError, "prefill staging"),
        ):
            self.new_ops().stage_prefill_attention(
                metadata,
                batch,
                current_kv,
                raw,
                compressed_cache,
                local_candidates,
                prefill_workspace(515, 2, 129),
                4,
            )

    def test_sparse_prefill_attention_uses_the_pinned_flashmla_abi(self) -> None:
        ops = self.new_ops()
        query, kv, indices, lengths, sink = sparse_prefill_inputs(7, 640)
        self.module.flash_mla.sparse_calls.clear()

        with patch.object(self.module, "_require_cuda_types"):
            output = ops.prefill_selected_attention(query, kv, indices, lengths, sink)

        self.assertEqual(output.shape, (7, 64, 512))
        self.assertEqual(
            self.module.flash_mla.sparse_calls,
            [
                {
                    "q": query,
                    "kv": kv,
                    "indices": indices,
                    "sm_scale": 512**-0.5,
                    "d_v": 512,
                    "attn_sink": sink,
                    "topk_length": lengths,
                }
            ],
        )

    def test_prefill_staging_accepts_max_context_capacities(self) -> None:
        batch = prefill_batch(1, 1)
        for ratio, page, cache_stride, index_width in (
            (4, 32, 19_008, 640),
            (128, 1, 1_152, 8_320),
        ):
            with self.subTest(ratio=ratio):
                capacity = model.MAX_CONTEXT_TOKENS // ratio
                metadata = model.DeepSeekV4PrefillMetadata(
                    query_start=torch.empty(2, dtype=torch.int32, device="meta"),
                    prefix_lengths=torch.empty(1, dtype=torch.int32, device="meta"),
                    state_slots=torch.empty(1, dtype=torch.int32, device="meta"),
                    block_table=torch.empty(
                        1, capacity // page, dtype=torch.int32, device="meta"
                    ),
                    block_table_base=torch.empty(1, dtype=torch.int32, device="meta"),
                )
                raw = model.DeepSeekV4RawAttention(
                    query=torch.empty(
                        1, 1, 64, 512, dtype=torch.bfloat16, device="meta"
                    ),
                    cache=torch.empty(1, 37_440, dtype=torch.uint8, device="meta"),
                    indices=torch.empty(
                        1, 1, index_width, dtype=torch.int32, device="meta"
                    ),
                    lengths=torch.empty(1, dtype=torch.int32, device="meta"),
                    sink=torch.empty(64, device="meta"),
                )
                workspace = model.DeepSeekV4PrefillWorkspace(
                    kv=torch.empty(capacity + 129, 1, 512, device="meta"),
                    raw_slots=torch.empty(1, dtype=torch.int32, device="meta"),
                )

                with patch.object(self.module, "_require_cuda_types"):
                    actual = self.module._validate_prefill_staging(
                        metadata,
                        batch,
                        torch.empty(1, 512, dtype=torch.bfloat16, device="meta"),
                        raw,
                        torch.empty(1, cache_stride, dtype=torch.uint8, device="meta"),
                        (
                            torch.empty(1, 512, dtype=torch.int32, device="meta")
                            if ratio == 4
                            else None
                        ),
                        workspace,
                        ratio,
                    )

                self.assertEqual(actual, capacity)

    def test_sparse_prefill_attention_requires_fixed_deepseek_geometry(self) -> None:
        valid = sparse_prefill_inputs()
        with patch.object(self.module, "_require_cuda_types"):
            self.module._validate_prefill_attention(*valid)
            self.module._validate_prefill_attention(
                *sparse_prefill_inputs(model.PREFILL_CHUNK_TOKENS, device="meta")
            )
            for values in (
                sparse_prefill_inputs(model.PREFILL_CHUNK_TOKENS + 1, device="meta"),
                (*valid[:1], torch.empty(0, 1, 512), *valid[2:]),
                (*valid[:2], torch.empty(1, 1, 64, dtype=torch.int32), *valid[3:]),
            ):
                with (
                    self.subTest(shapes=[tensor.shape for tensor in values]),
                    self.assertRaisesRegex(ValueError, "sparse prefill"),
                ):
                    self.module._validate_prefill_attention(*values)

    def test_c4_ignores_negative_topk_tail_for_short_and_padded_rows(self) -> None:
        ops = self.new_ops()
        size = 8
        batch, raw = common_inputs(size)
        main = compression(size, 1_024, 4, 19_008)
        index = compression(size, 256, 4, 2_176)
        table = torch.arange(17, dtype=torch.int32).expand(size, -1).contiguous()
        lengths = torch.tensor(
            (0, 1, 31, 511, 512, 513, 5, 128), dtype=torch.int32
        ).view(-1, 1)
        query, workspace = c4_inputs(table, lengths)

        with (
            patch.object(self.module, "_save"),
            patch.object(self.module, "_compress_main"),
            patch.object(self.module, "_compress_index"),
            patch.object(self.module, "_attention", return_value=(1, 2)),
            patch.object(self.module, "_require_cuda_types"),
        ):
            ops.validate_c4(
                batch, main, index, raw, query, workspace, model.LOCAL_QUERY_HEADS
            )
            ops.compress_c4_main(batch, main)
            ops.compress_c4_index(batch, index)
            ops.prepare_c4(query, workspace)
            ops.select_c4(index, query, workspace)
            self.assertEqual(
                ops.attend_c4(raw, main, workspace, "even", model.LOCAL_QUERY_HEADS),
                (1, 2),
            )

        self.assertTrue(
            torch.equal(
                workspace.mapped_c4[0],
                torch.full((1, model.INDEX_TOP_K), -1, dtype=torch.int32),
            )
        )
        for row, count in enumerate((0, 1, 31, 511, 512, 512, 5, 128)):
            self.assertTrue(
                torch.equal(
                    workspace.mapped_c4[row, 0, :count],
                    torch.arange(count, dtype=torch.int32),
                )
            )
        self.assertTrue(
            torch.equal(
                workspace.selected_lengths,
                torch.tensor((0, 1, 31, 511, 512, 512, 5, 128), dtype=torch.int32),
            )
        )

    def test_c128_uses_all_completed_history_rows(self) -> None:
        ops = self.new_ops()
        batch, raw = common_inputs()
        main = compression(1, 512, 128, 1_152)
        pool = model.DeepSeekV4AttentionPool(
            indices=torch.tensor([[[7, 2]]], dtype=torch.int32),
            lengths=torch.tensor([2], dtype=torch.int32),
        )

        with (
            patch.object(self.module, "_save") as save,
            patch.object(self.module, "_compress_main") as compress_main,
            patch.object(self.module, "_attention", return_value=(1, 2)) as attention,
            patch.object(self.module, "_require_cuda_types"),
        ):
            self.assertEqual(
                ops.decode_c128(
                    batch,
                    main,
                    raw,
                    pool,
                    metadata="odd",
                    output_heads=model.LOCAL_QUERY_HEADS,
                ),
                (1, 2),
            )

        save.assert_called_once_with(batch, main, 128, False, 8)
        compress_main.assert_called_once_with(batch, main, 128, False, 1, 16)
        attention.assert_called_once_with(
            raw,
            main.cache,
            pool.indices,
            pool.lengths,
            "odd",
            1,
            model.LOCAL_QUERY_HEADS,
        )

    def test_swa_uses_only_the_rolling_window_pool(self) -> None:
        ops = self.new_ops()
        batch, raw = common_inputs()
        self.module.flash_mla.calls.clear()

        with patch.object(self.module, "_validate_batch") as validate:
            output, lse = ops.decode_swa(
                batch,
                raw,
                metadata="swa",
                output_heads=model.NUM_QUERY_HEADS,
            )

        validate.assert_called_once_with(batch, raw)
        call = self.module.flash_mla.calls[-1]
        self.assertIsNone(call["extra_k_cache"])
        self.assertIsNone(call["extra_indices_in_kvcache"])
        self.assertIsNone(call["extra_topk_length"])
        self.assertEqual(output.shape, (1, 1, 64, 512))
        self.assertEqual(lse.shape, (1, 64, 1))

    def test_flashmla_uses_two_physical_pools_and_returns_local_head_views(
        self,
    ) -> None:
        _, raw = common_inputs()
        extra_cache = torch.empty(1, 19_008, dtype=torch.uint8)
        extra_indices = torch.zeros(1, 1, 512, dtype=torch.int32)
        extra_lengths = torch.tensor([512], dtype=torch.int32)

        output, lse = self.module._attention(
            raw,
            extra_cache,
            extra_indices,
            extra_lengths,
            "metadata",
            32,
            model.LOCAL_QUERY_HEADS,
        )

        call = self.module.flash_mla.calls[-1]
        self.assertEqual(call["k_cache"].shape, (1, 64, 1, 584))
        self.assertEqual(call["extra_k_cache"].shape, (1, 32, 1, 584))
        self.assertEqual(call["head_dim_v"], 512)
        self.assertEqual(call["softmax_scale"], 512**-0.5)
        self.assertFalse(call["causal"])
        self.assertTrue(call["is_fp8_kvcache"])
        self.assertIs(call["indices"], raw.indices)
        self.assertIs(call["topk_length"], raw.lengths)
        self.assertIs(call["extra_indices_in_kvcache"], extra_indices)
        self.assertIs(call["extra_topk_length"], extra_lengths)
        self.assertEqual(output.shape, (1, 1, 16, 512))
        self.assertEqual(lse.shape, (1, 16, 1))

        output, lse = self.module._attention(
            raw,
            extra_cache,
            extra_indices,
            extra_lengths,
            "metadata",
            32,
            model.NUM_QUERY_HEADS,
        )
        self.assertEqual(output.shape, (1, 1, 64, 512))
        self.assertEqual(lse.shape, (1, 64, 1))

    def test_flashmla_cache_views_preserve_physical_page_strides(self) -> None:
        _, raw = common_inputs()
        extra_indices = torch.zeros(1, 1, 512, dtype=torch.int32)
        extra_lengths = torch.tensor([512], dtype=torch.int32)

        for page, physical_stride in ((32, 19_008), (1, 1_152)):
            with self.subTest(page=page):
                extra_cache = torch.empty(2, physical_stride, dtype=torch.uint8)
                self.module._attention(
                    raw,
                    extra_cache,
                    extra_indices,
                    extra_lengths,
                    "metadata",
                    page,
                    model.LOCAL_QUERY_HEADS,
                )
                call = self.module.flash_mla.calls[-1]
                self.assertEqual(call["k_cache"].stride(), (37_440, 584, 584, 1))
                self.assertEqual(call["extra_k_cache"].shape, (2, page, 1, 584))
                self.assertEqual(
                    call["extra_k_cache"].stride(),
                    (physical_stride, 584, 584, 1),
                )

    def test_only_fixed_graph_buckets_are_accepted(self) -> None:
        self.assertEqual(model.DECODE_DESCRIPTOR_PARITIES, 2)
        with patch.object(self.module, "_require_cuda_types"):
            batch_sizes = (
                *model.DECODE_BATCH_SIZES,
                *(
                    size * model.DSPARK_VERIFY_WIDTH
                    for size in model.DECODE_BATCH_SIZES
                ),
            )
            for batch_size in batch_sizes:
                batch, raw = common_inputs(batch_size)
                self.assertEqual(self.module._validate_batch(batch, raw), batch_size)

        batch, raw = common_inputs(3)
        with (
            patch.object(self.module, "_require_cuda_types"),
            self.assertRaisesRegex(ValueError, "unsupported decode query"),
        ):
            self.module._validate_batch(batch, raw)

        table = torch.zeros(1, 16, dtype=torch.int32)
        _, even = c4_inputs(table)
        _, odd = c4_inputs(table.clone())
        self.assertTrue(
            all(
                getattr(even, name).data_ptr() != getattr(odd, name).data_ptr()
                for name in even.__dataclass_fields__
            )
        )

    def test_rejects_non_cuda_or_wrong_dtype_native_inputs(self) -> None:
        batch, raw = common_inputs()
        main = compression(1, 512, 128, 1_152)

        with self.assertRaisesRegex(TypeError, "expected CUDA"):
            self.module._validate_batch(batch, raw)

        with patch.object(self.module, "_require_cuda_types") as require:
            self.module._validate_compression(main, 1, 128, 512, 512, 1_152)
        self.assertEqual(require.call_count, 1)

    def test_c128_reductions_use_the_pinned_torch_trees(self) -> None:
        syntax = ast.parse(Path(self.module.__file__).read_text())
        functions = {
            node.name: node for node in syntax.body if isinstance(node, ast.FunctionDef)
        }

        for name, instruction in (
            ("_c128_add_rn", "add.rn.f32 $0, $1, $2;"),
            ("_c128_mul_rn", "mul.rn.f32 $0, $1, $2;"),
            ("_c128_div_rn", "div.rn.f32 $0, $1, $2;"),
            ("_c128_sub_rn", "sub.rn.f32 $0, $1, $2;"),
        ):
            expression = functions[name].body[0]
            self.assertIsInstance(expression, ast.Return)
            self.assertEqual(
                ast.unparse(expression.value),
                "tl.inline_asm_elementwise("
                f"'{instruction}', '=f,f,f', [left, right], "
                "dtype=tl.float32, is_pure=True, pack=1)",
            )

        sum_source = ast.unparse(functions["_c128_torch_sum_dim0_rn"])
        self.assertEqual(
            [ast.unparse(node) for node in functions["_c128_torch_sum_dim0_rn"].body],
            [
                (
                    "by_k = tl.trans(tl.reshape(product, (8, 4, 2, 2, 512)), "
                    "(4, 1, 2, 3, 0))"
                ),
                "k_even, k_odd = tl.split(tl.reshape(by_k, (512, 4, 2, 2, 4, 2)))",
                "k_04, k_26 = tl.split(tl.reshape(k_even, (512, 4, 2, 2, 2, 2)))",
                "k_15, k_37 = tl.split(tl.reshape(k_odd, (512, 4, 2, 2, 2, 2)))",
                "k_0, k_4 = tl.split(k_04)",
                "k_2, k_6 = tl.split(k_26)",
                "k_1, k_5 = tl.split(k_15)",
                "k_3, k_7 = tl.split(k_37)",
                "partial = _c128_add_rn(k_0, k_1)",
                *[
                    f"partial = _c128_add_rn(partial, k_{index})"
                    for index in range(2, 8)
                ],
                "by_a = tl.trans(partial, (0, 2, 3, 1))",
                "a_02, a_13 = tl.split(tl.reshape(by_a, (512, 2, 2, 2, 2)))",
                "a_0, a_2 = tl.split(a_02)",
                "a_1, a_3 = tl.split(a_13)",
                "partial = _c128_add_rn(a_0, a_1)",
                "partial = _c128_add_rn(partial, a_2)",
                "partial = _c128_add_rn(partial, a_3)",
                "y_02, y_13 = tl.split(partial)",
                "y_0, y_2 = tl.split(y_02)",
                "y_1, y_3 = tl.split(y_13)",
                "return _c128_add_rn(_c128_add_rn(y_0, y_2), _c128_add_rn(y_1, y_3))",
            ],
        )
        self.assertNotIn("tl.load", sum_source)
        self.assertNotIn("tl.store", sum_source)
        self.assertNotIn("_ptr", sum_source)

        self.assertEqual(
            [
                ast.unparse(statement)
                for statement in functions["_c128_reduce_halves_rn"].body
            ],
            [
                "left, right = tl.split(tl.trans(tl.reshape(values, (2, WIDTH // 2))))",
                "return _c128_add_rn(left, right)",
            ],
        )
        self.assertEqual(
            [
                ast.unparse(statement)
                for statement in functions["_c128_torch_variance"].body
            ],
            [
                "squared = _c128_mul_rn(pooled, pooled)",
                "columns_02, columns_13 = tl.split(tl.reshape(squared, (128, 2, 2)))",
                "column_0, column_2 = tl.split(columns_02)",
                "column_1, column_3 = tl.split(columns_13)",
                "partial = _c128_add_rn(column_0, column_1)",
                "partial = _c128_add_rn(partial, column_2)",
                "partial = _c128_add_rn(partial, column_3)",
                *[
                    f"partial = _c128_reduce_halves_rn(partial, WIDTH={width})"
                    for width in (128, 64, 32, 16, 8, 4, 2)
                ],
                "return _c128_mul_rn(partial, 1.0 / 512.0)",
            ],
        )
        main_body = [
            ast.unparse(statement) for statement in functions["_main_compress"].body
        ]
        softmax_branch = next(
            statement
            for statement in functions["_main_compress"].body
            if isinstance(statement, ast.If)
            and ast.unparse(statement.test) == "RATIO == 128"
            and "maximum = tl.full" in ast.unparse(statement)
        )
        c128_softmax = ast.unparse(softmax_branch)
        self.assertEqual(
            c128_softmax.count("tl.range(0, 128, loop_unroll_factor=1)"), 2
        )
        self.assertEqual(c128_softmax.count("libdevice.exp("), 2)
        self.assertIn(
            "weights = _c128_div_rn(numerator, denominator[None, :])", c128_softmax
        )
        self.assertIn("product = _c128_mul_rn(values, weights)", c128_softmax)
        self.assertIn("pooled = _c128_torch_sum_dim0_rn(product)", c128_softmax)
        self.assertNotIn("tl.store", c128_softmax)
        self.assertNotIn("tl.load(product", c128_softmax)
        self.assertEqual(
            [ast.unparse(statement) for statement in softmax_branch.orelse],
            [
                "weights = tl.softmax(score, dim=0)",
                "values = tl.load(source + dim[None, :], mask=valid, other=0.0)",
                "pooled = tl.sum(values * weights, axis=0).to(tl.bfloat16).to(tl.float32)",
            ],
        )
        self.assertIn(
            "if RATIO == 128:\n    variance = _c128_torch_variance(pooled)\nelse:\n"
            "    variance = tl.sum(pooled * pooled, axis=0) / HEAD_DIM",
            main_body,
        )
        self.assertIn(
            "normed = (pooled * tl.rsqrt(variance + 1e-06) * rms).to(tl.bfloat16).to(tl.float32)",
            main_body,
        )
        self.assertIn(
            "if RATIO == 128:\n"
            "    even_cosine = _c128_mul_rn(even, cosine)\n"
            "    odd_sine = _c128_mul_rn(odd, sine)\n"
            "    odd_cosine = _c128_mul_rn(odd, cosine)\n"
            "    even_sine = _c128_mul_rn(even, sine)\n"
            "    real = _c128_sub_rn(even_cosine, odd_sine)\n"
            "    imaginary = _c128_add_rn(odd_cosine, even_sine)\n"
            "    rotated = tl.interleave(real, imaginary)\n"
            "else:\n"
            "    rotated = tl.interleave(even * cosine - odd * sine, "
            "odd * cosine + even * sine)",
            main_body,
        )

    def test_provenance_and_corrected_bf16_boundaries_are_retained(self) -> None:
        source = Path(self.module.__file__).read_text()
        license_text = (
            Path(self.module.__file__)
            .with_name("compressor.LICENSE.txt")
            .read_text()
        )

        self.assertNotIn("tmp.", source)
        self.assertIn("f17b03efc1728875c586d848f49da5905032e87c", license_text)
        self.assertIn("70f70a054227978576c5f316a1d490cecf36fcd9", license_text)
        self.assertIn("MIT AND Apache-2.0", license_text)
        self.assertGreaterEqual(
            inspect.getsource(self.module._main_compress).count(
                ".to(tl.bfloat16).to(tl.float32)"
            ),
            2,
        )
        self.assertGreaterEqual(
            inspect.getsource(self.module._index_compress).count(
                ".to(tl.bfloat16).to(tl.float32)"
            ),
            2,
        )

    def test_triton_jit_functions_only_reference_compiler_safe_globals(self) -> None:
        path = Path(self.module.__file__)
        source = path.read_text()
        syntax = ast.parse(source)
        jit_functions = {
            node.name
            for node in syntax.body
            if isinstance(node, ast.FunctionDef)
            and any(
                ast.unparse(decorator) == "triton.jit"
                or (
                    isinstance(decorator, ast.Call)
                    and ast.unparse(decorator.func) == "triton.jit"
                )
                for decorator in node.decorator_list
            )
        }
        self.assertEqual(
            jit_functions,
            {
                "_c128_add_rn",
                "_c128_mul_rn",
                "_c128_div_rn",
                "_c128_sub_rn",
                "_c128_torch_sum_dim0_rn",
                "_c128_reduce_halves_rn",
                "_c128_torch_variance",
                "_power2_quantize",
                "_normalize_hash_route_weight",
                "_ordered_float_key",
                "_learned_route",
                "_hash_route",
                "_shared_swiglu_quantize",
                "_mxfp4_nibble",
                "_c4_query",
                "_fht_stage",
                "_c4_decode_query",
                "_stage_prefill_index_kernel",
                "_load_prefill_cache_row",
                "_stage_prefill_attention_kernel",
                "_save_state",
                "_main_compress",
                "_index_compress",
            },
        )
        constexpr_globals = {
            node.target.id
            for node in syntax.body
            if isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and ast.unparse(node.annotation) == "tl.constexpr"
        }
        tables = {
            table.get_name(): table
            for table in symtable.symtable(source, str(path), "exec").get_children()
        }
        allowed = {
            "tl",
            "libdevice",
            "float",
            "range",
            *jit_functions,
            *constexpr_globals,
        }
        unsafe = {
            name: sorted(
                identifier
                for identifier in tables[name].get_identifiers()
                if tables[name].lookup(identifier).is_global()
                and identifier not in allowed
            )
            for name in jit_functions
        }
        self.assertEqual(unsafe, {name: [] for name in jit_functions})


if __name__ == "__main__":
    unittest.main()

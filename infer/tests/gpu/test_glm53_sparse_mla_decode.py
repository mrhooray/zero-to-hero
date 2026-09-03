import ast
import types
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import torch

from infer.models.glm53_flash import checkpoint as glm53_checkpoint
from infer.models.glm53_flash import model as glm53_flash

pytestmark = pytest.mark.gpu

ROOT = Path(__file__).resolve().parents[2]
OPS_SOURCE = (ROOT / "src/infer/models/glm53_flash/ops/core.py").read_text()


class SparseMLADecodeModelTest(unittest.TestCase):
    def test_decode_weights_pack_key_and_ape(self) -> None:
        self.assertEqual(
            tuple(glm53_flash.SparseMLADecodeWeights.__dataclass_fields__),
            ("w_kc", "pool_ape"),
        )
        local_heads = glm53_flash.glm53_local_heads(glm53_flash.TP_SIZE)
        kv_b_heads = torch.empty((local_heads, 512, 512), dtype=torch.bfloat16)
        for head in range(local_heads):
            kv_b_heads[head, :256].fill_(head + 1)
            kv_b_heads[head, 256:].fill_(-(head + 1))
        ape = torch.arange(4 * 128, dtype=torch.bfloat16).view(4, 128)
        packed = glm53_checkpoint.pack_sparse_mla_decode_weights(
            {
                "kv_b_proj.weight": kv_b_heads.view(local_heads * 512, 512),
                "indexer.index_kpool_compress_ape": ape,
            },
            torch,
        )

        self.assertEqual(packed.w_kc.shape, (16, 256, 512))
        self.assertTrue(packed.w_kc.is_contiguous())
        self.assertTrue(torch.equal(packed.w_kc, kv_b_heads[:, :256]))
        self.assertIs(packed.pool_ape, ape)

    def test_output_weights_restore_values_in_exact_tep4_orientation(self) -> None:
        self.assertEqual(
            tuple(glm53_flash.SparseMLAOutputWeights.__dataclass_fields__),
            ("w_vc", "o_proj", "o_proj_scale_inv"),
        )
        specs = {spec.name: spec for spec in glm53_checkpoint.SPARSE_MLA_WEIGHTS}
        self.assertEqual(specs["kv_b_proj.weight"].local_shape(), (8192, 512))
        self.assertEqual(specs["o_proj.weight"].local_shape(), (4096, 4096))
        self.assertEqual(specs["o_proj.weight_scale_inv"].local_shape(), (32, 32))

        local_heads = glm53_flash.glm53_local_heads(glm53_flash.TP_SIZE)
        kv_b_heads = torch.zeros((local_heads, 512, 512), dtype=torch.bfloat16)
        latent = torch.zeros((1, 16, 512), dtype=torch.float32)
        expected = torch.zeros((1, 16, 256), dtype=torch.float32)
        for head in range(local_heads):
            kv_b_heads[head, 256 + head, head] = head + 1
            latent[0, head, head] = 2
            expected[0, head, head] = 2 * (head + 1)
        o_proj = torch.empty(1)
        o_proj_scale = torch.empty(1)
        packed = glm53_checkpoint.pack_sparse_mla_output_weights(
            {
                "kv_b_proj.weight": kv_b_heads.view(local_heads * 512, 512),
                "o_proj.weight": o_proj,
                "o_proj.weight_scale_inv": o_proj_scale,
            }
        )

        self.assertEqual(packed.w_vc.shape, (16, 512, 256))
        self.assertEqual(packed.w_vc.dtype, torch.bfloat16)
        self.assertTrue(packed.w_vc.is_contiguous())
        value_hbd = torch.bmm(latent.transpose(0, 1), packed.w_vc.float())
        projected = value_hbd.transpose(0, 1).contiguous().view(1, 4096)
        self.assertTrue(torch.equal(projected.view(1, 16, 256), expected))
        self.assertIs(packed.o_proj, o_proj)
        self.assertIs(packed.o_proj_scale_inv, o_proj_scale)

    def test_workspace_and_history_geometry_are_fixed(self) -> None:
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
        self.assertEqual(glm53_flash.SPARSE_MLA_COMPOUND_BLOCKS, 8192)
        self.assertEqual(glm53_flash.SPARSE_MLA_INDEX_CONTEXT_CAPACITY, 262_144)
        self.assertEqual(glm53_flash.SPARSE_MLA_INDEX_PAGE_BYTES, 32 * 128 + 32 * 4)
        self.assertEqual(glm53_flash.SPARSE_MLA_SPARSE_CAPACITY, 2052)
        self.assertEqual(
            glm53_flash.sparse_mla_decode_workspace_shapes(width),
            glm53_flash.SparseMLADecodeWorkspace(
                context_lengths=(width, 1),
                sequence_lengths=(width,),
                index_q=(width, 1, 32, 128),
                score_weights=(width, 32),
                main_q_hbd=(16, width, 256),
                absorbed_hbd=(16, width, 512),
                attention_q=(width, 1, 16, 512),
                schedule=(149, 2),
                logits=(width, 262_144),
                selected=(width, 512),
                topk_offsets=(width,),
                topk_rows=(1_048_576,),
                sparse_ids=(width, 1, 2052),
                sparse_lengths=(width,),
                flashinfer_workspace=(128 * 1024 * 1024,),
                counter=(608,),
                output=(width, 1, 16, 512),
            ),
        )
        self.assertEqual(
            [
                glm53_flash.sparse_mla_decode_workspace_shapes(n).counter
                for n in (*glm53_flash.GLM53_TARGET_DECODE_BATCH_SIZES, width, 128)
            ],
            [
                (608,),
                (608,),
                (608,),
                (608,),
                (1024,),
                (2048,),
                (4096,),
                (608,),
                (8192,),
            ],
        )
        self.assertEqual(
            glm53_flash.sparse_mla_decode_workspace_shapes(1455).output,
            (1455, 1, 16, 512),
        )
        for batch_size in (0, glm53_flash.KDA_CHUNK_SIZE + 1):
            with self.assertRaisesRegex(ValueError, rf"\[1, {glm53_flash.KDA_CHUNK_SIZE}\]"):
                glm53_flash.sparse_mla_decode_workspace_shapes(batch_size)
        self.assertEqual(
            tuple(glm53_flash.SparseMLAHistory.__dataclass_fields__),
            ("latent", "index_cache", "tail_key", "tail_gate"),
        )
        self.assertIn(
            "one caller-owned graph-buffer set per parity",
            str.lower(glm53_flash.SparseMLADecodeWorkspace.__doc__),
        )
        self.assertIn(
            "inactive rows skip writes", str.lower(glm53_flash.SparseMLADecodeBatch.__doc__)
        )

    def test_output_workspace_supports_live_prefill_tokens(self) -> None:
        width = glm53_flash.GLM53_TARGET_VERIFY_WIDTH
        self.assertEqual(
            glm53_flash.sparse_mla_output_workspace_shapes(width),
            glm53_flash.SparseMLAOutputWorkspace(
                value_hbd=(16, width, 256),
                projected=(width, 4096),
                projected_fp8=(width, 4096),
                projected_scale=(width, 32),
                output=(width, 4096),
            ),
        )
        self.assertEqual(
            [
                glm53_flash.sparse_mla_output_workspace_shapes(batch).projected
                for batch in (*glm53_flash.GLM53_TARGET_DECODE_BATCH_SIZES, width, 128)
            ],
            [
                (1, 4096),
                (2, 4096),
                (4, 4096),
                (8, 4096),
                (16, 4096),
                (32, 4096),
                (64, 4096),
                (width, 4096),
                (128, 4096),
            ],
        )
        self.assertEqual(
            glm53_flash.sparse_mla_output_workspace_shapes(1455).projected,
            (1455, 4096),
        )
        for batch_size in (0, glm53_flash.KDA_CHUNK_SIZE + 1):
            with self.assertRaisesRegex(ValueError, rf"\[1, {glm53_flash.KDA_CHUNK_SIZE}\]"):
                glm53_flash.sparse_mla_output_workspace_shapes(batch_size)
        self.assertIn(
            "one decode graph parity",
            str.lower(glm53_flash.SparseMLAOutputWorkspace.__doc__),
        )


class SparseMLADecodeOpsTest(unittest.TestCase):
    def test_verify_snapshots_grouped_request_major_tails(self) -> None:
        verify, launches = load_verify_sparse_mla()
        width = glm53_flash.GLM53_TARGET_VERIFY_WIDTH
        for groups in (1, 2, 16, 32):
            with self.subTest(groups=groups):
                rows = groups * width
                raw_lengths = torch.cat(
                    tuple(
                        torch.arange(10 * group + 1, 10 * group + width + 1)
                        for group in range(groups)
                    )
                ).to(torch.int32)
                state_slots = torch.arange(groups, dtype=torch.int32).repeat_interleave(
                    width
                )
                projection = SimpleNamespace(
                    latent=torch.zeros((rows, 1), dtype=torch.bfloat16),
                    key=torch.zeros((rows, 1), dtype=torch.bfloat16),
                    pool_gate=torch.zeros((rows, 1), dtype=torch.bfloat16),
                )
                batch = SimpleNamespace(
                    active=torch.ones(rows, dtype=torch.uint8),
                    raw_lengths=raw_lengths,
                    state_slots=state_slots,
                    block_table=torch.zeros((rows, 1), dtype=torch.int32),
                )
                history = glm53_flash.SparseMLAHistory(
                    latent=torch.zeros(1, dtype=torch.bfloat16),
                    index_cache=torch.zeros(1, dtype=torch.uint8),
                    tail_key=torch.full((2, 32, 4, 128), -1, dtype=torch.bfloat16),
                    tail_gate=torch.full((2, 32, 4, 128), -2, dtype=torch.bfloat16),
                )
                transaction_key = torch.empty(
                    (rows, 2, 1, 4, 128), dtype=torch.bfloat16
                )
                transaction_gate = torch.empty_like(transaction_key)
                workspace = SimpleNamespace(
                    context_lengths=torch.zeros((rows, 1), dtype=torch.int32),
                    sequence_lengths=torch.zeros(rows, dtype=torch.int32),
                )
                launches.clear()

                returned = verify(
                    None,
                    projection,
                    SimpleNamespace(pool_ape=torch.zeros(1, dtype=torch.bfloat16)),
                    batch,
                    history,
                    transaction_key,
                    transaction_gate,
                    workspace,
                )

                self.assertEqual(returned, "query-result")
                for row, raw_length in enumerate(raw_lengths.tolist()):
                    self.assertTrue(torch.all(transaction_key[row] == raw_length))
                    self.assertTrue(torch.all(transaction_gate[row] == -raw_length))
                self.assertEqual(
                    [(grid, keywords["ROW_START"]) for grid, keywords in launches],
                    [((groups,), row) for row in range(width)],
                )
                self.assertTrue(
                    all(
                        keywords["ROW_STRIDE"] == width
                        and keywords["WRITE_TRANSACTION"]
                        for _, keywords in launches
                    )
                )
        source = ast.unparse(functions_by_name(OPS_SOURCE)["verify_sparse_mla"])
        self.assertNotIn("torch.index_select", source)
        self.assertIn("range(GLM53_TARGET_VERIFY_WIDTH)", source)
        self.assertIn("ROW_STRIDE=GLM53_TARGET_VERIFY_WIDTH", source)

    def test_decode_composes_the_fixed_direct_native_path(self) -> None:
        method = functions_by_name(OPS_SOURCE)["decode_sparse_mla"]
        calls = list(ast_calls(method))
        source = ast.unparse(method)
        ordered = (
            "_sparse_mla_runtime",
            "_write_sparse_mla_history",
            "_run_sparse_mla_query",
        )
        positions = [source.index(name) for name in ordered]
        self.assertEqual(positions, sorted(positions))
        for name in ordered:
            self.assertEqual(count_calls(calls, name), 1)

        query = functions_by_name(OPS_SOURCE)["_run_sparse_mla_query"]
        query_calls = list(ast_calls(query))
        query_source = ast.unparse(query)
        ordered = (
            "bmm",
            "get_paged_mqa_logits_metadata_out",
            "fp8_paged_mqa_logits_out",
            "radix_topk_ragged_transform",
            "_map_sparse_mla_ids",
            "trtllm_batch_decode_with_kv_cache_mla",
            "torch.mul",
        )
        positions = [query_source.index(name) for name in ordered]
        self.assertEqual(positions, sorted(positions))
        for name in (*ordered[:-1], "mul"):
            self.assertEqual(count_calls(query_calls, name), 1)

        self.assertIn("weights.w_kc", query_source)
        self.assertIn("out=w.absorbed_hbd", query_source)
        self.assertIn("SPARSE_MLA_INDEX_SCORE_SCALE", query_source)
        self.assertIn("SPARSE_MLA_INDEX_CONTEXT_CAPACITY", query_source)
        self.assertIn("SPARSE_MLA_B200_SMS", query_source)
        self.assertIn("dsa_graph_safe=True", query_source)
        self.assertIn("row_starts=None", query_source)
        self.assertIn("backend='trtllm-gen'", query_source)
        self.assertIn("local_heads = weights.w_kc.shape[0]", query_source)
        self.assertIn("bmm1_scale=1 / local_heads", query_source)
        self.assertIn("uses_shared_paged_kv_idx=True", query_source)
        self.assertIn("return w.output[:, 0]", query_source)
        for allocation in ("empty", "zeros", "tensor", "arange", "clone", "item"):
            self.assertEqual(count_calls(query_calls, allocation), 0)
        for excluded in ("w_vc", "o_proj", "all_reduce", "fallback", "prefill"):
            self.assertNotIn(excluded, query_source.lower())

    def test_output_tail_restores_projects_then_reduces_bf16_once(self) -> None:
        method = functions_by_name(OPS_SOURCE)["decode_sparse_mla_output"]
        calls = list(ast_calls(method))
        source = ast.unparse(method)

        for name in (
            "bmm",
            "copy_",
            "_mn_scale_view",
            "_quantize",
            "_fp8_gemm",
            "all_reduce",
        ):
            self.assertEqual(count_calls(calls, name), 1)
        ordered = (
            "torch.bmm",
            "copy_",
            "_quantize",
            "_fp8_gemm",
            "return all_reduce(w.output)",
        )
        positions = [source.index(name) for name in ordered]
        self.assertEqual(positions, sorted(positions))
        for exact in (
            "latent.transpose(0, 1)",
            "weights.w_vc",
            "out=w.value_hbd",
            "w.value_hbd.transpose(0, 1)",
            "weights.o_proj",
            "weights.o_proj_scale_inv",
            "projected_scale = _mn_scale_view(w.projected_scale)",
            "return all_reduce(w.output)",
        ):
            self.assertIn(exact, source)
        for allocation in ("empty", "zeros", "tensor", "clone"):
            self.assertEqual(count_calls(calls, allocation), 0)

        validation = ast.unparse(
            functions_by_name(OPS_SOURCE)["_validate_sparse_mla_output"]
        )
        self.assertIn("dist.get_world_size(group) != TP_SIZE", validation)
        self.assertIn("(workspace.output, torch.bfloat16, expected.output)", validation)
        self.assertIn("(weights.o_proj, torch.float8_e4m3fn", validation)
        self.assertIn("weights.o_proj_scale_inv, torch.float32", validation)

    def test_writer_uses_compound_planar_history_and_exact_pooling(self) -> None:
        source = ast.unparse(functions_by_name(OPS_SOURCE)["_write_sparse_mla_history"])

        for exact in (
            "row = ROW_START + tl.program_id(0) * ROW_STRIDE",
            "position = tl.maximum(raw_length - 1, 0)",
            "row * 8192 + position // 128",
            "latent_page = 2 * block + within // 64",
            "latent = tl.load(latent_ptr + row * 512 + latent_dim)",
            "tl.store(latent_out + latent_dim, latent, mask=active)",
            "for parity in tl.static_range(2)",
            "if WRITE_TRANSACTION",
            "transaction = (row * 2 + parity) * 512 + transaction_offsets",
            "tl.store(transaction_key_ptr + transaction, key_snapshot, mask=active)",
            "tl.store(transaction_gate_ptr + transaction, gate_snapshot, mask=active)",
            "complete = active & (tail_row == 3)",
            "tl.softmax(logits, dim=0).to(tl.bfloat16)",
            "product = (probability * keys).to(tl.bfloat16).to(tl.float32)",
            (
                "product_02, product_13 = tl.split(tl.reshape(tl.trans(product), "
                "(128, 2, 2)))"
            ),
            "product_0, product_2 = tl.split(product_02)",
            "product_1, product_3 = tl.split(product_13)",
            "tl.maximum(tl.max(tl.abs(pooled), axis=0), 0.0001) / 448.0",
            "page = index_cache_ptr + block * 4224",
            "page + index_row * 128 + dim",
            "page + 4096 + index_row * 4 + byte",
            "tl.where(active, raw_length // 4, 0)",
            "tl.where(active, raw_length, 1)",
        ):
            self.assertIn(exact, source)
        self.assertGreaterEqual(source.count("mask=active"), 3)
        self.assertGreaterEqual(source.count("mask=complete"), 2)
        self.assertIn("bitcast=True", source)
        pooling = (
            "pooled = product_0 + product_1",
            "pooled = pooled + product_2",
            "pooled = pooled + product_3",
            "pooled = pooled.to(tl.bfloat16).to(tl.float32)",
        )
        positions = [source.index(expression) for expression in pooling]
        self.assertEqual(positions, sorted(positions))
        self.assertNotIn("tl.sum(product", source)

    def test_native_attention_requires_bf16_query_and_latent_history(self) -> None:
        decode = ast.unparse(
            functions_by_name(OPS_SOURCE)["_validate_sparse_mla_decode"]
        )
        workspace = ast.unparse(
            functions_by_name(OPS_SOURCE)["_validate_sparse_mla_query_workspace"]
        )

        self.assertIn("(history.latent, torch.bfloat16", decode)
        self.assertIn("if name == 'index_q'", workspace)
        self.assertIn(
            "('main_q_hbd', 'absorbed_hbd', 'attention_q', 'output')", workspace
        )
        self.assertNotIn("('index_q', 'attention_q')", workspace)

    def test_prefill_query_rejects_workspace_aliases_before_launch(self) -> None:
        validate_aliases = load_alias_validator()
        projection, weights, staged_query, history, workspace = alias_fixture()
        cases = (
            (
                replace(
                    workspace,
                    output=history.latent.view(workspace.output.shape),
                ),
                "workspace.output and history.latent overlap",
            ),
            (
                replace(
                    workspace,
                    main_q_hbd=projection.main_q.view(workspace.main_q_hbd.shape),
                ),
                "workspace.main_q_hbd and projection.main_q overlap",
            ),
            (
                replace(workspace, score_weights=projection.score_weights),
                "workspace.score_weights and projection.score_weights overlap",
            ),
        )
        for aliased, message in cases:
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(ValueError, message),
            ):
                validate_aliases(
                    projection,
                    weights,
                    staged_query,
                    history,
                    aliased,
                )

    def test_prefill_query_alias_contract_covers_every_mutable_and_read_only(
        self,
    ) -> None:
        validation = ast.unparse(
            functions_by_name(OPS_SOURCE)["_validate_sparse_mla_prefill_query"]
        )
        aliases = ast.unparse(
            functions_by_name(OPS_SOURCE)["_validate_sparse_mla_query_aliases"]
        )
        self.assertIn("history.tail_key, torch.bfloat16, tail_shape", validation)
        self.assertIn("history.tail_gate, torch.bfloat16, tail_shape", validation)
        self.assertIn("_validate_sparse_mla_query_aliases", validation)
        self.assertIn("_workspace_tensors('workspace', workspace)", aliases)
        self.assertIn("*_workspace_tensors('history', history)", aliases)
        for name in (
            "projection.main_q",
            "projection.index_q",
            "projection.score_weights",
            "weights.w_kc",
            "staged_query.active",
            "staged_query.raw_lengths",
            "staged_query.block_table",
            "staged_query.null_token",
        ):
            self.assertIn(name, aliases)

    def test_compound_mapper_covers_boundaries_and_null_rows(self) -> None:
        expected = {
            1: 1,
            3: 3,
            4: 4,
            5: 5,
            2047: 2047,
            2048: 2048,
            2051: 2051,
            2052: 2048,
            2053: 2049,
        }
        actual = {raw: 4 * min(raw // 4, 512) + raw % 4 for raw in expected}
        self.assertEqual(actual, expected)

        source = ast.unparse(functions_by_name(OPS_SOURCE)["_map_sparse_mla_ids"])
        for exact in (
            "tl.minimum(raw_length // 4, 512)",
            "4 * selected + offsets % 4",
            "raw_length - tail + offsets - selected_tokens",
            "row * 8192 + raw // 128",
            "physical = block * 128 + raw % 128",
            "tl.where(valid, physical, -1)",
            "~active & (offsets == 0)",
            "tl.where(active, active_length, 1)",
        ):
            self.assertIn(exact, source)

    def test_runtime_is_exact_aot_0618_and_sealed_deepgemm(self) -> None:
        runtime = ast.unparse(functions_by_name(OPS_SOURCE)["_sparse_mla_runtime"])
        for identity in (
            '"apache-tvm-ffi": "0.1.11"',
            "0.6.18.dev20260819",
            "0.6.18.dev20260819+cu129",
            '"nvidia-cutlass-dsl": "4.5.0"',
            '"nvidia-cutlass-dsl-libs-base": "4.5.0"',
            "61a6c651872a7d3f2f6dcc1ced61633d8f8ba3dd",
            "0.1.5.post3",
            "fafde24e6413a6be95ed0e094cf5746311ae8c87c3fe46466839f16b17e1896e",
        ):
            self.assertIn(identity, OPS_SOURCE)
        self.assertIn("gen_topk_module().is_aot", runtime)
        self.assertIn("gen_trtllm_gen_fmha_module().is_aot", runtime)
        self.assertIn("get_paged_mqa_logits_metadata_out", runtime)
        self.assertIn("fp8_paged_mqa_logits_out", runtime)
        self.assertIn("tf32_hc_prenorm_gemm", runtime)
        self.assertIn("deep_gemm.get_num_sms() != SPARSE_MLA_B200_SMS", runtime)
        self.assertIn("cutlass_cuda != (12, 9)", runtime)
        self.assertNotIn("0.6.17", OPS_SOURCE)
        self.assertNotIn("infer.ops.deepseek_v4", OPS_SOURCE)


def load_alias_validator():
    functions = functions_by_name(OPS_SOURCE)
    module = types.ModuleType("sparse_mla_alias_validator")
    module.torch = torch
    tree = ast.Module(
        body=[
            functions["_contiguous_tensors_overlap"],
            functions["_workspace_tensors"],
            functions["_validate_sparse_mla_query_aliases"],
        ],
        type_ignores=[],
    )
    code = compile(ast.fix_missing_locations(tree), "glm53.py", "exec")
    exec(code, module.__dict__)  # noqa: S102
    return module._validate_sparse_mla_query_aliases


def load_verify_sparse_mla():
    method = functions_by_name(OPS_SOURCE)["verify_sparse_mla"]
    for argument in (
        *method.args.posonlyargs,
        *method.args.args,
        *method.args.kwonlyargs,
    ):
        argument.annotation = None
    method.returns = None

    class Kernel:
        def __init__(self, launch):
            self.launch = launch

        def __getitem__(self, grid):
            return lambda *args, **kwargs: self.launch(grid, *args, **kwargs)

    launches = []

    def write_history(grid, *args, **kwargs):
        launches.append((grid, kwargs))
        for program in range(grid[0]):
            row = kwargs["ROW_START"] + program * kwargs["ROW_STRIDE"]
            raw_length = int(args[1][row])
            state_slot = int(args[2][row])
            args[10][:, state_slot].fill_(raw_length)
            args[11][:, state_slot].fill_(-raw_length)
            args[14][row].copy_(args[10][:, state_slot : state_slot + 1])
            args[15][row].copy_(args[11][:, state_slot : state_slot + 1])

    def require(_device, *specifications):
        for tensor, dtype, shape in specifications:
            if tensor.dtype != dtype or tuple(tensor.shape) != shape:
                raise ValueError("invalid transaction tail")

    module = types.ModuleType("verify_sparse_mla_test")
    module.__dict__.update(
        torch=torch,
        GLM53_TARGET_VERIFY_WIDTH=glm53_flash.GLM53_TARGET_VERIFY_WIDTH,
        SPARSE_MLA_INDEX_POOL_TOKENS=glm53_flash.SPARSE_MLA_INDEX_POOL_TOKENS,
        SPARSE_MLA_INDEXER_HEAD_DIM=glm53_flash.SPARSE_MLA_INDEXER_HEAD_DIM,
        _validate_sparse_mla_decode=lambda projection, *_args: projection.latent.shape[
            0
        ],
        _require_sparse_mla_tensors=require,
        _write_sparse_mla_history=Kernel(write_history),
        _run_sparse_mla_query=lambda *_args: "query-result",
        _sparse_mla_runtime=lambda: (),
    )
    tree = ast.Module(body=[method], type_ignores=[])
    exec(  # noqa: S102
        compile(ast.fix_missing_locations(tree), "glm53.py", "exec"),
        module.__dict__,
    )
    return module.verify_sparse_mla, launches


def alias_fixture():
    projection_shapes = glm53_flash.sparse_mla_projection_workspace_shapes(128)
    projection = SimpleNamespace(
        main_q=torch.empty(projection_shapes.main_q, dtype=torch.bfloat16),
        index_q=torch.empty(projection_shapes.index_q, dtype=torch.bfloat16),
        score_weights=torch.empty(projection_shapes.score_weights, dtype=torch.float32),
    )
    weights = SimpleNamespace(
        w_kc=torch.empty(
            (glm53_flash.glm53_local_heads(glm53_flash.TP_SIZE), 256, 512),
            dtype=torch.bfloat16,
        )
    )
    staged_query = SimpleNamespace(
        active=torch.empty(128, dtype=torch.uint8),
        raw_lengths=torch.empty(128, dtype=torch.int32),
        block_table=torch.empty(
            (128, glm53_flash.SPARSE_MLA_COMPOUND_BLOCKS), dtype=torch.int32
        ),
        null_token=torch.empty(1, dtype=torch.int32),
    )
    history_blocks = 16
    history = glm53_flash.SparseMLAHistory(
        latent=torch.empty(
            (
                2 * history_blocks,
                1,
                glm53_flash.SPARSE_MLA_LATENT_PAGE_TOKENS,
                glm53_flash.SPARSE_MLA_KV_LORA_RANK,
            ),
            dtype=torch.bfloat16,
        ),
        index_cache=torch.empty(
            (history_blocks, glm53_flash.SPARSE_MLA_INDEX_PAGE_BYTES), dtype=torch.uint8
        ),
        tail_key=torch.empty((2, 1, 4, 128), dtype=torch.bfloat16),
        tail_gate=torch.empty((2, 1, 4, 128), dtype=torch.bfloat16),
    )
    shapes = glm53_flash.sparse_mla_decode_workspace_shapes(128)
    values = {}
    for name in shapes.__dataclass_fields__:
        if name == "index_q":
            dtype = torch.float8_e4m3fn
        elif name in ("score_weights", "logits"):
            dtype = torch.float32
        elif name in ("main_q_hbd", "absorbed_hbd", "attention_q", "output"):
            dtype = torch.bfloat16
        elif name in ("topk_rows", "flashinfer_workspace", "counter"):
            dtype = torch.uint8
        else:
            dtype = torch.int32
        values[name] = torch.empty(getattr(shapes, name), dtype=dtype)
    workspace = glm53_flash.SparseMLADecodeWorkspace(**values)
    return projection, weights, staged_query, history, workspace


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

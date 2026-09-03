import ast
import unittest
from pathlib import Path

from infer.models.glm53_flash import checkpoint as glm53_checkpoint
from infer.models.glm53_flash import model as glm53_flash

MODEL_SOURCE = (
    Path(__file__).parents[1] / "src/infer/models/glm53_flash/checkpoint.py"
).read_text()
OPS_SOURCE = (
    Path(__file__).parents[1] / "src/infer/models/glm53_flash/ops/core.py"
).read_text()


def function(source: str, name: str) -> ast.FunctionDef:
    return next(
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def call_name(call: ast.Call) -> str:
    return call.func.attr if isinstance(call.func, ast.Attribute) else call.func.id


def calls(node: ast.AST) -> list[ast.Call]:
    return [child for child in ast.walk(node) if isinstance(child, ast.Call)]


def call_names(nodes: list[ast.stmt]) -> list[str]:
    return [call_name(call) for node in nodes for call in calls(node)]


class GLM53EndpointModelTest(unittest.TestCase):
    def test_pinned_checkpoint_layout_and_tp4_rows(self) -> None:
        self.assertEqual(glm53_flash.TOKENIZER_VOCAB_SIZE, 154_856)
        self.assertEqual(glm53_flash.VOCAB_SIZE - glm53_flash.TOKENIZER_VOCAB_SIZE, 24)
        specs = {
            spec.checkpoint_key: spec
            for spec in glm53_checkpoint.GLM53_ENDPOINT_WEIGHTS
        }
        self.assertEqual(
            [
                (name, spec.dtype, spec.shape, spec.shard_axis, spec.local_shape())
                for name, spec in specs.items()
            ],
            [
                (
                    "model.language_model.embed_tokens.weight",
                    "BF16",
                    (154880, 4096),
                    0,
                    (38720, 4096),
                ),
                ("lm_head.weight", "BF16", (154880, 4096), 0, (38720, 4096)),
                ("model.language_model.norm.weight", "BF16", (4096,), None, (4096,)),
            ],
        )
        self.assertEqual(
            tuple(glm53_checkpoint.GLM53_ENDPOINT_SHARDS.values()),
            (glm53_checkpoint.EMBEDDING_SHARD[0],) * 2
            + (glm53_checkpoint.FINAL_NORM_SHARD[0],),
        )
        self.assertEqual(
            glm53_checkpoint.EMBEDDING_SHARD,
            ("model-00001-of-00062.safetensors", 5_365_306_704, 93_512),
        )
        self.assertEqual(
            glm53_checkpoint.FINAL_NORM_SHARD,
            ("model-00062-of-00062.safetensors", 1_261_584_968, 39_488),
        )
        self.assertNotIn("shared_head", " ".join(specs))
        self.assertNotIn("mtp", " ".join(specs).lower())

    def test_loader_pins_index_headers_and_pread_path(self) -> None:
        loader = ast.unparse(function(MODEL_SOURCE, "load_glm53_endpoint_weights"))
        raw_loader = ast.unparse(function(MODEL_SOURCE, "_load_layer0_raw"))
        for contract in (
            "SAFETENSORS_INDEX",
            "weight_map",
            "GLM53_ENDPOINT_SHARDS",
            "header_size != expected_header_size",
            "_load_layer0_raw",
            "keys",
        ):
            self.assertIn(contract, loader)
        self.assertIn("backend='pread'", raw_loader)
        self.assertIn("_load_local_tensor", raw_loader)
        self.assertIn("embedding=raw['embed_tokens.weight']", loader)
        self.assertIn("lm_head=raw['lm_head.weight']", loader)

    def test_decode_and_nextn_prefill_workspace_contracts(self) -> None:
        self.assertEqual(
            glm53_flash.glm53_endpoint_workspace_shapes(1),
            glm53_flash.GLM53EndpointWorkspace(
                local_token=(1,),
                local_active=(1,),
                embedding=(1, 4096),
                streams=(1, 4, 4096),
                mean_f32=(1, 4096),
                collapsed=(1, 4096),
                normalized=(1, 4096),
                local_logits=(1, 38720),
                argmax=glm53_flash.GLM53DistributedArgmaxWorkspace(
                    local_values=(1,),
                    local_indices=(1,),
                    local_candidates=(1, 2),
                    gathered_candidates=(4, 2),
                    best_values=(1,),
                    select=(1,),
                ),
                token=(1,),
            ),
        )
        self.assertEqual(
            glm53_flash.glm53_endpoint_workspace_shapes(2).streams,
            (2, 4, 4096),
        )
        self.assertEqual(
            glm53_flash.glm53_endpoint_workspace_shapes(128),
            glm53_flash.GLM53EndpointWorkspace(
                local_token=(128,),
                local_active=(128,),
                embedding=(128, 4096),
                streams=(128, 4, 4096),
                mean_f32=(128, 4096),
                collapsed=(128, 4096),
                normalized=(128, 4096),
                local_logits=(128, 38720),
                argmax=glm53_flash.GLM53DistributedArgmaxWorkspace(
                    local_values=(128,),
                    local_indices=(128,),
                    local_candidates=(128, 2),
                    gathered_candidates=(512, 2),
                    best_values=(128,),
                    select=(128,),
                ),
                token=(128,),
            ),
        )
        width = glm53_flash.GLM53_TARGET_VERIFY_WIDTH
        self.assertEqual(
            glm53_flash.glm53_endpoint_workspace_shapes(width).streams,
            (width, 4, 4096),
        )
        self.assertEqual(
            glm53_flash.glm53_endpoint_workspace_shapes(1455).streams,
            (1455, 4, 4096),
        )
        for batch_size in (0, glm53_flash.KDA_CHUNK_SIZE + 1):
            with self.assertRaisesRegex(
                ValueError, rf"\[1, {glm53_flash.KDA_CHUNK_SIZE}\]"
            ):
                glm53_flash.glm53_endpoint_workspace_shapes(batch_size)

    def test_embedding_uses_full_vocab_for_dp_and_masked_reduce_for_tp(self) -> None:
        method = function(OPS_SOURCE, "decode_embedding")
        branch = next(node for node in method.body if isinstance(node, ast.If))
        branch_index = method.body.index(branch)
        self.assertEqual(ast.unparse(branch.test), "self._attention_tp_size == 1")
        dp_names = call_names(branch.body)
        tp_names = call_names(method.body[branch_index + 1 :])

        for name, count in (
            ("copy_", 2),
            ("index_select", 1),
            ("sub", 0),
            ("clamp", 0),
            ("ge", 0),
            ("lt", 0),
            ("mul_", 0),
            ("all_reduce", 0),
        ):
            self.assertEqual(dp_names.count(name), count)
        for name, count in (
            ("copy_", 1),
            ("index_select", 1),
            ("sub", 1),
            ("clamp", 1),
            ("ge", 1),
            ("lt", 1),
            ("mul_", 2),
            ("all_reduce", 1),
            ("all_gather_into_tensor", 0),
            ("broadcast", 0),
        ):
            self.assertEqual(tp_names.count(name), count)

        dp_source = ast.unparse(ast.Module(body=branch.body, type_ignores=[]))
        tp_source = ast.unparse(
            ast.Module(body=method.body[branch_index + 1 :], type_ignores=[])
        )
        self.assertIn(
            "torch.index_select(weights.embedding, 0, token_ids, out=w.embedding)",
            dp_source,
        )
        self.assertIn("w.streams.copy_(w.embedding.unsqueeze(1))", dp_source)
        self.assertLess(
            tp_source.index("torch.index_select"),
            tp_source.index("all_reduce(w.embedding)"),
        )
        self.assertIn("w.streams.copy_(embedding.unsqueeze(1))", tp_source)

    def test_prefill_embedding_sums_only_live_tokens(self) -> None:
        method = function(OPS_SOURCE, "prefill_embedding")
        self.assertEqual(
            [argument.arg for argument in method.args.args],
            ["self", "token_ids", "weights", "workspace", "process_group"],
        )
        source = ast.unparse(method)
        for total_tokens in (1, 7, 63, 64):
            self.assertIn(total_tokens, range(1, 65))
        self.assertIn("total_tokens = token_ids.shape[0]", source)
        self.assertIn("rows = slice(total_tokens)", source)
        self.assertIn("w.embedding[rows]", source)
        self.assertIn("dist.all_reduce(w.embedding", source)
        self.assertNotIn("[total_tokens:].zero_()", source)
        self.assertIn("('token_ids', token_ids)", source)
        self.assertIn("_workspace_tensors('weights', weights)", source)
        self.assertNotIn("torch.empty", source)
        self.assertNotIn("torch.zeros", source)

        collective = next(
            call for call in calls(method) if call_name(call) == "all_reduce"
        )
        self.assertEqual(ast.unparse(collective.args[0]), "w.embedding")
        self.assertEqual(
            {
                keyword.arg: ast.unparse(keyword.value)
                for keyword in collective.keywords
            },
            {"op": "dist.ReduceOp.SUM", "group": "process_group"},
        )

    def test_head_pins_rounding_math_gather_and_greedy_order(self) -> None:
        normalize = function(OPS_SOURCE, "normalize_head")
        method = function(OPS_SOURCE, "decode_head")
        nextn_helper = function(OPS_SOURCE, "_decode_glm53_head")
        project = function(OPS_SOURCE, "_project_glm53_head")
        normalize_names = [call_name(call) for call in calls(normalize)]
        project_names = [call_name(call) for call in calls(project)]

        for name, count in (
            ("add_", 3),
            ("_rmsnorm", 1),
            ("mm", 0),
            ("all_gather_into_tensor", 0),
            ("argmax", 0),
        ):
            self.assertEqual(normalize_names.count(name), count)
        for name, count in (
            ("mm", 1),
            ("glm53_distributed_argmax", 1),
            ("all_gather_into_tensor", 0),
            ("fill_", 0),
            ("argmax", 1),
            ("all_reduce", 0),
            ("broadcast", 0),
        ):
            self.assertEqual(project_names.count(name), count)
        normalize_source = ast.unparse(normalize)
        normalize_order = (
            "w.mean_f32.copy_(streams[:, 0])",
            "w.mean_f32.add_(streams[:, 1])",
            "w.mean_f32.add_(streams[:, 2])",
            "w.mean_f32.add_(streams[:, 3])",
            "w.mean_f32.mul_(0.25)",
            "w.collapsed.copy_(w.mean_f32)",
            "_rmsnorm(w.collapsed, weights.final_norm, w.normalized)",
        )
        self.assertEqual(
            sorted(normalize_source.index(item) for item in normalize_order),
            [normalize_source.index(item) for item in normalize_order],
        )
        method_source = ast.unparse(method)
        self.assertLess(
            method_source.index("self.normalize_head"),
            method_source.index("_project_glm53_head"),
        )
        nextn_source = ast.unparse(nextn_helper)
        self.assertLess(
            nextn_source.index("_rmsnorm"),
            nextn_source.index("_project_glm53_head"),
        )
        self.assertEqual(
            ast.unparse(project.body[0]),
            "torch.mm(normalized, weights.lm_head.t(), out=workspace.local_logits)",
        )
        branch = next(node for node in project.body if isinstance(node, ast.If))
        branch_index = project.body.index(branch)
        self.assertEqual(
            ast.unparse(branch.test), "weights.lm_head.shape[0] == VOCAB_SIZE"
        )
        dp_source = ast.unparse(ast.Module(body=branch.body, type_ignores=[]))
        tp_source = ast.unparse(
            ast.Module(body=project.body[branch_index + 1 :], type_ignores=[])
        )
        self.assertIn("workspace.local_logits[:, :TOKENIZER_VOCAB_SIZE]", dp_source)
        self.assertIn("torch.argmax", dp_source)
        self.assertNotIn("glm53_distributed_argmax", dp_source)
        self.assertIn(
            "glm53_distributed_argmax(workspace.local_logits, workspace.token, "
            "workspace.argmax, process_group)",
            tp_source,
        )
        self.assertNotIn("torch.argmax", tp_source)
        self.assertIsInstance(branch.body[-1], ast.Return)
        self.assertIsInstance(project.body[-1], ast.Return)
        project_source = ast.unparse(project)
        self.assertNotIn("workspace.gathered_logits", project_source)
        self.assertNotIn("workspace.logits", project_source)

    def test_validation_pins_cuda_layout_group_and_distinct_weights(self) -> None:
        validation = ast.unparse(function(OPS_SOURCE, "_validate_glm53_endpoint"))
        tensor_validation = ast.unparse(
            function(OPS_SOURCE, "_require_glm53_endpoint_tensors")
        )

        self.assertIn("dist.get_world_size(group) != TP_SIZE", validation)
        for dtype in ("torch.bfloat16", "torch.float32", "torch.int64"):
            self.assertIn(dtype, validation)
        self.assertIn(
            "weights.embedding.data_ptr() == weights.lm_head.data_ptr()", validation
        )
        self.assertIn("tensor.device != device", tensor_validation)
        self.assertIn("not tensor.is_cuda", tensor_validation)
        self.assertIn("not tensor.is_contiguous()", tensor_validation)

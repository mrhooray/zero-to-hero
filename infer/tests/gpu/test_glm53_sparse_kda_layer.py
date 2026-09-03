import ast
import importlib
import sys
import types
import unittest
from dataclasses import fields, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

import torch

from infer.models.glm53_flash import checkpoint as glm53_checkpoint
from infer.models.glm53_flash import model as glm53_flash

pytestmark = pytest.mark.gpu

ROOT = Path(__file__).resolve().parents[2]


def load_ops_module():
    distributed_argmax = types.ModuleType(
        "infer.models.glm53_flash.ops.distributed_argmax"
    )
    distributed_argmax.glm53_distributed_argmax = object()
    megafuse = types.ModuleType("infer.models.glm53_flash.ops.megafuse")
    megafuse.glm53_megafuse_decode = object()
    megafuse.glm53_megafuse_verify = object()
    prefill = types.ModuleType("infer.models.glm53_flash.ops.prefill_kda")
    prefill.GLM53PrefillMetadata = object
    prefill.glm53_prefill_kda = object()
    segmented = types.ModuleType("infer.models.glm53_flash.ops.segmented_conv")
    segmented.glm53_segmented_conv_prefill = object()
    sparse_prefill = types.ModuleType(
        "infer.models.glm53_flash.ops.sparse_mla_prefill"
    )
    sparse_prefill.GLM53StagedSparseMLAPrefillBatch = object
    sparse_prefill.glm53_sparse_mla_packed_prefill_append = object()
    modules = {
        "infer.models.glm53_flash.ops.distributed_argmax": distributed_argmax,
        "infer.models.glm53_flash.ops.megafuse": megafuse,
        "infer.models.glm53_flash.ops.prefill_kda": prefill,
        "infer.models.glm53_flash.ops.segmented_conv": segmented,
        "infer.models.glm53_flash.ops.sparse_mla_prefill": sparse_prefill,
    }
    with patch.dict(sys.modules, modules):
        sys.modules.pop("infer.models.glm53_flash.ops.core", None)
        return importlib.import_module("infer.models.glm53_flash.ops.core")


class SparseKDALayerLoaderTest(unittest.TestCase):
    def test_composite_loader_calls_each_component_once(self) -> None:
        self.assertEqual(
            tuple(field.name for field in fields(glm53_flash.GLM53SparseKDALayerWeights)),
            (
                "attention_mhc",
                "input_norm",
                "attention",
                "ffn_mhc",
                "post_attention_norm",
                "ffn",
            ),
        )
        shell = glm53_flash.GLM53SparseLayerShellWeights(
            attention_mhc=object(),
            input_norm=object(),
            ffn_mhc=object(),
            post_attention_norm=object(),
        )
        attention = object()
        ffn = object()
        with (
            patch.object(
                glm53_checkpoint,
                "load_glm53_sparse_layer_shell_weights",
                return_value=shell,
            ) as load_shell,
            patch.object(
                glm53_checkpoint,
                "load_glm53_sparse_kda_weights",
                return_value=attention,
            ) as load_attention,
            patch.object(
                glm53_checkpoint,
                "load_glm53_sparse_ffn_weights",
                return_value=ffn,
            ) as load_ffn,
        ):
            weights = glm53_checkpoint.load_glm53_sparse_kda_layer_weights(
                "/checkpoint", 4, 2, "cuda", attention_tp_size=1
            )

        expected = (Path("/checkpoint"), 4, 2, "cuda")
        load_shell.assert_called_once_with(*expected)
        load_attention.assert_called_once_with(*expected, 1)
        load_ffn.assert_called_once_with(*expected)
        self.assertIs(weights.attention_mhc, shell.attention_mhc)
        self.assertIs(weights.input_norm, shell.input_norm)
        self.assertIs(weights.attention, attention)
        self.assertIs(weights.ffn_mhc, shell.ffn_mhc)
        self.assertIs(weights.post_attention_norm, shell.post_attention_norm)
        self.assertIs(weights.ffn, ffn)

    def test_invalid_layer_and_rank_fail_before_io(self) -> None:
        with (
            patch.object(glm53_checkpoint, "load_glm53_sparse_layer_shell_weights") as shell,
            patch.object(glm53_checkpoint, "load_glm53_sparse_kda_weights") as attention,
            patch.object(glm53_checkpoint, "load_glm53_sparse_ffn_weights") as ffn,
        ):
            for layer_id in (-1, 0, 3, 7, 45):
                with self.assertRaisesRegex(ValueError, "not a main sparse KDA layer"):
                    glm53_checkpoint.load_glm53_sparse_kda_layer_weights(
                        "/missing", layer_id, 0, "cuda"
                    )
            for rank in (-1, 4):
                with self.assertRaisesRegex(ValueError, "tp_rank"):
                    glm53_checkpoint.load_glm53_sparse_kda_layer_weights(
                        "/missing", 4, rank, "cuda"
                    )

        shell.assert_not_called()
        attention.assert_not_called()
        ffn.assert_not_called()


class MetaTensor:
    def __init__(
        self,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        *,
        device: torch.device | None = None,
        contiguous: bool = True,
    ) -> None:
        self.shape = shape
        self.dtype = dtype
        self.device = device or torch.device("cuda:0")
        self.is_cuda = self.device.type == "cuda"
        self._contiguous = contiguous

    def is_contiguous(self) -> bool:
        return self._contiguous


class KDAValidationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.production = load_ops_module()

    def setUp(self) -> None:
        self.hidden = MetaTensor((1, 4096), torch.bfloat16)
        self.weights = glm53_flash.KDAWeights(
            projection=MetaTensor((6416, 4096), torch.bfloat16),
            gate_projections=MetaTensor((2, 2048, 128), torch.bfloat16),
            conv=MetaTensor((6144, 4), torch.float32),
            a_log=MetaTensor((16,), torch.float32),
            dt_bias=MetaTensor((2048,), torch.float32),
            o_norm=MetaTensor((128,), torch.bfloat16),
            o_projection=MetaTensor((4096, 2048), torch.bfloat16),
        )
        self.state = glm53_flash.KDAState(
            recurrent=MetaTensor((2, 16, 128, 128), torch.float32),
            conv=MetaTensor((2, 6144, 3), torch.bfloat16),
        )
        self.state_indices = MetaTensor((1,), torch.int32)
        self.cu_seqlens = MetaTensor((2,), torch.int32)
        self.workspace = glm53_flash.KDADecodeWorkspace(
            projection=MetaTensor((1, 6416), torch.bfloat16),
            output_gate=MetaTensor((1, 2048), torch.bfloat16),
            gate_raw=MetaTensor((1, 2048), torch.float32),
            local_output=MetaTensor((1, 16, 128), torch.bfloat16),
            output=MetaTensor((1, 4096), torch.bfloat16),
        )

    def validate(self, weights=None, state=None, workspace=None) -> int:
        return self.production._validate_kda_decode(
            self.hidden,
            weights or self.weights,
            state or self.state,
            self.state_indices,
            self.cu_seqlens,
            workspace or self.workspace,
        )

    def test_complete_pure_contract_accepts_exact_metadata(self) -> None:
        self.assertEqual(self.validate(), 1)

        with self.assertRaisesRegex(ValueError, "state.conv"):
            self.validate(
                state=replace(
                    self.state,
                    conv=MetaTensor((3, 6144, 3), torch.bfloat16),
                )
            )
        with self.assertRaisesRegex(ValueError, "weights.conv"):
            self.validate(
                weights=replace(
                    self.weights,
                    conv=MetaTensor((6144, 4), torch.bfloat16),
                )
            )
        with self.assertRaisesRegex(ValueError, "workspace.local_output"):
            self.validate(
                workspace=replace(
                    self.workspace,
                    local_output=MetaTensor(
                        (1, 16, 128), torch.bfloat16, contiguous=False
                    ),
                )
            )

    def test_post_state_failures_are_rejected_before_megafuse(self) -> None:
        engine = self.production.GLM53Ops()
        core = Mock()
        mm = Mock()
        malformed = (
            (
                replace(
                    self.weights,
                    o_projection=MetaTensor((4095, 2048), torch.bfloat16),
                ),
                self.workspace,
                "weights.o_projection",
            ),
            (
                self.weights,
                replace(
                    self.workspace,
                    output=MetaTensor((1, 4096), torch.float32),
                ),
                "workspace.output",
            ),
        )
        for weights, workspace, message in malformed:
            with (
                patch.object(self.production, "glm53_megafuse_decode", core),
                patch.object(self.production.torch, "mm", mm),
                self.assertRaisesRegex(ValueError, message),
            ):
                engine.decode_attention(
                    self.hidden,
                    weights,
                    self.state,
                    self.state_indices,
                    self.cu_seqlens,
                    workspace,
                    lambda output: output,
                )

        mm.assert_not_called()
        core.assert_not_called()


class SparseKDALayerCompositionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.production = load_ops_module()

    def setUp(self) -> None:
        self.engine = self.production.GLM53Ops()
        self.world = object()
        self.weights = glm53_flash.GLM53SparseKDALayerWeights(
            attention_mhc=object(),
            input_norm=object(),
            attention=object(),
            ffn_mhc=object(),
            post_attention_norm=object(),
            ffn=object(),
        )
        self.state = object()
        self.state_indices = object()
        self.cu_seqlens = object()
        self.all_reduce = object()
        self.kda_workspace = SimpleNamespace(output=self._bf16((1, 4096)))
        self.ffn_workspace = SimpleNamespace(output=self._bf16((1, 4096)))
        self.layer_workspace = SimpleNamespace(
            collapsed=self._bf16((1, 4096)),
            normalized=self._bf16((1, 4096)),
            streams_mid=self._bf16((1, 4, 4096)),
            streams_out=self._bf16((1, 4, 4096)),
        )

    @staticmethod
    def _bf16(shape: tuple[int, ...]) -> torch.Tensor:
        return torch.empty(shape, dtype=torch.bfloat16)

    def call(self, streams: torch.Tensor, group: object) -> torch.Tensor:
        return self.engine.decode_sparse_kda_layer(
            streams,
            self.weights,
            self.state,
            self.state_indices,
            self.cu_seqlens,
            self.kda_workspace,
            self.ffn_workspace,
            self.layer_workspace,
            group,
            self.all_reduce,
        )

    def test_exact_order_group_rank_collectives_and_return_owner(self) -> None:
        events: list[str] = []
        distributed = SimpleNamespace(
            group=SimpleNamespace(WORLD=self.world),
            get_world_size=lambda group: events.append("world_size") or 4,
            get_rank=lambda group: events.append("rank") or 2,
        )

        def mhc_pre(_streams, weights, _workspace, _rows) -> None:
            events.append(
                "attention_mhc" if weights is self.weights.attention_mhc else "ffn_mhc"
            )

        def rmsnorm(_input, weights, _output) -> None:
            events.append(
                "input_norm"
                if weights is self.weights.input_norm
                else "post_attention_norm"
            )

        def mhc_post(_hidden, _residual, _workspace, out, _rows) -> None:
            if out.data_ptr() == self.layer_workspace.streams_mid.data_ptr():
                events.append("attention_post")
            else:
                events.append("ffn_post")
                out.fill_(7)

        def attention(hidden, weights, state, indices, lengths, workspace, all_reduce):
            self.assertEqual(
                hidden.data_ptr(), self.layer_workspace.normalized.data_ptr()
            )
            self.assertIs(weights, self.weights.attention)
            self.assertIs(state, self.state)
            self.assertIs(indices, self.state_indices)
            self.assertIs(lengths, self.cu_seqlens)
            self.assertIs(workspace, self.kda_workspace)
            self.assertIs(all_reduce, self.all_reduce)
            events.append("attention_sum")
            return workspace.output

        def ffn(hidden, weights, workspace, rank, all_reduce):
            self.assertEqual(
                hidden.data_ptr(), self.layer_workspace.normalized.data_ptr()
            )
            self.assertIs(weights, self.weights.ffn)
            self.assertIs(workspace, self.ffn_workspace)
            self.assertEqual(rank, 2)
            self.assertIs(all_reduce, self.all_reduce)
            events.append("ffn_sum")
            return workspace.output

        with (
            patch.object(self.production, "dist", distributed),
            patch.object(
                self.production,
                "_validate_sparse_kda_layer",
                side_effect=lambda *_args: events.append("validate") or slice(1),
            ),
            patch.object(self.production, "_mhc_pre", side_effect=mhc_pre),
            patch.object(self.production, "_rmsnorm", side_effect=rmsnorm),
            patch.object(self.production, "_mhc_post", side_effect=mhc_post),
            patch.object(self.engine, "decode_attention", side_effect=attention),
            patch.object(self.engine, "decode_sparse_ffn", side_effect=ffn),
        ):
            returned = self.call(self.layer_workspace.streams_out, self.world)

        self.assertEqual(
            events,
            [
                "world_size",
                "rank",
                "validate",
                "attention_mhc",
                "input_norm",
                "attention_sum",
                "attention_post",
                "ffn_mhc",
                "post_attention_norm",
                "ffn_sum",
                "ffn_post",
            ],
        )
        self.assertEqual(
            returned.data_ptr(), self.layer_workspace.streams_out.data_ptr()
        )
        self.assertTrue(torch.equal(returned, torch.full_like(returned, 7)))

    def test_invalid_group_and_preflight_fail_before_state(self) -> None:
        core = Mock()
        distributed = SimpleNamespace(
            group=SimpleNamespace(WORLD=self.world),
            get_world_size=Mock(return_value=4),
            get_rank=Mock(return_value=1),
        )
        with (
            patch.object(self.production, "dist", distributed),
            patch.object(self.engine, "decode_attention", core),
            self.assertRaisesRegex(ValueError, "WORLD"),
        ):
            self.call(self.layer_workspace.streams_out, object())
        distributed.get_world_size.assert_not_called()
        distributed.get_rank.assert_not_called()

        distributed.get_world_size.return_value = 3
        with (
            patch.object(self.production, "dist", distributed),
            patch.object(self.engine, "decode_attention", core),
            self.assertRaisesRegex(ValueError, "TEP4"),
        ):
            self.call(self.layer_workspace.streams_out, self.world)
        distributed.get_rank.assert_not_called()

        distributed.get_world_size.return_value = 4
        with (
            patch.object(self.production, "dist", distributed),
            patch.object(
                self.production,
                "_validate_sparse_kda_layer",
                side_effect=ValueError("bad post-state output"),
            ),
            patch.object(self.engine, "decode_attention", core),
            self.assertRaisesRegex(ValueError, "bad post-state output"),
        ):
            self.call(self.layer_workspace.streams_out, self.world)
        core.assert_not_called()


class SparseKDAPrefillLayerCompositionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.production = load_ops_module()

    def test_live_lengths_and_resumed_state_order(self) -> None:
        world = object()
        distributed = SimpleNamespace(
            group=SimpleNamespace(WORLD=world),
            get_world_size=lambda group: 4,
            get_rank=lambda group: 3,
        )
        engine = self.production.GLM53Ops()
        weights = glm53_flash.GLM53SparseKDALayerWeights(
            attention_mhc=object(),
            input_norm=object(),
            attention=object(),
            ffn_mhc=object(),
            post_attention_norm=object(),
            ffn=object(),
        )

        for total_tokens in (1, 7, 63, 64):
            with self.subTest(total_tokens=total_tokens):
                events: list[str] = []
                state_indices = torch.tensor([5], dtype=torch.int32)
                has_initial = torch.tensor([True], dtype=torch.bool)
                batch = SimpleNamespace(
                    metadata=SimpleNamespace(total_tokens=total_tokens, batch_size=1),
                    state_indices=state_indices,
                    has_initial=has_initial,
                )
                streams = torch.full((total_tokens, 4, 4096), 9, dtype=torch.bfloat16)
                layer = SimpleNamespace(
                    collapsed=torch.empty((total_tokens, 4096), dtype=torch.bfloat16),
                    normalized=torch.empty((total_tokens, 4096), dtype=torch.bfloat16),
                    streams_mid=torch.full(
                        (total_tokens, 4, 4096), 9, dtype=torch.bfloat16
                    ),
                    streams_out=torch.full(
                        (total_tokens, 4, 4096), 9, dtype=torch.bfloat16
                    ),
                )
                kda = SimpleNamespace(output=torch.empty((total_tokens, 4096)))
                ffn = SimpleNamespace(output=torch.full((total_tokens, 4096), 5))

                def mhc_pre(_streams, mhc, _workspace, _rows, events=events):
                    events.append(
                        "attention_mhc" if mhc is weights.attention_mhc else "ffn_mhc"
                    )

                def rmsnorm(_input, norm, _output, events=events):
                    events.append(
                        "input_norm" if norm is weights.input_norm else "post_norm"
                    )

                def attention(
                    hidden,
                    attention_weights,
                    state,
                    receipt,
                    workspace,
                    total_tokens=total_tokens,
                    batch=batch,
                    kda=kda,
                    state_indices=state_indices,
                    has_initial=has_initial,
                    events=events,
                ):
                    self.assertEqual(hidden.shape, (total_tokens, 4096))
                    self.assertIs(attention_weights, weights.attention)
                    self.assertIs(receipt, batch)
                    self.assertIs(workspace, kda)
                    self.assertIs(receipt.state_indices, state_indices)
                    self.assertIs(receipt.has_initial, has_initial)
                    events.append("attention_sum")
                    return torch.full((total_tokens, 4096), 2, dtype=torch.bfloat16)

                def mhc_post(
                    _hidden,
                    _residual,
                    _workspace,
                    out,
                    _rows,
                    layer=layer,
                    events=events,
                ):
                    if out.data_ptr() == layer.streams_mid.data_ptr():
                        events.append("attention_post")
                        out.fill_(3)
                    else:
                        events.append("ffn_post")
                        out.fill_(7)

                nccl = object()

                def sparse_ffn(
                    hidden,
                    ffn_weights,
                    workspace,
                    rank,
                    all_reduce,
                    ffn=ffn,
                    events=events,
                    expected_all_reduce=nccl,
                    total_tokens=total_tokens,
                ):
                    self.assertEqual(hidden.shape, (total_tokens, 4096))
                    self.assertIs(ffn_weights, weights.ffn)
                    self.assertIs(workspace, ffn)
                    self.assertEqual(rank, 3)
                    self.assertIs(all_reduce, expected_all_reduce)
                    events.append("ffn_sum")
                    return workspace.output

                with (
                    patch.object(self.production, "dist", distributed),
                    patch.object(
                        self.production,
                        "_validate_sparse_kda_prefill_layer",
                        return_value=total_tokens,
                    ),
                    patch.object(self.production, "_mhc_pre", side_effect=mhc_pre),
                    patch.object(self.production, "_rmsnorm", side_effect=rmsnorm),
                    patch.object(self.production, "_mhc_post", side_effect=mhc_post),
                    patch.object(engine, "prefill_attention", side_effect=attention),
                    patch.object(engine, "decode_sparse_ffn", side_effect=sparse_ffn),
                ):
                    returned = engine.prefill_sparse_kda_layer(
                        streams,
                        weights,
                        object(),
                        batch,
                        kda,
                        ffn,
                        layer,
                        world,
                        nccl,
                    )

                self.assertEqual(
                    events,
                    [
                        "attention_mhc",
                        "input_norm",
                        "attention_sum",
                        "attention_post",
                        "ffn_mhc",
                        "post_norm",
                        "ffn_sum",
                        "ffn_post",
                    ],
                )
                self.assertIs(returned, layer.streams_out)
                self.assertTrue(torch.all(returned == 7))
                self.assertTrue(torch.all(streams[total_tokens:] == 0))

    def test_world_and_preflight_fail_before_state_or_stream_writes(self) -> None:
        world = object()
        distributed = SimpleNamespace(
            group=SimpleNamespace(WORLD=world),
            get_world_size=Mock(return_value=4),
            get_rank=Mock(return_value=0),
        )
        engine = self.production.GLM53Ops()
        streams = torch.full((64, 4, 4096), 9, dtype=torch.bfloat16)
        state_op = Mock()
        arguments = (
            streams,
            object(),
            object(),
            object(),
            object(),
            object(),
            object(),
        )
        with (
            patch.object(self.production, "dist", distributed),
            patch.object(engine, "prefill_attention", state_op),
            self.assertRaisesRegex(ValueError, "WORLD"),
        ):
            engine.prefill_sparse_kda_layer(*arguments, object(), object())
        state_op.assert_not_called()
        self.assertTrue(torch.all(streams == 9))

        with (
            patch.object(self.production, "dist", distributed),
            patch.object(
                self.production,
                "_validate_sparse_kda_prefill_layer",
                side_effect=ValueError("unsafe alias"),
            ),
            patch.object(engine, "prefill_attention", state_op),
            self.assertRaisesRegex(ValueError, "unsafe alias"),
        ):
            engine.prefill_sparse_kda_layer(*arguments, world, object())
        state_op.assert_not_called()
        self.assertTrue(torch.all(streams == 9))


class SparseKDALayerAliasTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.production = load_ops_module()

    @staticmethod
    def tensor(size: int = 16, dtype: torch.dtype = torch.uint8) -> torch.Tensor:
        return torch.empty(size, dtype=dtype)

    def fixture(self):
        streams = self.tensor(32, torch.bfloat16).view(1, 4, 8)
        attention = glm53_flash.KDAWeights(*(self.tensor() for _ in fields(glm53_flash.KDAWeights)))
        ffn = SimpleNamespace(router=self.tensor(), router_t=self.tensor())
        weights = glm53_flash.GLM53SparseKDALayerWeights(
            attention_mhc=SimpleNamespace(
                base=self.tensor(), fn=self.tensor(), scale=self.tensor()
            ),
            input_norm=self.tensor(),
            attention=attention,
            ffn_mhc=SimpleNamespace(
                base=self.tensor(), fn=self.tensor(), scale=self.tensor()
            ),
            post_attention_norm=self.tensor(),
            ffn=ffn,
        )
        state = glm53_flash.KDAState(recurrent=self.tensor(), conv=self.tensor())
        descriptors = (
            ("state_indices", self.tensor(dtype=torch.int32)),
            ("cu_seqlens", self.tensor(dtype=torch.int32)),
        )
        kda = glm53_flash.KDADecodeWorkspace(
            *(self.tensor() for _ in fields(glm53_flash.KDADecodeWorkspace))
        )
        ffn_workspace = SimpleNamespace(scores=self.tensor(), output=self.tensor())
        layer = SimpleNamespace(
            mhc_sqrsum=self.tensor().view(1, -1),
            mhc_dot=self.tensor().view(1, -1),
            post=self.tensor().view(1, -1),
            comb=self.tensor().view(1, -1),
            collapsed=self.tensor().view(1, -1),
            normalized=self.tensor().view(1, -1),
            streams_mid=self.tensor().view(1, -1),
            streams_out=streams,
        )
        return (
            streams,
            weights,
            state,
            descriptors,
            kda,
            ffn_workspace,
            layer,
        )

    def validate(self, values) -> None:
        self.production._validate_sparse_kda_layer_aliases(*values, slice(1))

    def test_prefill_embedding_rejects_token_workspace_alias_before_write(self) -> None:
        engine = self.production.GLM53Ops()
        world = object()
        token_ids = torch.tensor([7], dtype=torch.int64)
        workspace = SimpleNamespace(local_token=token_ids)
        distributed = SimpleNamespace(group=SimpleNamespace(WORLD=world))

        with (
            patch.object(self.production, "dist", distributed),
            patch.object(self.production, "_validate_glm53_endpoint"),
            patch.object(self.production, "_require_glm53_endpoint_tensors"),
            self.assertRaisesRegex(ValueError, "workspace.local_token and token_ids"),
        ):
            engine.prefill_embedding(
                token_ids,
                SimpleNamespace(embedding=SimpleNamespace(device=token_ids.device)),
                workspace,
                world,
            )
        self.assertEqual(token_ids.item(), 7)

    def test_prefill_embedding_collective_uses_live_length(self) -> None:
        engine = self.production.GLM53Ops()
        world = object()
        embedding = torch.arange(64 * 3, dtype=torch.bfloat16).view(64, 3)

        for total_tokens in (1, 7, 63, 64):
            with self.subTest(total_tokens=total_tokens):
                collectives = []

                def all_reduce(
                    tensor,
                    *,
                    op,
                    group,
                    total_tokens=total_tokens,
                    collectives=collectives,
                ):
                    self.assertIs(group, world)
                    self.assertEqual(tensor.shape, (total_tokens, 3))
                    collectives.append((op, group))

                distributed = SimpleNamespace(
                    group=SimpleNamespace(WORLD=world),
                    ReduceOp=SimpleNamespace(SUM=object()),
                    get_rank=lambda group: 0,
                    all_reduce=all_reduce,
                )
                token_ids = torch.arange(total_tokens, dtype=torch.int64)
                workspace = SimpleNamespace(
                    local_token=torch.empty(total_tokens, dtype=torch.int64),
                    local_active=torch.empty(total_tokens, dtype=torch.bool),
                    embedding=torch.full((total_tokens, 3), 9, dtype=torch.bfloat16),
                    streams=torch.full((total_tokens, 4, 3), 9, dtype=torch.bfloat16),
                )
                with (
                    patch.object(self.production, "dist", distributed),
                    patch.object(self.production, "LOCAL_VOCAB_SIZE", 64),
                    patch.object(self.production, "_validate_glm53_endpoint"),
                    patch.object(self.production, "_require_glm53_endpoint_tensors"),
                ):
                    result = engine.prefill_embedding(
                        token_ids,
                        SimpleNamespace(embedding=embedding),
                        workspace,
                        world,
                    )

                self.assertEqual(len(collectives), 1)
                expected = embedding[token_ids].unsqueeze(1).expand(-1, 4, -1)
                self.assertTrue(torch.equal(result, expected))

    def test_exact_streams_out_alias_is_the_only_stream_exception(self) -> None:
        values = self.fixture()
        self.validate(values)

        storage = self.tensor(33, torch.bfloat16)
        streams = storage[:-1].view(1, 4, 8)
        layer = values[-1]
        layer.streams_out = storage[1:].view(1, 4, 8)
        with self.assertRaisesRegex(
            ValueError, "streams and layer.streams_out overlap"
        ):
            self.validate((streams, *values[1:-1], layer))

    def test_rejects_state_scratch_internal_and_weight_overlap(self) -> None:
        values = self.fixture()
        streams, weights, state, descriptors, kda, ffn, layer = values
        with self.assertRaisesRegex(
            ValueError, "state.recurrent and kda.projection overlap"
        ):
            self.validate(
                (
                    streams,
                    weights,
                    replace(state, recurrent=kda.projection),
                    descriptors,
                    kda,
                    ffn,
                    layer,
                )
            )

        with self.assertRaisesRegex(
            ValueError, "kda.projection and kda.output overlap"
        ):
            self.validate(
                (
                    streams,
                    weights,
                    state,
                    descriptors,
                    replace(kda, output=kda.projection),
                    ffn,
                    layer,
                )
            )

        aliased_weights = replace(
            weights,
            attention=replace(weights.attention, projection=state.recurrent),
        )
        with self.assertRaisesRegex(
            ValueError, "state.recurrent and weights.attention.projection overlap"
        ):
            self.validate(
                (
                    streams,
                    aliased_weights,
                    state,
                    descriptors,
                    kda,
                    ffn,
                    layer,
                )
            )

    def test_rejects_prefill_descriptor_overlap(self) -> None:
        streams, weights, state, descriptors, kda, ffn, layer = self.fixture()
        descriptors = (*descriptors, ("batch.has_initial", kda.projection))

        with self.assertRaisesRegex(
            ValueError, "kda.projection and batch.has_initial overlap"
        ):
            self.validate((streams, weights, state, descriptors, kda, ffn, layer))


class SparseKDALayerSourceTest(unittest.TestCase):
    def test_source_pins_abi_order_collectives_and_no_allocations(self) -> None:
        tree = ast.parse(
            (ROOT / "src/infer/models/glm53_flash/ops/core.py").read_text()
        )
        functions = {
            node.name: node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
        }
        method = functions["decode_sparse_kda_layer"]
        self.assertEqual(
            [argument.arg for argument in method.args.args],
            [
                "self",
                "streams",
                "weights",
                "state",
                "state_indices",
                "cu_seqlens",
                "kda_workspace",
                "ffn_workspace",
                "layer_workspace",
                "process_group",
                "all_reduce",
            ],
        )
        self.assertEqual(
            statement_calls(method),
            [
                "_sparse_tep_rank",
                "_validate_sparse_kda_layer",
                "_mhc_pre",
                "_rmsnorm",
                "decode_attention",
                "_mhc_post",
                "_mhc_pre",
                "_rmsnorm",
                "decode_sparse_ffn",
                "_mhc_post",
            ],
        )
        allocation_names = {"empty", "empty_like", "zeros", "zeros_like", "new_empty"}
        self.assertFalse(
            allocation_names
            & {
                call_name(node)
                for node in ast.walk(method)
                if isinstance(node, ast.Call)
            }
        )

        attention = functions["decode_attention"]
        self.assertEqual(
            [argument.arg for argument in attention.args.args],
            [
                "self",
                "hidden_states",
                "weights",
                "state",
                "state_indices",
                "cu_seqlens",
                "workspace",
                "all_reduce",
            ],
        )
        self.assertEqual(
            statement_calls(attention)[:4],
            ["_validate_kda_decode", "mm", "mm", "zero_"],
        )
        self.assertIn("_project_kda_output", statement_calls(attention))

    def test_prefill_source_pins_b64_order_state_receipt_and_no_allocations(
        self,
    ) -> None:
        tree = ast.parse(
            (ROOT / "src/infer/models/glm53_flash/ops/core.py").read_text()
        )
        functions = {
            node.name: node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
        }
        method = functions["prefill_sparse_kda_layer"]
        self.assertEqual(
            [argument.arg for argument in method.args.args],
            [
                "self",
                "streams",
                "weights",
                "state",
                "batch",
                "kda_workspace",
                "ffn_workspace",
                "layer_workspace",
                "process_group",
                "all_reduce",
            ],
        )
        source = ast.unparse(method)
        ordered = (
            "_validate_sparse_kda_prefill_layer",
            "_mhc_pre",
            "_rmsnorm",
            "prefill_attention",
            "_mhc_post",
            "_mhc_pre",
            "_rmsnorm",
            "decode_sparse_ffn",
            "_mhc_post",
        )
        cursor = -1
        for name in ordered:
            cursor = source.index(name, cursor + 1)
        self.assertIn("batch", source)
        self.assertIn("rows = slice(total_tokens)", source)
        self.assertFalse(
            {"empty", "empty_like", "zeros", "zeros_like", "new_empty"}
            & {
                call_name(node)
                for node in ast.walk(method)
                if isinstance(node, ast.Call)
            }
        )
        ffn = functions["decode_sparse_ffn"]
        self.assertEqual(
            [argument.arg for argument in ffn.args.args],
            [
                "self",
                "hidden_states",
                "weights",
                "workspace",
                "tep_rank",
                "all_reduce",
            ],
        )
        self.assertEqual(
            sum(
                isinstance(node.func, ast.Name) and node.func.id == "all_reduce"
                for node in ast.walk(ffn)
                if isinstance(node, ast.Call)
            ),
            1,
        )


def statement_calls(function: ast.FunctionDef) -> list[str]:
    calls = []
    for statement in function.body:
        value = getattr(statement, "value", None)
        if isinstance(value, ast.Call):
            calls.append(call_name(value))
    return calls


def call_name(call: ast.Call) -> str:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    raise TypeError("unexpected call")


if __name__ == "__main__":
    unittest.main()

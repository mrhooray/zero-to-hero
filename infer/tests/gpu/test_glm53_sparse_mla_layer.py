import ast
import hashlib
import importlib
import sys
import types
import unittest
from dataclasses import fields, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

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


class SparseMLALayerLoaderTest(unittest.TestCase):
    def test_all_sparse_shell_routes_match_pinned_index(self) -> None:
        self.assertEqual(
            tuple(field.name for field in fields(glm53_flash.GLM53SparseLayerShellWeights)),
            (
                "attention_mhc",
                "input_norm",
                "ffn_mhc",
                "post_attention_norm",
            ),
        )
        self.assertEqual(
            tuple(
                (spec.name, spec.shape, spec.dtype, spec.shard_axis)
                for spec in glm53_checkpoint.SPARSE_LAYER_SHELL_WEIGHTS
            ),
            (
                ("hc_attn_base", (24,), "F32", None),
                ("hc_attn_fn", (24, 16384), "BF16", None),
                ("hc_attn_scale", (3,), "F32", None),
                ("hc_ffn_base", (24,), "F32", None),
                ("hc_ffn_fn", (24, 16384), "BF16", None),
                ("hc_ffn_scale", (3,), "F32", None),
                ("input_layernorm.weight", (4096,), "BF16", None),
                ("post_attention_layernorm.weight", (4096,), "BF16", None),
            ),
        )
        expected = {
            layer_id: (
                f"model-{primary:05d}-of-00062.safetensors",
                f"model-{secondary:05d}-of-00062.safetensors",
            )
            for layer_id, primary, secondary in (
                (3, 31, 32),
                (4, 46, 47),
                (5, 54, 56),
                (6, 56, 57),
                (7, 57, 59),
                (8, 59, 60),
                (9, 60, 61),
                (10, 3, 4),
                (11, 4, 5),
                (12, 5, 7),
                (13, 7, 8),
                (14, 8, 10),
                (15, 10, 11),
                (16, 11, 12),
                (17, 12, 14),
                (18, 14, 15),
                (19, 15, 17),
                (20, 17, 18),
                (21, 18, 19),
                (22, 19, 21),
                (23, 21, 22),
                (24, 22, 24),
                (25, 24, 25),
                (26, 25, 26),
                (27, 26, 28),
                (28, 28, 29),
                (29, 29, 31),
                (30, 32, 33),
                (31, 33, 35),
                (32, 35, 36),
                (33, 36, 38),
                (34, 38, 39),
                (35, 39, 40),
                (36, 40, 42),
                (37, 42, 43),
                (38, 43, 45),
                (39, 45, 46),
                (40, 47, 49),
                (41, 49, 50),
                (42, 50, 52),
                (43, 52, 53),
                (44, 53, 54),
            )
        }
        self.assertEqual(glm53_checkpoint.SPARSE_LAYER_SHELL_SHARDS, expected)
        self.assertIs(
            glm53_checkpoint.SPARSE_LAYER_SHARD_PINS,
            glm53_checkpoint.SPARSE_FFN_SHARD_PINS,
        )
        shell_shards = sorted(
            {filename for pair in expected.values() for filename in pair}
        )
        pin_evidence = "\n".join(
            f"{filename}:{glm53_checkpoint.SPARSE_LAYER_SHARD_PINS[filename][0]}:"
            f"{glm53_checkpoint.SPARSE_LAYER_SHARD_PINS[filename][1]}"
            for filename in shell_shards
        )
        self.assertEqual(len(shell_shards), 43)
        self.assertEqual(
            hashlib.sha256(pin_evidence.encode()).hexdigest(),
            "0244b2f137198c369da15e762237e31390ad27ff109100dc1448960f9cb4f88d",
        )

        for layer_id, (primary, secondary) in expected.items():
            prefix = glm53_checkpoint.DENSE_LAYER_PREFIX.format(layer_id=layer_id)
            specs = glm53_checkpoint._sparse_layer_shell_checkpoint_specs(layer_id)
            shards = glm53_checkpoint._sparse_layer_shell_checkpoint_shards(layer_id)
            self.assertEqual(len(specs), 8)
            self.assertEqual(tuple(shards.values()).count(primary), 7)
            self.assertEqual(tuple(shards.values()).count(secondary), 1)
            self.assertEqual(
                shards[f"{prefix}post_attention_layernorm.weight"], secondary
            )
            self.assertTrue(
                set(shards.values()) <= glm53_checkpoint.SPARSE_LAYER_SHARD_PINS.keys()
            )
            self.assertTrue(
                all(
                    size > 0 and header > 0
                    for filename in (primary, secondary)
                    for size, header in (
                        glm53_checkpoint.SPARSE_LAYER_SHARD_PINS[filename],
                    )
                )
            )

    def test_nonadjacent_shell_loader_routes_and_packs_layer44(self) -> None:
        layer_id = 44
        prefix = glm53_checkpoint.DENSE_LAYER_PREFIX.format(layer_id=layer_id)
        raw = {
            spec.name: torch.zeros(
                1,
                dtype=torch.float32 if spec.dtype == "F32" else torch.bfloat16,
            )
            for spec in glm53_checkpoint.SPARSE_LAYER_SHELL_WEIGHTS
        }
        keyed = {prefix + name: tensor for name, tensor in raw.items()}

        with patch.object(
            glm53_checkpoint, "_load_indexed_weights", return_value=keyed
        ) as load:
            weights = glm53_checkpoint.load_glm53_sparse_layer_shell_weights(
                "/checkpoint", layer_id, 2, "cuda"
            )

        args = load.call_args.args
        self.assertEqual(args[:3], (Path("/checkpoint"), 2, "cuda"))
        self.assertEqual(set(args[3]), set(keyed))
        self.assertEqual(
            args[4], glm53_checkpoint._sparse_layer_shell_checkpoint_shards(layer_id)
        )
        self.assertIs(args[5], glm53_checkpoint.SPARSE_LAYER_SHARD_PINS)
        self.assertEqual(args[6], glm53_checkpoint._sparse_layer_shell_scopes(layer_id))
        self.assertEqual(args[7], "sparse layer 44 shell")

        self.assertIs(weights.attention_mhc.base, raw["hc_attn_base"])
        self.assertEqual(weights.attention_mhc.fn.dtype, torch.float32)
        self.assertIs(weights.attention_mhc.scale, raw["hc_attn_scale"])
        self.assertIs(weights.input_norm, raw["input_layernorm.weight"])
        self.assertIs(weights.ffn_mhc.base, raw["hc_ffn_base"])
        self.assertEqual(weights.ffn_mhc.fn.dtype, torch.float32)
        self.assertIs(weights.ffn_mhc.scale, raw["hc_ffn_scale"])
        self.assertIs(
            weights.post_attention_norm, raw["post_attention_layernorm.weight"]
        )

    def test_shell_scope_ignores_attention_and_rejects_extra_shell_key(self) -> None:
        layer_id = 44
        expected = glm53_checkpoint._sparse_layer_shell_checkpoint_shards(layer_id)
        weight_map = {
            **expected,
            f"{glm53_checkpoint.SPARSE_KDA_PREFIX.format(layer_id=layer_id)}q_proj.weight": (
                glm53_checkpoint.SPARSE_KDA_DEFAULT_SHARDS[layer_id]
            ),
        }
        glm53_checkpoint._validate_indexed_weight_map(
            weight_map,
            expected,
            glm53_checkpoint._sparse_layer_shell_scopes(layer_id),
            "sparse layer 44 shell",
        )
        extra = (
            f"{glm53_checkpoint.DENSE_LAYER_PREFIX.format(layer_id=layer_id)}hc_attn_extra"
        )
        weight_map[extra] = glm53_checkpoint.SPARSE_LAYER_SHELL_SHARDS[layer_id][0]
        with self.assertRaisesRegex(ValueError, "hc_attn_extra"):
            glm53_checkpoint._validate_indexed_weight_map(
                weight_map,
                expected,
                glm53_checkpoint._sparse_layer_shell_scopes(layer_id),
                "sparse layer 44 shell",
            )

    def test_all_mla_layers_have_one_exact_26_key_route(self) -> None:
        for layer_id in glm53_flash.MAIN_SPARSE_MLA_LAYER_IDS:
            primary, secondary = glm53_checkpoint.SPARSE_LAYER_SHELL_SHARDS[layer_id]
            specs = glm53_checkpoint._sparse_mla_layer_checkpoint_specs(layer_id)
            shards = glm53_checkpoint._sparse_mla_layer_checkpoint_shards(layer_id)
            attention_prefix = glm53_checkpoint.SPARSE_MLA_PREFIX.format(
                layer_id=layer_id
            )

            self.assertEqual(len(specs), 26)
            self.assertEqual(set(specs), set(shards))
            self.assertEqual(len({spec.name for spec in specs.values()}), 26)
            self.assertEqual(tuple(shards.values()).count(primary), 7)
            self.assertEqual(tuple(shards.values()).count(secondary), 19)
            self.assertEqual(sum(key.startswith(attention_prefix) for key in specs), 18)
            self.assertTrue(
                all(
                    shards[key] == secondary
                    for key in specs
                    if key.startswith(attention_prefix)
                )
            )

    def test_generic_mla_loader_reads_26_once_and_reuses_kv_b(self) -> None:
        layer_id = 35
        specs = glm53_checkpoint._sparse_mla_layer_checkpoint_specs(layer_id)
        keyed = {key: object() for key in specs}
        kv_b = MagicMock()
        kv_b.ndim = 2
        kv_b.shape = (8192, 512)
        kv_b_heads = MagicMock()
        w_kc_slice = MagicMock()
        w_vc_slice = MagicMock()
        w_vc_transposed = MagicMock()
        w_kc = object()
        w_vc = object()
        kv_b.view.return_value = kv_b_heads
        kv_b_heads.__getitem__.side_effect = (w_kc_slice, w_vc_slice)
        w_kc_slice.contiguous.return_value = w_kc
        w_vc_slice.transpose.return_value = w_vc_transposed
        w_vc_transposed.contiguous.return_value = w_vc
        kv_b_key = glm53_checkpoint.sparse_mla_checkpoint_key(
            layer_id,
            next(
                spec
                for spec in glm53_checkpoint.SPARSE_MLA_WEIGHTS
                if spec.name == "kv_b_proj.weight"
            ),
        )
        keyed[kv_b_key] = kv_b
        raw = {spec.name: keyed[key] for key, spec in specs.items()}
        shell = glm53_flash.GLM53SparseLayerShellWeights(
            attention_mhc=object(),
            input_norm=object(),
            ffn_mhc=object(),
            post_attention_norm=object(),
        )
        projection_weights = object()
        ffn_weights = object()

        with (
            patch.object(
                glm53_checkpoint, "_load_indexed_weights", return_value=keyed
            ) as load,
            patch.object(
                glm53_checkpoint,
                "_pack_sparse_layer_shell_weights",
                return_value=shell,
            ) as pack_shell,
            patch.object(
                glm53_checkpoint,
                "pack_sparse_mla_projection_weights",
                return_value=projection_weights,
            ) as pack_projection,
            patch.object(
                glm53_checkpoint,
                "load_glm53_sparse_ffn_weights",
                return_value=ffn_weights,
            ) as load_ffn,
        ):
            weights = glm53_checkpoint.load_glm53_sparse_mla_layer_weights(
                "/checkpoint", layer_id, 2, "cuda"
            )

        args = load.call_args.args
        self.assertEqual(args[:3], (Path("/checkpoint"), 2, "cuda"))
        self.assertEqual(len(args[3]), 26)
        self.assertEqual(set(args[3]), set(keyed))
        self.assertEqual(
            args[4], glm53_checkpoint._sparse_mla_layer_checkpoint_shards(layer_id)
        )
        self.assertIs(args[5], glm53_checkpoint.SPARSE_LAYER_SHARD_PINS)
        self.assertEqual(
            args[6],
            (
                *glm53_checkpoint._sparse_layer_shell_scopes(layer_id),
                glm53_checkpoint.SPARSE_MLA_PREFIX.format(layer_id=layer_id),
            ),
        )
        self.assertEqual(args[7], "sparse MLA layer 35")
        pack_shell.assert_called_once_with(raw, torch)
        pack_projection.assert_called_once_with(raw, torch)
        kv_b.view.assert_called_once_with(16, 512, 512)
        self.assertIs(weights.attention_mhc, shell.attention_mhc)
        self.assertIs(weights.input_norm, shell.input_norm)
        self.assertIs(weights.mla_decode.w_kc, w_kc)
        self.assertIs(
            weights.mla_decode.pool_ape,
            raw["indexer.index_kpool_compress_ape"],
        )
        self.assertIs(weights.mla_output.w_vc, w_vc)
        self.assertIs(weights.mla_output.o_proj, raw["o_proj.weight"])
        self.assertIs(
            weights.mla_output.o_proj_scale_inv,
            raw["o_proj.weight_scale_inv"],
        )
        self.assertIs(weights.mla_projection, projection_weights)
        self.assertIs(weights.ffn_mhc, shell.ffn_mhc)
        self.assertIs(weights.post_attention_norm, shell.post_attention_norm)
        self.assertIs(weights.ffn, ffn_weights)
        load_ffn.assert_called_once_with(Path("/checkpoint"), layer_id, 2, "cuda")

    def test_combined_scope_rejects_missing_extra_and_wrong_mapping(self) -> None:
        layer_id = 39
        expected = glm53_checkpoint._sparse_mla_layer_checkpoint_shards(layer_id)
        weight_map = dict(expected)
        missing = next(iter(weight_map))
        weight_map.pop(missing)
        wrong = next(
            key
            for key in weight_map
            if key.startswith(
                glm53_checkpoint.SPARSE_MLA_PREFIX.format(layer_id=layer_id)
            )
        )
        weight_map[wrong] = glm53_checkpoint.SPARSE_LAYER_SHELL_SHARDS[layer_id][0]
        extra = glm53_checkpoint.SPARSE_MLA_PREFIX.format(
            layer_id=layer_id
        ) + "unexpected.weight"
        weight_map[extra] = glm53_checkpoint.SPARSE_LAYER_SHELL_SHARDS[layer_id][1]

        glm53_checkpoint._validate_indexed_weight_map(
            {
                **expected,
                "model.visual.ignored": "model-00001-of-00062.safetensors",
            },
            expected,
            (
                *glm53_checkpoint._sparse_layer_shell_scopes(layer_id),
                glm53_checkpoint.SPARSE_MLA_PREFIX.format(layer_id=layer_id),
            ),
            "sparse MLA layer 39",
        )
        with self.assertRaises(ValueError) as error:
            glm53_checkpoint._validate_indexed_weight_map(
                weight_map,
                expected,
                (
                    *glm53_checkpoint._sparse_layer_shell_scopes(layer_id),
                    glm53_checkpoint.SPARSE_MLA_PREFIX.format(layer_id=layer_id),
                ),
                "sparse MLA layer 39",
            )

        message = str(error.exception)
        self.assertIn(f"missing {missing}", message)
        self.assertIn(f"unexpected {extra}", message)
        self.assertIn(f"{wrong} maps to", message)

    def test_invalid_layer_ids_fail_before_io(self) -> None:
        with (
            patch.object(glm53_checkpoint, "_load_indexed_weights") as load,
            patch.object(glm53_checkpoint, "load_glm53_sparse_ffn_weights") as load_ffn,
        ):
            for layer_id in (-1, 0, 2, 45, 46):
                with self.assertRaisesRegex(ValueError, "not a main sparse layer"):
                    glm53_checkpoint.load_glm53_sparse_layer_shell_weights(
                        "/missing", layer_id, 0, "cuda"
                    )
            for layer_id in (-1, 0, 4, 44, 45):
                with self.assertRaisesRegex(ValueError, "not a main sparse MLA layer"):
                    glm53_checkpoint.load_glm53_sparse_mla_layer_weights(
                        "/missing", layer_id, 0, "cuda"
                    )

        load.assert_not_called()
        load_ffn.assert_not_called()

    def test_full_model_dispatch_inventory_matches_pinned_index(self) -> None:
        owned: list[tuple[str, str]] = []
        owned.extend(
            ("endpoint", spec.checkpoint_key)
            for spec in glm53_checkpoint.GLM53_ENDPOINT_WEIGHTS
        )
        for layer_id in glm53_flash.DENSE_LAYER_IDS:
            owned.extend(
                ("dense", key)
                for key in glm53_checkpoint._dense_layer_checkpoint_specs(layer_id)
            )
        for layer_id in glm53_flash.SPARSE_FFN_LAYER_IDS:
            owned.extend(
                ("shell", key)
                for key in glm53_checkpoint._sparse_layer_shell_checkpoint_specs(
                    layer_id
                )
            )
            owned.extend(
                ("ffn", key)
                for key, _spec, _expert_id in glm53_checkpoint._sparse_ffn_checkpoint_specs(
                    layer_id
                )
            )
        for layer_id in glm53_flash.SPARSE_KDA_LAYER_IDS:
            owned.extend(
                ("kda", key)
                for key in glm53_checkpoint._sparse_kda_checkpoint_specs(layer_id)
            )
        for layer_id in glm53_flash.MAIN_SPARSE_MLA_LAYER_IDS:
            owned.extend(
                ("mla", glm53_checkpoint.sparse_mla_checkpoint_key(layer_id, spec))
                for spec in glm53_checkpoint.SPARSE_MLA_WEIGHTS
            )

        counts = {
            owner: sum(actual_owner == owner for actual_owner, _key in owned)
            for owner in ("endpoint", "dense", "shell", "kda", "mla", "ffn")
        }
        keys = [key for _owner, key in owned]
        self.assertEqual(
            counts,
            {
                "endpoint": 3,
                "dense": 87,
                "shell": 336,
                "kda": 465,
                "mla": 198,
                "ffn": 72_912,
            },
        )
        self.assertEqual(len(keys), 74_001)
        self.assertEqual(len(set(keys)), 74_001)
        self.assertEqual(
            hashlib.sha256("\n".join(sorted(keys)).encode()).hexdigest(),
            "de715c2d450d237391f41fd7a8247ae89936c068ee41f6135b833e30e4e54e32",
        )
        self.assertFalse(
            any(
                key.startswith(("model.visual.", "model.language_model.layers.45."))
                for key in keys
            )
        )
        self.assertEqual(
            set(glm53_flash.SPARSE_KDA_LAYER_IDS)
            | set(glm53_flash.MAIN_SPARSE_MLA_LAYER_IDS),
            set(glm53_flash.SPARSE_FFN_LAYER_IDS),
        )
        self.assertFalse(
            set(glm53_flash.SPARSE_KDA_LAYER_IDS)
            & set(glm53_flash.MAIN_SPARSE_MLA_LAYER_IDS)
        )

        nextn_prefix = glm53_checkpoint.DENSE_LAYER_PREFIX.format(layer_id=45)
        nextn = {
            f"{nextn_prefix}{name}"
            for name in (
                "eh_proj.weight",
                "enorm.weight",
                "hnorm.weight",
                "input_layernorm.weight",
                "post_attention_layernorm.weight",
                "shared_head.norm.weight",
            )
        }
        nextn.update(
            f"{nextn_prefix}self_attn.{spec.name}"
            for spec in glm53_checkpoint.SPARSE_MLA_WEIGHTS
        )
        nextn.update(
            f"{nextn_prefix}mlp.{spec.name}"
            for spec in (
                glm53_checkpoint.SPARSE_FFN_ROUTER_WEIGHTS
                + glm53_checkpoint.SPARSE_FFN_SHARED_EXPERT_WEIGHTS
            )
        )
        nextn.update(
            f"{nextn_prefix}mlp.experts.{expert_id}.{spec.name}"
            for expert_id in range(glm53_flash.NUM_ROUTED_EXPERTS)
            for spec in glm53_checkpoint.SPARSE_FFN_ROUTED_EXPERT_TEMPLATE
        )
        self.assertEqual(len(nextn), 1_760)
        self.assertEqual(
            hashlib.sha256("\n".join(sorted(nextn)).encode()).hexdigest(),
            "b12f2fba05e311f0544d874bd2bb827df2bb0e1874b16d72da2452909fe0b348",
        )
        self.assertTrue(set(keys).isdisjoint(nextn))
        self.assertEqual(len(keys) + 347 + len(nextn), 76_108)
        self.assertEqual(
            glm53_checkpoint._glm53_target_checkpoint_keys(), frozenset(keys)
        )
        self.assertEqual(glm53_checkpoint._glm53_nextn_checkpoint_keys(), frozenset(nextn))

        visual = {
            f"model.visual.blocks.{block_id}.{suffix}"
            for block_id in range(24)
            for suffix in (
                "attn.k_norm.weight",
                "attn.proj.bias",
                "attn.proj.weight",
                "attn.q_norm.weight",
                "attn.qkv.bias",
                "attn.qkv.weight",
                "mlp.down_proj.bias",
                "mlp.down_proj.weight",
                "mlp.gate_proj.bias",
                "mlp.gate_proj.weight",
                "mlp.up_proj.bias",
                "mlp.up_proj.weight",
                "norm1.weight",
                "norm2.weight",
            )
        }
        visual.update(
            f"model.visual.{suffix}"
            for suffix in (
                "downsample.bias",
                "downsample.weight",
                "merger.down_proj.weight",
                "merger.gate_proj.weight",
                "merger.post_projection_norm.bias",
                "merger.post_projection_norm.weight",
                "merger.proj.weight",
                "merger.up_proj.weight",
                "patch_embed.proj.bias",
                "patch_embed.proj.weight",
                "post_layernorm.weight",
            )
        )
        self.assertEqual(len(visual), 347)
        self.assertEqual(
            hashlib.sha256("\n".join(sorted(visual)).encode()).hexdigest(),
            "979f4ffc275a60843257793bb9c3a331f362362ab3149a4abf49a131261677be",
        )
        weight_map = dict.fromkeys((*keys, *nextn, *visual), "pinned-shard")
        self.assertEqual(len(weight_map), 76_108)
        self.assertEqual(
            hashlib.sha256("\n".join(sorted(weight_map)).encode()).hexdigest(),
            "7baf28d0d672c695e66ac4c68f3a8bcf054e52b2f364a566e6f30fcfaa4cef2d",
        )
        glm53_checkpoint.validate_glm53_checkpoint_inventory(weight_map)

        changed = dict(weight_map)
        changed.pop(keys[0])
        changed.pop(next(iter(nextn)))
        changed.pop(next(iter(visual)))
        changed["model.unclassified.weight"] = "pinned-shard"
        with self.assertRaises(ValueError) as error:
            glm53_checkpoint.validate_glm53_checkpoint_inventory(changed)

        message = str(error.exception)
        self.assertIn("missing 1 target keys", message)
        self.assertIn("missing 1 layer-45 NextN keys", message)
        self.assertIn("found 346 visual keys, expected 347", message)
        self.assertIn("unexpected keys", message)


class SparseMLALayerCompositionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.production = load_ops_module()

    def setUp(self) -> None:
        self.engine = self.production.GLM53Ops()
        self.streams = self._bf16((1, 4, 4096))

        def tensor(shape, dtype):
            return SimpleNamespace(
                shape=shape,
                dtype=dtype,
                device=self.streams.device,
                is_cuda=True,
                is_contiguous=lambda: True,
            )

        attention_mhc = SimpleNamespace(
            base=tensor((24,), torch.float32),
            fn=tensor((24, 16384), torch.float32),
            scale=tensor((3,), torch.float32),
        )
        ffn_mhc = SimpleNamespace(
            base=tensor((24,), torch.float32),
            fn=tensor((24, 16384), torch.float32),
            scale=tensor((3,), torch.float32),
        )
        self.weights = glm53_flash.GLM53SparseMLALayerWeights(
            attention_mhc=attention_mhc,
            input_norm=tensor((4096,), torch.bfloat16),
            mla_projection="mla_projection",
            mla_decode="mla_decode",
            mla_output="mla_output",
            ffn_mhc=ffn_mhc,
            post_attention_norm=tensor((4096,), torch.bfloat16),
            ffn="ffn",
        )
        self.layer_workspace = SimpleNamespace(
            collapsed=self._bf16((1, 4096)),
            normalized=self._bf16((1, 4096)),
            streams_mid=self._bf16((1, 4, 4096)),
            streams_out=self._bf16((1, 4, 4096)),
        )
        self.projection_workspace = SimpleNamespace()
        self.decode_workspace = SimpleNamespace(output=self._bf16((1, 1, 16, 512)))
        self.output_workspace = SimpleNamespace(output=self._bf16((1, 4096)))
        self.ffn_workspace = SimpleNamespace(
            gathered=self._bf16((0, 4096)),
            scattered=self._bf16((0, 4096)),
            output=self._bf16((1, 4096)),
        )
        self.batch = object()
        self.history = object()
        self.world = object()
        self.all_reduce = object()

    @staticmethod
    def _bf16(shape: tuple[int, ...]) -> torch.Tensor:
        return torch.empty(shape, dtype=torch.bfloat16)

    def call(self, streams: torch.Tensor, group: object) -> torch.Tensor:
        return self.engine.decode_sparse_mla_layer(
            streams,
            self.weights,
            self.batch,
            self.history,
            self.projection_workspace,
            self.decode_workspace,
            self.output_workspace,
            self.ffn_workspace,
            self.layer_workspace,
            group,
            self.all_reduce,
        )

    def test_exact_order_group_rank_and_return_owner(self) -> None:
        events: list[str] = []
        latent = object()
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

        def projection(_hidden, weights, workspace):
            self.assertEqual(weights, "mla_projection")
            self.assertIs(workspace, self.projection_workspace)
            events.append("projection")
            return workspace

        def attention(projection_result, weights, batch, history, workspace):
            self.assertIs(projection_result, self.projection_workspace)
            self.assertEqual(weights, "mla_decode")
            self.assertIs(batch, self.batch)
            self.assertIs(history, self.history)
            self.assertIs(workspace, self.decode_workspace)
            events.append("history_attention")
            return latent

        def output(latent_result, weights, workspace, group, all_reduce):
            self.assertIs(latent_result, latent)
            self.assertEqual(weights, "mla_output")
            self.assertIs(workspace, self.output_workspace)
            self.assertIs(group, self.world)
            self.assertIs(all_reduce, self.all_reduce)
            events.append("attention_sum")
            return workspace.output

        def ffn(_hidden, weights, workspace, rank, all_reduce):
            self.assertEqual(weights, "ffn")
            self.assertIs(workspace, self.ffn_workspace)
            self.assertEqual(rank, 2)
            self.assertIs(all_reduce, self.all_reduce)
            events.append("ffn_sum")
            return workspace.output

        with (
            patch.object(self.production, "dist", distributed),
            patch.object(self.production, "_validate_sparse_mla_decode"),
            patch.object(self.production, "_validate_sparse_mla_output"),
            patch.object(self.production, "_validate_sparse_ffn"),
            patch.object(self.production, "_mhc_pre", side_effect=mhc_pre),
            patch.object(self.production, "_rmsnorm", side_effect=rmsnorm),
            patch.object(self.production, "_mhc_post", side_effect=mhc_post),
            patch.object(
                self.engine,
                "decode_sparse_mla_projection",
                side_effect=projection,
            ),
            patch.object(self.engine, "decode_sparse_mla", side_effect=attention),
            patch.object(self.engine, "decode_sparse_mla_output", side_effect=output),
            patch.object(self.engine, "decode_sparse_ffn", side_effect=ffn),
        ):
            returned = self.call(self.layer_workspace.streams_out, self.world)

        self.assertEqual(
            events,
            [
                "world_size",
                "rank",
                "attention_mhc",
                "input_norm",
                "projection",
                "history_attention",
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

    def test_invalid_group_fails_before_history(self) -> None:
        core = Mock()
        distributed = SimpleNamespace(
            group=SimpleNamespace(WORLD=self.world),
            get_world_size=Mock(return_value=4),
            get_rank=Mock(return_value=0),
        )
        with (
            patch.object(self.production, "dist", distributed),
            patch.object(self.engine, "decode_sparse_mla", core),
            self.assertRaisesRegex(ValueError, "WORLD"),
        ):
            self.call(object(), object())
        core.assert_not_called()
        distributed.get_world_size.assert_not_called()
        distributed.get_rank.assert_not_called()

        distributed.get_world_size.return_value = 3
        with (
            patch.object(self.production, "dist", distributed),
            patch.object(self.engine, "decode_sparse_mla", core),
            self.assertRaisesRegex(ValueError, "TEP4"),
        ):
            self.call(self.streams, self.world)
        core.assert_not_called()

        distributed.get_rank.assert_not_called()

    def test_rejects_unsafe_alias_before_history(self) -> None:
        core = Mock()
        distributed = SimpleNamespace(
            group=SimpleNamespace(WORLD=self.world),
            get_world_size=Mock(return_value=4),
            get_rank=Mock(return_value=0),
        )
        with (
            patch.object(self.production, "dist", distributed),
            patch.object(self.production, "_validate_sparse_mla_decode"),
            patch.object(self.production, "_validate_sparse_mla_output"),
            patch.object(self.production, "_validate_sparse_ffn"),
            patch.object(self.engine, "decode_sparse_mla", core),
            self.assertRaisesRegex(ValueError, "streams and streams_mid overlap"),
        ):
            self.call(self.layer_workspace.streams_mid, self.world)
        core.assert_not_called()

        self.projection_workspace.low_rank = self.streams.view(-1)[:2048].view(1, 2048)
        with (
            patch.object(self.production, "dist", distributed),
            patch.object(self.production, "_validate_sparse_mla_decode"),
            patch.object(self.production, "_validate_sparse_mla_output"),
            patch.object(self.production, "_validate_sparse_ffn"),
            patch.object(self.engine, "decode_sparse_mla", core),
            self.assertRaisesRegex(ValueError, "streams and projection.low_rank"),
        ):
            self.call(self.streams, self.world)
        core.assert_not_called()

        self.projection_workspace.low_rank = self._bf16((1, 2048))
        self.ffn_workspace.scores = self.layer_workspace.streams_mid.view(-1)[
            :288
        ].view(1, 288)
        with (
            patch.object(self.production, "dist", distributed),
            patch.object(self.production, "_validate_sparse_mla_decode"),
            patch.object(self.production, "_validate_sparse_mla_output"),
            patch.object(self.production, "_validate_sparse_ffn"),
            patch.object(self.engine, "decode_sparse_mla", core),
            self.assertRaisesRegex(ValueError, "streams_mid and ffn.scores"),
        ):
            self.call(self.streams, self.world)
        core.assert_not_called()

        del self.ffn_workspace.scores
        storage = self._bf16((1 + 4 * 4096,))
        self.streams = storage[:-1].view(1, 4, 4096)
        self.layer_workspace.streams_out = storage[1:].view(1, 4, 4096)
        with (
            patch.object(self.production, "dist", distributed),
            patch.object(self.production, "_validate_sparse_mla_decode"),
            patch.object(self.production, "_validate_sparse_mla_output"),
            patch.object(self.production, "_validate_sparse_ffn"),
            patch.object(self.engine, "decode_sparse_mla", core),
            self.assertRaisesRegex(ValueError, "streams and streams_out overlap"),
        ):
            self.call(self.streams, self.world)
        core.assert_not_called()

    def test_downstream_validation_fails_before_history(self) -> None:
        core = Mock()
        distributed = SimpleNamespace(
            group=SimpleNamespace(WORLD=self.world),
            get_world_size=Mock(return_value=4),
            get_rank=Mock(return_value=0),
        )
        with (
            patch.object(self.production, "dist", distributed),
            patch.object(self.production, "_validate_sparse_mla_decode"),
            patch.object(
                self.production,
                "_validate_sparse_mla_output",
                side_effect=ValueError("bad output workspace"),
            ),
            patch.object(self.engine, "decode_sparse_mla", core),
            self.assertRaisesRegex(ValueError, "bad output workspace"),
        ):
            self.call(self.streams, self.world)
        core.assert_not_called()

        bad_fn = SimpleNamespace(**vars(self.weights.ffn_mhc.fn))
        bad_fn.shape = (1, 1)
        bad_mhc = SimpleNamespace(**vars(self.weights.ffn_mhc))
        bad_mhc.fn = bad_fn
        with (
            patch.object(self, "weights", replace(self.weights, ffn_mhc=bad_mhc)),
            patch.object(self.production, "dist", distributed),
            patch.object(self.engine, "decode_sparse_mla", core),
            self.assertRaisesRegex(ValueError, "wrong shape or dtype"),
        ):
            self.call(self.streams, self.world)
        core.assert_not_called()

        self.weights = replace(
            self.weights,
            ffn=SimpleNamespace(router=torch.empty(288, 4096, dtype=torch.bfloat16)),
        )
        with (
            patch.object(self.production, "dist", distributed),
            patch.object(self.production, "_validate_sparse_mla_decode"),
            patch.object(self.production, "_validate_sparse_mla_output"),
            patch.object(
                self.production,
                "_validate_sparse_ffn",
                side_effect=ValueError("bad FFN workspace"),
            ),
            patch.object(self.engine, "decode_sparse_mla", core),
            self.assertRaisesRegex(ValueError, "bad FFN workspace"),
        ):
            self.call(self.streams, self.world)
        core.assert_not_called()

    def test_sparse_ffn_preflight_covers_full_tensor_contract(self) -> None:
        def tensor(shape, dtype):
            return SimpleNamespace(
                shape=shape,
                dtype=dtype,
                device="cuda:0",
                is_cuda=True,
                is_contiguous=lambda: True,
            )

        weight_fp8 = {
            "routed_up_gate",
            "routed_down",
            "shared_gate_up",
            "shared_down",
        }
        weights = glm53_flash.SparseFFNWeights(
            **{
                field.name: tensor(
                    getattr(glm53_flash.SPARSE_FFN_TEP4_SHAPES, field.name),
                    torch.float8_e4m3fn
                    if field.name in weight_fp8
                    else torch.bfloat16
                    if field.name in {"router", "router_t"}
                    else torch.float32,
                )
                for field in fields(glm53_flash.SparseFFNWeights)
            }
        )
        shapes = glm53_flash.sparse_ffn_decode_workspace_shapes(1)
        workspace_fp8 = {"hidden_fp8", "shared_fp8"}
        workspace_bf16 = {
            "gathered",
            "scattered",
            "routed",
            "shared_gate_up",
            "output",
        }
        workspace = glm53_flash.SparseFFNDecodeWorkspace(
            **{
                field.name: tensor(
                    getattr(shapes, field.name),
                    torch.float8_e4m3fn
                    if field.name in workspace_fp8
                    else torch.bfloat16
                    if field.name in workspace_bf16
                    else torch.int64
                    if field.name == "topk_ids64"
                    else torch.int32
                    if field.name == "topk_ids"
                    else torch.float32,
                )
                for field in fields(glm53_flash.SparseFFNDecodeWorkspace)
            }
        )
        hidden = tensor((1, 4096), torch.bfloat16)

        self.assertEqual(
            self.production._validate_sparse_ffn(hidden, weights, workspace, 2), 1
        )
        with self.assertRaisesRegex(ValueError, "scores workspace"):
            self.production._validate_sparse_ffn(
                hidden,
                weights,
                replace(workspace, scores=tensor((1, 1), torch.float32)),
                2,
            )
        with self.assertRaisesRegex(ValueError, "router_t weight"):
            self.production._validate_sparse_ffn(
                hidden,
                replace(
                    weights,
                    router_t=tensor(
                        glm53_flash.SPARSE_FFN_TEP4_SHAPES.router_t,
                        torch.float32,
                    ),
                ),
                workspace,
                2,
            )
        with self.assertRaisesRegex(ValueError, "shared_down weight"):
            self.production._validate_sparse_ffn(
                hidden,
                replace(
                    weights,
                    shared_down=tensor((1, 1), torch.float8_e4m3fn),
                ),
                workspace,
                2,
            )

    def test_dep_sparse_ffn_preflight_allows_an_uneven_local_batch(self) -> None:
        hidden = SimpleNamespace(
            shape=(1, glm53_flash.HIDDEN_SIZE),
            dtype=torch.bfloat16,
            device="cuda:0",
            is_cuda=True,
            is_contiguous=lambda: True,
        )
        weights = SimpleNamespace(
            router=SimpleNamespace(dtype=torch.bfloat16),
            __dataclass_fields__={},
        )
        workspace = SimpleNamespace(
            gathered=SimpleNamespace(shape=(8, glm53_flash.HIDDEN_SIZE)),
            scattered=SimpleNamespace(shape=(2, glm53_flash.HIDDEN_SIZE)),
            __dataclass_fields__={},
        )

        self.assertEqual(
            self.production._validate_sparse_ffn(hidden, weights, workspace, 2),
            1,
        )

    def test_dep_sparse_ffn_local_output_uses_reduce_scatter_buffer(self) -> None:
        local = torch.empty((1, 4096), dtype=torch.bfloat16)
        dep = SimpleNamespace(
            gathered=torch.empty((4, 4096), dtype=torch.bfloat16),
            scattered=local,
            output=torch.empty((4, 4096), dtype=torch.bfloat16),
        )
        tep = SimpleNamespace(
            gathered=torch.empty((0, 4096), dtype=torch.bfloat16),
            scattered=torch.empty((0, 4096), dtype=torch.bfloat16),
            output=local,
        )

        self.assertEqual(
            self.production._sparse_ffn_local_output(dep, 1).data_ptr(),
            local.data_ptr(),
        )
        self.assertEqual(
            self.production._sparse_ffn_local_output(tep, 1).data_ptr(),
            local.data_ptr(),
        )

    def test_source_pins_component_sequence_and_unchanged_ffn_abi(self) -> None:
        source = (ROOT / "src/infer/models/glm53_flash/ops/core.py").read_text()
        tree = ast.parse(source)
        functions = {
            node.name: node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
        }
        method = functions["decode_sparse_mla_layer"]
        calls = [
            call_name(statement.value)
            for statement in method.body
            if isinstance(getattr(statement, "value", None), ast.Call)
        ]
        self.assertEqual(
            calls,
            [
                "_sparse_tep_rank",
                "_validate_sparse_mla_layer",
                "_mhc_pre",
                "_rmsnorm",
                "decode_sparse_mla_projection",
                "decode_sparse_mla",
                "decode_sparse_mla_output",
                "_mhc_post",
                "_mhc_pre",
                "_rmsnorm",
                "decode_sparse_ffn",
                "_mhc_post",
            ],
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


class SparseMLAPrefillLayerCompositionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.production = load_ops_module()

    def test_live_lengths_and_resumed_history_order(self) -> None:
        world = object()
        distributed = SimpleNamespace(
            group=SimpleNamespace(WORLD=world),
            get_world_size=lambda group: 4,
            get_rank=lambda group: 2,
        )
        engine = self.production.GLM53Ops()
        weights = glm53_flash.GLM53SparseMLALayerWeights(
            attention_mhc=object(),
            input_norm=object(),
            mla_projection="projection_weights",
            mla_decode="decode_weights",
            mla_output="output_weights",
            ffn_mhc=object(),
            post_attention_norm=object(),
            ffn="ffn_weights",
        )

        for total_tokens in (1, 7, 63, 64):
            with self.subTest(total_tokens=total_tokens):
                events: list[str] = []
                staged_query = SimpleNamespace(
                    total_tokens=total_tokens,
                    start_token=17,
                    state_slot=5,
                    has_initial=True,
                )
                history = object()
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
                projection = object()
                decode = SimpleNamespace(
                    output=torch.full(
                        (total_tokens, 1, 16, 512), 4, dtype=torch.bfloat16
                    )
                )
                output = SimpleNamespace(
                    output=torch.full((total_tokens, 4096), 6, dtype=torch.bfloat16)
                )
                ffn = SimpleNamespace(
                    output=torch.full((total_tokens, 4096), 5, dtype=torch.bfloat16)
                )

                def mhc_pre(_streams, mhc, _workspace, _rows, events=events):
                    events.append(
                        "attention_mhc" if mhc is weights.attention_mhc else "ffn_mhc"
                    )

                def rmsnorm(_input, norm, _output, events=events):
                    events.append(
                        "input_norm" if norm is weights.input_norm else "post_norm"
                    )

                def project(
                    hidden,
                    projection_weights,
                    workspace,
                    projection=projection,
                    events=events,
                    total_tokens=total_tokens,
                ):
                    self.assertEqual(hidden.shape, (total_tokens, 4096))
                    self.assertEqual(projection_weights, "projection_weights")
                    self.assertIs(workspace, projection)
                    events.append("projection_b64")
                    return workspace

                def append(
                    projection_result,
                    decode_weights,
                    receipt,
                    state,
                    projection=projection,
                    staged_query=staged_query,
                    history=history,
                    events=events,
                ):
                    self.assertIs(projection_result, projection)
                    self.assertEqual(decode_weights, "decode_weights")
                    self.assertIs(receipt, staged_query)
                    self.assertIs(state, history)
                    events.append(f"append_{receipt.total_tokens}")
                    return state

                def query(
                    projection_result,
                    decode_weights,
                    receipt,
                    state,
                    workspace,
                    projection=projection,
                    staged_query=staged_query,
                    history=history,
                    decode=decode,
                    events=events,
                ):
                    self.assertIs(projection_result, projection)
                    self.assertEqual(decode_weights, "decode_weights")
                    self.assertIs(receipt, staged_query)
                    self.assertIs(state, history)
                    self.assertIs(workspace, decode)
                    events.append("causal_query_b64")
                    return workspace.output[:, 0]

                nccl = object()

                def attention_output(
                    latent,
                    output_weights,
                    workspace,
                    group,
                    all_reduce,
                    output=output,
                    events=events,
                    expected_all_reduce=nccl,
                    total_tokens=total_tokens,
                ):
                    self.assertEqual(latent.shape, (total_tokens, 16, 512))
                    self.assertEqual(output_weights, "output_weights")
                    self.assertIs(workspace, output)
                    self.assertIs(group, world)
                    self.assertIs(all_reduce, expected_all_reduce)
                    events.append("attention_sum")
                    return workspace.output

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
                    self.assertEqual(ffn_weights, "ffn_weights")
                    self.assertIs(workspace, ffn)
                    self.assertEqual(rank, 2)
                    self.assertIs(all_reduce, expected_all_reduce)
                    events.append("ffn_sum")
                    return workspace.output

                with (
                    patch.object(self.production, "dist", distributed),
                    patch.object(
                        self.production,
                        "_validate_sparse_mla_prefill_layer",
                        return_value=total_tokens,
                    ),
                    patch.object(self.production, "_mhc_pre", side_effect=mhc_pre),
                    patch.object(self.production, "_rmsnorm", side_effect=rmsnorm),
                    patch.object(self.production, "_mhc_post", side_effect=mhc_post),
                    patch.object(
                        engine, "decode_sparse_mla_projection", side_effect=project
                    ),
                    patch.object(
                        engine, "prefill_sparse_mla_history", side_effect=append
                    ),
                    patch.object(engine, "prefill_sparse_mla_query", side_effect=query),
                    patch.object(
                        engine, "decode_sparse_mla_output", side_effect=attention_output
                    ),
                    patch.object(engine, "decode_sparse_ffn", side_effect=sparse_ffn),
                ):
                    returned = engine.prefill_sparse_mla_layer(
                        streams,
                        weights,
                        staged_query,
                        history,
                        projection,
                        decode,
                        output,
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
                        "projection_b64",
                        f"append_{total_tokens}",
                        "causal_query_b64",
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

    def test_world_and_preflight_fail_before_history_or_stream_writes(self) -> None:
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
            object(),
            object(),
        )
        with (
            patch.object(self.production, "dist", distributed),
            patch.object(engine, "prefill_sparse_mla_history", state_op),
            self.assertRaisesRegex(ValueError, "WORLD"),
        ):
            engine.prefill_sparse_mla_layer(*arguments, object(), object())
        state_op.assert_not_called()
        self.assertTrue(torch.all(streams == 9))

        with (
            patch.object(self.production, "dist", distributed),
            patch.object(
                self.production,
                "_validate_sparse_mla_prefill_layer",
                side_effect=ValueError("unsafe alias"),
            ),
            patch.object(engine, "prefill_sparse_mla_history", state_op),
            self.assertRaisesRegex(ValueError, "unsafe alias"),
        ):
            engine.prefill_sparse_mla_layer(*arguments, world, object())
        state_op.assert_not_called()
        self.assertTrue(torch.all(streams == 9))

    def test_source_pins_b64_architecture_collectives_and_no_allocations(self) -> None:
        tree = ast.parse((ROOT / "src/infer/models/glm53_flash/ops/core.py").read_text())
        functions = {
            node.name: node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
        }
        method = functions["prefill_sparse_mla_layer"]
        self.assertEqual(
            [argument.arg for argument in method.args.args],
            [
                "self",
                "streams",
                "weights",
                "staged_query",
                "history",
                "projection_workspace",
                "decode_workspace",
                "output_workspace",
                "ffn_workspace",
                "layer_workspace",
                "process_group",
                "all_reduce",
            ],
        )
        source = ast.unparse(method)
        ordered = (
            "_validate_sparse_mla_prefill_layer",
            "_mhc_pre",
            "_rmsnorm",
            "decode_sparse_mla_projection",
            "prefill_sparse_mla_history",
            "prefill_sparse_mla_query",
            "decode_sparse_mla_output",
            "_mhc_post",
            "_mhc_pre",
            "_rmsnorm",
            "decode_sparse_ffn",
            "_mhc_post",
        )
        cursor = -1
        for name in ordered:
            cursor = source.index(name, cursor + 1)
        self.assertIn("staged_query", source)
        self.assertIn("rows = slice(total_tokens)", source)
        preflight = ast.unparse(functions["_validate_sparse_mla_prefill_layer"])
        self.assertLess(
            preflight.index("_reject_sparse_layer_aliases"),
            preflight.index("_sparse_mla_runtime"),
        )
        self.assertFalse(
            {"empty", "empty_like", "zeros", "zeros_like", "new_empty"}
            & {
                call_name(node)
                for node in ast.walk(method)
                if isinstance(node, ast.Call)
            }
        )
        self.assertEqual(
            sum(
                isinstance(node.func, ast.Name) and node.func.id == "all_reduce"
                for name in ("decode_sparse_mla_output", "decode_sparse_ffn")
                for node in ast.walk(functions[name])
                if isinstance(node, ast.Call)
            ),
            2,
        )


def call_name(call: ast.Call) -> str:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    raise TypeError("unexpected call")


if __name__ == "__main__":
    unittest.main()

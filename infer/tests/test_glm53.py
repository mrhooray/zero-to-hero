import ast
import hashlib
import json
import os
import sys
import tempfile
import unittest
from math import prod
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from infer.models.glm53_flash import checkpoint as glm53_checkpoint
from infer.models.glm53_flash import model as glm53_flash
from infer.models.glm53_flash.checkpoint import (
    SAFETENSORS_VERSION,
    WeightSpec,
    _read_local_bytes,
    _safetensor_data_offsets,
)
from infer.models.glm53_flash.model import (
    FP8_BLOCK_SIZE,
    HIDDEN_SIZE,
    MHC_STREAMS,
    RMS_NORM_EPS,
    GLM53DenseWorkspace,
    KDADecodeWorkspace,
    KDAPrefillWorkspace,
    KDAState,
    dense_workspace_shapes,
    kda_decode_workspace_shapes,
    kda_prefill_kernel_workspace_bytes,
    kda_prefill_workspace_shapes,
    kda_state_shapes,
)


def pinned_metadata(
    layer_id: int = 0,
) -> dict[str, tuple[str, tuple[int, ...]]]:
    tensors = {
        "A_log": ("F32", (64,)),
        "b_proj.weight": ("BF16", (64, 4096)),
        "dt_bias": ("F32", (8192,)),
        "f_a_proj.weight": ("BF16", (128, 4096)),
        "f_b_proj.weight": ("BF16", (8192, 128)),
        "g_a_proj.weight": ("BF16", (128, 4096)),
        "g_b_proj.weight": ("BF16", (8192, 128)),
        "k_conv1d.weight": ("BF16", (8192, 1, 4)),
        "k_proj.weight": ("BF16", (8192, 4096)),
        "o_norm.weight": ("BF16", (128,)),
        "o_proj.weight": ("BF16", (4096, 8192)),
        "q_conv1d.weight": ("BF16", (8192, 1, 4)),
        "q_proj.weight": ("BF16", (8192, 4096)),
        "v_conv1d.weight": ("BF16", (8192, 1, 4)),
        "v_proj.weight": ("BF16", (8192, 4096)),
    }
    prefix = f"model.language_model.layers.{layer_id}.self_attn."
    return {f"{prefix}{name}": metadata for name, metadata in tensors.items()}


def pinned_sparse_kda_layout() -> dict[int, tuple[tuple[str, int, int, int], ...]]:
    return {
        4: (("model-00047-of-00062.safetensors", 5_364_341_072, 172_528, 15),),
        5: (("model-00056-of-00062.safetensors", 5_364_341_080, 172_536, 15),),
        6: (("model-00057-of-00062.safetensors", 5_364_341_048, 172_504, 15),),
        8: (("model-00060-of-00062.safetensors", 5_364_341_064, 172_520, 15),),
        9: (
            ("model-00061-of-00062.safetensors", 5_303_993_768, 173_856, 12),
            ("model-00062-of-00062.safetensors", 1_261_584_968, 39_488, 3),
        ),
        10: (("model-00004-of-00062.safetensors", 5_364_342_296, 173_752, 15),),
        12: (("model-00007-of-00062.safetensors", 5_364_342_528, 173_984, 15),),
        13: (("model-00008-of-00062.safetensors", 5_364_342_304, 173_760, 15),),
        14: (("model-00010-of-00062.safetensors", 5_361_982_080, 173_984, 15),),
        16: (("model-00012-of-00062.safetensors", 5_364_342_288, 173_744, 15),),
        17: (("model-00014-of-00062.safetensors", 5_364_342_504, 173_960, 15),),
        18: (("model-00015-of-00062.safetensors", 5_364_342_304, 173_760, 15),),
        20: (("model-00018-of-00062.safetensors", 5_364_342_304, 173_760, 15),),
        21: (("model-00019-of-00062.safetensors", 5_364_342_272, 173_728, 15),),
        22: (("model-00021-of-00062.safetensors", 5_364_342_376, 173_832, 15),),
        24: (("model-00024-of-00062.safetensors", 5_364_342_640, 174_096, 15),),
        25: (("model-00025-of-00062.safetensors", 5_364_342_304, 173_760, 15),),
        26: (("model-00026-of-00062.safetensors", 5_364_342_272, 173_728, 15),),
        28: (("model-00029-of-00062.safetensors", 5_364_342_304, 173_760, 15),),
        29: (("model-00031-of-00062.safetensors", 5_364_341_544, 173_000, 15),),
        30: (("model-00033-of-00062.safetensors", 5_364_342_272, 173_728, 15),),
        32: (("model-00036-of-00062.safetensors", 5_364_342_304, 173_760, 15),),
        33: (("model-00038-of-00062.safetensors", 5_364_342_624, 174_080, 15),),
        34: (("model-00039-of-00062.safetensors", 5_364_342_296, 173_752, 15),),
        36: (("model-00042-of-00062.safetensors", 5_364_342_368, 173_824, 15),),
        37: (("model-00043-of-00062.safetensors", 5_364_342_296, 173_752, 15),),
        38: (("model-00045-of-00062.safetensors", 5_364_342_600, 174_056, 15),),
        40: (("model-00049-of-00062.safetensors", 5_364_342_344, 173_800, 15),),
        41: (("model-00050-of-00062.safetensors", 5_364_342_312, 173_768, 15),),
        42: (("model-00052-of-00062.safetensors", 5_364_342_576, 174_032, 15),),
        44: (("model-00054-of-00062.safetensors", 5_364_342_272, 173_728, 15),),
    }


def pinned_sparse_mla_metadata() -> dict[str, tuple[str, tuple[int, ...]]]:
    prefix = "model.language_model.layers.3.self_attn."
    tensors = {
        "indexer.index_kpool_compress_ape": ("BF16", (4, 128)),
        "indexer.index_kpool_compress_gate": ("BF16", (128, 4096)),
        "indexer.k_norm.bias": ("BF16", (128,)),
        "indexer.k_norm.weight": ("BF16", (128,)),
        "indexer.weights_proj.weight": ("BF16", (32, 4096)),
        "indexer.wk.weight": ("BF16", (128, 4096)),
        "indexer.wq_b.weight": ("BF16", (4096, 1536)),
        "kv_a_layernorm.weight": ("BF16", (512,)),
        "kv_a_proj_with_mqa.weight": ("F8_E4M3", (512, 4096)),
        "kv_a_proj_with_mqa.weight_scale_inv": ("F32", (4, 32)),
        "kv_b_proj.weight": ("BF16", (32768, 512)),
        "o_proj.weight": ("F8_E4M3", (4096, 16384)),
        "o_proj.weight_scale_inv": ("F32", (32, 128)),
        "q_a_layernorm.weight": ("BF16", (1536,)),
        "q_a_proj.weight": ("F8_E4M3", (1536, 4096)),
        "q_a_proj.weight_scale_inv": ("F32", (12, 32)),
        "q_b_proj.weight": ("F8_E4M3", (16384, 1536)),
        "q_b_proj.weight_scale_inv": ("F32", (128, 12)),
    }
    return {f"{prefix}{name}": metadata for name, metadata in tensors.items()}


def pinned_sparse_ffn_metadata(
    layer_id: int = 3,
) -> dict[str, tuple[str, tuple[int, ...]]]:
    prefix = f"model.language_model.layers.{layer_id}.mlp."
    non_expert = {
        "gate.weight": ("BF16", (288, 4096)),
        "gate.e_score_correction_bias": ("F32", (288,)),
        "shared_experts.gate_proj.weight": ("F8_E4M3", (2048, 4096)),
        "shared_experts.gate_proj.weight_scale_inv": ("F32", (16, 32)),
        "shared_experts.up_proj.weight": ("F8_E4M3", (2048, 4096)),
        "shared_experts.up_proj.weight_scale_inv": ("F32", (16, 32)),
        "shared_experts.down_proj.weight": ("F8_E4M3", (4096, 2048)),
        "shared_experts.down_proj.weight_scale_inv": ("F32", (32, 16)),
    }
    expert = {
        "gate_proj.weight": ("F8_E4M3", (2048, 4096)),
        "gate_proj.weight_scale_inv": ("F32", (16, 32)),
        "up_proj.weight": ("F8_E4M3", (2048, 4096)),
        "up_proj.weight_scale_inv": ("F32", (16, 32)),
        "down_proj.weight": ("F8_E4M3", (4096, 2048)),
        "down_proj.weight_scale_inv": ("F32", (32, 16)),
    }
    metadata = {f"{prefix}{name}": value for name, value in non_expert.items()}
    metadata.update(
        {
            f"{prefix}experts.{expert_id}.{name}": value
            for expert_id in range(288)
            for name, value in expert.items()
        }
    )
    return metadata


def pinned_sparse_ffn_layer_weight_map(layer_id: int) -> dict[str, str]:
    layouts = {
        5: (
            "model-00056-of-00062.safetensors",
            (
                (0, 1, "model-00054-of-00062.safetensors"),
                (1, 4, "model-00055-of-00062.safetensors"),
                (4, 10, "model-00056-of-00062.safetensors"),
                (10, 32, "model-00055-of-00062.safetensors"),
                (32, 100, "model-00056-of-00062.safetensors"),
                (100, 288, "model-00055-of-00062.safetensors"),
            ),
        ),
        44: (
            "model-00054-of-00062.safetensors",
            (
                (0, 2, "model-00053-of-00062.safetensors"),
                (2, 10, "model-00054-of-00062.safetensors"),
                (10, 18, "model-00053-of-00062.safetensors"),
                (18, 100, "model-00054-of-00062.safetensors"),
                (100, 178, "model-00053-of-00062.safetensors"),
                (178, 288, "model-00054-of-00062.safetensors"),
            ),
        ),
    }
    non_expert_shard, expert_ranges = layouts[layer_id]

    def shard(expert_id: int | None) -> str:
        if expert_id is None:
            return non_expert_shard
        return next(
            filename
            for start, stop, filename in expert_ranges
            if start <= expert_id < stop
        )

    return {
        key: shard(expert_id)
        for key, _spec, expert_id in glm53_checkpoint._sparse_ffn_checkpoint_specs(
            layer_id
        )
    }


class DenseKDAWeightsTest(unittest.TestCase):
    def test_tp4_shapes_match_head_sharding(self) -> None:
        expected = {
            "q_proj.weight": (2048, 4096),
            "k_proj.weight": (2048, 4096),
            "v_proj.weight": (2048, 4096),
            "q_conv1d.weight": (2048, 1, 4),
            "k_conv1d.weight": (2048, 1, 4),
            "v_conv1d.weight": (2048, 1, 4),
            "f_a_proj.weight": (128, 4096),
            "f_b_proj.weight": (2048, 128),
            "A_log": (16,),
            "dt_bias": (2048,),
            "b_proj.weight": (16, 4096),
            "g_a_proj.weight": (128, 4096),
            "g_b_proj.weight": (2048, 128),
            "o_norm.weight": (128,),
            "o_proj.weight": (4096, 2048),
        }

        self.assertEqual(
            {
                spec.name: spec.local_shape()
                for spec in glm53_checkpoint.DENSE_KDA_WEIGHTS
            },
            expected,
        )

    def test_dep4_conserves_per_rank_attention_capacity(self) -> None:
        self.assertEqual(
            glm53_flash.glm53_decode_batch_sizes(1), (1, 2, 4, 8, 16, 32)
        )
        self.assertEqual(
            glm53_flash.glm53_decode_batch_sizes(1, speculative=False),
            (1, 2, 4, 8, 16, 32),
        )
        self.assertEqual(glm53_flash.glm53_local_heads(1), 64)
        self.assertEqual(glm53_flash.glm53_local_heads(glm53_flash.TP_SIZE), 16)

        for dep, tep in (
            (
                glm53_flash.kda_state_shapes(16, 1),
                glm53_flash.kda_state_shapes(64, glm53_flash.TP_SIZE),
            ),
            (
                glm53_flash.kda_state_shapes(32, 1),
                glm53_flash.kda_state_shapes(128, glm53_flash.TP_SIZE),
            ),
        ):
            self.assertEqual(prod(dep.recurrent), prod(tep.recurrent))
            self.assertEqual(prod(dep.conv), prod(tep.conv))

        dep_endpoint = glm53_flash.glm53_endpoint_workspace_shapes(16, 1)
        tep_endpoint = glm53_flash.glm53_endpoint_workspace_shapes(
            64, glm53_flash.TP_SIZE
        )
        self.assertEqual(
            prod(dep_endpoint.local_logits), prod(tep_endpoint.local_logits)
        )
        dep_decode = glm53_flash.kda_decode_workspace_shapes(16, 1)
        tep_decode = glm53_flash.kda_decode_workspace_shapes(64, glm53_flash.TP_SIZE)
        self.assertEqual(prod(dep_decode.local_output), prod(tep_decode.local_output))
        dep_dense = glm53_flash.dense_workspace_shapes(16, 1)
        tep_dense = glm53_flash.dense_workspace_shapes(64, glm53_flash.TP_SIZE)
        self.assertEqual(prod(dep_dense.gate_up), prod(tep_dense.gate_up))


class DenseLayerWeightsTest(unittest.TestCase):
    def test_tp4_shapes_match_dense_ffn_sharding(self) -> None:
        specs = {spec.name: spec for spec in glm53_checkpoint.DENSE_LAYER_WEIGHTS}
        gate = specs["mlp.gate_proj.weight"].local_shape()
        up = specs["mlp.up_proj.weight"].local_shape()
        gate_scale = specs["mlp.gate_proj.weight_scale_inv"].local_shape()
        up_scale = specs["mlp.up_proj.weight_scale_inv"].local_shape()

        self.assertEqual(gate, (3072, 4096))
        self.assertEqual(up, (3072, 4096))
        self.assertEqual((gate[0] + up[0], gate[1]), (6144, 4096))
        self.assertEqual(gate_scale, (24, 32))
        self.assertEqual(up_scale, (24, 32))
        self.assertEqual((gate_scale[0] + up_scale[0], gate_scale[1]), (48, 32))
        self.assertEqual(specs["mlp.down_proj.weight"].local_shape(), (4096, 3072))
        self.assertEqual(
            specs["mlp.down_proj.weight_scale_inv"].local_shape(), (32, 24)
        )


class SparseFFNContractTest(unittest.TestCase):
    def test_full_index_and_sparse_shard_pins_match_checkpoint(self) -> None:
        self.assertEqual(
            glm53_checkpoint.SAFETENSORS_INDEX_PIN,
            (
                8_406_613,
                "3c3f40366a53c3fd7974b4eab7881a365a98c2a4329150befebab99fe7c18b05",
            ),
        )
        self.assertEqual(
            set(glm53_checkpoint.SPARSE_FFN_SHARD_PINS),
            {f"model-{shard_id:05d}-of-00062.safetensors" for shard_id in range(3, 62)},
        )
        self.assertEqual(
            {
                filename: glm53_checkpoint.SPARSE_FFN_SHARD_PINS[filename]
                for filename in (
                    "model-00003-of-00062.safetensors",
                    "model-00031-of-00062.safetensors",
                    "model-00032-of-00062.safetensors",
                    "model-00053-of-00062.safetensors",
                    "model-00054-of-00062.safetensors",
                    "model-00055-of-00062.safetensors",
                    "model-00056-of-00062.safetensors",
                    "model-00061-of-00062.safetensors",
                )
            },
            {
                "model-00003-of-00062.safetensors": (5_363_467_432, 175_048),
                "model-00031-of-00062.safetensors": (5_364_341_544, 173_000),
                "model-00032-of-00062.safetensors": (5_363_915_232, 177_792),
                "model-00053-of-00062.safetensors": (5_363_915_936, 178_496),
                "model-00054-of-00062.safetensors": (5_364_342_272, 173_728),
                "model-00055-of-00062.safetensors": (5_361_808_304, 179_112),
                "model-00056-of-00062.safetensors": (5_364_341_080, 172_536),
                "model-00061-of-00062.safetensors": (5_303_993_768, 173_856),
            },
        )

    def test_manifest_matches_pinned_layer3_shard_headers(self) -> None:
        metadata = pinned_sparse_ffn_metadata()

        glm53_checkpoint.validate_sparse_ffn_metadata(metadata)

        self.assertEqual(glm53_flash.SPARSE_FFN_LAYER_IDS, tuple(range(3, 45)))
        self.assertNotIn(45, glm53_flash.SPARSE_FFN_LAYER_IDS)
        self.assertEqual(glm53_flash.NUM_ROUTED_EXPERTS, 288)
        self.assertEqual(glm53_flash.TOP_K, 8)
        self.assertEqual(glm53_flash.MOE_INTERMEDIATE_SIZE, 2048)
        self.assertEqual(len(metadata), 1_736)
        generated = {
            glm53_checkpoint.sparse_ffn_checkpoint_key(3, spec): (
                spec.dtype,
                spec.shape,
            )
            for spec in (
                glm53_checkpoint.SPARSE_FFN_ROUTER_WEIGHTS
                + glm53_checkpoint.SPARSE_FFN_SHARED_EXPERT_WEIGHTS
            )
        }
        generated.update(
            {
                glm53_checkpoint.sparse_ffn_checkpoint_key(3, spec, expert_id): (
                    spec.dtype,
                    spec.shape,
                )
                for expert_id in range(288)
                for spec in glm53_checkpoint.SPARSE_FFN_ROUTED_EXPERT_TEMPLATE
            }
        )
        self.assertEqual(generated, metadata)

        for layer_id in glm53_flash.SPARSE_FFN_LAYER_IDS:
            specs = tuple(glm53_checkpoint._sparse_ffn_checkpoint_specs(layer_id))
            prefix = glm53_checkpoint.SPARSE_FFN_PREFIX.format(layer_id=layer_id)
            self.assertEqual(len(specs), 1_736)
            self.assertTrue(all(key.startswith(prefix) for key, _spec, _id in specs))

    def test_boundary_layer_index_layouts_match_pinned_routing(self) -> None:
        expected_counts = {
            5: {
                "model-00054-of-00062.safetensors": 6,
                "model-00055-of-00062.safetensors": 1_278,
                "model-00056-of-00062.safetensors": 452,
            },
            44: {
                "model-00053-of-00062.safetensors": 528,
                "model-00054-of-00062.safetensors": 1_208,
            },
        }

        for layer_id, counts in expected_counts.items():
            weight_map = pinned_sparse_ffn_layer_weight_map(layer_id)
            weight_map["model.visual.vision_model.embeddings.weight"] = (
                "model-00001-of-00062.safetensors"
            )
            layout = glm53_checkpoint._sparse_ffn_layer_layout(weight_map, layer_id)

            self.assertEqual(len(layout), 1_736)
            self.assertEqual(
                {
                    filename: tuple(layout.values()).count(filename)
                    for filename in set(layout.values())
                },
                counts,
            )

        with self.assertRaisesRegex(ValueError, "not a sparse FFN layer"):
            glm53_checkpoint._sparse_ffn_layer_layout({}, 46)

    def test_sparse_layout_rejects_missing_extra_and_unpinned_mapping(self) -> None:
        layer_id = 5
        prefix = glm53_checkpoint.SPARSE_FFN_PREFIX.format(layer_id=layer_id)
        weight_map = pinned_sparse_ffn_layer_weight_map(layer_id)
        missing = next(iter(weight_map))
        weight_map.pop(missing)
        wrong = next(iter(weight_map))
        weight_map[wrong] = "model-00062-of-00062.safetensors"
        extra = f"{prefix}unexpected.weight"
        weight_map[extra] = "model-00056-of-00062.safetensors"

        with self.assertRaises(ValueError) as error:
            glm53_checkpoint._sparse_ffn_layer_layout(weight_map, layer_id)

        message = str(error.exception)
        self.assertIn(f"missing {missing}", message)
        self.assertIn(f"unexpected {extra}", message)
        self.assertIn(f"{wrong} maps to unpinned shard", message)

    def test_checkpoint_keys_reject_invalid_layers_and_experts(self) -> None:
        router = glm53_checkpoint.SPARSE_FFN_ROUTER_WEIGHTS[0]
        expert = glm53_checkpoint.SPARSE_FFN_ROUTED_EXPERT_TEMPLATE[0]

        self.assertEqual(
            glm53_checkpoint.sparse_ffn_checkpoint_key(44, expert, 287),
            "model.language_model.layers.44.mlp.experts.287.gate_proj.weight",
        )
        with self.assertRaisesRegex(ValueError, "not a sparse FFN layer"):
            glm53_checkpoint.sparse_ffn_checkpoint_key(46, router)
        with self.assertRaisesRegex(ValueError, "require an expert_id"):
            glm53_checkpoint.sparse_ffn_checkpoint_key(3, expert)
        with self.assertRaisesRegex(ValueError, "expert_id"):
            glm53_checkpoint.sparse_ffn_checkpoint_key(3, expert, 288)
        with self.assertRaisesRegex(ValueError, "not a non-expert sparse FFN weight"):
            glm53_checkpoint.sparse_ffn_checkpoint_key(3, router, 0)

    def test_rejects_changed_layer3_scope_or_layout(self) -> None:
        metadata = pinned_sparse_ffn_metadata()
        metadata.pop("model.language_model.layers.3.mlp.experts.287.down_proj.weight")
        metadata["model.language_model.layers.3.mlp.gate.weight"] = (
            "F32",
            (1,),
        )
        metadata["model.language_model.layers.3.mlp.new.weight"] = ("BF16", (1,))

        with self.assertRaisesRegex(
            ValueError, "missing.*experts.287.down_proj.weight"
        ) as error:
            glm53_checkpoint.validate_sparse_ffn_metadata(metadata)

        message = str(error.exception)
        self.assertIn("unexpected", message)
        self.assertIn("has dtype F32, expected BF16", message)
        self.assertIn("has shape (1,), expected (288, 4096)", message)

    def test_tep4_owns_contiguous_numeric_experts(self) -> None:
        self.assertEqual(glm53_flash.LOCAL_ROUTED_EXPERTS, 72)
        for rank in range(4):
            self.assertEqual(
                glm53_checkpoint.local_sparse_ffn_expert_ids(rank),
                tuple(range(rank * 72, (rank + 1) * 72)),
            )

        with self.assertRaisesRegex(ValueError, "tep_rank must be"):
            glm53_checkpoint.local_sparse_ffn_expert_ids(4)
        with self.assertRaisesRegex(ValueError, "cannot shard 5 ways"):
            glm53_checkpoint.local_sparse_ffn_expert_ids(0, 5)

    def test_packed_runtime_shapes_and_bytes(self) -> None:
        shapes = glm53_flash.SPARSE_FFN_TEP4_SHAPES

        self.assertEqual(
            glm53_flash.SPARSE_FFN_SHARED_FC1_ORDER,
            ("gate_proj", "up_proj"),
        )
        self.assertEqual(shapes.router, (288, 4096))
        self.assertEqual(shapes.router_t, (4096, 288))
        self.assertEqual(shapes.correction_bias, (288,))
        self.assertEqual(shapes.routed_up_gate, (72, 4096, 4096))
        self.assertEqual(shapes.routed_up_gate_scale_inv, (72, 32, 32))
        self.assertEqual(shapes.routed_down, (72, 4096, 2048))
        self.assertEqual(shapes.routed_down_scale_inv, (72, 32, 16))
        self.assertEqual(shapes.routed_clamp, (72,))
        self.assertEqual(shapes.shared_gate_up, (1024, 4096))
        self.assertEqual(shapes.shared_gate_up_scale_inv, (8, 32))
        self.assertEqual(shapes.shared_down, (4096, 512))
        self.assertEqual(shapes.shared_down_scale_inv, (32, 4))

        shared = {
            spec.name: spec
            for spec in glm53_checkpoint.SPARSE_FFN_SHARED_EXPERT_WEIGHTS
        }
        self.assertEqual(
            shared["shared_experts.gate_proj.weight"].local_shape(), (512, 4096)
        )
        self.assertEqual(
            shared["shared_experts.up_proj.weight_scale_inv"].local_shape(), (4, 32)
        )
        self.assertEqual(
            shared["shared_experts.down_proj.weight"].local_shape(), (4096, 512)
        )
        self.assertEqual(
            shared["shared_experts.down_proj.weight_scale_inv"].local_shape(), (32, 4)
        )

    def test_workspace_geometry_supports_live_prefill_tokens(self) -> None:
        width = glm53_flash.GLM53_TARGET_VERIFY_WIDTH
        self.assertEqual(glm53_flash.KDA_CHUNK_SIZE, 8192)
        self.assertEqual(
            glm53_flash.GLM53_TARGET_DECODE_BATCH_SIZES, (1, 2, 4, 8, 16, 32, 64)
        )
        self.assertEqual(
            glm53_flash.KDA_CHUNK_SIZE
            + max(glm53_flash.GLM53_TARGET_DECODE_BATCH_SIZES) * width,
            8192 + 64 * width,
        )
        self.assertEqual(
            glm53_flash.SPARSE_FFN_DECODE_BATCH_SIZES,
            (
                *glm53_flash.GLM53_TARGET_DECODE_BATCH_SIZES,
                glm53_flash.GLM53_TARGET_VERIFY_WIDTH,
                glm53_flash.SPARSE_MLA_HISTORY_BLOCK_TOKENS,
                glm53_flash.KDA_CHUNK_SIZE,
            ),
        )
        self.assertEqual(
            glm53_flash.sparse_ffn_decode_workspace_shapes(width),
            glm53_flash.SparseFFNDecodeWorkspace(
                gathered=(0, 4096),
                scattered=(0, 4096),
                scores=(width, 288),
                selection=(width, 288),
                topk_values=(width, 8),
                topk_ids64=(width, 8),
                topk_ids=(width, 8),
                hidden_fp8=(width, 4096),
                hidden_scale_mn=(32, width),
                routed=(width, 4096),
                shared_gate_up=(width, 1024),
                shared_fp8=(width, 512),
                shared_scale=(width, 4),
                output=(width, 4096),
            ),
        )

        dep = glm53_flash.sparse_ffn_decode_workspace_shapes(16, glm53_flash.TP_SIZE)
        tep = glm53_flash.sparse_ffn_decode_workspace_shapes(64)
        self.assertEqual(dep.gathered, tep.output)
        self.assertEqual(dep.scattered, (16, glm53_flash.HIDDEN_SIZE))
        for name in dep.__dataclass_fields__:
            if name not in {"gathered", "scattered"}:
                self.assertEqual(getattr(dep, name), getattr(tep, name))
        for batch_size in (
            *glm53_flash.GLM53_TARGET_DECODE_BATCH_SIZES,
            glm53_flash.SPARSE_MLA_HISTORY_BLOCK_TOKENS,
        ):
            self.assertEqual(
                glm53_flash.sparse_ffn_decode_workspace_shapes(batch_size).output,
                (batch_size, 4096),
            )
        self.assertEqual(
            glm53_flash.sparse_ffn_decode_workspace_shapes(1455).output, (1455, 4096)
        )
        for batch_size in (0, glm53_flash.KDA_CHUNK_SIZE + 1):
            with self.assertRaisesRegex(
                ValueError, rf"\[1, {glm53_flash.KDA_CHUNK_SIZE}\]"
            ):
                glm53_flash.sparse_ffn_decode_workspace_shapes(batch_size)

    def test_loader_pins_index_headers_and_direct_packing(self) -> None:
        source = (
            Path(__file__).resolve().parents[1] / "src/infer/models/glm53_flash/checkpoint.py"
        ).read_text()
        functions = {
            node.name: node
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.FunctionDef)
        }
        loader = functions["load_glm53_sparse_ffn_weights"]
        packer = functions["_pack_sparse_ffn_weights"]
        loader_calls = list(ast_calls(loader))
        packed = ast.unparse(packer)

        for call in (
            "_read_checkpoint_weight_map",
            "_sparse_ffn_layer_layout",
            "safe_open",
            "_safetensor_data_offsets",
            "validate_sparse_ffn_metadata",
            "_load_local_tensor",
            "_pack_sparse_ffn_weights",
        ):
            self.assertGreaterEqual(count_calls(loader_calls, call), 1)
        self.assertIn("tep_rank must be", ast.unparse(loader))
        self.assertIn("header_size != expected_header_size", ast.unparse(loader))
        self.assertIn("router_t=router.T", packed)
        self.assertNotIn(".to(torch.float32)", packed)
        self.assertIn("SPARSE_FFN_CLAMP", packed)

        targets = next(
            node.value
            for node in ast.walk(packer)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "expert_targets"
                for target in node.targets
            )
        )
        self.assertEqual(
            [ast.literal_eval(item.elts[1]) for item in targets.elts],
            [
                "up_proj.weight",
                "gate_proj.weight",
                "up_proj.weight_scale_inv",
                "gate_proj.weight_scale_inv",
                "down_proj.weight",
                "down_proj.weight_scale_inv",
            ],
        )

    def test_layer5_loader_routes_three_shards_from_pinned_index(self) -> None:
        layer_id = 5
        weight_map = pinned_sparse_ffn_layer_weight_map(layer_id)
        weight_map.update(
            {
                "model.language_model.layers.45.mlp.gate.weight": (
                    "model-00061-of-00062.safetensors"
                ),
                "model.visual.vision_model.embeddings.weight": (
                    "model-00001-of-00062.safetensors"
                ),
            }
        )
        specs = {
            key: spec
            for key, spec, _expert_id in glm53_checkpoint._sparse_ffn_checkpoint_specs(
                layer_id
            )
        }
        shard_sizes = {
            "model-00054-of-00062.safetensors": 9,
            "model-00055-of-00062.safetensors": 10,
            "model-00056-of-00062.safetensors": 11,
        }
        metadata = {
            filename: {
                key: (spec.dtype, spec.shape)
                for key, spec in specs.items()
                if weight_map[key] == filename
            }
            for filename in shard_sizes
        }
        opens = []
        loads = []

        class Checkpoint:
            def __init__(self, values):
                self.values = values

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def offset_keys(self):
                return self.values

            def get_slice(self, name):
                dtype, shape = self.values[name]
                return type(
                    "Slice",
                    (),
                    {
                        "get_dtype": lambda _self: dtype,
                        "get_shape": lambda _self: shape,
                    },
                )()

        def safe_open(path, **kwargs):
            opens.append((Path(path).name, kwargs))
            return Checkpoint(metadata[Path(path).name])

        def load_local(descriptor, _offset, spec, tep_rank, device, _torch):
            value = (os.fstat(descriptor).st_size, spec.name, tep_rank, device)
            loads.append(value)
            return value

        def pack(load, tep_rank, device, _torch):
            expert = glm53_checkpoint.SPARSE_FFN_ROUTED_EXPERT_TEMPLATE[0]
            return {
                "router": load(glm53_checkpoint.SPARSE_FFN_ROUTER_WEIGHTS[0]),
                "expert_0": load(expert, 0),
                "expert_1": load(expert, 1),
                "expert_4": load(expert, 4),
                "rank_device": (tep_rank, device),
            }

        index_payload = json.dumps({"weight_map": weight_map}).encode()
        safetensors = ModuleType("safetensors")
        safetensors.safe_open = safe_open
        torch = ModuleType("torch")
        pins = {filename: (size, 0) for filename, size in shard_sizes.items()}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / glm53_checkpoint.SAFETENSORS_INDEX).write_bytes(index_payload)
            for filename, size in shard_sizes.items():
                path = root / filename
                path.write_bytes(bytes(8))
                with path.open("ab") as shard:
                    shard.truncate(size)

            with (
                patch.dict(sys.modules, {"safetensors": safetensors, "torch": torch}),
                patch.object(glm53_checkpoint, "SPARSE_FFN_SHARD_PINS", pins),
                patch.object(
                    glm53_checkpoint,
                    "SAFETENSORS_INDEX_PIN",
                    (len(index_payload), hashlib.sha256(index_payload).hexdigest()),
                ),
                patch.object(glm53_checkpoint, "_require_safetensors_version"),
                patch.object(
                    glm53_checkpoint,
                    "_safetensor_data_offsets",
                    side_effect=lambda path: dict.fromkeys(metadata[path.name], 0),
                ),
                patch.object(
                    glm53_checkpoint, "_load_local_tensor", side_effect=load_local
                ),
                patch.object(
                    glm53_checkpoint,
                    "_pack_sparse_ffn_weights",
                    side_effect=pack,
                ),
            ):
                output = glm53_checkpoint.load_glm53_sparse_ffn_weights(
                    root, layer_id, 3, "cuda"
                )

                wrong_map = dict(weight_map)
                wrong_key = glm53_checkpoint.sparse_ffn_checkpoint_key(
                    layer_id, glm53_checkpoint.SPARSE_FFN_ROUTED_EXPERT_TEMPLATE[0], 0
                )
                wrong_map[wrong_key] = "model-00055-of-00062.safetensors"
                (root / glm53_checkpoint.SAFETENSORS_INDEX).write_bytes(
                    json.dumps({"weight_map": wrong_map}).encode()
                )
                with self.assertRaisesRegex(ValueError, "size/SHA-256"):
                    glm53_checkpoint._read_checkpoint_weight_map(root)

        self.assertEqual([name for name, _kwargs in opens], sorted(shard_sizes))
        self.assertTrue(
            all(
                kwargs == {"framework": "pt", "device": "cpu", "backend": "pread"}
                for _name, kwargs in opens
            )
        )
        self.assertEqual(
            output,
            {
                "router": (11, "gate.weight", 3, "cuda"),
                "expert_0": (9, "gate_proj.weight", 3, "cuda"),
                "expert_1": (10, "gate_proj.weight", 3, "cuda"),
                "expert_4": (11, "gate_proj.weight", 3, "cuda"),
                "rank_device": (3, "cuda"),
            },
        )
        self.assertEqual(len(loads), 4)
        for layer_id in (2, 46):
            with self.assertRaisesRegex(ValueError, "not a sparse FFN layer"):
                glm53_checkpoint.load_glm53_sparse_ffn_weights(
                    "unused", layer_id, 0, "cuda"
                )

    def test_sparse_ffn_pins_tep4_overlap_sequence(self) -> None:
        source = (
            Path(__file__).resolve().parents[1] / "src/infer/models/glm53_flash/ops/core.py"
        ).read_text()
        method = next(
            node
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.FunctionDef) and node.name == "decode_sparse_ffn"
        )
        calls = list(ast_calls(method))
        moe = next(
            call
            for call in calls
            if call_name(call) == "trtllm_fp8_block_scale_routed_moe"
        )

        for name, count in (
            ("mm", 1),
            ("_route_sparse_ffn_pre_topk", 1),
            ("topk", 1),
            ("_route_sparse_ffn_post_topk", 1),
            ("_quantize", 1),
            ("_fp8_gemm", 2),
            ("_swiglu_kernel", 1),
            ("trtllm_fp8_block_scale_routed_moe", 1),
            ("Stream", 1),
            ("Event", 2),
            ("record", 2),
            ("wait_event", 2),
            ("all_reduce", 1),
        ):
            self.assertEqual(count_calls(calls, name), count)
        self.assertIn("self._sparse_ffn_overlap: tuple[object, object, object]", source)
        self.assertNotIn("all_to_all", ast.unparse(method))
        self.assertNotIn("hidden_f32", ast.unparse(method))
        self.assertIn("out_dtype=torch.float32", ast.unparse(method))
        self.assertIn("hidden_scale = w.hidden_scale_mn.T", ast.unparse(method))
        self.assertIn(
            "shared_scale = _mn_scale_view(w.shared_scale)", ast.unparse(method)
        )
        self.assertNotIn("weight_sum", ast.unparse(method))
        self.assertNotIn("topk_weights", ast.unparse(method))
        topk = next(call for call in calls if call_name(call) == "topk")
        self.assertEqual(keyword_constant(topk, "sorted"), False)

        body = ast.unparse(method)
        ordering = (
            "fork_event.record(current_stream)",
            "trtllm_fp8_block_scale_routed_moe(",
            "shared_stream.wait_event(fork_event)",
            "join_event.record(shared_stream)",
            "current_stream.wait_event(join_event)",
            "w.output.add_(w.routed)",
        )
        positions = [body.index(item) for item in ordering]
        self.assertEqual(positions, sorted(positions))

        expressions = {
            keyword.arg: ast.unparse(keyword.value) for keyword in moe.keywords
        }
        self.assertEqual(expressions["topk_ids"], "(w.topk_ids, w.topk_values)")
        self.assertEqual(expressions["routing_bias"], "None")
        self.assertEqual(
            expressions["local_expert_offset"], "tep_rank * LOCAL_ROUTED_EXPERTS"
        )
        self.assertEqual(expressions["local_num_experts"], "LOCAL_ROUTED_EXPERTS")
        self.assertEqual(expressions["gemm1_clamp_limit"], "weights.routed_clamp")
        self.assertEqual(expressions["output"], "w.routed")
        self.assertEqual(expressions["enable_pdl"], "True")
        self.assertEqual(expressions["use_shuffled_weight"], "False")
        self.assertEqual(
            expressions["routing_method_type"],
            "tllm_enums.RoutingMethodType.DeepSeekV3.value",
        )


class SparseMLAContractTest(unittest.TestCase):
    def test_manifest_matches_pinned_layer3_checkpoint_header(self) -> None:
        metadata = pinned_sparse_mla_metadata()

        self.assertEqual(
            glm53_flash.MAIN_SPARSE_MLA_LAYER_IDS,
            (3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43),
        )
        self.assertEqual(len(glm53_checkpoint.SPARSE_MLA_WEIGHTS), 18)
        self.assertEqual(
            {
                glm53_checkpoint.sparse_mla_checkpoint_key(3, spec): (
                    spec.dtype,
                    spec.shape,
                )
                for spec in glm53_checkpoint.SPARSE_MLA_WEIGHTS
            },
            metadata,
        )
        for layer_id in glm53_flash.MAIN_SPARSE_MLA_LAYER_IDS:
            self.assertTrue(
                all(
                    glm53_checkpoint.sparse_mla_checkpoint_key(
                        layer_id, spec
                    ).startswith(f"model.language_model.layers.{layer_id}.self_attn.")
                    for spec in glm53_checkpoint.SPARSE_MLA_WEIGHTS
                )
            )

        with self.assertRaisesRegex(ValueError, "not a main sparse MLA layer"):
            glm53_checkpoint.sparse_mla_checkpoint_key(
                4, glm53_checkpoint.SPARSE_MLA_WEIGHTS[0]
            )

    def test_tp4_shapes_match_sparse_mla_sharding(self) -> None:
        expected = {
            "q_a_proj.weight": (1536, 4096),
            "q_a_proj.weight_scale_inv": (12, 32),
            "q_a_layernorm.weight": (1536,),
            "kv_a_proj_with_mqa.weight": (512, 4096),
            "kv_a_proj_with_mqa.weight_scale_inv": (4, 32),
            "kv_a_layernorm.weight": (512,),
            "q_b_proj.weight": (4096, 1536),
            "q_b_proj.weight_scale_inv": (32, 12),
            "kv_b_proj.weight": (8192, 512),
            "o_proj.weight": (4096, 4096),
            "o_proj.weight_scale_inv": (32, 32),
            "indexer.wq_b.weight": (4096, 1536),
            "indexer.wk.weight": (128, 4096),
            "indexer.weights_proj.weight": (32, 4096),
            "indexer.index_kpool_compress_gate": (128, 4096),
            "indexer.index_kpool_compress_ape": (4, 128),
            "indexer.k_norm.weight": (128,),
            "indexer.k_norm.bias": (128,),
        }

        self.assertEqual(
            {
                spec.name: spec.local_shape()
                for spec in glm53_checkpoint.SPARSE_MLA_WEIGHTS
            },
            expected,
        )

    def test_compound_history_geometry(self) -> None:
        self.assertEqual(glm53_flash.SPARSE_MLA_HISTORY_BLOCK_TOKENS, 128)
        self.assertEqual(
            glm53_flash.SPARSE_MLA_BF16_LATENT_PAGES,
            ((1, 64, 512), (1, 64, 512)),
        )
        self.assertEqual(glm53_flash.SPARSE_MLA_INDEX_TAIL_K_BF16_SHAPE, (4, 128))


class KDAStateTest(unittest.TestCase):
    def test_tp4_state_geometry(self) -> None:
        self.assertEqual(
            kda_state_shapes(batch_size=2),
            KDAState(
                recurrent=(2, 16, 128, 128),
                conv=(2, 6144, 3),
            ),
        )
        shapes = kda_state_shapes(batch_size=1)
        self.assertEqual(
            prod(shapes.recurrent) * 4 + prod(shapes.conv) * 2,
            1_085_440,
        )

    def test_full_model_state_cross_checks_design_ledger(self) -> None:
        shapes = kda_state_shapes(batch_size=1, tp_size=1)
        state_bytes = prod(shapes.recurrent) * 4 + prod(shapes.conv) * 2
        self.assertEqual(state_bytes * 34, 147_619_840)

    def test_rejects_invalid_geometry(self) -> None:
        with self.assertRaisesRegex(ValueError, "batch_size must be positive"):
            kda_state_shapes(batch_size=0)
        with self.assertRaisesRegex(ValueError, "64 heads cannot shard 3 ways"):
            kda_state_shapes(batch_size=1, tp_size=3)


class KDADecodeContractTest(unittest.TestCase):
    def test_packed_weight_and_workspace_geometry(self) -> None:
        self.assertEqual(RMS_NORM_EPS, 1e-5)
        self.assertEqual(
            kda_decode_workspace_shapes(7),
            KDADecodeWorkspace(
                projection=(7, 6416),
                output_gate=(7, 2048),
                gate_raw=(7, 2048),
                local_output=(7, 16, 128),
                output=(7, 4096),
            ),
        )
        self.assertEqual(HIDDEN_SIZE, 4096)

    def test_decode_uses_the_fixed_operation_sequence(self) -> None:
        source = (
            Path(__file__).resolve().parents[1] / "src/infer/models/glm53_flash/ops/core.py"
        ).read_text()
        tree = ast.parse(source)
        method = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "decode_attention"
        )
        calls = [node for node in ast.walk(method) if isinstance(node, ast.Call)]

        self.assertEqual(count_calls(calls, "mm"), 2)
        self.assertEqual(count_calls(calls, "glm53_megafuse_decode"), 1)
        self.assertEqual(count_calls(calls, "_project_kda_output"), 1)
        megafuse = next(
            call for call in calls if call_name(call) == "glm53_megafuse_decode"
        )
        self.assertEqual(
            ast.unparse(
                next(k.value for k in megafuse.keywords if k.arg == "conv_state")
            ),
            "state.conv",
        )
        self.assertEqual(
            ast.unparse(next(k.value for k in megafuse.keywords if k.arg == "out")),
            "workspace.local_output",
        )
        self.assertEqual(
            ast.unparse(
                next(k.value for k in megafuse.keywords if k.arg == "state_indices")
            ),
            "state_indices",
        )

    def test_target_verify_uses_one_fused_recurrence(self) -> None:
        source = (
            Path(__file__).resolve().parents[1] / "src/infer/models/glm53_flash/ops/core.py"
        ).read_text()
        tree = ast.parse(source)
        method = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == "_verify_kda_attention"
        )
        calls = [node for node in ast.walk(method) if isinstance(node, ast.Call)]

        self.assertEqual(count_calls(calls, "glm53_megafuse_verify"), 1)
        self.assertEqual(count_calls(calls, "glm53_megafuse_decode"), 0)
        self.assertEqual(count_calls(calls, "_rmsnorm_sigmoid_gate"), 1)

    def test_target_verify_selects_request_major_groups_from_pooled_state(self) -> None:
        source = (
            Path(__file__).resolve().parents[1] / "src/infer/models/glm53_flash/ops/core.py"
        ).read_text()
        function = next(
            node
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.FunctionDef)
            and node.name == "_verify_kda_attention"
        )

        class Tensor:
            def __init__(self, shape=()) -> None:
                self.shape = shape

            def __getitem__(self, _index):
                return self

            def t(self):
                return self

            def zero_(self):
                return self

        class Kernel:
            def __getitem__(self, _grid):
                return lambda *_args, **_kwargs: None

        calls = []
        module = ModuleType("pooled_kda_verify_test")
        module.__dict__.update(
            GLM53_TARGET_VERIFY_WIDTH=glm53_flash.GLM53_TARGET_VERIFY_WIDTH,
            PACKED_QKV_SIZE=10,
            HEAD_DIM=2,
            LOCAL_HEADS=1,
            RMS_NORM_EPS=1e-5,
            _validate_kda_inputs=lambda hidden, *_args: hidden.shape[0],
            torch=SimpleNamespace(mm=lambda *_args, **_kwargs: None),
            glm53_megafuse_verify=lambda **kwargs: calls.append(kwargs),
            _rmsnorm_sigmoid_gate=Kernel(),
            _project_kda_output=lambda *_args: "output",
        )
        exec(  # noqa: S102
            compile(
                ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
                "glm53.py",
                "exec",
            ),
            module.__dict__,
        )
        state = SimpleNamespace(recurrent=Tensor((64, 1)), conv=Tensor())
        weights = SimpleNamespace(
            projection=Tensor(),
            gate_projections=(Tensor(), Tensor()),
            conv=Tensor(),
            a_log=Tensor((1,)),
            dt_bias=Tensor(),
            o_norm=Tensor(),
        )
        workspace = SimpleNamespace(
            projection=Tensor(),
            output_gate=Tensor(),
            gate_raw=Tensor(),
            local_output=Tensor((128, 1, 2)),
        )
        for groups in (1, 2, 16, 32):
            with self.subTest(groups=groups):
                rows = groups * glm53_flash.GLM53_TARGET_VERIFY_WIDTH
                transaction = SimpleNamespace(
                    recurrent=Tensor((rows, 1)), conv=Tensor()
                )
                state_indices = Tensor((groups,))
                returned = module._verify_kda_attention(
                    Tensor((rows, 1)),
                    weights,
                    state,
                    transaction,
                    state_indices,
                    workspace,
                    object(),
                )

                self.assertEqual(returned, "output")
                self.assertIs(calls[-1]["recurrent_state"], state.recurrent)
                self.assertIs(calls[-1]["state_indices"], state_indices)
                self.assertIs(calls[-1]["gate_raw"], workspace.gate_raw)
        self.assertEqual(len(calls), 4)
        with self.assertRaisesRegex(ValueError, "request-major Bx4 transactions"):
            module._verify_kda_attention(
                Tensor((8, 1)),
                weights,
                state,
                SimpleNamespace(recurrent=Tensor((8, 1)), conv=Tensor()),
                Tensor((1,)),
                workspace,
                object(),
            )


class KDAPrefillContractTest(unittest.TestCase):
    def test_packed_workspace_geometry(self) -> None:
        self.assertEqual(
            kda_prefill_workspace_shapes(total_tokens=4096, batch_size=7),
            KDAPrefillWorkspace(
                projection=(4096, 6416),
                gates=(2, 4096, 2048),
                beta=(1, 4096, 16),
                qkv=(3, 1, 4096, 16, 128),
                initial_state=(7, 16, 128, 128),
                kda_output=(1, 4096, 16, 128),
                final_state=(7, 16, 128, 128),
                kda_workspace_storage=(58_302_719,),
                kda_workspace=(58_302_592,),
                output=(4096, 4096),
            ),
        )
        self.assertEqual(kda_prefill_kernel_workspace_bytes(4096, 7), 58_302_592)
        self.assertEqual(
            kda_prefill_workspace_shapes(total_tokens=64, batch_size=1),
            KDAPrefillWorkspace(
                projection=(64, 6416),
                gates=(2, 64, 2048),
                beta=(1, 64, 16),
                qkv=(3, 1, 64, 16, 128),
                initial_state=(1, 16, 128, 128),
                kda_output=(1, 64, 16, 128),
                final_state=(1, 16, 128, 128),
                kda_workspace_storage=(1_108_223,),
                kda_workspace=(1_108_096,),
                output=(64, 4096),
            ),
        )

    def test_prefill_uses_the_fixed_packed_operation_sequence(self) -> None:
        source = (
            Path(__file__).resolve().parents[1] / "src/infer/models/glm53_flash/ops/core.py"
        ).read_text()
        tree = ast.parse(source)
        method = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "prefill_attention"
        )
        calls = [node for node in ast.walk(method) if isinstance(node, ast.Call)]

        self.assertEqual(count_calls(calls, "mm"), 2)
        self.assertEqual(count_calls(calls, "bmm"), 1)
        self.assertEqual(count_calls(calls, "glm53_segmented_conv_prefill"), 1)
        self.assertEqual(count_calls(calls, "glm53_prefill_kda"), 1)
        self.assertEqual(count_calls(calls, "_rmsnorm_sigmoid_gate"), 1)
        self.assertEqual(count_calls(calls, "all_reduce"), 1)
        self.assertEqual(count_calls(calls, "index_select"), 1)
        self.assertEqual(count_calls(calls, "index_copy_"), 1)
        self.assertEqual(count_calls(calls, "masked_fill_"), 0)
        self.assertEqual(count_calls(calls, "where"), 1)
        self.assertEqual(count_calls(calls, "mul"), 0)
        self.assertNotIn(".long()", ast.unparse(method))
        for allocation in ("empty", "empty_like", "zeros", "zeros_like", "tensor"):
            self.assertNotIn(f"torch.{allocation}", ast.unparse(method))
        self.assertIn("rows = slice(total_tokens)", ast.unparse(method))
        self.assertIn("workspace.projection[rows]", ast.unparse(method))
        self.assertIn("workspace.qkv[:, :, rows]", ast.unparse(method))

        gated_norm = next(
            call for call in calls if call_name(call) == "_rmsnorm_sigmoid_gate"
        )
        self.assertEqual(ast.unparse(gated_norm.args[0]), "core_output")
        self.assertEqual(ast.unparse(gated_norm.args[3]), "core_output")
        self.assertEqual(
            ast.unparse(gated_norm.func.slice), "(total_tokens * local_heads,)"
        )
        self.assertEqual(
            ast.unparse(next(k.value for k in gated_norm.keywords if k.arg == "WIDTH")),
            "HEAD_DIM",
        )

        where = next(call for call in calls if call_name(call) == "where")
        self.assertEqual(
            ast.unparse(where.args[0]),
            "batch.has_initial.view(batch_size, 1, 1, 1)",
        )
        self.assertEqual(
            ast.unparse(where.args[1]),
            "initial_state",
        )
        self.assertEqual(ast.unparse(where.args[2]), "final_state")
        self.assertEqual(
            ast.unparse(next(k.value for k in where.keywords if k.arg == "out")),
            "initial_state",
        )
        self.assertEqual(count_calls(calls, "zero_"), 1)

        conv = next(
            call for call in calls if call_name(call) == "glm53_segmented_conv_prefill"
        )
        self.assertEqual(
            ast.unparse(next(k.value for k in conv.keywords if k.arg == "out")),
            "qkv",
        )
        self.assertEqual(
            ast.unparse(next(k.value for k in conv.keywords if k.arg == "has_initial")),
            "batch.has_initial",
        )
        self.assertEqual(
            ast.unparse(
                next(k.value for k in conv.keywords if k.arg == "state_indices")
            ),
            "batch.state_indices",
        )

        kda = next(call for call in calls if call_name(call) == "glm53_prefill_kda")
        self.assertEqual(
            ast.unparse(next(k.value for k in kda.keywords if k.arg == "metadata")),
            "batch.metadata",
        )
        self.assertEqual(
            ast.unparse(next(k.value for k in kda.keywords if k.arg == "out")),
            "kda_output",
        )
        self.assertEqual(
            ast.unparse(next(k.value for k in kda.keywords if k.arg == "final_state")),
            "final_state",
        )
        self.assertEqual(
            ast.unparse(next(k.value for k in kda.keywords if k.arg == "workspace")),
            "workspace.kda_workspace",
        )
        index_copy = next(call for call in calls if call_name(call) == "index_copy_")
        self.assertEqual(
            ast.unparse(index_copy.args[1]),
            "batch.state_indices_int64",
        )
        index_select = next(call for call in calls if call_name(call) == "index_select")
        self.assertEqual(
            ast.unparse(index_select.args[2]),
            "batch.state_indices_int64",
        )

    def test_rejects_invalid_workspace_geometry(self) -> None:
        with self.assertRaisesRegex(ValueError, "total_tokens must be positive"):
            kda_prefill_workspace_shapes(total_tokens=0, batch_size=1)
        with self.assertRaisesRegex(ValueError, "batch_size must be positive"):
            kda_prefill_workspace_shapes(total_tokens=1, batch_size=0)


class DenseLayerContractTest(unittest.TestCase):
    def test_workspace_geometry(self) -> None:
        self.assertEqual(MHC_STREAMS, 4)
        self.assertEqual(FP8_BLOCK_SIZE, 128)
        self.assertEqual(
            dense_workspace_shapes(7),
            GLM53DenseWorkspace(
                mhc_sqrsum=(7, 8),
                mhc_dot=(7, 8, 24),
                post=(7, 4),
                comb=(7, 4, 4),
                collapsed=(7, 4096),
                normalized=(7, 4096),
                streams_mid=(7, 4, 4096),
                hidden_fp8=(7, 4096),
                hidden_scale=(7, 32),
                gate_up=(7, 6144),
                activated=(7, 3072),
                activated_fp8=(7, 3072),
                activated_scale=(7, 24),
                ffn_output=(7, 4096),
                streams_out=(7, 4, 4096),
            ),
        )
        with self.assertRaisesRegex(ValueError, "token_capacity must be positive"):
            dense_workspace_shapes(0)

    def test_mhc_prenorm_uses_qualified_splits_without_flat_workspace(self) -> None:
        self.assertEqual(glm53_flash.MHC_PRE_MAX_SPLITS, 8)
        source = (
            Path(__file__).resolve().parents[1] / "src/infer/models/glm53_flash/ops/core.py"
        ).read_text()
        functions = {
            node.name: node
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.FunctionDef)
        }
        mhc_pre = ast.unparse(functions["_mhc_pre"])

        self.assertIn("deep_gemm.tf32_hc_prenorm_gemm", mhc_pre)
        self.assertIn("num_splits = MHC_PRE_MAX_SPLITS", mhc_pre)
        self.assertIn("num_splits = 4", mhc_pre)
        self.assertIn("num_splits=num_splits", mhc_pre)
        finalize = next(
            call
            for call in ast.walk(functions["_mhc_pre"])
            if isinstance(call, ast.Call)
            if call_name(call) == "mhc_pre_big_fuse"
        )
        self.assertEqual(ast.unparse(finalize.args[-2]), "num_splits")
        self.assertNotIn("torch.mm", mhc_pre)
        self.assertNotIn("mhc_flat", mhc_pre)

    def test_layer_paths_have_one_attention_and_one_ffn_collective(self) -> None:
        source = (
            Path(__file__).resolve().parents[1] / "src/infer/models/glm53_flash/ops/core.py"
        ).read_text()
        tree = ast.parse(source)
        functions = {
            node.name: node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
        }
        finish = functions["_finish_dense_layer"]
        self.assertEqual(
            direct_call_names(finish),
            [
                "_mhc_post",
                "_mhc_pre",
                "_rmsnorm",
                "_mn_scale_view",
                "_quantize",
                "_fp8_gemm",
                "_swiglu_kernel",
                "_mn_scale_view",
                "_quantize",
                "_fp8_gemm",
                "_mhc_post",
            ],
        )
        self.assertEqual(count_calls(list(ast_calls(finish)), "all_reduce"), 1)

        for layer, attention in (
            ("prefill_dense_layer", "prefill_attention"),
            ("decode_dense_layer", "decode_attention"),
        ):
            method = functions[layer]
            self.assertEqual(
                direct_call_names(method),
                ["_prepare_dense_layer", attention, "_finish_dense_layer"],
            )
        self.assertEqual(
            count_calls(list(ast_calls(functions["prefill_attention"])), "all_reduce"),
            1,
        )
        self.assertEqual(
            count_calls(
                list(ast_calls(functions["_project_kda_output"])), "all_reduce"
            ),
            1,
        )

    def test_ops_pin_output_buffers_and_trtllm_gemm(self) -> None:
        source = (
            Path(__file__).resolve().parents[1] / "src/infer/models/glm53_flash/ops/core.py"
        ).read_text()
        tree = ast.parse(source)
        functions = {
            node.name: node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
        }
        rmsnorm = next(ast_calls(functions["_rmsnorm"]))
        gemm = next(
            call
            for call in ast_calls(functions["_fp8_gemm"])
            if call_name(call) == "gemm_fp8_nt_groupwise"
        )
        finish_calls = list(ast_calls(functions["_finish_dense_layer"]))
        quantize_calls = [
            call for call in finish_calls if call_name(call) == "_quantize"
        ]
        dense_gemms = [call for call in finish_calls if call_name(call) == "_fp8_gemm"]

        self.assertEqual(keyword_constant(rmsnorm, "enable_pdl"), True)
        self.assertEqual(
            ast.unparse(next(k.value for k in gemm.keywords if k.arg == "backend")),
            "'trtllm'",
        )
        self.assertEqual(
            ast.unparse(
                next(k.value for k in gemm.keywords if k.arg == "scale_major_mode")
            ),
            "None",
        )
        self.assertEqual(ast.unparse(gemm.args[3]), "weight_scale.T")
        self.assertEqual(
            ast.unparse(next(k.value for k in gemm.keywords if k.arg == "mma_sm")),
            "2 if input_.shape[0] >= 256 else 1",
        )
        self.assertEqual(
            ast.unparse(
                next(k.value for k in gemm.keywords if k.arg == "scale_granularity_mnk")
            ),
            "(1, FP8_BLOCK_SIZE, FP8_BLOCK_SIZE)",
        )
        self.assertTrue(any(keyword.arg == "out" for keyword in rmsnorm.keywords))
        self.assertTrue(any(keyword.arg == "out" for keyword in gemm.keywords))
        self.assertEqual(len(quantize_calls), 2)
        self.assertEqual(len(dense_gemms), 2)
        self.assertTrue(all(len(call.keywords) == 0 for call in dense_gemms))
        self.assertEqual(
            ast.unparse(functions["_mn_scale_view"]),
            "def _mn_scale_view(scale: torch.Tensor) -> torch.Tensor:\n"
            "    return scale.view(scale.shape[1], scale.shape[0]).T",
        )

        quantize_kernel = ast.unparse(functions["_quantize_fp8"])
        self.assertIn("do_not_specialize=['ROWS']", quantize_kernel)
        self.assertNotIn("ROWS: tl.constexpr", quantize_kernel)
        self.assertIn("scale_ptr + block * ROWS + row", quantize_kernel)
        self.assertNotIn("WRITE_", quantize_kernel)
        swiglu_kernel = ast.unparse(functions["_swiglu_kernel"])
        self.assertIn("block * tl.num_programs(0) + row", swiglu_kernel)

        gated_norm = ast.unparse(functions["_rmsnorm_sigmoid_gate"])
        self.assertIn("1.0 / tl.sqrt", gated_norm)
        self.assertIn("tl.sigmoid(gate)", gated_norm)
        self.assertIn("/ WIDTH + EPS", gated_norm)


class Layer0KDALoaderTest(unittest.TestCase):
    def test_pread_reads_only_the_local_axis_zero_slice(self) -> None:
        spec = WeightSpec("row", (4, 2), "BF16", 0)
        with tempfile.TemporaryFile() as checkpoint:
            checkpoint.write(bytes(range(16)))
            checkpoint.flush()

            loaded = _read_local_bytes(
                checkpoint.fileno(), 0, spec, tp_rank=1, tp_size=2
            )

        self.assertEqual(loaded, bytes(range(8, 16)))

    def test_pread_reads_only_the_local_axis_one_slice(self) -> None:
        spec = WeightSpec("column", (2, 4), "BF16", 1)
        with tempfile.TemporaryFile() as checkpoint:
            checkpoint.write(bytes(range(16)))
            checkpoint.flush()

            loaded = _read_local_bytes(
                checkpoint.fileno(), 0, spec, tp_rank=1, tp_size=2
            )

        self.assertEqual(loaded, bytes((4, 5, 6, 7, 12, 13, 14, 15)))

    def test_pread_uses_one_byte_fp8_elements(self) -> None:
        spec = WeightSpec("row", (4, 2), "F8_E4M3", 0)
        with tempfile.TemporaryFile() as checkpoint:
            checkpoint.write(bytes(range(8)))
            checkpoint.flush()

            loaded = _read_local_bytes(
                checkpoint.fileno(), 0, spec, tp_rank=1, tp_size=2
            )

        self.assertEqual(loaded, bytes(range(4, 8)))

    def test_offsets_are_relative_to_the_end_of_the_header(self) -> None:
        header = json.dumps(
            {
                "first": {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]},
                "second": {"dtype": "F32", "shape": [1], "data_offsets": [4, 8]},
            },
            separators=(",", ":"),
        ).encode()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.safetensors"
            path.write_bytes(len(header).to_bytes(8, "little") + header + bytes(8))

            offsets = _safetensor_data_offsets(path)

        self.assertEqual(
            offsets, {"first": 8 + len(header), "second": 12 + len(header)}
        )

    def test_loader_pins_pread_backend_and_resident_promotions(self) -> None:
        source = (
            Path(__file__).resolve().parents[1] / "src/infer/models/glm53_flash/checkpoint.py"
        ).read_text()
        tree = ast.parse(source)
        loader = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "_load_layer0_raw"
        )
        pack_kda = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "_pack_kda_weights"
        )
        pack_mhc = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "_pack_mhc_weights"
        )

        self.assertEqual(SAFETENSORS_VERSION, "0.8.0")
        calls = [node for node in ast.walk(loader) if isinstance(node, ast.Call)]
        safe_open = next(call for call in calls if call_name(call) == "safe_open")
        self.assertEqual(keyword_constant(safe_open, "backend"), "pread")
        self.assertEqual(ast.unparse(pack_kda).count(".to(torch.float32)"), 1)
        self.assertEqual(ast.unparse(pack_mhc).count(".to(torch.float32)"), 1)

    def test_dense_packer_fuses_gate_before_up(self) -> None:
        source = (
            Path(__file__).resolve().parents[1] / "src/infer/models/glm53_flash/checkpoint.py"
        ).read_text()
        tree = ast.parse(source)
        loader = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == "_pack_dense_layer_weights"
        )
        cats = [
            call
            for call in ast.walk(loader)
            if isinstance(call, ast.Call) and call_name(call) == "cat"
        ]

        self.assertEqual(len(cats), 2)
        self.assertEqual(
            ast.unparse(cats[0].args[0]),
            "(raw['mlp.gate_proj.weight'], raw['mlp.up_proj.weight'])",
        )
        self.assertEqual(
            ast.unparse(cats[1].args[0]),
            "(raw['mlp.gate_proj.weight_scale_inv'], "
            "raw['mlp.up_proj.weight_scale_inv'])",
        )


class SparseKDALoaderTest(unittest.TestCase):
    def test_all_layer_prefixes_metadata_routes_and_pins(self) -> None:
        expected_ids = (
            4,
            5,
            6,
            8,
            9,
            10,
            12,
            13,
            14,
            16,
            17,
            18,
            20,
            21,
            22,
            24,
            25,
            26,
            28,
            29,
            30,
            32,
            33,
            34,
            36,
            37,
            38,
            40,
            41,
            42,
            44,
        )
        layout = pinned_sparse_kda_layout()
        expected_pins = {
            filename: (size, header_size)
            for entries in layout.values()
            for filename, size, header_size, _count in entries
        }

        self.assertEqual(glm53_flash.SPARSE_KDA_LAYER_IDS, expected_ids)
        self.assertEqual(tuple(layout), expected_ids)
        self.assertEqual(glm53_checkpoint.SPARSE_KDA_SHARD_PINS, expected_pins)
        for layer_id, entries in layout.items():
            prefix = f"model.language_model.layers.{layer_id}.self_attn."
            metadata = pinned_metadata(layer_id)
            specs = glm53_checkpoint._sparse_kda_checkpoint_specs(layer_id)
            shards = glm53_checkpoint._sparse_kda_checkpoint_shards(layer_id, specs)
            counts = {
                filename: tuple(shards.values()).count(filename)
                for filename in set(shards.values())
            }

            self.assertEqual(set(specs), set(metadata))
            self.assertEqual(
                {key: (spec.dtype, spec.shape) for key, spec in specs.items()},
                metadata,
            )
            self.assertTrue(
                all(
                    spec.checkpoint_prefix == prefix and spec.checkpoint_key == key
                    for key, spec in specs.items()
                )
            )
            self.assertEqual(
                counts,
                {filename: count for filename, _size, _header, count in entries},
            )

    def test_layer9_secondary_shard_has_exact_three_weights(self) -> None:
        specs = glm53_checkpoint._sparse_kda_checkpoint_specs(9)
        shards = glm53_checkpoint._sparse_kda_checkpoint_shards(9, specs)
        secondary = "model-00062-of-00062.safetensors"

        self.assertEqual(
            {spec.name for key, spec in specs.items() if shards[key] == secondary},
            {"q_proj.weight", "v_conv1d.weight", "v_proj.weight"},
        )

    def test_invalid_layers_fail_before_checkpoint_io(self) -> None:
        with patch.object(glm53_checkpoint, "_load_indexed_weights") as load:
            for layer_id in (
                -1,
                *glm53_flash.DENSE_LAYER_IDS,
                *glm53_flash.MAIN_SPARSE_MLA_LAYER_IDS,
                45,
                46,
            ):
                with self.assertRaisesRegex(ValueError, "not a main sparse KDA layer"):
                    glm53_checkpoint.load_glm53_sparse_kda_weights(
                        "/missing-checkpoint", layer_id, 0, "cuda"
                    )

        load.assert_not_called()

    def test_layer9_loader_routes_fixture_across_two_shards(self) -> None:
        prefix = "model.language_model.layers.9.self_attn."
        primary = "model-00061-of-00062.safetensors"
        secondary = "model-00062-of-00062.safetensors"
        secondary_names = {"q_proj.weight", "v_conv1d.weight", "v_proj.weight"}
        weight_map = {
            key: secondary if key.removeprefix(prefix) in secondary_names else primary
            for key in pinned_metadata(9)
        }
        shard_sizes = {primary: 9, secondary: 10}
        metadata = {
            filename: {
                key: value
                for key, value in pinned_metadata(9).items()
                if weight_map[key] == filename
            }
            for filename in shard_sizes
        }
        opens = []
        loads = []

        class Checkpoint:
            def __init__(self, values):
                self.values = values

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def offset_keys(self):
                return self.values

            def get_slice(self, name):
                dtype, shape = self.values[name]
                return type(
                    "Slice",
                    (),
                    {
                        "get_dtype": lambda _self: dtype,
                        "get_shape": lambda _self: shape,
                    },
                )()

        def safe_open(path, **kwargs):
            opens.append((Path(path).name, kwargs))
            return Checkpoint(metadata[Path(path).name])

        def load_local(descriptor, _offset, spec, tp_rank, device, _torch, tp_size):
            size = os.fstat(descriptor).st_size
            loads.append((size, spec.name, tp_rank, device, tp_size))
            return (size, spec.name)

        safetensors = ModuleType("safetensors")
        safetensors.safe_open = safe_open
        torch = ModuleType("torch")
        pins = {filename: (size, 0) for filename, size in shard_sizes.items()}
        index_payload = json.dumps(
            {"weight_map": {**weight_map, "vision.weight": "ignored"}}
        ).encode()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / glm53_checkpoint.SAFETENSORS_INDEX).write_bytes(index_payload)
            for filename, size in shard_sizes.items():
                path = root / filename
                path.write_bytes(bytes(8))
                with path.open("ab") as shard:
                    shard.truncate(size)

            with (
                patch.dict(sys.modules, {"safetensors": safetensors, "torch": torch}),
                patch.object(glm53_checkpoint, "SPARSE_KDA_SHARD_PINS", pins),
                patch.object(
                    glm53_checkpoint,
                    "SAFETENSORS_INDEX_PIN",
                    (len(index_payload), hashlib.sha256(index_payload).hexdigest()),
                ),
                patch.object(glm53_checkpoint, "_require_safetensors_version"),
                patch.object(
                    glm53_checkpoint,
                    "_safetensor_data_offsets",
                    side_effect=lambda path: dict.fromkeys(metadata[path.name], 0),
                ),
                patch.object(
                    glm53_checkpoint, "_load_local_tensor", side_effect=load_local
                ),
                patch.object(
                    glm53_checkpoint,
                    "_pack_kda_weights",
                    side_effect=lambda raw, _torch: raw,
                ) as pack_kda,
            ):
                output = glm53_checkpoint.load_glm53_sparse_kda_weights(
                    root,
                    layer_id=9,
                    tp_rank=3,
                    device="cuda",
                    attention_tp_size=1,
                )

        self.assertEqual([name for name, _kwargs in opens], sorted(shard_sizes))
        self.assertTrue(
            all(
                kwargs == {"framework": "pt", "device": "cpu", "backend": "pread"}
                for _name, kwargs in opens
            )
        )
        self.assertEqual(len(loads), 15)
        self.assertEqual(sum(size == 9 for size, *_rest in loads), 12)
        self.assertEqual(sum(size == 10 for size, *_rest in loads), 3)
        self.assertEqual({rank for _size, _name, rank, _device, _tp in loads}, {3})
        self.assertEqual(
            {device for _size, _name, _rank, device, _tp in loads}, {"cuda"}
        )
        self.assertEqual({tp for *_rest, tp in loads}, {1})
        self.assertEqual(output["q_proj.weight"], (10, "q_proj.weight"))
        self.assertEqual(output["v_conv1d.weight"], (10, "v_conv1d.weight"))
        self.assertEqual(output["v_proj.weight"], (10, "v_proj.weight"))
        self.assertEqual(output["k_proj.weight"], (9, "k_proj.weight"))
        pack_kda.assert_called_once()


class DenseLayerLoaderTest(unittest.TestCase):
    def test_all_dense_layer_prefixes_and_pinned_shards(self) -> None:
        expected_shard_counts = {
            0: {"model-00002-of-00062.safetensors": 29},
            1: {
                "model-00002-of-00062.safetensors": 24,
                "model-00003-of-00062.safetensors": 5,
            },
            2: {"model-00017-of-00062.safetensors": 29},
        }

        for layer_id in glm53_flash.DENSE_LAYER_IDS:
            specs = glm53_checkpoint._dense_layer_checkpoint_specs(layer_id)
            prefix = glm53_checkpoint.DENSE_LAYER_PREFIX.format(layer_id=layer_id)
            shards = glm53_checkpoint._dense_layer_checkpoint_shards(layer_id, specs)
            counts = {
                name: tuple(shards.values()).count(name)
                for name in set(shards.values())
            }

            self.assertEqual(len(specs), 29)
            self.assertTrue(all(key.startswith(prefix) for key in specs))
            self.assertEqual(counts, expected_shard_counts[layer_id])

        self.assertEqual(
            glm53_checkpoint.DENSE_LAYER_SHARD_PINS,
            {
                "model-00002-of-00062.safetensors": (5_320_647_824, 160_856),
                "model-00003-of-00062.safetensors": (5_363_467_432, 175_048),
                "model-00017-of-00062.safetensors": (5_364_084_560, 168_216),
            },
        )

    def test_index_rejects_missing_extra_and_wrong_layer_mapping(self) -> None:
        layer_id = 1
        prefix = glm53_checkpoint.DENSE_LAYER_PREFIX.format(layer_id=layer_id)
        specs = glm53_checkpoint._dense_layer_checkpoint_specs(layer_id)
        expected = glm53_checkpoint._dense_layer_checkpoint_shards(layer_id, specs)
        weight_map = dict(expected)
        missing = next(iter(weight_map))
        weight_map.pop(missing)
        wrong = next(iter(weight_map))
        weight_map[wrong] = "model-00017-of-00062.safetensors"
        extra = f"{prefix}unexpected.weight"
        weight_map[extra] = "model-00002-of-00062.safetensors"

        with self.assertRaises(ValueError) as error:
            glm53_checkpoint._validate_indexed_weight_map(
                weight_map,
                expected,
                prefix,
                "dense layer 1",
            )

        message = str(error.exception)
        self.assertIn(f"missing {missing}", message)
        self.assertIn(f"unexpected {extra}", message)
        self.assertIn(f"{wrong} maps to", message)

    def test_layer1_loader_routes_fixture_across_two_shards(self) -> None:
        layer_id = 1
        specs = glm53_checkpoint._dense_layer_checkpoint_specs(layer_id)
        expected = glm53_checkpoint._dense_layer_checkpoint_shards(layer_id, specs)
        prefix = glm53_checkpoint.DENSE_LAYER_PREFIX.format(layer_id=layer_id)
        shard_sizes = {
            "model-00002-of-00062.safetensors": 9,
            "model-00003-of-00062.safetensors": 10,
        }
        metadata = {
            filename: {
                key: (spec.dtype, spec.shape)
                for key, spec in specs.items()
                if expected[key] == filename
            }
            for filename in shard_sizes
        }
        opens = []
        loads = []

        class Checkpoint:
            def __init__(self, values):
                self.values = values

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def offset_keys(self):
                return self.values

            def get_slice(self, name):
                dtype, shape = self.values[name]
                return type(
                    "Slice",
                    (),
                    {
                        "get_dtype": lambda _self: dtype,
                        "get_shape": lambda _self: shape,
                    },
                )()

        def safe_open(path, **kwargs):
            opens.append((Path(path).name, kwargs))
            return Checkpoint(metadata[Path(path).name])

        def load_local(descriptor, _offset, spec, tp_rank, device, _torch, tp_size):
            size = os.fstat(descriptor).st_size
            loads.append((size, spec.name, tp_rank, device, tp_size))
            return (size, spec.name)

        safetensors = ModuleType("safetensors")
        safetensors.safe_open = safe_open
        torch = ModuleType("torch")
        pins = {filename: (size, 0) for filename, size in shard_sizes.items()}
        index_payload = json.dumps(
            {"weight_map": {**expected, "vision.weight": "ignored"}}
        ).encode()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / glm53_checkpoint.SAFETENSORS_INDEX).write_bytes(index_payload)
            for filename, size in shard_sizes.items():
                path = root / filename
                path.write_bytes(bytes(8))
                with path.open("ab") as shard:
                    shard.truncate(size)

            with (
                patch.dict(sys.modules, {"safetensors": safetensors, "torch": torch}),
                patch.object(glm53_checkpoint, "DENSE_LAYER_SHARD_PINS", pins),
                patch.object(
                    glm53_checkpoint,
                    "SAFETENSORS_INDEX_PIN",
                    (len(index_payload), hashlib.sha256(index_payload).hexdigest()),
                ),
                patch.object(glm53_checkpoint, "_require_safetensors_version"),
                patch.object(
                    glm53_checkpoint,
                    "_safetensor_data_offsets",
                    side_effect=lambda path: dict.fromkeys(metadata[path.name], 0),
                ),
                patch.object(
                    glm53_checkpoint, "_load_local_tensor", side_effect=load_local
                ),
                patch.object(
                    glm53_checkpoint,
                    "_pack_dense_layer_weights",
                    side_effect=lambda raw, _torch: raw,
                ),
            ):
                output = glm53_checkpoint.load_glm53_dense_layer_weights(
                    root, layer_id, 3, "cuda", attention_tp_size=1
                )

        self.assertEqual([name for name, _kwargs in opens], sorted(shard_sizes))
        self.assertTrue(
            all(
                kwargs == {"framework": "pt", "device": "cpu", "backend": "pread"}
                for _name, kwargs in opens
            )
        )
        self.assertEqual(len(loads), 29)
        self.assertEqual({rank for _size, _name, rank, _device, _tp in loads}, {3})
        self.assertEqual(
            {device for _size, _name, _rank, device, _tp in loads}, {"cuda"}
        )
        self.assertEqual({tp for *_rest, tp in loads}, {1})
        self.assertEqual(
            output["q_proj.weight"],
            (shard_sizes["model-00003-of-00062.safetensors"], "q_proj.weight"),
        )
        self.assertEqual(
            output["k_proj.weight"],
            (shard_sizes["model-00002-of-00062.safetensors"], "k_proj.weight"),
        )
        self.assertTrue(all(key.startswith(prefix) for key in specs))

    def test_dense_packer_preserves_exact_order_and_promotions(self) -> None:
        class Tensor:
            def __init__(self, expression):
                self.expression = expression

            def squeeze(self, dimension):
                return Tensor(f"squeeze({self.expression},{dimension})")

            def to(self, dtype):
                return Tensor(f"to({self.expression},{dtype})")

        class Torch:
            float32 = "float32"

            @staticmethod
            def cat(tensors):
                return Tensor(
                    f"cat({','.join(tensor.expression for tensor in tensors)})"
                )

            @staticmethod
            def stack(tensors):
                return Tensor(
                    f"stack({','.join(tensor.expression for tensor in tensors)})"
                )

        raw = {
            spec.name: Tensor(spec.name)
            for spec in glm53_checkpoint.DENSE_LAYER_WEIGHTS
        }
        packed = glm53_checkpoint._pack_dense_layer_weights(raw, Torch)

        self.assertEqual(
            packed.attention.projection.expression,
            "cat(q_proj.weight,k_proj.weight,v_proj.weight,f_a_proj.weight,"
            "g_a_proj.weight,b_proj.weight)",
        )
        self.assertEqual(
            packed.attention.gate_projections.expression,
            "stack(f_b_proj.weight,g_b_proj.weight)",
        )
        self.assertEqual(
            packed.attention.conv.expression,
            "to(squeeze(cat(q_conv1d.weight,k_conv1d.weight,v_conv1d.weight),1),"
            "float32)",
        )
        self.assertEqual(packed.attention_mhc.fn.expression, "to(hc_attn_fn,float32)")
        self.assertEqual(packed.ffn_mhc.fn.expression, "to(hc_ffn_fn,float32)")
        self.assertEqual(
            packed.ffn.gate_up.expression,
            "cat(mlp.gate_proj.weight,mlp.up_proj.weight)",
        )
        self.assertEqual(
            packed.ffn.gate_up_scale_inv.expression,
            "cat(mlp.gate_proj.weight_scale_inv,mlp.up_proj.weight_scale_inv)",
        )
        self.assertIs(packed.ffn.down, raw["mlp.down_proj.weight"])
        self.assertIs(
            packed.ffn.down_scale_inv,
            raw["mlp.down_proj.weight_scale_inv"],
        )

    def test_public_loader_is_layer_driven(self) -> None:
        source = (
            Path(__file__).resolve().parents[1] / "src/infer/models/glm53_flash/checkpoint.py"
        ).read_text()
        function = next(
            node
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.FunctionDef)
            and node.name == "load_glm53_dense_layer_weights"
        )

        self.assertNotIn("LAYER0", ast.unparse(function))
        for layer_id in (-1, 3, 45):
            with self.assertRaisesRegex(ValueError, "not a main dense layer"):
                glm53_checkpoint.dense_layer_checkpoint_key(
                    layer_id, glm53_checkpoint.DENSE_LAYER_WEIGHTS[0]
                )


def count_calls(calls: list[ast.Call], name: str) -> int:
    return sum(call_name(call) == name for call in calls)


def ast_calls(node: ast.AST):
    return (child for child in ast.walk(node) if isinstance(child, ast.Call))


def direct_call_names(function: ast.FunctionDef) -> list[str | None]:
    calls = []
    for statement in function.body:
        value = getattr(statement, "value", None)
        if isinstance(value, ast.Call):
            calls.append(call_name(value))
    return calls


def call_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    if isinstance(call.func, ast.Subscript) and isinstance(call.func.value, ast.Name):
        return call.func.value.id
    return None


def keyword_constant(call: ast.Call, name: str) -> object:
    keyword = next(keyword for keyword in call.keywords if keyword.arg == name)
    if not isinstance(keyword.value, ast.Constant):
        raise TypeError(f"{name} is not constant")
    return keyword.value.value


if __name__ == "__main__":
    unittest.main()

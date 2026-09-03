import hashlib
import json
import os
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from importlib.metadata import version
from math import prod
from pathlib import Path

from infer.models.glm53_flash.model import (
    CONV_KERNEL_SIZE,
    DENSE_LAYER_IDS,
    HEAD_DIM,
    HIDDEN_SIZE,
    MAIN_SPARSE_MLA_LAYER_IDS,
    MOE_INTERMEDIATE_SIZE,
    NEXTN_LAYER_ID,
    NUM_HEADS,
    NUM_ROUTED_EXPERTS,
    PROJECTION_SIZE,
    SPARSE_FFN_CLAMP,
    SPARSE_FFN_LAYER_IDS,
    SPARSE_FFN_SHARED_FC1_ORDER,
    SPARSE_FFN_TEP4_SHAPES,
    SPARSE_KDA_LAYER_IDS,
    SPARSE_MLA_QK_NOPE_HEAD_DIM,
    SPARSE_MLA_VALUE_HEAD_DIM,
    TP_SIZE,
    VOCAB_SIZE,
    DenseFFNWeights,
    GLM53EndpointWeights,
    GLM53LayerWeights,
    GLM53SparseKDALayerWeights,
    GLM53SparseLayerShellWeights,
    GLM53SparseMLALayerWeights,
    KDAWeights,
    MHCWeights,
    ScalarType,
    Shape,
    SparseFFNWeights,
    SparseMLADecodeWeights,
    SparseMLAOutputWeights,
    SparseMLAProjectionWeights,
    glm53_local_heads,
)

DENSE_LAYER_PREFIX = "model.language_model.layers.{layer_id}."
SAFETENSORS_VERSION = "0.8.0"
EMBEDDING_SHARD = ("model-00001-of-00062.safetensors", 5_365_306_704, 93_512)
FINAL_NORM_SHARD = ("model-00062-of-00062.safetensors", 1_261_584_968, 39_488)
DENSE_LAYER_SHARDS = (
    ("model-00002-of-00062.safetensors", 5_320_647_824, 160_856),
    ("model-00003-of-00062.safetensors", 5_363_467_432, 175_048),
    ("model-00017-of-00062.safetensors", 5_364_084_560, 168_216),
)
DENSE_LAYER_SHARD_PINS = {
    filename: (size, header_size) for filename, size, header_size in DENSE_LAYER_SHARDS
}

NEXTN_LAYER_SHARDS = (EMBEDDING_SHARD, DENSE_LAYER_SHARDS[0])
NEXTN_SHARD_PINS = {
    filename: (size, header_size) for filename, size, header_size in NEXTN_LAYER_SHARDS
}
DENSE_LAYER_DEFAULT_SHARDS = {
    0: DENSE_LAYER_SHARDS[0][0],
    1: DENSE_LAYER_SHARDS[0][0],
    2: DENSE_LAYER_SHARDS[2][0],
}
LAYER1_SHARD3_WEIGHTS = frozenset(
    {
        "o_proj.weight",
        "q_conv1d.weight",
        "q_proj.weight",
        "v_conv1d.weight",
        "v_proj.weight",
    }
)
SPARSE_KDA_LAYER_SHARDS = (
    (4, "model-00047-of-00062.safetensors", 5_364_341_072, 172_528),
    (5, "model-00056-of-00062.safetensors", 5_364_341_080, 172_536),
    (6, "model-00057-of-00062.safetensors", 5_364_341_048, 172_504),
    (8, "model-00060-of-00062.safetensors", 5_364_341_064, 172_520),
    (9, "model-00061-of-00062.safetensors", 5_303_993_768, 173_856),
    (10, "model-00004-of-00062.safetensors", 5_364_342_296, 173_752),
    (12, "model-00007-of-00062.safetensors", 5_364_342_528, 173_984),
    (13, "model-00008-of-00062.safetensors", 5_364_342_304, 173_760),
    (14, "model-00010-of-00062.safetensors", 5_361_982_080, 173_984),
    (16, "model-00012-of-00062.safetensors", 5_364_342_288, 173_744),
    (17, "model-00014-of-00062.safetensors", 5_364_342_504, 173_960),
    (18, "model-00015-of-00062.safetensors", 5_364_342_304, 173_760),
    (20, "model-00018-of-00062.safetensors", 5_364_342_304, 173_760),
    (21, "model-00019-of-00062.safetensors", 5_364_342_272, 173_728),
    (22, "model-00021-of-00062.safetensors", 5_364_342_376, 173_832),
    (24, "model-00024-of-00062.safetensors", 5_364_342_640, 174_096),
    (25, "model-00025-of-00062.safetensors", 5_364_342_304, 173_760),
    (26, "model-00026-of-00062.safetensors", 5_364_342_272, 173_728),
    (28, "model-00029-of-00062.safetensors", 5_364_342_304, 173_760),
    (29, "model-00031-of-00062.safetensors", 5_364_341_544, 173_000),
    (30, "model-00033-of-00062.safetensors", 5_364_342_272, 173_728),
    (32, "model-00036-of-00062.safetensors", 5_364_342_304, 173_760),
    (33, "model-00038-of-00062.safetensors", 5_364_342_624, 174_080),
    (34, "model-00039-of-00062.safetensors", 5_364_342_296, 173_752),
    (36, "model-00042-of-00062.safetensors", 5_364_342_368, 173_824),
    (37, "model-00043-of-00062.safetensors", 5_364_342_296, 173_752),
    (38, "model-00045-of-00062.safetensors", 5_364_342_600, 174_056),
    (40, "model-00049-of-00062.safetensors", 5_364_342_344, 173_800),
    (41, "model-00050-of-00062.safetensors", 5_364_342_312, 173_768),
    (42, "model-00052-of-00062.safetensors", 5_364_342_576, 174_032),
    (44, "model-00054-of-00062.safetensors", 5_364_342_272, 173_728),
)
SPARSE_KDA_LAYER9_SECONDARY_SHARD = (
    "model-00062-of-00062.safetensors",
    1_261_584_968,
    39_488,
)

SPARSE_KDA_PREFIX = "model.language_model.layers.{layer_id}.self_attn."
SPARSE_KDA_DEFAULT_SHARDS = {
    layer_id: filename for layer_id, filename, *_rest in SPARSE_KDA_LAYER_SHARDS
}
SPARSE_KDA_SHARD_PINS = {
    filename: (size, header_size)
    for _layer_id, filename, size, header_size in (
        *SPARSE_KDA_LAYER_SHARDS,
        (9, *SPARSE_KDA_LAYER9_SECONDARY_SHARD),
    )
}
SPARSE_KDA_LAYER9_SECONDARY_WEIGHTS = frozenset(
    {"q_proj.weight", "v_conv1d.weight", "v_proj.weight"}
)


SPARSE_MLA_PREFIX = "model.language_model.layers.{layer_id}.self_attn."

_SPARSE_FFN_LOADABLE_LAYER_IDS = (*SPARSE_FFN_LAYER_IDS, NEXTN_LAYER_ID)

SPARSE_FFN_PREFIX = "model.language_model.layers.{layer_id}.mlp."

SAFETENSORS_INDEX = "model.safetensors.index.json"
SAFETENSORS_INDEX_PIN = (
    8_406_613,
    "3c3f40366a53c3fd7974b4eab7881a365a98c2a4329150befebab99fe7c18b05",
)
SPARSE_LAYER_SHARD_PINS = {
    "model-00003-of-00062.safetensors": (5_363_467_432, 175_048),
    "model-00004-of-00062.safetensors": (5_364_342_296, 173_752),
    "model-00005-of-00062.safetensors": (5_363_915_920, 178_480),
    "model-00006-of-00062.safetensors": (5_361_809_384, 180_192),
    "model-00007-of-00062.safetensors": (5_364_342_528, 173_984),
    "model-00008-of-00062.safetensors": (5_364_342_304, 173_760),
    "model-00009-of-00062.safetensors": (5_364_169_856, 180_216),
    "model-00010-of-00062.safetensors": (5_361_982_080, 173_984),
    "model-00011-of-00062.safetensors": (5_363_915_912, 178_472),
    "model-00012-of-00062.safetensors": (5_364_342_288, 173_744),
    "model-00013-of-00062.safetensors": (5_361_809_408, 180_216),
    "model-00014-of-00062.safetensors": (5_364_342_504, 173_960),
    "model-00015-of-00062.safetensors": (5_364_342_304, 173_760),
    "model-00016-of-00062.safetensors": (5_361_809_176, 179_984),
    "model-00017-of-00062.safetensors": (5_364_084_560, 168_216),
    "model-00018-of-00062.safetensors": (5_364_342_304, 173_760),
    "model-00019-of-00062.safetensors": (5_364_342_272, 173_728),
    "model-00020-of-00062.safetensors": (5_361_809_528, 180_336),
    "model-00021-of-00062.safetensors": (5_364_342_376, 173_832),
    "model-00022-of-00062.safetensors": (5_363_915_936, 178_496),
    "model-00023-of-00062.safetensors": (5_361_809_264, 180_072),
    "model-00024-of-00062.safetensors": (5_364_342_640, 174_096),
    "model-00025-of-00062.safetensors": (5_364_342_304, 173_760),
    "model-00026-of-00062.safetensors": (5_364_342_272, 173_728),
    "model-00027-of-00062.safetensors": (5_361_809_560, 180_368),
    "model-00028-of-00062.safetensors": (5_363_915_976, 178_536),
    "model-00029-of-00062.safetensors": (5_364_342_304, 173_760),
    "model-00030-of-00062.safetensors": (5_361_809_296, 180_104),
    "model-00031-of-00062.safetensors": (5_364_341_544, 173_000),
    "model-00032-of-00062.safetensors": (5_363_915_232, 177_792),
    "model-00033-of-00062.safetensors": (5_364_342_272, 173_728),
    "model-00034-of-00062.safetensors": (5_361_809_552, 180_360),
    "model-00035-of-00062.safetensors": (5_363_915_984, 178_544),
    "model-00036-of-00062.safetensors": (5_364_342_304, 173_760),
    "model-00037-of-00062.safetensors": (5_361_809_288, 180_096),
    "model-00038-of-00062.safetensors": (5_364_342_624, 174_080),
    "model-00039-of-00062.safetensors": (5_364_342_296, 173_752),
    "model-00040-of-00062.safetensors": (5_363_915_912, 178_472),
    "model-00041-of-00062.safetensors": (5_361_809_552, 180_360),
    "model-00042-of-00062.safetensors": (5_364_342_368, 173_824),
    "model-00043-of-00062.safetensors": (5_364_342_296, 173_752),
    "model-00044-of-00062.safetensors": (5_361_809_312, 180_120),
    "model-00045-of-00062.safetensors": (5_364_342_600, 174_056),
    "model-00046-of-00062.safetensors": (5_363_915_368, 177_928),
    "model-00047-of-00062.safetensors": (5_364_341_072, 172_528),
    "model-00048-of-00062.safetensors": (5_361_809_576, 180_384),
    "model-00049-of-00062.safetensors": (5_364_342_344, 173_800),
    "model-00050-of-00062.safetensors": (5_364_342_312, 173_768),
    "model-00051-of-00062.safetensors": (5_361_809_336, 180_144),
    "model-00052-of-00062.safetensors": (5_364_342_576, 174_032),
    "model-00053-of-00062.safetensors": (5_363_915_936, 178_496),
    "model-00054-of-00062.safetensors": (5_364_342_272, 173_728),
    "model-00055-of-00062.safetensors": (5_361_808_304, 179_112),
    "model-00056-of-00062.safetensors": (5_364_341_080, 172_536),
    "model-00057-of-00062.safetensors": (5_364_341_048, 172_504),
    "model-00058-of-00062.safetensors": (5_361_808_088, 178_896),
    "model-00059-of-00062.safetensors": (5_363_914_912, 177_472),
    "model-00060-of-00062.safetensors": (5_364_341_064, 172_520),
    "model-00061-of-00062.safetensors": (5_303_993_768, 173_856),
}
SPARSE_FFN_SHARD_PINS = SPARSE_LAYER_SHARD_PINS
SPARSE_LAYER_SHELL_SHARDS = {
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

@dataclass(frozen=True, slots=True)
class WeightSpec:
    name: str
    shape: Shape
    dtype: ScalarType
    shard_axis: int | None
    checkpoint_prefix: str = ""

    @property
    def checkpoint_key(self) -> str:
        return f"{self.checkpoint_prefix}{self.name}"

    def local_shape(self, tp_size: int = TP_SIZE) -> Shape:
        if self.shard_axis is None:
            return self.shape

        dimension = self.shape[self.shard_axis]
        if dimension % tp_size:
            raise ValueError(
                f"{self.name} axis {self.shard_axis} cannot shard {tp_size} ways"
            )

        shape = list(self.shape)
        shape[self.shard_axis] = dimension // tp_size
        return tuple(shape)

GLM53_ENDPOINT_WEIGHTS = (
    WeightSpec(
        "embed_tokens.weight",
        (VOCAB_SIZE, HIDDEN_SIZE),
        "BF16",
        0,
        "model.language_model.",
    ),
    WeightSpec("lm_head.weight", (VOCAB_SIZE, HIDDEN_SIZE), "BF16", 0, ""),
    WeightSpec("norm.weight", (HIDDEN_SIZE,), "BF16", None, "model.language_model."),
)
GLM53_ENDPOINT_SHARDS = {
    GLM53_ENDPOINT_WEIGHTS[0].checkpoint_key: EMBEDDING_SHARD[0],
    GLM53_ENDPOINT_WEIGHTS[1].checkpoint_key: EMBEDDING_SHARD[0],
    GLM53_ENDPOINT_WEIGHTS[2].checkpoint_key: FINAL_NORM_SHARD[0],
}


def load_glm53_endpoint_weights(
    checkpoint_dir: str | Path,
    tp_rank: int,
    device: str | int,
    attention_tp_size: int = TP_SIZE,
) -> GLM53EndpointWeights[object]:
    import torch

    if not 0 <= tp_rank < TP_SIZE:
        raise ValueError(f"tp_rank must be in [0, {TP_SIZE}), got {tp_rank}")
    glm53_local_heads(attention_tp_size)

    root = Path(checkpoint_dir)
    index = json.loads((root / SAFETENSORS_INDEX).read_bytes())
    weight_map = index.get("weight_map") if isinstance(index, Mapping) else None
    if not isinstance(weight_map, Mapping):
        raise TypeError("checkpoint index must contain a weight_map")
    if {key: weight_map.get(key) for key in GLM53_ENDPOINT_SHARDS} != (
        GLM53_ENDPOINT_SHARDS
    ):
        raise ValueError("invalid GLM endpoint tensor-to-shard mapping")

    raw: dict[str, object] = {}
    for filename, expected_size, expected_header_size in (
        EMBEDDING_SHARD,
        FINAL_NORM_SHARD,
    ):
        path = root / filename
        with path.open("rb") as shard:
            header_size = int.from_bytes(shard.read(8), "little")
        if path.stat().st_size != expected_size or header_size != expected_header_size:
            raise ValueError(f"invalid {filename} size/header")
        specs = tuple(
            spec
            for spec in GLM53_ENDPOINT_WEIGHTS
            if GLM53_ENDPOINT_SHARDS[spec.checkpoint_key] == filename
        )
        keys = frozenset(spec.checkpoint_key for spec in specs)
        raw.update(
            _load_layer0_raw(
                path,
                tp_rank,
                device,
                specs,
                lambda metadata, specs=specs: _validate_metadata(
                    metadata, specs, "", "GLM endpoint"
                ),
                torch,
                keys,
                attention_tp_size,
            )
        )

    return GLM53EndpointWeights(
        embedding=raw["embed_tokens.weight"],
        final_norm=raw["norm.weight"],
        lm_head=raw["lm_head.weight"],
    )


DENSE_KDA_WEIGHTS = (
    WeightSpec("q_proj.weight", (PROJECTION_SIZE, HIDDEN_SIZE), "BF16", 0),
    WeightSpec("k_proj.weight", (PROJECTION_SIZE, HIDDEN_SIZE), "BF16", 0),
    WeightSpec("v_proj.weight", (PROJECTION_SIZE, HIDDEN_SIZE), "BF16", 0),
    WeightSpec("q_conv1d.weight", (PROJECTION_SIZE, 1, CONV_KERNEL_SIZE), "BF16", 0),
    WeightSpec("k_conv1d.weight", (PROJECTION_SIZE, 1, CONV_KERNEL_SIZE), "BF16", 0),
    WeightSpec("v_conv1d.weight", (PROJECTION_SIZE, 1, CONV_KERNEL_SIZE), "BF16", 0),
    WeightSpec("f_a_proj.weight", (HEAD_DIM, HIDDEN_SIZE), "BF16", None),
    WeightSpec("f_b_proj.weight", (PROJECTION_SIZE, HEAD_DIM), "BF16", 0),
    WeightSpec("A_log", (NUM_HEADS,), "F32", 0),
    WeightSpec("dt_bias", (PROJECTION_SIZE,), "F32", 0),
    WeightSpec("b_proj.weight", (NUM_HEADS, HIDDEN_SIZE), "BF16", 0),
    WeightSpec("g_a_proj.weight", (HEAD_DIM, HIDDEN_SIZE), "BF16", None),
    WeightSpec("g_b_proj.weight", (PROJECTION_SIZE, HEAD_DIM), "BF16", 0),
    WeightSpec("o_norm.weight", (HEAD_DIM,), "BF16", None),
    WeightSpec("o_proj.weight", (HIDDEN_SIZE, PROJECTION_SIZE), "BF16", 1),
)
DENSE_LAYER_WEIGHTS = DENSE_KDA_WEIGHTS + (
    WeightSpec("hc_attn_base", (24,), "F32", None),
    WeightSpec("hc_attn_fn", (24, 4 * HIDDEN_SIZE), "BF16", None),
    WeightSpec("hc_attn_scale", (3,), "F32", None),
    WeightSpec("hc_ffn_base", (24,), "F32", None),
    WeightSpec("hc_ffn_fn", (24, 4 * HIDDEN_SIZE), "BF16", None),
    WeightSpec("hc_ffn_scale", (3,), "F32", None),
    WeightSpec("input_layernorm.weight", (HIDDEN_SIZE,), "BF16", None),
    WeightSpec(
        "post_attention_layernorm.weight",
        (HIDDEN_SIZE,),
        "BF16",
        None,
    ),
    WeightSpec(
        "mlp.gate_proj.weight",
        (12_288, HIDDEN_SIZE),
        "F8_E4M3",
        0,
    ),
    WeightSpec(
        "mlp.gate_proj.weight_scale_inv",
        (96, 32),
        "F32",
        0,
    ),
    WeightSpec(
        "mlp.up_proj.weight",
        (12_288, HIDDEN_SIZE),
        "F8_E4M3",
        0,
    ),
    WeightSpec(
        "mlp.up_proj.weight_scale_inv",
        (96, 32),
        "F32",
        0,
    ),
    WeightSpec(
        "mlp.down_proj.weight",
        (HIDDEN_SIZE, 12_288),
        "F8_E4M3",
        1,
    ),
    WeightSpec(
        "mlp.down_proj.weight_scale_inv",
        (32, 96),
        "F32",
        1,
    ),
)

SPARSE_LAYER_SHELL_WEIGHTS = (
    WeightSpec("hc_attn_base", (24,), "F32", None, ""),
    WeightSpec("hc_attn_fn", (24, 4 * HIDDEN_SIZE), "BF16", None, ""),
    WeightSpec("hc_attn_scale", (3,), "F32", None, ""),
    WeightSpec("hc_ffn_base", (24,), "F32", None, ""),
    WeightSpec("hc_ffn_fn", (24, 4 * HIDDEN_SIZE), "BF16", None, ""),
    WeightSpec("hc_ffn_scale", (3,), "F32", None, ""),
    WeightSpec("input_layernorm.weight", (HIDDEN_SIZE,), "BF16", None, ""),
    WeightSpec(
        "post_attention_layernorm.weight",
        (HIDDEN_SIZE,),
        "BF16",
        None,
        "",
    ),
)
SPARSE_LAYER_SHELL_SCOPE_SUFFIXES = (
    "hc_attn_",
    "hc_ffn_",
    "input_layernorm.",
    "post_attention_layernorm.",
)


def dense_layer_checkpoint_key(layer_id: int, spec: WeightSpec) -> str:
    if layer_id not in DENSE_LAYER_IDS:
        raise ValueError(f"layer {layer_id} is not a main dense layer")
    if spec in DENSE_KDA_WEIGHTS:
        suffix = f"self_attn.{spec.name}"
    elif spec in DENSE_LAYER_WEIGHTS:
        suffix = spec.name
    else:
        raise ValueError(f"{spec.name} is not a dense-layer weight")
    return f"{DENSE_LAYER_PREFIX.format(layer_id=layer_id)}{suffix}"


def _dense_layer_checkpoint_specs(layer_id: int) -> dict[str, WeightSpec]:
    return {
        dense_layer_checkpoint_key(layer_id, spec): spec for spec in DENSE_LAYER_WEIGHTS
    }


def _dense_layer_checkpoint_shards(
    layer_id: int,
    specs: Mapping[str, WeightSpec],
) -> dict[str, str]:
    default = DENSE_LAYER_DEFAULT_SHARDS[layer_id]
    return {
        key: (
            DENSE_LAYER_SHARDS[1][0]
            if layer_id == 1 and spec.name in LAYER1_SHARD3_WEIGHTS
            else default
        )
        for key, spec in specs.items()
    }


def _sparse_kda_checkpoint_specs(layer_id: int) -> dict[str, WeightSpec]:
    if layer_id not in SPARSE_KDA_LAYER_IDS:
        raise ValueError(f"layer {layer_id} is not a main sparse KDA layer")
    prefix = SPARSE_KDA_PREFIX.format(layer_id=layer_id)
    specs = tuple(
        WeightSpec(spec.name, spec.shape, spec.dtype, spec.shard_axis, prefix)
        for spec in DENSE_KDA_WEIGHTS
    )
    return {spec.checkpoint_key: spec for spec in specs}


def _sparse_kda_checkpoint_shards(
    layer_id: int,
    specs: Mapping[str, WeightSpec],
) -> dict[str, str]:
    primary = SPARSE_KDA_DEFAULT_SHARDS[layer_id]
    secondary = SPARSE_KDA_LAYER9_SECONDARY_SHARD[0]
    return {
        key: (
            secondary
            if layer_id == 9 and spec.name in SPARSE_KDA_LAYER9_SECONDARY_WEIGHTS
            else primary
        )
        for key, spec in specs.items()
    }


def _sparse_ffn_weight(
    name: str,
    shape: Shape,
    dtype: ScalarType,
    shard_axis: int | None = None,
) -> WeightSpec:
    return WeightSpec(name, shape, dtype, shard_axis)


SPARSE_FFN_ROUTER_WEIGHTS = (
    _sparse_ffn_weight("gate.weight", (NUM_ROUTED_EXPERTS, HIDDEN_SIZE), "BF16"),
    _sparse_ffn_weight("gate.e_score_correction_bias", (NUM_ROUTED_EXPERTS,), "F32"),
)
SPARSE_FFN_ROUTED_EXPERT_TEMPLATE = (
    _sparse_ffn_weight(
        "gate_proj.weight", (MOE_INTERMEDIATE_SIZE, HIDDEN_SIZE), "F8_E4M3"
    ),
    _sparse_ffn_weight("gate_proj.weight_scale_inv", (16, 32), "F32"),
    _sparse_ffn_weight(
        "up_proj.weight", (MOE_INTERMEDIATE_SIZE, HIDDEN_SIZE), "F8_E4M3"
    ),
    _sparse_ffn_weight("up_proj.weight_scale_inv", (16, 32), "F32"),
    _sparse_ffn_weight(
        "down_proj.weight", (HIDDEN_SIZE, MOE_INTERMEDIATE_SIZE), "F8_E4M3"
    ),
    _sparse_ffn_weight("down_proj.weight_scale_inv", (32, 16), "F32"),
)
SPARSE_FFN_SHARED_EXPERT_WEIGHTS = tuple(
    _sparse_ffn_weight(
        f"shared_experts.{spec.name}", spec.shape, spec.dtype, shard_axis
    )
    for spec, shard_axis in zip(
        SPARSE_FFN_ROUTED_EXPERT_TEMPLATE,
        (0, 0, 0, 0, 1, 1),
        strict=True,
    )
)


def _sparse_mla_weight(
    name: str,
    shape: Shape,
    dtype: ScalarType,
    shard_axis: int | None,
) -> WeightSpec:
    return WeightSpec(name, shape, dtype, shard_axis)


SPARSE_MLA_WEIGHTS = (
    _sparse_mla_weight("q_a_proj.weight", (1536, 4096), "F8_E4M3", None),
    _sparse_mla_weight("q_a_proj.weight_scale_inv", (12, 32), "F32", None),
    _sparse_mla_weight("q_a_layernorm.weight", (1536,), "BF16", None),
    _sparse_mla_weight("kv_a_proj_with_mqa.weight", (512, 4096), "F8_E4M3", None),
    _sparse_mla_weight("kv_a_proj_with_mqa.weight_scale_inv", (4, 32), "F32", None),
    _sparse_mla_weight("kv_a_layernorm.weight", (512,), "BF16", None),
    _sparse_mla_weight("q_b_proj.weight", (16384, 1536), "F8_E4M3", 0),
    _sparse_mla_weight("q_b_proj.weight_scale_inv", (128, 12), "F32", 0),
    _sparse_mla_weight("kv_b_proj.weight", (32768, 512), "BF16", 0),
    _sparse_mla_weight("o_proj.weight", (4096, 16384), "F8_E4M3", 1),
    _sparse_mla_weight("o_proj.weight_scale_inv", (32, 128), "F32", 1),
    _sparse_mla_weight("indexer.wq_b.weight", (4096, 1536), "BF16", None),
    _sparse_mla_weight("indexer.wk.weight", (128, 4096), "BF16", None),
    _sparse_mla_weight("indexer.weights_proj.weight", (32, 4096), "BF16", None),
    _sparse_mla_weight("indexer.index_kpool_compress_gate", (128, 4096), "BF16", None),
    _sparse_mla_weight("indexer.index_kpool_compress_ape", (4, 128), "BF16", None),
    _sparse_mla_weight("indexer.k_norm.weight", (128,), "BF16", None),
    _sparse_mla_weight("indexer.k_norm.bias", (128,), "BF16", None),
)


def _sparse_layer_shell_checkpoint_specs(
    layer_id: int,
) -> dict[str, WeightSpec]:
    if layer_id not in SPARSE_FFN_LAYER_IDS:
        raise ValueError(f"layer {layer_id} is not a main sparse layer")
    prefix = DENSE_LAYER_PREFIX.format(layer_id=layer_id)
    return {f"{prefix}{spec.name}": spec for spec in SPARSE_LAYER_SHELL_WEIGHTS}


def _sparse_layer_shell_checkpoint_shards(layer_id: int) -> dict[str, str]:
    specs = _sparse_layer_shell_checkpoint_specs(layer_id)
    primary, secondary = SPARSE_LAYER_SHELL_SHARDS[layer_id]
    return {
        key: (secondary if spec.name == "post_attention_layernorm.weight" else primary)
        for key, spec in specs.items()
    }


def _sparse_layer_shell_scopes(layer_id: int) -> tuple[str, ...]:
    if layer_id not in SPARSE_FFN_LAYER_IDS:
        raise ValueError(f"layer {layer_id} is not a main sparse layer")
    prefix = DENSE_LAYER_PREFIX.format(layer_id=layer_id)
    return tuple(f"{prefix}{suffix}" for suffix in SPARSE_LAYER_SHELL_SCOPE_SUFFIXES)

def sparse_ffn_checkpoint_key(
    layer_id: int,
    spec: WeightSpec,
    expert_id: int | None = None,
) -> str:
    if layer_id not in _SPARSE_FFN_LOADABLE_LAYER_IDS:
        raise ValueError(f"layer {layer_id} is not a sparse FFN layer")
    if spec in SPARSE_FFN_ROUTED_EXPERT_TEMPLATE:
        if expert_id is None:
            raise ValueError("routed expert weights require an expert_id")
        if not 0 <= expert_id < NUM_ROUTED_EXPERTS:
            raise ValueError(f"invalid expert_id {expert_id}")
        name = f"experts.{expert_id}.{spec.name}"
    elif expert_id is not None or spec not in (
        SPARSE_FFN_ROUTER_WEIGHTS + SPARSE_FFN_SHARED_EXPERT_WEIGHTS
    ):
        raise ValueError(f"{spec.name} is not a non-expert sparse FFN weight")
    else:
        name = spec.name
    return f"{SPARSE_FFN_PREFIX.format(layer_id=layer_id)}{name}"


def _sparse_ffn_checkpoint_specs(
    layer_id: int,
) -> Iterator[tuple[str, WeightSpec, int | None]]:
    for spec in SPARSE_FFN_ROUTER_WEIGHTS + SPARSE_FFN_SHARED_EXPERT_WEIGHTS:
        yield sparse_ffn_checkpoint_key(layer_id, spec), spec, None
    for expert_id in range(NUM_ROUTED_EXPERTS):
        for spec in SPARSE_FFN_ROUTED_EXPERT_TEMPLATE:
            yield sparse_ffn_checkpoint_key(layer_id, spec, expert_id), spec, expert_id


def local_sparse_ffn_expert_ids(
    tep_rank: int,
    tep_size: int = TP_SIZE,
) -> tuple[int, ...]:
    if tep_size < 1 or NUM_ROUTED_EXPERTS % tep_size:
        raise ValueError(f"{NUM_ROUTED_EXPERTS} experts cannot shard {tep_size} ways")
    if not 0 <= tep_rank < tep_size:
        raise ValueError(f"tep_rank must be in [0, {tep_size}), got {tep_rank}")
    local_experts = NUM_ROUTED_EXPERTS // tep_size
    return tuple(range(tep_rank * local_experts, (tep_rank + 1) * local_experts))


def validate_sparse_ffn_metadata(
    metadata: Mapping[str, tuple[str, Shape]],
    layer_id: int = 3,
) -> None:
    prefix = SPARSE_FFN_PREFIX.format(layer_id=layer_id)
    expected = {key: spec for key, spec, _ in _sparse_ffn_checkpoint_specs(layer_id)}
    _validate_metadata(metadata, expected, prefix, f"layer-{layer_id} sparse FFN")


def _sparse_ffn_layer_layout(
    weight_map: Mapping[str, str],
    layer_id: int,
) -> dict[str, str]:
    expected = {
        key for key, _spec, _expert_id in _sparse_ffn_checkpoint_specs(layer_id)
    }
    prefix = SPARSE_FFN_PREFIX.format(layer_id=layer_id)
    shards = {
        name: shard for name, shard in weight_map.items() if name.startswith(prefix)
    }
    errors = [f"missing {name}" for name in sorted(expected - shards.keys())]
    errors.extend(f"unexpected {name}" for name in sorted(shards.keys() - expected))
    shard_pins = (
        NEXTN_SHARD_PINS if layer_id == NEXTN_LAYER_ID else SPARSE_FFN_SHARD_PINS
    )
    errors.extend(
        f"{name} maps to unpinned shard {shard}"
        for name, shard in sorted(shards.items())
        if shard not in shard_pins
    )
    if errors:
        raise ValueError(
            f"invalid layer-{layer_id} sparse FFN tensor-to-shard mapping:\n"
            + "\n".join(errors)
        )
    return shards


def load_glm53_sparse_ffn_weights(
    checkpoint_dir: str | Path,
    layer_id: int,
    tep_rank: int,
    device: str | int,
) -> SparseFFNWeights[object]:
    if layer_id not in _SPARSE_FFN_LOADABLE_LAYER_IDS:
        raise ValueError(f"layer {layer_id} is not a sparse FFN layer")
    if not 0 <= tep_rank < TP_SIZE:
        raise ValueError(f"tep_rank must be in [0, {TP_SIZE}), got {tep_rank}")
    _require_safetensors_version()

    import torch
    from safetensors import safe_open

    root = Path(checkpoint_dir)
    weight_map = _read_checkpoint_weight_map(root)
    layer_shards = _sparse_ffn_layer_layout(weight_map, layer_id)

    metadata: dict[str, tuple[str, Shape]] = {}
    offsets: dict[str, dict[str, int]] = {}
    prefix = SPARSE_FFN_PREFIX.format(layer_id=layer_id)
    filenames = tuple(sorted(set(layer_shards.values())))
    shard_pins = (
        NEXTN_SHARD_PINS if layer_id == NEXTN_LAYER_ID else SPARSE_FFN_SHARD_PINS
    )
    for filename in filenames:
        expected_size, expected_header_size = shard_pins[filename]
        path = root / filename
        with path.open("rb") as shard:
            header_size = int.from_bytes(shard.read(8), "little")
        if path.stat().st_size != expected_size or header_size != expected_header_size:
            raise ValueError(f"invalid {filename} size/header")
        with safe_open(path, framework="pt", device="cpu", backend="pread") as shard:
            names = [name for name in shard.offset_keys() if name.startswith(prefix)]
            expected_names = {
                name
                for name, shard_name in layer_shards.items()
                if shard_name == filename
            }
            if set(names) != expected_names:
                raise ValueError(f"invalid {filename} sparse FFN tensor scope")
            metadata.update(
                {
                    name: (
                        shard.get_slice(name).get_dtype(),
                        tuple(shard.get_slice(name).get_shape()),
                    )
                    for name in names
                }
            )
        offsets[filename] = _safetensor_data_offsets(path)
    validate_sparse_ffn_metadata(metadata, layer_id)

    descriptors = {
        filename: os.open(root / filename, os.O_RDONLY) for filename in filenames
    }

    def load(spec: WeightSpec, expert_id: int | None = None) -> object:
        key = sparse_ffn_checkpoint_key(layer_id, spec, expert_id)
        filename = layer_shards[key]
        return _load_local_tensor(
            descriptors[filename],
            offsets[filename][key],
            spec,
            tep_rank,
            device,
            torch,
        )

    try:
        return _pack_sparse_ffn_weights(load, tep_rank, device, torch)
    finally:
        for descriptor in descriptors.values():
            os.close(descriptor)


def _pack_sparse_ffn_weights(
    load: Callable[[WeightSpec, int | None], object],
    tep_rank: int,
    device: str | int,
    torch: object,
) -> SparseFFNWeights[object]:
    routed = {spec.name: spec for spec in SPARSE_FFN_ROUTED_EXPERT_TEMPLATE}
    shared = {spec.name: spec for spec in SPARSE_FFN_SHARED_EXPERT_WEIGHTS}
    router_specs = {spec.name: spec for spec in SPARSE_FFN_ROUTER_WEIGHTS}
    shapes = SPARSE_FFN_TEP4_SHAPES

    def empty(name: str, dtype: object) -> object:
        return torch.empty(getattr(shapes, name), dtype=dtype, device=device)

    routed_up_gate = empty("routed_up_gate", torch.float8_e4m3fn)
    routed_up_gate_scale_inv = empty("routed_up_gate_scale_inv", torch.float32)
    routed_down = empty("routed_down", torch.float8_e4m3fn)
    routed_down_scale_inv = empty("routed_down_scale_inv", torch.float32)
    expert_targets = (
        (routed_up_gate[:, :MOE_INTERMEDIATE_SIZE], "up_proj.weight"),
        (routed_up_gate[:, MOE_INTERMEDIATE_SIZE:], "gate_proj.weight"),
        (routed_up_gate_scale_inv[:, :16], "up_proj.weight_scale_inv"),
        (routed_up_gate_scale_inv[:, 16:], "gate_proj.weight_scale_inv"),
        (routed_down, "down_proj.weight"),
        (routed_down_scale_inv, "down_proj.weight_scale_inv"),
    )
    expert_ids = local_sparse_ffn_expert_ids(tep_rank)
    for target, name in expert_targets:
        for local_index, expert_id in enumerate(expert_ids):
            target[local_index].copy_(load(routed[name], expert_id))

    def shared_pair(suffix: str) -> object:
        return torch.cat(
            tuple(
                load(shared[f"shared_experts.{projection}.{suffix}"], None)
                for projection in SPARSE_FFN_SHARED_FC1_ORDER
            )
        )

    router = load(router_specs["gate.weight"], None)
    return SparseFFNWeights(
        router=router,
        router_t=router.T,
        correction_bias=load(router_specs["gate.e_score_correction_bias"], None),
        routed_up_gate=routed_up_gate,
        routed_up_gate_scale_inv=routed_up_gate_scale_inv,
        routed_down=routed_down,
        routed_down_scale_inv=routed_down_scale_inv,
        routed_clamp=torch.full(
            shapes.routed_clamp,
            SPARSE_FFN_CLAMP,
            dtype=torch.float32,
            device=device,
        ),
        shared_gate_up=shared_pair("weight"),
        shared_gate_up_scale_inv=shared_pair("weight_scale_inv"),
        shared_down=load(shared["shared_experts.down_proj.weight"], None),
        shared_down_scale_inv=load(
            shared["shared_experts.down_proj.weight_scale_inv"], None
        ),
    )


def sparse_mla_checkpoint_key(layer_id: int, spec: WeightSpec) -> str:
    if layer_id not in MAIN_SPARSE_MLA_LAYER_IDS:
        raise ValueError(f"layer {layer_id} is not a main sparse MLA layer")
    return f"{SPARSE_MLA_PREFIX.format(layer_id=layer_id)}{spec.name}"


def _sparse_mla_layer_checkpoint_specs(layer_id: int) -> dict[str, WeightSpec]:
    if layer_id not in MAIN_SPARSE_MLA_LAYER_IDS:
        raise ValueError(f"layer {layer_id} is not a main sparse MLA layer")
    return {
        **_sparse_layer_shell_checkpoint_specs(layer_id),
        **{
            sparse_mla_checkpoint_key(layer_id, spec): spec
            for spec in SPARSE_MLA_WEIGHTS
        },
    }


def _sparse_mla_layer_checkpoint_shards(layer_id: int) -> dict[str, str]:
    shell = _sparse_layer_shell_checkpoint_shards(layer_id)
    secondary = SPARSE_LAYER_SHELL_SHARDS[layer_id][1]
    return {
        **shell,
        **{
            sparse_mla_checkpoint_key(layer_id, spec): secondary
            for spec in SPARSE_MLA_WEIGHTS
        },
    }


def pack_sparse_mla_projection_weights(
    raw: Mapping[str, object],
    torch: object,
) -> SparseMLAProjectionWeights[object]:
    return SparseMLAProjectionWeights(
        low_rank=torch.cat((raw["q_a_proj.weight"], raw["kv_a_proj_with_mqa.weight"])),
        low_rank_scale_inv=torch.cat(
            (
                raw["q_a_proj.weight_scale_inv"],
                raw["kv_a_proj_with_mqa.weight_scale_inv"],
            )
        ),
        q_norm=raw["q_a_layernorm.weight"],
        kv_norm=raw["kv_a_layernorm.weight"],
        q_b=raw["q_b_proj.weight"],
        q_b_scale_inv=raw["q_b_proj.weight_scale_inv"],
        wq_b=raw["indexer.wq_b.weight"],
        index_prep=torch.cat(
            (
                raw["indexer.wk.weight"],
                raw["indexer.index_kpool_compress_gate"],
                raw["indexer.weights_proj.weight"],
            )
        ),
        k_norm=raw["indexer.k_norm.weight"],
        k_bias=raw["indexer.k_norm.bias"],
    )


def pack_sparse_mla_decode_weights(
    raw: Mapping[str, object],
    torch: object,
) -> SparseMLADecodeWeights[object]:
    kv_b = _sparse_mla_kv_b_heads(raw["kv_b_proj.weight"])
    return SparseMLADecodeWeights(
        w_kc=kv_b[:, :SPARSE_MLA_QK_NOPE_HEAD_DIM].contiguous(),
        pool_ape=raw["indexer.index_kpool_compress_ape"],
    )


def pack_sparse_mla_output_weights(
    raw: Mapping[str, object],
) -> SparseMLAOutputWeights[object]:
    kv_b = _sparse_mla_kv_b_heads(raw["kv_b_proj.weight"])
    return SparseMLAOutputWeights(
        w_vc=kv_b[:, SPARSE_MLA_QK_NOPE_HEAD_DIM:].transpose(1, 2).contiguous(),
        o_proj=raw["o_proj.weight"],
        o_proj_scale_inv=raw["o_proj.weight_scale_inv"],
    )


def _sparse_mla_kv_b_heads(kv_b: object) -> object:
    head_width = SPARSE_MLA_QK_NOPE_HEAD_DIM + SPARSE_MLA_VALUE_HEAD_DIM
    if kv_b.ndim != 2 or kv_b.shape[0] % head_width:
        raise ValueError("invalid sparse MLA KV-B weight shape")
    return kv_b.view(kv_b.shape[0] // head_width, head_width, kv_b.shape[1])


def load_glm53_sparse_layer_shell_weights(
    checkpoint_dir: str | Path,
    layer_id: int,
    tp_rank: int,
    device: str | int,
) -> GLM53SparseLayerShellWeights[object]:
    specs = _sparse_layer_shell_checkpoint_specs(layer_id)
    shards = _sparse_layer_shell_checkpoint_shards(layer_id)

    import torch

    root = Path(checkpoint_dir)
    keyed = _load_indexed_weights(
        root,
        tp_rank,
        device,
        specs,
        shards,
        SPARSE_LAYER_SHARD_PINS,
        _sparse_layer_shell_scopes(layer_id),
        f"sparse layer {layer_id} shell",
        torch,
    )
    raw = {spec.name: keyed[key] for key, spec in specs.items()}
    return _pack_sparse_layer_shell_weights(raw, torch)


def load_glm53_sparse_mla_layer_weights(
    checkpoint_dir: str | Path,
    layer_id: int,
    tp_rank: int,
    device: str | int,
    attention_tp_size: int = TP_SIZE,
) -> GLM53SparseMLALayerWeights[object]:
    specs = _sparse_mla_layer_checkpoint_specs(layer_id)
    shards = _sparse_mla_layer_checkpoint_shards(layer_id)

    import torch

    root = Path(checkpoint_dir)
    keyed = _load_indexed_weights(
        root,
        tp_rank,
        device,
        specs,
        shards,
        SPARSE_LAYER_SHARD_PINS,
        (
            *_sparse_layer_shell_scopes(layer_id),
            SPARSE_MLA_PREFIX.format(layer_id=layer_id),
        ),
        f"sparse MLA layer {layer_id}",
        torch,
        attention_tp_size,
    )
    raw = {spec.name: keyed[key] for key, spec in specs.items()}
    shell = _pack_sparse_layer_shell_weights(raw, torch)
    kv_b = _sparse_mla_kv_b_heads(raw["kv_b_proj.weight"])
    return GLM53SparseMLALayerWeights(
        attention_mhc=shell.attention_mhc,
        input_norm=shell.input_norm,
        mla_projection=pack_sparse_mla_projection_weights(raw, torch),
        mla_decode=SparseMLADecodeWeights(
            w_kc=kv_b[:, :SPARSE_MLA_QK_NOPE_HEAD_DIM].contiguous(),
            pool_ape=raw["indexer.index_kpool_compress_ape"],
        ),
        mla_output=SparseMLAOutputWeights(
            w_vc=kv_b[:, SPARSE_MLA_QK_NOPE_HEAD_DIM:].transpose(1, 2).contiguous(),
            o_proj=raw["o_proj.weight"],
            o_proj_scale_inv=raw["o_proj.weight_scale_inv"],
        ),
        ffn_mhc=shell.ffn_mhc,
        post_attention_norm=shell.post_attention_norm,
        ffn=load_glm53_sparse_ffn_weights(root, layer_id, tp_rank, device),
    )


def _pack_sparse_layer_shell_weights(
    raw: Mapping[str, object], torch: object
) -> GLM53SparseLayerShellWeights[object]:
    return GLM53SparseLayerShellWeights(
        attention_mhc=_pack_mhc_weights(raw, "hc_attn", torch),
        input_norm=raw["input_layernorm.weight"],
        ffn_mhc=_pack_mhc_weights(raw, "hc_ffn", torch),
        post_attention_norm=raw["post_attention_layernorm.weight"],
    )


def load_glm53_sparse_kda_weights(
    checkpoint_dir: str | Path,
    layer_id: int,
    tp_rank: int,
    device: str | int,
    attention_tp_size: int = TP_SIZE,
) -> KDAWeights[object]:
    specs = _sparse_kda_checkpoint_specs(layer_id)

    import torch

    shards = _sparse_kda_checkpoint_shards(layer_id, specs)
    keyed = _load_indexed_weights(
        Path(checkpoint_dir),
        tp_rank,
        device,
        specs,
        shards,
        SPARSE_KDA_SHARD_PINS,
        SPARSE_KDA_PREFIX.format(layer_id=layer_id),
        f"sparse KDA layer {layer_id}",
        torch,
        attention_tp_size,
    )
    raw = {spec.name: keyed[key] for key, spec in specs.items()}
    return _pack_kda_weights(raw, torch)


def load_glm53_sparse_kda_layer_weights(
    checkpoint_dir: str | Path,
    layer_id: int,
    tp_rank: int,
    device: str | int,
    attention_tp_size: int = TP_SIZE,
) -> GLM53SparseKDALayerWeights[object]:
    if layer_id not in SPARSE_KDA_LAYER_IDS:
        raise ValueError(f"layer {layer_id} is not a main sparse KDA layer")
    if not 0 <= tp_rank < TP_SIZE:
        raise ValueError(f"tp_rank must be in [0, {TP_SIZE}), got {tp_rank}")

    root = Path(checkpoint_dir)
    shell = load_glm53_sparse_layer_shell_weights(root, layer_id, tp_rank, device)
    return GLM53SparseKDALayerWeights(
        attention_mhc=shell.attention_mhc,
        input_norm=shell.input_norm,
        attention=load_glm53_sparse_kda_weights(
            root, layer_id, tp_rank, device, attention_tp_size
        ),
        ffn_mhc=shell.ffn_mhc,
        post_attention_norm=shell.post_attention_norm,
        ffn=load_glm53_sparse_ffn_weights(root, layer_id, tp_rank, device),
    )


def load_glm53_dense_layer_weights(
    checkpoint_dir: str | Path,
    layer_id: int,
    tp_rank: int,
    device: str | int,
    attention_tp_size: int = TP_SIZE,
) -> GLM53LayerWeights[object]:
    import torch

    specs = _dense_layer_checkpoint_specs(layer_id)
    shards = _dense_layer_checkpoint_shards(layer_id, specs)
    keyed = _load_indexed_weights(
        Path(checkpoint_dir),
        tp_rank,
        device,
        specs,
        shards,
        DENSE_LAYER_SHARD_PINS,
        DENSE_LAYER_PREFIX.format(layer_id=layer_id),
        f"dense layer {layer_id}",
        torch,
        attention_tp_size,
    )
    raw = {spec.name: keyed[key] for key, spec in specs.items()}
    return _pack_dense_layer_weights(raw, torch)


def _pack_dense_layer_weights(
    raw: Mapping[str, object],
    torch: object,
) -> GLM53LayerWeights[object]:
    return GLM53LayerWeights(
        attention=_pack_kda_weights(raw, torch),
        attention_mhc=_pack_mhc_weights(raw, "hc_attn", torch),
        ffn_mhc=_pack_mhc_weights(raw, "hc_ffn", torch),
        input_norm=raw["input_layernorm.weight"],
        post_attention_norm=raw["post_attention_layernorm.weight"],
        ffn=DenseFFNWeights(
            gate_up=torch.cat((raw["mlp.gate_proj.weight"], raw["mlp.up_proj.weight"])),
            gate_up_scale_inv=torch.cat(
                (
                    raw["mlp.gate_proj.weight_scale_inv"],
                    raw["mlp.up_proj.weight_scale_inv"],
                )
            ),
            down=raw["mlp.down_proj.weight"],
            down_scale_inv=raw["mlp.down_proj.weight_scale_inv"],
        ),
    )


def _load_indexed_weights(
    root: Path,
    tp_rank: int,
    device: str | int,
    specs: Mapping[str, WeightSpec],
    expected_shards: Mapping[str, str],
    shard_pins: Mapping[str, tuple[int, int]],
    scope_prefix: str | tuple[str, ...],
    label: str,
    torch: object,
    tp_size: int = TP_SIZE,
) -> dict[str, object]:
    if not 0 <= tp_rank < TP_SIZE:
        raise ValueError(f"tp_rank must be in [0, {TP_SIZE}), got {tp_rank}")
    glm53_local_heads(tp_size)
    _require_safetensors_version()

    from safetensors import safe_open

    weight_map = _read_checkpoint_weight_map(root)
    _validate_indexed_weight_map(
        weight_map,
        expected_shards,
        scope_prefix,
        label,
    )
    shards = {key: weight_map[key] for key in specs}

    output: dict[str, object] = {}
    for filename in sorted(set(shards.values())):
        path = root / filename
        expected_size, expected_header_size = shard_pins[filename]
        with path.open("rb") as shard:
            header_size = int.from_bytes(shard.read(8), "little")
        if path.stat().st_size != expected_size or header_size != expected_header_size:
            raise ValueError(f"invalid {filename} size/header")

        shard_specs = {
            key: spec for key, spec in specs.items() if shards[key] == filename
        }
        with safe_open(
            path,
            framework="pt",
            device="cpu",
            backend="pread",
        ) as checkpoint:
            metadata = {
                name: (
                    checkpoint.get_slice(name).get_dtype(),
                    tuple(checkpoint.get_slice(name).get_shape()),
                )
                for name in checkpoint.offset_keys()
                if name.startswith(scope_prefix)
            }
            _validate_metadata(metadata, shard_specs, "", f"{label} {filename}")

        offsets = _safetensor_data_offsets(path)
        descriptor = os.open(path, os.O_RDONLY)
        try:
            output.update(
                {
                    key: _load_local_tensor(
                        descriptor,
                        offsets[key],
                        spec,
                        tp_rank,
                        device,
                        torch,
                        tp_size,
                    )
                    for key, spec in shard_specs.items()
                }
            )
        finally:
            os.close(descriptor)
    return output


def validate_glm53_checkpoint_inventory(weight_map: Mapping[str, str]) -> None:
    actual = frozenset(weight_map)
    target = _glm53_target_checkpoint_keys()
    nextn = _glm53_nextn_checkpoint_keys()
    visual = frozenset(key for key in actual if key.startswith("model.visual."))
    missing_target = target - actual
    missing_nextn = nextn - actual
    unexpected = actual - target - nextn - visual

    errors = []
    if missing_target:
        errors.append(f"missing {len(missing_target)} target keys")
    if missing_nextn:
        errors.append(f"missing {len(missing_nextn)} layer-45 NextN keys")
    if len(visual) != 347:
        errors.append(f"found {len(visual)} visual keys, expected 347")
    if unexpected:
        errors.append(
            f"found {len(unexpected)} unexpected keys, first {min(unexpected)}"
        )
    if errors:
        raise ValueError("invalid GLM checkpoint inventory:\n" + "\n".join(errors))


def _glm53_target_checkpoint_keys() -> frozenset[str]:
    keys = {spec.checkpoint_key for spec in GLM53_ENDPOINT_WEIGHTS}
    for layer_id in DENSE_LAYER_IDS:
        keys.update(_dense_layer_checkpoint_specs(layer_id))
    for layer_id in SPARSE_FFN_LAYER_IDS:
        keys.update(_sparse_layer_shell_checkpoint_specs(layer_id))
        keys.update(
            key for key, _spec, _expert_id in _sparse_ffn_checkpoint_specs(layer_id)
        )
    for layer_id in SPARSE_KDA_LAYER_IDS:
        keys.update(_sparse_kda_checkpoint_specs(layer_id))
    for layer_id in MAIN_SPARSE_MLA_LAYER_IDS:
        keys.update(
            sparse_mla_checkpoint_key(layer_id, spec) for spec in SPARSE_MLA_WEIGHTS
        )
    return frozenset(keys)


def _glm53_nextn_checkpoint_keys() -> frozenset[str]:
    prefix = DENSE_LAYER_PREFIX.format(layer_id=45)
    keys = {
        f"{prefix}{name}"
        for name in (
            "eh_proj.weight",
            "enorm.weight",
            "hnorm.weight",
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "shared_head.norm.weight",
        )
    }
    keys.update(f"{prefix}self_attn.{spec.name}" for spec in SPARSE_MLA_WEIGHTS)
    keys.update(
        f"{prefix}mlp.{spec.name}"
        for spec in SPARSE_FFN_ROUTER_WEIGHTS + SPARSE_FFN_SHARED_EXPERT_WEIGHTS
    )
    keys.update(
        f"{prefix}mlp.experts.{expert_id}.{spec.name}"
        for expert_id in range(NUM_ROUTED_EXPERTS)
        for spec in SPARSE_FFN_ROUTED_EXPERT_TEMPLATE
    )
    return frozenset(keys)


def _read_checkpoint_weight_map(root: Path) -> Mapping[str, str]:
    payload = (root / SAFETENSORS_INDEX).read_bytes()
    expected_size, expected_sha256 = SAFETENSORS_INDEX_PIN
    if len(payload) != expected_size or hashlib.sha256(payload).hexdigest() != (
        expected_sha256
    ):
        raise ValueError(f"invalid {SAFETENSORS_INDEX} size/SHA-256")
    index = json.loads(payload)
    weight_map = index.get("weight_map") if isinstance(index, Mapping) else None
    if not isinstance(weight_map, Mapping):
        raise TypeError("checkpoint index must contain a weight_map")
    return weight_map

def _validate_indexed_weight_map(
    weight_map: Mapping[str, str],
    expected: Mapping[str, str],
    scope_prefix: str | tuple[str, ...],
    label: str,
) -> None:
    actual = {
        name: shard
        for name, shard in weight_map.items()
        if name.startswith(scope_prefix)
    }
    errors = [f"missing {name}" for name in sorted(expected.keys() - actual.keys())]
    errors.extend(
        f"unexpected {name}" for name in sorted(actual.keys() - expected.keys())
    )
    errors.extend(
        f"{name} maps to {actual[name]}, expected {expected[name]}"
        for name in sorted(expected.keys() & actual.keys())
        if actual[name] != expected[name]
    )
    if errors:
        raise ValueError(
            f"invalid {label} tensor-to-shard mapping:\n" + "\n".join(errors)
        )


def _load_layer0_raw(
    shard_path: str | Path,
    tp_rank: int,
    device: str | int,
    specs: tuple[WeightSpec, ...],
    validator: Callable[[Mapping[str, tuple[str, Shape]]], None],
    torch: object,
    selected_keys: frozenset[str] | None = None,
    tp_size: int = TP_SIZE,
) -> dict[str, object]:
    if not 0 <= tp_rank < TP_SIZE:
        raise ValueError(f"tp_rank must be in [0, {TP_SIZE}), got {tp_rank}")
    glm53_local_heads(tp_size)
    _require_safetensors_version()

    from safetensors import safe_open

    path = Path(shard_path)
    with safe_open(
        path,
        framework="pt",
        device="cpu",
        backend="pread",
    ) as checkpoint:
        metadata = {
            name: (
                checkpoint.get_slice(name).get_dtype(),
                tuple(checkpoint.get_slice(name).get_shape()),
            )
            for name in checkpoint.offset_keys()
            if selected_keys is None or name in selected_keys
        }
        validator(metadata)

    # safetensors 0.8 reads a full tensor before slicing on CUDA's pread path.
    offsets = _safetensor_data_offsets(path)
    descriptor = os.open(path, os.O_RDONLY)
    try:
        return {
            spec.name: _load_local_tensor(
                descriptor,
                offsets[spec.checkpoint_key],
                spec,
                tp_rank,
                device,
                torch,
                tp_size,
            )
            for spec in specs
        }
    finally:
        os.close(descriptor)


def _require_safetensors_version() -> None:
    installed = version("safetensors")
    if installed != SAFETENSORS_VERSION:
        raise RuntimeError(
            f"safetensors is {installed}, expected {SAFETENSORS_VERSION}"
        )


def _pack_kda_weights(raw: Mapping[str, object], torch: object) -> KDAWeights[object]:
    projection = torch.cat(
        (
            raw["q_proj.weight"],
            raw["k_proj.weight"],
            raw["v_proj.weight"],
            raw["f_a_proj.weight"],
            raw["g_a_proj.weight"],
            raw["b_proj.weight"],
        )
    )
    gate_projections = torch.stack((raw["f_b_proj.weight"], raw["g_b_proj.weight"]))
    conv = (
        torch.cat(
            (
                raw["q_conv1d.weight"],
                raw["k_conv1d.weight"],
                raw["v_conv1d.weight"],
            )
        )
        .squeeze(1)
        .to(torch.float32)
    )

    return KDAWeights(
        projection=projection,
        gate_projections=gate_projections,
        conv=conv,
        a_log=raw["A_log"],
        dt_bias=raw["dt_bias"],
        o_norm=raw["o_norm.weight"],
        o_projection=raw["o_proj.weight"],
    )


def _pack_mhc_weights(
    raw: Mapping[str, object],
    prefix: str,
    torch: object,
) -> MHCWeights[object]:
    return MHCWeights(
        base=raw[f"{prefix}_base"],
        fn=raw[f"{prefix}_fn"].to(torch.float32),
        scale=raw[f"{prefix}_scale"],
    )



def _validate_metadata(
    metadata: Mapping[str, tuple[str, Shape]],
    specs: tuple[WeightSpec, ...] | Mapping[str, WeightSpec],
    prefix: str,
    label: str,
) -> None:
    expected = (
        specs
        if isinstance(specs, Mapping)
        else {spec.checkpoint_key: spec for spec in specs}
    )
    actual = {
        name: value for name, value in metadata.items() if name.startswith(prefix)
    }
    errors = [f"missing {name}" for name in sorted(expected.keys() - actual.keys())]
    errors.extend(
        f"unexpected {name}" for name in sorted(actual.keys() - expected.keys())
    )

    for name in sorted(expected.keys() & actual.keys()):
        dtype, shape = actual[name]
        spec = expected[name]
        if dtype != spec.dtype:
            errors.append(f"{name} has dtype {dtype}, expected {spec.dtype}")
        if tuple(shape) != spec.shape:
            errors.append(f"{name} has shape {tuple(shape)}, expected {spec.shape}")

    if errors:
        raise ValueError(f"invalid {label} metadata:\n" + "\n".join(errors))

def _safetensor_data_offsets(path: Path) -> dict[str, int]:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        header_size = int.from_bytes(_pread_exact(descriptor, 8, 0), "little")
        header = json.loads(_pread_exact(descriptor, header_size, 8))
    finally:
        os.close(descriptor)

    data_start = 8 + header_size
    return {
        name: data_start + value["data_offsets"][0]
        for name, value in header.items()
        if name != "__metadata__"
    }


def _load_local_tensor(
    descriptor: int,
    offset: int,
    spec: WeightSpec,
    tp_rank: int,
    device: str | int,
    torch: object,
    tp_size: int = TP_SIZE,
) -> object:
    data = _read_local_bytes(descriptor, offset, spec, tp_rank % tp_size, tp_size)
    dtype = {
        "BF16": torch.bfloat16,
        "F32": torch.float32,
        "F8_E4M3": torch.float8_e4m3fn,
    }[spec.dtype]
    tensor = torch.frombuffer(data, dtype=dtype).reshape(spec.local_shape(tp_size))
    return tensor.to(device=device, copy=True)


def _read_local_bytes(
    descriptor: int,
    offset: int,
    spec: WeightSpec,
    tp_rank: int,
    tp_size: int = TP_SIZE,
) -> bytearray:
    item_size = {"BF16": 2, "F32": 4, "F8_E4M3": 1}[spec.dtype]
    local_shape = spec.local_shape(tp_size)
    local_bytes = prod(local_shape) * item_size

    if spec.shard_axis is None:
        return _pread_exact(descriptor, local_bytes, offset)
    if spec.shard_axis == 0:
        return _pread_exact(
            descriptor,
            local_bytes,
            offset + tp_rank * local_bytes,
        )
    if spec.shard_axis != 1 or len(spec.shape) != 2:
        raise ValueError(f"unsupported shard axis for {spec.name}: {spec.shard_axis}")

    rows, columns = spec.shape
    local_columns = columns // tp_size
    row_bytes = local_columns * item_size
    full_row_bytes = columns * item_size
    output = bytearray(local_bytes)
    for row in range(rows):
        source = offset + row * full_row_bytes + tp_rank * row_bytes
        start = row * row_bytes
        output[start : start + row_bytes] = _pread_exact(
            descriptor,
            row_bytes,
            source,
        )
    return output


def _pread_exact(descriptor: int, size: int, offset: int) -> bytearray:
    output = bytearray()
    while len(output) < size:
        chunk = os.pread(descriptor, size - len(output), offset + len(output))
        if not chunk:
            raise EOFError(f"short pread at byte {offset + len(output)}")
        output.extend(chunk)
    return output

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from math import prod
from pathlib import Path
from types import MappingProxyType

from infer.models.deepseek_v4_flash import MODEL_ID, MODEL_REVISION
from infer.models.deepseek_v4_flash import model as deepseek_v4_flash

CONFIG_FILE = "config.json"
CONFIG_PIN = (
    1_888,
    "6c8f3d2d3b48707541b88f32f22ef3f0f8a6b57d8523281e2b8d3cdb0ae9a023",
)
SAFETENSORS_INDEX = "model.safetensors.index.json"
SAFETENSORS_INDEX_PIN = (
    5_602_871,
    "98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b",
)

_ROOT_TARGET_SPECS = {
    spec.name: spec
    for spec in (
        deepseek_v4_flash.WeightSpec(
            "embed.weight",
            (deepseek_v4_flash.VOCAB_SIZE, deepseek_v4_flash.HIDDEN_SIZE),
            "BF16",
            0,
        ),
        deepseek_v4_flash.WeightSpec("hc_head_base", (deepseek_v4_flash.MHC_STREAMS,), "F32"),
        deepseek_v4_flash.WeightSpec(
            "hc_head_fn",
            (
                deepseek_v4_flash.MHC_STREAMS,
                deepseek_v4_flash.MHC_STREAMS * deepseek_v4_flash.HIDDEN_SIZE,
            ),
            "F32",
        ),
        deepseek_v4_flash.WeightSpec("hc_head_scale", (1,), "F32"),
        deepseek_v4_flash.WeightSpec(
            "head.weight",
            (deepseek_v4_flash.VOCAB_SIZE, deepseek_v4_flash.HIDDEN_SIZE),
            "BF16",
            0,
        ),
        deepseek_v4_flash.WeightSpec("norm.weight", (deepseek_v4_flash.HIDDEN_SIZE,), "BF16"),
    )
}
_BLOCK_NAMES = (
    "ffn.gate.weight",
    "ffn.shared_experts.w1.scale",
    "ffn.shared_experts.w1.weight",
    "ffn.shared_experts.w2.scale",
    "ffn.shared_experts.w2.weight",
    "ffn.shared_experts.w3.scale",
    "ffn.shared_experts.w3.weight",
    "ffn_norm.weight",
    "hc_attn_base",
    "hc_attn_fn",
    "hc_attn_scale",
    "hc_ffn_base",
    "hc_ffn_fn",
    "hc_ffn_scale",
)
_BLOCK_SPECS = {
    spec.name: spec
    for spec in deepseek_v4_flash.LAYER0_NON_EXPERT_WEIGHTS
    if spec.name in (*_BLOCK_NAMES, "ffn.gate.tid2eid")
}
_GATE_BIAS_SPEC = deepseek_v4_flash.WeightSpec(
    "ffn.gate.bias", (deepseek_v4_flash.NUM_ROUTED_EXPERTS,), "F32"
)
_ROUTED_EXPERT_SPECS = {spec.name: spec for spec in deepseek_v4_flash.ROUTED_EXPERT_TEMPLATE}
_DSPARK_SPECIAL_SPECS = {
    spec.name: spec
    for spec in (
        *deepseek_v4_flash.DSPARK_STAGE0_WEIGHTS,
        *deepseek_v4_flash.DSPARK_ENDPOINT_WEIGHTS,
    )
}
_DTYPE_BYTES = {
    "BF16": 2,
    "F32": 4,
    "F8_E4M3": 1,
    "F8_E8M0": 1,
    "I8": 1,
    "I64": 8,
}


@dataclass(frozen=True, slots=True)
class _SafetensorsShard:
    header: Mapping[str, object]
    offsets: Mapping[str, int]
    storage: object


@dataclass(frozen=True, slots=True)
class DeepSeekV4CheckpointView:
    root: Path
    target_weight_map: Mapping[str, str]
    dspark_weight_map: Mapping[str, str]
    _shards: dict[str, _SafetensorsShard] = field(
        default_factory=dict, compare=False, repr=False
    )

    def load_target_tensor(
        self,
        name: str,
        tep_rank: int,
        device: str | int,
        *,
        sharded: bool,
    ) -> object:
        import torch

        deepseek_v4_flash.validate_tep_rank(tep_rank, deepseek_v4_flash.TEP_SIZE)
        try:
            filename = self.target_weight_map[name]
        except KeyError as error:
            raise KeyError(f"unknown DeepSeek V4 target tensor {name}") from error
        shard = self._shards.get(filename)
        if shard is None:
            shard = _open_target_shard(self, filename, torch)
            self._shards[filename] = shard
        return _load_local_tensor(
            shard.storage,
            shard.offsets[name],
            _target_spec(name),
            tep_rank,
            device,
            torch,
            sharded=sharded,
        )

    def load_dspark_tensor(
        self,
        name: str,
        ep_rank: int,
        device: str | int,
    ) -> object:
        import torch

        deepseek_v4_flash.validate_tep_rank(ep_rank, deepseek_v4_flash.EP_SIZE)
        try:
            filename = self.dspark_weight_map[name]
        except KeyError as error:
            raise KeyError(f"unknown DeepSeek V4 DSpark tensor {name}") from error
        shard = self._shards.get(filename)
        if shard is None:
            shard = _open_dspark_shard(self, filename, torch)
            self._shards[filename] = shard
        return _load_local_tensor(
            shard.storage,
            shard.offsets[name],
            _dspark_spec(name),
            ep_rank,
            device,
            torch,
            sharded=False,
        )


def open_deepseek_v4_checkpoint(
    checkpoint_dir: str | Path | None = None,
    *,
    repo_id: str | None = None,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
) -> DeepSeekV4CheckpointView:
    if repo_id is not None and (repo_id != MODEL_ID or revision != MODEL_REVISION):
        raise ValueError("DeepSeek V4 requires the pinned model and revision")

    if checkpoint_dir is not None:
        root = _resolve_model_source(
            checkpoint_dir,
            repo_id=repo_id,
            revision=revision,
            cache_dir=cache_dir,
        )
    else:
        root = _resolve_model_source(
            repo_id=repo_id,
            revision=revision,
            cache_dir=cache_dir,
        )
    return _read_checkpoint_view(root)


def validate_deepseek_v4_checkpoint_inventory(
    weight_map: Mapping[str, str],
) -> None:
    if not isinstance(weight_map, Mapping) or any(
        not isinstance(name, str) or not isinstance(shard, str)
        for name, shard in weight_map.items()
    ):
        raise TypeError("checkpoint weight_map must map strings to strings")

    target = _expected_target_weight_map()
    dspark = _expected_dspark_weight_map()
    expected = {**target, **dspark}
    actual_names = weight_map.keys()
    missing_target = target.keys() - actual_names
    missing_dspark = dspark.keys() - actual_names
    unclassified = actual_names - expected.keys()
    wrong_shards = {
        name
        for name in actual_names & expected.keys()
        if weight_map[name] != expected[name]
    }

    errors = []
    if missing_target:
        errors.append(f"missing {len(missing_target)} target tensors")
    if missing_dspark:
        errors.append(f"missing {len(missing_dspark)} DSpark tensors")
    if unclassified:
        errors.append(
            f"found {len(unclassified)} unclassified tensors, first {min(unclassified)}"
        )
    if wrong_shards:
        first = min(wrong_shards)
        errors.append(
            f"{len(wrong_shards)} tensors map to the wrong shard, first {first}: "
            f"{weight_map[first]} != {expected[first]}"
        )
    if errors:
        raise ValueError(
            "invalid DeepSeek V4 checkpoint inventory:\n" + "\n".join(errors)
        )


def _read_checkpoint_view(root: Path) -> DeepSeekV4CheckpointView:
    _read_pinned_json(root / CONFIG_FILE, CONFIG_PIN)
    weight_map = _read_pinned_weight_map(root)
    validate_deepseek_v4_checkpoint_inventory(weight_map)

    target = _expected_target_weight_map()
    dspark = _expected_dspark_weight_map()
    return DeepSeekV4CheckpointView(
        root=root,
        target_weight_map=MappingProxyType(
            {name: weight_map[name] for name in sorted(target)}
        ),
        dspark_weight_map=MappingProxyType(
            {name: weight_map[name] for name in sorted(dspark)}
        ),
    )


def _resolve_model_source(*args: object, **kwargs: object) -> Path:
    from infer.model_source import resolve_model_source

    return resolve_model_source(*args, **kwargs)


def _expected_target_weight_map() -> dict[str, str]:
    endpoint_shard = _shard_name(45)
    weight_map = dict.fromkeys(_ROOT_TARGET_SPECS, endpoint_shard)
    weight_map["embed.weight"] = _shard_name(1)
    for layer_id in range(deepseek_v4_flash.NUM_LAYERS):
        shard = _shard_name(layer_id + 2)
        weight_map.update(
            dict.fromkeys(
                (
                    f"layers.{layer_id}.{name}"
                    for name in _block_weight_names(
                        layer_id=layer_id,
                        dspark=False,
                    )
                ),
                shard,
            )
        )
    return weight_map


def _expected_dspark_weight_map() -> dict[str, str]:
    weight_map = {}
    for stage_id in range(deepseek_v4_flash.DSPARK_STAGE_COUNT):
        names = _block_weight_names(layer_id=stage_id, dspark=True)
        if stage_id == 0:
            names.update(spec.name for spec in deepseek_v4_flash.DSPARK_STAGE0_WEIGHTS)
        if stage_id == deepseek_v4_flash.DSPARK_STAGE_COUNT - 1:
            names.update(spec.name for spec in deepseek_v4_flash.DSPARK_ENDPOINT_WEIGHTS)
        weight_map.update(
            dict.fromkeys(
                (f"mtp.{stage_id}.{name}" for name in names),
                _shard_name(stage_id + 46),
            )
        )
    return weight_map


def _block_weight_names(*, layer_id: int, dspark: bool) -> set[str]:
    if dspark or layer_id < 2:
        attention = deepseek_v4_flash.COMPRESSED_ATTENTION_COMMON_WEIGHTS
    else:
        attention = deepseek_v4_flash.compressed_attention_weights(layer_id)
    names = {spec.name for spec in attention}
    names.update(_BLOCK_NAMES)
    names.add("ffn.gate.bias" if dspark or layer_id >= 3 else "ffn.gate.tid2eid")
    names.update(
        f"ffn.experts.{expert_id}.{spec.name}"
        for expert_id in range(deepseek_v4_flash.NUM_ROUTED_EXPERTS)
        for spec in deepseek_v4_flash.ROUTED_EXPERT_TEMPLATE
    )
    return names


def _target_spec(name: str) -> deepseek_v4_flash.WeightSpec:
    root_spec = _ROOT_TARGET_SPECS.get(name)
    if root_spec is not None:
        return root_spec
    _, layer_text, local_name = name.split(".", 2)
    layer_id = int(layer_text)
    if local_name.startswith("ffn.experts."):
        return _ROUTED_EXPERT_SPECS[local_name.split(".", 3)[3]]

    if local_name == "ffn.gate.tid2eid" and layer_id < 3:
        return _BLOCK_SPECS[local_name]
    if local_name == "ffn.gate.bias" and layer_id >= 3:
        return _GATE_BIAS_SPEC
    spec = _BLOCK_SPECS.get(local_name)
    if spec is not None:
        return spec
    attention = (
        deepseek_v4_flash.COMPRESSED_ATTENTION_COMMON_WEIGHTS
        if layer_id < 2
        else deepseek_v4_flash.compressed_attention_weights(layer_id)
    )
    return next(spec for spec in attention if spec.name == local_name)


def _dspark_spec(name: str) -> deepseek_v4_flash.WeightSpec:
    namespace, stage_text, local_name = name.split(".", 2)
    stage_id = int(stage_text)
    if namespace != "mtp" or not 0 <= stage_id < deepseek_v4_flash.DSPARK_STAGE_COUNT:
        raise KeyError(f"unknown DeepSeek V4 DSpark tensor {name}")
    special = _DSPARK_SPECIAL_SPECS.get(local_name)
    if special is not None:
        return special
    if local_name.startswith("ffn.experts."):
        return _ROUTED_EXPERT_SPECS[local_name.split(".", 3)[3]]
    if local_name == "ffn.gate.bias":
        return _GATE_BIAS_SPEC
    spec = _BLOCK_SPECS.get(local_name)
    if spec is not None:
        return spec
    for spec in deepseek_v4_flash.COMPRESSED_ATTENTION_COMMON_WEIGHTS:
        if spec.name == local_name:
            return spec
    raise KeyError(f"unknown DeepSeek V4 DSpark tensor {name}")


def _read_pinned_json(
    path: Path,
    pin: tuple[int, str],
) -> Mapping[str, object]:
    payload = path.read_bytes()
    expected_size, expected_sha256 = pin
    if len(payload) != expected_size or hashlib.sha256(payload).hexdigest() != (
        expected_sha256
    ):
        raise ValueError(f"invalid {path.name} size/SHA-256")
    value = json.loads(payload)
    if not isinstance(value, Mapping):
        raise TypeError(f"{path.name} must contain a JSON object")
    return value


def _read_pinned_weight_map(root: Path) -> Mapping[str, str]:
    index = _read_pinned_json(root / SAFETENSORS_INDEX, SAFETENSORS_INDEX_PIN)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, Mapping):
        raise TypeError("checkpoint index must contain a weight_map")
    return weight_map


def _open_target_shard(
    view: DeepSeekV4CheckpointView,
    filename: str,
    torch: object,
) -> _SafetensorsShard:
    return _open_checkpoint_shard(
        view.root,
        filename,
        view.target_weight_map,
        _target_spec,
        "target",
        torch,
    )


def _open_dspark_shard(
    view: DeepSeekV4CheckpointView,
    filename: str,
    torch: object,
) -> _SafetensorsShard:
    return _open_checkpoint_shard(
        view.root,
        filename,
        view.dspark_weight_map,
        _dspark_spec,
        "DSpark",
        torch,
    )


def _open_checkpoint_shard(
    root: Path,
    filename: str,
    weight_map: Mapping[str, str],
    spec_for_name: Callable[[str], deepseek_v4_flash.WeightSpec],
    label: str,
    torch: object,
) -> _SafetensorsShard:
    expected = {
        name: spec_for_name(name)
        for name, shard in weight_map.items()
        if shard == filename
    }
    if not expected:
        raise ValueError(f"{filename} is not a DeepSeek V4 {label} shard")

    path = root / filename
    size = path.stat().st_size
    with path.open("rb") as shard:
        prefix = shard.read(8)
        if len(prefix) != 8:
            raise ValueError(f"invalid {filename} safetensors header")
        header_size = int.from_bytes(prefix, "little")
        if header_size > size - 8:
            raise ValueError(f"invalid {filename} safetensors header")
        payload = shard.read(header_size)
    if len(payload) != header_size:
        raise ValueError(f"invalid {filename} safetensors header")
    try:
        header = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {filename} safetensors header") from error
    if (
        not isinstance(header, dict)
        or set(header) - {"__metadata__"} != expected.keys()
    ):
        raise ValueError(f"invalid {filename} {label} tensor inventory")

    ranges = []
    for name, spec in expected.items():
        record = header[name]
        shape = record.get("shape") if isinstance(record, Mapping) else None
        if (
            not isinstance(record, Mapping)
            or record.get("dtype") != spec.dtype
            or not isinstance(shape, list)
            or any(type(dimension) is not int for dimension in shape)
            or tuple(shape) != spec.shape
        ):
            raise ValueError(f"invalid {filename} dtype/shape for {name}")
        data_offsets = record.get("data_offsets")
        if (
            not isinstance(data_offsets, list)
            or len(data_offsets) != 2
            or any(type(offset) is not int for offset in data_offsets)
        ):
            raise ValueError(f"invalid {filename} offsets for {name}")
        start, end = data_offsets
        if start < 0 or end - start != prod(spec.shape) * _DTYPE_BYTES[spec.dtype]:
            raise ValueError(f"invalid {filename} offsets for {name}")
        ranges.append((start, end, name))

    data_size = size - 8 - header_size
    cursor = 0
    offsets = {}
    for start, end, name in sorted(ranges):
        if start != cursor or end > data_size:
            raise ValueError(f"invalid {filename} tensor data layout")
        offsets[name] = 8 + header_size + start
        cursor = end
    if cursor != data_size:
        raise ValueError(f"invalid {filename} tensor data layout")
    storage = torch.from_file(str(path), shared=False, size=size, dtype=torch.uint8)
    return _SafetensorsShard(header, offsets, storage)


def _load_local_tensor(
    storage: object,
    offset: int,
    spec: deepseek_v4_flash.WeightSpec,
    tep_rank: int,
    device: str | int,
    torch: object,
    *,
    sharded: bool = True,
) -> object:
    dtype = {
        "BF16": torch.bfloat16,
        "F32": torch.float32,
        "F8_E4M3": torch.float8_e4m3fn,
        "F8_E8M0": torch.float8_e8m0fnu,
        "I8": torch.int8,
        "I64": torch.int64,
    }[spec.dtype]
    data = storage[offset : offset + prod(spec.shape) * _DTYPE_BYTES[spec.dtype]]
    tensor = data.view(dtype).reshape(spec.shape)
    if sharded and spec.shard_axis is not None:
        tensor = tensor.chunk(deepseek_v4_flash.TEP_SIZE, dim=spec.shard_axis)[tep_rank]
    return tensor.contiguous().to(device=device, copy=True)


def _shard_name(index: int) -> str:
    return f"model-{index:05d}-of-00048.safetensors"

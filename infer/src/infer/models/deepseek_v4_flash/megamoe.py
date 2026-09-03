from collections.abc import Callable
from dataclasses import dataclass

from infer.models.deepseek_v4_flash import checkpoint
from infer.models.deepseek_v4_flash import model as deepseek_v4_flash

_MOE_NON_EXPERT_TEMPLATE = tuple(
    spec
    for spec in deepseek_v4_flash.LAYER0_NON_EXPERT_WEIGHTS
    if spec.name == "ffn.gate.weight" or spec.name.startswith("ffn.shared_experts.")
)
_HASH_ROUTING_SPEC = next(
    spec
    for spec in deepseek_v4_flash.LAYER0_NON_EXPERT_WEIGHTS
    if spec.name == "ffn.gate.tid2eid"
)
_LEARNED_ROUTING_SPEC = deepseek_v4_flash.WeightSpec(
    "ffn.gate.bias", (deepseek_v4_flash.NUM_ROUTED_EXPERTS,), "F32"
)


@dataclass(frozen=True, slots=True)
class DeepSeekV4MegaMoEWeights[TensorT]:
    l1: tuple[TensorT, TensorT]
    l2: tuple[TensorT, TensorT]


@dataclass(frozen=True, slots=True)
class DeepSeekV4MoEWeights[TensorT]:
    router_t: TensorT
    routing: TensorT
    shared_gate_up: TensorT
    shared_gate_up_scale: TensorT
    shared_down: TensorT
    shared_down_scale: TensorT
    routed: DeepSeekV4MegaMoEWeights[TensorT]


@dataclass(frozen=True, slots=True)
class DeepSeekV4MoEWorkspace[TensorT]:
    router_logits: TensorT
    topk_ids: TensorT
    topk_weights: TensorT
    hidden_fp8: TensorT
    hidden_scale: TensorT
    shared_gate_up: TensorT
    shared_activated_fp8: TensorT
    shared_activated_scale: TensorT
    routed: TensorT
    output: TensorT


def moe_workspace_shapes(
    token_count: int,
) -> DeepSeekV4MoEWorkspace[deepseek_v4_flash.Shape]:
    return _moe_workspace_shapes(token_count, deepseek_v4_flash.SHARED_INTERMEDIATE_SIZE)


def tep4_moe_workspace_shapes(
    token_count: int,
) -> DeepSeekV4MoEWorkspace[deepseek_v4_flash.Shape]:
    return _moe_workspace_shapes(
        token_count, deepseek_v4_flash.LOCAL_SHARED_INTERMEDIATE_SIZE
    )


def _moe_workspace_shapes(
    token_count: int, shared: int
) -> DeepSeekV4MoEWorkspace[deepseek_v4_flash.Shape]:
    if token_count < 0:
        raise ValueError("token_count must be non-negative")
    hidden = deepseek_v4_flash.HIDDEN_SIZE
    return DeepSeekV4MoEWorkspace(
        router_logits=(token_count, deepseek_v4_flash.NUM_ROUTED_EXPERTS),
        topk_ids=(token_count, deepseek_v4_flash.TOP_K),
        topk_weights=(token_count, deepseek_v4_flash.TOP_K),
        hidden_fp8=(token_count, hidden),
        hidden_scale=(token_count, hidden // deepseek_v4_flash.FP8_BLOCK_SIZE),
        shared_gate_up=(token_count, 2 * shared),
        shared_activated_fp8=(token_count, shared),
        shared_activated_scale=(
            token_count,
            shared // deepseek_v4_flash.FP8_BLOCK_SIZE,
        ),
        routed=(token_count, hidden),
        output=(token_count, hidden),
    )


def load_dep4_moe_weights(
    view: checkpoint.DeepSeekV4CheckpointView,
    layer_id: int,
    ep_rank: int,
    device: str | int,
) -> DeepSeekV4MoEWeights[object]:
    return _load_target_moe_weights(
        view, layer_id, ep_rank, device, tensor_parallel=False
    )


def load_tep4_moe_weights(
    view: checkpoint.DeepSeekV4CheckpointView,
    layer_id: int,
    rank: int,
    device: str | int,
) -> DeepSeekV4MoEWeights[object]:
    return _load_target_moe_weights(view, layer_id, rank, device, tensor_parallel=True)


def _load_target_moe_weights(
    view: checkpoint.DeepSeekV4CheckpointView,
    layer_id: int,
    rank: int,
    device: str | int,
    *,
    tensor_parallel: bool,
) -> DeepSeekV4MoEWeights[object]:
    import deep_gemm
    import torch

    local_routed_expert_ids(rank)

    def load(spec: deepseek_v4_flash.WeightSpec, expert_id: int | None) -> object:
        key = (
            spec.checkpoint_key_for_layer(layer_id)
            if expert_id is None
            else f"layers.{layer_id}.ffn.experts.{expert_id}.{spec.name}"
        )
        return view.load_target_tensor(
            key,
            rank,
            device,
            sharded=tensor_parallel,
        )

    return _pack_moe_weights(
        load,
        rank,
        device,
        torch,
        deep_gemm,
        hash_routing=deepseek_v4_flash.uses_hash_routing(layer_id),
        tensor_parallel=tensor_parallel,
    )


def load_dep4_dspark_moe_weights(
    view: checkpoint.DeepSeekV4CheckpointView,
    stage_id: int,
    ep_rank: int,
    device: str | int,
) -> DeepSeekV4MoEWeights[object]:
    import deep_gemm
    import torch

    if not 0 <= stage_id < deepseek_v4_flash.DSPARK_STAGE_COUNT:
        raise ValueError("DSpark stage_id is out of range")
    local_routed_expert_ids(ep_rank)

    def load(spec: deepseek_v4_flash.WeightSpec, expert_id: int | None) -> object:
        name = (
            f"mtp.{stage_id}.{spec.name}"
            if expert_id is None
            else f"mtp.{stage_id}.ffn.experts.{expert_id}.{spec.name}"
        )
        return view.load_dspark_tensor(name, ep_rank, device)

    return _pack_moe_weights(
        load,
        ep_rank,
        device,
        torch,
        deep_gemm,
        hash_routing=False,
        tensor_parallel=False,
    )


def local_routed_expert_ids(ep_rank: int) -> tuple[int, ...]:
    if not 0 <= ep_rank < deepseek_v4_flash.EP_SIZE:
        raise ValueError(f"ep_rank must be in [0, {deepseek_v4_flash.EP_SIZE})")
    start = ep_rank * deepseek_v4_flash.LOCAL_EXPERTS
    return tuple(range(start, start + deepseek_v4_flash.LOCAL_EXPERTS))


def _pack_megamoe_weights(
    load: Callable[[deepseek_v4_flash.WeightSpec, int], object],
    ep_rank: int,
    device: str | int,
    torch: object,
    deep_gemm: object,
) -> DeepSeekV4MegaMoEWeights[object]:
    hidden = deepseek_v4_flash.HIDDEN_SIZE
    intermediate = deepseek_v4_flash.EXPERT_INTERMEDIATE_SIZE
    local_experts = deepseek_v4_flash.LOCAL_EXPERTS
    specs = {spec.name: spec for spec in deepseek_v4_flash.ROUTED_EXPERT_TEMPLATE}

    l1_data = torch.empty(
        (local_experts, 2 * intermediate, hidden // 2),
        dtype=torch.int8,
        device=device,
    )
    l2_data = torch.empty(
        (local_experts, hidden, intermediate // 2),
        dtype=torch.int8,
        device=device,
    )
    l1_scale = torch.empty_strided(
        (local_experts, 2 * intermediate, hidden // deepseek_v4_flash.FP8_BLOCK_SIZE),
        (
            2 * intermediate * (hidden // deepseek_v4_flash.FP8_BLOCK_SIZE),
            1,
            2 * intermediate,
        ),
        dtype=torch.int32,
        device=device,
    )
    l2_scale = torch.empty_strided(
        (local_experts, hidden, intermediate // deepseek_v4_flash.FP8_BLOCK_SIZE),
        (
            hidden * (intermediate // deepseek_v4_flash.FP8_BLOCK_SIZE),
            1,
            hidden,
        ),
        dtype=torch.int32,
        device=device,
    )
    targets = (
        (l1_data[:, :intermediate], "w1.weight", False),
        (l1_data[:, intermediate:], "w3.weight", False),
        (l2_data, "w2.weight", False),
        (l1_scale[:, :intermediate], "w1.scale", True),
        (l1_scale[:, intermediate:], "w3.scale", True),
        (l2_scale, "w2.scale", True),
    )

    expert_ids = local_routed_expert_ids(ep_rank)
    for target, name, packed_scale in targets:
        spec = specs[name]
        for local_index, expert_id in enumerate(expert_ids):
            source = load(spec, expert_id)
            if packed_scale:
                _copy_packed_scales(target[local_index], source, torch)
            else:
                target[local_index].copy_(source)

    l1, l2 = deep_gemm.transform_weights_for_mega_moe(
        (l1_data, l1_scale),
        (l2_data, l2_scale),
    )
    _validate_transformed_weights(l1, l2, l1_data, l1_scale, l2_data, l2_scale)
    return DeepSeekV4MegaMoEWeights(l1=l1, l2=l2)


def _pack_moe_weights(
    load: Callable[[deepseek_v4_flash.WeightSpec, int | None], object],
    ep_rank: int,
    device: str | int,
    torch: object,
    deep_gemm: object,
    *,
    hash_routing: bool,
    tensor_parallel: bool,
) -> DeepSeekV4MoEWeights[object]:
    loaded_device = None

    def take(spec: deepseek_v4_flash.WeightSpec, expert_id: int | None = None) -> object:
        nonlocal loaded_device
        tensor = load(spec, expert_id)
        if loaded_device is None:
            loaded_device = tensor.device
        _validate_source_tensor(
            tensor, spec, loaded_device, torch, tensor_parallel=tensor_parallel
        )
        return tensor

    routing_spec = _HASH_ROUTING_SPEC if hash_routing else _LEARNED_ROUTING_SPEC
    raw = {spec.name: take(spec) for spec in (*_MOE_NON_EXPERT_TEMPLATE, routing_spec)}
    routed = _pack_megamoe_weights(take, ep_rank, device, torch, deep_gemm)

    routing = raw[routing_spec.name]
    if hash_routing:
        _validate_hash_routes_tensor(routing, torch)
        routing = routing.to(torch.int32)
    return DeepSeekV4MoEWeights(
        router_t=raw["ffn.gate.weight"].T.contiguous(),
        routing=routing,
        shared_gate_up=torch.cat(
            (
                raw["ffn.shared_experts.w1.weight"],
                raw["ffn.shared_experts.w3.weight"],
            )
        ).contiguous(),
        shared_gate_up_scale=torch.cat(
            (
                raw["ffn.shared_experts.w1.scale"],
                raw["ffn.shared_experts.w3.scale"],
            )
        )
        .to(torch.float32)
        .contiguous(),
        shared_down=raw["ffn.shared_experts.w2.weight"],
        shared_down_scale=raw["ffn.shared_experts.w2.scale"]
        .to(torch.float32)
        .contiguous(),
        routed=routed,
    )


def _copy_packed_scales(target: object, source: object, torch: object) -> None:
    target.copy_(source.contiguous().view(torch.uint8).view(torch.int32))


def _validate_hash_routes_tensor(routes: object, torch: object) -> None:
    if routes.shape != (deepseek_v4_flash.VOCAB_SIZE, deepseek_v4_flash.TOP_K) or (
        routes.dtype != torch.int64
    ):
        raise ValueError("hash routes have the wrong shape or dtype")
    ordered = routes.sort(dim=1).values
    if bool(
        ((ordered[:, 0] < 0) | (ordered[:, -1] >= deepseek_v4_flash.NUM_ROUTED_EXPERTS)).any()
    ):
        raise ValueError("hash routes contain an out-of-range expert ID")
    if bool((ordered[:, 1:] == ordered[:, :-1]).any()):
        raise ValueError("hash routes repeat an expert ID")


def _validate_source_tensor(
    tensor: object,
    spec: deepseek_v4_flash.WeightSpec,
    device: object,
    torch: object,
    *,
    tensor_parallel: bool,
) -> None:
    dtype = {
        "BF16": torch.bfloat16,
        "F32": torch.float32,
        "F8_E4M3": torch.float8_e4m3fn,
        "F8_E8M0": torch.float8_e8m0fnu,
        "I8": torch.int8,
        "I64": torch.int64,
    }[spec.dtype]
    expected_shape = spec.local_shape() if tensor_parallel else spec.shape
    if tuple(tensor.shape) != expected_shape or tensor.dtype != dtype:
        raise ValueError(
            f"{spec.name} tensor has {tuple(tensor.shape)}/{tensor.dtype}, "
            f"expected {expected_shape}/{dtype}"
        )
    if tensor.device != device:
        raise ValueError(f"{spec.name} tensor is on {tensor.device}, expected {device}")


def _validate_transformed_weights(
    l1: tuple[object, object],
    l2: tuple[object, object],
    l1_data: object,
    l1_scale: object,
    l2_data: object,
    l2_scale: object,
) -> None:
    expected = (l1_data, l1_scale, l2_data, l2_scale)
    actual = (*l1, *l2)
    if any(
        tuple(found.shape) != tuple(wanted.shape)
        or found.dtype != wanted.dtype
        or tuple(found.stride()) != tuple(wanted.stride())
        for found, wanted in zip(actual, expected, strict=True)
    ):
        raise RuntimeError("transformed MegaMoE weight ABI changed")

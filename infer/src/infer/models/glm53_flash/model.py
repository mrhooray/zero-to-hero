from dataclasses import dataclass
from typing import Literal

DENSE_LAYER_IDS = (0, 1, 2)
NEXTN_LAYER_ID = 45
TP_SIZE = 4
HIDDEN_SIZE = 4096
VOCAB_SIZE = 154_880
TOKENIZER_VOCAB_SIZE = 154_856
LOCAL_VOCAB_SIZE = VOCAB_SIZE // TP_SIZE
NUM_HEADS = 64
HEAD_DIM = 128
PROJECTION_SIZE = NUM_HEADS * HEAD_DIM
CONV_KERNEL_SIZE = 4
KDA_CHUNK_SIZE = 8192
FLASH_KDA_CHUNK_SIZE = 16
FLASH_KDA_WORKSPACE_ALIGNMENT = 128
FLASH_KDA_WORKSPACE_TILE_BYTES = 13_824
GATE_LOWER_BOUND = -5.0
RMS_NORM_EPS = 1e-5
MHC_STREAMS = 4
MHC_EPS = 1e-6
MHC_NORMALIZATION_ITERS = 20
MHC_PRE_MAX_SPLITS = 8
FP8_BLOCK_SIZE = 128

MAIN_SPARSE_MLA_LAYER_IDS = (3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43)
SPARSE_MLA_Q_LORA_RANK = 1536
SPARSE_MLA_KV_LORA_RANK = 512
SPARSE_MLA_QK_NOPE_HEAD_DIM = 256
SPARSE_MLA_VALUE_HEAD_DIM = 256
SPARSE_MLA_INDEXER_HEADS = 32
SPARSE_MLA_INDEXER_HEAD_DIM = 128
SPARSE_MLA_INDEX_POOL_TOKENS = 4
SPARSE_MLA_LAYER_NORM_EPS = 1e-6
SPARSE_MLA_INDEX_SCORE_SCALE = SPARSE_MLA_INDEXER_HEAD_DIM**-0.5
SPARSE_MLA_INDEX_WEIGHT_SCALE = SPARSE_MLA_INDEXER_HEADS**-0.5
GLM53_TARGET_DECODE_BATCH_SIZES = (1, 2, 4, 8, 16, 32, 64)
GLM53_DEP4_DECODE_BATCH_SIZES = (1, 2, 4, 8, 16, 32)
GLM53_DEP4_TARGET_ONLY_DECODE_BATCH_SIZES = GLM53_DEP4_DECODE_BATCH_SIZES
GLM53_TARGET_VERIFY_WIDTH = 4
SPARSE_MLA_HISTORY_BLOCK_TOKENS = 128
SPARSE_MLA_PROJECTION_BATCH_SIZES = (
    *GLM53_TARGET_DECODE_BATCH_SIZES,
    GLM53_TARGET_VERIFY_WIDTH,
    SPARSE_MLA_HISTORY_BLOCK_TOKENS,
    KDA_CHUNK_SIZE,
)
SPARSE_MLA_LATENT_PAGE_TOKENS = 64
SPARSE_MLA_MAX_CONTEXT_TOKENS = 1_048_576
SPARSE_MLA_INDEX_TOP_K = 512
SPARSE_MLA_SPARSE_CAPACITY = 4 * (SPARSE_MLA_INDEX_TOP_K + 1)
SPARSE_MLA_COMPOUND_BLOCKS = SPARSE_MLA_MAX_CONTEXT_TOKENS // 128
SPARSE_MLA_INDEX_CONTEXT_CAPACITY = SPARSE_MLA_MAX_CONTEXT_TOKENS // 4
SPARSE_MLA_INDEX_PAGE_BYTES = 32 * (SPARSE_MLA_INDEXER_HEAD_DIM + 4)
SPARSE_MLA_TOPK_ROW_STATES_BYTES = 1_048_576
SPARSE_MLA_FLASHINFER_WORKSPACE_BYTES = 128 * 1024 * 1024
SPARSE_MLA_B200_SMS = 148

SPARSE_FFN_LAYER_IDS = tuple(range(3, 45))
SPARSE_KDA_LAYER_IDS = tuple(
    layer_id
    for layer_id in SPARSE_FFN_LAYER_IDS
    if layer_id not in MAIN_SPARSE_MLA_LAYER_IDS
)
NUM_ROUTED_EXPERTS = 288
TOP_K = 8
MOE_INTERMEDIATE_SIZE = 2048
LOCAL_ROUTED_EXPERTS = NUM_ROUTED_EXPERTS // TP_SIZE
SPARSE_FFN_SHARED_FC1_ORDER = ("gate_proj", "up_proj")
SPARSE_FFN_DECODE_BATCH_SIZES = SPARSE_MLA_PROJECTION_BATCH_SIZES
SPARSE_FFN_ROUTE_SCALE = 2.5
SPARSE_FFN_CLAMP = 10.0

Shape = tuple[int, ...]
ScalarType = Literal["BF16", "F32", "F8_E4M3"]


@dataclass(frozen=True, slots=True)
class KDAWeights[TensorT]:
    projection: TensorT
    gate_projections: TensorT
    conv: TensorT
    a_log: TensorT
    dt_bias: TensorT
    o_norm: TensorT
    o_projection: TensorT


@dataclass(frozen=True, slots=True)
class MHCWeights[TensorT]:
    base: TensorT
    fn: TensorT
    scale: TensorT


@dataclass(frozen=True, slots=True)
class GLM53EndpointWeights[TensorT]:
    embedding: TensorT
    final_norm: TensorT
    lm_head: TensorT


@dataclass(frozen=True, slots=True)
class GLM53DistributedArgmaxWorkspace[TensorT]:
    local_values: TensorT
    local_indices: TensorT
    local_candidates: TensorT
    gathered_candidates: TensorT
    best_values: TensorT
    select: TensorT


@dataclass(frozen=True, slots=True)
class GLM53EndpointWorkspace[TensorT]:
    local_token: TensorT
    local_active: TensorT
    embedding: TensorT
    streams: TensorT
    mean_f32: TensorT
    collapsed: TensorT
    normalized: TensorT
    local_logits: TensorT
    argmax: GLM53DistributedArgmaxWorkspace[TensorT]
    token: TensorT


@dataclass(frozen=True, slots=True)
class DenseFFNWeights[TensorT]:
    gate_up: TensorT
    gate_up_scale_inv: TensorT
    down: TensorT
    down_scale_inv: TensorT


@dataclass(frozen=True, slots=True)
class SparseFFNWeights[TensorT]:
    router: TensorT
    router_t: TensorT
    correction_bias: TensorT
    routed_up_gate: TensorT
    routed_up_gate_scale_inv: TensorT
    routed_down: TensorT
    routed_down_scale_inv: TensorT
    routed_clamp: TensorT
    shared_gate_up: TensorT
    shared_gate_up_scale_inv: TensorT
    shared_down: TensorT
    shared_down_scale_inv: TensorT


@dataclass(frozen=True, slots=True)
class SparseMLAProjectionWeights[TensorT]:
    low_rank: TensorT
    low_rank_scale_inv: TensorT
    q_norm: TensorT
    kv_norm: TensorT
    q_b: TensorT
    q_b_scale_inv: TensorT
    wq_b: TensorT
    index_prep: TensorT
    k_norm: TensorT
    k_bias: TensorT


@dataclass(frozen=True, slots=True)
class SparseMLAProjectionWorkspace[TensorT]:
    """Caller-owned tensors; allocate one independent instance per parity."""

    hidden_fp8: TensorT
    hidden_scale: TensorT
    low_rank: TensorT
    q_resid: TensorT
    latent: TensorT
    q_resid_fp8: TensorT
    q_resid_scale: TensorT
    main_q: TensorT
    index_q: TensorT
    index_prep: TensorT
    key: TensorT
    pool_gate: TensorT
    score_weights: TensorT


@dataclass(frozen=True, slots=True)
class SparseMLADecodeWeights[TensorT]:
    w_kc: TensorT
    pool_ape: TensorT


@dataclass(frozen=True, slots=True)
class SparseMLAOutputWeights[TensorT]:
    w_vc: TensorT
    o_proj: TensorT
    o_proj_scale_inv: TensorT


@dataclass(frozen=True, slots=True)
class GLM53SparseLayerShellWeights[TensorT]:
    attention_mhc: MHCWeights[TensorT]
    input_norm: TensorT
    ffn_mhc: MHCWeights[TensorT]
    post_attention_norm: TensorT


@dataclass(frozen=True, slots=True)
class GLM53SparseKDALayerWeights[TensorT]:
    attention_mhc: MHCWeights[TensorT]
    input_norm: TensorT
    attention: KDAWeights[TensorT]
    ffn_mhc: MHCWeights[TensorT]
    post_attention_norm: TensorT
    ffn: SparseFFNWeights[TensorT]


@dataclass(frozen=True, slots=True)
class GLM53SparseMLALayerWeights[TensorT]:
    attention_mhc: MHCWeights[TensorT]
    input_norm: TensorT
    mla_projection: SparseMLAProjectionWeights[TensorT]
    mla_decode: SparseMLADecodeWeights[TensorT]
    mla_output: SparseMLAOutputWeights[TensorT]
    ffn_mhc: MHCWeights[TensorT]
    post_attention_norm: TensorT
    ffn: SparseFFNWeights[TensorT]


@dataclass(frozen=True, slots=True)
class SparseMLADecodeBatch[TensorT]:
    """Post-append parity; active rows have flag 1, positive length, and unique slots.

    Inactive rows skip writes, attend to a reserved zero history token, and return zero.
    """

    active: TensorT
    raw_lengths: TensorT
    state_slots: TensorT
    block_table: TensorT
    null_token: TensorT


@dataclass(frozen=True, slots=True)
class SparseMLAHistory[TensorT]:
    """BF16 latent history; FP8 index pages are 4096 values then 32 FP32 scales."""

    latent: TensorT
    index_cache: TensorT
    tail_key: TensorT
    tail_gate: TensorT


@dataclass(frozen=True, slots=True)
class SparseMLAPrefixSnapshots[TensorT]:
    """Private final history block and live tail for each cached endpoint."""

    latent: TensorT
    index_cache: TensorT
    tail_key: TensorT
    tail_gate: TensorT

    def capture(
        self,
        snapshot: int,
        history: SparseMLAHistory[TensorT],
        live_slot: int,
        physical_block: int,
    ) -> None:
        page = (physical_block + 1) * len(SPARSE_MLA_BF16_LATENT_PAGES)
        self.latent[snapshot].copy_(
            history.latent[page : page + len(SPARSE_MLA_BF16_LATENT_PAGES)]
        )
        self.index_cache[snapshot].copy_(history.index_cache[physical_block + 1])
        self.tail_key[:, snapshot].copy_(history.tail_key[:, live_slot])
        self.tail_gate[:, snapshot].copy_(history.tail_gate[:, live_slot])

    def restore(
        self,
        snapshot: int,
        history: SparseMLAHistory[TensorT],
        live_slot: int,
        physical_block: int,
    ) -> None:
        page = (physical_block + 1) * len(SPARSE_MLA_BF16_LATENT_PAGES)
        history.latent[page : page + len(SPARSE_MLA_BF16_LATENT_PAGES)].copy_(
            self.latent[snapshot]
        )
        history.index_cache[physical_block + 1].copy_(self.index_cache[snapshot])
        history.tail_key[:, live_slot].copy_(self.tail_key[:, snapshot])
        history.tail_gate[:, live_slot].copy_(self.tail_gate[:, snapshot])


def allocate_sparse_mla_prefix_snapshots(
    torch: object, device: str | int, snapshot_slot_count: int
) -> SparseMLAPrefixSnapshots:
    empty = lambda shape, dtype: torch.empty(shape, dtype=dtype, device=device)
    return SparseMLAPrefixSnapshots(
        latent=empty(
            (
                snapshot_slot_count,
                len(SPARSE_MLA_BF16_LATENT_PAGES),
                *SPARSE_MLA_BF16_LATENT_PAGES[0],
            ),
            torch.bfloat16,
        ),
        index_cache=empty(
            (snapshot_slot_count, SPARSE_MLA_INDEX_PAGE_BYTES), torch.uint8
        ),
        tail_key=empty(
            (2, snapshot_slot_count, *SPARSE_MLA_INDEX_TAIL_K_BF16_SHAPE),
            torch.bfloat16,
        ),
        tail_gate=empty(
            (2, snapshot_slot_count, *SPARSE_MLA_INDEX_TAIL_K_BF16_SHAPE),
            torch.bfloat16,
        ),
    )


@dataclass(frozen=True, slots=True)
class SparseMLADecodeWorkspace[TensorT]:
    """One caller-owned graph-buffer set per parity.

    Top-k offsets, row state, and the FlashInfer counter start zeroed.
    """

    context_lengths: TensorT
    sequence_lengths: TensorT
    index_q: TensorT
    score_weights: TensorT
    main_q_hbd: TensorT
    absorbed_hbd: TensorT
    attention_q: TensorT
    schedule: TensorT
    logits: TensorT
    selected: TensorT
    topk_offsets: TensorT
    topk_rows: TensorT
    sparse_ids: TensorT
    sparse_lengths: TensorT
    flashinfer_workspace: TensorT
    counter: TensorT
    output: TensorT


@dataclass(frozen=True, slots=True)
class SparseMLAOutputWorkspace[TensorT]:
    """Caller-owned sparse-MLA output tensors for one decode graph parity."""

    value_hbd: TensorT
    projected: TensorT
    projected_fp8: TensorT
    projected_scale: TensorT
    output: TensorT


SPARSE_FFN_TEP4_SHAPES = SparseFFNWeights(
    router=(NUM_ROUTED_EXPERTS, HIDDEN_SIZE),
    router_t=(HIDDEN_SIZE, NUM_ROUTED_EXPERTS),
    correction_bias=(NUM_ROUTED_EXPERTS,),
    routed_up_gate=(LOCAL_ROUTED_EXPERTS, 2 * MOE_INTERMEDIATE_SIZE, HIDDEN_SIZE),
    routed_up_gate_scale_inv=(LOCAL_ROUTED_EXPERTS, 32, 32),
    routed_down=(LOCAL_ROUTED_EXPERTS, HIDDEN_SIZE, MOE_INTERMEDIATE_SIZE),
    routed_down_scale_inv=(LOCAL_ROUTED_EXPERTS, 32, 16),
    routed_clamp=(LOCAL_ROUTED_EXPERTS,),
    shared_gate_up=(2 * MOE_INTERMEDIATE_SIZE // TP_SIZE, HIDDEN_SIZE),
    shared_gate_up_scale_inv=(8, 32),
    shared_down=(HIDDEN_SIZE, MOE_INTERMEDIATE_SIZE // TP_SIZE),
    shared_down_scale_inv=(32, 4),
)
@dataclass(frozen=True, slots=True)
class GLM53LayerWeights[TensorT]:
    attention: KDAWeights[TensorT]
    attention_mhc: MHCWeights[TensorT]
    ffn_mhc: MHCWeights[TensorT]
    input_norm: TensorT
    post_attention_norm: TensorT
    ffn: DenseFFNWeights[TensorT]


@dataclass(frozen=True, slots=True)
class KDAState[TensorT]:
    recurrent: TensorT
    conv: TensorT


@dataclass(frozen=True, slots=True)
class KDADecodeWorkspace[TensorT]:
    projection: TensorT
    output_gate: TensorT
    gate_raw: TensorT
    local_output: TensorT
    output: TensorT


@dataclass(frozen=True, slots=True)
class KDAPrefillWorkspace[TensorT]:
    """One fixed arena reused serially by every KDA layer in a worker.

    ``kda_workspace`` is the aligned view into ``kda_workspace_storage``, not
    a second allocation. No other fields alias without a GPU-validated layout.
    """

    projection: TensorT
    gates: TensorT
    beta: TensorT
    qkv: TensorT
    initial_state: TensorT
    kda_output: TensorT
    final_state: TensorT
    kda_workspace_storage: TensorT
    kda_workspace: TensorT
    output: TensorT


@dataclass(frozen=True, slots=True)
class GLM53DenseWorkspace[TensorT]:
    mhc_sqrsum: TensorT
    mhc_dot: TensorT
    post: TensorT
    comb: TensorT
    collapsed: TensorT
    normalized: TensorT
    streams_mid: TensorT
    hidden_fp8: TensorT
    hidden_scale: TensorT
    gate_up: TensorT
    activated: TensorT
    activated_fp8: TensorT
    activated_scale: TensorT
    ffn_output: TensorT
    streams_out: TensorT


@dataclass(frozen=True, slots=True)
class SparseFFNDecodeWorkspace[TensorT]:
    gathered: TensorT
    scattered: TensorT
    scores: TensorT
    selection: TensorT
    topk_values: TensorT
    topk_ids64: TensorT
    topk_ids: TensorT
    hidden_fp8: TensorT
    hidden_scale_mn: TensorT
    routed: TensorT
    shared_gate_up: TensorT
    shared_fp8: TensorT
    shared_scale: TensorT
    output: TensorT


def glm53_attention_tp_size(parallelism: str) -> int:
    if parallelism not in {"dep4", "tep4"}:
        raise ValueError("GLM parallelism must be dep4 or tep4")
    return 1 if parallelism == "dep4" else TP_SIZE


def glm53_decode_batch_sizes(
    attention_tp_size: int, *, speculative: bool = True
) -> tuple[int, ...]:
    if attention_tp_size == 1:
        return (
            GLM53_DEP4_DECODE_BATCH_SIZES
            if speculative
            else GLM53_DEP4_TARGET_ONLY_DECODE_BATCH_SIZES
        )
    if attention_tp_size == TP_SIZE:
        return GLM53_TARGET_DECODE_BATCH_SIZES
    raise ValueError("GLM attention TP size must be one or four")


def glm53_local_heads(attention_tp_size: int) -> int:
    if attention_tp_size not in {1, TP_SIZE}:
        raise ValueError("GLM attention TP size must be one or four")
    return NUM_HEADS // attention_tp_size


SPARSE_MLA_BF16_LATENT_PAGES = (
    (1, SPARSE_MLA_LATENT_PAGE_TOKENS, SPARSE_MLA_KV_LORA_RANK),
    (1, SPARSE_MLA_LATENT_PAGE_TOKENS, SPARSE_MLA_KV_LORA_RANK),
)
SPARSE_MLA_INDEX_TAIL_K_BF16_SHAPE = (
    SPARSE_MLA_INDEX_POOL_TOKENS,
    SPARSE_MLA_INDEXER_HEAD_DIM,
)


def validate_glm53_tep4_world(process_group: object) -> int:
    import torch.distributed as dist

    if not dist.is_initialized():
        raise RuntimeError("GLM requires initialized distributed state")
    if process_group is not dist.group.WORLD:
        raise ValueError("GLM requires the WORLD process group")
    if dist.get_world_size(process_group) != TP_SIZE:
        raise ValueError("GLM requires a TEP4 process group")
    return dist.get_rank(process_group)


def _validate_chunk_batch_size(batch_size: int, label: str) -> None:
    if not 1 <= batch_size <= KDA_CHUNK_SIZE:
        raise ValueError(f"{label} batch_size must be in [1, {KDA_CHUNK_SIZE}]")


def glm53_endpoint_workspace_shapes(
    batch_size: int, attention_tp_size: int = TP_SIZE
) -> GLM53EndpointWorkspace[Shape]:
    _validate_chunk_batch_size(batch_size, "GLM endpoint")
    glm53_local_heads(attention_tp_size)
    local_vocab_size = VOCAB_SIZE // attention_tp_size
    return GLM53EndpointWorkspace(
        local_token=(batch_size,),
        local_active=(batch_size,),
        embedding=(batch_size, HIDDEN_SIZE),
        streams=(batch_size, MHC_STREAMS, HIDDEN_SIZE),
        mean_f32=(batch_size, HIDDEN_SIZE),
        collapsed=(batch_size, HIDDEN_SIZE),
        normalized=(batch_size, HIDDEN_SIZE),
        local_logits=(batch_size, local_vocab_size),
        argmax=GLM53DistributedArgmaxWorkspace(
            local_values=(batch_size,),
            local_indices=(batch_size,),
            local_candidates=(batch_size, 2),
            gathered_candidates=(attention_tp_size * batch_size, 2),
            best_values=(batch_size,),
            select=(batch_size,),
        ),
        token=(batch_size,),
    )


def kda_state_shapes(batch_size: int, tp_size: int = TP_SIZE) -> KDAState[Shape]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if NUM_HEADS % tp_size:
        raise ValueError(f"{NUM_HEADS} heads cannot shard {tp_size} ways")

    local_heads = NUM_HEADS // tp_size
    local_channels = local_heads * HEAD_DIM
    return KDAState(
        recurrent=(batch_size, local_heads, HEAD_DIM, HEAD_DIM),
        conv=(batch_size, 3 * local_channels, CONV_KERNEL_SIZE - 1),
    )


def kda_decode_workspace_shapes(
    batch_size: int, attention_tp_size: int = TP_SIZE
) -> KDADecodeWorkspace[Shape]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    local_heads = glm53_local_heads(attention_tp_size)
    local_projection_size = local_heads * HEAD_DIM

    return KDADecodeWorkspace(
        projection=(batch_size, 3 * local_projection_size + 2 * HEAD_DIM + local_heads),
        output_gate=(batch_size, local_projection_size),
        gate_raw=(batch_size, local_projection_size),
        local_output=(batch_size, local_heads, HEAD_DIM),
        output=(batch_size, HIDDEN_SIZE),
    )


def dense_workspace_shapes(
    token_capacity: int, attention_tp_size: int = TP_SIZE
) -> GLM53DenseWorkspace[Shape]:
    if token_capacity < 1:
        raise ValueError("token_capacity must be positive")

    glm53_local_heads(attention_tp_size)
    local_intermediate = 12_288 // attention_tp_size
    return GLM53DenseWorkspace(
        mhc_sqrsum=(token_capacity, MHC_PRE_MAX_SPLITS),
        mhc_dot=(
            token_capacity,
            MHC_PRE_MAX_SPLITS,
            MHC_STREAMS * (MHC_STREAMS + 2),
        ),
        post=(token_capacity, MHC_STREAMS),
        comb=(token_capacity, MHC_STREAMS, MHC_STREAMS),
        collapsed=(token_capacity, HIDDEN_SIZE),
        normalized=(token_capacity, HIDDEN_SIZE),
        streams_mid=(token_capacity, MHC_STREAMS, HIDDEN_SIZE),
        hidden_fp8=(token_capacity, HIDDEN_SIZE),
        hidden_scale=(token_capacity, HIDDEN_SIZE // FP8_BLOCK_SIZE),
        gate_up=(token_capacity, 2 * local_intermediate),
        activated=(token_capacity, local_intermediate),
        activated_fp8=(token_capacity, local_intermediate),
        activated_scale=(token_capacity, local_intermediate // FP8_BLOCK_SIZE),
        ffn_output=(token_capacity, HIDDEN_SIZE),
        streams_out=(token_capacity, MHC_STREAMS, HIDDEN_SIZE),
    )


def sparse_ffn_decode_workspace_shapes(
    batch_size: int,
    moe_world_size: int = 1,
) -> SparseFFNDecodeWorkspace[Shape]:
    _validate_chunk_batch_size(batch_size, "sparse FFN")
    if moe_world_size not in {1, TP_SIZE}:
        raise ValueError("GLM MoE world size must be one or four")

    shared_intermediate = MOE_INTERMEDIATE_SIZE // TP_SIZE
    moe_batch_size = batch_size * moe_world_size
    return SparseFFNDecodeWorkspace(
        gathered=(moe_batch_size if moe_world_size > 1 else 0, HIDDEN_SIZE),
        scattered=(batch_size if moe_world_size > 1 else 0, HIDDEN_SIZE),
        scores=(moe_batch_size, NUM_ROUTED_EXPERTS),
        selection=(moe_batch_size, NUM_ROUTED_EXPERTS),
        topk_values=(moe_batch_size, TOP_K),
        topk_ids64=(moe_batch_size, TOP_K),
        topk_ids=(moe_batch_size, TOP_K),
        hidden_fp8=(moe_batch_size, HIDDEN_SIZE),
        hidden_scale_mn=(HIDDEN_SIZE // FP8_BLOCK_SIZE, moe_batch_size),
        routed=(moe_batch_size, HIDDEN_SIZE),
        shared_gate_up=(moe_batch_size, 2 * shared_intermediate),
        shared_fp8=(moe_batch_size, shared_intermediate),
        shared_scale=(moe_batch_size, shared_intermediate // FP8_BLOCK_SIZE),
        output=(moe_batch_size, HIDDEN_SIZE),
    )


def sparse_mla_projection_workspace_shapes(
    batch_size: int,
    attention_tp_size: int = TP_SIZE,
) -> SparseMLAProjectionWorkspace[Shape]:
    _validate_chunk_batch_size(batch_size, "sparse MLA projection")

    local_heads = glm53_local_heads(attention_tp_size)
    index_query_size = SPARSE_MLA_INDEXER_HEADS * SPARSE_MLA_INDEXER_HEAD_DIM
    return SparseMLAProjectionWorkspace(
        hidden_fp8=(batch_size, HIDDEN_SIZE),
        hidden_scale=(batch_size, HIDDEN_SIZE // FP8_BLOCK_SIZE),
        low_rank=(batch_size, SPARSE_MLA_Q_LORA_RANK + SPARSE_MLA_KV_LORA_RANK),
        q_resid=(batch_size, SPARSE_MLA_Q_LORA_RANK),
        latent=(batch_size, SPARSE_MLA_KV_LORA_RANK),
        q_resid_fp8=(batch_size, SPARSE_MLA_Q_LORA_RANK),
        q_resid_scale=(batch_size, SPARSE_MLA_Q_LORA_RANK // FP8_BLOCK_SIZE),
        main_q=(batch_size, local_heads * SPARSE_MLA_QK_NOPE_HEAD_DIM),
        index_q=(batch_size, index_query_size),
        index_prep=(
            batch_size,
            2 * SPARSE_MLA_INDEXER_HEAD_DIM + SPARSE_MLA_INDEXER_HEADS,
        ),
        key=(batch_size, SPARSE_MLA_INDEXER_HEAD_DIM),
        pool_gate=(batch_size, SPARSE_MLA_INDEXER_HEAD_DIM),
        score_weights=(batch_size, SPARSE_MLA_INDEXER_HEADS),
    )


def sparse_mla_decode_workspace_shapes(
    batch_size: int,
    attention_tp_size: int = TP_SIZE,
) -> SparseMLADecodeWorkspace[Shape]:
    _validate_chunk_batch_size(batch_size, "sparse MLA decode")
    local_heads = glm53_local_heads(attention_tp_size)
    counter_bytes = _align_up(max(batch_size * local_heads, SPARSE_MLA_B200_SMS), 8) * 4
    index_q = (batch_size, 1, SPARSE_MLA_INDEXER_HEADS, SPARSE_MLA_INDEXER_HEAD_DIM)
    return SparseMLADecodeWorkspace(
        context_lengths=(batch_size, 1),
        sequence_lengths=(batch_size,),
        index_q=index_q,
        score_weights=(batch_size, SPARSE_MLA_INDEXER_HEADS),
        main_q_hbd=(local_heads, batch_size, SPARSE_MLA_QK_NOPE_HEAD_DIM),
        absorbed_hbd=(local_heads, batch_size, SPARSE_MLA_KV_LORA_RANK),
        attention_q=(batch_size, 1, local_heads, SPARSE_MLA_KV_LORA_RANK),
        schedule=(SPARSE_MLA_B200_SMS + 1, 2),
        logits=(batch_size, SPARSE_MLA_INDEX_CONTEXT_CAPACITY),
        selected=(batch_size, SPARSE_MLA_INDEX_TOP_K),
        topk_offsets=(batch_size,),
        topk_rows=(SPARSE_MLA_TOPK_ROW_STATES_BYTES,),
        sparse_ids=(batch_size, 1, SPARSE_MLA_SPARSE_CAPACITY),
        sparse_lengths=(batch_size,),
        flashinfer_workspace=(SPARSE_MLA_FLASHINFER_WORKSPACE_BYTES,),
        counter=(counter_bytes,),
        output=(batch_size, 1, local_heads, SPARSE_MLA_KV_LORA_RANK),
    )


def sparse_mla_output_workspace_shapes(
    batch_size: int,
    attention_tp_size: int = TP_SIZE,
) -> SparseMLAOutputWorkspace[Shape]:
    _validate_chunk_batch_size(batch_size, "sparse MLA output")
    local_heads = glm53_local_heads(attention_tp_size)
    return SparseMLAOutputWorkspace(
        value_hbd=(local_heads, batch_size, SPARSE_MLA_VALUE_HEAD_DIM),
        projected=(batch_size, local_heads * SPARSE_MLA_VALUE_HEAD_DIM),
        projected_fp8=(batch_size, local_heads * SPARSE_MLA_VALUE_HEAD_DIM),
        projected_scale=(
            batch_size,
            local_heads * SPARSE_MLA_VALUE_HEAD_DIM // FP8_BLOCK_SIZE,
        ),
        output=(batch_size, HIDDEN_SIZE),
    )


def kda_prefill_workspace_shapes(
    total_tokens: int,
    batch_size: int,
    attention_tp_size: int = TP_SIZE,
) -> KDAPrefillWorkspace[Shape]:
    if total_tokens < 1:
        raise ValueError("total_tokens must be positive")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    local_heads = glm53_local_heads(attention_tp_size)
    local_projection_size = local_heads * HEAD_DIM
    packed_projection_size = 3 * local_projection_size + 2 * HEAD_DIM + local_heads
    kernel_workspace_bytes = kda_prefill_kernel_workspace_bytes(
        total_tokens, batch_size, attention_tp_size
    )
    return KDAPrefillWorkspace(
        projection=(total_tokens, packed_projection_size),
        gates=(2, total_tokens, local_projection_size),
        beta=(1, total_tokens, local_heads),
        qkv=(3, 1, total_tokens, local_heads, HEAD_DIM),
        initial_state=(batch_size, local_heads, HEAD_DIM, HEAD_DIM),
        kda_output=(1, total_tokens, local_heads, HEAD_DIM),
        final_state=(batch_size, local_heads, HEAD_DIM, HEAD_DIM),
        kda_workspace_storage=(
            kernel_workspace_bytes + FLASH_KDA_WORKSPACE_ALIGNMENT - 1,
        ),
        kda_workspace=(kernel_workspace_bytes,),
        output=(total_tokens, HIDDEN_SIZE),
    )


def kda_prefill_kernel_workspace_bytes(
    total_tokens: int, batch_size: int, attention_tp_size: int = TP_SIZE
) -> int:
    """Size the opaque workspace for the pinned patched FlashKDA build."""
    if total_tokens < 1:
        raise ValueError("total_tokens must be positive")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    local_heads = glm53_local_heads(attention_tp_size)
    tiles = (
        total_tokens + FLASH_KDA_CHUNK_SIZE - 1
    ) // FLASH_KDA_CHUNK_SIZE + batch_size
    prefix_bytes = _align_up(
        (batch_size + 1) * 4,
        FLASH_KDA_WORKSPACE_ALIGNMENT,
    )
    return (
        local_heads * tiles * FLASH_KDA_WORKSPACE_TILE_BYTES
        + prefix_bytes
        + total_tokens * local_heads * 2
    )


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment

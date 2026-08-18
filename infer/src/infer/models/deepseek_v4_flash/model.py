from dataclasses import dataclass
from math import prod
from typing import Literal

DEP_SIZE = 4
EP_SIZE = 4
TEP_SIZE = 4
DECODE_BATCH_SIZES = (1, 2, 4, 8, 16, 32, 64)
DECODE_DESCRIPTOR_PARITIES = 2
PREFILL_CHUNK_TOKENS = 4_096
MAX_PREFILL_REQUESTS = 4
NUM_LAYERS = 43
HIDDEN_SIZE = 4096
VOCAB_SIZE = 129_280
NUM_QUERY_HEADS = 64
HEAD_DIM = 512
ROPE_HEAD_DIM = 64
NOPE_HEAD_DIM = HEAD_DIM - ROPE_HEAD_DIM
NUM_ROUTED_EXPERTS = 256
TOP_K = 6
EXPERT_INTERMEDIATE_SIZE = 2048
SHARED_INTERMEDIATE_SIZE = 2048
RMS_NORM_EPS = 1.0e-6
MHC_STREAMS = 4
MHC_EPS = 1.0e-6
MHC_NORMALIZATION_ITERS = 20

DSPARK_STAGE_COUNT = 3
DSPARK_VERIFY_WIDTHS = (4, 6)
DSPARK_VERIFY_WIDTH = DSPARK_VERIFY_WIDTHS[-1]
DSPARK_BLOCK_SIZE = DSPARK_VERIFY_WIDTH - 1
DSPARK_WINDOW_TOKENS = 128
DSPARK_PAGE_TOKENS = 64
RAW_STATE_RING_TOKENS = 192
C4_STATE_RING_TOKENS = 16
C128_STATE_RING_TOKENS = 256
DSPARK_TARGET_LAYER_IDS = (40, 41, 42)
DSPARK_NOISE_TOKEN_ID = 128_799
DSPARK_MARKOV_RANK = 256

LOCAL_QUERY_HEADS = NUM_QUERY_HEADS // TEP_SIZE
LOCAL_EXPERTS = NUM_ROUTED_EXPERTS // EP_SIZE
LOCAL_SHARED_INTERMEDIATE_SIZE = SHARED_INTERMEDIATE_SIZE // TEP_SIZE

MAX_CONTEXT_TOKENS = 1_048_576
INDEX_COMPRESSION = 4
INDEX_TOP_K = 512
INDEX_CONTEXT_CAPACITY = MAX_CONTEXT_TOKENS // INDEX_COMPRESSION
TOPK_ROW_STATES_BYTES = 1_048_576


def validate_dspark_verify_width(width: int) -> None:
    if width not in DSPARK_VERIFY_WIDTHS:
        raise ValueError("DeepSeek DSpark verify width must be 4 or 6")


def configure_dspark_verify_width(width: int) -> None:
    validate_dspark_verify_width(width)
    global DSPARK_BLOCK_SIZE, DSPARK_VERIFY_WIDTH
    DSPARK_BLOCK_SIZE = width - 1
    DSPARK_VERIFY_WIDTH = width


CSA_LAYER_IDS = tuple(range(2, NUM_LAYERS, 2))
HCA_LAYER_IDS = tuple(range(3, NUM_LAYERS, 2))
COMPRESS_RATIOS = tuple(
    0 if layer_id < 2 else 4 if layer_id % 2 == 0 else 128
    for layer_id in range(NUM_LAYERS)
)
HASH_LAYER_IDS = (0, 1, 2)
# Hash layers replace only the learned top-k indices with tid2eid[input_ids].
ROUTER_SCALE = 1.5
ROUTER_DENOMINATOR_MIN = 1.0e-20
FP8_BLOCK_SIZE = 128

# Candidate physical state pending the design's cache-precision quality gate.
SWA_CANDIDATE_SCALE_BLOCK = 64
SWA_CANDIDATE_SCALE_BYTES = (
    NOPE_HEAD_DIM + SWA_CANDIDATE_SCALE_BLOCK - 1
) // SWA_CANDIDATE_SCALE_BLOCK
SWA_CANDIDATE_SCALE_PAD_BYTES = (
    -(NOPE_HEAD_DIM + 2 * ROPE_HEAD_DIM + SWA_CANDIDATE_SCALE_BYTES) % 8
)
SWA_CANDIDATE_ROW_BYTES = (
    NOPE_HEAD_DIM
    + 2 * ROPE_HEAD_DIM
    + SWA_CANDIDATE_SCALE_BYTES
    + SWA_CANDIDATE_SCALE_PAD_BYTES
)
SWA_WINDOW_TOKENS = 128

Shape = tuple[int, ...]
ScalarType = Literal["BF16", "F32", "F8_E4M3", "F8_E8M0", "I8", "I64", "U8"]
AttentionKind = Literal["swa", "csa", "hca"]


def _reset_flash_mla_metadata(metadata: object) -> None:
    if not metadata.have_initialized:
        return
    metadata.have_initialized = False
    metadata.config = None
    metadata.tile_scheduler_metadata = None
    metadata.num_splits = None


@dataclass(frozen=True, slots=True)
class PackedHistoryLayout:
    logical_view_shape: Shape
    physical_storage_bytes: int
    dtype: ScalarType = "U8"

    @property
    def logical_bytes(self) -> int:
        return prod(self.logical_view_shape)

    @property
    def physical_storage_shape(self) -> Shape:
        return (self.physical_storage_bytes,)


@dataclass(frozen=True, slots=True)
class DeepSeekV4TargetState[TensorT]:
    """One fixed target-only slot; compression cursors derive from raw length."""

    raw_window: TensorT
    c4_main: TensorT
    c4_index: TensorT
    c128_main: TensorT


@dataclass(frozen=True, slots=True)
class DeepSeekV4CompressionBatch[TensorT]:
    """One parity's contiguous device descriptors for a fixed graph bucket."""

    positions: TensorT
    request_indices: TensorT
    state_slots: TensorT
    output_slots: TensorT
    state_table: TensorT
    state_table_base: TensorT
    cos_sin: TensorT


@dataclass(frozen=True, slots=True)
class DeepSeekV4Compression[TensorT]:
    """Preprojected compressor tensors; C4 APE uses runtime-reordered rows."""

    kv_score: TensorT
    ape: TensorT
    state: TensorT
    norm_weight: TensorT
    cache: TensorT


@dataclass(frozen=True, slots=True)
class DeepSeekV4PrefillState[TensorT]:
    """Paged chunk scratch with bounded transfers to the retained state ring."""

    persistent: TensorT
    seed_source: TensorT
    seed_destination: TensorT
    retain_source: TensorT
    retain_destination: TensorT
    transfer: TensorT


@dataclass(frozen=True, slots=True)
class DeepSeekV4PrefillMetadata[TensorT]:
    query_start: TensorT
    prefix_lengths: TensorT
    state_slots: TensorT
    block_table: TensorT
    block_table_base: TensorT


@dataclass(frozen=True, slots=True)
class DeepSeekV4PrefillWorkspace[TensorT]:
    kv: TensorT
    raw_slots: TensorT


@dataclass(frozen=True, slots=True)
class DeepSeekV4RawAttention[TensorT]:
    """FlashMLA input with 64 heads; TEP4 pads the 48 non-local heads."""

    query: TensorT
    cache: TensorT
    indices: TensorT
    lengths: TensorT
    sink: TensorT


@dataclass(frozen=True, slots=True)
class DeepSeekV4IndexQuery[TensorT]:
    """Packed MXFP4 query and complete C4-pool lengths."""

    query: TensorT
    scale: TensorT
    weights: TensorT
    block_table: TensorT
    lengths: TensorT


@dataclass(frozen=True, slots=True)
class DeepSeekV4PrefillIndexWorkspace[TensorT]:
    """Fixed-shape C4 indexer scratch shared by packed prefill chunks."""

    staged: DeepSeekV4IndexQuery[TensorT]
    request_indices: TensorT
    schedule: TensorT
    logits: TensorT
    candidates: TensorT
    topk_offsets: TensorT
    topk_rows: TensorT


@dataclass(frozen=True, slots=True)
class DeepSeekV4AttentionPool[TensorT]:
    indices: TensorT
    lengths: TensorT


@dataclass(frozen=True, slots=True)
class DeepSeekV4DecodeWorkspace[TensorT]:
    """Caller-owned tensors; allocate one independent instance per parity.

    Row state is zero-initialized before capture; every field retains its
    address across graph replays.
    """

    schedule: TensorT
    logits: TensorT
    topk_rows: TensorT
    mapped_c4: TensorT
    selected_lengths: TensorT


@dataclass(frozen=True, slots=True)
class WeightSpec:
    name: str
    shape: Shape
    dtype: ScalarType
    shard_axis: int | None = None

    def checkpoint_key_for_layer(self, layer_id: int) -> str:
        _validate_layer_id(layer_id)
        return f"layers.{layer_id}.{self.name}"

    def local_shape(self, tep_size: int = TEP_SIZE) -> Shape:
        if self.shard_axis is None:
            return self.shape
        dimension = self.shape[self.shard_axis]
        if dimension % tep_size:
            raise ValueError(
                f"{self.name} axis {self.shard_axis} cannot shard {tep_size} ways"
            )
        shape = list(self.shape)
        shape[self.shard_axis] = dimension // tep_size
        return tuple(shape)


LAYER0_NON_EXPERT_WEIGHTS = (
    WeightSpec("ffn.gate.tid2eid", (VOCAB_SIZE, TOP_K), "I64"),
    WeightSpec("attn.attn_sink", (NUM_QUERY_HEADS,), "F32", 0),
    WeightSpec("hc_attn_base", (24,), "F32"),
    WeightSpec("hc_attn_fn", (24, 4 * HIDDEN_SIZE), "F32"),
    WeightSpec("hc_attn_scale", (3,), "F32"),
    WeightSpec("hc_ffn_base", (24,), "F32"),
    WeightSpec("hc_ffn_fn", (24, 4 * HIDDEN_SIZE), "F32"),
    WeightSpec("hc_ffn_scale", (3,), "F32"),
    WeightSpec("attn.kv_norm.weight", (HEAD_DIM,), "BF16"),
    WeightSpec("attn.q_norm.weight", (1024,), "BF16"),
    WeightSpec("attn_norm.weight", (HIDDEN_SIZE,), "BF16"),
    WeightSpec("ffn.gate.weight", (NUM_ROUTED_EXPERTS, HIDDEN_SIZE), "BF16"),
    WeightSpec("ffn_norm.weight", (HIDDEN_SIZE,), "BF16"),
    WeightSpec("attn.wkv.scale", (4, 32), "F8_E8M0"),
    WeightSpec("attn.wo_a.scale", (64, 32), "F8_E8M0", 0),
    WeightSpec("attn.wo_b.scale", (32, 64), "F8_E8M0", 1),
    WeightSpec("attn.wq_a.scale", (8, 32), "F8_E8M0"),
    WeightSpec("attn.wq_b.scale", (256, 8), "F8_E8M0", 0),
    WeightSpec("ffn.shared_experts.w1.scale", (16, 32), "F8_E8M0", 0),
    WeightSpec("ffn.shared_experts.w2.scale", (32, 16), "F8_E8M0", 1),
    WeightSpec("ffn.shared_experts.w3.scale", (16, 32), "F8_E8M0", 0),
    WeightSpec("attn.wkv.weight", (HEAD_DIM, HIDDEN_SIZE), "F8_E4M3"),
    WeightSpec("attn.wo_a.weight", (8192, HIDDEN_SIZE), "F8_E4M3", 0),
    WeightSpec("attn.wo_b.weight", (HIDDEN_SIZE, 8192), "F8_E4M3", 1),
    WeightSpec("attn.wq_a.weight", (1024, HIDDEN_SIZE), "F8_E4M3"),
    WeightSpec("attn.wq_b.weight", (32768, 1024), "F8_E4M3", 0),
    WeightSpec("ffn.shared_experts.w1.weight", (2048, HIDDEN_SIZE), "F8_E4M3", 0),
    WeightSpec("ffn.shared_experts.w2.weight", (HIDDEN_SIZE, 2048), "F8_E4M3", 1),
    WeightSpec("ffn.shared_experts.w3.weight", (2048, HIDDEN_SIZE), "F8_E4M3", 0),
)

COMPRESSED_ATTENTION_COMMON_WEIGHTS = (
    WeightSpec("attn_norm.weight", (HIDDEN_SIZE,), "BF16"),
    WeightSpec("attn.attn_sink", (NUM_QUERY_HEADS,), "F32", 0),
    WeightSpec("attn.wq_a.weight", (1024, HIDDEN_SIZE), "F8_E4M3"),
    WeightSpec("attn.wq_a.scale", (8, 32), "F8_E8M0"),
    WeightSpec("attn.q_norm.weight", (1024,), "BF16"),
    WeightSpec("attn.wq_b.weight", (32768, 1024), "F8_E4M3", 0),
    WeightSpec("attn.wq_b.scale", (256, 8), "F8_E8M0", 0),
    WeightSpec("attn.wkv.weight", (HEAD_DIM, HIDDEN_SIZE), "F8_E4M3"),
    WeightSpec("attn.wkv.scale", (4, 32), "F8_E8M0"),
    WeightSpec("attn.kv_norm.weight", (HEAD_DIM,), "BF16"),
    WeightSpec("attn.wo_a.weight", (8192, HIDDEN_SIZE), "F8_E4M3", 0),
    WeightSpec("attn.wo_a.scale", (64, 32), "F8_E8M0", 0),
    WeightSpec("attn.wo_b.weight", (HIDDEN_SIZE, 8192), "F8_E4M3", 1),
    WeightSpec("attn.wo_b.scale", (32, 64), "F8_E8M0", 1),
)

C4_ATTENTION_WEIGHTS = COMPRESSED_ATTENTION_COMMON_WEIGHTS + (
    WeightSpec("attn.compressor.ape", (4, 1024), "F32"),
    WeightSpec("attn.compressor.norm.weight", (HEAD_DIM,), "BF16"),
    WeightSpec("attn.compressor.wkv.weight", (1024, HIDDEN_SIZE), "BF16"),
    WeightSpec("attn.compressor.wgate.weight", (1024, HIDDEN_SIZE), "BF16"),
    WeightSpec("attn.indexer.wq_b.weight", (8192, 1024), "F8_E4M3"),
    WeightSpec("attn.indexer.wq_b.scale", (64, 8), "F8_E8M0"),
    WeightSpec("attn.indexer.weights_proj.weight", (64, HIDDEN_SIZE), "BF16"),
    WeightSpec("attn.indexer.compressor.ape", (4, 256), "F32"),
    WeightSpec("attn.indexer.compressor.norm.weight", (128,), "BF16"),
    WeightSpec("attn.indexer.compressor.wkv.weight", (256, HIDDEN_SIZE), "BF16"),
    WeightSpec("attn.indexer.compressor.wgate.weight", (256, HIDDEN_SIZE), "BF16"),
)

C128_ATTENTION_WEIGHTS = COMPRESSED_ATTENTION_COMMON_WEIGHTS + (
    WeightSpec("attn.compressor.ape", (128, HEAD_DIM), "F32"),
    WeightSpec("attn.compressor.norm.weight", (HEAD_DIM,), "BF16"),
    WeightSpec("attn.compressor.wkv.weight", (HEAD_DIM, HIDDEN_SIZE), "BF16"),
    WeightSpec("attn.compressor.wgate.weight", (HEAD_DIM, HIDDEN_SIZE), "BF16"),
)

C4_FP8_MAIN_HISTORY_CANDIDATE = PackedHistoryLayout((32, 1, 584), 19_008)
C4_FP4_INDEX_HISTORY_CANDIDATE = PackedHistoryLayout((32, 1, 68), 2_176)
C128_FP8_MAIN_HISTORY_CANDIDATE = PackedHistoryLayout((1, 1, 584), 1_152)
COMPRESSED_HISTORY_CANDIDATE_BLOCK_TOKENS = 128

TARGET_STATE_SLOT_SHAPES = DeepSeekV4TargetState(
    raw_window=(NUM_LAYERS, RAW_STATE_RING_TOKENS // DSPARK_PAGE_TOKENS, 37_440),
    c4_main=(len(CSA_LAYER_IDS), C4_STATE_RING_TOKENS, 2_048),
    c4_index=(len(CSA_LAYER_IDS), C4_STATE_RING_TOKENS, 512),
    c128_main=(len(HCA_LAYER_IDS), C128_STATE_RING_TOKENS, 1_024),
)
COMPRESS_STATE_INITIAL_VALUES = (0.0, float("-inf"))

ROUTED_EXPERT_TEMPLATE = (
    WeightSpec("w1.scale", (2048, 128), "F8_E8M0"),
    WeightSpec("w2.scale", (4096, 64), "F8_E8M0"),
    WeightSpec("w3.scale", (2048, 128), "F8_E8M0"),
    WeightSpec("w1.weight", (2048, 2048), "I8"),
    WeightSpec("w2.weight", (4096, 1024), "I8"),
    WeightSpec("w3.weight", (2048, 2048), "I8"),
)

DSPARK_STAGE0_WEIGHTS = (
    WeightSpec("main_norm.weight", (HIDDEN_SIZE,), "BF16"),
    WeightSpec(
        "main_proj.scale",
        (HIDDEN_SIZE // FP8_BLOCK_SIZE, 3 * HIDDEN_SIZE // FP8_BLOCK_SIZE),
        "F8_E8M0",
    ),
    WeightSpec("main_proj.weight", (HIDDEN_SIZE, 3 * HIDDEN_SIZE), "F8_E4M3"),
)
DSPARK_ENDPOINT_WEIGHTS = (
    WeightSpec(
        "confidence_head.proj.weight",
        (1, HIDDEN_SIZE + DSPARK_MARKOV_RANK),
        "BF16",
    ),
    WeightSpec("hc_head_base", (MHC_STREAMS,), "F32"),
    WeightSpec("hc_head_fn", (MHC_STREAMS, MHC_STREAMS * HIDDEN_SIZE), "F32"),
    WeightSpec("hc_head_scale", (1,), "F32"),
    WeightSpec(
        "markov_head.markov_w1.weight",
        (VOCAB_SIZE, DSPARK_MARKOV_RANK),
        "BF16",
    ),
    WeightSpec(
        "markov_head.markov_w2.weight",
        (VOCAB_SIZE, DSPARK_MARKOV_RANK),
        "BF16",
    ),
    WeightSpec("norm.weight", (HIDDEN_SIZE,), "BF16"),
)


def attention_kind(layer_id: int) -> AttentionKind:
    _validate_layer_id(layer_id)
    if layer_id < 2:
        return "swa"
    return "csa" if layer_id % 2 == 0 else "hca"


def uses_hash_routing(layer_id: int) -> bool:
    _validate_layer_id(layer_id)
    return layer_id in HASH_LAYER_IDS


def compressed_attention_weights(layer_id: int) -> tuple[WeightSpec, ...]:
    _validate_layer_id(layer_id)
    ratio = COMPRESS_RATIOS[layer_id]
    if ratio == 4:
        return C4_ATTENTION_WEIGHTS
    if ratio == 128:
        return C128_ATTENTION_WEIGHTS
    raise ValueError(f"layer {layer_id} does not use compressed attention")


def _validate_layer_id(layer_id: int) -> None:
    if not 0 <= layer_id < NUM_LAYERS:
        raise ValueError(f"layer_id must be in [0, {NUM_LAYERS}), got {layer_id}")


def validate_tep_rank(tep_rank: int, tep_size: int) -> None:
    if tep_size < 1:
        raise ValueError("tep_size must be positive")
    if not 0 <= tep_rank < tep_size:
        raise ValueError(f"tep_rank must be in [0, {tep_size}), got {tep_rank}")

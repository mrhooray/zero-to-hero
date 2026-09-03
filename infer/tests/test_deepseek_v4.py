import unittest

from infer.models.deepseek_v4_flash import MODEL_ID, MODEL_REVISION
from infer.models.deepseek_v4_flash import model as v4


class LayerPatternTest(unittest.TestCase):
    def test_pinned_identity_and_attention_pattern(self) -> None:
        self.assertEqual(MODEL_ID, "deepseek-ai/DeepSeek-V4-Flash-0731")
        self.assertEqual(
            MODEL_REVISION,
            "7872f01b1d1fe23eabc4c98b48bffcef5a386062",
        )
        self.assertEqual(v4.NUM_LAYERS, 43)
        self.assertEqual(v4.CSA_LAYER_IDS, tuple(range(2, 43, 2)))
        self.assertEqual(v4.HCA_LAYER_IDS, tuple(range(3, 43, 2)))
        self.assertEqual(v4.COMPRESS_RATIOS, (0, 0) + (4, 128) * 20 + (4,))
        self.assertEqual(
            tuple(v4.attention_kind(layer_id) for layer_id in range(43)),
            ("swa", "swa") + ("csa", "hca") * 20 + ("csa",),
        )

        with self.assertRaisesRegex(ValueError, "layer_id must be"):
            v4.attention_kind(43)


class CompressedAttentionContractTest(unittest.TestCase):
    def test_tep4_shapes_follow_query_head_slices(self) -> None:
        c4 = {spec.name: spec for spec in v4.C4_ATTENTION_WEIGHTS}
        expected_local = {
            "attn.attn_sink": (16,),
            "attn.wq_b.weight": (8192, 1024),
            "attn.wq_b.scale": (64, 8),
            "attn.wo_a.weight": (2048, 4096),
            "attn.wo_a.scale": (16, 32),
            "attn.wo_b.weight": (4096, 2048),
            "attn.wo_b.scale": (32, 16),
        }
        self.assertEqual(
            {name: c4[name].local_shape() for name in expected_local},
            expected_local,
        )
        self.assertTrue(
            all(
                spec.local_shape() == spec.shape
                for spec in v4.C4_ATTENTION_WEIGHTS
                if spec.name not in expected_local
            )
        )
        self.assertEqual(c4["attn.indexer.wq_b.weight"].local_shape(), (8192, 1024))
        self.assertEqual(c4["attn.indexer.wq_b.scale"].local_shape(), (64, 8))
        self.assertEqual(
            c4["attn.indexer.weights_proj.weight"].local_shape(), (64, 4096)
        )
        self.assertEqual(
            {
                spec.name: spec.local_shape()
                for spec in v4.C128_ATTENTION_WEIGHTS
                if spec.shard_axis is not None
            },
            expected_local,
        )

    def test_rejects_non_compressed_layers(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not use compressed attention"):
            v4.compressed_attention_weights(0)


class TEP4PlacementTest(unittest.TestCase):
    def test_attention_and_shared_expert_tensor_slices(self) -> None:
        specs = {spec.name: spec for spec in v4.LAYER0_NON_EXPERT_WEIGHTS}
        expected = {
            "attn.attn_sink": (16,),
            "attn.wq_b.weight": (8192, 1024),
            "attn.wq_b.scale": (64, 8),
            "attn.wo_a.weight": (2048, 4096),
            "attn.wo_a.scale": (16, 32),
            "attn.wo_b.weight": (4096, 2048),
            "attn.wo_b.scale": (32, 16),
            "ffn.shared_experts.w1.weight": (512, 4096),
            "ffn.shared_experts.w1.scale": (4, 32),
            "ffn.shared_experts.w2.weight": (4096, 512),
            "ffn.shared_experts.w2.scale": (32, 4),
        }

        self.assertEqual(v4.LOCAL_QUERY_HEADS, 16)
        self.assertEqual(v4.LOCAL_SHARED_INTERMEDIATE_SIZE, 512)
        self.assertEqual(
            {name: specs[name].local_shape() for name in expected},
            expected,
        )
        self.assertEqual(specs["attn.wkv.weight"].local_shape(), (512, 4096))
        self.assertEqual(specs["ffn.gate.weight"].local_shape(), (256, 4096))


class HashRoutingTest(unittest.TestCase):
    def test_first_three_layers_use_token_id_routes_and_learned_weights(self) -> None:
        self.assertEqual(
            tuple(v4.uses_hash_routing(layer_id) for layer_id in range(5)),
            (True, True, True, False, False),
        )
        self.assertEqual(v4.ROUTER_SCALE, 1.5)
        tid2eid = v4.LAYER0_NON_EXPERT_WEIGHTS[0]
        self.assertEqual((tid2eid.dtype, tid2eid.shape), ("I64", (129280, 6)))


class SWAStateTest(unittest.TestCase):
    def test_layer0_candidate_state_geometry(self) -> None:
        self.assertEqual(v4.NOPE_HEAD_DIM, 448)
        self.assertEqual(v4.SWA_CANDIDATE_SCALE_BYTES, 7)
        self.assertEqual(v4.SWA_CANDIDATE_SCALE_PAD_BYTES, 1)
        self.assertEqual(v4.SWA_CANDIDATE_ROW_BYTES, 584)


class CompressedStateTest(unittest.TestCase):
    def test_precision_gated_history_separates_views_from_storage(self) -> None:
        self.assertEqual(
            v4.C4_FP8_MAIN_HISTORY_CANDIDATE,
            v4.PackedHistoryLayout((32, 1, 584), 19_008),
        )
        self.assertEqual(
            v4.C4_FP4_INDEX_HISTORY_CANDIDATE,
            v4.PackedHistoryLayout((32, 1, 68), 2_176),
        )
        self.assertEqual(
            v4.C128_FP8_MAIN_HISTORY_CANDIDATE,
            v4.PackedHistoryLayout((1, 1, 584), 1_152),
        )
        self.assertEqual(v4.C4_FP8_MAIN_HISTORY_CANDIDATE.logical_bytes, 18_688)
        self.assertEqual(
            v4.C4_FP8_MAIN_HISTORY_CANDIDATE.physical_storage_shape, (19_008,)
        )
        self.assertEqual(v4.C128_FP8_MAIN_HISTORY_CANDIDATE.logical_bytes, 584)
        self.assertEqual(
            v4.C128_FP8_MAIN_HISTORY_CANDIDATE.physical_storage_shape, (1_152,)
        )
        self.assertEqual(v4.COMPRESSED_HISTORY_CANDIDATE_BLOCK_TOKENS, 128)

    def test_target_only_fixed_slot_geometry(self) -> None:
        self.assertEqual(
            v4.TARGET_STATE_SLOT_SHAPES,
            v4.DeepSeekV4TargetState(
                raw_window=(43, 3, 37_440),
                c4_main=(21, 16, 2_048),
                c4_index=(21, 16, 512),
                c128_main=(20, 256, 1_024),
            ),
        )
        self.assertEqual(v4.COMPRESS_STATE_INITIAL_VALUES, (0.0, float("-inf")))


if __name__ == "__main__":
    unittest.main()

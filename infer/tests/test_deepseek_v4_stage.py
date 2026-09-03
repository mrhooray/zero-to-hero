import unittest
from pathlib import Path

from infer.models.deepseek_v4_flash import model as deepseek_v4_flash

SOURCE = (
    Path(__file__).resolve().parents[1]
    / "src/infer/models/deepseek_v4_flash/ops/stage.py"
).read_text()


def descriptors(slot: int, position: int, block_table: tuple[int, ...]):
    raw_ring = deepseek_v4_flash.RAW_STATE_RING_TOKENS
    c4_ring = deepseek_v4_flash.C4_STATE_RING_TOKENS
    c128_ring = deepseek_v4_flash.C128_STATE_RING_TOKENS
    length = min(position + 1, deepseek_v4_flash.DSPARK_WINDOW_TOKENS)
    c4_base = max(position // 4 - 2, 0)
    c128_base = max(position // 8 - 16, 0)
    history = block_table[position // 128]
    return {
        "raw": tuple(
            slot * raw_ring + logical % raw_ring
            for logical in range(position - length + 1, position + 1)
        ),
        "raw_slot": slot * raw_ring + position % raw_ring,
        "c4_slot": slot * c4_ring + position % c4_ring,
        "c4_output": (
            history * 32 + position // 4 % 32 if (position + 1) % 4 == 0 else -1
        ),
        "c4_table": tuple(
            slot * (c4_ring // 4) + (c4_base + row) % (c4_ring // 4) for row in range(3)
        ),
        "c4_base": c4_base,
        "c4_length": (position + 1) // 4,
        "c128_slot": slot * c128_ring + position % c128_ring,
        "c128_output": history if (position + 1) % 128 == 0 else -1,
        "c128_table": tuple(
            slot * (c128_ring // 8) + (c128_base + row) % (c128_ring // 8)
            for row in range(17)
        ),
        "c128_base": c128_base,
        "c128_length": (position + 1) // 128,
    }


def verify_descriptors(
    slot: int,
    committed: int,
    block_table: tuple[int, ...],
    *,
    dummy_slot: int,
    active: bool = True,
):
    rows = []
    for step in range(6):
        live = active and committed + step < deepseek_v4_flash.MAX_CONTEXT_TOKENS
        row_slot = slot if live else dummy_slot
        position = committed + step if live else step
        table = block_table if live else (-1,) * len(block_table)
        staged = descriptors(row_slot, position, table)
        if not live:
            staged["c4_length"] = staged["c128_length"] = 0
        rows.append(
            {
                "live": live,
                "slot": row_slot,
                "position": position,
                **staged,
            }
        )
    return tuple(rows)


class DeepSeekV4StageTest(unittest.TestCase):
    def test_dspark_scalar_stores_use_scalar_masks(self) -> None:
        stage = SOURCE[
            SOURCE.index("def _stage_dspark(") : SOURCE.index("def _commit_dspark(")
        ]
        commit = SOURCE[
            SOURCE.index("def _commit_dspark(") : SOURCE.index(
                "def _seed_dspark_anchors("
            )
        ]

        self.assertNotIn("first = offsets == 0", stage)
        self.assertNotIn("mask=first", stage)
        self.assertIn("mask=offsets < context_length", stage)
        self.assertIn("mask=offsets < BLOCK", stage)
        self.assertIn("block_indices + row * PAGE + offsets", stage)
        self.assertNotIn("block_indices + row * BLOCK + offsets", stage)
        self.assertIn("first = tl.program_id(1) == 0", commit)
        self.assertNotIn("& (dimensions == 0)", commit)

    def test_ring_and_history_boundaries(self) -> None:
        table = (11, 13, 17, 19, 23, 29, 31, 37) * 1024

        initial = descriptors(3, 0, table)
        self.assertEqual(initial["raw"], (576,))
        self.assertEqual(initial["c4_table"], (12, 13, 14))
        self.assertEqual(initial["c128_table"], tuple(range(96, 113)))

        boundary = descriptors(3, 127, table)
        self.assertEqual(boundary["raw"], tuple(range(576, 704)))
        self.assertEqual(boundary["c4_output"], 11 * 32 + 31)
        self.assertEqual(boundary["c128_output"], 11)
        self.assertEqual(boundary["c4_base"], 29)
        self.assertEqual(boundary["c128_base"], 0)

        wrapped = descriptors(3, 128, table)
        self.assertEqual(wrapped["raw"], tuple(range(577, 705)))
        self.assertEqual(wrapped["c4_length"], 32)
        self.assertEqual(wrapped["c128_length"], 1)
        self.assertEqual(wrapped["c4_output"], -1)
        self.assertEqual(wrapped["c128_output"], -1)

        second = descriptors(3, 255, table)
        self.assertEqual(second["c4_output"], 13 * 32 + 31)
        self.assertEqual(second["c128_output"], 13)
        self.assertEqual(second["c128_base"], 15)

    def test_speculative_writes_do_not_alias_live_windows(self) -> None:
        rings = (
            (
                deepseek_v4_flash.TARGET_STATE_SLOT_SHAPES.raw_window[1]
                * deepseek_v4_flash.DSPARK_PAGE_TOKENS,
                deepseek_v4_flash.DSPARK_WINDOW_TOKENS,
            ),
            (deepseek_v4_flash.TARGET_STATE_SLOT_SHAPES.c4_main[1], 8),
            (deepseek_v4_flash.TARGET_STATE_SLOT_SHAPES.c128_main[1], 128),
        )
        for ring, window in rings:
            position = window - 1
            live = {
                logical % ring for logical in range(position - window + 1, position + 1)
            }
            drafts = {
                logical % ring
                for logical in range(
                    position + 1,
                    position + deepseek_v4_flash.DSPARK_VERIFY_WIDTH,
                )
            }
            self.assertTrue(live.isdisjoint(drafts))

    def test_fused_kernel_contains_the_qualified_formulas_without_host_table(
        self,
    ) -> None:
        for expression in (
            "slot * RAW_RING + logical % RAW_RING",
            "slot * C4_RING + position % C4_RING",
            "slot * (C4_RING // 4) + (c4_page + offsets) % (C4_RING // 4)",
            "history_block * 32 + (position // 4) % 32",
            "slot * (C128_RING // 8) + (c128_page + offsets) % (C128_RING // 8)",
            "execution_tables + control_offsets",
            "group = row // VERIFY_WIDTH",
            "step = row % VERIFY_WIDTH",
            "committed = tl.load(committed_lengths + control_slot)",
            "tl.where(live, (position + 1) // 4, 0)",
            "tl.where(live, (position + 1) // 128, 0)",
            "position = tl.load(committed_lengths + state_slot) + offset",
            "valid = active & (offset < accepted - 1)",
        ):
            self.assertIn(expression, SOURCE)
        self.assertNotIn(".tolist()", SOURCE)
        self.assertNotIn('device="cpu"', SOURCE)

    def test_verify_execution_ignores_acceptance_budget(self) -> None:
        stage = SOURCE[: SOURCE.index("def publish_deepseek_v4_prefill_lengths(")]
        kernel = SOURCE[
            SOURCE.index("def _stage_verify(") : SOURCE.index("def _stage_dspark(")
        ]

        self.assertNotIn("remaining", stage)
        self.assertNotIn("remaining", kernel)
        self.assertIn("tl.load(active_ptr + group)", kernel)
        self.assertIn("committed + step < MAX_CONTEXT_TOKENS", kernel)

    def test_target_only_staging_reads_one_sample_per_live_slot(self) -> None:
        target = SOURCE[
            SOURCE.index("def stage_deepseek_v4_target(") : SOURCE.index(
                "def publish_deepseek_v4_prefill_lengths("
            )
        ]
        kernel = SOURCE[
            SOURCE.index("def _stage_verify(") : SOURCE.index("def _stage_dspark(")
        ]

        self.assertIn("target.sampled_tokens", target)
        self.assertIn("        1,\n        True,\n", target)
        self.assertIn("candidate_token_ids + control_slot", kernel)

    def test_flattened_verify_rows_cover_full_window_and_history_boundaries(
        self,
    ) -> None:
        table = (11, 13, 17, 19, 23, 29, 31, 37) * 1024

        full = verify_descriptors(3, 3, table, dummy_slot=67)
        self.assertEqual(
            tuple((row["slot"], row["position"]) for row in full),
            ((3, 3), (3, 4), (3, 5), (3, 6), (3, 7), (3, 8)),
        )
        self.assertEqual(full[0]["raw"], (576, 577, 578, 579))
        self.assertEqual(full[0]["c4_output"], 11 * 32)

        boundary = verify_descriptors(3, 127, table, dummy_slot=67)
        self.assertEqual(
            tuple(row["position"] for row in boundary), tuple(range(127, 133))
        )
        self.assertTrue(all(row["slot"] == 3 for row in boundary))
        self.assertEqual(boundary[0]["c128_output"], 11)
        self.assertEqual(boundary[1]["raw"], tuple(range(577, 705)))
        self.assertEqual(boundary[4]["c4_output"], 13 * 32)

        after_boundary = verify_descriptors(3, 128, table, dummy_slot=67)
        self.assertEqual(
            tuple(row["position"] for row in after_boundary), tuple(range(128, 134))
        )
        self.assertEqual(after_boundary[0]["c4_length"], 32)
        self.assertEqual(after_boundary[0]["c128_length"], 1)

    def test_successor_plan_rebases_from_predecessor_device_commit(self) -> None:
        table = (11, 13, 17, 19, 23, 29, 31, 37) * 1024
        predecessor = verify_descriptors(3, 127, table, dummy_slot=67)
        accepted = 2
        successor = verify_descriptors(
            3,
            predecessor[0]["position"] + accepted,
            table,
            dummy_slot=67,
        )

        self.assertEqual(successor[0]["position"], 129)
        self.assertNotEqual(successor[0]["position"], 127 + 6)
        self.assertEqual(successor[2]["c4_output"], 13 * 32)
        self.assertEqual(successor[0]["c128_length"], 1)


if __name__ == "__main__":
    unittest.main()

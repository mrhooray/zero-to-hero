import unittest
from dataclasses import fields

from infer.protocol import (
    MAX_BATCH_PLAN_BYTES,
    BatchPlan,
    BatchPlanRow,
    Request,
    StepResult,
    StepResultRow,
    TableDelta,
    decode_batch_plan,
    encode_batch_plan,
)


class ProtocolTest(unittest.TestCase):
    def test_request_owns_only_scheduler_state(self) -> None:
        request = Request(7, (11, 12), 32, 0.0, 1.0, 99)
        request.slot = 3
        request.lane = 1
        request.output_token_ids.append(13)

        self.assertEqual(request.output_token_ids, [13])
        self.assertEqual(request.committed_tokens, 0)
        self.assertEqual(request.cached_tokens, 0)
        self.assertFalse(request.ignore_eos)
        other = Request(8, (11,), 1, 0.0, 1.0, 100)
        self.assertEqual(other.output_token_ids, [])

    def test_batch_plan_has_compact_deltas_and_no_phase_or_full_table(self) -> None:
        delta = TableDelta(start_block=2, physical_blocks=(9, 10))
        row = BatchPlanRow(
            slot=3,
            token_start=128,
            token_ids=(4, 5),
            max_accept_tokens=2,
            sample=True,
            table_delta=delta,
            ignore_eos=True,
            restore_slot=4,
            capture_slot=5,
        )
        plan = BatchPlan(step_id=5, lanes=((row,), ()), graph_bucket=None)

        self.assertEqual(plan.staging_index, 1)
        self.assertEqual(plan.lanes[0][0].table_delta, delta)
        self.assertEqual(
            [field.name for field in fields(BatchPlanRow)],
            [
                "slot",
                "token_start",
                "token_ids",
                "max_accept_tokens",
                "sample",
                "table_delta",
                "ignore_eos",
                "restore_slot",
                "capture_slot",
            ],
        )

    def test_result_identifies_resident_slots_only(self) -> None:
        row = StepResultRow(3, 1, (42,), True)
        result = StepResult(5, ((), (row,)), elapsed_ns=10)

        self.assertEqual(result.lanes[1][0].slot, 3)
        self.assertEqual(result.lanes[1][0].accepted_count, 1)
        self.assertNotIn("request_id", {field.name for field in fields(StepResultRow)})

    def test_batch_plan_binary_schema_round_trips_plan_and_none(self) -> None:
        rows = tuple(
            BatchPlanRow(
                slot=index,
                token_start=128 * index,
                token_ids=(index, index + 1),
                max_accept_tokens=2,
                sample=index == 1,
                table_delta=TableDelta(index, (index + 8,)),
                ignore_eos=index == 0,
                restore_slot=8 if index == 0 else None,
                capture_slot=9 if index == 1 else None,
            )
            for index in range(2)
        )
        plan = BatchPlan(2**32 + 7, (rows[:1], rows[1:]), 64)
        buffer = bytearray(MAX_BATCH_PLAN_BYTES)

        size = encode_batch_plan(plan, buffer)

        self.assertEqual(decode_batch_plan(bytes(buffer[:size])), plan)
        self.assertEqual(encode_batch_plan(None, buffer), 0)
        self.assertIsNone(decode_batch_plan(b""))

    def test_batch_plan_schema_covers_the_four_lane_4096_token_envelope(
        self,
    ) -> None:
        lanes = tuple(
            (
                BatchPlanRow(
                    lane,
                    0,
                    tuple(range(4096)),
                    4096,
                    False,
                    TableDelta(0, tuple(range(33))),
                ),
            )
            for lane in range(4)
        )
        plan = BatchPlan(9, lanes, None)
        buffer = bytearray(MAX_BATCH_PLAN_BYTES)

        size = encode_batch_plan(plan, buffer)

        self.assertGreater(size, 32 * 1024)
        self.assertLess(size, MAX_BATCH_PLAN_BYTES)
        self.assertEqual(decode_batch_plan(bytes(buffer[:size])), plan)

    def test_batch_plan_schema_covers_the_c64_full_restore_envelope(self) -> None:
        blocks = tuple(range(8_066))
        rows = tuple(
            BatchPlanRow(
                slot,
                0,
                tuple(range(128)),
                128,
                False,
                TableDelta(0, blocks),
                restore_slot=64 + slot,
                capture_slot=128 + slot,
            )
            for slot in range(64)
        )
        plan = BatchPlan(9, (rows,), None)
        buffer = bytearray(MAX_BATCH_PLAN_BYTES)

        size = encode_batch_plan(plan, buffer)

        self.assertEqual(size, 2_100_244)
        self.assertEqual(decode_batch_plan(bytes(buffer[:size])), plan)

    def test_batch_plan_binary_schema_rejects_unbounded_or_truncated_data(
        self,
    ) -> None:
        row = BatchPlanRow(
            0,
            0,
            (0,) * 64,
            1,
            False,
            TableDelta(0, ()),
        )
        with self.assertRaisesRegex(ValueError, "binary schema"):
            encode_batch_plan(
                BatchPlan(0, ((row,),), None),
                bytearray(64),
            )
        with self.assertRaisesRegex(ValueError, "transport envelope"):
            decode_batch_plan(bytes(MAX_BATCH_PLAN_BYTES + 1))
        with self.assertRaisesRegex(ValueError, "truncated"):
            decode_batch_plan(b"\x01")


if __name__ == "__main__":
    unittest.main()

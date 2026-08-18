import unittest

from tools.output_parity import compare_case, edit_distance


class OutputParityTest(unittest.TestCase):
    def test_edit_distance_covers_insert_delete_and_replace(self) -> None:
        self.assertEqual(edit_distance([1, 2, 3], [1, 4, 3, 5]), 2)
        self.assertEqual(edit_distance([], [1, 2]), 2)
        self.assertEqual(edit_distance([1, 2], []), 2)

    def test_case_reports_exact_and_tolerant_metrics(self) -> None:
        exact = compare_case("exact", [1, 2], [1, 2])
        changed = compare_case("changed", [1, 2, 3], [1, 4, 3, 5])

        self.assertTrue(exact["exact"])
        self.assertEqual(exact["token_agreement"], 1.0)
        self.assertFalse(changed["exact"])
        self.assertEqual(changed["common_prefix_tokens"], 1)
        self.assertEqual(changed["edit_distance"], 2)
        self.assertEqual(changed["token_agreement"], 0.5)


if __name__ == "__main__":
    unittest.main()

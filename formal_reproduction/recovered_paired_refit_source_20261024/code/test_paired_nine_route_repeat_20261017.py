import unittest
import pandas as pd
from summarise_paired_nine_route_repeat_20261017 import expected_recovery
from aggregate_paired_nine_route_repeats_20261017 import aggregate_tables


class RepeatRecoveryTest(unittest.TestCase):
    def test_both_models_must_recover_reference(self):
        ref = pd.Series(["E/NE/NE"] * 4)
        full = pd.Series(["E/NE/NE", "E/NE/NE", "E/E/NE", "NE/E/NE"])
        excluded = pd.Series(["E/NE/NE", "E/E/NE", "E/NE/NE", "NE/E/NE"])
        self.assertEqual(expected_recovery(ref, full, excluded).tolist(), [True, False, False, False])

    def test_incomplete_repeats_never_mark_final(self):
        one = pd.DataFrame({"homology_group_id": ["a", "b"], "matches_reference_both": [True, False], "repeat": [0, 0]})
        partial, complete = aggregate_tables([one], expected_repeats=2)
        self.assertFalse(complete)
        self.assertTrue(partial.final_refit_stable.isna().all())
        two = pd.DataFrame({"homology_group_id": ["a", "b"], "matches_reference_both": [True, True], "repeat": [1, 1]})
        final, complete = aggregate_tables([one, two], expected_repeats=2)
        self.assertTrue(complete)
        self.assertEqual(final.final_refit_stable.tolist(), [True, False])


if __name__ == "__main__":
    unittest.main()


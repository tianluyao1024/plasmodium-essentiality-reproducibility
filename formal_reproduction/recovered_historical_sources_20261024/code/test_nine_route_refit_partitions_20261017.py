"""Unit tests for paired five-fold resampling and label-scope restrictions."""
import unittest

import numpy as np
import pandas as pd

from nine_route_refit_partitions_20261017 import SCOPES, build_partition, make_scope_pools


class PartitionTests(unittest.TestCase):
    def setUp(self):
        self.pk = "Plasmodium_knowlesi"
        self.pf = "Plasmodium_falciparum"
        self.tg = "Toxoplasma_gondii"
        self.train = pd.DataFrame([
            ("p1", self.pk, "essential"), ("p2", self.pk, "nonessential"),
            ("p3", self.pk, "essential"), ("f1", self.pf, "essential"),
            ("t1", self.tg, "nonessential")], columns=["record_id", "species", "label"])
        self.ober = self.train[self.train.record_id.ne("p3")].copy()
        self.ober.loc[self.ober.record_id.eq("p1"), "label"] = "nonessential"

    def test_paired_sampling_is_deterministic_and_fivefold(self):
        y = np.r_[np.zeros(25, dtype=int), np.ones(25, dtype=int)]
        a = build_partition(y, .8, 123)
        b = build_partition(y, .8, 123)
        np.testing.assert_array_equal(a.selected_indices, b.selected_indices)
        np.testing.assert_array_equal(a.fold_assignment, b.fold_assignment)
        self.assertEqual(len(a.selected_indices), 40)
        self.assertEqual(set(a.fold_assignment), set(range(5)))
        self.assertEqual(len(np.unique(a.selected_indices)), 40)

    def test_target_fold_and_omitted_subsample_excluded(self):
        fit = self.train[self.train.record_id.eq("p1")]
        pools = make_scope_pools(self.pk, fit, self.train, self.ober)
        self.assertEqual(set(pools[SCOPES[0]].record_id), {"p1"})
        self.assertEqual(set(pools[SCOPES[1]].record_id), {"p1", "f1"})
        self.assertEqual(set(pools[SCOPES[2]].record_id), {"p1", "f1", "t1"})
        self.assertEqual(set(pools[SCOPES[3]].record_id), {"p1", "f1", "t1"})
        self.assertEqual(pools[SCOPES[3]].set_index("record_id").loc["p1", "label"], "nonessential")

    def test_outside_training_or_non_pk_label_changes_are_rejected(self):
        fit = self.train[self.train.record_id.eq("p1")]
        bad = pd.concat([self.ober, pd.DataFrame([("p4", self.pk, "essential")], columns=self.train.columns)])
        with self.assertRaises(ValueError):
            make_scope_pools(self.pk, fit, self.train, bad)
        bad = self.ober.copy()
        bad.loc[bad.record_id.eq("f1"), "label"] = "nonessential"
        with self.assertRaises(ValueError):
            make_scope_pools(self.pk, fit, self.train, bad)


if __name__ == "__main__":
    unittest.main()

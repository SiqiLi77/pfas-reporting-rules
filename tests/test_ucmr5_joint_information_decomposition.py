from itertools import permutations
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from analyze_ucmr5_joint_information_decomposition import (
    COUNTS, contributions, evaluation_weights, expected_pairing_losses,
    hierarchical_bootstrap, score_distribution,
)


class JointInformationTests(unittest.TestCase):
    def test_chain_rule_and_singleton_counts(self):
        rng = np.random.default_rng(1)
        raw = rng.dirichlet(np.ones(64), size=64)
        losses, _, _ = score_distribution(raw, np.arange(64))
        np.testing.assert_allclose(losses["pattern_nll"], losses["count_nll"] + losses["identity_given_count_nll"], atol=1e-14)
        self.assertEqual(losses["identity_given_count_nll"][0], 0)
        self.assertEqual(losses["identity_given_count_nll"][-1], 0)

    def test_only_identity_changed_keeps_count_loss(self):
        p = np.full((1, 64), 1 / 64)
        q = p.copy()
        q[0, 1], q[0, 2] = 1.5 / 64, 0.5 / 64
        a, _, _ = score_distribution(p, [1])
        b, _, _ = score_distribution(q, [1])
        self.assertAlmostEqual(a["count_nll"][0], b["count_nll"][0], places=14)
        self.assertGreater(a["identity_given_count_nll"][0], b["identity_given_count_nll"][0])

    def test_only_count_changed_keeps_uniform_identity(self):
        p = np.full((1, 64), 1 / 64)
        q = p * np.where(COUNTS == 2, 2, 1)[None, :]
        q /= q.sum()
        a, _, _ = score_distribution(p, [3])
        b, _, _ = score_distribution(q, [3])
        self.assertAlmostEqual(a["identity_given_count_nll"][0], b["identity_given_count_nll"][0], places=14)
        self.assertGreater(a["count_nll"][0], b["count_nll"][0])

    def test_exact_pairing_equals_enumeration_not_loss_of_mean(self):
        rng = np.random.default_rng(4)
        raw = rng.dirichlet(np.ones(64), size=3)
        codes = np.array([1, 3, 7])
        _, q, count_q = score_distribution(raw, codes)
        expected = expected_pairing_losses(q, count_q, codes, [np.arange(3)])
        enumerated = np.mean([[-np.log(q[p[i], codes[i]]) for i in range(3)] for p in permutations(range(3))], axis=0)
        np.testing.assert_allclose(expected["pattern_nll"], enumerated)
        self.assertTrue(np.all(expected["pattern_nll"] > -np.log(q.mean(axis=0)[codes])))

    def test_pairing_constant_and_singleton_have_zero_gain(self):
        raw = np.full((4, 64), 1 / 64)
        codes = np.array([0, 3, 7, 63])
        losses, q, cq = score_distribution(raw, codes)
        expected = expected_pairing_losses(q, cq, codes, [[0], [1, 2, 3]])
        np.testing.assert_allclose(expected["pattern_nll"], losses["pattern_nll"])

    def meta(self):
        return pd.DataFrame([
            {"location_id": l, "pws_id": p, "state": s, "outer_fold": f, "alpha": a}
            for l, p, s, f in [("a", "p1", "s1", 1), ("b", "p1", "s1", 1), ("c", "p2", "s1", 1), ("d", "p3", "s2", 2)]
            for a in [.1, .2]
        ])

    def test_unbalanced_weights_and_alpha_changing_groups(self):
        meta = self.meta()
        weights = evaluation_weights(meta)["jurisdiction_pws_equal"]
        np.testing.assert_allclose(weights, [.0625, .0625, .0625, .0625, .125, .125, .25, .25])
        values = {"gain": np.arange(8.) - 4}
        labels = np.array([0, 1, 1, 2, 0, 2, 1, 2])
        table = contributions(meta, values, weights, "K", labels)
        self.assertAlmostEqual(table.contribution_to_full_cohort.sum(), weights @ values["gain"])
        np.testing.assert_allclose(table.within_group_mean * table.full_cohort_weight_mass, table.contribution_to_full_cohort)

    def test_bootstrap_constant_and_chain_covariance(self):
        meta = self.meta()
        vectors = {"constant": np.ones(8), "count": np.arange(8.), "identity": -np.arange(8.) / 2, "total": np.arange(8.) / 2}
        summary, draws = hierarchical_bootstrap(meta, vectors, reps=100, seed=1)
        np.testing.assert_allclose(draws[:, 0], 1)
        np.testing.assert_allclose(draws[:, 1] + draws[:, 2], draws[:, 3])
        w = evaluation_weights(meta)["jurisdiction_pws_equal"]
        np.testing.assert_allclose(summary.estimate, [w @ x for x in vectors.values()])

    def test_bad_probability_is_rejected(self):
        with self.assertRaises(ValueError):
            score_distribution(np.zeros((1, 64)), [0])


if __name__ == "__main__":
    unittest.main()

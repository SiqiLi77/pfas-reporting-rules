from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from audit_ucmr5_information_nested_bootstrap import independent_nested_bootstrap, classify_interval
from analyze_ucmr5_joint_information_decomposition import evaluation_weights


class IndependentNestedBootstrapTests(unittest.TestCase):
    def meta(self):
        return pd.DataFrame([
            dict(location_id=l, pws_id=p, state=s, outer_fold=f, alpha=a)
            for l, p, s, f in [("a", "p1", "s1", 1), ("b", "p1", "s1", 1), ("c", "p2", "s1", 1), ("d", "p3", "s2", 2)]
            for a in [.1, .2]
        ])

    def test_weighting_constant_and_replicate_chain_rule(self):
        meta = self.meta()
        x = np.arange(len(meta), dtype=float)
        vectors = {"constant": np.ones(len(meta)), "count": x, "identity": -x / 2, "pattern": x / 2}
        summary, draws = independent_nested_bootstrap(meta, vectors, reps=1000, seed=6)
        w = evaluation_weights(meta)["jurisdiction_pws_equal"]
        np.testing.assert_allclose(summary.estimate, [w @ v for v in vectors.values()])
        np.testing.assert_allclose(draws[:, 0], 1)
        np.testing.assert_allclose(draws[:, 1] + draws[:, 2], draws[:, 3], atol=1e-14)

    def test_repeated_states_get_independent_inner_samples(self):
        # Two identical jurisdictions, each with PWS values [-1,+1].
        # Independent copies give Var(mean of two sampled PWS means)=1/4.
        # Reusing an inner sample for repeated jurisdictions gives 3/8.
        meta = pd.DataFrame([
            dict(location_id=f"l{s}{p}", pws_id=f"p{s}{p}", state=f"s{s}")
            for s in range(2) for p in range(2)
        ])
        _, draws = independent_nested_bootstrap(meta, {"value": np.array([-1., 1., -1., 1.])}, reps=40000, seed=9)
        self.assertLess(abs(float(draws[:, 0].var()) - .25), .008)
        self.assertGreater(abs(float(draws[:, 0].var()) - .375), .10)

    def test_batch_size_does_not_change_rng_or_results(self):
        meta = self.meta()
        vectors = {"value": np.arange(len(meta), dtype=float)}
        a, da = independent_nested_bootstrap(meta, vectors, reps=400, seed=5, batch_size=1)
        b, db = independent_nested_bootstrap(meta, vectors, reps=400, seed=5, batch_size=237)
        np.testing.assert_array_equal(da, db)
        pd.testing.assert_frame_equal(a, b)

    def test_single_pws_states_reduce_to_outer_bootstrap(self):
        meta = pd.DataFrame({"location_id": ["a", "b"], "pws_id": ["pa", "pb"], "state": ["s1", "s2"]})
        _, draws = independent_nested_bootstrap(meta, {"v": np.array([0., 2.])}, reps=1000, seed=3)
        rng = np.random.default_rng(3)
        expected = np.array([0., 2.])[rng.integers(2, size=(1000, 2))].mean(axis=1)
        np.testing.assert_array_equal(draws[:, 0], expected)

    def test_interval_classification_includes_boundary_zero(self):
        np.testing.assert_array_equal(classify_interval([1, -2, -1, 0], [2, -1, 1, 2]), ["positive", "negative", "includes_zero", "includes_zero"])

    def test_invalid_parameters_fail(self):
        meta = self.meta()
        for kwargs in ({"reps": 1}, {"batch_size": 0}):
            with self.assertRaises(ValueError):
                independent_nested_bootstrap(meta, {"v": np.ones(len(meta))}, **kwargs)


if __name__ == "__main__":
    unittest.main()

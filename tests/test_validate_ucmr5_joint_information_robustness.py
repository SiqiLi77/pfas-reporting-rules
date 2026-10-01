import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from validate_ucmr5_joint_information_robustness import (
    COUNTS, EPSILONS, METRICS, cluster_omissions, score_components,
    subgroup_rows, weights_for,
)


def toy_meta():
    rows = []
    for state, pws, locs in (("A", "a1", ("a11", "a12")), ("A", "a2", ("a21",)),
                             ("B", "b1", ("b11",)), ("C", "c1", ("c11", "c12", "c13"))):
        for loc in locs:
            for alpha in (.1, .5):
                rows.append({"state": state, "pws_id": pws, "location_id": loc,
                             "outer_fold": 1, "alpha": alpha,
                             "observed_count": int(alpha > .1), "old_visible_code": int(state == "B")})
    return pd.DataFrame(rows)


def check_chain_rule_and_singletons(epsilon):
    rng = np.random.default_rng(41)
    raw = rng.dirichlet(np.ones(64), size=64)
    raw[1, 1] = 0
    raw /= raw.sum(axis=1, keepdims=True)
    result = score_components(raw, np.arange(64), epsilon)
    np.testing.assert_allclose(result[:, 0], result[:, 1] + result[:, 2], atol=1e-14)
    assert result[0, 2] == result[63, 2] == 0
    q = (1 - epsilon) * raw + epsilon / 64
    for i in range(64):
        count_q = q[i, COUNTS == COUNTS[i]].sum()
        np.testing.assert_allclose(result[i, 1], -np.log(count_q), atol=1e-14)
        np.testing.assert_allclose(result[i, 2], -np.log(q[i, i] / count_q), atol=1e-14)


def check_no_silent_floor(epsilon):
    with np.testing.assert_raises(ValueError):
        score_components(np.ones((1, 64)) / 64, [0], epsilon)


def check_invalid_probability(case):
    q = np.ones((1, 64)) / 64
    codes = [1]
    if case == "negative":
        q[0, 1] = -1
    elif case == "bad_sum":
        q *= 2
    elif case == "nan":
        q[0, 1] = np.nan
    else:
        codes = [64]
    with np.testing.assert_raises(ValueError):
        score_components(q, codes, 1e-12)


def check_hierarchical_weights():
    meta = toy_meta()
    w = weights_for(meta)["jurisdiction_pws_equal"]
    np.testing.assert_allclose(w.sum(), 1)
    np.testing.assert_allclose(w[meta.state.eq("A")].sum(), 1/3)
    np.testing.assert_allclose(w[meta.pws_id.eq("a1")].sum(), 1/6)
    np.testing.assert_allclose(w[meta.location_id.eq("a11")].sum(), 1/12)
    np.testing.assert_allclose(w[meta.pws_id.eq("c1")].sum(), 1/3)


def check_duplicate_and_unequal_thresholds_rejected():
    meta = toy_meta()
    with np.testing.assert_raises(ValueError):
        weights_for(pd.concat([meta, meta.iloc[[0]]]))
    with np.testing.assert_raises(ValueError):
        weights_for(meta.iloc[1:])


def check_cluster_omissions_match_brute_force_including_singleton_jurisdictions():
    meta = toy_meta()
    rng = np.random.default_rng(55)
    two = rng.normal(size=(len(meta), 2))
    delta = np.column_stack([two.sum(axis=1), two])
    loo = cluster_omissions(meta, delta)
    assert len(loo) == (3 + 4) * 2 * 3
    for (level, state, pws), group in loo.groupby(["omission_level", "omitted_state", "omitted_pws_id"], dropna=False):
        keep = meta.state.ne(state) if level == "jurisdiction" else meta.pws_id.ne(pws)
        weight = weights_for(meta.loc[keep])
        for name, w in weight.items():
            expected = w @ delta[keep]
            actual = group[group.weighting.eq(name)].set_index("metric").loc[list(METRICS), "loss_reduction"]
            np.testing.assert_allclose(actual, expected, atol=1e-14)


def check_subgroups_keep_full_denominator():
    meta = toy_meta()
    delta = np.column_stack([np.ones(len(meta)) * 3, np.ones(len(meta)) * 2, np.ones(len(meta))])
    groups = pd.DataFrame(subgroup_rows(meta, delta, "test"))
    for (_, _, metric), part in groups.groupby(["grouping", "weighting", "metric"]):
        np.testing.assert_allclose(part.full_cohort_weight_mass.sum(), 1)
        np.testing.assert_allclose(part.contribution_to_full_cohort.sum(), {"pattern_nll": 3, "count_nll": 2, "identity_given_count_nll": 1}[metric])


class RobustnessTests(unittest.TestCase):
    def test_epsilon_1e8(self): check_chain_rule_and_singletons(1e-8)
    def test_epsilon_1e10(self): check_chain_rule_and_singletons(1e-10)
    def test_epsilon_1e12(self): check_chain_rule_and_singletons(1e-12)
    def test_epsilon_1e14(self): check_chain_rule_and_singletons(1e-14)
    def test_zero_epsilon(self): check_no_silent_floor(0)
    def test_negative_epsilon(self): check_no_silent_floor(-1)
    def test_one_epsilon(self): check_no_silent_floor(1)
    def test_over_one_epsilon(self): check_no_silent_floor(2)
    def test_negative_probability(self): check_invalid_probability("negative")
    def test_bad_normalization(self): check_invalid_probability("bad_sum")
    def test_nan(self): check_invalid_probability("nan")
    def test_bad_code(self): check_invalid_probability("bad_code")
    def test_weights(self): check_hierarchical_weights()
    def test_unequal_and_duplicate_queries(self): check_duplicate_and_unequal_thresholds_rejected()
    def test_omissions(self): check_cluster_omissions_match_brute_force_including_singleton_jurisdictions()
    def test_full_denominator(self): check_subgroups_keep_full_denominator()


if __name__ == "__main__":
    unittest.main()

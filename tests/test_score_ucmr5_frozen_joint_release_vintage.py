#!/usr/bin/env python3

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from score_ucmr5_frozen_joint_release_vintage import (  # noqa: E402
    CHEMICALS,
    SHOCKS,
    average_precision_if_identified,
    rates_from_record,
    raw_from_loading_record,
)


class FrozenJointReleaseVintageTest(unittest.TestCase):
    def test_average_precision_with_ties(self) -> None:
        outcome = np.asarray([1, 0, 1, 0, 0])
        score = np.asarray([0.9, 0.8, 0.8, 0.2, 0.1])
        # At the two positive recall steps, precision is 1 and 2/3.
        self.assertAlmostEqual(average_precision_if_identified(outcome, score), 5 / 6)
        self.assertIsNone(average_precision_if_identified(np.zeros(3), np.arange(3)))
        self.assertIsNone(average_precision_if_identified(np.ones(3), np.arange(3)))

    def test_loading_inverse_round_trip(self) -> None:
        loading = np.asarray(
            [[0.2, 0.3], [0.1, -0.4], [0.3, 0.2], [-0.2, 0.1], [0.4, 0.2], [0.1, 0.5]]
        )
        record = {"loadings": {chemical: loading[i].tolist() for i, chemical in enumerate(CHEMICALS)}}
        raw = raw_from_loading_record(record)
        reconstructed = 0.95 * raw / np.sqrt(1.0 + np.sum(raw * raw, axis=1, keepdims=True))
        np.testing.assert_allclose(reconstructed, loading, atol=1e-14, rtol=0)

    def test_shared_rate_order_is_boolean_code_order(self) -> None:
        names = [
            "_".join(CHEMICALS[index] for index in np.flatnonzero(shock))
            for shock in SHOCKS
        ]
        values = np.arange(1, 64, dtype=float) / np.arange(1, 64, dtype=float).sum()
        record = {"rates": dict(zip(reversed(names), reversed(values)))}
        np.testing.assert_array_equal(rates_from_record(record), values)


if __name__ == "__main__":
    unittest.main()

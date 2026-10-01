#!/usr/bin/env python3
"""Outcome-blind information projection of Ising dependence onto fixed marginals."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


CHEMICALS = ("PFBS", "PFHpA", "PFHxS", "PFNA", "PFOA", "PFOS")
PAIRS = tuple(combinations(range(len(CHEMICALS)), 2))


def enumerate_states() -> np.ndarray:
    return np.asarray(
        [
            [(state >> index) & 1 for index in range(len(CHEMICALS))]
            for state in range(2 ** len(CHEMICALS))
        ],
        dtype=np.float64,
    )


def marginal_preserving_projection(
    target_marginal: np.ndarray,
    interaction: np.ndarray,
    tolerance: float = 1e-10,
    max_cycles: int = 200,
) -> tuple[np.ndarray, dict]:
    """Return max-entropy Ising coupling with exactly fixed Bernoulli marginals.

    The operation uses no outcome labels. Cyclic iterative proportional fitting
    is equivalent to adjusting Ising unary potentials until the requested
    marginals are matched while preserving the learned pairwise interaction.
    """
    states = enumerate_states()
    pair_products = np.stack(
        [states[:, left] * states[:, right] for left, right in PAIRS], axis=1
    )
    target = np.clip(np.asarray(target_marginal, dtype=np.float64), 1e-8, 1 - 1e-8)
    log_base = (
        states[None, :, :] * np.log(target[:, None, :])
        + (1 - states[None, :, :]) * np.log1p(-target[:, None, :])
    ).sum(axis=2)
    energy = log_base + np.asarray(interaction, dtype=np.float64) @ pair_products.T
    energy -= energy.max(axis=1, keepdims=True)
    probability = np.exp(energy)
    probability /= probability.sum(axis=1, keepdims=True)
    cycles = 0
    for cycles in range(1, max_cycles + 1):
        for index in range(len(CHEMICALS)):
            current = probability @ states[:, index]
            factor = (
                states[:, index][None, :]
                * target[:, index, None]
                / np.clip(current[:, None], 1e-15, None)
                + (1 - states[:, index])[None, :]
                * (1 - target[:, index, None])
                / np.clip(1 - current[:, None], 1e-15, None)
            )
            probability *= factor
            probability /= probability.sum(axis=1, keepdims=True)
        mismatch = float(np.max(np.abs(probability @ states - target)))
        if mismatch <= tolerance:
            break
    if mismatch > tolerance:
        raise RuntimeError(
            f"Information projection did not converge: mismatch={mismatch:.3e}"
        )
    return probability, {
        "cycles": cycles,
        "max_marginal_mismatch": mismatch,
        "max_row_sum_error": float(
            np.max(np.abs(probability.sum(axis=1) - 1.0))
        ),
        "finite": bool(np.isfinite(probability).all()),
    }


def evaluate(probability: np.ndarray, test: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    states = enumerate_states()
    detected = test[
        [f"y_native_detect__{chemical}" for chemical in CHEMICALS]
    ].to_numpy(int)
    risk = 1 - test[
        [f"old_visible__{chemical}" for chemical in CHEMICALS]
    ].to_numpy(int)
    all_risk = risk.astype(bool).all(axis=1)
    y = detected[all_risk]
    p = probability[all_risk]
    codes = y @ (2 ** np.arange(len(CHEMICALS)))
    pattern_nll = -np.log(np.clip(p[np.arange(len(y)), codes], 1e-12, 1.0))
    state_counts = states.sum(axis=1)
    observed_counts = y.sum(axis=1)
    count_probability = np.stack(
        [p[:, state_counts == count].sum(axis=1) for count in range(7)], axis=1
    )
    count_cdf = np.cumsum(count_probability, axis=1)[:, :6]
    observed_cdf = (
        observed_counts[:, None] <= np.arange(6)[None, :]
    ).astype(float)
    count_crps = np.sum((count_cdf - observed_cdf) ** 2, axis=1)
    count_brier = np.sum(
        (count_probability - np.eye(7)[observed_counts]) ** 2, axis=1
    )
    marginal_brier = np.mean((p @ states - y) ** 2, axis=0)
    pair_rows = []
    for left, right in PAIRS:
        pair_probability = p @ (states[:, left] * states[:, right])
        pair_y = y[:, left] * y[:, right]
        pair_rows.append(
            {
                "chemical_left": CHEMICALS[left],
                "chemical_right": CHEMICALS[right],
                "brier": float(np.mean((pair_probability - pair_y) ** 2)),
                "positives": int(pair_y.sum()),
            }
        )
    metrics = {
        "all_risk_sites": int(all_risk.sum()),
        "joint_pattern_nll": float(pattern_nll.mean()),
        "count_crps": float(count_crps.mean()),
        "count_brier": float(count_brier.mean()),
        "macro_marginal_brier": float(marginal_brier.mean()),
    }
    return metrics, pd.DataFrame(pair_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-dir", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    independent = np.load(
        args.pilot_dir / "independent_test_joint_predictions.npz", allow_pickle=True
    )
    conditional = np.load(
        args.pilot_dir / "conditional_ising_test_joint_predictions.npz",
        allow_pickle=True,
    )
    for key in ("location_id", "state"):
        if not np.array_equal(independent[key], conditional[key]):
            raise ValueError(f"Prediction row mismatch for {key}")
    states = enumerate_states()
    independent_probability = independent["joint_probability"].astype(np.float64)
    target_marginal = independent_probability @ states
    projected, convergence = marginal_preserving_projection(
        target_marginal, conditional["interaction"]
    )
    cohort = pd.read_csv(
        args.cohort, dtype={"state": str, "region": str, "pws_id": str}
    )
    test = cohort[cohort["state_fold"].eq(0)].copy()
    if not np.array_equal(
        test["location_id"].astype(str).to_numpy(), independent["location_id"].astype(str)
    ):
        raise ValueError("Cohort and prediction location ordering differ")
    reference, reference_pairs = evaluate(independent_probability, test)
    candidate, candidate_pairs = evaluate(projected, test)
    pair_wins = int((candidate_pairs["brier"] < reference_pairs["brier"]).sum())
    criteria = {
        "joint_pattern_nll_better": candidate["joint_pattern_nll"]
        < reference["joint_pattern_nll"],
        "count_crps_better": candidate["count_crps"] < reference["count_crps"],
        "at_least_ten_of_fifteen_pair_brier_better": pair_wins >= 10,
        "marginal_brier_exactly_preserved": abs(
            candidate["macro_marginal_brier"] - reference["macro_marginal_brier"]
        )
        < 1e-12,
        "projection_normalized_and_finite": convergence["max_row_sum_error"]
        < 1e-10
        and convergence["finite"],
    }
    decision = {
        "status": (
            "MARGIN-PRESERVING DEVELOPMENT GO"
            if all(criteria.values())
            else "MARGIN-PRESERVING DEVELOPMENT NO-GO"
        ),
        "interpretation": (
            "Post-pilot architecture diagnostic selected after fold-0 outcomes were seen; "
            "this is development evidence, not confirmation."
        ),
        "criteria": criteria,
        "pair_brier_wins": pair_wins,
        "candidate_metrics": candidate,
        "reference_metrics": reference,
        "projection": convergence,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "margin_preserving_test_joint_predictions.npz",
        location_id=independent["location_id"],
        state=independent["state"],
        joint_probability=projected.astype(np.float32),
    )
    candidate_pairs.assign(model="margin_preserving_ising").to_csv(
        args.output_dir / "pairwise_metrics.csv", index=False
    )
    (args.output_dir / "decision.json").write_text(
        json.dumps(decision, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()

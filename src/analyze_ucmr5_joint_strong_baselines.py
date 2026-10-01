#!/usr/bin/env python3
"""Frozen strong-baseline audit of the six-PFAS conditional joint layer.

The script reuses locked out-of-fold candidate predictions.  It fits a global
pairwise Ising model and a smoothed empirical 64-pattern prior using training
fold labels only.  Every joint distribution is then information-projected to
the exact same site-specific six univariate marginals before scoring.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "work" / "vendor"))

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp

from analyze_ucmr5_margin_preserving_joint import (
    CHEMICALS,
    evaluate,
    enumerate_states,
    marginal_preserving_projection,
)


PAIRS = tuple(combinations(range(len(CHEMICALS)), 2))
FROZEN_FOLDS = (1, 2, 3, 4)
FROZEN_BOOTSTRAP_REPS = 20_000
FROZEN_SEED = 20260812
EMPIRICAL_ALPHA = 0.5
ISING_RIDGE = 1e-4


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_features() -> np.ndarray:
    states = enumerate_states()
    pair_products = np.stack(
        [states[:, left] * states[:, right] for left, right in PAIRS], axis=1
    )
    return np.concatenate([states, pair_products], axis=1)


def all_risk_labels(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    detected = frame[
        [f"y_native_detect__{chemical}" for chemical in CHEMICALS]
    ].to_numpy(int)
    old_visible = frame[
        [f"old_visible__{chemical}" for chemical in CHEMICALS]
    ].to_numpy(int)
    mask = np.all(old_visible == 0, axis=1)
    return detected[mask], mask


def fit_global_ising(labels: np.ndarray) -> tuple[np.ndarray, dict]:
    features = state_features()
    codes = labels @ (2 ** np.arange(len(CHEMICALS)))
    observed_mean = features[codes].mean(axis=0)
    prevalence = np.clip(labels.mean(axis=0), 1e-5, 1 - 1e-5)
    initial = np.zeros(features.shape[1], dtype=float)
    initial[: len(CHEMICALS)] = np.log(prevalence / (1 - prevalence))

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        energy = features @ theta
        log_partition = logsumexp(energy)
        probability = np.exp(energy - log_partition)
        model_mean = probability @ features
        penalty = ISING_RIDGE * float(np.mean(theta**2))
        loss = -float(observed_mean @ theta) + float(log_partition) + penalty
        gradient = (
            model_mean
            - observed_mean
            + (2 * ISING_RIDGE / len(theta)) * theta
        )
        return loss, gradient

    bounds = [(-8.0, 8.0)] * len(CHEMICALS) + [(-3.0, 3.0)] * len(PAIRS)
    result = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-8},
    )
    if not result.success:
        raise RuntimeError(f"Global Ising optimization failed: {result.message}")
    theta = np.asarray(result.x, dtype=float)
    return theta[len(CHEMICALS) :], {
        "complete_training_sites": int(len(labels)),
        "ridge": ISING_RIDGE,
        "iterations": int(result.nit),
        "objective": float(result.fun),
        "unary": theta[: len(CHEMICALS)].tolist(),
        "interaction": theta[len(CHEMICALS) :].tolist(),
        "gradient_max_abs": float(np.max(np.abs(result.jac))),
    }


def fit_empirical_prior(labels: np.ndarray) -> tuple[np.ndarray, dict]:
    codes = labels @ (2 ** np.arange(len(CHEMICALS)))
    counts = np.bincount(codes, minlength=2 ** len(CHEMICALS)).astype(float)
    probability = (counts + EMPIRICAL_ALPHA) / (
        counts.sum() + EMPIRICAL_ALPHA * len(counts)
    )
    return probability, {
        "complete_training_sites": int(len(labels)),
        "alpha": EMPIRICAL_ALPHA,
        "observed_patterns": int(np.sum(counts > 0)),
        "counts": counts.astype(int).tolist(),
        "probability": probability.tolist(),
    }


def project_prior_to_margins(
    target_marginal: np.ndarray,
    prior_probability: np.ndarray,
    tolerance: float = 1e-10,
    max_cycles: int = 500,
) -> tuple[np.ndarray, dict]:
    states = enumerate_states()
    target = np.clip(np.asarray(target_marginal, float), 1e-8, 1 - 1e-8)
    prior = np.asarray(prior_probability, float)
    if prior.ndim == 1:
        probability = np.broadcast_to(prior, (len(target), len(states))).copy()
    elif prior.shape == (len(target), len(states)):
        probability = prior.copy()
    else:
        raise ValueError(
            f"Prior shape {prior.shape} is incompatible with {(len(target), len(states))}"
        )
    probability /= probability.sum(axis=1, keepdims=True)
    mismatch = float("inf")
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
            f"Empirical-prior projection failed to converge: {mismatch:.3e}"
        )
    return probability, {
        "cycles": cycles,
        "max_marginal_mismatch": mismatch,
        "max_row_sum_error": float(
            np.max(np.abs(probability.sum(axis=1) - 1.0))
        ),
        "finite": bool(np.isfinite(probability).all()),
    }


def site_losses(probability: np.ndarray, frame: pd.DataFrame) -> pd.DataFrame:
    states = enumerate_states()
    labels, mask = all_risk_labels(frame)
    p = probability[mask]
    codes = labels @ (2 ** np.arange(len(CHEMICALS)))
    state_counts = states.sum(axis=1)
    observed_counts = labels.sum(axis=1)
    count_probability = np.stack(
        [p[:, state_counts == count].sum(axis=1) for count in range(7)], axis=1
    )
    count_cdf = np.cumsum(count_probability, axis=1)[:, :6]
    observed_cdf = (observed_counts[:, None] <= np.arange(6)[None, :]).astype(float)
    return pd.DataFrame(
        {
            "location_id": frame.loc[mask, "location_id"].astype(str).to_numpy(),
            "state": frame.loc[mask, "state"].astype(str).to_numpy(),
            "pattern_nll": -np.log(
                np.clip(p[np.arange(len(labels)), codes], 1e-12, 1.0)
            ),
            "count_crps": np.sum((count_cdf - observed_cdf) ** 2, axis=1),
        }
    )


def state_equal_bootstrap(
    site_contrast: pd.DataFrame,
    column: str,
    *,
    reps: int = FROZEN_BOOTSTRAP_REPS,
    seed: int = FROZEN_SEED,
) -> dict:
    by_state = site_contrast.groupby("state", sort=True)[column].mean()
    values = by_state.to_numpy(float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(reps, len(values)))
    samples = values[draws].mean(axis=1)
    return {
        "state_equal_mean": float(values.mean()),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
        "bootstrap_probability_positive": float(np.mean(samples > 0)),
        "states": int(len(values)),
        "reps": reps,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--locked-prediction-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    cohort = pd.read_csv(
        args.cohort, dtype={"state": str, "region": str, "pws_id": str}
    )
    states = enumerate_states()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pooled_frames = []
    pooled_probability: dict[str, list[np.ndarray]] = {
        "factorized_same_margins": [],
        "global_pairwise_ising": [],
        "global_empirical_64_pattern": [],
        "conditional_ising": [],
    }
    fit_records = []
    fold_metric_rows = []
    convergence_records = []

    for outer_fold in FROZEN_FOLDS:
        validation_fold = 1 + (outer_fold % 4)
        train = cohort[
            ~cohort["state_fold"].isin([outer_fold, validation_fold])
        ].copy()
        test = cohort[cohort["state_fold"].eq(outer_fold)].copy()
        prediction_path = (
            args.locked_prediction_dir
            / f"fold_{outer_fold}_ensemble_joint_predictions.npz"
        )
        locked = np.load(prediction_path, allow_pickle=True)
        if not np.array_equal(
            test["location_id"].astype(str).to_numpy(),
            locked["location_id"].astype(str),
        ):
            raise ValueError(f"Fold {outer_fold} location ordering differs")

        factorized = locked["factorized_probability"].astype(np.float64)
        factorized /= factorized.sum(axis=1, keepdims=True)
        stored_candidate = locked["coupled_probability"].astype(np.float64)
        target_marginal = factorized @ states
        # The locked archive is float32, while the original information projection
        # passed at float64 precision. Restore the frozen target margins without
        # labels while retaining all stored dependence odds ratios.
        candidate, candidate_convergence = project_prior_to_margins(
            target_marginal, stored_candidate
        )
        train_labels, _ = all_risk_labels(train)
        interaction, ising_fit = fit_global_ising(train_labels)
        global_ising, ising_convergence = marginal_preserving_projection(
            target_marginal,
            np.broadcast_to(interaction, (len(test), len(interaction))),
        )
        empirical_prior, empirical_fit = fit_empirical_prior(train_labels)
        empirical, empirical_convergence = project_prior_to_margins(
            target_marginal, empirical_prior
        )

        fold_models = {
            "factorized_same_margins": factorized,
            "global_pairwise_ising": global_ising,
            "global_empirical_64_pattern": empirical,
            "conditional_ising": candidate,
        }
        for model, probability in fold_models.items():
            metrics, _ = evaluate(probability, test)
            fold_metric_rows.append(
                {"outer_fold": outer_fold, "model": model, **metrics}
            )
            pooled_probability[model].append(probability)
        pooled_frames.append(test)
        fit_records.append(
            {
                "outer_fold": outer_fold,
                "validation_fold": validation_fold,
                "global_ising": ising_fit,
                "global_empirical_64_pattern": empirical_fit,
            }
        )
        convergence_records.extend(
            [
                {
                    "outer_fold": outer_fold,
                    "model": "conditional_ising_float64_margin_restore",
                    **candidate_convergence,
                },
                {
                    "outer_fold": outer_fold,
                    "model": "global_pairwise_ising",
                    **ising_convergence,
                },
                {
                    "outer_fold": outer_fold,
                    "model": "global_empirical_64_pattern",
                    **empirical_convergence,
                },
            ]
        )
        print(
            f"fold={outer_fold} train_complete={len(train_labels)} "
            f"ising_cycles={ising_convergence['cycles']} "
            f"empirical_cycles={empirical_convergence['cycles']}",
            flush=True,
        )

    pooled_frame = pd.concat(pooled_frames, ignore_index=True)
    pooled = {
        model: np.concatenate(parts, axis=0)
        for model, parts in pooled_probability.items()
    }
    metrics = {}
    pair_parts = []
    losses = {}
    for model, probability in pooled.items():
        model_metrics, pairs = evaluate(probability, pooled_frame)
        metrics[model] = model_metrics
        pairs.insert(0, "model", model)
        pair_parts.append(pairs)
        losses[model] = site_losses(probability, pooled_frame)

    candidate_name = "conditional_ising"
    baselines = ("global_pairwise_ising", "global_empirical_64_pattern")
    contrast_parts = []
    comparisons = {}
    pair_table = pd.concat(pair_parts, ignore_index=True)
    pair_pivot = pair_table.pivot(
        index=["chemical_left", "chemical_right"], columns="model", values="brier"
    )
    for baseline in baselines:
        contrast = losses[baseline][["location_id", "state"]].copy()
        if not np.array_equal(
            losses[baseline]["location_id"].to_numpy(),
            losses[candidate_name]["location_id"].to_numpy(),
        ):
            raise ValueError(f"Loss row mismatch for {baseline}")
        contrast["baseline"] = baseline
        contrast["joint_pattern_nll_improvement"] = (
            losses[baseline]["pattern_nll"]
            - losses[candidate_name]["pattern_nll"]
        )
        contrast["count_crps_improvement"] = (
            losses[baseline]["count_crps"] - losses[candidate_name]["count_crps"]
        )
        contrast_parts.append(contrast)
        comparisons[baseline] = {
            "joint_pattern_nll_improvement": state_equal_bootstrap(
                contrast, "joint_pattern_nll_improvement"
            ),
            "count_crps_improvement": state_equal_bootstrap(
                contrast, "count_crps_improvement"
            ),
            "pair_brier_wins": int(
                (pair_pivot[candidate_name] < pair_pivot[baseline]).sum()
            ),
        }

    target_marginal = pooled["factorized_same_margins"] @ states
    marginal_mismatch = {
        model: float(np.max(np.abs(probability @ states - target_marginal)))
        for model, probability in pooled.items()
    }
    criteria = {}
    for baseline in baselines:
        short = (
            "global_ising"
            if baseline == "global_pairwise_ising"
            else "empirical_64_pattern"
        )
        criteria[f"joint_nll_better_than_{short}"] = (
            metrics[candidate_name]["joint_pattern_nll"]
            < metrics[baseline]["joint_pattern_nll"]
        )
        criteria[f"count_crps_better_than_{short}"] = (
            metrics[candidate_name]["count_crps"] < metrics[baseline]["count_crps"]
        )
        criteria[f"joint_nll_state_ci_low_positive_vs_{short}"] = (
            comparisons[baseline]["joint_pattern_nll_improvement"]["ci95_low"] > 0
        )
        criteria[f"count_crps_state_ci_low_positive_vs_{short}"] = (
            comparisons[baseline]["count_crps_improvement"]["ci95_low"] > 0
        )
        criteria[f"at_least_ten_pair_brier_wins_vs_{short}"] = (
            comparisons[baseline]["pair_brier_wins"] >= 10
        )
    criteria["all_univariate_margins_preserved"] = all(
        value < 1e-8 for value in marginal_mismatch.values()
    )
    criteria["all_probabilities_normalized_and_finite"] = all(
        np.isfinite(probability).all()
        and float(np.max(np.abs(probability.sum(axis=1) - 1.0))) < 1e-10
        for probability in pooled.values()
    )

    decision = {
        "status": (
            "JOINT STRONG-BASELINE GO"
            if all(criteria.values())
            else "JOINT STRONG-BASELINE NO-GO"
        ),
        "evidence_class": (
            "Frozen post-development comparison on locked folds 1-4; global "
            "baselines fit with training-fold complete hidden states only."
        ),
        "criteria": criteria,
        "metrics": metrics,
        "comparisons": comparisons,
        "maximum_marginal_probability_difference": marginal_mismatch,
        "folds": FROZEN_FOLDS,
        "bootstrap_reps": FROZEN_BOOTSTRAP_REPS,
        "bootstrap_seed": FROZEN_SEED,
    }
    pd.DataFrame(fold_metric_rows).to_csv(
        args.output_dir / "per_fold_metrics.csv", index=False
    )
    pair_table.to_csv(args.output_dir / "pairwise_metrics.csv", index=False)
    pd.concat(contrast_parts, ignore_index=True).to_csv(
        args.output_dir / "site_level_contrasts.csv.gz", index=False
    )
    pd.DataFrame(convergence_records).to_csv(
        args.output_dir / "projection_convergence.csv", index=False
    )
    (args.output_dir / "fit_records.json").write_text(
        json.dumps(fit_records, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "decision.json").write_text(
        json.dumps(decision, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "run_info.json").write_text(
        json.dumps(
            {
                "cohort_sha256": sha256(args.cohort),
                "prediction_files": {
                    str(fold): sha256(
                        args.locked_prediction_dir
                        / f"fold_{fold}_ensemble_joint_predictions.npz"
                    )
                    for fold in FROZEN_FOLDS
                },
                "frozen_protocol": (
                    "outputs/UCMR5_JOINT_STRONG_BASELINE_FROZEN_PROTOCOL_20260812_ZH.md"
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()

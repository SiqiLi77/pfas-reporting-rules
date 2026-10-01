#!/usr/bin/env python3
"""Frozen prototype of monotone multi-threshold six-PFAS joint discovery curves."""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "work" / "vendor"))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from scipy.special import log_ndtr

from analyze_ucmr5_joint_strong_baselines import (
    fit_empirical_prior,
    fit_global_ising,
    project_prior_to_margins,
    state_equal_bootstrap,
)
from analyze_ucmr5_margin_preserving_joint import (
    CHEMICALS,
    enumerate_states,
    marginal_preserving_projection,
)


ALPHAS = (0.0, 0.25, 0.50, 0.75, 1.0)
OUTER_FOLDS = (1, 2, 3, 4)
BOOTSTRAP_REPS = 20_000
BOOTSTRAP_SEED = 20260812
NUMERICAL_TOLERANCE = 1e-8
DIAGNOSTIC_TOLERANCE = 1e-10
LOG_SCORE_EPSILON = 1e-12
PAIRS = tuple(combinations(range(len(CHEMICALS)), 2))
MODELS = ("factorized", "global_ising_by_threshold", "empirical_64_by_threshold")


def log_difference(log_b: np.ndarray, log_a: np.ndarray) -> np.ndarray:
    ratio = np.minimum(log_a - log_b, -1e-12)
    return log_b + np.log(-np.expm1(ratio))


def observed_labels(frame: pd.DataFrame, alpha: float) -> np.ndarray:
    labels = np.zeros((len(frame), len(CHEMICALS)), dtype=int)
    for index, chemical in enumerate(CHEMICALS):
        lower = frame[f"native_logmrl__{chemical}"].to_numpy(float)
        upper = frame[f"old_logmrl__{chemical}"].to_numpy(float)
        threshold = lower + alpha * (upper - lower)
        value = np.exp(lower) * np.expm1(
            frame[f"y_native_logratio__{chemical}"].to_numpy(float)
        )
        detected = frame[f"y_native_detect__{chemical}"].to_numpy(bool)
        log_value = np.log(np.maximum(value, 1e-30))
        labels[:, index] = (detected & (log_value >= threshold - 1e-12)).astype(int)
    return labels


def prediction_lookup(
    predictions: pd.DataFrame, test: pd.DataFrame
) -> pd.DataFrame:
    expected = pd.MultiIndex.from_product(
        [test["location_id"].astype(str), CHEMICALS],
        names=["location_id", "chemical"],
    )
    selected = predictions[
        predictions["location_id"].astype(str).isin(
            test["location_id"].astype(str)
        )
    ].copy()
    selected["location_id"] = selected["location_id"].astype(str)
    if selected.duplicated(["location_id", "chemical"]).any():
        raise ValueError("Duplicate distribution prediction rows")
    selected = selected.set_index(["location_id", "chemical"]).reindex(expected)
    if selected[["mu", "sigma", "probability_hidden"]].isna().any().any():
        raise ValueError("Missing distribution predictions after alignment")
    return selected


def threshold_margins(
    test: pd.DataFrame, lookup: pd.DataFrame, alpha: float
) -> np.ndarray:
    marginal = np.zeros((len(test), len(CHEMICALS)), dtype=float)
    for index, chemical in enumerate(CHEMICALS):
        rows = lookup.xs(chemical, level="chemical")
        lower = test[f"native_logmrl__{chemical}"].to_numpy(float)
        upper = test[f"old_logmrl__{chemical}"].to_numpy(float)
        threshold = lower + alpha * (upper - lower)
        old_visible = test[f"old_visible__{chemical}"].to_numpy(bool)
        occurrence = rows["probability_hidden"].to_numpy(float)
        mu = rows["mu"].to_numpy(float)
        sigma = rows["sigma"].to_numpy(float)
        log_upper = log_ndtr((upper - mu) / sigma)
        log_lower = log_ndtr((lower - mu) / sigma)
        log_threshold = log_ndtr((threshold - mu) / sigma)
        log_full_interval = log_difference(log_upper, log_lower)
        log_tail_interval = log_difference(log_upper, log_threshold)
        conditional_tail = np.exp(log_tail_interval - log_full_interval)
        conditional_tail = np.where(alpha >= 1.0, 0.0, conditional_tail)
        hidden_probability = occurrence * np.clip(conditional_tail, 0.0, 1.0)
        marginal[:, index] = np.where(old_visible, 1.0, hidden_probability)
    return np.clip(marginal, 1e-8, 1 - 1e-8)


def factorized_probability(marginal: np.ndarray) -> np.ndarray:
    states = enumerate_states()
    log_probability = (
        states[None, :, :] * np.log(marginal[:, None, :])
        + (1 - states[None, :, :]) * np.log1p(-marginal[:, None, :])
    ).sum(axis=2)
    log_probability -= log_probability.max(axis=1, keepdims=True)
    probability = np.exp(log_probability)
    return probability / probability.sum(axis=1, keepdims=True)


def evaluate_threshold(
    probability: np.ndarray,
    labels: np.ndarray,
    frame: pd.DataFrame,
    *,
    model: str,
    alpha: float,
    outer_fold: int,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    states = enumerate_states().astype(int)
    codes = labels @ (2 ** np.arange(len(CHEMICALS)))
    if np.min(probability) < -DIAGNOSTIC_TOLERANCE:
        raise ValueError(f"Materially negative state probability: {probability.min()}")
    cleaned = np.maximum(probability, 0.0)
    cleaned /= cleaned.sum(axis=1, keepdims=True)
    observed_raw = cleaned[np.arange(len(labels)), codes]
    # A fixed uniform-contamination mixture is itself a valid predictive
    # distribution.  This replaces the former eventwise probability floor,
    # which did not define a normalized distribution and therefore was not a
    # strictly proper log-score implementation.
    scored_probability = (
        (1.0 - LOG_SCORE_EPSILON) * cleaned + LOG_SCORE_EPSILON / len(states)
    )
    pattern_nll = -np.log(scored_probability[np.arange(len(labels)), codes])
    state_counts = states.sum(axis=1)
    observed_count = labels.sum(axis=1)
    count_probability = np.stack(
        [probability[:, state_counts == count].sum(axis=1) for count in range(7)],
        axis=1,
    )
    count_cdf = np.cumsum(count_probability, axis=1)[:, :6]
    observed_cdf = (observed_count[:, None] <= np.arange(6)[None, :]).astype(float)
    count_crps = np.sum((count_cdf - observed_cdf) ** 2, axis=1)
    count_brier = np.sum(
        (count_probability - np.eye(7)[observed_count]) ** 2, axis=1
    )
    k2_probability = probability[:, state_counts >= 2].sum(axis=1)
    k2_outcome = (observed_count >= 2).astype(float)
    k2_brier = (k2_probability - k2_outcome) ** 2
    pair_rows = []
    for left, right in PAIRS:
        pair_probability = probability @ (states[:, left] * states[:, right])
        pair_y = labels[:, left] * labels[:, right]
        pair_rows.append(
            {
                "outer_fold": outer_fold,
                "alpha": alpha,
                "model": model,
                "chemical_left": CHEMICALS[left],
                "chemical_right": CHEMICALS[right],
                "brier": float(np.mean((pair_probability - pair_y) ** 2)),
                "positives": int(pair_y.sum()),
            }
        )
    site = pd.DataFrame(
        {
            "location_id": frame["location_id"].astype(str).to_numpy(),
            "state": frame["state"].astype(str).to_numpy(),
            "outer_fold": outer_fold,
            "alpha": alpha,
            "model": model,
            "pattern_nll": pattern_nll,
            "count_crps": count_crps,
            "count_brier": count_brier,
            "k2_brier": k2_brier,
            "k2_probability": k2_probability,
            "k2_outcome": k2_outcome,
        }
    )
    metrics = {
        "outer_fold": outer_fold,
        "alpha": alpha,
        "model": model,
        "sites": int(len(labels)),
        "observed_k2_rate": float(k2_outcome.mean()),
        "pattern_nll": float(pattern_nll.mean()),
        "count_crps": float(count_crps.mean()),
        "count_brier": float(count_brier.mean()),
        "k2_brier": float(k2_brier.mean()),
        "macro_marginal_brier": float(
            np.mean((probability @ states - labels) ** 2)
        ),
        "log_score_uniform_contamination_epsilon": LOG_SCORE_EPSILON,
        "raw_zero_probability_events": int(np.sum(observed_raw == 0.0)),
        "negative_entries_cleaned_within_tolerance": int(np.sum(probability < 0.0)),
    }
    return metrics, site, pd.DataFrame(pair_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--distribution-predictions", type=Path, required=True)
    parser.add_argument("--prediction-model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    cohort = pd.read_csv(
        args.cohort, dtype={"state": str, "region": str, "pws_id": str}
    )
    prediction_columns = [
        "location_id",
        "chemical",
        "model",
        "mu",
        "sigma",
        "probability_hidden",
    ]
    predictions = pd.read_csv(
        args.distribution_predictions,
        usecols=prediction_columns,
        dtype={"location_id": str},
    )
    predictions = predictions[predictions["model"].eq(args.prediction_model)].copy()
    if len(predictions) != len(cohort) * len(CHEMICALS):
        raise ValueError(
            f"Expected {len(cohort) * len(CHEMICALS)} predictions, got {len(predictions)}"
        )

    metric_rows = []
    site_parts = []
    pair_parts = []
    convergence_rows = []
    curve_probability: dict[tuple[int, str], list[np.ndarray]] = {}
    maximum_margin_mismatch = {model: 0.0 for model in MODELS}

    for outer_fold in OUTER_FOLDS:
        validation_fold = 1 + (outer_fold % 4)
        train = cohort[
            ~cohort["state_fold"].isin([outer_fold, validation_fold])
        ].copy()
        test = cohort[cohort["state_fold"].eq(outer_fold)].copy()
        lookup = prediction_lookup(predictions, test)
        for alpha in ALPHAS:
            train_labels = observed_labels(train, alpha)
            test_labels = observed_labels(test, alpha)
            marginal = threshold_margins(test, lookup, alpha)
            factorized = factorized_probability(marginal)

            interaction, ising_fit = fit_global_ising(train_labels)
            ising, ising_convergence = marginal_preserving_projection(
                marginal,
                np.broadcast_to(interaction, (len(test), len(interaction))),
            )
            empirical_prior, empirical_fit = fit_empirical_prior(train_labels)
            empirical, empirical_convergence = project_prior_to_margins(
                marginal, empirical_prior
            )
            models = {
                "factorized": factorized,
                "global_ising_by_threshold": ising,
                "empirical_64_by_threshold": empirical,
            }
            for model, probability in models.items():
                metrics, site, pairs = evaluate_threshold(
                    probability,
                    test_labels,
                    test,
                    model=model,
                    alpha=alpha,
                    outer_fold=outer_fold,
                )
                metric_rows.append(metrics)
                site_parts.append(site)
                pair_parts.append(pairs)
                curve_probability.setdefault((outer_fold, model), []).append(
                    site["k2_probability"].to_numpy(float)
                )
                maximum_margin_mismatch[model] = max(
                    maximum_margin_mismatch[model],
                    float(np.max(np.abs(probability @ enumerate_states() - marginal))),
                )
            convergence_rows.extend(
                [
                    {
                        "outer_fold": outer_fold,
                        "alpha": alpha,
                        "model": "global_ising_by_threshold",
                        **ising_convergence,
                        "training_sites": len(train_labels),
                        "fit_iterations": ising_fit["iterations"],
                    },
                    {
                        "outer_fold": outer_fold,
                        "alpha": alpha,
                        "model": "empirical_64_by_threshold",
                        **empirical_convergence,
                        "training_sites": len(train_labels),
                        "observed_training_patterns": empirical_fit["observed_patterns"],
                    },
                ]
            )
            print(
                f"fold={outer_fold} alpha={alpha:.2f} "
                f"observed_k2={np.mean(test_labels.sum(axis=1) >= 2):.6f}",
                flush=True,
            )

    metrics = pd.DataFrame(metric_rows)
    sites = pd.concat(site_parts, ignore_index=True)
    pairs = pd.concat(pair_parts, ignore_index=True)
    integrated = (
        sites.groupby(["model", "location_id", "state"], as_index=False)[
            ["pattern_nll", "count_crps", "count_brier", "k2_brier"]
        ]
        .mean()
    )
    integrated_metrics = (
        integrated.groupby("model", as_index=False)[
            ["pattern_nll", "count_crps", "count_brier", "k2_brier"]
        ]
        .mean()
    )
    metric_lookup = integrated_metrics.set_index("model")
    comparisons = {}
    criteria = {}
    for candidate in ("global_ising_by_threshold", "empirical_64_by_threshold"):
        reference = integrated[integrated["model"].eq("factorized")].sort_values(
            "location_id"
        )
        alternative = integrated[integrated["model"].eq(candidate)].sort_values(
            "location_id"
        )
        if not np.array_equal(
            reference["location_id"].to_numpy(), alternative["location_id"].to_numpy()
        ):
            raise ValueError(f"Integrated row mismatch for {candidate}")
        contrast = reference[["location_id", "state"]].copy()
        for endpoint in ("pattern_nll", "count_crps", "count_brier", "k2_brier"):
            contrast[f"{endpoint}_improvement"] = (
                reference[endpoint].to_numpy() - alternative[endpoint].to_numpy()
            )
        intervals = {
            endpoint: state_equal_bootstrap(
                contrast,
                f"{endpoint}_improvement",
                reps=BOOTSTRAP_REPS,
                seed=BOOTSTRAP_SEED,
            )
            for endpoint in ("pattern_nll", "count_crps", "k2_brier")
        }
        threshold_crps_wins = int(
            (
                metrics[metrics["model"].eq(candidate)]
                .groupby("alpha")["count_crps"]
                .mean()
                < metrics[metrics["model"].eq("factorized")]
                .groupby("alpha")["count_crps"]
                .mean()
            ).sum()
        )
        comparisons[candidate] = {
            "intervals": intervals,
            "threshold_count_crps_wins": threshold_crps_wins,
        }
        criteria[candidate] = {
            "integrated_pattern_nll_better": bool(
                metric_lookup.loc[candidate, "pattern_nll"]
                < metric_lookup.loc["factorized", "pattern_nll"]
            ),
            "integrated_count_crps_better": bool(
                metric_lookup.loc[candidate, "count_crps"]
                < metric_lookup.loc["factorized", "count_crps"]
            ),
            "integrated_k2_brier_better": bool(
                metric_lookup.loc[candidate, "k2_brier"]
                < metric_lookup.loc["factorized", "k2_brier"]
            ),
            "pattern_nll_state_ci_low_positive": intervals["pattern_nll"][
                "ci95_low"
            ]
            > 0,
            "count_crps_state_ci_low_positive": intervals["count_crps"][
                "ci95_low"
            ]
            > 0,
            "at_least_four_threshold_crps_wins": threshold_crps_wins >= 4,
            "margins_preserved": maximum_margin_mismatch[candidate] < 1e-8,
        }

    monotonicity = {}
    for model in MODELS:
        violations = 0
        diagnostic_flags = 0
        maximum_increase = 0.0
        comparisons_count = 0
        for outer_fold in OUTER_FOLDS:
            matrix = np.stack(curve_probability[(outer_fold, model)], axis=1)
            increase = np.diff(matrix, axis=1)
            violations += int(np.sum(increase > NUMERICAL_TOLERANCE))
            diagnostic_flags += int(np.sum(increase > DIAGNOSTIC_TOLERANCE))
            maximum_increase = max(maximum_increase, float(np.max(increase)))
            comparisons_count += increase.size
        monotonicity[model] = {
            "k2_curve_violations": violations,
            "diagnostic_flags_above_1e-10": diagnostic_flags,
            "comparisons": comparisons_count,
            "maximum_increase": maximum_increase,
            "frozen_numerical_tolerance": NUMERICAL_TOLERANCE,
        }
    marginal_monotonicity_violations = 0
    # The analytic tail construction is checked independently of joint fitting.
    for outer_fold in OUTER_FOLDS:
        test = cohort[cohort["state_fold"].eq(outer_fold)].copy()
        lookup = prediction_lookup(predictions, test)
        matrix = np.stack(
            [threshold_margins(test, lookup, alpha) for alpha in ALPHAS], axis=2
        )
        marginal_monotonicity_violations += int(np.sum(np.diff(matrix, axis=2) > 1e-12))

    probability_valid = all(
        record["max_row_sum_error"] < 1e-10 and bool(record["finite"])
        for record in convergence_rows
    )
    go_models = [
        model
        for model, model_criteria in criteria.items()
        if all(model_criteria.values())
        and monotonicity[model]["k2_curve_violations"] == 0
    ]
    decision = {
        "status": (
            "MULTITHRESHOLD JOINT CURVE PROTOTYPE GO"
            if go_models and marginal_monotonicity_violations == 0 and probability_valid
            else "MULTITHRESHOLD JOINT CURVE PROTOTYPE NO-GO"
        ),
        "evidence_class": (
            "Frozen prototype using existing out-of-fold censored-distribution "
            "predictions and training-fold-only dependence skeletons."
        ),
        "go_models": go_models,
        "criteria": criteria,
        "integrated_metrics": integrated_metrics.to_dict(orient="records"),
        "comparisons": comparisons,
        "maximum_marginal_probability_difference": maximum_margin_mismatch,
        "marginal_monotonicity_violations": marginal_monotonicity_violations,
        "k2_curve_monotonicity": monotonicity,
        "probabilities_normalized_and_finite": probability_valid,
        "alphas": ALPHAS,
        "outer_folds": OUTER_FOLDS,
        "bootstrap_reps": BOOTSTRAP_REPS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "prediction_model": args.prediction_model,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output_dir / "threshold_metrics.csv", index=False)
    integrated_metrics.to_csv(args.output_dir / "integrated_metrics.csv", index=False)
    pairs.to_csv(args.output_dir / "pairwise_metrics.csv", index=False)
    sites.to_csv(args.output_dir / "site_threshold_losses.csv.gz", index=False)
    pd.DataFrame(convergence_rows).to_csv(
        args.output_dir / "projection_convergence.csv", index=False
    )
    (args.output_dir / "decision.json").write_text(
        json.dumps(decision, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()

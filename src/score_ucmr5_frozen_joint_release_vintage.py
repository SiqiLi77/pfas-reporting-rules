#!/usr/bin/env python3
"""Score frozen January joint models on later-added UCMR 5 events.

The script has two deliberately separate stages:

1. Reconstruct the joint distributions from the parameters serialized by the
   January analysis and replay them against the archived January site losses.
2. Without fitting or selecting anything, apply the same fold-specific
   parameters to records first present in the final release.

Only outer folds 1--4 are used because those were the folds in the frozen
conditional-joint analysis.  Fold 0 remains the architecture-development fold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "work" / "vendor"))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import torch

from analyze_ucmr5_conditional_gaussian_factor import GaussianFactorKernel
from analyze_ucmr5_conditional_joint_models import (
    CHEMICALS,
    QUERIES,
    SHOCKS,
    labels_at,
    margins_at,
    shock_probability,
)
from analyze_ucmr5_margin_preserving_joint import enumerate_states
from analyze_ucmr5_multithreshold_joint_curve import (
    evaluate_threshold,
    factorized_probability,
    prediction_lookup,
)


FORMAL_FOLDS = (1, 2, 3, 4)
MODELS = (
    "factorized",
    "conditional_shared_shock",
    "conditional_gaussian_factor2",
)
LOSS_COLUMNS = ("pattern_nll", "count_crps", "count_brier", "k2_brier")
REPLAY_COLUMNS = (
    "pattern_nll",
    "count_crps",
    "count_brier",
    "k2_brier",
    "k2_probability",
    "k2_outcome",
    "observed_state_probability_raw",
)
SHARED_REPLAY_TOLERANCE = 1e-10
GAUSSIAN_REPLAY_TOLERANCE = 2e-6


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def decision_records(path: Path, key: str) -> dict[int, dict]:
    decision = load_json(path)
    records = {
        int(record["outer_fold"]): record[key]
        for record in decision["fit_records"]
    }
    if set(records) != set(FORMAL_FOLDS):
        raise ValueError(f"{path} does not contain exactly outer folds 1--4")
    return records


def rates_from_record(record: dict) -> np.ndarray:
    stored = record["rates"]
    names = [
        "_".join(CHEMICALS[index] for index in np.flatnonzero(shock))
        for shock in SHOCKS
    ]
    if set(names) != set(stored):
        missing = sorted(set(names) - set(stored))
        extra = sorted(set(stored) - set(names))
        raise ValueError(f"Shared-shock parameter mismatch: missing={missing}, extra={extra}")
    rates = np.asarray([stored[name] for name in names], dtype=float)
    if rates.shape != (63,) or not np.isfinite(rates).all() or np.any(rates <= 0):
        raise ValueError("Invalid frozen shared-shock rates")
    return rates


def raw_from_loading_record(record: dict, maximum_loading_norm: float = 0.95) -> np.ndarray:
    loading = np.asarray([record["loadings"][chemical] for chemical in CHEMICALS], dtype=float)
    if loading.shape != (6, 2) or not np.isfinite(loading).all():
        raise ValueError("Invalid frozen Gaussian-factor loadings")
    residual = maximum_loading_norm**2 - np.sum(loading * loading, axis=1)
    if np.any(residual <= 0):
        raise ValueError("Frozen loading norm is outside the Gaussian kernel parameter space")
    return loading / np.sqrt(residual)[:, None]


def average_precision_if_identified(outcome: np.ndarray, score: np.ndarray) -> float | None:
    outcome = np.asarray(outcome, dtype=int)
    score = np.asarray(score, dtype=float)
    positives = int(outcome.sum())
    negatives = int(len(outcome) - positives)
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(-score, kind="mergesort")
    y = outcome[order]
    s = score[order]
    distinct_end = np.r_[np.flatnonzero(np.diff(s) != 0), len(s) - 1]
    true_positive = np.cumsum(y)[distinct_end]
    false_positive = (distinct_end + 1) - true_positive
    recall = true_positive / positives
    precision = true_positive / (true_positive + false_positive)
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def gaussian_probability(
    kernel: GaussianFactorKernel,
    marginal: np.ndarray,
    raw: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    return kernel.full_probability(marginal, raw, batch_size)


def add_probability_details(
    site: pd.DataFrame,
    probability: np.ndarray,
    marginal: np.ndarray,
    labels: np.ndarray,
    frame: pd.DataFrame,
) -> pd.DataFrame:
    states = enumerate_states().astype(int)
    state_counts = states.sum(axis=1)
    codes = labels @ (2 ** np.arange(len(CHEMICALS)))
    count_probability = np.stack(
        [probability[:, state_counts == count].sum(axis=1) for count in range(7)],
        axis=1,
    )
    result = site.copy()
    result["pws_id"] = frame["pws_id"].astype(str).to_numpy()
    result["observed_state_code"] = codes
    result["observed_count"] = labels.sum(axis=1)
    result["observed_state_probability_raw"] = probability[np.arange(len(frame)), codes]
    for index, chemical in enumerate(CHEMICALS):
        result[f"marginal_probability__{chemical}"] = marginal[:, index]
        result[f"observed_report__{chemical}"] = labels[:, index]
    for count in range(7):
        result[f"count_probability__{count}"] = count_probability[:, count]
    for code in range(64):
        result[f"state_probability__{code:02d}"] = probability[:, code]
    return result


def metric_summary(frame: pd.DataFrame, grouping: list[str]) -> pd.DataFrame:
    rows = []
    iterator = frame.groupby(grouping, sort=True, dropna=False)
    for keys, part in iterator:
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(grouping, keys))
        outcome = part["k2_outcome"].to_numpy(int)
        row.update(
            {
                "sites": int(part["location_id"].nunique()),
                "site_threshold_records": int(len(part)),
                "k2_positives": int(outcome.sum()),
                "k2_negatives": int(len(outcome) - outcome.sum()),
                "unique_k2_positive_sites": int(
                    part.loc[part["k2_outcome"].eq(1), "location_id"].nunique()
                ),
                "pattern_nll": float(part["pattern_nll"].mean()),
                "count_crps": float(part["count_crps"].mean()),
                "count_brier": float(part["count_brier"].mean()),
                "k2_brier": float(part["k2_brier"].mean()),
                "k2_average_precision": average_precision_if_identified(
                    outcome, part["k2_probability"].to_numpy(float)
                ),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def align_replay(reference: pd.DataFrame, current: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["location_id", "outer_fold", "alpha", "model"]
    left = reference.sort_values(keys).reset_index(drop=True)
    right = current.sort_values(keys).reset_index(drop=True)
    if len(left) != len(right):
        raise ValueError(f"Replay row count differs: archived={len(left)}, replayed={len(right)}")
    for key in keys:
        if not np.array_equal(left[key].to_numpy(), right[key].to_numpy()):
            raise ValueError(f"Replay key mismatch for {key}")
    return left, right


def replay_audit_row(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    outer_fold: int,
    alpha: float,
    model: str,
) -> dict:
    reference, current = align_replay(reference, current)
    row = {
        "outer_fold": outer_fold,
        "alpha": alpha,
        "model": model,
        "rows": len(current),
        "keys_match": True,
    }
    for column in REPLAY_COLUMNS:
        difference = np.abs(
            reference[column].to_numpy(float) - current[column].to_numpy(float)
        )
        row[f"maximum_absolute_difference__{column}"] = float(difference.max(initial=0.0))
        row[f"mean_absolute_difference__{column}"] = float(difference.mean())
    return row


def score_models_for_fold(
    frame: pd.DataFrame,
    lookup: pd.DataFrame,
    outer_fold: int,
    rates: np.ndarray,
    raw: np.ndarray,
    gaussian_kernel: GaussianFactorKernel,
    gaussian_batch_size: int,
) -> list[pd.DataFrame]:
    outputs = []
    for alpha in QUERIES:
        marginal = margins_at(frame, lookup, alpha, "log_mrl")
        labels = labels_at(frame, alpha, "log_mrl")
        probabilities = {
            "factorized": factorized_probability(marginal),
            "conditional_shared_shock": shock_probability(marginal, rates),
            "conditional_gaussian_factor2": gaussian_probability(
                gaussian_kernel, marginal, raw, gaussian_batch_size
            ),
        }
        for model, probability in probabilities.items():
            _, site, _ = evaluate_threshold(
                probability,
                labels,
                frame,
                model=model,
                alpha=alpha,
                outer_fold=outer_fold,
            )
            outputs.append(add_probability_details(site, probability, marginal, labels, frame))
    return outputs


def replay_january(
    cohort: pd.DataFrame,
    predictions: pd.DataFrame,
    shared_archive: pd.DataFrame,
    gaussian_archive: pd.DataFrame,
    shared_records: dict[int, dict],
    gaussian_records: dict[int, dict],
    gaussian_kernel: GaussianFactorKernel,
    gaussian_batch_size: int,
    sites_per_fold: int,
) -> pd.DataFrame:
    audits = []
    for outer_fold in FORMAL_FOLDS:
        test = cohort[cohort["state_fold"].eq(outer_fold)].copy()
        if sites_per_fold > 0 and sites_per_fold < len(test):
            # Hash selection is deterministic, order-independent, and outcome-blind.
            order = test["location_id"].astype(str).map(
                lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()
            )
            test = test.assign(_replay_order=order).sort_values(
                ["_replay_order", "location_id"]
            ).head(sites_per_fold).drop(columns="_replay_order")
        lookup = prediction_lookup(predictions, test)
        replayed = pd.concat(
            score_models_for_fold(
                test,
                lookup,
                outer_fold,
                rates_from_record(shared_records[outer_fold]),
                raw_from_loading_record(gaussian_records[outer_fold]),
                gaussian_kernel,
                gaussian_batch_size,
            ),
            ignore_index=True,
        )
        for alpha in QUERIES:
            for model in MODELS:
                current = replayed[
                    replayed["alpha"].eq(alpha) & replayed["model"].eq(model)
                ]
                archive = gaussian_archive if model == "conditional_gaussian_factor2" else shared_archive
                reference = archive[
                    archive["outer_fold"].eq(outer_fold)
                    & archive["alpha"].eq(alpha)
                    & archive["model"].eq(model)
                    & archive["location_id"].isin(current["location_id"])
                ]
                audits.append(replay_audit_row(reference, current, outer_fold, alpha, model))
        print(f"January replay fold {outer_fold} complete", flush=True)
    return pd.DataFrame(audits)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prospective-cohort", type=Path, required=True)
    parser.add_argument("--prospective-predictions", type=Path, required=True)
    parser.add_argument("--january-cohort", type=Path, required=True)
    parser.add_argument("--january-predictions", type=Path, required=True)
    parser.add_argument("--january-shared-site-losses", type=Path, required=True)
    parser.add_argument("--january-gaussian-site-losses", type=Path, required=True)
    parser.add_argument("--shared-decision", type=Path, required=True)
    parser.add_argument("--gaussian-decision", type=Path, required=True)
    parser.add_argument("--prediction-model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gaussian-quadrature-order", type=int, default=48)
    parser.add_argument("--gaussian-batch-size", type=int, default=256)
    parser.add_argument(
        "--january-replay-sites-per-fold",
        type=int,
        default=0,
        help="Outcome-blind deterministic replay subset; 0 replays every January formal-fold site.",
    )
    parser.add_argument("--skip-january-replay", action="store_true")
    args = parser.parse_args()

    shared_records = decision_records(args.shared_decision, "shared_shock")
    gaussian_records = decision_records(args.gaussian_decision, "gaussian_factor")
    gaussian_kernel = GaussianFactorKernel(
        rank=2,
        order=args.gaussian_quadrature_order,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        dtype=torch.float64,
    )

    prediction_columns = [
        "location_id", "chemical", "model", "mu", "sigma", "probability_hidden"
    ]
    prospective = pd.read_csv(
        args.prospective_cohort,
        dtype={"location_id": str, "pws_id": str, "state": str, "region": str},
    )
    formal = prospective[prospective["state_fold"].isin(FORMAL_FOLDS)].copy()
    if len(prospective) != 92 or len(formal) != 69:
        raise ValueError(
            f"Frozen release-vintage cohort gate failed: all={len(prospective)}, formal={len(formal)}"
        )
    if formal["location_id"].duplicated().any():
        raise ValueError("Prospective formal cohort contains duplicate location IDs")
    prospective_predictions = pd.read_csv(
        args.prospective_predictions,
        usecols=prediction_columns + ["outer_fold"],
        dtype={"location_id": str},
    )
    prospective_predictions = prospective_predictions[
        prospective_predictions["model"].eq(args.prediction_model)
        & prospective_predictions["outer_fold"].isin(FORMAL_FOLDS)
    ].copy()
    expected = len(formal) * len(CHEMICALS)
    if len(prospective_predictions) != expected:
        raise ValueError(
            f"Prospective prediction coverage failed: expected={expected}, observed={len(prospective_predictions)}"
        )
    expected_fold = formal.set_index("location_id")["state_fold"]
    observed_fold = prospective_predictions.set_index("location_id")["outer_fold"]
    if not np.array_equal(
        observed_fold.sort_index().to_numpy(),
        expected_fold.reindex(observed_fold.index).sort_index().to_numpy(),
    ):
        raise ValueError("External prediction fold does not equal the frozen state fold")

    replay = None
    if not args.skip_january_replay:
        january = pd.read_csv(
            args.january_cohort,
            dtype={"location_id": str, "pws_id": str, "state": str, "region": str},
        )
        january_predictions = pd.read_csv(
            args.january_predictions,
            usecols=prediction_columns,
            dtype={"location_id": str},
        )
        january_predictions = january_predictions[
            january_predictions["model"].eq(args.prediction_model)
        ].copy()
        shared_archive = pd.read_csv(
            args.january_shared_site_losses,
            dtype={"location_id": str, "pws_id": str, "state": str},
        )
        gaussian_archive = pd.read_csv(
            args.january_gaussian_site_losses,
            dtype={"location_id": str, "pws_id": str, "state": str},
        )
        replay = replay_january(
            january,
            january_predictions,
            shared_archive,
            gaussian_archive,
            shared_records,
            gaussian_records,
            gaussian_kernel,
            args.gaussian_batch_size,
            args.january_replay_sites_per_fold,
        )

    prospective_parts = []
    for outer_fold in FORMAL_FOLDS:
        test = formal[formal["state_fold"].eq(outer_fold)].copy()
        lookup = prediction_lookup(prospective_predictions, test)
        prospective_parts.extend(
            score_models_for_fold(
                test,
                lookup,
                outer_fold,
                rates_from_record(shared_records[outer_fold]),
                raw_from_loading_record(gaussian_records[outer_fold]),
                gaussian_kernel,
                args.gaussian_batch_size,
            )
        )
        print(f"Prospective scoring fold {outer_fold} complete", flush=True)
    site_predictions = pd.concat(prospective_parts, ignore_index=True)
    site_predictions = site_predictions.sort_values(
        ["outer_fold", "location_id", "alpha", "model"]
    ).reset_index(drop=True)

    fold_threshold = metric_summary(site_predictions, ["outer_fold", "alpha", "model"])
    threshold = metric_summary(site_predictions, ["alpha", "model"])
    integrated = metric_summary(site_predictions, ["model"])
    threshold_ap = (
        threshold.dropna(subset=["k2_average_precision"])
        .groupby("model", as_index=False)["k2_average_precision"]
        .agg(
            identified_k2_ap_thresholds="count",
            mean_threshold_k2_average_precision="mean",
        )
    )
    integrated = integrated.merge(threshold_ap, on="model", how="left", validate="one_to_one")
    site_integrated = (
        site_predictions.groupby(
            ["location_id", "pws_id", "state", "outer_fold", "model"], as_index=False
        )[list(LOSS_COLUMNS)]
        .mean()
    )
    reference = integrated.set_index("model").loc["factorized"]
    comparison_rows = []
    for model in MODELS[1:]:
        candidate = integrated.set_index("model").loc[model]
        comparison_rows.append(
            {
                "reference": "factorized",
                "candidate": model,
                **{
                    f"{column}_improvement_factorized_minus_candidate": float(
                        reference[column] - candidate[column]
                    )
                    for column in LOSS_COLUMNS
                },
                "k2_average_precision_difference_candidate_minus_factorized": (
                    None
                    if reference["k2_average_precision"] is None
                    or candidate["k2_average_precision"] is None
                    or pd.isna(reference["k2_average_precision"])
                    or pd.isna(candidate["k2_average_precision"])
                    else float(
                        candidate["k2_average_precision"]
                        - reference["k2_average_precision"]
                    )
                ),
                "mean_threshold_k2_average_precision_difference_candidate_minus_factorized": float(
                    candidate["mean_threshold_k2_average_precision"]
                    - reference["mean_threshold_k2_average_precision"]
                ),
            }
        )
    comparisons = pd.DataFrame(comparison_rows)

    replay_status = "SKIPPED"
    replay_maxima = {}
    if replay is not None:
        replay_status = "PASS"
        for model in MODELS:
            rows = replay[replay["model"].eq(model)]
            maximum = float(
                rows[
                    [f"maximum_absolute_difference__{column}" for column in REPLAY_COLUMNS]
                ].to_numpy(float).max()
            )
            replay_maxima[model] = maximum
            tolerance = (
                GAUSSIAN_REPLAY_TOLERANCE
                if model == "conditional_gaussian_factor2"
                else SHARED_REPLAY_TOLERANCE
            )
            if maximum > tolerance or not rows["keys_match"].all():
                replay_status = "FAIL"

    states = enumerate_states()
    state_columns = [f"state_probability__{code:02d}" for code in range(64)]
    probabilities = site_predictions[state_columns].to_numpy(float)
    margins = site_predictions[
        [f"marginal_probability__{chemical}" for chemical in CHEMICALS]
    ].to_numpy(float)
    maximum_normalization_residual = float(
        np.max(np.abs(probabilities.sum(axis=1) - 1.0))
    )
    maximum_margin_residual = {
        model: float(
            np.max(
                np.abs(
                    site_predictions[site_predictions["model"].eq(model)][state_columns]
                    .to_numpy(float)
                    @ states
                    - site_predictions[site_predictions["model"].eq(model)][
                        [f"marginal_probability__{chemical}" for chemical in CHEMICALS]
                    ].to_numpy(float)
                )
            )
        )
        for model in MODELS
    }
    decision = {
        "status": "PASS" if replay_status in ("PASS", "SKIPPED") else "FAIL",
        "analysis_role": (
            "No-retuning release-vintage evaluation on strict events first present in the "
            "final UCMR 5 archive"
        ),
        "scope_boundary": (
            "These are later-published records, not necessarily later-collected samples. "
            "The 69-event evaluation is small and descriptive; model selection and parameter "
            "estimation used only the frozen January archive. Integrated average precision pools "
            "repeated site-threshold records and is descriptive; threshold-specific average "
            "precision is reported only where both outcome classes occur."
        ),
        "formal_outer_folds": list(FORMAL_FOLDS),
        "excluded_development_fold": 0,
        "prospective_coverage": {
            "all_later_added_strict_events": len(prospective),
            "formal_fold_events": len(formal),
            "excluded_fold_0_events": int(len(prospective) - len(formal)),
            "formal_prediction_rows": len(prospective_predictions),
            "formal_site_threshold_model_rows": len(site_predictions),
            "pws": int(formal["pws_id"].nunique()),
            "jurisdictions": int(formal["state"].nunique()),
        },
        "frozen_protocol": {
            "prediction_model": args.prediction_model,
            "threshold_geometry": "log_mrl",
            "query_alphas": list(QUERIES),
            "gaussian_rank": 2,
            "gaussian_evaluation_quadrature_order": args.gaussian_quadrature_order,
            "proper_log_score_uniform_contamination_epsilon": 1e-12,
            "parameters_refit_or_retuned": False,
        },
        "january_replay": {
            "status": replay_status,
            "sites_per_fold": (
                "all" if args.january_replay_sites_per_fold == 0
                else args.january_replay_sites_per_fold
            ),
            "maximum_absolute_difference_by_model": replay_maxima,
            "tolerance": {
                "factorized": SHARED_REPLAY_TOLERANCE,
                "conditional_shared_shock": SHARED_REPLAY_TOLERANCE,
                "conditional_gaussian_factor2": GAUSSIAN_REPLAY_TOLERANCE,
            },
            "gaussian_tolerance_note": (
                "The archived decision serializes transformed float32 loadings rather than the "
                "raw fitted tensor; inverse transformation can introduce sub-micro numerical drift."
            ),
        },
        "integrated_metrics": integrated.where(pd.notna(integrated), None).to_dict(orient="records"),
        "model_contrasts": comparisons.where(pd.notna(comparisons), None).to_dict(orient="records"),
        "probability_audit": {
            "maximum_normalization_residual": maximum_normalization_residual,
            "maximum_supplied_margin_residual_by_model": maximum_margin_residual,
            "all_probabilities_finite": bool(np.isfinite(probabilities).all()),
            "minimum_state_probability": float(probabilities.min()),
        },
        "input_provenance": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in {
                "prospective_cohort": args.prospective_cohort,
                "prospective_predictions": args.prospective_predictions,
                "january_cohort": args.january_cohort,
                "january_predictions": args.january_predictions,
                "january_shared_site_losses": args.january_shared_site_losses,
                "january_gaussian_site_losses": args.january_gaussian_site_losses,
                "shared_decision": args.shared_decision,
                "gaussian_decision": args.gaussian_decision,
            }.items()
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    site_predictions.to_csv(
        args.output_dir / "prospective_joint_predictions.csv.gz",
        index=False,
        compression="gzip",
    )
    fold_threshold.to_csv(args.output_dir / "prospective_fold_threshold_metrics.csv", index=False)
    threshold.to_csv(args.output_dir / "prospective_threshold_metrics.csv", index=False)
    integrated.to_csv(args.output_dir / "prospective_integrated_metrics.csv", index=False)
    site_integrated.to_csv(
        args.output_dir / "prospective_site_integrated_losses.csv.gz",
        index=False,
        compression="gzip",
    )
    comparisons.to_csv(args.output_dir / "prospective_model_contrasts.csv", index=False)
    if replay is not None:
        replay.to_csv(args.output_dir / "january_replay_audit.csv", index=False)
    decision_path = args.output_dir / "decision.json"
    decision_path.write_text(json.dumps(decision, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(decision, indent=2, allow_nan=False), flush=True)
    if decision["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

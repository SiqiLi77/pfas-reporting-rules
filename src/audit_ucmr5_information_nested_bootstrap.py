#!/usr/bin/env python3
"""Independent-inner-draw audit of the frozen joint-information score intervals.

Each sampled jurisdiction copy gets an independent supply-system resample.
The inherited procedure, which reused one inner sample when a jurisdiction
appeared more than once, remains unchanged for provenance. This correction is
specified on algorithmic grounds, without selecting between the resulting CIs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_ucmr5_joint_information_decomposition import BASELINE, KEYS, METRICS, MODELS, NLL, evaluation_weights

ROOT = Path(__file__).resolve().parents[1]
SEED = 20260917
REPLICATES = 20_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def independent_nested_bootstrap(meta, vectors, reps=REPLICATES, seed=SEED, batch_size=128):
    """Resample states, then PWS independently for every sampled state copy.

    The second stage uses different random index arrays for repeated copies of
    a state in the same replicate. Query and location means stay integrated at
    their original levels. Identical resampling applies to all score columns.
    """
    if reps < 2 or batch_size < 1 or not vectors:
        raise ValueError("At least two replicates, a positive batch, and vectors required")
    columns = list(vectors)
    if len(meta) == 0 or any(len(values) != len(meta) for values in vectors.values()):
        raise ValueError("Nonempty metadata and equal-length score vectors required")
    frame = meta[["state", "pws_id", "location_id"]].copy()
    for name, values in vectors.items():
        frame[name] = np.asarray(values, dtype=float)
    if not np.isfinite(frame[columns].to_numpy()).all():
        raise ValueError("Nonfinite score contrast")
    integrated = frame.groupby(["state", "pws_id", "location_id"], sort=True)[columns].mean()
    pws = integrated.groupby(["state", "pws_id"], sort=True).mean()
    grouped = [group.to_numpy() for _, group in pws.groupby(level="state", sort=True)]
    point = np.mean([values.mean(axis=0) for values in grouped], axis=0)
    rng = np.random.default_rng(seed)
    choice = rng.integers(len(grouped), size=(reps, len(grouped)))
    # Each (replicate, sampled-slot) cell is filled once. Repeated states do not
    # share an inner sample, even if their slots belong to the same replicate.
    inner_means = np.empty((reps, len(grouped), len(columns)))
    for state_index, values in enumerate(grouped):
        replicate, slot = np.where(choice == state_index)
        if len(values) == 1:
            inner_means[replicate, slot] = values[0]
            continue
        for start in range(0, len(replicate), batch_size):
            end = min(start + batch_size, len(replicate))
            indices = rng.integers(len(values), size=(end - start, len(values)))
            inner_means[replicate[start:end], slot[start:end]] = values[indices].mean(axis=1)
    draws = inner_means.mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975], axis=0)
    summary = pd.DataFrame({"key": columns, "estimate": point, "ci95_low": low, "ci95_high": high})
    return summary, draws


def classify_interval(lower, upper):
    return np.where(np.asarray(lower) > 0, "positive", np.where(np.asarray(upper) < 0, "negative", "includes_zero"))


def build_vectors(frame):
    vectors = {}
    for reference, candidate in [("factorized", model) for model in MODELS[1:]] + [(BASELINE, model) for model in MODELS]:
        metrics = NLL if reference == "factorized" else METRICS
        for metric in metrics:
            vectors[f"{reference}|{candidate}|{metric}"] = (
                frame[f"{reference}__{metric}"] - frame[f"{candidate}__{metric}"]
            ).to_numpy()
    if len(vectors) != 24:
        raise ValueError("Expected exactly 24 prespecified score contrasts")
    return vectors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=ROOT / "outputs/joint_information_decomposition_20260916/analysis")
    parser.add_argument("--bootstrap-reps", type=int, default=REPLICATES)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    record_path = args.analysis_dir / "record_scores.csv.gz"
    inherited_path = args.analysis_dir / "paired_contrasts_primary.csv"
    columns = KEYS + [f"{model}__{metric}" for model in (*MODELS, BASELINE) for metric in METRICS]
    frame = pd.read_csv(record_path, usecols=columns, dtype={"location_id": str, "pws_id": str, "state": str})
    if len(frame) != 338496 or frame.location_id.nunique() != 21156:
        raise ValueError("Unexpected formal evaluation cohort")
    weights = evaluation_weights(frame)["jurisdiction_pws_equal"]
    vectors = build_vectors(frame)
    print(f"Independent-inner bootstrap: {args.bootstrap_reps} replicates, {len(vectors)} contrasts", flush=True)
    summary, draws = independent_nested_bootstrap(frame, vectors, args.bootstrap_reps, args.seed, args.batch_size)
    point_error = float(np.max(np.abs(summary.estimate.to_numpy() - np.array([weights @ values for values in vectors.values()]))))
    if point_error > 1e-12:
        raise ValueError("Nested bootstrap point estimate differs from target weighting")
    key_index = {key: index for index, key in enumerate(vectors)}
    chain_errors = []
    for reference, candidate in [("factorized", model) for model in MODELS[1:]] + [(BASELINE, model) for model in MODELS]:
        positions = [key_index[f"{reference}|{candidate}|{metric}"] for metric in NLL]
        error = float(np.max(np.abs(draws[:, positions[0]] - draws[:, positions[1]] - draws[:, positions[2]])))
        chain_errors.append({"reference": reference, "candidate": candidate, "maximum_replicate_chain_error": error})
        if error > 1e-12:
            raise ValueError("Joint resampling failed component covariance audit")
    split = summary.key.str.split("|", expand=True)
    summary["reference"], summary["candidate"], summary["metric"] = split[0], split[1], split[2]
    summary = summary.drop(columns="key")
    summary["weighting"] = "jurisdiction_pws_equal"
    summary["loss_reduction"] = summary.estimate
    summary["interval_method"] = "jurisdiction_resampling_with_independent_PWS_resampling_per_jurisdiction_copy"
    summary["bootstrap_reps"] = args.bootstrap_reps
    summary["bootstrap_seed"] = args.seed
    inherited = pd.read_csv(inherited_path)
    keys = ["reference", "candidate", "metric"]
    comparison = summary.merge(inherited[keys + ["estimate", "ci95_low", "ci95_high"]], on=keys, suffixes=("", "_inherited"), validate="one_to_one")
    if len(comparison) != 24:
        raise ValueError("Inherited interval coverage mismatch")
    inherited_point_error = float(np.max(np.abs(comparison.estimate - comparison.estimate_inherited)))
    if inherited_point_error > 1e-12:
        raise ValueError("Inherited and independent nested point estimates differ")
    comparison["interval_conclusion"] = classify_interval(comparison.ci95_low, comparison.ci95_high)
    comparison["inherited_interval_conclusion"] = classify_interval(comparison.ci95_low_inherited, comparison.ci95_high_inherited)
    comparison["interval_conclusion_changed"] = comparison.interval_conclusion.ne(comparison.inherited_interval_conclusion)
    summary_path = args.analysis_dir / "paired_contrasts_nested_independent.csv"
    comparison_path = args.analysis_dir / "bootstrap_method_comparison.csv"
    summary.to_csv(summary_path, index=False)
    comparison.to_csv(comparison_path, index=False)
    changed = comparison.loc[comparison.interval_conclusion_changed, keys + ["interval_conclusion", "inherited_interval_conclusion"]]
    audit = {
        "status": "PASS", "bootstrap_reps": args.bootstrap_reps, "bootstrap_seed": args.seed,
        "contrast_count": len(summary), "records": len(frame), "locations": int(frame.location_id.nunique()),
        "jurisdictions": int(frame.state.nunique()), "pws": int(frame.pws_id.nunique()),
        "method": "Outer resampling of jurisdictions with replacement. For each sampled jurisdiction copy, independently resample its PWS with replacement. Average locations and queries before resampling. All contrasts share all sampled indices.",
        "reason_for_correction": "The inherited algorithm reused a single inner PWS resample if the same jurisdiction was selected multiple times in a replicate. Independent copies now receive independent inner resamples. This method was selected on algorithmic grounds before comparing resulting intervals.",
        "main_interval_policy": "Use independent-inner nested intervals for this additional exploratory analysis. Preserve inherited intervals for provenance; do not choose between interval methods by result.",
        "uncertainty_scope": "Conditional on frozen predictions; no retraining, model-selection, or release uncertainty included. Pointwise percentile intervals, not simultaneous intervals.",
        "primary_point_max_error": point_error, "inherited_point_max_error": inherited_point_error,
        "replicate_chain_rule_checks": chain_errors,
        "interval_conclusions_changed": len(changed), "changed_contrasts": changed.to_dict(orient="records"),
        "input_sha256": {str(record_path): sha256(record_path), str(inherited_path): sha256(inherited_path)},
        "script_sha256": sha256(Path(__file__)),
        "output_sha256": {str(summary_path): sha256(summary_path), str(comparison_path): sha256(comparison_path)},
    }
    (args.analysis_dir / "BOOTSTRAP_AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(summary.to_string(index=False), flush=True)
    print(json.dumps({"interval_conclusions_changed": len(changed), "changed_contrasts": audit["changed_contrasts"]}), flush=True)


if __name__ == "__main__":
    main()

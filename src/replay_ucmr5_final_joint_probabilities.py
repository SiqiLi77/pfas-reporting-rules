#!/usr/bin/env python3
"""Replay, without fitting, the frozen final-release 64-state distributions.

The payload stores float64 raw probabilities. Downstream log-score analyses
must apply the original uniform-contamination epsilon to the normalized raw
distribution, rather than independently flooring derived probabilities.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "work" / "vendor"), str(ROOT / "src")]

import numpy as np
import pandas as pd
import torch

from analyze_ucmr5_conditional_gaussian_factor import GaussianFactorKernel
from analyze_ucmr5_conditional_joint_models import CHEMICALS, QUERIES, labels_at, margins_at, shock_probability
from analyze_ucmr5_margin_preserving_joint import enumerate_states
from analyze_ucmr5_multithreshold_joint_curve import LOG_SCORE_EPSILON, evaluate_threshold, factorized_probability, prediction_lookup
from score_ucmr5_frozen_joint_release_vintage import (
    FORMAL_FOLDS, MODELS, REPLAY_COLUMNS, decision_records, raw_from_loading_record,
    rates_from_record, replay_audit_row, sha256,
)

CANONICAL = ROOT / "outputs/ucmr5_final_20260828_r1/06_final_ml_full/canonical_server_run"


def metadata(frame: pd.DataFrame, fold: int, alpha: float, labels: np.ndarray) -> pd.DataFrame:
    result = frame[["location_id", "pws_id", "state"]].astype(str).reset_index(drop=True)
    result["outer_fold"] = int(fold)
    result["alpha"] = float(alpha)
    result["observed_state_code"] = labels @ (2 ** np.arange(6))
    result["observed_count"] = labels.sum(axis=1)
    visible = frame[[f"old_visible__{c}" for c in CHEMICALS]].to_numpy(int)
    result["old_visible_code"] = visible @ (2 ** np.arange(6))
    if np.any((result.observed_state_code.to_numpy(int) & result.old_visible_code.to_numpy(int)) != result.old_visible_code.to_numpy(int)):
        raise ValueError("Observed query labels contradict old-visible measurements")
    return result


def unique_gaussian_probability(kernel, marginal, raw, batch_size):
    # Exact duplicate elimination changes no numerical input or prediction.
    unique, inverse = np.unique(marginal, axis=0, return_inverse=True)
    return kernel.full_probability(unique, raw, batch_size)[inverse], len(unique)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cohort", type=Path, default=ROOT / "work/ucmr5_final_20260828_r1/ucmr5_artificial_censor_strict.csv.gz")
    p.add_argument("--canonical", type=Path, default=CANONICAL)
    p.add_argument("--output-dir", type=Path, default=ROOT / "outputs/joint_information_decomposition_20260916/payload")
    p.add_argument("--device", default="cpu")
    p.add_argument("--folds", default="1,2,3,4")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    folds = [int(s) for s in args.folds.split(",")]
    if not set(folds).issubset(FORMAL_FOLDS):
        raise ValueError("Only formal frozen folds 1--4 are admissible")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "cohort": args.cohort,
        "predictions": args.canonical / "marginal/merged/out_of_fold_distribution_predictions.csv.gz",
        "shared_decision": args.canonical / "joint/merged/decision.json",
        "gaussian_decision": args.canonical / "gaussian/merged/decision.json",
        "shared_loss": args.canonical / "joint/merged/site_threshold_losses.csv.gz",
        "gaussian_loss": args.canonical / "gaussian/merged/site_threshold_losses.csv.gz",
    }
    provenance = {k: {"path": str(v.resolve()), "sha256": sha256(v)} for k, v in paths.items()}
    cohort = pd.read_csv(args.cohort, dtype={"location_id": str, "pws_id": str, "state": str})
    formal = cohort[cohort.state_fold.isin(FORMAL_FOLDS)]
    if len(formal) != 21156 or formal.location_id.duplicated().any():
        raise ValueError("Frozen formal cohort coverage mismatch")
    shared_decision = json.loads(paths["shared_decision"].read_text())
    gaussian_decision = json.loads(paths["gaussian_decision"].read_text())
    if shared_decision["margin_logit_temperature"] != 1 or gaussian_decision["margin_logit_temperature"] != 1:
        raise ValueError("Replay expects frozen temperature 1")
    model = shared_decision["test_prediction_model"]
    if gaussian_decision["test_prediction_model"] != model:
        raise ValueError("Marginal prediction models differ")
    predictions = pd.read_csv(paths["predictions"], dtype={"location_id": str})
    predictions = predictions[predictions.model.eq(model)]
    shared_records = decision_records(paths["shared_decision"], "shared_shock")
    gaussian_records = decision_records(paths["gaussian_decision"], "gaussian_factor")
    archives = {
        "shared": pd.read_csv(paths["shared_loss"], dtype={"location_id": str, "pws_id": str, "state": str}),
        "gaussian": pd.read_csv(paths["gaussian_loss"], dtype={"location_id": str, "pws_id": str, "state": str}),
    }
    kernel = GaussianFactorKernel(2, gaussian_decision["evaluation_quadrature_order"], torch.device(args.device), torch.float64)
    states = enumerate_states()
    audits, distribution_audits = [], []
    for fold in folds:
        started = time.time()
        test = formal[formal.state_fold.eq(fold)].copy()
        lookup = prediction_lookup(predictions, test)
        rates = rates_from_record(shared_records[fold])
        raw = raw_from_loading_record(gaussian_records[fold])
        parts = {name: [] for name in MODELS}
        metadata_parts = []
        for alpha in QUERIES:
            marginal = margins_at(test, lookup, alpha, "log_mrl")
            labels = labels_at(test, alpha, "log_mrl")
            gaussian, unique_count = unique_gaussian_probability(kernel, marginal, raw, args.batch_size)
            probabilities = {
                "factorized": factorized_probability(marginal),
                "conditional_shared_shock": shock_probability(marginal, rates),
                "conditional_gaussian_factor2": gaussian,
            }
            codes = labels @ (2 ** np.arange(6))
            metadata_parts.append(metadata(test, fold, alpha, labels))
            for name, probability in probabilities.items():
                _, scored, _ = evaluate_threshold(probability, labels, test, model=name, alpha=alpha, outer_fold=fold)
                scored["observed_state_probability_raw"] = probability[np.arange(len(test)), codes]
                archive = archives["gaussian" if name == "conditional_gaussian_factor2" else "shared"]
                reference = archive[archive.outer_fold.eq(fold) & archive.alpha.eq(alpha) & archive.model.eq(name)]
                audit = replay_audit_row(reference, scored, fold, alpha, name)
                tolerance = 2e-6 if name == "conditional_gaussian_factor2" else 1e-10
                audit["tolerance"] = tolerance
                audit["passed"] = max(audit[f"maximum_absolute_difference__{c}"] for c in REPLAY_COLUMNS) <= tolerance
                audits.append(audit)
                if not audit["passed"]:
                    raise ValueError(f"Archived-score replay failed: {audit}")
                distribution_audits.append({
                    "outer_fold": fold, "alpha": alpha, "model": name, "rows": len(test),
                    "minimum_probability": float(probability.min()),
                    "maximum_row_sum_residual": float(np.max(np.abs(probability.sum(axis=1) - 1))),
                    "maximum_marginal_residual": float(np.max(np.abs(probability @ states - marginal))),
                    "gaussian_unique_marginal_vectors": unique_count,
                })
                parts[name].append(probability.astype(np.float64, copy=False))
            print(f"fold={fold} alpha={alpha:.2f} n={len(test)} unique={unique_count} elapsed={time.time()-started:.1f}s", flush=True)
        payload_path = args.output_dir / f"fold_{fold}.npz"
        metadata_path = args.output_dir / f"metadata_fold_{fold}.csv.gz"
        np.savez_compressed(payload_path, **{name: np.concatenate(values) for name, values in parts.items()})
        pd.concat(metadata_parts, ignore_index=True).to_csv(metadata_path, index=False)
        pd.DataFrame([a for a in audits if a["outer_fold"] == fold]).to_csv(args.output_dir / f"replay_audit_fold_{fold}.csv", index=False)
        pd.DataFrame([a for a in distribution_audits if a["outer_fold"] == fold]).to_csv(args.output_dir / f"distribution_audit_fold_{fold}.csv", index=False)
        report = {
            "status": "PASS", "outer_fold": fold, "locations": len(test), "rows": len(test)*len(QUERIES),
            "models": list(MODELS), "dtype": "float64", "raw_probabilities": True,
            "chemical_order_low_bit_first": list(CHEMICALS), "queries": list(QUERIES),
            "row_order": "query-major; original cohort order within each query",
            "uniform_contamination_epsilon": LOG_SCORE_EPSILON,
            "retrained": False, "device": args.device,
            "exact_duplicate_margin_deduplication": True,
            "source_files": provenance,
            "payload_sha256": sha256(payload_path), "metadata_sha256": sha256(metadata_path),
            "maximum_replay_difference_by_model": {
                name: max(a[f"maximum_absolute_difference__{c}"] for a in audits if a["outer_fold"] == fold and a["model"] == name for c in REPLAY_COLUMNS)
                for name in MODELS
            },
        }
        (args.output_dir / f"audit_fold_{fold}.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"Saved fold {fold}: {payload_path} elapsed={time.time()-started:.1f}s", flush=True)


if __name__ == "__main__":
    main()

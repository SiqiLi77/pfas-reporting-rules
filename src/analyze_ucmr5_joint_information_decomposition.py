#!/usr/bin/env python3
"""Exact chain-rule decomposition and train-composition/pairing diagnostics.

No model fitting or selection. See the accompanying frozen-analysis protocol.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EPSILON = 1e-12
COUNTS = np.array([int(i).bit_count() for i in range(64)])
KEYS = ["location_id", "pws_id", "state", "outer_fold", "alpha"]
MODELS = ("factorized", "conditional_shared_shock", "conditional_gaussian_factor2")
BASELINE = "training_composition"
METRICS = ("pattern_nll", "count_nll", "identity_given_count_nll", "count_crps", "count_brier", "k2_brier")
NLL = METRICS[:3]


def score_distribution(raw: np.ndarray, codes: np.ndarray):
    raw = np.asarray(raw, dtype=float)
    codes = np.asarray(codes, dtype=int)
    if raw.shape != (len(codes), 64) or not np.isfinite(raw).all():
        raise ValueError("Invalid probability shape/values")
    if np.any(raw < -1e-10) or np.any(raw.sum(axis=1) <= 0) or np.any((codes < 0) | (codes > 63)):
        raise ValueError("Invalid probability or state")
    clean = np.maximum(raw, 0)
    clean /= clean.sum(axis=1, keepdims=True)
    q = (1 - EPSILON) * clean + EPSILON / 64
    count_q = np.stack([q[:, COUNTS == k].sum(axis=1) for k in range(7)], axis=1)
    count_raw = np.stack([raw[:, COUNTS == k].sum(axis=1) for k in range(7)], axis=1)
    k = COUNTS[codes]
    rows = np.arange(len(codes))
    observed = q[rows, codes]
    observed_count_q = count_q[rows, k]
    losses = {
        "pattern_nll": -np.log(observed),
        "count_nll": -np.log(observed_count_q),
        "identity_given_count_nll": -np.log(observed / observed_count_q),
        "count_crps": np.sum((np.cumsum(count_raw, axis=1)[:, :6] - (k[:, None] <= np.arange(6))) ** 2, axis=1),
        "count_brier": np.sum((count_raw - np.eye(7)[k]) ** 2, axis=1),
        "k2_brier": (count_raw[:, 2:].sum(axis=1) - (k >= 2)) ** 2,
    }
    if np.max(np.abs(losses["pattern_nll"] - losses["count_nll"] - losses["identity_given_count_nll"])) > 1e-12:
        raise ValueError("Chain rule failed")
    if np.any(losses["identity_given_count_nll"][(k == 0) | (k == 6)] != 0):
        raise ValueError("Singleton count classes have nonzero identity loss")
    return losses, q, count_q


def expected_pairing_losses(q, count_q, codes, groups):
    """Exact mean over uniform within-group permutations, not loss of mean q."""
    codes = np.asarray(codes, int)
    k = COUNTS[codes]
    pattern = np.empty(len(codes))
    count = np.empty(len(codes))
    for indices in groups:
        indices = np.asarray(indices, int)
        pattern[indices] = (-np.log(q[indices]).mean(axis=0))[codes[indices]]
        count[indices] = (-np.log(count_q[indices]).mean(axis=0))[k[indices]]
    return {"pattern_nll": pattern, "count_nll": count, "identity_given_count_nll": pattern - count}


def evaluation_weights(meta):
    if meta.duplicated(KEYS).any():
        raise ValueError("Duplicate evaluation keys")
    loc = meta[["location_id", "pws_id", "state"]].drop_duplicates()
    if loc.location_id.duplicated().any():
        raise ValueError("Location has inconsistent PWS/state")
    loc["locations_in_pws"] = loc.groupby(["state", "pws_id"]).location_id.transform("size")
    pws = loc[["state", "pws_id"]].drop_duplicates()
    pws["pws_in_state"] = pws.groupby("state").pws_id.transform("size")
    loc = loc.merge(pws, on=["state", "pws_id"], validate="many_to_one")
    nquery = meta.groupby("location_id").size()
    nalpha = meta.groupby("location_id").alpha.nunique()
    if nquery.nunique() != 1 or not nquery.equals(nalpha):
        raise ValueError("Unequal or duplicate query sets")
    loc["weight"] = 1 / (loc.state.nunique() * loc.pws_in_state * loc.locations_in_pws)
    weights = meta.location_id.map(loc.set_index("location_id").weight).to_numpy() / int(nquery.iloc[0])
    if abs(weights.sum() - 1) > 1e-12:
        raise ValueError("Weights do not sum to one")
    return {"jurisdiction_pws_equal": weights, "record_equal": np.full(len(meta), 1 / len(meta))}


def contributions(meta, values, weights, group_name, labels):
    rows = []
    for group in sorted(pd.unique(labels)):
        mask = np.asarray(labels == group)
        mass = float(weights[mask].sum())
        part = meta.loc[mask]
        for metric, vector in values.items():
            contribution = float(np.dot(weights[mask], vector[mask]))
            rows.append({"grouping": group_name, "group": str(group), "metric": metric,
                         "records": int(mask.sum()), "locations": int(part.location_id.nunique()),
                         "pws": int(part.pws_id.nunique()), "jurisdictions": int(part.state.nunique()),
                         "full_cohort_weight_mass": mass, "within_group_mean": contribution / mass,
                         "contribution_to_full_cohort": contribution})
    frame = pd.DataFrame(rows)
    for metric, vector in values.items():
        part = frame[frame.metric.eq(metric)]
        if abs(part.full_cohort_weight_mass.sum() - 1) > 1e-12 or abs(part.contribution_to_full_cohort.sum() - np.dot(weights, vector)) > 1e-12:
            raise ValueError("Subgroup denominator/contribution audit failed")
    return frame


def hierarchical_bootstrap(meta, vectors, reps=20000, seed=20260916):
    """Paired state/PWS bootstrap, preserving all component covariance."""
    columns = list(vectors)
    frame = meta[["location_id", "pws_id", "state"]].copy()
    for name, vector in vectors.items():
        frame[name] = vector
    integrated = frame.groupby(["state", "pws_id", "location_id"], sort=True)[columns].mean()
    pws = integrated.groupby(["state", "pws_id"], sort=True).mean()
    grouped = [g.to_numpy() for _, g in pws.groupby(level="state", sort=True)]
    rng = np.random.default_rng(seed)
    state_draws = np.empty((reps, len(grouped), len(columns)))
    for s, values in enumerate(grouped):
        for start in range(0, reps, 200):
            end = min(start + 200, reps)
            indices = rng.integers(len(values), size=(end - start, len(values)))
            state_draws[start:end, s] = values[indices].mean(axis=1)
    choice = rng.integers(len(grouped), size=(reps, len(grouped)))
    draws = state_draws[np.arange(reps)[:, None], choice].mean(axis=1)
    point = np.mean([v.mean(axis=0) for v in grouped], axis=0)
    low, high = np.quantile(draws, [0.025, 0.975], axis=0)
    return pd.DataFrame({"key": columns, "estimate": point, "ci95_low": low, "ci95_high": high}), draws


def align_baseline(meta, directory):
    other = pd.read_csv(directory / "metadata.csv.gz", dtype={"location_id": str, "pws_id": str, "state": str})
    other["source_index"] = np.arange(len(other))
    if other.duplicated(KEYS).any():
        raise ValueError("Baseline duplicate keys")
    order = meta.merge(other, on=KEYS, how="left", suffixes=("", "_baseline"), validate="one_to_one")
    if order.source_index.isna().any() or len(other) != len(meta):
        raise ValueError("Baseline key coverage mismatch")
    for col in ("observed_state_code", "observed_count", "old_visible_code"):
        if not np.array_equal(order[col], order[col + "_baseline"]):
            raise ValueError(f"Baseline metadata mismatch: {col}")
    with np.load(directory / "probabilities.npz") as payload:
        q = payload["raw_probability"][order.source_index.to_numpy(int)]
    return q


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "outputs/joint_information_decomposition_20260916")
    parser.add_argument("--bootstrap-reps", type=int, default=20000)
    args = parser.parse_args()
    out = args.root / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    pieces = [pd.read_csv(args.root / "payload" / f"metadata_fold_{fold}.csv.gz", dtype={"location_id": str, "pws_id": str, "state": str}) for fold in range(1, 5)]
    meta = pd.concat(pieces, ignore_index=True)
    if len(meta) != 338496 or meta.location_id.nunique() != 21156:
        raise ValueError("Unexpected formal population")
    if not np.array_equal(COUNTS[meta.observed_state_code.to_numpy(int)], meta.observed_count):
        raise ValueError("Count/code mismatch")
    weights = evaluation_weights(meta)
    for name, vector in weights.items():
        meta[f"weight__{name}"] = vector
    wide = meta.copy()
    frequency_rows = []
    score_audit = []
    for model in (*MODELS, BASELINE):
        print(f"Scoring {model}", flush=True)
        if model == BASELINE:
            qraw = align_baseline(meta, args.root / "baseline")
        else:
            qraw = np.concatenate([np.load(args.root / "payload" / f"fold_{fold}.npz")[model] for fold in range(1, 5)])
        losses, q, count_q = score_distribution(qraw, meta.observed_state_code.to_numpy(int))
        for name, vector in losses.items():
            wide[f"{model}__{name}"] = vector
        groups = list(meta.groupby(["outer_fold", "old_visible_code", "alpha"], sort=True).indices.values())
        pair = expected_pairing_losses(q, count_q, meta.observed_state_code.to_numpy(int), groups)
        for name, vector in pair.items():
            wide[f"{model}__expected_repairing__{name}"] = vector
        for weighting, w in weights.items():
            pred = w @ q
            observed = np.bincount(meta.observed_state_code, weights=w, minlength=64)
            for code in range(64):
                frequency_rows.append({"weighting": weighting, "model": model, "state_code": code,
                                       "count": int(COUNTS[code]), "observed_frequency": float(observed[code]),
                                       "predicted_frequency": float(pred[code])})
        chain_error = np.max(np.abs(losses["pattern_nll"] - losses["count_nll"] - losses["identity_given_count_nll"]))
        score_audit.append({"model": model, "chain_rule_max_error": float(chain_error),
                            "minimum_raw_probability": float(qraw.min()), "row_sum_max_error": float(np.abs(qraw.sum(axis=1)-1).max())})
        del qraw, q, count_q, losses, pair
    print("Building summaries and paired contrasts", flush=True)
    model_rows, gain_rows, subgroup_frames, pairing_rows = [], [], [], []
    boot_vectors = {}
    pairs = [("factorized", m) for m in MODELS[1:]] + [(BASELINE, m) for m in MODELS]
    for reference, candidate in pairs:
        metrics = NLL if reference == "factorized" else METRICS
        delta = {metric: (wide[f"{reference}__{metric}"] - wide[f"{candidate}__{metric}"]).to_numpy() for metric in metrics}
        for metric, vector in delta.items():
            boot_vectors[f"{reference}|{candidate}|{metric}"] = vector
        for weighting, w in weights.items():
            for metric, vector in delta.items():
                gain_rows.append({"reference": reference, "candidate": candidate, "weighting": weighting,
                                  "metric": metric, "loss_reduction": float(w @ vector)})
            for group_name, labels in (("old_visibility", np.where(meta.old_visible_code.eq(0), "none_visible", "any_visible")),
                                       ("observed_count", meta.observed_count.to_numpy())):
                f = contributions(meta, delta, w, group_name, labels)
                f["reference"], f["candidate"], f["weighting"] = reference, candidate, weighting
                subgroup_frames.append(f)
    for model in (*MODELS, BASELINE):
        for weighting, w in weights.items():
            for metric in METRICS:
                model_rows.append({"model": model, "weighting": weighting, "metric": metric,
                                   "loss": float(w @ wide[f"{model}__{metric}"])})
        for group_name, mask in (("all", np.ones(len(meta), bool)), ("none_visible", meta.old_visible_code.eq(0).to_numpy()), ("any_visible", meta.old_visible_code.ne(0).to_numpy())):
            for metric in NLL:
                observed = wide.loc[mask, f"{model}__{metric}"].mean()
                randomized = wide.loc[mask, f"{model}__expected_repairing__{metric}"].mean()
                pairing_rows.append({"model": model, "stratum": group_name, "metric": metric, "weighting": "record_equal",
                                     "records": int(mask.sum()), "locations": int(meta.loc[mask].location_id.nunique()),
                                     "own_record_loss": float(observed), "expected_random_pairing_loss": float(randomized),
                                     "own_pairing_loss_reduction": float(randomized - observed)})
    metrics_table, gains = pd.DataFrame(model_rows), pd.DataFrame(gain_rows)
    metrics_table.to_csv(out / "model_metrics.csv", index=False)
    gains.to_csv(out / "paired_contrasts_point.csv", index=False)
    pd.concat(subgroup_frames, ignore_index=True).to_csv(out / "subgroup_contributions.csv", index=False)
    pd.DataFrame(pairing_rows).to_csv(out / "record_pairing_diagnostic.csv", index=False)
    frequency = pd.DataFrame(frequency_rows)
    frequency.to_csv(out / "pattern_frequencies.csv", index=False)
    frequency.groupby(["weighting", "model", "count"], as_index=False)[["observed_frequency", "predicted_frequency"]].sum().to_csv(out / "count_frequencies.csv", index=False)
    wide.to_csv(out / "record_scores.csv.gz", index=False)
    print(f"Running {args.bootstrap_reps} paired hierarchical bootstrap replicates", flush=True)
    intervals, _ = hierarchical_bootstrap(meta, boot_vectors, args.bootstrap_reps)
    fields = intervals.key.str.split("|", expand=True)
    intervals["reference"], intervals["candidate"], intervals["metric"] = fields[0], fields[1], fields[2]
    intervals = intervals.drop(columns=["key"])
    checked = gains[gains.weighting.eq("jurisdiction_pws_equal")].merge(intervals, on=["reference", "candidate", "metric"], validate="one_to_one")
    bootstrap_point_error = float(np.max(np.abs(checked.loss_reduction - checked.estimate)))
    if bootstrap_point_error > 1e-12:
        raise ValueError("Bootstrap aggregation differs from primary weighting")
    checked.to_csv(out / "paired_contrasts_primary.csv", index=False)
    fractions = []
    for weighting in weights:
        for model in MODELS[1:]:
            part = gains[gains.weighting.eq(weighting) & gains.reference.eq("factorized") & gains.candidate.eq(model)].set_index("metric").loss_reduction
            fractions.append({"candidate": model, "weighting": weighting,
                              "pattern_gain": float(part.pattern_nll), "count_gain": float(part.count_nll),
                              "identity_gain": float(part.identity_given_count_nll),
                              "count_share_of_net_gain": float(part.count_nll / part.pattern_nll),
                              "identity_share_of_net_gain": float(part.identity_given_count_nll / part.pattern_nll)})
    pd.DataFrame(fractions).to_csv(out / "gain_components.csv", index=False)
    audit = {"status": "PASS", "locations": int(meta.location_id.nunique()), "records": len(meta),
             "pws": int(meta.pws_id.nunique()), "jurisdictions": int(meta.state.nunique()),
             "queries": int(meta.alpha.nunique()), "bootstrap_reps": args.bootstrap_reps, "bootstrap_seed": 20260916,
             "bootstrap_primary_point_max_difference": bootstrap_point_error, "score_audits": score_audit,
             "analysis_role": "additional exploratory analysis of selected frozen models; no independent validation",
             "interval_scope": "Pointwise paired state/PWS bootstrap conditional on frozen predictions; excludes retraining/model selection/release uncertainty.",
             "pairing_scope": "Exact random reassignment expectation within fold/old-visible-mask; record-equal, no permutation p-value."}
    (out / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(pd.DataFrame(fractions).to_string(index=False), flush=True)
    print(checked.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

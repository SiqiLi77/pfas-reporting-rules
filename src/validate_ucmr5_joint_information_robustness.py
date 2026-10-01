#!/usr/bin/env python3
"""Frozen joint-score epsilon, subgroup and cluster-influence sensitivity.

Independent implementation of scoring and hierarchical weights. No model fits.
See outputs/joint_information_validation_20260916/robustness/PROTOCOL.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("factorized", "conditional_shared_shock", "conditional_gaussian_factor2")
METRICS = ("pattern_nll", "count_nll", "identity_given_count_nll")
EPSILONS = (1e-8, 1e-10, 1e-12, 1e-14)
COUNTS = np.asarray([int(i).bit_count() for i in range(64)])
KEYS = ["location_id", "pws_id", "state", "outer_fold", "alpha"]
IDTYPES = {"location_id": str, "pws_id": str, "state": str}


def score_components(raw, codes, epsilon):
    """All three losses arise from one row-normalized, contaminated q."""
    raw = np.asarray(raw, dtype=np.float64)
    codes = np.asarray(codes, dtype=np.int64)
    if raw.shape != (len(codes), 64) or not np.isfinite(raw).all():
        raise ValueError("Invalid probability array")
    if np.any(raw < -1e-10) or np.any((codes < 0) | (codes > 63)):
        raise ValueError("Negative probability or invalid observed code")
    if not 0 < epsilon < 1:
        raise ValueError("Zero smoothing is audited separately, not silently floored")
    q = np.maximum(raw, 0)
    total = q.sum(axis=1)
    if np.any(total <= 0) or np.max(np.abs(total - 1)) > 1e-8:
        raise ValueError("Invalid row normalization")
    q /= total[:, None]
    q *= 1 - epsilon
    q += epsilon / 64
    rows = np.arange(len(codes))
    obs = q[rows, codes]
    count = np.empty(len(codes))
    k = COUNTS[codes]
    for group in range(7):
        mask = k == group
        count[mask] = q[mask][:, COUNTS == group].sum(axis=1)
    result = np.column_stack((-np.log(obs), -np.log(count), -np.log(obs / count)))
    if np.max(np.abs(result[:, 0] - result[:, 1] - result[:, 2])) > 1e-12:
        raise ValueError("Chain-rule mismatch")
    if np.any(result[(k == 0) | (k == 6), 2] != 0):
        raise ValueError("Singleton count class has conditional identity loss")
    return result


def weights_for(meta):
    """Row weights for threshold/location/PWS/jurisdiction hierarchy."""
    if meta.duplicated(KEYS).any():
        raise ValueError("Duplicate evaluation key")
    loc = meta[["location_id", "pws_id", "state"]].drop_duplicates()
    if loc.location_id.duplicated().any():
        raise ValueError("Location maps to multiple PWS or jurisdictions")
    if loc.groupby("pws_id").state.nunique().max() != 1:
        raise ValueError("PWS maps to multiple jurisdictions")
    loc_n = loc.groupby(["state", "pws_id"]).location_id.transform("size")
    pws = loc[["state", "pws_id"]].drop_duplicates()
    pws_n = pws.groupby("state").size()
    base = 1 / (loc.state.nunique() * loc.state.map(pws_n) * loc_n)
    location_weight = pd.Series(base.to_numpy(), index=loc.location_id)
    query_n = meta.groupby("location_id").size()
    if query_n.nunique() != 1:
        raise ValueError("Unequal threshold records per location")
    primary = meta.location_id.map(location_weight).to_numpy() / meta.location_id.map(query_n).to_numpy()
    if abs(primary.sum() - 1) > 1e-12:
        raise ValueError("Weight sum differs from one")
    return {"jurisdiction_pws_equal": primary,
            "record_equal": np.ones(len(meta), dtype=float) / len(meta)}


def support(meta, mask):
    part = meta.loc[np.asarray(mask)]
    return {"records": int(len(part)), "locations": int(part.location_id.nunique()),
            "pws": int(part.pws_id.nunique()), "jurisdictions": int(part.state.nunique())}


def aggregate_rows(meta, delta, candidate, epsilon):
    rows = []
    for weighting, w in weights_for(meta).items():
        point = w @ delta
        for index, metric in enumerate(METRICS):
            rows.append({"candidate": candidate, "reference": "factorized", "epsilon": epsilon,
                         "weighting": weighting, "metric": metric, "loss_reduction": point[index],
                         "count_share_of_pattern_gain": point[1] / point[0] if point[0] != 0 else np.nan,
                         **support(meta, np.ones(len(meta), bool))})
    return rows


def subgroup_rows(meta, delta, candidate):
    rows = []
    for grouping, labels in (("old_visibility", np.where(meta.old_visible_code == 0, "none_visible", "any_visible")),
                             ("observed_count", meta.observed_count.to_numpy())):
        for weighting, w in weights_for(meta).items():
            contributions = []
            for label in sorted(np.unique(labels)):
                mask = labels == label
                mass = w[mask].sum()
                part = w[mask] @ delta[mask]
                contributions.append(part)
                for index, metric in enumerate(METRICS):
                    rows.append({"candidate": candidate, "reference": "factorized", "grouping": grouping,
                                 "group": str(label), "weighting": weighting, "metric": metric,
                                 "full_cohort_weight_mass": mass, "within_group_mean": part[index] / mass,
                                 "contribution_to_full_cohort": part[index], **support(meta, mask)})
            if np.max(np.abs(np.sum(contributions, axis=0) - w @ delta)) > 1e-12:
                raise ValueError("Subgroup contributions do not reconcile")
    return rows


def cluster_omissions(meta, delta):
    """Exact delete-cluster estimates with hierarchy recomputed after deletion."""
    frame = meta[["state", "pws_id", "location_id"]].copy()
    for j, metric in enumerate(METRICS):
        frame[metric] = delta[:, j]
    loc = frame.groupby(["state", "pws_id", "location_id"], sort=True)[list(METRICS)].mean()
    pws = loc.groupby(level=["state", "pws_id"], sort=True).mean()
    state = pws.groupby(level="state", sort=True).mean()
    ns = pws.groupby(level="state").size()
    nstate = len(state)
    if nstate < 2:
        raise ValueError("At least two jurisdictions required")
    state_total = state.sum().to_numpy()
    full = state_total / nstate
    record_total = delta.sum(axis=0)
    rec_state = frame.groupby("state")[list(METRICS)].sum()
    rec_pws = frame.groupby(["state", "pws_id"])[list(METRICS)].sum()
    nrec_state = frame.groupby("state").size()
    nrec_pws = frame.groupby(["state", "pws_id"]).size()
    nloc_state = meta.groupby("state").location_id.nunique()
    nloc_pws = meta.groupby(["state", "pws_id"]).location_id.nunique()
    result = []
    for key, value in state.iterrows():
        primary = (state_total - value.to_numpy()) / (nstate - 1)
        record = (record_total - rec_state.loc[key].to_numpy()) / (len(meta) - nrec_state.loc[key])
        for weighting, point in (("jurisdiction_pws_equal", primary), ("record_equal", record)):
            for j, metric in enumerate(METRICS):
                result.append({"omission_level": "jurisdiction", "omitted_state": key, "omitted_pws_id": "",
                               "weighting": weighting, "metric": metric, "loss_reduction": point[j],
                               "omitted_records": int(nrec_state.loc[key]), "omitted_locations": int(nloc_state.loc[key]),
                               "omitted_pws": int(ns.loc[key]), "remaining_jurisdictions": nstate - 1})
    for (s, p), value in pws.iterrows():
        n = int(ns.loc[s])
        old_state = state.loc[s].to_numpy()
        if n > 1:
            new_state = (n * old_state - value.to_numpy()) / (n - 1)
            primary = (state_total - old_state + new_state) / nstate
        else:
            primary = (state_total - old_state) / (nstate - 1)
        record = (record_total - rec_pws.loc[(s, p)].to_numpy()) / (len(meta) - nrec_pws.loc[(s, p)])
        for weighting, point in (("jurisdiction_pws_equal", primary), ("record_equal", record)):
            for j, metric in enumerate(METRICS):
                result.append({"omission_level": "pws", "omitted_state": s, "omitted_pws_id": p,
                               "weighting": weighting, "metric": metric, "loss_reduction": point[j],
                               "omitted_records": int(nrec_pws.loc[(s, p)]), "omitted_locations": int(nloc_pws.loc[(s, p)]),
                               "omitted_pws": 1, "remaining_jurisdictions": nstate - (n == 1)})
    if np.max(np.abs(weights_for(meta)["jurisdiction_pws_equal"] @ delta - full)) > 1e-12:
        raise ValueError("Hierarchy differs from row weighting")
    return pd.DataFrame(result)


def source_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "outputs/joint_information_decomposition_20260916")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/joint_information_validation_20260916/robustness")
    args = parser.parse_args()
    out, source = args.output, args.source
    if not (out / "PROTOCOL.md").exists():
        raise ValueError("Prespecified protocol must exist before calculation")
    meta_parts = [pd.read_csv(source / "payload" / f"metadata_fold_{f}.csv.gz", dtype=IDTYPES) for f in range(1, 5)]
    meta = pd.concat(meta_parts, ignore_index=True)
    if not np.array_equal(meta.observed_count, COUNTS[meta.observed_state_code.to_numpy(int)]):
        raise ValueError("Count/state mismatch")
    if len(meta) != 338496 or meta.location_id.nunique() != 21156 or meta.pws_id.nunique() != 8316:
        raise ValueError("Unexpected formal population")
    weights = weights_for(meta)
    scorecols = [f"{model}__{metric}" for model in MODELS for metric in METRICS]
    archived = pd.read_csv(source / "analysis/record_scores.csv.gz", dtype=IDTYPES,
                           usecols=KEYS + scorecols + [f"weight__{w}" for w in weights])
    if archived.duplicated(KEYS).any() or len(archived) != len(meta):
        raise ValueError("Invalid archived score keys")
    aligned = meta.merge(archived, on=KEYS, validate="one_to_one", how="left")
    if aligned[scorecols].isna().any().any():
        raise ValueError("Archived score coverage mismatch")
    weight_error = max(float(np.max(np.abs(weights[w] - aligned[f"weight__{w}"]))) for w in weights)
    if weight_error > 1e-12:
        raise ValueError("Independent weighting differs from archive")
    scores = {epsilon: {} for epsilon in EPSILONS}
    raw_observed = {}
    score_error = {}
    zero_rows = []
    chain_error = 0.0
    for model in MODELS:
        print(f"Scoring frozen {model}", flush=True)
        components = {epsilon: [] for epsilon in EPSILONS}
        raw_parts = []
        for fold, part in zip(range(1, 5), meta_parts):
            with np.load(source / "payload" / f"fold_{fold}.npz") as payload:
                raw = payload[model]
            codes = part.observed_state_code.to_numpy(int)
            raw_parts.append(raw[np.arange(len(part)), codes].copy())
            for epsilon in EPSILONS:
                loss = score_components(raw, codes, epsilon)
                chain_error = max(chain_error, float(np.max(np.abs(loss[:, 0] - loss[:, 1] - loss[:, 2]))))
                components[epsilon].append(loss)
            del raw
        raw_observed[model] = np.concatenate(raw_parts)
        for epsilon in EPSILONS:
            scores[epsilon][model] = np.concatenate(components[epsilon])
        score_error[model] = float(np.max(np.abs(scores[1e-12][model] - aligned[[f"{model}__{m}" for m in METRICS]].to_numpy())))
        if score_error[model] > 1e-12:
            raise ValueError(f"Archived score mismatch: {model}")
        zero = raw_observed[model] == 0
        for weighting, w in weights.items():
            positive = raw_observed[model][~zero]
            zero_rows.append({"model": model, "weighting": weighting, "epsilon": 0,
                              "unsmoothed_pattern_nll_defined": not bool(zero.any()),
                              "zero_probability_weight_mass": float(w[zero].sum()),
                              "minimum_positive_observed_probability": float(positive.min()) if len(positive) else np.nan,
                              "raw_zero_records": int(zero.sum()), **support(meta, zero)})
    pd.DataFrame(zero_rows).to_csv(out / "unsmoothed_zero_probability_audit.csv", index=False)
    epsilon_rows, subgroup, cuts, zeros, omissions = [], [], [], [], []
    for candidate in MODELS[1:]:
        for epsilon in EPSILONS:
            delta = scores[epsilon]["factorized"] - scores[epsilon][candidate]
            epsilon_rows.extend(aggregate_rows(meta, delta, candidate, epsilon))
            for subset, mask in (("candidate_raw_observed_zero", raw_observed[candidate] == 0),
                                 ("reference_raw_observed_zero", raw_observed["factorized"] == 0),
                                 ("either_raw_observed_zero", (raw_observed[candidate] == 0) | (raw_observed["factorized"] == 0))):
                for weighting, w in weights.items():
                    mass = w[mask].sum()
                    gain = w[mask] @ delta[mask]
                    for j, metric in enumerate(METRICS):
                        zeros.append({"candidate": candidate, "epsilon": epsilon, "subset": subset,
                                      "weighting": weighting, "metric": metric, "full_cohort_weight_mass": mass,
                                      "contribution_to_full_cohort": gain[j],
                                      "within_subset_mean": gain[j] / mass if mass > 0 else np.nan, **support(meta, mask)})
        delta = scores[1e-12]["factorized"] - scores[1e-12][candidate]
        subgroup.extend(subgroup_rows(meta, delta, candidate))
        for grouping in ("alpha", "outer_fold"):
            for group, indices in meta.groupby(grouping, sort=True).indices.items():
                rows = aggregate_rows(meta.iloc[indices], delta[indices], candidate, 1e-12)
                for row in rows:
                    row.update(grouping=grouping, group=group)
                cuts.extend(rows)
        omitted = cluster_omissions(meta, delta)
        omitted["candidate"] = candidate
        omissions.append(omitted)
    epsilon_table = pd.DataFrame(epsilon_rows)
    epsilon_table.to_csv(out / "epsilon_sensitivity.csv", index=False)
    subgroups = pd.DataFrame(subgroup)
    subgroups.to_csv(out / "subgroup_contributions.csv", index=False)
    subgroups[(subgroups.grouping == "observed_count") & subgroups.group.isin(["1", "2", "3", "4", "5"]) &
              (subgroups.metric == "identity_given_count_nll")].to_csv(out / "identity_contributions_by_count.csv", index=False)
    pd.DataFrame(cuts).to_csv(out / "threshold_and_fold_components.csv", index=False)
    pd.DataFrame(zeros).to_csv(out / "zero_probability_gain_contributions.csv", index=False)
    loo = pd.concat(omissions, ignore_index=True)
    loo.to_csv(out / "leave_one_cluster_out.csv.gz", index=False)
    loo_components = loo.pivot(index=["candidate", "omission_level", "omitted_state", "omitted_pws_id", "weighting"],
                               columns="metric", values="loss_reduction").reset_index()
    loo_components["count_share_of_pattern_gain"] = loo_components.count_nll / loo_components.pattern_nll
    loo_components.to_csv(out / "leave_one_cluster_out_components.csv.gz", index=False)
    fractions = loo_components.groupby(["candidate", "omission_level", "weighting"]).count_share_of_pattern_gain.agg(
        omissions="size", minimum="min", maximum="max").reset_index()
    fractions["interpretation"] = "algebraic share of net gain; influence range, not confidence interval"
    fractions.to_csv(out / "leave_one_cluster_out_count_share_summary.csv", index=False)
    summary_rows = []
    for (candidate, level, weighting, metric), part in loo.groupby(["candidate", "omission_level", "weighting", "metric"]):
        imin, imax = part.loss_reduction.idxmin(), part.loss_reduction.idxmax()
        summary_rows.append({"candidate": candidate, "omission_level": level, "weighting": weighting,
                             "metric": metric, "omissions": len(part), "minimum": part.loss_reduction.min(),
                             "maximum": part.loss_reduction.max(), "positive_estimates": int((part.loss_reduction > 0).sum()),
                             "negative_estimates": int((part.loss_reduction < 0).sum()),
                             "minimum_omitted_state": part.loc[imin, "omitted_state"],
                             "minimum_omitted_pws_id": part.loc[imin, "omitted_pws_id"],
                             "maximum_omitted_state": part.loc[imax, "omitted_state"],
                             "maximum_omitted_pws_id": part.loc[imax, "omitted_pws_id"],
                             "interpretation": "influence range, not confidence interval"})
    pd.DataFrame(summary_rows).to_csv(out / "leave_one_cluster_out_summary.csv", index=False)
    zero_listing = meta.copy()
    for model in MODELS:
        zero_listing[f"raw_observed_probability__{model}"] = raw_observed[model]
    zero_any = np.any(np.stack([raw_observed[m] == 0 for m in MODELS]), axis=0)
    zero_listing.loc[zero_any].to_csv(out / "observed_zero_probability_records.csv.gz", index=False)
    files = [source / "analysis/record_scores.csv.gz", out / "PROTOCOL.md", Path(__file__).resolve()]
    files += [source / "payload" / f"{prefix}{fold}{suffix}" for fold in range(1, 5)
              for prefix, suffix in (("metadata_fold_", ".csv.gz"), ("fold_", ".npz"))]
    audit = {"status": "PASS", "epsilon_values": EPSILONS, "primary_epsilon": 1e-12,
             "support": support(meta, np.ones(len(meta), bool)), "thresholds": sorted(meta.alpha.unique()),
             "outer_folds": sorted(int(f) for f in meta.outer_fold.unique()),
             "chain_rule_max_abs_error": chain_error, "archived_weight_max_abs_error": weight_error,
             "archived_score_max_abs_error": score_error,
             "source_sha256": {str(p.relative_to(ROOT)): source_hash(p) for p in files},
             "no_training_or_selection": True, "loo_is_not_confidence_interval": True}
    (out / "AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()

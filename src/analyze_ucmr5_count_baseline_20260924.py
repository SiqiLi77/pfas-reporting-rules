#!/usr/bin/env python3
"""Frozen historical count-prior benchmark on the final strict PFAS cohort.

Run --freeze before --run. The querywise benchmark is not assumed to define a
threshold-coherent joint process. A complete Boolean upper-set audit tests that
property using maximum-weight closure/min-cut, without enumerating all upsets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_ucmr5_joint_information_decomposition import (
    METRICS, NLL, evaluation_weights, hierarchical_bootstrap, score_distribution,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/significance_extension_20260924/count_baseline"
SOURCE = ROOT / "outputs/joint_information_decomposition_20260916"
COHORT = ROOT / "work/ucmr5_final_20260828_r1/ucmr5_artificial_censor_strict.csv.gz"
CHEMS = ("PFBS", "PFHpA", "PFHxS", "PFNA", "PFOA", "PFOS")
QUERIES = (0.05, .1, .15, .2, .3, .35, .4, .45, .55, .6, .65, .7, .8, .85, .9, .95)
ANCHORS = (0., .5, 1.)
MODEL = "count_only_log_3anchor"
MODELS = ("factorized", MODEL, "conditional_shared_shock", "conditional_gaussian_factor2")
STATES = ((np.arange(64)[:, None] >> np.arange(6)) & 1).astype(float)
COUNTS = STATES.sum(axis=1).astype(int)
COUNT_ALPHA = .5
TOLERANCE = 1e-11
BOOTSTRAP_REPS = 20000
SEED = 20260924


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "PREFIT_LOCK.json"
    if path.exists():
        raise FileExistsError("Never overwrite a prefit lock")
    inputs = [COHORT, Path(__file__),
              ROOT / "src/analyze_ucmr5_count_identity_ablation.py",
              ROOT / "src/analyze_ucmr5_joint_information_decomposition.py"]
    inputs += [SOURCE / "payload" / f"{stem}_{fold}.{ext}"
               for fold in range(1, 5)
               for stem, ext in (("fold", "npz"), ("metadata_fold", "csv.gz"))]
    protocol = {
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen before fitting and scoring this re-evaluation",
        "analysis_type": "exploratory post-selection historical-benchmark re-evaluation",
        "not_new_independent_test": True,
        "formal_folds": [1, 2, 3, 4], "validation_fold": "1 + outer_fold % 4",
        "training_folds": "all except designated outer and validation folds; includes development fold 0",
        "training_weighting": "location equal, exactly the historical count-prior rule",
        "chemical_bit_order": CHEMS, "anchors": ANCHORS, "queries": QUERIES,
        "count_class_pseudocount": COUNT_ALPHA,
        "prior": "P(K=k)/choose(6,k), uniform named identities at fixed count before marginal projection",
        "interpolation": "piecewise log-linear prior interpolation between fixed anchors",
        "projection": "iterative proportional fitting to the frozen factorized marginals; exact zero/one support, no artificial margin clipping",
        "projection_tolerance": TOLERANCE, "max_projection_cycles": 2000,
        "boundary_difference_from_legacy": "legacy clips marginals to [1e-8,1-1e-8]; current benchmark retains exact deterministic margins for a fair fixed-margin comparison",
        "primary_weighting": "jurisdiction equal, PWS equal within jurisdiction, location equal within PWS, query equal",
        "bootstrap": "paired hierarchical resampling of jurisdictions and PWS, all component covariance retained",
        "bootstrap_replicates": BOOTSTRAP_REPS, "seed": SEED,
        "models": MODELS, "metrics": METRICS,
        "coherence": "all Boolean upper sets by 64-node maximum-weight closure/min-cut for every distinct adjacent-query path; violation tolerance 1e-8",
        "interpretation_rule": "any upper-set violation means only a querywise diagnostic benchmark, not a deployable threshold-coherent replacement",
        "no_hyperparameter_selection": True,
        "sources": {str(p.relative_to(ROOT)): sha(p) for p in inputs},
    }
    dump(path, protocol)
    (OUT / "PROTOCOL.md").write_text(
        "# Frozen count-only benchmark re-evaluation\n\n"
        "This repeats a historical benchmark on the final strict cohort; it is not a new independent confirmation. "
        "Training excludes the designated test and validation jurisdictions. At alpha 0, 0.5 and 1, "
        "count frequencies receive 0.5 pseudocount per count class and are divided uniformly among the "
        "named patterns of that size. The prior is log-interpolated and projected to the exact frozen six margins. "
        "All 16 held-out query outcomes are used only for evaluation. No setting is selected by these outcomes.\n\n"
        "Primary scores use jurisdiction/PWS/location/query equal weighting. Twenty thousand paired "
        "hierarchical bootstrap replicates estimate contrast intervals. Full threshold stochastic ordering "
        "is checked through maximum-weight closure on the Boolean lattice; monotone P(K≥2) alone is insufficient. "
        "Any confirmed violation limits the benchmark to querywise diagnostic use.\n", encoding="utf-8")
    print("Frozen protocol saved", flush=True)


def labels_at(frame, alpha):
    lower = frame[[f"native_logmrl__{c}" for c in CHEMS]].to_numpy(float)
    upper = frame[[f"old_logmrl__{c}" for c in CHEMS]].to_numpy(float)
    lr = frame[[f"y_native_logratio__{c}" for c in CHEMS]].to_numpy(float)
    detected = frame[[f"y_native_detect__{c}" for c in CHEMS]].to_numpy(bool)
    value = np.exp(lower) * np.expm1(lr)
    return (detected & (np.log(np.maximum(value, 1e-30)) >= lower + alpha * (upper-lower) - 1e-12)).astype(int)


def fit_prior(frame, alpha):
    k = labels_at(frame, alpha).sum(axis=1)
    counts = np.bincount(k, minlength=7)
    pk = (counts + COUNT_ALPHA) / (len(k) + 7 * COUNT_ALPHA)
    prior = np.array([pk[t] / math.comb(6, t) for t in COUNTS])
    return prior, counts


def interpolate(alpha, priors):
    hi = int(np.searchsorted(ANCHORS, alpha))
    lo = hi - 1
    w = (alpha - ANCHORS[lo]) / (ANCHORS[hi] - ANCHORS[lo])
    z = (1-w)*np.log(priors[lo]) + w*np.log(priors[hi])
    p = np.exp(z - z.max())
    return p / p.sum()


def project(margins, prior):
    if np.min(margins) < -1e-12 or np.max(margins) > 1+1e-12:
        raise ValueError("Invalid input margins")
    target = np.clip(margins, 0, 1)
    unique, inverse = np.unique(target, axis=0, return_inverse=True)
    q = np.broadcast_to(prior, (len(unique), 64)).copy()
    for j in range(6):
        q[(unique[:, j] == 0), :] *= 1-STATES[:, j]
        q[(unique[:, j] == 1), :] *= STATES[:, j]
    q /= q.sum(axis=1, keepdims=True)
    for cycle in range(1, 2001):
        for j in range(6):
            hit = STATES[:, j].astype(bool)
            current = q[:, hit].sum(axis=1)
            yes = np.divide(unique[:, j], current, out=np.ones(len(q)), where=current > 0)
            no = np.divide(1-unique[:, j], 1-current, out=np.ones(len(q)), where=current < 1)
            q[:, hit] *= yes[:, None]
            q[:, ~hit] *= no[:, None]
            q /= q.sum(axis=1, keepdims=True)
        error = np.abs(q @ STATES - unique).max()
        if error <= TOLERANCE:
            break
    if error > TOLERANCE:
        raise ValueError(f"Projection did not converge: {error}")
    return q[inverse], {"cycles": cycle, "unique_margin_rows": len(unique),
                        "max_marginal_error": float(error),
                        "max_row_sum_error": float(np.abs(q.sum(axis=1)-1).max())}


def maximum_upset_gain(delta, nbits=6):
    """Exact max over upward-closed sets of sum(delta), via floating Dinic flow."""
    n = 2**nbits
    source, sink = n, n+1
    graph = [[] for _ in range(n+2)]
    def edge(u, v, cap):
        graph[u].append([v, len(graph[v]), float(cap)])
        graph[v].append([u, len(graph[u])-1, 0.])
    for s, w in enumerate(delta):
        if w > 0:
            edge(source, s, w)
        elif w < 0:
            edge(s, sink, -w)
        for j in range(nbits):
            if not (s & (1 << j)):
                edge(s, s | (1 << j), 2.)
    flow = 0.
    while True:
        level = [-1] * (n+2)
        level[source] = 0
        todo = deque([source])
        while todo:
            u = todo.popleft()
            for v, _, cap in graph[u]:
                if cap > 1e-14 and level[v] < 0:
                    level[v] = level[u] + 1
                    todo.append(v)
        if level[sink] < 0:
            break
        ptr = [0] * (n+2)
        def augment(u, cap):
            if u == sink:
                return cap
            while ptr[u] < len(graph[u]):
                e = graph[u][ptr[u]]
                v, rev, available = e
                if available > 1e-14 and level[v] == level[u]+1:
                    sent = augment(v, min(cap, available))
                    if sent > 1e-14:
                        e[2] -= sent
                        graph[v][rev][2] += sent
                        return sent
                ptr[u] += 1
            return 0.
        while True:
            sent = augment(source, 2.)
            if sent <= 1e-14:
                break
            flow += sent
    upset = [s for s in range(n) if level[s] >= 0]
    value = float(np.asarray(delta)[upset].sum())
    if abs(value - (np.maximum(delta, 0).sum()-flow)) > 1e-11:
        raise ValueError("Min-cut value/witness disagreement")
    for s in upset:
        if any((s | (1 << j)) not in upset for j in range(nbits)):
            raise ValueError("Returned witness is not upward closed")
    return value, upset


def tests():
    rng = np.random.default_rng(SEED)
    for nbits in (1, 2, 3):
        n = 2**nbits
        upsets = []
        for mask in range(2**n):
            ss = {s for s in range(n) if mask & (1 << s)}
            if all((s | (1 << j)) in ss for s in ss for j in range(nbits)):
                upsets.append(list(ss))
        for _ in range(20):
            a, b = rng.dirichlet(np.ones(n), 2)
            delta = b-a
            expected = max(float(delta[u].sum()) for u in upsets)
            actual, _ = maximum_upset_gain(delta, nbits)
            np.testing.assert_allclose(actual, expected, atol=1e-12)
    p = np.array([[0., .02, .04, 1., .3, .001], [1.]*6, [0.]*6])
    q, _ = project(p, np.ones(64)/64)
    np.testing.assert_allclose(q@STATES, p, atol=TOLERANCE)
    # Independent distribution from uniform prior is recovered exactly.
    product = np.prod(np.where(STATES[None] == 1, p[:, None], 1-p[:, None]), axis=2)
    np.testing.assert_allclose(q, product, atol=TOLERANCE)
    return {"max_closure_exhaustive_tests": 60, "exact_boundary_projection": "pass",
            "factorized_recovery": "pass"}


def coherence(meta, q, fold):
    n = meta.location_id.nunique()
    if len(q) != n * len(QUERIES):
        raise ValueError("Unexpected row count")
    paths = q.reshape(len(QUERIES), n, 64).transpose(1, 0, 2)
    unique, first, inverse, multiplicity = np.unique(paths.reshape(n, -1), axis=0, return_index=True, return_inverse=True, return_counts=True)
    unique = unique.reshape(-1, len(QUERIES), 64)
    counttail = np.array([(COUNTS >= k) for k in range(1, 7)]).T
    taildiff = np.diff(paths @ counttail, axis=1)
    principal = ((np.arange(64)[:, None] & np.arange(1, 64)[None]) == np.arange(1, 64)[None]).astype(float)
    principal_diff = np.diff(paths @ principal, axis=1)
    witness_rows, path_rows = [], []
    for i, path in enumerate(unique):
        for a in range(len(QUERIES)-1):
            gain, upset = maximum_upset_gain(path[a+1]-path[a])
            path_rows.append({"outer_fold": fold, "unique_path": i, "locations": int(multiplicity[i]),
                              "alpha_low": QUERIES[a], "alpha_high": QUERIES[a+1],
                              "maximum_upper_set_increase": gain,
                              "violation": gain > 1e-8})
            if gain > 1e-8:
                witness_rows.append({**path_rows[-1], "example_location_id": meta.iloc[first[i]].location_id,
                                     "upper_state_codes": ",".join(map(str, upset)),
                                     "probability_low": float(path[a, upset].sum()),
                                     "probability_high": float(path[a+1, upset].sum())})
    frame = pd.DataFrame(path_rows)
    audit = {"outer_fold": fold, "locations": n, "distinct_probability_paths": len(unique),
             "complete_upper_set_checks": len(frame),
             "violating_unique_path_transitions": int(frame.violation.sum()),
             "violating_location_transitions": int(frame.loc[frame.violation, "locations"].sum()),
             "maximum_upper_set_increase": float(frame.maximum_upper_set_increase.max()),
             "count_tail_violations": int((taildiff > 1e-8).sum()),
             "principal_upper_set_violations": int((principal_diff > 1e-8).sum())}
    return frame, pd.DataFrame(witness_rows), audit


def run():
    lock = json.loads((OUT / "PREFIT_LOCK.json").read_text())
    for name, expected in lock["sources"].items():
        if sha(ROOT/name) != expected:
            raise ValueError(f"Frozen source changed: {name}")
    dump(OUT / "UNIT_TESTS.json", tests())
    cohort = pd.read_csv(COHORT, dtype={"location_id": str, "pws_id": str, "state": str})
    if cohort.location_id.duplicated().any():
        raise ValueError("Duplicate cohort location")
    allmeta, countprobs, projection_rows, anchor_rows, folds, coherence_rows, witness_rows, coherence_audits = [], [], [], [], [], [], [], []
    for fold in range(1, 5):
        meta = pd.read_csv(SOURCE / "payload" / f"metadata_fold_{fold}.csv.gz", dtype={"location_id": str, "pws_id": str, "state": str})
        test = cohort[cohort.state_fold.eq(fold)].copy()
        val = 1 + fold % 4
        train = cohort[~cohort.state_fold.isin([fold, val])].copy()
        for key in ("location_id", "pws_id", "state"):
            if set(train[key]) & set(test[key]):
                raise ValueError(f"Train/test overlap: {key}")
        priors = []
        for alpha in ANCHORS:
            prior, counts = fit_prior(train, alpha)
            priors.append(prior)
            anchor_rows += [{"outer_fold": fold, "alpha": alpha, "count": k,
                             "training_count": int(counts[k]), "smoothed_count_probability": float(prior[COUNTS==k].sum())} for k in range(7)]
        with np.load(SOURCE / "payload" / f"fold_{fold}.npz") as payload:
            factorized = payload["factorized"]
        marginals = factorized @ STATES
        probs = []
        for alpha in QUERIES:
            idx = np.flatnonzero(meta.alpha.eq(alpha))
            if not np.array_equal(meta.iloc[idx].location_id, test.location_id):
                raise ValueError("Test/source row mismatch")
            np.testing.assert_array_equal(labels_at(test, alpha) @ (2**np.arange(6)), meta.iloc[idx].observed_state_code)
            q, audit = project(marginals[idx], interpolate(alpha, priors))
            probs.append(q)
            projection_rows.append({"outer_fold": fold, "alpha": alpha, **audit})
        q = np.concatenate(probs)
        np.savez_compressed(OUT / f"count_probabilities_fold_{fold}.npz", raw_probability=q, frozen_margins=marginals)
        cframe, witnesses, caudit = coherence(meta, q, fold)
        coherence_rows.append(cframe)
        if len(witnesses):
            witness_rows.append(witnesses)
        coherence_audits.append(caudit)
        folds.append({"outer_fold": fold, "validation_fold": val, "training_locations": len(train),
                      "evaluation_locations": len(test), "training_folds": ",".join(map(str, sorted(train.state_fold.unique()))),
                      "location_pws_jurisdiction_overlap": 0})
        allmeta.append(meta)
        countprobs.append(q)
        print(f"Fold {fold} fitted, frozen and audited: {caudit}", flush=True)
    meta = pd.concat(allmeta, ignore_index=True)
    if len(meta) != 338496 or meta.location_id.nunique() != 21156:
        raise ValueError("Wrong formal evaluation population")
    weights = evaluation_weights(meta)
    for key, w in weights.items():
        meta[f"weight__{key}"] = w
    meta.to_csv(OUT / "metadata.csv.gz", index=False)
    pd.DataFrame(anchor_rows).to_csv(OUT / "training_count_priors.csv", index=False)
    pd.DataFrame(folds).to_csv(OUT / "fold_audit.csv", index=False)
    pd.DataFrame(projection_rows).to_csv(OUT / "projection_audit.csv", index=False)
    pd.concat(coherence_rows).to_csv(OUT / "full_boolean_order_audit.csv.gz", index=False)
    pd.concat(witness_rows, ignore_index=True).to_csv(OUT / "coherence_counterexamples.csv.gz", index=False) if witness_rows else pd.DataFrame(columns=["outer_fold", "upper_state_codes"]).to_csv(OUT / "coherence_counterexamples.csv.gz", index=False)
    wide = meta.copy()
    scores, metric_rows, threshold_rows, frequency_rows, checks = {}, [], [], [], []
    for model in MODELS:
        if model == MODEL:
            raw = np.concatenate(countprobs)
        else:
            raw = np.concatenate([np.load(SOURCE / "payload" / f"fold_{f}.npz")[model] for f in range(1, 5)])
        losses, _, countq = score_distribution(raw, meta.observed_state_code.to_numpy(int))
        scores[model] = losses
        for metric, values in losses.items():
            wide[f"{model}__{metric}"] = values
            for weighting, w in weights.items():
                metric_rows.append({"model": model, "weighting": weighting, "metric": metric, "loss": float(w@values)})
                for alpha in QUERIES:
                    mask = meta.alpha.eq(alpha).to_numpy()
                    threshold_rows.append({"model": model, "weighting": weighting, "alpha": alpha, "metric": metric,
                                           "loss": float(w[mask]@values[mask]/w[mask].sum())})
        for weighting, w in weights.items():
            for k in range(7):
                frequency_rows.append({"model": model, "weighting": weighting, "count": k,
                                       "observed_frequency": float(w@(meta.observed_count.to_numpy()==k)),
                                       "predicted_frequency": float(w@countq[:, k])})
        checks.append({"model": model, "maximum_row_sum_error": float(np.abs(raw.sum(axis=1)-1).max()),
                       "minimum_raw_probability": float(raw.min()),
                       "chain_rule_error": float(np.abs(losses[NLL[0]]-losses[NLL[1]]-losses[NLL[2]]).max())})
        del raw
    metrics = pd.DataFrame(metric_rows)
    archived = pd.read_csv(SOURCE / "analysis/model_metrics.csv")
    replay = metrics[metrics.model.ne(MODEL)].merge(archived, on=["model", "weighting", "metric"], suffixes=("", "_archived"), validate="one_to_one")
    replay["absolute_difference"] = np.abs(replay.loss - replay.loss_archived)
    if replay.absolute_difference.max() > 1e-12:
        raise ValueError("Previously published score replay failure")
    replay.to_csv(OUT / "existing_model_replay.csv", index=False)
    metrics.to_csv(OUT / "model_metrics.csv", index=False)
    pd.DataFrame(threshold_rows).to_csv(OUT / "threshold_metrics.csv", index=False)
    pd.DataFrame(frequency_rows).to_csv(OUT / "count_frequencies.csv", index=False)
    wide.to_csv(OUT / "record_scores.csv.gz", index=False)
    vectors, contrast_rows = {}, []
    pairs = [("factorized", MODEL), (MODEL, "conditional_shared_shock"), (MODEL, "conditional_gaussian_factor2")]
    for reference, candidate in pairs:
        for metric in METRICS:
            d = scores[reference][metric]-scores[candidate][metric]
            vectors[f"{reference}|{candidate}|{metric}"] = d
            for weighting, w in weights.items():
                contrast_rows.append({"reference": reference, "candidate": candidate, "metric": metric,
                                      "weighting": weighting, "loss_reduction": float(w@d)})
    contrasts = pd.DataFrame(contrast_rows)
    contrasts.to_csv(OUT / "paired_contrasts_point.csv", index=False)
    print("Running 20,000 paired hierarchical bootstrap replicates", flush=True)
    intervals, draws = hierarchical_bootstrap(meta, vectors, BOOTSTRAP_REPS, SEED)
    bits = intervals.key.str.split("|", expand=True)
    intervals["reference"], intervals["candidate"], intervals["metric"] = bits[0], bits[1], bits[2]
    intervals.to_csv(OUT / "paired_contrasts_intervals.csv", index=False)
    np.savez_compressed(OUT / "bootstrap_draws.npz", values=draws, keys=np.array(list(vectors)))
    merged = contrasts[contrasts.weighting.eq("jurisdiction_pws_equal")].merge(intervals, on=["reference", "candidate", "metric"], validate="one_to_one")
    if np.abs(merged.loss_reduction-merged.estimate).max() > 1e-12:
        raise ValueError("Bootstrap point weighting mismatch")
    final = {"status": "COMPLETE", "analysis_type": lock["analysis_type"],
             "records": len(meta), "locations": int(meta.location_id.nunique()), "pws": int(meta.pws_id.nunique()),
             "jurisdictions": int(meta.state.nunique()), "bootstrap_replicates": BOOTSTRAP_REPS,
             "existing_model_maximum_replay_error": float(replay.absolute_difference.max()),
             "maximum_projection_margin_error": float(pd.DataFrame(projection_rows).max_marginal_error.max()),
             "score_audits": checks, "coherence_audits": coherence_audits,
             "threshold_coherent_on_all_evaluated_adjacent_queries": not any(x["violating_unique_path_transitions"] for x in coherence_audits),
             "continuous_threshold_coherence_proven": False,
             "caution": "Exploratory post-selection comparison; path audit covers evaluated queries, not every real alpha. Any failure makes this a querywise diagnostic baseline."}
    dump(OUT / "AUDIT.json", final)
    dump(OUT / "FILE_MANIFEST.json", {str(p.relative_to(OUT)): sha(p) for p in OUT.iterdir() if p.is_file() and p.name != "FILE_MANIFEST.json"})
    print(metrics.query("weighting == 'jurisdiction_pws_equal'").to_string(index=False), flush=True)
    print(intervals.to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.freeze:
        freeze()
    elif args.run:
        run()
    else:
        parser.error("Choose --freeze or --run")

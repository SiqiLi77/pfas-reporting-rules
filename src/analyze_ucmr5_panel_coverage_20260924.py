#!/usr/bin/env python3
"""Strict same-event Method 533 panel coverage; no concentration imputation."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from audit_ucmr_se1_strict_event_key_20260815 import read_rows, classify

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/significance_extension_20260924/panel_coverage"
ZIP5 = ROOT / "data/ucmr_raw/ucmr5-occurrence-data_final_20260828.zip"
ZIP3 = ROOT / "data/ucmr_raw/ucmr3-occurrence-data.zip"
SIX = ("PFBS", "PFHpA", "PFHxS", "PFNA", "PFOA", "PFOS")
OLD = dict(zip(SIX, (0.09, 0.01, 0.03, 0.02, 0.02, 0.04)))
REPS = 10000
SEED = 20260924


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def savejson(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def load():
    cols = ["PWSID", "FacilityID", "SamplePointID", "SampleEventCode", "CollectionDate", "SampleID", "Contaminant", "MRL", "Units", "MethodID", "AnalyticalResultsSign", "AnalyticalResultValue", "State", "Size", "FacilityWaterType"]
    parts = []
    total = 0
    with zipfile.ZipFile(ZIP5) as z, z.open("UCMR5_All.txt") as f:
        for c in pd.read_csv(f, sep="\t", encoding="latin-1", dtype=str, usecols=cols, keep_default_na=False, chunksize=200000):
            total += len(c)
            parts.append(c.loc[(c.SampleEventCode == "SE1") & (c.Contaminant != "lithium")].copy())
    d = pd.concat(parts, ignore_index=True)
    d["location_id"] = d.PWSID + "|" + d.FacilityID + "|" + d.SamplePointID
    d["value"] = pd.to_numeric(d.AnalyticalResultValue, errors="coerce")
    d["mrl"] = pd.to_numeric(d.MRL, errors="coerce")
    d["reported"] = d.AnalyticalResultsSign == "="
    d["valid"] = d.AnalyticalResultsSign.isin(["=", "<"]) & d.mrl.gt(0) & (~d.reported | (d.value.notna() & d.value.ge(d.mrl - 1e-12)))
    d["valid"] &= d.CollectionDate.ne("") & d.SampleID.ne("") & d.Units.eq("µg/L")
    return d, total


def strict_subset(d, analytes, method=None):
    x = d.loc[d.Contaminant.isin(analytes)].copy()
    stat = x.groupby("location_id").agg(n=("Contaminant", "size"), analytes=("Contaminant", "nunique"), dates=("CollectionDate", "nunique"), samples=("SampleID", "nunique"), valid=("valid", "all"), methods=("MethodID", "nunique"))
    ok = stat.n.eq(len(analytes)) & stat.analytes.eq(len(analytes)) & stat.dates.eq(1) & stat.samples.eq(1) & stat.valid
    if method is not None:
        methods_ok = x.groupby("location_id").MethodID.agg(lambda a: set(a) == {method})
        ok &= methods_ok
    eligible = stat.index[ok]
    return x.loc[x.location_id.isin(eligible)].copy(), stat.assign(eligible=ok)


def feature_spec(d, analytes):
    features = {"all": np.ones(len(d), dtype=float)}
    specs = []
    def add(name, num, den=None):
        features[name] = np.asarray(num, dtype=float)
        den_name = "all" if den is None else name + "__denominator"
        if den is not None:
            features[den_name] = np.asarray(den, dtype=float)
        specs.append((name, name, den_name))
    for col in ["k6_old", "k6_native", "k25_native"]:
        add(col + "_any", d[col] >= 1)
        add(col + "_multiple", d[col] >= 2)
        for k in range(0, int(d[col].max()) + 1):
            add(col + f"_count_{k}", d[col] == k)
    add("six_native_zero_to_panel_any", (d.k6_native == 0) & (d.k25_native >= 1), d.k6_native == 0)
    add("six_native_zero_to_panel_multiple", (d.k6_native == 0) & (d.k25_native >= 2), d.k6_native == 0)
    add("six_native_one_to_panel_multiple", (d.k6_native == 1) & (d.k25_native >= 2), d.k6_native == 1)
    add("panel_positive_missed_by_six", (d.k25_native >= 1) & (d.k6_native == 0), d.k25_native >= 1)
    add("panel_multiple_not_multiple_in_six", (d.k25_native >= 2) & (d.k6_native < 2), d.k25_native >= 2)
    add("extra_panel_any", d.k_added >= 1)
    for a in analytes:
        add("analyte__" + a, d["report__" + a])
    names = list(features)
    return np.column_stack([features[n] for n in names]), names, specs


def estimates(d, analytes, cohort):
    f, names, specs = feature_spec(d, analytes)
    # Every bootstrap samples PWS with replacement within the observed jurisdiction.
    # Whole event clusters travel together. Jurisdictions are treated as fixed strata.
    group = d.groupby(["State", "PWSID"], sort=True).indices
    gkeys = list(group)
    sums = np.stack([f[group[k]].sum(axis=0) for k in gkeys])
    counts = np.array([len(group[k]) for k in gkeys], dtype=float)
    means = sums / counts[:, None]
    states = sorted(d.State.unique())
    si = {s: np.array([i for i, k in enumerate(gkeys) if k[0] == s]) for s in states}
    point = {
        "event_equal": f.mean(axis=0),
        "pws_equal": means.mean(axis=0),
        "jurisdiction_pws_equal": np.stack([means[idx].mean(axis=0) for idx in si.values()]).mean(axis=0),
    }
    draws = {w: np.zeros((REPS, f.shape[1])) for w in point}
    draw_n = np.zeros(REPS)
    rng = np.random.default_rng(SEED)
    for idx in si.values():
        p = np.ones(len(idx)) / len(idx)
        for start in range(0, REPS, 200):
            end = min(start + 200, REPS)
            w = rng.multinomial(len(idx), p, size=end - start)
            draws["event_equal"][start:end] += w @ sums[idx]
            draw_n[start:end] += w @ counts[idx]
            draws["pws_equal"][start:end] += w @ means[idx] / len(gkeys)
            draws["jurisdiction_pws_equal"][start:end] += w @ means[idx] / len(idx) / len(states)
    draws["event_equal"] /= draw_n[:, None]
    rows = []
    for metric, numerator, denominator in specs:
        ni, di = names.index(numerator), names.index(denominator)
        for weight, p in point.items():
            b = draws[weight]
            vals = np.divide(b[:, ni], b[:, di], out=np.full(REPS, np.nan), where=b[:, di] > 0)
            q = np.nanquantile(vals, [0.025, 0.975])
            rows.append(dict(cohort=cohort, metric=metric, weighting=weight, numerator_events=int(f[:, ni].sum()), denominator_events=int(f[:, di].sum()), estimate=float(p[ni] / p[di]), ci95_lower=float(q[0]), ci95_upper=float(q[1]), finite_bootstrap_replicates=int(np.isfinite(vals).sum()), bootstrap_replicates=REPS))
    return pd.DataFrame(rows)


def pattern_table(d, analytes, cohort):
    combos = d.apply(lambda r: " + ".join(a for a in analytes if r["report__" + a]) or "None reported", axis=1)
    c = combos.value_counts().rename_axis("reported_combination").reset_index(name="events")
    c = c.sort_values(["events", "reported_combination"], ascending=[False, True]).reset_index(drop=True)
    c["cohort"] = cohort
    c["denominator_events"] = len(d)
    c["event_fraction"] = c.events / len(d)
    c["selected_top8_positive"] = False
    c.loc[c.loc[c.reported_combination != "None reported"].head(8).index, "selected_top8_positive"] = True
    return c


def verify_export_against_raw(events, metrics, analytes):
    """Second, csv-streaming calculation checks exported event flags and counts."""
    expected = events.to_dict(orient="index")
    observed = {key: [0, 0, 0, 0] for key in expected}
    with zipfile.ZipFile(ZIP5) as z, z.open("UCMR5_All.txt") as binary:
        reader = csv.DictReader((line.decode("latin-1") for line in binary), delimiter="\t")
        for r in reader:
            if r["SampleEventCode"] != "SE1" or r["MethodID"] != "EPA 533":
                continue
            key = "|".join(r[c] for c in ["PWSID", "FacilityID", "SamplePointID"])
            if key not in expected:
                continue
            ex = expected[key]
            assert (r["CollectionDate"], r["SampleID"]) == (ex["CollectionDate"], ex["SampleID"])
            a = r["Contaminant"]
            flag = r["AnalyticalResultsSign"] == "="
            assert a in analytes and flag == bool(ex["report__" + a])
            counts = observed[key]
            counts[3] += 1
            counts[2] += flag
            if a in SIX:
                counts[1] += flag
                counts[0] += bool(flag and float(r["AnalyticalResultValue"]) >= OLD[a])
    assert all(vals == [expected[key]["k6_old"], expected[key]["k6_native"], expected[key]["k25_native"], 25] for key, vals in observed.items())
    # Independent pandas grouping reproduces each primary hierarchical endpoint.
    for cohort, subset in [("matched", events.loc[events.matched]), ("all_strict6", events)]:
        for column in ["k6_old", "k6_native", "k25_native"]:
            test = subset[["State", "PWSID"]].copy()
            test["endpoint"] = subset[column].ge(2).astype(float)
            sys = test.groupby(["State", "PWSID"]).endpoint.mean()
            result = {"event_equal": test.endpoint.mean(), "pws_equal": sys.mean(), "jurisdiction_pws_equal": sys.groupby(level=0).mean().mean()}
            for weight, value in result.items():
                actual = metrics.loc[(metrics.cohort==cohort) & (metrics.metric==column+"_multiple") & (metrics.weighting==weight), "estimate"].iloc[0]
                assert abs(value-actual) < 1e-12
    return dict(status="PASS", raw_rows_checked=25*len(events), exported_events_checked=len(events), all_per_analyte_reporting_flags_match=True, all_three_count_columns_match=True, independent_grouped_primary_points_match=True)


def make_figure(m, d):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = ["#C5829A", "#2F8E9B", "#AC864C"]
    fig, axes = plt.subplots(1, 2, figsize=(9.1, 3.6), gridspec_kw={"width_ratios": [1, 1.15]})
    weights = ["event_equal", "jurisdiction_pws_equal"]
    labels = ["Six PFAS\nolder cutoffs", "Six PFAS\nnative cutoffs", "25 PFAS\nnative cutoffs"]
    for i, w in enumerate(weights):
        rows = m.loc[(m.weighting == w) & m.metric.isin(["k6_old_multiple", "k6_native_multiple", "k25_native_multiple"])].set_index("metric").loc[["k6_old_multiple", "k6_native_multiple", "k25_native_multiple"]]
        x = np.arange(3) + (i - .5) * .18
        axes[0].errorbar(x, rows.estimate * 100, yerr=np.vstack([rows.estimate - rows.ci95_lower, rows.ci95_upper - rows.estimate]) * 100, fmt="o" if i == 0 else "s", markersize=5, color=colors[i], capsize=3, lw=1.1, label="Event equal" if i == 0 else "Jurisdiction / system equal")
    axes[0].set_xticks(range(3), labels)
    axes[0].set_ylabel("Events with ≥2 PFAS reported (%)")
    axes[0].set_ylim(bottom=0)
    axes[0].legend(frameon=False, fontsize=7.5, loc="upper left")
    groups = [d.k6_native.eq(0), d.k6_native.eq(1)]
    names = ["Six PFAS: none reported", "Six PFAS: one reported"]
    left = np.zeros(2)
    for k, color, label in [(0, "#E3E4E5", "None"), (1, "#80B9BF", "One"), (2, "#C5829A", "Two or more")]:
        vals = np.array([100 * np.mean(np.minimum(d.loc[g, "k25_native"], 2) == k) for g in groups])
        axes[1].barh(np.arange(2), vals, left=left, color=color, edgecolor="white", height=.45, label=label)
        for j, v in enumerate(vals):
            if v >= 7:
                axes[1].text(left[j] + v / 2, j, f"{v:.1f}%", ha="center", va="center", fontsize=8)
        left += vals
    axes[1].set_yticks(np.arange(2), [f"{n}\n(n = {int(g.sum()):,})" for n, g in zip(names, groups)])
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, 100)
    axes[1].set_xlabel("Composition under the 25-PFAS panel (%)")
    axes[1].legend(frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(.5, 1.23), ncol=3, title="Number of PFAS reported", title_fontsize=8)
    for ax, panel in zip(axes, "ab"):
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(-.13, 1.08, panel, transform=ax.transAxes, fontsize=13, fontweight="bold")
        ax.tick_params(labelsize=8)
    fig.subplots_adjust(left=.085, right=.99, bottom=.24, top=.83, wspace=.72)
    fig.savefig(OUT / "panel_coverage_figure.svg")
    fig.savefig(OUT / "panel_coverage_figure.png", dpi=240)
    plt.close(fig)


def main():
    assert (OUT / "PROTOCOL.md").exists()
    sources_before = {str(p.relative_to(ROOT)): sha(p) for p in [ZIP5, ZIP3]}
    d, raw_rows = load()
    a25 = sorted(d.loc[d.MethodID.eq("EPA 533"), "Contaminant"].unique())
    a29 = sorted(d.Contaminant.unique())
    assert len(a25) == 25 and len(a29) == 29 and set(SIX).issubset(a25)
    print("Method533 analytes", a25, flush=True)
    s6, audit6 = strict_subset(d, SIX, "EPA 533")
    s25, audit25 = strict_subset(d, a25, "EPA 533")
    for field in ["PWSID", "FacilityID", "SamplePointID", "State", "Size", "FacilityWaterType"]:
        assert s25.groupby("location_id")[field].nunique().eq(1).all(), f"Within-event metadata conflict: {field}"
    # Small independent adversarial fixtures exercise the sample-key exclusion rules.
    fixture = s25.loc[s25.location_id == s25.location_id.iloc[0]].copy()
    assert len(strict_subset(fixture, a25, "EPA 533")[0]) == 25
    assert len(strict_subset(pd.concat([fixture, fixture.iloc[[0]]]), a25, "EPA 533")[0]) == 0
    assert len(strict_subset(fixture.iloc[1:], a25, "EPA 533")[0]) == 0
    for field, value in [("CollectionDate", "different date"), ("SampleID", "different sample"), ("valid", False), ("MethodID", "different method")]:
        bad = fixture.copy()
        bad.loc[bad.index[0], field] = value
        assert len(strict_subset(bad, a25, "EPA 533")[0]) == 0, field
    # Same date / same SampleID is deliberately stricter than administrative SE1 matching.
    s29, audit29 = strict_subset(d, a29)
    r3, _ = read_rows(ZIP3, "UCMR3")
    _, strict3, _, audit3 = classify(r3)
    u3ids = {"|".join(k) for k in strict3}
    rep6 = s6.pivot(index="location_id", columns="Contaminant", values="reported")[list(SIX)]
    val6 = s6.pivot(index="location_id", columns="Contaminant", values="value")[list(SIX)]
    k6 = rep6.sum(axis=1).astype(int)
    old6 = (rep6 & val6.ge(pd.Series(OLD), axis=1)).sum(axis=1).astype(int)
    matched = sorted(set(rep6.index) & u3ids)
    assert all(abs(strict3[tuple(k.split("|"))][a]["mrl"] - OLD[a]) < 1e-12 for k in matched for a in SIX)
    oldzero = [k for k in matched if k6[k] >= 2 and old6[k] == 0]
    reconcile = dict(strict6_events=len(rep6), matched_events=len(matched), ucmr3_native_multiple=sum(sum(strict3[tuple(k.split("|"))][a]["detected"] for a in SIX) >= 2 for k in matched), ucmr5_native_multiple=int((k6.loc[matched] >= 2).sum()), ucmr5_old_multiple=int((old6.loc[matched] >= 2).sum()), ucmr5_native_multiple_to_old_zero=len(oldzero))
    assert reconcile == dict(strict6_events=26461, matched_events=7677, ucmr3_native_multiple=36, ucmr5_native_multiple=829, ucmr5_old_multiple=10, ucmr5_native_multiple_to_old_zero=772), reconcile
    print("Reconciled", reconcile, flush=True)
    meta = s25.drop_duplicates("location_id").set_index("location_id")[["PWSID", "FacilityID", "SamplePointID", "CollectionDate", "SampleID", "State", "Size", "FacilityWaterType"]]
    rep25 = s25.pivot(index="location_id", columns="Contaminant", values="reported")[a25]
    events = meta.join(rep25.rename(columns=lambda a: "report__" + a).astype(int))
    events["k6_old"] = old6.reindex(events.index)
    events["k6_native"] = k6.reindex(events.index)
    events["k25_native"] = rep25.sum(axis=1).astype(int)
    events["k_added"] = events.k25_native - events.k6_native
    events["matched"] = events.index.isin(matched)
    events["original_native_multiple_to_old_zero"] = events.index.isin(oldzero)
    assert events.k6_old.notna().all() and events.k6_native.notna().all()
    events[["k6_old", "k6_native"]] = events[["k6_old", "k6_native"]].astype(int)
    assert (events.k6_old <= events.k6_native).all() and (events.k6_native <= events.k25_native).all()
    events.to_csv(OUT / "strict25_event_records.csv.gz", index=True)
    for name, audit in [("six", audit6), ("method533_25", audit25), ("all29_exact_sample", audit29)]:
        audit.to_csv(OUT / f"eligibility_{name}.csv.gz")
    # Audit same-date only separately; never call different SampleIDs one sample.
    date29ok = (audit29.n == 29) & (audit29.analytes == 29) & (audit29.dates == 1) & audit29.valid
    method_audit = dict(all29_unique_analyte_locations=int(((audit29.n == 29) & (audit29.analytes == 29)).sum()), all29_same_day_unique_locations=int(date29ok.sum()), all29_exact_same_sample_locations=int(audit29.eligible.sum()), matched_same_day29=int(sum(date29ok.reindex(matched, fill_value=False))), matched_same_sample29=int(sum(audit29.eligible.reindex(matched, fill_value=False))), policy="29-analyte results not pooled across methods/samples; this extension uses same-sample EPA 533 only")
    excluded = []
    for cohort, ids in [("matched", matched), ("all_strict6", rep6.index)]:
        for key in ids:
            if key not in events.index:
                status = audit25.loc[key].to_dict() if key in audit25.index else {"n": 0}
                excluded.append(dict(cohort=cohort, location_id=key, k6_native=int(k6[key]), k6_old=int(old6[key]), **status))
    pd.DataFrame(excluded, columns=["cohort", "location_id", "k6_native", "k6_old", "n", "analytes", "dates", "samples", "valid", "methods", "eligible"]).to_csv(OUT / "excluded_panel_events.csv", index=False)
    meta_a = d.groupby(["Contaminant", "MethodID", "MRL", "Units"]).size().rename("se1_rows").reset_index()
    meta_a["shared_six"] = meta_a.Contaminant.isin(SIX)
    meta_a.to_csv(OUT / "analytes_methods_cutoffs.csv", index=False)
    metrics, transitions, patterns, source_counts, pws_summary, subgroup_analytes, subgroup_combinations = [], [], [], [], [], [], []
    coverage = []
    for cohort, subset, base_n in [("matched", events.loc[events.matched], len(matched)), ("all_strict6", events, len(rep6))]:
        n = len(subset)
        coverage.append(dict(cohort=cohort, base_events=base_n, eligible_events=n, excluded_events=base_n-n, coverage_fraction=n/base_n, pws=subset.PWSID.nunique(), jurisdictions=subset.State.nunique()))
        print("Bootstrap", cohort, n, flush=True)
        metrics.append(estimates(subset, a25, cohort))
        patterns.append(pattern_table(subset, a25, cohort))
        groups = {"six_native_zero": subset.k6_native.eq(0), "six_native_one": subset.k6_native.eq(1), "six_native_multiple": subset.k6_native.ge(2), "six_old_zero": subset.k6_old.eq(0), "original_native_multi_to_old_zero": subset.original_native_multiple_to_old_zero}
        for subgroup, mask in groups.items():
            part = subset.loc[mask]
            for a in a25:
                if a not in SIX:
                    nrep = int(part["report__" + a].sum())
                    subgroup_analytes.append(dict(cohort=cohort, subgroup=subgroup, analyte=a, reported_events=nrep, denominator_events=len(part), event_fraction=nrep/len(part) if len(part) else None))
            pats = pattern_table(part, a25, cohort)
            pats["subgroup"] = subgroup
            # The top8 designation belongs to the original full eligible cohort only.
            subgroup_combinations.append(pats.drop(columns="selected_top8_positive"))
        for stage in ["k6_old", "k6_native", "k25_native"]:
            cnt = subset[stage].value_counts().sort_index()
            for k, count in cnt.items():
                source_counts.append(dict(cohort=cohort, stage=stage, reported_count=int(k), events=int(count), denominator_events=n, event_fraction=count/n))
            bypws = subset.groupby("PWSID")[stage].max()
            pws_summary.append(dict(cohort=cohort, stage=stage, denominator_systems=len(bypws), systems_any=int((bypws>=1).sum()), systems_multiple=int((bypws>=2).sum())))
        for before, after in [("k6_old", "k6_native"), ("k6_native", "k25_native"), ("k6_old", "k25_native")]:
            counts = subset.groupby([before, after]).size()
            for (a, b), count in counts.items():
                transitions.append(dict(cohort=cohort, before_stage=before, after_stage=after, before_count=int(a), after_count=int(b), events=int(count), denominator_events=n, event_fraction=count/n))
    met = pd.concat(metrics, ignore_index=True)
    independent_check = verify_export_against_raw(events, met, a25)
    savejson("independent_checks.json", independent_check)
    met.to_csv(OUT / "rates_with_cluster_bootstrap.csv", index=False)
    pd.concat(patterns, ignore_index=True).to_csv(OUT / "observed_25pfas_patterns.csv", index=False)
    pd.DataFrame(transitions).to_csv(OUT / "count_transitions.csv", index=False)
    pd.DataFrame(source_counts).to_csv(OUT / "count_distributions.csv", index=False)
    pd.DataFrame(pws_summary).to_csv(OUT / "system_level_counts.csv", index=False)
    pd.DataFrame(subgroup_analytes).to_csv(OUT / "added_analytes_by_six_panel_status.csv", index=False)
    pd.concat(subgroup_combinations, ignore_index=True).to_csv(OUT / "combinations_by_six_panel_status.csv", index=False)
    pd.DataFrame(coverage).to_csv(OUT / "cohort_coverage.csv", index=False)
    follow = events.loc[events.original_native_multiple_to_old_zero]
    follow_summary = dict(original_events=772, eligible_events=len(follow), excluded_events=772-len(follow), added_panel_any=int((follow.k_added>=1).sum()), count25_multiple=int((follow.k25_native>=2).sum()), count25_distribution={str(k):int(v) for k,v in follow.k25_native.value_counts().sort_index().items()})
    follow.to_csv(OUT / "original772_panel_followup.csv.gz")
    # Root integrates result figures after scientific audit; no plotting dependency required.
    main = met.loc[(met.cohort == "matched") & (met.weighting == "event_equal")].set_index("metric")
    weighted = met.loc[(met.cohort == "matched") & (met.weighting == "jurisdiction_pws_equal")].set_index("metric")
    def result(key):
        r = main.loc[key]
        return f"{int(r.numerator_events):,}/{int(r.denominator_events):,}（{100*r.estimate:.2f}%，95%区间 {100*r.ci95_lower:.2f}–{100*r.ci95_upper:.2f}%）"
    added = met.loc[(met.cohort=="matched") & (met.weighting=="event_equal") & met.metric.str.startswith("analyte__")].copy()
    added["analyte"] = added.metric.str.replace("analyte__", "", regex=False)
    added = added.loc[~added.analyte.isin(SIX)].sort_values(["numerator_events", "analyte"], ascending=[False,True])
    added.to_csv(OUT / "added_analyte_reporting_matched.csv", index=False)
    report = f"""# 同事件 PFAS 面板覆盖分析

## 结果及其适用范围

严格日期和 SampleID 规则复现全部 26,461 个六 PFAS 事件及 7,677 个跨周期匹配事件。原有的 36、829、10 个多 PFAS 事件及 772 个变为六种均未报告的事件完全复现。新增面板结果仅用于 UCMR 5 同记录组成描述，不扩大历史跨周期可比物种集合。

匹配队列中完整 EPA 533 的 25 种同样本面板覆盖 {coverage[0]['eligible_events']:,}/{coverage[0]['base_events']:,} 个事件，{coverage[0]['pws']:,} 个供水系统，{coverage[0]['jurisdictions']} 个辖区。全 strict6 队列覆盖 {coverage[1]['eligible_events']:,}/{coverage[1]['base_events']:,}。无样本日期合并，无低于报告限浓度插补。

匹配队列排除的15个事件中，14个在六种原生阈值均未报告，1个仅报告一种；829个原生多PFAS事件和10个旧阈值多PFAS事件全部保留。排除来自面板不完整或日期/SampleID不一致，逐例列于 excluded_panel_events.csv，不能把7,662的比例沿用7,677作分母。

### 阈值限制与面板限制

匹配同一批事件中，至少两种 PFAS 报告的事件分别为：六种旧阈值 {result('k6_old_multiple')}；六种原生阈值 {result('k6_native_multiple')}；25 种原生阈值 {result('k25_native_multiple')}。辖区内系统等权、再辖区等权的对应点估计为 {100*weighted.loc['k6_old_multiple','estimate']:.3f}%、{100*weighted.loc['k6_native_multiple','estimate']:.3f}%、{100*weighted.loc['k25_native_multiple','estimate']:.3f}%。这是同事件的报告范围变化，不是全国污染发生率或健康风险。

六种原生阈值均未报告的事件中，25 种面板至少报告一种为 {result('six_native_zero_to_panel_any')}，至少报告两种为 {result('six_native_zero_to_panel_multiple')}。六种原生阈值仅报告一种的事件中，25 种面板报告至少两种为 {result('six_native_one_to_panel_multiple')}。

25 种面板阳性事件中，被六种面板完全遗漏的比例为 {result('panel_positive_missed_by_six')}。25 种面板多 PFAS 事件中，六种面板未呈现为多 PFAS 的比例为 {result('panel_multiple_not_multiple_in_six')}。条件分母不同，不能互换。

原来 772 个六种原生多报告、旧阈值六种均未报告的事件中，完整25种面板可评价 {len(follow)} 个，其中 {follow_summary['added_panel_any']} 个还报告六种之外的 PFAS。因为这组按六种原生多报告定义，其25种原生多报告计数必然全保留；该计数不是独立发现。

### 新增分析物与组合

匹配队列19种新增分析物中，报告最多的五种为 {', '.join(str(r.analyte)+' '+str(int(r.numerator_events)) for _, r in added.head(5).iterrows())}。每种分析物及所有观测完整模式均另存数据表；展示组合按原生频数前八位选定，不按效应大小筛选。组合不被解释为污染源或迁移机制。

### 29种面板边界

29种各一次的行政SE1位置有 {method_audit['all29_unique_analyte_locations']:,} 个；其中同一采样日 {method_audit['all29_same_day_unique_locations']:,} 个，同日且全部相同 SampleID 只有 {method_audit['all29_exact_same_sample_locations']:,} 个。方法间相同日期不自动证明同一水样。本轮主分析严格限于 EPA 533 的25种，不拼接另一天或另一标识的样本。

## 对稿件的意义

这里直接检验历史共同六种面板对当前可观察组成的覆盖。若只保留六种共同 PFAS，即使沿用现代低报告限，仍会遗漏部分报告事件或把多种共同报告描述为单一报告。可比性分析应保留六种共同阈值口径，同时用严格同事件的更宽原生面板描述现代可观察组成。不能把面板变宽所产生的计数增量解释为环境浓度升高，也不能由报告频率推断毒性负担。

## 统计和溯源

区间为固定辖区分层、辖区内供水系统整簇重抽样10,000次的百分位区间（seed={SEED}），不代表全国概率抽样不确定性。事件等权与辖区/系统等权是不同的目标总体。单一SE1事件每位置，无时间趋势估计。原始浓度、报告标识、方法、MRL和单位均由原始ZIP核查；低于MRL仅编码为该阈值下未报告，不被当成化学浓度零。
"""
    (OUT / "FINDINGS_ZH.md").write_text(report, encoding="utf-8")
    sources_after = {str(p.relative_to(ROOT)): sha(p) for p in [ZIP5, ZIP3]}
    assert sources_after == sources_before
    savejson("audit.json", dict(status="PASS", raw_rows=raw_rows, se1_pfas_rows=len(d), panel25_analytes=a25, all29_analytes=a29, signs=d.AnalyticalResultsSign.value_counts().to_dict(), invalid_rows=int((~d.valid).sum()), eligibility_fixture_checks="PASS: complete accepted; duplicate, missing analyte, different date, different SampleID, invalid result, different method rejected", within_panel_metadata="PASS", old_cutoffs_match_all_matched_ucmr3_mrls=True, reconciliation=reconcile, coverage=coverage, method29_audit=method_audit, original772_followup=follow_summary, source_sha256=sources_before, protocol_sha256=sha(OUT/"PROTOCOL.md"), script_sha256=sha(Path(__file__)), bootstrap=dict(replicates=REPS, seed=SEED, clusters="PWS", strata="jurisdiction fixed", scheme="PWS sampling with replacement within observed jurisdiction", interval="percentile95")))
    savejson("output_manifest.json", {str(p.relative_to(OUT)): sha(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name != "output_manifest.json"})
    print(report, flush=True)


if __name__ == "__main__":
    main()

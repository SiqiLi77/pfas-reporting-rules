#!/usr/bin/env python3
"""Build the locked, temporally ordered PFAS cohort. No model fitting or scoring."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import zipfile
from collections import Counter, defaultdict
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/monitoring_decision_20260925/cohort"
ARCHIVE = ROOT / "data/ucmr_raw/ucmr5-occurrence-data_final_20260828.zip"
CACHE = ROOT / "work/ucmr5_final_20260828_r1/ucmr5_artificial_censor_strict.csv.gz"
REFERENCE = ROOT / "outputs/significance_extension_20260924/panel_coverage/strict25_event_records.csv.gz"
PROTOCOL = ROOT / "outputs/monitoring_decision_20260925/ANALYSIS_PROTOCOL.md"
SIX = ("PFBS", "PFHpA", "PFHxS", "PFNA", "PFOA", "PFOS")
MRL = {
    "11Cl-PF3OUdS": .005, "4:2 FTS": .003, "6:2 FTS": .005,
    "8:2 FTS": .005, "9Cl-PF3ONS": .002, "ADONA": .003,
    "HFPO-DA": .005, "NFDHA": .020, "PFBA": .005, "PFBS": .003,
    "PFDA": .003, "PFDoA": .003, "PFEESA": .003, "PFHpA": .003,
    "PFHpS": .003, "PFHxA": .003, "PFHxS": .003, "PFMBA": .003,
    "PFMPA": .004, "PFNA": .004, "PFOA": .004, "PFOS": .004,
    "PFPeA": .003, "PFPeS": .004, "PFUnA": .002,
}
A25 = tuple(MRL)
META_MAP = {"PWSID": "pws_id", "State": "state", "Region": "region",
            "Size": "pws_size", "FacilityWaterType": "water_type"}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def savejson(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                     allow_nan=False), encoding="utf-8")


@lru_cache(maxsize=10000)
def parsedate(value):
    return datetime.strptime(value.strip(), "%m/%d/%Y").date()


def location(row):
    return "|".join(row[k].strip() for k in ("PWSID", "FacilityID", "SamplePointID"))


def raw_rows():
    with zipfile.ZipFile(ARCHIVE) as z, z.open("UCMR5_All.txt") as f:
        yield from csv.DictReader(io.TextIOWrapper(f, encoding="latin-1"), delimiter="\t")


def row_reasons(row):
    reasons = []
    a = row["Contaminant"]
    sign = row["AnalyticalResultsSign"].strip()
    try:
        mrl = float(row["MRL"])
    except ValueError:
        mrl = float("nan")
    if not math.isfinite(mrl) or mrl <= 0 or abs(mrl - MRL[a]) > 1e-12:
        reasons.append("invalid_or_unexpected_MRL")
    if row["MethodID"].strip() != "EPA 533":
        reasons.append("not_EPA533")
    if row["Units"].strip() != "µg/L":
        reasons.append("invalid_units")
    if sign not in {"=", "<"}:
        reasons.append("invalid_sign")
    value = row["AnalyticalResultValue"].strip()
    if sign == "=":
        try:
            v = float(value)
        except ValueError:
            v = float("nan")
        if not math.isfinite(v) or v < mrl - 1e-12:
            reasons.append("invalid_reported_value")
    elif value:
        try:
            v = float(value)
        except ValueError:
            v = float("nan")
        if not math.isfinite(v) or v >= mrl:
            reasons.append("invalid_censored_value")
    if not row["SampleID"].strip():
        reasons.append("blank_sample_id")
    return reasons


def panel_reasons(rows, analytes):
    reasons = []
    counts = Counter(r["Contaminant"] for r in rows)
    if set(counts) != set(analytes):
        reasons.append("missing_or_unexpected_analyte")
    if len(rows) != len(analytes) or any(v != 1 for v in counts.values()):
        reasons.append("duplicate_or_incorrect_row_count")
    for field, label in [("CollectionDate", "multiple_dates"),
                         ("SampleID", "multiple_sample_ids"),
                         ("SampleEventCode", "multiple_event_codes")]:
        if len({r[field].strip() for r in rows}) != 1:
            reasons.append(label)
    for field in META_MAP:
        if len({r[field].strip() for r in rows}) != 1:
            reasons.append("inconsistent_panel_" + field)
    reasons += [x for r in rows for x in row_reasons(r)]
    return list(dict.fromkeys(reasons))


def keep_first_date(first_dates, first_rows, key, dt, row):
    if key not in first_dates or dt < first_dates[key]:
        first_dates[key] = dt
        first_rows[key] = [row]
    elif dt == first_dates[key]:
        first_rows[key].append(row)


def fixture_checks():
    good = [{"Contaminant": a, "MRL": str(MRL[a]), "MethodID": "EPA 533",
             "Units": "µg/L", "AnalyticalResultsSign": "<", "AnalyticalResultValue": "",
             "CollectionDate": "2/1/2024", "SampleID": "s2", "SampleEventCode": "SE2",
             "PWSID": "001", "State": "01", "Region": "1", "Size": "L",
             "FacilityWaterType": "GW"} for a in A25]
    assert panel_reasons(good, A25) == []
    assert panel_reasons(good[:-1], A25)
    assert panel_reasons(good + [good[0].copy()], A25)
    for key, val in [("SampleID", "other"), ("SampleEventCode", "SE3"),
                     ("MethodID", "EPA 537.1"), ("Units", "ng/L"),
                     ("MRL", "0"), ("AnalyticalResultsSign", "?")]:
        altered = [r.copy() for r in good]
        altered[0][key] = val
        assert panel_reasons(altered, A25), key
    days, records = {}, defaultdict(list)
    for r in good:
        keep_first_date(days, records, "x", parsedate("3/1/2024"), r.copy())
    for r in good[:-1]:
        keep_first_date(days, records, "x", parsedate("2/1/2024"), r.copy())
    assert days["x"] == parsedate("2/1/2024")
    assert "missing_or_unexpected_analyte" in panel_reasons(records["x"], A25)
    # A complete later day cannot repair an incomplete earlier day, even if seen first.
    for r in good:
        keep_first_date(days, records, "x", parsedate("4/1/2024"), r.copy())
    assert len(records["x"]) == 24
    days, records = {}, defaultdict(list)
    for sample in ("one", "two"):
        for r in good:
            rr = dict(r, SampleID=sample)
            keep_first_date(days, records, "x", parsedate("2/1/2024"), rr)
    assert "multiple_sample_ids" in panel_reasons(records["x"], A25)
    return "PASS: complete accepted; missing, duplicate, SampleID, event code, method, units, MRL and sign defects rejected; earliest incomplete day retained; competing same-day complete samples excluded"


def independent_followup_replay(d, candidates, first_dates, raw_count):
    """Different parser/vectorized joins; no call to panel_reasons or raw_rows."""
    cols = ["PWSID", "FacilityID", "SamplePointID", "CollectionDate", "SampleID",
            "SampleEventCode", "Contaminant", "MethodID", "MRL", "Units",
            "AnalyticalResultsSign", "AnalyticalResultValue"]
    parts, count = [], 0
    with zipfile.ZipFile(ARCHIVE) as z, z.open("UCMR5_All.txt") as f:
        for chunk in pd.read_csv(f, sep="\t", encoding="latin-1", dtype=str,
                                 keep_default_na=False, usecols=cols, chunksize=200000):
            count += len(chunk)
            x = chunk.loc[chunk.SampleEventCode.isin(["SE2", "SE3", "SE4"]) & chunk.Contaminant.isin(A25)].copy()
            x["location_id"] = x.PWSID + "|" + x.FacilityID + "|" + x.SamplePointID
            x = x.loc[x.location_id.isin(candidates)]
            parts.append(x)
    assert count == raw_count
    x = pd.concat(parts, ignore_index=True)
    x["dt"] = pd.to_datetime(x.CollectionDate, format="%m/%d/%Y", errors="coerce")
    start = pd.Series({k: b["date_se1"] for k, b in candidates.items()})
    x["start"] = pd.to_datetime(x.location_id.map(start))
    later = x.loc[x.dt > x.start].copy()
    first = later.groupby("location_id").dt.min()
    expected = pd.Series({k: pd.Timestamp(v) for k, v in first_dates.items()})
    pd.testing.assert_series_equal(first.sort_index(), expected.sort_index(), check_names=False)
    chosen = later.loc[later.dt.eq(later.location_id.map(first)) & later.location_id.isin(d.location_id)].copy()
    assert len(chosen) == 25 * len(d)
    assert chosen.groupby("location_id").size().eq(25).all()
    for col in ["SampleID", "SampleEventCode", "CollectionDate"]:
        assert chosen.groupby("location_id")[col].nunique().eq(1).all()
    assert chosen.MethodID.eq("EPA 533").all() and chosen.Units.eq("µg/L").all()
    assert not chosen.duplicated(["location_id", "Contaminant"]).any()
    numeric_mrl = pd.to_numeric(chosen.MRL)
    assert np.allclose(numeric_mrl, chosen.Contaminant.map(MRL), atol=1e-12, rtol=0)
    assert chosen.AnalyticalResultsSign.isin(["=", "<"]).all()
    chosen["report"] = chosen.AnalyticalResultsSign.eq("=").astype(int)
    flags = chosen.pivot(index="location_id", columns="Contaminant", values="report").loc[d.location_id, list(A25)].to_numpy(int)
    assert np.array_equal(flags, d[["future_report__" + a for a in A25]].to_numpy(int))
    k25 = flags.sum(axis=1)
    k6 = flags[:, [A25.index(a) for a in SIX]].sum(axis=1)
    assert np.array_equal(k25, d.future_k25.to_numpy())
    assert np.array_equal(k6, d.future_k6.to_numpy())
    assert np.array_equal((k25 >= 2) & (k6 <= 1), d.target_expansion_multi.to_numpy(bool))
    savejson("independent_replay_audit.json", {
        "status": "PASS", "parser": "pandas chunk parser independent of csv.DictReader preparation",
        "raw_rows": count, "earliest_dates_recomputed_for_locations": len(first),
        "paired_raw_rows": len(chosen), "paired_locations": len(d),
        "all_25_flags_exact": True, "six_and_25_counts_exact": True,
        "expansion_target_exact": True, "keys_dates_sampleids_eventcodes_units_methods_mrls": "PASS",
    })


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source_hashes = {str(p.relative_to(ROOT)): sha(p) for p in [ARCHIVE, CACHE, REFERENCE, PROTOCOL]}
    fixture_result = fixture_checks()
    cache = pd.read_csv(CACHE, dtype=str, keep_default_na=False)
    assert len(cache) == 26461 and cache.location_id.is_unique
    assert cache.groupby("pws_id").state_fold.nunique().max() == 1
    byid = cache.set_index("location_id").to_dict("index")
    baseline_raw = defaultdict(list)
    counter = Counter()
    print("Reading and independently validating all strict SE1 six-analyte panels", flush=True)
    for r in raw_rows():
        counter["raw_rows_pass1"] += 1
        if r["SampleEventCode"] != "SE1" or r["Contaminant"] not in SIX:
            continue
        key = location(r)
        if key in byid:
            baseline_raw[key].append(r)
    assert set(baseline_raw) == set(byid)
    baseline = {}
    errors = []
    max_abs_conc_error_ugL = 0.0
    max_abs_cached_log1p_error = 0.0
    for key, old in byid.items():
        rows = baseline_raw[key]
        reasons = panel_reasons(rows, SIX)
        if reasons:
            errors.append({"location_id": key, "reasons": ";".join(reasons)})
            continue
        first = rows[0]
        dt = parsedate(first["CollectionDate"])
        b = {"location_id": key, "date_se1": dt.isoformat(),
             "sample_id_se1": first["SampleID"].strip(),
             "state_fold": int(old["state_fold"])}
        for rawcol, newcol in META_MAP.items():
            assert first[rawcol].strip() == old[newcol], (key, newcol)
            b[newcol] = old[newcol]
        b["month_sin"] = float(old["month_sin"])
        b["month_cos"] = float(old["month_cos"])
        assert abs(b["month_sin"] - math.sin(2 * math.pi * dt.month / 12)) < 1e-12
        assert abs(b["month_cos"] - math.cos(2 * math.pi * dt.month / 12)) < 1e-12
        native_count = 0
        for r in rows:
            a = r["Contaminant"]
            reported = int(r["AnalyticalResultsSign"].strip() == "=")
            mrl = float(r["MRL"])
            c = float(r["AnalyticalResultValue"]) if reported else 0.0
            assert reported == int(old["y_native_detect__" + a]), (key, a, "report_flag")
            cached_log1p = float(old["y_native_logratio__" + a])
            restored = math.exp(float(old["native_logmrl__" + a])) * math.expm1(cached_log1p)
            discrepancy = abs(restored - c)
            max_abs_conc_error_ugL = max(max_abs_conc_error_ugL, discrepancy)
            max_abs_cached_log1p_error = max(max_abs_cached_log1p_error, abs(cached_log1p - math.log1p(c / mrl)))
            assert discrepancy <= 1e-6, (key, a, "concentration", discrepancy)
            assert abs(cached_log1p - math.log1p(c / mrl)) <= 1e-6, (key, a, "cached_log1p")
            assert abs(math.exp(float(old["native_logmrl__" + a])) - mrl) <= 1e-12
            b["baseline_report__" + a] = reported
            b["baseline_logratio__" + a] = math.log(c / mrl) if reported else 0.0
            b["baseline_conc_ngL__" + a] = c * 1000
            b["baseline_mrl_ngL__" + a] = mrl * 1000
            native_count += reported
        assert native_count == int(old["n_native_detect"]), key
        b["baseline_k6"] = native_count
        b["baseline_pattern6"] = sum(b["baseline_report__" + a] << j for j, a in enumerate(SIX))
        baseline[key] = b
    if errors:
        pd.DataFrame(errors).to_csv(OUT / "BASELINE_ERRORS.csv", index=False)
        raise RuntimeError(f"Baseline raw-panel discrepancies: {len(errors)}; stopped before follow-up construction")
    assert len(baseline) == len(cache)
    candidates = {k: b for k, b in baseline.items() if b["baseline_k6"] <= 1}
    assert len(candidates) == 24456
    dates1 = {k: datetime.fromisoformat(b["date_se1"]).date() for k, b in candidates.items()}
    first_dates, first_rows = {}, defaultdict(list)
    bad_dates = Counter()
    seen_later_codes = Counter()
    print(f"Validated {len(baseline):,} baselines; collecting first later date for {len(candidates):,} candidates", flush=True)
    for r in raw_rows():
        counter["raw_rows_pass2"] += 1
        if r["SampleEventCode"] not in {"SE2", "SE3", "SE4"} or r["Contaminant"] not in MRL:
            continue
        key = location(r)
        if key not in candidates:
            continue
        seen_later_codes[key] += 1
        try:
            dt = parsedate(r["CollectionDate"])
        except ValueError:
            bad_dates[key] += 1
            continue
        if dt <= dates1[key]:
            counter["SE2_to_SE4_rows_not_strictly_later"] += 1
            continue
        keep_first_date(first_dates, first_rows, key, dt, r)
    reference = pd.read_csv(REFERENCE, dtype=str, keep_default_na=False).set_index("location_id")
    assert reference.index.is_unique
    paired, availability = [], []
    unique_mrls = defaultdict(set)
    for key, b in candidates.items():
        a = dict(b)
        a["has_later_se2_to_se4_rows"] = int(key in seen_later_codes)
        a["has_strictly_later_date"] = int(key in first_dates)
        a["invalid_followup_date_rows"] = int(bad_dates[key])
        reasons = []
        if bad_dates[key]:
            reasons.append("unparseable_followup_date")
        if key not in first_dates:
            reasons.append("no_strictly_later_date")
        else:
            a["date_first_followup"] = first_dates[key].isoformat()
            a["first_followup_rows"] = len(first_rows[key])
            reasons += panel_reasons(first_rows[key], A25)
        a["eligible_followup"] = int(not reasons)
        a["exclusion_primary"] = reasons[0] if reasons else "included"
        a["exclusion_all"] = ";".join(dict.fromkeys(reasons))
        availability.append(a)
        if reasons:
            continue
        rows = first_rows[key]
        row = dict(b)
        row["date_followup"] = first_dates[key].isoformat()
        row["lag_days"] = (first_dates[key] - dates1[key]).days
        row["sample_id_followup"] = rows[0]["SampleID"].strip()
        row["event_code_followup"] = rows[0]["SampleEventCode"].strip()
        for r in rows:
            a = r["Contaminant"]
            row["future_report__" + a] = int(r["AnalyticalResultsSign"].strip() == "=")
            unique_mrls[a].add(float(r["MRL"]))
        row["future_k6"] = sum(row["future_report__" + a] for a in SIX)
        row["future_k25"] = sum(row["future_report__" + a] for a in A25)
        row["target_all_multi"] = int(row["future_k25"] >= 2)
        row["target_expansion_multi"] = int(row["future_k25"] >= 2 and row["future_k6"] <= 1)
        row["reference_se1_k25"] = np.nan
        row["reference_se1_complete25"] = int(key in reference.index)
        for a in A25:
            row["reference_se1_report__" + a] = np.nan
        if key in reference.index:
            ref = reference.loc[key]
            assert parsedate(ref["CollectionDate"]) == dates1[key]
            assert ref["SampleID"] == b["sample_id_se1"]
            assert int(ref["k6_native"]) == b["baseline_k6"]
            row["reference_se1_k25"] = int(ref["k25_native"])
            for a in A25:
                row["reference_se1_report__" + a] = int(ref["report__" + a])
        paired.append(row)
    d = pd.DataFrame(paired).sort_values("location_id").reset_index(drop=True)
    av = pd.DataFrame(availability).sort_values("location_id").reset_index(drop=True)
    assert d.location_id.is_unique and d.lag_days.gt(0).all()
    assert d.groupby("pws_id").state_fold.nunique().max() == 1
    assert d.groupby("state").state_fold.nunique().max() == 1
    assert (d.future_k25 >= d.future_k6).all()
    assert (d.target_expansion_multi <= d.target_all_multi).all()
    conc = d[["baseline_conc_ngL__" + a for a in SIX]].to_numpy(float)
    assert np.array_equal(conc.sum(axis=1), conc.max(axis=1))
    print("Independently replaying all earliest dates and follow-up labels with pandas parser", flush=True)
    independent_followup_replay(d, candidates, first_dates, counter["raw_rows_pass2"])
    d.to_csv(OUT / "paired_analysis.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    av.to_csv(OUT / "baseline_followup_availability.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    counts = av.exclusion_primary.value_counts().sort_index()
    pd.DataFrame([{"stage": "strict_six_SE1", "locations": len(baseline)},
                  {"stage": "exclude_SE1_six_count_at_least_two", "locations": len(baseline)-len(candidates)},
                  {"stage": "SE1_candidate_zero_or_one", "locations": len(candidates)}] +
                 [{"stage": str(k), "locations": int(v)} for k, v in counts.items()]).to_csv(OUT / "exclusion_flow.csv", index=False)
    all_reason_counts = Counter(x for v in av.exclusion_all for x in v.split(";") if x)
    pd.DataFrame([{"reason": k, "locations": v} for k, v in sorted(all_reason_counts.items())]).to_csv(OUT / "exclusion_reasons_nonexclusive.csv", index=False)
    availability_rows = []
    for columns in [["water_type"], ["baseline_k6"], ["water_type", "baseline_k6"], ["pws_size"], ["state"], ["state_fold"]]:
        for key, g in av.groupby(columns, dropna=False):
            key = key if isinstance(key, tuple) else (key,)
            availability_rows.append({"grouping": "+".join(columns), "group": "|".join(map(str, key)),
                                      "candidate_locations": len(g), "eligible_followup_locations": int(g.eligible_followup.sum()),
                                      "eligible_followup_fraction": float(g.eligible_followup.mean())})
    pd.DataFrame(availability_rows).to_csv(OUT / "followup_availability_by_baseline.csv", index=False)
    support = []
    for cols in [[], ["baseline_k6"], ["water_type"], ["state_fold"], ["event_code_followup"]]:
        groups = [("all", d)] if not cols else d.groupby(cols, dropna=False)
        for key, g in groups:
            key = key if isinstance(key, tuple) else (key,)
            support.append({"grouping": "+".join(cols) or "all", "group": "|".join(map(str, key)),
                            "locations": len(g), "pws": g.pws_id.nunique(), "states": g.state.nunique(),
                            "all_multi": int(g.target_all_multi.sum()), "expansion_multi": int(g.target_expansion_multi.sum()),
                            "lag_min": int(g.lag_days.min()), "lag_median": float(g.lag_days.median()), "lag_max": int(g.lag_days.max())})
    pd.DataFrame(support).to_csv(OUT / "outcome_support.csv", index=False)
    pd.DataFrame({"analyte": A25, "native_MRL_ug_L": [MRL[a] for a in A25]}).to_csv(OUT / "fixed_method533_analytes.csv", index=False)
    allowed = ["region", "pws_size", "water_type", "month_sin", "month_cos"] + ["baseline_report__"+a for a in SIX] + ["baseline_logratio__"+a for a in SIX]
    savejson("column_dictionary.json", {
        "six_analyte_order": list(SIX), "fixed_25_analytes": list(A25),
        "allowed_model_features": allowed,
        "rule_only_features": ["baseline_conc_ngL__"+a for a in SIX] + ["baseline_pattern6", "baseline_k6"],
        "baseline_logratio_definition": "natural log(C/MRL) only for reported measurements; zero feature code otherwise; report flag always accompanies it",
        "old_cache_logratio_definition": "y_native_logratio is log1p(C/MRL), so independent concentration replay uses MRL*expm1(cache value)",
        "future_prefix": "Labels/audits only; prohibited predictors",
        "reference_prefix": "SE1 complete-panel information-rich reference only; prohibited in primary restricted-information models",
        "id_read_policy": "Read location_id, pws_id, state and both sample IDs as strings. Leading zeroes are significant.",
        "date_and_lag_policy": "Dates and lag are audit/sensitivity fields, not primary model predictors",
        "all_columns": list(d.columns),
    })
    hashes_after = {str(p.relative_to(ROOT)): sha(p) for p in [ARCHIVE, CACHE, REFERENCE]}
    assert all(source_hashes[k] == v for k, v in hashes_after.items())
    audit = {
        "status": "PASS", "analysis_type": "cohort preparation only; no fitting/scoring",
        "raw_rows": dict(counter), "strict_baselines_validated": len(baseline),
        "candidate_locations": len(candidates), "paired_locations": len(d),
        "paired_pws": int(d.pws_id.nunique()), "paired_jurisdictions": int(d.state.nunique()),
        "target_all_multi": int(d.target_all_multi.sum()), "target_expansion_multi": int(d.target_expansion_multi.sum()),
        "reference_complete25": int(d.reference_se1_complete25.sum()),
        "max_abs_concentration_cache_replay_error_ugL": max_abs_conc_error_ugL,
        "max_abs_cached_log1p_error": max_abs_cached_log1p_error,
        "baseline_comparison_tolerance": 1e-6, "duplicate_or_invalid_baseline_panels": 0,
        "PWS_cross_fold": 0, "jurisdiction_cross_fold": 0, "nonpositive_pair_lags": 0,
        "max_equals_sum_on_candidate_reported_concentrations": True,
        "fixtures": fixture_result,
        "earliest_date_rule": "First strictly later date among all SE2-SE4 rows whose analyte belongs to the fixed Method533 panel; selected before checking completeness or report status",
        "invalid_date_policy": "Any unparseable SE2-SE4 date for a candidate excludes it because earliest date is unidentified",
        "future_MRL_values": {a: sorted(v) for a, v in unique_mrls.items()},
        "source_sha256_at_start": source_hashes, "source_data_sha256_unchanged": True,
        "script_sha256": sha(Path(__file__)), "protocol_sha256_at_finish": sha(PROTOCOL),
        "primary_label_from_parent_lock": "target_expansion_multi", "secondary_label": "target_all_multi",
    }
    savejson("cohort_audit.json", audit)
    savejson("output_manifest.json", {p.name: sha(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name != "output_manifest.json"})
    print(json.dumps({k: audit[k] for k in ["status", "candidate_locations", "paired_locations", "paired_pws", "paired_jurisdictions", "target_all_multi", "target_expansion_multi", "reference_complete25"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()

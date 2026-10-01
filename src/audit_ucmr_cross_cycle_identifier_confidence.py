#!/usr/bin/env python3
"""Major-revision audit of cross-cycle identifier-matched UCMR entry points."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import re
import unicodedata
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


CHEMICALS = ("PFBS", "PFHpA", "PFHxS", "PFNA", "PFOA", "PFOS")
BOOTSTRAP_REPS = 20_000
SEED = 20260813


def month_from_date(value: str) -> int:
    """Parse the month without assuming that both UCMR cycles use one date format."""
    text = (value or "").strip()
    if re.fullmatch(r"\d{1,2}/\d{1,2}/\d{4}", text):
        return int(text.split("/")[0])
    if re.fullmatch(r"\d{4}-\d{1,2}-\d{1,2}", text):
        return int(text.split("-")[1])
    raise ValueError(f"Unsupported collection date: {value!r}")


def season(month: int) -> str:
    if month in {12, 1, 2}:
        return "DJF"
    if month in {3, 4, 5}:
        return "MAM"
    if month in {6, 7, 8}:
        return "JJA"
    return "SON"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_name(value: str) -> str:
    text = unicodedata.normalize("NFKD", value or "").encode("ascii", "ignore").decode()
    text = re.sub(r"\b(THE|INC|LLC|LTD|CORP|CORPORATION|COMPANY|CO)\b", "", text.upper())
    return re.sub(r"[^A-Z0-9]+", "", text)


def read_panels(path: Path, cycle: str) -> tuple[dict, dict]:
    expected = f"{cycle}_All.txt"
    with zipfile.ZipFile(path) as archive:
        members = [name for name in archive.namelist() if name.endswith(expected)]
        if len(members) != 1:
            raise ValueError(f"Expected one {expected}, found {members}")
        raw = defaultdict(lambda: defaultdict(list))
        audit = Counter()
        with archive.open(members[0]) as binary:
            reader = csv.DictReader(
                (line.decode("latin-1") for line in binary), delimiter="\t"
            )
            for row in reader:
                audit["raw_rows"] += 1
                if row["SampleEventCode"] != "SE1" or row["Contaminant"] not in CHEMICALS:
                    continue
                audit["shared_pfas_se1_rows"] += 1
                key = (row["PWSID"], row["FacilityID"], row["SamplePointID"])
                detected = row["AnalyticalResultsSign"].strip() == "="
                value_text = row["AnalyticalResultValue"].strip()
                if detected and not value_text:
                    audit["detected_missing_value_rows"] += 1
                    continue
                raw[key][row["Contaminant"]].append(
                    {
                        "detected": detected,
                        "value": float(value_text) if value_text else None,
                        "mrl": float(row["MRL"]),
                        "state": row["State"],
                        "size": row["Size"],
                        "water_type": row["FacilityWaterType"],
                        "sample_point_type": row["SamplePointType"],
                        "date": row["CollectionDate"],
                        "pws_name": row["PWSName"],
                        "facility_name": row["FacilityName"],
                        "sample_point_name": row["SamplePointName"],
                        "method_id": row["MethodID"],
                    }
                )
    complete = {}
    for key, panel in raw.items():
        if set(panel) != set(CHEMICALS):
            audit["incomplete_panel_sites"] += 1
            continue
        if any(len(panel[chemical]) != 1 for chemical in CHEMICALS):
            audit["ambiguous_duplicate_sites"] += 1
            continue
        complete[key] = {chemical: panel[chemical][0] for chemical in CHEMICALS}
    audit["complete_unambiguous_se1_sites"] = len(complete)
    audit["pws_with_complete_se1"] = len({key[0] for key in complete})
    return complete, dict(audit)


def cluster_bootstrap(values: np.ndarray, cluster: np.ndarray, seed: int) -> dict:
    names = sorted(set(cluster.astype(str)))
    grouped = [values[cluster.astype(str) == name] for name in names]
    totals = np.asarray([part.sum() for part in grouped])
    counts = np.asarray([len(part) for part in grouped])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(names), size=(BOOTSTRAP_REPS, len(names)))
    estimates = totals[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return {
        "estimate": float(values.mean()),
        "ci95_low": float(np.quantile(estimates, 0.025)),
        "ci95_high": float(np.quantile(estimates, 0.975)),
        "clusters": len(names),
        "observations": len(values),
        "reps": BOOTSTRAP_REPS,
    }


def pws_equal_bootstrap(values: np.ndarray, pws: np.ndarray, seed: int) -> dict:
    names = sorted(set(pws.astype(str)))
    means = np.asarray([values[pws.astype(str) == name].mean() for name in names])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(names), size=(BOOTSTRAP_REPS, len(names)))
    estimates = means[draws].mean(axis=1)
    return {
        "estimate": float(means.mean()),
        "ci95_low": float(np.quantile(estimates, 0.025)),
        "ci95_high": float(np.quantile(estimates, 0.975)),
        "clusters": len(names),
        "observations": len(values),
        "reps": BOOTSTRAP_REPS,
    }


def state_pws_bootstrap(
    values: np.ndarray, state: np.ndarray, pws: np.ndarray, seed: int
) -> dict:
    state_names = sorted(set(state.astype(str)))
    by_state = {}
    for state_name in state_names:
        mask = state.astype(str) == state_name
        pws_names = sorted(set(pws[mask].astype(str)))
        by_state[state_name] = np.asarray(
            [values[mask & (pws.astype(str) == name)].mean() for name in pws_names]
        )
    rng = np.random.default_rng(seed)
    # For each state, generate all within-state PWS bootstrap means at once.
    # Then resample state columns per replicate. This is algebraically the same
    # two-level bootstrap as the scalar implementation but avoids millions of
    # Python-level random-choice calls.
    state_bootstrap = np.empty((BOOTSTRAP_REPS, len(state_names)), dtype=float)
    for column, state_name in enumerate(state_names):
        pws_values = by_state[state_name]
        draws = rng.integers(
            0, len(pws_values), size=(BOOTSTRAP_REPS, len(pws_values))
        )
        state_bootstrap[:, column] = pws_values[draws].mean(axis=1)
    state_draws = rng.integers(
        0, len(state_names), size=(BOOTSTRAP_REPS, len(state_names))
    )
    estimates = state_bootstrap[
        np.arange(BOOTSTRAP_REPS)[:, None], state_draws
    ].mean(axis=1)
    # The point estimate must use exactly the same state-equal/PWS-equal estimand
    # as the two-level bootstrap.  An earlier implementation accidentally used
    # event-equal state means here while bootstrapping PWS-equal state means.
    observed_state_means = [by_state[name].mean() for name in state_names]
    return {
        "estimate": float(np.mean(observed_state_means)),
        "ci95_low": float(np.quantile(estimates, 0.025)),
        "ci95_high": float(np.quantile(estimates, 0.975)),
        "states": len(state_names),
        "pws": int(len(set(pws.astype(str)))),
        "observations": len(values),
        "reps": BOOTSTRAP_REPS,
        "estimand": "state-equal mean of within-state PWS-equal means",
    }


def paired_table(before: np.ndarray, after: np.ndarray) -> dict:
    before = before.astype(bool)
    after = after.astype(bool)
    only_before = int(np.sum(before & ~after))
    only_after = int(np.sum(~before & after))
    discordant = only_before + only_after
    smaller = min(only_before, only_after)
    exact_p = min(
        1.0,
        2.0
        * sum(math.comb(discordant, k) for k in range(smaller + 1))
        / (2**discordant),
    ) if discordant else 1.0
    return {
        "only_ucmr3": only_before,
        "only_ucmr5": only_after,
        "both": int(np.sum(before & after)),
        "neither": int(np.sum(~before & ~after)),
        "discordant_pairs": discordant,
        "exact_mcnemar_two_sided_p": float(exact_p),
    }


def shapley_mrl_reclassification(y5: np.ndarray, y5_old: np.ndarray) -> dict:
    """Exact six-analyte Shapley decomposition over all 6! threshold orders."""
    values = {}
    for mask in range(1 << len(CHEMICALS)):
        use_old = np.asarray([(mask >> j) & 1 for j in range(len(CHEMICALS))], dtype=bool)
        switched = np.where(use_old[None, :], y5_old, y5)
        values[mask] = float(np.mean(switched.sum(axis=1) >= 2))
    contributions = np.zeros(len(CHEMICALS), dtype=float)
    orders = list(itertools.permutations(range(len(CHEMICALS))))
    for order in orders:
        mask = 0
        for analyte in order:
            updated = mask | (1 << analyte)
            contributions[analyte] += values[mask] - values[updated]
            mask = updated
    contributions /= len(orders)
    total = values[0] - values[(1 << len(CHEMICALS)) - 1]
    return {
        "orders": len(orders),
        "endpoint": "change in Pr(K6 >= 2) when UCMR5 native MRLs are replaced by UCMR3 MRLs",
        "contribution_pp": {
            chemical: float(100 * value)
            for chemical, value in zip(CHEMICALS, contributions)
        },
        "total_reclassification_pp": float(100 * total),
        "sum_contributions_pp": float(100 * contributions.sum()),
        "additivity_residual_pp": float(100 * (total - contributions.sum())),
    }


def cycle_cohort_summary(panels: dict, matched_keys: set, cycle: str) -> list[dict]:
    """Describe exact-identifier matched and unmatched complete SE1 cohorts."""
    rows = []
    for cohort, want_matched in (("identifier_matched", True), ("unmatched", False)):
        selected = [(key, panel) for key, panel in panels.items() if (key in matched_keys) == want_matched]
        if not selected:
            continue
        first = [panel[CHEMICALS[0]] for _, panel in selected]
        k6 = np.asarray(
            [sum(panel[chemical]["detected"] for chemical in CHEMICALS) for _, panel in selected]
        )
        row = {
            "cycle": cycle,
            "cohort": cohort,
            "entry_point_events": len(selected),
            "pws": len({key[0] for key, _ in selected}),
            "states": len({item["state"] for item in first}),
            "native_k2_rate": float(np.mean(k6 >= 2)),
            "state_counts": dict(sorted(Counter(item["state"] for item in first).items())),
            "size_counts": dict(sorted(Counter(item["size"] for item in first).items())),
            "water_type_counts": dict(
                sorted(Counter(item["water_type"] for item in first).items())
            ),
            "analyte_native_reporting_rates": {
                chemical: float(
                    np.mean([panel[chemical]["detected"] for _, panel in selected])
                )
                for chemical in CHEMICALS
            },
        }
        rows.append(row)
    return rows


def analyze_subset(records: list[dict], subset: str, seed: int) -> dict:
    selected = [record for record in records if record[subset]]
    if not selected:
        return {"entry_point_pairs": 0}
    state = np.asarray([record["state"] for record in selected], dtype=str)
    pws = np.asarray([record["pws"] for record in selected], dtype=str)
    y3 = np.stack([record["y3"] for record in selected])
    y5 = np.stack([record["y5"] for record in selected])
    y5_old = np.stack([record["y5_old"] for record in selected])
    k3 = y3.sum(axis=1)
    k5 = y5.sum(axis=1)
    k5_old = y5_old.sum(axis=1)
    native = (k5 >= 2).astype(float) - (k3 >= 2).astype(float)
    standardized = (k5_old >= 2).astype(float) - (k3 >= 2).astype(float)
    reclassification = (k5 >= 2).astype(float) - (k5_old >= 2).astype(float)
    endpoint = {}
    for name, values in (
        ("native_period_contrast", native),
        ("common_ucmr3_mrl_period_contrast", standardized),
        ("within_ucmr5_mrl_reclassification", reclassification),
    ):
        endpoint[name] = {
            "site_equal_state_cluster": cluster_bootstrap(values, state, seed),
            "pws_equal": pws_equal_bootstrap(values, pws, seed + 1000),
            "state_pws_two_level": state_pws_bootstrap(values, state, pws, seed + 2000),
        }
        seed += 17
    denominator = k5 >= 2
    numerator = denominator & (k5_old < 2)
    return {
        "entry_point_pairs": len(selected),
        "pws": int(len(set(pws))),
        "states": int(len(set(state))),
        "ucmr3_native_k2_rate": float(np.mean(k3 >= 2)),
        "ucmr5_native_k2_rate": float(np.mean(k5 >= 2)),
        "ucmr5_at_ucmr3_mrl_k2_rate": float(np.mean(k5_old >= 2)),
        "ucmr5_native_k2_observations": int(denominator.sum()),
        "ucmr5_native_k2_reclassified_below_ucmr3_mrl": int(numerator.sum()),
        "fraction_ucmr5_native_k2_reclassified": float(numerator.sum() / denominator.sum())
        if denominator.any()
        else None,
        "decomposition_pp": {
            "native_period_contrast": float(100 * native.mean()),
            "within_ucmr5_mrl_reclassification": float(100 * reclassification.mean()),
            "common_ucmr3_mrl_period_contrast": float(100 * standardized.mean()),
            "identity_residual": float(
                100 * (native.mean() - reclassification.mean() - standardized.mean())
            ),
        },
        "paired_tables": {
            "native_mrl": paired_table(k3 >= 2, k5 >= 2),
            "common_ucmr3_mrl": paired_table(k3 >= 2, k5_old >= 2),
        },
        "exact_six_analyte_shapley": shapley_mrl_reclassification(y5, y5_old),
        "inference": endpoint,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ucmr3-zip", type=Path, required=True)
    parser.add_argument("--ucmr5-zip", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    u3, audit3 = read_panels(args.ucmr3_zip, "UCMR3")
    u5, audit5 = read_panels(args.ucmr5_zip, "UCMR5")
    keys = sorted(set(u3) & set(u5))
    records = []
    mismatch = Counter()
    method_pairs = Counter()
    water_pairs = Counter()
    for key in keys:
        a = u3[key][CHEMICALS[0]]
        b = u5[key][CHEMICALS[0]]
        stable_fields = {
            "state": a["state"] == b["state"],
            "water_type": a["water_type"] == b["water_type"],
            "sample_point_type": a["sample_point_type"] == b["sample_point_type"],
            "size": a["size"] == b["size"],
        }
        for field, same in stable_fields.items():
            mismatch[f"{field}_mismatch"] += int(not same)
        name_fields = {
            field: normalize_name(a[field]) == normalize_name(b[field])
            and bool(normalize_name(a[field]))
            for field in ("pws_name", "facility_name", "sample_point_name")
        }
        for field, same in name_fields.items():
            mismatch[f"{field}_mismatch"] += int(not same)
        y3 = np.asarray([u3[key][c]["detected"] for c in CHEMICALS], dtype=np.int8)
        y5 = np.asarray([u5[key][c]["detected"] for c in CHEMICALS], dtype=np.int8)
        y5_old = np.asarray(
            [
                u5[key][c]["detected"]
                and u5[key][c]["value"] is not None
                and u5[key][c]["value"] >= u3[key][c]["mrl"]
                for c in CHEMICALS
            ],
            dtype=np.int8,
        )
        metadata_stable = all(stable_fields.values())
        name_corroborated = metadata_stable and all(name_fields.values())
        water = b["water_type"].upper()
        month3 = month_from_date(a["date"])
        month5 = month_from_date(b["date"])
        cyclic_month_distance = min(abs(month5 - month3), 12 - abs(month5 - month3))
        records.append(
            {
                "pws": key[0],
                "state": b["state"],
                "y3": y3,
                "y5": y5,
                "y5_old": y5_old,
                "identifier_matched": True,
                "metadata_stable": metadata_stable,
                "name_corroborated": name_corroborated,
                "stable_groundwater": metadata_stable and water in {"GW", "GU"},
                "stable_surface_water": metadata_stable and water in {"SW", "SU"},
                "same_calendar_month": month3 == month5,
                "same_meteorological_season": season(month3) == season(month5),
                "cyclic_month_distance": cyclic_month_distance,
            }
        )
        water_pairs[(a["water_type"], b["water_type"])] += 1
        for chemical in CHEMICALS:
            method_pairs[(u3[key][chemical]["method_id"], u5[key][chemical]["method_id"])] += 1
    subsets = (
        "identifier_matched",
        "metadata_stable",
        "name_corroborated",
        "stable_groundwater",
        "stable_surface_water",
        "same_calendar_month",
        "same_meteorological_season",
    )
    analyses = {
        subset: analyze_subset(records, subset, SEED + index * 10_000)
        for index, subset in enumerate(subsets)
    }
    matched_key_set = set(keys)
    cohort_comparison = cycle_cohort_summary(u3, matched_key_set, "UCMR3")
    cohort_comparison.extend(cycle_cohort_summary(u5, matched_key_set, "UCMR5"))
    month_distance = np.asarray([record["cyclic_month_distance"] for record in records])
    payload = {
        "status": "IDENTIFIER-PAIR AND ESTIMAND AUDIT COMPLETE",
        "primary_unit": (
            "Identifier-matched pair of SE1 entry-point events: exact PWSID + FacilityID + "
            "SamplePointID with one complete, unambiguous six-PFAS SE1 event in each cycle. "
            "The 7,620 unit count denotes pairs, not individual events."
        ),
        "cooccurrence_definition": (
            "K6 is the number of the six shared PFAS reported at or above the specified "
            "analyte-specific MRL within the same SE1 sampling event; it is not annual-ever-detected."
        ),
        "source": {
            "ucmr3": str(args.ucmr3_zip),
            "ucmr3_sha256": sha256(args.ucmr3_zip),
            "ucmr5": str(args.ucmr5_zip),
            "ucmr5_sha256": sha256(args.ucmr5_zip),
        },
        "cycle_audits": {"UCMR3": audit3, "UCMR5": audit5},
        "identifier_matched_entry_point_pairs": len(keys),
        "events_represented_by_pairs": 2 * len(keys),
        "collection_timing": {
            "same_calendar_month_pairs": int(np.sum(month_distance == 0)),
            "same_calendar_month_fraction": float(np.mean(month_distance == 0)),
            "same_meteorological_season_pairs": int(
                sum(record["same_meteorological_season"] for record in records)
            ),
            "same_meteorological_season_fraction": float(
                np.mean([record["same_meteorological_season"] for record in records])
            ),
            "cyclic_month_distance_median": float(np.median(month_distance)),
            "cyclic_month_distance_counts": {
                str(int(value)): int(np.sum(month_distance == value))
                for value in sorted(set(month_distance))
            },
        },
        "matched_vs_unmatched_cohort_comparison": cohort_comparison,
        "mismatches": dict(mismatch),
        "water_type_pairs": [
            {"ucmr3": a, "ucmr5": b, "sites": count}
            for (a, b), count in sorted(water_pairs.items())
        ],
        "method_pairs": [
            {"ucmr3": a, "ucmr5": b, "analyte_site_results": count}
            for (a, b), count in sorted(method_pairs.items())
        ],
        "subsets": analyses,
        "bootstrap": {"reps": BOOTSTRAP_REPS, "seed": SEED},
        "unresolved_physical_identity": [
            "No sampling-point coordinates in the occurrence files",
            "No hydraulic connectivity or source-switch history",
            "No treatment-upgrade history",
            "Names and identifiers may be administratively revised",
        ],
        "interpretation_boundary": (
            "Metadata and exact normalized-name agreement corroborate administrative identity "
            "but do not prove an unchanged physical or hydraulic sampling location."
        ),
        "protocol": "outputs/CENMIXQ_MAJOR_REVISION_FROZEN_PROTOCOL_20260813_ZH.md",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "decision.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    compact = {
        "status": payload["status"],
        "identifier_matched_entry_point_pairs": len(keys),
        "collection_timing": payload["collection_timing"],
        "mismatches": payload["mismatches"],
        "subsets": {
            key: {
                field: value
                for field, value in result.items()
                if field not in {"inference"}
            }
            for key, result in analyses.items()
        },
    }
    print(json.dumps(compact, indent=2), flush=True)


if __name__ == "__main__":
    main()

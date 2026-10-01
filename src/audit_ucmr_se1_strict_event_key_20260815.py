#!/usr/bin/env python3
"""Audit SE1 panels under a strict date- and sample-specific event key.

The legacy cohort groups the six shared PFAS by PWSID, FacilityID, and
SamplePointID after filtering SampleEventCode == SE1. This audit distinguishes
an administrative location from a laboratory sample event. A strict event is
defined by:

    PWSID + FacilityID + SamplePointID + CollectionDate + SampleID

A location is retained only if all six shared PFAS occur exactly once in one
strict event and no additional shared-PFAS SE1 rows occur at that location.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path


PFAS = ("PFBS", "PFHpA", "PFHxS", "PFNA", "PFOA", "PFOS")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def archive_member(path: Path, cycle: str) -> str:
    target = f"{cycle}_All.txt"
    with zipfile.ZipFile(path) as archive:
        matches = [name for name in archive.namelist() if name.endswith(target)]
    if len(matches) != 1:
        raise ValueError(f"Expected one {target}, found {matches}")
    return matches[0]


def read_rows(path: Path, cycle: str) -> tuple[dict, Counter]:
    by_location = defaultdict(list)
    audit = Counter()
    member = archive_member(path, cycle)
    with zipfile.ZipFile(path) as archive, archive.open(member) as binary:
        reader = csv.DictReader(
            (line.decode("latin-1") for line in binary), delimiter="\t"
        )
        for row in reader:
            audit["raw_rows"] += 1
            if row["SampleEventCode"] != "SE1" or row["Contaminant"] not in PFAS:
                continue
            audit["shared_pfas_se1_rows"] += 1
            key = (row["PWSID"], row["FacilityID"], row["SamplePointID"])
            sign = row["AnalyticalResultsSign"].strip()
            value_text = row["AnalyticalResultValue"].strip()
            by_location[key].append(
                {
                    "chemical": row["Contaminant"],
                    "date": row["CollectionDate"].strip(),
                    "sample_id": row["SampleID"].strip(),
                    "method_id": row["MethodID"].strip(),
                    "mrl": float(row["MRL"]),
                    "detected": sign == "=",
                    "value": float(value_text) if value_text else None,
                    "state": row["State"],
                    "size": row["Size"],
                    "water_type": row["FacilityWaterType"],
                    "sample_point_type": row["SamplePointType"],
                    "pws_name": row["PWSName"],
                    "facility_name": row["FacilityName"],
                    "sample_point_name": row["SamplePointName"],
                }
            )
    return dict(by_location), audit


def classify(rows_by_location: dict) -> tuple[dict, dict, list[dict], Counter]:
    legacy = {}
    strict = {}
    anomalies = []
    audit = Counter()
    for location, rows in sorted(rows_by_location.items()):
        by_chemical = defaultdict(list)
        for row in rows:
            by_chemical[row["chemical"]].append(row)
        if set(by_chemical) != set(PFAS):
            audit["legacy_incomplete_locations"] += 1
            continue
        if any(len(by_chemical[chemical]) != 1 for chemical in PFAS):
            audit["legacy_duplicate_ambiguous_locations"] += 1
            continue
        legacy[location] = {chemical: by_chemical[chemical][0] for chemical in PFAS}

        dates = sorted({row["date"] for row in rows})
        sample_ids = sorted({row["sample_id"] for row in rows})
        if len(dates) > 1:
            audit["legacy_panels_multiple_dates"] += 1
        if len(sample_ids) > 1:
            audit["legacy_panels_multiple_sample_ids"] += 1

        event_groups = defaultdict(list)
        for row in rows:
            event_groups[(row["date"], row["sample_id"])].append(row)
        complete_events = []
        for event_key, event_rows in event_groups.items():
            event_chemicals = Counter(row["chemical"] for row in event_rows)
            if set(event_chemicals) == set(PFAS) and all(
                event_chemicals[chemical] == 1 for chemical in PFAS
            ):
                complete_events.append((event_key, event_rows))

        strict_ok = len(rows) == len(PFAS) and len(complete_events) == 1
        if strict_ok:
            event_key, event_rows = complete_events[0]
            strict[location] = {
                row["chemical"]: row for row in event_rows
            }
            audit["strict_complete_unambiguous_locations"] += 1
        else:
            audit["legacy_panels_failing_strict_event_key"] += 1
            first = rows[0]
            anomalies.append(
                {
                    "pws_id": location[0],
                    "facility_id": location[1],
                    "sample_point_id": location[2],
                    "state": first["state"],
                    "size": first["size"],
                    "pws_name": first["pws_name"],
                    "facility_name": first["facility_name"],
                    "sample_point_name": first["sample_point_name"],
                    "dates": "|".join(dates),
                    "sample_ids": "|".join(sample_ids),
                    "n_dates": len(dates),
                    "n_sample_ids": len(sample_ids),
                    "n_event_groups": len(event_groups),
                    "n_complete_strict_events": len(complete_events),
                    "methods": "|".join(sorted({row["method_id"] for row in rows})),
                }
            )
    audit["legacy_complete_unambiguous_locations"] = len(legacy)
    return legacy, strict, anomalies, audit


def read_benchmark(path: Path) -> dict:
    opener = gzip.open if path.suffix == ".gz" else Path.open
    kwargs = {"mode": "rt", "encoding": "utf-8", "newline": ""}
    if path.suffix != ".gz":
        kwargs = {"mode": "r", "encoding": "utf-8", "newline": ""}
    with opener(path, **kwargs) as handle:
        return {row["location_id"]: row for row in csv.DictReader(handle)}


def endpoint_summary(keys: list, panels3: dict, panels5: dict) -> dict:
    counts = Counter()
    for key in keys:
        k3 = sum(int(panels3[key][chemical]["detected"]) for chemical in PFAS)
        k5 = sum(int(panels5[key][chemical]["detected"]) for chemical in PFAS)
        k5_old = sum(
            int(
                panels5[key][chemical]["detected"]
                and panels5[key][chemical]["value"] is not None
                and panels5[key][chemical]["value"] >= panels3[key][chemical]["mrl"]
            )
            for chemical in PFAS
        )
        counts["ucmr3_native_k2"] += int(k3 >= 2)
        counts["ucmr5_native_k2"] += int(k5 >= 2)
        counts["ucmr5_at_ucmr3_cutoff_k2"] += int(k5_old >= 2)
    n = len(keys)
    return {
        "pairs": n,
        **dict(counts),
        "ucmr3_native_rate": counts["ucmr3_native_k2"] / n if n else None,
        "ucmr5_native_rate": counts["ucmr5_native_k2"] / n if n else None,
        "ucmr5_at_ucmr3_cutoff_rate": (
            counts["ucmr5_at_ucmr3_cutoff_k2"] / n if n else None
        ),
        "native_contrast_pp": (
            100
            * (counts["ucmr5_native_k2"] - counts["ucmr3_native_k2"])
            / n
            if n
            else None
        ),
        "common_numeric_cutoff_contrast_pp": (
            100
            * (
                counts["ucmr5_at_ucmr3_cutoff_k2"]
                - counts["ucmr3_native_k2"]
            )
            / n
            if n
            else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ucmr3", type=Path, required=True)
    parser.add_argument("--ucmr5", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    rows3, raw_audit3 = read_rows(args.ucmr3, "UCMR3")
    rows5, raw_audit5 = read_rows(args.ucmr5, "UCMR5")
    legacy3, strict3, anomalies3, panel_audit3 = classify(rows3)
    legacy5, strict5, anomalies5, panel_audit5 = classify(rows5)
    benchmark = read_benchmark(args.benchmark)

    for row in anomalies5:
        location_id = "|".join(
            [row["pws_id"], row["facility_id"], row["sample_point_id"]]
        )
        cohort = benchmark.get(location_id)
        row["state_fold"] = int(cohort["state_fold"]) if cohort else None
        row["formal_fold_1_4"] = bool(
            cohort and int(cohort["state_fold"]) in {1, 2, 3, 4}
        )
        row["legacy_cross_cycle_matched"] = (
            row["pws_id"],
            row["facility_id"],
            row["sample_point_id"],
        ) in legacy3

    legacy_keys = sorted(set(legacy3) & set(legacy5))
    strict_keys = sorted(set(strict3) & set(strict5))
    strict5_location_ids = {"|".join(location) for location in strict5}
    method_pairs = Counter()
    mrl_pairs = {}
    size_transitions = Counter()
    for key in strict_keys:
        size_transitions[(strict3[key][PFAS[0]]["size"], strict5[key][PFAS[0]]["size"])] += 1
        for chemical in PFAS:
            before = strict3[key][chemical]
            after = strict5[key][chemical]
            method_pairs[(before["method_id"], after["method_id"])] += 1
            mrl_pairs[chemical] = [before["mrl"], after["mrl"]]

    payload = {
        "status": "STRICT SE1 EVENT-KEY AUDIT COMPLETE",
        "strict_event_key": [
            "PWSID",
            "FacilityID",
            "SamplePointID",
            "CollectionDate",
            "SampleID",
        ],
        "source_sha256": {
            "ucmr3": sha256(args.ucmr3),
            "ucmr5": sha256(args.ucmr5),
        },
        "UCMR3": {
            "raw": dict(raw_audit3),
            "panels": dict(panel_audit3),
            "anomalies": len(anomalies3),
        },
        "UCMR5": {
            "raw": dict(raw_audit5),
            "panels": dict(panel_audit5),
            "anomalies": len(anomalies5),
            "anomalies_in_formal_folds_1_4": sum(
                row["formal_fold_1_4"] for row in anomalies5
            ),
            "anomalies_in_fold_0": sum(row["state_fold"] == 0 for row in anomalies5),
            "anomalies_in_legacy_cross_cycle_matched": sum(
                row["legacy_cross_cycle_matched"] for row in anomalies5
            ),
        },
        "benchmark_flow": {
            "legacy_locations": len(benchmark),
            "fold_0_development_locations": sum(
                int(row["state_fold"]) == 0 for row in benchmark.values()
            ),
            "formal_fold_1_4_locations": sum(
                int(row["state_fold"]) in {1, 2, 3, 4}
                for row in benchmark.values()
            ),
            "strict_locations_after_exclusion": len(strict5),
            "strict_formal_fold_1_4_locations_after_exclusion": sum(
                int(row["state_fold"]) in {1, 2, 3, 4}
                and location_id in strict5_location_ids
                for location_id, row in benchmark.items()
            ),
        },
        "cross_cycle": {
            "legacy": endpoint_summary(legacy_keys, legacy3, legacy5),
            "strict": endpoint_summary(strict_keys, strict3, strict5),
            "legacy_pairs_removed_by_strict_key": len(legacy_keys) - len(strict_keys),
            "strict_pairs_pws": len({key[0] for key in strict_keys}),
            "strict_pairs_jurisdictions": len(
                {strict5[key][PFAS[0]]["state"] for key in strict_keys}
            ),
            "size_transitions": {
                f"{before}->{after}": count
                for (before, after), count in sorted(size_transitions.items())
            },
            "method_pairs_by_analyte_measurement": {
                f"{before}->{after}": count
                for (before, after), count in sorted(method_pairs.items())
            },
            "mrl_ug_l": {
                chemical: {"UCMR3": values[0], "UCMR5": values[1]}
                for chemical, values in sorted(mrl_pairs.items())
            },
        },
        "interpretation": (
            "Applying UCMR3 numeric cutoffs to UCMR5 EPA 533 measurements "
            "standardizes the cutoff values, not the analytical methods."
        ),
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = list(anomalies5[0]) if anomalies5 else [
        "pws_id",
        "facility_id",
        "sample_point_id",
    ]
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(anomalies5)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

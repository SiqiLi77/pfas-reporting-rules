#!/usr/bin/env python3
"""Audit how UCMR reporting limits alter apparent PFAS changes across cycles.

The audit uses only SE1 observations for the six PFAS shared by UCMR3 and
UCMR5. Exact sampling-point matches are defined by PWSID, FacilityID, and
SamplePointID. A site is retained only when all six analytes have one
unambiguous SE1 result in both cycles.

This script deliberately performs no below-MRL imputation. It separates the
directly identified part of the problem:

* native detection under each cycle's MRL;
* UCMR5 detections that would remain detectable under the UCMR3 MRL;
* UCMR5 detections enabled only by the lower UCMR5 MRL.

Latent concentration recovery below the UCMR3 MRL is a subsequent modeling
task and must be validated by artificial censoring experiments.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path


COMMON_PFAS = ("PFBS", "PFHpA", "PFHxS", "PFNA", "PFOA", "PFOS")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def archive_member(path: Path, cycle: str) -> str:
    expected = f"{cycle}_All.txt"
    with zipfile.ZipFile(path) as archive:
        matches = [name for name in archive.namelist() if name.endswith(expected)]
    if len(matches) != 1:
        raise ValueError(f"Expected one {expected} in {path}, found {matches}")
    return matches[0]


def read_se1(path: Path, cycle: str) -> tuple[dict, Counter]:
    """Return exact site -> analyte -> measurement for complete SE1 panels."""
    member = archive_member(path, cycle)
    raw: dict[tuple[str, str, str], dict[str, list[dict]]] = defaultdict(
        lambda: defaultdict(list)
    )
    audit = Counter()
    with zipfile.ZipFile(path) as archive, archive.open(member) as binary:
        lines = (line.decode("latin-1") for line in binary)
        reader = csv.DictReader(lines, delimiter="\t")
        for row in reader:
            audit["raw_rows"] += 1
            if row["SampleEventCode"] != "SE1" or row["Contaminant"] not in COMMON_PFAS:
                continue
            audit["shared_pfas_se1_rows"] += 1
            key = (row["PWSID"], row["FacilityID"], row["SamplePointID"])
            sign = row["AnalyticalResultsSign"].strip()
            value_text = row["AnalyticalResultValue"].strip()
            mrl_text = row["MRL"].strip()
            if not mrl_text:
                audit["missing_mrl_rows"] += 1
                continue
            detected = sign == "="
            value = float(value_text) if value_text else None
            if detected and value is None:
                audit["detected_missing_value_rows"] += 1
                continue
            raw[key][row["Contaminant"]].append(
                {
                    "detected": detected,
                    "value": value,
                    "mrl": float(mrl_text),
                    "state": row["State"],
                    "region": row["Region"],
                    "size": row["Size"],
                    "water_type": row["FacilityWaterType"],
                    "sample_point_type": row["SamplePointType"],
                    "date": row["CollectionDate"],
                }
            )

    complete = {}
    for key, panel in raw.items():
        if set(panel) != set(COMMON_PFAS):
            audit["incomplete_panel_sites"] += 1
            continue
        if any(len(panel[chemical]) != 1 for chemical in COMMON_PFAS):
            audit["ambiguous_duplicate_sites"] += 1
            continue
        complete[key] = {chemical: panel[chemical][0] for chemical in COMMON_PFAS}
    audit["complete_unambiguous_se1_sites"] = len(complete)
    audit["pws_with_complete_se1"] = len({key[0] for key in complete})
    return complete, audit


def wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = successes / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    margin = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denominator
    return center - margin, center + margin


def analyze(ucmr3: dict, ucmr5: dict) -> tuple[list[dict], dict]:
    matched = sorted(set(ucmr3) & set(ucmr5))
    stable_water_type = [
        key
        for key in matched
        if ucmr3[key][COMMON_PFAS[0]]["water_type"]
        == ucmr5[key][COMMON_PFAS[0]]["water_type"]
    ]
    rows = []
    for chemical in COMMON_PFAS:
        old_mrls = {ucmr3[key][chemical]["mrl"] for key in matched}
        new_mrls = {ucmr5[key][chemical]["mrl"] for key in matched}
        if len(old_mrls) != 1 or len(new_mrls) != 1:
            raise ValueError(
                f"MRL not constant for {chemical}: UCMR3={old_mrls}, UCMR5={new_mrls}"
            )
        old_mrl = next(iter(old_mrls))
        new_mrl = next(iter(new_mrls))
        native3 = native5 = standardized5 = enabled5 = 0
        nd3_to_native5 = nd3_to_enabled5 = persistent_native = 0
        paired_native_increase = paired_native_decrease = 0
        paired_standardized_increase = paired_standardized_decrease = 0
        state_enabled = Counter()
        state_native5 = Counter()
        for key in matched:
            before = ucmr3[key][chemical]
            after = ucmr5[key][chemical]
            detected3 = before["detected"]
            detected5 = after["detected"]
            above_old5 = bool(detected5 and after["value"] >= old_mrl)
            enabled = bool(detected5 and not above_old5)
            native3 += int(detected3)
            native5 += int(detected5)
            standardized5 += int(above_old5)
            enabled5 += int(enabled)
            nd3_to_native5 += int(not detected3 and detected5)
            nd3_to_enabled5 += int(not detected3 and enabled)
            persistent_native += int(detected3 and detected5)
            paired_native_increase += int(not detected3 and detected5)
            paired_native_decrease += int(detected3 and not detected5)
            paired_standardized_increase += int(not detected3 and above_old5)
            paired_standardized_decrease += int(detected3 and not above_old5)
            state_native5[after["state"]] += int(detected5)
            state_enabled[after["state"]] += int(enabled)

        enabled_ci = wilson_interval(enabled5, native5)
        new_enabled_ci = wilson_interval(nd3_to_enabled5, nd3_to_native5)
        stable_native3 = sum(
            int(ucmr3[key][chemical]["detected"]) for key in stable_water_type
        )
        stable_native5 = sum(
            int(ucmr5[key][chemical]["detected"]) for key in stable_water_type
        )
        stable_standardized5 = sum(
            int(
                ucmr5[key][chemical]["detected"]
                and ucmr5[key][chemical]["value"] >= old_mrl
            )
            for key in stable_water_type
        )
        stable_enabled5 = stable_native5 - stable_standardized5
        rows.append(
            {
                "chemical": chemical,
                "matched_sites": len(matched),
                "ucmr3_mrl_ug_l": old_mrl,
                "ucmr5_mrl_ug_l": new_mrl,
                "mrl_fold_reduction": old_mrl / new_mrl,
                "ucmr3_native_detect_sites": native3,
                "ucmr5_native_detect_sites": native5,
                "ucmr5_detect_sites_at_ucmr3_mrl": standardized5,
                "ucmr5_measurement_enabled_detect_sites": enabled5,
                "fraction_ucmr5_detects_below_ucmr3_mrl": enabled5 / native5 if native5 else 0,
                "fraction_ucmr5_detects_below_ucmr3_mrl_ci95_low": enabled_ci[0],
                "fraction_ucmr5_detects_below_ucmr3_mrl_ci95_high": enabled_ci[1],
                "ucmr3_nd_to_ucmr5_native_detect": nd3_to_native5,
                "ucmr3_nd_to_ucmr5_below_old_mrl": nd3_to_enabled5,
                "fraction_apparent_new_detects_below_ucmr3_mrl": (
                    nd3_to_enabled5 / nd3_to_native5 if nd3_to_native5 else 0
                ),
                "fraction_apparent_new_detects_below_ucmr3_mrl_ci95_low": new_enabled_ci[0],
                "fraction_apparent_new_detects_below_ucmr3_mrl_ci95_high": new_enabled_ci[1],
                "native_paired_increase": paired_native_increase,
                "native_paired_decrease": paired_native_decrease,
                "old_mrl_standardized_paired_increase": paired_standardized_increase,
                "old_mrl_standardized_paired_decrease": paired_standardized_decrease,
                "persistent_native_detect": persistent_native,
                "states_with_ucmr5_native_detect": sum(value > 0 for value in state_native5.values()),
                "states_with_measurement_enabled_detect": sum(value > 0 for value in state_enabled.values()),
                "stable_water_type_sites": len(stable_water_type),
                "stable_water_type_ucmr3_native_detect_sites": stable_native3,
                "stable_water_type_ucmr5_native_detect_sites": stable_native5,
                "stable_water_type_ucmr5_detect_sites_at_ucmr3_mrl": stable_standardized5,
                "stable_water_type_fraction_ucmr5_detects_below_ucmr3_mrl": (
                    stable_enabled5 / stable_native5 if stable_native5 else 0
                ),
            }
        )

    summary = {
        "matched_exact_sampling_points": len(matched),
        "matched_pws": len({key[0] for key in matched}),
        "states": len({ucmr5[key][COMMON_PFAS[0]]["state"] for key in matched}),
        "state_mismatches": sum(
            ucmr3[key][COMMON_PFAS[0]]["state"]
            != ucmr5[key][COMMON_PFAS[0]]["state"]
            for key in matched
        ),
        "water_type_mismatches": sum(
            ucmr3[key][COMMON_PFAS[0]]["water_type"]
            != ucmr5[key][COMMON_PFAS[0]]["water_type"]
            for key in matched
        ),
        "sample_point_type_mismatches": sum(
            ucmr3[key][COMMON_PFAS[0]]["sample_point_type"]
            != ucmr5[key][COMMON_PFAS[0]]["sample_point_type"]
            for key in matched
        ),
        "size_category_mismatches": sum(
            ucmr3[key][COMMON_PFAS[0]]["size"]
            != ucmr5[key][COMMON_PFAS[0]]["size"]
            for key in matched
        ),
        "primary_unit": "exact PWSID + FacilityID + SamplePointID with one complete SE1 panel in both cycles",
        "identification": (
            "Directly identifies the share of UCMR5 detections below UCMR3 MRLs. "
            "It does not identify UCMR3 concentrations below UCMR3 MRLs."
        ),
    }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ucmr3-zip", type=Path, required=True)
    parser.add_argument("--ucmr5-zip", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    ucmr3, audit3 = read_se1(args.ucmr3_zip, "UCMR3")
    ucmr5, audit5 = read_se1(args.ucmr5_zip, "UCMR5")
    rows, summary = analyze(ucmr3, ucmr5)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "source": {
            "ucmr3_zip": str(args.ucmr3_zip),
            "ucmr3_sha256": sha256(args.ucmr3_zip),
            "ucmr5_zip": str(args.ucmr5_zip),
            "ucmr5_sha256": sha256(args.ucmr5_zip),
        },
        "cycle_audits": {"UCMR3": dict(audit3), "UCMR5": dict(audit5)},
        "matched_summary": summary,
        "per_chemical": rows,
    }
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

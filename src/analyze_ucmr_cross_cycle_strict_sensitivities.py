#!/usr/bin/env python3
"""Strict-event cross-cycle identity, timing, cohort, and Shapley sensitivities.

This analysis complements the count-spectrum/MRL-buffer analysis.  It uses the
same strict event key in both UCMR cycles, restricts the cross-cycle estimand to
exact identifier matches, and evaluates whether the K6 >= 2 decomposition is
stable under stronger metadata, name, source-water, and collection-timing
requirements.  It also computes the exact 6! analyte-order Shapley
decomposition of within-UCMR5 MRL reclassification.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_ucmr_cross_cycle_count_mrl_robustness import (
    bootstrap_feature_estimands,
    weighted_points,
)
from audit_ucmr_cross_cycle_identifier_confidence import (
    CHEMICALS,
    SEED,
    cycle_cohort_summary,
    month_from_date,
    normalize_name,
    season,
    sha256,
)
from audit_ucmr_se1_strict_event_key_20260815 import (
    classify as classify_strict_events,
    read_rows as read_strict_event_rows,
)


SUBSETS = (
    "identifier_matched",
    "metadata_stable",
    "name_corroborated",
    "stable_groundwater",
    "stable_surface_water",
    "same_calendar_month",
    "same_meteorological_season",
    "cyclic_month_distance_le_1",
    "cyclic_month_distance_le_2",
    "cyclic_month_distance_le_3",
)


def read_strict_panels(path: Path, cycle: str) -> tuple[dict, dict]:
    rows, raw_audit = read_strict_event_rows(path, cycle)
    _, panels, anomalies, panel_audit = classify_strict_events(rows)
    audit = {
        **dict(raw_audit),
        **dict(panel_audit),
        "strict_event_key_anomalies": len(anomalies),
    }
    return panels, audit


def event_shapley_contributions(y_native: np.ndarray, y_old: np.ndarray) -> np.ndarray:
    """Return per-event exact Shapley contributions over all 6! switch orders."""
    if y_native.shape != y_old.shape or y_native.shape[1] != len(CHEMICALS):
        raise ValueError("Expected paired event-by-six-analyte arrays")
    contributions = np.zeros_like(y_native, dtype=np.float64)
    for order in itertools.permutations(range(len(CHEMICALS))):
        current = y_native.copy()
        before = current.sum(axis=1) >= 2
        for analyte in order:
            current[:, analyte] = y_old[:, analyte]
            after = current.sum(axis=1) >= 2
            contributions[:, analyte] += before.astype(float) - after.astype(float)
            before = after
    contributions /= float(math.factorial(len(CHEMICALS)))
    event_total = (y_native.sum(axis=1) >= 2).astype(float) - (
        y_old.sum(axis=1) >= 2
    ).astype(float)
    residual = np.max(np.abs(contributions.sum(axis=1) - event_total))
    if residual > 1e-12:
        raise RuntimeError(f"Event-level Shapley additivity failed: {residual}")
    return contributions


def build_records(u3: dict, u5: dict) -> tuple[list[dict], dict]:
    records = []
    mismatch_counts: dict[str, int] = {}
    keys = sorted(set(u3) & set(u5))
    for key in keys:
        first3 = u3[key][CHEMICALS[0]]
        first5 = u5[key][CHEMICALS[0]]
        stable_fields = {
            "state": first3["state"] == first5["state"],
            "water_type": first3["water_type"] == first5["water_type"],
            "sample_point_type": first3["sample_point_type"]
            == first5["sample_point_type"],
            "size": first3["size"] == first5["size"],
        }
        name_fields = {
            field: bool(normalize_name(first3[field]))
            and normalize_name(first3[field]) == normalize_name(first5[field])
            for field in ("pws_name", "facility_name", "sample_point_name")
        }
        for field, agrees in {**stable_fields, **name_fields}.items():
            label = f"{field}_mismatch"
            mismatch_counts[label] = mismatch_counts.get(label, 0) + int(not agrees)

        y3 = np.asarray([u3[key][chemical]["detected"] for chemical in CHEMICALS])
        y5 = np.asarray([u5[key][chemical]["detected"] for chemical in CHEMICALS])
        y5_old = np.asarray(
            [
                u5[key][chemical]["detected"]
                and u5[key][chemical]["value"] is not None
                and u5[key][chemical]["value"]
                >= u3[key][chemical]["mrl"] - 1e-12
                for chemical in CHEMICALS
            ]
        )
        metadata_stable = all(stable_fields.values())
        water = str(first5["water_type"]).upper()
        month3 = month_from_date(first3["date"])
        month5 = month_from_date(first5["date"])
        distance = min(abs(month5 - month3), 12 - abs(month5 - month3))
        records.append(
            {
                "pws": key[0],
                "state": first5["state"],
                "y3": y3.astype(np.int8),
                "y5": y5.astype(np.int8),
                "y5_old": y5_old.astype(np.int8),
                "cyclic_month_distance": distance,
                "identifier_matched": True,
                "metadata_stable": metadata_stable,
                "name_corroborated": metadata_stable and all(name_fields.values()),
                "stable_groundwater": metadata_stable and water in {"GW", "GU"},
                "stable_surface_water": metadata_stable and water in {"SW", "SU"},
                "same_calendar_month": distance == 0,
                "same_meteorological_season": season(month3) == season(month5),
                "cyclic_month_distance_le_1": distance <= 1,
                "cyclic_month_distance_le_2": distance <= 2,
                "cyclic_month_distance_le_3": distance <= 3,
            }
        )
    return records, mismatch_counts


def analyze_subset(
    records: list[dict], subset: str, reps: int, chunk_size: int, seed: int
) -> tuple[list[dict], list[dict], dict]:
    selected = [record for record in records if record[subset]]
    state = np.asarray([record["state"] for record in selected], dtype=str)
    pws = np.asarray([record["pws"] for record in selected], dtype=str)
    y3 = np.stack([record["y3"] for record in selected])
    y5 = np.stack([record["y5"] for record in selected])
    y5_old = np.stack([record["y5_old"] for record in selected])
    shapley = event_shapley_contributions(y5, y5_old)
    features = np.column_stack(
        [
            (y3.sum(axis=1) >= 2).astype(float),
            (y5.sum(axis=1) >= 2).astype(float),
            (y5_old.sum(axis=1) >= 2).astype(float),
            shapley,
        ]
    )
    points = weighted_points(features, state, pws)
    bootstraps = bootstrap_feature_estimands(
        features, state, pws, reps, seed, chunk_size
    )
    contrasts = {
        "native_period": np.asarray([-1.0, 1.0, 0.0] + [0.0] * 6),
        "common_ucmr3_mrl_period": np.asarray([-1.0, 0.0, 1.0] + [0.0] * 6),
        "within_ucmr5_mrl_reclassification": np.asarray(
            [0.0, 1.0, -1.0] + [0.0] * 6
        ),
    }
    inference_rows = []
    shapley_rows = []
    for bootstrap_name, draws in bootstraps.items():
        point_name = (
            "event_equal"
            if bootstrap_name == "event_equal_state_cluster"
            else bootstrap_name
        )
        point = points[point_name]
        for contrast_name, vector in contrasts.items():
            values = draws @ vector
            inference_rows.append(
                {
                    "subset": subset,
                    "estimand": bootstrap_name,
                    "contrast": contrast_name,
                    "estimate": float(point @ vector),
                    "estimate_pp": float(100 * (point @ vector)),
                    "ci95_low_pp": float(100 * np.quantile(values, 0.025)),
                    "ci95_high_pp": float(100 * np.quantile(values, 0.975)),
                    "bootstrap_reps": reps,
                }
            )
        for analyte_index, chemical in enumerate(CHEMICALS):
            values = draws[:, 3 + analyte_index]
            shapley_rows.append(
                {
                    "subset": subset,
                    "estimand": bootstrap_name,
                    "chemical": chemical,
                    "orders": 720,
                    "contribution_pp": float(100 * point[3 + analyte_index]),
                    "ci95_low_pp": float(100 * np.quantile(values, 0.025)),
                    "ci95_high_pp": float(100 * np.quantile(values, 0.975)),
                    "bootstrap_reps": reps,
                }
            )
    point_residual = max(
        abs(points[name][3:].sum() - (points[name][1] - points[name][2]))
        for name in points
    )
    summary = {
        "pairs": len(selected),
        "pws": int(len(np.unique(pws))),
        "states": int(len(np.unique(state))),
        "maximum_shapley_additivity_residual": float(point_residual),
    }
    return inference_rows, shapley_rows, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ucmr3-zip", type=Path, required=True)
    parser.add_argument("--ucmr5-zip", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-reps", type=int, default=20_000)
    parser.add_argument("--bootstrap-chunk-size", type=int, default=250)
    args = parser.parse_args()

    u3, audit3 = read_strict_panels(args.ucmr3_zip, "UCMR3")
    u5, audit5 = read_strict_panels(args.ucmr5_zip, "UCMR5")
    records, mismatches = build_records(u3, u5)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    inference_rows, shapley_rows, subset_summaries = [], [], {}
    for index, subset in enumerate(SUBSETS):
        inference, shapley, summary = analyze_subset(
            records,
            subset,
            args.bootstrap_reps,
            args.bootstrap_chunk_size,
            SEED + 10_000 * index,
        )
        inference_rows.extend(inference)
        shapley_rows.extend(shapley)
        subset_summaries[subset] = summary

    inference = pd.DataFrame(inference_rows)
    shapley = pd.DataFrame(shapley_rows)
    inference.to_csv(output / "subset_inference.csv", index=False)
    shapley.to_csv(output / "shapley_contributions.csv", index=False)

    keys = set(u3) & set(u5)
    cohort_rows = cycle_cohort_summary(u3, keys, "UCMR3")
    cohort_rows.extend(cycle_cohort_summary(u5, keys, "UCMR5"))
    cohort = pd.DataFrame(cohort_rows)
    cohort.to_csv(output / "matched_unmatched_cohort_comparison.csv", index=False)

    distances = np.asarray([record["cyclic_month_distance"] for record in records])
    primary = inference[
        inference["subset"].eq("identifier_matched")
        & inference["estimand"].eq("state_pws_two_level")
    ]
    primary_shapley = shapley[
        shapley["subset"].eq("identifier_matched")
        & shapley["estimand"].eq("state_pws_two_level")
    ]
    decision = {
        "status": "STRICT CROSS-CYCLE SENSITIVITY AND SHAPLEY ANALYSIS COMPLETE",
        "strict_event_key": True,
        "source": {
            "ucmr3": str(args.ucmr3_zip),
            "ucmr3_sha256": sha256(args.ucmr3_zip),
            "ucmr5": str(args.ucmr5_zip),
            "ucmr5_sha256": sha256(args.ucmr5_zip),
        },
        "cycle_audits": {"UCMR3": audit3, "UCMR5": audit5},
        "identifier_matched_pairs": len(records),
        "subsets": subset_summaries,
        "collection_timing": {
            "cyclic_month_distance_counts": {
                str(distance): int(np.sum(distances == distance))
                for distance in sorted(np.unique(distances))
            },
            "median_cyclic_month_distance": float(np.median(distances)),
        },
        "metadata_mismatches": mismatches,
        "primary_state_pws_inference": {
            row.contrast: {
                "estimate_pp": float(row.estimate_pp),
                "ci95_pp": [float(row.ci95_low_pp), float(row.ci95_high_pp)],
            }
            for row in primary.itertuples()
        },
        "primary_state_pws_shapley": {
            row.chemical: {
                "contribution_pp": float(row.contribution_pp),
                "ci95_pp": [float(row.ci95_low_pp), float(row.ci95_high_pp)],
            }
            for row in primary_shapley.itertuples()
        },
        "shapley_orders": 720,
        "maximum_shapley_additivity_residual": float(
            max(item["maximum_shapley_additivity_residual"] for item in subset_summaries.values())
        ),
        "primary_estimand": "state-equal mean of within-state PWS-equal means",
        "bootstrap": {"reps": args.bootstrap_reps, "seed": SEED},
        "interpretation_boundary": (
            "Metadata and normalized-name restrictions corroborate administrative identity "
            "but do not prove unchanged hydraulics, treatment, or physical sampling location. "
            "Shapley values decompose the classification effect of analyte-specific MRL "
            "changes and are not contamination-source attributions."
        ),
    }
    (output / "decision.json").write_text(
        json.dumps(decision, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(decision, indent=2), flush=True)


if __name__ == "__main__":
    main()

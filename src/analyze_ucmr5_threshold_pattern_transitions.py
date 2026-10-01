#!/usr/bin/env python3
"""Describe how UCMR 5 six-PFAS reporting patterns change at UCMR 3 cutoffs.

The analysis is deliberately descriptive and uses the same strict, identifier-
matched SE1 cohort as the cross-cycle decomposition.  It never treats results
below an older reporting cutoff as false measurements.  Instead, it records
how the monitoring-derived pattern changes when the same UCMR 5 concentrations
are evaluated under the older numeric cutoffs.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from audit_ucmr_cross_cycle_identifier_confidence import CHEMICALS, sha256
from audit_ucmr_se1_strict_event_key_20260815 import classify, read_rows


def code_from_flags(flags: list[bool]) -> int:
    return int(sum((1 << index) for index, value in enumerate(flags) if value))


def pattern_name(code: int) -> str:
    members = [chemical for index, chemical in enumerate(CHEMICALS) if code & (1 << index)]
    return "+".join(members) if members else "none"


def category(count: int) -> str:
    if count == 0:
        return "K6 = 0"
    if count == 1:
        return "K6 = 1"
    return "K6 >= 2"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ucmr3", type=Path, required=True)
    parser.add_argument("--ucmr5", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-patterns", type=int, default=8)
    args = parser.parse_args()

    rows3, raw3 = read_rows(args.ucmr3, "UCMR3")
    rows5, raw5 = read_rows(args.ucmr5, "UCMR5")
    _, strict3, anomalies3, audit3 = classify(rows3)
    _, strict5, anomalies5, audit5 = classify(rows5)
    keys = sorted(set(strict3) & set(strict5))

    records: list[dict] = []
    for key in keys:
        native_flags = [bool(strict5[key][chemical]["detected"]) for chemical in CHEMICALS]
        common_flags = [
            bool(
                strict5[key][chemical]["detected"]
                and strict5[key][chemical]["value"] is not None
                and strict5[key][chemical]["value"]
                >= strict3[key][chemical]["mrl"] - 1e-12
            )
            for chemical in CHEMICALS
        ]
        native_code = code_from_flags(native_flags)
        common_code = code_from_flags(common_flags)
        first = strict5[key][CHEMICALS[0]]
        records.append(
            {
                "location_id": "|".join(key),
                "pws_id": str(key[0]),
                "state": str(first["state"]),
                "native_code": native_code,
                "native_pattern": pattern_name(native_code),
                "native_count": int(sum(native_flags)),
                "common_code": common_code,
                "common_pattern": pattern_name(common_code),
                "common_count": int(sum(common_flags)),
            }
        )

    frame = pd.DataFrame(records)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    count_grid = (
        frame.groupby(["native_count", "common_count"], as_index=False)
        .size()
        .rename(columns={"size": "events"})
    )
    full_grid = pd.MultiIndex.from_product(
        [range(7), range(7)], names=["native_count", "common_count"]
    ).to_frame(index=False)
    count_grid = full_grid.merge(count_grid, how="left").fillna({"events": 0})
    count_grid["events"] = count_grid["events"].astype(int)
    count_grid.to_csv(output / "count_transition_matrix.csv", index=False)

    exact = (
        frame.groupby(
            ["native_code", "native_pattern", "native_count", "common_code", "common_pattern", "common_count"],
            as_index=False,
        )
        .size()
        .rename(columns={"size": "events"})
        .sort_values(["native_count", "native_code", "events"], ascending=[True, True, False])
    )
    exact.to_csv(output / "exact_pattern_transitions.csv", index=False)

    native_multi = frame[frame["native_count"].ge(2)].copy()
    native_multi_n = len(native_multi)
    destination = (
        native_multi.assign(common_category=native_multi["common_count"].map(category))
        .groupby("common_category", as_index=False)
        .size()
        .rename(columns={"size": "events"})
    )
    order = pd.DataFrame({"common_category": ["K6 = 0", "K6 = 1", "K6 >= 2"]})
    destination = order.merge(destination, how="left").fillna({"events": 0})
    destination["events"] = destination["events"].astype(int)
    destination["percent_of_native_multi_events"] = 100 * destination["events"] / native_multi_n
    destination.to_csv(output / "native_multi_destination.csv", index=False)

    summary_rows = []
    for (code, name, count), part in native_multi.groupby(
        ["native_code", "native_pattern", "native_count"], observed=True
    ):
        destinations = Counter(category(value) for value in part["common_count"])
        exact_destinations = Counter(part["common_pattern"])
        summary_rows.append(
            {
                "native_code": int(code),
                "native_pattern": name,
                "native_count": int(count),
                "events": len(part),
                "share_of_native_multi_events_percent": 100 * len(part) / native_multi_n,
                "to_K6_0": destinations["K6 = 0"],
                "to_K6_1": destinations["K6 = 1"],
                "to_K6_ge_2": destinations["K6 >= 2"],
                "retained_multi_percent": 100 * destinations["K6 >= 2"] / len(part),
                "most_common_result_at_old_cutoffs": exact_destinations.most_common(1)[0][0],
                "most_common_result_events": exact_destinations.most_common(1)[0][1],
            }
        )
    pattern_summary = pd.DataFrame(summary_rows).sort_values(
        ["events", "native_count", "native_code"], ascending=[False, False, True]
    )
    pattern_summary["native_rank"] = np.arange(1, len(pattern_summary) + 1)
    pattern_summary["cumulative_share_percent"] = pattern_summary[
        "share_of_native_multi_events_percent"
    ].cumsum()
    pattern_summary.to_csv(output / "native_multi_pattern_summary.csv", index=False)
    pattern_summary.head(args.top_patterns).to_csv(
        output / "top_native_multi_patterns.csv", index=False
    )

    rank_parts = []
    for regime, code_col, pattern_col, count_col in (
        ("UCMR 5 native reporting limits", "native_code", "native_pattern", "native_count"),
        ("UCMR 3 numeric cutoffs", "common_code", "common_pattern", "common_count"),
    ):
        current = (
            frame.groupby([code_col, pattern_col, count_col], as_index=False)
            .size()
            .rename(
                columns={
                    code_col: "pattern_code",
                    pattern_col: "pattern",
                    count_col: "reported_count",
                    "size": "events",
                }
            )
            .sort_values(["events", "pattern_code"], ascending=[False, True])
        )
        current["rank"] = np.arange(1, len(current) + 1)
        current["event_percent"] = 100 * current["events"] / len(frame)
        current.insert(0, "regime", regime)
        rank_parts.append(current)
    pd.concat(rank_parts, ignore_index=True).to_csv(
        output / "pattern_frequency_ranking.csv", index=False
    )

    complexity_rows = []
    for regime, count_col in (
        ("UCMR 5 native reporting limits", "native_count"),
        ("UCMR 3 numeric cutoffs", "common_count"),
    ):
        any_reported = int(frame[count_col].ge(1).sum())
        multi_reported = int(frame[count_col].ge(2).sum())
        complexity_rows.append(
            {
                "regime": regime,
                "matched_events": len(frame),
                "events_with_at_least_one_reported_pfas": any_reported,
                "events_with_at_least_two_reported_pfas": multi_reported,
                "multi_pfas_share_among_any_reported_percent": 100 * multi_reported / any_reported,
            }
        )
    pd.DataFrame(complexity_rows).to_csv(output / "conditional_mixture_complexity.csv", index=False)

    pws_rows = []
    for regime, count_col in (
        ("UCMR 5 native reporting limits", "native_count"),
        ("UCMR 3 numeric cutoffs", "common_count"),
    ):
        positive = frame[frame[count_col].ge(2)]
        pws_rows.append(
            {
                "regime": regime,
                "matched_public_water_systems": int(frame["pws_id"].nunique()),
                "public_water_systems_with_any_multi_pfas_event": int(positive["pws_id"].nunique()),
                "reporting_jurisdictions_with_any_multi_pfas_event": int(positive["state"].nunique()),
            }
        )
    pd.DataFrame(pws_rows).to_csv(output / "monitoring_consequences.csv", index=False)

    top = pattern_summary.head(args.top_patterns)
    decision = {
        "status": "THRESHOLD PATTERN TRANSITION ANALYSIS COMPLETE",
        "scope": "strict identifier-matched UCMR 3-UCMR 5 SE1 entry points",
        "interpretation": (
            "Transitions describe monitoring visibility under alternative numeric cutoffs. "
            "They do not classify measurements below the older cutoffs as false contamination."
        ),
        "sources": {
            "ucmr3": str(args.ucmr3),
            "ucmr3_sha256": sha256(args.ucmr3),
            "ucmr5": str(args.ucmr5),
            "ucmr5_sha256": sha256(args.ucmr5),
        },
        "raw_rows": {"ucmr3": int(raw3["raw_rows"]), "ucmr5": int(raw5["raw_rows"])},
        "strict_locations": {
            "ucmr3": int(audit3["strict_complete_unambiguous_locations"]),
            "ucmr5": int(audit5["strict_complete_unambiguous_locations"]),
        },
        "strict_event_anomalies": {"ucmr3": len(anomalies3), "ucmr5": len(anomalies5)},
        "matched_events": len(frame),
        "matched_public_water_systems": int(frame["pws_id"].nunique()),
        "native_multi_events": native_multi_n,
        "native_multi_destinations": {
            row.common_category: {
                "events": int(row.events),
                "percent": float(row.percent_of_native_multi_events),
            }
            for row in destination.itertuples(index=False)
        },
        "top_pattern_rule": f"{args.top_patterns} most frequent exact native-limit patterns among K6 >= 2 events",
        "top_patterns_cover_percent": float(top["share_of_native_multi_events_percent"].sum()),
        "audits": {
            "count_transition_sum": int(count_grid["events"].sum()),
            "exact_transition_sum": int(exact["events"].sum()),
            "native_pattern_summary_sum": int(pattern_summary["events"].sum()),
            "common_is_subset_of_native_for_every_event": bool(
                np.all((frame["common_code"] & frame["native_code"]) == frame["common_code"])
            ),
        },
    }
    if decision["audits"]["count_transition_sum"] != len(frame):
        raise RuntimeError("Count transition table does not sum to the matched cohort")
    if decision["audits"]["exact_transition_sum"] != len(frame):
        raise RuntimeError("Exact transition table does not sum to the matched cohort")
    if decision["audits"]["native_pattern_summary_sum"] != native_multi_n:
        raise RuntimeError("Native multi-pattern table does not sum to K6 >= 2 events")
    if not decision["audits"]["common_is_subset_of_native_for_every_event"]:
        raise RuntimeError("A common-cutoff pattern is not a subset of its native pattern")
    (output / "decision.json").write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()

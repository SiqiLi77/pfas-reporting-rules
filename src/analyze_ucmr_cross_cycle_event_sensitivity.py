#!/usr/bin/env python3
"""Assess whether the cross-cycle PFAS reporting contrast depends on SE1.

For each scheduled sample-event code (SE1-SE4), this script constructs one
complete and unambiguous six-PFAS event at each administrative sampling
location in UCMR 3 and UCMR 5. It then pairs locations across cycles and
repeats the native-cutoff and harmonized-numeric-cutoff comparison. The
analysis is descriptive: administrative identifiers do not establish an
unchanged intake, treatment train, or water source between cycles.
"""

from __future__ import annotations

import argparse
import csv
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


PFAS = ("PFBS", "PFHpA", "PFHxS", "PFNA", "PFOA", "PFOS")
EVENT_CODES = ("SE1", "SE2", "SE3", "SE4")


def archive_member(path: Path, cycle: str) -> str:
    target = f"{cycle}_All.txt"
    with zipfile.ZipFile(path) as archive:
        matches = [name for name in archive.namelist() if name.endswith(target)]
    if len(matches) != 1:
        raise ValueError(f"Expected one {target}, found {matches}")
    return matches[0]


def read_rows(path: Path, cycle: str) -> tuple[dict, Counter]:
    rows_by_location_event: dict[tuple, list[dict]] = defaultdict(list)
    audit = Counter()
    member = archive_member(path, cycle)
    with zipfile.ZipFile(path) as archive, archive.open(member) as binary:
        reader = csv.DictReader(
            (line.decode("latin-1") for line in binary), delimiter="\t"
        )
        for row in reader:
            audit["raw_rows"] += 1
            event_code = row["SampleEventCode"].strip()
            chemical = row["Contaminant"].strip()
            if event_code not in EVENT_CODES or chemical not in PFAS:
                continue
            audit[f"shared_pfas_{event_code.lower()}_rows"] += 1
            location = (row["PWSID"], row["FacilityID"], row["SamplePointID"])
            sign = row["AnalyticalResultsSign"].strip()
            value_text = row["AnalyticalResultValue"].strip()
            rows_by_location_event[(location, event_code)].append(
                {
                    "chemical": chemical,
                    "date": row["CollectionDate"].strip(),
                    "sample_id": row["SampleID"].strip(),
                    "mrl": float(row["MRL"]),
                    "detected": sign == "=",
                    "value": float(value_text) if value_text else None,
                    "state": row["State"].strip(),
                    "pws": row["PWSID"].strip(),
                    "size": row["Size"].strip(),
                    "water_type": row["FacilityWaterType"].strip(),
                    "sample_point_type": row["SamplePointType"].strip(),
                }
            )
    return dict(rows_by_location_event), audit


def classify(rows_by_location_event: dict) -> tuple[dict, Counter]:
    panels: dict[str, dict[tuple, dict[str, dict]]] = {
        event: {} for event in EVENT_CODES
    }
    audit = Counter()
    for (location, event_code), rows in sorted(rows_by_location_event.items()):
        event_groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for row in rows:
            event_groups[(row["date"], row["sample_id"])].append(row)

        complete_events: list[list[dict]] = []
        for event_rows in event_groups.values():
            counts = Counter(row["chemical"] for row in event_rows)
            if set(counts) == set(PFAS) and all(counts[name] == 1 for name in PFAS):
                complete_events.append(event_rows)

        # Require the event-code panel to contain exactly one six-PFAS sample
        # and no additional shared-PFAS rows at the same location/event code.
        if len(complete_events) != 1 or len(rows) != len(PFAS):
            audit[f"{event_code.lower()}_excluded_ambiguous_or_incomplete"] += 1
            continue
        event_rows = complete_events[0]
        panels[event_code][location] = {
            row["chemical"]: row for row in event_rows
        }
        audit[f"{event_code.lower()}_complete_unambiguous"] += 1
    return panels, audit


def state_pws_mean(values: np.ndarray, states: np.ndarray, pws: np.ndarray) -> np.ndarray:
    state_values = []
    for state in sorted(set(states.tolist())):
        state_mask = states == state
        pws_values = []
        for system in sorted(set(pws[state_mask].tolist())):
            pws_values.append(values[state_mask & (pws == system)].mean(axis=0))
        state_values.append(np.stack(pws_values).mean(axis=0))
    return np.stack(state_values).mean(axis=0)


def bootstrap_state_pws(
    values: np.ndarray,
    states: np.ndarray,
    pws: np.ndarray,
    *,
    reps: int,
    seed: int,
    chunk_size: int = 500,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    state_names = np.asarray(sorted(set(states.tolist())), dtype=object)
    within_state: dict[str, np.ndarray] = {}
    for state in state_names:
        state_mask = states == state
        systems = sorted(set(pws[state_mask].tolist()))
        within_state[str(state)] = np.stack(
            [values[state_mask & (pws == system)].mean(axis=0) for system in systems]
        )

    # Match the two-level resampling used by the primary analysis: draw
    # jurisdictions with replacement, and independently draw PWSs within each
    # jurisdiction. Multinomial weights avoid a Python loop over replicates.
    state_weights = rng.multinomial(
        len(state_names),
        np.full(len(state_names), 1.0 / len(state_names)),
        size=reps,
    )
    draws = np.zeros((reps, values.shape[1]), dtype=float)
    for state_index, state in enumerate(state_names):
        pws_values = within_state[str(state)]
        group_count = len(pws_values)
        probabilities = np.full(group_count, 1.0 / group_count)
        for start in range(0, reps, chunk_size):
            stop = min(start + chunk_size, reps)
            pws_weights = rng.multinomial(
                group_count, probabilities, size=stop - start
            )
            state_means = pws_weights @ pws_values / group_count
            draws[start:stop] += (
                state_weights[start:stop, state_index, None]
                * state_means
                / len(state_names)
            )
    return draws


def summarize_event(
    event_code: str,
    u3: dict,
    u5: dict,
    *,
    reps: int,
    seed: int,
) -> tuple[dict, list[dict]]:
    keys = sorted(set(u3) & set(u5))
    if not keys:
        return {"event_code": event_code, "pairs": 0}, []

    values = np.zeros((len(keys), 3), dtype=float)
    states = np.empty(len(keys), dtype=object)
    pws = np.empty(len(keys), dtype=object)
    water_types = Counter()
    size_transitions = Counter()
    sample_point_types = Counter()
    for index, key in enumerate(keys):
        first3 = u3[key][PFAS[0]]
        first5 = u5[key][PFAS[0]]
        states[index] = first5["state"]
        pws[index] = first5["pws"]
        water_types[first5["water_type"]] += 1
        size_transitions[f"{first3['size']}->{first5['size']}"] += 1
        sample_point_types[first5["sample_point_type"]] += 1

        k3 = sum(int(u3[key][chemical]["detected"]) for chemical in PFAS)
        k5 = sum(int(u5[key][chemical]["detected"]) for chemical in PFAS)
        k5_old = sum(
            int(
                u5[key][chemical]["detected"]
                and u5[key][chemical]["value"] is not None
                and u5[key][chemical]["value"] >= u3[key][chemical]["mrl"] - 1e-12
            )
            for chemical in PFAS
        )
        values[index] = (k3 >= 2, k5 >= 2, k5_old >= 2)

    point = state_pws_mean(values, states, pws)
    draws = bootstrap_state_pws(values, states, pws, reps=reps, seed=seed)
    contrasts = {
        "native_period": (values[:, 1] - values[:, 0], point[1] - point[0], draws[:, 1] - draws[:, 0]),
        "within_ucmr5_reclassification": (values[:, 1] - values[:, 2], point[1] - point[2], draws[:, 1] - draws[:, 2]),
        "harmonized_numeric_cutoff_period": (values[:, 2] - values[:, 0], point[2] - point[0], draws[:, 2] - draws[:, 0]),
    }
    contrast_rows = []
    for name, (event_diff, estimate, bootstrap) in contrasts.items():
        low, high = np.quantile(bootstrap, [0.025, 0.975])
        contrast_rows.append(
            {
                "event_code": event_code,
                "contrast": name,
                "unweighted_difference_pp": 100 * float(event_diff.mean()),
                "primary_difference_pp": 100 * float(estimate),
                "ci95_low_pp": 100 * float(low),
                "ci95_high_pp": 100 * float(high),
                "pairs": len(keys),
                "pws": len(set(pws.tolist())),
                "jurisdictions": len(set(states.tolist())),
            }
        )

    summary = {
        "event_code": event_code,
        "pairs": len(keys),
        "pws": len(set(pws.tolist())),
        "jurisdictions": len(set(states.tolist())),
        "unweighted_counts": {
            "ucmr3_native_k6_ge_2": int(values[:, 0].sum()),
            "ucmr5_native_k6_ge_2": int(values[:, 1].sum()),
            "ucmr5_at_ucmr3_numeric_cutoffs_k6_ge_2": int(values[:, 2].sum()),
        },
        "unweighted_rates": {
            "ucmr3_native": float(values[:, 0].mean()),
            "ucmr5_native": float(values[:, 1].mean()),
            "ucmr5_at_ucmr3_numeric_cutoffs": float(values[:, 2].mean()),
        },
        "primary_rates": {
            "ucmr3_native": float(point[0]),
            "ucmr5_native": float(point[1]),
            "ucmr5_at_ucmr3_numeric_cutoffs": float(point[2]),
        },
        "water_type_counts_ucmr5": dict(sorted(water_types.items())),
        "size_transitions": dict(sorted(size_transitions.items())),
        "sample_point_type_counts_ucmr5": dict(sorted(sample_point_types.items())),
    }
    return summary, contrast_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ucmr3", type=Path, required=True)
    parser.add_argument("--ucmr5", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-reps", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260825)
    args = parser.parse_args()

    rows3, raw3 = read_rows(args.ucmr3, "UCMR3")
    rows5, raw5 = read_rows(args.ucmr5, "UCMR5")
    panels3, audit3 = classify(rows3)
    panels5, audit5 = classify(rows5)

    event_summaries = []
    contrast_rows = []
    for offset, event_code in enumerate(EVENT_CODES):
        summary, rows = summarize_event(
            event_code,
            panels3[event_code],
            panels5[event_code],
            reps=args.bootstrap_reps,
            seed=args.seed + offset,
        )
        event_summaries.append(summary)
        contrast_rows.extend(rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "CROSS-CYCLE SAMPLE-EVENT SENSITIVITY COMPLETE",
        "estimand": "jurisdiction-equal mean of within-jurisdiction PWS-equal means",
        "bootstrap": {
            "design": "two-level jurisdiction and PWS resampling",
            "reps": args.bootstrap_reps,
            "seed": args.seed,
        },
        "event_key": [
            "PWSID",
            "FacilityID",
            "SamplePointID",
            "SampleEventCode",
            "CollectionDate",
            "SampleID",
        ],
        "raw_audit": {"UCMR3": dict(raw3), "UCMR5": dict(raw5)},
        "panel_audit": {"UCMR3": dict(audit3), "UCMR5": dict(audit5)},
        "events": event_summaries,
        "interpretive_boundary": (
            "Event-code analyses are descriptive administrative matches and do not "
            "establish unchanged source water, intake configuration, treatment, or exposure."
        ),
    }
    (args.output_dir / "decision.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "event_contrasts.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(contrast_rows[0]))
        writer.writeheader()
        writer.writerows(contrast_rows)
    print(json.dumps({"events": event_summaries, "contrasts": contrast_rows}, indent=2))


if __name__ == "__main__":
    main()

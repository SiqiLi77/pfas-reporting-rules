#!/usr/bin/env python3
"""Count-spectrum and MRL-buffer robustness for matched UCMR3/UCMR5 panels.

This is a prespecified-style sensitivity extension of the corrected identifier
audit.  It evaluates every K6 >= r endpoint (r=1,...,6), and raises the common
comparison threshold to c times each analyte's UCMR3 MRL for c >= 1.  Both
cycles are re-thresholded, so the native contrast has the exact decomposition

  U5_native - U3_native
    = (U5_native - U5_c) + (U5_c - U3_c) + (U3_c - U3_native).

No value below a cycle's reporting limit is imputed.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from audit_ucmr_cross_cycle_identifier_confidence import (
    BOOTSTRAP_REPS,
    CHEMICALS,
    SEED,
    normalize_name,
    read_panels,
    sha256,
)
from audit_ucmr_se1_strict_event_key_20260815 import (
    classify as classify_strict_events,
    read_rows as read_strict_event_rows,
)


DEFAULT_BUFFERS = (1.0, 1.10, 1.25, 1.50, 2.0)
SUBSETS = ("identifier_matched", "metadata_stable", "name_corroborated")


def buffer_label(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".")


def weighted_points(features: np.ndarray, state: np.ndarray, pws: np.ndarray) -> dict:
    """Return event-, PWS-, and state/PWS-equal point estimands."""
    state = state.astype(str)
    pws = pws.astype(str)
    pws_names = np.unique(pws)
    pws_means = np.stack([features[pws == name].mean(axis=0) for name in pws_names])
    state_names = np.unique(state)
    state_pws_means = []
    for state_name in state_names:
        mask = state == state_name
        names = np.unique(pws[mask])
        state_pws_means.append(
            np.stack([features[mask & (pws == name)].mean(axis=0) for name in names]).mean(axis=0)
        )
    return {
        "event_equal": features.mean(axis=0),
        "pws_equal": pws_means.mean(axis=0),
        "state_pws_two_level": np.stack(state_pws_means).mean(axis=0),
    }


def _multinomial_chunks(
    rng: np.random.Generator,
    group_features: np.ndarray,
    draws_per_replicate: int,
    reps: int,
    chunk_size: int,
):
    groups = len(group_features)
    probabilities = np.full(groups, 1.0 / groups)
    for start in range(0, reps, chunk_size):
        stop = min(start + chunk_size, reps)
        weights = rng.multinomial(draws_per_replicate, probabilities, size=stop - start)
        yield start, stop, weights @ group_features / draws_per_replicate


def bootstrap_feature_estimands(
    features: np.ndarray,
    state: np.ndarray,
    pws: np.ndarray,
    reps: int,
    seed: int,
    chunk_size: int,
) -> dict[str, np.ndarray]:
    """Bootstrap all feature rates once, then reuse them for every contrast."""
    state = state.astype(str)
    pws = pws.astype(str)
    state_names = np.unique(state)
    pws_names = np.unique(pws)
    output = {
        "event_equal_state_cluster": np.empty((reps, features.shape[1]), dtype=np.float64),
        "pws_equal": np.empty((reps, features.shape[1]), dtype=np.float64),
        "state_pws_two_level": np.zeros((reps, features.shape[1]), dtype=np.float64),
    }

    # Resampling state clusters while retaining the event-equal estimand.
    state_totals = np.stack([features[state == name].sum(axis=0) for name in state_names])
    state_counts = np.asarray([np.sum(state == name) for name in state_names], dtype=float)
    rng = np.random.default_rng(seed)
    probabilities = np.full(len(state_names), 1.0 / len(state_names))
    for start in range(0, reps, chunk_size):
        stop = min(start + chunk_size, reps)
        weights = rng.multinomial(len(state_names), probabilities, size=stop - start)
        output["event_equal_state_cluster"][start:stop] = (
            weights @ state_totals
        ) / (weights @ state_counts)[:, None]

    # Equal-PWS bootstrap.
    pws_features = np.stack([features[pws == name].mean(axis=0) for name in pws_names])
    rng = np.random.default_rng(seed + 1000)
    for start, stop, values in _multinomial_chunks(
        rng, pws_features, len(pws_names), reps, chunk_size
    ):
        output["pws_equal"][start:stop] = values

    # State-equal mean of within-state PWS-equal means, with two resampling levels.
    rng = np.random.default_rng(seed + 2000)
    state_weights = rng.multinomial(
        len(state_names), np.full(len(state_names), 1.0 / len(state_names)), size=reps
    )
    for state_index, state_name in enumerate(state_names):
        mask = state == state_name
        names = np.unique(pws[mask])
        group_features = np.stack(
            [features[mask & (pws == name)].mean(axis=0) for name in names]
        )
        for start, stop, values in _multinomial_chunks(
            rng, group_features, len(names), reps, chunk_size
        ):
            output["state_pws_two_level"][start:stop] += (
                state_weights[start:stop, state_index, None] * values / len(state_names)
            )
    return output


def make_records(u3: dict, u5: dict, buffers: tuple[float, ...]) -> tuple[list[dict], Counter]:
    records = []
    mismatches = Counter()
    for key in sorted(set(u3) & set(u5)):
        first3 = u3[key][CHEMICALS[0]]
        first5 = u5[key][CHEMICALS[0]]
        stable_fields = {
            "state": first3["state"] == first5["state"],
            "water_type": first3["water_type"] == first5["water_type"],
            "sample_point_type": first3["sample_point_type"] == first5["sample_point_type"],
            "size": first3["size"] == first5["size"],
        }
        name_fields = {
            field: bool(normalize_name(first3[field]))
            and normalize_name(first3[field]) == normalize_name(first5[field])
            for field in ("pws_name", "facility_name", "sample_point_name")
        }
        for field, same in {**stable_fields, **name_fields}.items():
            mismatches[f"{field}_mismatch"] += int(not same)

        y3_native = np.asarray([u3[key][c]["detected"] for c in CHEMICALS], dtype=np.int8)
        y5_native = np.asarray([u5[key][c]["detected"] for c in CHEMICALS], dtype=np.int8)
        k3_common, k5_common = {}, {}
        for multiplier in buffers:
            label = buffer_label(multiplier)
            k3_common[label] = int(
                sum(
                    u3[key][c]["detected"]
                    and u3[key][c]["value"] is not None
                    and u3[key][c]["value"] >= multiplier * u3[key][c]["mrl"] - 1e-12
                    for c in CHEMICALS
                )
            )
            k5_common[label] = int(
                sum(
                    u5[key][c]["detected"]
                    and u5[key][c]["value"] is not None
                    and u5[key][c]["value"] >= multiplier * u3[key][c]["mrl"] - 1e-12
                    for c in CHEMICALS
                )
            )
        metadata_stable = all(stable_fields.values())
        records.append(
            {
                "pws": key[0],
                "state": first5["state"],
                "k3_native": int(y3_native.sum()),
                "k5_native": int(y5_native.sum()),
                "k3_common": k3_common,
                "k5_common": k5_common,
                "identifier_matched": True,
                "metadata_stable": metadata_stable,
                "name_corroborated": metadata_stable and all(name_fields.values()),
            }
        )
    return records, mismatches


def feature_design(records: list[dict], buffers: tuple[float, ...]):
    regimes = [("ucmr3_native", None), ("ucmr5_native", None)]
    for multiplier in buffers:
        label = buffer_label(multiplier)
        regimes.extend([(f"ucmr3_common_c{label}", label), (f"ucmr5_common_c{label}", label)])
    specs = [(regime, count) for regime, _ in regimes for count in range(7)]
    matrix = np.zeros((len(records), len(specs)), dtype=np.float64)
    counts_by_regime = {}
    for regime, label in regimes:
        if regime == "ucmr3_native":
            counts = np.asarray([row["k3_native"] for row in records])
        elif regime == "ucmr5_native":
            counts = np.asarray([row["k5_native"] for row in records])
        elif regime.startswith("ucmr3"):
            counts = np.asarray([row["k3_common"][label] for row in records])
        else:
            counts = np.asarray([row["k5_common"][label] for row in records])
        counts_by_regime[regime] = counts
        offset = regimes.index((regime, label)) * 7
        matrix[np.arange(len(records)), offset + counts] = 1.0
    index = {spec: column for column, spec in enumerate(specs)}
    return matrix, index, counts_by_regime


def tail_vector(index: dict, regime: str, k: int) -> np.ndarray:
    vector = np.zeros(len(index), dtype=float)
    for count in range(k, 7):
        vector[index[(regime, count)]] = 1.0
    return vector


def contrast_design(index: dict, buffers: tuple[float, ...]):
    specs, columns = [], []
    for k in range(1, 7):
        specs.append({"contrast": "native_period", "buffer": None, "k": k})
        columns.append(tail_vector(index, "ucmr5_native", k) - tail_vector(index, "ucmr3_native", k))
    for multiplier in buffers:
        label = buffer_label(multiplier)
        u3c, u5c = f"ucmr3_common_c{label}", f"ucmr5_common_c{label}"
        for k in range(1, 7):
            u3n = tail_vector(index, "ucmr3_native", k)
            u5n = tail_vector(index, "ucmr5_native", k)
            u3 = tail_vector(index, u3c, k)
            u5 = tail_vector(index, u5c, k)
            for name, vector in (
                ("common_threshold_period", u5 - u3),
                ("within_ucmr5_threshold_shift", u5n - u5),
                ("within_ucmr3_threshold_shift", u3 - u3n),
            ):
                specs.append({"contrast": name, "buffer": multiplier, "k": k})
                columns.append(vector)
    return specs, np.stack(columns, axis=1)


def mrl_table(u3: dict, u5: dict) -> pd.DataFrame:
    rows = []
    for cycle, panels in (("UCMR3", u3), ("UCMR5", u5)):
        for chemical in CHEMICALS:
            values = np.asarray([panel[chemical]["mrl"] for panel in panels.values()])
            counts = Counter(float(value) for value in values)
            rows.append(
                {
                    "cycle": cycle,
                    "chemical": chemical,
                    "sites": len(values),
                    "minimum_mrl": float(values.min()),
                    "median_mrl": float(np.median(values)),
                    "maximum_mrl": float(values.max()),
                    "unique_mrls": len(counts),
                    "modal_mrl": counts.most_common(1)[0][0],
                    "modal_mrl_fraction": counts.most_common(1)[0][1] / len(values),
                    "mrl_value_counts": json.dumps(dict(sorted(counts.items()))),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ucmr3-zip", type=Path, required=True)
    parser.add_argument("--ucmr5-zip", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--buffers", default=",".join(map(str, DEFAULT_BUFFERS)))
    parser.add_argument("--bootstrap-reps", type=int, default=BOOTSTRAP_REPS)
    parser.add_argument("--bootstrap-chunk-size", type=int, default=250)
    parser.add_argument(
        "--strict-event-key",
        action="store_true",
        help=(
            "Retain only one-to-one SE1 events under PWSID + FacilityID + "
            "SamplePointID + CollectionDate + SampleID."
        ),
    )
    args = parser.parse_args()
    buffers = tuple(float(value) for value in args.buffers.split(","))
    if any(value < 1.0 for value in buffers):
        raise ValueError("MRL multipliers below 1 are not identifiable from censored reports")

    if args.strict_event_key:
        rows3, raw_audit3 = read_strict_event_rows(args.ucmr3_zip, "UCMR3")
        rows5, raw_audit5 = read_strict_event_rows(args.ucmr5_zip, "UCMR5")
        _, u3, anomalies3, panel_audit3 = classify_strict_events(rows3)
        _, u5, anomalies5, panel_audit5 = classify_strict_events(rows5)
        audit3 = {
            **dict(raw_audit3),
            **dict(panel_audit3),
            "strict_event_key_anomalies": len(anomalies3),
        }
        audit5 = {
            **dict(raw_audit5),
            **dict(panel_audit5),
            "strict_event_key_anomalies": len(anomalies5),
        }
    else:
        u3, audit3 = read_panels(args.ucmr3_zip, "UCMR3")
        u5, audit5 = read_panels(args.ucmr5_zip, "UCMR5")
    records, mismatches = make_records(u3, u5, buffers)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    rate_rows, transition_rows, subset_summaries = [], [], {}
    primary_payload = None
    for subset_index, subset in enumerate(SUBSETS):
        selected = [row for row in records if row[subset]]
        feature, feature_index, counts = feature_design(selected, buffers)
        contrast_specs, contrast_matrix = contrast_design(feature_index, buffers)
        state = np.asarray([row["state"] for row in selected], dtype=str)
        pws = np.asarray([row["pws"] for row in selected], dtype=str)
        points = weighted_points(feature, state, pws)
        subset_summaries[subset] = {
            "pairs": len(selected),
            "pws": int(len(np.unique(pws))),
            "states": int(len(np.unique(state))),
        }

        for estimand, feature_point in points.items():
            for regime in sorted(counts):
                for k in range(1, 7):
                    rate_rows.append(
                        {
                            "subset": subset,
                            "estimand": estimand,
                            "regime": regime,
                            "k": k,
                            "rate": float(feature_point @ tail_vector(feature_index, regime, k)),
                        }
                    )

        for multiplier in buffers:
            label = buffer_label(multiplier)
            for before_name, after_name, transition in (
                ("ucmr3_native", f"ucmr3_common_c{label}", "ucmr3_native_to_buffer"),
                ("ucmr5_native", f"ucmr5_common_c{label}", "ucmr5_native_to_common_buffer"),
                (f"ucmr3_common_c{label}", f"ucmr5_common_c{label}", "common_threshold_cross_cycle"),
            ):
                before, after = counts[before_name], counts[after_name]
                table = np.zeros((7, 7), dtype=int)
                np.add.at(table, (before, after), 1)
                for left in range(7):
                    for right in range(7):
                        transition_rows.append(
                            {
                                "subset": subset,
                                "buffer": multiplier,
                                "transition": transition,
                                "count_before": left,
                                "count_after": right,
                                "pairs": int(table[left, right]),
                            }
                        )

        # Full bootstrap inference is intentionally restricted to the primary
        # identifier-matched cohort; corroborated subsets remain transparent
        # point-estimate sensitivity analyses.
        if subset == "identifier_matched":
            bootstrap = bootstrap_feature_estimands(
                feature,
                state,
                pws,
                args.bootstrap_reps,
                SEED + subset_index * 10_000,
                args.bootstrap_chunk_size,
            )
            inference_rows = []
            for bootstrap_estimand, draws in bootstrap.items():
                point_key = "event_equal" if bootstrap_estimand == "event_equal_state_cluster" else bootstrap_estimand
                point = points[point_key] @ contrast_matrix
                contrast_draws = draws @ contrast_matrix
                for column, spec in enumerate(contrast_specs):
                    inference_rows.append(
                        {
                            "subset": subset,
                            "estimand": bootstrap_estimand,
                            **spec,
                            "estimate": float(point[column]),
                            "estimate_pp": float(100 * point[column]),
                            "ci95_low": float(np.quantile(contrast_draws[:, column], 0.025)),
                            "ci95_high": float(np.quantile(contrast_draws[:, column], 0.975)),
                            "ci95_low_pp": float(100 * np.quantile(contrast_draws[:, column], 0.025)),
                            "ci95_high_pp": float(100 * np.quantile(contrast_draws[:, column], 0.975)),
                            "bootstrap_reps": args.bootstrap_reps,
                        }
                    )
            inference = pd.DataFrame(inference_rows)
            inference.to_csv(output / "endpoint_inference.csv", index=False)
            primary_payload = (feature, feature_index, counts, points, inference)

    pd.DataFrame(rate_rows).to_csv(output / "tail_rate_table.csv", index=False)
    pd.DataFrame(transition_rows).to_csv(output / "count_transition_table.csv", index=False)
    mrl_table(u3, u5).to_csv(output / "analyte_mrl_audit.csv", index=False)

    if primary_payload is None:
        raise RuntimeError("Primary cohort was not evaluated")
    _, feature_index, _, points, inference = primary_payload
    primary = inference[inference["estimand"].eq("state_pws_two_level")].copy()
    native = primary[primary["contrast"].eq("native_period")]
    common = primary[primary["contrast"].eq("common_threshold_period")]
    identity = []
    for estimand, feature_point in points.items():
        for multiplier in buffers:
            label = buffer_label(multiplier)
            for k in range(1, 7):
                native_value = feature_point @ (
                    tail_vector(feature_index, "ucmr5_native", k)
                    - tail_vector(feature_index, "ucmr3_native", k)
                )
                parts = (
                    feature_point @ (tail_vector(feature_index, "ucmr5_native", k) - tail_vector(feature_index, f"ucmr5_common_c{label}", k))
                    + feature_point @ (tail_vector(feature_index, f"ucmr5_common_c{label}", k) - tail_vector(feature_index, f"ucmr3_common_c{label}", k))
                    + feature_point @ (tail_vector(feature_index, f"ucmr3_common_c{label}", k) - tail_vector(feature_index, "ucmr3_native", k))
                )
                identity.append(
                    {
                        "estimand": estimand,
                        "buffer": multiplier,
                        "k": k,
                        "identity_residual": float(native_value - parts),
                    }
                )
    identity_frame = pd.DataFrame(identity)
    identity_frame.to_csv(output / "decomposition_identity_audit.csv", index=False)

    decision = {
        "status": "COUNT-SPECTRUM AND MRL-BUFFER ROBUSTNESS COMPLETE",
        "source": {
            "ucmr3": str(args.ucmr3_zip),
            "ucmr3_sha256": sha256(args.ucmr3_zip),
            "ucmr5": str(args.ucmr5_zip),
            "ucmr5_sha256": sha256(args.ucmr5_zip),
        },
        "cycle_audits": {"UCMR3": audit3, "UCMR5": audit5},
        "strict_event_key": bool(args.strict_event_key),
        "subsets": subset_summaries,
        "buffers": buffers,
        "endpoints": "Pr(K6 >= r), r=1,...,6",
        "primary_estimand": "state-equal mean of within-state PWS-equal means",
        "bootstrap": {"reps": args.bootstrap_reps, "seed": SEED},
        "native_state_pws_contrasts_pp": {
            str(int(row.k)): {
                "estimate": float(row.estimate_pp),
                "ci95": [float(row.ci95_low_pp), float(row.ci95_high_pp)],
            }
            for row in native.itertuples()
        },
        "common_threshold_state_pws_contrasts_pp": {
            buffer_label(multiplier): {
                str(int(row.k)): {
                    "estimate": float(row.estimate_pp),
                    "ci95": [float(row.ci95_low_pp), float(row.ci95_high_pp)],
                }
                for row in common[np.isclose(common["buffer"], multiplier)].itertuples()
            }
            for multiplier in buffers
        },
        "maximum_decomposition_identity_residual": float(
            identity_frame["identity_residual"].abs().max()
        ),
        "identification_boundary": (
            "Only thresholds at or above each UCMR3 analyte-specific MRL are evaluated; "
            "sub-MRL concentrations are not recovered or imputed. K>=4--6 estimates may be "
            "descriptively sparse and are retained to show the complete count spectrum."
        ),
    }
    (output / "decision.json").write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(decision, indent=2), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Locked Gate-A audit of PFAS mixture observability across UCMR cycles.

The analysis uses exact PWSID + FacilityID + SamplePointID matches with one
complete, unambiguous SE1 panel for the six PFAS shared by UCMR3 and UCMR5.
It performs no below-MRL imputation.  Three directly observed regimes are
contrasted:

* UCMR3 at its native (legacy) MRLs;
* UCMR5 at its native MRLs;
* UCMR5 detections re-thresholded at the UCMR3 MRLs.

The primary endpoint is the paired change in P(K >= 2).  State-cluster
bootstrap resamples whole state/primacy entities and recomputes nonlinear
mixture-assembly statistics within every draw.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from audit_ucmr_cross_cycle_observability import COMMON_PFAS, read_se1


REGIMES = ("ucmr3_native", "ucmr5_native", "ucmr5_at_ucmr3_mrl")


def quantiles(values: np.ndarray) -> tuple[float, float]:
    return tuple(float(value) for value in np.quantile(values, [0.025, 0.975]))


def poisson_binomial_tail(margins: np.ndarray, threshold: int) -> float:
    probability = np.zeros(len(margins) + 1, dtype=float)
    probability[0] = 1.0
    active = 0
    for margin in margins:
        active += 1
        previous = probability.copy()
        probability[: active + 1] = 0.0
        probability[:active] += previous[:active] * (1.0 - margin)
        probability[1 : active + 1] += previous[:active] * margin
    return float(probability[threshold:].sum())


def build_arrays(ucmr3: dict, ucmr5: dict) -> dict:
    matched = sorted(set(ucmr3) & set(ucmr5))
    if not matched:
        raise ValueError("No exact sampling-point matches")

    state = np.asarray(
        [ucmr5[key][COMMON_PFAS[0]]["state"] for key in matched], dtype=object
    )
    pws = np.asarray([key[0] for key in matched], dtype=object)
    stable_water = np.asarray(
        [
            ucmr3[key][COMMON_PFAS[0]]["water_type"]
            == ucmr5[key][COMMON_PFAS[0]]["water_type"]
            for key in matched
        ],
        dtype=bool,
    )

    y3 = np.zeros((len(matched), len(COMMON_PFAS)), dtype=np.int8)
    y5 = np.zeros_like(y3)
    y5_old = np.zeros_like(y3)
    old_mrl = np.zeros(len(COMMON_PFAS), dtype=float)
    new_mrl = np.zeros(len(COMMON_PFAS), dtype=float)
    for j, chemical in enumerate(COMMON_PFAS):
        old_values = {ucmr3[key][chemical]["mrl"] for key in matched}
        new_values = {ucmr5[key][chemical]["mrl"] for key in matched}
        if len(old_values) != 1 or len(new_values) != 1:
            raise ValueError(
                f"Nonconstant MRL for {chemical}: UCMR3={old_values}, UCMR5={new_values}"
            )
        old_mrl[j] = next(iter(old_values))
        new_mrl[j] = next(iter(new_values))
        for i, key in enumerate(matched):
            before = ucmr3[key][chemical]
            after = ucmr5[key][chemical]
            y3[i, j] = int(before["detected"])
            y5[i, j] = int(after["detected"])
            y5_old[i, j] = int(
                after["detected"] and after["value"] >= old_mrl[j]
            )

    return {
        "keys": matched,
        "state": state,
        "pws": pws,
        "stable_water": stable_water,
        "old_mrl": old_mrl,
        "new_mrl": new_mrl,
        "panels": {
            "ucmr3_native": y3,
            "ucmr5_native": y5,
            "ucmr5_at_ucmr3_mrl": y5_old,
        },
    }


def cluster_indices(labels: np.ndarray) -> tuple[list[str], list[np.ndarray]]:
    names = sorted(set(str(value) for value in labels))
    return names, [np.flatnonzero(labels == name) for name in names]


def aggregate_endpoint_by_state(
    state: np.ndarray, values: np.ndarray
) -> tuple[list[str], np.ndarray, np.ndarray]:
    names, indices = cluster_indices(state)
    totals = np.asarray([values[index].sum() for index in indices], dtype=float)
    counts = np.asarray([len(index) for index in indices], dtype=float)
    return names, totals, counts


def bootstrap_paired_mean(
    state: np.ndarray,
    values: np.ndarray,
    *,
    reps: int,
    seed: int,
) -> dict:
    names, totals, counts = aggregate_endpoint_by_state(state, values)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(names), size=(reps, len(names)))
    estimates = totals[draws].sum(axis=1) / counts[draws].sum(axis=1)
    low, high = quantiles(estimates)
    return {
        "estimate": float(values.mean()),
        "ci95_low": low,
        "ci95_high": high,
        "bootstrap_probability_positive": float(np.mean(estimates > 0)),
        "bootstrap_probability_negative": float(np.mean(estimates < 0)),
        "clusters": len(names),
        "observations": len(values),
        "reps": reps,
    }


def pws_equal_values(
    state: np.ndarray, pws: np.ndarray, site_values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for state_value, pws_value, value in zip(state, pws, site_values, strict=True):
        grouped[(str(state_value), str(pws_value))].append(float(value))
    states = []
    values = []
    for (state_value, _), group_values in sorted(grouped.items()):
        states.append(state_value)
        values.append(float(np.mean(group_values)))
    return np.asarray(states, dtype=object), np.asarray(values, dtype=float)


def bootstrap_ratio(
    state: np.ndarray,
    numerator: np.ndarray,
    denominator: np.ndarray,
    *,
    reps: int,
    seed: int,
) -> dict:
    names, indices = cluster_indices(state)
    numerator_totals = np.asarray(
        [numerator[index].sum() for index in indices], dtype=float
    )
    denominator_totals = np.asarray(
        [denominator[index].sum() for index in indices], dtype=float
    )
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(names), size=(reps, len(names)))
    draw_denominator = denominator_totals[draws].sum(axis=1)
    valid = draw_denominator > 0
    estimates = numerator_totals[draws].sum(axis=1)[valid] / draw_denominator[valid]
    low, high = quantiles(estimates)
    observed_denominator = float(denominator.sum())
    return {
        "numerator": int(numerator.sum()),
        "denominator": int(denominator.sum()),
        "estimate": float(numerator.sum() / observed_denominator),
        "ci95_low": low,
        "ci95_high": high,
        "clusters": len(names),
        "valid_reps": int(valid.sum()),
        "reps": reps,
    }


def sufficient_statistics(
    state: np.ndarray, panels: dict[str, np.ndarray]
) -> tuple[list[str], dict[str, dict[str, np.ndarray]]]:
    names, indices = cluster_indices(state)
    pairs = list(itertools.combinations(range(len(COMMON_PFAS)), 2))
    result = {}
    for regime, panel in panels.items():
        count = panel.sum(axis=1)
        result[regime] = {
            "n": np.asarray([len(index) for index in indices], dtype=float),
            "marginal": np.stack(
                [panel[index].sum(axis=0) for index in indices]
            ).astype(float),
            "histogram": np.stack(
                [
                    np.bincount(count[index], minlength=len(COMMON_PFAS) + 1)
                    for index in indices
                ]
            ).astype(float),
            "pair": np.stack(
                [
                    np.asarray(
                        [
                            np.sum(panel[index, a] * panel[index, b])
                            for a, b in pairs
                        ]
                    )
                    for index in indices
                ]
            ).astype(float),
        }
    return names, result


def metrics_from_sufficient(stats: dict[str, np.ndarray]) -> dict:
    n = float(stats["n"].sum())
    marginal = stats["marginal"].sum(axis=0) / n
    histogram = stats["histogram"].sum(axis=0) / n
    pair = stats["pair"].sum(axis=0) / n
    independent_tail = {
        str(k): poisson_binomial_tail(marginal, k) for k in (1, 2, 3)
    }
    joint_tail = {str(k): float(histogram[k:].sum()) for k in (1, 2, 3)}
    assembly = {
        str(k): joint_tail[str(k)] - independent_tail[str(k)] for k in (1, 2, 3)
    }
    pairs = list(itertools.combinations(range(len(COMMON_PFAS)), 2))
    pair_excess = np.asarray(
        [pair[i] - marginal[a] * marginal[b] for i, (a, b) in enumerate(pairs)]
    )
    return {
        "n": int(n),
        "marginal": marginal,
        "histogram": histogram,
        "pair_joint": pair,
        "pair_excess": pair_excess,
        "joint_tail": joint_tail,
        "independent_tail": independent_tail,
        "assembly": assembly,
    }


def subset_stats(
    sufficient: dict[str, dict[str, np.ndarray]], draw: np.ndarray | None = None
) -> dict[str, dict]:
    metrics = {}
    for regime, stats in sufficient.items():
        selected = stats if draw is None else {key: value[draw] for key, value in stats.items()}
        metrics[regime] = metrics_from_sufficient(selected)
    return metrics


def bootstrap_nonlinear_metrics(
    state: np.ndarray,
    panels: dict[str, np.ndarray],
    *,
    reps: int,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    names, sufficient = sufficient_statistics(state, panels)
    observed = subset_stats(sufficient)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(names), size=(reps, len(names)))

    assembly_values = {
        (regime, k): np.empty(reps, dtype=float)
        for regime in REGIMES
        for k in (1, 2, 3)
    }
    pair_joint_values = {
        (regime, i): np.empty(reps, dtype=float)
        for regime in REGIMES
        for i in range(15)
    }
    pair_excess_values = {
        (regime, i): np.empty(reps, dtype=float)
        for regime in REGIMES
        for i in range(15)
    }
    for draw_index, draw in enumerate(draws):
        result = subset_stats(sufficient, draw)
        for regime in REGIMES:
            for k in (1, 2, 3):
                assembly_values[(regime, k)][draw_index] = result[regime][
                    "assembly"
                ][str(k)]
            for i in range(15):
                pair_joint_values[(regime, i)][draw_index] = result[regime][
                    "pair_joint"
                ][i]
                pair_excess_values[(regime, i)][draw_index] = result[regime][
                    "pair_excess"
                ][i]

    assembly_rows = []
    for k in (1, 2, 3):
        for regime in REGIMES:
            values = assembly_values[(regime, k)]
            low, high = quantiles(values)
            assembly_rows.append(
                {
                    "metric": f"MAS_K_GE_{k}",
                    "contrast": regime,
                    "estimate": observed[regime]["assembly"][str(k)],
                    "ci95_low": low,
                    "ci95_high": high,
                    "clusters": len(names),
                    "reps": reps,
                }
            )
        for contrast, newer, older in (
            ("ucmr5_native_minus_ucmr3_native", "ucmr5_native", "ucmr3_native"),
            (
                "ucmr5_at_ucmr3_mrl_minus_ucmr3_native",
                "ucmr5_at_ucmr3_mrl",
                "ucmr3_native",
            ),
        ):
            values = assembly_values[(newer, k)] - assembly_values[(older, k)]
            low, high = quantiles(values)
            assembly_rows.append(
                {
                    "metric": f"MAS_K_GE_{k}",
                    "contrast": contrast,
                    "estimate": observed[newer]["assembly"][str(k)]
                    - observed[older]["assembly"][str(k)],
                    "ci95_low": low,
                    "ci95_high": high,
                    "clusters": len(names),
                    "reps": reps,
                }
            )

    pair_rows = []
    pairs = list(itertools.combinations(range(len(COMMON_PFAS)), 2))
    for i, (a, b) in enumerate(pairs):
        for metric_name, values_by_regime, observed_key in (
            ("joint_rate", pair_joint_values, "pair_joint"),
            ("excess_over_same_margin_independence", pair_excess_values, "pair_excess"),
        ):
            for contrast, newer, older in (
                ("ucmr5_native_minus_ucmr3_native", "ucmr5_native", "ucmr3_native"),
                (
                    "ucmr5_at_ucmr3_mrl_minus_ucmr3_native",
                    "ucmr5_at_ucmr3_mrl",
                    "ucmr3_native",
                ),
            ):
                values = values_by_regime[(newer, i)] - values_by_regime[(older, i)]
                low, high = quantiles(values)
                pair_rows.append(
                    {
                        "chemical_a": COMMON_PFAS[a],
                        "chemical_b": COMMON_PFAS[b],
                        "metric": metric_name,
                        "contrast": contrast,
                        "estimate": float(
                            observed[newer][observed_key][i]
                            - observed[older][observed_key][i]
                        ),
                        "ci95_low": low,
                        "ci95_high": high,
                        "clusters": len(names),
                        "reps": reps,
                    }
                )
    return assembly_rows, pair_rows


def state_conditional_assembly(
    state: np.ndarray,
    panels: dict[str, np.ndarray],
    *,
    reps: int,
    seed: int,
) -> list[dict]:
    """Average within-state MAS so between-state prevalence cannot create it."""
    names, indices = cluster_indices(state)
    n_by_state = np.asarray([len(index) for index in indices], dtype=float)
    values: dict[tuple[str, int], np.ndarray] = {}
    for regime, panel in panels.items():
        count = panel.sum(axis=1)
        for k in (1, 2, 3):
            state_values = []
            for index in indices:
                margins = panel[index].mean(axis=0)
                observed_tail = float(np.mean(count[index] >= k))
                independent_tail = poisson_binomial_tail(margins, k)
                state_values.append(observed_tail - independent_tail)
            values[(regime, k)] = np.asarray(state_values, dtype=float)

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(names), size=(reps, len(names)))
    rows = []
    for k in (1, 2, 3):
        for weighting in ("state_equal", "site_weighted_within_state"):
            for contrast, newer, older in (
                ("ucmr5_native_minus_ucmr3_native", "ucmr5_native", "ucmr3_native"),
                (
                    "ucmr5_at_ucmr3_mrl_minus_ucmr3_native",
                    "ucmr5_at_ucmr3_mrl",
                    "ucmr3_native",
                ),
            ):
                difference = values[(newer, k)] - values[(older, k)]
                if weighting == "state_equal":
                    estimate = float(difference.mean())
                    estimates = difference[draws].mean(axis=1)
                else:
                    estimate = float(np.average(difference, weights=n_by_state))
                    estimates = (
                        (difference[draws] * n_by_state[draws]).sum(axis=1)
                        / n_by_state[draws].sum(axis=1)
                    )
                low, high = quantiles(estimates)
                rows.append(
                    {
                        "metric": f"WITHIN_STATE_MAS_K_GE_{k}",
                        "weighting": weighting,
                        "contrast": contrast,
                        "estimate": estimate,
                        "ci95_low": low,
                        "ci95_high": high,
                        "bootstrap_probability_positive": float(
                            np.mean(estimates > 0)
                        ),
                        "bootstrap_probability_negative": float(
                            np.mean(estimates < 0)
                        ),
                        "clusters": len(names),
                        "reps": reps,
                    }
                )
    return rows


def descriptive_rows(panels: dict[str, np.ndarray], cohort: str) -> list[dict]:
    rows = []
    for regime, panel in panels.items():
        counts = panel.sum(axis=1)
        histogram = np.bincount(counts, minlength=len(COMMON_PFAS) + 1)
        margins = panel.mean(axis=0)
        for k, count in enumerate(histogram):
            rows.append(
                {
                    "cohort": cohort,
                    "regime": regime,
                    "metric": f"K_EQ_{k}",
                    "count": int(count),
                    "rate": float(count / len(panel)),
                }
            )
        for k in (1, 2, 3):
            rows.append(
                {
                    "cohort": cohort,
                    "regime": regime,
                    "metric": f"K_GE_{k}",
                    "count": int(np.sum(counts >= k)),
                    "rate": float(np.mean(counts >= k)),
                }
            )
            independent = poisson_binomial_tail(margins, k)
            joint = float(np.mean(counts >= k))
            rows.append(
                {
                    "cohort": cohort,
                    "regime": regime,
                    "metric": f"INDEPENDENT_K_GE_{k}",
                    "count": "",
                    "rate": independent,
                }
            )
            rows.append(
                {
                    "cohort": cohort,
                    "regime": regime,
                    "metric": f"MAS_K_GE_{k}",
                    "count": "",
                    "rate": joint - independent,
                }
            )
        rows.append(
            {
                "cohort": cohort,
                "regime": regime,
                "metric": "MEAN_K",
                "count": "",
                "rate": float(counts.mean()),
            }
        )
    return rows


def paired_endpoint_rows(
    state: np.ndarray,
    pws: np.ndarray,
    panels: dict[str, np.ndarray],
    *,
    cohort: str,
    reps: int,
    seed: int,
) -> list[dict]:
    rows = []
    counts = {regime: panel.sum(axis=1) for regime, panel in panels.items()}
    endpoint_index = 0
    for endpoint in ("K_GE_1", "K_GE_2", "K_GE_3", "MEAN_K"):
        if endpoint == "MEAN_K":
            endpoint_values = {regime: value.astype(float) for regime, value in counts.items()}
        else:
            k = int(endpoint[-1])
            endpoint_values = {
                regime: (value >= k).astype(float) for regime, value in counts.items()
            }
        for contrast, newer, older in (
            ("ucmr5_native_minus_ucmr3_native", "ucmr5_native", "ucmr3_native"),
            (
                "ucmr5_at_ucmr3_mrl_minus_ucmr3_native",
                "ucmr5_at_ucmr3_mrl",
                "ucmr3_native",
            ),
        ):
            values = endpoint_values[newer] - endpoint_values[older]
            result = bootstrap_paired_mean(
                state, values, reps=reps, seed=seed + endpoint_index
            )
            result.update(
                {
                    "cohort": cohort,
                    "weighting": "site_equal",
                    "metric": endpoint,
                    "contrast": contrast,
                }
            )
            rows.append(result)

            pws_state, pws_values = pws_equal_values(state, pws, values)
            pws_result = bootstrap_paired_mean(
                pws_state,
                pws_values,
                reps=reps,
                seed=seed + 1000 + endpoint_index,
            )
            pws_result.update(
                {
                    "cohort": cohort,
                    "weighting": "pws_equal_mean_of_site_endpoints",
                    "metric": endpoint,
                    "contrast": contrast,
                }
            )
            rows.append(pws_result)
            endpoint_index += 1
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def transition_rows(panels: dict[str, np.ndarray], cohort: str) -> list[dict]:
    rows = []
    count3 = panels["ucmr3_native"].sum(axis=1)
    for target_regime in ("ucmr5_native", "ucmr5_at_ucmr3_mrl"):
        count5 = panels[target_regime].sum(axis=1)
        matrix = np.zeros(
            (len(COMMON_PFAS) + 1, len(COMMON_PFAS) + 1), dtype=int
        )
        for before, after in zip(count3, count5, strict=True):
            matrix[int(before), int(after)] += 1
        for before in range(len(COMMON_PFAS) + 1):
            for after in range(len(COMMON_PFAS) + 1):
                rows.append(
                    {
                        "cohort": cohort,
                        "target_regime": target_regime,
                        "ucmr3_k": before,
                        "ucmr5_k": after,
                        "count": int(matrix[before, after]),
                        "rate": float(matrix[before, after] / len(count3)),
                    }
                )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ucmr3-zip", type=Path, required=True)
    parser.add_argument("--ucmr5-zip", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reps", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260812)
    args = parser.parse_args()

    ucmr3, audit3 = read_se1(args.ucmr3_zip, "UCMR3")
    ucmr5, audit5 = read_se1(args.ucmr5_zip, "UCMR5")
    data = build_arrays(ucmr3, ucmr5)
    panels = data["panels"]
    state = data["state"]
    pws = data["pws"]
    stable = data["stable_water"]

    descriptive = descriptive_rows(panels, "all_exact_matched_sites")
    stable_panels = {regime: panel[stable] for regime, panel in panels.items()}
    descriptive.extend(descriptive_rows(stable_panels, "stable_water_type_sites"))

    paired = paired_endpoint_rows(
        state,
        pws,
        panels,
        cohort="all_exact_matched_sites",
        reps=args.reps,
        seed=args.seed,
    )
    paired.extend(
        paired_endpoint_rows(
            state[stable],
            pws[stable],
            stable_panels,
            cohort="stable_water_type_sites",
            reps=args.reps,
            seed=args.seed + 10000,
        )
    )

    counts5 = panels["ucmr5_native"].sum(axis=1)
    counts5_old = panels["ucmr5_at_ucmr3_mrl"].sum(axis=1)
    enabled = {}
    for k in (1, 2, 3):
        denominator = counts5 >= k
        numerator = denominator & (counts5_old < k)
        enabled[f"K_GE_{k}"] = bootstrap_ratio(
            state,
            numerator.astype(float),
            denominator.astype(float),
            reps=args.reps,
            seed=args.seed + 20000 + k,
        )
        stable_denominator = denominator[stable]
        stable_numerator = numerator[stable]
        enabled[f"K_GE_{k}"]["stable_water_type"] = bootstrap_ratio(
            state[stable],
            stable_numerator.astype(float),
            stable_denominator.astype(float),
            reps=args.reps,
            seed=args.seed + 21000 + k,
        )
        counts3 = panels["ucmr3_native"].sum(axis=1)
        apparent_new_denominator = (counts3 < k) & (counts5 >= k)
        apparent_new_numerator = apparent_new_denominator & (counts5_old < k)
        enabled[f"K_GE_{k}"]["among_apparent_new_vs_ucmr3"] = bootstrap_ratio(
            state,
            apparent_new_numerator.astype(float),
            apparent_new_denominator.astype(float),
            reps=args.reps,
            seed=args.seed + 22000 + k,
        )

    assembly, pair = bootstrap_nonlinear_metrics(
        state, panels, reps=args.reps, seed=args.seed + 30000
    )
    conditional_assembly = state_conditional_assembly(
        state, panels, reps=args.reps, seed=args.seed + 40000
    )

    def paired_result(cohort: str, weighting: str, metric: str, contrast: str) -> dict:
        matches = [
            row
            for row in paired
            if row["cohort"] == cohort
            and row["weighting"] == weighting
            and row["metric"] == metric
            and row["contrast"] == contrast
        ]
        if len(matches) != 1:
            raise ValueError(f"Expected one paired result, found {len(matches)}")
        return matches[0]

    primary_native = paired_result(
        "all_exact_matched_sites",
        "site_equal",
        "K_GE_2",
        "ucmr5_native_minus_ucmr3_native",
    )
    primary_old = paired_result(
        "all_exact_matched_sites",
        "site_equal",
        "K_GE_2",
        "ucmr5_at_ucmr3_mrl_minus_ucmr3_native",
    )
    stable_native = paired_result(
        "stable_water_type_sites",
        "site_equal",
        "K_GE_2",
        "ucmr5_native_minus_ucmr3_native",
    )
    stable_old = paired_result(
        "stable_water_type_sites",
        "site_equal",
        "K_GE_2",
        "ucmr5_at_ucmr3_mrl_minus_ucmr3_native",
    )
    pws_native = paired_result(
        "all_exact_matched_sites",
        "pws_equal_mean_of_site_endpoints",
        "K_GE_2",
        "ucmr5_native_minus_ucmr3_native",
    )
    pws_old = paired_result(
        "all_exact_matched_sites",
        "pws_equal_mean_of_site_endpoints",
        "K_GE_2",
        "ucmr5_at_ucmr3_mrl_minus_ucmr3_native",
    )

    criteria = {
        "native_k_ge_2_state_ci_low_positive": primary_native["ci95_low"] > 0,
        "old_mrl_k_ge_2_state_ci_high_negative": primary_old["ci95_high"] < 0,
        "at_least_80pct_ucmr5_native_mixture_sites_enabled_by_lower_mrl": enabled[
            "K_GE_2"
        ]["estimate"]
        >= 0.80,
        "stable_water_type_preserves_directions": stable_native["estimate"] > 0
        and stable_old["estimate"] < 0,
        "pws_equal_preserves_directions": pws_native["estimate"] > 0
        and pws_old["estimate"] < 0,
    }
    decision = {
        "status": "MIXTURE OBSERVABILITY GATE-A GO"
        if all(criteria.values())
        else "MIXTURE OBSERVABILITY GATE-A NO-GO",
        "evidence_class": (
            "Directly identified exact-sampling-point cross-cycle audit; no below-MRL imputation."
        ),
        "criteria": criteria,
        "primary_k_ge_2": {
            "native_cycle_specific_mrl": primary_native,
            "ucmr3_mrl_standardized": primary_old,
            "measurement_enabled_fraction": enabled["K_GE_2"],
            "stable_water_type_native": stable_native,
            "stable_water_type_old_mrl": stable_old,
            "pws_equal_native": pws_native,
            "pws_equal_old_mrl": pws_old,
        },
        "cohort": {
            "exact_matched_sites": len(state),
            "matched_pws": len(set(pws)),
            "states": len(set(state)),
            "stable_water_type_sites": int(stable.sum()),
        },
        "mrl_ug_l": {
            chemical: {
                "ucmr3": float(data["old_mrl"][j]),
                "ucmr5": float(data["new_mrl"][j]),
            }
            for j, chemical in enumerate(COMMON_PFAS)
        },
        "cycle_audits": {"UCMR3": dict(audit3), "UCMR5": dict(audit5)},
        "bootstrap": {"reps": args.reps, "seed": args.seed},
        "interpretation_boundary": (
            "The standardized contrast identifies changes at or above UCMR3 MRLs. "
            "It does not identify UCMR3 mixture occurrence below those MRLs."
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "count_distribution.csv", descriptive)
    transitions = transition_rows(panels, "all_exact_matched_sites")
    transitions.extend(
        transition_rows(stable_panels, "stable_water_type_sites")
    )
    write_csv(args.output_dir / "count_transition_matrix.csv", transitions)
    write_csv(args.output_dir / "paired_endpoint_bootstrap.csv", paired)
    write_csv(args.output_dir / "mixture_assembly_bootstrap.csv", assembly)
    write_csv(
        args.output_dir / "state_conditional_mixture_assembly_bootstrap.csv",
        conditional_assembly,
    )
    write_csv(args.output_dir / "pairwise_bootstrap.csv", pair)
    (args.output_dir / "decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()

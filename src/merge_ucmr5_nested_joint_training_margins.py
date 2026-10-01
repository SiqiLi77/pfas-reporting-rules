#!/usr/bin/env python3
"""Merge and audit per-outer-fold nested marginal predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


CHEMICALS = ("PFBS", "PFHpA", "PFHxS", "PFNA", "PFOA", "PFOS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--input-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    cohort = pd.read_csv(
        args.cohort,
        usecols=["location_id", "state_fold"],
        dtype={"location_id": str},
    )
    if cohort["location_id"].duplicated().any():
        raise ValueError("Cohort contains duplicate location_id values")
    fold_by_location = cohort.set_index("location_id")["state_fold"]

    parts = []
    run_infos = []
    for directory in args.input_dirs:
        parts.append(
            pd.read_csv(
                directory / "nested_joint_training_margins.csv.gz",
                dtype={"location_id": str, "pws_id": str, "state": str},
            )
        )
        run_infos.append(json.loads((directory / "run_info.json").read_text()))
    predictions = pd.concat(parts, ignore_index=True)

    key = ["outer_joint_fold", "location_id", "chemical"]
    if predictions.duplicated(key).any():
        raise ValueError("Duplicate outer-fold/location/analyte nested prediction keys")
    outer_folds = sorted(predictions["outer_joint_fold"].unique().astype(int).tolist())
    if outer_folds != [1, 2, 3, 4]:
        raise ValueError(f"Expected outer joint folds 1..4, found {outer_folds}")
    if set(predictions["chemical"]) != set(CHEMICALS):
        raise ValueError("Nested predictions do not contain exactly the six shared PFAS")

    mapped_fold = predictions["location_id"].map(fold_by_location)
    if mapped_fold.isna().any():
        raise ValueError("Nested predictions contain locations absent from the cohort")
    if not mapped_fold.astype(int).eq(predictions["inner_target_fold"].astype(int)).all():
        raise ValueError("inner_target_fold disagrees with the frozen cohort fold")

    coverage = []
    for outer_fold in outer_folds:
        validation_fold = 1 + (outer_fold % 4)
        expected_folds = sorted(set(range(5)) - {outer_fold, validation_fold})
        subset = predictions[predictions["outer_joint_fold"].eq(outer_fold)]
        found_folds = sorted(subset["inner_target_fold"].unique().astype(int).tolist())
        if found_folds != expected_folds:
            raise ValueError(
                f"Outer fold {outer_fold}: expected target folds {expected_folds}, found {found_folds}"
            )
        expected_locations = cohort[cohort["state_fold"].isin(expected_folds)][
            "location_id"
        ]
        expected_rows = len(expected_locations) * len(CHEMICALS)
        if len(subset) != expected_rows:
            raise ValueError(
                f"Outer fold {outer_fold}: expected {expected_rows} rows, found {len(subset)}"
            )
        if subset.groupby("location_id")["chemical"].nunique().ne(len(CHEMICALS)).any():
            raise ValueError(f"Outer fold {outer_fold}: incomplete analyte panel")
        coverage.append(
            {
                "outer_joint_fold": outer_fold,
                "joint_validation_fold": validation_fold,
                "inner_target_folds": expected_folds,
                "predicted_sites": len(expected_locations),
                "prediction_rows": len(subset),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions.sort_values(key).to_csv(
        args.output_dir / "nested_joint_training_margins.csv.gz",
        index=False,
        compression="gzip",
    )
    training = [
        record
        for info in run_infos
        for record in info.get("training", [])
    ]
    training.sort(
        key=lambda row: (int(row["outer_joint_fold"]), int(row["inner_target_fold"]))
    )
    payload = {
        "status": "NESTED MARGINS MERGED AND AUDITED",
        "design": (
            "Outer-test-clean inner cross-fitting. For each joint outer fold, its test "
            "fold and the validation fold fixed for the reported analysis are absent from marginal-model "
            "training; each predicted target fold is also absent from training."
        ),
        "cohort": str(args.cohort),
        "cohort_sites": len(cohort),
        "outer_folds": outer_folds,
        "prediction_rows": len(predictions),
        "coverage": coverage,
        "training": training,
        "source_run_info": run_infos,
    }
    (args.output_dir / "run_info.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: payload[key] for key in ("status", "cohort_sites", "prediction_rows", "coverage")}, indent=2))


if __name__ == "__main__":
    main()

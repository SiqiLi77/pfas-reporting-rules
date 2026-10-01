#!/usr/bin/env python3
"""Merge independently trained censored-distribution outer-fold outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from run_ucmr5_censored_distribution_screen import summarize


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--design", required=True)
    args = parser.parse_args()

    predictions = pd.concat(
        [
            pd.read_csv(
                path / "out_of_fold_distribution_predictions.csv.gz",
                dtype={"location_id": str, "pws_id": str, "state": str},
            )
            for path in args.input_dirs
        ],
        ignore_index=True,
    )
    run_infos = [
        json.loads((path / "run_info.json").read_text()) for path in args.input_dirs
    ]
    key = ["model", "location_id", "chemical"]
    if predictions.duplicated(key).any():
        raise ValueError("Duplicate model/location/chemical keys across folds")
    folds = sorted(predictions["outer_fold"].unique().astype(int).tolist())
    if folds != [0, 1, 2, 3, 4]:
        raise ValueError(f"Expected outer folds 0..4, found {folds}")
    metrics, macro = summarize(predictions)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(
        args.output_dir / "out_of_fold_distribution_predictions.csv.gz",
        index=False,
        compression="gzip",
    )
    metrics.to_csv(args.output_dir / "per_analyte_distribution_metrics.csv", index=False)
    macro.to_csv(args.output_dir / "macro_distribution_metrics.csv", index=False)
    payload = {
        "design": args.design,
        "folds": folds,
        "sites": int(predictions["location_id"].nunique()),
        "models": sorted(predictions["model"].unique().tolist()),
        "source_run_info": run_infos,
        "macro_metrics": macro.to_dict(orient="records"),
    }
    (args.output_dir / "run_info.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(macro.to_string(index=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Outer-test-clean, inner-cross-fitted margins for conditional joint fitting.

For each joint-model outer fold, this script predicts every site used to fit the
dependence model with a marginal model that saw neither that site's state fold
nor the joint outer-test fold. The validation fold fixed for the reported analysis is used
only for neural early stopping and Platt calibration.  The remaining two folds
train the neural model.  These margins are intentionally separate from the
ordinary five-fold OOF predictions used to score the outer test sites.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from run_ucmr5_artificial_censor_screen import (
    CHEMICALS,
    fit_platt,
    probability_to_logit,
)
from run_ucmr5_censored_distribution_screen import (
    distribution_summaries,
    moment_match,
    target_arrays,
)
from run_ucmr5_echo_hurdle_ablation import train_one
from run_ucmr5_echo_source_ablation import attach_source_features


DEFAULT_TRAINING_MODEL = "panel_hurdle_echo_strict"
DEFAULT_OUTPUT_MODEL = "panel_hurdle_tri2022radial_strict_nested"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--source-features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--outer-folds", default="1,2,3,4")
    parser.add_argument("--seeds", default="20260810,20260811,20260812")
    parser.add_argument(
        "--training-model",
        default=DEFAULT_TRAINING_MODEL,
        help=(
            "Hurdle-model family passed to train_one. Use panel_hurdle_echo_null "
            "for the source-null/no-TRI analysis."
        ),
    )
    parser.add_argument(
        "--output-model",
        default=DEFAULT_OUTPUT_MODEL,
        help="Model label stored in the nested marginal prediction table.",
    )
    args = parser.parse_args()

    torch.set_num_threads(min(8, max(1, torch.get_num_threads())))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cohort = pd.read_csv(
        args.cohort, dtype={"state": str, "region": str, "pws_id": str}
    )
    frame = attach_source_features(cohort, args.source_features)
    outer_folds = [int(value) for value in args.outer_folds.split(",") if value]
    seeds = [int(value) for value in args.seeds.split(",") if value]
    all_folds = sorted(frame["state_fold"].unique().astype(int))
    prediction_parts = []
    training_records = []

    for outer_fold in outer_folds:
        joint_validation_fold = 1 + (outer_fold % 4)
        target_folds = [
            fold
            for fold in all_folds
            if fold not in {outer_fold, joint_validation_fold}
        ]
        for target_fold in target_folds:
            training_folds = [
                fold
                for fold in all_folds
                if fold not in {outer_fold, joint_validation_fold, target_fold}
            ]
            train = frame[frame["state_fold"].isin(training_folds)].copy()
            validation = frame[frame["state_fold"].eq(joint_validation_fold)].copy()
            target = frame[frame["state_fold"].eq(target_fold)].copy()
            validation_targets = target_arrays(validation)
            target_targets = target_arrays(target)
            val_probabilities, val_mus, val_sigmas = [], [], []
            target_probabilities, target_mus, target_sigmas = [], [], []
            runs = []
            for seed in seeds:
                values = train_one(
                    args.training_model, train, validation, target, seed, device
                )
                (
                    val_p,
                    val_mu,
                    val_sigma,
                    target_p,
                    target_mu,
                    target_sigma,
                    external_values,
                    info,
                ) = values
                if external_values is not None:
                    raise AssertionError(
                        "Nested training unexpectedly received external predictions"
                    )
                val_probabilities.append(val_p)
                val_mus.append(val_mu)
                val_sigmas.append(val_sigma)
                target_probabilities.append(target_p)
                target_mus.append(target_mu)
                target_sigmas.append(target_sigma)
                runs.append(info)

            validation_probability = np.mean(val_probabilities, axis=0)
            target_probability_raw = np.mean(target_probabilities, axis=0)
            target_mu, target_sigma = moment_match(target_mus, target_sigmas)
            calibrated_probability = np.zeros_like(target_probability_raw)
            for chemical_index in range(len(CHEMICALS)):
                risk = validation_targets[1][:, chemical_index].astype(bool)
                calibrator = fit_platt(
                    probability_to_logit(validation_probability[risk, chemical_index]),
                    validation_targets[0][risk, chemical_index].astype(int),
                )
                calibrated_probability[:, chemical_index] = calibrator(
                    probability_to_logit(target_probability_raw[:, chemical_index])
                )
            _, conditional_mean, q05, q95, log_interval = distribution_summaries(
                target_mu, target_sigma, target_targets[3], target_targets[4]
            )
            for chemical_index, chemical in enumerate(CHEMICALS):
                prediction_parts.append(
                    pd.DataFrame(
                        {
                            "location_id": target["location_id"].astype(str).to_numpy(),
                            "pws_id": target["pws_id"].astype(str).to_numpy(),
                            "state": target["state"].astype(str).to_numpy(),
                            "outer_joint_fold": outer_fold,
                            "inner_target_fold": target_fold,
                            "joint_validation_fold": joint_validation_fold,
                            "model": args.output_model,
                            "chemical": chemical,
                            "risk": target_targets[1][:, chemical_index].astype(int),
                            "y_hidden": (
                                target_targets[0][:, chemical_index]
                                * target_targets[1][:, chemical_index]
                            ).astype(int),
                            "mu": target_mu[:, chemical_index],
                            "sigma": target_sigma[:, chemical_index],
                            "probability_hidden_raw": target_probability_raw[:, chemical_index],
                            "probability_hidden": calibrated_probability[:, chemical_index],
                            "conditional_mean_log_concentration": conditional_mean[:, chemical_index],
                            "q05_log_concentration": q05[:, chemical_index],
                            "q95_log_concentration": q95[:, chemical_index],
                            "hurdle_log_interval": log_interval[:, chemical_index],
                        }
                    )
                )
            training_records.append(
                {
                    "outer_joint_fold": outer_fold,
                    "joint_validation_fold": joint_validation_fold,
                    "inner_target_fold": int(target_fold),
                    "neural_training_folds": [int(fold) for fold in training_folds],
                    "outer_test_fold_excluded": True,
                    "target_fold_excluded": True,
                    "train_sites": len(train),
                    "validation_sites": len(validation),
                    "predicted_sites": len(target),
                    "runs": runs,
                }
            )
            print(
                f"outer={outer_fold} target={target_fold} train={training_folds} complete",
                flush=True,
            )

    predictions = pd.concat(prediction_parts, ignore_index=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(
        args.output_dir / "nested_joint_training_margins.csv.gz",
        index=False,
        compression="gzip",
    )
    run_info = {
        "design": (
            "Outer-test-clean inner cross-fitting for the sitewise margins used to fit "
            "conditional joint models. Each target fold and the joint outer-test fold "
            "are excluded from neural training; the validation fold fixed for the reported analysis "
            "is restricted to early stopping and Platt calibration."
        ),
        "device": str(device),
        "model": args.output_model,
        "training_model": args.training_model,
        "seeds": seeds,
        "outer_folds": outer_folds,
        "prediction_rows": len(predictions),
        "training": training_records,
    }
    (args.output_dir / "run_info.json").write_text(
        json.dumps(run_info, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

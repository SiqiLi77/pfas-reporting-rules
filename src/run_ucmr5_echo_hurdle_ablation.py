#!/usr/bin/env python3
"""Source-gated hurdle model for legacy-MRL hidden PFAS concentrations.

The hurdle probability models whether concentration falls in [native MRL,
legacy MRL), conditional on being below the legacy MRL. A separate truncated
normal head models concentration within that interval. This preserves coherent
threshold probabilities while allowing source geography to improve occurrence
ranking without forcing the same shift into conditional concentration.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import log_ndtr
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from run_ucmr5_artificial_censor_screen import (
    BASE_NUMERIC,
    CAT_COLUMNS,
    CHEMICALS,
    fit_platt,
    probability_to_logit,
    set_seed,
)
from run_ucmr5_censored_distribution_screen import (
    distribution_summaries,
    log_difference,
    moment_match,
    summarize,
    target_arrays,
)
from run_ucmr5_echo_source_ablation import (
    SourceTokenPreprocessor,
    attach_source_features,
    model_bundle,
    source_columns,
)


MODELS = {
    "panel_hurdle_echo_null",
    "panel_hurdle_echo_strict",
    "panel_hurdle_echo_broad",
    "panel_hurdle_echo_strictnull",
}


class SourceGatedHurdleTransformer(nn.Module):
    def __init__(
        self,
        category_sizes: list[int],
        source_dim: int,
        mu_offset: np.ndarray,
        occurrence_prevalence: np.ndarray,
        d_model: int = 64,
    ):
        super().__init__()
        self.register_buffer("mu_offset", torch.from_numpy(mu_offset.astype(np.float32)))
        self.chemical = nn.Embedding(len(CHEMICALS), d_model)
        self.token_projection = nn.Linear(2, d_model)
        self.category_embeddings = nn.ModuleList(
            [nn.Embedding(size + 1, 8) for size in category_sizes]
        )
        context_dim = len(BASE_NUMERIC) + 8 * len(category_sizes)
        self.context = nn.Sequential(
            nn.Linear(context_dim, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.source_encoder = nn.Sequential(
            nn.Linear(source_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, d_model),
        )
        self.source_gate = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.Sigmoid())
        nn.init.zeros_(self.source_encoder[-1].weight)
        nn.init.zeros_(self.source_encoder[-1].bias)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=128,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.concentration_output = nn.Linear(d_model, 2)
        nn.init.zeros_(self.concentration_output.weight)
        nn.init.zeros_(self.concentration_output.bias)
        self.occurrence_output = nn.Linear(d_model, 1, bias=False)
        nn.init.zeros_(self.occurrence_output.weight)
        prevalence = np.clip(occurrence_prevalence, 1e-4, 1 - 1e-4)
        logits = np.log(prevalence / (1 - prevalence)).astype(np.float32)
        self.occurrence_bias = nn.Parameter(torch.from_numpy(logits))

    def forward(
        self,
        token: torch.Tensor,
        context: torch.Tensor,
        categories: torch.Tensor,
        source: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        chemical_ids = torch.arange(len(CHEMICALS), device=token.device)
        hidden = self.token_projection(token) + self.chemical(chemical_ids)[None, :, :]
        embedded = [
            embedding(categories[:, index])
            for index, embedding in enumerate(self.category_embeddings)
        ]
        context_hidden = self.context(torch.cat([context] + embedded, dim=1))
        hidden = self.encoder(hidden + context_hidden[:, None, :])
        concentration = self.concentration_output(hidden)

        source_hidden = self.source_encoder(source)
        gate = self.source_gate(torch.cat([context_hidden, source_hidden], dim=1))
        occurrence_hidden = hidden + (gate * source_hidden)[:, None, :]
        occurrence_logit = (
            self.occurrence_output(occurrence_hidden).squeeze(-1)
            + self.occurrence_bias[None, :]
        )
        mu = self.mu_offset[None, :] + concentration[:, :, 0]
        return occurrence_logit, mu, concentration[:, :, 1]


def hurdle_nll(
    occurrence_logit: torch.Tensor,
    mu: torch.Tensor,
    raw_sigma: torch.Tensor,
    detected: torch.Tensor,
    risk: torch.Tensor,
    z: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> torch.Tensor:
    sigma = 0.05 + nn.functional.softplus(raw_sigma)
    log_cdf_upper = torch.special.log_ndtr((upper - mu) / sigma)
    log_cdf_lower = torch.special.log_ndtr((lower - mu) / sigma)
    ratio = torch.clamp(log_cdf_lower - log_cdf_upper, max=-1e-7)
    log_interval = log_cdf_upper + torch.log(-torch.expm1(ratio))
    conditional_exact_nll = (
        0.5 * ((z - mu) / sigma) ** 2
        + torch.log(sigma)
        + 0.5 * math.log(2 * math.pi)
        + log_interval
    )
    exact_nll = -nn.functional.logsigmoid(occurrence_logit) + conditional_exact_nll
    nondetect_nll = -nn.functional.logsigmoid(-occurrence_logit)
    element = torch.where(detected > 0, exact_nll, nondetect_nll)
    return (element * risk).sum() / risk.sum().clamp_min(1)


def predict(
    model: nn.Module,
    inputs: tuple[np.ndarray, ...],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tensors = [torch.from_numpy(value).to(device) for value in inputs]
    model.eval()
    with torch.no_grad():
        logits, mu, raw_sigma = model(*tensors)
        probability = torch.sigmoid(logits)
        sigma = 0.05 + nn.functional.softplus(raw_sigma)
    return probability.cpu().numpy(), mu.cpu().numpy(), sigma.cpu().numpy()


def train_one(
    model_name: str,
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    seed: int,
    device: torch.device,
    external_frame: pd.DataFrame | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    dict,
]:
    set_seed(seed)
    bundle = model_bundle(model_name)
    columns = source_columns(train_frame, bundle)
    preprocessor = SourceTokenPreprocessor.fit(
        train_frame, columns, null_bundle=bundle in {"null", "strictnull"}
    )
    train_inputs = preprocessor.transform(train_frame)
    validation_inputs = preprocessor.transform(validation_frame)
    test_inputs = preprocessor.transform(test_frame)
    external_inputs = (
        preprocessor.transform(external_frame) if external_frame is not None else None
    )
    train_targets = target_arrays(train_frame)
    validation_targets = target_arrays(validation_frame)
    mu_offset = train_targets[3][0] - 0.75
    occurrence_prevalence = (train_targets[0] * train_targets[1]).sum(0) / np.maximum(
        train_targets[1].sum(0), 1
    )
    category_sizes = [
        len(preprocessor.base.category_maps[column]) for column in CAT_COLUMNS
    ]
    model = SourceGatedHurdleTransformer(
        category_sizes,
        train_inputs[-1].shape[1],
        mu_offset,
        occurrence_prevalence,
    ).to(device)
    dataset = TensorDataset(
        *[torch.from_numpy(value) for value in train_inputs],
        *[torch.from_numpy(value) for value in train_targets],
    )
    loader = DataLoader(
        dataset,
        batch_size=512,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    validation_inputs_t = [torch.from_numpy(value).to(device) for value in validation_inputs]
    validation_targets_t = [
        torch.from_numpy(value).to(device) for value in validation_targets
    ]
    best_loss = math.inf
    best_epoch = 0
    best_state = None
    stale = 0
    for epoch in range(1, 81):
        model.train()
        for batch in loader:
            n_inputs = len(train_inputs)
            inputs = [value.to(device) for value in batch[:n_inputs]]
            targets = [value.to(device) for value in batch[n_inputs:]]
            logits, mu, raw_sigma = model(*inputs)
            loss = hurdle_nll(logits, mu, raw_sigma, *targets)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            logits, mu, raw_sigma = model(*validation_inputs_t)
            validation_loss = float(
                hurdle_nll(logits, mu, raw_sigma, *validation_targets_t).cpu()
            )
        if validation_loss < best_loss - 1e-5:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if stale >= 12:
            break
    if best_state is None:
        raise RuntimeError(f"No finite validation state for {model_name}")
    model.load_state_dict(best_state)
    validation_probability, validation_mu, validation_sigma = predict(
        model, validation_inputs, device
    )
    test_probability, test_mu, test_sigma = predict(model, test_inputs, device)
    external_predictions = (
        predict(model, external_inputs, device)
        if external_inputs is not None
        else None
    )
    return (
        validation_probability,
        validation_mu,
        validation_sigma,
        test_probability,
        test_mu,
        test_sigma,
        external_predictions,
        {
            "seed": seed,
            "best_epoch": best_epoch,
            "best_validation_hurdle_nll": best_loss,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "source_bundle": bundle,
            "source_feature_count": len(columns),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--source-features", type=Path, required=True)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--models", default=",".join(sorted(MODELS)))
    parser.add_argument("--seeds", default="20260810,20260811,20260812")
    parser.add_argument("--outer-folds", default="0,1,2,3,4")
    parser.add_argument(
        "--external-cohort",
        type=Path,
        help=(
            "Optional independently versioned cohort to score with each reconstructed "
            "outer-fold model. Rows are routed by their preassigned state_fold and never "
            "enter training, validation, calibration, or early stopping."
        ),
    )
    parser.add_argument(
        "--external-output",
        type=Path,
        help="Required with --external-cohort; receives external distribution predictions.",
    )
    parser.add_argument(
        "--model-prefix",
        default="",
        help="Optional output label prefix; the source bundle is appended (null/strict/broad).",
    )
    parser.add_argument(
        "--temporal-scope",
        default=(
            "Current ECHO features are restricted to modern artificial-censor validation; "
            "historical UCMR3 use requires a pre-UCMR3 source-vintage analysis."
        ),
    )
    args = parser.parse_args()

    if (args.external_cohort is None) != (args.external_output is None):
        raise ValueError("--external-cohort and --external-output must be supplied together")

    torch.set_num_threads(min(8, max(1, torch.get_num_threads())))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cohort = pd.read_csv(
        args.cohort, dtype={"state": str, "region": str, "pws_id": str}
    )
    frame = attach_source_features(cohort, args.source_features)
    external_frame = None
    if args.external_cohort is not None:
        external_cohort = pd.read_csv(
            args.external_cohort,
            dtype={"state": str, "region": str, "pws_id": str},
        )
        if external_cohort["location_id"].duplicated().any():
            raise ValueError("External cohort contains duplicate location_id values")
        external_frame = attach_source_features(external_cohort, args.source_features)
    models = [value.strip() for value in args.models.split(",") if value.strip()]
    unknown = set(models) - MODELS
    if unknown:
        raise ValueError(f"Unknown models: {sorted(unknown)}")
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    folds = [int(value) for value in args.outer_folds.split(",") if value.strip()]
    n_folds = int(frame["state_fold"].max()) + 1
    prediction_frames = []
    external_prediction_frames = []
    training_info = []
    for outer_fold in folds:
        validation_fold = (outer_fold + 1) % n_folds
        train_frame = frame[~frame["state_fold"].isin([outer_fold, validation_fold])].copy()
        validation_frame = frame[frame["state_fold"] == validation_fold].copy()
        test_frame = frame[frame["state_fold"] == outer_fold].copy()
        fold_external_frame = (
            external_frame[external_frame["state_fold"] == outer_fold].copy()
            if external_frame is not None
            else None
        )
        external_scoring_frame = (
            fold_external_frame
            if fold_external_frame is not None and len(fold_external_frame)
            else None
        )
        validation_targets = target_arrays(validation_frame)
        test_targets = target_arrays(test_frame)
        external_targets = (
            target_arrays(external_scoring_frame)
            if external_scoring_frame is not None
            else None
        )
        for model_name in models:
            output_model_name = (
                f"{args.model_prefix}{model_bundle(model_name)}"
                if args.model_prefix
                else model_name
            )
            val_probabilities = []
            val_mus = []
            val_sigmas = []
            test_probabilities = []
            test_mus = []
            test_sigmas = []
            external_probabilities = []
            external_mus = []
            external_sigmas = []
            runs = []
            for seed in seeds:
                values = train_one(
                    model_name,
                    train_frame,
                    validation_frame,
                    test_frame,
                    seed,
                    device,
                    external_scoring_frame,
                )
                (
                    val_p,
                    val_mu,
                    val_sigma,
                    test_p,
                    test_mu,
                    test_sigma,
                    external_values,
                    info,
                ) = values
                val_probabilities.append(val_p)
                val_mus.append(val_mu)
                val_sigmas.append(val_sigma)
                test_probabilities.append(test_p)
                test_mus.append(test_mu)
                test_sigmas.append(test_sigma)
                if external_values is not None:
                    external_p, external_mu, external_sigma = external_values
                    external_probabilities.append(external_p)
                    external_mus.append(external_mu)
                    external_sigmas.append(external_sigma)
                runs.append(info)
            validation_probability = np.mean(val_probabilities, axis=0)
            test_probability = np.mean(test_probabilities, axis=0)
            validation_mu, validation_sigma = moment_match(val_mus, val_sigmas)
            test_mu, test_sigma = moment_match(test_mus, test_sigmas)
            if external_probabilities:
                external_probability = np.mean(external_probabilities, axis=0)
                external_mu, external_sigma = moment_match(
                    external_mus, external_sigmas
                )
            else:
                external_probability = external_mu = external_sigma = None
            _, conditional_mean, q05, q95, log_interval = distribution_summaries(
                test_mu, test_sigma, test_targets[3], test_targets[4]
            )
            calibrated_probability = np.zeros_like(test_probability)
            calibrated_external_probability = (
                np.zeros_like(external_probability)
                if external_probability is not None
                else None
            )
            for chemical_index in range(len(CHEMICALS)):
                validation_risk = validation_targets[1][:, chemical_index].astype(bool)
                calibrator = fit_platt(
                    probability_to_logit(
                        validation_probability[validation_risk, chemical_index]
                    ),
                    validation_targets[0][validation_risk, chemical_index].astype(int),
                )
                calibrated_probability[:, chemical_index] = calibrator(
                    probability_to_logit(test_probability[:, chemical_index])
                )
                if calibrated_external_probability is not None:
                    calibrated_external_probability[:, chemical_index] = calibrator(
                        probability_to_logit(
                            external_probability[:, chemical_index]
                        )
                    )
            training_info.append(
                {
                    "outer_fold": outer_fold,
                    "validation_fold": validation_fold,
                    "model": output_model_name,
                    "training_model": model_name,
                    "train_sites": len(train_frame),
                    "validation_sites": len(validation_frame),
                    "test_sites": len(test_frame),
                    "runs": runs,
                }
            )
            for chemical_index, chemical in enumerate(CHEMICALS):
                target_z = test_targets[2][:, chemical_index].astype(float)
                target_z[test_targets[0][:, chemical_index] == 0] = np.nan
                prediction_frames.append(
                    pd.DataFrame(
                        {
                            "location_id": test_frame["location_id"].to_numpy(),
                            "pws_id": test_frame["pws_id"].to_numpy(),
                            "state": test_frame["state"].to_numpy(),
                            "outer_fold": outer_fold,
                            "model": output_model_name,
                            "chemical": chemical,
                            "risk": test_targets[1][:, chemical_index].astype(int),
                            "y_hidden": (
                                test_targets[0][:, chemical_index]
                                * test_targets[1][:, chemical_index]
                            ).astype(int),
                            "target_log_concentration": target_z,
                            "mu": test_mu[:, chemical_index],
                            "sigma": test_sigma[:, chemical_index],
                            "probability_hidden_raw": test_probability[:, chemical_index],
                            "probability_hidden": calibrated_probability[:, chemical_index],
                            "conditional_mean_log_concentration": conditional_mean[
                                :, chemical_index
                            ],
                            "q05_log_concentration": q05[:, chemical_index],
                            "q95_log_concentration": q95[:, chemical_index],
                            # Keep the established conditional-concentration NLL:
                            # density given that a hidden detection occurred in
                            # [native MRL, legacy MRL). Occurrence quality is
                            # evaluated separately by AP/Brier/calibration.
                            "log_interval_probability": log_interval[:, chemical_index],
                            "hurdle_log_interval": log_interval[:, chemical_index],
                            "source_service_zip_count": test_frame[
                                "source__service_zip_count"
                            ].to_numpy(),
                            "source_feature_missing": test_frame[
                                "source__feature_missing"
                            ].to_numpy(),
                        }
                    )
                )
            if (
                fold_external_frame is not None
                and external_targets is not None
                and external_mu is not None
                and external_sigma is not None
                and external_probability is not None
                and calibrated_external_probability is not None
            ):
                (
                    _,
                    external_conditional_mean,
                    external_q05,
                    external_q95,
                    external_log_interval,
                ) = distribution_summaries(
                    external_mu,
                    external_sigma,
                    external_targets[3],
                    external_targets[4],
                )
                for chemical_index, chemical in enumerate(CHEMICALS):
                    external_target_z = external_targets[2][
                        :, chemical_index
                    ].astype(float)
                    external_target_z[
                        external_targets[0][:, chemical_index] == 0
                    ] = np.nan
                    external_prediction_frames.append(
                        pd.DataFrame(
                            {
                                "location_id": fold_external_frame[
                                    "location_id"
                                ].to_numpy(),
                                "pws_id": fold_external_frame["pws_id"].to_numpy(),
                                "state": fold_external_frame["state"].to_numpy(),
                                "outer_fold": outer_fold,
                                "model": output_model_name,
                                "chemical": chemical,
                                "risk": external_targets[1][
                                    :, chemical_index
                                ].astype(int),
                                "y_hidden": (
                                    external_targets[0][:, chemical_index]
                                    * external_targets[1][:, chemical_index]
                                ).astype(int),
                                "target_log_concentration": external_target_z,
                                "mu": external_mu[:, chemical_index],
                                "sigma": external_sigma[:, chemical_index],
                                "probability_hidden_raw": external_probability[
                                    :, chemical_index
                                ],
                                "probability_hidden": calibrated_external_probability[
                                    :, chemical_index
                                ],
                                "conditional_mean_log_concentration": external_conditional_mean[
                                    :, chemical_index
                                ],
                                "q05_log_concentration": external_q05[
                                    :, chemical_index
                                ],
                                "q95_log_concentration": external_q95[
                                    :, chemical_index
                                ],
                                "log_interval_probability": external_log_interval[
                                    :, chemical_index
                                ],
                                "hurdle_log_interval": external_log_interval[
                                    :, chemical_index
                                ],
                                "source_service_zip_count": fold_external_frame[
                                    "source__service_zip_count"
                                ].to_numpy(),
                                "source_feature_missing": fold_external_frame[
                                    "source__feature_missing"
                                ].to_numpy(),
                                "scoring_role": "release-vintage external",
                            }
                        )
                    )
            print(f"fold={outer_fold} model={model_name} runs={len(runs)}")

    predictions = pd.concat(prediction_frames, ignore_index=True)
    metrics, macro = summarize(predictions)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(
        args.output_dir / "out_of_fold_distribution_predictions.csv.gz",
        index=False,
        compression="gzip",
    )
    metrics.to_csv(args.output_dir / "per_analyte_distribution_metrics.csv", index=False)
    macro.to_csv(args.output_dir / "macro_distribution_metrics.csv", index=False)
    external_audit = None
    if external_frame is not None:
        if not external_prediction_frames:
            raise ValueError("External cohort was supplied but no rows were scored")
        external_predictions = pd.concat(external_prediction_frames, ignore_index=True)
        eligible_external_frame = external_frame[
            external_frame["state_fold"].isin(folds)
        ].copy()
        expected_rows = len(eligible_external_frame) * len(CHEMICALS) * len(models)
        if len(external_predictions) != expected_rows:
            raise ValueError(
                f"External prediction coverage failed: expected {expected_rows}, "
                f"got {len(external_predictions)}"
            )
        expected_keys = {
            (str(location_id), model, chemical)
            for location_id in eligible_external_frame["location_id"]
            for model in [
                f"{args.model_prefix}{model_bundle(model_name)}"
                if args.model_prefix
                else model_name
                for model_name in models
            ]
            for chemical in CHEMICALS
        }
        observed_keys = set(
            zip(
                external_predictions["location_id"].astype(str),
                external_predictions["model"].astype(str),
                external_predictions["chemical"].astype(str),
            )
        )
        if observed_keys != expected_keys:
            raise ValueError("External prediction key coverage is not exact")
        args.external_output.parent.mkdir(parents=True, exist_ok=True)
        external_predictions.to_csv(
            args.external_output, index=False, compression="gzip"
        )
        external_metrics, external_macro = summarize(external_predictions)
        external_metrics.to_csv(
            args.external_output.with_name("external_per_analyte_metrics.csv"),
            index=False,
        )
        external_macro.to_csv(
            args.external_output.with_name("external_macro_metrics.csv"),
            index=False,
        )
        external_audit = {
            "cohort": str(args.external_cohort),
            "output": str(args.external_output),
            "sites_supplied": len(external_frame),
            "sites_scored": len(eligible_external_frame),
            "sites_outside_requested_folds": len(external_frame)
            - len(eligible_external_frame),
            "prediction_rows": len(external_predictions),
            "coverage_gate": "PASS",
            "fold_site_counts": {
                str(key): int(value)
                for key, value in eligible_external_frame["state_fold"]
                .value_counts()
                .sort_index()
                .items()
            },
            "macro_metrics": external_macro.to_dict(orient="records"),
        }
    payload = {
        "design": (
            "A source-gated occurrence hurdle is separated from the truncated-normal "
            "concentration head. Source-null bundles retain the source encoder and gate but "
            "receive only zeros: strictnull uses the exact strict source-feature layout, "
            "whereas null uses the broad layout. Five outer state/primacy folds."
        ),
        "temporal_scope": args.temporal_scope,
        "device": str(device),
        "models": models,
        "seeds": seeds,
        "outer_folds": folds,
        "sites": len(frame),
        "source_audit": str(args.source_audit),
        "training": training_info,
        "macro_metrics": macro.to_dict(orient="records"),
        "external_scoring": external_audit,
    }
    (args.output_dir / "run_info.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(macro.to_string(index=False))


if __name__ == "__main__":
    main()

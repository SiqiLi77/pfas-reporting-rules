#!/usr/bin/env python3
"""Locked ECHO source-landscape ablation for the artificial-censor benchmark.

Only label-free PWS features constructed from EPA ECHO and UCMR service ZIPs
are added. Outer state/primacy folds, validation folds, likelihood, calibration,
and target architecture match the established censored-distribution benchmark.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from run_ucmr5_artificial_censor_screen import (
    BASE_NUMERIC,
    CAT_COLUMNS,
    CHEMICALS,
    FlatPreprocessor,
    TokenPreprocessor,
    fit_platt,
    probability_to_logit,
    set_seed,
)
from run_ucmr5_censored_distribution_screen import (
    LinearTobit,
    distribution_summaries,
    moment_match,
    predict_distribution,
    summarize,
    target_arrays,
    truncated_censored_nll,
)


MODELS = {
    "linear_tobit_echo_strict",
    "linear_tobit_echo_broad",
    "panel_censored_transformer_echo_null",
    "panel_censored_transformer_echo_strict",
    "panel_censored_transformer_echo_broad",
    "panel_censored_transformer_echo_strictnull",
}


def normalize_pws(value: object) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    text = text.upper()
    return text.zfill(9) if text.isdigit() else text


def source_columns(frame: pd.DataFrame, bundle: str) -> list[str]:
    all_columns = sorted(column for column in frame if column.startswith("source__"))
    if bundle in {"broad", "null"}:
        return all_columns
    if bundle in {"strict", "strictnull"}:
        keep = [
            "source__service_zip_count",
            "source__max_evidence_support",
            "source__fraction_zips_with_high_confidence_facility",
            "source__log1p_sum_high_confidence",
            "source__log1p_mean_high_confidence",
            "source__single_zip",
            "source__feature_missing",
        ]
        keep += [column for column in all_columns if "group_high__" in column]
        keep += [
            column
            for column in all_columns
            if "high_confidence" in column and "radius" in column
        ]
        keep += [
            column
            for column in all_columns
            if "distance_high_confidence" in column
        ]
        # Explicit, reporting-year TRI PFAS signals are stricter than an
        # industry-only match and remain available in time-locked ablations.
        keep += [column for column in all_columns if "source__tri2022_" in column]
        return [column for column in keep if column in all_columns]
    raise ValueError(bundle)


def model_bundle(model_name: str) -> str:
    return model_name.rsplit("_", 1)[-1]


def attach_source_features(cohort: pd.DataFrame, path: Path) -> pd.DataFrame:
    features = pd.read_csv(path, dtype={"pws_id": str})
    features["_pws_join"] = features["pws_id"].map(normalize_pws)
    features = features.drop(columns="pws_id").drop_duplicates("_pws_join")
    frame = cohort.copy()
    frame["_pws_join"] = frame["pws_id"].map(normalize_pws)
    frame = frame.merge(features, on="_pws_join", how="left", validate="many_to_one")
    raw_source = [column for column in frame if column.startswith("source__")]
    missing = frame[raw_source].isna().all(axis=1)
    frame[raw_source] = frame[raw_source].fillna(0.0)
    frame["source__feature_missing"] = missing.astype(np.float32)
    return frame


@dataclass
class SourceFlatPreprocessor:
    base: FlatPreprocessor
    columns: list[str]
    mean: np.ndarray
    scale: np.ndarray
    null_bundle: bool

    @classmethod
    def fit(
        cls, frame: pd.DataFrame, columns: list[str], null_bundle: bool
    ) -> "SourceFlatPreprocessor":
        base = FlatPreprocessor.fit(frame)
        values = frame[columns].to_numpy(np.float32)
        if null_bundle:
            values = np.zeros_like(values)
        mean = values.mean(0)
        scale = values.std(0)
        scale[scale < 1e-6] = 1.0
        return cls(base, columns, mean, scale, null_bundle)

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        base = self.base.transform(frame)
        values = frame[self.columns].to_numpy(np.float32)
        if self.null_bundle:
            values = np.zeros_like(values)
        values = (values - self.mean) / self.scale
        return np.concatenate([base, values], axis=1).astype(np.float32)


@dataclass
class SourceTokenPreprocessor:
    base: TokenPreprocessor
    columns: list[str]
    mean: np.ndarray
    scale: np.ndarray
    null_bundle: bool

    @classmethod
    def fit(
        cls, frame: pd.DataFrame, columns: list[str], null_bundle: bool
    ) -> "SourceTokenPreprocessor":
        base = TokenPreprocessor.fit(frame)
        values = frame[columns].to_numpy(np.float32)
        if null_bundle:
            values = np.zeros_like(values)
        mean = values.mean(0)
        scale = values.std(0)
        scale[scale < 1e-6] = 1.0
        return cls(base, columns, mean, scale, null_bundle)

    def transform(
        self, frame: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        token, context, categories = self.base.transform(frame)
        values = frame[self.columns].to_numpy(np.float32)
        if self.null_bundle:
            values = np.zeros_like(values)
        values = (values - self.mean) / self.scale
        return token, context, categories, values.astype(np.float32)


class GatedSourcePanelTransformer(nn.Module):
    def __init__(
        self,
        category_sizes: list[int],
        source_dim: int,
        mu_offset: np.ndarray,
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
        self.output = nn.Linear(d_model, 2)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        token: torch.Tensor,
        context: torch.Tensor,
        categories: torch.Tensor,
        source: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        chemical_ids = torch.arange(len(CHEMICALS), device=token.device)
        hidden = self.token_projection(token) + self.chemical(chemical_ids)[None, :, :]
        embedded = [
            embedding(categories[:, index])
            for index, embedding in enumerate(self.category_embeddings)
        ]
        context_hidden = self.context(torch.cat([context] + embedded, dim=1))
        source_hidden = self.source_encoder(source)
        gate = self.source_gate(torch.cat([context_hidden, source_hidden], dim=1))
        fused = context_hidden + gate * source_hidden
        hidden = self.encoder(hidden + fused[:, None, :])
        output = self.output(hidden)
        return self.mu_offset[None, :] + output[:, :, 0], output[:, :, 1]


def make_inputs(
    model_name: str,
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
) -> tuple[object, tuple[np.ndarray, ...], tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    bundle = model_bundle(model_name)
    columns = source_columns(train_frame, bundle)
    null_bundle = bundle in {"null", "strictnull"}
    if model_name.startswith("linear_tobit"):
        preprocessor = SourceFlatPreprocessor.fit(
            train_frame, columns, null_bundle=null_bundle
        )
    else:
        preprocessor = SourceTokenPreprocessor.fit(
            train_frame, columns, null_bundle=null_bundle
        )
    train_values = preprocessor.transform(train_frame)
    validation_values = preprocessor.transform(validation_frame)
    test_values = preprocessor.transform(test_frame)
    if isinstance(train_values, np.ndarray):
        train_values = (train_values,)
        validation_values = (validation_values,)
        test_values = (test_values,)
    return preprocessor, train_values, validation_values, test_values


def build_model(
    model_name: str,
    preprocessor: object,
    train_inputs: tuple[np.ndarray, ...],
    mu_offset: np.ndarray,
) -> nn.Module:
    if model_name.startswith("linear_tobit"):
        return LinearTobit(train_inputs[0].shape[1], mu_offset)
    assert isinstance(preprocessor, SourceTokenPreprocessor)
    category_sizes = [
        len(preprocessor.base.category_maps[column]) for column in CAT_COLUMNS
    ]
    return GatedSourcePanelTransformer(
        category_sizes, train_inputs[-1].shape[1], mu_offset
    )


def train_one(
    model_name: str,
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    seed: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    set_seed(seed)
    preprocessor, train_inputs, validation_inputs, test_inputs = make_inputs(
        model_name, train_frame, validation_frame, test_frame
    )
    train_targets = target_arrays(train_frame)
    validation_targets = target_arrays(validation_frame)
    mu_offset = train_targets[3][0] - 0.75
    model = build_model(model_name, preprocessor, train_inputs, mu_offset).to(device)
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
    is_linear = model_name.startswith("linear_tobit")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-3 if is_linear else 1e-3, weight_decay=1e-4
    )
    validation_inputs_t = [torch.from_numpy(value).to(device) for value in validation_inputs]
    validation_targets_t = [
        torch.from_numpy(value).to(device) for value in validation_targets
    ]
    best_loss = math.inf
    best_epoch = 0
    best_state = None
    stale = 0
    patience = 12
    max_epochs = 120 if is_linear else 80
    for epoch in range(1, max_epochs + 1):
        model.train()
        for batch in loader:
            n_inputs = len(train_inputs)
            inputs = [value.to(device) for value in batch[:n_inputs]]
            targets = [value.to(device) for value in batch[n_inputs:]]
            mu, raw_sigma = model(*inputs)
            loss = truncated_censored_nll(mu, raw_sigma, *targets)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_mu, validation_raw_sigma = model(*validation_inputs_t)
            validation_loss = float(
                truncated_censored_nll(
                    validation_mu, validation_raw_sigma, *validation_targets_t
                ).cpu()
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
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError(f"No finite validation state for {model_name}")
    model.load_state_dict(best_state)
    validation_mu, validation_sigma = predict_distribution(
        model, validation_inputs, device
    )
    test_mu, test_sigma = predict_distribution(model, test_inputs, device)
    return validation_mu, validation_sigma, test_mu, test_sigma, {
        "seed": seed,
        "best_epoch": best_epoch,
        "best_validation_conditional_nll": best_loss,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "source_bundle": model_bundle(model_name),
        "source_feature_count": len(source_columns(train_frame, model_bundle(model_name))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--source-features", type=Path, required=True)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--models", default=",".join(sorted(MODELS)))
    parser.add_argument("--seeds", default="20260810,20260811,20260812")
    parser.add_argument("--outer-folds", default="0,1,2,3,4")
    args = parser.parse_args()

    torch.set_num_threads(min(8, max(1, torch.get_num_threads())))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cohort = pd.read_csv(
        args.cohort, dtype={"state": str, "region": str, "pws_id": str}
    )
    frame = attach_source_features(cohort, args.source_features)
    models = [value.strip() for value in args.models.split(",") if value.strip()]
    unknown = set(models) - MODELS
    if unknown:
        raise ValueError(f"Unknown models: {sorted(unknown)}")
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    outer_folds = [int(value) for value in args.outer_folds.split(",") if value.strip()]
    n_folds = int(frame["state_fold"].max()) + 1
    if not set(outer_folds) <= set(range(n_folds)):
        raise ValueError("outer-folds outside available state_fold values")

    prediction_frames: list[pd.DataFrame] = []
    training_info: list[dict] = []
    for outer_fold in outer_folds:
        validation_fold = (outer_fold + 1) % n_folds
        train_frame = frame[~frame["state_fold"].isin([outer_fold, validation_fold])].copy()
        validation_frame = frame[frame["state_fold"] == validation_fold].copy()
        test_frame = frame[frame["state_fold"] == outer_fold].copy()
        validation_targets = target_arrays(validation_frame)
        test_targets = target_arrays(test_frame)
        for model_name in models:
            model_seeds = seeds[:1] if model_name.startswith("linear_tobit") else seeds
            validation_mus: list[np.ndarray] = []
            validation_sigmas: list[np.ndarray] = []
            test_mus: list[np.ndarray] = []
            test_sigmas: list[np.ndarray] = []
            runs = []
            for seed in model_seeds:
                values = train_one(
                    model_name,
                    train_frame,
                    validation_frame,
                    test_frame,
                    seed,
                    device,
                )
                validation_mu, validation_sigma, test_mu, test_sigma, info = values
                validation_mus.append(validation_mu)
                validation_sigmas.append(validation_sigma)
                test_mus.append(test_mu)
                test_sigmas.append(test_sigma)
                runs.append(info)
            validation_mu, validation_sigma = moment_match(
                validation_mus, validation_sigmas
            )
            test_mu, test_sigma = moment_match(test_mus, test_sigmas)
            validation_probability, _, _, _, _ = distribution_summaries(
                validation_mu,
                validation_sigma,
                validation_targets[3],
                validation_targets[4],
            )
            test_probability, conditional_mean, q05, q95, log_interval = (
                distribution_summaries(
                    test_mu, test_sigma, test_targets[3], test_targets[4]
                )
            )
            calibrated_probability = np.zeros_like(test_probability)
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
            training_info.append(
                {
                    "outer_fold": outer_fold,
                    "validation_fold": validation_fold,
                    "model": model_name,
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
                            "model": model_name,
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
                            "log_interval_probability": log_interval[:, chemical_index],
                            "source_service_zip_count": test_frame[
                                "source__service_zip_count"
                            ].to_numpy(),
                            "source_feature_missing": test_frame[
                                "source__feature_missing"
                            ].to_numpy(),
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
    source_audit = json.loads(args.source_audit.read_text())
    payload = {
        "design": (
            "Locked ECHO source-covariate ablation under the established five outer "
            "state/primacy folds and conditional truncated-censored likelihood. The null "
            "model has the same broad-source encoder parameterization but receives zeros."
        ),
        "temporal_scope": (
            "Current ECHO source geography is evaluated only as a modern artificial-censor "
            "recovery feature. It is not authorized for the UCMR3 historical counterfactual "
            "until a pre-UCMR3 facility-vintage sensitivity analysis is completed."
        ),
        "device": str(device),
        "models": models,
        "seeds": seeds,
        "outer_folds": outer_folds,
        "sites": len(frame),
        "source_feature_missing_sites": int(frame["source__feature_missing"].sum()),
        "source_feature_columns": {
            bundle: source_columns(frame, bundle) for bundle in ["strict", "broad"]
        },
        "source_audit_sha256": source_audit["sources"]["echo_exporter_sha256"],
        "training": training_info,
        "macro_metrics": macro.to_dict(orient="records"),
    }
    (args.output_dir / "run_info.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(macro.to_string(index=False))


if __name__ == "__main__":
    main()

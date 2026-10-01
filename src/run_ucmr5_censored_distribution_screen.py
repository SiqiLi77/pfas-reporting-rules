#!/usr/bin/env python3
"""State-held-out reconstruction of PFAS concentrations hidden by legacy MRLs.

For analyte a, the modeled risk set is old_visible[a] == 0.  That observation
implies concentration C_a < old_MRL_a.  UCMR5 then either supplies an exact
concentration above its lower native MRL, or a left-censored observation below
that native MRL.  Models are therefore trained with the likelihood conditional
on C_a < old_MRL_a, rather than with substituted non-detect values.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import log_ndtr, ndtr, ndtri
from scipy.stats import norm
from sklearn.metrics import average_precision_score, brier_score_loss
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from run_ucmr5_artificial_censor_screen import (
    CHEMICALS,
    FlatPreprocessor,
    TokenPreprocessor,
    ece10,
    fit_platt,
    probability_to_logit,
    safe_auc,
    set_seed,
)


NEURAL_MODELS = {"flat_censored_mlp", "panel_censored_transformer"}
BASELINE_MODELS = {"marginal_lognormal", "linear_tobit"}


def inverse_softplus(value: float) -> float:
    return math.log(math.expm1(value))


class MarginalLogNormal(nn.Module):
    def __init__(self, mu_offset: np.ndarray):
        super().__init__()
        self.register_buffer("mu_offset", torch.from_numpy(mu_offset.astype(np.float32)))
        self.mu_residual = nn.Parameter(torch.zeros(len(CHEMICALS)))
        self.raw_sigma = nn.Parameter(
            torch.full((len(CHEMICALS),), inverse_softplus(0.95))
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n = len(x)
        mu = (self.mu_offset + self.mu_residual)[None, :].expand(n, -1)
        raw_sigma = self.raw_sigma[None, :].expand(n, -1)
        return mu, raw_sigma


class LinearTobit(nn.Module):
    def __init__(self, input_dim: int, mu_offset: np.ndarray):
        super().__init__()
        self.register_buffer("mu_offset", torch.from_numpy(mu_offset.astype(np.float32)))
        self.mean = nn.Linear(input_dim, len(CHEMICALS))
        nn.init.zeros_(self.mean.weight)
        nn.init.zeros_(self.mean.bias)
        self.raw_sigma = nn.Parameter(
            torch.full((len(CHEMICALS),), inverse_softplus(0.95))
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu = self.mu_offset[None, :] + self.mean(x)
        raw_sigma = self.raw_sigma[None, :].expand(len(x), -1)
        return mu, raw_sigma


class FlatCensoredMLP(nn.Module):
    def __init__(self, input_dim: int, mu_offset: np.ndarray):
        super().__init__()
        self.register_buffer("mu_offset", torch.from_numpy(mu_offset.astype(np.float32)))
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 2 * len(CHEMICALS)),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.network(x).reshape(len(x), len(CHEMICALS), 2)
        return self.mu_offset[None, :] + output[:, :, 0], output[:, :, 1]


class PanelCensoredTransformer(nn.Module):
    def __init__(self, category_sizes: list[int], mu_offset: np.ndarray, d_model: int = 64):
        super().__init__()
        self.register_buffer("mu_offset", torch.from_numpy(mu_offset.astype(np.float32)))
        self.chemical = nn.Embedding(len(CHEMICALS), d_model)
        self.token_projection = nn.Linear(2, d_model)
        self.category_embeddings = nn.ModuleList(
            [nn.Embedding(size + 1, 8) for size in category_sizes]
        )
        context_dim = 3 + 8 * len(category_sizes)
        self.context = nn.Sequential(
            nn.Linear(context_dim, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
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
        self, token: torch.Tensor, context: torch.Tensor, categories: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        chemical_ids = torch.arange(len(CHEMICALS), device=token.device)
        hidden = self.token_projection(token) + self.chemical(chemical_ids)[None, :, :]
        embedded = [
            embedding(categories[:, index])
            for index, embedding in enumerate(self.category_embeddings)
        ]
        context_hidden = self.context(torch.cat([context] + embedded, dim=1))
        hidden = self.encoder(hidden + context_hidden[:, None, :])
        output = self.output(hidden)
        return self.mu_offset[None, :] + output[:, :, 0], output[:, :, 1]


def target_arrays(frame: pd.DataFrame) -> tuple[np.ndarray, ...]:
    detected = frame[[f"y_native_detect__{c}" for c in CHEMICALS]].to_numpy(np.float32)
    risk = 1 - frame[[f"old_visible__{c}" for c in CHEMICALS]].to_numpy(np.float32)
    lower = frame[[f"native_logmrl__{c}" for c in CHEMICALS]].to_numpy(np.float32)
    upper = frame[[f"old_logmrl__{c}" for c in CHEMICALS]].to_numpy(np.float32)
    logratio = frame[[f"y_native_logratio__{c}" for c in CHEMICALS]].to_numpy(np.float32)
    value = np.exp(lower) * np.expm1(logratio)
    z = np.where(detected > 0, np.log(np.maximum(value, 1e-12)), lower)
    return detected, risk, z.astype(np.float32), lower, upper


def truncated_censored_nll(
    mu: torch.Tensor,
    raw_sigma: torch.Tensor,
    detected: torch.Tensor,
    risk: torch.Tensor,
    z: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> torch.Tensor:
    sigma = 0.05 + nn.functional.softplus(raw_sigma)
    standardized_upper = (upper - mu) / sigma
    log_cdf_upper = torch.special.log_ndtr(standardized_upper)
    log_cdf_lower = torch.special.log_ndtr((lower - mu) / sigma)
    exact_nll = (
        0.5 * ((z - mu) / sigma) ** 2
        + torch.log(sigma)
        + 0.5 * math.log(2 * math.pi)
        + log_cdf_upper
    )
    censored_nll = log_cdf_upper - log_cdf_lower
    element = torch.where(detected > 0, exact_nll, censored_nll)
    return (element * risk).sum() / risk.sum().clamp_min(1)


def make_inputs(
    model_name: str,
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
) -> tuple[object, tuple[np.ndarray, ...], tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    if model_name in {"marginal_lognormal", "linear_tobit", "flat_censored_mlp"}:
        preprocessor = FlatPreprocessor.fit(train_frame)
        return (
            preprocessor,
            (preprocessor.transform(train_frame),),
            (preprocessor.transform(validation_frame),),
            (preprocessor.transform(test_frame),),
        )
    if model_name == "panel_censored_transformer":
        preprocessor = TokenPreprocessor.fit(train_frame)
        return (
            preprocessor,
            preprocessor.transform(train_frame),
            preprocessor.transform(validation_frame),
            preprocessor.transform(test_frame),
        )
    raise ValueError(model_name)


def build_model(
    model_name: str,
    preprocessor: object,
    inputs: tuple[np.ndarray, ...],
    mu_offset: np.ndarray,
) -> nn.Module:
    if model_name == "marginal_lognormal":
        return MarginalLogNormal(mu_offset)
    if model_name == "linear_tobit":
        return LinearTobit(inputs[0].shape[1], mu_offset)
    if model_name == "flat_censored_mlp":
        return FlatCensoredMLP(inputs[0].shape[1], mu_offset)
    if model_name == "panel_censored_transformer":
        assert isinstance(preprocessor, TokenPreprocessor)
        category_sizes = [len(preprocessor.category_maps[column]) for column in preprocessor.category_maps]
        return PanelCensoredTransformer(category_sizes, mu_offset)
    raise ValueError(model_name)


def predict_distribution(
    model: nn.Module, inputs: tuple[np.ndarray, ...], device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    tensors = [torch.from_numpy(value).to(device) for value in inputs]
    model.eval()
    with torch.no_grad():
        mu, raw_sigma = model(*tensors)
        sigma = 0.05 + nn.functional.softplus(raw_sigma)
    return mu.cpu().numpy(), sigma.cpu().numpy()


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
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=512, shuffle=True, generator=generator)
    learning_rate = 3e-3 if model_name in BASELINE_MODELS else 1e-3
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    validation_tensors = [torch.from_numpy(value).to(device) for value in validation_inputs]
    validation_target_tensors = [
        torch.from_numpy(value).to(device) for value in validation_targets
    ]

    best_loss = math.inf
    best_epoch = 0
    best_state = None
    stale = 0
    patience = 12
    max_epochs = 120 if model_name in BASELINE_MODELS else 80
    for epoch in range(1, max_epochs + 1):
        model.train()
        for batch in loader:
            n_inputs = len(train_inputs)
            features = [value.to(device) for value in batch[:n_inputs]]
            targets = [value.to(device) for value in batch[n_inputs:]]
            mu, raw_sigma = model(*features)
            loss = truncated_censored_nll(mu, raw_sigma, *targets)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_mu, validation_raw_sigma = model(*validation_tensors)
            validation_loss = float(
                truncated_censored_nll(
                    validation_mu, validation_raw_sigma, *validation_target_tensors
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
    validation_mu, validation_sigma = predict_distribution(model, validation_inputs, device)
    test_mu, test_sigma = predict_distribution(model, test_inputs, device)
    return validation_mu, validation_sigma, test_mu, test_sigma, {
        "seed": seed,
        "best_epoch": best_epoch,
        "best_validation_conditional_nll": best_loss,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    }


def moment_match(mus: list[np.ndarray], sigmas: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    stacked_mu = np.stack(mus)
    stacked_sigma = np.stack(sigmas)
    mean = stacked_mu.mean(axis=0)
    second_moment = (stacked_sigma**2 + stacked_mu**2).mean(axis=0)
    variance = np.maximum(second_moment - mean**2, 0.05**2)
    return mean, np.sqrt(variance)


def log_difference(log_b: np.ndarray, log_a: np.ndarray) -> np.ndarray:
    """Return log(exp(log_b) - exp(log_a)) for log_b >= log_a."""
    ratio = np.minimum(log_a - log_b, -1e-12)
    return log_b + np.log(-np.expm1(ratio))


def distribution_summaries(
    mu: np.ndarray, sigma: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    alpha = (lower - mu) / sigma
    beta = (upper - mu) / sigma
    log_cdf_alpha = log_ndtr(alpha)
    log_cdf_beta = log_ndtr(beta)
    log_interval = log_difference(log_cdf_beta, log_cdf_alpha)
    probability_hidden = np.exp(log_interval - log_cdf_beta)

    cdf_alpha = ndtr(alpha)
    cdf_beta = ndtr(beta)
    interval_probability = np.maximum(cdf_beta - cdf_alpha, 1e-12)
    conditional_mean = mu + sigma * (
        norm.pdf(alpha) - norm.pdf(beta)
    ) / interval_probability
    q05_probability = np.clip(cdf_alpha + 0.05 * interval_probability, 1e-10, 1 - 1e-10)
    q95_probability = np.clip(cdf_alpha + 0.95 * interval_probability, 1e-10, 1 - 1e-10)
    q05 = mu + sigma * ndtri(q05_probability)
    q95 = mu + sigma * ndtri(q95_probability)
    return probability_hidden, conditional_mean, q05, q95, log_interval


def summarize(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for (model, chemical), group in predictions.groupby(["model", "chemical"]):
        risk_group = group[group["risk"] == 1]
        y = risk_group["y_hidden"].to_numpy(int)
        probability = risk_group["probability_hidden"].to_numpy(float)
        hidden = risk_group[risk_group["y_hidden"] == 1]
        residual = hidden["conditional_mean_log_concentration"] - hidden["target_log_concentration"]
        standardized = (
            hidden["target_log_concentration"] - hidden["mu"]
        ) / hidden["sigma"]
        log_density = norm.logpdf(standardized) - np.log(hidden["sigma"])
        conditional_nll = -log_density + hidden["log_interval_probability"]
        rows.append(
            {
                "model": model,
                "chemical": chemical,
                "n_risk": len(risk_group),
                "hidden_positives": int(y.sum()),
                "average_precision": float(average_precision_score(y, probability)),
                "roc_auc": safe_auc(y, probability),
                "brier": float(brier_score_loss(y, probability)),
                "ece10": ece10(y, probability),
                "hidden_log_mae": float(np.mean(np.abs(residual))),
                "hidden_log_rmse": float(np.sqrt(np.mean(residual**2))),
                "hidden_conditional_nll": float(np.mean(conditional_nll)),
                "hidden_central90_coverage": float(
                    np.mean(
                        (hidden["target_log_concentration"] >= hidden["q05_log_concentration"])
                        & (hidden["target_log_concentration"] <= hidden["q95_log_concentration"])
                    )
                ),
                "hidden_central90_width": float(
                    np.mean(hidden["q95_log_concentration"] - hidden["q05_log_concentration"])
                ),
            }
        )
    metrics = pd.DataFrame(rows)
    macro = (
        metrics.groupby("model", as_index=False)
        .agg(
            macro_ap=("average_precision", "mean"),
            macro_roc_auc=("roc_auc", "mean"),
            macro_brier=("brier", "mean"),
            macro_ece10=("ece10", "mean"),
            macro_hidden_log_mae=("hidden_log_mae", "mean"),
            macro_hidden_log_rmse=("hidden_log_rmse", "mean"),
            macro_hidden_conditional_nll=("hidden_conditional_nll", "mean"),
            macro_hidden_central90_coverage=("hidden_central90_coverage", "mean"),
            macro_hidden_central90_width=("hidden_central90_width", "mean"),
        )
        .sort_values("macro_hidden_conditional_nll")
    )
    return metrics, macro


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--models",
        default="marginal_lognormal,linear_tobit,flat_censored_mlp,panel_censored_transformer",
    )
    parser.add_argument("--seeds", default="20260809,20260810,20260811")
    args = parser.parse_args()

    torch.set_num_threads(min(8, max(1, torch.get_num_threads())))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame = pd.read_csv(args.cohort, dtype={"state": str, "region": str, "pws_id": str})
    models = [item.strip() for item in args.models.split(",") if item.strip()]
    if not set(models) <= NEURAL_MODELS | BASELINE_MODELS:
        raise ValueError(f"Unknown models: {sorted(set(models) - NEURAL_MODELS - BASELINE_MODELS)}")
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    n_folds = int(frame["state_fold"].max()) + 1
    prediction_frames = []
    training_info = []

    for outer_fold in range(n_folds):
        validation_fold = (outer_fold + 1) % n_folds
        train_frame = frame[~frame["state_fold"].isin([outer_fold, validation_fold])].copy()
        validation_frame = frame[frame["state_fold"] == validation_fold].copy()
        test_frame = frame[frame["state_fold"] == outer_fold].copy()
        validation_targets = target_arrays(validation_frame)
        test_targets = target_arrays(test_frame)

        for model_name in models:
            model_seeds = seeds if model_name in NEURAL_MODELS else seeds[:1]
            validation_mus = []
            validation_sigmas = []
            test_mus = []
            test_sigmas = []
            model_runs = []
            for seed in model_seeds:
                validation_mu, validation_sigma, test_mu, test_sigma, info = train_one(
                    model_name,
                    train_frame,
                    validation_frame,
                    test_frame,
                    seed,
                    device,
                )
                validation_mus.append(validation_mu)
                validation_sigmas.append(validation_sigma)
                test_mus.append(test_mu)
                test_sigmas.append(test_sigma)
                model_runs.append(info)
            validation_mu, validation_sigma = moment_match(validation_mus, validation_sigmas)
            test_mu, test_sigma = moment_match(test_mus, test_sigmas)
            validation_probability, _, _, _, _ = distribution_summaries(
                validation_mu,
                validation_sigma,
                validation_targets[3],
                validation_targets[4],
            )
            test_probability, conditional_mean, q05, q95, log_interval = distribution_summaries(
                test_mu, test_sigma, test_targets[3], test_targets[4]
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
                    "runs": model_runs,
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
                            "conditional_mean_log_concentration": conditional_mean[:, chemical_index],
                            "q05_log_concentration": q05[:, chemical_index],
                            "q95_log_concentration": q95[:, chemical_index],
                            "log_interval_probability": log_interval[:, chemical_index],
                        }
                    )
                )

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
    payload = {
        "design": (
            "Five state/primacy-entity outer folds. For every analyte, likelihood is "
            "conditioned on nonvisibility below its UCMR3 MRL; UCMR5 non-detects are "
            "left-censored at the native MRL and UCMR5 detections are exact. Validation "
            "uses the next held-out fold; remaining three folds train."
        ),
        "device": str(device),
        "models": models,
        "seeds": seeds,
        "training": training_info,
        "macro_metrics": macro.to_dict(orient="records"),
    }
    (args.output_dir / "run_info.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(macro.to_string(index=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Screen whether models can recover UCMR5 detections hidden by UCMR3 MRLs.

The primary risk set for analyte a contains sites where a is not visible at the
UCMR3 MRL. The label is whether the actual UCMR5 result was detected between
the modern and old MRLs. Five outer folds are disjoint by state/primacy entity;
the next fold is used for calibration/early stopping and the other three for
training. Every site receives one out-of-fold prediction.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


CHEMICALS = ("PFBS", "PFHpA", "PFHxS", "PFNA", "PFOA", "PFOS")
SEEDS = (20260809, 20260810, 20260811)
CAT_COLUMNS = ("region", "pws_size", "water_type")
BASE_NUMERIC = ("month_sin", "month_cos", "n_old_visible")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -30, 30)
    return 1 / (1 + np.exp(-values))


def probability_to_logit(probability: np.ndarray) -> np.ndarray:
    probability = np.clip(probability, 1e-6, 1 - 1e-6)
    return np.log(probability / (1 - probability))


def ece10(y: np.ndarray, p: np.ndarray) -> float:
    bins = np.linspace(0, 1, 11)
    total = len(y)
    value = 0.0
    for left, right in zip(bins[:-1], bins[1:]):
        mask = (p >= left) & (p < right if right < 1 else p <= right)
        if mask.any():
            value += mask.mean() * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return value


def safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")


def fit_platt(validation_score: np.ndarray, validation_y: np.ndarray):
    if len(np.unique(validation_y)) < 2:
        constant = float(validation_y.mean())
        return lambda score: np.full(len(score), constant, dtype=float)
    model = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
    model.fit(validation_score.reshape(-1, 1), validation_y)
    return lambda score: model.predict_proba(np.asarray(score).reshape(-1, 1))[:, 1]


@dataclass
class FlatPreprocessor:
    numeric_columns: list[str]
    category_levels: dict[str, list[str]]
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, frame: pd.DataFrame) -> "FlatPreprocessor":
        numeric = list(BASE_NUMERIC)
        for chemical in CHEMICALS:
            numeric += [f"old_visible__{chemical}", f"old_logratio__{chemical}"]
        values = frame[numeric].to_numpy(dtype=np.float32)
        mean = values.mean(0)
        scale = values.std(0)
        scale[scale < 1e-6] = 1.0
        levels = {
            column: sorted(frame[column].fillna("<MISSING>").astype(str).unique())
            for column in CAT_COLUMNS
        }
        return cls(numeric, levels, mean, scale)

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        numeric = (frame[self.numeric_columns].to_numpy(dtype=np.float32) - self.mean) / self.scale
        pieces = [numeric]
        for column in CAT_COLUMNS:
            values = frame[column].fillna("<MISSING>").astype(str).to_numpy()
            for level in self.category_levels[column]:
                pieces.append((values == level).astype(np.float32)[:, None])
        return np.concatenate(pieces, axis=1).astype(np.float32)


@dataclass
class TokenPreprocessor:
    context_mean: np.ndarray
    context_scale: np.ndarray
    category_maps: dict[str, dict[str, int]]

    @classmethod
    def fit(cls, frame: pd.DataFrame) -> "TokenPreprocessor":
        context = frame[list(BASE_NUMERIC)].to_numpy(dtype=np.float32)
        mean = context.mean(0)
        scale = context.std(0)
        scale[scale < 1e-6] = 1.0
        maps = {}
        for column in CAT_COLUMNS:
            levels = sorted(frame[column].fillna("<MISSING>").astype(str).unique())
            maps[column] = {level: index + 1 for index, level in enumerate(levels)}
        return cls(mean, scale, maps)

    def transform(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        token = np.stack(
            [
                frame[[f"old_visible__{chemical}", f"old_logratio__{chemical}"]]
                .to_numpy(dtype=np.float32)
                for chemical in CHEMICALS
            ],
            axis=1,
        )
        context = (frame[list(BASE_NUMERIC)].to_numpy(dtype=np.float32) - self.context_mean) / self.context_scale
        categories = np.zeros((len(frame), len(CAT_COLUMNS)), dtype=np.int64)
        for index, column in enumerate(CAT_COLUMNS):
            mapping = self.category_maps[column]
            values = frame[column].fillna("<MISSING>").astype(str)
            categories[:, index] = [mapping.get(value, 0) for value in values]
        return token, context.astype(np.float32), categories


class FlatMLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, len(CHEMICALS)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class PanelTransformer(nn.Module):
    def __init__(self, category_sizes: list[int], d_model: int = 64):
        super().__init__()
        self.chemical = nn.Embedding(len(CHEMICALS), d_model)
        self.token_projection = nn.Linear(2, d_model)
        self.category_embeddings = nn.ModuleList(
            [nn.Embedding(size + 1, 8) for size in category_sizes]
        )
        context_dim = len(BASE_NUMERIC) + 8 * len(category_sizes)
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
        self.output = nn.Linear(d_model, 1)

    def forward(
        self, token: torch.Tensor, context: torch.Tensor, categories: torch.Tensor
    ) -> torch.Tensor:
        chemical_ids = torch.arange(len(CHEMICALS), device=token.device)
        hidden = self.token_projection(token) + self.chemical(chemical_ids)[None, :, :]
        embedded = [
            embedding(categories[:, index])
            for index, embedding in enumerate(self.category_embeddings)
        ]
        context_hidden = self.context(torch.cat([context] + embedded, dim=1))
        hidden = hidden + context_hidden[:, None, :]
        hidden = self.encoder(hidden)
        return self.output(hidden).squeeze(-1)


def macro_ap(y: np.ndarray, probability: np.ndarray, risk: np.ndarray) -> float:
    scores = []
    for index in range(len(CHEMICALS)):
        mask = risk[:, index].astype(bool)
        scores.append(average_precision_score(y[mask, index], probability[mask, index]))
    return float(np.mean(scores))


def train_neural(
    mode: str,
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    seed: int,
    device: torch.device,
) -> tuple[np.ndarray, dict]:
    set_seed(seed)
    y_train = train_frame[[f"y_hidden_by_old_mrl__{c}" for c in CHEMICALS]].to_numpy(np.float32)
    y_validation = validation_frame[[f"y_hidden_by_old_mrl__{c}" for c in CHEMICALS]].to_numpy(np.float32)
    risk_train = 1 - train_frame[[f"old_visible__{c}" for c in CHEMICALS]].to_numpy(np.float32)
    risk_validation = (
        1
        - validation_frame[
            [f"old_visible__{c}" for c in CHEMICALS]
        ].to_numpy(np.int8)
    ).astype(bool)

    if mode == "flat_mlp":
        preprocessor = FlatPreprocessor.fit(train_frame)
        train_inputs = (preprocessor.transform(train_frame),)
        validation_inputs = (preprocessor.transform(validation_frame),)
        test_inputs = (preprocessor.transform(test_frame),)
        model = FlatMLP(train_inputs[0].shape[1]).to(device)
    elif mode == "panel_transformer":
        preprocessor = TokenPreprocessor.fit(train_frame)
        train_inputs = preprocessor.transform(train_frame)
        validation_inputs = preprocessor.transform(validation_frame)
        test_inputs = preprocessor.transform(test_frame)
        category_sizes = [len(preprocessor.category_maps[column]) for column in CAT_COLUMNS]
        model = PanelTransformer(category_sizes).to(device)
    else:
        raise ValueError(mode)

    tensor_train = [torch.from_numpy(value) for value in train_inputs]
    dataset = TensorDataset(
        *tensor_train, torch.from_numpy(y_train), torch.from_numpy(risk_train)
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=512, shuffle=True, generator=generator)

    positives = (y_train * risk_train).sum(0)
    negatives = risk_train.sum(0) - positives
    pos_weight = np.clip(negatives / np.maximum(positives, 1), 1, 50).astype(np.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_state = None
    best_score = -math.inf
    best_epoch = 0
    patience = 10
    stale = 0

    validation_tensors = [torch.from_numpy(value).to(device) for value in validation_inputs]
    for epoch in range(1, 81):
        model.train()
        for batch in loader:
            *features, y_batch, risk_batch = [value.to(device) for value in batch]
            logits = model(*features)
            element = nn.functional.binary_cross_entropy_with_logits(
                logits,
                y_batch,
                pos_weight=torch.from_numpy(pos_weight).to(device),
                reduction="none",
            )
            loss = (element * risk_batch).sum() / risk_batch.sum().clamp_min(1)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(*validation_tensors).cpu().numpy()
        score = macro_ap(y_validation, sigmoid(val_logits), risk_validation)
        if score > best_score + 1e-5:
            best_score = score
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break

    model.load_state_dict(best_state)
    model.eval()
    validation_tensors = [torch.from_numpy(value).to(device) for value in validation_inputs]
    test_tensors = [torch.from_numpy(value).to(device) for value in test_inputs]
    with torch.no_grad():
        val_logits = model(*validation_tensors).cpu().numpy()
        test_logits = model(*test_tensors).cpu().numpy()

    calibrated = np.zeros_like(test_logits, dtype=float)
    for index in range(len(CHEMICALS)):
        mask = risk_validation[:, index]
        calibrator = fit_platt(val_logits[mask, index], y_validation[mask, index])
        calibrated[:, index] = calibrator(test_logits[:, index])
    return calibrated, {
        "seed": seed,
        "best_epoch": best_epoch,
        "best_validation_macro_ap": best_score,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    }


def train_tabular(
    mode: str,
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    seed: int,
) -> np.ndarray:
    preprocessor = FlatPreprocessor.fit(train_frame)
    x_train = preprocessor.transform(train_frame)
    x_validation = preprocessor.transform(validation_frame)
    x_test = preprocessor.transform(test_frame)
    probability = np.zeros((len(test_frame), len(CHEMICALS)), dtype=float)
    for index, chemical in enumerate(CHEMICALS):
        train_mask = train_frame[f"old_visible__{chemical}"].to_numpy() == 0
        validation_mask = validation_frame[f"old_visible__{chemical}"].to_numpy() == 0
        y_train = train_frame.loc[train_mask, f"y_hidden_by_old_mrl__{chemical}"].to_numpy(int)
        y_validation = validation_frame.loc[
            validation_mask, f"y_hidden_by_old_mrl__{chemical}"
        ].to_numpy(int)
        if mode == "logistic":
            model = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
        elif mode == "hgb":
            model = HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=150,
                max_leaf_nodes=15,
                min_samples_leaf=30,
                l2_regularization=1.0,
                random_state=seed,
            )
        else:
            raise ValueError(mode)
        model.fit(x_train[train_mask], y_train)
        val_probability = model.predict_proba(x_validation[validation_mask])[:, 1]
        calibrator = fit_platt(probability_to_logit(val_probability), y_validation)
        test_probability = model.predict_proba(x_test)[:, 1]
        probability[:, index] = calibrator(probability_to_logit(test_probability))
    return probability


def summarize(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics = []
    burden = []
    jurisdiction = []
    for model, model_frame in predictions.groupby("model"):
        for chemical in CHEMICALS:
            frame = model_frame[model_frame["chemical"] == chemical]
            risk = frame["risk"].to_numpy(bool)
            y = frame.loc[risk, "y_hidden"].to_numpy(int)
            p = frame.loc[risk, "probability_hidden"].to_numpy(float)
            metrics.append(
                {
                    "model": model,
                    "chemical": chemical,
                    "n_risk": len(y),
                    "hidden_positives": int(y.sum()),
                    "prevalence": float(y.mean()),
                    "average_precision": float(average_precision_score(y, p)),
                    "roc_auc": safe_auc(y, p),
                    "brier": float(brier_score_loss(y, p)),
                    "ece10": ece10(y, p),
                }
            )
            observed_native = frame["y_native_detect"].mean()
            predicted_native = (
                frame["old_visible"] + (1 - frame["old_visible"]) * frame["probability_hidden"]
            ).mean()
            burden.append(
                {
                    "model": model,
                    "chemical": chemical,
                    "observed_native_prevalence": observed_native,
                    "predicted_native_prevalence": predicted_native,
                    "absolute_error": abs(predicted_native - observed_native),
                }
            )
            for state, group in frame.groupby("state"):
                observed = group["y_native_detect"].mean()
                predicted = (
                    group["old_visible"]
                    + (1 - group["old_visible"]) * group["probability_hidden"]
                ).mean()
                jurisdiction.append(
                    {
                        "model": model,
                        "chemical": chemical,
                        "state": state,
                        "n": len(group),
                        "observed_native_prevalence": observed,
                        "predicted_native_prevalence": predicted,
                        "absolute_error": abs(predicted - observed),
                    }
                )
    return pd.DataFrame(metrics), pd.DataFrame(burden), pd.DataFrame(jurisdiction)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--models", default="marginal,logistic,hgb,flat_mlp,panel_transformer")
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    args = parser.parse_args()

    torch.set_num_threads(min(8, max(1, torch.get_num_threads())))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame = pd.read_csv(args.cohort, dtype={"state": str, "region": str, "pws_id": str})
    models = [value.strip() for value in args.models.split(",") if value.strip()]
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    n_folds = int(frame["state_fold"].max()) + 1
    prediction_frames = []
    training_info = []

    for outer_fold in range(n_folds):
        validation_fold = (outer_fold + 1) % n_folds
        train_frame = frame[~frame["state_fold"].isin([outer_fold, validation_fold])].copy()
        validation_frame = frame[frame["state_fold"] == validation_fold].copy()
        test_frame = frame[frame["state_fold"] == outer_fold].copy()
        y_test = test_frame[[f"y_hidden_by_old_mrl__{c}" for c in CHEMICALS]].to_numpy(int)
        risk_test = (
            1
            - test_frame[[f"old_visible__{c}" for c in CHEMICALS]].to_numpy(np.int8)
        ).astype(bool)

        for model_name in models:
            if model_name == "marginal":
                probability = np.zeros((len(test_frame), len(CHEMICALS)), dtype=float)
                for index, chemical in enumerate(CHEMICALS):
                    mask = train_frame[f"old_visible__{chemical}"].to_numpy() == 0
                    prevalence = train_frame.loc[
                        mask, f"y_hidden_by_old_mrl__{chemical}"
                    ].mean()
                    probability[:, index] = prevalence
                info = []
            elif model_name in {"logistic", "hgb"}:
                probability = train_tabular(
                    model_name, train_frame, validation_frame, test_frame, seeds[0]
                )
                info = []
            elif model_name in {"flat_mlp", "panel_transformer"}:
                seed_predictions = []
                info = []
                for seed in seeds:
                    predicted, metadata = train_neural(
                        model_name,
                        train_frame,
                        validation_frame,
                        test_frame,
                        seed,
                        device,
                    )
                    seed_predictions.append(predicted)
                    info.append(metadata)
                probability = np.mean(seed_predictions, axis=0)
            else:
                raise ValueError(f"Unknown model {model_name}")

            training_info.append(
                {
                    "outer_fold": outer_fold,
                    "validation_fold": validation_fold,
                    "model": model_name,
                    "train_sites": len(train_frame),
                    "validation_sites": len(validation_frame),
                    "test_sites": len(test_frame),
                    "neural_runs": info,
                }
            )
            for index, chemical in enumerate(CHEMICALS):
                prediction_frames.append(
                    pd.DataFrame(
                        {
                            "location_id": test_frame["location_id"].to_numpy(),
                            "pws_id": test_frame["pws_id"].to_numpy(),
                            "state": test_frame["state"].to_numpy(),
                            "outer_fold": outer_fold,
                            "model": model_name,
                            "chemical": chemical,
                            "old_visible": test_frame[f"old_visible__{chemical}"].to_numpy(int),
                            "risk": risk_test[:, index],
                            "y_hidden": y_test[:, index],
                            "y_native_detect": test_frame[
                                f"y_native_detect__{chemical}"
                            ].to_numpy(int),
                            "probability_hidden": probability[:, index],
                        }
                    )
                )

    predictions = pd.concat(prediction_frames, ignore_index=True)
    metrics, burden, jurisdiction = summarize(predictions)
    macro = (
        metrics.groupby("model", as_index=False)
        .agg(
            macro_ap=("average_precision", "mean"),
            macro_roc_auc=("roc_auc", "mean"),
            macro_brier=("brier", "mean"),
            macro_ece10=("ece10", "mean"),
        )
        .merge(
            burden.groupby("model", as_index=False).agg(
                macro_absolute_prevalence_error=("absolute_error", "mean")
            ),
            on="model",
        )
        .merge(
            jurisdiction.groupby("model", as_index=False).agg(
                macro_jurisdiction_absolute_error=("absolute_error", "mean")
            ),
            on="model",
        )
        .sort_values("macro_ap", ascending=False)
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(
        args.output_dir / "out_of_fold_predictions.csv.gz",
        index=False,
        compression="gzip",
    )
    metrics.to_csv(args.output_dir / "per_analyte_metrics.csv", index=False)
    burden.to_csv(args.output_dir / "prevalence_recovery.csv", index=False)
    jurisdiction.to_csv(args.output_dir / "jurisdiction_recovery.csv", index=False)
    macro.to_csv(args.output_dir / "macro_metrics.csv", index=False)
    run_info = {
        "design": (
            "Five outcome-blind state/primacy folds; outer test and next-fold validation; "
            "three remaining folds train. Primary target is native UCMR5 detection hidden "
            "below UCMR3 MRL among old-MRL nonvisible results."
        ),
        "device": str(device),
        "models": models,
        "seeds": seeds,
        "training": training_info,
        "macro_metrics": macro.to_dict(orient="records"),
    }
    (args.output_dir / "run_info.json").write_text(
        json.dumps(run_info, indent=2), encoding="utf-8"
    )
    print(macro.to_string(index=False))


if __name__ == "__main__":
    main()

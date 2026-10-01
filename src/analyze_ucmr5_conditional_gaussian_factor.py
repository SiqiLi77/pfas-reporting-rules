#!/usr/bin/env python3
"""Nested conditional low-rank Gaussian-copula baseline for six-PFAS states.

The model is a multivariate probit with site-specific supplied Bernoulli
margins.  Dependence is learned only on outer-test-clean, inner-cross-fitted
training margins, matching the conditional shared-shock evaluation protocol.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "work" / "vendor"))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import torch

from analyze_ucmr5_conditional_joint_models import (
    ANCHORS,
    CHEMICALS,
    QUERIES,
    labels_at,
    margins_at,
    temperature_transform,
)
from analyze_ucmr5_margin_preserving_joint import enumerate_states
from analyze_ucmr5_multithreshold_joint_curve import LOG_SCORE_EPSILON, evaluate_threshold, prediction_lookup


SEEDS = (20260815, 20260851, 20260852)
LOSS_COLUMNS = ("pattern_nll", "count_crps", "count_brier", "k2_brier")


def normal_quadrature(rank: int, order: int) -> tuple[np.ndarray, np.ndarray]:
    roots, weights = np.polynomial.hermite.hermgauss(order)
    roots = np.sqrt(2.0) * roots
    weights = weights / np.sqrt(np.pi)
    node_grid = np.asarray(list(itertools.product(roots, repeat=rank)), dtype=float)
    weight_grid = np.asarray(
        [np.prod(values) for values in itertools.product(weights, repeat=rank)], dtype=float
    )
    weight_grid /= weight_grid.sum()
    return node_grid, weight_grid


class GaussianFactorKernel:
    def __init__(
        self,
        rank: int,
        order: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        maximum_loading_norm: float = 0.95,
    ):
        nodes, weights = normal_quadrature(rank, order)
        self.rank = rank
        self.maximum_loading_norm = maximum_loading_norm
        self.device = device
        self.dtype = dtype
        self.nodes = torch.as_tensor(nodes, dtype=dtype, device=device)
        self.log_weights = torch.log(torch.as_tensor(weights, dtype=dtype, device=device))
        self.states = torch.as_tensor(enumerate_states(), dtype=dtype, device=device)

    def loadings(self, raw: torch.Tensor) -> torch.Tensor:
        scale = torch.sqrt(1.0 + torch.sum(raw * raw, dim=1, keepdim=True))
        return self.maximum_loading_norm * raw / scale

    def conditional_probability(self, marginal: torch.Tensor, raw: torch.Tensor) -> torch.Tensor:
        loading = self.loadings(raw)
        uniqueness = torch.sqrt(torch.clamp(1.0 - torch.sum(loading * loading, dim=1), min=1e-5))
        # 1-1e-8 rounds to exactly one in float32, making ndtri's backward
        # derivative infinite even when the forward loss is finite.
        margin_epsilon = 1e-6 if marginal.dtype == torch.float32 else 1e-10
        probability_epsilon = 1e-6 if marginal.dtype == torch.float32 else 1e-12
        threshold = torch.special.ndtri(
            torch.clamp(marginal, margin_epsilon, 1 - margin_epsilon)
        )
        factor_shift = self.nodes @ loading.T
        standardized = (threshold[:, None, :] - factor_shift[None, :, :]) / uniqueness[None, None, :]
        return torch.clamp(
            torch.special.ndtr(standardized), probability_epsilon, 1 - probability_epsilon
        )

    def observed_log_probability(
        self, marginal: torch.Tensor, labels: torch.Tensor, raw: torch.Tensor
    ) -> torch.Tensor:
        conditional = self.conditional_probability(marginal, raw)
        log_probability = (
            labels[:, None, :] * torch.log(conditional)
            + (1.0 - labels[:, None, :]) * torch.log1p(-conditional)
        ).sum(dim=2)
        return torch.logsumexp(log_probability + self.log_weights[None, :], dim=1)

    def full_probability(
        self, marginal: np.ndarray, raw: np.ndarray, batch_size: int
    ) -> np.ndarray:
        parameter = torch.as_tensor(raw, dtype=self.dtype, device=self.device)
        parts = []
        with torch.no_grad():
            for start in range(0, len(marginal), batch_size):
                current = torch.as_tensor(
                    marginal[start : start + batch_size], dtype=self.dtype, device=self.device
                )
                conditional = self.conditional_probability(current, parameter)
                log_absent = torch.log1p(-conditional)
                log_odds = torch.log(conditional) - log_absent
                state_log = (
                    torch.einsum("bqj,sj->bqs", log_odds, self.states)
                    + log_absent.sum(dim=2)[:, :, None]
                )
                log_probability = torch.logsumexp(
                    state_log + self.log_weights[None, :, None], dim=1
                )
                probability = torch.exp(log_probability)
                probability /= probability.sum(dim=1, keepdim=True)
                parts.append(probability.cpu().numpy())
        return np.ascontiguousarray(np.concatenate(parts).astype(float))


def conditional_nll(
    kernel: GaussianFactorKernel,
    marginal: torch.Tensor,
    labels: torch.Tensor,
    raw: torch.Tensor,
) -> torch.Tensor:
    log_probability = kernel.observed_log_probability(marginal, labels, raw)
    probability = torch.exp(log_probability)
    scored = (1.0 - LOG_SCORE_EPSILON) * probability + LOG_SCORE_EPSILON / 64.0
    return -torch.log(scored).mean()


def initial_raw(seed: int, rank: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.normal(0.0, 0.15, size=(len(CHEMICALS), rank))
    if seed == SEEDS[0]:
        raw[:, 0] = 0.50 + rng.normal(0.0, 0.05, size=len(CHEMICALS))
        if rank > 1:
            raw[:, 1:] *= 0.25
    elif seed == SEEDS[1]:
        raw[:, 0] = np.linspace(0.20, 0.80, len(CHEMICALS))
    return raw.astype(np.float32)


def fit_gaussian_factor(
    margins: np.ndarray,
    labels: np.ndarray,
    rank: int,
    order: int,
    device: torch.device,
    maximum_epochs: int,
    batch_size: int,
) -> dict:
    kernel = GaussianFactorKernel(rank, order, device)
    x = torch.as_tensor(margins, dtype=torch.float32, device=device)
    y = torch.as_tensor(labels, dtype=torch.float32, device=device)
    attempts = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        raw = torch.nn.Parameter(
            torch.as_tensor(initial_raw(seed, rank), dtype=torch.float32, device=device)
        )
        optimizer = torch.optim.Adam([raw], lr=0.025)
        generator = torch.Generator(device=device).manual_seed(seed + 1000)
        best_loss, best_raw, best_epoch, stale = np.inf, None, 0, 0
        for epoch in range(1, maximum_epochs + 1):
            order_rows = torch.randperm(len(x), generator=generator, device=device)
            for start in range(0, len(x), batch_size):
                rows = order_rows[start : start + batch_size]
                loss = conditional_nll(kernel, x[rows], y[rows], raw)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_([raw], 10.0)
                optimizer.step()
            with torch.no_grad():
                total = 0.0
                for start in range(0, len(x), batch_size):
                    value = conditional_nll(
                        kernel, x[start : start + batch_size], y[start : start + batch_size], raw
                    )
                    total += float(value) * len(x[start : start + batch_size])
                current = total / len(x)
            if current < best_loss - 1e-7:
                best_loss = current
                best_raw = raw.detach().clone()
                best_epoch = epoch
                stale = 0
            else:
                stale += 1
            if stale >= 25:
                break
        if best_raw is None:
            raise RuntimeError("No finite Gaussian-factor fit")
        with torch.no_grad():
            loading = kernel.loadings(best_raw).cpu().numpy().astype(float)
        attempts.append(
            {
                "seed": seed,
                "objective": best_loss,
                "best_epoch": best_epoch,
                "epochs_run": epoch,
                "raw": best_raw.cpu().numpy().astype(float),
                "loadings": loading,
            }
        )
    selected = min(attempts, key=lambda item: item["objective"])
    raw = selected.pop("raw")
    loadings = selected["loadings"]
    selected.pop("loadings")
    correlations = loadings @ loadings.T
    np.fill_diagonal(correlations, 1.0)
    return {
        "raw": raw,
        "fit": {
            **selected,
            "success": True,
            "rank": rank,
            "quadrature_order": order,
            "quadrature_nodes": order**rank,
            "nominal_parameters": len(CHEMICALS) * rank,
            "rotation_invariance_not_counted_as_identified_parameters": True,
            "loadings": {chemical: loadings[index].tolist() for index, chemical in enumerate(CHEMICALS)},
            "implied_latent_correlation": correlations.tolist(),
            "multistart": [
                {
                    "seed": item["seed"],
                    "objective": item["objective"],
                    "best_epoch": item["best_epoch"],
                    "epochs_run": item["epochs_run"],
                }
                for item in attempts
            ],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--test-predictions", type=Path, required=True)
    parser.add_argument("--test-prediction-model", required=True)
    parser.add_argument("--nested-training-margins", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--outer-folds", default="1,2,3,4")
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--quadrature-order", type=int, default=16)
    parser.add_argument("--evaluation-quadrature-order", type=int)
    parser.add_argument("--load-fit-decision", type=Path)
    parser.add_argument("--maximum-epochs", type=int, default=180)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--evaluation-batch-size", type=int, default=256)
    parser.add_argument("--geometry", default="log_mrl")
    parser.add_argument("--margin-logit-temperature", type=float, default=1.0)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = f"conditional_gaussian_factor{args.rank}"
    cohort = pd.read_csv(
        args.cohort,
        dtype={"location_id": str, "pws_id": str, "state": str, "region": str},
    )
    columns = ["location_id", "chemical", "model", "mu", "sigma", "probability_hidden"]
    test_prediction = pd.read_csv(
        args.test_predictions, usecols=columns, dtype={"location_id": str}
    )
    test_prediction = test_prediction[
        test_prediction["model"].eq(args.test_prediction_model)
    ].copy()
    nested = pd.read_csv(args.nested_training_margins, dtype={"location_id": str})
    states = enumerate_states()
    outer_folds = [int(value) for value in args.outer_folds.split(",") if value]
    metric_rows, site_parts, fit_records = [], [], []
    maximum_margin_residual = 0.0
    maximum_normalization_residual = 0.0
    minimum_probability = 1.0
    maximum_margin_increase = 0.0

    for outer_fold in outer_folds:
        validation_fold = 1 + outer_fold % 4
        train = cohort[~cohort["state_fold"].isin([outer_fold, validation_fold])].copy()
        test = cohort[cohort["state_fold"].eq(outer_fold)].copy()
        train_prediction = nested[nested["outer_joint_fold"].eq(outer_fold)].copy()
        train_lookup = prediction_lookup(train_prediction, train)
        test_lookup = prediction_lookup(test_prediction, test)
        train_margins, train_labels = [], []
        for alpha in ANCHORS:
            train_margins.append(
                temperature_transform(
                    margins_at(train, train_lookup, alpha, args.geometry),
                    args.margin_logit_temperature,
                )
            )
            train_labels.append(labels_at(train, alpha, args.geometry))
        if args.load_fit_decision is None:
            fitted = fit_gaussian_factor(
                np.concatenate(train_margins),
                np.concatenate(train_labels),
                args.rank,
                args.quadrature_order,
                device,
                args.maximum_epochs,
                args.batch_size,
            )
            raw = fitted.pop("raw")
        else:
            prior = json.loads(args.load_fit_decision.read_text())
            record = next(
                item for item in prior["fit_records"] if int(item["outer_fold"]) == outer_fold
            )
            prior_fit = record["gaussian_factor"]
            loading = np.asarray([prior_fit["loadings"][c] for c in CHEMICALS], dtype=float)
            norm_squared = np.sum(loading * loading, axis=1)
            raw = loading / np.sqrt(0.95**2 - norm_squared)[:, None]
            fitted = {
                "fit": {
                    **prior_fit,
                    "parameters_loaded_without_refit": True,
                    "loaded_fit_decision": str(args.load_fit_decision),
                }
            }
        fit_records.append(
            {
                "outer_fold": outer_fold,
                "validation_fold": validation_fold,
                "train_sites": len(train),
                "training_margin_design": "outer-test-clean inner cross-fitting",
                "gaussian_factor": fitted["fit"],
            }
        )
        evaluation_order = args.evaluation_quadrature_order or args.quadrature_order
        evaluation_kernel = GaussianFactorKernel(
            args.rank, evaluation_order, device, dtype=torch.float64
        )
        previous_margin = None
        for alpha in QUERIES:
            marginal = temperature_transform(
                margins_at(test, test_lookup, alpha, args.geometry),
                args.margin_logit_temperature,
            )
            if previous_margin is not None:
                maximum_margin_increase = max(
                    maximum_margin_increase, float(np.max(marginal - previous_margin))
                )
            previous_margin = marginal
            probability = evaluation_kernel.full_probability(
                marginal, raw, args.evaluation_batch_size
            )
            minimum_probability = min(minimum_probability, float(probability.min()))
            maximum_normalization_residual = max(
                maximum_normalization_residual,
                float(np.max(np.abs(probability.sum(axis=1) - 1.0))),
            )
            maximum_margin_residual = max(
                maximum_margin_residual,
                float(np.max(np.abs(probability @ states - marginal))),
            )
            labels = labels_at(test, alpha, args.geometry)
            codes = labels @ (2 ** np.arange(len(CHEMICALS)))
            aggregate, site, _ = evaluate_threshold(
                probability,
                labels,
                test,
                model=model_name,
                alpha=alpha,
                outer_fold=outer_fold,
            )
            metric_rows.append(aggregate)
            site["pws_id"] = test["pws_id"].astype(str).to_numpy()
            site["observed_state_probability_raw"] = probability[np.arange(len(test)), codes]
            site_parts.append(site)
        print(f"fold={outer_fold} objective={fitted['fit']['objective']:.6f} complete", flush=True)

    metrics = pd.DataFrame(metric_rows)
    sites = pd.concat(site_parts, ignore_index=True)
    integrated = sites.groupby(["location_id", "pws_id", "state", "model"], as_index=False)[
        list(LOSS_COLUMNS)
    ].mean()
    decision = {
        "status": "CONDITIONAL GAUSSIAN-FACTOR BASELINE COMPLETE",
        "analysis_role": "post-failure exploratory comparator under the frozen conditional protocol",
        "model": model_name,
        "estimand": "sitewise conditional P(Y_i | p_i) composite log likelihood across three anchors",
        "rank": args.rank,
        "training_quadrature_order": args.quadrature_order,
        "evaluation_quadrature_order": args.evaluation_quadrature_order
        or args.quadrature_order,
        "test_prediction_model": args.test_prediction_model,
        "training_margins": str(args.nested_training_margins),
        "margin_logit_temperature": args.margin_logit_temperature,
        "outer_folds": outer_folds,
        "device": str(device),
        "fit_records": fit_records,
        "integrated_metrics": integrated.groupby("model")[list(LOSS_COLUMNS)].mean().to_dict(orient="index"),
        "minimum_state_probability": minimum_probability,
        "maximum_probability_normalization_residual": maximum_normalization_residual,
        "maximum_supplied_margin_residual": maximum_margin_residual,
        "maximum_supplied_margin_increase_across_query_thresholds": maximum_margin_increase,
        "cross_threshold_coherence": (
            "Analytic: all thresholds share the same latent Gaussian factors and residual normals; "
            "decreasing supplied margins therefore induce nested Bernoulli states sitewise."
        ),
        "interpretation_boundary": (
            "The low-rank latent correlation is a predictive residual-dependence parameter after "
            "conditioning on cross-fitted margins, not an identified chemical source mechanism."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output_dir / "threshold_metrics.csv", index=False)
    sites.to_csv(args.output_dir / "site_threshold_losses.csv.gz", index=False, compression="gzip")
    integrated.to_csv(args.output_dir / "integrated_site_losses.csv.gz", index=False, compression="gzip")
    (args.output_dir / "decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    print(json.dumps(decision, indent=2), flush=True)


if __name__ == "__main__":
    main()

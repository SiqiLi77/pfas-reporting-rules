#!/usr/bin/env python3
"""Sitewise conditional comparison of coherent six-PFAS joint models.

The dependence likelihood is conditioned on outer-test-clean, inner-cross-
fitted site margins.  It therefore targets P(Y_i | p_i), not an aggregate
state histogram evaluated at an average marginal vector.
"""

from __future__ import annotations

import argparse
import ctypes
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
from scipy.optimize import minimize
from scipy.special import log_ndtr, logsumexp

from analyze_ucmr5_margin_preserving_joint import CHEMICALS, enumerate_states
from analyze_ucmr5_multithreshold_joint_curve import (
    LOG_SCORE_EPSILON,
    evaluate_threshold,
    factorized_probability,
    prediction_lookup,
)


ANCHORS = (0.0, 0.5, 1.0)
QUERIES = (
    0.05, 0.10, 0.15, 0.20, 0.30, 0.35, 0.40, 0.45,
    0.55, 0.60, 0.65, 0.70, 0.80, 0.85, 0.90, 0.95,
)
GRID = tuple(np.round(np.linspace(0.0, 1.0, 21), 2))
LOSS_COLUMNS = ("pattern_nll", "count_crps", "count_brier", "k2_brier")
SEEDS = (20260813, 20260831, 20260832)


def temperature_transform(marginal: np.ndarray, temperature: float) -> np.ndarray:
    """Apply a monotone logit-temperature transform to supplied margins."""
    if temperature <= 0:
        raise ValueError("margin temperature must be positive")
    if np.isclose(temperature, 1.0):
        return marginal
    clipped = np.clip(marginal, 1e-8, 1 - 1e-8)
    logit = np.log(clipped) - np.log1p(-clipped)
    scaled = np.clip(logit / temperature, -40.0, 40.0)
    return np.clip(1.0 / (1.0 + np.exp(-scaled)), 1e-8, 1 - 1e-8)


def subset_arrays() -> tuple[np.ndarray, np.ndarray]:
    codes = np.arange(64)
    states = ((codes[:, None] >> np.arange(6)) & 1).astype(bool)
    shocks = states[1:]
    mobius = np.zeros((64, 64), dtype=float)
    for state in codes:
        for superset in codes:
            if (superset & state) == state:
                mobius[state, superset] = (-1.0) ** (
                    states[superset].sum() - states[state].sum()
                )
    return shocks, mobius


SHOCKS, MOBIUS = subset_arrays()


def threshold_log_values(frame: pd.DataFrame, alpha: float, geometry: str) -> np.ndarray:
    lower = frame[[f"native_logmrl__{c}" for c in CHEMICALS]].to_numpy(float)
    upper = frame[[f"old_logmrl__{c}" for c in CHEMICALS]].to_numpy(float)
    if geometry == "log_mrl":
        return lower + alpha * (upper - lower)
    if geometry == "concentration_linear":
        return np.log((1 - alpha) * np.exp(lower) + alpha * np.exp(upper))
    raise KeyError(geometry)


def labels_at(frame: pd.DataFrame, alpha: float, geometry: str) -> np.ndarray:
    threshold = threshold_log_values(frame, alpha, geometry)
    lower = frame[[f"native_logmrl__{c}" for c in CHEMICALS]].to_numpy(float)
    logratio = frame[[f"y_native_logratio__{c}" for c in CHEMICALS]].to_numpy(float)
    detected = frame[[f"y_native_detect__{c}" for c in CHEMICALS]].to_numpy(bool)
    value = np.exp(lower) * np.expm1(logratio)
    return (detected & (np.log(np.maximum(value, 1e-30)) >= threshold - 1e-12)).astype(int)


def log_difference(log_b: np.ndarray, log_a: np.ndarray) -> np.ndarray:
    ratio = np.minimum(log_a - log_b, -1e-12)
    return log_b + np.log(-np.expm1(ratio))


def margins_at(
    frame: pd.DataFrame, lookup: pd.DataFrame, alpha: float, geometry: str
) -> np.ndarray:
    thresholds = threshold_log_values(frame, alpha, geometry)
    marginal = np.zeros((len(frame), len(CHEMICALS)), dtype=float)
    for index, chemical in enumerate(CHEMICALS):
        rows = lookup.xs(chemical, level="chemical")
        lower = frame[f"native_logmrl__{chemical}"].to_numpy(float)
        upper = frame[f"old_logmrl__{chemical}"].to_numpy(float)
        threshold = thresholds[:, index]
        old_visible = frame[f"old_visible__{chemical}"].to_numpy(bool)
        occurrence = rows["probability_hidden"].to_numpy(float)
        mu = rows["mu"].to_numpy(float)
        sigma = rows["sigma"].to_numpy(float)
        log_upper = log_ndtr((upper - mu) / sigma)
        log_lower = log_ndtr((lower - mu) / sigma)
        log_threshold = log_ndtr((threshold - mu) / sigma)
        full_interval = log_difference(log_upper, log_lower)
        tail_interval = log_difference(log_upper, log_threshold)
        tail = np.exp(tail_interval - full_interval)
        tail = np.where(alpha >= 1.0, 0.0, tail)
        marginal[:, index] = np.where(
            old_visible, 1.0, occurrence * np.clip(tail, 0.0, 1.0)
        )
    return np.clip(marginal, 1e-8, 1 - 1e-8)


def component_definitions() -> list[dict]:
    definitions = [
        {"name": "independent", "blocks": [[i] for i in range(6)]},
        {"name": "all_shared", "blocks": [list(range(6))]},
    ]
    for left, right in itertools.combinations(range(6), 2):
        definitions.append(
            {
                "name": f"pair_shared__{CHEMICALS[left]}__{CHEMICALS[right]}",
                "blocks": [[left, right], *[[i] for i in range(6) if i not in (left, right)]],
            }
        )
    return definitions


def partition_probability(marginal: np.ndarray, blocks: list[list[int]]) -> np.ndarray:
    states = enumerate_states().astype(bool)
    probability = np.ones((len(marginal), len(states)), dtype=float)
    for block in blocks:
        block_state = states[:, block]
        for code in range(len(states)):
            present = np.asarray(block)[block_state[code]]
            absent = np.asarray(block)[~block_state[code]]
            upper = np.min(marginal[:, present], axis=1) if len(present) else np.ones(len(marginal))
            lower = np.max(marginal[:, absent], axis=1) if len(absent) else np.zeros(len(marginal))
            probability[:, code] *= np.maximum(upper - lower, 0.0)
    probability /= probability.sum(axis=1, keepdims=True)
    return probability


def mixture_probability(
    marginal: np.ndarray, definitions: list[dict], weights: np.ndarray
) -> np.ndarray:
    probability = np.zeros((len(marginal), 64), dtype=float)
    for definition, weight in zip(definitions, weights):
        probability += float(weight) * partition_probability(marginal, definition["blocks"])
    probability /= probability.sum(axis=1, keepdims=True)
    return np.ascontiguousarray(probability)


def shock_probability(marginal: np.ndarray, rates: np.ndarray) -> np.ndarray:
    analyte_rate = SHOCKS.T @ rates
    threshold = -np.log(np.clip(marginal, 1e-15, 1.0)) / analyte_rate[None, :]
    order = np.argsort(-threshold, axis=1)
    probability = np.empty((len(marginal), 64), dtype=float)
    shock_codes = np.arange(1, 64)
    for permutation in np.unique(order, axis=0):
        rows = np.all(order == permutation[None, :], axis=1)
        coefficient = np.zeros((64, 6), dtype=float)
        for state in range(1, 64):
            prior = 0
            for analyte in permutation:
                bit = 1 << int(analyte)
                if state & bit:
                    eligible = ((shock_codes & bit) != 0) & ((shock_codes & prior) == 0)
                    coefficient[state, analyte] = rates[eligible].sum()
                    prior |= bit
        moments = np.exp(-threshold[rows] @ coefficient.T)
        moments[:, 0] = 1.0
        current = moments @ MOBIUS.T
        current[np.abs(current) < 1e-14] = 0.0
        current = np.maximum(current, 0.0)
        current /= current.sum(axis=1, keepdims=True)
        probability[rows] = current
    return np.ascontiguousarray(probability)


def load_transport(path: Path):
    library = ctypes.CDLL(str(path.resolve()))
    function = library.boolean_monotone_batch
    array = np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags="C_CONTIGUOUS")
    function.argtypes = [
        array, array, ctypes.c_size_t, ctypes.c_double,
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_size_t),
    ]
    function.restype = ctypes.c_int
    return function


class ConditionalShockKernel:
    """Piecewise-smooth batched Marshall--Olkin likelihood kernel."""

    def __init__(self, device: torch.device):
        states = enumerate_states().astype(bool)
        permutations = list(itertools.permutations(range(6)))
        powers = 6 ** np.arange(6)
        code_to_index = np.full(6**6, -1, dtype=np.int64)
        eligibility = np.zeros((len(permutations), 64, 6, 63), dtype=np.float32)
        shock_codes = np.arange(1, 64)
        for permutation_index, permutation in enumerate(permutations):
            code_to_index[int(np.asarray(permutation) @ powers)] = permutation_index
            for state in range(1, 64):
                prior = 0
                for analyte in permutation:
                    bit = 1 << analyte
                    if state & bit:
                        eligible = ((shock_codes & bit) != 0) & (
                            (shock_codes & prior) == 0
                        )
                        eligibility[permutation_index, state, analyte, eligible] = 1.0
                        prior |= bit
        self.device = device
        self.shocks = torch.as_tensor(SHOCKS.T, dtype=torch.float32, device=device)
        self.mobius = torch.as_tensor(MOBIUS, dtype=torch.float32, device=device)
        self.eligibility = torch.as_tensor(eligibility, device=device).reshape(
            len(permutations), 64 * 6, 63
        )
        self.code_to_index = torch.as_tensor(code_to_index, device=device)
        self.powers = torch.as_tensor(powers, device=device)

    def observed_probability(
        self, logits: torch.Tensor, marginal: torch.Tensor, codes: torch.Tensor
    ) -> torch.Tensor:
        rates = torch.softmax(logits, dim=0)
        analyte_rate = self.shocks @ rates
        threshold = -torch.log(torch.clamp(marginal, 1e-8, 1 - 1e-8)) / analyte_rate
        order = torch.argsort(-threshold, dim=1)
        permutation_code = (order * self.powers[None, :]).sum(dim=1)
        permutation_index = self.code_to_index[permutation_code]
        # Compute coefficients for every one of the 6! threshold orderings in
        # one batched matrix multiplication, then gather the ordering relevant
        # to each site. This removes hundreds of small GPU kernel launches per
        # minibatch while retaining the exact piecewise Marshall--Olkin kernel.
        all_coefficients = (self.eligibility @ rates).reshape(-1, 64, 6)
        coefficient = all_coefficients[permutation_index]
        moments = torch.exp(-(threshold[:, None, :] * coefficient).sum(dim=2))
        probability = moments @ self.mobius.T
        return probability.gather(1, codes[:, None]).squeeze(1)


def conditional_nll(
    kernel: ConditionalShockKernel,
    logits: torch.Tensor,
    marginal: torch.Tensor,
    codes: torch.Tensor,
) -> torch.Tensor:
    observed = torch.clamp(kernel.observed_probability(logits, marginal, codes), min=0.0)
    scored = (1.0 - LOG_SCORE_EPSILON) * observed + LOG_SCORE_EPSILON / 64.0
    return -torch.log(scored).mean()


def initial_logits(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    shock_order = SHOCKS.sum(axis=1)
    if seed == SEEDS[0]:
        # Near-independence start, but not so close to the simplex boundary
        # that shared-rate gradients vanish numerically.
        initial = np.where(shock_order == 1, 0.0, -6.0)
    elif seed == SEEDS[1]:
        # Diffuse shared-shock start probes a strongly dependent basin.
        initial = np.zeros(63)
    else:
        # Low-order hierarchy favors singles and pairs while leaving higher
        # order shocks available to the likelihood.
        initial = np.select(
            [shock_order == 1, shock_order == 2], [0.0, -1.5], default=-3.0
        )
    initial = initial.astype(np.float32)
    initial += rng.normal(0.0, 0.25, size=63).astype(np.float32)
    return initial


def fit_conditional_shock(
    marginal: np.ndarray,
    codes: np.ndarray,
    device: torch.device,
    maximum_epochs: int,
    batch_size: int,
) -> dict:
    kernel = ConditionalShockKernel(device)
    x = torch.as_tensor(marginal, dtype=torch.float32, device=device)
    y = torch.as_tensor(codes, dtype=torch.long, device=device)
    attempts = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        logits = torch.nn.Parameter(
            torch.as_tensor(initial_logits(seed), dtype=torch.float32, device=device)
        )
        optimizer = torch.optim.Adam([logits], lr=0.025)
        generator = torch.Generator(device=device).manual_seed(seed + 1000)
        best_loss = np.inf
        best_logits = None
        best_epoch = 0
        stale = 0
        for epoch in range(1, maximum_epochs + 1):
            order = torch.randperm(len(x), generator=generator, device=device)
            epoch_sum = 0.0
            for start in range(0, len(order), batch_size):
                rows = order[start : start + batch_size]
                loss = conditional_nll(kernel, logits, x[rows], y[rows])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_([logits], 10.0)
                optimizer.step()
                epoch_sum += float(loss.detach()) * len(rows)
            # Evaluate the current endpoint parameters on the complete
            # training set; the minibatch accumulation above mixes parameter
            # values from different optimizer steps and is not a valid model-
            # selection objective.
            with torch.no_grad():
                exact_sum = 0.0
                for start in range(0, len(x), batch_size):
                    current_rows = slice(start, start + batch_size)
                    exact_sum += float(
                        conditional_nll(
                            kernel, logits, x[current_rows], y[current_rows]
                        )
                    ) * len(x[current_rows])
            current = exact_sum / len(x)
            if current < best_loss - 1e-7:
                best_loss = current
                best_logits = logits.detach().clone()
                best_epoch = epoch
                stale = 0
            else:
                stale += 1
            if stale >= 25:
                break
        if best_logits is None:
            raise RuntimeError("No finite conditional shared-shock fit")
        with torch.no_grad():
            losses = []
            for start in range(0, len(x), batch_size):
                losses.append(
                    float(
                        conditional_nll(
                            kernel,
                            best_logits,
                            x[start : start + batch_size],
                            y[start : start + batch_size],
                        )
                    )
                    * len(x[start : start + batch_size])
                )
            exact_objective = sum(losses) / len(x)
        rates = torch.softmax(best_logits, dim=0).cpu().numpy().astype(float)
        attempts.append(
            {
                "seed": seed,
                "objective": exact_objective,
                "best_training_objective": best_loss,
                "best_epoch": best_epoch,
                "epochs_run": epoch,
                "rates": rates,
            }
        )
    selected = min(attempts, key=lambda item: item["objective"])
    stability_marginal = marginal[: min(512, len(marginal))]
    induced = [shock_probability(stability_marginal, item["rates"]) for item in attempts]
    stability = []
    for left, right in itertools.combinations(range(len(attempts)), 2):
        left_rate = attempts[left]["rates"]
        right_rate = attempts[right]["rates"]
        total_variation = 0.5 * np.abs(induced[left] - induced[right]).sum(axis=1)
        stability.append(
            {
                "seed_left": attempts[left]["seed"],
                "seed_right": attempts[right]["seed"],
                "rate_l1": float(np.abs(left_rate - right_rate).sum()),
                "rate_cosine": float(
                    (left_rate @ right_rate)
                    / np.sqrt((left_rate @ left_rate) * (right_rate @ right_rate))
                ),
                "mean_induced_distribution_total_variation": float(total_variation.mean()),
                "maximum_induced_distribution_total_variation": float(total_variation.max()),
            }
        )
    rates = selected.pop("rates")
    return {
        "rates": rates,
        "fit": {
            **selected,
            "success": True,
            "multistart": [
                {
                    key: value
                    for key, value in attempt.items()
                    if key != "rates"
                }
                for attempt in attempts
            ],
            "multistart_stability": {
                "sites_used_for_induced_distribution_comparison": len(stability_marginal),
                "pairwise": stability,
            },
            "site_threshold_observations": int(len(marginal)),
            "fit_alphas": ANCHORS,
            "objective_definition": "sitewise conditional mean log score",
            "nominal_simplex_coordinates": 62,
            "active_rates_above_1e4": int(np.sum(rates > 1e-4)),
            "inverse_simpson_rate_concentration": float(1.0 / np.sum(rates**2)),
            "rate_concentration_is_not_dimension": True,
            "rates": {
                "_".join(CHEMICALS[index] for index in np.flatnonzero(shock)): float(rate)
                for shock, rate in zip(SHOCKS, rates)
            },
        },
    }


def observed_component_likelihood(
    marginal: np.ndarray, codes: np.ndarray, definitions: list[dict]
) -> np.ndarray:
    likelihood = np.empty((len(marginal), len(definitions)), dtype=float)
    for index, definition in enumerate(definitions):
        probability = partition_probability(marginal, definition["blocks"])
        likelihood[:, index] = probability[np.arange(len(marginal)), codes]
    return likelihood


def fit_conditional_partition(likelihood: np.ndarray, definitions: list[dict]) -> dict:
    scored = (1.0 - LOG_SCORE_EPSILON) * likelihood + LOG_SCORE_EPSILON / 64.0

    def unpack(theta: np.ndarray) -> np.ndarray:
        logits = np.concatenate([[0.0], theta])
        return np.exp(logits - logsumexp(logits))

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        weight = unpack(theta)
        mixture = scored @ weight
        loss = -float(np.log(mixture).mean())
        gradient_weight = -(scored / mixture[:, None]).mean(axis=0)
        centered = gradient_weight - float(gradient_weight @ weight)
        return loss, weight[1:] * centered[1:]

    result = minimize(
        objective,
        np.zeros(len(definitions) - 1),
        jac=True,
        method="L-BFGS-B",
        bounds=[(-20.0, 20.0)] * (len(definitions) - 1),
        options={"maxiter": 3000, "ftol": 1e-13, "gtol": 1e-9},
    )
    weights = unpack(result.x)
    return {
        "weights": weights,
        "fit": {
            "success": bool(result.success),
            "message": str(result.message),
            "objective": float(result.fun),
            "iterations": int(result.nit),
            "site_threshold_observations": len(likelihood),
            "fit_alphas": ANCHORS,
            "weights": {
                definition["name"]: float(weight)
                for definition, weight in zip(definitions, weights)
            },
        },
    }


def pair_dependence_audit(labels: np.ndarray, marginal: np.ndarray) -> list[dict]:
    rows = []
    for left, right in itertools.combinations(range(6), 2):
        observed_joint = float(np.mean(labels[:, left] * labels[:, right]))
        independent_joint = float(np.mean(marginal[:, left] * marginal[:, right]))
        rows.append(
            {
                "chemical_left": CHEMICALS[left],
                "chemical_right": CHEMICALS[right],
                "observed_joint": observed_joint,
                "conditional_independence_joint": independent_joint,
                "joint_minus_product": observed_joint - independent_joint,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--test-predictions", type=Path, required=True)
    parser.add_argument("--test-prediction-model", required=True)
    parser.add_argument("--nested-training-margins", type=Path, required=True)
    parser.add_argument("--shared-library", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--geometry", default="log_mrl", choices=("log_mrl", "concentration_linear"))
    parser.add_argument("--maximum-epochs", type=int, default=180)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--outer-folds", default="1,2,3,4")
    parser.add_argument("--margin-logit-temperature", type=float, default=1.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transport = load_transport(args.shared_library)
    cohort = pd.read_csv(
        args.cohort,
        dtype={"location_id": str, "pws_id": str, "state": str, "region": str},
    )
    prediction_columns = ["location_id", "chemical", "model", "mu", "sigma", "probability_hidden"]
    test_prediction = pd.read_csv(
        args.test_predictions, usecols=prediction_columns, dtype={"location_id": str}
    )
    test_prediction = test_prediction[
        test_prediction["model"].eq(args.test_prediction_model)
    ].copy()
    nested = pd.read_csv(args.nested_training_margins, dtype={"location_id": str})
    definitions = component_definitions()
    states = enumerate_states()
    state_count = states.sum(axis=1)
    metric_rows, site_parts, fit_records, pair_audits, coupling_records = [], [], [], [], []
    maximum_residual = {"conditional_shared_shock": 0.0, "conditional_partition17": 0.0, "factorized": 0.0}
    minimum_probability = {key: 1.0 for key in maximum_residual}
    outer_folds = [int(value) for value in args.outer_folds.split(",") if value]

    for outer_fold in outer_folds:
        validation_fold = 1 + (outer_fold % 4)
        train = cohort[~cohort["state_fold"].isin([outer_fold, validation_fold])].copy()
        test = cohort[cohort["state_fold"].eq(outer_fold)].copy()
        train_prediction = nested[nested["outer_joint_fold"].eq(outer_fold)].copy()
        train_lookup = prediction_lookup(train_prediction, train)
        test_lookup = prediction_lookup(test_prediction, test)
        train_margins, train_codes, partition_likelihoods = [], [], []
        for alpha in ANCHORS:
            marginal = temperature_transform(
                margins_at(train, train_lookup, alpha, args.geometry),
                args.margin_logit_temperature,
            )
            labels = labels_at(train, alpha, args.geometry)
            codes = labels @ (2 ** np.arange(6))
            train_margins.append(marginal)
            train_codes.append(codes)
            partition_likelihoods.append(
                observed_component_likelihood(marginal, codes, definitions)
            )
            for row in pair_dependence_audit(labels, marginal):
                pair_audits.append({"outer_fold": outer_fold, "alpha": alpha, **row})
        stacked_margins = np.concatenate(train_margins)
        stacked_codes = np.concatenate(train_codes)
        shock = fit_conditional_shock(
            stacked_margins,
            stacked_codes,
            device,
            args.maximum_epochs,
            args.batch_size,
        )
        rates = shock.pop("rates")
        partition = fit_conditional_partition(
            np.concatenate(partition_likelihoods), definitions
        )
        weights = partition.pop("weights")
        fit_records.append(
            {
                "outer_fold": outer_fold,
                "validation_fold": validation_fold,
                "train_sites": len(train),
                "training_margin_design": "outer-test-clean inner cross-fitting",
                "shared_shock": shock["fit"],
                "partition17": partition["fit"],
            }
        )

        previous = {"conditional_shared_shock": None, "conditional_partition17": None}
        for alpha in GRID:
            marginal = temperature_transform(
                margins_at(test, test_lookup, alpha, args.geometry),
                args.margin_logit_temperature,
            )
            probabilities = {
                "factorized": factorized_probability(marginal),
                "conditional_partition17": mixture_probability(marginal, definitions, weights),
                "conditional_shared_shock": shock_probability(marginal, rates),
            }
            for model, probability in probabilities.items():
                minimum_probability[model] = min(minimum_probability[model], float(probability.min()))
                maximum_residual[model] = max(
                    maximum_residual[model],
                    float(np.max(np.abs(probability @ states - marginal))),
                )
                if alpha in QUERIES:
                    labels = labels_at(test, alpha, args.geometry)
                    codes = labels @ (2 ** np.arange(6))
                    aggregate, site, _ = evaluate_threshold(
                        probability,
                        labels,
                        test,
                        model=model,
                        alpha=alpha,
                        outer_fold=outer_fold,
                    )
                    metric_rows.append(aggregate)
                    site["pws_id"] = test["pws_id"].astype(str).to_numpy()
                    site["observed_state_probability_raw"] = probability[
                        np.arange(len(test)), codes
                    ]
                    site_parts.append(site)
            for model in previous:
                probability = probabilities[model]
                if previous[model] is not None:
                    maximum = ctypes.c_double()
                    failures = ctypes.c_size_t()
                    code = transport(
                        previous[model],
                        probability,
                        len(test),
                        1e-9,
                        ctypes.byref(maximum),
                        ctypes.byref(failures),
                    )
                    if code:
                        raise RuntimeError(f"Transport library returned {code}")
                    coupling_records.append(
                        {
                            "outer_fold": outer_fold,
                            "model": model,
                            "alpha_lower": alpha - 0.05,
                            "alpha_upper": alpha,
                            "sites": len(test),
                            "infeasible_sites": int(failures.value),
                            "maximum_untransported_mass": float(maximum.value),
                        }
                    )
                previous[model] = probability
        print(
            f"fold={outer_fold} shock={shock['fit']['objective']:.6f} "
            f"partition={partition['fit']['objective']:.6f} complete",
            flush=True,
        )

    metrics = pd.DataFrame(metric_rows)
    sites = pd.concat(site_parts, ignore_index=True)
    integrated = sites.groupby(["location_id", "pws_id", "state", "model"], as_index=False)[
        list(LOSS_COLUMNS)
    ].mean()
    coupling = pd.DataFrame(coupling_records)
    conditional_coupling = {
        str(model): int(value)
        for model, value in coupling.groupby("model")["infeasible_sites"].sum().items()
    }
    decision = {
        "status": (
            "CONDITIONAL JOINT REFIT COMPLETE"
            if sum(conditional_coupling.values()) == 0
            else "CONDITIONAL JOINT COHERENCE FAILURE"
        ),
        "analysis_role": "post-failure exploratory repair evaluated under outer-test-clean nested margins",
        "estimand": "sitewise conditional P(Y_i | p_i) composite log likelihood across three anchors fixed for the reported analysis",
        "geometry": args.geometry,
        "margin_logit_temperature": args.margin_logit_temperature,
        "test_prediction_model": args.test_prediction_model,
        "training_margins": str(args.nested_training_margins),
        "device": str(device),
        "proper_log_score": {
            "uniform_contamination_epsilon": LOG_SCORE_EPSILON,
            "eventwise_probability_floor_used": False,
        },
        "shared_shock_parameterization": {
            "nominal_nonempty_subset_rates": 63,
            "nominal_simplex_coordinates": 62,
            "effective_dimension_claimed": False,
        },
        "fit_records": fit_records,
        "integrated_metrics": integrated.groupby("model")[list(LOSS_COLUMNS)].mean().to_dict(orient="index"),
        "minimum_state_probability_after_numerical_cleanup": minimum_probability,
        "maximum_supplied_margin_residual": maximum_residual,
        "boolean_lattice_audit": {
            "infeasible_comparisons_by_model": conditional_coupling,
            "maximum_untransported_mass": float(coupling["maximum_untransported_mass"].max()),
            "tolerance": 1e-9,
        },
        "interpretation_boundary": (
            "The fitted weights/rates describe predictive residual dependence after conditioning "
            "on cross-fitted site margins. They are not identified chemical source mechanisms."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output_dir / "threshold_metrics.csv", index=False)
    sites.to_csv(args.output_dir / "site_threshold_losses.csv.gz", index=False, compression="gzip")
    integrated.to_csv(args.output_dir / "integrated_site_losses.csv.gz", index=False, compression="gzip")
    pd.DataFrame(pair_audits).to_csv(args.output_dir / "training_pair_dependence_audit.csv", index=False)
    coupling.to_csv(args.output_dir / "boolean_lattice_audit.csv", index=False)
    (args.output_dir / "decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    print(json.dumps(decision, indent=2), flush=True)


if __name__ == "__main__":
    main()

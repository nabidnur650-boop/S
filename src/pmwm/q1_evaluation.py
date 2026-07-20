from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import t

from .memory import memory_forecast
from .q1_baselines import BASELINE_CLASSES
from .q1_common import Q1_ROOT, ensure_q1_directories, q1_json, verify_protocol_lock
from .q1_features import TARGET_FEATURES, load_q1_features
from .q1_memory import load_q1_calibration, load_q1_memory
from .q1_model import load_q1_predictions


def _persistence(prediction: dict[str, np.ndarray], config: dict[str, Any]) -> np.ndarray:
    store = load_q1_features()
    origin = prediction["origin"]
    site = prediction["site"]
    current = store.target_seasonal[origin, site] + store.target_z[origin, site] * store.target_scale[site]
    values = []
    for horizon in config["model"]["horizons"]:
        seasonal = store.target_seasonal[origin + int(horizon), site]
        values.append((current - seasonal) / store.target_scale[site])
    return np.stack(values, axis=1).astype(np.float32)


def _basic_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = predicted - actual
    actual_flat = actual.ravel().astype(np.float64)
    predicted_flat = predicted.ravel().astype(np.float64)
    actual_centered = actual_flat - actual_flat.mean()
    predicted_centered = predicted_flat - predicted_flat.mean()
    denominator = np.sqrt(np.square(actual_centered).sum() * np.square(predicted_centered).sum())
    correlation = float(np.sum(actual_centered * predicted_centered) / denominator) if denominator > 0 else np.nan
    return {
        "rmse_z": float(np.sqrt(np.mean(np.square(error)))),
        "mae_z": float(np.mean(np.abs(error))),
        "bias_z": float(np.mean(error)),
        "acc": correlation,
    }


def _bh_adjust(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(p)
    ranked = p[order]
    adjusted_ranked = np.minimum.accumulate((ranked * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = np.clip(adjusted_ranked, 0, 1)
    return adjusted


def _block_effects(
    predictions: list[dict[str, np.ndarray]],
    memory_means: list[np.ndarray],
    block_days: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    reference = predictions[0]
    unique_origin = np.unique(reference["origin"])
    unique_site = np.unique(reference["site"])
    n_origin, n_site = len(unique_origin), len(unique_site)
    block_id = unique_origin // (4 * block_days)
    unique_block = np.unique(block_id)
    tensor = np.empty((len(predictions), len(unique_block), n_site, 4, 4), dtype=np.float64)
    for seed_index, (prediction, memory_mean) in enumerate(zip(predictions, memory_means)):
        if not np.array_equal(prediction["origin"], reference["origin"]) or not np.array_equal(
            prediction["site"], reference["site"]
        ):
            raise ValueError("Seed predictions are not aligned")
        difference = (
            np.square(prediction["target_z"] - prediction["mean"])
            - np.square(prediction["target_z"] - memory_mean)
        ).reshape(n_origin, n_site, 4, 4)
        for block_index, block in enumerate(unique_block):
            tensor[seed_index, block_index] = difference[block_id == block].mean(axis=0)
    return tensor, unique_block, unique_site


def _bootstrap_effects(
    block_effect: np.ndarray,
    replicates: int,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    _, n_blocks, n_sites, n_horizons, n_targets = block_effect.shape
    draws = np.empty((replicates, n_horizons, n_targets), dtype=np.float64)
    for replicate in range(replicates):
        sampled_blocks = rng.integers(0, n_blocks, n_blocks)
        sampled_sites = rng.integers(0, n_sites, n_sites)
        draws[replicate] = block_effect[:, sampled_blocks][:, :, sampled_sites].mean(axis=(0, 1, 2))
    rows = []
    raw_p = []
    for horizon in range(n_horizons):
        for target in range(n_targets):
            values = draws[:, horizon, target]
            probability_positive = float((values > 0).mean())
            p_value = min(1.0, 2 * min(probability_positive, 1 - probability_positive))
            raw_p.append(p_value)
            rows.append(
                {
                    "horizon_index": horizon,
                    "target_index": target,
                    "mse_reduction_z": float(block_effect[..., horizon, target].mean()),
                    "ci_lower": float(np.quantile(values, 0.025)),
                    "ci_upper": float(np.quantile(values, 0.975)),
                    "bootstrap_probability_positive": probability_positive,
                    "p_value": p_value,
                }
            )
    adjusted = _bh_adjust(np.asarray(raw_p))
    for row, value in zip(rows, adjusted):
        row["p_value_bh"] = float(value)
        row["significant_positive_bh"] = bool(row["ci_lower"] > 0 and value < 0.05)
    aggregate_draw = draws.mean(axis=(1, 2))
    return pd.DataFrame(rows), aggregate_draw


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def run_q1_evaluation(config: dict[str, Any], force: bool = False) -> Path:
    ensure_q1_directories()
    lock = verify_protocol_lock()
    output = Q1_ROOT / "results" / "summary.json"
    if output.exists() and not force:
        return output
    seeds = [int(value) for value in config["model"]["seeds"]]
    predictions: list[dict[str, np.ndarray]] = []
    full_forecasts: list[dict[str, np.ndarray]] = []
    uniform_forecasts: list[dict[str, np.ndarray]] = []
    reservoir_forecasts: list[dict[str, np.ndarray]] = []
    for seed in seeds:
        prediction = load_q1_predictions("confirmatory", seed)
        predictions.append(prediction)
        for kind, collection in [
            ("full", full_forecasts), ("uniform", uniform_forecasts), ("reservoir", reservoir_forecasts)
        ]:
            collection.append(
                memory_forecast(
                    prediction,
                    load_q1_memory(seed, kind),
                    load_q1_calibration(seed, kind),
                    int(config["memory"]["top_k"]),
                )
            )
    actual = predictions[0]["target_z"]
    persistence = _persistence(predictions[0], config)
    model_rows: list[dict[str, Any]] = []
    detailed_rows: list[dict[str, Any]] = []
    model_predictions: dict[str, list[np.ndarray]] = {
        "Neural backbone": [prediction["mean"] for prediction in predictions],
        "PMWM full": [forecast["mean"] for forecast in full_forecasts],
        "Uniform consolidated memory": [forecast["mean"] for forecast in uniform_forecasts],
        "Reservoir memory": [forecast["mean"] for forecast in reservoir_forecasts],
        "Latent analog": [forecast["analog_mean"] for forecast in full_forecasts],
    }
    for model_name, means in model_predictions.items():
        for seed, mean in zip(seeds, means):
            model_rows.append({"model": model_name, "seed": seed, **_basic_metrics(actual, mean)})
            for horizon_index, horizon_label in enumerate(config["model"]["horizon_labels"]):
                for target_index, target_name in enumerate(TARGET_FEATURES):
                    detailed_rows.append(
                        {
                            "model": model_name,
                            "seed": seed,
                            "horizon": horizon_label,
                            "target": target_name,
                            **_basic_metrics(actual[:, horizon_index, target_index], mean[:, horizon_index, target_index]),
                        }
                    )
    for model_name, mean in [("Seasonal climatology", np.zeros_like(actual)), ("Persistence", persistence)]:
        model_rows.append({"model": model_name, "seed": -1, **_basic_metrics(actual, mean)})

    # Add completed modern baselines without allowing missing experiments to be silently represented.
    for name in BASELINE_CLASSES:
        baseline_means = []
        for seed in seeds[: int(config["evaluation"]["minimum_neural_baseline_seeds"])]:
            path = Q1_ROOT / "artifacts" / f"predictions_{name}_confirmatory_seed{seed}.npz"
            if path.exists():
                prediction = _load_npz(path)
                baseline_means.append(prediction["mean"])
                model_rows.append({"model": name, "seed": seed, **_basic_metrics(actual, prediction["mean"])})
        if baseline_means:
            model_predictions[name] = baseline_means
    ridge_path = Q1_ROOT / "artifacts" / "predictions_ridge_confirmatory.npz"
    if ridge_path.exists():
        ridge = _load_npz(ridge_path)["mean"]
        model_rows.append({"model": "Ridge-AR", "seed": -1, **_basic_metrics(actual, ridge)})

    seed_metrics = pd.DataFrame(model_rows)
    seed_metrics.to_csv(Q1_ROOT / "results" / "tables" / "model_seed_metrics.csv", index=False)
    pd.DataFrame(detailed_rows).to_csv(Q1_ROOT / "results" / "tables" / "metrics_by_seed_horizon_target.csv", index=False)
    summary_rows = []
    for model_name, group in seed_metrics.groupby("model", sort=False):
        row: dict[str, Any] = {"model": model_name, "runs": len(group)}
        for metric in ["rmse_z", "mae_z", "bias_z", "acc"]:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_sd"] = float(group[metric].std(ddof=1)) if len(group) > 1 else np.nan
        summary_rows.append(row)
    leaderboard = pd.DataFrame(summary_rows).sort_values("rmse_z_mean")
    leaderboard.to_csv(Q1_ROOT / "results" / "tables" / "leaderboard.csv", index=False)

    block_effect, blocks, sites = _block_effects(predictions, [item["mean"] for item in full_forecasts], int(config["evaluation"]["bootstrap_block_days"]))
    effect_table, aggregate_draw = _bootstrap_effects(
        block_effect, int(config["evaluation"]["bootstrap_replicates"]), int(config["protocol"]["seed"])
    )
    effect_table["horizon"] = [config["model"]["horizon_labels"][index] for index in effect_table.horizon_index]
    effect_table["target"] = [TARGET_FEATURES[index] for index in effect_table.target_index]
    effect_table.to_csv(Q1_ROOT / "results" / "tables" / "confirmatory_effects.csv", index=False)
    np.savez_compressed(Q1_ROOT / "artifacts" / "bootstrap_draws.npz", aggregate_mse_reduction=aggregate_draw)

    seed_effect = block_effect.mean(axis=(1, 2, 3, 4))
    seed_ci_half = float(t.ppf(0.975, len(seed_effect) - 1) * seed_effect.std(ddof=1) / np.sqrt(len(seed_effect)))
    cell_effect = block_effect.mean(axis=(0, 1, 3, 4))
    cell_table = pd.DataFrame(
        {
            "site": sites,
            "cell_id": load_q1_features().cell_id[sites],
            "mse_reduction_z": cell_effect,
            "improved": cell_effect > 0,
        }
    )
    cell_table.to_csv(Q1_ROOT / "results" / "tables" / "confirmatory_cell_effects.csv", index=False)
    seed_table = pd.DataFrame({"seed": seeds, "mse_reduction_z": seed_effect, "improved": seed_effect > 0})
    seed_table.to_csv(Q1_ROOT / "results" / "tables" / "confirmatory_seed_effects.csv", index=False)

    calibration_rows = []
    for seed, prediction, forecast in zip(seeds, predictions, full_forecasts):
        sigma = np.sqrt(forecast["variance"])
        for level, z_value in [(0.5, 0.67448975), (0.8, 1.28155157), (0.9, 1.64485363), (0.95, 1.95996398)]:
            covered = np.abs(prediction["target_z"] - forecast["mean"]) <= z_value * sigma
            calibration_rows.append(
                {
                    "seed": seed,
                    "nominal": level,
                    "empirical": float(covered.mean()),
                    "mean_width_z": float((2 * z_value * sigma).mean()),
                }
            )
    pd.DataFrame(calibration_rows).to_csv(Q1_ROOT / "results" / "tables" / "calibration.csv", index=False)

    pmwm = seed_metrics[seed_metrics.model == "PMWM full"]
    backbone = seed_metrics[seed_metrics.model == "Neural backbone"]
    aggregate_ci = [float(np.quantile(aggregate_draw, 0.025)), float(np.quantile(aggregate_draw, 0.975))]
    decision = bool(
        aggregate_ci[0] > 0
        and float((seed_effect > 0).mean()) > 0.5
        and float((cell_effect > 0).mean()) > 0.5
    )
    summary = {
        "protocol_id": config["protocol"]["id"],
        "protocol_hash": lock["combined_sha256"],
        "confirmatory_opened_after_lock": True,
        "confirmatory_samples_per_seed": len(actual),
        "confirmatory_cells": len(sites),
        "model_seeds": seeds,
        "backbone_rmse_mean": float(backbone.rmse_z.mean()),
        "backbone_rmse_sd": float(backbone.rmse_z.std(ddof=1)),
        "pmwm_rmse_mean": float(pmwm.rmse_z.mean()),
        "pmwm_rmse_sd": float(pmwm.rmse_z.std(ddof=1)),
        "relative_rmse_improvement_percent": float(100 * (backbone.rmse_z.mean() - pmwm.rmse_z.mean()) / backbone.rmse_z.mean()),
        "primary_mse_reduction": float(block_effect.mean()),
        "primary_data_bootstrap_ci": aggregate_ci,
        "seed_mean_mse_reduction": float(seed_effect.mean()),
        "seed_t_interval": [float(seed_effect.mean() - seed_ci_half), float(seed_effect.mean() + seed_ci_half)],
        "positive_seed_fraction": float((seed_effect > 0).mean()),
        "positive_cell_fraction": float((cell_effect > 0).mean()),
        "significant_target_horizon_count_bh": int(effect_table.significant_positive_bh.sum()),
        "primary_hypothesis_supported": decision,
        "decision_rule": "bootstrap lower bound > 0 and majority of seeds and cells improve",
    }
    q1_json("results/summary.json", summary)
    return output

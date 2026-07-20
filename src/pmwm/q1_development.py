from __future__ import annotations

import copy
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from .common import device_name, set_seed
from .memory import MemoryBank, calibrate_memory, load_memory, memory_forecast, normalize_keys, save_memory
from .model import PersistentWorldModel
from .q1_baselines import ITransformerForecaster, TCNForecaster
from .q1_common import Q1_ROOT, ensure_q1_directories, q1_json, verify_protocol_lock
from .q1_features import Q1FeatureStore, load_q1_features
from .q1_memory import _bank_from, _latitude_band, _partition
from .q1_model import _loader, make_q1_datasets


def persistence_for(
    origins: np.ndarray,
    sites: np.ndarray,
    store: Q1FeatureStore,
    horizons: list[int],
) -> np.ndarray:
    current = store.target_seasonal[origins, sites] + store.target_z[origins, sites] * store.target_scale[sites]
    return np.stack(
        [
            (current - store.target_seasonal[origins + horizon, sites]) / store.target_scale[sites]
            for horizon in horizons
        ],
        axis=1,
    ).astype(np.float32)


def _model(config: dict[str, Any], store: Q1FeatureStore) -> PersistentWorldModel:
    cfg = config["model"]
    return PersistentWorldModel(
        n_input=store.input_z.shape[-1],
        n_target=store.target_z.shape[-1],
        n_horizons=len(cfg["horizons"]),
        static_dim=store.static.shape[-1],
        hidden_dim=int(cfg["hidden_dim"]),
        latent_dim=int(cfg["latent_dim"]),
        dropout=float(cfg["dropout"]),
    )


def _baseline_batch(
    origin: torch.Tensor,
    site: torch.Tensor,
    store: Q1FeatureStore,
    horizons: list[int],
    device: torch.device,
) -> torch.Tensor:
    values = persistence_for(origin.numpy(), site.numpy(), store, horizons)
    return torch.from_numpy(values).to(device, non_blocking=True)


@torch.no_grad()
def _validate(
    model: PersistentWorldModel,
    loader: Any,
    store: Q1FeatureStore,
    horizons: list[int],
    device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    for x, target, static, origin, site in loader:
        x, target, static = x.to(device), target.to(device), static.to(device)
        baseline = _baseline_batch(origin, site, store, horizons, device)
        correction, _, _, _ = model(x, torch.ones_like(x), static)
        total += float(torch.square(baseline + correction - target).sum())
        count += target.numel()
    return total / count


def train_residual_development(config: dict[str, Any], seed: int = 3407, force: bool = False) -> Path:
    """Development-only persistence-residual backbone; never reads the v2 confirmation split."""
    ensure_q1_directories()
    verify_protocol_lock()
    output = Q1_ROOT / "checkpoints" / f"development_residual_seed{seed}.pt"
    history = Q1_ROOT / "results" / "tables" / f"development_residual_training_seed{seed}.csv"
    if output.exists() and history.exists() and not force:
        return output
    set_seed(seed)
    store = load_q1_features()
    datasets = make_q1_datasets(config, store)
    cfg = config["model"]
    horizons = [int(value) for value in cfg["horizons"]]
    device = torch.device(device_name())
    model = _model(config, store).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"])
    )
    train_loader = _loader(datasets["train"], int(cfg["batch_size"]), True, seed)
    validation_loader = _loader(datasets["validation"], int(cfg["batch_size"]), False, seed)
    best_mse = float("inf")
    best_state = None
    stale = 0
    rows = []
    for epoch in range(1, int(cfg["epochs"]) + 1):
        model.train()
        losses = []
        for x, target, static, origin, site in train_loader:
            x, target, static = x.to(device), target.to(device), static.to(device)
            baseline = _baseline_batch(origin, site, store, horizons, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                correction, logvar, _, _ = model(x, torch.ones_like(x), static)
                error = target - (baseline + correction)
                loss = 0.5 * (logvar + torch.square(error) * torch.exp(-logvar)).mean() + 0.05 * torch.square(error).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        validation_mse = _validate(model, validation_loader, store, horizons, device)
        rows.append(
            {
                "seed": seed,
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "validation_rmse_z": math.sqrt(validation_mse),
            }
        )
        print(f"development residual seed={seed} epoch={epoch:02d} val_rmse={math.sqrt(validation_mse):.4f}", flush=True)
        if validation_mse < best_mse - 1e-4:
            best_mse = validation_mse
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= int(cfg["patience"]):
                break
    if best_state is None:
        raise RuntimeError("Residual development training failed")
    torch.save(
        {
            "state_dict": best_state,
            "seed": seed,
            "best_validation_mse": best_mse,
            "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        },
        output,
    )
    pd.DataFrame(rows).to_csv(history, index=False)
    return output


@torch.no_grad()
def predict_residual_development(
    config: dict[str, Any], split: str, seed: int = 3407, force: bool = False
) -> Path:
    if split == "confirmatory":
        raise ValueError("Development model is deliberately prohibited from reading v2 confirmatory data")
    output = Q1_ROOT / "artifacts" / f"predictions_development_residual_{split}_seed{seed}.npz"
    if output.exists() and not force:
        return output
    store = load_q1_features()
    dataset = make_q1_datasets(config, store)[split]
    model = _model(config, store)
    model.load_state_dict(torch.load(Q1_ROOT / "checkpoints" / f"development_residual_seed{seed}.pt", map_location="cpu", weights_only=False)["state_dict"])
    device = torch.device(device_name())
    model = model.to(device).eval()
    horizons = [int(value) for value in config["model"]["horizons"]]
    parts: dict[str, list[np.ndarray]] = {
        "mean": [], "logvar": [], "latent": [], "baseline": [], "target_z": [], "origin": [], "site": []
    }
    for x, target, static, origin, site in _loader(dataset, int(config["model"]["batch_size"]), False, seed):
        x, static = x.to(device), static.to(device)
        baseline = _baseline_batch(origin, site, store, horizons, device)
        correction, logvar, _, latent = model(x, torch.ones_like(x), static)
        parts["mean"].append((baseline + correction).float().cpu().numpy())
        parts["logvar"].append(logvar.float().cpu().numpy())
        parts["latent"].append(latent.float().cpu().numpy())
        parts["baseline"].append(baseline.float().cpu().numpy())
        parts["target_z"].append(target.numpy())
        parts["origin"].append(origin.numpy())
        parts["site"].append(site.numpy())
    np.savez_compressed(output, **{key: np.concatenate(value) for key, value in parts.items()})
    return output


def train_tcn_residual_development(config: dict[str, Any], seed: int = 3407, force: bool = False) -> Path:
    """Development-only TCN correction of the registered persistence baseline."""
    ensure_q1_directories()
    verify_protocol_lock()
    output = Q1_ROOT / "checkpoints" / f"development_tcn_residual_seed{seed}.pt"
    history = Q1_ROOT / "results" / "tables" / f"development_tcn_residual_training_seed{seed}.csv"
    if output.exists() and history.exists() and not force:
        return output
    set_seed(seed)
    store = load_q1_features()
    datasets = make_q1_datasets(config, store)
    cfg = config["model"]
    horizons = [int(value) for value in cfg["horizons"]]
    device = torch.device(device_name())
    model = TCNForecaster(
        n_input=store.input_z.shape[-1],
        n_horizons=len(horizons),
        n_target=store.target_z.shape[-1],
        static_dim=store.static.shape[-1],
        dropout=float(cfg["dropout"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"])
    )
    train_loader = _loader(datasets["train"], int(cfg["batch_size"]), True, seed)
    validation_loader = _loader(datasets["validation"], int(cfg["batch_size"]), False, seed)
    best_mse = float("inf")
    best_state = None
    stale = 0
    rows = []
    for epoch in range(1, int(cfg["epochs"]) + 1):
        model.train()
        losses = []
        for x, target, static, origin, site in train_loader:
            x, target, static = x.to(device), target.to(device), static.to(device)
            baseline = _baseline_batch(origin, site, store, horizons, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                correction = model(x, static)
                loss = torch.square(baseline + correction - target).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        validation_mse = 0.0
        count = 0
        model.eval()
        with torch.no_grad():
            for x, target, static, origin, site in validation_loader:
                x, target, static = x.to(device), target.to(device), static.to(device)
                baseline = _baseline_batch(origin, site, store, horizons, device)
                predicted = baseline + model(x, static)
                validation_mse += float(torch.square(predicted - target).sum())
                count += target.numel()
        validation_mse /= count
        rows.append(
            {
                "seed": seed,
                "epoch": epoch,
                "train_mse": float(np.mean(losses)),
                "validation_rmse_z": math.sqrt(validation_mse),
            }
        )
        print(f"development residual-TCN seed={seed} epoch={epoch:02d} val_rmse={math.sqrt(validation_mse):.4f}", flush=True)
        if validation_mse < best_mse - 1e-4:
            best_mse = validation_mse
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= int(cfg["patience"]):
                break
    if best_state is None:
        raise RuntimeError("Residual TCN development training failed")
    torch.save(
        {
            "state_dict": best_state,
            "seed": seed,
            "best_validation_mse": best_mse,
            "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        },
        output,
    )
    pd.DataFrame(rows).to_csv(history, index=False)
    return output


@torch.no_grad()
def predict_tcn_residual_development(
    config: dict[str, Any], split: str, seed: int = 3407, force: bool = False
) -> Path:
    if split == "confirmatory":
        raise ValueError("Development model is deliberately prohibited from reading v2 confirmatory data")
    output = Q1_ROOT / "artifacts" / f"predictions_development_tcn_residual_{split}_seed{seed}.npz"
    if output.exists() and not force:
        return output
    store = load_q1_features()
    dataset = make_q1_datasets(config, store)[split]
    horizons = [int(value) for value in config["model"]["horizons"]]
    model = TCNForecaster(
        n_input=store.input_z.shape[-1],
        n_horizons=len(horizons),
        n_target=store.target_z.shape[-1],
        static_dim=store.static.shape[-1],
        dropout=float(config["model"]["dropout"]),
    )
    checkpoint = torch.load(
        Q1_ROOT / "checkpoints" / f"development_tcn_residual_seed{seed}.pt", map_location="cpu", weights_only=False
    )
    model.load_state_dict(checkpoint["state_dict"])
    device = torch.device(device_name())
    model = model.to(device).eval()
    parts: dict[str, list[np.ndarray]] = {
        "mean": [], "logvar": [], "latent": [], "baseline": [], "target_z": [], "origin": [], "site": []
    }
    for x, target, static, origin, site in _loader(dataset, int(config["model"]["batch_size"]), False, seed):
        x, static = x.to(device), static.to(device)
        baseline = _baseline_batch(origin, site, store, horizons, device)
        correction = model(x, static)
        parts["mean"].append((baseline + correction).float().cpu().numpy())
        parts["logvar"].append(np.zeros_like(correction.float().cpu().numpy()))
        parts["latent"].append(model.encode(x).float().cpu().numpy())
        parts["baseline"].append(baseline.float().cpu().numpy())
        parts["target_z"].append(target.numpy())
        parts["origin"].append(origin.numpy())
        parts["site"].append(site.numpy())
    np.savez_compressed(output, **{key: np.concatenate(value) for key, value in parts.items()})
    return output


def _load_prediction(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _development_prediction_path(split: str, seed: int) -> Path:
    return Q1_ROOT / "artifacts" / f"predictions_development_tcn_residual_{split}_seed{seed}.npz"


def _development_memory_path(kind: str, seed: int) -> Path:
    return Q1_ROOT / "artifacts" / f"development_memory_{kind}_seed{seed}.npz"


def _development_calibration_path(kind: str, seed: int) -> Path:
    return Q1_ROOT / "artifacts" / f"development_calibration_{kind}_seed{seed}.npz"


def build_tcn_residual_memory_development(
    config: dict[str, Any], seed: int = 3407, force: bool = False
) -> list[Path]:
    """Fit matched-capacity memories without reading either confirmation set."""
    ensure_q1_directories()
    verify_protocol_lock()
    outputs = [_development_memory_path(kind, seed) for kind in ("full", "uniform", "reservoir")]
    calibration_outputs = [
        _development_calibration_path(kind, seed) for kind in ("full", "uniform", "reservoir")
    ]
    if all(path.exists() for path in outputs + calibration_outputs) and not force:
        return outputs

    for split in ("train", "validation"):
        predict_tcn_residual_development(config, split, seed, force=force)
    train = _load_prediction(_development_prediction_path("train", seed))
    validation = _load_prediction(_development_prediction_path("validation", seed))
    store = load_q1_features()
    memory_cfg = config["memory"]
    keys = normalize_keys(train["latent"])
    outcomes = train["target_z"].astype(np.float32)
    residuals = (outcomes - train["mean"]).astype(np.float32)
    scores = np.max(np.abs(outcomes), axis=(1, 2)).astype(np.float32)
    dates = pd.DatetimeIndex(store.time[train["origin"]])
    months = dates.month.to_numpy(dtype=np.int16)
    zones = _latitude_band(store.latitude[train["site"]])
    threshold = float(np.quantile(scores, float(memory_cfg["event_quantile"])))
    event_indices = np.flatnonzero(scores >= threshold)
    regular_indices = np.flatnonzero(scores < threshold)
    all_indices = np.arange(len(keys), dtype=np.int64)
    capacity = int(memory_cfg["capacity"])

    regular, _ = _partition(
        keys,
        outcomes,
        residuals,
        scores,
        months,
        zones,
        train["site"],
        train["origin"],
        regular_indices,
        int(memory_cfg["prototype_slots"]),
        int(memory_cfg["kmeans_sample"]),
        int(memory_cfg["kmeans_iterations"]),
        seed,
    )
    events, _ = _partition(
        keys,
        outcomes,
        residuals,
        scores,
        months,
        zones,
        train["site"],
        train["origin"],
        event_indices,
        int(memory_cfg["event_slots"]),
        min(60000, int(memory_cfg["kmeans_sample"])),
        int(memory_cfg["kmeans_iterations"]),
        seed + 1,
    )
    uniform, _ = _partition(
        keys,
        outcomes,
        residuals,
        scores,
        months,
        zones,
        train["site"],
        train["origin"],
        all_indices,
        capacity,
        int(memory_cfg["kmeans_sample"]),
        int(memory_cfg["kmeans_iterations"]),
        seed + 2,
    )
    rng = np.random.default_rng(seed)
    selected = rng.choice(len(keys), size=capacity, replace=False)
    global_variance = np.var(residuals, axis=0, ddof=1).astype(np.float32)
    banks = {
        "full": _bank_from([(regular, "regular"), (events, "event")]),
        "uniform": _bank_from([(uniform, "uniform")]),
        "reservoir": MemoryBank(
            keys=keys[selected],
            outcomes=outcomes[selected],
            residuals=residuals[selected],
            residual_variance=np.broadcast_to(
                global_variance[None], (capacity, *global_variance.shape)
            ).copy(),
            event_score=scores[selected],
            month=months[selected],
            zone=zones[selected],
            site=train["site"][selected].astype(np.int16),
            origin=train["origin"][selected],
            count=np.ones(capacity, dtype=np.int64),
            kind=np.asarray(["reservoir"] * capacity),
        ),
    }
    calibration_summary = {}
    for kind, bank in banks.items():
        if bank.capacity != capacity:
            raise ValueError(f"{kind} memory capacity is {bank.capacity}, expected {capacity}")
        save_memory(bank, _development_memory_path(kind, seed))
        calibration = calibrate_memory(bank, validation, config)
        np.savez_compressed(_development_calibration_path(kind, seed), **calibration)
        calibration_summary[kind] = {
            "temperature": float(calibration["temperature"]),
            "gate_mean": float(calibration["gate"].mean()),
            "validation_mse_z": float(calibration["validation_mse_z"]),
        }
    q1_json(
        f"artifacts/development_memory_seed{seed}_manifest.json",
        {
            "seed": seed,
            "training_contexts": len(keys),
            "capacity": capacity,
            "compression_ratio": len(keys) / capacity,
            "event_threshold_z": threshold,
            "event_pool": len(event_indices),
            "event_slots": int(memory_cfg["event_slots"]),
            "matched_capacity_ablation": True,
            "calibration": calibration_summary,
        },
    )
    return outputs


def evaluate_tcn_residual_development(
    config: dict[str, Any], seed: int = 3407, force: bool = False
) -> Path:
    """Write the development-only selection table used to freeze the v3 candidate."""
    output = Q1_ROOT / "results" / "tables" / f"development_selection_seed{seed}.csv"
    if output.exists() and not force:
        return output
    predict_tcn_residual_development(config, "development_spatial", seed, force=force)
    build_tcn_residual_memory_development(config, seed, force=False)
    prediction = _load_prediction(_development_prediction_path("development_spatial", seed))
    actual = prediction["target_z"]
    candidates: dict[str, np.ndarray] = {
        "Persistence": prediction["baseline"],
        "Persistence-residual TCN": prediction["mean"],
        "Seasonal climatology": np.zeros_like(actual),
    }
    registered = {
        "Neural backbone": Q1_ROOT / "artifacts" / f"predictions_development_spatial_seed{seed}.npz",
        "DLinear": Q1_ROOT / "artifacts" / f"predictions_dlinear_development_spatial_seed{seed}.npz",
        "TCN": Q1_ROOT / "artifacts" / f"predictions_tcn_development_spatial_seed{seed}.npz",
        "PatchTST-style": Q1_ROOT / "artifacts" / f"predictions_patchtst_development_spatial_seed{seed}.npz",
        "iTransformer-style": Q1_ROOT / "artifacts" / f"predictions_itransformer_development_spatial_seed{seed}.npz",
        "Ridge-AR": Q1_ROOT / "artifacts" / "predictions_ridge_development_spatial.npz",
    }
    for label, path in registered.items():
        if path.exists():
            baseline_prediction = _load_prediction(path)
            if not np.array_equal(baseline_prediction["origin"], prediction["origin"]) or not np.array_equal(
                baseline_prediction["site"], prediction["site"]
            ):
                raise ValueError(f"Development baseline is not aligned: {label}")
            candidates[label] = baseline_prediction["mean"]
    for kind, label in (
        ("full", "PMWM-R event-aware"),
        ("uniform", "PMWM-R uniform"),
        ("reservoir", "PMWM-R reservoir"),
    ):
        bank = load_memory(_development_memory_path(kind, seed))
        with np.load(_development_calibration_path(kind, seed), allow_pickle=False) as archive:
            calibration = {key: archive[key] for key in archive.files}
        candidates[label] = memory_forecast(
            prediction, bank, calibration, int(config["memory"]["top_k"])
        )["mean"]
    rows = []
    for name, mean in candidates.items():
        error = mean - actual
        rows.append(
            {
                "model": name,
                "seed": seed,
                "rmse_z": float(np.sqrt(np.mean(np.square(error)))),
                "mae_z": float(np.mean(np.abs(error))),
                "mse_z": float(np.mean(np.square(error))),
            }
        )
    table = pd.DataFrame(rows).sort_values("rmse_z")
    table.to_csv(output, index=False)
    return output


def augmented_retrieval_keys(
    latent: np.ndarray,
    base_mean: np.ndarray,
    sites: np.ndarray,
    origins: np.ndarray,
    store: Q1FeatureStore,
    forecast_weight: float,
    static_weight: float,
    season_weight: float,
) -> np.ndarray:
    """Build a target-free key from encoder state, forecast state, location, and season."""
    encoder = normalize_keys(latent)
    forecast = normalize_keys(base_mean.reshape(len(base_mean), -1))
    static = normalize_keys(store.static[sites])
    month = pd.DatetimeIndex(store.time[origins]).month.to_numpy()
    angle = 2 * np.pi * (month - 1) / 12
    season = np.column_stack((np.sin(angle), np.cos(angle))).astype(np.float32)
    return normalize_keys(
        np.concatenate(
            (
                encoder,
                float(forecast_weight) * forecast,
                float(static_weight) * static,
                float(season_weight) * season,
            ),
            axis=1,
        )
    )


def augment_prediction_keys(
    prediction: dict[str, np.ndarray],
    store: Q1FeatureStore,
    parameters: dict[str, float],
) -> dict[str, np.ndarray]:
    augmented = dict(prediction)
    augmented["latent"] = augmented_retrieval_keys(
        prediction["latent"],
        prediction["mean"],
        prediction["site"],
        prediction["origin"],
        store,
        parameters["forecast_weight"],
        parameters["static_weight"],
        parameters["season_weight"],
    )
    return augmented


def augment_memory_keys(
    bank: MemoryBank,
    store: Q1FeatureStore,
    parameters: dict[str, float],
) -> MemoryBank:
    base_mean = bank.outcomes - bank.residuals
    keys = augmented_retrieval_keys(
        bank.keys,
        base_mean,
        bank.site.astype(np.int64),
        bank.origin,
        store,
        parameters["forecast_weight"],
        parameters["static_weight"],
        parameters["season_weight"],
    )
    return replace(bank, keys=keys)


def _prediction_subset(prediction: dict[str, np.ndarray], stride: int) -> dict[str, np.ndarray]:
    indices = np.arange(0, len(prediction["origin"]), stride, dtype=np.int64)
    return {key: value[indices] for key, value in prediction.items()}


def _augmented_paths(kind: str, seed: int) -> tuple[Path, Path]:
    return (
        Q1_ROOT / "artifacts" / f"development_augmented_memory_{kind}_seed{seed}.npz",
        Q1_ROOT / "artifacts" / f"development_augmented_calibration_{kind}_seed{seed}.npz",
    )


def build_augmented_memories(
    config: dict[str, Any],
    seed: int,
    parameters: dict[str, float],
    force: bool = False,
) -> list[Path]:
    """Apply locked target-free key weights to every matched-capacity memory."""
    outputs = [_augmented_paths(kind, seed)[0] for kind in ("full", "uniform", "reservoir")]
    calibrations = [_augmented_paths(kind, seed)[1] for kind in ("full", "uniform", "reservoir")]
    if all(path.exists() for path in outputs + calibrations) and not force:
        return outputs
    build_tcn_residual_memory_development(config, seed, force=False)
    validation = _load_prediction(_development_prediction_path("validation", seed))
    store = load_q1_features()
    validation_augmented = augment_prediction_keys(validation, store, parameters)
    tuned_config = copy.deepcopy(config)
    tuned_config["memory"]["top_k"] = int(parameters["top_k"])
    for kind in ("full", "uniform", "reservoir"):
        bank = load_memory(_development_memory_path(kind, seed))
        augmented_bank = augment_memory_keys(bank, store, parameters)
        memory_path, calibration_path = _augmented_paths(kind, seed)
        save_memory(augmented_bank, memory_path)
        calibration = calibrate_memory(augmented_bank, validation_augmented, tuned_config)
        calibration["top_k"] = np.asarray(int(parameters["top_k"]), dtype=np.int16)
        calibration["forecast_weight"] = np.asarray(parameters["forecast_weight"], dtype=np.float32)
        calibration["static_weight"] = np.asarray(parameters["static_weight"], dtype=np.float32)
        calibration["season_weight"] = np.asarray(parameters["season_weight"], dtype=np.float32)
        np.savez_compressed(calibration_path, **calibration)
    return outputs


def run_augmented_key_development_sweep(
    config: dict[str, Any], seed: int = 3407, force: bool = False
) -> Path:
    """Select target-free retrieval-key weights on a deterministic development subsample."""
    output = Q1_ROOT / "results" / "tables" / f"development_augmented_key_sweep_seed{seed}.csv"
    selection_output = Q1_ROOT / "results" / "tables" / "development_selection_v3.csv"
    parameter_output = Q1_ROOT / "results" / "tables" / "development_augmented_key_parameters.json"
    if all(path.exists() for path in (output, selection_output, parameter_output)) and not force:
        return selection_output
    for split in ("validation", "development_spatial"):
        predict_tcn_residual_development(config, split, seed, force=False)
    build_tcn_residual_memory_development(config, seed, force=False)
    validation = _load_prediction(_development_prediction_path("validation", seed))
    development = _load_prediction(_development_prediction_path("development_spatial", seed))
    store = load_q1_features()
    validation_subset = _prediction_subset(validation, 12)
    development_subset = _prediction_subset(development, 4)
    bank = load_memory(_development_memory_path("full", seed))
    rows = []

    key_grid = [
        {"forecast_weight": forecast, "static_weight": static, "season_weight": season, "top_k": 16.0}
        for forecast in (0.15, 0.35, 0.70, 1.20)
        for static in (0.0, 0.25, 0.50)
        for season in (0.0, 0.20)
    ]
    for parameters in key_grid:
        augmented_bank = augment_memory_keys(bank, store, parameters)
        validation_augmented = augment_prediction_keys(validation_subset, store, parameters)
        development_augmented = augment_prediction_keys(development_subset, store, parameters)
        calibration = calibrate_memory(augmented_bank, validation_augmented, config)
        forecast = memory_forecast(
            development_augmented,
            augmented_bank,
            calibration,
            int(parameters["top_k"]),
        )["mean"]
        rows.append(
            {
                **parameters,
                "validation_mse_z": float(calibration["validation_mse_z"]),
                "development_rmse_z": float(
                    np.sqrt(np.mean(np.square(forecast - development_subset["target_z"])))
                ),
            }
        )
        print(
            "augmented key "
            f"forecast={parameters['forecast_weight']:.2f} static={parameters['static_weight']:.2f} "
            f"season={parameters['season_weight']:.2f} dev={rows[-1]['development_rmse_z']:.4f}",
            flush=True,
        )
    first_stage = pd.DataFrame(rows).sort_values("development_rmse_z")
    best_key = first_stage.iloc[0].to_dict()
    for top_k in (4, 8, 16, 32, 64):
        parameters = {
            "forecast_weight": float(best_key["forecast_weight"]),
            "static_weight": float(best_key["static_weight"]),
            "season_weight": float(best_key["season_weight"]),
            "top_k": float(top_k),
        }
        augmented_bank = augment_memory_keys(bank, store, parameters)
        validation_augmented = augment_prediction_keys(validation_subset, store, parameters)
        development_augmented = augment_prediction_keys(development_subset, store, parameters)
        tuned_config = copy.deepcopy(config)
        tuned_config["memory"]["top_k"] = top_k
        calibration = calibrate_memory(augmented_bank, validation_augmented, tuned_config)
        forecast = memory_forecast(development_augmented, augmented_bank, calibration, top_k)["mean"]
        rows.append(
            {
                **parameters,
                "validation_mse_z": float(calibration["validation_mse_z"]),
                "development_rmse_z": float(
                    np.sqrt(np.mean(np.square(forecast - development_subset["target_z"])))
                ),
            }
        )
    sweep = pd.DataFrame(rows).drop_duplicates(
        ["forecast_weight", "static_weight", "season_weight", "top_k"], keep="last"
    )
    sweep.sort_values("development_rmse_z").to_csv(output, index=False)
    winner = sweep.sort_values("development_rmse_z").iloc[0]
    parameters = {
        "forecast_weight": float(winner.forecast_weight),
        "static_weight": float(winner.static_weight),
        "season_weight": float(winner.season_weight),
        "top_k": int(winner.top_k),
    }
    parameter_output.write_text(json.dumps(parameters, indent=2) + "\n", encoding="utf-8")
    build_augmented_memories(config, seed, parameters, force=True)
    development_augmented = augment_prediction_keys(development, store, parameters)
    memory_path, calibration_path = _augmented_paths("full", seed)
    augmented_bank = load_memory(memory_path)
    with np.load(calibration_path, allow_pickle=False) as archive:
        calibration = {key: archive[key] for key in archive.files}
    full_forecast = memory_forecast(
        development_augmented, augmented_bank, calibration, int(parameters["top_k"])
    )["mean"]
    augmented_row = {
        "model": "PMWM-R event-aware augmented",
        "seed": seed,
        "rmse_z": float(np.sqrt(np.mean(np.square(full_forecast - development["target_z"])))),
        "mae_z": float(np.mean(np.abs(full_forecast - development["target_z"]))),
        "mse_z": float(np.mean(np.square(full_forecast - development["target_z"]))),
    }
    original = pd.read_csv(evaluate_tcn_residual_development(config, seed, force=False))
    combined = pd.concat([original, pd.DataFrame([augmented_row])], ignore_index=True).sort_values("rmse_z")
    combined.to_csv(selection_output, index=False)
    return selection_output


def _itransformer_model(config: dict[str, Any], store: Q1FeatureStore) -> ITransformerForecaster:
    return ITransformerForecaster(
        context=int(config["model"]["context_steps"]),
        n_input=store.input_z.shape[-1],
        n_horizons=len(config["model"]["horizons"]),
        n_target=store.target_z.shape[-1],
        static_dim=store.static.shape[-1],
        dropout=float(config["model"]["dropout"]),
    )


def _itransformer_latent(
    model: ITransformerForecaster, x: torch.Tensor, static: torch.Tensor
) -> torch.Tensor:
    tokens = model.encode(x)
    return model.head[1](model.head[0](torch.cat((tokens.flatten(1), static), dim=1)))


def train_itransformer_residual_development(
    config: dict[str, Any], seed: int = 3407, force: bool = False
) -> Path:
    """Development-only inverted-Transformer correction of physical persistence."""
    ensure_q1_directories()
    verify_protocol_lock()
    output = Q1_ROOT / "checkpoints" / f"development_itransformer_residual_seed{seed}.pt"
    history = (
        Q1_ROOT
        / "results"
        / "tables"
        / f"development_itransformer_residual_training_seed{seed}.csv"
    )
    if output.exists() and history.exists() and not force:
        return output
    set_seed(seed)
    store = load_q1_features()
    datasets = make_q1_datasets(config, store)
    model_cfg = config["model"]
    horizons = [int(value) for value in model_cfg["horizons"]]
    device = torch.device(device_name())
    model = _itransformer_model(config, store).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(model_cfg["learning_rate"]),
        weight_decay=float(model_cfg["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(model_cfg["epochs"])
    )
    train_loader = _loader(datasets["train"], int(model_cfg["batch_size"]), True, seed)
    validation_loader = _loader(
        datasets["validation"], int(model_cfg["batch_size"]), False, seed
    )
    best_mse = float("inf")
    best_state = None
    stale = 0
    rows = []
    for epoch in range(1, int(model_cfg["epochs"]) + 1):
        model.train()
        losses = []
        for x, target, static, origin, site in train_loader:
            x, target, static = x.to(device), target.to(device), static.to(device)
            baseline = _baseline_batch(origin, site, store, horizons, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                correction = model(x, static)
                loss = torch.square(baseline + correction - target).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        scheduler.step()
        model.eval()
        squared = 0.0
        count = 0
        with torch.no_grad():
            for x, target, static, origin, site in validation_loader:
                x, target, static = x.to(device), target.to(device), static.to(device)
                baseline = _baseline_batch(origin, site, store, horizons, device)
                squared += float(torch.square(baseline + model(x, static) - target).sum())
                count += target.numel()
        validation_mse = squared / count
        rows.append(
            {
                "seed": seed,
                "epoch": epoch,
                "train_mse": float(np.mean(losses)),
                "validation_rmse_z": math.sqrt(validation_mse),
            }
        )
        print(
            f"development residual-iTransformer seed={seed} epoch={epoch:02d} "
            f"val_rmse={math.sqrt(validation_mse):.4f}",
            flush=True,
        )
        if validation_mse < best_mse - 1e-4:
            best_mse = validation_mse
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= int(model_cfg["patience"]):
                break
    if best_state is None:
        raise RuntimeError("Residual iTransformer development training failed")
    torch.save(
        {
            "state_dict": best_state,
            "seed": seed,
            "best_validation_mse": best_mse,
            "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        },
        output,
    )
    pd.DataFrame(rows).to_csv(history, index=False)
    return output


def _itransformer_prediction_path(split: str, seed: int) -> Path:
    return (
        Q1_ROOT
        / "artifacts"
        / f"predictions_development_itransformer_residual_{split}_seed{seed}.npz"
    )


@torch.no_grad()
def predict_itransformer_residual_development(
    config: dict[str, Any], split: str, seed: int = 3407, force: bool = False
) -> Path:
    if split == "confirmatory":
        raise ValueError("Development model cannot read the opened confirmation split")
    output = _itransformer_prediction_path(split, seed)
    if output.exists() and not force:
        return output
    store = load_q1_features()
    dataset = make_q1_datasets(config, store)[split]
    horizons = [int(value) for value in config["model"]["horizons"]]
    model = _itransformer_model(config, store)
    checkpoint = torch.load(
        Q1_ROOT / "checkpoints" / f"development_itransformer_residual_seed{seed}.pt",
        map_location="cpu",
        weights_only=False,
    )
    model.load_state_dict(checkpoint["state_dict"])
    device = torch.device(device_name())
    model = model.to(device).eval()
    parts: dict[str, list[np.ndarray]] = {
        "mean": [],
        "logvar": [],
        "latent": [],
        "baseline": [],
        "target_z": [],
        "origin": [],
        "site": [],
    }
    for x, target, static, origin, site in _loader(
        dataset, int(config["model"]["batch_size"]), False, seed
    ):
        x, static = x.to(device), static.to(device)
        baseline = _baseline_batch(origin, site, store, horizons, device)
        correction = model(x, static)
        parts["mean"].append((baseline + correction).float().cpu().numpy())
        parts["logvar"].append(np.zeros_like(correction.float().cpu().numpy()))
        parts["latent"].append(_itransformer_latent(model, x, static).float().cpu().numpy())
        parts["baseline"].append(baseline.float().cpu().numpy())
        parts["target_z"].append(target.numpy())
        parts["origin"].append(origin.numpy())
        parts["site"].append(site.numpy())
    np.savez_compressed(output, **{key: np.concatenate(value) for key, value in parts.items()})
    return output


def _itransformer_memory_path(kind: str, seed: int) -> tuple[Path, Path]:
    return (
        Q1_ROOT / "artifacts" / f"development_itransformer_memory_{kind}_seed{seed}.npz",
        Q1_ROOT
        / "artifacts"
        / f"development_itransformer_calibration_{kind}_seed{seed}.npz",
    )


def build_itransformer_residual_memories(
    config: dict[str, Any], seed: int = 3407, force: bool = False
) -> list[Path]:
    outputs = [_itransformer_memory_path(kind, seed)[0] for kind in ("full", "uniform", "reservoir")]
    calibrations = [
        _itransformer_memory_path(kind, seed)[1] for kind in ("full", "uniform", "reservoir")
    ]
    if all(path.exists() for path in outputs + calibrations) and not force:
        return outputs
    for split in ("train", "validation"):
        predict_itransformer_residual_development(config, split, seed, force=force)
    train = _load_prediction(_itransformer_prediction_path("train", seed))
    validation = _load_prediction(_itransformer_prediction_path("validation", seed))
    store = load_q1_features()
    memory_cfg = config["memory"]
    keys = normalize_keys(train["latent"])
    outcomes = train["target_z"].astype(np.float32)
    residuals = (outcomes - train["mean"]).astype(np.float32)
    scores = np.max(np.abs(outcomes), axis=(1, 2)).astype(np.float32)
    dates = pd.DatetimeIndex(store.time[train["origin"]])
    months = dates.month.to_numpy(dtype=np.int16)
    zones = _latitude_band(store.latitude[train["site"]])
    threshold = float(np.quantile(scores, float(memory_cfg["event_quantile"])))
    event_indices = np.flatnonzero(scores >= threshold)
    regular_indices = np.flatnonzero(scores < threshold)
    all_indices = np.arange(len(keys), dtype=np.int64)
    capacity = int(memory_cfg["capacity"])
    regular, _ = _partition(
        keys,
        outcomes,
        residuals,
        scores,
        months,
        zones,
        train["site"],
        train["origin"],
        regular_indices,
        int(memory_cfg["prototype_slots"]),
        int(memory_cfg["kmeans_sample"]),
        int(memory_cfg["kmeans_iterations"]),
        seed + 10,
    )
    events, _ = _partition(
        keys,
        outcomes,
        residuals,
        scores,
        months,
        zones,
        train["site"],
        train["origin"],
        event_indices,
        int(memory_cfg["event_slots"]),
        min(60000, int(memory_cfg["kmeans_sample"])),
        int(memory_cfg["kmeans_iterations"]),
        seed + 11,
    )
    uniform, _ = _partition(
        keys,
        outcomes,
        residuals,
        scores,
        months,
        zones,
        train["site"],
        train["origin"],
        all_indices,
        capacity,
        int(memory_cfg["kmeans_sample"]),
        int(memory_cfg["kmeans_iterations"]),
        seed + 12,
    )
    rng = np.random.default_rng(seed + 10)
    selected = rng.choice(len(keys), size=capacity, replace=False)
    variance = np.var(residuals, axis=0, ddof=1).astype(np.float32)
    banks = {
        "full": _bank_from([(regular, "regular"), (events, "event")]),
        "uniform": _bank_from([(uniform, "uniform")]),
        "reservoir": MemoryBank(
            keys=keys[selected],
            outcomes=outcomes[selected],
            residuals=residuals[selected],
            residual_variance=np.broadcast_to(variance[None], (capacity, *variance.shape)).copy(),
            event_score=scores[selected],
            month=months[selected],
            zone=zones[selected],
            site=train["site"][selected].astype(np.int16),
            origin=train["origin"][selected],
            count=np.ones(capacity, dtype=np.int64),
            kind=np.asarray(["reservoir"] * capacity),
        ),
    }
    calibration_summary = {}
    for kind, bank in banks.items():
        memory_path, calibration_path = _itransformer_memory_path(kind, seed)
        save_memory(bank, memory_path)
        calibration = calibrate_memory(bank, validation, config)
        np.savez_compressed(calibration_path, **calibration)
        calibration_summary[kind] = {
            "validation_mse_z": float(calibration["validation_mse_z"]),
            "gate_mean": float(calibration["gate"].mean()),
        }
    q1_json(
        f"artifacts/development_itransformer_memory_seed{seed}_manifest.json",
        {
            "seed": seed,
            "training_contexts": len(keys),
            "latent_dimension": keys.shape[1],
            "capacity": capacity,
            "compression_ratio": len(keys) / capacity,
            "event_threshold_z": threshold,
            "event_slots": int(memory_cfg["event_slots"]),
            "matched_capacity_ablation": True,
            "calibration": calibration_summary,
        },
    )
    return outputs


def evaluate_itransformer_residual_development(
    config: dict[str, Any], seed: int = 3407, force: bool = False
) -> Path:
    output = Q1_ROOT / "results" / "tables" / "development_selection_final.csv"
    if output.exists() and not force:
        return output
    predict_itransformer_residual_development(
        config, "development_spatial", seed, force=force
    )
    build_itransformer_residual_memories(config, seed, force=force)
    prediction = _load_prediction(_itransformer_prediction_path("development_spatial", seed))
    candidates = {"Persistence-residual iTransformer": prediction["mean"]}
    for kind, label in (
        ("full", "PMWM-IR event-aware"),
        ("uniform", "PMWM-IR uniform"),
        ("reservoir", "PMWM-IR reservoir"),
    ):
        memory_path, calibration_path = _itransformer_memory_path(kind, seed)
        bank = load_memory(memory_path)
        with np.load(calibration_path, allow_pickle=False) as archive:
            calibration = {key: archive[key] for key in archive.files}
        candidates[label] = memory_forecast(
            prediction, bank, calibration, int(config["memory"]["top_k"])
        )["mean"]
    rows = []
    for name, mean in candidates.items():
        error = mean - prediction["target_z"]
        rows.append(
            {
                "model": name,
                "seed": seed,
                "rmse_z": float(np.sqrt(np.mean(np.square(error)))),
                "mae_z": float(np.mean(np.abs(error))),
                "mse_z": float(np.mean(np.square(error))),
            }
        )
    previous = pd.read_csv(
        Q1_ROOT / "results" / "tables" / "development_selection_v3.csv"
    )
    combined = pd.concat((previous, pd.DataFrame(rows)), ignore_index=True).sort_values("rmse_z")
    combined.to_csv(output, index=False)
    return output

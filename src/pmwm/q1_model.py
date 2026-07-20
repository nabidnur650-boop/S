from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .common import device_name, set_seed
from .model import PersistentWorldModel
from .q1_common import Q1_ROOT, ensure_q1_directories, q1_json, verify_protocol_lock
from .q1_features import Q1FeatureStore, load_q1_features


@dataclass
class SplitSpec:
    name: str
    years: tuple[int, int]
    partition: str
    stride_days: int


class Q1SequenceDataset(Dataset):
    def __init__(
        self,
        store: Q1FeatureStore,
        origins: np.ndarray,
        sites: np.ndarray,
        context_steps: int,
        horizons: list[int],
    ) -> None:
        self.store = store
        self.origins = origins.astype(np.int64)
        self.sites = sites.astype(np.int64)
        self.context_steps = int(context_steps)
        self.horizons = [int(value) for value in horizons]

    def __len__(self) -> int:
        return len(self.origins)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        origin = int(self.origins[index])
        site = int(self.sites[index])
        start = origin - self.context_steps + 1
        x = np.array(self.store.input_z[start : origin + 1, site], dtype=np.float32, copy=True)
        y = np.stack([self.store.target_z[origin + horizon, site] for horizon in self.horizons]).astype(np.float32)
        return (
            torch.from_numpy(x),
            torch.from_numpy(y),
            torch.from_numpy(np.array(self.store.static[site], dtype=np.float32, copy=True)),
            torch.tensor(origin, dtype=torch.int64),
            torch.tensor(site, dtype=torch.int64),
        )


def _strict_origins(
    store: Q1FeatureStore,
    years: tuple[int, int],
    context: int,
    max_horizon: int,
    stride_days: int,
) -> np.ndarray:
    dates = pd.DatetimeIndex(store.time)
    start, end = years
    candidates = np.flatnonzero((dates.hour == 0) & (dates.year >= start) & (dates.year <= end))
    context_start = candidates - context + 1
    target_end = candidates + max_horizon
    valid_bounds = (context_start >= 0) & (target_end < len(dates))
    candidates = candidates[valid_bounds]
    context_start = context_start[valid_bounds]
    target_end = target_end[valid_bounds]
    inside = (dates.year[context_start] >= start) & (dates.year[target_end] <= end)
    candidates = candidates[inside]
    return candidates[:: max(1, int(stride_days))].astype(np.int64)


def _pair(origins: np.ndarray, sites: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    origin_grid, site_grid = np.meshgrid(origins, sites, indexing="ij")
    return origin_grid.ravel(), site_grid.ravel()


def make_q1_datasets(config: dict[str, Any], store: Q1FeatureStore | None = None) -> dict[str, Q1SequenceDataset]:
    store = store or load_q1_features()
    context = int(config["model"]["context_steps"])
    horizons = [int(value) for value in config["model"]["horizons"]]
    eval_stride = int(config["splits"]["evaluation_origin_stride_days"])
    specs = [
        SplitSpec("train", tuple(config["splits"]["train"]), "train", int(config["splits"]["train_origin_stride_days"])),
        SplitSpec("validation", tuple(config["splits"]["validation"]), "train", eval_stride),
        SplitSpec("development_seen", tuple(config["splits"]["development"]), "train", eval_stride),
        SplitSpec("development_spatial", tuple(config["splits"]["development"]), "development", eval_stride),
        SplitSpec("confirmatory", tuple(config["splits"]["confirmatory"]), "confirmatory", eval_stride),
    ]
    datasets: dict[str, Q1SequenceDataset] = {}
    for spec in specs:
        origins = _strict_origins(store, spec.years, context, max(horizons), spec.stride_days)
        sites = np.flatnonzero(store.partition == spec.partition)
        paired_origin, paired_site = _pair(origins, sites)
        datasets[spec.name] = Q1SequenceDataset(store, paired_origin, paired_site, context, horizons)
    return datasets


def _loader(dataset: Dataset, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
        drop_last=False,
    )


def _masked(x: torch.Tensor, probability: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    observed = torch.rand_like(x) > probability
    forced = torch.rand_like(x[:, -1]) < max(0.30, probability)
    observed[:, -1] &= ~forced
    empty = observed.sum(dim=-1) == 0
    if empty.any():
        observed[..., 0] |= empty
    mask = observed.to(x.dtype)
    return x * mask, mask, ~observed[:, -1]


def _loss(
    mean: torch.Tensor,
    logvar: torch.Tensor,
    target: torch.Tensor,
    imputation: torch.Tensor,
    actual_last: torch.Tensor,
    final_missing: torch.Tensor,
    imputation_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    squared = (target - mean) ** 2
    forecast = 0.5 * (logvar + squared * torch.exp(-logvar)).mean() + 0.05 * squared.mean()
    missing = final_missing.to(imputation.dtype)
    imputation_loss = (((imputation - actual_last) ** 2) * missing).sum() / missing.sum().clamp_min(1.0)
    return forecast + imputation_weight * imputation_loss, forecast, imputation_loss


@torch.no_grad()
def _validation(model: PersistentWorldModel, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    squared_sum = 0.0
    loss_sum = 0.0
    count = 0
    for x, y, static, _, _ in loader:
        x, y, static = x.to(device), y.to(device), static.to(device)
        mean, logvar, _, _ = model(x, torch.ones_like(x), static)
        squared_sum += float(torch.square(mean - y).sum())
        loss_sum += float((0.5 * (logvar + torch.square(y - mean) * torch.exp(-logvar))).sum())
        count += y.numel()
    return loss_sum / count, math.sqrt(squared_sum / count)


def checkpoint_path(seed: int) -> Path:
    return Q1_ROOT / "checkpoints" / f"backbone_seed{seed}.pt"


def train_q1_backbone(config: dict[str, Any], seed: int, force: bool = False) -> Path:
    ensure_q1_directories()
    lock = verify_protocol_lock()
    output = checkpoint_path(seed)
    history_path = Q1_ROOT / "results" / "tables" / f"training_seed{seed}.csv"
    if output.exists() and history_path.exists() and not force:
        return output
    if seed not in [int(value) for value in config["model"]["seeds"]]:
        raise ValueError(f"Seed {seed} was not protocol-specified")
    set_seed(seed)
    store = load_q1_features()
    datasets = make_q1_datasets(config, store)
    cfg = config["model"]
    batch_size = int(cfg["batch_size"])
    train_loader = _loader(datasets["train"], batch_size, True, seed)
    validation_loader = _loader(datasets["validation"], batch_size, False, seed)
    device = torch.device(device_name())
    model = PersistentWorldModel(
        n_input=store.input_z.shape[-1],
        n_target=store.target_z.shape[-1],
        n_horizons=len(cfg["horizons"]),
        static_dim=store.static.shape[-1],
        hidden_dim=int(cfg["hidden_dim"]),
        latent_dim=int(cfg["latent_dim"]),
        dropout=float(cfg["dropout"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"])
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(cfg["epochs"]))
    best: dict[str, torch.Tensor] | None = None
    best_loss = float("inf")
    stale = 0
    rows: list[dict[str, float]] = []
    started = time.perf_counter()
    for epoch in range(1, int(cfg["epochs"]) + 1):
        model.train()
        totals: list[float] = []
        forecasts: list[float] = []
        imputations: list[float] = []
        epoch_started = time.perf_counter()
        for x, y, static, _, _ in train_loader:
            x, y, static = x.to(device), y.to(device), static.to(device)
            masked, mask, final_missing = _masked(x, float(cfg["mask_probability"]))
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                mean, logvar, imputation, _ = model(masked, mask, static)
                total, forecast, imputation_loss = _loss(
                    mean,
                    logvar,
                    y,
                    imputation,
                    x[:, -1],
                    final_missing,
                    float(cfg["auxiliary_imputation_weight"]),
                )
            total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            totals.append(float(total.detach()))
            forecasts.append(float(forecast.detach()))
            imputations.append(float(imputation_loss.detach()))
        scheduler.step()
        val_loss, val_rmse = _validation(model, validation_loader, device)
        row = {
            "seed": seed,
            "epoch": epoch,
            "train_loss": float(np.mean(totals)),
            "train_forecast_loss": float(np.mean(forecasts)),
            "train_imputation_mse": float(np.mean(imputations)),
            "validation_nll": val_loss,
            "validation_rmse_z": val_rmse,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "epoch_seconds": time.perf_counter() - epoch_started,
        }
        rows.append(row)
        print(
            f"q1 seed={seed} epoch={epoch:02d} train={row['train_loss']:.4f} "
            f"val_rmse={val_rmse:.4f} ({row['epoch_seconds']:.1f}s)",
            flush=True,
        )
        if val_loss < best_loss - 1e-4:
            best_loss = val_loss
            best = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= int(cfg["patience"]):
                break
    if best is None:
        raise RuntimeError("No Q1 checkpoint was produced")
    torch.save(
        {
            "state_dict": best,
            "model_config": {
                "n_input": store.input_z.shape[-1],
                "n_target": store.target_z.shape[-1],
                "n_horizons": len(cfg["horizons"]),
                "static_dim": store.static.shape[-1],
                "hidden_dim": int(cfg["hidden_dim"]),
                "latent_dim": int(cfg["latent_dim"]),
                "dropout": float(cfg["dropout"]),
            },
            "seed": seed,
            "protocol_hash": lock["combined_sha256"],
            "best_validation_nll": best_loss,
        },
        output,
    )
    pd.DataFrame(rows).to_csv(history_path, index=False)
    q1_json(
        f"artifacts/model_seed{seed}_manifest.json",
        {
            "seed": seed,
            "protocol_hash": lock["combined_sha256"],
            "train_samples": len(datasets["train"]),
            "validation_samples": len(datasets["validation"]),
            "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
            "epochs": len(rows),
            "best_validation_nll": best_loss,
            "total_seconds": time.perf_counter() - started,
        },
    )
    return output


def load_q1_model(seed: int) -> PersistentWorldModel:
    checkpoint = torch.load(checkpoint_path(seed), map_location="cpu", weights_only=False)
    verify_protocol_lock()
    model = PersistentWorldModel(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["state_dict"])
    return model


@torch.no_grad()
def predict_q1_dataset(
    model: PersistentWorldModel,
    dataset: Q1SequenceDataset,
    batch_size: int,
    seed: int,
) -> dict[str, np.ndarray]:
    device = torch.device(device_name())
    model = model.to(device).eval()
    outputs: dict[str, list[np.ndarray]] = {
        "mean": [], "logvar": [], "latent": [], "origin": [], "site": [], "target_z": []
    }
    for x, y, static, origin, site in _loader(dataset, batch_size, False, seed):
        x, static = x.to(device), static.to(device)
        mean, logvar, _, latent = model(x, torch.ones_like(x), static)
        outputs["mean"].append(mean.float().cpu().numpy())
        outputs["logvar"].append(logvar.float().cpu().numpy())
        outputs["latent"].append(latent.float().cpu().numpy())
        outputs["origin"].append(origin.numpy())
        outputs["site"].append(site.numpy())
        outputs["target_z"].append(y.numpy())
    return {key: np.concatenate(values) for key, values in outputs.items()}


def encode_q1_seed(config: dict[str, Any], seed: int, force: bool = False) -> list[Path]:
    ensure_q1_directories()
    datasets = make_q1_datasets(config)
    names = ["train", "validation", "development_spatial", "confirmatory"]
    outputs = [Q1_ROOT / "artifacts" / f"predictions_{name}_seed{seed}.npz" for name in names]
    if all(path.exists() for path in outputs) and not force:
        return outputs
    model = load_q1_model(seed)
    for name, output in zip(names, outputs):
        print(f"q1 encoding seed={seed} {name}: {len(datasets[name]):,}", flush=True)
        prediction = predict_q1_dataset(model, datasets[name], int(config["model"]["batch_size"]), seed)
        np.savez_compressed(output, **prediction)
    return outputs


def load_q1_predictions(name: str, seed: int) -> dict[str, np.ndarray]:
    path = Q1_ROOT / "artifacts" / f"predictions_{name}_seed{seed}.npz"
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def train_and_encode_all_seeds(config: dict[str, Any], force: bool = False) -> None:
    for seed in [int(value) for value in config["model"]["seeds"]]:
        train_q1_backbone(config, seed, force=force)
        encode_q1_seed(config, seed, force=force)


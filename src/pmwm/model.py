from __future__ import annotations

import copy
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .common import ROOT, atomic_json, device_name, ensure_directories, set_seed
from .features import FeatureStore, actual_targets, load_features, origin_indices, paired_samples


class SiteSequenceDataset(Dataset):
    def __init__(
        self,
        store: FeatureStore,
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
        y = np.stack(
            [self.store.target_z[origin + horizon, site] for horizon in self.horizons], axis=0
        ).astype(np.float32)
        static = np.array(self.store.static[site], dtype=np.float32, copy=True)
        return (
            torch.from_numpy(x),
            torch.from_numpy(y),
            torch.from_numpy(static),
            torch.tensor(origin, dtype=torch.int64),
            torch.tensor(site, dtype=torch.int64),
        )


class MultiScaleEncoder(nn.Module):
    def __init__(self, n_features: int, static_dim: int, hidden_dim: int, latent_dim: int, dropout: float) -> None:
        super().__init__()
        self.n_features = n_features
        self.native_gru = nn.GRU(2 * n_features, hidden_dim, batch_first=True)
        self.daily_gru = nn.GRU(3 * n_features, hidden_dim, batch_first=True)
        self.weekly_mlp = nn.Sequential(
            nn.Linear(6 * n_features, 2 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.static_mlp = nn.Sequential(nn.Linear(static_dim, 16), nn.GELU(), nn.Linear(16, 16))
        self.fusion = nn.Sequential(
            nn.Linear(3 * hidden_dim + 16, 2 * latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    @staticmethod
    def _masked_statistics(x: torch.Tensor, mask: torch.Tensor, axis: int) -> tuple[torch.Tensor, ...]:
        count = mask.sum(dim=axis).clamp_min(1.0)
        mean = (x * mask).sum(dim=axis) / count
        variance = (((x - mean.unsqueeze(axis)) ** 2) * mask).sum(dim=axis) / count
        availability = count / x.shape[axis]
        return mean, torch.sqrt(variance.clamp_min(1e-6)), availability

    def forward(self, x: torch.Tensor, mask: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        batch, steps, features = x.shape
        if steps % 28 != 0 or steps % 4 != 0:
            raise ValueError("context length must be divisible by both 4 and 28")
        native = torch.cat([x[:, -16:], mask[:, -16:]], dim=-1)
        _, native_hidden = self.native_gru(native)

        days = steps // 4
        x_daily = x.reshape(batch, days, 4, features)
        m_daily = mask.reshape(batch, days, 4, features)
        daily_mean, daily_std, daily_availability = self._masked_statistics(x_daily, m_daily, axis=2)
        daily_tokens = torch.cat([daily_mean, daily_std, daily_availability], dim=-1)
        _, daily_hidden = self.daily_gru(daily_tokens)

        half = steps // 2
        first = self._masked_statistics(x[:, :half], mask[:, :half], axis=1)
        second = self._masked_statistics(x[:, half:], mask[:, half:], axis=1)
        weekly = self.weekly_mlp(torch.cat([*first, *second], dim=-1))
        static_embedding = self.static_mlp(static)
        fused = torch.cat([native_hidden[-1], daily_hidden[-1], weekly, static_embedding], dim=-1)
        return self.fusion(fused)


class PersistentWorldModel(nn.Module):
    def __init__(
        self,
        n_input: int,
        n_target: int,
        n_horizons: int,
        static_dim: int,
        hidden_dim: int,
        latent_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.n_input = n_input
        self.n_target = n_target
        self.n_horizons = n_horizons
        self.encoder = MultiScaleEncoder(n_input, static_dim, hidden_dim, latent_dim, dropout)
        self.mean_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(latent_dim, n_horizons * n_target)
        )
        self.logvar_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2), nn.GELU(), nn.Linear(latent_dim // 2, n_horizons * n_target)
        )
        self.imputation_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim), nn.GELU(), nn.Linear(latent_dim, n_input)
        )

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor, static: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        latent = self.encoder(x, mask, static)
        mean = self.mean_head(latent).reshape(-1, self.n_horizons, self.n_target)
        logvar = self.logvar_head(latent).reshape(-1, self.n_horizons, self.n_target).clamp(-7.0, 4.0)
        imputation = self.imputation_head(latent)
        return mean, logvar, imputation, latent


def make_datasets(config: dict[str, Any], store: FeatureStore | None = None) -> dict[str, SiteSequenceDataset]:
    store = store or load_features()
    model_cfg = config["model"]
    split_cfg = config["splits"]
    context = int(model_cfg["context_steps"])
    horizons = [int(value) for value in model_cfg["horizons"]]
    max_horizon = max(horizons)
    seen_sites = np.flatnonzero(~store.holdout)
    unseen_sites = np.flatnonzero(store.holdout)

    def build(name: str, start: int, end: int, sites: np.ndarray, stride: int) -> SiteSequenceDataset:
        base_origins = origin_indices(store, start, end, context, max_horizon, stride)
        origins, site_ids = paired_samples(base_origins, sites)
        return SiteSequenceDataset(store, origins, site_ids, context, horizons)

    return {
        "train": build(
            "train",
            int(split_cfg["train_start"]),
            int(split_cfg["train_end"]),
            seen_sites,
            int(model_cfg["train_origin_stride"]),
        ),
        "validation": build(
            "validation",
            int(split_cfg["validation_start"]),
            int(split_cfg["validation_end"]),
            seen_sites,
            int(model_cfg["evaluation_origin_stride"]),
        ),
        "test_seen": build(
            "test_seen",
            int(split_cfg["test_start"]),
            int(split_cfg["test_end"]),
            seen_sites,
            int(model_cfg["evaluation_origin_stride"]),
        ),
        "test_unseen": build(
            "test_unseen",
            int(split_cfg["test_start"]),
            int(split_cfg["test_end"]),
            unseen_sites,
            int(model_cfg["evaluation_origin_stride"]),
        ),
    }


def _loader(dataset: Dataset, batch_size: int, shuffle: bool) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(3407)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
        drop_last=False,
    )


def _masked_input(x: torch.Tensor, probability: float, training: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not training or probability <= 0:
        mask = torch.ones_like(x)
        final_missing = torch.zeros_like(x[:, -1], dtype=torch.bool)
        return x, mask, final_missing
    observed = torch.rand_like(x) > probability
    forced_final = torch.rand_like(x[:, -1]) < max(0.35, probability)
    observed[:, -1] &= ~forced_final
    # Always leave at least one feature observed at each time step.
    empty = observed.sum(dim=-1) == 0
    if empty.any():
        observed[..., 0] |= empty
    mask = observed.to(x.dtype)
    return x * mask, mask, ~observed[:, -1]


def _forecast_loss(mean: torch.Tensor, logvar: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    squared = (target - mean) ** 2
    return 0.5 * (logvar + squared * torch.exp(-logvar)).mean() + 0.05 * squared.mean()


@torch.no_grad()
def _validate(model: PersistentWorldModel, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    squared: list[float] = []
    absolute: list[float] = []
    for x, y, static, _, _ in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        static = static.to(device, non_blocking=True)
        mask = torch.ones_like(x)
        mean, logvar, _, _ = model(x, mask, static)
        losses.append(float(_forecast_loss(mean, logvar, y)))
        squared.append(float(((mean - y) ** 2).mean()))
        absolute.append(float((mean - y).abs().mean()))
    return {
        "loss": float(np.mean(losses)),
        "rmse_z": float(math.sqrt(np.mean(squared))),
        "mae_z": float(np.mean(absolute)),
    }


def train_model(config: dict[str, Any], force: bool = False) -> Path:
    ensure_directories()
    checkpoint_path = ROOT / "artifacts" / "pmwm_backbone.pt"
    history_path = ROOT / "results" / "training_history.csv"
    if checkpoint_path.exists() and history_path.exists() and not force:
        return checkpoint_path

    seed = int(config["project"]["seed"])
    set_seed(seed)
    store = load_features()
    datasets = make_datasets(config, store)
    cfg = config["model"]
    batch_size = int(cfg["batch_size"])
    train_loader = _loader(datasets["train"], batch_size, shuffle=True)
    validation_loader = _loader(datasets["validation"], batch_size, shuffle=False)
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
        model.parameters(),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(cfg["epochs"]))
    best_state: dict[str, torch.Tensor] | None = None
    best_loss = float("inf")
    stale = 0
    history: list[dict[str, float]] = []
    started_training = time.perf_counter()

    for epoch in range(1, int(cfg["epochs"]) + 1):
        model.train()
        batch_losses: list[float] = []
        forecast_losses: list[float] = []
        imputation_losses: list[float] = []
        epoch_started = time.perf_counter()
        for x, y, static, _, _ in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            static = static.to(device, non_blocking=True)
            masked_x, mask, final_missing = _masked_input(x, float(cfg["mask_probability"]), training=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                mean, logvar, imputation, _ = model(masked_x, mask, static)
                forecast = _forecast_loss(mean, logvar, y)
                missing_float = final_missing.to(imputation.dtype)
                denominator = missing_float.sum().clamp_min(1.0)
                imputation_loss = (((imputation - x[:, -1]) ** 2) * missing_float).sum() / denominator
                loss = forecast + float(cfg["auxiliary_imputation_weight"]) * imputation_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            batch_losses.append(float(loss.detach()))
            forecast_losses.append(float(forecast.detach()))
            imputation_losses.append(float(imputation_loss.detach()))
        scheduler.step()
        validation = _validate(model, validation_loader, device)
        epoch_row = {
            "epoch": float(epoch),
            "train_loss": float(np.mean(batch_losses)),
            "train_forecast_loss": float(np.mean(forecast_losses)),
            "train_imputation_mse": float(np.mean(imputation_losses)),
            "validation_loss": validation["loss"],
            "validation_rmse_z": validation["rmse_z"],
            "validation_mae_z": validation["mae_z"],
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "epoch_seconds": time.perf_counter() - epoch_started,
        }
        history.append(epoch_row)
        print(
            f"epoch {epoch:02d} train={epoch_row['train_loss']:.4f} "
            f"val={validation['loss']:.4f} rmse={validation['rmse_z']:.4f} "
            f"({epoch_row['epoch_seconds']:.1f}s)",
            flush=True,
        )
        if validation["loss"] < best_loss - 1e-4:
            best_loss = validation["loss"]
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= int(cfg["patience"]):
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)
    torch.save(
        {
            "state_dict": model.state_dict(),
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
            "best_validation_loss": best_loss,
        },
        checkpoint_path,
    )
    pd.DataFrame(history).to_csv(history_path, index=False)
    atomic_json(
        ROOT / "artifacts" / "model_manifest.json",
        {
            "checkpoint": str(checkpoint_path.relative_to(ROOT)),
            "device": str(device),
            "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
            "train_samples": len(datasets["train"]),
            "validation_samples": len(datasets["validation"]),
            "epochs_completed": len(history),
            "best_validation_loss": best_loss,
            "total_training_seconds": time.perf_counter() - started_training,
        },
    )
    return checkpoint_path


def load_model(config: dict[str, Any], path: Path | None = None) -> PersistentWorldModel:
    path = path or ROOT / "artifacts" / "pmwm_backbone.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = PersistentWorldModel(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["state_dict"])
    return model


@torch.no_grad()
def predict_dataset(
    model: PersistentWorldModel,
    dataset: SiteSequenceDataset,
    batch_size: int,
    missing_rate: float = 0.0,
    seed: int = 3407,
) -> dict[str, np.ndarray]:
    device = torch.device(device_name())
    model = model.to(device).eval()
    loader = _loader(dataset, batch_size, shuffle=False)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    outputs: dict[str, list[np.ndarray]] = {
        "mean": [], "logvar": [], "latent": [], "imputation": [], "origin": [], "site": [], "final_mask": []
    }
    for x, _, static, origin, site in loader:
        x = x.to(device, non_blocking=True)
        static = static.to(device, non_blocking=True)
        mask = torch.ones_like(x)
        if missing_rate > 0:
            final_missing = torch.rand(
                x.shape[0], x.shape[2], device=device, generator=generator
            ) < missing_rate
            mask[:, -1] = (~final_missing).to(x.dtype)
        else:
            final_missing = torch.zeros(x.shape[0], x.shape[2], device=device, dtype=torch.bool)
        masked = x * mask
        mean, logvar, imputation, latent = model(masked, mask, static)
        outputs["mean"].append(mean.float().cpu().numpy())
        outputs["logvar"].append(logvar.float().cpu().numpy())
        outputs["latent"].append(latent.float().cpu().numpy())
        outputs["imputation"].append(imputation.float().cpu().numpy())
        outputs["origin"].append(origin.numpy())
        outputs["site"].append(site.numpy())
        outputs["final_mask"].append(final_missing.cpu().numpy())
    return {name: np.concatenate(parts, axis=0) for name, parts in outputs.items()}


def generate_backbone_predictions(config: dict[str, Any], force: bool = False) -> dict[str, Path]:
    ensure_directories()
    expected = {
        name: ROOT / "artifacts" / f"predictions_{name}.npz"
        for name in ["train", "validation", "test_seen", "test_unseen"]
    }
    if all(path.exists() for path in expected.values()) and not force:
        return expected
    store = load_features()
    datasets = make_datasets(config, store)
    model = load_model(config)
    for name, dataset in datasets.items():
        print(f"encoding {name}: {len(dataset):,} samples", flush=True)
        result = predict_dataset(model, dataset, int(config["model"]["batch_size"]))
        targets_z, targets_physical = actual_targets(
            result["origin"], result["site"], config["model"]["horizons"], store
        )
        np.savez_compressed(
            expected[name],
            **result,
            target_z=targets_z,
            target_physical=targets_physical,
        )
    return expected


def load_predictions(name: str) -> dict[str, np.ndarray]:
    path = ROOT / "artifacts" / f"predictions_{name}.npz"
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


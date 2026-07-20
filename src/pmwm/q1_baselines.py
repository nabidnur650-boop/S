from __future__ import annotations

import copy
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from torch import nn

from .common import device_name, set_seed
from .q1_common import Q1_ROOT, ensure_q1_directories, q1_json, verify_protocol_lock
from .q1_features import Q1FeatureStore, load_q1_features
from .q1_model import _loader, make_q1_datasets


class DeterministicForecaster(nn.Module):
    def forward(self, x: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class DLinearForecaster(DeterministicForecaster):
    def __init__(self, context: int, n_input: int, n_horizons: int, n_target: int, static_dim: int) -> None:
        super().__init__()
        self.context = context
        self.n_input = n_input
        self.n_horizons = n_horizons
        self.n_target = n_target
        self.seasonal = nn.Linear(context, n_horizons)
        self.trend = nn.Linear(context, n_horizons)
        self.head = nn.Linear(n_input * n_horizons + static_dim, n_horizons * n_target)

    def forward(self, x: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        # Centered moving average uses only values inside the observed context.
        padded = nn.functional.pad(x.transpose(1, 2), (12, 12), mode="replicate")
        trend = nn.functional.avg_pool1d(padded, kernel_size=25, stride=1).transpose(1, 2)
        seasonal = x - trend
        components = self.seasonal(seasonal.transpose(1, 2)) + self.trend(trend.transpose(1, 2))
        fused = torch.cat([components.flatten(1), static], dim=1)
        return self.head(fused).reshape(-1, self.n_horizons, self.n_target)


class CausalBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = 2 * dilation
        self.padding = padding
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, dilation=dilation, padding=padding)
        self.norm = nn.GroupNorm(1, channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        value = self.conv(x)
        if self.padding:
            value = value[..., : -self.padding]
        return residual + self.dropout(torch.nn.functional.gelu(self.norm(value)))


class TCNForecaster(DeterministicForecaster):
    def __init__(self, n_input: int, n_horizons: int, n_target: int, static_dim: int, dropout: float) -> None:
        super().__init__()
        channels = 64
        self.n_horizons = n_horizons
        self.n_target = n_target
        self.input = nn.Conv1d(n_input, channels, kernel_size=1)
        self.blocks = nn.Sequential(*[CausalBlock(channels, dilation, dropout) for dilation in [1, 2, 4, 8]])
        self.head = nn.Sequential(
            nn.Linear(channels + static_dim, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, n_horizons * n_target),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.input(x.transpose(1, 2)))[..., -1]

    def forward(self, x: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        hidden = self.encode(x)
        return self.head(torch.cat([hidden, static], dim=1)).reshape(-1, self.n_horizons, self.n_target)


class PatchTransformerForecaster(DeterministicForecaster):
    def __init__(
        self, context: int, n_input: int, n_horizons: int, n_target: int, static_dim: int, dropout: float
    ) -> None:
        super().__init__()
        self.patch = 4
        self.n_patches = context // self.patch
        self.n_horizons = n_horizons
        self.n_target = n_target
        dimension = 64
        self.embedding = nn.Linear(self.patch * n_input, dimension)
        self.position = nn.Parameter(torch.zeros(1, self.n_patches, dimension))
        layer = nn.TransformerEncoderLayer(
            d_model=dimension, nhead=4, dim_feedforward=128, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Sequential(
            nn.LayerNorm(dimension + static_dim),
            nn.Linear(dimension + static_dim, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, n_horizons * n_target),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        tokens = x.reshape(batch, self.n_patches, -1)
        return self.encoder(self.embedding(tokens) + self.position).mean(dim=1)

    def forward(self, x: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        encoded = self.encode(x)
        return self.head(torch.cat([encoded, static], dim=1)).reshape(-1, self.n_horizons, self.n_target)


class ITransformerForecaster(DeterministicForecaster):
    def __init__(
        self, context: int, n_input: int, n_horizons: int, n_target: int, static_dim: int, dropout: float
    ) -> None:
        super().__init__()
        self.n_horizons = n_horizons
        self.n_target = n_target
        dimension = 64
        self.embedding = nn.Linear(context, dimension)
        self.variable = nn.Parameter(torch.zeros(1, n_input, dimension))
        layer = nn.TransformerEncoderLayer(
            d_model=dimension, nhead=4, dim_feedforward=128, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Sequential(
            nn.Linear(n_input * dimension + static_dim, 192), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(192, n_horizons * n_target),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(self.embedding(x.transpose(1, 2)) + self.variable)

    def forward(self, x: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        tokens = self.encode(x)
        return self.head(torch.cat([tokens.flatten(1), static], dim=1)).reshape(
            -1, self.n_horizons, self.n_target
        )


BASELINE_CLASSES = {
    "dlinear": DLinearForecaster,
    "tcn": TCNForecaster,
    "patchtst": PatchTransformerForecaster,
    "itransformer": ITransformerForecaster,
}


def _make_model(name: str, config: dict[str, Any], store: Q1FeatureStore) -> DeterministicForecaster:
    cfg = config["model"]
    common = {
        "n_input": store.input_z.shape[-1],
        "n_horizons": len(cfg["horizons"]),
        "n_target": store.target_z.shape[-1],
        "static_dim": store.static.shape[-1],
    }
    if name == "dlinear":
        return DLinearForecaster(context=int(cfg["context_steps"]), **common)
    if name == "tcn":
        return TCNForecaster(dropout=float(cfg["dropout"]), **common)
    if name in {"patchtst", "itransformer"}:
        return BASELINE_CLASSES[name](
            context=int(cfg["context_steps"]), dropout=float(cfg["dropout"]), **common
        )
    raise ValueError(f"Unknown baseline: {name}")


def baseline_checkpoint(name: str, seed: int) -> Path:
    return Q1_ROOT / "checkpoints" / f"{name}_seed{seed}.pt"


@torch.no_grad()
def _validate(model: DeterministicForecaster, loader: Any, device: torch.device) -> float:
    model.eval()
    total = 0.0
    count = 0
    for x, target, static, _, _ in loader:
        x, target, static = x.to(device), target.to(device), static.to(device)
        predicted = model(x, static)
        total += float(torch.square(predicted - target).sum())
        count += target.numel()
    return total / count


def train_neural_baseline(
    config: dict[str, Any], name: str, seed: int, force: bool = False
) -> Path:
    ensure_q1_directories()
    lock = verify_protocol_lock()
    output = baseline_checkpoint(name, seed)
    history_path = Q1_ROOT / "results" / "tables" / f"training_{name}_seed{seed}.csv"
    if output.exists() and history_path.exists() and not force:
        return output
    set_seed(seed)
    store = load_q1_features()
    datasets = make_q1_datasets(config, store)
    cfg = config["model"]
    device = torch.device(device_name())
    model = _make_model(name, config, store).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"])
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(cfg["epochs"]))
    train_loader = _loader(datasets["train"], int(cfg["batch_size"]), True, seed)
    validation_loader = _loader(datasets["validation"], int(cfg["batch_size"]), False, seed)
    best_state: dict[str, torch.Tensor] | None = None
    best_mse = float("inf")
    stale = 0
    rows = []
    for epoch in range(1, int(cfg["epochs"]) + 1):
        model.train()
        losses = []
        began = time.perf_counter()
        for x, target, static, _, _ in train_loader:
            x, target, static = x.to(device), target.to(device), static.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                predicted = model(x, static)
                loss = torch.square(predicted - target).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        scheduler.step()
        validation_mse = _validate(model, validation_loader, device)
        rows.append(
            {
                "model": name,
                "seed": seed,
                "epoch": epoch,
                "train_mse": float(np.mean(losses)),
                "validation_rmse_z": math.sqrt(validation_mse),
                "epoch_seconds": time.perf_counter() - began,
            }
        )
        print(
            f"q1 {name} seed={seed} epoch={epoch:02d} val_rmse={math.sqrt(validation_mse):.4f}",
            flush=True,
        )
        if validation_mse < best_mse - 1e-4:
            best_mse = validation_mse
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= int(cfg["patience"]):
                break
    if best_state is None:
        raise RuntimeError(f"No checkpoint for {name}")
    torch.save(
        {
            "state_dict": best_state,
            "name": name,
            "seed": seed,
            "protocol_hash": lock["combined_sha256"],
            "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        },
        output,
    )
    pd.DataFrame(rows).to_csv(history_path, index=False)
    return output


def load_neural_baseline(config: dict[str, Any], name: str, seed: int) -> DeterministicForecaster:
    store = load_q1_features()
    checkpoint = torch.load(baseline_checkpoint(name, seed), map_location="cpu", weights_only=False)
    model = _make_model(name, config, store)
    model.load_state_dict(checkpoint["state_dict"])
    return model


@torch.no_grad()
def predict_neural_baseline(
    config: dict[str, Any], name: str, seed: int, split: str, force: bool = False
) -> Path:
    output = Q1_ROOT / "artifacts" / f"predictions_{name}_{split}_seed{seed}.npz"
    if output.exists() and not force:
        return output
    datasets = make_q1_datasets(config)
    model = load_neural_baseline(config, name, seed)
    device = torch.device(device_name())
    model = model.to(device).eval()
    parts: dict[str, list[np.ndarray]] = {"mean": [], "target_z": [], "origin": [], "site": []}
    for x, target, static, origin, site in _loader(
        datasets[split], int(config["model"]["batch_size"]), False, seed
    ):
        predicted = model(x.to(device), static.to(device))
        parts["mean"].append(predicted.float().cpu().numpy())
        parts["target_z"].append(target.numpy())
        parts["origin"].append(origin.numpy())
        parts["site"].append(site.numpy())
    np.savez_compressed(output, **{key: np.concatenate(value) for key, value in parts.items()})
    return output


def _multiscale_features(origins: np.ndarray, sites: np.ndarray, store: Q1FeatureStore) -> np.ndarray:
    x = store.input_z
    cumulative = np.concatenate(
        [np.zeros((1, x.shape[1], x.shape[2]), dtype=np.float32), np.cumsum(x, axis=0, dtype=np.float32)]
    )
    cumulative_square = np.concatenate(
        [np.zeros((1, x.shape[1], x.shape[2]), dtype=np.float32), np.cumsum(np.square(x), axis=0, dtype=np.float32)]
    )

    def stats(width: int) -> tuple[np.ndarray, np.ndarray]:
        end = origins + 1
        start = end - width
        mean = (cumulative[end, sites] - cumulative[start, sites]) / width
        square = (cumulative_square[end, sites] - cumulative_square[start, sites]) / width
        return mean, np.sqrt(np.maximum(square - np.square(mean), 1e-6))

    mean4, _ = stats(4)
    mean28, std28 = stats(28)
    mean56, std56 = stats(56)
    current = x[origins, sites]
    tendency = current - x[origins - 4, sites]
    return np.concatenate([current, mean4, mean28, std28, mean56, std56, tendency, store.static[sites]], axis=1)


def fit_and_predict_ridge(config: dict[str, Any], force: bool = False) -> list[Path]:
    outputs = [Q1_ROOT / "artifacts" / f"predictions_ridge_{split}.npz" for split in ["development_spatial", "confirmatory"]]
    model_output = Q1_ROOT / "artifacts" / "ridge_model.npz"
    if all(path.exists() for path in [*outputs, model_output]) and not force:
        return outputs
    store = load_q1_features()
    datasets = make_q1_datasets(config, store)
    x_train = _multiscale_features(datasets["train"].origins, datasets["train"].sites, store)
    y_train = np.stack(
        [store.target_z[datasets["train"].origins + h, datasets["train"].sites] for h in config["model"]["horizons"]],
        axis=1,
    ).reshape(len(x_train), -1)
    x_val = _multiscale_features(datasets["validation"].origins, datasets["validation"].sites, store)
    y_val = np.stack(
        [store.target_z[datasets["validation"].origins + h, datasets["validation"].sites] for h in config["model"]["horizons"]],
        axis=1,
    ).reshape(len(x_val), -1)
    best: tuple[float, Ridge] | None = None
    for alpha in [0.1, 1.0, 10.0, 100.0]:
        model = Ridge(alpha=alpha).fit(x_train, y_train)
        mse = float(np.mean(np.square(model.predict(x_val) - y_val)))
        if best is None or mse < best[0]:
            best = (mse, model)
    assert best is not None
    np.savez_compressed(
        model_output,
        coefficient=best[1].coef_.astype(np.float32),
        intercept=best[1].intercept_.astype(np.float32),
        alpha=np.asarray(best[1].alpha, dtype=np.float32),
    )
    for split, output in zip(["development_spatial", "confirmatory"], outputs):
        dataset = datasets[split]
        features = _multiscale_features(dataset.origins, dataset.sites, store)
        mean = best[1].predict(features).reshape(len(features), len(config["model"]["horizons"]), -1).astype(np.float32)
        target = np.stack(
            [store.target_z[dataset.origins + h, dataset.sites] for h in config["model"]["horizons"]], axis=1
        ).astype(np.float32)
        np.savez_compressed(output, mean=mean, target_z=target, origin=dataset.origins, site=dataset.sites)
    q1_json("artifacts/ridge_manifest.json", {"alpha": float(best[1].alpha), "validation_mse_z": best[0]})
    return outputs


def run_registered_neural_baselines(config: dict[str, Any], force: bool = False) -> None:
    seeds = [int(value) for value in config["model"]["seeds"][: int(config["evaluation"]["minimum_neural_baseline_seeds"])]]
    for name in BASELINE_CLASSES:
        for seed in seeds:
            train_neural_baseline(config, name, seed, force=force)
            for split in ["development_spatial", "confirmatory"]:
                predict_neural_baseline(config, name, seed, split, force=force)
    fit_and_predict_ridge(config, force=force)

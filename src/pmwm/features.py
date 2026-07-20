from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .common import ROOT, atomic_json, ensure_directories, sha256_file
from .data import load_stream

INPUT_FEATURES = ["temperature_c", "pressure_hpa", "u_wind_ms", "v_wind_ms", "log1p_precipitation_mm"]
TARGET_FEATURES = ["temperature_c", "pressure_hpa", "wind_speed_ms", "log1p_precipitation_mm"]
PHYSICAL_TARGET_FEATURES = ["temperature_c", "pressure_hpa", "wind_speed_ms", "precipitation_mm"]
TARGET_UNITS = ["°C", "hPa", "m s$^{-1}$", "mm / 6 h"]


@dataclass
class FeatureStore:
    time: np.ndarray
    input_z: np.ndarray
    target_z: np.ndarray
    target_physical: np.ndarray
    target_seasonal: np.ndarray
    input_scale: np.ndarray
    target_scale: np.ndarray
    static: np.ndarray
    site_names: np.ndarray
    site_lat: np.ndarray
    site_lon: np.ndarray
    source_lat: np.ndarray
    source_lon: np.ndarray
    zones: np.ndarray
    holdout: np.ndarray
    input_features: np.ndarray
    target_features: np.ndarray


def time_design(times: np.ndarray, annual_harmonics: int = 4, daily_harmonics: int = 2) -> np.ndarray:
    """Fourier design for a smooth, training-only seasonal climatology."""
    ns = times.astype("datetime64[ns]").astype(np.int64).astype(np.float64)
    seconds = ns / 1e9
    tropical_year = 365.2425 * 86400.0
    day = 86400.0
    columns = [np.ones(len(times), dtype=np.float64)]
    for harmonic in range(1, annual_harmonics + 1):
        phase = 2 * np.pi * harmonic * seconds / tropical_year
        columns.extend([np.sin(phase), np.cos(phase)])
    for harmonic in range(1, daily_harmonics + 1):
        phase = 2 * np.pi * harmonic * seconds / day
        columns.extend([np.sin(phase), np.cos(phase)])
    return np.column_stack(columns)


def _raw_features(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    inputs = raw.copy().astype(np.float32)
    inputs[..., 4] = np.log1p(np.maximum(inputs[..., 4], 0.0))
    wind_speed = np.sqrt(np.square(raw[..., 2]) + np.square(raw[..., 3]))
    target_physical = np.stack([raw[..., 0], raw[..., 1], wind_speed, raw[..., 4]], axis=-1).astype(np.float32)
    target_model = target_physical.copy()
    target_model[..., 3] = np.log1p(np.maximum(target_model[..., 3], 0.0))
    return inputs, target_model, target_physical


def _fit_fourier_climatology(
    values: np.ndarray, design: np.ndarray, train_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_time, n_site, n_feature = values.shape
    train_design = design[train_mask]
    train_values = values[train_mask].reshape(train_mask.sum(), -1).astype(np.float64)
    coefficients, *_ = np.linalg.lstsq(train_design, train_values, rcond=None)
    seasonal = (design @ coefficients).reshape(n_time, n_site, n_feature).astype(np.float32)
    residual = values[train_mask] - seasonal[train_mask]
    standard = np.nanstd(residual, axis=0, ddof=1)
    median = np.nanmedian(residual, axis=0)
    mad = 1.4826 * np.nanmedian(np.abs(residual - median), axis=0)
    scale = np.maximum(0.5 * standard + 0.5 * mad, 1e-3).astype(np.float32)
    return seasonal, scale, coefficients.astype(np.float32)


def _static_coordinates(latitudes: np.ndarray, longitudes: np.ndarray) -> np.ndarray:
    lat_radians = np.deg2rad(latitudes.astype(np.float64))
    lon_radians = np.deg2rad(longitudes.astype(np.float64))
    return np.column_stack(
        [
            np.sin(lat_radians),
            np.cos(lat_radians),
            np.sin(lon_radians),
            np.cos(lon_radians),
        ]
    ).astype(np.float32)


def prepare_features(config: dict[str, Any], force: bool = False) -> Path:
    ensure_directories()
    output = ROOT / "artifacts" / "feature_store.npz"
    manifest_path = ROOT / "artifacts" / "feature_manifest.json"
    if output.exists() and manifest_path.exists() and not force:
        return output

    stream = load_stream()
    times = stream["time"]
    raw = stream["data"].astype(np.float32)
    inputs, target_model, target_physical = _raw_features(raw)
    date_index = pd.DatetimeIndex(times)
    train_mask = (date_index.year >= config["splits"]["train_start"]) & (
        date_index.year <= config["splits"]["train_end"]
    )
    design = time_design(times)
    input_seasonal, input_scale, input_coeff = _fit_fourier_climatology(inputs, design, train_mask)
    target_seasonal, target_scale, target_coeff = _fit_fourier_climatology(target_model, design, train_mask)
    input_z = ((inputs - input_seasonal) / input_scale[None, ...]).astype(np.float32)
    target_z = ((target_model - target_seasonal) / target_scale[None, ...]).astype(np.float32)
    input_z = np.clip(input_z, -12.0, 12.0)
    target_z = np.clip(target_z, -12.0, 12.0)
    static = _static_coordinates(stream["site_lat"], stream["site_lon"])

    np.savez_compressed(
        output,
        time=times.astype("datetime64[ns]").astype(np.int64),
        input_z=input_z,
        target_z=target_z,
        target_physical=target_physical,
        target_seasonal=target_seasonal,
        input_scale=input_scale,
        target_scale=target_scale,
        input_coeff=input_coeff,
        target_coeff=target_coeff,
        static=static,
        site_names=stream["site_names"],
        site_lat=stream["site_lat"],
        site_lon=stream["site_lon"],
        source_lat=stream["source_lat"],
        source_lon=stream["source_lon"],
        zones=stream["zones"],
        holdout=stream["holdout"],
        input_features=np.asarray(INPUT_FEATURES),
        target_features=np.asarray(TARGET_FEATURES),
    )
    manifest = {
        "artifact": str(output.relative_to(ROOT)),
        "sha256": sha256_file(output),
        "shape_input": list(input_z.shape),
        "shape_target": list(target_z.shape),
        "climatology": {
            "fit_period": [config["splits"]["train_start"], config["splits"]["train_end"]],
            "annual_harmonics": 4,
            "daily_harmonics": 2,
            "normalization": "mean of residual standard deviation and robust MAD scale",
            "note": "Climatology is fit per anchor; held-out anchors never enter model or memory training.",
        },
        "input_features": INPUT_FEATURES,
        "target_features": TARGET_FEATURES,
        "finite_fraction": float(np.isfinite(input_z).mean()),
    }
    atomic_json(manifest_path, manifest)
    return output


def load_features(path: Path | None = None) -> FeatureStore:
    path = path or ROOT / "artifacts" / "feature_store.npz"
    with np.load(path, allow_pickle=False) as archive:
        payload = {name: archive[name] for name in archive.files}
    return FeatureStore(
        time=payload["time"].astype("datetime64[ns]"),
        input_z=payload["input_z"],
        target_z=payload["target_z"],
        target_physical=payload["target_physical"],
        target_seasonal=payload["target_seasonal"],
        input_scale=payload["input_scale"],
        target_scale=payload["target_scale"],
        static=payload["static"],
        site_names=payload["site_names"],
        site_lat=payload["site_lat"],
        site_lon=payload["site_lon"],
        source_lat=payload["source_lat"],
        source_lon=payload["source_lon"],
        zones=payload["zones"],
        holdout=payload["holdout"],
        input_features=payload["input_features"],
        target_features=payload["target_features"],
    )


def origin_indices(
    store: FeatureStore,
    start_year: int,
    end_year: int,
    context_steps: int,
    max_horizon: int,
    stride: int = 4,
) -> np.ndarray:
    dates = pd.DatetimeIndex(store.time)
    valid = (
        (dates.year >= start_year)
        & (dates.year <= end_year)
        & (dates.hour == 0)
    )
    indices = np.flatnonzero(valid)
    indices = indices[(indices >= context_steps - 1) & (indices + max_horizon < len(store.time))]
    if stride > 4:
        indices = indices[:: max(1, stride // 4)]
    return indices.astype(np.int64)


def paired_samples(origins: np.ndarray, site_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    origin_grid, site_grid = np.meshgrid(origins, site_ids, indexing="ij")
    return origin_grid.ravel().astype(np.int64), site_grid.ravel().astype(np.int64)


def standardized_to_physical(
    prediction_z: np.ndarray,
    origins: np.ndarray,
    sites: np.ndarray,
    horizons: list[int],
    store: FeatureStore,
) -> np.ndarray:
    output = np.empty_like(prediction_z, dtype=np.float32)
    for horizon_idx, horizon in enumerate(horizons):
        future = origins + int(horizon)
        seasonal = store.target_seasonal[future, sites]
        scale = store.target_scale[sites]
        transformed = seasonal + prediction_z[:, horizon_idx] * scale
        output[:, horizon_idx, :3] = transformed[:, :3]
        output[:, horizon_idx, 3] = np.maximum(np.expm1(transformed[:, 3]), 0.0)
    return output


def actual_targets(
    origins: np.ndarray, sites: np.ndarray, horizons: list[int], store: FeatureStore
) -> tuple[np.ndarray, np.ndarray]:
    target_z = np.stack(
        [store.target_z[origins + int(horizon), sites] for horizon in horizons], axis=1
    ).astype(np.float32)
    target_physical = np.stack(
        [store.target_physical[origins + int(horizon), sites] for horizon in horizons], axis=1
    ).astype(np.float32)
    return target_z, target_physical


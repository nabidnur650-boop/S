from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .common import sha256_file
from .features import time_design
from .q1_common import Q1_ROOT, ensure_q1_directories, q1_json, verify_protocol_lock
from .q1_data import load_q1_stream

INPUT_FEATURES = ["temperature_c", "pressure_hpa", "u_wind_ms", "v_wind_ms", "log1p_precipitation_mm"]
TARGET_FEATURES = ["temperature_c", "pressure_hpa", "wind_speed_ms", "log1p_precipitation_mm"]


@dataclass
class Q1FeatureStore:
    time: np.ndarray
    input_z: np.ndarray
    target_z: np.ndarray
    target_physical: np.ndarray
    target_seasonal: np.ndarray
    input_scale: np.ndarray
    target_scale: np.ndarray
    static: np.ndarray
    cell_id: np.ndarray
    partition: np.ndarray
    latitude: np.ndarray
    longitude: np.ndarray
    input_features: np.ndarray
    target_features: np.ndarray


def _sphere(latitude: np.ndarray, longitude: np.ndarray) -> np.ndarray:
    lat = np.deg2rad(latitude.astype(np.float64))
    lon = np.deg2rad(longitude.astype(np.float64))
    return np.column_stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])


def _static(latitude: np.ndarray, longitude: np.ndarray) -> np.ndarray:
    xyz = _sphere(latitude, longitude)
    return np.column_stack([xyz, np.sin(2 * np.deg2rad(latitude)), np.cos(2 * np.deg2rad(latitude))]).astype(np.float32)


def _raw(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    inputs = raw.astype(np.float32, copy=True)
    inputs[..., 4] = np.log1p(np.maximum(inputs[..., 4], 0.0))
    wind = np.sqrt(np.square(raw[..., 2]) + np.square(raw[..., 3]))
    physical = np.stack([raw[..., 0], raw[..., 1], wind, raw[..., 4]], axis=-1).astype(np.float32)
    target = physical.copy()
    target[..., 3] = np.log1p(np.maximum(target[..., 3], 0.0))
    return inputs, target, physical


def _fit_training_statistics(
    values: np.ndarray,
    design: np.ndarray,
    time_mask: np.ndarray,
    training_sites: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    train_design = design[time_mask]
    y = values[time_mask][:, training_sites].reshape(time_mask.sum(), -1).astype(np.float64)
    coefficient, *_ = np.linalg.lstsq(train_design, y, rcond=None)
    coefficient = coefficient.reshape(design.shape[1], len(training_sites), values.shape[-1])
    fitted = np.einsum("td,dsf->tsf", train_design, coefficient, optimize=True)
    residual = values[time_mask][:, training_sites] - fitted
    standard = np.std(residual, axis=0, ddof=1)
    median = np.median(residual, axis=0)
    mad = 1.4826 * np.median(np.abs(residual - median), axis=0)
    scale = np.maximum(0.5 * standard + 0.5 * mad, 1e-3)
    return coefficient.astype(np.float32), scale.astype(np.float32)


def _interpolate_statistics(
    coefficient: np.ndarray,
    scale: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    training_sites: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xyz = _sphere(latitude, longitude)
    train_xyz = xyz[training_sites]
    similarity = np.clip(xyz @ train_xyz.T, -1.0, 1.0)
    angular = np.arccos(similarity)
    nearest = np.argpartition(angular, kth=min(3, len(training_sites) - 1), axis=1)[:, :4]
    distance = np.take_along_axis(angular, nearest, axis=1)
    weights = 1.0 / np.maximum(distance, 1e-4) ** 2
    weights /= weights.sum(axis=1, keepdims=True)
    all_coefficient = np.einsum("sk,dskf->dsf", weights, coefficient[:, nearest, :], optimize=True)
    all_scale = np.einsum("sk,skf->sf", weights, scale[nearest], optimize=True)
    # Training sites retain their exact local statistics.
    for local_index, site in enumerate(training_sites):
        all_coefficient[:, site] = coefficient[:, local_index]
        all_scale[site] = scale[local_index]
    return all_coefficient.astype(np.float32), all_scale.astype(np.float32), weights.astype(np.float32)


def prepare_q1_features(config: dict[str, Any], force: bool = False) -> Path:
    ensure_q1_directories()
    lock = verify_protocol_lock()
    output = Q1_ROOT / "artifacts" / "feature_store.npz"
    if output.exists() and not force:
        return output
    stream = load_q1_stream()
    inputs, targets, physical = _raw(stream["data"])
    dates = pd.DatetimeIndex(stream["time"])
    train_start, train_end = config["splits"]["train"]
    time_mask = (dates.year >= train_start) & (dates.year <= train_end)
    training_sites = np.flatnonzero(stream["partition"] == "train")
    design = time_design(stream["time"])
    input_coefficient, input_train_scale = _fit_training_statistics(inputs, design, time_mask, training_sites)
    target_coefficient, target_train_scale = _fit_training_statistics(targets, design, time_mask, training_sites)
    input_all_coefficient, input_scale, weights = _interpolate_statistics(
        input_coefficient, input_train_scale, stream["latitude"], stream["longitude"], training_sites
    )
    target_all_coefficient, target_scale, _ = _interpolate_statistics(
        target_coefficient, target_train_scale, stream["latitude"], stream["longitude"], training_sites
    )
    input_seasonal = np.einsum("td,dsf->tsf", design, input_all_coefficient, optimize=True).astype(np.float32)
    target_seasonal = np.einsum("td,dsf->tsf", design, target_all_coefficient, optimize=True).astype(np.float32)
    input_z = np.clip((inputs - input_seasonal) / input_scale[None], -12, 12).astype(np.float32)
    target_z = np.clip((targets - target_seasonal) / target_scale[None], -12, 12).astype(np.float32)
    np.savez_compressed(
        output,
        time=stream["time"].astype(np.int64),
        input_z=input_z,
        target_z=target_z,
        target_physical=physical,
        target_seasonal=target_seasonal,
        input_scale=input_scale,
        target_scale=target_scale,
        static=_static(stream["latitude"], stream["longitude"]),
        cell_id=stream["cell_id"],
        partition=stream["partition"],
        latitude=stream["latitude"],
        longitude=stream["longitude"],
        input_features=np.asarray(INPUT_FEATURES),
        target_features=np.asarray(TARGET_FEATURES),
        interpolation_weights=weights,
        training_site_indices=training_sites,
    )
    q1_json(
        "artifacts/feature_manifest.json",
        {
            "protocol_hash": lock["combined_sha256"],
            "artifact_sha256": sha256_file(output),
            "input_shape": list(input_z.shape),
            "target_shape": list(target_z.shape),
            "training_cells_used_for_statistics": int(len(training_sites)),
            "heldout_cells_used_for_statistics": 0,
            "finite_fraction": float(np.isfinite(input_z).mean()),
            "normalization": "training-cell Fourier climatology; four-neighbor spherical interpolation for held-out cells",
        },
    )
    return output


def load_q1_features() -> Q1FeatureStore:
    with np.load(Q1_ROOT / "artifacts" / "feature_store.npz", allow_pickle=False) as archive:
        payload = {key: archive[key] for key in archive.files}
    return Q1FeatureStore(
        time=payload["time"].astype("datetime64[ns]"),
        input_z=payload["input_z"],
        target_z=payload["target_z"],
        target_physical=payload["target_physical"],
        target_seasonal=payload["target_seasonal"],
        input_scale=payload["input_scale"],
        target_scale=payload["target_scale"],
        static=payload["static"],
        cell_id=payload["cell_id"],
        partition=payload["partition"],
        latitude=payload["latitude"],
        longitude=payload["longitude"],
        input_features=payload["input_features"],
        target_features=payload["target_features"],
    )


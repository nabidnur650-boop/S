from __future__ import annotations

import copy
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import dask
import numpy as np
import pandas as pd
import torch
import xarray as xr
import yaml
from scipy.stats import t
from sklearn.metrics import average_precision_score, roc_auc_score

from .common import device_name, sha256_file
from .evaluation import gaussian_crps
from .features import time_design
from .memory import load_memory, memory_forecast
from .q1_baselines import (
    BASELINE_CLASSES,
    _make_model,
    _multiscale_features,
    baseline_checkpoint,
    fit_and_predict_ridge,
)
from .q1_common import Q1_ROOT, load_q1_config, q1_json, verify_protocol_lock
from .q1_data import RAW_FEATURES, _physical
from .q1_development import (
    _itransformer_latent,
    _itransformer_memory_path,
    _itransformer_model,
    build_itransformer_residual_memories,
    persistence_for,
    predict_itransformer_residual_development,
    train_itransformer_residual_development,
)
from .q1_features import (
    INPUT_FEATURES,
    TARGET_FEATURES,
    Q1FeatureStore,
    _fit_training_statistics,
    _raw,
    _static,
    load_q1_features,
)
from .q1_model import Q1SequenceDataset, _loader, _pair, _strict_origins, load_q1_model, predict_q1_dataset
from .q1_online import _forecast, _safe_update

V3_ROOT = Q1_ROOT / "v3"


def _v3_directories() -> None:
    for relative in (
        "artifacts",
        "figures/png",
        "figures/pdf",
        "logs",
        "manuscript",
        "notebooks",
        "results/tables",
    ):
        (V3_ROOT / relative).mkdir(parents=True, exist_ok=True)


def _sphere(latitude: np.ndarray, longitude: np.ndarray) -> np.ndarray:
    lat = np.deg2rad(latitude.astype(np.float64))
    lon = np.deg2rad(longitude.astype(np.float64))
    return np.column_stack((np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)))


def _sha256(path: Path) -> str:
    return sha256_file(path)


def verify_v3_lock() -> dict[str, Any]:
    lock_path = V3_ROOT / "LOCKED_PROTOCOL.json"
    if not lock_path.exists():
        raise RuntimeError("Fresh-confirmation v3 protocol has not been locked")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    for name, expected in lock["files"].items():
        if _sha256(V3_ROOT / name) != expected:
            raise RuntimeError(f"Locked v3 protocol file changed: {name}")
    canonical = {key: value for key, value in lock.items() if key != "combined_sha256"}
    observed = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if observed != lock["combined_sha256"]:
        raise RuntimeError("Fresh-confirmation v3 combined hash is invalid")
    return lock


def prepare_v3_protocol(selected_model: str, selection_table: Path) -> Path:
    """Freeze a fresh-cell protocol after development selection and before value access."""
    _v3_directories()
    predecessor = verify_protocol_lock()
    lock_path = V3_ROOT / "LOCKED_PROTOCOL.json"
    if lock_path.exists():
        verify_v3_lock()
        return lock_path
    table = pd.read_csv(selection_table).sort_values("rmse_z")
    if selected_model not in set(table.model):
        raise ValueError(f"Selected model is absent from {selection_table}")
    if table.iloc[0].model != selected_model:
        raise ValueError(
            f"Fresh confirmation may only lock the development winner ({table.iloc[0].model}), not {selected_model}"
        )
    if selected_model != "PMWM-IR event-aware":
        raise ValueError(
            "The persistence-residual iTransformer memory did not win development; v3 is not justified"
        )
    non_memory = table[~table.model.str.startswith("PMWM-")]
    strongest_baseline = str(non_memory.iloc[0].model)

    config = load_q1_config()
    dataset = xr.open_zarr(
        config["data"]["source_url"], chunks={}, consolidated=True, storage_options={"token": "anon"}
    )
    latitudes = np.asarray(dataset.latitude.values, dtype=np.float64)
    longitudes = np.asarray(dataset.longitude.values, dtype=np.float64)
    v2_sites = pd.read_csv(Q1_ROOT / "sites.csv")
    excluded = {
        (int(row.latitude_index), int(row.longitude_index)) for row in v2_sites.itertuples(index=False)
    }
    pilot_path = Q1_ROOT.parent / "artifacts" / "era5_anchor_stream_1959_2022.npz"
    if pilot_path.exists():
        with np.load(pilot_path, allow_pickle=False) as pilot:
            excluded.update(
                {
                    (
                        int(np.abs(latitudes - latitude).argmin()),
                        int(np.abs(longitudes - (longitude % 360)).argmin()),
                    )
                    for latitude, longitude in zip(pilot["source_lat"], pilot["source_lon"])
                }
            )
    polar = int(config["data"]["site_design"]["excluded_polar_rows"])
    candidates = [
        (lat_index, lon_index)
        for lat_index in range(polar, len(latitudes) - polar)
        for lon_index in range(len(longitudes))
        if (lat_index, lon_index) not in excluded
    ]
    candidate_xyz = _sphere(
        np.asarray([latitudes[i] for i, _ in candidates]),
        np.asarray([longitudes[j] for _, j in candidates]),
    )
    existing = list(excluded)
    existing_xyz = _sphere(
        np.asarray([latitudes[i] for i, _ in existing]),
        np.asarray([longitudes[j] for _, j in existing]),
    )
    nearest_distance = np.min(1.0 - candidate_xyz @ existing_xyz.T, axis=1)
    chosen: list[int] = []
    for _ in range(32):
        index = int(np.argmax(nearest_distance))
        chosen.append(index)
        nearest_distance = np.minimum(nearest_distance, 1.0 - candidate_xyz @ candidate_xyz[index])
        nearest_distance[chosen] = -1.0
    rows = []
    for position, index in enumerate(chosen):
        latitude_index, longitude_index = candidates[index]
        longitude = float(longitudes[longitude_index])
        rows.append(
            {
                "cell_id": f"V3{position:03d}",
                "partition": "fresh_confirmatory",
                "latitude_index": latitude_index,
                "longitude_index": longitude_index,
                "latitude": float(latitudes[latitude_index]),
                "longitude": float(((longitude + 180.0) % 360.0) - 180.0),
            }
        )
    sites_path = V3_ROOT / "sites.csv"
    pd.DataFrame(rows).to_csv(sites_path, index=False)
    selection_copy = V3_ROOT / "DEVELOPMENT_SELECTION.csv"
    table.to_csv(selection_copy, index=False)

    v3_config = copy.deepcopy({key: value for key, value in config.items() if not key.startswith("_")})
    v3_config["protocol"] = {
        "id": "pmwm-q1-v3-fresh-2026-07-20",
        "seed": 20260721,
        "status": "locked-before-fresh-cell-value-access",
        "predecessor_protocol_hash": predecessor["combined_sha256"],
        "selected_model": selected_model,
        "selected_backbone": "persistence-residual iTransformer",
        "strongest_development_baseline": strongest_baseline,
        "selection_source": "DEVELOPMENT_SELECTION.csv",
    }
    v3_config["data"]["site_design"] = {
        "fresh_confirmatory_cells": 32,
        "selection": "farthest-point design excluding every pilot and v2 cell",
        "normalization": "per-cell 1959-1994 Fourier climatology and robust scale",
    }
    v3_config["evaluation"]["primary_contrasts"] = [
        "PMWM-IR event-aware minus persistence-residual iTransformer",
        "PMWM-IR event-aware minus persistence",
        f"PMWM-IR event-aware minus {strongest_baseline}",
    ]
    config_path = V3_ROOT / "config_v3.yaml"
    config_path.write_text(yaml.safe_dump(v3_config, sort_keys=False), encoding="utf-8")
    protocol_path = V3_ROOT / "PROTOCOL.md"
    protocol_path.write_text(
        f"""# Locked fresh-confirmation protocol: PMWM Q1 v3

Protocol ID: `pmwm-q1-v3-fresh-2026-07-20`

This protocol was frozen after model selection on the v2 development cells and before any values at the 32 v3 cells were accessed. The opened v2 confirmation remains reported as a failed persistence gate and is never reused for v3 selection.

## Locked candidate and decision rule

The locked candidate is **{selected_model}**, selected because it had the lowest aggregate RMSE in `DEVELOPMENT_SELECTION.csv`. The strongest non-memory development comparator is **{strongest_baseline}**. The candidate must beat that comparator, its matched persistence-residual iTransformer, and raw persistence on 2017-2022 fresh-cell data. Each paired 95% moving-block bootstrap interval must have a strictly positive lower bound for competitor MSE minus candidate MSE; effects must also be positive for a majority of seeds and cells.

## Data boundary

The 32 cells are a deterministic global farthest-point sample excluding all pilot and v2 cells. Models and memory are fitted only on the original 64 training cells in 1959-1994 and calibrated in 1995-2004. Per-cell Fourier climatology and robust scale may use 1959-1994 observations at a v3 cell; therefore this is historical-normal adaptation to a new location, not strict zero-shot spatial transfer. Forecast targets are 2017-2022 and were untouched at lock time. Context and all forecast horizons remain inside the confirmation interval.

## Reporting

Five seeds, 32 cells, four variables, four horizons, matched 2,048-slot memory ablations, 5,000 paired 28-day block-bootstrap replicates, BH-adjusted secondary tests, physical-unit metrics, calibration, efficiency, and every negative result are reported. No post-confirmation model changes are allowed under this protocol ID.
""",
        encoding="utf-8",
    )
    files = [config_path, protocol_path, sites_path, selection_copy]
    lock: dict[str, Any] = {
        "protocol_id": v3_config["protocol"]["id"],
        "status": "LOCKED BEFORE FRESH CELL VALUE ACCESS",
        "files": {path.name: _sha256(path) for path in files},
        "predecessor_protocol_hash": predecessor["combined_sha256"],
        "selected_model": selected_model,
        "strongest_development_baseline": strongest_baseline,
        "fresh_cells": len(rows),
    }
    canonical = json.dumps(lock, sort_keys=True, separators=(",", ":")).encode()
    lock["combined_sha256"] = hashlib.sha256(canonical).hexdigest()
    lock_path.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    return lock_path


def load_v3_config() -> dict[str, Any]:
    verify_v3_lock()
    with (V3_ROOT / "config_v3.yaml").open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def stream_v3_era5(force: bool = False) -> Path:
    """Stream only the 32 fresh cells; the full ERA5 archive is never materialized."""
    _v3_directories()
    lock = verify_v3_lock()
    output = V3_ROOT / "artifacts" / "era5_v3_stream.npz"
    manifest = V3_ROOT / "artifacts" / "stream_manifest.json"
    if output.exists() and manifest.exists() and not force:
        return output
    config = load_v3_config()
    sites = pd.read_csv(V3_ROOT / "sites.csv")
    variables = list(config["data"]["variables"])
    dataset = xr.open_zarr(
        config["data"]["source_url"], chunks={}, consolidated=True, storage_options={"token": "anon"}
    )
    site_coord = xr.DataArray(np.arange(len(sites)), dims="site")
    latitude = xr.DataArray(sites.latitude_index.to_numpy(), dims="site", coords={"site": site_coord})
    longitude = xr.DataArray(sites.longitude_index.to_numpy(), dims="site", coords={"site": site_coord})
    blocks: list[np.ndarray] = []
    times: list[np.ndarray] = []
    rows = []
    width = int(config["data"]["stream_block_years"])
    for start in range(int(config["data"]["start_year"]), int(config["data"]["end_year"]) + 1, width):
        end = min(start + width - 1, int(config["data"]["end_year"]))
        request = (
            dataset[variables]
            .sel(time=slice(f"{start}-01-01", f"{end}-12-31T18:00:00"))
            .isel(latitude=latitude, longitude=longitude)
        )
        began = time.perf_counter()
        with dask.config.set(scheduler="threads", num_workers=16):
            block = request.compute()
        values = _physical(block, variables)
        block_time = np.asarray(block.time.values).astype("datetime64[ns]")
        blocks.append(values)
        times.append(block_time)
        rows.append(
            {
                "start_year": start,
                "end_year": end,
                "time_steps": len(block_time),
                "retained_mb": values.nbytes / 1e6,
                "elapsed_seconds": time.perf_counter() - began,
            }
        )
        print(f"v3 stream {start}-{end}: {len(block_time):,} x {len(sites)}", flush=True)
    data = np.concatenate(blocks)
    time_values = np.concatenate(times)
    missing = np.argwhere(~np.isfinite(data))
    if len(missing):
        allowed = (missing[:, 0] < 2) & (missing[:, 2] == 4)
        if not allowed.all():
            raise ValueError(f"Unexpected v3 non-finite values: {len(missing)}")
        for time_index, site_index, feature_index in missing:
            future = data[time_index + 1 :, site_index, feature_index]
            data[time_index, site_index, feature_index] = future[np.isfinite(future)][0]
    hours = time_values.astype("datetime64[h]").astype(np.int64)
    if data.shape != (93504, 32, len(RAW_FEATURES)) or not np.all(np.diff(hours) == 6):
        raise ValueError(f"Invalid v3 stream shape or cadence: {data.shape}")
    np.savez_compressed(
        output,
        data=data,
        time=time_values.astype(np.int64),
        cell_id=sites.cell_id.to_numpy(dtype=str),
        partition=sites.partition.to_numpy(dtype=str),
        latitude=sites.latitude.to_numpy(np.float32),
        longitude=sites.longitude.to_numpy(np.float32),
        features=np.asarray(RAW_FEATURES),
    )
    pd.DataFrame(rows).to_csv(V3_ROOT / "logs" / "stream_blocks.csv", index=False)
    payload = {
        "protocol_hash": lock["combined_sha256"],
        "source": config["data"]["source_url"],
        "shape": list(data.shape),
        "time_start": str(time_values[0]),
        "time_end": str(time_values[-1]),
        "artifact_sha256": sha256_file(output),
        "finite_fraction": float(np.isfinite(data).mean()),
        "boundary_repairs": int(len(missing)),
        "blocks": rows,
    }
    (manifest).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return output


def _load_v3_stream() -> dict[str, np.ndarray]:
    with np.load(V3_ROOT / "artifacts" / "era5_v3_stream.npz", allow_pickle=False) as archive:
        payload = {key: archive[key] for key in archive.files}
    payload["time"] = payload["time"].astype("datetime64[ns]")
    return payload


def prepare_v3_features(force: bool = False) -> Path:
    """Fit local normals on pre-confirmation years; never fit on 2017-2022 labels."""
    lock = verify_v3_lock()
    output = V3_ROOT / "artifacts" / "feature_store.npz"
    manifest = V3_ROOT / "artifacts" / "feature_manifest.json"
    if output.exists() and manifest.exists() and not force:
        return output
    stream_v3_era5(force=force)
    stream = _load_v3_stream()
    inputs, targets, physical = _raw(stream["data"])
    dates = pd.DatetimeIndex(stream["time"])
    fit_mask = (dates.year >= 1959) & (dates.year <= 1994)
    design = time_design(stream["time"])
    sites = np.arange(len(stream["cell_id"]), dtype=np.int64)
    input_coefficient, input_scale = _fit_training_statistics(inputs, design, fit_mask, sites)
    target_coefficient, target_scale = _fit_training_statistics(targets, design, fit_mask, sites)
    input_seasonal = np.einsum("td,dsf->tsf", design, input_coefficient, optimize=True).astype(np.float32)
    target_seasonal = np.einsum("td,dsf->tsf", design, target_coefficient, optimize=True).astype(np.float32)
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
    )
    payload = {
        "protocol_hash": lock["combined_sha256"],
        "artifact_sha256": sha256_file(output),
        "input_shape": list(input_z.shape),
        "fit_years": [1959, 1994],
        "confirmation_years_used_for_statistics": 0,
        "normalization": "exact per-cell Fourier climatology and robust scale from 1959-1994",
        "claim_boundary": "historical-normal adaptation at a new location, not strict zero-shot transfer",
    }
    manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return output


def load_v3_features() -> Q1FeatureStore:
    with np.load(V3_ROOT / "artifacts" / "feature_store.npz", allow_pickle=False) as archive:
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


def make_v3_dataset(config: dict[str, Any], store: Q1FeatureStore | None = None) -> Q1SequenceDataset:
    store = store or load_v3_features()
    horizons = [int(value) for value in config["model"]["horizons"]]
    origins = _strict_origins(
        store,
        tuple(config["splits"]["confirmatory"]),
        int(config["model"]["context_steps"]),
        max(horizons),
        int(config["splits"]["evaluation_origin_stride_days"]),
    )
    paired_origin, paired_site = _pair(origins, np.arange(len(store.cell_id), dtype=np.int64))
    return Q1SequenceDataset(
        store,
        paired_origin,
        paired_site,
        int(config["model"]["context_steps"]),
        horizons,
    )


@torch.no_grad()
def predict_v3_pmwm(config: dict[str, Any], seed: int, force: bool = False) -> Path:
    output = V3_ROOT / "artifacts" / f"predictions_pmwm_r_seed{seed}.npz"
    if output.exists() and not force:
        return output
    store = load_v3_features()
    dataset = make_v3_dataset(config, store)
    model = _itransformer_model(config, load_q1_features())
    checkpoint = torch.load(
        Q1_ROOT / "checkpoints" / f"development_itransformer_residual_seed{seed}.pt",
        map_location="cpu",
        weights_only=False,
    )
    model.load_state_dict(checkpoint["state_dict"])
    device = torch.device(device_name())
    model = model.to(device).eval()
    horizons = [int(value) for value in config["model"]["horizons"]]
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
        baseline = torch.from_numpy(
            persistence_for(origin.numpy(), site.numpy(), store, horizons)
        ).to(device)
        correction = model(x, static)
        parts["mean"].append((baseline + correction).float().cpu().numpy())
        parts["logvar"].append(np.zeros_like(correction.float().cpu().numpy()))
        parts["latent"].append(_itransformer_latent(model, x, static).float().cpu().numpy())
        parts["baseline"].append(baseline.float().cpu().numpy())
        parts["target_z"].append(target.numpy())
        parts["origin"].append(origin.numpy())
        parts["site"].append(site.numpy())
    prediction = {key: np.concatenate(values) for key, values in parts.items()}
    for kind in ("full", "uniform", "reservoir"):
        memory_path, calibration_path = _itransformer_memory_path(kind, seed)
        bank = load_memory(memory_path)
        with np.load(calibration_path, allow_pickle=False) as archive:
            calibration = {key: archive[key] for key in archive.files}
        forecast = memory_forecast(
            prediction, bank, calibration, int(config["memory"]["top_k"])
        )
        prediction[f"{kind}_mean"] = forecast["mean"]
        prediction[f"{kind}_variance"] = forecast["variance"]
    np.savez_compressed(output, **prediction)
    return output


@torch.no_grad()
def predict_v3_registered_baseline(
    config: dict[str, Any], name: str, seed: int, force: bool = False
) -> Path:
    output = V3_ROOT / "artifacts" / f"predictions_{name}_seed{seed}.npz"
    if output.exists() and not force:
        return output
    store = load_v3_features()
    dataset = make_v3_dataset(config, store)
    if name == "gru_backbone":
        result = predict_q1_dataset(
            load_q1_model(seed), dataset, int(config["model"]["batch_size"]), seed
        )
    else:
        model = _make_model(name, config, load_q1_features())
        checkpoint = torch.load(baseline_checkpoint(name, seed), map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
        device = torch.device(device_name())
        model = model.to(device).eval()
        parts: dict[str, list[np.ndarray]] = {
            "mean": [],
            "target_z": [],
            "origin": [],
            "site": [],
        }
        for x, target, static, origin, site in _loader(
            dataset, int(config["model"]["batch_size"]), False, seed
        ):
            parts["mean"].append(model(x.to(device), static.to(device)).float().cpu().numpy())
            parts["target_z"].append(target.numpy())
            parts["origin"].append(origin.numpy())
            parts["site"].append(site.numpy())
        result = {key: np.concatenate(values) for key, values in parts.items()}
    np.savez_compressed(output, **result)
    return output


def predict_v3_ridge(config: dict[str, Any], force: bool = False) -> Path:
    output = V3_ROOT / "artifacts" / "predictions_ridge.npz"
    if output.exists() and not force:
        return output
    store = load_v3_features()
    dataset = make_v3_dataset(config, store)
    with np.load(Q1_ROOT / "artifacts" / "ridge_model.npz", allow_pickle=False) as archive:
        coefficient = archive["coefficient"]
        intercept = archive["intercept"]
    features = _multiscale_features(dataset.origins, dataset.sites, store)
    mean = (features @ coefficient.T + intercept).reshape(
        len(features), len(config["model"]["horizons"]), -1
    )
    target = np.stack(
        [
            store.target_z[dataset.origins + int(horizon), dataset.sites]
            for horizon in config["model"]["horizons"]
        ],
        axis=1,
    )
    np.savez_compressed(
        output,
        mean=mean.astype(np.float32),
        target_z=target.astype(np.float32),
        origin=dataset.origins,
        site=dataset.sites,
    )
    return output


def train_and_predict_v3(force: bool = False) -> None:
    config = load_v3_config()
    seeds = [int(value) for value in config["model"]["seeds"]]
    # Complete every fitting and calibration operation before opening the fresh
    # confirmation feature store. The later loop is prediction-only.
    for seed in seeds:
        train_itransformer_residual_development(config, seed, force=force)
        for split in ("train", "validation"):
            predict_itransformer_residual_development(config, split, seed, force=force)
        build_itransformer_residual_memories(config, seed, force=force)
    fit_and_predict_ridge(config, force=force)
    prepare_v3_features(force=force)
    for seed in seeds:
        predict_v3_pmwm(config, seed, force=force)
        predict_v3_registered_baseline(config, "gru_backbone", seed, force=force)
    for name in BASELINE_CLASSES:
        for seed in seeds[: int(config["evaluation"]["minimum_neural_baseline_seeds"])]:
            predict_v3_registered_baseline(config, name, seed, force=force)
    predict_v3_ridge(config, force=force)


def run_v3_causal_online(config: dict[str, Any], seed: int, force: bool = False) -> Path:
    """Delayed prequential update: forecast first, insert only fully matured outcomes."""
    output = V3_ROOT / "artifacts" / f"predictions_causal_online_seed{seed}.npz"
    audit_path = V3_ROOT / "results" / "tables" / f"causal_update_audit_seed{seed}.csv"
    if output.exists() and audit_path.exists() and not force:
        return output
    prediction = _load_npz(V3_ROOT / "artifacts" / f"predictions_pmwm_r_seed{seed}.npz")
    store = load_v3_features()
    memory_path, calibration_path = _itransformer_memory_path("full", seed)
    bank = load_memory(memory_path)
    with np.load(calibration_path, allow_pickle=False) as archive:
        calibration = {key: archive[key] for key in archive.files}
    top_k = int(config["memory"]["top_k"])
    max_horizon = max(int(value) for value in config["model"]["horizons"])
    replacements = int(config["memory"]["causal_replacements_per_update"])
    unique_origins = np.unique(prediction["origin"])
    origin_to_indices = {
        int(origin): np.flatnonzero(prediction["origin"] == origin) for origin in unique_origins
    }
    mean = np.empty_like(prediction["mean"])
    rows = []
    for origin in unique_origins:
        query = origin_to_indices[int(origin)]
        mean[query] = _forecast(
            prediction["latent"][query], prediction["mean"][query], bank, calibration, top_k
        )
        matured_origin = int(origin) - max_horizon
        matured_indices = origin_to_indices.get(matured_origin)
        if matured_indices is None:
            continue
        matured = {
            key: prediction[key][matured_indices]
            for key in ("latent", "mean", "target_z", "origin", "site")
        }
        bank, inserted = _safe_update(
            bank,
            matured,
            replacements,
            int(origin),
            max_horizon,
            store.latitude,
            store.time,
        )
        rows.append(
            {
                "forecast_origin": int(origin),
                "matured_origin": matured_origin,
                "inserted": len(inserted),
                "maximum_inserted_origin": max(inserted),
                "causal_margin_steps": int(origin) - max(inserted) - max_horizon,
                "capacity": bank.capacity,
            }
        )
    np.savez_compressed(
        output,
        mean=mean,
        target_z=prediction["target_z"],
        origin=prediction["origin"],
        site=prediction["site"],
    )
    audit = pd.DataFrame(rows)
    audit.to_csv(audit_path, index=False)
    if audit.empty or (audit.causal_margin_steps < 0).any() or (audit.capacity != int(config["memory"]["capacity"])).any():
        raise RuntimeError("The v3 causal-online audit failed")
    manifest = {
        "seed": seed,
        "protocol_hash": verify_v3_lock()["combined_sha256"],
        "forecast_origins": len(unique_origins),
        "updates": len(audit),
        "insertions": int(audit.inserted.sum()),
        "minimum_causal_margin_steps": int(audit.causal_margin_steps.min()),
        "capacity": int(config["memory"]["capacity"]),
        "leakage_detected": False,
    }
    (V3_ROOT / "artifacts" / f"causal_online_seed{seed}_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return output


def run_all_v3_causal_online(force: bool = False) -> Path:
    config = load_v3_config()
    rows = []
    for seed in [int(value) for value in config["model"]["seeds"]]:
        print(f"v3 causal online seed={seed}", flush=True)
        path = run_v3_causal_online(config, seed, force=force)
        online = _load_npz(path)
        static = _load_npz(V3_ROOT / "artifacts" / f"predictions_pmwm_r_seed{seed}.npz")
        for name, mean in (
            ("Static PMWM-IR", static["full_mean"]),
            ("Causal online PMWM-IR", online["mean"]),
        ):
            rows.append({"model": name, "seed": seed, **_metrics(static["target_z"], mean)})
    output = V3_ROOT / "results" / "tables" / "causal_online_metrics.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    return output


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = predicted - actual
    actual_flat = actual.ravel().astype(np.float64)
    predicted_flat = predicted.ravel().astype(np.float64)
    actual_centered = actual_flat - actual_flat.mean()
    predicted_centered = predicted_flat - predicted_flat.mean()
    denominator = np.sqrt(
        np.square(actual_centered).sum() * np.square(predicted_centered).sum()
    )
    return {
        "rmse_z": float(np.sqrt(np.mean(np.square(error)))),
        "mae_z": float(np.mean(np.abs(error))),
        "bias_z": float(np.mean(error)),
        "acc": float(np.sum(actual_centered * predicted_centered) / denominator)
        if denominator > 0
        else np.nan,
    }


def _physical_values(
    store: Q1FeatureStore,
    origins: np.ndarray,
    sites: np.ndarray,
    horizons: list[int],
    standardized: np.ndarray,
) -> np.ndarray:
    seasonal = np.stack(
        [store.target_seasonal[origins + horizon, sites] for horizon in horizons], axis=1
    )
    transformed = seasonal + standardized * store.target_scale[sites, None, :]
    transformed[..., 3] = np.expm1(np.clip(transformed[..., 3], -10, 10))
    transformed[..., 3] = np.maximum(transformed[..., 3], 0.0)
    return transformed.astype(np.float32)


def _paired_draws(
    competitor_errors: np.ndarray,
    candidate_errors: np.ndarray,
    origins: np.ndarray,
    sites: np.ndarray,
    replicates: int,
    block_days: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Paired overlapping moving-block bootstrap over time plus cell resampling."""
    unique_origins = np.unique(origins)
    unique_sites = np.unique(sites)
    n_origin, n_site = len(unique_origins), len(unique_sites)
    if len(origins) != n_origin * n_site:
        raise ValueError("Paired bootstrap requires a complete origin-by-cell panel")
    if not np.array_equal(origins, np.repeat(unique_origins, n_site)) or not np.array_equal(
        sites, np.tile(unique_sites, n_origin)
    ):
        raise ValueError("Paired bootstrap requires origin-major, cell-minor ordering")
    origin_stride = np.diff(unique_origins)
    if not len(origin_stride) or not np.all(origin_stride == origin_stride[0]):
        raise ValueError("Paired bootstrap requires regularly spaced forecast origins")
    # ERA5 indices are six-hourly. Evaluation origins are daily (stride four), so
    # a protocol block of 28 days corresponds to 28 entries on this origin axis.
    block_length = max(1, int(round(4 * block_days / int(origin_stride[0]))))
    block_length = min(block_length, n_origin)
    difference = np.square(competitor_errors) - np.square(candidate_errors)
    difference = difference.reshape(difference.shape[0], n_origin, n_site, 4, 4)
    mean_difference = difference.mean(axis=0, dtype=np.float64)
    cumulative = np.concatenate(
        [np.zeros_like(mean_difference[:1]), np.cumsum(mean_difference, axis=0, dtype=np.float64)],
        axis=0,
    )
    full_block_sums = cumulative[block_length:] - cumulative[:-block_length]
    blocks_per_draw = int(np.ceil(n_origin / block_length))
    remainder = n_origin - (blocks_per_draw - 1) * block_length
    partial_block_sums = cumulative[remainder:] - cumulative[:-remainder]
    rng = np.random.default_rng(seed)
    draws = np.empty((replicates, 4, 4), dtype=np.float64)
    for replicate in range(replicates):
        sampled_starts = rng.integers(0, len(full_block_sums), blocks_per_draw)
        sampled_sites = rng.integers(0, n_site, n_site)
        temporal_sum = full_block_sums[sampled_starts[:-1]].sum(axis=0)
        temporal_sum += partial_block_sums[sampled_starts[-1]]
        site_counts = np.bincount(sampled_sites, minlength=n_site)
        draws[replicate] = (temporal_sum * site_counts[:, None, None]).sum(axis=0) / (n_origin * n_site)
    seed_effect = difference.mean(axis=(1, 2, 3, 4))
    cell_effect = difference.mean(axis=(0, 1, 3, 4))
    return draws, seed_effect, cell_effect


def _bh(p_values: np.ndarray) -> np.ndarray:
    order = np.argsort(p_values)
    ranked = p_values[order]
    adjusted_ranked = np.minimum.accumulate(
        (ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1]
    )[::-1]
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = np.clip(adjusted_ranked, 0, 1)
    return adjusted


def evaluate_v3(force: bool = False) -> Path:
    verify_v3_lock()
    output = V3_ROOT / "results" / "summary.json"
    if output.exists() and not force:
        return output
    config = load_v3_config()
    seeds = [int(value) for value in config["model"]["seeds"]]
    predictions = [_load_npz(V3_ROOT / "artifacts" / f"predictions_pmwm_r_seed{seed}.npz") for seed in seeds]
    actual = predictions[0]["target_z"]
    origins = predictions[0]["origin"]
    sites = predictions[0]["site"]
    model_values: dict[str, list[tuple[int, np.ndarray]]] = {
        "PMWM-IR event-aware": [(seed, item["full_mean"]) for seed, item in zip(seeds, predictions)],
        "Persistence-residual iTransformer": [(seed, item["mean"]) for seed, item in zip(seeds, predictions)],
        "PMWM-IR uniform": [(seed, item["uniform_mean"]) for seed, item in zip(seeds, predictions)],
        "PMWM-IR reservoir": [(seed, item["reservoir_mean"]) for seed, item in zip(seeds, predictions)],
        "Persistence": [(-1, predictions[0]["baseline"])],
        "Seasonal climatology": [(-1, np.zeros_like(actual))],
    }
    for seed in seeds:
        path = V3_ROOT / "artifacts" / f"predictions_gru_backbone_seed{seed}.npz"
        if path.exists():
            model_values.setdefault("GRU backbone", []).append((seed, _load_npz(path)["mean"]))
    for name in BASELINE_CLASSES:
        for seed in seeds[: int(config["evaluation"]["minimum_neural_baseline_seeds"])]:
            path = V3_ROOT / "artifacts" / f"predictions_{name}_seed{seed}.npz"
            if path.exists():
                model_values.setdefault(name, []).append((seed, _load_npz(path)["mean"]))
    ridge_path = V3_ROOT / "artifacts" / "predictions_ridge.npz"
    if ridge_path.exists():
        model_values["Ridge-AR"] = [(-1, _load_npz(ridge_path)["mean"])]
    rows = []
    for name, values in model_values.items():
        for seed, mean in values:
            rows.append({"model": name, "seed": seed, **_metrics(actual, mean)})
    seed_table = pd.DataFrame(rows)
    seed_table.to_csv(V3_ROOT / "results" / "tables" / "model_seed_metrics.csv", index=False)
    leaderboard = (
        seed_table.groupby("model", sort=False)
        .agg(
            runs=("seed", "size"),
            rmse_z_mean=("rmse_z", "mean"),
            rmse_z_sd=("rmse_z", "std"),
            mae_z_mean=("mae_z", "mean"),
            bias_z_mean=("bias_z", "mean"),
            acc_mean=("acc", "mean"),
        )
        .reset_index()
        .sort_values("rmse_z_mean")
    )
    leaderboard.to_csv(V3_ROOT / "results" / "tables" / "leaderboard.csv", index=False)

    store = load_v3_features()
    horizons = [int(value) for value in config["model"]["horizons"]]
    actual_physical = np.stack(
        [store.target_physical[origins + horizon, sites] for horizon in horizons], axis=1
    )
    units = ["degC", "hPa", "m s-1", "mm per 6 h"]
    physical_rows = []
    for name, values in model_values.items():
        for seed, mean in values:
            predicted_physical = _physical_values(store, origins, sites, horizons, mean)
            for horizon_index, horizon in enumerate(config["model"]["horizon_labels"]):
                for target_index, (target, unit) in enumerate(zip(TARGET_FEATURES, units)):
                    error = (
                        predicted_physical[:, horizon_index, target_index]
                        - actual_physical[:, horizon_index, target_index]
                    )
                    physical_rows.append(
                        {
                            "model": name,
                            "seed": seed,
                            "horizon": horizon,
                            "target": target,
                            "unit": unit,
                            "rmse": float(np.sqrt(np.mean(np.square(error)))),
                            "mae": float(np.mean(np.abs(error))),
                            "bias": float(np.mean(error)),
                        }
                    )
    pd.DataFrame(physical_rows).to_csv(
        V3_ROOT / "results" / "tables" / "physical_metrics.csv", index=False
    )

    calibration_rows = []
    extreme_rows = []
    probabilistic_rows = []
    threshold = float(config["evaluation"]["extreme_z_threshold"])
    extreme_actual = np.max(np.abs(actual), axis=(1, 2)) >= threshold
    for seed, prediction in zip(seeds, predictions):
        variance = np.maximum(prediction["full_variance"], 1e-6)
        sigma = np.sqrt(variance)
        error = actual - prediction["full_mean"]
        interval_scores = []
        for nominal, z_value in (
            (0.50, 0.67448975),
            (0.80, 1.28155157),
            (0.90, 1.64485363),
            (0.95, 1.95996398),
        ):
            covered = np.abs(actual - prediction["full_mean"]) <= z_value * sigma
            calibration_rows.append(
                {
                    "seed": seed,
                    "nominal": nominal,
                    "empirical": float(covered.mean()),
                    "mean_width_z": float((2 * z_value * sigma).mean()),
                }
            )
            alpha = 1.0 - nominal
            lower = prediction["full_mean"] - z_value * sigma
            upper = prediction["full_mean"] + z_value * sigma
            interval_score = (
                upper
                - lower
                + (2 / alpha) * (lower - actual) * (actual < lower)
                + (2 / alpha) * (actual - upper) * (actual > upper)
            )
            interval_scores.append((alpha / 2) * interval_score)
        probabilistic_rows.append(
            {
                "seed": seed,
                "gaussian_crps_z": float(
                    gaussian_crps(actual, prediction["full_mean"], variance).mean()
                ),
                "gaussian_nll_z": float(
                    (0.5 * (np.log(2 * np.pi * variance) + np.square(error) / variance)).mean()
                ),
                "weighted_interval_score_z": float(np.mean(np.stack(interval_scores), axis=(0, 1, 2, 3))),
            }
        )
        score = np.max(np.abs(prediction["full_mean"]), axis=(1, 2))
        extreme_rows.append(
            {
                "seed": seed,
                "threshold_z": threshold,
                "prevalence": float(extreme_actual.mean()),
                "auroc": float(roc_auc_score(extreme_actual, score)),
                "auprc": float(average_precision_score(extreme_actual, score)),
                "extreme_rmse_z": float(
                    np.sqrt(np.mean(np.square(prediction["full_mean"][extreme_actual] - actual[extreme_actual])))
                ),
                "non_extreme_rmse_z": float(
                    np.sqrt(np.mean(np.square(prediction["full_mean"][~extreme_actual] - actual[~extreme_actual])))
                ),
            }
        )
    pd.DataFrame(calibration_rows).to_csv(
        V3_ROOT / "results" / "tables" / "calibration.csv", index=False
    )
    pd.DataFrame(extreme_rows).to_csv(
        V3_ROOT / "results" / "tables" / "extreme_events.csv", index=False
    )
    pd.DataFrame(probabilistic_rows).to_csv(
        V3_ROOT / "results" / "tables" / "probabilistic_metrics.csv", index=False
    )

    candidate = np.stack([item["full_mean"] - actual for item in predictions])
    residual = np.stack([item["mean"] - actual for item in predictions])
    persistence = np.broadcast_to((predictions[0]["baseline"] - actual)[None], candidate.shape)
    strongest_name = str(config["protocol"]["strongest_development_baseline"])
    strongest_map = {
        "Neural backbone": "gru_backbone",
        "DLinear": "dlinear",
        "TCN": "tcn",
        "PatchTST-style": "patchtst",
        "iTransformer-style": "itransformer",
    }
    if strongest_name in strongest_map:
        artifact_name = strongest_map[strongest_name]
        comparison_seeds = seeds if artifact_name == "gru_backbone" else seeds[: int(config["evaluation"]["minimum_neural_baseline_seeds"])]
        strongest = np.stack(
            [
                _load_npz(V3_ROOT / "artifacts" / f"predictions_{artifact_name}_seed{seed}.npz")["mean"]
                - actual
                for seed in comparison_seeds
            ]
        )
        strongest_candidate = candidate[: len(comparison_seeds)]
    elif strongest_name == "Ridge-AR":
        ridge_error = _load_npz(V3_ROOT / "artifacts" / "predictions_ridge.npz")["mean"] - actual
        strongest = np.broadcast_to(ridge_error[None], candidate.shape)
        strongest_candidate = candidate
    elif strongest_name == "Persistence":
        strongest = persistence
        strongest_candidate = candidate
    elif strongest_name == "Persistence-residual iTransformer":
        strongest = residual
        strongest_candidate = candidate
    else:
        raise ValueError(f"Unsupported strongest development baseline: {strongest_name}")
    contrasts: dict[str, dict[str, Any]] = {}
    detailed_rows = []
    bootstrap_payload = {}
    contrast_inputs = [
        ("residual_itransformer_minus_pmwm_ir", residual, candidate),
        ("persistence_minus_pmwm_ir", persistence, candidate),
    ]
    if strongest_name not in {"Persistence-residual iTransformer", "Persistence"}:
        contrast_inputs.append(("strongest_baseline_minus_pmwm_ir", strongest, strongest_candidate))
    for contrast, competitor, candidate_errors in contrast_inputs:
        draws, seed_effect, cell_effect = _paired_draws(
            competitor,
            candidate_errors,
            origins,
            sites,
            int(config["evaluation"]["bootstrap_replicates"]),
            int(config["evaluation"]["bootstrap_block_days"]),
            int(config["protocol"]["seed"]) + len(contrasts),
        )
        aggregate = draws.mean(axis=(1, 2))
        point_grid = (np.square(competitor) - np.square(candidate_errors)).mean(axis=(0, 1))
        raw_p = []
        local_rows = []
        for horizon_index, horizon in enumerate(config["model"]["horizon_labels"]):
            for target_index, target in enumerate(TARGET_FEATURES):
                values = draws[:, horizon_index, target_index]
                probability = float((values > 0).mean())
                p_value = min(1.0, 2 * min(probability, 1 - probability))
                raw_p.append(p_value)
                local_rows.append(
                    {
                        "contrast": contrast,
                        "horizon": horizon,
                        "target": target,
                        "mse_reduction_z": float(point_grid[horizon_index, target_index]),
                        "ci_lower": float(np.quantile(values, 0.025)),
                        "ci_upper": float(np.quantile(values, 0.975)),
                        "p_value": p_value,
                    }
                )
        adjusted = _bh(np.asarray(raw_p))
        for row, p_adjusted in zip(local_rows, adjusted):
            row["p_value_bh"] = float(p_adjusted)
            row["significant_positive_bh"] = bool(row["ci_lower"] > 0 and p_adjusted < 0.05)
        detailed_rows.extend(local_rows)
        ci_half = float(t.ppf(0.975, len(seed_effect) - 1) * seed_effect.std(ddof=1) / np.sqrt(len(seed_effect)))
        contrasts[contrast] = {
            "mse_reduction_z": float(point_grid.mean()),
            "bootstrap_ci_95": [float(np.quantile(aggregate, 0.025)), float(np.quantile(aggregate, 0.975))],
            "seed_mean": float(seed_effect.mean()),
            "seed_t_interval_95": [float(seed_effect.mean() - ci_half), float(seed_effect.mean() + ci_half)],
            "positive_seed_fraction": float((seed_effect > 0).mean()),
            "positive_cell_fraction": float((cell_effect > 0).mean()),
        }
        bootstrap_payload[contrast] = aggregate
        effect_seeds = seeds[: len(seed_effect)]
        pd.DataFrame(
            {"seed": effect_seeds, "mse_reduction_z": seed_effect, "improved": seed_effect > 0}
        ).to_csv(V3_ROOT / "results" / "tables" / f"{contrast}_seed_effects.csv", index=False)
        pd.DataFrame(
            {
                "site": np.arange(len(cell_effect)),
                "cell_id": load_v3_features().cell_id,
                "mse_reduction_z": cell_effect,
                "improved": cell_effect > 0,
            }
        ).to_csv(V3_ROOT / "results" / "tables" / f"{contrast}_cell_effects.csv", index=False)
    pd.DataFrame(detailed_rows).to_csv(V3_ROOT / "results" / "tables" / "confirmatory_effects.csv", index=False)
    np.savez_compressed(V3_ROOT / "artifacts" / "bootstrap_draws.npz", **bootstrap_payload)
    sensitivity_rows = []
    primary_block_days = int(config["evaluation"]["bootstrap_block_days"])
    for block_days in (7, 14, primary_block_days, 56, 84):
        for contrast_index, (contrast, competitor, candidate_errors) in enumerate(contrast_inputs):
            if block_days == primary_block_days:
                sensitivity_aggregate = bootstrap_payload[contrast]
            else:
                sensitivity_draws, _, _ = _paired_draws(
                    competitor,
                    candidate_errors,
                    origins,
                    sites,
                    int(config["evaluation"]["bootstrap_replicates"]),
                    block_days,
                    int(config["protocol"]["seed"]) + 10_000 + 100 * block_days + contrast_index,
                )
                sensitivity_aggregate = sensitivity_draws.mean(axis=(1, 2))
            sensitivity_rows.append(
                {
                    "contrast": contrast,
                    "block_days": block_days,
                    "prespecified_primary": block_days == primary_block_days,
                    "replicates": len(sensitivity_aggregate),
                    "mse_reduction_z": float(
                        (np.square(competitor) - np.square(candidate_errors)).mean()
                    ),
                    "ci_lower": float(np.quantile(sensitivity_aggregate, 0.025)),
                    "ci_upper": float(np.quantile(sensitivity_aggregate, 0.975)),
                    "probability_positive": float((sensitivity_aggregate > 0).mean()),
                }
            )
    pd.DataFrame(sensitivity_rows).to_csv(
        V3_ROOT / "results" / "tables" / "block_length_sensitivity.csv", index=False
    )
    primary_supported = all(
        value["bootstrap_ci_95"][0] > 0
        and value["positive_seed_fraction"] > 0.5
        and value["positive_cell_fraction"] > 0.5
        for value in contrasts.values()
    )
    summary = {
        "protocol_id": config["protocol"]["id"],
        "protocol_hash": verify_v3_lock()["combined_sha256"],
        "fresh_confirmatory_samples": int(len(actual)),
        "fresh_cells": int(len(np.unique(sites))),
        "seeds": seeds,
        "contrasts": contrasts,
        "strongest_development_baseline": strongest_name,
        "primary_hypothesis_supported": primary_supported,
        "pmwm_rmse_z_mean": float(leaderboard.loc[leaderboard.model == "PMWM-IR event-aware", "rmse_z_mean"].iloc[0]),
        "persistence_rmse_z": float(leaderboard.loc[leaderboard.model == "Persistence", "rmse_z_mean"].iloc[0]),
        "mean_90_interval_coverage": float(pd.DataFrame(calibration_rows).query("nominal == 0.90").empirical.mean()),
        "mean_extreme_auroc": float(pd.DataFrame(extreme_rows).auroc.mean()),
        "mean_gaussian_crps_z": float(pd.DataFrame(probabilistic_rows).gaussian_crps_z.mean()),
        "mean_gaussian_nll_z": float(pd.DataFrame(probabilistic_rows).gaussian_nll_z.mean()),
        "claim_boundary": "historical-normal adaptation at unseen cells; not strict zero-shot transfer",
    }
    q1_json("v3/results/summary.json", summary)
    return output

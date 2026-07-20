from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import dask
import numpy as np
import pandas as pd
import xarray as xr

from .common import sha256_file
from .q1_common import Q1_ROOT, ensure_q1_directories, q1_json, verify_protocol_lock

RAW_FEATURES = ["temperature_c", "pressure_hpa", "u_wind_ms", "v_wind_ms", "precipitation_mm"]


def _physical(block: xr.Dataset, variables: list[str]) -> np.ndarray:
    values = np.stack([np.asarray(block[name].values) for name in variables], axis=-1).astype(np.float32)
    values[..., 0] -= np.float32(273.15)
    values[..., 1] /= np.float32(100.0)
    values[..., 4] = np.maximum(values[..., 4] * np.float32(1000.0), 0.0)
    return values


def stream_q1_era5(config: dict[str, Any], force: bool = False) -> Path:
    ensure_q1_directories()
    lock = verify_protocol_lock()
    output = Q1_ROOT / "artifacts" / "era5_q1_stream.npz"
    manifest_path = Q1_ROOT / "artifacts" / "stream_manifest.json"
    if output.exists() and manifest_path.exists() and not force:
        return output

    sites = pd.read_csv(Q1_ROOT / "sites.csv")
    cfg = config["data"]
    variables = list(cfg["variables"])
    dataset = xr.open_zarr(
        cfg["source_url"], chunks={}, consolidated=True, storage_options={"token": "anon"}
    )
    site_coord = xr.DataArray(np.arange(len(sites)), dims="site")
    latitude = xr.DataArray(sites.latitude_index.to_numpy(), dims="site", coords={"site": site_coord})
    longitude = xr.DataArray(sites.longitude_index.to_numpy(), dims="site", coords={"site": site_coord})
    blocks: list[np.ndarray] = []
    times: list[np.ndarray] = []
    logs: list[dict[str, Any]] = []
    width = int(cfg["stream_block_years"])
    for start in range(int(cfg["start_year"]), int(cfg["end_year"]) + 1, width):
        end = min(start + width - 1, int(cfg["end_year"]))
        request = (
            dataset[variables]
            .sel(time=slice(f"{start}-01-01", f"{end}-12-31T18:00:00"))
            .isel(latitude=latitude, longitude=longitude)
        )
        began = time.perf_counter()
        with dask.config.set(scheduler="threads", num_workers=16):
            block = request.compute()
        values = _physical(block, variables)
        block_times = np.asarray(block.time.values).astype("datetime64[ns]")
        elapsed = time.perf_counter() - began
        blocks.append(values)
        times.append(block_times)
        logs.append(
            {
                "start_year": start,
                "end_year": end,
                "time_steps": len(block_times),
                "retained_mb": values.nbytes / 1e6,
                "elapsed_seconds": elapsed,
            }
        )
        print(f"q1 stream {start}-{end}: {len(block_times):,} × {len(sites)} in {elapsed:.1f}s", flush=True)

    data = np.concatenate(blocks)
    time_values = np.concatenate(times)
    missing = np.argwhere(~np.isfinite(data))
    if len(missing):
        allowed = (missing[:, 0] < 2) & (missing[:, 2] == 4)
        if not allowed.all():
            raise ValueError(f"Unexpected non-finite ERA5 values: {len(missing)}")
        for time_index, site_index, feature_index in missing:
            future = data[time_index + 1 :, site_index, feature_index]
            data[time_index, site_index, feature_index] = future[np.isfinite(future)][0]
    hours = time_values.astype("datetime64[h]").astype(np.int64)
    if not np.all(np.diff(hours) == 6):
        raise ValueError("Q1 ERA5 stream is not a complete six-hour sequence")
    if data.shape != (93504, len(sites), len(RAW_FEATURES)):
        raise ValueError(f"Unexpected Q1 stream shape: {data.shape}")

    np.savez_compressed(
        output,
        data=data,
        time=time_values.astype(np.int64),
        cell_id=sites.cell_id.to_numpy(dtype=str),
        partition=sites.partition.to_numpy(dtype=str),
        latitude=sites.latitude.to_numpy(np.float32),
        longitude=sites.longitude.to_numpy(np.float32),
        latitude_index=sites.latitude_index.to_numpy(np.int16),
        longitude_index=sites.longitude_index.to_numpy(np.int16),
        features=np.asarray(RAW_FEATURES),
    )
    pd.DataFrame(logs).to_csv(Q1_ROOT / "logs" / "stream_blocks.csv", index=False)
    q1_json(
        "artifacts/stream_manifest.json",
        {
            "protocol_hash": lock["combined_sha256"],
            "source": cfg["source_url"],
            "shape": list(data.shape),
            "time_start": str(time_values[0]),
            "time_end": str(time_values[-1]),
            "finite_fraction": float(np.isfinite(data).mean()),
            "boundary_repairs": int(len(missing)),
            "partition_counts": sites.partition.value_counts().sort_index().to_dict(),
            "artifact_sha256": sha256_file(output),
            "blocks": logs,
        },
    )
    return output


def load_q1_stream() -> dict[str, np.ndarray]:
    with np.load(Q1_ROOT / "artifacts" / "era5_q1_stream.npz", allow_pickle=False) as archive:
        payload = {key: archive[key] for key in archive.files}
    payload["time"] = payload["time"].astype("datetime64[ns]")
    return payload

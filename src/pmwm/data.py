from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import dask
import numpy as np
import pandas as pd
import xarray as xr

from .common import ROOT, atomic_json, ensure_directories, sha256_file

RAW_FEATURES = ["temperature_c", "pressure_hpa", "u_wind_ms", "v_wind_ms", "precipitation_mm"]


def _nearest_indices(ds: xr.Dataset, sites: list[dict[str, Any]]) -> tuple[list[int], list[int]]:
    latitudes = np.asarray(ds.latitude.values)
    longitudes = np.asarray(ds.longitude.values)
    lat_idx = [int(np.abs(latitudes - site["lat"]).argmin()) for site in sites]
    lon_idx = [int(np.abs(longitudes - (site["lon"] % 360)).argmin()) for site in sites]
    return lat_idx, lon_idx


def _to_physical(block: xr.Dataset, variables: list[str]) -> np.ndarray:
    values = np.stack([np.asarray(block[name].values) for name in variables], axis=-1).astype(np.float32)
    values[..., 0] -= np.float32(273.15)
    values[..., 1] /= np.float32(100.0)
    values[..., 4] = np.maximum(values[..., 4] * np.float32(1000.0), 0.0)
    return values


def _quality_report(data: np.ndarray, times: np.ndarray, site_names: list[str]) -> dict[str, Any]:
    dt_hours = np.diff(times.astype("datetime64[h]").astype(np.int64))
    duplicate_grids = len(site_names) - len(set(site_names))
    ranges = {
        "temperature_c": [-100.0, 65.0],
        "pressure_hpa": [800.0, 1100.0],
        "u_wind_ms": [-100.0, 100.0],
        "v_wind_ms": [-100.0, 100.0],
        "precipitation_mm": [0.0, 500.0],
    }
    checks: dict[str, Any] = {}
    for feature_idx, feature in enumerate(RAW_FEATURES):
        values = data[..., feature_idx]
        low, high = ranges[feature]
        checks[feature] = {
            "finite_fraction": float(np.isfinite(values).mean()),
            "minimum": float(np.nanmin(values)),
            "maximum": float(np.nanmax(values)),
            "outside_plausible_fraction": float(((values < low) | (values > high)).mean()),
        }
    return {
        "n_time": int(len(times)),
        "n_sites": int(data.shape[1]),
        "time_start": str(times[0]),
        "time_end": str(times[-1]),
        "six_hour_step_fraction": float((dt_hours == 6).mean()),
        "monotonic_time": bool(np.all(dt_hours > 0)),
        "duplicate_site_name_count": duplicate_grids,
        "feature_checks": checks,
    }


def stream_era5(config: dict[str, Any], force: bool = False) -> Path:
    """Stream chronological ERA5 blocks and retain only the requested anchor cells."""
    ensure_directories()
    output = ROOT / "artifacts" / "era5_anchor_stream_1959_2022.npz"
    manifest_path = ROOT / "artifacts" / "stream_manifest.json"
    log_path = ROOT / "logs" / "stream_blocks.csv"
    if output.exists() and manifest_path.exists() and not force:
        return output

    data_cfg = config["data"]
    variables = list(data_cfg["variables"])
    sites = list(data_cfg["sites"])
    ds = xr.open_zarr(
        data_cfg["source_url"],
        chunks={},
        consolidated=True,
        storage_options={"token": "anon"},
    )
    lat_idx, lon_idx = _nearest_indices(ds, sites)
    site_index = xr.DataArray(np.arange(len(sites)), dims="site")
    lat_selection = xr.DataArray(lat_idx, dims="site", coords={"site": site_index})
    lon_selection = xr.DataArray(lon_idx, dims="site", coords={"site": site_index})

    start_year = int(data_cfg["start_year"])
    end_year = int(data_cfg["end_year"])
    width = int(data_cfg["stream_block_years"])
    blocks: list[np.ndarray] = []
    time_blocks: list[np.ndarray] = []
    stream_rows: list[dict[str, Any]] = []

    for block_start in range(start_year, end_year + 1, width):
        block_end = min(block_start + width - 1, end_year)
        request = (
            ds[variables]
            .sel(time=slice(f"{block_start}-01-01", f"{block_end}-12-31T18:00:00"))
            .isel(latitude=lat_selection, longitude=lon_selection)
        )
        started = time.perf_counter()
        with dask.config.set(scheduler="threads", num_workers=int(data_cfg["dask_workers"])):
            block = request.compute()
        elapsed = time.perf_counter() - started
        values = _to_physical(block, variables)
        times = np.asarray(block.time.values).astype("datetime64[ns]")
        blocks.append(values)
        time_blocks.append(times)

        native_chunks = int(np.ceil(len(times) / 100)) * len(variables)
        logical_bytes = native_chunks * 100 * 64 * 32 * 4
        stream_rows.append(
            {
                "start_year": block_start,
                "end_year": block_end,
                "time_steps": len(times),
                "source_chunks_estimate": native_chunks,
                "source_megabytes_estimate": logical_bytes / 1e6,
                "retained_megabytes": values.nbytes / 1e6,
                "elapsed_seconds": elapsed,
                "estimated_source_MB_per_second": logical_bytes / 1e6 / max(elapsed, 1e-9),
            }
        )
        print(
            f"streamed {block_start}-{block_end}: {len(times):,} steps, "
            f"{elapsed:.1f}s, retained {values.nbytes / 1e6:.2f} MB",
            flush=True,
        )

    data = np.concatenate(blocks, axis=0)
    times = np.concatenate(time_blocks, axis=0)
    repair_log: list[dict[str, Any]] = []
    nonfinite = np.argwhere(~np.isfinite(data))
    if len(nonfinite):
        # WeatherBench2's rolling 6 h precipitation derivative is undefined for
        # the first two archive timestamps because no preceding accumulation is
        # available. No other variable/time is repaired; unexpected gaps fail.
        permitted = (nonfinite[:, 0] < 2) & (nonfinite[:, 2] == 4)
        if not permitted.all():
            raise ValueError(
                f"ERA5 stream contains {len(nonfinite)} non-finite values outside the documented precipitation boundary"
            )
        for time_index, site_index_value, feature_index in nonfinite:
            later = data[time_index + 1 :, site_index_value, feature_index]
            valid = later[np.isfinite(later)]
            if not len(valid):
                raise ValueError("Cannot repair precipitation boundary: no subsequent finite value")
            data[time_index, site_index_value, feature_index] = valid[0]
        repair_log.append(
            {
                "feature": RAW_FEATURES[4],
                "count": int(len(nonfinite)),
                "locations": "first two archive timestamps across 16 anchors",
                "policy": "nearest subsequent finite value (boundary backfill)",
                "reason": "rolling 6 h accumulation is undefined without pre-archive values",
            }
        )
    if not np.all(np.diff(times.astype("datetime64[h]").astype(np.int64)) == 6):
        raise ValueError("ERA5 stream is not a complete monotonic 6-hour sequence")

    source_lat = np.asarray(ds.latitude.values)[lat_idx]
    source_lon = np.asarray(ds.longitude.values)[lon_idx]
    np.savez_compressed(
        output,
        data=data,
        time=times.astype("datetime64[ns]").astype(np.int64),
        site_names=np.asarray([site["name"] for site in sites]),
        site_lat=np.asarray([site["lat"] for site in sites], dtype=np.float32),
        site_lon=np.asarray([site["lon"] for site in sites], dtype=np.float32),
        source_lat=source_lat.astype(np.float32),
        source_lon=source_lon.astype(np.float32),
        zones=np.asarray([site["zone"] for site in sites]),
        holdout=np.asarray([site["holdout"] for site in sites], dtype=bool),
        features=np.asarray(RAW_FEATURES),
    )
    pd.DataFrame(stream_rows).to_csv(log_path, index=False)
    quality = _quality_report(data, times, [site["name"] for site in sites])
    manifest = {
        "source": data_cfg["source_name"],
        "source_url": data_cfg["source_url"],
        "selection_policy": "nearest 64x32 conservative ERA5 grid cell for each anchor",
        "raw_variables": variables,
        "retained_features": RAW_FEATURES,
        "source_grid": {"longitude": int(ds.sizes["longitude"]), "latitude": int(ds.sizes["latitude"])},
        "source_time_steps": int(ds.sizes["time"]),
        "retained_shape": list(data.shape),
        "retained_uncompressed_bytes": int(data.nbytes),
        "stream_blocks": stream_rows,
        "repairs": repair_log,
        "quality": quality,
        "site_grid_mapping": [
            {
                "name": site["name"],
                "requested_lat": site["lat"],
                "requested_lon": site["lon"],
                "source_lat": float(source_lat[i]),
                "source_lon": float(((source_lon[i] + 180) % 360) - 180),
                "holdout": bool(site["holdout"]),
            }
            for i, site in enumerate(sites)
        ],
    }
    manifest["artifact_sha256"] = sha256_file(output)
    atomic_json(manifest_path, manifest)
    return output


def load_stream(path: Path | None = None) -> dict[str, np.ndarray]:
    path = path or ROOT / "artifacts" / "era5_anchor_stream_1959_2022.npz"
    with np.load(path, allow_pickle=False) as archive:
        payload = {name: archive[name] for name in archive.files}
    payload["time"] = payload["time"].astype("datetime64[ns]")
    return payload

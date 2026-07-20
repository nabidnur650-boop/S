from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .memory import (
    MemoryBank,
    _consolidate_partition,
    calibrate_memory,
    load_memory,
    normalize_keys,
    save_memory,
)
from .q1_common import Q1_ROOT, ensure_q1_directories, q1_json, verify_protocol_lock
from .q1_features import load_q1_features
from .q1_model import load_q1_predictions


def _latitude_band(latitude: np.ndarray) -> np.ndarray:
    absolute = np.abs(latitude)
    return np.select([absolute < 23.5, absolute < 45.0, absolute < 66.5], [0, 1, 2], default=3).astype(np.int16)


def _partition(
    keys: np.ndarray,
    outcomes: np.ndarray,
    residuals: np.ndarray,
    scores: np.ndarray,
    months: np.ndarray,
    zones: np.ndarray,
    sites: np.ndarray,
    origins: np.ndarray,
    indices: np.ndarray,
    slots: int,
    sample: int,
    iterations: int,
    seed: int,
) -> tuple[dict[str, np.ndarray], float]:
    return _consolidate_partition(
        keys,
        outcomes,
        residuals,
        scores,
        months,
        zones,
        sites,
        origins,
        indices,
        slots,
        min(sample, len(indices)),
        iterations,
        seed,
    )


def _bank_from(parts: list[tuple[dict[str, np.ndarray], str]]) -> MemoryBank:
    fields = [
        "keys", "outcomes", "residuals", "residual_variance", "event_score",
        "month", "zone", "site", "origin", "count",
    ]
    payload = {field: np.concatenate([part[field] for part, _ in parts], axis=0) for field in fields}
    payload["kind"] = np.concatenate(
        [np.asarray([kind] * len(part["keys"])) for part, kind in parts]
    )
    return MemoryBank(**payload)


def q1_memory_path(seed: int, kind: str = "full") -> Path:
    return Q1_ROOT / "artifacts" / f"memory_{kind}_seed{seed}.npz"


def q1_calibration_path(seed: int, kind: str = "full") -> Path:
    return Q1_ROOT / "artifacts" / f"calibration_{kind}_seed{seed}.npz"


def build_q1_memory(config: dict[str, Any], seed: int, force: bool = False) -> Path:
    ensure_q1_directories()
    lock = verify_protocol_lock()
    kinds = ["full", "uniform", "reservoir"]
    if all(q1_memory_path(seed, kind).exists() and q1_calibration_path(seed, kind).exists() for kind in kinds) and not force:
        return q1_memory_path(seed)
    train = load_q1_predictions("train", seed)
    validation = load_q1_predictions("validation", seed)
    store = load_q1_features()
    cfg = config["memory"]
    rng = np.random.default_rng(seed)
    keys = normalize_keys(train["latent"])
    outcomes = train["target_z"].astype(np.float32)
    residuals = (outcomes - train["mean"]).astype(np.float32)
    scores = np.max(np.abs(outcomes), axis=(1, 2)).astype(np.float32)
    dates = pd.DatetimeIndex(store.time[train["origin"]])
    months = dates.month.to_numpy(dtype=np.int16)
    zones = _latitude_band(store.latitude[train["site"]])
    event_threshold = float(np.quantile(scores, float(cfg["event_quantile"])))
    event_indices = np.flatnonzero(scores >= event_threshold)
    regular_indices = np.flatnonzero(scores < event_threshold)
    all_indices = np.arange(len(keys), dtype=np.int64)
    began = time.perf_counter()
    regular, _ = _partition(
        keys, outcomes, residuals, scores, months, zones, train["site"], train["origin"],
        regular_indices, int(cfg["prototype_slots"]), int(cfg["kmeans_sample"]),
        int(cfg["kmeans_iterations"]), seed,
    )
    events, _ = _partition(
        keys, outcomes, residuals, scores, months, zones, train["site"], train["origin"],
        event_indices, int(cfg["event_slots"]), min(60000, int(cfg["kmeans_sample"])),
        int(cfg["kmeans_iterations"]), seed + 1,
    )
    uniform, _ = _partition(
        keys, outcomes, residuals, scores, months, zones, train["site"], train["origin"],
        all_indices, int(cfg["capacity"]), int(cfg["kmeans_sample"]),
        int(cfg["kmeans_iterations"]), seed + 2,
    )
    full_bank = _bank_from([(regular, "regular"), (events, "event")])
    uniform_bank = _bank_from([(uniform, "uniform")])
    chosen = rng.choice(len(keys), size=int(cfg["capacity"]), replace=False)
    global_variance = np.var(residuals, axis=0, ddof=1).astype(np.float32)
    reservoir_bank = MemoryBank(
        keys=keys[chosen],
        outcomes=outcomes[chosen],
        residuals=residuals[chosen],
        residual_variance=np.broadcast_to(global_variance[None], (len(chosen), *global_variance.shape)).copy(),
        event_score=scores[chosen],
        month=months[chosen],
        zone=zones[chosen],
        site=train["site"][chosen].astype(np.int16),
        origin=train["origin"][chosen],
        count=np.ones(len(chosen), dtype=np.int64),
        kind=np.asarray(["reservoir"] * len(chosen)),
    )
    banks = {"full": full_bank, "uniform": uniform_bank, "reservoir": reservoir_bank}
    calibration_summary: dict[str, Any] = {}
    for kind, bank in banks.items():
        if bank.capacity != int(cfg["capacity"]):
            raise ValueError(f"{kind} bank violates matched capacity")
        save_memory(bank, q1_memory_path(seed, kind))
        calibration = calibrate_memory(bank, validation, config)
        np.savez_compressed(q1_calibration_path(seed, kind), **calibration)
        calibration_summary[kind] = {
            "temperature": float(calibration["temperature"]),
            "gate_mean": float(calibration["gate"].mean()),
            "validation_mse_z": float(calibration["validation_mse_z"]),
        }
    q1_json(
        f"artifacts/memory_seed{seed}_manifest.json",
        {
            "seed": seed,
            "protocol_hash": lock["combined_sha256"],
            "training_contexts": len(keys),
            "capacity": int(cfg["capacity"]),
            "compression_ratio": len(keys) / int(cfg["capacity"]),
            "event_threshold_z": event_threshold,
            "event_pool": len(event_indices),
            "event_slots": int(cfg["event_slots"]),
            "matched_capacity_ablation": True,
            "calibration": calibration_summary,
            "build_seconds": time.perf_counter() - began,
        },
    )
    return q1_memory_path(seed)


def load_q1_memory(seed: int, kind: str = "full") -> MemoryBank:
    return load_memory(q1_memory_path(seed, kind))


def load_q1_calibration(seed: int, kind: str = "full") -> dict[str, np.ndarray]:
    with np.load(q1_calibration_path(seed, kind), allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def build_all_q1_memories(config: dict[str, Any], force: bool = False) -> None:
    for seed in [int(value) for value in config["model"]["seeds"]]:
        print(f"q1 memory seed={seed}", flush=True)
        build_q1_memory(config, seed, force=force)


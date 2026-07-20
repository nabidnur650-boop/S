from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .memory import MemoryBank, aggregate_retrieval, normalize_keys
from .q1_common import Q1_ROOT, ensure_q1_directories, q1_json, verify_protocol_lock
from .q1_features import load_q1_features
from .q1_memory import load_q1_calibration, load_q1_memory
from .q1_model import load_q1_predictions


def _retrieve_numpy(keys: np.ndarray, bank: MemoryBank, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    similarity = normalize_keys(keys) @ normalize_keys(bank.keys).T
    partial = np.argpartition(similarity, -top_k, axis=1)[:, -top_k:]
    partial_similarity = np.take_along_axis(similarity, partial, axis=1)
    order = np.argsort(partial_similarity, axis=1)[:, ::-1]
    indices = np.take_along_axis(partial, order, axis=1)
    values = np.take_along_axis(similarity, indices, axis=1)
    return values.astype(np.float32), indices.astype(np.int32)


def _forecast(
    latent: np.ndarray,
    base: np.ndarray,
    bank: MemoryBank,
    calibration: dict[str, np.ndarray],
    top_k: int,
) -> np.ndarray:
    similarity, indices = _retrieve_numpy(latent, bank, top_k)
    retrieval = aggregate_retrieval(similarity, indices, bank, float(calibration["temperature"]))
    return (base + calibration["gate"][None] * retrieval["residual"]).astype(np.float32)


def _safe_update(
    bank: MemoryBank,
    matured: dict[str, np.ndarray],
    n_new: int,
    current_origin: int,
    max_horizon: int,
    latitude: np.ndarray,
    times: np.ndarray,
) -> tuple[MemoryBank, list[int]]:
    if np.any(matured["origin"] + max_horizon > current_origin):
        raise RuntimeError("Causal memory attempted to insert an unobserved target")
    residual = matured["target_z"] - matured["mean"]
    event_score = np.max(np.abs(matured["target_z"]), axis=(1, 2))
    similarity = normalize_keys(matured["latent"]) @ normalize_keys(bank.keys).T
    novelty = 1.0 - similarity.max(axis=1)
    event_index = int(np.argmax(event_score))
    novelty[event_index] = -np.inf
    novelty_index = int(np.argmax(novelty))
    selected = np.asarray([event_index, novelty_index][:n_new], dtype=np.int64)

    origin_min = float(bank.origin.min())
    origin_span = max(float(current_origin) - origin_min, 1.0)
    recency = (bank.origin - origin_min) / origin_span
    count_priority = np.log1p(bank.count) / max(float(np.log1p(bank.count).max()), 1.0)
    event_priority = bank.event_score / max(float(bank.event_score.max()), 1e-6)
    keep_priority = 0.40 * count_priority + 0.30 * event_priority + 0.30 * recency
    keep = np.argsort(keep_priority)[-(bank.capacity - len(selected)) :]
    variance = np.var(residual, axis=0, ddof=1).astype(np.float32)
    months = pd.DatetimeIndex(times[matured["origin"][selected]]).month.to_numpy(dtype=np.int16)
    absolute_latitude = np.abs(latitude[matured["site"][selected]])
    zones = np.select(
        [absolute_latitude < 23.5, absolute_latitude < 45.0, absolute_latitude < 66.5],
        [0, 1, 2],
        default=3,
    ).astype(np.int16)
    updated = MemoryBank(
        keys=np.concatenate([bank.keys[keep], normalize_keys(matured["latent"][selected])]),
        outcomes=np.concatenate([bank.outcomes[keep], matured["target_z"][selected]]),
        residuals=np.concatenate([bank.residuals[keep], residual[selected]]),
        residual_variance=np.concatenate(
            [bank.residual_variance[keep], np.broadcast_to(variance[None], (len(selected), *variance.shape))]
        ),
        event_score=np.concatenate([bank.event_score[keep], event_score[selected]]),
        month=np.concatenate([bank.month[keep], months]),
        zone=np.concatenate([bank.zone[keep], zones]),
        site=np.concatenate([bank.site[keep], matured["site"][selected].astype(np.int16)]),
        origin=np.concatenate([bank.origin[keep], matured["origin"][selected]]),
        count=np.concatenate([bank.count[keep], np.ones(len(selected), dtype=np.int64)]),
        kind=np.concatenate([bank.kind[keep], np.asarray(["causal_online"] * len(selected))]),
    )
    return updated, matured["origin"][selected].astype(int).tolist()


def run_causal_online(config: dict[str, Any], seed: int, force: bool = False) -> Path:
    ensure_q1_directories()
    lock = verify_protocol_lock()
    output = Q1_ROOT / "artifacts" / f"predictions_causal_online_seed{seed}.npz"
    if output.exists() and not force:
        return output
    prediction = load_q1_predictions("confirmatory", seed)
    store = load_q1_features()
    bank = load_q1_memory(seed, "full")
    calibration = load_q1_calibration(seed, "full")
    top_k = int(config["memory"]["top_k"])
    max_horizon = max(int(value) for value in config["model"]["horizons"])
    replacements = int(config["memory"]["causal_replacements_per_update"])
    unique_origins = np.unique(prediction["origin"])
    output_mean = np.empty_like(prediction["mean"])
    inserted_origins: list[int] = []
    update_rows = []
    origin_to_indices = {
        int(origin): np.flatnonzero(prediction["origin"] == origin) for origin in unique_origins
    }
    for step, origin in enumerate(unique_origins):
        query_indices = origin_to_indices[int(origin)]
        output_mean[query_indices] = _forecast(
            prediction["latent"][query_indices], prediction["mean"][query_indices], bank, calibration, top_k
        )
        matured_origin = int(origin) - max_horizon
        matured_indices = origin_to_indices.get(matured_origin)
        if matured_indices is not None:
            matured = {key: value[matured_indices] for key, value in prediction.items()}
            bank, inserted = _safe_update(
                bank, matured, replacements, int(origin), max_horizon, store.latitude, store.time
            )
            inserted_origins.extend(inserted)
            update_rows.append(
                {
                    "forecast_origin": int(origin),
                    "matured_origin": matured_origin,
                    "inserted": len(inserted),
                    "maximum_inserted_origin": max(inserted),
                    "causal_margin_steps": int(origin) - max(inserted) - max_horizon,
                }
            )
    np.savez_compressed(
        output,
        mean=output_mean,
        target_z=prediction["target_z"],
        origin=prediction["origin"],
        site=prediction["site"],
    )
    update_table = pd.DataFrame(update_rows)
    update_table.to_csv(Q1_ROOT / "results" / "tables" / f"causal_update_audit_seed{seed}.csv", index=False)
    q1_json(
        f"artifacts/causal_online_seed{seed}_manifest.json",
        {
            "seed": seed,
            "protocol_hash": lock["combined_sha256"],
            "forecast_origins": len(unique_origins),
            "updates": len(update_rows),
            "insertions": len(inserted_origins),
            "capacity": bank.capacity,
            "minimum_causal_margin_steps": int(update_table.causal_margin_steps.min()),
            "leakage_detected": bool((update_table.causal_margin_steps < 0).any()),
        },
    )
    return output


def run_all_causal_online(config: dict[str, Any], force: bool = False) -> None:
    for seed in [int(value) for value in config["model"]["seeds"]]:
        print(f"q1 causal online seed={seed}", flush=True)
        run_causal_online(config, seed, force=force)

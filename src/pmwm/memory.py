from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import MiniBatchKMeans

from .common import ROOT, atomic_json, device_name, ensure_directories, set_seed
from .features import load_features
from .model import load_predictions


@dataclass
class MemoryBank:
    keys: np.ndarray
    outcomes: np.ndarray
    residuals: np.ndarray
    residual_variance: np.ndarray
    event_score: np.ndarray
    month: np.ndarray
    zone: np.ndarray
    site: np.ndarray
    origin: np.ndarray
    count: np.ndarray
    kind: np.ndarray

    @property
    def capacity(self) -> int:
        return int(len(self.keys))


def normalize_keys(keys: np.ndarray) -> np.ndarray:
    keys = np.asarray(keys, dtype=np.float32)
    norm = np.linalg.norm(keys, axis=1, keepdims=True)
    return keys / np.maximum(norm, 1e-8)


def _modal_label(assignments: np.ndarray, labels: np.ndarray, n_clusters: int, default: int = -1) -> np.ndarray:
    result = np.full(n_clusters, default, dtype=np.int16)
    for cluster in range(n_clusters):
        selected = labels[assignments == cluster]
        if len(selected):
            result[cluster] = int(np.bincount(selected.astype(np.int64)).argmax())
    return result


def _select_diverse_events(
    scores: np.ndarray, origins: np.ndarray, sites: np.ndarray, times: np.ndarray, n_events: int
) -> np.ndarray:
    dates = pd.DatetimeIndex(times[origins])
    group = sites.astype(np.int64) * 100000 + dates.year.to_numpy() * 100 + dates.month.to_numpy()
    order = np.argsort(scores)[::-1]
    selected: list[int] = []
    used: set[int] = set()
    for index in order:
        code = int(group[index])
        if code in used:
            continue
        selected.append(int(index))
        used.add(code)
        if len(selected) >= n_events:
            break
    if len(selected) < n_events:
        selected_set = set(selected)
        selected.extend(int(index) for index in order if int(index) not in selected_set)
    return np.asarray(selected[:n_events], dtype=np.int64)


def _consolidate_partition(
    keys: np.ndarray,
    outcomes: np.ndarray,
    residuals: np.ndarray,
    scores: np.ndarray,
    months: np.ndarray,
    zones: np.ndarray,
    sites: np.ndarray,
    origins: np.ndarray,
    indices: np.ndarray,
    n_clusters: int,
    sample_size: int,
    max_iter: int,
    seed: int,
) -> tuple[dict[str, np.ndarray], float]:
    """Learn a bounded vector-quantized summary of one memory partition."""
    rng = np.random.default_rng(seed)
    fit_indices = (
        rng.choice(indices, size=sample_size, replace=False) if len(indices) > sample_size else indices
    )
    started = time.perf_counter()
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=4096,
        max_iter=max_iter,
        n_init=1,
        init="k-means++",
        random_state=seed,
        reassignment_ratio=0.005,
    )
    kmeans.fit(keys[fit_indices])
    assignments = kmeans.predict(keys[indices])
    elapsed = time.perf_counter() - started
    counts = np.bincount(assignments, minlength=n_clusters).astype(np.int64)
    safe_count = np.maximum(counts, 1).astype(np.float64)
    key_sum = np.zeros((n_clusters, keys.shape[1]), dtype=np.float64)
    outcome_sum = np.zeros((n_clusters, *outcomes.shape[1:]), dtype=np.float64)
    residual_sum = np.zeros_like(outcome_sum)
    residual_square_sum = np.zeros_like(outcome_sum)
    event_sum = np.zeros(n_clusters, dtype=np.float64)
    origin_sum = np.zeros(n_clusters, dtype=np.float64)
    np.add.at(key_sum, assignments, keys[indices])
    np.add.at(outcome_sum, assignments, outcomes[indices])
    np.add.at(residual_sum, assignments, residuals[indices])
    np.add.at(residual_square_sum, assignments, np.square(residuals[indices]))
    np.add.at(event_sum, assignments, scores[indices])
    np.add.at(origin_sum, assignments, origins[indices])
    mean_residual = (residual_sum / safe_count[:, None, None]).astype(np.float32)
    partition = {
        "keys": normalize_keys((key_sum / safe_count[:, None]).astype(np.float32)),
        "outcomes": (outcome_sum / safe_count[:, None, None]).astype(np.float32),
        "residuals": mean_residual,
        "residual_variance": np.maximum(
            residual_square_sum / safe_count[:, None, None] - np.square(mean_residual), 1e-4
        ).astype(np.float32),
        "event_score": (event_sum / safe_count).astype(np.float32),
        "month": _modal_label(assignments, months[indices], n_clusters, default=1),
        "zone": _modal_label(assignments, zones[indices], n_clusters, default=0),
        "site": _modal_label(assignments, sites[indices], n_clusters, default=0),
        "origin": np.rint(origin_sum / safe_count).astype(np.int64),
        "count": counts,
    }
    return partition, elapsed


def build_memory(config: dict[str, Any], force: bool = False) -> Path:
    ensure_directories()
    output = ROOT / "artifacts" / "memory_bank.npz"
    random_output = ROOT / "artifacts" / "memory_bank_reservoir.npz"
    calibration_output = ROOT / "artifacts" / "memory_calibration.npz"
    if output.exists() and random_output.exists() and calibration_output.exists() and not force:
        return output

    seed = int(config["project"]["seed"])
    set_seed(seed)
    rng = np.random.default_rng(seed)
    memory_cfg = config["memory"]
    train = load_predictions("train")
    validation = load_predictions("validation")
    store = load_features()
    keys = normalize_keys(train["latent"])
    outcomes = train["target_z"].astype(np.float32)
    residuals = (outcomes - train["mean"]).astype(np.float32)
    scores = np.max(np.abs(outcomes), axis=(1, 2)).astype(np.float32)
    dates = pd.DatetimeIndex(store.time[train["origin"]])
    months = dates.month.to_numpy(dtype=np.int16)
    zone_names = sorted(set(store.zones.tolist()))
    zone_to_id = {name: index for index, name in enumerate(zone_names)}
    zones = np.asarray([zone_to_id[store.zones[site]] for site in train["site"]], dtype=np.int16)

    event_slots = int(memory_cfg["event_slots"])
    prototype_slots = int(memory_cfg["prototype_slots"])
    # The rarest 10% are consolidated separately. This preserves event coverage
    # without allowing single noisy extremes to dominate nearest-neighbor votes.
    event_threshold = float(np.quantile(scores, 0.90))
    event_indices = np.flatnonzero(scores >= event_threshold)
    regular_indices = np.flatnonzero(scores < event_threshold)
    regular, regular_seconds = _consolidate_partition(
        keys,
        outcomes,
        residuals,
        scores,
        months,
        zones,
        train["site"],
        train["origin"],
        regular_indices,
        prototype_slots,
        min(int(memory_cfg["kmeans_sample"]), len(regular_indices)),
        int(memory_cfg["kmeans_iterations"]),
        seed,
    )
    events, event_seconds = _consolidate_partition(
        keys,
        outcomes,
        residuals,
        scores,
        months,
        zones,
        train["site"],
        train["origin"],
        event_indices,
        event_slots,
        min(30000, len(event_indices)),
        int(memory_cfg["kmeans_iterations"]),
        seed + 1,
    )
    kmeans_seconds = regular_seconds + event_seconds
    global_variance = np.var(residuals, axis=0, ddof=1).astype(np.float32)
    bank = MemoryBank(
        keys=np.concatenate([regular["keys"], events["keys"]], axis=0),
        outcomes=np.concatenate([regular["outcomes"], events["outcomes"]], axis=0),
        residuals=np.concatenate([regular["residuals"], events["residuals"]], axis=0),
        residual_variance=np.concatenate([regular["residual_variance"], events["residual_variance"]], axis=0),
        event_score=np.concatenate([regular["event_score"], events["event_score"]], axis=0),
        month=np.concatenate([regular["month"], events["month"]], axis=0),
        zone=np.concatenate([regular["zone"], events["zone"]], axis=0),
        site=np.concatenate([regular["site"], events["site"]], axis=0),
        origin=np.concatenate([regular["origin"], events["origin"]], axis=0),
        count=np.concatenate([regular["count"], events["count"]], axis=0),
        kind=np.asarray(["prototype"] * prototype_slots + ["event_prototype"] * event_slots),
    )
    save_memory(bank, output)

    reservoir_indices = rng.choice(len(keys), size=bank.capacity, replace=False)
    reservoir_bank = MemoryBank(
        keys=keys[reservoir_indices],
        outcomes=outcomes[reservoir_indices],
        residuals=residuals[reservoir_indices],
        residual_variance=np.broadcast_to(global_variance[None], (bank.capacity, *global_variance.shape)).copy(),
        event_score=scores[reservoir_indices],
        month=months[reservoir_indices],
        zone=zones[reservoir_indices],
        site=train["site"][reservoir_indices].astype(np.int16),
        origin=train["origin"][reservoir_indices],
        count=np.ones(bank.capacity, dtype=np.int64),
        kind=np.asarray(["reservoir"] * bank.capacity),
    )
    save_memory(reservoir_bank, random_output)
    calibration = calibrate_memory(bank, validation, config)
    np.savez_compressed(calibration_output, **calibration)

    source_points = len(train["origin"]) * int(config["model"]["context_steps"]) * train["target_z"].shape[-1]
    memory_bytes = sum(
        getattr(bank, field).nbytes
        for field in [
            "keys", "outcomes", "residuals", "residual_variance", "event_score", "month", "zone", "site", "origin", "count", "kind"
        ]
    )
    atomic_json(
        ROOT / "artifacts" / "memory_manifest.json",
        {
            "capacity": bank.capacity,
            "prototype_slots": prototype_slots,
            "event_slots": event_slots,
            "event_pool_size": len(event_indices),
            "event_threshold_z": event_threshold,
            "training_memories_before_consolidation": len(keys),
            "compression_ratio": len(keys) / bank.capacity,
            "effective_raw_points_per_memory": source_points / bank.capacity,
            "memory_bytes": memory_bytes,
            "kmeans_fit_seconds": kmeans_seconds,
            "top_k": int(memory_cfg["top_k"]),
            "temperature": float(calibration["temperature"]),
            "gate_mean": float(calibration["gate"].mean()),
            "zone_vocabulary": zone_names,
        },
    )
    return output


def save_memory(bank: MemoryBank, path: Path) -> None:
    np.savez_compressed(
        path,
        keys=bank.keys,
        outcomes=bank.outcomes,
        residuals=bank.residuals,
        residual_variance=bank.residual_variance,
        event_score=bank.event_score,
        month=bank.month,
        zone=bank.zone,
        site=bank.site,
        origin=bank.origin,
        count=bank.count,
        kind=bank.kind,
    )


def load_memory(path: Path | None = None) -> MemoryBank:
    path = path or ROOT / "artifacts" / "memory_bank.npz"
    with np.load(path, allow_pickle=False) as archive:
        return MemoryBank(**{name: archive[name] for name in archive.files})


def retrieve_neighbors(
    query_keys: np.ndarray,
    bank: MemoryBank,
    top_k: int,
    batch_size: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    device = torch.device(device_name())
    memory = torch.from_numpy(normalize_keys(bank.keys)).to(device)
    all_similarity: list[np.ndarray] = []
    all_indices: list[np.ndarray] = []
    for start in range(0, len(query_keys), batch_size):
        query = torch.from_numpy(normalize_keys(query_keys[start : start + batch_size])).to(device)
        similarity = query @ memory.T
        values, indices = torch.topk(similarity, k=min(top_k, bank.capacity), dim=1)
        all_similarity.append(values.float().cpu().numpy())
        all_indices.append(indices.cpu().numpy())
    return np.concatenate(all_similarity), np.concatenate(all_indices)


def aggregate_retrieval(
    similarity: np.ndarray,
    indices: np.ndarray,
    bank: MemoryBank,
    temperature: float,
) -> dict[str, np.ndarray]:
    logits = (similarity - similarity.max(axis=1, keepdims=True)) / max(temperature, 1e-4)
    weights = np.exp(logits)
    weights /= weights.sum(axis=1, keepdims=True)
    expanded = weights[..., None, None]
    residual_values = bank.residuals[indices]
    outcome_values = bank.outcomes[indices]
    residual_mean = np.sum(expanded * residual_values, axis=1)
    outcome_mean = np.sum(expanded * outcome_values, axis=1)
    within = bank.residual_variance[indices]
    between = np.square(residual_values - residual_mean[:, None])
    residual_variance = np.sum(expanded * (within + between), axis=1)
    return {
        "weights": weights.astype(np.float32),
        "residual": residual_mean.astype(np.float32),
        "outcome": outcome_mean.astype(np.float32),
        "variance": np.maximum(residual_variance, 1e-5).astype(np.float32),
    }


def calibrate_memory(bank: MemoryBank, validation: dict[str, np.ndarray], config: dict[str, Any]) -> dict[str, np.ndarray]:
    top_k = int(config["memory"]["top_k"])
    similarity, indices = retrieve_neighbors(validation["latent"], bank, top_k)
    target = validation["target_z"]
    base = validation["mean"]
    delta = target - base
    best: dict[str, Any] | None = None
    for temperature in [0.03, 0.05, 0.08, 0.12, 0.20, 0.35]:
        retrieval = aggregate_retrieval(similarity, indices, bank, temperature)
        residual = retrieval["residual"]
        numerator = np.sum(delta * residual, axis=0)
        denominator = np.sum(np.square(residual), axis=0) + 1e-6
        gate = np.clip(numerator / denominator, 0.0, 1.5).astype(np.float32)
        prediction = base + gate[None] * residual
        mse = float(np.mean(np.square(target - prediction)))
        if best is None or mse < best["mse"]:
            best = {"temperature": temperature, "gate": gate, "mse": mse, "retrieval": retrieval}
    assert best is not None
    base_variance = np.exp(validation["logvar"])
    fused_variance = base_variance + np.square(best["gate"])[None] * best["retrieval"]["variance"]
    error_ratio = np.abs(target - (base + best["gate"][None] * best["retrieval"]["residual"])) / np.sqrt(
        np.maximum(fused_variance, 1e-6)
    )
    quantile = np.quantile(error_ratio, 0.90, axis=0)
    variance_scale = np.square(np.clip(quantile / 1.6448536269514722, 0.5, 4.0)).astype(np.float32)
    return {
        "temperature": np.asarray(best["temperature"], dtype=np.float32),
        "gate": best["gate"],
        "variance_scale": variance_scale,
        "validation_mse_z": np.asarray(best["mse"], dtype=np.float32),
    }


def load_calibration() -> dict[str, np.ndarray]:
    with np.load(ROOT / "artifacts" / "memory_calibration.npz", allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def memory_forecast(
    prediction: dict[str, np.ndarray],
    bank: MemoryBank,
    calibration: dict[str, np.ndarray],
    top_k: int,
) -> dict[str, np.ndarray]:
    similarity, indices = retrieve_neighbors(prediction["latent"], bank, top_k)
    retrieval = aggregate_retrieval(similarity, indices, bank, float(calibration["temperature"]))
    gate = calibration["gate"]
    mean = prediction["mean"] + gate[None] * retrieval["residual"]
    variance = (
        np.exp(prediction["logvar"]) + np.square(gate)[None] * retrieval["variance"]
    ) * calibration["variance_scale"][None]
    return {
        "mean": mean.astype(np.float32),
        "variance": np.maximum(variance, 1e-6).astype(np.float32),
        "analog_mean": retrieval["outcome"],
        "similarity": similarity.astype(np.float32),
        "indices": indices.astype(np.int32),
        "weights": retrieval["weights"],
        "retrieval_variance": retrieval["variance"],
    }


def adapt_memory(
    bank: MemoryBank,
    prediction: dict[str, np.ndarray],
    n_new: int,
    seed: int,
) -> MemoryBank:
    """Event-aware bounded update used for prequential continual-learning analysis."""
    rng = np.random.default_rng(seed)
    residual = prediction["target_z"] - prediction["mean"]
    score = np.max(np.abs(prediction["target_z"]), axis=(1, 2))
    event_count = n_new // 2
    event_indices = np.argsort(score)[-event_count:]
    remaining = np.setdiff1d(np.arange(len(score)), event_indices, assume_unique=False)
    random_indices = rng.choice(remaining, size=n_new - event_count, replace=False)
    selected = np.concatenate([event_indices, random_indices])
    keep_priority = np.log1p(bank.count) + 0.35 * bank.event_score
    keep = np.argsort(keep_priority)[-(bank.capacity - n_new) :]
    global_variance = np.var(residual, axis=0, ddof=1).astype(np.float32)
    store = load_features()
    dates = pd.DatetimeIndex(store.time[prediction["origin"][selected]])
    zone_names = sorted(set(store.zones.tolist()))
    zone_to_id = {name: index for index, name in enumerate(zone_names)}
    new_zone = np.asarray([zone_to_id[store.zones[site]] for site in prediction["site"][selected]], dtype=np.int16)
    return MemoryBank(
        keys=np.concatenate([bank.keys[keep], normalize_keys(prediction["latent"][selected])]),
        outcomes=np.concatenate([bank.outcomes[keep], prediction["target_z"][selected]]),
        residuals=np.concatenate([bank.residuals[keep], residual[selected]]),
        residual_variance=np.concatenate(
            [bank.residual_variance[keep], np.broadcast_to(global_variance[None], (n_new, *global_variance.shape))]
        ),
        event_score=np.concatenate([bank.event_score[keep], score[selected]]),
        month=np.concatenate([bank.month[keep], dates.month.to_numpy(dtype=np.int16)]),
        zone=np.concatenate([bank.zone[keep], new_zone]),
        site=np.concatenate([bank.site[keep], prediction["site"][selected].astype(np.int16)]),
        origin=np.concatenate([bank.origin[keep], prediction["origin"][selected]]),
        count=np.concatenate([bank.count[keep], np.ones(n_new, dtype=np.int64)]),
        kind=np.concatenate([bank.kind[keep], np.asarray(["adaptive"] * n_new)]),
    )

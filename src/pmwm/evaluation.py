from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.special import ndtr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .common import ROOT, atomic_json, ensure_directories, set_seed
from .features import (
    PHYSICAL_TARGET_FEATURES,
    load_features,
    standardized_to_physical,
)
from .memory import (
    MemoryBank,
    adapt_memory,
    aggregate_retrieval,
    calibrate_memory,
    load_calibration,
    load_memory,
    memory_forecast,
    retrieve_neighbors,
)
from .model import SiteSequenceDataset, load_model, load_predictions, make_datasets, predict_dataset


def _multiscale_features(origins: np.ndarray, sites: np.ndarray, store: Any) -> np.ndarray:
    x = store.input_z
    cumulative = np.concatenate(
        [np.zeros((1, x.shape[1], x.shape[2]), dtype=np.float64), np.cumsum(x, axis=0, dtype=np.float64)], axis=0
    )
    cumulative_square = np.concatenate(
        [
            np.zeros((1, x.shape[1], x.shape[2]), dtype=np.float64),
            np.cumsum(np.square(x), axis=0, dtype=np.float64),
        ],
        axis=0,
    )

    def stats(width: int) -> tuple[np.ndarray, np.ndarray]:
        end = origins + 1
        start = end - width
        total = cumulative[end, sites] - cumulative[start, sites]
        square = cumulative_square[end, sites] - cumulative_square[start, sites]
        mean = total / width
        std = np.sqrt(np.maximum(square / width - np.square(mean), 1e-6))
        return mean.astype(np.float32), std.astype(np.float32)

    mean4, _ = stats(4)
    mean28, std28 = stats(28)
    mean56, std56 = stats(56)
    current = x[origins, sites]
    tendency = current - x[origins - 4, sites]
    return np.concatenate(
        [current, mean4, mean28, std28, mean56, std56, tendency, store.static[sites]], axis=1
    ).astype(np.float32)


def _fit_ridge(
    train: dict[str, np.ndarray], validation: dict[str, np.ndarray], store: Any
) -> tuple[Ridge, float]:
    x_train = _multiscale_features(train["origin"], train["site"], store)
    y_train = train["target_z"].reshape(len(x_train), -1)
    x_validation = _multiscale_features(validation["origin"], validation["site"], store)
    y_validation = validation["target_z"].reshape(len(x_validation), -1)
    best_model: Ridge | None = None
    best_alpha = 0.0
    best_mse = float("inf")
    for alpha in [0.1, 1.0, 10.0, 100.0]:
        model = Ridge(alpha=alpha).fit(x_train, y_train)
        mse = float(np.mean(np.square(model.predict(x_validation) - y_validation)))
        if mse < best_mse:
            best_model, best_alpha, best_mse = model, alpha, mse
    assert best_model is not None
    return best_model, best_alpha


def _ridge_predict(model: Ridge, prediction: dict[str, np.ndarray], store: Any) -> np.ndarray:
    features = _multiscale_features(prediction["origin"], prediction["site"], store)
    shape = prediction["target_z"].shape
    return model.predict(features).reshape(shape).astype(np.float32)


def _persistence_z(prediction: dict[str, np.ndarray], store: Any, horizons: list[int]) -> np.ndarray:
    current_model = (
        store.target_seasonal[prediction["origin"], prediction["site"]]
        + store.target_z[prediction["origin"], prediction["site"]] * store.target_scale[prediction["site"]]
    )
    output = []
    for horizon in horizons:
        seasonal = store.target_seasonal[prediction["origin"] + horizon, prediction["site"]]
        output.append((current_model - seasonal) / store.target_scale[prediction["site"]])
    return np.stack(output, axis=1).astype(np.float32)


def gaussian_crps(target: np.ndarray, mean: np.ndarray, variance: np.ndarray) -> np.ndarray:
    sigma = np.sqrt(np.maximum(variance, 1e-8))
    z = (target - mean) / sigma
    phi = np.exp(-0.5 * np.square(z)) / math.sqrt(2 * math.pi)
    return sigma * (z * (2 * ndtr(z) - 1) + 2 * phi - 1 / math.sqrt(math.pi))


def _corr(actual: np.ndarray, predicted: np.ndarray) -> float:
    actual = actual.ravel().astype(np.float64)
    predicted = predicted.ravel().astype(np.float64)
    actual -= actual.mean()
    predicted -= predicted.mean()
    denominator = np.sqrt(np.sum(np.square(actual)) * np.sum(np.square(predicted)))
    return float(np.sum(actual * predicted) / denominator) if denominator > 0 else float("nan")


def metric_tables(
    predictions_z: dict[str, np.ndarray],
    variances: dict[str, np.ndarray],
    metadata: dict[str, np.ndarray],
    store: Any,
    horizons: list[int],
    horizon_labels: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    actual_z = metadata["target_z"]
    actual_physical = metadata["target_physical"]
    physical_predictions = {
        name: standardized_to_physical(values, metadata["origin"], metadata["site"], horizons, store)
        for name, values in predictions_z.items()
    }
    rows: list[dict[str, Any]] = []
    leaderboard: list[dict[str, Any]] = []
    climatology_mse = np.mean(np.square(actual_z), axis=0)
    persistence_mse = np.mean(np.square(actual_z - predictions_z["Persistence"]), axis=0)
    for model_name, predicted_z in predictions_z.items():
        for h_idx, horizon_label in enumerate(horizon_labels):
            for feature_idx, feature in enumerate(PHYSICAL_TARGET_FEATURES):
                az = actual_z[:, h_idx, feature_idx]
                pz = predicted_z[:, h_idx, feature_idx]
                ap = actual_physical[:, h_idx, feature_idx]
                pp = physical_predictions[model_name][:, h_idx, feature_idx]
                mse_z = float(np.mean(np.square(az - pz)))
                mse_p = float(np.mean(np.square(ap - pp)))
                variance = variances.get(model_name)
                row = {
                    "model": model_name,
                    "horizon": horizon_label,
                    "target": feature,
                    "rmse_z": math.sqrt(mse_z),
                    "mae_z": float(np.mean(np.abs(az - pz))),
                    "bias_z": float(np.mean(pz - az)),
                    "acc": _corr(az, pz),
                    "skill_vs_climatology": 1 - mse_z / max(float(climatology_mse[h_idx, feature_idx]), 1e-9),
                    "skill_vs_persistence": 1 - mse_z / max(float(persistence_mse[h_idx, feature_idx]), 1e-9),
                    "rmse_physical": math.sqrt(mse_p),
                    "mae_physical": float(np.mean(np.abs(ap - pp))),
                    "bias_physical": float(np.mean(pp - ap)),
                    "r2_physical": 1 - mse_p / max(float(np.var(ap)), 1e-9),
                }
                if variance is not None:
                    var = variance[:, h_idx, feature_idx]
                    row["crps_z"] = float(np.mean(gaussian_crps(az, pz, var)))
                    row["nll_z"] = float(
                        np.mean(0.5 * (np.log(2 * np.pi * var) + np.square(az - pz) / var))
                    )
                    sigma = np.sqrt(var)
                    row["coverage_90"] = float(np.mean(np.abs(az - pz) <= 1.6448536269514722 * sigma))
                    row["interval_width_90_z"] = float(np.mean(2 * 1.6448536269514722 * sigma))
                rows.append(row)
        error = actual_z - predicted_z
        flat_mse = float(np.mean(np.square(error)))
        leaderboard.append(
            {
                "model": model_name,
                "rmse_z": math.sqrt(flat_mse),
                "mae_z": float(np.mean(np.abs(error))),
                "acc": _corr(actual_z, predicted_z),
                "skill_vs_climatology": 1 - flat_mse / float(np.mean(np.square(actual_z))),
                "skill_vs_persistence": 1 - flat_mse / float(np.mean(np.square(actual_z - predictions_z["Persistence"]))),
                "crps_z": float(np.mean(gaussian_crps(actual_z, predicted_z, variances[model_name])))
                if model_name in variances
                else np.nan,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(leaderboard), physical_predictions


def _probabilistic_table(
    actual: np.ndarray, mean: np.ndarray, variance: np.ndarray, levels: list[float]
) -> pd.DataFrame:
    rows = []
    sigma = np.sqrt(variance)
    from scipy.stats import norm

    for level in levels:
        z = float(norm.ppf((1 + level) / 2))
        covered = np.abs(actual - mean) <= z * sigma
        rows.append(
            {
                "nominal_coverage": level,
                "empirical_coverage": float(covered.mean()),
                "calibration_error": float(covered.mean() - level),
                "mean_width_z": float(np.mean(2 * z * sigma)),
            }
        )
    return pd.DataFrame(rows)


def _event_metrics(
    validation: dict[str, np.ndarray],
    validation_mean: np.ndarray,
    validation_variance: np.ndarray,
    test: dict[str, np.ndarray],
    test_mean: np.ndarray,
    test_variance: np.ndarray,
    threshold_z: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    def probabilities(mean: np.ndarray, variance: np.ndarray) -> np.ndarray:
        sigma = np.sqrt(np.maximum(variance, 1e-8))
        return 1 - (ndtr((threshold_z - mean) / sigma) - ndtr((-threshold_z - mean) / sigma))

    val_label = (np.abs(validation["target_z"]) > threshold_z).ravel().astype(int)
    val_probability = probabilities(validation_mean, validation_variance).ravel()
    thresholds = np.linspace(0.01, 0.99, 99)
    f1_values = [f1_score(val_label, val_probability >= threshold, zero_division=0) for threshold in thresholds]
    chosen = float(thresholds[int(np.argmax(f1_values))])
    label = (np.abs(test["target_z"]) > threshold_z).ravel().astype(int)
    probability = probabilities(test_mean, test_variance).ravel()
    metrics = pd.DataFrame(
        [
            {
                "auroc": roc_auc_score(label, probability),
                "auprc": average_precision_score(label, probability),
                "brier": brier_score_loss(label, probability),
                "f1": f1_score(label, probability >= chosen, zero_division=0),
                "decision_threshold": chosen,
                "event_prevalence": float(label.mean()),
            }
        ]
    )
    bins = np.linspace(0, 1, 11)
    bin_index = np.clip(np.digitize(probability, bins) - 1, 0, 9)
    reliability = []
    for index in range(10):
        selected = bin_index == index
        if selected.any():
            reliability.append(
                {
                    "bin": index,
                    "predicted_probability": float(probability[selected].mean()),
                    "observed_frequency": float(label[selected].mean()),
                    "count": int(selected.sum()),
                }
            )
    return metrics, pd.DataFrame(reliability)


def _retrieval_quality(
    forecast: dict[str, np.ndarray], prediction: dict[str, np.ndarray], bank: MemoryBank, store: Any
) -> pd.DataFrame:
    query_month = pd.DatetimeIndex(store.time[prediction["origin"]]).month.to_numpy()
    query_zone_names = store.zones[prediction["site"]]
    zone_vocabulary = sorted(set(store.zones.tolist()))
    zone_to_id = {name: index for index, name in enumerate(zone_vocabulary)}
    query_zone = np.asarray([zone_to_id[name] for name in query_zone_names])
    indices = forecast["indices"]
    month_distance = np.abs(bank.month[indices] - query_month[:, None])
    month_relevant = np.minimum(month_distance, 12 - month_distance) <= 1
    zone_relevant = bank.zone[indices] == query_zone[:, None]
    relevance = month_relevant & zone_relevant
    discounts = 1 / np.log2(np.arange(indices.shape[1]) + 2)
    dcg = np.sum(relevance * discounts[None], axis=1)
    ideal_count = np.minimum(relevance.sum(axis=1), indices.shape[1])
    idcg = np.asarray([discounts[:count].sum() if count else 1.0 for count in ideal_count])
    query_outcome = prediction["target_z"].reshape(len(indices), -1)
    memory_outcome = bank.outcomes[indices[:, 0]].reshape(len(indices), -1)
    outcome_cosine = np.sum(query_outcome * memory_outcome, axis=1) / np.maximum(
        np.linalg.norm(query_outcome, axis=1) * np.linalg.norm(memory_outcome, axis=1), 1e-8
    )
    return pd.DataFrame(
        [
            {
                "precision_at_k": float(relevance.mean()),
                "hit_rate_at_k": float(relevance.any(axis=1).mean()),
                "ndcg_at_k": float(np.mean(dcg / idcg)),
                "same_zone_at_k": float(zone_relevant.mean()),
                "season_match_at_k": float(month_relevant.mean()),
                "top1_outcome_cosine": float(np.mean(outcome_cosine)),
                "mean_top1_similarity": float(forecast["similarity"][:, 0].mean()),
                "k": int(indices.shape[1]),
            }
        ]
    )


def _classification_probes(
    train: dict[str, np.ndarray], test_seen: dict[str, np.ndarray], test_unseen: dict[str, np.ndarray], store: Any
) -> pd.DataFrame:
    zone_names = sorted(set(store.zones.tolist()))
    zone_to_id = {name: index for index, name in enumerate(zone_names)}

    def labels(prediction: dict[str, np.ndarray], kind: str) -> np.ndarray:
        if kind == "zone":
            return np.asarray([zone_to_id[store.zones[site]] for site in prediction["site"]])
        month = pd.DatetimeIndex(store.time[prediction["origin"]]).month.to_numpy()
        return ((month % 12) // 3).astype(int)

    rng = np.random.default_rng(3407)
    sample = rng.choice(len(train["latent"]), size=min(60000, len(train["latent"])), replace=False)
    rows = []
    for kind in ["zone", "season"]:
        classifier = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=3407),
        )
        classifier.fit(train["latent"][sample], labels(train, kind)[sample])
        for domain_name, prediction in [("seen", test_seen), ("held-out", test_unseen)]:
            actual = labels(prediction, kind)
            predicted = classifier.predict(prediction["latent"])
            rows.append(
                {
                    "probe": kind,
                    "domain": domain_name,
                    "balanced_accuracy": balanced_accuracy_score(actual, predicted),
                    "macro_f1": f1_score(actual, predicted, average="macro", zero_division=0),
                }
            )
    return pd.DataFrame(rows)


def _imputation_study(config: dict[str, Any], test_dataset: SiteSequenceDataset, store: Any) -> pd.DataFrame:
    rng = np.random.default_rng(3407)
    subset_indices = rng.choice(len(test_dataset), size=min(16000, len(test_dataset)), replace=False)
    subset = SiteSequenceDataset(
        store,
        test_dataset.origins[subset_indices],
        test_dataset.sites[subset_indices],
        test_dataset.context_steps,
        test_dataset.horizons,
    )
    model = load_model(config)
    rows = []
    actual = store.input_z[subset.origins, subset.sites]
    previous = store.input_z[subset.origins - 1, subset.sites]
    for rate in [0.1, 0.3, 0.5, 0.7]:
        result = predict_dataset(model, subset, int(config["model"]["batch_size"]), missing_rate=rate, seed=3407)
        missing = result["final_mask"]
        for name, estimate in [
            ("Seasonal mean", np.zeros_like(actual)),
            ("Last observation", previous),
            ("PMWM imputation head", result["imputation"]),
        ]:
            error = (estimate - actual)[missing]
            rows.append(
                {
                    "missing_fraction": rate,
                    "method": name,
                    "rmse_z": float(np.sqrt(np.mean(np.square(error)))),
                    "mae_z": float(np.mean(np.abs(error))),
                    "masked_values": int(missing.sum()),
                }
            )
    return pd.DataFrame(rows)


def _bootstrap_effect(
    prediction: dict[str, np.ndarray], full_mean: np.ndarray, replicates: int
) -> pd.DataFrame:
    sample_full = np.mean(np.square(prediction["target_z"] - full_mean), axis=(1, 2))
    sample_base = np.mean(np.square(prediction["target_z"] - prediction["mean"]), axis=(1, 2))
    block = prediction["origin"] // (4 * 7)
    frame = pd.DataFrame({"block": block, "difference": sample_base - sample_full})
    block_values = frame.groupby("block")["difference"].mean().to_numpy()
    rng = np.random.default_rng(3407)
    draws = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        draws[index] = rng.choice(block_values, size=len(block_values), replace=True).mean()
    return pd.DataFrame(
        [
            {
                "mse_reduction_z": float(block_values.mean()),
                "ci_lower": float(np.quantile(draws, 0.025)),
                "ci_upper": float(np.quantile(draws, 0.975)),
                "bootstrap_probability_improvement": float((draws > 0).mean()),
                "n_week_blocks": len(block_values),
                "replicates": replicates,
            }
        ]
    )


def _continual_study(
    prediction: dict[str, np.ndarray], bank: MemoryBank, calibration: dict[str, np.ndarray], config: dict[str, Any], store: Any
) -> tuple[pd.DataFrame, pd.DataFrame]:
    years = pd.DatetimeIndex(store.time[prediction["origin"]]).year.to_numpy()
    eras = config["evaluation"]["continual_eras"]
    rng = np.random.default_rng(3407)
    task_predictions: list[dict[str, np.ndarray]] = []
    for start, end in eras:
        indices = np.flatnonzero((years >= start) & (years <= end))
        if len(indices) > 10000:
            indices = np.sort(rng.choice(indices, size=10000, replace=False))
        task_predictions.append({name: values[indices] for name, values in prediction.items()})
    current = bank
    matrix_rows = []
    for state in range(len(eras) + 1):
        for task_idx, (task, (start, end)) in enumerate(zip(task_predictions, eras)):
            forecast = memory_forecast(task, current, calibration, int(config["memory"]["top_k"]))
            base_mse = float(np.mean(np.square(task["target_z"] - task["mean"])))
            memory_mse = float(np.mean(np.square(task["target_z"] - forecast["mean"])))
            matrix_rows.append(
                {
                    "memory_state": state,
                    "task": task_idx,
                    "task_label": f"{start}–{end}",
                    "memory_skill_vs_backbone": 1 - memory_mse / base_mse,
                    "rmse_z": math.sqrt(memory_mse),
                }
            )
        if state < len(eras):
            # Conservative bounded updates replace <1% of the bank per era;
            # validation experiments showed aggressive replacement causes avoidable forgetting.
            current = adapt_memory(current, task_predictions[state], n_new=8, seed=3407 + state)
    matrix = pd.DataFrame(matrix_rows)
    learned = []
    final_state = len(eras)
    for task_idx in range(len(eras)):
        at_learning = matrix[(matrix.memory_state == task_idx + 1) & (matrix.task == task_idx)][
            "memory_skill_vs_backbone"
        ].iloc[0]
        final = matrix[(matrix.memory_state == final_state) & (matrix.task == task_idx)][
            "memory_skill_vs_backbone"
        ].iloc[0]
        learned.append(final - at_learning)
    summary = pd.DataFrame(
        [
            {
                "backward_transfer": float(np.mean(learned[:-1])) if len(learned) > 1 else float(learned[0]),
                "worst_task_forgetting": float(-min(learned)),
                "final_average_memory_skill": float(
                    matrix[matrix.memory_state == final_state]["memory_skill_vs_backbone"].mean()
                ),
            }
        ]
    )
    return matrix, summary


def _zero_shot_study(
    prediction: dict[str, np.ndarray], bank: MemoryBank, calibration: dict[str, np.ndarray], config: dict[str, Any], store: Any
) -> pd.DataFrame:
    years = pd.DatetimeIndex(store.time[prediction["origin"]]).year.to_numpy()
    dates = pd.DatetimeIndex(store.time[prediction["origin"]])
    adaptation_indices = np.flatnonzero(years == config["splits"]["test_start"])
    adaptation_train = np.flatnonzero(
        (years == config["splits"]["test_start"]) & (dates.month.to_numpy() <= 6)
    )
    adaptation_validation = np.flatnonzero(
        (years == config["splits"]["test_start"]) & (dates.month.to_numpy() > 6)
    )
    evaluation_indices = np.flatnonzero(years > config["splits"]["test_start"])
    evaluation = {name: values[evaluation_indices] for name, values in prediction.items()}
    global_forecast = memory_forecast(prediction, bank, calibration, int(config["memory"]["top_k"]))

    def domain_bank(indices: np.ndarray) -> MemoryBank:
        residual = prediction["target_z"][indices] - global_forecast["mean"][indices]
        variance = np.var(residual, axis=0, ddof=1).astype(np.float32)
        n_items = len(indices)
        return MemoryBank(
            keys=prediction["latent"][indices],
            outcomes=prediction["target_z"][indices],
            residuals=residual,
            residual_variance=np.broadcast_to(variance[None], (n_items, *variance.shape)).copy(),
            event_score=np.max(np.abs(prediction["target_z"][indices]), axis=(1, 2)),
            month=dates.month.to_numpy(dtype=np.int16)[indices],
            zone=np.zeros(n_items, dtype=np.int16),
            site=prediction["site"][indices].astype(np.int16),
            origin=prediction["origin"][indices],
            count=np.ones(n_items, dtype=np.int64),
            kind=np.asarray(["domain"] * n_items),
        )

    # The first half of the adaptation year creates domain memory; its second
    # half selects retrieval softness and a residual gate. The reported period
    # starts in the following year, so adaptation tuning never sees final-test targets.
    half_bank = domain_bank(adaptation_train)
    validation_delta = (
        prediction["target_z"][adaptation_validation]
        - global_forecast["mean"][adaptation_validation]
    )
    best: dict[str, Any] | None = None
    similarity, indices = retrieve_neighbors(
        prediction["latent"][adaptation_validation], half_bank, int(config["memory"]["top_k"])
    )
    for temperature in [0.01, 0.03, 0.05, 0.10, 0.20]:
        local = aggregate_retrieval(similarity, indices, half_bank, temperature)["residual"]
        gate = np.clip(
            np.sum(validation_delta * local, axis=0) / (np.sum(np.square(local), axis=0) + 1e-6),
            0.0,
            1.5,
        )
        estimate = global_forecast["mean"][adaptation_validation] + gate[None] * local
        mse = float(np.mean(np.square(prediction["target_z"][adaptation_validation] - estimate)))
        if best is None or mse < best["mse"]:
            best = {"temperature": temperature, "gate": gate.astype(np.float32), "mse": mse}
    assert best is not None

    full_domain_bank = domain_bank(adaptation_indices)
    eval_similarity, eval_indices = retrieve_neighbors(
        prediction["latent"][evaluation_indices], full_domain_bank, int(config["memory"]["top_k"])
    )
    local_residual = aggregate_retrieval(
        eval_similarity, eval_indices, full_domain_bank, float(best["temperature"])
    )["residual"]
    zero_mean = global_forecast["mean"][evaluation_indices]
    few_mean = zero_mean + best["gate"][None] * local_residual
    rows = []
    for model_name, mean in [
        ("Backbone zero-shot", evaluation["mean"]),
        ("PMWM zero-shot", zero_mean),
        ("PMWM + 1-year memory", few_mean),
    ]:
        for site in np.unique(evaluation["site"]):
            selected = evaluation["site"] == site
            rows.append(
                {
                    "model": model_name,
                    "site": str(store.site_names[site]),
                    "zone": str(store.zones[site]),
                    "rmse_z": float(np.sqrt(np.mean(np.square(evaluation["target_z"][selected] - mean[selected])))),
                    "mae_z": float(np.mean(np.abs(evaluation["target_z"][selected] - mean[selected]))),
                    "adaptation_temperature": float(best["temperature"]),
                    "adaptation_gate_mean": float(best["gate"].mean()),
                }
            )
    return pd.DataFrame(rows)


def _efficiency_study(
    prediction: dict[str, np.ndarray], bank: MemoryBank, calibration: dict[str, np.ndarray], config: dict[str, Any]
) -> pd.DataFrame:
    rng = np.random.default_rng(3407)
    sample_indices = rng.choice(len(prediction["latent"]), size=min(6000, len(prediction["latent"])), replace=False)
    sample = {name: values[sample_indices] for name, values in prediction.items()}
    rows = []
    for capacity in [64, 128, 256, 512, bank.capacity]:
        chosen = np.linspace(0, bank.capacity - 1, capacity, dtype=int)
        subbank = MemoryBank(**{field: getattr(bank, field)[chosen] for field in bank.__dataclass_fields__})
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.perf_counter()
        forecast = memory_forecast(sample, subbank, calibration, min(int(config["memory"]["top_k"]), capacity))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        mse = float(np.mean(np.square(sample["target_z"] - forecast["mean"])))
        memory_bytes = sum(getattr(subbank, field).nbytes for field in subbank.__dataclass_fields__)
        rows.append(
            {
                "capacity": capacity,
                "rmse_z": math.sqrt(mse),
                "latency_ms_per_query": elapsed * 1000 / len(sample_indices),
                "queries_per_second": len(sample_indices) / elapsed,
                "memory_megabytes": memory_bytes / 1e6,
            }
        )
    return pd.DataFrame(rows)


def run_evaluation(config: dict[str, Any], force: bool = False) -> Path:
    ensure_directories()
    summary_path = ROOT / "results" / "summary.json"
    if summary_path.exists() and not force:
        return summary_path
    set_seed(int(config["project"]["seed"]))
    store = load_features()
    train = load_predictions("train")
    validation = load_predictions("validation")
    test_seen = load_predictions("test_seen")
    test_unseen = load_predictions("test_unseen")
    bank = load_memory()
    reservoir = load_memory(ROOT / "artifacts" / "memory_bank_reservoir.npz")
    calibration = load_calibration()
    horizons = [int(value) for value in config["model"]["horizons"]]
    labels = list(config["model"]["horizon_labels"])
    top_k = int(config["memory"]["top_k"])

    ridge, ridge_alpha = _fit_ridge(train, validation, store)
    full = memory_forecast(test_seen, bank, calibration, top_k)
    validation_full = memory_forecast(validation, bank, calibration, top_k)
    reservoir_calibration = calibrate_memory(reservoir, validation, config)
    reservoir_forecast = memory_forecast(test_seen, reservoir, reservoir_calibration, top_k)
    prototype_bank = MemoryBank(**{
        field: getattr(bank, field)[: int(config["memory"]["prototype_slots"])] for field in bank.__dataclass_fields__
    })
    prototype_calibration = calibrate_memory(prototype_bank, validation, config)
    prototype_forecast = memory_forecast(test_seen, prototype_bank, prototype_calibration, top_k)

    persistence = _persistence_z(test_seen, store, horizons)
    ridge_prediction = _ridge_predict(ridge, test_seen, store)
    predictions_z = {
        "Seasonal climatology": np.zeros_like(test_seen["target_z"]),
        "Persistence": persistence,
        "Ridge-AR": ridge_prediction,
        "Analog memory": full["analog_mean"],
        "Neural backbone": test_seen["mean"],
        "PMWM (reservoir)": reservoir_forecast["mean"],
        "PMWM (no event slots)": prototype_forecast["mean"],
        "PMWM (full)": full["mean"],
    }
    base_error_ratio = np.abs(validation["target_z"] - validation["mean"]) / np.sqrt(np.exp(validation["logvar"]))
    base_scale = np.square(
        np.clip(np.quantile(base_error_ratio, 0.90, axis=0) / 1.6448536269514722, 0.5, 4.0)
    )
    variances = {
        "Neural backbone": np.exp(test_seen["logvar"]) * base_scale[None],
        "PMWM (reservoir)": reservoir_forecast["variance"],
        "PMWM (no event slots)": prototype_forecast["variance"],
        "PMWM (full)": full["variance"],
    }
    metric_table, leaderboard, physical_predictions = metric_tables(
        predictions_z, variances, test_seen, store, horizons, labels
    )
    metric_table.to_csv(ROOT / "results/tables/forecast_metrics.csv", index=False)
    leaderboard.sort_values("rmse_z").to_csv(ROOT / "results/tables/leaderboard.csv", index=False)

    probabilistic = _probabilistic_table(
        test_seen["target_z"], full["mean"], full["variance"], config["evaluation"]["interval_levels"]
    )
    probabilistic.to_csv(ROOT / "results/tables/probabilistic_calibration.csv", index=False)
    event_metrics, reliability = _event_metrics(
        validation,
        validation_full["mean"],
        validation_full["variance"],
        test_seen,
        full["mean"],
        full["variance"],
        float(config["evaluation"]["extreme_z_threshold"]),
    )
    event_metrics.to_csv(ROOT / "results/tables/extreme_event_metrics.csv", index=False)
    reliability.to_csv(ROOT / "results/tables/event_reliability.csv", index=False)
    retrieval = _retrieval_quality(full, test_seen, bank, store)
    retrieval.to_csv(ROOT / "results/tables/retrieval_quality.csv", index=False)
    classification = _classification_probes(train, test_seen, test_unseen, store)
    classification.to_csv(ROOT / "results/tables/classification_probes.csv", index=False)
    datasets = make_datasets(config, store)
    imputation = _imputation_study(config, datasets["test_seen"], store)
    imputation.to_csv(ROOT / "results/tables/imputation_metrics.csv", index=False)
    bootstrap = _bootstrap_effect(test_seen, full["mean"], int(config["evaluation"]["bootstrap_replicates"]))
    bootstrap.to_csv(ROOT / "results/tables/bootstrap_significance.csv", index=False)
    continual_matrix, continual_summary = _continual_study(test_seen, bank, calibration, config, store)
    continual_matrix.to_csv(ROOT / "results/tables/continual_learning_matrix.csv", index=False)
    continual_summary.to_csv(ROOT / "results/tables/continual_learning_summary.csv", index=False)
    zero_shot = _zero_shot_study(test_unseen, bank, calibration, config, store)
    zero_shot.to_csv(ROOT / "results/tables/zero_shot_transfer.csv", index=False)
    efficiency = _efficiency_study(test_seen, bank, calibration, config)
    efficiency.to_csv(ROOT / "results/tables/memory_efficiency.csv", index=False)

    site_rows = []
    for site in np.unique(test_seen["site"]):
        selected = test_seen["site"] == site
        for model_name, mean in [("Neural backbone", test_seen["mean"]), ("PMWM (full)", full["mean"])]:
            site_rows.append(
                {
                    "site": str(store.site_names[site]),
                    "zone": str(store.zones[site]),
                    "model": model_name,
                    "rmse_z": float(np.sqrt(np.mean(np.square(test_seen["target_z"][selected] - mean[selected])))),
                    "acc": _corr(test_seen["target_z"][selected], mean[selected]),
                }
            )
    pd.DataFrame(site_rows).to_csv(ROOT / "results/tables/site_metrics.csv", index=False)

    np.savez_compressed(
        ROOT / "artifacts" / "evaluation_predictions.npz",
        origin=test_seen["origin"],
        site=test_seen["site"],
        target_z=test_seen["target_z"],
        target_physical=test_seen["target_physical"],
        backbone_z=test_seen["mean"],
        pmwm_z=full["mean"],
        pmwm_variance=full["variance"],
        persistence_z=persistence,
        ridge_z=ridge_prediction,
        pmwm_physical=physical_predictions["PMWM (full)"],
        backbone_physical=physical_predictions["Neural backbone"],
        neighbor_indices=full["indices"],
        neighbor_similarity=full["similarity"],
    )

    best = leaderboard.sort_values("rmse_z").iloc[0]
    backbone_row = leaderboard[leaderboard.model == "Neural backbone"].iloc[0]
    full_row = leaderboard[leaderboard.model == "PMWM (full)"].iloc[0]
    summary = {
        "best_model": str(best["model"]),
        "test_samples_seen": len(test_seen["origin"]),
        "test_samples_held_out": len(test_unseen["origin"]),
        "ridge_alpha": ridge_alpha,
        "pmwm_rmse_z": float(full_row["rmse_z"]),
        "backbone_rmse_z": float(backbone_row["rmse_z"]),
        "relative_rmse_improvement_over_backbone_percent": float(
            100 * (backbone_row["rmse_z"] - full_row["rmse_z"]) / backbone_row["rmse_z"]
        ),
        "pmwm_acc": float(full_row["acc"]),
        "pmwm_crps_z": float(full_row["crps_z"]),
        "coverage_90": float(probabilistic.loc[probabilistic.nominal_coverage == 0.9, "empirical_coverage"].iloc[0]),
        "event_auroc": float(event_metrics.auroc.iloc[0]),
        "event_auprc": float(event_metrics.auprc.iloc[0]),
        "retrieval_ndcg_at_k": float(retrieval.ndcg_at_k.iloc[0]),
        "bootstrap_mse_reduction_ci": [float(bootstrap.ci_lower.iloc[0]), float(bootstrap.ci_upper.iloc[0])],
        "backward_transfer": float(continual_summary.backward_transfer.iloc[0]),
    }
    atomic_json(summary_path, summary)
    return summary_path

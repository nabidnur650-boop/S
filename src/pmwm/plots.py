from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy import stats
from sklearn.decomposition import PCA

from .common import ROOT, ensure_directories, sha256_file
from .features import load_features, standardized_to_physical
from .memory import load_memory

COLORS = {
    "navy": "#16324F",
    "blue": "#2F6690",
    "teal": "#3A7D7C",
    "green": "#5B8E7D",
    "orange": "#D97941",
    "red": "#B84A62",
    "gold": "#C9A227",
    "grey": "#7A8793",
    "light": "#E9EFF3",
}


def journal_style() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 320,
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.facecolor": "#FBFCFD",
            "figure.facecolor": "white",
        }
    )


def _panel_labels(axes: Any) -> None:
    array = np.asarray(axes, dtype=object).ravel()
    for index, axis in enumerate(array):
        axis.text(
            -0.10,
            1.04,
            chr(ord("a") + index),
            transform=axis.transAxes,
            fontsize=11,
            fontweight="bold",
            va="bottom",
        )


def _save(fig: plt.Figure, stem: str, caption: str, manifest: list[dict[str, Any]]) -> None:
    png = ROOT / "figures/png" / f"{stem}.png"
    pdf = ROOT / "figures/pdf" / f"{stem}.pdf"
    fig.savefig(png, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    manifest.append(
        {
            "figure": stem.split("_")[0].replace("fig", ""),
            "stem": stem,
            "png": str(png.relative_to(ROOT)),
            "pdf": str(pdf.relative_to(ROOT)),
            "caption": caption,
            "png_sha256": sha256_file(png),
            "pdf_sha256": sha256_file(pdf),
        }
    )


def _read_table(name: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / "results/tables" / name)


def _load_evaluation() -> dict[str, np.ndarray]:
    with np.load(ROOT / "artifacts" / "evaluation_predictions.npz", allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def _fig01_architecture(manifest: list[dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 3.8))
    ax.set_xlim(0, 11.5)
    ax.set_ylim(0, 4)
    ax.axis("off")
    boxes = [
        (0.2, 1.4, 1.35, 1.25, "ERA5 Zarr\nstream", COLORS["blue"]),
        (1.9, 1.4, 1.35, 1.25, "Multi-scale\ntokenizer", COLORS["teal"]),
        (3.6, 1.4, 1.35, 1.25, "Hierarchical\nGRU encoder", COLORS["green"]),
        (5.3, 2.3, 1.45, 1.15, "Event\ndetector", COLORS["orange"]),
        (5.3, 0.55, 1.45, 1.15, "Probabilistic\nbackbone", COLORS["navy"]),
        (7.15, 2.3, 1.45, 1.15, "Learned\nconsolidation", COLORS["gold"]),
        (7.15, 0.55, 1.45, 1.15, "Top-k analog\nretrieval", COLORS["red"]),
        (9.05, 1.4, 1.55, 1.25, "Gated memory\nforecast", COLORS["navy"]),
    ]
    for x, y, width, height, text, color in boxes:
        patch = FancyBboxPatch(
            (x, y), width, height, boxstyle="round,pad=0.04,rounding_size=0.12", fc=color, ec="none", alpha=0.95
        )
        ax.add_patch(patch)
        ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", color="white", fontweight="bold")
    arrows = [
        ((1.55, 2.02), (1.9, 2.02)),
        ((3.25, 2.02), (3.6, 2.02)),
        ((4.95, 2.02), (5.3, 2.85)),
        ((4.95, 2.02), (5.3, 1.12)),
        ((6.75, 2.85), (7.15, 2.85)),
        ((7.88, 2.3), (7.88, 1.7)),
        ((6.75, 1.12), (7.15, 1.12)),
        ((8.6, 1.12), (9.05, 2.02)),
    ]
    for start, end in arrows:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12, color=COLORS["grey"], lw=1.4))
    ax.text(5.75, 3.78, "Persistent Memory World Model (PMWM)", ha="center", fontsize=14, fontweight="bold", color=COLORS["navy"])
    ax.text(5.75, 0.08, "Predict → observe → consolidate → retrieve, with bounded memory and no raw-corpus materialization", ha="center", color=COLORS["grey"])
    _save(fig, "fig01_system_architecture", "Streaming PMWM architecture and information flow.", manifest)


def _fig02_coverage(store: Any, manifest: list[dict[str, Any]]) -> None:
    fig = plt.figure(figsize=(10.5, 4.2))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2, projection="mollweide")
    ax1.plot([1959, 1994], [0, 0], lw=12, color=COLORS["blue"], solid_capstyle="butt", label="Train")
    ax1.plot([1995, 2004], [0, 0], lw=12, color=COLORS["gold"], solid_capstyle="butt", label="Validation")
    ax1.plot([2005, 2022], [0, 0], lw=12, color=COLORS["red"], solid_capstyle="butt", label="Test")
    ax1.scatter([1959, 1994, 2004, 2022], [0] * 4, color="white", edgecolor=COLORS["navy"], zorder=5)
    ax1.set_xlim(1957, 2024)
    ax1.set_ylim(-0.5, 0.5)
    ax1.set_yticks([])
    ax1.set_xlabel("Year")
    ax1.set_title("64-year chronological protocol")
    ax1.legend(loc="upper center", ncol=3)
    ax1.grid(axis="x", alpha=0.25)
    lon = np.deg2rad(((store.source_lon + 180) % 360) - 180)
    lat = np.deg2rad(store.source_lat)
    colors = np.where(store.holdout, COLORS["red"], COLORS["blue"])
    ax2.scatter(lon, lat, s=np.where(store.holdout, 65, 38), c=colors, edgecolor="white", linewidth=0.7, zorder=3)
    for i, name in enumerate(store.site_names):
        if store.holdout[i]:
            ax2.text(lon[i], lat[i] + 0.08, name, fontsize=7, ha="center", color=COLORS["red"])
    ax2.grid(True, alpha=0.25)
    ax2.set_title("16 ERA5 anchor cells (red = held out)")
    _panel_labels([ax1, ax2])
    _save(fig, "fig02_data_coverage_and_anchors", "Chronological data splits and globally distributed evaluation anchors.", manifest)


def _fig03_stream(store: Any, manifest: list[dict[str, Any]]) -> None:
    dates = pd.DatetimeIndex(store.time)
    selected_time = (dates >= "2018-07-01") & (dates < "2018-08-01")
    names = ["Seoul", "Singapore"]
    indices = [int(np.flatnonzero(store.site_names == name)[0]) for name in names]
    fig, axes = plt.subplots(4, 1, figsize=(10.5, 7.0), sharex=True)
    labels = ["Temperature (°C)", "Pressure (hPa)", "Wind speed (m s$^{-1}$)", "Precipitation (mm / 6 h)"]
    for feature, ax in enumerate(axes):
        for site, color in zip(indices, [COLORS["blue"], COLORS["orange"]]):
            ax.plot(dates[selected_time], store.target_physical[selected_time, site, feature], lw=1.1, color=color, label=store.site_names[site])
        ax.set_ylabel(labels[feature])
        if feature == 3:
            ax.set_yscale("symlog", linthresh=0.5)
    axes[0].legend(ncol=2)
    axes[0].set_title("A multivariate month from the retained ERA5 stream")
    axes[-1].set_xlabel("UTC time")
    _save(fig, "fig03_multivariate_era5_stream", "Illustrative six-hourly ERA5 streams for contrasting climate anchors.", manifest)


def _fig04_climatology(store: Any, manifest: list[dict[str, Any]]) -> None:
    dates = pd.DatetimeIndex(store.time)
    train = dates.year <= 1994
    matrix = np.zeros((len(store.site_names), 12))
    for site in range(len(store.site_names)):
        for month in range(1, 13):
            matrix[site, month - 1] = store.target_physical[train & (dates.month == month), site, 0].mean()
    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    sns.heatmap(matrix, cmap="coolwarm", center=15, xticklabels=["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"], yticklabels=store.site_names, cbar_kws={"label": "Mean 2 m temperature (°C)"}, ax=ax)
    ax.set_xlabel("Month")
    ax.set_ylabel("")
    ax.set_title("Training-period seasonal climatology")
    _save(fig, "fig04_seasonal_climatology", "Training-period monthly temperature climatology by anchor.", manifest)


def _fig05_events(store: Any, evaluation: dict[str, np.ndarray], manifest: list[dict[str, Any]]) -> None:
    dates = pd.DatetimeIndex(store.time[evaluation["origin"]])
    score = np.max(np.abs(evaluation["target_z"][:, 1]), axis=1)
    frame = pd.DataFrame({"date": dates.normalize(), "score": score}).groupby("date").max()
    window = frame.loc["2010-01-01":"2012-12-31"]
    threshold = 2.5
    fig, ax = plt.subplots(figsize=(10.5, 3.8))
    ax.plot(window.index, window.score, color=COLORS["navy"], lw=0.8)
    ax.fill_between(window.index, threshold, window.score, where=window.score >= threshold, color=COLORS["red"], alpha=0.5)
    ax.axhline(threshold, color=COLORS["red"], ls="--", lw=1, label="event threshold")
    peaks = window.nlargest(5, "score")
    ax.scatter(peaks.index, peaks.score, color=COLORS["orange"], zorder=4)
    for index, row in peaks.iterrows():
        ax.annotate(index.strftime("%Y-%m-%d"), (index, row.score), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=7)
    ax.set_ylabel("Maximum |standardized anomaly|")
    ax.set_xlabel("Forecast origin")
    ax.set_title("Event discovery promotes rare states into persistent memory")
    ax.legend()
    _save(fig, "fig05_event_discovery_timeline", "Detected high-anomaly event states in the evaluation stream.", manifest)


def _fig06_training(manifest: list[dict[str, Any]]) -> None:
    history = pd.read_csv(ROOT / "results/training_history.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.8))
    axes[0].plot(history.epoch, history.train_loss, marker="o", color=COLORS["blue"], label="train")
    axes[0].plot(history.epoch, history.validation_loss, marker="o", color=COLORS["red"], label="validation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Gaussian multi-task loss")
    axes[0].set_title("Optimization convergence")
    axes[0].legend()
    axes[1].plot(history.epoch, history.validation_rmse_z, marker="o", color=COLORS["teal"], label="forecast RMSE")
    axes[1].plot(history.epoch, np.sqrt(history.train_imputation_mse), marker="s", color=COLORS["orange"], label="imputation RMSE")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Standardized RMSE")
    axes[1].set_title("Forecast and auxiliary task")
    axes[1].legend()
    _panel_labels(axes)
    _save(fig, "fig06_training_convergence", "Training, validation, and auxiliary imputation convergence.", manifest)


def _fig07_leaderboard(manifest: list[dict[str, Any]]) -> None:
    frame = _read_table("leaderboard.csv").sort_values("rmse_z", ascending=True)
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    colors = [COLORS["red"] if name == "PMWM (full)" else COLORS["blue"] if "Neural" in name else COLORS["grey"] for name in frame.model]
    bars = ax.barh(frame.model, frame.rmse_z, color=colors)
    ax.invert_yaxis()
    ax.set_xlabel("Aggregate standardized RMSE (lower is better)")
    ax.set_title("Out-of-time forecasting leaderboard, 2005–2022")
    for bar, value in zip(bars, frame.rmse_z):
        ax.text(value + 0.004, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center", fontsize=8)
    _save(fig, "fig07_forecast_leaderboard", "Aggregate out-of-time forecast performance across targets and horizons.", manifest)


def _fig08_skill_horizon(manifest: list[dict[str, Any]]) -> None:
    metrics = _read_table("forecast_metrics.csv")
    keep = ["Persistence", "Ridge-AR", "Neural backbone", "PMWM (full)"]
    frame = metrics[metrics.model.isin(keep)].groupby(["model", "horizon"], as_index=False).skill_vs_climatology.mean()
    order = ["6 h", "24 h", "72 h", "168 h"]
    palette = {"Persistence": COLORS["grey"], "Ridge-AR": COLORS["gold"], "Neural backbone": COLORS["blue"], "PMWM (full)": COLORS["red"]}
    fig, ax = plt.subplots(figsize=(8.4, 4.5))
    for model in keep:
        selected = frame[frame.model == model].set_index("horizon").reindex(order)
        ax.plot(order, selected.skill_vs_climatology, marker="o", lw=2, label=model, color=palette[model])
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("MSE skill vs seasonal climatology")
    ax.set_xlabel("Forecast horizon")
    ax.set_title("Forecast skill across lead times")
    ax.legend(ncol=2)
    _save(fig, "fig08_skill_by_horizon", "Forecast skill relative to climatology as lead time increases.", manifest)


def _fig09_target_heatmap(manifest: list[dict[str, Any]]) -> None:
    metrics = _read_table("forecast_metrics.csv")
    full = metrics[metrics.model == "PMWM (full)"].set_index(["horizon", "target"])
    base = metrics[metrics.model == "Neural backbone"].set_index(["horizon", "target"])
    improvement = 100 * (base.rmse_z - full.rmse_z) / base.rmse_z
    matrix = improvement.unstack("target").reindex(["6 h", "24 h", "72 h", "168 h"])
    matrix = matrix.rename(
        columns={
            "precipitation_mm": "Precipitation",
            "pressure_hpa": "Pressure",
            "temperature_c": "Temperature",
            "wind_speed_ms": "Wind speed",
        }
    )
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    sns.heatmap(matrix, annot=True, fmt=".1f", cmap="RdYlGn", center=0, cbar_kws={"label": "RMSE reduction vs backbone (%)"}, ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("Forecast horizon")
    ax.set_title("Where persistent retrieval improves the neural backbone")
    _save(fig, "fig09_target_horizon_improvement", "PMWM RMSE improvement over the backbone for every target–horizon pair.", manifest)


def _fig10_site_map(store: Any, manifest: list[dict[str, Any]]) -> None:
    frame = _read_table("site_metrics.csv").pivot(index="site", columns="model", values="rmse_z")
    improvement = 100 * (frame["Neural backbone"] - frame["PMWM (full)"]) / frame["Neural backbone"]
    values = np.asarray([improvement.get(name, np.nan) for name in store.site_names])
    seen = ~store.holdout
    lon = np.deg2rad(((store.source_lon[seen] + 180) % 360) - 180)
    lat = np.deg2rad(store.source_lat[seen])
    fig = plt.figure(figsize=(9.6, 4.8))
    ax = fig.add_subplot(111, projection="mollweide")
    size = 40 + 20 * np.clip(np.abs(values[seen]), 0, 8)
    scatter = ax.scatter(lon, lat, c=values[seen], s=size, cmap="RdYlGn", vmin=-4, vmax=8, edgecolor="white", linewidth=0.8)
    offsets = {
        "Seoul": (-12, 8),
        "Tokyo": (14, 8),
        "Cairo": (-8, 10),
        "Nairobi": (14, 2),
        "London": (8, 8),
        "Reykjavik": (-8, 8),
    }
    for x, y, name in zip(lon, lat, store.site_names[seen]):
        dx, dy = offsets.get(str(name), (0, 8))
        ax.annotate(name, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=6.5, ha="center")
    ax.grid(alpha=0.25)
    ax.set_title("Site-wise RMSE reduction from persistent memory")
    fig.colorbar(scatter, ax=ax, orientation="horizontal", pad=0.08, shrink=0.65, label="RMSE reduction vs backbone (%)")
    _save(fig, "fig10_spatial_skill_map", "Spatial distribution of PMWM improvements at seen anchors.", manifest)


def _case_window(evaluation: dict[str, np.ndarray]) -> np.ndarray:
    score = np.max(np.abs(evaluation["target_z"][:, 1]), axis=1)
    center = int(np.argmax(score))
    site = evaluation["site"][center]
    site_indices = np.flatnonzero(evaluation["site"] == site)
    position = int(np.flatnonzero(site_indices == center)[0])
    return site_indices[max(0, position - 35) : position + 36]


def _fig11_case(store: Any, evaluation: dict[str, np.ndarray], manifest: list[dict[str, Any]]) -> None:
    indices = _case_window(evaluation)
    dates = pd.DatetimeIndex(store.time[evaluation["origin"][indices] + 4])
    site_name = store.site_names[evaluation["site"][indices[0]]]
    actual = evaluation["target_physical"][indices, 1]
    full = evaluation["pmwm_physical"][indices, 1]
    base = evaluation["backbone_physical"][indices, 1]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.2), sharex=True)
    labels = ["Temperature (°C)", "Pressure (hPa)", "Wind speed (m s$^{-1}$)", "Precipitation (mm / 6 h)"]
    for feature, ax in enumerate(axes.ravel()):
        ax.plot(dates, actual[:, feature], color="black", lw=1.5, label="ERA5")
        ax.plot(dates, base[:, feature], color=COLORS["blue"], lw=1.0, alpha=0.8, label="backbone")
        ax.plot(dates, full[:, feature], color=COLORS["red"], lw=1.3, label="PMWM")
        ax.set_ylabel(labels[feature])
        if feature == 3:
            ax.set_yscale("symlog", linthresh=0.3)
    axes[0, 0].legend(ncol=3)
    axes[0, 0].set_title(f"24 h event case study — {site_name}")
    for label in [*axes[-1, 0].get_xticklabels(), *axes[-1, 1].get_xticklabels()]:
        label.set_rotation(18)
        label.set_ha("right")
    _panel_labels(axes)
    _save(fig, "fig11_extreme_event_case_study", "Observed and predicted trajectories around the strongest held-out event window.", manifest)


def _fig12_intervals(store: Any, evaluation: dict[str, np.ndarray], manifest: list[dict[str, Any]]) -> None:
    indices = _case_window(evaluation)
    horizon_idx, horizon = 1, 4
    origins = evaluation["origin"][indices]
    sites = evaluation["site"][indices]
    sigma = np.sqrt(evaluation["pmwm_variance"][indices, horizon_idx])
    lower_z = evaluation["pmwm_z"][indices].copy()
    upper_z = evaluation["pmwm_z"][indices].copy()
    lower_z[:, horizon_idx] -= 1.6448536 * sigma
    upper_z[:, horizon_idx] += 1.6448536 * sigma
    lower = standardized_to_physical(lower_z, origins, sites, [1, 4, 12, 28], store)[:, horizon_idx]
    upper = standardized_to_physical(upper_z, origins, sites, [1, 4, 12, 28], store)[:, horizon_idx]
    dates = pd.DatetimeIndex(store.time[origins + horizon])
    fig, axes = plt.subplots(2, 1, figsize=(10.2, 5.4), sharex=True)
    for ax, feature, label in [(axes[0], 0, "Temperature (°C)"), (axes[1], 2, "Wind speed (m s$^{-1}$)")]:
        ax.fill_between(dates, lower[:, feature], upper[:, feature], color=COLORS["red"], alpha=0.18, label="90% interval")
        ax.plot(dates, evaluation["target_physical"][indices, horizon_idx, feature], color="black", lw=1.3, label="ERA5")
        ax.plot(dates, evaluation["pmwm_physical"][indices, horizon_idx, feature], color=COLORS["red"], lw=1.2, label="PMWM mean")
        ax.set_ylabel(label)
    axes[0].legend(ncol=3)
    axes[0].set_title("Calibrated predictive uncertainty during the event window")
    _panel_labels(axes)
    _save(fig, "fig12_predictive_intervals", "Calibrated PMWM uncertainty intervals for an extreme-event window.", manifest)


def _fig13_calibration(manifest: list[dict[str, Any]]) -> None:
    coverage = _read_table("probabilistic_calibration.csv")
    reliability = _read_table("event_reliability.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))
    axes[0].plot([0, 1], [0, 1], ls="--", color=COLORS["grey"])
    axes[0].plot(coverage.nominal_coverage, coverage.empirical_coverage, marker="o", color=COLORS["red"], lw=2)
    axes[0].set_xlabel("Nominal interval coverage")
    axes[0].set_ylabel("Empirical coverage")
    axes[0].set_xlim(0.45, 1.0)
    axes[0].set_ylim(0.45, 1.0)
    axes[0].set_title("Forecast interval calibration")
    axes[1].plot([0, 1], [0, 1], ls="--", color=COLORS["grey"])
    axes[1].plot(reliability.predicted_probability, reliability.observed_frequency, marker="o", color=COLORS["orange"], lw=2)
    axes[1].set_xlabel("Predicted extreme probability")
    axes[1].set_ylabel("Observed extreme frequency")
    axes[1].set_title("Extreme-event reliability")
    _panel_labels(axes)
    _save(fig, "fig13_probabilistic_calibration", "Coverage and extreme-event reliability diagnostics.", manifest)


def _fig14_retrieval(evaluation: dict[str, np.ndarray], manifest: list[dict[str, Any]]) -> None:
    quality = _read_table("retrieval_quality.csv").iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.0))
    axes[0].hist(evaluation["neighbor_similarity"][:, 0], bins=45, color=COLORS["blue"], alpha=0.85, density=True)
    axes[0].axvline(evaluation["neighbor_similarity"][:, 0].mean(), color=COLORS["red"], ls="--")
    axes[0].set_xlabel("Top-1 cosine similarity")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Analog retrieval confidence")
    labels = ["Precision@k", "Hit@k", "NDCG@k", "Zone match", "Season match"]
    values = [quality.precision_at_k, quality.hit_rate_at_k, quality.ndcg_at_k, quality.same_zone_at_k, quality.season_match_at_k]
    axes[1].barh(labels, values, color=[COLORS["teal"], COLORS["green"], COLORS["gold"], COLORS["blue"], COLORS["orange"]])
    axes[1].set_xlim(0, 1)
    axes[1].set_xlabel("Score")
    axes[1].set_title("Retrieval relevance diagnostics")
    _panel_labels(axes)
    _save(fig, "fig14_retrieval_quality", "Similarity distribution and retrieval relevance metrics.", manifest)


def _fig15_memory_embedding(bank: Any, manifest: list[dict[str, Any]]) -> None:
    projection = PCA(n_components=2, random_state=3407).fit_transform(bank.keys)
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), layout="constrained")
    scatter = axes[0].scatter(projection[:, 0], projection[:, 1], c=bank.month, cmap="twilight_shifted", s=14 + 10 * (bank.kind == "event"), alpha=0.75)
    axes[0].set_title("Memory space by calendar regime")
    axes[0].set_xlabel("PC 1")
    axes[0].set_ylabel("PC 2")
    fig.colorbar(scatter, ax=axes[0], label="Month")
    scatter2 = axes[1].scatter(projection[:, 0], projection[:, 1], c=bank.event_score, cmap="magma", s=14 + 14 * (bank.kind == "event"), alpha=0.78)
    axes[1].set_title("Event-aware memory allocation")
    axes[1].set_xlabel("PC 1")
    axes[1].set_ylabel("PC 2")
    fig.colorbar(scatter2, ax=axes[1], label="Event score")
    _panel_labels(axes)
    _save(fig, "fig15_memory_embedding", "Two-dimensional view of consolidated memory keys by season and event intensity.", manifest)


def _fig16_memory_graph(bank: Any, manifest: list[dict[str, Any]]) -> None:
    selected = np.argsort(bank.event_score)[-36:]
    keys = bank.keys[selected]
    similarity = keys @ keys.T
    graph = nx.Graph()
    for i in range(len(selected)):
        graph.add_node(i)
        neighbors = np.argsort(similarity[i])[-4:-1]
        for j in neighbors:
            if similarity[i, j] > 0.35:
                graph.add_edge(i, int(j), weight=float(similarity[i, j]))
    position = nx.spring_layout(graph, seed=3407, weight="weight", k=0.55)
    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    edge_width = [0.5 + 2.2 * graph[u][v]["weight"] for u, v in graph.edges]
    nx.draw_networkx_edges(graph, position, width=edge_width, alpha=0.25, edge_color=COLORS["grey"], ax=ax)
    node = nx.draw_networkx_nodes(
        graph,
        position,
        node_color=bank.month[selected],
        cmap="twilight_shifted",
        node_size=80 + 90 * bank.event_score[selected] / max(bank.event_score[selected].max(), 1e-6),
        edgecolors="white",
        linewidths=0.7,
        ax=ax,
    )
    fig.colorbar(node, ax=ax, label="Month")
    ax.set_title("Temporal memory graph of high-impact analogs")
    ax.axis("off")
    _save(fig, "fig16_temporal_memory_graph", "Graph linking high-impact memories by latent-state similarity.", manifest)


def _fig17_ablation(manifest: list[dict[str, Any]]) -> None:
    frame = _read_table("leaderboard.csv")
    order = ["Neural backbone", "PMWM (reservoir)", "PMWM (no event slots)", "PMWM (full)"]
    frame = frame.set_index("model").reindex(order).reset_index()
    backbone = float(frame.loc[frame.model == "Neural backbone", "rmse_z"].iloc[0])
    frame["improvement"] = 100 * (backbone - frame.rmse_z) / backbone
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    colors = [COLORS["grey"], COLORS["gold"], COLORS["blue"], COLORS["red"]]
    bars = ax.bar(frame.model, frame.improvement, color=colors)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("RMSE reduction vs backbone (%)")
    ax.set_title("Ablation: consolidation and event retention both matter")
    ax.tick_params(axis="x", rotation=18)
    ax.set_ylim(-0.04, max(frame.improvement) + 0.22)
    for bar, value in zip(bars, frame.improvement):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.035, f"{value:.2f}%", ha="center", fontsize=8)
    _save(fig, "fig17_memory_ablation", "Ablation of consolidation and event-specific memory slots.", manifest)


def _fig18_efficiency(manifest: list[dict[str, Any]]) -> None:
    frame = _read_table("memory_efficiency.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.0))
    axes[0].plot(frame.capacity, frame.rmse_z, marker="o", color=COLORS["red"], lw=2)
    axes[0].set_xscale("log", base=2)
    axes[0].set_xlabel("Memory capacity")
    axes[0].set_ylabel("Standardized RMSE")
    axes[0].set_title("Accuracy–capacity frontier")
    scatter = axes[1].scatter(frame.latency_ms_per_query, frame.rmse_z, c=frame.capacity, s=70, cmap="viridis")
    for _, row in frame.iterrows():
        axes[1].annotate(str(int(row.capacity)), (row.latency_ms_per_query, row.rmse_z), xytext=(4, 4), textcoords="offset points", fontsize=7)
    axes[1].set_xlabel("Retrieval latency (ms / query)")
    axes[1].set_ylabel("Standardized RMSE")
    axes[1].set_title("Latency–accuracy Pareto view")
    fig.colorbar(scatter, ax=axes[1], label="Capacity")
    _panel_labels(axes)
    _save(fig, "fig18_memory_efficiency_pareto", "Memory capacity, accuracy, and retrieval latency trade-offs.", manifest)


def _fig19_continual(manifest: list[dict[str, Any]]) -> None:
    frame = _read_table("continual_learning_matrix.csv")
    matrix = frame.pivot(index="memory_state", columns="task_label", values="memory_skill_vs_backbone")
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    sns.heatmap(matrix * 100, annot=True, fmt=".2f", cmap="RdYlGn", center=0, cbar_kws={"label": "Memory skill vs backbone (%)"}, ax=ax)
    ax.set_xlabel("Evaluation era")
    ax.set_ylabel("Memory state after adaptation stage")
    ax.set_title("Prequential continual learning and backward transfer")
    _save(fig, "fig19_continual_learning_matrix", "Continual-learning performance across sequential test eras.", manifest)


def _fig20_zero_shot(manifest: list[dict[str, Any]]) -> None:
    frame = _read_table("zero_shot_transfer.csv")
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    sns.barplot(data=frame, x="site", y="rmse_z", hue="model", palette=[COLORS["grey"], COLORS["blue"], COLORS["red"]], ax=ax)
    ax.set_xlabel("Held-out climate anchor")
    ax.set_ylabel("Standardized RMSE")
    ax.set_title("Zero-shot transfer and one-year memory adaptation")
    ax.legend(title="", ncol=3, loc="upper center")
    _save(fig, "fig20_zero_shot_domain_adaptation", "Forecasting transfer to four held-out climate anchors before and after memory adaptation.", manifest)


def _fig21_imputation(manifest: list[dict[str, Any]]) -> None:
    frame = _read_table("imputation_metrics.csv")
    fig, ax = plt.subplots(figsize=(8.4, 4.5))
    palette = {"Seasonal mean": COLORS["grey"], "Last observation": COLORS["gold"], "PMWM imputation head": COLORS["red"]}
    for method, group in frame.groupby("method"):
        ax.plot(group.missing_fraction, group.rmse_z, marker="o", lw=2, label=method, color=palette[method])
    ax.set_xlabel("Fraction of current-step variables missing")
    ax.set_ylabel("Masked-value RMSE (standardized)")
    ax.set_title("Robust current-step imputation under increasing missingness")
    ax.legend()
    _save(fig, "fig21_missing_value_imputation", "Masked current-step imputation performance as missingness increases.", manifest)


def _autocorrelation(values: np.ndarray, max_lag: int) -> np.ndarray:
    values = values - values.mean()
    denominator = np.dot(values, values)
    return np.asarray([np.dot(values[:-lag], values[lag:]) / denominator for lag in range(1, max_lag + 1)])


def _fig22_residuals(evaluation: dict[str, np.ndarray], manifest: list[dict[str, Any]]) -> None:
    selected_site = np.min(evaluation["site"])
    selected = evaluation["site"] == selected_site
    base_site = (evaluation["target_z"] - evaluation["backbone_z"])[selected, 1, 0]
    full_site = (evaluation["target_z"] - evaluation["pmwm_z"])[selected, 1, 0]
    base = (evaluation["target_z"] - evaluation["backbone_z"])[:, 1, 0]
    full = (evaluation["target_z"] - evaluation["pmwm_z"])[:, 1, 0]
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.0))
    lags = np.arange(1, 31)
    axes[0].plot(lags, _autocorrelation(base_site, 30), color=COLORS["blue"], marker="o", ms=3, label="backbone")
    axes[0].plot(lags, _autocorrelation(full_site, 30), color=COLORS["red"], marker="o", ms=3, label="PMWM")
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set_xlabel("Daily-origin lag")
    axes[0].set_ylabel("Residual autocorrelation")
    axes[0].set_title("24 h temperature residual memory — Seoul")
    axes[0].legend()
    quantiles = np.linspace(0.005, 0.995, 199)
    theoretical = stats.norm.ppf(quantiles)
    axes[1].plot(theoretical, np.quantile(base, quantiles), color=COLORS["blue"], label="backbone")
    axes[1].plot(theoretical, np.quantile(full, quantiles), color=COLORS["red"], label="PMWM")
    axes[1].plot([-3, 3], [-3, 3], ls="--", color=COLORS["grey"])
    axes[1].set_xlabel("Normal theoretical quantile")
    axes[1].set_ylabel("Residual quantile")
    axes[1].set_title("Residual tail diagnostics")
    axes[1].legend()
    _panel_labels(axes)
    _save(fig, "fig22_residual_diagnostics", "Autocorrelation and quantile diagnostics for backbone and PMWM residuals.", manifest)


def _fig23_bootstrap(evaluation: dict[str, np.ndarray], manifest: list[dict[str, Any]]) -> None:
    rng = np.random.default_rng(3407)
    rows = []
    for horizon in range(4):
        for target in range(4):
            base_error = np.square(evaluation["target_z"][:, horizon, target] - evaluation["backbone_z"][:, horizon, target])
            full_error = np.square(evaluation["target_z"][:, horizon, target] - evaluation["pmwm_z"][:, horizon, target])
            block = evaluation["origin"] // 28
            frame = pd.DataFrame({"block": block, "delta": base_error - full_error}).groupby("block").delta.mean().to_numpy()
            draws = np.asarray([rng.choice(frame, len(frame), replace=True).mean() for _ in range(200)])
            rows.append(
                {
                    "label": f"{['6 h','24 h','72 h','168 h'][horizon]} · {['Temp','Pressure','Wind','Precip'][target]}",
                    "mean": frame.mean(),
                    "lower": np.quantile(draws, 0.025),
                    "upper": np.quantile(draws, 0.975),
                }
            )
    frame = pd.DataFrame(rows).sort_values("mean")
    fig, ax = plt.subplots(figsize=(8.8, 6.6))
    y = np.arange(len(frame))
    ax.errorbar(frame["mean"], y, xerr=[frame["mean"] - frame.lower, frame.upper - frame["mean"]], fmt="o", color=COLORS["red"], ecolor=COLORS["grey"], capsize=2)
    ax.axvline(0, color="black", lw=0.9, ls="--")
    ax.set_yticks(y, frame.label)
    ax.set_xlabel("Block-bootstrap MSE reduction (positive favors PMWM)")
    ax.set_title("Uncertainty of the memory contribution")
    _save(fig, "fig23_bootstrap_effects", "Block-bootstrap confidence intervals for PMWM MSE reductions.", manifest)


def _fig24_scorecard(manifest: list[dict[str, Any]]) -> None:
    with (ROOT / "results/summary.json").open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    memory = json.loads((ROOT / "artifacts/memory_manifest.json").read_text(encoding="utf-8"))
    values = [
        ("RMSE improvement", f"{summary['relative_rmse_improvement_over_backbone_percent']:.2f}%"),
        ("Anomaly correlation", f"{summary['pmwm_acc']:.3f}"),
        ("90% interval coverage", f"{summary['coverage_90']:.3f}"),
        ("Extreme-event AUROC", f"{summary['event_auroc']:.3f}"),
        ("Retrieval NDCG@k", f"{summary['retrieval_ndcg_at_k']:.3f}"),
        ("Memory compression", f"{memory['compression_ratio']:.0f}×"),
    ]
    fig, ax = plt.subplots(figsize=(10.4, 3.8))
    ax.axis("off")
    for index, (label, value) in enumerate(values):
        x = 0.03 + (index % 3) * 0.325
        y = 0.57 if index < 3 else 0.12
        box = FancyBboxPatch((x, y), 0.285, 0.32, transform=ax.transAxes, boxstyle="round,pad=0.02", fc=COLORS["light"], ec="none")
        ax.add_patch(box)
        ax.text(x + 0.142, y + 0.205, value, transform=ax.transAxes, ha="center", va="center", fontsize=17, fontweight="bold", color=COLORS["navy"])
        ax.text(x + 0.142, y + 0.075, label, transform=ax.transAxes, ha="center", va="center", fontsize=8.5, color=COLORS["grey"])
    ax.set_title("PMWM empirical scorecard", fontsize=14, color=COLORS["navy"], pad=8)
    _save(fig, "fig24_empirical_scorecard", "Compact scorecard of forecasting, calibration, retrieval, and efficiency results.", manifest)


def generate_figures(config: dict[str, Any], force: bool = False) -> Path:
    ensure_directories()
    manifest_path = ROOT / "results" / "figure_manifest.csv"
    if manifest_path.exists() and not force:
        return manifest_path
    journal_style()
    store = load_features()
    evaluation = _load_evaluation()
    bank = load_memory()
    manifest: list[dict[str, Any]] = []
    _fig01_architecture(manifest)
    _fig02_coverage(store, manifest)
    _fig03_stream(store, manifest)
    _fig04_climatology(store, manifest)
    _fig05_events(store, evaluation, manifest)
    _fig06_training(manifest)
    _fig07_leaderboard(manifest)
    _fig08_skill_horizon(manifest)
    _fig09_target_heatmap(manifest)
    _fig10_site_map(store, manifest)
    _fig11_case(store, evaluation, manifest)
    _fig12_intervals(store, evaluation, manifest)
    _fig13_calibration(manifest)
    _fig14_retrieval(evaluation, manifest)
    _fig15_memory_embedding(bank, manifest)
    _fig16_memory_graph(bank, manifest)
    _fig17_ablation(manifest)
    _fig18_efficiency(manifest)
    _fig19_continual(manifest)
    _fig20_zero_shot(manifest)
    _fig21_imputation(manifest)
    _fig22_residuals(evaluation, manifest)
    _fig23_bootstrap(evaluation, manifest)
    _fig24_scorecard(manifest)
    pd.DataFrame(manifest).to_csv(manifest_path, index=False)
    return manifest_path

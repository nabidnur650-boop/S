from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from .common import sha256_file
from .q1_common import Q1_ROOT
from .q1_v3 import V3_ROOT, _load_npz, load_v3_config, load_v3_features, verify_v3_lock

COLORS = {
    "navy": "#17324D",
    "blue": "#3B6FB6",
    "teal": "#0F9D8A",
    "orange": "#E76F51",
    "gold": "#E9B949",
    "gray": "#687684",
    "light": "#E9EEF2",
    "red": "#B23A48",
}


def _style() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.titlesize": 13,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _save(fig: plt.Figure, name: str) -> None:
    png = V3_ROOT / "figures" / "png" / f"{name}.png"
    pdf = V3_ROOT / "figures" / "pdf" / f"{name}.pdf"
    png.parent.mkdir(parents=True, exist_ok=True)
    pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=360, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _box(
    axis: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    color: str,
) -> None:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.025",
        linewidth=1.2,
        edgecolor=color,
        facecolor=mpl.colors.to_rgba(color, 0.10),
    )
    axis.add_patch(patch)
    axis.text(xy[0] + width / 2, xy[1] + height / 2, text, ha="center", va="center", weight="bold")


def figure_protocol_timeline() -> None:
    fig, axis = plt.subplots(figsize=(10.2, 3.5), constrained_layout=True)
    periods = [
        (1959, 1994, "Fit\n64 cells", COLORS["navy"]),
        (1995, 2004, "Validate", COLORS["blue"]),
        (2005, 2016, "Develop\n16 cells", COLORS["gold"]),
        (2017, 2022, "Fresh confirm\n32 cells", COLORS["teal"]),
    ]
    for start, end, label, color in periods:
        axis.barh(0, end - start + 1, left=start, height=0.52, color=color, alpha=0.92)
        axis.text((start + end) / 2, 0, label, ha="center", va="center", color="white", weight="bold")
    for value, text in [(2004.5, "architecture frozen"), (2016.5, "v3 lock"), (2022.5, "final readout")]:
        axis.axvline(value, color=COLORS["gray"], linestyle="--", linewidth=1)
        axis.text(value, 0.47, text, rotation=35, ha="left", va="bottom", color=COLORS["gray"])
    axis.set_xlim(1958, 2024)
    axis.set_ylim(-0.6, 0.9)
    axis.set_yticks([])
    axis.set_xlabel("ERA5 year (6-hour cadence; daily forecast origins)")
    axis.set_title("Leakage-audited chronological protocol", loc="left", weight="bold")
    _save(fig, "01_protocol_timeline")


def figure_spatial_design() -> None:
    v2 = pd.read_csv(Q1_ROOT / "sites.csv")
    v3 = pd.read_csv(V3_ROOT / "sites.csv")
    fig, axis = plt.subplots(figsize=(10.5, 4.8), constrained_layout=True)
    axis.set_facecolor("#F7F9FB")
    for latitude in (-60, -30, 0, 30, 60):
        axis.axhline(latitude, color="white", linewidth=1.2, zorder=0)
    palette = {
        "train": (COLORS["navy"], "o", 32, "Fit cells (64)"),
        "development": (COLORS["gold"], "s", 44, "Development cells (16)"),
        "confirmatory": (COLORS["orange"], "^", 44, "Opened v2 confirmation (32)"),
    }
    for partition, group in v2.groupby("partition"):
        color, marker, size, label = palette[partition]
        axis.scatter(group.longitude, group.latitude, s=size, marker=marker, color=color, label=label, alpha=0.85)
    axis.scatter(
        v3.longitude,
        v3.latitude,
        s=62,
        marker="*",
        color=COLORS["teal"],
        edgecolor="white",
        linewidth=0.6,
        label="Fresh v3 confirmation (32)",
        zorder=4,
    )
    axis.set(xlim=(-185, 185), ylim=(-82, 82), xlabel="Longitude", ylabel="Latitude")
    axis.set_title("Global cell partitions are disjoint by construction", loc="left", weight="bold")
    axis.legend(ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.30), frameon=False)
    _save(fig, "02_spatial_design")


def figure_architecture() -> None:
    fig, axis = plt.subplots(figsize=(11.2, 4.5), constrained_layout=True)
    axis.set(xlim=(0, 1), ylim=(0, 1))
    axis.axis("off")
    blocks = [
        ((0.02, 0.37), 0.14, 0.26, "56-step\nmultivariate\ncontext", COLORS["navy"]),
        ((0.21, 0.37), 0.14, 0.26, "Inverted-variable\nTransformer", COLORS["blue"]),
        ((0.40, 0.56), 0.15, 0.22, "Persistence-\nresidual head", COLORS["gold"]),
        ((0.40, 0.19), 0.15, 0.22, "192-D retrieval\nkey", COLORS["teal"]),
        ((0.61, 0.19), 0.16, 0.22, "Bounded\nevent-aware\nmemory", COLORS["teal"]),
        ((0.82, 0.37), 0.16, 0.26, "Calibrated\nprobabilistic\nforecast", COLORS["orange"]),
    ]
    for xy, width, height, text, color in blocks:
        _box(axis, xy, width, height, text, color)
    arrows = [
        ((0.16, 0.50), (0.21, 0.50)),
        ((0.35, 0.50), (0.40, 0.67)),
        ((0.35, 0.50), (0.40, 0.30)),
        ((0.55, 0.30), (0.61, 0.30)),
        ((0.55, 0.67), (0.82, 0.55)),
        ((0.77, 0.30), (0.82, 0.45)),
    ]
    for start, end in arrows:
        axis.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12, color=COLORS["gray"]))
    axis.text(0.50, 0.91, "PMWM-IR: persistence establishes the safety floor; memory retrieves residual corrections", ha="center", weight="bold", size=12)
    axis.text(0.69, 0.08, "2,048 slots · top-16 retrieval · fixed backbone weights", ha="center", color=COLORS["gray"])
    _save(fig, "03_pmwm_ir_architecture")


def figure_memory_design() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.8), constrained_layout=True)
    titles = ["Discover", "Consolidate", "Retrieve + update"]
    for axis, title in zip(axes, titles):
        axis.set(xlim=(0, 1), ylim=(0, 1))
        axis.axis("off")
        axis.set_title(title, loc="left", weight="bold")
    rng = np.random.default_rng(8)
    points = rng.normal(size=(180, 2))
    score = np.linalg.norm(points, axis=1)
    axes[0].scatter(points[:, 0] / 7 + 0.5, points[:, 1] / 7 + 0.5, s=10, c=np.where(score > np.quantile(score, 0.9), COLORS["orange"], COLORS["blue"]), alpha=0.65)
    axes[0].text(0.05, 0.05, "top-decile outcomes reserve event slots", color=COLORS["gray"])
    for index in range(12):
        x = 0.13 + (index % 4) * 0.20
        y = 0.75 - (index // 4) * 0.25
        color = COLORS["orange"] if index >= 9 else COLORS["teal"]
        axes[1].add_patch(plt.Circle((x, y), 0.055, color=color, alpha=0.8))
    axes[1].text(0.08, 0.05, "1,536 regular + 512 event prototypes", color=COLORS["gray"])
    _box(axes[2], (0.04, 0.55), 0.25, 0.20, "new key", COLORS["blue"])
    _box(axes[2], (0.39, 0.55), 0.25, 0.20, "top-k", COLORS["teal"])
    _box(axes[2], (0.70, 0.55), 0.25, 0.20, "residual", COLORS["orange"])
    axes[2].add_patch(FancyArrowPatch((0.29, 0.65), (0.39, 0.65), arrowstyle="-|>", mutation_scale=12))
    axes[2].add_patch(FancyArrowPatch((0.64, 0.65), (0.70, 0.65), arrowstyle="-|>", mutation_scale=12))
    axes[2].text(0.05, 0.22, "forecast first", weight="bold")
    axes[2].text(0.36, 0.22, "wait 168 h", weight="bold")
    axes[2].text(0.70, 0.22, "insert", weight="bold")
    axes[2].text(0.06, 0.05, "strict delayed prequential order", color=COLORS["gray"])
    _save(fig, "04_memory_consolidation_and_update")


def figure_v2_gate() -> None:
    table = pd.read_csv(Q1_ROOT / "results" / "tables" / "leaderboard.csv").sort_values("rmse_z_mean")
    fig, axis = plt.subplots(figsize=(8.5, 5.6), constrained_layout=True)
    colors = [COLORS["teal"] if name == "PMWM full" else COLORS["orange"] if name == "Persistence" else COLORS["gray"] for name in table.model]
    axis.barh(table.model, table.rmse_z_mean, color=colors)
    axis.invert_yaxis()
    axis.set_xlabel("Confirmatory standardized RMSE (lower is better)")
    axis.set_title("v2 falsification gate: memory beat its backbone, but persistence won", loc="left", weight="bold")
    for index, value in enumerate(table.rmse_z_mean):
        axis.text(value + 0.01, index, f"{value:.3f}", va="center", size=8)
    _save(fig, "05_v2_persistence_gate")


def figure_v2_effects() -> None:
    table = pd.read_csv(Q1_ROOT / "results" / "tables" / "confirmatory_effects.csv")
    matrix = table.pivot(index="target", columns="horizon", values="mse_reduction_z")
    order = [label for label in ["6 h", "24 h", "72 h", "168 h"] if label in matrix.columns]
    fig, axis = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    sns.heatmap(matrix[order], center=0, cmap="vlag", annot=True, fmt=".3f", linewidths=0.5, ax=axis, cbar_kws={"label": "backbone MSE − PMWM MSE"})
    axis.set_title("v2 memory effect was heterogeneous across targets and horizons", loc="left", weight="bold")
    axis.set(xlabel="Forecast horizon", ylabel="Target")
    _save(fig, "06_v2_target_horizon_effects")


def figure_development_selection() -> None:
    table = pd.read_csv(V3_ROOT / "DEVELOPMENT_SELECTION.csv").sort_values("rmse_z")
    fig, axis = plt.subplots(figsize=(8.8, 6.0), constrained_layout=True)
    colors = [COLORS["teal"] if name == "PMWM-IR event-aware" else COLORS["orange"] if name == "Persistence" else COLORS["gray"] for name in table.model]
    axis.barh(table.model, table.rmse_z, color=colors)
    axis.invert_yaxis()
    axis.set_xlabel("Development spatial RMSE (seed 3407; lower is better)")
    axis.set_title("Development-only selection before the fresh protocol lock", loc="left", weight="bold")
    for index, value in enumerate(table.rmse_z):
        axis.text(value + 0.008, index, f"{value:.3f}", va="center", size=8)
    _save(fig, "07_development_model_selection")


def figure_v3_leaderboard() -> None:
    table = pd.read_csv(V3_ROOT / "results" / "tables" / "leaderboard.csv").sort_values("rmse_z_mean")
    fig, axis = plt.subplots(figsize=(9.0, 6.2), constrained_layout=True)
    colors = [COLORS["teal"] if name == "PMWM-IR event-aware" else COLORS["orange"] if name == "Persistence" else COLORS["gray"] for name in table.model]
    error = table.rmse_z_sd.fillna(0)
    y = np.arange(len(table))
    for index, row in enumerate(table.itertuples(index=False)):
        axis.errorbar(
            row.rmse_z_mean,
            index,
            xerr=error.iloc[index],
            fmt="o",
            color=colors[index],
            ecolor=colors[index],
            capsize=3,
            markersize=6,
        )
        axis.text(row.rmse_z_mean + 0.006, index, f"{row.rmse_z_mean:.3f}", va="center", size=7)
    axis.set_yticks(y, table.model)
    axis.invert_yaxis()
    axis.set_xlim(table.rmse_z_mean.min() - 0.025, table.rmse_z_mean.max() + 0.055)
    axis.set_xlabel("Fresh-confirmation standardized RMSE (mean ± seed SD)")
    axis.set_title("Fresh 2017–2022 confirmation", loc="left", weight="bold")
    _save(fig, "08_v3_fresh_leaderboard")


def figure_bootstrap() -> None:
    with np.load(V3_ROOT / "artifacts" / "bootstrap_draws.npz", allow_pickle=False) as archive:
        keys = archive.files
        titles = {
            "residual_itransformer_minus_pmwm_ir": "Matched residual iTransformer − PMWM-IR",
            "persistence_minus_pmwm_ir": "Persistence − PMWM-IR",
        }
        fig, axes = plt.subplots(len(keys), 1, figsize=(8.4, 2.7 * len(keys)), constrained_layout=True, squeeze=False)
        for axis, key in zip(axes[:, 0], keys):
            values = archive[key]
            sns.histplot(values, bins=45, stat="density", color=COLORS["teal"], alpha=0.75, ax=axis)
            axis.axvline(0, color=COLORS["red"], linestyle="--", linewidth=1.2)
            lower, upper = np.quantile(values, [0.025, 0.975])
            axis.axvspan(lower, upper, color=COLORS["gold"], alpha=0.18)
            axis.set_title(titles.get(key, key.replace("_", " ").title()), loc="left", weight="bold")
            axis.set_xlabel("Competitor MSE − PMWM-IR MSE")
            axis.text(
                0.99,
                0.90,
                f"95% CI [{lower:.4f}, {upper:.4f}]  ·  n={len(values):,}",
                transform=axis.transAxes,
                ha="right",
                va="top",
                size=8,
                bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none", "pad": 3},
            )
    _save(fig, "09_primary_block_bootstrap")


def figure_primary_forest() -> None:
    summary = json.loads((V3_ROOT / "results" / "summary.json").read_text())
    rows = []
    for name, value in summary["contrasts"].items():
        rows.append(
            {
                "contrast": name.replace("_minus_pmwm_ir", "").replace("_", " "),
                "effect": value["mse_reduction_z"],
                "lower": value["bootstrap_ci_95"][0],
                "upper": value["bootstrap_ci_95"][1],
            }
        )
    table = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, len(table), figsize=(5.2 * len(table), 3.6), constrained_layout=True, squeeze=False)
    for axis, row in zip(axes[0], table.itertuples(index=False)):
        axis.errorbar(
            row.effect,
            0,
            xerr=[[row.effect - row.lower], [row.upper - row.effect]],
            fmt="o",
            color=COLORS["navy"],
            ecolor=COLORS["blue"],
            capsize=5,
            markersize=7,
        )
        axis.axvline(0, color=COLORS["red"], linestyle="--")
        span = max(row.upper - row.lower, abs(row.effect) * 0.20, 1e-3)
        axis.set_xlim(min(0, row.lower) - 0.25 * span, max(0, row.upper) + 0.25 * span)
        axis.set_yticks([])
        axis.set_title(row.contrast, loc="left", weight="bold")
        axis.set_xlabel("Competitor MSE − PMWM-IR MSE")
        axis.text(
            0.02,
            0.10,
            f"{row.effect:.4f} [{row.lower:.4f}, {row.upper:.4f}]",
            transform=axis.transAxes,
            size=8,
        )
    fig.suptitle("Prespecified fresh-confirmation contrasts (95% moving-block CIs)", x=0.01, ha="left", weight="bold")
    _save(fig, "10_primary_effect_forest")


def figure_effect_heatmaps() -> None:
    table = pd.read_csv(V3_ROOT / "results" / "tables" / "confirmatory_effects.csv")
    contrasts = table.contrast.unique()
    fig, axes = plt.subplots(1, len(contrasts), figsize=(5.1 * len(contrasts), 4.3), constrained_layout=True, squeeze=False)
    for axis, contrast in zip(axes[0], contrasts):
        subset = table[table.contrast == contrast]
        matrix = subset.pivot(index="target", columns="horizon", values="mse_reduction_z")
        matrix = matrix[[label for label in ["6 h", "24 h", "72 h", "168 h"] if label in matrix.columns]]
        sns.heatmap(matrix, center=0, cmap="vlag", annot=True, fmt=".3f", linewidths=0.4, ax=axis, cbar=False)
        axis.set_title(contrast.replace("_", " "), loc="left", weight="bold")
        axis.set(xlabel="Horizon", ylabel="Target")
    _save(fig, "11_target_horizon_confirmatory_effects")


def figure_spatial_effects() -> None:
    sites = pd.read_csv(V3_ROOT / "sites.csv")
    table = pd.read_csv(V3_ROOT / "results" / "tables" / "persistence_minus_pmwm_ir_cell_effects.csv")
    merged = sites.merge(table, on="cell_id")
    limit = float(np.max(np.abs(merged.mse_reduction_z)))
    fig, axis = plt.subplots(figsize=(10.5, 4.8), constrained_layout=True)
    scatter = axis.scatter(merged.longitude, merged.latitude, c=merged.mse_reduction_z, cmap="vlag", vmin=-limit, vmax=limit, s=90, edgecolor="white", linewidth=0.7)
    axis.axhline(0, color=COLORS["light"], linewidth=1)
    axis.set(xlim=(-185, 185), ylim=(-82, 82), xlabel="Longitude", ylabel="Latitude")
    axis.set_title("Cell-wise skill relative to persistence", loc="left", weight="bold")
    fig.colorbar(scatter, ax=axis, label="Persistence MSE − PMWM-IR MSE")
    _save(fig, "12_spatial_robustness")


def figure_seed_effects() -> None:
    files = sorted((V3_ROOT / "results" / "tables").glob("*_minus_pmwm_ir_seed_effects.csv"))
    fig, axis = plt.subplots(figsize=(8.8, 4.8), constrained_layout=True)
    width = 0.22
    for index, path in enumerate(files):
        table = pd.read_csv(path)
        x = np.arange(len(table)) + (index - (len(files) - 1) / 2) * width
        axis.bar(x, table.mse_reduction_z, width=width, label=path.stem.replace("_seed_effects", "").replace("_", " "))
    axis.axhline(0, color=COLORS["red"], linestyle="--", linewidth=1)
    axis.set_xticks(np.arange(len(table)), table.seed.astype(str))
    axis.set(xlabel="Training seed", ylabel="Competitor MSE − PMWM-IR MSE")
    axis.set_title("Between-seed robustness", loc="left", weight="bold")
    axis.legend(frameon=False)
    _save(fig, "13_seed_robustness")


def figure_memory_ablation() -> None:
    table = pd.read_csv(V3_ROOT / "results" / "tables" / "leaderboard.csv")
    names = ["Persistence-residual iTransformer", "PMWM-IR reservoir", "PMWM-IR uniform", "PMWM-IR event-aware"]
    subset = table.set_index("model").loc[names].reset_index()
    fig, axis = plt.subplots(figsize=(8.2, 4.3), constrained_layout=True)
    colors = [COLORS["gray"], COLORS["blue"], COLORS["gold"], COLORS["teal"]]
    axis.bar(subset.model, subset.rmse_z_mean, yerr=subset.rmse_z_sd.fillna(0), color=colors, capsize=3)
    axis.tick_params(axis="x", rotation=18)
    axis.set_ylabel("Fresh-confirmation RMSE")
    axis.set_title("Matched-capacity memory ablation", loc="left", weight="bold")
    _save(fig, "14_memory_capacity_ablation")


def figure_physical_skill() -> None:
    table = pd.read_csv(V3_ROOT / "results" / "tables" / "physical_metrics.csv")
    averaged = table.groupby(["model", "target", "horizon"], as_index=False).rmse.mean()
    pmwm = averaged[averaged.model == "PMWM-IR event-aware"].set_index(["target", "horizon"]).rmse
    persistence = averaged[averaged.model == "Persistence"].set_index(["target", "horizon"]).rmse
    skill = (100 * (persistence - pmwm) / persistence).rename("skill").reset_index()
    matrix = skill.pivot(index="target", columns="horizon", values="skill")
    matrix = matrix[[label for label in ["6 h", "24 h", "72 h", "168 h"] if label in matrix.columns]]
    fig, axis = plt.subplots(figsize=(7.3, 4.4), constrained_layout=True)
    sns.heatmap(matrix, center=0, cmap="vlag", annot=True, fmt=".1f", linewidths=0.5, ax=axis, cbar_kws={"label": "RMSE skill vs persistence (%)"})
    axis.set_title("Physical-unit skill relative to persistence", loc="left", weight="bold")
    axis.set(xlabel="Horizon", ylabel="Target")
    _save(fig, "15_physical_unit_skill")


def figure_calibration() -> None:
    table = pd.read_csv(V3_ROOT / "results" / "tables" / "calibration.csv")
    summary = table.groupby("nominal").empirical.agg(["mean", "std"]).reset_index()
    fig, axis = plt.subplots(figsize=(5.7, 5.0), constrained_layout=True)
    axis.plot([0, 1], [0, 1], linestyle="--", color=COLORS["gray"], label="ideal")
    axis.errorbar(summary.nominal, summary["mean"], yerr=summary["std"], marker="o", color=COLORS["teal"], capsize=3, label="PMWM-IR")
    axis.set(xlim=(0.45, 1.0), ylim=(0.45, 1.0), xlabel="Nominal coverage", ylabel="Empirical coverage")
    axis.set_title("Predictive interval calibration", loc="left", weight="bold")
    axis.legend(frameon=False)
    _save(fig, "16_calibration_reliability")


def figure_extremes() -> None:
    table = pd.read_csv(V3_ROOT / "results" / "tables" / "extreme_events.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0), constrained_layout=True)
    metrics = table[["auroc", "auprc"]].melt(var_name="metric", value_name="score")
    sns.barplot(data=metrics, x="metric", y="score", errorbar="sd", color=COLORS["teal"], ax=axes[0])
    axes[0].axhline(table.prevalence.mean(), color=COLORS["gray"], linestyle="--", label="event prevalence")
    axes[0].set(title="Extreme-event ranking", xlabel="", ylabel="Score", ylim=(0, 1))
    axes[0].legend(frameon=False)
    rmse = table[["extreme_rmse_z", "non_extreme_rmse_z"]].melt(var_name="subset", value_name="rmse")
    sns.barplot(data=rmse, x="subset", y="rmse", errorbar="sd", color=COLORS["orange"], ax=axes[1])
    axes[1].tick_params(axis="x", rotation=15)
    axes[1].set(title="Conditional forecast error", xlabel="", ylabel="RMSE")
    _save(fig, "17_extreme_event_diagnostics")


def figure_causal_online() -> None:
    table = pd.read_csv(V3_ROOT / "results" / "tables" / "causal_online_metrics.csv")
    audits = []
    for path in sorted((V3_ROOT / "results" / "tables").glob("causal_update_audit_seed*.csv")):
        frame = pd.read_csv(path)
        frame["seed"] = path.stem.split("seed")[-1]
        frame["update_index"] = np.arange(len(frame))
        audits.append(frame)
    audit = pd.concat(audits, ignore_index=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), constrained_layout=True)
    names = ["Static PMWM-IR", "Causal online PMWM-IR"]
    paired = table.pivot(index="seed", columns="model", values="rmse_z")[names]
    x = np.arange(2)
    for row in paired.itertuples(index=False):
        axes[0].plot(x, row, color=COLORS["light"], linewidth=1, alpha=0.9, zorder=1)
        axes[0].scatter(x, row, color=[COLORS["gray"], COLORS["teal"]], s=25, alpha=0.9, zorder=2)
    means = paired.mean().to_numpy()
    stds = paired.std().to_numpy()
    axes[0].errorbar(x, means, yerr=stds, fmt="none", ecolor=COLORS["navy"], capsize=5, linewidth=2, zorder=3)
    axes[0].scatter(x, means, color=COLORS["navy"], marker="D", s=48, label="mean ± seed SD", zorder=4)
    degradation = 100 * (means[1] / means[0] - 1)
    pad = max(0.002, 0.25 * (paired.to_numpy().max() - paired.to_numpy().min()))
    axes[0].set_ylim(paired.to_numpy().min() - pad, paired.to_numpy().max() + pad)
    axes[0].set_xticks(x, ["Static", "Causal online"])
    axes[0].set(title="Delayed prequential performance", xlabel="", ylabel="Standardized RMSE")
    axes[0].text(
        0.03,
        0.96,
        f"Online RMSE is higher in {(paired.iloc[:, 1] > paired.iloc[:, 0]).sum()}/{len(paired)} seeds ({degradation:+.2f}%)",
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        size=8,
    )
    axes[0].legend(frameon=False, loc="lower right")
    for _, group in audit.groupby("seed"):
        axes[1].plot(group.update_index, group.inserted.cumsum(), alpha=0.65)
    axes[1].set(
        title="Insertions only after target maturity",
        xlabel="Daily update index",
        ylabel="Cumulative inserted slots",
    )
    axes[1].text(
        0.03,
        0.96,
        f"5 seeds  ·  capacity={int(audit.capacity.max()):,}  ·  minimum causal margin={int(audit.causal_margin_steps.min())}",
        transform=axes[1].transAxes,
        ha="left",
        va="top",
        size=8,
        bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none", "pad": 3},
    )
    _save(fig, "18_causal_online_learning")


def _efficiency_table(config: dict[str, Any]) -> pd.DataFrame:
    rows = []
    seed = int(config["model"]["seeds"][0])
    files = {
        "Persistence-residual iTransformer": Q1_ROOT / "checkpoints" / f"development_itransformer_residual_seed{seed}.pt",
        "GRU backbone": Q1_ROOT / "checkpoints" / f"backbone_seed{seed}.pt",
        "DLinear": Q1_ROOT / "checkpoints" / f"dlinear_seed{seed}.pt",
        "TCN": Q1_ROOT / "checkpoints" / f"tcn_seed{seed}.pt",
        "PatchTST-style": Q1_ROOT / "checkpoints" / f"patchtst_seed{seed}.pt",
        "iTransformer-style": Q1_ROOT / "checkpoints" / f"itransformer_seed{seed}.pt",
    }
    leaderboard = pd.read_csv(V3_ROOT / "results" / "tables" / "leaderboard.csv").set_index("model")
    for name, path in files.items():
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        parameter_count = checkpoint.get("parameter_count")
        if parameter_count is None:
            parameter_count = int(sum(value.numel() for value in checkpoint["state_dict"].values()))
        label = name
        leaderboard_label = {
            "DLinear": "dlinear",
            "TCN": "tcn",
            "PatchTST-style": "patchtst",
            "iTransformer-style": "itransformer",
        }.get(name, name)
        if leaderboard_label not in leaderboard.index:
            continue
        rows.append(
            {
                "model": label,
                "parameters": parameter_count,
                "storage_mb": path.stat().st_size / 1e6,
                "rmse_z": float(leaderboard.loc[leaderboard_label, "rmse_z_mean"]),
            }
        )
    memory_path = Q1_ROOT / "artifacts" / f"development_itransformer_memory_full_seed{seed}.npz"
    base = next(row for row in rows if row["model"] == "Persistence-residual iTransformer")
    rows.append(
        {
            "model": "PMWM-IR event-aware",
            "parameters": base["parameters"],
            "storage_mb": base["storage_mb"] + memory_path.stat().st_size / 1e6,
            "rmse_z": float(leaderboard.loc["PMWM-IR event-aware", "rmse_z_mean"]),
        }
    )
    table = pd.DataFrame(rows)
    table.to_csv(V3_ROOT / "results" / "tables" / "efficiency.csv", index=False)
    return table


def figure_efficiency() -> None:
    table = _efficiency_table(load_v3_config())
    fig, axis = plt.subplots(figsize=(7.3, 5.0), constrained_layout=True)
    for row in table.itertuples(index=False):
        color = COLORS["teal"] if row.model == "PMWM-IR event-aware" else COLORS["gray"]
        axis.scatter(row.storage_mb, row.rmse_z, s=65, color=color)
        axis.annotate(row.model, (row.storage_mb, row.rmse_z), xytext=(4, 4), textcoords="offset points", size=7)
    axis.set_xscale("log")
    axis.set(xlabel="Checkpoint + memory storage (MB, log scale)", ylabel="Fresh-confirmation RMSE")
    axis.set_title("Accuracy–storage trade-off", loc="left", weight="bold")
    _save(fig, "19_efficiency_tradeoff")


def figure_case_study() -> None:
    config = load_v3_config()
    seeds = [int(value) for value in config["model"]["seeds"]]
    predictions = [_load_npz(V3_ROOT / "artifacts" / f"predictions_pmwm_r_seed{seed}.npz") for seed in seeds]
    reference = predictions[0]
    unique_origins = np.unique(reference["origin"])
    unique_sites = np.unique(reference["site"])
    shape = (len(unique_origins), len(unique_sites), 4, 4)
    actual = reference["target_z"].reshape(shape)
    pmwm = np.mean([item["full_mean"] for item in predictions], axis=0).reshape(shape)
    persistence = reference["baseline"].reshape(shape)
    horizon_index, target_index = 3, 0
    location = np.unravel_index(np.argmax(np.abs(actual[:, :, horizon_index, target_index])), actual[:, :, horizon_index, target_index].shape)
    origin_index, site_index = location
    start = max(0, origin_index - 60)
    end = min(len(unique_origins), origin_index + 61)
    store = load_v3_features()
    dates = pd.DatetimeIndex(store.time[unique_origins[start:end] + int(config["model"]["horizons"][horizon_index])])
    fig, axis = plt.subplots(figsize=(10.4, 4.2), constrained_layout=True)
    axis.plot(dates, actual[start:end, site_index, horizon_index, target_index], color=COLORS["navy"], linewidth=1.5, label="ERA5 target")
    axis.plot(dates, persistence[start:end, site_index, horizon_index, target_index], color=COLORS["gray"], linewidth=1.0, label="Persistence")
    axis.plot(dates, pmwm[start:end, site_index, horizon_index, target_index], color=COLORS["teal"], linewidth=1.4, label="PMWM-IR")
    axis.axvline(dates[origin_index - start], color=COLORS["orange"], linestyle="--", linewidth=1)
    axis.set(ylabel="Standardized temperature anomaly", xlabel="Target date")
    cell_id = str(store.cell_id[site_index])
    site = pd.read_csv(V3_ROOT / "sites.csv").set_index("cell_id").loc[cell_id]
    axis.set_title(
        f"168-hour extreme case at {cell_id} ({site.latitude:.1f}°, {site.longitude:.1f}°)",
        loc="left",
        weight="bold",
    )
    axis.annotate(
        "largest |target anomaly|",
        xy=(dates[origin_index - start], actual[origin_index, site_index, horizon_index, target_index]),
        xytext=(8, 8),
        textcoords="offset points",
        size=8,
        color=COLORS["orange"],
    )
    axis.legend(frameon=False, ncol=3)
    _save(fig, "20_extreme_case_study")


def generate_q1_figures() -> list[Path]:
    verify_v3_lock()
    _style()
    functions = [
        figure_protocol_timeline,
        figure_spatial_design,
        figure_architecture,
        figure_memory_design,
        figure_v2_gate,
        figure_v2_effects,
        figure_development_selection,
        figure_v3_leaderboard,
        figure_bootstrap,
        figure_primary_forest,
        figure_effect_heatmaps,
        figure_spatial_effects,
        figure_seed_effects,
        figure_memory_ablation,
        figure_physical_skill,
        figure_calibration,
        figure_extremes,
        figure_causal_online,
        figure_efficiency,
        figure_case_study,
    ]
    for function in functions:
        print(f"figure: {function.__name__}", flush=True)
        function()
    png = sorted((V3_ROOT / "figures" / "png").glob("*.png"))
    rows = []
    for raster in png:
        vector = V3_ROOT / "figures" / "pdf" / f"{raster.stem}.pdf"
        rows.append(
            {
                "figure": raster.stem,
                "png_bytes": raster.stat().st_size,
                "png_sha256": sha256_file(raster),
                "pdf_bytes": vector.stat().st_size,
                "pdf_sha256": sha256_file(vector),
            }
        )
    pd.DataFrame(rows).to_csv(
        V3_ROOT / "results" / "tables" / "figure_manifest.csv", index=False
    )
    return png

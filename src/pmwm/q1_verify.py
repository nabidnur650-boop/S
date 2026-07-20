from __future__ import annotations

import hashlib
import json
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import nbformat
import numpy as np
import pandas as pd
import torch
from PIL import Image

from .q1_common import Q1_ROOT, verify_protocol_lock
from .q1_v3 import V3_ROOT, _load_npz, load_v3_config, load_v3_features, make_v3_dataset, verify_v3_lock


def _record(rows: list[dict[str, Any]], check: str, passed: bool, detail: Any) -> None:
    rows.append({"check": check, "passed": bool(passed), "detail": str(detail)})


def verify_q1_release() -> Path:
    rows: list[dict[str, Any]] = []
    v2_lock = verify_protocol_lock()
    v3_lock = verify_v3_lock()
    _record(rows, "v2 protocol hash", len(v2_lock["combined_sha256"]) == 64, v2_lock["combined_sha256"])
    _record(rows, "v3 protocol hash", len(v3_lock["combined_sha256"]) == 64, v3_lock["combined_sha256"])

    v2_sites = pd.read_csv(Q1_ROOT / "sites.csv")
    v3_sites = pd.read_csv(V3_ROOT / "sites.csv")
    v2_pairs = set(map(tuple, v2_sites[["latitude_index", "longitude_index"]].to_numpy()))
    v3_pairs = set(map(tuple, v3_sites[["latitude_index", "longitude_index"]].to_numpy()))
    _record(rows, "fresh cells are unique", len(v3_pairs) == 32, len(v3_pairs))
    _record(rows, "fresh cells exclude v2", not (v2_pairs & v3_pairs), len(v2_pairs & v3_pairs))

    config = load_v3_config()
    store = load_v3_features()
    dataset = make_v3_dataset(config, store)
    dates = pd.DatetimeIndex(store.time)
    context = int(config["model"]["context_steps"])
    max_horizon = max(int(value) for value in config["model"]["horizons"])
    context_inside = (dates.year[dataset.origins - context + 1] >= 2017) & (dates.year[dataset.origins] <= 2022)
    target_inside = (dates.year[dataset.origins + max_horizon] >= 2017) & (
        dates.year[dataset.origins + max_horizon] <= 2022
    )
    _record(rows, "strict context boundary", context_inside.all(), f"{context_inside.sum()}/{len(context_inside)}")
    _record(rows, "strict target boundary", target_inside.all(), f"{target_inside.sum()}/{len(target_inside)}")
    feature_manifest = json.loads((V3_ROOT / "artifacts" / "feature_manifest.json").read_text())
    _record(
        rows,
        "confirmation labels excluded from normalization",
        feature_manifest["confirmation_years_used_for_statistics"] == 0,
        feature_manifest["fit_years"],
    )

    seeds = [int(value) for value in config["model"]["seeds"]]
    reference = None
    for seed in seeds:
        path = V3_ROOT / "artifacts" / f"predictions_pmwm_r_seed{seed}.npz"
        exists = path.exists()
        _record(rows, f"PMWM-IR prediction seed {seed}", exists, path.name)
        if not exists:
            continue
        prediction = _load_npz(path)
        finite = all(np.isfinite(value).all() for key, value in prediction.items() if key not in {"origin", "site"})
        _record(rows, f"finite prediction seed {seed}", finite, len(prediction["origin"]))
        if reference is None:
            reference = prediction
        else:
            aligned = np.array_equal(reference["origin"], prediction["origin"]) and np.array_equal(
                reference["site"], prediction["site"]
            )
            _record(rows, f"aligned prediction seed {seed}", aligned, len(prediction["origin"]))

    baseline_seeds = seeds[: int(config["evaluation"]["minimum_neural_baseline_seeds"])]
    for model in ("dlinear", "tcn", "patchtst", "itransformer"):
        complete = all((V3_ROOT / "artifacts" / f"predictions_{model}_seed{seed}.npz").exists() for seed in baseline_seeds)
        _record(rows, f"baseline complete: {model}", complete, baseline_seeds)

    summary = json.loads((V3_ROOT / "results" / "summary.json").read_text())
    leaderboard = pd.read_csv(V3_ROOT / "results" / "tables" / "leaderboard.csv")
    pmwm_value = float(leaderboard.loc[leaderboard.model == "PMWM-IR event-aware", "rmse_z_mean"].iloc[0])
    _record(rows, "summary RMSE matches table", np.isclose(pmwm_value, summary["pmwm_rmse_z_mean"]), pmwm_value)
    for name, value in summary["contrasts"].items():
        _record(rows, f"positive bootstrap lower bound: {name}", value["bootstrap_ci_95"][0] > 0, value["bootstrap_ci_95"])
        _record(rows, f"majority seeds: {name}", value["positive_seed_fraction"] > 0.5, value["positive_seed_fraction"])
        _record(rows, f"majority cells: {name}", value["positive_cell_fraction"] > 0.5, value["positive_cell_fraction"])

    sensitivity_path = V3_ROOT / "results" / "tables" / "block_length_sensitivity.csv"
    sensitivity_ok = False
    sensitivity_detail: Any = "missing"
    if sensitivity_path.exists():
        sensitivity = pd.read_csv(sensitivity_path)
        sensitivity_ok = (
            set(sensitivity.block_days) == {7, 14, 28, 56, 84}
            and set(sensitivity.contrast) == set(summary["contrasts"])
            and len(sensitivity) == 5 * len(summary["contrasts"])
            and sensitivity.loc[sensitivity.prespecified_primary, "block_days"].eq(28).all()
            and int(sensitivity.prespecified_primary.sum()) == len(summary["contrasts"])
            and np.isfinite(sensitivity[["mse_reduction_z", "ci_lower", "ci_upper"]]).all().all()
        )
        sensitivity_detail = f"rows={len(sensitivity)}, blocks={sorted(sensitivity.block_days.unique())}"
    _record(rows, "post-confirmation block sensitivity complete", sensitivity_ok, sensitivity_detail)

    audits = sorted((V3_ROOT / "results" / "tables").glob("causal_update_audit_seed*.csv"))
    _record(rows, "causal audits for all seeds", len(audits) == len(seeds), len(audits))
    for path in audits:
        audit = pd.read_csv(path)
        _record(rows, f"nonnegative causal margin: {path.stem}", (audit.causal_margin_steps >= 0).all(), audit.causal_margin_steps.min())
        _record(rows, f"bounded capacity: {path.stem}", (audit.capacity == int(config["memory"]["capacity"])).all(), audit.capacity.unique())

    png = sorted((V3_ROOT / "figures" / "png").glob("*.png"))
    pdf = sorted((V3_ROOT / "figures" / "pdf").glob("*.pdf"))
    _record(rows, "at least 15 raster figures", len(png) >= 15, len(png))
    _record(rows, "matched raster/vector figures", {path.stem for path in png} == {path.stem for path in pdf}, f"{len(png)}/{len(pdf)}")
    figure_manifest = pd.read_csv(V3_ROOT / "results" / "tables" / "figure_manifest.csv")
    manifest_ok = len(figure_manifest) == len(png) == len(pdf)
    for entry in figure_manifest.itertuples(index=False):
        png_path = V3_ROOT / "figures" / "png" / f"{entry.figure}.png"
        pdf_path = V3_ROOT / "figures" / "pdf" / f"{entry.figure}.pdf"
        manifest_ok = manifest_ok and (
            png_path.exists()
            and pdf_path.exists()
            and png_path.stat().st_size == int(entry.png_bytes)
            and pdf_path.stat().st_size == int(entry.pdf_bytes)
            and hashlib.sha256(png_path.read_bytes()).hexdigest() == entry.png_sha256
            and hashlib.sha256(pdf_path.read_bytes()).hexdigest() == entry.pdf_sha256
        )
    _record(rows, "figure manifest sizes and hashes", manifest_ok, len(figure_manifest))
    for path in png:
        with Image.open(path) as image:
            _record(rows, f"publication raster: {path.name}", min(image.size) >= 900, image.size)
    for path in pdf:
        _record(rows, f"nonempty vector figure: {path.name}", path.stat().st_size > 5000, path.stat().st_size)

    notebooks = sorted((V3_ROOT / "notebooks").glob("*.ipynb"))
    _record(rows, "four executed notebooks", len(notebooks) == 4, len(notebooks))
    for path in notebooks:
        notebook = nbformat.read(path, as_version=4)
        code = [cell for cell in notebook.cells if cell.cell_type == "code"]
        executed = all(cell.execution_count is not None for cell in code)
        errors = [output for cell in code for output in cell.get("outputs", []) if output.output_type == "error"]
        _record(rows, f"executed notebook: {path.name}", executed and not errors, f"cells={len(code)}, errors={len(errors)}")

    manuscript_files = [
        V3_ROOT / "manuscript" / "main.md",
        V3_ROOT / "manuscript" / "supplement.md",
        V3_ROOT / "manuscript" / "references.bib",
        V3_ROOT / "manuscript" / "reporting_checklist.md",
    ]
    for path in manuscript_files:
        _record(rows, f"manuscript artifact: {path.name}", path.exists() and path.stat().st_size > 500, path)

    repository_files = [
        Q1_ROOT.parent / ".gitignore",
        Q1_ROOT.parent / "pyproject.toml",
        Q1_ROOT.parent / "LICENSE",
        Q1_ROOT.parent / "CITATION.cff",
        Q1_ROOT.parent / "CONTRIBUTING.md",
        Q1_ROOT.parent / "SECURITY.md",
        Q1_ROOT.parent / ".github" / "workflows" / "ci.yml",
    ]
    for path in repository_files:
        _record(rows, f"GitHub artifact: {path.name}", path.exists(), path.relative_to(Q1_ROOT.parent))

    table = pd.DataFrame(rows)
    table_path = V3_ROOT / "results" / "tables" / "verification_checks.csv"
    table.to_csv(table_path, index=False)
    failed = table[~table.passed]
    scientific_pass = bool(summary["primary_hypothesis_supported"])
    artifact_failures = failed[~failed.check.str.startswith(("positive bootstrap", "majority seeds", "majority cells"))]
    report = {
        "status": "PASS" if failed.empty else "FAIL",
        "scientific_gate_passed": scientific_pass,
        "artifact_gate_passed": artifact_failures.empty,
        "checks": len(table),
        "passed": int(table.passed.sum()),
        "failed": int((~table.passed).sum()),
        "failed_checks": failed.check.tolist(),
        "q1_readiness": bool(scientific_pass and failed.empty),
        "protocol_hash": v3_lock["combined_sha256"],
    }
    output = V3_ROOT / "results" / "verification_report.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    packages = {}
    for name in (
        "torch",
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "matplotlib",
        "xarray",
        "dask",
        "zarr",
    ):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "not installed"
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "torch_cuda_available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "packages": packages,
        "protocol_hash": v3_lock["combined_sha256"],
    }
    (V3_ROOT / "results" / "environment.json").write_text(
        json.dumps(environment, indent=2) + "\n", encoding="utf-8"
    )
    return output

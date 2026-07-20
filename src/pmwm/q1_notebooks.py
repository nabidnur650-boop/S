from __future__ import annotations

from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

from .q1_v3 import V3_ROOT, verify_v3_lock


def _header(title: str, purpose: str) -> list[nbf.NotebookNode]:
    return [
        nbf.v4.new_markdown_cell(
            f"# {title}\n\n{purpose}\n\n"
            "This notebook is generated from machine-readable artifacts. Numerical conclusions are not typed into code cells."
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "import json, sys\n"
            "import numpy as np\n"
            "import pandas as pd\n"
            "ROOT = Path.cwd().resolve().parents[2]\n"
            "sys.path.insert(0, str(ROOT / 'src'))\n"
            "pd.set_option('display.max_columns', 40)\n"
            "ROOT"
        ),
    ]


def _notebooks() -> dict[str, nbf.NotebookNode]:
    notebooks: dict[str, nbf.NotebookNode] = {}

    cells = _header(
        "01 — Protocol and leakage audit",
        "Verify immutable protocol hashes, disjoint spatial cells, and strict context/target boundaries.",
    )
    cells.extend(
        [
            nbf.v4.new_code_cell(
                "from pmwm.q1_common import verify_protocol_lock\n"
                "from pmwm.q1_v3 import verify_v3_lock\n"
                "{'v2': verify_protocol_lock()['combined_sha256'], 'v3': verify_v3_lock()['combined_sha256']}"
            ),
            nbf.v4.new_code_cell(
                "v2 = pd.read_csv(ROOT/'q1/sites.csv')\n"
                "v3 = pd.read_csv(ROOT/'q1/v3/sites.csv')\n"
                "pairs2 = set(map(tuple, v2[['latitude_index','longitude_index']].to_numpy()))\n"
                "pairs3 = set(map(tuple, v3[['latitude_index','longitude_index']].to_numpy()))\n"
                "{'v2_counts': v2.partition.value_counts().to_dict(), 'v3_cells': len(v3), 'overlap': len(pairs2 & pairs3)}"
            ),
            nbf.v4.new_code_cell(
                "from pmwm.q1_v3 import load_v3_config, load_v3_features, make_v3_dataset\n"
                "cfg = load_v3_config(); store = load_v3_features(); ds = make_v3_dataset(cfg, store)\n"
                "dates = pd.DatetimeIndex(store.time)\n"
                "context = cfg['model']['context_steps']; max_h = max(cfg['model']['horizons'])\n"
                "checks = {\n"
                " 'all_contexts_in_2017_2022': bool(((dates.year[ds.origins-context+1] >= 2017) & (dates.year[ds.origins] <= 2022)).all()),\n"
                " 'all_targets_in_2017_2022': bool(((dates.year[ds.origins+max_h] >= 2017) & (dates.year[ds.origins+max_h] <= 2022)).all()),\n"
                " 'samples': len(ds), 'cells': len(np.unique(ds.sites))}\n"
                "checks"
            ),
            nbf.v4.new_code_cell(
                "manifest = json.loads((ROOT/'q1/v3/artifacts/feature_manifest.json').read_text())\n"
                "manifest"
            ),
        ]
    )
    notebooks["01_protocol_and_leakage_audit.ipynb"] = nbf.v4.new_notebook(cells=cells)

    cells = _header(
        "02 — Falsification gate and development selection",
        "Preserve the v2 persistence failure and document the development-only candidate selection.",
    )
    cells.extend(
        [
            nbf.v4.new_code_cell(
                "v2_summary = json.loads((ROOT/'q1/results/summary.json').read_text())\n"
                "{k:v2_summary[k] for k in ['backbone_rmse_mean','pmwm_rmse_mean','primary_hypothesis_supported']}"
            ),
            nbf.v4.new_code_cell(
                "v2_leaderboard = pd.read_csv(ROOT/'q1/results/tables/leaderboard.csv')\n"
                "v2_leaderboard.sort_values('rmse_z_mean')"
            ),
            nbf.v4.new_code_cell(
                "selection = pd.read_csv(ROOT/'q1/v3/DEVELOPMENT_SELECTION.csv').sort_values('rmse_z')\n"
                "selection"
            ),
            nbf.v4.new_code_cell(
                "lock = json.loads((ROOT/'q1/v3/LOCKED_PROTOCOL.json').read_text())\n"
                "{'selected_model': lock['selected_model'], 'strongest_baseline': lock['strongest_development_baseline'], 'hash': lock['combined_sha256']}"
            ),
        ]
    )
    notebooks["02_falsification_and_selection.ipynb"] = nbf.v4.new_notebook(cells=cells)

    cells = _header(
        "03 — Fresh confirmatory evaluation",
        "Reconstruct the main leaderboard, prespecified bootstrap contrasts, physical metrics, calibration, and extremes.",
    )
    cells.extend(
        [
            nbf.v4.new_code_cell(
                "summary = json.loads((ROOT/'q1/v3/results/summary.json').read_text())\n"
                "summary"
            ),
            nbf.v4.new_code_cell(
                "pd.read_csv(ROOT/'q1/v3/results/tables/leaderboard.csv').sort_values('rmse_z_mean')"
            ),
            nbf.v4.new_code_cell(
                "effects = pd.read_csv(ROOT/'q1/v3/results/tables/confirmatory_effects.csv')\n"
                "effects.groupby('contrast').agg(mean_effect=('mse_reduction_z','mean'), significant_cells=('significant_positive_bh','sum'))"
            ),
            nbf.v4.new_code_cell(
                "sensitivity = pd.read_csv(ROOT/'q1/v3/results/tables/block_length_sensitivity.csv')\n"
                "sensitivity[['contrast','block_days','prespecified_primary','mse_reduction_z','ci_lower','ci_upper']]"
            ),
            nbf.v4.new_code_cell(
                "physical = pd.read_csv(ROOT/'q1/v3/results/tables/physical_metrics.csv')\n"
                "physical[physical.model.isin(['PMWM-IR event-aware','Persistence'])].groupby(['model','target','horizon']).rmse.mean().unstack('horizon')"
            ),
            nbf.v4.new_code_cell(
                "calibration = pd.read_csv(ROOT/'q1/v3/results/tables/calibration.csv')\n"
                "extremes = pd.read_csv(ROOT/'q1/v3/results/tables/extreme_events.csv')\n"
                "{'calibration': calibration.groupby('nominal').empirical.mean().to_dict(), 'extremes': extremes.mean(numeric_only=True).to_dict()}"
            ),
        ]
    )
    notebooks["03_fresh_confirmatory_results.ipynb"] = nbf.v4.new_notebook(cells=cells)

    cells = _header(
        "04 — Causal online audit and publication inventory",
        "Check delayed memory updates and enumerate paired raster/vector figures and report artifacts.",
    )
    cells.extend(
        [
            nbf.v4.new_code_cell(
                "online = pd.read_csv(ROOT/'q1/v3/results/tables/causal_online_metrics.csv')\n"
                "online.groupby('model').agg(rmse_mean=('rmse_z','mean'), rmse_sd=('rmse_z','std'))"
            ),
            nbf.v4.new_code_cell(
                "audits = [pd.read_csv(p) for p in sorted((ROOT/'q1/v3/results/tables').glob('causal_update_audit_seed*.csv'))]\n"
                "{'seeds': len(audits), 'minimum_margin': min(int(x.causal_margin_steps.min()) for x in audits), 'capacity_constant': all(x.capacity.nunique()==1 for x in audits)}"
            ),
            nbf.v4.new_code_cell(
                "png = sorted((ROOT/'q1/v3/figures/png').glob('*.png'))\n"
                "pdf = sorted((ROOT/'q1/v3/figures/pdf').glob('*.pdf'))\n"
                "{'png_count': len(png), 'pdf_count': len(pdf), 'matched_stems': {p.stem for p in png} == {p.stem for p in pdf}}"
            ),
            nbf.v4.new_code_cell(
                "verification = json.loads((ROOT/'q1/v3/results/verification_report.json').read_text()) if (ROOT/'q1/v3/results/verification_report.json').exists() else {'status':'verification runs after notebook execution'}\n"
                "verification"
            ),
        ]
    )
    notebooks["04_causal_online_and_inventory.ipynb"] = nbf.v4.new_notebook(cells=cells)
    return notebooks


def generate_q1_notebooks() -> list[Path]:
    verify_v3_lock()
    output_dir = V3_ROOT / "notebooks"
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, notebook in _notebooks().items():
        notebook.metadata.kernelspec = {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        }
        notebook.metadata.language_info = {"name": "python", "version": "3.12"}
        path = output_dir / name
        nbf.write(notebook, path)
        paths.append(path)
    return paths


def execute_q1_notebooks(timeout: int = 900) -> list[Path]:
    paths = generate_q1_notebooks()
    for path in paths:
        print(f"execute notebook: {path.name}", flush=True)
        notebook = nbf.read(path, as_version=4)
        client = NotebookClient(
            notebook,
            timeout=timeout,
            kernel_name="python3",
            resources={"metadata": {"path": str(V3_ROOT / "notebooks")}},
            allow_errors=False,
        )
        client.execute()
        nbf.write(notebook, path)
    return paths

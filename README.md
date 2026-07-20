# PMWM-IR on streamed ERA5

A leakage-audited study of whether bounded persistent retrieval memory improves a strong time-series forecaster. The repository preserves the pilot, an opened v2 confirmation that failed the persistence gate, and a separately locked v3 confirmation on 32 previously unused ERA5 cells.

> **Outcome:** the software and artifact release passes its audit, but the prespecified memory-superiority hypothesis does not. PMWM-IR beats raw persistence, yet it does not beat its matched residual iTransformer and is worse than the direct iTransformer-style baseline. This repository therefore supports a transparent falsification/negative-result paper—not a memory-accuracy or state-of-the-art claim.

![Fresh-confirmation leaderboard](q1/v3/figures/png/08_v3_fresh_leaderboard.png)

## Release status

| Gate | Status | Evidence |
|---|---|---|
| Artifact integrity | **PASS** | All non-scientific checks pass; 20 PNG/PDF pairs and 4 executed notebooks |
| Leakage controls | **PASS** | Frozen hashes, disjoint cells, strict context/target boundaries, causal update audit |
| Beats persistence | **PASS** | PMWM-IR RMSE 0.9174 vs persistence 1.1995 |
| Beats matched backbone | **FAIL** | MSE effect -0.0008; 95% block CI [-0.0038, 0.0017]; 1/5 seeds positive |
| Strict Q1 method-readiness | **NO** | The central superiority claim was not confirmed |

The machine-readable decision is in [`verification_report.json`](q1/v3/results/verification_report.json), and the publication assessment is in [`Q1_READINESS.md`](Q1_READINESS.md).

## Fresh-confirmation result

The v3 test contains 69,440 forecast origins across 32 global cells during 2017–2022. Local normalization uses only 1959–1994 history. The candidate was run with five seeds; modern learned baselines use three seeds. Primary uncertainty uses 5,000 paired 28-day moving-block bootstrap replicates.

| Model | Runs | Standardized RMSE |
|---|---:|---:|
| iTransformer-style | 3 | **0.8896** |
| PatchTST-style | 3 | 0.8899 |
| TCN | 3 | 0.8968 |
| GRU backbone | 5 | 0.9042 |
| PMWM-IR uniform memory | 5 | 0.9165 |
| Matched residual iTransformer | 5 | 0.9170 |
| PMWM-IR event-aware | 5 | 0.9174 |
| Persistence | 1 | 1.1995 |

The event-aware allocation also trails uniform memory, and delayed causal online replacement degrades RMSE by 1.13%. These null/negative outcomes are retained in the manuscript and figures.

## Repository map

```text
S/
├── src/pmwm/                 Installable implementation and CLI
├── tests/                    Fast unit and protocol-integrity tests
├── q1/                       Locked v2 study and preserved failure
│   └── v3/                   Fresh, final publication-decision layer
│       ├── manuscript/       Main paper, supplement, BibTeX, checklist
│       ├── notebooks/        Four executed audit/result notebooks
│       ├── figures/          20 matched 320-dpi PNG and vector PDF figures
│       ├── results/          Machine-readable tables and release audit
│       └── artifacts/        Local arrays/checkpoints; ignored by Git
├── notebooks/, figures/      Original pilot record
├── docs/                     Scope and repository documentation
└── .github/                  CI, issue forms, and pull-request template
```

Start with:

- [`q1/v3/manuscript/main.md`](q1/v3/manuscript/main.md) — negative-result manuscript
- [`q1/v3/RESULTS.md`](q1/v3/RESULTS.md) — concise final results
- [`q1/v3/PROTOCOL.md`](q1/v3/PROTOCOL.md) — immutable decision rules
- [`q1/v3/results/tables/leaderboard.csv`](q1/v3/results/tables/leaderboard.csv) — model comparison
- [`q1/v3/results/tables/figure_manifest.csv`](q1/v3/results/tables/figure_manifest.csv) — figure checksums
- [`DATA_CARD.md`](DATA_CARD.md) — provenance, transformations, and limitations

## Install and verify

Python 3.11 or newer is required.

```bash
python -m venv .venv
.venv/bin/pip install -e '.[dev]'
make check
```

Fast checks require no ERA5 download. To verify the immutable protocols:

```bash
PYTHONPATH=src .venv/bin/python -c \
  "from pmwm.q1_common import verify_protocol_lock; from pmwm.q1_v3 import verify_v3_lock; print(verify_protocol_lock()['combined_sha256']); print(verify_v3_lock()['combined_sha256'])"
```

For a local checkout containing the ignored large artifacts, run the 103-check release audit:

```bash
make release-audit
```

The audit exits normally and writes JSON/CSV evidence even when a scientific gate fails. A clean GitHub clone can regenerate the full v3 workflow with `make fresh`; it streams only selected WeatherBench2 Zarr chunks rather than downloading global ERA5. This is compute- and network-intensive.

## Scientific boundaries

- The experiment is pointwise multivariate forecasting on a coarse 64×32 reanalysis grid, not global numerical weather prediction.
- Fresh cells use their own 1959–1994 history for climatology and scaling. This is historical-normal adaptation, not strict zero-shot transfer.
- PatchTST-style and iTransformer-style are controlled in-repository implementations, not full reproductions of pretrained external checkpoints.
- The evidence covers ERA5 only and does not establish a general time-series foundation model.
- The frozen v2/v3 files must not be edited. A new model claim requires a new protocol and genuinely untouched data.

## Data provenance

The workflow streams the public WeatherBench2 ERA5 Zarr store at six-hour cadence and retains only selected cells and variables. Large source-derived arrays, predictions, and checkpoints are excluded from Git; small manifests, tables, figures, notebooks, and protocol hashes are release artifacts. See [`DATA_CARD.md`](DATA_CARD.md) for the exact source, variables, transformations, and license notes.

## Citation and metadata

Software citation metadata are provided in [`CITATION.cff`](CITATION.cff). Before publishing to GitHub or submitting a manuscript, replace the contributor placeholder with the real author list and add the final repository/DOI URLs; no identity or remote URL has been invented.

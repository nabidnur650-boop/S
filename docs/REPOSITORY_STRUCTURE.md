# Repository structure

```text
S/
├── src/pmwm/                  Reusable Python package
│   ├── q1_v3.py               Frozen-v3 orchestration and evaluation
│   ├── q1_reporting.py        Journal figure generation
│   ├── q1_notebooks.py        Notebook generation/execution
│   ├── q1_manuscript.py       Manuscript and supplement generation
│   └── q1_verify.py           103-check release audit
├── tests/                     Fast, data-independent tests
├── q1/
│   ├── PROTOCOL.md            Immutable v2 protocol
│   ├── LOCKED_PROTOCOL.json   v2 hash record
│   ├── results/               Preserved opened-v2 outcome
│   └── v3/
│       ├── PROTOCOL.md        Immutable fresh decision rules
│       ├── LOCKED_PROTOCOL.json
│       ├── DEVELOPMENT_SELECTION.csv
│       ├── sites.csv          32 fresh coordinates
│       ├── manuscript/        Main text, supplement, BibTeX, checklist
│       ├── notebooks/         Four executed audit notebooks
│       ├── figures/           20 matched PNG/PDF figures
│       ├── results/           Tables, summary, audit, environment
│       └── artifacts/         Large local outputs, ignored by Git
├── notebooks/, figures/       Original pilot provenance
├── artifacts/                 Pilot local arrays/checkpoints, ignored
├── docs/                      Scope and repository guidance
├── .github/                   CI, issue forms, PR template
├── pyproject.toml             Package and tool configuration
├── Makefile                   Reproduction and audit entry points
├── Q1_READINESS.md            Honest publication decision
└── README.md                  Main entry point
```

## Versioned scientific layers

The pilot is exploratory provenance. V2 is an opened confirmation whose persistence failure is retained. V3 is the final untouched decision layer. Locked inputs in v2 or v3 must never be edited under their existing protocol IDs.

## Git boundary

Git includes source, locks, manifests, small result tables, publication figures, executed notebooks, and manuscript text. It excludes raw/streamed arrays, prediction tensors, checkpoints, caches, logs, secrets, and local environments. The current unignored files are below GitHub's per-file size limit; ignored artifacts remain available in this local workspace for verification.

Before public release, add the actual author metadata and final GitHub/DOI URLs. They are deliberately not guessed in this repository.

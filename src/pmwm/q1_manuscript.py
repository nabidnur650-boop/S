from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .q1_common import Q1_ROOT
from .q1_v3 import V3_ROOT, load_v3_config, verify_v3_lock


def _format_ci(values: list[float]) -> str:
    return f"[{values[0]:.4f}, {values[1]:.4f}]"


def generate_manuscript() -> list[Path]:
    lock = verify_v3_lock()
    config = load_v3_config()
    v2_summary = json.loads((Q1_ROOT / "results" / "summary.json").read_text())
    v2_board = pd.read_csv(Q1_ROOT / "results" / "tables" / "leaderboard.csv").set_index("model")
    summary = json.loads((V3_ROOT / "results" / "summary.json").read_text())
    board = pd.read_csv(V3_ROOT / "results" / "tables" / "leaderboard.csv").set_index("model")
    selection = pd.read_csv(V3_ROOT / "DEVELOPMENT_SELECTION.csv").sort_values("rmse_z")
    effects = pd.read_csv(V3_ROOT / "results" / "tables" / "confirmatory_effects.csv")
    calibration = pd.read_csv(V3_ROOT / "results" / "tables" / "calibration.csv")
    extremes = pd.read_csv(V3_ROOT / "results" / "tables" / "extreme_events.csv")
    probabilistic = pd.read_csv(V3_ROOT / "results" / "tables" / "probabilistic_metrics.csv")
    online = pd.read_csv(V3_ROOT / "results" / "tables" / "causal_online_metrics.csv")
    sensitivity = pd.read_csv(V3_ROOT / "results" / "tables" / "block_length_sensitivity.csv")
    pmwm = float(board.loc["PMWM-IR event-aware", "rmse_z_mean"])
    persistence = float(board.loc["Persistence", "rmse_z_mean"])
    residual = float(board.loc["Persistence-residual iTransformer", "rmse_z_mean"])
    direct_itransformer = float(board.loc["itransformer", "rmse_z_mean"])
    strongest = str(summary["strongest_development_baseline"])
    improvement_persistence = 100 * (persistence - pmwm) / persistence
    improvement_residual = 100 * (residual - pmwm) / residual
    improvement_direct = 100 * (direct_itransformer - pmwm) / direct_itransformer
    supported = bool(summary["primary_hypothesis_supported"])
    outcome_sentence = (
        "All prespecified fresh-confirmation gates were satisfied."
        if supported
        else "The prespecified memory-superiority hypothesis was not supported; the result is reported as non-confirmatory."
    )
    contrast_lines = []
    for name, value in summary["contrasts"].items():
        contrast_lines.append(
            f"- `{name}`: MSE reduction {value['mse_reduction_z']:.4f}, 95% moving-block-bootstrap CI "
            f"{_format_ci(value['bootstrap_ci_95'])}, positive in {100*value['positive_seed_fraction']:.1f}% of seeds "
            f"and {100*value['positive_cell_fraction']:.1f}% of cells."
        )
    contrast_text = "\n".join(contrast_lines)
    significant = effects.groupby("contrast").significant_positive_bh.sum().astype(int).to_dict()
    coverage90 = float(calibration.query("nominal == 0.90").empirical.mean())
    extreme_auroc = float(extremes.auroc.mean())
    extreme_auprc = float(extremes.auprc.mean())
    mean_crps = float(probabilistic.gaussian_crps_z.mean())
    mean_nll = float(probabilistic.gaussian_nll_z.mean())
    static_online = online.groupby("model").rmse_z.mean().to_dict()
    online_change = 100 * (
        static_online["Static PMWM-IR"] - static_online["Causal online PMWM-IR"]
    ) / static_online["Static PMWM-IR"]
    selection_markdown = selection.to_markdown(index=False, floatfmt=".4f")
    leaderboard_markdown = board.reset_index().sort_values("rmse_z_mean").to_markdown(index=False, floatfmt=".4f")
    sensitivity_markdown = sensitivity.to_markdown(index=False, floatfmt=".4f")
    manuscript_dir = V3_ROOT / "manuscript"
    manuscript_dir.mkdir(parents=True, exist_ok=True)

    main = rf"""---
bibliography: references.bib
link-citations: true
---

# When persistent memory does not beat a strong forecaster: a leakage-audited ERA5 study

## Abstract

External memories are often evaluated against weak neural backbones or on test sets that also influence model design. We study whether a fixed-capacity event-aware memory adds reproducible forecast skill after a strong persistence floor is enforced. The proposed PMWM-IR system combines an inverted-variable Transformer, a persistence-residual head, and a 2,048-slot memory that consolidates regular and rare contexts separately. A pilot and an initially locked 32-cell confirmation were preserved as exploratory/opened evidence after the memory model improved its backbone but lost to persistence. We then selected one residual-memory candidate using development cells, froze a new protocol, and evaluated it once on 32 previously unused global ERA5 cells during 2017–2022. All fitting used 1959–1994, calibration used 1995–2004, and five seeds were evaluated with paired 28-day moving-block bootstrap intervals. PMWM-IR attained mean standardized RMSE {pmwm:.4f}, compared with {persistence:.4f} for persistence and {residual:.4f} for its matched residual iTransformer; the direct iTransformer-style baseline was best at {direct_itransformer:.4f}. The corresponding signed RMSE improvements for PMWM-IR were {improvement_persistence:+.2f}%, {improvement_residual:+.2f}%, and {improvement_direct:+.2f}%, respectively. {outcome_sentence} The nominal 90% interval achieved {100*coverage90:.2f}% empirical coverage, Gaussian CRPS was {mean_crps:.3f}, and extreme-event ranking reached mean AUROC {extreme_auroc:.3f}. The claim is deliberately bounded to historical-normal adaptation at unseen coarse-grid locations; it is not evidence for a general time-series foundation model or operational weather prediction.

## 1. Introduction

Finite context windows discard long records that may contain analog regimes and rare outcomes. Retrieval memory offers a computationally attractive alternative: encode a recent context, query a bounded bank of consolidated past cases, and combine a retrieved correction with a parametric forecast. The central scientific question, however, is not whether retrieval can improve a convenient backbone. It is whether the gain survives strong trivial and modern learned baselines, spatial and temporal shift, repeated seeds, and a genuinely untouched confirmation.

This work makes four bounded contributions. First, it defines PMWM-IR, a persistence-residual inverted-Transformer forecasting model with fixed-capacity event-aware consolidation. Second, it implements delayed prequential updates whose audit guarantees that a memory is inserted only after the 168-hour target has matured. Third, it separates pilot, development, opened confirmation, and fresh confirmation, preserving the initial persistence failure rather than replacing it. Fourth, it reports five-seed, cell-wise, target-wise, horizon-wise, physical-unit, calibration, extreme-event, and efficiency evidence from a streamed WeatherBench2 ERA5 source.

We do not claim a foundation model, causal event graph, or global numerical weather prediction system. The experiment is pointwise multivariate forecasting at a deliberately coarse 64×32 reanalysis grid. This narrower claim is essential: it makes the statistical unit, leakage boundary, and operational information set explicit.

![Study protocol](../figures/png/01_protocol_timeline.png)

## 2. Related work

Long-horizon forecasting has moved from decomposition-based linear models [@zeng2023dlinear] to patch-token Transformers [@nie2023patchtst] and inverted-variable attention [@liu2024itransformer]. Pretrained time-series models such as TimesFM and Moirai broaden the training distribution and task interface [@das2024timesfm; @woo2024moirai], but this study does not reproduce their external pretraining budgets. Its controlled baselines instead isolate architecture and memory effects under one data and optimization protocol.

ERA5 provides a globally complete reanalysis record [@hersbach2020era5], while WeatherBench2 supplies standardized data products and evaluation conventions for data-driven weather modeling [@rasp2024weatherbench2]. PMWM-IR addresses a narrower evaluation problem: retrieved outcomes can produce optimistic evidence when they overlap the test interval, capacity can confound allocation ablations, and weak backbone-only comparisons can conceal a persistence failure. The contribution is therefore the locked evaluation and causal memory audit as much as the retrieval mechanism itself.

## 3. Data and protocol

We accessed the public WeatherBench2 ERA5 Zarr store [@rasp2024weatherbench2] at six-hour cadence and retained only selected grid cells and five single-level variables: 2 m temperature, mean sea-level pressure, 10 m u/v wind, and six-hour precipitation. Wind speed was derived from u/v components; precipitation was modeled after `log1p`. The source was streamed in eight-year blocks, so the global archive was never materialized locally.

The v2 design contained 64 fitting cells, 16 development cells, and 32 confirmation cells. Forecast origins were admitted only when the complete 56-step context and all targets at 6, 24, 72, and 168 hours remained in the declared split. The v2 protocol hash was `{v2_summary['protocol_hash']}`. Its primary memory-versus-backbone hypothesis passed, but persistence achieved RMSE {float(v2_board.loc['Persistence','rmse_z_mean']):.4f}, versus {float(v2_board.loc['PMWM full','rmse_z_mean']):.4f} for PMWM. We therefore classified v2 as a failed readiness gate, did not tune on its target values, and opened a development-only residual route.

For v3, the development winner was frozen before any value from 32 new cells was accessed. These cells exclude every pilot and v2 grid coordinate. The lock hash is `{lock['combined_sha256']}`. Per-cell Fourier climatology and robust scale use only 1959–1994 at the new location. Consequently, v3 evaluates transfer of model weights and memory to a new cell with historical normals; it is not strict zero-shot normalization.

![Spatial partitions](../figures/png/02_spatial_design.png)

## 4. Method

Let $x_t\in\mathbb{{R}}^5$ be the standardized context and $y_{{t,h}}\in\mathbb{{R}}^4$ a target at horizon $h$. A persistence forecast $p_{{t,h}}$ carries the current physical state through the future cell-specific seasonal transform. An inverted-variable Transformer predicts a correction $f_\theta(x_{{t-55:t}},s)$, giving

$$\hat y^{{base}}_{{t,h}} = p_{{t,h}} + f_\theta(x_{{t-55:t}},s).$$

The learned forecast representation is normalized into a retrieval key. Training cases are split by the 90th percentile of maximum absolute target anomaly. Mini-batch k-means consolidates 1,536 regular prototypes and 512 event prototypes. Every memory stores a key, target, base-model residual, within-cluster residual variance, event score, month, zone, source cell, origin, and count. Full, uniform, and reservoir memories all contain exactly 2,048 slots.

For query key $q$, cosine top-16 neighbors receive temperature-scaled weights. Their weighted residual $r(q)$ corrects the base forecast with validation-fitted horizon-target gate $g$:

$$\hat y^{{PMWM-IR}}_{{t,h}} = \hat y^{{base}}_{{t,h}} + g_h\,r_h(q).$$

Predictive variance combines base and retrieved residual variance and is conformally scaled on validation data. During online evaluation, the system forecasts first. Only when origin $t-h_{{max}}$ has fully matured may two cases—one high-event and one high-novelty—replace low-priority slots. The memory remains bounded, and every insertion records a causal margin.

![Architecture](../figures/png/03_pmwm_ir_architecture.png)

## 5. Comparators and evaluation

Comparators include persistence, seasonal climatology, Ridge-AR, DLinear, a dilated TCN, PatchTST-style and iTransformer-style encoders, the original probabilistic GRU, the matched persistence-residual iTransformer, uniform consolidated memory, reservoir memory, and full event-aware PMWM-IR. “Style” labels are used where the implementation captures the published architectural principle but is trained under this study's controlled budget rather than reproducing an external full-scale checkpoint.

The primary metric is equal-weight standardized MSE across four targets and four horizons. The fresh decision rule requires a strictly positive paired 95% interval for competitor MSE minus PMWM-IR MSE against persistence, the matched residual iTransformer, and the strongest non-memory development comparator. It also requires positive effects for majorities of seeds and cells. We use 5,000 paired moving-block replicates, 28-day blocks, spatial cell resampling, five PMWM seeds, and three seeds for modern neural baselines. Moving blocks retain local serial dependence [@kunsch1989bootstrap]. Sixteen target-horizon secondary tests per contrast are adjusted with the Benjamini–Hochberg procedure [@benjamini1995fdr].

## 6. Results

### 6.1 Development selection

The complete development-only selection was:

{selection_markdown}

Only the top candidate was eligible for the new lock. The strongest non-memory development comparator was {strongest}.

### 6.2 Fresh confirmation

{leaderboard_markdown}

{outcome_sentence}

The direct iTransformer-style baseline was best overall at RMSE {float(board.loc['itransformer', 'rmse_z_mean']):.4f}; PatchTST-style was statistically close at {float(board.loc['patchtst', 'rmse_z_mean']):.4f}. Event-aware memory was slightly worse than its matched residual backbone and also worse than uniform memory. Thus the fresh data reject both the memory-superiority claim and the development ranking, while supporting the narrower conclusion that learned forecasters beat raw persistence in this historical-normal setting.

{contrast_text}

The explicitly post-confirmation block-length sensitivity reached the same qualitative decision at 7, 14, 28, 56, and 84 days: every matched-backbone interval crossed zero and every persistence interval remained positive. The 28-day analysis alone remains the prespecified primary result.

BH-significant positive target-horizon effects by contrast were `{significant}` out of 16 tests each. Aggregate effects should therefore not be interpreted as uniform skill across every variable and lead time.

![Fresh leaderboard](../figures/png/08_v3_fresh_leaderboard.png)

![Primary effects](../figures/png/10_primary_effect_forest.png)

### 6.3 Ablations, calibration, and extremes

Matched capacity isolates allocation policy: event-aware, uniform, and reservoir memories store the same number of slots and use the same retrieval count. The physical-unit table reports RMSE, MAE, and bias for every target and horizon. The mean empirical coverage of nominal 90% intervals was {100*coverage90:.2f}%; mean Gaussian CRPS was {mean_crps:.3f} and Gaussian NLL was {mean_nll:.3f}. Extreme events, defined by maximum absolute target anomaly above {config['evaluation']['extreme_z_threshold']:.1f}, had mean AUROC {extreme_auroc:.3f} and AUPRC {extreme_auprc:.3f}.

![Memory ablation](../figures/png/14_memory_capacity_ablation.png)

![Calibration](../figures/png/16_calibration_reliability.png)

### 6.4 Continual update

All five delayed-prequential audits retained a 2,048-slot memory and nonnegative causal margins. Online updating changed aggregate RMSE by {online_change:+.2f}% relative to the static memory. A negative value is a degradation and is retained as such; bounded causal updating is a systems property, not automatic evidence of improved forecast skill.

![Causal online evaluation](../figures/png/18_causal_online_learning.png)

## 7. Discussion

The two-lock sequence is scientifically informative. The v2 result would have appeared favorable if evaluated only against its GRU backbone; the persistence gate reversed the substantive conclusion. A residual formulation then made persistence part of the model rather than an external afterthought, while fresh cells prevented that redesign from inheriting confirmatory status. This workflow separates algorithm development from evidence intended to support a publication claim.

The fresh result does not support event-aware memory as an accuracy contribution. Four of five seeds favored the matched backbone, the aggregate block interval crossed zero, uniform allocation was better than the event partition, and causal replacement degraded RMSE. These aligned failures suggest that retrieved residuals were unstable under the normalization and spatial shift rather than merely underpowered in one aggregate test.

Event-aware allocation is useful only if it improves both aggregate and rare-regime behavior at matched capacity. The horizon-target and cell maps reveal where this condition holds and where it does not. Similarly, good standardized skill does not establish operational weather value: physical errors at coarse cells, reanalysis uncertainty, missing orography, and absence of global spatial coupling remain material.

## 8. Limitations

1. ERA5 is reanalysis rather than direct observation, and the 64×32 conservative grid smooths coastlines, topography, and extremes.
2. Pointwise cells omit spatial fields and dynamical conservation; the system is not comparable to a global numerical or neural weather model.
3. New-cell climatology uses historical values from that cell. This is a realistic normals-assisted setting but not strict zero-shot spatial transfer.
4. The study covers one domain. It does not support claims about healthcare, finance, traffic, or a universal foundation model.
5. “PatchTST-style” and “iTransformer-style” are controlled implementations, not reproductions of every external training recipe or pretrained checkpoint.
6. Five method seeds and three baseline seeds quantify material stochastic variation but cannot eliminate it.
7. Moving-block interval width depends on the chosen 28-day dependence scale; the supplement reports an explicitly post-confirmation sensitivity analysis.
8. Online memory updates are heuristic and bounded. Their causal safety is audited, but their forecast benefit may be null or negative.

## 9. Reproducibility and data availability

The repository contains immutable protocol hashes, deterministic cell lists, streaming code, model and memory implementations, machine-readable tables, 20 paired PNG/PDF figures, four executed notebooks, tests, CI, and a verification report. Large source-derived arrays and checkpoints are excluded from Git and regenerated from manifests. WeatherBench2/ERA5 terms govern source data. Exact commands and artifact boundaries appear in the repository README and data card.

## 10. Conclusion

PMWM-IR tests a modest but consequential proposition: bounded retrieval memory should add skill only after persistence, strong learned comparators, fresh spatial-temporal confirmation, and causal update order are enforced. {outcome_sentence} Regardless of sign, the preserved v2 failure and the fresh v3 decision provide a more defensible foundation than a test-tuned backbone comparison.

## References

The bibliography is generated from `references.bib` using the citation keys embedded above.
"""
    main_path = manuscript_dir / "main.md"
    main_path.write_text(main, encoding="utf-8")

    supplement = f"""# Supplementary material

## S1. Immutable protocol chain

- v2 protocol: `{v2_summary['protocol_id']}`
- v2 SHA-256: `{v2_summary['protocol_hash']}`
- v3 protocol: `{summary['protocol_id']}`
- v3 SHA-256: `{lock['combined_sha256']}`
- v3 predecessor hash: `{lock['predecessor_protocol_hash']}`

The v3 cell list and development selection table are inputs to the v3 hash. Any edit invalidates `verify_v3_lock()`.

## S2. Data flow and leakage controls

1. Select coordinates from metadata only.
2. Hash coordinates, configuration, protocol, and development table.
3. Fit the candidate and memory using the original training/validation cells.
4. Stream values at fresh cells.
5. Fit only 1959–1994 local climatology and scales at those cells.
6. Admit 2017–2022 origins only if the full context and 168-hour target remain in the interval.
7. Evaluate once; never refit under the same protocol ID.

The feature manifest records zero confirmation years used for normalization. The online audit records `forecast_origin - inserted_origin - max_horizon >= 0` for every update.

## S3. Model and optimization details

- Context: {config['model']['context_steps']} six-hour steps.
- Horizons: {config['model']['horizon_labels']}.
- Batch size: {config['model']['batch_size']}.
- Epoch cap / patience: {config['model']['epochs']} / {config['model']['patience']}.
- AdamW learning rate / weight decay: {config['model']['learning_rate']} / {config['model']['weight_decay']}.
- Seeds: {config['model']['seeds']}.
- Memory: {config['memory']['capacity']} slots, {config['memory']['event_slots']} event slots, top-{config['memory']['top_k']} retrieval.
- Event threshold: training target-score quantile {config['memory']['event_quantile']}.

## S4. Statistical details

The paired squared-error difference is formed before resampling. A bootstrap replicate samples temporal blocks and cells with replacement and then averages across available seeds, cells, horizons, and targets. Target-horizon intervals use the same paired draws. Two-sided empirical p-values are adjusted within each 16-test contrast using Benjamini–Hochberg. Between-seed t intervals are reported separately from data-axis moving-block-bootstrap intervals.

The protocol names the matched residual backbone and the strongest non-memory development comparator as separate gates. They are the same selected model—Persistence-residual iTransformer—so the three named gates reduce to two unique paired contrasts rather than duplicating an identical statistical test.

Primary outcomes:

{contrast_text}

The following block-length sensitivity analysis was added after confirmation as a robustness diagnostic. Only the 28-day row was prespecified and used for the scientific decision.

{sensitivity_markdown}

## S5. Negative-result preservation

The v2 primary contrast versus its exact backbone was positive: MSE reduction {v2_summary['primary_mse_reduction']:.4f}, 95% CI {_format_ci(v2_summary['primary_data_bootstrap_ci'])}. Nevertheless, persistence RMSE {float(v2_board.loc['Persistence','rmse_z_mean']):.4f} was lower than PMWM RMSE {float(v2_board.loc['PMWM full','rmse_z_mean']):.4f}. Both facts are retained. No v2 target value was used to choose the v3 architecture.

## S6. Artifact map

- `results/tables/model_seed_metrics.csv`: run-level standardized metrics.
- `results/tables/physical_metrics.csv`: target/horizon physical metrics.
- `results/tables/confirmatory_effects.csv`: effect sizes, intervals, and BH results.
- `results/tables/*_seed_effects.csv`: stochastic robustness.
- `results/tables/*_cell_effects.csv`: spatial robustness.
- `results/tables/calibration.csv`: interval reliability.
- `results/tables/extreme_events.csv`: extreme ranking and conditional errors.
- `results/tables/causal_update_audit_seed*.csv`: delayed-update proof.
- `artifacts/bootstrap_draws.npz`: aggregate resampling draws.
- `figures/png` and `figures/pdf`: paired publication visuals.

## S7. Claim boundary

The supported unit of inference is the sampled set of coarse WeatherBench2 cells and the 2017–2022 interval under historical-normal adaptation. Generalization beyond this domain or information set requires a new protocol and external data.

## S8. Pre-release implementation correction

An internal code audit before repository release found that a preliminary analysis grouped observations into disjoint 28-day calendar bins although the frozen protocol specified an overlapping moving-block bootstrap. The implementation was corrected to sample overlapping 28-day blocks, concatenate them to the observed series length, truncate the final block, and resample cells independently. No model, prediction, hyperparameter, contrast, block length, or replicate count changed. The matched-backbone interval changed from [-0.0037, 0.0017] to {_format_ci(summary['contrasts']['residual_itransformer_minus_pmwm_ir']['bootstrap_ci_95'])}; the non-confirmatory decision was unchanged. A synthetic unit test fixes the corrected resampling behavior.
"""
    supplement_path = manuscript_dir / "supplement.md"
    supplement_path.write_text(supplement, encoding="utf-8")

    references = """@article{hersbach2020era5,
  title={The ERA5 global reanalysis},
  author={Hersbach, Hans and others},
  journal={Quarterly Journal of the Royal Meteorological Society},
  year={2020},
  doi={10.1002/qj.3803}
}

@article{rasp2024weatherbench2,
  title={WeatherBench 2: A benchmark for the next generation of data-driven global weather models},
  author={Rasp, Stephan and others},
  journal={Journal of Advances in Modeling Earth Systems},
  year={2024},
  doi={10.1029/2023MS004019}
}

@inproceedings{zeng2023dlinear,
  title={Are Transformers Effective for Time Series Forecasting?},
  author={Zeng, Ailing and Chen, Muxi and Zhang, Lei and Xu, Qiang},
  booktitle={AAAI Conference on Artificial Intelligence},
  year={2023}
}

@inproceedings{nie2023patchtst,
  title={A Time Series is Worth 64 Words: Long-term Forecasting with Transformers},
  author={Nie, Yuqi and Nguyen, Nam H. and Sinthong, Phanwadee and Kalagnanam, Jayant},
  booktitle={International Conference on Learning Representations},
  year={2023}
}

@inproceedings{liu2024itransformer,
  title={iTransformer: Inverted Transformers Are Effective for Time Series Forecasting},
  author={Liu, Yong and Hu, Tengge and Zhang, Haoran and Wu, Haixu and Wang, Shiyu and Ma, Lintao and Long, Mingsheng},
  booktitle={International Conference on Learning Representations},
  year={2024}
}

@inproceedings{das2024timesfm,
  title={A Decoder-only Foundation Model for Time-series Forecasting},
  author={Das, Abhimanyu and Kong, Weihao and Sen, Rajat and Zhou, Yichen},
  booktitle={International Conference on Machine Learning},
  year={2024}
}

@inproceedings{woo2024moirai,
  title={Unified Training of Universal Time Series Forecasting Transformers},
  author={Woo, Gerald and others},
  booktitle={International Conference on Machine Learning},
  year={2024}
}

@article{benjamini1995fdr,
  title={Controlling the false discovery rate: a practical and powerful approach to multiple testing},
  author={Benjamini, Yoav and Hochberg, Yosef},
  journal={Journal of the Royal Statistical Society: Series B},
  year={1995}
}

@article{kunsch1989bootstrap,
  title={The jackknife and the bootstrap for general stationary observations},
  author={Kunsch, Hans R.},
  journal={The Annals of Statistics},
  year={1989}
}
"""
    references_path = manuscript_dir / "references.bib"
    references_path.write_text(references, encoding="utf-8")

    checklist = f"""# Reporting checklist

- [x] Primary hypothesis and contrasts frozen before fresh-cell value access.
- [x] Protocol files and development selection included in SHA-256 lock.
- [x] Pilot and failed v2 persistence gate retained.
- [x] Training, validation, development, and fresh confirmation separated.
- [x] Full context and every target constrained to its split.
- [x] Five PMWM seeds and at least three modern-baseline seeds.
- [x] Persistence, climatology, Ridge, DLinear, TCN, patch Transformer, inverted Transformer, GRU, and matched residual backbone.
- [x] Event-aware, uniform, and reservoir memory use equal capacity.
- [x] Effect sizes, moving-block-bootstrap intervals, seed intervals, cell robustness, and BH correction.
- [x] Post-confirmation block-length sensitivity clearly separated from the 28-day primary rule.
- [x] Standardized and physical-unit metrics.
- [x] Predictive calibration and extreme-event diagnostics.
- [x] Delayed-prequential update audit and bounded capacity.
- [x] Negative and contradictory outcomes preserved.
- [x] Data provenance, transformations, and claim boundary documented.
- [x] Code, tests, CI, notebooks, figures, tables, and verification script included.
- [{'x' if supported else ' '}] All prespecified fresh scientific decision gates passed.

- Protocol: `{summary['protocol_id']}`
- Hash: `{lock['combined_sha256']}`
"""
    checklist_path = manuscript_dir / "reporting_checklist.md"
    checklist_path.write_text(checklist, encoding="utf-8")
    return [main_path, supplement_path, references_path, checklist_path]

# Project scope derived from the research brief

The original brief proposed a persistent foundation memory for massive continuous time-series streams, potentially spanning weather, energy, traffic, finance, and health. That full program is larger than one auditable experiment.

This repository implements a bounded first study:

- domain: WeatherBench2 ERA5 only;
- task: pointwise multivariate forecasts at 6, 24, 72, and 168 hours;
- method: fixed-capacity retrieval of consolidated forecast residuals;
- data access: streamed Zarr selection rather than full-corpus materialization;
- evidence: temporal/spatial shift, modern controlled baselines, repeated seeds, paired block inference, calibration, extremes, and causal online updates;
- publication rule: favorable pilot evidence cannot become confirmation after redesign.

It does not implement or claim a multi-domain foundation model, causal world graph, global spatial weather model, or indefinite autonomous learning. Those ideas remain future research and require independent protocols and data.

---
tags: [project, thesis]
type: project
status: active
deadline: defence pending
---

# Solar GHI Forecasting

**Physics-Gated Cross-Modal Fusion for Short-Horizon Solar Irradiance Forecasting**
M.Tech Computer Engineering · Singapore · 1.3521°N 103.8198°E

## In one paragraph

PV output is almost exactly proportional to irradiance, and irradiance without cloud is
pure astronomy — so the entire forecasting problem is cloud. This project forecasts
[[GHI]] at t+1h, t+2h and t+3h by fusing a weather time series ([[Temporal branch]]) with
geostationary satellite imagery of the cloud field ([[Image branch]]), blended by a small
[[Physics gate]] conditioned on physical state. It outputs a mean *and* an uncertainty per
horizon, because reserve sizing needs an interval.

## The finding

**Negative, and defensible.** Adding the satellite branch made forecasts *worse*, the gate
collapsed to a near-constant, and gradient-boosted trees on 11 tabular features beat every
deep variant. See [[Ablation results]] and [[Gate collapse]] for the mechanism.

## Map

- **Background** → [[Problem statement]] · [[Why 1-3 hours]]
- **Data** → [[Data sources]] · [[Dataset]] · [[Feature list]] · [[Pipeline]]
- **Model** → [[Model architecture]] · [[Temporal branch]] · [[Image branch]] · [[Physics gate]] · [[Prediction head]] · [[Training setup]]
- **Results** → [[Baseline results]] · [[Ablation results]] · [[Calibration results]] · [[Gate collapse]]
- **Deployment** → [[Live pipeline]] · [[Verification loop]] · [[kt bias diagnostic]]
- **Quality** → [[Open issues]] · [[Fixed issues]] · [[Limitations]]
- **Defence** → [[Defence prep]] · [[60-second answer]] · [[Q and A bank]] · [[Three hard questions]]
- **Repo** → [[Code map]]

## Status

> [!danger] Blocking before the defence
> [[ISSUE-TEST18 Test set collapsed to 18 samples]] — the deep-model results table is
> currently scored on 18 samples over 2 days. One-line fix. Do this first.

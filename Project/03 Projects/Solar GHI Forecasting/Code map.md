---
tags: [reference]
type: note
---

# Code map

| File | Does what | Notes |
| --- | --- | --- |
| `model.py` | **Single source of truth** for architecture + inference helpers | [[Model architecture]] |
| `solar_pv_main.ipynb` | Training, ablations, evaluation | §1 EDA · §2 baselines · §3 images · §3.5 SwimSeg · §4 RAM+flow · §5 dataset · §6 model · §7 train · §8 evaluate · §9 plots |
| `historical_data.py` | Open-Meteo ERA5 + pvlib clear-sky | [[Data sources]] |
| ~~`himawari_data.py`~~ | Historical satellite tiles | ⚠️ **deleted** — restore it |
| `combined_dataset.py` | Aligns everything into one table | [[Pipeline]] |
| `current_data.py` | Live weather + satellite, with `validate_tile()` | [[Live pipeline]] |
| `predict.py` | Live forecast → `forecast_latest.json` | [[Live pipeline]] |
| `verify.py` | Verification + [[kt bias diagnostic]] report | [[Verification loop]] |
| `daily_run.sh` | Cron entry point | ⚠️ expiry `20260816` |
| `fetch_solcast.py` | PVGIS PV reference (fetched, excluded as leaky) | |

## Artefacts

| Path | What |
| --- | --- |
| `data/combined_dataset.csv` | 9,590 rows → [[Dataset]] |
| `best_model*.pt` + `.config.json` | 6 checkpoints with architecture sidecars |
| `swimseg_encoder.pt` | [[SwimSeg pretraining]] output |
| `train_stats.json` | Normalisation statistics |
| `results/*.csv` | [[Baseline results]] · [[Ablation results]] |
| `results/plots/` | eda · training_curves · horizon_degradation · gate_analysis · forecast_vs_actual |
| `analysis/` | Per-model prediction dumps, α analysis, figure builder |
| `deck/` | `build.py` → the defence pptx |
| `verification_log.csv` | Post-fix live log |
| `verification_log_prefix.csv` | Pre-fix live log (the broken era) |

## Key functions

- `model.py: build_lookback_window` ← [[ISSUE-B5 Lookback window night contamination]]
- `model.py: compute_gate_features` ← [[ISSUE-B4 Gate cloud cover synthesised]]
- `model.py: compute_clearsky_hour_mean` ← the fix that [[ISSUE-C1 Clear-sky time convention]] needs applied upstream
- `model.py: smart_persistence_forecast` / `skill_score` → [[Forecast skill score]]
- notebook cell 19 `GHIForecastDataset` ← [[ISSUE-TEST18 Test set collapsed to 18 samples]]

# Solar Irradiance Forecasting for Singapore

This project forecasts **Global Horizontal Irradiance (GHI)**, the sunlight reaching the ground, over Singapore at **1, 2 and 3 hours ahead**. It fuses a 24-step hourly weather window — daylight hours only, so roughly 2.4 days of history — with three **Himawari-8/9 satellite frames** taken ten minutes apart. Each forecast gives a **mean and an uncertainty** for every horizon.

> Solar panel output is almost exactly proportional to irradiance, and clear-sky irradiance is pure astronomy. **The entire forecasting problem is cloud.** A satellite sees cloud *upwind* of the site before it arrives. That is the information that neither sky cameras (useful for under 30 minutes) nor numerical weather models (useful beyond 6 hours) provide in the 1–3 hour window.

**Headline result (v3):** replayed over **147 days (1,029 forecasts) it never saw in training**, the shipped v3 ensemble scores **64.1 W/m² MAE against 86.9 for smart persistence (26% lower error)**. It beats persistence on **128 of 147 days**, and **91.0%** of actual values fall inside its 90% interval. Against LightGBM on the held-out test split it scores 63.6 vs 66.2 W/m². That gap is **not statistically significant** (Diebold-Mariano p = 0.054), so the honest claim is "on par with LightGBM, plus calibrated uncertainty". See [Results](#results).

---

## How it works (v3)

```
24 daylight steps (14 features) ─► Conv1D stem ─► BiLSTM ─► attention pool ─► H_tab ────┐
                                                                                         ├─► cross-attention ─► fused
3 satellite frames (t, t−10, t−20 min) + zoomed crop ─► frozen EfficientNet-B0 (shared) ─► patch tokens ──┘
                                                                                         │
fused ‖ future clear-sky (3) ─► MLP head ─► μ, σ of the clear-sky index k_t at t+1h, t+2h, t+3h
                                                   │
                                   GHI forecast = min(k_t, 1.15) × clear-sky GHI
```

1. **Predict the clear-sky index, not raw GHI.** The target is k_t = GHI / clear-sky GHI. This removes the daily sun cycle, so the model spends its capacity on cloud rather than re-learning the time of day. The Gaussian negative log-likelihood loss is weighted by future clear-sky irradiance, so errors near noon count for more than errors near sunset.
2. **Tabular branch.** 14 features: the clear-sky index, cloud cover, temperature, rain, wind, humidity, cyclical hour and month encodings, clear-sky GHI, and GHI lags at 1, 2 and 3 hours. The window is **24 hourly steps of daylight only** (08:00–17:00 SGT), so it reaches back about 2.4 calendar days rather than one. Training never saw a night row, and feeding live 24-hour windows made 14 of every 24 steps night-time zeros the model had no basis for — see the note in `build_lookback_window`. They pass through a 1D convolution stem (for interactions between lags), a 2-layer bidirectional LSTM, and attention pooling.
3. **Image branch.** One **ImageNet-pretrained EfficientNet-B0**, frozen and shared across the three frames and a zoomed crop around Singapore. The frames keep RGB because the AWS composite carries infrared in the blue channel.
4. **Cross-attention fusion.** The weather representation attends over the satellite patch tokens.
5. **Probabilistic head and ensemble.** Three models trained with different seeds are combined as an **equal-weight Gaussian mixture**: the mean is the plain average of the seed means, and the variance is the mean seed variance plus the spread of the seed means (σ² = mean(σᵢ²) + mean((μᵢ − μ̄)²)). Until 2026-11-11 the seeds were combined with precision weighting, which shrank σ by about 1/√3 and threw away the disagreement between seeds, so its 90% intervals covered only 71.9% of the test split. The mixture rule covers 91.2%, with no calibration factor needed. A physics cap (k_t ≤ 1.15) is applied in k_t space before converting back to W/m², so the uncertainty interval doesn't collapse to a single point.

The model has **4.8M parameters, 1.24M of them trainable.** v2 had 15.5M.

### Why v3 exists: what v2 taught

The v2 model had two EfficientNet-B2 encoders that were later unfrozen, and a small "physics gate" that blended the weather and image branches. It **did not beat gradient-boosted trees**:
- Its validation loss turned upward as soon as the image encoders unfroze. That was overfitting: 14.4M image parameters against about 5k training samples.
- The gate collapsed to a near-constant blending weight, so it never adapted to conditions.
- The deep model saw 11 features while the baselines saw 14.

v3 fixes each of these directly with a frozen shared encoder, cross-attention in place of the gate, the same 14 features as the baselines, a k_t target, and checkpoint selection on MAE. Detailed notes on the v2 investigation are in [`Project/`](Project/).

---

## Results

### Backtest: 147 unseen days

`scripts/backtest.py` replays the full pipeline over the test split's dates (2026-04-21 → 2026-09-14). The model forecasts every hour from 08:00 to 14:00 SGT, using only the weather and satellite frames available at that moment, and each forecast is scored against what happened (`verification_log_backtest.csv`, summarised in `report/data/results.json`).

| | t+1h | t+2h | t+3h | All |
|---|---:|---:|---:|---:|
| v3 ensemble MAE (W/m²) | 53.8 | 66.2 | 72.4 | **64.1** |
| Smart persistence MAE | 62.1 | 93.1 | 105.7 | 86.9 |
| Skill vs smart persistence | +13.4% | +28.9% | +31.5% | **+26.2%** |
| Inside the 90% interval | 89.5% | 91.0% | 92.6% | **91.0%** |

- 1,029 forecast runs, 3,087 scored pairs. v3 has the lower daily mean error on 128 of 147 days.
- Skill grows with horizon, which is what you'd expect if the model carries information about how cloud is evolving.
- These are the same dates as the offline test split below, so this is the same held-out period scored end to end through the live code path, not an independent second test.
- `verification_log_backtest_precision.csv` keeps the run under the old precision-weighted rule (MAE 63.9, coverage 71.7%) so the before/after comparison can be reproduced.

### Held-out test split

Chronological last 15%, n = 1,015; all models scored on identical rows (`analysis/v3_results_test_mixture.json`):

| Model | MAE (W/m²) | RMSE | R² | Skill vs smart persistence |
|-------|-----:|-----:|-----:|-----:|
| **v3 ensemble (3 seeds, mixture; shipped)** | **63.60** | **91.1** | **0.767** | **+26.1%** |
| v3 ensemble (precision-weighted; retired 2026-11-11) | 63.38 | 91.3 | 0.767 | +26.3% |
| v3 seed 42 | 64.69 | 92.2 | 0.764 | +24.8% |
| v3 seed 2024 | 65.14 | 93.7 | 0.753 | +24.3% |
| v3 seed 1337 | 65.18 | 94.4 | 0.751 | +24.2% |
| LightGBM | 66.23 | 94.9 | 0.738 | +23.0% |
| Random Forest | 66.34 | 93.6 | 0.747 | +22.9% |
| Smart persistence | 86.01 | 122.0 | 0.568 | – |
| Linear Regression | 91.26 | 118.8 | 0.596 | −6.1% |

**How to read these numbers:**
- **Every seed beats LightGBM on its own**, so the result doesn't depend on one lucky training run.
- Against LightGBM, the shipped ensemble is **63.60 vs 66.23 W/m², p = 0.054** (Diebold-Mariano test with HAC correction): **not significant at 5%**. The retired precision-weighted mean gave p = 0.040, but that rule's intervals were badly overconfident. On synthetic null data of this size the test fired 8% of the time at a nominal 5%, so even 0.040 was borderline. The fair summary is that v3 matches LightGBM on point accuracy, and it also produces a calibrated interval (91.2% coverage at a nominal 90%) that LightGBM doesn't.
- No ablation was run for v3, so these results don't show that the satellite branch is what improves accuracy.
- On the validation split the gap is larger (7.25 W/m², p = 0.0002). Checkpoints were selected on those rows, though, so that figure is optimistic, and the test numbers above are the ones to quote.
- **Live check:** v3 forecast live from 2026-09-21 to 2026-10-16 (63 forecasts over 21 days, `verification_log_v3.csv`), using the old precision-weighted intervals. It was ahead of smart persistence at t+2h (98.1 vs 112.1 W/m²) and t+3h (115.9 vs 129.7). At t+1h the two were level (89.3 vs 85.7, paired p = 0.88). The backtest shows v3 ahead at t+1h, and the cause of the difference was never isolated, so nothing is claimed about live t+1h performance. Live runs stopped on 2026-10-28 in favour of the backtest.

The files in `results/` describe the **retired v2 model**. Its numbers were never reproduced across machines (a CUDA run and a local CPU run disagree), so they're kept only as a record and shouldn't be quoted.

---

## Data

| Source | Provides |
|--------|----------|
| [Open-Meteo](https://open-meteo.com/) ERA5 archive | Hourly GHI (target), temperature, humidity, rain, wind, cloud cover |
| Himawari-8/9 via [NOAA on AWS](https://registry.opendata.aws/noaa-himawari/) | Full-disk scans every 10 minutes (public, no account needed) |
| [pvlib](https://pvlib-python.readthedocs.io/) (Ineichen model) | Clear-sky GHI, averaged over the preceding hour to match Open-Meteo's labelling |

- `data/combined_dataset_v2.csv` has **9,880 daylight rows** (08:00–17:00 SGT) from **2024-01-01 to 2026-09-14**, with satellite frames matched for 9,837 of them.
- The split is 70 / 15 / 15 and **strictly chronological**, to prevent look-ahead leakage.
- Satellite imagery (~51 GB) is **not committed**. `scripts/runpod_setup.sh` downloads just the ~2.4 GB of frames that training uses.

**Limitation:** the target is ERA5 *reanalysis*, not a ground pyranometer, so it's spatially and temporally smoothed. Live verification uses the same product, so it's a consistency check rather than independent ground truth.

---

## Repository layout

| Path | Purpose |
|------|---------|
| `model.py` | Model definitions (v3 `WinnerModel` and legacy v2) and shared helpers |
| `train.py` | v3 training: one seed per run, configured by `winner.yaml` |
| `eval_v3.py` | Scores the seeds, the ensemble and the tabular baselines on identical rows, with a Diebold-Mariano test |
| `predict_v3.py` | Live forecast from the v3 ensemble → `forecast_latest_v3.json` |
| `verify.py` | Scores past forecasts against what actually happened |
| `scripts/backtest.py` | Replays the pipeline over the test-split dates → `verification_log_backtest.csv` |
| `webapp/` | Dashboard (forecast, verification and backtest tabs); `export_static.py` builds the static `site/` |
| `run_v3.sh` | Full pipeline: trains 3 seeds, then evaluates |
| `best_model_v3_seed*.pt` | The three v3 checkpoints (~19 MB each), each with a `.stats.json` normalisation file and a `.history.json` training log |
| `historical_data.py`, `himawari_aws.py`, `combined_dataset.py` | Data collection and alignment |
| `scripts/build_npy_cache.py` | Builds the 224×224 frame cache from the command line |
| `solar_pv_main.ipynb` | The original v2 notebook: exploratory analysis, baselines and ablations |
| `Project/` | Obsidian vault with design notes, experiments, issues and limitations (written during v2) |
| `RUNPOD.md` | Guide to training on a RunPod GPU instance |

---

## Getting started

```bash
git clone https://github.com/Qasim1507/Solar-Prediction.git
cd Solar-Prediction
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-runpod.txt
```

**Live forecast** (best run between 09:00 and 14:00 SGT; the model is trained on daylight hours only):

```bash
python predict_v3.py              # fetch live weather + satellite, write forecast_latest_v3.json
python verify.py --report         # score past forecasts once the target hours have passed
```

**Reproduce the results:**

```bash
bash scripts/runpod_setup.sh                          # download the training frames (~2.4 GB)
python scripts/build_npy_cache.py                     # build the 224×224 frame cache
DEV=cuda bash run_v3.sh                               # train 3 seeds + validation evaluation
python eval_v3.py --seeds 42 1337 2024 --split test   # held-out test numbers
python scripts/backtest.py --report                   # backtest summary from the verification log
```

A CUDA GPU is recommended. See [`RUNPOD.md`](RUNPOD.md) for sizing.

## Tech stack
Python · PyTorch · timm (EfficientNet) · pandas · NumPy · scikit-learn · LightGBM · SciPy · pvlib · OpenCV · Open-Meteo API · NOAA Himawari (AWS)

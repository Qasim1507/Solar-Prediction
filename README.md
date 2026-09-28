# Solar Irradiance Forecasting for Singapore

**Physics-Gated Cross-Modal Fusion for Short-Horizon Solar Irradiance Forecasting**

This project forecasts **Global Horizontal Irradiance (GHI)**, the sunlight reaching the ground, over Singapore at **1, 2 and 3 hours ahead**. It combines a weather time series with **Himawari-8/9 geostationary satellite imagery** of the surrounding cloud field. Each forecast gives a **mean and an uncertainty** for every horizon.

> Solar panel output is almost exactly proportional to irradiance, and clear-sky irradiance is pure astronomy. **The entire forecasting problem is cloud.** A satellite sees cloud *upwind* of the site before it arrives. That is the information that neither sky cameras (useful for under 30 minutes) nor numerical weather models (useful beyond 6 hours) provide in the 1–3 hour window.

Grid operators must commit reserve generation hours in advance, so they need both a forecast and a confidence interval.

---

## How it works

```
tabular_seq  (24 steps × 11 features) ─► Temporal branch ─► H_t (256) ──┐
                                                                          ├─► Physics gate α ─► fused (256)
satellite frames t, t−10, t−20 min ─┐                                     │
centre crop (region of interest)   ─┴─► Image branch ─► Cross-attention ─► H_a (256) ──┘
                                                                          │
gate features [k_t, cloud cover, optical flow vx, vy] ─────────────────────┘

fused ‖ future clear-sky GHI (3) ─► Prediction head ─► μ (3 horizons), σ (3 horizons)
```

### 1. Temporal branch
- The input is a 24-step window of 11 daylight-hour features. These are the clear-sky index, cloud cover, temperature, rain, wind speed, humidity, cyclical hour and month encodings, and the previous hour's GHI.
- A 2-layer **bidirectional LSTM** reads the window, and **attention pooling** weights the most relevant timesteps. For example, it can focus on the last few hours during a ramp.

### 2. Image branch
- Two **EfficientNet-B2** encoders, pre-trained on the **SwimSeg** Singapore sky/cloud segmentation dataset, process the images:
  - a global encoder for three satellite frames (t, t−10 min, t−20 min, at Himawari's native cadence)
  - a region-of-interest encoder for a zoomed crop centred on Singapore
- This produces 196 image patches. The temporal representation attends to them through **cross-attention**.

### 3. Physics gate (the core idea)
A tiny 193-parameter network computes a blending weight:

```
α = sigmoid(MLP([k_t, cloud_cover, v_x, v_y]))
fused = α · H_t + (1 − α) · H_a
```

The gate uses physical state to decide which input to trust. On a clear day (k_t close to 1), the time series should be enough. Under broken, moving cloud (shown by optical flow), the satellite view should dominate.

### 4. Probabilistic prediction head
- The fused vector and the known future clear-sky GHI go through an MLP.
- The head outputs **μ and σ** for each horizon and is trained with a **Gaussian negative log-likelihood** loss, so the model produces calibrated prediction intervals.

### 5. Live forecasting and self-verification
- `predict.py` fetches live weather and validated satellite frames, runs the model, applies a clear-sky **physics clamp**, and writes `forecast_latest.json`.
- `verify.py` later fetches what actually happened and logs the error and interval coverage. It also runs a **clear-sky-index bias diagnostic**, which separates a constant offset (for example, a train/serve mismatch) from genuinely unpredictable error.

---

## Data

| Source | Provides |
|--------|----------|
| [Open-Meteo](https://open-meteo.com/) ERA5 archive | Hourly GHI (target), temperature, humidity, rain, wind, cloud cover |
| Himawari-8/9 via [NOAA on AWS](https://registry.opendata.aws/noaa-himawari/) | Full-disk satellite scans every 10 minutes (public, no account needed) |
| [pvlib](https://pvlib-python.readthedocs.io/) (Ineichen model) | Clear-sky GHI, used for the clear-sky index, future clear-sky input and physics clamp |
| SwimSeg (NTU) | Singapore sky images with cloud masks, used only for encoder pre-training |

- `data/combined_dataset.csv` covers **2024-01 → 2026-08**, 08:00–17:00 SGT only (night GHI is trivially zero).
- The train/validation/test split is 70 / 15 / 15 and **strictly chronological**, to prevent look-ahead leakage.
- Satellite imagery (~51 GB) is **not committed**. It can be downloaded again with `scripts/runpod_setup.sh`.

---

## Results

MAE in W/m² from the latest full retrain (`results/*.csv`).

**Deep model variants (ablation study)**

| Variant | MAE | RMSE | R² | 90% interval coverage | Skill vs smart persistence |
|---------|----:|-----:|---:|------:|------:|
| Naive concat fusion | 65.9 | 91.8 | 0.763 | 94.1% | +44.6% |
| CNN-only | 66.4 | 92.9 | 0.758 | 92.4% | +44.2% |
| Physics-Gated (no GHI lag) | 66.5 | 92.0 | 0.762 | 94.4% | +44.1% |
| LSTM-only | 66.9 | 94.4 | 0.748 | 95.2% | +43.7% |
| **Physics-Gated (large) ★** | 67.6 | 94.2 | 0.752 | 92.8% | +43.2% |
| Physics-Gated (small) | 72.9 | 98.9 | 0.727 | 94.1% | +38.7% |
| Smart persistence | 119.0 | 151.1 | 0.323 | – | – |

**Tabular baselines (MAE by horizon)**

| Model | t+1h | t+2h | t+3h |
|-------|-----:|-----:|-----:|
| Smart persistence | 76.2 | 106.6 | 111.4 |
| Linear Regression | 60.2 | 82.7 | 88.0 |
| Random Forest | 55.4 | 79.6 | 83.9 |
| LightGBM | 55.2 | 83.9 | 85.2 |

**Key findings**
- All deep variants beat smart persistence by about **40–45%**, and their prediction intervals are well calibrated. The 90% intervals cover 92–95% of outcomes.
- The satellite branch and physics gate give **no clear advantage** over simpler variants. The top five models are within about 2 W/m² of each other. The detailed analysis (in `Project/`) traces this to overfitting (14.4M image parameters against about 5k training samples) and to the gate collapsing to a near-constant α. Gradient-boosted trees on tabular features remain a strong baseline.

---

## Repository layout

| Path | Purpose |
|------|---------|
| `model.py` | Model architecture and inference helpers (single source of truth) |
| `solar_pv_main.ipynb` | End-to-end training: exploratory analysis, baselines, image caching, SwimSeg pre-training, training, ablations, evaluation and plots |
| `historical_data.py` | Downloads ERA5 weather and computes clear-sky GHI |
| `himawari_aws.py` | Downloads Himawari scans from NOAA's AWS archive |
| `combined_dataset.py` | Aligns weather rows with satellite frames → `data/combined_dataset.csv` |
| `current_data.py` | Fetches live weather and satellite frames, with tile validation and fallback sources |
| `predict.py` | Runs a live forecast → `forecast_latest.json` |
| `verify.py` | Scores past forecasts and runs the bias diagnostic report |
| `daily_run.sh` | Cron entry point for scheduled predict/verify runs |
| `best_model*.pt` + `.config.json` | Trained checkpoints (main model and ablations) with architecture sidecars |
| `swimseg_encoder.pt` | Pre-trained cloud-segmentation encoder |
| `analysis/`, `results/` | Evaluation scripts, figures and result tables |
| `Project/` | Obsidian vault with detailed notes on design, experiments, issues and limitations |
| `RUNPOD.md` | Guide to retraining on a RunPod GPU instance |

---

## Getting started

```bash
git clone https://github.com/Qasim1507/Solar-Prediction.git
cd Solar-Prediction
python -m venv .venv && source .venv/bin/activate
pip install torch timm pandas numpy pvlib requests pytz pillow opencv-python scikit-learn lightgbm matplotlib jupyter
cp .env.example .env        # optional: configure satellite sources
```

> The repo is about 350 MB because the trained checkpoints are included.

**Run a live forecast** (best between 09:00 and 14:00 SGT, because the model was trained on daylight hours):

```bash
python predict.py
python verify.py            # after the forecast hours have passed
python verify.py --report   # bias diagnostic over the whole log
```

**Retrain from scratch:** download the satellite frames, then run `solar_pv_main.ipynb` from top to bottom. A CUDA GPU and 16 GB+ RAM are recommended. See [`RUNPOD.md`](RUNPOD.md) for details.

```bash
bash scripts/runpod_setup.sh    # ~2.4 GB of training frames
```

## Tech stack
Python · PyTorch · timm (EfficientNet) · pandas · NumPy · scikit-learn · LightGBM · pvlib · OpenCV · Open-Meteo API · NOAA Himawari (AWS)

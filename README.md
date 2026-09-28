# Solar Irradiance Forecasting for Singapore

This project forecasts **Global Horizontal Irradiance (GHI)**, the sunlight reaching the ground, over Singapore at **1, 2 and 3 hours ahead**. It fuses 24 hours of weather data with three **Himawari-8/9 satellite frames** taken ten minutes apart. Each forecast gives a **mean and an uncertainty** for every horizon.

> Solar panel output is almost exactly proportional to irradiance, and clear-sky irradiance is pure astronomy. **The entire forecasting problem is cloud.** A satellite sees cloud *upwind* of the site before it arrives. That is the information that neither sky cameras (useful for under 30 minutes) nor numerical weather models (useful beyond 6 hours) provide in the 1–3 hour window.

**Headline result (v3):** on a held-out test split of 1,015 samples that was never used for training or model selection, the v3 ensemble scores **63.4 W/m² MAE**, compared with LightGBM's 66.2. That's a 2.84 W/m² improvement (Diebold-Mariano p = 0.040, which is borderline; see [Results](#results)).

---

## How it works (v3)

```
24 h weather window (14 features) ─► Conv1D stem ─► BiLSTM ─► attention pool ─► H_tab ──┐
                                                                                         ├─► cross-attention ─► fused
3 satellite frames (t, t−10, t−20 min) + zoomed crop ─► frozen EfficientNet-B0 (shared) ─► patch tokens ──┘
                                                                                         │
fused ‖ future clear-sky (3) ─► MLP head ─► μ, σ of the clear-sky index k_t at t+1h, t+2h, t+3h
                                                   │
                                   GHI forecast = min(k_t, 1.15) × clear-sky GHI
```

1. **Predict the clear-sky index, not raw GHI.** The target is k_t = GHI / clear-sky GHI. This removes the daily sun cycle, so the model spends its capacity on cloud rather than re-learning the time of day. The Gaussian negative log-likelihood loss is weighted by future clear-sky irradiance, so errors near noon count for more than errors near sunset.
2. **Tabular branch.** 14 features: the clear-sky index, cloud cover, temperature, rain, wind, humidity, cyclical hour and month encodings, clear-sky GHI, and GHI lags at 1, 2 and 3 hours. They pass through a 1D convolution stem (for interactions between lags), a 2-layer bidirectional LSTM, and attention pooling.
3. **Image branch.** One **ImageNet-pretrained EfficientNet-B0**, frozen and shared across the three frames and a zoomed crop around Singapore. The frames keep RGB because the AWS composite carries infrared in the blue channel.
4. **Cross-attention fusion.** The weather representation attends over the satellite patch tokens.
5. **Probabilistic head and ensemble.** Three models trained with different seeds are combined with **precision weighting**, which is the correct rule for Gaussian outputs: a model that is confident about a sample gets more say on it. A physics cap (k_t ≤ 1.15) is applied in k_t space before converting back to W/m², so the uncertainty interval doesn't collapse to a single point.

The model has **4.8M parameters, 1.24M of them trainable.** v2 had 15.5M.

### Why v3 exists: what v2 taught

The v2 model had two EfficientNet-B2 encoders that were later unfrozen, and a small "physics gate" that blended the weather and image branches. It **did not beat gradient-boosted trees**:
- Its validation loss turned upward as soon as the image encoders unfroze. That was overfitting: 14.4M image parameters against about 5k training samples.
- The gate collapsed to a near-constant blending weight, so it never adapted to conditions.
- The deep model saw 11 features while the baselines saw 14.

v3 fixes each of these directly with a frozen shared encoder, cross-attention in place of the gate, the same 14 features as the baselines, a k_t target, and checkpoint selection on MAE. Detailed notes on the v2 investigation are in [`Project/`](Project/).

---

## Results

**Held-out test split** (chronological last 15%, n = 1,015; all models scored on identical rows; `analysis/v3_results_test.json`):

| Model | MAE (W/m²) | RMSE | R² | Skill vs smart persistence |
|-------|-----:|-----:|-----:|-----:|
| **v3 ensemble (3 seeds)** | **63.38** | **91.3** | **0.767** | **+26.3%** |
| v3 seed 42 | 64.69 | 92.2 | 0.764 | +24.8% |
| v3 seed 2024 | 65.14 | 93.7 | 0.753 | +24.3% |
| v3 seed 1337 | 65.18 | 94.4 | 0.751 | +24.2% |
| LightGBM | 66.23 | 94.9 | 0.738 | +23.0% |
| Random Forest | 66.34 | 93.6 | 0.747 | +22.9% |
| Smart persistence | 86.01 | 122.0 | 0.568 | – |
| Linear Regression | 91.26 | 118.8 | 0.596 | −6.1% |

**How to read these numbers:**
- **Every seed beats LightGBM on its own**, so the result doesn't depend on one lucky training run.
- The ensemble-vs-LightGBM difference has **p = 0.040** (Diebold-Mariano test with HAC correction). On synthetic null data of this size the test fired 8% of the time at a nominal 5%, so treat this as **borderline, not settled**.
- On the validation split the gap is larger (7.25 W/m², p = 0.0002). Checkpoints were selected on those rows, though, so that figure is optimistic, and the test numbers above are the ones to quote.
- **Live check:** the v3 ensemble has been forecasting daily since 2026-09-21 (`verification_log_v3.csv`). Over that first small sample (18 forecasts across 6 days) it has **underperformed smart persistence**, especially at t+1h. `predict_v3.py` now publishes a smart-persistence t+1h forecast alongside every v3 forecast so the two can be compared fairly as more days come in.

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
```

A CUDA GPU is recommended. See [`RUNPOD.md`](RUNPOD.md) for sizing.

## Tech stack
Python · PyTorch · timm (EfficientNet) · pandas · NumPy · scikit-learn · LightGBM · SciPy · pvlib · OpenCV · Open-Meteo API · NOAA Himawari (AWS)

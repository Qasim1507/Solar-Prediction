# Running this project on RunPod

## What is and isn't in git

The satellite imagery is **not** committed. The full archive is ~51 GB, which
exceeds GitHub's push limit, and it doesn't need to be in git: it is
re-downloadable from NOAA's public Himawari archive on AWS with no account and
no credentials. A freshly downloaded scan is byte-identical to the one used for
training (verified by SHA-256).

In git: all code, `solar_pv_main.ipynb`, `data/*.csv` (3.6 MB),
`best_model.pt`, and `swimseg_encoder.pt`.

Not in git, rebuilt on the pod:

| Path | Size | How it comes back |
|---|---|---|
| `data/satellite_aws/` | ~7 GB (training frames) | `scripts/runpod_setup.sh` |
| `data/satellite_aws_npy/` | 5.6 GB | notebook Section 3 |
| `swimseg/` | 221 MB | not needed — see below |

## 1. Setup

```bash
git clone https://github.com/Qasim1507/Project.git
cd Project
bash scripts/runpod_setup.sh
```

This installs dependencies and downloads **only the scans the training set
actually uses**: each row's frame series (t, t-10min, t-20min), about 29,500
scans or ~7 GB, rather than the full 51 GB.

The reason: the archive holds a scan every 10 minutes, but the rows are hourly
(ERA5 weather is hourly), so training reads roughly three scans per hour rather
than all six. You only need the full download to **rebuild**
`combined_dataset.csv` against every available scan:

```bash
MODE=full bash scripts/runpod_setup.sh    # ~51 GB, ~3.5 h
```

Both modes are resumable — finished scans are skipped, so re-run after any
interruption or pod restart.

## 2. Train

Open `solar_pv_main.ipynb` and run from Section 0. What each section does:

| Section | What it does | Notes |
|---|---|---|
| 0–1 | Setup, load `data/combined_dataset.csv`, EDA | fast |
| 2 | Persistence baselines | fast; the bar the model must beat |
| 3 | PNG → 224×224 `.npy` cache | **required**, builds the 5.6 GB `NPY_DIR`; one-off |
| 3.5 | SwimSeg CNN pretraining | **skipped** — `swimseg_encoder.pt` is in the repo |
| 4 | Load images into RAM | ~9 GB free RAM (~29,500 frames, cached float16) |
| 5–6 | DataLoaders, model | fast |
| 7 | Training | 150 epochs, early stop at patience 20 |
| 8 | Evaluation + ablation study | retrains 5 variants → `best_model_*.pt` |
| 9 | Plots | writes into `results/` |

Section 3.5 is why `swimseg_encoder.pt` is committed: without it you would also
need the uncommitted 221 MB `swimseg/` dataset to pretrain the encoder.

**Pod sizing:** ≥9 GB free RAM for the Section 4 image cache (16 GB+ total is
comfortable), ~25 GB disk for imagery plus the `.npy` cache, and any CUDA GPU.

## Frame cadence

Each row feeds the model three satellite frames. Those are taken at **t,
t-10min and t-20min** — Himawari's native scan cadence — not at the previous
dataframe rows, which would space them an hour apart.

This matters because two of the four physics-gate inputs are an optical-flow
vector computed between frames. Measured on 300 samples, the flow explains
12.3% of the frame-to-frame change at 10-minute spacing versus 5.1% at
1-hour: over an hour a cloud field decorrelates rather than merely moving, so
the vector is largely noise.

The offsets live in `combined_dataset.FRAME_OFFSETS_MINUTES` and are imported
by `current_data.fetch_frame_series()`, so training and live inference cannot
drift apart — a mismatch there silently destroys the flow signal (ISSUE-B2/B3).

Himawari skips its full-disk scan during daily housekeeping at 02:40 UTC, so
the t-20min slot genuinely does not exist for ~10% of rows. Those fall back to
the nearest earlier scan (t-30min) rather than a zero-filled frame.

The targets stay hourly. Every non-satellite column — `ghi` included — comes
from ERA5 reanalysis, which is natively hourly; resampling them to 10 minutes
would mean interpolating the prediction target.

## 3. Save your outputs

A pod's container disk is **ephemeral** — anything not pushed is lost when the
pod stops. Training writes `best_model.pt` whenever validation improves, but
that file still only exists on the pod.

Push the outputs back to GitHub (they are small — 60 MB model plus configs,
`train_stats.json`, and `results/`):

```bash
git config user.name  "Qasim1507"
git config user.email "nalawalaq@gmail.com"
git add best_model.pt best_model*.config.json train_stats.json results/
git commit -m "Retrained on RunPod"
git push
```

A push needs credentials the HTTPS clone doesn't carry. Use a GitHub personal
access token with `repo` scope:

```bash
git remote set-url origin https://<TOKEN>@github.com/Qasim1507/Project.git
```

Treat that token as a secret: it goes in the pod shell only, never into a
committed file. Revoke it when the pod is done.

Push during a long run, not only at the end — an interrupted pod otherwise
takes the checkpoint with it.

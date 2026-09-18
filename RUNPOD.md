# Running this project on RunPod

The satellite imagery is **not** stored in git. It is ~51 GB (76,886 scans),
which exceeds GitHub's push and repo limits, and it is fully re-downloadable
from NOAA's public Himawari archive on AWS — no account, no credentials.
A freshly downloaded scan is byte-identical to the one used for training
(verified against `himawari_sg_20260910_020000.png`).

Pulling from AWS on a pod is also much faster than a 51 GB clone would be.

## Setup

```bash
git clone https://github.com/Qasim1507/Project.git
cd Project
bash scripts/runpod_setup.sh
```

That installs the dependencies and downloads the full imagery set
(2024-01-01 → 2026-09-15) into `data/satellite_aws/`.

At 21,000 scans/hour the full range takes roughly **3.5 hours**; a pod with
more bandwidth and `WORKERS=64` is typically faster. The download is
**resumable** — finished scans are skipped, so re-run it after any
interruption or pod restart.

To train on a shorter span first:

```bash
START=2025-01-01 END=2025-06-30 bash scripts/runpod_setup.sh
```

## What is in git

- All training/inference code and `solar_pv_main.ipynb`
- `data/*.csv` — the PV dataset and ground truth (3.6 MB)
- `best_model*.pt` checkpoints

## What is not

- `data/satellite_aws/` — regenerate with the script above
- `data/satellite_aws_npy/` — derived intermediates, not read by the
  current pipeline

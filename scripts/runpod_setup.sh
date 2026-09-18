#!/usr/bin/env bash
# runpod_setup.sh — bootstrap this project on a fresh RunPod pod.
#
# The Himawari imagery is NOT in git: it is re-downloadable from NOAA's public
# AWS archive, which is far faster from a pod than any git clone would be.
#
# Two modes:
#   training (default) — only the 9,843 scans referenced by
#                        data/combined_dataset.csv. ~2.4 GB, ~25 min.
#                        This is everything the notebook actually reads.
#   full               — every 10-minute daylight scan in the range.
#                        ~51 GB, ~3.5 h. Only needed to REBUILD the dataset
#                        CSV at finer cadence via combined_dataset.py.
#
# Usage, from the repo root on the pod:
#     bash scripts/runpod_setup.sh
#     MODE=full bash scripts/runpod_setup.sh
#
# Both are resumable: finished scans are skipped, so re-run after any
# interruption or pod restart.

set -euo pipefail

MODE="${MODE:-training}"
WORKERS="${WORKERS:-32}"
OUT="${OUT:-data/satellite_aws}"
START="${START:-2024-01-01}"
END="${END:-2026-09-15}"

cd "$(dirname "$0")/.."

echo "==> Installing Python dependencies"
pip install --quiet --upgrade numpy pillow requests h5py pandas psutil \
    matplotlib scikit-learn torch torchvision

mkdir -p "$OUT" logs

if [ "$MODE" = "full" ]; then
    echo "==> FULL download ${START} .. ${END} → ${OUT}  (~51 GB, ~3.5 h)"
    python himawari_aws.py --start "$START" --end "$END" \
        --out "$OUT" --workers "$WORKERS" 2>&1 | tee -a logs/himawari_download.log
else
    echo "==> TRAINING-SET download → ${OUT}  (~2.4 GB, ~25 min)"
    python scripts/fetch_training_images.py \
        --out "$OUT" --workers "$WORKERS" 2>&1 | tee -a logs/himawari_download.log
fi

echo "==> Scans on disk: $(find "$OUT" -name '*.png' | wc -l)"
echo "==> Next: open solar_pv_main.ipynb and run Section 0 onward."
echo "    Section 3 builds the 224x224 .npy cache; Section 3.5 is skipped"
echo "    automatically because swimseg_encoder.pt is in the repo."

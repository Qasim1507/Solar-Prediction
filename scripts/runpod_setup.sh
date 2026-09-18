#!/usr/bin/env bash
# runpod_setup.sh — bootstrap this project on a fresh RunPod pod.
#
# The Himawari imagery (~51 GB, 76k scans) is deliberately NOT in git: it is
# re-downloadable from NOAA's public AWS archive, which is far faster from a
# pod than any git clone would be. This script fetches it into data/satellite_aws.
#
# Usage, from the repo root on the pod:
#     bash scripts/runpod_setup.sh
#
# Override the range or parallelism with env vars:
#     START=2025-01-01 END=2025-06-30 WORKERS=64 bash scripts/runpod_setup.sh
#
# The download is resumable: finished scans are skipped, so re-running after
# an interruption (or a pod restart) picks up where it left off.

set -euo pipefail

START="${START:-2024-01-01}"
END="${END:-2026-09-15}"
WORKERS="${WORKERS:-64}"
OUT="${OUT:-data/satellite_aws}"

cd "$(dirname "$0")/.."

echo "==> Installing Python dependencies"
pip install --quiet --upgrade numpy pillow requests h5py pandas torch

echo "==> Downloading Himawari imagery ${START} .. ${END} into ${OUT}"
echo "    (~51 GB at full range; resumable, so safe to re-run)"
mkdir -p "$OUT" logs
python himawari_aws.py \
    --start "$START" \
    --end   "$END" \
    --out   "$OUT" \
    --workers "$WORKERS" 2>&1 | tee -a logs/himawari_download.log

echo "==> Done. Scans on disk:"
find "$OUT" -name '*.png' | wc -l

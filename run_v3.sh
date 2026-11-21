#!/bin/bash
# Full v3 pipeline (recipe Steps 1-6). Assumes data/combined_dataset_v2.csv and
# the .npy frame cache already exist.
set -e
cd "$(dirname "$0")"
PY=.venv/bin/python
DEV=${DEV:-mps}          # cuda on RunPod
mkdir -p outputs/logs outputs/analysis

for s in 42 1337 2024; do
  echo "=== seed $s ==="
  $PY -u src/train.py --seed "$s" --device "$DEV" 2>&1 | tee "outputs/logs/train_v3_seed$s.log"
done

$PY -u src/eval_v3.py --seeds 42 1337 2024 --device "$DEV" 2>&1 | tee outputs/logs/eval_v3.log
echo "done -> outputs/analysis/v3_results.json"

#!/bin/bash
# Daily forecast + verification runs, active until 2026-08-16 (inclusive).
# Installed via cron:
#   0 10 * * *  → predict (10:00 SGT)
#   30 13 * * * → verify  (13:30 SGT, after all 3 target hours have passed)
cd "$(dirname "$0")"
[ "$(date +%Y%m%d)" -le 20260816 ] || exit 0

mkdir -p logs
PY=.venv/bin/python3

case "$1" in
  predict) echo "=== $(date) ===" >> logs/predict_cron.log
           $PY predict.py        >> logs/predict_cron.log 2>&1 ;;
  verify)  echo "=== $(date) ===" >> logs/verify_cron.log
           $PY verify.py         >> logs/verify_cron.log  2>&1 ;;
  *)       echo "usage: daily_run.sh predict|verify" ;;
esac

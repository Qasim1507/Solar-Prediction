"""
scripts/backtest.py - replay the daily loop over past dates.

    .venv/bin/python scripts/backtest.py                       # whole test split
    .venv/bin/python scripts/backtest.py --start 2026-07-01 --end 2026-07-03
    .venv/bin/python scripts/backtest.py --report              # report only

For every issue hour (08:00-14:00 SGT by default, so t+3h lands by 17:00, the
end of the training window) it runs the same two steps as a live day, back to
back:

    predict_v3.py --as-of "<day> <hour>" --out backtest/forecasts/<run_id>.json
    verify.py --forecast <that file> --log verification_log_backtest.csv --no-model-fallback

Both run in this process, so torch and the CSV load once rather than ~1,000
times. Each step's own output goes to logs/backtest.log; the terminal gets one
line per forecast.

Resumable: issue times already in the log are skipped. Hours with no archived
satellite scan are skipped and reported at the end.

The default range is the v3 test split: train.py cuts the CSV 70/15/15 by
date, and the test part runs 2026-04-20 12:00 -> 2026-09-14 (the end of the
CSV). Starting on 04-21 keeps every issue day wholly inside it, so none of
these forecasts were seen in training or checkpoint selection. Dates before
2026-04-20 would score too well - the model has seen them.
"""
import argparse
import contextlib
import os
import sys
import time
import traceback
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)                  # both scripts use ./data, ./best_model_v3_*.pt
sys.path.insert(0, str(ROOT))

import predict_v3               # noqa: E402
import verify                   # noqa: E402

LOG = "verification_log_backtest.csv"
FC_DIR = ROOT / "backtest" / "forecasts"
RUN_LOG = ROOT / "logs" / "backtest.log"


def issue_times(start, end, first_hour, last_hour):
    days = pd.date_range(start, end, freq="D")
    return [d + pd.Timedelta(hours=h) for d in days
            for h in range(first_hour, last_hour + 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-04-21")
    ap.add_argument("--end", default="2026-09-14")
    ap.add_argument("--first-hour", type=int, default=8)
    ap.add_argument("--last-hour", type=int, default=14)
    ap.add_argument("--log", default=LOG)
    ap.add_argument("--report", action="store_true",
                    help="only print verify.py's report on the backtest log")
    a = ap.parse_args()

    if a.report:
        verify.report(log_path=a.log)
        return

    times = issue_times(a.start, a.end, a.first_hour, a.last_hour)
    done = set(verify.read_log(a.log)["forecast_time"].astype(str))
    todo = [t for t in times if t.strftime("%Y-%m-%d %H:%M") not in done]
    print(f"{len(times)} issue times {a.start} -> {a.end}, "
          f"{a.first_hour:02d}:00-{a.last_hour:02d}:00 SGT; "
          f"{len(times) - len(todo)} already logged, {len(todo)} to run")
    print(f"step output -> {RUN_LOG.relative_to(ROOT)}\n")

    FC_DIR.mkdir(parents=True, exist_ok=True)
    RUN_LOG.parent.mkdir(exist_ok=True)
    ghi = verify.csv_ghi_series()
    skipped, t0 = [], time.time()

    with open(RUN_LOG, "a") as runlog:
        for i, t in enumerate(todo, 1):
            stamp = t.strftime("%Y-%m-%d %H:%M")
            out = FC_DIR / f"{t:%Y%m%d_%H%M}.json"
            try:
                with contextlib.redirect_stdout(runlog):
                    print(f"\n{'#' * 74}\n# {stamp}\n{'#' * 74}")
                    predict_v3.main(["--as-of", stamp, "--out", str(out)])
                    verify.verify(model_fallback=False, log_path=a.log,
                                  forecast_path=str(out), ghi_cache=ghi)
            except Exception as e:
                skipped.append((stamp, str(e)))
                runlog.write(traceback.format_exc())
                print(f"[{i}/{len(todo)}] {stamp}  SKIPPED: {e}")
                continue
            rate = (time.time() - t0) / i
            print(f"[{i}/{len(todo)}] {stamp}  ok  "
                  f"(~{rate * (len(todo) - i) / 60:.0f} min left)")

    if skipped:
        print(f"\n{len(skipped)} skipped:")
        for s, why in skipped:
            print(f"  {s}  {why}")
    verify.report(log_path=a.log)


if __name__ == "__main__":
    main()

"""
compare_t1.py — decide who serves t+1h: the v3 ensemble or smart persistence.

    python scripts/compare_t1.py
    python scripts/compare_t1.py --since 2026-09-27     # only the new trial days

Read-only. Reads verification_log_v3.csv and prints a verdict.

The decision rule, fixed in advance so the answer is computed rather than
eyeballed. Smart persistence takes over t+1h only if BOTH hold:

  * its t+1h MAE is more than MIN_GAIN% lower than v3's, and
  * it wins on most of the days compared.

Two thresholds because one blown day out of five can carry a mean on its own.

Two arms are compared:
  * "paired"  — days where BOTH a v3 t+1h row and a committed t+1h-sp row exist.
                This is the real A/B: both forecasts were published in advance.
  * "pooled"  — every v3 t+1h row against the sp_forecast_wm2 column verify.py
                has always computed retrospectively. That column is honest (it
                uses only the fully-elapsed issue hour, no look-ahead), so it is
                legitimate to report, and it covers the days before the A/B
                started. Shown as context, not as the verdict.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import skill_score

LOG = "./verification_log_v3.csv"
MIN_GAIN = 15.0          # percent MAE reduction smart persistence must clear
V3_METHOD = "v3_ensemble"


def day_table(log_path, since=None):
    df = pd.read_csv(log_path)
    df = df[df["source"] == "measured"].copy()
    if "method" not in df.columns:
        df["method"] = V3_METHOD
    df["method"] = df["method"].fillna(V3_METHOD)
    df["target_time"] = pd.to_datetime(df["target_time"])
    df["date"] = df["target_time"].dt.date
    for c in ["abs_error_wm2", "sp_abs_error_wm2", "forecast_wm2", "actual_wm2"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if since:
        df = df[df["date"] >= pd.Timestamp(since).date()]
    v3 = df[(df["method"] == V3_METHOD) & (df["horizon"] == "t+1h")]
    sp = df[df["method"] == "smart_persistence"]
    return df, v3, sp


def picp(s):
    return 100 * s.astype(str).str.lower().eq("true").mean() if len(s) else np.nan


def paired_t(d):
    """Paired t on day-level differences. Returns (t, n) or (nan, n)."""
    d = np.asarray(d, dtype=float)
    n = len(d)
    if n < 2 or d.std(ddof=1) == 0:
        return float("nan"), n
    return float(d.mean() / (d.std(ddof=1) / np.sqrt(n))), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=LOG)
    ap.add_argument("--since", default=None,
                    help="only days on or after this date (YYYY-MM-DD)")
    ap.add_argument("--min-gain", type=float, default=MIN_GAIN)
    a = ap.parse_args()

    if not os.path.exists(a.log):
        raise SystemExit(f"no log at {a.log}")

    df, v3, sp = day_table(a.log, a.since)
    if v3.empty:
        raise SystemExit("no measured v3 t+1h rows in the log")

    print(f"\n{'='*70}\n  t+1h: v3 ENSEMBLE vs SMART PERSISTENCE\n{'='*70}")
    print(f"  log {a.log}"
          + (f"   (days from {a.since})" if a.since else ""))

    # ── the real A/B: days with both arms committed in advance ────────────────
    j = (v3.groupby("date")["abs_error_wm2"].mean().rename("v3")
         .to_frame()
         .join(sp.groupby("date")["abs_error_wm2"].mean().rename("sp"),
               how="inner"))

    print(f"\n  PAIRED — both arms published in advance ({len(j)} day(s))")
    if j.empty:
        print("    none yet. Run predict_v3.py with --sp-t1 (the default), then")
        print("    verify.py, for five days.")
    else:
        print(f"    {'date':12} {'v3':>8} {'smart-pers':>11} {'winner':>8}")
        for d, r in j.iterrows():
            print(f"    {str(d):12} {r['v3']:>8.1f} {r['sp']:>11.1f} "
                  f"{'sp' if r['sp'] < r['v3'] else 'v3':>8}")
        mv, ms = j["v3"].mean(), j["sp"].mean()
        gain = 100 * (1 - ms / mv) if mv > 0 else float("nan")
        wins = int((j["sp"] < j["v3"]).sum())
        t, n = paired_t(j["v3"] - j["sp"])
        print(f"    {'MAE':12} {mv:>8.1f} {ms:>11.1f}")
        print(f"\n    smart persistence MAE gain   {gain:+.1f}% "
              f"(needs > {a.min_gain:.0f}%)")
        print(f"    days won by smart persistence {wins}/{len(j)} "
              f"(needs > {len(j)//2})")
        print(f"    paired day-level t            "
              + (f"{t:+.2f} on n={n}" if np.isfinite(t) else f"n/a (n={n})")
              + "   — context only, ~5 days rarely reaches p<0.05")

    # ── PICP for each arm ────────────────────────────────────────────────────
    print(f"\n  INTERVAL COVERAGE (nominal 90%)")
    print(f"    v3                {picp(v3['in_90ci']):>5.0f}%  (n={len(v3)})")
    if not sp.empty:
        print(f"    smart persistence {picp(sp['in_90ci']):>5.0f}%  (n={len(sp)})")

    # ── pooled context from the retrospective column ─────────────────────────
    pool = v3.dropna(subset=["sp_abs_error_wm2"])
    if not pool.empty:
        pv = pool["abs_error_wm2"].mean()
        ps = pool["sp_abs_error_wm2"].mean()
        pdays = pool["date"].nunique()
        print(f"\n  POOLED CONTEXT — v3 vs the retrospective sp column "
              f"({len(pool)} rows over {pdays} day(s))")
        print(f"    v3 {pv:.1f}   smart-pers {ps:.1f}   "
              f"v3 skill {100*skill_score(pv, ps):+.1f}%")

    # ── verdict ──────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    if j.empty or len(j) < 3:
        print(f"  VERDICT: inconclusive — {len(j)} paired day(s), need ~5")
    else:
        mv, ms = j["v3"].mean(), j["sp"].mean()
        gain = 100 * (1 - ms / mv) if mv > 0 else float("nan")
        wins = int((j["sp"] < j["v3"]).sum())
        beats_mae = np.isfinite(gain) and gain > a.min_gain
        beats_days = wins > len(j) // 2
        if beats_mae and beats_days:
            print("  VERDICT: persistence — switch t+1h to smart persistence.")
            print("    Both gates cleared. Note this routes around the live t+1h")
            print("    regression rather than fixing it: offline, v3 beats")
            print("    persistence at t+1h (~69 vs 76 W/m2).")
        elif beats_mae or beats_days:
            print("  VERDICT: inconclusive — one gate cleared, not both.")
            print(f"    MAE gate {'PASS' if beats_mae else 'fail'}, "
                  f"days gate {'PASS' if beats_days else 'fail'}. Keep logging.")
        else:
            print("  VERDICT: v3 — keep the ensemble on t+1h.")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

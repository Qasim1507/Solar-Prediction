"""
verify.py — Verify past forecasts, and diagnose *how* they are wrong
=====================================================================
Two jobs:

  VERIFY   Load forecast_latest.json, fetch what actually happened from
           Open-Meteo, and append one row per horizon to verification_log.csv.

  DIAGNOSE Read the whole log and answer the question raw MAE cannot:
           is the model wrong by a constant factor, or is it wrong at random?

Why the clear-sky index matters here
------------------------------------
Absolute error (W/m²) scales with how much irradiance there is, so a forecast
issued at 08:00 and one issued at noon are not comparable on MAE. Dividing by
clear-sky GHI removes the sun and leaves only the cloud signal:

    k_t = GHI / GHI_clearsky            (0 = black overcast, ~1 = cloudless)
    k_t bias = k_t_forecast − k_t_actual

If `k_t bias` is large but its spread across horizons and days is small, the
model is not confused — it is *offset*. A constant offset has a cause you can
find and fix (a train/serve mismatch, a normalisation error). Scattered bias
means the model genuinely cannot read the sky. Those are completely different
problems, and MAE cannot tell them apart.

Clear-sky convention
--------------------
Open-Meteo labels each hourly value with the END of its averaging window (the
value at 14:00 is the 13:00→14:00 mean). So k_t must use the preceding-hour
MEAN clear-sky, not the instantaneous value — `compute_clearsky_hour_mean`.
Using the instantaneous value produces a spurious drift in k_t from morning to
evening that has nothing to do with cloud.

Usage
-----
    python verify.py                 # verify the latest forecast, then report
    python verify.py --report        # report on the existing log only
    python verify.py --no-model-fallback
    python verify.py --log verification_log_prefix.csv --report
"""

import os
import sys
import json
import argparse
import warnings
from datetime import timedelta

import numpy as np
import pandas as pd
import requests
import torch
import pytz

from model import (
    load_model,
    run_model,
    load_historical_df,
    extend_with_recent,
    build_lookback_window,
    compute_clearsky_ghi,
    compute_clearsky_hour_mean,
    SG_LAT, SG_LON,
)

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
CSV_PATH      = "./data/combined_dataset_v2.csv"
# The v2 checkpoint has been retired; forecasts come from predict_v3.py.
# The model fallback below only runs with --forecast pointing at a v2 JSON
# and a v2 checkpoint restored from git history.
MODEL_PATH    = "./best_model.pt"
STATS_PATH    = "./train_stats.json"
FORECAST_PATH = "./forecast_latest.json"
SATELLITE_IMG = "./datanow/satellite/himawari_current.png"
LOG_PATH      = "./verification_log.csv"

SGT = pytz.timezone("Asia/Singapore")

LOG_COLUMNS = [
    "verified_at", "forecast_time", "target_time", "horizon",
    "forecast_wm2", "actual_wm2", "source", "abs_error_wm2", "in_90ci",
    # diagnostic columns
    "clearsky_wm2", "kt_forecast", "kt_actual", "kt_bias",
    "sp_forecast_wm2", "sp_abs_error_wm2",
]


# ══════════════════════════════════════════════════════════════════════════════
# DATA FETCH
# ══════════════════════════════════════════════════════════════════════════════

def fetch_ghi_series(past_days: int = 7) -> dict:
    """
    Hourly GHI (shortwave radiation, W/m²) from Open-Meteo for the recent past.

    Only hours that have fully elapsed are returned — anything later is
    Open-Meteo's own forecast, not an analysis, and verifying a forecast
    against another forecast is meaningless.

    NOTE: this is ERA5-family reanalysis, not a pyranometer. It is the same
    product used to build the training targets, so this is a consistency
    check against the training distribution — not independent ground truth.

    Returns {pd.Timestamp (SGT naive, floored to the hour): ghi}
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude":      SG_LAT,
        "longitude":     SG_LON,
        "hourly":        "shortwave_radiation",
        "past_days":     past_days,
        "forecast_days": 1,
        "timezone":      "Asia/Singapore",
    }
    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        hourly = resp.json()["hourly"]
    except Exception as e:
        print(f"  ⚠️  Could not fetch GHI from Open-Meteo: {e}")
        return {}

    cutoff = pd.Timestamp.now(tz="Asia/Singapore").tz_localize(None).floor("h")
    return {
        pd.Timestamp(t).floor("h"): float(v)
        for t, v in zip(pd.to_datetime(hourly["time"]),
                        hourly["shortwave_radiation"])
        if v is not None and pd.Timestamp(t).floor("h") <= cutoff
    }


# ══════════════════════════════════════════════════════════════════════════════
# MODEL RE-ESTIMATE (fallback — self-consistency only, not true verification)
# ══════════════════════════════════════════════════════════════════════════════

def fetch_data_for_time(target_time_sgt):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from current_data import WeatherCollector, SatelliteCollector

    print(f"    Fetching data for "
          f"{target_time_sgt.strftime('%Y-%m-%d %H:%M SGT')}...", end=" ")
    try:
        WeatherCollector().fetch_data(
            date_time=target_time_sgt.strftime("%Y-%m-%dT%H:%M:%S"))
        SatelliteCollector().fetch_image(date_time=target_time_sgt)
        print("✓")
    except Exception as e:
        print(f"⚠️  {str(e)[:40]}")
    return SATELLITE_IMG


def model_reestimate(model, device, df, train_stats, forecast, idx, target_time):
    sat_path    = fetch_data_for_time(target_time)
    tabular_seq = build_lookback_window(df, train_stats, target_time)
    future_cs   = [compute_clearsky_ghi(pd.Timestamp(f["time_sgt"]).tz_localize(SGT))
                   for f in forecast["forecasts"]]
    mu, _ = run_model(model, tabular_seq, sat_path,
                      torch.tensor([future_cs], dtype=torch.float32),
                      df, target_time, device)
    mu_np = mu.cpu().numpy()[0]
    return float(np.clip(mu_np[idx] * float(train_stats["ghi_std"])
                         + float(train_stats["ghi_mean"]), 0, None))


# ══════════════════════════════════════════════════════════════════════════════
# LOG I/O
# ══════════════════════════════════════════════════════════════════════════════

def read_log(path: str) -> pd.DataFrame:
    """Read the log, tolerating an empty file and an older/narrower schema."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return pd.DataFrame(columns=LOG_COLUMNS)
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=LOG_COLUMNS)

    # Backfill diagnostic columns for rows written by an older verify.py
    if "clearsky_wm2" not in df.columns or df["clearsky_wm2"].isna().any():
        print("  ↻ Backfilling clear-sky / k_t columns for older log rows...")
        for c in LOG_COLUMNS:
            if c not in df.columns:
                df[c] = np.nan
        need = df["clearsky_wm2"].isna()
        for i in df.index[need]:
            tt = pd.Timestamp(df.at[i, "target_time"])
            cs = compute_clearsky_hour_mean(SGT.localize(tt.to_pydatetime()))
            df.at[i, "clearsky_wm2"] = round(cs, 1)
        df["kt_forecast"] = df["forecast_wm2"] / df["clearsky_wm2"]
        df["kt_actual"]   = df["actual_wm2"]   / df["clearsky_wm2"]
        df["kt_bias"]     = df["kt_forecast"]  - df["kt_actual"]

    return df.reindex(columns=LOG_COLUMNS)


def append_log(path: str, rows: list) -> int:
    """Append rows, skipping any (forecast_time, horizon) pair already present."""
    new = pd.DataFrame(rows).reindex(columns=LOG_COLUMNS)
    prev = read_log(path)
    if len(prev):
        seen = set(zip(prev["forecast_time"].astype(str),
                       prev["horizon"].astype(str)))
        new = new[~new.apply(
            lambda r: (str(r["forecast_time"]), str(r["horizon"])) in seen,
            axis=1)]
    if new.empty:
        return 0
    combined = pd.concat([prev, new], ignore_index=True) if len(prev) else new
    combined.to_csv(path, index=False)
    return len(new)


# ══════════════════════════════════════════════════════════════════════════════
# VERIFY
# ══════════════════════════════════════════════════════════════════════════════

def verify(model_fallback=True, log_path=LOG_PATH, forecast_path=FORECAST_PATH):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*74}")
    print("  FORECAST VERIFICATION")
    print(f"{'='*74}")

    with open(forecast_path) as f:
        forecast = json.load(f)
    if forecast.get("model"):
        print(f"  Model: {forecast['model']}")

    ftime = SGT.localize(
        pd.to_datetime(forecast["forecast_time_sgt"]).replace(tzinfo=None))
    print(f"  Forecast made at: {ftime.strftime('%Y-%m-%d %H:%M SGT')}\n")

    targets = [SGT.localize(pd.to_datetime(f["time_sgt"]).replace(tzinfo=None))
               for f in forecast["forecasts"]]

    now = pd.Timestamp.now(tz="Asia/Singapore")
    pending = [t for t in targets if pd.Timestamp(t) > now]
    if pending:
        print(f"  ⏳ {len(pending)}/{len(targets)} forecast hours haven't happened yet"
              f" — run again after {max(targets).strftime('%H:%M')} SGT.\n")
        if len(pending) == len(targets):
            print("  Nothing to verify yet.\n")
            return

    print("  Fetching GHI from Open-Meteo...")
    series = fetch_ghi_series()
    actuals = {pd.Timestamp(t).replace(tzinfo=None).floor("h"): series[k]
               for t in targets
               for k in [pd.Timestamp(t).replace(tzinfo=None).floor("h")]
               if k in series}
    print(f"  ✓ Actuals for {len(actuals)}/{len(targets)} target hours")

    # GHI at issue time — needed for the smart-persistence reference
    issue_key = pd.Timestamp(ftime).replace(tzinfo=None).floor("h")
    ghi_issue = series.get(issue_key)
    cs_issue  = compute_clearsky_hour_mean(ftime)
    kt_issue  = (ghi_issue / cs_issue) if (ghi_issue is not None and cs_issue > 1) else None
    if kt_issue is None:
        print("  ⚠️  No GHI at issue time — smart-persistence reference unavailable")
    else:
        print(f"  Issue-time k_t = {kt_issue:.3f}  "
              f"(GHI {ghi_issue:.0f} / clear-sky {cs_issue:.0f})")
    print()

    model = df = stats = None
    if model_fallback and len(actuals) < len(targets):
        with open(STATS_PATH) as f:
            stats = json.load(f)
        model = load_model(MODEL_PATH, device)
        df    = extend_with_recent(load_historical_df(CSV_PATH))

    hdr = (f"  {'Horizon':<7} {'Fcst':>8} {'Actual':>8} {'Err':>7} "
           f"{'CS':>7} {'kt_f':>6} {'kt_a':>6} {'kt bias':>8} {'CI':>4}")
    print("-" * 74); print(hdr); print("-" * 74)

    rows = []
    for idx, f in enumerate(forecast["forecasts"]):
        tt   = targets[idx]
        key  = pd.Timestamp(tt).replace(tzinfo=None).floor("h")
        fcst = float(f["ghi_forecast_wm2"])

        if pd.Timestamp(tt) > now:
            print(f"  {f['horizon']:<7} {fcst:>8.1f} {'(pending)':>8}")
            continue

        if key in actuals:
            actual, source = actuals[key], "measured"
        elif model is not None:
            actual = model_reestimate(model, device, df, stats, forecast, idx, tt)
            source = "model*"
        else:
            print(f"  {f['horizon']:<7} {fcst:>8.1f} {'—':>8}")
            continue

        cs    = compute_clearsky_hour_mean(tt)
        kt_f  = fcst / cs if cs > 1 else np.nan
        kt_a  = actual / cs if cs > 1 else np.nan
        bias  = kt_f - kt_a
        err   = abs(fcst - actual)
        in_ci = bool(f["ghi_lower_90"] <= actual <= f["ghi_upper_90"])

        sp = sp_err = np.nan
        if kt_issue is not None and cs > 1:
            sp     = float(np.clip(kt_issue, 0, 1.3)) * cs
            sp_err = abs(sp - actual)

        print(f"  {f['horizon']:<7} {fcst:>8.1f} {actual:>8.1f} {err:>7.1f} "
              f"{cs:>7.0f} {kt_f:>6.3f} {kt_a:>6.3f} {bias:>+8.3f} "
              f"{'yes' if in_ci else 'NO':>4}")

        rows.append({
            "verified_at":      pd.Timestamp.now(tz="Asia/Singapore").strftime("%Y-%m-%d %H:%M"),
            "forecast_time":    forecast["forecast_time_sgt"],
            "target_time":      f["time_sgt"],
            "horizon":          f["horizon"],
            "forecast_wm2":     round(fcst, 1),
            "actual_wm2":       round(actual, 1),
            "source":           source,
            "abs_error_wm2":    round(err, 1),
            "in_90ci":          in_ci,
            "clearsky_wm2":     round(cs, 1),
            "kt_forecast":      round(kt_f, 4),
            "kt_actual":        round(kt_a, 4),
            "kt_bias":          round(bias, 4),
            "sp_forecast_wm2":  round(sp, 1) if np.isfinite(sp) else "",
            "sp_abs_error_wm2": round(sp_err, 1) if np.isfinite(sp_err) else "",
        })

    print("-" * 74)

    if rows:
        n = append_log(log_path, rows)
        print(f"\n  ✓ Logged {n} new row(s) → {log_path}")
        mb = np.nanmean([r["kt_bias"] for r in rows])
        print(f"  This run: MAE {np.mean([r['abs_error_wm2'] for r in rows]):.1f} W/m²  |  "
              f"mean k_t bias {mb:+.3f} "
              f"({'under' if mb < 0 else 'over'}-forecasting the sky)")
        if any(r["source"] == "model*" for r in rows):
            print("  * model re-estimate — self-consistency only, not verification.")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# DIAGNOSE
# ══════════════════════════════════════════════════════════════════════════════

def report(log_path=LOG_PATH):
    df = read_log(log_path)
    df = df[df["source"] == "measured"].copy()
    if df.empty:
        print(f"\n  No measured rows in {log_path} yet — nothing to report.\n")
        return

    df["target_time"] = pd.to_datetime(df["target_time"])
    df["issue_hour"]  = pd.to_datetime(df["forecast_time"]).dt.hour
    for c in ["forecast_wm2", "actual_wm2", "abs_error_wm2", "clearsky_wm2",
              "kt_forecast", "kt_actual", "kt_bias", "sp_abs_error_wm2"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    ndays = df["target_time"].dt.date.nunique()

    print(f"\n{'='*74}")
    print(f"  DIAGNOSTIC REPORT — {log_path}")
    print(f"  {len(df)} verified hours over {ndays} day(s), "
          f"{df['target_time'].min().date()} → {df['target_time'].max().date()}")
    print(f"{'='*74}\n")

    # ── 1. Accuracy, absolute and in k_t ──────────────────────────────────────
    print("  1. ACCURACY BY HORIZON")
    print(f"     {'':6} {'n':>3} {'MAE':>7} {'bias':>8} {'actual sd':>10} "
          f"{'kt bias':>9} {'kt sd':>7} {'PICP':>6}")
    for h in ["t+1h", "t+2h", "t+3h"]:
        s = df[df["horizon"] == h]
        if s.empty:
            continue
        print(f"     {h:6} {len(s):>3} {s['abs_error_wm2'].mean():>7.1f} "
              f"{(s['forecast_wm2']-s['actual_wm2']).mean():>+8.1f} "
              f"{s['actual_wm2'].std():>10.1f} "
              f"{s['kt_bias'].mean():>+9.3f} {s['kt_bias'].std():>7.3f} "
              f"{100*s['in_90ci'].astype(str).str.lower().eq('true').mean():>5.0f}%")

    # ── 2. The constant-offset test ───────────────────────────────────────────
    b      = df["kt_bias"].dropna()
    mb, sb = b.mean(), b.std()
    print(f"\n  2. IS THE ERROR A CONSTANT OFFSET?")
    print(f"     mean k_t bias      {mb:+.3f}")
    print(f"     sd of k_t bias     {sb:.3f}" if np.isfinite(sb) else
          "     sd of k_t bias     n/a (need >1 row)")
    if np.isfinite(sb) and abs(mb) > 1e-9:
        ratio = sb / abs(mb)
        print(f"     sd / |mean|        {ratio:.2f}", end="   ")
        if ratio < 0.5:
            print("→ STRONGLY systematic. One offset, one cause. Fixable.")
        elif ratio < 1.0:
            print("→ mostly systematic, with real scatter on top.")
        else:
            print("→ scatter dominates. Not a simple offset.")

    corrected = (df["kt_forecast"] - mb) * df["clearsky_wm2"]
    cerr = (corrected - df["actual_wm2"]).abs()
    raw  = df["abs_error_wm2"].mean()
    print(f"\n     Remove that single constant from every forecast:")
    print(f"       MAE {raw:.1f} → {cerr.mean():.1f} W/m²", end="   ")
    if raw > 0:
        print(f"({100*(1-cerr.mean()/raw):.0f}% of all error is one offset)")

    # ── 3. Does the forecast track reality at all? ────────────────────────────
    print(f"\n  3. DOES THE FORECAST TRACK REALITY?")
    print("     (a near-constant forecast can still score a decent MAE —")
    print("      this is the check that catches it)")
    if ndays < 4:
        print(f"     Need ≥4 days at the same issue hour; have {ndays}.")
    for hr, g0 in df.groupby("issue_hour"):
        for h in ["t+1h", "t+2h", "t+3h"]:
            s = g0[g0["horizon"] == h]
            if len(s) < 4:
                continue
            r = s["forecast_wm2"].corr(s["actual_wm2"])
            fsd, asd = s["forecast_wm2"].std(), s["actual_wm2"].std()
            flag = "  ← FLAT / ANTI-CORRELATED" if (r < 0.2 or fsd < 0.3*asd) else ""
            print(f"     issue {hr:02d}:00 {h}: n={len(s)}  corr={r:+.2f}  "
                  f"fcst sd={fsd:.1f} vs actual sd={asd:.1f}{flag}")

    # ── 4. Skill against smart persistence ────────────────────────────────────
    sp = df.dropna(subset=["sp_abs_error_wm2"])
    print(f"\n  4. SKILL vs SMART PERSISTENCE")
    if sp.empty:
        print("     No reference available (issue-hour GHI was missing).")
    else:
        for h in ["t+1h", "t+2h", "t+3h"]:
            s = sp[sp["horizon"] == h]
            if s.empty:
                continue
            m, r = s["abs_error_wm2"].mean(), s["sp_abs_error_wm2"].mean()
            print(f"     {h}: model {m:6.1f}  smart-pers {r:6.1f}  "
                  f"skill {100*(1-m/r):+6.1f}%" if r > 0 else "")
        m, r = sp["abs_error_wm2"].mean(), sp["sp_abs_error_wm2"].mean()
        if r > 0:
            sk = 100*(1-m/r)
            print(f"     overall skill {sk:+.1f}%", end="   ")
            print("→ beating the free benchmark." if sk > 0 else
                  "→ NOT beating the free benchmark.")

    # ── 5. Trend ──────────────────────────────────────────────────────────────
    print(f"\n  5. IS THE BIAS DRIFTING?")
    daily = df.groupby(df["target_time"].dt.date)["kt_bias"].mean()
    if len(daily) < 2:
        print("     Need ≥2 days.")
    else:
        for d, v in daily.items():
            bar = "█" * max(1, int(abs(v) * 60))
            print(f"     {d}  {v:+.3f}  {bar}")
        print(f"     first → last: {daily.iloc[0]:+.3f} → {daily.iloc[-1]:+.3f}")

    print(f"\n{'='*74}")
    print("  HOW TO READ THIS")
    print("    A large mean k_t bias with small sd = one systematic cause.")
    print("    Prime suspects, in order: the inference lookback window "
          "containing\n    night hours the model never trained on; gate features "
          "computed\n    differently at serve time than at train time; a "
          "clear-sky convention\n    mismatch between the features and the target.")
    print(f"{'='*74}\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--report", action="store_true",
                   help="Only analyse the existing log; skip verification")
    p.add_argument("--no-model-fallback", action="store_true",
                   help="Only use measured actuals; never re-estimate")
    p.add_argument("--log", default=LOG_PATH, help="Path to the verification log")
    p.add_argument("--forecast", default=FORECAST_PATH,
                   help="Forecast JSON to verify (use forecast_latest_v3.json "
                        "for the v3 ensemble, with its own --log)")
    a = p.parse_args()

    if not a.report:
        verify(model_fallback=not a.no_model_fallback, log_path=a.log,
               forecast_path=a.forecast)
    report(log_path=a.log)

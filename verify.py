"""
verify.py — Verify past forecasts against measured GHI
=======================================================
1. Load the forecast made earlier (forecast_latest.json)
2. For each target time (t+1h, t+2h, t+3h):
   - Primary: fetch ACTUAL GHI (shortwave radiation) for that hour from
     Open-Meteo (analysis of recent past — real verification data)
   - Fallback: if actuals are unavailable, re-estimate with the model
     using data fetched for the target time (self-consistency check only)
3. Compare original forecast vs actual, report MAE/RMSE and CI coverage

Usage:
    python verify.py
    python verify.py --no-model-fallback
"""

import os
import sys
import json
import argparse
import warnings
import numpy as np
import pandas as pd
import requests
import torch
from datetime import timedelta
import pytz

from model import (
    load_model,
    run_model,
    load_historical_df,
    build_lookback_window,
    compute_clearsky_ghi,
    SG_LAT, SG_LON,
)

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
CSV_PATH      = "./data/combined_dataset.csv"
MODEL_PATH    = "./best_model.pt"
STATS_PATH    = "./train_stats.json"
FORECAST_PATH = "./forecast_latest.json"
WEATHER_JSON  = "./datanow/weather/weather_current.json"
SATELLITE_IMG = "./datanow/satellite/himawari_current.png"

SGT = pytz.timezone("Asia/Singapore")


# ══════════════════════════════════════════════════════════════════════════════
# ACTUAL GHI (real verification data)
# ══════════════════════════════════════════════════════════════════════════════

def fetch_actual_ghi(target_times) -> dict:
    """
    Fetch actual hourly GHI (shortwave radiation, W/m²) for the given SGT
    timestamps from Open-Meteo. Uses the forecast API with past_days, which
    serves analysis (measured/assimilated) values for the recent past.

    Returns {pd.Timestamp (SGT naive, floored to hour): ghi} for the hours found.
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude":  SG_LAT,
        "longitude": SG_LON,
        "hourly":    "shortwave_radiation",
        "past_days": 7,
        "forecast_days": 1,
        "timezone":  "Asia/Singapore",
    }
    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        hourly = resp.json()["hourly"]
        times  = pd.to_datetime(hourly["time"])
        values = hourly["shortwave_radiation"]
        lookup = {t: v for t, v in zip(times, values) if v is not None}
    except Exception as e:
        print(f"  ⚠️  Could not fetch actual GHI from Open-Meteo: {e}")
        return {}

    out = {}
    for tt in target_times:
        key = pd.Timestamp(tt).replace(tzinfo=None).floor("h")
        if key in lookup:
            out[key] = float(lookup[key])
    return out


# ══════════════════════════════════════════════════════════════════════════════
# MODEL RE-ESTIMATE (fallback — self-consistency only, not true verification)
# ══════════════════════════════════════════════════════════════════════════════

def fetch_data_for_time(target_time_sgt):
    """Fetch weather + satellite for the target time via current_data collectors."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from current_data import WeatherCollector, SatelliteCollector

    print(f"    Fetching data for "
          f"{target_time_sgt.strftime('%Y-%m-%d %H:%M SGT')}...", end=" ")
    try:
        time_str = target_time_sgt.strftime("%Y-%m-%dT%H:%M:%S")
        WeatherCollector().fetch_data(date_time=time_str)
        SatelliteCollector().fetch_image(date_time=target_time_sgt)
        print("✓")
    except Exception as e:
        print(f"⚠️  Error: {str(e)[:40]}")
    return WEATHER_JSON, SATELLITE_IMG


def model_reestimate(model, device, df, train_stats, forecast, idx, target_time):
    """Re-run the model for target_time and return the de-normalised estimate."""
    _, sat_path = fetch_data_for_time(target_time)

    tabular_seq = build_lookback_window(df, train_stats, target_time)

    future_cs = [compute_clearsky_ghi(
                     pd.Timestamp(f["time_sgt"]).tz_localize(SGT))
                 for f in forecast["forecasts"]]
    future_clearsky = torch.tensor([future_cs], dtype=torch.float32)

    # Gate features are computed inside run_model at the TARGET time
    mu, _ = run_model(model, tabular_seq, sat_path, future_clearsky,
                      df, target_time, device)

    mu_np = mu.cpu().numpy()[0]
    ghi_mean = float(train_stats["ghi_mean"])
    ghi_std  = float(train_stats["ghi_std"])
    return float(np.clip(mu_np[idx] * ghi_std + ghi_mean, 0, None))


# ══════════════════════════════════════════════════════════════════════════════
# VERIFICATION
# ══════════════════════════════════════════════════════════════════════════════

def main(model_fallback=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*70}")
    print(f"  FORECAST VERIFICATION")
    print(f"{'='*70}")

    with open(FORECAST_PATH) as f:
        forecast = json.load(f)

    forecast_time = SGT.localize(
        pd.to_datetime(forecast["forecast_time_sgt"]).replace(tzinfo=None))
    print(f"  Forecast made at: "
          f"{forecast_time.strftime('%Y-%m-%d %H:%M:%S SGT')}\n")

    target_times = [SGT.localize(pd.to_datetime(f["time_sgt"])
                                 .replace(tzinfo=None))
                    for f in forecast["forecasts"]]

    # ── Primary: real actuals ─────────────────────────────────────────────────
    print("  Fetching actual GHI from Open-Meteo...")
    actuals = fetch_actual_ghi(target_times)
    if actuals:
        print(f"  ✓ Actuals found for {len(actuals)}/{len(target_times)} "
              f"target hours\n")
    else:
        print("  ⚠️  No actuals available (targets may be >7 days old "
              "or in the future)\n")

    # ── Fallback setup ────────────────────────────────────────────────────────
    model, df, train_stats = None, None, None
    need_fallback = model_fallback and len(actuals) < len(target_times)
    if need_fallback:
        with open(STATS_PATH) as f:
            train_stats = json.load(f)
        model = load_model(MODEL_PATH, device)
        df    = load_historical_df(CSV_PATH)

    # ── Compare ───────────────────────────────────────────────────────────────
    print(f"{'='*70}")
    print(f"  {'Horizon':<8} {'Forecast':>10} {'Actual':>12} {'Source':>10} "
          f"{'Diff':>8} {'In 90% CI':>10}")
    print("-" * 70)

    results = []
    for idx, f in enumerate(forecast["forecasts"]):
        target_time  = target_times[idx]
        original_ghi = f["ghi_forecast_wm2"]
        key = pd.Timestamp(target_time).replace(tzinfo=None).floor("h")

        if key in actuals:
            actual, source = actuals[key], "measured"
        elif model is not None:
            actual = model_reestimate(model, device, df, train_stats,
                                      forecast, idx, target_time)
            source = "model*"
        else:
            print(f"  {f['horizon']:<8} {original_ghi:>10.1f} "
                  f"{'—':>12} {'n/a':>10}")
            continue

        diff   = abs(original_ghi - actual)
        in_ci  = f["ghi_lower_90"] <= actual <= f["ghi_upper_90"]
        print(f"  {f['horizon']:<8} {original_ghi:>10.1f} {actual:>12.1f} "
              f"{source:>10} {diff:>8.1f} {'yes' if in_ci else 'NO':>10}")
        results.append({"diff": diff, "in_ci": in_ci, "source": source})

    print("-" * 70)

    if results:
        mae  = np.mean([r["diff"] for r in results])
        rmse = np.sqrt(np.mean([r["diff"] ** 2 for r in results]))
        cov  = 100 * np.mean([r["in_ci"] for r in results])
        print(f"\n  📊 Statistics (n={len(results)}):")
        print(f"     MAE:  {mae:.1f} W/m²")
        print(f"     RMSE: {rmse:.1f} W/m²")
        print(f"     90% CI coverage: {cov:.0f}%")
        if any(r["source"] == "model*" for r in results):
            print(f"\n  * model re-estimate (no measured data available) — "
                  f"self-consistency only,")
            print(f"    not true verification.")

    print(f"{'='*70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-model-fallback", action="store_true",
                        help="Only use measured actuals; skip model re-estimates")
    args = parser.parse_args()
    main(model_fallback=not args.no_model_fallback)

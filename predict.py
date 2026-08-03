"""
predict.py — Real-time GHI Forecasting
=======================================
1. Runs current_data.py to fetch live weather + satellite image
2. Reads datanow/weather/weather_current.json and datanow/satellite/himawari_current.png
3. Builds 24h lookback window from historical CSV
4. Runs Physics-Gated Fusion model (v1, from model.py)
5. Outputs GHI forecast for t+1h, t+2h, t+3h

Usage:
    python predict.py
    python predict.py --model ./best_model.pt --csv ./data/combined_dataset.csv
"""

import os
import sys
import json
import argparse
import warnings
import subprocess
import numpy as np
import torch
from datetime import datetime, timedelta
import pytz

from model import (
    load_model,
    run_model,
    load_weather_from_json,
    load_historical_df,
    build_lookback_window,
    compute_clearsky_ghi,
    denormalise_forecast,
)

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
CSV_PATH      = "./combined_dataset.csv"
MODEL_PATH    = "./best_model.pt"
STATS_PATH    = "./train_stats.json"
WEATHER_JSON  = "./datanow/weather/weather_current.json"
SATELLITE_IMG = "./datanow/satellite/himawari_current.png"


def predict(model_path=MODEL_PATH, csv_path=CSV_PATH, stats_path=STATS_PATH,
            skip_fetch=False):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  GHI FORECAST — Physics-Gated Fusion Model")
    print(f"{'='*60}")
    print(f"  Device: {device}")

    # ── Step 1: Run current_data.py ───────────────────────────────────────────
    if not skip_fetch:
        print("\n  Step 1: Fetching live data via current_data.py...")
        print("-" * 60)
        script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "current_data.py")
        result = subprocess.run([sys.executable, script])
        if result.returncode != 0:
            print("  ⚠️  current_data.py had errors — continuing with saved files")
        print("-" * 60)

    # ── Step 2: Load training stats ───────────────────────────────────────────
    with open(stats_path) as f:
        train_stats = json.load(f)

    # ── Step 3: Load model ────────────────────────────────────────────────────
    print("\n  Loading model...")
    model = load_model(model_path, device)
    print(f"  ✓ Model loaded from {model_path}")

    # ── Step 4: Current time ──────────────────────────────────────────────────
    sgt     = pytz.timezone("Asia/Singapore")
    now_sgt = datetime.now(sgt).replace(minute=0, second=0, microsecond=0)
    print(f"\n  Current time (SGT): {now_sgt.strftime('%Y-%m-%d %H:%M')}")

    # ── Step 5: Load weather ──────────────────────────────────────────────────
    print(f"\n  Loading weather from {WEATHER_JSON}...")
    weather = load_weather_from_json(WEATHER_JSON)
    print(f"    Temp: {weather['temperature_2m']:.1f}°C  "
          f"Rain: {weather['rain']:.1f}mm  "
          f"RH: {weather['relative_humidity_2m']:.1f}%  "
          f"Wind: {weather['wind_speed_10m']:.1f}km/h")

    # ── Step 6: Build lookback window ─────────────────────────────────────────
    print("\n  Building 24h lookback window...")
    df          = load_historical_df(csv_path)
    tabular_seq = build_lookback_window(df, train_stats, now_sgt)

    # ── Step 7: Future clearsky GHI ───────────────────────────────────────────
    future_cs = [compute_clearsky_ghi(now_sgt + timedelta(hours=h))
                 for h in [1, 2, 3]]
    future_clearsky = torch.tensor(future_cs, dtype=torch.float32).unsqueeze(0)
    print(f"  Clearsky GHI → t+1h:{future_cs[0]:.0f}  t+2h:{future_cs[1]:.0f}  "
          f"t+3h:{future_cs[2]:.0f} W/m²")

    # ── Step 8: Inference (image + gate inputs built per architecture) ────────
    print(f"\n  Loading satellite image from {SATELLITE_IMG}...")
    mu, sigma = run_model(model, tabular_seq, SATELLITE_IMG, future_clearsky,
                          df, now_sgt, device)

    mu_real, lo, hi = denormalise_forecast(mu.cpu().numpy()[0],
                                           sigma.cpu().numpy()[0], train_stats)

    # ── Print results ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  GHI FORECAST from {now_sgt.strftime('%Y-%m-%d %H:%M')} SGT")
    print(f"{'='*60}")
    print(f"  {'Horizon':<10} {'Forecast':>12} {'90% CI':>24}  {'Clearsky':>10}")
    print(f"  {'-'*58}")
    for h in range(3):
        dt_str = (now_sgt + timedelta(hours=h + 1)).strftime("%H:%M")
        print(f"  t+{h+1}h ({dt_str})  {mu_real[h]:>8.1f} W/m²  "
              f"[{lo[h]:>6.1f} – {hi[h]:>6.1f}]  "
              f"{future_cs[h]:>8.0f} W/m²")
    print(f"{'='*60}\n")

    # ── Save forecast ─────────────────────────────────────────────────────────
    output = {
        "forecast_time_sgt": now_sgt.strftime("%Y-%m-%d %H:%M"),
        "forecasts": [
            {
                "horizon":          f"t+{h+1}h",
                "time_sgt":         (now_sgt + timedelta(hours=h + 1)
                                     ).strftime("%Y-%m-%d %H:%M"),
                "ghi_forecast_wm2": round(float(mu_real[h]), 1),
                "ghi_lower_90":     round(float(lo[h]), 1),
                "ghi_upper_90":     round(float(hi[h]), 1),
                "clearsky_wm2":     round(future_cs[h], 1),
            }
            for h in range(3)
        ],
        "current_weather": weather,
    }
    out_path = "./forecast_latest.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Forecast saved → {out_path}")

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      default=MODEL_PATH)
    parser.add_argument("--csv",        default=CSV_PATH)
    parser.add_argument("--stats",      default=STATS_PATH)
    parser.add_argument("--skip-fetch", action="store_true",
                        help="Use already-saved datanow/ files instead of fetching")
    args = parser.parse_args()
    predict(model_path=args.model, csv_path=args.csv, stats_path=args.stats,
            skip_fetch=args.skip_fetch)

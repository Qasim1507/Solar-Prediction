"""
predict.py — Real-time GHI Forecasting
=======================================
Pipeline
--------
1. Runs current_data.py to fetch live weather + a validated 3-frame satellite
   series (t, t-1h, t-2h — the v2 model needs all three for optical flow)
2. Loads the trained model, auto-detecting v1/v2 from its config sidecar
3. Builds a 24h lookback window, topped up with live hours so it ends at the
   current hour rather than trailing the archive API by ~5 days
4. Runs the Physics-Gated Fusion model
5. Applies a clear-sky physics cap and writes forecast_latest.json

Usage
-----
    python predict.py
    python predict.py --skip-fetch                    # reuse saved datanow/ files
    python predict.py --model ./best_model.pt --csv ./data/combined_dataset.csv

Notes
-----
The model was trained on daylight hours only (08:00–17:00 SGT). Outside that
window the forecast is dominated by the physics cap, not the network — run
between 09:00 and 14:00 SGT for a meaningful test.
"""

import os
import sys
import json
import time
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
    extend_with_recent,
    build_lookback_window,
    compute_clearsky_ghi,
    compute_clearsky_hour_mean,
    denormalise_forecast,
)

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
# The dataset lives in data/ locally, or at the repo root on RunPod where data/
# is gitignored — use whichever exists.
CSV_PATH        = ("./data/combined_dataset.csv"
                   if os.path.exists("./data/combined_dataset.csv")
                   else "./combined_dataset.csv")
MODEL_PATH      = "./best_model.pt"
STATS_PATH      = "./train_stats.json"
WEATHER_JSON    = "./datanow/weather/weather_current.json"
SATELLITE_IMG   = "./datanow/satellite/himawari_current.png"
SATELLITE_PREV1 = "./datanow/satellite/himawari_prev1.png"
SATELLITE_PREV2 = "./datanow/satellite/himawari_prev2.png"

# Physics: measured GHI can slightly exceed the clear-sky model at cloud edges
# (cloud enhancement), so allow 15% headroom rather than a hard ceiling.
CLEARSKY_HEADROOM = 1.15

# The model only ever saw these hours during training.
TRAIN_HOUR_START, TRAIN_HOUR_END = 8, 17


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def fetch_live_data():
    """Run current_data.py to refresh datanow/. Non-fatal on failure."""
    print("\n  Step 1: Fetching live data via current_data.py...")
    print("-" * 60)
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "current_data.py")
    result = subprocess.run([sys.executable, script])
    if result.returncode != 0:
        print("  ⚠️  current_data.py had errors — continuing with saved files")
    print("-" * 60)


def check_satellite_inputs(now_sgt):
    """
    Report on the satellite inputs before inference. A dead or missing frame
    silently corrupts the image branch, so surface it rather than hide it.
    Returns True if the current frame looks usable.
    """
    for path, label in [(SATELLITE_PREV1, "t-1h"), (SATELLITE_PREV2, "t-2h")]:
        if not os.path.exists(path):
            print(f"  ⚠️  Previous frame {label} missing — optical flow "
                  f"degraded for this run")

    if not os.path.exists(SATELLITE_IMG):
        print(f"  ⚠️  No satellite image at {SATELLITE_IMG} — the image branch "
              f"will see zeros")
        return False

    age_h = (time.time() - os.path.getmtime(SATELLITE_IMG)) / 3600
    if age_h > 2:
        print(f"  ⚠️  Satellite image is {age_h:.1f}h old — live fetch may "
              f"have failed")

    # Training images never fall near zero in daylight (mean ranges ~19 at
    # 06:00 SGT to ~88 at 11:00), so a black/flat daylight tile is a bad fetch.
    try:
        from PIL import Image
        arr = np.asarray(Image.open(SATELLITE_IMG).convert("L"),
                         dtype=np.float32)
        daylight = 7 <= now_sgt.hour <= 18
        if daylight and (arr.mean() < 3 or arr.std() < 2):
            print(f"  ⚠️  Satellite image looks DEAD "
                  f"(mean={arr.mean():.2f}, std={arr.std():.2f}) in daylight — "
                  f"treat this forecast with caution")
            return False
        print(f"  ✓ Image quality OK (mean={arr.mean():.1f}, "
              f"std={arr.std():.1f})")
        return True
    except Exception as e:
        print(f"  ⚠️  Could not inspect satellite image: {e}")
        return False


def apply_physics_cap(mu_real, lo, hi, now_sgt):
    """
    Cap the forecast at CLEARSKY_HEADROOM × clear-sky GHI.

    The cap uses the PRECEDING-HOUR MEAN clear-sky value because Open-Meteo
    labels each hourly figure with the end of its averaging window — the
    instantaneous value is badly wrong near sunrise and sunset.

    The mean is clamped first, then the interval is rebuilt around it from the
    original sigma. Clamping mu, lo and hi independently collapses the interval
    to a point whenever the cap binds.
    """
    cs_cap   = CLEARSKY_HEADROOM * np.array(
        [compute_clearsky_hour_mean(now_sgt + timedelta(hours=h))
         for h in (1, 2, 3)])
    sig_real = (hi - mu_real) / 1.645          # recover sigma before clamping
    mu_capped = np.minimum(mu_real, cs_cap)
    lo_new = np.clip(mu_capped - 1.645 * sig_real, 0, cs_cap)
    hi_new = np.clip(mu_capped + 1.645 * sig_real, 0, cs_cap)
    return mu_capped, lo_new, hi_new, cs_cap


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def predict(model_path=MODEL_PATH, csv_path=CSV_PATH, stats_path=STATS_PATH,
            skip_fetch=False):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  GHI FORECAST — Physics-Gated Fusion Model")
    print(f"{'='*60}")
    print(f"  Device: {device}")

    # ── Step 1: live data ─────────────────────────────────────────────────────
    if not skip_fetch:
        fetch_live_data()

    # ── Step 2: training stats ────────────────────────────────────────────────
    if not os.path.exists(stats_path):
        raise FileNotFoundError(
            f"{stats_path} not found — copy it back from the training run. "
            "Without it the model's outputs cannot be de-normalised.")
    with open(stats_path) as f:
        train_stats = json.load(f)

    # ── Step 3: model ─────────────────────────────────────────────────────────
    print("\n  Loading model...")
    model = load_model(model_path, device)
    print(f"  ✓ Model loaded from {model_path}")

    # ── Step 4: current time ──────────────────────────────────────────────────
    sgt     = pytz.timezone("Asia/Singapore")
    now_sgt = datetime.now(sgt).replace(minute=0, second=0, microsecond=0)
    print(f"\n  Current time (SGT): {now_sgt.strftime('%Y-%m-%d %H:%M')}")

    # ── Step 5: weather ───────────────────────────────────────────────────────
    print(f"\n  Loading weather from {WEATHER_JSON}...")
    weather = load_weather_from_json(WEATHER_JSON)
    print(f"    Temp: {weather['temperature_2m']:.1f}°C  "
          f"Rain: {weather['rain']:.1f}mm  "
          f"RH: {weather['relative_humidity_2m']:.1f}%  "
          f"Wind: {weather['wind_speed_10m']:.1f}km/h")

    # ── Step 6: lookback window ───────────────────────────────────────────────
    print("\n  Building 24h lookback window...")
    df = load_historical_df(csv_path)
    df = extend_with_recent(df)      # live API fills the archive's ~5-day lag
    data_age_h = (now_sgt.replace(tzinfo=None)
                  - df["timestamp"].max()).total_seconds() / 3600
    if data_age_h > 6:
        print(f"  ⚠️  Lookback data still ends {df['timestamp'].max()} "
              f"({data_age_h:.0f}h ago) — live top-up may have failed")
    tabular_seq = build_lookback_window(df, train_stats, now_sgt)

    # ── Step 7: future clear-sky ──────────────────────────────────────────────
    future_cs = [compute_clearsky_ghi(now_sgt + timedelta(hours=h))
                 for h in (1, 2, 3)]
    future_clearsky = torch.tensor(future_cs, dtype=torch.float32).unsqueeze(0)
    print(f"  Clearsky GHI → t+1h:{future_cs[0]:.0f}  t+2h:{future_cs[1]:.0f}  "
          f"t+3h:{future_cs[2]:.0f} W/m²")

    # ── Step 8: inference ─────────────────────────────────────────────────────
    print(f"\n  Checking satellite inputs...")
    image_ok = check_satellite_inputs(now_sgt)

    mu, sigma = run_model(model, tabular_seq, SATELLITE_IMG, future_clearsky,
                          df, now_sgt, device,
                          prev_paths=(SATELLITE_PREV1, SATELLITE_PREV2))

    mu_real, lo, hi = denormalise_forecast(mu.cpu().numpy()[0],
                                           sigma.cpu().numpy()[0], train_stats)

    # ── Step 9: physics cap ───────────────────────────────────────────────────
    mu_real, lo, hi, cs_cap = apply_physics_cap(mu_real, lo, hi, now_sgt)

    outside_training = not (TRAIN_HOUR_START <= now_sgt.hour <= TRAIN_HOUR_END)
    if outside_training:
        print(f"\n  ⚠️  {now_sgt.strftime('%H:%M')} SGT is outside the training "
              f"window ({TRAIN_HOUR_START:02d}:00–{TRAIN_HOUR_END:02d}:00).")
        print(f"      These figures are dominated by the clear-sky cap, not "
              f"the model. Run between 09:00 and 14:00 SGT to test the model.")

    # ── Results ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  GHI FORECAST from {now_sgt.strftime('%Y-%m-%d %H:%M')} SGT")
    print(f"{'='*60}")
    print(f"  {'Horizon':<10} {'Forecast':>12} {'90% CI':>24}  {'Clearsky':>10}")
    print(f"  {'-'*58}")
    for h in range(3):
        dt_str  = (now_sgt + timedelta(hours=h + 1)).strftime("%H:%M")
        capped  = "*" if mu_real[h] >= cs_cap[h] - 1e-6 else " "
        print(f"  t+{h+1}h ({dt_str})  {mu_real[h]:>8.1f}{capped} W/m²  "
              f"[{lo[h]:>6.1f} – {hi[h]:>6.1f}]  "
              f"{future_cs[h]:>8.0f} W/m²")
    if any(mu_real[h] >= cs_cap[h] - 1e-6 for h in range(3)):
        print(f"\n  * capped at {CLEARSKY_HEADROOM:.2f}x clear-sky "
              f"(hour-mean): the model wanted to predict higher")
    print(f"{'='*60}\n")

    # ── Save ──────────────────────────────────────────────────────────────────
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
                "capped":           bool(mu_real[h] >= cs_cap[h] - 1e-6),
            }
            for h in range(3)
        ],
        "current_weather": weather,
        "diagnostics": {
            "image_ok":              bool(image_ok),
            "outside_training_hours": bool(outside_training),
            "lookback_ends":         str(df["timestamp"].max()),
        },
    }
    out_path = "./forecast_latest.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Forecast saved → {out_path}")

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time GHI forecast")
    parser.add_argument("--model",      default=MODEL_PATH)
    parser.add_argument("--csv",        default=CSV_PATH)
    parser.add_argument("--stats",      default=STATS_PATH)
    parser.add_argument("--skip-fetch", action="store_true",
                        help="Reuse saved datanow/ files instead of fetching")
    args = parser.parse_args()
    predict(model_path=args.model, csv_path=args.csv, stats_path=args.stats,
            skip_fetch=args.skip_fetch)
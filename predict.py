"""
predict.py — Real-time GHI Forecasting
=======================================
1. Runs current_data.py to fetch live weather + satellite image
2. Reads datanow/weather/weather_current.json and datanow/satellite/himawari_current.png
3. Builds 24h lookback window from historical CSV
4. Runs Physics-Gated Fusion model
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
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from datetime import datetime, timedelta
import pytz
import pvlib

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
CSV_PATH        = "./data/combined_dataset.csv"
MODEL_PATH      = "./best_model.pt"
STATS_PATH      = "./train_stats.json"
WEATHER_JSON    = "./datanow/weather/weather_current.json"
SATELLITE_IMG   = "./datanow/satellite/himawari_current.png"

# ── Image normalisation (ImageNet) ────────────────────────────────────────────
IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMG_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ── Tabular feature columns (must match training) ─────────────────────────────
TABULAR_COLS = [
    "clearsky_ratio", "cloud_cover", "temperature_2m",
    "rain", "wind_speed_10m", "relative_humidity_2m",
    "sin_hour", "cos_hour", "sin_month", "cos_month",
    "ghi_lag1",
]

WINDOW_SIZE = 24


# ══════════════════════════════════════════════════════════════════════════════
# MODEL DEFINITION (must match training exactly)
# ══════════════════════════════════════════════════════════════════════════════

class TemporalSelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )
    def forward(self, x):
        w = torch.softmax(self.attn(x), dim=1)
        return torch.sum(x * w, dim=1)

class BiLSTMEncoder(nn.Module):
    def __init__(self, input_dim=11, hidden_dim=128):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2,
                            batch_first=True, bidirectional=True, dropout=0.1)
        self.attn    = TemporalSelfAttention(hidden_dim * 2)
        self.out_dim = hidden_dim * 2
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.attn(out)

class EfficientNetB2Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        import timm
        base = timm.create_model("efficientnet_b2", pretrained=False, features_only=True)
        self.backbone     = base
        self.img_channels = 352
    def forward(self, x):
        return self.backbone(x)[-1]

class SimplifiedCrossAttention(nn.Module):
    def __init__(self, query_dim, key_dim, hidden_dim=256):
        super().__init__()
        self.q_proj  = nn.Linear(query_dim, hidden_dim)
        self.k_proj  = nn.Linear(key_dim,   hidden_dim)
        self.v_proj  = nn.Linear(key_dim,   hidden_dim)
        self.scale   = hidden_dim ** -0.5
        self.out_dim = hidden_dim
    def forward(self, query, keys, values):
        Q    = self.q_proj(query).unsqueeze(1)
        K    = self.k_proj(keys)
        V    = self.v_proj(values)
        attn = torch.softmax((Q @ K.transpose(-2,-1)) * self.scale, dim=-1)
        return (attn @ V).squeeze(1), attn.squeeze(1)

class PhysicsGatedFusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.temporal     = BiLSTMEncoder(input_dim=len(TABULAR_COLS))
        self.image_enc    = EfficientNetB2Encoder()
        self.temp_dim     = self.temporal.out_dim
        self.img_channels = self.image_enc.img_channels
        self.D            = 256

        self.cross_attn = SimplifiedCrossAttention(self.temp_dim, self.img_channels, self.D)
        self.gate = nn.Sequential(
            nn.Linear(2, 16), nn.ReLU(),
            nn.Linear(16, 1), nn.Sigmoid()
        )
        self.enrich = nn.Sequential(
            nn.Linear(self.D + 3, self.D), nn.ReLU(), nn.Dropout(0.15)
        )
        self.head = nn.Sequential(
            nn.Linear(self.D, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(128, 64),    nn.LayerNorm(64),  nn.GELU(),
            nn.Linear(64, 6),
        )

    def forward(self, tabular_seq, image, future_clearsky, gate_features):
        B    = tabular_seq.shape[0]
        H_t  = self.temporal(tabular_seq)
        smap = self.image_enc(image)
        _, C, H, W = smap.shape
        img_features = smap.view(B, C, H*W).transpose(1, 2)
        H_a, _  = self.cross_attn(H_t, img_features, img_features)
        alpha   = self.gate(gate_features)
        pad     = H_t[:, :self.D] if self.temp_dim >= self.D else \
                  nn.functional.pad(H_t, (0, self.D - self.temp_dim))
        fused   = self.enrich(torch.cat([alpha * pad + (1 - alpha) * H_a,
                                          future_clearsky], dim=1))
        out     = self.head(fused)
        mu      = out[:, :3]
        sigma   = nn.functional.softplus(out[:, 3:]) + 1e-4
        return mu, sigma


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING FROM current_data.py OUTPUT
# ══════════════════════════════════════════════════════════════════════════════

def load_weather_from_json(path):
    """Parse weather_current.json into flat feature dict"""
    with open(path) as f:
        data = json.load(f)

    def avg_readings(key):
        readings = data.get(key, {}).get("readings", [])
        vals = [r["value"] for r in readings if "value" in r]
        return float(np.mean(vals)) if vals else 0.0

    return {
        "temperature_2m":       avg_readings("air-temperature"),
        "rain":                 avg_readings("rainfall"),
        "relative_humidity_2m": avg_readings("relative-humidity"),
        "wind_speed_10m":       avg_readings("wind-speed"),
    }


def load_satellite_image(path):
    """Load himawari_current.png, resize and normalise for model input"""
    if not os.path.exists(path):
        print(f"  ⚠️  Satellite image not found at {path} — using zeros")
        return torch.zeros((3, 224, 224), dtype=torch.float32)
    try:
        img = Image.open(path).convert("RGB")
        img = img.resize((224, 224), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = (arr - IMG_MEAN) / IMG_STD
        print(f"  ✓ Satellite image loaded from {path}")
        return torch.from_numpy(arr.transpose(2, 0, 1)).float()
    except Exception as e:
        print(f"  ⚠️  Could not load satellite image: {e} — using zeros")
        return torch.zeros((3, 224, 224), dtype=torch.float32)


def compute_clearsky_ghi(dt_sgt):
    """Compute GHI clearsky using pvlib for Singapore"""
    loc   = pvlib.location.Location(1.3521, 103.8198, tz="Asia/Singapore")
    times = pd.DatetimeIndex([dt_sgt])
    cs    = loc.get_clearsky(times)
    return float(cs["ghi"].iloc[0])


def build_lookback_window(csv_path, train_stats, now_sgt):
    """Pull last 24 hourly rows from historical CSV, normalise and return tensor"""
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    df["clearsky_ratio"] = df["ghi"] / (df["ghi_clearsky"] + 1e-6)
    df["hour"]           = df["timestamp"].dt.hour
    df["month"]          = df["timestamp"].dt.month
    df["sin_hour"]       = np.sin(2 * np.pi * df["hour"] / 24)
    df["cos_hour"]       = np.cos(2 * np.pi * df["hour"] / 24)
    df["sin_month"]      = np.sin(2 * np.pi * df["month"] / 12)
    df["cos_month"]      = np.cos(2 * np.pi * df["month"] / 12)
    df["ghi_lag1"]       = df["ghi"].shift(1).fillna(0)

    now_naive = now_sgt.replace(tzinfo=None)
    past      = df[df["timestamp"] <= now_naive].tail(WINDOW_SIZE)

    if len(past) < WINDOW_SIZE:
        print(f"  ⚠️  Only {len(past)} historical rows — padding with zeros")
        pad = pd.DataFrame(
            np.zeros((WINDOW_SIZE - len(past), len(TABULAR_COLS))),
            columns=TABULAR_COLS)
        tab = pd.concat([pad, past[TABULAR_COLS]], ignore_index=True)
    else:
        tab = past[TABULAR_COLS].copy()

    mean = np.array(train_stats["mean"], dtype=np.float32)
    std  = np.array(train_stats["std"],  dtype=np.float32)
    arr  = (tab.values.astype(np.float32) - mean) / std

    return torch.from_numpy(arr).float().unsqueeze(0)   # [1, 24, features]


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def predict(model_path=MODEL_PATH, csv_path=CSV_PATH, stats_path=STATS_PATH):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  GHI FORECAST — Physics-Gated Fusion Model")
    print(f"{'='*60}")
    print(f"  Device: {device}")

    # ── Step 1: Run current_data.py ───────────────────────────────────────────
    print("\n  Step 1: Fetching live data via current_data.py...")
    print("-" * 60)
    current_data_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "current_data.py")
    result = subprocess.run([sys.executable, current_data_script])
    if result.returncode != 0:
        print("  ⚠️  current_data.py had errors — continuing with saved files")
    print("-" * 60)

    # ── Step 2: Load training stats ───────────────────────────────────────────
    with open(stats_path) as f:
        train_stats = json.load(f)

    # ── Step 3: Load model ────────────────────────────────────────────────────
    print("\n  Loading model...")
    model = PhysicsGatedFusionModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
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

    # ── Step 6: Load satellite image ──────────────────────────────────────────
    print(f"\n  Loading satellite image from {SATELLITE_IMG}...")
    image_tensor = load_satellite_image(SATELLITE_IMG).unsqueeze(0).to(device)

    # ── Step 7: Build lookback window ─────────────────────────────────────────
    print("\n  Building 24h lookback window...")
    tabular_seq = build_lookback_window(csv_path, train_stats, now_sgt).to(device)

    # ── Step 8: Future clearsky GHI ───────────────────────────────────────────
    future_cs = [compute_clearsky_ghi(now_sgt + timedelta(hours=h)) for h in [1, 2, 3]]
    future_clearsky = torch.tensor(future_cs, dtype=torch.float32).unsqueeze(0).to(device)
    print(f"  Clearsky GHI → t+1h:{future_cs[0]:.0f}  t+2h:{future_cs[1]:.0f}  t+3h:{future_cs[2]:.0f} W/m²")

    # ── Step 9: Gate features ─────────────────────────────────────────────────
    df_tail  = pd.read_csv(csv_path, parse_dates=["timestamp"]).sort_values("timestamp").tail(1)
    ghi_last = float(df_tail["ghi"].iloc[0]) if len(df_tail) else 400.0
    ghi_cs   = compute_clearsky_ghi(now_sgt)
    cr       = float(np.clip(ghi_last / (ghi_cs + 1e-6), 0, 1.5))
    cc       = float(np.clip((1 - cr) * 100, 0, 100))
    gate_features = torch.tensor([[cr, cc / 100.0]], dtype=torch.float32).to(device)

    # ── Step 10: Inference ────────────────────────────────────────────────────
    with torch.no_grad():
        mu, sigma = model(tabular_seq, image_tensor, future_clearsky, gate_features)

    mu_np    = mu.cpu().numpy()[0]
    sigma_np = sigma.cpu().numpy()[0]

    ghi_mean = float(train_stats["ghi_mean"])
    ghi_std  = float(train_stats["ghi_std"])
    mu_real  = np.clip(mu_np  * ghi_std + ghi_mean, 0, None)
    sig_real = sigma_np * ghi_std
    lo       = np.clip(mu_real - 1.645 * sig_real, 0, None)
    hi       = mu_real + 1.645 * sig_real

    # ── Print results ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  GHI FORECAST from {now_sgt.strftime('%Y-%m-%d %H:%M')} SGT")
    print(f"{'='*60}")
    print(f"  {'Horizon':<10} {'Forecast':>12} {'90% CI':>24}  {'Clearsky':>10}")
    print(f"  {'-'*58}")
    for h in range(3):
        dt_str = (now_sgt + timedelta(hours=h+1)).strftime("%H:%M")
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
                "time_sgt":         (now_sgt + timedelta(hours=h+1)).strftime("%Y-%m-%d %H:%M"),
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


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  default=MODEL_PATH)
    parser.add_argument("--csv",    default=CSV_PATH)
    parser.add_argument("--stats",  default=STATS_PATH)
    args = parser.parse_args()
    predict(model_path=args.model, csv_path=args.csv, stats_path=args.stats)
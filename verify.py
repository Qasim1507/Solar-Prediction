"""
verify.py — Verify forecasts by re-estimating GHI at verification times
==========================================================================
1. Load the forecast made 3 hours ago (from forecast_latest.json)
2. For each target time (t+1h, t+2h, t+3h from original forecast):
   - Call current_data.py with that target time to fetch weather + satellite
   - Run the model with that data to re-estimate "actual" GHI
3. Compare original forecast vs re-estimated GHI

Usage:
    python verify.py
"""

import os
import sys
import json
import warnings
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
FORECAST_PATH   = "./forecast_latest.json"
CURRENT_DATA_SCRIPT = "./current_data.py"

# ── Image normalisation ───────────────────────────────────────────────────────
IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMG_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ── Tabular features ─────────────────────────────────────────────────────────
TABULAR_COLS = [
    "clearsky_ratio", "cloud_cover", "temperature_2m",
    "rain", "wind_speed_10m", "relative_humidity_2m",
    "sin_hour", "cos_hour", "sin_month", "cos_month",
    "ghi_lag1",
]

WINDOW_SIZE = 24


# ══════════════════════════════════════════════════════════════════════════════
# MODEL (from predict.py)
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
# HELPERS
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
    """Load himawari_current.png, resize and normalise"""
    if not os.path.exists(path):
        return torch.zeros((3, 224, 224), dtype=torch.float32)
    try:
        img = Image.open(path).convert("RGB")
        img = img.resize((224, 224), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = (arr - IMG_MEAN) / IMG_STD
        return torch.from_numpy(arr.transpose(2, 0, 1)).float()
    except:
        return torch.zeros((3, 224, 224), dtype=torch.float32)


def compute_clearsky_ghi(dt_sgt):
    """Compute GHI clearsky for target time"""
    loc   = pvlib.location.Location(1.3521, 103.8198, tz="Asia/Singapore")
    times = pd.DatetimeIndex([dt_sgt])
    cs    = loc.get_clearsky(times)
    return float(cs["ghi"].iloc[0])


def build_lookback_window(csv_path, train_stats, reference_time):
    """Build 24h lookback window up to reference_time"""
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

    ref_naive = reference_time.replace(tzinfo=None)
    past      = df[df["timestamp"] <= ref_naive].tail(WINDOW_SIZE)

    if len(past) < WINDOW_SIZE:
        pad = pd.DataFrame(np.zeros((WINDOW_SIZE - len(past), len(TABULAR_COLS))), columns=TABULAR_COLS)
        tab = pd.concat([pad, past[TABULAR_COLS]], ignore_index=True)
    else:
        tab = past[TABULAR_COLS].copy()

    mean = np.array(train_stats["mean"], dtype=np.float32)
    std  = np.array(train_stats["std"],  dtype=np.float32)
    arr  = (tab.values.astype(np.float32) - mean) / std

    return torch.from_numpy(arr).float().unsqueeze(0)


def fetch_data_for_time(target_time_sgt):
    """
    Fetch weather + satellite for target time by importing current_data directly.
    Uses the same collection method as current_data.py but for historical times.
    """
    # Import current_data module dynamically
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from current_data import WeatherCollector, SatelliteCollector
    
    print(f"    Fetching data for {target_time_sgt.strftime('%Y-%m-%d %H:%M SGT')}...", end=" ")
    
    try:
        # Format time for API
        time_str = target_time_sgt.strftime("%Y-%m-%dT%H:%M:%S")
        
        # Fetch weather for this time
        weather_collector = WeatherCollector()
        weather_collector.fetch_data(date_time=time_str)
        
        # Fetch satellite for this time
        sat_collector = SatelliteCollector()
        sat_collector.fetch_image(date_time=target_time_sgt)
        
        print("✓")
    except Exception as e:
        print(f"⚠️  Error: {str(e)[:40]}")
    
    return "./datanow/weather/weather_current.json", "./datanow/satellite/himawari_current.png"


# ══════════════════════════════════════════════════════════════════════════════
# VERIFICATION
# ══════════════════════════════════════════════════════════════════════════════

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\n{'='*70}")
    print(f"  FORECAST VERIFICATION")
    print(f"{'='*70}")
    
    # Load original forecast
    with open(FORECAST_PATH) as f:
        forecast = json.load(f)
    
    forecast_time = pd.to_datetime(forecast['forecast_time_sgt'])
    sgt = pytz.timezone("Asia/Singapore")
    forecast_time = sgt.localize(forecast_time.replace(tzinfo=None))
    
    print(f"  Forecast made at: {forecast_time.strftime('%Y-%m-%d %H:%M:%S SGT')}\n")
    
    # Load model and stats
    with open(STATS_PATH) as f:
        train_stats = json.load(f)
    
    model = PhysicsGatedFusionModel().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()
    
    ghi_mean = float(train_stats["ghi_mean"])
    ghi_std  = float(train_stats["ghi_std"])
    
    print(f"{'='*70}")
    print(f"{'Horizon':<10} {'Original':>12} {'Re-Estimated':>14} {'Diff':>10} {'Error %':>10}")
    print("-" * 70)
    
    results = []
    
    with torch.no_grad():
        for idx, f in enumerate(forecast['forecasts']):
            target_time = pd.to_datetime(f['time_sgt'])
            target_time = sgt.localize(target_time.replace(tzinfo=None))
            original_ghi = f['ghi_forecast_wm2']
            
            # Fetch current data for this target time
            weather_path, sat_path = fetch_data_for_time(target_time)
            
            # Load data
            weather = load_weather_from_json(weather_path)
            image_tensor = load_satellite_image(sat_path).unsqueeze(0).to(device)
            
            # Build lookback window
            tabular_seq = build_lookback_window(CSV_PATH, train_stats, target_time).to(device)
            
            # Future clearsky (all 3 values)
            future_cs_vals = [compute_clearsky_ghi(pd.Timestamp(f_item['time_sgt']).tz_localize(sgt)) for f_item in forecast['forecasts']]
            future_clearsky = torch.tensor([future_cs_vals], dtype=torch.float32).to(device)
            
            # Gate features
            df_tail  = pd.read_csv(CSV_PATH, parse_dates=["timestamp"]).sort_values("timestamp").tail(1)
            ghi_last = float(df_tail["ghi"].iloc[0]) if len(df_tail) else 400.0
            ghi_cs   = compute_clearsky_ghi(target_time)
            cr       = float(np.clip(ghi_last / (ghi_cs + 1e-6), 0, 1.5))
            cc       = float(np.clip((1 - cr) * 100, 0, 100))
            gate_features = torch.tensor([[cr, cc / 100.0]], dtype=torch.float32).to(device)
            
            # Inference
            mu, sigma = model(tabular_seq, image_tensor, future_clearsky, gate_features)
            mu_np = mu.cpu().numpy()[0]
            
            # De-normalize
            re_estimated_ghi = float(np.clip(mu_np[idx] * ghi_std + ghi_mean, 0, None))
            
            diff = abs(original_ghi - re_estimated_ghi)
            error_pct = (diff / max(original_ghi, 1)) * 100
            
            print(f"  {f['horizon']:<8} {original_ghi:>10.1f}  {re_estimated_ghi:>12.1f}  "
                  f"{diff:>8.1f}  {error_pct:>8.1f}%")
            
            results.append({'original': original_ghi, 're_est': re_estimated_ghi, 'diff': diff})
    
    print("-" * 70)
    
    if results:
        mae = np.mean([r['diff'] for r in results])
        rmse = np.sqrt(np.mean([r['diff']**2 for r in results]))
        print(f"\n  📊 Statistics (n={len(results)}):")
        print(f"     MAE:  {mae:.1f} W/m²")
        print(f"     RMSE: {rmse:.1f} W/m²")
    
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
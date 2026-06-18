"""
model.py — Shared model definition and inference helpers
=========================================================
Single source of truth for PhysicsGatedFusionModel and all
helper functions used by predict.py and verify.py.
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
import pvlib

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
# MODEL
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
    def __init__(self, pretrained_encoder_path: str = None):
        """
        Parameters
        ----------
        pretrained_encoder_path : str, optional
            Path to swimseg_encoder.pt produced by swimseg_pretrain.py.
            When provided, loads cloud-segmentation-pretrained weights into
            the backbone before fusion model training begins.
        """
        super().__init__()
        import timm
        base = timm.create_model("efficientnet_b2", pretrained=False, features_only=True)
        self.backbone     = base
        self.img_channels = 352

        if pretrained_encoder_path and os.path.exists(pretrained_encoder_path):
            state = torch.load(pretrained_encoder_path, map_location="cpu", weights_only=True)
            missing, unexpected = self.backbone.load_state_dict(state, strict=False)
            print(f"  ✓ Loaded SwimSeg pretrained encoder from {pretrained_encoder_path}")
            if missing:
                print(f"    Missing keys  : {len(missing)}")
            if unexpected:
                print(f"    Unexpected keys: {len(unexpected)}")
        elif pretrained_encoder_path:
            print(f"  ⚠️  Pretrained encoder not found at {pretrained_encoder_path} — using random init")

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
        attn = torch.softmax((Q @ K.transpose(-2, -1)) * self.scale, dim=-1)
        return (attn @ V).squeeze(1), attn.squeeze(1)


class PhysicsGatedFusionModel(nn.Module):
    def __init__(self, pretrained_encoder_path: str = None):
        """
        Parameters
        ----------
        pretrained_encoder_path : str, optional
            Path to SwimSeg-pretrained encoder weights (swimseg_encoder.pt).
            Pass this when training to initialise the CNN with cloud-aware weights.
        """
        super().__init__()
        self.temporal     = BiLSTMEncoder(input_dim=len(TABULAR_COLS))
        self.image_enc    = EfficientNetB2Encoder(pretrained_encoder_path)
        self.temp_dim     = self.temporal.out_dim
        self.img_channels = self.image_enc.img_channels
        self.D            = 256

        self.cross_attn = SimplifiedCrossAttention(self.temp_dim, self.img_channels, self.D)
        self.gate = nn.Sequential(
            nn.Linear(2, 16), nn.ReLU(),
            nn.Linear(16, 1), nn.Sigmoid(),
        )
        self.enrich = nn.Sequential(
            nn.Linear(self.D + 3, self.D), nn.ReLU(), nn.Dropout(0.15),
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
        img_features = smap.view(B, C, H * W).transpose(1, 2)
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


def load_model(model_path: str, device: torch.device,
               pretrained_encoder_path: str = None) -> PhysicsGatedFusionModel:
    """
    Load PhysicsGatedFusionModel weights from disk.

    Parameters
    ----------
    pretrained_encoder_path : str, optional
        Only used when loading a model for training (not inference).
        For inference, the saved checkpoint already contains trained weights.
    """
    model = PhysicsGatedFusionModel(pretrained_encoder_path).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    return model


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def load_weather_from_json(path: str) -> dict:
    """Parse weather_current.json into flat feature dict."""
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
        "cloud_cover":          avg_readings("cloud-cover"),
    }


def load_satellite_image(path: str) -> torch.Tensor:
    """Load himawari_current.png, resize and normalise for model input."""
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


def compute_clearsky_ghi(dt_sgt) -> float:
    """Compute GHI clearsky using pvlib for Singapore."""
    loc   = pvlib.location.Location(1.3521, 103.8198, tz="Asia/Singapore")
    times = pd.DatetimeIndex([dt_sgt])
    cs    = loc.get_clearsky(times)
    return float(cs["ghi"].iloc[0])


def build_lookback_window(
    df: pd.DataFrame,
    train_stats: dict,
    reference_time,
) -> torch.Tensor:
    """
    Build a normalised [1, WINDOW_SIZE, n_features] tensor from a pre-loaded
    historical DataFrame up to reference_time.
    """
    df = df.copy()
    df["clearsky_ratio"] = df["ghi"] / (df["ghi_clearsky"] + 1e-6)
    df["hour"]           = df["timestamp"].dt.hour
    df["month"]          = df["timestamp"].dt.month
    df["sin_hour"]       = np.sin(2 * np.pi * df["hour"] / 24)
    df["cos_hour"]       = np.cos(2 * np.pi * df["hour"] / 24)
    df["sin_month"]      = np.sin(2 * np.pi * df["month"] / 12)
    df["cos_month"]      = np.cos(2 * np.pi * df["month"] / 12)
    df["ghi_lag1"]       = df["ghi"].shift(1).fillna(0)

    ref_naive = pd.Timestamp(reference_time).replace(tzinfo=None)
    past      = df[df["timestamp"] <= ref_naive].tail(WINDOW_SIZE)

    if len(past) < WINDOW_SIZE:
        print(f"  ⚠️  Only {len(past)} historical rows — padding with zeros")
        pad = pd.DataFrame(
            np.zeros((WINDOW_SIZE - len(past), len(TABULAR_COLS))),
            columns=TABULAR_COLS,
        )
        tab = pd.concat([pad, past[TABULAR_COLS]], ignore_index=True)
    else:
        tab = past[TABULAR_COLS].copy()

    mean = np.array(train_stats["mean"], dtype=np.float32)
    std  = np.array(train_stats["std"],  dtype=np.float32)
    arr  = (tab.values.astype(np.float32) - mean) / std

    return torch.from_numpy(arr).float().unsqueeze(0)  # [1, 24, features]


def compute_gate_features(
    df: pd.DataFrame,
    reference_time,
) -> torch.Tensor:
    """Compute [clearsky_ratio, cloud_cover_normed] gate features at reference_time."""
    ref_naive = pd.Timestamp(reference_time).replace(tzinfo=None)
    row       = df[df["timestamp"] <= ref_naive].tail(1)
    ghi_last  = float(row["ghi"].iloc[0]) if len(row) else 400.0
    ghi_cs    = compute_clearsky_ghi(reference_time)
    cr        = float(np.clip(ghi_last / (ghi_cs + 1e-6), 0, 1.5))
    cc        = float(np.clip((1 - cr) * 100, 0, 100))
    return torch.tensor([[cr, cc / 100.0]], dtype=torch.float32)


def load_historical_df(csv_path: str) -> pd.DataFrame:
    """Load and sort the historical CSV once for reuse across helper calls."""
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)

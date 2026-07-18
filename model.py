"""
model.py — Shared model definition and inference helpers
=========================================================
Single source of truth for the model architecture and all inference
helpers. predict.py and verify.py import from here — do not duplicate.

Two architectures live here:

  PhysicsGatedFusionModel   (v1) — DEPLOYED. Matches the trained
      checkpoints on disk (best_model*.pt): single EfficientNetB2
      image encoder, simplified cross-attention, 2-input physics gate.

  PhysicsGatedFusionModelV2 (v2) — trained by solar_pv_main.ipynb.
      Dual CNN branches (global + RoI), 8-head cross-attention over 196
      patches, 4-input gate with optical flow.

load_model() auto-detects which architecture a checkpoint file contains,
so predict.py / verify.py work with either — including new checkpoints
saved by the notebook after retraining.
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import pvlib
import timm

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
FLOW_SIZE   = 64          # resolution used when computing optical flow (v2)
ZEROS_IMG   = torch.zeros((3, 224, 224), dtype=torch.float32)

SG_LAT, SG_LON = 1.3521, 103.8198


# ══════════════════════════════════════════════════════════════════════════════
# SHARED BUILDING BLOCKS
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
    """
    EfficientNetB2 feature extractor (last stage: 352 ch, 7x7).
    Optionally initialised from a SwimSeg-pretrained encoder checkpoint.
    """
    def __init__(self, pretrained_encoder_path: str = None,
                 imagenet_fallback: bool = False):
        super().__init__()
        use_imagenet = imagenet_fallback and (
            pretrained_encoder_path is None or
            not os.path.exists(pretrained_encoder_path))
        self.backbone     = timm.create_model("efficientnet_b2",
                                              pretrained=use_imagenet,
                                              features_only=True)
        self.img_channels = 352

        if (pretrained_encoder_path is not None and
                os.path.exists(pretrained_encoder_path)):
            state = torch.load(pretrained_encoder_path, map_location="cpu",
                               weights_only=True)
            missing, unexpected = self.backbone.load_state_dict(state, strict=False)
            print(f"  ✓ SwimSeg encoder loaded ({len(missing)} missing, "
                  f"{len(unexpected)} unexpected)")

    def forward(self, x):
        return self.backbone(x)[-1]   # (B, 352, 7, 7)


# ══════════════════════════════════════════════════════════════════════════════
# V1 MODEL — matches the trained checkpoints (best_model*.pt)
# ══════════════════════════════════════════════════════════════════════════════

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
    """
    Physics-Gated Cross-Modal Attention Fusion (v1 — deployed).

    Inputs
    ------
    tabular_seq    : (B, 24, 11)
    image          : (B, 3, 224, 224)
    future_clearsky: (B, 3)
    gate_features  : (B, 2)  [clearsky_ratio, cloud_cover]
    """
    def __init__(self):
        super().__init__()
        self.temporal     = BiLSTMEncoder(input_dim=len(TABULAR_COLS))
        self.image_enc    = EfficientNetB2Encoder()
        self.temp_dim     = self.temporal.out_dim
        self.img_channels = self.image_enc.img_channels
        self.D            = 256

        self.cross_attn = SimplifiedCrossAttention(self.temp_dim,
                                                   self.img_channels, self.D)
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
        img_features = smap.view(B, C, H * W).transpose(1, 2)
        H_a, _  = self.cross_attn(H_t, img_features, img_features)
        alpha   = self.gate(gate_features)
        pad     = H_t[:, :self.D] if self.temp_dim >= self.D else \
                  F.pad(H_t, (0, self.D - self.temp_dim))
        fused   = self.enrich(torch.cat([alpha * pad + (1 - alpha) * H_a,
                                         future_clearsky], dim=1))
        out     = self.head(fused)
        mu      = out[:, :3]
        sigma   = F.softplus(out[:, 3:]) + 1e-4
        return mu, sigma


def load_model(model_path: str, device: torch.device) -> nn.Module:
    """
    Load a trained checkpoint ready for inference.
    Auto-detects the architecture from the checkpoint keys:
      - 'global_enc.*' present  → v2 (PhysicsGatedFusionModelV2)
      - otherwise               → v1 (PhysicsGatedFusionModel)
    """
    state = torch.load(model_path, map_location=device, weights_only=True)
    is_v2 = any(k.startswith("global_enc.") for k in state)
    model = (PhysicsGatedFusionModelV2() if is_v2
             else PhysicsGatedFusionModel()).to(device)
    model.load_state_dict(state)
    model.eval()
    print(f"  ✓ Detected {'v2' if is_v2 else 'v1'} architecture")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# V2 MODEL — experimental, no trained checkpoint yet
# ══════════════════════════════════════════════════════════════════════════════

class MultiHeadCrossAttention(nn.Module):
    """
    8-head cross-attention.
    Query = BiLSTM temporal summary.
    Keys/values = 196 image patches (3 global frames x 49 + 1 RoI x 49).
    """
    def __init__(self, query_dim, kv_dim, hidden_dim=256, num_heads=8):
        super().__init__()
        self.q_proj  = nn.Linear(query_dim, hidden_dim)
        self.kv_proj = nn.Linear(kv_dim,    hidden_dim)
        self.attn    = nn.MultiheadAttention(hidden_dim, num_heads,
                                             dropout=0.1, batch_first=True)
        self.out_dim = hidden_dim

    def forward(self, query, kv):
        Q      = self.q_proj(query).unsqueeze(1)   # (B, 1, D)
        KV     = self.kv_proj(kv)                  # (B, N, D)
        out, w = self.attn(Q, KV, KV)              # (B,1,D), (B,1,N)
        return out.squeeze(1), w.squeeze(1)


class PhysicsGatedFusionModelV2(nn.Module):
    """
    Physics-Gated Cross-Modal Attention Fusion (v2 — experimental).

    Inputs
    ------
    tabular_seq    : (B, 24, 11)
    multi_frame    : (B, 9, 224, 224)   frames at t, t-1h, t-2h stacked
    roi_image      : (B, 3, 224, 224)   centre crop of current frame
    future_clearsky: (B, 3)
    gate_features  : (B, 4)  [clearsky_ratio, cloud_cover, flow_vx, flow_vy]
    """
    def __init__(self, ablation=None, pretrained_encoder_path=None,
                 imagenet_init=False):
        # imagenet_init: only set True when training from scratch; leave False
        # when loading a full checkpoint (avoids a pointless weight download).
        super().__init__()
        self.ablation = ablation
        input_dim = len(TABULAR_COLS)

        self.temporal = BiLSTMEncoder(input_dim=input_dim)
        self.temp_dim = self.temporal.out_dim

        self.global_enc = EfficientNetB2Encoder(pretrained_encoder_path,
                                                imagenet_fallback=imagenet_init)
        self.roi_enc    = EfficientNetB2Encoder(pretrained_encoder_path,
                                                imagenet_fallback=imagenet_init)
        self.img_ch     = 352
        self.D          = 256

        self.cross_attn = MultiHeadCrossAttention(self.temp_dim, self.img_ch,
                                                  self.D, num_heads=8)
        self.gate = nn.Sequential(
            nn.Linear(4, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid()
        )
        in_dim = self.D if ablation != "concat" else self.D * 2
        self.enrich = nn.Sequential(
            nn.Linear(in_dim + 3, self.D), nn.ReLU(), nn.Dropout(0.15)
        )
        self.head = nn.Sequential(
            nn.Linear(self.D, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(128, 64),    nn.LayerNorm(64),  nn.GELU(),
            nn.Linear(64, 6),
        )
        self.freeze_cnn()

    def freeze_cnn(self):
        for enc in [self.global_enc, self.roi_enc]:
            for p in enc.parameters():
                p.requires_grad = False

    def unfreeze_cnn(self):
        for enc in [self.global_enc, self.roi_enc]:
            for p in enc.parameters():
                p.requires_grad = True

    def _image_patches(self, multi_frame, roi_image):
        B = multi_frame.shape[0]
        def to_patches(feat):
            _, C, H, W = feat.shape
            return feat.view(B, C, H * W).transpose(1, 2)
        f0 = self.global_enc(multi_frame[:, 0:3])
        f1 = self.global_enc(multi_frame[:, 3:6])
        f2 = self.global_enc(multi_frame[:, 6:9])
        fr = self.roi_enc(roi_image)
        return torch.cat([to_patches(f0), to_patches(f1),
                          to_patches(f2), to_patches(fr)], dim=1)  # (B,196,352)

    def forward(self, tabular_seq, multi_frame, roi_image,
                future_clearsky, gate_features, return_attn=False):
        H_t      = self.temporal(tabular_seq)
        patches  = self._image_patches(multi_frame, roi_image)
        H_a, aw  = self.cross_attn(H_t, patches)
        alpha    = self.gate(gate_features)

        pad = (H_t[:, :self.D] if self.temp_dim >= self.D
               else F.pad(H_t, (0, self.D - self.temp_dim)))

        if self.ablation == "lstm":
            fused = self.enrich(torch.cat([pad, future_clearsky], dim=1))
        elif self.ablation == "cnn":
            fused = self.enrich(torch.cat([H_a, future_clearsky], dim=1))
        elif self.ablation == "concat":
            fused = self.enrich(torch.cat([pad, H_a, future_clearsky], dim=1))
        else:
            fused = self.enrich(torch.cat(
                [alpha * pad + (1 - alpha) * H_a, future_clearsky], dim=1))

        out   = self.head(fused)
        mu    = out[:, :3]
        sigma = F.softplus(out[:, 3:]) + 1e-4

        if return_attn:
            return mu, sigma, alpha, aw
        return mu, sigma


# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE HELPERS (shared by predict.py and verify.py)
# ══════════════════════════════════════════════════════════════════════════════

def load_weather_from_json(path: str) -> dict:
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


def load_image_tensor(path: str) -> torch.Tensor:
    """Load a PNG/NPY satellite image as a normalised (3,224,224) tensor."""
    if not path or not os.path.exists(path):
        print(f"  ⚠️  Satellite image not found at {path} — using zeros")
        return ZEROS_IMG.clone()
    try:
        if path.endswith(".npy"):
            arr = np.load(path).astype(np.float32)
        else:
            img = Image.open(path).convert("RGB").resize((224, 224), Image.BILINEAR)
            arr = np.array(img, dtype=np.float32) / 255.0
        arr = (arr - IMG_MEAN) / IMG_STD
        return torch.from_numpy(arr.transpose(2, 0, 1)).float()
    except Exception as e:
        print(f"  ⚠️  Could not load {path}: {e} — using zeros")
        return ZEROS_IMG.clone()


def compute_clearsky_ghi(dt_sgt) -> float:
    loc   = pvlib.location.Location(SG_LAT, SG_LON, tz="Asia/Singapore")
    times = pd.DatetimeIndex([dt_sgt])
    cs    = loc.get_clearsky(times)
    return float(cs["ghi"].iloc[0])


def load_historical_df(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


def _add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["clearsky_ratio"] = df["ghi"] / (df["ghi_clearsky"] + 1e-6)
    df["hour"]           = df["timestamp"].dt.hour
    df["month"]          = df["timestamp"].dt.month
    df["sin_hour"]       = np.sin(2 * np.pi * df["hour"] / 24)
    df["cos_hour"]       = np.cos(2 * np.pi * df["hour"] / 24)
    df["sin_month"]      = np.sin(2 * np.pi * df["month"] / 12)
    df["cos_month"]      = np.cos(2 * np.pi * df["month"] / 12)
    df["ghi_lag1"]       = df["ghi"].shift(1).fillna(0)
    return df


def build_lookback_window(df: pd.DataFrame, train_stats: dict,
                          reference_time) -> torch.Tensor:
    """Build the normalised (1, 24, 11) lookback tensor ending at reference_time."""
    df = _add_engineered_features(df)

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
    return torch.from_numpy(arr).float().unsqueeze(0)   # (1, 24, 11)


def compute_gate_features(df: pd.DataFrame, reference_time) -> torch.Tensor:
    """
    Returns the (1, 2) gate tensor [clearsky_ratio, cloud_cover] evaluated
    at reference_time (uses the last CSV row at or before that time).
    """
    ref_naive = pd.Timestamp(reference_time).replace(tzinfo=None)
    row       = df[df["timestamp"] <= ref_naive].tail(1)
    ghi_last  = float(row["ghi"].iloc[0]) if len(row) else 400.0
    ghi_cs    = compute_clearsky_ghi(reference_time)
    cr        = float(np.clip(ghi_last / (ghi_cs + 1e-6), 0, 1.5))
    cc        = float(np.clip((1 - cr) * 100, 0, 100))
    return torch.tensor([[cr, cc / 100.0]], dtype=torch.float32)


def _compute_optical_flow(frame_prev: torch.Tensor,
                          frame_curr: torch.Tensor) -> torch.Tensor:
    """Mean optical flow in the centre RoI. Returns (vx, vy). Needs opencv."""
    try:
        import cv2

        def to_gray(t):
            arr = t[0].numpy()
            arr = np.clip(arr * 0.229 + 0.485, 0, 1)
            arr = (arr * 255).astype(np.uint8)
            return cv2.resize(arr, (FLOW_SIZE, FLOW_SIZE),
                              interpolation=cv2.INTER_LINEAR)

        g0   = to_gray(frame_prev)
        g1   = to_gray(frame_curr)
        flow = cv2.calcOpticalFlowFarneback(
            g0, g1, None,
            pyr_scale=0.5, levels=3, winsize=11,
            iterations=3, poly_n=5, poly_sigma=1.1, flags=0
        )
        c  = slice(FLOW_SIZE // 4, 3 * FLOW_SIZE // 4)
        vx = float(flow[c, c, 0].mean()) / FLOW_SIZE
        vy = float(flow[c, c, 1].mean()) / FLOW_SIZE
    except Exception:
        vx, vy = 0.0, 0.0
    return torch.tensor([vx, vy], dtype=torch.float32)


def load_satellite_inputs(current_path: str,
                          prev1_path: str = None,
                          prev2_path: str = None):
    """
    Load v2 image inputs: multi-frame stack + RoI crop + optical flow.
    Missing previous frames are zero-filled.

    Returns
    -------
    multi_frame : (1, 9, 224, 224)
    roi_image   : (1, 3, 224, 224)
    flow        : (2,)
    """
    frame_t  = load_image_tensor(current_path)
    frame_t1 = (load_image_tensor(prev1_path)
                if prev1_path and os.path.exists(prev1_path)
                else ZEROS_IMG.clone())
    frame_t2 = (load_image_tensor(prev2_path)
                if prev2_path and os.path.exists(prev2_path)
                else ZEROS_IMG.clone())

    multi_frame = torch.cat([frame_t, frame_t1, frame_t2], dim=0).unsqueeze(0)

    roi = frame_t[:, 56:168, 56:168]
    roi = F.interpolate(roi.unsqueeze(0), size=(224, 224),
                        mode="bilinear", align_corners=False)

    flow = _compute_optical_flow(frame_t1, frame_t)
    return multi_frame, roi, flow


def compute_gate_features_v2(df: pd.DataFrame, reference_time,
                             flow: torch.Tensor) -> torch.Tensor:
    """(1, 4) gate tensor for v2: [clearsky_ratio, cloud_cover, vx, vy]."""
    base = compute_gate_features(df, reference_time)   # (1, 2)
    return torch.cat([base, flow.unsqueeze(0)], dim=1)


def run_model(model, tabular_seq, sat_path, future_clearsky,
              df, reference_time, device, prev_paths=(None, None)):
    """
    Run inference with either architecture. Builds the correct image and
    gate inputs based on the model version returned by load_model().
    Returns (mu, sigma) — normalised.
    """
    tabular_seq     = tabular_seq.to(device)
    future_clearsky = future_clearsky.to(device)

    with torch.no_grad():
        if isinstance(model, PhysicsGatedFusionModelV2):
            multi, roi, flow = load_satellite_inputs(sat_path, *prev_paths)
            gate = compute_gate_features_v2(df, reference_time, flow)
            mu, sigma = model(tabular_seq, multi.to(device), roi.to(device),
                              future_clearsky, gate.to(device))
        else:
            image = load_image_tensor(sat_path).unsqueeze(0).to(device)
            gate  = compute_gate_features(df, reference_time).to(device)
            mu, sigma = model(tabular_seq, image, future_clearsky, gate)
    return mu, sigma


def denormalise_forecast(mu_np, sigma_np, train_stats: dict):
    """Convert normalised model outputs to W/m² with a 90% CI."""
    ghi_mean = float(train_stats["ghi_mean"])
    ghi_std  = float(train_stats["ghi_std"])
    mu_real  = np.clip(mu_np * ghi_std + ghi_mean, 0, None)
    sig_real = sigma_np * ghi_std
    lo       = np.clip(mu_real - 1.645 * sig_real, 0, None)
    hi       = mu_real + 1.645 * sig_real
    return mu_real, lo, hi

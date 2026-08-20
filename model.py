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
    def __init__(self, input_dim=len(TABULAR_COLS), hidden_dim=128):
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
    EfficientNet feature extractor (last stage only).
    Channel count is read from the backbone, never hardcoded:
        efficientnet_b2 → 352 ch,  efficientnet_b0 → 320 ch  (both 7x7)
    Optionally initialised from a SwimSeg-pretrained encoder checkpoint.
    """
    def __init__(self, pretrained_encoder_path: str = None,
                 imagenet_fallback: bool = False,
                 backbone: str = "efficientnet_b2"):
        super().__init__()
        use_imagenet = imagenet_fallback and (
            pretrained_encoder_path is None or
            not os.path.exists(pretrained_encoder_path))
        self.backbone     = timm.create_model(backbone,
                                              pretrained=use_imagenet,
                                              features_only=True)
        self.img_channels = self.backbone.feature_info.channels()[-1]

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


def save_checkpoint(model: nn.Module, path: str):
    """
    Save weights plus a sidecar `<path>.config.json` describing the
    architecture. Needed because num_heads cannot be recovered from tensor
    shapes — loading with the wrong head count silently produces garbage
    rather than raising an error.
    """
    torch.save(model.state_dict(), path)
    cfg = getattr(model, "cfg", {"version": "v1"})
    with open(path + ".config.json", "w") as f:
        json.dump(cfg, f, indent=2)


def load_model(model_path: str, device: torch.device) -> nn.Module:
    """
    Load a trained checkpoint ready for inference.

    Architecture resolution order:
      1. `<model_path>.config.json` sidecar (written by save_checkpoint)
      2. Inferred from tensor shapes (assumes num_heads=8 — warns)
    v1 vs v2 is detected from the presence of 'global_enc.*' keys.
    """
    state = torch.load(model_path, map_location=device, weights_only=True)
    is_v2 = any(k.startswith("global_enc.") for k in state)

    if not is_v2:
        model = PhysicsGatedFusionModel().to(device)
        model.load_state_dict(state)
        model.eval()
        print("  ✓ Detected v1 architecture")
        return model

    cfg_path = model_path + ".config.json"
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
        kwargs = {k: cfg[k] for k in
                  ("backbone", "lstm_hidden", "num_heads", "hidden_dim",
                   "dropout", "n_tabular", "ablation") if k in cfg}
        print(f"  ✓ Detected v2 architecture (config: {cfg.get('backbone')}, "
              f"D={cfg.get('hidden_dim')}, heads={cfg.get('num_heads')}, "
              f"lstm={cfg.get('lstm_hidden')})")
    else:
        # Infer what the shapes allow; num_heads is unknowable → assume 8.
        D, img_ch   = state["cross_attn.kv_proj.weight"].shape
        ih          = state["temporal.lstm.weight_ih_l0"].shape
        kwargs = {
            "hidden_dim":  int(D),
            "lstm_hidden": int(ih[0] // 4),
            "n_tabular":   int(ih[1]),
            "backbone":    "efficientnet_b0" if int(img_ch) == 320
                           else "efficientnet_b2",
            "num_heads":   8,
        }
        print(f"  ✓ Detected v2 architecture (inferred: {kwargs['backbone']}, "
              f"D={kwargs['hidden_dim']}, lstm={kwargs['lstm_hidden']})")
        print(f"  ⚠️  No {os.path.basename(cfg_path)} — assuming num_heads=8. "
              f"If this model was trained with a different head count, its "
              f"predictions will be wrong. Re-save with save_checkpoint().")

    model = PhysicsGatedFusionModelV2(**kwargs).to(device)
    model.load_state_dict(state)
    model.eval()
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
                 imagenet_init=False, backbone="efficientnet_b2",
                 lstm_hidden=128, num_heads=8, hidden_dim=256,
                 dropout=0.15, n_tabular=None):
        """
        imagenet_init : True only when training from scratch; False when
                        loading a full checkpoint (skips a weight download).
        backbone      : "efficientnet_b2" (default) or "efficientnet_b0"
        lstm_hidden   : BiLSTM hidden size per direction (128 → 64 to shrink)
        num_heads     : cross-attention heads (8 → 4 to shrink)
        hidden_dim    : cross-attention / fusion width D (256 → 128 to shrink)
        n_tabular     : number of tabular features; defaults to len(TABULAR_COLS).
                        Set explicitly when ablating features (e.g. dropping
                        ghi_lag1) so the LSTM input width matches the data.

        DEFAULTS REPRODUCE THE ORIGINAL ARCHITECTURE EXACTLY, so existing
        checkpoints keep loading. Override them to train a smaller model.
        """
        super().__init__()
        self.ablation = ablation
        input_dim = n_tabular if n_tabular is not None else len(TABULAR_COLS)

        self.temporal = BiLSTMEncoder(input_dim=input_dim,
                                      hidden_dim=lstm_hidden)
        self.temp_dim = self.temporal.out_dim

        self.global_enc = EfficientNetB2Encoder(pretrained_encoder_path,
                                                imagenet_fallback=imagenet_init,
                                                backbone=backbone)
        self.roi_enc    = EfficientNetB2Encoder(pretrained_encoder_path,
                                                imagenet_fallback=imagenet_init,
                                                backbone=backbone)
        self.img_ch     = self.global_enc.img_channels
        self.D          = hidden_dim

        self.cross_attn = MultiHeadCrossAttention(self.temp_dim, self.img_ch,
                                                  self.D, num_heads=num_heads)
        self.gate = nn.Sequential(
            nn.Linear(4, self.D // 8), nn.ReLU(),
            nn.Linear(self.D // 8, 1), nn.Sigmoid()
        )
        in_dim = self.D if ablation != "concat" else self.D * 2
        self.enrich = nn.Sequential(
            nn.Linear(in_dim + 3, self.D), nn.ReLU(), nn.Dropout(dropout)
        )
        h1, h2 = self.D // 2, self.D // 4
        self.head = nn.Sequential(
            nn.Linear(self.D, h1), nn.LayerNorm(h1), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(h1, h2),     nn.LayerNorm(h2), nn.GELU(),
            nn.Linear(h2, 6),
        )
        # Recorded so save_checkpoint() can write a sidecar config — num_heads
        # is NOT recoverable from tensor shapes, so it must be persisted.
        self.cfg = {
            "version": "v2", "ablation": ablation, "backbone": backbone,
            "lstm_hidden": lstm_hidden, "num_heads": num_heads,
            "hidden_dim": hidden_dim, "dropout": dropout,
            "n_tabular": input_dim,
        }
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


def compute_clearsky_hour_mean(dt_sgt) -> float:
    """
    Mean clearsky GHI over the hour PRECEDING dt_sgt.
    Open-Meteo labels each hourly value with the end of its averaging
    window (the value at 18:00 is the 17:00→18:00 mean), so physical
    caps on forecasts must use this convention — the instantaneous
    value is badly wrong near sunrise/sunset.
    """
    loc   = pvlib.location.Location(SG_LAT, SG_LON, tz="Asia/Singapore")
    end   = pd.Timestamp(dt_sgt)
    times = pd.date_range(end - pd.Timedelta(minutes=50), end, freq="10min")
    cs    = loc.get_clearsky(pd.DatetimeIndex(times))
    return float(cs["ghi"].mean())


def load_historical_df(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


def fetch_recent_df(past_days: int = 3) -> pd.DataFrame:
    """
    Fetch the most recent hourly weather + GHI from Open-Meteo's live API.
    The archive API lags ~5 days behind; this fills that gap so the
    lookback window ends at the current hour, not last week.
    Returns a DataFrame with the columns build_lookback_window needs,
    or None if the fetch fails.
    """
    import requests
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude":  SG_LAT,
        "longitude": SG_LON,
        "hourly": ("temperature_2m,relative_humidity_2m,rain,"
                   "wind_speed_10m,cloud_cover,shortwave_radiation"),
        "past_days": past_days,
        "forecast_days": 1,
        "timezone": "Asia/Singapore",
    }
    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        hourly = resp.json()["hourly"]
    except Exception as e:
        print(f"  ⚠️  Could not fetch recent hours from Open-Meteo: {e}")
        return None

    df = pd.DataFrame(hourly)
    df["timestamp"] = pd.to_datetime(df["time"])
    df = df.drop(columns=["time"]).rename(columns={"shortwave_radiation": "ghi"})

    # Only keep hours that have passed (later rows are forecasts, not data)
    now = pd.Timestamp.now(tz="Asia/Singapore").tz_localize(None).floor("h")
    df = df[df["timestamp"] <= now].reset_index(drop=True)

    # Clearsky GHI via pvlib (needed for clearsky_ratio feature)
    loc   = pvlib.location.Location(SG_LAT, SG_LON, tz="Asia/Singapore")
    times = pd.DatetimeIndex(df["timestamp"], tz="Asia/Singapore")
    df["ghi_clearsky"] = loc.get_clearsky(times)["ghi"].values
    return df


def extend_with_recent(df: pd.DataFrame) -> pd.DataFrame:
    """Append live recent hours to the historical CSV dataframe."""
    recent = fetch_recent_df()
    if recent is None or recent.empty:
        return df
    new = recent[recent["timestamp"] > df["timestamp"].max()]
    if new.empty:
        return df
    print(f"  ✓ Appended {len(new)} recent hours from live API "
          f"(lookback now ends {new['timestamp'].max()})")
    return (pd.concat([df, new], ignore_index=True)
            .sort_values("timestamp").reset_index(drop=True))


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


# ══════════════════════════════════════════════════════════════════════════════
# BASELINES — a model is only "good" if it beats these
# ══════════════════════════════════════════════════════════════════════════════

def persistence_forecast(ghi_now, horizons=3):
    """
    Persistence: GHI at t+h equals GHI at t.
    The standard reference for short-horizon irradiance forecasting — any
    model that cannot beat it has learned nothing useful.
    """
    ghi_now = np.asarray(ghi_now, dtype=np.float32)
    return np.repeat(ghi_now[:, None], horizons, axis=1)


def smart_persistence_forecast(ghi_now, cs_now, cs_future):
    """
    Smart (clear-sky) persistence: holds the CLEAR-SKY INDEX constant rather
    than raw GHI, so it accounts for the sun's known movement.
        ghi(t+h) = ghi(t)/cs(t) * cs(t+h)
    This is the honest baseline for a solar model — plain persistence flatters
    you at midday and punishes you near sunrise/sunset.

    ghi_now, cs_now : (N,)      cs_future : (N, H)
    """
    ghi_now   = np.asarray(ghi_now,   dtype=np.float32)
    cs_now    = np.asarray(cs_now,    dtype=np.float32)
    cs_future = np.asarray(cs_future, dtype=np.float32)
    k = np.clip(ghi_now / (cs_now + 1e-6), 0, 1.5)     # clear-sky index
    return np.clip(k[:, None] * cs_future, 0, None)


def skill_score(mae_model: float, mae_reference: float) -> float:
    """
    Forecast skill vs a reference baseline.
      > 0 : better than the baseline (1.0 = perfect)
      = 0 : no better than the baseline
      < 0 : WORSE than the baseline
    """
    if mae_reference <= 0:
        return float("nan")
    return 1.0 - (mae_model / mae_reference)


def denormalise_forecast(mu_np, sigma_np, train_stats: dict):
    """Convert normalised model outputs to W/m² with a 90% CI."""
    ghi_mean = float(train_stats["ghi_mean"])
    ghi_std  = float(train_stats["ghi_std"])
    mu_real  = np.clip(mu_np * ghi_std + ghi_mean, 0, None)
    sig_real = sigma_np * ghi_std
    lo       = np.clip(mu_real - 1.645 * sig_real, 0, None)
    hi       = mu_real + 1.645 * sig_real
    return mu_real, lo, hi

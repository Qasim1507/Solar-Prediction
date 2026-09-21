"""
train.py — v3 training: k_t target, 14 features, WinnerModel, MAE-selected.

    python train.py --seed 42 --config winner.yaml

Differences from the v2 notebook run, in the order they matter:

  1. Target is k_t = GHI / clearsky, not GHI. The diurnal cycle dominates the
     GHI loss, so most of the gradient goes into re-learning the time of day.
  2. The NLL is weighted by future clear-sky, so a k_t error at noon counts for
     more than the same error near sunset. Unweighted, the model over-fits the
     rare large-k_t samples that occur when clearsky is small.
  3. 14 tabular features, matching what the tabular baselines get. The deep
     model previously ran on 11 and was losing to LightGBM on that handicap.
  4. Checkpoints are selected on validation MAE, not validation NLL. The metric
     being reported is MAE; selecting on NLL optimises a different thing.
  5. bf16 rather than fp16 — the v2 run had to skip NaN batches.
  6. The CNN stays frozen. The v2 validation curve turned upward at the epoch-20
     unfreeze while train NLL kept falling.
"""
import argparse, json, math, os, sys, time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import WinnerModel, TABULAR_COLS_V3
from combined_dataset import FRAME_OFFSETS_MINUTES

ROOT = os.path.dirname(os.path.abspath(__file__))
SAT_DIR = os.path.join(ROOT, "data/satellite_aws")
NPY_DIR = os.path.join(ROOT, "data/satellite_aws_npy")
FRAME_COLS = ["image_path"] + [f"image_path_prev{n}"
                               for n in range(1, len(FRAME_OFFSETS_MINUTES))]
WINDOW = 24
IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMG_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
ZEROS = torch.zeros((3, 224, 224), dtype=torch.float32)


def npy_for(p):
    rel = os.path.relpath(os.path.join(ROOT, p), SAT_DIR)
    return os.path.join(NPY_DIR, rel).replace(".png", ".npy")


_cache = {}


def load_frame(p, cache_limit=4000):
    if not isinstance(p, str):
        return ZEROS
    if p in _cache:
        return _cache[p]
    f = npy_for(p)
    if not os.path.exists(f):
        return ZEROS
    a = (np.load(f).astype(np.float32) - IMG_MEAN) / IMG_STD
    t = torch.from_numpy(a.transpose(2, 0, 1)).float()
    if len(_cache) < cache_limit:
        _cache[p] = t
    return t


def engineer(df):
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["clearsky_ratio"] = df["ghi"] / (df["ghi_clearsky"] + 1e-6)
    df["hour"] = df["timestamp"].dt.hour
    df["month"] = df["timestamp"].dt.month
    df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12)
    df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12)
    # Lags are TIME-indexed, not positional: the frame is daylight-only, so a
    # positional shift would reach back to the previous evening across the gap.
    ts = df.set_index("timestamp")
    for L in (1, 2, 3):
        df[f"ghi_lag{L}"] = ts["ghi"].reindex(
            df["timestamp"] - pd.Timedelta(hours=L)).values
    for c in ("ghi_lag1", "ghi_lag2", "ghi_lag3"):
        df[c] = df[c].ffill().fillna(df["ghi"])
    return df


class KtDataset(Dataset):
    """Windows of 24 daylight hours -> k_t at t+1, t+2, t+3."""

    def __init__(self, df, is_train, stats=None):
        d = df.reset_index(drop=True)
        self.tab = d[TABULAR_COLS_V3].values.astype(np.float32)
        self.ghi = d["ghi"].values.astype(np.float32)
        self.cs = d["ghi_clearsky"].values.astype(np.float32)
        self.frames = [d[c].values for c in FRAME_COLS]
        self.ts = d["timestamp"].values

        def contiguous(i):
            return all((self.ts[i + k] - self.ts[i]) / np.timedelta64(1, "h") == k
                       for k in (1, 2, 3))

        self.idx = [i for i in range(WINDOW - 1, len(d) - 3)
                    if isinstance(d["image_path"].iloc[i], str)
                    and contiguous(i)]

        kt_all = np.clip(self.ghi / (self.cs + 1e-6), 0.0, 1.5)
        if is_train:
            self.mean = self.tab.mean(0)
            self.std = self.tab.std(0) + 1e-6
            w = np.array(self.idx)
            kt_tr = np.stack([kt_all[w + h] for h in (1, 2, 3)], 1)
            self.kt_mean = float(kt_tr.mean())
            self.kt_std = float(kt_tr.std() + 1e-6)
        else:
            self.mean = np.asarray(stats["mean"], dtype=np.float32)
            self.std = np.asarray(stats["std"], dtype=np.float32)
            self.kt_mean, self.kt_std = stats["kt_mean"], stats["kt_std"]
        self.tabn = (self.tab - self.mean) / self.std
        self.kt = kt_all

    def stats(self):
        return {"mean": self.mean.tolist(), "std": self.std.tolist(),
                "kt_mean": self.kt_mean, "kt_std": self.kt_std,
                "features": TABULAR_COLS_V3}

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, k):
        t = self.idx[k]
        cs_f = self.cs[t + 1:t + 4]
        kt = self.kt[t + 1:t + 4]
        return {
            "tab": torch.from_numpy(self.tabn[t - WINDOW + 1:t + 1]),
            "frame_t": load_frame(self.frames[0][t]),
            "frame_t1": load_frame(self.frames[1][t]),
            "frame_t2": load_frame(self.frames[2][t]),
            "roi": F.interpolate(
                load_frame(self.frames[0][t])[:, 56:168, 56:168].unsqueeze(0),
                size=(224, 224), mode="bilinear", align_corners=False).squeeze(0),
            "cs_future": torch.from_numpy(cs_f.copy()),
            "kt_target": torch.from_numpy(
                ((kt - self.kt_mean) / self.kt_std).astype(np.float32)),
            "ghi_target": torch.from_numpy(self.ghi[t + 1:t + 4].copy()),
        }


def weighted_nll(mu, sigma, target, weight):
    """Gaussian NLL in k_t space, weighted by future clear-sky.

    Without the weight the model chases large k_t values, which occur exactly
    when clearsky is small and the resulting GHI error is negligible.
    """
    nll = torch.log(sigma) + (target - mu) ** 2 / (2 * sigma ** 2)
    return (nll * weight).sum() / weight.sum().clamp_min(1e-6)


def ghi_mae(mu_kt, batch, kt_mean, kt_std, kt_cap=1.15):
    kt = (mu_kt * kt_std + kt_mean).clamp(0.0, kt_cap)
    pred = kt * batch["cs_future"].to(kt.device)
    return (pred - batch["ghi_target"].to(kt.device)).abs()


def run_epoch(model, loader, dev, stats, opt=None, scaler=None, amp_dtype=None,
              clip=1.0):
    train = opt is not None
    model.train(train)
    tot_loss = tot_n = 0.0
    abs_err = []
    for b in loader:
        tab = b["tab"].to(dev)
        fr = [b[k].to(dev) for k in ("frame_t", "frame_t1", "frame_t2", "roi")]
        csf = b["cs_future"].to(dev)
        tgt = b["kt_target"].to(dev)
        w = csf / csf.mean().clamp_min(1e-6)
        ctx = (torch.autocast(dev.type, dtype=amp_dtype)
               if amp_dtype else torch.enable_grad() if train else torch.no_grad())
        with torch.set_grad_enabled(train):
            with ctx if amp_dtype else _null():
                mu, sigma = model(tab, fr[0], fr[1], fr[2], fr[3], csf)
                loss = weighted_nll(mu.float(), sigma.float(), tgt, w)
            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                opt.step()
        tot_loss += loss.item() * tgt.size(0)
        tot_n += tgt.size(0)
        abs_err.append(ghi_mae(mu.float().detach(), b,
                               stats["kt_mean"], stats["kt_std"]).cpu())
    return tot_loss / max(tot_n, 1), torch.cat(abs_err).mean().item()


class _null:
    def __enter__(self): return None
    def __exit__(self, *a): return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--config", default="winner.yaml")
    ap.add_argument("--csv", default="data/combined_dataset_v2.csv")
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    cfg = {}
    if os.path.exists(a.config):
        import yaml
        cfg = yaml.safe_load(open(a.config)) or {}
    g = lambda k, d: cfg.get(k, d)

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    import random; random.seed(a.seed)

    if a.device:
        dev = torch.device(a.device)
    elif torch.cuda.is_available():
        dev = torch.device("cuda")
    else:
        dev = torch.device("cpu")   # MPS is not trusted for this model
    print(f"device={dev} seed={a.seed} csv={a.csv}")

    df = pd.read_csv(a.csv, parse_dates=["timestamp"])
    df = engineer(df)
    df = df[df["image_path"].notna()].reset_index(drop=True)
    n = len(df); t_end = int(n * .70); v_end = int(n * .85)
    tr_df, va_df = df.iloc[:t_end], df.iloc[t_end:v_end]

    tr = KtDataset(tr_df, True)
    st = tr.stats()
    va = KtDataset(va_df, False, st)
    print(f"train={len(tr)} val={len(va)}  features={len(TABULAR_COLS_V3)}")

    bs = g("batch_size", 64)
    nw = g("num_workers", 0)
    trl = DataLoader(tr, batch_size=bs, shuffle=True, num_workers=nw)
    val = DataLoader(va, batch_size=bs, shuffle=False, num_workers=nw)

    model = WinnerModel(n_tab=len(TABULAR_COLS_V3),
                        hidden=g("hidden", 256),
                        lstm_hidden=g("lstm_hidden", 128),
                        dropout=g("dropout", 0.15),
                        freeze_cnn=g("freeze_cnn", True)).to(dev)
    ntr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {ntr:,}")

    epochs = g("epochs", 80)
    warm = g("warmup_epochs", 3)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=g("lr", 1e-3), weight_decay=g("weight_decay", 0.05))
    amp_dtype = torch.bfloat16 if (g("precision", "bf16") == "bf16"
                                   and dev.type == "cuda") else None

    def lr_at(e):
        if e < warm:
            return (e + 1) / max(warm, 1)
        p = (e - warm) / max(epochs - warm, 1)
        return 0.5 * (1 + math.cos(math.pi * p))

    out = a.out or f"best_model_v3_seed{a.seed}.pt"
    best_mae, best_ep = float("inf"), -1
    patience = g("early_stop", 15)
    hist = []
    print(f"\n{'Ep':>3} | {'tr NLL':>8} {'tr MAE':>8} | {'va NLL':>8} "
          f"{'va MAE':>8} | status")
    for e in range(epochs):
        for pg in opt.param_groups:
            pg["lr"] = g("lr", 1e-3) * lr_at(e)
        t0 = time.time()
        trl_loss, trl_mae = run_epoch(model, trl, dev, st, opt,
                                      amp_dtype=amp_dtype,
                                      clip=g("grad_clip", 1.0))
        vl_loss, vl_mae = run_epoch(model, val, dev, st, amp_dtype=amp_dtype)
        hist.append({"epoch": e + 1, "train_nll": trl_loss, "train_mae": trl_mae,
                     "val_nll": vl_loss, "val_mae": vl_mae})
        # SELECT ON VALIDATION MAE, not NLL — MAE is the reported metric.
        if vl_mae < best_mae:
            best_mae, best_ep = vl_mae, e
            torch.save(model.state_dict(), out)
            with open(out + ".stats.json", "w") as f:
                json.dump(st, f)
            status = "saved"
        else:
            status = f"patience {e - best_ep}/{patience}"
        print(f"{e+1:>3} | {trl_loss:>8.4f} {trl_mae:>7.1f}W | "
              f"{vl_loss:>8.4f} {vl_mae:>7.1f}W | {status}  {time.time()-t0:.0f}s")
        if e - best_ep >= patience:
            print(f"early stop at epoch {e+1}")
            break

    json.dump(hist, open(out + ".history.json", "w"), indent=1)
    print(f"\nbest val MAE {best_mae:.2f} W/m2 at epoch {best_ep+1} -> {out}")


if __name__ == "__main__":
    main()

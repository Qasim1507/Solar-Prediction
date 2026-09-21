"""
current_vs_tabular.py — deep models vs tabular baselines on EXACTLY the same rows.

Replaces repro_eval.py + tabular_same_rows.py for the current pipeline. Those two
were written against the August snapshot and have since diverged from training in
six ways: they read `combined_dataset.csv` from the repo root, point at
`data/satellite` / `data/satellite_npy` (the retired NICT cache), truncate to
`< 2026-08-16`, evaluate on MPS, build the frame series from ROW offsets (1 hour
apart, where training now uses the 10-minute `image_path_prev*` columns), and
convert frames to GRAYSCALE — which notebook cell 8 explicitly stopped doing for
the AWS composite, because its blue channel carries infrared.

Rather than patch six divergences in a stale reimplementation, this script builds
its state by executing the notebook's own cells, so the dataset, frame series,
normalisation and split are training's by construction and cannot drift.

Read-only w.r.t. the repository except for its own JSON output in analysis/.
"""
import json, os, sys, time
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from scipy import stats as sps
import lightgbm as lgb

os.environ.setdefault("MPLBACKEND", "Agg")
ROOT = "/Users/qasimnalawala/Desktop/Project"
os.chdir(ROOT)
sys.path.insert(0, ROOT)

OUT = "analysis/current_comparison.json"

# ── Build the notebook's state ────────────────────────────────────────────────
# Section 7 (training) is skipped; Section 8's ablation loop is disabled with
# RUN_ONLY = [] so it defines evaluate_model and the baselines without training.
CELLS = [2, 4, 6, 8, 9, 10, 12, 13, 14, 16, 17, 19, 21, 25]
nb = json.load(open("solar_pv_main.ipynb"))
G = {"__name__": "__main__"}
t0 = time.time()
for idx in CELLS:
    src = "".join(nb["cells"][idx]["source"])
    if idx == 25:
        assert "RUN_ONLY   = None" in src, "RUN_ONLY anchor missing"
        src = src.replace("RUN_ONLY   = None", "RUN_ONLY   = []")
    try:
        exec(compile(src, f"<cell {idx}>", "exec"), G)
    except Exception as e:
        # Cell 25's summary line raises on an empty Skill_vs_SP column; the
        # definitions above it are what we need and they have already run.
        if idx == 25:
            print(f"  (cell 25 summary raised {type(e).__name__}, expected)")
        else:
            raise
print(f"notebook state built in {time.time()-t0:.0f}s")

df_deep       = G["df_deep"]
train_dataset = G["train_dataset"]
val_dataset   = G["val_dataset"]
test_dataset  = G["test_dataset"]
train_stats   = G["train_stats"]
evaluate_model = G["evaluate_model"]
build_model   = G["build_model"]
skill_score   = G["skill_score"]
smart_persistence_forecast = G["smart_persistence_forecast"]
persistence_forecast       = G["persistence_forecast"]
ACTIVE_PRESET = G["ACTIVE_PRESET"]
FEATURE_COLS  = G["FEATURE_COLS"]

# Pin CPU: the same checkpoints scored 11-19 W/m2 worse on MPS than on CUDA.
G["device"], G["USE_AMP"] = torch.device("cpu"), False
device = G["device"]

_enc = G.get("SWIMSEG_ENC")
_enc_path = _enc if _enc and os.path.exists(_enc) else None

# ── The rows every model is scored on ─────────────────────────────────────────
# A deep sample at index t uses features at row t and targets at t+1..t+3, and
# the dataset's own contiguity guard guarantees those are +1/+2/+3 hours. So the
# tabular baselines get row t of the same frame, with shift-based targets. The
# assert below proves the two target constructions agree.
BASELINE_FEATURES = [
    "clearsky_ratio", "cloud_cover", "temperature_2m", "rain",
    "wind_speed_10m", "relative_humidity_2m", "sin_hour", "cos_hour",
    "sin_month", "cos_month", "ghi_clearsky", "ghi_lag1", "ghi_lag2", "ghi_lag3",
]


def rows_for(ds):
    """The dataframe rows a deep dataset actually yields, in loader order."""
    d = ds.df.reset_index(drop=True)
    idx = np.asarray(ds.valid_indices)
    feats = d.loc[idx, BASELINE_FEATURES].astype(np.float32).values
    ghi = d["ghi"].values.astype(np.float32)
    cs = d["ghi_clearsky"].values.astype(np.float32)
    tgt = np.stack([ghi[idx + h] for h in (1, 2, 3)], axis=1)
    cs_f = np.stack([cs[idx + h] for h in (1, 2, 3)], axis=1)
    return feats, tgt, cs_f, ghi[idx], cs[idx], idx


def dm_test(d, lags=None):
    """Diebold-Mariano on a loss difference, Bartlett HAC + Harvey correction."""
    N = len(d)
    dbar = float(d.mean())
    if lags is None:
        lags = max(int(np.floor(1.5 * N ** (1 / 3))), 1)
    dc = d - dbar
    s = float(dc @ dc) / N
    for l in range(1, lags + 1):
        s += 2.0 * (1.0 - l / (lags + 1.0)) * (float(dc[:-l] @ dc[l:]) / N)
    dm = dbar / np.sqrt(max(s, 1e-12) / N)
    dm *= np.sqrt(max((N - 1) / N, 1e-9))
    return dm, 2.0 * (1.0 - sps.t.cdf(abs(dm), df=N - 1)), lags


CKPTS = [
    ("LSTM-only",               ACTIVE_PRESET, "lstm",   None, "best_model_lstm.pt"),
    ("CNN-only",                ACTIVE_PRESET, "cnn",    None, "best_model_cnn.pt"),
    ("Naive concat fusion",     ACTIVE_PRESET, "concat", None, "best_model_concat.pt"),
    ("Physics-Gated (large)",   "large",       None,     None, "best_model.pt"),
    ("Physics-Gated (small)",   "small",       None,     None, "best_model_small.pt"),
]

results = {}
for split, ds, loader in [("val", val_dataset, G["val_loader"]),
                          ("test", test_dataset, G["test_loader"])]:
    feats, tgt, cs_f, ghi_now, cs_now, idx = rows_for(ds)
    n = len(idx)
    print(f"\n{'='*72}\n{split}: {n} imaged rows\n{'='*72}")

    # baselines on these rows
    sp = smart_persistence_forecast(ghi_now, cs_now, cs_f)
    pe = persistence_forecast(ghi_now)
    ref = mean_absolute_error(tgt, sp)
    split_res = {"n": int(n), "models": {}}
    for nm, P in [("Smart persistence", sp), ("Persistence", pe)]:
        split_res["models"][nm] = {
            "MAE": float(mean_absolute_error(tgt, P)),
            "RMSE": float(np.sqrt(mean_squared_error(tgt, P))),
            "R2": float(r2_score(tgt, P)),
        }
    print(f"  smart persistence MAE {ref:.2f}")

    # tabular baselines, fitted on the deep model's TRAIN rows
    tr_feats, tr_tgt, *_ = rows_for(train_dataset)
    per_sample = {}
    for nm, mk in [("Linear Regression", lambda: LinearRegression()),
                   ("Random Forest", lambda: RandomForestRegressor(
                       n_estimators=100, n_jobs=-1, random_state=42)),
                   ("LightGBM", lambda: lgb.LGBMRegressor(
                       n_estimators=200, random_state=42, verbose=-1))]:
        P = np.zeros((n, 3), dtype=np.float64)
        for h in range(3):
            m = mk()
            m.fit(tr_feats, tr_tgt[:, h])
            P[:, h] = m.predict(feats)
        mae = float(mean_absolute_error(tgt, P))
        split_res["models"][nm] = {
            "MAE": mae, "RMSE": float(np.sqrt(mean_squared_error(tgt, P))),
            "R2": float(r2_score(tgt, P)),
            "Skill_vs_SP": float(skill_score(mae, ref)),
        }
        per_sample[nm] = np.abs(P - tgt).mean(axis=1)
        print(f"  {nm:24s} MAE={mae:7.2f}")

    # deep checkpoints on the identical loader
    for label, preset, ab, fcols, ck in CKPTS:
        if not os.path.exists(ck):
            print(f"  skip {label}: no {ck}")
            continue
        m = build_model(preset, ablation=ab, feature_cols=fcols,
                        enc_path=_enc_path).to(device)
        g, hr, mu, sig, tg = evaluate_model(m, loader, label, ck)
        assert len(mu) == n, f"{label}: {len(mu)} preds vs {n} rows"
        assert np.allclose(tg, tgt, atol=1.0), (
            f"{label}: deep targets disagree with shift-based targets")
        g["Skill_vs_SP"] = float(skill_score(g["MAE_Wm2"], ref))
        split_res["models"][label] = {
            "MAE": float(g["MAE_Wm2"]), "RMSE": float(g["RMSE_Wm2"]),
            "R2": float(g["R2"]), "PICP": float(g["PICP_pct"]),
            "Skill_vs_SP": g["Skill_vs_SP"],
        }
        per_sample[label] = np.abs(mu - tg).mean(axis=1)
        print(f"  {label:24s} MAE={g['MAE_Wm2']:7.2f}")
        del m

    # the question: best deep model vs LightGBM, on identical rows
    deep = [k for k in per_sample if k not in
            ("Linear Regression", "Random Forest", "LightGBM")]
    if deep and "LightGBM" in per_sample:
        best = min(deep, key=lambda k: per_sample[k].mean())
        d = per_sample[best] - per_sample["LightGBM"]
        dm, p, lags = dm_test(d)
        split_res["deep_vs_lightgbm"] = {
            "best_deep": best,
            "best_deep_MAE": float(per_sample[best].mean()),
            "lightgbm_MAE": float(per_sample["LightGBM"].mean()),
            "delta_MAE": float(d.mean()),
            "DM": float(dm), "p_value": float(p), "hac_lags": int(lags),
            "significant": bool(p < 0.05),
        }
        who = best if d.mean() < 0 else "LightGBM"
        print(f"\n  {best} vs LightGBM: dMAE={d.mean():+.2f} "
              f"({who} lower)  DM={dm:+.2f} p={p:.4f}")
    results[split] = split_res

with open(OUT, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nwritten -> {OUT}")

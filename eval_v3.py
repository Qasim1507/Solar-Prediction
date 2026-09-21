"""
eval_v3.py — score the v3 seeds, their Gaussian ensemble, and the tabular
baselines on EXACTLY the same rows.

    python eval_v3.py --seeds 42 1337 2024                  # val  (optimistic)
    python eval_v3.py --seeds 42 1337 2024 --split test     # test (unbiased)

--split val reports on the rows checkpoint selection was performed on, so the
deep models carry a selection advantage the tabular baselines do not. --split
test is held out from both training and selection and is the number to quote.

The ensemble is precision-weighted, which is the right combination rule for
Gaussian heads: a seed that is confident on a sample gets more say on it than
one that is not. Plain averaging throws that information away.

Everything is scored on the rows KtDataset actually yields, so the deep models
and LightGBM see one identical row set and the comparison is honest.
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from scipy import stats as sps
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import WinnerModel, TABULAR_COLS_V3, persistence_forecast, \
    smart_persistence_forecast, skill_score
import train as T

KT_CAP = 1.15


def dm_test(d, lags=None):
    """Diebold-Mariano, Bartlett HAC + Harvey small-sample correction."""
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


def metrics(actual, pred, ref_mae=None):
    mae = float(mean_absolute_error(actual, pred))
    out = {"MAE": mae,
           "RMSE": float(np.sqrt(mean_squared_error(actual, pred))),
           "R2": float(r2_score(actual, pred))}
    if ref_mae:
        out["Skill_vs_SP"] = float(skill_score(mae, ref_mae))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 1337, 2024])
    ap.add_argument("--csv", default="data/combined_dataset_v2.csv")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="analysis/v3_results.json")
    ap.add_argument("--split", choices=["val", "test"], default="val",
                    help="val = the rows checkpoints were selected on "
                         "(optimistic); test = held out from both training "
                         "and selection (unbiased)")
    a = ap.parse_args()

    dev = torch.device(a.device) if a.device else (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu"))

    df = T.engineer(pd.read_csv(a.csv, parse_dates=["timestamp"]))
    df = df[df["image_path"].notna()].reset_index(drop=True)
    n = len(df); t_end = int(n * .70); v_end = int(n * .85)
    tr_df = df.iloc[:t_end]
    ev_df = df.iloc[t_end:v_end] if a.split == "val" else df.iloc[v_end:]

    tr = T.KtDataset(tr_df, True)
    st = tr.stats()
    va = T.KtDataset(ev_df, False, st)
    loader = DataLoader(va, batch_size=64, shuffle=False)
    note = ("selection was performed on these rows - optimistic"
            if a.split == "val" else
            "held out from training AND selection - unbiased")
    print(f"train={len(tr)} {a.split}={len(va)} device={dev}\n({note})")

    # ── truth and baselines on exactly these rows ────────────────────────────
    vi = np.asarray(va.idx)
    actual = np.stack([va.ghi[vi + h] for h in (1, 2, 3)], 1)
    cs_fut = np.stack([va.cs[vi + h] for h in (1, 2, 3)], 1)
    sp = smart_persistence_forecast(va.ghi[vi], va.cs[vi], cs_fut)
    ref = float(mean_absolute_error(actual, sp))

    results = {"n": int(len(vi)), "split": a.split, "models": {}}
    per_sample = {}
    results["models"]["Smart persistence"] = metrics(actual, sp)
    results["models"]["Persistence"] = metrics(
        actual, persistence_forecast(va.ghi[vi]))
    per_sample["Smart persistence"] = np.abs(sp - actual).mean(1)
    print(f"smart persistence MAE {ref:.2f}")

    # ── tabular baselines, fitted on the same train rows ─────────────────────
    ti = np.asarray(tr.idx)
    Xtr = tr.tab[ti]
    Ytr = np.stack([tr.ghi[ti + h] for h in (1, 2, 3)], 1)
    Xva = va.tab[vi]
    for nm, mk in [("Linear Regression", lambda: LinearRegression()),
                   ("Random Forest", lambda: RandomForestRegressor(
                       n_estimators=100, n_jobs=-1, random_state=42)),
                   ("LightGBM", lambda: lgb.LGBMRegressor(
                       n_estimators=200, random_state=42, verbose=-1))]:
        P = np.zeros_like(actual, dtype=np.float64)
        for h in range(3):
            m = mk(); m.fit(Xtr, Ytr[:, h]); P[:, h] = m.predict(Xva)
        results["models"][nm] = metrics(actual, P, ref)
        per_sample[nm] = np.abs(P - actual).mean(1)
        print(f"  {nm:22s} MAE={results['models'][nm]['MAE']:7.2f}")

    # ── v3 seeds ─────────────────────────────────────────────────────────────
    mus, sigs = [], []
    for s in a.seeds:
        ck = f"best_model_v3_seed{s}.pt"
        if not os.path.exists(ck):
            print(f"  skip seed {s}: no {ck}")
            continue
        m = WinnerModel(n_tab=len(TABULAR_COLS_V3), imagenet_init=False).to(dev)
        m.load_state_dict(torch.load(ck, map_location=dev)); m.eval()
        MU, SG = [], []
        with torch.no_grad():
            for b in loader:
                mu, sg = m(b["tab"].to(dev), b["frame_t"].to(dev),
                           b["frame_t1"].to(dev), b["frame_t2"].to(dev),
                           b["roi"].to(dev), b["cs_future"].to(dev))
                MU.append(mu.float().cpu()); SG.append(sg.float().cpu())
        mu_kt = torch.cat(MU).numpy() * st["kt_std"] + st["kt_mean"]
        sg_kt = torch.cat(SG).numpy() * st["kt_std"]
        P = np.clip(mu_kt, 0, KT_CAP) * cs_fut
        results["models"][f"v3 seed {s}"] = metrics(actual, P, ref)
        per_sample[f"v3 seed {s}"] = np.abs(P - actual).mean(1)
        mus.append(mu_kt); sigs.append(np.maximum(sg_kt, 1e-6))
        print(f"  v3 seed {s:<16d} MAE={results['models'][f'v3 seed {s}']['MAE']:7.2f}")
        del m

    # ── precision-weighted Gaussian ensemble ─────────────────────────────────
    if len(mus) >= 2:
        w = [1.0 / (s ** 2) for s in sigs]
        mu_ens = sum(m * wi for m, wi in zip(mus, w)) / sum(w)
        sg_ens = np.sqrt(1.0 / sum(w))
        P = np.clip(mu_ens, 0, KT_CAP) * cs_fut
        results["models"]["v3 ensemble"] = metrics(actual, P, ref)
        results["models"]["v3 ensemble"]["mean_sigma_kt"] = float(sg_ens.mean())
        per_sample["v3 ensemble"] = np.abs(P - actual).mean(1)
        print(f"  {'v3 ensemble':22s} MAE={results['models']['v3 ensemble']['MAE']:7.2f}"
              f"  ({len(mus)} seeds)")

    # ── the question ─────────────────────────────────────────────────────────
    deep = [k for k in per_sample if k.startswith("v3")]
    if deep and "LightGBM" in per_sample:
        best = min(deep, key=lambda k: per_sample[k].mean())
        d = per_sample[best] - per_sample["LightGBM"]
        dm, p, lags = dm_test(d)
        results["deep_vs_lightgbm"] = {
            "best_deep": best,
            "best_deep_MAE": float(per_sample[best].mean()),
            "lightgbm_MAE": float(per_sample["LightGBM"].mean()),
            "delta_MAE": float(d.mean()), "DM": float(dm),
            "p_value": float(p), "hac_lags": int(lags),
            "deep_wins": bool(d.mean() < 0), "significant": bool(p < 0.05)}
        print(f"\n{best} vs LightGBM: dMAE={d.mean():+.2f} W/m2  "
              f"DM={dm:+.2f} p={p:.4f}  "
              f"{'DEEP WINS' if d.mean() < 0 else 'LightGBM wins'}"
              f"{' (significant)' if p < 0.05 else ' (not significant)'}")

    os.makedirs("analysis", exist_ok=True)
    json.dump(results, open(a.out, "w"), indent=2)
    print(f"\nwritten -> {a.out}")


if __name__ == "__main__":
    main()

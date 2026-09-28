"""
predict_v3.py — live GHI forecast from the v3 ensemble.

    python predict_v3.py                  # fetch live data, then forecast
    python predict_v3.py --skip-fetch     # reuse datanow/, no network
    python predict_v3.py --seeds 42       # single seed instead of the ensemble
    python predict_v3.py --no-sp-t1       # drop the persistence challenger

Differences from predict.py, which serves the v2 model:

  * The network predicts k_t, so the forecast is k_t x clear-sky. The physics
    cap is applied in k_t space before that multiply - capping in W/m2 would
    collapse the interval to a point wherever it binds.
  * Clear-sky everywhere is the MEAN OVER THE PRECEDING HOUR, matching how
    combined_dataset_v2.csv was built and how Open-Meteo labels its hourly
    values. Using the instantaneous value would inflate the forecast by ~5% at
    noon and more towards sunrise and sunset.
  * Three checkpoints are combined precision-weighted, the correct rule for
    Gaussian heads: a seed that is confident on a sample gets more say on it.
  * Normalisation comes from the checkpoint's .stats.json sidecar (14 feature
    means/stds plus kt_mean/kt_std), not train_stats.json, which describes the
    11-feature v2 model.

On top of the three ensemble forecasts, a smart-persistence forecast for t+1h is
published as a FOURTH entry labelled "t+1h-sp" (--sp-t1, on by default). Live,
the ensemble is losing t+1h to that free baseline by 134 vs 45 W/m2 MAE, having
beaten it offline by ~10%, so both are committed in advance and scored side by
side rather than the question being settled retrospectively. The three v3
entries are untouched: v3 remains the primary forecast at all three horizons.
"""
import argparse, json, os, sys
from datetime import timedelta

import numpy as np
import pandas as pd
import pytz
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import (WinnerModel, TABULAR_COLS_V3, load_historical_df,
                   extend_with_recent, _add_engineered_features,
                   load_satellite_inputs, compute_clearsky_hour_mean,
                   check_inputs_in_distribution,
                   TRAIN_HOUR_START, TRAIN_HOUR_END, WINDOW_SIZE)
from predict import fetch_live_data, check_satellite_inputs

CSV = "./data/combined_dataset_v2.csv"
SAT = "./datanow/satellite/himawari_current.png"
PREV1 = "./datanow/satellite/himawari_prev1.png"
PREV2 = "./datanow/satellite/himawari_prev2.png"
KT_CAP = 1.15


def build_window_v3(df, stats, ref_time):
    """(1, 24, 14) normalised lookback, daylight rows only."""
    df = _add_engineered_features(df)
    ref = pd.Timestamp(ref_time).replace(tzinfo=None)
    past = df[(df["timestamp"] <= ref)
              & (df["timestamp"].dt.hour.between(TRAIN_HOUR_START,
                                                 TRAIN_HOUR_END))
              ].tail(WINDOW_SIZE)
    if len(past) < WINDOW_SIZE:
        raise ValueError(f"only {len(past)} daylight rows for a "
                         f"{WINDOW_SIZE}-step window ending {ref}")
    tab = past[TABULAR_COLS_V3].values.astype(np.float32)
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)
    return torch.from_numpy((tab - mean) / std).float().unsqueeze(0), past


def persistence_kt_sigma(df, issue_hour=None, min_n=100, fallback=0.13):
    """
    sigma of the t+1h smart-persistence error, in k_t units.

    Smart persistence asserts k_t is unchanged an hour from now, so its error IS
    the hour-to-hour change in k_t. Estimating it that way keeps the challenger's
    interval empirical rather than invented, and costs no extra I/O — the frame is
    already loaded. Restricted to the daylight rows the model trains on, and to
    genuinely consecutive hours, so overnight gaps are not counted as one step.

    Conditioned on the issue hour, because the volatility is not flat across the
    day: pooled over 08:00-17:00 the sd is 0.168, but that is dragged up by 09:00
    (0.228, convective onset) and the late afternoon (0.16). At the midday hours
    these forecasts are actually issued it is ~0.125-0.138. Roughly 992 samples
    per hour, so conditioning costs nothing in precision.
    """
    d = _add_engineered_features(df).sort_values("timestamp")
    d = d[d["timestamp"].dt.hour.between(TRAIN_HOUR_START, TRAIN_HOUR_END)]
    kt = d["clearsky_ratio"].clip(0.0, KT_CAP)
    step = kt.diff().where(d["timestamp"].diff() == pd.Timedelta(hours=1))
    if issue_hour is not None:
        at_hour = step.where(d["timestamp"].dt.hour == issue_hour).dropna()
        if len(at_hour) >= min_n:
            s = float(at_hour.std())
            if np.isfinite(s) and s > 0:
                return s
    s = float(step.std())                       # pooled fallback
    return s if np.isfinite(s) and s > 0 else fallback


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 1337, 2024])
    ap.add_argument("--skip-fetch", action="store_true")
    ap.add_argument("--csv", default=CSV)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="./forecast_latest_v3.json")
    ap.add_argument("--sp-t1", dest="sp_t1", action="store_true", default=True,
                    help="publish a smart-persistence t+1h challenger (default)")
    ap.add_argument("--no-sp-t1", dest="sp_t1", action="store_false",
                    help="v3 only, as before")
    a = ap.parse_args()

    dev = torch.device(a.device) if a.device else (
        torch.device("cuda") if torch.cuda.is_available() else
        torch.device("cpu"))
    print(f"\n{'='*62}\n  GHI FORECAST - v3 ensemble\n{'='*62}\n  Device: {dev}")

    if not a.skip_fetch:
        fetch_live_data()

    cks = [(s, f"best_model_v3_seed{s}.pt") for s in a.seeds]
    cks = [(s, c) for s, c in cks if os.path.exists(c)]
    if not cks:
        raise SystemExit("no v3 checkpoints found")
    stats = json.load(open(cks[0][1] + ".stats.json"))
    assert stats["features"] == TABULAR_COLS_V3, "feature list drifted"
    print(f"  Seeds: {[s for s, _ in cks]}")

    sgt = pytz.timezone("Asia/Singapore")
    now = pd.Timestamp.now(tz=sgt).replace(minute=0, second=0, microsecond=0)
    print(f"  Now (SGT): {now:%Y-%m-%d %H:%M}")

    df = extend_with_recent(load_historical_df(a.csv))
    tab, past = build_window_v3(df, stats, now)

    # Clear-sky for t+1..t+3 on the training convention (hour mean).
    cs_future = np.array([compute_clearsky_hour_mean(now + timedelta(hours=h))
                          for h in (1, 2, 3)], dtype=np.float32)
    print(f"  Clear-sky (hour mean) t+1h:{cs_future[0]:.0f}  "
          f"t+2h:{cs_future[1]:.0f}  t+3h:{cs_future[2]:.0f} W/m2")

    check_inputs_in_distribution(tab, None, stats)

    print("\n  Satellite inputs:")
    image_ok = check_satellite_inputs(now)
    multi, roi, _flow = load_satellite_inputs(SAT, PREV1, PREV2)
    f_t, f_t1, f_t2 = multi[:, 0:3], multi[:, 3:6], multi[:, 6:9]
    cs_t = torch.from_numpy(cs_future).unsqueeze(0)

    mus, sigs = [], []
    for s, ck in cks:
        m = WinnerModel(n_tab=len(TABULAR_COLS_V3), imagenet_init=False).to(dev)
        m.load_state_dict(torch.load(ck, map_location=dev))
        m.eval()
        with torch.no_grad():
            mu, sg = m(tab.to(dev), f_t.to(dev), f_t1.to(dev), f_t2.to(dev),
                       roi.to(dev), cs_t.to(dev))
        mus.append(mu.float().cpu().numpy()[0] * stats["kt_std"] + stats["kt_mean"])
        sigs.append(np.maximum(sg.float().cpu().numpy()[0] * stats["kt_std"], 1e-6))
        del m

    if len(mus) > 1:                      # precision-weighted Gaussian combine
        w = [1.0 / (s ** 2) for s in sigs]
        kt = sum(m * wi for m, wi in zip(mus, w)) / sum(w)
        kt_sig = np.sqrt(1.0 / sum(w))
    else:
        kt, kt_sig = mus[0], sigs[0]

    kt_c = np.clip(kt, 0.0, KT_CAP)       # cap in k_t space, then scale
    ghi = kt_c * cs_future
    lo = np.clip((kt_c - 1.645 * kt_sig), 0.0, KT_CAP) * cs_future
    hi = np.clip((kt_c + 1.645 * kt_sig), 0.0, KT_CAP) * cs_future

    # ── smart-persistence challenger for t+1h ────────────────────────────────
    # extend_with_recent() falls back to the stale CSV without raising if the
    # Open-Meteo call fails, so a days-old kt_issue is possible and would look
    # like nothing worse than an inexplicably bad day in the log. Check that the
    # lookback really ends at the issue hour before trusting it.
    lookback_fresh = bool(past["timestamp"].max() == now.replace(tzinfo=None))
    sp = None
    if a.sp_t1:
        if not lookback_fresh:
            print(f"\n  ⚠️  Lookback ends {past['timestamp'].max()}, not "
                  f"{now:%Y-%m-%d %H:%M} — skipping the persistence challenger "
                  f"rather than issuing it off stale data.")
        else:
            kt_i = float(np.clip(past.iloc[-1]["clearsky_ratio"], 0.0, KT_CAP))
            sp_sig = persistence_kt_sigma(df, issue_hour=now.hour)
            sp = {
                "ghi": kt_i * cs_future[0],
                "lo": float(np.clip(kt_i - 1.645 * sp_sig, 0.0, KT_CAP)) * cs_future[0],
                "hi": float(np.clip(kt_i + 1.645 * sp_sig, 0.0, KT_CAP)) * cs_future[0],
                "kt": kt_i,
                "sig": sp_sig,
            }
            print(f"\n  Persistence challenger: k_t at issue hour = {kt_i:.3f}"
                  f"  (sigma {sp_sig:.3f} from hour-to-hour k_t change)")

    outside = not (TRAIN_HOUR_START <= now.hour <= TRAIN_HOUR_END)
    if outside:
        print(f"\n  ⚠️  {now:%H:%M} SGT is outside the "
              f"{TRAIN_HOUR_START:02d}:00-{TRAIN_HOUR_END:02d}:00 training "
              f"window - treat this with caution.")

    print(f"\n{'='*62}\n  FORECAST from {now:%Y-%m-%d %H:%M} SGT\n{'='*62}")
    print(f"  {'Horizon':<14}{'GHI':>10}{'90% CI':>22}{'k_t':>8}"
          f"{'clearsky':>11}   method")
    for h in range(3):
        t = (now + timedelta(hours=h + 1)).strftime("%H:%M")
        print(f"  t+{h+1}h ({t}){ghi[h]:>10.1f}"
              f"   [{lo[h]:>6.1f} - {hi[h]:>6.1f}]{kt_c[h]:>8.3f}"
              f"{cs_future[h]:>11.0f}   v3")
    if sp:
        t = (now + timedelta(hours=1)).strftime("%H:%M")
        print(f"  t+1h ({t}){sp['ghi']:>10.1f}"
              f"   [{sp['lo']:>6.1f} - {sp['hi']:>6.1f}]{sp['kt']:>8.3f}"
              f"{cs_future[0]:>11.0f}   smart-persistence")
    print("=" * 62)

    out = {
        "model": "v3 ensemble (WinnerModel, k_t target)",
        "seeds": [s for s, _ in cks],
        "forecast_time_sgt": now.strftime("%Y-%m-%d %H:%M"),
        "forecasts": [
            {"horizon": f"t+{h+1}h",
             "time_sgt": (now + timedelta(hours=h + 1)).strftime("%Y-%m-%d %H:%M"),
             "ghi_forecast_wm2": round(float(ghi[h]), 1),
             "ghi_lower_90": round(float(lo[h]), 1),
             "ghi_upper_90": round(float(hi[h]), 1),
             "kt_forecast": round(float(kt_c[h]), 4),
             "kt_sigma": round(float(kt_sig[h]), 4),
             "clearsky_hour_mean_wm2": round(float(cs_future[h]), 1),
             "kt_capped": bool(kt[h] > KT_CAP),
             "method": "v3_ensemble"}
            for h in range(3)],
        "diagnostics": {
            "image_ok": bool(image_ok),
            "outside_training_hours": bool(outside),
            "lookback_ends": str(past["timestamp"].max()),
            "lookback_hours": sorted(past["timestamp"].dt.hour.unique().tolist()),
            "lookback_fresh": lookback_fresh,
            "sp_t1_enabled": bool(a.sp_t1),
            "kt_issue": round(sp["kt"], 4) if sp else None,
        },
    }

    # A distinct horizon label, not a second "t+1h": append_log de-dupes on
    # (forecast_time, horizon), so reusing the label would silently drop one arm.
    if sp:
        out["forecasts"].append({
            "horizon": "t+1h-sp",
            "time_sgt": (now + timedelta(hours=1)).strftime("%Y-%m-%d %H:%M"),
            "ghi_forecast_wm2": round(float(sp["ghi"]), 1),
            "ghi_lower_90": round(float(sp["lo"]), 1),
            "ghi_upper_90": round(float(sp["hi"]), 1),
            "kt_forecast": round(float(sp["kt"]), 4),
            "kt_sigma": round(float(sp["sig"]), 4),
            "clearsky_hour_mean_wm2": round(float(cs_future[0]), 1),
            "kt_capped": False,
            "method": "smart_persistence",
        })
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Saved -> {a.out}")


if __name__ == "__main__":
    main()

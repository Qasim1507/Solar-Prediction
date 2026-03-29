import os
import re
import argparse
import pandas as pd
import pytz
from datetime import datetime

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SGT = pytz.timezone("Asia/Singapore")
UTC = pytz.utc

IMAGE_PATTERN = re.compile(
    r"himawari_4d_(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})\.png$"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_image_utc(filename: str) -> datetime | None:
    m = IMAGE_PATTERN.search(filename)
    if not m:
        return None
    y, mo, d, h, mi, s = (int(x) for x in m.groups())
    return datetime(y, mo, d, h, mi, s, tzinfo=UTC)


def sgt_to_utc(ts: pd.Timestamp) -> pd.Timestamp:
    if ts.tzinfo is None:
        ts = SGT.localize(ts)
    return ts.astimezone(UTC)


def scan_satellite_images(satellite_dir: str) -> pd.DataFrame:
    records = []
    for root, _dirs, files in os.walk(satellite_dir):
        for fname in files:
            if not fname.lower().endswith(".png"):
                continue
            utc_dt = parse_image_utc(fname)
            if utc_dt is None:
                continue
            rel_path = os.path.join(root, fname).replace("\\", "/")
            records.append({"utc_dt": utc_dt, "image_path": rel_path})

    if not records:
        raise FileNotFoundError(
            f"No Himawari PNG images found under '{satellite_dir}'. "
            "Check the directory exists and filenames follow: himawari_4d_YYYYMMDD_HHMMSS.png"
        )

    df = pd.DataFrame(records).sort_values("utc_dt").reset_index(drop=True)
    print(f"  Found {len(df):,} satellite images "
          f"({df['utc_dt'].min().date()} → {df['utc_dt'].max().date()})")
    return df


def match_images(
    weather_df: pd.DataFrame,
    sat_df: pd.DataFrame,
    tolerance_minutes: int,
) -> pd.DataFrame:
    sat_ts_series = pd.DatetimeIndex(sat_df["utc_dt"])
    image_paths, time_diffs = [], []

    for utc_ts in weather_df["utc_timestamp"]:
        idx = sat_ts_series.searchsorted(utc_ts)
        candidates = []
        for i in [idx - 1, idx]:
            if 0 <= i < len(sat_df):
                diff = abs((sat_ts_series[i] - utc_ts).total_seconds() / 60)
                candidates.append((diff, i))

        if not candidates:
            image_paths.append(None)
            time_diffs.append(None)
            continue

        best_diff, best_idx = min(candidates, key=lambda x: x[0])
        if best_diff <= tolerance_minutes:
            image_paths.append(sat_df.iloc[best_idx]["image_path"])
            time_diffs.append(round(best_diff, 1))
        else:
            image_paths.append(None)
            time_diffs.append(None)

    weather_df = weather_df.copy()
    weather_df["image_path"] = image_paths
    weather_df["time_diff_minutes"] = time_diffs
    return weather_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_combined_dataset(
    pv_csv: str,
    ground_truth_csv: str,
    satellite_dir: str,
    output_path: str,
    tolerance_minutes: int,
    daylight_only: bool,
) -> pd.DataFrame:

    print("\n" + "=" * 65)
    print("BUILD COMBINED DATASET")
    print("=" * 65)

    # ------------------------------------------------------------------
    # 1. Load weather / pvlib data
    # ------------------------------------------------------------------
    print(f"\n[1/5] Loading weather data from '{pv_csv}' ...")
    if not os.path.exists(pv_csv):
        raise FileNotFoundError(f"Not found: {pv_csv}")

    df = pd.read_csv(pv_csv, parse_dates=["timestamp"])
    df["timestamp"] = df["timestamp"].dt.floor("h")
    print(f"  Loaded {len(df):,} rows "
          f"({df['timestamp'].min().date()} → {df['timestamp'].max().date()})")

    # ------------------------------------------------------------------
    # 2. Merge pv_actual ground truth
    # ------------------------------------------------------------------
    print(f"\n[2/5] Merging PV ground truth from '{ground_truth_csv}' ...")
    if not os.path.exists(ground_truth_csv):
        raise FileNotFoundError(
            f"Not found: {ground_truth_csv}\n"
            "  → Run fetch_pv_ground_truth.py first."
        )

    gt = pd.read_csv(ground_truth_csv, parse_dates=["timestamp"])
    gt["timestamp"] = gt["timestamp"].dt.floor("h")
    gt = gt[["timestamp", "pv_actual"]].drop_duplicates(subset=["timestamp"])

    before = len(df)
    df = df.merge(gt, on="timestamp", how="left")

    matched_gt = df["pv_actual"].notna().sum()
    print(f"  Ground truth rows matched: {matched_gt:,} / {before:,}")

    if matched_gt == 0:
        print("  ⚠️  WARNING: No ground truth matched. Check that timestamps in")
        print("     pv_ground_truth_sg.csv overlap with pv_dataset_sg.csv.")
    elif matched_gt < before * 0.5:
        sample = df[df["pv_actual"].isna()]["timestamp"].head(3).tolist()
        print(f"  ⚠️  WARNING: <50% of rows matched. Sample unmatched: {[str(t) for t in sample]}")
        print(f"     This is expected if ground truth only covers 2020-2023 and")
        print(f"     your weather data starts from 2024. Both columns will be kept.")

    # ------------------------------------------------------------------
    # 3. Build timestamp columns + daylight filter
    # ------------------------------------------------------------------
    print("\n[3/5] Building timestamp columns ...")
    df["sg_timestamp_naive"] = df["timestamp"]
    df["utc_timestamp"] = df["timestamp"].apply(sgt_to_utc)

    if daylight_only:
        before = len(df)
        mask = (df["timestamp"].dt.hour >= 8) & (df["timestamp"].dt.hour <= 17)
        df = df[mask].reset_index(drop=True)
        print(f"  Daylight filter (08:00–17:00 SGT): {before:,} → {len(df):,} rows")

    # ------------------------------------------------------------------
    # 4. Scan and match satellite images
    # ------------------------------------------------------------------
    print(f"\n[4/5] Scanning satellite images in '{satellite_dir}' ...")
    sat_df = scan_satellite_images(satellite_dir)

    print(f"\n[5/5] Matching to satellite images (tolerance: {tolerance_minutes} min) ...")
    df = match_images(df, sat_df, tolerance_minutes)

    matched_img   = df["image_path"].notna().sum()
    unmatched_img = df["image_path"].isna().sum()
    print(f"  Image matched:   {matched_img:,} rows")
    print(f"  Image unmatched: {unmatched_img:,} rows")

    # ------------------------------------------------------------------
    # 5. Final column order and save
    # ------------------------------------------------------------------
    output_cols = [
        "timestamp",
        "temperature_2m",
        "relative_humidity_2m",
        "rain",
        "wind_speed_10m",
        "cloud_cover",
        "ghi",
        "direct_normal_irradiance",
        "diffuse_radiation",
        "ghi_clearsky",
        "pv_power_predicted",   # pvlib simulation — useful as a model feature/baseline
        "pv_actual",            # PVGIS/Open-Meteo ground truth — use as training target
        "sg_timestamp_naive",
        "utc_timestamp",
        "image_path",
        "time_diff_minutes",
    ]
    output_cols = [c for c in output_cols if c in df.columns]
    df = df[output_cols]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    pv_pred_count   = df["pv_power_predicted"].notna().sum() if "pv_power_predicted" in df.columns else 0
    pv_actual_count = df["pv_actual"].notna().sum() if "pv_actual" in df.columns else 0
    img_count       = df["image_path"].notna().sum()

    print(f"\n{'=' * 65}")
    print(f"✓ Combined dataset saved → '{output_path}'")
    print(f"  Total rows : {len(df):,}")
    print(f"  Columns    : {len(df.columns)}")
    print(f"  Date range : {df['timestamp'].min().date()} → {df['timestamp'].max().date()}")
    print(f"\n  Column coverage:")
    print(f"    pv_power_predicted (pvlib baseline) : {pv_pred_count:,} rows")
    print(f"    pv_actual (training target)         : {pv_actual_count:,} rows")
    print(f"    image_path (satellite matched)      : {img_count:,} rows")
    if df["time_diff_minutes"].notna().any():
        print(f"    avg satellite time diff             : {df['time_diff_minutes'].mean():.1f} min")
    print(f"\n  Fully usable rows (pv_actual + image both present): "
          f"{(df['pv_actual'].notna() & df['image_path'].notna()).sum():,}")
    print("=" * 65 + "\n")

    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Build combined dataset: weather + satellite images + PV ground truth"
    )
    p.add_argument("--pv-csv",             default="data/pv_dataset_sg.csv")
    p.add_argument("--ground-truth",       default="data/pv_ground_truth_sg.csv",
                   help="Output of fetch_pv_ground_truth.py")
    p.add_argument("--satellite-dir",      default="data/satellite")
    p.add_argument("--output",             default="data/combined_dataset.csv")
    p.add_argument("--tolerance",          type=int, default=60,
                   help="Max minutes between weather row and satellite image (default: 60)")
    p.add_argument("--no-daylight-filter", action="store_true",
                   help="Keep all hours (default: 08:00–17:00 SGT only)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_combined_dataset(
        pv_csv=args.pv_csv,
        ground_truth_csv=args.ground_truth,
        satellite_dir=args.satellite_dir,
        output_path=args.output,
        tolerance_minutes=args.tolerance,
        daylight_only=not args.no_daylight_filter,
    )
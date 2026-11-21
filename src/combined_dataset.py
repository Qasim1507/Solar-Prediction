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
    r"himawari_(?:4d|sg)_(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})\.png$"
)

# Frame series fed to the model: t, t-10min, t-20min (Himawari's native
# 10-minute cadence). current_data.fetch_frame_series() MUST use the same
# offsets — a training/inference spacing mismatch silently destroys the
# optical-flow signal (see ISSUE-B2/B3).
FRAME_OFFSETS_MINUTES = (0, 10, 20)


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
            "Check the directory exists and filenames follow: himawari_{4d,sg}_YYYYMMDD_HHMMSS.png"
        )

    df = pd.DataFrame(records).sort_values("utc_dt").reset_index(drop=True)
    print(f"  Found {len(df):,} satellite images "
          f"({df['utc_dt'].min().date()} → {df['utc_dt'].max().date()})")
    return df


def _match_at_offset(
    sat_ts_series: pd.DatetimeIndex,
    sat_df: pd.DataFrame,
    target_times,
    tolerance_minutes: int,
    backfill_minutes: int = 0,
    taken=None,
):
    """Nearest satellite image to each target time, or None beyond tolerance.

    backfill_minutes > 0 additionally accepts the nearest scan EARLIER than
    the target, out to that many minutes, when nothing lands within tolerance.
    Himawari skips a full-disk scan during daily housekeeping (02:40 and 14:40
    UTC), so the exact slot genuinely does not exist for ~10% of rows; a
    slightly older frame is a far better CNN input than a zero-filled one.

    `taken` is a per-row set of already-assigned paths, so a backfill cannot
    duplicate a frame the series already holds.
    """
    image_paths, time_diffs = [], []

    for row, utc_ts in enumerate(target_times):
        idx = sat_ts_series.searchsorted(utc_ts)
        candidates = []
        for i in [idx - 1, idx]:
            if 0 <= i < len(sat_df):
                diff = abs((sat_ts_series[i] - utc_ts).total_seconds() / 60)
                candidates.append((diff, i))

        best_diff, best_idx = (min(candidates, key=lambda x: x[0])
                               if candidates else (None, None))

        if best_diff is not None and best_diff <= tolerance_minutes:
            chosen, diff = best_idx, best_diff
        elif backfill_minutes:
            # Walk back from the target to the most recent usable scan.
            chosen, diff = None, None
            j = idx - 1 if idx <= len(sat_df) else len(sat_df) - 1
            while j >= 0:
                gap = (utc_ts - sat_ts_series[j]).total_seconds() / 60
                if gap > backfill_minutes:
                    break
                if gap >= 0 and (taken is None
                                 or sat_df.iloc[j]["image_path"] not in taken[row]):
                    chosen, diff = j, gap
                    break
                j -= 1
        else:
            chosen, diff = None, None

        if chosen is None:
            image_paths.append(None)
            time_diffs.append(None)
        else:
            path = sat_df.iloc[chosen]["image_path"]
            image_paths.append(path)
            time_diffs.append(round(diff, 1))
            if taken is not None:
                taken[row].add(path)

    return image_paths, time_diffs


def match_images(
    weather_df: pd.DataFrame,
    sat_df: pd.DataFrame,
    tolerance_minutes: int,
    frame_offsets_minutes=FRAME_OFFSETS_MINUTES,
) -> pd.DataFrame:
    """Attach the frame series each row feeds the model.

    The rows are hourly (ERA5 weather is hourly), but the Himawari archive has
    a scan every 10 minutes. The previous frames are therefore taken at
    t-10min / t-20min rather than at the previous row, so that the optical
    flow between them measures actual cloud advection: over a full hour a
    cloud field decorrelates rather than merely moving, which makes flow
    across that gap mostly noise.

    Offsets must match current_data.fetch_frame_series(), or live inference
    sees a different frame spacing than training did.
    """
    sat_ts_series = pd.DatetimeIndex(sat_df["utc_dt"])
    weather_df = weather_df.copy()
    taken = [set() for _ in range(len(weather_df))]

    # One scan cadence of slack, so a housekeeping gap falls back to the
    # previous scan instead of zero-filling. Frame t is never backfilled: it
    # defines the sample's time and must be the real observation.
    cadence = (min(b - a for a, b in
                   zip(frame_offsets_minutes, frame_offsets_minutes[1:]))
               if len(frame_offsets_minutes) > 1 else 0)

    for n, offset in enumerate(frame_offsets_minutes):
        targets = weather_df["utc_timestamp"] - pd.Timedelta(minutes=offset)
        paths, diffs = _match_at_offset(
            sat_ts_series, sat_df, targets, tolerance_minutes,
            backfill_minutes=0 if n == 0 else cadence + tolerance_minutes,
            taken=taken)

        col = "image_path" if n == 0 else f"image_path_prev{n}"
        weather_df[col] = paths
        if n == 0:
            weather_df["time_diff_minutes"] = diffs
        matched = sum(p is not None for p in paths)
        exact = sum(1 for d in diffs if d is not None and d <= tolerance_minutes)
        note = "" if n == 0 else f" ({matched - exact:,} backfilled)"
        print(f"    t-{offset:>3} min → {col:<18} {matched:,} matched{note}")

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
    print(f"  Frame series offsets: "
          f"{', '.join(f't-{o}min' for o in FRAME_OFFSETS_MINUTES)}")
    df = match_images(df, sat_df, tolerance_minutes)

    matched_img   = df["image_path"].notna().sum()
    unmatched_img = df["image_path"].isna().sum()
    print(f"  Image matched:   {matched_img:,} rows")
    print(f"  Image unmatched: {unmatched_img:,} rows")

    prev_cols = [f"image_path_prev{n}"
                 for n in range(1, len(FRAME_OFFSETS_MINUTES))]
    if prev_cols:
        series_cols = ["image_path"] + prev_cols
        full_series = df[series_cols].notna().all(axis=1).sum()
        print(f"  Complete frame series (all {len(FRAME_OFFSETS_MINUTES)} "
              f"frames present): {full_series:,} rows")

        # A frame repeated within a row means two series slots resolved to the
        # same scan — the model would see a zero flow vector and a duplicated
        # channel group. Only possible if tolerance >= half the scan cadence.
        present = df[series_cols].notna().all(axis=1)
        dupes = df.loc[present, series_cols].nunique(axis=1) < len(series_cols)
        if dupes.any():
            print(f"  ⚠️  {dupes.sum():,} rows repeat a frame within their series "
                  f"— lower --tolerance (currently {tolerance_minutes} min)")

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
        *(f"image_path_prev{n}" for n in range(1, len(FRAME_OFFSETS_MINUTES))),
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
    p.add_argument("--satellite-dir",      default="data/satellite_aws")
    p.add_argument("--output",             default="data/combined_dataset.csv")
    p.add_argument("--tolerance",          type=int, default=5,
                   help="Max minutes between weather row and satellite image (default: 5; "
                        "must stay under half the 10-min scan cadence, or a missing "
                        "scan lets two frame-series slots match the same image. "
                        "AWS scans are every 10 min, so this only allows a neighbouring scan)")
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
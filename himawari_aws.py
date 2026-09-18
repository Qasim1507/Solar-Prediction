"""
himawari_aws.py — Singapore Himawari crops from NOAA's public AWS archive
=========================================================================
Uses the ISatSS "sectorized" product (AHI-L2-FLDK-ISatSS): every 10-minute
full-disk scan is pre-cut into 88 tiles per band, stored as small NetCDF
files with calibrated values. Singapore lies in tile T036, so each scan
needs only two small downloads (no account, no decompression, no
reprojection):

    C03  0.64 µm visible   0.5 km   ~4.7 MB   reflectance (0–~1.2)
    C13  10.4 µm infrared  2 km     ~0.3 MB   brightness temperature (K)

The crop is a fixed 572×572 window of the satellite's native 0.5 km grid
centred on Singapore (~370 km E–W × 286 km N–S). C13 is upsampled ×4
(nearest) onto the same grid.

Himawari-9 is operational except Oct–Nov 2025, when JMA switched back to
Himawari-8; the right bucket is picked automatically.

Outputs, under <out>/YYYY/MM/:
    sg_YYYYMMDD_HHMM.npz               float16 "bands" (2, 572, 572)
    himawari_sg_YYYYMMDD_HHMM00.png    RGB composite (bands_to_rgb)

Usage (resumable — finished scans are skipped):
    python himawari_aws.py --start 2024-01-01 --end 2026-09-15 --out data/satellite_aws --workers 32

Requires: numpy, pillow, requests, h5py
"""

import argparse
import csv
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

import numpy as np

# ── Crop definition (native full-disk pixel grid) ─────────────────────────────
TILE        = "T036"                  # full-disk tile containing Singapore
TILE_ROW0   = 8800                    # C03 pixel offset of T036 (verified)
TILE_COL0   = 2200
# Window inside the C03 tile; Singapore is at tile pixel (1914, 1437)
ROWS        = slice(1628, 2200)
COLS        = slice(1152, 1724)
IR_FACTOR   = 4                       # C13 (2 km) → C03 (0.5 km)

# Daytime window in UTC minutes-of-day: 22:30–11:30 UTC = 06:30–19:30 SGT
DAY_START_MIN = 22 * 60 + 30
DAY_END_MIN   = 11 * 60 + 30
CADENCE_MIN   = 10

# BT range mapped to the blue channel: 310 K (warm surface) → 0, 190 K → 1
BT_WARM, BT_COLD = 310.0, 190.0

BUCKETS     = ("noaa-himawari9", "noaa-himawari8")
PREFIX      = "AHI-L2-FLDK-ISatSS"
BANDS       = (("005", "M1C03"), ("020", "M1C13"))   # (resolution tag, channel)

MISSING_CSV = "missing.csv"


# ── Pure helpers ──────────────────────────────────────────────────────────────

def bands_to_rgb(bands: np.ndarray) -> np.ndarray:
    """(2,H,W) [reflectance, BT K] → (H,W,3) float32 in 0–1.

    R = G = visible reflectance, B = inverted IR (cold cloud tops bright).
    Shared by the PNG writer and model.load_image_tensor so training and
    inference see identical images.
    """
    refl = np.nan_to_num(bands[0].astype(np.float32), nan=0.0)
    bt   = np.nan_to_num(bands[1].astype(np.float32), nan=BT_WARM)
    vis  = np.clip(refl, 0.0, 1.0)
    ir   = np.clip((BT_WARM - bt) / (BT_WARM - BT_COLD), 0.0, 1.0)
    return np.stack([vis, vis, ir], axis=-1)


def rgb_to_uint8(rgb: np.ndarray) -> np.ndarray:
    return (np.clip(rgb, 0.0, 1.0) * 255 + 0.5).astype(np.uint8)


def output_paths(dt: datetime, out_root: str):
    d = os.path.join(out_root, dt.strftime("%Y"), dt.strftime("%m"))
    npz = os.path.join(d, f"sg_{dt:%Y%m%d_%H%M}.npz")
    png = os.path.join(d, f"himawari_sg_{dt:%Y%m%d_%H%M}00.png")
    return npz, png


def is_daytime(dt: datetime) -> bool:
    m = dt.hour * 60 + dt.minute
    return m >= DAY_START_MIN or m <= DAY_END_MIN


def daytime_times(start: datetime, end: datetime):
    """10-minute UTC timestamps in [start, end] inside the daytime window."""
    t = start.replace(second=0, microsecond=0)
    t -= timedelta(minutes=t.minute % CADENCE_MIN)
    if t < start:
        t += timedelta(minutes=CADENCE_MIN)
    while t <= end:
        if is_daytime(t):
            yield t
        t += timedelta(minutes=CADENCE_MIN)


# ── S3 access (plain HTTPS, anonymous) ────────────────────────────────────────

_local = threading.local()


def _session():
    if not hasattr(_local, "s"):
        import requests
        _local.s = requests.Session()
    return _local.s


def _get(url, **kw):
    for attempt in range(4):
        try:
            r = _session().get(url, timeout=60, **kw)
            if r.status_code == 200:
                return r
            if r.status_code == 404:
                return None
        except Exception:
            pass
        time.sleep(2 ** attempt)
    raise IOError(f"GET failed: {url}")


def scan_files(dt: datetime):
    """[(bucket, key)] for C03 and C13 tile T036, or None if not archived."""
    for bucket in BUCKETS:
        keys = []
        for res, ch in BANDS:
            prefix = f"{PREFIX}/{dt:%Y/%m/%d/%H%M}/OR_HFD-{res}-"
            r = _get(f"https://{bucket}.s3.amazonaws.com/",
                     params={"list-type": "2", "prefix": prefix})
            found = [k for k in re.findall(r"<Key>([^<]+)</Key>", r.text if r else "")
                     if f"-{ch}-{TILE}_" in k]
            if not found:
                break
            keys.append((bucket, found[0]))
        if len(keys) == len(BANDS):
            return keys
    return None


def _read_tile(bucket: str, key: str):
    # h5py (not netCDF4): the netCDF C library is not thread-safe
    import io
    import h5py
    r = _get(f"https://{bucket}.s3.amazonaws.com/{key}")
    if r is None:
        raise IOError(f"vanished: {key}")
    with h5py.File(io.BytesIO(r.content), "r") as h:
        scale = 1 if "C03" in key else IR_FACTOR
        if (int(h.attrs["tile_row_offset"][0]) * scale != TILE_ROW0
                or int(h.attrs["tile_column_offset"][0]) * scale != TILE_COL0):
            raise ValueError(f"unexpected tile offsets in {key}")
        v = h["Sectorized_CMI"]
        raw = v[()]
        out = (raw.astype(np.float32) * float(v.attrs["scale_factor"][0])
               + float(v.attrs["add_offset"][0]))
        if "_FillValue" in v.attrs:
            out[raw == v.attrs["_FillValue"][0]] = np.nan
        return out


def extract_scan(dt: datetime, out_root: str, overwrite: bool = False):
    """Fetch, crop and save one scan. Returns (status, info).

    status: "ok" | "skip" (already done) | "missing" (not in archive)
    Raises on download/processing errors (caller logs them for retry).
    """
    npz_path, png_path = output_paths(dt, out_root)
    if not overwrite and os.path.exists(npz_path) and os.path.exists(png_path):
        return "skip", png_path

    keys = scan_files(dt)
    if keys is None:
        return "missing", "tile not in archive"

    vis = _read_tile(*keys[0])[ROWS, COLS]
    ir_rows = slice(ROWS.start // IR_FACTOR, ROWS.stop // IR_FACTOR)
    ir_cols = slice(COLS.start // IR_FACTOR, COLS.stop // IR_FACTOR)
    ir = _read_tile(*keys[1])[ir_rows, ir_cols]
    ir = ir.repeat(IR_FACTOR, axis=0).repeat(IR_FACTOR, axis=1)
    bands = np.stack([vis, ir]).astype(np.float16)

    from PIL import Image
    os.makedirs(os.path.dirname(npz_path), exist_ok=True)
    # Write to temp names then rename, so an interrupted run never leaves a
    # half-written file that the resume check would treat as done.
    with open(npz_path + ".tmp", "wb") as f:
        np.savez_compressed(f, bands=bands)
    Image.fromarray(rgb_to_uint8(bands_to_rgb(bands))).save(
        png_path + ".tmp", format="PNG")
    os.replace(npz_path + ".tmp", npz_path)
    os.replace(png_path + ".tmp", png_path)
    return "ok", png_path


def _worker(dt: datetime, out_root: str):
    try:
        return dt, *extract_scan(dt, out_root)
    except Exception as e:
        return dt, "error", f"{type(e).__name__}: {e}"[:200]


# ── Batch driver ──────────────────────────────────────────────────────────────

def _load_missing(out_root: str) -> set:
    path = os.path.join(out_root, MISSING_CSV)
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return {row["timestamp"] for row in csv.DictReader(f)
                if row["reason"] == "missing"}


def _log_missing(out_root: str, dt: datetime, reason: str, detail: str):
    path = os.path.join(out_root, MISSING_CSV)
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["timestamp", "reason", "detail"])
        w.writerow([dt.strftime("%Y-%m-%dT%H:%M"), reason, detail])


def run_range(start: datetime, end: datetime, out_root: str, workers: int = 32):
    os.makedirs(out_root, exist_ok=True)
    known_missing = _load_missing(out_root)

    todo = []
    for dt in daytime_times(start, end):
        npz, png = output_paths(dt, out_root)
        if os.path.exists(npz) and os.path.exists(png):
            continue
        if dt.strftime("%Y-%m-%dT%H:%M") in known_missing:
            continue
        todo.append(dt)

    print(f"{start:%Y-%m-%d %H:%M} → {end:%Y-%m-%d %H:%M} UTC: "
          f"{len(todo)} scans to process ({workers} workers)", flush=True)
    counts = {"ok": 0, "skip": 0, "missing": 0, "error": 0}
    if not todo:
        return counts

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_worker, dt, out_root) for dt in todo]
        for i, fut in enumerate(as_completed(futures), 1):
            dt, status, info = fut.result()
            counts[status] += 1
            if status in ("missing", "error"):
                _log_missing(out_root, dt, status, info)
                if status == "error":
                    print(f"  ✗ {dt:%Y-%m-%d %H:%M} {info}", flush=True)
            if i % 200 == 0 or i == len(todo):
                el = time.time() - t0
                eta = el / i * (len(todo) - i)
                print(f"  [{i}/{len(todo)}] ok={counts['ok']} "
                      f"missing={counts['missing']} error={counts['error']} "
                      f"({i / el * 3600:.0f} scans/h, ETA {eta / 3600:.1f} h)",
                      flush=True)
    return counts


def _parse_time(s: str, is_end: bool) -> datetime:
    dt = datetime.fromisoformat(s)
    if len(s) == 10 and is_end:          # bare date → include the whole day
        dt += timedelta(days=1) - timedelta(minutes=1)
    return dt.replace(tzinfo=None)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--start", required=True, help="UTC, e.g. 2024-01-01 or 2024-01-01T04:00")
    p.add_argument("--end",   required=True, help="UTC, inclusive (bare date = whole day)")
    p.add_argument("--out",   required=True, help="output root directory")
    p.add_argument("--workers", type=int, default=32)
    args = p.parse_args()
    run_range(_parse_time(args.start, False), _parse_time(args.end, True),
              args.out, args.workers)


if __name__ == "__main__":
    main()

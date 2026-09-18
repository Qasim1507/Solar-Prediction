#!/usr/bin/env python3
"""
fetch_training_images.py — download only the scans the training set uses.

data/combined_dataset.csv is hourly, while the Himawari archive holds a scan
every 10 minutes. Training therefore touches 9,843 images, not the 76,886 a
full-range download produces: ~2.4 GB instead of ~51 GB.

This reads the image_path column and fetches exactly those timestamps,
reusing himawari_aws.extract_scan (same crop, same output, byte-identical).
Resumable: scans already on disk are skipped.

    python scripts/fetch_training_images.py
    python scripts/fetch_training_images.py --csv data/combined_dataset.csv --workers 32
"""

import argparse
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from himawari_aws import extract_scan

STAMP = re.compile(r"himawari_sg_(\d{8})_(\d{6})\.png$")


def timestamps_from_csv(csv_path: str):
    """Parse the scan times referenced by the dataset's image_path column."""
    col = pd.read_csv(csv_path, usecols=["image_path"])["image_path"].dropna()
    out = []
    for p in col.unique():
        m = STAMP.search(str(p))
        if m:
            out.append(datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S"))
    return sorted(out)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--csv", default="data/combined_dataset.csv")
    p.add_argument("--out", default="data/satellite_aws")
    p.add_argument("--workers", type=int, default=32)
    args = p.parse_args()

    times = timestamps_from_csv(args.csv)
    if not times:
        raise SystemExit(f"No himawari_sg_*.png timestamps found in {args.csv}")
    print(f"{len(times):,} scans referenced by {args.csv} "
          f"({times[0]:%Y-%m-%d} .. {times[-1]:%Y-%m-%d}), {args.workers} workers",
          flush=True)

    counts = {"ok": 0, "skip": 0, "missing": 0, "error": 0}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(extract_scan, t, args.out): t for t in times}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                status, _ = f.result()
            except Exception as e:
                status = "error"
                print(f"  {futs[f]:%Y-%m-%d %H:%M} {type(e).__name__}: {e}"[:160],
                      flush=True)
            counts[status] = counts.get(status, 0) + 1
            if i % 250 == 0 or i == len(times):
                print(f"  [{i}/{len(times)}] " +
                      " ".join(f"{k}={v}" for k, v in counts.items()), flush=True)

    print(f"\nDone. {counts['ok']} downloaded, {counts['skip']} already present, "
          f"{counts['missing']} not in archive, {counts['error']} errors.")
    if counts["error"]:
        print("Re-run to retry the errors (finished scans are skipped).")


if __name__ == "__main__":
    main()

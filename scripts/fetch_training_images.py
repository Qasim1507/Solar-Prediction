#!/usr/bin/env python3
"""
fetch_training_images.py — download only the scans the training set uses.

data/combined_dataset_v2.csv is hourly, while the Himawari archive holds a scan
every 10 minutes. Training therefore touches a small fraction of the 76,886
images a full-range download produces.

Each row feeds the model a frame series — t, t-10min, t-20min, per
combined_dataset.FRAME_OFFSETS_MINUTES — so this fetches those offsets around
every timestamp in the image_path column: ~29,500 scans, ~7 GB, well under the
51 GB full range.

The offsets are derived from the timestamps rather than read from the
image_path_prev* columns, so this works on a CSV built before those columns
existed — which is the point, since the prev frames must be on disk before
combined_dataset.py can match them.

Uses himawari_aws.extract_scan, so output is byte-identical to a full
download. Resumable: scans already on disk are skipped.

    python scripts/fetch_training_images.py
    python scripts/fetch_training_images.py --csv data/combined_dataset_v2.csv --workers 32
"""

import argparse
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from combined_dataset import FRAME_OFFSETS_MINUTES
from himawari_aws import extract_scan

STAMP = re.compile(r"himawari_sg_(\d{8})_(\d{6})\.png$")


def timestamps_from_csv(csv_path: str, offsets=FRAME_OFFSETS_MINUTES):
    """Scan times the dataset needs: each image_path, expanded by the offsets."""
    col = pd.read_csv(csv_path, usecols=["image_path"])["image_path"].dropna()
    out = set()
    for p in col.unique():
        m = STAMP.search(str(p))
        if not m:
            continue
        t = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
        for off in offsets:
            out.add(t - timedelta(minutes=off))
    return sorted(out)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--csv", default="data/combined_dataset_v2.csv")
    p.add_argument("--out", default="data/satellite_aws")
    p.add_argument("--workers", type=int, default=32)
    args = p.parse_args()

    times = timestamps_from_csv(args.csv)
    if not times:
        raise SystemExit(f"No himawari_sg_*.png timestamps found in {args.csv}")
    spacing = ", ".join(f"t-{o}min" if o else "t" for o in FRAME_OFFSETS_MINUTES)
    print(f"{len(times):,} scans referenced by {args.csv} "
          f"({times[0]:%Y-%m-%d} .. {times[-1]:%Y-%m-%d})", flush=True)
    print(f"  frame series: {spacing} | {args.workers} workers", flush=True)

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

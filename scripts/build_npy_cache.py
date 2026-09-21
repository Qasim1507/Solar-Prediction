"""
build_npy_cache.py — PNG -> 224x224 float32 .npy cache, headless.

Same output as notebook Section 3 (cell 9), runnable without Jupyter so a pod
can go straight from download to training.

    python scripts/build_npy_cache.py --csv data/combined_dataset_v2.csv --workers 8

RGB is preserved: the AWS composite carries visible reflectance in R/G and
infrared in B, so a grayscale conversion would throw the IR signal away.
"""
import argparse, os, sys, time
import numpy as np
import pandas as pd
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from combined_dataset import FRAME_OFFSETS_MINUTES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRAME_COLS = ["image_path"] + [f"image_path_prev{n}"
                               for n in range(1, len(FRAME_OFFSETS_MINUTES))]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="data/combined_dataset_v2.csv")
    p.add_argument("--sat-dir", default="data/satellite_aws")
    p.add_argument("--npy-dir", default="data/satellite_aws_npy")
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()

    df = pd.read_csv(a.csv)
    cols = [c for c in FRAME_COLS if c in df.columns]
    paths = pd.unique(df[cols].values.ravel("K"))
    paths = [q for q in paths if isinstance(q, str)]

    def npy_for(q):
        rel = os.path.relpath(os.path.join(ROOT, q), os.path.join(ROOT, a.sat_dir))
        return os.path.join(ROOT, a.npy_dir, rel).replace(".png", ".npy")

    todo = [q for q in paths if not os.path.exists(npy_for(q))]
    print(f"{len(paths)} frames referenced, {len(paths)-len(todo)} cached, "
          f"{len(todo)} to build")
    if not todo:
        return

    done = [0]
    t0 = time.time()

    def build(q):
        src = os.path.join(ROOT, q)
        dst = npy_for(q)
        try:
            if not os.path.exists(src):
                return f"missing PNG: {q}"
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with Image.open(src) as im:
                rgb = im.convert("RGB").resize((224, 224), Image.BILINEAR)
                arr = np.array(rgb, dtype=np.float32) / 255.0
            np.save(dst, arr)
        except Exception as e:
            return f"{q}: {e}"
        finally:
            done[0] += 1
            if done[0] % 2000 == 0:
                el = time.time() - t0
                print(f"  {done[0]}/{len(todo)}  {el/60:.1f}m  "
                      f"~{(len(todo)-done[0])/max(done[0]/el,1e-9)/60:.1f}m left",
                      flush=True)
        return None

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        errs = [e for e in ex.map(build, todo) if e]

    print(f"built {len(todo)-len(errs)} in {(time.time()-t0)/60:.1f}m")
    for e in errs[:20]:
        print("  ERR", e)
    if len(errs) > 20:
        print(f"  ... {len(errs)-20} more")


if __name__ == "__main__":
    main()

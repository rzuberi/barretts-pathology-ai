"""Part 2 step 0b: slide QC over the ISBI slide table (shardable, CPU).
Per slide: kept tiles, tissue tiles, grid tiles (from h5 attrs), and on 64 sampled tiles read from the WSI:
Laplacian-variance / Tenengrad medians, brightness, white fraction. Exclusion applied by the finalizer with
thresholds FIXED in advance: kept_tiles < 50, or white_frac > 0.98, or lapvar_median < corpus 1st percentile.
Env: SHARD, N_SHARDS. Output: analysis/isbi_rerun/qc/qc_shard<k>.csv"""
import csv, os
import numpy as np, pandas as pd, h5py, openslide
from scipy.ndimage import laplace, sobel
A = "/mnt/scratche/slow/fmlab/zuberi01/analysis/isbi_rerun"; Q = A + "/qc"; os.makedirs(Q, exist_ok=True)
s = pd.read_csv(A + "/slides.csv", dtype={"idx": int}); SHARD, NS = int(os.environ.get("SHARD", 0)), int(os.environ.get("N_SHARDS", 1))
K = 64; out = f"{Q}/qc_shard{SHARD}.csv"
done = set(pd.read_csv(out).idx) if os.path.exists(out) else set()
if not os.path.exists(out):
    with open(out, "w", newline="") as f: csv.writer(f).writerow(["idx", "kept_tiles", "tissue_tiles", "grid_tiles", "n_scored", "lapvar_median", "lapvar_p10", "tenengrad_median", "brightness_median", "white_frac", "error"])
for i in range(SHARD, len(s), NS):
    if i in done: continue
    r = s.iloc[i]; row = [i] + [""] * 10
    try:
        with h5py.File(r.h5) as h:
            coords = np.asarray(h["coords"]); level = int(h.attrs.get("level", 0)); sp = str(h.attrs.get("slide_path", ""))
            kept = len(coords); tissue = int(h.attrs.get("tissue_tiles", kept)); grid = int(h.attrs.get("grid_tiles", 0))
        row[1:4] = [kept, tissue, grid]
        if kept == 0: row[10] = "no_tiles"
        elif not os.path.exists(sp): row[10] = "slide_missing"
        else:
            sl = openslide.OpenSlide(sp); ds = sl.level_downsamples[level]
            pick = coords[np.random.RandomState(0).choice(kept, min(K, kept), replace=False)]
            lap, ten, bri, white = [], [], [], []
            for x, y in pick:
                try: a = np.asarray(sl.read_region((int(x * ds), int(y * ds)), level, (224, 224)).convert("L"), dtype=np.float32)
                except Exception: continue
                lap.append(float(laplace(a).var())); gx, gy = sobel(a, 0), sobel(a, 1); ten.append(float((gx ** 2 + gy ** 2).mean()))
                bri.append(float(a.mean())); white.append(float((a > 230).mean()))
            sl.close()
            if lap: row[4:10] = [len(lap), round(np.median(lap), 2), round(np.percentile(lap, 10), 2), round(np.median(ten), 1), round(np.median(bri), 1), round(float(np.mean(white)), 3)]
            else: row[10] = "read_failed"
    except Exception as e:
        row[10] = type(e).__name__
    with open(out, "a", newline="") as f: csv.writer(f).writerow(row)
print(f"qc shard {SHARD}/{NS} complete", flush=True)

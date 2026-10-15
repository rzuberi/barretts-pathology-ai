"""Slide QC (Rehan 2026-09-03): blur/quality screen over ERIN WSIs.

Per slide: sample K tiles at the stored feature coords, compute
variance-of-Laplacian (focus), tenengrad, brightness and saturation stats.
Output per-slide summary rows; thresholds chosen at analysis time from the
corpus distribution (no hard-coded blur cutoff). CPU-only, shardable.
"""
import csv, os
import h5py
import numpy as np
import pandas as pd
from scipy.ndimage import laplace, sobel

T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"
OUT = os.environ.get("OUTDIR", ".")
SHARD, N_SHARDS = int(os.environ.get("SHARD", 0)), int(os.environ.get("N_SHARDS", 1))
K = int(os.environ.get("K_TILES", 64))
import openslide

m = pd.read_csv(T + "/labeller/erin_master.csv", dtype=str).dropna(subset=["h5"]).drop_duplicates("h5")
m = m.iloc[SHARD::N_SHARDS]
out = os.path.join(OUT, f"qc_shard{SHARD}.csv")
done = set()
if os.path.exists(out):
    done = set(pd.read_csv(out)["h5"])
if not os.path.exists(out):
    with open(out, "w", newline="") as f:
        csv.writer(f).writerow(["h5", "slide_path", "n_tiles_scored",
                                "lapvar_median", "lapvar_p10", "lapvar_p90",
                                "tenengrad_median", "brightness_median",
                                "sat_frac_white", "error"])

def score_slide(h5p):
    with h5py.File(h5p) as h:
        sp = h.attrs.get("slide_path", "")
        level = int(h.attrs.get("level", 0))
        coords = np.asarray(h["coords"])
    if not os.path.exists(sp):
        return sp, None, "slide_missing"
    sl = openslide.OpenSlide(sp)
    ds = sl.level_downsamples[level]
    rng = np.random.RandomState(0)
    pick = coords[rng.choice(len(coords), min(K, len(coords)), replace=False)]
    lap, ten, bri, white = [], [], [], []
    for x, y in pick:
        try:
            img = sl.read_region((int(x), int(y)), level, (224, 224)).convert("L")
        except Exception:
            continue
        a = np.asarray(img, dtype=np.float32)
        lap.append(float(laplace(a).var()))
        gx, gy = sobel(a, 0), sobel(a, 1)
        ten.append(float((gx ** 2 + gy ** 2).mean()))
        bri.append(float(a.mean()))
        white.append(float((a > 230).mean()))
    sl.close()
    if not lap:
        return sp, None, "no_tiles_read"
    lap, ten, bri, white = map(np.array, (lap, ten, bri, white))
    return sp, {"n": len(lap),
                "lv_med": np.median(lap), "lv_p10": np.percentile(lap, 10),
                "lv_p90": np.percentile(lap, 90),
                "tg_med": np.median(ten), "br_med": np.median(bri),
                "white": float(white.mean())}, ""

todo = [h for h in m["h5"] if h not in done]
print(f"shard {SHARD}: {len(todo)} slides", flush=True)
for i, h5p in enumerate(todo):
    try:
        sp, s, err = score_slide(h5p)
    except Exception as e:
        sp, s, err = "", None, f"{type(e).__name__}"
    with open(out, "a", newline="") as f:
        w = csv.writer(f)
        if s:
            w.writerow([h5p, sp, s["n"], round(s["lv_med"], 2), round(s["lv_p10"], 2),
                        round(s["lv_p90"], 2), round(s["tg_med"], 2),
                        round(s["br_med"], 1), round(s["white"], 3), ""])
        else:
            w.writerow([h5p, sp, 0, "", "", "", "", "", "", err])
    if i % 20 == 0: print(f"{i}/{len(todo)}", flush=True)
print("shard complete", flush=True)

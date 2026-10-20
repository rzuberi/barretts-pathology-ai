"""Phase 1 (CPU, one task per slide): download a public WSI (GDC / CAMELYON16 S3) if missing, build a tissue mask, and read every tissue
tile (224 px at 0.5 um/px) into data/tiles/{ds}/{slide}.npy (uint8, N x 224 x 224 x 3) + {slide}.json (level-0 top-left coords, timings).
Usage: tile_slide.py DS SLIDE [--workers 4].  Idempotent: skips if the json exists. See DESIGN.md for the tissue rule."""
import argparse, json, os, sys, time, urllib.request
import threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd
from PIL import Image, ImageFilter
A = "/mnt/scratche/slow/fmlab/zuberi01/analysis/outreach_tilebio"; TILE, MPP = 224, 0.5
ap = argparse.ArgumentParser(); ap.add_argument("ds"); ap.add_argument("slide"); ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
a = ap.parse_args()
row = pd.read_csv(A + "/manifests/slides.csv").set_index(["ds", "slide"]).loc[(a.ds, a.slide)]
W_DIR, T_DIR = f"{A}/data/wsi/{a.ds}", os.environ.get("TILE_OUT", f"{A}/data/tiles/{a.ds}"); os.makedirs(W_DIR, exist_ok=True); os.makedirs(T_DIR, exist_ok=True)
wsi, out_npy, out_js = f"{W_DIR}/{row.fname}", f"{T_DIR}/{a.slide}.npy", f"{T_DIR}/{a.slide}.json"
if os.path.exists(out_js) and os.path.exists(out_npy): print("done already"); sys.exit(0)
t0 = time.time()
if not (os.path.exists(wsi) and os.path.getsize(wsi) == int(row["size"])):
    for attempt in range(4):
        try:
            with urllib.request.urlopen(row.url, timeout=1800) as r, open(wsi + ".part", "wb") as fh:
                while True:
                    b = r.read(1 << 23)
                    if not b: break
                    fh.write(b)
            assert os.path.getsize(wsi + ".part") == int(row["size"]), "size mismatch"
            os.replace(wsi + ".part", wsi); break
        except Exception as e:
            print(f"download attempt {attempt}: {type(e).__name__}: {e}", flush=True); time.sleep(20 * (attempt + 1))
    else: sys.exit("download failed")
t_dl = time.time() - t0
import openslide
sl = openslide.OpenSlide(wsi); W0, H0 = sl.dimensions
mpp0 = float(sl.properties.get("openslide.mpp-x") or 0)
if not mpp0:   # generic TIFF without openslide mpp: fall back to the TIFF resolution tag (pixels per cm)
    xr = sl.properties.get("tiff.XResolution"); unit = sl.properties.get("tiff.ResolutionUnit", "")
    mpp0 = (1e4 / float(xr) if unit == "centimeter" else 25400 / float(xr)) if xr else 0.25
lv = max([l for l in range(sl.level_count) if mpp0 * sl.level_downsamples[l] <= MPP * 1.05] or [0])
ds_l = sl.level_downsamples[lv]; mpp_l = mpp0 * ds_l; R = int(round(TILE * MPP / mpp_l))   # read R x R at level lv -> resize to 224
S0 = R * ds_l                                                                               # tile side in level-0 px
# ---- tissue mask on a thumbnail with 8 px per tile ----
t1 = time.time(); sc = S0 / 8.0; tw, th = int(W0 / sc), int(H0 / sc)
th_img = sl.get_thumbnail((tw, th)).convert("RGB"); tw, th = th_img.size; sx, sy = W0 / tw, H0 / th
hsv = np.asarray(th_img.convert("HSV")); S = np.asarray(Image.fromarray(hsv[..., 1]).filter(ImageFilter.MedianFilter(7))).astype(np.float32); V = hsv[..., 2]
def otsu(x):
    h, e = np.histogram(x, 256, (0, 256)); p = h / h.sum(); w = np.cumsum(p); mu = np.cumsum(p * np.arange(256)); mt = mu[-1]
    sb = (mt * w - mu) ** 2 / np.maximum(w * (1 - w), 1e-12); return float(np.argmax(sb))
thr = max(15.0, otsu(S)); tissue = (S > thr) & (V > 40)
nx, ny = int(W0 // S0), int(H0 // S0); coords = []
for j in range(ny):
    for i in range(nx):
        x0, y0 = i * S0, j * S0; blk = tissue[int(y0 / sy):max(int(y0 / sy) + 1, int((y0 + S0) / sy)), int(x0 / sx):max(int(x0 / sx) + 1, int((x0 + S0) / sx))]
        if blk.size and blk.mean() >= 0.25: coords.append((int(round(x0)), int(round(y0))))
coords = np.array(coords, np.int64).reshape(-1, 2); N = len(coords); t_mask = time.time() - t1
print(f"{a.ds}/{a.slide}: {W0}x{H0} mpp0={mpp0:.4f} level={lv} read={R}px otsu={thr:.0f} tissue tiles {N} of grid {nx*ny}", flush=True)
# ---- read tiles with threads (forking after OpenSlide is open deadlocked in the smoke; ctypes reads release the GIL) ----
sl.close(); part = out_npy + ".part.npy"; mm = np.lib.format.open_memmap(part, mode="w+", dtype=np.uint8, shape=(max(N, 0), TILE, TILE, 3))
tls = threading.local()
def work(chunk):
    if not hasattr(tls, "s"): tls.s = openslide.OpenSlide(wsi)
    for k in chunk:
        x, y = coords[k]; im = tls.s.read_region((int(x), int(y)), lv, (R, R)).convert("RGB")
        if R != TILE: im = im.resize((TILE, TILE), Image.BILINEAR)
        mm[k] = np.asarray(im)
    return len(chunk)
t2 = time.time()
if N:
    with ThreadPoolExecutor(a.workers) as ex: assert sum(ex.map(work, np.array_split(np.arange(N), max(1, min(N, a.workers * 8))))) == N
mm.flush(); del mm
t_read = time.time() - t2
os.replace(part, out_npy)
meta = {"ds": a.ds, "slide": a.slide, "n_tiles": int(N), "grid": int(nx * ny), "W0": W0, "H0": H0, "mpp0": mpp0, "level": lv, "read_px": R,
        "tile_side_level0": S0, "otsu": thr, "coords_level0_topleft": coords.tolist(), "t_download_s": round(t_dl, 1), "t_mask_s": round(t_mask, 2),
        "t_read_s": round(t_read, 1), "read_workers": a.workers, "bytes": int(N * TILE * TILE * 3)}
json.dump(meta, open(out_js + ".part", "w")); os.replace(out_js + ".part", out_js)
print(f"done: {N} tiles, download {t_dl:.0f}s, mask {t_mask:.1f}s, read {t_read:.0f}s ({a.workers} workers)", flush=True)

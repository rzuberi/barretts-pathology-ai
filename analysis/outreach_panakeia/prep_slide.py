"""CPU unit, one slide: (1) make sure the GDC slide is present and its MD5 matches (download via the GDC API otherwise),
(2) read up to 2,000 random tissue tiles at 0.5 um/px (224 px) into tiles/{patient}.npy + .json.
Tissue rule copied from campaigns/extract_one.py (thumbnail mean < 220 and std > 4; tile mean <= 235 and std >= 8).
Usage: prep_slide.py PATIENT"""
import hashlib, json, os, sys, time, urllib.request, zlib
import numpy as np, pandas as pd

A = "/mnt/scratche/slow/fmlab/zuberi01/analysis/outreach_panakeia"; T = A + "/tiles"; os.makedirs(T, exist_ok=True)
CAP, TILE, MPP = 2000, 224, 0.5
pid = sys.argv[1]
r = pd.read_csv(A + "/cohort.csv").set_index("patient").loc[pid]
out = f"{T}/{pid}.npy"
if os.path.exists(out) and os.path.exists(f"{T}/{pid}.json"): print("done already"); sys.exit(0)


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""): h.update(b)
    return h.hexdigest()


path = r.slide_path; t0 = time.time()
ok = os.path.exists(path) and os.path.getsize(path) == r.file_size and md5(path) == r.md5
if not ok:
    tmp = path + ".part"
    for attempt in range(4):
        try:
            with urllib.request.urlopen(f"https://api.gdc.cancer.gov/data/{r.file_id}", timeout=300) as u, open(tmp, "wb") as f:
                while True:
                    b = u.read(1 << 24)
                    if not b: break
                    f.write(b)
            if os.path.getsize(tmp) == r.file_size and md5(tmp) == r.md5: os.replace(tmp, path); ok = True; break
            print("size/md5 mismatch, retry", attempt, flush=True)
        except Exception as e:
            print("download error", attempt, e, flush=True); time.sleep(20 * (attempt + 1))
    if not ok: sys.exit("download failed")
    print(f"downloaded {r.file_size / 1e9:.2f} GB in {time.time() - t0:.0f}s", flush=True)
else:
    print(f"present, md5 ok ({time.time() - t0:.0f}s)", flush=True)

import openslide
from PIL import Image
sl = openslide.OpenSlide(path)
if "openslide.mpp-x" not in sl.properties:   # e.g. 1600x1200 Aperio snapshots filed as diagnostic slides: excluded, counted in the flow
    np.save(out + ".part.npy", np.zeros((0, TILE, TILE, 3), np.uint8)); os.replace(out + ".part.npy", out)
    json.dump({"patient": pid, "slide_path": path, "excluded": "no mpp metadata", "dims0": list(sl.dimensions), "n_tiles": 0,
               "coords_level0_topleft": [], "mpp0": None, "tile_px_level0": None}, open(f"{T}/{pid}.json", "w"))
    print(f"{pid}: no mpp metadata, dims {sl.dimensions}: excluded (0 tiles)"); sys.exit(0)
mpp0 = float(sl.properties["openslide.mpp-x"])
f = MPP / mpp0                     # level-0 px per target px (about 2 for 40x scans, 1 for 20x)
lvl = max([l for l in range(sl.level_count) if sl.level_downsamples[l] <= f * 1.05] or [0])
ds = sl.level_downsamples[lvl]; S0 = TILE * f; read = int(round(S0 / ds))
W0, H0 = sl.dimensions
thumb = np.asarray(sl.get_thumbnail((2048, 2048)).convert("L"), dtype=np.float32)
sy, sx = thumb.shape[0] / H0, thumb.shape[1] / W0
cells = []
for y in np.arange(0, H0 - S0, S0):
    for x in np.arange(0, W0 - S0, S0):
        p = thumb[int(y * sy):max(int(y * sy) + 1, int((y + S0) * sy)), int(x * sx):max(int(x * sx) + 1, int((x + S0) * sx))]
        if p.size and p.mean() < 220 and p.std() > 4: cells.append((int(x), int(y)))
rng = np.random.RandomState(zlib.crc32(pid.encode()))
order = rng.permutation(len(cells))
tiles, kept = [], []
for i in order:                    # walk the random order until CAP tiles pass the per-tile check
    x, y = cells[i]
    t = sl.read_region((x, y), lvl, (read, read)).convert("RGB")
    if read != TILE: t = t.resize((TILE, TILE), Image.BILINEAR)
    t = np.asarray(t)
    if t.mean() > 235 or t.std() < 8: continue
    tiles.append(t); kept.append((x, y))
    if len(tiles) == CAP: break
X = np.stack(tiles) if tiles else np.zeros((0, TILE, TILE, 3), np.uint8)
np.save(out + ".part.npy", X); os.replace(out + ".part.npy", out)
meta = {"patient": pid, "slide_path": path, "mpp0": mpp0, "level_read": lvl, "read_px": read, "tile_px_level0": S0,
        "n_tissue_cells": len(cells), "n_tiles": len(kept), "coords_level0_topleft": kept, "dims0": [W0, H0]}
json.dump(meta, open(f"{T}/{pid}.json.part", "w")); os.replace(f"{T}/{pid}.json.part", f"{T}/{pid}.json")
print(f"{pid}: mpp0={mpp0:.3f} lvl={lvl} read={read} cells={len(cells)} tiles={len(kept)} total {time.time() - t0:.0f}s")

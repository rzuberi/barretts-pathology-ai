"""Exploration E1 (docs/paper_explore_E1.md), CPU only: per-tile pixel statistics at the 256 stored UNI2-h tile positions of every analysed slide
(the R55 reading code of rv7_round7.py, extended from a per-slide count to per-tile values). No pixels are saved, only per-tile numbers:
white = share of pixels with all RGB > 220 (R55: background tile if > 0.5); dark = share with grey < 50; sat = mean HSV saturation; grey = mean grey;
ink = share of saturated green-blue pixels (hue 80-250 degrees, saturation > 0.25, not white; marker pen); lapvar = variance of the Laplacian of grey (focus);
mask_own / mask_nb = tissue-mask share inside the tile and inside the 3 x 3-tile neighbourhood (edge tile = low neighbourhood share).
Row-level output (cluster only): killcoyne_mm/explore_E1/px/chunk_<k>.csv.   TASK = px_<k> (k = 0..39)"""
import os, glob, numpy as np, pandas as pd
import openslide
from PIL import Image
from scipy.ndimage import laplace
Image.MAX_IMAGE_PIXELS = None
T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"; M = T + "/feasibility/paper_plan/killcoyne_mm"; KD = T + "/feasibility/paper_plan/killcoyne"; HZ = M + "/horizons"
OD = M + "/explore_E1/px"; os.makedirs(OD, exist_ok=True); NCH = 40
remap = lambda p: p.replace("/scratchc/", "/mnt/scratche/fast/") if isinstance(p, str) and p.startswith("/scratchc/") else p
ch = int(os.environ["TASK"].split("_")[1]); outf = f"{OD}/chunk_{ch:02d}.csv"
if os.path.exists(outf): raise SystemExit(0)
ids = set(pd.read_csv(HZ + "/samples.csv", dtype={"Sample": str}).Sample)
mp = pd.read_csv(KD + "/uni2_npz_map.csv", dtype=str); mp = mp[mp.cnv.isin(ids)].drop_duplicates(["cnv", "npz"]).sort_values(["cnv", "npz"]).reset_index(drop=True)
out = []
for r in mp.iloc[ch::NCH].itertuples():
    with np.load(r.npz, allow_pickle=True) as z: sp = remap(str(z["slide_path"])); lv = int(z["level"]); ts = int(z["tile_size"]); xy = np.asarray(z["coords_level"])
    s = openslide.OpenSlide(sp); ds = s.level_downsamples[lv]; W_, H_ = s.level_dimensions[lv]
    md = os.path.join(os.path.dirname(os.path.dirname(sp)), "masks"); cand = sorted(glob.glob(os.path.join(md, glob.escape(os.path.splitext(os.path.basename(sp))[0]) + "*")))
    mk = (np.array(Image.open(cand[0]).convert("L")) > 0) if cand else None
    for t, (x, yy) in enumerate(xy):
        im = np.asarray(s.read_region((int(round(x * ds)), int(round(yy * ds))), lv, (ts, ts)).convert("RGB")).astype(np.float32)
        wh = (im > 220).all(-1); grey = im.mean(-1); mx = im.max(-1); mn = im.min(-1); sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0)
        rr, gg, bb = im[..., 0], im[..., 1], im[..., 2]; d = np.maximum(mx - mn, 1e-6)
        hue = np.where(mx == rr, (60 * (gg - bb) / d) % 360, np.where(mx == gg, 60 * (bb - rr) / d + 120, 60 * (rr - gg) / d + 240))
        ink = (hue >= 80) & (hue <= 250) & (sat > 0.25) & ~wh
        o = {"key": r.cnv, "npz": r.npz, "tile": t, "white": float(wh.mean()), "dark": float((grey < 50).mean()), "sat": float(sat.mean()), "grey": float(grey.mean()),
             "ink": float(ink.mean()), "lapvar": float(laplace(grey).var())}
        if mk is not None:
            mh, mw = mk.shape; fx, fy = mw / W_, mh / H_
            def share(x0, y0, x1, y1):
                a = mk[int(np.clip(np.floor(y0 * fy), 0, mh - 1)):int(np.clip(np.ceil(y1 * fy), 1, mh)), int(np.clip(np.floor(x0 * fx), 0, mw - 1)):int(np.clip(np.ceil(x1 * fx), 1, mw))]
                return float(a.mean()) if a.size else np.nan
            o["mask_own"] = share(x, yy, x + ts, yy + ts); o["mask_nb"] = share(x - ts, yy - ts, x + 2 * ts, yy + 2 * ts)
        else: o["mask_own"] = o["mask_nb"] = np.nan
        out.append(o)
    s.close()
pd.DataFrame(out).to_csv(outf, index=False); print("EXE1 PX DONE", ch, len(out), flush=True)

"""Two papers, agent ART (docs/two_papers_ART.md), phase 1 (CPU only; the only script that opens slides): per-tile artefact / QC statistics for every
tile of every L-IMG bag (set C, 676 samples; 256 UNI2-h tiles per slide, 224 px at level 2, about 0.88 um/px), read at the CORRECT position.
Tile rule of extract_foundation_from_masks.py (barretts_retraining): coords_level are tile CENTRES at the stored level; top-left = centre - tile_size // 2,
clamped to the level; read_region at level 2. (ex_e1_px.py read coords_level as top-left corners: a half-tile shift.)
Bag order = km_prep.py: the sample's slides sorted by npz path, stored tile order (asserted against bag_index n_tiles in the merge).
Per tile (no pixels saved): pixel classes (HSV saturation s, grey g = RGB mean)
  white  = all RGB > 220 (R55 / E1 background pixel)
  tissue = s > 0.15 and not white (P1B panel criterion)
  glass  = not white, s < 0.12, g >= 160 (grey glass, faint unstained areas)
  dlow   = s < 0.12, g < 160 (dark unstained: coverslip edges, bubble rims, debris, black marker)
  ink    = hue 80-250 deg, s > 0.25, not white (green / blue marker; E1 definition)
  dark   = g < 50 (E1)
focus: lapvar = variance of the Laplacian of grey over the whole tile (E1); lapvar_t = the same over tissue pixels only (1-px border dropped);
       ten_t = mean squared Sobel gradient over tissue pixels.
lines (straight coverslip edge): line = max over rows and columns (3-line running mean) of the share of non-tissue non-white pixels (glass + dlow);
       line_d = the same for dlow pixels only; line_w = the same for white + glass pixels;
       step = max over rows and columns of |mean grey of the 8 lines after - the 8 lines before| (a straight brightness step).
mask_own / mask_nb = tissue-mask share in the tile and in its 3 x 3-tile neighbourhood (E1 definitions, correct position).
Also, per slide: openslide vendor / product, mpp. Row-level outputs (cluster only): killcoyne_mm/two_papers_ART/qc/chunk_<k>.csv(.slides.csv).
TASK = qc_<k> (k = 0..NCH-1)"""
import os, glob, numpy as np, pandas as pd
import openslide
from PIL import Image
from scipy.ndimage import laplace, sobel, uniform_filter1d
Image.MAX_IMAGE_PIXELS = None
T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"; M = T + "/feasibility/paper_plan/killcoyne_mm"; KD = T + "/feasibility/paper_plan/killcoyne"
OD = M + "/two_papers_ART/qc"; os.makedirs(OD, exist_ok=True); NCH = 60
remap = lambda p: p.replace("/scratchc/", "/mnt/scratche/fast/") if isinstance(p, str) and p.startswith("/scratchc/") else p
def bag_slides():
    """(sample, npz) rows in bag order (km_prep.py: set C samples, slides sorted by npz)"""
    ids = pd.read_csv(M + "/set_C.csv", dtype=str).Sample.values; mp = pd.read_csv(KD + "/uni2_npz_map.csv", dtype=str)
    return mp[mp.cnv.isin(set(ids))].sort_values(["cnv", "npz"]).reset_index(drop=True)
def tile_top_left(c, ts, W, H):
    half = ts // 2; return max(0, min(int(c[0] - half), W - ts)), max(0, min(int(c[1] - half), H - ts))
def pixel_stats(im):
    """per-tile statistics of an RGB float array (H x W x 3)"""
    wh = (im > 220).all(-1); g = im.mean(-1); mx = im.max(-1); mn = im.min(-1); s = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0)
    rr, gg, bb = im[..., 0], im[..., 1], im[..., 2]; d = np.maximum(mx - mn, 1e-6)
    hue = np.where(mx == rr, (60 * (gg - bb) / d) % 360, np.where(mx == gg, 60 * (bb - rr) / d + 120, 60 * (rr - gg) / d + 240))
    tis = (s > 0.15) & ~wh; glass = ~wh & (s < 0.12) & (g >= 160); dlow = (s < 0.12) & (g < 160); ink = (hue >= 80) & (hue <= 250) & (s > 0.25) & ~wh
    L = laplace(g); Sx = sobel(g, 0); Sy = sobel(g, 1); inner = np.zeros_like(tis); inner[1:-1, 1:-1] = True; tm = tis & inner
    lmax = lambda a: max(uniform_filter1d(a.mean(1), 3).max(), uniform_filter1d(a.mean(0), 3).max())
    nt = (glass | dlow).astype(np.float32); line = lmax(nt); line_d = lmax(dlow.astype(np.float32)); line_w = lmax((wh | glass).astype(np.float32))
    rm, cm = g.mean(1), g.mean(0); k = 8
    step = max(max(abs(v[i:i + k].mean() - v[i - k:i].mean()) for i in range(k, len(v) - k + 1)) for v in (rm, cm))
    return {"white": float(wh.mean()), "tissue": float(tis.mean()), "glass": float(glass.mean()), "dlow": float(dlow.mean()), "ink": float(ink.mean()),
            "dark": float((g < 50).mean()), "sat": float(s.mean()), "sat_t": float(s[tis].mean()) if tis.any() else np.nan, "grey": float(g.mean()),
            "grey_t": float(g[tis].mean()) if tis.any() else np.nan, "lapvar": float(L.var()),
            "lapvar_t": float(L[tm].var()) if tm.sum() >= 500 else np.nan, "ten_t": float((Sx[tm] ** 2 + Sy[tm] ** 2).mean()) if tm.sum() >= 500 else np.nan,
            "line": float(line), "line_d": float(line_d), "line_w": float(line_w), "step": float(step)}
if __name__ == "__main__" and os.environ["TASK"] == "smoke":
    # reading check: with E1's (top-left) convention, white / lapvar must equal E1's px values for the same tiles; with the centre rule they must not
    E1 = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(M + "/explore_E1/px/chunk_0[0-1].csv"))]); mp = bag_slides(); rows = []
    for npz in list(E1.npz.unique())[:4]:
        with np.load(npz, allow_pickle=True) as z: sp = remap(str(z["slide_path"])); lv = int(z["level"]); ts = int(z["tile_size"]); xy = np.asarray(z["coords_level"])
        s = openslide.OpenSlide(sp); ds = float(s.level_downsamples[lv]); W_, H_ = s.level_dimensions[lv]; e = E1[E1.npz == npz].set_index("tile")
        for t in range(0, len(xy), 16):
            a = pixel_stats(np.asarray(s.read_region((int(round(xy[t][0] * ds)), int(round(xy[t][1] * ds))), lv, (ts, ts)).convert("RGB")).astype(np.float32))
            x0, y0 = tile_top_left(xy[t], ts, W_, H_); b = pixel_stats(np.asarray(s.read_region((int(round(x0 * ds)), int(round(y0 * ds))), lv, (ts, ts)).convert("RGB")).astype(np.float32))
            rows.append({"d_white_e1conv": abs(a["white"] - e.loc[t, "white"]), "d_lap_e1conv": abs(a["lapvar"] - e.loc[t, "lapvar"]) / max(e.loc[t, "lapvar"], 1e-9),
                         "d_white_centre": abs(b["white"] - e.loc[t, "white"])})
        s.close()
    R = pd.DataFrame(rows); print("TPART QC SMOKE n", len(R), "max d_white (E1 convention)", R.d_white_e1conv.max(), "max rel d_lapvar", R.d_lap_e1conv.max(),
                                 "mean d_white (centre rule)", R.d_white_centre.mean(), "share of tiles differing > 0.05", (R.d_white_centre > 0.05).mean(), flush=True)
elif __name__ == "__main__":
    ch = int(os.environ["TASK"].split("_")[1]); outf = f"{OD}/chunk_{ch:02d}.csv"
    if os.path.exists(outf): raise SystemExit(0)
    mp = bag_slides(); keys = sorted(mp.npz.unique()); mine = set(keys[ch::NCH]); out, sl = [], []
    for r in mp[mp.npz.isin(mine)].drop_duplicates("npz").itertuples():
        with np.load(r.npz, allow_pickle=True) as z: sp = remap(str(z["slide_path"])); lv = int(z["level"]); ts = int(z["tile_size"]); xy = np.asarray(z["coords_level"])
        s = openslide.OpenSlide(sp); ds = float(s.level_downsamples[lv]); W_, H_ = s.level_dimensions[lv]; pr = s.properties
        sl.append({"npz": r.npz, "vendor": pr.get("openslide.vendor", ""), "product": pr.get("hamamatsu.Product", pr.get("aperio.ScanScope ID", "")), "mpp": pr.get("openslide.mpp-x", ""),
                   "level": lv, "level_mpp": float(pr.get("openslide.mpp-x", "nan")) * ds, "tile_size": ts, "n_tiles": len(xy)})
        md = os.path.join(os.path.dirname(os.path.dirname(sp)), "masks"); cand = sorted(glob.glob(os.path.join(md, glob.escape(os.path.splitext(os.path.basename(sp))[0]) + "*")))
        mk = (np.array(Image.open(cand[0]).convert("L")) > 0) if cand else None
        for t, c in enumerate(xy):
            x0, y0 = tile_top_left(c, ts, W_, H_)
            im = np.asarray(s.read_region((int(round(x0 * ds)), int(round(y0 * ds))), lv, (ts, ts)).convert("RGB")).astype(np.float32)
            o = {"npz": r.npz, "tile": t, "cx": int(c[0]), "cy": int(c[1]), "x0": x0, "y0": y0, **pixel_stats(im)}
            if mk is not None:
                mh, mw = mk.shape; fx, fy = mw / W_, mh / H_
                def share(a0, b0, a1, b1):
                    a = mk[int(np.clip(np.floor(b0 * fy), 0, mh - 1)):int(np.clip(np.ceil(b1 * fy), 1, mh)), int(np.clip(np.floor(a0 * fx), 0, mw - 1)):int(np.clip(np.ceil(a1 * fx), 1, mw))]
                    return float(a.mean()) if a.size else np.nan
                o["mask_own"] = share(x0, y0, x0 + ts, y0 + ts); o["mask_nb"] = share(x0 - ts, y0 - ts, x0 + 2 * ts, y0 + 2 * ts)
            else: o["mask_own"] = o["mask_nb"] = np.nan
            out.append(o)
        s.close()
    pd.DataFrame(sl).to_csv(outf.replace(".csv", ".slides.csv"), index=False); pd.DataFrame(out).to_csv(outf, index=False); print("TPART QC DONE", ch, len(out), flush=True)

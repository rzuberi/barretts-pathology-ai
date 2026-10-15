"""Horizons H2 (docs/paper_survival_horizons.md @ ee51db8): per set C sample, scanner (hamamatsu.Product) and tissue tiles available (non-overlapping
level-2 224-px cells whose centre is in the tissuetector mask, as dd_describe.py wsi), summed over the sample's slides. Row-level output stays on the
cluster: killcoyne_mm/horizons/slide_desc.csv."""
import os, glob, numpy as np, pandas as pd, openslide
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
Image.MAX_IMAGE_PIXELS = None
T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"; K = T + "/feasibility/paper_plan/killcoyne"; M = K + "_mm"
remap = lambda p: p.replace("/scratchc/", "/mnt/scratche/fast/") if p.startswith("/scratchc/") else p
C = pd.read_csv(M + "/set_C.csv", dtype=str); mp = pd.read_csv(K + "/uni2_npz_map.csv", dtype=str); mp = mp[mp.cnv.isin(C.Sample)].drop_duplicates(["cnv", "npz"])
def one(r):
    with np.load(r.npz, allow_pickle=True) as z: sp = remap(str(z["slide_path"])); lv = int(z["level"]); ts = int(z["tile_size"])
    s = openslide.OpenSlide(sp); prod = s.properties.get("hamamatsu.Product"); W, H = s.level_dimensions[lv]; s.close()
    md = os.path.join(os.path.dirname(os.path.dirname(sp)), "masks"); cand = sorted(glob.glob(os.path.join(md, glob.escape(os.path.splitext(os.path.basename(sp))[0]) + "*")))
    tt = np.nan
    if cand:
        mk = np.array(Image.open(cand[0]).convert("L")) > 0; mh, mw = mk.shape
        xi = np.clip(((np.arange(W // ts) * ts + ts / 2) * mw / W).round().astype(int), 0, mw - 1); yi = np.clip(((np.arange(H // ts) * ts + ts / 2) * mh / H).round().astype(int), 0, mh - 1)
        tt = int(mk[np.ix_(yi, xi)].sum())
    return {"Sample": r.cnv, "scanner": prod, "tissue_tiles": tt}
with ThreadPoolExecutor(8) as ex: R = pd.DataFrame(list(ex.map(one, [r for r in mp.itertuples()])))
G = R.groupby("Sample").agg(scanner=("scanner", lambda s: "/".join(sorted(set(map(str, s))))), tissue_tiles=("tissue_tiles", "sum"), n_slides=("scanner", "size")).reset_index()
G.to_csv(M + "/horizons/slide_desc.csv", index=False); print("HZ SLIDES DONE", len(G), G.scanner.value_counts().to_dict())

"""CNV-quiet morphology (docs/paper_cnvquiet_morphology.md), phase 1 (CPU): for every slide in a manifest, re-read the stored UNI2 tiles
(npz coords_level = tile CENTRES at pyramid level 2, 224 px; top-left = centre - 112, clamped, as
barretts_training/scripts/extract_foundation_from_masks.py extract_tiles) from the .ndpi at level 0 and resample them to 0.25 um/px (the HoVer-Net PanNuke training
resolution). Saves JPEG (quality 95) per tile plus a meta.json per slide. GPU jobs never read slides; they read these files.
usage: python cq_extract.py MANIFEST.csv SHARD NSHARDS   (manifest columns: slide_key, npz, Sample)
Row-level output (cluster only): killcoyne_mm/cnvquiet_morphology/tiles/<slide_key>/."""
import os, sys, json, numpy as np, pandas as pd, openslide
from PIL import Image
W = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis/feasibility/paper_plan/killcoyne_mm/cnvquiet_morphology"; TARGET_MPP = 0.25; TD = os.environ.get("CQ_TILEDIR", "tiles")
remap = lambda p: p.replace("/scratchc/", "/mnt/scratche/fast/") if p.startswith("/scratchc/") else p
man, shard, nsh = pd.read_csv(sys.argv[1], dtype=str), int(sys.argv[2]), int(sys.argv[3])
for r in man.iloc[shard::nsh].itertuples():
    out = f"{W}/{TD}/{r.slide_key}"; done = f"{out}/meta.json"
    if os.path.exists(done): print("skip", r.slide_key, flush=True); continue
    os.makedirs(out, exist_ok=True)
    with np.load(r.npz, allow_pickle=False) as z: sp = remap(str(z["slide_path"])); lv = int(z["level"]); ts = int(z["tile_size"]); co = np.asarray(z["coords_level"])
    s = openslide.OpenSlide(sp); lw, lh = s.level_dimensions[lv]; co = np.c_[np.clip(co[:, 0] - ts // 2, 0, lw - ts), np.clip(co[:, 1] - ts // 2, 0, lh - ts)]  # centres -> top-left
    ds = float(s.level_downsamples[lv]); mpp0 = float(s.properties["openslide.mpp-x"]); mppy = float(s.properties["openslide.mpp-y"])
    size0 = int(round(ts * ds)); outpx = int(round(size0 * mpp0 / TARGET_MPP)); corr = []
    for i, (x, y) in enumerate(co):
        im = s.read_region((int(round(x * ds)), int(round(y * ds))), 0, (size0, size0)).convert("RGB")
        if i < 4:  # coordinate check: the level-0 read, downsampled, must match the level-2 read the embedding used
            ref = np.asarray(s.read_region((int(round(x * ds)), int(round(y * ds))), lv, (ts, ts)).convert("L"), dtype=float)
            dn = np.asarray(im.convert("L").resize((ts, ts), Image.BILINEAR), dtype=float); corr.append(float(np.corrcoef(ref.ravel(), dn.ravel())[0, 1]))
        im.resize((outpx, outpx), Image.LANCZOS).save(f"{out}/t{i:03d}.jpg", quality=95)
    s.close()
    json.dump({"slide_key": r.slide_key, "Sample": r.Sample, "level": lv, "tile_size_level": ts, "downsample": ds, "mpp0_x": mpp0, "mpp0_y": mppy, "size0_px": size0,
               "out_px": outpx, "out_mpp": size0 * mpp0 / outpx, "n_tiles": int(len(co)), "coord_check_corr": corr, "coords": "centre"}, open(done, "w"), indent=1)
    print("done", r.slide_key, len(co), outpx, [round(c, 3) for c in corr], flush=True)
print("CQ EXTRACT DONE", shard, nsh)

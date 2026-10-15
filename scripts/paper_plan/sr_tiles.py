"""Scanner robustness, item 1, CPU phase (docs/paper_scanner_robustness.md @ abe96e9): read the 256 stored tiles of each slide exactly as
barretts_training/scripts/extract_foundation_from_masks.py extract_tiles (coords_level = tile centres at the stored level; top-left = centre - 112
clamped to the level bounds; read_region at that level, 224 x 224, RGB) and save them as uint8 arrays (256 x 224 x 224 x 3) on scratch.
Also checks that the stored Prov-GigaPath npz has the same coords. Only these CPU jobs open slide files. TASK = smoke | c<i>of<n>."""
import os, sys, json, time, numpy as np, pandas as pd, openslide
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sr_common import ROW, slide_map, chunk, remap, gp_of, sha_of
TASK = os.environ["TASK"]; os.makedirs(ROW + "/tiles", exist_ok=True); os.makedirs(ROW + "/tiles_check", exist_ok=True)
rows = []
for f in chunk(slide_map(), TASK):
    t0 = time.time(); out = f"{ROW}/tiles/{sha_of(f)}.npy"
    with np.load(f, allow_pickle=True) as z: c = np.asarray(z["coords_level"]); lv = int(z["level"]); ts = int(z["tile_size"]); sp = remap(str(z["slide_path"])); n_emb = int(z["embeddings"].shape[0])
    with np.load(gp_of(f), allow_pickle=True) as g: gp_eq = bool(np.array_equal(np.asarray(g["coords_level"]), c)); gp_n = int(g["embeddings"].shape[0])
    if not os.path.exists(out):
        s = openslide.OpenSlide(sp); W, H = s.level_dimensions[lv]; ds = float(s.level_downsamples[lv]); half = ts // 2; T_ = np.empty((len(c), ts, ts, 3), np.uint8)
        for i, (cx, cy) in enumerate(c):
            x0 = max(0, min(int(cx - half), W - ts)); y0 = max(0, min(int(cy - half), H - ts))
            T_[i] = np.asarray(s.read_region((int(round(x0 * ds)), int(round(y0 * ds))), lv, (ts, ts)).convert("RGB"))
        s.close(); tmp = out[:-4] + f".tmp{os.getpid()}.npy"; np.save(tmp, T_); os.replace(tmp, out)
    rows.append({"sha": sha_of(f), "n_coords": len(c), "n_emb_uni2": n_emb, "gp_coords_equal": gp_eq, "n_emb_gp": gp_n, "level": lv, "tile_size": ts, "sec": round(time.time() - t0, 1)})
    print("tiles", sha_of(f), rows[-1], flush=True)
pd.DataFrame(rows).to_csv(f"{ROW}/tiles_check/{TASK}.csv", index=False); print("SR TILES DONE", TASK, len(rows), flush=True)

"""E5 two-phase extraction, phase 1 (CPU only; the only E5 step that reads slide pixels). For the slides in killcoyne_mm/explore_E5/manifest_missing.txt
(shard SHARD_ID of N_SHARDS): the tiling of campaigns/allslides/extract_worker.py unchanged (level closest to 0.5 um/px, 224 px grid, thumbnail tissue test
mean < 220 and sd > 4, <= 8,000 tiles evenly thinned, tile kept if mean <= 235 and sd >= 8), so the features match SWGCohort/features_uni2h_05um.
Writes explore_E5/tiles/<stem>.npy (uint8 n x 224 x 224 x 3), <stem>.coords.npy and <stem>.json (attributes); phase 2 (ex_e5_embed.py, GPU) embeds them."""
import os, sys, json, numpy as np, openslide
E5 = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis/feasibility/paper_plan/killcoyne_mm/explore_E5"; OD = E5 + "/tiles"; os.makedirs(OD, exist_ok=True)
sh, ns = int(os.environ["SHARD_ID"]), int(os.environ["N_SHARDS"]); TILE, MAX_TILES = 224, 8000
paths = [l.strip() for l in open(E5 + "/manifest_missing.txt") if l.strip()]
def safe_grey_thumb(sl, lvl, target=2048, chunk_rows=32):   # extract_worker.py, unchanged
    from PIL import Image
    W, H = sl.level_dimensions[lvl]; sW, sH = sl.level_dimensions[-1]
    if sW * sH <= 50_000_000: return np.asarray(sl.get_thumbnail((target, target)).convert("L"), dtype=np.float32)
    ds = sl.level_downsamples[lvl]; sc = max(W, H) / float(target); tw, th = max(1, int(W / sc)), max(1, int(H / sc)); out = np.zeros((th, tw), dtype=np.float32)
    for oy in range(0, th, chunk_rows):
        oh = min(chunk_rows, th - oy); y0 = int(oy * H / th); y1 = int((oy + oh) * H / th)
        band = sl.read_region((0, int(y0 * ds)), lvl, (W, max(1, y1 - y0))).convert("L"); out[oy:oy + oh] = np.asarray(band.resize((tw, oh), Image.BILINEAR), dtype=np.float32)
    return out
for q, path in enumerate(paths):
    if q % ns != sh: continue
    stem = os.path.splitext(os.path.basename(path))[0]
    if os.path.exists(f"{OD}/{stem}.json"): continue
    sl = openslide.OpenSlide(path)
    try: mpp0 = float(sl.properties.get("openslide.mpp-x", "0.25"))
    except Exception: mpp0 = 0.25
    lvl = min(range(sl.level_count), key=lambda l: abs(mpp0 * sl.level_downsamples[l] - 0.5)); ds = sl.level_downsamples[lvl]; W, H = sl.level_dimensions[lvl]
    thumb = safe_grey_thumb(sl, lvl); ty, tx = thumb.shape; sy, sx = ty / H, tx / W; coords = []
    for yy in range(0, H - TILE, TILE):
        for xx in range(0, W - TILE, TILE):
            p = thumb[int(yy * sy):max(int(yy * sy) + 1, int((yy + TILE) * sy)), int(xx * sx):max(int(xx * sx) + 1, int((xx + TILE) * sx))]
            if p.size and p.mean() < 220 and p.std() > 4: coords.append((xx, yy))
    grid = (H // TILE) * (W // TILE); ntis = len(coords)
    if len(coords) > MAX_TILES: coords = [coords[i] for i in np.linspace(0, len(coords) - 1, MAX_TILES).astype(int)]
    tiles, kept = [], []
    for (xx, yy) in coords:
        t = np.asarray(sl.read_region((int(xx * ds), int(yy * ds)), lvl, (TILE, TILE)).convert("RGB"))
        if t.mean() > 235 or t.std() < 8: continue
        tiles.append(t); kept.append((xx, yy))
    A = np.stack(tiles).astype(np.uint8) if tiles else np.zeros((0, TILE, TILE, 3), np.uint8)
    np.save(f"{OD}/{stem}.npy.tmp.npy", A); os.replace(f"{OD}/{stem}.npy.tmp.npy", f"{OD}/{stem}.npy"); np.save(f"{OD}/{stem}.coords.npy", np.array(kept, dtype=np.int32).reshape(-1, 2))
    json.dump({"slide_path": path, "level": int(lvl), "mpp_used": float(mpp0 * ds), "grid_tiles": int(grid), "tissue_tiles": int(ntis), "kept_tiles": int(len(kept)), "level_dims": [int(W), int(H)]},
              open(f"{OD}/{stem}.json", "w"))
    print("tiles", q, len(kept), flush=True)
print("E5 TILES DONE", sh, flush=True)

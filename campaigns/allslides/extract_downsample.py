#!/usr/bin/env python3
"""Re-extract single-level slides at 0.5 um/px by reading 2x tiles and resizing.

Four ERIN slides have ONE pyramid level, so extract_worker.py had no 0.5 um/px
level to choose and stored features from 0.25 um/px tiles: physically 56 um
across instead of 112 um, i.e. not comparable to the other 11,788 slides. Here a
tile is read as TILE*ds px at level 0 and resized to 224, which reproduces the
corpus scale. Attributes record the resize so the difference stays auditable.

Env: SLIDES (comma-separated paths), OUTDIR.
"""
import os, sys, time, numpy as np, h5py, openslide, torch, timm
from PIL import Image

OUTDIR = os.environ["OUTDIR"]
# NB: Slurm --export=ALL,K=V splits on commas, so a comma-joined list arrives truncated
# to its first element. Pass SLIDELIST (one path per line) instead.
if os.environ.get("SLIDELIST"):
    SLIDES = [l.strip() for l in open(os.environ["SLIDELIST"]) if l.strip()]
else:
    SLIDES = [s for s in os.environ.get("SLIDES", "").split(",") if s.strip()]
JOB = os.environ.get("SLURM_JOB_ID", "nojob"); PART = os.environ.get("SLURM_JOB_PARTITION", "?")
TILE, BATCH, MAX_TILES, TARGET_MPP = 224, 128, 8000, 0.5
T0 = time.time()

dev = "cuda" if torch.cuda.is_available() else "cpu"
m = timm.create_model("vit_giant_patch14_224", pretrained=False, img_size=224, patch_size=14, depth=24,
                      num_heads=24, init_values=1e-5, embed_dim=1536, mlp_ratio=2.66667 * 2, num_classes=0,
                      no_embed_class=True, mlp_layer=timm.layers.SwiGLUPacked, act_layer=torch.nn.SiLU,
                      reg_tokens=8, dynamic_img_size=True)
m.load_state_dict(torch.load("/mnt/scratche/slow/fmlab/zuberi01/phd/UNI2-h/pytorch_model.bin", map_location="cpu"), strict=True)
m.eval().to(dev)
MEAN = np.array([0.485, 0.456, 0.406]); STD = np.array([0.229, 0.224, 0.225])
ENV = f"torch{torch.__version__}|timm{timm.__version__}|{torch.cuda.get_device_name(0) if dev=='cuda' else 'cpu'}"
print(f"model loaded on {dev} in {time.time()-T0:.0f}s ({ENV})", flush=True)


def grey_thumb(sl, lvl, target=2048, chunk_rows=32):
    W, H = sl.level_dimensions[lvl]; ds = sl.level_downsamples[lvl]
    sc = max(W, H) / float(target); tw, th = max(1, int(W / sc)), max(1, int(H / sc))
    out = np.zeros((th, tw), dtype=np.float32)
    for oy in range(0, th, chunk_rows):
        oh = min(chunk_rows, th - oy)
        y0 = int(oy * H / th); y1 = int((oy + oh) * H / th)
        band = sl.read_region((0, int(y0 * ds)), lvl, (W, max(1, y1 - y0))).convert("L")
        out[oy:oy + oh] = np.asarray(band.resize((tw, oh), Image.BILINEAR), dtype=np.float32)
    return out


def run(path):
    out = os.path.join(OUTDIR, os.path.splitext(os.path.basename(path))[0] + ".h5")
    sl = openslide.OpenSlide(path)
    try: mpp0 = float(sl.properties.get("openslide.mpp-x", "0.25"))
    except Exception: mpp0 = 0.25
    lvl = min(range(sl.level_count), key=lambda l: abs(mpp0 * sl.level_downsamples[l] - TARGET_MPP))
    ds_lvl = sl.level_downsamples[lvl]
    factor = max(1, int(round(TARGET_MPP / (mpp0 * ds_lvl))))   # extra downsample we must do ourselves
    read_px = TILE * factor
    W, H = sl.level_dimensions[lvl]
    thumb = grey_thumb(sl, lvl); ty, tx = thumb.shape; sy, sx = ty / H, tx / W
    coords = []
    for yy in range(0, H - read_px, read_px):
        for xx in range(0, W - read_px, read_px):
            p = thumb[int(yy * sy):max(int(yy * sy) + 1, int((yy + read_px) * sy)),
                      int(xx * sx):max(int(xx * sx) + 1, int((xx + read_px) * sx))]
            if p.size and p.mean() < 220 and p.std() > 4: coords.append((xx, yy))
    grid = (H // read_px) * (W // read_px)
    if len(coords) > MAX_TILES:
        coords = [coords[i] for i in np.linspace(0, len(coords) - 1, MAX_TILES).astype(int)]
    feats, batch, kept = [], [], []
    with torch.inference_mode():
        for (xx, yy) in coords:
            im = sl.read_region((int(xx * ds_lvl), int(yy * ds_lvl)), lvl, (read_px, read_px)).convert("RGB")
            if factor > 1: im = im.resize((TILE, TILE), Image.BILINEAR)
            t = np.asarray(im)
            if t.mean() > 235 or t.std() < 8: continue
            batch.append(((t / 255.0 - MEAN) / STD).transpose(2, 0, 1)); kept.append((xx, yy))
            if len(batch) == BATCH:
                feats.append(m(torch.tensor(np.stack(batch), dtype=torch.float32, device=dev)).cpu().numpy()); batch = []
        if batch:
            feats.append(m(torch.tensor(np.stack(batch), dtype=torch.float32, device=dev)).cpu().numpy())
    F = np.vstack(feats).astype(np.float32) if feats else np.zeros((0, 1536), np.float32)
    tmp = out + ".part"
    with h5py.File(tmp, "w") as h:
        h.create_dataset("features", data=F); h.create_dataset("coords", data=np.array(kept, dtype=np.int32))
        h.attrs["slide_path"] = path; h.attrs["level"] = lvl; h.attrs["mpp_used"] = mpp0 * ds_lvl * factor
        h.attrs["grid_tiles"] = grid; h.attrs["tissue_tiles"] = len(coords); h.attrs["kept_tiles"] = len(kept)
        h.attrs["level_dims"] = (W, H); h.attrs["env"] = ENV; h.attrs["job"] = f"{JOB}@{PART}"
        h.attrs["tile_px_at_level"] = read_px; h.attrs["resized_to"] = TILE; h.attrs["downsample_factor"] = factor
    os.replace(tmp, out)
    print(f"[done] {os.path.basename(path)} tiles={F.shape[0]}/{grid} factor={factor} "
          f"read_px={read_px} mpp={mpp0*ds_lvl*factor} in {time.time()-T0:.0f}s", flush=True)


for p in SLIDES:
    try: run(p)
    except Exception as e: print(f"[fail] {p}: {type(e).__name__}: {e}", flush=True)
print("[exit]", flush=True)

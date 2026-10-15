"""Pull-worker UNI2-h feature extraction (ERIN all-slides campaign).

One Slurm job = one worker. It loads the model once, then keeps claiming slides
from the manifest until the queue is drained or its walltime budget is nearly
spent. Per-slide unit is idempotent and lock-protected, so any number of workers
on any partition can run concurrently; late duplicates find nothing to do and
exit. Identical tiling/threshold logic to extract_one.py (features comparable).

Env: MANIFEST, OUTDIR, WALL_SECONDS (job walltime), MARGIN_SECONDS (default 600)
Lock: <out>.lock/ dir holding job_id; stale if owner not in squeue.
"""
import os, sys, time, random, signal, subprocess, socket
import numpy as np, h5py

MANIFEST = os.environ["MANIFEST"]; OUTDIR = os.environ["OUTDIR"]
WALL = int(os.environ.get("WALL_SECONDS", "10800")); MARGIN = int(os.environ.get("MARGIN_SECONDS", "600"))
JOB = os.environ.get("SLURM_JOB_ID", f"nojob-{os.getpid()}"); PART = os.environ.get("SLURM_JOB_PARTITION", "?")
T0 = time.time()
os.makedirs(OUTDIR, exist_ok=True)
paths = [l.strip() for l in open(MANIFEST) if l.strip()]
start = random.randrange(len(paths)) if paths else 0
order = paths[start:] + paths[:start]           # each worker walks the queue from a random offset
print(f"worker job={JOB} part={PART} host={socket.gethostname()} manifest={len(paths)} start={start}", flush=True)

def out_for(p): return os.path.join(OUTDIR, os.path.splitext(os.path.basename(p))[0] + ".h5")

def owner_alive(lock):
    try: jid = open(os.path.join(lock, "job_id")).read().strip()
    except OSError:
        return time.time() - os.path.getmtime(lock) < 1200   # unreadable: treat as alive for 20 min
    if not jid or jid.startswith("nojob"): return time.time() - os.path.getmtime(lock) < 1200
    r = subprocess.run(["squeue", "-h", "-j", jid], capture_output=True, text=True)
    return bool(r.stdout.strip())

def claim(p):
    out = out_for(p)
    if os.path.exists(out): return None
    lock = out + ".lock"
    try: os.mkdir(lock)
    except FileExistsError:
        if owner_alive(lock): return None
        try:                                              # stale lock: owner gone, reclaim
            for f in os.listdir(lock): os.remove(os.path.join(lock, f))
            os.rmdir(lock); os.mkdir(lock)
        except OSError: return None
    with open(os.path.join(lock, "job_id"), "w") as f: f.write(JOB)
    return lock

held = {"lock": None, "part": None}
def release():
    if held["part"] and os.path.exists(held["part"]):
        try: os.remove(held["part"])
        except OSError: pass
    if held["lock"] and os.path.isdir(held["lock"]):
        for f in os.listdir(held["lock"]):
            try: os.remove(os.path.join(held["lock"], f))
            except OSError: pass
        try: os.rmdir(held["lock"])
        except OSError: pass
    held["lock"] = held["part"] = None
def on_term(sig, frm):
    print(f"[signal {sig}] releasing lock and exiting", flush=True); release(); sys.exit(0)
signal.signal(signal.SIGTERM, on_term); signal.signal(signal.SIGINT, on_term)

import torch, timm, openslide
TILE, BATCH, MAX_TILES = 224, 128, 8000
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

def safe_grey_thumb(sl, lvl, target=2048, chunk_rows=32):
    """Greyscale thumbnail of level `lvl` built from banded reads.

    openslide.get_thumbnail() decodes the SMALLEST pyramid level whole; on slides
    whose smallest level is still gigapixel (generic-tiff without a real pyramid)
    that is tens of GB of RAM and the job is OOM-killed. These files are tiled,
    so banded read_region + per-band resize costs <1 GB. Same output semantics as
    np.asarray(get_thumbnail(...).convert("L")) for pyramidal slides.
    """
    from PIL import Image
    W, H = sl.level_dimensions[lvl]
    sW, sH = sl.level_dimensions[-1]
    if sW * sH <= 50_000_000:                       # normal pyramid: original path
        return np.asarray(sl.get_thumbnail((target, target)).convert("L"), dtype=np.float32)
    ds = sl.level_downsamples[lvl]
    sc = max(W, H) / float(target)
    tw, th = max(1, int(W / sc)), max(1, int(H / sc))
    out = np.zeros((th, tw), dtype=np.float32)
    for oy in range(0, th, chunk_rows):
        oh = min(chunk_rows, th - oy)
        y0 = int(oy * H / th); y1 = int((oy + oh) * H / th)
        band = sl.read_region((0, int(y0 * ds)), lvl, (W, max(1, y1 - y0))).convert("L")
        out[oy:oy + oh] = np.asarray(band.resize((tw, oh), Image.BILINEAR), dtype=np.float32)
    return out

def extract(path, out):
    sl = openslide.OpenSlide(path)
    try: mpp0 = float(sl.properties.get("openslide.mpp-x", "0.25"))
    except Exception: mpp0 = 0.25
    lvl = min(range(sl.level_count), key=lambda l: abs(mpp0 * sl.level_downsamples[l] - 0.5))
    ds = sl.level_downsamples[lvl]; W, H = sl.level_dimensions[lvl]
    thumb = safe_grey_thumb(sl, lvl)
    ty, tx = thumb.shape; sy, sx = ty / H, tx / W
    coords = []
    for yy in range(0, H - TILE, TILE):
        for xx in range(0, W - TILE, TILE):
            p = thumb[int(yy * sy):max(int(yy * sy) + 1, int((yy + TILE) * sy)),
                      int(xx * sx):max(int(xx * sx) + 1, int((xx + TILE) * sx))]
            if p.size and p.mean() < 220 and p.std() > 4: coords.append((xx, yy))
    grid = (H // TILE) * (W // TILE)
    if len(coords) > MAX_TILES:
        coords = [coords[i] for i in np.linspace(0, len(coords) - 1, MAX_TILES).astype(int)]
    feats, batch, kept = [], [], []
    with torch.inference_mode():
        for (xx, yy) in coords:
            t = np.asarray(sl.read_region((int(xx * ds), int(yy * ds)), lvl, (TILE, TILE)).convert("RGB"))
            if t.mean() > 235 or t.std() < 8: continue
            batch.append(((t / 255.0 - MEAN) / STD).transpose(2, 0, 1)); kept.append((xx, yy))
            if len(batch) == BATCH:
                feats.append(m(torch.tensor(np.stack(batch), dtype=torch.float32, device=dev)).cpu().numpy()); batch = []
        if batch:
            feats.append(m(torch.tensor(np.stack(batch), dtype=torch.float32, device=dev)).cpu().numpy())
    F = np.vstack(feats).astype(np.float32) if feats else np.zeros((0, 1536), np.float32)
    tmp = out + ".part"; held["part"] = tmp
    with h5py.File(tmp, "w") as h:
        h.create_dataset("features", data=F); h.create_dataset("coords", data=np.array(kept, dtype=np.int32))
        h.attrs["slide_path"] = path; h.attrs["level"] = lvl; h.attrs["mpp_used"] = mpp0 * ds
        h.attrs["grid_tiles"] = grid; h.attrs["tissue_tiles"] = len(coords); h.attrs["kept_tiles"] = len(kept)
        h.attrs["level_dims"] = (W, H); h.attrs["env"] = ENV; h.attrs["job"] = f"{JOB}@{PART}"
    os.replace(tmp, out); held["part"] = None
    return F.shape[0], grid

done = skipped = failed = 0
for p in order:
    if time.time() - T0 > WALL - MARGIN:
        print(f"[budget] {WALL-MARGIN}s reached, stopping", flush=True); break
    lock = claim(p)
    if lock is None: skipped += 1; continue
    held["lock"] = lock; t1 = time.time()
    try:
        n, grid = extract(p, out_for(p)); done += 1
        print(f"[done] {os.path.basename(p)} tiles={n}/{grid} in {time.time()-t1:.0f}s", flush=True)
    except Exception as e:
        failed += 1; print(f"[fail] {p}: {type(e).__name__}: {e}", flush=True)
        with open(os.path.join(OUTDIR, "_failed.log"), "a") as f: f.write(f"{p}\t{JOB}\t{type(e).__name__}: {e}\n")
    finally:
        release()
print(f"[exit] job={JOB} done={done} skipped={skipped} failed={failed} elapsed={time.time()-T0:.0f}s", flush=True)

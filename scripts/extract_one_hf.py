"""Encoder-sweep tile extractor (EXECUTION_PLAN 2.5): one slide per array element.

ENCODER env: virchow2 | gigapath (weights already in the shared HF cache; runs
offline). Mirrors extract_one.py contract: index = OFFSET + SLURM_ARRAY_TASK_ID,
--manifest FILE --outdir DIR [--name-from dir|file]; skips if output h5 exists.
"""
import argparse, os, sys, time
import numpy as np, h5py

os.environ.setdefault("HF_HOME", "/mnt/scratche/fast/fmlab/zuberi01/hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch, timm

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", required=True)
ap.add_argument("--outdir", required=True)
ap.add_argument("--name-from", default="dir", choices=["dir", "file"])
a = ap.parse_args()
ENC = os.environ.get("ENCODER", "virchow2")
IDX = int(os.environ.get("OFFSET", "0")) + int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
TILE, BATCH, MAX_TILES = 224, 64, 2000

paths = [l.strip() for l in open(a.manifest) if l.strip()]
if IDX >= len(paths):
    print(f"[skip] index {IDX} beyond manifest"); sys.exit(0)
path = paths[IDX]
base = os.path.basename(os.path.dirname(path)) if a.name_from == "dir" else \
       os.path.splitext(os.path.basename(path))[0]
out = os.path.join(a.outdir, base + ".h5")
os.makedirs(a.outdir, exist_ok=True)
if os.path.exists(out):
    print(f"[skip] exists {out}"); sys.exit(0)

if ENC == "virchow2":
    from timm.layers import SwiGLUPacked
    model = timm.create_model("hf-hub:paige-ai/Virchow2", pretrained=True,
                              mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)
    def embed(x):
        o = model(x)                       # (B, 261, 1280)
        return torch.cat([o[:, 0], o[:, 5:].mean(1)], dim=-1)  # (B, 2560)
elif ENC == "gigapath":
    model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
    def embed(x): return model(x)          # (B, 1536)
elif ENC == "hoptimus0":
    model = timm.create_model("hf-hub:bioptimus/H-optimus-0", pretrained=True,
                              init_values=1e-5, dynamic_img_size=False)
    def embed(x): return model(x)          # (B, 1536)
elif ENC == "hoptimus1":
    model = timm.create_model("hf-hub:bioptimus/H-optimus-1", pretrained=True,
                              init_values=1e-5, dynamic_img_size=False)
    def embed(x): return model(x)          # (B, 1536)
elif ENC == "phikon2":
    from transformers import AutoModel
    model = AutoModel.from_pretrained("owkin/phikon-v2")
    def embed(x): return model(pixel_values=x).last_hidden_state[:, 0]  # (B, 1024)
else:
    sys.exit(f"unknown ENCODER {ENC}")
dev = "cuda" if torch.cuda.is_available() else "cpu"
model.eval().to(dev)
if ENC == "phikon2":
    MEAN = np.array([0.485, 0.456, 0.406]); STD = np.array([0.229, 0.224, 0.225])
else:
    cfg = timm.data.resolve_data_config({}, model=model)
    MEAN = np.array(cfg["mean"]); STD = np.array(cfg["std"])

import openslide
sl = openslide.OpenSlide(path)
try: mpp0 = float(sl.properties.get("openslide.mpp-x", "0.25"))
except Exception: mpp0 = 0.25
lvl = min(range(sl.level_count), key=lambda l: abs(mpp0 * sl.level_downsamples[l] - 0.5))
W, H = sl.level_dimensions[lvl]
def is_tissue(t):
    g = t.mean(axis=2)
    return (g < 220).mean() > 0.4 and t.std() > 12

feats, batch, kept, t0 = [], [], 0, time.time()
for yy in range(0, H - TILE, TILE):
    for xx in range(0, W - TILE, TILE):
        loc = (int(xx * sl.level_downsamples[lvl]), int(yy * sl.level_downsamples[lvl]))
        t = np.asarray(sl.read_region(loc, lvl, (TILE, TILE)).convert("RGB"))
        if not is_tissue(t): continue
        batch.append(((t / 255.0 - MEAN) / STD).transpose(2, 0, 1)); kept += 1
        if len(batch) == BATCH:
            with torch.inference_mode():
                feats.append(embed(torch.tensor(np.stack(batch), dtype=torch.float32, device=dev)).cpu().numpy())
            batch = []
        if kept >= MAX_TILES: break
    if kept >= MAX_TILES: break
if batch:
    with torch.inference_mode():
        feats.append(embed(torch.tensor(np.stack(batch), dtype=torch.float32, device=dev)).cpu().numpy())
DIMS = {"virchow2": 2560, "gigapath": 1536, "hoptimus0": 1536, "hoptimus1": 1536, "phikon2": 1024}
F = np.vstack(feats) if feats else np.zeros((0, DIMS[ENC]), dtype=np.float32)
tmp = out + ".tmp"
with h5py.File(tmp, "w") as h:
    h.create_dataset("features", data=F.astype(np.float32))
os.replace(tmp, out)
print(f"[done] {base}: {F.shape} in {time.time()-t0:.0f}s ({ENC})")

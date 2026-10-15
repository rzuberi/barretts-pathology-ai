"""Ch3/Ch4 GATE: extract UNI2-h features for N ERIN slides end-to-end; time per slide
and estimate the 1,045-slide campaign. Self-contained (local weights, no HF download)."""
import os, sys, glob, json, time, numpy as np

OUT = os.environ.get("OUTDIR", ".")
N_SLIDES = int(os.environ.get("N_SLIDES", "6"))
TILE, BATCH, MAX_TILES = 224, 64, 512
ERIN = "/mnt/scratche/fast/fmlab/datasets/imaging/ERIN/slides"
WEIGHTS = "/mnt/scratche/slow/fmlab/zuberi01/phd/UNI2-h/pytorch_model.bin"

report = {"env": {}, "slides": [], "errors": []}
import torch
report["env"]["torch"] = torch.__version__
report["env"]["cuda"] = torch.cuda.is_available()
report["env"]["device_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
try:
    import openslide
    reader = "openslide"
except Exception as e:
    report["errors"].append(f"openslide import failed: {e}")
    import tifffile
    reader = "tifffile"
report["env"]["reader"] = reader
print(report["env"])

import timm
timm_kwargs = dict(img_size=224, patch_size=14, depth=24, num_heads=24, init_values=1e-5,
                   embed_dim=1536, mlp_ratio=2.66667 * 2, num_classes=0, no_embed_class=True,
                   mlp_layer=timm.layers.SwiGLUPacked, act_layer=torch.nn.SiLU,
                   reg_tokens=8, dynamic_img_size=True)
model = timm.create_model("vit_giant_patch14_224", pretrained=False, **timm_kwargs)
sd = torch.load(WEIGHTS, map_location="cpu")
missing, unexpected = model.load_state_dict(sd, strict=False)
report["env"]["missing_keys"] = len(missing); report["env"]["unexpected_keys"] = len(unexpected)
dev = "cuda" if torch.cuda.is_available() else "cpu"
model.eval().to(dev)
MEAN = np.array([0.485, 0.456, 0.406]); STD = np.array([0.229, 0.224, 0.225])
print("model loaded; missing", len(missing), "unexpected", len(unexpected))

def tiles_from_slide(path):
    """Yield RGB uint8 tiles at ~20x from largest usable level."""
    if reader == "openslide":
        sl = openslide.OpenSlide(path)
        # pick level closest to 0.5 mpp if metadata present, else level 0
        try:
            mpp0 = float(sl.properties.get("openslide.mpp-x", "0.25"))
        except Exception:
            mpp0 = 0.25
        best = min(range(sl.level_count),
                   key=lambda l: abs(mpp0 * sl.level_downsamples[l] - 0.5))
        W, H = sl.level_dimensions[best]
        step = TILE
        for yy in range(0, H - TILE, step):
            for xx in range(0, W - TILE, step):
                loc = (int(xx * sl.level_downsamples[best]), int(yy * sl.level_downsamples[best]))
                t = np.asarray(sl.read_region(loc, best, (TILE, TILE)).convert("RGB"))
                yield t
    else:
        import tifffile
        arr = tifffile.imread(path, level=min(2, len(tifffile.TiffFile(path).series[0].levels) - 1))
        if arr.ndim == 2: return
        H, W = arr.shape[:2]
        for yy in range(0, H - TILE, TILE):
            for xx in range(0, W - TILE, TILE):
                yield arr[yy:yy + TILE, xx:xx + TILE, :3]

def is_tissue(t):
    g = t.mean(axis=2)
    return (g < 220).mean() > 0.4 and t.std() > 12

cands = sorted(glob.glob(os.path.join(ERIN, "*", "*.tiff")) + glob.glob(os.path.join(ERIN, "*", "*.tif")))
by_dir = {}
for c in cands: by_dir.setdefault(os.path.dirname(c), c)
picks = list(by_dir.values())[:N_SLIDES]
print(f"found {len(by_dir)} slide dirs; testing {len(picks)}")
report["env"]["slide_dirs_found"] = len(by_dir)

for path in picks:
    t0 = time.time(); kept = 0; feats = []; batch = []
    try:
        for t in tiles_from_slide(path):
            if not is_tissue(t): continue
            batch.append(((t / 255.0 - MEAN) / STD).transpose(2, 0, 1))
            kept += 1
            if len(batch) == BATCH:
                with torch.inference_mode():
                    x = torch.tensor(np.stack(batch), dtype=torch.float32, device=dev)
                    feats.append(model(x).cpu().numpy())
                batch = []
            if kept >= MAX_TILES: break
        if batch:
            with torch.inference_mode():
                x = torch.tensor(np.stack(batch), dtype=torch.float32, device=dev)
                feats.append(model(x).cpu().numpy())
        dt = time.time() - t0
        F = np.vstack(feats) if feats else np.zeros((0, 1536))
        np.save(os.path.join(OUT, os.path.basename(os.path.dirname(path)) + "_uni2.npy"), F)
        report["slides"].append({"slide": path, "tiles": int(kept), "feat_shape": list(F.shape),
                                 "seconds": round(dt, 1)})
        print(report["slides"][-1])
    except Exception as e:
        report["errors"].append(f"{path}: {type(e).__name__}: {e}")
        print("ERROR", path, e)

ok = [s for s in report["slides"] if s["tiles"] > 0]
if ok:
    sec = np.mean([s["seconds"] for s in ok])
    # smoke caps at MAX_TILES; production slides likely have more tiles — scale estimate x4 as guardrail
    report["estimate"] = {"mean_sec_per_slide_capped": round(float(sec), 1),
                          "campaign_1045_slides_hours_capped": round(1045 * sec / 3600, 1),
                          "campaign_guardrail_x4_hours": round(4 * 1045 * sec / 3600, 1)}
    print(report["estimate"])
json.dump(report, open(os.path.join(OUT, "results.json"), "w"), indent=2)
print("wrote results.json;", len(ok), "slides OK,", len(report["errors"]), "errors")
if not ok: sys.exit(1)

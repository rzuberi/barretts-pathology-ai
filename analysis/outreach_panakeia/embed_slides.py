"""GPU unit: UNI2-h embeddings for a group of slides whose tile packs exist (tiles/{patient}.npy from prep_slide.py).
Model construction + local weights copied from campaigns/extract_one.py; bf16 autocast. Writes feats/{patient}.h5
(features fp16 (n,1536), coords level-0 top-left) and deletes nothing. Usage: embed_slides.py GROUP_FILE"""
import json, os, sys, time
import h5py, numpy as np, torch, timm

A = "/mnt/scratche/slow/fmlab/zuberi01/analysis/outreach_panakeia"; T = A + "/tiles"; O = A + "/feats"; os.makedirs(O, exist_ok=True)
pids = [l.strip() for l in open(sys.argv[1]) if l.strip()]
todo = [p for p in pids if not os.path.exists(f"{O}/{p}.h5")]
if not todo: print("done already"); sys.exit(0)
missing = [p for p in todo if not os.path.exists(f"{T}/{p}.json")]
if missing: sys.exit(f"tile packs missing for {missing}")
assert torch.cuda.is_available(), "no GPU"
dev = "cuda"; t0 = time.time()
m = timm.create_model("vit_giant_patch14_224", pretrained=False, img_size=224, patch_size=14, depth=24, num_heads=24,
                      init_values=1e-5, embed_dim=1536, mlp_ratio=2.66667 * 2, num_classes=0, no_embed_class=True,
                      mlp_layer=timm.layers.SwiGLUPacked, act_layer=torch.nn.SiLU, reg_tokens=8, dynamic_img_size=True)
m.load_state_dict(torch.load("/mnt/scratche/slow/fmlab/zuberi01/phd/UNI2-h/pytorch_model.bin", map_location="cpu"), strict=True)
m.eval().to(dev)
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
print(f"model loaded in {time.time() - t0:.0f}s on {torch.cuda.get_device_name()}", flush=True)
for p in todo:
    t1 = time.time(); X = np.load(f"{T}/{p}.npy"); meta = json.load(open(f"{T}/{p}.json")); assert len(X) == meta["n_tiles"]
    feats = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, len(X), 256):
            x = torch.from_numpy(X[i:i + 256]).to(dev).permute(0, 3, 1, 2).float().div_(255)
            feats.append(m((x - MEAN) / STD).float().cpu().numpy())
    F = np.concatenate(feats) if feats else np.zeros((0, 1536), np.float32)
    assert np.isfinite(F).all()
    with h5py.File(f"{O}/{p}.h5.part", "w") as h:
        h["features"] = F.astype(np.float16); h["coords"] = np.array(meta["coords_level0_topleft"], np.int32).reshape(-1, 2)
        for k in ("slide_path", "mpp0", "tile_px_level0"):
            if meta.get(k) is not None: h.attrs[k] = meta[k]
        if meta.get("excluded"): h.attrs["excluded"] = meta["excluded"]
        h.attrs["mpp_target"] = 0.5; h.attrs["model"] = "UNI2-h"
    os.replace(f"{O}/{p}.h5.part", f"{O}/{p}.h5")
    print(f"{p}: {F.shape} {time.time() - t1:.1f}s", flush=True)
print(f"group done in {time.time() - t0:.0f}s")

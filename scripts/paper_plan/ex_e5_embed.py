"""E5 two-phase extraction, phase 2 (GPU; reads only the uint8 tile arrays written by ex_e5_tiles.py, never a slide). UNI2-h exactly as
campaigns/allslides/extract_worker.py (timm ViT-giant config, ImageNet mean/sd, fp32, batch 128). Shard SHARD_ID of N_SHARDS over the manifest.
Writes explore_E5/features_uni2h_05um/<stem>.h5 (features, coords, attrs as the shared SWG files)."""
import os, json, time, numpy as np, h5py, torch, timm
E5 = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis/feasibility/paper_plan/killcoyne_mm/explore_E5"; TD = E5 + "/tiles"; OD = E5 + "/features_uni2h_05um"; os.makedirs(OD, exist_ok=True)
sh, ns = int(os.environ["SHARD_ID"]), int(os.environ["N_SHARDS"]); BATCH = 128; dev = "cuda"
paths = [l.strip() for l in open(E5 + "/manifest_missing.txt") if l.strip()]
m = timm.create_model("vit_giant_patch14_224", pretrained=False, img_size=224, patch_size=14, depth=24, num_heads=24, init_values=1e-5, embed_dim=1536, mlp_ratio=2.66667 * 2,
                      num_classes=0, no_embed_class=True, mlp_layer=timm.layers.SwiGLUPacked, act_layer=torch.nn.SiLU, reg_tokens=8, dynamic_img_size=True)
m.load_state_dict(torch.load("/mnt/scratche/slow/fmlab/zuberi01/phd/UNI2-h/pytorch_model.bin", map_location="cpu"), strict=True); m.eval().to(dev)
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
ENV = f"torch{torch.__version__}|timm{timm.__version__}|{torch.cuda.get_device_name(0)}"; JOB = f"{os.environ.get('SLURM_JOB_ID', '?')}@{os.environ.get('SLURM_JOB_PARTITION', '?')}"
def embed(stem):
    out = f"{OD}/{stem}.h5"; A = np.load(f"{TD}/{stem}.npy", mmap_mode="r"); C = np.load(f"{TD}/{stem}.coords.npy"); meta = json.load(open(f"{TD}/{stem}.json")); t0 = time.time(); F = []
    with torch.inference_mode():
        for i in range(0, len(A), BATCH):
            x = torch.from_numpy(np.ascontiguousarray(A[i:i + BATCH])).to(dev).permute(0, 3, 1, 2).float() / 255.0
            F.append(m((x - MEAN) / STD).cpu().numpy())
    F = np.vstack(F).astype(np.float32) if F else np.zeros((0, 1536), np.float32)
    with h5py.File(out + ".part", "w") as h:
        h.create_dataset("features", data=F); h.create_dataset("coords", data=C)
        for k, v in meta.items(): h.attrs[k] = v
        h.attrs["env"] = ENV; h.attrs["job"] = JOB
    os.replace(out + ".part", out); print("embedded", stem[:0], F.shape, f"{time.time() - t0:.0f}s", flush=True)
todo = [os.path.splitext(os.path.basename(p))[0] for q, p in enumerate(paths) if q % ns == sh]; T0 = time.time()
while True:   # phase-1 tiles may still be arriving: embed whatever is ready, wait for the rest (<= 2 h)
    left = [st for st in todo if not os.path.exists(f"{OD}/{st}.h5")]
    if not left: break
    ready = [st for st in left if os.path.exists(f"{TD}/{st}.json")]
    for st in ready: embed(st)
    if not ready:
        assert time.time() - T0 < 7200, f"{len(left)} tile sets never arrived"; time.sleep(30)
print("E5 EMBED DONE", sh, flush=True)

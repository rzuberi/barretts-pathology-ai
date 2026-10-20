"""GPU check: Virchow2 built from explicit config + safetensors reproduces the hf-hub build used in the smoke (cosine on 64 tiles)."""
import os, time, numpy as np, torch, timm
os.environ["HF_HUB_OFFLINE"] = "1"; os.environ.setdefault("HF_HOME", "/mnt/scratche/slow/fmlab/zuberi01/hf_cache")
from huggingface_hub import snapshot_download; from safetensors.torch import load_file
A = "/mnt/scratche/slow/fmlab/zuberi01/analysis/outreach_tilebio"; sl = "TCGA-18-3406-01Z-00-DX1.8D07F006-425C-4724-BBB3-5BA099401234"
t = time.time(); m = timm.create_model("vit_huge_patch14_224", pretrained=False, img_size=224, init_values=1e-5, num_classes=0, reg_tokens=4, mlp_ratio=5.3375,
    global_pool="", dynamic_img_size=True, mlp_layer=timm.layers.SwiGLUPacked, act_layer=torch.nn.SiLU)
m.load_state_dict(load_file(snapshot_download("paige-ai/Virchow2") + "/model.safetensors"), strict=True); m.eval().cuda(); print(f"load {time.time()-t:.1f}s")
X = np.load(f"{A}/data/tiles/tcga/{sl}.npy", mmap_mode="r")[:64].copy(); ref = np.load(f"{A}/data/feats/virchow2/tcga/{sl}.npy", mmap_mode="r")[:64].astype(np.float32)
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda(); STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
    x = torch.from_numpy(X).cuda().permute(0, 3, 1, 2).float().div(255).sub(MEAN).div(STD); o = m(x); E = torch.cat([o[:, 0], o[:, 5:].mean(1)], 1).float().cpu().numpy()
c = (E * ref).sum(1) / np.linalg.norm(E, axis=1) / np.linalg.norm(ref, axis=1); print("cosine min/median", c.min(), np.median(c))

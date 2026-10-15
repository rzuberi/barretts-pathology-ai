"""Scanner robustness, item 1, GPU phase (docs/paper_scanner_robustness.md @ abe96e9): embed the saved tiles (sr_tiles.py) with one extractor.
Never opens slide files. MODEL = virchow2 | hoptimus0 | gigapath (gigapath: smoke validation only). TASK = smoke | c<i>of<n>.
Model-card preprocessing, fp16 autocast, features stored as fp16 (as the stored UNI2-h / Prov-GigaPath embeddings).
  virchow2  : timm hf-hub:paige-ai/Virchow2 (SwiGLUPacked, SiLU), timm transform from pretrained_cfg; CLS + mean of patch tokens 5+ -> 2,560-d
  hoptimus0 : timm hf-hub:bioptimus/H-optimus-0 (init_values 1e-5, dynamic_img_size False); ToTensor + its mean/std; CLS -> 1,536-d
  gigapath  : timm hf_hub:prov-gigapath/prov-gigapath; Resize 256 bicubic, CenterCrop 224, ImageNet mean/std (original extraction transform); CLS"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sr_common import ROW, HF, slide_map, chunk, sha_of
os.environ["HF_HOME"] = HF; os.environ.setdefault("HF_HUB_OFFLINE", "1")   # before importing timm / huggingface_hub
import numpy as np, torch, timm
from PIL import Image
from torchvision import transforms
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
MODEL = os.environ["MODEL"]; TASK = os.environ["TASK"]; BS = 128; DEV = "cuda"; assert torch.cuda.is_available()
OD = f"{ROW}/emb/{MODEL}"; os.makedirs(OD, exist_ok=True)
if MODEL == "virchow2":
    model = timm.create_model("hf-hub:paige-ai/Virchow2", pretrained=True, mlp_layer=timm.layers.SwiGLUPacked, act_layer=torch.nn.SiLU)
    tf = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    feat = lambda o: torch.cat([o[:, 0], o[:, 5:].mean(1)], dim=-1)
elif MODEL == "hoptimus0":
    model = timm.create_model("hf-hub:bioptimus/H-optimus-0", pretrained=True, init_values=1e-5, dynamic_img_size=False)
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.707223, 0.578729, 0.703617), std=(0.211883, 0.230117, 0.177517))])
    feat = lambda o: o
elif MODEL == "gigapath":
    model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
    tf = transforms.Compose([transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC), transforms.CenterCrop(224), transforms.ToTensor(),
                             transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
    feat = lambda o: o
else: raise ValueError(MODEL)
model = model.eval().to(DEV); print("model", MODEL, type(model).__name__, torch.__version__, timm.__version__, torch.cuda.get_device_name(0), flush=True)
t_all = time.time(); n_tiles = 0
for f in chunk(slide_map(), TASK):
    out = f"{OD}/{sha_of(f)}.npz"
    if os.path.exists(out): continue
    t0 = time.time(); X = np.load(f"{ROW}/tiles/{sha_of(f)}.npy"); E = []
    with torch.inference_mode():
        for i in range(0, len(X), BS):
            b = torch.stack([tf(Image.fromarray(x)) for x in X[i:i + BS]]).to(DEV)
            with torch.autocast(device_type="cuda", dtype=torch.float16): o = model(b)
            E.append(feat(o).to(torch.float16).cpu().numpy())
    E = np.concatenate(E); assert E.ndim == 2 and len(E) == len(X) and np.isfinite(E.astype(np.float32)).all()
    tmp = out[:-4] + f".tmp{os.getpid()}.npz"; np.savez(tmp, embeddings=E); os.replace(tmp, out); n_tiles += len(E)
    print("emb", MODEL, sha_of(f), E.shape, round(time.time() - t0, 1), flush=True)
print("SR EMBED DONE", MODEL, TASK, n_tiles, "tiles", round(time.time() - t_all, 1), "s", flush=True)

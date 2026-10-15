"""Phase 2 (GPU): CONCH ViT-B/16 image embeddings + zero-shot probabilities for the tiles stored by conch_read_unit.py
(JPEG crops; no slide-file access). One pass covers both tile selections (masks is_subset / is_attn64 are carried through).
Text weights: zeroshot/text_weights.npy (existing prompt ensemble, computed once by make_text_weights.py). Idempotent per slide.
Env: UNIT, N_UNITS. Output: zeroshot/conch/<idx>.npz (tile_idx, is_subset, is_attn64, emb fp16 512-d, prob fp16 6-class, n_total)."""
import io, os, sys, time
import numpy as np, pandas as pd, torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
sys.path.insert(0, "/mnt/scratche/slow/fmlab/zuberi01/phd/CONCH")
from conch.open_clip_custom import create_model_from_pretrained
B = "/mnt/scratche/slow/fmlab/zuberi01"; ISBI = B + "/analysis/isbi_rerun"; Z = B + "/analysis/paperB_extras/zeroshot"; TI = Z + "/tiles"; OUT = Z + "/conch"; os.makedirs(OUT, exist_ok=True)
UNIT, NU = int(os.environ.get("UNIT", 0)), int(os.environ.get("N_UNITS", 1)); DEV = "cuda" if torch.cuda.is_available() else "cpu"; HALF = DEV == "cuda"
NCPU = int(os.environ.get("SLURM_CPUS_PER_TASK", 8)); torch.set_num_threads(NCPU)
s = pd.read_csv(ISBI + "/slides.csv", dtype={"idx": int}).fillna(""); excl = set(pd.read_csv(ISBI + "/qc/exclusions.csv").idx); s = s[~s.idx.isin(excl)].reset_index(drop=True)
lo, hi = (len(s) * UNIT) // NU, (len(s) * (UNIT + 1)) // NU
todo = [int(i) for i in s.iloc[lo:hi].idx if not os.path.exists(f"{OUT}/{i}.npz") and os.path.exists(f"{TI}/{i}.npz")]
missing = [int(i) for i in s.iloc[lo:hi].idx if not os.path.exists(f"{TI}/{i}.npz")]
print(f"unit {UNIT}/{NU}: slides {lo}-{hi}, {len(todo)} to score, {len(missing)} without tiles yet | {DEV}", flush=True)
if missing: print("tiles not ready for", len(missing), "slides -> not marking done"); sys.exit(2)
if not todo: sys.exit(0)
model, preprocess = create_model_from_pretrained("conch_ViT-B-16", checkpoint_path=B + "/phd/CONCH/checkpoints/conch/pytorch_model.bin"); model = model.to(DEV).eval(); model = model.half() if HALF else model
W = torch.tensor(np.load(f"{Z}/text_weights.npy"), device=DEV); scale = float(open(f"{Z}/logit_scale.txt").read())
class Crops(Dataset):
    def __init__(self, idxs): self.items = []; self.meta = {}
    def add(self, i, z):
        data, off = z["data"], z["offsets"]; start = len(self.items)
        self.items += [(data, off[k], off[k + 1]) for k in range(len(off) - 1)]; self.meta[i] = (start, len(self.items), z["tile_idx"], z["is_subset"], z["is_attn64"], int(z["n_total"]))
    def __len__(self): return len(self.items)
    def __getitem__(self, k): data, a, b = self.items[k]; return preprocess(Image.open(io.BytesIO(data[a:b].tobytes())).convert("RGB"))
ds = Crops(todo)
for i in todo:
    try: ds.add(i, np.load(f"{TI}/{i}.npz"))
    except Exception as e: print("ERR load", i, repr(e)[:150], flush=True)
print(f"{len(ds.meta)} slides, {len(ds)} tiles", flush=True); t0 = time.time(); n = 0; E, P = [], []
with torch.inference_mode():
    for b in DataLoader(ds, batch_size=256, num_workers=NCPU, pin_memory=HALF):
        f = model.encode_image(b.to(DEV).half() if HALF else b.to(DEV), proj_contrast=True, normalize=True).float()
        E.append(f.half().cpu().numpy()); P.append(torch.softmax(scale * f @ W.T, -1).half().cpu().numpy()); n += len(f)
        if n % 10240 < 256: print(f"{n} tiles, {n / (time.time() - t0):.0f} tiles/s", flush=True)
E = np.concatenate(E); P = np.concatenate(P)
for i, (a, b, tidx, su, at, nt) in ds.meta.items():
    tmp = f"{OUT}/{i}.part.npz"; np.savez(tmp, tile_idx=tidx, is_subset=su, is_attn64=at, emb=E[a:b], prob=P[a:b], n_total=nt); os.replace(tmp, f"{OUT}/{i}.npz")
print(f"unit {UNIT} done: {n} tiles in {(time.time() - t0) / 60:.1f} min ({n / max(time.time() - t0, 1):.0f} tiles/s)", flush=True)

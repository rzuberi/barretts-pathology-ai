"""Phase 1 (CPU, epyc): read the tiles CONCH needs ONCE per slide and store them as JPEG bytes, so the GPU phase never touches
the slide files (cold-opening a generic-TIFF on Lustre costs 10-40 s and per-tile reads ~80/s; that made the direct GPU plan >100 GPU-h).
Tiles per slide = union of (a) the ISBI fixed random <=1,200-tile subset (seed = slide idx, identical to the supervised bags) and
(b) the 64 highest-attention tiles of the section-trained MC-MIL (task_conch_zeroshot.py protocol). Idempotent per slide.
Env: UNIT, N_UNITS, NPROC (parallel readers, default SLURM_CPUS_PER_TASK). Output: zeroshot/tiles/<idx>.npz"""
import io, os, sys, time
from multiprocessing import Pool
import h5py, numpy as np, pandas as pd, torch, torch.nn as nn, openslide
from PIL import Image
B = "/mnt/scratche/slow/fmlab/zuberi01"; T = B + "/phd/thesis"; ISBI = B + "/analysis/isbi_rerun"; Z = B + "/analysis/paperB_extras/zeroshot"
OUT = Z + "/tiles"; os.makedirs(OUT, exist_ok=True); CAP, K64, Q = 1200, 64, 90
UNIT, NU = int(os.environ.get("UNIT", 0)), int(os.environ.get("N_UNITS", 1)); NPROC = int(os.environ.get("NPROC", os.environ.get("SLURM_CPUS_PER_TASK", 4)))
torch.set_num_threads(max(1, NPROC))
s = pd.read_csv(ISBI + "/slides.csv", dtype={"idx": int}).fillna(""); excl = set(pd.read_csv(ISBI + "/qc/exclusions.csv").idx); s = s[~s.idx.isin(excl)].reset_index(drop=True)
lo, hi = (len(s) * UNIT) // NU, (len(s) * (UNIT + 1)) // NU; todo = [r for r in s.iloc[lo:hi].itertuples() if not os.path.exists(f"{OUT}/{r.idx}.npz")]
print(f"unit {UNIT}/{NU}: slides {lo}-{hi}, {len(todo)} to do, {NPROC} readers", flush=True)
if not todo: sys.exit(0)
class MC_MIL(nn.Module):
    def __init__(self, d_in=1536, n_cls=6):
        super().__init__(); self.emb = nn.Sequential(nn.Linear(d_in, 512), nn.GELU(), nn.Dropout(0.1))
        self.att_v = nn.Linear(512, 128); self.att_u = nn.Linear(512, 128); self.att_w = nn.Linear(128, 1); self.head = nn.Linear(512, n_cls)
    def tiles(self, bag):
        h = self.emb(bag); a = self.att_w(torch.tanh(self.att_v(h)) * torch.sigmoid(self.att_u(h))).softmax(0); return h, a.squeeze(-1)
mil = MC_MIL(); mil.load_state_dict(torch.load(T + "/feasibility/runs/erin_tilemaps/output/mcmil_section.pt", map_location="cpu")); mil.eval()
_SL = {}
def read_chunk(args):
    path, lvl, ds, pts = args
    if path not in _SL: _SL.clear(); _SL[path] = openslide.OpenSlide(path)
    sl = _SL[path]; out = []
    for x, y in pts:
        im = sl.read_region((int(x * ds), int(y * ds)), lvl, (224, 224)).convert("RGB"); b = io.BytesIO(); im.save(b, "JPEG", quality=Q); out.append(b.getvalue())
    return out
t0 = time.time(); ntiles = 0; pool = Pool(NPROC)
for r in todo:
    try:
        t1 = time.time()
        with h5py.File(r.h5) as h: X = np.asarray(h["features"], np.float32); coords = np.asarray(h["coords"]); attrs = dict(h.attrs); n = len(coords)
        sub = np.sort(np.random.RandomState(int(r.idx)).choice(n, CAP, replace=False)) if n > CAP else np.arange(n)
        with torch.no_grad(): _, a = mil.tiles(torch.tensor(X)); att = np.sort(np.argsort(-a.numpy())[:K64])
        sel = np.union1d(sub, att); is_sub = np.isin(sel, sub); is_att = np.isin(sel, att)
        sl = openslide.OpenSlide(attrs["slide_path"]); lvl = int(attrs.get("level", 0)); ds = sl.level_downsamples[lvl]; sl.close(); t_open = time.time() - t1
        order = np.lexsort((coords[sel][:, 0], coords[sel][:, 1])); sel, is_sub, is_att = sel[order], is_sub[order], is_att[order]
        pts = [(int(x), int(y)) for x, y in coords[sel]]; chunks = [pts[i::NPROC] for i in range(NPROC)]; idx_chunks = [list(range(len(pts)))[i::NPROC] for i in range(NPROC)]
        res = pool.map(read_chunk, [(attrs["slide_path"], lvl, ds, c) for c in chunks if c])
        jpegs = [None] * len(pts)
        for ids, outs in zip([c for c in idx_chunks if c], res):
            for i, b in zip(ids, outs): jpegs[i] = b
        data = np.frombuffer(b"".join(jpegs), dtype=np.uint8); offsets = np.cumsum([0] + [len(b) for b in jpegs]).astype(np.int64)
        tmp = f"{OUT}/{r.idx}.part.npz"; np.savez(tmp, data=data, offsets=offsets, tile_idx=sel.astype(np.int32), is_subset=is_sub, is_attn64=is_att, n_total=n, level=lvl, ds=ds); os.replace(tmp, f"{OUT}/{r.idx}.npz")
        ntiles += len(pts); print(f"slide {r.idx}: {len(pts)} tiles, open {t_open:.1f}s, total {time.time() - t1:.1f}s, {len(data) / 1e6:.1f} MB", flush=True)
    except Exception as e: print("ERR", r.idx, repr(e)[:200], flush=True)
pool.close(); print(f"unit {UNIT} done: {ntiles} tiles in {(time.time() - t0) / 60:.1f} min ({ntiles / max(time.time() - t0, 1):.0f} tiles/s)", flush=True)

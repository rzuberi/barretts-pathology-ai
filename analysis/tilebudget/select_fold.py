"""P1 (GPU unit, one per fold F): shared selection resources for the tile-budget study.
(1) attention of the reference arm-B binary models (3 seeds, fold F) over ALL tiles -> attn_all_fold{F}.npy (fp16, pack_all global order)
(2) per-slide tile orderings (indices into the slide's h5 order), int32 (N, 1200) padded -1, in sel_fold{F}.npz:
    U1200 rules: rand_u_0/1/2, grid_u, attn_top_u, attn_bot_u, mixed_u, conch_u, diverse_u;  ALL rules: attn_top_all, grid_all.
Everything downstream (P2 units) reads these; nothing recomputes attention or selections. Idempotent (skips if outputs exist)."""
import json, os, sys, time
import numpy as np, pandas as pd, torch, torch.nn as nn
B = "/mnt/scratche/slow/fmlab/zuberi01"; ISBI = B + "/analysis/isbi_rerun"; A = B + "/analysis/tilebudget"; PA = A + "/pack_all"; Z = B + "/analysis/paperB_extras/zeroshot/conch"
F = int(os.environ["FOLD"]); CAP = 1200; dev = "cuda"
if os.path.exists(f"{A}/sel_fold{F}.npz") and os.path.exists(f"{A}/attn_all_fold{F}.npy"): print("done already"); sys.exit(0)
s = pd.read_csv(ISBI + "/slides.csv", dtype={"idx": int}).fillna(""); excl = set(pd.read_csv(ISBI + "/qc/exclusions.csv").idx); s = s[~s.idx.isin(excl)].reset_index(drop=True)
ix = json.load(open(PA + "/index_all.json")); chunks = ix["chunks"]; goff = {}; g = 0
for c in chunks: c["goff"] = g; g += c["n_tiles"]
TOT = g; print(f"fold {F}: {len(s)} slides, {TOT} tiles in pack_all", flush=True)
t0 = time.time(); X = np.empty((TOT, 1536), np.float16)
for c in chunks: a = np.load(f"{PA}/{c['file']}"); X[c["goff"]:c["goff"] + c["n_tiles"]] = a
print(f"pack_all in RAM: {X.nbytes / 1e9:.1f} GB in {time.time() - t0:.0f}s", flush=True)
def span(idx):
    e = ix["slides"][str(idx)]; c = chunks[e["chunk"]]; st = c["goff"] + e["off"]; return st, st + e["n"]
class ABMIL(nn.Module):
    def __init__(self, d_in=1536, d=384, d_attn=128, p_drop=0.25, n_out=1):
        super().__init__(); self.emb = nn.Sequential(nn.Linear(d_in, d), nn.ReLU(), nn.Dropout(p_drop))
        self.attn_v = nn.Sequential(nn.Linear(d, d_attn), nn.Tanh()); self.attn_u = nn.Sequential(nn.Linear(d, d_attn), nn.Sigmoid()); self.attn_w = nn.Linear(d_attn, 1); self.head = nn.Linear(d, n_out)
    def tile_scores(self, x): h = self.emb(x); return self.attn_w(self.attn_v(h) * self.attn_u(h)).squeeze(-1)
models = []
for sd in (0, 1, 2):
    m = ABMIL().to(dev).eval(); m.load_state_dict(torch.load(f"{ISBI}/models/patient_B_bin_seed{sd}_fold{F}.pt", map_location=dev)); models.append(m)
attn = np.zeros(TOT, np.float16)
if os.path.exists(f"{A}/attn_all_fold{F}.npy"): attn = np.load(f"{A}/attn_all_fold{F}.npy"); print("attention loaded (already computed)", flush=True)
else:
  with torch.inference_mode():
    for st in range(0, TOT, 65536):
        xb = torch.from_numpy(X[st:st + 65536]).to(dev).float(); attn[st:st + 65536] = torch.stack([m.tile_scores(xb) for m in models]).mean(0).half().cpu().numpy()
  np.save(f"{A}/attn_all_fold{F}.part.npy", attn); os.replace(f"{A}/attn_all_fold{F}.part.npy", f"{A}/attn_all_fold{F}.npy"); print("attention over all tiles saved", flush=True)
def fps_order(P, k):
    """farthest-point order (greedy) over rows of P (numpy float32, CPU: per-iteration cost is tiny, GPU kernel launches dominated)."""
    n = len(P); k = min(k, n)
    if n == 0: return np.zeros(0, np.int64)
    d = np.full(n, np.inf, np.float32); cur = int(np.argmin(((P - P.mean(0)) ** 2).sum(1))); out = [cur]
    for _ in range(k - 1):
        d = np.minimum(d, ((P - P[cur]) ** 2).sum(1)); d[out] = -1; cur = int(np.argmax(d)); out.append(cur)
    return np.array(out)
PROJ = (np.random.RandomState(32).randn(1536, 32) / np.sqrt(32)).astype(np.float32)
N = len(s); R = {k: np.full((N, CAP), -1, np.int32) for k in ["rand_u_0", "rand_u_1", "rand_u_2", "grid_u", "attn_top_u", "attn_bot_u", "mixed_u", "conch_u", "diverse_u", "attn_top_all", "grid_all"]}
n_all = np.zeros(N, np.int32); n_sub = np.zeros(N, np.int32); t1 = time.time()
for j, r in enumerate(s.itertuples()):
    idx = int(r.idx); st, en = span(idx); n = en - st; n_all[j] = n
    sub = np.sort(np.random.RandomState(idx).choice(n, CAP, replace=False)) if n > CAP else np.arange(n); k = len(sub); n_sub[j] = k   # == pack_bags.py
    coords = np.load(f"{PA}/coords_{idx}.npy").astype(np.float32); a = attn[st:en].astype(np.float32)
    for d in range(3): R[f"rand_u_{d}"][j, :k] = sub[np.random.RandomState(1000 * d + idx).permutation(k)]
    o = np.argsort(-a[sub], kind="stable"); R["attn_top_u"][j, :k] = sub[o]; R["attn_bot_u"][j, :k] = sub[o[::-1]]
    top, rnd = list(sub[o]), list(R["rand_u_0"][j, :k]); seen, mix = set(), []
    for t_, q in zip(top, rnd):
        for v in (t_, q):
            if v not in seen: seen.add(v); mix.append(v)
    R["mixed_u"][j, :k] = mix[:k]
    cz = f"{Z}/{idx}.npz"
    if os.path.exists(cz):
        z = np.load(cz); pl = {int(t_): float(p) for t_, p, m_ in zip(z["tile_idx"], z["prob"][:, 3:].astype(np.float32).sum(1), z["is_subset"]) if m_}
        R["conch_u"][j, :k] = sorted(sub, key=lambda t_: -pl.get(int(t_), -1.0))
    else: R["conch_u"][j, :k] = R["rand_u_0"][j, :k]
    R["grid_u"][j, :k] = sub[fps_order(coords[sub], k)]
    R["diverse_u"][j, :k] = sub[fps_order(X[st:en][sub].astype(np.float32) @ PROJ, k)]   # FPS in a fixed 32-d random projection (full 1536-d FPS cost 2.4 s/slide)
    oa = np.argsort(-a, kind="stable")[:CAP]; R["attn_top_all"][j, :len(oa)] = oa
    ga = fps_order(coords, CAP); R["grid_all"][j, :len(ga)] = ga
    if j % 1000 == 0: print(f"  {j}/{N} slides, {time.time() - t1:.0f}s", flush=True)
np.savez(f"{A}/sel_fold{F}.part.npz", idx=s.idx.values.astype(np.int32), n_all=n_all, n_sub=n_sub, **R); os.replace(f"{A}/sel_fold{F}.part.npz", f"{A}/sel_fold{F}.npz")
print(f"fold {F} selections saved in {(time.time() - t0) / 60:.1f} min", flush=True)

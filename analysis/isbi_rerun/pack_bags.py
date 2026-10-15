"""Part 2 step 1: pack all slides into fp16 chunk files so a training job can hold the whole
dataset on the GPU. Fixed random subset of <=CAP tiles per slide (seed 0 + slide index), identical
for every arm. Shardable: each shard writes its own disjoint rows. Index = slides.csv order.
Env: SHARD, N_SHARDS. Output: pack/chunk_<k>.npy (contiguous slide ranges) + meta_shard<k>.json; index.json by the finalizer."""
import json, os, sys
import numpy as np, pandas as pd, h5py
A = "/mnt/scratche/slow/fmlab/zuberi01/analysis/isbi_rerun"; P = A + "/pack"; os.makedirs(P, exist_ok=True)
CAP, D = 1200, 1536
s = pd.read_csv(A + "/slides.csv", dtype={"idx": int})
N = len(s); SHARD, NS = int(os.environ.get("SHARD", 0)), int(os.environ.get("N_SHARDS", 1))
# contiguous slide range per shard -> its own chunk file (no shared file, no Lustre lock contention)
lo, hi = (N * SHARD) // NS, (N * (SHARD + 1)) // NS
arr = np.zeros((hi - lo, CAP, D), dtype=np.float16)
meta = []
for i in range(lo, hi):
    r = s.iloc[i]
    try:
        with h5py.File(r.h5) as h: X = np.asarray(h["features"], dtype=np.float32)
    except Exception as e:
        meta.append({"idx": i, "n": 0, "err": repr(e)[:80]}); continue
    n = len(X)
    if n > CAP:
        sel = np.sort(np.random.RandomState(i).choice(n, CAP, replace=False)); X = X[sel]
    arr[i - lo, :len(X)] = X.astype(np.float16)
    meta.append({"idx": i, "n": int(len(X)), "n_total": int(n)})
tmp = f"{P}/chunk_{SHARD:02d}.part.npy"; np.save(tmp, arr); os.replace(tmp, f"{P}/chunk_{SHARD:02d}.npy")   # np.save appends .npy unless present
json.dump({"shard": SHARD, "lo": lo, "hi": hi, "rows": meta}, open(f"{P}/meta_shard{SHARD}.json", "w"))
print(f"shard {SHARD}/{NS}: packed {sum(1 for m in meta if m['n'] > 0)} slides, errors {sum(1 for m in meta if m['n'] == 0)}", flush=True)

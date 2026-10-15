"""P0 (CPU unit): fp16 pack of ALL tiles of every ISBI slide (slides.csv order), shard = contiguous slide range.
Writes tilebudget/pack_all/chunk_<k>.npy (concatenated tiles) + meta_<k>.json (per-slide offsets). Idempotent per shard.
Env: SHARD, N_SHARDS."""
import json, os, sys
import numpy as np, pandas as pd, h5py
B = "/mnt/scratche/slow/fmlab/zuberi01"; ISBI = B + "/analysis/isbi_rerun"; P = B + "/analysis/tilebudget/pack_all"; os.makedirs(P, exist_ok=True)
SHARD, NS = int(os.environ.get("SHARD", 0)), int(os.environ.get("N_SHARDS", 1))
if os.path.exists(f"{P}/chunk_{SHARD:03d}.npy") and os.path.exists(f"{P}/meta_{SHARD:03d}.json"): print("done already"); sys.exit(0)
s = pd.read_csv(ISBI + "/slides.csv", dtype={"idx": int}); N = len(s); lo, hi = (N * SHARD) // NS, (N * (SHARD + 1)) // NS
parts, meta, off = [], [], 0
for i in range(lo, hi):
    r = s.iloc[i]
    try:
        with h5py.File(r.h5) as h: X = np.asarray(h["features"], dtype=np.float16); C = np.asarray(h["coords"], dtype=np.int32)
    except Exception as e: meta.append({"idx": int(r.idx), "n": 0, "err": repr(e)[:80]}); continue
    parts.append(X); meta.append({"idx": int(r.idx), "n": int(len(X)), "off": off}); off += len(X)
    np.save(f"{P}/coords_{int(r.idx)}.npy", C)
arr = np.concatenate(parts) if parts else np.zeros((0, 1536), np.float16)
tmp = f"{P}/chunk_{SHARD:03d}.part.npy"; np.save(tmp, arr); os.replace(tmp, f"{P}/chunk_{SHARD:03d}.npy")
json.dump({"shard": SHARD, "lo": lo, "hi": hi, "n_tiles": int(off), "rows": meta}, open(f"{P}/meta_{SHARD:03d}.json", "w"))
print(f"shard {SHARD}/{NS}: slides {lo}-{hi}, tiles {off}, {arr.nbytes / 1e9:.2f} GB", flush=True)

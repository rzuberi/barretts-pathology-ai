"""Phase 3 (CPU, one task per ds x model): concatenate per-slide fp16 features into one pack, materialised once
(tile-budget lesson: per-slide npz/mmap reads over Lustre were the bottleneck). Writes data/packs/{model}_{ds}.npy + _index.json
(slide -> [start, end) rows; slides.csv order). Usage: pack_feats.py DS MODEL"""
import json, os, sys
import numpy as np, pandas as pd
A = "/mnt/scratche/slow/fmlab/zuberi01/analysis/outreach_tilebio"; ds, model = sys.argv[1:3]
SMOKE = os.environ.get("SMOKE") == "1"; P = f"{A}/data/packs" + ("_smoke" if SMOKE else ""); os.makedirs(P, exist_ok=True); out = f"{P}/{model}_{ds}.npy"; ix_js = f"{P}/{model}_{ds}_index.json"
if os.path.exists(out) and os.path.exists(ix_js): print("done already"); sys.exit(0)
s = pd.read_csv(A + "/manifests/slides.csv"); s = s[(s.ds == ds) & (s.use | SMOKE)]
if SMOKE and os.environ.get("SMOKE_SLIDES"): s = s[s.slide.isin(os.environ["SMOKE_SLIDES"].split())]
have = [r.slide for r in s.itertuples() if os.path.exists(f"{A}/data/feats/{model}/{ds}/{r.slide}.npy")]
miss = sorted(set(s.slide) - set(have)); print(f"{ds}/{model}: {len(have)} slides with features, {len(miss)} missing", flush=True)
shapes = {}
for sl in have:
    with open(f"{A}/data/feats/{model}/{ds}/{sl}.npy", "rb") as f:
        v = np.lib.format.read_magic(f); shapes[sl] = np.lib.format._read_array_header(f, v)[0]
D = {sh[1] for sh in shapes.values()}; assert len(D) == 1, D; D = D.pop(); N = sum(sh[0] for sh in shapes.values())
mm = np.lib.format.open_memmap(out + ".part.npy", mode="w+", dtype=np.float16, shape=(N, D)); idx = {}; o = 0
for sl in have:
    x = np.load(f"{A}/data/feats/{model}/{ds}/{sl}.npy"); mm[o:o + len(x)] = x; idx[sl] = [o, o + len(x)]; o += len(x)
mm.flush(); del mm; os.replace(out + ".part.npy", out)
json.dump({"n_tiles": int(N), "dim": int(D), "slides": idx, "missing": miss}, open(ix_js, "w")); print(f"packed {N} tiles x {D}", flush=True)

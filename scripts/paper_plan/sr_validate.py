"""Scanner robustness, smoke validation (docs/paper_scanner_robustness.md @ abe96e9, item 1): Prov-GigaPath re-embedded from the saved tiles vs the
stored Prov-GigaPath embeddings of the same tiles; per-tile cosine similarity must have median >= 0.99 and minimum >= 0.95. Also reports the
smoke embeddings' shapes for every extractor. Aggregates -> results/paper_final/scanner_robustness/smoke_validation.json."""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sr_common import ROW, OUT, slide_map, chunk, gp_of, sha_of
os.makedirs(OUT, exist_ok=True); res = {"prespec": "abe96e9", "slides": [], "shapes": {}}
cos_all = []
for f in chunk(slide_map(), "smoke"):
    with np.load(gp_of(f), allow_pickle=True) as g: S = np.asarray(g["embeddings"], np.float32)
    N = np.load(f"{ROW}/emb/gigapath/{sha_of(f)}.npz")["embeddings"].astype(np.float32)
    c = (S * N).sum(1) / (np.linalg.norm(S, axis=1) * np.linalg.norm(N, axis=1)); cos_all.append(c)
    res["slides"].append({"cos_median": round(float(np.median(c)), 5), "cos_min": round(float(c.min()), 5), "n": int(len(c))})
    for m in ("virchow2", "hoptimus0"): res["shapes"].setdefault(m, []).append(list(np.load(f"{ROW}/emb/{m}/{sha_of(f)}.npz")["embeddings"].shape))
c = np.concatenate(cos_all); res["cos_median"] = round(float(np.median(c)), 5); res["cos_min"] = round(float(c.min()), 5)
res["passed"] = bool(np.median(c) >= 0.99 and c.min() >= 0.95)
json.dump(res, open(f"{OUT}/smoke_validation.json", "w"), indent=1); print(json.dumps(res), flush=True); print("SR VALIDATE DONE", res["passed"], flush=True)

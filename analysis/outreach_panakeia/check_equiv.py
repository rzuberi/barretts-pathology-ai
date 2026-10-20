"""Smoke check: on a native-20x slide the new tiles sit on the old extract_one.py grid; matched tiles must embed alike."""
import h5py, numpy as np, json
A = "/mnt/scratche/slow/fmlab/zuberi01/analysis/outreach_panakeia"; p = "TCGA-HF-7134"
n = h5py.File(f"{A}/feats/{p}.h5"); sp = n.attrs["slide_path"]
o = h5py.File(sp.replace("/slides/", "/features/20x_224px/features_uni_v2/").replace(".svs", ".h5"))
from scipy.spatial import cKDTree
Fo = o["features"][:]; Fn = n["features"][:].astype(np.float32); dist, j = cKDTree(o["coords"][:]).query(n["coords"][:])
m = [(i, int(jj)) for i, (d, jj) in enumerate(zip(dist, j)) if d <= 16]   # grid steps differ slightly (mpp0 0.499 vs 0.5)
a = Fn[[i for i, _ in m]]; b = Fo[[j for _, j in m]]
cos = (a * b).sum(1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)
r = {"patient": p, "matched_tiles": len(m), "cos_median": float(np.median(cos)), "cos_p05": float(np.percentile(cos, 5))}
json.dump(r, open(f"{A}/results/smoke_equivalence.json", "w"), indent=1); print(r)

"""Killcoyne checks item 5 (reused for docs/paper_plan_killcoyne_cv.md item 4 with model name and spec arguments): add UNI2 extraction settings (read from the embedding files) and a SHA-256 manifest to the frozen model folder."""
import json, os, sys, hashlib, glob, numpy as np, pandas as pd
D = sys.argv[1]; M = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis/feasibility/paper_plan/killcoyne_mm"
mp = pd.read_csv("/mnt/scratche/slow/fmlab/zuberi01/phd/thesis/feasibility/paper_plan/killcoyne/uni2_npz_map.csv", dtype=str); C = set(pd.read_csv(M + "/set_C.csv", dtype=str).Sample)
vals = {}
for f in mp[mp.cnv.isin(C)].npz:
    with np.load(f, allow_pickle=True) as a:
        for k in ["level", "tile_size", "tiles_requested", "model_name", "embedding_dim"]:
            v = a[k]; v = v.item() if v.shape == () else v.tolist(); vals.setdefault(k, set()).add(json.dumps(v))
        vals.setdefault("mpp_x", set()).add(round(float(np.asarray(a["mpp_x"]).ravel()[0]), 3))
json.dump({"foundation_model": "UNI2", **{k: sorted(v) for k, v in vals.items()}, "tiles_per_slide": 256, "tile_aggregation": "mean over tiles", "note": "values observed across the set-C embedding files"}, open(D + "/extraction_settings.json", "w"), indent=1, default=str)
files = sorted(f for f in os.listdir(D) if f != "MANIFEST.json")
man = {"model": sys.argv[2] if len(sys.argv) > 2 else "killcoyne_frozen_v1", "spec": sys.argv[3] if len(sys.argv) > 3 else "docs/paper_plan_killcoyne_checks.md @ 828ebf7 item 5", "files": {f: hashlib.sha256(open(os.path.join(D, f), "rb").read()).hexdigest() for f in files}}
man["bundle_sha256"] = hashlib.sha256("".join(f"{k}:{v}\n" for k, v in sorted(man["files"].items())).encode()).hexdigest()
json.dump(man, open(D + "/MANIFEST.json", "w"), indent=1); print(json.dumps(man, indent=1))

"""Killcoyne protocol with imaging (docs/paper_plan_killcoyne_multimodal.md @ 3904c44), prep: common set C (published Killcoyne samples
with a UNI2 bag), per-sample tile bags (union of the sample's slides), slide-mean image matrix, neural bag index. Row-level outputs stay on
the cluster under feasibility/paper_plan/killcoyne_mm/."""
import os, json, numpy as np, pandas as pd
K = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis/feasibility/paper_plan/killcoyne"; M = K + "_mm"; os.makedirs(M + "/bags", exist_ok=True)
smp = pd.read_csv(K + "/kr_samples.csv", dtype=str); mp = pd.read_csv(K + "/uni2_npz_map.csv", dtype=str)
mp = mp[mp.cnv.isin(smp.Sample)].sort_values(["cnv", "npz"])
rows, img = [], {}
for s, g in mp.groupby("cnv"):
    bags = []
    for f in g.npz:
        with np.load(f, allow_pickle=False) as a: bags.append(np.asarray(a["embeddings"], dtype=np.float32))
    b = np.concatenate(bags, 0); assert b.ndim == 2 and b.shape[1] == 1536 and np.isfinite(b).all(), s
    out = f"{M}/bags/{s}.npz"; np.savez(out, embeddings=b); rows.append({"sample_id": s, "npz_path": out, "n_slides": len(bags), "n_tiles": b.shape[0]}); img[s] = b.mean(0)
idx = pd.DataFrame(rows); C = smp[smp.Sample.isin(idx.sample_id)].copy()
idx.to_csv(M + "/bag_index.csv", index=False); C.to_csv(M + "/set_C.csv", index=False)
X = pd.DataFrame(np.vstack([img[s] for s in C.Sample]), columns=[f"u{i}" for i in range(1536)]); X.insert(0, "Sample", C.Sample.values); X.to_csv(M + "/img_mean.csv", index=False)
rel = set(pd.read_csv(K + "/rel_rows.csv", dtype=str).Sample)
info = {"C_samples": int(len(C)), "C_patients": int(C.Patient.nunique()), "C_P_patients": int(C[C.Status == "P"].Patient.nunique()), "C_P_samples": int((C.Status == "P").sum()),
        "multi_slide_samples": int((idx.n_slides > 1).sum()), "tiles_per_sample": idx.n_tiles.describe().round(1).to_dict(), "release_rows_in_C": int(C.Sample.isin(rel).sum()),
        "excluded_no_image": int(len(smp) - len(C)), "excluded_patients": sorted(set(smp.Patient) - set(C.Patient)), "pathology_C": C.Pathology.value_counts().to_dict()}
json.dump(info, open(M + "/prep_info.json", "w"), indent=1); print(json.dumps(info, indent=1)); print("PREP DONE")

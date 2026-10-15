"""2.41 (Astra A6): patient-clustered uncertainty for the slide-level ERIN
contrasts. The 2.38 / 2.38b bootstraps resampled slides; slides nest in
patients (~1.4 per patient), so here we resample WHOLE PATIENTS and recompute
the paired deltas from the saved unit predictions. Reports iid vs clustered
CI widths (design-effect ratio) and whether any conclusion changes.
"""
import glob, json, os
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"
OUT = os.environ.get("OUTDIR", ".")
N_BOOT = 2000
POS = {"LGD", "HGD", "CANCER"}
CLASSES = ["NORMAL_OTHER", "NDBE", "IND", "LGD", "HGD", "CANCER"]
C_OF = {c: i for i, c in enumerate(CLASSES)}

lab = pd.read_csv(T + "/labeller/erin_slide_labels_v2.csv", dtype=str)
m = pd.read_csv(T + "/labeller/erin_master.csv", dtype=str).dropna(subset=["h5", "anon_id"]).drop_duplicates("h5")
lab = lab.merge(m[["h5", "anon_id"]], on="h5").set_index("h5")
pat_of = lab["anon_id"].to_dict()

def load_units(d, field):
    acc = {"case": {}, "slide": {}}
    for f in glob.glob(os.path.join(d, "*.npz")):
        arm = os.path.basename(f).split("_")[0]
        z = np.load(f, allow_pickle=True)
        for k, p in zip(z["keys"], z[field]):
            acc[arm].setdefault(str(k), []).append(p)
    return acc

def qwk(a, b, n=6):
    O = np.zeros((n, n))
    for i, j in zip(a, b): O[i, j] += 1
    W = np.array([[(i - j) ** 2 for j in range(n)] for i in range(n)], float) / (n - 1) ** 2
    E = np.outer(O.sum(1), O.sum(0)) / max(O.sum(), 1)
    return 1 - (W * O).sum() / max((W * E).sum(), 1e-9)

def macro_auc(yt, P):
    aucs = [roc_auc_score((yt == c).astype(int), P[:, c]) for c in range(P.shape[1])
            if 0 < (yt == c).sum() < len(yt)]
    return float(np.mean(aucs))

def both_cis(stat_fn, keys, pats):
    """stat_fn(idx) -> delta. Returns iid (slide) and clustered (patient) bootstrap CIs."""
    n = len(keys)
    rng = np.random.RandomState(0)
    iid = []
    for _ in range(N_BOOT):
        idx = rng.randint(0, n, n)
        v = stat_fn(idx)
        if v is not None: iid.append(v)
    upats = sorted(set(pats)); slides_of = {}
    for i, p in enumerate(pats): slides_of.setdefault(p, []).append(i)
    rng = np.random.RandomState(1)
    clu = []
    for _ in range(N_BOOT):
        sp = rng.randint(0, len(upats), len(upats))
        idx = np.concatenate([slides_of[upats[j]] for j in sp])
        v = stat_fn(idx)
        if v is not None: clu.append(v)
    def summ(v):
        lo, hi = np.percentile(v, [2.5, 97.5])
        return {"mean": round(float(np.mean(v)), 4), "ci": [round(float(lo), 4), round(float(hi), 4)],
                "width": round(float(hi - lo), 4)}
    a, b = summ(iid), summ(clu)
    return {"iid_slide_bootstrap": a, "patient_clustered_bootstrap": b,
            "width_ratio_clustered_over_iid": round(b["width"] / max(a["width"], 1e-9), 3),
            "n_slides": n, "n_patients": len(upats),
            "excludes_zero_iid": bool(a["ci"][0] > 0 or a["ci"][1] < 0),
            "excludes_zero_clustered": bool(b["ci"][0] > 0 or b["ci"][1] < 0)}

res = {"_meta": {"n_boot": N_BOOT}}

# ---- 2.38 binary ----
acc = load_units(T + "/feasibility/svc_units", "preds")
keys = sorted(set(acc["case"]) & set(acc["slide"]) & set(lab.index))
y = np.array([lab.loc[k, "worst_grade"] in POS for k in keys], int)
pc = np.array([np.mean(acc["case"][k]) for k in keys]); ps = np.array([np.mean(acc["slide"][k]) for k in keys])
pats = [pat_of[k] for k in keys]
def d_bin(idx):
    if len(set(y[idx])) < 2: return None
    return roc_auc_score(y[idx], ps[idx]) - roc_auc_score(y[idx], pc[idx])
res["binary_2_38_delta_slide_minus_case_auc"] = both_cis(d_bin, keys, pats)
print("binary", res["binary_2_38_delta_slide_minus_case_auc"], flush=True)

# ---- 2.38b six-class ----
acc5 = load_units(T + "/feasibility/svc5_units", "probs")
keys5 = sorted(set(acc5["case"]) & set(acc5["slide"]) & set(lab.index))
yt = np.array([C_OF[lab.loc[k, "worst_grade"]] for k in keys5])
Pc = np.stack([np.mean(acc5["case"][k], axis=0) for k in keys5])
Ps = np.stack([np.mean(acc5["slide"][k], axis=0) for k in keys5])
pats5 = [pat_of[k] for k in keys5]
def d_macro(idx):
    if len(set(yt[idx])) < 3: return None
    return macro_auc(yt[idx], Ps[idx]) - macro_auc(yt[idx], Pc[idx])
def d_qwk(idx):
    return qwk(yt[idx], Ps[idx].argmax(1)) - qwk(yt[idx], Pc[idx].argmax(1))
res["sixclass_2_38b_delta_macro_auc"] = both_cis(d_macro, keys5, pats5)
res["sixclass_2_38b_delta_qwk"] = both_cis(d_qwk, keys5, pats5)
print("6class", res["sixclass_2_38b_delta_macro_auc"], flush=True)

res["verdict"] = {
    k: ("conclusion unchanged" if v["excludes_zero_iid"] == v["excludes_zero_clustered"]
        else "CONCLUSION CHANGES under patient clustering")
    for k, v in res.items() if k != "_meta"}
json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=2)
print(json.dumps(res, indent=2))

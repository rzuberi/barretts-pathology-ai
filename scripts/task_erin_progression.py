"""ERIN pre-registered SECONDARY endpoint: future progression to HGD+.

Index slides = slides of each progression-cohort patient's index case (grade <=LGD,
>=1 later timepoint). Arms: histology ABMIL, prior-clinical logistic, late fusion.
Binary AUC primary here (progressed_to_HGDplus), patient-disjoint folds, 3 seeds.
"""
import json, os, sys
import h5py, numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from abmil_clf import train_abmil_clf_fold, bootstrap_auc, patient_folds
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"
OUT = os.environ.get("OUTDIR", ".")
SEEDS = [0, 1, 2]
NUM = {"NDBE": 0, "IND": 1, "LGD": 2, "HGD": 3, "CANCER": 4}

m = pd.read_csv(T + "/labeller/erin_master.csv", dtype=str).dropna(subset=["h5", "anon_id"]).drop_duplicates("h5")
m["date"] = pd.to_datetime(m["CollectedOrOrdered"], errors="coerce", dayfirst=True)
coh = pd.read_csv(T + "/labeller/erin_progression_cohort_v3.csv", dtype=str)
coh["index_date"] = pd.to_datetime(coh["index_date"], errors="coerce")
coh["y"] = (coh["progressed_to_HGDplus"] == "True").astype(int)

# index slides: patient's slides on (or before) the index date, closest first
rows = []
for _, r in coh.iterrows():
    s = m[(m["anon_id"] == r["anon_id"]) & (m["date"] <= r["index_date"])]
    if s.empty: continue
    s = s[s["date"] == s["date"].max()]
    for _, sl in s.iterrows():
        rows.append({"h5": sl["h5"], "anon_id": r["anon_id"], "y": r["y"],
                     "index_date": r["index_date"], "slide_date": sl["date"]})
idx = pd.DataFrame(rows).drop_duplicates("h5")
print(f"index slides={len(idx)} patients={idx['anon_id'].nunique()} progressors={idx.groupby('anon_id')['y'].max().sum()}", flush=True)

# prior-clinical features (strictly before index date)
labels_all = pd.read_csv(T + "/labeller/erin_labels_jury_final.csv", dtype=str)
labels_all["date"] = pd.to_datetime(labels_all["CollectedOrOrdered"], errors="coerce", dayfirst=True)
labels_all["num"] = labels_all["final_label"].map(NUM)
hist = labels_all.dropna(subset=["date"]).sort_values("date")
# 1.23 leakage ablation: three clinical variants.
#   full       — reports up to and including the index date (original arm)
#   strict_pre — landmarked: strictly before the index date (Terra-Pro design)
#   noprior    — grade history dropped entirely (count + duration only)
clin_variants = {"full": {}, "strict_pre": {}, "noprior": {}}
for _, r in idx.iterrows():
    upto = hist[(hist["anon_id"] == r["anon_id"]) & (hist["date"] <= r["index_date"])]
    before = upto[upto["date"] < r["index_date"]]
    for vname, pri in (("full", upto), ("strict_pre", before)):
        pn = pri["num"].dropna()
        clin_variants[vname][r["h5"]] = [float(pn.max()) if len(pn) else -1.0, float(len(pri)),
                                         float((r["index_date"] - pri["date"].min()).days) if len(pri) else -1.0]
    clin_variants["noprior"][r["h5"]] = clin_variants["full"][r["h5"]][1:]
clin = clin_variants["full"]

bags = {}
for _, r in idx.iterrows():
    with h5py.File(r["h5"]) as h:
        bags[r["h5"]] = np.asarray(h["features"])
y = dict(zip(idx["h5"], idx["y"].astype(int)))
pat = dict(zip(idx["h5"], idx["anon_id"]))
keys = sorted(bags)
folds = patient_folds(keys, pat, y, 5, seed=0)
Ck = {k: i for i, k in enumerate(keys)}

def make_clin_fold(vname):
    Cv = np.array([clin_variants[vname][k] for k in keys])
    def clin_fold(tr, te, seed):
        sc = StandardScaler().fit(Cv[[Ck[k] for k in tr]])
        lr = LogisticRegression(max_iter=2000, class_weight="balanced")
        lr.fit(sc.transform(Cv[[Ck[k] for k in tr]]), [y[k] for k in tr])
        return dict(zip(te, map(float, lr.predict_proba(sc.transform(Cv[[Ck[k] for k in te]]))[:, 1])))
    return clin_fold

arms = {"hist_abmil": lambda tr, te, s: train_abmil_clf_fold(bags, tr, te, y, s),
        "clin_only": make_clin_fold("full"),
        "clin_strict_pre": make_clin_fold("strict_pre"),
        "clin_noprior": make_clin_fold("noprior")}
oof = {}
for name, fn in arms.items():
    per = []
    for s in SEEDS:
        o = {}
        for i in range(5):
            te = folds[i]; tr = [k for j, f in enumerate(folds) if j != i for k in f]
            o.update(fn(tr, te, s))
        per.append(o); print(name, "seed", s, "done", flush=True)
    oof[name] = {k: float(np.mean([p[k] for p in per])) for k in keys}
oof["late_fusion"] = {k: (oof["hist_abmil"][k] + oof["clin_only"][k]) / 2 for k in keys}
oof["late_fusion_strict_pre"] = {k: (oof["hist_abmil"][k] + oof["clin_strict_pre"][k]) / 2 for k in keys}
oof["late_fusion_noprior"] = {k: (oof["hist_abmil"][k] + oof["clin_noprior"][k]) / 2 for k in keys}

res = {"_meta": {"n_slides": len(keys), "n_patients": int(idx["anon_id"].nunique()),
                 "progressor_patients": int(idx.groupby("anon_id")["y"].max().sum()),
                 "endpoint": "progressed_to_HGDplus", "seeds": SEEDS}}
for name, o in oof.items():
    ref = oof["hist_abmil"] if name.startswith("late_fusion") else None
    res[name] = bootstrap_auc(o, y, prob_b=ref)
    print(name, res[name], flush=True)
json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=2)
print("wrote results.json")

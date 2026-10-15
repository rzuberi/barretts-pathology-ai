"""EXECUTION_PLAN 2.2: the ERIN Ch3 arm, per docs/erin_ch3_preregistration.md.

Primary endpoint: current-grade classification, NDBE vs {LGD,HGD,CANCER} (IND
excluded from primary). Labels: jury train-eligible/adjudicated only (unsure held
out per Rehan's rule). Arms: histology ABMIL; clinical-only (PRIOR-timepoint
variables only, to avoid label circularity); late fusion; early-ish fusion
(clinical appended to pooled features, logistic). Patient-disjoint 5-fold, 3 seeds.
"""
import json, os, sys
import h5py, numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from abmil_clf import train_abmil_clf_fold, bootstrap_auc, patient_folds
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"
MASTER = T + "/labeller/erin_master.csv"
OUT = os.environ.get("OUTDIR", ".")
SEEDS = [0, 1, 2]
NUM = {"NDBE": 0, "IND": 1, "LGD": 2, "HGD": 3, "CANCER": 4}

m = pd.read_csv(MASTER, dtype=str)
FEAT_SUB = os.environ.get("FEAT_SUB", "")  # e.g. features_virchow2 for the encoder sweep
if FEAT_SUB:
    m["h5"] = m["h5"].str.replace("features_uni_v2", FEAT_SUB, regex=False)
    m = m[m["h5"].map(os.path.exists)]
m = m[m["label_status"].isin(["train_eligible", "adjudicated"])]
m = m[m["final_label"].isin(["NDBE", "LGD", "HGD", "CANCER"])]  # IND excluded (pre-reg)
m["date"] = pd.to_datetime(m["CollectedOrOrdered"], errors="coerce", dayfirst=True)
m = m.dropna(subset=["h5", "anon_id", "date"]).drop_duplicates("h5")
m["y"] = (m["final_label"] != "NDBE").astype(int)

# prior-timepoint clinical features per slide (no current-label leakage)
labels_all = pd.read_csv(T + "/labeller/erin_labels_jury_final.csv", dtype=str)
labels_all["date"] = pd.to_datetime(labels_all["CollectedOrOrdered"], errors="coerce", dayfirst=True)
labels_all["num"] = labels_all["final_label"].map(NUM)
hist = labels_all.dropna(subset=["date"]).sort_values("date")
clin_rows = {}
for _, r in m.iterrows():
    prior = hist[(hist["anon_id"] == r["anon_id"]) & (hist["date"] < r["date"])]
    pnum = prior["num"].dropna()
    clin_rows[r["h5"]] = [
        float(pnum.max()) if len(pnum) else -1.0,
        float(pnum.iloc[-1]) if len(pnum) else -1.0,
        float(len(prior)),
        float((r["date"] - prior["date"].max()).days) if len(prior) else -1.0,
    ]

bags = {}
for _, r in m.iterrows():
    with h5py.File(r["h5"]) as h:
        bags[r["h5"]] = np.asarray(h["features"])
y = dict(zip(m["h5"], m["y"]))
if os.environ.get("SHUFFLE"):  # pre-registered control: patient-wise label permutation
    import numpy as _np
    rng = _np.random.RandomState(0)
    pats = m.groupby("anon_id")["y"].max()
    perm = pd.Series(rng.permutation(pats.values), index=pats.index)
    y = {r["h5"]: int(perm[r["anon_id"]]) for _, r in m.iterrows()}
pat = dict(zip(m["h5"], m["anon_id"]))
keys = sorted(bags)
print(f"slides={len(keys)} patients={m['anon_id'].nunique()} pos={sum(y.values())}", flush=True)

folds = patient_folds(keys, pat, y, 5, seed=0)
C = np.array([clin_rows[k] for k in keys]); Ck = {k: i for i, k in enumerate(keys)}

def clin_fold(tr, te, seed):
    sc = StandardScaler().fit(C[[Ck[k] for k in tr]])
    lr = LogisticRegression(max_iter=2000, class_weight="balanced")
    lr.fit(sc.transform(C[[Ck[k] for k in tr]]), [y[k] for k in tr])
    pr = lr.predict_proba(sc.transform(C[[Ck[k] for k in te]]))[:, 1]
    return dict(zip(te, map(float, pr)))

def early_fold(tr, te, seed):
    P = {k: np.concatenate([bags[k].mean(0), clin_rows[k]]) for k in tr + te}
    X = np.stack([P[k] for k in tr]); sc = StandardScaler().fit(X)
    lr = LogisticRegression(max_iter=3000, class_weight="balanced", C=0.5)
    lr.fit(sc.transform(X), [y[k] for k in tr])
    pr = lr.predict_proba(sc.transform(np.stack([P[k] for k in te])))[:, 1]
    return dict(zip(te, map(float, pr)))

arms = {"hist_abmil": lambda tr, te, s: train_abmil_clf_fold(bags, tr, te, y, s),
        "clin_only": clin_fold, "early_fusion": early_fold}
oof = {}
for name, fn in arms.items():
    per = []
    for s in SEEDS:
        o = {}
        for i in range(5):
            te = folds[i]; tr = [k for j, f in enumerate(folds) if j != i for k in f]
            o.update(fn(tr, te, s))
        per.append(o)
        print(f"{name} seed{s} done", flush=True)
    oof[name] = {k: float(np.mean([p[k] for p in per])) for k in keys}
oof["late_fusion"] = {k: (oof["hist_abmil"][k] + oof["clin_only"][k]) / 2 for k in keys}

res = {"_meta": {"n_slides": len(keys), "n_patients": int(m["anon_id"].nunique()),
                 "pos": int(sum(y.values())), "endpoint": "NDBE vs LGD+", "seeds": SEEDS}}
for name, o in oof.items():
    ref = oof["hist_abmil"] if "fusion" in name else None
    res[name] = bootstrap_auc(o, y, prob_b=ref)
    print(name, res[name], flush=True)
json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=2)
print("wrote results.json")

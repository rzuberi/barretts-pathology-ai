"""ERIN necessity probes at full scale (n≈2,280 slides — our largest cohort,
never probed): can histology alone predict (a) current grade [floor for Ch3's
hist arm], (b) prior max grade, (c) patient age, (d) future progression?
Mean-pools each slide's tiles (cached), then the standard linear-probe protocol
with PATIENT-GROUPED folds to avoid same-patient leakage."""
import json, os
import numpy as np, pandas as pd, h5py
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"
OUT = os.environ.get("OUTDIR", ".")
CACHE = os.path.join(OUT, "erin_pooled_uni2.npz")
NUM = {"NDBE": 0, "IND": 1, "LGD": 2, "HGD": 3, "CANCER": 4}

m = pd.read_csv(T + "/labeller/erin_master.csv", dtype=str).dropna(subset=["h5"]).drop_duplicates("h5")
if os.path.exists(CACHE):
    d = np.load(CACHE, allow_pickle=True); X, h5s = d["X"], list(d["h5s"])
else:
    X, h5s = [], []
    for i, p in enumerate(m["h5"]):
        try:
            with h5py.File(p) as h:
                X.append(np.asarray(h["features"]).mean(0)); h5s.append(p)
        except Exception as e:
            print("skip", p, e)
        if i % 200 == 0: print(f"pooled {i}/{len(m)}", flush=True)
    X = np.vstack(X); np.savez(CACHE, X=X, h5s=np.array(h5s))
m = m.set_index("h5").loc[h5s].reset_index()
print("pooled slides:", len(m), flush=True)

lab_all = pd.read_csv(T + "/labeller/erin_labels_jury_final.csv", dtype=str)
lab_all["date"] = pd.to_datetime(lab_all["CollectedOrOrdered"], errors="coerce", dayfirst=True)
lab_all["num"] = lab_all["final_label"].map(NUM)
rep_meta = pd.read_csv("/mnt/scratche/fast/fmlab/datasets/imaging/ERIN/data/PathologyReport_AnonIds.csv",
                       dtype=str, low_memory=False)[["CaseName", "AgeAtInvestigation"]].drop_duplicates("CaseName")
m = m.merge(rep_meta, on="CaseName", how="left")
assert len(m) == len(X), f"row/feature mismatch after merge: {len(m)} vs {len(X)}"
prog = pd.read_csv(T + "/labeller/erin_progression_cohort_v3.csv", dtype=str)
prog_map = dict(zip(prog["anon_id"], prog["progressed_to_HGDplus"] == "True"))

elig = m["label_status"].isin(["train_eligible", "adjudicated"])
grade_num = m["final_label"].map(NUM)
hist_sorted = lab_all.dropna(subset=["date"]).sort_values("date")
m["date"] = pd.to_datetime(m["CollectedOrOrdered"], errors="coerce", dayfirst=True)
prior_max = []
for _, r in m.iterrows():
    pri = hist_sorted[(hist_sorted["anon_id"] == r["anon_id"]) & (hist_sorted["date"] < r["date"])]["num"].dropna()
    prior_max.append(pri.max() if len(pri) else np.nan)
m["prior_max"] = prior_max
age = pd.to_numeric(m["AgeAtInvestigation"], errors="coerce")

targets = {
    "grade_from_hist": np.where(elig & grade_num.notna(), (grade_num >= 2).astype(float), np.nan),
    "prior_dysplasia_from_hist": np.where(pd.Series(m["prior_max"]).notna(),
                                          (pd.to_numeric(m["prior_max"]) >= 2).astype(float), np.nan),
    "age_from_hist": np.where(age.notna(), (age > age.median()).astype(float), np.nan),
    "progression_from_hist": np.array([float(prog_map[a]) if a in prog_map else np.nan
                                       for a in m["anon_id"]]),
}
groups_all = m["anon_id"].values

def probe(y):
    mask = ~np.isnan(y) & pd.Series(groups_all).notna().values
    yy, Xi, gg = y[mask].astype(int), X[mask], groups_all[mask]
    if len(set(yy)) < 2 or min(np.bincount(yy)) < 20: return {"skipped": True}
    aucs = []
    for s in range(3):
        rng = np.random.RandomState(s)
        order = rng.permutation(len(yy))
        yy2, Xi2, gg2 = yy[order], Xi[order], gg[order]
        oof = np.zeros(len(yy2))
        for tr, te in GroupKFold(5).split(Xi2, yy2, gg2):
            p = Pipeline([("s", StandardScaler()), ("p", PCA(64)),
                          ("l", LogisticRegression(C=0.5, class_weight="balanced", max_iter=4000))])
            p.fit(Xi2[tr], yy2[tr]); oof[te] = p.predict_proba(Xi2[te])[:, 1]
        aucs.append(roc_auc_score(yy2, oof))
    return {"n": int(mask.sum()), "pos": int(yy.sum()),
            "auc": round(float(np.mean(aucs)), 3), "auc_sd": round(float(np.std(aucs)), 3)}

res = {name: probe(y) for name, y in targets.items()}
for k, v in res.items(): print(k, v, flush=True)
json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=2)
print("wrote results.json")

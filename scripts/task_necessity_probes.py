"""EXECUTION_PLAN 2.17 (new, joint 2026-08-21): necessity-triangle probes on the
TCGA OAC+GEJ pool — can histology alone predict the OTHER modalities' variables?
Linear probes on mean-pooled UNI2 features: tp53, wgd (replication at n≈446),
stage (I/II vs III/IV), age (> median). Protocol as prior probes: PCA64+logreg,
5-fold x 5 seeds, shuffled controls.
"""
import glob, json, os, re
import numpy as np, pandas as pd, h5py
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"
FEATS = {"OAC": "/mnt/scratche/fast/fmlab/datasets/imaging/tcga_esca/features/20x_224px/features_uni_v2",
         "GEJ": "/mnt/scratche/fast/fmlab/datasets/imaging/tcga_stad/features/20x_224px/features_uni_v2"}
LABELS = {"OAC": T + "/data/tcga_oac_labels.csv", "GEJ": T + "/data/tcga_stad_labels.csv"}
OUT = os.environ.get("OUTDIR", ".")

X, rows = [], []
for grp, fd in FEATS.items():
    lab = pd.read_csv(LABELS[grp]).drop_duplicates("barcode").set_index("barcode")
    for f in sorted(glob.glob(os.path.join(fd, "*.h5"))):
        m = re.search(r"(TCGA-\w{2}-\w{4})", os.path.basename(f))
        if not m or m.group(1) not in lab.index: continue
        with h5py.File(f) as h:
            X.append(np.asarray(h["features"]).mean(0))
        r = lab.loc[m.group(1)]
        stage = re.search(r"Stage (I{1,3}V?)", str(r["ajcc_stage"]))
        rows.append({"tp53": r.get("tp53_mut"), "wgd": r.get("wgd"),
                     "stage_hi": {"I": 0, "II": 0, "III": 1, "IV": 1}.get(stage.group(1) if stage else None),
                     "age": r.get("age")})
X = np.vstack(X); md = pd.DataFrame(rows)
md["age_hi"] = (pd.to_numeric(md["age"]) > pd.to_numeric(md["age"]).median()).astype(float)
print("pool n:", len(md))

def probe(name, yser):
    y = pd.to_numeric(yser, errors="coerce")
    mask = y.notna().values
    yy, Xi = y[mask].astype(int).values, X[mask]
    if len(set(yy)) < 2 or min(np.bincount(yy)) < 10: return {"skipped": True}
    def run(labels, seed):
        oof = np.zeros(len(labels))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(Xi, labels):
            p = Pipeline([("s", StandardScaler()), ("p", PCA(min(64, len(tr) - 1))),
                          ("l", LogisticRegression(C=0.5, class_weight="balanced", max_iter=4000))])
            p.fit(Xi[tr], labels[tr]); oof[te] = p.predict_proba(Xi[te])[:, 1]
        return roc_auc_score(labels, oof)
    aucs = [run(yy, s) for s in range(5)]
    shuf = [run(np.random.RandomState(s).permutation(yy), s) for s in range(5)]
    r = {"n": int(mask.sum()), "pos": int(yy.sum()), "auc": round(float(np.mean(aucs)), 3),
         "auc_sd": round(float(np.std(aucs)), 3), "shuffled": round(float(np.mean(shuf)), 3)}
    print(name, r, flush=True)
    return r

res = {v: probe(v, md[c]) for v, c in
       [("tp53_from_hist", "tp53"), ("wgd_from_hist", "wgd"),
        ("stage_from_hist", "stage_hi"), ("age_from_hist", "age_hi")]}
json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=2)
print("wrote results.json")

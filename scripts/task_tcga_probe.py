"""Ch2 Part A replication: can H&E features predict TP53/WGD in TCGA-OAC?

Mirrors the OCCAMS probe (mean-pooled UNI2, PCA64 + logistic regression,
5-fold x 5 seeds, shuffled-label controls). Pre-registered expectation from the
OCCAMS negative: AUC near 0.5. Cluster-side; needs tcga features + labels.
"""
import glob, json, os, re
import numpy as np, pandas as pd, h5py
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

FEAT = "/mnt/scratche/fast/fmlab/datasets/imaging/tcga_esca/features/20x_224px/features_uni_v2"
LABELS = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis/data/tcga_oac_labels.csv"
OUT = os.environ.get("OUTDIR", ".")

H_map = {}
for f in sorted(glob.glob(os.path.join(FEAT, "*.h5"))):
    m = re.search(r"(TCGA-\w{2}-\w{4})", os.path.basename(f))
    if not m: continue
    with h5py.File(f) as h:
        H_map[m.group(1)] = np.asarray(h["features"]).mean(0)
print("feature cases:", len(H_map))

lab = pd.read_csv(LABELS).set_index("barcode")

def probe(name, y_series):
    y_series = y_series.dropna()
    cases = sorted(set(H_map) & set(y_series.index))
    y = y_series.loc[cases].astype(int).values
    X = np.vstack([H_map[c] for c in cases])
    if len(set(y)) < 2 or min(np.bincount(y)) < 8:
        return {"n": len(cases), "skipped": "class too small"}
    def run(yy, seed):
        oof = np.zeros(len(yy))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, yy):
            p = Pipeline([("s", StandardScaler()), ("p", PCA(min(64, len(tr) - 1))),
                          ("l", LogisticRegression(C=0.5, class_weight="balanced", max_iter=4000))])
            p.fit(X[tr], yy[tr]); oof[te] = p.predict_proba(X[te])[:, 1]
        return roc_auc_score(yy, oof)
    aucs = [run(y, s) for s in range(5)]
    # 50-permutation empirical null: at n<=65 with heavy class imbalance the
    # shuffled AUC has sd ~0.1, so a 5-replicate "control" cannot be read as a
    # pipeline check — report the null distribution and the real AUC's percentile.
    null = [run(np.random.RandomState(s).permutation(y), s) for s in range(50)]
    real = float(np.mean(aucs))
    r = {"n": len(cases), "pos": int(y.sum()), "auc": round(real, 3),
         "auc_sd": round(float(np.std(aucs)), 3),
         "null_mean": round(float(np.mean(null)), 3),
         "null_sd": round(float(np.std(null)), 3),
         "null_pctile_of_real": round(float(np.mean([n_ <= real for n_ in null])), 3)}
    print(name, r)
    return r

res = {"tp53": probe("tp53", lab["tp53_mut"]), "wgd": probe("wgd", lab["wgd"])}
json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=2)
print("wrote results.json")

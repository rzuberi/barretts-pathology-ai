"""Diagnostic: per-slide mean features from the packs and from the per-slide files -> sklearn CV AUROC; duplicate check; tile/feature counts."""
import json, os, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
A = "/mnt/scratche/slow/fmlab/zuberi01/analysis/outreach_tilebio"; s = pd.read_csv(A + "/manifests/slides.csv").set_index("slide")
for ds in ("tcga", "c16"):
    for mo in ("uni2h", "virchow2"):
        ix = json.load(open(f"{A}/data/packs/{mo}_{ds}_index.json")); sl = list(ix["slides"])
        X = np.load(f"{A}/data/packs/{mo}_{ds}.npy", mmap_mode="r")
        M = np.stack([np.asarray(X[a:b], np.float32).mean(0) for a, b in (ix["slides"][k] for k in sl)])
        y = s.loc[sl, "label"].values
        p = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000)), M, y, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
        # tiles count consistency
        bad = sum(json.load(open(f"{A}/data/tiles/{ds}/{k}.json"))["n_tiles"] != ix["slides"][k][1] - ix["slides"][k][0] for k in sl[:60])
        Mn = M / np.linalg.norm(M, axis=1, keepdims=True); C = Mn @ Mn.T; np.fill_diagonal(C, 0)
        print(ds, mo, "n", len(sl), "CV AUROC meanpool", round(roc_auc_score(y, p), 3), "| count mismatches in first 60:", bad, "| max off-diag cos", round(C.max(), 5), "pairs>0.9999:", int((C > 0.9999).sum() // 2), flush=True)
        if ds == "c16":
            tr = np.array([s.loc[k, "split"] == "train" for k in sl]); clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000)).fit(M[tr], y[tr])
            print("   official split meanpool AUROC", round(roc_auc_score(y[~tr], clf.predict_proba(M[~tr])[:, 1]), 3), "train AUROC", round(roc_auc_score(y[tr], clf.predict_proba(M[tr])[:, 1]), 3))

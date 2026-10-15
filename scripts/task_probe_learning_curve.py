"""Follow-up to the visibility reversal: TP53/WGD/stage-from-histology AUC as a
function of n (subsampled from the TCGA pool). Explains the small-n negatives.
10 resamples per n; linear probe protocol identical to task_necessity_probes."""
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
NS = [65, 100, 140, 200, 300, 446]

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
                     "stage_hi": {"I": 0, "II": 0, "III": 1, "IV": 1}.get(stage.group(1) if stage else None)})
X = np.vstack(X); md = pd.DataFrame(rows)

def auc_at(Xi, yy, seed):
    oof = np.zeros(len(yy))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(Xi, yy):
        p = Pipeline([("s", StandardScaler()), ("p", PCA(min(64, len(tr) - 1))),
                      ("l", LogisticRegression(C=0.5, class_weight="balanced", max_iter=4000))])
        p.fit(Xi[tr], yy[tr]); oof[te] = p.predict_proba(Xi[te])[:, 1]
    return roc_auc_score(yy, oof)

res = {}
for target in ["tp53", "wgd", "stage_hi"]:
    y = pd.to_numeric(md[target], errors="coerce")
    mask = y.notna().values
    yy_all, X_all = y[mask].astype(int).values, X[mask]
    curve = {}
    for n in NS:
        if n >= len(yy_all):  # full-n: evaluate directly, same protocol as the probes task
            aucs_f = [auc_at(X_all, yy_all, s) for s in range(5)]
            curve[str(len(yy_all))] = {"auc_mean": round(float(np.mean(aucs_f)), 3),
                                       "auc_sd": round(float(np.std(aucs_f)), 3),
                                       "resamples": 5, "mode": "full"}
            print(target, "full", curve[str(len(yy_all))], flush=True)
            break
        aucs = []
        for s in range(10):
            rng = np.random.RandomState(s)
            # stratified subsample
            pos = np.where(yy_all == 1)[0]; neg = np.where(yy_all == 0)[0]
            k_pos = max(8, int(round(n * len(pos) / len(yy_all))))
            k_neg = n - k_pos
            if k_pos > len(pos) or k_neg > len(neg): continue
            idx = np.concatenate([rng.choice(pos, k_pos, replace=False),
                                  rng.choice(neg, k_neg, replace=False)])
            aucs.append(auc_at(X_all[idx], yy_all[idx], s))
        if aucs:
            curve[str(n)] = {"auc_mean": round(float(np.mean(aucs)), 3),
                             "auc_sd": round(float(np.std(aucs)), 3), "resamples": len(aucs)}
        print(target, n, curve.get(str(n)), flush=True)
        if n == len(yy_all): break
    res[target] = curve
json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=2)
print("wrote results.json")

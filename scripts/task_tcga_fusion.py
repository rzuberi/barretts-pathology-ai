"""Ch2 TCGA arm first-pass: H&E + genomics fusion -> 2-year OS in TCGA-OAC.

Endpoint: OS >= 730 days; patients censored before 730 days are excluded (CDR
gives event indicator + time, so censoring is handled by exclusion, matching the
OCCAMS v2 design). Arms: histology (mean-pooled UNI2), genomics (TP53, ploidy,
WGD, stage-coded), early fusion, late fusion. Same probe protocol as elsewhere:
PCA64+logreg, 5-fold x 5 seeds, shuffled controls, paired bootstrap on OOF.
"""
import glob, json, os, re
import numpy as np, pandas as pd, h5py
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss

FEAT = "/mnt/scratche/fast/fmlab/datasets/imaging/tcga_esca/features/20x_224px/features_uni_v2"
LABELS = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis/data/tcga_oac_labels.csv"
OUT = os.environ.get("OUTDIR", ".")
CUT = 730.0

H_map = {}
for f in sorted(glob.glob(os.path.join(FEAT, "*.h5"))):
    m = re.search(r"(TCGA-\w{2}-\w{4})", os.path.basename(f))
    if m:
        with h5py.File(f) as h:
            H_map[m.group(1)] = np.asarray(h["features"]).mean(0)

lab = pd.read_csv(LABELS)
lab = lab[lab["barcode"].isin(H_map)]
# label: 1 = survived past CUT; exclude censored-before-CUT
alive_past = lab["os_days"] >= CUT
dead_before = (lab["os_event"] == 1) & (lab["os_days"] < CUT)
lab = lab[alive_past | dead_before].copy()
lab["y"] = alive_past[alive_past | dead_before].astype(int)
stage = lab["ajcc_stage"].astype(str).str.extract(r"Stage (I{1,3}V?)", expand=False)
lab["stage_num"] = stage.map({"I": 1, "II": 2, "III": 3, "IV": 4}).fillna(2.5)

y = lab["y"].values
H = np.vstack([H_map[b] for b in lab["barcode"]])
G = np.column_stack([lab["tp53_mut"].fillna(0), lab["ploidy"].fillna(2.0),
                     lab["wgd"].fillna(0), lab["stage_num"], lab["age"].fillna(lab["age"].median())])
print(f"n={len(y)} pos(2yr-survivor)={int(y.sum())} H={H.shape} G={G.shape}")

def oof_probs(Xi, yy, seed, pca=None):
    oof = np.zeros(len(yy))
    steps = [("s", StandardScaler())]
    if pca: steps.append(("p", PCA(min(pca, Xi.shape[1], int(len(yy) * 0.8) - 2))))
    steps.append(("l", LogisticRegression(C=0.5, class_weight="balanced", max_iter=4000)))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(Xi, yy):
        p = Pipeline(steps); p.fit(Xi[tr], yy[tr]); oof[te] = p.predict_proba(Xi[te])[:, 1]
    return oof

arms = {"hist": lambda s, yy: oof_probs(H, yy, s, pca=64),
        "gen": lambda s, yy: oof_probs(G, yy, s),
        "early": lambda s, yy: oof_probs(np.column_stack([H, G * 10]), yy, s, pca=64)}
res, oofs = {}, {}
for name, fn in arms.items():
    per = [fn(s, y) for s in range(5)]
    oofs[name] = np.mean(per, axis=0)
    shuf = [roc_auc_score(np.random.RandomState(s).permutation(y),
                          fn(s, np.random.RandomState(s).permutation(y))) for s in range(5)]
    res[name] = {"auc": round(float(np.mean([roc_auc_score(y, p) for p in per])), 3),
                 "brier": round(float(brier_score_loss(y, oofs[name])), 3),
                 "shuffled_auc": round(float(np.mean(shuf)), 3)}
    print(name, res[name])
oofs["late"] = (oofs["hist"] + oofs["gen"]) / 2
res["late"] = {"auc": round(float(roc_auc_score(y, oofs["late"])), 3),
               "brier": round(float(brier_score_loss(y, oofs["late"])), 3)}

rng = np.random.RandomState(0)
for fus in ("late", "early"):
    d = []
    for _ in range(2000):
        b = rng.randint(0, len(y), len(y))
        if len(set(y[b])) < 2: continue
        d.append(roc_auc_score(y[b], oofs[fus][b]) - roc_auc_score(y[b], oofs["hist"][b]))
    res[fus]["delta_auc_vs_hist"] = {"mean": round(float(np.mean(d)), 3),
                                     "ci": [round(float(np.percentile(d, 2.5)), 3),
                                            round(float(np.percentile(d, 97.5)), 3)]}
    print(fus, "vs hist:", res[fus]["delta_auc_vs_hist"])

res["_meta"] = {"n": int(len(y)), "endpoint": "OS>=730d, early-censored excluded"}
json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=2)
print("wrote results.json")

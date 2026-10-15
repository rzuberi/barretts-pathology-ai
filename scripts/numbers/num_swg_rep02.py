#!/usr/bin/env python3
"""Item 13: second CV repeat of the frozen SWG release (new patient split, seed 20260924) vs the original repeat.
Patient level (max over samples), 2,000 patient bootstraps seed 0. Two late-mean constructions for BOTH repeats: late_mean_prob = plain mean of the image_only and cnv_only
probabilities (this IS the release late_mean family, verified corr 1.0 with its OOF), and late_mean_z = fold-local z-mean."""
import glob, json, os, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
F = "/mnt/scratche/slow/fmlab/zuberi01/phd/barretts_retraining/barretts_training/analysis/chapter1_lgd2_final_pre_event_20260713_final"
REPS = {"rep01": F + "/training_final_nested_cv_v1", "rep02": F + "_rep02/training_rep02_nested_cv"}; OUT = os.environ.get("OUTDIR", "."); NB = 2000
FAMS = ["image_only", "cnv_only", "early_fusion", "intermediate_fusion", "coattention_fusion"]
def load(root, fam):
    fs = glob.glob(f"{root}/{fam}/fold*/outer_test_predictions.csv")
    return pd.concat([pd.read_csv(f, dtype={"sample_id": str, "patient_id": str}) for f in fs]).reset_index(drop=True) if len(fs) == 5 else None
def zfold(v, folds):
    o = np.zeros(len(v))
    for f in np.unique(folds): te = folds == f; o[te] = (v[te] - v[te].mean()) / (v[te].std() + 1e-9)
    return o
res = {"_meta": {"n_boot": NB, "seed": 0, "rep02_split": "StratifiedKFold on patient-level label, seed 20260924, see REPEAT02_README.json"}, "reps": {}}
for rep, root in REPS.items():
    d = {f: load(root, f) for f in FAMS}; have = [f for f in FAMS if d[f] is not None]
    if not have: continue
    base = d[have[0]][["sample_id", "patient_id", "outer_fold", "y_true"]].copy()
    for f in have: base[f] = d[f].set_index("sample_id").y_prob.reindex(base.sample_id).values
    if "image_only" in have and "cnv_only" in have: base["late_mean_z"] = (zfold(base.image_only.values, base.outer_fold.values) + zfold(base.cnv_only.values, base.outer_fold.values)) / 2; base["late_mean_prob"] = (base.image_only.values + base.cnv_only.values) / 2; have = have + ["late_mean_z", "late_mean_prob"]
    g = base.groupby("patient_id"); y = g.y_true.max().values.astype(int); P = {f: g[f].max().values for f in have}
    rng = np.random.RandomState(0); n = len(y); B = []
    while len(B) < NB:
        s = rng.choice(n, n)
        if len(set(y[s])) < 2: continue
        B.append(s)
    ci = lambda v: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
    r = {"n_patients": int(n), "pos": int(y.sum()), "families_complete": have, "arms": {}}
    for f in have:
        r["arms"][f] = {"auroc": round(float(roc_auc_score(y, P[f])), 4), "auroc_ci": ci([roc_auc_score(y[s], P[f][s]) for s in B]), "auprc": round(float(average_precision_score(y, P[f])), 4)}
        if f != "image_only" and "image_only" in have: r["arms"][f]["delta_vs_image_only"] = round(float(roc_auc_score(y, P[f]) - roc_auc_score(y, P["image_only"])), 4); r["arms"][f]["delta_ci"] = ci([roc_auc_score(y[s], P[f][s]) - roc_auc_score(y[s], P["image_only"][s]) for s in B])
    res["reps"][rep] = r
if "rep01" in res["reps"] and "rep02" in res["reps"]:
    res["rep02_minus_rep01_point"] = {f: round(res["reps"]["rep02"]["arms"][f]["auroc"] - res["reps"]["rep01"]["arms"][f]["auroc"], 4) for f in res["reps"]["rep02"]["arms"] if f in res["reps"]["rep01"]["arms"]}
os.makedirs(OUT, exist_ok=True); json.dump(res, open(os.path.join(OUT, "swg_rep02.json"), "w"), indent=1); print(json.dumps(res, indent=None))

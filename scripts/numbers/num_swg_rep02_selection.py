#!/usr/bin/env python3
"""Item 13 follow-up: selection-adjusted test on the SECOND repeat, same machinery as task_swg_selection_adjusted.py
(labels permuted 2,000x with all arms' predictions fixed; statistic = max over fusion arms of AUROC(arm) - AUROC(image_only)),
run on rep01 (recomputed, fold-local late_mean) and rep02, plus the two-repeat average delta per arm."""
import glob, json, os, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
F = "/mnt/scratche/slow/fmlab/zuberi01/phd/barretts_retraining/barretts_training/analysis/chapter1_lgd2_final_pre_event_20260713_final"
REPS = {"rep01": F + "/training_final_nested_cv_v1", "rep02": F + "_rep02/training_rep02_nested_cv"}; OUT = os.environ.get("OUTDIR", "."); NP = 2000
FAMS = ["image_only", "cnv_only", "early_fusion", "intermediate_fusion", "coattention_fusion"]; FUS = ["early_fusion", "intermediate_fusion", "coattention_fusion", "late_mean_prob", "late_mean_z"]
def zfold(v, folds):
    o = np.zeros(len(v))
    for f in np.unique(folds): te = folds == f; o[te] = (v[te] - v[te].mean()) / (v[te].std() + 1e-9)
    return o
res = {"_meta": {"n_perm": NP, "seed": 0, "statistic": "max over fusion arms of AUROC(arm) - AUROC(image_only), labels permuted, predictions fixed"}}
deltas = {}
for rep, root in REPS.items():
    d = {f: pd.concat([pd.read_csv(x, dtype={"sample_id": str, "patient_id": str}) for x in glob.glob(f"{root}/{f}/fold*/outer_test_predictions.csv")]).reset_index(drop=True) for f in FAMS}
    base = d["image_only"][["sample_id", "patient_id", "outer_fold", "y_true"]].copy()
    for f in FAMS: base[f] = d[f].set_index("sample_id").y_prob.reindex(base.sample_id).values
    base["late_mean_z"] = (zfold(base.image_only.values, base.outer_fold.values) + zfold(base.cnv_only.values, base.outer_fold.values)) / 2; base["late_mean_prob"] = (base.image_only.values + base.cnv_only.values) / 2
    g = base.groupby("patient_id"); y = g.y_true.max().values.astype(int); P = {f: g[f].max().values for f in FAMS + ["late_mean_prob", "late_mean_z"]}
    obs = {f: roc_auc_score(y, P[f]) - roc_auc_score(y, P["image_only"]) for f in FUS}; deltas[rep] = obs
    rng = np.random.RandomState(0); perm = [rng.permutation(len(y)) for _ in range(NP)]
    dperm = {f: np.array([roc_auc_score(y[ix], P[f]) - roc_auc_score(y[ix], P["image_only"]) for ix in perm]) for f in FUS}
    best = max(FUS, key=lambda f: obs[f]); t_obs = obs[best]; t_null = np.max(np.stack([dperm[f] for f in FUS]), axis=0)
    res[rep] = {"observed_delta_vs_image_only": {f: round(float(v), 4) for f, v in obs.items()}, "selected_arm": best, "t_obs": round(float(t_obs), 4),
                "p_selection_adjusted": round(float((1 + (t_null >= t_obs).sum()) / (NP + 1)), 5), "p_selected_arm_unadjusted": round(float((1 + (dperm[best] >= t_obs).sum()) / (NP + 1)), 5),
                "p_unadjusted_per_arm": {f: round(float((1 + (dperm[f] >= obs[f]).sum()) / (NP + 1)), 5) for f in FUS}, "null_t_q95": round(float(np.percentile(t_null, 95)), 4)}
    print(rep, json.dumps(res[rep]), flush=True)
res["two_repeat_mean_delta_vs_image_only"] = {f: round(float((deltas["rep01"][f] + deltas["rep02"][f]) / 2), 4) for f in FUS}
os.makedirs(OUT, exist_ok=True); json.dump(res, open(os.path.join(OUT, "swg_rep02_selection.json"), "w"), indent=1); print(json.dumps(res["two_repeat_mean_delta_vs_image_only"]))

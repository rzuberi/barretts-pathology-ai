#!/usr/bin/env python3
"""Items 14-18 (24 Sep 2026) on the frozen SWG release (outer-fold OOF, patient level = max over samples).
14 thresholds fixed on the OTHER folds' OOF (train-fold proxy) instead of post hoc on all OOF;
15 paired bootstrap CI on Brier, late_mean - image_only (and all arms vs image_only);
16 recalibration inside CV (Platt on other folds' OOF logits): slope/intercept/Brier before vs after;
17 detection-intensity confound: time windows + horizons excluding samples taken <=183 d after the previous biopsy;
18 composition of the 12-36 month window vs the other windows."""
import glob, json, os
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss, roc_curve
from sklearn.linear_model import LogisticRegression
F = "/mnt/scratche/slow/fmlab/zuberi01/phd/barretts_retraining/barretts_training/analysis/chapter1_lgd2_final_pre_event_20260713_final"
R = F + "/training_final_nested_cv_v1"; OUT = os.environ.get("OUTDIR", "."); NB = 2000
ARMS = ["image_only", "cnv_only", "early_fusion", "intermediate_fusion", "coattention_fusion", "late_mean", "late_stack_logit"]; REF = "image_only"
coh = pd.read_csv(F + "/pre_event_cohort.csv", dtype=str); m = coh.assign(sample_id=coh.SampleID.astype(str)).set_index("sample_id")
cols = ["MonthsBeforeLastBiopsy", "DaysSincePreviousBiopsy", "CurrentGradeInt", "BiopsiesTotalForPatient", "IsFirstBiopsy"] + [f"Progress_in_{k}" for k in range(1, 6)]
fam = {a: pd.concat([pd.read_csv(f, dtype={"sample_id": str, "patient_id": str}) for f in glob.glob(f"{R}/{a}/fold*/outer_test_predictions.csv")]).reset_index(drop=True).merge(m[cols], left_on="sample_id", right_index=True, how="left").reset_index(drop=True) for a in ARMS}
def pat(d, score="y_prob"): return d.groupby("patient_id").agg(y=("y_true", "max"), p=(score, "max"), fold=("outer_fold", "first"))
P = {a: pat(fam[a]) for a in ARMS}; common = sorted(set.intersection(*(set(v.index) for v in P.values())))
y = P[REF].loc[common, "y"].values.astype(int); fold = P[REF].loc[common, "fold"].values; S = {a: P[a].loc[common, "p"].values for a in ARMS}; n = len(y)
rng = np.random.RandomState(0); boots = []
while len(boots) < NB:
    s = rng.choice(n, n)
    if len(set(y[s])) < 2: continue
    boots.append(s)
def ci(v): return [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
res = {"_meta": {"n_patients": n, "pos": int(y.sum()), "n_boot": NB, "seed": 0, "release": R}}
# ---- 14: thresholds from other folds ----
def thr_sens(yy, pp, se):
    fpr, tpr, thr = roc_curve(yy, pp); k = np.where(tpr >= se - 1e-9)[0]; k = k[np.argmax(1 - fpr[k])]; return float(thr[k])
def thr_spec(yy, pp, sp):
    fpr, tpr, thr = roc_curve(yy, pp); k = np.where(1 - fpr >= sp)[0]; k = k[np.argmax(tpr[k])]; return float(thr[k])
res["item14_thresholds_from_other_folds"] = {}
for a in ARMS:
    o = {}
    for nm, fn, tgt in (("sens_0.95", thr_sens, 0.95), ("sens_1.0", thr_sens, 1.0), ("spec_0.80", thr_spec, 0.80), ("spec_0.86", thr_spec, 0.86)):
        flag = np.zeros(n, bool)
        for f in np.unique(fold):
            te = fold == f; tr = ~te; t = fn(y[tr], S[a][tr], tgt); flag[te] = S[a][te] >= t
        sens = float(flag[y == 1].mean()); spec = float((~flag[y == 0]).mean())
        bs_s = [float(flag[s][y[s] == 1].mean()) for s in boots]; bs_p = [float((~flag[s][y[s] == 0]).mean()) for s in boots]
        o[nm] = {"achieved_sensitivity": round(sens, 4), "sens_ci": ci(bs_s), "achieved_specificity": round(spec, 4), "spec_ci": ci(bs_p), "flagged_of_n": [int(flag.sum()), n]}
    res["item14_thresholds_from_other_folds"][a] = o
# ---- 15: Brier paired ----
res["item15_brier"] = {}
for a in ARMS:
    b = brier_score_loss(y, S[a]); d = np.array([brier_score_loss(y[s], S[a][s]) - brier_score_loss(y[s], S[REF][s]) for s in boots])
    res["item15_brier"][a] = {"brier": round(float(b), 4), "brier_ci": ci([brier_score_loss(y[s], S[a][s]) for s in boots]), "delta_vs_image_only": round(float(b - brier_score_loss(y, S[REF])), 4), "delta_ci": ci(d)}
# ---- 16: recalibration within CV (Platt on other folds' OOF logits) ----
def logit(p): p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))
def slope_int(yy, pp):
    lr = LogisticRegression(C=1e6).fit(logit(pp)[:, None], yy); return float(lr.coef_[0][0]), float(lr.intercept_[0])
res["item16_recalibration"] = {}
for a in ("image_only", "cnv_only", "late_mean", "late_stack_logit"):
    p0 = S[a]; p1 = np.zeros(n)
    for f in np.unique(fold):
        te = fold == f; tr = ~te; lr = LogisticRegression(C=1e6).fit(logit(p0[tr])[:, None], y[tr]); p1[te] = lr.predict_proba(logit(p0[te])[:, None])[:, 1]
    s0, i0 = slope_int(y, p0); s1, i1 = slope_int(y, p1)
    res["item16_recalibration"][a] = {"before": {"slope": round(s0, 3), "intercept": round(i0, 3), "brier": round(float(brier_score_loss(y, p0)), 4), "auroc": round(float(roc_auc_score(y, p0)), 4)},
                                       "after_platt_cv": {"slope": round(s1, 3), "intercept": round(i1, 3), "brier": round(float(brier_score_loss(y, p1)), 4), "brier_ci": ci([brier_score_loss(y[s], p1[s]) for s in boots]), "auroc": round(float(roc_auc_score(y, p1)), 4)},
                                       "brier_delta_after_minus_before_ci": ci([brier_score_loss(y[s], p1[s]) - brier_score_loss(y[s], p0[s]) for s in boots])}
# ---- 17: detection-intensity exclusion ----
def windows(d, excl):
    mb = pd.to_numeric(d.MonthsBeforeLastBiopsy, errors="coerce").values; yy = d.y_true.values.astype(int); pp = d.y_prob.values; g = d.patient_id.values; dsp = pd.to_numeric(d.DaysSincePreviousBiopsy, errors="coerce").values
    keep = ~(dsp <= 183) if excl else np.ones(len(d), bool)
    o = {"samples_kept": int(keep.sum()), "samples_total": int(len(d))}
    for lo, hi, nm in [(0, 12, "0-12m"), (12, 36, "12-36m"), (36, 1e9, ">36m")]:
        sel = keep & ((yy == 0) | ((yy == 1) & (mb >= lo) & (mb < hi))); ys, ps, gs = yy[sel], pp[sel], g[sel]; npos = int(ys.sum())
        if npos < 5 or (ys == 0).sum() < 5: o[nm] = {"n_progressor_samples": npos, "auroc": None}; continue
        r2 = np.random.RandomState(0); up = np.unique(gs); idx_of = {u: np.where(gs == u)[0] for u in up}; out = []
        while len(out) < 1000:
            s = np.concatenate([idx_of[u] for u in r2.choice(up, len(up))])
            if len(set(ys[s])) < 2: continue
            out.append(roc_auc_score(ys[s], ps[s]))
        o[nm] = {"n_progressor_samples": npos, "n_nonprogressor_samples": int((ys == 0).sum()), "auroc": round(float(roc_auc_score(ys, ps)), 4), "ci": ci(out)}
    return o
def horizons(d, excl):
    dsp = pd.to_numeric(d.DaysSincePreviousBiopsy, errors="coerce").values; keep = ~(dsp <= 183) if excl else np.ones(len(d), bool); dd = d[keep]; o = {}
    for k in range(1, 6):
        t = pd.to_numeric(dd[f"Progress_in_{k}"], errors="coerce"); ok = t.notna(); pt_ = dd[ok].assign(t=t[ok].astype(int)).groupby("patient_id").agg(y=("t", "max"), p=("y_prob", "max"))
        if pt_.y.sum() < 3: o[f"Progress_in_{k}"] = {"n_patients": len(pt_), "pos": int(pt_.y.sum()), "auroc": None}; continue
        o[f"Progress_in_{k}"] = {"n_patients": len(pt_), "pos": int(pt_.y.sum()), "auroc": round(float(roc_auc_score(pt_.y, pt_.p)), 4)}
    return o
res["item17_detection_intensity"] = {"rule": "exclude samples with DaysSincePreviousBiopsy <= 183 (a repeat within 6 months of the previous biopsy)"}
for a in ("image_only", "cnv_only", "late_mean"):
    res["item17_detection_intensity"][a] = {"all_samples": {"windows": windows(fam[a], False), "horizons": horizons(fam[a], False)},
                                            "excluding_repeats_within_6m": {"windows": windows(fam[a], True), "horizons": horizons(fam[a], True)}}
# ---- 18: composition of the 12-36 m window ----
d = fam["image_only"]; mb = pd.to_numeric(d.MonthsBeforeLastBiopsy, errors="coerce"); prog = d[d.y_true.astype(int) == 1].copy(); prog["mb"] = mb[prog.index]
prog["window"] = pd.cut(prog.mb, [-0.01, 12, 36, 1e9], labels=["0-12m", "12-36m", ">36m"])
def summ(g):
    return {"n_samples": int(len(g)), "n_patients": int(g.patient_id.nunique()), "grade_int_dist": pd.to_numeric(g.CurrentGradeInt, errors="coerce").value_counts().sort_index().to_dict(),
            "mean_grade_int": round(float(pd.to_numeric(g.CurrentGradeInt, errors="coerce").mean()), 3), "median_biopsies_total_for_patient": float(pd.to_numeric(g.BiopsiesTotalForPatient, errors="coerce").median()),
            "median_days_since_previous": float(pd.to_numeric(g.DaysSincePreviousBiopsy, errors="coerce").median()), "frac_first_biopsy": round(float(g.IsFirstBiopsy.astype(str).str.lower().isin(["true", "1"]).mean()), 3),
            "image_only_median_prob": round(float(g.y_prob.median()), 3)}
res["item18_window_composition_progressor_samples"] = {str(w): summ(g) for w, g in prog.groupby("window", observed=True)}
neg = d[d.y_true.astype(int) == 0]; res["item18_window_composition_progressor_samples"]["non_progressor_samples_reference"] = summ(neg)
res["item18_window_composition_progressor_samples"]["patients_in_12_36m_window"] = sorted(prog[prog.window == "12-36m"].patient_id.unique().tolist())
os.makedirs(OUT, exist_ok=True); json.dump(res, open(os.path.join(OUT, "swg_robustness.json"), "w"), indent=1, default=str); print(json.dumps(res, indent=None, default=str)[:6000])

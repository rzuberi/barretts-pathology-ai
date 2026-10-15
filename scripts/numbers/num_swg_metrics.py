"""NUMBERS C for SWG: per-arm AUROC/AUPRC with patient-bootstrap CIs, sens/spec at Youden and fixed
specificities, Brier + calibration slope/intercept + reliability bins, grade-only clinical baseline,
performance by time-before-progression and by Progress_in_k horizon, folds/seeds actually present."""
import glob, json, os
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, brier_score_loss
from sklearn.linear_model import LogisticRegression
F = "/mnt/scratche/slow/fmlab/zuberi01/phd/barretts_retraining/barretts_training/analysis/chapter1_lgd2_final_pre_event_20260713_final"
R = F + "/training_final_nested_cv_v1"; OUT = os.environ.get("OUTDIR", "."); NB = 2000
coh = pd.read_csv(F + "/pre_event_cohort.csv", dtype=str)
fam = {}
for d in sorted(glob.glob(R + "/*")):
    fs = glob.glob(d + "/fold*/outer_test_predictions.csv")
    if fs: fam[os.path.basename(d)] = pd.concat([pd.read_csv(f, dtype={"sample_id": str, "patient_id": str}) for f in fs])
res = {"families": sorted(fam), "folds_seen": {k: sorted(v.outer_fold.unique().tolist()) for k, v in fam.items()},
       "seeds_seen": {k: sorted(v.seed.unique().tolist()) for k, v in fam.items()}, "rows_per_family": {k: len(v) for k, v in fam.items()},
       "design_note": "single nested-CV run, 5 outer folds, patient-disjoint (fold_id_rep01); one seed per family in release"}
def boot(y, p, g, fn):
    rng = np.random.RandomState(0); up = np.unique(g); idx_of = {u: np.where(g == u)[0] for u in up}; out = []
    for _ in range(NB):
        s = np.concatenate([idx_of[u] for u in rng.choice(up, len(up))])
        if len(set(y[s])) < 2: continue
        out.append(fn(y[s], p[s]))
    return [round(float(np.percentile(out, 2.5)), 4), round(float(np.percentile(out, 97.5)), 4)]
def sens_spec(y, p):
    fpr, tpr, thr = roc_curve(y, p); j = np.argmax(tpr - fpr); o = {"youden": {"threshold": float(thr[j]), "sensitivity": float(tpr[j]), "specificity": float(1 - fpr[j])}}
    for sp in (0.8, 0.9):
        k = np.where(1 - fpr >= sp)[0]; k = k[np.argmax(tpr[k])]; o[f"at_specificity_{sp}"] = {"threshold": float(thr[k]), "sensitivity": float(tpr[k]), "specificity": float(1 - fpr[k])}
    prev = float(np.mean(y))
    for se in (0.95, 1.0):   # rule-out operating points: lock sensitivity, report specificity (+ NPV/PPV at cohort prevalence and at 0.5%/yr real-world)
        k = np.where(tpr >= se - 1e-9)[0]; k = k[np.argmax(1 - fpr[k])]; sens, spec = float(tpr[k]), float(1 - fpr[k])
        def npv(pr): return (spec * (1 - pr)) / (spec * (1 - pr) + (1 - sens) * pr) if (spec * (1 - pr) + (1 - sens) * pr) > 0 else None
        def ppv(pr): return (sens * pr) / (sens * pr + (1 - spec) * (1 - pr)) if (sens * pr + (1 - spec) * (1 - pr)) > 0 else None
        o[f"at_sensitivity_{se}"] = {"threshold": float(thr[k]), "sensitivity": sens, "specificity": spec, "n_flagged_of_n": [int((p >= thr[k]).sum()), int(len(p))],
                                    "npv_cohort_prevalence": npv(prev), "ppv_cohort_prevalence": ppv(prev), "npv_at_0.5pct": npv(0.005), "ppv_at_0.5pct": ppv(0.005)}
    return o
def calib(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6); lg = np.log(p / (1 - p))[:, None]
    lr = LogisticRegression(C=1e6).fit(lg, y); bins = pd.qcut(p, 10, duplicates="drop")
    tab = pd.DataFrame({"p": p, "y": y, "b": bins}).groupby("b", observed=True).agg(mean_pred=("p", "mean"), obs_rate=("y", "mean"), n=("y", "size"))
    return {"brier": float(brier_score_loss(y, p)), "calibration_slope": float(lr.coef_[0][0]), "calibration_intercept": float(lr.intercept_[0]),
            "reliability_bins": [{"mean_pred": round(float(r.mean_pred), 3), "obs_rate": round(float(r.obs_rate), 3), "n": int(r.n)} for r in tab.itertuples()]}
res["arms"] = {}
for k, d in fam.items():
    pat = d.groupby("patient_id").agg(y=("y_true", "max"), p=("y_prob", "max")); y, p = pat.y.values.astype(int), pat.p.values; g = pat.index.values
    ys, ps, gs = d.y_true.values.astype(int), d.y_prob.values, d.patient_id.values
    res["arms"][k] = {"patient_level": {"n": len(y), "pos": int(y.sum()), "auroc": round(float(roc_auc_score(y, p)), 4), "auroc_ci": boot(y, p, g, roc_auc_score),
                                        "auprc": round(float(average_precision_score(y, p)), 4), "auprc_ci": boot(y, p, g, average_precision_score),
                                        "operating_points": sens_spec(y, p), "calibration": calib(y, p)},
                      "sample_level": {"n": len(ys), "pos": int(ys.sum()), "auroc": round(float(roc_auc_score(ys, ps)), 4), "auroc_ci": boot(ys, ps, gs, roc_auc_score),
                                       "auprc": round(float(average_precision_score(ys, ps)), 4)}}
# grade-only baseline (current pathology grade as the score)
m = coh.assign(sample_id=coh.SampleID.astype(str)).set_index("sample_id")
anyf = fam[res["families"][0]]; gj = anyf.merge(m[["Label", "MonthsBeforeLastBiopsy"] + [c for c in coh.columns if c.startswith("Progress_in_")]], left_on="sample_id", right_index=True, how="left")
gr = pd.to_numeric(gj.Label, errors="coerce").fillna(0).values; ys = gj.y_true.values.astype(int)
pg = gj.assign(gr=gr).groupby("patient_id").agg(y=("y_true", "max"), gr=("gr", "max"))
res["clinical_baseline_current_grade"] = {"sample_level_auroc": round(float(roc_auc_score(ys, gr)), 4), "patient_level_auroc_max_grade": round(float(roc_auc_score(pg.y.astype(int), pg.gr)), 4),
                                         "note": "pathologist grade of the biopsy (0-4 code) used directly as the risk score; no other clinical variables in the release"}
# time structure: AUC by months-before-last-biopsy bucket (progressor samples in bucket vs all non-progressor samples), per family
res["by_time_before_event"] = {}
for k, d in fam.items():
    j = d.merge(m[["MonthsBeforeLastBiopsy"] + [c for c in coh.columns if c.startswith("Progress_in_")]], left_on="sample_id", right_index=True, how="left")
    mb = pd.to_numeric(j.MonthsBeforeLastBiopsy, errors="coerce"); yy = j.y_true.values.astype(int); pp = j.y_prob.values
    o = {}
    for lo, hi, nm in [(0, 12, "0-12m"), (12, 36, "12-36m"), (36, 1e9, ">36m")]:
        sel = (yy == 0) | ((yy == 1) & (mb >= lo) & (mb < hi))
        npos = int(((yy == 1) & (mb >= lo) & (mb < hi)).sum())
        o[nm] = {"n_progressor_samples": npos, "auroc": round(float(roc_auc_score(yy[sel], pp[sel])), 4) if npos >= 5 else None}
    for c in [c for c in j.columns if c.startswith("Progress_in_")]:
        t = pd.to_numeric(j[c], errors="coerce"); ok = t.notna()
        if ok.sum() > 20 and 0 < t[ok].sum() < ok.sum(): o[c] = {"n_pos": int(t[ok].sum()), "n": int(ok.sum()), "auroc": round(float(roc_auc_score(t[ok].astype(int), pp[ok])), 4)}
    res["by_time_before_event"][k] = o
json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=2, default=str); print(json.dumps({k: res[k] for k in ("families", "folds_seen", "seeds_seen", "clinical_baseline_current_grade")}, indent=1, default=str))

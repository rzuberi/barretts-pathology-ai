#!/usr/bin/env python3
"""Item 19: patient-resampled bootstrap CIs (2,000, seed 0) for every CNV-as-text run present, plus paired CIs vs the
pathologist grade-as-score baseline and vs the trained CNV-only model's OOF (release cnv_only arm)."""
import glob, json, os, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
def fast_auc(y, s):
    r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"; F = "/mnt/scratche/slow/fmlab/zuberi01/phd/barretts_retraining/barretts_training/analysis/chapter1_lgd2_final_pre_event_20260713_final"
OUT = os.environ.get("OUTDIR", "."); NB = 2000
man = pd.read_csv(F + "/training_manifest.csv", dtype=str); coh = pd.read_csv(F + "/pre_event_cohort.csv", dtype=str).set_index("SampleID")
man["grade"] = pd.to_numeric(pd.Series(coh.Label.reindex(man.sample_id).values), errors="coerce").fillna(0).values
cnv = pd.concat([pd.read_csv(f, dtype={"sample_id": str, "patient_id": str}) for f in glob.glob(F + "/training_final_nested_cv_v1/cnv_only/fold*/outer_test_predictions.csv")])
man = man.merge(cnv[["sample_id", "y_prob"]].rename(columns={"y_prob": "cnv_model"}), on="sample_id", how="left")
def ci(v): return [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
res = {"_meta": {"n_boot": NB, "seed": 0, "unit": "patient (max risk over samples)"}, "runs": {}}
for f in sorted(glob.glob(T + "/feasibility/runs/cnvtext*/output/cnv_text_*.csv")):
    run = f.split("/runs/")[1].split("/")[0]; d = pd.read_csv(f, dtype={"sample_id": str}); d["risk"] = pd.to_numeric(d.risk, errors="coerce"); d = d.merge(man, on="sample_id"); d = d[d.risk.notna()]
    pt = d.groupby("patient_id").agg(y=("y_progressor", lambda s: s.astype(int).max()), r=("risk", "max"), g=("grade", "max"), c=("cnv_model", "max")); n = len(pt); y = pt.y.values
    print(run, "patients", n, "class counts", dict(zip(*np.unique(y, return_counts=True))), flush=True)
    if n < 10 or len(set(y)) < 2: print(run, "SKIPPED (degenerate)", flush=True); continue
    rng = np.random.RandomState(0); B = []; tries = 0
    while len(B) < NB and tries < 50 * NB:
        tries += 1; s = rng.choice(n, n)
        if len(set(y[s])) < 2: continue
        B.append(s)
    def A(col, s): return fast_auc(y[s], pt[col].values[s])
    res["runs"][run] = {"file": os.path.basename(f), "n_samples_parsed": int(len(d)), "n_patients": n, "pos": int(y.sum()),
        "patient_auroc": round(float(roc_auc_score(y, pt.r)), 4), "ci": ci([A("r", s) for s in B]),
        "sample_auroc": round(float(roc_auc_score(d.y_progressor.astype(int), d.risk)), 4),
        "grade_baseline_auroc": round(float(roc_auc_score(y, pt.g)), 4), "delta_vs_grade": ci([A("r", s) - A("g", s) for s in B]),
        "trained_cnv_model_auroc": round(float(roc_auc_score(y, pt.c)), 4), "delta_vs_trained_cnv": ci([A("r", s) - A("c", s) for s in B])}
    print(run, json.dumps(res["runs"][run]))
os.makedirs(OUT, exist_ok=True); json.dump(res, open(os.path.join(OUT, "cnvtext_cis.json"), "w"), indent=1)

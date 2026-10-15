#!/usr/bin/env python3
"""Items 2-3 (24 Sep 2026): SWG frozen release, patient-level horizon AUROCs (Progress_in_1..5) for
image_only / cnv_only / late_mean with 2,000 patient bootstraps (seed 0) and paired late_mean - image_only;
time-to-event window AUROCs (0-12, 12-36, >36 months; progressor samples in window vs all non-progressor
samples, as item 22) with patient-clustered bootstrap CIs.
Patient-level horizon label = any evaluable sample of the patient has Progress_in_k = 1; score = max y_prob
over the patient's evaluable samples; patients with no evaluable sample for that k are dropped."""
import glob, json, os
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
F = "/mnt/scratche/slow/fmlab/zuberi01/phd/barretts_retraining/barretts_training/analysis/chapter1_lgd2_final_pre_event_20260713_final"
R = F + "/training_final_nested_cv_v1"; OUT = os.environ.get("OUTDIR", "."); NB = 2000
ARMS = ["image_only", "cnv_only", "late_mean"]
coh = pd.read_csv(F + "/pre_event_cohort.csv", dtype=str); m = coh.assign(sample_id=coh.SampleID.astype(str)).set_index("sample_id")
HZ = [f"Progress_in_{k}" for k in range(1, 6)]
fam = {a: pd.concat([pd.read_csv(f, dtype={"sample_id": str, "patient_id": str}) for f in glob.glob(f"{R}/{a}/fold*/outer_test_predictions.csv")])
          .merge(m[["MonthsBeforeLastBiopsy"] + HZ], left_on="sample_id", right_index=True, how="left") for a in ARMS}
def ci(v): return [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
res = {"_meta": {"arms": ARMS, "n_boot": NB, "seed": 0, "source_predictions": R, "source_cohort": F + "/pre_event_cohort.csv"}, "horizons_patient_level": {}, "time_windows_sample_level": {}}
# ---- horizons, patient level
for k in HZ:
    pats = {}
    for a in ARMS:
        d = fam[a]; t = pd.to_numeric(d[k], errors="coerce"); ok = t.notna()
        pats[a] = d[ok].assign(t=t[ok].astype(int)).groupby("patient_id").agg(y=("t", "max"), p=("y_prob", "max"))
    common = sorted(set.intersection(*(set(v.index) for v in pats.values())))
    y = pats[ARMS[0]].loc[common, "y"].values.astype(int); P = {a: pats[a].loc[common, "p"].values for a in ARMS}
    rng = np.random.RandomState(0); n = len(y); boots = []
    while len(boots) < NB:
        s = rng.choice(n, n)
        if len(set(y[s])) < 2: continue
        boots.append(s)
    B = {a: np.array([roc_auc_score(y[s], P[a][s]) for s in boots]) for a in ARMS}
    res["horizons_patient_level"][k] = {"n_patients": n, "pos": int(y.sum()),
        **{a: {"auroc": round(float(roc_auc_score(y, P[a])), 4), "ci": ci(B[a])} for a in ARMS},
        "late_mean_minus_image_only": {"delta": round(float(roc_auc_score(y, P["late_mean"]) - roc_auc_score(y, P["image_only"])), 4), "ci": ci(B["late_mean"] - B["image_only"])}}
# ---- time windows, sample level, patient-clustered bootstrap
for a in ARMS:
    d = fam[a]; mb = pd.to_numeric(d.MonthsBeforeLastBiopsy, errors="coerce").values; yy = d.y_true.values.astype(int); pp = d.y_prob.values; g = d.patient_id.values
    o = {}
    for lo, hi, nm in [(0, 12, "0-12m"), (12, 36, "12-36m"), (36, 1e9, ">36m")]:
        sel = (yy == 0) | ((yy == 1) & (mb >= lo) & (mb < hi)); ys, ps, gs = yy[sel], pp[sel], g[sel]
        npos = int(ys.sum()); nneg = int((ys == 0).sum())
        rng = np.random.RandomState(0); up = np.unique(gs); idx_of = {u: np.where(gs == u)[0] for u in up}; out = []
        while len(out) < NB:
            s = np.concatenate([idx_of[u] for u in rng.choice(up, len(up))])
            if len(set(ys[s])) < 2: continue
            out.append(roc_auc_score(ys[s], ps[s]))
        o[nm] = {"n_progressor_samples": npos, "n_nonprogressor_samples": nneg, "n_patients": int(len(up)), "auroc": round(float(roc_auc_score(ys, ps)), 4), "ci": ci(out)}
    res["time_windows_sample_level"][a] = o
os.makedirs(OUT, exist_ok=True); json.dump(res, open(os.path.join(OUT, "swg_horizons_patient.json"), "w"), indent=1)
print(json.dumps(res, indent=None))

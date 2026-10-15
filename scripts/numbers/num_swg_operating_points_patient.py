#!/usr/bin/env python3
"""Patient-level operating points for all seven arms of the frozen SWG release.

Unit = patient (max y_true, max y_prob over the patient's samples, exactly as
task_swg_oof_analyses.py), outer-fold OOF predictions. Threshold rule and
bootstrap identical to item 19 (num_swg_metrics.py): lock sensitivity >= s and
take the threshold with the highest specificity; lock specificity >= s and take
the highest sensitivity; 2,000 patient-resampled bootstraps, RandomState(0),
percentile 95% CIs. Paired deltas vs image_only use the same resample indices.
"""
import glob, json, os
import numpy as np, pandas as pd
from sklearn.metrics import roc_curve

R = "/mnt/scratche/slow/fmlab/zuberi01/phd/barretts_retraining/barretts_training/analysis/chapter1_lgd2_final_pre_event_20260713_final/training_final_nested_cv_v1"
OUT = os.environ.get("OUTDIR", "."); NB = 2000; REF = "image_only"
ARMS = ["image_only", "cnv_only", "early_fusion", "intermediate_fusion", "coattention_fusion", "late_mean", "late_stack_logit"]

pat = {}
for a in ARMS:
    df = pd.concat([pd.read_csv(f, dtype={"sample_id": str, "patient_id": str}) for f in glob.glob(f"{R}/{a}/fold*/outer_test_predictions.csv")])
    pat[a] = df.groupby("patient_id").agg(y=("y_true", "max"), p=("y_prob", "max"))
common = sorted(set.intersection(*(set(v.index) for v in pat.values())))
y = pat[REF].loc[common, "y"].values.astype(int); P = {a: pat[a].loc[common, "p"].values for a in ARMS}
n, npos = len(y), int(y.sum())

def spec_at_sens(yy, pp, se):
    fpr, tpr, thr = roc_curve(yy, pp); k = np.where(tpr >= se - 1e-9)[0]; k = k[np.argmax(1 - fpr[k])]
    return float(1 - fpr[k]), float(tpr[k]), float(thr[k]), int((pp >= thr[k]).sum())
def sens_at_spec(yy, pp, sp):
    fpr, tpr, thr = roc_curve(yy, pp); k = np.where(1 - fpr >= sp)[0]; k = k[np.argmax(tpr[k])]
    return float(tpr[k]), float(1 - fpr[k]), float(thr[k])

# shared bootstrap indices: patient = unit, same generator as item 19
rng = np.random.RandomState(0); up = np.arange(n); boots = []
while len(boots) < NB:
    s = rng.choice(up, n)
    if len(set(y[s])) < 2: continue
    boots.append(s)
def ci(vals): return [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)]

res = {"_meta": {"unit": "patient (max over samples)", "n_patients": n, "positives": npos, "arms": ARMS, "reference_arm": REF,
                 "threshold_rule": "item 19 / num_swg_metrics.py: lock sens>=s, max spec; lock spec>=s, max sens", "n_boot": NB, "seed": 0,
                 "score": "y_prob (uncalibrated), outer-fold OOF"}, "arms": {}}
bs = {a: {"s95": [], "s100": [], "se80": [], "se86": []} for a in ARMS}
for s in boots:
    for a in ARMS:
        bs[a]["s95"].append(spec_at_sens(y[s], P[a][s], 0.95)[0]); bs[a]["s100"].append(spec_at_sens(y[s], P[a][s], 1.0)[0])
        bs[a]["se80"].append(sens_at_spec(y[s], P[a][s], 0.80)[0]); bs[a]["se86"].append(sens_at_spec(y[s], P[a][s], 0.86)[0])
rows = []
for a in ARMS:
    sp95, se95, t95, f95 = spec_at_sens(y, P[a], 0.95); sp100, se100, t100, f100 = spec_at_sens(y, P[a], 1.0)
    se80, sp80, _ = sens_at_spec(y, P[a], 0.80); se86, sp86, _ = sens_at_spec(y, P[a], 0.86)
    d95 = np.array(bs[a]["s95"]) - np.array(bs[REF]["s95"])
    r = {"spec_at_sens_0.95": round(sp95, 4), "spec_at_sens_0.95_ci": ci(bs[a]["s95"]), "sens_achieved_0.95": round(se95, 4), "flagged_of_n_0.95": [f95, n],
         "spec_at_sens_1.0": round(sp100, 4), "spec_at_sens_1.0_ci": ci(bs[a]["s100"]), "flagged_of_n_1.0": [f100, n],
         "sens_at_spec_0.80": round(se80, 4), "sens_at_spec_0.80_ci": ci(bs[a]["se80"]), "spec_achieved_0.80": round(sp80, 4),
         "sens_at_spec_0.86": round(se86, 4), "sens_at_spec_0.86_ci": ci(bs[a]["se86"]), "spec_achieved_0.86": round(sp86, 4),
         "delta_spec_at_sens_0.95_vs_image_only": round(sp95 - res["arms"][REF]["spec_at_sens_0.95"], 4) if REF in res["arms"] else 0.0,
         "delta_spec_at_sens_0.95_vs_image_only_ci": ci(d95) if a != REF else [0.0, 0.0]}
    res["arms"][a] = r
    rows.append(f"| {a} | {sp95:.3f} [{r['spec_at_sens_0.95_ci'][0]:.3f}, {r['spec_at_sens_0.95_ci'][1]:.3f}] | {f95}/{n} | {sp100:.3f} [{r['spec_at_sens_1.0_ci'][0]:.3f}, {r['spec_at_sens_1.0_ci'][1]:.3f}] | {f100}/{n} | "
                f"{se80:.3f} [{r['sens_at_spec_0.80_ci'][0]:.3f}, {r['sens_at_spec_0.80_ci'][1]:.3f}] | {se86:.3f} [{r['sens_at_spec_0.86_ci'][0]:.3f}, {r['sens_at_spec_0.86_ci'][1]:.3f}] | "
                f"{r['delta_spec_at_sens_0.95_vs_image_only']:+.3f} [{r['delta_spec_at_sens_0.95_vs_image_only_ci'][0]:+.3f}, {r['delta_spec_at_sens_0.95_vs_image_only_ci'][1]:+.3f}] |")
# fix reference delta after loop (image_only computed first, so others are already relative to it)
res["arms"][REF]["delta_spec_at_sens_0.95_vs_image_only"] = 0.0
hdr = (f"| arm | spec @ sens 0.95 [95% CI] | flagged/n | spec @ sens 1.0 [95% CI] | flagged/n | sens @ spec 0.80 [95% CI] | sens @ spec 0.86 [95% CI] | Δ spec@sens0.95 vs image_only [paired CI] |\n"
       "|---|---|---|---|---|---|---|---|")
table = hdr + "\n" + "\n".join(rows)
res["table_markdown"] = table
os.makedirs(OUT, exist_ok=True)
json.dump(res, open(os.path.join(OUT, "swg_operating_points_patient.json"), "w"), indent=1)
print(f"patients={n} positives={npos} boots={len(boots)}"); print(table)

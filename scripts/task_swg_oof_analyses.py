"""Two cheap analyses on the frozen SWG release OOF (no retraining):

1. WINNER'S-CURSE quantification (GLM proposal): bootstrap patients 500x; on
   each resample pick the best of the 7 families by AUPRC. Readout: win rate
   per family, late-mean's win rate, and the optimism gap (winner's resample
   AUPRC minus its full-sample AUPRC).
2. RESIDUAL COMPLEMENTARITY (Terra-Pro proposal): nested logistic models on
   patient-level OOF risks — does CNV risk add to histology risk (and vice
   versa)? Likelihood-ratio tests + paired bootstrap delta AUC.
"""
import glob, json, os
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from scipy.stats import chi2

R = "/mnt/scratche/slow/fmlab/zuberi01/phd/barretts_retraining/barretts_training/analysis/chapter1_lgd2_final_pre_event_20260713_final/training_final_nested_cv_v1"
OUT = os.environ.get("OUTDIR", ".")
N_BOOT = 500

fam_oof = {}
for d in sorted(glob.glob(os.path.join(R, "*"))):
    fam = os.path.basename(d)
    files = glob.glob(os.path.join(d, "fold*/outer_test_predictions.csv"))
    if not files: continue
    df = pd.concat([pd.read_csv(f) for f in files])
    pat = df.groupby("patient_id").agg(y=("y_true", "max"), p=("y_prob", "max"))
    fam_oof[fam] = pat
fams = sorted(fam_oof)
common = sorted(set.intersection(*(set(v.index) for v in fam_oof.values())))
y = fam_oof[fams[0]].loc[common, "y"].values.astype(int)
P = {f: fam_oof[f].loc[common, "p"].values for f in fams}
print(f"families={fams} patients={len(common)} pos={y.sum()}", flush=True)

full_auprc = {f: average_precision_score(y, P[f]) for f in fams}
rng = np.random.RandomState(0)
wins = {f: 0 for f in fams}; optimism = []
for b in range(N_BOOT):
    idx = rng.randint(0, len(y), len(y))
    if y[idx].sum() in (0, len(y)): continue
    scores = {f: average_precision_score(y[idx], P[f][idx]) for f in fams}
    w = max(scores, key=scores.get)
    wins[w] += 1
    optimism.append(scores[w] - full_auprc[w])
tot = sum(wins.values())
res = {"_meta": {"n_patients": len(common), "pos": int(y.sum()), "n_boot": tot},
       "full_sample_auprc": {f: round(v, 4) for f, v in full_auprc.items()},
       "winners_curse": {
           "win_rate": {f: round(wins[f] / tot, 3) for f in fams},
           "late_mean_win_rate": round(wins.get("late_mean", 0) / tot, 3),
           "mean_winner_optimism_auprc": round(float(np.mean(optimism)), 4)}}
print("winner's curse:", res["winners_curse"], flush=True)

def z(v): return (v - v.mean()) / (v.std() + 1e-9)
def ll(model, X, yy):
    p = np.clip(model.predict_proba(X)[:, 1], 1e-9, 1 - 1e-9)
    return float(np.sum(yy * np.log(p) + (1 - yy) * np.log(1 - p)))
Xh = z(P["image_only"])[:, None]; Xc = z(P["cnv_only"])[:, None]
Xhc = np.hstack([Xh, Xc])
res["complementarity"] = {}
for name, Xbase, Xfull in [("cnv_adds_to_hist", Xh, Xhc), ("hist_adds_to_cnv", Xc, Xhc)]:
    m0 = LogisticRegression(penalty=None, max_iter=4000).fit(Xbase, y)
    m1 = LogisticRegression(penalty=None, max_iter=4000).fit(Xfull, y)
    lrt = 2 * (ll(m1, Xfull, y) - ll(m0, Xbase, y))
    p0 = m0.predict_proba(Xbase)[:, 1]; p1 = m1.predict_proba(Xfull)[:, 1]
    deltas = []
    for b in range(2000):
        idx = rng.randint(0, len(y), len(y))
        if y[idx].sum() in (0, len(y)): continue
        deltas.append(roc_auc_score(y[idx], p1[idx]) - roc_auc_score(y[idx], p0[idx]))
    res["complementarity"][name] = {
        "lrt_chi2": round(lrt, 3), "lrt_p": float(chi2.sf(lrt, 1)),
        "auc_base": round(float(roc_auc_score(y, p0)), 4),
        "auc_full": round(float(roc_auc_score(y, p1)), 4),
        "delta_auc_ci": [round(float(np.percentile(deltas, 2.5)), 4),
                         round(float(np.percentile(deltas, 97.5)), 4)]}
    print(name, res["complementarity"][name], flush=True)
json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=2)
print("wrote results.json")

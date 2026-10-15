"""2.39: selection-adjusted inference for the SWG headline (Astra A1).

The C1 claim (late_mean fusion beats histology, p_holm=0.0096) selected
late_mean among competing arm families on the same OOF data. Two remedies,
both on the frozen release OOF (no retraining):

1. Max-over-arms permutation test: statistic T = max_k [metric(arm_k) -
   metric(image_only)] over the candidate set; null from patient-level label
   permutations with predictions fixed (arm correlation preserved), so the
   null distribution of the BEST arm's advantage already includes selection.
   p = (1 + #{T_b >= T_obs}) / (B + 1). Primary metric AUC (matches C1);
   AUPRC secondary. Candidate sets: fusion-only, and all-but-histology
   (sensitivity).
2. Selection-honest effect size: bootstrap resamples select the best arm
   in-bag (by AUC vs image_only) and record that arm's delta on the
   OUT-OF-BAG patients — the debiased "delta you get after selecting".
"""
import glob, json, os
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

R = "/mnt/scratche/slow/fmlab/zuberi01/phd/barretts_retraining/barretts_training/analysis/chapter1_lgd2_final_pre_event_20260713_final/training_final_nested_cv_v1"
OUT = os.environ.get("OUTDIR", ".")
N_PERM = 20000
N_BOOT = 2000
HIST = "image_only"

fam_oof = {}
for d in sorted(glob.glob(os.path.join(R, "*"))):
    files = glob.glob(os.path.join(d, "fold*/outer_test_predictions.csv"))
    if not files: continue
    df = pd.concat([pd.read_csv(f) for f in files])
    fam_oof[os.path.basename(d)] = df.groupby("patient_id").agg(
        y=("y_true", "max"), p=("y_prob", "max"))
fams = sorted(fam_oof)
common = sorted(set.intersection(*(set(v.index) for v in fam_oof.values())))
y = fam_oof[fams[0]].loc[common, "y"].values.astype(int)
P = {f: fam_oof[f].loc[common, "p"].values for f in fams}
assert HIST in P, fams
unimodal = {HIST, "cnv_only"}
sets = {"fusion_arms": [f for f in fams if f not in unimodal],
        "all_but_histology": [f for f in fams if f != HIST]}
print(f"families={fams} patients={len(common)} pos={int(y.sum())}", flush=True)

METRICS = {"auc": roc_auc_score, "auprc": average_precision_score}
res = {"_meta": {"n_patients": len(common), "pos": int(y.sum()),
                 "families": fams, "comparator": HIST,
                 "n_perm": N_PERM, "n_boot": N_BOOT,
                 "candidate_sets": {k: v for k, v in sets.items()}}}

rng = np.random.RandomState(0)
perm_idx = [rng.permutation(len(y)) for _ in range(N_PERM)]
for mname, met in METRICS.items():
    m_obs = {f: met(y, P[f]) for f in fams}
    deltas_obs = {f: m_obs[f] - m_obs[HIST] for f in fams}
    # per-permutation metric for every family (predictions fixed, labels permuted)
    m_perm = {f: np.array([met(y[ix], P[f]) for ix in perm_idx]) for f in fams}
    d_perm = {f: m_perm[f] - m_perm[HIST] for f in fams}
    block = {"observed": {f: round(v, 4) for f, v in m_obs.items()},
             "observed_delta_vs_hist": {f: round(v, 4) for f, v in deltas_obs.items()}}
    for sname, cand in sets.items():
        t_obs = max(deltas_obs[f] for f in cand)
        best = max(cand, key=lambda f: deltas_obs[f])
        t_null = np.max(np.stack([d_perm[f] for f in cand]), axis=0)
        p_adj = (1 + int((t_null >= t_obs).sum())) / (N_PERM + 1)
        # unadjusted single-arm reference: same null, selected arm only
        p_single = (1 + int((d_perm[best] >= deltas_obs[best]).sum())) / (N_PERM + 1)
        block[sname] = {"selected_arm": best, "t_obs": round(t_obs, 4),
                        "p_selection_adjusted": round(p_adj, 5),
                        "p_selected_arm_unadjusted": round(p_single, 5),
                        "null_t_q95": round(float(np.quantile(t_null, 0.95)), 4)}
    res[mname] = block
    print(mname, {k: block[k] for k in sets}, flush=True)

# selection-honest effect size (AUC): select in-bag, evaluate out-of-bag
rng = np.random.RandomState(1)
oob_deltas, sel_count = [], {}
cand = sets["fusion_arms"]
b = 0
while b < N_BOOT:
    idx = rng.randint(0, len(y), len(y))
    oob = np.setdiff1d(np.arange(len(y)), idx)
    if y[idx].sum() in (0, len(y)) or y[oob].sum() in (0, len(oob)):
        continue
    b += 1
    ib = {f: roc_auc_score(y[idx], P[f][idx]) - roc_auc_score(y[idx], P[HIST][idx])
          for f in cand}
    w = max(ib, key=ib.get)
    sel_count[w] = sel_count.get(w, 0) + 1
    oob_deltas.append(roc_auc_score(y[oob], P[w][oob]) - roc_auc_score(y[oob], P[HIST][oob]))
oob_deltas = np.array(oob_deltas)
naive = res["auc"]["observed_delta_vs_hist"]["late_mean"] if "late_mean" in fams else None
res["selection_honest_delta_auc"] = {
    "mean_oob_delta_of_selected_arm": round(float(oob_deltas.mean()), 4),
    "ci": [round(float(np.percentile(oob_deltas, 2.5)), 4),
           round(float(np.percentile(oob_deltas, 97.5)), 4)],
    "frac_oob_delta_gt_0": round(float((oob_deltas > 0).mean()), 4),
    "in_bag_selection_rate": {f: round(c / N_BOOT, 3) for f, c in sorted(sel_count.items())},
    "naive_full_sample_late_mean_delta": naive}
json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=2)
print(json.dumps(res, indent=2))

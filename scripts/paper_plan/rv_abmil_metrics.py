"""R13 of docs/paper_review_round1_analyses.md (baa47ab): metrics for N-IMG (rv_abmil_train.py) against the stored linear kv_cv.R predictions.
All pre-event samples (horizons/samples.csv pre). Fold-stratified per-sample AUROC (case-control pairs within the same outer fold of a repeat,
averaged over repeats; as rs_strata.py fs_auc). N-LATE = (N-IMG + L-CNV)/2 and L-LATE = (L-CNV + L-IMG)/2 per repeat (L-CNV = cfg 0, their
matrix; L-IMG = cfg 2). Patient bootstrap: 2,000 draws, RandomState(0), draws without both classes redrawn, resampled patients keep their
folds, predictions fixed. SMOKE=1: only the N-IMG fold files present (rep x fold), output to the cluster row folder, not results/.
Out: results/paper_final/review_round1/r13_abmil.json."""
import os, glob, json, numpy as np, pandas as pd
from scipy.stats import rankdata
T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"; M = T + "/feasibility/paper_plan/killcoyne_mm"; HZ = M + "/horizons"; A = M + "/review_round1/abmil"
OUT = T + "/results/paper_final/review_round1"; NB = 2000; REPS = list(range(1, 11)); SMOKE = os.environ.get("SMOKE") == "1"
r3 = lambda x: None if x is None or not np.isfinite(x) else round(float(x), 3)
SM = pd.read_csv(HZ + "/samples.csv", dtype={"Sample": str, "Patient": str}); n = len(SM); ids = SM.Sample.values
y = SM.y.values.astype(int); pat = SM.Patient.values; pre = SM.pre.astype(bool).values
def per_rep(D, col="pred"):
    P = D.pivot_table(index="Sample", columns="rep", values=col).reindex(ids); Fo = D.pivot_table(index="Sample", columns="rep", values="fold").reindex(ids); return P, Fo
kv = lambda c: pd.concat([pd.read_csv(f, dtype={"Sample": str}) for f in sorted(glob.glob(f"{M}/cv/preds/cfg_{c:02d}_rep_*.csv"))])
Pc, FO = per_rep(kv(0)); Pi, F2 = per_rep(kv(2)); assert Pc.notna().all().all() and Pi.notna().all().all() and (FO.values == F2.values).all()
FO = FO[REPS].values.astype(int)
nf = sorted(glob.glob(f"{A}/nimg_rep_*_fold_*.csv")); N = pd.concat([pd.read_csv(f, dtype={"Sample": str, "Patient": str}) for f in nf])
assert not N.duplicated(["Sample", "rep"]).any() and (N.y.values == y[pd.Series(np.arange(n), index=ids)[N.Sample].values]).all()
Pn, Fn = per_rep(N); Pn = Pn.reindex(columns=REPS); Fn = Fn.reindex(columns=REPS)
have = Pn.notna().values                                                   # (n x 10) which sample x repeat has an N-IMG prediction
assert (Fn.values[have] == FO[have]).all()
if not SMOKE: assert have.all() and len(nf) == 100, f"{len(nf)} N-IMG fold files, {(~have).sum()} missing sample x repeat cells"
X = {"L-CNV": Pc[REPS].values, "L-IMG": Pi[REPS].values, "N-IMG": Pn.values}; X["L-LATE"] = (X["L-CNV"] + X["L-IMG"]) / 2; X["N-LATE"] = (X["N-IMG"] + X["L-CNV"]) / 2
MODELS = ["N-IMG", "N-LATE", "L-IMG", "L-LATE", "L-CNV"]; PAIRS = [("N-IMG", "L-IMG"), ("N-LATE", "L-LATE")]
def fs_auc(m, rr):
    yr = y[rr]; out = []
    for j in range(10):
        f = FO[rr, j]; x = X[m][rr, j]; h = have[rr, j]; num = den = 0.0
        for k in np.unique(f[h]):
            mk = (f == k) & h; yk = yr[mk]; n1 = yk.sum(); n0 = len(yk) - n1
            if n1 == 0 or n0 == 0: continue
            rk = rankdata(x[mk]); num += rk[yk == 1].sum() - n1 * (n1 + 1) / 2; den += n1 * n0
        if den: out.append(num / den)
    return float(np.mean(out)) if out else np.nan
rows = np.where(pre)[0]
pp = pat[rows]; up = np.unique(pp); ro = {q: rows[pp == q] for q in up}; rng = np.random.RandomState(0); boots = []
while len(boots) < NB:
    b = np.concatenate([ro[up[i]] for i in rng.choice(len(up), len(up))])
    if 0 < y[b].sum() < len(b): boots.append(b)
obs = {m: fs_auc(m, rows) for m in MODELS}; BS = {m: np.array([fs_auc(m, b) for b in boots]) for m in MODELS}
pct = lambda a: [r3(np.nanpercentile(a, 2.5)), r3(np.nanpercentile(a, 97.5))]
p2 = lambda a: round(float(min(1.0, 2 * min(np.nanmean(a <= 0), np.nanmean(a >= 0)))), 4)
res = {"item": "R13", "spec": "docs/paper_review_round1_analyses.md @ baa47ab", "population": "pre-event samples", "smoke": SMOKE,
       "n_samples": int(len(rows)), "n_patients": int(len(up)), "n_progressor_samples": int(y[rows].sum()), "n_progressor_patients": int(len(set(pat[rows][y[rows] == 1]))),
       "auroc_type": "fold-stratified per-sample (pairs within the same outer fold of a repeat, mean over repeats)",
       "bootstrap": {"draws": NB, "seed": "RandomState(0)", "unit": "patient", "redraw_if_one_class": True, "folds_kept": True, "predictions_fixed": True},
       "auroc": {m: {"value": r3(obs[m]), "ci95": pct(BS[m])} for m in MODELS},
       "paired": {f"{a} - {b}": {"delta": r3(obs[a] - obs[b]), "ci95": pct(BS[a] - BS[b]), "p_bootstrap_unadjusted": p2(BS[a] - BS[b])} for a, b in PAIRS}}
def rep_auc(m, j):   # one repeat's fold-stratified AUROC on the pre-event rows (spread across repeats, descriptive)
    yr = y[rows]; f = FO[rows, j]; x = X[m][rows, j]; hh = have[rows, j]; num = den = 0.0
    for k in np.unique(f[hh]):
        mk = (f == k) & hh; yk = yr[mk]; n1 = yk.sum(); n0 = len(yk) - n1
        if n1 and n0: rk = rankdata(x[mk]); num += rk[yk == 1].sum() - n1 * (n1 + 1) / 2; den += n1 * n0
    return num / den if den else np.nan
res["auroc_by_repeat_range"] = {m: [r3(np.nanmin(v)), r3(np.nanmax(v))] for m in MODELS for v in [[rep_auc(m, j) for j in range(10)]]}
E = N.drop_duplicates(["rep", "fold"])
res["training"] = {"fold_fits": int(len(E)), "seed": "= repeat", "device": "cpu (as the stored km_neural.py runs)",
                   "best_epoch_mean": r3(E.best_epoch.mean()), "best_epoch_range": [int(E.best_epoch.min()), int(E.best_epoch.max())],
                   "epochs_run_mean": r3(E.epochs_run.mean()), "n_train_range": [int(E.n_train.min()), int(E.n_train.max())],
                   "val_patients_range": [int(E.val_patients.min()), int(E.val_patients.max())], "val_progressor_patients_range": [int(E.val_pos_patients.min()), int(E.val_pos_patients.max())],
                   "fits_with_no_progressor_in_validation": int((E.val_pos_patients == 0).sum())}
dest = (A + "/smoke_r13_abmil.json") if SMOKE else (OUT + "/r13_abmil.json"); os.makedirs(os.path.dirname(dest), exist_ok=True)
json.dump(res, open(dest, "w"), indent=1); print(json.dumps(res, indent=1)); print("R13 METRICS DONE", dest)

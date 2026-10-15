"""Horizon answers (docs/paper_horizon_answers.md @ 81473df): L-CLIN and L-CLIN without grade under the stratified 10-fold x 10 CV.
L2 logistic (scikit-learn, C = 1, no tuning) on [ordinal sample grade, age at BE diagnosis, sex (M = 1), Prague M with training-fold median fill + missing
indicator], z-scored on the training rows; label = sheet Status (ever-progression). Outer folds read from kv_cv.R predictions (cfg_00, per repeat);
inner folds read from hz_fit.R inner files (inner_fold), so the inner out-of-fold predictions use the same splits as the other arms.
Row-level outputs (cluster only): feasibility/paper_plan/killcoyne_mm/horizon_answers/clin_{outer,inner}.csv"""
import glob, os, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"; M = T + "/feasibility/paper_plan/killcoyne_mm"; HZ = M + "/horizons"; OUT = M + "/horizon_answers"; os.makedirs(OUT, exist_ok=True)
S = "/mnt/scratche/fast/fmlab/datasets/imaging/SWGCohort"
SM = pd.read_csv(HZ + "/samples.csv", dtype={"Sample": str, "Patient": str}); n = len(SM)
dm = pd.read_csv(S + "/Demographics_full.csv", dtype=str); dmi = {}
for _, r in dm.iterrows():
    for c in ("Study Number", "Alternate Study Number"):
        v = r[c].strip() if isinstance(r[c], str) else ""
        if v and v not in dmi: dmi[v] = r
num = lambda v: pd.to_numeric(pd.Series([v]), errors="coerce").iloc[0]
grade = SM.Pathology.map({"NDBE": 0, "ID": 1, "LGD": 2, "HGD": 3, "IMC": 4}).astype(float).values
agedx = np.array([num(dmi[s]["Age at diagnosis"]) if s in dmi else np.nan for s in SM.study_number])
sex = np.array([1.0 if s in dmi and str(dmi[s]["Sex"]).strip() == "M" else 0.0 if s in dmi and str(dmi[s]["Sex"]).strip() == "F" else np.nan for s in SM.study_number])
pm = np.array([num(dmi[s]["Maximal"]) if s in dmi else np.nan for s in SM.study_number])
assert np.isfinite(grade).all() and np.isfinite(agedx).all() and np.isfinite(sex).all(), "grade/age/sex must be complete"
y = SM.y.values.astype(int)
def design(tr, rows, with_grade):
    med = np.nanmedian(pm[tr]); m = np.where(np.isnan(pm), med, pm); miss = np.isnan(pm).astype(float)
    X = np.column_stack(([grade] if with_grade else []) + [agedx, sex, m, miss]); mu = X[tr].mean(0); sd = X[tr].std(0); sd[sd == 0] = 1
    return (X[rows] - mu) / sd, (X[tr] - mu) / sd
def fitpred(tr, te, with_grade):
    Xte, Xtr = design(tr, te, with_grade); return LogisticRegression(C=1.0, penalty="l2", max_iter=2000).fit(Xtr, y[tr]).predict_proba(Xte)[:, 1]
ix = pd.Series(np.arange(n), index=SM.Sample)
outer, inner = [], []
for f in sorted(glob.glob(M + "/cv/preds/cfg_00_rep_*.csv")):
    D = pd.read_csv(f, dtype={"Sample": str}); r = int(D.rep.iloc[0]); fold = pd.Series(D.fold.values, index=ix[D.Sample].values).reindex(np.arange(n)).values
    for k in range(1, 11):
        te = np.where(fold == k)[0]; tr = np.where(fold != k)[0]
        for wg, nm in ((True, "clin"), (False, "clin_nograde")):
            outer.append(pd.DataFrame({"Sample": SM.Sample.values[te], "rep": r, "fold": k, "model": nm, "pred": fitpred(tr, te, wg)}))
        I = pd.read_csv(f"{HZ}/inner/cnv_their_rep_{r:02d}_fold_{k:02d}.csv", dtype={"Sample": str}); irow = ix[I.Sample].values; ifo = I.inner_fold.values
        assert set(irow) == set(tr), (r, k)
        for j in range(1, 6):
            tei = irow[ifo == j]; tri = irow[ifo != j]
            for wg, nm in ((True, "clin"), (False, "clin_nograde")):
                inner.append(pd.DataFrame({"Sample": SM.Sample.values[tei], "rep": r, "fold": k, "inner_fold": j, "model": nm, "pred": fitpred(tri, tei, wg)}))
    print("rep", r, flush=True)
pd.concat(outer).to_csv(OUT + "/clin_outer.csv", index=False); pd.concat(inner).to_csv(OUT + "/clin_inner.csv", index=False)
print("HA CLIN DONE", "prague_M_missing", int(np.isnan(pm).sum()), "of", n)

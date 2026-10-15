"""Item 5 metrics (docs/paper_triage_robustness.md @ 32653f1): within-scanner fold-stratified AUROC (mean over 10 repeats), cross-scanner plain AUROC,
patient-bootstrap CIs (2,000 draws, seed 0, predictions fixed); triage bands on the test scanner with t_low (95% sens) / c* (90% spec) chosen on the
training scanner's inner OOF predictions. Aggregates -> results/paper_final/triage_robustness/scanner.json."""
import os, glob, json, numpy as np, pandas as pd
from scipy.stats import rankdata
from tb_common import OUT, ROW, r3, pct, thr_sens, c_star
D = ROW + "/scanner"; os.makedirs(OUT, exist_ok=True); rng0 = 0
def auc(yy, s):
    r = rankdata(s); n1 = yy.sum(); n0 = len(yy) - n1; return (r[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if 0 < n1 < len(yy) else np.nan
def boots(df, nb=2000):
    up = np.unique(df.Patient); ro = {q: np.where(df.Patient.values == q)[0] for q in up}; rng = np.random.RandomState(0); out = []
    while len(out) < nb:
        b = np.concatenate([ro[up[i]] for i in rng.choice(len(up), len(up))])
        if 0 < df.y.values[b].sum() < len(b): out.append(b)
    return out
res = {"prespec": "32653f1", "within": {}, "cross": {}}
for sc in ("s39", "s10"):
    W = pd.concat([pd.read_csv(f, dtype={"Sample": str, "Patient": str}) for f in sorted(glob.glob(f"{D}/within_{sc}_*.csv"))]); assert W.rep.nunique() == 10
    P = W.pivot_table(index="Sample", columns="rep", values="pred"); F = W.pivot_table(index="Sample", columns="rep", values="fold").astype(int); S = W.drop_duplicates("Sample").set_index("Sample").reindex(P.index)
    def fs(b):
        yy = S.y.values[b]; out = []
        for j in range(10):
            f = F.values[b, j]; x = P.values[b, j]; num = den = 0.0
            for k in range(1, 11):
                m = f == k; yk = yy[m]; n1 = yk.sum(); n0 = len(yk) - n1
                if n1 == 0 or n0 == 0: continue
                rk = rankdata(x[m]); num += rk[yk == 1].sum() - n1 * (n1 + 1) / 2; den += n1 * n0
            if den: out.append(num / den)
        return np.mean(out) if out else np.nan
    bs = boots(S.reset_index()); allr = np.arange(len(S))
    res["within"][sc] = {"auroc_fold_stratified": r3(fs(allr)), "ci95": pct([fs(b) for b in bs]), "n_samples": int(len(S)), "n_patients": int(S.Patient.nunique()), "n_P_patients": int(S[S.y == 1].Patient.nunique())}
for f in sorted(glob.glob(f"{D}/cross_*.csv")):
    job = os.path.basename(f)[:-4]; X = pd.read_csv(f, dtype={"Sample": str, "Patient": str})
    if (X.role == "not_estimable").any():
        r = X.iloc[0]; res["cross"][job] = {"estimable": False, **{k: int(r[k]) for k in ("n_train", "n_train_patients", "n_train_P_patients", "n_test", "n_test_patients", "n_test_P_patients")}}; continue
    te = X[X.role == "test"].reset_index(drop=True); inn = X[X.role == "inner"]; bs = boots(te)
    yi = inn.y.values == 1; tl = thr_sens(inn.pred.values[yi], 0.95); cs = c_star(inn.pred.values[~yi], 0.90)
    def bands(df): s = df.pred.values; b = np.where(s < tl, 0, np.where(s > cs, 2, 1)) if tl <= cs else np.where(s > cs, 2, 0); return {nm: {"all": r3(np.mean(b == g)), "progressor": r3(np.mean(b[df.y.values == 1] == g)), "non_progressor": r3(np.mean(b[df.y.values == 0] == g))} for g, nm in ((0, "low"), (1, "mid"), (2, "high"))}
    res["cross"][job] = {"estimable": True, "auroc": r3(auc(te.y.values, te.pred.values)), "ci95": pct([auc(te.y.values[b], te.pred.values[b]) for b in bs]),
                         "n_test": int(len(te)), "n_test_patients": int(te.Patient.nunique()), "n_test_P_patients": int(te[te.y == 1].Patient.nunique()),
                         "n_train": int(inn.Sample.nunique()), "n_train_patients": int(inn.Patient.nunique()), "n_train_P_patients": int(inn[inn.y == 1].Patient.nunique()),
                         "inner_auroc_training_scanner": r3(auc(inn.y.values, inn.pred.values)), "cutoffs": {"t_low": r3(tl), "c_star": r3(cs), "no_middle_band": bool(tl > cs)},
                         "bands_test_scanner": bands(te), "bands_training_scanner_inner_oof": bands(inn)}
json.dump(res, open(f"{OUT}/scanner.json", "w"), indent=1); print("TB SCANNER METRICS DONE", flush=True)

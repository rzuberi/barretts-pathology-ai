"""Round 3 post-hoc addition (not in the cf244e1 pre-specification; reported beside the pre-specified result): the CNV-QC
stratum probe of R1 without the read count and its missing indicator, because read counts exist only for the discovery
sheet and the indicator identifies the stratum by construction. Same folds, model and non-progressor set as R1."""
import json, os, numpy as np, pandas as pd, warnings
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression; from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")
F = "/mnt/scratche/slow/fmlab/zuberi01/phd/barretts_retraining/barretts_training/analysis/chapter1_lgd2_final_pre_event_20260713_final"; R = F + "/training_final_nested_cv_v1"; T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"; ROW = T + "/feasibility/paper_plan"; AGG = os.environ.get("OUTDIR", T + "/results/paper_plan")
def auc(y, s):
    y = np.asarray(y).astype(int); r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1; return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if 0 < n1 < len(y) else float("nan")
man = pd.read_csv(F + "/training_manifest.csv", dtype=str).set_index("sample_id"); ids = list(man.index); pid = man.patient_id.values; folds = man.fold_id_rep01.astype(int).values
pt = pd.read_csv(ROW + "/round3_patient_table.csv", dtype={"patient_id": str}).set_index("patient_id"); pats = pt.index.values; y = pt.y.values.astype(int); lab = (pt.stratum2.values == "discovery").astype(int); pfold = pd.Series(folds, index=pid).groupby(level=0).first().reindex(pats).values
qc = pd.read_csv(T + "/feasibility/closeout/swg_cnv_qc.csv"); cxs = pd.read_csv(F + "/feature_views/cnv/cx.csv", dtype=str).set_index("sample_id").reindex(ids); q = qc.set_index("cnv_id").reindex(cxs.cnv_id.values)
Xrow = np.column_stack([q.noise_mapd.values, q.n_segments.values, q.frac_altered_0p15.values, pd.to_numeric(cxs.cx, errors="coerce").values]); Xp = pd.DataFrame(Xrow, index=pid).groupby(level=0).mean().reindex(pats).values
inner = {k: pd.read_csv(f"{R}/image_only/fold{k}/inner_fold_assignments.csv", dtype=str).set_index("patient_id").inner_fold.astype(int) for k in range(1, 6)}
neg = y == 0; idx = np.where(neg)[0]; pr_ = np.full(len(pats), np.nan)
for k in range(1, 6):
    tr = idx[pfold[idx] != k]; te = idx[pfold[idx] == k]; sc = StandardScaler().fit(Xp[tr]); Xtr, Xte = sc.transform(Xp[tr]), sc.transform(Xp[te]); inn = inner[k].reindex(pats[tr]).values; best, bestC = -1, 1.0
    for C in [0.01, 0.1, 1.0, 10.0]:
        pv = np.zeros(len(tr))
        for j in np.unique(inn):
            v = inn == j
            if len(set(lab[tr][~v])) < 2 or v.sum() == 0: continue
            pv[v] = LogisticRegression(C=C, max_iter=5000).fit(Xtr[~v], lab[tr][~v]).predict_proba(Xtr[v])[:, 1]
        a = auc(lab[tr], pv)
        if a > best: best, bestC = a, C
    pr_[te] = LogisticRegression(C=bestC, max_iter=5000).fit(Xtr, lab[tr]).predict_proba(Xte)[:, 1]
ok = ~np.isnan(pr_); yy = lab[ok]; rng = np.random.RandomState(0); B = []
while len(B) < 2000:
    s = rng.choice(len(yy), len(yy))
    if len(set(yy[s])) > 1: B.append(auc(yy[s], pr_[ok][s]))
uni = {n: round(float(auc(yy, Xp[ok][:, i])), 3) for i, n in enumerate(["noise_mapd", "n_segments", "frac_altered_0p15", "cx"])}
res = {"cnv_qc_without_reads": {"features": ["noise_mapd", "n_segments", "frac_altered_0p15", "cx"], "n_nonprogressors": int(ok.sum()), "discovery": int(yy.sum()), "validation": int((1 - yy).sum()), "auroc": round(float(auc(yy, pr_[ok])), 3), "ci": [round(float(np.percentile(B, 2.5)), 3), round(float(np.percentile(B, 97.5)), 3)], "univariate_auroc_raw_direction": uni}, "note": "post hoc, added after seeing the pre-specified QC probe reach 1.0 through the read-count missing indicator"}
json.dump(res, open(AGG + "/round3_extra.json", "w"), indent=1); print(json.dumps(res, indent=1))

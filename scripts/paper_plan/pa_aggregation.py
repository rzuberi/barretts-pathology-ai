"""Aggregation sensitivity (pre-specified in docs/paper_plan_aggregation.md @ d1958ea): mean-over-rows vs max-over-rows
patient scores for every whiteboard arm, discovery and stratified, both endpoints; K3 probes with max aggregation.
Output: results/paper_final/aggregation_sensitivity.json."""
import glob, json, os, warnings, numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression; from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")
F = "/mnt/scratche/slow/fmlab/zuberi01/phd/barretts_retraining/barretts_training/analysis/chapter1_lgd2_final_pre_event_20260713_final"; R = F + "/training_final_nested_cv_v1"; T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"; ROW = T + "/feasibility/paper_plan"; AGG = os.environ.get("OUTDIR", T + "/results/paper_final"); os.makedirs(AGG, exist_ok=True); NB = 2000; SEED = 0
def auc(y, s):
    y = np.asarray(y).astype(int); r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1; return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if 0 < n1 < len(y) else float("nan")
def r3(x): return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 3)
man = pd.read_csv(F + "/training_manifest.csv", dtype=str).set_index("sample_id"); coh = pd.read_csv(F + "/pre_event_cohort.csv", dtype=str).set_index("SampleID").loc[man.index]; ids = list(man.index); pid = man.patient_id.values; folds = man.fold_id_rep01.astype(int).values; yrow = man.y_progressor.astype(int).values
def oof(fam): return pd.concat([pd.read_csv(f, dtype={"sample_id": str}) for f in glob.glob(f"{R}/{fam}/fold*/outer_test_predictions.csv")]).set_index("sample_id").y_prob.reindex(ids).values
ROWS = {f: oof(f) for f in ["cnv_only", "image_only", "early_fusion", "intermediate_fusion", "late_mean", "coattention_fusion", "late_stack_logit"]}; ROWS["cnv_km"] = pd.read_csv(ROW + "/f1_cnv_km_oof.csv", dtype={"sample_id": str}).set_index("sample_id").cnv_km.reindex(ids).values
inner = {k: pd.read_csv(f"{R}/image_only/fold{k}/inner_fold_assignments.csv", dtype=str).set_index("patient_id").inner_fold.astype(int) for k in range(1, 6)}
def nested_logistic(X, yv, fl, pids, Cs=(0.01, 0.1, 1.0, 10.0)):
    p = np.zeros(len(yv))
    for k in np.unique(fl):
        te = fl == k; tr = ~te; mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9; Xtr = (X[tr] - mu) / sd; inn = inner[int(k)].reindex(pids[tr]).values; best, bestC = -1, 1.0
        for C in Cs:
            pv = np.zeros(tr.sum())
            for j in np.unique(inn): v = inn == j; pv[v] = LogisticRegression(C=C, max_iter=5000).fit(Xtr[~v], yv[tr][~v]).predict_proba(Xtr[v])[:, 1]
            g = pd.DataFrame({"p": pids[tr], "s": pv, "y": yv[tr]}).groupby("p"); a = auc(g.y.max().values, g.s.max().values)
            if a > best: best, bestC = a, C
        p[te] = LogisticRegression(C=bestC, max_iter=5000).fit(Xtr, yv[tr]).predict_proba((X[te] - mu) / sd)[:, 1]
    return p
X2 = np.column_stack([pd.to_numeric(coh.Label, errors="coerce").fillna(0).values, pd.to_numeric(coh.MaxPathologySoFar, errors="coerce").fillna(0).values]).astype(float); ROWS["C2_grade_maxsofar"] = nested_logistic(X2, yrow, folds, pid)
pt = pd.read_csv(ROW + "/round3_patient_table.csv", dtype={"patient_id": str}).set_index("patient_id"); pats = pt.index.values; y = pt.y.values.astype(int); yH = pt.y_hgd.values.astype(int); strat2 = pt.stratum2.values; disc = strat2 == "discovery"
def agg(s, how): g = pd.Series(s, index=pid).groupby(level=0); return (g.max() if how == "max" else g.mean()).reindex(pats).values
def strat_auc(yy, s, g2):
    num = den = 0.0
    for l in np.unique(g2):
        m = g2 == l; n1 = yy[m].sum(); n0 = m.sum() - n1
        if n1 > 0 and n0 > 0: num += n1 * n0 * auc(yy[m], s[m]); den += n1 * n0
    return num / den
def bidx(yy, strata=None):
    rng = np.random.RandomState(SEED); out = []
    while len(out) < NB:
        s = rng.choice(len(yy), len(yy)) if strata is None else np.concatenate([rng.choice(np.where(strata == l)[0], (strata == l).sum()) for l in np.unique(strata)])
        if (len(set(yy[s])) > 1) and (strata is None or all(len(set(yy[s][strata[s] == l])) > 1 for l in np.unique(strata))): out.append(s)
    return out
RES = {"_spec": "docs/paper_plan_aggregation.md @ d1958ea", "arms": {}, "max_check_vs_round3_table": {}}
for a, s in ROWS.items(): RES["max_check_vs_round3_table"][a] = r3(np.max(np.abs(agg(s, "max") - pt[a].values))) if a in pt.columns else "not in table"
B = {("discovery", "LGD2plus"): bidx(y[disc]), ("discovery", "E_HGD"): bidx(yH[disc]), ("stratified", "LGD2plus"): bidx(y, strat2), ("stratified", "E_HGD"): bidx(yH, strat2)}
for a, s in ROWS.items():
    smax, smean = agg(s, "max"), agg(s, "mean"); RES["arms"][a] = {}
    for pop in ["discovery", "stratified"]:
        for ep, yy in [("LGD2plus", y), ("E_HGD", yH)]:
            if pop == "discovery": m = disc; f = lambda yv, sv, g=None: auc(yv, sv); yy2, a1, a2, g2 = yy[m], smax[m], smean[m], None
            else: f = lambda yv, sv, g: strat_auc(yv, sv, g); yy2, a1, a2, g2 = yy, smax, smean, strat2
            Bs = B[(pop, ep)]; v = [f(yy2[q], a2[q], g2[q] if g2 is not None else None) - f(yy2[q], a1[q], g2[q] if g2 is not None else None) for q in Bs]
            RES["arms"][a][f"{pop}__{ep}"] = {"n": int(len(yy2)), "events": int(yy2.sum()), "auroc_max": r3(f(yy2, a1, g2)), "auroc_mean": r3(f(yy2, a2, g2)), "delta_mean_minus_max": r3(f(yy2, a2, g2) - f(yy2, a1, g2)), "ci": [r3(np.percentile(v, 2.5)), r3(np.percentile(v, 97.5))]}
    print("arm", a, flush=True)
# K3 probes with max aggregation
Lat = T + "/feasibility/paper_plan/latent"; pfold = pd.Series(folds, index=pid).groupby(level=0).first().reindex(pats).values; Bd = bidx(y[disc]); Ba = bidx(y)
def pagg(M, how): g = pd.DataFrame(M, index=pid).groupby(level=0); return (g.max() if how == "max" else g.mean()).reindex(pats).values
def probe_patient(E, how):
    pr_ = np.zeros(len(pats))
    for k in range(1, 6):
        Pm = pagg(E[k], how); tr = pfold != k; te = ~tr; sc = StandardScaler().fit(Pm[tr]); Xtr, Xte = sc.transform(Pm[tr]), sc.transform(Pm[te]); ytr = y[tr]; inn = inner[k].reindex(pats[tr]).values; best, bestC = -1, 1.0
        for C in [0.01, 0.1, 1.0, 10.0]:
            pv = np.zeros(tr.sum())
            for j in np.unique(inn): v = inn == j; pv[v] = LogisticRegression(C=C, max_iter=5000).fit(Xtr[~v], ytr[~v]).predict_proba(Xtr[v])[:, 1]
            a_ = auc(ytr, pv)
            if a_ > best: best, bestC = a_, C
        pr_[te] = LogisticRegression(C=bestC, max_iter=5000).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
    return pr_
def probe_rows_max(E):
    pr_ = np.zeros(len(ids))
    for k in range(1, 6):
        M = E[k]; tr = folds != k; te = ~tr; sc = StandardScaler().fit(M[tr]); Xtr, Xte = sc.transform(M[tr]), sc.transform(M[te]); inn = inner[k].reindex(pid[tr]).values; best, bestC = -1, 1.0
        for C in [0.01, 0.1, 1.0, 10.0]:
            pv = np.zeros(tr.sum())
            for j in np.unique(inn): v = inn == j; pv[v] = LogisticRegression(C=C, max_iter=5000).fit(Xtr[~v], yrow[tr][~v]).predict_proba(Xtr[v])[:, 1]
            g = pd.DataFrame({"p": pid[tr], "s": pv, "y": yrow[tr]}).groupby("p"); a_ = auc(g.y.max().values, g.s.max().values)
            if a_ > best: best, bestC = a_, C
        pr_[te] = LogisticRegression(C=bestC, max_iter=5000).fit(Xtr, yrow[tr]).predict_proba(Xte)[:, 1]
    return agg(pr_, "max")
RES["k3_probes"] = {}
for fam in ["image_only", "cnv_only", "early_fusion", "intermediate_fusion", "coattention_fusion"]:
    E = {k: np.load(f"{Lat}/emb_{fam}_fold{k}.npy") for k in range(1, 6)}; pm, px, prm = probe_patient(E, "mean"), probe_patient(E, "max"), probe_rows_max(E); out = {}
    for pop, m, Bs in [("discovery", disc, Bd), ("all_150", np.ones(len(pats), bool), Ba)]:
        yy = y[m]; ci = lambda a_, b_: [r3(np.percentile([auc(yy[q], a_[q]) - auc(yy[q], b_[q]) for q in Bs], 2.5)), r3(np.percentile([auc(yy[q], a_[q]) - auc(yy[q], b_[q]) for q in Bs], 97.5))]
        out[pop] = {"n": int(m.sum()), "events": int(yy.sum()), "probe_mean_repr_K3": r3(auc(yy, pm[m])), "probe_max_repr": r3(auc(yy, px[m])), "delta_maxrepr_minus_mean": [r3(auc(yy, px[m]) - auc(yy, pm[m]))] + ci(px[m], pm[m]), "row_probe_max_over_rows": r3(auc(yy, prm[m])), "delta_rowmax_minus_mean": [r3(auc(yy, prm[m]) - auc(yy, pm[m]))] + ci(prm[m], pm[m])}
    RES["k3_probes"][fam] = out; print("probe", fam, flush=True)
json.dump(RES, open(AGG + "/aggregation_sensitivity.json", "w"), indent=1); print("AGG DONE", flush=True)

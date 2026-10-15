"""Round 3 R1, R2 (endpoints), R3, R4, R5, R6, pre-specified in docs/paper_plan_round3.md @ cf244e1. Frozen release only.
Inputs: follow-up patient scores/strata (feasibility/paper_plan), release tables, image embeddings (pp_latent), closeout QC.
Output: results/paper_plan/round3_main.json, figs/km_v3/."""
import glob, json, os, warnings, numpy as np, pandas as pd, h5py
from scipy.stats import rankdata, spearmanr, mannwhitneyu, chi2
from sklearn.linear_model import LogisticRegression; from sklearn.decomposition import PCA; from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm; from statsmodels.stats.contingency_tables import StratifiedTable
from lifelines import KaplanMeierFitter, CoxPHFitter; from lifelines.statistics import multivariate_logrank_test
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
F = "/mnt/scratche/slow/fmlab/zuberi01/phd/barretts_retraining/barretts_training/analysis/chapter1_lgd2_final_pre_event_20260713_final"; R = F + "/training_final_nested_cv_v1"
T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"; S = "/mnt/scratche/fast/fmlab/datasets/imaging/SWGCohort"; E = "/mnt/scratche/slow/fmlab/zuberi01/barretts_db_export"; ROW = T + "/feasibility/paper_plan"; AGG = os.environ.get("OUTDIR", T + "/results/paper_plan"); FIG = T + "/results/paper_plan/figs/km_v3"; os.makedirs(AGG, exist_ok=True); os.makedirs(FIG, exist_ok=True)
NB = 2000; SEED = 0
def auc(y, s):
    y = np.asarray(y).astype(int); r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1; return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if 0 < n1 < len(y) else float("nan")
def r3(x): return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 3)
def pf(p): return None if p is None or (isinstance(p, float) and np.isnan(p)) else ("<0.001" if p < 0.001 else round(float(p), 3))
def logit(p): p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6); return np.log(p / (1 - p))
RES = {"_spec": "docs/paper_plan_round3.md @ cf244e1"}
# ------------------------------------------------------------- data
man = pd.read_csv(F + "/training_manifest.csv", dtype=str).set_index("sample_id"); coh = pd.read_csv(F + "/pre_event_cohort.csv", dtype=str).set_index("SampleID").loc[man.index]
man["y"] = man.y_progressor.astype(int); man["fold"] = man.fold_id_rep01.astype(int); man["date"] = pd.to_datetime(coh.Date); man["next_label"] = pd.to_numeric(coh.NextBiopsyLabel, errors="coerce"); man["nbd"] = pd.to_datetime(coh.NextBiopsyDate, errors="coerce"); man["grade"] = pd.to_numeric(coh.Label, errors="coerce").fillna(0); man["maxsofar"] = pd.to_numeric(coh.MaxPathologySoFar, errors="coerce").fillna(0)
ids = list(man.index); pid = man.patient_id.values; folds = man.fold.values
sp = pd.read_csv(ROW + "/followup_patient_scores.csv", dtype={"patient_id": str}).set_index("patient_id"); pats = sp.index.values; y = sp.y.values.astype(int); st = pd.read_csv(ROW + "/f2_strata.csv", dtype=str).set_index("patient_id").reindex(pats)
disc = st.stratum.str.startswith("discovery").values; strat2 = np.where(disc, "discovery", "validation"); strat3 = st.stratum.values; pfold = pd.Series(folds, index=pid).groupby(level=0).first().reindex(pats).values
ARMS = ["C1_grade", "C2_grade_maxsofar", "C3_plus_streak", "C4_plus_surveillance_3a", "cnv_only", "cnv_km", "image_only", "early_fusion", "intermediate_fusion", "late_mean", "late_mean_km", "coattention_fusion", "late_stack_logit"]; P = {a: sp[a].values for a in ARMS}
# C2 + modality fold-z combos at patient level (recompute row-level z-means as in pf_main)
def oof(fam): return pd.concat([pd.read_csv(f, dtype={"sample_id": str}) for f in glob.glob(f"{R}/{fam}/fold*/outer_test_predictions.csv")]).set_index("sample_id").y_prob.reindex(ids).values
f1 = pd.read_csv(ROW + "/f1_cnv_km_oof.csv", dtype={"sample_id": str}).set_index("sample_id").reindex(ids); rowsc = {"image_only": oof("image_only"), "cnv_only": oof("cnv_only"), "late_mean": oof("late_mean"), "cnv_km": f1.cnv_km.values}
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
c2_row = nested_logistic(man[["grade", "maxsofar"]].values.astype(float), man.y.values, folds, pid)
def zf(v):
    v = np.asarray(v, float); o = np.zeros(len(v))
    for f_ in np.unique(folds): te = folds == f_; o[te] = (v[te] - v[te].mean()) / (v[te].std() + 1e-9)
    return o
def pmax(s): return pd.Series(s, index=pid).groupby(level=0).max().reindex(pats).values
COMBO = {"C2+image": ["image_only"], "C2+cnv_only": ["cnv_only"], "C2+cnv_km": ["cnv_km"], "C2+late_mean": ["late_mean"], "C2+image+cnv_only": ["image_only", "cnv_only"]}
for name, mods in COMBO.items(): P[name] = pmax((zf(c2_row) + sum(zf(rowsc[m_]) for m_ in mods)) / (1 + len(mods)))
ALL = ARMS + list(COMBO)
# ------------------------------------------------------------- helpers: stratified AUROC and bootstraps
def strat_auc(yy, s, g2):
    num = den = 0.0
    for lvl in np.unique(g2):
        m = g2 == lvl; n1 = yy[m].sum(); n0 = m.sum() - n1
        if n1 > 0 and n0 > 0: num += n1 * n0 * auc(yy[m], s[m]); den += n1 * n0
    return num / den if den else float("nan")
def boots(yy, strata=None):
    rng = np.random.RandomState(SEED); out = []
    while len(out) < NB:
        if strata is None: s = rng.choice(len(yy), len(yy))
        else: s = np.concatenate([rng.choice(np.where(strata == l)[0], (strata == l).sum()) for l in np.unique(strata)])
        if len(set(yy[s])) > 1 and (strata is None or all(len(set(yy[s][strata[s] == l])) > 1 for l in np.unique(strata))): out.append(s)
    return out
B0 = boots(y); BS = boots(y, strat2)
def ci_f(f, yy, Bs, a, b=None, extra=None): v = [f(yy[s], a[s], *(x[s] for x in (extra or []))) - (f(yy[s], b[s], *(x[s] for x in (extra or []))) if b is not None else 0) for s in Bs]; return [r3(np.percentile(v, 2.5)), r3(np.percentile(v, 97.5))]
# ------------------------------------------------------------- R1 probes among non-progressors
neg = y == 0; lab_st = disc.astype(int)
cnv_df = pd.read_csv(F + "/feature_views/cnv/features_5mb_armdiff.csv", low_memory=False).set_index("sample_id"); arms_df = pd.read_csv(F + "/feature_views/cnv/features_arms.csv").set_index("sample_id"); cxdf = pd.read_csv(F + "/feature_views/cnv/cx.csv").set_index("sample_id")
cnv_df.index = cnv_df.index.astype(str); arms_df.index = arms_df.index.astype(str); cxdf.index = cxdf.index.astype(str)
META = {"source_cnv_feature_id", "patient_id", "cnv_id", "slide_ref"}; Xrow = np.column_stack([cnv_df.loc[ids, [c for c in cnv_df.columns if c not in META]].to_numpy(float), arms_df.loc[ids, [c for c in arms_df.columns if c not in META]].to_numpy(float), cxdf.loc[ids, ["cx"]].to_numpy(float)])
Xarm = np.column_stack([arms_df.loc[ids, [c for c in arms_df.columns if c not in META]].to_numpy(float), cxdf.loc[ids, ["cx"]].to_numpy(float)])
def pmean_rows(M): return pd.DataFrame(M, index=pid).groupby(level=0).mean().reindex(pats).values
Xcnv_p = pmean_rows(np.where(np.isfinite(Xrow), Xrow, np.nanmedian(Xrow, 0))); Xarm_p = pmean_rows(np.where(np.isfinite(Xarm), Xarm, np.nanmedian(Xarm, 0)))
L = T + "/feasibility/paper_plan/latent"; EMB = {k: np.load(f"{L}/emb_image_only_fold{k}.npy") for k in range(1, 6)}
qc = pd.read_csv(T + "/feasibility/closeout/swg_cnv_qc.csv"); cxs = pd.read_csv(F + "/feature_views/cnv/cx.csv", dtype=str).set_index("sample_id").reindex(ids); fd = pd.read_csv(S + "/sWGS_777_samples_cleaned_202401_Leanne_fullDetails (3) (1).csv", dtype=str); fd.columns = [c.strip().replace("\n", " ") for c in fd.columns]; fd = fd.drop_duplicates("combined_name").set_index("combined_name")
q = qc.set_index("cnv_id").reindex(cxs.cnv_id.values); reads = pd.to_numeric(pd.Series(cxs.cnv_id.values).map(fd["Number of reads"]).str.replace(",", ""), errors="coerce").values
Xqc_row = np.column_stack([q.noise_mapd.values, q.n_segments.values, q.frac_altered_0p15.values, cxdf.loc[ids, "cx"].values.astype(float), reads]); Xqc_p = pmean_rows(Xqc_row); miss = np.isnan(Xqc_p[:, 4]).astype(float)
def probe(Xp, name, pca=None):
    pr_ = np.full(len(pats), np.nan); idx = np.where(neg)[0]
    for k in range(1, 6):
        tr = idx[pfold[idx] != k]; te = idx[pfold[idx] == k]
        if len(te) == 0 or len(set(lab_st[tr])) < 2: continue
        Xtr, Xte = Xp[tr].copy(), Xp[te].copy()
        if name == "cnv_qc": med = np.nanmedian(Xtr[:, 4]); Xtr[np.isnan(Xtr[:, 4]), 4] = med; Xte[np.isnan(Xte[:, 4]), 4] = med; Xtr = np.column_stack([Xtr, miss[tr]]); Xte = np.column_stack([Xte, miss[te]])
        sc = StandardScaler().fit(Xtr); Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
        if pca: p_ = PCA(min(pca, Xtr.shape[0] - 1), random_state=0).fit(Xtr); Xtr, Xte = p_.transform(Xtr), p_.transform(Xte)
        inn = inner[k].reindex(pats[tr]).values; best, bestC = -1, 1.0
        for C in [0.01, 0.1, 1.0, 10.0]:
            pv = np.zeros(len(tr))
            for j in np.unique(inn):
                v = inn == j
                if len(set(lab_st[tr][~v])) < 2 or v.sum() == 0: continue
                pv[v] = LogisticRegression(C=C, max_iter=5000).fit(Xtr[~v], lab_st[tr][~v]).predict_proba(Xtr[v])[:, 1]
            a = auc(lab_st[tr], pv)
            if a > best: best, bestC = a, C
        pr_[te] = LogisticRegression(C=bestC, max_iter=5000).fit(Xtr, lab_st[tr]).predict_proba(Xte)[:, 1]
    ok = ~np.isnan(pr_); yy = lab_st[ok]; Bn = boots(yy); return {"n_nonprogressors": int(ok.sum()), "discovery": int(yy.sum()), "validation": int((1 - yy).sum()), "auroc": r3(auc(yy, pr_[ok])), "ci": ci_f(auc, yy, Bn, pr_[ok])}
R1 = {"probes_stratum_from_inputs_nonprogressors": {"cnv_features_pca20": probe(Xcnv_p, "cnv", pca=20), "cnv_arms_cx_no_pca": probe(Xarm_p, "arm"), "cnv_qc": probe(Xqc_p, "cnv_qc")}}
# image embedding probe: fold-k embeddings for fold-k patients
img_p = np.zeros((len(pats), 256)); 
for k in range(1, 6): pm = pmean_rows(EMB[k]); img_p[pfold == k] = pm[pfold == k]
img_p_train = {k: pmean_rows(EMB[k]) for k in range(1, 6)}
def probe_img():
    pr_ = np.full(len(pats), np.nan); idx = np.where(neg)[0]
    for k in range(1, 6):
        tr = idx[pfold[idx] != k]; te = idx[pfold[idx] == k]; Xtr, Xte = img_p_train[k][tr], img_p_train[k][te]; sc = StandardScaler().fit(Xtr); Xtr, Xte = sc.transform(Xtr), sc.transform(Xte); inn = inner[k].reindex(pats[tr]).values; best, bestC = -1, 1.0
        for C in [0.01, 0.1, 1.0, 10.0]:
            pv = np.zeros(len(tr))
            for j in np.unique(inn):
                v = inn == j
                if len(set(lab_st[tr][~v])) < 2 or v.sum() == 0: continue
                pv[v] = LogisticRegression(C=C, max_iter=5000).fit(Xtr[~v], lab_st[tr][~v]).predict_proba(Xtr[v])[:, 1]
            a = auc(lab_st[tr], pv)
            if a > best: best, bestC = a, C
        pr_[te] = LogisticRegression(C=bestC, max_iter=5000).fit(Xtr, lab_st[tr]).predict_proba(Xte)[:, 1]
    ok = ~np.isnan(pr_); yy = lab_st[ok]; Bn = boots(yy); return {"n_nonprogressors": int(ok.sum()), "discovery": int(yy.sum()), "validation": int((1 - yy).sum()), "auroc": r3(auc(yy, pr_[ok])), "ci": ci_f(auc, yy, Bn, pr_[ok])}
R1["probes_stratum_from_inputs_nonprogressors"]["image_patient_embedding_256d"] = probe_img()
# stratified AUROC for all arms
n_pairs = {l: int(y[strat2 == l].sum() * (1 - y[strat2 == l]).sum()) for l in ["discovery", "validation"]}
R1["stratified_auroc"] = {"pairs_weights": n_pairs, "n_by_stratum": {l: [int((strat2 == l).sum()), int(y[strat2 == l].sum())] for l in ["discovery", "validation"]}, "arms": {}}
SA = lambda yy, s, g: strat_auc(yy, s, g)
for a in ALL: R1["stratified_auroc"]["arms"][a] = {"pooled": r3(auc(y, P[a])), "pooled_ci": ci_f(auc, y, B0, P[a]), "stratified": r3(strat_auc(y, P[a], strat2)), "stratified_ci": ci_f(SA, y, BS, P[a], extra=[strat2]), "discovery": r3(auc(y[disc], P[a][disc])), "validation": r3(auc(y[~disc], P[a][~disc]))}
R1["stratified_paired_differences"] = {}
for a, b in [("image_only", "cnv_only"), ("image_only", "cnv_km"), ("late_mean", "image_only"), ("late_mean_km", "image_only"), ("late_mean", "cnv_only")] + [(c, "C2_grade_maxsofar") for c in COMBO]:
    R1["stratified_paired_differences"][f"{a}_minus_{b}"] = {"stratified_delta": r3(strat_auc(y, P[a], strat2) - strat_auc(y, P[b], strat2)), "ci": ci_f(SA, y, BS, P[a], P[b], extra=[strat2]), "pooled_delta": r3(auc(y, P[a]) - auc(y, P[b])), "pooled_ci": ci_f(auc, y, B0, P[a], P[b])}
# stratum-adjusted incremental value
def lr_incr(base_cols, a):
    Xb = sm.add_constant(np.column_stack(base_cols)); m0 = sm.Logit(y, Xb).fit(disp=0, maxiter=200); Xa = sm.add_constant(np.column_stack(base_cols + [logit(P[a])])); m1 = sm.Logit(y, Xa).fit(disp=0, maxiter=200)
    lr = 2 * (m1.llf - m0.llf); out = {"LR_stat": r3(lr), "LR_p": pf(chi2.sf(lr, 1)), "auroc_base_insample": r3(auc(y, m0.predict(Xb))), "auroc_base_plus_arm_insample": r3(auc(y, m1.predict(Xa))), "delta_auroc_insample": r3(auc(y, m1.predict(Xa)) - auc(y, m0.predict(Xb)))}
    pb = np.zeros(len(y)); pa = np.zeros(len(y))
    for k in range(1, 6):
        tr = pfold != k; te = ~tr; Xb_ = np.column_stack(base_cols); Xa_ = np.column_stack(base_cols + [logit(P[a])])
        pb[te] = LogisticRegression(C=1e6, max_iter=5000).fit(Xb_[tr], y[tr]).predict_proba(Xb_[te])[:, 1]; pa[te] = LogisticRegression(C=1e6, max_iter=5000).fit(Xa_[tr], y[tr]).predict_proba(Xa_[te])[:, 1]
    out["auroc_base_foldhonest"] = r3(auc(y, pb)); out["auroc_base_plus_arm_foldhonest"] = r3(auc(y, pa)); out["delta_auroc_foldhonest"] = r3(auc(y, pa) - auc(y, pb)); out["delta_ci_foldhonest"] = ci_f(auc, y, B0, pa, pb); return out
R1["stratum_adjusted_incremental_value"] = {a: lr_incr([disc.astype(float)], a) for a in ALL}; R1["stratum_alone"] = {"auroc_insample": r3(auc(y, disc.astype(float))), "n": int(len(y)), "events": int(y.sum())}
RES["R1"] = R1; print("R1 done", flush=True)
# ------------------------------------------------------------- R2 endpoints
ptype = man[man.y == 1].groupby("patient_id").next_label.max().reindex(pats); hgd = (ptype >= 3).fillna(False).values; lgd_only = (y == 1) & ~hgd
yH = hgd.astype(int); keep = ~lgd_only; BH = boots(yH); BHs = boots(yH, strat2); BHe = boots(yH[keep]); BHes = boots(yH[keep], strat2[keep])
R2 = {"progressor_types": {"HGD_IMC": int(hgd.sum()), "second_LGD_only": int(lgd_only.sum()), "by_stratum": {l: {"HGD_IMC": int((hgd & (strat2 == l)).sum()), "second_LGD_only": int((lgd_only & (strat2 == l)).sum())} for l in ["discovery", "validation"]}}, "E_HGD": {"n": int(len(yH)), "events": int(yH.sum()), "arms": {}}, "E_HGD_excl": {"n": int(keep.sum()), "events": int(yH[keep].sum()), "arms": {}}}
for a in ALL:
    R2["E_HGD"]["arms"][a] = {"pooled": r3(auc(yH, P[a])), "pooled_ci": ci_f(auc, yH, BH, P[a]), "stratified": r3(strat_auc(yH, P[a], strat2)), "stratified_ci": ci_f(SA, yH, BHs, P[a], extra=[strat2]), "discovery": r3(auc(yH[disc], P[a][disc])), "validation": r3(auc(yH[~disc], P[a][~disc]))}
    R2["E_HGD_excl"]["arms"][a] = {"pooled": r3(auc(yH[keep], P[a][keep])), "pooled_ci": ci_f(auc, yH[keep], BHe, P[a][keep]), "stratified": r3(strat_auc(yH[keep], P[a][keep], strat2[keep])), "stratified_ci": ci_f(SA, yH[keep], BHes, P[a][keep], extra=[strat2[keep]])}
R2["second_LGD_only_progressors_scores_rank"] = {a: {"median_percentile_among_all_patients": r3(np.median(rankdata(P[a])[lgd_only] / len(y) * 100))} for a in ["cnv_only", "cnv_km", "image_only", "late_mean", "C2_grade_maxsofar"]}
RES["R2"] = R2; print("R2 done", flush=True)
# ------------------------------------------------------------- R3 tissue amount
ui = pd.read_csv(F + "/feature_views/uni2/uni2_index.csv", dtype=str).set_index("sample_id").reindex(ids); kept = []; grid = []
for b in ui.image_basename:
    with h5py.File(f"{S}/features_uni2h_05um/{os.path.splitext(str(b))[0]}.h5") as h: kept.append(float(h.attrs["kept_tiles"])); grid.append(float(h.attrs["grid_tiles"]))
man["kept"] = kept; man["tfrac"] = np.array(kept) / np.array(grid); pk = man.groupby("patient_id").kept.mean().reindex(pats).values; ptf = man.groupby("patient_id").tfrac.mean().reindex(pats).values
slm = pd.read_csv(T + "/feasibility/closeout/swg_slide_meta.csv", dtype=str).set_index("sample_id").reindex(ids); man["scanner"] = slm["tiff.Model"].values; pscan = man.groupby("patient_id").scanner.agg(lambda s: s.mode().iloc[0]).reindex(pats).values
def signed_score(v):
    s = np.zeros(len(v))
    for k in range(1, 6): tr = pfold != k; te = ~tr; sgn = 1.0 if auc(y[tr], v[tr]) >= 0.5 else -1.0; s[te] = sgn * v[te]
    return s
sk = signed_score(pk); stf = signed_score(ptf)
R3 = {"kept_tiles_as_score": {"pooled": {"auroc": r3(auc(y, sk)), "ci": ci_f(auc, y, B0, sk), "n": int(len(y)), "events": int(y.sum()), "sign_note": "sign chosen on training folds per fold"}, "stratified": r3(strat_auc(y, sk, strat2)), **{l: {"auroc": r3(auc(y[strat2 == l], sk[strat2 == l])), "n": int((strat2 == l).sum()), "events": int(y[strat2 == l].sum())} for l in ["discovery", "validation"]}, "raw_pooled_auroc_more_tiles_higher": r3(auc(y, pk))},
      "tissue_frac_as_score": {"pooled_auroc": r3(auc(y, stf)), "ci": ci_f(auc, y, B0, stf)}, "kept_tiles_distribution": {"by_stratum": {l: {"n": int((strat3 == l).sum()), "median": r3(np.median(pk[strat3 == l])), "iqr": [r3(np.percentile(pk[strat3 == l], 25)), r3(np.percentile(pk[strat3 == l], 75))]} for l in np.unique(strat3)}, "discovery_vs_validation_mannwhitney_p": pf(mannwhitneyu(pk[disc], pk[~disc]).pvalue),
      "by_scanner": {s_: {"n": int((pscan == s_).sum()), "median": r3(np.median(pk[pscan == s_])), "iqr": [r3(np.percentile(pk[pscan == s_], 25)), r3(np.percentile(pk[pscan == s_], 75))]} for s_ in np.unique(pscan.astype(str))}, "by_label": {"progressors_median": r3(np.median(pk[y == 1])), "nonprogressors_median": r3(np.median(pk[y == 0])), "mannwhitney_p": pf(mannwhitneyu(pk[y == 1], pk[y == 0]).pvalue)}}}
R3["spearman_arm_score_vs_kept_tiles_nonprogressors"] = {a: {"rho": r3(spearmanr(P[a][neg], pk[neg]).correlation), "p": pf(spearmanr(P[a][neg], pk[neg]).pvalue)} for a in ALL}
R3["stratum_plus_tiles_incremental_value"] = {a: lr_incr([disc.astype(float), np.log(pk)], a) for a in ALL}; R3["stratum_plus_tiles_base"] = {"auroc_insample": R3["stratum_plus_tiles_incremental_value"]["image_only"]["auroc_base_insample"], "auroc_foldhonest": R3["stratum_plus_tiles_incremental_value"]["image_only"]["auroc_base_foldhonest"]}
fn = (sp.pred_late_mean.values == 0) & (y == 1); R3["late_mean_FN_tiles"] = []
for i in np.where(fn)[0]:
    l = strat2[i]; prog = pk[(strat2 == l) & (y == 1)]; nonp = pk[(strat2 == l) & (y == 0)]; R3["late_mean_FN_tiles"].append({"stratum": l, "kept_tiles": r3(pk[i]), "percentile_among_stratum_progressors": r3((prog < pk[i]).mean() * 100), "percentile_among_stratum_nonprogressors": r3((nonp < pk[i]).mean() * 100)})
R3["late_mean_FN_summary"] = {"n": int(fn.sum()), "median_kept": r3(np.median(pk[fn])), "median_percentile_vs_stratum_progressors": r3(np.median([d["percentile_among_stratum_progressors"] for d in R3["late_mean_FN_tiles"]])), "median_percentile_vs_stratum_nonprogressors": r3(np.median([d["percentile_among_stratum_nonprogressors"] for d in R3["late_mean_FN_tiles"]])), "by_stratum": pd.Series(strat2[fn]).value_counts().to_dict()}
RES["R3"] = R3; print("R3 done", flush=True)
# ------------------------------------------------------------- R4 risk groups
PT = pd.read_csv(T + "/feasibility/closeout/swg_patient_table.csv", dtype=str).set_index("patient_id"); g = man.groupby("patient_id"); first = g.date.min().reindex(pats); lastrow = g.date.max().reindex(pats); last_next = g.nbd.max().reindex(pats)
t_event = pd.to_numeric(PT.days_first_to_event.reindex(pats), errors="coerce").values; timeB = np.where(y == 1, t_event, (last_next.fillna(lastrow) - first).dt.days.values); timeB = np.maximum(np.nan_to_num(timeB, nan=0), 0.5)
def groups_tertile(a):
    gg = np.zeros(len(y), int)
    for k in range(1, 6):
        tr = pfold != k; te = ~tr; q1, q2 = np.quantile(P[a][tr], [1 / 3, 2 / 3]); gg[te] = np.where(P[a][te] > q2, 2, np.where(P[a][te] > q1, 1, 0))
    return gg
def wilson(k, n, z=1.96):
    if n == 0: return [None, None]
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d; return [r3(c - h), r3(c + h)]
R4 = {"version_A_superseded": "the paper-plan item-5 and follow-up F4 version-A results (censoring from MonthsBeforeLastBiopsy) are superseded by version B (biopsy-date censoring); earlier documents are not edited", "validation_stratum_survival": {}, "pooled_rates_and_ORs": {}}
val = ~disc
for a in ["late_mean", "cnv_only", "cnv_km", "C2_grade_maxsofar", "image_only"]:
    gg = groups_tertile(a); out = {"n": int(val.sum()), "events": int(y[val].sum()), "groups": {}}
    for gi, gn in enumerate(["low", "moderate", "high"]): m = val & (gg == gi); out["groups"][gn] = {"n": int(m.sum()), "events": int(y[m].sum()), "median_time_d": r3(np.median(timeB[m])) if m.sum() else None}
    df = pd.DataFrame({"T": timeB[val], "E": y[val], "g": gg[val], "mod": (gg[val] == 1).astype(int), "high": (gg[val] == 2).astype(int)})
    try: out["logrank_p"] = pf(multivariate_logrank_test(df["T"], df.g, df.E).p_value)
    except Exception as e: out["logrank_error"] = str(e)[:80]
    feasible = all(out["groups"][gn]["events"] >= 3 for gn in ["low", "high"]); out["cox_feasible_rule_ge3_events_low_and_high"] = feasible
    if feasible:
        try: cph = CoxPHFitter().fit(df[["T", "E", "mod", "high"]], "T", "E"); out["cox_hr_high_vs_low"] = [r3(np.exp(cph.params_["high"]))] + [r3(v) for v in np.exp(cph.confidence_intervals_.loc["high"]).values]; out["cox_hr_moderate_vs_low"] = [r3(np.exp(cph.params_["mod"]))] + [r3(v) for v in np.exp(cph.confidence_intervals_.loc["mod"]).values]
        except Exception as e: out["cox_error"] = str(e)[:80]
    fig, ax = plt.subplots(figsize=(5, 4)); km = KaplanMeierFitter()
    for gi, gn, c in [(0, "low", "#1f77b4"), (1, "moderate", "#ff7f0e"), (2, "high", "#d62728")]:
        m = df.g == gi
        if m.sum(): km.fit(df["T"][m] / 365.25, df.E[m], label=f"{gn} (n={int(m.sum())}, events={int(df.E[m].sum())})").plot_survival_function(ax=ax, ci_show=False, color=c)
    ax.set_xlabel("years from earliest release row"); ax.set_ylabel("endpoint-free"); ax.set_title(f"{a} tertiles, validation stratum, version B"); fn_ = f"{FIG}/km_validation_{a}.png"; fig.tight_layout(); fig.savefig(fn_, dpi=130); plt.close(fig); out["figure"] = fn_.replace(T + "/", "")
    R4["validation_stratum_survival"][a] = out
    # pooled rates + ORs + MH
    po = {"groups": {}}
    for gi, gn in enumerate(["low", "moderate", "high"]): m = gg == gi; po["groups"][gn] = {"n": int(m.sum()), "events": int(y[m].sum()), "rate": r3(y[m].mean()) if m.sum() else None, "wilson": wilson(int(y[m].sum()), int(m.sum()))}
    hl = gg != 1; hi = (gg == 2).astype(int); a_, b_, c_, d_ = y[gg == 2].sum() + .5, (1 - y[gg == 2]).sum() + .5, y[gg == 0].sum() + .5, (1 - y[gg == 0]).sum() + .5; po["OR_high_vs_low_haldane"] = r3((a_ * d_) / (b_ * c_))
    for nm, strat in [("MH_two_strata", strat2), ("MH_three_strata", strat3)]:
        tabs = []
        for l in np.unique(strat):
            m = hl & (strat == l); t = np.array([[((hi == 1) & (y == 1) & m).sum(), ((hi == 1) & (y == 0) & m).sum()], [((hi == 0) & (y == 1) & m).sum(), ((hi == 0) & (y == 0) & m).sum()]], float)
            if t.sum() > 0: tabs.append(t)
        try: stt = StratifiedTable(tabs); po[nm] = {"OR": r3(stt.oddsratio_pooled), "ci": [r3(v) for v in stt.oddsratio_pooled_confint()], "strata_used": len(tabs)}
        except Exception as e: po[nm] = {"error": str(e)[:80]}
    R4["pooled_rates_and_ORs"][a] = po
RES["R4"] = R4; print("R4 done", flush=True)
# ------------------------------------------------------------- R5 false positives
LT = pd.read_csv(ROW + "/later_disease_patient.csv", dtype={"patient_id": str}).set_index("patient_id").reindex(pats); bgr = pd.to_numeric(PT.baseline_grade.reindex(pats), errors="coerce").fillna(0).values; mxs = g.maxsofar.max().reindex(pats).values; fu = LT.db_followup_days.fillna(0).values
laterL = ((LT.db_max_grade_after >= 2) | (LT.release_excluded_max_grade >= 2) | (LT.slidematch_max_grade_after >= 2)).values; laterH = ((LT.db_max_grade_after >= 3) | (LT.release_excluded_max_grade >= 3) | (LT.slidematch_max_grade_after >= 3) | (LT.hgd_table_entries_after > 0)).values
PRED = {a: sp[f"pred_{a}"].values.astype(int) for a in ["late_mean", "image_only", "cnv_only", "cnv_km", "C2_grade_maxsofar", "clinical_3a", "v4_exploratory"]}
flag = pd.read_csv(ROW + "/f5_later_hgd_nonprogressors.csv", dtype={"patient_id": str}); flagged = set(flag[flag.hgd_report_is_the_next_release_biopsy.astype(str) == "True"].patient_id)
def f5(a, pred, negmask, label):
    fp = (pred == 1) & negmask; out = {"n_FP": int(fp.sum()), "n_TN": int((negmask & (pred == 0)).sum()), "FP_TN_by_stratum": {l: [int((fp & (strat2 == l)).sum()), int((negmask & (pred == 0) & (strat2 == l)).sum())] for l in ["discovery", "validation"]}}
    for nm, lab in [("later_HGDplus", laterH), ("later_LGDplus", laterL)]:
        Xd = pd.DataFrame({"FP": fp[negmask].astype(float), "baseline_grade": bgr[negmask], "max_grade_so_far": mxs[negmask], "discovery": disc[negmask].astype(float), "followup_days": fu[negmask] / 365.25}); yy = lab[negmask].astype(int)
        try: mres = sm.Logit(yy, sm.add_constant(Xd)).fit(disp=0, maxiter=300); ci_ = np.exp(mres.conf_int().loc["FP"]); out[nm] = {"OR_FP_adjusted": r3(np.exp(mres.params["FP"])), "ci": [r3(ci_[0]), r3(ci_[1])], "p": pf(mres.pvalues["FP"]), "OR_discovery": r3(np.exp(mres.params["discovery"])), "OR_followup_per_year": r3(np.exp(mres.params["followup_days"])), "events": int(yy.sum()), "n": int(len(yy)), "raw_FP_TN": [int(lab[fp].sum()), int(lab[negmask & (pred == 0)].sum())]}
        except Exception as e: out[nm] = {"error": str(e)[:100], "raw_FP_TN": [int(lab[fp].sum()), int(lab[negmask & (pred == 0)].sum())]}
    return out
R5 = {"adjusted_with_stratum_and_followup": {a: f5(a, PRED[a], neg, a) for a in PRED}, "flagged_patients_reclassified": len(flagged)}
neg2 = neg & ~pd.Series(pats).isin(flagged).values; y2 = y.copy(); y2[pd.Series(pats).isin(flagged).values] = 1
R5["rerun_with_flagged_as_progressor"] = {a: f5(a, PRED[a], neg2, a) for a in PRED}; R5["rerun_with_flagged_as_progressor"]["late_mean_pooled_auroc_relabelled"] = {"auroc": r3(auc(y2, P["late_mean"])), "events": int(y2.sum()), "original": r3(auc(y, P["late_mean"]))}
RES["R5"] = R5; print("R5 done", flush=True)
# ------------------------------------------------------------- R6 headline
R6 = {}
for a in ALL: R6[a] = {"pooled": [R1["stratified_auroc"]["arms"][a]["pooled"]] + R1["stratified_auroc"]["arms"][a]["pooled_ci"], "stratified": [R1["stratified_auroc"]["arms"][a]["stratified"]] + R1["stratified_auroc"]["arms"][a]["stratified_ci"], "E_HGD_pooled": [R2["E_HGD"]["arms"][a]["pooled"]] + R2["E_HGD"]["arms"][a]["pooled_ci"], "E_HGD_stratified": [R2["E_HGD"]["arms"][a]["stratified"]] + R2["E_HGD"]["arms"][a]["stratified_ci"], "delta_over_stratum_plus_tiles_foldhonest": [R3["stratum_plus_tiles_incremental_value"][a]["delta_auroc_foldhonest"]] + R3["stratum_plus_tiles_incremental_value"][a]["delta_ci_foldhonest"]}
R6["_n_events"] = {"pooled": [int(len(y)), int(y.sum())], "E_HGD": [int(len(yH)), int(yH.sum())]}; RES["R6"] = R6
pd.DataFrame({"patient_id": pats, "y": y, "y_hgd": yH, "stratum2": strat2, "kept_tiles": pk, **{a: P[a] for a in ALL}}).to_csv(ROW + "/round3_patient_table.csv", index=False)
json.dump(RES, open(AGG + "/round3_main.json", "w"), indent=1, default=str); print("ROUND3 MAIN DONE", flush=True)

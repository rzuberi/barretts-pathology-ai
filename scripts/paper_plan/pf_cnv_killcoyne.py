"""Follow-up F1 (pre-specified in docs/paper_plan_followup.md @ 0db4075): Killcoyne-method CNV arm = elastic-net logistic
regression (l1_ratio 0.9, C by release inner folds) on the release CNV features (5-Mb window-minus-arm + arms + cx),
standardised on outer training rows; release outer folds; our endpoint. Plus faithfulness check vs the published LOPO
predictions, item 2/4 reruns with this arm, and the diagnosis of cnv_only on the 25 overlap-discovery patients.
Outputs: results/paper_plan/f1_cnv_killcoyne.json; row OOF -> feasibility/paper_plan/f1_cnv_km_oof.csv."""
import glob, json, os, sys, warnings, numpy as np, pandas as pd, joblib
from scipy.stats import rankdata, spearmanr, mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
warnings.filterwarnings("ignore")
F = "/mnt/scratche/slow/fmlab/zuberi01/phd/barretts_retraining/barretts_training/analysis/chapter1_lgd2_final_pre_event_20260713_final"; R = F + "/training_final_nested_cv_v1"
B = "/mnt/scratche/slow/fmlab/zuberi01/phd/barretts_retraining/barretts_training/multimodal-barretts-progression"; sys.path.insert(0, B + "/src"); from barrett.training.data import load_cnv_matrix
T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"; S = "/mnt/scratche/fast/fmlab/datasets/imaging/SWGCohort"; K = "/mnt/scratche/slow/fmlab/zuberi01/phd/killcoyne_data_from_paper"
ROW = T + "/feasibility/paper_plan"; AGG = os.environ.get("OUTDIR", T + "/results/paper_plan"); os.makedirs(AGG, exist_ok=True); NB = 2000; SEED = 0
def auc(y, s):
    y = np.asarray(y).astype(int); r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1; return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if 0 < n1 < len(y) else float("nan")
def ap(y, s): return float(average_precision_score(y, s)) if 0 < np.sum(y) < len(y) else float("nan")
def r3(x): return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 3)
def pf(p): return None if p is None or (isinstance(p, float) and np.isnan(p)) else ("<0.001" if p < 0.001 else round(float(p), 3))
man = pd.read_csv(F + "/training_manifest.csv", dtype=str).set_index("sample_id"); ids = list(man.index); y = man.y_progressor.astype(int).values; fold = man.fold_id_rep01.astype(int).values; pid = man.patient_id.values
cnv, feats = load_cnv_matrix(F + "/feature_views/cnv"); X = cnv.set_index("sample_id").loc[ids, feats].to_numpy(np.float64); print("features", len(feats), flush=True)
inner = {k: pd.read_csv(f"{R}/image_only/fold{k}/inner_fold_assignments.csv", dtype=str).set_index("patient_id").inner_fold.astype(int) for k in range(1, 6)}
def pat_auc(idx, s): g = pd.DataFrame({"p": pid[idx], "s": s, "y": y[idx]}).groupby("p"); return auc(g.y.max().values, g.s.max().values)
CS = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
def fit_en(alpha):
    oof = np.zeros(len(y)); chosen = {}; nnz = {}
    for k in range(1, 6):
        tr = fold != k; te = ~tr; med = np.nanmedian(X[tr], 0); Xtr = np.where(np.isfinite(X[tr]), X[tr], med); Xte = np.where(np.isfinite(X[te]), X[te], med); mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9; Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
        inn = inner[k].reindex(pid[tr]).values; best, bestC = -1, 1.0; tri = np.where(tr)[0]
        for C in CS:
            pv = np.zeros(tr.sum())
            for j in np.unique(inn): v = inn == j; pv[v] = LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=alpha, C=C, max_iter=20000, tol=1e-4).fit(Xtr[~v], y[tr][~v]).predict_proba(Xtr[v])[:, 1]
            a = pat_auc(tri, pv)
            if a > best: best, bestC = a, C
        m = LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=alpha, C=bestC, max_iter=20000, tol=1e-4).fit(Xtr, y[tr]); oof[te] = m.predict_proba(Xte)[:, 1]; chosen[k] = bestC; nnz[k] = int((m.coef_ != 0).sum()); print("alpha", alpha, "fold", k, "C", bestC, "nnz", nnz[k], flush=True)
    return oof, chosen, nnz
oof_km, C_km, nnz_km = fit_en(0.9)
sens = {}
for a_ in [0.1, 0.5]:
    o, c, n = fit_en(a_); sens[str(a_)] = {"oof_patient_auroc": r3(pat_auc(np.arange(len(y)), o)), "C": c, "nnz": n}
def oof_rel(fam): return pd.concat([pd.read_csv(f, dtype={"sample_id": str}) for f in glob.glob(f"{R}/{fam}/fold*/outer_test_predictions.csv")]).set_index("sample_id").y_prob.reindex(ids).values
img, cnv_rel, late = oof_rel("image_only"), oof_rel("cnv_only"), oof_rel("late_mean"); late_km = (img + oof_km) / 2
pd.DataFrame({"sample_id": ids, "patient_id": pid, "fold": fold, "y": y, "cnv_km": oof_km, "cnv_only": cnv_rel, "image_only": img, "late_mean": late, "late_mean_km": late_km}).to_csv(ROW + "/f1_cnv_km_oof.csv", index=False)
# subsets
pairs = pd.read_csv(T + "/feasibility/closeout/erin_swg_pairs.csv", dtype=str); OV = set(pairs.swg_patient_id)
cx = pd.read_csv(F + "/feature_views/cnv/cx.csv", dtype=str).set_index("sample_id").reindex(ids); fd = pd.read_csv(S + "/sWGS_777_samples_cleaned_202401_Leanne_fullDetails (3) (1).csv", dtype=str); fd.columns = [c.strip().replace("\n", " ") for c in fd.columns]
kdisc = pd.Series(cx.cnv_id.isin(set(fd.combined_name)).values, index=pid).groupby(level=0).any()
def patient(cols, rows=None):
    d = pd.DataFrame(cols, index=ids).assign(y=y, p=pid); d = d if rows is None else d[rows]; g = d.groupby("p"); return g.y.max().values.astype(int), {c: g[c].max().values for c in cols}, g.size().index.values
def bidx(yy):
    rng = np.random.RandomState(SEED); out = []
    while len(out) < NB:
        s = rng.choice(len(yy), len(yy))
        if len(set(yy[s])) > 1: out.append(s)
    return out
def cis(yy, Bs, f, a, b=None): v = [f(yy[s], a[s]) - (f(yy[s], b[s]) if b is not None else 0) for s in Bs]; return [r3(np.percentile(v, 2.5)), r3(np.percentile(v, 97.5))]
def permp(yy, a, b): rng = np.random.RandomState(SEED); obs = auc(yy, a) - auc(yy, b); null = np.array([auc(yy[ix], a) - auc(yy[ix], b) for ix in (rng.permutation(len(yy)) for _ in range(NB))]); return round(float((1 + (null >= obs).sum()) / (NB + 1)), 4)
COLS = {"cnv_km": oof_km, "cnv_only": cnv_rel, "image_only": img, "late_mean": late, "late_mean_km": late_km}
sub_ov = pd.Series(pid).isin(OV).values; sub_k = pd.Series(pid).map(kdisc).values.astype(bool)
RES = {"_spec": "docs/paper_plan_followup.md @ 0db4075 F1", "features": {"n_total": len(feats), "n_5mb_window_minus_arm": int(sum(1 for f in feats if ":" in f)), "n_arm": int(sum(1 for f in feats if f.startswith("chr") and ":" not in f)), "cx": int("cx" in feats)}, "model": "sklearn LogisticRegression(penalty=elasticnet, solver=saga, l1_ratio=0.9), C by release inner folds (patient AUROC), per-feature standardisation and median imputation on outer training rows", "C_per_fold": C_km, "nonzero_coefs_per_fold": nnz_km, "alpha_sensitivity": sens, "subsets": {}}
for label, rows in [("all_150", None), ("never_in_ERIN_96", ~sub_ov), ("also_in_ERIN_54", sub_ov), ("killcoyne_discovery_82", sub_k), ("other_68", ~sub_k)]:
    yy, P, pats = patient(COLS, rows); Bs = bidx(yy); out = {"n": int(len(yy)), "events": int(yy.sum()), "arms": {a: {"auroc": r3(auc(yy, P[a])), "auroc_ci": cis(yy, Bs, auc, P[a]), "auprc": r3(ap(yy, P[a])), "auprc_ci": cis(yy, Bs, ap, P[a])} for a in COLS}}
    out["item2_image_vs_cnv_km"] = {"delta": r3(auc(yy, P["image_only"]) - auc(yy, P["cnv_km"])), "ci": cis(yy, Bs, auc, P["image_only"], P["cnv_km"]), "perm_p": permp(yy, P["image_only"], P["cnv_km"]), "lower_ci_vs_noninferiority_margin_-0.05": None}
    out["item2_image_vs_cnv_only"] = {"delta": r3(auc(yy, P["image_only"]) - auc(yy, P["cnv_only"])), "ci": cis(yy, Bs, auc, P["image_only"], P["cnv_only"]), "perm_p": permp(yy, P["image_only"], P["cnv_only"])}
    for k_ in ["item2_image_vs_cnv_km", "item2_image_vs_cnv_only"]: out[k_]["noninferior_at_-0.05"] = bool(out[k_]["ci"][0] is not None and out[k_]["ci"][0] > -0.05)
    out["item4_late_km_deltas"] = {f"late_mean_km_vs_{ref}": {"delta": r3(auc(yy, P["late_mean_km"]) - auc(yy, P[ref])), "ci": cis(yy, Bs, auc, P["late_mean_km"], P[ref]), "perm_p": permp(yy, P["late_mean_km"], P[ref])} for ref in ["image_only", "cnv_km", "cnv_only", "late_mean"]}
    out["item4_late_mean_deltas"] = {f"late_mean_vs_{ref}": {"delta": r3(auc(yy, P["late_mean"]) - auc(yy, P[ref])), "ci": cis(yy, Bs, auc, P["late_mean"], P[ref]), "perm_p": permp(yy, P["late_mean"], P[ref])} for ref in ["image_only", "cnv_only", "cnv_km"]}
    RES["subsets"][label] = out; print("subset", label, flush=True)
# faithfulness vs published LOPO
kp = pd.read_excel(K + "/41591_2020_1033_MOESM4_ESM.xlsx", sheet_name="Supporting data for Figure 2a", header=1).rename(columns={"Samplename": "cnv_id", "Probability": "k_prob"}); m = pd.DataFrame({"cnv_id": cx.cnv_id.values, "cnv_km": oof_km, "cnv_only": cnv_rel, "y": y, "p": pid}).merge(kp[["cnv_id", "k_prob"]], on="cnv_id")
g = m.groupby("p"); yk = g.y.max().values
RES["faithfulness_vs_published_LOPO"] = {"matched_rows": int(len(m)), "matched_patients": int(m.p.nunique()), "spearman_cnv_km_vs_killcoyne": r3(spearmanr(m.cnv_km, m.k_prob).correlation), "spearman_cnv_only_vs_killcoyne": r3(spearmanr(m.cnv_only, m.k_prob).correlation), "spearman_cnv_km_vs_cnv_only": r3(spearmanr(m.cnv_km, m.cnv_only).correlation),
    "patient_auroc_matched": {"killcoyne_max": r3(auc(yk, g.k_prob.max().values)), "cnv_km": r3(auc(yk, g.cnv_km.max().values)), "cnv_only": r3(auc(yk, g.cnv_only.max().values)), "n": int(len(yk)), "events": int(yk.sum())}}
# diagnosis of cnv_only on the 25 overlap-discovery patients
d25 = sub_ov & sub_k; p25 = set(pid[d25]); RES["diagnosis_cnv_only_overlap_discovery"] = {"n_patients": int(len(p25)), "n_rows": int(d25.sum()), "per_fold": {}, "training_fold_class_balance": {}}
for k in range(1, 6):
    te = (fold == k) & d25; tr = fold != k
    yy_, P_, _ = patient({"cnv_only": cnv_rel, "cnv_km": oof_km}, te) if te.sum() else (np.array([]), {}, [])
    RES["diagnosis_cnv_only_overlap_discovery"]["per_fold"][k] = {"held_out_25_patients": int(len(yy_)), "events": int(yy_.sum()) if len(yy_) else 0, "auroc_cnv_only": r3(auc(yy_, P_["cnv_only"])) if len(yy_) and 0 < yy_.sum() < len(yy_) else None, "auroc_cnv_km": r3(auc(yy_, P_["cnv_km"])) if len(yy_) and 0 < yy_.sum() < len(yy_) else None}
    gtr = pd.DataFrame({"p": pid[tr], "y": y[tr]}).groupby("p").y.max(); RES["diagnosis_cnv_only_overlap_discovery"]["training_fold_class_balance"][k] = {"rows": int(tr.sum()), "positive_rows": int(y[tr].sum()), "patients": int(len(gtr)), "progressor_patients": int(gtr.sum()), "rows_from_25_patients_in_training": int((tr & d25).sum()), "positive_rows_from_25_in_training": int(y[tr & d25].sum())}
ARM = [i for i, f in enumerate(feats) if ":" not in f]; zabs = np.zeros(len(y)); pcn = np.zeros(len(y)); rfp = cnv_rel
for k in range(1, 6):
    tr = fold != k; te = fold == k; med = np.nanmedian(X[tr], 0); Xa = np.where(np.isfinite(X), X, med); mu, sd = Xa[tr].mean(0), Xa[tr].std(0) + 1e-9; Z = (Xa - mu) / sd; zabs[te] = np.abs(Z[te][:, ARM]).mean(1)
    pipe = joblib.load(f"{R}/cnv_only/fold{k}/model.joblib"); pcn[te] = np.linalg.norm(pipe.named_steps["pca"].transform(pipe.named_steps["scale"].transform(pipe.named_steps["impute"].transform(X[te]))), axis=1)
def mw(a, b): return {"median_25": r3(np.median(a)), "median_rest": r3(np.median(b)), "p": pf(mannwhitneyu(a, b).pvalue)}
RES["diagnosis_cnv_only_overlap_discovery"]["feature_distributions_rows"] = {"mean_abs_z_arm_and_cx": mw(zabs[d25], zabs[~d25]), "pca64_score_norm": mw(pcn[d25], pcn[~d25]), "cx_raw": mw(X[d25, feats.index("cx")], X[~d25, feats.index("cx")]), "rows_25": int(d25.sum()), "rows_rest": int((~d25).sum())}
RES["diagnosis_cnv_only_overlap_discovery"]["rf_probability_rows"] = {"25_progressor_rows": {"n": int((d25 & (y == 1)).sum()), "median": r3(np.median(rfp[d25 & (y == 1)]))}, "25_nonprogressor_rows": {"n": int((d25 & (y == 0)).sum()), "median": r3(np.median(rfp[d25 & (y == 0)]))}, "rest_progressor_rows": {"n": int((~d25 & (y == 1)).sum()), "median": r3(np.median(rfp[~d25 & (y == 1)]))}, "rest_nonprogressor_rows": {"n": int((~d25 & (y == 0)).sum()), "median": r3(np.median(rfp[~d25 & (y == 0)]))}, "cnv_km_25_prog_median": r3(np.median(oof_km[d25 & (y == 1)])), "cnv_km_25_nonprog_median": r3(np.median(oof_km[d25 & (y == 0)]))}
json.dump(RES, open(AGG + "/f1_cnv_killcoyne.json", "w"), indent=1, default=str); print(json.dumps({k: RES[k] for k in ["C_per_fold", "nonzero_coefs_per_fold", "faithfulness_vs_published_LOPO"]}, indent=1)); print("F1 DONE", flush=True)

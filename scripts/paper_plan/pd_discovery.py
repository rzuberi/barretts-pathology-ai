"""Discovery-stratum analysis (pre-specified in docs/paper_plan_discovery.md @ 30b2317): the 82 Killcoyne-discovery patients,
endpoints LGD2+ and E-HGD, every R6 arm, paired differences, selection-adjusted p, and the F5 FP analysis within discovery
non-progressors. Output: results/paper_plan/discovery_main.json."""
import json, os, warnings, numpy as np, pandas as pd
from scipy.stats import rankdata, fisher_exact
from sklearn.metrics import average_precision_score
import statsmodels.api as sm
warnings.filterwarnings("ignore")
F = "/mnt/scratche/slow/fmlab/zuberi01/phd/barretts_retraining/barretts_training/analysis/chapter1_lgd2_final_pre_event_20260713_final"; T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"; ROW = T + "/feasibility/paper_plan"; AGG = os.environ.get("OUTDIR", T + "/results/paper_plan"); os.makedirs(AGG, exist_ok=True); NB = 2000; SEED = 0
def auc(y, s):
    y = np.asarray(y).astype(int); r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1; return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if 0 < n1 < len(y) else float("nan")
def ap(y, s): return float(average_precision_score(y, s)) if 0 < np.sum(y) < len(y) else float("nan")
def r3(x): return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 3)
def pf(p): return None if p is None or (isinstance(p, float) and np.isnan(p)) else ("<0.001" if p < 0.001 else round(float(p), 3))
pt = pd.read_csv(ROW + "/round3_patient_table.csv", dtype={"patient_id": str}).set_index("patient_id"); d = pt[pt.stratum2 == "discovery"]; pats = d.index.values
ARMS = [c for c in pt.columns if c not in ("y", "y_hgd", "stratum2", "kept_tiles")]; FUS = ["early_fusion", "intermediate_fusion", "late_mean", "coattention_fusion", "late_stack_logit"]
man = pd.read_csv(F + "/training_manifest.csv", dtype=str); pfold = man.groupby("patient_id").fold_id_rep01.first().astype(int).reindex(pats).values; pfold_all = man.groupby("patient_id").fold_id_rep01.first().astype(int)
def boots(y):
    rng = np.random.RandomState(SEED); out = []
    while len(out) < NB:
        s = rng.choice(len(y), len(y))
        if len(set(y[s])) > 1: out.append(s)
    return out
def cis(y, B, f, a, b=None): v = [f(y[s], a[s]) - (f(y[s], b[s]) if b is not None else 0) for s in B]; return [r3(np.percentile(v, 2.5)), r3(np.percentile(v, 97.5))]
RES = {"_spec": "docs/paper_plan_discovery.md @ 30b2317", "n_patients": int(len(d)), "endpoints": {}}
for ep, y in [("LGD2plus", d.y.values.astype(int)), ("E_HGD", d.y_hgd.values.astype(int))]:
    P = {a: d[a].values for a in ARMS}; B = boots(y); rng = np.random.RandomState(SEED); perms = [rng.permutation(len(y)) for _ in range(NB)]
    out = {"n": int(len(y)), "events": int(y.sum()), "arms": {a: {"auroc": r3(auc(y, P[a])), "auroc_ci": cis(y, B, auc, P[a]), "auprc": r3(ap(y, P[a])), "auprc_ci": cis(y, B, ap, P[a])} for a in ARMS}, "differences": {}}
    def diff(a, b):
        obs = auc(y, P[a]) - auc(y, P[b]); null = np.array([auc(y[ix], P[a]) - auc(y[ix], P[b]) for ix in perms]); return {"delta": r3(obs), "ci": cis(y, B, auc, P[a], P[b]), "perm_p": round(float((1 + (null >= obs).sum()) / (NB + 1)), 4)}, null
    for a, b in [("image_only", "cnv_only"), ("image_only", "cnv_km"), ("late_mean", "cnv_only"), ("late_mean_km", "cnv_km")] + [(c, "C2_grade_maxsofar") for c in ARMS if c.startswith("C2+")]: out["differences"][f"{a}_minus_{b}"], _ = diff(a, b)
    nulls = {}
    for f_ in FUS: out["differences"][f"{f_}_minus_image_only"], nulls[f_] = diff(f_, "image_only")
    mx = np.max(np.stack([nulls[f_] for f_ in FUS]), 0)
    for f_ in FUS: out["differences"][f"{f_}_minus_image_only"]["perm_p_selection_adjusted_max_over_5_fusion_arms"] = round(float((1 + (mx >= (auc(y, P[f_]) - auc(y, P["image_only"]))).sum()) / (NB + 1)), 4)
    RES["endpoints"][ep] = out; print("endpoint", ep, flush=True)
# FP analysis within discovery non-progressors
sp = pd.read_csv(ROW + "/followup_patient_scores.csv", dtype={"patient_id": str}).set_index("patient_id").reindex(pats); LT = pd.read_csv(ROW + "/later_disease_patient.csv", dtype={"patient_id": str}).set_index("patient_id").reindex(pats); PT = pd.read_csv(T + "/feasibility/closeout/swg_patient_table.csv", dtype=str).set_index("patient_id")
y = d.y.values.astype(int); neg = y == 0; bgr = pd.to_numeric(PT.baseline_grade.reindex(pats), errors="coerce").fillna(0).values; mxs = pd.to_numeric(PT.max_grade.reindex(pats), errors="coerce").fillna(0).values
laterL = ((LT.db_max_grade_after >= 2) | (LT.release_excluded_max_grade >= 2) | (LT.slidematch_max_grade_after >= 2)).values; laterH = ((LT.db_max_grade_after >= 3) | (LT.release_excluded_max_grade >= 3) | (LT.slidematch_max_grade_after >= 3) | (LT.hgd_table_entries_after > 0)).values
MODELS = ["late_mean", "image_only", "cnv_only", "cnv_km", "C2_grade_maxsofar", "clinical_3a", "v4_exploratory"]
SCORE_COL = {"clinical_3a": "C4_plus_surveillance_3a"}; sp0 = pd.read_csv(ROW + "/patient_scores_predictions.csv", dtype={"patient_id": str}).set_index("patient_id").reindex(pats)
def refit_pred(a):
    s = sp0[a].values if a == "v4_exploratory" else sp[SCORE_COL.get(a, a)].values; pred = np.zeros(len(y), int)
    for k in range(1, 6):
        tr = pfold != k; te = ~tr; ss = np.sort(np.unique(s[tr]))[::-1]; t = ss[-1]
        for t_ in ss:
            if (s[tr][y[tr] == 1] >= t_).mean() >= 0.8: t = t_; break
        pred[te] = (s[te] >= t).astype(int)
    return pred
def fpa(pred):
    fp = (pred == 1) & neg; tn = (pred == 0) & neg; out = {"n_FP": int(fp.sum()), "n_TN": int(tn.sum())}
    for nm, lab in [("later_HGDplus", laterH), ("later_LGDplus", laterL)]:
        k1, k0 = int(lab[fp].sum()), int(lab[tn].sum()); o = {"FP": [k1, int(fp.sum())], "TN": [k0, int(tn.sum())], "fisher_p": pf(fisher_exact([[k1, fp.sum() - k1], [k0, tn.sum() - k0]])[1]) if fp.sum() and tn.sum() else None}
        try:
            Xd = pd.DataFrame({"FP": fp[neg].astype(float), "baseline_grade": bgr[neg], "max_grade_so_far": mxs[neg]}); m_ = sm.Logit(lab[neg].astype(int), sm.add_constant(Xd)).fit(disp=0, maxiter=300); ci_ = np.exp(m_.conf_int().loc["FP"]); o["OR_FP_adjusted"] = r3(np.exp(m_.params["FP"])); o["ci"] = [r3(ci_[0]), r3(ci_[1])]; o["p"] = pf(m_.pvalues["FP"])
            if not np.isfinite(ci_).all() or ci_[1] > 1e4: o["note"] = "near-separation: CI unreliable"
        except Exception as e: o["logit_error"] = str(e)[:80]
        out[nm] = o
    return out
RES["fp_analysis_discovery_nonprogressors"] = {"n_nonprogressors": int(neg.sum()), "later_HGDplus_total": int(laterH[neg].sum()), "later_LGDplus_total": int(laterL[neg].sum()), "models": {a: {"pooled_thresholds": fpa(sp[f"pred_{a}"].values.astype(int)), "discovery_refit_thresholds": fpa(refit_pred(a))} for a in MODELS}}
json.dump(RES, open(AGG + "/discovery_main.json", "w"), indent=1, default=str); print("DISCOVERY DONE", flush=True)

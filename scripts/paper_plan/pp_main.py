"""Paper plan items 1-5, 7 (summary), 8a, 9, 10 (tables + pack manifest), pre-specified in docs/paper_plan_answers.md @ db236a0.
Frozen release only (item 0 found no confirmed errors). Aggregates -> results/paper_plan/main_items.json (+ figs);
row-level material -> feasibility/paper_plan/ (cluster only). Env: erin."""
import glob, json, os, re, sys, warnings, numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr, mannwhitneyu, fisher_exact
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import multivariate_logrank_test, logrank_test
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
F = "/mnt/scratche/slow/fmlab/zuberi01/phd/barretts_retraining/barretts_training/analysis/chapter1_lgd2_final_pre_event_20260713_final"; R = F + "/training_final_nested_cv_v1"
T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"; S = "/mnt/scratche/fast/fmlab/datasets/imaging/SWGCohort"; E = "/mnt/scratche/slow/fmlab/zuberi01/barretts_db_export"
ROW = T + "/feasibility/paper_plan"; AGG = os.environ.get("OUTDIR", T + "/results/paper_plan"); FIG = T + "/results/paper_plan/figs"; os.makedirs(ROW, exist_ok=True); os.makedirs(AGG, exist_ok=True); os.makedirs(FIG + "/km", exist_ok=True)
NB = 2000; SEED = 0
def auc(y, s):
    y = np.asarray(y).astype(int); r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1; return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if 0 < n1 < len(y) else float("nan")
def ap(y, s): return float(average_precision_score(y, s)) if 0 < np.sum(y) < len(y) else float("nan")
def r3(x): return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 3)
def pf(p): return None if p is None or (isinstance(p, float) and np.isnan(p)) else ("<0.001" if p < 0.001 else round(float(p), 3))
RES = {"_spec": "docs/paper_plan_answers.md @ db236a0", "_release": "frozen only (item 0: no confirmed errors)"}
# ------------------------------------------------------------------ data
man = pd.read_csv(F + "/training_manifest.csv", dtype=str).set_index("sample_id"); coh = pd.read_csv(F + "/pre_event_cohort.csv", dtype=str).set_index("SampleID"); cohall = coh.copy(); coh = coh.loc[man.index]
man["y"] = man.y_progressor.astype(int); man["fold"] = man.fold_id_rep01.astype(int); man["date"] = pd.to_datetime(coh.Date, errors="coerce"); man["grade"] = pd.to_numeric(coh.Label, errors="coerce").fillna(0)
man["maxsofar"] = pd.to_numeric(coh.MaxPathologySoFar, errors="coerce").fillna(0); man["streak"] = pd.to_numeric(coh.LGDStreakSoFar, errors="coerce").fillna(0); man["bidx"] = pd.to_numeric(coh.BiopsyIndex, errors="coerce"); man["dsp"] = pd.to_numeric(coh.DaysSincePreviousBiopsy, errors="coerce").fillna(0); man["year"] = man.date.dt.year
man["days_to_next"] = pd.to_numeric(coh.DaysToNextBiopsy, errors="coerce"); man["next_label"] = pd.to_numeric(coh.NextBiopsyLabel, errors="coerce"); man["slide"] = coh.ImageAbsPath.map(os.path.basename); man["participant"] = coh.participant_id.str.replace(r"\.0$", "", regex=True)
FAM = {"cnv_only": "CNV", "image_only": "WSI", "early_fusion": "Early fusion", "intermediate_fusion": "Intermediate fusion", "late_mean": "Late fusion (mean)", "coattention_fusion": "Co-attention fusion (extra)", "late_stack_logit": "Late stack-logit fusion (extra)"}
def oof(fam): return pd.concat([pd.read_csv(f, dtype={"sample_id": str}) for f in glob.glob(f"{R}/{fam}/fold*/outer_test_predictions.csv")]).set_index("sample_id").y_prob.reindex(man.index).values
for fam in FAM: man[fam] = oof(fam)
ids = man.index.values; y_s = man.y.values; folds = man.fold.values; pid = man.patient_id.values
pairs = pd.read_csv(T + "/feasibility/closeout/erin_swg_pairs.csv", dtype=str); OV = set(pairs.swg_patient_id)
PT = pd.read_csv(T + "/feasibility/closeout/swg_patient_table.csv", dtype=str).set_index("patient_id")   # closeout item C/E patient table (commit 602a40e)
inner = {k: pd.read_csv(f"{R}/image_only/fold{k}/inner_fold_assignments.csv", dtype=str).set_index("patient_id").inner_fold.astype(int) for k in range(1, 6)}
def patient(cols, rows=None):
    d = man.assign(**cols) if cols else man; d = d if rows is None else d[rows]; g = d.groupby("patient_id"); return g.y.max().values.astype(int), {c: g[c].max().values for c in (cols or {})}, g.size().index.values, g.fold.first().values
def bidx(y):
    rng = np.random.RandomState(SEED); out = []
    while len(out) < NB:
        s = rng.choice(len(y), len(y))
        if len(set(y[s])) > 1: out.append(s)
    return out
def ci_stat(y, B, f, a, b=None): v = [f(y[s], a[s]) - (f(y[s], b[s]) if b is not None else 0) for s in B]; return [r3(np.percentile(v, 2.5)), r3(np.percentile(v, 97.5))]
def perm_p(y, a, b, n=NB):
    rng = np.random.RandomState(SEED); obs = auc(y, a) - auc(y, b); null = np.array([auc(y[ix], a) - auc(y[ix], b) for ix in (rng.permutation(len(y)) for _ in range(n))]); return round(float((1 + (null >= obs).sum()) / (n + 1)), 4), null
# ------------------------------------------------------------------ item 3 clinical arms (nested CV, release folds + inner folds)
def nested_logistic(X, y, fl, pids, Cs=(0.01, 0.1, 1.0, 10.0)):
    p = np.zeros(len(y)); chosen = {}
    for k in np.unique(fl):
        te = fl == k; tr = ~te; mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9; Xtr = (X[tr] - mu) / sd; inn = inner[int(k)].reindex(pids[tr]).values; best, bestC = -1, 1.0
        for C in Cs:
            pv = np.zeros(tr.sum())
            for j in np.unique(inn): v = inn == j; pv[v] = LogisticRegression(C=C, max_iter=5000).fit(Xtr[~v], y[tr][~v]).predict_proba(Xtr[v])[:, 1]
            g = pd.DataFrame({"p": pids[tr], "s": pv, "y": y[tr]}).groupby("p"); a = auc(g.y.max().values, g.s.max().values)
            if a > best: best, bestC = a, C
        p[te] = LogisticRegression(C=bestC, max_iter=5000).fit(Xtr, y[tr]).predict_proba((X[te] - mu) / sd)[:, 1]; chosen[int(k)] = bestC
    return p, chosen
CLIN3A = ["grade", "maxsofar", "streak", "bidx", "dsp", "year"]; cov = {c: r3(man[c].notna().groupby(pid).all().mean()) for c in CLIN3A}; use3a = [c for c in CLIN3A if cov[c] >= 0.9]
man["clinical_3a"], C3a = nested_logistic(man[use3a].fillna(0).values.astype(float), y_s, folds, pid)
dem = pd.read_csv(S + "/Demographics_full.csv", dtype=str); dem.columns = [c.strip() for c in dem.columns]; dem = dem.set_index("Study Number")
man["age"] = man.year - pd.to_datetime(man.patient_id.map(dem["Date of birth"]), errors="coerce").dt.year.values; man["sex_m"] = man.patient_id.map(dem.Sex).map({"M": 1, "F": 0}); man["pragueC"] = pd.to_numeric(man.patient_id.map(dem["Circumference"]), errors="coerce"); man["pragueM"] = pd.to_numeric(man.patient_id.map(dem["Maximal"]), errors="coerce"); man["smoke"] = man.patient_id.map(dem["Smoking Status"]).map({"Y": 1, "N": 0})
CLIN3B = use3a + ["age", "sex_m", "pragueC", "pragueM", "smoke"]; cc = man[CLIN3B].notna().all(axis=1).values; cc_pats = set(man.patient_id[cc]); cc = man.patient_id.isin(cc_pats).values & cc
man["clinical_3b"] = np.nan; sub3b = man[cc]
if sub3b.patient_id.nunique() >= 20 and sub3b.groupby("patient_id").y.max().sum() >= 5:
    p3b, C3b = nested_logistic(sub3b[CLIN3B].values.astype(float), sub3b.y.values, sub3b.fold.values, sub3b.patient_id.values); man.loc[cc, "clinical_3b"] = p3b
else: C3b = {}
RES["item3"] = {"3a_covariates_all": CLIN3A, "3a_patient_coverage": cov, "3a_used": use3a, "3a_C_per_fold": C3a, "3b_covariates": CLIN3B, "3b_patients_in_demographics": int(man.patient_id.isin(dem.index).groupby(pid).first().sum()), "3b_complete_case_patients": int(len(cc_pats)), "3b_complete_case_rows": int(cc.sum()), "3b_C_per_fold": C3b, "model": "L2 logistic, C by inner CV (release inner folds, patient-level AUROC), standardisation on outer training rows, patient = max over rows"}
# ------------------------------------------------------------------ items 2 & 4: main table on subsets
ARMS = ["clinical_3a"] + list(FAM); FUS = ["early_fusion", "intermediate_fusion", "late_mean", "coattention_fusion", "late_stack_logit"]
def table_on(rows, label):
    y, P, pats, pf_ = patient({a: man[a].values for a in ARMS}, rows); B = bidx(y); out = {"label": label, "n": int(len(y)), "events": int(y.sum()), "arms": {}, "differences": {}}
    for a in ARMS: out["arms"][a] = {"auroc": r3(auc(y, P[a])), "auroc_ci": ci_stat(y, B, auc, P[a]), "auprc": r3(ap(y, P[a])), "auprc_ci": ci_stat(y, B, ap, P[a])}
    rng = np.random.RandomState(SEED); perms = [rng.permutation(len(y)) for _ in range(NB)]
    for ref in ["image_only", "cnv_only"]:
        nulls = {}
        for f_ in FUS:
            obs = auc(y, P[f_]) - auc(y, P[ref]); null = np.array([auc(y[ix], P[f_]) - auc(y[ix], P[ref]) for ix in perms]); nulls[f_] = null
            out["differences"][f"{f_}_vs_{ref}"] = {"delta_auroc": r3(obs), "ci": ci_stat(y, B, auc, P[f_], P[ref]), "delta_auprc": r3(ap(y, P[f_]) - ap(y, P[ref])), "auprc_ci": ci_stat(y, B, ap, P[f_], P[ref]), "perm_p_naive": round(float((1 + (null >= obs).sum()) / (NB + 1)), 4)}
        mx = np.max(np.stack([nulls[f_] for f_ in FUS]), axis=0)
        for f_ in FUS: out["differences"][f"{f_}_vs_{ref}"]["perm_p_selection_adjusted_max_over_5_fusion_arms"] = round(float((1 + (mx >= (auc(y, P[f_]) - auc(y, P[ref]))).sum()) / (NB + 1)), 4)
    # item 2
    obs = auc(y, P["image_only"]) - auc(y, P["cnv_only"]); pp_, _ = perm_p(y, P["image_only"], P["cnv_only"])
    out["item2_image_vs_cnv"] = {"delta_auroc": r3(obs), "ci": ci_stat(y, B, auc, P["image_only"], P["cnv_only"]), "perm_p": pp_, "delta_auprc": r3(ap(y, P["image_only"]) - ap(y, P["cnv_only"])), "auprc_ci": ci_stat(y, B, ap, P["image_only"], P["cnv_only"])}
    return out, (y, P, pats, pf_)
sub_ov = man.patient_id.isin(OV).values
RES["item4_tables"] = {}; TAB = {}
for label, rows in [("all_150", None), ("never_in_ERIN_96", ~sub_ov), ("also_in_ERIN_54", sub_ov)]:
    RES["item4_tables"][label], TAB[label] = table_on(rows, label); print("table", label, flush=True)
# 3b subset table
if C3b:
    y, P, pats, _ = patient({a: man[a].values for a in ARMS + ["clinical_3b"]}, cc); B = bidx(y)
    RES["item3"]["3b_subset_table"] = {"n": int(len(y)), "events": int(y.sum()), "arms": {a: {"auroc": r3(auc(y, P[a])), "auroc_ci": ci_stat(y, B, auc, P[a]), "auprc": r3(ap(y, P[a])), "auprc_ci": ci_stat(y, B, ap, P[a])} for a in ["clinical_3b"] + ARMS}, "delta_3b_minus_3a": ci_stat(y, B, auc, P["clinical_3b"], P["clinical_3a"])}
# item 1: CNV-only on Killcoyne-overlap patients vs rest
kk = pd.read_excel("/mnt/scratche/slow/fmlab/zuberi01/phd/killcoyne_data_from_paper/41591_2020_1033_MOESM4_ESM.xlsx", sheet_name="Supporting data for Figure 2a", header=1); cx = pd.read_csv(F + "/feature_views/cnv/cx.csv", dtype=str).set_index("sample_id").reindex(man.index); man["in_k"] = cx.cnv_id.isin(set(kk.Samplename.astype(str))).values
kpat = man.groupby("patient_id").in_k.any(); RES["item1_overlap"] = {}
for label, sel in [("in_killcoyne_discovery_table", kpat[kpat].index), ("not_in_table", kpat[~kpat].index)]:
    rows = man.patient_id.isin(sel).values; y, P, _, _ = patient({"cnv_only": man.cnv_only.values, "image_only": man.image_only.values, "late_mean": man.late_mean.values}, rows); B = bidx(y)
    RES["item1_overlap"][label] = {"n": int(len(y)), "events": int(y.sum()), "rows": int(rows.sum()), **{a: [r3(auc(y, P[a]))] + ci_stat(y, B, auc, P[a]) for a in P}}
# ------------------------------------------------------------------ operating point (sens >= 0.80 on training-fold patients)
def op_threshold(y_tr, s_tr, target=0.80):
    ss = np.sort(np.unique(s_tr))[::-1]
    for t in ss:
        if (s_tr[y_tr == 1] >= t).mean() >= target: return t
    return ss[-1]
noov = pd.read_csv(T + "/feasibility/runs/p32_head_noov/output/swg_imputed_fields.csv", index_col=0); noov.index = noov.index.astype(str); gh = noov.grade_LGDplus.reindex(man.index).values
def zf(v):
    v = np.asarray(v, float); o = np.zeros(len(v))
    for f_ in np.unique(folds): te = folds == f_; o[te] = (v[te] - v[te].mean()) / (v[te].std() + 1e-9)
    return o
man["v4_exploratory"] = (zf(man.image_only) + zf(man.cnv_only) + zf(gh)) / 3
OPM = ["late_mean", "image_only", "cnv_only", "clinical_3a", "v4_exploratory"]; y, P, pats, pfold = patient({a: man[a].values for a in OPM}); pat_sub = np.array(["also_in_ERIN" if p in OV else "never_in_ERIN" for p in pats])
PRED = {}; RES["operating_point"] = {}
for a in OPM:
    pred = np.zeros(len(y), int); per = {}
    for k in range(1, 6):
        tr = pfold != k; te = ~tr; t = op_threshold(y[tr], P[a][tr]); pred[te] = (P[a][te] >= t).astype(int); per[k] = {"threshold": r3(t), "n_test": int(te.sum()), "events": int(y[te].sum()), "sensitivity": r3((pred[te][y[te] == 1] == 1).mean()) if y[te].sum() else None, "specificity": r3((pred[te][y[te] == 0] == 0).mean())}
    PRED[a] = pred; RES["operating_point"][a] = {"per_fold": per, "pooled_sensitivity": r3((pred[y == 1] == 1).mean()), "pooled_specificity": r3((pred[y == 0] == 0).mean()), "flagged": int(pred.sum()), "n": int(len(y)), "events": int(y.sum())}
pd.DataFrame({"patient_id": pats, "y": y, **{a: P[a] for a in OPM}, **{f"pred_{a}": PRED[a] for a in OPM}}).to_csv(ROW + "/patient_scores_predictions.csv", index=False)
# ------------------------------------------------------------------ item 5 risk groups
def wilson(k, n, z=1.96):
    if n == 0: return [None, None]
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d; return [r3(c - h), r3(c + h)]
first = pd.to_datetime(PT.first_date.reindex(pats)); t_event = pd.to_numeric(PT.days_first_to_event.reindex(pats), errors="coerce").values; fu = pd.to_numeric(PT.followup_months_first_to_last_biopsy.reindex(pats), errors="coerce").values * 30.44
time = np.where(y == 1, t_event, fu); time = np.where(np.isnan(time), 0, time); time = np.maximum(time, 0.5); event = y.copy()
RES["item5"] = {"time_definition": "progressors: earliest release row -> endpoint biopsy (closeout item E); non-progressors: earliest row -> last biopsy (max MonthsBeforeLastBiopsy x 30.44 d), censored", "groups": {}, "figures": {}}
GROUPS = {}
def groups_tertile(a):
    g = np.zeros(len(y), int)
    for k in range(1, 6):
        tr = pfold != k; te = ~tr; q1, q2 = np.quantile(P[a][tr], [1 / 3, 2 / 3]); g[te] = np.where(P[a][te] > q2, 2, np.where(P[a][te] > q1, 1, 0))
    return g
def group_stats(a, g, name):
    out = {"n": int(len(y)), "events": int(y.sum()), "groups": {}}
    for gi, gn in enumerate(["low", "moderate", "high"]):
        m = g == gi; k_ = int(y[m].sum()); out["groups"][gn] = {"n": int(m.sum()), "events": k_, "rate": r3(k_ / m.sum()) if m.sum() else None, "wilson_ci": wilson(k_, int(m.sum()))}
    lo, hi = g == 0, g == 2; a_, b_, c_, d_ = y[hi].sum() + .5, (1 - y[hi]).sum() + .5, y[lo].sum() + .5, (1 - y[lo]).sum() + .5; out["odds_ratio_high_vs_low_haldane"] = r3((a_ * d_) / (b_ * c_)); out["or_log_se"] = r3(np.sqrt(1 / a_ + 1 / b_ + 1 / c_ + 1 / d_))
    df = pd.DataFrame({"T": time, "E": event, "g": g, "mod": (g == 1).astype(int), "high": (g == 2).astype(int)})
    try:
        out["logrank_p_3groups"] = pf(multivariate_logrank_test(df["T"], df.g, df.E).p_value); cph = CoxPHFitter().fit(df[["T", "E", "mod", "high"]], "T", "E"); out["cox_hr_high_vs_low"] = [r3(np.exp(cph.params_["high"]))] + [r3(v) for v in np.exp(cph.confidence_intervals_.loc["high"]).values]; out["cox_hr_moderate_vs_low"] = [r3(np.exp(cph.params_["mod"]))] + [r3(v) for v in np.exp(cph.confidence_intervals_.loc["mod"]).values]
    except Exception as e: out["cox_error"] = str(e)[:120]
    fig, ax = plt.subplots(figsize=(5, 4)); km = KaplanMeierFitter()
    for gi, gn, c in [(0, "low", "#1f77b4"), (1, "moderate", "#ff7f0e"), (2, "high", "#d62728")]:
        m = g == gi
        if m.sum(): km.fit(df["T"][m] / 365.25, df.E[m], label=f"{gn} (n={m.sum()}, events={int(y[m].sum())})").plot_survival_function(ax=ax, ci_show=False, color=c)
    ax.set_xlabel("years from earliest release row"); ax.set_ylabel("endpoint-free"); ax.set_title(f"{a} / {name}"); fn = f"{FIG}/km/km_{a}_{name}.png"; fig.tight_layout(); fig.savefig(fn, dpi=130); plt.close(fig); out["figure"] = fn.replace(T + "/", ""); return out
for a in ["clinical_3a", "cnv_only", "image_only", "late_mean"]:
    g = groups_tertile(a); GROUPS[(a, "tertile")] = g; RES["item5"]["groups"][f"{a}__tertile_trainfold"] = group_stats(a, g, "tertile")
for a in ["cnv_only", "late_mean"]:
    g = np.where(P[a] >= 0.5, 2, np.where(P[a] > 0.3, 1, 0)); GROUPS[(a, "killcoyne")] = g; RES["item5"]["groups"][f"{a}__killcoyne_fixed_0.3_0.5"] = group_stats(a, g, "killcoyne")
ct = pd.crosstab(GROUPS[("cnv_only", "tertile")], GROUPS[("late_mean", "tertile")]); mv = GROUPS[("cnv_only", "tertile")] != GROUPS[("late_mean", "tertile")]
RES["item5"]["crosstab_cnv_vs_late_tertile"] = {"counts_rows_cnv_cols_late": ct.values.tolist(), "movers": int(mv.sum()), "movers_progression_rate": r3(y[mv].mean()) if mv.sum() else None, "stayers_progression_rate": r3(y[~mv].mean()), "moved_up_n_rate": [int((GROUPS[("late_mean", "tertile")] > GROUPS[("cnv_only", "tertile")]).sum()), r3(y[GROUPS[("late_mean", "tertile")] > GROUPS[("cnv_only", "tertile")]].mean())], "moved_down_n_rate": [int((GROUPS[("late_mean", "tertile")] < GROUPS[("cnv_only", "tertile")]).sum()), r3(y[GROUPS[("late_mean", "tertile")] < GROUPS[("cnv_only", "tertile")]].mean())]}
RES["item5"]["killcoyne_threshold_source"] = "Killcoyne et al. 2020 Nat Med, results text: low Pr <= 0.3, moderate 0.3-0.5, high Pr >= 0.5 (p. 1727)"
# ------------------------------------------------------------------ item 8a ranking
pc, pl = rankdata(P["cnv_only"]) / len(y) * 100, rankdata(P["late_mean"]) / len(y) * 100; dr = pl - pc
gc, gl = GROUPS[("cnv_only", "tertile")], GROUPS[("late_mean", "tertile")]
def nri(y_, gc_, gl_): up = gl_ > gc_; dn = gl_ < gc_; ev = y_ == 1; return ((up[ev].mean() - dn[ev].mean()) if ev.sum() else np.nan) + ((dn[~ev].mean() - up[~ev].mean()) if (~ev).sum() else np.nan)
B = bidx(y); nri_b = [nri(y[s], gc[s], gl[s]) for s in B]
RES["item8a"] = {"spearman_cnv_vs_late_patient_scores": r3(spearmanr(P["cnv_only"], P["late_mean"]).correlation), "n": int(len(y)), "events": int(y.sum()), "progressors_up_gt20pct": int(((dr > 20) & (y == 1)).sum()), "progressors_down_gt20pct": int(((dr < -20) & (y == 1)).sum()), "nonprogressors_up_gt20pct": int(((dr > 20) & (y == 0)).sum()), "nonprogressors_down_gt20pct": int(((dr < -20) & (y == 0)).sum()),
                 "median_abs_rank_change_pct": r3(np.median(np.abs(dr))), "categorical_NRI_late_vs_cnv_tertile_groups": [r3(nri(y, gc, gl)), r3(np.percentile(nri_b, 2.5)), r3(np.percentile(nri_b, 97.5))], "event_NRI": r3((gl > gc)[y == 1].mean() - (gl < gc)[y == 1].mean()), "nonevent_NRI": r3((gl < gc)[y == 0].mean() - (gl > gc)[y == 0].mean())}
# ------------------------------------------------------------------ item 9 false positives (later disease)
pt = pd.read_csv(E + "/pathology_text_normalised_full.csv", dtype=str, usecols=["participant_id", "receiveddatetime", "highestgradedysconf"]); pt["d"] = pd.to_datetime(pt.receiveddatetime, errors="coerce"); pt["g"] = pd.to_numeric(pt.highestgradedysconf, errors="coerce").map({2: 0, 3: 1, 4: 2, 5: 3, 6: 4, 8: 4})
hg = pd.read_parquet(E + "/hgd_pathology_table.parquet"); hg["d"] = pd.to_datetime(hg.receiveddatetime, dayfirst=True, errors="coerce"); hg["participant_id"] = hg.participant_id.astype(str)
sm = pd.read_csv(S + "/slide_matching.csv", dtype=str); sm["d"] = pd.to_datetime(sm.EndoscopyDate, errors="coerce"); sm["g"] = sm.Pathology.str.upper().map({"NDBE": 0, "BE": 0, "ID": 1, "IND": 1, "LGD": 2, "HGD": 3, "IMC": 4, "OAC": 4})
last_row = man.groupby("patient_id").date.max().reindex(pats); part = man.groupby("patient_id").participant.first().reindex(pats); excl = cohall[cohall.exclusion_reason.notna() & (cohall.exclusion_reason != "")]; excl["d"] = pd.to_datetime(excl.Date, errors="coerce"); excl["g"] = pd.to_numeric(excl.Label, errors="coerce")
later = []
for i, p in enumerate(pats):
    l = last_row.iloc[i]; rec = {"patient_id": p, "y": int(y[i]), "last_release_row": l}
    db = pt[(pt.participant_id == part.iloc[i]) & (pt.d > l)].sort_values("d"); rec["db_reports_after"] = int(len(db)); rec["db_followup_days"] = float((db.d.max() - l).days) if len(db) else 0.0; rec["db_max_grade_after"] = float(db.g.max()) if db.g.notna().any() else np.nan
    rec["db_first_lgdplus_days"] = float((db[db.g >= 2].d.min() - l).days) if (db.g >= 2).any() else np.nan; rec["db_first_hgdplus_days"] = float((db[db.g >= 3].d.min() - l).days) if (db.g >= 3).any() else np.nan
    ex = excl[(excl.PatientID.astype(str) == str(man.PatientID.iloc[0]) if False else excl.index.isin([])) ]  # placeholder replaced below
    exr = excl[(excl.get("PatientID_real", pd.Series(dtype=str)) == p) if "PatientID_real" in excl else excl.index.isin([])]; rec["release_excluded_rows"] = int(len(exr)); rec["release_excluded_max_grade"] = float(exr.g.max()) if len(exr) and exr.g.notna().any() else np.nan; rec["release_excluded_reasons"] = "|".join(sorted(set(exr.exclusion_reason))) if len(exr) else ""
    h = hg[(hg.participant_id == part.iloc[i]) & (hg.d > l)]; rec["hgd_table_entries_after"] = int(len(h))
    s_ = sm[(sm.AlternateID == p) & (sm.d > l)]; rec["slidematch_rows_after"] = int(len(s_)); rec["slidematch_max_grade_after"] = float(s_.g.max()) if len(s_) and s_.g.notna().any() else np.nan
    later.append(rec)
LT = pd.DataFrame(later).set_index("patient_id"); LT.to_csv(ROW + "/later_disease_patient.csv")
RES["item9"] = {"sources": {"barretts_db_pathology_reports": {"participants_linked": int(part.notna().sum()), "date_coverage": [str(pt.d.min().date()), str(pt.d.max().date())]}, "release_excluded_rows_for_patient": "pre_event_cohort.csv rows with exclusion_reason (at_event/post_event/endpoint_not_evaluable)", "hgd_pathology_table": {"rows": int(len(hg)), "date_coverage": [str(hg.d.min().date()), str(hg.d.max().date())]}, "slide_matching_rows_after_last_release_row": "SWGCohort/slide_matching.csv Pathology"}, "models": {}}
cxp = pd.to_numeric(PT.cx_max.reindex(pats), errors="coerce").values; bgr = pd.to_numeric(PT.baseline_grade.reindex(pats), errors="coerce").values; p53 = PT.p53_ihc_any_aberrant.reindex(pats).values
for a in OPM:
    neg = y == 0; fp = (PRED[a] == 1) & neg; tn = (PRED[a] == 0) & neg; m_ = {"n_FP": int(fp.sum()), "n_TN": int(tn.sum())}
    anyl = (LT.db_max_grade_after >= 2) | (LT.release_excluded_max_grade >= 2) | (LT.slidematch_max_grade_after >= 2); anyh = (LT.db_max_grade_after >= 3) | (LT.release_excluded_max_grade >= 3) | (LT.slidematch_max_grade_after >= 3) | (LT.hgd_table_entries_after > 0)
    for nm, flag in [("later_LGDplus", anyl.values), ("later_HGDplus_or_hgd_table", anyh.values)]:
        k1, k0 = int(flag[fp].sum()), int(flag[tn].sum()); m_[nm] = {"FP": [k1, int(fp.sum()), r3(k1 / max(fp.sum(), 1))], "TN": [k0, int(tn.sum()), r3(k0 / max(tn.sum(), 1))], "fisher_p": pf(fisher_exact([[k1, fp.sum() - k1], [k0, tn.sum() - k0]])[1]) if fp.sum() and tn.sum() else None}
    m_["db_followup_days_median_FP_TN"] = [r3(np.median(LT.db_followup_days.values[fp])) if fp.sum() else None, r3(np.median(LT.db_followup_days.values[tn])) if tn.sum() else None, pf(mannwhitneyu(LT.db_followup_days.values[fp], LT.db_followup_days.values[tn]).pvalue) if fp.sum() > 1 and tn.sum() > 1 else None]
    m_["any_db_report_after_FP_TN"] = [int((LT.db_reports_after.values[fp] > 0).sum()), int((LT.db_reports_after.values[tn] > 0).sum())]
    for nm, v in [("baseline_grade_mean", bgr), ("cx_max_median", cxp)]: m_[nm] = [r3(np.nanmean(v[fp]) if nm.endswith("mean") else np.nanmedian(v[fp])) if fp.sum() else None, r3(np.nanmean(v[tn]) if nm.endswith("mean") else np.nanmedian(v[tn])) if tn.sum() else None, pf(mannwhitneyu(v[fp][~np.isnan(v[fp])], v[tn][~np.isnan(v[tn])]).pvalue) if fp.sum() > 1 and tn.sum() > 1 else None]
    m_["p53_aberrant_FP_TN_of_with_ihc"] = [[int((p53[fp] == "aberrant").sum()), int(pd.notna(p53[fp]).sum())], [int((p53[tn] == "aberrant").sum()), int(pd.notna(p53[tn]).sum())]]
    tt = np.where(np.isnan(LT.db_first_lgdplus_days.values), LT.db_followup_days.values, LT.db_first_lgdplus_days.values); ee = (~np.isnan(LT.db_first_lgdplus_days.values)).astype(int); ok = neg & (tt > 0)
    if ee[ok & fp].sum() + ee[ok & tn].sum() >= 5 and (ok & fp).sum() and (ok & tn).sum():
        m_["km_logrank_p_time_to_first_later_LGDplus_FP_vs_TN"] = pf(logrank_test(tt[ok & fp], tt[ok & tn], ee[ok & fp], ee[ok & tn]).p_value); fig, ax = plt.subplots(figsize=(5, 4)); km = KaplanMeierFitter()
        for m2, lab, c in [(ok & fp, "FP", "#d62728"), (ok & tn, "TN", "#1f77b4")]: km.fit(tt[m2] / 365.25, ee[m2], label=f"{lab} (n={m2.sum()}, later LGD+={ee[m2].sum()})").plot_survival_function(ax=ax, ci_show=False, color=c)
        ax.set_xlabel("years after last release row"); ax.set_ylabel("free of later LGD+ (DB)"); ax.set_title(a); fn = f"{FIG}/km/later_lgd_FP_TN_{a}.png"; fig.tight_layout(); fig.savefig(fn, dpi=130); plt.close(fig); m_["km_figure"] = fn.replace(T + "/", "")
    else: m_["km"] = "not drawn (fewer than 5 later LGD+ events among non-progressors with follow-up)"
    RES["item9"]["models"][a] = m_
# ------------------------------------------------------------------ item 10 false negatives
qc = pd.read_csv(T + "/feasibility/closeout/swg_cnv_qc.csv"); cxr = pd.read_csv(F + "/feature_views/cnv/cx.csv", dtype=str).set_index("sample_id").reindex(man.index); man["cnv_id"] = cxr.cnv_id.values; man = man.join(qc.set_index("cnv_id")[["noise_mapd", "n_segments", "frac_altered_0p15"]], on="cnv_id")
fd = pd.read_csv(S + "/sWGS_777_samples_cleaned_202401_Leanne_fullDetails (3) (1).csv", dtype=str); fd.columns = [c.strip().replace("\n", " ") for c in fd.columns]; fd = fd.drop_duplicates("combined_name").set_index("combined_name"); man["reads"] = pd.to_numeric(man.cnv_id.map(fd["Number of reads"]).str.replace(",", ""), errors="coerce")
import h5py
ui = pd.read_csv(F + "/feature_views/uni2/uni2_index.csv", dtype=str).set_index("sample_id").reindex(man.index); kept = []; grid = []
for b in ui.image_basename:
    p_ = f"{S}/features_uni2h_05um/{os.path.splitext(str(b))[0]}.h5"
    if os.path.exists(p_):
        with h5py.File(p_) as h: kept.append(float(h.attrs.get("kept_tiles", np.nan))); grid.append(float(h.attrs.get("grid_tiles", np.nan)))
    else: kept.append(np.nan); grid.append(np.nan)
man["tiles_kept"] = kept; man["tissue_frac"] = np.array(kept) / np.array(grid)
slm = pd.read_csv(T + "/feasibility/closeout/swg_slide_meta.csv", dtype=str).set_index("sample_id").reindex(man.index); man["scanner"] = slm["tiff.Model"].values
g = man.groupby("patient_id"); PF = pd.DataFrame({"n_rows": g.size(), "first_to_event": pd.to_numeric(PT.days_first_to_event, errors="coerce"), "endpoint_label": man[man.y == 1].groupby("patient_id").next_label.max(), "baseline_grade": pd.to_numeric(PT.baseline_grade, errors="coerce"), "max_sofar": g.maxsofar.max(), "cx": pd.to_numeric(PT.cx_max, errors="coerce"),
                                       "noise": g.noise_mapd.mean(), "segments": g.n_segments.mean(), "frac_alt": g.frac_altered_0p15.mean(), "reads": g.reads.mean(), "tiles_kept": g.tiles_kept.mean(), "tissue_frac": g.tissue_frac.mean(), "scanner": g.scanner.agg(lambda s: s.mode().iloc[0] if s.notna().any() else None), "p53": PT.p53_ihc_any_aberrant}).reindex(pats); PF["subgroup"] = pat_sub
RES["item10"] = {"models": {}, "slide_qc_note": "no blur/focus QC exists for SWG slides; tiles_kept and tissue fraction (kept/grid) from the 0.5 um h5 attributes are the available slide-level metrics"}
def cmp(mask1, mask2, lab1, lab2):
    out = {}
    for c in ["first_to_event", "n_rows", "baseline_grade", "max_sofar", "cx", "noise", "segments", "frac_alt", "reads", "tiles_kept", "tissue_frac"]:
        a_, b_ = PF[c].values[mask1].astype(float), PF[c].values[mask2].astype(float); a_, b_ = a_[~np.isnan(a_)], b_[~np.isnan(b_)]
        out[c] = {lab1: [r3(np.median(a_)) if len(a_) else None, int(len(a_))], lab2: [r3(np.median(b_)) if len(b_) else None, int(len(b_))], "mannwhitney_p": pf(mannwhitneyu(a_, b_).pvalue) if len(a_) > 1 and len(b_) > 1 else None}
    for c in ["endpoint_label", "scanner", "subgroup", "p53"]:
        t = pd.crosstab(PF[c].fillna("missing").values[mask1 | mask2], np.where(mask1[mask1 | mask2], lab1, lab2)); out[c] = {"table": {str(k): {str(kk): int(vv) for kk, vv in v.items()} for k, v in t.to_dict().items()}, "fisher_p": pf(fisher_exact(t.values)[1]) if t.shape == (2, 2) else None}
    return out
for a in OPM:
    pos = y == 1; fn_ = (PRED[a] == 0) & pos; tp = (PRED[a] == 1) & pos; RES["item10"]["models"][a] = {"n_FN": int(fn_.sum()), "n_TP": int(tp.sum()), "comparison": cmp(fn_, tp, "FN", "TP")}
# review pack manifest (late_mean): all FN slides (all rows of FN patients), equal numbers of TP and TN slides drawn with RandomState(0), shuffled
fn_p = pats[(PRED["late_mean"] == 0) & (y == 1)]; tp_p = pats[(PRED["late_mean"] == 1) & (y == 1)]; tn_p = pats[(PRED["late_mean"] == 0) & (y == 0)]
rows_fn = man[man.patient_id.isin(fn_p)]; nsl = len(rows_fn); rng = np.random.RandomState(SEED)
rows_tp = man[man.patient_id.isin(tp_p)].sample(min(nsl, man.patient_id.isin(tp_p).sum()), random_state=SEED); rows_tn = man[man.patient_id.isin(tn_p)].sample(min(nsl, man.patient_id.isin(tn_p).sum()), random_state=SEED)
pack = pd.concat([rows_fn.assign(cls="FN"), rows_tp.assign(cls="TP"), rows_tn.assign(cls="TN")]); pack = pack.iloc[rng.permutation(len(pack))]; pack["pack_id"] = [f"R{i:03d}" for i in range(1, len(pack) + 1)]
pack[["pack_id", "patient_id", "cls", "slide", "grade", "y"]].rename_axis("sample_id").to_csv(ROW + "/review_pack_manifest_SECRET.csv"); pack[["pack_id", "slide"]].rename_axis("sample_id").to_csv(ROW + "/review_pack_slides.csv")
RES["item10"]["review_pack"] = {"FN_patients": int(len(fn_p)), "FN_slides": int(nsl), "TP_slides": int(len(rows_tp)), "TN_slides": int(len(rows_tn)), "manifest": "feasibility/paper_plan/review_pack_manifest_SECRET.csv (cluster only)", "thumbnails": "made by scripts/paper_plan/pp_review_pack.py"}
# montage patient set for item 7 (10 progressors, 10 non-progressors, RandomState(0) from sorted ids; row = highest image_only score)
rng = np.random.RandomState(SEED); mp = list(rng.choice(sorted(pats[y == 1]), 10, replace=False)) + list(rng.choice(sorted(pats[y == 0]), 10, replace=False)); rows_m = man[man.patient_id.isin(mp)].sort_values("image_only", ascending=False).groupby("patient_id").head(1)
rows_m = rows_m.iloc[np.random.RandomState(1).permutation(len(rows_m))]; rows_m["montage_id"] = [f"M{i:02d}" for i in range(1, len(rows_m) + 1)]; rows_m[["montage_id", "patient_id", "y", "image_only", "late_mean"]].rename_axis("sample_id").to_csv(ROW + "/montage_manifest_SECRET.csv"); rows_m[["montage_id"]].rename_axis("sample_id").to_csv(ROW + "/montage_rows.csv")
# ------------------------------------------------------------------ item 7 summary (if pp_latent produced the row metrics)
att_path = glob.glob(T + "/feasibility/runs/pp_latent/output/attention_item7_rows.csv") + glob.glob(AGG + "/attention_item7_rows.csv")
if att_path:
    A7 = pd.read_csv(att_path[0], dtype={"sample_id": str}).set_index("sample_id"); A7["y"] = man.y.reindex(A7.index).values; A7["patient_id"] = man.patient_id.reindex(A7.index).values
    predmap = {a: dict(zip(pats, PRED[a])) for a in ["image_only", "late_mean"]}; A7["correct_image_only"] = [int(predmap["image_only"][p] == yy) for p, yy in zip(A7.patient_id, A7.y)]
    I7 = {}
    for fam, s in A7.groupby("fusion_model"):
        def summ(d_): return {"n_rows": int(len(d_)), "spearman_median": r3(d_.spearman.median()), "spearman_iqr": [r3(d_.spearman.quantile(.25)), r3(d_.spearman.quantile(.75))], "jaccard_top5pct_median": r3(d_.jaccard_top5pct.median()), "jaccard_top50_median": r3(d_.jaccard_top50.median()), "entropy_image_only_median": r3(d_.entropy_image_only.median()), "entropy_fusion_median": r3(d_.entropy_fusion.median()), "max_entropy_ln256": r3(np.log(256))}
        I7[fam] = {"overall": summ(s), "progressor_rows": summ(s[s.y == 1]), "nonprogressor_rows": summ(s[s.y == 0]), "image_only_correct_patient": summ(s[s.correct_image_only == 1]), "image_only_incorrect_patient": summ(s[s.correct_image_only == 0])}
    I7["_note"] = "early_fusion has no attention module (mean pooling); intermediate = image attention module; co-attention = CNV-conditioned attention; correctness at the primary operating point of image_only, patient level"
    RES["item7"] = I7
else: RES["item7"] = {"status": "attention rows not found (pp_latent pending)"}
json.dump(RES, open(AGG + "/main_items.json", "w"), indent=1, default=str); print(json.dumps({k: RES[k] for k in ["item3", "operating_point", "item8a"]}, indent=1, default=str)[:5000]); print("MAIN DONE", flush=True)

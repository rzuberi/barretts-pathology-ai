"""Follow-up F2, F3, F4, F5, F6a, F6 attention mass, F9 and the updated whiteboard table (pre-specified in
docs/paper_plan_followup.md @ 0db4075). Frozen release only. Reads the F1 arm from feasibility/paper_plan/f1_cnv_km_oof.csv.
Aggregates -> results/paper_plan/followup_main.json (+ figs/km_v2); row-level -> feasibility/paper_plan/."""
import glob, json, os, warnings, numpy as np, pandas as pd, h5py
from scipy.stats import rankdata, mannwhitneyu, fisher_exact, chi2_contingency
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from lifelines import KaplanMeierFitter, CoxPHFitter; from lifelines.statistics import multivariate_logrank_test
import statsmodels.api as sm
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
F = "/mnt/scratche/slow/fmlab/zuberi01/phd/barretts_retraining/barretts_training/analysis/chapter1_lgd2_final_pre_event_20260713_final"; R = F + "/training_final_nested_cv_v1"
T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"; S = "/mnt/scratche/fast/fmlab/datasets/imaging/SWGCohort"; E = "/mnt/scratche/slow/fmlab/zuberi01/barretts_db_export"; ROW = T + "/feasibility/paper_plan"; AGG = os.environ.get("OUTDIR", T + "/results/paper_plan"); FIG = T + "/results/paper_plan/figs/km_v2"; os.makedirs(AGG, exist_ok=True); os.makedirs(FIG, exist_ok=True)
NB = 2000; SEED = 0
def auc(y, s):
    y = np.asarray(y).astype(int); r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1; return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if 0 < n1 < len(y) else float("nan")
def ap(y, s): return float(average_precision_score(y, s)) if 0 < np.sum(y) < len(y) else float("nan")
def r3(x): return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 3)
def pf(p): return None if p is None or (isinstance(p, float) and np.isnan(p)) else ("<0.001" if p < 0.001 else round(float(p), 3))
RES = {"_spec": "docs/paper_plan_followup.md @ 0db4075", "_release": "frozen only"}
man = pd.read_csv(F + "/training_manifest.csv", dtype=str).set_index("sample_id"); coh = pd.read_csv(F + "/pre_event_cohort.csv", dtype=str).set_index("SampleID"); cohall = coh.copy(); coh = coh.loc[man.index]
man["y"] = man.y_progressor.astype(int); man["fold"] = man.fold_id_rep01.astype(int); man["date"] = pd.to_datetime(coh.Date, errors="coerce"); man["grade"] = pd.to_numeric(coh.Label, errors="coerce").fillna(0); man["maxsofar"] = pd.to_numeric(coh.MaxPathologySoFar, errors="coerce").fillna(0)
man["streak"] = pd.to_numeric(coh.LGDStreakSoFar, errors="coerce").fillna(0); man["bidx"] = pd.to_numeric(coh.BiopsyIndex, errors="coerce").fillna(0); man["dsp"] = pd.to_numeric(coh.DaysSincePreviousBiopsy, errors="coerce").fillna(0); man["year"] = man.date.dt.year
man["mbl"] = pd.to_numeric(coh.MonthsBeforeLastBiopsy, errors="coerce"); man["nbd"] = pd.to_datetime(coh.NextBiopsyDate, errors="coerce"); man["dtn"] = pd.to_numeric(coh.DaysToNextBiopsy, errors="coerce"); man["next_label"] = pd.to_numeric(coh.NextBiopsyLabel, errors="coerce"); man["participant"] = coh.participant_id.str.replace(r"\.0$", "", regex=True)
FAM = ["cnv_only", "image_only", "early_fusion", "intermediate_fusion", "late_mean", "coattention_fusion", "late_stack_logit"]
def oof(fam): return pd.concat([pd.read_csv(f, dtype={"sample_id": str}) for f in glob.glob(f"{R}/{fam}/fold*/outer_test_predictions.csv")]).set_index("sample_id").y_prob.reindex(man.index).values
for fam in FAM: man[fam] = oof(fam)
f1 = pd.read_csv(ROW + "/f1_cnv_km_oof.csv", dtype={"sample_id": str}).set_index("sample_id").reindex(man.index); man["cnv_km"] = f1.cnv_km.values; man["late_mean_km"] = f1.late_mean_km.values
ids = man.index.values; y_s = man.y.values; folds = man.fold.values; pid = man.patient_id.values
inner = {k: pd.read_csv(f"{R}/image_only/fold{k}/inner_fold_assignments.csv", dtype=str).set_index("patient_id").inner_fold.astype(int) for k in range(1, 6)}
pairs = pd.read_csv(T + "/feasibility/closeout/erin_swg_pairs.csv", dtype=str); OV = set(pairs.swg_patient_id); PT = pd.read_csv(T + "/feasibility/closeout/swg_patient_table.csv", dtype=str).set_index("patient_id")
def patient(cols, rows=None):
    d = man.assign(**cols) if cols else man; d = d if rows is None else d[rows]; g = d.groupby("patient_id"); return g.y.max().values.astype(int), {c: g[c].max().values for c in (cols or {})}, g.size().index.values, g.fold.first().values
def bidx(y):
    rng = np.random.RandomState(SEED); out = []
    while len(out) < NB:
        s = rng.choice(len(y), len(y))
        if len(set(y[s])) > 1: out.append(s)
    return out
def cis(y, B, f, a, b=None): v = [f(y[s], a[s]) - (f(y[s], b[s]) if b is not None else 0) for s in B]; return [r3(np.percentile(v, 2.5)), r3(np.percentile(v, 97.5))]
def permp(y, a, b): rng = np.random.RandomState(SEED); obs = auc(y, a) - auc(y, b); null = np.array([auc(y[ix], a) - auc(y[ix], b) for ix in (rng.permutation(len(y)) for _ in range(NB))]); return round(float((1 + (null >= obs).sum()) / (NB + 1)), 4)
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
# ------------------------------------------------------------------ F3 clinical nested versions
CSETS = {"C1_grade": ["grade"], "C2_grade_maxsofar": ["grade", "maxsofar"], "C3_plus_streak": ["grade", "maxsofar", "streak"], "C4_plus_surveillance_3a": ["grade", "maxsofar", "streak", "bidx", "dsp", "year"]}
F3 = {"C_per_fold": {}}
for c, cols in CSETS.items(): man[c], F3["C_per_fold"][c] = nested_logistic(man[cols].values.astype(float), y_s, folds, pid)
ARMS = list(CSETS) + ["cnv_only", "cnv_km", "image_only", "early_fusion", "intermediate_fusion", "late_mean", "late_mean_km", "coattention_fusion", "late_stack_logit"]
def table_on(rows):
    y, P, pats, _ = patient({a: man[a].values for a in ARMS}, rows); B = bidx(y); return {"n": int(len(y)), "events": int(y.sum()), "arms": {a: {"auroc": r3(auc(y, P[a])), "auroc_ci": cis(y, B, auc, P[a]), "auprc": r3(ap(y, P[a])), "auprc_ci": cis(y, B, ap, P[a])} for a in ARMS}}, (y, P, pats)
F3["table_all_150"], (y, P, pats) = table_on(None); B = bidx(y)
def zf(v):
    v = np.asarray(v, float); o = np.zeros(len(v))
    for f_ in np.unique(folds): te = folds == f_; o[te] = (v[te] - v[te].mean()) / (v[te].std() + 1e-9)
    return o
def logit(p): p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6); return np.log(p / (1 - p))
COMBOS = {"C2+image": ["image_only"], "C2+cnv_only": ["cnv_only"], "C2+cnv_km": ["cnv_km"], "C2+late_mean": ["late_mean"], "C2+image+cnv_only": ["image_only", "cnv_only"], "C2+image+cnv_km": ["image_only", "cnv_km"]}
F3["C2_plus_modality"] = {}
for name, mods in COMBOS.items():
    zmean = (zf(man.C2_grade_maxsofar) + sum(zf(man[m_]) for m_ in mods)) / (1 + len(mods)); Xst = np.column_stack([logit(man.C2_grade_maxsofar)] + [logit(man[m_]) for m_ in mods]); stack, Cst = nested_logistic(Xst, y_s, folds, pid)
    yy, Pc, _, _ = patient({"c2": man.C2_grade_maxsofar.values, "z": zmean, "st": stack}); F3["C2_plus_modality"][name] = {"fold_z_mean": {"auroc": r3(auc(yy, Pc["z"])), "delta_vs_C2": r3(auc(yy, Pc["z"]) - auc(yy, Pc["c2"])), "ci": cis(yy, B, auc, Pc["z"], Pc["c2"]), "perm_p": permp(yy, Pc["z"], Pc["c2"])},
        "L2_stack_on_logits": {"auroc": r3(auc(yy, Pc["st"])), "delta_vs_C2": r3(auc(yy, Pc["st"]) - auc(yy, Pc["c2"])), "ci": cis(yy, B, auc, Pc["st"], Pc["c2"]), "perm_p": permp(yy, Pc["st"], Pc["c2"]), "C_per_fold": Cst}}
for m_ in ["image_only", "cnv_only", "cnv_km", "late_mean", "late_mean_km"]: F3[f"{m_}_vs_C2"] = {"delta": r3(auc(y, P[m_]) - auc(y, P["C2_grade_maxsofar"])), "ci": cis(y, B, auc, P[m_], P["C2_grade_maxsofar"]), "perm_p_two_sided_note": "one-sided p that modality > C2", "perm_p": permp(y, P[m_], P["C2_grade_maxsofar"])}
RES["F3"] = F3; print("F3 done", flush=True)
# ------------------------------------------------------------------ F2 design strata
cx = pd.read_csv(F + "/feature_views/cnv/cx.csv", dtype=str).set_index("sample_id").reindex(man.index); man["cnv_id"] = cx.cnv_id.values
fd = pd.read_csv(S + "/sWGS_777_samples_cleaned_202401_Leanne_fullDetails (3) (1).csv", dtype=str); fd.columns = [c.strip().replace("\n", " ") for c in fd.columns]; fd = fd.drop_duplicates("combined_name").set_index("combined_name")
va = pd.read_csv(S + "/sWGS_validation_cleaned_Leanne (4) (1).csv", dtype=str).drop_duplicates("sample_id").set_index("sample_id")
man["sheet_status"] = man.cnv_id.map(fd.Status); man["in_disc"] = man.cnv_id.isin(fd.index); man["in_val"] = man.cnv_id.isin(va.index)
g = man.groupby("patient_id"); st = {}
for p_, d in g:
    if d.in_disc.any(): s_ = d.sheet_status.dropna().mode(); st[p_] = "discovery_case" if len(s_) and s_.iloc[0] == "P" else "discovery_control"
    elif d.in_val.any(): st[p_] = "validation_sheet"
    else: st[p_] = "other"
stratum = pd.Series(st).reindex(pats).values; sub = np.array(["also_in_ERIN" if p_ in OV else "never_in_ERIN" for p_ in pats])
F2 = {"stratum_definition": "discovery_case/control = any row CNV id in the 777 discovery sheet, sheet Status P/NP (mode); validation_sheet = any row in the 268 validation sheet; other = neither", "crosstab_stratum_vs_overlap": pd.crosstab(stratum, sub).to_dict(), "stratum_events": pd.crosstab(stratum, y).to_dict(), "tables": {}}
for s_ in np.unique(stratum):
    m_ = stratum == s_; rows = pd.Series(pid).map(dict(zip(pats, m_))).values
    if m_.sum() >= 20 and y[m_].sum() >= 5 and (m_.sum() - y[m_].sum()) >= 5: F2["tables"][s_], _ = table_on(rows)
    else: F2["tables"][s_] = {"n": int(m_.sum()), "events": int(y[m_].sum()), "skipped": "fewer than 20 patients or 5 events/non-events"}
m_ = stratum == "discovery_case"; m2 = stratum == "discovery_control"; rowsd = pd.Series(pid).map(dict(zip(pats, m_ | m2))).values
if (m_ | m2).sum() >= 20 and y[m_ | m2].sum() >= 5: F2["tables"]["discovery_all"], _ = table_on(rowsd)
# matching check within discovery: progressors vs non-progressors (our label)
dem = pd.read_csv(S + "/Demographics_full.csv", dtype=str); dem.columns = [c.strip() for c in dem.columns]; dem = dem.set_index("Study Number")
D = pd.DataFrame({"y": y, "stratum": stratum, "age": pd.to_numeric(dem["Age at diagnosis"].reindex(pats), errors="coerce").values, "sex": dem.Sex.reindex(pats).values, "pragueM": pd.to_numeric(dem["Maximal"].reindex(pats), errors="coerce").values, "pragueC": pd.to_numeric(dem["Circumference"].reindex(pats), errors="coerce").values, "fu_months": pd.to_numeric(PT.followup_months_first_to_last_biopsy.reindex(pats), errors="coerce").values, "sheet_status_case": stratum == "discovery_case"}, index=pats)
dd = D[D.stratum.str.startswith("discovery")]; F2["discovery_matching_check"] = {}
for c in ["age", "pragueM", "pragueC", "fu_months"]:
    a_, b_ = dd[c][dd.y == 1].dropna(), dd[c][dd.y == 0].dropna(); F2["discovery_matching_check"][c] = {"progressors_median_n": [r3(a_.median()) if len(a_) else None, int(len(a_))], "nonprogressors_median_n": [r3(b_.median()) if len(b_) else None, int(len(b_))], "mannwhitney_p": pf(mannwhitneyu(a_, b_).pvalue) if len(a_) > 1 and len(b_) > 1 else None}
t = pd.crosstab(dd.sex.fillna("missing"), dd.y); F2["discovery_matching_check"]["sex"] = {"table": t.to_dict(), "fisher_p": pf(fisher_exact(t.loc[[i for i in t.index if i != "missing"]].values)[1]) if t.loc[[i for i in t.index if i != "missing"]].shape == (2, 2) else None}
F2["discovery_matching_check"]["our_label_vs_sheet_status"] = pd.crosstab(dd.sheet_status_case, dd.y).to_dict()
# stratum-only score and scores by stratum among non-progressors
pf_ = pd.Series(folds, index=pid).groupby(level=0).first().reindex(pats).values; sc = np.zeros(len(y))
for k in range(1, 6):
    tr = pf_ != k; rates = pd.Series(y[tr]).groupby(stratum[tr]).mean(); sc[~tr] = pd.Series(stratum[~tr]).map(rates).fillna(y[tr].mean()).values
F2["stratum_only_score"] = {"auroc": r3(auc(y, sc)), "ci": cis(y, B, auc, sc), "n": int(len(y)), "events": int(y.sum())}
neg = y == 0; F2["arm_scores_by_stratum_nonprogressors_discovery_vs_validation"] = {}
for a in ["C2_grade_maxsofar", "cnv_only", "cnv_km", "image_only", "late_mean", "late_mean_km"]:
    a_ = P[a][neg & np.isin(stratum, ["discovery_case", "discovery_control"])]; b_ = P[a][neg & (stratum == "validation_sheet")]
    F2["arm_scores_by_stratum_nonprogressors_discovery_vs_validation"][a] = {"discovery_median_n": [r3(np.median(a_)), int(len(a_))], "validation_median_n": [r3(np.median(b_)), int(len(b_))], "mannwhitney_p": pf(mannwhitneyu(a_, b_).pvalue)}
RES["F2"] = F2; pd.DataFrame({"patient_id": pats, "stratum": stratum, "subgroup": sub, "y": y}).to_csv(ROW + "/f2_strata.csv", index=False); print("F2 done", flush=True)
# ------------------------------------------------------------------ F4 survival
g = man.groupby("patient_id"); first = g.date.min().reindex(pats); lastrow = g.date.max().reindex(pats); mbl_first = man.sort_values("date").groupby("patient_id").mbl.first().reindex(pats); mbl_max = g.mbl.max().reindex(pats); last_next = g.nbd.max().reindex(pats)
t_event = pd.to_numeric(PT.days_first_to_event.reindex(pats), errors="coerce").values
timeA = np.where(y == 1, t_event, mbl_max.values * 30.44); timeB = np.where(y == 1, t_event, (last_next.fillna(lastrow) - first).dt.days.values); timeA = np.maximum(np.nan_to_num(timeA, nan=0), 0.5); timeB = np.maximum(np.nan_to_num(timeB, nan=0), 0.5)
mono = man.sort_values("date").groupby("patient_id").mbl.apply(lambda s: bool((s.diff().dropna() < 0).all()) if len(s) > 1 else None)
F4 = {"censoring_derivation": {"MonthsBeforeLastBiopsy_first_row_vs_days_first_to_max_NextBiopsyDate_spearman": r3(pd.Series(mbl_first.values * 30.44).corr(pd.Series((last_next - first).dt.days.values), method="spearman")), "median_abs_diff_days": r3(np.nanmedian(np.abs(mbl_first.values * 30.44 - (last_next - first).dt.days.values))), "patients_with_mbl_strictly_decreasing_along_rows": int(mono.dropna().sum()), "patients_with_more_than_one_row": int(mono.notna().sum()),
    "reading": "MonthsBeforeLastBiopsy is not consistently measured from each row (it decreases along rows for only part of the patients) and its first-row value does not equal the interval to the last NextBiopsyDate; version A (used in the paper-plan item 5) = first row + max MonthsBeforeLastBiopsy; version B = first row -> max NextBiopsyDate from biopsy dates"},
      "nonprogressor_time_days": {"A_median": r3(np.median(timeA[y == 0])), "B_median": r3(np.median(timeB[y == 0])), "A_iqr": [r3(np.percentile(timeA[y == 0], 25)), r3(np.percentile(timeA[y == 0], 75))], "B_iqr": [r3(np.percentile(timeB[y == 0], 25)), r3(np.percentile(timeB[y == 0], 75))]}, "progressor_time_days": {"median": r3(np.median(timeA[y == 1])), "iqr": [r3(np.percentile(timeA[y == 1], 25)), r3(np.percentile(timeA[y == 1], 75))]}, "groups": {}}
def groups_tertile(a):
    gg = np.zeros(len(y), int)
    for k in range(1, 6):
        tr = pf_ != k; te = ~tr; q1, q2 = np.quantile(P[a][tr], [1 / 3, 2 / 3]); gg[te] = np.where(P[a][te] > q2, 2, np.where(P[a][te] > q1, 1, 0))
    return gg
def surv(a, gg, time, tag):
    out = {"groups": {}}
    for gi, gn in enumerate(["low", "moderate", "high"]):
        m = gg == gi; out["groups"][gn] = {"n": int(m.sum()), "events": int(y[m].sum()), "rate": r3(y[m].mean()) if m.sum() else None, "median_time_progressors_d": r3(np.median(time[m & (y == 1)])) if (m & (y == 1)).sum() else None, "median_time_nonprogressors_d": r3(np.median(time[m & (y == 0)])) if (m & (y == 0)).sum() else None}
    df = pd.DataFrame({"T": time, "E": y, "g": gg, "mod": (gg == 1).astype(int), "high": (gg == 2).astype(int)})
    try: out["logrank_p"] = pf(multivariate_logrank_test(df["T"], df.g, df.E).p_value); cph = CoxPHFitter().fit(df[["T", "E", "mod", "high"]], "T", "E"); out["cox_hr_high_vs_low"] = [r3(np.exp(cph.params_["high"]))] + [r3(v) for v in np.exp(cph.confidence_intervals_.loc["high"]).values]; out["cox_hr_moderate_vs_low"] = [r3(np.exp(cph.params_["mod"]))] + [r3(v) for v in np.exp(cph.confidence_intervals_.loc["mod"]).values]
    except Exception as e: out["cox_error"] = str(e)[:100]
    fig, ax = plt.subplots(figsize=(5, 4)); km = KaplanMeierFitter()
    for gi, gn, c in [(0, "low", "#1f77b4"), (1, "moderate", "#ff7f0e"), (2, "high", "#d62728")]:
        m = gg == gi
        if m.sum(): km.fit(df["T"][m] / 365.25, df.E[m], label=f"{gn} (n={m.sum()}, events={int(y[m].sum())})").plot_survival_function(ax=ax, ci_show=False, color=c)
    ax.set_xlabel("years from earliest release row"); ax.set_ylabel("endpoint-free"); ax.set_title(f"{a} tertiles, time version {tag}"); fn = f"{FIG}/km_{a}_{tag}.png"; fig.tight_layout(); fig.savefig(fn, dpi=130); plt.close(fig); out["figure"] = fn.replace(T + "/", ""); return out
for a in ["cnv_only", "cnv_km", "late_mean", "late_mean_km", "C2_grade_maxsofar"]:
    gg = groups_tertile(a); F4["groups"][a] = {"version_A_release_MonthsBeforeLastBiopsy": surv(a, gg, timeA, "A"), "version_B_biopsy_dates": surv(a, gg, timeB, "B")}
    lo, hi = gg == 0, gg == 2; a_, b_, c_, d_ = y[hi].sum() + .5, (1 - y[hi]).sum() + .5, y[lo].sum() + .5, (1 - y[lo]).sum() + .5; F4["groups"][a]["odds_ratio_high_vs_low_haldane"] = r3((a_ * d_) / (b_ * c_))
    disc = np.isin(stratum, ["discovery_case", "discovery_control"]); F4["groups"][a]["discovery_stratum_rates_only"] = {gn: {"n": int((gg == gi)[disc].sum()), "events": int(y[disc & (gg == gi)].sum()), "rate": r3(y[disc & (gg == gi)].mean()) if (disc & (gg == gi)).sum() else None} for gi, gn in enumerate(["low", "moderate", "high"])}
F4["time_to_event_interpretability"] = "in the Killcoyne discovery stratum controls were selected on >= 3 y follow-up (matched case-control), so censoring time is a selection criterion, not an outcome; for that stratum only rates and ORs are reported"
F4["probability_deciles"] = {a: [r3(v) for v in np.percentile(P[a], np.arange(0, 101, 10))] for a in ["cnv_only", "cnv_km", "late_mean", "late_mean_km"]}
gk = np.where(P["cnv_km"] >= 0.5, 2, np.where(P["cnv_km"] > 0.3, 1, 0)); F4["killcoyne_fixed_classes_on_cnv_km"] = {gn: {"n": int((gk == gi).sum()), "events": int(y[gk == gi].sum()), "rate": r3(y[gk == gi].mean()) if (gk == gi).sum() else None} for gi, gn in enumerate(["low", "moderate", "high"])}
RES["F4"] = F4; print("F4 done", flush=True)
# ------------------------------------------------------------------ F5 false positives
sp = pd.read_csv(ROW + "/patient_scores_predictions.csv", dtype={"patient_id": str}).set_index("patient_id").reindex(pats); LT = pd.read_csv(ROW + "/later_disease_patient.csv", dtype={"patient_id": str}).set_index("patient_id").reindex(pats)
bgr = pd.to_numeric(PT.baseline_grade.reindex(pats), errors="coerce").fillna(0).values; mxs = g.maxsofar.max().reindex(pats).values
def op_pred(a):
    pred = np.zeros(len(y), int)
    for k in range(1, 6):
        tr = pf_ != k; te = ~tr; s_tr = P[a][tr]; ss = np.sort(np.unique(s_tr))[::-1]; t = ss[-1]
        for t_ in ss:
            if (s_tr[y[tr] == 1] >= t_).mean() >= 0.8: t = t_; break
        pred[te] = (P[a][te] >= t).astype(int)
    return pred
PRED = {a: sp[f"pred_{a}"].values.astype(int) for a in ["late_mean", "image_only", "cnv_only", "clinical_3a", "v4_exploratory"]}; PRED["C2_grade_maxsofar"] = op_pred("C2_grade_maxsofar"); PRED["cnv_km"] = op_pred("cnv_km"); PRED["late_mean_km"] = op_pred("late_mean_km")
laterL = ((LT.db_max_grade_after >= 2) | (LT.release_excluded_max_grade >= 2) | (LT.slidematch_max_grade_after >= 2)).values; laterH = ((LT.db_max_grade_after >= 3) | (LT.release_excluded_max_grade >= 3) | (LT.slidematch_max_grade_after >= 3) | (LT.hgd_table_entries_after > 0)).values
F5 = {"adjusted_logistic": {}, "within_baseline_NDBE": {}}
for a, pred in PRED.items():
    fp = (pred == 1) & neg; out = {"n_FP": int(fp.sum()), "n_TN": int((neg & (pred == 0)).sum())}
    for nm, lab in [("later_HGDplus", laterH), ("later_LGDplus", laterL)]:
        Xd = pd.DataFrame({"FP": fp[neg].astype(float), "baseline_grade": bgr[neg], "max_grade_so_far": mxs[neg]}); yy = lab[neg].astype(int)
        try:
            mres = sm.Logit(yy, sm.add_constant(Xd)).fit(disp=0, maxiter=200); ci_ = np.exp(mres.conf_int().loc["FP"]); out[nm] = {"OR_FP_adjusted": r3(np.exp(mres.params["FP"])), "ci": [r3(ci_[0]), r3(ci_[1])], "p": pf(mres.pvalues["FP"]), "OR_baseline_grade": r3(np.exp(mres.params["baseline_grade"])), "OR_max_grade": r3(np.exp(mres.params["max_grade_so_far"])), "events": int(yy.sum()), "n": int(len(yy))}
            m0 = sm.Logit(yy, sm.add_constant(Xd[["FP"]])).fit(disp=0); out[nm]["OR_FP_unadjusted"] = r3(np.exp(m0.params["FP"]))
        except Exception as e: out[nm] = {"error": str(e)[:100]}
    F5["adjusted_logistic"][a] = out
    nd = neg & (bgr == 0); fpn = fp & nd; tnn = (pred == 0) & nd
    F5["within_baseline_NDBE"][a] = {"n_FP": int(fpn.sum()), "n_TN": int(tnn.sum()), "later_HGDplus_FP_TN": [int(laterH[fpn].sum()), int(laterH[tnn].sum())], "fisher_p_HGD": pf(fisher_exact([[laterH[fpn].sum(), fpn.sum() - laterH[fpn].sum()], [laterH[tnn].sum(), tnn.sum() - laterH[tnn].sum()]])[1]) if fpn.sum() and tnn.sum() else None, "later_LGDplus_FP_TN": [int(laterL[fpn].sum()), int(laterL[tnn].sum())], "fisher_p_LGD": pf(fisher_exact([[laterL[fpn].sum(), fpn.sum() - laterL[fpn].sum()], [laterL[tnn].sum(), tnn.sum() - laterL[tnn].sum()]])[1]) if fpn.sum() and tnn.sum() else None}
# later HGD+ non-progressors: why not release progressors
pt = pd.read_csv(E + "/pathology_text_normalised_full.csv", dtype=str, usecols=["participant_id", "receiveddatetime", "highestgradedysconf"]); pt["d"] = pd.to_datetime(pt.receiveddatetime, errors="coerce"); pt["g"] = pd.to_numeric(pt.highestgradedysconf, errors="coerce").map({2: 0, 3: 1, 4: 2, 5: 3, 6: 4, 8: 4})
part = g.participant.first().reindex(pats); rowsH = []
for i, p_ in enumerate(pats):
    if not (neg[i] and laterH[i]): continue
    l = lastrow.iloc[i]; db = pt[(pt.participant_id == part.iloc[i]) & (pt.d > l) & (pt.g >= 3)].sort_values("d"); src = []
    if len(db): src.append("db_report")
    if LT.release_excluded_max_grade.iloc[i] >= 3: src.append("release_excluded_row")
    if LT.slidematch_max_grade_after.iloc[i] >= 3: src.append("slide_matching")
    if LT.hgd_table_entries_after.iloc[i] > 0: src.append("hgd_table")
    d_h = db.d.min() if len(db) else pd.NaT; days = (d_h - l).days if pd.notna(d_h) else None
    lastnext = last_next.iloc[i]; is_next = bool(pd.notna(d_h) and pd.notna(lastnext) and abs((d_h - lastnext).days) <= 30)
    rowsH.append({"patient_id": p_, "days_after_last_release_row": days, "sources": "+".join(src), "hgd_report_is_the_next_release_biopsy": is_next, "last_release_row_next_label": float(man[man.patient_id == p_].sort_values("date").next_label.iloc[-1]), "excluded_reasons": LT.release_excluded_reasons.iloc[i] if isinstance(LT.release_excluded_reasons.iloc[i], str) else ""})
H = pd.DataFrame(rowsH); H.to_csv(ROW + "/f5_later_hgd_nonprogressors.csv", index=False)
F5["later_HGDplus_nonprogressors"] = {"n": int(len(H)), "days_after_last_release_row_median_iqr": [r3(H.days_after_last_release_row.median()), r3(H.days_after_last_release_row.quantile(.25)), r3(H.days_after_last_release_row.quantile(.75))] if len(H) else None, "sources": H.sources.value_counts().to_dict() if len(H) else {}, "flagged_hgd_is_next_release_biopsy": int(H.hgd_report_is_the_next_release_biopsy.sum()) if len(H) else 0, "with_release_excluded_rows": int((H.excluded_reasons != "").sum()) if len(H) else 0, "excluded_reason_counts": H.excluded_reasons.value_counts().to_dict() if len(H) else {},
    "why_not_progressors": "the endpoint is evaluated on the NEXT biopsy after each strict pre-event row; a later HGD+ report that is not that next biopsy (later surveillance, or after an endpoint-not-evaluable gap) does not enter the release label"}
for a in ["late_mean", "image_only"]:
    fp = (PRED[a] == 1) & neg; F5[f"grades_behind_counts_{a}"] = {"FP_later_LGDplus_n": int(laterL[fp].sum()), "FP_later_HGDplus_n": int(laterH[fp].sum()), "FP_max_later_grade_counts": pd.Series(np.nanmax(np.column_stack([LT.db_max_grade_after.values, LT.release_excluded_max_grade.values, LT.slidematch_max_grade_after.values]), axis=1)[fp]).value_counts(dropna=False).to_dict(), "definition": "LGD+ = max later grade >= 2 (LGD/HGD/IMC); HGD+ = max later grade >= 3 or an hgd_table entry; identical counts mean every FP with a later LGD+ also reached HGD+"}
RES["F5"] = F5; print("F5 done", flush=True)
# ------------------------------------------------------------------ F6a + attention mass
ui = pd.read_csv(F + "/feature_views/uni2/uni2_index.csv", dtype=str).set_index("sample_id").reindex(man.index); kept = []
for b in ui.image_basename:
    with h5py.File(f"{S}/features_uni2h_05um/{os.path.splitext(str(b))[0]}.h5") as h: kept.append(float(h.attrs["kept_tiles"]))
man["kept"] = kept; pk = g.kept.mean().reindex(pats).values if "kept" in man else None; pk = man.groupby("patient_id").kept.mean().reindex(pats).values; ter = pd.qcut(pk, 3, labels=["T1_low", "T2", "T3_high"]).astype(str)
F6 = {"image_only_by_tiles_kept_tertile": {}}
for t_ in ["T1_low", "T2", "T3_high"]:
    m = ter == t_; yy = y[m]; Bm = bidx(yy); F6["image_only_by_tiles_kept_tertile"][t_] = {"n": int(m.sum()), "events": int(yy.sum()), "tiles_kept_range": [r3(pk[m].min()), r3(pk[m].max())], "image_only_auroc": r3(auc(yy, P["image_only"][m])), "ci": cis(yy, Bm, auc, P["image_only"][m]), "late_mean_auroc": r3(auc(yy, P["late_mean"][m])), "cnv_only_auroc": r3(auc(yy, P["cnv_only"][m]))}
L = T + "/feasibility/paper_plan/latent"; ids_l = list(np.load(L + "/ids.npy")); fold_l = pd.Series(folds, index=ids).reindex(ids_l).values; F6["attention_mass"] = {}
for fam in ["image_only", "intermediate_fusion", "coattention_fusion"]:
    top5 = []; top10 = []; mx = []
    for k in range(1, 6):
        A = np.load(f"{L}/attn_{fam}_fold{k}.npy")[fold_l == k]; s = -np.sort(-A, axis=1); top5 += list(s[:, :13].sum(1)); top10 += list(s[:, :26].sum(1)); mx += list(s[:, 0])
    top5, top10, mx = np.array(top5), np.array(top10), np.array(mx); F6["attention_mass"][fam] = {"rows": int(len(top5)), "top5pct_mass_median_iqr": [r3(np.median(top5)), r3(np.percentile(top5, 25)), r3(np.percentile(top5, 75))], "top10pct_mass_median_iqr": [r3(np.median(top10)), r3(np.percentile(top10, 25)), r3(np.percentile(top10, 75))], "uniform_top5_top10": [0.05, 0.1], "max_weight_median": r3(np.median(mx)), "uniform_weight": r3(1 / 256), "frac_rows_top5_mass_below_0.06": r3((top5 < 0.06).mean())}
RES["F6_cpu"] = F6; print("F6 done", flush=True)
# ------------------------------------------------------------------ F9 non-inferiority
d_km = auc(y, P["image_only"]) - auc(y, P["cnv_km"]); d_rf = auc(y, P["image_only"]) - auc(y, P["cnv_only"]); RES["F9_noninferiority"] = {"margin_prespecified": -0.05, "WSI_minus_cnv_only": {"delta": r3(d_rf), "ci": cis(y, B, auc, P["image_only"], P["cnv_only"])}, "WSI_minus_cnv_km": {"delta": r3(d_km), "ci": cis(y, B, auc, P["image_only"], P["cnv_km"])}}
for k_ in ["WSI_minus_cnv_only", "WSI_minus_cnv_km"]: RES["F9_noninferiority"][k_]["lower_bound_above_margin"] = bool(RES["F9_noninferiority"][k_]["ci"][0] > -0.05)
pd.DataFrame({"patient_id": pats, "y": y, **{a: P[a] for a in ARMS}, **{f"pred_{a}": v for a, v in PRED.items()}}).to_csv(ROW + "/followup_patient_scores.csv", index=False)
json.dump(RES, open(AGG + "/followup_main.json", "w"), indent=1, default=str); print("FOLLOWUP MAIN DONE", flush=True)

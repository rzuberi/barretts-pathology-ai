"""Closeout for review (docs/closeout_for_review.md): items A, B, C, D, E, F, G, I, J, K, L, M computed from data.
Pre-specified in docs/closeout_for_review.md before this script was run. Nothing here reads a number from any prose file.

Conventions (every item): SWG release chapter1_lgd2_final_pre_event_20260713_final, 707 rows / 150 patients / 50 progressor
patients; arm scores are release OOF probabilities (image_only, cnv_only) or ERIN-head imputed probabilities per slide;
late fusion = mean of fold-local z-scores (z within each release outer fold); patient score = MAX over the patient's rows;
AUROC rank-based; 95 % CI = percentile bootstrap over PATIENTS (2,000 resamples, numpy RandomState(0)); permutation tests
permute PATIENT-level labels (2,000, RandomState(0)). Row-level tables -> feasibility/closeout (cluster only);
aggregates -> results/closeout/closeout_main.json and results/closeout/tables.md (markdown tables for the report)."""
import glob, json, os, re, sys, hashlib, numpy as np, pandas as pd
from scipy.stats import rankdata, mannwhitneyu, fisher_exact, chi2_contingency
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score
F = "/mnt/scratche/slow/fmlab/zuberi01/phd/barretts_retraining/barretts_training/analysis/chapter1_lgd2_final_pre_event_20260713_final"
T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"; R = T + "/feasibility/runs"; E = "/mnt/scratche/slow/fmlab/zuberi01/barretts_db_export"; S = "/mnt/scratche/fast/fmlab/datasets/imaging/SWGCohort"
ROW = T + "/feasibility/closeout"; AGG = os.environ.get("OUTDIR", T + "/results/closeout"); os.makedirs(ROW, exist_ok=True); os.makedirs(AGG, exist_ok=True); NB = 2000; SEED = 0
sys.path.insert(0, T + "/scripts"); from abmil_clf import patient_folds
def auc(y, s):
    y = np.asarray(y).astype(int); r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if 0 < n1 < len(y) else float("nan")
def r3(x): return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 3)
RES = {"_conventions": __doc__}; MD = []
def to_md(df):
    df = df.astype(str); cols = list(df.columns)
    return "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n" + "\n".join("| " + " | ".join(str(v).replace("|", "/") for v in row) + " |" for row in df.values.tolist())
def md(title, df):
    MD.append(f"\n**{title}**\n\n" + to_md(df) + "\n")
# ------------------------------------------------------------------ data
man = pd.read_csv(F + "/training_manifest.csv", dtype=str).set_index("sample_id"); coh = pd.read_csv(F + "/pre_event_cohort.csv", dtype=str).set_index("SampleID").loc[man.index]
man["y"] = man.y_progressor.astype(int); man["fold"] = man.fold_id_rep01.astype(int); man["date"] = pd.to_datetime(coh.Date, errors="coerce"); man["grade"] = pd.to_numeric(coh.Label, errors="coerce")
man["days_to_next"] = pd.to_numeric(coh.DaysToNextBiopsy, errors="coerce"); man["next_label"] = pd.to_numeric(coh.NextBiopsyLabel, errors="coerce"); man["d_cur_event"] = pd.to_numeric(coh.DaysFromCurrentToEvent, errors="coerce")
def oof(root, fam): return pd.concat([pd.read_csv(f, dtype={"sample_id": str}) for f in glob.glob(f"{root}/{fam}/fold*/outer_test_predictions.csv")]).set_index("sample_id").y_prob.reindex(man.index).values
man["img"] = oof(F + "/training_final_nested_cv_v1", "image_only"); man["cnv"] = oof(F + "/training_final_nested_cv_v1", "cnv_only")
m2 = pd.read_csv(F + "_rep02/training_manifest_v2.csv", dtype={"sample_id": str}).set_index("sample_id").loc[man.index]; man["fold2"] = m2.fold_id_rep01.astype(int).values
man["img2"] = oof(F + "_rep02/training_rep02_nested_cv", "image_only"); man["cnv2"] = oof(F + "_rep02/training_rep02_nested_cv", "cnv_only")
def imp(d):
    x = pd.read_csv(d + "/swg_imputed_fields.csv", index_col=0); x.index = x.index.astype(str); return x.reindex(man.index)
IMP = {"p32b": imp(R + "/p32b_fields/output"), "noov": imp(R + "/p32_head_noov/output"), "repeat": imp(R + "/p32_head_repeat/output"), "perm": imp(R + "/p32_head_perm/output"),
       "pass1": pd.concat([imp(R + f"/p32_fields_g{i}/output") for i in range(4)], axis=1).dropna(axis=1, how="all")}
for k, v in IMP.items(): IMP[k] = v.loc[:, ~v.columns.duplicated()]
for k, d in [("h_treat", os.environ.get("H_TREAT_DIR")), ("h_im", os.environ.get("H_IM_DIR")), ("h_perm_noov", os.environ.get("H_PERM_DIR"))]:
    if d and os.path.exists(d + "/swg_imputed_fields.csv"): IMP[k] = imp(d)
y_s = man.y.values; folds = man.fold.values; folds2 = man.fold2.values; pid = man.patient_id.values
def zf(v, fl):
    v = np.asarray(v, float); o = np.zeros(len(v))
    for f in np.unique(fl): te = fl == f; o[te] = (v[te] - v[te].mean()) / (v[te].std() + 1e-9)
    return o
def cvlog(X, y, fl, C=1.0):
    X = np.asarray(X, float); p = np.zeros(len(y))
    for f in np.unique(fl):
        te = fl == f; tr = ~te; mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9; p[te] = LogisticRegression(C=C, max_iter=5000).fit((X[tr] - mu) / sd, y[tr]).predict_proba((X[te] - mu) / sd)[:, 1]
    return p
def fuse(*arms, fl=None): fl = folds if fl is None else fl; return sum(zf(a, fl) for a in arms) / len(arms)
def patient(cols, rows=None):
    d = man.assign(**cols); d = d if rows is None else d[rows]; g = d.groupby("patient_id"); return g.y.max().values.astype(int), {c: g[c].max().values for c in cols}, g.size().index.values
def boots(y, n=NB, seed=SEED, strata=None):
    rng = np.random.RandomState(seed); out = []
    while len(out) < n:
        if strata is None: s = rng.choice(len(y), len(y))
        else: s = np.concatenate([rng.choice(np.where(strata == g)[0], (strata == g).sum()) for g in np.unique(strata)])
        if len(set(y[s])) > 1: out.append(s)
    return out
def ci(y, B, a, b=None): v = [auc(y[s], a[s]) - (auc(y[s], b[s]) if b is not None else 0) for s in B]; return [r3(np.percentile(v, 2.5)), r3(np.percentile(v, 97.5))]
def perm_p(y, a, b, n=NB, seed=SEED):
    rng = np.random.RandomState(seed); obs = auc(y, a) - auc(y, b); null = np.array([auc(y[ix], a) - auc(y[ix], b) for ix in (rng.permutation(len(y)) for _ in range(n))]); return round(float((1 + (null >= obs).sum()) / (n + 1)), 4)
# ------------------------------------------------------------------ A. reconciliation
grade_b = IMP["p32b"].grade_LGDplus.values; grade_n = IMP["noov"].grade_LGDplus.values
A = {"v1_p32b_fields6_logistic": {"third_arm": cvlog(IMP["p32b"].values, y_s, folds), "head_alone_label": "CV logistic on 6 imputed fields (incl-overlap heads)"},
     "v2_p32b_grade_only": {"third_arm": grade_b, "head_alone_label": "imputed grade LGD+ (incl-overlap head)"},
     "v3_noov_fields2_logistic": {"third_arm": cvlog(IMP["noov"].values, y_s, folds), "head_alone_label": "CV logistic on 2 leak-free imputed fields (grade, inflammation)"},
     "v4_noov_grade_only_CANONICAL": {"third_arm": grade_n, "head_alone_label": "imputed grade LGD+ (leak-free head, 55 ERIN ids / 54 SWG patients excluded)"}}
cols = {"img": man.img.values, "cnv": man.cnv.values, "fuse2": fuse(man.img, man.cnv)}
for k, v in A.items(): cols[k + "_head"] = v["third_arm"]; cols[k + "_fuse3"] = fuse(man.img, man.cnv, v["third_arm"])
y, P, pats = patient(cols); B = boots(y); n_all, pos_all = len(y), int(y.sum())
rowsA = []
for k, v in A.items():
    rowsA.append({"version": k, "n": n_all, "events": pos_all, "head_alone": r3(auc(y, P[k + "_head"])), "head_ci": ci(y, B, P[k + "_head"]), "fuse2": r3(auc(y, P["fuse2"])), "fuse3": r3(auc(y, P[k + "_fuse3"])), "fuse3_ci": ci(y, B, P[k + "_fuse3"]),
                  "gain": r3(auc(y, P[k + "_fuse3"]) - auc(y, P["fuse2"])), "gain_ci": ci(y, B, P[k + "_fuse3"], P["fuse2"]), "perm_p_gain": perm_p(y, P[k + "_fuse3"], P["fuse2"]), "third_arm": v["head_alone_label"]})
RES["A_versions"] = rowsA; md("A. Versions of the fusion result recomputed (all 150 patients, 50 progressors)", pd.DataFrame(rowsA))
RES["A_arms"] = {"img": [r3(auc(y, P["img"]))] + ci(y, B, P["img"]), "cnv": [r3(auc(y, P["cnv"]))] + ci(y, B, P["cnv"]), "fuse2": [r3(auc(y, P["fuse2"]))] + ci(y, B, P["fuse2"])}
# ------------------------------------------------------------------ B. overlap
pairs = pd.read_csv(ROW + "/erin_swg_pairs.csv", dtype=str); memb = pd.read_csv(ROW + "/swg_overlap_training_membership.csv", dtype=str)
ov = set(pairs.swg_patient_id); tcols = [c for c in memb.columns if c.startswith("p32")]; memb[tcols] = memb[tcols].astype(int)
in_train_any = set(memb[memb[[c for c in tcols if c != "p32_head_noov"]].sum(axis=1) > 0].swg_patient_id)
mm = pd.read_csv(T + "/labeller/erin_master.csv", dtype=str); er_ids = set(pairs.erin_anon_id); imaged = mm[mm.anon_id.isin(er_ids)].drop_duplicates("CaseName")
RES["B_overlap"] = {"swg_patients_linked_to_erin": len(ov), "erin_anon_ids_linked": len(er_ids), "pairs": len(pairs), "how": pairs.how.value_counts().to_dict(), "swg_patients_with_case_in_any_inclusive_training_table": len(in_train_any),
    "swg_patients_with_case_in_noov_training": int((memb.p32_head_noov > 0).sum()), "imaged_erin_cases_of_linked_ids": int(len(imaged)), "imaged_label_status": imaged.label_status.value_counts().to_dict(),
    "cases_in_p32b_training_from_linked_ids": int(memb.p32b_fields.sum()), "exclusion_file_ids": len(set(open(T + "/feasibility/erin_fusion/erin_swg_overlap_anon_ids.txt").read().split())),
    "note": "ERIN heads are 5-fold CV over all training cases; every included case trains 4 of the 5 fold models whose average is applied to SWG, so any included case is in training for the imputation. The 21 linked patients without a case in the tables have no imaged ERIN case at all (their ERIN link is report-only)."}
# ------------------------------------------------------------------ C. subgroup characterisation
sub = pd.Series(np.where(pd.Index(pats).isin(ov), "also_in_ERIN", "never_in_ERIN"), index=pats)
g = man.groupby("patient_id"); PT = pd.DataFrame({"y": g.y.max(), "n_rows": g.size(), "first_date": g.date.min(), "last_date": g.date.max(), "first_year": g.date.min().dt.year,
    "baseline_grade": man.sort_values("date").groupby("patient_id").grade.first(), "max_grade": g.grade.max(), "biopsies_total": pd.to_numeric(coh.BiopsiesTotalForPatient, errors="coerce").groupby(man.patient_id).max(),
    "span_days": (g.date.max() - g.date.min()).dt.days}); PT["subgroup"] = sub.reindex(PT.index)
# event date per progressor = date of the earliest positive row + its DaysToNextBiopsy (the endpoint biopsy)
posr = man[man.y == 1].assign(ev=lambda d: d.date + pd.to_timedelta(d.days_to_next, unit="D")); ev = posr.groupby("patient_id").agg(event_date=("ev", "min"), endpoint_label=("next_label", "max"))
PT = PT.join(ev); PT["days_first_to_event"] = (PT.event_date - PT.first_date).dt.days
mbl = pd.to_numeric(coh.MonthsBeforeLastBiopsy, errors="coerce").groupby(man.patient_id).max(); PT["followup_months_first_to_last_biopsy"] = mbl.reindex(PT.index)
# demographics / sequencing / slides / db
dem = pd.read_csv(S + "/Demographics_full.csv", dtype=str); dem.columns = [c.strip() for c in dem.columns]; dem = dem.set_index("Study Number")
PT["sex_demographics"] = dem.Sex.reindex(PT.index); PT["age_at_diagnosis"] = pd.to_numeric(dem["Age at diagnosis"], errors="coerce").reindex(PT.index)
PT["prague_C"] = pd.to_numeric(dem["Circumference"], errors="coerce").reindex(PT.index); PT["prague_M"] = pd.to_numeric(dem["Maximal"], errors="coerce").reindex(PT.index); PT["smoking"] = dem["Smoking Status"].reindex(PT.index)
bd = pd.read_csv(S + "/barretts_database_230809.csv", dtype=str).set_index("participant_id"); ppid = coh.participant_id.str.replace(r"\.0$", "", regex=True).groupby(man.patient_id).first(); PT["gender_id_code"] = ppid.map(bd.gender_id).reindex(PT.index)
en = pd.read_parquet(E + "/mat_endoscopy_full.parquet"); en["pid"] = en.participant_id.astype(str); en["d"] = pd.to_datetime(en.endoscopydate, errors="coerce"); first_en = en.sort_values("d").groupby("pid").first()
PT["referral_hospital_db"] = ppid.map(first_en.referral_hospital).reindex(PT.index); PT["referral_hospital_db"] = PT.referral_hospital_db.fillna("not in DB")
cx = pd.read_csv(F + "/feature_views/cnv/cx.csv", dtype=str).set_index("sample_id").reindex(man.index); man["cnv_id"] = cx.cnv_id.values; man["cx"] = pd.to_numeric(cx.cx, errors="coerce").values; man["slx_run"] = man.cnv_id.str.extract(r"^(SLX-\d+)")[0]
fd = pd.read_csv(S + "/sWGS_777_samples_cleaned_202401_Leanne_fullDetails (3) (1).csv", dtype=str); fd.columns = [c.strip().replace("\n", " ") for c in fd.columns]; fd = fd.drop_duplicates("combined_name").set_index("combined_name")
va = pd.read_csv(S + "/sWGS_validation_cleaned_Leanne (4) (1).csv", dtype=str).drop_duplicates("sample_id").set_index("sample_id")
man["seq_sheet"] = np.where(man.cnv_id.isin(fd.index), "discovery_777", np.where(man.cnv_id.isin(va.index), "validation_268", "neither"))
man["seq_batch"] = man.cnv_id.map(fd.Batch); man["n_reads"] = pd.to_numeric(man.cnv_id.map(fd["Number of reads"]).str.replace(",", ""), errors="coerce"); man["cellularity"] = pd.to_numeric(man.cnv_id.map(fd["% Barrett's cellularity"]), errors="coerce"); man["p53_seqsheet"] = man.cnv_id.map(fd["p53 status"])
qc = pd.read_csv(ROW + "/swg_cnv_qc.csv").set_index("cnv_id"); man = man.join(qc[["n_bins", "noise_mapd", "sd_logratio", "n_segments", "frac_altered_0p15", "frac_altered_0p3"]], on="cnv_id")
sm = pd.read_csv(S + "/slide_matching.csv", dtype=str).drop_duplicates("Slide file").set_index("Slide file"); man["slide_file"] = coh.ImageAbsPath.map(os.path.basename).values; man["p53_ihc"] = man.slide_file.map(sm.p53IHC)
sl = pd.read_csv(ROW + "/swg_slide_meta.csv", dtype=str).set_index("sample_id").reindex(man.index); man["scanner_model"] = sl["tiff.Model"].values; man["scanner_serial"] = sl["hamamatsu.NDP.S/N"].values; man["scan_date"] = pd.to_datetime(sl["hamamatsu.Created"], format="%Y/%m/%d", errors="coerce").values
man["scan_year"] = pd.DatetimeIndex(man.scan_date).year; man["slide_age_at_scan_days"] = (man.scan_date - man.date).dt.days; man["mpp"] = pd.to_numeric(sl["openslide.mpp-x"], errors="coerce").values; man["source_lens"] = sl["hamamatsu.SourceLens"].values
gp = man.groupby("patient_id")
for c, f_ in [("n_reads_mean", ("n_reads", "mean")), ("cellularity_mean", ("cellularity", "mean")), ("cx_max", ("cx", "max")), ("noise_mapd_mean", ("noise_mapd", "mean")), ("n_segments_mean", ("n_segments", "mean")), ("frac_altered_0p15_mean", ("frac_altered_0p15", "mean")), ("frac_altered_0p30_mean", ("frac_altered_0p3", "mean")),
              ("slide_age_at_scan_days_mean", ("slide_age_at_scan_days", "mean")), ("scan_year_first", ("scan_year", "min")), ("mpp_mean", ("mpp", "mean"))]: PT[c] = gp[f_[0]].agg(f_[1]).reindex(PT.index)
for c in ["seq_sheet", "seq_batch", "slx_run", "scanner_model", "scanner_serial", "source_lens"]: PT[c + "_mode"] = gp[c].agg(lambda s: s.mode().iloc[0] if s.notna().any() else "missing").reindex(PT.index)
PT["p53_ihc_any_aberrant"] = gp.p53_ihc.agg(lambda s: ("aberrant" if (s == "aberrant").any() else ("normal" if (s == "normal").any() else np.nan))).reindex(PT.index)
PT["p53_seqsheet_any"] = gp.p53_seqsheet.agg(lambda s: ("1" if (s == "1").any() else ("0" if (s == "0").any() else np.nan))).reindex(PT.index)
PT["endpoint_label_name"] = PT.endpoint_label.map({2: "LGD (second consecutive)", 3: "HGD", 4: "IMC/cancer", 5: "IMC/cancer"})
PT["grade_source_mode"] = coh.GradeSource.groupby(man.patient_id).agg(lambda s: s.mode().iloc[0]).reindex(PT.index); PT["next_label_source_mode"] = coh.NextBiopsyLabel_source.groupby(man.patient_id).agg(lambda s: s.mode().iloc[0]).reindex(PT.index)
PT.to_csv(ROW + "/swg_patient_table.csv")
CONT = ["n_rows", "first_year", "span_days", "biopsies_total", "followup_months_first_to_last_biopsy", "days_first_to_event", "age_at_diagnosis", "prague_C", "prague_M", "n_reads_mean", "cellularity_mean", "cx_max", "noise_mapd_mean", "n_segments_mean", "frac_altered_0p15_mean", "frac_altered_0p30_mean", "slide_age_at_scan_days_mean", "scan_year_first", "mpp_mean"]
CAT = ["y", "baseline_grade", "max_grade", "endpoint_label_name", "sex_demographics", "gender_id_code", "smoking", "referral_hospital_db", "seq_sheet_mode", "seq_batch_mode", "slx_run_mode", "scanner_model_mode", "scanner_serial_mode", "source_lens_mode", "p53_ihc_any_aberrant", "p53_seqsheet_any", "grade_source_mode", "next_label_source_mode"]
a_, b_ = PT[PT.subgroup == "also_in_ERIN"], PT[PT.subgroup == "never_in_ERIN"]; rowsC = []
def fmt(x): return f"{x:.0f}" if abs(x) >= 100 else (f"{x:.1f}" if abs(x) >= 10 else f"{x:.3g}")
def qs(s): s = s.dropna(); return f"{fmt(s.median())} [{fmt(s.quantile(.25))}, {fmt(s.quantile(.75))}] (n={len(s)})" if len(s) else "n=0"
def pf(p): return None if p is None or (isinstance(p, float) and np.isnan(p)) else ("<0.001" if p < 0.001 else round(float(p), 3))
for c in CONT:
    x, z = a_[c].dropna(), b_[c].dropna(); p = mannwhitneyu(x, z).pvalue if len(x) > 1 and len(z) > 1 else np.nan
    rowsC.append({"variable": c, "type": "continuous (median [IQR])", "also_in_ERIN_54": qs(a_[c]), "never_in_ERIN_96": qs(b_[c]), "test": "Mann-Whitney", "p": pf(p)})
for c in CAT:
    tab = pd.crosstab(PT[c].fillna("missing"), PT.subgroup); 
    if tab.shape[0] < 2: rowsC.append({"variable": c, "type": "categorical", "also_in_ERIN_54": dict(tab.get("also_in_ERIN", pd.Series(dtype=int))), "never_in_ERIN_96": dict(tab.get("never_in_ERIN", pd.Series(dtype=int))), "test": "constant", "p": None}); continue
    if tab.shape[0] == 2: p = fisher_exact(tab.values)[1]; tn = "Fisher exact"
    else: p = chi2_contingency(tab.values)[1]; tn = f"chi-square ({tab.shape[0]} levels; Fisher only defined for 2x2)"
    rowsC.append({"variable": c, "type": "categorical (counts)", "also_in_ERIN_54": {k: int(v) for k, v in tab["also_in_ERIN"].items()}, "never_in_ERIN_96": {k: int(v) for k, v in tab["never_in_ERIN"].items()}, "test": tn, "p": pf(p)})
RES["C_characterisation"] = rowsC; md("C. Subgroup characterisation (patient level)", pd.DataFrame(rowsC).astype(str))
RES["C_not_available"] = ["staining batch (no record in any SWG table)", "referral pathway beyond DB referral_hospital of first endoscopy", "sequencing depth in x (reads only; read length not recorded in the sheet)", "sex for patients absent from Demographics_full.csv (gender_id code from barretts_database_230809.csv reported as code, mapping not documented)", "p53 IHC for slides with blank p53IHC in slide_matching.csv", "4x resequencing (none exists; see L)"]
# per-subgroup AUROCs with the canonical head
colsS = {"img": man.img.values, "cnv": man.cnv.values, "head": grade_n, "fuse2": fuse(man.img, man.cnv), "fuse3": fuse(man.img, man.cnv, grade_n)}
y, P, pats = patient(colsS); grp = sub.reindex(pats).values; rowsS = []
for gname in ["all", "also_in_ERIN", "never_in_ERIN"]:
    m_ = np.ones(len(y), bool) if gname == "all" else grp == gname; yy = y[m_]; Bg = boots(yy, seed=SEED); r = {"subgroup": gname, "n": int(m_.sum()), "events": int(yy.sum())}
    for k in colsS: r[k] = r3(auc(yy, P[k][m_])); r[k + "_ci"] = ci(yy, Bg, P[k][m_])
    r["gain"] = r3(auc(yy, P["fuse3"][m_]) - auc(yy, P["fuse2"][m_])); r["gain_ci"] = ci(yy, Bg, P["fuse3"][m_], P["fuse2"][m_]); rowsS.append(r)
RES["C_subgroup_aurocs"] = rowsS; md("C. Arm AUROCs per subgroup (canonical leak-free grade head)", pd.DataFrame(rowsS))
# CNV within strata
PTi = PT.reindex(pats); rowsK = []
for strat, lab in [("seq_sheet_mode", "sequencing sheet (discovery vs validation)"), ("slx_run_mode", "SLX run"), ("seq_batch_mode", "Leanne batch (discovery only)")]:
    for level in sorted(PTi[strat].dropna().unique()):
        for gname in ["also_in_ERIN", "never_in_ERIN"]:
            m_ = (PTi[strat].values == level) & (grp == gname); yy = y[m_]
            if m_.sum() >= 10 and 0 < yy.sum() and yy.sum() < m_.sum(): rowsK.append({"stratum": f"{lab}={level}", "subgroup": gname, "n": int(m_.sum()), "events": int(yy.sum()), "cnv_auroc": r3(auc(yy, P["cnv"][m_])), "cnv_ci": ci(yy, boots(yy), P["cnv"][m_]) if yy.sum() >= 3 else None})
            else: rowsK.append({"stratum": f"{lab}={level}", "subgroup": gname, "n": int(m_.sum()), "events": int(yy.sum()), "cnv_auroc": None, "cnv_ci": None})
rt = pd.qcut(PTi.n_reads_mean, 3, labels=["reads_T1_low", "reads_T2", "reads_T3_high"])
for level in ["reads_T1_low", "reads_T2", "reads_T3_high"]:
    for gname in ["also_in_ERIN", "never_in_ERIN"]:
        m_ = (rt.values == level) & (grp == gname); yy = y[m_]
        rowsK.append({"stratum": f"read-count tertile={level}", "subgroup": gname, "n": int(m_.sum()), "events": int(yy.sum()), "cnv_auroc": r3(auc(yy, P["cnv"][m_])) if 0 < yy.sum() < m_.sum() else None, "cnv_ci": ci(yy, boots(yy), P["cnv"][m_]) if yy.sum() >= 3 and yy.sum() < m_.sum() else None})
RES["C_cnv_by_stratum"] = rowsK; md("C. CNV-only AUROC within sequencing strata by subgroup", pd.DataFrame(rowsK))
RES["C_4x"] = "NOT AVAILABLE: no resequenced data exists; see item L"
# ------------------------------------------------------------------ D. interaction (pre-specified)
gA = grp == "also_in_ERIN"; gN = ~gA
def gain(m_, yy=None, P_=None): yy = y if yy is None else yy; P_ = P if P_ is None else P_; return auc(yy[m_], P_["fuse3"][m_]) - auc(yy[m_], P_["fuse2"][m_])
obs = gain(gA) - gain(gN); rng = np.random.RandomState(SEED); bs = []
while len(bs) < NB:
    sA = rng.choice(np.where(gA)[0], gA.sum()); sN = rng.choice(np.where(gN)[0], gN.sum())
    if len(set(y[sA])) < 2 or len(set(y[sN])) < 2: continue
    bs.append((auc(y[sA], P["fuse3"][sA]) - auc(y[sA], P["fuse2"][sA])) - (auc(y[sN], P["fuse3"][sN]) - auc(y[sN], P["fuse2"][sN])))
def perm_membership(preserve_events):
    rng = np.random.RandomState(SEED); null = []
    for _ in range(NB):
        lab = np.zeros(len(y), bool)
        if preserve_events:
            for cls in (0, 1): idx = np.where(y == cls)[0]; lab[rng.choice(idx, int((gA & (y == cls)).sum()), replace=False)] = True
        else: lab[rng.choice(len(y), int(gA.sum()), replace=False)] = True
        if len(set(y[lab])) < 2 or len(set(y[~lab])) < 2: null.append(np.nan); continue
        null.append(gain(lab) - gain(~lab))
    null = np.array(null); null = null[~np.isnan(null)]
    return {"n_valid_perms": int(len(null)), "p_two_sided": round(float((1 + (np.abs(null) >= abs(obs)).sum()) / (len(null) + 1)), 4), "p_one_sided_ge": round(float((1 + (null >= obs).sum()) / (len(null) + 1)), 4), "null_q025_q975": [r3(np.percentile(null, 2.5)), r3(np.percentile(null, 97.5))]}
RES["D_interaction"] = {"gain_also_in_ERIN": r3(gain(gA)), "gain_never_in_ERIN": r3(gain(gN)), "difference": r3(obs), "bootstrap_ci_stratified": [r3(np.percentile(bs, 2.5)), r3(np.percentile(bs, 97.5))], "n_boot": NB, "seed": SEED,
    "perm_sizes_only": perm_membership(False), "perm_sizes_and_events": perm_membership(True), "n_events": {"also_in_ERIN": [int(gA.sum()), int(y[gA].sum())], "never_in_ERIN": [int(gN.sum()), int(y[gN].sum())]}}
# ------------------------------------------------------------------ E. prevalent vs future
prog = PT[PT.y == 1]; iv = prog.days_first_to_event; bins = [-1, 183, 365, 730, 1095, 1826, 1e9]; labels = ["0-6m", "6-12m", "12-24m", "24-36m", "36-60m", ">60m"]
RES["E_interval"] = {"n_progressors": int(len(prog)), "n_with_interval": int(iv.notna().sum()), "median_days": r3(iv.median()), "iqr_days": [r3(iv.quantile(.25)), r3(iv.quantile(.75))], "min_max": [r3(iv.min()), r3(iv.max())],
    "hist_bins": pd.cut(iv, bins, labels=labels).value_counts().reindex(labels).astype(int).to_dict(), "definition": "earliest release row date -> date of the endpoint biopsy (positive row date + DaysToNextBiopsy, earliest positive row)",
    "secondary_DaysFromCurrentToEvent_min_per_progressor": {"median": r3(man[man.y == 1].groupby("patient_id").d_cur_event.min().median()), "n": int(man[man.y == 1].groupby("patient_id").d_cur_event.min().notna().sum())},
    "progressors_with_single_row": int((prog.n_rows == 1).sum())}
rowsE = []
for thr, name in [(0, "none"), (183, "exclude progressors with event <= 6 m from first row"), (365, "exclude progressors with event <= 12 m from first row")]:
    keep_p = set(PT.index) - set(prog[prog.days_first_to_event <= thr].index) if thr else set(PT.index); m_ = np.isin(pats, list(keep_p)); yy = y[m_]; Bg = boots(yy)
    r = {"exclusion": name, "n": int(m_.sum()), "events": int(yy.sum())}
    for k in ["head", "fuse2", "fuse3"]: r[k] = r3(auc(yy, P[k][m_])); r[k + "_ci"] = ci(yy, Bg, P[k][m_])
    r["gain"] = r3(auc(yy, P["fuse3"][m_]) - auc(yy, P["fuse2"][m_])); r["gain_ci"] = ci(yy, Bg, P["fuse3"][m_], P["fuse2"][m_]); rowsE.append(r)
for thr, name in [(183, "SECONDARY: drop positive ROWS with DaysToNextBiopsy <= 183 (patient keeps other rows)"), (365, "SECONDARY: drop positive ROWS with DaysToNextBiopsy <= 365")]:
    keep = ~((man.y == 1) & (man.days_to_next <= thr)).values; _, Pk, pk = patient(colsS, rows=keep); yy = PT.y.reindex(pk).values.astype(int); Bg = boots(yy)   # patient label = ORIGINAL progressor status; only rows are dropped
    r = {"exclusion": name, "n": int(len(yy)), "events": int(yy.sum()), "rows_kept": int(keep.sum())}
    for k in ["head", "fuse2", "fuse3"]: r[k] = r3(auc(yy, Pk[k])); r[k + "_ci"] = ci(yy, Bg, Pk[k])
    r["gain"] = r3(auc(yy, Pk["fuse3"]) - auc(yy, Pk["fuse2"])); r["gain_ci"] = ci(yy, Bg, Pk["fuse3"], Pk["fuse2"]); rowsE.append(r)
RES["E_exclusions"] = rowsE; md("E. Excluding near-baseline progressors", pd.DataFrame(rowsE).astype(str))
# ------------------------------------------------------------------ F. baseline grade
fams = sorted(os.path.basename(p) for p in glob.glob(F + "/training_final_nested_cv_v1/*") if os.path.isdir(p) and glob.glob(p + "/fold1/outer_test_predictions.csv"))
gr = man.grade.fillna(0).values; maxsofar = pd.to_numeric(coh.MaxPathologySoFar, errors="coerce").fillna(0).values
colsF = {"grade_arm": cvlog(gr[:, None], y_s, folds), "grade_plus_head": fuse(cvlog(gr[:, None], y_s, folds), grade_n), "grade_maxsofar_arm": cvlog(np.column_stack([gr, maxsofar]), y_s, folds), "grade_maxsofar_plus_head": fuse(cvlog(np.column_stack([gr, maxsofar]), y_s, folds), grade_n),
         "fuse2_plus_grade": fuse(man.img, man.cnv, cvlog(gr[:, None], y_s, folds)), "fuse3_plus_grade": fuse(man.img, man.cnv, grade_n, cvlog(gr[:, None], y_s, folds)), "head": grade_n, "fuse2": fuse(man.img, man.cnv), "fuse3": fuse(man.img, man.cnv, grade_n), "grade_raw_max": gr}
yF, PF, pF = patient(colsF); BF = boots(yF); rowsF = [{"arm": k, "n": len(yF), "events": int(yF.sum()), "auroc": r3(auc(yF, PF[k])), "ci": ci(yF, BF, PF[k])} for k in colsF]
RES["F_arms"] = {"release_families": fams, "clinical_arm_exists": False, "grade_coding": "pre_event_cohort.Label per row: 0 NDBE, 1 IND, 2 LGD (no HGD+ rows in the 707); MaxPathologySoFar = max grade up to the row", "rows": rowsF}
md("F. Grade arms (all 150 patients)", pd.DataFrame(rowsF).astype(str))
rowsFn = []
for name, keep_p in [("first release row NDBE (Label 0)", PT.index[PT.baseline_grade == 0]), ("all release rows NDBE", PT.index[PT.max_grade == 0])]:
    m_ = np.isin(pF, keep_p); yy = yF[m_]; Bg = boots(yy); r = {"definition": name, "n": int(m_.sum()), "events": int(yy.sum())}
    for k in ["head", "fuse2", "fuse3", "img", "cnv"]:
        v = PF[k][m_] if k in PF else P[k][np.isin(pats, keep_p)]; r[k] = r3(auc(yy, v)); r[k + "_ci"] = ci(yy, Bg, v)
    r["gain"] = r3(auc(yy, PF["fuse3"][m_]) - auc(yy, PF["fuse2"][m_])); r["gain_ci"] = ci(yy, Bg, PF["fuse3"][m_], PF["fuse2"][m_]); rowsFn.append(r)
RES["F_within_NDBE"] = rowsFn; md("F. Within baseline-NDBE patients", pd.DataFrame(rowsFn).astype(str))
# ------------------------------------------------------------------ G. sanity gate reconciliation
lab2 = (man.grade.fillna(0).values >= 2).astype(int)
def stem(x): x = str(x).lower().strip(); x = re.sub(r"[\s_].*$", "", x); return re.sub(r"[^a-z0-9]", "", x)
smr = pd.read_parquet(E + "/swg_matched_reports_v2.parquet"); rows_ = []
for r in smr.itertuples():
    for p_ in str(r.swg_path_ids).split(";"):
        if p_.strip(): rows_.append({"stem": stem(p_), "conf": pd.to_numeric(r.highestgradedysconf, errors="coerce"), "rid": str(r.pathology_text_id)})
rep = pd.DataFrame(rows_).drop_duplicates("stem").set_index("stem"); MAP = {2: 0, 3: 1, 4: 2, 5: 3, 6: 4, 8: 4}
stems = coh.BiopsyID_real.map(stem); conf = stems.map(rep.conf); okc = conf.isin(MAP).values; db2 = (conf[okc].map(MAP).values >= 2).astype(int)
jg = None
try:
    jo = pd.read_csv(f"{T}/feasibility/runs/swg_text_arm/output/swg_text_arm_oof.csv", dtype={"sample_id": str}).set_index("sample_id")
    if "jury_grade" in jo: jg = jo.jury_grade.reindex(man.index)
except Exception: pass
G = {"metric": "slide-level AUROC of the head's imputed P(LGD+) against the binary pathologist grade LGD+ (Label >= 2) on the 707 release rows; the 707 rows contain no HGD+ so LGD+ = LGD", "n_rows": 707, "n_LGD_rows": int(lab2.sum()),
     "grade_source_of_Label_707": coh.GradeSource.value_counts().to_dict(), "observed": {k: r3(auc(lab2, IMP[k].grade_LGDplus.values)) for k in ["pass1", "p32b", "repeat", "perm", "noov"]},
     "vs_db_confirmed_code": {"n_rows_with_code": int(okc.sum()), "n_LGDplus_by_code": int(db2.sum()), **{k: r3(auc(db2, IMP[k].grade_LGDplus.values[okc])) for k in ["p32b", "noov"]}},
     "read_vs_read_on_coded_rows": {"Label_vs_DBcode_two_tier_agreement": r3(((lab2[okc]) == db2).mean()), "kappa": r3(cohen_kappa_score(lab2[okc], db2)), "auroc_Label_as_score_vs_DBcode": r3(auc(db2, lab2[okc])), "n": int(okc.sum()),
                                   "note": "GradeSource shows Label is itself 'scraped_confirmed' (the DB code) for most rows, so this is not an independent second pathologist read"},
     "head_vs_each_read_same_rows": {"vs_Label": {k: r3(auc(lab2[okc], IMP[k].grade_LGDplus.values[okc])) for k in ["p32b", "noov"]}, "vs_DBcode": {k: r3(auc(db2, IMP[k].grade_LGDplus.values[okc])) for k in ["p32b", "noov"]}},
     "second_pathologist_read": "NONE: highestgradedysresearch is 7 (not done) on 325/327 matched reports; no other pathologist re-read table exists in the release, the SWGCohort folder or the DB export"}
if jg is not None:
    okj = jg.notna().values; jb = (jg[okj].values >= 2).astype(int)
    G["vs_LLM_jury_grade_of_report_NOT_a_pathologist"] = {"n_rows": int(okj.sum()), "jury_LGDplus_rows": int(jb.sum()), "head_p32b_vs_jury": r3(auc(jb, IMP["p32b"].grade_LGDplus.values[okj])), "head_noov_vs_jury": r3(auc(jb, IMP["noov"].grade_LGDplus.values[okj])), "Label_vs_jury_two_tier": r3((lab2[okj] == jb).mean())}
RES["G_sanity"] = G
# ------------------------------------------------------------------ I. selection across fields
cands = {}
for c in IMP["pass1"].columns: cands[f"pass1_0.88um:{c}"] = IMP["pass1"][c].fillna(IMP["pass1"][c].mean()).values
cands["pass1_0.88um:logistic_all13"] = cvlog(IMP["pass1"].fillna(IMP["pass1"].mean()).values, y_s, folds)
for c in IMP["p32b"].columns: cands[f"pass2_0.5um:{c}"] = IMP["p32b"][c].values
cands["pass2_0.5um:logistic_all6"] = cvlog(IMP["p32b"].values, y_s, folds)
for c in IMP["p32b"].columns: cands[f"pass2_0.5um:logistic_without_{c}"] = cvlog(IMP["p32b"].drop(columns=[c]).values, y_s, folds)
cands["pass2_0.5um:img2_rep02_ensemble_control"] = man.img2.values
colsI = {"fuse2": fuse(man.img, man.cnv), **{f"f3:{k}": fuse(man.img, man.cnv, v) for k, v in cands.items()}}
yI, PI, _ = patient(colsI); obsI = {k: auc(yI, PI[f"f3:{k}"]) - auc(yI, PI["fuse2"]) for k in cands}; best = max(obsI, key=obsI.get); tI = obsI[best]
rng = np.random.RandomState(SEED); perms = [rng.permutation(len(yI)) for _ in range(NB)]; K = [f"f3:{k}" for k in cands]
nullI = np.array([max(auc(yI[ix], PI[k]) - auc(yI[ix], PI["fuse2"]) for k in K) for ix in perms])
canon = auc(yI, P["fuse3"]) - auc(yI, P["fuse2"])
RES["I_selection"] = {"n_candidate_third_arms": len(cands), "candidates_observed_gain": {k: r3(v) for k, v in obsI.items()}, "selected_max": best, "t_obs_max": r3(tI), "p_selection_adjusted_max_over_all_candidates": round(float((1 + (nullI >= tI).sum()) / (NB + 1)), 4),
    "p_unadjusted_for_selected": round(float((1 + (np.array([auc(yI[ix], PI[f"f3:{best}"]) - auc(yI[ix], PI["fuse2"]) for ix in perms]) >= tI).sum()) / (NB + 1)), 4), "canonical_noov_grade_gain": r3(canon), "p_canonical_against_max_null": round(float((1 + (nullI >= canon).sum()) / (NB + 1)), 4),
    "what_was_maximised_over": "every third arm ever fused with image + CNV on SWG: 13 pass-1 single fields (0.88 um imputation), pass-1 all-13 logistic, 6 pass-2 single fields (0.5 um), pass-2 all-6 logistic, 6 pass-2 leave-one-field-out logistics, the rep02 image ensemble control; fusion rule fixed (fold-local z-mean) so not maximised over", "n_perm": NB, "seed": SEED}
# ------------------------------------------------------------------ J. recompute P32 ERIN CIs at 2,000 (were 1,000)
mm2 = mm.dropna(subset=["h5", "anon_id"]).drop_duplicates("CaseName"); c2a = dict(zip(mm2.CaseName, mm2.anon_id)); rowsJ = []
for d, tag in [(R + "/p32b_fields/output", "p32b_fields"), (R + "/p32_head_noov/output", "p32_head_noov")] + [(R + f"/p32_fields_g{i}/output", f"p32_fields_g{i}") for i in range(4)]:
    for f_ in sorted(glob.glob(d + "/oof_*.csv")):
        o = pd.read_csv(f_); o["anon"] = o.CaseName.map(c2a); o = o.dropna(subset=["anon"]); yy = o.y.values.astype(int); pp = o.p.values; gg = o.anon.values; up = np.unique(gg); idx_of = {u: np.where(gg == u)[0] for u in up}
        rng = np.random.RandomState(0); Bc = []
        while len(Bc) < NB:
            s_ = np.concatenate([idx_of[u] for u in rng.choice(up, len(up))])
            if len(set(yy[s_])) > 1: Bc.append(auc(yy[s_], pp[s_]))
        rowsJ.append({"run": tag, "field": os.path.basename(f_)[4:-4], "n_cases": int(len(o)), "n_patients": int(len(up)), "pos": int(yy.sum()), "auroc": r3(auc(yy, pp)), "ci_2000_patient_clustered": [r3(np.percentile(Bc, 2.5)), r3(np.percentile(Bc, 97.5))]})
RES["J_p32_erin_cis_2000"] = rowsJ; md("J. P32 ERIN field AUROCs with 2,000 patient-clustered resamples (source runs used 1,000)", pd.DataFrame(rowsJ).astype(str))
# ------------------------------------------------------------------ K. fold grouping assertions
K_ = {"swg_rep01_patients_in_more_than_one_outer_fold": int((man.groupby("patient_id").fold.nunique() > 1).sum()), "swg_rep02_patients_in_more_than_one_outer_fold": int((man.groupby("patient_id").fold2.nunique() > 1).sum()), "swg_inner_folds": {}}
for fam in ["image_only", "cnv_only"]:
    bad = 0
    for fp in sorted(glob.glob(F + f"/training_final_nested_cv_v1/{fam}/fold*/inner_fold_assignments.csv")):
        ia = pd.read_csv(fp, dtype=str); bad += int((ia.groupby("patient_id").inner_fold.nunique() > 1).sum())
    K_["swg_inner_folds"][fam] = {"patients_in_more_than_one_inner_fold": bad, "note": "inner_fold_assignments.csv is keyed by patient_id: inner folds are patient-level by construction"}
K_["erin_tasks"] = {}
for tp in sorted(glob.glob(T + "/feasibility/erin_fusion/tasks/T*.csv")):
    d = pd.read_csv(tp, dtype=str); keys = list(d.sample_id); pat = dict(zip(keys, d.anon_id)); yv = dict(zip(keys, d.y.astype(int))); fl = patient_folds(keys, pat, yv, 5, seed=0)
    fold_of = {k: i for i, f_ in enumerate(fl) for k in f_}; pf = pd.Series({k: fold_of[k] for k in keys}).groupby(pd.Series(pat)).nunique()
    K_["erin_tasks"][os.path.basename(tp)[:-4]] = {"n_units": len(keys), "n_patients": int(len(pf)), "patients_in_more_than_one_fold": int((pf > 1).sum()), "units_assigned": len(fold_of)}
p31 = pd.read_csv(R + "/p31_validate_v2/output/p31_fields.csv", dtype=str); dP = mm2.merge(p31, on="CaseName"); keys = list(dP.CaseName); pat = dict(zip(keys, dP.anon_id)); yv = dict(zip(keys, dP.grade.isin(["LGD", "HGD", "CANCER"]).astype(int)))
fl = patient_folds(keys, pat, yv, 5, seed=0); fold_of = {k: i for i, f_ in enumerate(fl) for k in f_}; pf = pd.Series({k: fold_of[k] for k in keys}).groupby(pd.Series(pat)).nunique()
K_["p32_erin_heads"] = {"n_cases": len(keys), "n_patients": int(len(pf)), "patients_in_more_than_one_fold": int((pf > 1).sum())}
K_["aggregation"] = {"swg_unit": "release row = one slide + one CNV profile (707 rows; 26 rows share a CNV profile with another row)", "patient_score": "max over the patient's rows of the arm score (after fold-local z for fused arms)", "patient_label": "max of y_progressor over rows",
    "baseline": "the release has no single baseline row: every strict pre-event row of a patient is a unit in training and in evaluation; 'baseline' in items C/E means the earliest release row", "training": "rows of one patient always share an outer fold (assertion above) so no patient is split across train/test"}
RES["K_folds"] = K_
# ------------------------------------------------------------------ L. definitions from code / config
meta = json.load(open(F + "/cohort_release_metadata.json")); smeta = json.load(open(F + "/split_release_metadata.json")); tasks = json.load(open(F + "/tasks_chapter1_lgd2_final.json"))
RES["L_ours"] = {"endpoint_name": meta["endpoint"], "endpoint_rule_code": "src/barrett/labels/lgd2.py derive_next_biopsy_lgd2plus: positive if next biopsy label >= HGD, or next biopsy = LGD and current LGD streak >= 1 (two consecutive LGD)",
    "eligibility": "strict pre-event rows only: exclusion_counts " + json.dumps(meta["exclusion_counts"]), "rows_patients": [smeta["n_units"], smeta["n_patients"]], "progressor_patients": pos_all, "positive_rows": int(y_s.sum()),
    "split": f"{smeta['n_folds']}-fold patient-level, split_seed {smeta['split_seed']}; nested inner CV per outer fold", "primary_metric_release": tasks["tasks"][0]["primary_metric"], "cnv_representation": "QDNAseq 50 kb bins -> arm-level features (features_arms.csv, 39 arm columns) + complexity cx",
    "depth_statement": "0.4x stated in docs/NUMBERS.md (Leanne's slide, 6 Mar 2026); not verifiable from data here; read counts from the sequencing sheet summarised in item C", "reads_median_per_sample_sheet": r3(man.n_reads.median()), "reads_n_samples_with_value": int(man.n_reads.notna().sum()),
    "killcoyne_discovery_overlap_by_cnv_id": int(man.cnv_id.isin(fd.index).sum()), "validation_sheet_cnv_ids": int(man.cnv_id.isin(va.index).sum()), "code_commit_release": meta["code_git_commit"], "fold1_git_commit": json.load(open(F + "/training_final_nested_cv_v1/image_only/fold1/fold_metadata.json"))["git_commit"]}
RES["L_4x"] = {"status": "NOT AVAILABLE", "looked": [S + " (50kb, copy_number_hg38/train, dna_seq_bam: 1,035 BAMs, validation_genomics: id lists only)", "barretts_training/leanne_files_25_03_2026 (777 discovery + 268 validation sample sheets, no depth column)", "docs/NUMBERS.md, docs/status_ledger_2026-09-24.md item 2, questions_for_people Q2 (question to Leanne, unanswered)"]}
# ------------------------------------------------------------------ M. calibration and decision curve
def platt_cv(score, fl):
    p = np.zeros(len(score))
    for f in np.unique(fl): te = fl == f; tr = ~te; lr = LogisticRegression(C=1e6, max_iter=5000).fit(score[tr][:, None], y_s[tr]); p[te] = lr.predict_proba(score[te][:, None])[:, 1]
    return p
lm = (man.img.values + man.cnv.values) / 2; z2 = fuse(man.img, man.cnv); z3 = fuse(man.img, man.cnv, grade_n)
colsM = {"image_only_prob": man.img.values, "late_mean_prob_release": lm, "fuse2_z_plattCV": platt_cv(z2, folds), "fuse3_head_z_plattCV": platt_cv(z3, folds), "head_prob": grade_n}
yM, PM, _ = patient(colsM); BM = boots(yM); rowsM = []; thr = np.round(np.arange(0.10, 0.501, 0.05), 2)
def calib(yy, pp):
    lp = np.log(np.clip(pp, 1e-6, 1 - 1e-6) / (1 - np.clip(pp, 1e-6, 1 - 1e-6))); sl = LogisticRegression(C=1e6, max_iter=5000).fit(lp[:, None], yy)
    inter = float(np.log(yy.mean() / (1 - yy.mean())) - np.log(pp.mean() / (1 - pp.mean())))   # calibration-in-the-large: logit(observed) - logit(mean predicted)
    return float(sl.coef_[0][0]), float(sl.intercept_[0]), inter
for k, v in PM.items():
    slope, int_slopefit, itl = calib(yM, v); br = float(np.mean((v - yM) ** 2)); brc = [r3(np.percentile([np.mean((v[s] - yM[s]) ** 2) for s in BM], q)) for q in (2.5, 97.5)]
    nb = {}
    for t in thr: pred = v >= t; tp = (pred & (yM == 1)).sum() / len(yM); fp = (pred & (yM == 0)).sum() / len(yM); nb[str(t)] = r3(tp - fp * t / (1 - t))
    rowsM.append({"arm": k, "n": len(yM), "events": int(yM.sum()), "auroc": r3(auc(yM, v)), "mean_pred": r3(v.mean()), "calib_slope": r3(slope), "calib_intercept_at_slope_fit": r3(int_slopefit), "calib_in_the_large": r3(itl), "brier": r3(br), "brier_ci": brc, "net_benefit": nb})
nb_all = {str(t): r3(yM.mean() - (1 - yM.mean()) * t / (1 - t)) for t in thr}
RES["M_calibration"] = {"rows": rowsM, "treat_all_net_benefit": nb_all, "treat_none": 0.0, "prevalence": r3(yM.mean()), "note": "fused arms are z-score means with no probability scale; fuse2/fuse3 are put on a probability scale by a Platt logistic fitted on the other four outer folds (CV), then max over rows; image_only and late_mean are release probabilities"}
md("M. Calibration, Brier and decision-curve net benefit (patient level)", pd.DataFrame([{**{kk: vv for kk, vv in r.items() if kk != "net_benefit"}, **{"NB@" + kk: vv for kk, vv in r["net_benefit"].items()}} for r in rowsM]).astype(str))
# p53 IHC
p53 = PT.p53_ihc_any_aberrant.reindex(pats); okp = p53.notna().values; yp = y[okp]; xp = (p53[okp].values == "aberrant").astype(float); Bp = boots(yp)
hp = P["head"][okp]; f2p = P["fuse2"][okp]; comb = (rankdata(xp) / len(xp) + rankdata(hp) / len(hp)) / 2; comb2 = (rankdata(xp) / len(xp) + rankdata(f2p) / len(f2p)) / 2
RES["M_p53"] = {"source": "slide_matching.csv p53IHC per slide (normal/aberrant/blank); patient = aberrant if any release slide aberrant, normal if any normal and none aberrant", "n_patients_with_ihc": int(okp.sum()), "events": int(yp.sum()), "n_aberrant": int(xp.sum()),
    "p53_auroc": [r3(auc(yp, xp))] + ci(yp, Bp, xp), "head_auroc_same_patients": [r3(auc(yp, hp))] + ci(yp, Bp, hp), "p53_plus_head_rankmean": [r3(auc(yp, comb))] + ci(yp, Bp, comb), "delta_p53_plus_head_minus_p53": ci(yp, Bp, comb, xp),
    "fuse2_same_patients": [r3(auc(yp, f2p))] + ci(yp, Bp, f2p), "fuse2_plus_p53_rankmean": [r3(auc(yp, comb2))] + ci(yp, Bp, comb2), "delta_fuse2_plus_p53_minus_fuse2": ci(yp, Bp, comb2, f2p), "note": "rank-mean combination (no fitted weights) because a fitted combination on 150 patients would need its own CV"}
# ------------------------------------------------------------------ write
json.dump(RES, open(AGG + "/closeout_main.json", "w"), indent=1, default=str); open(AGG + "/tables.md", "w").write("\n".join(MD)); print(json.dumps({k: v for k, v in RES.items() if k in ("A_versions", "D_interaction", "E_interval", "G_sanity", "I_selection", "K_folds", "M_p53")}, indent=1, default=str)[:12000])

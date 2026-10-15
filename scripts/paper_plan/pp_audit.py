"""Paper plan item 0 (pre-specified in docs/paper_plan_answers.md @ db236a0): CNV/label chain audit of the 54 also_in_ERIN
patients and 25 never_in_ERIN comparators. Row-level table -> feasibility/paper_plan/audit_rows.csv (cluster only);
summary -> results/paper_plan/audit_item0.json. Also: Killcoyne 2020 per-sample predictions vs ours; count reconciliation;
the pre-specified confirmed-error decision."""
import glob, json, os, re, numpy as np, pandas as pd
from scipy.stats import spearmanr, rankdata
F = "/mnt/scratche/slow/fmlab/zuberi01/phd/barretts_retraining/barretts_training/analysis/chapter1_lgd2_final_pre_event_20260713_final"
T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"; S = "/mnt/scratche/fast/fmlab/datasets/imaging/SWGCohort"; E = "/mnt/scratche/slow/fmlab/zuberi01/barretts_db_export"; K = "/mnt/scratche/slow/fmlab/zuberi01/phd/killcoyne_data_from_paper"
ROW = T + "/feasibility/paper_plan"; AGG = os.environ.get("OUTDIR", T + "/results/paper_plan"); os.makedirs(ROW, exist_ok=True); os.makedirs(AGG, exist_ok=True)
def auc(y, s):
    y = np.asarray(y).astype(int); r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1; return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if 0 < n1 < len(y) else float("nan")
def stem(x):
    x = str(x).strip().lower()
    if x in ("", "nan", "none"): return None
    x = re.split(r"[\s_]", x)[0]; x = x.split("-")[0] if re.match(r"^ps\d\d\.\d+-", x) else x
    m = re.match(r"^s?(?:ps)?(\d{2})[.\-]?(\d{3,6})$", x.replace("ps", "", 1) if x.startswith("ps") else x)
    if x.startswith("ps"): m = re.match(r"^ps(\d{2})[.\-]?(\d{3,6})$", x)
    elif x.startswith("s"): m = re.match(r"^s(\d{2})[.\-]?(\d{3,6})$", x)
    return f"ps{m.group(1)}{int(m.group(2))}" if m else re.sub(r"[^a-z0-9]", "", x)
def slide_stem(fn):   # v2 parser: 'PS00 4239 B2100 1', 'S08 6265 2 1 G3497', 'PSyy.nnnnn 1 L1-3', '0009H053851 PR1 HIN 042 B2506' (no accession -> None)
    t = str(fn).replace(".ndpi", "").split()
    if not t: return None
    if re.match(r"^(ps|PS)\d\d\.\d+", t[0]): return stem(t[0])
    if re.match(r"^(ps|PS|s|S)\d\d$", t[0]) and len(t) > 1 and t[1].isdigit(): return f"ps{t[0][-2:]}{int(t[1])}"
    return None
man = pd.read_csv(F + "/training_manifest.csv", dtype=str).set_index("sample_id"); coh = pd.read_csv(F + "/pre_event_cohort.csv", dtype=str).set_index("SampleID").loc[man.index]
cx = pd.read_csv(F + "/feature_views/cnv/cx.csv", dtype=str).set_index("sample_id").reindex(man.index)
d = pd.DataFrame({"patient_id": man.patient_id, "y_row": man.y_progressor.astype(int), "fold": man.fold_id_rep01, "cnv_id": cx.cnv_id, "acc": coh.BiopsyID_real, "date": pd.to_datetime(coh.Date, errors="coerce"), "date_source": coh.DateSource, "acc_match": coh.BiopsyIDMatchType,
                  "grade": pd.to_numeric(coh.Label, errors="coerce"), "grade_source": coh.GradeSource, "next": pd.to_numeric(coh.NextBiopsyLabel, errors="coerce"), "next_source": coh.NextBiopsyLabel_source, "next_orig": pd.to_numeric(coh.NextBiopsyLabel_orig, errors="coerce"), "next_scrape": pd.to_numeric(coh.NextBiopsyLabel_scrape, errors="coerce"),
                  "slide": coh.ImageAbsPath.map(os.path.basename), "participant": coh.participant_id.str.replace(r"\.0$", "", regex=True)}, index=man.index)
d["acc_stem"] = d.acc.map(stem); d["slide_stem"] = d.slide.map(slide_stem)
pairs = pd.read_csv(T + "/feasibility/closeout/erin_swg_pairs.csv", dtype=str); ov = sorted(set(pairs.swg_patient_id)); never = sorted(set(d.patient_id) - set(ov)); comp = sorted(np.random.RandomState(0).choice(never, 25, replace=False))
d["subgroup"] = np.where(d.patient_id.isin(ov), "also_in_ERIN", np.where(d.patient_id.isin(comp), "comparator_25", "other"))
# sources
sm = pd.read_csv(S + "/slide_matching.csv", dtype=str); sm["pid_num"] = sm.PatientID.str.replace(r"\.0$", "", regex=True); num2alt = sm.dropna(subset=["AlternateID"]).drop_duplicates("pid_num").set_index("pid_num").AlternateID.to_dict(); smf = sm.drop_duplicates("Slide file").set_index("Slide file")
fd = pd.read_csv(S + "/sWGS_777_samples_cleaned_202401_Leanne_fullDetails (3) (1).csv", dtype=str); fd.columns = [c.strip().replace("\n", " ") for c in fd.columns]; fd = fd.drop_duplicates("combined_name").set_index("combined_name")
va = pd.read_csv(S + "/sWGS_validation_cleaned_Leanne (4) (1).csv", dtype=str).drop_duplicates("sample_id").set_index("sample_id")
pt = pd.read_csv(E + "/pathology_text_normalised_full.csv", dtype=str, usecols=["participant_id", "specimennumber", "receiveddatetime", "highestgradedysconf"]); pt["stem"] = pt.specimennumber.map(stem); pt["d"] = pd.to_datetime(pt.receiveddatetime, errors="coerce"); pt["code"] = pd.to_numeric(pt.highestgradedysconf, errors="coerce")
MAP = {2: 0, 3: 1, 4: 2, 5: 3, 6: 4, 8: 4}; pt["g"] = pt.code.map(MAP); ptc = pt.dropna(subset=["stem", "g"]).drop_duplicates("stem").set_index("stem")
G_SHEET = {"BE": 0, "NDBE": 0, "ID": 1, "IND": 1, "LGD": 2, "HGD": 3, "IMC": 4, "OAC": 4, "EAC": 4}
def tier(g): return None if g is None or (isinstance(g, float) and np.isnan(g)) else int(g >= 2)
# v2: the discovery sheet's numeric PatientID is a different numbering from slide_matching.PatientID; map each numeric id to the
# release patient that the majority of its cnv_ids belong to, and flag rows whose sheet id maps elsewhere (mixed-patient sheet id)
fdnum = fd.PatientID.astype(str).str.replace(".0", "", regex=False); tmp = d[["patient_id", "cnv_id"]].dropna(); tmp["num"] = tmp.cnv_id.map(fdnum); tmp = tmp.dropna(subset=["num"])
num2code = tmp.groupby("num").patient_id.agg(lambda s: s.mode().iloc[0]).to_dict(); num_mixed = tmp.groupby("num").patient_id.nunique(); num_mixed = sorted(num_mixed[num_mixed > 1].index)
rows = []
for sid, r in d.iterrows():
    o = {"sample_id": sid, "patient_id": r.patient_id, "subgroup": r.subgroup, "cnv_id": r.cnv_id, "release_acc": r.acc, "release_acc_stem": r.acc_stem, "release_date": r.date, "date_source": r.date_source, "acc_match": r.acc_match, "release_grade": r.grade, "grade_source": r.grade_source, "release_next": r.next, "next_source": r.next_source, "next_orig": r.next_orig, "next_scrape": r.next_scrape, "slide": r.slide, "slide_stem": r.slide_stem, "y_row": r.y_row}
    if r.cnv_id in fd.index:
        f = fd.loc[r.cnv_id]; o.update(sheet="discovery", sheet_patient=num2code.get(str(f.PatientID).replace(".0", ""), f"num:{f.PatientID}"), sheet_patient_num=str(f.PatientID).replace(".0", ""), sheet_acc_stem=stem(f["Path ID"]), sheet_year=pd.to_numeric(f["Endoscopy Year"], errors="coerce"), sheet_grade=G_SHEET.get(str(f.Pathology).strip().upper()), sheet_status=f.Status)
    elif r.cnv_id in va.index:
        f = va.loc[r.cnv_id]; o.update(sheet="validation", sheet_patient=f.PatientID, sheet_acc_stem=stem(f.Block), sheet_year=pd.to_numeric(f["Endoscopy Year"], errors="coerce"), sheet_grade=G_SHEET.get(str(f.Pathology2).strip().upper()), sheet_status=f.Status)
    else: o.update(sheet="none")
    if r.slide in smf.index:
        s = smf.loc[r.slide]; o.update(sm_patient=s.AlternateID, sm_acc_stem=stem(s.PathCaseID), sm_date=pd.to_datetime(s.EndoscopyDate, errors="coerce"), sm_grade=G_SHEET.get(str(s.Pathology).strip().upper()), p53=s.p53IHC)
    if r.acc_stem in ptc.index:
        p = ptc.loc[r.acc_stem]; o.update(db_participant=str(p.participant_id), db_date=p.d, db_grade=p.g)
        nxt = pt[(pt.participant_id == str(p.participant_id)) & (pt.d > r.date + pd.Timedelta(days=1))].sort_values("d")
        if len(nxt) and nxt.g.notna().any(): o.update(db_next_grade=nxt.g.dropna().iloc[0], db_next_date=nxt.d.iloc[0])
    rows.append(o)
A = pd.DataFrame(rows)
# flags
A["f_patient_sheet"] = A.apply(lambda r: r.get("sheet_patient") is not None and isinstance(r.get("sheet_patient"), str) and not r.sheet_patient.startswith("num:") and r.sheet_patient != r.patient_id, axis=1)
A["f_patient_slidematch"] = A.apply(lambda r: isinstance(r.get("sm_patient"), str) and r.sm_patient != r.patient_id, axis=1)
A["f_acc_sheet"] = A.apply(lambda r: isinstance(r.get("sheet_acc_stem"), str) and r.release_acc_stem is not None and r.sheet_acc_stem != r.release_acc_stem, axis=1)
A["f_acc_slidematch"] = A.apply(lambda r: isinstance(r.get("sm_acc_stem"), str) and r.release_acc_stem is not None and r.sm_acc_stem != r.release_acc_stem, axis=1)
A["f_acc_slidefile"] = A.apply(lambda r: r.slide_stem is not None and r.release_acc_stem is not None and r.slide_stem != r.release_acc_stem, axis=1)
A["f_cnv_from_other_endoscopy"] = A.apply(lambda r: isinstance(r.get("sheet_acc_stem"), str) and r.slide_stem is not None and r.sheet_acc_stem != r.slide_stem, axis=1)
A["f_year_sheet"] = A.apply(lambda r: pd.notna(r.get("sheet_year")) and pd.notna(r.release_date) and int(r.sheet_year) != r.release_date.year, axis=1)
A["f_date_slidematch"] = A.apply(lambda r: pd.notna(r.get("sm_date")) and pd.notna(r.release_date) and not (r.sm_date.month == 1 and r.sm_date.day == 1) and abs((r.sm_date - r.release_date).days) > 366, axis=1)
A["f_date_db"] = A.apply(lambda r: pd.notna(r.get("db_date")) and pd.notna(r.release_date) and abs((r.db_date - r.release_date).days) > 366, axis=1)
A["f_grade_sheet"] = A.apply(lambda r: pd.notna(r.get("sheet_grade")) and pd.notna(r.release_grade) and tier(r.sheet_grade) != tier(r.release_grade), axis=1)
A["f_grade_slidematch"] = A.apply(lambda r: pd.notna(r.get("sm_grade")) and pd.notna(r.release_grade) and tier(r.sm_grade) != tier(r.release_grade), axis=1)
A["f_grade_db"] = A.apply(lambda r: pd.notna(r.get("db_grade")) and pd.notna(r.release_grade) and tier(r.db_grade) != tier(r.release_grade), axis=1)
A["f_next_orig_vs_scrape"] = A.apply(lambda r: pd.notna(r.next_orig) and pd.notna(r.next_scrape) and tier(r.next_orig) != tier(r.next_scrape), axis=1)
A["f_next_vs_db"] = A.apply(lambda r: pd.notna(r.get("db_next_grade")) and pd.notna(r.release_next) and tier(r.db_next_grade) != tier(r.release_next), axis=1)
# confirmed error: two independent non-release sources agree with each other against the release (patient, accession or date)
def confirmed(r):
    out = []
    ps = [p for p in [r.get("sheet_patient") if isinstance(r.get("sheet_patient"), str) and not str(r.get("sheet_patient")).startswith("num:") else None, r.get("sm_patient") if isinstance(r.get("sm_patient"), str) else None] if p]
    if len(ps) == 2 and ps[0] == ps[1] and ps[0] != r.patient_id: out.append("patient")
    ac = [a for a in [r.get("sheet_acc_stem"), r.get("sm_acc_stem"), r.slide_stem] if isinstance(a, str)]
    if len(ac) >= 2 and len(set(ac)) == 1 and r.release_acc_stem is not None and ac[0] != r.release_acc_stem: out.append("accession")
    yrs = [y for y in [r.get("sheet_year"), r.get("sm_date").year if pd.notna(r.get("sm_date")) and not (r.get("sm_date").month == 1 and r.get("sm_date").day == 1) else None, r.get("db_date").year if pd.notna(r.get("db_date")) else None] if y is not None and pd.notna(y)]
    if len(yrs) >= 2 and len(set(int(y) for y in yrs)) == 1 and pd.notna(r.release_date) and int(yrs[0]) != r.release_date.year: out.append("date")
    return "+".join(out)
A["confirmed_error"] = A.apply(confirmed, axis=1)
A.to_csv(ROW + "/audit_rows.csv", index=False)
flags = [c for c in A.columns if c.startswith("f_")]; summ = {}
for g in ["also_in_ERIN", "comparator_25", "other"]:
    s = A[A.subgroup == g]; summ[g] = {"rows": int(len(s)), "patients": int(s.patient_id.nunique()), "sheet": s.sheet.value_counts().to_dict(), "rows_with_slidematch": int(s.sm_patient.notna().sum()) if "sm_patient" in s else 0, "rows_with_db_report": int(s.db_grade.notna().sum()) if "db_grade" in s else 0,
               **{f: {"rows": int(s[f].sum()), "patients": int(s.loc[s[f], "patient_id"].nunique())} for f in flags}, "confirmed_error_rows": int((s.confirmed_error != "").sum()), "confirmed_error_types": s.confirmed_error[s.confirmed_error != ""].value_counts().to_dict()}
# Killcoyne per-sample predictions
kp = pd.read_excel(K + "/41591_2020_1033_MOESM4_ESM.xlsx", sheet_name="Supporting data for Figure 2a", header=1).rename(columns={"Samplename": "cnv_id", "Probability": "k_prob", "Risk class": "k_class"})
kc = pd.read_excel(K + "/41591_2020_1033_MOESM7_ESM.xlsx", sheet_name=0).rename(columns={"Samplename": "cnv_id", "Status": "k_status", "Complexity Score": "k_cx"})
kk = kp.merge(kc, on="cnv_id", how="outer"); kk["cnv_id"] = kk.cnv_id.astype(str)
def oof(fam): return pd.concat([pd.read_csv(f, dtype={"sample_id": str}) for f in glob.glob(f"{F}/training_final_nested_cv_v1/{fam}/fold*/outer_test_predictions.csv")]).set_index("sample_id").y_prob.reindex(d.index)
d["cnv_oof"] = oof("cnv_only").values; d["img_oof"] = oof("image_only").values
m = d.reset_index().merge(kk, on="cnv_id", how="left"); mm = m.dropna(subset=["k_prob"])
g = d.groupby("patient_id"); y = g.y_row.max(); sub = pd.Series(np.where(y.index.isin(ov), "also_in_ERIN", "never_in_ERIN"), index=y.index)
gm = mm.groupby("patient_id").agg(k_max=("k_prob", "max"), k_status=("k_status", "first"), cnv_max=("cnv_oof", "max"), n=("cnv_id", "size")); gm["y"] = y.reindex(gm.index); gm["subgroup"] = sub.reindex(gm.index)
KJ = {"killcoyne_table_rows": int(len(kk)), "matched_rows": int(len(mm)), "matched_patients": int(mm.patient_id.nunique()), "matched_rows_by_subgroup": mm.assign(sg=mm.patient_id.map(sub)).sg.value_counts().to_dict(),
      "status_vs_our_label_patients": pd.crosstab(gm.k_status, gm.y).to_dict(), "spearman_kprob_vs_our_cnv_oof_rows": {"rho": round(float(spearmanr(mm.k_prob, mm.cnv_oof).correlation), 3), "n": int(len(mm))},
      "patient_auroc_killcoyne_maxprob_for_our_endpoint": {sg: {"n": int((gm.subgroup == sg).sum()) if sg != "all" else int(len(gm)), "events": int(gm.y[gm.subgroup == sg].sum()) if sg != "all" else int(gm.y.sum()), "auroc_killcoyne": round(auc(gm.y[gm.subgroup == sg] if sg != "all" else gm.y, gm.k_max[gm.subgroup == sg] if sg != "all" else gm.k_max), 3), "auroc_our_cnv_only": round(auc(gm.y[gm.subgroup == sg] if sg != "all" else gm.y, gm.cnv_max[gm.subgroup == sg] if sg != "all" else gm.cnv_max), 3)} for sg in ["all", "also_in_ERIN", "never_in_ERIN"]},
      "killcoyne_class_by_our_row_label": pd.crosstab(mm.k_class, mm.y_row).to_dict()}
# count reconciliation
disc = d.cnv_id.isin(fd.index); ink = d.cnv_id.isin(kk.cnv_id); pat_any_disc = d[disc].patient_id.nunique(); pat_any_k = d[ink].patient_id.nunique(); pat_all_disc = int((g.apply(lambda s: s.cnv_id.isin(fd.index).all())).sum()); pat_all_k = int((g.apply(lambda s: s.cnv_id.isin(kk.cnv_id).all())).sum())
modal = int((g.apply(lambda s: (s.cnv_id.isin(fd.index).mean() >= 0.5))).sum())
CR = {"rows_in_discovery_sheet": int(disc.sum()), "rows_in_killcoyne_pred_table": int(ink.sum()), "patients_any_row_discovery_sheet": int(pat_any_disc), "patients_any_row_in_killcoyne_table": int(pat_any_k), "patients_all_rows_discovery_sheet": pat_all_disc, "patients_all_rows_in_killcoyne_table": pat_all_k, "patients_modal_sheet_discovery": modal,
      "note": "'82/150' (closeout C) = modal sheet per patient; '69/150' (pipeline doc) matches none of these definitions exactly unless stated otherwise below"}
for k_, v_ in list(CR.items()):
    if v_ == 69: CR["note"] = f"69 matches definition {k_}"
# v2: label-source sensitivity: patient progressor status under the release rule with the next-biopsy label taken from (a) release, (b) DB scrape where present else release, (c) master where present else release
def lgd2(next_lab, streak): return ((next_lab >= 3) | ((next_lab == 2) & (streak >= 1))).astype(float).where(next_lab.notna())
streak = pd.to_numeric(coh.LGDStreakSoFar, errors="coerce").reindex(d.index)
alt = pd.DataFrame({"release": d.y_row.astype(float), "db_scrape_preferred": lgd2(d.next_scrape.fillna(d.next), streak), "master_preferred": lgd2(d.next_orig.fillna(d.next), streak)}, index=d.index)
altp = alt.groupby(d.patient_id).max(); altp["subgroup"] = np.where(altp.index.isin(ov), "also_in_ERIN", "never_in_ERIN")
SENS = {"patient_status_changes_vs_release": {src: {sg: {"patients": int(((altp[src] != altp.release) & (altp.subgroup == sg)).sum()), "to_progressor": int(((altp[src] == 1) & (altp.release == 0) & (altp.subgroup == sg)).sum()), "to_nonprogressor": int(((altp[src] == 0) & (altp.release == 1) & (altp.subgroup == sg)).sum())} for sg in ["also_in_ERIN", "never_in_ERIN"]} for src in ["db_scrape_preferred", "master_preferred"]},
        "rows_with_both_next_sources": int((d.next_orig.notna() & d.next_scrape.notna()).sum()), "rows_both_sources_disagree_two_tier": int(((d.next_orig >= 2) != (d.next_scrape >= 2))[d.next_orig.notna() & d.next_scrape.notna()].sum()),
        "grade_rows_release_LGD_vs_sheet_and_slidematch_benign": int(((A.release_grade >= 2) & (A.sheet_grade.fillna(-1) < 2) & (A.sm_grade.fillna(-1) < 2) & A.sheet_grade.notna() & A.sm_grade.notna()).sum()), "grade_rows_release_benign_vs_sheet_and_slidematch_LGDplus": int(((A.release_grade < 2) & (A.sheet_grade >= 2) & (A.sm_grade >= 2)).sum()),
        "note": "grade and next-label disagreements are label-source disagreements, not chain errors; the pre-specified confirmed-error rule covers patient, accession and date only"}
DEC = {"confirmed_error_rows_total": int((A.confirmed_error != "").sum()), "confirmed_error_patients": int(A.loc[A.confirmed_error != "", "patient_id"].nunique()), "decision": "corrected release required" if (A.confirmed_error != "").any() else "no confirmed errors: all items run on the frozen release only"}
res = {"_spec": "docs/paper_plan_answers.md @ db236a0 item 0", "n_overlap_patients": len(ov), "comparator_patients": comp, "summary_by_subgroup": summ, "killcoyne": KJ, "count_reconciliation": CR, "decision": DEC, "label_source_sensitivity": SENS, "sheet_numeric_ids_mapping_to_more_than_one_release_patient": num_mixed, "version": "v2 (5bf5419+): slide-filename parser fixed, sheet patient id mapped by majority through cnv_id; v1 = commit 187dda8 output audit_item0.json",
       "grade_source_by_subgroup": pd.crosstab(A.subgroup, A.grade_source).to_dict(), "next_source_by_subgroup": pd.crosstab(A.subgroup, A.next_source).to_dict()}
json.dump(res, open(AGG + "/audit_item0_v2.json", "w"), indent=1, default=str); print(json.dumps(res, indent=1, default=str)[:6000])

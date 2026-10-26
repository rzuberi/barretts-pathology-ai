"""ERL step 1 + 3: ERIN patient-endoscopy table for an external longitudinal H&E test of the Paper 2 no-sequencing tier.
Row-level outputs stay in killcoyne_mm/two_papers_ERL/ (cluster); only counts (cells < 5 -> "<5") go to results/paper_final/two_papers_ERL/.
Endoscopy = (anon_id, collection date); all reports of that day pooled. Event = first endoscopy with an HGD/CANCER report label (jury final_label);
primary event definition additionally needs an oesophageal/GOJ v3 section graded HGD/CANCER when v3 exists for that day (so gastric/other-organ
cancers are not counted). Therapy = report text mentions RFA/EMR/ESD/ablation/APC/cryo/neosquamous (regex of num_t4_treatment_split.py), or a v3
oesophageal section typed EMR/RESECTION; endoscopies from the first therapy mention onwards are never pre-event. Pre-event Barrett's endoscopy =
before min(event, therapy), with an oesophageal/GOJ section (v3) or, without v3, a report label NDBE/IND/LGD. Scanned = >= 1 H&E slide whose
organ_final is OESOPHAGUS/GOJ and whose specimen type is not EMR/RESECTION. Killcoyne-overlap patients excluded (closeout pairs file U the
70-patient union of explore_E5). No outcome is scored against any model."""
import json, os, re, numpy as np, pandas as pd
T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"; W = T + "/feasibility/paper_plan/killcoyne_mm/two_papers_ERL"; R = T + "/results/paper_final/two_papers_ERL"
os.makedirs(W, exist_ok=True); os.makedirs(R, exist_ok=True)
ERIN = "/mnt/scratche/fast/fmlab/datasets/imaging/ERIN/data/PathologyReport_AnonIds.csv"
ORD = {"NDBE": 0, "IND": 1, "LGD": 2, "HGD": 3, "CANCER": 4}; OES = {"OESOPHAGUS", "GOJ"}
def sup(n): n = int(n); return n if (n == 0 or n >= 5) else "<5"
info = {}
# ---------------- sources (read-only)
rep = pd.read_csv(ERIN, dtype=str).fillna(""); rep["d"] = pd.to_datetime(rep.CollectedOrOrdered, dayfirst=True, errors="coerce")
lab = pd.read_csv(T + "/labeller/erin_labels_jury_final.csv", dtype=str).drop_duplicates("CaseName")
v3 = pd.read_csv(T + "/labeller/erin_sections_v3_consensus.csv", dtype=str)
sl = pd.read_csv(T + "/labeller/erin_slide_organ_v3.csv", dtype=str)
allsl = pd.read_csv(T + "/campaigns/allslides/erin_slides_all.csv", dtype=str, low_memory=False)
isbi = pd.read_csv("/mnt/scratche/slow/fmlab/zuberi01/analysis/isbi_rerun/slides.csv", dtype=str, usecols=["file_uuid", "CaseName"])
ex = set(pd.read_csv(T + "/feasibility/closeout/erin_swg_pairs.csv", dtype=str).erin_anon_id.dropna())
ex5 = set(l.strip() for l in open(T + "/feasibility/paper_plan/killcoyne_mm/explore_E5/erin_exclude_anon_ids.txt") if l.strip())
EXCL = ex | ex5
info["sources"] = {"reports": len(rep), "report_cases": int(rep.CaseName.nunique()), "report_patients": int(rep.anon_id.nunique()), "labelled_cases": len(lab),
                   "v3_sections": len(v3), "v3_cases": int(v3.CaseName.nunique()), "slides": len(sl), "isbi_oes_slides": len(isbi),
                   "exclude_pairs_file": len(ex), "exclude_e5_union": len(ex5), "exclude_union": len(EXCL),
                   "report_date_min_year": int(rep.d.min().year), "report_date_max": str(rep.d.max().date())[:7], "report_date_missing": int(rep.d.isna().sum())}
info["specimen_protocol_top"] = {k: int(v) for k, v in rep.SpecimenProtocol.str.upper().value_counts().head(25).items()}
# ---------------- per-case flags
TXT = rep.ClinicalInformation_redacted + " " + rep.GrossDescription_redacted + " " + rep.MicroscopicDescription_redacted + " " + rep.FinalDiagnosis_redacted
rx_ther = re.compile(r"\bRFA\b|radiofrequency|ablat|\bEMR\b|\bESD\b|mucosal resection|neosquam|argon|\bAPC\b|cryo", re.I)
rx_be = re.compile(r"barrett|intestinal metaplasia|columnar[- ]lined|\bCLO\b|goblet", re.I)
rx_prior = re.compile(r"(previous|prior|known|hx|history of|h/o)\W+(\w+\W+){0,3}(high[- ]grade|HGD|intramucosal|IMC|adenocarcinoma|carcinoma|cancer|oesophagectomy|esophagectomy)", re.I)
rep["ther_txt"] = TXT.str.contains(rx_ther); rep["be_txt"] = (rep.MicroscopicDescription_redacted + " " + rep.FinalDiagnosis_redacted + " " + rep.ClinicalInformation_redacted).str.contains(rx_be)
rep["prior_ev_txt"] = rep.ClinicalInformation_redacted.str.contains(rx_prior)
rep["resect_prot"] = rep.SpecimenProtocol.str.upper().str.contains("RESECT")
case = rep.drop_duplicates("CaseName").set_index("CaseName")[["anon_id", "d", "ther_txt", "be_txt", "prior_ev_txt", "resect_prot", "AgeAtInvestigation"]].copy()
lab = lab.set_index("CaseName"); case["label"] = lab.final_label.reindex(case.index); case["status"] = lab.label_status.reindex(case.index)
case["g"] = case.label.map(ORD); case["unsure"] = case.label.eq("unsure")
v3["oes"] = v3.site.isin(OES); v3["hgdp"] = v3.worst_grade.isin(["HGD", "CANCER"]); v3["emr"] = v3.specimen_type.isin(["EMR", "RESECTION"])
g3 = v3.groupby("CaseName"); case["v3"] = case.index.isin(g3.groups.keys())
case["v3_oes_n"] = g3.oes.sum().reindex(case.index).fillna(0).astype(int)
case["v3_oes_hgdp"] = v3[v3.oes & v3.hgdp].groupby("CaseName").size().reindex(case.index).fillna(0).astype(int) > 0
case["v3_oes_emr"] = v3[v3.oes & v3.emr].groupby("CaseName").size().reindex(case.index).fillna(0).astype(int) > 0
# scanned oesophageal H&E slides per case
sl = sl.merge(allsl[["file_uuid", "path", "h5_new", "size_mb"]], on="file_uuid", how="left")
sl["he"] = sl.is_he.astype(str).str.lower().eq("true"); sl["oes_slide"] = sl.he & sl.organ_final.isin(OES) & ~sl.specimen_type.isin(["EMR", "RESECTION"])
sl.loc[sl.file_uuid.isin(isbi.file_uuid), "in_isbi"] = True; sl["in_isbi"] = sl.in_isbi.fillna(False).astype(bool)
info["slides_he"] = int(sl.he.sum()); info["slides_oes_he_nonemr"] = int(sl.oes_slide.sum()); info["slides_isbi_and_oes"] = int((sl.oes_slide & sl.in_isbi).sum())
case["n_oes_slides"] = sl[sl.oes_slide].groupby("CaseName").size().reindex(case.index).fillna(0).astype(int)
case = case[case.d.notna()]
# ---------------- endoscopy table (patient x date)
case = case.reset_index().rename(columns={"index": "CaseName"})
def agg(g):
    lbl = g.g.dropna()
    return pd.Series({"cases": "|".join(g.CaseName), "n_cases": len(g), "g": lbl.max() if len(lbl) else np.nan, "unsure": bool(g.unsure.any()),
                      "labelled": bool(g.label.notna().any()), "v3": bool(g.v3.any()), "v3_oes_n": int(g.v3_oes_n.sum()), "v3_oes_hgdp": bool(g.v3_oes_hgdp.any()),
                      "ther": bool(g.ther_txt.any() | g.v3_oes_emr.any() | g.resect_prot.any()), "be": bool(g.be_txt.any()), "prior_ev": bool(g.prior_ev_txt.any()),
                      "n_oes_slides": int(g.n_oes_slides.sum()), "age": pd.to_numeric(g.AgeAtInvestigation, errors="coerce").max()})
E = case.groupby(["anon_id", "d"]).apply(agg).reset_index().sort_values(["anon_id", "d"])
E["ev_report"] = E.g >= 3
E["ev_primary"] = E.ev_report & (E.v3_oes_hgdp | ~E.v3)
E["ev_v3only"] = E.v3_oes_hgdp            # sensitivity: oesophageal HGD+ section even when the report label says otherwise
E["barrett_endo"] = (E.v3 & (E.v3_oes_n > 0)) | (~E.v3 & E.g.isin([0, 1, 2]))
E["scanned"] = E.n_oes_slides > 0
last_rep = rep.groupby("anon_id").d.max()
info["endoscopies_all"] = len(E); info["patients_all"] = int(E.anon_id.nunique())

def build(evcol, unsure_is_event=False, nd_only=False, use_prior=False):
    """Per patient: pre-event Barrett's endoscopies (all, scanned), event date, follow-up."""
    rows = []
    for pid, g in E.groupby("anon_id", sort=False):
        ev = g[evcol] | (g.unsure if unsure_is_event else False)
        t_ev = g.d[ev].min() if ev.any() else pd.NaT
        stop = [t_ev] + [g.d[g.ther].min()] if g.ther.any() else [t_ev]
        if use_prior and g.prior_ev.any(): stop.append(g.d[g.prior_ev].min())
        stop = min([s for s in stop if pd.notna(s)], default=pd.NaT)
        pre = g[g.barrett_endo & ~ev & ((g.d < stop) if pd.notna(stop) else True)]
        if nd_only: pre = pre[pre.g.isna() | (pre.g <= 1)]
        be = bool(g.be[g.d <= (pre.d.max() if len(pre) else g.d.min())].any()) if len(pre) else False
        sc = pre[pre.scanned]
        rows.append({"anon_id": pid, "prog": bool(pd.notna(t_ev)), "t_ev": t_ev, "excluded_overlap": pid in EXCL, "barretts": be,
                     "therapy_before_event": bool(g.ther.any() and (pd.isna(t_ev) or g.d[g.ther].min() < t_ev)),
                     "n_pre": len(pre), "n_pre_scanned": len(sc), "first_pre": pre.d.min(), "last_pre": pre.d.max(),
                     "third_pre": pre.d.iloc[2] if len(pre) >= 3 else pd.NaT, "second_pre": pre.d.iloc[1] if len(pre) >= 2 else pd.NaT,
                     "first_sc": sc.d.min(), "last_sc": sc.d.max(), "last_report": last_rep.get(pid), "later_lgd": bool((g.g == 2).any()),
                     "pre_dates": "|".join(pre.d.dt.strftime("%Y-%m-%d")), "pre_scanned_dates": "|".join(sc.d.dt.strftime("%Y-%m-%d")),
                     "pre_slides": int(pre.n_oes_slides.sum()), "pre_oes_sections": int(pre.v3_oes_n.sum()), "age_first_pre": pre.age.iloc[0] if len(pre) else np.nan})
    P = pd.DataFrame(rows)
    yrs = lambda a, b: (a - b).dt.days / 365.25
    P["fu_after_last_sc_y"] = np.where(P.prog, yrs(P.t_ev, P.last_sc), yrs(P.last_report, P.last_sc))
    P["fu_after_first_pre_y"] = np.where(P.prog, yrs(P.t_ev, P.first_pre), yrs(P.last_report, P.first_pre))
    P["fu_after_third_pre_y"] = np.where(P.prog, yrs(P.t_ev, P.third_pre), yrs(P.last_report, P.third_pre))
    P["fu_after_second_pre_y"] = np.where(P.prog, yrs(P.t_ev, P.second_pre), yrs(P.last_report, P.second_pre))
    return P

def counts(P, scanned=True):
    el = P[P.barretts & ~P.excluded_overlap]
    k = "n_pre_scanned" if scanned else "n_pre"; fu = "fu_after_last_sc_y" if scanned else "fu_after_third_pre_y"
    out = {"patients_barretts_nonoverlap": sup(len(el)), "overlap_excluded_with_any_pre": sup((P.excluded_overlap & (P[k] > 0)).sum())}
    for m in (1, 2, 3):
        s = el[el[k] >= m]; pr, np_ = s[s.prog], s[~s.prog]
        d = {"progressors": sup(len(pr)), "non_progressors": sup(len(np_))}
        for y in (1, 2, 3, 5): d[f"np_fu_ge_{y}y"] = sup((np_[fu] >= y).sum())
        if len(pr): d["prog_years_last_to_event_median"] = round(float(pr[fu].median()), 2) if len(pr) >= 5 else "<5 patients"
        d["endoscopies"] = sup(s[k].sum()); d["prog_endoscopies"] = sup(pr[k].sum())
        out[f"ge{m}"] = d
    return out

P = build("ev_primary")
info["step1_scanned_primary"] = counts(P)
info["step1_scanned_unsure_as_event"] = counts(build("ev_primary", unsure_is_event=True))
info["step1_scanned_report_event"] = counts(build("ev_report"))
info["step1_scanned_v3_event"] = counts(build("ev_v3only"))
info["step1_scanned_nd_only"] = counts(build("ev_primary", nd_only=True))
info["step1_scanned_prior_event_text_excluded"] = counts(build("ev_primary", use_prior=True))
# scanned years of the scanned pre-event endoscopies (all eligible)
el = P[P.barretts & ~P.excluded_overlap]
info["scanned_pre_event_endoscopy_years"] = {str(k): sup(v) for k, v in pd.Series([x[:4] for s in el.pre_scanned_dates for x in s.split("|") if x]).value_counts().sort_index().items()}
info["first_scanned_year_of_progressors_ge2"] = {str(k): sup(v) for k, v in el[el.prog & (el.n_pre_scanned >= 2)].first_sc.dt.year.value_counts().sort_index().items()}
info["event_year_of_progressors_ge2"] = {str(k): sup(v) for k, v in el[el.prog & (el.n_pre_scanned >= 2)].t_ev.dt.year.value_counts().sort_index().items()}
# ---------------- step 3: whole report history (scanned or not)
info["step3_report_history_primary"] = counts(P, scanned=False)
info["step3_report_history_nd_only"] = counts(build("ev_primary", nd_only=True), scanned=False)
P.to_csv(W + "/patients_primary.csv", index=False); E.to_csv(W + "/endoscopies.csv", index=False)
json.dump(info, open(R + "/cohort_counts.json", "w"), indent=1, default=str); print(json.dumps(info, indent=1, default=str))

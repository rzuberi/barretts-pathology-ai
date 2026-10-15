"""Jury v3 consensus + what it tells us (ERIN all-slides campaign).

Inputs: feasibility/runs/sec3full_<juror>_s<k>/output/sections_v3_*.csv (10 jurors).
Outputs (labeller/, cluster only):
  erin_sections_v3_consensus.csv  one row per (CaseName, section): site, specimen_type,
                                  grades (>=half of jurors who saw the section), subtypes,
                                  vote counts, n_jurors
  erin_slide_organ_v3.csv         one row per WSI file: organ (metadata / report / both), section grades
  results/sections_v3_summary.json  the numbers
"""
import glob, json, os, re
from collections import Counter, defaultdict
import pandas as pd

T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"
L = T + "/labeller"; C = T + "/campaigns/allslides"
ORD = {"NORMAL_OTHER": -1, "NDBE": 0, "IND": 1, "LGD": 2, "HGD": 3, "CANCER": 4}

# ---- gather votes ----
votes = defaultdict(lambda: defaultdict(lambda: {"grades": Counter(), "subs": Counter(), "site": Counter(),
                                                 "spec": Counter(), "jurors": set()}))
parse = Counter(); rows_per = Counter(); nsec_by_juror = defaultdict(list)
for f in glob.glob(T + "/feasibility/runs/sec3full_*/output/sections_v3_*.csv"):
    juror = os.path.basename(f).replace("sections_v3_", "").rsplit("_shard", 1)[0]
    d = pd.read_csv(f, dtype=str).fillna("").drop_duplicates("CaseName")
    for _, r in d.iterrows():
        rows_per[juror] += 1
        if r.sections_json.startswith("PARSE_FAIL"): parse[juror] += 1; continue
        secs = json.loads(r.sections_json); nsec_by_juror[juror].append(len(secs))
        seen = {}
        for s in secs:
            key = s["section"].upper().strip(); key = "WHOLE" if key in ("WHOLE", "WHOLE_REPORT", "") else key[:2]
            e = seen.setdefault(key, {"grades": set(), "subs": set(), "site": None, "spec": None})
            e["grades"] |= set(s["grades"]); e["subs"] |= set(s.get("cancer_subtypes", []))
            e["site"] = e["site"] or s.get("site"); e["spec"] = e["spec"] or s.get("specimen_type")
        for key, e in seen.items():
            v = votes[r.CaseName][key]; v["jurors"].add(juror)
            v["grades"].update(e["grades"]); v["subs"].update(e["subs"])
            if e["site"]: v["site"][e["site"]] += 1
            if e["spec"]: v["spec"][e["spec"]] += 1
jurors = sorted(rows_per)
N = len(jurors)

# ---- consensus ----
rows = []
for case, secs in votes.items():
    for key, v in secs.items():
        nj = len(v["jurors"])
        if nj < 3: continue
        thr = (nj + 1) // 2                                  # >= half of the jurors that saw the section
        grades = sorted([g for g, n in v["grades"].items() if n >= thr], key=lambda g: ORD.get(g, 9))
        if not grades: continue
        subs = sorted([s for s, n in v["subs"].items() if n >= thr])
        site, site_n = (v["site"].most_common(1)[0] if v["site"] else ("UNSTATED", 0))
        spec, spec_n = (v["spec"].most_common(1)[0] if v["spec"] else ("UNSTATED", 0))
        rows.append({"CaseName": case, "section": key, "n_jurors": nj,
                     "site": site, "site_frac": round(site_n / nj, 3), "site_votes": json.dumps(dict(v["site"])),
                     "specimen_type": spec, "spec_frac": round(spec_n / nj, 3),
                     "grades": "|".join(grades), "worst_grade": max(grades, key=lambda g: ORD[g]),
                     "grade_votes": json.dumps(dict(v["grades"])),
                     "cancer_subtypes": "|".join(subs), "subtype_votes": json.dumps(dict(v["subs"])),
                     "unanimous_grades": all(v["grades"][g] == nj for g in grades) and
                                         all(n < thr for g, n in v["grades"].items() if g not in grades)})
sec = pd.DataFrame(rows)
sec.to_csv(L + "/erin_sections_v3_consensus.csv", index=False); os.chmod(L + "/erin_sections_v3_consensus.csv", 0o600)

# ---- per-juror agreement with consensus (grade set, site) ----
juror_stats = {}
for j in jurors:
    g_agree = g_n = s_agree = s_n = 0
    for _, r in sec.iterrows():
        pass
juror_grade_agree = defaultdict(lambda: [0, 0]); juror_site_agree = defaultdict(lambda: [0, 0])
cons_idx = {(r.CaseName, r.section): r for r in sec.itertuples()}
for f in glob.glob(T + "/feasibility/runs/sec3full_*/output/sections_v3_*.csv"):
    juror = os.path.basename(f).replace("sections_v3_", "").rsplit("_shard", 1)[0]
    d = pd.read_csv(f, dtype=str).fillna("").drop_duplicates("CaseName")
    for _, r in d.iterrows():
        if r.sections_json.startswith("PARSE_FAIL"): continue
        for s in json.loads(r.sections_json):
            key = s["section"].upper().strip(); key = "WHOLE" if key in ("WHOLE", "WHOLE_REPORT", "") else key[:2]
            c = cons_idx.get((r.CaseName, key))
            if c is None: continue
            juror_grade_agree[juror][1] += 1
            if max(s["grades"], key=lambda g: ORD.get(g, -9)) == c.worst_grade: juror_grade_agree[juror][0] += 1
            juror_site_agree[juror][1] += 1
            if s.get("site") == c.site: juror_site_agree[juror][0] += 1

# ---- report-level: section-max vs whole-report jury (v1) ----
case_max = sec.groupby("CaseName")["worst_grade"].agg(lambda g: max(g, key=lambda x: ORD[x]))
whole = pd.read_csv(L + "/erin_labels_jury_final.csv", dtype=str).fillna("").drop_duplicates("CaseName").set_index("CaseName")
cmp = pd.DataFrame({"section_max": case_max}).join(whole[["final_label", "label_status"]], how="inner")
cmp_ok = cmp[cmp.label_status != "unsure_held_out"]
xt = pd.crosstab(cmp_ok.section_max, cmp_ok.final_label)
smax_ndbe = cmp_ok.section_max.replace({"NORMAL_OTHER": "NDBE"})
# oesophageal-only section max (site in OESOPHAGUS/GOJ)
oes = sec[sec.site.isin(["OESOPHAGUS", "GOJ"])]
oes_max = oes.groupby("CaseName")["worst_grade"].agg(lambda g: max(g, key=lambda x: ORD[x]))
cmp2 = pd.DataFrame({"oes_max": oes_max}).join(whole[["final_label", "label_status"]], how="inner")
cmp2 = cmp2[cmp2.label_status != "unsure_held_out"]

# ---- slides: organ per WSI ----
sl = pd.read_csv(C + "/erin_slides_all.csv", dtype=str).fillna("")
sl["section"] = sl.section.str.upper().str.strip()
m = sl.merge(sec[["CaseName", "section", "site", "site_frac", "specimen_type", "grades", "worst_grade", "cancer_subtypes", "n_jurors"]],
             on=["CaseName", "section"], how="left")
# fall back to WHOLE-section row when the report has no lettered sections
whole_rows = sec[sec.section == "WHOLE"].set_index("CaseName")
miss = m.site.isna() & m.CaseName.isin(whole_rows.index)
for col in ["site", "site_frac", "specimen_type", "grades", "worst_grade", "cancer_subtypes", "n_jurors"]:
    m.loc[miss, col] = m.loc[miss, "CaseName"].map(whole_rows[col])
META2SITE = {"OESOPHAGUS": "OESOPHAGUS", "STOMACH": "STOMACH", "DUODENUM": "DUODENUM", "COLORECTAL": "COLORECTAL",
             "LYMPH NODE": "LYMPH_NODE", "POLYP": None}
m["organ_report"] = m.site.fillna("")
m["organ_meta_norm"] = m.organ_meta.map(lambda x: META2SITE.get(x, "OTHER" if x else ""))
def final_organ(r):
    rep = r.organ_report if r.organ_report not in ("", "UNSTATED") else ""
    rep_o = "OESOPHAGUS" if rep == "GOJ" else rep          # GOJ counts as oesophageal for the filter
    return rep_o or r.organ_meta_norm or "UNKNOWN"
m["organ_final"] = m.apply(final_organ, axis=1)
m["organ_source"] = m.apply(lambda r: "report" if r.organ_report not in ("", "UNSTATED") else ("metadata" if r.organ_meta_norm else "none"), axis=1)
both = m[(m.organ_meta_norm != "") & (m.organ_meta_norm != "OTHER") & (m.organ_report != "") & (m.organ_report != "UNSTATED")]
both_rep = both.organ_report.replace({"GOJ": "OESOPHAGUS"})
m[["file_uuid", "CaseName", "section", "stain_type", "is_he", "organ_meta", "site", "site_frac", "organ_final", "organ_source",
   "specimen_type", "grades", "worst_grade", "cancer_subtypes", "n_jurors", "has_features"]].to_csv(L + "/erin_slide_organ_v3.csv", index=False)
os.chmod(L + "/erin_slide_organ_v3.csv", 0o600)
he = m[m.is_he == "True"]

summary = {
 "jurors": jurors, "reports_per_juror": dict(rows_per), "parse_fail_per_juror": dict(parse),
 "sections_per_report_by_juror": {j: round(sum(v) / len(v), 2) for j, v in nsec_by_juror.items()},
 "consensus": {"reports": int(sec.CaseName.nunique()), "sections": len(sec),
               "sections_per_report": round(len(sec) / sec.CaseName.nunique(), 2),
               "n_jurors_dist": sec.n_jurors.value_counts().sort_index().to_dict(),
               "multi_grade_sections": int((sec.grades.str.contains(r"\|")).sum()),
               "unanimous_grade_sections": int(sec.unanimous_grades.sum()),
               "grade_mentions": dict(Counter(g for gs in sec.grades for g in gs.split("|")).most_common()),
               "grade_set_top": sec.grades.value_counts().head(12).to_dict(),
               "worst_grade_dist": sec.worst_grade.value_counts().to_dict(),
               "site_dist": sec.site.value_counts().to_dict(),
               "site_plurality_frac_mean": round(float(sec.site_frac.mean()), 3),
               "site_unanimous": int((sec.site_frac == 1.0).sum()),
               "specimen_dist": sec.specimen_type.value_counts().to_dict(),
               "subtypes": dict(Counter(s for ss in sec.cancer_subtypes for s in ss.split("|") if s).most_common()),
               "grade_by_site": pd.crosstab(sec.site, sec.worst_grade).to_dict()},
 "juror_agreement_with_consensus": {j: {"worst_grade": round(a[0] / a[1], 4), "site": round(juror_site_agree[j][0] / juror_site_agree[j][1], 4), "n": a[1]}
                                    for j, a in juror_grade_agree.items()},
 "report_level": {"n_compared": int(len(cmp_ok)),
                  "section_max_vs_whole_report_exact": round(float((cmp_ok.section_max == cmp_ok.final_label).mean()), 4),
                  "section_max(NORMAL->NDBE)_vs_whole_report": round(float((smax_ndbe == cmp_ok.final_label).mean()), 4),
                  "crosstab_sectionmax_rows_vs_whole_cols": xt.to_dict(),
                  "oesophageal_only_max_vs_whole_report": round(float((cmp2.oes_max.replace({"NORMAL_OTHER": "NDBE"}) == cmp2.final_label).mean()), 4),
                  "reports_with_no_oesophageal_section": int(len(set(sec.CaseName) - set(oes.CaseName))),
                  "reports_by_section_max": case_max.value_counts().to_dict(),
                  "reports_by_oesophageal_max": oes_max.value_counts().to_dict()},
 "slides": {"all": len(m), "he": len(he),
            "organ_source_he": he.organ_source.value_counts().to_dict(),
            "organ_final_he": he.organ_final.value_counts().to_dict(),
            "organ_final_all": m.organ_final.value_counts().to_dict(),
            "he_oesophageal_incl_goj": int(he.organ_final.eq("OESOPHAGUS").sum()),
            "he_with_section_grade": int(he.worst_grade.notna().sum()),
            "he_worst_grade_dist": he.worst_grade.value_counts().to_dict(),
            "he_oesophageal_worst_grade_dist": he[he.organ_final == "OESOPHAGUS"].worst_grade.value_counts().to_dict(),
            "he_features_done_by_organ": he[he.has_features == "True"].organ_final.value_counts().to_dict(),
            "metadata_vs_report_organ": {"n_both": int(len(both)),
                                         "agreement": round(float((both.organ_meta_norm == both_rep).mean()), 4),
                                         "confusion_meta_rows_vs_report_cols": pd.crosstab(both.organ_meta_norm, both_rep).to_dict()}},
}
os.makedirs(T + "/results", exist_ok=True)
json.dump(summary, open(T + "/results/sections_v3_summary.json", "w"), indent=1, default=int)
print(json.dumps(summary, indent=1, default=int))

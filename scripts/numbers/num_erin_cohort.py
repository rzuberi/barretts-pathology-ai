"""NUMBERS A/D/E for ERIN: reports, patients, years, slides (all-slides table), stains/organs, features
extracted vs remaining, total patches from h5 attrs, jury label distributions (report/patient/section),
per-section structure, full-corpus inter-model agreement (pairwise + Fleiss), follow-up structure."""
import glob, json, os, re
from collections import Counter, defaultdict
from itertools import combinations
import numpy as np, pandas as pd, h5py
T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"; OUT = os.environ.get("OUTDIR", ".")
REP = "/mnt/scratche/fast/fmlab/datasets/imaging/ERIN/data/PathologyReport_AnonIds.csv"
FEAT = "/mnt/scratche/fast/fmlab/datasets/imaging/ERIN/features/20x_224px/features_uni_v2_all"
def mr(s): s = pd.Series(s).dropna(); return {"median": float(s.median()), "min": float(s.min()), "max": float(s.max()), "n": int(len(s))} if len(s) else None
res = {}
rep = pd.read_csv(REP, dtype=str, usecols=["anon_id", "CaseName", "CollectedOrOrdered", "SpecimenProtocol", "CaseStatus", "AgeAtInvestigation"])
rep["d"] = pd.to_datetime(rep.CollectedOrOrdered, errors="coerce", dayfirst=True)
per = rep.groupby("anon_id").size()
res["reports"] = {"n": len(rep), "patients": int(rep.anon_id.nunique()), "years": {"min": int(rep.d.min().year), "max": int(rep.d.max().year), "by_year": rep.d.dt.year.value_counts().sort_index().to_dict()},
                  "reports_per_patient": mr(per), "patients_with_ge2_reports": int((per >= 2).sum()),
                  "followup_span_years_patients_ge2": mr((rep.groupby("anon_id").d.max() - rep.groupby("anon_id").d.min()).dt.days[per >= 2] / 365.25),
                  "SpecimenProtocol_top": rep.SpecimenProtocol.value_counts().head(12).to_dict(), "CaseStatus": rep.CaseStatus.value_counts().to_dict(),
                  "age_at_investigation": mr(pd.to_numeric(rep.AgeAtInvestigation, errors="coerce"))}
sl = pd.read_csv(T + "/campaigns/allslides/erin_slides_all.csv", dtype=str, low_memory=False)
he = sl[sl.is_he.astype(str).str.lower() == "true"]
res["slides"] = {"all_files": len(sl), "cases": int(sl.CaseName.nunique()), "patients": int(sl.anon_id.nunique()), "he_slides": len(he), "non_he": int(len(sl) - len(he)),
                 "stain_type_counts": sl.stain_type.fillna("(blank)").value_counts().head(15).to_dict(), "organ_meta_counts": sl.organ_meta.fillna("(blank)").value_counts().head(15).to_dict(),
                 "organ_meta_known_frac": round(float(sl.organ_meta.notna().mean()), 3), "tissue_meta_top": sl.tissue_meta.fillna("(blank)").value_counts().head(10).to_dict(),
                 "he_slides_per_case": mr(he.groupby("CaseName").size()), "he_slides_per_patient": mr(he.groupby("anon_id").size()),
                 "size_mb_per_slide": mr(pd.to_numeric(sl.size_mb, errors="coerce")), "total_tb": round(float(pd.to_numeric(sl.size_mb, errors="coerce").sum() / 1e6), 2)}
h5s = glob.glob(FEAT + "/*.h5"); kept, grid, tissue, envs, bad = [], [], [], Counter(), 0
for f in h5s:
    try:
        with h5py.File(f) as h:
            a = dict(h.attrs); kept.append(int(a.get("kept_tiles", h["features"].shape[0]))); grid.append(int(a.get("grid_tiles", -1))); tissue.append(int(a.get("tissue_tiles", -1))); envs[str(a.get("env", "?"))[:40]] += 1
    except Exception: bad += 1
failed = FEAT + "/_failed.log"
res["features"] = {"h5_files": len(h5s), "unreadable": bad, "failed_log_lines": sum(1 for _ in open(failed)) if os.path.exists(failed) else 0,
                   "manifest_he_to_extract": sum(1 for _ in open(T + "/campaigns/allslides/erin_manifest_all_he.txt")) if os.path.exists(T + "/campaigns/allslides/erin_manifest_all_he.txt") else None,
                   "he_slides_with_h5": int(he.file_uuid.isin({os.path.basename(f)[:-3] for f in h5s}).sum()), "he_slides_without_h5": int((~he.file_uuid.isin({os.path.basename(f)[:-3] for f in h5s})).sum()),
                   "patches_total_kept": int(np.sum(kept)), "patches_per_slide_kept": mr(kept), "grid_tiles_per_slide": mr([g for g in grid if g > 0]), "env_counts": dict(envs),
                   "tile": "224 px at 0.5 mpp (20x), UNI2-h 1536-d"}
lab = pd.read_csv(T + "/labeller/erin_labels_jury_final.csv", dtype=str)
lab["d"] = pd.to_datetime(lab.CollectedOrOrdered, errors="coerce", dayfirst=True)
ORD = {"NDBE": 0, "IND": 1, "LGD": 2, "HGD": 3, "CANCER": 4}
res["jury_report_level"] = {"status": lab.label_status.value_counts().to_dict(), "final_label_all": lab.final_label.value_counts().to_dict(),
                            "final_label_train_eligible": lab[lab.label_status == "train_eligible"].final_label.value_counts().to_dict(),
                            "patient_max_grade": lab.assign(g=lab.final_label.map(ORD)).groupby("anon_id").g.max().map({v: k for k, v in ORD.items()}).value_counts().to_dict(),
                            "patient_first_report_grade": lab.sort_values("d").groupby("anon_id").final_label.first().value_counts().to_dict(),
                            "reports_with_he_slide_on_disk": int(lab.CaseName.isin(set(he.CaseName)).sum()), "reports_with_any_slide_file": int(lab.CaseName.isin(set(sl.CaseName)).sum())}
m = pd.read_csv(T + "/labeller/erin_master.csv", dtype=str)
res["imaged_cases_original_one_per_case"] = {"rows": len(m), "cases": int(m.CaseName.nunique()), "patients": int(m.anon_id.nunique()), "label_status": m.label_status.value_counts().to_dict(), "final_label": m.final_label.value_counts().to_dict()}
if os.path.exists(T + "/labeller/erin_progression_cohort_v3.csv"):
    pc = pd.read_csv(T + "/labeller/erin_progression_cohort_v3.csv")
    res["progression_cohort_v3"] = {"patients": len(pc), "progressors": int(pc.progressed_to_HGDplus.sum()), "index_grade": pc.index_grade.value_counts().to_dict(),
                                    "tte_days_progressors": mr(pc[pc.progressed_to_HGDplus].tte_days), "followup_days_nonprogressors": mr(pc[~pc.progressed_to_HGDplus].tte_days), "n_future_reports": mr(pc.n_future)}
# per-section structure + consensus grade distribution at SECTION level (same rule as build_slide_labels)
GRADE_ORD = {"NORMAL_OTHER": -1, "NDBE": 0, "IND": 1, "LGD": 2, "HGD": 3, "CANCER": 4}
juror_secs = defaultdict(dict)
for f in glob.glob(T + "/feasibility/runs/sections_*/output/sections_*.csv"):
    juror = f.split("/output/sections_")[1].rsplit("_shard", 1)[0]; d = pd.read_csv(f, dtype=str).fillna("")
    for _, r in d.iterrows():
        if r.sections_json == "PARSE_FAIL": continue
        try: secs = json.loads(r.sections_json)
        except Exception: continue
        mm = {}
        for s in secs:
            key = str(s.get("section", "?")).upper().strip(); key = "WHOLE" if key in ("WHOLE", "WHOLE_REPORT", "") else key[:2].strip()
            mm.setdefault(key, (set(), set())); mm[key][0].update(s.get("grades", [])); mm[key][1].update(s.get("cancer_subtypes", []))
        juror_secs[r.CaseName][juror] = mm
sec_rows, nsec, subs = [], [], Counter()
for case, jm in juror_secs.items():
    if len(jm) < 3: continue
    all_secs = Counter()
    for mm in jm.values(): all_secs.update(mm.keys())
    n_j = len(jm); kept_secs = 0
    for sec, seen in all_secs.items():
        if seen < max(3, n_j - 2): continue
        gv, sv = Counter(), Counter()
        for mm in jm.values():
            if sec in mm: gv.update(mm[sec][0]); sv.update(mm[sec][1])
        grades = [g for g, c in gv.items() if c >= 3 and g in GRADE_ORD]
        if not grades: continue
        kept_secs += 1; sec_rows.append(max(grades, key=lambda g: GRADE_ORD[g])); subs.update(s for s, c in sv.items() if c >= 3)
    nsec.append(kept_secs)
res["per_section"] = {"reports_with_consensus": len(nsec), "sections_per_report": mr(nsec), "section_worst_grade_dist": dict(Counter(sec_rows)), "cancer_subtype_sections": dict(subs),
                      "jurors": sorted({j for jm in juror_secs.values() for j in jm})}
if os.path.exists(T + "/labeller/erin_slide_labels_v2.csv"):
    sv2 = pd.read_csv(T + "/labeller/erin_slide_labels_v2.csv", dtype=str)
    res["per_section"]["dual_labelled_slides"] = {"n": len(sv2), "worst_grade": sv2.worst_grade.value_counts().to_dict(), "case_max": sv2.case_max.value_counts().to_dict(), "frac_differ": round(float((sv2.worst_grade != sv2.case_max).mean()), 4)}
# inter-model agreement, full corpus (whole-report jury)
ev = {}
for f in glob.glob(T + "/labeller/llm_full/llm_grades_*.csv"):
    model = os.path.basename(f).replace("llm_grades_", "").rsplit("_shard", 1)[0]; d = pd.read_csv(f, dtype=str); d = d[d.llm_grade.isin(ORD)]
    ev.setdefault(model, {}).update(dict(zip(d.CaseName, d.llm_grade)))
models = sorted(ev); common = sorted(set.intersection(*(set(v) for v in ev.values())))
pair = {}
for a, b in combinations(models, 2): pair[f"{a}|{b}"] = round(float(np.mean([ev[a][c] == ev[b][c] for c in common])), 4)
M = np.array([[ORD[ev[mo][c]] for mo in models] for c in common]); k = len(ORD); n = len(models)
P = np.array([[np.sum(row == j) for j in range(k)] for row in M]) / n
Pi = ((P ** 2).sum(1) * n - 1) / (n - 1); Pbar = Pi.mean(); pj = P.mean(0); Pe = (pj ** 2).sum(); fleiss = (Pbar - Pe) / (1 - Pe)
unan = float(np.mean([len(set(row)) == 1 for row in M]))
res["inter_model_agreement_full_corpus"] = {"models": models, "n_reports_all_models": len(common), "mean_pairwise_agreement": round(float(np.mean(list(pair.values()))), 4), "min_pairwise": min(pair.values()), "max_pairwise": max(pair.values()),
                                           "fleiss_kappa": round(float(fleiss), 4), "frac_unanimous": round(unan, 4), "pairwise": pair}
json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=2, default=str); print(json.dumps({k: v for k, v in res.items() if k in ("reports", "slides", "features", "per_section")}, indent=1, default=str)[:4000])

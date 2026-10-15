#!/usr/bin/env python3
"""Item 7 (24 Sep 2026): ERIN patient-level counts from the report table + jury labels + all-slides table."""
import json, os, numpy as np, pandas as pd
T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"; OUT = os.environ.get("OUTDIR", ".")
REP = "/mnt/scratche/fast/fmlab/datasets/imaging/ERIN/data/PathologyReport_AnonIds.csv"
rep = pd.read_csv(REP, dtype=str).fillna("")
lab = pd.read_csv(f"{T}/labeller/erin_labels_jury_final.csv", dtype=str).drop_duplicates("CaseName")
rep = rep.merge(lab[["CaseName", "final_label", "label_status"]], on="CaseName", how="left")
rep["d"] = pd.to_datetime(rep.CollectedOrOrdered, dayfirst=True, errors="coerce")
rep["res"] = rep.SpecimenProtocol.str.upper().str.contains("RESECT|ECTOMY")
ORD = {"NDBE": 0, "IND": 1, "LGD": 2, "HGD": 3, "CANCER": 4}; rep["g"] = rep.final_label.map(ORD)
pts = rep.groupby("anon_id")
gmax = pts.g.max()
per = pts.size(); fu = (pts.d.max() - pts.d.min()).dt.days
sl = pd.read_csv(f"{T}/campaigns/allslides/erin_slides_all.csv", dtype=str).fillna("")
sl = sl.merge(rep[["CaseName", "d", "anon_id"]].drop_duplicates("CaseName"), on="CaseName", how="left", suffixes=("", "_rep"))
he = sl[sl.stain_type.str.upper().str.contains("H&E|HE|H_E|HAEM", regex=True) | sl.stain_type.eq("")]
acq = pd.to_datetime(sl.acquired, errors="coerce")
res = {"source_report_table": REP, "source_labels": f"{T}/labeller/erin_labels_jury_final.csv", "source_slides": f"{T}/campaigns/allslides/erin_slides_all.csv",
  "reports": len(rep), "patients": int(rep.anon_id.nunique()), "reports_graded": int(rep.g.notna().sum()),
  "patients_any_CANCER": int((gmax >= 4).sum()), "patients_any_HGD_plus": int((gmax >= 3).sum()), "patients_any_LGD_plus": int((gmax >= 2).sum()),
  "patients_any_IND_plus": int((gmax >= 1).sum()), "patients_max_NDBE": int((gmax == 0).sum()), "patients_no_graded_report": int(gmax.isna().sum()),
  "resection_reports": int(rep.res.sum()), "resection_patients": int(rep[rep.res].anon_id.nunique()),
  "reports_per_patient": {"median": float(per.median()), "min": int(per.min()), "max": int(per.max()), "mean": round(float(per.mean()), 2), "one_report_only": int((per == 1).sum())},
  "followup_first_to_last_report_days": {"median": float(fu.median()), "q25": float(fu.quantile(.25)), "q75": float(fu.quantile(.75)), "max": int(fu.max()),
                                          "patients_with_ge2_reports": int((per >= 2).sum()), "median_among_ge2": float(fu[per >= 2].median()), "q25_ge2": float(fu[per >= 2].quantile(.25)), "q75_ge2": float(fu[per >= 2].quantile(.75))},
  "report_year_range": [int(rep.d.min().year), int(rep.d.max().year)], "reports_by_year": rep.d.dt.year.value_counts().sort_index().to_dict(),
  "slides_rows": len(sl), "slides_matched_to_report": int(sl.d.notna().sum()), "slide_report_year_range": [int(sl.d.min().year), int(sl.d.max().year)],
  "slides_by_report_year": sl.d.dt.year.value_counts().sort_index().to_dict(),
  "slide_scan_date_range": [str(acq.min().date()) if acq.notna().any() else None, str(acq.max().date()) if acq.notna().any() else None],
  "stain_type_dist": sl.stain_type.value_counts().head(8).to_dict(), "patients_with_any_slide": int(sl.anon_id.nunique())}
os.makedirs(OUT, exist_ok=True); json.dump(res, open(os.path.join(OUT, "erin_patient_counts.json"), "w"), indent=1, default=str); print(json.dumps(res, indent=None, default=str))

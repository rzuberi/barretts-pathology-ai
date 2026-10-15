"""NUMBERS A/B for the SWG Barrett's cohort: counts, per-patient structure, years, grades,
progression timing, CNV coverage, UNI2 patch counts, modality intersection."""
import json, os, re, glob
import numpy as np, pandas as pd
F = "/mnt/scratche/slow/fmlab/zuberi01/phd/barretts_retraining/barretts_training/analysis/chapter1_lgd2_final_pre_event_20260713_final"
PS = "/mnt/scratche/slow/fmlab/zuberi01/phd/barretts_retraining/barretts_training/multimodal-barretts-progression/PROJECT_STATE.md"
T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"
OUT = os.environ.get("OUTDIR", ".")
def mr(s): s = pd.Series(s).dropna(); return {"median": float(s.median()), "min": float(s.min()), "max": float(s.max()), "n": int(len(s))} if len(s) else None
man = pd.read_csv(F + "/training_manifest.csv", dtype=str)
coh = pd.read_csv(F + "/pre_event_cohort.csv", dtype=str).merge(man, left_on="SampleID", right_on="sample_id")
coh["Date"] = pd.to_datetime(coh["Date"], errors="coerce")
num = lambda c: pd.to_numeric(coh[c], errors="coerce")
res = {"source": F}
res["patients"] = int(man.patient_id.nunique()); res["samples_rows"] = len(man)
pp = man.groupby("patient_id").y_progressor.agg(lambda v: int(v.astype(int).max()))
res["progressor_patients"] = int(pp.sum()); res["positive_samples_next_biopsy_lgd2"] = int(man.y_progressor.astype(int).sum())
res["endpoint_note"] = "y_progressor is PER SAMPLE: the next biopsy after this sample shows LGD confirmed on two reads or worse; a patient is a progressor if any sample is positive"
res["alt_Progressor_label_col"] = {"positive_samples": int((coh.Progressor_label.astype(str) == "1").sum()), "positive_patients": int(coh[coh.Progressor_label.astype(str) == "1"].patient_id.nunique())}
res["folds"] = sorted(man.fold_id_rep01.unique().tolist())
per = coh.groupby("patient_id").size(); res["samples_per_patient"] = mr(per)
res["years"] = {"min": int(coh.Date.min().year), "max": int(coh.Date.max().year), "by_year": coh.Date.dt.year.value_counts().sort_index().to_dict()}
GR = {"0": "NDBE", "1": "IND", "2": "LGD", "3": "HGD", "4": "IMC/cancer"}
lab = coh["Label"].astype(str).str.strip()
res["grade_dist_sample_level_codes"] = lab.value_counts().to_dict()
res["grade_code_assumed_names"] = GR
maxg = coh.assign(g=pd.to_numeric(lab, errors="coerce")).groupby("patient_id").g.max()
res["grade_dist_patient_max"] = maxg.value_counts().sort_index().to_dict()
first = coh.sort_values("Date").groupby("patient_id").first()
res["entry_grade_first_sample"] = first["Label"].astype(str).value_counts().to_dict()
res["max_pathology_col_dist"] = coh["max_pathology"].value_counts().to_dict() if "max_pathology" in coh else None
progp = set(pp[pp == 1].index); prog = coh[coh.patient_id.isin(progp)]; nonp = coh[~coh.patient_id.isin(progp)]
res["progression_timing"] = {
    "Time_to_progression_per_progressor_patient": mr(pd.to_numeric(prog.groupby("patient_id")["Time_to_progression"].first(), errors="coerce")),
    "Time_to_progression_units_note": "days (median 1095 = 3 years); non-null on the samples of progressor patients",
    "Time_to_progression_all_nonnull_samples_days": mr(num("Time_to_progression")),
    "MonthsBeforeLastBiopsy_nonprogressor_samples": mr(pd.to_numeric(nonp["MonthsBeforeLastBiopsy"], errors="coerce")),
    "biopsies_total_for_patient": mr(pd.to_numeric(coh["BiopsiesTotalForPatient"], errors="coerce")),
    "MonthsBeforeLastBiopsy_progressor_samples": mr(pd.to_numeric(prog["MonthsBeforeLastBiopsy"], errors="coerce")),
    "first_to_last_sample_span_months_progressors": mr((prog.groupby("patient_id").Date.max() - prog.groupby("patient_id").Date.min()).dt.days / 30.44),
    "first_to_last_sample_span_months_nonprogressors": mr((nonp.groupby("patient_id").Date.max() - nonp.groupby("patient_id").Date.min()).dt.days / 30.44),
    "DaysToNextBiopsy_all": mr(num("DaysToNextBiopsy")),
    "Progress_in_k_sample_counts": {k: int(pd.to_numeric(coh[k], errors="coerce").fillna(0).sum()) for k in ["Progress_in_1", "Progress_in_2", "Progress_in_3", "Progress_in_4", "Progress_in_5"] if k in coh},
    "condition_col_dist": man["condition"].value_counts().to_dict()}
cx = pd.read_csv(F + "/feature_views/cnv/cx.csv", dtype=str)
res["cnv"] = {"samples_with_cx": int(cx.sample_id.nunique()), "patients": int(cx.patient_id.nunique()), "unique_cnv_ids": int(cx.cnv_id.nunique())}
u = pd.read_csv(F + "/feature_views/uni2/uni2_index.csv", dtype=str); ok = u[u.status == "ok"]
res["uni2"] = {"slides_ok": len(ok), "status_counts": u.status.value_counts().to_dict(),
               "patches_per_slide": mr(pd.to_numeric(ok.n_instances, errors="coerce")), "feat_dim": ok.feat_dim.iloc[0],
               "magnification_note": "uni2_tile224_lvl2 = 224px tiles at pyramid level 2 (see release); thesis states 20x/224px"}
both = set(ok.sample_id) & set(cx.sample_id) & set(man.sample_id)
res["modality_intersection"] = {"samples_uni2_and_cnv_in_manifest": len(both), "patients": int(man[man.sample_id.isin(both)].patient_id.nunique()),
                                "manifest_samples_missing_uni2": int((~man.sample_id.isin(ok.sample_id)).sum()), "manifest_samples_missing_cx": int((~man.sample_id.isin(cx.sample_id)).sum())}
sl = pd.read_csv(T + "/docs/swg_barretts_slides.csv", dtype=str)
res["slides_on_disk_csv"] = {"rows": len(sl), "cols": list(sl.columns), "with_patient": int(sl.iloc[:, 0].notna().sum()), "patients": int(sl.iloc[:, 0].nunique()),
                             "grade_col_dist": sl.iloc[:, 2].value_counts(dropna=False).to_dict(), "P_NP": sl.iloc[:, 3].value_counts(dropna=False).to_dict(),
                             "slides_per_patient": mr(sl.dropna(subset=[sl.columns[0]]).groupby(sl.columns[0]).size())}
try:
    txt = open(PS).read().splitlines()
    res["project_state_lines"] = [l for l in txt if re.search(r"match|control|resolution|\bbin|kb\b|sWGS|repeat|rep0|nested|seed|hospital|Addenbrooke|LGD2|two read|confirm", l, re.I)][:60]
except Exception as e: res["project_state_lines"] = str(e)
json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=2, default=str); print(json.dumps(res, indent=1, default=str)[:3000])

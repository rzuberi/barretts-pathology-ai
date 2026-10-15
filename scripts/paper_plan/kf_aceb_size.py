"""Killcoyne final item 3 (docs/paper_plan_killcoyne_final.md @ 8342ac0): ACE-B size from the sample manifest joined to the Barrett's-database
timeline through the official case list. Aggregates only -> feasibility/paper_plan/killcoyne_mm/final/aceb_size.json."""
import json, glob, os, pandas as pd
A = "/mnt/scratche/slow/fmlab/zuberi01/phd/aceb_meta"; T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"; O = T + "/feasibility/paper_plan/killcoyne_mm/final"; os.makedirs(O, exist_ok=True)
s = pd.read_csv(sorted(glob.glob(A + "/aceb_samples_direct_patientid_link_*.csv"))[-1], dtype=str)
oc = pd.read_csv(A + "/aceb_official_case_presence_in_barretts_db_20260527_150857.csv", dtype=str)
tl = pd.read_csv(T + "/feasibility/runs/num_aceb_db_v2/output/aceb_participant_timeline.csv", dtype=str)
case2pid = oc.set_index("study_number_normalized").participant_ids   # 1 participant per case except one case with 3 (then no unique status)
st = tl.set_index("pid")
def status(case):
    p = case2pid.get(case)
    if not isinstance(p, str) or not p or len(p.replace(";", ",").split(",")) != 1 or p not in st.index: return "no_database_status"
    r = st.loc[p]; return "prevalent" if r.prevalent_HGD_plus == "True" else ("progressed" if r.progressed_to_HGD_plus == "True" else "non_progressor")
s["status"] = s.PatientID_normalized.map(status); im = s[s.Pathology == "IM"]
cases = im.groupby("PatientID_normalized").agg(n_im=("Pathology", "size"), status=("status", "first"))
res = {"manifest_samples": int(len(s)), "manifest_cases": int(s.PatientID_normalized.nunique()), "manifest_pathology": s.Pathology.fillna("missing").value_counts().to_dict(),
       "IM_samples": int(len(im)), "cases_with_IM": int(len(cases)), "IM_cases_by_status": cases.status.value_counts().to_dict(), "IM_samples_by_status": im.status.value_counts().to_dict(),
       "owner_split": {"progressors": 11, "non_progressors": 93, "prevalent_HGD_IMC": 30, "patients": 134, "samples": 294, "source": "docs/NUMBERS.md item 25 (Leanne, 21 Sep 2026)"},
       "timeline_source": "feasibility/runs/num_aceb_db_v2/output/aceb_participant_timeline.csv (LLM-jury grades from the Barrett's DB; cross-check, not trial-adjudicated)"}
bs = res["IM_cases_by_status"]; ss = res["IM_samples_by_status"]
res["scenarios"] = {"S1": {"n_P": bs.get("prevalent", 0) + bs.get("progressed", 0), "n_NP": bs.get("non_progressor", 0), "n_samples": ss.get("prevalent", 0) + ss.get("progressed", 0) + ss.get("non_progressor", 0), "label": "manifest IM samples; P = prevalent or progressed (database)"},
                    "S2": {"n_P": bs.get("progressed", 0), "n_NP": bs.get("non_progressor", 0), "n_samples": ss.get("progressed", 0) + ss.get("non_progressor", 0), "label": "S1 without prevalent patients"},
                    "S3": {"n_P": 41, "n_NP": 93, "n_samples": None, "label": "owner's split, P = 11 progressors + 30 prevalent, NP = 93, all assumed to contribute NDBE samples"}}
# added post hoc (sample-count-matched simulation): NDBE samples per case, by status
res["IM_per_case_hist"] = {k: {str(n): int(c) for n, c in g.n_im.value_counts().sort_index().items()} for k, g in cases.groupby("status")}
json.dump(res, open(O + "/aceb_size.json", "w"), indent=1); print(json.dumps(res, indent=1))

"""Demographics completion D2/D3 add-on (docs/demographics_completion.md @ 24b2972): per-patient agreement between Demographics_full.csv (sheet) and
BE_Progression_Project.db Patient (SQLite) for the discovery subset and release; implausible database ages; the 777-sheet columns that could hold
demographics. Reads feasibility/paper_plan/demographics/d3_patient_rows.csv (dm_analyse.py PART=d3). -> results/paper_final/demographics/d3_agree.json"""
import re, json, numpy as np, pandas as pd
T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"; D = T + "/feasibility/paper_plan/demographics"; S = "/mnt/scratche/fast/fmlab/datasets/imaging/SWGCohort"
R = pd.read_csv(D + "/d3_patient_rows.csv"); out = {}
for coh in ("discovery", "release"):
    g = R[R.cohort == coh]; o = {}
    def eq(a, b, f=lambda x: x):
        m = g[a].notna() & g[b].notna(); return {"both": int(m.sum()), "agree": int((g.loc[m, a].map(f) == g.loc[m, b]).sum()) if m.any() else 0}
    o["sex"] = eq("sex_sheet", "sex_sqlite", lambda x: {"M": "male", "F": "female"}.get(x, x))
    m = g.age_dx_sheet.notna() & g.age_dx_sqlite.notna(); o["age_dx"] = {"both": int(m.sum()), "agree_within_0.5y": int(((g.age_dx_sheet - g.age_dx_sqlite).abs()[m] <= 0.5).sum())}
    for v in ("prague_c", "prague_m"):
        m = g[f"{v}_sheet"].notna() & g[f"{v}_sqlite"].notna(); o[v] = {"both": int(m.sum()), "agree": int((g[f"{v}_sheet"] == g[f"{v}_sqlite"])[m].sum())}
    o["smoking"] = eq("smoking_sheet", "smoking_sqlite", lambda x: x); m = g.smoking_sheet.notna() & g.smoking_sqlite.notna()
    o["smoking"]["agree_Y_eq_former_or_current"] = int(((g.smoking_sheet == "Y") & g.smoking_sqlite.isin(["former", "current"]) | (g.smoking_sheet == "N") & (g.smoking_sqlite == "never"))[m].sum())
    for a, b in (("age_dx_db", "age_dx_sheet"), ("age_first_sample_db", "age_first_sample_sheet")):
        m = g[a].notna() & g[b].notna(); d = (g[a] - g[b])[m]; o[f"{a}_minus_{b}"] = {"both": int(m.sum()), "median_abs_years": None if not m.any() else round(float(d.abs().median()), 2), "within_1y": int((d.abs() <= 1).sum())}
    o["age_first_sample_db_below_18"] = int((g.age_first_sample_db < 18).sum()); o["age_dx_db_below_18"] = int((g.age_dx_db < 18).sum())
    o["sqlite_only_patients_sex"] = int((g.sex_sheet.isna() & g.sex_sqlite.notna()).sum()); o["sqlite_only_patients_smoking"] = int((g.smoking_sheet.isna() & g.smoking_sqlite.notna()).sum())
    out[coh] = o
fd = pd.read_csv(S + "/sWGS_777_samples_cleaned_202401_Leanne_fullDetails (3) (1).csv", nrows=0); cols = [c.strip().replace("\n", " ") for c in fd.columns]
out["sheet777_columns"] = cols; out["sheet777_demographic_columns"] = [c for c in cols if re.search(r"age|sex|gender|smok|length|prague|circum|maxim|diagnos|birth|dob|bmi|weight|height", c, re.I)]
json.dump(out, open(T + "/results/paper_final/demographics/d3_agree.json", "w"), indent=1); print("DM AGREE DONE")

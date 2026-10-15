"""Demographics completion D2 (docs/demographics_completion.md @ 24b2972): Supplementary Table 1 (Killcoyne 2020, UK discovery cohort) summaries recomputed
on (a) the discovery subset (80 patients, sheet Status) and (b) all Demographics_full.csv rows (P/NP column), and Supplementary Table 2 (validation) on the
release's 268-sheet patients (validation-sheet Status) from BE_Progression_Project.db. Same summaries as the tables: mean ± SD, counts. No tests.
-> results/paper_final/demographics/d2_st_compare.json"""
import json, sqlite3, numpy as np, pandas as pd
T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"; S = "/mnt/scratche/fast/fmlab/datasets/imaging/SWGCohort"; M = T + "/feasibility/paper_plan/killcoyne_mm"; AM = "/mnt/scratche/slow/fmlab/zuberi01/phd/aceb_meta"
ms = lambda v: (lambda a: {"n": int(len(a)), "mean": round(float(a.mean()), 1), "sd": round(float(a.std(ddof=1)), 1)} if len(a) > 1 else {"n": int(len(a))})(pd.to_numeric(pd.Series(v), errors="coerce").dropna())
dm = pd.read_csv(S + "/Demographics_full.csv", dtype=str)
for c in ("Study Number", "Alternate Study Number"): dm[c] = dm[c].str.strip()
dm["lab"] = dm["P/NP"].str.strip()
def summ(g):
    sx = g["Sex"].str.strip(); sm = g["Smoking Status"].str.strip()
    return {"patients": int(len(g)), "age_at_diagnosis": ms(g["Age at diagnosis"]), "be_length_maximal_cm": ms(g["Maximal"]), "be_length_not_recorded": int(pd.to_numeric(g["Maximal"], errors="coerce").isna().sum()),
            "female": int((sx == "F").sum()), "male": int((sx == "M").sum()), "smoking_Y": int((sm == "Y").sum()), "smoking_N": int((sm == "N").sum()), "smoking_not_recorded": int((~sm.isin(["Y", "N"])).sum()),
            "followup_years": ms(g["Total Followup (years)"]), "followup_range": [pd.to_numeric(g["Total Followup (years)"], errors="coerce").min(), pd.to_numeric(g["Total Followup (years)"], errors="coerce").max()]}
out = {"demographics_full_all": {lab: summ(g) for lab, g in dm.groupby("lab")}}
C = pd.read_csv(M + "/set_C.csv", dtype=str); fd = pd.read_csv(S + "/sWGS_777_samples_cleaned_202401_Leanne_fullDetails (3) (1).csv", dtype=str); fd.columns = [c.strip().replace("\n", " ") for c in fd.columns]
C["hrid"] = C.Sample.map(fd.drop_duplicates("combined_name").set_index("combined_name")["Hospital_Research_ID_updated"].str.strip())
pat = C.groupby("Patient").agg(hrid=("hrid", lambda s: s.dropna().mode().iloc[0]), status=("Status", "first"))
key = {}
for i, r in dm.iterrows():
    for c in ("Study Number", "Alternate Study Number"):
        if isinstance(r[c], str) and r[c] and r[c] not in key: key[r[c]] = i
pat["row"] = pat.hrid.map(key); sub = dm.loc[pat.row.dropna().astype(int)].copy(); sub["lab"] = pat.dropna(subset=["row"]).status.map({"P": "P", "NP": "NP"}).values
out["discovery_subset_80"] = {lab: summ(g) for lab, g in sub.groupby("lab")}; out["discovery_subset_matched_rows"] = int(len(sub))
# ST2: release 268-sheet patients, labels from the validation sheet Status, demographics from BE_Progression_Project.db
va = pd.read_csv(S + "/sWGS_validation_cleaned_Leanne (4) (1).csv", dtype=str); va["PatientID"] = va.PatientID.str.strip(); vs = va.groupby("PatientID").Status.agg(lambda s: s.dropna().str.strip().mode().iloc[0] if s.notna().any() else None)
con = sqlite3.connect(f"file:{AM}/BE_Progression_Project.db?mode=ro", uri=True)
SQ = pd.read_sql("select ID, Gender, AgeAtDiagnosis, SmokingStatus, Maximal from Patient", con).set_index("ID"); ALT = pd.read_sql("select PatientID, AlternateID from AlternatePatientID", con); ALT["AlternateID"] = ALT.AlternateID.astype(str).str.strip()
sq_of = dict(zip(ALT.AlternateID, ALT.PatientID)); rows = []
for p, st in vs.items():
    q = sq_of.get(p)
    if q is None or q not in SQ.index: continue
    r = SQ.loc[q]; rows.append({"lab": st, "age": r.AgeAtDiagnosis, "max": r.Maximal, "sex": r.Gender, "smk": r.SmokingStatus})
V = pd.DataFrame(rows)
out["validation_sheet_patients_sqlite"] = {lab: {"patients": int(len(g)), "age_at_diagnosis": ms(g.age), "be_length_maximal_cm": ms(g["max"]), "female": int((g.sex == "female").sum()), "male": int((g.sex == "male").sum()),
                                                 "smoking_yes_former_or_current": int(g.smk.isin(["former", "current"]).sum()), "smoking_no_never": int((g.smk == "never").sum()), "smoking_not_recorded": int(g.smk.isna().sum() + (g.smk == "None").sum())} for lab, g in V.groupby("lab")}
out["validation_sheet_patients_total"] = int(len(vs)); out["validation_sheet_status_values"] = va.Status.astype(str).value_counts().to_dict()
json.dump(out, open(T + "/results/paper_final/demographics/d2_st_compare.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)); print("DM ST DONE")

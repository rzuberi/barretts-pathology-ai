"""Horizons H0 (docs/paper_survival_horizons.md @ ee51db8): per-sample event time, event, pre-event and NDBE flags, age at sample and sex for the
discovery subset (set C). Row-level output stays on the cluster: feasibility/paper_plan/killcoyne_mm/horizons/samples.csv. Event logic copied from
kc_merge.py:30-32 / kv_merge.py."""
import os, numpy as np, pandas as pd
T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"; K = T + "/feasibility/paper_plan/killcoyne"; M = K + "_mm"; S = "/mnt/scratche/fast/fmlab/datasets/imaging/SWGCohort"
F = "/mnt/scratche/slow/fmlab/zuberi01/phd/barretts_retraining/barretts_training/analysis/chapter1_lgd2_final_pre_event_20260713_final"
OUT = M + "/horizons"; os.makedirs(OUT, exist_ok=True)
C = pd.read_csv(M + "/set_C.csv", dtype=str); C["y"] = (C.Status == "P").astype(int)
fd = pd.read_csv(S + "/sWGS_777_samples_cleaned_202401_Leanne_fullDetails (3) (1).csv", dtype=str); fd.columns = [c.strip().replace("\n", " ") for c in fd.columns]; fd = fd.drop_duplicates("combined_name").set_index("combined_name")
A = pd.read_csv(K + "/kr_samples.csv", dtype=str).set_index("Sample"); A["mbf"] = pd.to_numeric(A.index.map(fd["Months before final"]), errors="coerce"); A["ogd"] = A.index.map(fd["Path_class_per_OGD"])
hg = A[(A.Pathology.isin(["HGD", "IMC"])) | (A.ogd.isin(["HGD", "IMC", "HGD/IMC"]))]; tev = hg.groupby("Patient").mbf.max()
Ppats = sorted(C[C.y == 1].Patient.unique()); fallback = [p for p in Ppats if p not in tev.index]; tev = pd.concat([tev, pd.Series(0.0, index=fallback)])
C["mbf"] = C.Sample.map(A.mbf).astype(float); assert C.mbf.notna().all()
C["tev"] = [tev[p] if y == 1 else np.nan for p, y in zip(C.Patient, C.y)]
C["fallback_endpoint"] = C.Patient.isin(fallback) & (C.y == 1)
C["pre"] = [True if y == 0 else (m > t) for y, m, t in zip(C.y, C.mbf, C.tev)]
C["time"] = np.where(C.y == 1, (C.mbf - C.tev) / 12.0, C.mbf / 12.0); C["event"] = C.y
C["ndbe"] = C.Pathology == "NDBE"
# age at sample and sex (Demographics_full.csv via the sheet's Hospital Research ID)
C["hrid"] = C.Sample.map(fd["Hospital_Research_ID_updated"].str.strip())
pe = pd.read_csv(F + "/pre_event_cohort.csv", dtype=str); pe["cnv"] = pe.CNVAbsPath.map(lambda p: os.path.basename(str(p))); pe["d"] = pd.to_datetime(pe.Date, errors="coerce")
pdate = pe.dropna(subset=["d"]).drop_duplicates("cnv").set_index("cnv").d
dm = pd.read_csv(S + "/Demographics_full.csv", dtype=str); dmi = {}
for _, r in dm.iterrows():
    for c in ("Study Number", "Alternate Study Number"):
        v = r[c].strip() if isinstance(r[c], str) else ""
        if v and v not in dmi: dmi[v] = r
first = C.sort_values(["Patient", "mbf"], ascending=[True, False]).groupby("Patient").first()
age0 = {}
for p, r in first.iterrows():
    h = C.loc[C.Patient == p, "hrid"].dropna().mode(); h = h.iloc[0] if len(h) else None; d = dmi.get(h)
    dob = pd.to_datetime(d["Date of birth"], errors="coerce") if d is not None else pd.NaT; fdt = pdate.get(r.Sample, pd.NaT)
    age0[p] = ((fdt - dob).days / 365.25, r.mbf, (d["Sex"].strip() if d is not None and isinstance(d["Sex"], str) else None), h)
C["age"] = [age0[p][0] + (age0[p][1] - m) / 12.0 for p, m in zip(C.Patient, C.mbf)]
C["sex_M"] = [1.0 if age0[p][2] == "M" else 0.0 if age0[p][2] == "F" else np.nan for p in C.Patient]
C["study_number"] = [age0[p][3] for p in C.Patient]
cols = ["Sample", "Patient", "study_number", "Status", "y", "Pathology", "P53 IHC", "k_prob", "mbf", "tev", "fallback_endpoint", "pre", "ndbe", "time", "event", "age", "sex_M"]
C[cols].to_csv(OUT + "/samples.csv", index=False)
print("HZ PREP DONE rows", len(C), "pre", int(C.pre.sum()), "pre&ndbe", int((C.pre & C.ndbe).sum()), "age_na", int(C.age.isna().sum()), "sex_na", int(C.sex_M.isna().sum()), "fallback_patients", len(fallback))

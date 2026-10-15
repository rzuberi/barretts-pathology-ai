"""Demographics completion: demographic columns of aceb_meta/BE_Progression_Project.db (Patient: Gender, AgeAtDiagnosis, SmokingStatus, Height, Weight,
DateInitialDiagnosis, Circumference, Maximal, Type; AlternatePatientID) -- value distributions and identifier formats only. Status and DateProgressed
are not read. Output: feasibility/paper_plan/demographics/probe4_sqlite.json"""
import sqlite3, json, re, pandas as pd
f = "/mnt/scratche/slow/fmlab/zuberi01/phd/aceb_meta/BE_Progression_Project.db"; con = sqlite3.connect(f"file:{f}?mode=ro", uri=True)
P = pd.read_sql("select ID, UUID, Gender, AgeAtDiagnosis, SmokingStatus, Height, Weight, DateInitialDiagnosis, Circumference, Maximal, Type from Patient", con)
A = pd.read_sql("select PatientID, AlternateID, AltIDStudy from AlternatePatientID", con)
fmt = lambda v: re.sub(r"[0-9]", "9", re.sub(r"[A-Za-z]", "A", str(v)))
out = {"Patient_rows": len(P), "values": {c: P[c].astype(str).value_counts(dropna=False).head(15).to_dict() for c in ("Gender", "SmokingStatus", "Type")},
       "nonnull": {c: int(P[c].notna().sum()) for c in P.columns}, "UUID_formats": pd.Series([fmt(v) for v in P.UUID]).value_counts().head(5).to_dict(),
       "Alt_rows": len(A), "AltIDStudy": A.AltIDStudy.astype(str).value_counts().to_dict(), "AlternateID_formats": pd.Series([fmt(v) for v in A.AlternateID]).value_counts().head(8).to_dict()}
json.dump(out, open("/mnt/scratche/slow/fmlab/zuberi01/phd/thesis/feasibility/paper_plan/demographics/probe4_sqlite.json", "w"), indent=1, default=str); print("PROBE4 DONE")

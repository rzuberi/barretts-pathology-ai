"""Dataset description, probe 2: identifier formats for the release -> Barrett's DB and release -> Demographics joins. Prints formats and match counts only."""
import re, pandas as pd
E = "/mnt/scratche/slow/fmlab/zuberi01/barretts_db_export"
F = "/mnt/scratche/slow/fmlab/zuberi01/phd/barretts_retraining/barretts_training/analysis/chapter1_lgd2_final_pre_event_20260713_final"
S = "/mnt/scratche/fast/fmlab/datasets/imaging/SWGCohort"
fmt = lambda v: re.sub(r"[0-9]", "9", re.sub(r"[A-Za-z]", "A", str(v)))
pe = pd.read_csv(F + "/pre_event_cohort.csv", dtype=str)
print("pe participant_id formats", pd.Series([fmt(v) for v in pe.participant_id.dropna()]).value_counts().head(5).to_dict())
print("pe PatientID_real formats", pd.Series([fmt(v) for v in pe.PatientID_real.dropna()]).value_counts().head(5).to_dict())
ih = pd.read_parquet(E + "/initial_history.parquet"); en = pd.read_parquet(E + "/endoscopy.parquet")
for name, d in [("initial_history", ih), ("endoscopy", en)]:
    for c in d.columns:
        if re.search("participant|patient|anonref|hospitalnumber|selected", c, re.I):
            vals = d[c].dropna().astype(str)
            print(name, c, pd.Series([fmt(v) for v in vals]).value_counts().head(4).to_dict())
pid = set(pe.participant_id.dropna().astype(str).str.replace(r"\.0$", "", regex=True))
rid = set(pe.PatientID_real.dropna().astype(str))
for name, d in [("initial_history", ih), ("endoscopy", en)]:
    for c in d.columns:
        vals = set(d[c].dropna().astype(str).str.replace(r"\.0$", "", regex=True))
        a, b = len(pid & vals), len(rid & vals)
        if a or b: print("MATCH", name, c, "participant_id", a, "PatientID_real", b)
print("ih smoking all", ih.smoking.astype(str).value_counts().head(10).to_dict())
print("ih praguec all nonnull", ih.praguec.notna().sum(), ih.praguec.dropna().astype(str).head(5).tolist())
print("en date cols", [c for c in en.columns if re.search("date", c, re.I)])
dm = pd.read_csv(S + "/Demographics_full.csv", dtype=str)
for c in ["Study Number", "Alternate Study Number"]: print("dm", c, pd.Series([fmt(v) for v in dm[c].dropna()]).value_counts().head(4).to_dict(), len(set(dm[c].dropna()) & rid))
fd = pd.read_csv(S + "/sWGS_777_samples_cleaned_202401_Leanne_fullDetails (3) (1).csv", dtype=str)
print("777 cols", [c.replace("\n", " ") for c in fd.columns])
for c in fd.columns:
    if re.search("patient|study|hospital", c, re.I): print("777", c.replace("\n", " "), pd.Series([fmt(v) for v in fd[c].dropna()]).value_counts().head(3).to_dict(), len(set(fd[c].dropna()) & rid))
va = pd.read_csv(S + "/sWGS_validation_cleaned_Leanne (4) (1).csv", dtype=str)
print("val PatientID", pd.Series([fmt(v) for v in va.PatientID.dropna()]).value_counts().head(3).to_dict(), len(set(va.PatientID.dropna()) & rid))
print("DONE_PROBE2")

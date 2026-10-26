"""ABD step 2: key formats only (letters->A, digits->9), counts; masked header names; Feb-2026
scrape header; study-name membership counts. No values printed."""
import re, csv, glob, os
from collections import Counter
import pandas as pd

Z = "/mnt/scratche/slow/fmlab/zuberi01"; META = f"{Z}/phd/aceb_meta"; DB = f"{Z}/barretts_db_export"
FEB = f"{Z}/phd/barretts_retraining/barretts_training/data/barretts_scrape_20260220_172725"
def fmt(s): return re.sub(r"\d", "9", re.sub(r"[A-Za-z]", "A", str(s).strip()))
def top(series, k=12):
    c = Counter(fmt(x) for x in series.dropna()); n = sum(c.values())
    return [(f, v if v >= 5 else "<5") for f, v in c.most_common(k)], n
def hdr(p):
    with open(p, newline="", encoding="utf-8", errors="replace") as f: return next(csv.reader(f))

print("== header names containing ACE/PS (digit runs masked) ==")
for p in sorted(glob.glob(f"{META}/*.csv")):
    h = hdr(p); m = [re.sub(r"\d{3,}", "<d>", t) for t in h if re.search(r"ACE|PS|\d{3,}", t, re.I)]
    if m: print(os.path.basename(p), m)
print("== Feb-2026 scrape folder ==")
for p in sorted(os.listdir(FEB)):
    q = f"{FEB}/{p}"
    print(p, os.path.getsize(q), hdr(q) if p.endswith(".csv") else "")

s = pd.read_csv(f"{META}/ACEB_samples_for Rehan.csv", usecols=["Batch", "Name", "PatientID"], dtype=str)
print("== ACE-B sheet ==", len(s), "rows;", s.PatientID.nunique(), "PatientID;", s.Name.nunique(), "Name; batches", s.Batch.nunique())
print("Name formats", top(s.Name)); print("PatientID formats", top(s.PatientID))
c = pd.read_csv(f"{META}/ACE-B cases for Rehan.csv", usecols=["Study Number", "Date AFI", "Date WLE"], dtype=str)
print("== case list ==", len(c), "rows;", c["Study Number"].nunique(), "study numbers")
print("Study Number formats", top(c["Study Number"])); print("Date AFI formats", top(c["Date AFI"])); print("Date WLE formats", top(c["Date WLE"]))
t = pd.read_csv(f"{META}/aceb_rehan_study_number_match_table_20260527_1323.csv",
                usecols=["study_number_raw", "study_number_normalized", "found_in_scrape_filter", "participant_ids_in_filter"], dtype=str)
print("== May study-number match table ==", len(t), "rows")
for col in t.columns: print(col, top(t[col], 6))
m = pd.read_csv(f"{META}/aceb_ps_to_participant_mapping_20260527_143713.csv", usecols=[0, 1, 2], dtype=str)
print("== May PS->participant mapping ==", len(m), "rows; cols", list(m.columns))
for col in m.columns: print(col, top(m[col], 6))

pt = pd.read_csv(f"{DB}/pathology_text_normalised_full.csv",
                 usecols=["pathology_text_id", "participant_id", "receiveddatetime", "specimennumber", "impdate",
                          "newatlastimport", "updatedatlastimport", "rowupdatedatetime"], dtype=str)
print("== Aug-2026 export pathology ==", len(pt), "rows;", pt.participant_id.nunique(), "participants")
for col in pt.columns: print(col, top(pt[col], 8))
for col in ["receiveddatetime", "impdate", "rowupdatedatetime"]:
    d = pd.to_datetime(pt[col], errors="coerce", utc=True)
    print(col, "min", d.min(), "max", d.max(), "n>=2026-06-01:", int((d >= pd.Timestamp("2026-06-01", tz="UTC")).sum()))
    print("  by year (last 6):", d.dt.year.value_counts().sort_index().tail(6).to_dict())
for f in ["endoscopy.parquet", "mat_endoscopy_full.parquet"]:
    e = pd.read_parquet(f"{DB}/{f}", columns=["participant_id", "endoscopydate", "study_id", "rowupdatetimestamp"])
    for col in ["endoscopydate", "rowupdatetimestamp"]:
        d = pd.to_datetime(e[col], errors="coerce", utc=True)
        print(f, col, "max", d.max(), "n>=2026-06-01:", int((d >= pd.Timestamp("2026-06-01", tz="UTC")).sum()))
    print(f, "study_id counts", e.study_id.value_counts(dropna=False).head(15).to_dict())

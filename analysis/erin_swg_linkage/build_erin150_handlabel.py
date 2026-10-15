"""150 ERIN oesophageal/GOJ sections (2023-2025 reports), stratified by jury-v3 grade with
IND/LGD/HGD oversampled, loaded into the cluster grading app for grader 'rehan'.
Read-only on ERIN. Writes: hand_grading/cases.json (appended block 'erin150'),
hand_grading/erin150_key.csv (strata + jury grade; NOT for graders), assignments rows.
Then the app must be restarted (it loads cases.json at start)."""
import json, os, re, sqlite3, shutil, datetime
import numpy as np, pandas as pd

T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"; H = "/mnt/scratche/slow/fmlab/zuberi01/hand_grading"
ERIN = "/mnt/scratche/fast/fmlab/datasets/imaging/ERIN/data/PathologyReport_AnonIds.csv"
TARGET = {"IND": 30, "LGD": 30, "HGD": 30, "CANCER": 20, "NDBE": 25, "NORMAL_OTHER": 15}   # = 150
rng = np.random.RandomState(150)

sec = pd.read_csv(T + "/labeller/erin_sections_v3_consensus.csv", dtype=str).fillna("")
rep = pd.read_csv(ERIN, dtype=str, low_memory=False).fillna("").drop_duplicates("CaseName").set_index("CaseName")
rep["year"] = pd.to_datetime(rep.CollectedOrOrdered, dayfirst=True, errors="coerce").dt.year
sec = sec[sec.site.isin(["OESOPHAGUS", "GOJ"]) & (sec.section != "WHOLE")]
sec = sec[sec.CaseName.map(lambda c: rep.year.get(c, 0) >= 2023)]
# exclude cases already in the app or in the earlier hand-label/adjudication sets
existing = {c["case"].split("#")[0] for c in json.load(open(H + "/cases.json"))}
for f in [T + "/labeller/handlabel_sample_key.csv", T + "/labeller/adjudications.csv"]:
    if os.path.exists(f): existing |= set(pd.read_csv(f, dtype=str).CaseName.dropna())
sec = sec[~sec.CaseName.isin(existing)]
# one section per report, one report per patient
sec = sec.assign(anon=sec.CaseName.map(rep.anon_id)).sample(frac=1, random_state=150)
sec = sec.drop_duplicates("CaseName").drop_duplicates("anon")
print("eligible sections after filters:", len(sec), sec.worst_grade.value_counts().to_dict())
picks = []
for g, n in TARGET.items():
    pool = sec[sec.worst_grade == g]
    take = pool.sample(min(n, len(pool)), random_state=int(rng.randint(1e6)))
    picks.append(take); print(f"  {g}: wanted {n}, got {len(take)}")
sample = pd.concat(picks).sample(frac=1, random_state=7).reset_index(drop=True)

cases = []
for _, r in sample.iterrows():
    R = rep.loc[r.CaseName]
    text = (f"*** GRADE SECTION {r.section} ONLY (add a row for section {r.section}; other sections optional) ***\n\n"
            f"GROSS DESCRIPTION:\n{R.GrossDescription_redacted}\n\nFINAL DIAGNOSIS:\n{R.FinalDiagnosis_redacted}\n\n"
            f"MICROSCOPIC DESCRIPTION:\n{R.MicroscopicDescription_redacted}")
    cases.append({"case": f"{r.CaseName}#{r.section}", "block": "erin150", "text": text})
key = sample[["CaseName", "section", "site", "worst_grade", "grades", "n_jurors", "site_frac"]].copy()
key["case_id"] = [c["case"] for c in cases]

# ---- write ----
ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")
shutil.copy2(H + "/cases.json", f"{H}/cases.json.bak_{ts}"); shutil.copy2(H + "/grading.db", f"{H}/grading.db.bak_{ts}")
allc = json.load(open(H + "/cases.json"))
allc = [c for c in allc if c.get("block") != "erin150"] + cases
json.dump(allc, open(H + "/cases.json", "w")); os.chmod(H + "/cases.json", 0o600)
key.to_csv(H + "/erin150_key.csv", index=False); os.chmod(H + "/erin150_key.csv", 0o600)
db = sqlite3.connect(H + "/grading.db")
n_lab = db.execute("SELECT COUNT(*) FROM labels WHERE grader='rehan'").fetchone()[0]
old = db.execute("SELECT COUNT(*) FROM assignments WHERE grader='rehan'").fetchone()[0]
if n_lab == 0:
    db.execute("DELETE FROM assignments WHERE grader='rehan'")          # 40 unlabelled core+pool rows replaced
db.executemany("INSERT OR IGNORE INTO assignments VALUES (?,?,?)", [("rehan", c["case"], i) for i, c in enumerate(cases)])
db.commit()
now = db.execute("SELECT COUNT(*) FROM assignments WHERE grader='rehan'").fetchone()[0]
print(f"rehan: previous assignments {old} (labels {n_lab}) -> now {now}")
print("cases.json total:", len(allc), "| erin150:", len(cases), "| grade mix:", key.worst_grade.value_counts().to_dict(), "| GOJ share:", round((key.site == 'GOJ').mean(), 2))

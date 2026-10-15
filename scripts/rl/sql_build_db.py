"""Project 3: structured ERIN label database (no free text). Out: analysis/rl_sql/erin_labels.sqlite + SCHEMA.md.
IDs re-keyed to opaque integers (mapping file kept on the cluster only, chmod 600)."""
import json, os, sqlite3, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__)); from common import *
O = B + "/analysis/rl_sql"; os.makedirs(O, exist_ok=True); os.chmod(O, 0o700)
rep = pd.read_csv(ERIN, dtype=str, low_memory=False).fillna("").drop_duplicates("CaseName")
sec = pd.read_csv(B + "/phd/thesis/labeller/erin_sections_v3_consensus.csv", dtype=str).fillna("")
sl = pd.read_csv(B + "/phd/thesis/campaigns/allslides/erin_slides_all.csv", dtype=str).fillna("")
rng = np.random.RandomState(7)
pmap = {a: i + 1 for i, a in enumerate(rng.permutation(sorted(rep.anon_id.unique())))}
rmap = {c: i + 1 for i, c in enumerate(rng.permutation(sorted(rep.CaseName.unique())))}
rep["date"] = pd.to_datetime(rep.CollectedOrOrdered, dayfirst=True, errors="coerce").dt.strftime("%Y-%m-%d")
reports = pd.DataFrame({"report_id": rep.CaseName.map(rmap), "patient_id": rep.anon_id.map(pmap), "report_date": rep.date,
                        "specimen_protocol": rep.SpecimenProtocol.replace("", None), "age_at_report": pd.to_numeric(rep.AgeAtInvestigation, errors="coerce")})
patients = pd.DataFrame({"patient_id": sorted(pmap.values())})
def vf(r):
    try: return round(json.loads(r.grade_votes).get(r.worst_grade, 0) / int(r.n_jurors), 3)
    except Exception: return None
sec = sec[sec.CaseName.isin(rmap) & sec.worst_grade.isin(GRADES)]
sites = pd.DataFrame({"report_id": sec.CaseName.map(rmap), "site_letter": sec.section, "grade": sec.worst_grade, "jury_vote_fraction": sec.apply(vf, axis=1),
                      "organ_site": sec.site.replace("", None), "specimen_type": sec.specimen_type.replace("", None)})
sl = sl[sl.CaseName.isin(rmap)].reset_index(drop=True)
slides = pd.DataFrame({"slide_id": np.arange(1, len(sl) + 1), "report_id": sl.CaseName.map(rmap), "site_letter": sl.section.str.upper().str.strip().replace("", None),
                       "stain": np.where(sl.is_he == "True", "H&E", sl.stain_type.replace("", "OTHER").str.upper())})
db = O + "/erin_labels.sqlite"
if os.path.exists(db): os.remove(db)
con = sqlite3.connect(db)
con.executescript("""CREATE TABLE patients(patient_id INTEGER PRIMARY KEY);
CREATE TABLE reports(report_id INTEGER PRIMARY KEY, patient_id INTEGER REFERENCES patients, report_date TEXT, specimen_protocol TEXT, age_at_report REAL);
CREATE TABLE sites(report_id INTEGER REFERENCES reports, site_letter TEXT, grade TEXT, jury_vote_fraction REAL, organ_site TEXT, specimen_type TEXT, PRIMARY KEY(report_id, site_letter));
CREATE TABLE slides(slide_id INTEGER PRIMARY KEY, report_id INTEGER REFERENCES reports, site_letter TEXT, stain TEXT);""")
for name, df in (("patients", patients), ("reports", reports), ("sites", sites.drop_duplicates(["report_id", "site_letter"])), ("slides", slides)):
    df.to_sql(name, con, if_exists="append", index=False)
con.commit(); con.close(); os.chmod(db, 0o600)
pd.DataFrame({"anon_id": list(pmap), "patient_id": list(pmap.values())}).to_csv(O + "/id_map_patients_DO_NOT_SHARE.csv", index=False); os.chmod(O + "/id_map_patients_DO_NOT_SHARE.csv", 0o600)
schema = f"""# erin_labels.sqlite — schema (structured labels only; no report text; opaque integer IDs)

- **patients**(patient_id) — one row per ERIN patient ({len(patients)}). Sex and birth year are not in the source data.
- **reports**(report_id, patient_id, report_date 'YYYY-MM-DD', specimen_protocol, age_at_report) — one pathology report ({len(reports)});
  specimen_protocol = the lab's specimen category (e.g. OESOPHAGUS); age_at_report in years.
- **sites**(report_id, site_letter, grade, jury_vote_fraction, organ_site, specimen_type) — one lettered biopsy site/section of a report
  ({len(sites)}). grade = worst grade in that site: NORMAL_OTHER < NDBE < IND < LGD < HGD < CANCER (NDBE = non-dysplastic Barrett's,
  IND = indefinite for dysplasia, LGD/HGD = low/high-grade dysplasia). "HGD+" means HGD or CANCER; "dysplasia" means LGD, HGD or CANCER.
  jury_vote_fraction = share of the 10 LLM jurors agreeing with that grade (0–1). organ_site ∈ OESOPHAGUS, GOJ, STOMACH, DUODENUM,
  COLORECTAL, LYMPH_NODE, OTHER, UNSTATED. specimen_type ∈ BIOPSY, EMR_ESD, RESECTION, LYMPH_NODE, OTHER, UNSTATED.
- **slides**(slide_id, report_id, site_letter, stain) — one scanned slide ({len(slides)}); stain = 'H&E' or the special stain name.
Report-level ("case-level") grade = the worst site grade in the report.
"""
open(O + "/SCHEMA.md", "w").write(schema); print(schema)

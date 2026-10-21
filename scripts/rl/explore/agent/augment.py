"""Explore/agent, training-diversity step 1 (CPU): new parameter values for the 12 answerable TRAINING templates (same gold SQL
templates as sql_questions.py; answers executed on the read-only database). Test and validation templates are never touched.
Out: $O2/aug_base.jsonl (new base questions, split=train, variant=0, aug=1)."""
import itertools, json, os, sqlite3, sys
RL = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, RL)
from sql_common import O, DB
O2 = "/mnt/scratche/slow/fmlab/zuberi01/analysis/rl_explore/agent"; os.makedirs(O2, exist_ok=True)
con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
ORDSQL = "(CASE grade WHEN 'NORMAL_OTHER' THEN 0 WHEN 'NDBE' THEN 1 WHEN 'IND' THEN 2 WHEN 'LGD' THEN 3 WHEN 'HGD' THEN 4 WHEN 'CANCER' THEN 5 END)"
G = ["NDBE", "IND", "LGD", "HGD", "CANCER"]; G6 = ["NORMAL_OTHER"] + G
GN = {"NORMAL_OTHER": "normal/other", "NDBE": "non-dysplastic Barrett's", "IND": "indefinite for dysplasia", "LGD": "low-grade dysplasia", "HGD": "high-grade dysplasia", "CANCER": "cancer"}
ON = {"OESOPHAGUS": "oesophagus", "GOJ": "gastro-oesophageal junction", "STOMACH": "stomach", "DUODENUM": "duodenum", "COLORECTAL": "colon or rectum", "LYMPH_NODE": "lymph nodes", "OTHER": "other (unlisted) organ", "UNSTATED": "unstated organ"}
SPN = {"BIOPSY": "biopsy", "EMR_ESD": "endoscopic resection (EMR/ESD)", "RESECTION": "surgical resection", "LYMPH_NODE": "lymph node", "OTHER": "other-type"}
split = json.load(open(O + "/templates_agent.json"))["split"]
# (id, type, question template, sql template, NEW param grid only)
T = [
 ("goj_grade", "count", "How many gastro-oesophageal junction sites were graded {g}?", "SELECT COUNT(*) FROM sites WHERE organ_site='GOJ' AND grade='{G}'", [{"G": "NORMAL_OTHER"}]),
 ("worst_lgd_multi", "multi_report", "How many patients whose worst grade is low-grade dysplasia have at least {k} reports?",
  "WITH w AS (SELECT r.patient_id, MAX({ORD}) m FROM sites s JOIN reports r USING(report_id) GROUP BY r.patient_id), n AS (SELECT patient_id, COUNT(*) c FROM reports GROUP BY patient_id) "
  "SELECT COUNT(*) FROM w JOIN n USING(patient_id) WHERE w.m=3 AND n.c>={k}", [{"k": k} for k in range(10, 16)]),
 ("progression_lgd_hgd", "progression", "How many patients went from low-grade dysplasia to high-grade dysplasia or cancer within {n} years?",
  "WITH f AS (SELECT r.patient_id, MIN(r.report_date) d FROM sites s JOIN reports r USING(report_id) WHERE s.grade='LGD' GROUP BY r.patient_id), "
  "h AS (SELECT r.patient_id, r.report_date d FROM sites s JOIN reports r USING(report_id) WHERE s.grade IN ('HGD','CANCER')) "
  "SELECT COUNT(DISTINCT f.patient_id) FROM f JOIN h ON f.patient_id=h.patient_id WHERE h.d>f.d AND julianday(h.d)-julianday(f.d)<=365.25*{n}", [{"n": n} for n in range(9, 13)]),
 ("confident_grade", "aggregate", "How many sites graded {g} have a jury vote fraction of at least {t}?", "SELECT COUNT(*) FROM sites WHERE grade='{G}' AND jury_vote_fraction>={t}", [{"G": g, "t": t} for g in G for t in (0.55, 0.65, 0.75, 0.85, 0.95)]),
 ("patients_min_reports", "multi_report", "How many patients have at least {k} reports?", "SELECT COUNT(*) FROM (SELECT patient_id FROM reports GROUP BY patient_id HAVING COUNT(*)>={k})", [{"k": k} for k in range(16, 26)]),
 ("reports_many_sites", "count", "How many reports have more than {k} biopsy sites?", "SELECT COUNT(*) FROM (SELECT report_id FROM sites GROUP BY report_id HAVING COUNT(*)>{k})", [{"k": k} for k in range(9, 15)]),
 ("spec_type_grade", "count", "How many {spec} specimens were graded {g}?", "SELECT COUNT(*) FROM sites WHERE specimen_type='{SP}' AND grade='{G}'", [{"SP": sp, "G": g} for sp in ("LYMPH_NODE", "OTHER") for g in G] + [{"SP": sp, "G": "NORMAL_OTHER"} for sp in ("BIOPSY", "EMR_ESD", "RESECTION")]),
 ("grade_dist_organ", "distribution", "What is the distribution of site grades in the {organ}?", "SELECT grade, COUNT(*) FROM sites WHERE organ_site='{O}' GROUP BY grade ORDER BY grade", [{"O": o} for o in ("OTHER", "UNSTATED")]),
]
rows = []
for tid, typ, qt, st, grid in T:
    assert split[tid] == "train", tid
    for p in grid:
        q = dict(p)
        if "G" in p: q.update(g=GN[p["G"]], o=G6.index(p["G"]))
        if "O" in p: q.update(organ=ON[p["O"]])
        if "SP" in p: q.update(spec=SPN[p["SP"]])
        sql = st.replace("{ORD}", ORDSQL).format(**q); text = qt.format(**q); ans = [list(r) for r in con.execute(sql).fetchall()]
        rows.append({"template": tid, "type": typ, "answerable": True, "question": text, "gold_sql": sql, "answer": ans, "split": "train", "variant": 0, "aug": 1})
with open(O2 + "/aug_base.jsonl", "w") as f:
    for r in rows: f.write(json.dumps(r) + "\n")
os.chmod(O2 + "/aug_base.jsonl", 0o600)
from collections import Counter; print(len(rows), "new base training questions", dict(Counter(r["template"] for r in rows)))

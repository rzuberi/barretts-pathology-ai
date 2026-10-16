"""Project 3: question templates with gold SQL -> instantiated questions with executed gold answers, split BY TEMPLATE.
Out: analysis/rl_sql/questions.jsonl, templates.json. Paraphrases are added later by sql_paraphrase.py (GPU, local LLM)."""
import json, os, random, sqlite3
O = "/mnt/scratche/slow/fmlab/zuberi01/analysis/rl_sql"; con = sqlite3.connect(O + "/erin_labels.sqlite")
ORDSQL = "(CASE grade WHEN 'NORMAL_OTHER' THEN 0 WHEN 'NDBE' THEN 1 WHEN 'IND' THEN 2 WHEN 'LGD' THEN 3 WHEN 'HGD' THEN 4 WHEN 'CANCER' THEN 5 END)"
G = ["NDBE", "IND", "LGD", "HGD", "CANCER"]; YRS = [str(y) for y in range(2014, 2026)]; ORG = ["OESOPHAGUS", "GOJ", "STOMACH", "DUODENUM", "COLORECTAL", "LYMPH_NODE"]
import itertools
GN = {"NORMAL_OTHER": "normal/other", "NDBE": "non-dysplastic Barrett's", "IND": "indefinite for dysplasia", "LGD": "low-grade dysplasia", "HGD": "high-grade dysplasia", "CANCER": "cancer"}
# (id, type, question template, sql template, param grid)
T = [
 ("count_patients", "count", "How many patients are in the database?", "SELECT COUNT(*) FROM patients", [{}]),
 ("reports_in_year", "per_year", "How many pathology reports were dated in {Y}?", "SELECT COUNT(*) FROM reports WHERE substr(report_date,1,4)='{Y}'", [{"Y": y} for y in YRS]),
 ("reports_per_year", "per_year", "How many reports are there per year?", "SELECT substr(report_date,1,4) AS yr, COUNT(*) FROM reports WHERE report_date IS NOT NULL GROUP BY yr ORDER BY yr", [{}]),
 ("sites_with_grade", "count", "How many biopsy sites were graded {g}?", "SELECT COUNT(*) FROM sites WHERE grade='{G}'", [{"G": g} for g in G]),
 ("patients_any_grade", "count", "How many patients had at least one site with {g}?", "SELECT COUNT(DISTINCT r.patient_id) FROM sites s JOIN reports r USING(report_id) WHERE s.grade='{G}'", [{"G": g} for g in G]),
 ("patients_worst_grade", "worst_grade", "How many patients have {g} as their worst grade ever recorded?",
  "SELECT COUNT(*) FROM (SELECT r.patient_id, MAX({ORD}) m FROM sites s JOIN reports r USING(report_id) GROUP BY r.patient_id) WHERE m={o}", [{"G": g} for g in G]),
 ("grade_dist_organ", "distribution", "What is the distribution of site grades in the {organ}?", "SELECT grade, COUNT(*) FROM sites WHERE organ_site='{O}' GROUP BY grade ORDER BY grade", [{"O": o} for o in ORG]),
 ("patients_min_reports", "multi_report", "How many patients have at least {k} reports?", "SELECT COUNT(*) FROM (SELECT patient_id FROM reports GROUP BY patient_id HAVING COUNT(*)>={k})", [{"k": k} for k in range(2, 16)]),
 ("progression_lgd_hgd", "progression", "How many patients went from low-grade dysplasia to high-grade dysplasia or cancer within {n} years?",
  "WITH f AS (SELECT r.patient_id, MIN(r.report_date) d FROM sites s JOIN reports r USING(report_id) WHERE s.grade='LGD' GROUP BY r.patient_id), "
  "h AS (SELECT r.patient_id, r.report_date d FROM sites s JOIN reports r USING(report_id) WHERE s.grade IN ('HGD','CANCER')) "
  "SELECT COUNT(DISTINCT f.patient_id) FROM f JOIN h ON f.patient_id=h.patient_id WHERE h.d>f.d AND julianday(h.d)-julianday(f.d)<=365.25*{n}", [{"n": n} for n in range(1, 9)]),
 ("ndbe_to_dysplasia", "progression", "How many patients whose first report showed non-dysplastic Barrett's later had dysplasia (LGD or worse)?",
  "WITH c AS (SELECT r.patient_id, r.report_date d, MAX({ORD}) m FROM sites s JOIN reports r USING(report_id) GROUP BY r.report_id), "
  "f AS (SELECT patient_id, MIN(d) d FROM c GROUP BY patient_id) SELECT COUNT(DISTINCT f.patient_id) FROM f JOIN c c1 ON c1.patient_id=f.patient_id AND c1.d=f.d AND c1.m=1 "
  "JOIN c c2 ON c2.patient_id=f.patient_id AND c2.d>f.d AND c2.m>=3", [{}]),
 ("time_to_hgd", "time_to_event", "Among patients who ever had high-grade dysplasia or cancer, what is the average number of days from their first report to the first such finding?",
  "WITH f AS (SELECT patient_id, MIN(report_date) d0 FROM reports GROUP BY patient_id), h AS (SELECT r.patient_id, MIN(r.report_date) d1 FROM sites s JOIN reports r USING(report_id) WHERE s.grade IN ('HGD','CANCER') GROUP BY r.patient_id) "
  "SELECT ROUND(AVG(julianday(h.d1)-julianday(f.d0)),1) FROM f JOIN h USING(patient_id)", [{}]),
 ("slides_stain", "count", "How many slides were stained with {S}?", "SELECT COUNT(*) FROM slides WHERE stain='{S}'", [{"S": "H&E"}]),
 ("avg_slides_per_report", "aggregate", "What is the average number of slides per report, among reports with at least one slide?", "SELECT ROUND(AVG(n),2) FROM (SELECT report_id, COUNT(*) n FROM slides GROUP BY report_id)", [{}]),
 ("reports_many_sites", "count", "How many reports have more than {k} biopsy sites?", "SELECT COUNT(*) FROM (SELECT report_id FROM sites GROUP BY report_id HAVING COUNT(*)>{k})", [{"k": k} for k in range(1, 9)]),
 ("low_confidence_sites", "aggregate", "What fraction of sites have a jury vote fraction below {t}?", "SELECT ROUND(AVG(jury_vote_fraction<{t}),3) FROM sites", [{"t": round(t, 2)} for t in [0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]]),
 ("case_level_grade", "case_level", "How many reports have a case-level (worst site) grade of {g}?", "SELECT COUNT(*) FROM (SELECT report_id, MAX({ORD}) m FROM sites GROUP BY report_id) WHERE m={o}", [{"G": g} for g in G]),
 ("spec_type_grade", "count", "How many {spec} specimens were graded {g}?", "SELECT COUNT(*) FROM sites WHERE specimen_type='{SP}' AND grade='{G}'", [{"SP": sp, "G": g} for sp in ("BIOPSY", "EMR_ESD", "RESECTION") for g in G]),
 ("older_hgd", "count", "How many patients were over {a} at a report showing high-grade dysplasia or cancer?",
  "SELECT COUNT(DISTINCT r.patient_id) FROM sites s JOIN reports r USING(report_id) WHERE s.grade IN ('HGD','CANCER') AND r.age_at_report>{a}", [{"a": a} for a in range(40, 90, 5)]),
 ("avg_age_case_grade", "aggregate", "What is the average age at report for reports whose worst grade is {g}?",
  "SELECT ROUND(AVG(r.age_at_report),1) FROM reports r JOIN (SELECT report_id, MAX({ORD}) m FROM sites GROUP BY report_id) c USING(report_id) WHERE c.m={o}", [{"G": g} for g in G]),
 ("year_most_grade", "per_year", "In which year were the most sites graded {g}?",
  "SELECT substr(r.report_date,1,4) yr FROM sites s JOIN reports r USING(report_id) WHERE s.grade='{G}' AND r.report_date IS NOT NULL GROUP BY yr ORDER BY COUNT(*) DESC, yr LIMIT 1", [{"G": g} for g in G]),
 ("dysplasia_reports_year", "per_year", "How many reports from {Y} contained any dysplasia (LGD or worse)?",
  "SELECT COUNT(DISTINCT s.report_id) FROM sites s JOIN reports r USING(report_id) WHERE s.grade IN ('LGD','HGD','CANCER') AND substr(r.report_date,1,4)='{Y}'", [{"Y": y} for y in YRS]),
 ("patients_two_years", "multi_report", "How many patients had reports in both {Y} and {Y2}?",
  "SELECT COUNT(*) FROM (SELECT patient_id FROM reports WHERE substr(report_date,1,4) IN ('{Y}','{Y2}') GROUP BY patient_id HAVING COUNT(DISTINCT substr(report_date,1,4))=2)", [{"Y": a, "Y2": b} for a, b in itertools.combinations(YRS, 2)]),
 ("goj_grade", "count", "How many gastro-oesophageal junction sites were graded {g}?", "SELECT COUNT(*) FROM sites WHERE organ_site='GOJ' AND grade='{G}'", [{"G": g} for g in G]),
 ("patients_emr", "count", "How many patients had an endoscopic resection (EMR/ESD) specimen?", "SELECT COUNT(DISTINCT r.patient_id) FROM sites s JOIN reports r USING(report_id) WHERE s.specimen_type='EMR_ESD'", [{}]),
 ("reports_no_slides", "count", "How many reports have no scanned slides?", "SELECT COUNT(*) FROM reports WHERE report_id NOT IN (SELECT DISTINCT report_id FROM slides)", [{}]),
 ("worst_lgd_multi", "multi_report", "How many patients whose worst grade is low-grade dysplasia have at least {k} reports?",
  "WITH w AS (SELECT r.patient_id, MAX({ORD}) m FROM sites s JOIN reports r USING(report_id) GROUP BY r.patient_id), n AS (SELECT patient_id, COUNT(*) c FROM reports GROUP BY patient_id) "
  "SELECT COUNT(*) FROM w JOIN n USING(patient_id) WHERE w.m=3 AND n.c>={k}", [{"k": k} for k in range(2, 10)]),
 ("reports_half_year", "per_year", "How many reports were dated in the first half (January to June) of {Y}?", "SELECT COUNT(*) FROM reports WHERE report_date BETWEEN '{Y}-01-01' AND '{Y}-06-30'", [{"Y": y} for y in YRS]),
 ("ind_followed_by_dysplasia", "progression", "How many patients with an indefinite-for-dysplasia site later had LGD or worse on a subsequent report?",
  "WITH i AS (SELECT r.patient_id, MIN(r.report_date) d FROM sites s JOIN reports r USING(report_id) WHERE s.grade='IND' GROUP BY r.patient_id) "
  "SELECT COUNT(DISTINCT i.patient_id) FROM i JOIN reports r ON r.patient_id=i.patient_id AND r.report_date>i.d JOIN sites s ON s.report_id=r.report_id AND s.grade IN ('LGD','HGD','CANCER')", [{}]),
 ("prop_patients_cancer", "aggregate", "What proportion of patients ever had a cancer site?", "SELECT ROUND(1.0*(SELECT COUNT(DISTINCT r.patient_id) FROM sites s JOIN reports r USING(report_id) WHERE s.grade='CANCER')/(SELECT COUNT(*) FROM patients),3)", [{}]),
 ("site_vs_case_lgd", "case_level", "How many sites are graded LGD, and how many reports contain at least one LGD site?",
  "SELECT (SELECT COUNT(*) FROM sites WHERE grade='LGD'), (SELECT COUNT(DISTINCT report_id) FROM sites WHERE grade='LGD')", [{}]),
 ("patients_per_organ", "count", "How many patients had at least one {organ} site?", "SELECT COUNT(DISTINCT r.patient_id) FROM sites s JOIN reports r USING(report_id) WHERE s.organ_site='{O}'", [{"O": o} for o in ORG]),
 ("grade_in_year", "per_year", "How many sites graded {g} were reported in {Y}?", "SELECT COUNT(*) FROM sites s JOIN reports r USING(report_id) WHERE s.grade='{G}' AND substr(r.report_date,1,4)='{Y}'", [{"G": g, "Y": y} for g in G for y in YRS]),
 ("grade_in_organ", "count", "How many {organ} sites were graded {g}?", "SELECT COUNT(*) FROM sites WHERE organ_site='{O}' AND grade='{G}'", [{"O": o, "G": g} for o in ORG for g in G]),
 ("patients_grade_year", "per_year", "How many patients had a {g} site on a report from {Y}?", "SELECT COUNT(DISTINCT r.patient_id) FROM sites s JOIN reports r USING(report_id) WHERE s.grade='{G}' AND substr(r.report_date,1,4)='{Y}'", [{"G": g, "Y": y} for g in G for y in YRS]),
 ("confident_grade", "aggregate", "How many sites graded {g} have a jury vote fraction of at least {t}?", "SELECT COUNT(*) FROM sites WHERE grade='{G}' AND jury_vote_fraction>={t}", [{"G": g, "t": t} for g in G for t in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)]),
 ("first_report_year", "per_year", "How many patients had their first report in {Y}?", "SELECT COUNT(*) FROM (SELECT patient_id, MIN(report_date) d FROM reports GROUP BY patient_id) WHERE substr(d,1,4)='{Y}'", [{"Y": y} for y in YRS]),
]
UNANS = [  # (id, question) gold = CANNOT_ANSWER
 ("u_sex", "How many female patients had high-grade dysplasia?"), ("u_ethnicity", "What is the ethnic breakdown of patients with Barrett's?"),
 ("u_smoking", "How many smokers progressed to cancer?"), ("u_treatment", "How many patients received radiofrequency ablation after LGD?"),
 ("u_recent", "How many patients had dysplasia recently?"), ("u_length", "What is the average Barrett's segment length?"),
 ("u_tumour_size", "What was the mean tumour size in the cancer resections?"), ("u_survival", "How many cancer patients died within five years?"),
 ("u_hospital", "Which hospital sent the most biopsies?"), ("u_pathologist", "Which pathologist reported the most high-grade dysplasia?"),
 ("u_bmi", "Do patients with higher BMI have more dysplasia?"), ("u_p53", "How many sites were p53-positive?"),
]
NAME = {"G": None}
qs = []
for tid, typ, qt, st, grid in T:
    for p in grid:
        q = dict(p); 
        if "G" in p: q.update(g=GN[p["G"]], o={"NDBE": 1, "IND": 2, "LGD": 3, "HGD": 4, "CANCER": 5}[p["G"]])
        if "O" in p: q.update(organ={"OESOPHAGUS": "oesophagus", "GOJ": "gastro-oesophageal junction", "STOMACH": "stomach", "DUODENUM": "duodenum", "COLORECTAL": "colon or rectum", "LYMPH_NODE": "lymph nodes"}[p["O"]])
        if "SP" in p: q.update(spec={"BIOPSY": "biopsy", "EMR_ESD": "endoscopic resection (EMR/ESD)", "RESECTION": "surgical resection"}[p["SP"]])
        sql = st.replace("{ORD}", ORDSQL).format(**q); text = qt.format(**q)
        try: ans = [list(r) for r in con.execute(sql).fetchall()]
        except Exception as e: raise SystemExit(f"gold SQL failed for {tid}: {e}\n{sql}")
        qs.append({"template": tid, "type": typ, "answerable": True, "question": text, "gold_sql": sql, "answer": ans})
for tid, text in UNANS: qs.append({"template": tid, "type": "unanswerable", "answerable": False, "question": text, "gold_sql": None, "answer": "CANNOT_ANSWER"})
# more unanswerable items: missing fields crossed with grades, and vague time windows (each family = one template for the split)
MISSING = {"u_sex_grade": "How many {sexw} patients had at least one {g} site?", "u_smoking_grade": "How many smokers had a {g} site?",
           "u_treat_grade": "How many patients with {g} were treated with ablation?", "u_length_grade": "What is the average Barrett's segment length in patients with {g}?",
           "u_death_grade": "How many patients with {g} have died?", "u_hospital_grade": "Which hospital reported the most {g} sites?",
           "u_p53_grade": "How many {g} sites were p53 abnormal?", "u_bmi_grade": "What is the mean BMI of patients with {g}?"}
for tid, qt in MISSING.items():
    for g in G:
        for sexw in (["female", "male"] if "{sexw}" in qt else [""]):
            qs.append({"template": tid, "type": "unanswerable", "answerable": False, "question": qt.format(g=GN[g], sexw=sexw), "gold_sql": None, "answer": "CANNOT_ANSWER"})
VAGUE = {"u_recent_grade": "How many patients had {g} recently?", "u_lately_reports": "How many {g} reports have there been lately?",
         "u_soon_progress": "How many patients with {g} progressed quickly?", "u_most_patients": "Which patients are most at risk of {g}?"}
for tid, qt in VAGUE.items():
    for g in G: qs.append({"template": tid, "type": "unanswerable", "answerable": False, "question": qt.format(g=GN[g]), "gold_sql": None, "answer": "CANNOT_ANSWER"})
tids = sorted({q["template"] for q in qs if q["answerable"]}); uids = sorted({q["template"] for q in qs if not q["answerable"]})
rnd = random.Random(20261015); rnd.shuffle(tids); rnd.shuffle(uids)
def split(ids):
    n = len(ids); return {t: ("train" if i < 0.6 * n else "val" if i < 0.75 * n else "test") for i, t in enumerate(ids)}
sp = {**split(tids), **split(uids)}
for q in qs: q["split"] = sp[q["template"]]
json.dump({"templates": [t[0] for t in T], "unanswerable": sorted(uids), "split": sp}, open(O + "/templates.json", "w"), indent=1)
with open(O + "/questions_base.jsonl", "w") as f:
    for q in qs: f.write(json.dumps(q) + "\n")
from collections import Counter
print(len(qs), "base questions;", Counter(q["split"] for q in qs), "| templates", Counter(sp.values()), "| unanswerable", len(UNANS))
print("example:", qs[8]["question"], "->", qs[8]["answer"])

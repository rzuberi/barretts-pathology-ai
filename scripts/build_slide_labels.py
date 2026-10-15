"""2.38 step 1: build SLIDE-LEVEL ERIN labels from the per-section jury.

Majority vote per (report, section): a grade is accepted when >=3/5 jurors list
it; cancer subtypes likewise. Sections are joined to individual slides through
Shiv Sakthivel's matched_image_pathology.csv (slide <-> report Section), then
to our feature h5s via slide uuid. Outputs labeller/erin_slide_labels_v2.csv
plus join-rate diagnostics (no silent losses).
"""
import glob, json, os, re
from collections import Counter, defaultdict
import pandas as pd

T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"
SHIV = "/mnt/scratche/slow/fmlab/sakthi01/erin/data/matched_image_pathology.csv"
OUT = os.environ.get("OUTDIR", ".")
GRADE_ORD = {"NORMAL_OTHER": -1, "NDBE": 0, "IND": 1, "LGD": 2, "HGD": 3, "CANCER": 4}

# ---- per-section majority across jurors ----
juror_secs = defaultdict(dict)   # case -> juror -> {section: (grades, subtypes)}
for f in glob.glob(T + "/feasibility/runs/sections_*/output/sections_*.csv"):
    juror = f.split("/output/sections_")[1].rsplit("_shard", 1)[0]
    d = pd.read_csv(f, dtype=str).fillna("")
    for _, r in d.iterrows():
        if r["sections_json"] == "PARSE_FAIL": continue
        try:
            secs = json.loads(r["sections_json"])
        except Exception:
            continue
        m = {}
        for s in secs:
            key = str(s.get("section", "?")).upper().strip()
            key = "WHOLE" if key in ("WHOLE", "WHOLE_REPORT", "") else key[:2].strip()
            m.setdefault(key, (set(), set()))
            m[key][0].update(s.get("grades", []))
            m[key][1].update(s.get("cancer_subtypes", []))
        juror_secs[r["CaseName"]][juror] = m

rows = []
for case, jm in juror_secs.items():
    if len(jm) < 3: continue
    all_secs = Counter()
    for m in jm.values(): all_secs.update(m.keys())
    n_jurors = len(jm)
    for sec, seen in all_secs.items():
        if seen < max(3, n_jurors - 2): continue   # section itself must be seen by majority
        gv, sv = Counter(), Counter()
        for m in jm.values():
            if sec in m:
                gv.update(m[sec][0]); sv.update(m[sec][1])
        grades = sorted(g for g, c in gv.items() if c >= 3 and g in GRADE_ORD)
        subs = sorted(s for s, c in sv.items() if c >= 3)
        if not grades: continue
        worst = max(grades, key=lambda g: GRADE_ORD[g])
        rows.append({"CaseName": case, "section": sec, "grades": "|".join(grades),
                     "worst_grade": worst, "cancer_subtypes": "|".join(subs),
                     "n_jurors_seen": int(seen)})
sec_df = pd.DataFrame(rows)
print(f"section-level consensus rows: {len(sec_df)} over {sec_df['CaseName'].nunique()} reports", flush=True)

# ---- join sections to slides via Shiv's table ----
shiv = pd.read_csv(SHIV, dtype=str, low_memory=False).fillna("")
shiv["sec_norm"] = shiv["Section"].astype(str).str.upper().str.strip().str[:2].str.strip()
shiv["uuid"] = shiv["Filepath"].map(lambda p: next(
    (t for t in re.findall(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", str(p))), ""))
m = pd.read_csv(T + "/labeller/erin_master.csv", dtype=str).dropna(subset=["h5"]).drop_duplicates("h5")
m["uuid_h5"] = m["h5"].map(lambda p: os.path.basename(str(p)).replace(".h5", ""))
uuid2h5 = dict(zip(m["uuid_h5"], m["h5"]))
joined = shiv.merge(sec_df, left_on=["CaseName", "sec_norm"], right_on=["CaseName", "section"])
joined["h5"] = joined["uuid"].map(uuid2h5)
slide = joined[joined["h5"].notna()].drop_duplicates(["h5"])[
    ["h5", "CaseName", "section", "grades", "worst_grade", "cancer_subtypes"]]
# case-max = worst grade across ALL consensus sections of the report,
# computed BEFORE saving so the training contrast can read it
case_worst = sec_df.groupby("CaseName")["worst_grade"].agg(
    lambda g: max(g, key=lambda x: GRADE_ORD[x]))
slide["case_max"] = slide["CaseName"].map(case_worst)
slide.to_csv(T + "/labeller/erin_slide_labels_v2.csv", index=False)
diff = (slide["worst_grade"] != slide["case_max"]).mean()
res = {"section_rows": len(sec_df), "reports_with_consensus": int(sec_df["CaseName"].nunique()),
       "shiv_rows": len(shiv), "joined_rows": int(len(joined)),
       "slides_labelled": int(len(slide)),
       "slides_where_section_differs_from_case_max": round(float(diff), 4),
       "grade_dist": slide["worst_grade"].value_counts().to_dict(),
       "subtype_any": int((slide["cancer_subtypes"] != "").sum())}
json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=2)
print(json.dumps(res, indent=2))

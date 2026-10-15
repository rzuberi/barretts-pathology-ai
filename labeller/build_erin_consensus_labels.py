"""EXECUTION_PLAN 2.10 + 2.12: consensus soft labels and rebuilt progression cohort.

Per Rehan's decision (2026-08-19): where pathladder and the feasibility grader agree,
the label is confident; where they disagree, the case gets the midpoint grade with an
`uncertain` flag (e.g. one says CANCER(4), other NDBE(0) -> 2.0, uncertain). Rehan's
adjudications (see make_adjudication_pack.py) override both graders where present.

Also rebuilds the longitudinal progression cohort from consensus grades:
per patient, index = first timepoint with grade <= LGD; event = any later timepoint
with confident grade >= HGD; time = days index -> event/last follow-up.
"""
import json, os, re, sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pathladder import label_report, BARRETTS_LADDER
from run_erin_audit_v2 import feas_sev  # the verbatim feasibility grader

ERIN = "/mnt/scratche/fast/fmlab/datasets/imaging/ERIN/data/PathologyReport_AnonIds.csv"
ADJ = os.path.join(HERE, "adjudications.csv")  # CaseName,decision (yes=carcinoma current)
OUT = os.environ.get("OUTDIR", HERE)
NUM = {"NDBE": 0, "IND": 1, "LGD": 2, "HGD": 3, "CANCER": 4}
TEXT = ["FinalDiagnosis_redacted", "MicroscopicDescription_redacted",
        "Addendum1_redacted", "Addendum2_redacted", "Addendum3_redacted"]

rep = pd.read_csv(ERIN, dtype=str, low_memory=False).fillna("")
rep["_text"] = rep[TEXT].agg(" ".join, axis=1)
rep["plad"] = [label_report(t, BARRETTS_LADDER)["grade"] for t in rep["_text"]]
rep["feas"] = rep["_text"].map(feas_sev)

def consensus(row):
    a, b = row["plad"], row["feas"]
    if a is None and b is None: return None, None, "unlabelled"
    if a is None or b is None:
        g = a or b; return float(NUM[g]), g, "single_grader"
    if a == b: return float(NUM[a]), a, "confident"
    mid = (NUM[a] + NUM[b]) / 2
    return mid, None, "uncertain"

rep[["grade_num", "grade_label", "confidence"]] = rep.apply(
    consensus, axis=1, result_type="expand")

if os.path.exists(ADJ):
    adj = pd.read_csv(ADJ, dtype=str)
    adj_map = dict(zip(adj["CaseName"], adj["decision"].str.lower()))
    hit = rep["CaseName"].map(adj_map)
    rep.loc[hit == "yes", ["grade_num", "grade_label", "confidence"]] = [4.0, "CANCER", "adjudicated"]
    # 'no' -> fall back to the non-CANCER grader's call
    mask = hit == "no"
    fallback = rep.loc[mask].apply(
        lambda r: r["plad"] if r["plad"] not in (None, "CANCER") else
                  (r["feas"] if r["feas"] not in (None, "CANCER") else "NDBE"), axis=1)
    rep.loc[mask, "grade_label"] = fallback
    rep.loc[mask, "grade_num"] = fallback.map(NUM).astype(float)
    rep.loc[mask, "confidence"] = "adjudicated"
    print(f"adjudications applied: {int((hit.isin(['yes','no'])).sum())}")
else:
    print("no adjudications.csv yet — consensus only (rerun after Rehan's pass)")

print(rep["confidence"].value_counts(dropna=False).to_dict())
rep[["anon_id", "CaseName", "CollectedOrOrdered", "plad", "feas",
     "grade_num", "grade_label", "confidence"]].to_csv(
    os.path.join(OUT, "erin_consensus_labels.csv"), index=False)

# --- progression rebuild (2.12) ---
lab = rep.copy()
lab["date"] = pd.to_datetime(lab["CollectedOrOrdered"], errors="coerce")
lab = lab.dropna(subset=["date"])
rows = []
for pid, g in lab.sort_values("date").groupby("anon_id"):
    g = g[g["grade_num"].notna()]
    if len(g) < 2: continue
    idx = g[g["grade_num"] <= 2.0]
    if idx.empty: continue
    t0 = idx.iloc[0]["date"]
    fut = g[g["date"] > t0]
    if fut.empty: continue
    ev = fut[(fut["grade_num"] >= 3.0) & (fut["confidence"].isin(["confident", "adjudicated"]))]
    rows.append({"anon_id": pid, "index_date": t0.date(),
                 "index_grade": idx.iloc[0]["grade_label"] or idx.iloc[0]["grade_num"],
                 "progressed_to_HGDplus": bool(len(ev)),
                 "tte_days": int(((ev.iloc[0]["date"] if len(ev) else fut.iloc[-1]["date"]) - t0).days),
                 "n_future_reports": int(len(fut))})
coh = pd.DataFrame(rows)
coh.to_csv(os.path.join(OUT, "erin_progression_cohort_v2.csv"), index=False)
summary = {"reports": int(len(rep)),
           "confidence_dist": {str(k): int(v) for k, v in rep["confidence"].value_counts(dropna=False).items()},
           "progression_cohort_n": int(len(coh)),
           "progressors": int(coh["progressed_to_HGDplus"].sum()) if len(coh) else 0}
json.dump(summary, open(os.path.join(OUT, "consensus_summary.json"), "w"), indent=2)
print(summary)

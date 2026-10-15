"""Apply pathladder's barretts_ladder to ERIN reports; validate vs manual grades.

CLUSTER-SIDE script (paths below). Run after the maintenance window:
    conda activate pathology && python run_erin_validation.py

Truth: the report-derived manual grading from the ERIN feasibility work
(2,446 graded patients). This validates pathladder against the hand-built
extraction it replaces, then against any expert-graded subset if available.
"""
import json, os, sys
import pandas as pd

ERIN_REPORTS = "/mnt/scratche/fast/fmlab/datasets/imaging/ERIN/data/PathologyReport_AnonIds.csv"
FEAS_LABELS = "/mnt/scratche/slow/fmlab/zuberi01/phd/erin_multimodal_feasibility/trainable_progression_cohort.csv"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, OUT_DIR)
from pathladder import label_frame, BARRETTS_LADDER

rep = pd.read_csv(ERIN_REPORTS, dtype=str, low_memory=False)
print(f"ERIN reports: {len(rep)}")
text_cols = [c for c in rep.columns if any(k in c.lower() for k in
             ("finaldiagnosis", "final_diagnosis", "microscopic", "diagnosis"))]
print("text columns found:", text_cols)
rep["_text"] = rep[text_cols].fillna("").agg(" ".join, axis=1)

labels = label_frame(rep, "_text", BARRETTS_LADDER)
rep["pathladder_grade"] = labels["grade"]
dist = rep["pathladder_grade"].value_counts(dropna=False).to_dict()
print("grade distribution:", dist)

res = {"n_reports": int(len(rep)), "distribution": {str(k): int(v) for k, v in dist.items()}}

if os.path.exists(FEAS_LABELS):
    feas = pd.read_csv(FEAS_LABELS, dtype=str)
    print("feasibility label columns:", list(feas.columns))
    key = next((c for c in ("CaseName", "case_id", "accession") if c in feas.columns and c in rep.columns), None)
    gcol = next((c for c in feas.columns if "grade" in c.lower() or "label" in c.lower()), None)
    if key and gcol:
        m = rep[[key, "pathladder_grade"]].merge(feas[[key, gcol]], on=key).dropna()
        agree = float((m["pathladder_grade"] == m[gcol]).mean())
        res["vs_feasibility_labels"] = {"n": int(len(m)), "agreement": round(agree, 4),
                                        "confusion": pd.crosstab(m["pathladder_grade"], m[gcol]).to_dict()}
        print(f"agreement vs feasibility labels: {agree:.3f} on {len(m)}")
    else:
        res["vs_feasibility_labels"] = f"join not automatic; key={key} gcol={gcol} — inspect manually"

rep[["anon_id", "CaseName", "pathladder_grade"]].to_csv(
    os.path.join(OUT_DIR, "erin_pathladder_labels.csv"), index=False)
json.dump(res, open(os.path.join(OUT_DIR, "erin_validation_results.json"), "w"), indent=2)
print("wrote erin_pathladder_labels.csv + erin_validation_results.json")

"""EXECUTION_PLAN 2.11: build Rehan's adjudication pack.

Produces adjudication_pack.md: numbered disagreement cases (both directions of the
CANCER boundary), each with the report text and one question:
    "Is carcinoma a CURRENT diagnosis in this report? yes / no / unsure"
Rehan reads the pack, records himself saying e.g. "case 12 yes", transcribes, and
feeds the transcript to parse_adjudication.py (accepts free-ish text).
"""
import os, sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pathladder import label_report, BARRETTS_LADDER
from run_erin_audit_v2 import feas_sev

ERIN = "/mnt/scratche/fast/fmlab/datasets/imaging/ERIN/data/PathologyReport_AnonIds.csv"
OUT = os.environ.get("OUTDIR", HERE)
N_PER_SIDE = 40
TEXT = ["FinalDiagnosis_redacted", "MicroscopicDescription_redacted",
        "Addendum1_redacted", "Addendum2_redacted", "Addendum3_redacted"]

rep = pd.read_csv(ERIN, dtype=str, low_memory=False).fillna("")
rep["_text"] = rep[TEXT].agg(" ".join, axis=1)
rep["plad"] = [label_report(t, BARRETTS_LADDER)["grade"] for t in rep["_text"]]
rep["feas"] = rep["_text"].map(feas_sev)

a = rep[(rep["plad"] == "CANCER") & (rep["feas"] != "CANCER")].head(N_PER_SIDE)
b = rep[(rep["feas"] == "CANCER") & (rep["plad"] != "CANCER")].sample(
    min(N_PER_SIDE, (rep["feas"] == "CANCER").sum()), random_state=0)
b = b[b["plad"] != "CANCER"].head(N_PER_SIDE)
pack = pd.concat([a.assign(side="pathladder_only"), b.assign(side="feasibility_only")])
pack = pack.reset_index(drop=True)
pack.index += 1

lines = ["# ERIN adjudication pack",
         "",
         "One question per case: **is carcinoma a CURRENT diagnosis in this report?**",
         "Answer per case: `yes` / `no` / `unsure`. Say it as: *\"case 12 yes\"*.",
         f"{len(pack)} cases: 1-{len(a)} pathladder-only CANCER, "
         f"{len(a)+1}-{len(pack)} feasibility-only CANCER.", ""]
for i, r in pack.iterrows():
    lines += [f"---\n\n## Case {i}",
              f"*graders: pathladder={r['plad']}, feasibility={r['feas']}*", "",
              r["_text"].strip()[:2500], ""]
open(os.path.join(OUT, "adjudication_pack.md"), "w").write("\n".join(lines))
pack[["CaseName", "side", "plad", "feas"]].to_csv(
    os.path.join(OUT, "adjudication_index.csv"), index=True, index_label="case_no")
print(f"wrote adjudication_pack.md ({len(pack)} cases) + adjudication_index.csv")

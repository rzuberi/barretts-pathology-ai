"""EXECUTION_PLAN 1.3 (corrected): per-REPORT pathladder vs the feasibility grader.

The first audit compared pathladder's max-over-timepoints grade to the feasibility
cohort's index-timepoint severity — mis-specified (progression makes them differ by
design). This compares both graders on the SAME report text, which is the apples-to-
apples validation, and isolates the two design differences:
  (a) addenda included in the text or not,
  (b) first-match cascade (feasibility) vs sentence-negation + worst-wins (pathladder).
"""
import json, os, re, sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pathladder import label_report, BARRETTS_LADDER

ERIN = "/mnt/scratche/fast/fmlab/datasets/imaging/ERIN/data/PathologyReport_AnonIds.csv"
OUT = os.environ.get("OUTDIR", HERE)
TEXT_COLS = ["FinalDiagnosis_redacted", "MicroscopicDescription_redacted"]
ADDENDA = ["Addendum1_redacted", "Addendum2_redacted", "Addendum3_redacted"]

# --- feasibility grader, copied verbatim from erin_feas.py (first-match cascade) ---
def feas_sev(t):
    t = t.lower()
    if re.search(r"adenocarcinoma|invasive carcinoma|malignan|carcinoma", t) and not re.search(
            r"no evidence of malignan|no malignan|nor evidence of malignan|not evidence of malignan", t):
        return "CANCER"
    if re.search(r"high[- ]grade dysplasia|high grade glandular dysplasia|\bhgd\b", t): return "HGD"
    if re.search(r"low[- ]grade dysplasia|low grade glandular dysplasia|\blgd\b", t): return "LGD"
    if re.search(r"indefinite for dysplasia", t): return "IND"
    if re.search(r"intestinal metaplasia|barrett", t): return "NDBE"
    if re.search(r"no dysplasia|no significant abnormalit|negative|reactive|no active inflammation", t): return "NDBE"
    return None

rep = pd.read_csv(ERIN, dtype=str, low_memory=False).fillna("")
rep["text_core"] = rep[TEXT_COLS].agg(" ".join, axis=1)
rep["text_full"] = rep[TEXT_COLS + ADDENDA].agg(" ".join, axis=1)
print(f"reports: {len(rep)}")

rep["feas"] = rep["text_full"].map(feas_sev)
rep["feas_core"] = rep["text_core"].map(feas_sev)
rep["plad"] = [label_report(t, BARRETTS_LADDER)["grade"] for t in rep["text_full"]]
rep["plad_core"] = [label_report(t, BARRETTS_LADDER)["grade"] for t in rep["text_core"]]

res = {"n_reports": int(len(rep))}
for a, b, name in [("plad", "feas", "full_text"), ("plad_core", "feas_core", "core_text")]:
    m = rep[rep[a].notna() & rep[b].notna()]
    res[name] = {
        "n_both_labelled": int(len(m)),
        "agreement": round(float((m[a] == m[b]).mean()), 4),
        "confusion_pathladder_rows": pd.crosstab(m[a], m[b]).to_dict(),
        "pathladder_dist": rep[a].value_counts(dropna=False).astype(int).to_dict(),
        "feasibility_dist": rep[b].value_counts(dropna=False).astype(int).to_dict(),
    }
    print(name, "agreement", res[name]["agreement"], "on", len(m))

# where pathladder says CANCER and feasibility does not (full text)
dis = rep[(rep["plad"] == "CANCER") & (rep["feas"] != "CANCER")]
t = dis["text_full"].str.lower()
res["pathladder_cancer_not_feas"] = {
    "n": int(len(dis)),
    "negation_context": int(t.str.contains(r"no (?:evidence of )?(?:residual )?(?:malignan|carcinoma)").sum()),
    "history_mention": int(t.str.contains(r"history of|previous|known|treated|neoadjuvant").sum()),
    "resection_specimen": int(t.str.contains(r"resection|oesophagectomy|esophagectomy|gastrectomy|\bemr\b|\besd\b").sum()),
}
dis[["CaseName"] + TEXT_COLS].head(40).to_csv(os.path.join(OUT, "audit_plad_cancer_only.csv"), index=False)

# and the reverse
rev = rep[(rep["feas"] == "CANCER") & (rep["plad"] != "CANCER")]
res["feas_cancer_not_pathladder"] = {"n": int(len(rev))}
rev[["CaseName"] + TEXT_COLS].head(40).to_csv(os.path.join(OUT, "audit_feas_cancer_only.csv"), index=False)
print("plad-only CANCER:", res["pathladder_cancer_not_feas"], "| feas-only CANCER:", len(rev))

json.dump(res, open(os.path.join(OUT, "erin_audit_v2_results.json"), "w"), indent=2)
print("wrote erin_audit_v2_results.json + two disagreement samples")

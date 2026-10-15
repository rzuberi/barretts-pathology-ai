"""EXECUTION_PLAN 1.3: audit pathladder's CANCER over-call on ERIN.

(1) Patient-level comparison: max pathladder grade per patient vs feasibility
    index_sev (join on anon_id — the level at which the feasibility labels exist).
(2) Categorise CANCER-labelled reports by suspected cause markers (history/known/
    previous, resection specimens, 'no residual').
(3) Export 30 uncategorised CANCER report texts for manual reading.
"""
import json, os, re, sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ERIN = "/mnt/scratche/fast/fmlab/datasets/imaging/ERIN/data/PathologyReport_AnonIds.csv"
FEAS = "/mnt/scratche/slow/fmlab/zuberi01/phd/erin_multimodal_feasibility/trainable_progression_cohort.csv"
LBL = os.path.join(HERE, "erin_pathladder_labels.csv")
OUT = os.environ.get("OUTDIR", HERE)
ORDER = {"NDBE": 0, "IND": 1, "LGD": 2, "HGD": 3, "CANCER": 4}

lab = pd.read_csv(LBL, dtype=str)
rep = pd.read_csv(ERIN, dtype=str, low_memory=False)
rep["_text"] = rep[["MicroscopicDescription_redacted", "FinalDiagnosis_redacted"]].fillna("").agg(" ".join, axis=1)
df = lab.merge(rep[["CaseName", "_text"]], on="CaseName", how="left")

# (1) patient-level max grade vs feasibility index_sev
pl = (df.dropna(subset=["pathladder_grade"])
        .assign(rank=lambda d: d["pathladder_grade"].map(ORDER))
        .groupby("anon_id")["rank"].max()
        .map({v: k for k, v in ORDER.items()}))
feas = pd.read_csv(FEAS, dtype=str).drop_duplicates("anon_id").set_index("anon_id")
m = pd.DataFrame({"pathladder_max": pl}).join(feas["index_sev"], how="inner").dropna()
m["index_sev"] = m["index_sev"].str.upper().str.strip()
agree = float((m["pathladder_max"] == m["index_sev"]).mean())
conf = pd.crosstab(m["pathladder_max"], m["index_sev"])
print(f"patient-level join n={len(m)} exact agreement={agree:.3f}")
print(conf.to_string())

# (2) categorise CANCER-labelled reports
canc = df[df["pathladder_grade"] == "CANCER"].copy()
t = canc["_text"].str.lower()
canc["hx"] = t.str.contains(r"\b(?:history of|previous|known|treated|post[- ]?chemo|neoadjuvant)\b", regex=True)
canc["resection"] = t.str.contains(r"\b(?:resection|oesophagectomy|esophagectomy|gastrectomy|emr|esd)\b", regex=True)
canc["no_residual"] = t.str.contains(r"\bno (?:residual|evidence of residual)\b", regex=True)
cats = {"history_mention": int(canc["hx"].sum()),
        "resection_specimen": int(canc["resection"].sum()),
        "no_residual_neg_miss": int(canc["no_residual"].sum()),
        "none_of_the_above": int((~(canc["hx"] | canc["resection"] | canc["no_residual"])).sum()),
        "total_cancer_reports": int(len(canc))}
print(cats)

# (3) sample for manual read
sample = canc[~(canc["hx"] | canc["resection"] | canc["no_residual"])].head(30)
sample[["CaseName", "_text"]].to_csv(os.path.join(OUT, "cancer_audit_sample.csv"), index=False)

json.dump({"patient_level": {"n": int(len(m)), "agreement": round(agree, 4),
                             "confusion": conf.to_dict()},
           "cancer_categories": cats},
          open(os.path.join(OUT, "cancer_audit_results.json"), "w"), indent=2)
print("wrote cancer_audit_results.json + cancer_audit_sample.csv")

"""Deep-dive D (extends 2.7): what are the 204 held-out 'unsure' reports?

Compares unsure vs train-eligible on: jury label spread, hedging density, report
length, year, addenda, IND-adjacency. The Ch4 table showing the holdout is
principled clinical ambiguity, not noise.
"""
import json, os
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("OUTDIR", HERE)
lab = pd.read_csv(os.path.join(HERE, "erin_labels_jury_final.csv"), dtype=str)
votes = pd.read_csv(os.path.join(HERE, "llm_full", "jury_votes.csv"), index_col=0)
rep = pd.read_csv("/mnt/scratche/fast/fmlab/datasets/imaging/ERIN/data/PathologyReport_AnonIds.csv",
                  dtype=str, low_memory=False).fillna("")
TEXT = ["FinalDiagnosis_redacted", "MicroscopicDescription_redacted"]
rep["_text"] = rep[TEXT].agg(" ".join, axis=1)
df = lab.merge(rep[["CaseName", "_text", "Addendum1_redacted"]], on="CaseName", how="left")
df = df.merge(votes, left_on="CaseName", right_index=True, how="left", suffixes=("", "_votes"))
df["unsure"] = df["label_status"] == "unsure_held_out"

t = df["_text"].fillna("")
df["hedging"] = t.str.lower().str.count(r"suspicious|indefinite|cannot exclude|difficult|equivocal|\?")
df["length"] = t.str.len()
df["has_addendum"] = (df["Addendum1_redacted"].fillna("").str.len() > 0)
df["year"] = pd.to_numeric(df["CollectedOrOrdered"].str[-4:], errors="coerce")

res = {}
for grpname, g in df.groupby("unsure"):
    key = "unsure" if grpname else "train_eligible"
    res[key] = {"n": int(len(g)),
                "hedging_mean": round(float(g["hedging"].mean()), 2),
                "length_mean": round(float(g["length"].mean()), 0),
                "addendum_rate": round(float(g["has_addendum"].mean()), 3),
                "year_median": float(g["year"].median()),
                "jury_entropy_mean": round(float(pd.to_numeric(g["jury_entropy"], errors="coerce").mean()), 3),
                "jury_top_label_dist": g["jury_label"].value_counts().head(5).to_dict()}
u = df[df["unsure"]]
res["unsure_ind_involved"] = int(u["jury_label"].isin(["IND"]).sum())
json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=2)
print(json.dumps(res, indent=2))

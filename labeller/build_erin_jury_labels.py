"""EXECUTION_PLAN 2.16: final ERIN labels from the adopted jury-vote scheme.

Rules (Rehan, 2026-08-20):
  - train-eligible: jury majority >= 6/8 on a real grade (not NA), i.e. jury_frac >= 0.75
  - Rehan/Claude adjudications override the jury where present
  - everything else -> "unsure": kept in a held-aside file, NEVER trained on
Also rebuilds the progression cohort (v3) using train-eligible grades only for
both index and event assignment.
"""
import json, os
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
VOTES = os.path.join(HERE, "llm_full", "jury_votes.csv")
ERIN = "/mnt/scratche/fast/fmlab/datasets/imaging/ERIN/data/PathologyReport_AnonIds.csv"
ADJ = os.path.join(HERE, "adjudications.csv")
OUT = os.environ.get("OUTDIR", HERE)
NUM = {"NDBE": 0, "IND": 1, "LGD": 2, "HGD": 3, "CANCER": 4}

votes = pd.read_csv(VOTES, index_col=0)
rep = pd.read_csv(ERIN, dtype=str, low_memory=False)
df = rep[["anon_id", "CaseName", "CollectedOrOrdered"]].merge(
    votes, left_on="CaseName", right_index=True, how="left")

eligible = (df["jury_frac"] >= 0.75) & df["jury_label"].isin(NUM)
df["final_label"] = np.where(eligible, df["jury_label"], "unsure")
df["label_status"] = np.where(eligible, "train_eligible", "unsure_held_out")

if os.path.exists(ADJ):
    adj = pd.read_csv(ADJ, dtype=str)
    amap = dict(zip(adj["CaseName"], adj["decision"].str.lower()))
    hit = df["CaseName"].map(amap)
    df.loc[hit == "yes", ["final_label", "label_status"]] = ["CANCER", "adjudicated"]
    # 'no' = not cancer: keep jury label if it's a confident non-cancer grade, else unsure
    no_m = (hit == "no")
    keep = no_m & eligible & (df["jury_label"] != "CANCER")
    df.loc[no_m & ~keep, ["final_label", "label_status"]] = ["unsure", "unsure_held_out"]
    df.loc[keep, "label_status"] = "adjudicated"
    df.loc[hit == "unsure", ["final_label", "label_status"]] = ["unsure", "unsure_held_out"]

df.to_csv(os.path.join(OUT, "erin_labels_jury_final.csv"), index=False)
df[df["label_status"] == "unsure_held_out"].to_csv(
    os.path.join(OUT, "erin_labels_unsure_heldout.csv"), index=False)

# progression cohort v3: train-eligible grades only
lab = df[df["label_status"] != "unsure_held_out"].copy()
lab["date"] = pd.to_datetime(lab["CollectedOrOrdered"], errors="coerce", dayfirst=True)
lab["num"] = lab["final_label"].map(NUM)
lab = lab.dropna(subset=["date", "num"])
rows = []
for pid, g in lab.sort_values("date").groupby("anon_id"):
    idx = g[g["num"] <= 2]
    if idx.empty: continue
    t0 = idx.iloc[0]["date"]
    fut = g[g["date"] > t0]
    if fut.empty: continue
    ev = fut[fut["num"] >= 3]
    rows.append({"anon_id": pid, "index_date": t0.date(),
                 "index_grade": idx.iloc[0]["final_label"],
                 "progressed_to_HGDplus": bool(len(ev)),
                 "tte_days": int(((ev.iloc[0]["date"] if len(ev) else fut.iloc[-1]["date"]) - t0).days),
                 "n_future": int(len(fut))})
coh = pd.DataFrame(rows)
coh.to_csv(os.path.join(OUT, "erin_progression_cohort_v3.csv"), index=False)

summary = {"reports": int(len(df)),
           "status_dist": df["label_status"].value_counts().to_dict(),
           "label_dist_train_eligible": df.loc[df.label_status == "train_eligible",
                                               "final_label"].value_counts().to_dict(),
           "label_dist_usable_incl_adjudicated": df.loc[df.label_status != "unsure_held_out",
                                                        "final_label"].value_counts().to_dict(),
           "progression_v3": {"patients": int(len(coh)),
                              "progressors": int(coh["progressed_to_HGDplus"].sum()) if len(coh) else 0}}
json.dump(summary, open(os.path.join(OUT, "erin_jury_labels_summary.json"), "w"), indent=2)
print(json.dumps(summary, indent=2))

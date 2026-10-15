"""1.19: revalidate the pathladder negation-window fix on a FRESH sample.

The 130-char window was tuned on the same 80-case audit used to condemn the
feasibility grader (label-function leakage, Grok's review point). Test: on all
corpus reports EXCLUDING the 80 adjudicated cases, does pathladder agree with
the 8-model jury consensus — overall, and specifically on negation-bearing
reports (the pattern class the fix touched)?
"""
import json, os, re
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("OUTDIR", HERE)

lab = pd.read_csv(os.path.join(HERE, "erin_labels_jury_final.csv"), dtype=str)
plad = pd.read_csv(os.path.join(HERE, "erin_pathladder_labels.csv"), dtype=str)
adj = pd.read_csv(os.path.join(HERE, "adjudications.csv"), dtype=str)
rep = pd.read_csv("/mnt/scratche/fast/fmlab/datasets/imaging/ERIN/data/PathologyReport_AnonIds.csv",
                  dtype=str, low_memory=False).fillna("")
rep["_text"] = rep[["FinalDiagnosis_redacted", "MicroscopicDescription_redacted"]].agg(" ".join, axis=1)

pcol = "pathladder_grade"
df = lab.merge(plad[["CaseName", pcol]].drop_duplicates("CaseName"), on="CaseName", how="inner")
df = df.merge(rep[["CaseName", "_text"]].drop_duplicates("CaseName"), on="CaseName", how="left")
audited = set(adj["CaseName"]) if "CaseName" in adj.columns else set()
df["fresh"] = ~df["CaseName"].isin(audited)

NEG = re.compile(r"\bno evidence|negative for|without|absence of|not seen|no dysplasia|"
                 r"free of|excluded?\b", re.I)
df["negation_bearing"] = df["_text"].fillna("").str.contains(NEG)
# confident jury rows only — the comparison target must itself be trustworthy
conf = df[df["label_status"].isin(["train_eligible", "adjudicated"])].copy()
conf["agree"] = conf[pcol] == conf["jury_label"]

def block(d):
    if not len(d):
        return {"n": 0}
    cm = (d[~d["agree"]].groupby([pcol, "jury_label"]).size()
          .sort_values(ascending=False).head(6))
    return {"n": int(len(d)), "agreement": round(float(d["agree"].mean()), 4),
            "top_confusions": {f"plad={a}|jury={b}": int(v) for (a, b), v in cm.items()}}

res = {"audit_sample": block(conf[~conf["fresh"]]),
       "fresh_all": block(conf[conf["fresh"]]),
       "fresh_negation_bearing": block(conf[conf["fresh"] & conf["negation_bearing"]]),
       "fresh_non_negation": block(conf[conf["fresh"] & ~conf["negation_bearing"]]),
       "_meta": {"n_audited_excluded": len(audited), "pathladder_col": pcol,
                 "target": "jury consensus (train-eligible/adjudicated only)"}}
json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=2)
print(json.dumps(res, indent=2))

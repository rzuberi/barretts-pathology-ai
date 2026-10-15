"""EXECUTION_PLAN 2.7 jury analysis: parse viability, agreement with adjudicated
truth, pairwise agreement, Fleiss kappa, and jury-vote label candidates.

Usage: python analyze_jury.py <dir_of_llm_grades_csvs> [--adjudicated]
With --adjudicated, scores each model against adjudications.csv (cancer boundary).
"""
import glob, json, os, sys
import numpy as np, pandas as pd

D = sys.argv[1]
ADJ_MODE = "--adjudicated" in sys.argv
HERE = os.path.dirname(os.path.abspath(__file__))
GRADES = ["NDBE", "IND", "LGD", "HGD", "CANCER", "NA"]

frames = {}
for f in sorted(glob.glob(os.path.join(D, "llm_grades_*.csv"))):
    model = os.path.basename(f).replace("llm_grades_", "").rsplit("_shard", 1)[0]
    try:
        df = pd.read_csv(f).drop_duplicates("CaseName")
    except Exception:  # legacy unquoted rows where the id itself contains a comma
        recs = []
        for line in open(f).read().splitlines()[1:]:
            if "," in line:
                name, g = line.rsplit(",", 1)
                recs.append({"CaseName": name.strip('"'), "llm_grade": g})
        df = pd.DataFrame(recs).drop_duplicates("CaseName")
    frames.setdefault(model, []).append(df)
jury = {m: pd.concat(fs).drop_duplicates("CaseName").set_index("CaseName")["llm_grade"]
        for m, fs in frames.items()}
print("models:", list(jury))

wide = pd.DataFrame(jury)
res = {"n_reports": int(len(wide)), "models": {}}
for m in wide.columns:
    v = wide[m]
    res["models"][m] = {"parse_rate": round(float((v != "PARSE_FAIL").mean()), 4),
                        "dist": v.value_counts().to_dict()}

if ADJ_MODE and os.path.exists(os.path.join(HERE, "adjudications.csv")):
    adj = pd.read_csv(os.path.join(HERE, "adjudications.csv")).drop_duplicates("CaseName")
    adj = adj[adj["decision"].isin(["yes", "no"])].set_index("CaseName")["decision"]
    for m in wide.columns:
        both = wide[m].reindex(adj.index).dropna()
        both = both[both != "PARSE_FAIL"]
        acc = float(((both == "CANCER") == (adj.reindex(both.index) == "yes")).mean())
        res["models"][m]["cancer_agreement_vs_adjudication"] = round(acc, 4)

# pairwise agreement (on mutually parsed reports)
models = list(wide.columns)
pair = {}
for i, a in enumerate(models):
    for b in models[i + 1:]:
        ok = wide[[a, b]].replace("PARSE_FAIL", np.nan).dropna()
        if len(ok): pair[f"{a}|{b}"] = round(float((ok[a] == ok[b]).mean()), 4)
res["pairwise_agreement"] = pair

# Fleiss kappa over reports rated by all models
full = wide.replace("PARSE_FAIL", np.nan).dropna()
if len(full) and len(models) > 1:
    counts = np.stack([(full.values == g).sum(axis=1) for g in GRADES], axis=1)
    n = counts.sum(axis=1)[0]
    P_i = ((counts * (counts - 1)).sum(axis=1)) / (n * (n - 1))
    p_j = counts.sum(axis=0) / counts.sum()
    P_bar, P_e = P_i.mean(), (p_j ** 2).sum()
    res["fleiss_kappa"] = round(float((P_bar - P_e) / (1 - P_e)), 4)
    res["n_fully_rated"] = int(len(full))

# jury vote: majority label + entropy
def vote(row):
    v = row.replace("PARSE_FAIL", np.nan).dropna()
    if not len(v): return pd.Series({"jury_label": None, "jury_frac": np.nan, "jury_entropy": np.nan})
    c = v.value_counts(normalize=True)
    ent = float(-(c * np.log2(c)).sum())
    return pd.Series({"jury_label": c.index[0], "jury_frac": float(c.iloc[0]), "jury_entropy": ent})
votes = wide.apply(vote, axis=1)
votes.to_csv(os.path.join(D, "jury_votes.csv"))
res["jury_label_dist"] = votes["jury_label"].value_counts().to_dict()
res["unanimous_frac"] = round(float((votes["jury_frac"] == 1.0).mean()), 4)

json.dump(res, open(os.path.join(D, "jury_analysis.json"), "w"), indent=2)
print(json.dumps(res, indent=2)[:2500])
print("wrote jury_analysis.json + jury_votes.csv")

"""P31 check: second juror (gemma3:27b, v2 prompt) vs MedGemma-27B v2 on the same reports, per field: exact agreement,
Cohen's kappa, and the jury final label as a third reference for grade. Env: A_DIRS, B_DIRS (colon lists), OUTDIR."""
import glob, json, os, numpy as np, pandas as pd
from sklearn.metrics import cohen_kappa_score
T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"; OUT = os.environ.get("OUTDIR", ".")
def load(dirs): return pd.DataFrame([json.loads(l) for d in dirs.split(":") for f in glob.glob(d + "/fields_*_shard*.jsonl") for l in open(f)]).drop_duplicates("CaseName").set_index("CaseName")
A = load(os.environ["A_DIRS"]); B = load(os.environ["B_DIRS"]); common = A.index.intersection(B.index)
F = ["grade", "site", "specimen_type", "intestinal_metaplasia", "goblet_cells", "inflammation", "ulceration_or_erosion", "squamous_only", "gastric_mucosa_present", "treatment_effect", "p53", "diagnostic_certainty"]
res = {"n_common": int(len(common)), "model_A": str(A.model.iloc[0]), "model_B": str(B.model.iloc[0]), "fields": {}}
for f in F:
    a, b = A.loc[common, f], B.loc[common, f]; ok = (a != "PARSE_FAIL") & (b != "PARSE_FAIL")
    res["fields"][f] = {"n": int(ok.sum()), "exact": round(float((a[ok] == b[ok]).mean()), 4), "kappa": round(float(cohen_kappa_score(a[ok], b[ok])), 4) if a[ok].nunique() > 1 and b[ok].nunique() > 1 else None,
                        "B_parse_fail": round(float((b == "PARSE_FAIL").mean()), 4), "B_marginal": b.value_counts().to_dict()}
lab = pd.read_csv(T + "/labeller/erin_labels_jury_final.csv", dtype=str).drop_duplicates("CaseName").set_index("CaseName"); el = lab.index.intersection(common); el = el[lab.loc[el, "label_status"] == "train_eligible"]
G = ["NDBE", "IND", "LGD", "HGD", "CANCER"]
for nm, X in (("A", A), ("B", B)):
    g = X.loc[el, "grade"]; j = lab.loc[el, "final_label"]; ok = g.isin(G)
    res[f"grade_{nm}_vs_jury"] = {"n": int(ok.sum()), "exact": round(float((g[ok] == j[ok]).mean()), 4), "two_tier": round(float((g[ok].isin(G[2:]) == j[ok].isin(G[2:])).mean()), 4)}
os.makedirs(OUT, exist_ok=True); json.dump(res, open(os.path.join(OUT, "p31_agreement.json"), "w"), indent=1); print(json.dumps(res, indent=None)[:3000])

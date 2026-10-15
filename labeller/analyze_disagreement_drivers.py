"""EXECUTION_PLAN 2.18b: what makes the jury disagree? (existing votes, no new gradings)

Per ERIN report: jury entropy (from llm_full votes) regressed on report features:
text length, negation-cue count, addenda presence, specimen-section count, report
age, uppercase fraction (OCR/format proxy). Logistic (disagree vs unanimous) +
feature effect sizes. Output: results.json + per-feature table.
"""
import json, os, re, sys
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("OUTDIR", HERE)
votes = pd.read_csv(os.path.join(HERE, "llm_full", "jury_votes.csv"), index_col=0)
rep = pd.read_csv("/mnt/scratche/fast/fmlab/datasets/imaging/ERIN/data/PathologyReport_AnonIds.csv",
                  dtype=str, low_memory=False).fillna("")
TEXT = ["FinalDiagnosis_redacted", "MicroscopicDescription_redacted",
        "Addendum1_redacted", "Addendum2_redacted", "Addendum3_redacted"]
rep["_text"] = rep[TEXT].agg(" ".join, axis=1)
df = rep.merge(votes, left_on="CaseName", right_index=True, how="inner")
df = df[df["jury_frac"].notna()]

t = df["_text"]
feats = pd.DataFrame({
    "len_chars": t.str.len(),
    "negation_cues": t.str.lower().str.count(r"\bno\b|\bnot\b|negative|without|nor\b"),
    "has_addendum": (df["Addendum1_redacted"].str.len() > 0).astype(int),
    "n_specimens": t.str.count(r"(?m)^\s*[A-H][.)]\s"),
    "year": pd.to_numeric(df["CollectedOrOrdered"].str[-4:], errors="coerce"),
    "upper_frac": t.map(lambda s: sum(c.isupper() for c in s) / max(len(s), 1)),
    "hedging": t.str.lower().str.count(r"suspicious|indefinite|cannot exclude|difficult|equivocal|\?"),
})
y = (df["jury_frac"] < 1.0).astype(int).values  # any disagreement vs unanimity
mask = feats.notna().all(axis=1).values
X, yv = feats[mask].values, y[mask]
print(f"n={len(yv)} disagree_rate={yv.mean():.3f}")

sc = StandardScaler().fit(X)
lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(X), yv)
auc = cross_val_score(LogisticRegression(max_iter=2000, class_weight="balanced"),
                      sc.transform(X), yv, cv=5, scoring="roc_auc")
coefs = dict(zip(feats.columns, [round(float(c), 3) for c in lr.coef_[0]]))
uni = {}
for c in feats.columns:
    hi = yv[X[:, list(feats.columns).index(c)] > np.median(X[:, list(feats.columns).index(c)])]
    lo = yv[X[:, list(feats.columns).index(c)] <= np.median(X[:, list(feats.columns).index(c)])]
    uni[c] = {"disagree_rate_above_median": round(float(hi.mean()), 3),
              "below_median": round(float(lo.mean()), 3)}
res = {"n": int(len(yv)), "disagree_rate": round(float(yv.mean()), 4),
       "model_auc_cv": round(float(auc.mean()), 3),
       "std_logistic_coefs": coefs, "univariate": uni}
json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=2)
print(json.dumps(res, indent=2))

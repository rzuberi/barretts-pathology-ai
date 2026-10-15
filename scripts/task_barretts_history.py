"""2.26 analysis (runs after the full-corpus jury campaign): Barrett's-DB
natural history + jury-at-scale validation.

(i) Majority-vote jury grades over the 13,645-report corpus, validated against
    the database's own structured dysplasia grading — the largest
    jury-vs-structured-truth comparison available (hardening for the
    self-validation criticism).
(ii) Per-patient longitudinal grade sequences -> empirical transition matrix
     between NDBE/IND/LGD/HGD/CANCER + dwell times (real-world natural history).
(iii) Text-only progression baseline: future HGD+ from grade-history features.
"""
import glob, json, os
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"
EXP = "/mnt/scratche/slow/fmlab/zuberi01/barretts_db_export"
OUT = os.environ.get("OUTDIR", ".")
LADDER = ["NDBE", "IND", "LGD", "HGD", "CANCER"]
NUM = {g: i for i, g in enumerate(LADDER)}

votes = {}
for f in glob.glob(T + "/feasibility/runs/jury_full_*/output/llm_grades_*.csv"):
    model = os.path.basename(f).replace("llm_grades_", "").rsplit("_shard", 1)[0]
    rows = []
    for line in open(f).read().splitlines()[1:]:
        cid, _, g = line.rpartition(",")
        if g in NUM: rows.append((cid, g))
    votes.setdefault(model, {}).update(dict(rows))
print("models:", {k: len(v) for k, v in votes.items()}, flush=True)
assert len(votes) >= 6, "need >=6 jurors' outputs"

all_ids = sorted(set.union(*(set(v) for v in votes.values())))
maj, frac = {}, {}
for cid in all_ids:
    vs = [v[cid] for v in votes.values() if cid in v]
    if len(vs) < 5: continue
    top = max(set(vs), key=vs.count)
    maj[cid] = top; frac[cid] = vs.count(top) / len(vs)
jury = pd.DataFrame({"pathology_text_id": list(maj), "jury_grade": list(maj.values()),
                     "jury_frac": [frac[c] for c in maj]})
print(f"jury-labelled reports: {len(jury)}; confident(>=0.75): {(jury['jury_frac']>=0.75).sum()}", flush=True)

rep = pd.read_csv(EXP + "/pathology_text_normalised_full.csv", dtype=str, low_memory=False)
date_col = next((c for c in rep.columns if "date" in c.lower()), None)
pid_col = next((c for c in rep.columns if "patient" in c.lower() or "participant" in c.lower()), None)
print("rep columns:", list(rep.columns)[:12], "| date:", date_col, "| pid:", pid_col, flush=True)
rep = rep.merge(jury, on="pathology_text_id", how="inner")
rep["date"] = pd.to_datetime(rep[date_col], errors="coerce") if date_col else pd.NaT
rep["num"] = rep["jury_grade"].map(NUM)

res = {"_meta": {"jurors": sorted(votes), "n_labelled": len(jury),
                 "n_confident": int((jury["jury_frac"] >= 0.75).sum())}}

# (i) validation vs the corpus's OWN per-report structured grade columns
GMAP = {"0": "NDBE", "1": "IND", "2": "LGD", "3": "HGD", "4": "CANCER",
        "ndbe": "NDBE", "no dysplasia": "NDBE", "negative": "NDBE",
        "indefinite": "IND", "ind": "IND", "lgd": "LGD", "low": "LGD",
        "hgd": "HGD", "high": "HGD", "imc": "CANCER", "cancer": "CANCER",
        "adenocarcinoma": "CANCER", "oac": "CANCER", "ac": "CANCER"}
def map_struct(v):
    v = str(v).strip().lower()
    if v in GMAP: return GMAP[v]
    for k, g in GMAP.items():
        if len(k) > 2 and k in v: return g
    return None
scol = next((c for c in ("highestgradedysconfirmed", "highestgradedys")
             if c in rep.columns), None)
if scol:
    res["structured_value_counts"] = rep[scol].value_counts().head(10).to_dict()
    rep["_struct"] = rep[scol].map(map_struct)
    both = rep.dropna(subset=["_struct"])
    conf = both[both["jury_frac"] >= 0.75]
    if len(both) > 200:
        res["jury_vs_structured"] = {
            "source_column": scol, "n_matched": len(both),
            "agreement": round(float((both["jury_grade"] == both["_struct"]).mean()), 4),
            "n_confident": len(conf),
            "agreement_confident": round(float((conf["jury_grade"] == conf["_struct"]).mean()), 4),
            "confusions": {f"jury={a}|struct={b}": int(n) for (a, b), n in
                           both[both["jury_grade"] != both["_struct"]]
                           .groupby(["jury_grade", "_struct"]).size()
                           .sort_values(ascending=False).head(8).items()}}

# (ii) transition matrix on confident labels
if pid_col and date_col:
    seq = rep[(rep["jury_frac"] >= 0.75)].dropna(subset=["date", "num"]).sort_values("date")
    trans = np.zeros((5, 5), int); dwell = []
    n_pat = 0
    for pid, g in seq.groupby(pid_col):
        states = g[["date", "num"]].values
        if len(states) < 2: continue
        n_pat += 1
        for i in range(1, len(states)):
            a, b = int(states[i-1][1]), int(states[i][1])
            trans[a][b] += 1
            dwell.append(((states[i][0] - states[i-1][0]).days / 365.25, a, b))
    res["natural_history"] = {
        "n_patients_with_sequences": n_pat,
        "transitions": {f"{LADDER[a]}->{LADDER[b]}": int(trans[a][b])
                        for a in range(5) for b in range(5) if trans[a][b] > 0},
        "median_years_ndbe_to_first_dysplasia": round(float(np.median(
            [d for d, a, b in dwell if a == 0 and b >= 2])), 2) if any(
            a == 0 and b >= 2 for _, a, b in dwell) else None}
    # (iii) text-only progression baseline
    feats, ys, gps = [], [], []
    for pid, g in seq.groupby(pid_col):
        g = g.sort_values("date")
        for i in range(1, len(g)):
            past = g.iloc[:i]
            fut = g.iloc[i:]
            ys.append(int((fut["num"] >= 3).any()))
            feats.append([past["num"].max(), len(past),
                          (past["date"].iloc[-1] - past["date"].iloc[0]).days / 365.25])
            gps.append(pid)
    if len(ys) > 500 and 0 < sum(ys) < len(ys):
        Xf, yf, gp = np.array(feats), np.array(ys), np.array(gps)
        oof = np.zeros(len(yf))
        for tr, te in GroupKFold(5).split(Xf, yf, gp):
            lr = LogisticRegression(class_weight="balanced", max_iter=2000).fit(Xf[tr], yf[tr])
            oof[te] = lr.predict_proba(Xf[te])[:, 1]
        res["text_only_progression"] = {"n": len(yf), "pos": int(yf.sum()),
                                        "auc": round(float(roc_auc_score(yf, oof)), 4)}
json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=2)
print(json.dumps(res, indent=2)[:3000])

"""POST HOC (not pre-registered; added 2026-10-19 after seeing results/main.json): is the STAD WGD signal more than subtype
detection? WGD AUROC of the pre-registered pooled ABMIL / mean-pool OOF scores (strat CV, averaged over 3 repeats) within
STAD tumours of the CIN subtype and within MSS STAD tumours. Writes results/posthoc.json."""
import glob, json
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
A = "/mnt/scratche/slow/fmlab/zuberi01/analysis/outreach_panakeia"; NB = 2000
c = pd.read_csv(A + "/cohort.csv").set_index("patient")
def wauc(score, y, W):
    o = np.argsort(score, kind="stable"); yo = y[o].astype(np.float32); Wo = W[:, o]; neg = Wo * (1 - yo)
    cumneg = np.cumsum(neg, axis=1) - neg; num = (Wo * yo * cumneg).sum(1); den = (Wo * yo).sum(1) * neg.sum(1)
    return np.where(den > 0, num / np.maximum(den, 1e-9), np.nan)
out = {"note": "post hoc, not pre-registered"}
for model in ("abmil", "meanlr"):
    d = [pd.read_csv(f).set_index("patient") for f in sorted(glob.glob(f"{A}/preds/wgd_strat_{model}_r*.csv"))]
    s = pd.concat([x.score for x in d], axis=1).mean(1); y = d[0].y
    for name, m in (("stad_cin_only", (c.loc[s.index, "cohort"] == "stad") & (c.loc[s.index, "subtype"] == "STAD_CIN")),
                    ("stad_mss_only", (c.loc[s.index, "cohort"] == "stad") & (c.loc[s.index, "msi"] == 0)),
                    ("stad_noncin_only", (c.loc[s.index, "cohort"] == "stad") & c.loc[s.index, "subtype"].notna() & (c.loc[s.index, "subtype"] != "STAD_CIN"))):
        ss, yy = s[m.values].values, y[m.values].values.astype(int)
        if yy.sum() < 5 or (1 - yy).sum() < 5: out[f"{model}_{name}"] = {"n": int(len(yy)), "n_pos": int(yy.sum()), "note": "too few of one class"}; continue
        r = np.random.RandomState(5); W = np.stack([np.bincount(r.randint(0, len(yy), len(yy)), minlength=len(yy)) for _ in range(NB)]).astype(np.float32)
        b = wauc(ss, yy, W); out[f"{model}_{name}"] = {"point": round(float(roc_auc_score(yy, ss)), 4), "ci": [round(float(np.nanpercentile(b, q)), 4) for q in (2.5, 97.5)], "n": int(len(yy)), "n_pos": int(yy.sum())}
json.dump(out, open(A + "/results/posthoc.json", "w"), indent=1); print(json.dumps(out, indent=1))

"""2.45 (Astra A12): genotype-visibility curves with PREVALENCE MATCHED.

2.22 matched total n across strata but TP53 prevalence differs (~83% OAC vs
~46% STAD/GEJ), so 'population contrast' and 'prevalence contrast' were
confounded. Here every (stratum, n) cell draws n/2 positives + n/2 negatives
(so AUC is estimated on balanced samples everywhere), and mixed-cohort cells
add (i) a cohort-identity-only baseline and (ii) within-cohort label
permutation nulls. Reuses the 2.22 pooled-feature cache; standard probe
(StandardScaler -> PCA64 -> logistic), 5-fold CV inside each draw.
"""
import json, os
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"
CACHE = T + "/feasibility/runs/visibility_curve/output/pooled_features.npz"
OUT = os.environ.get("OUTDIR", ".")
GRID = [40, 50, 65, 100, 141, 200]
REPS, PERMS_PER_REP = 20, 3

d = np.load(CACHE, allow_pickle=True)
X = d["X"]; meta = pd.DataFrame({"stratum": d["stratum"], "tp53": d["tp53"], "wgd": d["wgd"]})
STRATA = {"tcga_oac": meta["stratum"] == "tcga_oac", "occams_oac": meta["stratum"] == "occams_oac",
          "oac_combined": meta["stratum"].isin(["tcga_oac", "occams_oac"]),
          "stad_gej": meta["stratum"] == "stad_gej", "mixed_all": meta["stratum"].notna()}
MIXED = {"oac_combined", "mixed_all"}

def probe(Xs, ys, seed, Xextra=None):
    oof = np.zeros(len(ys))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(Xs, ys):
        p = Pipeline([("s", StandardScaler()), ("p", PCA(min(64, len(tr) - 1))),
                      ("l", LogisticRegression(C=0.5, class_weight="balanced", max_iter=2000))])
        p.fit(Xs[tr], ys[tr]); oof[te] = p.predict_proba(Xs[te])[:, 1]
    return roc_auc_score(ys, oof)

def identity_probe(S, ys, seed):
    oof = np.zeros(len(ys))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(S, ys):
        l = LogisticRegression(max_iter=1000).fit(S[tr], ys[tr]); oof[te] = l.predict_proba(S[te])[:, 1]
    return roc_auc_score(ys, oof)

res = {"_meta": {"grid": GRID, "reps": REPS, "design": "balanced n/2 pos + n/2 neg per draw",
                 "counts": meta["stratum"].value_counts().to_dict()}}
for target in ["tp53", "wgd"]:
    res[target] = {}
    for sname, mask in STRATA.items():
        ok = mask.values & meta[target].notna().values
        Xs_all, ys_all = X[ok], meta[target][ok].astype(int).values
        strat_all = meta["stratum"][ok].values
        pos_i, neg_i = np.where(ys_all == 1)[0], np.where(ys_all == 0)[0]
        cell = {"n_available": int(len(ys_all)), "pos": int(len(pos_i)), "neg": int(len(neg_i)),
                "prevalence": round(float(len(pos_i) / max(len(ys_all), 1)), 3), "curve": {}}
        for n in GRID:
            h = n // 2
            if h > len(pos_i) or h > len(neg_i): continue
            aucs, nulls, ident = [], [], []
            for rep in range(REPS):
                rng = np.random.RandomState(1000 * rep + n)
                idx = np.concatenate([rng.choice(pos_i, h, replace=False), rng.choice(neg_i, h, replace=False)])
                ys = ys_all[idx]; Xs = Xs_all[idx]; st = strat_all[idx]
                aucs.append(probe(Xs, ys, rep))
                for pi in range(PERMS_PER_REP):
                    yp = ys.copy()
                    if sname in MIXED:   # permute labels WITHIN cohort
                        for s in set(st):
                            w = np.where(st == s)[0]; yp[w] = rng.permutation(ys[w])
                    else:
                        yp = rng.permutation(ys)
                    if 0 < yp.sum() < len(yp): nulls.append(probe(Xs, yp, rep * 10 + pi))
                if sname in MIXED:
                    S = pd.get_dummies(pd.Series(st)).values.astype(float)
                    if S.shape[1] > 1: ident.append(identity_probe(S, ys, rep))
            cell["curve"][str(n)] = {
                "auc_mean": round(float(np.mean(aucs)), 4), "auc_sd": round(float(np.std(aucs)), 4),
                "null_mean": round(float(np.mean(nulls)), 4), "null_q95": round(float(np.quantile(nulls, 0.95)), 4),
                "cohort_identity_only_auc": round(float(np.mean(ident)), 4) if ident else None}
            print(target, sname, n, cell["curve"][str(n)], flush=True)
        res[target][sname] = cell
json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=2)
print("wrote results.json")

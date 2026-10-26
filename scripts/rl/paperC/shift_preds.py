"""Paper C, items 1 and 4: label-free per-knockdown shift predictors, the list-size-only score and the leave-one-out oracle.
All written as Task F-format preds (constant per perturbation) into $O/preds, plus $O/shift_pred.json (per-pert predictions)
and $O/shift_eval.json (how well each predictor recovers the held-out knockdown's true up-fraction / DE rate).

Predictors of a test knockdown p's up-fraction u(p) (fraction "up" among its changed genes) and DE rate d(p):
- shiftknn_k{10,20,50}: similarity-weighted mean of u / d over the k most similar *training* knockdowns (mean of KG Jaccard and
  gene-summary TF-IDF cosine, as in knn_baseline.py). No gene-level information, no test labels.
- shift_ridge: ridge regression on sparse KG-neighbour-set + TF-IDF features (RidgeCV over alphas, 5-fold on training perts).
- shift_kridge: kernel ridge with the (PSD) mean-similarity matrix as kernel, alpha by 5-fold CV on training perts.
- listsize_only: DE score = number of test pairs listed for p (the benchmark artefact); DIR score 0.5.
- oracle_loo: p's test DE rate / up-fraction over its *other* test pairs (diagnostic upper bound, uses test labels).
"""
import json, random, sys
import numpy as np, pandas as pd
from collections import defaultdict
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from pc_common import *

from pc_kg import feat, features, sims


def kridge_fit_predict(Ktr, ytr, Kte, alphas, seed=0):
    """Kernel ridge (precomputed kernel), alpha by 5-fold CV on the training rows. Returns predictions and chosen alpha."""
    kf = KFold(5, shuffle=True, random_state=seed); best = None
    for a in alphas:
        err = 0.0
        for tr, va in kf.split(Ktr):
            coef = np.linalg.solve(Ktr[np.ix_(tr, tr)] + a * np.eye(len(tr)), ytr[tr] - ytr[tr].mean())
            err += np.mean((Ktr[np.ix_(va, tr)] @ coef + ytr[tr].mean() - ytr[va]) ** 2)
        if best is None or err < best[0]: best = (err, a)
    a = best[1]; coef = np.linalg.solve(Ktr + a * np.eye(len(Ktr)), ytr - ytr.mean())
    return Kte @ coef + ytr.mean(), a


PRED = defaultdict(list); SHIFT = {}; EVAL = {}
for c in CELLS:
    de, dr, dmap = load_cell(c); df = test_frame(c)
    tr = de[de.split == "train"]; te = de[de.split == "test"]
    perts = sorted(set(de.pert)); pidx = {p: i for i, p in enumerate(perts)}
    Xkg, T = features(perts); S = sims(Xkg, T)
    # training-perturbation targets
    d_tr = tr.groupby("pert").label.mean().to_dict()
    ups = defaultdict(list)
    for a, b, l in zip(tr.pert, tr.gene, tr.label):
        if l == 1 and (a, b) in dmap: ups[a].append(dmap[(a, b)])
    u_tr = {p: float(np.mean(v)) for p, v in ups.items()}
    trainp = [p for p in perts if p in d_tr]; testp = sorted(set(te.pert)); itr = np.array([pidx[p] for p in trainp]); ite = np.array([pidx[p] for p in testp])
    utr = np.array([u_tr.get(p, 0.5) for p in trainp]); dtr = np.array([d_tr[p] for p in trainp]); has_u = np.array([p in u_tr for p in trainp])
    # true test shift (for evaluation only)
    d_te = te.groupby("pert").label.mean().to_dict(); ups_te = defaultdict(list)
    for a, b, l in zip(te.pert, te.gene, te.label):
        if l == 1 and (a, b) in dmap: ups_te[a].append(dmap[(a, b)])
    u_te = {p: float(np.mean(v)) for p, v in ups_te.items()}
    preds = {}
    # kNN shift
    Ste = S[np.ix_(ite, itr)]
    for k in (10, 20, 50):
        up = np.zeros(len(testp)); dd = np.zeros(len(testp))
        for i in range(len(testp)):
            top = np.argsort(-Ste[i])[:k]; w = np.maximum(Ste[i, top], 1e-6)
            dd[i] = np.average(dtr[top], weights=w); tu = top[has_u[top]]; up[i] = np.average(utr[tu], weights=np.maximum(Ste[i, tu], 1e-6)) if len(tu) else 0.5
        preds[f"shiftknn_k{k}"] = (dd, up)
    # ridge on sparse KG + text features
    Xf = sparse.hstack([Xkg, T]).tocsr(); alphas = np.logspace(-2, 4, 13)
    rd = RidgeCV(alphas=alphas, cv=5).fit(Xf[itr], dtr); ru = RidgeCV(alphas=alphas, cv=5).fit(Xf[itr][has_u], utr[has_u])
    preds["shift_ridge"] = (rd.predict(Xf[ite]), ru.predict(Xf[ite])); EVAL[f"{c}_ridge_alpha"] = [float(rd.alpha_), float(ru.alpha_)]
    # kernel ridge on the mean-similarity kernel
    Ktr = S[np.ix_(itr, itr)]; Kte = S[np.ix_(ite, itr)]
    pd_, ad = kridge_fit_predict(Ktr, dtr, Kte, alphas); pu_, au = kridge_fit_predict(Ktr[np.ix_(has_u, has_u)], utr[has_u], Kte[:, has_u], alphas)
    preds["shift_kridge"] = (pd_, pu_); EVAL[f"{c}_kridge_alpha"] = [float(ad), float(au)]
    # evaluation of the shift predictors against the true held-out shift
    tu = np.array([u_te.get(p, np.nan) for p in testp]); td = np.array([d_te[p] for p in testp]); ok = ~np.isnan(tu)
    for nm, (dd, up) in preds.items():
        EVAL[f"{c}_{nm}"] = {"spearman_up": float(spearmanr(up[ok], tu[ok]).correlation), "spearman_de": float(spearmanr(dd, td).correlation),
                             "n_perts": int(len(testp)), "n_perts_with_up": int(ok.sum())}
    SHIFT[c] = {p: {nm: [float(v[0][i]), float(v[1][i])] for nm, v in preds.items()} for i, p in enumerate(testp)}
    # emit preds (constant per perturbation)
    tp = {p: i for i, p in enumerate(testp)}
    for nm, (dd, up) in preds.items():
        PRED[nm] += [{"cell": c, "pert": p, "gene": g, "label": lab, "de": int(d), "dir": (None if di < 0 else int(di)), "pde": dd[tp[p]], "pdir": up[tp[p]]}
                     for p, g, lab, d, di in zip(df.pert, df.gene, ["x"] * len(df), df.de, df["dir"])]
    # list size only
    sz = df.groupby("pert").size().to_dict(); mx = max(sz.values())
    PRED["listsize_only"] += [{"cell": c, "pert": p, "gene": g, "label": "x", "de": int(d), "dir": (None if di < 0 else int(di)), "pde": sz[p] / (mx + 1), "pdir": 0.5}
                              for p, g, d, di in zip(df.pert, df.gene, df.de, df["dir"])]
    # leave-one-out oracle (uses test labels; diagnostic only)
    nde = df.groupby("pert").de.sum().to_dict(); npair = sz; nup = defaultdict(int); ndir = defaultdict(int)
    for p, d, di in zip(df.pert, df.de, df["dir"]):
        if d == 1 and di >= 0: ndir[p] += 1; nup[p] += int(di == 1)
    rows = []
    for p, g, d, di in zip(df.pert, df.gene, df.de, df["dir"]):
        pde = (nde[p] - d) / max(npair[p] - 1, 1)
        if d == 1 and di >= 0: n = ndir[p] - 1; pdir = (nup[p] - int(di == 1)) / n if n > 0 else 0.5
        else: pdir = nup[p] / ndir[p] if ndir[p] else 0.5
        rows.append({"cell": c, "pert": p, "gene": g, "label": "x", "de": int(d), "dir": (None if di < 0 else int(di)), "pde": pde, "pdir": pdir})
    PRED["oracle_loo"] += rows
    print(c, {k: v for k, v in EVAL.items() if k.startswith(c)}, flush=True)

for tag, rows in PRED.items(): write_preds(tag, rows)
json.dump(SHIFT, open(O + "/shift_pred.json", "w")); json.dump(EVAL, open(O + "/shift_eval.json", "w"), indent=1)
print("SHIFT DONE", list(PRED))

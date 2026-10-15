"""P32 payoff test: do imputed report fields add to image + CNV on SWG (150 patients, release endpoint)?
Text-surrogate arm = CV logistic (release folds) on the imputed field vector; late fusion = fold-local z-mean.
3-way (image, cnv, fields) vs 2-way (image, cnv) on ALL 150 patients, patient level, 2,000 bootstraps.
Env: P32DIRS (comma list of dirs with swg_imputed_fields.csv), OUTDIR."""
import glob, json, os, numpy as np, pandas as pd, sys
print('p32_swg_fuse start', flush=True)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score as _sk_auc
from scipy.stats import rankdata
def roc_auc_score(y, s):   # rank-based AUROC, ~50x faster than sklearn for the bootstrap loops
    y = np.asarray(y); r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
F = "/mnt/scratche/slow/fmlab/zuberi01/phd/barretts_retraining/barretts_training/analysis/chapter1_lgd2_final_pre_event_20260713_final"; OUT = os.environ.get("OUTDIR", "."); NB = 2000
imp = pd.concat([pd.read_csv(p + "/swg_imputed_fields.csv", index_col=0) for p in os.environ["P32DIRS"].split(",")], axis=1); imp = imp.loc[:, ~imp.columns.duplicated()].dropna(axis=1, how="all"); imp.index = imp.index.astype(str)
print("imputed table", imp.shape, flush=True)
man = pd.read_csv(F + "/training_manifest.csv", dtype=str).set_index("sample_id"); man = man.loc[man.index.intersection(imp.index)]
def oof(fam): return pd.concat([pd.read_csv(f, dtype={"sample_id": str}) for f in glob.glob(f"{F}/training_final_nested_cv_v1/{fam}/fold*/outer_test_predictions.csv")]).set_index("sample_id").y_prob
d = man.assign(img=oof("image_only").reindex(man.index).values, cnv=oof("cnv_only").reindex(man.index).values, y=man.y_progressor.astype(int).values, fold=man.fold_id_rep01.astype(int).values)
X = imp.loc[d.index].values; folds = d.fold.values
def cvlog(X, y):
    p = np.zeros(len(y))
    for f in np.unique(folds):
        te = folds == f; tr = ~te; mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9; p[te] = LogisticRegression(C=1.0, max_iter=3000).fit((X[tr] - mu) / sd, y[tr]).predict_proba((X[te] - mu) / sd)[:, 1]
    return p
def zf(v):
    o = np.zeros(len(v))
    for f in np.unique(folds): te = folds == f; o[te] = (v[te] - v[te].mean()) / (v[te].std() + 1e-9)
    return o
d["fields"] = cvlog(X, d.y.values); d["fuse2"] = (zf(d.img.values) + zf(d.cnv.values)) / 2; d["fuse3"] = (zf(d.img.values) + zf(d.cnv.values) + zf(d.fields.values)) / 3; d["fuse2_prob"] = (d.img + d.cnv) / 2
for c in imp.columns: d[f"f_{c}"] = imp.loc[d.index, c].values
g = d.groupby("patient_id"); y = g.y.max().values; P = {k: g[k].max().values for k in ["img", "cnv", "fields", "fuse2", "fuse2_prob", "fuse3"] + [f"f_{c}" for c in imp.columns]}
rng = np.random.RandomState(0); n = len(y); B = []
if n < 10 or len(set(y)) < 2: sys.exit(f"degenerate patient set: n={n} classes={set(y)} (index mismatch between imputed table and manifest?)")
while len(B) < NB:
    s = rng.choice(n, n)
    if len(set(y[s])) < 2: continue
    B.append(s)
ci = lambda a, b=None: [round(float(np.percentile([roc_auc_score(y[s], a[s]) - (roc_auc_score(y[s], b[s]) if b is not None else 0) for s in B], q)), 4) for q in (2.5, 97.5)]
res = {"n_patients": int(n), "pos": int(y.sum()), "fields_used": list(imp.columns), **{k: {"auroc": round(float(roc_auc_score(y, v)), 4), "ci": ci(v)} for k, v in P.items()},
       "delta_fuse3_minus_fuse2": ci(P["fuse3"], P["fuse2"]), "delta_fuse3_minus_img": ci(P["fuse3"], P["img"]), "delta_fields_minus_img": ci(P["fields"], P["img"]),
       "sample_level": {k: round(float(roc_auc_score(d.y, d[k])), 4) for k in ["img", "cnv", "fields", "fuse2", "fuse3"]}}
os.makedirs(OUT, exist_ok=True); json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=1); print(json.dumps(res, indent=None))

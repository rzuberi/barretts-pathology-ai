"""P32 pass-2 controls for the 3-way fusion gain on SWG (150 patients): (a) which imputed field carries it (single-field
3-way fusions and leave-one-out); (b) label-permutation p for delta fuse3 - fuse2 with all arms fixed; (c) ENSEMBLE
CONTROL: replace the imputed-field arm with a second image-only progression model on the same slides (the rep02 release
image_only OOF) - if fuse(img, cnv, img_rep02) gains as much, the gain is ensembling two image models, not report fields;
(d) fuse2 + imputed grade only. Rank-based AUROC, patient level (max over samples), 2,000 boots / perms, seed 0."""
import glob, json, os, numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
F = "/mnt/scratche/slow/fmlab/zuberi01/phd/barretts_retraining/barretts_training/analysis/chapter1_lgd2_final_pre_event_20260713_final"
R = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis/feasibility/runs"; OUT = os.environ.get("OUTDIR", "."); NB = 2000
def auc(y, s): r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1; return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
imp = pd.read_csv(R + "/p32b_fields/output/swg_imputed_fields.csv", index_col=0); imp.index = imp.index.astype(str)
man = pd.read_csv(F + "/training_manifest.csv", dtype=str).set_index("sample_id").loc[imp.index]
def oof(root, fam): return pd.concat([pd.read_csv(f, dtype={"sample_id": str}) for f in glob.glob(f"{root}/{fam}/fold*/outer_test_predictions.csv")]).set_index("sample_id").y_prob
d = man.assign(img=oof(F + "/training_final_nested_cv_v1", "image_only").reindex(man.index).values, cnv=oof(F + "/training_final_nested_cv_v1", "cnv_only").reindex(man.index).values,
               img2=oof(F + "_rep02/training_rep02_nested_cv", "image_only").reindex(man.index).values, y=man.y_progressor.astype(int).values, fold=man.fold_id_rep01.astype(int).values)
folds = d.fold.values
def zf(v):
    o = np.zeros(len(v))
    for f in np.unique(folds): te = folds == f; o[te] = (v[te] - v[te].mean()) / (v[te].std() + 1e-9)
    return o
def cvlog(X, y):
    p = np.zeros(len(y))
    for f in np.unique(folds):
        te = folds == f; tr = ~te; mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9; p[te] = LogisticRegression(C=1.0, max_iter=3000).fit((X[tr] - mu) / sd, y[tr]).predict_proba((X[te] - mu) / sd)[:, 1]
    return p
cols = list(imp.columns); Ximp = imp.loc[d.index].values
def fuse(*arms): return sum(zf(a) for a in arms) / len(arms)
arms = {"img": d.img.values, "cnv": d.cnv.values, "img2_rep02": d.img2.values, "fields_all": cvlog(Ximp, d.y.values)}
for c in cols: arms[f"field_{c}"] = imp.loc[d.index, c].values
for c in cols: arms[f"fields_without_{c}"] = cvlog(imp.loc[d.index, [k for k in cols if k != c]].values, d.y.values)
S = {"fuse2": fuse(arms["img"], arms["cnv"]), "fuse3_fields": fuse(arms["img"], arms["cnv"], arms["fields_all"]), "fuse3_img2_control": fuse(arms["img"], arms["cnv"], arms["img2_rep02"]),
     "fuse3_grade_only": fuse(arms["img"], arms["cnv"], arms["field_grade_LGDplus"]), "fuse3_inflam_only": fuse(arms["img"], arms["cnv"], arms["field_inflammation_mod_severe"]),
     "fuse4_fields_img2": fuse(arms["img"], arms["cnv"], arms["fields_all"], arms["img2_rep02"])}
for c in cols: S[f"fuse3_fields_without_{c}"] = fuse(arms["img"], arms["cnv"], arms[f"fields_without_{c}"])
g = d.assign(**S, **{k: v for k, v in arms.items()}).groupby("patient_id"); y = g.y.max().values; P = {k: g[k].max().values for k in list(S) + list(arms)}
rng = np.random.RandomState(0); n = len(y); B = [rng.choice(n, n) for _ in range(NB)]; B = [s for s in B if len(set(y[s])) > 1]
ci = lambda a, b=None: [round(float(np.percentile([auc(y[s], a[s]) - (auc(y[s], b[s]) if b is not None else 0) for s in B], q)), 4) for q in (2.5, 97.5)]
perm = [rng.permutation(n) for _ in range(NB)]
def perm_p(a, b):
    obs = auc(y, a) - auc(y, b); null = np.array([auc(y[ix], a) - auc(y[ix], b) for ix in perm]); return round(float((1 + (null >= obs).sum()) / (NB + 1)), 5)
res = {"n_patients": int(n), "pos": int(y.sum()), "aurocs": {k: round(auc(y, v), 4) for k, v in P.items()},
       "delta_fuse3_fields_minus_fuse2": {"delta": round(auc(y, P["fuse3_fields"]) - auc(y, P["fuse2"]), 4), "ci": ci(P["fuse3_fields"], P["fuse2"]), "perm_p": perm_p(P["fuse3_fields"], P["fuse2"])},
       "CONTROL_delta_fuse3_img2_minus_fuse2": {"delta": round(auc(y, P["fuse3_img2_control"]) - auc(y, P["fuse2"]), 4), "ci": ci(P["fuse3_img2_control"], P["fuse2"]), "perm_p": perm_p(P["fuse3_img2_control"], P["fuse2"])},
       "delta_fuse3_fields_minus_fuse3_img2": {"delta": round(auc(y, P["fuse3_fields"]) - auc(y, P["fuse3_img2_control"]), 4), "ci": ci(P["fuse3_fields"], P["fuse3_img2_control"])},
       "delta_fuse3_grade_only_minus_fuse2": {"delta": round(auc(y, P["fuse3_grade_only"]) - auc(y, P["fuse2"]), 4), "ci": ci(P["fuse3_grade_only"], P["fuse2"])},
       "delta_fuse4_minus_fuse3_img2": {"delta": round(auc(y, P["fuse4_fields_img2"]) - auc(y, P["fuse3_img2_control"]), 4), "ci": ci(P["fuse4_fields_img2"], P["fuse3_img2_control"])},
       "leave_one_field_out_fuse3": {c: round(auc(y, P[f"fuse3_fields_without_{c}"]), 4) for c in cols},
       "imputed_field_alone_vs_img": {c: {"auroc": round(auc(y, P[f"field_{c}"]), 4), "delta_vs_img_ci": ci(P[f"field_{c}"], P["img"])} for c in cols}}
os.makedirs(OUT, exist_ok=True); json.dump(res, open(os.path.join(OUT, "p32_fuse_controls.json"), "w"), indent=1); print(json.dumps(res, indent=None))

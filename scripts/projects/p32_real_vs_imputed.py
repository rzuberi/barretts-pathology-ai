"""P32 check: on SWG samples whose clinical report is in the Barrett's DB (428 slides / 65 patients), compare the REAL
report fields (P31 v2 on the DB report) with the IMPUTED fields (ERIN heads on the slide): per-field AUROC of imputed
score vs real field; and the fusion test on this subset with real fields vs imputed fields vs neither.
Env: DB_DIRS (colon list of p31 db shard output dirs), OUTDIR."""
import glob, json, os, re, numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
F = "/mnt/scratche/slow/fmlab/zuberi01/phd/barretts_retraining/barretts_training/analysis/chapter1_lgd2_final_pre_event_20260713_final"; E = "/mnt/scratche/slow/fmlab/zuberi01/barretts_db_export"; R = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis/feasibility/runs"; OUT = os.environ.get("OUTDIR", "."); NB = 2000
def auc(y, s): r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1; return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if 0 < n1 < len(y) else float("nan")
def stem(x): x = str(x).lower().strip(); x = re.sub(r"[\s_].*$", "", x); return re.sub(r"[^a-z0-9]", "", x)
real = pd.DataFrame([json.loads(l) for d in os.environ["DB_DIRS"].split(":") for f in glob.glob(d + "/fields_*_shard*.jsonl") for l in open(f)]).drop_duplicates("CaseName").set_index("CaseName")
sm = pd.read_parquet(E + "/swg_matched_reports_v2.parquet"); rows = []
for r in sm.itertuples():
    for pid in str(r.swg_path_ids).split(";"):
        if pid.strip(): rows.append({"stem": stem(pid), "rid": str(r.pathology_text_id)})
rep = pd.DataFrame(rows).drop_duplicates("stem").set_index("stem")
imp = pd.read_csv(R + "/p32b_fields/output/swg_imputed_fields.csv", index_col=0); imp.index = imp.index.astype(str)
man = pd.read_csv(F + "/training_manifest.csv", dtype=str).set_index("sample_id").loc[imp.index]; coh = pd.read_csv(F + "/pre_event_cohort.csv", dtype=str).set_index("SampleID")
man["rid"] = coh.BiopsyID_real.reindex(man.index).map(stem).map(rep.rid); d = man[man.rid.isin(real.index)].copy(); print("matched samples", len(d), "patients", d.patient_id.nunique(), flush=True)
RF = {"grade_LGDplus": real.grade.isin(["LGD", "HGD", "CANCER"]), "im_present": real.intestinal_metaplasia.eq("present"), "treatment_effect": real.treatment_effect.eq("yes"), "ulceration": real.ulceration_or_erosion.eq("yes"), "squamous_only": real.squamous_only.eq("yes"), "inflammation_mod_severe": real.inflammation.isin(["moderate", "severe"])}
res = {"n_samples": int(len(d)), "n_patients": int(d.patient_id.nunique()), "imputed_vs_real_field_auroc": {}}
for c in imp.columns:
    yr = RF[c].reindex(d.rid).astype(int).values; res["imputed_vs_real_field_auroc"][c] = {"real_pos": int(yr.sum()), "auroc": round(auc(yr, imp.loc[d.index, c].values), 4)}
def oof(fam): return pd.concat([pd.read_csv(f, dtype={"sample_id": str}) for f in glob.glob(f"{F}/training_final_nested_cv_v1/{fam}/fold*/outer_test_predictions.csv")]).set_index("sample_id").y_prob
d["img"] = oof("image_only").reindex(d.index).values; d["cnv"] = oof("cnv_only").reindex(d.index).values; d["y"] = d.y_progressor.astype(int); folds = d.fold_id_rep01.astype(int).values
def zf(v):
    o = np.zeros(len(v))
    for f in np.unique(folds): te = folds == f; o[te] = (v[te] - v[te].mean()) / (v[te].std() + 1e-9)
    return o
Xr = np.column_stack([RF[c].reindex(d.rid).astype(float).values for c in imp.columns]); Xi = imp.loc[d.index].values
def cvlog(X, y):
    p = np.zeros(len(y))
    for f in np.unique(folds):
        te = folds == f; tr = ~te
        if len(set(y[tr])) < 2: p[te] = y[tr].mean(); continue
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9; p[te] = LogisticRegression(C=1.0, max_iter=3000).fit((X[tr] - mu) / sd, y[tr]).predict_proba((X[te] - mu) / sd)[:, 1]
    return p
d["real_f"] = cvlog(Xr, d.y.values); d["imp_f"] = cvlog(Xi, d.y.values); d["real_grade"] = Xr[:, list(imp.columns).index("grade_LGDplus")]; d["imp_grade"] = Xi[:, list(imp.columns).index("grade_LGDplus")]
d["fuse2"] = (zf(d.img.values) + zf(d.cnv.values)) / 2; d["f3_real"] = (zf(d.img.values) + zf(d.cnv.values) + zf(d.real_f.values)) / 3; d["f3_imp"] = (zf(d.img.values) + zf(d.cnv.values) + zf(d.imp_f.values)) / 3
d["f3_real_grade"] = (zf(d.img.values) + zf(d.cnv.values) + zf(d.real_grade.values)) / 3; d["f3_imp_grade"] = (zf(d.img.values) + zf(d.cnv.values) + zf(d.imp_grade.values)) / 3
g = d.groupby("patient_id"); y = g.y.max().values; P = {k: g[k].max().values for k in ("img", "cnv", "real_f", "imp_f", "real_grade", "imp_grade", "fuse2", "f3_real", "f3_imp", "f3_real_grade", "f3_imp_grade")}
rng = np.random.RandomState(0); B = [s for s in (rng.choice(len(y), len(y)) for _ in range(NB)) if len(set(y[s])) > 1]
ci = lambda a, b=None: [round(float(np.percentile([auc(y[s], a[s]) - (auc(y[s], b[s]) if b is not None else 0) for s in B], q)), 4) for q in (2.5, 97.5)]
res["progression_matched_subset"] = {"n_patients": int(len(y)), "pos": int(y.sum()), **{k: {"auroc": round(auc(y, v), 4), "ci": ci(v)} for k, v in P.items()}, "delta_f3_real_minus_fuse2": ci(P["f3_real"], P["fuse2"]), "delta_f3_imp_minus_fuse2": ci(P["f3_imp"], P["fuse2"]), "delta_f3_imp_minus_f3_real": ci(P["f3_imp"], P["f3_real"]), "delta_imp_grade_minus_real_grade": ci(P["imp_grade"], P["real_grade"])}
os.makedirs(OUT, exist_ok=True); json.dump(res, open(os.path.join(OUT, "p32_real_vs_imputed.json"), "w"), indent=1); print(json.dumps(res, indent=None))

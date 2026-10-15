"""Overnight fluke-checks for the P32 SWG result (CPU; uses saved OOF/imputed CSVs).
1 selection-adjusted permutation: max over the 6 single-field 3-way fusions of (fuse3_field - fuse2), labels permuted;
2 replication on the rep02 split: fuse3 with rep02 image/cnv OOF and rep02 folds;
3 imputed grade vs the Barrett's-DB CONFIRMED code (the better human anchor) on matched SWG reports, and vs spreadsheet;
4 calibration of the transferred grade head on SWG (slope/intercept vs pathologist LGD+), imputed positive rate vs ERIN;
5 arms from the overnight GPU jobs if present: repeat-fold grade head (FOLD_SEED 1), permuted-label ERIN head (random head
  control), SWG-trained image on 0.5um feats (img05) -> fuse2' = img05 + cnv, fuse3' = img05 + cnv + grade head.
Env: OUTDIR, plus optional REP_DIR, PERM_DIR, IMG05_DIR."""
import glob, json, os, re, numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score
F = "/mnt/scratche/slow/fmlab/zuberi01/phd/barretts_retraining/barretts_training/analysis/chapter1_lgd2_final_pre_event_20260713_final"
E = "/mnt/scratche/slow/fmlab/zuberi01/barretts_db_export"; T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"; R = T + "/feasibility/runs"; OUT = os.environ.get("OUTDIR", "."); NB = 2000
def auc(y, s): r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1; return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
imp = pd.read_csv(R + "/p32b_fields/output/swg_imputed_fields.csv", index_col=0); imp.index = imp.index.astype(str); cols = list(imp.columns)
man = pd.read_csv(F + "/training_manifest.csv", dtype=str).set_index("sample_id").loc[imp.index]; coh = pd.read_csv(F + "/pre_event_cohort.csv", dtype=str).set_index("SampleID")
def oof(root, fam): return pd.concat([pd.read_csv(f, dtype={"sample_id": str}) for f in glob.glob(f"{root}/{fam}/fold*/outer_test_predictions.csv")]).set_index("sample_id").y_prob
def zf(v, folds):
    o = np.zeros(len(v))
    for f in np.unique(folds): te = folds == f; o[te] = (v[te] - v[te].mean()) / (v[te].std() + 1e-9)
    return o
def cvlog(X, y, folds):
    p = np.zeros(len(y))
    for f in np.unique(folds):
        te = folds == f; tr = ~te; mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9; p[te] = LogisticRegression(C=1.0, max_iter=3000).fit((X[tr] - mu) / sd, y[tr]).predict_proba((X[te] - mu) / sd)[:, 1]
    return p
def patient(d, cols_): g = d.groupby("patient_id"); return g.y.max().values, {c: g[c].max().values for c in cols_}
def boots(y): rng = np.random.RandomState(0); return [s for s in (rng.choice(len(y), len(y)) for _ in range(NB)) if len(set(y[s])) > 1]
def ci(y, B, a, b=None): return [round(float(np.percentile([auc(y[s], a[s]) - (auc(y[s], b[s]) if b is not None else 0) for s in B], q)), 4) for q in (2.5, 97.5)]
res = {}
# ---- rep01 base
y_s = man.y_progressor.astype(int).values; f1 = man.fold_id_rep01.astype(int).values
img1 = oof(F + "/training_final_nested_cv_v1", "image_only").reindex(man.index).values; cnv1 = oof(F + "/training_final_nested_cv_v1", "cnv_only").reindex(man.index).values
d1 = man.assign(y=y_s, img=img1, cnv=cnv1, fuse2=(zf(img1, f1) + zf(cnv1, f1)) / 2)
for c in cols: d1[f"f3_{c}"] = (zf(img1, f1) + zf(cnv1, f1) + zf(imp.loc[man.index, c].values, f1)) / 3
d1["f3_all"] = (zf(img1, f1) + zf(cnv1, f1) + zf(cvlog(imp.loc[man.index].values, y_s, f1), f1)) / 3
y, P = patient(d1, ["img", "cnv", "fuse2", "f3_all"] + [f"f3_{c}" for c in cols]); B = boots(y)
# 1 selection-adjusted over the 6 single-field fusions
obs = {c: auc(y, P[f"f3_{c}"]) - auc(y, P["fuse2"]) for c in cols}; best = max(obs, key=obs.get); t_obs = obs[best]
rng = np.random.RandomState(0); perm = [rng.permutation(len(y)) for _ in range(NB)]
null = np.array([max(auc(y[ix], P[f"f3_{c}"]) - auc(y[ix], P["fuse2"]) for c in cols) for ix in perm])
null_all = np.array([auc(y[ix], P["f3_all"]) - auc(y[ix], P["fuse2"]) for ix in perm]); obs_all = auc(y, P["f3_all"]) - auc(y, P["fuse2"])
res["1_selection_adjusted_over_fields"] = {"observed_delta_by_field": {c: round(v, 4) for c, v in obs.items()}, "selected_field": best, "t_obs": round(t_obs, 4), "p_selection_adjusted": round(float((1 + (null >= t_obs).sum()) / (NB + 1)), 5),
    "p_unadjusted_selected": round(float((1 + (np.array([auc(y[ix], P[f"f3_{best}"]) - auc(y[ix], P["fuse2"]) for ix in perm]) >= t_obs).sum()) / (NB + 1)), 5), "f3_all_fields_delta": round(obs_all, 4), "p_f3_all": round(float((1 + (null_all >= obs_all).sum()) / (NB + 1)), 5)}
# 2 rep02 replication
R2 = F + "_rep02/training_rep02_nested_cv"; m2 = pd.read_csv(F + "_rep02/training_manifest_v2.csv", dtype={"sample_id": str}).set_index("sample_id").loc[man.index]; f2 = m2.fold_id_rep01.astype(int).values
img2 = oof(R2, "image_only").reindex(man.index).values; cnv2 = oof(R2, "cnv_only").reindex(man.index).values
d2 = man.assign(y=y_s, img=img2, cnv=cnv2, fuse2=(zf(img2, f2) + zf(cnv2, f2)) / 2, f3_grade=(zf(img2, f2) + zf(cnv2, f2) + zf(imp.loc[man.index, "grade_LGDplus"].values, f2)) / 3, f3_all=(zf(img2, f2) + zf(cnv2, f2) + zf(cvlog(imp.loc[man.index].values, y_s, f2), f2)) / 3)
y2, P2 = patient(d2, ["img", "cnv", "fuse2", "f3_grade", "f3_all"])
res["2_rep02_split_replication"] = {k: round(auc(y2, v), 4) for k, v in P2.items()}; res["2_rep02_split_replication"]["delta_f3_all_minus_fuse2"] = ci(y2, B, P2["f3_all"], P2["fuse2"]); res["2_rep02_split_replication"]["delta_f3_grade_minus_fuse2"] = ci(y2, B, P2["f3_grade"], P2["fuse2"])
res["2_rep02_split_replication"]["note"] = "ERIN heads unchanged (trained on ERIN); SWG image/cnv arms and folds from the rep02 split"
# 3 imputed grade vs DB confirmed code and vs spreadsheet
def stem(x): x = str(x).lower().strip(); x = re.sub(r"[\s_].*$", "", x); return re.sub(r"[^a-z0-9]", "", x)
sm = pd.read_parquet(E + "/swg_matched_reports_v2.parquet"); rows = []
for r in sm.itertuples():
    for pid in str(r.swg_path_ids).split(";"):
        if pid.strip(): rows.append({"stem": stem(pid), "conf": pd.to_numeric(r.highestgradedysconf, errors="coerce")})
rep = pd.DataFrame(rows).drop_duplicates("stem").set_index("stem"); MAP = {2: 0, 3: 1, 4: 2, 5: 3, 6: 4, 8: 4}
stems = coh.BiopsyID_real.reindex(man.index).map(stem); conf = stems.map(rep.conf); ok = conf.isin(MAP).values
lab = pd.to_numeric(coh.Label.reindex(man.index), errors="coerce").fillna(0).values; g = imp.loc[man.index, "grade_LGDplus"].values
res["3_imputed_grade_anchors"] = {"vs_spreadsheet_LGDplus_auroc_all707": round(auc((lab >= 2).astype(int), g), 4), "n_with_db_code": int(ok.sum()),
    "vs_db_confirmed_LGDplus_auroc": round(auc((conf[ok].map(MAP).values >= 2).astype(int), g[ok]), 4) if ok.sum() > 20 else None,
    "vs_spreadsheet_on_same_subset": round(auc((lab[ok] >= 2).astype(int), g[ok]), 4) if ok.sum() > 20 else None,
    "spreadsheet_vs_db_code_two_tier_on_subset": round(float(((lab[ok] >= 2) == (conf[ok].map(MAP).values >= 2)).mean()), 4) if ok.sum() > 20 else None}
# 4 calibration / prevalence
lp = np.log(np.clip(g, 1e-6, 1 - 1e-6) / (1 - np.clip(g, 1e-6, 1 - 1e-6))); lr = LogisticRegression(C=1e6).fit(lp[:, None], (lab >= 2).astype(int))
erin = pd.read_csv(R + "/p32b_fields/output/oof_grade_LGDplus.csv"); res["4_transfer_calibration"] = {"swg_imputed_mean_p": round(float(g.mean()), 4), "swg_frac_p_gt_0.5": round(float((g > 0.5).mean()), 4), "swg_spreadsheet_LGDplus_rate": round(float((lab >= 2).mean()), 4),
    "erin_oof_mean_p": round(float(erin.p.mean()), 4), "erin_label_LGDplus_rate": round(float(erin.y.mean()), 4), "calib_slope_vs_spreadsheet": round(float(lr.coef_[0][0]), 3), "calib_intercept": round(float(lr.intercept_[0]), 3)}
# 5 optional overnight arms
extra = {}
for tag, env in (("grade_repeat_foldseed1", "REP_DIR"), ("grade_permuted_labels", "PERM_DIR")):
    dd = os.environ.get(env)
    if dd and os.path.exists(dd + "/swg_imputed_fields.csv"):
        x = pd.read_csv(dd + "/swg_imputed_fields.csv", index_col=0); x.index = x.index.astype(str); v = x.loc[man.index, "grade_LGDplus"].values
        d1[f"f3_{tag}"] = (zf(img1, f1) + zf(cnv1, f1) + zf(v, f1)) / 3; d1[f"arm_{tag}"] = v; extra[tag] = True
if os.environ.get("IMG05_DIR") and os.path.exists(os.environ["IMG05_DIR"] + "/swg_img05_oof.csv"):
    x = pd.read_csv(os.environ["IMG05_DIR"] + "/swg_img05_oof.csv", dtype={"sample_id": str}).set_index("sample_id").p.reindex(man.index).values
    d1["img05"] = x; d1["fuse2_05"] = (zf(x, f1) + zf(cnv1, f1)) / 2; d1["f3_05_grade"] = (zf(x, f1) + zf(cnv1, f1) + zf(imp.loc[man.index, "grade_LGDplus"].values, f1)) / 3; extra["img05"] = True
if extra:
    cols5 = [c for c in d1.columns if c.startswith(("f3_grade_", "arm_", "img05", "fuse2_05", "f3_05"))]; y5, P5 = patient(d1, cols5 + ["fuse2", "img"])
    res["5_overnight_arms"] = {k: {"auroc": round(auc(y5, v), 4), "delta_vs_fuse2": ci(y5, B, v, P5["fuse2"])} for k, v in P5.items() if k not in ("fuse2", "img")}
    if "img05" in extra: res["5_overnight_arms"]["img05_minus_release_img"] = ci(y5, B, P5["img05"], P5["img"]); res["5_overnight_arms"]["f3_05_grade_minus_fuse2_05"] = ci(y5, B, P5["f3_05_grade"], P5["fuse2_05"])
os.makedirs(OUT, exist_ok=True); json.dump(res, open(os.path.join(OUT, "p32_checks.json"), "w"), indent=1); print(json.dumps(res, indent=None))

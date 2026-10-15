"""Closeout item H (pre-specified in docs/closeout_for_review.md before the heads were trained): non-grade ensemble control.
Reads the leak-free ERIN heads for treatment_effect and im_present (and a leak-free permuted-label grade head) trained by
scripts/projects/p32_fields_from_image.py with EXCLUDE_PATIENTS, transfers each to SWG as the third arm of the fold-local
z-mean fusion, and reports head alone, image + CNV + head, and gain vs image + CNV, side by side with the canonical grade
head and the two shuffled-label heads. Same conventions as co_main.py. Env: H_TREAT_DIR, H_IM_DIR, H_PERM_DIR."""
import glob, json, os, numpy as np, pandas as pd
from scipy.stats import rankdata
F = "/mnt/scratche/slow/fmlab/zuberi01/phd/barretts_retraining/barretts_training/analysis/chapter1_lgd2_final_pre_event_20260713_final"
T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"; R = T + "/feasibility/runs"; AGG = os.environ.get("OUTDIR", T + "/results/closeout"); NB = 2000; SEED = 0
def auc(y, s): y = np.asarray(y).astype(int); r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1; return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
def r3(x): return round(float(x), 3)
man = pd.read_csv(F + "/training_manifest.csv", dtype=str).set_index("sample_id"); man["y"] = man.y_progressor.astype(int); folds = man.fold_id_rep01.astype(int).values
def oof(fam): return pd.concat([pd.read_csv(f, dtype={"sample_id": str}) for f in glob.glob(f"{F}/training_final_nested_cv_v1/{fam}/fold*/outer_test_predictions.csv")]).set_index("sample_id").y_prob.reindex(man.index).values
img, cnv = oof("image_only"), oof("cnv_only")
def imp(d, c): x = pd.read_csv(d + "/swg_imputed_fields.csv", index_col=0); x.index = x.index.astype(str); return x.reindex(man.index)[c].values
def zf(v):
    v = np.asarray(v, float); o = np.zeros(len(v))
    for f in np.unique(folds): te = folds == f; o[te] = (v[te] - v[te].mean()) / (v[te].std() + 1e-9)
    return o
def fuse(*a): return sum(zf(x) for x in a) / len(a)
heads = {"grade_LGDplus_leakfree_CANONICAL": (R + "/p32_head_noov/output", "grade_LGDplus"), "inflammation_mod_severe_leakfree": (R + "/p32_head_noov/output", "inflammation_mod_severe"),
         "treatment_effect_leakfree_NEW": (os.environ.get("H_TREAT_DIR"), "treatment_effect"), "im_present_leakfree_NEW": (os.environ.get("H_IM_DIR"), "im_present"),
         "grade_permuted_labels_leakfree_NEW": (os.environ.get("H_PERM_DIR"), "grade_LGDplus"), "grade_permuted_labels_incl_overlap": (R + "/p32_head_perm/output", "grade_LGDplus"),
         "treatment_effect_incl_overlap_pass2": (R + "/p32b_fields/output", "treatment_effect"), "im_present_incl_overlap_pass2": (R + "/p32b_fields/output", "im_present")}
cols = {"fuse2": fuse(img, cnv)}; erin = {}
for k, (d, c) in heads.items():
    if d and os.path.exists(d + "/swg_imputed_fields.csv"):
        cols[k + "_head"] = imp(d, c); cols[k + "_fuse3"] = fuse(img, cnv, cols[k + "_head"]); rj = json.load(open(d + "/results.json")); erin[k] = rj["fields"].get(c, {}); erin[k]["n_cases_training"] = rj["_meta"]["n_cases"]; erin[k]["excluded_file"] = rj["_meta"].get("excluded_patients_file"); erin[k]["perm_seed"] = rj["_meta"].get("perm_seed")
g = man.assign(**cols).groupby("patient_id"); y = g.y.max().values.astype(int); P = {c: g[c].max().values for c in cols}
rng = np.random.RandomState(SEED); B = []
while len(B) < NB:
    s = rng.choice(len(y), len(y))
    if len(set(y[s])) > 1: B.append(s)
ci = lambda a, b=None: [r3(np.percentile([auc(y[s], a[s]) - (auc(y[s], b[s]) if b is not None else 0) for s in B], q)) for q in (2.5, 97.5)]
perm = [rng.permutation(len(y)) for _ in range(NB)]
rows = []
for k in heads:
    if k + "_head" not in P: rows.append({"head": k, "status": "NOT AVAILABLE (run missing)"}); continue
    gain = auc(y, P[k + "_fuse3"]) - auc(y, P["fuse2"]); null = np.array([auc(y[ix], P[k + "_fuse3"]) - auc(y[ix], P["fuse2"]) for ix in perm])
    rows.append({"head": k, "n": len(y), "events": int(y.sum()), "erin_oof_auroc": erin[k].get("auroc"), "erin_n_pos": [erin[k].get("n"), erin[k].get("pos")], "erin_training_cases": erin[k]["n_cases_training"], "head_alone_swg": r3(auc(y, P[k + "_head"])), "head_ci": ci(P[k + "_head"]),
                 "fuse2": r3(auc(y, P["fuse2"])), "fuse3": r3(auc(y, P[k + "_fuse3"])), "fuse3_ci": ci(P[k + "_fuse3"]), "gain": r3(gain), "gain_ci": ci(P[k + "_fuse3"], P["fuse2"]), "perm_p": round(float((1 + (null >= gain).sum()) / (NB + 1)), 4)})
res = {"rows": rows, "n_boot": NB, "n_perm": NB, "seed": SEED, "fusion": "fold-local z-mean of image_only OOF, cnv_only OOF and the head's imputed probability; patient = max over rows"}
os.makedirs(AGG, exist_ok=True); json.dump(res, open(AGG + "/h_nongrade_controls.json", "w"), indent=1); df_ = pd.DataFrame(rows).astype(str); open(AGG + "/tables_h.md", "w").write("| " + " | ".join(df_.columns) + " |\n|" + "---|" * len(df_.columns) + "\n" + "\n".join("| " + " | ".join(r) + " |" for r in df_.values.tolist())); print(json.dumps(res, indent=1))

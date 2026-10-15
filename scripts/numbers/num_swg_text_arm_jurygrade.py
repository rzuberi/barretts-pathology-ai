#!/usr/bin/env python3
"""Item 38 addendum: the nomic-embedding text arm was null for progression (0.43). A fairer text representation is the
jury's structured grade of the SAME matched report (the extraction that agrees 0.82 two-tier with the pathologist).
Same 428 matched samples / 65 patients as swg_text_arm; text score = jury grade ordinal (NDBE 0 .. CANCER 4) of the
matched DB report; compared with the pathologist grade-as-score, image OOF, CNV OOF and fold-local late fusions."""
import glob, json, os, re, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
F = "/mnt/scratche/slow/fmlab/zuberi01/phd/barretts_retraining/barretts_training/analysis/chapter1_lgd2_final_pre_event_20260713_final"
E = "/mnt/scratche/slow/fmlab/zuberi01/barretts_db_export"; T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"; OUT = os.environ.get("OUTDIR", "."); NB = 2000
O = {"NDBE": 0, "IND": 1, "LGD": 2, "HGD": 3, "CANCER": 4}
oof = pd.read_csv(f"{T}/feasibility/runs/swg_text_arm/output/swg_text_arm_oof.csv", dtype={"sample_id": str, "patient_id": str})
def stem(x): x = str(x).lower().strip(); x = re.sub(r"[\s_].*$", "", x); return re.sub(r"[^a-z0-9]", "", x)
coh = pd.read_csv(F + "/pre_event_cohort.csv", dtype=str).set_index("SampleID"); oof["stem"] = coh.BiopsyID_real.reindex(oof.sample_id).map(stem).values
sm = pd.read_parquet(E + "/swg_matched_reports_v2.parquet"); rows = []
for r in sm.itertuples():
    for pid in str(r.swg_path_ids).split(";"):
        if pid.strip(): rows.append({"stem": stem(pid), "rid": str(r.pathology_text_id)})
rep = pd.DataFrame(rows).drop_duplicates("stem"); oof = oof.merge(rep, on="stem", how="left")
votes = {}
for f in glob.glob(f"{T}/feasibility/runs/jury_full_*/output/llm_grades_*.csv"):
    d = pd.read_csv(f, dtype=str, on_bad_lines="skip"); d = d[d.llm_grade.isin(O)]
    for c, g in zip(d.CaseName, d.llm_grade): votes.setdefault(str(c), []).append(O[g])
jg = {c: max(set(v), key=v.count) for c, v in votes.items() if len(v) >= 4}
oof["jury_grade"] = oof.rid.map(jg); oof = oof.dropna(subset=["jury_grade"]); print("samples with a jury grade:", len(oof), "patients:", oof.patient_id.nunique(), flush=True)
folds = oof.fold.values
def zfold(v):
    o = np.zeros(len(v))
    for f in np.unique(folds): te = folds == f; o[te] = (v[te] - v[te].mean()) / (v[te].std() + 1e-9)
    return o
oof["fuse2"] = (zfold(oof.img.values) + zfold(oof.cnv.values)) / 2; oof["fuse3_jg"] = (zfold(oof.img.values) + zfold(oof.cnv.values) + zfold(oof.jury_grade.values.astype(float))) / 3
g = oof.groupby("patient_id"); y = g.y.max().values.astype(int); P = {k: g[k].max().values for k in ("jury_grade", "grade", "img", "cnv", "fuse2", "fuse3_jg")}
rng = np.random.RandomState(0); n = len(y); B = []
while len(B) < NB:
    s = rng.choice(n, n)
    if len(set(y[s])) < 2: continue
    B.append(s)
ci = lambda a, b=None: [round(float(np.percentile([roc_auc_score(y[s], a[s]) - (roc_auc_score(y[s], b[s]) if b is not None else 0) for s in B], q)), 4) for q in (2.5, 97.5)]
res = {"n_patients": int(n), "pos": int(y.sum()), "n_samples": int(len(oof)), **{k: {"auroc": round(float(roc_auc_score(y, v)), 4), "ci": ci(v)} for k, v in P.items()},
       "delta_jury_grade_minus_pathologist_grade": ci(P["jury_grade"], P["grade"]), "delta_fuse3_jg_minus_fuse2": ci(P["fuse3_jg"], P["fuse2"]), "delta_fuse3_jg_minus_img": ci(P["fuse3_jg"], P["img"]),
       "sample_level": {k: round(float(roc_auc_score(oof.y, oof[k])), 4) for k in ("jury_grade", "grade", "img", "cnv", "fuse2", "fuse3_jg")},
       "note": "jury_grade = whole-report jury majority on the matched DB report; grade = release pathologist code. Progression = release y_progressor, patient level = max over samples, matched subset only."}
os.makedirs(OUT, exist_ok=True); json.dump(res, open(os.path.join(OUT, "swg_text_arm_jurygrade.json"), "w"), indent=1); print(json.dumps(res, indent=None))

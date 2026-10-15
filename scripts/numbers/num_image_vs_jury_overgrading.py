#!/usr/bin/env python3
"""Item 35: do image models over-grade the same specimens the jury over-grades? Unit = release sample (slide) whose
accession stem matches exactly one DB specimen pair with the same pathologist grade. Image score = CONCH zero-shot
P(LGD+) from conch_swg_slides.csv (the only per-slide GRADE score on SWG; the trained image arm predicts progression,
not grade). Among pathologist-NDBE specimens: does CONCH P(LGD+) differ between jury-over-called and jury-agreed ones?"""
import glob, json, os, re, numpy as np, pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.metrics import roc_auc_score
F = "/mnt/scratche/slow/fmlab/zuberi01/phd/barretts_retraining/barretts_training/analysis/chapter1_lgd2_final_pre_event_20260713_final"
E = "/mnt/scratche/slow/fmlab/zuberi01/barretts_db_export"; T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"; OUT = os.environ.get("OUTDIR", ".")
O = {"NDBE": 0, "IND": 1, "LGD": 2, "HGD": 3, "CANCER": 4}; SWG = {"BE": 0, "ID": 1, "LGD": 2, "HGD": 3, "IMC": 4}
def stem(x): x = str(x).lower().strip(); x = re.sub(r"[\s_].*$", "", x); return re.sub(r"[^a-z0-9]", "", x)
coh = pd.read_csv(F + "/pre_event_cohort.csv", dtype=str); coh["stem"] = coh.BiopsyID_real.map(stem); coh["lab"] = pd.to_numeric(coh.Label, errors="coerce")
cz = pd.read_csv(glob.glob(T + "/feasibility/runs/conch_swg/output/conch_swg_slides.csv")[0], dtype={"sample_id": str})
cz = cz.merge(coh[["SampleID", "stem", "lab"]], left_on="sample_id", right_on="SampleID")
sp = pd.read_parquet(E + "/specimen_pairs.parquet"); sp["truth"] = sp.swg_grade_code.map(SWG); sp["stem"] = sp.path_id.map(stem)
jv = pd.read_csv(E + "/jury_specimen/jury_votes.csv", dtype=str); sp = sp.merge(jv, on="CaseName", how="left"); sp["j"] = sp.jury_label.map(O); sp = sp.dropna(subset=["j", "truth"])
# unique link: (stem, pathologist grade) -> exactly one specimen pair
key = sp.groupby(["stem", "truth"]).agg(n=("j", "size"), j=("j", "first"), jf=("jury_frac", "first")).reset_index(); key = key[key.n == 1]
m = cz.merge(key, left_on=["stem", "lab"], right_on=["stem", "truth"])
m["jury_over"] = (m.j > m.truth).astype(int); m["jury_diff"] = m.j - m.truth; m["img_resid"] = m.mean_plgd - m.groupby("truth").mean_plgd.transform("median")
res = {"linked_samples": int(len(m)), "of_release_samples": int(len(cz)), "truth_dist": m.truth.value_counts().sort_index().to_dict(), "jury_overcall_rate_linked": round(float(m.jury_over.mean()), 4)}
nd = m[m.truth == 0]
if len(nd) > 10 and nd.jury_over.nunique() > 1:
    a, b = nd[nd.jury_over == 1], nd[nd.jury_over == 0]
    res["pathologist_NDBE"] = {"n": int(len(nd)), "n_jury_overcalled": int(len(a)), "conch_mean_plgd_median_overcalled": round(float(a.mean_plgd.median()), 4), "conch_mean_plgd_median_agreed": round(float(b.mean_plgd.median()), 4),
        "mannwhitney_p": float(mannwhitneyu(a.mean_plgd, b.mean_plgd).pvalue), "auroc_conch_predicts_jury_overcall": round(float(roc_auc_score(nd.jury_over, nd.mean_plgd)), 4),
        "spearman_conch_vs_jury_grade_within_NDBE": [round(float(x), 4) for x in spearmanr(nd.mean_plgd, nd.j)]}
res["all_linked"] = {"spearman_img_resid_vs_jury_diff": [round(float(x), 4) for x in spearmanr(m.img_resid, m.jury_diff)], "note": "residuals = CONCH P(LGD+) minus the median for that pathologist grade; jury_diff = jury minus pathologist ordinal"}
os.makedirs(OUT, exist_ok=True); json.dump(res, open(os.path.join(OUT, "image_vs_jury_overgrading.json"), "w"), indent=1); print(json.dumps(res, indent=None))

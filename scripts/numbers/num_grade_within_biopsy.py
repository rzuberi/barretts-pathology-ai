#!/usr/bin/env python3
"""Item 24: re-read the ERIN grade model (case-max-trained and section-trained ABMIL, saved OOF units from
feasibility/svc_units) within biopsy specimens only, within resections only, and with specimen type as a covariate.
Truth = section (slide) label from erin_slide_labels_v2 (LGD+), as in slide_vs_casemax.json."""
import glob, json, os, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"; OUT = os.environ.get("OUTDIR", "."); NB = 2000
rep = pd.read_csv("/mnt/scratche/fast/fmlab/datasets/imaging/ERIN/data/PathologyReport_AnonIds.csv", dtype=str).fillna("")
rep["res"] = rep.SpecimenProtocol.str.upper().str.contains("RESECT|ECTOMY"); spec = dict(zip(rep.CaseName, rep.res))
m = pd.read_csv(T + "/labeller/erin_master.csv", dtype=str).dropna(subset=["h5"]).drop_duplicates("h5")
v2 = pd.read_csv(T + "/labeller/erin_slide_labels_v2.csv", dtype=str); v2 = v2.drop(columns=[c for c in ("anon_id", "CaseName") if c in v2.columns]).merge(m[["h5", "anon_id", "CaseName"]], on="h5", how="left")
v2["key"] = v2.h5.map(lambda p: os.path.basename(str(p)))
def load(arm):
    oof = {}
    for f in glob.glob(f"{T}/feasibility/svc_units/{arm}_*_*.npz"):
        z = np.load(f, allow_pickle=True); seed = int(os.path.basename(f).split("_")[1])
        for k, p in zip(z["keys"], z["preds"]): oof.setdefault(str(k), []).append(float(p))
    return {k: float(np.mean(v)) for k, v in oof.items()}
res = {"_meta": {"units_dir": T + "/feasibility/svc_units", "truth": "section label LGD+ (erin_slide_labels_v2.worst_grade)", "n_boot": NB, "seed": 0}}
for arm in ("case", "slide"):
    oof = load(arm); keys = [k for k in v2.key if k in oof] or [k for k in v2.h5 if k in oof]
    d = v2[v2.key.isin(oof) | v2.h5.isin(oof)].copy(); d["p"] = [oof.get(k, oof.get(h)) for k, h in zip(d.key, d.h5)]
    d["y"] = d.worst_grade.isin(["LGD", "HGD", "CANCER"]).astype(int); d["res"] = d.CaseName.map(spec).fillna(False).astype(int); d = d.dropna(subset=["p"])
    y, p, r, g = d.y.values, d.p.values.astype(float), d.res.values, d.anon_id.values
    up = np.unique(g); idx_of = {u: np.where(g == u)[0] for u in up}; rng = np.random.RandomState(0); B = []
    while len(B) < NB:
        s = np.concatenate([idx_of[u] for u in rng.choice(up, len(up))])
        if len(set(y[s])) < 2 or len(set(y[s][r[s] == 0])) < 2: continue
        B.append(s)
    def ci(v): return [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
    # covariate-adjusted: logistic on [logit p, is_resection] within-fold is not available; report the AUROC of specimen type alone and stratified AUROCs
    lp = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    strat = float(np.average([roc_auc_score(y[r == k], p[r == k]) for k in (0, 1) if len(set(y[r == k])) > 1], weights=[(r == k).sum() for k in (0, 1) if len(set(y[r == k])) > 1]))
    res[f"{arm}_max_trained" if arm == "case" else "section_trained"] = {
        "n_slides": int(len(d)), "n_resection_slides": int(r.sum()), "pos_rate_biopsy": round(float(y[r == 0].mean()), 4), "pos_rate_resection": round(float(y[r == 1].mean()), 4),
        "auroc_all": round(float(roc_auc_score(y, p)), 4), "auroc_all_ci": ci([roc_auc_score(y[s], p[s]) for s in B]),
        "auroc_biopsy_only": round(float(roc_auc_score(y[r == 0], p[r == 0])), 4), "auroc_biopsy_only_ci": ci([roc_auc_score(y[s][r[s] == 0], p[s][r[s] == 0]) for s in B]),
        "auroc_resection_only": round(float(roc_auc_score(y[r == 1], p[r == 1])), 4) if len(set(y[r == 1])) > 1 else None,
        "auroc_specimen_type_alone": round(float(roc_auc_score(y, r)), 4), "auroc_within_stratum_weighted": round(strat, 4),
        "delta_all_minus_biopsy_only": ci([roc_auc_score(y[s], p[s]) - roc_auc_score(y[s][r[s] == 0], p[s][r[s] == 0]) for s in B])}
    print(arm, json.dumps(res[f"{arm}_max_trained" if arm == "case" else "section_trained"]))
os.makedirs(OUT, exist_ok=True); json.dump(res, open(os.path.join(OUT, "grade_within_biopsy.json"), "w"), indent=1)

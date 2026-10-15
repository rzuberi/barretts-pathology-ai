"""Killcoyne reconciliation A0 (pre-specified in docs/paper_plan_killcoyne_reconcile.md @ a655e08): AUROC of the published LOPO
probabilities (MOESM4, Fig 2a) against sheet Status, per sample and per patient. Output: OUTDIR/kr_a0.json."""
import json, os, sys, numpy as np, pandas as pd
from scipy.stats import rankdata
T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"; S = "/mnt/scratche/fast/fmlab/datasets/imaging/SWGCohort"; K = "/mnt/scratche/slow/fmlab/zuberi01/phd/killcoyne_data_from_paper"
F = "/mnt/scratche/slow/fmlab/zuberi01/phd/barretts_retraining/barretts_training/analysis/chapter1_lgd2_final_pre_event_20260713_final"
OUT = os.environ.get("OUTDIR", T + "/results/paper_final"); os.makedirs(OUT, exist_ok=True)
def auc(y, s):
    y = np.asarray(y).astype(int); r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1; return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if 0 < n1 < len(y) else float("nan")
def summ(d, lab="y", sc="k"):   # d has columns p (patient), y, k; bootstrap + permutation over patients
    pats = np.array(sorted(d.p.unique())); g = d.groupby("p"); yp = g[lab].max().reindex(pats).values; pm = g[sc].mean().reindex(pats).values; px = g[sc].max().reindex(pats).values
    rows_by = {p: np.where(d.p.values == p)[0] for p in pats}; ys, ss = d[lab].values, d[sc].values
    stats = lambda idx: (auc(np.concatenate([ys[rows_by[pats[i]]] for i in idx]), np.concatenate([ss[rows_by[pats[i]]] for i in idx])), auc(yp[idx], pm[idx]), auc(yp[idx], px[idx]))
    pt = stats(np.arange(len(pats))); rng = np.random.RandomState(0); bs = []
    while len(bs) < 2000:
        idx = rng.choice(len(pats), len(pats))
        if 0 < yp[idx].sum() < len(idx): bs.append(stats(idx))
    bs = np.array(bs); rng = np.random.RandomState(0); perm = []
    for _ in range(2000):
        yperm = rng.permutation(yp); ymap = dict(zip(pats, yperm)); yr = d.p.map(ymap).values
        perm.append((auc(yr, ss), auc(yperm, pm), auc(yperm, px)))
    perm = np.array(perm); out = {"n_samples": int(len(d)), "n_patients": int(len(pats)), "n_P_patients": int(yp.sum())}
    for j, k in enumerate(["per_sample", "patient_mean", "patient_max"]):
        out[k] = {"auroc": round(pt[j], 3), "auroc_4dp": round(pt[j], 4), "ci95": [round(float(np.percentile(bs[:, j], 2.5)), 3), round(float(np.percentile(bs[:, j], 97.5)), 3)], "perm_p": round(float((1 + (perm[:, j] >= pt[j]).sum()) / 2001), 4)}
    return out
fd = pd.read_csv(S + "/sWGS_777_samples_cleaned_202401_Leanne_fullDetails (3) (1).csv", dtype=str); fd.columns = [c.strip().replace("\n", " ") for c in fd.columns]; fd = fd.drop_duplicates("combined_name").set_index("combined_name")
kp = pd.read_excel(K + "/41591_2020_1033_MOESM4_ESM.xlsx", sheet_name="Supporting data for Figure 2a", header=1).rename(columns={"Samplename": "cnv_id", "Probability": "k"})
kp["p"] = kp.cnv_id.map(fd.Patient); kp["status"] = kp.cnv_id.map(fd.Status); kp["path"] = kp.cnv_id.map(fd.Pathology); kp["set"] = kp.cnv_id.map(fd.Set)
kp["excluded"] = kp.cnv_id.map(fd.excluded); kp["remove"] = kp.cnv_id.map(fd.remove); unm = int(kp.p.isna().sum()); kp = kp.dropna(subset=["p", "status"]); kp["y"] = (kp.status == "P").astype(int)
res = {"_spec": "docs/paper_plan_killcoyne_reconcile.md @ a655e08 A0", "published_rows": 773, "unmatched_to_sheet": unm,
       "pathology_counts": kp.path.value_counts().to_dict(), "all_773": summ(kp), "excl_HGD_IMC": summ(kp[~kp.path.isin(["HGD", "IMC"])])}
# extra descriptive cuts using sheet flags (same scores, subsets only)
res["sheet_Set_Training"] = summ(kp[kp.set == "Training"]); res["sheet_excluded_0"] = summ(kp[kp.excluded == "0"]); res["sheet_remove_0"] = summ(kp[kp.remove == "0"])
man = pd.read_csv(F + "/training_manifest.csv", dtype=str).set_index("sample_id"); cx = pd.read_csv(F + "/feature_views/cnv/cx.csv", dtype=str).set_index("sample_id").reindex(man.index)
rel = pd.DataFrame({"cnv_id": cx.cnv_id.values, "rp": man.patient_id.values}); m = kp.merge(rel.dropna(), on="cnv_id")
res["release_rows_matched"] = summ(m); res["release_rows_matched"]["note"] = "published probability on release rows that match a published sample; patient = sheet Patient"
json.dump(res, open(OUT + "/kr_a0.json", "w"), indent=1); print(json.dumps(res, indent=1)); print("A0 DONE")

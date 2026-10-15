"""Round 3 R2 faithfulness retrain (pre-specified in docs/paper_plan_round3.md @ cf244e1): elastic-net (l1_ratio 0.9) on the
release CNV features, within the 82 Killcoyne-discovery patients, label = sheet status P/NP, leave-one-patient-out with C by
5-fold patient-grouped CV on the remaining 81. Output: results/paper_plan/round3_lopo.json."""
import json, os, sys, warnings, numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LogisticRegression; from sklearn.model_selection import GroupKFold
warnings.filterwarnings("ignore")
F = "/mnt/scratche/slow/fmlab/zuberi01/phd/barretts_retraining/barretts_training/analysis/chapter1_lgd2_final_pre_event_20260713_final"
B = "/mnt/scratche/slow/fmlab/zuberi01/phd/barretts_retraining/barretts_training/multimodal-barretts-progression"; sys.path.insert(0, B + "/src"); from barrett.training.data import load_cnv_matrix
T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"; S = "/mnt/scratche/fast/fmlab/datasets/imaging/SWGCohort"; K = "/mnt/scratche/slow/fmlab/zuberi01/phd/killcoyne_data_from_paper"; AGG = os.environ.get("OUTDIR", T + "/results/paper_plan"); os.makedirs(AGG, exist_ok=True)
def auc(y, s):
    y = np.asarray(y).astype(int); r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1; return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if 0 < n1 < len(y) else float("nan")
def r3(x): return round(float(x), 3)
man = pd.read_csv(F + "/training_manifest.csv", dtype=str).set_index("sample_id"); ids = list(man.index); pid = man.patient_id.values
cnv, feats = load_cnv_matrix(F + "/feature_views/cnv"); X = cnv.set_index("sample_id").loc[ids, feats].to_numpy(np.float64)
cx = pd.read_csv(F + "/feature_views/cnv/cx.csv", dtype=str).set_index("sample_id").reindex(ids); fd = pd.read_csv(S + "/sWGS_777_samples_cleaned_202401_Leanne_fullDetails (3) (1).csv", dtype=str); fd.columns = [c.strip().replace("\n", " ") for c in fd.columns]; fd = fd.drop_duplicates("combined_name").set_index("combined_name")
st = pd.Series(cx.cnv_id.map(fd.Status).values, index=ids); disc_rows = st.notna().values | pd.Series(pid).map(pd.Series(st.values, index=pid).groupby(level=0).apply(lambda s: s.notna().any())).values
ps = pd.Series(st.values, index=pid).dropna().groupby(level=0).agg(lambda s: s.mode().iloc[0]); dpats = sorted(ps.index); rows = np.where(pd.Series(pid).isin(dpats))[0]
yp = (ps == "P").astype(int); yr = np.array([yp[p] for p in pid[rows]]); Xd = X[rows]; pr = pid[rows]; print("discovery rows", len(rows), "patients", len(dpats), "sheet P patients", int(yp.sum()), flush=True)
CS = [0.01, 0.1, 1.0, 10.0]; oof = np.zeros(len(rows)); chosen = []
SHARD = os.environ.get("SHARD_ID"); NSH = int(os.environ.get("N_SHARDS", "1")); my = [p for i, p in enumerate(dpats) if SHARD is None or i % NSH == int(SHARD)]   # sharded LOPO: this job handles patients i with i % N_SHARDS == SHARD_ID
def prep(Xtr, Xte): med = np.nanmedian(Xtr, 0); Xtr = np.where(np.isfinite(Xtr), Xtr, med); Xte = np.where(np.isfinite(Xte), Xte, med); mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9; return (Xtr - mu) / sd, (Xte - mu) / sd
def pat_auc(p_, y_, s_): g = pd.DataFrame({"p": p_, "s": s_, "y": y_}).groupby("p"); return auc(g.y.max().values, g.s.max().values)
for i, p in enumerate(dpats):
    if p not in my: continue
    te = pr == p; tr = ~te; best, bestC = -1, 1.0; gk = GroupKFold(5)
    for C in CS:
        pv = np.zeros(tr.sum()); Xt, yt, gt = Xd[tr], yr[tr], pr[tr]
        for a, b in gk.split(Xt, yt, gt):
            Xa, Xb = prep(Xt[a], Xt[b]); pv[b] = LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=0.9, C=C, max_iter=20000, tol=1e-4).fit(Xa, yt[a]).predict_proba(Xb)[:, 1]
        a_ = pat_auc(gt, yt, pv)
        if a_ > best: best, bestC = a_, C
    Xa, Xb = prep(Xd[tr], Xd[te]); oof[te] = LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=0.9, C=bestC, max_iter=20000, tol=1e-4).fit(Xa, yr[tr]).predict_proba(Xb)[:, 1]; chosen.append(bestC)
    if (i + 1) % 10 == 0: print("lopo", i + 1, "of", len(dpats), flush=True)
if SHARD is not None:   # write this shard's rows and stop; pr3_lopo_merge.py assembles the summary
    os.makedirs(T + "/feasibility/paper_plan/lopo_shards", exist_ok=True); pd.DataFrame({"sample_id": np.array(ids)[rows][np.isin(pr, my)], "lopo": oof[np.isin(pr, my)], "C": [dict(zip(my, chosen)).get(p_) for p_ in pr[np.isin(pr, my)]]}).to_csv(f"{T}/feasibility/paper_plan/lopo_shards/shard_{SHARD}_of_{NSH}.csv", index=False); print("SHARD DONE", SHARD, flush=True); sys.exit(0)
kp = pd.read_excel(K + "/41591_2020_1033_MOESM4_ESM.xlsx", sheet_name="Supporting data for Figure 2a", header=1).rename(columns={"Samplename": "cnv_id", "Probability": "k_prob"})
f1 = pd.read_csv(T + "/feasibility/paper_plan/f1_cnv_km_oof.csv", dtype={"sample_id": str}).set_index("sample_id")
d = pd.DataFrame({"sample_id": np.array(ids)[rows], "p": pr, "y_sheet": yr, "lopo": oof, "cnv_id": cx.cnv_id.values[rows]}); d["cnv_km_f1"] = f1.cnv_km.reindex(d.sample_id).values; d["y_ours"] = man.y_progressor.astype(int).reindex(d.sample_id).values; m = d.merge(kp[["cnv_id", "k_prob"]], on="cnv_id")
g = d.groupby("p"); gm = m.groupby("p")
res = {"_spec": "docs/paper_plan_round3.md @ cf244e1 R2 faithfulness", "discovery_rows": int(len(rows)), "patients": len(dpats), "sheet_P_patients": int(yp.sum()), "C_chosen_counts": pd.Series(chosen).value_counts().to_dict(),
       "lopo_vs_sheet_status": {"row_auroc": r3(auc(yr, oof)), "patient_auroc_max": r3(auc(g.y_sheet.max().values, g.lopo.max().values))}, "matched_rows_with_published": int(len(m)), "matched_patients": int(m.p.nunique()),
       "spearman_lopo_vs_published_rows": r3(spearmanr(m.lopo, m.k_prob).correlation), "spearman_f1arm_vs_published_rows": r3(spearmanr(m.cnv_km_f1, m.k_prob).correlation), "spearman_lopo_vs_f1arm_rows": r3(spearmanr(d.lopo, d.cnv_km_f1).correlation),
       "published_vs_sheet_status": {"row_auroc": r3(auc(m.y_sheet, m.k_prob)), "patient_auroc_max": r3(auc(gm.y_sheet.max().values, gm.k_prob.max().values))}, "lopo_vs_sheet_status_matched_rows": {"row_auroc": r3(auc(m.y_sheet, m.lopo)), "patient_auroc_max": r3(auc(gm.y_sheet.max().values, gm.lopo.max().values))},
       "lopo_vs_our_endpoint": {"patient_auroc_max": r3(auc(g.y_ours.max().values, g.lopo.max().values)), "events": int(g.y_ours.max().sum())}, "published_vs_our_endpoint_matched": {"patient_auroc_max": r3(auc(gm.y_ours.max().values, gm.k_prob.max().values))}}
json.dump(res, open(AGG + "/round3_lopo.json", "w"), indent=1); print(json.dumps(res, indent=1)); print("LOPO DONE", flush=True)

"""Assembles the sharded LOPO outputs (feasibility/paper_plan/lopo_shards/shard_*_of_N.csv) into results/paper_plan/round3_lopo.json
with the same summary as the unsharded pr3_lopo.py. Exits non-zero if shards are incomplete."""
import glob, json, os, sys, numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr
F = "/mnt/scratche/slow/fmlab/zuberi01/phd/barretts_retraining/barretts_training/analysis/chapter1_lgd2_final_pre_event_20260713_final"; T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"; S = "/mnt/scratche/fast/fmlab/datasets/imaging/SWGCohort"; K = "/mnt/scratche/slow/fmlab/zuberi01/phd/killcoyne_data_from_paper"; AGG = os.environ.get("OUTDIR", T + "/results/paper_plan"); os.makedirs(AGG, exist_ok=True)
N = int(os.environ.get("N_SHARDS", "8")); files = [f"{T}/feasibility/paper_plan/lopo_shards/shard_{k}_of_{N}.csv" for k in range(N)]
if not all(os.path.exists(f) for f in files): sys.exit("shards incomplete: " + str([f for f in files if not os.path.exists(f)]))
def auc(y, s):
    y = np.asarray(y).astype(int); r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1; return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if 0 < n1 < len(y) else float("nan")
def r3(x): return round(float(x), 3)
sh = pd.concat([pd.read_csv(f, dtype={"sample_id": str}) for f in files]).set_index("sample_id")
man = pd.read_csv(F + "/training_manifest.csv", dtype=str).set_index("sample_id"); cx = pd.read_csv(F + "/feature_views/cnv/cx.csv", dtype=str).set_index("sample_id")
fd = pd.read_csv(S + "/sWGS_777_samples_cleaned_202401_Leanne_fullDetails (3) (1).csv", dtype=str); fd.columns = [c.strip().replace("\n", " ") for c in fd.columns]; fd = fd.drop_duplicates("combined_name").set_index("combined_name")
d = pd.DataFrame({"p": man.patient_id.reindex(sh.index).values, "lopo": sh.lopo.values, "cnv_id": cx.cnv_id.reindex(sh.index).values, "y_ours": man.y_progressor.astype(int).reindex(sh.index).values}, index=sh.index)
st = pd.Series(cx.cnv_id.map(fd.Status).values, index=cx.index); ps = pd.Series(st.values, index=man.patient_id.reindex(cx.index).values).dropna().groupby(level=0).agg(lambda s: s.mode().iloc[0]); d["y_sheet"] = (d.p.map(ps) == "P").astype(int).values
kp = pd.read_excel(K + "/41591_2020_1033_MOESM4_ESM.xlsx", sheet_name="Supporting data for Figure 2a", header=1).rename(columns={"Samplename": "cnv_id", "Probability": "k_prob"}); f1 = pd.read_csv(T + "/feasibility/paper_plan/f1_cnv_km_oof.csv", dtype={"sample_id": str}).set_index("sample_id"); d["cnv_km_f1"] = f1.cnv_km.reindex(d.index).values
m = d.reset_index().merge(kp[["cnv_id", "k_prob"]], on="cnv_id"); g = d.groupby("p"); gm = m.groupby("p")
res = {"_spec": "docs/paper_plan_round3.md @ cf244e1 R2 faithfulness (sharded run, merged by pr3_lopo_merge.py)", "discovery_rows": int(len(d)), "patients": int(d.p.nunique()), "sheet_P_patients": int(g.y_sheet.max().sum()), "C_chosen_counts": sh.C.value_counts().to_dict(),
       "lopo_vs_sheet_status": {"row_auroc": r3(auc(d.y_sheet, d.lopo)), "patient_auroc_max": r3(auc(g.y_sheet.max().values, g.lopo.max().values))}, "matched_rows_with_published": int(len(m)), "matched_patients": int(m.p.nunique()),
       "spearman_lopo_vs_published_rows": r3(spearmanr(m.lopo, m.k_prob).correlation), "spearman_f1arm_vs_published_rows": r3(spearmanr(m.cnv_km_f1, m.k_prob).correlation), "spearman_lopo_vs_f1arm_rows": r3(spearmanr(d.lopo, d.cnv_km_f1).correlation),
       "published_vs_sheet_status": {"row_auroc": r3(auc(m.y_sheet, m.k_prob)), "patient_auroc_max": r3(auc(gm.y_sheet.max().values, gm.k_prob.max().values))}, "lopo_vs_sheet_status_matched_rows": {"row_auroc": r3(auc(m.y_sheet, m.lopo)), "patient_auroc_max": r3(auc(gm.y_sheet.max().values, gm.lopo.max().values))},
       "lopo_vs_our_endpoint": {"patient_auroc_max": r3(auc(g.y_ours.max().values, g.lopo.max().values)), "events": int(g.y_ours.max().sum())}, "published_vs_our_endpoint_matched": {"patient_auroc_max": r3(auc(gm.y_ours.max().values, gm.k_prob.max().values))}}
json.dump(res, open(AGG + "/round3_lopo.json", "w"), indent=1); print(json.dumps(res, indent=1)); print("MERGE DONE")

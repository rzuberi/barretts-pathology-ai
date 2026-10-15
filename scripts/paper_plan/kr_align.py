"""Killcoyne reconciliation, added after the pre-specification (labelled post hoc in the report): the package ships its 773 x 634 training
matrix (be_model$fit.data) without sample names. Align its rows to our samples by row-wise Pearson correlation with our features built the
model's way (kr_assemble.R): candidate orderings from the sheet are scored first; an optimal one-to-one assignment (Hungarian) is the fallback.
Writes their matrix keyed by Sample for the LOPO grid (feature source 'their'). Row-level outputs stay on the cluster."""
import json, numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment
O = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis/feasibility/paper_plan/killcoyne"; S = "/mnt/scratche/fast/fmlab/datasets/imaging/SWGCohort"
ours = pd.read_csv(O + "/ours_modelway_cohortstd.csv").set_index("Sample"); fd = pd.read_csv(O + "/fitdata.csv"); assert list(fd.columns) == list(ours.columns)
smp = pd.read_csv(O + "/kr_samples.csv", dtype=str).set_index("Sample"); ours = ours.reindex([s for s in smp.index if s in ours.index])
def rz(M): M = M - M.mean(1, keepdims=True); return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
A = rz(ours.to_numpy(float)); Bm = rz(fd.to_numpy(float)); C = A @ Bm.T   # C[i, j] = corr(our sample i, their row j)
res = {"our_rows": int(len(A)), "their_rows": int(len(Bm))}
sheet = pd.read_csv(S + "/sWGS_777_samples_cleaned_202401_Leanne_fullDetails (3) (1).csv", dtype=str); sheet.columns = [c.strip().replace("\n", " ") for c in sheet.columns]
sheet = sheet.drop_duplicates("combined_name"); sh773 = [s for s in sheet.combined_name if s in ours.index]
pos = {s: i for i, s in enumerate(ours.index)}
def score(order):
    if len(order) != len(Bm): return None
    return round(float(np.mean([C[pos[s], j] for j, s in enumerate(order)])), 4)
kp = pd.read_excel("/mnt/scratche/slow/fmlab/zuberi01/phd/killcoyne_data_from_paper/41591_2020_1033_MOESM4_ESM.xlsx", sheet_name="Supporting data for Figure 2a", header=1)
cand = {"sheet_order": sh773, "published_table_order": [s for s in kp.Samplename if s in pos],
        "patient_numeric_then_sheet": sorted(sh773, key=lambda s: (int(smp.Patient[s]) if smp.Patient[s].isdigit() else 10**9, sh773.index(s))),
        "patient_string_then_sheet": sorted(sh773, key=lambda s: (smp.Patient[s], sh773.index(s))), "sample_name_sorted": sorted(sh773)}
res["candidate_order_mean_diag_corr"] = {k: score(v) for k, v in cand.items()}
rng = np.random.RandomState(0); res["random_order_mean_diag_corr"] = round(float(np.mean([C[rng.permutation(len(A)), np.arange(len(Bm))].mean() for _ in range(20)])), 4)
r, c = linear_sum_assignment(-C); asg = pd.Series(np.array(ours.index)[r], index=c).sort_index()   # their row j -> our sample
mc = C[r, c]; rowbest = C.argmax(0)   # for each their-row, our best sample
second = np.sort(C, axis=0)[-2, :]
res["hungarian"] = {"mean_corr": round(float(mc.mean()), 4), "median_corr": round(float(np.median(mc)), 4), "q05_corr": round(float(np.quantile(mc, 0.05)), 4),
                    "agree_with_column_argmax": int((np.array([pos[s] for s in asg.values]) == rowbest).sum()), "median_margin_best_minus_second": round(float(np.median(C.max(0) - second)), 4)}
best = max(res["candidate_order_mean_diag_corr"], key=lambda k: res["candidate_order_mean_diag_corr"][k] or -1)
cand_order = cand[best]; agree = int(sum(a == b for a, b in zip(cand_order, asg.values))) if len(cand_order) == len(asg) else None
res["best_candidate"] = best; res["best_candidate_agrees_with_hungarian_rows"] = agree
use = cand_order if (res["candidate_order_mean_diag_corr"][best] or 0) >= res["hungarian"]["mean_corr"] - 0.01 else list(asg.values)
res["alignment_used"] = best if use is cand_order else "hungarian"
# block check: consecutive their-rows belonging to the same patient under the chosen alignment
pp = [smp.Patient[s] for s in use]; res["patient_runs"] = int(1 + sum(pp[i] != pp[i - 1] for i in range(1, len(pp)))); res["patients"] = int(len(set(pp)))
out = fd.copy(); out.insert(0, "Sample", use); out.to_csv(O + "/their_features.csv", index=False)
ins = pd.read_csv(O + "/be_model_insample.csv"); pd.DataFrame({"Sample": use, "insample": ins.insample.values}).to_csv(O + "/their_insample_aligned.csv", index=False)
json.dump(res, open(O + "/kr_align.json", "w"), indent=1); print(json.dumps(res, indent=1)); print("ALIGN DONE")

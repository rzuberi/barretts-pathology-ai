"""CNV-quiet morphology (docs/paper_cnvquiet_morphology.md), step 4: group comparisons. Unit = patient (mean of the patient's eligible samples
within the group). Primary: (a) vs (c), two-sided Mann-Whitney U (scipy, exact when there are no ties), Holm over the features tested.
Effect: difference in patient-level means (a - c) with a 95% patient-bootstrap CI (2,000 draws, RandomState(0), patients resampled within group)
and AUC = U / (n_a n_c). (b): descriptive only, with (b - c) and (b - a) differences and bootstrap CIs, no p. Secondary, descriptive:
3-year-only (a) vs the (c) samples matched to them (unadjusted p); sample-level means; matched-set differences; QC composition.
Aggregates: results/paper_final/cnvquiet_morphology/stats.json; per-patient values for the figure stay on the cluster (fig_patient_values.csv)."""
import json, numpy as np, pandas as pd
from scipy.stats import mannwhitneyu
T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"; W = T + "/feasibility/paper_plan/killcoyne_mm/cnvquiet_morphology"; R = T + "/results/paper_final/cnvquiet_morphology"
FEAT = {"F1_area_mean": "epithelial nuclear area, mean (um^2)", "F2_area_sd": "epithelial nuclear area, SD (um^2)", "F3_circularity": "epithelial nuclear circularity, mean",
        "F4_basal_crowding": "basal-layer crowding (epithelial nuclei within 20 um)", "F5_pel_per100e": "peri-epithelial inflammatory nuclei per 100 epithelial nuclei"}
import os; TAG = os.environ.get("CQ_TAG", "")
F = pd.read_csv(W + f"/features_sample{TAG}.csv"); G = pd.read_csv(W + "/groups.csv"); G = G[G.group.isin(set(F.group))]; D = G.merge(F.drop(columns="group"), on="Sample"); GS = sorted(set(G.group))
assert len(D) == len(G)
def pat(df, f): return df.dropna(subset=[f]).groupby("Patient")[f].mean()
def boot_diff(x, y, n=2000, seed=0):
    rs = np.random.RandomState(seed); x, y = np.asarray(x), np.asarray(y)
    d = [rs.choice(x, len(x)).mean() - rs.choice(y, len(y)).mean() for _ in range(n)]; return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
def mwu(x, y):
    ties = len(np.unique(np.r_[x, y])) < len(x) + len(y); r = mannwhitneyu(x, y, alternative="two-sided", method="asymptotic" if ties else "exact")
    return float(r.pvalue), float(r.statistic / (len(x) * len(y))), "asymptotic" if ties else "exact"
def holm(ps):
    k = [i for i, p in enumerate(ps) if p == p]; o = sorted(k, key=lambda i: ps[i]); m = len(k); adj = [np.nan] * len(ps); run = 0
    for r, i in enumerate(o): run = max(run, min(1.0, (m - r) * ps[i])); adj[i] = run
    return adj
out = {"features": FEAT, "primary": {}, "descriptive": {}, "secondary_3y": {}, "matched_sets": {}, "qc": {}}
raw = []
for f in FEAT:
    PV = {g: pat(D[D.group == g], f) for g in GS}; pa, pc = PV["a"], PV["c"]; pb = PV.get("b")
    row = {"n_patients": {g: len(v) for g, v in PV.items()}, "n_samples": {g: int(D[(D.group == g)][f].notna().sum()) for g in GS},
           "patient_mean": {g: float(v.mean()) for g, v in PV.items()}, "patient_sd": {g: float(v.std(ddof=1)) for g, v in PV.items()},
           "patient_median": {g: float(v.median()) for g, v in PV.items()},
           "sample_mean": {g: float(D[D.group == g][f].mean()) for g in GS}, "sample_sd": {g: float(D[D.group == g][f].std(ddof=1)) for g in GS}}
    if len(pa) >= 3 and len(pc) >= 3:
        p, auc, meth = mwu(pa.values, pc.values); row.update({"diff_a_minus_c": float(pa.mean() - pc.mean()), "diff_ci95": boot_diff(pa.values, pc.values), "auc_a_gt_c": auc, "p_raw": p, "test": meth})
    else: row.update({"p_raw": np.nan, "note": "not estimable (< 3 patients with the feature in a or c)"})
    raw.append(row["p_raw"]); out["primary"][f] = row
    if pb is not None: out["descriptive"][f] = {"diff_b_minus_c": float(pb.mean() - pc.mean()), "diff_b_minus_c_ci95": boot_diff(pb.values, pc.values), "diff_b_minus_a": float(pb.mean() - pa.mean()),
                             "diff_b_minus_a_ci95": boot_diff(pb.values, pa.values), "auc_b_gt_c": mwu(pb.values, pc.values)[1]}
    a3 = D[(D.group == "a") & D.a_3y]; c3 = D[(D.group == "c") & D.matched_to.isin(a3.Sample)]; p3a, p3c = pat(a3, f), pat(c3, f)
    out["secondary_3y"][f] = {"n_patients": [len(p3a), len(p3c)], "mean_a": float(p3a.mean()), "mean_c": float(p3c.mean()), "diff": float(p3a.mean() - p3c.mean()),
                              "p_unadjusted": mwu(p3a.values, p3c.values)[0] if len(p3a) >= 3 and len(p3c) >= 3 else np.nan}
    A = D[D.group == "a"].set_index("Sample")[f]; C = D[D.group == "c"].groupby("matched_to")[f].mean(); dd = (A - C.reindex(A.index)).dropna()
    out["matched_sets"][f] = {"n_sets": int(len(dd)), "mean_diff": float(dd.mean()), "n_a_above_matches": int((dd > 0).sum())}
for f, p in zip(FEAT, holm(raw)): out["primary"][f]["p_holm"] = p
for g in GS:
    s = D[D.group == g]; ne = s.n_e.sum()
    out["qc"][g] = {"samples": int(len(s)), "patients": int(s.Patient.nunique()), "nuclei_total": int(s[["n_neo", "n_epi5", "n_inf", "n_con", "n_dead"]].sum().sum()),
                    "epithelial_nuclei": int(ne), "share_epithelial_called_neoplastic": float(s.n_neo.sum() / ne), "inflammatory_per_100_epithelial_all": float(100 * s.n_inf.sum() / ne),
                    "median_epithelial_nuclei_per_sample": float(s.n_e.median()), "share_tiles_with_epithelium": float(s.tiles_with_e.sum() / s.n_tiles.sum()),
                    "samples_ineligible_F1_F3_F5": int(s.F1_area_mean.isna().sum()), "samples_ineligible_F4": int(s.F4_basal_crowding.isna().sum())}
out["groups_present"] = GS; json.dump(out, open(R + f"/stats{TAG}.json", "w"), indent=1, default=float)
pv = pd.concat([pat(D[D.group == g], f).rename("value").reset_index(drop=True).to_frame().assign(group=g, feature=f) for f in FEAT for g in GS])
pv.to_csv(W + f"/fig_patient_values{TAG}.csv", index=False)
for f in FEAT: r = out["primary"][f]; print(f, {g: round(v, 3) for g, v in r["patient_mean"].items()}, "p", r["p_raw"], "holm", r["p_holm"])
print("CQ STATS DONE")

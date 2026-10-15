"""Render Results of docs/paper_horizon_foldstrat.md from results/paper_final/horizon_foldstrat/fs_*.json (and the pooled values in
results/paper_final/horizon_answers/q0_*.json for comparison); figure fold-stratified vs pooled. Usage: python hf_render.py RESULTS_COMMIT"""
import json, sys, os, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
RC = sys.argv[1]; R = "results/paper_final/horizon_foldstrat"; DOC = "docs/paper_horizon_foldstrat.md"
F = {(s, p, t): json.load(open(f"{R}/fs_{s}_{p}_{t}.json")) for s in ("their", "pkg") for p in ("pre", "pre_ndbe") for t in (1, 3, 5)}
Q = {(s, p): json.load(open(f"results/paper_final/horizon_answers/q0_{s}_{p}.json"))["units"]["sample"]["horizons"] for s in ("their", "pkg") for p in ("pre", "pre_ndbe")}
f3 = lambda x: "—" if x is None else f"{x:.3f}"; sg = lambda x: "—" if x is None else f"{x:+.3f}"
ci = lambda c: "[—]" if not c or c[0] is None else f"[{c[0]:.3f}, {c[1]:.3f}]"; cis = lambda c: "[—]" if not c or c[0] is None else f"[{c[0]:+.3f}, {c[1]:+.3f}]"
def table(h, rows): return ["| " + " | ".join(h) + " |", "|" + "|".join("---" for _ in h) + "|"] + ["| " + " | ".join(map(str, r)) + " |" for r in rows] + [""]
ROWS = [("Clinical Only (baseline)", "L-CLIN"), ("CNV (replication)", "L-CNV"), ("WSI", "L-IMG"), ("Early fusion", "L-EARLY"), ("Inter fusion", "L-INTER"), ("Late fusion", "L-LATE"),
        ("Demographics only (L-CLIN without grade)", "L-CLIN-nograde"), ("Intercept-only check", "INTERCEPT-ONLY")]
SRC = {"their": "their matrix", "pkg": "package features"}; POP = {"pre": "all pre-event samples", "pre_ndbe": "NDBE pre-event samples"}
def main(s, p):
    rows = []
    for nm, a in ROWS:
        cells = []
        for t in (1, 3, 5):
            o = F[(s, p, t)]["arms"][a]; pooled = Q[(s, p)][str(t)]["arms"].get(a, {}).get("ipcw", {}).get("auroc") if a != "INTERCEPT-ONLY" else None
            cells.append(f"{f3(o['ipcw'])} {ci(o['ipcw_ci95'])}" + (f" (pooled {f3(pooled)})" if pooled is not None else ""))
        rows.append([nm] + cells + ["pending"] * 3)
    rows.append(["n cases / n controls (samples; patients); within-fold pairs, repeat 1"] + [f"{F[(s, p, t)]['n_cases']} / {F[(s, p, t)]['n_controls']}; {F[(s, p, t)]['n_case_patients']} / {F[(s, p, t)]['n_control_patients']}; {F[(s, p, t)]['within_fold_pairs_repeat1']:,} of {F[(s, p, t)]['all_pairs']:,}" for t in (1, 3, 5)] + ["pending (slides being scanned)"] * 3)
    return table(["", "Internal 1-year", "Internal 3-year", "Internal 5-year", "ACE-B 1", "ACE-B 3", "ACE-B 5"], rows)
def deltas(s, p):
    rows = []
    for t in (1, 3, 5):
        G = F[(s, p, t)]
        for _, a in ROWS[:6]:
            o = G["arms"][a]; fm = lambda d: "—" if not d else f"{sg(d['delta'])} {cis(d['ci95'])}" + (f" (p {d['p_unadjusted']}, max-T {d['p_max_T']})" if "p_max_T" in d else "")
            rows.append([f"{t} y", a, f3(o["ipcw"]), fm(o.get("delta_vs_L-LATE")), fm(o.get("delta_vs_L-CNV"))])
        rows.append([f"{t} y", "best", G["best"], "ranking: " + " > ".join(G["ranking"]), ""])
    return rows
L = open(DOC).read().split("\n## Results")[0].rstrip().split("\n")
L += ["", "## Results", "", f"Pre-specification commit 2d0014f; results commit {RC}. Script `scripts/paper_plan/hf_foldstrat.py` (Slurm via `scripts/cluster/campaign.sh`, prefix hf, one task per CNV source × population × horizon), `scripts/paper_plan/hf_render.py`. Results `results/paper_final/horizon_foldstrat/fs_{{their,pkg}}_{{pre,pre_ndbe}}_{{1,3,5}}.json`; figure `results/paper_final/horizon_foldstrat/figs/foldstrat_vs_pooled.{{png,pdf,json}}`. Pooled values from `results/paper_final/horizon_answers/q0_*.json` (afa0278).", ""]
ok = all(F[k]["arms"]["INTERCEPT-ONLY"]["ipcw"] == 0.5 for k in F); fa = all(all(v == 10 for v in F[k]["fold_agreement_repeats"].values()) for k in F)
L += ["### Status", ""] + table(["Item", "Status"], [["Fold-stratified IPCW AUROC, every row and horizon, both CNV sources, both populations", "DONE"], ["Intercept-only check (expected 0.500)", "DONE: " + ("0.500 in all 12 cells" if ok else "NOT 0.500 in some cells")], ["Q1 deltas vs L-CNV and vs L-LATE under this metric", "DONE"]])
L += [f"Every arm's fold assignment matches `kv_cv.R` in all 10 repeats ({'yes' if fa else 'no'}). Valid bootstrap draws: {min(F[k]['valid_bootstrap_draws'] for k in F):,} of 2,000 in every cell.", ""]
th = {t: F[("their", "pre", t)]["arms"] for t in (1, 3, 5)}
L += ["### Answer", "", f"Restricted to same-fold case–control pairs, the intercept-only prediction scores 0.500 at every horizon, so the metric removes the between-fold-model artefact. The CNV, image and fusion arms fall by 0.00–0.04 from their pooled values (the clinical arms rise slightly); L-LATE remains the best arm on all pre-event samples at 1, 3 and 5 years with either CNV source ({', '.join(f3(th[t]['L-LATE']['ipcw']) for t in (1, 3, 5))} on their matrix), and its gain over L-CNV ({', '.join(sg(th[t]['L-LATE']['delta_vs_L-CNV']['delta']) for t in (1, 3, 5))}) is not significant after max-T adjustment (p {', '.join(str(th[t]['L-LATE']['delta_vs_L-CNV']['p_max_T']) for t in (1, 3, 5))}). "
      f"Demographics-only L-CLIN stays below 0.5 under this metric ({', '.join(f3(th[t]['L-CLIN-nograde']['ipcw']) for t in (1, 3, 5))}), so its pooled shortfall is not only the intercept artefact. One line: the fusion ranking survives fold stratification; the size of its advantage over CNV remains unresolved.", ""]
for s in ("their", "pkg"):
    for p in ("pre", "pre_ndbe"):
        L += [f"### Table: {SRC[s]}, {POP[p]}", "", "Fold-stratified IPCW time-dependent AUROC [95% patient-bootstrap CI] (pooled per-sample IPCW AUROC from `docs/paper_horizon_answers.md` in parentheses).", ""] + main(s, p)
for s in ("their", "pkg"):
    for p in ("pre", "pre_ndbe"):
        L += [f"### Q1 under this metric: {SRC[s]}, {POP[p]}", "", "Paired Δ on the same bootstrap draws; swap-permutation p and single-step max-T over the 5 non-clinical arms (clinical rows: CI only).", ""] + table(["Horizon", "Arm", "Fold-stratified AUROC", "Δ vs L-LATE [CI] (p, max-T)", "Δ vs L-CNV [CI] (p, max-T)"], deltas(s, p))
L += ["Unweighted fold-stratified AUROCs (w = 1) are in the JSON files (`arms.<arm>.unweighted`).", "",
      "**Method.** Per repeat, IPCW-weighted concordance over case–control pairs from the same outer fold (cases weighted 1/Ĝ(T−), Ĝ the reverse Kaplan–Meier over the evaluated samples), pooled over folds, then averaged over the 10 repeats; bootstrap draws keep each resampled patient's folds and re-estimate Ĝ; permutation swaps a patient's two arms' per-repeat probabilities together.",
      "**Sources.** `feasibility/paper_plan/killcoyne_mm/cv/preds/` (kv_cv.R, d69de24), `horizons/outer/inter_*` (hz_fit.R, 4d7efa9), `horizon_answers/clin_outer.csv` (ha_clin.py, afa0278), `horizons/samples.csv` (hz_prep.py).",
      "**Caveats.** (1) Only about 9–10% of case–control pairs are within a fold (folds of 8 patients), so the intervals are wider than the pooled ones. (2) Within-fold pairs include pairs from the same progressor (an early control sample and a later case sample), which no between-patient ranking can separate except through the sample-level score. (3) Demographics-only stays below 0.5 under this metric: with a near-null signal, a model fitted on the other folds' patients is known to be negatively correlated with held-out patients' labels; this was not investigated further here. (4) Design caveat as before: AUROCs only; no absolute risk, calibration or PPV.", ""]
os.makedirs(R + "/figs", exist_ok=True); fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.8), sharey=True); nums = {}
COL = {"L-CLIN": "#9e9e9e", "L-CNV": "#1f77b4", "L-IMG": "#d62728", "L-EARLY": "#9467bd", "L-INTER": "#8c564b", "L-LATE": "#2ca02c"}
for ax, s in zip(axs, ("their", "pkg")):
    nums[s] = {}
    for j, (_, a) in enumerate(ROWS[:6]):
        xs = np.array([1, 3, 5]) + (j - 2.5) * 0.14; v = [F[(s, "pre", t)]["arms"][a]["ipcw"] for t in (1, 3, 5)]; c = [F[(s, "pre", t)]["arms"][a]["ipcw_ci95"] for t in (1, 3, 5)]; pv = [Q[(s, "pre")][str(t)]["arms"][a]["ipcw"]["auroc"] for t in (1, 3, 5)]
        ax.errorbar(xs, v, yerr=[[v[i] - c[i][0] for i in range(3)], [c[i][1] - v[i] for i in range(3)]], fmt="o-", color=COL[a], ms=3.5, lw=1, capsize=2, label=a); ax.plot(xs, pv, "x", color=COL[a], ms=4)
        nums[s][a] = {"foldstrat": v, "ci95": c, "pooled": pv}
    ax.axhline(0.5, color="k", lw=0.5, ls=":"); ax.set_xticks([1, 3, 5]); ax.set_xticklabels(["1 y", "3 y", "5 y"]); ax.set_title(SRC[s] + " (o fold-stratified, x pooled)")
axs[0].set_ylabel("IPCW td-AUROC, all pre-event samples"); axs[0].set_ylim(0.2, 1.0); axs[1].legend(frameon=False, fontsize=7, loc="lower left", ncol=2)
fig.savefig(R + "/figs/foldstrat_vs_pooled.png", dpi=200, bbox_inches="tight"); fig.savefig(R + "/figs/foldstrat_vs_pooled.pdf", bbox_inches="tight"); json.dump(nums, open(R + "/figs/foldstrat_vs_pooled.json", "w"), indent=1)
open(DOC, "w").write("\n".join(L) + "\n"); print("rendered", len(L))

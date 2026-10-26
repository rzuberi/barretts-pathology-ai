"""Figure for docs/two_papers_P1B.md: robustness forest (AUROC per model and paired differences across confound and sensitivity rows).
Reads results/paper_final/two_papers_P1B/*.json (aggregates only); writes paper/two_papers/figs_P1B/p1b_robustness_forest.png (local, never committed)."""
import json, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
H = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(H, "../../results/paper_final/two_papers_P1B"); OUT = os.path.join(H, "../../paper/two_papers/figs_P1B")
J = lambda f: json.load(open(f"{R}/{f}.json"))
B = J("batch")["batch"]; D = J("depth")["depth"]; G = J("grade")["grade"]; S = J("sens")["sens"]; X = J("select_extractors_scales_poolings_ensembles")["select"]
cut = D["depth_tertiles"]["cuts_x"]
ROWS = [("All pairs (primary)", B["overall"]["sample"]), ("All pairs, patient-weighted", B["overall"]["patient_weighted"]),
        ("Within sequencing batch", B["within_batch"]["sample"]), ("Within scanner", B["within_scanner"]["sample"]), ("Within batch and scanner", B["within_batch_and_scanner"]["sample"]),
        (f"Depth < {cut[0]}×", D["depth_tertiles"]["0"]), (f"Depth {cut[0]}–{cut[1]}×", D["depth_tertiles"]["1"]), (f"Depth > {cut[1]}×", D["depth_tertiles"]["2"]),
        ("Small biopsy (< 256 tiles)", D["biopsy_size"]["1"]), ("Large biopsy (≥ 256 tiles)", D["biopsy_size"]["0"]),
        ("NDBE samples only", G["within_grade"]["ndbe_only"]), ("Pairs sharing grade", G["within_grade"]["pairs_sharing_grade"]),
        ("Same block only", S["same_block_only"]), ("Database event dates", S["db_event_dates_eval_only"]), ("Database-updated labels (refit)", S["refits"]["r54"]),
        ("Linkable samples added (refit)", S["refits"]["r52a"])]
NEST = {"L-IMG": X["auroc"]["IMG-nested"], "L-LATE": X["auroc"]["LATE-nested"], "IMG_minus_CNV": X["nested_IMG_minus_L_CNV"], "LATE_minus_IMG": X["nested_LATE_minus_nested_IMG"]}
COL = {"L-CNV": "#2a78d6", "L-IMG": "#eb6834", "L-LATE": "#1baf7a"}; MK = {"L-CNV": "o", "L-IMG": "s", "L-LATE": "D"}
plt.rcParams.update({"font.size": 9})
fig, ax = plt.subplots(1, 3, figsize=(11, 6.6), dpi=200, gridspec_kw={"width_ratios": [1.5, 1, 1]}, sharey=True)
n = len(ROWS) + 1; ys = list(range(n))[::-1]
for i, (lab, h) in enumerate(ROWS):
    yv = ys[i]
    for k, m in enumerate(("L-CNV", "L-IMG", "L-LATE")):
        a = h["auroc"][m]; o = (k - 1) * 0.22; ax[0].errorbar(a["value"], yv + o, xerr=[[a["value"] - a["ci95"][0]], [a["ci95"][1] - a["value"]]], fmt=MK[m], ms=4, color=COL[m], lw=1, label=m if i == 0 else None)
    for j, key in enumerate(("IMG_minus_CNV", "LATE_minus_IMG")):
        dl = h[key]; ax[j + 1].errorbar(dl["delta"], yv, xerr=[[dl["delta"] - dl["ci95"][0]], [dl["ci95"][1] - dl["delta"]]], fmt="o", ms=4, color="#3d3d3a", lw=1)
yv = ys[-1]
for k, m in enumerate(("L-IMG", "L-LATE")):
    a = NEST[m]; o = (k) * 0.22; ax[0].errorbar(a["value"], yv + o, xerr=[[a["value"] - a["ci95"][0]], [a["ci95"][1] - a["value"]]], fmt=MK[m], ms=4, color=COL[m], lw=1, mfc="white")
for j, key in enumerate(("IMG_minus_CNV", "LATE_minus_IMG")):
    dl = NEST[key]; ax[j + 1].errorbar(dl["delta"], yv, xerr=[[dl["delta"] - dl["ci95"][0]], [dl["ci95"][1] - dl["delta"]]], fmt="o", ms=4, color="#3d3d3a", lw=1, mfc="white")
ax[0].set_yticks(ys); ax[0].set_yticklabels([r[0] for r in ROWS] + ["Image model chosen in inner CV (30 variants)"])
ax[0].set_xlabel("Fold-stratified AUROC (95% CI)"); ax[0].legend(loc="upper left", frameon=False, fontsize=9)
ax[1].set_xlabel("L-IMG − L-CNV"); ax[2].set_xlabel("L-LATE − L-IMG")
for a_ in ax[1:]: a_.axvline(0, color="#8c8c88", lw=0.8, ls=":")
for a_ in ax:
    a_.grid(axis="x", color="#e6e6e3", lw=0.6); a_.set_axisbelow(True)
    for s in ("top", "right"): a_.spines[s].set_visible(False)
    for gi in (1, 4, 7, 9, 11, 15): a_.axhline(ys[gi] - 0.5, color="#c3c2b7", lw=0.6)   # group separators below rows 1, 4, 7, 9, 11, 15
fig.suptitle("Paper 1 confound and robustness checks: 571 pre-event samples, 75 patients (32 progressors)", fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.97)); os.makedirs(OUT, exist_ok=True); fig.savefig(f"{OUT}/p1b_robustness_forest.png", dpi=200); print("saved")

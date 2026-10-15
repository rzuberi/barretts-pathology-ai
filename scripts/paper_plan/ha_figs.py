"""Horizon answers figures (docs/paper_horizon_answers.md @ 81473df) from aggregate JSON only. PNG + PDF + plotted numbers as JSON in
results/paper_final/horizon_answers/figs/."""
import json, os, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = "results/paper_final/horizon_answers"; O = R + "/figs"; os.makedirs(O, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
def save(fig, name, nums):
    fig.savefig(f"{O}/{name}.png", dpi=200, bbox_inches="tight"); fig.savefig(f"{O}/{name}.pdf", bbox_inches="tight"); plt.close(fig); json.dump(nums, open(f"{O}/{name}.json", "w"), indent=1)
ARMS = ["L-CLIN", "L-CLIN-nograde", "L-CNV", "L-IMG", "L-EARLY", "L-INTER", "L-LATE"]
COL = dict(zip(ARMS, ["#9e9e9e", "#cfcfcf", "#1f77b4", "#d62728", "#9467bd", "#8c564b", "#2ca02c"]))
fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.8), sharey=True); nums = {}
for ax, src in zip(axs, ("their", "pkg")):
    d = json.load(open(f"{R}/q0_{src}_pre.json"))["units"]["sample"]["horizons"]; nums[src] = {}
    for j, a in enumerate(ARMS):
        xs = np.array([1, 3, 5]) + (j - 3) * 0.12; v = [d[str(t)]["arms"][a]["ipcw"]["auroc"] for t in (1, 3, 5)]; ci = [d[str(t)]["arms"][a]["ipcw"]["ci95"] for t in (1, 3, 5)]
        ax.errorbar(xs, v, yerr=[[v[i] - ci[i][0] for i in range(3)], [ci[i][1] - v[i] for i in range(3)]], fmt="o-", color=COL[a], ms=3.5, lw=1, capsize=2, label=a); nums[src][a] = {"auroc": v, "ci95": ci}
    nums[src]["n"] = {t: [d[str(t)]["n_cases"], d[str(t)]["n_controls"], d[str(t)]["n_case_patients"], d[str(t)]["n_control_patients"]] for t in (1, 3, 5)}
    ax.axhline(0.5, color="k", lw=0.5, ls=":"); ax.set_xticks([1, 3, 5]); ax.set_xticklabels([f"{t} y\n{d[str(t)]['n_cases']}/{d[str(t)]['n_controls']}" for t in (1, 3, 5)])
    ax.set_title("their CNV matrix" if src == "their" else "package features"); ax.set_xlabel("horizon (cases / controls, samples)")
axs[0].set_ylabel("IPCW time-dependent AUROC (95% CI)"); axs[0].set_ylim(0.1, 1.0); axs[1].legend(frameon=False, fontsize=7, loc="lower left", ncol=2)
save(fig, "q0_auroc_by_horizon", nums)
q2 = json.load(open(f"{R}/q2.json")); fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.4), sharey=True); nums = {"best": q2["best_by_horizon_their_pre"]}
for ax, src in zip(axs, ("their", "pkg")):
    nums[src] = {}; xs = np.arange(3); w = 0.2
    for i, (g, k, col, lab) in enumerate((("cases", "CNVneg_bestpos", "#2ca02c", "cases newly captured"), ("cases", "CNVpos_bestneg", "#a6dba0", "cases lost"), ("controls", "CNVpos_bestneg", "#1f77b4", "controls newly cleared"), ("controls", "CNVneg_bestpos", "#aec7e8", "controls newly flagged"))):
        v = []
        for t in (1, 3, 5):
            b = q2["best_by_horizon_their_pre"][str(t)]; f_ = b if b != "L-CNV" else "L-LATE"; v.append(q2["results"][src][f"{t}|sample"][f"{f_}_vs_L-CNV"]["reclassification"][g][k])
        ax.bar(xs + (i - 1.5) * w, v, w, color=col, label=lab); nums[src][lab] = v
    ax.set_xticks(xs); ax.set_xticklabels(["1 y", "3 y", "5 y"]); ax.set_title(("their matrix" if src == "their" else "package features") + ": L-CNV → best arm, 80% sens.")
axs[0].set_ylabel("samples"); axs[1].legend(frameon=False, fontsize=7)
save(fig, "q2_reclassification", nums)
fig, ax = plt.subplots(figsize=(4.8, 3.4)); nums = {}
for j, src in enumerate(("their", "pkg")):
    d = json.load(open(f"results/paper_final/horizons/h3_{src}.json"))["results"]; v = [d[str(t)]["ratio_mean"] for t in (1, 3, 5)]; ci = [d[str(t)]["ratio_ci95"] for t in (1, 3, 5)]
    xs = np.array([1, 3, 5]) + (j - 0.5) * 0.2; ax.errorbar(xs, v, yerr=[[v[i] - ci[i][0] for i in range(3)], [ci[i][1] - v[i] for i in range(3)]], fmt="o-", capsize=3, label="their matrix" if src == "their" else "package features")
    nums[src] = {"wsi_share_mean_over_fits": v, "ci95": ci, "source": f"results/paper_final/horizons/h3_{src}.json"}
ax.axhline(0.5, color="k", lw=0.5, ls=":"); ax.set_xticks([1, 3, 5]); ax.set_xticklabels(["1 y", "3 y", "5 y"]); ax.set_ylabel("WSI share c / (b + c)"); ax.legend(frameon=False, fontsize=8)
save(fig, "q3_wsi_share", nums); print("figs done")

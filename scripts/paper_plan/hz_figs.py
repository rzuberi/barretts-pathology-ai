"""Horizons figures (docs/paper_survival_horizons.md @ ee51db8) from the aggregate JSON in results/paper_final/horizons/ (no row-level data).
Writes PNG + PDF and the plotted numbers as JSON to results/paper_final/horizons/figs/."""
import json, os, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = "results/paper_final/horizons"; O = R + "/figs"; os.makedirs(O, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
def save(fig, name, nums):
    fig.savefig(f"{O}/{name}.png", dpi=200, bbox_inches="tight"); fig.savefig(f"{O}/{name}.pdf", bbox_inches="tight"); plt.close(fig)
    json.dump(nums, open(f"{O}/{name}.json", "w"), indent=1)
ARMS = ["L-GRADE", "L-GRADE+age+sex", "L-CNV", "L-IMG", "L-EARLY", "L-INTER", "L-LATE"]; COL = dict(zip(ARMS, ["#9e9e9e", "#616161", "#1f77b4", "#d62728", "#9467bd", "#8c564b", "#2ca02c"]))
# 1. td-AUROC per arm at 1/3/5 y
fig, axs = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True); nums = {}
for ax, src in zip(axs, ("their", "pkg")):
    d = json.load(open(f"{R}/h1_{src}_pre.json"))["units"]["sample"]["horizons"]; nums[src] = {}
    for j, a in enumerate(ARMS):
        xs = np.array([1, 3, 5]) + (j - 3) * 0.12; v = [d[str(t)]["arms"][a]["ipcw"]["auroc"] for t in (1, 3, 5)]; ci = [d[str(t)]["arms"][a]["ipcw"]["ci95"] for t in (1, 3, 5)]
        ax.errorbar(xs, v, yerr=[[v[i] - ci[i][0] for i in range(3)], [ci[i][1] - v[i] for i in range(3)]], fmt="o-", color=COL[a], ms=3.5, lw=1, capsize=2, label=a)
        nums[src][a] = {"auroc": v, "ci95": ci}
    nums[src]["n_cases_controls"] = {t: [d[str(t)]["n_cases"], d[str(t)]["n_controls"]] for t in (1, 3, 5)}
    ax.axhline(0.5, color="k", lw=0.5, ls=":"); ax.set_xticks([1, 3, 5]); ax.set_xticklabels([f"{t} y\n{d[str(t)]['n_cases']} / {d[str(t)]['n_controls']}" for t in (1, 3, 5)])
    ax.set_title(f"CNV source: {'their matrix' if src == 'their' else 'package features'}"); ax.set_xlabel("horizon (cases / controls, samples)")
axs[0].set_ylabel("IPCW time-dependent AUROC (95% CI)"); axs[1].legend(frameon=False, fontsize=7, loc="lower left", ncol=2); axs[0].set_ylim(0.1, 1.0)
save(fig, "h1_td_auroc", nums)
# 2. H2 reclassification (operating point a, all pre-event samples)
h2 = json.load(open(f"{R}/h2.json")); best = h2["best_model"]; fig, axs = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True); nums = {"best_model": best}
for ax, src in zip(axs, ("their", "pkg")):
    nums[src] = {}; xs = np.arange(3); w = 0.2
    for i, (g, k, col) in enumerate((("cases", "CNVneg_fusionpos", "#2ca02c"), ("cases", "CNVpos_fusionneg", "#a6dba0"), ("controls", "CNVpos_fusionneg", "#1f77b4"), ("controls", "CNVneg_fusionpos", "#aec7e8"))):
        v = [h2["results"][src][f"a_sens80|pre|sample|{t}"]["L-LATE_vs_L-CNV"]["reclassification"][g][k] for t in (1, 3, 5)]
        ax.bar(xs + (i - 1.5) * w, v, w, color=col, label=f"{g}: {k.replace('_', ' → ').replace('CNVneg', 'CNV−').replace('CNVpos', 'CNV+').replace('fusionpos', 'L-LATE+').replace('fusionneg', 'L-LATE−')}"); nums[src][f"{g}:{k}"] = v
    ax.set_xticks(xs); ax.set_xticklabels(["1 y", "3 y", "5 y"]); ax.set_title(f"{'their matrix' if src == 'their' else 'package features'}: L-CNV → L-LATE, 80% sens.")
axs[0].set_ylabel("samples reclassified"); axs[1].legend(frameon=False, fontsize=7)
save(fig, "h2_reclassification", nums)
# 3. H3 weights by horizon
fig, ax = plt.subplots(figsize=(4.8, 3.4)); nums = {}
for j, src in enumerate(("their", "pkg")):
    d = json.load(open(f"{R}/h3_{src}.json"))["results"]; v = [d[str(t)]["ratio_of_means"] for t in (1, 3, 5)]; ci = [d[str(t)]["ratio_of_means_ci95"] for t in (1, 3, 5)]
    xs = np.array([1, 3, 5]) + (j - 0.5) * 0.2; ax.errorbar(xs, v, yerr=[[v[i] - ci[i][0] for i in range(3)], [ci[i][1] - v[i] for i in range(3)]], fmt="o-", capsize=3, label="their matrix" if src == "their" else "package features")
    nums[src] = {"ratio_of_means": v, "ci95": ci, "ratio_mean_over_fits": [d[str(t)]["ratio_mean"] for t in (1, 3, 5)]}
ax.axhline(0.5, color="k", lw=0.5, ls=":"); ax.set_xticks([1, 3, 5]); ax.set_xticklabels(["1 y", "3 y", "5 y"]); ax.set_ylabel("WSI weight c / (b + c)"); ax.legend(frameon=False, fontsize=8)
save(fig, "h3_weights", nums)
# 4. H4 stability by depth (only if run)
if os.path.exists(f"{R}/h4_depth.json"):
    d = json.load(open(f"{R}/h4_depth.json")); dep = [k for k in d["by_depth"]]
    fig, ax = plt.subplots(figsize=(4.8, 3.4)); v = [d["by_depth"][k]["icc_vs_native"] for k in dep]; ax.plot(dep, v, "o-"); ax.set_ylabel("ICC(A,1) vs native"); ax.set_xlabel("target depth")
    save(fig, "h4_stability", {"depths": dep, "icc_vs_native": v})
print("figs done")

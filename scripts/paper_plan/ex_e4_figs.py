"""Exploration E4 figures (local, not committed) from results/paper_final/explore_E4/*.json -> paper/figures_explore_E4/ (PNG, 200 dpi, fonts >= 9 pt)."""
import json, glob, os, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); JD = ROOT + "/results/paper_final/explore_E4"; FD = ROOT + "/paper/figures_explore_E4"; os.makedirs(FD, exist_ok=True)
J = {os.path.basename(f)[5:-5]: json.load(open(f)) for f in sorted(glob.glob(JD + "/eval_*.json"))}
ORDER = [s for s in ["coxx", "coxs", "coxp", "h3", "h5", "wt", "wt2", "coxx_img+stored_cnv", "stored_img+coxx_cnv", "wt_img+stored_cnv", "h5_img+stored_cnv", "ens_coxx", "ens_h3", "ens_h5", "ens_wt2", "stored+ensh3late"] if s in J]
NAME = {"coxx": "Cox, extended FU", "coxs": "Cox, study FU", "h3": "3-y horizon", "h5": "5-y horizon", "wt": "Certainty-weighted", "coxx_img+stored_cnv": "Cox H&E + stored CN",
        "stored_img+coxx_cnv": "Stored H&E + Cox CN", "wt_img+stored_cnv": "Weighted H&E + stored CN", "h5_img+stored_cnv": "5-y H&E + stored CN",
        "coxp": "Cox, pre-event only", "wt2": "Strongly weighted", "ens_coxx": "Ensemble stored + Cox", "ens_h3": "Ensemble stored + 3-y", "ens_h5": "Ensemble stored + 5-y", "ens_wt2": "Ensemble stored + strong wt", "stored+ensh3late": "Stored bands, ens. 3-y fusion"}
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"; INK = "#0b0b0b"; INK2 = "#52514e"; GRID = "#e4e3df"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False})
def forest(ax, items, title, xlabel):
    """items: list of (label, [(series_name, colour, delta, lo, hi)])"""
    yy = np.arange(len(items))[::-1]; k = max(len(s) for _, s in items); off = np.linspace(-0.22, 0.22, k) if k > 1 else [0]
    seen = set()
    for y0, (lab, ser) in zip(yy, items):
        for o, (nm, col, d, lo, hi) in zip(off, ser):
            ax.plot([lo, hi], [y0 + o] * 2, color=col, lw=2, solid_capstyle="round"); ax.plot(d, y0 + o, "o", ms=7, color=col, mec="white", mew=1.5, label=None if nm in seen else nm); seen.add(nm)
    ax.axvline(0, color=INK2, lw=1); ax.set_yticks(yy); ax.set_yticklabels([l for l, _ in items]); ax.set_xlabel(xlabel); ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
    pass
# Figure 1: Δ non-progressor patients flagged and Δ progressor patients caught vs R49 for each scheme's rebuilt rules
RULES = [("R49-new t_low 95/t_high 90", "Gate (R49 cut-offs)", C1), ("P1-new 90%", "H&E alone 90%", C2), ("P3-new 90%", "Sequence all, fusion 90%", C3)]
fig, axs = plt.subplots(1, 2, figsize=(10, 0.45 * len(ORDER) + 2.2), sharey=True)
for ax, met, xl in ((axs[0], "non_progressor_patients_flagged", "Δ non-progressor patients flagged vs R49 (of 43)"), (axs[1], "progressor_patients_caught", "Δ progressor patients caught vs R49 (of 32)")):
    items = [(NAME[s], [(lab, col, J[s]["rules"][r]["delta_vs_R49"][met]["delta"], *J[s]["rules"][r]["delta_vs_R49"][met]["ci95"]) for r, lab, col in RULES]) for s in ORDER]
    forest(ax, items, "Fewer false flags ←" if met.startswith("non") else "More caught →", xl)
h, l = axs[0].get_legend_handles_labels(); fig.legend(h[::-1], l[::-1], loc="upper center", ncol=3, frameon=False, fontsize=10); fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(FD + "/e4_rules_vs_r49.png", dpi=200); plt.close(fig)
# Figure 2: specificity gain from time-aware training at R49's endoscopy sensitivity (vs the stored score, same matching)
fig, ax = plt.subplots(figsize=(7.5, 0.45 * len(ORDER) + 2.0))
items = []
for s in ORDER:
    R = J[s]["matched_sensitivity"]["R49"]; ser = []
    for nm, lab, col in (("new H&E max (P1-new score)", "H&E score", C1), ("new fusion max (P3-new score, sequence all)", "Fusion score", C2)):
        d = R[nm]["delta_vs_stored_score_at_same_sensitivity"]["endoscopy_specificity"]; ser.append((lab, col, d["delta"], *d["ci95"]))
    items.append((NAME[s], ser))
forest(ax, items, "Specificity at R49's sensitivity: new − stored score", "Δ endoscopy specificity"); h, l = ax.get_legend_handles_labels(); fig.legend(h[::-1], l[::-1], loc="upper center", ncol=2, frameon=False, fontsize=10); fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(FD + "/e4_spec_matched.png", dpi=200); plt.close(fig)
# Figure 3: discrimination of the fused score (Δ vs stored L-LATE): AUROC, td-AUROC 1/3/5, C-index (extended FU)
MS = [("auroc_sample", "AUROC"), ("td_auroc_1y", "td-AUROC 1 y"), ("td_auroc_3y", "td-AUROC 3 y"), ("td_auroc_5y", "td-AUROC 5 y"), ("cindex_ext", "C-index")]
fig, axs = plt.subplots(1, 2, figsize=(11, 0.45 * len(ORDER) + 2.2), sharey=True)
for ax, m, ttl in ((axs[0], "L-IMG", "H&E score"), (axs[1], "L-LATE", "Fusion score")):
    items = [(NAME[s], [(lab, col, *(lambda d: (d["delta"]["delta"], *d["delta"]["ci95"]))(J[s]["discrimination"][f"{k}_{m}"])) for (k, lab), col in zip([MS[0], MS[2], MS[4]], (C1, C2, C3))]) for s in ORDER]
    forest(ax, items, f"{ttl}: new − stored", "Δ (fold-stratified)")
h, l = axs[0].get_legend_handles_labels(); fig.legend(h[::-1], l[::-1], loc="upper center", ncol=3, frameon=False, fontsize=10); fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(FD + "/e4_discrimination.png", dpi=200); plt.close(fig)
print("figs ->", FD)

"""Two papers, agent P2N: figures of the selection-aware null from the aggregate results/paper_final/two_papers_P2N/merge.json (local; PNG 200 dpi, not committed).
fig_p2n_null.png: null distribution of the best-of-search gain (NP flagged vs R49, tolerance 1 progressor) for the three nulls, real value marked.
fig_p2n_null_scatter.png: the selected cell of every null label set (Δ caught, Δ NP flagged) against the real selection."""
import json, os, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
H = os.path.dirname(os.path.abspath(__file__)); LIMG = os.environ.get("LIMG", "stored"); SFX = "" if LIMG == "stored" else "_clean"   # clean: ART's clean-tile L-IMG
R = json.load(open(H + f"/../../results/paper_final/two_papers_P2N/merge{SFX}.json")); OD = H + "/../../paper/two_papers/figs_P2N"; os.makedirs(OD, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e", "ytick.color": "#52514e"})
BLUE, INK, MUTED = "#2a78d6", "#0b0b0b", "#8a8984"
PAN = [("a_free", "(a) Full refit of L-IMG and L-CNV\n+ whole rule search"), ("b_free", "(b) Stored predictions\n+ whole rule search"), ("b_strat", "(b') As (b), labels permuted within\nendoscopy-count strata")]
PAN = [(k, t) for k, t in PAN if k in R]
def hist_fig(key, fname, xlab):
    fig, axes = plt.subplots(1, len(PAN), figsize=(3.6 * len(PAN), 3.4)); axes = np.atleast_1d(axes)
    lo = min(min(min(R[k][key]["null_dNP_values"]) for k, _ in PAN), min(R[k][key]["dNP"]["real"] for k, _ in PAN)) - 1.0
    hi = max(max(R[k][key]["null_dNP_values"]) for k, _ in PAN) + 1.0
    for ax, (k, t) in zip(axes, PAN):
        o = R[k][key]; v = np.array(o["null_dNP_values"], float); bins = np.arange(np.floor(lo), np.ceil(hi) + 0.5, 0.5); rv = o["dNP"]["real"]
        ax.hist(v, bins=bins, color=BLUE, edgecolor="white", linewidth=0.8); ax.axvline(rv, color=INK, lw=2); ymax = ax.get_ylim()[1]; ax.set_ylim(0, ymax * 1.7)
        B = R[k]["B"]; n_le = o["dNP"]["n_null_at_or_below_real"]; ptxt = f"p < {1 / (B + 1) + 1e-9:.3f}" if n_le == 0 else f"p = {o['dNP']['p']:.3f}"
        ax.text(0.98, 0.97, f"real search {rv:+.1f} (black line)\n{n_le} of {B} null searches\nat or below it; {ptxt}", transform=ax.transAxes, ha="right", va="top", fontsize=9, color=INK)
        ax.set_title(t, fontsize=10, loc="left"); ax.set_xlabel(xlab); ax.set_xlim(np.floor(lo), np.ceil(hi))
    axes[0].set_ylabel("Label permutations")
    if LIMG == "clean": fig.suptitle("Clean-tile L-IMG", fontsize=11, x=0.01, ha="left"); fig.tight_layout(rect=(0, 0, 1, 0.95))
    else: fig.tight_layout()
    fig.savefig(OD + "/" + fname, dpi=200); plt.close(fig)
hist_fig("tol1.0", f"fig_p2n{SFX}_null.png", "Best cell: Δ NP patients flagged vs R49\n(of 43; caught ≥ R49 − 1)")
hist_fig("tier0_tol1.0", f"fig_p2n{SFX}_null_tier0.png", "Best no-sequencing cell: Δ NP flagged vs R49\n(of 43; caught ≥ R49 − 1)")
fig, axes = plt.subplots(1, len(PAN), figsize=(3.6 * len(PAN), 3.4), sharex=True, sharey=True); axes = np.atleast_1d(axes)
for ax, (k, t) in zip(axes, PAN):
    o = R[k]["tol1.0"]; x = np.array(o["null_dC_values"], float); y = np.array(o["null_dNP_values"], float); rng = np.random.RandomState(0)
    ax.scatter(x + rng.uniform(-0.04, 0.04, len(x)), y, s=10, color=BLUE, alpha=0.35, linewidths=0)
    ax.scatter([o["real_dC"]], [o["dNP"]["real"]], s=60, color=INK, zorder=3); ax.annotate("real", (o["real_dC"], o["dNP"]["real"]), xytext=(6, 4), textcoords="offset points", fontsize=9, color=INK)
    ax.set_title(t, fontsize=10, loc="left"); ax.set_xlabel("Δ progressor patients caught vs R49 (of 32)")
axes[0].set_ylabel("Δ NP patients flagged vs R49 (of 43)")
fig.tight_layout(); fig.savefig(OD + f"/fig_p2n{SFX}_null_scatter.png", dpi=200); plt.close(fig); print("figs written", OD)

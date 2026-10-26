"""Paper C draft figures (public PerturbQA results only). Usage: python pc_figs.py RESULTS_DIR OUT_DIR
(a) fig_a_dumbbell.png   raw vs two-way corrected AUROC for every method, DE and DIR
(b) fig_b_variance.png   variance decomposition of the DIR score: knockdown / gene / interaction share
(c) fig_c_shortcuts.png  list-size and shift-only shortcut baselines, raw vs corrected
(d) fig_d_copy.png       "the LLM copies the retrieval": per-knockdown mean score of the retrieval-prompted LLM vs its own kNN
Inputs: results.json (always), robust.json and copy_scatter.json (for b and d; skipped with a message if absent).
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
res = json.load(open(f"{R}/results.json"))
plt.rcParams.update({"font.size": 8, "figure.dpi": 200, "savefig.bbox": "tight", "axes.spines.top": False, "axes.spines.right": False})
RAWC, CORC = "#9aa0a6", "#1f5fa9"

LABEL = {"knn_mean_k20": "kNN mean-sim k20 (retrieval)", "knn_mean_k10": "kNN mean-sim k10", "knn_kg_k20": "kNN KG-only k20",
         "corevote_meansim": "CORE scheme, our similarity", "corevote_full": "CORE-Voting re-impl. (full)",
         "corevote_contrast": "CORE re-impl. contrast only", "corevote_anchor": "CORE re-impl. anchor only",
         "F_zero_shot": "Qwen3-8B zero-shot", "F_sft": "Qwen3-8B SFT", "F_grpo": "Qwen3-8B SFT→GRPO",
         "sft_r": "SFT, retrieval prompt", "sft_kr": "SFT, descriptions + retrieval", "grpo_kr": "GRPO on descriptions + retrieval",
         "listsize_only": "list size only", "shiftknn_k20": "predicted shift (label-free)", "shiftknn_k10": "predicted shift k10",
         "shiftknn_k50": "predicted shift k50", "shift_ridge": "predicted shift, ridge", "shift_kridge": "predicted shift, kernel ridge",
         "knn_k20_shiftpart": "kNN per-knockdown mean (shift part)", "oracle_loo": "oracle, leave-one-out (raw only)", "oracle_self": "oracle, self-inclusive (leaky)",
         "pc2_think_plain": "Qwen3-8B thinking, zero-shot", "pc2_think_r": "Qwen3-8B thinking + retrieval"}
MAIN = ["knn_mean_k20", "knn_mean_k10", "knn_kg_k20", "corevote_meansim", "corevote_full", "corevote_contrast", "corevote_anchor",
        "pc2_think_r", "pc2_think_plain", "sft_r", "sft_kr", "grpo_kr", "F_sft", "F_grpo", "F_zero_shot", "listsize_only", "shiftknn_k20"]
RAW_ONLY = {"oracle_loo", "oracle_self"}   # the oracles use test labels; no corrected value is meaningful (anti-correlated by construction)
SHORT = ["listsize_only", "shiftknn_k10", "shiftknn_k20", "shiftknn_k50", "shift_ridge", "shift_kridge", "knn_k20_shiftpart", "oracle_loo"]


def pt(tag, m):
    return res[tag][m]["mean"] if tag in res and m in res[tag] else None


def dumbbell(ax, tags, raw_m, cor_m, title):
    tags = [t for t in tags if pt(t, raw_m) and pt(t, cor_m)]
    tags = sorted(tags, key=lambda t: (pt(t, raw_m)[0] if t in RAW_ONLY else pt(t, cor_m)[0]))
    y = np.arange(len(tags))
    for i, t in enumerate(tags):
        r, c = pt(t, raw_m), pt(t, cor_m)
        if t not in RAW_ONLY:
            ax.plot([r[0], c[0]], [i, i], color="#cfd4d9", lw=1.4, zorder=1)
        ax.plot(r[0], i, "o", ms=4.5, color=RAWC, zorder=2)
        if t not in RAW_ONLY:
            ax.plot(c[0], i, "o", ms=4.5, color=CORC, zorder=3)
            ax.hlines(i, c[1], c[2], color=CORC, lw=1.0, alpha=0.55, zorder=2)
    ax.axvline(0.5, color="#b00020", lw=0.8, ls="--", zorder=0)
    ax.set_yticks(y); ax.set_yticklabels([LABEL.get(t, t) for t in tags])
    ax.set_xlabel("AUROC (per-gene)"); ax.set_title(title, loc="left", fontsize=9)
    ax.set_xlim(0.40, 0.82)


fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.6)); fig.subplots_adjust(wspace=0.62)
dumbbell(axes[0], MAIN, "DE_raw", "DE_2w", "a1  Differential expression")
dumbbell(axes[1], MAIN, "DIR_raw", "DIR_2w", "a2  Direction")
h = [plt.Line2D([], [], marker="o", ls="", color=RAWC, label="raw auc_per_gene"),
     plt.Line2D([], [], marker="o", ls="", color=CORC, label="two-way corrected (95% CI)"),
     plt.Line2D([], [], color="#b00020", ls="--", lw=0.8, label="chance")]
fig.legend(handles=h, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.04))
fig.savefig(f"{OUT}/fig_a_dumbbell.png"); plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.3)); fig.subplots_adjust(wspace=0.66)
dumbbell(axes[0], SHORT, "DE_raw", "DE_2w", "c1  Shortcut baselines, DE")
dumbbell(axes[1], SHORT, "DIR_raw", "DIR_2w", "c2  Shortcut baselines, direction")
fig.legend(handles=h, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.1))
fig.savefig(f"{OUT}/fig_c_shortcuts.png"); plt.close(fig)

# (b) variance decomposition of the DIR score: how much of the score's variance an additive knockdown + gene model explains
vd = f"{R}/vardecomp.json"
if os.path.exists(vd):
    V = json.load(open(vd))
    tags = [t for t in MAIN if t in V and not any(np.isnan(V[t]["DIR"][k]) for k in ("knockdown", "gene", "interaction"))]
    fig, ax = plt.subplots(figsize=(6.6, 0.32 * len(tags) + 1.3))
    y = np.arange(len(tags)); cols = ["#1f5fa9", "#6fa8d6", "#d9dde1"]
    left = np.zeros(len(tags))
    for part, col in zip(["knockdown", "gene", "interaction"], cols):
        w = np.array([V[t]["DIR"][part] for t in tags])
        ax.barh(y, w, left=left, color=col, label=part, height=0.7); left += w
    ax.set_yticks(y); ax.set_yticklabels([LABEL.get(t, t) for t in tags]); ax.set_xlim(0, 1)
    ax.set_xlabel("share of the direction score's variance"); ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.28))
    ax.set_title("b  Variance decomposition of the direction score", loc="left", fontsize=9)
    fig.savefig(f"{OUT}/fig_b_variance.png"); plt.close(fig)
else:
    print("no vardecomp.json; skipped figure b")

# (d) LLM copies retrieval
cs = f"{R}/copy_scatter.json"
if os.path.exists(cs):
    C = json.load(open(cs)); cells = list(C)
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.6))
    for ax, key, name in zip(axes, ["pert_de", "pert_dir"], ["d1  DE score", "d2  direction score"]):
        for c in cells:
            a, b = C[c][key]
            ax.plot(b, a, "o", ms=3, alpha=0.55, label=c.upper())
        lo = min(min(min(C[c][key][0]), min(C[c][key][1])) for c in cells)
        hi = max(max(max(C[c][key][0]), max(C[c][key][1])) for c in cells)
        ax.plot([lo, hi], [lo, hi], color="#b00020", lw=0.8, ls="--")
        ax.set_xlabel("kNN k10 retrieval, per-knockdown mean"); ax.set_ylabel("retrieval-prompted LLM, per-knockdown mean")
        ax.set_title(name, loc="left", fontsize=9)
    rho = {k: np.mean([C[c][k] for c in cells]) for k in ("pair_de_spearman", "pair_dir_spearman")}
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle(f"d  The retrieval-prompted LLM tracks its own retrieval (per-pair Spearman: DE {rho['pair_de_spearman']:.2f}, direction {rho['pair_dir_spearman']:.2f})",
                 fontsize=9, x=0.02, ha="left")
    fig.savefig(f"{OUT}/fig_d_copy.png"); plt.close(fig)
else:
    print("no copy_scatter.json; skipped figure d")
print("FIGS DONE ->", OUT)

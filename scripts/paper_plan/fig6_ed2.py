"""Extended Data Figure 2, sequencing batch (docs/paper_figures_v6.md @ d46b9a5; R31), from results/paper_final/review_round4/r31.json. (a) batch x class
composition of the 571 pre-event biopsies (12 batch labels; single-class batches marked); (b) fold-stratified AUROCs of the three models on all within-fold
pairs and on pairs restricted to the same batch or the same sequencing run; (c) one-vs-rest batch probe on the UNI2-h slide-mean embedding, per batch
with >= 5 patients, on all pre-event biopsies and on non-progressor biopsies only. Runs locally."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fig6_style import *
R = J("review_round4/r31.json")["R31"]; tab = R["a_cross_tabulation"]["batch"]["by_pre_event_sample"]["table"]; fisher = R["a_cross_tabulation"]["batch"]["by_pre_event_sample"]["fisher"]
C = R["c_within_batch_pairs"]; Pb = R["d_batch_probe_uni2h"]; names = sorted(tab, key=lambda k: -(tab[k]["progressor"] + tab[k]["non_progressor"]))
fig = plt.figure(figsize=(183 * MM, 80 * MM))
ax = fig.add_axes([0.14, 0.22, 0.22, 0.66])
for i, b in enumerate(names):
    p, q = tab[b]["progressor"], tab[b]["non_progressor"]; y_ = len(names) - 1 - i; single = p == 0 or q == 0
    ax.barh(y_, p, color=CLASS["P"], height=0.7); ax.barh(y_, q, left=p, color=CLASS["N"], height=0.7)
    if single: ax.text(p + q + 1.5, y_, "single class", fontsize=5.2, va="center", color="#333333")
ax.set_yticks(range(len(names))); ax.set_yticklabels(names[::-1], fontsize=5.8); ax.set_xlabel("pre-event biopsies"); ax.barh([-5], [0], color=CLASS["P"], label="progressor"); ax.barh([-5], [0], color=CLASS["N"], label="non-progressor"); ax.set_ylim(-0.6, len(names) - 0.4)
ax.legend(loc="lower right", fontsize=5.6); ax.set_title(f"Fisher (Monte Carlo) p {'< 0.0001' if fisher['p'] < 1e-4 else fisher['p']}", fontsize=6); letter(fig, ax, "a", dx=-0.13, dy=0.02)
bx = fig.add_axes([0.46, 0.22, 0.18, 0.66]); sets = [("all within-fold pairs", {m: {"value": C["unstratified_reference"][m], "ci95": J("triage/their_pre.json")["section1"]["auroc"][m]["ci95"]} for m in ("L-CNV", "L-IMG", "L-LATE")}), ("same-batch pairs", C["batch"]["auroc"]), ("same-run pairs", C["slx_run"]["auroc"])]
for i, (lab, d) in enumerate(sets):
    for k, m in enumerate(("L-CNV", "L-IMG", "L-LATE")): ci_point(bx, i + (k - 1) * 0.22, d[m]["value"], d[m]["ci95"], MODEL[m], ms=3.5, label=MODEL_NAME[m] if i == 0 else None)
bx.set_xticks(range(3)); bx.set_xticklabels([s[0].replace(" pairs", "\npairs") for s in sets], fontsize=5.6); bx.set_ylim(0.6, 1.0); bx.set_ylabel("AUROC, fold-stratified [95% CI]"); bx.legend(loc="lower center", bbox_to_anchor=(0.5, -0.30), ncol=3, fontsize=5.4); letter(fig, bx, "b", dx=-0.07, dy=0.02)
d = C["batch"]["IMG_minus_CNV"]; bx.text(1, 0.615, f"same batch: L-IMG − L-CNV {signed(d['delta'])}\nbound {mi(d['one_sided_95_lower'])}", ha="center", fontsize=5.2, color="#333333")
cx = fig.add_axes([0.79, 0.22, 0.20, 0.66]); pb = Pb["pre_event_samples"]["batches"]; nb = Pb["non_progressor_pre_event_samples"]["batches"]; bl = sorted(pb, key=lambda k: -pb[k]["auroc_one_vs_rest"])
for i, b in enumerate(bl):
    cx.barh(i + 0.18, pb[b]["auroc_one_vs_rest"], height=0.36, color="#555555", label="all pre-event biopsies" if i == 0 else None)
    if b in nb: cx.barh(i - 0.18, nb[b]["auroc_one_vs_rest"], height=0.36, color="#BBBBBB")
cx.axvline(0.5, color="black", lw=0.6, ls=":"); cx.set_yticks(range(len(bl))); cx.set_yticklabels([f"{b} ({pb[b]['n_patients']} pts)" for b in bl], fontsize=5.4); cx.set_xlim(0.3, 1.0); cx.set_xlabel("batch probe AUROC (one vs rest)")
cx.barh([-5], [0], height=0.36, color="#BBBBBB", label="non-progressor biopsies only"); cx.set_ylim(len(bl) - 0.5, -0.6); cx.legend(loc="lower right", fontsize=5.2); cx.set_title(f"mean {Pb['pre_event_samples']['auroc_mean_over_batches']:.3f}; non-progressors {Pb['non_progressor_pre_event_samples']['auroc_mean_over_batches']:.3f}", fontsize=6); letter(fig, cx, "c", dx=-0.13, dy=0.02)
save(fig, "ED2"); dump("ED2", {"batch_by_class_pre_event": tab, "fisher": fisher, "auroc_within_pairs": {s[0]: s[1] for s in sets}, "IMG_minus_CNV_within_batch": d, "batch_probe": Pb})

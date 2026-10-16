"""Extended Data Figure 2 (manuscript v7), sequencing batch, from results/paper_final/review_round4/r31.json. (a) batch x class composition of the
pre-event biopsies with single-class batches marked; (b) fold-stratified AUROCs of the models on all within-fold pairs and on same-batch / same-run pairs,
with batch alone (fold-honest target encoding) as a raw score; (c) one-vs-rest batch probe on the UNI2-h embedding per batch (>= 5 patients), on all
pre-event biopsies and within non-progressor biopsies, with mixed and single-class batches labelled. Runs locally."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fig7_style import *
R = J("review_round4/r31.json")["R31"]; tab = R["a_cross_tabulation"]["batch"]["by_pre_event_sample"]["table"]; fisher = R["a_cross_tabulation"]["batch"]["by_pre_event_sample"]["fisher"]
C = R["c_within_batch_pairs"]; Pb = R["d_batch_probe_uni2h"]; raw = R["b_raw_score_auroc"]; names = sorted(tab, key=lambda k: -(tab[k]["progressor"] + tab[k]["non_progressor"])); single = {b: (tab[b]["progressor"] == 0 or tab[b]["non_progressor"] == 0) for b in tab}
fig = plt.figure(figsize=(183 * MM, 80 * MM)); ax = fig.add_axes([0.14, 0.22, 0.22, 0.66])
for i, b in enumerate(names):
    p, q = tab[b]["progressor"], tab[b]["non_progressor"]; y_ = len(names) - 1 - i; ax.barh(y_, p, color=CLASS["P"], height=0.7); ax.barh(y_, q, left=p, color=CLASS["N"], height=0.7)
    if single[b]: ax.text(p + q + 1.5, y_, "single class", fontsize=5.2, va="center", color="#333333")
ax.set_yticks(range(len(names))); ax.set_yticklabels(names[::-1], fontsize=5.8); ax.set_xlabel("pre-event biopsies"); ax.barh([-5], [0], color=CLASS["P"], label="progressor"); ax.barh([-5], [0], color=CLASS["N"], label="non-progressor"); ax.set_ylim(-0.6, len(names) - 0.4)
ax.legend(loc="lower right", fontsize=5.6); ax.set_title(f"Fisher (Monte Carlo) p {'< 0.0001' if fisher['p'] < 1e-4 else fisher['p']}", fontsize=6); letter(fig, ax, "a", dx=-0.13, dy=0.02)
bx = fig.add_axes([0.46, 0.22, 0.20, 0.66]); sets = [("all\npairs", {m: {"value": C["unstratified_reference"][m], "ci95": J("triage/their_pre.json")["section1"]["auroc"][m]["ci95"]} for m in ("L-CNV", "L-IMG", "L-LATE")}), ("same\nbatch", C["batch"]["auroc"]), ("same\nrun", C["slx_run"]["auroc"])]
for i, (lab, d) in enumerate(sets):
    for k, m in enumerate(("L-CNV", "L-IMG", "L-LATE")): ci_point(bx, i + (k - 1) * 0.22, d[m]["value"], d[m]["ci95"], MODEL[m], ms=3.5, label=MODEL_NAME[m] if i == 0 else None)
ci_point(bx, 3, raw["batch_target_encoded"]["value"], raw["batch_target_encoded"]["ci95"], GREY, ms=3.5, label="batch alone (target-encoded)")
bx.set_xticks(range(4)); bx.set_xticklabels([s[0] for s in sets] + ["batch\nalone"], fontsize=5.6); bx.set_ylim(0.6, 1.0); bx.set_xlim(-0.6, 3.6); bx.set_ylabel("AUROC, fold-stratified [95% CI]"); bx.legend(loc="lower center", bbox_to_anchor=(0.5, -0.32), ncol=2, fontsize=5.2); letter(fig, bx, "b", dx=-0.07, dy=0.02)
d = C["batch"]["IMG_minus_CNV"]; bx.text(1, 0.615, f"same batch: L-IMG − L-CNV\n{signed(d['delta'])}, bound {mi(d['one_sided_95_lower'])}", ha="center", fontsize=5.0, color="#333333")
cx = fig.add_axes([0.80, 0.22, 0.17, 0.66]); pb = Pb["pre_event_samples"]["batches"]; nb = Pb["non_progressor_pre_event_samples"]["batches"]; bl = sorted(pb, key=lambda k: -pb[k]["auroc_one_vs_rest"])
for i, b in enumerate(bl):
    cx.barh(i + 0.18, pb[b]["auroc_one_vs_rest"], height=0.36, color="#555555")
    if b in nb: cx.barh(i - 0.18, nb[b]["auroc_one_vs_rest"], height=0.36, color="#BBBBBB")
cx.axvline(0.5, color="black", lw=0.6, ls=":"); cx.set_yticks(range(len(bl))); cx.set_yticklabels([f"{b} ({pb[b]['n_patients']} pts; {'single-class' if single[b] else 'mixed'})" for b in bl], fontsize=5.2); cx.set_xlim(0.3, 1.0); cx.set_xlabel("batch probe AUROC (one vs rest)")
cx.barh([-5], [0], height=0.36, color="#555555", label="all pre-event biopsies"); cx.barh([-5], [0], height=0.36, color="#BBBBBB", label="non-progressor biopsies only"); cx.set_ylim(len(bl) - 0.5, -0.6); cx.legend(loc="lower center", bbox_to_anchor=(0.5, -0.32), fontsize=5.2)
cx.set_title(f"mean {Pb['pre_event_samples']['auroc_mean_over_batches']:.3f} (NP only {Pb['non_progressor_pre_event_samples']['auroc_mean_over_batches']:.3f})", fontsize=6); letter(fig, cx, "c", dx=-0.17, dy=0.02)
save(fig, "ED2"); dump("ED2", {"batch_by_class_pre_event": tab, "single_class": single, "fisher": fisher, "auroc_within_pairs": {s[0].replace("\n", " "): s[1] for s in sets}, "batch_alone_target_encoded": raw["batch_target_encoded"], "IMG_minus_CNV_within_batch": d, "batch_probe": Pb})

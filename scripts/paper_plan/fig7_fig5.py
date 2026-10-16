"""Figure 5 (manuscript v7), robustness: three forest plots on one row list (single model throughout), from the round-4 robustness tables
(results/paper_final/review_round4/r33_merge.json: triage vs B*, the registered test vs (B), L-IMG vs L-CNV), the within-batch row (review_round4/
r31.json) and the round-6 label subsets (review_round6/r38_merge.json). Two-sided 95% CIs, truncated CIs drawn with arrow heads, the one-sided 95% lower
bound as a tick, the -0.05 margin dashed; a row is 'outside' when its bound is <= -0.05 (the pre-specified rule; -0.050 counts as outside). Runs locally."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fig7_style import *
R = J("review_round4/r33_merge.json")["R33"]; T = {r["analysis"]: r for r in R["claim_T_triage_vs_Bstar_training_matched_single_model"]}; H = {r["analysis"]: r for r in R["claim_H_LIMG_vs_LCNV_dAUROC"]}
PB = {r["analysis"]: (r.get("no_vote") or {}) | {"status": r["status"]} for r in R["prespecified_test_triage_vs_B_voted_and_single_model"]}
wb = J("review_round4/r31.json")["R31"]["c_within_batch_pairs"]["batch"]; H["within-batch pairs"] = {"status": "done", "d_auroc": wb["IMG_minus_CNV"]["delta"], "ci95": wb["IMG_minus_CNV"]["ci95"], "one_sided_lower": wb["IMG_minus_CNV"]["one_sided_95_lower"], "within_-0.05": wb["IMG_minus_CNV"]["non_inferior_at_-0.05"]}
for r in J("review_round6/r38_merge.json")["R38"]["rows"][1:]:   # label subsets
    k = "label: " + r["analysis"]; b, bb, h = r["vs_Bstar"], r["vs_B"], r["IMG_minus_CNV"]
    T[k] = {"status": "done", "d_sens": b["d_sens"], "ci95": b["ci95"], "one_sided_lower": b["one_sided_lower"], "within_-0.05": b["within_-0.05"], "share_sequenced": r["triage"]["share_sequenced"]}
    PB[k] = {"status": "done", "d_sens": bb["d_sens"], "ci95": bb["ci95"], "one_sided_lower": bb["one_sided_lower"], "within_-0.05": bb["one_sided_lower"] > -0.05}
    H[k] = {"status": "done", "d_auroc": h["delta"], "ci95": h["ci95"], "one_sided_lower": h["one_sided_lower"], "within_-0.05": h["within_-0.05"]}
for k, v in PB.items():
    if v.get("one_sided_lower") is not None and "within_-0.05" not in v: v["within_-0.05"] = v["one_sided_lower"] > -0.05
grid = lambda ql, qh: f"cut-off grid cell: t_low {ql}% training sensitivity, t_high {qh}% training specificity" + (" (primary cell)" if (ql, qh) == (95, 90) else "")
ROWS = [("Population", "Primary: all pre-event biopsies", "primary: all pre-event samples"), ("Population", "NDBE pre-event biopsies", "NDBE pre-event samples"), ("Population", "Never-dysplastic patients", "patients never dysplastic before the event"),
        ("Population", "Excl. progressor biopsies within 1 year", "excluding progressor samples within 1 year"), ("Population", "Earliest endoscopy per patient", "earliest pre-event endoscopy per patient (all its samples)"),
        ("Population", "Earliest biopsy per patient", "earliest pre-event sample per patient (Table 5)"), ("Population", "Patient-weighted", "patient-weighted (R34)"), ("Population", "Within-batch pairs", "within-batch pairs"),
        ("Labels", "'Excluded' flag removed", "label: (a) 'excluded' flag removed"), ("Labels", "All sheet flags removed", "label: (a) 'excluded', 'remove' and 'inconsistency' flags removed"),
        ("Labels", "NP biopsies < 1 y to last endoscopy removed", "label: (b) non-progressor samples < 1 year before the last sequenced endoscopy removed"), ("Labels", "NP biopsies < 2 y to last endoscopy removed", "label: (b) non-progressor samples < 2 years before the last sequenced endoscopy removed"),
        ("Labels", "NP patients with later HGD removed", "label: (c) non-progressor patients with a later HGD+ report removed"), ("Labels", "All three label exclusions", "label: (d) (a) + (b, 1 year) + (c)"),
        ("Features", "Package features, joint segmentation", "package features"), ("Features", "Package, per-sample segmentation", "per-sample re-segmented copy number (R36)"), ("Features", "Package, past-only segmentation", "past-only re-segmented copy number (R36, extra row)"),
        ("Features", "H-optimus-0 extractor", "H-optimus-0 gate (F5)"), ("Features", "Tissue- and year-adjusted H&E score", "tissue-adjusted gate (R32b)"),
        ("Training", "Pre-event-only training", "pre-event-only training (F1)"), ("Training", "Package, pre-event-only training", "package features, pre-event-only training (Table 5)"), ("Training", "λ by deviance", "lambda by deviance (B5)"),
        ("Tissue", "Slides with < 256 tissue tiles", "tissue stratum: < 256 tissue tiles (R32a)"), ("Tissue", "Slides with ≥ 256 tissue tiles", "tissue stratum: >= 256 tissue tiles (R32a)"), ("Tissue", "Tissue-matched pairs", "tissue-matched pairs: same tissue tertile (Table 5)")]
ROWS += [("Cut-offs", f"t_low {ql}% sens. / t_high {qh}% spec.", grid(ql, qh)) for ql in (90, 95, 98) for qh in (85, 90, 95) if (ql, qh) != (95, 90)]
ROWS += [("Inference", "Cut-offs re-chosen per bootstrap draw", "selection-inclusive (every cut-off re-chosen in each draw)"), ("Inference", "Evaluation-matched B*", "evaluation-matched B*"), ("Inference", "10-model vote", "10-model vote"),
         ("Inference", "Between-repeat variance added", "between-repeat variance added"), ("Inference", "Refit bootstrap, 200 draws", "refit bootstrap (F3, 200 draws)")]
fig = plt.figure(figsize=(183 * MM, 215 * MM)); ys = []; yy = 0; prev = None
for g, *_ in ROWS:
    if prev is not None and g != prev: yy += 0.9
    ys.append(yy); yy += 1; prev = g
ys = np.array(ys); top = yy - 1; Y = lambda i: top - ys[i]; src = {"rows": []}
def forest(ax, table, dkey, xlab, xlim, share=False):
    ax.axvspan(-0.05, xlim[1], color="#F4F4F4", lw=0, zorder=0); ax.axvline(-0.05, color=MODEL["L-CNV"], lw=0.8, ls="--", zorder=1); ax.axvline(0, color=LIGHT, lw=0.6, zorder=1)
    for i, (g, lab, key) in enumerate(ROWS):
        r = table.get(key); y_ = Y(i)
        if r is None or r.get("status") != "done" or r.get(dkey) is None: ax.text(0, y_, "n/a", fontsize=5.2, color=GREY, ha="center", va="center"); continue
        d, ci, lo = r[dkey], r["ci95"], r["one_sided_lower"]; ok = bool(r.get("within_-0.05")); c_ = "#222222" if ok else MODEL["L-CNV"]; primary = lab.startswith("Primary")
        ci_arrow(ax, y_, ci[0], ci[1], xlim, c_); ax.plot([lo], [y_], marker="|", color=c_, ms=4, mew=0.8, lw=0, zorder=3, clip_on=False)
        if xlim[0] <= d <= xlim[1]: ax.plot([d], [y_], marker="D" if primary else "o", color=c_, ms=4 if primary else 3.2, lw=0, zorder=4)
        txt = f"{signed(d)} [{mi(ci[0])}, {mi(ci[1])}]" + (f"  {100 * r['share_sequenced']:.0f}%" if share and r.get("share_sequenced") is not None else "")
        ax.text(xlim[1] + 0.02 * (xlim[1] - xlim[0]) / 0.16, y_, txt, fontsize=4.5, va="center", ha="left", color=c_, clip_on=False)
    ax.set_xlim(*xlim); ax.set_ylim(-0.7, top + 0.7); ax.set_yticks([]); ax.spines["left"].set_visible(False); ax.set_xlabel(xlab, fontsize=6); ax.tick_params(labelsize=5.8)
for g, lab, key in ROWS: src["rows"].append({"group": g, "label": lab, "triage_vs_Bstar": T.get(key), "registered_test_vs_B": PB.get(key), "IMG_minus_CNV": H.get(key)})
Y0, HT = 0.07, 0.855; axL = fig.add_axes([0.245, Y0, 0.11, HT]); axM = fig.add_axes([0.525, Y0, 0.105, HT]); axR = fig.add_axes([0.76, Y0, 0.105, HT])
forest(axL, T, "d_sens", "Δ sensitivity, triage − B*", (-0.10, 0.06), share=True); forest(axM, PB, "d_sens", "Δ sensitivity, triage − (B)", (-0.08, 0.12)); forest(axR, H, "d_auroc", "Δ AUROC, L-IMG − L-CNV", (-0.12, 0.16))
for ax_, t_, hd in ((axL, "Triage vs B* (matched specificity)", "Δ [95% CI]  sequenced"), (axM, "Registered test: triage vs (B)", "Δ [95% CI]"), (axR, "H&E model vs copy number", "Δ [95% CI]")):
    ax_.set_title(t_, fontsize=6.2, pad=16, loc="left"); ax_.text(ax_.get_xlim()[1] + 0.02 * (ax_.get_xlim()[1] - ax_.get_xlim()[0]) / 0.16, top + 1.2, hd, fontsize=5.0, va="bottom", ha="left", clip_on=False)
lab_ax = fig.add_axes([0.0, Y0, 0.24, HT]); lab_ax.set_xlim(0, 1); lab_ax.set_ylim(-0.7, top + 0.7); lab_ax.axis("off")
for i, (g, lab, key) in enumerate(ROWS):
    lab_ax.text(0.99, Y(i), lab, fontsize=5.2, ha="right", va="center", fontweight="bold" if lab.startswith("Primary") else "normal")
    if i == 0 or ROWS[i - 1][0] != g: lab_ax.text(0.02, Y(i) + 0.62, g, fontsize=6, ha="left", va="bottom", fontweight="bold", color="#333333")
from matplotlib.lines import Line2D
h = [Line2D([], [], marker="D", color="#222222", lw=0, ms=4, label="primary analysis"), Line2D([], [], marker="o", color="#222222", lw=0, ms=3.2, label="sensitivity analysis, bound > −0.05 (within margin)"),
     Line2D([], [], marker="o", color=MODEL["L-CNV"], lw=0, ms=3.2, label="sensitivity analysis, bound ≤ −0.05 (outside margin)"), Line2D([], [], marker="|", color="#222222", lw=0, ms=4, mew=0.8, label="one-sided 95% lower bound"),
     Line2D([], [], color=MODEL["L-CNV"], lw=0.8, ls="--", label="−0.05 margin"), Line2D([], [], marker=">", color="#222222", lw=0, ms=3.5, label="CI truncated by the axis")]
fig.legend(handles=h, loc="lower center", ncol=3, fontsize=5.6, bbox_to_anchor=(0.5, 0.0))
letter(fig, lab_ax, "a", dx=0.005, dy=0.03); letter(fig, axM, "b", dx=-0.025, dy=0.03); letter(fig, axR, "c", dx=-0.025, dy=0.03)
save(fig, "Fig5"); dump("Fig5", src)

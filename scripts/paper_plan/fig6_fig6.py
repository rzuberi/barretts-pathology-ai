"""Figure 6, robustness (docs/paper_figures_v6.md @ d46b9a5), from results/paper_final/review_round4/r33_merge.json (R33 Claim T and Claim H rows, incl.
the R36 rows) plus the within-batch Claim H row (review_round4/r31.json, Table 6). Two forest plots sharing one row list: triage - B* delta sensitivity
(single model, training-matched) and L-IMG - L-CNV delta AUROC, two-sided 95% patient-bootstrap CIs, the one-sided 95% lower bound as a tick and the
-0.05 margin. Rows grouped by type; the primary row is marked distinctly from the pre-specified sensitivity analyses (rounds 2 and 4). Runs locally."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fig6_style import *
R = J("review_round4/r33_merge.json")["R33"]; T = {r["analysis"]: r for r in R["claim_T_triage_vs_Bstar_training_matched_single_model"]}; H = {r["analysis"]: r for r in R["claim_H_LIMG_vs_LCNV_dAUROC"]}
wb = J("review_round4/r31.json")["R31"]["c_within_batch_pairs"]["batch"]; H["within-batch pairs"] = {"status": "done", "L-IMG": wb["auroc"]["L-IMG"]["value"], "L-CNV": wb["auroc"]["L-CNV"]["value"], "d_auroc": wb["IMG_minus_CNV"]["delta"], "ci95": wb["IMG_minus_CNV"]["ci95"], "one_sided_lower": wb["IMG_minus_CNV"]["one_sided_95_lower"], "within_-0.05": wb["IMG_minus_CNV"]["non_inferior_at_-0.05"]}
grid = lambda ql, qh: f"cut-off grid cell: t_low {ql}% training sensitivity, t_high {qh}% training specificity" + (" (primary cell)" if (ql, qh) == (95, 90) else "")
ROWS = [("Population", "Primary: all pre-event biopsies", "primary: all pre-event samples", "primary: all pre-event samples"), ("Population", "NDBE pre-event biopsies", "NDBE pre-event samples", "NDBE pre-event samples"),
        ("Population", "Never-dysplastic patients", "patients never dysplastic before the event", "patients never dysplastic before the event"), ("Population", "Excl. progressor biopsies within 1 year", "excluding progressor samples within 1 year", "excluding progressor samples within 1 year"),
        ("Population", "Earliest endoscopy per patient", "earliest pre-event endoscopy per patient (all its samples)", "earliest pre-event endoscopy per patient (all its samples)"), ("Population", "Earliest biopsy per patient", "earliest pre-event sample per patient (Table 5)", "earliest pre-event sample per patient (Table 5)"),
        ("Population", "Patient-weighted", "patient-weighted (R34)", "patient-weighted (R34)"), ("Population", "Within-batch pairs", None, "within-batch pairs"),
        ("Features", "Package features, joint segmentation", "package features", "package features"), ("Features", "Package, per-sample segmentation", "per-sample re-segmented copy number (R36)", "per-sample re-segmented copy number (R36)"),
        ("Features", "Package, past-only segmentation", "past-only re-segmented copy number (R36, extra row)", "past-only re-segmented copy number (R36, extra row)"), ("Features", "H-optimus-0 extractor", "H-optimus-0 gate (F5)", "H-optimus-0 gate (F5)"),
        ("Features", "Tissue- and year-adjusted H&E score", "tissue-adjusted gate (R32b)", "tissue-adjusted gate (R32b)"),
        ("Training", "Pre-event-only training", "pre-event-only training (F1)", "pre-event-only training (F1)"), ("Training", "Package, pre-event-only training", "package features, pre-event-only training (Table 5)", "package features, pre-event-only training (Table 5)"),
        ("Training", "λ by deviance", "lambda by deviance (B5)", "lambda by deviance (B5)"),
        ("Tissue", "Slides with < 256 tissue tiles", "tissue stratum: < 256 tissue tiles (R32a)", "tissue stratum: < 256 tissue tiles (R32a)"), ("Tissue", "Slides with ≥ 256 tissue tiles", "tissue stratum: >= 256 tissue tiles (R32a)", "tissue stratum: >= 256 tissue tiles (R32a)"),
        ("Tissue", "Tissue-matched pairs", None, "tissue-matched pairs: same tissue tertile (Table 5)")]
ROWS += [("Cut-offs", f"t_low {ql}% sens. / t_high {qh}% spec.", grid(ql, qh), None) for ql in (90, 95, 98) for qh in (85, 90, 95) if (ql, qh) != (95, 90)]
ROWS += [("Inference", "Cut-offs re-chosen per bootstrap draw", "selection-inclusive (every cut-off re-chosen in each draw)", None), ("Inference", "Evaluation-matched B*", "evaluation-matched B*", None), ("Inference", "10-model vote", "10-model vote", None),
         ("Inference", "Between-repeat variance added", "between-repeat variance added", "between-repeat variance added"), ("Inference", "Refit bootstrap, 200 draws", "refit bootstrap (F3, 200 draws)", "refit bootstrap (F3, 200 draws)")]
fig = plt.figure(figsize=(183 * MM, 195 * MM)); nrow = len(ROWS); groups = list(dict.fromkeys(g for g, *_ in ROWS)); ngap = len(groups) - 1
ys = []; yy = 0; prev = None
for g, *_ in ROWS:
    if prev is not None and g != prev: yy += 0.9
    ys.append(yy); yy += 1; prev = g
ys = np.array(ys); top = yy - 1; Y = lambda i: top - ys[i]
src = {"rows": []}
def forest(ax, key, src_key, xlab, xlim, dkey):
    ax.axvspan(-0.05, xlim[1], color="#F4F4F4", lw=0, zorder=0); ax.axvline(-0.05, color=MODEL["L-CNV"], lw=0.8, ls="--", zorder=1); ax.axvline(0, color=LIGHT, lw=0.6, zorder=1)
    for i, (g, lab, kT, kH) in enumerate(ROWS):
        k = kT if key == "T" else kH; r = (T if key == "T" else H).get(k) if k else None; y_ = Y(i)
        if r is None or r.get("status") != "done" or r.get(dkey) is None: ax.text(0, y_, "n/a", fontsize=5.4, color=GREY, ha="center", va="center"); continue
        d, ci, lo = r[dkey], r["ci95"], r["one_sided_lower"]; ok = bool(r.get("within_-0.05")); col = "#222222" if ok else MODEL["L-CNV"]; primary = lab.startswith("Primary")
        ax.plot(ci, [y_, y_], color=col, lw=0.9, zorder=2, solid_capstyle="butt"); ax.plot([lo], [y_], marker="|", color=col, ms=4, mew=0.8, lw=0, zorder=3)
        ax.plot([d], [y_], marker="D" if primary else "o", color=col, ms=4.2 if primary else 3.4, mfc=col, lw=0, zorder=4)
        txt = f"{signed(d)} [{mi(ci[0])}, {mi(ci[1])}]" + (f"   {100 * r['share_sequenced']:.0f}%" if key == "T" and r.get("share_sequenced") is not None else "")
        ax.text(xlim[1] + 0.012, y_, txt, fontsize=5.4, va="center", ha="left", color=col, clip_on=False)
    ax.set_xlim(*xlim); ax.set_ylim(-0.7, top + 0.7); ax.set_yticks([]); ax.spines["left"].set_visible(False); ax.set_xlabel(xlab, fontsize=6.5)
for i, (g, lab, kT, kH) in enumerate(ROWS):
    src["rows"].append({"group": g, "label": lab, "claim_T": T.get(kT) if kT else None, "claim_H": H.get(kH) if kH else None})
axL = fig.add_axes([0.235, 0.075, 0.22, 0.875]); axR = fig.add_axes([0.635, 0.075, 0.22, 0.875])
forest(axL, "T", "T", "Δ sensitivity, triage − B* (single model)", (-0.12, 0.06), "d_sens"); forest(axR, "H", "H", "Δ AUROC, L-IMG − L-CNV", (-0.14, 0.16), "d_auroc")
axL.text(0.06 + 0.012, top + 1.3, "Δ [95% CI]   sequenced", fontsize=5.6, va="bottom", ha="left", clip_on=False)
axR.text(0.16 + 0.012, top + 1.3, "Δ [95% CI]", fontsize=5.6, va="bottom", ha="left", clip_on=False)
lab_ax = fig.add_axes([0.0, 0.075, 0.23, 0.875]); lab_ax.set_xlim(0, 1); lab_ax.set_ylim(-0.7, top + 0.7); lab_ax.axis("off")
for i, (g, lab, kT, kH) in enumerate(ROWS):
    lab_ax.text(0.99, Y(i), lab, fontsize=5.6, ha="right", va="center", fontweight="bold" if lab.startswith("Primary") else "normal")
    if i == 0 or ROWS[i - 1][0] != g: lab_ax.text(0.02, Y(i) + 0.62, g, fontsize=6.2, ha="left", va="bottom", fontweight="bold", color="#333333")
axL.set_title("Triage against B* in sensitivity", fontsize=7, pad=14); axR.set_title("H&E model against copy number in AUROC", fontsize=7, pad=14)
from matplotlib.lines import Line2D
h = [Line2D([], [], marker="D", color="#222222", lw=0, ms=4.2, label="primary analysis"), Line2D([], [], marker="o", color="#222222", lw=0, ms=3.4, label="pre-specified sensitivity analysis (rounds 2 and 4), bound > −0.05"),
     Line2D([], [], marker="o", color=MODEL["L-CNV"], lw=0, ms=3.4, label="pre-specified sensitivity analysis, bound ≤ −0.05"), Line2D([], [], marker="|", color="#222222", lw=0, ms=4, mew=0.8, label="one-sided 95% lower bound"),
     Line2D([], [], color=MODEL["L-CNV"], lw=0.8, ls="--", label="−0.05 margin")]
fig.legend(handles=h, loc="lower center", ncol=3, fontsize=5.8, bbox_to_anchor=(0.5, 0.0))
letter(fig, lab_ax, "a", dx=0.005, dy=0.012); letter(fig, axR, "b", dx=-0.04, dy=0.012)
save(fig, "Fig6"); dump("Fig6", src)

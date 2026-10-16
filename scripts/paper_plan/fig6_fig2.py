"""Figure 2, discrimination (docs/paper_figures_v6.md @ d46b9a5), drawn on the cluster from the stored held-out predictions (tb_common.load, R36 per-sample
refit under killcoyne_mm/review_round4/r36_outer). (a) ROC curves pooled over the repeat-mean held-out predictions of the 571 pre-event biopsies, labelled
with the committed fold-stratified AUROCs; the triage operating point and the comparators (A), (B), (C), B*, C*, (D) from the committed tables.
(b) L-IMG repeat-mean score by class with the frozen median t_low and t_high. (c) L-IMG against L-CNV repeat-mean scores by class, within-class Spearman.
(d) precision-recall curves at the cohort prevalence. Source data: aggregate curves only (no sample identifiers)."""
import os, sys, glob, numpy as np, pandas as pd
from scipy.stats import spearmanr, rankdata
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fig6_style import *
from tb_common import *
from tb_common import _outer
O, I, FO = load("their"); rows = np.where(popmask("pre"))[0]; yr = y[rows]
Ops = _outer(sorted(glob.glob(f"{M}/review_round4/r36_outer/cnv_pkgps_rep_*.csv")))[0]; assert Ops.shape == (n, 10) and np.isfinite(Ops[rows]).all()
S = {"L-IMG": O["L-IMG"].mean(1), "L-CNV": O["L-CNV"].mean(1), "L-CNV-ps": Ops.mean(1), "L-LATE": O["L-LATE"].mean(1), "P": SM.k_prob.values.astype(float),
     "grade": SM.Pathology.map({"NDBE": 0, "ID": 1, "LGD": 2, "HGD": 3, "IMC": 4}).values.astype(float)}
def roc(s, yy):
    o = np.argsort(-s, kind="mergesort"); s, yy = s[o], yy[o]; P = yy.sum(); N = len(yy) - P; tp = np.cumsum(yy); fp = np.cumsum(1 - yy)
    keep = np.r_[np.diff(s) != 0, True]; return np.r_[0, fp[keep] / N], np.r_[0, tp[keep] / P]
def auc_plain(s, yy): r = rankdata(s); n1 = yy.sum(); return float((r[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(yy) - n1)))
def prc(s, yy):
    o = np.argsort(-s, kind="mergesort"); s, yy = s[o], yy[o]; tp = np.cumsum(yy); k = np.arange(1, len(yy) + 1); keep = np.r_[np.diff(s) != 0, True]
    rec = tp[keep] / yy.sum(); prec = tp[keep] / k[keep]; return np.r_[0, rec], np.r_[1, prec]
# committed fold-stratified AUROCs for the curve labels (Table 5)
tp = J("triage/their_pre.json")["section1"]["auroc"]; r36 = J("review_round4/r36.json")["R36"]["persample"]["pre"]["discrimination"]["auroc"]["L-CNV"]; dp = J("risk_strata/disc_pre.json")["models"]
LAB = {"L-IMG": tp["L-IMG"], "L-CNV": tp["L-CNV"], "L-CNV-ps": r36, "L-LATE": tp["L-LATE"], "P": dp["P"]["auroc"], "grade": dp["GRADE"]["auroc"]}
# committed operating points (single model unless stated): Tables 2 and 3
pr = J("review_round4/r33_base_primary.json")["R33_base"]["claim_T"]; r14 = J("review_round2/r14_their_pre.json")["R14"]["b_no_vote_paired"]; r24 = J("review_round2/r24.json")["R24"]["comparator_D"]["D1_grade_ID_or_LGD"]["sample"]
OP = {"Triage": (pr["triage_no_vote"]["sensitivity"], pr["triage_no_vote"]["specificity"]), "(A)": (r14["A"]["comparator_sensitivity"], r14["A"]["comparator_specificity"]),
      "(B)": (pr["no_vote_vs_B"]["comparator_sensitivity"], pr["no_vote_vs_B"]["comparator_specificity"]), "(C)": (pr["no_vote_vs_C"]["comparator_sensitivity"], pr["no_vote_vs_C"]["comparator_specificity"]),
      "B*": (pr["no_vote_vs_Bstar_training_matched"]["comparator_sensitivity"], pr["no_vote_vs_Bstar_training_matched"]["comparator_specificity"]),
      "C*": (pr["no_vote_vs_Cstar_training_matched"]["comparator_sensitivity"], pr["no_vote_vs_Cstar_training_matched"]["comparator_specificity"]), "(D)": (r24["sensitivity"], r24["specificity"])}
cut = J("review_round4/r33_base_primary.json")["R33_base"]["cutoffs_median"]
fig = plt.figure(figsize=(183 * MM, 150 * MM)); src = {"n_samples": int(len(rows)), "n_progressor_samples": int(yr.sum()), "prevalence": float(yr.mean())}
# ------------------------------------------------------------------ (a) ROC
ax = fig.add_axes([0.07, 0.63, 0.40, 0.34]); src["roc"] = {}
for m in ("L-IMG", "L-CNV", "L-CNV-ps", "L-LATE", "P", "grade"):
    fpr, tpr = roc(S[m][rows], yr); a = auc_plain(S[m][rows], yr); L = LAB[m]
    ax.plot(fpr, tpr, color=MODEL[m], lw=1.1 if m in ("L-IMG", "L-CNV-ps", "L-LATE") else 0.9, ls="-" if m != "grade" else "--", drawstyle="steps-post" if m == "grade" else "default",
            label=f"{MODEL_NAME[m]}: {L['value']:.3f} [{L['ci95'][0]:.3f}, {L['ci95'][1]:.3f}]")
    src["roc"][m] = {"fpr": np.round(fpr, 4).tolist(), "tpr": np.round(tpr, 4).tolist(), "auroc_pooled_repeat_mean": round(a, 3), "auroc_fold_stratified_committed": L}
ax.plot([0, 1], [0, 1], color=LIGHT, lw=0.6, ls=":")
MK = {"Triage": ("*", "black", 9), "(A)": ("o", MODEL["L-CNV"], 4), "(B)": ("o", MODEL["L-LATE"], 4), "(C)": ("o", MODEL["L-IMG"], 4), "B*": ("s", MODEL["L-LATE"], 4), "C*": ("s", MODEL["L-IMG"], 4), "(D)": ("D", MODEL["grade"], 4)}
OFF = {"Triage": (-0.115, 0.035), "(A)": (0.02, -0.03), "(B)": (-0.055, 0.0), "(C)": (0.02, -0.035), "B*": (0.02, 0.015), "C*": (0.025, 0.012), "(D)": (0.02, -0.01)}
for k, (se, sp) in OP.items():
    mk, c, ms = MK[k]; ax.plot([1 - sp], [se], marker=mk, color=c, ms=ms, mec="black", mew=0.4, lw=0, zorder=5); ax.text(1 - sp + OFF[k][0], se + OFF[k][1], k, fontsize=6, ha="left", va="center", zorder=6)
ax.set_xlabel("1 − specificity"); ax.set_ylabel("sensitivity"); ax.set_xlim(-0.01, 1); ax.set_ylim(0, 1.01); ax.set_aspect("equal")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=5.4, title="fold-stratified AUROC [95% CI]", title_fontsize=5.6, handlelength=1.4, columnspacing=1.0); src["operating_points_sens_spec"] = OP
letter(fig, ax, "a", dx=-0.05)
# ------------------------------------------------------------------ (b) L-IMG distributions by class
bx = fig.add_axes([0.58, 0.63, 0.40, 0.34]); bins = np.linspace(0, 1, 26); src["img_hist"] = {"bins": bins.tolist()}
for b, (l, r) in zip(("low", "mid", "high"), ((0, cut["t_low"]), (cut["t_low"], cut["c_star"]), (cut["c_star"], 1))): bx.axvspan(l, r, color=BAND[b], alpha=0.25, lw=0)
for cls, v in (("P", 1), ("N", 0)):
    h, _ = np.histogram(S["L-IMG"][rows][yr == v], bins); src["img_hist"][CLASS_NAME[cls]] = h.tolist()
    bx.hist(S["L-IMG"][rows][yr == v], bins, histtype="step", color=CLASS[cls], lw=1.1, label=f"{CLASS_NAME[cls]} (n {int((yr == v).sum())})"); bx.hist(S["L-IMG"][rows][yr == v], bins, histtype="stepfilled", color=CLASS[cls], alpha=0.18, lw=0)
yl = bx.get_ylim()[1]
for v, lab in ((cut["t_low"], f"t_low {cut['t_low']:.3f}"), (cut["c_star"], f"t_high {cut['c_star']:.3f}")): bx.axvline(v, color="black", lw=0.7, ls="--"); bx.text(v + 0.01, yl * 0.97, lab, fontsize=6, va="top")
for b, xc in (("mid", (cut["t_low"] + cut["c_star"]) / 2), ("high", (1 + cut["c_star"]) / 2)): bx.text(xc, yl * 0.80, b, ha="center", fontsize=6, color="#333333")
bx.text(0.004, yl * 0.80, "low", ha="left", fontsize=6, color="#333333")
bx.set_xlabel("L-IMG score (mean over the 10 repeats)"); bx.set_ylabel("pre-event biopsies"); bx.legend(loc="center right"); bx.set_xlim(0, 1)
letter(fig, bx, "b", dx=-0.06)
# ------------------------------------------------------------------ (c) L-IMG vs L-CNV
cx = fig.add_axes([0.07, 0.07, 0.40, 0.34]); src["spearman_within_class"] = {}
for cls, v in (("N", 0), ("P", 1)):
    m = yr == v; cx.scatter(S["L-IMG"][rows][m], S["L-CNV"][rows][m], s=7, color=CLASS[cls], alpha=0.65, lw=0, label=CLASS_NAME[cls]); rho = spearmanr(S["L-IMG"][rows][m], S["L-CNV"][rows][m]).correlation
    src["spearman_within_class"][CLASS_NAME[cls]] = round(float(rho), 3)
cx.text(0.03, 0.97, "\n".join(f"Spearman ρ, {CLASS_NAME[c]}: {src['spearman_within_class'][CLASS_NAME[c]]:.2f}" for c in ("P", "N")), transform=cx.transAxes, va="top", fontsize=6)
cx.set_xlabel("L-IMG score (repeat mean)"); cx.set_ylabel("L-CNV score, shipped matrix (repeat mean)"); cx.set_xlim(-0.02, 1.02); cx.set_ylim(-0.02, 1.02); cx.set_aspect("equal"); cx.legend(loc="lower right")
letter(fig, cx, "c", dx=-0.05)
# ------------------------------------------------------------------ (d) precision-recall
dx_ = fig.add_axes([0.58, 0.07, 0.40, 0.34]); src["pr"] = {}
for m in ("L-IMG", "L-CNV", "L-LATE"):
    rec, prec = prc(S[m][rows], yr); ap = float(np.sum(np.diff(rec) * prec[1:])); dx_.plot(rec, prec, color=MODEL[m], lw=1.1, label=f"{MODEL_NAME[m]} (AP {ap:.2f})"); src["pr"][m] = {"recall": np.round(rec, 4).tolist(), "precision": np.round(prec, 4).tolist(), "average_precision": round(ap, 3)}
dx_.axhline(yr.mean(), color=LIGHT, lw=0.7, ls=":"); dx_.text(0.02, yr.mean() + 0.015, f"prevalence {100 * yr.mean():.1f}%", fontsize=6, color=GREY)
dx_.set_xlabel("recall (sensitivity)"); dx_.set_ylabel("precision (PPV at cohort prevalence)"); dx_.set_xlim(0, 1); dx_.set_ylim(0, 1.02); dx_.legend(loc="lower left"); dx_.set_aspect("equal")
letter(fig, dx_, "d", dx=-0.06)
save(fig, "Fig2"); dump("Fig2", src)

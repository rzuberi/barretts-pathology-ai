"""Figure 2 (manuscript v7), discrimination, drawn on the cluster from the stored held-out predictions. (a) single-model ROC curves: for each fitted model
the ROC of each repeat's held-out predictions is averaged over the 10 repeats (vertical averaging on a common false-positive grid), so curves and the
single-model operating points (means over repeats) share one basis; the published model P is one curve and the pathologist's grade a connected ROC
through its three thresholds; p53 alone and current care plus p53 are shown on the 450-biopsy p53 subset. Legend: committed fold-stratified AUROCs
(fitted models) and plain AUROCs (grade, P). (b) repeat-mean L-IMG score by class as densities with the frozen cut-offs; (c) repeat-mean L-IMG against
L-CNV with within-class Spearman correlations in the colour key. Aggregate curves only in the source data."""
import os, sys, glob, numpy as np, pandas as pd
from scipy.stats import spearmanr, rankdata
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fig7_style import *
from tb_common import *
from tb_common import _outer
O, I, FO = load("their"); rows = np.where(popmask("pre"))[0]; yr = y[rows]
Ops = _outer(sorted(glob.glob(f"{M}/review_round4/r36_outer/cnv_pkgps_rep_*.csv")))[0]; assert Ops.shape == (n, 10) and np.isfinite(Ops[rows]).all()
MAT = {"L-IMG": O["L-IMG"], "L-CNV": O["L-CNV"], "L-CNV-ps": Ops, "L-LATE": O["L-LATE"]}; PK = SM.k_prob.values.astype(float); GRD = SM.Pathology.map({"NDBE": 0, "ID": 1, "LGD": 2, "HGD": 3, "IMC": 4}).values.astype(float)
def roc(s, yy):
    o = np.argsort(-s, kind="mergesort"); s, yy = s[o], yy[o]; P = yy.sum(); N = len(yy) - P; tp = np.cumsum(yy); fp = np.cumsum(1 - yy); keep = np.r_[np.diff(s) != 0, True]
    return np.r_[0, fp[keep] / N], np.r_[0, tp[keep] / P]
def auc_plain(s, yy): r = rankdata(s); n1 = yy.sum(); return float((r[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(yy) - n1)))
grid = np.linspace(0, 1, 401)
def mean_roc(Smat):   # vertical average over repeats of each repeat's pooled held-out ROC
    T = []
    for j in range(10): f, t = roc(Smat[rows, j], yr); T.append(np.interp(grid, f, t))
    return grid, np.mean(T, 0), float(np.mean([auc_plain(Smat[rows, j], yr) for j in range(10)]))
tp = J("triage/their_pre.json")["section1"]["auroc"]; r36 = J("review_round4/r36.json")["R36"]["persample"]["pre"]["discrimination"]["auroc"]["L-CNV"]; dp = J("risk_strata/disc_pre.json")["models"]
LAB = {"L-IMG": tp["L-IMG"], "L-CNV": tp["L-CNV"], "L-CNV-ps": r36, "L-LATE": tp["L-LATE"], "P": dp["P"]["auroc"], "grade": dp["GRADE"]["auroc"]}
pr = J("review_round4/r33_base_primary.json")["R33_base"]["claim_T"]; r14 = J("review_round2/r14_their_pre.json")["R14"]["b_no_vote_paired"]; r24 = J("review_round2/r24.json")["R24"]["comparator_D"]["D1_grade_ID_or_LGD"]["sample"]
r37 = J("review_round6/r37.json")["R37"]["b_c_d_samples_with_p53"]["rules"]
OP = {"Triage": (pr["triage_no_vote"]["sensitivity"], pr["triage_no_vote"]["specificity"]), "(A)": (r14["A"]["comparator_sensitivity"], r14["A"]["comparator_specificity"]),
      "(B)": (pr["no_vote_vs_B"]["comparator_sensitivity"], pr["no_vote_vs_B"]["comparator_specificity"]), "(C)": (pr["no_vote_vs_C"]["comparator_sensitivity"], pr["no_vote_vs_C"]["comparator_specificity"]),
      "B*": (pr["no_vote_vs_Bstar_training_matched"]["comparator_sensitivity"], pr["no_vote_vs_Bstar_training_matched"]["comparator_specificity"]),
      "C*": (pr["no_vote_vs_Cstar_training_matched"]["comparator_sensitivity"], pr["no_vote_vs_Cstar_training_matched"]["comparator_specificity"]), "(D)": (r24["sensitivity"], r24["specificity"]),
      "p53": (r37["p53"]["sample"]["sensitivity"], r37["p53"]["sample"]["specificity"]), "(D)+p53": (r37["D+p53"]["sample"]["sensitivity"], r37["D+p53"]["sample"]["specificity"])}
cut = J("review_round4/r33_base_primary.json")["R33_base"]["cutoffs_median"]
fig = plt.figure(figsize=(183 * MM, 95 * MM)); src = {"n_samples": int(len(rows)), "n_progressor_samples": int(yr.sum()), "roc": {}}
# ------------------------------------------------------------------ (a)
ax = fig.add_axes([0.05, 0.33, 0.30, 0.62])
for m in ("L-IMG", "L-CNV", "L-CNV-ps", "L-LATE"):
    f, t, a = mean_roc(MAT[m]); L = LAB[m]; ax.plot(f, t, color=MODEL[m], lw=1.1, label=f"{MODEL_NAME[m]}: {L['value']:.3f} [{L['ci95'][0]:.3f}, {L['ci95'][1]:.3f}]")
    src["roc"][m] = {"fpr_grid": np.round(f[::4], 4).tolist(), "tpr_mean_over_repeats": np.round(t[::4], 4).tolist(), "auroc_pooled_mean_over_repeats": round(a, 3), "auroc_fold_stratified_committed": L}
f, t = roc(PK[rows], yr); ax.plot(f, t, color=MODEL["P"], lw=0.9, label=f"{MODEL_NAME['P']}: {LAB['P']['value']:.3f} [{LAB['P']['ci95'][0]:.3f}, {LAB['P']['ci95'][1]:.3f}]"); src["roc"]["P"] = {"fpr": np.round(f, 4).tolist(), "tpr": np.round(t, 4).tolist(), "auroc_plain": round(auc_plain(PK[rows], yr), 3), "auroc_committed": LAB["P"]}
gf, gt = [1.0], [1.0]
for thr in (0, 1, 2): pos = GRD[rows] > thr; gf.append(float(pos[yr == 0].mean())); gt.append(float(pos[yr == 1].mean()))
gf.append(0.0); gt.append(0.0); ax.plot(gf, gt, color=MODEL["grade"], lw=0.9, ls="--", marker="o", ms=2.5, label=f"{MODEL_NAME['grade']}: {LAB['grade']['value']:.3f} [{LAB['grade']['ci95'][0]:.3f}, {LAB['grade']['ci95'][1]:.3f}]"); src["roc"]["grade"] = {"fpr": gf, "tpr": gt, "thresholds": "> NDBE, > ID, > LGD", "auroc_committed": LAB["grade"]}
ax.plot([0, 1], [0, 1], color=LIGHT, lw=0.6, ls=":")
MK = {"Triage": ("*", "black", 9), "(A)": ("o", MODEL["L-CNV"], 4), "(B)": ("o", MODEL["L-LATE"], 4), "(C)": ("o", MODEL["L-IMG"], 4), "B*": ("s", MODEL["L-LATE"], 4), "C*": ("s", MODEL["L-IMG"], 4), "(D)": ("D", MODEL["grade"], 4), "p53": ("v", MODEL["p53"], 4.5), "(D)+p53": ("^", MODEL["p53"], 4.5)}
OFF = {"Triage": (-0.125, 0.035), "(A)": (0.02, -0.03), "(B)": (-0.06, 0.0), "(C)": (0.02, -0.035), "B*": (0.02, 0.015), "C*": (0.025, 0.012), "(D)": (0.02, -0.01), "p53": (0.02, 0.0), "(D)+p53": (0.02, 0.0)}
for k, (se, sp) in OP.items():
    mk, c, ms = MK[k]; ax.plot([1 - sp], [se], marker=mk, color=c, ms=ms, mec="black", mew=0.4, lw=0, zorder=5, mfc=c if k not in ("p53", "(D)+p53") else "white"); ax.text(1 - sp + OFF[k][0], se + OFF[k][1], k, fontsize=5.8, ha="left", va="center", zorder=6)
ax.set_xlabel("1 − specificity"); ax.set_ylabel("sensitivity"); ax.set_xlim(-0.01, 1); ax.set_ylim(0, 1.01); ax.set_aspect("equal")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=1, fontsize=5.3, title="fold-stratified AUROC [95% CI]; plain AUROC for P and grade", title_fontsize=5.5, handlelength=1.4); src["operating_points_sens_spec"] = OP
letter(fig, ax, "a", dx=-0.035)
# ------------------------------------------------------------------ (b)
bx = fig.add_axes([0.44, 0.33, 0.25, 0.62]); img = O["L-IMG"].mean(1)[rows]; bins = np.linspace(0, 1, 26); src["img_density"] = {"bins": bins.tolist()}
for b, (l, r) in zip(("low", "mid", "high"), ((0, cut["t_low"]), (cut["t_low"], cut["c_star"]), (cut["c_star"], 1))): bx.axvspan(l, r, color=BAND[b], alpha=0.25, lw=0)
for cls, v in (("P", 1), ("N", 0)):
    h, _ = np.histogram(img[yr == v], bins, density=True); src["img_density"][CLASS_NAME[cls]] = np.round(h, 4).tolist()
    bx.hist(img[yr == v], bins, density=True, histtype="step", color=CLASS[cls], lw=1.1, label=f"{CLASS_NAME[cls]} (n {int((yr == v).sum())})"); bx.hist(img[yr == v], bins, density=True, histtype="stepfilled", color=CLASS[cls], alpha=0.18, lw=0)
yl = bx.get_ylim()[1]
for v, lab in ((cut["t_low"], f"t_low {cut['t_low']:.3f}"), (cut["c_star"], f"t_high {cut['c_star']:.3f}")): bx.axvline(v, color="black", lw=0.7, ls="--"); bx.text(v + 0.01, yl * 0.97, lab, fontsize=5.8, va="top")
for b, xc in (("mid", (cut["t_low"] + cut["c_star"]) / 2), ("high", (1 + cut["c_star"]) / 2)): bx.text(xc, yl * 0.82, b, ha="center", fontsize=6, color="#333333")
bx.text(0.004, yl * 0.82, "low", ha="left", fontsize=6, color="#333333"); bx.set_xlabel("L-IMG score (mean over the 10 repeats)"); bx.set_ylabel("density"); bx.legend(loc="center right", fontsize=5.6); bx.set_xlim(0, 1)
letter(fig, bx, "b", dx=-0.05)
# ------------------------------------------------------------------ (c)
cx = fig.add_axes([0.745, 0.33, 0.25, 0.62]); cnv = O["L-CNV"].mean(1)[rows]; src["spearman_within_class"] = {}
for cls, v in (("N", 0), ("P", 1)):
    m = yr == v; rho = float(spearmanr(img[m], cnv[m]).correlation); src["spearman_within_class"][CLASS_NAME[cls]] = round(rho, 3)
    cx.scatter(img[m], cnv[m], s=6, color=CLASS[cls], alpha=0.65, lw=0, label=f"{CLASS_NAME[cls]}: Spearman ρ {rho:.2f}")
cx.set_xlabel("L-IMG score (repeat mean)"); cx.set_ylabel("L-CNV score, published matrix (repeat mean)"); cx.set_xlim(-0.02, 1.02); cx.set_ylim(-0.02, 1.02); cx.set_aspect("equal"); cx.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), fontsize=5.6)
letter(fig, cx, "c", dx=-0.05, dy=0.07)
save(fig, "Fig2"); dump("Fig2", src)

"""Extended Data Figure 4, calibration (docs/paper_figures_v6.md @ d46b9a5; R3), from results/paper_final/review_round1/r3.json: decile plots of the
observed progressor share against the mean repeat-mean prediction for L-CNV, L-IMG and L-LATE on the 571 pre-event biopsies, with calibration slopes
[95% CI]. Relative only in a matched case-control design (legend). Runs locally."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fig6_style import *
R = J("review_round1/r3.json")["R3"]; fig = plt.figure(figsize=(183 * MM, 62 * MM))
for q, m in enumerate(("L-CNV", "L-IMG", "L-LATE")):
    ax = fig.add_axes([0.07 + q * 0.32, 0.20, 0.25, 0.70]); d = R[m]; x = [t["mean_pred"] for t in d["deciles"]]; o = [t["observed"] for t in d["deciles"]]
    ax.plot([0, 1], [0, 1], color=LIGHT, lw=0.6, ls=":"); ax.plot(x, o, "-o", color=MODEL[m], ms=3.5, lw=0.9)
    ax.text(0.03, 0.97, f"slope {d['slope']['value']:.2f} [{d['slope']['ci95'][0]:.2f}, {d['slope']['ci95'][1]:.2f}]\nintercept {mi(d['intercept']['value'])}\nmean prediction {d['mean_pred']:.3f}; observed {d['observed']:.3f}", transform=ax.transAxes, va="top", fontsize=5.6)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal"); ax.set_xlabel("mean predicted probability (decile)"); ax.set_title(MODEL_NAME[m], fontsize=6.5)
    if q == 0: ax.set_ylabel("observed progressor share")
    letter(fig, ax, "abc"[q], dx=-0.045)
save(fig, "ED4"); dump("ED4", R)

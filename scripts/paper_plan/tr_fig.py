"""Figure 13_F_triage (docs/paper_triage.md @ f5944d2) from aggregates only: results/paper_final/triage/their_pre.json (triage with L-LATE).
Panel a: sample flow (all pre-event samples -> modal band over 10 repeats -> final >= 6/10 call), progressor / non-progressor counts per box.
Panel b: projected share sequenced against progressor-sample prevalence with 95% patient-bootstrap band. Writes ~/Downloads/be_paper_figs/v3/13_F_triage.{pdf,png}."""
import json, os, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
J = json.load(open("results/paper_final/triage/their_pre.json")); OUT = os.path.expanduser("~/Downloads/be_paper_figs/v3"); F = J["flow"]["triage_LATE"]
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
fig = plt.figure(figsize=(13, 5.2)); ax = fig.add_axes([0.01, 0.04, 0.60, 0.86]); ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
COL = {"low": "#1f77b4", "mid": "#ff7f0e", "high": "#d62728", "all": "#7f7f7f"}
def box(x, yc, w, h, title, p, np_, col):
    ax.add_patch(FancyBboxPatch((x, yc - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.15", fc=col, alpha=0.13, ec=col, lw=1.4))
    ax.text(x + w / 2, yc + 0.25, title, ha="center", va="center", fontsize=8.5, weight="bold", linespacing=1.1); ax.text(x + w / 2, yc - 0.38, f"{p} progressor / {np_} non-progressor", ha="center", va="center", fontsize=8)
def arrow(x0, y0, x1, y1): ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="->", color="#555555", lw=1.1))
P = J["n_progressor_samples"]; N = J["n_samples"] - P
box(0.05, 5, 2.55, 1.5, f"All pre-event samples\n(n {J['n_samples']})", P, N, COL["all"])
names = {"low": "Low H&E score: cleared,\nnot sequenced", "mid": "Middle H&E score:\nsequenced, L-LATE call", "high": "High H&E score: flagged,\nnot sequenced"}; ys = {"low": 1.8, "mid": 5, "high": 8.2}
for b in ("low", "mid", "high"):
    f = F[b]; box(3.2, ys[b], 3.3, 1.6, names[b] + f" (n {f['progressor'] + f['non_progressor']})", f["progressor"], f["non_progressor"], COL[b]); arrow(2.6, 5, 3.2, ys[b])
    cp, cn = f["called_positive"]["progressor"], f["called_positive"]["non_progressor"]; outs = [("called high risk", cp, cn, "#d62728"), ("called low risk", f["progressor"] - cp, f["non_progressor"] - cn, "#1f77b4")]
    outs = [o for o in outs if o[1] + o[2] > 0]; off = [0.0] if len(outs) == 1 else [0.8, -0.8]   # empty (0 / 0) outcome boxes omitted
    for (t_, a_, b_, c_), dy in zip(outs, off): box(7.15, ys[b] + dy, 2.8, 1.15, t_, a_, b_, c_); arrow(6.5, ys[b], 7.15, ys[b] + dy)
ax.text(0.05, 9.85, "a  Sample flow (band = most frequent of 10 repeats; call = positive in ≥ 6 of 10 repeats)", fontsize=10, weight="bold", va="top")
bx = fig.add_axes([0.69, 0.14, 0.29, 0.70]); C = J["curve"]; pv = [100 * p for p in C["prevalence"]]
bx.fill_between(pv, [100 * v for v in C["share_mid_lo"]], [100 * v for v in C["share_mid_hi"]], color=COL["mid"], alpha=0.2, lw=0); bx.plot(pv, [100 * v for v in C["share_mid"]], color=COL["mid"], lw=2)
for (k, lab), off in zip((("0.02", "2%"), ("0.05", "5%"), ("0.10", "10%"), ("observed", "observed")), ((-4, -16), (2, 9), (2, 9), (-30, 9))):
    q = J["prevalence_projection"][k]; bx.plot(100 * q["prevalence"], 100 * q["mid"]["value"], "o", color="#333333", ms=4); bx.annotate(f"{lab}: {100 * q['mid']['value']:.0f}%", (100 * q["prevalence"], 100 * q["mid"]["value"]), textcoords="offset points", xytext=off, fontsize=8)
bx.axhline(70, color="#999999", lw=0.8, ls="--"); bx.text(30, 71, "70% criterion", ha="right", fontsize=7.5, color="#666666")
bx.set_xlim(0, 30); bx.set_ylim(0, 100); bx.set_xlabel("progressor-sample prevalence (%)"); bx.set_ylabel("samples sequenced (%)")
bx.set_title("b  Share sequenced at other prevalences\n(projection: class-wise band rates held fixed; 95% patient-bootstrap band)", fontsize=9.5, loc="left")
fig.suptitle("H&E-first triage, their CNV matrix, all pre-event samples (571 samples, 75 patients); matched case–control cohort", fontsize=10, y=0.99)
for ext in ("pdf", "png"): fig.savefig(f"{OUT}/13_F_triage.{ext}", dpi=200, bbox_inches="tight")
print("TR FIG DONE")

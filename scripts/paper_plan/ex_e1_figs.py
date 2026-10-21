"""Exploration E1 figures (local only, never committed) from the aggregate JSONs -> paper/figures_explore_E1/."""
import json, os, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
T = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); RD = T + "/results/paper_final/explore_E1"; FD = T + "/paper/figures_explore_E1"; os.makedirs(FD, exist_ok=True)
J = lambda f: json.load(open(f"{RD}/{f}.json")) if os.path.exists(f"{RD}/{f}.json") else None
BLUE, ORANGE, GREY, INK, INK2, SURF = "#2a78d6", "#eb6834", "#8a8984", "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "axes.spines.top": False, "axes.spines.right": False})
def forest(items, title, fn, xlabel):
    """items: list of (label, delta, lo, hi, colour)"""
    h = 0.32 * len(items) + 1.2; fig, ax = plt.subplots(figsize=(7.2, h)); yy = np.arange(len(items))[::-1]
    for y_, (lab, d, lo, hi, col) in zip(yy, items):
        ax.plot([lo, hi], [y_, y_], color=col, lw=2, solid_capstyle="round"); ax.plot(d, y_, "o", color=col, ms=6, mec=SURF, mew=1.5)
    ax.axvline(0, color=GREY, lw=1, ls=(0, (3, 3))); ax.set_yticks(yy); ax.set_yticklabels([i[0] for i in items]); ax.set_xlabel(xlabel); ax.set_title(title, loc="left", fontsize=10, color=INK)
    fig.tight_layout(); fig.savefig(f"{FD}/{fn}", dpi=200); plt.close(fig)
c1 = J("char")
if c1:
    for rn, tag in (("R49", "r49"), ("P1 90%", "p1")):
        r = c1["E1_1"]["rules"][rn]["endoscopy"]; it = []
        for nm, o in r.items():
            if nm.startswith("later_db"): continue
            c = o["contrasts"]["FP-NP - TP-P"]; g = o["groups"]; sd = np.sqrt(((g["FP-NP"]["sd"] or 0) ** 2 + (g["TP-P"]["sd"] or 0) ** 2) / 2)
            if not sd or c["delta"] is None or c["ci95"][0] is None: continue
            it.append((nm[:48], c["delta"] / sd, c["ci95"][0] / sd, c["ci95"][1] / sd))
        it = sorted(it, key=lambda t: -abs(t[1]))[:18]; it = [(a, b, c, d, ORANGE if (c > 0 or d < 0) else BLUE) for a, b, c, d in it]
        forest(it, f"{rn}: false-positive NP vs true-positive P endoscopies", f"e1_char_{tag}.png", "standardised difference (FP-NP − TP-P), 95% CI")
t2 = J("tiles")
if t2:
    r = t2["E1_2"]["rules"]["R49"]; it = []
    for nm, o in r.items():
        c = o["contrasts"]["FP-NP - TP-P"]; g = o["groups"]; sd = np.sqrt(((g["FP-NP"]["sd"] or 0) ** 2 + (g["TP-P"]["sd"] or 0) ** 2) / 2)
        if not sd or c["delta"] is None or c["ci95"][0] is None: continue
        it.append((nm[:48], c["delta"] / sd, c["ci95"][0] / sd, c["ci95"][1] / sd))
    it = sorted(it, key=lambda t: -abs(t[1])); it = [(a, b, c, d, ORANGE if (c > 0 or d < 0) else BLUE) for a, b, c, d in it]
    forest(it, "R49: tile-score summaries, FP-NP vs TP-P", "e1_tiles_r49.png", "standardised difference (FP-NP − TP-P), 95% CI")
sm = J("summary")
if sm:
    rows = sm["E1_3_summary"]["all"]; fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for fam, col, lab in (("score", BLUE, "score substitution"), ("veto", GREY, "single-feature veto"), ("filter", ORANGE, "second-stage filter"), ("nsamp", "#1baf7a", "sample-count penalty")):
        rr = [r for r in rows if r["family"] == fam and r.get("deployable", True)]; ax.scatter([r["np_flagged"] for r in rr], [r["caught"] for r in rr], s=16, color=col, alpha=0.75, label=lab, edgecolors="none")
    rr = [r for r in rows if not r.get("deployable", True)]
    ax.scatter([r["np_flagged"] for r in rr], [r["caught"] for r in rr], s=22, facecolors="none", edgecolors=INK2, lw=1, label="uses batch / slide metadata (not deployable)")
    b = sm["E1_3_summary"]["base"]
    for k, o in b.items():
        ax.scatter(o["value"]["non_progressor_patients_flagged"], o["value"]["progressor_patients_caught"], s=70, marker="D", color=INK, zorder=5)
        ax.annotate(k, (o["value"]["non_progressor_patients_flagged"], o["value"]["progressor_patients_caught"]), xytext=(6, -12), textcoords="offset points", fontsize=9, color=INK)
    ax.set_xlabel("non-progressor patients ever flagged (of 43)"); ax.set_ylabel("progressor patients caught (of 32)"); ax.legend(frameon=False, loc="lower left", fontsize=9)
    ax.set_title("Nested fixes against R49 and P1 90% (single model)", loc="left", fontsize=10); fig.tight_layout(); fig.savefig(f"{FD}/e1_fixes.png", dpi=200); plt.close(fig)
print("figs done", sorted(os.listdir(FD)))

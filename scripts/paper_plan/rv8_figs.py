"""Figures for docs/paper_review_round8_landscape.md (pre-specification b396cdc) from the aggregate JSONs in results/paper_final/review_round8/.
Writes PNG (200 dpi) to paper/figures_r8/ (local, not committed). RV8_JSON overrides the JSON directory (smoke checks)."""
import json, os, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
T = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
R = os.environ.get("RV8_JSON", T + "/results/paper_final/review_round8"); OUTD = os.environ.get("RV8_FIGS", T + "/paper/figures_r8"); os.makedirs(OUTD, exist_ok=True)
J = lambda f: json.load(open(f"{R}/{f}"))
plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9, "figure.dpi": 100, "savefig.dpi": 200, "axes.spines.top": False, "axes.spines.right": False})
FAM = {"P1": "P1 H&E only", "P2": "P2 copy number, sequence all", "P3": "P3 sequence all, H&E + CN", "P4": "P4 one random sample", "P5": "P5 sample-level triage", "P6": "P6 R49 gate",
       "P7": "P7 sequence once, then H&E", "P8": "P8 re-sequence when H&E rises", "P9": "P9 sequence every N years", "P10": "P10 current care (grade)"}
COL = dict(zip(FAM, ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#000000"]))
MRK = dict(zip(FAM, ["o", "s", "D", "^", "v", "P", "o", "X", "h", "*"]))
L1 = J("l1.json")["L1"]; cells = [c for c in L1["cells"] if c.get("params", {}).get("subset") is None]
def short(c):
    p = c["params"]; f = c["family"]
    if f in ("P1", "P2", "P3", "P4"): return f"{f} {int(p['q'] * 100)}"
    if f in ("P5", "P6"): return f"{f} {int(p['ql'] * 100)}/{int(p['qh'] * 100)}"
    if f == "P7": return f"P7 {dict(zip(('1 top-H&E', '1 random', '2 top-H&E', 'all'), ('top1', 'rnd1', 'top2', 'all')))[p['k']]} {int(p['qb'] * 100)}/{int(p['qs'] * 100)}"
    if f == "P8": return f"P8 {int(p['qb'] * 100)}/{int(p['qmid'] * 100)}"
    if f == "P9": return f"P9 N={p['N']}"
    return "P10 grade"
def place_labels(ax, pts, labels, fig):
    """greedy placement: for each point try offsets on growing rings; keep the first whose text box overlaps no placed box and no marker"""
    fig.canvas.draw(); rend = fig.canvas.get_renderer(); placed = []; disp = ax.transData.transform(np.array(pts)); ax_bb = ax.get_window_extent(rend)
    offs = [(r * np.cos(a), r * np.sin(a)) for r in (9, 15, 22, 30, 40, 52) for a in np.deg2rad(np.arange(0, 360, 30) + 15)]
    for (x, y_), lab, d in zip(pts, labels, disp):
        done = False
        for ox, oy in offs:
            t = ax.annotate(lab, (x, y_), xytext=(ox, oy), textcoords="offset points", fontsize=9, ha="center", va="center", fontweight="bold",
                            arrowprops=dict(arrowstyle="-", lw=0.5, color="0.35", shrinkA=0, shrinkB=3) if abs(ox) + abs(oy) > 20 else None, zorder=6)
            bb = t.get_window_extent(rend).expanded(1.15, 1.15)
            ok = not any(bb.overlaps(q) for q in placed) and ax_bb.contains(bb.x0, bb.y0) and ax_bb.contains(bb.x1, bb.y1)
            ok = ok and not any(bb.expanded(1.3, 1.3).contains(px, py) for px, py in disp)
            if ok: placed.append(bb); done = True; break
            t.remove()
        if not done: t = ax.annotate(lab, (x, y_), xytext=(0, 12), textcoords="offset points", fontsize=9, ha="center", fontweight="bold"); placed.append(t.get_window_extent(rend))
def scatter(xkey, xlabel, fname, frontier_key, sizekey=True, connect=False, title=""):
    fig, ax = plt.subplots(figsize=(11, 6.8)); YMIN = 28.4
    sz = lambda c: 25 + 22 * c["samples_sequenced_per_patient"] if sizekey else 40
    on = [c for c in cells if c["progressor_patients_caught"] >= YMIN]; off = [c for c in cells if c["progressor_patients_caught"] < YMIN]
    for f in FAM:
        cs = [c for c in on if c["family"] == f]
        if cs: ax.scatter([c[xkey] for c in cs], [c["progressor_patients_caught"] for c in cs], s=[sz(c) for c in cs], c=COL[f], marker=MRK[f], alpha=0.75, edgecolors="white", linewidths=0.5, zorder=3)
    fr = sorted([c for c in on if c[frontier_key]], key=lambda c: (c[xkey], -c["progressor_patients_caught"]))
    ax.scatter([c[xkey] for c in fr], [c["progressor_patients_caught"] for c in fr], s=[sz(c) * 2.2 + 30 for c in fr], facecolors="none", edgecolors="black", linewidths=1.1, zorder=4)
    if connect:
        ax.plot([c[xkey] for c in fr], [c["progressor_patients_caught"] for c in fr], "-", color="black", lw=1, zorder=2)
    xs = [c[xkey] for c in cells]; pad = (max(xs) - min(xs)) * 0.06; ax.set_xlim(min(xs) - pad, max(xs) + pad); ax.set_ylim(YMIN - 0.25, 32.3)
    for c in off:   # off-scale points (current care): drawn at the lower edge with their value
        ax.scatter([c[xkey]], [YMIN - 0.12], marker="v", s=70, c=COL[c["family"]], zorder=5, clip_on=False)
        ax.annotate(f"{short(c)}: {c['progressor_patients_caught']:.0f} caught (off scale)" + (" *" if c[frontier_key] else ""), (c[xkey], YMIN - 0.12), xytext=(8, 4), textcoords="offset points", fontsize=9, va="bottom")
    ax.set_xlabel(xlabel); ax.set_ylabel("Progressor patients caught (of 32)"); ax.set_title(title, loc="left"); ax.grid(alpha=0.25, lw=0.5)
    place_labels(ax, [(c[xkey], c["progressor_patients_caught"]) for c in fr], [str(i + 1) for i in range(len(fr))], fig)
    h1 = [Line2D([], [], marker=MRK[f], color="w", markerfacecolor=COL[f], markersize=8, label=FAM[f]) for f in FAM if any(c["family"] == f for c in cells)]
    h1.append(Line2D([], [], marker="o", color="w", markerfacecolor="none", markeredgecolor="black", markersize=10, label="Pareto frontier (numbered; cost / caught)" if not sizekey else "Pareto frontier (numbered; caught,\nNP ever flagged, sequencing)"))
    if sizekey: h1 += [Line2D([], [], marker="o", color="w", markerfacecolor="0.6", markersize=np.sqrt(25 + 22 * v), label=f"{v} samples sequenced / patient") for v in (0, 2, 8)]
    lg_ = ax.legend(handles=h1, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, borderaxespad=0)
    key = "\n".join(f"{i + 1:>2}  {short(c)}" for i, c in enumerate(fr)) + ("\n *  off-scale point on the frontier" if any(c[frontier_key] for c in off) else "")
    ax.text(1.02, 0.42 if sizekey else 0.55, "Frontier points\n" + key, transform=ax.transAxes, va="top", ha="left", fontsize=9, family="monospace")
    fig.tight_layout(); fig.savefig(f"{OUTD}/{fname}", bbox_inches="tight"); plt.close(fig)
scatter("non_progressor_patients_ever_flagged", "Non-progressor patients ever flagged (of 43)", "r8_a_caught_vs_np_flagged.png", "pareto_caught_falseflags_sequencing",
        title="(a) Caught against non-progressors ever flagged (marker size = samples sequenced per patient)")
scatter("flags_per_non_progressor_patient_year", "Flagged non-progressor endoscopies per non-progressor patient-year", "r8_b_caught_vs_flags_per_py.png", "pareto_caught_falseflags_sequencing",
        title="(b) Caught against flags per non-progressor patient-year (marker size = samples sequenced per patient)")
scatter("cost_per_patient_gbp", "Cost per patient (£; £85 per sequenced sample + £535 per flagged endoscopy)", "r8_c_cost_vs_caught.png", "pareto_caught_cost", sizekey=False, connect=True,
        title="(c) Cost per patient against caught (line = cost / caught Pareto frontier)")
# ---------------------------------------------------------------- L2 forest
L2 = []
for fam in ("overall", "heq", "grade", "tissue", "tte", "position", "nsamp"):
    try: L2 += J(f"l2_{fam}.json")["L2"]["strata"]
    except FileNotFoundError: pass
if L2:
    fig, axs = plt.subplots(1, 2, figsize=(11, 0.42 * len(L2) + 1.6), sharey=True); yy = np.arange(len(L2))[::-1]
    for ax, key, ttl in ((axs[0], "LATE_minus_IMG", "Δ AUROC, L-LATE − L-IMG"), (axs[1], "CNV_minus_IMG", "Δ AUROC, L-CNV − L-IMG")):
        d = np.array([s[key]["delta"] for s in L2]); lo = np.array([s[key]["ci95"][0] for s in L2]); hi = np.array([s[key]["ci95"][1] for s in L2])
        ax.errorbar(d, yy, xerr=[d - lo, hi - d], fmt="o", color="#4C72B0", ecolor="0.4", capsize=2, ms=5); ax.axvline(0, color="0.3", lw=0.8, ls="--"); ax.set_title(ttl); ax.grid(axis="x", alpha=0.3, lw=0.5)
        ax.set_xlabel("Δ fold-stratified AUROC (95% patient-bootstrap CI)")
    axs[0].set_yticks(yy); axs[0].set_yticklabels([s["stratum"] for s in L2])
    fig.suptitle("L2. Where copy number adds value: Δ AUROC by stratum (descriptive)", x=0.01, ha="left", fontsize=11); fig.tight_layout(); fig.savefig(f"{OUTD}/r8_l2_forest.png", bbox_inches="tight"); plt.close(fig)
# ---------------------------------------------------------------- L3 trajectory
L3 = J("l3.json")["L3"]; fig, axs = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
for ax, m in zip(axs, ("L-IMG", "L-CNV")):
    for cl, col, lab in (("progressor", "#C44E52", "progressors (time to event)"), ("non_progressor", "#4C72B0", "non-progressors (time to last sequenced endoscopy)")):
        tr = [t for t in L3[m]["trajectory"] if t["class"] == cl and t["mean"] is not None]
        xs = [(t["bin_years"][0] + min(t["bin_years"][1], 11)) / 2 for t in tr]; ax.errorbar(xs, [t["mean"] for t in tr], yerr=[t["se"] or 0 for t in tr], fmt="-o", color=col, capsize=3, label=lab, ms=5)
    mm = L3[m]["mixed_model"].get("tte:cls"); sub = f"\nmixed model time × class {mm['estimate']:+.3f} [{mm['ci95'][0]:+.3f}, {mm['ci95'][1]:+.3f}] per year" if mm else ""
    ax.set_xlim(10.2, -0.3); ax.set_xlabel("Years before the event / last sequenced endoscopy (bin midpoint)"); ax.set_title(f"{m}: endoscopy max logit (mean ± SE){sub}", fontsize=10); ax.grid(alpha=0.3, lw=0.5)
axs[0].set_ylabel("Mean logit score"); axs[0].legend(loc="upper left", frameon=False)
fig.suptitle("L3. Score trajectory, patients with ≥ 2 pre-event endoscopies (last bin = 8+ years)", x=0.01, ha="left", fontsize=11); fig.tight_layout(); fig.savefig(f"{OUTD}/r8_l3_trajectory.png", bbox_inches="tight"); plt.close(fig)
# ---------------------------------------------------------------- L4 bars
L4 = J("l4.json")["L4"]; fig, ax = plt.subplots(figsize=(10, 4.4)); grp = [(lv, m) for lv in ("endoscopy", "patient") for m in ("L-IMG", "L-CNV")]; W = 0.26
for i, (k, lab, col) in enumerate((("max", "maximum", "#4C72B0"), ("sd", "within-unit SD", "#DD8452"), ("max_plus_sd", "max + SD (rank sum)", "#55A868"))):
    v = [L4[lv][m]["auroc"][k]["value"] for lv, m in grp]; lo = [L4[lv][m]["auroc"][k]["ci95"][0] for lv, m in grp]; hi = [L4[lv][m]["auroc"][k]["ci95"][1] for lv, m in grp]
    xs = np.arange(len(grp)) + (i - 1) * W; ax.bar(xs, v, W, color=col, label=lab, yerr=[np.array(v) - lo, np.array(hi) - v], capsize=3, ecolor="0.3")
    for x, val, h in zip(xs, v, hi): ax.text(x, h + 0.008, f"{val:.2f}", ha="center", va="bottom", fontsize=9)
ax.set_xticks(np.arange(len(grp))); ax.set_xticklabels([f"{lv} level\n{m} (n = {L4[lv]['n_units']}, {L4[lv]['n_progressor_units']} progressor)" for lv, m in grp]); ax.set_ylim(0.4, 1.1); ax.axhline(0.5, color="0.3", lw=0.8, ls="--")
ax.set_ylabel("Fold-stratified AUROC (95% CI)"); ax.legend(frameon=False, loc="upper right", ncol=3); ax.set_title("L4. Disagreement within an endoscopy (units with ≥ 2 samples)", loc="left")
fig.tight_layout(); fig.savefig(f"{OUTD}/r8_l4_auroc.png", bbox_inches="tight"); plt.close(fig)
# ---------------------------------------------------------------- L5 2 x 2
L5 = J("l5.json")["L5"]; ct = L5["cross_tab_progressor_patients_mean_over_repeats"]; fig, ax = plt.subplots(figsize=(7.2, 4.2)); ax.axis("off"); ax.set_xlim(0, 3.05); ax.set_ylim(-0.05, 3)
fmt = lambda v: v if isinstance(v, str) else f"{v:.1f}"
grid = {(0, 0): ("both", ct["both"]), (1, 0): ("copy number only", ct["copy number only"]), (0, 1): ("H&E only", ct["H&E only"]), (1, 1): ("neither", ct["neither"])}
for (ci_, ri), (lab, v) in grid.items():   # column 0 = H&E caught, row 0 = copy number caught
    x0, y0 = 1 + ci_, 1 - ri; ax.add_patch(plt.Rectangle((x0, y0), 1, 1, fill=True, facecolor="#E8EEF7" if lab == "both" else "white", edgecolor="black", lw=1.2))
    ax.text(x0 + 0.5, y0 + 0.58, fmt(v), ha="center", va="center", fontsize=16); ax.text(x0 + 0.5, y0 + 0.25, lab, ha="center", va="center", fontsize=9, color="0.35")
ax.text(1.5, 2.12, "H&E only (P1)\ncaught", ha="center", va="bottom", fontsize=10); ax.text(2.5, 2.12, "H&E only (P1)\nmissed", ha="center", va="bottom", fontsize=10)
ax.text(0.95, 1.5, "Copy number only (P2)\ncaught", ha="right", va="center", fontsize=10); ax.text(0.95, 0.5, "Copy number only (P2)\nmissed", ha="right", va="center", fontsize=10)
ax.text(0.0, 2.95, f"L5. Progressor patients (of {L5['n_progressor_patients']}), mean over repeats; both policies at\n{L5['target_non_progressor_patients_flagged_mean']:.1f} of {L5['n_non_progressor_patients']} non-progressors ever flagged (cells < 5 shown as '<5')", ha="left", va="top", fontsize=10)
fig.tight_layout(); fig.savefig(f"{OUTD}/r8_l5_2x2.png", bbox_inches="tight"); plt.close(fig)
print("figures ->", OUTD)

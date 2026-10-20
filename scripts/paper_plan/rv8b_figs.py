"""Figures for docs/paper_review_round8b_step3.md (local, not committed): paper/figures_r8/r8b_*.png from results/paper_final/review_round8b/*.json.
S1 paired-difference forest; S2 frontier-membership bars; S3 cost-frontier membership across the grid; S4 ACE-B-shaped distributions with the proposed bounds.
R8BDIR / R8BFIG override the input and output folders (smoke preview)."""
import json, os, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
T = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
R = os.environ.get("R8BDIR", T + "/results/paper_final/review_round8b"); FIG = os.environ.get("R8BFIG", T + "/paper/figures_r8"); os.makedirs(FIG, exist_ok=True)
J = lambda f: json.load(open(f"{R}/{f}"))["S" + f[1]]
BLUE, ORANGE, AQUA, INK, INK2, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#d9d8d4"
plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 200, "font.family": "DejaVu Sans"})
def save(fig, name): fig.savefig(f"{FIG}/{name}", bbox_inches="tight", facecolor="white"); plt.close(fig); print("wrote", f"{FIG}/{name}")
# ---------------------------------------------------------------- S1 forest
s = J("s1.json"); MN = [("non_progressor_patients_flagged", "NP patients flagged\n(of 43)"), ("endoscopy_specificity", "Endoscopy\nspecificity"), ("flagged_np_endoscopies_per_np_patient_year", "Flagged NP endoscopies\nper NP patient-year"),
                        ("progressor_patients_caught", "Progressor patients\ncaught (of 32)"), ("samples_sequenced_per_patient", "Samples sequenced\nper patient")]
ROWS = [("a_R49_vs_P1", "i_evaluation_matched", "R49 − P1, matched sensitivity", BLUE, "o"), ("a_R49_vs_P1", "ii_P1_at_90", "R49 − P1 at 90%", BLUE, "s"),
        ("b_P3_90_vs_P1", "i_evaluation_matched", "P3 90% − P1, matched sensitivity", ORANGE, "o"), ("b_P3_90_vs_P1", "ii_P1_at_90", "P3 90% − P1 at 90%", ORANGE, "s")]
fig, ax = plt.subplots(1, 5, figsize=(14, 3.6), sharey=True); yy = np.arange(len(ROWS))[::-1]
for a, (k, lab) in zip(ax, MN):
    for yv, (c, op, rl, col, mk) in zip(yy, ROWS):
        d = s["contrasts"][c][op]["delta_rule_minus_P1"][k]; a.plot(d["ci95"], [yv, yv], color=col, lw=2, solid_capstyle="round"); a.plot(d["delta"], yv, mk, color=col, ms=8, mec="white", mew=1.5)
    a.axvline(0, color=INK2, lw=1, ls=(0, (3, 3))); a.set_title(lab); a.grid(axis="x", color=GRID, lw=0.6); a.tick_params(axis="y", length=0)
ax[0].set_yticks(yy); ax[0].set_yticklabels([r[2] for r in ROWS]); ax[0].set_ylim(-0.6, len(ROWS) - 0.4)
st = s["statement"]; fig.suptitle(f"S1. Paired differences against H&E alone (Δ = rule − P1, 95% CI; circle = P1 at matched sensitivity, square = P1 at 90%)\nBinding rule, (a)(i) upper 95% limit for NP patients flagged {st['upper_95_limit']:+.2f}: \"{st['statement']}\"", fontsize=11, x=0.01, ha="left", y=1.1)
save(fig, "r8b_s1_forest.png")
# ---------------------------------------------------------------- S2 frontier membership
s = J("s2.json"); C = [c for c in s["cells"] if c["share_draws_on_pareto"] >= 0.05 or c["share_draws_on_cost_frontier"] >= 0.05 or c["observed_on_pareto"] or c["observed_on_cost_frontier"]
                       or (c["family"], c["label"]) in (("P6", "t_low 95% / t_high 90%"), ("P1", "90%"))]
C.sort(key=lambda c: (c["share_draws_on_pareto"] + c["share_draws_on_cost_frontier"]))
lab = lambda c: ("R49 (P6 95/90)" if (c["family"], c["label"]) == ("P6", "t_low 95% / t_high 90%") else f"{c['family']} {c['label']}").replace("; t, t_s 90%", "").replace(", t_c 90%", "").replace("t_low ", "").replace(" / t_high ", "/")
fig, ax = plt.subplots(1, 2, figsize=(11, 0.32 * len(C) + 1.4), sharey=True); yy = np.arange(len(C))
for a, key, ok, col, ttl in ((ax[0], "share_draws_on_pareto", "observed_on_pareto", BLUE, "Pareto frontier (caught, NP flagged, sequenced)"), (ax[1], "share_draws_on_cost_frontier", "observed_on_cost_frontier", ORANGE, "Cost frontier (caught, cost per patient)")):
    v = np.array([c[key] for c in C]); a.barh(yy, v, height=0.7, color=col, edgecolor="white", linewidth=1)
    for yv, c, x in zip(yy, C, v):
        a.text(min(x, 1) + 0.015, yv, f"{100 * x:.0f}%" + (" ●" if c[ok] else ""), va="center", fontsize=9, color=INK)
    a.set_xlim(0, 1.18); a.set_xticks([0, 0.25, 0.5, 0.75, 1]); a.set_xticklabels(["0", "25%", "50%", "75%", "100%"]); a.set_title(ttl); a.grid(axis="x", color=GRID, lw=0.6); a.set_axisbelow(True); a.tick_params(axis="y", length=0)
ax[0].set_yticks(yy); ax[0].set_yticklabels([lab(c) for c in C]); ax[0].set_ylim(-0.6, len(C) - 0.4)
for t_, c in zip(ax[0].get_yticklabels(), C):
    if (c["family"], c["label"]) in (("P6", "t_low 95% / t_high 90%"), ("P1", "90%")): t_.set_fontweight("bold")
fig.suptitle(f"S2. Share of {s['draws']:,} bootstrap draws in which each policy cell is on the frontier (● = on it in the observed data)", fontsize=11, x=0.01, ha="left")
fig.tight_layout(); save(fig, "r8b_s2_frontier_share.png")
# ---------------------------------------------------------------- S3 grid
s = J("s3.json"); names = s["rules"]; ab = {"P1 80%": "P1 80%", "P1 90%": "P1 90%", "P1 95%": "P1 95%", "R49": "R49", "P3 90%": "P3 90%", "P8 t_b 90% / t_mid 80%": "P8 90/80", "grade ID/LGD": "Grade"}
G = [g for g in s["grid"] if not (g["unit"] == "np_py" and g["progressor_weight"] != "cohort")]
rl = [f"£{g['seq_price']:.0f} seq · £{g['flag_cost']:,.0f} flag · " + ("per patient, " + ("cohort" if g["progressor_weight"] == "cohort" else "10% prog.") if g["unit"] == "patient" else "per NP patient-year") for g in G]
M_ = np.array([[k in g["cost_frontier"] for k in names] for g in G], float)
fig, ax = plt.subplots(figsize=(11.5, 0.33 * len(G) + 1.6)); ax.imshow(M_, cmap=matplotlib.colors.ListedColormap(["#f1f0ec", BLUE]), aspect="auto", vmin=0, vmax=1)
ax.set_xticks(range(len(names))); ax.set_xticklabels([ab[k] for k in names]); ax.xaxis.tick_top(); ax.set_yticks(range(len(G))); ax.set_yticklabels(rl); ax.tick_params(length=0)
for i in range(len(G) + 1): ax.axhline(i - 0.5, color="white", lw=2)
for j in range(len(names) + 1): ax.axvline(j - 0.5, color="white", lw=2)
for i, g in enumerate(G):
    for j, k in enumerate(names): ax.text(j, i, f"£{g['cost'][k]['value']:,.0f}", ha="center", va="center", fontsize=9, color="white" if M_[i, j] else INK2)
    d = g["delta_cost_R49_minus_P1_90"]; ax.text(len(names) - 0.4, i, f"  {d['delta']:+,.0f} [{d['ci95'][0]:+,.0f}, {d['ci95'][1]:+,.0f}]", ha="left", va="center", fontsize=9, color=INK)
ax.text(len(names) - 0.4, -0.95, "  Δ cost R49 − P1 90% [95% CI]", ha="left", va="bottom", fontsize=9, color=INK)
for sp in ax.spines.values(): sp.set_visible(False)
ax.set_title("S3. Rules on the cost frontier (blue) across the cost grid; cell = cost per unit", loc="left", pad=28, fontsize=11)
save(fig, "r8b_s3_cost_grid.png")
# ---------------------------------------------------------------- S4 distributions
s = J("s4.json"); pb = s["proposed_pass_bounds"]; KB = pb["secondary_sensitivity"]["proposed_bound_progressors_detected_at_least"]
P4 = [("auroc_patient_max_L_IMG", "H&E patient-level AUROC", [(0.70, "0.70", ORANGE)]), ("auroc_hanley_mcneil_lower_95", "AUROC lower 95% bound (Hanley–McNeil)", [(0.70, "pass > 0.70", ORANGE)]),
      ("r49_progressor_patients_detected_of_15", "R49 progressors detected (of 15)", [(KB, f"proposed ≥ {KB}\n(= panel's ≥ 12)" if KB == 12 else f"proposed ≥ {KB}", ORANGE)] + ([] if KB == 12 else [(12, "≥ 12", AQUA)])),
      ("r49_endoscopy_specificity", "R49 endoscopy specificity", []), ("delta_specificity_r49_minus_p1_matched", "Δ specificity, R49 − P1 (matched)", [(0.0, "0", INK2)]),
      ("r49_share_samples_sequenced", "R49 share of samples sequenced", [(pb["secondary_share_sequenced"]["proposed_upper_bound"], f"proposed ≤ {pb['secondary_share_sequenced']['proposed_upper_bound']:.2f}", ORANGE)])]
fig, ax = plt.subplots(2, 3, figsize=(13, 6.6))
for a, (k, ttl, lines) in zip(ax.ravel(), P4):
    o = s[k]; e = np.array(o["hist_edges"]); c = np.array(o["hist_counts"]); nz = np.where(c > 0)[0]; lo, hi = e[max(nz[0] - 1, 0)], e[min(nz[-1] + 2, len(e) - 1)]
    a.bar(e[:-1], c / c.sum(), width=np.diff(e), align="edge", color=BLUE, edgecolor="white", linewidth=0.8)
    a.axvspan(o["p2_5"], o["p97_5"], color=BLUE, alpha=0.08, lw=0); a.axvline(o["median"], color=INK, lw=1.5)
    for xv, tx, col in lines:
        a.axvline(xv, color=col, lw=2, ls=(0, (4, 2))); lo = min(lo, xv - 0.02 * (hi - lo)); hi = max(hi, xv + 0.02 * (hi - lo))
    a.set_xlim(lo, hi); top = a.get_ylim()[1]; a.set_ylim(0, top * 1.32)
    for i_, (xv, tx, col) in enumerate(lines):   # label on the side away from the median line
        lf = xv <= o["median"]; a.text(xv, top * (1.22 - 0.11 * i_), f"{tx} " if lf else f" {tx}", color=INK, fontsize=9, ha="right" if lf else "left", va="center")
    dg = 1 if "detected" in k else 3; a.set_title(f"{ttl}\nmedian {o['median']:.{dg}f}; 95% range {o['p2_5']:.{dg}f}–{o['p97_5']:.{dg}f}", fontsize=10)
    a.set_yticks([]); a.spines["left"].set_visible(False); a.grid(axis="x", color=GRID, lw=0.6); a.set_axisbelow(True)
fig.suptitle(f"S4. ACE-B-shaped discovery subsamples ({s['design']['draws']:,} draws; 15 progressor / 67 non-progressor patients, 1 endoscopy each, ≤ 2 samples)\nblack line = median, shading = 2.5–97.5% range, dashed = proposed or pre-specified bound",
             fontsize=11, x=0.01, ha="left"); fig.tight_layout(); save(fig, "r8b_s4_distributions.png")

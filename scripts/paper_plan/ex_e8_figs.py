"""Exploration E8 figures (local, from aggregate JSON only) -> paper/figures_explore_E8/*.png (200 dpi)."""
import json, os, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
H = os.path.dirname(os.path.abspath(__file__)); D = H + "/../../results/paper_final/explore_E8"; O = H + "/../../paper/figures_explore_E8"; os.makedirs(O, exist_ok=True)
J = lambda n: json.load(open(f"{D}/{n}.json"))
BLUE, ORANGE, AQUA, GRAY, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#a3a29c", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})
def cell(task, label_start):
    r = J(task); cs = r.get("cells") or r[task]["cells"]
    for c in cs:
        if c["label"].startswith(label_start): return c
    raise KeyError(label_start)
# ---------------------------------------------------------------- 1. arm specificity: random 10-arm sets vs the OAC check and named controls
r = J("rand")["rand"]["random_arm_sets"]["non-OAC arms only"]; A = np.array(r["draws_dNP_dcaught"], float)
rng = np.random.RandomState(0); jit = lambda k: rng.uniform(-0.04, 0.04, k)
fig, ax = plt.subplots(figsize=(7.2, 4.6))
ax.scatter(-A[:, 0] + jit(len(A)), A[:, 1] + jit(len(A)), s=9, color=GRAY, alpha=0.45, lw=0, label="1,000 random 10-arm sets (non-OAC arms)")
named = [("OAC check (final rule)", "base", None, BLUE), ("17p loss only", "arms", "A12 + check on 17p loss only", ORANGE), ("GC-rich arms, any change", "arms", "A12 + check on GC-rich non-OAC arms 19p 19q 22q 16p, any change", AQUA),
         ("FHIT + WWOX focal loss", "arms", "A12 + check on FHIT + WWOX focal loss k2", "#4a3aa7"), ("Fraction genome altered", "flat", "A12 + check on global measure: fraction genome altered", "#e34948")]
for nm, t, lab, col in named:
    c = J("base")["final_full"] if lab is None else cell(t, lab); x = -c["dNP_vs_A12"]["delta"]; y = c["dP_vs_A12"]["delta"]
    ax.scatter([x], [y], s=70, color=col, edgecolor="white", lw=1.5, zorder=3); ax.annotate(nm, (x, y), xytext=(6, 6 if nm != "17p loss only" else -14), textcoords="offset points", fontsize=9, color=INK)
ax.axhline(0, color=MUTED, lw=0.8, ls=":"); ax.axvline(0, color=MUTED, lw=0.8, ls=":")
ax.set_xlabel("Non-progressors cleared vs A12 (patients)"); ax.set_ylabel("Progressors caught vs A12 (patients)")
ax.set_title("The OAC arms beat every random arm set (0 of 1,000 match it)", fontsize=11, loc="left", color=INK); ax.legend(loc="lower right", frameon=False, fontsize=9)
fig.tight_layout(); fig.savefig(O + "/e8_arm_specificity.png", dpi=200); plt.close(fig)
# ---------------------------------------------------------------- 2. timing: quiet share of progressor endoscopies by years before the event
t = J("time")["time"]["progressor_quiet_by_time_to_event"]["all progressor endoscopies"]["bands"]; bands = ["<1 y before event", "1-3 y", ">=3 y"]
y = [t[b]["quiet_share"] for b in bands]; lo = [t[b]["ci95"][0] for b in bands]; hi = [t[b]["ci95"][1] for b in bands]; nE = [t[b]["endoscopies_per_repeat"] for b in bands]
npq = J("time")["time"]["np_quiet_share_all_endoscopies"]
fig, ax = plt.subplots(figsize=(6.0, 4.0)); xs = np.arange(3)
ax.bar(xs, y, width=0.55, color=BLUE, zorder=2); ax.errorbar(xs, y, yerr=[np.array(y) - lo, np.array(hi) - y], fmt="none", ecolor=INK, lw=1.2, capsize=4, zorder=3)
ax.axhline(npq, color=ORANGE, lw=2, ls="--"); ax.text(2.35, npq + 0.01, f"non-progressor endoscopies: {npq:.0%}", ha="right", va="bottom", fontsize=9, color=INK)
ax.set_xticks(xs); ax.set_xticklabels([f"{b.replace('>=', '≥')}\n({n:.0f} endoscopies)" for b, n in zip(["<1 y", "1–3 y", ">=3 y"], nE)])
ax.set_ylabel("Share quiet on the 10 OAC arms"); ax.set_xlabel("Progressor endoscopies, years before the event"); ax.set_ylim(0, 0.45); ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
ax.set_title("Quiet progressor endoscopies are early, not near the event", fontsize=10.5, loc="left", color=INK)
fig.tight_layout(); fig.savefig(O + "/e8_quiet_timing.png", dpi=200); plt.close(fig)
# ---------------------------------------------------------------- 3. technical confounding: AUROC of each QC metric for quiet vs not (A12-flagged), by class
q = J("qc")["qc"]["quiet_vs_not_among_A12_flagged"]; mets = [("log10_reads", "Read depth"), ("log_mapd_50kb", "50-kb noise (MAPD)"), ("log_segment_residual_mad", "Segment-residual noise"),
                                                          ("qdnaseq_expected_variance", "QDNAseq expected variance"), ("arm_level_robust_sd", "Arm-level noise"), ("barretts_cellularity_pct", "Barrett's cellularity (%)*")]
fig, ax = plt.subplots(figsize=(7.0, 4.2)); ys = np.arange(len(mets))[::-1]
for off, cl, col, nm in ((0.15, "non_progressor", ORANGE, "Non-progressor endoscopies"), (-0.15, "progressor", BLUE, "Progressor endoscopies")):
    for yy, (k, _) in zip(ys, mets):
        a = q[k][cl]["auroc_metric_higher_in_quiet"]; ax.errorbar([a["auroc"]], [yy + off], xerr=[[a["auroc"] - a["ci95"][0]], [a["ci95"][1] - a["auroc"]]], fmt="o", color=col, ms=6, lw=1.4, capsize=3, label=nm if k == "log10_reads" else None)
ax.axvline(0.5, color=MUTED, lw=0.8, ls=":"); ax.set_yticks(ys); ax.set_yticklabels([m for _, m in mets]); ax.set_xlim(0, 1)
ax.set_xlabel("AUROC: metric higher in quiet than non-quiet flagged endoscopies"); ax.legend(frameon=False, fontsize=9, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
ax.set_title("Quiet calls come from cleaner, deeper samples", fontsize=11, loc="left", color=INK, pad=24)
fig.text(0.01, 0.01, "* cellularity recorded for few samples. Left of 0.5 = lower in quiet endoscopies.", fontsize=8.5, color=MUTED)
fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(O + "/e8_qc_confounding.png", dpi=200); plt.close(fig)
# ---------------------------------------------------------------- 4. cost: NP flagged vs samples per patient (needs soft2)
try:
    pts = [("A12", J("base")["A12_full"]), ("Final rule", J("base")["final_full"]), ("Sequential check", J("seq")["seq"]["sequential"])]
    for lab, nm in (("A12 + band check: cheapest band matching the hard veto on training, sequence in band, sequential", "Nested band check + sequential"),
                    ("A12 + soft veto (noloss, bar cn90), sequence only where it can change the call", "Soft veto (nested δ), band sequencing"),
                    ("A12 + band check, fixed delta 5.0", "Band δ=5 + sequential (post hoc)"), ("A12 + band check, fixed delta 3.0", "Band δ=3 + sequential (post hoc)")):
        try: pts.append((nm, cell("soft", lab)))
        except KeyError: pass
    fig, ax = plt.subplots(figsize=(7.2, 4.4)); cols = [GRAY, BLUE, AQUA, ORANGE, "#4a3aa7", "#e34948", "#e87ba4"]
    for (nm, c), col in zip(pts, cols):
        x = c["samples_sequenced_per_patient"]; yv = c["non_progressor_patients_ever_flagged"]; ax.scatter([x], [yv], s=70, color=col, edgecolor="white", lw=1.5, zorder=3, label=nm)
    ax.set_xlabel("Samples sequenced per patient"); ax.set_ylabel("Non-progressors ever flagged (of 43)"); ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax.set_title("Every rule here catches 30.5 of 32 progressors; staging the check saves samples", fontsize=11, loc="left", color=INK); fig.tight_layout(); fig.savefig(O + "/e8_sequencing_cost.png", dpi=200); plt.close(fig)
except FileNotFoundError: pass
print("figs done")

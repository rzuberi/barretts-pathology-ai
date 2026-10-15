"""Generate every thesis figure from results/*.json (no patient data touched).
Output: thesis_tex/figures/*.pdf. Deterministic; re-run after any result change.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

T = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(T, "results"); OUT = os.path.join(T, "thesis_tex", "figures")
os.makedirs(OUT, exist_ok=True)
J = lambda f: json.load(open(os.path.join(R, f)))
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.bbox": "tight"})
C_NULL, C_POS, C_NEG, C_GREY = "#4c72b0", "#2a9d5c", "#c44e52", "#8c8c8c"
def save(fig, name): fig.savefig(os.path.join(OUT, name)); plt.close(fig); print("wrote", name)

# ---------------- Fig 1: fusion delta forest ----------------
rows = []
sa = J("swg_selection_adjusted.json"); occ = J("occams_v3.json"); ep = J("erin_prog_ablation.json")
to = J("tcga_abmil.json"); tp = J("tcga_pool_fusion.json"); po = J("porpoise_baselines.json")
rows.append(("SWG late-mean vs histology (naive, AUC)", sa["auc"]["observed_delta_vs_hist"]["late_mean"], None, None))
h = sa["selection_honest_delta_auc"]
rows.append(("SWG selected arm, out-of-bag (AUC)", h["mean_oob_delta_of_selected_arm"], *h["ci"]))
def d(x): return x.get("delta_mean", x.get("delta_vs_ref", {}).get("mean")), *(x.get("delta_ci") or x["delta_vs_ref"]["ci"])
rows.append(("OCCAMS hist+genomics vs hist (C)", *d(occ["late_hist_gen"])))
rows.append(("OCCAMS hist+clinical vs hist (C)", *d(occ["late_hist_clin"])))
rows.append(("ERIN progression late fusion (AUC)", *d(ep["late_fusion"])))
rows.append(("ERIN progression, strictly pre-index clinical (AUC)", *d(ep["late_fusion_strict_pre"])))
rows.append(("TCGA-OAC hist+genomics (C)", *d(to["late_hist_gen"])))
rows.append(("TCGA pool hist+genomics (C)", *d(tp["late_hist_gen"])))
rows.append(("PORPOISE MMF vs AMIL, TCGA (C)", po["mmf_minus_amil"]["delta_c"], *po["mmf_minus_amil"]["ci"]))
es = J("encoder_sweep_surv.json")
for enc in es["_meta"]["encoders_attempted"]:
    for coh in ("occams", "tcga_oac", "tcga_pool"):
        x = es.get(enc, {}).get(coh)
        if x and "late_fusion" in x:
            rows.append((f"{coh.upper().replace('_', '-')} late fusion, {enc} (C)", x["late_fusion"]["delta_mean"], *x["late_fusion"]["delta_ci"]))
fig, ax = plt.subplots(figsize=(7.2, 0.28 * len(rows) + 1.2))
y = np.arange(len(rows))[::-1]
ax.axvspan(-0.075, 0.075, color=C_GREY, alpha=0.12, lw=0, label="below minimum detectable delta (0.075)")
ax.axvline(0, color="k", lw=0.8)
for yi, (lab, m, lo, hi) in zip(y, rows):
    col = C_POS if (lo is not None and lo > 0) else (C_NEG if (hi is not None and hi < 0) else C_NULL)
    if lo is not None: ax.plot([lo, hi], [yi, yi], color=col, lw=1.6)
    ax.plot(m, yi, "o", color=col, ms=5)
ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=7.5)
ax.set_xlabel("fusion minus best unimodal arm (95% paired bootstrap CI)")
ax.legend(loc="lower right", fontsize=7, frameon=False)
save(fig, "fig_fusion_forest.pdf")

# ---------------- Fig 2: power map ----------------
pm = J("power_map_v2.json"); deltas = [x for x in pm["_meta"]["deltas"] if x > 0]
cohorts = [c for c in pm if c != "_meta"]
fig, axes = plt.subplots(1, 3, figsize=(8.5, 2.9), sharey=True)
for ax, rho in zip(axes, (0.6, 0.8, 0.9)):
    M = np.full((len(cohorts), len(deltas)), np.nan)
    for i, c in enumerate(cohorts):
        for j, dd in enumerate(deltas):
            cell = pm[c].get(f"rho{rho}_d{dd}", {})
            if "power" in cell: M[i, j] = cell["power"]
    im = ax.imshow(M, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    for i in range(len(cohorts)):
        for j in range(len(deltas)):
            v = M[i, j]
            ax.text(j, i, "n/a" if np.isnan(v) else f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="w" if (np.isnan(v) or v < 0.6) else "k")
    ax.set_xticks(range(len(deltas))); ax.set_xticklabels(deltas, fontsize=7)
    ax.set_title(f"arm correlation rho = {rho}", fontsize=9); ax.set_xlabel("true fusion delta")
axes[0].set_yticks(range(len(cohorts))); axes[0].set_yticklabels(cohorts, fontsize=8)
fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02, label="power (paired-bootstrap CI excludes 0)")
save(fig, "fig_power_map.pdf")

# ---------------- Fig 3: selection-adjusted SWG ----------------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(8, 3), gridspec_kw={"width_ratios": [1.5, 1]})
od = sa["auc"]["observed_delta_vs_hist"]; fams = [f for f in od if f != "image_only"]
vals = [od[f] for f in fams]
cols = [C_POS if f == "late_mean" else C_NULL for f in fams]
a1.barh(fams, vals, color=cols)
q95 = sa["auc"]["fusion_arms"]["null_t_q95"]
a1.axvline(q95, color=C_NEG, ls="--", lw=1.2, label=f"95th pct of best-arm advantage under no signal ({q95:.3f})")
a1.axvline(0, color="k", lw=0.8); a1.set_xlabel("AUC minus histology-only (observed)")
a1.legend(fontsize=7, frameon=False, loc="lower right"); a1.tick_params(axis="y", labelsize=8)
a2.errorbar([0], [od["late_mean"]], fmt="o", color=C_POS, label="naive (full sample)")
a2.errorbar([1], [h["mean_oob_delta_of_selected_arm"]], yerr=[[h["mean_oob_delta_of_selected_arm"] - h["ci"][0]], [h["ci"][1] - h["mean_oob_delta_of_selected_arm"]]],
            fmt="o", color=C_NULL, capsize=4, label="select in-bag, evaluate out-of-bag")
a2.axhline(0, color="k", lw=0.8); a2.set_xticks([0, 1]); a2.set_xticklabels(["naive", "selection-honest"], fontsize=8)
a2.set_ylabel("late-mean minus histology (AUC)"); a2.set_xlim(-0.6, 1.6)
a2.set_title(f"adjusted p = {sa['auc']['fusion_arms']['p_selection_adjusted']:.2f} (naive {sa['auc']['fusion_arms']['p_selected_arm_unadjusted']:.4f})", fontsize=8)
save(fig, "fig_selection_adjusted.pdf")

# ---------------- Fig 4: prevalence-matched visibility ----------------
vm = J("visibility_matched.json")
fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.2), sharey=True)
palette = {"tcga_oac": "#d62728", "occams_oac": "#ff7f0e", "oac_combined": "#9467bd", "stad_gej": "#1f77b4", "mixed_all": "#2ca02c"}
for ax, tgt in zip(axes, ("tp53", "wgd")):
    for s, cell in vm[tgt].items():
        ns = sorted(int(n) for n in cell["curve"])
        if not ns: continue
        ax.plot(ns, [cell["curve"][str(n)]["auc_mean"] for n in ns], "-o", ms=4, color=palette[s], label=s)
        ax.plot(ns, [cell["curve"][str(n)]["null_q95"] for n in ns], ":", color=palette[s], lw=1)
        io = [cell["curve"][str(n)]["cohort_identity_only_auc"] for n in ns]
        if any(v is not None for v in io):
            ax.plot(ns, [v if v is not None else np.nan for v in io], "--", color=palette[s], lw=1)
    ax.axhline(0.5, color="k", lw=0.8); ax.set_title(f"{tgt.upper()} from H&E, balanced draws", fontsize=9)
    ax.set_xlabel("n (n/2 positives + n/2 negatives)")
axes[0].set_ylabel("AUC (mean of 20 draws)")
axes[1].plot([], [], "k-o", ms=4, label="probe"); axes[1].plot([], [], "k:", label="permutation 95th pct")
axes[1].plot([], [], "k--", label="cohort-identity-only")
axes[1].legend(fontsize=7, frameon=False, ncol=2)
save(fig, "fig_visibility_matched.pdf")

# ---------------- Fig 5: transfer matrix ----------------
tm = J("transfer_matrix_excl.json"); cells = [k for k in tm if k != "_meta"]
fig, ax = plt.subplots(figsize=(6.5, 2.8))
x = np.arange(len(cells)); w = 0.38
ax.bar(x - w / 2, [tm[c]["within_train_cohort_cv"] for c in cells], w, color=C_NULL, label="within training cohort (CV)")
ax.bar(x + w / 2, [tm[c]["transfer_auc"] for c in cells], w, color=C_NEG, label="transferred to other cohort")
ax.axhline(0.5, color="k", lw=0.8); ax.set_xticks(x); ax.set_xticklabels([c.replace("_", "\n") for c in cells], fontsize=7)
ax.set_ylabel("AUC"); ax.set_ylim(0.4, 1); ax.legend(fontsize=7, frameon=False)
save(fig, "fig_transfer_matrix.pdf")

# ---------------- Fig 6: jury ----------------
js = J("erin_jury_labels_summary.json"); lofo = J("lofo_jury.json"); uc = J("unsure_characterization.json")
fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(9, 2.8), gridspec_kw={"width_ratios": [1.2, 1.2, 0.8]})
dist = js["label_dist_train_eligible"]; order = ["NDBE", "IND", "LGD", "HGD", "CANCER"]
a1.bar(order, [dist.get(k, 0) for k in order], color=C_NULL); a1.set_ylabel("train-eligible reports"); a1.set_title("jury labels (n=6,867)", fontsize=9)
for k, v in zip(order, [dist.get(k, 0) for k in order]): a1.text(k, v, str(v), ha="center", va="bottom", fontsize=7)
drops = [k for k in lofo if k.startswith("drop_")]
a2.barh([k.replace("drop_", "") for k in drops], [100 * lofo[k]["label_flip_rate"] for k in drops], color=C_NULL)
a2.set_xlabel("% labels flipped when family removed"); a2.set_title("leave-one-family-out", fontsize=9); a2.tick_params(axis="y", labelsize=8)
a3.bar(["train-eligible", "unsure"], [uc["train_eligible"]["hedging_mean"], uc["unsure"]["hedging_mean"]], color=[C_NULL, C_NEG])
a3.set_ylabel("hedging-language rate"); a3.set_title("why reports are unsure", fontsize=9)
save(fig, "fig_jury.pdf")

# ---------------- Fig 7: pan-cancer ----------------
ph = J("pancancer_hardening.json")["pancancer"]; studies = [s for s in ph if s.endswith("_tcga")]
fig, ax = plt.subplots(figsize=(6, 2.8)); x = np.arange(len(studies)); w = 0.38
ag = [ph[s]["two_tier"]["agreement"] for s in studies]; lo = [ph[s]["two_tier"]["wilson95"][0] for s in studies]; hi = [ph[s]["two_tier"]["wilson95"][1] for s in studies]
ax.bar(x - w / 2, ag, w, color=C_POS, yerr=[np.array(ag) - lo, np.array(hi) - ag], capsize=3, label="jury vs registry (two-tier)")
ax.bar(x + w / 2, [ph[s]["two_tier"]["majority_class_baseline_agreement"] for s in studies], w, color=C_GREY, label="always-majority-class baseline")
ax.set_xticks(x); ax.set_xticklabels([f"{s.split('_')[0].upper()}\n(n={ph[s]['n_jury_graded']})" for s in studies], fontsize=8)
ax.set_ylim(0.4, 1.02); ax.set_ylabel("agreement with registry grade"); ax.legend(fontsize=7, frameon=False, loc="lower left")
save(fig, "fig_pancancer.pdf")

# ---------------- Fig 8: slide vs case-max ----------------
sl = J("slide_labels_v2.json"); cc = J("clustered_cis.json")
fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.5, 3), gridspec_kw={"width_ratios": [1, 1.3]})
g = sl["grade_dist"]; order6 = ["NORMAL_OTHER", "NDBE", "IND", "LGD", "HGD", "CANCER"]
a1.bar(order6, [g.get(k, 0) for k in order6], color=C_NULL); a1.tick_params(axis="x", rotation=45, labelsize=7)
a1.set_ylabel("slides"); a1.set_title(f"section-resolved slide grades (n={sl['slides_labelled']}); {100*sl['slides_where_section_differs_from_case_max']:.0f}% differ from case-max", fontsize=8)
items = [("2.38 binary AUC", "binary_2_38_delta_slide_minus_case_auc"), ("2.38b macro-AUC", "sixclass_2_38b_delta_macro_auc"), ("2.38b weighted kappa", "sixclass_2_38b_delta_qwk")]
for i, (lab, key) in enumerate(items):
    for j, (kind, col) in enumerate((("iid_slide_bootstrap", C_NULL), ("patient_clustered_bootstrap", C_NEG))):
        v = cc[key][kind]; yy = i + (0.15 if j else -0.15)
        a2.plot(v["ci"], [yy, yy], color=col, lw=1.6); a2.plot(v["mean"], yy, "o", color=col, ms=4)
a2.axvline(0, color="k", lw=0.8); a2.set_yticks(range(len(items))); a2.set_yticklabels([i[0] for i in items], fontsize=8)
a2.set_xlabel("trained on slide labels minus trained on case-max (95% CI)")
a2.plot([], [], color=C_NULL, label="slide bootstrap"); a2.plot([], [], color=C_NEG, label="patient-clustered bootstrap"); a2.legend(fontsize=7, frameon=False)
save(fig, "fig_slide_vs_casemax.pdf")

# ---------------- Fig 9: VLM ----------------
vp = J("vlm_pretrain.json"); vs0 = J("vlm_swg.json"); vs1 = J("vlm_swg_excl.json")
fig, (a1, a2) = plt.subplots(1, 2, figsize=(8, 2.8))
labs = ["ERIN test", "TCGA", "SWG (all pairs)", "SWG (overlap removed)"]
r1 = [vp["erin_test"]["recall_at_1"], vp["tcga_transfer"]["recall_at_1"], vs0["retrieval"]["recall_at_1"], vs1["retrieval"]["recall_at_1"]]
ch = [vp["erin_test"]["chance_at_1"], vp["tcga_transfer"]["chance_at_1"], vs0["retrieval"]["chance_at_1"], vs1["retrieval"]["chance_at_1"]]
x = np.arange(4); w = 0.38
a1.bar(x - w / 2, r1, w, color=C_NULL, label="recall@1"); a1.bar(x + w / 2, ch, w, color=C_GREY, label="chance")
a1.set_xticks(x); a1.set_xticklabels(labs, fontsize=7, rotation=20); a1.set_ylabel("slide-to-report retrieval"); a1.legend(fontsize=7, frameon=False)
aucs = [vp["erin_test"]["zeroshot_grade_auc"], vp["tcga_transfer"]["zeroshot_site_auc"], vs0["zeroshot_grade_vs_pathologist"]["auc_ndbe_vs_lgdplus"], vs1["zeroshot_grade_vs_pathologist"]["auc_ndbe_vs_lgdplus"]]
a2.bar(labs, aucs, color=[C_POS, C_NULL, C_NULL, C_NEG]); a2.axhline(0.5, color="k", lw=0.8)
a2.set_xticklabels(labs, fontsize=7, rotation=20); a2.set_ylabel("zero-shot AUC"); a2.set_ylim(0.4, 1)
a2.set_title("grade (ERIN, SWG) / site (TCGA)", fontsize=8)
save(fig, "fig_vlm.pdf")

# ---------------- Fig 10: trajectory baselines ----------------
tb = J("swg_trajectory_baselines.json")
fig, ax = plt.subplots(figsize=(5, 2.6))
arms = ["persist", "persist_grade", "hist_only", "hist_plus"]
ax.bar(arms, [tb["cx_next"]["rho"][a] for a in arms], color=[C_GREY, C_GREY, C_NULL, C_NULL])
ax.set_ylabel("Spearman rho with next-biopsy CNV complexity"); ax.tick_params(axis="x", labelsize=8)
inc = tb["cx_next"]["increment_hist_plus_minus_persist_grade"]
ax.set_title(f"adding histology to persistence: {inc['observed']:+.3f} [{inc['patient_clustered_ci'][0]:.3f}, {inc['patient_clustered_ci'][1]:.3f}]", fontsize=8)
save(fig, "fig_trajectory_baselines.pdf")

# ---------------- Fig 11: ERIN encoders ----------------
ee = J("erin_encoder_sweep.json"); encs = list(ee)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(8, 2.8))
a1.bar(encs, [ee[e]["hist_abmil"]["auc"] for e in encs], color=C_NULL, yerr=[[ee[e]["hist_abmil"]["auc"] - ee[e]["hist_abmil"]["auc_ci"][0] for e in encs], [ee[e]["hist_abmil"]["auc_ci"][1] - ee[e]["hist_abmil"]["auc"] for e in encs]], capsize=3)
a1.set_ylim(0.85, 0.96); a1.set_ylabel("histology-only AUC (NDBE vs LGD+)"); a1.tick_params(axis="x", labelsize=8)
for i, e in enumerate(encs):
    for j, (arm, col) in enumerate((("late_fusion", C_POS), ("early_fusion", C_NEG))):
        v = ee[e][arm]["delta_vs_ref"]; yy = i + (0.15 if j else -0.15)
        a2.plot(v["ci"], [yy, yy], color=col, lw=1.6); a2.plot(v["mean"], yy, "o", color=col, ms=4)
a2.axvline(0, color="k", lw=0.8); a2.set_yticks(range(len(encs))); a2.set_yticklabels(encs, fontsize=8)
a2.set_xlabel("fusion minus histology (AUC, 95% CI)"); a2.plot([], [], color=C_POS, label="late fusion"); a2.plot([], [], color=C_NEG, label="early fusion"); a2.legend(fontsize=7, frameon=False)
save(fig, "fig_erin_encoders.pdf")

# ---------------- Fig 12: label-source cross-evaluation ----------------
xe = J("ch4_labelsource_xeval.json")["cross_eval"]; src = ["jury", "pathladder", "feas_grader"]
M = np.array([[xe[f"train_{a}_eval_{b}"]["auc"] for b in src] for a in src])
fig, ax = plt.subplots(figsize=(4, 3.2))
im = ax.imshow(M, cmap="viridis", vmin=0.55, vmax=0.95)
for i in range(3):
    for j in range(3): ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", color="w" if M[i, j] < 0.8 else "k", fontsize=8)
ax.set_xticks(range(3)); ax.set_xticklabels(src, fontsize=8); ax.set_yticks(range(3)); ax.set_yticklabels(src, fontsize=8)
ax.set_xlabel("evaluation labels"); ax.set_ylabel("training labels"); fig.colorbar(im, fraction=0.046, label="AUC")
save(fig, "fig_labelsource_xeval.pdf")

# ---------------- Fig 13: natural history transitions ----------------
bh = J("barretts_history.json")["natural_history"]; tr = bh["transitions"]; st = ["NDBE", "IND", "LGD", "HGD", "CANCER"]
M = np.array([[tr.get(f"{a}->{b}", 0) for b in st] for a in st], float); P = M / M.sum(1, keepdims=True)
fig, ax = plt.subplots(figsize=(4.4, 3.6))
im = ax.imshow(P, cmap="Blues", vmin=0, vmax=1)
for i in range(5):
    for j in range(5): ax.text(j, i, f"{P[i, j]:.2f}\n({int(M[i, j])})", ha="center", va="center", fontsize=6.5, color="k" if P[i, j] < 0.6 else "w")
ax.set_xticks(range(5)); ax.set_xticklabels(st, fontsize=8); ax.set_yticks(range(5)); ax.set_yticklabels(st, fontsize=8)
ax.set_xlabel("grade at next report"); ax.set_ylabel("grade at this report")
ax.set_title(f"report-to-report transitions, {bh['n_patients_with_sequences']:,} patients", fontsize=8)
save(fig, "fig_natural_history.pdf")

# ---------------- Fig 14: MDT + adjudication ----------------
md = J("mdt_erin.json")
fig, ax = plt.subplots(figsize=(4.2, 2.6))
ax.bar(["independent\nmajority", "MDT chair\nafter debate"], [md["adjudicated"]["r1_independent_majority_accuracy"], md["adjudicated"]["mdt_chair_cancer_accuracy"]], color=[C_NULL, C_POS])
ax.set_ylim(0.9, 1.0); ax.set_ylabel(f"accuracy vs adjudication (n={md['adjudicated']['n']})")
ax.set_title(f"{md['deliberation']['converged_unanimous_by_r2']}/{md['deliberation']['cases_with_r1_disagreement']} disagreements converged; 0 conformity losses", fontsize=8)
save(fig, "fig_mdt.pdf")
print("all figures written to", OUT)

"""Figure 14_F_budget_curve (docs/paper_triage_robustness.md @ 32653f1) from results/paper_final/triage_robustness/budget_their_pre.json:
sensitivity and specificity (95% patient-bootstrap bands) against target share sequenced; (A) and (B) as horizontal reference lines."""
import json, os, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
J = json.load(open("results/paper_final/triage_robustness/budget_their_pre.json")); OUT = os.path.expanduser("~/Downloads/be_paper_figs/v3"); G = J["grid"]
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
x = [100 * g["target_share"] for g in G]; fig, ax = plt.subplots(figsize=(7.2, 4.6))
for key, ci, col, lab in (("sensitivity", "sens_ci95", "#d62728", "sensitivity"), ("specificity", "spec_ci95", "#1f77b4", "specificity")):
    ax.fill_between(x, [g[ci][0] for g in G], [g[ci][1] for g in G], color=col, alpha=0.15, lw=0); ax.plot(x, [g[key] for g in G], "-o", color=col, ms=3, lw=1.8, label=f"triage {lab}")
    for ref, ls in (("A", "--"), ("B", ":")):
        ax.axhline(J[ref][key], color=col, ls=ls, lw=1.1, alpha=0.9, label=f"({ref}) sequence everyone, {'L-CNV' if ref == 'A' else 'L-LATE'}: {lab}")
for ref, yy in (("A", 0.96), ("B", 0.91)):
    s = J[f"smallest_s_vs_{ref}"]["value"]
    if s == s and s is not None: ax.axvline(100 * s, color="#555555", lw=0.8, ls="-."); ax.text(100 * s + 1, yy, f"matches ({ref}) from {100 * s:.0f}%", fontsize=7.5, color="#333333")
ax.set_xlim(0, 100); ax.set_ylim(0, 1.0); ax.set_xlabel("target share of samples sequenced (%)"); ax.set_ylabel("per-sample sensitivity / specificity")
ax.set_title("Sequencing budget: H&E-first triage, their CNV matrix, all pre-event samples (571 samples, 75 patients)\n0% = H&E only at the Youden threshold; 100% = sequence everyone with L-LATE (= B)", fontsize=9, loc="left")
ax.legend(fontsize=7, frameon=False, loc="lower left", ncol=1); fig.tight_layout()
for ext in ("pdf", "png"): fig.savefig(f"{OUT}/14_F_budget_curve.{ext}", dpi=200)
print("TB FIG DONE")

"""Extended Data Figure 7, Kaplan-Meier by Killcoyne risk class (docs/paper_figures_v6.md @ d46b9a5; risk_strata), from results/paper_final/risk_strata/
km.json (sample level, pre-event biopsies, 95% patient-bootstrap bands) and sep_pre.json (Cox hazard ratios, patient-clustered robust variance).
One panel per model: published model (P), L-CNV and L-LATE; classes low (Pr <= 0.3), moderate (0.3-0.5) and high (Pr >= 0.5). Runs locally."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fig6_style import *
R = J("risk_strata/km.json")["sample"]; S = J("risk_strata/sep_pre.json")["models"]; NAMES = {"P": MODEL_NAME["P"], "C": "L-CNV, shipped matrix", "L": "L-LATE"}; COL = {"low": BAND["low"], "moderate": BAND["mid"], "high": BAND["high"]}; AT = [0, 2, 4, 6, 8]
fig = plt.figure(figsize=(183 * MM, 80 * MM)); src = {}
for j, m in enumerate(("P", "C", "L")):
    ax = fig.add_axes([0.09 + j * 0.315, 0.40, 0.235, 0.52]); tab = fig.add_axes([0.09 + j * 0.315, 0.04, 0.235, 0.15]); tab.axis("off"); D = R[m]; src[m] = {}
    for i, c in enumerate(("low", "moderate", "high")):
        d = D[c]; ax.step(d["grid"], d["km"], where="post", color=COL[c], lw=1.2, label=f"{c} (n {d['n']}, events {d['events']})"); ax.fill_between(d["grid"], d["lo"], d["hi"], step="post", color=COL[c], alpha=0.2, lw=0)
        tab.text(-0.09, 0.85 - i * 0.32, c, color="black", ha="right", va="center", fontsize=5.6, transform=tab.transAxes)
        for a, v in zip(AT, d["at_risk"]): tab.text(a / 10, 0.85 - i * 0.32, str(v), ha="center", va="center", fontsize=5.6, transform=tab.transAxes)
        src[m][c] = {"n": d["n"], "patients": d["patients"], "events": d["events"], "at_risk": dict(zip(AT, d["at_risk"])), "km_grid": d["grid"], "km": d["km"], "lo": d["lo"], "hi": d["hi"]}
    tab.text(0.5, 1.12, "number at risk", ha="center", va="bottom", fontsize=5.6, transform=tab.transAxes); cox = S[m]["primary"]["cox"]; src[m]["cox"] = cox
    ax.text(0.03, 0.06, f"HR high vs low {cox['HR_high_vs_low']:.1f} [{cox['HR_high_ci95'][0]:.1f}, {cox['HR_high_ci95'][1]:.1f}]\nHR moderate vs low {cox['HR_moderate_vs_low']:.1f} [{cox['HR_moderate_ci95'][0]:.1f}, {cox['HR_moderate_ci95'][1]:.1f}]", transform=ax.transAxes, fontsize=5.6, va="bottom")
    ax.set_xlim(0, 10); ax.set_ylim(0, 1.3); ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0]); ax.set_xticks(AT + [10]); ax.set_xlabel("years from the biopsy"); ax.set_title(NAMES[m], fontsize=6.5); ax.legend(loc="upper right", bbox_to_anchor=(1.0, 1.0), fontsize=5.2)
    if j == 0: ax.set_ylabel("free of HGD/IMC")
    letter(fig, ax, "abc"[j], dx=-0.055)
save(fig, "ED7"); dump("ED7", src)

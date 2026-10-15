"""Kaplan-Meier by risk class (docs/paper_risk_strata.md Section 3), from results/paper_final/risk_strata/km.json (aggregates only).
Style of be_paper_figs/v3 (matplotlib, blue/orange/red = low/moderate/high, n / events in titles). Writes 10_F_risk_KM.{pdf,png} (sample level) and
10_F_risk_KM_patient_supplementary.{pdf,png} (patient level) to ~/Downloads/be_paper_figs/v3/, plus the plotted numbers as JSON next to the results."""
import json, os, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = json.load(open("results/paper_final/risk_strata/km.json")); OUT = os.path.expanduser("~/Downloads/be_paper_figs/v3")
COL = {"low": "#1f77b4", "moderate": "#ff7f0e", "high": "#d62728"}; NAMES = {"P": "Killcoyne published model", "C": "CNV (L-CNV)", "L": "Late fusion (L-LATE)"}; AT = [0, 2, 4, 6, 8]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
def draw(level, fname, title_unit):
    fig = plt.figure(figsize=(13.5, 6.0)); nums = {}
    for j, m in enumerate(("P", "C", "L")):
        ax = fig.add_axes([0.07 + j * 0.32, 0.40, 0.26, 0.47]); tab = fig.add_axes([0.07 + j * 0.32, 0.11, 0.26, 0.15]); tab.axis("off"); D = R[level][m]; nums[m] = {}
        tot = sum(D[c]["n"] for c in D if D[c]); evs = sum(D[c]["events"] for c in D if D[c])
        for i, c in enumerate(("low", "moderate", "high")):
            d = D.get(c)
            if not d: continue
            ax.step(d["grid"], d["km"], where="post", color=COL[c], lw=1.8, label=f"{c} (n {d['n']}, events {d['events']})"); ax.fill_between(d["grid"], d["lo"], d["hi"], step="post", color=COL[c], alpha=0.15, lw=0)
            tab.text(-0.07, 0.8 - i * 0.33, c, color=COL[c], ha="right", va="center", fontsize=9, transform=tab.transAxes)
            for a, v in zip(AT, d["at_risk"]): tab.text(a / 10, 0.8 - i * 0.33, str(v), ha="center", va="center", fontsize=9, transform=tab.transAxes)
            nums[m][c] = {"n": d["n"], "patients": d["patients"], "events": d["events"], "at_risk": dict(zip(AT, d["at_risk"])), "surv_at": {str(a): d["km"][int(round(a / 0.05))] for a in (2, 4, 6, 8)}}
        tab.text(0.5, 1.05, "number at risk", ha="center", va="bottom", fontsize=9, transform=tab.transAxes)
        ax.set_xlim(0, 10); ax.set_ylim(0, 1.02); ax.set_xticks(AT + [10]); ax.set_xlabel("years from " + title_unit); ax.set_title(f"{NAMES[m]}\n(n {tot} {'samples' if title_unit == 'sample' else 'patients'}, events {evs})", fontsize=10)
        if j == 0: ax.set_ylabel("progression-free (HGD/IMC)")
        ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.text(0.5, 0.995, "Killcoyne risk classes (low Pr ≤ 0.3, moderate 0.3–0.5, high Pr ≥ 0.5), pre-event samples; bands: 95% patient-bootstrap", ha="center", va="top", fontsize=10)
    for ext in ("pdf", "png"): fig.savefig(f"{OUT}/{fname}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig); return nums
nums = {"sample": draw("sample", "10_F_risk_KM", "sample"), "patient": draw("patient", "10_F_risk_KM_patient_supplementary", "earliest pre-event sample")}
os.makedirs("results/paper_final/risk_strata/figs", exist_ok=True); json.dump(nums, open("results/paper_final/risk_strata/figs/10_F_risk_KM.json", "w"), indent=1); print("KM FIG DONE")

"""Extended Data Figure 5, negative controls (docs/paper_figures_v6.md @ d46b9a5; R25, F6), drawn on the cluster from the per-seed random-gate outputs
(killcoyne_mm/review_round2/randgate/seed_*.json, 100 seeds, single model) and the committed permuted-label gate results (5 permutations, F6).
Distributions of share sequenced, sensitivity and specificity with the real H&E gate marked. Source data: per-seed values (no identifiers)."""
import os, sys, glob, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fig6_style import *
from tb_common import M
seeds = [json.load(open(f)) for f in sorted(glob.glob(f"{M}/review_round2/randgate/seed_*.json"))]; assert len(seeds) == 100
f6 = J("review_round2/f6_r25b.json")["F6_R25b"]; r25 = J("review_round2/r25.json")["R25"]; real = f6["real_gate_no_vote"]; pm = f6["per_permutation"]
fig = plt.figure(figsize=(183 * MM, 60 * MM)); src = {"n_seeds": 100, "real_gate": real, "random_gate_committed_R25": {k: r25[k] for k in ("random_gate_share_sequenced", "random_gate_sensitivity", "random_gate_specificity")}, "permuted_gate_committed_F6": pm}
for q, (key, lab) in enumerate((("share_sequenced", "share of biopsies sequenced"), ("sensitivity", "sensitivity"), ("specificity", "specificity"))):
    ax = fig.add_axes([0.06 + q * 0.32, 0.20, 0.26, 0.66]); v = np.array([s[key] for s in seeds]); src[f"random_{key}"] = np.round(v, 4).tolist()
    ax.hist(v, bins=15, color=GREY, alpha=0.8, label="random gate, 100 seeds" if q == 0 else None); yl = ax.get_ylim()[1]
    pv = [pm[k]["no_vote"][key] for k in pm]; src[f"permuted_{key}"] = pv
    for i, x in enumerate(pv): ax.plot([x, x], [0, yl * 0.35], color="#AA4499", lw=1.2, label="permuted-label gate, 5 permutations" if (q == 0 and i == 0) else None)
    xl0, xl1 = min(v.min(), min(pv), real[key]) - 0.03, max(v.max(), max(pv), real[key]) + 0.03; left = real[key] < (xl0 + xl1) / 2; ax.axvline(real[key], color=MODEL["L-IMG"], lw=1.4, label="H&E gate (UNI2-h)" if q == 0 else None); ax.text(real[key] + (0.005 if left else -0.005), yl * 1.02, f"H&E gate {real[key]:.3f}", color=MODEL["L-IMG"], fontsize=6, ha="left" if left else "right", va="bottom")
    ax.text(0.97 if left else 0.03, 0.97, f"random: mean {v.mean():.3f}\n[{np.percentile(v, 2.5):.3f}, {np.percentile(v, 97.5):.3f}]\npermuted: mean {np.mean(pv):.3f}", transform=ax.transAxes, va="top", ha="right" if left else "left", fontsize=5.6)
    ax.set_xlabel(lab); ax.set_ylabel("seeds" if q == 0 else ""); ax.set_xlim(xl0, xl1); ax.set_ylim(0, yl * 1.15)
    letter(fig, ax, "abc"[q], dx=-0.05)
fig.legend(loc="lower center", ncol=3, fontsize=6, bbox_to_anchor=(0.5, -0.01))
save(fig, "ED5"); dump("ED5", src)

"""Extended Data Figure 8, performance by time to event and by grade (docs/paper_figures_v6.md @ d46b9a5), drawn on the cluster from the stored held-out
predictions (tb_common.load). Descriptive fold-stratified AUROC (mean over repeats of within-fold case-control concordance) of L-CNV (shipped matrix),
L-IMG and L-LATE: (a) progressor biopsies taken < 1, 1-3 and > 3 years before the first HGD/IMC endoscopy, each against all 410 non-progressor pre-event
biopsies; (b) biopsies graded NDBE, ID or LGD (both classes within the grade). 95% patient-bootstrap CIs (2,000 draws, models held fixed). Not a
pre-specified analysis; aggregate values only."""
import os, sys, numpy as np, pandas as pd
from scipy.stats import rankdata
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fig6_style import *
from tb_common import *
O, I, FO = load("their"); rows = np.where(popmask("pre"))[0]; TIME = SM.time.values.astype(float); GR = SM.Pathology.values; MODELS = ("L-CNV", "L-IMG", "L-LATE")
def fsa(S, b, mask):
    out = []
    for j in range(10):
        x = S[b, j]; f = FO[b, j]; l = y[b]; ok = mask[b]; num = den = 0.0
        for k in np.unique(f[ok]):
            m = ok & (f == k); yk = l[m]; n1 = int(yk.sum()); n0 = len(yk) - n1
            if n1 == 0 or n0 == 0: continue
            rk = rankdata(x[m]); num += rk[yk == 1].sum() - n1 * (n1 + 1) / 2; den += n1 * n0
        if den: out.append(num / den)
    return float(np.mean(out)) if out else np.nan
strata = {"time": [("< 1 year", (y == 0) | ((y == 1) & (TIME < 1))), ("1–3 years", (y == 0) | ((y == 1) & (TIME >= 1) & (TIME <= 3))), ("> 3 years", (y == 0) | ((y == 1) & (TIME > 3)))],
          "grade": [(g, GR == g) for g in ("NDBE", "ID", "LGD")]}
boots = boots_for(rows); res = {}
for part, lst in strata.items():
    res[part] = {}
    for lab, mk in lst:
        rr = rows[mk[rows]]; obs = {m: fsa(O[m], rows, mk) for m in MODELS}; B = {m: np.array([fsa(O[m], b, mk) for b in boots]) for m in MODELS}
        res[part][lab] = {"n_progressor": int(y[rr].sum()), "n_non_progressor": int((y[rr] == 0).sum()), "n_patients": int(len(set(pat[rr]))), **{m: {"value": round(obs[m], 3), "ci95": pct(B[m][np.isfinite(B[m])])} for m in MODELS}}
        print("ED8", part, lab, res[part][lab], flush=True)
fig = plt.figure(figsize=(183 * MM, 75 * MM))
for q, (part, xl) in enumerate((("time", "time before the first HGD/IMC endoscopy (progressor biopsies) vs all non-progressor biopsies"), ("grade", "pathologist's grade of the biopsy (both classes)"))):
    ax = fig.add_axes([0.07 + q * 0.50, 0.22, 0.42, 0.68]); labs = [l for l, _ in strata[part]]
    for i, lab in enumerate(labs):
        d = res[part][lab]
        for k, m in enumerate(MODELS): ci_point(ax, i + (k - 1) * 0.22, d[m]["value"], d[m]["ci95"], MODEL[m], ms=3.5, label=MODEL_NAME[m] if (i == 0 and q == 0) else None)
        ax.text(i, 0.41, f"{d['n_progressor']} P / {d['n_non_progressor']} NP", ha="center", fontsize=5.6, color="#333333")
    ax.axhline(0.5, color=LIGHT, lw=0.6, ls=":"); ax.set_xticks(range(len(labs))); ax.set_xticklabels(labs); ax.set_ylim(0.38, 1.0); ax.set_xlim(-0.6, len(labs) - 0.4); ax.set_xlabel(xl, fontsize=6)
    if q == 0: ax.set_ylabel("AUROC, fold-stratified [95% CI]"); ax.legend(loc="lower center", bbox_to_anchor=(0.5, 0.08), fontsize=5.6, ncol=3)
    letter(fig, ax, "ab"[q], dx=-0.05)
save(fig, "ED8"); dump("ED8", {"draws": len(boots), "strata": res, "note": "descriptive; time strata use the progressor biopsies of the stratum against all non-progressor pre-event biopsies; grade strata use both classes within the grade"})

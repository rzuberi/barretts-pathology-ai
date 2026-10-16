"""Extended Data Figure 3 (manuscript v7), scanner transfer, from results/paper_final/review_round1/r4.json, review_round2/f2.json, review_round2/r19.json
and triage_robustness/scanner.json. (a) band shares on the C13210 pre-event biopsies; (b) triage sensitivity and specificity in the same three settings;
(c) L-IMG AUROC per setting with the estimand labelled (fold-stratified within a cross-validation, plain for a transfer model), both directions and
both within-scanner cross-validations. Runs locally."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fig7_style import *
r4 = J("review_round1/r4.json")["R4"]; f2 = J("review_round2/f2.json")["F2"]; r19 = J("review_round2/r19.json")["R19"]; sc = J("triage_robustness/scanner.json")
sets = [("mixed-scanner CV\n(10-model vote)", r19["mixed_scanner_cv_same_samples"], "band_shares_mean_over_repeats"), ("C13239-01 → C13210\ntransfer model", r19["transfer_unreset"], "band_shares"), ("transfer model,\nlabel-free reset", r19["transfer_fallback_reset"], "band_shares")]
fig = plt.figure(figsize=(183 * MM, 80 * MM)); ax = fig.add_axes([0.15, 0.20, 0.20, 0.54])
for i, (lab, d, key) in enumerate(sets):
    left = 0
    for b in ("low", "mid", "high"): v = d[key][b]; ax.barh(len(sets) - 1 - i, 100 * v, left=left, color=BAND[b], height=0.6, label=BAND_NAME[b] if i == 0 else None); ax.text(left + 50 * v, len(sets) - 1 - i, f"{100 * v:.0f}%", ha="center", va="center", fontsize=5.6); left += 100 * v
ax.set_yticks(range(len(sets))); ax.set_yticklabels([s[0] for s in sets][::-1], fontsize=5.8); ax.set_xlabel("C13210 pre-event biopsies (%)"); ax.set_xlim(0, 100); ax.legend(loc="lower left", bbox_to_anchor=(-0.1, 1.02), ncol=3, fontsize=5.2); letter(fig, ax, "a", dx=-0.14, dy=0.09)
bx = fig.add_axes([0.44, 0.20, 0.15, 0.54])
for i, (lab, d, key) in enumerate(sets):
    ci_point(bx, i - 0.15, d["sensitivity"], d["sens_ci95"], CLASS["P"], ms=3.5, label="sensitivity" if i == 0 else None); ci_point(bx, i + 0.15, d["specificity"], d["spec_ci95"], CLASS["N"], ms=3.5, label="specificity" if i == 0 else None)
bx.set_xticks(range(len(sets))); bx.set_xticklabels(["mixed CV\n(vote)", "transfer\nmodel", "transfer,\nreset"], fontsize=5.6); bx.set_ylim(0.4, 1.02); bx.set_ylabel("triage on C13210\n(139 biopsies, 9 progressors)"); bx.legend(loc="lower left", fontsize=5.4); letter(fig, bx, "b", dx=-0.08, dy=0.09)
cx = fig.add_axes([0.80, 0.13, 0.19, 0.76]); a = r4["a_main_cv_by_scanner"]; s10 = f2["by_scanner"]["C13210"]; s39 = f2["by_scanner"]["C13239-01"]; W = sc["within"]; X = sc["cross"]
items = [("mixed-scanner CV, scored on C13239-01 (432)\nfold-stratified", a["C13239-01"]["L-IMG"], MODEL["L-IMG"]), ("mixed-scanner CV, 48 training patients,\non C13239-01; fold-stratified", s39["size48_mixed_scanner"], "#7AA6C9"),
         ("mixed-scanner CV, scored on C13210 (139)\nfold-stratified", a["C13210"]["L-IMG"], MODEL["L-IMG"]), ("mixed-scanner CV, 48 training patients,\non C13210; fold-stratified", s10["size48_mixed_scanner"], "#7AA6C9"),
         ("within C13239-01 CV (71 patients)\nfold-stratified", {"value": W["s39"]["auroc_fold_stratified"], "ci95": W["s39"]["ci95"]}, "#7AA6C9"), ("within C13210 CV (27 patients, 9 progressors)\nfold-stratified", {"value": W["s10"]["auroc_fold_stratified"], "ci95": W["s10"]["ci95"]}, "#7AA6C9"),
         ("C13239-01 only (48 patients) → C13210\nplain", {"value": X["cross_b_s39_none"]["auroc"], "ci95": X["cross_b_s39_none"]["ci95"]}, MODEL["L-CNV"]), ("C13210 only (27 patients) → C13239-01\nplain", {"value": X["cross_a_s10_none"]["auroc"], "ci95": X["cross_a_s10_none"]["ci95"]}, MODEL["L-CNV"])]
for i, (lab, d, c_) in enumerate(items): ci_point(cx, d["value"], len(items) - 1 - i, d["ci95"], c_, horizontal=True, ms=3.6)
cx.axvline(0.5, color=LIGHT, lw=0.6, ls=":"); cx.set_yticks(range(len(items))); cx.set_yticklabels([it[0] for it in items][::-1], fontsize=4.9); cx.set_xlim(0.4, 1.0); cx.set_xlabel("L-IMG AUROC [95% CI]"); letter(fig, cx, "c", dx=-0.2, dy=0.02)
save(fig, "ED3"); dump("ED3", {"settings": {s[0].replace("\n", " "): {"band_shares": s[1][s[2]], "sensitivity": s[1]["sensitivity"], "sens_ci95": s[1]["sens_ci95"], "specificity": s[1]["specificity"], "spec_ci95": s[1]["spec_ci95"]} for s in sets}, "auroc": {it[0].replace("\n", "; "): it[1] for it in items}, "n_C13210": r19["setting"]})

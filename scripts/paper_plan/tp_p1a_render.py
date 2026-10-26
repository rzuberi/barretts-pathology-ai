"""Render the P1A aggregate JSONs (results/paper_final/two_papers_P1A/) as markdown tables for docs/two_papers_P1A.md. Runs locally on aggregates only."""
import json, sys, os
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../results/paper_final/two_papers_P1A")
J = lambda f: json.load(open(os.path.join(D, f)))
SM, PT, MN, RB = J("smoke.json"), J("point.json"), J("main.json"), J("refit_boot.json")
f3 = lambda v: "—" if v is None else f"{v:.3f}"
sg = lambda v: "—" if v is None else f"{v:+.3f}"
ci = lambda c: f"[{c[0]:.3f}, {c[1]:.3f}]"
cis = lambda c: f"[{c[0]:+.3f}, {c[1]:+.3f}]"
LV = [("sample", "Sample"), ("patient_weighted", "Patient-weighted"), ("endo_max", "Endoscopy max"), ("endo_mean", "Endoscopy mean"), ("pat_max", "Patient max"), ("pat_mean", "Patient mean")]
out = []
def T(head, rows): out.extend(["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(map(str, r)) + " |" for r in rows] + [""])
# smoke
out += ["### Smoke (known numbers reproduced)", ""]
T(["Quantity", "Expected", "Reproduced"], [[k, SM["expected"].get(k if k != "P published" else "P published (fold-strat)"), v] for k, v in SM["sample_level"].items()] +
  [["L-LATE − L-IMG, old family max-T", "+0.042 [+0.002, +0.078], 0.066", f"{sg(SM['old_family_late_minus_img']['delta'])} {cis(SM['old_family_late_minus_img']['ci95'])}, {SM['old_family_late_minus_img']['p_maxT_family_EARLY_INTER_LATE']}"]] +
  [[f"Refit anchor ({k.split('_')[1]} features, set C rows, 100-λ, class min) vs stored L-CNV fold predictions", "identical", f"max abs Δ {v['max_abs_diff']:.1e}, same λ"] for k, v in SM.items() if k.startswith("anchor")] +
  [["Refit bootstrap (F3) Δ L-IMG − L-CNV, sample level", "+0.070 [−0.114, +0.302]; draw-mean AUROC 0.718 / 0.788 / 0.821", f"{sg(RB['levels']['sample']['L-IMG - L-CNV']['refit_draw_mean'])} {cis(RB['levels']['sample']['L-IMG - L-CNV']['refit_ci95_percentile'])}; {RB['levels']['sample']['draw_mean_auroc']['pred_cnv']:.3f} / {RB['levels']['sample']['draw_mean_auroc']['pred_img']:.3f} / {RB['levels']['sample']['draw_mean_auroc']['late']:.3f}"]])
# item 1
out += ["### 1. Model P refitted in our folds (sample-level and patient-weighted fold-stratified AUROC, 571 pre-event samples)", ""]
VT = PT["P_refit_variants"]; rows = []
nm = {"their": "shipped matrix", "pkg": "our counts via package"}; un = {"s773": "P's population (773 samples, 88 patients)", "C": "set C rows only"}
for k, v in VT.items():
    f, u, _, r = k.split("|"); rows.append([nm[f], un[u], r + (" (pre-specified primary)" if k == MN["primary_P_refit"] else "") + (" (not fold-honest)" if r == "published" else ""), f3(v["sample"]), f3(v["patient_weighted"]), f3(v["endo_max"]), f"{v['lambda_median']:.3f}", int(v["nnz_median"])])
T(["Features", "Training rows", "λ rule", "Sample", "Patient-weighted", "Endoscopy max", "λ median", "Non-zero coef. median"], rows)
# item 2: AUROC table
out += ["### 2. AUROC by unit of analysis (fold-stratified; patient-bootstrap 95% CI, 2,000 draws)", ""]
rows = []
for m in MN["auroc"]:
    rows.append([m] + [f"{f3(MN['auroc'][m][lv]['value'])} {ci(MN['auroc'][m][lv]['ci95'])}" for lv, _ in LV])
T(["Model"] + [h for _, h in LV], rows)
# classification
out += ["**Classification at P's published thresholds** (low ≤ 0.3, high ≥ 0.5; endoscopy score = max sample probability; per repeat, averaged). Sensitivity / specificity [95% CI].", ""]
rows = []; C = MN["classification"]
models = []
for k in C:
    m = k.split("::")[0]
    if m not in models: models.append(m)
for m in models:
    r = [m]
    for lvl in ("endo", "sample"):
        for cl in ("high", "not_low"):
            v = C.get(f"{m}::{lvl}::{cl}"); r.append("—" if v is None else f"{v['sensitivity']:.3f} {ci(v['sens_ci95'])} / {v['specificity']:.3f} {ci(v['spec_ci95'])}")
    rows.append(r)
T(["Model", "Endoscopy, high (≥ 0.5)", "Endoscopy, not low (> 0.3)", "Sample, high", "Sample, not low"], rows)
# item 3: differences
out += ["### 3. Paired differences (patient bootstrap, predictions fixed)", "",
        "Max-T (7) adjusts over the 7 contrasts in the table within the same unit of analysis; max-T (2) over the two primary contrasts only (L-LATE − L-IMG and L-IMG − L-CNV).", ""]
rows = []
for lv, h in LV:
    d = MN["differences"][lv]; f2 = d["maxT_other_families"]["two_primary_LATE-IMG_and_IMG-LCNV"]
    for c, v in d.items():
        if c.startswith("maxT"): continue
        rows.append([h, c, f"{sg(v['delta'])} {cis(v['ci95'])}", v["p_unadjusted"], v["p_maxT"], f2.get(c, "—")])
T(["Unit", "Contrast", "Δ [95% CI]", "p", "max-T p (7)", "max-T p (2)"], rows)
# refit bootstrap
out += ["**Refit bootstrap** (200 draws; one stratified 10-fold CV per draw with λ re-chosen; copies share every split). Draw mean, 95% percentile interval, SD; next to the stored estimate and the fixed-prediction bootstrap SE.", ""]
rows = []
for lv, h in LV:
    for c, v in RB["levels"][lv].items():
        if c == "draw_mean_auroc": continue
        rows.append([h, c, f"{sg(v['refit_draw_mean'])} {cis(v['refit_ci95_percentile'])}", f3(v["refit_sd"]), f3(v["share_draws_le_0"]), sg(v["stored_delta"]), f3(v["fixed_prediction_se"])])
T(["Unit", "Contrast", "Refit mean [95% interval]", "Refit SD", "Share of draws ≤ 0", "Stored Δ", "Fixed-prediction SE"], rows)
out += ["Draw-mean AUROC (refit models, ~48 distinct patients per draw): " + "; ".join(f"{h}: " + ", ".join(f"{k} {v:.3f}" for k, v in RB['levels'][lv]['draw_mean_auroc'].items()) for lv, h in LV) + ".", ""]
# item 4
out += ["### 4. Design checks", "", "**Within-stratum concordance** (strata: sex × age at diagnosis ≤ / > " + str(PT["matching"]["age_dx_median_75"]) + " y × Prague M ≤ 4 / > 4 cm / missing; " +
        f"Prague M missing for {PT['matching']['missing']['prague_m']} patients; within-stratum pairs are {PT['stratum_pairs_share_of_all_pairs']:.1%} of all case–control sample pairs).", ""]
SC = MN["stratum_concordance"]; rows = []
for m in ("L-IMG", "L-CNV", "L-LATE", "P refit", "CN joint package"):
    r = [m]
    for sc in ("unstratified_across_folds", "stratum_across_folds", "stratum_within_fold"):
        for lv in ("sample", "patient_weighted"):
            v = SC.get(f"{m}::{sc}::{lv}"); r.append("—" if v is None else f"{f3(v['value'])} {ci(v['ci95'])}")
    rows.append(r)
for c in ("L-IMG - L-CNV", "L-IMG - P refit", "L-IMG - CN joint package", "L-LATE - L-IMG"):
    r = [c, "—", "—"]
    for sc in ("stratum_across_folds", "stratum_within_fold"):
        for lv in ("sample", "patient_weighted"):
            v = SC.get(f"{c}::{sc}::{lv}"); r.append("—" if v is None else f"{sg(v['delta'])} {cis(v['ci95'])}")
    rows.append(r)
T(["Model / contrast", "All pairs, across folds: sample", "patient-weighted", "Same stratum, across folds: sample", "patient-weighted", "Same stratum and fold: sample", "patient-weighted"], rows)
CL = PT["conditional_logistic_patient_mean_logit"]
T(["Model (patient-mean logit, per SD)", "Conditional OR [95% CI], stratum = matching cell", "Unconditional OR [95% CI]"], [[m, f"{v['conditional_OR_per_SD']:.1f} [{v['conditional_ci95'][0]:.1f}, {v['conditional_ci95'][1]:.1f}]", f"{v['unconditional_OR_per_SD']:.1f} [{v['unconditional_ci95'][0]:.1f}, {v['unconditional_ci95'][1]:.1f}]"] for m, v in CL.items()])
CB = MN["calibration"]; rows = []
for m in ("L-IMG", "L-CNV", "L-LATE", "P refit", "CN joint package", "P published"):
    rows.append([m] + [f"{f3(CB[f'{m}::{w}::{s}']['value'])} {ci(CB[f'{m}::{w}::{s}']['ci95'])}" for w in ("unweighted", "patient_weighted") for s in ("slope", "citl")])
out += ["**Calibration** (logistic recalibration of status on logit(probability), sample level, per repeat and averaged; calibration-in-the-large = intercept with slope fixed at 1).", ""]
T(["Model", "Slope", "Calibration-in-the-large", "Slope, patient-weighted", "CITL, patient-weighted"], rows)
print("\n".join(out))

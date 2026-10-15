"""Append results to docs/paper_plan_killcoyne_cv.md (the pre-specification above is kept as is)."""
import json, sys
R = json.load(open("results/paper_final/killcoyne_cv.json")); FZ = json.load(open("models/killcoyne_frozen_pkg_v1/MANIFEST.json")); MI = json.load(open("models/killcoyne_frozen_pkg_v1/model_info.json"))
commit = sys.argv[1] if len(sys.argv) > 1 else "<results commit>"
f = lambda x: "—" if x is None else f"{x:.3f}"; ci = lambda c: "" if not c else f" [{f(c[0])}, {f(c[1])}]"
dl = lambda d: "—" if not d else f"{d['delta']:+.3f}{ci(d['ci95'])} (p {d['perm_p']})"
SUBN = {"all_C": "All of set C", "a_NDBE_only": "(a) NDBE samples only", "b_before_first_HGD_IMC": "(b) before the first HGD/IMC", "c_NDBE_and_before": "(c) NDBE and before the first HGD/IMC"}
L = ["", "---", "", "## Results", "", f"Sources: `results/paper_final/killcoyne_cv.json` · scripts `scripts/paper_plan/kv_*.{{py,R}}` · frozen models `models/killcoyne_frozen_pkg_v1/` · ACE-B plan `docs/aceb_primary_killcoyne.md` · results commit {commit}. Row-level outputs on the cluster under `feasibility/paper_plan/killcoyne_mm/cv/`.", "",
     "### Status", "", "| Item | Status |", "|---|---|", "| 1 Stratified repeated CV | DONE |", "| 2 LOPO artefact | DONE |", "| 3 Grade arms under the stratified CV | DONE |", "| 4 Frozen package models and ACE-B primary | DONE |", ""]
fs = R["fold_sizes"]; L += ["Fold composition (repeat 1): " + ", ".join(f"fold {k} {v['patients']} patients / {v['P']} P" for k, v in fs.items()) + ".", ""]
L += ["### 1. Stratified 10-fold × 10 CV vs LOPO", ""]
for src in ["their", "pkg"]:
    for sub in ["all_C", "a_NDBE_only", "b_before_first_HGD_IMC", "c_NDBE_and_before"]:
        X = R["item1"][src][sub]; s0 = R["subsets"][sub]
        L += [f"**{SUBN[sub]}**, CNV source {src} ({s0['n_samples']} samples, {s0['n_patients']} patients, {s0['n_P_patients']} P).", "",
              "| Arm | CV per-sample AUROC | CV Δ vs L-CNV | CV patient-max AUROC | CV Δ vs L-CNV | LOPO per sample / Δ | LOPO patient max / Δ |", "|---|---|---|---|---|---|---|"]
        for a, v in X.items():
            d = v.get("delta_vs_L_CNV", {}); lo = v["LOPO"]; ld = lo.get("delta_vs_L_CNV") or {}
            L.append(f"| {a} | {f(v['per_sample_auroc'])}{ci(v['ci95']['per_sample_auroc'])} | {dl(d.get('per_sample_auroc'))} | {f(v['patient_max_auroc'])}{ci(v['ci95']['patient_max_auroc'])} | {dl(d.get('patient_max_auroc'))} | {f(lo['per_sample_auroc'])} / {dl(ld.get('per_sample_auroc'))} | {f(lo['patient_max_auroc'])} / {dl(ld.get('patient_max_auroc'))} |")
        L.append("")
I2 = R["item2"]; L += ["### 2. The LOPO artefact", "", f"LOPO intercepts recovered by refitting at the chosen λ; recomputed predictions match the stored ones to {I2['lopo_refit_max_abs_pred_diff']} (max absolute difference).", "",
      "| Arm | LOPO: corr(label, intercept) Pearson / Spearman | Stratified CV: Pearson / Spearman |", "|---|---|---|"]
for a, v in I2["intercept_label_correlation"].items():
    lo, cv = v["LOPO"], v["stratified_CV"]; L.append(f"| {a} | {f(lo['pearson'])} / {f(lo['spearman'])} | {f(cv['pearson'])} / {f(cv['spearman'])} |")
L += ["", "Intercept-only model (prediction = training-row prevalence), per-sample AUROC:", "", "| Subset | LOPO | Stratified CV |", "|---|---|---|"] + [f"| {SUBN[k]} | {f(v['LOPO'])} | {f(v['stratified_CV'])} |" for k, v in I2["intercept_only_per_sample_auroc"].items()] + [""]
L += ["### 3. Grade arms under the stratified CV", "", "Raw grade as a score: " + "; ".join(f"{SUBN[k]} per sample {f(v['per_sample_auroc'])}{ci(v['ci95']['per_sample_auroc'])}, patient max {f(v['patient_max_auroc'])}{ci(v['ci95']['patient_max_auroc'])}" for k, v in R["item3"]["raw_grade"].items()) + ".", ""]
for src in ["their", "pkg"]:
    for sub in ["all_C", "a_NDBE_only"]:
        X = R["item3"][src][sub]; L += [f"**{SUBN[sub]}**, CNV source {src}.", "", "| Arm | Per-sample AUROC | Patient-max AUROC | Δ per sample | Δ patient max |", "|---|---|---|---|---|"]
        for a, v in X.items():
            d = v.get("delta_vs_L_CNV+GRADE") or v.get("delta_vs_L_CNV"); ref = " vs L-CNV+GRADE" if "delta_vs_L_CNV+GRADE" in v else (" vs L-CNV" if d else "")
            L.append(f"| {a} | {f(v['per_sample_auroc'])}{ci(v['ci95']['per_sample_auroc'])} | {f(v['patient_max_auroc'])}{ci(v['ci95']['patient_max_auroc'])} | {dl(d['per_sample_auroc']) + ref if d else '—'} | {dl(d['patient_max_auroc']) + ref if d else '—'} |")
        L.append("")
D = R["item4_development_estimate"]
L += ["### 4. Frozen package-feature models", "", f"Bundle SHA-256 `{FZ['bundle_sha256']}`. Reconstruction of the set-C package block from raw tiles with the frozen constants: max absolute difference {MI['reconstruction_max_abs_diff']:.2e}.", "",
      "| Model | λ | Features | Non-zero coefficients | In-sample AUROC (not a performance estimate) |", "|---|---|---|---|---|"]
for nm in ["L_CNV", "L_IMG", "L_EARLY"]:
    v = MI[nm]; L.append(f"| {nm.replace('_', '-')} | {v['lambda']:.5f} | {v['n_features']} | {v['nonzero']} | {MI['insample_check']['auroc_' + nm]:.3f} |")
L += [f"| L-LATE | mean of L-CNV and L-IMG | — | — | {MI['insample_check']['auroc_L_LATE']:.3f} |", "",
      f"Development estimate of the ACE-B primary contrast ({D['contrast']}; {D['n_samples']} samples, {D['n_patients']} patients, {D['n_P_patients']} P): Δ {D['delta']:+.3f}, one-sided 95% lower bound {D['one_sided_95_lower']:+.3f}, two-sided 95% CI [{D['two_sided_95'][0]:+.3f}, {D['two_sided_95'][1]:+.3f}].", "", "INTERPRETATION_LINE", ""]
open("docs/paper_plan_killcoyne_cv.md", "a").write("\n".join(L)); print("rendered")

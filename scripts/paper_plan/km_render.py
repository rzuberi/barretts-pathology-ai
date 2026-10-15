"""Append results to docs/paper_plan_killcoyne_multimodal.md (the pre-specification above is kept as is)."""
import json, sys
R = json.load(open("results/paper_final/killcoyne_multimodal.json")); commit = sys.argv[1] if len(sys.argv) > 1 else "<results commit>"
f = lambda x: "—" if x is None else f"{x:.3f}"; ci = lambda c: "" if not c else f" [{f(c[0])}, {f(c[1])}]"
P = R["prep"]; L = ["", "---", "", "## Results", "", f"Sources: `results/paper_final/killcoyne_multimodal.json` · scripts `scripts/paper_plan/km_*.{{py,R}}` · results commit {commit}. Row-level outputs on the cluster under `feasibility/paper_plan/killcoyne_mm/`.", ""]
L += ["### Status", "", "| Arm | Status |", "|---|---|"] + [f"| {k} | {v} |" for k, v in R["status"].items() if "[" not in k] + [""]
L += ["### Set C", "", f"{P['C_samples']} samples, {P['C_patients']} patients ({P['C_P_patients']} P patients, {P['C_P_samples']} P samples); pathology {P['pathology_C']}; {P['multi_slide_samples']} samples with more than one slide (the pre-specification said 7); {P['excluded_no_image']} published samples without a UNI2 bag, patients {', '.join(P['excluded_patients'])} have none; {P['release_rows_in_C']} of the samples are release rows.", ""]
A = R["anchors_773"]; L += ["CNV-only anchors on all 773 samples (reconciliation): " + "; ".join(f"{k} {v:.3f}" for k, v in A.items()) + ".", ""]
for src in ["their", "pkg"]:
    L += [f"### Primary rule (class-error λ min per left-out patient; neural = mean of 3 seeds), CNV source: {'their shipped matrix' if src == 'their' else 'package features from our counts'}", "",
          "| Arm | Per-sample AUROC | Patient-mean AUROC | Patient-max AUROC | Per-sample AUPRC | Patient-max AUPRC | Release rows: per sample / patient max | Δ per sample vs L-CNV (p) | Δ patient max vs L-CNV (p) |", "|---|---|---|---|---|---|---|---|---|"]
    for nm, v in R["arms"].items():
        if "[" in nm: continue
        if src == "their" and "(pkg" in nm: continue
        if src == "pkg" and not ("(pkg" in nm or nm == "N-IMG" or nm == "L-IMG"): continue
        d = R["deltas_vs_L_CNV"].get(nm); rr = v["release_rows"]
        ds = "—" if not d else f"{d['per_sample_auroc']['delta']:+.3f}{ci(d['per_sample_auroc']['ci95'])} ({d['per_sample_auroc']['perm_p']})"
        dm = "—" if not d else f"{d['patient_max_auroc']['delta']:+.3f}{ci(d['patient_max_auroc']['ci95'])} ({d['patient_max_auroc']['perm_p']})"
        if src == "pkg" and nm in ("N-IMG", "L-IMG"): ds = dm = "(see their-source table)"
        L.append(f"| {nm} | {f(v['per_sample_auroc'])}{ci(v.get('ci95', {}).get('per_sample_auroc'))} | {f(v['patient_mean_auroc'])} | {f(v['patient_max_auroc'])}{ci(v.get('ci95', {}).get('patient_max_auroc'))} | {f(v['per_sample_auprc'])} | {f(v['patient_max_auprc'])} | {f(rr['per_sample_auroc'])} / {f(rr['patient_max_auroc'])} | {ds} | {dm} |")
    L.append("")
L += ["### Post hoc: same predictions scored without the HGD/IMC samples", "", "Added after the pre-specification: under the paper's label the HGD/IMC diagnostic samples are progressor samples, and their slides show the diagnosis itself. No model is refitted.", "",
      "| Arm | Samples | Per-sample AUROC | Patient-max AUROC |", "|---|---|---|---|"] + [f"| {nm} | {v['excl_HGD_IMC_post_hoc']['n_samples']} | {f(v['excl_HGD_IMC_post_hoc']['per_sample_auroc'])} | {f(v['excl_HGD_IMC_post_hoc']['patient_max_auroc'])} |" for nm, v in R["arms"].items() if "[" not in nm] + [""]
L += ["### Secondary λ rules (linear tier)", "", "| Arm | Per-sample AUROC | Patient-max AUROC |", "|---|---|---|"] + [f"| {nm} | {f(v['per_sample_auroc'])} | {f(v['patient_max_auroc'])} |" for nm, v in R["arms"].items() if "[" in nm] + [""]
L += ["### Neural seeds", "", "Per-sample AUROC by seed: " + "; ".join(f"{m} {s}" for m, s in R["neural_seed_auroc"].items()) + ".", "", "Best epoch (early stopping) by model: " + "; ".join(f"{m} mean {v['mean']} (min {v['min']}, max {v['max']})" for m, v in R.get("neural_best_epoch", {}).items()) + ".", "", "INTERPRETATION_LINE", ""]
open("docs/paper_plan_killcoyne_multimodal.md", "a").write("\n".join(L)); print("rendered")

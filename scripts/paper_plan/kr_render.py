"""Append the results of the Killcoyne reconciliation to docs/paper_plan_killcoyne_reconcile.md (pre-specification above is kept as is)."""
import json, sys
R = json.load(open("results/paper_final/killcoyne_reconcile.json")); commit = sys.argv[1] if len(sys.argv) > 1 else "<results commit>"
f = lambda x: "—" if x is None else (f"{x:.3f}" if isinstance(x, float) else str(x))
ci = lambda c: "" if not c else f" [{f(c[0])}, {f(c[1])}]"
L = ["", "---", "", "## Results", "", f"Sources: `results/paper_final/killcoyne_reconcile.json` · scripts `scripts/paper_plan/kr_*.{{py,R}}` · results commit {commit}. Row-level outputs on the cluster under `feasibility/paper_plan/killcoyne/`.", ""]
A0 = R["A0"]; L += ["### Status", "", "| Item | Status |", "|---|---|"]
st = {"A0": "DONE", "A": "DONE", "B": "DONE" if "B" in R else "NOT AVAILABLE", "C": "PARTIAL (release features exist for release rows only; release features x 773 / 711 NOT AVAILABLE)", "D": "DONE"}
L += [f"| {k} | {v} |" for k, v in st.items()] + [""]
L += ["### A0. Published probabilities against sheet status", "", "| Population | Samples | Patients (P) | Per sample | Patient mean | Patient max |", "|---|---|---|---|---|---|"]
for k, lab in [("all_773", "All 773 published"), ("excl_HGD_IMC", "Without HGD/IMC samples"), ("release_rows_matched", "Our 500 release rows (82 patients)"), ("sheet_Set_Training", "Sheet Set = Training"), ("sheet_excluded_0", "Sheet excluded = 0"), ("sheet_remove_0", "Sheet remove = 0")]:
    v = A0[k]; L.append(f"| {lab} | {v['n_samples']} | {v['n_patients']} ({v['n_P_patients']}) | {v['per_sample']['auroc_4dp']:.4f}{ci(v['per_sample']['ci95'])} | {v['patient_mean']['auroc']:.3f}{ci(v['patient_mean']['ci95'])} | {v['patient_max']['auroc']:.3f}{ci(v['patient_max']['ci95'])} |")
L += ["", f"Label-permutation p is {A0['all_773']['per_sample']['perm_p']} for every row above. No population gives 0.8713; the value closest to the quoted 0.8713 is the patient-max AUROC on our 82 patients, 0.873.", ""]
A = R["A"]; a = A["all_segmented"]; al = R["alignment"]
L += ["### A. Their frozen model on our counts", "", f"Segmented {A['segmented']} of 773 samples; {A['qc_pass']} pass the package QC (varMAD ≤ 0.008). The model was trained on these samples, so this is a fidelity check.", "",
      "| Input | Samples | Per sample | Patient mean | Patient max | Spearman vs published |", "|---|---|---|---|---|---|"]
for k, lab in [("all_segmented", "Our counts, all segmented"), ("qc_pass_only", "Our counts, QC pass"), ("release_rows", "Our counts, release rows"), ("shipped_matrix_insample", "Their shipped training matrix (in sample)")]:
    v = A[k]; L.append(f"| {lab} | {v['n_samples']} | {f(v['per_sample'])}{ci(v.get('ci95', {}).get('per_sample'))} | {f(v['patient_mean'])} | {f(v['patient_max'])}{ci(v.get('ci95', {}).get('patient_max'))} | {f(v['spearman_vs_published'])} |")
L += ["", f"Row alignment of the shipped matrix to our samples [post hoc]: method {al['alignment_used']}; mean row correlation of the aligned pairs {al['hungarian']['mean_corr']} (random pairing {al['random_order_mean_diag_corr']}); candidate orders {al['candidate_order_mean_diag_corr']}; aligned rows form {al['patient_runs']} patient runs over {al['patients']} patients.", ""]
if "B" in R:
    b = R["B"]; L += ["### B. Faithful LOPO retrain (package features from our counts, 773 samples, α 0.9, class-error λ min)", "", "| Evaluation | Samples | Per sample | Patient mean | Patient max | Spearman vs published |", "|---|---|---|---|---|---|"]
    for k, lab in [("own", "All 773"), ("rel500", "Release rows")]:
        v = b[k]; L.append(f"| {lab} | {v['n_samples']} | {f(v['per_sample'])}{ci(v['ci95']['per_sample'])} | {f(v['patient_mean'])}{ci(v['ci95']['patient_mean'])} | {f(v['patient_max'])}{ci(v['ci95']['patient_max'])} | {f(v['spearman_vs_published'])} |")
    L += ["", f"Permutation p (all 773, per sample / patient max): {b['own']['perm_p']['per_sample']} / {b['own']['perm_p']['patient_max']}.", ""]
L += ["### C. Grid (every configuration and λ rule)", "", R["criterion"] + ". Rows tagged post hoc were added after the pre-specification, from the shipped model object. Picking a best row from this grid is a post hoc selection.", "",
      "| cfg | tag | set | features | standardisation | glmnet standardize | nλ | α | λ rule | own per sample | own patient max | 82 patients max | release rows per sample | release rows max | Spearman vs published | reproduces |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for r in sorted(R["grid"], key=lambda r: (r["cfg"], r["rule"])):
    L.append(f"| {r['cfg']} | {r['tag']} | {r['set']} | {r['feat']} | {r['std']} | {r['stdz']} | {r['nlam']} | {r['alpha']} | {r['rule']} | {f(r['own']['per_sample'])} | {f(r['own']['patient_max'])} | {f(r['disc82']['patient_max'])} | {f(r['rel500']['per_sample'])} | {f(r['rel500']['patient_max'])} | {f(r['own']['spearman_vs_published'])} | {r.get('reproduces', '—')} |")
L += ["", "### D. Step table from our 0.800 to the published value", "", "| Step | Release rows patient max | Release rows per sample | 773 per sample | Spearman vs published |", "|---|---|---|---|---|"]
for d in R["D"]:
    if d.get("status"): L.append(f"| {d['step']} | {d['status']} | | | |"); continue
    L.append(f"| {d['step']} | {f(d['rel500_patient_max'])}{ci(d.get('rel500_patient_max_ci'))} | {f(d['rel500_per_sample'])} | {f(d['s773_per_sample'])} | {f(d['spearman_vs_published'])} |")
L += ["", "RECONCILIATION_LINE", ""]
open("docs/paper_plan_killcoyne_reconcile.md", "a").write("\n".join(L)); print("rendered")

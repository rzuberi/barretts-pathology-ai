"""Render docs/two_papers_P1B.md result tables from results/paper_final/two_papers_P1B/*.json (aggregates only). Prints markdown to stdout."""
import json, os, sys
R = os.environ.get("P1B_R") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../results/paper_final/two_papers_P1B")
J = lambda f: json.load(open(f"{R}/{f}.json"))
def ci(c): return f"[{c[0]:.3f}, {c[1]:.3f}]" if c and c[0] is not None else "[–]"
def a(x): return f"{x['value']:.3f} {ci(x['ci95'])}"
def d(x): return f"{x['delta']:+.3f} {ci(x['ci95'])}"
def hrow(name, h, extra=""):
    A = h["auroc"]; return f"| {name}{extra} | {h['n_samples']} / {h['n_patients']} ({h['n_progressor_patients']}) | {a(A['L-CNV'])} | {a(A['L-IMG'])} | {a(A['L-LATE'])} | {d(h['IMG_minus_CNV'])} | {d(h['LATE_minus_IMG'])} |"
HDR = "| Rows | Samples / patients (P) | L-CNV | L-IMG | L-LATE | Δ IMG − CNV | Δ LATE − IMG |\n|---|---|---|---|---|---|---|"
out = []
B = J("batch")["batch"]
out += ["### 1. Batch", "", HDR]
for k, nm in (("overall", "All pairs"), ("within_batch", "Within sequencing batch"), ("within_scanner", "Within scanner (H&E batch)"), ("within_batch_and_scanner", "Within batch and scanner")):
    for w, wn in (("sample", ""), ("patient_weighted", ", patient-weighted")):
        h = B[k][w]; out.append(hrow(nm + wn, h).replace(f"| {h['n_samples']} /", f"| {h['pairs_per_repeat']:.0f} pairs; {h['n_samples']} /"))
bs = B["batch_by_scanner"]; ov = B["one_variable_auroc"]
out += ["", f"Batch × scanner: Cramér's V {bs['cramers_v']}; {bs['batches_with_both_scanners']} of {bs['n_batches']} batches hold slides from both scanners; patients with slides from both scanners {bs['patients_with_both_scanners']}, with samples in several batches {bs['patients_with_several_batches']}.",
        f"One-variable AUROC (fold-honest target encoding): batch {a(ov['batch_target_encoded'])}; scanner {a(ov['scanner_target_encoded'])}; batch × scanner {a(ov['batch_x_scanner_target_encoded'])}.", "",
        "| Batch probe (mean one-vs-rest AUROC over batches) | Copy-number matrix | UNI2-h slide mean |", "|---|---|---|"]
for k, nm in (("pre_event_all", "All pre-event samples (≥ 5 patients per batch)"), ("non_progressor_pre_event", "Non-progressors only (≥ 3)"), ("progressor_pre_event", "Progressors only (≥ 3)")):
    p = B["batch_probe"][k]; out.append(f"| {nm} | {p['copy_number_matrix']['auroc_mean_over_batches']} ({p['copy_number_matrix']['n_batches_probed']} batches) | {p['uni2h_slide_mean']['auroc_mean_over_batches']} ({p['uni2h_slide_mean']['n_batches_probed']}) |")
sv = B["score_batch_association_within_class"]
out += ["", "| Held-out score vs batch, within class (patient level) | η² non-progressors (perm p) | η² progressors (perm p) |", "|---|---|---|"]
for m in ("L-CNV", "L-IMG"): out.append(f"| {m} | {sv[m]['non_progressor']['eta_squared']} ({sv[m]['non_progressor']['perm_p']}; null {sv[m]['non_progressor']['null_mean_eta_squared']}) | {sv[m]['progressor']['eta_squared']} ({sv[m]['progressor']['perm_p']}; null {sv[m]['progressor']['null_mean_eta_squared']}) |")
D = J("depth")["depth"]; dd = D["depth_distribution"]; b = dd["bam_based_pre_event"]; s = dd["sheet_reads_x50bp_pre_event"]
out += ["", "### 2. Depth, sWGS QC and biopsy size", "",
        f"Realised depth, pre-event samples: median {b['median']}× (IQR {b['iqr'][0]}–{b['iqr'][1]}; 10th–90th centile {b['p10_p90'][0]}–{b['p10_p90'][1]}; range {b['min']}–{b['max']}); {b['n_below_0.1x']} samples below 0.1×, {b['n_above_1x']} above 1×. Sheet read count × 50 bp: median {s['median']}× (IQR {s['iqr'][0]}–{s['iqr'][1]}). Progressors {dd['by_class_pre_event']['progressor']['median']}×, non-progressors {dd['by_class_pre_event']['non_progressor']['median']}× (Δ median {d(D['depth_progressor_minus_nonprogressor_median'])}).", "", HDR]
T3 = D["depth_tertiles"]; cut = T3["cuts_x"]
for k, nm in (("0", f"Depth low tertile (< {cut[0]}×)"), ("1", f"Depth middle ({cut[0]}–{cut[1]}×)"), ("2", f"Depth high (> {cut[1]}×)")):
    out.append(hrow(nm, T3[k]) if "auroc" in T3[k] else f"| {nm} | {T3[k]['status']} |")
out.append(hrow("Pairs within the same depth tertile", T3["within_depth_tertile_pairs"]))
for k, nm in (("1.0", "sWGS QC pass"), ("0.0", "sWGS QC fail")):
    x = D["sWGS_QC_pass"]["strata"].get(k); out.append(hrow(nm, x) if x and "auroc" in x else f"| {nm} | {x['n_samples'] if x else '–'} samples; {x['status'] if x else ''} | | | | | |")
V = D["sWGS_noise_varMAD_tertiles"]
for k, nm in (("0", "varMAD low tertile (least noisy)"), ("1", "varMAD middle"), ("2", "varMAD high (noisiest)")): out.append(hrow(nm, V[k]) if "auroc" in V[k] else f"| {nm} | {V[k]['status']} |")
S = D["biopsy_size"]
for k, nm in (("1", "Small biopsy (< 256 tissue tiles)"), ("0", "Large biopsy (≥ 256)")): out.append(hrow(nm, S[k]))
for k, nm in (("0", "Tissue low tertile (< 200 tiles)"), ("1", "Tissue middle (200–302)"), ("2", "Tissue high (> 302)")): out.append(hrow(nm, S["tissue_tertiles_200_302"][k]))
out.append(hrow("Small biopsy, depth-matched pairs", S["depth_matched_pairs"]["small"])); out.append(hrow("Large biopsy, depth-matched pairs", S["depth_matched_pairs"]["large"]))
lm = T3.get("low_minus_high_tertile", {}); sl = S["small_minus_large"]; dm = S["depth_matched_pairs"]["small_minus_large"]
out += ["", "| Interaction (difference in differences) | Estimate [95% CI] | p |", "|---|---|---|"]
for nm, x in (("Small − large biopsy: Δ IMG − CNV", sl["IMG_minus_CNV"]), ("Small − large: L-CNV AUROC", sl["L-CNV"]), ("Small − large: L-IMG AUROC", sl["L-IMG"]), ("Small − large: Δ LATE − IMG", sl["LATE_minus_IMG"]),
              ("Small − large, depth-matched pairs: Δ IMG − CNV", dm["IMG_minus_CNV"]), ("Small − large, depth-matched pairs: L-CNV", dm["L-CNV"]),
              ("Low − high depth tertile: Δ IMG − CNV", lm.get("IMG_minus_CNV")), ("Low − high depth: L-CNV", lm.get("L-CNV")), ("Low − high depth: L-IMG", lm.get("L-IMG")),
              ("Low − high tissue tertile: Δ IMG − CNV", S["tissue_tertiles_200_302"].get("low_minus_high", {}).get("IMG_minus_CNV")), ("Least − most noisy varMAD tertile: L-CNV", V.get("low_noise_minus_high_noise", {}).get("L-CNV"))):
    if x: out.append(f"| {nm} | {d(x)} | {x['p_two_sided']} |")
sv2 = D["size_vs_depth"]
out += ["", f"Small biopsies are not low depth: Spearman (tissue tiles, depth) {sv2['spearman_tissue_tiles_vs_depth']['value']} {ci(sv2['spearman_tissue_tiles_vs_depth']['ci95'])}; median depth small {sv2['median_depth_small']}× vs large {sv2['median_depth_large']}× (Δ {d(sv2['median_depth_small_minus_large'])}). Small-biopsy samples by depth tertile (low / mid / high): {sv2['small_by_depth_tertile'].get('1')}; large: {sv2['small_by_depth_tertile'].get('0')}."]
out += ["", "### 3. Image-model selection", "", "| Candidate set | n | Nested AUROC | Nested − L-IMG | L-IMG chosen (of 100 folds) | Rank of L-IMG | Apparent best (held-out, optimistic) | Optimism: apparent best − nested | Nested IMG − L-CNV | Nested LATE | Nested LATE − L-CNV |", "|---|---|---|---|---|---|---|---|---|---|---|"]
for f, nm in (("select_e3_9", "E3 extractors (9)"), ("select_e3_11", "E3 scales and concatenations (11)"), ("select_poolings_11", "Poolings (11)"), ("select_extractors_scales_poolings_ensembles", "Everything with inner predictions")):
    if not os.path.exists(f"{R}/{f}.json"): continue
    x = J(f)["select"]; ab = x["apparent_best_candidate"]
    out.append(f"| {nm} | {x['n_candidates']} | {a(x['auroc']['IMG-nested'])} | {d(x['nested_minus_L_IMG'])} | {x['times_chosen_of_100_folds'].get('base')} | {x['rank_of_L_IMG_among_candidates']} | {ab['auroc']:.3f} ({ab['name']}) | {d(x['optimism_apparent_best_minus_nested'])} | {d(x['nested_IMG_minus_L_CNV'])} | {a(x['auroc']['LATE-nested'])} | {d(x['nested_LATE_minus_L_CNV'])} |")
G = J("grade")["grade"]; GA = G["auroc"]
out += ["", "### 4. Grade and p53", "", "| Model (held-out, nested) | AUROC [95% CI] |", "|---|---|"] + [f"| {k} | {a(v)} |" for k, v in GA.items()]
out += ["", "| Difference | Δ [95% CI] | p |", "|---|---|---|"] + [f"| {k.replace('_minus_', ' − ')} | {d(v)} | {v['p_two_sided']} |" for k, v in G["differences"].items()]
out += ["", HDR] + [hrow(nm, G["within_grade"][k]) for k, nm in (("ndbe_only", "NDBE samples only"), ("ndbe_only_patient_weighted", "NDBE only, patient-weighted"), ("pairs_sharing_grade", "Pairs sharing grade"), ("p53_normal_only", "p53-normal samples only"), ("ndbe_and_p53_normal", "NDBE and p53-normal"))]
out += ["", "Apparent (non-nested, optimistic): " + "; ".join(f"{k} {v}" for k, v in G["apparent_optimistic"].items()) + "."]
SN = J("sens")["sens"]
out += ["", "### 5. Sensitivity rows", "", HDR, hrow("Primary: original (Killcoyne sheet) labels", SN["primary_original_labels"])]
for k, nm in (("db_event_dates_eval_only", "Database event dates (R53; eval-only)"), ("sheet_review_inconsistent_removed", "Sheet review 'inconsistent' removed (eval-only)"), ("sheet_excluded_flag_removed", "Sheet 'excluded' flag removed (eval-only)"), ("same_block_only", "Same block only (eval-only)")):
    out.append(hrow(nm, SN[k], f" (−{SN[k]['removed_samples']})"))
for k, x in SN["refits"].items():
    if "auroc" in x: out.append(hrow(f"Refit {k}: {x['description']}", x))
out += ["", f"Label check: sheet Status equals the sheet's original P/NP column for {SN['label_check']['pre_event_samples_status_equals_original_column']} of {SN['label_check']['n']} pre-event samples. Refit reproduction (repeat 1, max |Δ| vs stored): {SN['refits']['base'].get('repeat1_max_abs_diff_vs_stored')}. Same-block labels: {SN['same_block_only']['labels_pre_event']}."]
E = J("exclsel")["exclsel"]; c = E["counts"]
out += ["", f"**Samples without a slide.** {c['without_slide']} of {c['published_samples']} published samples (progressor {c['without_slide_by_class']['progressor']}, non-progressor {c['without_slide_by_class']['non_progressor']}; pre-event {c['without_slide_pre_event_by_class']}); {c['patients_without_any_slide']} patients have none, {c['without_slide_from_patients_with_other_slides']} samples come from patients who do.", "",
        "| Published model P AUROC | Samples (P) | AUROC [95% CI] |", "|---|---|---|"] + [f"| {k} | {v['n_samples']} ({v['n_progressor']}) | {v['auroc_published_P_probability']:.3f} {ci(v['ci95'])} |" for k, v in E["published_model_P_auroc"].items()]
out += ["", "| Within class: without − with slide | P probability | CN burden | Reads | Grade | p53 aberrant | Year |", "|---|---|---|---|---|---|---|"]
for k, v in E["within_class_comparison"].items():
    out.append(f"| {k} | " + " | ".join((f"{v[q]['mean_without']} vs {v[q]['mean_with']}; SMD {d(v[q]['std_mean_difference_without_minus_with'])}" if isinstance(v[q]['std_mean_difference_without_minus_with'], dict) else "n < 5") for q in ("k", "cn_abs", "reads", "grade", "p53", "year")) + " |")
print("\n".join(out))

"""Render docs/paper_review_round4_analyses.md results (pre-specification 45b77bf) from results/paper_final/review_round4/*.json.
Replaces only the '(appended after running)' placeholder; the pre-specification above the '---' line is not touched. Idempotent."""
import json, os
T = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
R = T + "/results/paper_final/review_round4"; DOC = T + "/docs/paper_review_round4_analyses.md"
J = lambda f: json.load(open(f"{R}/{f}"))
f3 = lambda x: "" if x is None else (f"{x:+.3f}" if isinstance(x, float) and x < 1 else f"{x}")
ci = lambda c: f"[{c[0]:+.3f}, {c[1]:+.3f}]" if c else ""
yn = lambda b: "yes" if b else "no"
L = []; A = L.append

r31 = J("r31.json")["R31"]; r32a = J("r32a.json")["R32a"]; r32b = J("r32b.json")["R32b"]; r33 = J("r33_merge.json")["R33"]
r34 = J("r34.json")["R34"]; r35 = J("r35.json")["R35"]; r36 = J("r36.json")["R36"]; b5 = J("b5.json")["B5"]

A("Rendered by `scripts/paper_plan/rv4_render.py` from `results/paper_final/review_round4/` (Part A results 9d50255, Part B results 658c9fc).\n")
A("| Item | Status |\n|---|---|")
for it, st in [("R31 batch and processing variables", "DONE"), ("R32 tissue strata, tissue-adjusted gate", "DONE"), ("R33 robustness tables", "DONE (all rows filled)"),
               ("R34 patient-weighted triage", "DONE"), ("R35 copy-number-only middle band, frozen cut-offs, band stability", "DONE"),
               ("R36 per-sample re-segmentation", "DONE (plus past-only variant)"), ("B5 inner predictions with λ by deviance", "DONE")]:
    A(f"| {it} | {st} |")

# R31
A("\n### R31. Sequencing batch and processing variables\n")
tb = r31["a_cross_tabulation"]["batch"]["by_pre_event_sample"]["table"]
A("Pre-event samples by batch (progressor / non-progressor). Fisher's exact test (Monte Carlo, 10,000): p < 0.0001.\n")
A("| Batch | Progressor | Non-progressor |\n|---|---|---|")
for b, v in tb.items(): A(f"| {b} | {v['progressor']} | {v['non_progressor']} |")
rb = r31["b_raw_score_auroc"]
A("\nFold-stratified AUROC as raw scores (fold-honest target encoding for batch and run):\n")
A("| Variable | AUROC [95% CI] |\n|---|---|")
for k, nm in [("batch_target_encoded", "Batch"), ("slx_run_target_encoded", "SLX run"), ("number_of_reads", "Number of reads"), ("pct_barretts_cellularity", "% Barrett's cellularity (111 samples)"), ("pilot_flag", "Pilot study")]:
    A(f"| {nm} | {rb[k]['value']:.3f} {ci(rb[k]['ci95'])} |")
wb = r31["c_within_batch_pairs"]
A("\nWithin-batch pairs only (283 to 596 pairs per repeat):\n")
A("| Pairs | L-CNV | L-IMG | L-LATE | Δ L-IMG − L-CNV [95% CI] | One-sided bound |\n|---|---|---|---|---|---|")
for k, nm in [("batch", "same batch"), ("slx_run", "same SLX run")]:
    w = wb[k]; a = w["auroc"]; d = w["IMG_minus_CNV"]
    A(f"| {nm} | {a['L-CNV']['value']:.3f} | {a['L-IMG']['value']:.3f} | {a['L-LATE']['value']:.3f} | {d['delta']:+.3f} {ci(d['ci95'])} | {d['one_sided_95_lower']:+.3f} |")
pr = r31["d_batch_probe_uni2h"]["pre_event_samples"]
A(f"\nBatch probe on the UNI2-h embedding (patient-grouped 5-fold, one-vs-rest, batches with ≥ 5 patients): mean AUROC {pr['auroc_mean_over_batches']:.3f}; "
  + ", ".join(f"{b} {v['auroc_one_vs_rest']:.3f}" for b, v in pr["batches"].items()) + ".")
e = r31["e_triage_largest_batch"]
A(f"\nLargest batch ({e['largest_batch']['batch']}, {e['largest_batch']['n_samples']} samples) holds no progressor samples, so Δ sensitivity is not estimable there; triage specificity {e['largest_batch']['specificity_single_model']['T']['value']:.3f} against B\\* {e['largest_batch']['specificity_single_model']['B*']['value']:.3f}.")
fl = r31["f_exclusion_flags"]
A(f"\nSheet flags among the 571 pre-event samples (progressor / non-progressor): excluded {fl['excluded']['pre_event_571']['progressor']} / {fl['excluded']['pre_event_571']['non_progressor']}; remove {fl['remove']['pre_event_571']['progressor']} / {fl['remove']['pre_event_571']['non_progressor']}; inconsistency {fl['inconsistency']['pre_event_571']['progressor']} / {fl['inconsistency']['pre_event_571']['non_progressor']}; label-review note {fl['notes_on_label_review']['pre_event_571']['progressor']} / {fl['notes_on_label_review']['pre_event_571']['non_progressor']}.")
A("\nBatch is strongly associated with status by design (several batches are single-class), but discrimination within batch is close to the overall values, and the image embedding encodes batch only weakly.")

# R32
def trow(nm, d, share):
    b = d["no_vote_vs_Bstar_training_matched"]; bb = d["no_vote_vs_B"]
    return (f"| {nm} | {d['triage_no_vote']['sensitivity']:.3f} / {d['triage_no_vote']['specificity']:.3f} | {share:.3f} | "
            f"{b['d_sens_triage_minus']['delta']:+.3f} {ci(b['d_sens_triage_minus']['ci95'])} | {b['d_sens_triage_minus']['one_sided_95_lower']:+.3f} | {b['d_spec_triage_minus']['delta']:+.3f} | "
            f"{bb['d_sens_triage_minus']['delta']:+.3f} | {bb['d_sens_triage_minus']['one_sided_95_lower']:+.3f} |")
A("\n### R32. Tissue strata and a tissue-adjusted gate\n")
A("Single model. Δ = triage − comparator.\n")
A("| Setting | Triage sensitivity / specificity | Share sequenced | Δ sens vs B\\* [95% CI] | Bound | Δ spec vs B\\* | Δ sens vs (B) | Bound |\n|---|---|---|---|---|---|---|---|")
for k, nm in [("lt256", "< 256 tissue tiles (293 samples)"), ("ge256", "≥ 256 tissue tiles (278 samples)")]:
    s = r32a["strata"][k]["claim_T"]; A(trow(nm, s, s["share_sequenced"]["per_repeat_mean"]))
s = r32b["claim_T"]; A(trow("tissue- and year-adjusted gate (all 571)", s, s["share_sequenced"]["per_repeat_mean"]))
it = r32a["interaction_lt256_minus_ge256"]
A(f"\nInteraction (< 256 minus ≥ 256): Δ sensitivity against B\\* {it['d_sens_vs_Bstar_training_matched_no_vote']['delta']:+.3f} {ci(it['d_sens_vs_Bstar_training_matched_no_vote']['ci95'])}; ΔAUROC L-IMG − L-CNV {it['d_auroc_IMG_minus_CNV']['delta']:+.3f} {ci(it['d_auroc_IMG_minus_CNV']['ci95'])}.")

# R33
A("\n### R33. Robustness tables for both headline claims\n")
A("No summary rule (decision fixed in the pre-specification). \"Within\" = one-sided 95% lower bound > −0.05; the 70% cap applies to the share sequenced.\n")
A("**Claim T: triage versus B\\* (training-matched), single model.**\n")
A("| # | Analysis | Δ sensitivity [95% CI] | One-sided bound | Within | Δ specificity | Share sequenced | Cap met |\n|---|---|---|---|---|---|---|---|")
for r in r33["claim_T_triage_vs_Bstar_training_matched_single_model"]:
    if r.get("status") != "done": A(f"| {r['item']} | {r['analysis']} | {r['status']} | | | | | |"); continue
    A(f"| {r['item']} | {r['analysis']} | {r['d_sens']:+.3f} {ci(r.get('ci95'))} | {r['one_sided_lower']:+.3f} | {yn(r['within_-0.05'])} | {f3(r.get('d_spec'))} | {r['share_sequenced']:.3f} | {yn(r['cap_0.70_met'])} |")
A("\n**Claim H: L-IMG versus L-CNV, ΔAUROC.**\n")
A("| # | Analysis | Δ AUROC | One-sided bound | Within |\n|---|---|---|---|---|")
for r in r33["claim_H_LIMG_vs_LCNV_dAUROC"]:
    if r.get("status") != "done": A(f"| {r['item']} | {r['analysis']} | {r['status']} | | |"); continue
    A(f"| {r['item']} | {r['analysis']} | {f3(r.get('delta', r.get('d_auroc')))} | {r['one_sided_lower']:+.3f} | {yn(r['within_-0.05'])} |")
s = r33["summary"]
A(f"\nClaim T: {s['claim_T']['rows_within_-0.05']} of {s['claim_T']['rows_done']} analyses within the margin; the share sequenced met the cap in {s['claim_T']['rows_cap_met']}. "
  f"Claim H: {s['claim_H']['rows_within_-0.05']} of {s['claim_H']['rows_done']} within the margin.")
mc = r33["monte_carlo_error_refit_bootstrap"]
A("\nRefit bootstrap (200 draws), Monte Carlo error of the 5th percentile:\n")
A("| Claim | 5th percentile | MC SE | MC 95% interval | Share of draws < −0.05 |\n|---|---|---|---|---|")
for k, nm in [("claim_T_d_sens_triage_minus_Bstar", "T: triage − B\\*"), ("claim_H_d_auroc_IMG_minus_CNV", "H: L-IMG − L-CNV"), ("prespecified_d_sens_triage_minus_B", "Pre-specified: triage − (B)")]:
    m = mc[k]; A(f"| {nm} | {m['one_sided_5th_percentile']:+.3f} | {m['mc_se_of_5th_percentile']:.4f} | {ci(m.get('mc_ci95_of_5th_percentile'))} | {m['share_of_draws_below_-0.05']:.3f} |")

# R34
A("\n### R34. Patient-weighted triage\n")
w = r34["weighted_single_model"]; wd = r34["weighted_delta_no_vote"]
A(f"Weights 1/nᵢ. Weighted sensitivity / specificity: triage {w['T']['sensitivity']:.3f} / {w['T']['specificity']:.3f}; (B) {w['B']['sensitivity']:.3f} / {w['B']['specificity']:.3f}; B\\* {w['B*']['sensitivity']:.3f} / {w['B*']['specificity']:.3f}; C\\* {w['C*']['sensitivity']:.3f} / {w['C*']['specificity']:.3f}. Weighted share sequenced {r34['weighted_share_sequenced']['value']:.3f}.")
A(f"Δ sensitivity vs (B) {wd['B']['d_sens_triage_minus']['delta']:+.3f} {ci(wd['B']['d_sens_triage_minus']['ci95'])}, bound {wd['B']['d_sens_triage_minus']['one_sided_95_lower']:+.3f}; vs B\\* {wd['B*']['d_sens_triage_minus']['delta']:+.3f} {ci(wd['B*']['d_sens_triage_minus']['ci95'])}, bound {wd['B*']['d_sens_triage_minus']['one_sided_95_lower']:+.3f}; vs C\\* {wd['C*']['d_sens_triage_minus']['delta']:+.3f}, bound {wd['C*']['d_sens_triage_minus']['one_sided_95_lower']:+.3f}.")

# R35
A("\n### R35. The deployable object and band stability\n")
a = r35["a_triage_CNV_middle_band"]["claim_T"]
A(f"**(a) Copy-number-only middle band, single model:** sensitivity {a['triage_no_vote']['sensitivity']:.3f}, specificity {a['triage_no_vote']['specificity']:.3f}; vs (B) {a['no_vote_vs_B']['d_sens_triage_minus']['delta']:+.3f} (bound {a['no_vote_vs_B']['d_sens_triage_minus']['one_sided_95_lower']:+.3f}); vs re-matched B\\* {a['no_vote_vs_Bstar_training_matched']['d_sens_triage_minus']['delta']:+.3f} (bound {a['no_vote_vs_Bstar_training_matched']['d_sens_triage_minus']['one_sided_95_lower']:+.3f}).")
b = r35["b_frozen_cutoffs_on_heldout"]
A(f"\n**(b) {b['label']}:** share {b['triage_no_vote']['share_sequenced']:.3f}, sensitivity {b['triage_no_vote']['sensitivity']:.3f}, specificity {b['triage_no_vote']['specificity']:.3f}; vs (B) at 0.314 {b['vs_B_frozen_0.314_no_vote']['d_sens_triage_minus']['delta']:+.3f} (bound {b['vs_B_frozen_0.314_no_vote']['d_sens_triage_minus']['one_sided_95_lower']:+.3f}, Δspec {b['vs_B_frozen_0.314_no_vote']['d_spec_triage_minus']['delta']:+.3f}); vs B\\* {b['vs_Bstar_training_matched_no_vote']['d_sens_triage_minus']['delta']:+.3f} (bound {b['vs_Bstar_training_matched_no_vote']['d_sens_triage_minus']['one_sided_95_lower']:+.3f}, Δspec {b['vs_Bstar_training_matched_no_vote']['d_spec_triage_minus']['delta']:+.3f}).")
c = r35["c_band_stability"]
A(f"\n**(c) Band stability:** {c['share_all_10_repeats_same_band']:.0%} of samples in the same band in all 10 repeats; mean {c['mean_repeats_in_modal_band']['all']:.1f} repeats in the modal band (progressors {c['mean_repeats_in_modal_band']['progressor']:.1f}, non-progressors {c['mean_repeats_in_modal_band']['non_progressor']:.1f}). The single-model call differs between repeats for {c['share_single_model_call_differs_between_repeats']['all']:.0%} of samples (modal low band {c['share_single_model_call_differs_between_repeats']['by_modal_band']['low']:.0%}, middle {c['share_single_model_call_differs_between_repeats']['by_modal_band']['mid']:.0%}, high {c['share_single_model_call_differs_between_repeats']['by_modal_band']['high']:.0%}).")

# R36
A("\n### R36. Per-sample segmentation of copy number\n")
ch = r36["checks"]
A(f"Checks: joint re-segmentation of one patient reproduced the stored package features exactly (max |Δ| {ch['joint_segmentation_reproduction']['pat_005.csv']['max_abs_tile_diff']}), so the stored features were segmented jointly per patient; the refit path reproduced stored package predictions exactly (max |Δ| {ch['refit_outer_cnv_pkg_rep_01_vs_stored_cfg_01_rep_01']['max_abs_pred_diff']}).")
fd = r36["persample"]["feature_differences_vs_joint"]
A(f"Per-sample versus joint features: per-sample correlation of raw 5-Mb tiles median {fd['per_sample_pearson_r']['raw_5mb_tiles']['median']:.3f}, of the final standardised block median {fd['per_sample_pearson_r']['final_feature_block']['median']:.3f}; cx correlation {fd['cx_raw']['pearson_r']:.3f}.\n")
A("| Copy-number features | L-CNV | L-LATE | Δ L-IMG − L-CNV (bound) | Triage − (B) Δ sens (bound) | Triage − B\\* (bound) | Triage − C\\* (bound) |\n|---|---|---|---|---|---|---|")
for k, nm in [("persample", "per-sample segmentation"), ("past", "past-only segmentation"), ("joint_package_reference", "joint (stored package features)")]:
    p = r36[k]["pre"] if "pre" in r36[k] else r36[k]
    dd = p["discrimination"]; cr = lambda key: p[key]
    A(f"| {nm} | {dd['auroc']['L-CNV']['value']:.3f} | {dd['auroc']['L-LATE']['value']:.3f} | {dd['IMG_minus_CNV']['delta']:+.3f} ({dd['IMG_minus_CNV']['one_sided_95_lower']:+.3f}) | "
      f"{cr('criterion_vs_B_no_vote')['delta_sens']:+.3f} ({cr('criterion_vs_B_no_vote')['one_sided_95_lower']:+.3f}) | {cr('criterion_vs_B*_no_vote')['delta_sens']:+.3f} ({cr('criterion_vs_B*_no_vote')['one_sided_95_lower']:+.3f}) | "
      f"{cr('criterion_vs_C*_no_vote')['delta_sens']:+.3f} ({cr('criterion_vs_C*_no_vote')['one_sided_95_lower']:+.3f}) |")
A("\nRemoving joint segmentation lowered copy-number discrimination; the H&E model was then non-inferior with a lower bound above zero, and the triage comparisons were unchanged.")

# B5
A("\n### B5. λ by deviance for the triage\n")
A("Deviance-λ outer (round-2 F4) and inner (B5) predictions, cut-offs re-chosen on them; full fields in `b5.json`. The triage row appears in the Claim T table (row 11).")

A("\n### Checks\n")
A("- Part A reproduced round-1 and round-2 values to three decimals (primary B\\* −0.016 / −0.032; (B) +0.033; R18 0.905 / 0.799; R24 −0.014 / −0.035; R15 −0.046; F3 −0.069).")
A("- Part B: joint segmentation, stored block, outer refit and all 200 inner class-error files reproduced exactly; the joint package reference reproduces 0.845 / 0.847 / 0.903.")
A("\n### Implementation notes\n")
for t in ["R31: target encoding uses the pre-event samples of the training folds; the sheet's pilot column is empty, so 'pilot' is the batch label containing 'Pilot'; the batch probe standardises inside training folds (L2, C = 1). The largest batch is single-class, so the largest mixed batch (Batch 1) is also reported in `r31.json`.",
          "R32b: inner residualisation fits on each inner set's pre-event rows.",
          "R33 rows 7, 13 and the earliest-sample row re-evaluate stored calls on the subset; selection-inclusive and between-repeat rows recompute R16 and R15 logic.",
          "R36: 'each sample on its own' = one `segmentRawData` call per sample (single-sample branch); arm subtraction and cx as the stored block's construction. Six samples hit a package edge case when no sample of a call passes QC; descriptor columns were selected by name (values unchanged). Past-only = per endoscopy, joint segmentation with the patient's samples up to that endoscopy.",
          "B5: deviance λ with the same clamp as round 2; no fold model with zero features."]:
    A(f"- {t}")

doc = open(DOC).read(); head, sep, tail = doc.partition("\n---\n")
assert "(appended after running)" in tail or "Rendered by" in tail
res = "\n## Results\n\n" + "\n".join(L) + "\n"
open(DOC, "w").write(head + sep + res)
print("rendered", len(L), "lines")

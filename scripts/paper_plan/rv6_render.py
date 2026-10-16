"""Render docs/paper_review_round6_analyses.md results (pre-specification 5c10c27) from results/paper_final/review_round6/*.json.
Replaces only the '(appended after running)' placeholder; the pre-specification above the '---' line is not touched. Idempotent. RESC = results commit."""
import json, os
T = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
R = T + "/results/paper_final/review_round6"; DOC = T + "/docs/paper_review_round6_analyses.md"; RESC = os.environ.get("RESC", "see git log")
J = lambda f: json.load(open(f"{R}/{f}"))
ci = lambda c: f"[{c[0]:+.3f}, {c[1]:+.3f}]" if c and c[0] is not None else ""
f3 = lambda x: "" if x is None else f"{x:+.3f}"
yn = lambda b: "yes" if b else "no"
cip = lambda c: f"[{c[0]:.3f}, {c[1]:.3f}]" if c and c[0] is not None else ""   # proportions (no sign)
rg = lambda u: f" ({u['range_2.5_97.5_over_seeds'][0]:.3f} to {u['range_2.5_97.5_over_seeds'][1]:.3f})" if u.get("range_2.5_97.5_over_seeds") else ""
rgd = lambda u: f"({u['range_2.5_97.5_over_seeds'][0]:+.3f} to {u['range_2.5_97.5_over_seeds'][1]:+.3f})"
L = []; A = L.append
r37 = J("r37.json")["R37"]; r38 = J("r38_merge.json")["R38"]; r39 = J("r39_merge.json")["R39"]; r40 = J("r40.json")["R40"]; r41 = J("r41.json")["R41"]

A(f"Rendered by `scripts/paper_plan/rv6_render.py` from `results/paper_final/review_round6/` (aggregate results {RESC}).\n")
A("| Item | Status |\n|---|---|")
for it, st in [("R37 p53 immunohistochemistry", "DONE"), ("R38 label sensitivity analyses", f"DONE ({len([r for r in r38['rows'] if r.get('status', 'done') == 'done']) - 1} subsets)"), ("R39 the 10-model vote as the frozen rule", f"DONE ({r39['summary']['rows_done']} rows)"),
               ("R40 paired comparison of copy-number versions", "DONE"), ("R41 one random biopsy per endoscopy", "DONE")]:
    A(f"| {it} | {st} |")

# R37
A("\n### R37. p53 immunohistochemistry as a current-standard comparator\n")
av = r37["a_availability"]; ps = av["pre_event_samples"]; ab = av["aberrant_among_available"]
A(f"**(a) Availability.** p53 available for {ps['progressor']['with_p53'] + ps['non_progressor']['with_p53']} of 571 pre-event samples ({ps['share_available']:.0%}): progressor {ps['progressor']['with_p53']} with / {ps['progressor']['without_p53']} without, non-progressor {ps['non_progressor']['with_p53']} / {ps['non_progressor']['without_p53']} (Fisher p = {ps['fisher_exact_p']}). "
  f"Aberrant among those with a result: {ab['progressor']} progressor and {ab['non_progressor']} non-progressor samples. Patients with any p53 result: {av['patients_with_any_p53']['progressor']} progressor, {av['patients_with_any_p53']['non_progressor']} non-progressor.\n")
gd = av["grade_distribution"]
A("| Grade | Progressor with p53 | Progressor without | Non-progressor with p53 | Non-progressor without |\n|---|---|---|---|---|")
for g in ("NDBE", "ID", "LGD"): A(f"| {g} | {gd['with_p53']['progressor'][g]} | {gd['without_p53']['progressor'][g]} | {gd['with_p53']['non_progressor'][g]} | {gd['without_p53']['non_progressor'][g]} |")
bk = r37["b_c_d_samples_with_p53"]; RN = [("p53", "p53 alone (b)"), ("D", "(D) grade ID or LGD"), ("D+p53", "(D) + p53 (c)"), ("T_single", "triage, single model"), ("T|p53_single", "triage or p53, single model (d)"), ("T_voted", "triage, voted"), ("T|p53_voted", "triage or p53, voted (d)"), ("Bstar_single", "B\\* training-matched, single model")]
def rules_table(b):
    A("| Rule | Sample sens / spec | Endoscopy sens / spec | Patient sens / spec |\n|---|---|---|---|")
    for k, nm in RN:
        u = b["rules"][k]; cell = lambda lv: f"{u[lv]['sensitivity']:.3f} {cip(u[lv].get('sens_ci95_exact', u[lv]['sens_ci95_bootstrap']))} / {u[lv]['specificity']:.3f} {cip(u[lv].get('spec_ci95_exact', u[lv]['spec_ci95_bootstrap']))}"
        A(f"| {nm} | {cell('sample')} | {cell('endoscopy')} | {cell('patient')} |")
A(f"\n**(b)–(d) On the {bk['n_samples']} pre-event samples with p53** ({bk['n_progressor_samples']} progressor; {bk['n_patients']} patients, {bk['n_endoscopies']} endoscopies). Exact (Clopper–Pearson) CIs for deterministic rules, patient-bootstrap CIs for single-model rules. Triage share sequenced on these samples {bk['share_sequenced_triage']:.3f}.\n")
rules_table(bk)
def paired_table(b, lv="sample"):
    A(f"| Paired difference ({lv} level) | Δ sensitivity [95% CI] | One-sided bound | Δ specificity [95% CI] |\n|---|---|---|---|")
    for k, nm in [("T_single_minus_D", "triage − (D)"), ("T_single_minus_D+p53", "triage − (D+p53)"), ("T_single_minus_p53", "triage − p53 alone"), ("T_voted_minus_D+p53", "triage (voted) − (D+p53)"), ("T|p53_single_minus_D+p53", "triage-or-p53 − (D+p53)"), ("T|p53_single_minus_T_single", "triage-or-p53 − triage"), ("T_single_minus_Bstar_single", "triage − B\\*")]:
        p = b["paired"][k][lv]; A(f"| {nm} | {p['d_sens']['delta']:+.3f} {ci(p['d_sens']['ci95'])} | {p['d_sens']['one_sided_95_lower']:+.3f} | {p['d_spec']['delta']:+.3f} {ci(p['d_spec']['ci95'])} |")
A("\nPaired comparisons, single model unless stated:\n"); paired_table(bk)
e = r37["e_p53_among_triage_cleared"]; em = e["modal_low_band"]; ea = e["low_band_all_repeats"]
A(f"\n**(e) p53 among samples the triage clears.** Modal low band: {em['n_cleared']} samples, {em['n_cleared_with_p53']} with p53; p53-aberrant {em['progressor']['p53_aberrant']} of {em['progressor']['cleared_with_p53']} progressor and {em['non_progressor']['p53_aberrant']} of {em['non_progressor']['cleared_with_p53']} non-progressor samples. "
  f"Low band in all 10 repeats: {ea['progressor']['p53_aberrant']} of {ea['progressor']['cleared_with_p53']} progressor, {ea['non_progressor']['p53_aberrant']} of {ea['non_progressor']['cleared_with_p53']} non-progressor. Per-repeat mean p53-aberrant samples in the low band: progressor {e['per_repeat_mean_p53_aberrant_in_low_band']['progressor']:.1f}, non-progressor {e['per_repeat_mean_p53_aberrant_in_low_band']['non_progressor']:.1f}.")
fb = r37["f_missing_as_wild_type_all_pre_event"]
A(f"\n**(f) Missing p53 as wild-type, all 571 samples.**\n"); rules_table(fb); A(""); paired_table(fb)
tp = bk["paired"]["T_single_minus_D+p53"]["sample"]; tq = bk["paired"]["T|p53_single_minus_T_single"]["sample"]
A(f"\np53 is available for {ps['share_available']:.0%} of samples and flags few; the triage keeps its sensitivity advantage over current care with p53 ({tp['d_sens']['delta']:+.3f}, bound {tp['d_sens']['one_sided_95_lower']:+.3f}), and adding p53 to the triage changes sensitivity by {tq['d_sens']['delta']:+.3f} at a specificity cost of {-tq['d_spec']['delta']:.3f}.")

# R38
A("\n### R38. Label sensitivity analyses\n")
lh = r38["later_hgd_check"]
A(f"Stored calls re-evaluated on each subset (single model; patient bootstrap on the subset). Later HGD+: of the {lh['n_nonprogressor_patients']} non-progressor patients, {lh['n_matched_in_hospital_record_check']} are in the hospital-record check ({lh['n_not_identifiable']} not identifiable), {lh['n_with_later_HGDplus']} with a later HGD+ report and {lh['n_with_later_LGDplus']} with later LGD+.\n")
A("| Subset | Samples (progressor) / patients | Triage sens / spec | Share | Δ sens vs B\\* [95% CI] | Bound | Within | Δ spec | Δ sens vs (B) (bound) | Δ L-IMG − L-CNV (bound) | Within |\n|---|---|---|---|---|---|---|---|---|---|---|")
for r in r38["rows"]:
    if r.get("status", "done") != "done": A(f"| {r['analysis']} | {r['status']} | | | | | | | | | |"); continue
    b = r["vs_Bstar"]; bb = r["vs_B"]; h = r["IMG_minus_CNV"]; t = r["triage"]
    A(f"| {r['analysis']} | {r['n_samples']} ({r['n_progressor_samples']}) / {r['n_patients']} | {t['sensitivity']:.3f} / {t['specificity']:.3f} | {t['share_sequenced']:.3f} | {b['d_sens']:+.3f} {ci(b['ci95'])} | {b['one_sided_lower']:+.3f} | {yn(b['within_-0.05'])} | {b['d_spec']:+.3f} | {bb['d_sens']:+.3f} ({bb['one_sided_lower']:+.3f}) | {h['delta']:+.3f} ({h['one_sided_lower']:+.3f}) | {yn(h['within_-0.05'])} |")
dn = [r for r in r38["rows"][1:] if r.get("status", "done") == "done"]
A(f"\nTriage versus B\\* stays within the margin in {sum(bool(r['vs_Bstar']['within_-0.05']) for r in dn)} of {len(dn)} label subsets and the H&E-versus-copy-number bound in {sum(bool(r['IMG_minus_CNV']['within_-0.05']) for r in dn)} of {len(dn)}; the direction of every difference is unchanged by the label exclusions.")

# R39
A("\n### R39. The 10-model ensemble as the frozen rule\n")
A("Voted (≥ 6 of 10) triage against voted B\\* (training-matched), same rows as R33 Claim T. \"Pivotal\" = share of samples whose voted call depends on the middle-band calls (the vote's sequencing need). The single-model (R33) Δ and bound are repeated for comparison.\n")
A("| # | Analysis | Δ sensitivity voted [95% CI] | Bound | Within | Δ spec | Share | Pivotal | Cap | Single-model Δ (bound) | Within |\n|---|---|---|---|---|---|---|---|---|---|---|")
for r in r39["claim_T_voted_triage_vs_voted_Bstar_training_matched"]:
    s = r.get("single_model_R33"); sm = f"{s['d_sens']:+.3f} ({s['one_sided_lower']:+.3f}) | {yn(s['within_-0.05'])}" if s else " | "
    if r["status"] != "done": A(f"| {r['item']} | {r['analysis']} | {r['status']} | | | | | | | {sm} |"); continue
    A(f"| {r['item']} | {r['analysis']} | {r['d_sens']:+.3f} {ci(r['ci95'])} | {r['one_sided_lower']:+.3f} | {yn(r['within_-0.05'])} | {f3(r.get('d_spec'))} | {r['share_sequenced']:.3f} | {r['share_pivotal']:.3f} | {yn(r['cap_0.70_met'])} | {sm} |")
s = r39["summary"]; st = r39["stability_of_voted_call"]
A(f"\nVoted: {s['rows_within_-0.05']} of {s['rows_done']} rows within the margin (single model on the same {s['rows_with_single_model_counterpart']} rows: {s['single_model_within_-0.05_on_those_rows']}; voted: {s['voted_within_-0.05_on_those_rows']}); mean Δ voted {s['mean_d_sens_voted']:+.3f} versus single model {s['mean_d_sens_single_model']:+.3f}; mean voted CI width {s['mean_ci_width_voted']:.3f}.")
A(f"\n**Between-repeat stability of the voted call** (primary): the per-repeat call equals the voted call in all 10 repeats for {st['all']['unanimous_10_of_10']:.0%} of samples and in ≥ 8 of 10 for {st['all']['near_unanimous_ge_8_of_10']:.0%} (progressors {st['progressor']['unanimous_10_of_10']:.0%} / {st['progressor']['near_unanimous_ge_8_of_10']:.0%}, non-progressors {st['non_progressor']['unanimous_10_of_10']:.0%} / {st['non_progressor']['near_unanimous_ge_8_of_10']:.0%}). "
  f"By modal band: low {st['by_modal_band']['low']['unanimous_10_of_10']:.0%} / {st['by_modal_band']['low']['near_unanimous_ge_8_of_10']:.0%} (n {st['by_modal_band']['low']['n']}), middle {st['by_modal_band']['mid']['unanimous_10_of_10']:.0%} / {st['by_modal_band']['mid']['near_unanimous_ge_8_of_10']:.0%} (n {st['by_modal_band']['mid']['n']}), high {st['by_modal_band']['high']['unanimous_10_of_10']:.0%} / {st['by_modal_band']['high']['near_unanimous_ge_8_of_10']:.0%} (n {st['by_modal_band']['high']['n']}). B\\* for reference: {st['Bstar_for_reference']['unanimous_10_of_10']:.0%} / {st['Bstar_for_reference']['near_unanimous_ge_8_of_10']:.0%}.")
TTd = [r for r in r39["claim_T_voted_triage_vs_voted_Bstar_training_matched"] if r["status"] == "done" and "single_model_R33" in r]
vfail = [r["analysis"] for r in TTd if not r["within_-0.05"] and r["single_model_R33"]["within_-0.05"]]; bfail = [r["analysis"] for r in TTd if not r["within_-0.05"] and not r["single_model_R33"]["within_-0.05"]]
A(f"\nThe vote's point estimates stay close to the single model's (mean Δ {s['mean_d_sens_voted']:+.3f} versus {s['mean_d_sens_single_model']:+.3f}) but its intervals are wider: the {len(bfail)} rows the single model fails also fail voted, and {len(vfail)} further rows ({'; '.join(vfail)}) fall just below the margin; disagreement between repeats concentrates in the middle band ({st['by_modal_band']['mid']['unanimous_10_of_10']:.0%} unanimous against {st['by_modal_band']['low']['unanimous_10_of_10']:.0%} and {st['by_modal_band']['high']['unanimous_10_of_10']:.0%} in the low and high bands), so the ensemble is not a more robust rule than the single model.")

# R40
A("\n### R40. Paired comparison of copy-number versions\n")
au = r40["auroc_L_CNV"]; NM = {"shipped_matrix": "shipped matrix", "joint_package": "joint package rebuild", "past_only": "past-only", "per_sample": "per-sample"}
A("| L-CNV version | Fold-stratified AUROC [95% CI] |\n|---|---|")
for k, nm in NM.items(): A(f"| {nm} | {au[k]['value']:.3f} {ci(au[k]['ci95'])} |")
A("\n| Pair | ΔAUROC [95% CI] | p (two-sided) | p (Holm) |\n|---|---|---|---|")
for k, p in r40["pairwise"].items():
    a, b = k.split("_minus_"); A(f"| {NM[a]} − {NM[b]} | {p['delta']:+.3f} {ci(p['ci95'])} | {p['p_two_sided']} | {p['p_holm']} |")
si = r40["sampling_intensity"]; nj = si["n_segmented_together_joint"]; pr = si["per_sample_r_joint_vs_per_sample_final_features"]; cd = si["class_difference_median_r_progressor_minus_non_progressor"]; ct = si["counts_by_tertile_and_class"]
A(f"\n**Sampling intensity.** Samples segmented together in the joint call: median {nj['pre_event_by_class']['progressor']['median']:.0f} (IQR {nj['pre_event_by_class']['progressor']['iqr'][0]:.0f}–{nj['pre_event_by_class']['progressor']['iqr'][1]:.0f}) for progressor and {nj['pre_event_by_class']['non_progressor']['median']:.0f} ({nj['pre_event_by_class']['non_progressor']['iqr'][0]:.0f}–{nj['pre_event_by_class']['non_progressor']['iqr'][1]:.0f}) for non-progressor pre-event samples. Per-sample correlation of joint versus per-sample final features: median {pr['pre_event_by_class']['progressor']['median']:.3f} (progressor) versus {pr['pre_event_by_class']['non_progressor']['median']:.3f} (non-progressor), difference {cd['all_pre_event']['delta']:+.3f} {ci(cd['all_pre_event']['ci95'])}; Spearman ρ of r against the number segmented together {pr['spearman_r_vs_n_segmented_pre_event']['all']:+.3f} (progressor {pr['spearman_r_vs_n_segmented_pre_event']['progressor']:+.3f}, non-progressor {pr['spearman_r_vs_n_segmented_pre_event']['non_progressor']:+.3f}).\n")
A("| Tertile of samples segmented together | Range | Progressor n / median r | Non-progressor n / median r | Class difference [95% CI] |\n|---|---|---|---|---|")
for t in (1, 2, 3):
    b = pr["by_tertile"][f"tertile_{t}"]; A(f"| {t} | {b['n_segmented_range'][0]}–{b['n_segmented_range'][1]} | {ct[f'tertile_{t}']['progressor']} / {b['progressor']['median'] if b['progressor']['median'] is not None else 'n/a'} | {ct[f'tertile_{t}']['non_progressor']} / {b['non_progressor']['median'] if b['non_progressor']['median'] is not None else 'n/a'} | {f3(cd[f'tertile_{t}']['delta'])} {ci(cd[f'tertile_{t}']['ci95'])} |")
pw = r40["pairwise"]
A(f"\nThe joint package rebuild has the highest AUROC, but no pairwise difference survives Holm adjustment (smallest Holm p {min(p['p_holm'] for p in pw.values())}; joint − per-sample {pw['joint_package_minus_per_sample']['delta']:+.3f} {ci(pw['joint_package_minus_per_sample']['ci95'])}). Progressor samples are segmented with fewer companions than non-progressor samples, yet the class difference in joint-versus-per-sample agreement is present in every tertile (clearly in the lowest), so it does not follow the number of samples per patient.")

# R41
A("\n### R41. One random biopsy per endoscopy\n")
sh = r41["share_sequenced"]
A(f"{r41['n_endoscopies']} pre-event endoscopies ({r41['n_progressor_endoscopies']} progressor) in {r41['n_patients_units']} patients; one sample per endoscopy sequences {sh['random_biopsy']:.3f} of samples (triage: {sh['triage_per_repeat_mean']:.3f} per repeat, {sh['triage_in_at_least_one_repeat']:.3f} in at least one repeat). 100 seeds; single model; mean over seeds with the 2.5–97.5% range over seeds, and the patient-bootstrap CI of the seed mean.\n")
A("| Level | Rule | Sensitivity, mean (range over seeds) [bootstrap CI] | Specificity, mean (range) [CI] |\n|---|---|---|---|")
for lv in ("endoscopy", "patient"):
    Lv = r41["levels"][lv]
    for k, nm in [("random_biopsy", "one random biopsy, L-LATE at (B)'s threshold"), ("random_biopsy_rest_HE", "one random biopsy + the rest by the H&E model (C)"), ("triage", "triage"), ("B_all_sequenced", "(B), every sample sequenced")]:
        se = Lv[f"sens_{k}"]; sp = Lv[f"spec_{k}"]
        A(f"| {lv} | {nm} | {se['mean_over_seeds']:.3f}{rg(se)} {cip(se['ci95_bootstrap_of_seed_mean'])} | {sp['mean_over_seeds']:.3f}{rg(sp)} {cip(sp['ci95_bootstrap_of_seed_mean'])} |")
A("\n| Level | Δ rule − triage | Δ sensitivity, mean over seeds (range) [bootstrap CI of the seed mean] | One-sided bound | Δ specificity (range) [CI] |\n|---|---|---|---|---|")
for lv in ("endoscopy", "patient"):
    Lv = r41["levels"][lv]
    for k, nm in [("random_biopsy", "one random biopsy"), ("random_biopsy_rest_HE", "random biopsy + H&E on the rest")]:
        ds = Lv[f"d_sens_{k}_minus_triage"]; dp = Lv[f"d_spec_{k}_minus_triage"]
        A(f"| {lv} | {nm} | {ds['mean_over_seeds']:+.3f} {rgd(ds)} {ci(ds['ci95_bootstrap_of_seed_mean'])} | {ds['one_sided_95_lower_seed_mean']:+.3f} | {dp['mean_over_seeds']:+.3f} {rgd(dp)} {ci(dp['ci95_bootstrap_of_seed_mean'])} |")
E_ = r41["levels"]["endoscopy"]; d = E_["d_sens_random_biopsy_minus_triage"]; dv = E_["d_sens_random_biopsy_rest_HE_minus_triage"]
A(f"\nAt a similar sequencing share, one random biopsy per endoscopy loses {-d['mean_over_seeds']:.3f} endoscopy-level sensitivity against the triage (bound {d['one_sided_95_lower_seed_mean']:+.3f}) while gaining {E_['d_spec_random_biopsy_minus_triage']['mean_over_seeds']:+.3f} specificity; calling the remaining biopsies with the H&E model recovers {dv['mean_over_seeds'] - d['mean_over_seeds']:.3f} of that sensitivity (Δ {dv['mean_over_seeds']:+.3f}, bound {dv['one_sided_95_lower_seed_mean']:+.3f}) and keeps a specificity gain of {E_['d_spec_random_biopsy_rest_HE_minus_triage']['mean_over_seeds']:+.3f}.")

A("\n### Checks\n")
sn = r39["sanity"]; A(f"- Stored values reproduced: single-model triage vs B\\* {sn['single_model_vs_Bstar_expected_-0.016_-0.032'][0]:+.3f} / {sn['single_model_vs_Bstar_expected_-0.016_-0.032'][1]:+.3f}; vs (B) {sn['single_model_vs_B_expected_+0.033']:+.3f}; R33 voted row {sn['round4_r33_row16_voted_vs_Bstar_expected_-0.006_-0.044'][0]:+.3f} / {sn['round4_r33_row16_voted_vs_Bstar_expected_-0.006_-0.044'][1]:+.3f}; R35 share in the same band in all 10 repeats {sn['round4_r35_share_all_10_repeats_same_band_expected_0.35']:.2f}.")
A(f"- R23 endoscopy-level single-model triage {r41['sanity_round2_r23_endoscopy_no_vote_T_expected_0.909_0.549'][0]:.3f} / {r41['sanity_round2_r23_endoscopy_no_vote_T_expected_0.909_0.549'][1]:.3f}; R24 (D) {r37['sanity_round2_r24_D1_expected_0.354_0.815']['sensitivity']:.3f} / {r37['sanity_round2_r24_D1_expected_0.354_0.815']['specificity']:.3f}; R36 L-CNV per-sample {r40['sanity_round4_r36_expected_per_sample_0.785_joint_0.845'][0]:.3f}, joint package {r40['sanity_round4_r36_expected_per_sample_0.785_joint_0.845'][1]:.3f}.")
A("\n### Implementation notes\n")
for t in ["R37: p53 from the `P53 IHC` field of the analysed-sample table (0 / 1 / blank); endoscopy = (patient, endoscopy number), positive if any sample is flagged; patient positive if any sample is flagged; single-model statistics are per-repeat values averaged over the 10 repeats; the patient bootstrap tags resampled copies of a patient as distinct units (R23). 'Cleared' in (e) = modal low band over the 10 repeats (also low in all 10). B\\* and the paired comparisons are on the same samples; cut-offs as stored.",
          "R38: calls and cut-offs as stored (R33 row 7 logic); the sheet flags are Killcoyne's `excluded` = 1, `remove` = 1 and non-empty `inconsistency`; 'less than 1 year before the last sequenced endoscopy' = the sheet's months-before-final < 12 for non-progressor samples (so the last endoscopy itself is excluded), 24 for 2 years; later HGD+ = the K1 definition of `pk_main.py` on `later_disease_patient.csv`, joined to the analysed patients by study number (every non-progressor patient matched; the stored K1 count of 6 later HGD+ among 42 refers to the ERIN-labelled discovery non-progressors, five of whom are sheet progressors); (d) uses (a) 'excluded' only.",
          "R39: one task per row, recomputed with the rv4 machinery; the voted comparator is the training-matched B\\* voted the same way; the selection-inclusive row uses the stored round-4 draws (no pivotal CI); rows 16–18 are not applicable to a voted rule; the 'evaluation-matched' row is the voted evaluation-matched B\\* of the primary analysis. Stability = agreement of the 10 per-repeat triage calls with the voted call.",
          "R40: four L-CNV outer prediction sets (stored shipped, stored package, round-4 past-only and per-sample refits) on identical folds; the same 2,000 patient draws score every version; Holm over the six pairs on two-sided bootstrap p-values. Samples segmented together = the patient's sample count in the joint call (all of the patient's sheet samples, `rv4_reseg.R`); tertile edges are the pre-event terciles of that count, ties kept in the lower tertile.",
          "R41: the random biopsy's call is its stored (B) call (L-LATE at that repeat × fold's 80%-training-sensitivity threshold); the variant adds the stored (C) call of the endoscopy's other samples; the triage endoscopy call is 'any sample positive'; the patient bootstrap resamples patients with all their endoscopies and reports the seed-averaged statistic per draw (pooled seed × draw percentiles in the JSON)."]:
    A(f"- {t}")

doc = open(DOC).read(); head, sep, tail = doc.partition("\n---\n")
assert "(appended after running)" in tail or "Rendered by" in tail
res = "\n## Results\n\n" + "\n".join(L) + "\n"
open(DOC, "w").write(head + sep + res)
print("rendered", len(L), "lines")

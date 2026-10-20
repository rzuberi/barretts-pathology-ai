"""Render docs/paper_review_round7d_sensitivity_first.md results (pre-specification 832d184) from results/paper_final/review_round7d/*.json.
Appends below the document's final '---' line; the pre-specification above it is not touched. Idempotent (replaces anything after that line).
Counts of 1-4 patients, samples or endoscopies are shown as '<5'. RESC = results commit."""
import json, os
T = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
R = T + "/results/paper_final/review_round7d"; DOC = T + "/docs/paper_review_round7d_sensitivity_first.md"; RESC = os.environ.get("RESC", "see git log")
J = lambda f: json.load(open(f"{R}/{f}"))
ci = lambda c: f"[{c[0]:+.3f}, {c[1]:+.3f}]" if c and c[0] is not None else ""
cip = lambda c: f"[{c[0]:.3f}, {c[1]:.3f}]" if c and c[0] is not None else ""
yn = lambda b: "yes" if b else "no"
f3_ = lambda v: "n/a" if v is None else f"{v:.3f}"
pc_ = lambda v: "n/a" if v is None else f"{100 * v:.1f}%"
def c5(k):
    k = int(k); return "<5" if 0 < k < 5 else str(k)
def gbp(v): return "n/a" if v is None else f"£{v:,.0f}"
L = []; A = L.append
PQ = J("prereq.json")["prereq"]; PR = J("row_primary.json")["R7D"]; CV = J("curve.json")["curve"]; MG = J("merge.json")["R7D"]; F3 = J("f3.json")["R7D"]; ND = J("row_ndbe.json")["R7D"]
RL = {"V0": "**V0** (clear on the slide, confirm top sample by L-CNV)", "V1": "V1 (confirm by L-LATE)", "V2": "V2 (sequence all samples of an uncleared endoscopy)", "V4": "V4 (high band flagged unsequenced)",
      "V0ovr": "V0+override (ID / LGD / p53-aberrant → guideline)", "R49": "R49 gate (round 7, reference)"}
CL = {"H": "(H) H&E alone", "SLATE": "(S-LATE) sequence everything, L-LATE", "SCNV": "(S-CNV) sequence everything, L-CNV", "R1": "(R1) one random sample, L-LATE (seeds 0-99)"}
TLB = {"V0": "V0", "V0ovr": "V0+override", "H": "(H) H&E alone", "SLATE": "(S-LATE) sequence everything", "R49": "R49 gate", "GP53": "grade + p53 (current care)"}
pc = PR["primary_contrast"]; ks = PR["key_secondary"]; v0 = PR["rules"]["V0"]

A("\n## Results\n")
A(f"Rendered by `scripts/paper_plan/rv7d_render.py` from `results/paper_final/review_round7d/` (aggregate results {RESC}); script `scripts/paper_plan/rv7d_sensfirst.py`. No model was refitted; stored outer and inner out-of-fold predictions were re-scored. "
  "Unit = pre-event endoscopy unless stated. Single model = per-repeat values averaged over the 10 repeats. Intervals: **re-derived** = patient bootstrap, 2,000 draws (`RandomState(0)`), every threshold re-derived in each draw from the resampled training patients' inner predictions (the primary interval); "
  "**fixed** = the same draws with thresholds and calls fixed (rounds 4–7). Comparators are evaluation-matched: specificity read at the rule's per-repeat endoscopy sensitivity on the comparator's empirical ROC (linear interpolation), re-matched in each draw.\n")

# ---------------------------------------------------------------- prerequisites
A("### Prerequisites\n")
A("| Stored arrays | Outer (10 repeats, finite) | Inner (repeat × fold sets, img / cnv) | Repeat-folds with missing inner values | Inner rows in their own outer fold | Status |\n|---|---|---|---|---|---|")
for k, v in PQ.items():
    if "outer_finite" in v: A(f"| {k} | {yn(v['outer_finite'] and v['outer_repeats'] == 10)} | {v['inner_repeat_folds_img']} / {v['inner_repeat_folds_cnv']} | {v['repeat_folds_with_missing_inner_values']} | {v['inner_rows_in_own_outer_fold']} | {'available' if v['available'] else 'not available'} |")
f3p = PQ["refit bootstrap draws (review_round2/f3)"]; p53 = PQ["p53_field"]
A(f"| refit bootstrap draws (review_round2/f3) | {f3p['draws_with_outer_and_inner_cnv_and_img']} of {f3p['of']} draws with outer and inner L-CNV and L-IMG | | | | {'available' if f3p['available'] else 'not available'} |")
A(f"| R41 random-one seeds | regenerated (seeds 0-99, R41 draw order) | | | | available |")
A(f"\nIn training, the endoscopy score, top sample and cleared status come from inner out-of-fold L-IMG and the copy-number confirmation from inner out-of-fold L-CNV (L-LATE for V1). p53 (analysed-sample table, `P53 IHC`, as R37): available for {p53['available_pre_event_samples']} pre-event samples, aberrant in {c5(p53['aberrant_pre_event_samples'])}; identical to the field of the endoscopy table: {yn(p53['equal_to_endoscopy_table_field'])}. Nothing was 'not available'; nothing was refitted.\n" if all(v.get("available", True) for v in PQ.values() if isinstance(v, dict)) else "\nRows whose arrays are missing are 'not available'; nothing was refitted.\n")

# ---------------------------------------------------------------- primary
A("### Primary contrast (fixed sequence, step 1)\n")
A(f"{PR['n_endoscopies']} pre-event endoscopies ({PR['n_progressor_endoscopies']} progressor), {PR['n_samples']} samples, {PR['n_patients']} patients ({PR['n_progressor_patients']} progressor).\n")
A("| Rule | Endoscopy sens [re-derived] [fixed] | Endoscopy spec [re-derived] [fixed] | Samples sequenced | Endoscopies sequenced |\n|---|---|---|---|---|")
A(f"| {RL['V0']} | {v0['endoscopy_sensitivity']:.3f} {cip(v0['sens_ci95_rederived'])} {cip(v0['sens_ci95_fixed'])} | {v0['endoscopy_specificity']:.3f} {cip(v0['spec_ci95_rederived'])} {cip(v0['spec_ci95_fixed'])} | {pc_(v0['share_samples_sequenced'])} | {pc_(v0['share_endoscopies_sequenced'])} |")
c = PR["comparators"]["V0"]["H"]
A(f"| (H) H&E alone at V0's sensitivity | (matched) | {c['specificity']:.3f} {cip(c['spec_ci95_rederived'])} {cip(c['spec_ci95_fixed'])} | 0.0% | 0.0% |")
d = PR["contrasts"]["V0_minus_H"]
A(f"\n**Δ endoscopy specificity, V0 − (H): {d['delta']:+.3f}**, 95% CI {ci(d['rederived']['ci95'])} (re-derived, primary interval); {ci(d['fixed']['ci95'])} (fixed). Realised V0 endoscopy sensitivity {pc['v0_endoscopy_sensitivity']:.3f}. Per repeat: {', '.join(f'{x:+.3f}' for x in d['per_repeat'])}.\n")
A(f"- Lower 95% limit > 0: {yn(pc['lower_limit_above_0'])}. Sensitivity ≥ 0.90: {yn(pc['sensitivity_condition_met'])}. Clinically relevant (Δ ≥ 0.05): {yn(pc['clinically_relevant_delta_ge_0.05'])}.")
if pc["success"]:
    A("\n**Verdict: success.** The lower 95% limit of Δ endoscopy specificity (V0 − (H)) is above 0 and V0's realised endoscopy sensitivity is at least 0.90. The study is hypothesis-generating (post hoc reframing); rule V0 and its threshold procedure are frozen for external testing.")
else:
    A("\n**Verdict: failure.** Limited sequencing did not remove more false positives than raising the H&E threshold. No variant or curve point is promoted, and the reframing is reported as tested and not supported.")

A("\n### Key secondary contrast (fixed sequence, step 2): V0 − (S-LATE)\n")
A(f"Status: **{ks['status']}**.")
A(f"\nΔ endoscopy specificity V0 − (S-LATE) {ks['delta']:+.3f}; one-sided 95% lower bound {ks['one_sided_95_lower_rederived']:+.3f} (re-derived), {ks['one_sided_95_lower_fixed']:+.3f} (fixed); two-sided 95% CI {ci(ks['ci95_rederived'])} (re-derived), {ci(ks['ci95_fixed'])} (fixed). Margin −0.05 (about 50 extra flagged non-progressor endoscopies per 1,000; about £23.6k at £471 each).")
if ks["status"] == "tested": A(f"\nNon-inferior to sequencing everything at −0.05 (re-derived bound > −0.05): **{yn(ks['non_inferior_at_-0.05_rederived'])}**.")
else: A("\nComputed for completeness only; because the primary contrast failed, it is not tested and supports no claim.")

# ---------------------------------------------------------------- comparators
A("\n### Comparators at V0's sensitivity (all evaluation-matched)\n")
A("| Comparator | Endoscopy spec at V0's sensitivity [re-derived] | Samples sequenced | V0 − comparator, Δ spec [re-derived] | [fixed] |\n|---|---|---|---|---|")
shc = {"H": "0.0%", "SLATE": "100%", "SCNV": "100%", "R1": pc_(PR["n_endoscopies"] / PR["n_samples"])}
for k_, lab in CL.items():
    c = PR["comparators"]["V0"][k_]; d = PR["contrasts"][f"V0_minus_{k_}"]
    A(f"| {lab} | {c['specificity']:.3f} {cip(c['spec_ci95_rederived'])} | {shc[k_]} | {d['delta']:+.3f} {ci(d['rederived']['ci95'])} | {ci(d['fixed']['ci95'])} |")
cc = PR["current_care"]; cv = CV["current_care"]
A(f"\nCurrent care (deterministic; exact intervals): grade ID/LGD, endoscopy sensitivity {cc['grade_ID_LGD_all']['endoscopy_sensitivity']:.3f} {cip(cc['grade_ID_LGD_all']['sens_cp95'])}, specificity {cc['grade_ID_LGD_all']['endoscopy_specificity']:.3f} {cip(cc['grade_ID_LGD_all']['spec_cp95'])}; "
  f"grade ID/LGD or p53 aberrant on the p53 subset ({cc['grade_or_p53_on_p53_subset']['endoscopies']} endoscopies, {cc['grade_or_p53_on_p53_subset']['progressor_endoscopies']} progressor): {cc['grade_or_p53_on_p53_subset']['endoscopy_sensitivity']:.3f} / {cc['grade_or_p53_on_p53_subset']['endoscopy_specificity']:.3f}. "
  f"The V0 family at grade's specificity reaches sensitivity {f3_(cv['grade_ID_LGD']['V0_family_sensitivity_at_this_specificity'])}, and at grade's sensitivity specificity {f3_(cv['grade_ID_LGD']['V0_family_specificity_at_this_sensitivity'])}; "
  f"on the p53 subset against grade + p53: {f3_(cv['grade_or_p53_on_p53_subset']['V0_family_sensitivity_at_this_specificity'])} and {f3_(cv['grade_or_p53_on_p53_subset']['V0_family_specificity_at_this_sensitivity'])} (n/a = outside the family's range: the nine V0 grid points span sensitivity "
  f"{min(g['endoscopy_sensitivity'] for g in CV['grid']):.3f}–{max(g['endoscopy_sensitivity'] for g in CV['grid']):.3f} and specificity {min(g['endoscopy_specificity'] for g in CV['grid']):.3f}–{max(g['endoscopy_specificity'] for g in CV['grid']):.3f}; on the p53 subset "
  f"{min(g['endoscopy_sensitivity'] for g in CV['grid_p53_subset']):.3f}–{max(g['endoscopy_sensitivity'] for g in CV['grid_p53_subset']):.3f} and {min(g['endoscopy_specificity'] for g in CV['grid_p53_subset']):.3f}–{max(g['endoscopy_specificity'] for g in CV['grid_p53_subset']):.3f}).")

# ---------------------------------------------------------------- override, NDBE
A("\n### V0+override\n")
for nm, X, lab in (("all pre-event endoscopies", PR, "primary unit"), ("NDBE endoscopies", ND, "NDBE unit")):
    o = X["rules"]["V0ovr"]; d = X["contrasts"]["V0ovr_minus_H"]; s = X["contrasts"]["V0ovr_minus_SLATE"]
    A(f"- {nm}: sensitivity {o['endoscopy_sensitivity']:.3f} {cip(o['sens_ci95_rederived'])}, specificity {o['endoscopy_specificity']:.3f} {cip(o['spec_ci95_rederived'])}, samples sequenced {pc_(o['share_samples_sequenced'])}, endoscopies sequenced {pc_(o['share_endoscopies_sequenced'])}; "
      f"Δ spec vs (H) at its sensitivity {d['delta']:+.3f} {ci(d['rederived']['ci95'])} (fixed {ci(d['fixed']['ci95'])}); vs (S-LATE) {s['delta']:+.3f} {ci(s['rederived']['ci95'])}.")
A(f"\nOverride endoscopies (any ID / LGD / p53-aberrant sample): {c5(cc['override_endoscopies'])} of {PR['n_endoscopies']} ({c5(cc['override_progressor_endoscopies'])} progressor). The rule acts on the rest with V0's thresholds; p53 missing counts as wild-type.")

A("\n### NDBE endoscopies (key secondary unit 1)\n")
v = ND["rules"]["V0"]; d = ND["contrasts"]["V0_minus_H"]; s = ND["contrasts"]["V0_minus_SLATE"]
A(f"{ND['n_endoscopies']} endoscopies whose samples are all NDBE ({ND['n_progressor_endoscopies']} progressor; {ND['n_patients']} patients, {ND['n_progressor_patients']} progressor). V0 (thresholds as the primary) sensitivity {v['endoscopy_sensitivity']:.3f} {cip(v['sens_ci95_rederived'])}, specificity {v['endoscopy_specificity']:.3f} {cip(v['spec_ci95_rederived'])}; samples sequenced {pc_(v['share_samples_sequenced'])}, endoscopies {pc_(v['share_endoscopies_sequenced'])}.\n")
A(f"- Δ spec V0 − (H): {d['delta']:+.3f} {ci(d['rederived']['ci95'])} (re-derived), {ci(d['fixed']['ci95'])} (fixed); (H) spec {d['comparator_specificity']:.3f}.")
A(f"- Δ spec V0 − (S-LATE): {s['delta']:+.3f} {ci(s['rederived']['ci95'])}; one-sided bound {s['rederived']['one_sided_95_lower']:+.3f} (re-derived), {s['fixed']['one_sided_95_lower']:+.3f} (fixed).")

# ---------------------------------------------------------------- variants
A("\n### Variants (all reported; none promoted)\n")
A("| Rule | Endo sens [re-derived] | Endo spec [re-derived] | Samples / endoscopies sequenced | (H) spec at its sens | Δ spec vs (H) [re-derived] [fixed] | Δ spec vs (S-LATE) [re-derived] |\n|---|---|---|---|---|---|---|")
for nm in ("V0", "V1", "V2", "V4", "V0ovr", "R49"):
    o = PR["rules"][nm]; d = PR["contrasts"][f"{nm}_minus_H"]; s = PR["contrasts"][f"{nm}_minus_SLATE"]
    A(f"| {RL[nm]} | {o['endoscopy_sensitivity']:.3f} {cip(o['sens_ci95_rederived'])} | {o['endoscopy_specificity']:.3f} {cip(o['spec_ci95_rederived'])} | {pc_(o['share_samples_sequenced'])} / {pc_(o['share_endoscopies_sequenced'])} | {d['comparator_specificity']:.3f} | {d['delta']:+.3f} {ci(d['rederived']['ci95'])} {ci(d['fixed']['ci95'])} | {s['delta']:+.3f} {ci(s['rederived']['ci95'])} |")
p = PR["rules"]["Vp53"]; q = PR["rules"]["V0p53"]; d = PR["contrasts"]["Vp53_minus_V0p53_matched"]; ns = PR["n_p53_subset"]
A(f"\nV-p53 on the p53 subset ({ns['endoscopies']} endoscopies, {ns['progressor_endoscopies']} progressor, {ns['samples']} samples): sensitivity {p['endoscopy_sensitivity']:.3f} {cip(p['sens_ci95_rederived'])}, specificity {p['endoscopy_specificity']:.3f} {cip(p['spec_ci95_rederived'])}; p53 read on {pc_(p['share_endoscopies_sequenced'])} of these endoscopies. "
  f"V0 on the same subset at its own thresholds: {q['endoscopy_sensitivity']:.3f} / {q['endoscopy_specificity']:.3f}; V0 at V-p53's sensitivity (t_cnv evaluation-matched, step 1 fixed): specificity {d['comparator_specificity']:.3f}. Δ spec V-p53 − V0 {d['delta']:+.3f} {ci(d['rederived']['ci95'])} (re-derived), {ci(d['fixed']['ci95'])} (fixed).")

# ---------------------------------------------------------------- patient level
A("\n### Patient level (key secondary unit 2; bias-corrected, unmatched, descriptive)\n")
PL = PR["patient_level"]; pn = PL["n"]
A(f"{pn['patients']} patients ({pn['progressor']} progressor, {pn['non_progressor']} non-progressor); pre-event endoscopies per patient: progressor {pn['endoscopies_per_patient_mean']['progressor']:.2f}, non-progressor {pn['endoscopies_per_patient_mean']['non_progressor']:.2f}; non-progressor follow-up {pn['non_progressor_patient_years']:.1f} patient-years. "
  "(H) and (S-LATE) here use the threshold reaching V0's per-repeat endoscopy sensitivity (binary calls).\n")
A(f"| Rule | Progressor patients detected per repeat: median (range) of {pn['progressor']} [exact 95% CI, median repeat] | Non-progressor patients ever flagged (of {pn['non_progressor']}) | Per-patient share of endoscopies flagged, equal weights: progressor / non-progressor | Last 2 endoscopies: progressor detected share / non-progressor patients flagged (of {pn['non_progressor']}) | Flagged non-progressor endoscopies per patient-year |\n|---|---|---|---|---|---|")
for nm in ("V0", "V0ovr", "H", "SLATE", "R49", "GP53"):
    o = PL[nm]; dp = o["progressor_patients_detected_per_repeat"]
    A(f"| {TLB[nm]} | {dp['median']:.1f} ({c5(dp['min'])}–{c5(dp['max'])}) {cip(o['exact_ci95_median_repeat'])} | {o['non_progressor_patients_ever_flagged_mean']:.1f} | {o['per_patient_share_of_endoscopies_flagged_equal_weight']['progressor']:.3f} / {o['per_patient_share_of_endoscopies_flagged_equal_weight']['non_progressor']:.3f} | "
      f"{o['ever_flagged_last2_endoscopies']['progressor_share']:.3f} / {o['ever_flagged_last2_endoscopies']['non_progressor_mean_count']:.1f} | {o['flagged_non_progressor_endoscopies_per_patient_year']:.3f} |")
o = PL["V0"]
A(f"\nV0 progressor-patient detection averaged over repeats {o['detected_proportion']:.3f}, patient bootstrap 95% CI {cip(o['detected_proportion_ci95_rederived'])} (re-derived), {cip(o['detected_proportion_ci95_fixed'])} (fixed). With {pn['progressor']} progressors, detecting all still has an exact lower limit of about 0.89.")

# ---------------------------------------------------------------- timing
A("\n### Timing (descriptive)\n")
TM = PR["timing"]; tv = TM["V0"]
A("| Rule | Endo sens, < 2 y | 2–5 y | > 5 y | Years first flag → event: median of repeat medians (pooled IQR) | Progressor patients first flagged only at last pre-event endoscopy (of whom ≥ 2 endoscopies) | Missed progressor patients (range over repeats) |\n|---|---|---|---|---|---|---|")
for nm in ("V0", "V0ovr", "H", "SLATE", "R49", "GP53"):
    t = TM[nm]; b = t["endoscopy_sensitivity_by_years_to_event"]; y_ = t["years_first_flag_to_event"]
    A(f"| {TLB[nm]} | {f3_(b['<2']['sensitivity'])} | {f3_(b['2-5']['sensitivity'])} | {f3_(b['>5']['sensitivity'])} | {y_['median_of_per_repeat_medians']:.2f} ({y_['pooled_iqr'][0]:.2f}–{y_['pooled_iqr'][1]:.2f}) | {t['progressor_patients_first_flagged_only_at_last_pre_event_endoscopy_mean']:.1f} ({t['of_which_patients_with_ge2_endoscopies_mean']:.1f}) | {t['missed_progressor_patients_mean']:.1f} ({c5(t['missed_progressor_patients_per_repeat_range'][0])}–{c5(t['missed_progressor_patients_per_repeat_range'][1])}) |")
b = tv["endoscopy_sensitivity_by_years_to_event"]
A(f"\nProgressor endoscopies per band: < 2 y {c5(b['<2']['n_endoscopies'])}, 2–5 y {c5(b['2-5']['n_endoscopies'])}, > 5 y {c5(b['>5']['n_endoscopies'])}. Counts per patient are means over the 10 repeats.")
cn = tv["progressor_endoscopies_cleared_by_copy_number"]
A(f"\nV0, progressor endoscopies cleared by copy number at step 2: {cn['mean_per_repeat']:.1f} per repeat; a later pre-event endoscopy of the same patient is flagged for {cn['later_endoscopy_flagged_mean']:.1f}, a later one exists but is not flagged for {cn['later_endoscopy_not_flagged_mean']:.1f}, and there is no later pre-event endoscopy for {cn['no_later_pre_event_endoscopy_mean']:.1f}. A missed progressor is a patient never flagged before the event.")

# ---------------------------------------------------------------- samples
A("\n### Samples (secondary unit)\n")
A("| Rule | Sample sens | Sample spec |\n|---|---|---|")
for nm in ("V0", "V0ovr", "H", "SLATE", "R49", "GP53"): A(f"| {TLB[nm]} | {PR['sample_unit'][nm]['sensitivity']:.3f} | {PR['sample_unit'][nm]['specificity']:.3f} |")
A("\nEach sample takes its endoscopy's call.")

# ---------------------------------------------------------------- clinical output
A("\n### Clinical output: per 1,000 NDBE surveillance endoscopies\n")
CO = PR["clinical_per_1000_ndbe"]; cr = CO["rules"]; nu = CO["ndbe_unit"]; cs = CO["costs"]
A(f"Class-conditional rates on the {nu['endoscopies']} NDBE endoscopies ({nu['progressor_endoscopies']} progressor), projected to a 2% progressor-endoscopy prevalence. **Costs are the pre-specification's assumptions**, not measured: sequencing £85 per sample (sensitivity £150, £250); 3% sequencing failures, treated as flags; "
  f"each flag £{cs['flag_gbp']:.0f} (repeat endoscopy £471 + expert pathology review £44, an NHS histopathology proxy, + p53 £20); slide scanning and AI £5 per slide for every rule except grade + p53. The cost of misses is not modelled. (H) and (S-LATE) use the threshold reaching V0's per-repeat sensitivity on the NDBE endoscopies.\n")
A("| Rule | Endoscopies with sequencing | Endoscopies flagged | Progressor endoscopies missed | Cost at £85 | at £150 | at £250 | Extra cost per additional progressor patient detected vs grade + p53 (£85) |\n|---|---|---|---|---|---|---|---|")
for nm in ("V0", "V0ovr", "H", "SLATE", "R49", "GP53"):
    o = cr[nm]; ex = o.get("extra_cost_per_additional_progressor_patient_detected_vs_grade_p53_seq_85")
    A(f"| {TLB[nm]} | {o['endoscopies_with_sequencing']:.1f} | {o['endoscopies_flagged']:.1f} | {o['progressor_endoscopies_missed']:.2f} | {gbp(o['cost_gbp_seq_85'])} | {gbp(o['cost_gbp_seq_150'])} | {gbp(o['cost_gbp_seq_250'])} | {'(reference)' if nm == 'GP53' else ('not defined (no additional detection)' if ex is None else gbp(ex))} |")
A(f"\nAdditional progressor patients per 1,000 endoscopies are projected as (20 progressor endoscopies ÷ {nu['progressor_endoscopies_per_progressor_patient']:.2f} progressor NDBE endoscopies per progressor patient) × the rule's patient detection on the NDBE endoscopies. Extra cost per additional progressor endoscopy detected (£85): " +
  "; ".join(f"{TLB[nm]} {gbp(cr[nm].get('extra_cost_per_additional_progressor_endoscopy_detected_vs_grade_p53_seq_85'))}" for nm in ("V0", "V0ovr", "H", "SLATE", "R49")) + ".")

# ---------------------------------------------------------------- pathway
SB = PR["same_block"]
A("\n### Assumed clinical pathway\n")
A(f"V0 sequences {SB['v0_sequenced_samples_per_repeat_mean']:.1f} samples per repeat; {SB['v0_sequenced_not_same_block_per_repeat_mean']:.1f} per repeat are not labelled same-block pairs. Over the 10 repeats {SB['v0_sequenced_in_any_repeat']} distinct samples are sequenced at least once, {c5(SB['v0_sequenced_in_any_repeat_not_same_block'])} of them not same-block (different block, block not recorded, or case unparseable; round-7 lookup Q1 labels). "
  "Whether the scanned slides are the diagnostic H&E or research sections is unconfirmed (limitation); turnaround ≤ 4 weeks is assumed.")

# ---------------------------------------------------------------- curve
A("\n### Trade-off curve (descriptive)\n")
A("| t_low: training step-1 sensitivity | Two-stage target | Endo sens | Endo spec | Samples sequenced | Endoscopies sequenced | (H) spec at same sens | (S-LATE) spec at same sens | Folds where the target was unreachable |\n|---|---|---|---|---|---|---|---|---|")
for g in CV["grid"]:
    A(f"| {100 * g['t_low_training_step1_sensitivity']:.0f}% | {100 * g['two_stage_training_target']:.0f}% | {g['endoscopy_sensitivity']:.3f} | {g['endoscopy_specificity']:.3f} | {pc_(g['share_samples_sequenced'])} | {pc_(g['share_endoscopies_sequenced'])} | {g['H_specificity_at_same_sensitivity']:.3f} | {g['SLATE_specificity_at_same_sensitivity']:.3f} | {g['folds_target_unreachable_all_uncleared_flagged']} |")
A(f"| current care: grade ID/LGD | — | {cv['grade_ID_LGD']['endoscopy_sensitivity']:.3f} | {cv['grade_ID_LGD']['endoscopy_specificity']:.3f} | 0.0% | 0.0% | | | |")
A(f"| current care: grade or p53 (p53 subset) | — | {cv['grade_or_p53_on_p53_subset']['endoscopy_sensitivity']:.3f} | {cv['grade_or_p53_on_p53_subset']['endoscopy_specificity']:.3f} | 0.0% | 0.0% | | | |")
A("\nThe primary V0 is the 98% / 95% cell. Where the two-stage target exceeds what step 1 leaves in training, no copy-number threshold reaches it and every uncleared endoscopy is flagged (last column; such points coincide)." + ("" if pc["success"] else " Because the primary contrast failed, no curve point is promoted."))

# ---------------------------------------------------------------- stability
A("\n### Threshold stability (100 folds)\n")
TS = PR["threshold_stability"]; qs = lambda q: f"{q['median']:.3f} (IQR {q['iqr'][0]:.3f}–{q['iqr'][1]:.3f}; range {q['range'][0]:.3f}–{q['range'][1]:.3f})"
A(f"- t_low: {qs(TS['t_low'])}; t_cnv: {qs(TS['t_cnv'])}; folds where no copy-number threshold met the target (all uncleared flagged): {TS['folds_t_cnv_infeasible']}.")
A(f"- Training progressor endoscopies per fold: {TS['train_progressor_endoscopies_per_fold']['median']:.0f} (range {TS['train_progressor_endoscopies_per_fold']['range'][0]:.0f}–{TS['train_progressor_endoscopies_per_fold']['range'][1]:.0f}). Folds by training progressor endoscopies below t_low: none {TS['folds_by_train_progressor_endoscopies_below_t_low']['0']}, 1–4 {TS['folds_by_train_progressor_endoscopies_below_t_low']['1-4']}, ≥ 5 {TS['folds_by_train_progressor_endoscopies_below_t_low']['>=5']}; by training progressor patients below t_low: none {TS['folds_by_train_progressor_patients_below_t_low']['0']}, 1–4 {TS['folds_by_train_progressor_patients_below_t_low']['1-4']}, ≥ 5 {TS['folds_by_train_progressor_patients_below_t_low']['>=5']}.")
A(f"- Targets against realised: step 1 target {TS['target_step1']:.2f}, training {qs(TS['train_step1_sensitivity'])}, outer per fold {qs(TS['outer_step1_sensitivity_per_fold'])}; two-stage target {TS['target_two_stage']:.2f}, training {qs(TS['train_two_stage_sensitivity'])}, outer per fold {qs(TS['outer_two_stage_sensitivity_per_fold'])}; outer per repeat {', '.join(f'{x:.3f}' for x in TS['outer_two_stage_sensitivity_per_repeat'])}.")

# ---------------------------------------------------------------- robustness
A("\n### Robustness of V0 (primary contrast)\n")
A("| Row | Endoscopies (progressor) / samples / patients (progressor) | V0 endo sens / spec | (H) spec at V0's sens | Δ spec V0 − (H) [re-derived] | [fixed] | Lower limit > 0 | Samples / endoscopies sequenced | Δ spec V0 − (S-LATE) (bound) |\n|---|---|---|---|---|---|---|---|---|")
for r in MG["table"]:
    lab = r["label"] + (" (reference)" if r["row"] == "primary" else "")
    if r["status"] != "done": A(f"| {lab} | | | | {r['status']} | | | | |"); continue
    nn = r.get("n"); ns_ = f"{c5(nn['n_endoscopies'])} ({c5(nn['n_progressor_endoscopies'])}) / {c5(nn['n_samples'])} / {c5(nn['n_patients'])} ({c5(nn['n_progressor_patients'])})" if nn else ""
    ss = f"{r['v0_sens_spec'][0]:.3f} / {r['v0_sens_spec'][1]:.3f}" if "v0_sens_spec" in r else ""; hs = f"{r['H_spec']:.3f}" if "H_spec" in r else ""
    sq = (f"{pc_(r['share_samples_sequenced'])}" + (f" / {pc_(r['share_endoscopies_sequenced'])}" if r.get("share_endoscopies_sequenced") is not None else "")) if "share_samples_sequenced" in r else ""
    sl = (f"{r['d_spec_vs_SLATE']:+.3f}" + (f" ({r['d_spec_vs_SLATE_lower_rederived']:+.3f})" if r.get("d_spec_vs_SLATE_lower_rederived") is not None else "")) if "d_spec_vs_SLATE" in r else ""
    lab2 = {"between_repeat": "variance-inflated CI", "f3": "percentile over draws"}.get(r["row"])
    A(f"| {lab} | {ns_} | {ss} | {hs} | {r['d_spec']:+.3f} {ci(r['ci95_rederived'])}{' (' + lab2 + ')' if lab2 else ''} | {ci(r.get('ci95_fixed'))} | {yn(r['lower_above_0_rederived'])} | {sq} | {sl} |")
SUM = MG["summary"]
A(f"\nOf {SUM['rows']} robustness rows, the lower 95% limit of Δ spec V0 − (H) is above 0 in {SUM['lower_limit_above_0']} and not above 0 in {SUM['not_above_0']}" + (f"; not available: {'; '.join(SUM['not_available'])}" if SUM["not_available"] else "; none was not available") + ". Reporting only; the verdict rests on the primary row.")
A(f"\nRefit bootstrap: V0 − (H) Δ spec positive in {F3['d_spec_vs_H']['share_of_draws_above_0']:.3f} of the {F3['draws_scored']} draws; held-out pre-event endoscopies per draw median {F3['n_endoscopies_per_draw']['median']:.0f} (range {F3['n_endoscopies_per_draw']['min']}–{F3['n_endoscopies_per_draw']['max']}).")

# ---------------------------------------------------------------- checks
A("\n### Checks\n")
CK = PR["checks"]; c = CK["r49_vs_round7"]; fl = lambda v: ", ".join(f"{x:.3f}" for x in v)
A(f"- R49 through this code: sens, spec, samples, endoscopies {fl(c['got'][:4])}, sensitivity CI (fixed) {cip(c['got'][4])} (round 7: {fl(c['expected'][:4])}, {cip(c['expected'][4])}); reproduced: {yn(c['reproduced'])}. Pre-specification: 0.907 / 0.560, 25.5% of samples.")
c = CK["H_at_R49_sensitivity_vs_Cstar_endo"]
A(f"- (H) at R49's sensitivity ({c['R49_sensitivity']:.3f}): specificity {c['H_specificity_evaluation_matched']:.3f} (evaluation-matched); C\\*-endo (training-matched, round 7c) {fl(c['Cstar_endo_sens_spec_training_matched'])}; (H) at C\\*-endo's own sensitivity: specificity {c['H_specificity_at_Cstar_endo_sensitivity']:.3f} (evaluation matching favours (H)).")
c = CK["SLATE_at_Bstar_endo_sensitivity"]
A(f"- (S-LATE) at B\\*-endo's sensitivity: specificity {c['SLATE_specificity_evaluation_matched']:.3f}; B\\*-endo {fl(c['Bstar_endo_sens_spec'])} (expected 0.923 / 0.530, reproduced: {yn(c['Bstar_endo_reproduced'])}); difference in specificity {c['SLATE_minus_Bstar_endo_spec']:+.3f} (evaluation matching favours the comparator).")
c = (F3.get("checks") or {}).get("r49_mean_sens_spec_vs_round7c_f3")
if c: A(f"- Refit bootstrap: R49 re-scored on the 200 stored F3 draws, mean sens / spec {fl(c['got'])} (round 7c: {fl(c['expected'])}); reproduced: {yn(c['reproduced'])}.")
lab = PR["same_block"]["pre_event_sample_labels"]
A(f"- Same-block labels of the {PR['n_samples']} pre-event samples (round-7 lookup Q1: same / different / unknown = 556 / 6 / 9): " + ", ".join(f"{k} {c5(v)}" for k, v in sorted(lab.items())) + ".")

A("\n### Implementation notes\n")
NOTES = [
 "Endoscopy = (patient, endoscopy number) over its pre-event samples; `s_E` = maximum L-IMG, top sample = the sample attaining it. In each (repeat, fold) the training endoscopies are the fold's training pre-event samples grouped the same way, scored by inner out-of-fold L-IMG (score, top sample, cleared) and inner L-CNV / L-LATE of the top sample (confirmation).",
 "`t_low` = Harrell–Davis 0.02 quantile (`scipy.stats.mstats.hdquantiles`) of the training progressor endoscopies' `s_E`; cleared if `s_E` < `t_low`. `t_cnv` = the largest threshold at which the two-stage rule flags ≥ 95% of training progressor endoscopies (cleared ones count as not flagged; the ⌈0.95 n⌉-th largest confirmation score among them). If no threshold reaches the target, every uncleared endoscopy is flagged (counted under threshold stability).",
 "V1 confirms by the top sample's L-LATE; V2 by the maximum L-CNV over all samples of an uncleared endoscopy (all its samples sequenced); V4 keeps a high band `s_E` > `t_high` (the training non-progressor endoscopies' 90% specificity point, the c\\* convention of the triage, which guarantees ≥ 90% training specificity) flagged unsequenced, with `t_cnv` set for the rule as a whole. Each variant's thresholds are derived the same way.",
 "V0+override uses V0's thresholds unchanged: endoscopies with any ID / LGD or p53-aberrant sample are flagged by pathology and not sequenced; p53 missing counts as wild-type (as R47). V-p53: endoscopies restricted to their samples with p53; step 1 with V0's `t_low`; flag if the top sample (by L-IMG among them) is p53-aberrant. 'V0 on the same subset at matched sensitivity' varies `t_cnv` with step 1 fixed (evaluation-matched to V-p53's per-repeat sensitivity).",
 "Comparators: (H) = `s_E`, (S-LATE) / (S-CNV) = maximum sample L-LATE / L-CNV, (R1) = L-LATE of one random sample per endoscopy (seeds 0–99, R41 draw order, specificity averaged over seeds). For each repeat the empirical ROC is pooled across folds, and specificity is read at the rule's realised endoscopy sensitivity in that repeat by linear interpolation between adjacent operating points; values are averaged over repeats and re-matched in each draw. The pre-specification's 'comparator matching re-derived from the training patients' is read as this evaluation-set re-matching, since no comparator threshold is set on training data.",
 "Bootstrap: patients of the analysed population drawn with replacement (`RandomState(0)`, the R49 loop; draws whose evaluated endoscopies are one-class are rejected), multiplicities used as integer weights. Re-derived: in each draw every fold's `t_low` (Harrell–Davis on the replicated training progressor endoscopies), `t_cnv`, the variants' thresholds and R49's triage cut-offs (tb_common.triage with weights) are recomputed; fitted models fixed. Fixed: the same draws, observed calls. The primary row's draws coincide with R49's, so R49's fixed interval reproduces round 7.",
 "Units: subsets (NDBE endoscopies, earliest endoscopy, ≥ 2 samples, tissue strata) evaluate the frozen V0 (thresholds from all training pre-event endoscopies) on that subset of endoscopies; draws resample all patients and evaluate the subset. The R52(a) and package rows use their own stored predictions with every threshold re-derived. Earliest = the endoscopy with the largest months-before-final per patient; tissue strata by the tissue tiles of the sample with the highest L-IMG averaged over repeats (as round 7c).",
 "Patient level: a patient is flagged if any of its pre-event endoscopies is. (H) and (S-LATE) use the binary threshold reaching V0's per-repeat sensitivity (`thr_sens`), which can give them slightly higher sensitivity than V0. Equal window: each patient's last 2 pre-event endoscopies (smallest months-before-final; for non-progressors the last sequenced endoscopy is the reference). Patient-years: span from first to last pre-event endoscopy (R47). Exact interval = Clopper–Pearson on the median repeat's count (lower median for an even split).",
 "Timing: endoscopy time to event = the samples' years to event; first flag = the earliest flagged pre-event endoscopy; 'first flagged only at the last pre-event endoscopy' = only the last endoscopy flagged. 'Cleared by copy number' = progressor endoscopies sequenced and not flagged in that repeat.",
 "Clinical table: rates from the NDBE endoscopies, means over repeats; a sequencing failure (3% per sequenced sample, independent) turns an unflagged endoscopy into a flag; samples sequenced, flags and slides are projected to 1,000 endoscopies at 2% prevalence as π·(progressor rate) + (1 − π)·(non-progressor rate). Extra cost per additional progressor patient is a projection through the cohort's progressor NDBE endoscopies per progressor patient (stated above); grade + p53 uses grade or p53-aberrant with missing p53 as wild-type and costs no scanning.",
 "Same-block status: the round-7 lookup Q1 labels (clinical_lookup/q6_q1fix.py logic copied, read-only); only aggregate counts are reported.",
 "Refit bootstrap: the 200 stored F3 draws (one 10-fold CV per patient-bootstrap draw); V0's thresholds re-derived per draw and fold from the draw's inner predictions; units = resampled patient copy × endoscopy; (H) and (S-LATE) evaluation-matched to V0's sensitivity in the draw.",
 "Between-repeat variance: mean of the 10 per-repeat Δ ± 1.96 √(V_boot + V_rep), V_boot from the re-derived draws (R33 / R15 rule).",
 "Costs are the pre-specification's values, labelled as assumptions. Checks compare values at three decimals with a tolerance of 0.0015.",
]
for t in NOTES: A(f"- {t}")
txt = open(DOC).read(); head = txt[:txt.rindex("\n---") + 4]
open(DOC, "w").write(head + "\n" + "\n".join(L) + "\n"); print("RENDERED", len(L), "lines")

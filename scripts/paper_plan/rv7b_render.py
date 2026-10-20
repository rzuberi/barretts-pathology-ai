"""Render docs/paper_review_round7b_gate.md results (pre-specification 0764a2e) from results/paper_final/review_round7b/*.json.
Appends below the document's final '---' line; the pre-specification above it is not touched. Idempotent (replaces anything after that line).
Counts of 1-4 patients, samples or endoscopies are shown as '<5'. RESC = results commit."""
import json, os
T = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
R = T + "/results/paper_final/review_round7b"; DOC = T + "/docs/paper_review_round7b_gate.md"; RESC = os.environ.get("RESC", "see git log")
J = lambda f: json.load(open(f"{R}/{f}"))
ci = lambda c: f"[{c[0]:+.3f}, {c[1]:+.3f}]" if c and c[0] is not None else ""
cip = lambda c: f"[{c[0]:.3f}, {c[1]:.3f}]" if c and c[0] is not None else ""
yn = lambda b: "yes" if b else "no"
vl = lambda s: s.replace(" | ", " · ") if s else s   # variant labels inside markdown tables
def c5(k):
    k = int(k); return "<5" if 0 < k < 5 else str(k)
L = []; A = L.append
ROWS = {"primary": "primary (all pre-event samples)", "a_ndbe": "(a) NDBE-only samples", "b_earliest": "(b) earliest pre-event endoscopy per patient", "c_ge2": "(c) endoscopies with ≥ 2 samples", "d_r52a": "(d) R52(a) cohort, refit predictions"}
X = {r: J(f"merge_{r}.json")["R7B"] for r in ROWS}; P = X["primary"]; CV = J("curve.json")["curve"]
LAB = {"nested": "**nested gate**", "Bstar_endo": "B\\*-endo (sequence all, B\\* threshold)", "B_endo": "(B)-endo (sequence all, (B) threshold)", "triage": "sample-level triage, aggregated", "rand_own": "random-one, own threshold",
       "rand_matched": "random-one, matched to the nested gate's endoscopy specificity", "r49_gate": "R49 gate (max, sample-level, highest, L-LATE)", "r49_all": "R49 variant (max, sample-level, all, L-LATE)"}

A("\n## Results\n")
A(f"Rendered by `scripts/paper_plan/rv7b_render.py` from `results/paper_final/review_round7b/` (aggregate results {RESC}). Single model = per-repeat values averaged over the 10 repeats; patient bootstrap 2,000 draws (`RandomState(0)`) with fitted models, cut-offs and selections fixed; bound = one-sided 95% lower bound (5th percentile). Unit = pre-event endoscopy.\n")
A("| Item | Status |\n|---|---|")
for it in ("nested gate against B\\*-endo (primary)", "against triage, (B)-endo, random-one, R49", "patient columns", "variant choice frequencies", "apparent results of the 36 variants (optimistic)", "robustness (a)–(e)", "trade-off curve"): A(f"| {it} | DONE |")
A(f"\nAll inner out-of-fold predictions needed (L-IMG, L-LATE and L-CNV `cnv_their`, stored under `killcoyne_mm/horizons/inner/`; R52(a) refit inner files for (d)) exist, so no variant was dropped from the nested grid.")

# ---------------------------------------------------------------- primary
A("\n### Nested gate (primary for this exploration)\n")
A(f"{P['n_endoscopies']} pre-event endoscopies ({P['n_progressor_endoscopies']} progressor), {P['n_samples']} samples, {P['n_patients']} patients ({P['n_progressor_patients']} progressor). In {P['n_folds'] - P['fallback_folds']} of {P['n_folds']} outer folds at least one variant was eligible on the inner predictions; {P['fallback_folds']} fell back to the sample-level triage. Nested endoscopy bands (per repeat): low {P['nested_endoscopy_bands']['low']:.3f}, middle {P['nested_endoscopy_bands']['middle']:.3f}, high {P['nested_endoscopy_bands']['high']:.3f}, fallback {P['nested_endoscopy_bands']['fallback_triage']:.3f}.\n")
A("| Rule | Endoscopy sens [95% CI] | Endoscopy spec [95% CI] | Samples sequenced | Endoscopies sequenced | Progressor patients missed (of 32) / non-progressor flagged (of 43) |\n|---|---|---|---|---|---|")
for nm, lab in LAB.items():
    g = P["rules"][nm]; A(f"| {lab} | {g['endoscopy_sensitivity']:.3f} {cip(g['sens_ci95'])} | {g['endoscopy_specificity']:.3f} {cip(g['spec_ci95'])} | {g['share_samples_sequenced']:.3f} | {g['share_endoscopies_sequenced']:.3f} | {g['progressor_patients_missed']:.1f} / {g['non_progressor_patients_flagged']:.1f} |")
A("\n| Nested gate − | Δ endoscopy sens [95% CI] | Bound | Within −0.05 | Δ endoscopy spec [95% CI] | Δ share of samples sequenced [95% CI] | Δ share of endoscopies sequenced |\n|---|---|---|---|---|---|---|")
for nm in list(LAB)[1:]:
    d = P["nested_minus"][nm]; s = d["d_endoscopy_sens"]
    A(f"| {LAB[nm]} | {s['delta']:+.3f} {ci(s['ci95'])} | {s['one_sided_95_lower']:+.3f} | {yn(s['non_inferior_at_-0.05'])} | {d['d_endoscopy_spec']['delta']:+.3f} {ci(d['d_endoscopy_spec']['ci95'])} | {d['d_share_samples']['delta']:+.3f} {ci(d['d_share_samples']['ci95'])} | {d['d_share_endoscopies']:+.3f} |")

# ---------------------------------------------------------------- choice
A("\n### Variant choice across the 100 outer folds\n")
A("| Variant (aggregation · cut-offs · sequenced · call) | Folds chosen |\n|---|---|")
for k, v in sorted(P["choice_frequency"].items(), key=lambda kv: -kv[1]): A(f"| {vl(k)} | {v} |")
cb = P["choice_by_factor"]; A("\nBy factor (folds with a variant chosen): " + "; ".join(f"{f}: " + ", ".join(f"{lv} {c}" for lv, c in d.items()) for f, d in cb.items()) + ".")
ii = P["inner"]
A(f"\nInner (training patients, inner out-of-fold predictions; medians over folds): eligible variants {ii['eligible_variants_per_fold']['median']:.0f} (range {ii['eligible_variants_per_fold']['min']}–{ii['eligible_variants_per_fold']['max']}) of 36; B\\*-endo sensitivity / specificity {ii['Bstar_endo_sens_spec_median'][0]:.3f} / {ii['Bstar_endo_sens_spec_median'][1]:.3f}; sample-level triage {ii['triage_endo_sens_spec_share_median'][0]:.3f} / {ii['triage_endo_sens_spec_share_median'][1]:.3f}, share sequenced {ii['triage_endo_sens_spec_share_median'][2]:.3f}; chosen variant {ii['chosen_sens_spec_share_median'][0]:.3f} / {ii['chosen_sens_spec_share_median'][1]:.3f}, share sequenced {ii['chosen_sens_spec_share_median'][2]:.3f}.")

# ---------------------------------------------------------------- robustness
A("\n### Robustness of the nested gate\n")
A("Same nested rule re-run on each population (cut-offs, comparators, inner selection and evaluation all restricted to it).\n")
A("| Row | Endoscopies (progressor) / samples / patients (progressor) | Nested endo sens / spec | Δ endo sens vs B\\*-endo [95% CI] | Bound | Within −0.05 | Δ endo spec vs B\\*-endo | Samples sequenced: nested / triage | Endoscopies sequenced: nested / triage | Δ endo sens vs triage (bound) | Fallback folds | Most-chosen variant |\n|---|---|---|---|---|---|---|---|---|---|---|---|")
for r, lab in ROWS.items():
    x = X[r]; g = x["rules"]["nested"]; t = x["rules"]["triage"]; d = x["nested_minus"]["Bstar_endo"]; dt = x["nested_minus"]["triage"]["d_endoscopy_sens"]; s = d["d_endoscopy_sens"]
    A(f"| {lab} | {c5(x['n_endoscopies'])} ({c5(x['n_progressor_endoscopies'])}) / {c5(x['n_samples'])} / {c5(x['n_patients'])} ({c5(x['n_progressor_patients'])}) | {g['endoscopy_sensitivity']:.3f} / {g['endoscopy_specificity']:.3f} | {s['delta']:+.3f} {ci(s['ci95'])} | {s['one_sided_95_lower']:+.3f} | {yn(s['non_inferior_at_-0.05'])} | {d['d_endoscopy_spec']['delta']:+.3f} | "
      f"{g['share_samples_sequenced']:.3f} / {t['share_samples_sequenced']:.3f} | {g['share_endoscopies_sequenced']:.3f} / {t['share_endoscopies_sequenced']:.3f} | {dt['delta']:+.3f} ({dt['one_sided_95_lower']:+.3f}) | {x['fallback_folds']} | {vl(x['most_chosen_variant'])} |")
br = P["between_repeat"]
A(f"\n**(e) Between-repeat variance** (primary, R33 / R15 logic): per-repeat Δ endoscopy sensitivity nested − B\\*-endo ranges {min(br['per_repeat_d_endoscopy_sens']):+.3f} to {max(br['per_repeat_d_endoscopy_sens']):+.3f}; the per-repeat bound stays within −0.05 in {br['n_repeats_lower_above_-0.05']} of 10 repeats ({br['share_repeats_lower_above_-0.05']:.0%}). With the between-repeat variance added: Δ {br['delta_repeat_mean']:+.3f}, 95% interval {ci(br['variance_inflated_ci95'])}, bound {br['variance_inflated_one_sided_lower']:+.3f} ({'within' if br['non_inferior_at_-0.05'] else 'not within'} −0.05).")

# ---------------------------------------------------------------- curve
A("\n### Trade-off curve (descriptive)\n")
A(f"Most-chosen variant ({vl(CV['variant'])}) applied in every fold (not nested); sample-level triage; random-one matched to the gate's endoscopy specificity at each grid point (own threshold: {CV['random_one_own_threshold']['endoscopy_sensitivity']:.3f} / {CV['random_one_own_threshold']['endoscopy_specificity']:.3f}, {CV['random_one_own_threshold']['share_samples_sequenced']:.3f} of samples). Endoscopy sens / spec, share of samples sequenced.\n")
A("| t_low sens / t_high spec | Gate variant | Triage | Random-one (matched) |\n|---|---|---|---|")
fm = lambda g: f"{g['endoscopy_sensitivity']:.3f} / {g['endoscopy_specificity']:.3f}, {g['share_samples_sequenced']:.3f}"
for g in CV["grid"]: A(f"| {g['t_low_training_sensitivity']:.2f} / {g['t_high_training_specificity']:.2f} | {fm(g['gate_most_chosen_variant'])} | {fm(g['triage'])} | {fm(g['random_one_matched_to_gate_endoscopy_specificity'])} |")

# ---------------------------------------------------------------- apparent
A("\n### Apparent results of the 36 variants (optimistic: not nested, chosen after seeing R49)\n")
A("| Variant | Endo sens | Endo spec | Samples sequenced | Endoscopies sequenced | Patients missed / flagged | Δ endo sens vs B\\*-endo (bound) | Δ endo spec vs B\\*-endo | Δ samples vs triage | Folds chosen (nested) |\n|---|---|---|---|---|---|---|---|---|---|")
for k, g in P["apparent_optimistic"].items():
    s = g["minus_Bstar_endo"]["d_endoscopy_sens"]
    A(f"| {vl(k)} | {g['endoscopy_sensitivity']:.3f} | {g['endoscopy_specificity']:.3f} | {g['share_samples_sequenced']:.3f} | {g['share_endoscopies_sequenced']:.3f} | {g['progressor_patients_missed']:.1f} / {g['non_progressor_patients_flagged']:.1f} | {s['delta']:+.3f} ({s['one_sided_95_lower']:+.3f}) | {g['minus_Bstar_endo']['d_endoscopy_spec']['delta']:+.3f} | {g['minus_triage_d_share_samples']:+.3f} | {g['times_chosen']} |")

# ---------------------------------------------------------------- what follows
nb = P["nested_minus"]["Bstar_endo"]["d_endoscopy_sens"]; fewer = P["rules"]["nested"]["share_samples_sequenced"] < P["rules"]["triage"]["share_samples_sequenced"]; win = nb["non_inferior_at_-0.05"]
A("\n### What follows\n")
A(f"Nested gate against B\\*-endo: bound {nb['one_sided_95_lower']:+.3f} ({'within' if win else 'not within'} −0.05); samples sequenced {P['rules']['nested']['share_samples_sequenced']:.3f} against {P['rules']['triage']['share_samples_sequenced']:.3f} for the sample-level triage ({'fewer' if fewer else 'not fewer'}).")
if win and fewer: A("\nBoth conditions hold, so by the pre-specified rule the nested gate becomes the manuscript's headline development result, described as \"selected by nested cross-validation, after R49\", and the proposed primary rule for ACE-B registration; the sample-level triage remains reported as the original pre-specified analysis.")
else: A("\nThe conditions are not both met, so by the pre-specified rule the sample-level triage stays the headline and the gate is reported as exploratory.")

# ---------------------------------------------------------------- checks
ck = P["checks"]; f = lambda v: ", ".join("" if x is None else f"{x:.3f}" for x in v)
A("\n### Checks\n")
A(f"- R49 gate through this code: {f(ck['r49_gate']['got'])} (R49: {f(ck['r49_gate']['expected'])}; endoscopy sens, spec, samples, endoscopies, patients missed, flagged); reproduced: {yn(ck['r49_gate']['reproduced'])}; sensitivity CI {cip(ck['r49_gate']['ci_got_expected'][0])} (R49 {cip(ck['r49_gate']['ci_got_expected'][1])}).")
A(f"- R49 all-samples variant: {f(ck['r49_all']['got'])} (R49: {f(ck['r49_all']['expected'])}); reproduced: {yn(ck['r49_all']['reproduced'])}.")
A(f"- B\\*-endo: {f(ck['Bstar_endo']['got'])} (Table 1 endoscopy column: {f(ck['Bstar_endo']['expected'])}); reproduced: {yn(ck['Bstar_endo']['reproduced'])}.")
A(f"- Sample-level triage aggregated: {f(ck['triage']['got'])} (R49: {f(ck['triage']['expected'])}); reproduced: {yn(ck['triage']['reproduced'])}.")
A(f"- Random-one at its own threshold: {f(ck['rand_own']['got'])} (R46: {f(ck['rand_own']['expected'])}); reproduced: {yn(ck['rand_own']['reproduced'])}.")
A("- The B\\* fold threshold recomputed for the inner evaluation reproduces the stored outer B\\* calls in every fold (asserted in each selection job).")

A("\n### Implementation notes\n")
NOTES = [
 "Inner evaluation: for each outer (repeat, fold), every variant is scored on the inner out-of-fold predictions of that fold's training patients (the stored inner files from which the triage's t_low, t_high and t_seq are chosen), grouped into the training patients' pre-event endoscopies. The sample-level cut-offs and t_seq are the fold's own (chosen on the same inner predictions); the endoscopy-level cut-offs are chosen on the same inner endoscopy scores. So inner performance is apparent with respect to the cut-offs, out-of-fold with respect to the models.",
 "Inner B\\*-endo: the fold's B\\* threshold (L-LATE, training-matched to the triage's inner specificity, exactly as `matched_calls`) applied to the inner L-LATE of the training samples; endoscopy flagged if any sample is positive. Eligibility uses 'at least' with a 1e-9 tolerance for floating ties. 'Share of samples sequenced' compares integer counts, so equal shares tie exactly and go to the lower variant in the pre-specified order (aggregation, then cut-off type, then sequenced sample, then call).",
 "Middle-band call: L-LATE at the triage's fold t_seq for every variant, including those with endoscopy-level cut-offs (t_seq is not re-chosen). L-CNV threshold = the fold t_seq of the rv4 copy-number-only middle band (`triage(..., mid='L-CNV')`, R35 a): 80% sensitivity of inner L-CNV among the middle-band training progressor samples, with the same fallback.",
 "'Uncertain' sample = the sample whose L-IMG is closest to (t_low + t_high) / 2 of the cut-offs in use (sample-level or endoscopy-level); ties to the first sample in sorted order. Top-2 mean = mean of the two highest L-IMG (the maximum when there is one sample). Endoscopy bands as R49: low if score < t_low, high if score > t_high, no middle band when t_low > t_high.",
 "Endoscopy-level cut-offs: t_low = the largest threshold keeping ≥ 95% of training progressor endoscopies at or above it (`thr_sens`), t_high = the ⌈0.90 n⌉-th smallest training non-progressor endoscopy score (`c_star`), per aggregation.",
 "Fallback (no eligible variant): the held-out endoscopies of that fold are called by the sample-level triage (flagged if any sample is triage-positive; sequenced samples = middle-band samples).",
 "Bootstrap: the R49 draw loop (2,000 patient draws, RandomState(0), draws with only one class rejected), endoscopies of a resampled patient kept together; calls, cut-offs and chosen variants fixed. Patient columns: patient flagged if any of its pre-event endoscopies is flagged; counts averaged over repeats.",
 "Random-one: R41 seeds 0–99, R41 draw order over the row's endoscopies in sorted order; own threshold = the stored (B) call of the picked sample; matched = one L-LATE threshold per seed × repeat, pooled across folds, set to the nested gate's endoscopy specificity in that repeat and re-set in each draw (as R46, with a 1e-9 guard on ⌈s·n⌉).",
 "Robustness rows re-run the whole procedure on the restricted population: cut-offs, t_seq, B\\*, (B) and inner selection are re-chosen on the training samples of that population (as R50 for NDBE). (a) NDBE pre-event samples; endoscopies with no NDBE sample drop out. (b) earliest pre-event endoscopy = the patient's pre-event endoscopy with the largest months-before-final; unlike R33 row 13, cut-offs are re-chosen on these endoscopies. (c) endoscopies with ≥ 2 pre-event samples. (d) the R52(a) set and its refit outer and inner predictions (`killcoyne_mm/review_round7/refit/r52a`), endoscopy = the set's (patient, endoscopy).",
 "Checks: values are compared at three decimals with a tolerance of 0.0015; the R49 all-samples variant's specificity prints 0.543 here against 0.542 in R49 because the value is exactly 0.5425 (915 of 2,000 non-progressor endoscopy-repeats flagged) and the two scripts' summation order decides the rounding; all other values are identical. The curve task ran before the scripts commit with byte-identical code.",
 "(e) as R15 / R33: per-repeat Δ with its own 5th-percentile bound; variance-inflated bound = mean Δ − 1.645 √(V_boot + V_rep), V_boot = variance of the bootstrap repeat-mean Δ, V_rep = variance of the 10 per-repeat Δ.",
 "Trade-off curve: the most-chosen variant (ties to the simpler) applied in all folds with cut-offs at each grid point (sample-level cut-offs and t_seq from the triage rule at those quantiles; endoscopy-level cut-offs at the same quantiles on inner endoscopy scores). Random-one has no cut-off quantiles, so it is shown at its own threshold and matched (evaluation-matched, per repeat) to the gate's endoscopy specificity at each grid point. No intervals.",
]
for t in NOTES: A(f"- {t}")
txt = open(DOC).read(); head = txt[:txt.rindex("\n---") + 4]
open(DOC, "w").write(head + "\n" + "\n".join(L) + "\n"); print("RENDERED", len(L), "lines")

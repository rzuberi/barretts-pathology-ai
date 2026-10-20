"""Render docs/paper_review_round7_analyses.md results (pre-specification 724ff97) from results/paper_final/review_round7/*.json.
Appends below the document's final '---' line; the pre-specification above it is not touched. Idempotent (replaces anything after that line).
Counts of 1-4 patients, samples or endoscopies are shown as '<5'. RESC = results commit."""
import json, os
T = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
R = T + "/results/paper_final/review_round7"; DOC = T + "/docs/paper_review_round7_analyses.md"; RESC = os.environ.get("RESC", "see git log")
J = lambda f: json.load(open(f"{R}/{f}"))
ci = lambda c: f"[{c[0]:+.3f}, {c[1]:+.3f}]" if c and c[0] is not None else ""
cip = lambda c: f"[{c[0]:.3f}, {c[1]:.3f}]" if c and c[0] is not None else ""
f3 = lambda x: "" if x is None else f"{x:+.3f}"
p3 = lambda x: "" if x is None else f"{x:.3f}"
yn = lambda b: "yes" if b else "no"
def c5(k):
    k = int(k); return "<5" if 0 < k < 5 else str(k)
L = []; A = L.append
r42 = J("r42.json")["R42"]; r44 = J("r44.json")["R44"]; r45 = J("r45.json")["R45"]; r46 = J("r46.json")["R46"]; r47 = J("r47.json")["R47"]; r48 = J("r48.json")["R48"]; r49 = J("r49.json")["R49"]
r50 = J("r50.json")["R50"]; r51 = J("r51.json")["R51"]; r53 = J("r53.json")["R53"]; r55 = J("r55.json")["R55"]; r56 = J("r56.json")["R56"]; r57 = J("r57.json")["R57"]; chk = J("check.json")["checks"]
prep = J("prep.json")["sets"]; RF = {v: J(f"refit_{v}.json")["refit"] for v in ("r43a", "r43b", "r43c", "r52a", "r52b", "r54")}
r38 = json.load(open(T + "/results/paper_final/review_round6/r38_merge.json"))["R38"]["rows"]

A("\n## Results\n")
A(f"Rendered by `scripts/paper_plan/rv7_render.py` from `results/paper_final/review_round7/` (aggregate results {RESC}). Single model = per-repeat values averaged over the 10 repeats; patient bootstrap 2,000 draws (`RandomState(0)`); bound = one-sided 95% lower bound (5th percentile).\n")
A("| Item | Status |\n|---|---|")
for it, st in [("R42 stability of the 10-model vote", "DONE"), ("R43 control exclusions with training refitted", "DONE (3 refits)"), ("R44 p53 among all triage-negative samples", "DONE"),
               ("R45 current care with p53 at matched specificity", "DONE"), ("R46 one random sample per endoscopy at matched specificity", "DONE"), ("R47 burden, prevalence and cost", "DONE"),
               ("R48 two-band rule", "DONE"), ("R49 per-endoscopy gate", "DONE"), ("R50 NDBE-only block", "DONE"), ("R51 falsely cleared progressor endoscopies", "DONE"),
               ("R52 samples of the 8 progressors previously reported without a slide", "DONE (2 refits)"), ("R53 event dates from the Barrett's database", "DONE"), ("R54 later disease in non-progressors", "DONE (1 refit)"),
               ("R55 image quality and within-class confound audit", "DONE"), ("R56 Killcoyne validation cohort: audit only", f"DONE (gate {'passed' if r56['gate']['passes'] else 'not passed'})"),
               ("R57 achievable precision (simulated, no ACE-B data)", "DONE"), ("R58 4x copy number of the same cohort", "NOT RUN: data not yet available")]:
    A(f"| {it} | {st} |")

def trirow(lab, o, extra=""):
    b = o["vs_Bstar"]; bb = o["vs_B"]; h = o.get("IMG_minus_CNV"); t = o["triage"]
    hs = f"{h['delta']:+.3f} ({h['one_sided_lower']:+.3f})" if h else ""; hw = yn(h["within_-0.05"]) if h else ""
    return (f"| {lab} | {o['n_samples']} ({o['n_progressor_samples']}) / {o['n_patients']} ({o['n_progressor_patients']}){extra} | {t['sensitivity']:.3f} / {t['specificity']:.3f} | {t['share_sequenced']:.3f} | "
            f"{b['d_sens']:+.3f} {ci(b['ci95'])} | {b['one_sided_lower']:+.3f} | {yn(b['within_-0.05'])} | {b['d_spec']:+.3f} | {bb['d_sens']:+.3f} ({bb['one_sided_lower']:+.3f}) | {hs} | {hw} |")
TRIH = "| Analysis | Pre-event samples (progressor) / patients (progressor) | Triage sens / spec | Share sequenced | Δ sens vs B\\* [95% CI] | Bound | Within −0.05 | Δ spec vs B\\* | Δ sens vs (B) (bound) | Δ L-IMG − L-CNV (bound) | Within |\n|---|---|---|---|---|---|---|---|---|---|---|"
prim = chk["r33_row1_expected_-0.016_-0.032_+0.033_+0.045"]

# ---------------------------------------------------------------- R42
A("\n### R42. Stability of the 10-model vote\n")
a5 = r42["two_5_model_votes_agree"]; a1 = r42["two_single_repeats_agree"]
A(f"Pre-event samples ({r42['n_samples']}). Two 5-model votes (≥ 3 of 5) over all {r42['n_splits']} splits of the 10 repeats into two sets of 5, and two single repeats over all {r42['n_single_pairs']} pairs: share of samples on which the two calls agree, mean (2.5–97.5% range over splits or pairs).\n")
A("| Samples | n | Two 5-model votes agree | Two single repeats agree |\n|---|---|---|---|")
for k, lab in (("all", "all pre-event"), ("modal_low", "modal band low"), ("modal_middle", "modal band middle"), ("modal_high", "modal band high"), ("progressor", "progressor"), ("non_progressor", "non-progressor")):
    A(f"| {lab} | {a5[k]['n_samples']} | {a5[k]['mean']:.3f} ({a5[k]['range_2.5_97.5'][0]:.3f}–{a5[k]['range_2.5_97.5'][1]:.3f}) | {a1[k]['mean']:.3f} ({a1[k]['range_2.5_97.5'][0]:.3f}–{a1[k]['range_2.5_97.5'][1]:.3f}) |")
A(f"\nTwo independent 5-model votes give the same call on {a5['all']['mean']:.0%} of samples, against {a1['all']['mean']:.0%} for two single repeats; the disagreement is in the middle band ({a5['modal_middle']['mean']:.0%} for votes, {a1['modal_middle']['mean']:.0%} for single repeats). The round-6 '62% unanimous' was the share of samples on which all 10 single repeats agree with the vote, not the stability of the vote itself.")

# ---------------------------------------------------------------- R43
A("\n### R43. Control exclusions with training refitted\n")
A("The same non-progressor samples as R38 (b) and (c) are removed from training and evaluation, and the whole pipeline (outer and inner fits, cut-offs, matched comparators) is refitted. Existing patients keep their folds.\n")
A(TRIH); A(trirow("primary (stored, R33 row 1)", prim))
for v, lab in (("r43a", "(a) non-progressor samples < 1 year before the last sequenced endoscopy removed, refitted"), ("r43b", "(b) < 2 years removed, refitted"), ("r43c", "(c) the non-progressor patient with a later HGD report removed, refitted")): A(trirow(lab, RF[v]))
A("\nR38 rows relabelled (stored calls, evaluation-only):\n")
A("| R38 row | Triage sensitivity | Δ spec vs B\\* | Note |\n|---|---|---|---|")
for r in r38:
    if r.get("status", "done") != "done": continue
    if r["analysis"].startswith("(b)") or r["analysis"].startswith("(c)"):
        A(f"| {r['analysis']} | {r['triage']['sensitivity']:.3f} | {r['vs_Bstar']['d_spec']:+.3f} | evaluation-only: sensitivity unchanged by construction; Δ specificity shown |")
nin = sum(RF[v]["vs_Bstar"]["within_-0.05"] for v in ("r43a", "r43b", "r43c")); hin = sum(RF[v]["IMG_minus_CNV"]["within_-0.05"] for v in ("r43a", "r43b", "r43c"))
A(f"\nWith training refitted, the triage stays within −0.05 of B\\* in {nin} of 3 exclusions and L-IMG within −0.05 of L-CNV in {hin} of 3.")

# ---------------------------------------------------------------- R44
A("\n### R44. p53 among all triage-negative samples\n")
tn = r44["triage_negative_per_repeat"]; lo = r44["low_band_only_per_repeat"]; md = r44["modal_low_band_R37e"]
A("Pre-event samples called negative by the triage in a repeat (low band, or middle band negative on L-LATE); counts are means over the 10 repeats (range over repeats).\n")
A("| Set | Progressor: negative / with p53 / p53-aberrant | Non-progressor: negative / with p53 / p53-aberrant |\n|---|---|---|")
for lab, d in (("all triage-negative (R44)", tn), ("low band only (R37 e, per repeat)", lo)):
    A(f"| {lab} | {d['progressor']['negative_samples_mean']:.1f} / {d['progressor']['negative_with_p53_mean']:.1f} / {d['progressor']['p53_aberrant_mean_over_repeats']:.1f} ({d['progressor']['p53_aberrant_range'][0]}–{d['progressor']['p53_aberrant_range'][1]}) | "
      f"{d['non_progressor']['negative_samples_mean']:.1f} / {d['non_progressor']['negative_with_p53_mean']:.1f} / {d['non_progressor']['p53_aberrant_mean_over_repeats']:.1f} ({d['non_progressor']['p53_aberrant_range'][0]}–{d['non_progressor']['p53_aberrant_range'][1]}) |")
A(f"| modal low band (R37 e) | — / {md['progressor']['cleared_with_p53']} / {c5(md['progressor']['p53_aberrant'])} | — / {md['non_progressor']['cleared_with_p53']} / {c5(md['non_progressor']['p53_aberrant'])} |")
A(f"\nOf {r44['p53_aberrant_pre_event']['progressor']} p53-aberrant progressor samples, a mean of {tn['progressor']['p53_aberrant_mean_over_repeats']:.1f} per repeat are triage-negative (all negatives), against {lo['progressor']['p53_aberrant_mean_over_repeats']:.1f} in the low band alone.")

# ---------------------------------------------------------------- R45
A("\n### R45. Current care with p53 at matched specificity\n")
g = r45["grade_or_p53"]
A(f"On the {r45['n_samples']} pre-event samples with p53 ({r45['n_progressor_samples']} progressor; {r45['n_patients']} patients). Grade ID/LGD or p53 aberrant: per sample sensitivity {g['sample']['sensitivity']:.3f}, specificity {g['sample']['specificity']:.3f}; per patient {g['patient']['sensitivity']:.3f} / {g['patient']['specificity']:.3f}. Each model's threshold is set per repeat so that its specificity on these samples equals grade-or-p53's (evaluation-matched; re-matched in each draw).\n")
A("| Level | Model at grade-or-p53's specificity | Sensitivity [95% CI] | Δ sensitivity vs grade-or-p53 [95% CI] | Bound |\n|---|---|---|---|---|")
for lv, lab in (("per_sample", "sample"), ("per_patient", "patient")):
    for m, ml in (("L-LATE", "L-LATE (sequence all)"), ("L-IMG", "L-IMG (H&E only)")):
        u = r45[lv][m]; d = u["d_sens_minus_grade_or_p53"]; A(f"| {lab} | {ml} | {u['sensitivity_at_matched_specificity']:.3f} {cip(u['sens_ci95'])} | {d['delta']:+.3f} {ci(d['ci95'])} | {d['one_sided_95_lower']:+.3f} |")
to = r45["triage_own_operating_point"]
A(f"\nSecondary, triage at its own operating point on these samples (share sequenced {r45['share_sequenced_triage']:.3f}): per sample {to['sample']['sensitivity']:.3f} / {to['sample']['specificity']:.3f} (Δ sensitivity {to['sample']['d_sens_minus_grade_or_p53']['delta']:+.3f} {ci(to['sample']['d_sens_minus_grade_or_p53']['ci95'])}, Δ specificity {to['sample']['d_spec_minus_grade_or_p53']['delta']:+.3f} {ci(to['sample']['d_spec_minus_grade_or_p53']['ci95'])}); per patient {to['patient']['sensitivity']:.3f} / {to['patient']['specificity']:.3f} (Δ sensitivity {to['patient']['d_sens_minus_grade_or_p53']['delta']:+.3f}, Δ specificity {to['patient']['d_spec_minus_grade_or_p53']['delta']:+.3f} {ci(to['patient']['d_spec_minus_grade_or_p53']['ci95'])}).")

# ---------------------------------------------------------------- R46
A("\n### R46. One random sample per endoscopy at matched endoscopy specificity\n")
a = r46["a_matched_specificity"]; dd = a["d_endoscopy_sensitivity_triage_minus_random"]
rg = lambda u: f" ({u['range_2.5_97.5_over_seeds'][0]:.3f}–{u['range_2.5_97.5_over_seeds'][1]:.3f})" if u.get("range_2.5_97.5_over_seeds") else ""
A(f"{r46['n_endoscopies']} pre-event endoscopies ({r46['n_progressor_endoscopies']} progressor); R41 seeds (100). Mean over seeds (2.5–97.5% range over seeds) [bootstrap CI of the seed mean].\n")
A("| Rule | Endoscopy sensitivity | Endoscopy specificity |\n|---|---|---|")
for k, lab in (("triage", "triage"), ("random_matched", "one random sample, L-LATE threshold matched to the triage's endoscopy specificity"), ("random_own_threshold", "one random sample, (B)'s fold threshold (R41)")):
    u = a[k]; A(f"| {lab} | {u['endoscopy_sensitivity']['mean_over_seeds']:.3f}{rg(u['endoscopy_sensitivity'])} {cip(u['endoscopy_sensitivity']['ci95_bootstrap_of_seed_mean'])} | {u['endoscopy_specificity']['mean_over_seeds']:.3f}{rg(u['endoscopy_specificity'])} {cip(u['endoscopy_specificity']['ci95_bootstrap_of_seed_mean'])} |")
A(f"\n**(a)** Δ endoscopy sensitivity, triage − random at matched specificity: {dd['mean_over_seeds']:+.3f} (seeds {dd['range_2.5_97.5_over_seeds'][0]:+.3f} to {dd['range_2.5_97.5_over_seeds'][1]:+.3f}) {ci(dd['ci95_bootstrap_of_seed_mean'])}, bound {dd['one_sided_95_lower']:+.3f}.\n")
b = r46["b_per_patient"]
A(f"**(b) Per patient** (of {b['n_progressor_patients']} progressor / {b['n_non_progressor_patients']} non-progressor patients; means over seeds and repeats): random rule at its own threshold misses {b['random_own_threshold']['progressor_patients_missed']:.1f} and flags {b['random_own_threshold']['non_progressor_patients_flagged']:.1f}; at the matched threshold misses {b['random_matched']['progressor_patients_missed']:.1f} and flags {b['random_matched']['non_progressor_patients_flagged']:.1f}; triage misses {b['triage']['progressor_patients_missed']:.1f} and flags {b['triage']['non_progressor_patients_flagged']:.1f}.\n")
c = r46["c_burden"]
A("**(c) Sequencing burden.**\n\n| Rule | Share of samples sequenced | Share of endoscopies with ≥ 1 sample sequenced |\n|---|---|---|")
for k, lab in (("random", "one random sample per endoscopy"), ("triage", "triage"), ("Bstar", "B\\*"), ("B", "(B)")): A(f"| {lab} | {c[k]['share_samples_sequenced']:.3f} | {c[k]['share_endoscopies_with_a_sequenced_sample']:.3f} |")

# ---------------------------------------------------------------- R47
A("\n### R47. Burden, prevalence and cost (descriptive)\n")
RN = [("T", "triage"), ("2B", "two-band rule (R48)"), ("B*", "B\\*"), ("B", "(B)"), ("C", "(C) H&E only"), ("D", "(D) grade ID or LGD"), ("D+p53", "grade or p53 aberrant")]
A(f"**(a) Flag rate.** Flagged non-progressor endoscopies per non-progressor patient-year ({r47['n_non_progressor_patients']} patients, {r47['non_progressor_patient_years']:.1f} patient-years between first and last pre-event endoscopy).\n")
A("| Rule | Sensitivity | False-positive rate | Share sequenced (progressor / non-progressor) | Flagged non-progressor endoscopies (mean) | Per patient-year |\n|---|---|---|---|---|---|")
for k, lab in RN:
    u = r47["rules"][k]; A(f"| {lab} | {u['sensitivity']:.3f} | {u['false_positive_rate']:.3f} | {u['share_sequenced_progressor']:.3f} / {u['share_sequenced_non_progressor']:.3f} | {u['flagged_non_progressor_endoscopies_mean']:.1f} | {u['a_flagged_non_progressor_endoscopies_per_patient_year']:.3f} |")
A(f"\n**(b), (c) Projection per 1,000 surveillance samples** (class-conditional rates from this cohort; mean {r47['samples_per_endoscopy']['mean']:.2f} samples per endoscopy; £85 per sequenced sample, £471 per flagged endoscopy; illustrative, no threshold set).\n")
A("| Prevalence | Rule | Samples sequenced (share, c) | Flagged samples | Flagged endoscopies | Progressor samples missed | Cost (£) |\n|---|---|---|---|---|---|---|")
for pv in ("0.005", "0.02", "0.05"):
    for k, lab in RN:
        u = r47["rules"][k]["projection_per_1000_samples"][pv]; A(f"| {float(pv):.1%} | {lab} | {u['samples_sequenced']:.0f} ({u['c_share_sequenced_reweighted']:.3f}) | {u['flagged_samples']:.0f} | {u['flagged_endoscopies']:.0f} | {u['progressor_samples_missed']:.1f} | {u['cost_gbp']:,} |")

# ---------------------------------------------------------------- R48
A("\n### R48. Two-band rule (secondary)\n")
tb = r48["two_band"]
A(f"Low band cleared; middle and high bands sequenced and called by L-LATE at the triage's fold threshold. Sensitivity {tb['sensitivity']:.3f} {cip(tb['sens_ci95'])}, specificity {tb['specificity']:.3f} {cip(tb['spec_ci95'])}, share sequenced {tb['share_sequenced']:.3f} {cip(tb['share_ci95'])}. Share of high-band samples that L-LATE calls negative: {tb['high_band_called_negative_share_of_high']:.3f}.\n")
A("| Two-band − | Δ sensitivity [95% CI] | Bound | Δ specificity [95% CI] |\n|---|---|---|---|")
for k, lab in (("T", "triage"), ("B*", "B\\*"), ("B", "(B)")):
    u = r48[f"two_band_minus_{k}"]; A(f"| {lab} | {u['d_sens']['delta']:+.3f} {ci(u['d_sens']['ci95'])} | {u['d_sens']['one_sided_95_lower']:+.3f} | {u['d_spec']['delta']:+.3f} {ci(u['d_spec']['ci95'])} |")
def t1(tab, names):
    A("| Strategy | Sequenced | Sensitivity | Specificity | Endoscopy sens / spec | Patients missed / flagged |\n|---|---|---|---|---|---|")
    for k, lab in names:
        u = tab["rules"][k]; sh = "" if u["share_sequenced"] is None else f"{u['share_sequenced']:.1%}"
        A(f"| {lab} | {sh} | {u['sensitivity']:.3f} {cip(u['sens_ci95'])} | {u['specificity']:.3f} {cip(u['spec_ci95'])} | {u['endoscopy_sensitivity']:.3f} / {u['endoscopy_specificity']:.3f} | {u['progressor_patients_missed']:.1f} / {u['non_progressor_patients_flagged']:.1f} |")
A(""); t1(r48["table1"], (("2B", "two-band rule"), ("T", "triage"), ("B*", "B\\*"), ("B", "(B)")))
A(f"\nBurden (R47): two-band sequences {r47['rules']['2B']['share_sequenced_cohort']:.3f} of samples against {r47['rules']['T']['share_sequenced_cohort']:.3f} for the triage, at the same flag rate ({r47['rules']['2B']['a_flagged_non_progressor_endoscopies_per_patient_year']:.3f} vs {r47['rules']['T']['a_flagged_non_progressor_endoscopies_per_patient_year']:.3f} per non-progressor patient-year).")

# ---------------------------------------------------------------- R49
A("\n### R49. Per-endoscopy gate (secondary, exploratory)\n")
bs = r49["endoscopy_band_shares"]
A(f"{r49['n_endoscopies']} pre-event endoscopies ({r49['n_progressor_endoscopies']} progressor). Endoscopy bands (per repeat mean): low {bs['low']:.3f}, middle {bs['middle']:.3f}, high {bs['high']:.3f}.\n")
A("| Rule | Endoscopy sens [95% CI] | Endoscopy spec [95% CI] | Samples sequenced | Endoscopies sequenced | Patients missed / flagged | Δ endoscopy sens vs triage [95% CI] (bound) | Δ endoscopy spec [95% CI] |\n|---|---|---|---|---|---|---|---|")
for k, lab in (("gate", "gate: top-L-IMG sample sequenced"), ("gate_all_mid_sequenced", "variant: all middle-endoscopy samples sequenced"), ("triage", "sample-level triage, aggregated")):
    u = r49["rules"][k]; d = u.get("minus_triage")
    ds = f"{d['d_endoscopy_sens']['delta']:+.3f} {ci(d['d_endoscopy_sens']['ci95'])} ({d['d_endoscopy_sens']['one_sided_95_lower']:+.3f})" if d else ""; dp = f"{d['d_endoscopy_spec']['delta']:+.3f} {ci(d['d_endoscopy_spec']['ci95'])}" if d else ""
    A(f"| {lab} | {u['endoscopy_sensitivity']:.3f} {cip(u['sens_ci95'])} | {u['endoscopy_specificity']:.3f} {cip(u['spec_ci95'])} | {u['share_samples_sequenced']:.3f} | {u['share_endoscopies_sequenced']:.3f} | {u['progressor_patients_missed']:.1f} / {u['non_progressor_patients_flagged']:.1f} | {ds} | {dp} |")

# ---------------------------------------------------------------- R50
A("\n### R50. NDBE-only block\n")
nn = r50["n"]
A(f"Pre-event NDBE samples: {nn['n_samples']} ({nn['n_progressor_samples']} progressor), {nn['n_patients']} patients ({nn['n_progressor_patients']} progressor), {nn['n_endoscopies']} endoscopies ({nn['n_progressor_endoscopies']} progressor). (D) flags nothing on NDBE samples, so (D)+p53 = p53 aberrant (missing as wild-type).\n")
t1(r50, (("T", "H&E-first triage"), ("B*", "B\\*"), ("B", "(B)"), ("C*", "C\\*"), ("D+p53", "(D)+p53")))
ct = r50["claim_T_H"]
A(f"\nTriage − B\\* {ct['vs_Bstar']['d_sens']:+.3f} {ci(ct['vs_Bstar']['ci95'])} (bound {ct['vs_Bstar']['one_sided_lower']:+.3f}); triage − (B) {ct['vs_B']['d_sens']:+.3f} (bound {ct['vs_B']['one_sided_lower']:+.3f}); L-IMG − L-CNV {ct['IMG_minus_CNV']['delta']:+.3f} (bound {ct['IMG_minus_CNV']['one_sided_lower']:+.3f}). On the {r50['d_plus_p53_on_samples_with_p53']['n_samples']} NDBE samples with a p53 result, (D)+p53 = {r50['d_plus_p53_on_samples_with_p53']['sensitivity']:.3f} / {r50['d_plus_p53_on_samples_with_p53']['specificity']:.3f}.")

# ---------------------------------------------------------------- R51
A("\n### R51. Falsely cleared progressor endoscopies (descriptive)\n")
A(f"{r51['n_progressor_endoscopies']} progressor pre-event endoscopies. An endoscopy is falsely cleared in a repeat when every sample is triage-negative.\n")
A("| Time to event | Progressor endoscopies | Falsely cleared, mean per repeat | In ≥ 1 repeat | In ≥ 6 of 10 | Grade of those cleared in ≥ 1 repeat (NDBE / ID / LGD) |\n|---|---|---|---|---|---|")
for k, u in r51["by_time_to_event"].items():
    gg = u["grade_of_endoscopies_cleared_in_any_repeat"]; A(f"| {k} | {c5(u['progressor_endoscopies'])} | {u['falsely_cleared_mean_over_repeats']:.1f} | {c5(u['falsely_cleared_in_any_repeat'])} | {c5(u['falsely_cleared_in_majority'])} | {c5(gg['NDBE'])} / {c5(gg['ID'])} / {c5(gg['LGD'])} |")
mp_ = r51["progressor_patients_missed_in_at_least_one_repeat"]
if mp_["n"]:
    A(f"\nProgressor patients missed in ≥ 1 repeat: {c5(mp_['n'])} of {mp_['n_progressor_patients']}. Pre-event endoscopies median {mp_['n_pre_event_endoscopies']['median']:.0f} (range {mp_['n_pre_event_endoscopies']['min']}–{mp_['n_pre_event_endoscopies']['max']}); highest grade " + ", ".join(f"{k} {c5(v)}" for k, v in mp_["max_grade"].items()) +
      "; tissue tertile " + ", ".join(f"T{k} {c5(v)}" for k, v in mp_["tissue_tertile"].items()) + f"; years from first pre-event sample to event median {mp_['years_first_sample_to_event']['median']:.1f} (range {mp_['years_first_sample_to_event']['min']:.1f}–{mp_['years_first_sample_to_event']['max']:.1f}); repeats missed median {mp_['repeats_missed']['median']:.0f} (range {mp_['repeats_missed']['min']}–{mp_['repeats_missed']['max']}).")

# ---------------------------------------------------------------- R52 / R54
def refit_tab(vs):
    A(TRIH)
    A(trirow("primary (stored, R33 row 1)", prim))
    for v, lab in vs: A(trirow(lab, RF[v]))
    A("\n| Analysis | AUROC L-CNV [95% CI] | L-IMG | L-LATE | Patients missed / flagged: triage | B\\* | (B) | C\\* |\n|---|---|---|---|---|---|---|---|")
    for v, lab in vs:
        o = RF[v]; au = o["auroc"]; t = o["table1"]["rules"]; pm = lambda k: f"{t[k]['progressor_patients_missed']:.1f} / {t[k]['non_progressor_patients_flagged']:.1f}"
        A(f"| {lab} | {au['L-CNV']['value']:.3f} {cip(au['L-CNV']['ci95'])} | {au['L-IMG']['value']:.3f} {cip(au['L-IMG']['ci95'])} | {au['L-LATE']['value']:.3f} {cip(au['L-LATE']['ci95'])} | {pm('T')} | {pm('B*')} | {pm('B')} | {pm('C*')} |")
A("\n### R52. Samples of the 8 progressors previously reported without a slide\n")
pa, pb = prep["r52a"], prep["r52b"]
A(f"(a) adds {pa['added_samples']} samples ({pa['added_pre_event_samples']} pre-event) from {c5(pa['added_patients'])} patients: {pa['samples']} samples; pre-event {pa['pre_event_samples']} / {pa['pre_event_patients']} / {pa['pre_event_progressor_patients']} progressors. (b) adds {pb['added_samples']} ({pb['added_pre_event_samples']} pre-event) from {pb['added_patients']} patients: {pb['samples']} samples; pre-event {pb['pre_event_samples']} / {pb['pre_event_patients']} / {pb['pre_event_progressor_patients']}. Every pipeline step refitted; added patients assigned to folds by the pre-specified rule.\n")
refit_tab((("r52a", "(a) + same case and block"), ("r52b", "(b) + block unrecorded")))
A("\n### R53. Event dates from the Barrett's database\n")
s_ = r53["event_date_source_progressor_patients"]; dg = r53["diagnostics_pre_event_progressor_samples"]; p_ = r53["primary_on_or_after"]; w_ = r53["same_endoscopy_within_14_days_or_after"]
A(f"Progressor event date = earlier of the sheet event endoscopy and the first database HGD+ report. Progressor patients: both dates, database earlier {s_['both_db_earlier']}; both, sheet earlier or same day {s_['both_sheet_earlier_or_same']}; sheet only {c5(s_['sheet_only'])}; database only {s_['db_only']}; not linked {c5(s_['not_linked'])}. "
  f"Pre-event progressor samples dated on or after the sheet event date: {c5(dg['dated_on_or_after_sheet_event_date'])}; on or after the first database HGD+ report: {c5(dg['dated_on_or_after_db_first_hgdplus'])} (more than 14 days after: {c5(dg['dated_more_than_14_days_after_db_first_hgdplus'])}; within 14 days: {c5(dg['dated_within_14_days_of_db_first_hgdplus'])}). Stored predictions re-scored.\n")
A(TRIH); A(trirow("primary (stored, R33 row 1)", prim))
A(trirow(f"pre-event samples on or after the event date removed ({p_['n_removed_samples']} samples, {c5(p_['n_removed_patients_affected'])} patients)", p_))
A(trirow(f"same, plus samples within 14 days before a database report ({w_['n_removed_samples']} samples)", w_))
A("\n### R54. Later disease in non-progressors\n")
p4 = prep["r54"]
A(f"Barrett's database, confirmed grade more than 14 days after the last sequenced endoscopy ({p4['linked']} of {p4['non_progressor_patients']} non-progressor patients linked): {c5(p4['relabelled_later_HGDplus'])} relabelled progressor (later HGD+; the same patient as the stored check: {c5(p4['relabelled_also_in_stored_check'])}), {c5(p4['excluded_later_LGD'])} excluded (later LGD; {p4['excluded_samples']} samples). Pre-event {p4['pre_event_samples']} samples / {p4['pre_event_patients']} patients / {p4['pre_event_progressor_patients']} progressors. Refitted.\n")
refit_tab((("r54", "later HGD relabelled, later LGD excluded"),))

# ---------------------------------------------------------------- R55
A("\n### R55. Image quality and within-class confound audit\n")
bg = r55["a_background"]; q_ = lambda u: f"{u['median']:.3f} ({u['iqr'][0]:.3f}–{u['iqr'][1]:.3f}); > 0.25: {c5(u['n_above_0.25'])}"
A(f"**(a) Background tiles.** {bg['slides']['n']} slides; slides with background share > 0.25: {bg['slides']['n_slides_above_0.25']} ({bg['slides']['share_slides_above_0.25']:.1%}). Background share per pre-event sample, median (IQR):\n")
A("| Stratum | Progressor | Non-progressor |\n|---|---|---|")
A(f"| all pre-event | {q_(bg['pre_event_samples_by_class']['progressor'])} (n {bg['pre_event_samples_by_class']['progressor']['n']}) | {q_(bg['pre_event_samples_by_class']['non_progressor'])} (n {bg['pre_event_samples_by_class']['non_progressor']['n']}) |")
for k, u in bg["pre_event_samples_by_tissue_tertile"].items(): A(f"| tissue {k.replace('_', ' ')} | {q_(u['progressor'])} (n {u['progressor']['n']}) | {q_(u['non_progressor'])} (n {u['non_progressor']['n']}) |")
ex = bg["excluding_samples_above_0.25"]
A(f"\nExcluding samples with background share > 0.25 ({ex['n_removed']} removed, {c5(ex['n_removed_progressor'])} progressor), stored calls:\n")
A(TRIH); A(trirow("primary (stored, R33 row 1)", prim)); A(trirow("background share ≤ 0.25", ex))
sa = r55["b_slide_age"]; sp = sa["spearman_L_IMG_within_class"]
A(f"\n**(b) Slide age at scan** (years): progressor median {sa['pre_event_by_class']['progressor']['median']:.1f} (IQR {sa['pre_event_by_class']['progressor']['q1']:.1f}–{sa['pre_event_by_class']['progressor']['q3']:.1f}), non-progressor {sa['pre_event_by_class']['non_progressor']['median']:.1f} ({sa['pre_event_by_class']['non_progressor']['q1']:.1f}–{sa['pre_event_by_class']['non_progressor']['q3']:.1f}). AUROC of slide age alone (younger = higher): {sa['auroc_slide_age_alone']['value']:.3f} {cip(sa['auroc_slide_age_alone']['ci95'])}.\n")
A("| Within class | Spearman L-IMG vs slide age | vs tissue tiles | vs biopsy year |\n|---|---|---|---|")
for k, lab in (("progressor", "progressor"), ("non_progressor", "non-progressor")): A(f"| {lab} (n {sp[k]['n']}) | {sp[k]['slide_age']:+.3f} | {sp[k]['tissue_tiles']:+.3f} | {sp[k]['biopsy_year']:+.3f} |")
cm_ = r55["c_metadata_alone"]; di = r55["d_incremental"]
A(f"\n**(c) Metadata alone** (slide age, log tissue tiles, scanner, batch composition; {cm_['batch_groups']['single_class_batches']} single-class and {cm_['batch_groups']['mixed_batches']} mixed batches): out-of-fold AUROC {cm_['auroc']:.3f} {cip(cm_['ci95'])}.\n")
A(f"**(d) Incremental value.** L-IMG + metadata {di['auroc_L_IMG_plus_metadata']:.3f} {cip(di['ci95'])} (L-IMG alone {di['auroc_L_IMG_alone']:.3f}); Δ (L-IMG + metadata) − metadata {di['d_L_IMG_plus_metadata_minus_metadata']['delta']:+.3f} {ci(di['d_L_IMG_plus_metadata_minus_metadata']['ci95'])} (bound {di['d_L_IMG_plus_metadata_minus_metadata']['one_sided_95_lower']:+.3f}); Δ (L-IMG + metadata) − L-IMG {di['d_L_IMG_plus_metadata_minus_L_IMG']['delta']:+.3f} {ci(di['d_L_IMG_plus_metadata_minus_L_IMG']['ci95'])}.")

# ---------------------------------------------------------------- R56
A("\n### R56. Killcoyne validation cohort: audit only\n")
ai = r56["a_identity"]; bl = r56["b_linkage"]; cm6 = r56["c_metadata_alone"]; gt = r56["gate"]
A("No outcome-based score of the slide model was computed.\n")
A(f"**(a) Identity.** The sheet holds {ai['validation_sheet']['samples']} samples from {ai['validation_sheet']['patients']} patients. {ai['validation_samples_in_release']} of its samples ({ai['validation_patients_with_a_sample_in_release']} patients, {ai['share_validation_patients_in_release_batch']:.0%} of its patients) are release rows; the release holds {ai['release_rows_from_validation_folders']['rows']} rows from the validation copy-number folders ({ai['release_rows_from_validation_folders']['patients']} patients). It is therefore {'the same source as the' if ai['is_the_rejected_batch'] else 'distinct from the'} release validation batch rejected as an external test. Patients overlapping the 80: {c5(ai['validation_patients_overlapping_the_80'])}.\n")
pb_ = bl["patients_with_slide_by_status"]; sb_ = bl["samples_with_slide_by_status"]
A(f"**(b) Linkage.** {bl['samples_with_slide']} samples with ≥ 1 same-case, same-block embedded slide ({bl['slides']} slides): progressor {sb_.get('P', 0)} samples / {pb_.get('P', 0)} patients, non-progressor {sb_.get('NP', 0)} / {pb_.get('NP', 0)}. Sequencing runs {bl['sequencing_runs_SLXID1']} (progressor {bl['sequencing_runs_by_status']['P']}, non-progressor {bl['sequencing_runs_by_status']['NP']}; {bl['runs_with_both_classes']} with both classes). "
  f"Scanner: progressor {', '.join(f'{k} {c5(v)}' for k, v in bl['scanner_by_status']['P'].items())}; non-progressor {', '.join(f'{k} {c5(v)}' for k, v in bl['scanner_by_status']['NP'].items())}. Tissue tiles median (IQR): progressor {bl['tissue_tiles_by_status']['P']['median']:.0f} ({bl['tissue_tiles_by_status']['P']['iqr'][0]:.0f}–{bl['tissue_tiles_by_status']['P']['iqr'][1]:.0f}), non-progressor {bl['tissue_tiles_by_status']['NP']['median']:.0f} ({bl['tissue_tiles_by_status']['NP']['iqr'][0]:.0f}–{bl['tissue_tiles_by_status']['NP']['iqr'][1]:.0f}).\n")
A(f"**(c) Metadata alone** (scanner, sequencing run, tissue tiles, biopsy year) for status: out-of-fold AUROC {cm6['auroc']:.3f} {cip(cm6['ci95'])}, one-sided lower bound {cm6['one_sided_95_lower']:.3f} ({cm6['n_samples']} samples, {cm6['n_patients']} patients, {cm6['n_progressor_patients']} progressor).\n")
A("| Gate condition | Met |\n|---|---|")
for k, lab in (("not_the_rejected_batch", "not the rejected batch"), ("at_least_10_progressor_patients_with_slides", "≥ 10 progressor patients with slides"), ("metadata_auroc_point_below_0.65", "metadata AUROC point estimate < 0.65"), ("metadata_auroc_one_sided_lower_95_below_0.65", "metadata AUROC lower 95% bound < 0.65")): A(f"| {lab} | {yn(gt[k])} |")
A(f"\nGate {'passed: a later commit may pre-register the patient-level check' if gt['passes'] else 'not passed: no check of the slide model on this cohort is pre-registered'}.")

# ---------------------------------------------------------------- R57
A("\n### R57. Achievable precision (simulated from the discovery cohort; no ACE-B data read)\n")
pr_ = r57["a_precision"]; pp_ = r57["b_pass_rule_ge_12_of_15"]; sm_ = r57["c_slide_model_patient_auroc"]
A(f"{r57['draws']:,} draws of 15 progressor and 67 non-progressor patients, 2 pre-event samples each.\n")
A("| Quantity | Mean | CI half-width | 2.5–97.5% over draws |\n|---|---|---|---|")
for k, lab in (("sens", "per-sample sensitivity (triage)"), ("spec", "per-sample specificity (triage)")): u = pr_[k]; A(f"| {lab} | {u['mean']:.3f} | {u['ci_half_width']:.3f} | {u['range_2.5_97.5'][0]:.3f}–{u['range_2.5_97.5'][1]:.3f} |")
A(f"| per-patient AUROC (slide model, maximum over the 2 samples) | {sm_['mean']:.3f} | {sm_['ci_half_width']:.3f} | {sm_['range_2.5_97.5'][0]:.3f}–{sm_['range_2.5_97.5'][1]:.3f} |")
bp = pp_["binomial_probability"]
A(f"\n**(b) Pass rule '≥ 12 of 15 progressor patients detected'.** Simulated probability {pp_['simulated_probability']:.3f} (per-patient detection rate with 2 samples {pp_['mean_detection_rate_in_simulation']:.3f}; cohort rate with all samples {pp_['cohort_per_patient_detection_rate_all_samples']:.3f}). Binomial probability at a true per-patient detection rate of 0.70: {bp['0.70']:.3f}; 0.80: {bp['0.80']:.3f}; 0.90: {bp['0.90']:.3f}.")

# ---------------------------------------------------------------- R58
A("\n### R58. 4× copy number of the same cohort\n")
A("Not run: data not yet available. No 4× data for this cohort exist on the cluster (docs/paper_horizon_answers.md); the analysis stays registered as written above.")

# ---------------------------------------------------------------- checks
A("\n### Checks\n")
oc, oi = chk["outer_cnv_their_rep_01_vs_stored_cfg_00"], chk["outer_img_rep_01_vs_stored_cfg_02"]; icn, iim = chk["inner_cnv_their_rep_01_10_folds_vs_stored"], chk["inner_img_rep_01_10_folds_vs_stored"]
A(f"- Refit pipeline with no change (`rv7_refit.R`, VAR=base, repeat 1) reproduces the stored predictions: outer L-CNV {oc['n']} rows, max |Δ prediction| {oc['max_abs_pred_diff']:.1e}; outer L-IMG {oi['n']} rows, {oi['max_abs_pred_diff']:.1e}; inner L-CNV 10 splits, {icn['n']} rows, {icn['max_abs_pred_diff']:.1e}; inner L-IMG {iim['n']} rows, {iim['max_abs_pred_diff']:.1e}; folds and λ identical: {yn(chk['all_passed'])}.")
A(f"- Stored predictions through the round-7 scoring path reproduce R33 row 1: triage − B\\* {prim['vs_Bstar']['d_sens']:+.3f} (bound {prim['vs_Bstar']['one_sided_lower']:+.3f}), triage − (B) {prim['vs_B']['d_sens']:+.3f}, L-IMG − L-CNV {prim['IMG_minus_CNV']['delta']:+.3f} (expected −0.016 / −0.032, +0.033, +0.045).")
A(f"- Added samples: (a) {pa['added_samples']} → {pa['samples']} samples, pre-event {pa['pre_event_samples']} / {pa['pre_event_patients']} / {pa['pre_event_progressor_patients']} (pre-specified 693; 577 / 77 / 34); (b) {pb['added_samples']} → {pb['samples']}, {pb['pre_event_samples']} / {pb['pre_event_patients']} / {pb['pre_event_progressor_patients']} (pre-specified 700; 582 / 79 / 36). Their slide-mean image rows use the existing UNI2-h tiles; the km_prep.py construction reproduces stored rows to {prep['img_mean_reproduction_max_abs_diff_5_samples']:.1e}; their copy-number rows come from the shipped matrix (stored block reproduced to {prep['their_features_vs_cnv_their_C_max_abs_diff']:.1e}).")
A(f"- R38 evaluation-only rows: triage sensitivity " + "; ".join(f"{k}: {v:.3f}" for k, v in chk["r38_triage_sensitivity_by_row"].items()) + " (the (b) and (c) rows equal the primary 0.865, as expected when only non-progressor samples are removed).")
A(f"- R43 removals equal R38's: (a) {prep['r43a']['pre_event_samples']}, (b) {prep['r43b']['pre_event_samples']}, (c) {prep['r43c']['pre_event_samples']} pre-event samples (R38: 478, 456, 564).")
A(f"- R45 and R44 restate R37: grade-or-p53 {g['sample']['sensitivity']:.3f} / {g['sample']['specificity']:.3f}, triage − grade-or-p53 {to['sample']['d_sens_minus_grade_or_p53']['delta']:+.3f} (R37: 0.432 / 0.794, +0.439); low-band p53-aberrant per repeat {lo['progressor']['p53_aberrant_mean_over_repeats']:.1f} / {lo['non_progressor']['p53_aberrant_mean_over_repeats']:.1f} (R37 e: 0.6 / 0.7). R46 own-threshold rule {a['random_own_threshold']['endoscopy_sensitivity']['mean_over_seeds']:.3f} / {a['random_own_threshold']['endoscopy_specificity']['mean_over_seeds']:.3f} (R41: 0.827 / 0.740). R48 triage {r48['triage_reproduction_expected_0.865_0.697'][0]:.3f} / {r48['triage_reproduction_expected_0.865_0.697'][1]:.3f}.")
A("- R55 tissue tiles recomputed from the masks equal the stored `slide_desc.csv` values on the first pixel chunk (18 of 18 samples) and the stored tile coordinates are level coordinates (tile centres fall in the tissue mask).")

# ---------------------------------------------------------------- implementation notes
A("\n### Implementation notes\n")
NOTES = [
 "General: stored predictions and cut-offs are scored with the rv6_round6.py helpers (copied unchanged into rv7_round7.py). 'Single model' everywhere; endoscopy = (patient, endoscopy number), positive if any sample is flagged; patient positive if any sample is flagged. Patient bootstraps that need endoscopy or patient units tag resampled copies as distinct units (R23).",
 "Refits (R43, R52, R54): `rv7_refit.R` runs the kv_cv.R outer loop and the hz_fit.R inner loop verbatim on a variant set; only the set changes. Existing patients keep their set-C fold in every repeat, for the outer folds and also for the inner 5-fold splits (the inner split of the stored set C, 1000 × REP + FOLD); added patients get a fold by the pre-specified rule (fewest patients of their class, sorted order, ties to the lowest index), applied separately to the outer folds and to each inner split. The λ-selection shuffles (shuffle_groups, unchanged code) are drawn on the variant's patients, so they differ from the stored ones when patients are added or removed. A few refit files (base repeat 1, and one unit each of r52b and r54) were produced by the smoke test with the committed code and kept.",
 "R42: 126 = the 252 ordered 5/5 splits counted once (repeat 1 always in the first half); 5-model vote = ≥ 3 of 5; modal band as R39.",
 "R43: the R38 definitions are reused exactly (months before final < 12 / < 24 for non-progressor samples; the later-HGD patient of the stored hospital-record check, K1 logic), so (a)/(b) remove samples but no patient and (c) removes one patient.",
 "R44: triage-negative = band low, or band middle with L-LATE below the fold t_seq, per repeat; reported as per-repeat means with the range over repeats.",
 "R45: evaluation-matched threshold per repeat, pooled over folds, on the 450 samples (as the evaluation-matched B\\* of R33): positive if score > the ⌈s·n⌉-th smallest non-progressor score; re-matched in each bootstrap draw to grade-or-p53's specificity in that draw. Per patient: patient score = maximum over its samples with p53; grade-or-p53 per patient = any sample flagged.",
 "R46: the R41 seed loop (RandomState(seed), seeds 0–99, endoscopies in sorted order) is reproduced to pick the same samples; the matched threshold is one L-LATE threshold per seed × repeat pooled across folds, set to the triage's endoscopy specificity in that repeat (and re-set in each draw). 'Triage share of endoscopies sequenced' = endoscopies with ≥ 1 middle-band sample, per repeat.",
 "R47: non-progressor patient-years = sum over non-progressor patients of the span between their first and last pre-event endoscopy (months before final / 12); a patient with one endoscopy contributes 0. Projections use per-sample class-conditional rates (sensitivity, false-positive rate, share sequenced by class); flagged endoscopies assume samples within an endoscopy flag independently, with the cohort's samples-per-endoscopy distribution. Grade-or-p53 treats missing p53 as wild-type on all 571 samples (R37 f). (C), (D) and grade-or-p53 sequence nothing. No uncertainty is attached (descriptive).",
 "R48: high-band samples are sequenced and called at the triage's middle-band threshold t_seq of their fold.",
 "R49: the endoscopy's fold is the patient's (all samples share it); its score is compared with that fold's t_low and c\\* (high band = score > c\\*, as the sample-level rule); when t_low > c\\* there is no middle band. Share of samples sequenced = sequenced samples / pre-event samples, per repeat.",
 "R50: cut-offs and matched comparators chosen on NDBE pre-event training samples, as R33 row 2 and the R39 'ndbe' row; Table 1 columns from those calls.",
 "R51: endoscopy time to event = its samples' time; < 1 year, 1–3 years (inclusive), > 3 years. 'Missed patient' = no pre-event sample flagged in that repeat. Tissue tertile = tertile of the patient's median tissue tiles against pre-event sample tertile edges.",
 "R52: tiers T1 and T2b of `a2_candidate_map_8pt.csv`; all their samples (pre- and post-event) are added, as for every other patient in set C. Pre-event, time and NDBE flags follow hz_prep.py (event = the patient's first HGD/IMC sample in kr_samples.csv or the sheet's per-endoscopy class). Slide-mean image rows = mean over the union of the sample's stored UNI2-h tiles (km_prep.py); for T2b samples with several candidate slides of the case, all of them.",
 "R53: Barrett's database linkage and grade coding copied from clinical_lookup/r2_main.py; sheet event date = earliest exact date of the patient's event samples; sample dates = pre_event_cohort.Date (all 571 dated). 'On or after' is taken literally (sample date ≥ event date); a second row also removes samples dated up to 14 days before the event (the lookup's same-endoscopy convention). The database report date is the specimen-received date. Calls, cut-offs and comparators are the stored ones (no refit).",
 "R54: later = confirmed grade in a pathology report (or an HGD-register entry) more than 14 days after the patient's last sequenced endoscopy (latest exact date among all of the patient's sheet samples, slide-table date as fallback), as the lookup's B2. HGD+ → relabelled progressor (all its samples, all pre-event); highest later grade LGD → patient excluded.",
 "R55 (a): the 256 stored tile positions are read at the stored level and tile size; a sample's share is pooled over its slides; samples (not single slides) above 0.25 are excluded, because re-scoring cannot drop one slide of a sample. (b): scan date = hamamatsu.Created (earliest over the sample's slides); L-IMG = mean of the 10 outer predictions; slide-age AUROC uses the stored folds (fold-stratified, fixed score, sign chosen so younger slides score higher). (c)/(d): unpenalised logistic regression on training-standardised features; batch composition = whether the sample's sheet batch contains one class only among the 676; training rows = the stored inner rows of each outer split, where L-IMG enters as its inner out-of-fold prediction (and as its outer prediction on the held-out fold), so no fold sees its own test data.",
 "R56: 'the rejected batch' = the release's 268-sheet stratum (`docs/dataset_description.md`: release rows whose CNV sample is in the validation sheet); the sheet counts as that batch if any of its patients has a release row. Slide link = case + block (lookup functions ckey / vblock) with an embedded slide in uni2_npz_map.csv. The sheet has no Batch column, so 'batch' = sequencing run (SLXID1). Metadata model: L2 logistic (C = 1, because the one-hot run levels are many for the sample size), patient-grouped folds stratified by patient status; status = the sample's sheet Status (P / NP only). The gate's 'lower 95% bound' is the one-sided 5th percentile, as elsewhere.",
 "R57: patients drawn with replacement (67 > 43 non-progressor patients available); one random repeat per draw; CI half-width = half the 2.5–97.5% range of the estimate over draws (the spread an ACE-B-sized study would see), not a within-study interval. The pass-rule probability at a given true detection rate is binomial (15 patients).",
 "R58: not run; no 4× data exist.",
]
for t in NOTES: A(f"- {t}")
txt = open(DOC).read(); head = txt[:txt.rindex("\n---") + 4]
open(DOC, "w").write(head + "\n" + "\n".join(L) + "\n"); print("RENDERED", len(L), "lines")

"""Append results to docs/paper_review_round2_analyses.md below the pre-specification (d681783); text above the closing '---' line is not changed.
Usage: python rv2_render.py. Inputs: results/paper_final/review_round2/*.json (Part A r14-r30 from rv2_round2.py; Part B check, f1-f6, f3_draws,
f5_r28, f6_r25b, r21b from rv2_refit_metrics.py). Reads JSONs only."""
import json
RC_B, RC_A = "41f84cb", "c8934b6"; R = "results/paper_final/review_round2"; DOC = "docs/paper_review_round2_analyses.md"
def J(f): return json.load(open(f"{R}/{f}.json"))
def f3(x): return "—" if x is None else (str(x) if isinstance(x, int) and not isinstance(x, bool) else f"{x + 0.0 if x else 0.0:.3f}")   # -0.0 -> 0.000
def ci(c): return "" if not c or c[0] is None else f" [{c[0]:.3f}, {c[1]:.3f}]"
def sg(x): return "—" if x is None else f"{x if x else 0.0:+.3f}"
def cis(c): return "" if not c or c[0] is None else f" [{sg(c[0])}, {sg(c[1])}]"
def rg(c): return f"{f3(c[0])}–{f3(c[1])}"
def rgs(c): return f"{sg(c[0])} to {sg(c[1])}"   # signed range
def ni(lo): return "non-inferior at −0.05" if lo is not None and lo > -0.05 else "not non-inferior at −0.05"
def ss(v, sk="sensitivity", sc="sens_ci95", pk="specificity", pc="spec_ci95"): return f"{f3(v[sk])}{ci(v.get(sc))} / {f3(v[pk])}{ci(v.get(pc))}"
def dd(v, one=False, tost=False):   # delta [ci]; one-sided lower; TOST 90% CI
    s = f"{sg(v['delta'])}{cis(v['ci95'])}"
    if one: s += f"; lower {f3(v.get('one_sided_95_lower'))}"
    if tost and "ci90_tost" in v: s += f"; 90% CI{cis(v['ci90_tost'])}"
    return s
def nm(s): return "triage" if s == "T" else s
POPL = {"their_pre": "their, all pre-event", "their_pre_ndbe": "their, NDBE pre-event", "pkg_pre": "package, all pre-event"}
PL = {"pre": "all pre-event", "pre_ndbe": "NDBE pre-event"}; SL = {"their": "their", "pkg": "package"}
doc = open(DOC).read(); head = doc[: doc.rindex("\n---")].rstrip("\n") + "\n\n---\n"
L = ["", "## Results", "", f"Pre-specification commit d681783; results commits {RC_B} (Part B) and {RC_A} (Part A). Scripts `scripts/paper_plan/rv2_round2.py` (Part A), `rv2_refit.R` and `rv2_refit_metrics.py` (Part B), run on Slurm via `scripts/cluster/campaign.sh` (prefix rv2); renderer `rv2_render.py`. Aggregates `results/paper_final/review_round2/`.", ""]
# ------------------------------------------------------------------ status
ST = [("R14 cost of the vote; no-vote comparisons", "DONE"), ("R15 between-repeat variability", "DONE"), ("R16 selection-inclusive intervals", "DONE"), ("R17 multiplicity", "DONE"), ("R18 tissue amount", "DONE"),
      ("R19 scanner fallback", "DONE"), ("R20 patients without a slide", "DONE"), ("R21 leave-one-patient-out bias", "DONE ((b) from stored predictions; R21b)"), ("R22 package-feature tables", "DONE"), ("R23 endoscopy and patient level", "DONE"),
      ("R24 current care, grade, sampling", "DONE"), ("R25 uninformative gate", "DONE ((a) random gate; (b) = F6)"), ("R26 cut-off transfer", "DONE (fold coefficient counts not stored)"), ("R27 pairs and time-dependent AUROC", "DONE"), ("R28 H-optimus-0", "DONE (triage = F5)"),
      ("R29 likelihood ratios", "DONE"), ("R30 lead time", "DONE"), ("F1 pre-event-only training", "DONE"), ("F2 training-size control", "DONE"), ("F3 refit bootstrap", "DONE (200 draws)"), ("F4 λ by deviance", "DONE"),
      ("F5 H-optimus-0 inner predictions", "DONE"), ("F6 permutation-trained gate", "DONE"), ("F7 leave-one-patient-out L-CNV", "NOT RUN (stored predictions exist; R21b)")]
L += ["### Status", "", "| Item | Status |", "|---|---|"] + [f"| {k} | {v} |" for k, v in ST] + [""]
# ------------------------------------------------------------------ R14
R14 = {k: J(f"r14_{k}")["R14"] for k in POPL}
L += ["### R14. Cost of the vote and the comparisons without the vote", "", "(a) Sequencing needed (share of samples; patient-bootstrap 95% CI).", "", "| Population | Samples (progressor) | Per-repeat mean middle band | Middle band in ≥ 1 repeat | Vote-pivotal | Progressor samples: mean / ≥ 1 repeat / pivotal | Non-progressor samples: mean / ≥ 1 repeat / pivotal |", "|---|---|---|---|---|---|---|"]
for k, x in R14.items():
    a = x["a_sequencing_cost"]; tri = lambda sfx: " / ".join(f3(a[f"{q}{sfx}"]["value"]) for q in ("share_mid", "share_mid_any", "share_pivotal"))
    L.append(f"| {POPL[k]} | {x['n_samples']} ({x['n_progressor_samples']}) | {f3(a['share_mid']['value'])}{ci(a['share_mid']['ci95'])} | {f3(a['share_mid_any']['value'])}{ci(a['share_mid_any']['ci95'])} | {f3(a['share_pivotal']['value'])}{ci(a['share_pivotal']['ci95'])} | {tri('_P')} | {tri('_N')} |")
L += ["", "Triage operating points (sensitivity / specificity).", "", "| Population | No vote | Voted |", "|---|---|---|"] + [f"| {POPL[k]} | {ss(x['triage_no_vote'])} | {ss(x['triage_voted'])} |" for k, x in R14.items()]
L += ["", "(b) No-vote paired comparisons (training-matched A\\*, B\\*, C\\* as in R1, thresholds per repeat; one-sided lower bound where pre-specified).", "", "| Population | Comparator | Comparator sens / spec | Δ sens triage − comparator [95% CI]; lower | Δ spec [95% CI] |", "|---|---|---|---|---|"]
for k, x in R14.items():
    for s, v in x["b_no_vote_paired"].items():
        L.append(f"| {POPL[k]} | {s} | {ss(v, 'comparator_sensitivity', 'comparator_sens_ci95', 'comparator_specificity', 'comparator_spec_ci95')} | {dd(v['d_sens_triage_minus'], one='one_sided_95_lower' in v['d_sens_triage_minus'])} | {dd(v['d_spec_triage_minus'])} |")
L += ["", "(c) Evaluation-matched comparators without the vote (threshold re-chosen in every bootstrap draw).", "", "| Population | Comparator | Comparator sens at the triage's realised specificity | Δ sens [95% CI]; lower |", "|---|---|---|---|"]
for k, x in R14.items():
    for s in ("A*", "B*", "C*"):
        v = x["c_no_vote_evaluation_matched"][s]; L.append(f"| {POPL[k]} | {s} | {f3(v['comparator_sensitivity'])}{ci(v['comparator_sens_ci95'])} | {dd(v['d_sens_triage_minus'], one=True)} |")
b = R14["their_pre"]["b_no_vote_paired"]; bB, bS = b["B"]["d_sens_triage_minus"], b["B*"]["d_sens_triage_minus"]
L += ["", f"One line: without the vote (their matrix, all pre-event) Δ sensitivity is {sg(bB['delta'])} against (B) (lower {f3(bB['one_sided_95_lower'])}, {ni(bB['one_sided_95_lower'])}) and {sg(bS['delta'])} against B\\* (lower {f3(bS['one_sided_95_lower'])}, {ni(bS['one_sided_95_lower'])}); {f3(R14['their_pre']['a_sequencing_cost']['share_pivotal']['value'])} of samples are vote-pivotal under the voted rule.", ""]
# ------------------------------------------------------------------ R15
r15 = J("r15")["R15"]; L += ["### R15. Between-repeat variability (their matrix, all pre-event)", "", "| Statistic | Repeat mean | SD over repeats | Range | Repeats with own lower bound > −0.05 | Standard one-sided lower | Variance-inflated lower (V_boot, V_rep) |", "|---|---|---|---|---|---|---|"]
for k, v in r15.items():
    L.append(f"| {k.replace('_', ' ')} | {sg(v['delta_repeat_mean'])} | {f3(v['sd_over_repeats'])} | {sg(v['range_over_repeats'][0])} to {sg(v['range_over_repeats'][1])} | {v['n_repeats_lower_above_-0.05']} / 10 | {f3(v['standard_one_sided_lower'])} | {f3(v['variance_inflated_one_sided_lower'])} ({v['V_boot']:.6f}, {v['V_rep']:.6f}) |")
v = r15["dAUROC_LIMG_minus_LCNV"]; w = r15["no_vote_d_sens_triage_minus_Bstar"]
L += ["", f"One line: with between-repeat variance added, ΔAUROC L-IMG − L-CNV has lower bound {f3(v['variance_inflated_one_sided_lower'])} ({ni(v['variance_inflated_one_sided_lower'])}) and the no-vote Δ sensitivity against B\\* has lower bound {f3(w['variance_inflated_one_sided_lower'])} ({ni(w['variance_inflated_one_sided_lower'])}).", ""]
# ------------------------------------------------------------------ R16
L += ["### R16. Selection-inclusive intervals (every cut-off re-chosen in each draw; 2,000 draws)", "", "| Population | Mode | Comparator | Triage sens / spec | Comparator sens / spec | Δ sens [95% CI]; lower; 90% CI (TOST) | Δ spec [95% CI] |", "|---|---|---|---|---|---|---|"]
R16 = {k: J(f"r16_{k}")["R16"] for k in ("their_pre", "their_pre_ndbe")}
for k, x in R16.items():
    for m in ("voted", "no_vote"):
        t = x["selection_inclusive_training_matched"][m]; tri = ss(t["triage"]); ml = m.replace("_", " ")
        for s, lab in (("A*", "A* (training-matched)"), ("B*", "B* (training-matched)"), ("C*", "C* (training-matched)"), ("vs_B", "(B)")):
            v = t[s]; L.append(f"| {POPL[k]} | {ml} | {lab} | {tri} | {ss(v, 'comparator_sensitivity', 'comparator_sens_ci95', 'comparator_specificity', 'comparator_spec_ci95')} | {dd(v['d_sens_triage_minus'], one=True, tost=True)} | {dd(v['d_spec_triage_minus'])} |")
        for s in ("A*", "B*", "C*"):
            v = x["selection_inclusive_evaluation_matched"][m][s]; L.append(f"| {POPL[k]} | {ml} | {s} (evaluation-matched) | {tri} | {f3(v['comparator_sensitivity'])}{ci(v['comparator_sens_ci95'])} / — | {dd(v['d_sens_triage_minus'], one=True, tost=True)} | — |")
L += ["", "Bias of the primary criterion (Δ sensitivity against (B)) and selection-inclusive CIs of the share sequenced.", "", "| Population | Mode | Observed | Bootstrap median / mean | Bias-corrected (2 × observed − mean) | 95% CI; lower | Share sequenced 95% CI: per-repeat mean / ≥ 1 repeat / vote-pivotal |", "|---|---|---|---|---|---|---|"]
for k, x in R16.items():
    sh = x["share_sequenced_ci95"]
    for m, kk in (("voted", "bias_primary_criterion_voted_vs_B"), ("no vote", "bias_primary_criterion_no_vote_vs_B")):
        v = x[kk]; L.append(f"| {POPL[k]} | {m} | {sg(v['observed'])} | {sg(v['bootstrap_median'])} / {sg(v['bootstrap_mean'])} | {sg(v['bias_corrected_2obs_minus_mean'])} |{cis(v['ci95'])}; {f3(v['one_sided_95_lower'])} |{ci(sh['share_mid'])} /{ci(sh['share_mid_any'])} /{ci(sh['share_pivotal'])} |")
x = R16["their_pre"]; nvB = x["selection_inclusive_training_matched"]["no_vote"]["B*"]["d_sens_triage_minus"]; vB = x["selection_inclusive_training_matched"]["voted"]["B*"]["d_sens_triage_minus"]; chk = x["check_stored_tb_selci_draws"]
L += ["", f"One line: selection-inclusive, the no-vote Δ sensitivity against B\\* has lower bound {f3(nvB['one_sided_95_lower'])} ({ni(nvB['one_sided_95_lower'])}) and the voted one {f3(vB['one_sided_95_lower'])} ({ni(vB['one_sided_95_lower'])}); the recomputed voted draws against (B) match the stored `tb_selci` draws (max abs difference {chk['max_abs_diff_vs_recomputed']}).", ""]
# ------------------------------------------------------------------ R17
L += ["### R17. Multiplicity across A\\*, B\\*, C\\* (Δ sensitivity, triage − comparator)", "", "| Population | Mode, matching | Comparator | Δ [95% CI] | p (percentile) | p Holm | p max-T |", "|---|---|---|---|---|---|---|"]
ML = {"voted_training_matched": "voted, training-matched", "voted_evaluation_matched": "voted, evaluation-matched", "no_vote_training_matched": "no vote, training-matched", "no_vote_evaluation_matched": "no vote, evaluation-matched"}
for k in POPL:
    x = J(f"r17_{k}")["R17"]
    for kk, lab in ML.items():
        for s in ("A*", "B*", "C*"):
            v = x[kk][s]; L.append(f"| {POPL[k]} | {lab} | {s} | {sg(v['delta'])}{cis(v['ci95'])} | {v['p_unadjusted_percentile']} | {v['p_holm']} | {v['p_max_T']} |")
x = J("r17_their_pre")["R17"]["no_vote_training_matched"]; L += ["", f"One line: their matrix, all pre-event, no vote, training-matched: max-T p {x['A*']['p_max_T']} (A\\*), {x['B*']['p_max_T']} (B\\*), {x['C*']['p_max_T']} (C\\*).", ""]
# ------------------------------------------------------------------ R18
r18 = J("r18")["R18"]; L += ["### R18. Tissue amount (their matrix, all pre-event)", "", "| Stratum | Samples (progressor) / patients | L-CNV | L-IMG | L-LATE | Δ L-IMG − L-CNV [95% CI]; lower | Triage voted sens / spec; mid in ≥ 1 repeat; pivotal | Triage no vote sens / spec; share sequenced |", "|---|---|---|---|---|---|---|---|"]
for s, lab in (("overlap_lt256", "tissue tiles < 256 (overlapping tiles)"), ("no_overlap_ge256", "tissue tiles ≥ 256")):
    v = r18["strata"][s]; a = v["auroc"]; tv = v["triage_voted"]; tn = v["triage_no_vote"]
    L.append(f"| {lab} | {v['n_samples']} ({v['n_progressor_samples']}) / {v['n_patients']} | " + " | ".join(f"{f3(a[m]['value'])}{ci(a[m]['ci95'])}" for m in ("L-CNV", "L-IMG", "L-LATE")) + f" | {dd(v['IMG_minus_CNV'], one=True)} | {ss(tv)}; {f3(tv['share_mid_any'])}; {f3(tv['share_pivotal'])} | {ss(tn)}; {f3(tn['share_sequenced'])}{ci(tn['share_ci95'])} |")
it = r18["interaction_dIMGCNV_lt256_minus_ge256"]; tm = r18["tissue_matched_auroc"]; u = r18["unstratified_for_reference"]; d1 = r18["strata"]["overlap_lt256"]["IMG_minus_CNV"]; d2 = r18["strata"]["no_overlap_ge256"]["IMG_minus_CNV"]
L += ["", f"Interaction (Δ in the < 256 stratum minus Δ in the ≥ 256 stratum): {dd(it)} (p {it['p_two_sided']}).", "",
      f"Tissue-matched AUROC (pairs within the same tissue-tile tertile; tertile edges {', '.join(f3(e) for e in tm['tertile_edges_tissue_tiles'])} tiles): " + ", ".join(f"{m} {f3(v['value'])}{ci(v['ci95'])}" for m, v in tm["auroc"].items()) + f"; Δ L-IMG − L-CNV {dd(tm['IMG_minus_CNV'], one=True)} (unstratified reference " + ", ".join(f"{m} {f3(v)}" for m, v in u.items()) + ").", "",
      "Samples by tertile (progressor / non-progressor): " + "; ".join(f"{t.replace('_', ' ')} {v['progressor']} / {v['non_progressor']}" for t, v in tm["counts_by_tertile_and_class"].items()) + ".", "",
      f"One line: the L-IMG advantage is confined to the overlapping-tile stratum ({sg(d1['delta'])}, lower {f3(d1['one_sided_95_lower'])}) and absent in the ≥ 256 stratum ({sg(d2['delta'])}, lower {f3(d2['one_sided_95_lower'])}, {ni(d2['one_sided_95_lower'])}); tissue-matched Δ {sg(tm['IMG_minus_CNV']['delta'])}, lower {f3(tm['IMG_minus_CNV']['one_sided_95_lower'])} ({ni(tm['IMG_minus_CNV']['one_sided_95_lower'])}).", ""]
# ------------------------------------------------------------------ R19
r19 = J("r19")["R19"]; st = r19["setting"]; mx = r19["mixed_scanner_cv_same_samples"]
L += [f"### R19. Scanner fallback on the stored transfer predictions (C13239-01 → C13210; {st['n_samples']} samples, {st['n_patients']} patients, {st['n_P_patients']} progressors; t_seq frozen at {f3(st['t_seq_frozen'])})", "", "| Rule | t_low | c* | Sensitivity [95% CI] | Specificity [95% CI] | Bands low / mid / high |", "|---|---|---|---|---|---|"]
for k, lab in (("transfer_unreset", "transfer, unreset cut-offs"), ("transfer_fallback_reset", f"transfer, label-free reset to the internal low / high shares ({f3(st['internal_target_shares']['low'])} / {f3(st['internal_target_shares']['high'])})")):
    v = r19[k]; bs = v["band_shares"]; L.append(f"| {lab} | {f3(v['cutoffs']['t_low'])} | {f3(v['cutoffs']['c_star'])} | {f3(v['sensitivity'])}{ci(v['sens_ci95'])} | {f3(v['specificity'])}{ci(v['spec_ci95'])} | {f3(bs['low'])} / {f3(bs['mid'])} / {f3(bs['high'])} |")
bs = mx["band_shares_mean_over_repeats"]; L.append(f"| mixed-scanner CV, same samples (voted) | — | — | {f3(mx['sensitivity'])}{ci(mx['sens_ci95'])} | {f3(mx['specificity'])}{ci(mx['spec_ci95'])} | {f3(bs['low'])} / {f3(bs['mid'])} / {f3(bs['high'])} |")
cw = r19["class_wise_band_rates"]; bnd = lambda d: f"{f3(d['low'])} / {f3(d['mid'])} / {f3(d['high'])}"
L += ["", f"Internal class-wise band rates (low / mid / high): progressor {bnd(cw['progressor'])}; non-progressor {bnd(cw['non_progressor'])}.", "", "| Progressor-sample prevalence | Expected low | Expected middle | Expected high |", "|---|---|---|---|"]
L += [f"| {k.replace('observed_', '') + (' (observed)' if k.startswith('observed') else '')} | {f3(v['low'])} | {f3(v['mid'])} | {f3(v['high'])} |" for k, v in r19["expected_shares_by_prevalence"].items()]
tp = r19["trigger_prediction_interval_117_patients"]
L += ["", f"Trigger interval for a 117-patient cohort ({tp['draws']} draws of {tp['n_internal_patients']} internal patients, mean {tp['mean_samples_per_draw']:.1f} samples per draw): " + "; ".join(f"{b} mean {f3(tp[b]['mean'])}, 95% interval{ci(tp[b]['pi95'])}" for b in ("low", "mid", "high")) + ".", "",
      f"One line: the label-free reset moves C13210 sensitivity from {f3(r19['transfer_unreset']['sensitivity'])} to {f3(r19['transfer_fallback_reset']['sensitivity'])} and specificity from {f3(r19['transfer_unreset']['specificity'])} to {f3(r19['transfer_fallback_reset']['specificity'])}, against {f3(mx['sensitivity'])} / {f3(mx['specificity'])} for the mixed-scanner CV on the same samples.", ""]
# ------------------------------------------------------------------ R20
r20 = J("r20")["R20"]; L += [f"### R20. The {r20['n_without_slide']} patients without a slide (all progressors; published P and the sample sheet)", "", "| Group | Samples | Patients | P per sample: mean, median (IQR) | P per patient mean: mean, median (IQR) | Samples per patient: median (IQR) | Grade mix | Share of samples with P ≥ 0.5 |", "|---|---|---|---|---|---|---|---|"]
for g, lab in (("without_slide_8", "without slide"), ("included_progressors_37", "included progressors")):
    for s, sl in (("all_samples", "all samples"), ("pre_event_samples", "pre-event samples")):
        v = r20[g][s]; ps = v["P_per_sample"]; pp = v["P_per_patient_mean"]; sp = v["samples_per_patient"]
        L.append(f"| {lab}, {sl} | {v['n_samples']} | {v['n_patients']} | {f3(ps['mean'])}, {f3(ps['median'])} ({rg(ps['iqr'])}) | {f3(pp['mean'])}, {f3(pp['median'])} ({rg(pp['iqr'])}) | {f3(sp['median'])} ({rg(sp['iqr'])}) | {', '.join(f'{k} {n}' for k, n in v['grade_mix'].items())} | {f3(v['share_P_ge_0.5_samples'])} |")
L += ["", "| Plain per-sample AUROC of P, pre-event samples | Samples / patients (progressors) | AUROC [95% CI] |", "|---|---|---|"]
L += [f"| {k.replace('_', ' ')} | {v['n_samples']} / {v['n_patients']} ({v['n_P_patients']}) | {f3(v['auroc'])}{ci(v['ci95'])} |" for k, v in r20["plain_auroc_P_pre_event"].items()]
sc = r20["reason_no_slide"]["source_columns_checked"]; sm = sc["slide_matching"]; TXT = ("Pathology notes", "inconsistency", "notes on label review", "Set", "Batch")
cols = "; ".join(f"{k} {v['n_nonempty']}" + (f" ({', '.join(v['distinct_values'])})" if v["distinct_values"] and k in TXT else "") for k, v in sc.items() if k != "slide_matching")
pa, pb = r20["without_slide_8"]["pre_event_samples"]["P_per_patient_mean"]["mean"], r20["included_progressors_37"]["pre_event_samples"]["P_per_patient_mean"]["mean"]; au = r20["plain_auroc_P_pre_event"]
L += ["", f"Reason no slide exists: no column of the sample sheet records one. Non-empty rows among the {sm['n_sheet_rows_8_patients']} sheet rows of these patients: {cols}. Slide matching: {sm['n_matched_slide_rows']} matched rows, {sm['n_with_slide_file']} with a slide file. Conclusion: not recorded.", "",
      f"One line: the published P of the 8 patients is of the same order as the included progressors' (pre-event per-patient mean {f3(pa)} against {f3(pb)}), and P's plain AUROC on all 88 patients ({f3(au['all_88_patients']['auroc'])}) is close to that on the analysed samples ({f3(au['80_patients_571_analysed_samples']['auroc'])}).", ""]
# ------------------------------------------------------------------ R21 and R21b
r21 = J("r21")["R21"]; r21b = J("r21b")["R21b"]
L += ["### R21. Leave-one-patient-out bias of the published model", "", f"(a) Intercept-only leave-one-patient-out predictions (`{r21['intercept_only_lopo']['path']}`, {r21['intercept_only_lopo']['n_patients']} patients). (b) Stored leave-one-patient-out L-CNV from our pipeline (`{r21['stored_lopo_L_CNV']['path']}`; λ range {rg(r21['stored_lopo_L_CNV']['lopo_lambda_range'])}), so F7 was not refitted.", "",
      "| Population | Score | Fold-stratified AUROC on the 10 × 10 pairs [95% CI] | Plain pooled AUROC [95% CI] |", "|---|---|---|---|"]
for pop in PL:
    for m, v in r21[pop]["fold_stratified_10x10_pairs"].items():
        p = r21[pop]["plain_pooled_auroc"][m]; L.append(f"| {PL[pop]} ({r21[pop]['n_samples']} samples) | {m.replace('_', ' ')} | {f3(v['value'])}{ci(v['ci95'])} | {f3(p['value'])}{ci(p['ci95'])} |")
L += ["", "| Population | Difference | Δ [95% CI] | p |", "|---|---|---|---|"]
for pop in PL:
    for k, lab in (("a_intercept_only_vs_P", "(a) intercept-only LOPO − P"), ("b_LOPO_minus_10x10_L_CNV", "(b) LOPO L-CNV − 10 × 10 L-CNV, fold-stratified"), ("b_LOPO_minus_10x10_L_CNV_plain", "(b) LOPO L-CNV − 10 × 10 L-CNV, plain pooled"), ("P_minus_10x10_L_CNV", "P − 10 × 10 L-CNV")):
        v = r21[pop][k]; L.append(f"| {PL[pop]} | {lab} | {dd(v)} | {v['p_two_sided']} |")
    L.append(f"| {PL[pop]} | (a) intercept-only LOPO − 0.5 | {sg(r21[pop]['a_intercept_only_vs_0.5_fold_stratified'])} | — |")
def r21b_line(pop):
    x = r21b[pop]; fa = x["fold_stratified_auroc_on_10x10_pairs"]; pa = x["plain_pooled_auroc"]
    return f"{PL[pop]} LOPO {f3(fa['LOPO_L-CNV']['value'])}{ci(fa['LOPO_L-CNV']['ci95'])} against 10 × 10 {f3(fa['CV_L-CNV_10x10']['value'])}{ci(fa['CV_L-CNV_10x10']['ci95'])}, Δ {dd(x['d_LOPO_minus_CV'])} (p {x['d_LOPO_minus_CV']['p_two_sided']}); plain LOPO {f3(pa['LOPO_L-CNV'])} against {f3(pa['CV_L-CNV_repeat_mean_score'])}"
p1 = r21["pre"]; L += ["", "R21b (`rv2_refit_metrics.py`, same stored predictions): " + "; ".join(r21b_line(pop) for pop in PL) + ".", "",
      f"One line: the intercept-only leave-one-patient-out score is perfectly anti-concordant (AUROC {f3(p1['fold_stratified_10x10_pairs']['intercept_only_LOPO']['value'])}), and our leave-one-patient-out L-CNV differs from the 10 × 10 L-CNV on the same pairs by {dd(p1['b_LOPO_minus_10x10_L_CNV'])} (all pre-event), so the leave-one-patient-out design does not account for P − L-CNV ({dd(p1['P_minus_10x10_L_CNV'])}).", ""]
# ------------------------------------------------------------------ R22
r22 = J("r22")["R22"]; L += ["### R22. Package-feature triage (all pre-event samples; reporting of stored results)", "", "| Level | Strategy | Sensitivity [95% CI] | Specificity [95% CI] |", "|---|---|---|---|"]
for s, v in r22["sample_level_voted"].items(): L.append(f"| sample, voted | {s} | {f3(v['sensitivity']['value'])}{ci(v['sensitivity']['ci95'])} | {f3(v['specificity']['value'])}{ci(v['specificity']['ci95'])} |")
for s, v in r22["endoscopy_level_voted"].items(): L.append(f"| endoscopy, voted | {s} | {f3(v['sensitivity'])}{ci(v['sens_ci95'])} | {f3(v['specificity'])}{ci(v['spec_ci95'])} |")
r8 = r22["sample_level_no_vote_R8"]
for s in ("triage_LATE", "A", "B", "C"):
    v = r8[s]; L.append(f"| sample, no vote (mean over repeats, range) | {s} | {f3(v['sensitivity_mean'])} ({rg(v['sensitivity_range'])}) | {f3(v['specificity_mean'])} ({rg(v['specificity_range'])}) |")
m1 = r22["matched_comparators_voted_R1"]; nv14 = r22["no_vote_paired_R14"]["b_no_vote_paired"]
L += ["", "| Matched comparator (voted, R1) | Comparator sens / spec | Δ sens triage − comparator [95% CI] | Δ spec [95% CI] | No-vote Δ sens [95% CI]; lower (R14) |", "|---|---|---|---|---|"]
for s, v in m1["matched"].items():
    nv = nv14[s]["d_sens_triage_minus"]; L.append(f"| {s} ({v['model']}) | {ss(v)} | {dd(v['d_sens_triage_minus'])} | {dd(v['d_spec_triage_minus'])} | {dd(nv, one='one_sided_95_lower' in nv)} |")
pc = r22["primary_criterion"]; pv = r22["patient_level_voted"]; sq = r22["sequenced"]; nvB = nv14["B*"]["d_sens_triage_minus"]
L += ["", "Patient level (voted): " + "; ".join(f"{s} caught {v['progressor_caught']}, missed {v['progressor_missed']}, non-progressors flagged {v['nonprogressor_false_positive']} of {v['nonprogressor_false_positive'] + v['nonprogressor_true_negative']}" for s, v in pv.items()) + ".", "",
      "Bands (all [95% CI] / progressor / non-progressor): " + "; ".join(f"{b} {f3(v['all']['value'])}{ci(v['all']['ci95'])} / {f3(v['progressor']['value'])} / {f3(v['non_progressor']['value'])}" for b, v in r22["bands"].items()) + f". Share of samples sequenced {f3(sq['share_samples_sequenced'])}; endoscopies with any sample sequenced {f3(sq['share_endoscopies_with_any_sample_sequenced'])}; per-repeat share {f3(r8['share_sequenced_per_repeat']['mean'])} ({rg(r8['share_sequenced_per_repeat']['range'])}), in the middle band in ≥ 1 repeat {f3(r8['share_middle_band_in_at_least_one_repeat'])}.", "",
      f"One line: the package-feature primary criterion (voted) is Δ sensitivity {sg(pc['delta_sens_vs_B'])} against (B), lower {f3(pc['one_sided_95_lower'])}, share sequenced {f3(pc['share_sequenced'])}: {'met' if pc['met'] else 'not met'}; no vote, against B\\* {dd(nvB, one=True)} ({ni(nvB['one_sided_95_lower'])}).", ""]
# ------------------------------------------------------------------ R23
r23 = J("r23")["R23"]; el = r23["endoscopy_level"]; ES = ("T", "B*", "C*", "A", "B", "C")
L += [f"### R23. Endoscopy and patient level (their matrix, all pre-event; {el['n_endoscopies']} endoscopies, {el['n_progressor_endoscopies']} from progressors)", "", "| Mode | Strategy | Sensitivity [95% CI] | Specificity [95% CI] | Δ sens triage − strategy [95% CI] | Δ spec [95% CI] |", "|---|---|---|---|---|---|"]
for m in ("voted", "no_vote"):
    for s in ES:
        v = el[m][s]; p = el[m]["paired_triage_minus"].get(s); L.append(f"| endoscopy, {m.replace('_', ' ')} | {nm(s)} | {f3(v['sensitivity'])}{ci(v['sens_ci95'])} | {f3(v['specificity'])}{ci(v['spec_ci95'])} | {dd(p['d_sens']) if p else '—'} | {dd(p['d_spec']) if p else '—'} |")
pl = r23["patient_level_voted"]; L += ["", f"Patient level, voted ({pl['n_progressor_patients']} progressor and {pl['n_nonprogressor_patients']} non-progressor patients; exact Clopper–Pearson 95% CIs).", "", "| Strategy | Progressor patients missed (share [95% CI]) | Non-progressor patients flagged (share [95% CI]) | Δ missed share, triage − strategy [95% CI] (patients) | Δ flagged share [95% CI] (patients) |", "|---|---|---|---|---|"]
for s in ES:
    v = pl[s]; p = pl["paired_triage_minus"].get(s)
    dm = f"{dd(p['d_missed_share'])} ({p['d_missed_count']:+d})" if p else "—"; df = f"{dd(p['d_flagged_share'])} ({p['d_flagged_count']:+d})" if p else "—"
    L.append(f"| {nm(s)} | {v['progressor_patients_missed']} ({f3(v['missed_share'])}{ci(v['missed_ci95_clopper_pearson'])}) | {v['nonprogressor_patients_flagged']} ({f3(v['flagged_share'])}{ci(v['flagged_ci95_clopper_pearson'])}) | {dm} | {df} |")
bu = r23["burden"]; bt = bu["T"]["voted"]; dist = lambda d: " / ".join(f3(d[c]) for c in ("0", "1", "2", "3"))
L += ["", f"Burden per non-progressor (follow-up first to last pre-event endoscopy; {bt['n_nonprogressors_multi_endoscopy']} non-progressors with > 1 endoscopy, {bt['patient_years']:.1f} patient-years; {bt['single_endoscopy_nonprogressors']['n']} with a single endoscopy).", "", "| Strategy | Flagged endoscopies per non-progressor patient-year: voted / no vote | Flagged endoscopies per non-progressor patient, voted: 0 / 1 / 2 / ≥ 3 | Per progressor patient, voted: 0 / 1 / 2 / ≥ 3 | Per non-progressor patient, no vote: 0 / 1 / 2 / ≥ 3 |", "|---|---|---|---|---|"]
for s in ("T", "B", "C", "B*", "C*"):
    v = bu[s]; dv = v["voted"]["distribution_flagged_per_patient"]; dn = v["no_vote_mean"]["distribution_flagged_per_patient"]
    L.append(f"| {nm(s)} | {f3(v['voted']['flagged_endoscopies_per_nonprogressor_patient_year'])} / {f3(v['no_vote_mean']['flagged_endoscopies_per_nonprogressor_patient_year'])} | {dist(dv['non_progressor'])} | {dist(dv['progressor'])} | {dist(dn['non_progressor'])} |")
L += ["", f"One line: at patient level the voted triage misses {pl['T']['progressor_patients_missed']} progressor and flags {pl['T']['nonprogressor_patients_flagged']} of {pl['n_nonprogressor_patients']} non-progressors, with {f3(bt['flagged_endoscopies_per_nonprogressor_patient_year'])} flagged endoscopies per non-progressor patient-year against {f3(bu['B']['voted']['flagged_endoscopies_per_nonprogressor_patient_year'])} for (B) and {f3(bu['C']['voted']['flagged_endoscopies_per_nonprogressor_patient_year'])} for (C).", ""]
# ------------------------------------------------------------------ R24
r24 = J("r24")["R24"]; L += ["### R24. Current care, grade and sampling intensity (their matrix, all pre-event)", "", "| Comparator | Sample sens [95% CI] / spec [95% CI]; share flagged | Endoscopy sens / spec | Patients: progressors caught / missed; non-progressors flagged / unflagged |", "|---|---|---|---|"]
for k, v in r24["comparator_D"].items():
    s, e, p = v["sample"], v["endoscopy"], v["patient"]; L.append(f"| {k.replace('_', ' ')} | {ss(s)}; {f3(s['share_flagged'])} | {f3(e['sensitivity'])} / {f3(e['specificity'])} | {p['progressor_caught']} / {p['progressor_missed']}; {p['nonprogressor_flagged']} / {p['nonprogressor_unflagged']} |")
L += ["| Table 4 reference (voted) | " + "; ".join(f"{nm(s)} {f3(v['sensitivity'])} / {f3(v['specificity'])}" for s, v in r24["table4_reference_voted"].items()) + " | — | — |", ""]
mb = r24["modal_band_by_grade_and_class"]; L += ["Modal band × grade (counts low / mid / high): " + "; ".join(f"{cl.replace('_', '-')} {g} {v['low']} / {v['mid']} / {v['high']}" for cl, gg in mb.items() for g, v in gg.items()) + ".", "",
      "| Status | Patients | With any pre-event ID or LGD | Pre-event endoscopies per patient: mean, median (range) | Pre-event samples per patient: mean, median (range) | Span first to last pre-event endoscopy, years: mean, median (range) |", "|---|---|---|---|---|---|"]
qq = lambda d: f"{f3(d['mean'])}, {f3(d['median'])} ({rg(d['range'])})"
for stn, v in r24["worst_grade_and_sampling_by_status"].items():
    L.append(f"| {stn.replace('_', '-')} | {v['n_patients']} | {v['patients_with_any_pre_event_ID_or_LGD']} | {qq(v['endoscopies_per_patient'])} | {qq(v['samples_per_patient'])} | {qq(v['span_years_first_to_last_pre_event_endoscopy'])} |")
ei = r24["endoscopy_index_auroc"]; nd = r24["never_dysplastic_patients"]; d1s = r24["comparator_D"]["D1_grade_ID_or_LGD"]["sample"]
L += ["", f"Endoscopy index within the patient (1 = earliest) as a raw score: fold-stratified AUROC {f3(ei['value'])}{ci(ei['ci95'])}.", "",
      f"Never-dysplastic patients ({nd['n_samples']} samples, {nd['n_progressor_samples']} progressor; {nd['n_patients']} patients, {nd['n_progressor_patients']} progressors): AUROC " + ", ".join(f"{m} {f3(v['value'])}{ci(v['ci95'])}" for m, v in nd["auroc"].items()) + f"; Δ L-IMG − L-CNV {dd(nd['IMG_minus_CNV'], one=True)}; triage − (B) Δ sensitivity voted {dd(nd['triage_criterion_voted_vs_B']['d_sens_triage_minus'], one=True)}, no vote {dd(nd['triage_no_vote_vs_B']['d_sens_triage_minus'], one=True)} (Δ specificity no vote {dd(nd['triage_no_vote_vs_B']['d_spec_triage_minus'])}); triage − B\\* no vote {dd(nd['triage_no_vote_vs_Bstar']['d_sens_triage_minus'], one=True)}; share sequenced (no vote) {f3(nd['share_sequenced_no_vote'])}.", "",
      f"One line: current care (D1) has sample sensitivity {f3(d1s['sensitivity'])} at specificity {f3(d1s['specificity'])}; in never-dysplastic patients Δ L-IMG − L-CNV is {ni(nd['IMG_minus_CNV']['one_sided_95_lower'])} (lower {f3(nd['IMG_minus_CNV']['one_sided_95_lower'])}) while the triage criterion against (B) holds (no-vote lower {f3(nd['triage_no_vote_vs_B']['d_sens_triage_minus']['one_sided_95_lower'])}).", ""]
# ------------------------------------------------------------------ R25
r25 = J("r25")["R25"]; rr = r25["real_gate_reference"]; s2r = r25["share_to_reach_real_sensitivity"]; pr_ = lambda k: f"{f3(r25[k]['mean'])} ({rg(r25[k]['p2.5_97.5'])})"
L += [f"### R25. Negative control: uniform random H&E gate ({r25['n_seeds']} seeds; no vote; their matrix, all pre-event)", "", "| Gate | Share sequenced | Sensitivity | Specificity |", "|---|---|---|---|",
      f"| random (mean, 2.5th–97.5th percentile over seeds) | {pr_('random_gate_share_sequenced')} | {pr_('random_gate_sensitivity')} | {pr_('random_gate_specificity')} |", f"| real gate | {f3(rr['share_sequenced'])} | {f3(rr['sensitivity'])} | {f3(rr['specificity'])} |", "",
      f"Share that must be sequenced for the random gate to reach the real no-vote sensitivity ({f3(r25['target_real_no_vote_sensitivity'])}): not reached in {s2r['n_seeds_not_reached']} of {s2r['n_seeds']} seeds even with the middle band widened to all samples (sensitivity at full sequencing, mean over seeds, {f3(r25['sensitivity_at_full_sequencing_mean'])}). (b) The permutation-trained gate is in F6.", "",
      f"One line: a random gate sequences {f3(r25['random_gate_share_sequenced']['mean'])} of samples for sensitivity {f3(r25['random_gate_sensitivity']['mean'])} and does not reach the real gate's {f3(rr['sensitivity'])} at any share.", ""]
# ------------------------------------------------------------------ R26
r26 = J("r26")["R26"]; L += ["### R26. Cut-off transfer from the inner predictions", "", "| CNV source | Cut-off | Median | IQR | Range over the 100 folds |", "|---|---|---|---|---|"]
for src, v in r26["spread_of_fold_cutoffs"].items():
    for c, q in v.items(): L.append(f"| {SL[src]} | {c} | {f3(q['median'])} | {rg(q['iqr'])} | {rg(q['range'])} |")
L += ["", "| CNV source | Strategy | Training specificity: median (IQR) | Held-out specificity: median (IQR) | Held-out − training: median (IQR; range) |", "|---|---|---|---|---|"]
for src, v in r26["training_vs_heldout_operating_points"].items():
    for s, q in v.items(): L.append(f"| {SL[src]} | {'triage (s*)' if s == 'T' else s} | {f3(q['training_specificity']['median'])} ({rg(q['training_specificity']['iqr'])}) | {f3(q['heldout_specificity']['median'])} ({rg(q['heldout_specificity']['iqr'])}) | {sg(q['heldout_minus_training']['median'])} ({rgs(q['heldout_minus_training']['iqr'])}; {rgs(q['heldout_minus_training']['range'])}) |")
fz = r26["frozen_model_in_sample"]; nz = fz["nonzero_coefficients"]; bc = fz["band_shares_by_class"]; ck = fz["reproduction_check"]; lam = nz["lambda"]; hm = r26["training_vs_heldout_operating_points"]["their"]["T"]["heldout_minus_training"]["median"]
L += ["", f"Frozen their-matrix bundle on the 571 pre-event samples, {fz['label']}: cut-offs " + ", ".join(f"{k} {f3(v)}" for k, v in fz["cutoffs"].items()) + f"; bands low / mid / high {bnd(fz['band_shares_571'])} (progressor {bnd(bc['progressor'])}; non-progressor {bnd(bc['non_progressor'])}); in-sample triage sensitivity {f3(fz['triage_in_sample']['sensitivity'])}, specificity {f3(fz['triage_in_sample']['specificity'])}; (B) in sample {f3(fz['B_in_sample']['sensitivity'])} / {f3(fz['B_in_sample']['specificity'])}. Non-zero coefficients: frozen L-CNV {nz['frozen_L_CNV']}, frozen L-IMG {nz['frozen_L_IMG']}; λ frozen L-CNV {f3(lam['frozen_L_CNV'])} (fold range {rg(lam['fold_L_CNV_range'])}), frozen L-IMG {f3(lam['frozen_L_IMG'])} (fold range {rg(lam['fold_L_IMG_range'])}). Bundle reproduction: in-sample AUROC L-CNV {f3(ck['auroc_L_CNV_676'])} (model_info {f3(ck['model_info']['auroc_L_CNV'])}), L-IMG {f3(ck['auroc_L_IMG_676'])} (model_info {f3(ck['model_info']['auroc_L_IMG'])}).", "",
      f"One line: held-out specificity exceeds the training target s\\* by a median {sg(hm)} (their matrix), and the frozen in-sample band shares ({bnd(fz['band_shares_571'])}) are not a performance estimate.", ""]
# ------------------------------------------------------------------ R27
R27 = {pop: J(f"r27_{pop}")["R27"] for pop in PL}
L += ["### R27. Case–control pairs and time-dependent AUROC without within-patient pairs (their matrix)", "", "| Population | Samples / patients | Sample-level within-fold pairs, mean over repeats | Patient-level within-fold pairs, mean | Fold × repeat cells lacking one class | Sample pairs without folds |", "|---|---|---|---|---|---|"]
for pop, x in R27.items():
    p = x["pairs"]; L.append(f"| {PL[pop]} | {p['n_samples']} / {p['n_patients']} | {p['sample_pairs_mean']:.1f} | {p['patient_pairs_mean']:.1f} | {p['cells_lacking_total']} | {p['all_sample_pairs_without_folds']} |")
L += ["", "| Population | Horizon (years) | Cases / controls | Within-patient pairs removed (repeat 1, of within-fold pairs) | Weighting, pairs | L-CNV | L-IMG | L-LATE | Δ L-IMG − L-CNV [95% CI]; lower | Δ L-LATE − L-CNV [95% CI] |", "|---|---|---|---|---|---|---|---|---|---|"]
WL = (("ipcw_excluding_within_patient", "IPCW, excluding within-patient"), ("ipcw_all_within_fold_pairs_reference", "IPCW, all pairs (reference)"), ("unweighted_excluding_within_patient", "unweighted, excluding within-patient"), ("unweighted_all_within_fold_pairs_reference", "unweighted, all pairs (reference)"))
for pop, x in R27.items():
    for h, v in x["time_dependent_excluding_within_patient_pairs"].items():
        for kk, lab in WL:
            w = v[kk]; L.append(f"| {PL[pop]} | {h} | {v['n_cases']} / {v['n_controls']} | {v['within_patient_pairs_removed']} of {v['within_fold_pairs_repeat1']} | {lab} | " + " | ".join(f"{f3(w[m]['auroc'])}{ci(w[m]['ci95'])}" for m in ("L-CNV", "L-IMG", "L-LATE")) + f" | {dd(w['IMG_minus_CNV'], one=True)} | {dd(w['LATE_minus_CNV'])} |")
x = R27["pre"]["time_dependent_excluding_within_patient_pairs"]
L += ["", "One line: excluding within-patient pairs (IPCW, all pre-event), Δ L-IMG − L-CNV is " + "; ".join(f"{sg(x[h]['ipcw_excluding_within_patient']['IMG_minus_CNV']['delta'])} at the {h}-year horizon (lower {f3(x[h]['ipcw_excluding_within_patient']['IMG_minus_CNV']['one_sided_95_lower'])}, {ni(x[h]['ipcw_excluding_within_patient']['IMG_minus_CNV']['one_sided_95_lower'])})" for h in x) + ".", ""]
# ------------------------------------------------------------------ R28
r28 = J("r28")["R28"]; L += ["### R28. H-optimus-0 downstream predictions (stored; folds equal to the main CV)", "", "| Population | Model | AUROC [95% CI] |", "|---|---|---|"]
for pop in PL:
    for m, v in r28[pop]["auroc"].items(): L.append(f"| {PL[pop]} ({r28[pop]['n_samples']} samples) | {m.replace('_', ' ')} | {f3(v['value'])}{ci(v['ci95'])} |")
L += ["", "| Population | Difference | Δ [95% CI]; lower |", "|---|---|---|"]
for pop in PL:
    for k, lab in (("IMG_Hoptimus0_minus_CNV", "L-IMG (H-optimus-0) − L-CNV"), ("IMG_UNI2h_minus_CNV_reference", "L-IMG (UNI2-h) − L-CNV, reference"), ("LATE_Hoptimus0_minus_CNV", "L-LATE (H-optimus-0) − L-CNV"), ("IMG_Hoptimus0_minus_IMG_UNI2h", "L-IMG (H-optimus-0) − L-IMG (UNI2-h)")):
        v = r28[pop][k]; L.append(f"| {PL[pop]} | {lab} | {dd(v, one='one_sided_95_lower' in v)} |")
hp = r28["pre"]["IMG_Hoptimus0_minus_CNV"]; L += ["", f"One line: with H-optimus-0, Δ L-IMG − L-CNV is {sg(hp['delta'])} (lower {f3(hp['one_sided_95_lower'])}, {ni(hp['one_sided_95_lower'])}; all pre-event) and L-IMG (H-optimus-0) − L-IMG (UNI2-h) is {dd(r28['pre']['IMG_Hoptimus0_minus_IMG_UNI2h'])}; the H-optimus-0 triage is in F5.", ""]
# ------------------------------------------------------------------ R29
r29 = J("r29")["R29"]; lr = r29["likelihood_ratios"]; L += ["### R29. Band likelihood ratios and natural frequencies (their matrix, all pre-event)", "", "| Band | Rate in progressor samples [95% CI] | Rate in non-progressor samples [95% CI] | Likelihood ratio [95% CI] |", "|---|---|---|---|"]
for b_, v in r29["class_wise_band_rates"].items(): L.append(f"| {b_} | {f3(v['progressor']['value'])}{ci(v['progressor']['ci95'])} | {f3(v['non_progressor']['value'])}{ci(v['non_progressor']['ci95'])} | {f3(lr[b_]['value'])}{ci(lr[b_]['ci95'])} |")
c1 = lambda v: f"{v['value']:.1f} [{v['ci95'][0]:.1f}, {v['ci95'][1]:.1f}]"
L += ["", "Per 1,000 biopsies (no vote: sequenced = middle band; voted: sequenced = vote-pivotal; 95% CIs).", "", "| Prevalence | Mode | Sequenced | Progressor biopsies caught | Progressor biopsies missed | False flags | In the middle band in ≥ 1 repeat |", "|---|---|---|---|---|---|---|"]
for pi, mm in r29["natural_frequencies_per_1000"].items():
    for m, v in mm.items(): L.append(f"| {pi} | {m.replace('_', ' ')} | {c1(v['sequenced'])} | {c1(v['progressor_biopsies_caught'])} | {c1(v['progressor_biopsies_missed'])} | {c1(v['false_flags'])} | {c1(v['sequenced_any_repeat'])} |")
n5 = r29["natural_frequencies_per_1000"]["0.05"]["no_vote"]
L += ["", f"One line: LR(high) {f3(lr['high']['value'])}, LR(low) {f3(lr['low']['value'])}, LR(middle) {f3(lr['mid']['value'])}; at 5% prevalence the no-vote triage sequences {n5['sequenced']['value']:.1f} of 1,000 biopsies, catches {n5['progressor_biopsies_caught']['value']:.1f} of 50 progressor biopsies and flags {n5['false_flags']['value']:.1f} non-progressor biopsies (high-band PPV at 5%, R12 check, {f3(r29['sanity_round12']['PPV_high_band_0.05'])}).", ""]
# ------------------------------------------------------------------ R30
r30 = J("r30")["R30"]; vo = r30["voted"]; nvr = r30["no_vote_mean_over_repeats"]
L += [f"### R30. Lead time ({r30['n_progressor_patients_with_pre_event_samples']} progressor patients with pre-event samples; years from the first positive pre-event endoscopy to the first HGD/IMC)", "", "| Strategy | Voted: patients caught | Voted: median (IQR), mean | No vote (mean over repeats): patients caught | No vote: median (IQR), mean |", "|---|---|---|---|---|"]
for s in ("T", "B", "C", "B*", "C*"):
    v = vo[s]; w = nvr.get(s); nvs = f"{f3(w['median_mean'])} ({rg(w['iqr_mean'])}), {f3(w['mean_mean'])}" if w else "—"; nc = f"{w['n_caught_mean']:.1f}" if w else "—"
    L.append(f"| {nm(s)} | {v['n']} | {f3(v['median'])} ({rg(v['iqr'])}), {f3(v['mean'])} | {nc} | {nvs} |")
pC, pB, pn = vo["paired_T_minus_C_caught_by_both"], vo["paired_T_minus_B_caught_by_both"], nvr["paired_T_minus_C_caught_by_both"]
L += ["", f"Paired difference in patients caught by both (voted): triage − (C) median {f3(pC['median'])} (IQR {rg(pC['iqr'])}), mean {sg(pC['mean'])}, n {pC['n']}; triage − (B) median {f3(pB['median'])} (IQR {rg(pB['iqr'])}), mean {sg(pB['mean'])}, n {pB['n']}. No vote, triage − (C): median {f3(pn['median_mean'])}, mean {sg(pn['mean_mean'])}, n {pn['n_mean']:.1f}.", "",
      f"One line: lead time is the same for the triage and (C) in the patients both catch (median difference {f3(pC['median'])} years), with the triage's median {f3(vo['T']['median'])} years against {f3(vo['B']['median'])} for (B).", ""]
# ------------------------------------------------------------------ F1
F1 = J("f1")["F1"]; L += ["### F1. Pre-event-only training (outer 10 × 10 and inner 5-fold refits; training rows restricted to pre-event samples)", "",
      "Training rows per outer fold: " + "; ".join(f"{SL[src]} mean {F1[src]['training_rows_per_fold']['mean']:.1f} (range {F1[src]['training_rows_per_fold']['range'][0]}–{F1[src]['training_rows_per_fold']['range'][1]})" for src in F1) + f" (stored main CV: {F1['their']['training_rows_per_fold']['stored_main_cv']}).", "",
      "| CNV source | Population | L-CNV | L-IMG | L-LATE | Δ L-IMG − L-CNV [95% CI]; lower | Δ L-LATE − L-CNV [95% CI] | Stored reference L-CNV / L-IMG / L-LATE; stored Δ L-IMG − L-CNV lower |", "|---|---|---|---|---|---|---|---|"]
for src in F1:
    for pop in PL:
        t = F1[src][pop]; a = t["discrimination"]["auroc"]; sr = t.get("stored_reference", {}); sra = sr.get("auroc", {})
        L.append(f"| {SL[src]} | {PL[pop]} | " + " | ".join(f"{f3(a[m]['value'])}{ci(a[m]['ci95'])}" for m in ("L-CNV", "L-IMG", "L-LATE")) + f" | {dd(t['discrimination']['IMG_minus_CNV'], one=True)} | {dd(t['discrimination']['LATE_minus_CNV'])} | {f3(sra.get('L-CNV'))} / {f3(sra.get('L-IMG'))} / {f3(sra.get('L-LATE'))}; {f3(sr.get('img_vs_cnv', {}).get('one_sided_95_lower'))} |")
L += ["", "| CNV source | Population | Mode | Triage sens / spec | Share sequenced (per-repeat mean [95% CI]; ≥ 1 repeat) | Criterion vs (B): Δ sens; lower; met | A\\* Δ sens [95% CI]; lower | B\\* Δ sens [95% CI]; lower | C\\* Δ sens [95% CI]; lower |", "|---|---|---|---|---|---|---|---|---|"]
def strat_rows(block, src_lab, pop_lab):
    out = []
    for mode in ("voted", "no_vote"):
        t = block["strategies"][mode]["triage"]; cr = block[f"criterion_vs_B_{mode}"]; sh = block["share_sequenced"]
        out.append(f"| {src_lab} | {pop_lab} | {mode.replace('_', ' ')} | {ss(t)} | {f3(sh['per_repeat_mean'])}{ci(sh['ci95'])}; {f3(sh['in_at_least_one_repeat'])} | {sg(cr['delta_sens'])}; {f3(cr['one_sided_95_lower'])}; {'met' if cr['met'] else 'not met'} | " + " | ".join(dd(t[f"vs_{s}"]["d_sens"], one=True) for s in ("A*", "B*", "C*")) + " |")
    return out
for src in F1:
    for pop in PL: L += strat_rows(F1[src][pop], SL[src], PL[pop])
t = F1["their"]["pre"]; L += ["", "Cut-offs (their matrix, all pre-event; median, range over folds): " + ", ".join(f"{k} {f3(v['median'])} ({rg(v['range'])})" for k, v in t["cutoffs"].items()) + f"; training specificity target s\\* {f3(t['training_specificity_target']['median'])} ({rg(t['training_specificity_target']['range'])}); folds without a middle band {t['folds_no_middle_band']}, t_seq fallback {t['t_seq_fallback']}.", "",
      f"One line: with pre-event-only training (their matrix, all pre-event) Δ L-IMG − L-CNV is {sg(t['discrimination']['IMG_minus_CNV']['delta'])} (lower {f3(t['discrimination']['IMG_minus_CNV']['one_sided_95_lower'])}, {ni(t['discrimination']['IMG_minus_CNV']['one_sided_95_lower'])}) and the no-vote Δ sensitivity against B\\* is {dd(t['strategies']['no_vote']['triage']['vs_B*']['d_sens'], one=True)} ({ni(t['strategies']['no_vote']['triage']['vs_B*']['d_sens']['one_sided_95_lower'])}).", ""]
# ------------------------------------------------------------------ F2
F2 = J("f2")["F2"]; L += [f"### F2. Training-size control ({F2['training_patients']} training patients per fold, {F2['P_patients_in_48']['mean']:.1f} progressors (range {F2['P_patients_in_48']['range'][0]}–{F2['P_patients_in_48']['range'][1]}); full size {F2['full_training_patients']['mean']:.1f} patients, {F2['full_training_patients']['P_mean']:.1f} progressors)", "", "| Held-out samples | n | L-IMG, size-48 mixed-scanner [95% CI] | L-IMG, full-size main CV [95% CI] | Δ size-48 − full [95% CI] | Reference |", "|---|---|---|---|---|---|"]
for scn, v in F2["by_scanner"].items():
    ref = f"transfer model C13239-01 → C13210 {f3(v['transfer_model_C13239-01_to_C13210_reference'])}" if "transfer_model_C13239-01_to_C13210_reference" in v else f"within-C13239-01 CV {f3(v['within_C13239-01_cv_reference'])}"
    L.append(f"| {scn} | {v['n_samples']} | {f3(v['size48_mixed_scanner']['value'])}{ci(v['size48_mixed_scanner']['ci95'])} | {f3(v['full_size_main_cv']['value'])}{ci(v['full_size_main_cv']['ci95'])} | {dd(v['d_size48_minus_full'])} | {ref} |")
ap = F2["all_pre_event"]; L.append(f"| all pre-event | — | {f3(ap['size48']['value'])}{ci(ap['size48']['ci95'])} | {f3(ap['full_size'])} | — | — |")
c10 = F2["by_scanner"]["C13210"]; L += ["", f"One line: size-48 mixed-scanner models change by {sg(c10['d_size48_minus_full']['delta'])} on C13210 ({f3(c10['size48_mixed_scanner']['value'])}, between the full-size {f3(c10['full_size_main_cv']['value'])} and the transfer model's {f3(c10['transfer_model_C13239-01_to_C13210_reference'])}), so training size explains part of the transfer drop.", ""]
# ------------------------------------------------------------------ F3
F3 = J("f3")["F3"]; fd = J("f3_draws"); pi_ = lambda v: f"{sg(v['mean'])} / {sg(v['median'])} |{cis(v['ci95_percentile'])} | {f3(v['one_sided_95_lower'])} | {f3(v['sd'])}"
L += [f"### F3. Refit bootstrap ({F3['draws_scored']} draws; one stratified 10-fold CV per draw, λ and all cut-offs re-chosen; no vote; {fd['distinct_patients_per_draw']['mean']:.1f} distinct patients per draw, range {fd['distinct_patients_per_draw']['range'][0]}–{fd['distinct_patients_per_draw']['range'][1]}; mean {F3['n_eval_rows_mean']:.1f} pre-event evaluation rows)", "",
      "| Statistic | Refit bootstrap mean / median | 95% percentile interval | One-sided 5th percentile | SD | Conditional stored: voted | Conditional stored: no vote |", "|---|---|---|---|---|---|---|"]
for k, lab in (("d_auroc_IMG_minus_CNV", "Δ AUROC L-IMG − L-CNV"), ("d_sens_triage_vs_B", "Δ sensitivity triage − (B)"), ("d_spec_triage_vs_B", "Δ specificity triage − (B)"), ("d_sens_triage_vs_Bstar", "Δ sensitivity triage − B*")):
    v = F3[k]; cv = v.get("conditional_stored_voted", v.get("conditional_stored")); cn = v.get("conditional_stored_no_vote")
    L.append(f"| {lab} | {pi_(v['refit_bootstrap'])} | {dd(cv, one='one_sided_95_lower' in cv)} | {dd(cn, one='one_sided_95_lower' in cn) if cn else '—'} |")
tpd = F3["triage_per_draw"]; am = F3["auroc_refit_draw_mean"]
L += ["", "Per-draw triage (mean / median; 95% percentile interval): " + "; ".join(f"{k.replace('_', ' ')} {f3(v['mean'])} / {f3(v['median'])}{ci(v['ci95_percentile'])}" for k, v in tpd.items()) + f". Draw-mean AUROC: L-CNV {f3(am['L-CNV'])}, L-IMG {f3(am['L-IMG'])}, L-LATE {f3(am['L-LATE'])}.", "",
      f"One line: under the full refit bootstrap the 5th percentile of Δ AUROC L-IMG − L-CNV is {f3(F3['d_auroc_IMG_minus_CNV']['refit_bootstrap']['one_sided_95_lower'])} ({ni(F3['d_auroc_IMG_minus_CNV']['refit_bootstrap']['one_sided_95_lower'])}), of Δ sensitivity against (B) {f3(F3['d_sens_triage_vs_B']['refit_bootstrap']['one_sided_95_lower'])} ({ni(F3['d_sens_triage_vs_B']['refit_bootstrap']['one_sided_95_lower'])}) and against B\\* {f3(F3['d_sens_triage_vs_Bstar']['refit_bootstrap']['one_sided_95_lower'])} ({ni(F3['d_sens_triage_vs_Bstar']['refit_bootstrap']['one_sided_95_lower'])}).", ""]
# ------------------------------------------------------------------ F4
F4 = J("f4")["F4"]; sf = F4["share_fold_models_with_no_selected_feature"]; rp = F4["reproduction_class_lambda_vs_stored_max_abs_pred_diff"]
L += ["### F4. λ at the deviance minimum (outer 10 × 10 refits; their matrix)", "", "| Arm | Share of fold models with no selected feature: class-error λ / deviance λ | Non-zero coefficients, median: class-error / deviance | λ median: class-error / deviance | Share of folds with deviance λ < class-error λ | Max abs prediction difference, class-error refit vs stored |", "|---|---|---|---|---|---|"]
for m, v in sf.items(): L.append(f"| {m} | {f3(v['class_error_lambda'])} / {f3(v['deviance_lambda'])} | {f3(v['n_nonzero_median']['class_error_lambda'])} / {f3(v['n_nonzero_median']['deviance_lambda'])} | {f3(v['lambda_median']['class_error'])} / {f3(v['lambda_median']['deviance'])} | {f3(v['deviance_lambda_smaller_share'])} | {rp[m]} |")
L += ["", "| Population | λ rule | L-CNV | L-IMG | L-LATE | Δ L-IMG − L-CNV [95% CI]; lower | Δ L-LATE − L-CNV [95% CI] |", "|---|---|---|---|---|---|---|"]
for pop in PL:
    for k, lab in (("deviance_lambda", "deviance"), ("class_error_lambda_stored", "class error (stored)")):
        a = F4[pop][k]; L.append(f"| {PL[pop]} | {lab} | " + " | ".join(f"{f3(a['auroc'][m]['value'])}{ci(a['auroc'][m]['ci95'])}" for m in ("L-CNV", "L-IMG", "L-LATE")) + f" | {dd(a['IMG_minus_CNV'], one=True)} | {dd(a['LATE_minus_CNV'])} |")
dv = F4["pre"]["deviance_lambda"]["IMG_minus_CNV"]; L += ["", f"One line: with the deviance λ (all pre-event) Δ L-IMG − L-CNV is {sg(dv['delta'])} with lower bound {f3(dv['one_sided_95_lower'])} ({ni(dv['one_sided_95_lower'])}); no fold model is empty under either rule.", ""]
# ------------------------------------------------------------------ F5 / R28 triage
F5 = J("f5_r28")["F5_R28"]; L += [f"### F5. H-optimus-0 gate triage ({F5['gate']}; L-CNV {F5['L-CNV']})", "", "| CNV source | Population | Mode | Triage sens / spec | Share sequenced (per-repeat mean [95% CI]; ≥ 1 repeat) | Criterion vs (B): Δ sens; lower; met | A\\* Δ sens [95% CI]; lower | B\\* Δ sens [95% CI]; lower | C\\* Δ sens [95% CI]; lower |", "|---|---|---|---|---|---|---|---|---|"]
for pop in PL: L += strat_rows(F5[pop], "their, H-optimus-0 gate", PL[pop])
t = F5["pre"]; sr = t["stored_reference_UNI2h"]
L += ["", "Cut-offs (all pre-event; median, range over folds): " + ", ".join(f"{k} {f3(v['median'])} ({rg(v['range'])})" for k, v in t["cutoffs"].items()) + f"; s\\* {f3(t['training_specificity_target']['median'])} ({rg(t['training_specificity_target']['range'])}). Stored UNI2-h reference (all pre-event): share sequenced {f3(sr['share_sequenced_voted_mean'])}, voted Δ sensitivity against (B) {dd(sr['triage_vs_B_sens_voted'], one=True)}, voted B\\* {dd(sr['matched_voted']['B*'])}.", "",
      f"One line: the H-optimus-0 gate sequences {f3(t['share_sequenced']['per_repeat_mean'])} of samples (UNI2-h {f3(sr['share_sequenced_voted_mean'])}) with the criterion against (B) {'met' if t['criterion_vs_B_voted']['met'] else 'not met'} (voted lower {f3(t['criterion_vs_B_voted']['one_sided_95_lower'])}; no vote lower {f3(t['criterion_vs_B_no_vote']['one_sided_95_lower'])}) and no-vote Δ sensitivity against B\\* {dd(t['strategies']['no_vote']['triage']['vs_B*']['d_sens'], one=True)}.", ""]
# ------------------------------------------------------------------ F6 / R25(b)
F6 = J("f6_r25b")["F6_R25b"]; L += [f"### F6. Permutation-trained gate ({F6['permutations']} patient-level label permutations; outer and inner L-IMG refitted; their matrix, all pre-event)", "", "| Permutation | Gate AUROC (fold-stratified) | L-LATE AUROC with the permuted gate | No vote: share sequenced / sensitivity / specificity | Voted: sensitivity / specificity; middle band in ≥ 1 repeat |", "|---|---|---|---|---|"]
for p, v in F6["per_permutation"].items():
    nv, vo_ = v["no_vote"], v["voted"]; L.append(f"| {p} | {f3(v['gate_auroc_fold_stratified'])} | {f3(v['late_auroc_with_permuted_gate'])} | {f3(nv['share_sequenced'])} / {f3(nv['sensitivity'])} / {f3(nv['specificity'])} | {f3(vo_['sensitivity'])} / {f3(vo_['specificity'])}; {f3(vo_['share_sequenced_in_at_least_one_repeat'])} |")
mn, mv, rg_ = F6["mean_over_permutations_no_vote"], F6["mean_over_permutations_voted"], F6["real_gate_no_vote"]
L += [f"| mean | {f3(F6['gate_auroc_mean'])} | — | {f3(mn['share_sequenced'])} / {f3(mn['sensitivity'])} / {f3(mn['specificity'])} | {f3(mv['sensitivity'])} / {f3(mv['specificity'])}; — |", f"| real gate (no vote) | — | — | {f3(rg_['share_sequenced'])} / {f3(rg_['sensitivity'])} / {f3(rg_['specificity'])} | — |", "",
      f"One line: a gate fitted on permuted labels (AUROC {f3(F6['gate_auroc_mean'])}) sequences {f3(mn['share_sequenced'])} of samples for sensitivity {f3(mn['sensitivity'])} and specificity {f3(mn['specificity'])}, against {f3(rg_['share_sequenced'])}, {f3(rg_['sensitivity'])} and {f3(rg_['specificity'])} for the real gate.", ""]
# ------------------------------------------------------------------ F7
L += ["### F7. Leave-one-patient-out L-CNV", "", "Not refitted: stored leave-one-patient-out predictions cover all 676 samples and 80 patients (check.json), so R21(b) used them (R21b above).", ""]
# ------------------------------------------------------------------ implementation notes and checks
L += ["### Implementation notes", "",
      "- R14c/R1-style evaluation matching uses the smallest held-out specificity ≥ the triage's (ceil rule, positive if score > threshold).",
      "- R16 draws are the tb_selci draws (20 shards × 100).",
      "- R17 max-T is single-step over centred bootstrap-t draws; Holm from percentile p.",
      "- R19 trigger interval resamples 117 internal patients with all their samples (mean 891 samples per draw, not 234), so it is narrower than an interval for a 2-samples-per-patient cohort would be.",
      "- R21b used the stored LOPO L-CNV predictions (killcoyne_mm/linear), so F7 was not refitted.",
      "- R26 fold-model coefficient counts were not stored; λ ranges are reported instead.",
      "- R27 same-patient exclusion also removes cross-copy pairs of a bootstrapped patient.",
      "- F1 pre-event filter applies to outer and inner training rows; inner folds re-stratified over pre-event patients.",
      "- F2: 48 patients, progressor count matched to the fold's training share.",
      "- F3: the first 200 boots_for draws over the 75 pre-event patients; copies share their patient's fold; each draw has about 48 distinct patients, so refit models are weaker than the stored ones (per-draw triage sens 0.83, spec 0.60, share 0.57) and the interval is likely pessimistic.",
      "- F4: deviance with 1e-5 clamp as km_linear.R; ties → largest λ.",
      "- F6: one patient-level permutation per (perm, rep, fold), shared by outer and inner fits; cut-offs chosen against real labels.", ""]
CK = J("check")["checks"]; s14 = R14["their_pre"]["sanity_round1"]
L += ["### Checks", ""] + [f"- {k}: n {v['n']}, max abs prediction difference {v['max_abs_pred_diff']}, folds equal {v['folds_equal']}, λ max abs difference {v['lambda_max_abs_diff']}: {'passed' if v['passed'] else 'FAILED'}." for k, v in CK.items() if isinstance(v, dict) and "max_abs_pred_diff" in v]
h0 = CK["hoptimus0_outer_auroc_pre"]; lp = CK["lopo_L-CNV_their_stored"]
L += [f"- H-optimus-0 stored outer AUROC (all pre-event) {f3(h0['value'])} against the documented {f3(h0['reference_docs_paper_scanner_robustness'])}, folds equal {h0['folds_equal_stored']}: {'passed' if h0['passed'] else 'FAILED'}.",
      f"- Stored leave-one-patient-out L-CNV: {lp['rows']} rows, {lp['patients']} patients, all samples covered {lp['all_samples_covered']} ({lp['source']}); F7 needed: {lp['F7_needed']}. All checks passed: {CK['all_passed']}.",
      f"- Round-1 numbers reproduced from the stored predictions: voted Δ sensitivity against B\\* {sg(s14['voted_d_sens_vs_Bstar'])}, against (B) {sg(s14['voted_d_sens_vs_B'])}, middle band in ≥ 1 repeat {f3(s14['share_mid_any'])}, per-repeat share sequenced {f3(s14['share_mid_mean'])} (R14); R4 transfer sensitivity {f3(r19['transfer_unreset']['sensitivity'])} (R19); R12 high-band PPV at 5% {f3(r29['sanity_round12']['PPV_high_band_0.05'])} (R29); recomputed selection-inclusive draws against the stored tb_selci draws, max abs difference {chk['max_abs_diff_vs_recomputed']} (R16).", ""]
open(DOC, "w").write(head + "\n".join(L) + "\n"); print("rendered", len(L), "lines")

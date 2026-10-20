"""Render docs/paper_review_round7c_gate_robustness.md results (pre-specification 6794545) from results/paper_final/review_round7c/*.json.
Appends below the document's final '---' line; the pre-specification above it is not touched. Idempotent (replaces anything after that line).
Counts of 1-4 patients, samples or endoscopies are shown as '<5'. RESC = results commit."""
import json, os
T = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
R = T + "/results/paper_final/review_round7c"; DOC = T + "/docs/paper_review_round7c_gate_robustness.md"; RESC = os.environ.get("RESC", "see git log")
J = lambda f: json.load(open(f"{R}/{f}"))
ci = lambda c: f"[{c[0]:+.3f}, {c[1]:+.3f}]" if c and c[0] is not None else ""
cip = lambda c: f"[{c[0]:.3f}, {c[1]:.3f}]" if c and c[0] is not None else ""
yn = lambda b: "yes" if b else "no"
f3_ = lambda v: "" if v is None else f"{v:.3f}"
def c5(k):
    k = int(k); return "<5" if 0 < k < 5 else str(k)
L = []; A = L.append
M = J("merge.json")["R7C"]; TAB = M["table"]; SUM = M["summary"]; CK = M["checks"]; P = M["primary"]; F3 = J("f3.json")["R7C"]
LAB = {"r49": "**R49 gate**", "Bstar_endo": "B\\*-endo (sequence all, B\\* threshold)", "B_endo": "(B)-endo (sequence all, (B) threshold)", "triage": "sample-level triage, aggregated", "Cstar_endo": "C\\*-endo (H&E alone, training-matched)",
       "rand_matched": "random-one, matched to R49's endoscopy specificity", "rand_own": "random-one, own threshold"}
VD = {"within": "within", "shift": "fails: shift", "width": "fails: width"}

A("\n## Results\n")
A(f"Rendered by `scripts/paper_plan/rv7c_render.py` from `results/paper_final/review_round7c/` (aggregate results {RESC}). R49 exactly as in round 7 (max L-IMG endoscopy score, the triage's per-fold t_low / t_high / t_seq, top-L-IMG sample sequenced in the middle band). Single model = per-repeat values averaged over the 10 repeats; patient bootstrap 2,000 draws (`RandomState(0)`) with fitted models and cut-offs fixed; bound = one-sided 95% lower bound (5th percentile). Unit = pre-event endoscopy. No model was refitted for this item (refit rows read stored round-2 / round-4 / round-7 refit predictions).\n")
A("| Item | Status |\n|---|---|")
A("| primary comparison (R49 against B\\*-endo) and the other comparators | DONE |")
for r in TAB:
    if r["row"] == "primary": continue
    A(f"| {r['item']}. {r['label']} | {'DONE' if r['status'] == 'done' else r['status']} |")

# ---------------------------------------------------------------- primary
A("\n### Primary comparison\n")
A(f"{P['n_endoscopies']} pre-event endoscopies ({P['n_progressor_endoscopies']} progressor), {P['n_samples']} samples, {P['n_patients']} patients ({P['n_progressor_patients']} progressor). R49 endoscopy bands (per repeat): low {P['r49_endoscopy_band_shares']['low']:.3f}, middle {P['r49_endoscopy_band_shares']['middle']:.3f}, high {P['r49_endoscopy_band_shares']['high']:.3f}.\n")
A(f"| Rule | Endoscopy sens [95% CI] | Endoscopy spec [95% CI] | Samples sequenced | Endoscopies sequenced | Progressor patients missed (of {P['n_progressor_patients_in_units']}) / non-progressor flagged (of {P['n_non_progressor_patients_in_units']}) |\n|---|---|---|---|---|---|")
for nm, lab in LAB.items():
    g = P["rules"][nm]; A(f"| {lab} | {g['endoscopy_sensitivity']:.3f} {cip(g['sens_ci95'])} | {g['endoscopy_specificity']:.3f} {cip(g['spec_ci95'])} | {g['share_samples_sequenced']:.3f} | {g['share_endoscopies_sequenced']:.3f} | {g['progressor_patients_missed']:.1f} / {g['non_progressor_patients_flagged']:.1f} |")
A("\n| R49 − | Δ endoscopy sens [95% CI] | Bound | Within −0.05 | Δ endoscopy spec [95% CI] | Δ share of samples sequenced [95% CI] | Δ share of endoscopies sequenced | Δ patients missed / flagged |\n|---|---|---|---|---|---|---|---|")
for nm in list(LAB)[1:]:
    d = P["r49_minus"][nm]; s = d["d_endoscopy_sens"]; pp = f"{d['d_patients_missed']:+.1f} / {d['d_patients_flagged']:+.1f}" if "d_patients_missed" in d else ""
    A(f"| {LAB[nm]} | {s['delta']:+.3f} {ci(s['ci95'])} | {s['one_sided_95_lower']:+.3f} | {yn(s['non_inferior_at_-0.05'])} | {d['d_endoscopy_spec']['delta']:+.3f} {ci(d['d_endoscopy_spec']['ci95'])} | {d['d_share_samples']['delta']:+.3f} {ci(d['d_share_samples']['ci95'])} | {d['d_share_endoscopies']:+.3f} | {pp} |")

# ---------------------------------------------------------------- robustness table
A("\n### Robustness table\n")
A("Each row recomputes R49 and B\\*-endo on that row's predictions, cut-offs and population (the R33 Claim T definition of the row; see the implementation notes). Fails by: shift = two-sided 95% CI entirely below 0; width = CI includes 0 but the bound is below −0.05.\n")
A("| Item | Row | Endoscopies (progressor) / samples / patients (progressor) | R49 endo sens / spec | Δ endo sens vs B\\*-endo [95% CI] | Bound | Within −0.05 | Verdict | Δ endo spec vs B\\*-endo | Samples sequenced: R49 / triage | Endoscopies sequenced (R49) | Δ endo sens vs triage (bound) |\n|---|---|---|---|---|---|---|---|---|---|---|---|")
for r in TAB:
    lab = r["label"] + (" (reference)" if r["row"] == "primary" else "")
    if r["status"] != "done": A(f"| {r['item']} | {lab} | | | | | | {r['status']} | | | | |"); continue
    if r["row"] == "per_repeat":
        pr_ = r["per_repeat_d_sens"]; lw = r["per_repeat_one_sided_lower"]
        A(f"| {r['item']} | {lab} | | | per repeat {min(pr_):+.3f} to {max(pr_):+.3f} | {min(lw):+.3f} to {max(lw):+.3f} | {r['n_repeats_within']} of 10 | (count) | | | | |"); continue
    nn = r.get("n"); ns = f"{c5(nn['n_endoscopies'])} ({c5(nn['n_progressor_endoscopies'])}) / {c5(nn['n_samples'])} / {c5(nn['n_patients'])} ({c5(nn['n_progressor_patients'])})" if nn else ""
    ss = f"{r['r49_sens_spec'][0]:.3f} / {r['r49_sens_spec'][1]:.3f}" if "r49_sens_spec" in r else ""
    sq = f"{r['share_samples_sequenced']:.3f} / {r['triage_share_samples_sequenced']:.3f}" if r.get("triage_share_samples_sequenced") is not None else f"{r['share_samples_sequenced']:.3f}"
    dt = f"{r['d_sens_vs_triage']:+.3f} ({r['d_sens_vs_triage_lower']:+.3f})" if "d_sens_vs_triage" in r else ""
    A(f"| {r['item'] if r['row'] != 'primary' else '—'} | {lab} | {ns} | {ss} | {r['d_sens']:+.3f} {ci(r['ci95'])} | {r['one_sided_lower']:+.3f} | {yn(r['within_-0.05'])} | {VD[r['verdict']] if r['row'] != 'primary' else '(reference) ' + VD[r['verdict']]} | {r['d_spec']:+.3f} | {sq} | {r['share_endoscopies_sequenced']:.3f} | {dt} |")
fr = [r for r in TAB if r["row"] == "f3"][0]
if fr["status"] == "done":
    A(f"\nRefit bootstrap: Monte Carlo SE of the 5th percentile {fr['mc_se_of_5th_percentile']:.4f}; share of the {F3['draws_scored']} draws with Δ below −0.05: {fr['share_of_draws_below_-0.05']:.3f}; held-out pre-event endoscopies per draw median {F3['n_endoscopies_per_draw']['median']:.0f} (range {F3['n_endoscopies_per_draw']['min']}–{F3['n_endoscopies_per_draw']['max']}).")
tv = J("row_tis_lt256.json")["R7C"]["all_endoscopies_top_sample_stratum_changes_in_some_repeat_share"]
A(f"\nTissue strata: for {tv:.3f} of the {P['n_endoscopies']} endoscopies the stratum of R49's per-repeat top sample differs from that of the repeat-averaged top sample (the assignment used) in at least one repeat.")

# ---------------------------------------------------------------- summary
A("\n### Summary (reporting only)\n")
A(f"Of {SUM['rows_in_summary']} robustness rows (items 1–15; the primary row and the per-repeat count row are not counted): **{SUM['within_-0.05']} within −0.05**, **{SUM['fail_by_shift']} fail by a shift** (CI entirely below 0), **{SUM['fail_by_width']} fail by width**, **{SUM['not_available']} not available**" + (f" ({'; '.join(SUM['not_available_rows'])})" if SUM["not_available_rows"] else "") + f". R49 sequences fewer samples than the sample-level triage in {SUM['rows_with_lower_share_sequenced_than_triage']} of the rows where both are reported. No pass/fail threshold is set; whether R49 is promoted is the authors' decision.")

# ---------------------------------------------------------------- checks
A("\n### Checks\n")
cp = CK["primary"]; fl = lambda v: ", ".join("" if x is None else f"{x:.3f}" for x in v)
A(f"- R49 through this code: {fl(cp['r49_gate_vs_R49']['got'])} (R49: {fl(cp['r49_gate_vs_R49']['expected'])}; endoscopy sens, spec, samples, endoscopies, patients missed, flagged); reproduced: {yn(cp['r49_gate_vs_R49']['reproduced'])}; sensitivity CI {cip(cp['r49_gate_vs_R49']['sens_ci_got_expected'][0])} (R49 {cip(cp['r49_gate_vs_R49']['sens_ci_got_expected'][1])}). Against the pre-specification's 0.907 / 0.560, 25.5% of samples, 50.5% of endoscopies: reproduced: {yn(cp['r49_spec_samples_endoscopies_vs_prespec_0.907_0.560_0.255_0.505']['reproduced'])}.")
A(f"- B\\*-endo: {fl(cp['Bstar_endo_vs_0.923_0.530']['got'])} (expected 0.923, 0.530); reproduced: {yn(cp['Bstar_endo_vs_0.923_0.530']['reproduced'])}.")
A(f"- Sample-level triage aggregated: {fl(cp['triage_vs_R49']['got'])} (R49: {fl(cp['triage_vs_R49']['expected'])}); reproduced: {yn(cp['triage_vs_R49']['reproduced'])}. Random-one at its own threshold: {fl(cp['rand_own_vs_R46']['got'])} (R46: {fl(cp['rand_own_vs_R46']['expected'])}); reproduced: {yn(cp['rand_own_vs_R46']['reproduced'])}.")
A(f"- Cut-off grid primary cell (95 / 90) equals the primary row (rules, deltas, between-repeat block and counts identical): {yn(CK.get('grid_primary_cell_equals_primary_row'))}.")
b7 = cp.get("vs_round7b_apparent_R49_variant")
if b7: A(f"- Primary against round 7b's apparent R49-variant row (same population and bootstrap): {fl(b7['got'])} (7b: {fl(b7['expected'])}; the last two = Δ endoscopy sens vs B\\*-endo and bound); reproduced: {yn(b7['reproduced'])}.")
for r, lab in (("ndbe", "NDBE-only"), ("ge2", "≥ 2 samples"), ("r52a", "R52(a)")):
    c = CK.get(f"{r}_vs_round7b_apparent")
    if c: A(f"- Row '{lab}' against round 7b's apparent R49-variant row on the same population: {fl(c['got'])} (7b: {fl(c['expected'])}); reproduced: {yn(c['reproduced'])}.")
c = CK.get("f3", {}).get("sample_level_triage_minus_Bstar_mean_and_5th_vs_R33")
if c: A(f"- Refit bootstrap: the sample-level triage − B\\* Δ sensitivity re-derived from the stored F3 draws, mean and 5th percentile {fl(c['got'])} (R33 row 18: {fl(c['expected'])}).")

A("\n### Implementation notes\n")
NOTES = [
 "R49, B\\*-endo, (B)-endo, the aggregated triage, the random-one rules and the bootstrap are the rv7_round7.py r49 / rv7b_gate.py merge code, unchanged; C\\*-endo = the training-matched C\\* call (L-IMG at the triage's training specificity, `matched_calls`) with the endoscopy flagged if any sample is positive, sequencing nothing. Random-one matched = one L-LATE threshold per seed × repeat, pooled across folds, giving R49's endoscopy specificity in that repeat (re-set in each draw, as R46).",
 "Row cut-offs follow the R33 Claim T implementation of each row. Rows that R33 scored on stored calls (never-dysplastic, earliest endoscopy, excluding progressor samples within 1 year, tissue strata, R38(a), R53, R55) keep the primary cut-offs, B\\* and calls and change only the evaluated samples; R49 is then recomputed on the remaining samples of each endoscopy (endoscopy score = max over its remaining pre-event samples; endoscopies left with no sample drop out). Rows with their own predictions or population (NDBE, package features, R36, F5, deviance λ, tissue- and year-adjusted gate, the refits) re-choose every cut-off and B\\* on that row's training samples, as in R33.",
 "Earliest pre-event endoscopy (row 3) uses R33 row 13 (stored cut-offs; earliest = largest months-before-final), not round 7b (b), which re-chose cut-offs. Endoscopies with ≥ 2 samples (row 4) is not an R33 row; it follows round 7b (c) (cut-offs re-chosen on endoscopies with ≥ 2 pre-event samples).",
 "Tissue strata (row 7): the endoscopy keeps all its pre-event samples and is assigned by the tissue tiles of its top sample, taken as the sample with the highest L-IMG averaged over the 10 repeats (a fixed assignment; R49 sequences the per-repeat top sample, and the share of endoscopies where the two differ in stratum in some repeat is reported above). The 'pure' rows keep endoscopies whose samples all fall in one stratum.",
 "Tissue- and year-adjusted gate (row 12): the R32b residualised L-IMG (year and log tissue tiles, outer per repeat, inner per (repeat, fold)) replaces L-IMG everywhere in the gate (endoscopy score, bands, sequenced sample, C\\*); the middle-band call stays the stored L-LATE, as in R32b.",
 "Label rows (13): R38(a) = the sheet's 'excluded' flag only (the R38 'also report' variant with 'remove' and 'inconsistency' is not a separate row); R53 = its primary 'on or after' row (the 14-day variant is not run); R43(a)–(c), R54 and R52(a)/(b) use the round-7 refit outer and inner predictions with all cut-offs re-chosen (the refit sets' own endoscopies).",
 "Inference rows (15): between-repeat variance = the R33 / R15 rule on the primary row (mean of the 10 per-repeat Δ ± z √(V_boot + V_rep)); per-repeat bounds = each repeat's own 5th-percentile bound (a count, so it is listed but not counted in the summary). The refit bootstrap is run because stored outer and inner predictions exist for all 200 F3 draws; the per-draw cut-offs are not stored files but are re-derived from each draw's stored inner predictions by the fixed R33 F3 rule (rv4_round4.py r33_f3, unchanged), which reproduces R33 row 18. Endoscopy units in a draw = (resampled patient copy, endoscopy).",
 "Summary verdicts: within = bound > −0.05; shift = not within and the two-sided 95% CI (percentile interval for the refit bootstrap, variance-inflated interval for the between-repeat row) entirely below 0; width = not within and the CI includes 0. No row was refitted and every needed stored prediction set existed, so nothing is 'not available'." if SUM["not_available"] == 0 else
 "Summary verdicts: within = bound > −0.05; shift = not within and the two-sided 95% CI entirely below 0; width = not within and the CI includes 0. Rows whose stored predictions were missing are 'not available' and were not refitted.",
 "Checks compare values at three decimals with a tolerance of 0.0015.",
]
for t in NOTES: A(f"- {t}")
txt = open(DOC).read(); head = txt[:txt.rindex("\n---") + 4]
open(DOC, "w").write(head + "\n" + "\n".join(L) + "\n"); print("RENDERED", len(L), "lines")

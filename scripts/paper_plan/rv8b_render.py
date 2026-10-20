"""Render docs/paper_review_round8b_step3.md results (pre-specification c39fd82) from results/paper_final/review_round8b/*.json.
Appends below the document's final '---' line; the pre-specification above it is not touched. Idempotent (replaces anything after that line).
Counts of 1-4 patients in patient cells are shown as '<5'. RESC = results commit. R8BDIR overrides the JSON folder (smoke preview only)."""
import json, os
T = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
R = os.environ.get("R8BDIR", T + "/results/paper_final/review_round8b"); DOC = os.environ.get("R8BDOC", T + "/docs/paper_review_round8b_step3.md"); RESC = os.environ.get("RESC", "see git log")
def J(f):
    try: return json.load(open(f"{R}/{f}"))
    except FileNotFoundError: return None
ci = lambda c, d=3: f"[{c[0]:+.{d}f}, {c[1]:+.{d}f}]" if c and c[0] is not None else ""
cip = lambda c, d=3: f"[{c[0]:.{d}f}, {c[1]:.{d}f}]" if c and c[0] is not None else ""
gbp = lambda v, sg=False: ("+" if sg and v >= 0 else "") + ("−" if v < 0 else "") + f"£{abs(v):,.0f}"
cig = lambda c: f"[{gbp(c[0])}, {gbp(c[1])}]" if c and c[0] is not None else ""
yn = lambda b: "yes" if b else "no"
pc = lambda v: "n/a" if v is None else f"{100 * v:.1f}%"
L = []; A = L.append
S1 = J("s1.json"); S2 = J("s2.json"); S3 = J("s3.json"); S4 = J("s4.json")
A("\n## Results\n")
A(f"Rendered by `scripts/paper_plan/rv8b_render.py` from `results/paper_final/review_round8b/` (aggregate results {RESC}); script `scripts/paper_plan/rv8b_step3.py` (policy definitions, thresholds and outcomes executed unchanged from `rv8_landscape.py`); "
  "figures by `scripts/paper_plan/rv8b_figs.py` (local, not committed). No model was refitted and no ACE-B file was read. Single model = per-repeat values averaged over the 10 repeats. "
  "Intervals: paired patient bootstrap, 2,000 draws (`RandomState(0)`), thresholds and calls fixed unless stated. These analyses do not change the main rule (R49).\n")
# ---------------------------------------------------------------- S1
A("### S1. Does sequencing reduce false flags compared with H&E alone? (paired)\n")
if S1 is None: A("not available\n")
else:
    s = S1["S1"]; st = s["statement"]
    MN = [("non_progressor_patients_flagged", "Δ NP patients flagged (of 43)", 1), ("endoscopy_specificity", "Δ endoscopy specificity", 3), ("flagged_np_endoscopies_per_np_patient_year", "Δ flagged NP endoscopies / NP patient-year", 3),
          ("progressor_patients_caught", "Δ progressor patients caught (of 32)", 1), ("samples_sequenced_per_patient", "Δ samples sequenced / patient", 2)]
    A("Δ = rule − P1 (H&E alone). (i) P1's threshold set per repeat so that its endoscopy sensitivity equals the rule's realised endoscopy sensitivity (re-matched in every draw); (ii) P1 at its 90% training cell. Values: rule / P1, then Δ [95% CI].\n")
    A("| Contrast | Operating point | " + " | ".join(m[1] for m in MN) + " |\n|---|---|" + "---|" * len(MN))
    for cn, cl in (("a_R49_vs_P1", "(a) R49 − P1"), ("b_P3_90_vs_P1", "(b) P3 90% − P1")):
        for op, ol in (("i_evaluation_matched", "(i) evaluation-matched"), ("ii_P1_at_90", "(ii) P1 at 90%")):
            o = s["contrasts"][cn][op]; cells = []
            for k, _, d in MN:
                dd = o["delta_rule_minus_P1"][k]; cells.append(f"{o['rule'][k]:.{d}f} / {o['P1'][k]:.{d}f}; **{dd['delta']:+.{d}f}** {ci(dd['ci95'], d)}")
            A(f"| {cl} | {ol} | " + " | ".join(cells) + " |")
    A("")
    for cn, cl in (("a_R49_vs_P1", "(a)"), ("b_P3_90_vs_P1", "(b)")):
        mt = s["contrasts"][cn]["i_evaluation_matched"]["matching"]; o = s["contrasts"][cn]["i_evaluation_matched"]
        A(f"- {cl}(i) matching: endoscopy sensitivity rule {o['rule']['endoscopy_sensitivity']:.3f}, P1 {o['P1']['endoscopy_sensitivity']:.3f} (P1 − rule observed {mt['P1_minus_rule_endoscopy_sensitivity_observed']:+.3f}; over draws median {mt['P1_minus_rule_endoscopy_sensitivity_draws_median_max'][0]:+.3f}, max {mt['P1_minus_rule_endoscopy_sensitivity_draws_median_max'][1]:+.3f}).")
    d = s["contrasts"]["a_R49_vs_P1"]["i_evaluation_matched"]["delta_rule_minus_P1"]["non_progressor_patients_flagged"]
    A(f"\n**Binding manuscript statement (pre-specified rule):** Δ non-progressor patients flagged in (a)(i) = {d['delta']:+.1f} {ci(d['ci95'], 1)}; upper 95% limit {st['upper_95_limit']:+.3f} {'<' if st['met'] else '≥'} 0. The manuscript states: **\"{st['statement']}\"**\n")
    ck = s["check_round8"]; g = ck["got"]
    A(f"Check (round-8 settings, observed): R49 {g['R49']['caught']:.1f} caught / {g['R49']['np_flagged']:.1f} NP flagged; P1 90% {g['P1 90%']['caught']:.1f} / {g['P1 90%']['np_flagged']:.1f}; P3 90% {g['P3 90%']['caught']:.1f} / {g['P3 90%']['np_flagged']:.1f} "
      f"(round 8: 31.1 / 31.8, 31.1 / 36.0, 31.2 / 31.8); reproduced: {yn(ck['reproduced'])}.\n")
# ---------------------------------------------------------------- S2
A("### S2. How stable is the frontier?\n")
if S2 is None: A("not available\n")
else:
    s = S2["S2"]
    if s["missing_families"]: A(f"Families not available: {', '.join(s['missing_families'])}.\n")
    A(f"{s['n_points']} round-8 policy cells (the p53-subset current-care cell excluded, as in round 8), {s['draws']:,} draws. Pareto = not dominated on progressor patients caught, non-progressor patients ever flagged and samples sequenced per patient; "
      "cost = not dominated on caught and cost per patient (£85 per sample, £535 per flagged endoscopy). Observed = round-8 single-model frontier membership.\n")
    fs = s["frontier_size_per_draw"]; A(f"Points on the frontier per draw, median [IQR]: Pareto {fs['pareto_median_iqr'][0]:.0f} [{fs['pareto_median_iqr'][1]:.0f}, {fs['pareto_median_iqr'][2]:.0f}], cost {fs['cost_median_iqr'][0]:.0f} [{fs['cost_median_iqr'][1]:.0f}, {fs['cost_median_iqr'][2]:.0f}].\n")
    for nm, lab in (("R49", "R49 (P6 t_low 95% / t_high 90%)"), ("P1_90", "P1 at 90%")):
        o = s[nm]; td = lambda L_: "; ".join(f"{x['family']} {x['label']} ({pc(x['share_draws_dominating'])})" for x in L_) or "none"
        A(f"- **{lab}**: dominated in {pc(o['share_draws_dominated_pareto'])} of draws on the Pareto criteria (most often by {td(o['top_dominators_pareto'])}) and in {pc(o['share_draws_dominated_cost'])} on cost (most often by {td(o['top_dominators_cost'])}).")
    shown = lambda c: c["share_draws_on_pareto"] >= 0.05 or c["share_draws_on_cost_frontier"] >= 0.05 or c["observed_on_pareto"] or c["observed_on_cost_frontier"] or (c["family"], c["label"]) == ("P6", "t_low 95% / t_high 90%")
    A("\nCells on either frontier in ≥ 5% of draws or in the observed data, plus R49:\n")
    A("| Policy | Cell | Caught | NP flagged | Sequenced / patient | Cost / patient | Observed: Pareto / cost | Share of draws on Pareto | on cost frontier |\n|---|---|---|---|---|---|---|---|---|")
    for c in s["cells"]:
        if shown(c):
            o = c["observed"]; A(f"| {c['family']} | {c['label']} | {o['caught']:.1f} | {o['np_flagged']:.1f} | {o['seq_per_patient']:.2f} | £{o['cost']:,.0f} | {yn(c['observed_on_pareto'])} / {yn(c['observed_on_cost_frontier'])} | {pc(c['share_draws_on_pareto'])} | {pc(c['share_draws_on_cost_frontier'])} |")
    nr = sum(1 for c in s["cells"] if not shown(c))
    ck = s["check_round8_frontier"]
    A(f"\nThe other {nr} cells were on neither frontier in the observed data and on each in < 5% of draws (all shares in `s2.json`). Check: observed frontier membership equals round 8 for {ck['cells_compared'] - len(ck['mismatches'])} of {ck['cells_compared']} cells; reproduced: {yn(ck['reproduced'])}"
      + (f" (differences: {', '.join(ck['mismatches'])})." if ck["mismatches"] else ".") + "\n")
# ---------------------------------------------------------------- S3
A("### S3. Cost sensitivity (descriptive)\n")
if S3 is None: A("not available\n")
else:
    s = S3["S3"]; names = s["rules"]; ab = {"P1 80%": "P1 80", "P1 90%": "P1 90", "P1 95%": "P1 95", "R49": "R49", "P3 90%": "P3 90", "P8 t_b 90% / t_mid 80%": "P8 90/80", "grade ID/LGD": "grade"}
    A("Progressor patients caught (of 32, fixed across the grid): " + "; ".join(f"{ab[k]} {s['progressor_patients_caught'][k]:.1f}" for k in names) + ". Cost per patient or per non-progressor patient-year (NP py); "
      "'10%' = progressor and non-progressor patient means mixed 10% / 90% (applies to the per-patient unit only). Frontier = rules not dominated on caught and cost.\n")
    A("| Sequencing £ | Flag £ | Unit | Weight | " + " | ".join(ab[k] for k in names) + " | Cost frontier | Δ R49 − P1 90 [CI] |\n|---|---|---|---|" + "---|" * len(names) + "---|---|")
    for g in s["grid"]:
        if g["unit"] == "np_py" and g["progressor_weight"] != "cohort": continue
        A(f"| £{g['seq_price']:.0f} | £{g['flag_cost']:,.0f} | {'patient' if g['unit'] == 'patient' else 'NP py'} | {g['progressor_weight'] if g['unit'] == 'patient' else '–'} | " + " | ".join(f"£{g['cost'][k]['value']:,.0f}" for k in names)
          + f" | {', '.join(ab[k] for k in g['cost_frontier'])} | {gbp(g['delta_cost_R49_minus_P1_90']['delta'], True)} {cig(g['delta_cost_R49_minus_P1_90']['ci95'])} |")
    G = s["grid"]; ng = len(G); cnt = {k: sum(k in g["cost_frontier"] for g in G) for k in names}
    A(f"\nPer-NP-patient-year rows do not depend on the progressor weighting (their duplicates are omitted; all {ng} grid points are in `s3.json`). Times on the cost frontier over the {ng} grid points: " + "; ".join(f"{ab[k]} {cnt[k]}" for k in names) + ".")
    neg = sum(1 for g in G if g["delta_cost_R49_minus_P1_90"]["ci95"][1] is not None and g["delta_cost_R49_minus_P1_90"]["ci95"][1] < 0); pos = sum(1 for g in G if g["delta_cost_R49_minus_P1_90"]["ci95"][0] is not None and g["delta_cost_R49_minus_P1_90"]["ci95"][0] > 0)
    A(f"Δ cost R49 − P1 90%: interval entirely below 0 at {neg} and entirely above 0 at {pos} of {ng} grid points.")
    ck = s["check_round8_cost"]; A(f"\nCheck (£85 / £535, per patient, cohort weights): caught and cost equal the round-8 cells: {yn(ck['reproduced'])} (R49 £{ck['got_caught_cost']['R49'][1]:,.2f} vs £{ck['expected_round8']['R49'][1]:,.2f}; P1 90% £{ck['got_caught_cost']['P1 90%'][1]:,.2f} vs £{ck['expected_round8']['P1 90%'][1]:,.2f}).\n")
# ---------------------------------------------------------------- S4
A("### S4. ACE-B-shaped precision (no ACE-B data)\n")
if S4 is None: A("not available\n")
else:
    s = S4["S4"]; dz = s["design"]
    A(f"{dz['draws']:,} discovery subsamples: {dz['progressor_patients']} progressor patients (without replacement, of {dz['cohort_progressor_patients']}) and {dz['non_progressor_patients']} non-progressor patients "
      f"(**with replacement**: only {dz['cohort_non_progressor_patients']} exist); one random pre-event endoscopy per drawn patient, at most {dz['max_samples_per_endoscopy']} of its samples at random. Stored outer predictions and R49's round-7 fold cut-offs fixed.\n")
    A("| Quantity | Median | 2.5–97.5% range |\n|---|---|---|")
    for k, lab in (("auroc_patient_max_L_IMG", "H&E model patient-level AUROC (max L-IMG)"), ("r49_endoscopy_sensitivity", "R49 endoscopy sensitivity"), ("r49_endoscopy_specificity", "R49 endoscopy specificity"),
                   ("p1_specificity_matched", "P1 specificity at R49's sensitivity"), ("delta_specificity_r49_minus_p1_matched", "Δ specificity, R49 − P1 at matched sensitivity"),
                   ("r49_share_samples_sequenced", "R49 share of samples sequenced"), ("r49_progressor_patients_detected_of_15", "R49 progressor patients detected (of 15)"), ("auroc_hanley_mcneil_lower_95", "AUROC lower 95% bound (Hanley–McNeil)")):
        o = s[k]; dg = 1 if "detected" in k else 3; A(f"| {lab} | {o['median']:.{dg}f} | {o['p2_5']:.{dg}f} – {o['p97_5']:.{dg}f} |")
    A(f"\nShare of draws in which \"≥ 12 of 15 progressor patients detected by R49\" passes: {pc(s['share_pass_ge12_of_15_repeat_mean'])} (detected count averaged over the 10 repeats); {pc(s['share_pass_ge12_of_15_per_repeat_mean'])} averaged per repeat.\n")
    pb = s["proposed_pass_bounds"]; p = pb["primary"]; se = pb["secondary_sensitivity"]; sq = pb["secondary_share_sequenced"]
    A("**Proposed pass bounds for the ACE-B registration draft** (not manuscript results):\n")
    A(f"- Primary (H&E model transfer): patient-level AUROC with a lower 95% bound above {p['bound']:.2f}, as pre-specified. Expected AUROC in an ACE-B-shaped sample: median {p['expected_auroc_median']:.3f}; "
      f"the lower bound exceeded 0.70 in {pc(p['share_draws_lower_bound_above_0_70_hanley_mcneil'])} of draws (Hanley–McNeil), the point estimate in {pc(p['share_draws_point_auroc_above_0_70'])}.")
    A(f"- Secondary (R49 gate), sensitivity: ≥ {se['proposed_bound_progressors_detected_at_least']} of {se['of']} progressor patients detected ({se['rule']}; passed in {pc(se['share_draws_passing'])} of draws). The pre-specified check '≥ 12 of 15' passed in {pc(se['share_draws_passing_ge12'])}.")
    A(f"- Secondary (R49 gate), share of samples sequenced: ≤ {sq['proposed_upper_bound']:.2f} ({sq['rule']}; passed in {pc(sq['share_draws_passing'])} of draws).\n")
# ---------------------------------------------------------------- figures and notes
A("### Figures (local, not committed)\n")
A("`paper/figures_r8/`: `r8b_s1_forest.png` (S1 paired differences), `r8b_s2_frontier_share.png` (S2 frontier membership), `r8b_s3_cost_grid.png` (S3 cost-frontier membership across the grid), `r8b_s4_distributions.png` (S4 distributions with the proposed bounds).\n")
A("### Implementation notes\n")
for t in [
    "Reuse: `rv8b_step3.py` executes `rv8_landscape.py` unchanged up to its task dispatch, so setup, endoscopy units, thresholds (`thr_arr`), policies (`family`, `r49_calls`, `p8_calls`), outcomes (`patient_arrays`) and bootstrap draws (`pdraws`, `RandomState(0)`, one-class draws rejected) are round 8's. Outcomes are recomputed from the same weighted sums as round 8's `lstats`.",
    "S1 contrast (b) uses P3 at its 90% cell (the round-8 check cell). Counts are shares × 32 or × 43 in every draw, as round 8. Δ samples sequenced equals the rule's own value because P1 sequences nothing.",
    "S1 (i) matching: per repeat, P1's threshold is the largest value of max L-IMG reaching at least the rule's realised endoscopy sensitivity over the evaluated endoscopies (bootstrap multiplicities as integer weights; round 7d's binary-call matching); with continuous scores the realised P1 sensitivity can exceed the rule's slightly (gap reported).",
    "S1 statement: the binding rule is applied to the 95% percentile interval of Δ non-progressor patients flagged (count of 43) in (a)(i).",
    "S2: every round-8 cell's outcomes are recomputed per draw with thresholds and calls fixed and rounded as round 8 (counts 2 dp, samples sequenced 3 dp, cost 2 dp) before the dominance test; ties are not dominance. 'Dominated' = not on that frontier in the draw.",
    "S3: P8 at 90/80 = t_b 90%, t_mid 80% (t_c 90%); grade = P10 grade ID/LGD on all pre-event endoscopies. Cost per non-progressor patient-year = sequencing and flagged-endoscopy costs incurred in non-progressor patients divided by their patient-years (round 8's flags-per-NP-patient-year analogue), so progressor reweighting does not apply to it. "
    "Reweighting to 10% progressors mixes the progressor and non-progressor per-patient mean costs 10% / 90%. The cost frontier is observed (single-model means), among the seven rules only; Δ cost has a paired bootstrap interval.",
    "S4: the pre-specification asks for 67 non-progressor patients without replacement, but the cohort has 43; non-progressors are therefore drawn with replacement (each copy picks its own endoscopy and samples), progressors without replacement. One RandomState(0) stream for all 10,000 draws.",
    "S4: AUROC is pooled over folds within each repeat (an external test set is scored by one model; fold-stratified AUROC is unstable with 15 progressors) and averaged over repeats; R49 uses the stored round-7 cut-offs (t_low 95%, c* 90%, t_seq) of the drawn patient's outer fold, applied to the max L-IMG of the chosen ≤ 2 samples, the top one sequenced in the middle band. "
    "'≥ 12 of 15 detected' is judged on the repeat-mean count (also reported per repeat). The Hanley–McNeil lower bound and the proposal rules (2.5th percentile of sensitivity, floored to a count of 15; 97.5th percentile of the share sequenced, rounded up) were added to turn the distributions into bounds; they are proposals for the registration draft only.",
    "Checks were run first in a 20-draw smoke campaign (aggregates written aside, not committed) and repeated in the full run: S1 point estimates at round-8 settings, S2 observed frontier membership (76 of 76 cells) and S3 base costs all equal round 8.",
    "Privacy: only aggregates are committed; per-draw arrays stay on the cluster (killcoyne_mm/review_round8b/); no identifiers or dates."]:
    A(f"- {t}")
txt = open(DOC).read(); head = txt[:txt.rindex("\n---") + 4]
open(DOC, "w").write(head + "\n" + "\n".join(L) + "\n"); print("rendered", DOC)

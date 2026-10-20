"""Render docs/paper_review_round8_landscape.md results (pre-specification b396cdc) from results/paper_final/review_round8/*.json.
Appends below the document's final '---' line; the pre-specification above it is not touched. Idempotent (replaces anything after that line).
Counts of 1-4 patients, samples or endoscopies in patient cells are shown as '<5'. RESC = results commit."""
import json, os
T = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
R = T + "/results/paper_final/review_round8"; DOC = T + "/docs/paper_review_round8_landscape.md"; RESC = os.environ.get("RESC", "see git log")
J = lambda f: json.load(open(f"{R}/{f}"))
ci = lambda c: f"[{c[0]:+.3f}, {c[1]:+.3f}]" if c and c[0] is not None else ""
cip = lambda c: f"[{c[0]:.3f}, {c[1]:.3f}]" if c and c[0] is not None else ""
cig = lambda c: f"[£{c[0]:,.0f}, £{c[1]:,.0f}]" if c and c[0] is not None else ""
yn = lambda b: "yes" if b else "no"
f1 = lambda v: "n/a" if v is None else f"{v:.1f}"
f2 = lambda v: "n/a" if v is None else f"{v:.2f}"
f3_ = lambda v: "n/a" if v is None else f"{v:.3f}"
def c5(k):
    if isinstance(k, str): return k
    if k is None: return "n/a"
    return "<5" if 0 < k < 5 else (f"{k:.1f}" if abs(k - round(k)) > 1e-9 else str(int(round(k))))
L = []; A = L.append
PR = J("prep.json")["prep"]; L1 = J("l1.json")["L1"]; L3 = J("l3.json")["L3"]; L4 = J("l4.json")["L4"]; L5 = J("l5.json")["L5"]; dd = PR["dates"]
FAMN = {"P1": "P1. H&E only", "P2": "P2. Copy number only, sequence all", "P3": "P3. Sequence all, H&E + copy number", "P4": "P4. One random sample per endoscopy", "P5": "P5. Sample-level triage",
        "P6": "P6. R49 gate", "P7": "P7. Sequence once, then H&E", "P8": "P8. Sequence once, re-sequence when H&E rises", "P9": "P9. Sequence every N years", "P10": "P10. Current care"}
A("\n## Results\n")
A(f"Rendered by `scripts/paper_plan/rv8_render.py` from `results/paper_final/review_round8/` (aggregate results {RESC}); script `scripts/paper_plan/rv8_landscape.py`; figures by `scripts/paper_plan/rv8_figs.py` (local, not committed). "
  "**Everything below is descriptive and hypothesis-generating; no comparison is confirmatory and nothing is adjusted for multiplicity.** No model was refitted; stored patient-grouped outer and inner out-of-fold predictions were re-scored. "
  "Single model = per-repeat values averaged over the 10 repeats. Intervals: patient bootstrap, 2,000 draws (`RandomState(0)`), thresholds and calls fixed.\n")
A("### Population, dates and availability\n")
A(f"{PR['n_endoscopies']} pre-event endoscopies ({PR['n_progressor_endoscopies']} progressor), {PR['n_samples']} samples, {PR['n_patients']} patients ({PR['n_progressor_patients']} progressor, {PR['n_non_progressor_patients']} non-progressor). "
  f"Endoscopies per patient: progressors {PR['endoscopies_per_patient']['progressor']['mean']:.2f} (≥ 2 in {PR['endoscopies_per_patient']['progressor']['ge2']}), non-progressors {PR['endoscopies_per_patient']['non_progressor']['mean']:.2f} (≥ 2 in {PR['endoscopies_per_patient']['non_progressor']['ge2']}).\n")
src = dd["reference_date_source_patients"]; ad = dd["abs_days_exact_vs_implied_reference"]; py = dd["patient_years"]; sp = dd.get("endoscopy_date_spread_days_when_more_than_one_date") or {}
A(f"- Exact dates (`pre_event_cohort.Date`) for {dd['analysed_samples_with_exact_date']} of {dd['analysed_samples']} analysed samples. Event date (progressors) = exact date of the sheet event endoscopy (first HGD/IMC endoscopy): {src['progressor_exact_event_endoscopy']} patients; "
  f"{c5(src['progressor_implied'])} had no exact date on that endoscopy's samples and use the date implied by their analysed samples' exact dates and months before final. Last sequenced endoscopy (non-progressors) = latest exact date over the patient's sequenced samples on the sheet: {src['non_progressor_sheet_exact']} patients.")
A(f"- Exact reference dates against those implied by months before final: progressors median {ad['progressor']['median']:.0f} days (90th percentile {ad['progressor']['p90']:.0f}, max {ad['progressor']['max']:.0f}; > 90 days in {c5(ad['progressor']['n_over_90_days'])}); "
  f"non-progressors median {ad['non_progressor']['median']:.0f} days (90th percentile {ad['non_progressor']['p90']:.0f}, max {ad['non_progressor']['max']:.0f}; > 90 days in {c5(ad['non_progressor']['n_over_90_days'])}). "
  f"Progressor pre-event samples, years to event from exact dates against months before final: median absolute difference {PR['exact_vs_months_before_final_time_to_event_progressor_samples']['median_abs_diff_years']:.2f} years (max {PR['exact_vs_months_before_final_time_to_event_progressor_samples']['max_abs_diff_years']:.2f}).")
A(f"- Endoscopy units (patient, endoscopy number, as rounds 6–7d) whose samples carry more than one exact date: {dd['endoscopies_with_more_than_one_sample_date']} (spread median {sp.get('median', 0):.0f} days, max {sp.get('max', 0):.0f}; all dates in one calendar year in {sp.get('units_with_all_dates_in_one_calendar_year', 'n/a')}); the unit's date is its earliest sample date. "
  f"Endoscopies dated after the reference date: {dd['endoscopies_after_reference_date']}; progressor pre-event endoscopies dated on the event date: {c5(dd['progressor_endoscopies_dated_on_the_event_date'])}. Exact-date order equals months-before-final order in {dd['patients_with_date_order_equal_to_months_before_final_order']} of {PR['n_patients']} patients.")
A(f"- Patient-years: progressors {py['progressor']:.1f} (first pre-event endoscopy to event), non-progressors {py['non_progressor']:.1f} (first pre-event endoscopy to last sequenced endoscopy; round 7d, months-before-final span: {dd['non_progressor_patient_years_months_before_final_span_as_round7']:.1f}), all {py['all']:.1f}.")
av = PR["available"]; A(f"- Arrays: outer predictions finite in 10 repeats: {yn(av['outer_repeats_finite'])}; inner repeat × fold sets {av['inner_repeat_folds']['L-IMG']} (L-IMG) / {av['inner_repeat_folds']['L-CNV']} (L-CNV); stored time-dependent AUROC at 1 / 3 / 5 years: {' / '.join(yn(v) for v in av['time_dependent_auroc_stored'].values())}. Nothing was 'not available'.\n")
# ---------------------------------------------------------------- L1
A("### L1. Policy map\n")
A("Per policy and grid cell (single model). Caught = progressor patients (of 32) flagged at ≥ 1 pre-event endoscopy; NP flagged = non-progressor patients (of 43) ever flagged; shares with 95% CI in brackets. Flags / NP py = flagged non-progressor endoscopies per non-progressor patient-year. "
  "First flag → event = median years over caught progressors. Last 2 = caught / NP flagged within each patient's last 2 pre-event endoscopies. Sequenced = samples per patient (per 100 patient-years). Cost = £85 per sequenced sample + £535 per flagged endoscopy, per patient (all 75). "
  "F = on the Pareto frontier (caught, NP flagged, samples sequenced); C = on the cost / caught frontier.\n")
A("| Policy | Cell | Caught [share CI] | NP flagged [share CI] | Flags / NP py [CI] | First flag → event (y) | Last 2: caught / NP flagged | Sequenced / patient (/100 py) | Cost / patient [CI] | Endo sens / spec | Frontier |\n|---|---|---|---|---|---|---|---|---|---|---|")
for c in L1["cells"]:
    fr = ("F" if c["pareto_caught_falseflags_sequencing"] else "") + (" C" if c["pareto_caught_cost"] else "")
    nP, nN = c["n_progressor_patients"], c["n_non_progressor_patients"]
    A(f"| {c['family']} | {c['label']} | {f1(c['progressor_patients_caught'])} / {nP} {cip(c['caught_share_ci95'])} | {f1(c['non_progressor_patients_ever_flagged'])} / {nN} {cip(c['np_flagged_share_ci95'])} | {f3_(c['flags_per_non_progressor_patient_year'])} {cip(c['flags_per_np_py_ci95'])} | "
      f"{f2(c['years_first_flag_to_event_median'])} | {f1(c['last2_window']['progressor_patients_caught'])} / {f1(c['last2_window']['non_progressor_patients_flagged'])} | {f2(c['samples_sequenced_per_patient'])} ({f1(c['samples_sequenced_per_100_patient_years'])}) | "
      f"£{c['cost_per_patient_gbp']:,.0f} {cig(c['cost_ci95'])} | {f3_(c['endoscopy_sensitivity'])} / {f3_(c['endoscopy_specificity'])} | {fr.strip()} |")
A("")
nm = sum(c["folds_target_not_met"] for c in L1["cells"])
A(f"Folds in which a training sensitivity target could not be met: {nm} (over all policies and cells). P5 / P6 folds without a middle band, summed over the 9 cells: P5 {sum(c.get('folds_no_middle_band', 0) for c in L1['cells'] if c['family'] == 'P5')}, P6 {sum(c.get('folds_no_middle_band', 0) for c in L1['cells'] if c['family'] == 'P6')}. "
  "The P10 grade-or-p53 row uses only samples with a p53 result (its own patients; counts of its own denominators) and is excluded from both frontiers.\n")
A("**Pareto frontier** (not dominated on caught, non-progressor patients ever flagged and samples sequenced per patient):\n")
A("| Policy | Cell | Caught | NP flagged | Sequenced / patient | Flags / NP py | Cost / patient | Endo sens / spec |\n|---|---|---|---|---|---|---|---|")
for c in L1["cells"]:
    if c["pareto_caught_falseflags_sequencing"]:
        A(f"| {c['family']} | {c['label']} | {f1(c['progressor_patients_caught'])} | {f1(c['non_progressor_patients_ever_flagged'])} | {f2(c['samples_sequenced_per_patient'])} | {f3_(c['flags_per_non_progressor_patient_year'])} | £{c['cost_per_patient_gbp']:,.0f} | {f3_(c['endoscopy_sensitivity'])} / {f3_(c['endoscopy_specificity'])} |")
A("\n**Cost / caught frontier:** " + "; ".join(f"{c['family']} {c['label']} ({f1(c['progressor_patients_caught'])} caught, £{c['cost_per_patient_gbp']:,.0f})" for c in sorted([c for c in L1["cells"] if c["pareto_caught_cost"]], key=lambda c: c["cost_per_patient_gbp"])) + ".\n")
ck = L1["checks"]
A("**Checks**\n")
g = ck["check_R49"]; A(f"- P6 primary cell (t_low 95% / t_high 90%) against R49: endoscopy sens / spec {g['got'][0]:.3f} / {g['got'][1]:.3f}, samples sequenced {100 * g['got'][2]:.1f}%, progressor patients missed {g['got'][3]:.1f}, non-progressor patients flagged {g['got'][4]:.1f} "
  f"(R49: {g['expected'][0]:.3f} / {g['expected'][1]:.3f}, {100 * g['expected'][2]:.1f}%, {g['expected'][3]:.1f}, {g['expected'][4]:.1f}); reproduced: {yn(g['reproduced'])}.")
g = ck["check_triage_R49"]; A(f"- P5 primary cell against the round-7 triage aggregated to endoscopies: {g['got'][0]:.3f} / {g['got'][1]:.3f}, {100 * g['got'][2]:.1f}%, missed {g['got'][3]:.1f}, flagged {g['got'][4]:.1f} (expected {g['expected'][0]:.3f} / {g['expected'][1]:.3f}, {100 * g['expected'][2]:.1f}%, {g['expected'][3]:.1f}, {g['expected'][4]:.1f}); reproduced: {yn(g['reproduced'])}.")
g = ck["check_Bstar"]; A(f"- P3 at the B\\* operating point (round-7 B\\* sample threshold, endoscopy flagged if any sample is): {g['got'][0]:.3f} / {g['got'][1]:.3f} (B\\*-endo 0.923 / 0.530); reproduced: {yn(g['reproduced'])}. Nearest P3 grid cell: {g['nearest_grid_cell']} ({g['nearest_grid_cell_sens_spec'][0]:.3f} / {g['nearest_grid_cell_sens_spec'][1]:.3f}).")
g = ck["check_Table1_D_patient"]; A(f"- P10 grade ID/LGD per patient: {g['got_missed_flagged'][0]:.0f} missed / {g['got_missed_flagged'][1]:.0f} flagged (Table 1 (D): {g['expected_missed_flagged'][0]} / {g['expected_missed_flagged'][1]}); reproduced: {yn(g['reproduced'])}.\n")
# ---------------------------------------------------------------- L2
A("### L2. Where copy number adds value\n")
A("Fold-stratified AUROC on pre-event samples within each stratum (single model), Δ with 95% patient-bootstrap CI. Call change = share of endoscopy × repeat cells whose P1 and P3 calls at 90% training endoscopy sensitivity differ (progressor / non-progressor endoscopies); "
  "gained / lost = P3 flags and P1 does not / P1 flags and P3 does not.\n")
A("| Stratum | Samples (progressor) / patients (progressor) | AUROC L-IMG / L-CNV / L-LATE | Δ L-LATE − L-IMG [CI] | Δ L-CNV − L-IMG [CI] | Endoscopies (progressor) | Call change P / NP | Progressor gained / lost | NP gained / lost |\n|---|---|---|---|---|---|---|---|---|")
for fam in ("overall", "heq", "grade", "tissue", "tte", "position", "nsamp"):
    try: S2 = J(f"l2_{fam}.json")["L2"]["strata"]
    except FileNotFoundError: A(f"| {fam} | not available | | | | | | | |"); continue
    for s in S2:
        a = s["auroc"]; cc = s["call_change_P1_vs_P3_at_90"]
        A(f"| {s['stratum']} | {c5(s['n_samples'])} ({c5(s['n_progressor_samples'])}) / {c5(s['n_patients'])} ({c5(s['n_progressor_patients'])}) | {f3_(a['L-IMG']['value'])} / {f3_(a['L-CNV']['value'])} / {f3_(a['L-LATE']['value'])} | "
          f"{s['LATE_minus_IMG']['delta']:+.3f} {ci(s['LATE_minus_IMG']['ci95'])} | {s['CNV_minus_IMG']['delta']:+.3f} {ci(s['CNV_minus_IMG']['ci95'])} | {c5(cc['n_endoscopies'])} ({c5(cc['n_progressor_endoscopies'])}) | "
          f"{f3_(cc['share_changed_progressor'])} / {f3_(cc['share_changed_non_progressor'])} | {f3_(cc['progressor_gained_flag_share'])} / {f3_(cc['progressor_lost_flag_share'])} | {f3_(cc['non_progressor_gained_flag_share'])} / {f3_(cc['non_progressor_lost_flag_share'])} |")
A("\nTime-to-event strata compare the stratum's progressor samples with all non-progressor samples (so the non-progressor call-change columns repeat). Strata with few progressor patients have wide intervals; no stratum contrast is tested.\n")
# ---------------------------------------------------------------- L3
A("### L3. Change over time within a patient\n")
n3 = L3["n"]; A(f"Patients with ≥ 2 pre-event endoscopies: {n3['patients_ge2_endoscopies']} ({n3['progressor']} progressor, {n3['non_progressor']} non-progressor), {n3['endoscopies']} endoscopies. Endoscopy score = mean over repeats of the maximum logit outer prediction. "
  "Time = years before the event (progressors) or before the last sequenced endoscopy (non-progressors); a **negative** slope against time to event means the score **rises** towards the event.\n")
A("| Modality | Mean slope / year: progressors (median) | non-progressors (median) | Δ mean slope P − NP [CI] | Mixed model time × class [95% CI], p | Time (NP) | Class | Last − first: progressors [CI] (share up) | non-progressors [CI] (share up) |\n|---|---|---|---|---|---|---|---|---|")
for m in ("L-IMG", "L-CNV"):
    o = L3[m]; s = o["slope_per_year_vs_time_to_event"]; d = o["difference_progressor_minus_non_progressor"]; mm = o["mixed_model"]
    mmt = f"{mm['tte:cls']['estimate']:+.3f} [{mm['tte:cls']['ci95'][0]:+.3f}, {mm['tte:cls']['ci95'][1]:+.3f}], p = {mm['tte:cls']['p']:.3f}" if "tte:cls" in mm else mm.get("status", "n/a")
    lp, ln = o["last_minus_first_progressor"], o["last_minus_first_non_progressor"]
    A(f"| {m} | {s['progressor_mean']:+.3f} ({s['progressor_median']:+.3f}) | {s['non_progressor_mean']:+.3f} ({s['non_progressor_median']:+.3f}) | {d['delta']:+.3f} {ci(d['ci95'])} | {mmt} | {mm['tte']['estimate']:+.3f} [{mm['tte']['ci95'][0]:+.3f}, {mm['tte']['ci95'][1]:+.3f}] | "
      f"{mm['cls']['estimate']:+.3f} [{mm['cls']['ci95'][0]:+.3f}, {mm['cls']['ci95'][1]:+.3f}] | {lp['mean']:+.3f} {ci(lp['ci95'])} ({lp['share_increased']:.2f}) | {ln['mean']:+.3f} {ci(ln['ci95'])} ({ln['share_increased']:.2f}) |")
A("")
for m in ("L-IMG", "L-CNV"):
    mm = L3[m]["mixed_model"]
    if "converged" in mm: A(f"- {m} mixed model: random-intercept variance {mm['random_intercept_variance']:.3f}, residual variance {mm['residual_variance']:.3f}, converged {yn(mm['converged'])}, {mm['n_endoscopies']} endoscopies / {mm['n_patients']} patients.")
A("\nTime-dependent AUROC (stored, fold-stratified IPCW, `results/paper_final/horizon_foldstrat`; re-tabulated, not recomputed):\n")
A("| Horizon | Cases / controls (samples) | L-CNV [CI] | L-IMG [CI] | L-LATE [CI] |\n|---|---|---|---|---|")
for h, v in L3["time_dependent_auroc_stored"].items():
    if isinstance(v, str): A(f"| {h} year(s) | not available | | | |"); continue
    A(f"| {h} year{'s' if h != '1' else ''} | {v['n_cases']} / {v['n_controls']} | {v['L-CNV']['ipcw']:.3f} {cip(v['L-CNV']['ci95'])} | {v['L-IMG']['ipcw']:.3f} {cip(v['L-IMG']['ci95'])} | {v['L-LATE']['ipcw']:.3f} {cip(v['L-LATE']['ci95'])} |")
A("")
# ---------------------------------------------------------------- L4
A("### L4. Disagreement within an endoscopy as a signal\n")
A("| Unit | Modality | AUROC max [CI] | AUROC SD [CI] | AUROC max + SD [CI] | Δ (max + SD) − max [CI] |\n|---|---|---|---|---|---|")
for lv in ("endoscopy", "patient"):
    for m in ("L-IMG", "L-CNV"):
        o = L4[lv][m]; a = o["auroc"]
        A(f"| {lv} (n = {L4[lv]['n_units']}, {L4[lv]['n_progressor_units']} progressor) | {m} | {a['max']['value']:.3f} {cip(a['max']['ci95'])} | {a['sd']['value']:.3f} {cip(a['sd']['ci95'])} | {a['max_plus_sd']['value']:.3f} {cip(a['max_plus_sd']['ci95'])} | {o['max_plus_sd_minus_max']['delta']:+.3f} {ci(o['max_plus_sd_minus_max']['ci95'])} |")
A("\nEndoscopy level: endoscopies with ≥ 2 pre-event samples; patient level: SD over all of a patient's pre-event samples (patients with ≥ 2).\n")
# ---------------------------------------------------------------- L5
A("### L5. Who each modality misses\n")
gp = L5["grid_position_mean_0_80_1_90_2_95"]
A(f"P1 (H&E only) and P2 (copy number only) set to the same non-progressor ever-flagged count per repeat: target {L5['target_non_progressor_patients_flagged_mean']:.1f} of {L5['n_non_progressor_patients']} (mean over repeats; the mean of the two policies' 90% cells). "
  f"Mean grid position (0 = 80%, 1 = 90%, 2 = 95%): H&E {gp['H&E']:.2f}, copy number {gp['copy number']:.2f}; repeats clamped to a grid end: H&E {L5['repeats_clamped_to_grid_end']['H&E']}, copy number {L5['repeats_clamped_to_grid_end']['copy number']}. "
  f"Progressor patients caught: H&E {L5['progressor_patients_caught_mean']['H&E']:.1f}, copy number {L5['progressor_patients_caught_mean']['copy number']:.1f} of {L5['n_progressor_patients']}.\n")
ct = L5["cross_tab_progressor_patients_mean_over_repeats"]; cn = L5["cross_tab_non_progressor_patients_mean_over_repeats"]
A("| Progressor patients (mean over repeats) | Copy number caught | Copy number missed |\n|---|---|---|")
A(f"| H&E caught | {c5(ct['both'])} (both) | {c5(ct['H&E only'])} (H&E only) |"); A(f"| H&E missed | {c5(ct['copy number only'])} (copy number only) | {c5(ct['neither'])} (neither) |")
A(f"\nNon-progressor patients at the same operating point: both {c5(cn['both'])}, H&E only {c5(cn['H&E only'])}, copy number only {c5(cn['copy number only'])}, neither {c5(cn['neither'])}.\n")
A("Progressor patients by modal category (highest mean probability over repeats) and the patients discordant in at least one repeat, described in aggregate:\n")
A("| Group | Patients | Highest pre-event grade | Pre-event endoscopies, median [IQR] | Years first sample → event, median [IQR] | Tissue tertile 1 / 2 / 3 | Batch type | Scanner |\n|---|---|---|---|---|---|---|---|")
for k, o in L5["progressor_patients_by_modal_category"].items():
    if "highest_pre_event_grade" not in o: A(f"| {k} | {c5(o['n_patients'])} | {o.get('note', '')} | | | | | |"); continue
    tt = o["tissue_tertile"]; dct = lambda d: ", ".join(f"{a} {c5(b)}" for a, b in d.items())
    A(f"| {k} | {c5(o['n_patients'])} | {dct(o['highest_pre_event_grade'])} | {o['pre_event_endoscopies']['median']:.1f} [{o['pre_event_endoscopies']['iqr'][0]:.1f}, {o['pre_event_endoscopies']['iqr'][1]:.1f}] | "
      f"{o['years_first_sample_to_event']['median']:.1f} [{o['years_first_sample_to_event']['iqr'][0]:.1f}, {o['years_first_sample_to_event']['iqr'][1]:.1f}] | {' / '.join(c5(tt.get(str(t), 0)) for t in (1, 2, 3))} | {dct(o['batch_type'])} | {dct(o['scanner'])} |")
dg = L5["progressor_patients_by_modal_category"].get("discordant in at least one repeat (H&E only or copy number only)", {})
if "direction_mean_repeats" in dg: A(f"\nDiscordant-in-≥ 1-repeat group: mean per repeat {dg['direction_mean_repeats']['H&E only']:.2f} caught by H&E only and {dg['direction_mean_repeats']['copy number only']:.2f} by copy number only; modal category 'both' for {c5(dg['of_which_modal_both'])} of them.")
A("")
# ---------------------------------------------------------------- figures and notes
A("### Figures (local, not committed)\n")
A("`paper/figures_r8/`: `r8_a_caught_vs_np_flagged.png` (plot a; marker size = samples sequenced per patient), `r8_b_caught_vs_flags_per_py.png` (plot b), `r8_c_cost_vs_caught.png` (plot c, cost / caught frontier joined), "
  "`r8_l2_forest.png` (Δ by stratum), `r8_l3_trajectory.png` (mean endoscopy logit by years to reference, both modalities), `r8_l4_auroc.png`, `r8_l5_2x2.png`. Frontier points are numbered in plots a–c with a key beside each plot.\n")
A("### Implementation notes\n")
for t in [
    "Units and population: pre-event samples of the analysed cohort (571 samples, 288 endoscopies, 75 patients); endoscopy = (patient, endoscopy number) as rounds 6–7d, so the checks against R49, B\\*-endo and Table 1 (D) use the same units. A patient is in one outer fold per repeat, so every threshold applied to a patient's history comes from that fold.",
    "Thresholds (all policies): per (repeat, fold), on the fold's training patients' pre-event endoscopies scored by inner out-of-fold predictions, the largest threshold flagging ≥ the target share of training progressor endoscopies (`thr_sens`; endoscopy level). No policy uses a patient-level threshold: P7–P9 apply endoscopy-level thresholds along each patient's history. P5 and P6 use the stored sample-level triage cut-offs (`tb_common.triage`) at the 9 cells (t_low 90/95/98% sample sensitivity × t_high 85/90/95% specificity), as round 7b/7c.",
    "Endoscopy scores: P1 max L-IMG; P2 max L-CNV; P3 max L-LATE; P4 L-LATE of one random sample (R41 seeds 0–99; the same pick for an endoscopy whether it is in training or evaluation; one threshold per seed; outcomes averaged over seeds). Top-H&E sample = the sample with the highest L-IMG (inner in training, outer in evaluation).",
    "P7: the first pre-event endoscopy (earliest exact date) is sequenced (k = 1 top-H&E, 1 random, 2 top-H&E, all) and flagged if the max L-LATE over the sequenced samples ≥ t_b; 'always flag' is read as: a patient flagged there becomes high risk and **every later endoscopy is flagged** without further sequencing. Otherwise later endoscopies are flagged if max L-IMG ≥ t_s (the P1 threshold). t_b is set on all training endoscopies with the same k-sample score (for k = all it equals P3's threshold).",
    "P8: as P7 with k = 1 top-H&E; t_b keeps P7's grid (80/90/95%), so 3 × 2 cells; for a patient not high risk, a later endoscopy with max L-IMG ≥ t_mid (the P1 threshold at 80/90%) has its top-H&E sample sequenced and is flagged if that L-LATE ≥ t_c (the top-H&E L-LATE threshold at 90% training endoscopy sensitivity, unconditional on t_mid). Endoscopies below t_mid are not flagged.",
    "P9: the top-H&E sample is sequenced at the first endoscopy and at any endoscopy ≥ N years (exact dates, 365.25-day years) after the last sequenced one, flagged if its L-LATE ≥ t (90%); other endoscopies flagged if max L-IMG ≥ t_s (90%). A positive sequencing result does not change later rules.",
    "P10: grade ID/LGD at any sample of the endoscopy; grade ID/LGD or p53 aberrant on the samples with a p53 result (endoscopies and patients formed from those samples only), so its counts have their own denominators.",
    "Outcomes: patient caught / ever flagged = flagged at ≥ 1 pre-event endoscopy; time first flag → event = from the earliest flagged endoscopy's exact date to the event, median over caught progressors per repeat, then averaged; last-2 window = each patient's two latest pre-event endoscopies by exact date (for non-progressors the last sequenced endoscopy is the reference of the time axis); sequencing counts samples; cost = £85 × samples sequenced + £535 × flagged endoscopies (£471 + £44 + £20, round 7d), divided by all 75 patients.",
    "Dates: exact sample dates from `pre_event_cohort.Date` (CNV folder name), never the year-only slide-table date. Event = earliest exact date among the samples of the sheet event endoscopy (largest months before final among HGD/IMC samples, hz_prep.py); if none of its samples has an exact date, the date implied by the patient's analysed samples (exact date + (months before final − event months) × 30.44 days, midrange). Last sequenced endoscopy (non-progressors) = latest exact date over the patient's sequenced samples on the sheet. The endoscopy unit's date is its earliest sample date. Pre-event status keeps the study's months-before-final definition.",
    "Frontier: computed on observed single-model means (no intervals); dominance requires ≥ on all three criteria and > on one. P1 cells sequence nothing, so they are on the three-way frontier by construction whenever no other zero-sequencing point dominates them.",
    "L2: strata at the sample level for AUROC (H&E quintile edges from the repeat-mean L-IMG of all pre-event samples; tissue tertiles of tissue tiles; time to event from exact dates; position = the patient's first pre-event endoscopy). Folds lacking one class within a stratum are skipped (fsa). Call-change strata at the endoscopy level (quintile of the endoscopy's max repeat-mean L-IMG, highest grade, tissue tertile of the top-H&E sample, progressor endoscopy time to event with all non-progressor endoscopies).",
    "L3: endoscopy score = mean over the 10 repeats of the endoscopy's maximum logit (outer prediction, clipped at 1e-6); slopes by least squares per patient; Δ mean slope with a patient bootstrap of the 65 patients; mixed model `score ~ time × class + (1 | patient)` (statsmodels MixedLM, REML) on the same endoscopies; last − first by exact date. The trajectory plot bins endoscopies by years to the reference date; bins with < 5 patients are not plotted.",
    "L4: within-unit SD (ddof 1) of logit predictions per repeat; max + SD = sum of the two ranks among the units in that repeat (fixed equal weights), ranks held fixed in the bootstrap; fold-stratified AUROC with the unit's patient fold.",
    "L5: the common non-progressor ever-flagged target per repeat is the mean of P1's and P2's counts at their 90% cells; each policy is interpolated linearly in that count between adjacent grid cells, a patient's catch probability being the same mix of its two cells' calls; cross-tab cells are expected counts with the two mixes independent. Groups of fewer than 5 patients are not described.",
    "Privacy: only aggregates are committed; counts of 1–4 patients in patient cells are shown as '<5'; the row-level endoscopy date table stays on the cluster (killcoyne_mm/review_round8/)."]:
    A(f"- {t}")
txt = open(DOC).read(); head = txt[:txt.rindex("\n---") + 4]
open(DOC, "w").write(head + "\n" + "\n".join(L) + "\n"); print("rendered", DOC)

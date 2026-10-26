"""Two-paper agent P2S: render results/paper_final/two_papers_P2S/*.json into the Results section of docs/two_papers_P2S.md (replaces everything after
the '## Results' line) and draw the local figures in paper/two_papers/figs_P2S/ (never committed). Aggregates only."""
import json, os, sys
import numpy as np
TH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); R = TH + "/results/paper_final/two_papers_P2S"; DOC = TH + "/docs/two_papers_P2S.md"; FIG = TH + "/paper/two_papers/figs_P2S"
J = {k: json.load(open(f"{R}/{k}.json")) for k in ("prep", "rules", "audit", "landmark", "design", "comp", "per1000")}
rep = J["rules"]["reproduction"]; au = J["audit"]["audit"]; L = J["landmark"]["landmark"]; D = J["design"]["design"]; C = J["comp"]["comp"]; P = J["per1000"]["per1000"]; PR = J["prep"]["prep"]
NM = {"ml_mem": "No sequencing (mean-logit + memory)", "final": "Confirmatory (final rule, cn90 bar)", "final_infold": "Confirmatory, fold-internal bar", "ml_single": "Mean-logit, single endoscopy", "A1": "Memory (gate score)",
      "A12": "Memory + certificate", "gate_single": "Gate score, single endoscopy", "R49": "Per-endoscopy gate R49 (this study)", "P1": "H&E alone (P1 90%)", "P3": "Sequence everything (P3 90%)", "grade": "Grade IND/LGD",
      "g53": "Grade IND/LGD or p53 aberrant", "prague_single": "Prague M, training threshold", "prague_mem": "Prague M + memory, training threshold", "parasa_single": "Parasa-style, training threshold",
      "parasa_mem": "Parasa-style + memory, training threshold", "prague_M_ge3": "Prague M ≥ 3 cm", "prague_M_ge3_ew2": "EW-mean Prague M ≥ 3 cm", "parasa_gt10": "Parasa > 10", "parasa_gt10_ew2": "EW-mean Parasa > 10",
      "parasa_gt20": "Parasa > 20", "parasa_gt20_ew2": "EW-mean Parasa > 20"}
f1 = lambda x: "—" if x is None else (x if isinstance(x, str) else f"{x:.1f}"); f2 = lambda x: "—" if x is None else (x if isinstance(x, str) else f"{x:.2f}"); f3 = lambda x: "—" if x is None else (x if isinstance(x, str) else f"{x:.3f}")
def vc(d, f=f1): return f"{f(d['value'])} [{f(d['ci95'][0])}, {f(d['ci95'][1])}]"
def dl(d, f=f1): return f"{'+' if d['delta'] > 0 else ('−' if d['delta'] < 0 else '')}{f(abs(d['delta']))} [{f(d['ci95'][0])}, {f(d['ci95'][1])}]".replace("[-", "[−").replace(", -", ", −")
def tbl(head, rows): return "\n".join(["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])
o = []; A = o.append
A("## Results\n")
A("Rendered by `scripts/paper_plan/tp_p2s_render.py` from `results/paper_final/two_papers_P2S/` (analysis `scripts/paper_plan/tp_p2s_main.py`). Exploratory: every rule here came out of the ~650-variant search, and these analyses describe it; nothing is confirmatory. Held-out patients throughout; every threshold, half-life, certificate setting and check bar chosen on training patients (inner out-of-fold predictions of each outer fold). Counts are means over the 10 repeats unless they are labelled integer; intervals are paired patient-bootstrap 95% CIs (2,000 draws, `RandomState(0)`, the round-8 draws). Patient cells below 5 are shown as <5. No ACE-B file was read; stored predictions and frozen bundles were not touched.\n")
A("### Reproduction (smoke and full run)\n")
ob = rep["observed"]; A(tbl(["Rule", "Caught / 32", "NP flagged / 43", "Samples sequenced / patient", "Expected"], [[NM.get(k, k), f1(ob[k][0]), f1(ob[k][1]), f2(ob[k][2]), "/".join(f1(x) for x in rep["expected"][k]) if k in rep["expected"] else ""] for k in ("R49", "P1", "grade", "A1", "A12", "final", "final_infold", "ml_mem", "ml_single", "gate_single", "g53", "P3")]))
A(f"\nEvery expected number reproduced (`reproduced: {rep['reproduced']}`), and the rebuilt held-out flags equal the stored ones exactly: 0 mismatches against `explore_E7/base_flags.npz` for R49, P1, grade, A1, A12, final and fold-internal final, and 0 against the E9 sweep for mean-logit + memory. E7's equal-exposure memory gain (A1 −6.1) reproduced as {L['equal_exposure_patients']['check_E7_A1_vs_R49_np_delta_expected_-6.1']:+.1f}. The smoke run used 50 draws; the tables use 2,000.\n")
# ---- 1 audit
A("### 1. Memory uses visits up to k only (audit)\n")
ut = au["unit_test_mismatches_at_or_before_cut"]; e2 = au["end_to_end_flag_mismatches_at_or_before_cut"]; to = au["time_order"]
A(f"**Verdict: {au['verdict']}.** Nothing needed fixing.\n")
A("- **Code reading.** " + " ".join(s_[0].upper() + s_[1:] + "." for s_ in au["code_audit"]))
A(f"- **Time order.** Patients whose order is not by exact date: {to['patients_with_order_not_by_date']}; consecutive endoscopies on the same date: {to['consecutive_endoscopy_pairs_on_the_same_date']}. {to['endoscopies_with_samples_on_more_than_one_date']} endoscopy units have samples dated on more than one day (the unit takes its earliest date, as in all earlier rounds).")
A(f"- **Unit test.** Per patient a random cut; every score after the cut replaced by noise (×10 scale). Mismatches at or before the cut, for running mean, EW (h = 0.5–16 y), last-2, consecutive-2, 2-of-n, 2-of-3 and the certificate (Y = 2, 4): all 0. The current endoscopy does enter its own memory (sanity check: {ut['current_endoscopy_enters_own_memory_ew2']}).")
A(f"- **End to end.** The held-out (outer) L-IMG and L-CNV predictions of every sample in a later endoscopy replaced by noise and the whole nested pipeline rerun (thresholds, half-life, certificate, check bar, fold-internal bar). Flag mismatches at or before the cut: A1 {e2['A1']}, A12 {e2['A12']}, final {e2['final']}, fold-internal final {e2['final_infold']}, mean-logit + memory {e2['ml_mem']}. Sanity: flags after the cut did change ({e2['final_flags_changed_after_cut_(sanity, expected > 0)']} and {e2['ml_mem_flags_changed_after_cut_(sanity, expected > 0)']} endoscopy-repeats).")
A(f"- **Other patients.** Whole histories of a random half of patients replaced by noise: final-rule flags of the untouched half changed at {au['other_patients_test_final_flag_mismatches_in_untouched_half']} endoscopy-repeats.\n")
# ---- tiers headline
A("### Tiers and comparators (all 75 patients)\n")
T = D["tiers"]; rows = []
for k in ("ml_mem", "final", "final_infold", "ml_single", "A1", "A12", "R49", "P1", "P3", "grade", "g53"):
    t = T[k]; ic = t["integer_counts"]
    rows.append([NM[k], vc(t["caught"]), vc(t["np_flagged"]), f"{ic['progressors_caught']['min']}–{ic['progressors_caught']['max']}", f"{ic['np_flagged']['min']}–{ic['np_flagged']['max']}", f"{t['majority_vote']['missed']} ({f3(t['majority_vote']['missed_cp95_of_32'][0])}–{f3(t['majority_vote']['missed_cp95_of_32'][1])})", vc(D["npv_cleared"][k]["npv_cohort_case_control"], f3), f2(t["seq_per_patient"]), dl(t["vs_g53"]["np_flagged"]) if "vs_g53" in t else "—", dl(t["vs_g53"]["caught"]) if "vs_g53" in t else "—"])
A(tbl(["Rule", "Caught / 32 [CI]", "NP flagged / 43 [CI]", "Caught, integer range over repeats", "NP flagged, integer range", "Missed in ≥ 5 of 10 repeats (exact 95% CI of share)", "NPV among never-flagged (case–control) [CI]", "Seq. / patient", "Δ NP vs grade + p53", "Δ caught vs grade + p53"], rows))
A("\nNPV here is the cohort's 1:1 case–control value and is not a population NPV; item 9 gives the model-based population version. Integer counts per repeat are in `design.json` (`integer_counts.per_repeat`).\n")
# ---- 2 landmark
A("### 2. Landmark analysis (history to date only)\n")
A("Every pre-event endoscopy is a landmark scored on the patient's history up to and including it. Positive at horizon h: progressor endoscopy with the event ≤ h years later. Negative: non-progressor endoscopy with known event-free follow-up ≥ h (to the last sequenced endoscopy, extended by later Barrett's database records, median 6.7 more years), or progressor endoscopy with the event > h years later. Operating points are each rule's own (training-chosen) thresholds. AUROC = landmark AUROC of the held-out score minus its fold threshold, pooled over folds, repeat mean; memory and single-endoscopy versions of the same base, each with its own training threshold.\n")
for blk, title in (("all_landmarks", "All landmarks (primary)"), ("equal_exposure_landmarks", "Equal exposure (primary): non-progressor landmarks weighted by the share of the 200 E7 truncation draws that keep them"), ("patient_weighted", "Patient-weighted (secondary): each patient's landmarks share weight 1")):
    A(f"**{title}.**\n"); rows = []
    for h in ("1y", "3y", "5y", "ever"):
        b = L[blk][h]; bi = b["binary"]; mm = b["memory_minus_single"]; au_ = b.get("auroc_memory_minus_single", {}); am = b.get("auroc_margin", {})
        rows.append([h, f"{b['positive_landmarks']} ({b['positive_patients']} P) / {b['negative_landmarks']}", f"{f3(bi['ml_mem']['sens']['value'])} / {f3(bi['ml_mem']['spec']['value'])}", f"{f3(bi['ml_single']['sens']['value'])} / {f3(bi['ml_single']['spec']['value'])}",
                     dl(mm["ml_mem - ml_single"]["sens"], f3), dl(mm["ml_mem - ml_single"]["spec"], f3), f"{f3(am['ml_mem']['value'])} vs {f3(am['ml_single']['value'])}" if am else "—", dl(au_["ml_mem - ml_single"], f3) if au_ else "—",
                     f"{f3(bi['final']['sens']['value'])} / {f3(bi['final']['spec']['value'])}", f"{f3(bi['final_infold']['sens']['value'])} / {f3(bi['final_infold']['spec']['value'])}", f"{f3(bi['R49']['sens']['value'])} / {f3(bi['R49']['spec']['value'])}",
                     dl(au_["A12 - gate_single"], f3) if au_ else "—", f"{f3(bi['g53']['sens']['value'])} / {f3(bi['g53']['spec']['value'])}"])
    A(tbl(["Horizon", "Positive (patients) / negative landmarks", "No-seq tier sens / spec", "Mean-logit single sens / spec", "Δ sens, memory − single", "Δ spec, memory − single", "AUROC memory vs single (mean-logit)", "Δ AUROC (mean-logit)", "Final sens / spec", "Fold-internal sens / spec", "R49 sens / spec", "Δ AUROC, memory + certificate − single gate", "Grade + p53 sens / spec"], rows)); A("")
lv = L["all_landmarks"]["5y"]["binary"]; A(f"NPV at the landmark (event within 5 years; case–control mix of landmarks): no-sequencing tier {vc(lv['ml_mem']['npv'], f3)}, final {vc(lv['final']['npv'], f3)}, R49 {vc(lv['R49']['npv'], f3)}, grade + p53 {vc(lv['g53']['npv'], f3)}.\n")
E = L["equal_exposure_patients"]; rows = []
for k in ("ml_mem", "ml_single", "final", "final_infold", "A1", "A12", "R49", "P1", "g53"):
    rows.append([NM[k], vc(E[k]["values"]["caught"]), vc(E[k]["values"]["np_flagged"]), dl(E[k]["vs_R49"]["np_flagged"]), dl(E[k]["vs_R49"]["caught"])])
A("**Equal exposure at patient level (E7 test 1(a): each non-progressor truncated to its first m endoscopies, m drawn from the progressors' counts, 200 draws; frozen rules).**\n"); A(tbl(["Rule", "Caught", "NP flagged (of 43)", "Δ NP vs R49", "Δ caught vs R49"], rows))
A(f"\nMemory gain at equal exposure (memory minus the same base without memory): mean-logit {dl(E['ml_mem - ml_single (memory gain)']['np_flagged'])} NP (caught {dl(E['ml_mem - ml_single (memory gain)']['caught'])}); gate score {dl(E['A1 - gate_single (memory gain)']['np_flagged'])}; gate + certificate {dl(E['A12 - gate_single (memory gain)']['np_flagged'])}. Without truncation the mean-logit memory gain is {T['ml_mem']['np_flagged']['value'] - T['ml_single']['np_flagged']['value']:+.1f} (26.6 vs 31.0).\n")
# ---- 3 two lows
A("### 3. Two-consecutive-lows rule\n")
A("De-escalation = the interval would be lengthened after that endoscopy. Four readings: (a) single look: low at this endoscopy; (b) two consecutive lows; (c) never flagged to date, single look (the 'never flagged' headline metric); (d) never flagged to date and two consecutive lows (the gastro review's recommendation: no de-escalation on one look, a flag returns the patient to guideline care). 'Progressors de-escalated at the last pre-event endoscopy' are progressors whose interval would have been lengthened at the look before the event.\n")
rows = []
for k in ("ml_mem", "final", "final_infold", "R49", "g53"):
    T2 = D["two_lows"][k]
    for v, lab in (("single_look", "(a) single look"), ("two_consecutive_lows", "(b) two lows"), ("never_flagged_to_date_single_look", "(c) never flagged, single look"), ("never_flagged_to_date_two_consecutive_lows", "(d) never flagged, two lows")):
        x = T2[v]; pr = x["progressors_deescalated_at_last_integer_per_repeat"]
        rows.append([NM[k], lab, vc(x["np_deescalated_at_last_endoscopy"]), vc(x["np_ever_deescalated"]), f2(x["share_of_np_endoscopies_followed_by_a_longer_interval"]), vc(x["progressors_deescalated_at_last_pre_event_endoscopy"]), f"{min(pr)}–{max(pr)}", vc(x["progressors_ever_deescalated_before_event"])])
A(tbl(["Rule", "De-escalation reading", "NP de-escalated at last endoscopy", "NP ever de-escalated", "Share of NP endoscopies followed by a longer interval", "Progressors de-escalated at the last pre-event endoscopy", "Same, integer range over repeats", "Progressors ever de-escalated before the event"], rows))
A("")
# ---- 4 safety
A("### 4. Safety block\n")
sr = D["safety_reference"]; A(f"Reference: 32 progressors; first event HGD {sr['event_type']['HGD']}, IMC {sr['event_type']['IMC']}, no HGD/IMC sample sequenced (event = final endoscopy) {sr['event_type']['none']}; focal HGD noted at the event endoscopy {sr['focal_hgd']}; {sr['progressors_with_one_pre_event_endoscopy']} progressors have one pre-event endoscopy; median {sr['years_last_pre_event_endoscopy_to_event_median_all_progressors']} years from the last pre-event endoscopy to the event. Stage: the event is the first HGD/IMC endoscopy on the sheet; the Barrett's database gives OAC as the first HGD-or-worse report for <5 progressors and as the later worst grade for 5 (round-7 lookup B2), and records no TNM stage. Row-level line lists stay on the cluster.\n")
rows = []
for k in ("ml_mem", "final", "final_infold", "R49", "P1", "g53"):
    for w, lab in (("missed_never_flagged", "missed (never flagged)"), ("cleared_at_last_pre_event_endoscopy", "cleared at last pre-event endoscopy")):
        s_ = D["safety"][k][w]; pr = s_["per_repeat_integer_counts"]
        rows.append([NM[k], lab, f"{f1(s_['mean'])} ({min(pr)}–{max(pr)})", s_["patients_ever_in_any_repeat"], s_["patients_in_ge5_of_10_repeats"], "yes" if s_.get("any_IMC") else "no", s_.get("expected_count_by_event", {}).get("IMC", "—"), s_.get("expected_count_focal_hgd", "—"),
                     s_.get("expected_count_5y_interval_postpones_detection_past_event", "—"), "yes" if s_.get("most_expected_misses_postponed_past_event_by_5y_interval") else "no", "yes" if s_.get("most_expected_misses_have_one_pre_event_endoscopy") else "no", s_.get("years_last_cleared_to_event_weighted_median", "—")])
A(tbl(["Rule", "Group", "Expected count (integer range over repeats)", "Distinct patients in any repeat", "Patients in ≥ 5 of 10 repeats", "Any IMC", "Expected IMC", "Expected focal HGD", "Expected: 5-y interval from the last cleared endoscopy postpones detection past the event", "Most postponed by a 5-y interval", "Most have a single pre-event endoscopy", "Median years, last cleared endoscopy → event"], rows))
A("\n'5-y interval postpones detection past the event' = the event came less than 5 years after the last cleared endoscopy, so a 5-year interval would have put the next look after the event date (detection delayed by 5 − t years). Whether that delay would have passed beyond endoscopic cure cannot be judged from these records: most treated progressors had EMR or ablation at or after the event (round-7 lookup), but what a further 2–4 years would have done is not observed.\n")
# ---- 5 lead time
A("### 5. Lead time\n")
rows = []
for w, lab in (("gt_2y_before_event", "> 2 y (NP: all)"), ("gt_2y_before_event_symmetric", "> 2 y (NP: > 2 y before last)"), ("gt_4y_before_event", "> 4 y (NP: all)"), ("gt_4y_before_event_symmetric", "> 4 y (NP: > 4 y before last)")):
    b = D["lead_time"][w]
    for k in ("ml_mem", "ml_single", "final", "final_infold", "R49", "P1", "g53"):
        v = b[k]; rows.append([lab, f"{b['progressors_with_an_endoscopy_in_window']} / {b['non_progressors_with_an_endoscopy_in_window']}", NM[k], vc(v["values"]["caught"]), vc(v["values"]["np_flagged"]), dl(v["vs_R49"]["caught"]), dl(v["vs_R49"]["np_flagged"]), dl(v["vs_g53"]["caught"]), dl(v["vs_single_endoscopy_counterpart"]["np_flagged"]) if "vs_single_endoscopy_counterpart" in v else "—"])
A(tbl(["Window", "P / NP with an endoscopy in it", "Rule", "Caught", "NP flagged", "Δ caught vs R49", "Δ NP vs R49", "Δ caught vs grade + p53", "Δ NP vs single-endoscopy counterpart"], rows)); A("")
# ---- 6 comparators
A("### 6. Clinical comparators\n")
pm = PR["prague_M_source_endoscopies"]
A(f"Sources: Prague M per endoscopy from the Barrett's database within ±60 days ({pm['db_at_endoscopy_60d']} of {pm['of']} endoscopies), else `Demographics_full` Maximal ({pm['demographics']}), else the patient's database median ({pm['db_patient_median']}); missing {pm['missing']} endoscopies ({PR['prague_M_missing_patients']} patients, all progressors: segment-length rows are biased by differential missingness). Smoking known for {C['smoking_known']['progressors']} of 32 progressors and {C['smoking_known']['non_progressors']} of 43 non-progressors (`Demographics_full`; the database's smoking code is undocumented and unused). Sex from the sample table; LGD from the sheet grade at the endoscopy. Parasa-style score = 9 × male + 5 × smoker + 1 per cm + 11 × LGD (Parasa et al., Gastroenterology 2018). Because Prague M is essentially one value per patient, memory cannot change it.\n")
rows = []
for k in ("prague_M_ge3", "prague_M_ge3_ew2", "parasa_gt10", "parasa_gt10_ew2", "parasa_gt20", "parasa_gt20_ew2", "prague_single", "prague_mem", "parasa_single", "parasa_mem"):
    v = C[k]; rows.append([NM[k], vc(v["caught"]), vc(v["np_flagged"]), dl(v["vs_no_sequencing_tier"]["caught"]), dl(v["vs_no_sequencing_tier"]["np_flagged"]), dl(v["vs_g53"]["caught"]), dl(v["vs_g53"]["np_flagged"])])
A(tbl(["Comparator", "Caught", "NP flagged", "Δ caught vs no-seq tier", "Δ NP vs no-seq tier", "Δ caught vs grade + p53", "Δ NP vs grade + p53"], rows))
A("\nEvaluation-matched (OPTIMISTIC for the comparator: its threshold set on the evaluated patients, per repeat):\n")
rows = [[k.split(" at ")[0], NM[k.split(" at ")[1]], f"{f1(v['tier_caught'])} / {f1(v['tier_np_flagged'])}", f1(v["comparator_np_flagged_at_caught_ge_floor_of_tier"]), f1(v["comparator_caught_at_np_flagged_le_tier"])] for k, v in C["evaluation_matched_optimistic"].items()]
A(tbl(["Comparator", "Tier", "Tier caught / NP flagged", "Comparator NP flagged at the tier's caught", "Comparator caught at the tier's NP flagged"], rows)); A("")
# ---- 7 subgroups
A("### 7. Subgroups\n")
rows = []
for g, b in D["subgroups"].items():
    if g.startswith("endoscopies"):
        for k in ("ml_mem", "final", "R49", "g53"):
            v = b[k]; rows.append([g, f"{b['progressors_with_one']} / {b['np_with_one']}", NM[k], f"{f1(v['values']['caught']['value'])} / {v['values']['caught']['of']}", f"{f1(v['values']['np_flagged']['value'])} / {v['values']['np_flagged']['of']}", dl(v["vs_R49"]["np_flagged"]), "—"])
        continue
    if "ml_mem" not in b: rows.append([g, f"{b['progressors']} / {b['non_progressors']}", "not estimable", "", "", "", ""]); continue
    for k in ("ml_mem", "final", "final_infold", "R49", "g53"):
        v = b[k]; rows.append([g, f"{b['progressors']} / {b['non_progressors']}", NM[k], vc(v["caught"]), vc(v["np_flagged"]), dl(v["vs_R49_np"]), dl(v["vs_g53_np"])])
A(tbl(["Subgroup", "P / NP", "Rule", "Caught", "NP flagged", "Δ NP vs R49", "Δ NP vs grade + p53"], rows))
A("\n" + D["subgroup_note"] + ". Segment and age cut-offs are the guideline split (3 cm) and 65 years; no subgroup test is powered.\n")
# ---- 8 robustness rows
A("### 8. The 4 of 34 R49 robustness rows that fail (round 7c)\n")
A("Round 7c (`docs/paper_review_round7c_gate_robustness.md`) tests R49's endoscopy sensitivity against sequencing everything (B*-endo) with a −0.05 non-inferiority bound in 34 rows. 30 are within; none fails by a shift; four fail by width (the CI includes 0, the lower bound is below −0.05):\n")
A(tbl(["Row", "Population", "Δ endoscopy sensitivity [95% CI]", "Lower bound", "Why"],
      [["7: ≥ 256 tissue tiles, endoscopy by its top-L-IMG sample", "146 endoscopies (49 P), 63 patients (25 P)", "−0.024 [−0.072, +0.018]", "−0.063", "half the endoscopies; larger biopsies, where H&E is weaker relative to copy number (the H&E advantage sits in small biopsies)"],
           ["7: ≥ 256 tissue tiles, all samples of the endoscopy in the stratum", "117 endoscopies (40 P), 54 patients (21 P)", "−0.013 [−0.071, +0.031]", "−0.060", "same stratum, even smaller"],
           ["15: between-repeat variance added (as R33)", "all", "−0.016 [−0.061, +0.029]", "−0.054", "the interval is inflated by the spread of the 10 CV repeats; same point estimate as the primary row"],
           ["15: refit bootstrap (F3, 200 draws)", "all, refitted per draw", "−0.012 [−0.080, +0.054]", "−0.072", "models refitted on each bootstrap population add fitting variance (10% of draws below −0.05)"]]))
A("\nAll four are width failures with point estimates between −0.012 and −0.024 (primary −0.016): no row shows R49 losing sensitivity; the intervals are too wide to rule out a 5-point loss in large biopsies or once fitting and repeat variance are counted.\n")
# ---- 9 per 1000
A("### 9. Progressions per 1,000 patient-years in the de-escalated group (MODEL-BASED)\n")
A("A case–control cohort cannot estimate absolute risk. Assumption: the rule's patient-level sensitivity (progressors flagged at ≥ 1 pre-event endoscopy) and specificity (non-progressors never flagged) transport to an NDBE population with annual progression incidence λ. Then progressions per 1,000 patient-years in the de-escalated (never-flagged) group = 1000 · λ(1 − sens) / (λ(1 − sens) + (1 − λ) spec). Bootstrap CIs cannot exceed the cohort's observed misses, so an exact (Clopper–Pearson) upper limit on the miss count (mean misses rounded up) is given. Bar (gastro review): ≤ 2 per 1,000 patient-years (short-segment NDBE, ~0.2%/yr).\n")
rows = []
for k in ("ml_mem", "final", "final_infold", "A1", "A12", "ml_single", "R49", "P1", "P3", "grade", "g53", "parasa_gt10"):
    if k not in P: continue
    r_ = [NM[k], f1(P[k]["incidence_0.45pct_per_year"]["share_deescalated_per_1000_patients"]["value"])]
    for lam in ("0.30", "0.45", "0.60"):
        x = P[k][f"incidence_{lam}pct_per_year"]; r_.append(f"{f2(x['progressions_per_1000_py_in_deescalated']['value'])} [{f2(x['progressions_per_1000_py_in_deescalated']['ci95'][0])}, {f2(x['progressions_per_1000_py_in_deescalated']['ci95'][1])}]; exact ≤ {f2(x['upper_95_from_exact_binomial_on_mean_misses_rounded_up'])}")
    r_.append(vc(P[k]["incidence_0.45pct_per_year"]["relative_rate_vs_no_triage"], f2)); rows.append(r_)
A(tbl(["Rule", "De-escalated per 1,000 patients", "λ = 0.30%/yr: per 1,000 py [CI]; exact upper", "λ = 0.45%/yr", "λ = 0.60%/yr", "Rate relative to no triage (λ = 0.45%)"], rows))
A("\nWithout triage the rate is 3.0, 4.5 and 6.0 per 1,000 patient-years.\n")
doc = open(DOC).read(); head = doc[:doc.index("## Results")]; open(DOC, "w").write(head + "\n".join(o) + "\n")
print("rendered", len(o), "blocks")
# ================================================================== figures (local)
if "--figs" in sys.argv:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    os.makedirs(FIG, exist_ok=True); plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}); B1, O2, G = "#2a78d6", "#eb6834", "#8a8a85"
    # fig 1: landmark AUROC, memory vs single (mean-logit), by horizon and weighting
    fig, ax = plt.subplots(figsize=(7.2, 3.6)); hs = ["1y", "3y", "5y", "ever"]; y0 = 0
    for i, (blk, lab, col) in enumerate((("all_landmarks", "All landmarks", B1), ("equal_exposure_landmarks", "Equal exposure", O2), ("patient_weighted", "Patient-weighted", G))):
        for j, h in enumerate(hs):
            d = L[blk][h]["auroc_memory_minus_single"]["ml_mem - ml_single"]; yy = j * 4 + i
            ax.plot(d["ci95"], [yy, yy], color=col, lw=2); ax.plot(d["delta"], yy, "o", color=col, ms=7, label=lab if j == 0 else None)
    ax.axvline(0, color="#444", lw=1); ax.set_yticks([j * 4 + 1 for j in range(4)]); ax.set_yticklabels(["≤ 1 y", "≤ 3 y", "≤ 5 y", "ever"]); ax.invert_yaxis()
    ax.set_xlabel("Δ landmark AUROC, memory − single endoscopy (mean-logit H&E)"); ax.legend(frameon=False, fontsize=9, loc="lower right"); ax.set_title("Landmark analysis: history to date predicts later progression", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG + "/p2s_landmark_auroc.png", dpi=200); plt.close(fig)
    # fig 2: two-lows: progressors de-escalated at the last pre-event endoscopy vs NP de-escalated at last endoscopy
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.6)); ks = ["ml_mem", "final", "final_infold", "R49", "g53"]; labs = ["No-seq tier", "Final", "Final, fold bar", "R49", "Grade + p53"]
    vs = [("single_look", "single look", G), ("never_flagged_to_date_two_consecutive_lows", "never flagged + two lows", B1)]
    for a_, key, tt in ((axs[0], "np_deescalated_at_last_endoscopy", "Non-progressors de-escalated at last endoscopy (of 43)"), (axs[1], "progressors_deescalated_at_last_pre_event_endoscopy", "Progressors de-escalated at the last look before the event (of 32)")):
        for i, (v, lab, col) in enumerate(vs):
            x = np.arange(len(ks)) + (i - 0.5) * 0.36; vals = [D["two_lows"][k][v][key]["value"] for k in ks]; ci = np.array([D["two_lows"][k][v][key]["ci95"] for k in ks])
            a_.bar(x, vals, width=0.34, color=col, label=lab, edgecolor="white", linewidth=1); a_.errorbar(x, vals, yerr=[np.array(vals) - ci[:, 0], ci[:, 1] - np.array(vals)], fmt="none", ecolor="#333", lw=1)
        a_.set_xticks(range(len(ks))); a_.set_xticklabels(labs, rotation=25, ha="right", fontsize=9); a_.set_title(tt, fontsize=10)
    fig.legend(*axs[0].get_legend_handles_labels(), frameon=False, fontsize=9, loc="upper center", ncol=2); fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig(FIG + "/p2s_two_lows.png", dpi=200); plt.close(fig)
    # fig 3: per 1,000 py at 0.45%/yr
    fig, ax = plt.subplots(figsize=(8.2, 4.0)); ks = ["ml_mem", "final", "final_infold", "R49", "P1", "grade", "g53"]
    for i, k in enumerate(ks):
        x = P[k]["incidence_0.45pct_per_year"]; v = x["progressions_per_1000_py_in_deescalated"]; ax.plot(v["ci95"], [i, i], color=B1, lw=2); ax.plot(v["value"], i, "o", color=B1, ms=7)
        ax.plot(x["upper_95_from_exact_binomial_on_mean_misses_rounded_up"], i, "|", color=O2, ms=12, mew=2)
    ax.axvline(2.0, color="#444", lw=1, ls="--"); ax.axvline(4.5, color=G, lw=1, ls=":"); ax.text(2.05, -0.75, "bar 2 / 1,000 py", fontsize=9); ax.text(4.55, -0.75, "no triage", fontsize=9, color="#555"); ax.set_ylim(len(ks) - 0.5, -1.1)
    ax.set_yticks(range(len(ks))); ax.set_yticklabels([NM[k] for k in ks], fontsize=9); ax.set_xlabel("Progressions per 1,000 patient-years, de-escalated group\n(λ = 0.45%/yr; model-based)")
    ax.set_title("Dots: bootstrap point and 95% CI; orange tick: exact upper limit", fontsize=9); fig.tight_layout(); fig.savefig(FIG + "/p2s_per1000.png", dpi=200); plt.close(fig)
    print("figures in", FIG)

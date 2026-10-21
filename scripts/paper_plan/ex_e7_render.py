"""Render the E7 results (results/paper_final/explore_E7/base.json, merge.json) as markdown on stdout (appended to docs/paper_explore_E7.md by hand)."""
import json, os, sys
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../results/paper_final/explore_E7")
B = json.load(open(R + "/base.json")); MG = json.load(open(R + "/merge.json")) if os.path.exists(R + "/merge.json") else {}
f1 = lambda x: "–" if x is None else f"{x:+.1f}"; f0 = lambda x: "–" if x is None else f"{x:.1f}"
def ci(d, k="np_flagged", sgn=True):
    x = d[k]; a, b = x["ci95"]; fm = (lambda v: f"{v:+.1f}") if sgn else (lambda v: f"{v:.1f}")
    return f"{fm(x['delta'])} [{fm(a)}, {fm(b)}]" if a is not None else fm(x["delta"])
def ci3(d, k):
    x = d[k]; a, b = x["ci95"]; return f"{x['delta']:+.3f} [{a:+.3f}, {b:+.3f}]"
L = []; P = L.append
NM = {"A1": "A1 memory", "A12": "A12 memory + certificate", "final": "final (A12 + check, cn90 bar)", "final_infold": "final, check bar re-chosen in fold", "P1": "P1 H&E alone 90%", "grade": "pathologist grade ID/LGD"}
P("### Reproduction\n")
P(f"Engine re-implementation against the stored round-9 flags: {B['reproduction']}. Observed (all patients): " + "; ".join(f"{NM.get(k, k)} {v['caught']} caught / {v['np_flagged']} NP flagged / {v['seq_per_patient']} samples per patient" for k, v in B["check"].items()) + ".\n")
P("### 1. Exposure equalised\n")
P("**(a) Frozen rules, truncated held-out histories** (memory is causal, so the scores on kept endoscopies are those of the full rule; " + B["test1_exposure_frozen"]["note"] + ").\n")
P("| Truncation | NP endoscopies per NP patient | R49 NP flagged / caught | Rule | NP flagged | caught | Δ NP vs R49 [95% CI] | Δ caught vs R49 | Δ NP vs P1 | Δ flags per NP patient-year vs R49 | spread of Δ NP over draws |")
P("|---|---|---|---|---|---|---|---|---|---|---|")
for nm, o in B["test1_exposure_frozen"].items():
    if nm == "note": continue
    r = o["R49"]["values"]
    for k in ("A1", "A12", "final", "P1"):
        v = o[k]; sp = v["per_draw_delta_np_vs_R49"]
        P(f"| {nm} ({o['draws']} draw{'s' if o['draws'] > 1 else ''}) | {o['np_endoscopies_per_np_patient']} | {f0(r['np_flagged']['value'])} / {f0(r['caught']['value'])} | {NM[k]} | {f0(v['values']['np_flagged']['value'])} | {f0(v['values']['caught']['value'])} | {ci(v['vs_R49'])} | {ci(v['vs_R49'], 'caught')} | {ci(v['vs_P1']) if k != 'P1' else ''} | {ci3(v['vs_R49'], 'flags_per_np_py')} | {sp['mean']} ({sp['p2.5_p97.5'][0]} to {sp['p2.5_p97.5'][1]}) |")
if MG.get("test1_exposure_full_rerun"):
    P("\n**(b) Whole selection rerun on the truncated cohort** (" + MG["test1_exposure_full_rerun"]["note"] + ").\n")
    P("| Truncation | draws | R49 NP flagged / caught | Rule | NP flagged | caught | Δ NP vs R49 [95% CI] | Δ caught vs R49 |"); P("|---|---|---|---|---|---|---|---|")
    for nm, o in MG["test1_exposure_full_rerun"].items():
        if nm == "note": continue
        r = o["R49"]["values"]
        for k in ("A1", "A12", "final", "final_infold"):
            v = o[k]; P(f"| {nm} | {o['draws']} | {f0(r['np_flagged']['value'])} / {f0(r['caught']['value'])} | {NM[k]} | {f0(v['values']['np_flagged']['value'])} | {f0(v['values']['caught']['value'])} | {ci(v['vs_R49'])} | {ci(v['vs_R49'], 'caught')} |")
def draws_table(T, title):
    P(f"| {title} | Rule | NP flagged | caught | Δ NP vs R49 [95% CI] | Δ caught vs R49 [95% CI] | Δ NP vs P1 | Δ NP over draws (2.5–97.5%) |"); P("|---|---|---|---|---|---|---|---|")
    for var, o in T.items():
        if var == "note" or not isinstance(o, dict) or "A1" not in o: continue
        for k in ("A1", "A12", "final", "final_infold"):
            v = o[k]; sp = v["per_draw_delta_np_vs_R49"]
            P(f"| {var} ({o['draws']} draws) | {NM[k]} | {f0(v['values']['np_flagged']['value'])} | {f0(v['values']['caught']['value'])} | {ci(v['vs_R49'])} | {ci(v['vs_R49'], 'caught')} | {ci(v['vs_P1'])} | {sp['mean']} ({sp['p2.5_p97.5'][0]} to {sp['p2.5_p97.5'][1]}) |")
if MG.get("test2_order_shuffle"):
    P("\n### 2. Order shuffled within patient\n"); P(MG["test2_order_shuffle"]["note"] + ". Unshuffled: A1 −8.2, A12 −10.3, final −14.6 NP flagged against R49.\n"); draws_table(MG["test2_order_shuffle"], "Variant")
P("\n### 3. Endoscopies close to the event excluded\n")
P("Caught counted out of all 32 (a progressor with no endoscopy left counts as missed). Rules frozen (thresholds from full training histories).\n")
P("| Exclusion | progressors with an endoscopy left | R49 | A1 | A12 | final | P1 | grade ID/LGD | Δ final vs R49 | Δ final vs grade | Δ NP vs R49 (symmetric version): A1 / final |"); P("|---|---|---|---|---|---|---|---|---|---|---|")
for nm, o in B["test3_prevalent_exclusion"].items():
    g = lambda k: o[k]["caught_of_32"]["value"]; sym = "vs_R49" in o["A1"]
    P(f"| {nm} | {o['progressors_with_an_evaluated_endoscopy']} | {g('R49')} | {g('A1')} | {g('A12')} | {g('final')} | {g('P1')} | {g('grade')} | {ci(o['final']['caught_of_32'], 'delta_vs_ref')} | {ci(o['final'], 'caught_of_32_vs_grade') if False else f1(o['final']['caught_of_32_vs_grade']['delta'])} | {(ci(o['A1']['vs_R49']) + ' / ' + ci(o['final']['vs_R49'])) if sym else ''} |")
P("\n### 4. One look per patient\n")
P("| Variant | caught | NP flagged | samples per patient | Δ caught vs A12 [95% CI] | Δ NP vs A12 | Δ caught vs R49 | Δ NP vs R49 |"); P("|---|---|---|---|---|---|---|---|")
for nm, v in B["test4_one_look"].items():
    if "values" not in v: continue
    P(f"| {nm} | {v['values']['caught']} | {v['values']['np_flagged']} | {v['values']['seq_per_patient']} | {ci(v['vs_A12'], 'caught')} | {ci(v['vs_A12'])} | {ci(v['vs_R49'], 'caught')} | {ci(v['vs_R49'])} |")
fe = B["test4_one_look"]["first_endoscopy_only_fixed_cn90"]
P(f"\nFirst endoscopy only (ACE-B-like; {fe['note']}): R49 {fe['R49']['caught']} caught / {fe['R49']['np_flagged']} NP flagged; A12 {fe['A12']['caught']} / {fe['A12']['np_flagged']}; A12 + check {fe['A12_plus_check']['caught']} / {fe['A12_plus_check']['np_flagged']} "
  f"(check vs A12: Δ caught {ci(fe['check_vs_A12'], 'caught')}, Δ NP {ci(fe['check_vs_A12'])}).\n")
P("### 5. Check bar sweep\n")
P("| Bar (share of training progressor endoscopies above it) | caught | NP flagged | Δ NP vs R49 [95% CI] | Δ caught vs R49 [95% CI] | Δ caught vs A12 [95% CI] | Δ NP vs A12 |"); P("|---|---|---|---|---|---|---|")
for nm, v in B["test5_bar_sweep"].items():
    P(f"| {nm}{' (picks: ' + ', '.join(f'{a}: {b}' for a, b in sorted(v['picks'].items())) + ')' if 'picks' in v else ''} | {v['values']['caught']} | {v['values']['np_flagged']} | {ci(v['vs_R49'])} | {ci(v['vs_R49'], 'caught')} | {ci(v['vs_A12'], 'caught')} | {ci(v['vs_A12'])} |")
if MG.get("test6_missing_records"):
    P("\n### 6. Missing records\n"); P(MG["test6_missing_records"]["note"] + ".\n")
    for p in ("20pct", "40pct"): P(f"\n**{p} missing**\n"); draws_table(MG["test6_missing_records"][p], "Variant")
if MG.get("test7_permutation"):
    T7 = MG["test7_permutation"]; P("\n### 7. Optimism of the search (label permutation)\n"); P(T7["note"] + ".\n")
    re = T7["real"]; P(f"Real labels: R49 {re['R49']['caught']} / {re['R49']['np_flagged']}; nested " + "; ".join(f"{NM[k]} {v['caught']} / {v['np_flagged']}" for k, v in re["nested"].items()) +
                     f"; best of grid (optimistic, {re['grid_size']} cells) {re['best_of_grid_optimistic']}; best of grid by excess {re.get('best_of_grid_excess_optimistic')}.\n")
    P("| Null | permutations | Rule | null Δ NP vs R49: mean (2.5–97.5%) | null Δ caught mean | null excess Δ share (NP − progressor): mean (2.5–97.5%) | real Δ NP | real excess | share of null at or below real excess |"); P("|---|---|---|---|---|---|---|---|---|")
    for kind in ("free", "strat"):
        if kind not in T7: continue
        o = T7[kind]
        for k in ("A1", "A12", "final", "final_infold", "best_of_grid_excess"):
            if k not in o: continue
            v = o[k]; P(f"| {kind} | {o['permutations']} | {NM.get(k, k + ' (OPTIMISTIC)')} | {v.get('null_delta_np_mean')} ({v.get('null_delta_np_p2.5_p97.5', [None, None])[0]} to {v.get('null_delta_np_p2.5_p97.5', [None, None])[1]}) | {v.get('null_delta_caught_mean')} | {v.get('null_excess_mean')} ({v.get('null_excess_p2.5_p97.5', [None, None])[0]} to {v.get('null_excess_p2.5_p97.5', [None, None])[1]}) | {v.get('real_delta_np')} | {v.get('real_excess')} | {v.get('share_null_at_or_below_real_excess')} |")
P("\n### 8. Batch and scanner\n"); T8 = B["test8_batch"]; P(T8["note"] + ".\n")
P("| Class | pairs | pairs with scanner change | with batch change | |Δ H&E logit|: scanner change coef [CI] | batch change coef | |Δ gate score|: scanner coef | flip: scanner coef | flip: batch coef | mean flip rate no change / change |"); P("|---|---|---|---|---|---|---|---|---|---|")
for cl in ("non_progressor", "progressor"):
    o = T8[cl]; c = lambda y, t: f"{o[y][t]['coef']:+.2f} [{o[y][t]['ci95'][0]:+.2f}, {o[y][t]['ci95'][1]:+.2f}]"
    P(f"| {cl} | {o['consecutive_pairs']} | {o['pairs_scanner_change']} | {o['pairs_batch_change']} | {c('d_img', 'scanner_change')} | {c('d_img', 'batch_change')} | {c('d_gate', 'scanner_change')} | {c('flip', 'scanner_change')} | {c('flip', 'batch_change')} | {o['flip_mean_by_scanner_change']['no_change']} / {o['flip_mean_by_scanner_change']['change']} |")
P("\nICC(1) of repeat-mean endoscopy scores: " + "; ".join(f"{cl} {sc}: {v['icc1_unadjusted']} unadjusted, {v['icc1_scanner_adjusted']} scanner-adjusted, {v['icc1_scanner_and_batch_adjusted']} scanner+batch-adjusted" for cl, o in T8["icc_on_repeat_mean_scores"].items() for sc, v in o.items()) + ".\n")
for nm, v in T8["memory_gain_by_scanner_history"].items():
    P(f"- **{nm}** ({v['non_progressors']} NP, {v['progressors']} P): A1 vs R49 Δ NP {ci(v['A1_vs_R49']) if 'A1_vs_R49' in v else '–'}; final vs R49 Δ NP {ci(v['final_vs_R49']) if 'final_vs_R49' in v else '–'}.")
print("\n".join(L))

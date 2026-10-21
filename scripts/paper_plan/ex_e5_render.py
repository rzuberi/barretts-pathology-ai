"""E5: render results/paper_final/explore_E5/*.json into markdown tables (printed; pasted under Results in docs/paper_explore_E5.md)."""
import json, sys, os
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../results/paper_final/explore_E5")
J = lambda n: json.load(open(f"{R}/{n}.json"))
f3 = lambda v: "–" if v is None else f"{v:+.3f}"
def d(x, nd=1, sign=True):
    v, (lo, hi) = x["delta"], x["ci95"]; fm = (lambda z: f"{z:+.{nd}f}") if sign else (lambda z: f"{z:.{nd}f}")
    return f"{fm(v)} [{fm(lo)}, {fm(hi)}]"
def rows(blocks):
    out = []
    for blk in blocks:
        for key, o in blk.items(): out.append((key, o))
    return out
def table(items, title):
    print(f"\n#### {title}\n")
    print("| Rule | Caught (of 32) | NP flagged (of 43) | Δ NP flagged vs R49 | Δ caught vs R49 | Δ NP flagged vs P1 90% | Δ caught vs P1 90% | Endo sens / spec | Δ spec vs P1 at matched sens | Seq / patient |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for key, o in items:
        ob = o["observed"]; r = o["delta_vs_R49"]; p = o["delta_vs_P1 90%"]
        print(f"| {key} | {ob['progressor_patients_caught']:.1f} | {ob['non_progressor_patients_flagged']:.1f} | {d(r['non_progressor_patients_flagged'])} | {d(r['progressor_patients_caught'])} | "
              f"{d(p['non_progressor_patients_flagged'])} | {d(p['progressor_patients_caught'])} | {ob['endoscopy_sensitivity']:.3f} / {ob['endoscopy_specificity']:.3f} | "
              f"{d(o['delta_spec_vs_P1_matched_to_rule_sensitivity'], 3)} | {ob['samples_sequenced_per_patient']:.2f} |")
A = J("e5_a")["A"]; B = J("e5_b"); C = J("e5_c"); U = J("e5_auc")["AUC"]; MM = J("e5_mem")["MEM"]
print("#### AUROC (progression; sample-level fold-stratified, endoscopy max, patient max)\n")
print("| Score | Sample AUROC [CI] | Δ vs L-IMG | Endoscopy max | Δ vs L-IMG | Patient max | Δ vs L-IMG |"); print("|---|---|---|---|---|---|---|")
for k in U:
    if k in ("endoscopy_max", "patient_max", "transfer_vs_recorded_grade_pre_event"): continue
    o = U[k]; e = U["endoscopy_max"][k]; p = U["patient_max"][k]
    dd = lambda z: d(z, 3) if z else "–"
    print(f"| {k} | {o['sample_fold_stratified_auroc']:.3f} [{o['ci95'][0]:.3f}, {o['ci95'][1]:.3f}] | {dd(o.get('delta_vs_L-IMG'))} | {e['auroc']:.3f} | {dd(e.get('delta_vs_L-IMG'))} | {p['auroc']:.3f} | {dd(p.get('delta_vs_L-IMG'))} |")
print("\n#### Transfer check on our slides (pre-event samples, recorded grade)\n")
print("| Score | LGD vs NDBE | ID/LGD vs NDBE | Spearman with L-IMG |"); print("|---|---|---|---|")
for k, o in U["transfer_vs_recorded_grade_pre_event"].items(): print(f"| {k} | {o['LGD_vs_NDBE']} | {o['ID_or_LGD_vs_NDBE']} | {o['spearman_with_repeat_mean_L-IMG']} |")
table(rows([A]), "(a) ERIN score alone (P1-like rule)")
table(rows([B["B"]]), "(b) Stacked with L-IMG"); print("\nStack coefficients:", json.dumps(B["stack_coefficients"]))
table(rows([C["C"]]), "(c) Veto / confirmation inside R49"); print("\nVeto touch:", json.dumps(C.get("veto_touch_primary_q95")))
table(rows([MM]), "(m) Round-9 EW memory rule with ERIN inside (nested)")
for k, o in MM.items(): print(k, o.get("chosen_counts"), "infeasible", o.get("infeasible_folds"))
# score rules matched to R49's sensitivity
print("\n#### Score rules: specificity at R49's realised endoscopy sensitivity, minus R49's\n"); print("| Rule | Δ spec [CI] |"); print("|---|---|")
for blk in (A, B["B"]):
    for k, o in blk.items():
        if "delta_spec_score_matched_to_R49_sensitivity_minus_R49" in o: print(f"| {k} | {d(o['delta_spec_score_matched_to_R49_sensitivity_minus_R49'], 3)} |")

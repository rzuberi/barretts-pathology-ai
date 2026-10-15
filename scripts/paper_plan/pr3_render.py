"""Renders docs/paper_plan_round3.md from results/paper_plan/round3_main.json and round3_lopo.json. Args: PRESPEC RESULTS_COMMIT PRESPEC_TEXT_FILE."""
import json, os, sys
R = "results/paper_plan"; PRESPEC, RESC, PRETXT = (sys.argv + ["cf244e1", "pending", ""])[1:4]; PRE = open(PRETXT).read() if PRETXT and os.path.exists(PRETXT) else ""
M = json.load(open(f"{R}/round3_main.json")); LO = json.load(open(f"{R}/round3_lopo.json")) if os.path.exists(f"{R}/round3_lopo.json") else None
def f3(x): return "n/a" if x is None else (f"{x:.3f}" if isinstance(x, (int, float)) and not isinstance(x, bool) else str(x))
def s3(x): return "n/a" if x is None else f"{x:+.3f}"
def ci(c): return "n/a" if not c or c[0] is None else (f"[{c[0]:+.3f}, {c[1]:+.3f}]" if (c[0] < 0 or c[1] < 0) else f"[{c[0]:.3f}, {c[1]:.3f}]")
def tb(hdr, rows): return "| " + " | ".join(hdr) + " |\n|" + "---|" * len(hdr) + "\n" + "\n".join("| " + " | ".join(str(v) for v in r) + " |" for r in rows)
SRC = lambda f, s: f"`results/paper_plan/{f}` · `scripts/paper_plan/{s}` · commit {RESC}"
NAME = {"C1_grade": "Clinical C1", "C2_grade_maxsofar": "Clinical C2 (primary)", "C3_plus_streak": "Clinical C3", "C4_plus_surveillance_3a": "Clinical C4 (=3a)", "cnv_only": "CNV (release RF)", "cnv_km": "CNV (Killcoyne method)", "image_only": "WSI", "early_fusion": "Early fusion", "intermediate_fusion": "Intermediate fusion", "late_mean": "Late fusion (image + RF CNV)", "late_mean_km": "Late fusion (image + KM CNV)", "coattention_fusion": "Co-attention (extra)", "late_stack_logit": "Late stack (extra)", "C2+image": "C2 + WSI", "C2+cnv_only": "C2 + CNV(RF)", "C2+cnv_km": "C2 + CNV(KM)", "C2+late_mean": "C2 + late fusion", "C2+image+cnv_only": "C2 + WSI + CNV(RF)"}
R1, R2, R3, R4, R5, R6 = M["R1"], M["R2"], M["R3"], M["R4"], M["R5"], M["R6"]; L = []; P = L.append
pr = R1["probes_stratum_from_inputs_nonprogressors"]; sa = R1["stratified_auroc"]["arms"]; sd = R1["stratified_paired_differences"]; iv = R1["stratum_adjusted_incremental_value"]
P(f"""# BE paper plan, round 3: confounding checks before writing (R1–R6)

## 1. Header
- Date: 26 September 2026. Commit at start: `e7ee492`. Pre-specification commit: `{PRESPEC}` (verbatim in §5). Results commit: `{RESC}`; this text is the next commit. Frozen release only.
- Scripts (`scripts/paper_plan/`): `pr3_main.py`, `pr3_lopo.py`, `pr3_render.py`, `pr3_check_report.py`. Results: `results/paper_plan/round3_main.json`, `round3_lopo.json`, figures `results/paper_plan/figs/km_v3/`; cluster-only rows `feasibility/paper_plan/round3_patient_table.csv`.
- Strata: discovery = 82 Killcoyne 777-sheet patients (40 progressors), validation = 68 268-sheet patients (10 progressors); stratified AUROC = pair-weighted within-stratum AUROC (weights {R1['stratified_auroc']['pairs_weights']}); stratified bootstrap resamples patients within stratum.

## 2. Status table
| item | status | key number |
|---|---|---|
| R1 Stratum as a shortcut | DONE | stratum from inputs (non-progressors): CNV features {f3(pr['cnv_features_pca20']['auroc'])} {ci(pr['cnv_features_pca20']['ci'])}, image embedding {f3(pr['image_patient_embedding_256d']['auroc'])} {ci(pr['image_patient_embedding_256d']['ci'])}, CNV QC {f3(pr['cnv_qc']['auroc'])}; stratified AUROC late fusion {f3(sa['late_mean']['stratified'])} {ci(sa['late_mean']['stratified_ci'])} vs pooled {f3(sa['late_mean']['pooled'])} |
| R2 Endpoint sensitivity | {'DONE' if LO else 'PARTIAL'} | E-HGD ({R2['E_HGD']['events']} events): late fusion pooled {f3(R2['E_HGD']['arms']['late_mean']['pooled'])}, CNV(RF) {f3(R2['E_HGD']['arms']['cnv_only']['pooled'])}, WSI {f3(R2['E_HGD']['arms']['image_only']['pooled'])}; LOPO retrain on sheet labels vs published: Spearman {f3(LO['spearman_lopo_vs_published_rows']) if LO else 'pending'} |
| R3 Tissue amount | DONE | kept_tiles alone AUROC {f3(R3['kept_tiles_as_score']['pooled']['auroc'])} {ci(R3['kept_tiles_as_score']['pooled']['ci'])} pooled, {f3(R3['kept_tiles_as_score']['discovery']['auroc'])} / {f3(R3['kept_tiles_as_score']['validation']['auroc'])} by stratum; stratum + tiles alone (fold-honest) {f3(R3['stratum_plus_tiles_base']['auroc_foldhonest'])} |
| R4 Risk groups (valid version) | DONE | validation stratum ({R4['validation_stratum_survival']['late_mean']['n']} / {R4['validation_stratum_survival']['late_mean']['events']}): late fusion log-rank p {R4['validation_stratum_survival']['late_mean'].get('logrank_p')}, Cox feasible {R4['validation_stratum_survival']['late_mean']['cox_feasible_rule_ge3_events_low_and_high']}; pooled MH OR high vs low {R4['pooled_rates_and_ORs']['late_mean']['MH_two_strata'].get('OR')} {ci(R4['pooled_rates_and_ORs']['late_mean']['MH_two_strata'].get('ci'))} |
| R5 FP: stratum and follow-up | DONE | late fusion later-HGD+ OR adjusted for stratum and follow-up {f3(R5['adjusted_with_stratum_and_followup']['late_mean']['later_HGDplus'].get('OR_FP_adjusted'))} {ci(R5['adjusted_with_stratum_and_followup']['late_mean']['later_HGDplus'].get('ci'))}; 1 patient reclassified |
| R6 Headline table | DONE | §R6 |

## 3. Items

### R1. Stratum as a shortcut
**Question.** Can the inputs predict the design stratum without the label, and how much of each arm survives within-stratum comparison?

**Status.** DONE. **Pre-specification.** `{PRESPEC}` §R1.

**Probes of stratum (discovery vs validation) among the 100 non-progressors** (fold-honest, release folds):

""" + tb(["input", "n non-progressors (discovery / validation)", "AUROC [CI]"], [[k, f"{v['n_nonprogressors']} ({v['discovery']} / {v['validation']})", f"{f3(v['auroc'])} {ci(v['ci'])}"] for k, v in pr.items()]) + f"""

**Stratified AUROC, all arms** (n {R1['stratified_auroc']['n_by_stratum']['discovery'][0]} / {R1['stratified_auroc']['n_by_stratum']['discovery'][1]} discovery, {R1['stratified_auroc']['n_by_stratum']['validation'][0]} / {R1['stratified_auroc']['n_by_stratum']['validation'][1]} validation; stratum alone in-sample AUROC {f3(R1['stratum_alone']['auroc_insample'])}):

""" + tb(["arm", "pooled [CI]", "stratified [CI]", "within discovery", "within validation"], [[NAME[a], f"{f3(v['pooled'])} {ci(v['pooled_ci'])}", f"{f3(v['stratified'])} {ci(v['stratified_ci'])}", f3(v["discovery"]), f3(v["validation"])] for a, v in sa.items()]) + "\n\n**Paired differences** (stratified vs pooled):\n\n" + tb(["difference", "stratified Δ [CI]", "pooled Δ [CI]"], [[k.replace("_minus_", " − "), f"{s3(v['stratified_delta'])} {ci(v['ci'])}", f"{s3(v['pooled_delta'])} {ci(v['pooled_ci'])}"] for k, v in sd.items()]) + "\n\n**Stratum-adjusted incremental value** (label ~ stratum, then + arm logit):\n\n" + tb(["arm", "LR stat (1 df)", "LR p", "AUROC stratum alone → + arm (in-sample)", "ΔAUROC fold-honest [CI]"], [[NAME[a], v["LR_stat"], v["LR_p"], f"{f3(v['auroc_base_insample'])} → {f3(v['auroc_base_plus_arm_insample'])}", f"{s3(v['delta_auroc_foldhonest'])} {ci(v['delta_ci_foldhonest'])}"] for a, v in iv.items()]) + f"""

**Method.** `pr3_main.py` R1. **Sources.** {SRC('round3_main.json', 'pr3_main.py')} (`R1`). **Caveats.** The image-embedding probe uses fold-k model embeddings for fold-k patients (held-out) and for training patients (in-sample for the image model); the CNV probe uses PCA-20 on 632 features; probes are among non-progressors only (100 patients).
""")
# R2
e1, e2 = R2["E_HGD"], R2["E_HGD_excl"]
P(f"""### R2. Endpoint sensitivity (HGD/IMC vs LGD2+)
**Question.** Do the arms predict HGD/IMC progression rather than second-LGD events, and is the F1 arm a faithful reimplementation?

**Status.** {'DONE' if LO else 'PARTIAL (LOPO retrain pending)'}. **Pre-specification.** `{PRESPEC}` §R2.

Progressor types: HGD/IMC {R2['progressor_types']['HGD_IMC']}, second-LGD only {R2['progressor_types']['second_LGD_only']}; by stratum {R2['progressor_types']['by_stratum']}. Second-LGD-only progressors' median percentile among all patients: {R2['second_LGD_only_progressors_scores_rank']}.

**E-HGD** (n {e1['n']}, events {e1['events']}; second-LGD progressors counted as non-events) and **E-HGD-excl** (n {e2['n']}, events {e2['events']}):

""" + tb(["arm", "E-HGD pooled [CI]", "E-HGD stratified [CI]", "E-HGD discovery / validation", "E-HGD-excl pooled [CI]", "E-HGD-excl stratified [CI]"], [[NAME[a], f"{f3(v['pooled'])} {ci(v['pooled_ci'])}", f"{f3(v['stratified'])} {ci(v['stratified_ci'])}", f"{f3(v['discovery'])} / {f3(v['validation'])}", f"{f3(e2['arms'][a]['pooled'])} {ci(e2['arms'][a]['pooled_ci'])}", f"{f3(e2['arms'][a]['stratified'])} {ci(e2['arms'][a]['stratified_ci'])}"] for a, v in e1["arms"].items()]) + (f"""

**Faithfulness retrain within the {LO['patients']} discovery patients on the sheet labels ({LO['sheet_P_patients']} sheet P), leave-one-patient-out** (`pr3_lopo.py`): LOPO vs sheet status row AUROC {f3(LO['lopo_vs_sheet_status']['row_auroc'])}, patient AUROC {f3(LO['lopo_vs_sheet_status']['patient_auroc_max'])}; published LOPO vs sheet status on the {LO['matched_rows_with_published']} matched rows: row {f3(LO['published_vs_sheet_status']['row_auroc'])}, patient {f3(LO['published_vs_sheet_status']['patient_auroc_max'])} (ours on the same rows: {f3(LO['lopo_vs_sheet_status_matched_rows']['row_auroc'])} / {f3(LO['lopo_vs_sheet_status_matched_rows']['patient_auroc_max'])}); Spearman ours vs published {f3(LO['spearman_lopo_vs_published_rows'])} (F1 arm vs published {f3(LO['spearman_f1arm_vs_published_rows'])}; ours vs F1 arm {f3(LO['spearman_lopo_vs_f1arm_rows'])}); ours vs our LGD2+ endpoint {f3(LO['lopo_vs_our_endpoint']['patient_auroc_max'])} ({LO['lopo_vs_our_endpoint']['events']} events), published vs our endpoint {f3(LO['published_vs_our_endpoint_matched']['patient_auroc_max'])}. C chosen: {LO['C_chosen_counts']}.""" if LO else "\n\nLOPO retrain: pending.") + f"""

**Method.** OOF scores re-evaluated against the alternative labels; no retraining except the LOPO faithfulness check. **Sources.** {SRC('round3_main.json', 'pr3_main.py')} (`R2`); {SRC('round3_lopo.json', 'pr3_lopo.py')}. **Caveats.** E-HGD has {e1['events']} events; the LOPO retrain uses sheet labels for 82 patients with in-cohort C selection and is not the glmnet pipeline.
""")
# R3
kt = R3["kept_tiles_as_score"]; kd = R3["kept_tiles_distribution"]
P(f"""### R3. Tissue amount as a shortcut
**Question.** Does the number of tissue tiles predict the label, and do the arms track it?

**Status.** DONE. **Pre-specification.** `{PRESPEC}` §R3.

`kept_tiles` alone (patient mean, sign from training folds): pooled AUROC {f3(kt['pooled']['auroc'])} {ci(kt['pooled']['ci'])} (raw direction more-tiles-higher {f3(kt['raw_pooled_auroc_more_tiles_higher'])}); stratified {f3(kt['stratified'])}; discovery {f3(kt['discovery']['auroc'])} (n {kt['discovery']['n']}, events {kt['discovery']['events']}), validation {f3(kt['validation']['auroc'])} (n {kt['validation']['n']}, events {kt['validation']['events']}). Tissue fraction alone: {f3(R3['tissue_frac_as_score']['pooled_auroc'])} {ci(R3['tissue_frac_as_score']['ci'])}.

Distribution: by stratum {kd['by_stratum']} (discovery vs validation Mann-Whitney p {kd['discovery_vs_validation_mannwhitney_p']}); by scanner {kd['by_scanner']}; by label {kd['by_label']}.

Spearman of arm score with `kept_tiles` among non-progressors: """ + "; ".join(f"{NAME[a]} {v['rho']} (p {v['p']})" for a, v in R3["spearman_arm_score_vs_kept_tiles_nonprogressors"].items()) + f""".

**Label ~ stratum + log tiles, then + arm** (base AUROC in-sample {f3(R3['stratum_plus_tiles_base']['auroc_insample'])}, fold-honest {f3(R3['stratum_plus_tiles_base']['auroc_foldhonest'])}):

""" + tb(["arm", "LR stat", "LR p", "ΔAUROC in-sample", "ΔAUROC fold-honest [CI]"], [[NAME[a], v["LR_stat"], v["LR_p"], s3(v["delta_auroc_insample"]), f"{s3(v['delta_auroc_foldhonest'])} {ci(v['delta_ci_foldhonest'])}"] for a, v in R3["stratum_plus_tiles_incremental_value"].items()]) + f"""

**late_mean false negatives** ({R3['late_mean_FN_summary']['n']}; by stratum {R3['late_mean_FN_summary']['by_stratum']}): median `kept_tiles` {R3['late_mean_FN_summary']['median_kept']}; median percentile within their stratum's progressors {R3['late_mean_FN_summary']['median_percentile_vs_stratum_progressors']} and non-progressors {R3['late_mean_FN_summary']['median_percentile_vs_stratum_nonprogressors']}. Per patient: {R3['late_mean_FN_tiles']}.

**Sources.** {SRC('round3_main.json', 'pr3_main.py')} (`R3`). **Caveats.** `kept_tiles` comes from the 0.44 µm/px extraction, not the release's 256-tile sample; scanner groups are unbalanced.
""")
# R4
vs = R4["validation_stratum_survival"]; po = R4["pooled_rates_and_ORs"]
P(f"""### R4. Risk groups: valid version only
**Question.** What do risk groups show when time-to-event is only used where it is interpretable?

**Status.** DONE. **Pre-specification.** `{PRESPEC}` §R4. {R4['version_A_superseded']}.

**Validation stratum, version B censoring** (n {vs['late_mean']['n']}, events {vs['late_mean']['events']}; training-fold tertiles):

""" + tb(["model", "groups: n / events / median time (d)", "log-rank p", "Cox feasible (≥3 events low and high)", "Cox HR high vs low", "Cox HR moderate vs low", "figure"], [[NAME[a], " / ".join(f"{gn}: {g['n']} / {g['events']} / {g['median_time_d']}" for gn, g in v["groups"].items()), v.get("logrank_p", v.get("logrank_error")), v["cox_feasible_rule_ge3_events_low_and_high"], v.get("cox_hr_high_vs_low", v.get("cox_error", "not fitted")), v.get("cox_hr_moderate_vs_low", ""), f"`{v['figure']}`"] for a, v in vs.items()]) + "\n\n**Pooled cohort: rates and odds ratios only.**\n\n" + tb(["model", "low: n / events / rate [Wilson]", "moderate", "high", "OR high vs low (Haldane)", "MH OR two strata [CI]", "MH OR three strata [CI]"], [[NAME[a], *(f"{v['groups'][gn]['n']} / {v['groups'][gn]['events']} / {f3(v['groups'][gn]['rate'])} {ci(v['groups'][gn]['wilson'])}" for gn in ["low", "moderate", "high"]), v["OR_high_vs_low_haldane"], f"{v['MH_two_strata'].get('OR')} {ci(v['MH_two_strata'].get('ci'))}", f"{v['MH_three_strata'].get('OR')} {ci(v['MH_three_strata'].get('ci'))}"] for a, v in po.items()]) + f"""

**Sources.** {SRC('round3_main.json', 'pr3_main.py')} (`R4`); figures `results/paper_plan/figs/km_v3/`. **Caveats.** With 10 events in the validation stratum the log-rank test and any Cox model are low-powered; the three-stratum MH OR conditions on the Killcoyne case/control status, which is itself an outcome-derived label.
""")
# R5
a5 = R5["adjusted_with_stratum_and_followup"]; b5 = R5["rerun_with_flagged_as_progressor"]
def f5row(a, v): return [NAME.get(a, a), f"{v['n_FP']} / {v['n_TN']}", str(v["FP_TN_by_stratum"]), (f"{f3(v['later_HGDplus'].get('OR_FP_adjusted'))} {ci(v['later_HGDplus'].get('ci'))}, p {v['later_HGDplus'].get('p')}; OR discovery {f3(v['later_HGDplus'].get('OR_discovery'))}, OR follow-up/yr {f3(v['later_HGDplus'].get('OR_followup_per_year'))}; raw {v['later_HGDplus'].get('raw_FP_TN')}" if "error" not in v["later_HGDplus"] else v["later_HGDplus"]["error"]), (f"{f3(v['later_LGDplus'].get('OR_FP_adjusted'))} {ci(v['later_LGDplus'].get('ci'))}, p {v['later_LGDplus'].get('p')}; raw {v['later_LGDplus'].get('raw_FP_TN')}" if "error" not in v["later_LGDplus"] else v["later_LGDplus"]["error"])]
P(f"""### R5. False positives: stratum and follow-up
**Question.** Does the FP excess of later disease survive adjustment for stratum and follow-up, and does the flagged mislabelled patient change it?

**Status.** DONE. **Pre-specification.** `{PRESPEC}` §R5.

**Adjusted for baseline grade, max grade, stratum (discovery) and follow-up (years after last release row):**

""" + tb(["model", "FP / TN", "FP, TN by stratum {discovery: [FP, TN], validation: [FP, TN]}", "later HGD+: OR FP [CI], p; covariate ORs; raw counts FP / TN", "later LGD+"], [f5row(a, v) for a, v in a5.items()]) + f"""

**Rerun with the {R5['flagged_patients_reclassified']} flagged patient moved to the progressors** (late_mean pooled AUROC {f3(b5['late_mean_pooled_auroc_relabelled']['original'])} → {f3(b5['late_mean_pooled_auroc_relabelled']['auroc'])}, events {b5['late_mean_pooled_auroc_relabelled']['events']}):

""" + tb(["model", "FP / TN", "by stratum", "later HGD+", "later LGD+"], [f5row(a, v) for a, v in b5.items() if a != "late_mean_pooled_auroc_relabelled"]) + f"""

**Sources.** {SRC('round3_main.json', 'pr3_main.json'.replace('.json', '.py'))} (`R5`). **Caveats.** 100 non-progressors and 15 later-HGD+ events with five covariates: the Wald CIs are wide and some fits may be near-separated.
""")
# R6
ne = R6["_n_events"]
P(f"""### R6. Headline numbers under each view
**Status.** DONE. n / events: pooled and stratified {ne['pooled']}; E-HGD {ne['E_HGD']}; Δ over stratum + tiles is fold-honest (R3).

""" + tb(["arm", "pooled AUROC [CI]", "stratified AUROC [CI]", "E-HGD pooled [CI]", "E-HGD stratified [CI]", "ΔAUROC over stratum + tiles [CI]"], [[NAME[a], f"{f3(v['pooled'][0])} {ci(v['pooled'][1:])}", f"{f3(v['stratified'][0])} {ci(v['stratified'][1:])}", f"{f3(v['E_HGD_pooled'][0])} {ci(v['E_HGD_pooled'][1:])}", f"{f3(v['E_HGD_stratified'][0])} {ci(v['E_HGD_stratified'][1:])}", f"{s3(v['delta_over_stratum_plus_tiles_foldhonest'][0])} {ci(v['delta_over_stratum_plus_tiles_foldhonest'][1:])}"] for a, v in R6.items() if not a.startswith("_")]) + f"""

**Sources.** {SRC('round3_main.json', 'pr3_main.py')} (`R6`, assembled from R1–R3).

## 4. Discrepancies found
1. Pooled vs stratified: late fusion {f3(sa['late_mean']['pooled'])} pooled vs {f3(sa['late_mean']['stratified'])} stratified; CNV (release RF) {f3(sa['cnv_only']['pooled'])} vs {f3(sa['cnv_only']['stratified'])}; WSI {f3(sa['image_only']['pooled'])} vs {f3(sa['image_only']['stratified'])} (R1). The within-discovery AUROCs are {f3(sa['late_mean']['discovery'])} (late), {f3(sa['cnv_only']['discovery'])} (CNV RF), {f3(sa['image_only']['discovery'])} (WSI).
2. Stratum is predictable from the inputs among non-progressors: CNV features {f3(pr['cnv_features_pca20']['auroc'])}, image embedding {f3(pr['image_patient_embedding_256d']['auroc'])}, CNV QC {f3(pr['cnv_qc']['auroc'])} (R1).
3. Tissue amount alone reaches {f3(kt['pooled']['auroc'])} pooled and stratum + tiles {f3(R3['stratum_plus_tiles_base']['auroc_foldhonest'])} fold-honest (R3); the follow-up's item-5 version-A survival results are superseded (R4).
4. Under E-HGD the CNV arms move to {f3(e1['arms']['cnv_only']['pooled'])} (RF) and {f3(e1['arms']['cnv_km']['pooled'])} (KM) pooled (R2).

## 5. Pre-specification text as committed at `{PRESPEC}` (verbatim)

""" + "\n".join("> " + l if l.strip() else ">" for l in PRE.splitlines()))
open("docs/paper_plan_round3.md", "w").write("\n".join(L)); print("written", sum(len(x) for x in L))

"""Renders docs/paper_plan_followup.md from results/paper_plan/f*.json and followup_main.json. Args: PRESPEC RESULTS_COMMIT PRESPEC_TEXT_FILE."""
import json, os, sys
R = "results/paper_plan"; PRESPEC, RESC, PRETXT = (sys.argv + ["0db4075", "pending", ""])[1:4]; PRE = open(PRETXT).read() if PRETXT and os.path.exists(PRETXT) else ""
J = lambda f: json.load(open(f"{R}/{f}")) if os.path.exists(f"{R}/{f}") else None
F1, M, F6, F7, F8a, F8b = J("f1_cnv_killcoyne.json"), J("followup_main.json"), J("f6_image_tiles.json"), J("f7_modality_ablation.json"), J("f8_train_vs_heldout.json"), J("f8_probe_vs_output.json")
def f3(x): return "n/a" if x is None or x == "None" else (f"{x:.3f}" if isinstance(x, (int, float)) and not isinstance(x, bool) else str(x))
def s3(x): return "n/a" if x is None else f"{x:+.3f}"
def ci(c): return "n/a" if not c or c[0] is None else (f"[{c[0]:+.3f}, {c[1]:+.3f}]" if (c[0] < 0 or c[1] < 0) else f"[{c[0]:.3f}, {c[1]:.3f}]")
def tb(hdr, rows): return "| " + " | ".join(hdr) + " |\n|" + "---|" * len(hdr) + "\n" + "\n".join("| " + " | ".join(str(v) for v in r) + " |" for r in rows)
SRC = lambda f, s: f"`results/paper_plan/{f}` · `scripts/paper_plan/{s}` · commit {RESC}"
NAME = {"C1_grade": "Clinical C1 (grade)", "C2_grade_maxsofar": "Clinical C2 (grade + max so far) [primary clinical]", "C3_plus_streak": "Clinical C3 (+ LGD streak)", "C4_plus_surveillance_3a": "Clinical C4 (+ surveillance = 3a)", "cnv_only": "CNV (release, PCA64 + RF)", "cnv_km": "CNV (Killcoyne method, elastic net)", "image_only": "WSI", "early_fusion": "Early fusion", "intermediate_fusion": "Intermediate fusion", "late_mean": "Late fusion (mean, image + CNV release)", "late_mean_km": "Late fusion (mean, image + CNV Killcoyne method)", "coattention_fusion": "Co-attention fusion (extra)", "late_stack_logit": "Late stack-logit (extra)"}
def arm_tb(t, arms=None): arms = arms or [a for a in NAME if a in t["arms"]]; return tb(["arm", "AUROC [CI]", "AUPRC [CI]", "n", "events"], [[NAME.get(a, a), f"{f3(t['arms'][a]['auroc'])} {ci(t['arms'][a]['auroc_ci'])}", f"{f3(t['arms'][a]['auprc'])} {ci(t['arms'][a]['auprc_ci'])}", t["n"], t["events"]] for a in arms])
L = []; P = L.append
T0 = M["F3"]["table_all_150"] if M else None; S1 = F1["subsets"]["all_150"] if F1 else None
P(f"""# BE paper plan follow-up (F1–F9)

## 1. Header
- Date: 25–26 September 2026. Commit at start: `38a83f2`. Pre-specification commit: `{PRESPEC}` (verbatim in §6). Script commits: `a8e5311`, `367d512`, `f99bde9`, `7d62569`. Results commit (result files, figures, scripts): `{RESC}`; this rendered text is the next commit. Frozen release only.
- Scripts (`scripts/paper_plan/`): `pf_cnv_killcoyne.py` (F1), `pf_main.py` (F2–F5, F6a, attention mass, F9), `pf_gpu.py` (F6b/c, F7, F8 train-vs-held-out), `pf_latent_figs.py` (F8), `pf_render.py`, `pf_check_report.py`.
- Results (`results/paper_plan/`): `f1_cnv_killcoyne.json`, `followup_main.json`, `f6_image_tiles.json`, `f7_modality_ablation.json`, `f8_train_vs_heldout.json`, `f8_probe_vs_output.json`; figures `figs/km_v2/`, `figs/latent_folds/`. Cluster-only rows: `feasibility/paper_plan/` (F1 OOF, strata, patient scores, later-HGD list, F6 draws, F6c checkpoints).
- Conventions as in `docs/paper_plan_answers.md`.
""")
if F1 and M and F6 and F7 and F8a and F8b:
    D = F1["diagnosis_cnv_only_overlap_discovery"]; FA = F1["faithfulness_vs_published_LOPO"]; F2 = M["F2"]; F3 = M["F3"]; F4 = M["F4"]; F5 = M["F5"]; F6c = M["F6_cpu"]; F9 = M["F9_noninferiority"]
    ka = F1["subsets"]["killcoyne_discovery_82"]["arms"]; sa = S1["arms"]
    P(f"""## 2. Status table
| item | status | key number |
|---|---|---|
| F1 Killcoyne-method CNV arm | DONE | all 150: {f3(sa['cnv_km']['auroc'])} {ci(sa['cnv_km']['auroc_ci'])} vs release CNV {f3(sa['cnv_only']['auroc'])}; on the 82 discovery patients {f3(ka['cnv_km']['auroc'])} vs {f3(ka['cnv_only']['auroc'])}; Spearman with published LOPO {f3(FA['spearman_cnv_km_vs_killcoyne'])} (rows {FA['matched_rows']}) |
| F2 Cohort design heterogeneity | DONE | stratum-only score AUROC {f3(F2['stratum_only_score']['auroc'])} {ci(F2['stratum_only_score']['ci'])}; strata × overlap in §F2 |
| F3 Clinical arm | DONE | C1 {f3(T0['arms']['C1_grade']['auroc'])}, C2 {f3(T0['arms']['C2_grade_maxsofar']['auroc'])} (primary), C3 {f3(T0['arms']['C3_plus_streak']['auroc'])}, C4 {f3(T0['arms']['C4_plus_surveillance_3a']['auroc'])}; C2 + late_mean (z-mean) Δ {s3(F3['C2_plus_modality']['C2+late_mean']['fold_z_mean']['delta_vs_C2'])} {ci(F3['C2_plus_modality']['C2+late_mean']['fold_z_mean']['ci'])} |
| F4 Survival inconsistency | DONE | version B (biopsy dates) late_mean HR high vs low {F4['groups']['late_mean']['version_B_biopsy_dates'].get('cox_hr_high_vs_low')}; version A {F4['groups']['late_mean']['version_A_release_MonthsBeforeLastBiopsy'].get('cox_hr_high_vs_low')} |
| F5 FP confounding | DONE | late_mean adjusted OR later HGD+ {F5['adjusted_logistic']['late_mean']['later_HGDplus'].get('OR_FP_adjusted')} {ci(F5['adjusted_logistic']['late_mean']['later_HGDplus'].get('ci'))}; {F5['later_HGDplus_nonprogressors']['n']} later-HGD+ non-progressors, {F5['later_HGDplus_nonprogressors']['flagged_hgd_is_next_release_biopsy']} flagged |
| F6 Attention and tiles | DONE | image-only top-5 % mass {F6c['attention_mass']['image_only']['top5pct_mass_median_iqr'][0]} (uniform 0.05); image-only 0.44 µm ≤2,048 tiles {f3(F6['F6c_image_only_044um_le2048']['auroc'])} {ci(F6['F6c_image_only_044um_le2048']['ci'])}; 256-tile draw SD median {F6['F6b_256_tile_resampling']['patient_score_sd_across_10_draws']['median']} |
| F7 Fusion uses CNV? | DONE | CNV jointly permuted ΔAUROC: early {s3(F7['early_fusion']['cnv_permuted_jointly']['delta_auroc_mean_over_repeats'])}, intermediate {s3(F7['intermediate_fusion']['cnv_permuted_jointly']['delta_auroc_mean_over_repeats'])}, co-attention {s3(F7['coattention_fusion']['cnv_permuted_jointly']['delta_auroc_mean_over_repeats'])}; image permuted: {s3(F7['early_fusion']['image_bags_permuted']['delta_auroc_mean_over_repeats'])} / {s3(F7['intermediate_fusion']['image_bags_permuted']['delta_auroc_mean_over_repeats'])} / {s3(F7['coattention_fusion']['image_bags_permuted']['delta_auroc_mean_over_repeats'])} |
| F8 Latent follow-ups | DONE | probe − model output: early {s3(F8b['probe_vs_model_output']['early_fusion']['delta_probe_minus_model'])} {ci(F8b['probe_vs_model_output']['early_fusion']['ci'])}, intermediate {s3(F8b['probe_vs_model_output']['intermediate_fusion']['delta_probe_minus_model'])} {ci(F8b['probe_vs_model_output']['intermediate_fusion']['ci'])}; five-fold figures `results/paper_plan/figs/latent_folds/` |
| F9 Housekeeping | DONE | WSI − CNV lower CI bound vs −0.05 margin: release CNV {F9['WSI_minus_cnv_only']['ci'][0]:+.3f}, Killcoyne-method CNV {F9['WSI_minus_cnv_km']['ci'][0]:+.3f} |

## 3. Items

### F1. Killcoyne-method CNV arm
**Question.** Does a CNV arm built with Killcoyne's method (elastic net on windows + arms) recover the CNV signal that the release's random forest loses on the overlap patients?

**Status.** DONE.

**Pre-specification.** `{PRESPEC}` §F1. As built: features {F1['features']} (the release's window-minus-arm encoding; the paper reports 589 windows and 44 arms, our release table carries {F1['features']['n_5mb_window_minus_arm']} windows and {F1['features']['n_arm']} arms plus cx); model {F1['model']}; C per fold {F1['C_per_fold']}; non-zero coefficients per fold {F1['nonzero_coefs_per_fold']} (paper: 74). α sensitivity (patient AUROC, all 150): {", ".join(f"α {a}: {v['oof_patient_auroc']}" for a, v in F1['alpha_sensitivity'].items())}.

**Result: subsets.**

""" + tb(["patients", "n", "events", "CNV (Killcoyne method)", "CNV (release RF)", "WSI", "late (image + RF CNV)", "late (image + KM CNV)"], [[k, v["n"], v["events"]] + [f"{f3(v['arms'][a]['auroc'])} {ci(v['arms'][a]['auroc_ci'])}" for a in ["cnv_km", "cnv_only", "image_only", "late_mean", "late_mean_km"]] for k, v in F1["subsets"].items()]) + f"""

**Faithfulness vs the published LOPO predictions** ({FA['matched_rows']} rows / {FA['matched_patients']} patients): Spearman Killcoyne-method arm vs published {f3(FA['spearman_cnv_km_vs_killcoyne'])}; release RF vs published {f3(FA['spearman_cnv_only_vs_killcoyne'])}; KM arm vs RF {f3(FA['spearman_cnv_km_vs_cnv_only'])}. Patient AUROC for our endpoint on the matched patients (n {FA['patient_auroc_matched']['n']}, events {FA['patient_auroc_matched']['events']}): published max prob {f3(FA['patient_auroc_matched']['killcoyne_max'])}, KM arm {f3(FA['patient_auroc_matched']['cnv_km'])}, RF {f3(FA['patient_auroc_matched']['cnv_only'])}.

**Item 2 rerun (WSI vs CNV) and item 4 rerun (late fusion with the KM arm), per subset:**

""" + tb(["patients", "WSI − CNV(KM) Δ [CI], perm p", "WSI − CNV(RF) Δ [CI], perm p", "late(KM) − WSI", "late(KM) − CNV(KM)", "late(KM) − late(RF)", "late(RF) − WSI"], [[k, f"{s3(v['item2_image_vs_cnv_km']['delta'])} {ci(v['item2_image_vs_cnv_km']['ci'])}, {v['item2_image_vs_cnv_km']['perm_p']}", f"{s3(v['item2_image_vs_cnv_only']['delta'])} {ci(v['item2_image_vs_cnv_only']['ci'])}, {v['item2_image_vs_cnv_only']['perm_p']}", f"{s3(v['item4_late_km_deltas']['late_mean_km_vs_image_only']['delta'])} {ci(v['item4_late_km_deltas']['late_mean_km_vs_image_only']['ci'])}, p {v['item4_late_km_deltas']['late_mean_km_vs_image_only']['perm_p']}", f"{s3(v['item4_late_km_deltas']['late_mean_km_vs_cnv_km']['delta'])} {ci(v['item4_late_km_deltas']['late_mean_km_vs_cnv_km']['ci'])}", f"{s3(v['item4_late_km_deltas']['late_mean_km_vs_late_mean']['delta'])} {ci(v['item4_late_km_deltas']['late_mean_km_vs_late_mean']['ci'])}", f"{s3(v['item4_late_mean_deltas']['late_mean_vs_image_only']['delta'])} {ci(v['item4_late_mean_deltas']['late_mean_vs_image_only']['ci'])}, p {v['item4_late_mean_deltas']['late_mean_vs_image_only']['perm_p']}"] for k, v in F1["subsets"].items()]) + f"""

**Diagnosis of the release RF on the {D['n_patients']} overlap-discovery patients ({D['n_rows']} rows).**

""" + tb(["fold", "held-out of the 25", "events", "RF AUROC", "KM-arm AUROC", "training rows (positive)", "training patients (progressors)", "rows from the 25 in training (positive)"], [[k, v["held_out_25_patients"], v["events"], f3(v["auroc_cnv_only"]), f3(v["auroc_cnv_km"]), f"{D['training_fold_class_balance'][k]['rows']} ({D['training_fold_class_balance'][k]['positive_rows']})", f"{D['training_fold_class_balance'][k]['patients']} ({D['training_fold_class_balance'][k]['progressor_patients']})", f"{D['training_fold_class_balance'][k]['rows_from_25_patients_in_training']} ({D['training_fold_class_balance'][k]['positive_rows_from_25_in_training']})"] for k, v in D["per_fold"].items()]) + f"""

Feature distributions, rows of the 25 vs the rest ({D['feature_distributions_rows']['rows_25']} vs {D['feature_distributions_rows']['rows_rest']} rows): mean |z| over arm + cx features {D['feature_distributions_rows']['mean_abs_z_arm_and_cx']}; PCA-64 score norm {D['feature_distributions_rows']['pca64_score_norm']}; raw cx {D['feature_distributions_rows']['cx_raw']}. RF probability medians: 25 progressor rows {D['rf_probability_rows']['25_progressor_rows']}, 25 non-progressor rows {D['rf_probability_rows']['25_nonprogressor_rows']}, rest progressor {D['rf_probability_rows']['rest_progressor_rows']}, rest non-progressor {D['rf_probability_rows']['rest_nonprogressor_rows']}; KM-arm medians on the 25: progressor rows {D['rf_probability_rows']['cnv_km_25_prog_median']}, non-progressor rows {D['rf_probability_rows']['cnv_km_25_nonprog_median']}.

**Method.** `pf_cnv_killcoyne.py`; release outer folds and inner folds; standardisation and imputation on training rows. **Sources.** {SRC('f1_cnv_killcoyne.json', 'pf_cnv_killcoyne.py')}; OOF rows `feasibility/paper_plan/f1_cnv_km_oof.csv` (cluster); Killcoyne Methods p.1733, Ext Data Fig 9. **Caveats.** Not a re-run of glmnet: sklearn saga with l1_ratio 0.9 and C by inner CV; the paper standardised across the whole cohort, here per training fold; our endpoint differs from theirs; the published LOPO probabilities were produced with their labels, so agreement is expected to be partial.
""")
    # F2
    tabs = {k: v for k, v in F2["tables"].items() if "arms" in v}
    P(f"""### F2. Cohort design heterogeneity
**Question.** Do results differ by cohort design stratum, and does stratum itself predict the label?

**Status.** DONE.

**Pre-specification.** `{PRESPEC}` §F2. {F2['stratum_definition']}.

Stratum × overlap (patients): {F2['crosstab_stratum_vs_overlap']}. Stratum × label (patients, columns 0/1): {F2['stratum_events']}. Strata skipped (fewer than 20 patients or 5 events/non-events): {[k for k, v in F2['tables'].items() if 'skipped' in v]}.

""" + "\n\n".join(f"**{k}** (n {v['n']}, events {v['events']})\n\n" + arm_tb(v) for k, v in tabs.items()) + f"""

**Matching check within the discovery stratum (our progressors vs non-progressors).** {F2['discovery_matching_check']}. Our LGD2+ label vs the sheet's case/control status (patients): {F2['discovery_matching_check']['our_label_vs_sheet_status']}.

**Stratum-only score** (training-fold progression rate per stratum applied to held-out patients): AUROC {f3(F2['stratum_only_score']['auroc'])} {ci(F2['stratum_only_score']['ci'])} (n {F2['stratum_only_score']['n']}, events {F2['stratum_only_score']['events']}).

**Arm scores by stratum among non-progressors** (discovery vs validation, patient max score):

""" + tb(["arm", "discovery median (n)", "validation median (n)", "Mann-Whitney p"], [[NAME.get(a, a), f"{v['discovery_median_n'][0]} ({v['discovery_median_n'][1]})", f"{v['validation_median_n'][0]} ({v['validation_median_n'][1]})", v["mannwhitney_p"]] for a, v in F2["arm_scores_by_stratum_nonprogressors_discovery_vs_validation"].items()]) + f"""

**Sources.** {SRC('followup_main.json', 'pf_main.py')} (`F2`); strata `feasibility/paper_plan/f2_strata.csv`. **Caveats.** Demographics cover 66/150 patients; the sheet case/control status is Killcoyne's HGD/IMC endpoint, not ours; strata are post hoc.
""")
    # F3
    P(f"""### F3. Clinical arm without endpoint-definition features
**Question.** How much of the clinical arm is the endpoint rule or surveillance history, and does any modality add to the primary clinical arm C2?

**Status.** DONE. Primary clinical arm = **C2 (grade + MaxPathologySoFar)**; 3b (27 patients / 9 events) is in the supplement of `docs/paper_plan_answers.md` only.

**All 150 patients (50 events)**

""" + arm_tb(T0) + f"""

C per fold: {F3['C_per_fold']}.

**Does a modality add to C2?**

""" + tb(["combination", "fold-z mean: AUROC, Δ vs C2 [CI], perm p", "L2 stack on logits: AUROC, Δ vs C2 [CI], perm p"], [[k, f"{f3(v['fold_z_mean']['auroc'])}, {s3(v['fold_z_mean']['delta_vs_C2'])} {ci(v['fold_z_mean']['ci'])}, {v['fold_z_mean']['perm_p']}", f"{f3(v['L2_stack_on_logits']['auroc'])}, {s3(v['L2_stack_on_logits']['delta_vs_C2'])} {ci(v['L2_stack_on_logits']['ci'])}, {v['L2_stack_on_logits']['perm_p']}"] for k, v in F3["C2_plus_modality"].items()]) + "\n\nSingle modality arms vs C2 (Δ = arm − C2): " + "; ".join(f"{NAME.get(m_, m_)} {s3(F3[m_ + '_vs_C2']['delta'])} {ci(F3[m_ + '_vs_C2']['ci'])}, p {F3[m_ + '_vs_C2']['perm_p']}" for m_ in ["image_only", "cnv_only", "cnv_km", "late_mean", "late_mean_km"]) + f""".

**Method.** Same nested logistic as 3a; combinations fitted on training rows (C by inner folds). **Sources.** {SRC('followup_main.json', 'pf_main.py')} (`F3`). **Caveats.** C2's `MaxPathologySoFar` includes the row's own grade; the L2 stack has two to three inputs and 570 training rows per fold.
""")
    # F4
    rows4 = []
    for a, v in F4["groups"].items():
        for ver in ["version_A_release_MonthsBeforeLastBiopsy", "version_B_biopsy_dates"]:
            w = v[ver]; rows4.append([NAME.get(a, a), ver.split("_")[1], " / ".join(f"{gn}: n {x['n']}, ev {x['events']}, rate {f3(x['rate'])}, t_prog {x['median_time_progressors_d']}, t_nonprog {x['median_time_nonprogressors_d']}" for gn, x in w["groups"].items()), w.get("logrank_p", w.get("cox_error")), w.get("cox_hr_high_vs_low"), w.get("cox_hr_moderate_vs_low"), v["odds_ratio_high_vs_low_haldane"], f"`{w['figure']}`"])
    P(f"""### F4. Risk-group survival inconsistency
**Question.** Why do rates/ORs and hazard ratios disagree, and how is censoring time derived?

**Status.** DONE.

**Censoring derivation.** {F4['censoring_derivation']}. Non-progressor time (days): version A median {F4['nonprogressor_time_days']['A_median']} IQR {F4['nonprogressor_time_days']['A_iqr']}; version B median {F4['nonprogressor_time_days']['B_median']} IQR {F4['nonprogressor_time_days']['B_iqr']}. Progressor time median {F4['progressor_time_days']['median']} IQR {F4['progressor_time_days']['iqr']}.

""" + tb(["model (tertiles)", "time version", "groups: n, events, rate, median time progressors / non-progressors (d)", "log-rank p", "Cox HR high vs low", "Cox HR moderate vs low", "OR high vs low (Haldane)", "figure"], rows4) + f"""

Discovery-stratum rates only (case–control design; time not interpretable): {json.dumps({a: v['discovery_stratum_rates_only'] for a, v in F4['groups'].items()})}. {F4['time_to_event_interpretability']}.

Probability deciles (patient max score, 0–100 % in steps of 10): {F4['probability_deciles']}. Killcoyne fixed classes applied to the Killcoyne-method arm: {F4['killcoyne_fixed_classes_on_cnv_km']}.

**Sources.** {SRC('followup_main.json', 'pf_main.py')} (`F4`); figures `results/paper_plan/figs/km_v2/`. **Caveats.** Non-progressor follow-up ends at the last biopsy in the release tables; progressor time is to the endpoint biopsy; both versions share the progressor times.
""")
    # F5
    P(f"""### F5. False positives: confounding and label check
**Question.** Is the FP excess of later disease explained by baseline grade, and are any "non-progressors" with later HGD+ mislabelled?

**Status.** DONE.

**Adjusted logistic regression among non-progressors** (later outcome ~ FP + baseline grade + max grade so far; statsmodels Logit, Wald CIs).

""" + tb(["model", "FP / TN", "later HGD+: OR FP adjusted [CI], p (unadjusted OR)", "later LGD+: OR FP adjusted [CI], p (unadjusted OR)"], [[NAME.get(a, a), f"{v['n_FP']} / {v['n_TN']}", (f"{f3(v['later_HGDplus'].get('OR_FP_adjusted'))} {ci(v['later_HGDplus'].get('ci'))}, p {v['later_HGDplus'].get('p')} ({f3(v['later_HGDplus'].get('OR_FP_unadjusted'))})" if "error" not in v["later_HGDplus"] else v["later_HGDplus"]["error"]), (f"{f3(v['later_LGDplus'].get('OR_FP_adjusted'))} {ci(v['later_LGDplus'].get('ci'))}, p {v['later_LGDplus'].get('p')} ({f3(v['later_LGDplus'].get('OR_FP_unadjusted'))})" if "error" not in v["later_LGDplus"] else v["later_LGDplus"]["error"])] for a, v in F5["adjusted_logistic"].items()]) + "\n\n**Within baseline-NDBE non-progressors.**\n\n" + tb(["model", "FP", "TN", "later HGD+ FP / TN", "Fisher p", "later LGD+ FP / TN", "Fisher p"], [[NAME.get(a, a), v["n_FP"], v["n_TN"], v["later_HGDplus_FP_TN"], v["fisher_p_HGD"], v["later_LGDplus_FP_TN"], v["fisher_p_LGD"]] for a, v in F5["within_baseline_NDBE"].items()]) + f"""

**Non-progressors with later HGD+** (n {F5['later_HGDplus_nonprogressors']['n']}): days after the last release row median [IQR] {F5['later_HGDplus_nonprogressors']['days_after_last_release_row_median_iqr']}; sources {F5['later_HGDplus_nonprogressors']['sources']}; with release-excluded rows {F5['later_HGDplus_nonprogressors']['with_release_excluded_rows']} ({F5['later_HGDplus_nonprogressors']['excluded_reason_counts']}); **flagged as the next release biopsy (would be progressors): {F5['later_HGDplus_nonprogressors']['flagged_hgd_is_next_release_biopsy']}**. {F5['later_HGDplus_nonprogressors']['why_not_progressors']}. Patient list `feasibility/paper_plan/f5_later_hgd_nonprogressors.csv` (cluster).

**LGD+/HGD+ count check.** late_mean: {F5['grades_behind_counts_late_mean']}; image_only: {F5['grades_behind_counts_image_only']}.

**Sources.** {SRC('followup_main.json', 'pf_main.py')} (`F5`). **Caveats.** Later grades are DB confirmed codes; 100 non-progressors with ~15 later HGD+ events give wide CIs; adjustment covariates are coarse ordinal grades.
""")
    # F6
    P(f"""### F6. Attention and tile sampling
**Question.** How concentrated is attention, does tile count drive image-only performance, and how much do 256-tile draws move patient scores?

**Status.** DONE (F6c retrained with a scale change, labelled).

**Attention mass** (held-out rows):

""" + tb(["model", "rows", "top-5 % (13 tiles) mass median [IQR]", "top-10 % mass median [IQR]", "uniform", "max weight median (uniform 1/256 = 0.004)", "rows with top-5 % mass < 0.06"], [[fam, v["rows"], f"{v['top5pct_mass_median_iqr'][0]} [{v['top5pct_mass_median_iqr'][1]}, {v['top5pct_mass_median_iqr'][2]}]", f"{v['top10pct_mass_median_iqr'][0]} [{v['top10pct_mass_median_iqr'][1]}, {v['top10pct_mass_median_iqr'][2]}]", "0.05 / 0.10", v["max_weight_median"], v["frac_rows_top5_mass_below_0.06"]] for fam, v in F6c["attention_mass"].items()]) + f"""

**F6a: image-only by tiles-kept tertile** (patient mean of `kept_tiles` at 0.44 µm/px):

""" + tb(["tertile", "tiles kept range", "n", "events", "image-only AUROC [CI]", "late_mean", "cnv_only"], [[t_, v["tiles_kept_range"], v["n"], v["events"], f"{f3(v['image_only_auroc'])} {ci(v['ci'])}", f3(v["late_mean_auroc"]), f3(v["cnv_only_auroc"])] for t_, v in F6c["image_only_by_tiles_kept_tertile"].items()]) + f"""

**F6c: image-only retrained on 0.44 µm/px UNI2-h bags, ≤2,048 tiles** (release ABMIL config, release folds, epochs per fold {F6['F6c_image_only_044um_le2048']['epochs_per_fold']}, seed 0): AUROC {f3(F6['F6c_image_only_044um_le2048']['auroc'])} {ci(F6['F6c_image_only_044um_le2048']['ci'])} (n {F6['F6c_image_only_044um_le2048']['n']}, events {F6['F6c_image_only_044um_le2048']['events']}); Δ vs release image-only {s3(F6['F6c_image_only_044um_le2048']['delta_vs_release_image_only'][0])} {ci(F6['F6c_image_only_044um_le2048']['delta_vs_release_image_only'][1:])}. {F6['F6c_image_only_044um_le2048']['note']}.

**F6b: 10 random 256-tile draws through the F6c fold models.** Patient-score SD across draws median {F6['F6b_256_tile_resampling']['patient_score_sd_across_10_draws']['median']} IQR {F6['F6b_256_tile_resampling']['patient_score_sd_across_10_draws']['iqr']} max {F6['F6b_256_tile_resampling']['patient_score_sd_across_10_draws']['max']}; AUROC per draw {F6['F6b_256_tile_resampling']['auroc_per_draw']} (full bags {F6['F6b_256_tile_resampling']['auroc_full_bag_F6c']}); patients changing class across draws {F6['F6b_256_tile_resampling']['patients_changing_class_across_draws']} of 150 (threshold per fold {F6['F6b_256_tile_resampling']['thresholds_per_fold']}); the {F6['F6b_256_tile_resampling']['late_mean_FN_patients']} late_mean FN patients were predicted positive in {F6['F6b_256_tile_resampling']['FN_predicted_positive_draws_of_10']} of 10 draws (score SD {F6['F6b_256_tile_resampling']['FN_score_sd']}).

**Sources.** {SRC('followup_main.json', 'pf_main.py')} (`F6_cpu`); {SRC('f6_image_tiles.json', 'pf_gpu.py')}; checkpoints `feasibility/paper_plan/f6c_image05_fold*.pt`. **Caveats.** F6c changes tile scale and tile count at once; single training seed; F6b draws are of 0.44 µm tiles through the F6c model, not of the release model.
""")
    # F7
    P(f"""### F7. Does fusion actually use CNV?
**Question.** How much does each learned fusion model lose when its CNV input, or its image input, is destroyed?

**Status.** DONE.

""" + tb(["model", "baseline AUROC", "CNV permuted jointly: Δ mean (SD over 50), [CI of mean-permuted]", "CNV → training mean: Δ [CI]", "image bags permuted: Δ mean (SD), [CI]", "image → mean tile: Δ [CI]"], [[fam, f3(v["baseline_auroc"]), (f"{s3(v['cnv_permuted_jointly']['delta_auroc_mean_over_repeats'])} ({v['cnv_permuted_jointly']['delta_sd_over_repeats']}) {ci(v['cnv_permuted_jointly']['delta_of_mean_permuted_score_ci'])}" if "cnv_permuted_jointly" in v else "—"), (f"{s3(v['cnv_replaced_by_training_mean']['delta_auroc'])} {ci(v['cnv_replaced_by_training_mean']['ci'])}" if "cnv_replaced_by_training_mean" in v else "—"), f"{s3(v['image_bags_permuted']['delta_auroc_mean_over_repeats'])} ({v['image_bags_permuted']['delta_sd_over_repeats']}) {ci(v['image_bags_permuted']['delta_of_mean_permuted_score_ci'])}", f"{s3(v['image_replaced_by_training_mean_tile']['delta_auroc'])} {ci(v['image_replaced_by_training_mean_tile']['ci'])}"] for fam, v in F7.items() if not fam.startswith("_")]) + f"""

**8b noise floor** (one random 5-Mb window feature permuted, 10 windows × 50 repeats × 5 folds): """ + "; ".join(f"{fam}: mean {v['random_window_delta_auroc']['mean']}, SD {v['random_window_delta_auroc']['sd']}, 95th pct {v['random_window_delta_auroc']['q95']}, 99th {v['random_window_delta_auroc']['q99']}" for fam, v in F7["_8b_noise_floor"].items()) + f""". Per-fold arm importances (mean of 10 repeats per fold) are in the JSON (`_8b_noise_floor.<model>.arm_features_per_fold_delta_auroc_mean_of_10_repeats`).

**Method.** {F7['_method']}. **Sources.** {SRC('f7_modality_ablation.json', 'pf_gpu.py')}. **Caveats.** Two different quantities are shown for the permutation ablations: the mean over 50 repeats of the per-repeat ΔAUROC, and the ΔAUROC of the score averaged over the 50 permutations (whose CI is given); averaging permuted scores removes part of the damage, so the second is smaller. Replacing CNV by the training mean is a single deterministic ablation; the image ablation for early fusion replaces the mean-pooled bag.
""")
    # F8
    P(f"""### F8. Latent space: small follow-ups
**Question.** Are the probes really above the models' own outputs, and do the projections hold across folds?

**Status.** DONE.

""" + tb(["representation", "probe AUROC", "model OOF AUROC", "Δ probe − model [CI]", "n / events"], [[fam, f3(v["probe_auroc"]), f3(v["model_oof_auroc"]), f"{s3(v['delta_probe_minus_model'])} {ci(v['ci'])}", f"{v['n']} / {v['events']}"] for fam, v in F8b["probe_vs_model_output"].items()]) + "\n\n**Training-fold vs held-out AUROC per fold (model outputs, patient level).**\n\n" + tb(["family", "fold", "training AUROC (n patients)", "held-out AUROC (n patients)"], [[fam, k, f"{v['train_auroc']} ({v['n_train_patients']})", f"{v['heldout_auroc']} ({v['n_heldout_patients']})"] for fam, d in F8a["train_vs_heldout_auroc_patient_level"].items() for k, v in d.items()]) + f"""

Five-fold PCA/UMAP figures (projection fitted on each fold's training patients; held-out patients shown; rings = also_in_ERIN): {", ".join("`" + v + "`" for v in F8b['figures'].values())}.

**Method.** {F8b['_method']}. **Sources.** {SRC('f8_probe_vs_output.json', 'pf_latent_figs.py')}; {SRC('f8_train_vs_heldout.json', 'pf_gpu.py')}. **Caveats.** The probe is a second model fitted on top of the fusion embedding with its own inner CV; the comparison is probe-on-embedding vs the network's own head.
""")
    # F9
    P(f"""### F9. Housekeeping
**Status.** DONE.

**Page mapping.** `pdftotext` page N of `s41591-020-1033-y.pdf` = journal page 1725 + N for N ≤ 9 (Letters 1726–1732, Methods 1733–1734); N = 10–24 are Extended Data Figs 1–10 (online only, no journal page). Revised references for the item-1 table in `docs/paper_plan_answers.md`: abstract and Fig 1 legend → p.1726–1727; risk classes and validation paragraph → p.1727; Methods (cohorts, sequencing 0.4× HiSeq, QDNAseq 50-kb, elastic net, LOPO, endpoint) → p.1733; Ext Data Fig 2 (aggregation) and Fig 9 (bin size / penalty, AUC 0.87 / 0.84) → online Extended Data; supplementary `MOESM15` Ext Data Fig 9c → 50 kb discovery AUC 0.865 [0.839, 0.891].

**Non-inferiority (pre-specified margin −0.05 AUROC).** WSI − CNV (release RF): {s3(F9['WSI_minus_cnv_only']['delta'])} {ci(F9['WSI_minus_cnv_only']['ci'])}, lower bound above margin: {F9['WSI_minus_cnv_only']['lower_bound_above_margin']}. WSI − CNV (Killcoyne method): {s3(F9['WSI_minus_cnv_km']['delta'])} {ci(F9['WSI_minus_cnv_km']['ci'])}, lower bound above margin: {F9['WSI_minus_cnv_km']['lower_bound_above_margin']}. (n 150, 50 events.)

**Sources.** {SRC('followup_main.json', 'pf_main.py')} (`F9_noninferiority`); `f1_cnv_killcoyne.json` (`subsets.*.item2_*`). **Caveats.** The margin was chosen for this follow-up, not in the original paper plan.

## 4. Updated filled whiteboard table (SWG)

""" + tb(["", "SWG progressor cohort (n 150, 50 progressors): AUROC [CI] · AUPRC [CI]", "ACE-B"], [[NAME[a], f"{f3(T0['arms'][a]['auroc'])} {ci(T0['arms'][a]['auroc_ci'])} · {f3(T0['arms'][a]['auprc'])} {ci(T0['arms'][a]['auprc_ci'])}", "NOT AVAILABLE"] for a in ["C2_grade_maxsofar", "cnv_only", "cnv_km", "image_only", "early_fusion", "intermediate_fusion", "late_mean", "late_mean_km", "coattention_fusion", "late_stack_logit"]]) + f"""

Source: `results/paper_plan/followup_main.json` (`F3.table_all_150`), commit {RESC}.

## 5. Discrepancies found
1. The Killcoyne-method arm does not recover the CNV signal: on the 82 discovery patients it scores {f3(ka['cnv_km']['auroc'])} against the release RF {f3(ka['cnv_only']['auroc'])} and the published LOPO predictions {f3(FA['patient_auroc_matched']['killcoyne_max'])}, and its agreement with the published probabilities ({f3(FA['spearman_cnv_km_vs_killcoyne'])}) is no higher than the RF's ({f3(FA['spearman_cnv_only_vs_killcoyne'])}). Both of our CNV models are trained on our LGD2+ labels; the published predictions were trained on their HGD/IMC case–control labels, which disagree with ours for 11 of the 82 patients (paper plan item 0). The difference is therefore in labels and cohort design, not in the model class.
1b. Design stratum alone predicts our label with AUROC {f3(F2['stratum_only_score']['auroc'])} {ci(F2['stratum_only_score']['ci'])}: 34 of 39 Killcoyne discovery cases and 6 of 43 discovery controls are our progressors, while the validation-sheet stratum has 10 events in 68 patients (F2). Within the discovery-control stratum the CNV arms score below 0.5.
2. Killcoyne fixed classes place no release-RF patient in the high class because the RF probabilities never reach 0.5 (deciles in F4); the classes were designed for a logistic-regression probability scale.
3. The paper-plan item-5 censoring time (version A) does not equal the biopsy-date interval (version B) for most non-progressors (F4 censoring derivation).
4. Clinical arm 3a's advantage over C2 comes from features that encode the endpoint rule or surveillance history (F3, C1→C4).
5. Later-HGD+ non-progressors flagged as the next release biopsy: {F5['later_HGDplus_nonprogressors']['flagged_hgd_is_next_release_biopsy']} (F5); if > 0 these are label discrepancies in the release.

## 6. Not done
- glmnet itself (R) was not run; the elastic net is the sklearn saga implementation.
- F6c uses 0.44 µm/px features (the only multi-tile pool available), so tile count and scale change together; a 0.88 µm/px multi-tile re-extraction was not done. F6c scored below the release image arm, so the tile-count hypothesis is not supported by this test but is not isolated by it either.
- ACE-B: no data.

## 7. Pre-specification text as committed at `{PRESPEC}` (verbatim)

""" + "\n".join("> " + l if l.strip() else ">" for l in PRE.splitlines()))
else:
    P("## Results pending: missing " + ", ".join(n for n, v in [("F1", F1), ("main", M), ("F6", F6), ("F7", F7), ("F8a", F8a), ("F8b", F8b)] if v is None))
open("docs/paper_plan_followup.md", "w").write("\n".join(L)); print("written", sum(len(x) for x in L))

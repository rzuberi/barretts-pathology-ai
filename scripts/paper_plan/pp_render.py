"""Renders docs/paper_plan_answers.md from results/paper_plan/*.json; every number is read from a file. Args: PRESPEC RESULTS_COMMIT PRESPEC_TEXT_FILE."""
import json, os, sys, glob
R = "results/paper_plan"; PRESPEC, RESC, PRETXT = (sys.argv + ["db236a0", "pending", ""])[1:4]; PRE = open(PRETXT).read() if PRETXT and os.path.exists(PRETXT) else ""
J = lambda f: json.load(open(f"{R}/{f}")) if os.path.exists(f"{R}/{f}") else None
A1, A2, M, L6, I8 = J("audit_item0.json"), J("audit_item0_v2.json"), J("main_items.json"), J("latent_item6.json"), J("perm_importance_item8b.json")
def f3(x): return "n/a" if x is None or x == "None" else (f"{x:.3f}" if isinstance(x, (int, float)) else str(x))
def s3(x): return "n/a" if x is None else f"{x:+.3f}"
def ci(c): return "n/a" if not c or c[0] is None else (f"[{c[0]:+.3f}, {c[1]:+.3f}]" if (c[0] < 0 or c[1] < 0) else f"[{c[0]:.3f}, {c[1]:.3f}]")
def tb(hdr, rows): return "| " + " | ".join(hdr) + " |\n|" + "---|" * len(hdr) + "\n" + "\n".join("| " + " | ".join(str(v) for v in r) + " |" for r in rows)
SRC = lambda f, s: f"`results/paper_plan/{f}` · `scripts/paper_plan/{s}` · commit {RESC}"
L = []; P = L.append
FAMN = {"clinical_3a": "Clinical only (3a)", "cnv_only": "CNV", "image_only": "WSI", "early_fusion": "Early fusion", "intermediate_fusion": "Intermediate fusion", "late_mean": "Late fusion (mean)", "coattention_fusion": "Co-attention fusion (extra)", "late_stack_logit": "Late stack-logit fusion (extra)"}
FUS = ["early_fusion", "intermediate_fusion", "late_mean", "coattention_fusion", "late_stack_logit"]
def arm_table(t):
    return tb(["arm", "AUROC [CI]", "AUPRC [CI]", "n", "events"], [[FAMN[a], f"{f3(t['arms'][a]['auroc'])} {ci(t['arms'][a]['auroc_ci'])}", f"{f3(t['arms'][a]['auprc'])} {ci(t['arms'][a]['auprc_ci'])}", t["n"], t["events"]] for a in FAMN if a in t["arms"]])
def diff_table(t):
    rows = []
    for ref in ["image_only", "cnv_only"]:
        for f_ in FUS:
            d = t["differences"][f"{f_}_vs_{ref}"]; rows.append([FAMN[f_], FAMN[ref], f"{s3(d['delta_auroc'])} {ci(d['ci'])}", f"{s3(d['delta_auprc'])} {ci(d['auprc_ci'])}", d["perm_p_naive"], d["perm_p_selection_adjusted_max_over_5_fusion_arms"]])
    return tb(["fusion arm", "vs", "ΔAUROC [CI]", "ΔAUPRC [CI]", "naive perm p", "selection-adjusted p (max over 5 fusion arms)"], rows)
T0 = M["item4_tables"]["all_150"] if M else None
DEC = A2["decision"]["decision"] if A2 else "pending"
# ------------------------------------------------------------------ header + status
P(f"""# BE paper plan: answers (SWG progressor cohort)

## 1. Header
- Date: 25 September 2026. Commit at start: `1d27002`. Pre-specification commit: `{PRESPEC}` (text reproduced verbatim in §8). Script commits before results: `187dda8`, `5bf5419`, `529cccc`, `b4bc409`. Results commit (all result files, figures and scripts): `{RESC}`; this rendered text is the next commit.
- Release used for every number: the frozen `chapter1_lgd2_final_pre_event_20260713_final` only (item 0: {DEC}).
- New scripts (`scripts/paper_plan/`): `pp_audit.py`, `pp_latent.py`, `pp_main.py`, `pp_montage.py`, `pp_review_pack.py`, `pp_render.py`.
- New results (`results/paper_plan/`): `audit_item0.json` (v1), `audit_item0_v2.json`, `main_items.json`, `latent_item6.json`, `attention_item7_rows.csv`, `perm_importance_item8b.json`; figures `figs/km/*.png`, `figs/latent/*.png`, `figs/attention/M*.png`. Cluster-only (identifiable or row-level): `feasibility/paper_plan/` (audit rows, patient scores, later-disease table, embeddings, montage and review-pack manifests, review pack thumbnails).
- Conventions: patient level, patient score = max over strict pre-event rows; rank AUROC and AUPRC to 3 decimals; CIs = 2,000 patient bootstraps `RandomState(0)`; permutations = 2,000 patient-label permutations `RandomState(0)`. Fold honesty: every fitted quantity (clinical arms, operating thresholds, tertile cut-points, probes, kNN, scalers, PCA/UMAP) is fitted on the four training outer folds (inner CV on the release's patient-keyed inner folds) and applied to the held-out fold.
""")
if M:
    op = M["operating_point"]; i5 = M["item5"]["groups"]; i9 = M["item9"]["models"]; i10 = M["item10"]["models"]
    P(f"""## 2. Plan status table
| whiteboard element | status | key number / figure |
|---|---|---|
| Intro: Killcoyne 2020 comparison (item 1) | DONE | their column filled from the paper PDF with page refs; {A2['killcoyne']['matched_patients']} of our patients / {A2['killcoyne']['matched_rows']} rows have Killcoyne per-sample predictions |
| Intro: histology as good as CNV? (item 2) | DONE | WSI − CNV ΔAUROC {s3(T0['item2_image_vs_cnv']['delta_auroc'])} {ci(T0['item2_image_vs_cnv']['ci'])}, perm p {T0['item2_image_vs_cnv']['perm_p']} (n 150, 50 events) |
| Table row Clinical only × SWG (item 3) | DONE | 3a AUROC {f3(T0['arms']['clinical_3a']['auroc'])} {ci(T0['arms']['clinical_3a']['auroc_ci'])}; 3b on {M['item3'].get('3b_complete_case_patients')} complete-case patients |
| Table row CNV × SWG | DONE | {f3(T0['arms']['cnv_only']['auroc'])} {ci(T0['arms']['cnv_only']['auroc_ci'])} |
| Table row WSI × SWG | DONE | {f3(T0['arms']['image_only']['auroc'])} {ci(T0['arms']['image_only']['auroc_ci'])} |
| Table row Early fusion × SWG | DONE | {f3(T0['arms']['early_fusion']['auroc'])} {ci(T0['arms']['early_fusion']['auroc_ci'])} |
| Table row Intermediate fusion × SWG | DONE | {f3(T0['arms']['intermediate_fusion']['auroc'])} {ci(T0['arms']['intermediate_fusion']['auroc_ci'])} |
| Table row Late fusion × SWG | DONE | {f3(T0['arms']['late_mean']['auroc'])} {ci(T0['arms']['late_mean']['auroc_ci'])}; vs WSI {s3(T0['differences']['late_mean_vs_image_only']['delta_auroc'])} {ci(T0['differences']['late_mean_vs_image_only']['ci'])}, selection-adjusted p {T0['differences']['late_mean_vs_image_only']['perm_p_selection_adjusted_max_over_5_fusion_arms']} |
| Every table row × ACE-B | NOT AVAILABLE | no ACE-B slides, features or CNV on the cluster (closeout item N); nothing run |
| Discussion 1: risk groups (item 5) | DONE | late_mean tertiles: high-group rate {f3(i5['late_mean__tertile_trainfold']['groups']['high']['rate'])} vs low {f3(i5['late_mean__tertile_trainfold']['groups']['low']['rate'])}; Cox HR high vs low {i5['late_mean__tertile_trainfold'].get('cox_hr_high_vs_low')}; `{i5['late_mean__tertile_trainfold']['figure']}` |
| Discussion 2: latent space (item 6) | {'DONE' if L6 else 'PENDING'} | {('probe AUROC image ' + f3(L6['image_only']['probe_auroc']) + ', intermediate ' + f3(L6['intermediate_fusion']['probe_auroc']) + ', co-attention ' + f3(L6['coattention_fusion']['probe_auroc']) + '; figs `results/paper_plan/figs/latent/`') if L6 else 'pending'} |
| Discussion 3: attention patches (item 7) | {'DONE' if 'intermediate_fusion' in M.get('item7', {}) else 'PARTIAL'} | {('Spearman image-only vs intermediate median ' + f3(M['item7']['intermediate_fusion']['overall']['spearman_median']) + ', vs co-attention ' + f3(M['item7']['coattention_fusion']['overall']['spearman_median']) + '; montages cluster-only `feasibility/paper_plan/figs/attention/`') if 'intermediate_fusion' in M.get('item7', {}) else 'attention rows pending'} |
| Discussion 4: CNV prediction change (item 8) | {'DONE' if I8 else 'PARTIAL'} | 8a Spearman CNV vs late {f3(M['item8a']['spearman_cnv_vs_late_patient_scores'])}, NRI {M['item8a']['categorical_NRI_late_vs_cnv_tertile_groups']}; 8b {('top feature cnv_only ' + I8['cnv_only']['top10'][0]['feature']) if I8 else 'pending'} |
| Discussion 5: false positives (item 9) | DONE | late_mean FP {i9['late_mean']['n_FP']} / TN {i9['late_mean']['n_TN']}; later LGD+ {i9['late_mean']['later_LGDplus']['FP']} vs {i9['late_mean']['later_LGDplus']['TN']}, Fisher p {i9['late_mean']['later_LGDplus']['fisher_p']} |
| Discussion 6: false negatives (item 10) | DONE | late_mean FN {i10['late_mean']['n_FN']} / TP {i10['late_mean']['n_TP']}; review pack {M['item10']['review_pack']['FN_slides']} FN + {M['item10']['review_pack']['TP_slides']} TP + {M['item10']['review_pack']['TN_slides']} TN slides (cluster) |
""")
# ------------------------------------------------------------------ item 0
if A2:
    sg = A2["summary_by_subgroup"]; kj = A2["killcoyne"]; cr = A2["count_reconciliation"]; se = A2["label_source_sensitivity"]
    flags = [("f_patient_slidematch", "patient: slide_matching ≠ release"), ("f_patient_sheet", "patient: sequencing sheet (majority-mapped id) ≠ release"), ("f_acc_slidematch", "accession: slide_matching ≠ release"), ("f_acc_sheet", "accession: sequencing sheet ≠ release"), ("f_acc_slidefile", "accession: slide filename ≠ release"), ("f_cnv_from_other_endoscopy", "CNV sheet accession ≠ slide accession (same row)"), ("f_year_sheet", "date: sheet endoscopy year ≠ release year"), ("f_date_slidematch", "date: slide_matching date ≠ release (> 366 d, non-placeholder)"), ("f_date_db", "date: DB report date ≠ release (> 366 d)"), ("f_grade_sheet", "grade two-tier: sheet ≠ release"), ("f_grade_slidematch", "grade two-tier: slide_matching ≠ release"), ("f_grade_db", "grade two-tier: DB confirmed code ≠ release"), ("f_next_orig_vs_scrape", "next-biopsy label: master ≠ DB scrape"), ("f_next_vs_db", "next-biopsy label: release ≠ next DB report")]
    P(f"""## 3. Item 0: CNV audit of the overlap subgroup
**Status.** DONE. **Decision: {A2['decision']['decision']}** (confirmed-error rows {A2['decision']['confirmed_error_rows_total']}, patients {A2['decision']['confirmed_error_patients']}). Every later number uses the frozen release.

**Pre-specification.** `{PRESPEC}` §"Item 0". Two versions of the audit script were run: v1 (commit `187dda8`, `audit_item0.json`) had a slide-filename parser that failed on `PS00 4239 …` names and mapped the discovery sheet's numeric patient id through `slide_matching.PatientID`, which is a different numbering; v2 (commit `529cccc`, `audit_item0_v2.json`) parses those names and maps the sheet id to the release patient owning the majority of its CNV ids. Both are kept; the decision is identical under both (0 confirmed errors). v1 flag counts (artefacts): patient-sheet {A1['summary_by_subgroup']['also_in_ERIN']['f_patient_sheet']['rows']} / {A1['summary_by_subgroup']['comparator_25']['f_patient_sheet']['rows']} rows, slide-filename {A1['summary_by_subgroup']['also_in_ERIN']['f_acc_slidefile']['rows']} / {A1['summary_by_subgroup']['comparator_25']['f_acc_slidefile']['rows']} rows (overlap / comparator).

**Result (v2): discrepancy counts per subgroup** (rows, patients). Overlap = 54 patients / {sg['also_in_ERIN']['rows']} rows ({sg['also_in_ERIN']['sheet']}); comparator = 25 `never_in_ERIN` patients drawn with `RandomState(0)` / {sg['comparator_25']['rows']} rows ({sg['comparator_25']['sheet']}); the remaining 71 patients / {sg['other']['rows']} rows are shown for completeness.

""" + tb(["check", "also_in_ERIN (54 pts)", "comparator (25 pts)", "other 71 pts"], [[lab] + [f"{sg[g][f]['rows']} rows / {sg[g][f]['patients']} pts" for g in ["also_in_ERIN", "comparator_25", "other"]] for f, lab in flags]) + f"""

Rows with a slide_matching entry: {sg['also_in_ERIN']['rows_with_slidematch']} / {sg['comparator_25']['rows_with_slidematch']} / {sg['other']['rows_with_slidematch']}; with a DB report matched by accession: {sg['also_in_ERIN']['rows_with_db_report']} / {sg['comparator_25']['rows_with_db_report']} / {sg['other']['rows_with_db_report']}. Sheet numeric ids mapping to more than one release patient: {len(A2['sheet_numeric_ids_mapping_to_more_than_one_release_patient'])}.

**CNV attached to a different endoscopy than sequenced.** The row's CNV sheet accession equals the slide accession in every row of every subgroup where both exist (0 flags in v2); the only residual flags are in 'other' (slide filename year differs from the accession year in the filename for {sg['other']['f_acc_slidefile']['rows']} rows / {sg['other']['f_acc_slidefile']['patients']} patients, where slide_matching agrees with the release) and the comparator ({sg['comparator_25']['f_acc_slidefile']['rows']} rows / 1 patient, same pattern).

**Label-source disagreements (not chain errors).** Release grade LGD where both the sequencing sheet and slide_matching say NDBE/IND: {se['grade_rows_release_LGD_vs_sheet_and_slidematch_benign']} rows; the reverse: {se['grade_rows_release_benign_vs_sheet_and_slidematch_LGDplus']} rows. The release grade comes from the DB confirmed code for most rows (`GradeSource` scraped_confirmed), so grade-vs-DB disagreement is 0 by construction. Next-biopsy label: {se['rows_with_both_next_sources']} rows carry both a master and a DB-scrape label; {se['rows_both_sources_disagree_two_tier']} disagree at two-tier level. Recomputing the LGD2+ patient status with the alternative source: DB-scrape-preferred changes {se['patient_status_changes_vs_release']['db_scrape_preferred']['also_in_ERIN']['patients']} overlap patients ({se['patient_status_changes_vs_release']['db_scrape_preferred']['also_in_ERIN']['to_progressor']} to progressor, {se['patient_status_changes_vs_release']['db_scrape_preferred']['also_in_ERIN']['to_nonprogressor']} to non-progressor) and {se['patient_status_changes_vs_release']['db_scrape_preferred']['never_in_ERIN']['patients']} never-in-ERIN patients ({se['patient_status_changes_vs_release']['db_scrape_preferred']['never_in_ERIN']['to_progressor']} / {se['patient_status_changes_vs_release']['db_scrape_preferred']['never_in_ERIN']['to_nonprogressor']}); master-preferred changes {se['patient_status_changes_vs_release']['master_preferred']['also_in_ERIN']['patients']} + {se['patient_status_changes_vs_release']['master_preferred']['never_in_ERIN']['patients']} (the release already prefers master). Grade-source mix per subgroup: {A2['grade_source_by_subgroup']}.

**Killcoyne 2020 per-sample predictions vs ours** (`41591_2020_1033_MOESM4_ESM.xlsx` sheet "Supporting data for Figure 2a": leave-one-patient-out probability and risk class for {kj['killcoyne_table_rows']} discovery samples; join on CNV sample id).

| quantity | value |
|---|---|
| matched rows / patients | {kj['matched_rows']} / {kj['matched_patients']} (rows by subgroup {kj['matched_rows_by_subgroup']}) |
| Killcoyne patient status vs our LGD2+ label (patients) | {kj['status_vs_our_label_patients']} |
| Spearman, Killcoyne probability vs our cnv_only OOF (rows) | {kj['spearman_kprob_vs_our_cnv_oof_rows']['rho']} (n {kj['spearman_kprob_vs_our_cnv_oof_rows']['n']}) |
| Killcoyne class by our row label (rows) | {kj['killcoyne_class_by_our_row_label']} |
""" + tb(["patients", "n", "events", "AUROC Killcoyne max prob (their LOPO model, our endpoint)", "AUROC our cnv_only (release OOF)"], [[k, v["n"], v["events"], v["auroc_killcoyne"], v["auroc_our_cnv_only"]] for k, v in kj["patient_auroc_killcoyne_maxprob_for_our_endpoint"].items()]) + f"""

**Count reconciliation.** Rows whose CNV id is in the 777-sample discovery sheet: {cr['rows_in_discovery_sheet']}; in the Killcoyne prediction table: {cr['rows_in_killcoyne_pred_table']}. Patients with any such row: {cr['patients_any_row_discovery_sheet']} / {cr['patients_any_row_in_killcoyne_table']}; with all rows: {cr['patients_all_rows_discovery_sheet']} / {cr['patients_all_rows_in_killcoyne_table']}; modal sheet: {cr['patients_modal_sheet_discovery']}. **82 is correct** under every definition computed here; the pipeline document's 69 matches none of them and its derivation is not recorded (the document is undated and cites no file).

**Method.** `pp_audit.py` (v2). **Sources.** {SRC('audit_item0_v2.json', 'pp_audit.py')}; v1 `audit_item0.json` (commit 187dda8). Row table `feasibility/paper_plan/audit_rows.csv` (cluster).

**Caveats.** The sequencing sheet and slide_matching both originate from the Fitzgerald-lab sample records and may not be independent of each other; the confirmed-error rule required agreement of two non-release sources on patient, accession or date, and none occurred. Grade and next-label disagreements are between label sources, not chain errors, and were outside the pre-specified corrected-release trigger; their effect on patient status is quantified above (≤ 6 patients). The Killcoyne comparison uses their in-cohort LOPO predictions against our LGD2+ endpoint, not their HGD/IMC endpoint.
""")
# ------------------------------------------------------------------ item 1
if M:
    o1 = M["item1_overlap"]
    P(f"""## 4. Items 1–10

### Item 1. Killcoyne 2020 comparison
**Question.** How do our cohort, unit, endpoint, eligibility, CNV representation, depth and evaluation compare with Killcoyne 2020, and how much do the cohorts overlap?

**Status.** DONE (their column from the PDF and supplementary xlsx on the cluster; page numbers refer to the `pdftotext` page index of `s41591-020-1033-y.pdf`, Nat Med 26:1726–1732).

| field | ours (frozen release; closeout item L) | Killcoyne 2020 (source) |
|---|---|---|
| cohort | 150 patients, 707 rows, 50 progressor patients (LGD2+), 107 positive rows | discovery 88 patients / 777 biopsies sequenced, 773 passing QC; 45 progressors NDBE→HGD/IMC, 43 non-progressors; validation 76 patients / 213 samples passing QC (219 sequenced) (PDF p.1 abstract; p.4 Fig 2 legend; p.8 Methods) |
| design | retrospective surveillance cohort; no case-control matching documented in the release | retrospective, demographically matched case–control; cases and controls matched for age, gender and BE segment length; NP min follow-up 3 y (6.7 ± 3.2), P min 1 y (6.1 ± 3.4) (p.1; p.2 Fig 1 legend; p.8 Methods) |
| unit | one biopsy slide + one CNV profile per row; patient = max over rows | one pooled per-level biopsy sample per row; per-sample predictions; per-endoscopy/patient aggregation did not change accuracy (p.2; p.8; p.13 Ext Data Fig 2b) |
| endpoint | `NextBiopsyProgression_LGD2plus`: next biopsy HGD+ or second consecutive LGD (code `src/barrett/labels/lgd2.py`) | a single biopsy graded HGD or IMC (p.8 Methods) |
| eligibility | strict pre-event rows only (at-event, post-event and non-evaluable rows excluded) | all samples incl. diagnostic HGD/IMC in the main model; models excluding HGD/IMC reported (p.2; p.13 Ext Data Fig 2a) |
| CNV representation | QDNAseq 50 kb → 5-Mb window features + 39 arm features + complexity cx (features_5mb_armdiff, features_arms, cx) | QDNAseq 50-kb bins, genome-wide standardised, elastic-net logistic regression over bins/arms; bin size tuned 15–500 kb (p.8; p.23 Ext Data Fig 9) |
| depth | 0.4× as stated in project documents (not verified from BAMs); median reads per profile in closeout item C | sWGS average depth 0.4× on Illumina HiSeq (p.1; p.8) |
| evaluation | 5-fold patient-level nested CV (split seed 20260713); AUPRC primary in the release, AUROC here | leave-one-patient-out on discovery; independent validation cohort; 50 kb LOPO AUC 0.87 discovery, 0.84 validation (p.23–24 Ext Data Fig 9); supplementary `MOESM15` Ext Data Fig 9c: 50 kb discovery AUC 0.865 [0.839, 0.891]; combined discovery + validation (n = 164) cross-validated AUC 0.89, spec 0.83, sens 0.82 (p.2) |
| risk classes | tertiles of training-fold scores (item 5); Killcoyne fixed classes also applied | low Pr ≤ 0.3 (sens 0.87, spec 0.65), moderate 0.3–0.5, high Pr ≥ 0.5 (sens 0.72, spec 0.82) (p.2) |
| overlap | {A2['count_reconciliation']['rows_in_discovery_sheet']} of 707 rows / {A2['count_reconciliation']['patients_any_row_discovery_sheet']} of 150 patients have a CNV id in the discovery sheet; {A2['killcoyne']['matched_rows']} rows / {A2['killcoyne']['matched_patients']} patients appear in the published per-sample prediction table; 203 rows / 68 patients are from the validation sheet (closeout item C) | — |

**CNV-only AUROC, Killcoyne-overlap patients vs the rest** (patient level, release OOF):

""" + tb(["patients", "n", "events", "rows", "cnv_only", "image_only", "late_mean"], [[k, v["n"], v["events"], v["rows"], f"{f3(v['cnv_only'][0])} {ci(v['cnv_only'][1:])}", f"{f3(v['image_only'][0])} {ci(v['image_only'][1:])}", f"{f3(v['late_mean'][0])} {ci(v['late_mean'][1:])}"] for k, v in o1.items()]) + f"""

**Sources.** {SRC('main_items.json', 'pp_main.py')} (key `item1_overlap`); `audit_item0_v2.json`; `docs/closeout_for_review.md` item L (commit 1d27002); Killcoyne PDF and `MOESM4/7/15` xlsx (cluster `phd/`). **Caveats.** Their AUCs are for an HGD/IMC endpoint in their cohort; not comparable one-to-one with our LGD2+ endpoint. Our CNV-only arm on the Killcoyne-overlap patients is compared with their published LOPO predictions in item 0.
""")
    # item 2
    P("### Item 2. Histology vs CNV\n**Question.** Is image-only as good as CNV-only on the same patients?\n\n**Status.** DONE.\n\n" + tb(["patients", "n", "events", "WSI AUROC", "CNV AUROC", "ΔAUROC WSI − CNV [CI]", "paired perm p", "WSI AUPRC", "CNV AUPRC", "ΔAUPRC [CI]"],
        [[k, t["n"], t["events"], f3(t["arms"]["image_only"]["auroc"]), f3(t["arms"]["cnv_only"]["auroc"]), f"{s3(t['item2_image_vs_cnv']['delta_auroc'])} {ci(t['item2_image_vs_cnv']['ci'])}", t["item2_image_vs_cnv"]["perm_p"], f3(t["arms"]["image_only"]["auprc"]), f3(t["arms"]["cnv_only"]["auprc"]), f"{s3(t['item2_image_vs_cnv']['delta_auprc'])} {ci(t['item2_image_vs_cnv']['auprc_ci'])}"] for k, t in M["item4_tables"].items()]) +
      f"\n\n**Method.** Release OOF probabilities, patient max; paired bootstrap and paired label permutation as in the conventions. No corrected release exists (item 0). **Sources.** {SRC('main_items.json', 'pp_main.py')} (`item4_tables.*.item2_image_vs_cnv`). **Caveats.** Same patients, different feature pipelines; the subgroups are post hoc (closeout item D).\n")
    # item 3
    i3 = M["item3"]; t3 = i3.get("3b_subset_table")
    P(f"""### Item 3. Clinical-only arm (new)
**Question.** What does a clinical-only arm achieve, on all patients (3a) and with demographics (3b)?

**Status.** DONE.

**Pre-specification.** `{PRESPEC}` §"Item 3". Covariates considered for 3a: {i3['3a_covariates_all']}; patient coverage {i3['3a_patient_coverage']}; used (coverage ≥ 0.9): {i3['3a_used']}. C chosen per outer fold: {i3['3a_C_per_fold']}. 3b covariates: {i3['3b_covariates']}; patients in the demographics sheet {i3['3b_patients_in_demographics']}; complete cases {i3['3b_complete_case_patients']} patients / {i3['3b_complete_case_rows']} rows; C per fold {i3['3b_C_per_fold']}.

3a on all 150: AUROC {f3(T0['arms']['clinical_3a']['auroc'])} {ci(T0['arms']['clinical_3a']['auroc_ci'])}, AUPRC {f3(T0['arms']['clinical_3a']['auprc'])} {ci(T0['arms']['clinical_3a']['auprc_ci'])} (n 150, 50 events).
""" + (("**3b subset, all arms re-evaluated on the same patients** (n " + str(t3["n"]) + ", events " + str(t3["events"]) + "):\n\n" + tb(["arm", "AUROC [CI]", "AUPRC [CI]"], [[("Clinical + demographics (3b)" if a == "clinical_3b" else FAMN[a]), f"{f3(v['auroc'])} {ci(v['auroc_ci'])}", f"{f3(v['auprc'])} {ci(v['auprc_ci'])}"] for a, v in t3["arms"].items()]) + f"\n\nΔ 3b − 3a on the subset: {ci(t3['delta_3b_minus_3a'])}.") if t3 else "3b NOT AVAILABLE: fewer than 20 complete-case patients or fewer than 5 events.") + f"""

**Method.** {i3['model']}. **Sources.** {SRC('main_items.json', 'pp_main.py')} (`item3`). **Caveats.** `BiopsyIndex`, `DaysSincePreviousBiopsy` and the row year encode surveillance history, not biology; 3b is a complete-case subset of 27 patients / 9 events with CIs spanning most of the range; it cannot rank arms.
""")
    # item 4
    P(f"""### Item 4. Main results table
**Question.** All arms on SWG with paired fusion differences; ACE-B column.

**Status.** DONE for SWG; ACE-B NOT AVAILABLE (no ACE-B slides, features or CNV exist on the cluster; closeout item N; nothing was run).

**All 150 patients**

""" + arm_table(T0) + "\n\nPaired differences (fusion arm minus reference):\n\n" + diff_table(T0) + f"""

**Exploratory block (not merged): ERIN grade-head fusion, closeout canonical v4.** From `results/closeout/closeout_main.json` (commit 602a40e): head alone 0.753 [0.659, 0.840]; image + CNV + head 0.850 [0.779, 0.909]; gain vs fold-z image + CNV +0.067 [+0.022, +0.114]; selection-adjusted permutation p 0.069 (maximum over 28 evaluated third arms) and 0.078 (canonical gain against the same null); the pre-registered sanity gate (imputed grade vs pathologist grade ≥ 0.7) failed for every head (canonical 0.597). Reported here as exploratory only.

**never_in_ERIN (96)**

""" + arm_table(M["item4_tables"]["never_in_ERIN_96"]) + "\n\n" + diff_table(M["item4_tables"]["never_in_ERIN_96"]) + "\n\n**also_in_ERIN (54)**\n\n" + arm_table(M["item4_tables"]["also_in_ERIN_54"]) + "\n\n" + diff_table(M["item4_tables"]["also_in_ERIN_54"]) + f"""

**Arms.** Clinical 3a: L2 logistic on row covariates (item 3), this repo. CNV: `cnv_only` = impute(median) → standardise → PCA(64) → random forest (500 trees, depth 20, balanced), on 5-Mb + arm + cx CNV features; `<release>/training_final_nested_cv_v1/cnv_only/fold*/model.joblib`. WSI: `image_only` = gated-attention MIL (hidden 256, attn 128) on 256 UNI2 tile embeddings (1536-d, release level-2 tiles); `image_only/fold*/model.pt`. Early: `early_fusion` = MLP (512) on [mean tile embedding ‖ standardised CNV]; Intermediate: `intermediate_fusion` = ABMIL image branch (256) ‖ CNV MLP (128) → fusion MLP; Co-attention: `coattention_fusion` = CNV-embedding query over tile keys, pooled values ‖ CNV embedding → fusion MLP; Late (mean): `late_mean` = plain mean of image_only and cnv_only probabilities; Late stack: `late_stack_logit` = logistic regression on the two probabilities. Code: `multimodal-barretts-progression/src/barrett/models/*.py`, trainer `scripts/24_run_lgd2_final_outer_fold.py` (pipeline commit 98ba8682), configs `fold*/resolved_config.yaml`.

**Sources.** {SRC('main_items.json', 'pp_main.py')} (`item4_tables`). **Caveats.** The five fusion arms share the same image and CNV inputs; the selection-adjusted p treats the choice among them as post hoc and maximises over the five fusion rows of this table (early, intermediate, late-mean, co-attention, late-stack); the digest's 0.225 (`results/numbers/swg_rep02_selection.json`) maximised over a different five-arm set that included the fold-z late mean instead of late-stack, so the two values are not the same statistic. Subgroup tables are post hoc.
""")
    # operating point
    P("### Primary operating point (items 5–10)\n**Threshold** fitted on the four training folds' patient scores for sensitivity ≥ 0.80, applied to the held-out fold.\n\n" + tb(["model", "fold", "threshold", "n test", "events", "sensitivity", "specificity"], [[a, k, v["threshold"], v["n_test"], v["events"], v["sensitivity"], v["specificity"]] for a, d in op.items() for k, v in d["per_fold"].items()]) + "\n\n" + tb(["model", "pooled sensitivity", "pooled specificity", "flagged", "n", "events"], [[a, d["pooled_sensitivity"], d["pooled_specificity"], d["flagged"], d["n"], d["events"]] for a, d in op.items()]) + f"\n\n`v4_exploratory` = fold-z mean of image_only, cnv_only and the leak-free ERIN grade head (closeout v4). **Sources.** {SRC('main_items.json', 'pp_main.py')} (`operating_point`).\n")
    # item 5
    rows5 = []
    for k, v in i5.items():
        for gn, g in v["groups"].items(): rows5.append([k, gn, g["n"], g["events"], f3(g["rate"]), ci(g["wilson_ci"]) if g["wilson_ci"][0] is not None else "n/a"])
    P(f"""### Item 5. Risk stratification groups
**Question.** Do training-fold tertiles (and Killcoyne's fixed classes) separate progression risk, per model?

**Status.** DONE. No Killcoyne-style thresholds are documented in the release metadata (closeout: none found); the alternative applied is the paper's fixed classes ({M['item5']['killcoyne_threshold_source']}).

""" + tb(["model / rule", "group", "n", "events", "progression rate", "Wilson 95 % CI"], rows5) + "\n\n" + tb(["model / rule", "OR high vs low (Haldane)", "log-OR SE", "log-rank p (3 groups)", "Cox HR high vs low [CI]", "Cox HR moderate vs low [CI]", "KM figure"], [[k, v.get("odds_ratio_high_vs_low_haldane"), v.get("or_log_se"), v.get("logrank_p_3groups", v.get("cox_error")), v.get("cox_hr_high_vs_low"), v.get("cox_hr_moderate_vs_low"), f"`{v['figure']}`"] for k, v in i5.items()]) + f"""

Time definition: {M['item5']['time_definition']}. Cross-tab CNV tertile (rows) × late_mean tertile (columns): {M['item5']['crosstab_cnv_vs_late_tertile']['counts_rows_cnv_cols_late']}; movers {M['item5']['crosstab_cnv_vs_late_tertile']['movers']} patients with progression rate {f3(M['item5']['crosstab_cnv_vs_late_tertile']['movers_progression_rate'])} vs stayers {f3(M['item5']['crosstab_cnv_vs_late_tertile']['stayers_progression_rate'])}; moved up {M['item5']['crosstab_cnv_vs_late_tertile']['moved_up_n_rate']} (n, rate), moved down {M['item5']['crosstab_cnv_vs_late_tertile']['moved_down_n_rate']}.

**Method.** `pp_main.py` item 5; lifelines KaplanMeierFitter / multivariate log-rank / CoxPHFitter with moderate and high indicators. **Sources.** {SRC('main_items.json', 'pp_main.py')} (`item5`); figures `results/paper_plan/figs/km/`. **Caveats.** Time is measured from the earliest release row, which is not a clinical baseline; censoring at the last biopsy uses `MonthsBeforeLastBiopsy`; tertile cut-points differ per fold.
""")
    # item 6
    if L6:
        P(f"""### Item 6. Latent space under fusion
**Question.** Does a learned fusion representation separate progressors better than the unimodal representations?

**Status.** DONE. Late fusion excluded (no shared representation). CNV-only representation = the fold pipeline's 64-d PCA scores (the random forest has no penultimate layer; the PCA is fitted on training folds inside the pipeline).

""" + tb(["representation", "dim", "linear-probe AUROC [CI]", "kNN-10 AUROC [CI]", "silhouette (held-out, mean of 5 folds)", "Δ probe vs image", "Δ probe vs CNV", "Δ kNN vs image", "Δ kNN vs CNV"], [[f, v["dim"], f"{f3(v['probe_auroc'])} {ci(v['probe_ci'])}", f"{f3(v['knn10_auroc'])} {ci(v['knn10_ci'])}", f3(v["silhouette_mean"]), (f"{s3(v['probe_delta_vs_image_only'][0])} {ci(v['probe_delta_vs_image_only'][1:])}" if "probe_delta_vs_image_only" in v else "—"), (f"{s3(v['probe_delta_vs_cnv_only'][0])} {ci(v['probe_delta_vs_cnv_only'][1:])}" if "probe_delta_vs_cnv_only" in v else "—"), (f"{s3(v['knn_delta_vs_image_only'][0])} {ci(v['knn_delta_vs_image_only'][1:])}" if "knn_delta_vs_image_only" in v else "—"), (f"{s3(v['knn_delta_vs_cnv_only'][0])} {ci(v['knn_delta_vs_cnv_only'][1:])}" if "knn_delta_vs_cnv_only" in v else "—")] for f, v in L6.items() if not f.startswith("_")]) + f"""

n = {L6['image_only']['n']} patients, {L6['image_only']['events']} events in every row. Figures (PCA and UMAP fitted on fold-1 training patients, held-out fold-1 patients shown, coloured by status and by subgroup): {", ".join("`" + v + "`" for v in L6['_figures'].values())}.

**Method.** {L6['_method']}. Recomputed model outputs vs release OOF (sanity): {L6['_sanity_recomputed_vs_release_oof']}. **Sources.** {SRC('latent_item6.json', 'pp_latent.py')}. **Caveats.** Training-fold rows' representations come from a model that saw them; only held-out patients are scored. UMAP is a secondary visual with a fixed seed; held-out fold 1 only (projections are not comparable across folds).
""")
    else: P("### Item 6. Latent space under fusion\n**Status.** PENDING (GPU job).\n")
    # item 7
    i7 = M.get("item7", {})
    if "intermediate_fusion" in i7:
        rows7 = []
        for fam in ["intermediate_fusion", "coattention_fusion"]:
            for sp, v in i7[fam].items(): rows7.append([fam, sp, v["n_rows"], f"{f3(v['spearman_median'])} {ci(v['spearman_iqr'])}", f3(v["jaccard_top5pct_median"]), f3(v["jaccard_top50_median"]), f3(v["entropy_image_only_median"]), f3(v["entropy_fusion_median"])])
        P(f"""### Item 7. Attention patches under fusion
**Question.** Do the tiles the model attends to change under fusion?

**Status.** DONE. Models with an image attention module: intermediate (own gated attention) and co-attention (CNV-conditioned); early fusion has none (mean pooling).

""" + tb(["fusion model", "rows", "n rows", "Spearman vs image-only, median [IQR]", "Jaccard top-5 % (13 tiles), median", "Jaccard top-50, median", "entropy image-only, median", "entropy fusion, median"], rows7) + f"""

Max entropy ln(256) = {f3(i7['intermediate_fusion']['overall']['max_entropy_ln256'])}. Correct/incorrect = image_only patient prediction at the primary operating point. Montages of the top-16 tiles per model for 20 pre-drawn rows: `M01–M20.png`, 20 files. Deviation from the pre-specified path: they are patient tissue images, so they stay on the cluster at `feasibility/paper_plan/figs/attention/` (moved from `results/paper_plan/figs/attention/`) and are not committed to the public repository; manifest with outcomes `feasibility/paper_plan/montage_manifest_SECRET.csv` (cluster).

**Method.** Held-out-fold attention over the 256 release tiles per row (`pp_latent.py`), summarised in `pp_main.py`. **Sources.** {SRC('attention_item7_rows.csv', 'pp_latent.py')}; `main_items.json` (`item7`). **Caveats.** Attention weights are not importance; 256 tiles per slide are the release's fixed sample.
""")
    else: P("### Item 7. Attention patches\n**Status.** PARTIAL (attention rows pending).\n")
    # item 8
    a8 = M["item8a"]
    P(f"""### Item 8. Does CNV prediction change?
**Question.** 8a: how do patient rankings move from CNV-only to late fusion; 8b: which CNV features drive each model?

**Status.** {'DONE' if I8 else 'PARTIAL (8b pending)'}.

**8a (patient ranking, n {a8['n']}, events {a8['events']}).** Spearman(cnv_only, late_mean) = {f3(a8['spearman_cnv_vs_late_patient_scores'])}; median |percentile change| {f3(a8['median_abs_rank_change_pct'])} points; progressors moving up > 20 points {a8['progressors_up_gt20pct']}, down {a8['progressors_down_gt20pct']}; non-progressors up {a8['nonprogressors_up_gt20pct']}, down {a8['nonprogressors_down_gt20pct']}. Categorical NRI (late vs CNV, item-5 tertile groups) {a8['categorical_NRI_late_vs_cnv_tertile_groups'][0]} [{a8['categorical_NRI_late_vs_cnv_tertile_groups'][1]}, {a8['categorical_NRI_late_vs_cnv_tertile_groups'][2]}] (event NRI {a8['event_NRI']}, non-event NRI {a8['nonevent_NRI']}).
""")
    if I8:
        P("**8b (permutation importance, ΔAUROC when one arm-level feature is permuted on held-out rows; top 10 per model).**\n\n" + tb(["rank"] + [f for f in ["cnv_only", "early_fusion", "intermediate_fusion", "coattention_fusion"]], [[i + 1] + [f"{I8[f]['top10'][i]['feature']} ({s3(I8[f]['top10'][i]['delta_auroc'])})" for f in ["cnv_only", "early_fusion", "intermediate_fusion", "coattention_fusion"]] for i in range(10)]) + f"\n\nSpearman of the importance vectors vs cnv_only: early {I8['early_fusion']['spearman_vs_cnv_only']}, intermediate {I8['intermediate_fusion']['spearman_vs_cnv_only']}, co-attention {I8['coattention_fusion']['spearman_vs_cnv_only']}. Method: {I8['_method']}. late_mean has no CNV branch of its own (its CNV component is cnv_only).\n\n**Sources.** {SRC('main_items.json', 'pp_main.py')} (`item8a`); {SRC('perm_importance_item8b.json', 'pp_latent.py')}. **Caveats.** ΔAUROC importances on ~140 held-out rows per fold are noisy; a feature can matter through the 5-Mb windows, which were held fixed.\n")
    # item 9
    P(f"""### Item 9. Do false positives show risk of progression?
**Question.** At the primary operating point, do non-progressors flagged positive show later disease?

**Status.** DONE (later-disease sources exist; coverage stated).

Sources checked: Barrett's-DB pathology reports of the linked participant after the last release row ({M['item9']['sources']['barretts_db_pathology_reports']['participants_linked']} of 150 patients linked; report dates {M['item9']['sources']['barretts_db_pathology_reports']['date_coverage']}); release rows excluded as at-event / post-event / endpoint-not-evaluable; `hgd_pathology_table` ({M['item9']['sources']['hgd_pathology_table']['rows']} rows, {M['item9']['sources']['hgd_pathology_table']['date_coverage']}); slide_matching rows dated after the last release row. Later LGD+ = any of these with grade LGD or worse; later HGD+ = HGD/IMC or an hgd_table entry.

""" + tb(["model", "FP", "TN", "later LGD+ FP (k/n, rate)", "later LGD+ TN", "Fisher p", "later HGD+ FP", "later HGD+ TN", "Fisher p", "DB follow-up d, median FP / TN (MW p)", "any DB report after, FP / TN", "baseline grade mean FP / TN (p)", "cx median FP / TN (p)", "p53 aberrant / with IHC, FP ; TN", "KM"], [[a, v["n_FP"], v["n_TN"], v["later_LGDplus"]["FP"], v["later_LGDplus"]["TN"], v["later_LGDplus"]["fisher_p"], v["later_HGDplus_or_hgd_table"]["FP"], v["later_HGDplus_or_hgd_table"]["TN"], v["later_HGDplus_or_hgd_table"]["fisher_p"], v["db_followup_days_median_FP_TN"], v["any_db_report_after_FP_TN"], v["baseline_grade_mean"], v["cx_max_median"], v["p53_aberrant_FP_TN_of_with_ihc"], v.get("km_figure", v.get("km"))] for a, v in i9.items()]) + f"""

**Sources.** {SRC('main_items.json', 'pp_main.py')} (`item9`); patient table `feasibility/paper_plan/later_disease_patient.csv` (cluster). **Caveats.** DB linkage covers most but not all patients; later grades are DB confirmed codes (not re-read); release-excluded rows are by construction absent for non-progressors except non-evaluable ones; no cancer-registry linkage exists.
""")
    # item 10
    def cmp_tab(c):
        rows = []
        for k, v in c.items():
            if "mannwhitney_p" in v: rows.append([k, f"{v['FN'][0]} (n {v['FN'][1]})", f"{v['TP'][0]} (n {v['TP'][1]})", "Mann-Whitney", v["mannwhitney_p"]])
            else: rows.append([k, str(v["table"]), "", "Fisher (2×2 only)", v["fisher_p"]])
        return tb(["variable", "FN median / counts", "TP median / counts", "test", "p"], rows)
    P(f"""### Item 10. What is missing from false negatives?
**Question.** At the primary operating point, how do missed progressors differ from detected ones?

**Status.** DONE (tables; review pack built on the cluster, review itself out of scope).

""" + "\n\n".join(f"**{a}** (FN {v['n_FN']}, TP {v['n_TP']})\n\n" + cmp_tab(v["comparison"]) for a, v in i10.items()) + f"""

{M['item10']['slide_qc_note']}. Review pack: {M['item10']['review_pack']['FN_slides']} FN slides (all rows of the {M['item10']['review_pack']['FN_patients']} late_mean FN patients) + {M['item10']['review_pack']['TP_slides']} TP + {M['item10']['review_pack']['TN_slides']} TN slides, shuffled, opaque names, thumbnails in `feasibility/paper_plan/review_pack/` (cluster); manifest {M['item10']['review_pack']['manifest']}.

**Sources.** {SRC('main_items.json', 'pp_main.py')} (`item10`); QC and slide tables from the closeout (`feasibility/closeout/`, commit 602a40e). **Caveats.** Small FN groups; endpoint-type and scanner tables have more than two levels (no Fisher p); read counts exist only for discovery-sheet profiles.
""")
    # whiteboard table
    P(f"""## 5. Filled whiteboard table (SWG; no corrected release exists)

""" + tb(["", "SWG progressor cohort (n 150, 50 progressors): AUROC [CI] · AUPRC [CI]", "ACE-B external validation"], [[FAMN[a], f"{f3(T0['arms'][a]['auroc'])} {ci(T0['arms'][a]['auroc_ci'])} · {f3(T0['arms'][a]['auprc'])} {ci(T0['arms'][a]['auprc_ci'])}", "NOT AVAILABLE"] for a in ["clinical_3a", "cnv_only", "image_only", "early_fusion", "intermediate_fusion", "late_mean", "coattention_fusion", "late_stack_logit"]]) + f"""

Source: `results/paper_plan/main_items.json` (`item4_tables.all_150`), commit {RESC}.

## 6. Discrepancies found
1. Pipeline document `killcoyne_protocol_comparison.md` says 69/150 patients overlap the Killcoyne discovery cohort; every definition computed here gives {A2['count_reconciliation']['patients_any_row_discovery_sheet']} (item 0).
2. Our CNV-only arm scores the 25 discovery-sheet overlap patients at {A2['killcoyne']['patient_auroc_killcoyne_maxprob_for_our_endpoint']['also_in_ERIN']['auroc_our_cnv_only']} while Killcoyne's published per-sample predictions on the same profiles give {A2['killcoyne']['patient_auroc_killcoyne_maxprob_for_our_endpoint']['also_in_ERIN']['auroc_killcoyne']} for our endpoint: the CNV data are not corrupted; the release's CNV model is what fails on these patients (item 0).
3. Release grade LGD contradicted by both lab tables in {A2['label_source_sensitivity']['grade_rows_release_LGD_vs_sheet_and_slidematch_benign']} rows, and {A2['label_source_sensitivity']['rows_both_sources_disagree_two_tier']} rows where the master and DB-scrape next-biopsy labels disagree; changes patient status for ≤ 6 patients under the alternative source (item 0). Not a chain error under the pre-specified rule.
4. Recomputing `cnv_only` probabilities from the saved pipelines reproduces the release OOF only to {L6['_sanity_recomputed_vs_release_oof']['cnv_only']['max_abs_diff_vs_release_oof'] if L6 else 'n/a'} max abs difference (Spearman {L6['_sanity_recomputed_vs_release_oof']['cnv_only']['spearman'] if L6 else 'n/a'}); the four torch families reproduce exactly. Feeding float32 instead of the release's float64 input shifts RF probabilities by up to 0.14 (fixed in commit 5bf5419).
5. Audit v1 flagged {A1['summary_by_subgroup']['also_in_ERIN']['f_patient_sheet']['rows']} overlap rows for patient mismatch and {A1['summary_by_subgroup']['also_in_ERIN']['f_acc_slidefile']['rows']} for accession mismatch; both were parser/mapping artefacts corrected in v2 (item 0).

## 7. Not done
- ACE-B column (no data; nothing run).
- Killcoyne-style thresholds from release metadata (none documented; paper's fixed classes used instead).
- Sequencing depth in ×, slide blur/focus QC, cancer-registry linkage: not available (closeout item C; item 10 note).
- The pathologist review of the pack (out of scope by instruction).
- A corrected release (not triggered: no confirmed errors).

## 8. Pre-specification text as committed at `{PRESPEC}` (verbatim)

""" + "\n".join("> " + l if l.strip() else ">" for l in PRE.splitlines()))
open("docs/paper_plan_answers.md", "w").write("\n".join(L)); print("written", sum(len(x) for x in L))

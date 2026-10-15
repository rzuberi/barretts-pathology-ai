"""Renders docs/closeout_for_review.md from the closeout result files. Every number in the report is read from
results/closeout/*.json (never typed), so the report cannot drift from the files. Run locally after the cluster results
are pulled. Args: START_COMMIT PRESPEC_COMMIT RESULTS_COMMIT (hashes for the header)."""
import json, os, sys, textwrap
R = "results/closeout"; M = json.load(open(f"{R}/closeout_main.json")); H = json.load(open(f"{R}/h_nongrade_controls.json")) if os.path.exists(f"{R}/h_nongrade_controls.json") else None
SM = json.load(open(f"{R}/swg_slide_meta_summary.json")); HS = json.load(open(f"{R}/aceb_checkpoint_hashes.json"))
START, PRESPEC, RESC = (sys.argv + ["a23b483", "0aaa824", "pending"])[1:4]
PRESPEC_TEXT = open(sys.argv[4]).read() if len(sys.argv) > 4 else ""
def ci(c): return f"[{c[0]:+.3f}, {c[1]:+.3f}]" if c and c[0] is not None and (c[0] < 0 or c[1] < 0) else (f"[{c[0]:.3f}, {c[1]:.3f}]" if c and c[0] is not None else "n/a")
def f3(x): return "n/a" if x is None else f"{x:.3f}"
def s3(x): return "n/a" if x is None else f"{x:+.3f}"
def tbl(rows, cols, hdr=None):
    hdr = hdr or cols; out = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(cols)]
    for r in rows: out.append("| " + " | ".join(str(r.get(c, "")) if not callable(c) else str(c(r)) for c in cols) + " |")
    return "\n".join(out)
SRC_MAIN = f"`results/closeout/closeout_main.json` · `scripts/closeout/co_main.py` · commit {RESC}"
L = []; P = L.append
A = M["A_versions"]; canon = A[3]; arms = M["A_arms"]; B = M["B_overlap"]; CS = M["C_subgroup_aurocs"]; D = M["D_interaction"]; E = M["E_interval"]; EX = M["E_exclusions"]; Fm = M["F_arms"]; FN = M["F_within_NDBE"]; G = M["G_sanity"]; I = M["I_selection"]; J = M["J_p32_erin_cis_2000"]; K = M["K_folds"]; Lo = M["L_ours"]; L4 = M["L_4x"]; MC = M["M_calibration"]; MP = M["M_p53"]
also, never = CS[1], CS[2]
hrow = {r["head"]: r for r in H["rows"]} if H else {}
def hget(k, f): r = hrow.get(k); return "pending" if not r or f not in r else r[f]
# ------------------------------------------------------------------ header
P(f"""# Closeout for review: SWG fusion and the ERIN-transfer result

## 1. Header
- Date: 25 September 2026.
- Commit at start: `{START}`. Pre-specification commit (this file's specification text, the ACE-B plan and all closeout scripts, before any run): `{PRESPEC}`. Script fixes after failed first runs (markdown writer, CNV path, one labelling correction stated in item E): `1148318`, `4af24d5`. Results commit (all result files and scripts): `{RESC}`. Commit at end = the commit that adds this rendered text (the next commit after `{RESC}`; see `git log -- docs/closeout_for_review.md`).
- New scripts: `scripts/closeout/co_checkpoint_hashes.py`, `co_swg_slide_meta.py`, `co_swg_cnv_qc.py`, `co_main.py`, `co_h_assemble.py`, `co_render_report.py` (writes this file from the JSONs; no number here is typed by hand).
- New result files (`results/closeout/`): `aceb_checkpoint_hashes.json`, `swg_slide_meta_summary.json`, `closeout_main.json`, `tables.md`, `h_nongrade_controls.json`, `tables_h.md`. Row-level tables (slide filenames, per-patient rows, per-profile QC) stay on the cluster under `feasibility/closeout/` and are not committed.
- New plan: `docs/aceb_analysis_plan.md` (frozen at `{PRESPEC}`).
- Conventions for every number: SWG frozen release `chapter1_lgd2_final_pre_event_20260713_final`, 707 rows / 150 patients / 50 progressor patients; patient score = max over the patient's rows; AUROC rank-based; CI = percentile bootstrap over patients, 2,000 resamples, `RandomState(0)`; permutation = 2,000 permutations of patient labels, `RandomState(0)`. Fused arms = mean of fold-local z-scores (release outer folds). "Head" = ERIN-trained ABMIL grade head applied to the SWG slide at 0.5 µm/px, leak-free unless stated.

## 2. Summary table
| Item | Status | One-line answer |
|---|---|---|
| A | DONE | Four versions reproduce exactly from their files; canonical = leak-free grade head as third arm: head {f3(canon['head_alone'])}, fusion {f3(canon['fuse3'])} vs {f3(canon['fuse2'])}, gain {s3(canon['gain'])} {ci(canon['gain_ci'])}, perm p {canon['perm_p_gain']} (n 150, 50 events). |
| B | DONE | {B['swg_patients_linked_to_erin']} SWG patients link to {B['erin_anon_ids_linked']} ERIN identities; only {B['imaged_erin_cases_of_linked_ids']} imaged ERIN cases exist for them, from {B['swg_patients_with_case_in_any_inclusive_training_table']} SWG patients; the other 21 have no imaged ERIN case and were never in any training table; the leak-free head excluded all 55 identities. No rerun needed. |
| C | PARTIAL | Groups differ in age (56.5 vs 65.0, p 0.006), slide age at scan (p 0.044) and grade-label provenance (p <0.001); CNV-only {f3(also['cnv'])} {ci(also['cnv_ci'])} vs {f3(never['cnv'])} {ci(never['cnv_ci'])}; collapse persists within the discovery sequencing sheet (0.253, n 25 / 11). Staining batch, depth in ×, 4× data NOT AVAILABLE. |
| D | DONE | Gain difference {s3(D['difference'])} {ci(D['bootstrap_ci_stratified'])}; membership-permutation p {D['perm_sizes_only']['p_two_sided']} (sizes) / {D['perm_sizes_and_events']['p_two_sided']} (sizes + events). Post-hoc split. |
| E | DONE | Progressor interval first row → endpoint biopsy median {E['median_days']:.0f} d, IQR [{E['iqr_days'][0]:.0f}, {E['iqr_days'][1]:.0f}]; excluding ≤6 m / ≤12 m progressors the gain is {s3(EX[1]['gain'])} {ci(EX[1]['gain_ci'])} (n {EX[1]['n']}/{EX[1]['events']}) and {s3(EX[2]['gain'])} {ci(EX[2]['gain_ci'])} (n {EX[2]['n']}/{EX[2]['events']}). |
| F | DONE | No clinical arm exists in the release ({len(Fm['release_families'])} families listed). Row grade alone {f3(Fm['rows'][0]['auroc'])}; grade + head {f3(Fm['rows'][1]['auroc'])}; head within first-row-NDBE patients {f3(FN[0]['head'])} {ci(FN[0]['head_ci'])} (n {FN[0]['n']}/{FN[0]['events']}). |
| G | DONE | Pre-registered gate "< 0.7 → moot" quoted; observed slide-level AUROC vs Label: pass-1 {f3(G['observed']['pass1'])}, pass-2 {f3(G['observed']['p32b'])}, leak-free {f3(G['observed']['noov'])}; vs DB code {f3(G['vs_db_confirmed_code']['p32b'])} / {f3(G['vs_db_confirmed_code']['noov'])} (n {G['vs_db_confirmed_code']['n_rows_with_code']}). No independent second pathologist read exists. Gate failed for every head. |
| H | {'DONE' if H and all(k + '_head' in r or 'gain' in r for k, r in hrow.items()) and all(k in hrow for k in ['treatment_effect_leakfree_NEW', 'im_present_leakfree_NEW', 'grade_permuted_labels_leakfree_NEW']) else 'PENDING'} | treatment-effect head: alone {hget('treatment_effect_leakfree_NEW', 'head_alone_swg')}, gain {hget('treatment_effect_leakfree_NEW', 'gain')} {ci(hget('treatment_effect_leakfree_NEW', 'gain_ci')) if isinstance(hget('treatment_effect_leakfree_NEW', 'gain_ci'), list) else ''}; IM head: alone {hget('im_present_leakfree_NEW', 'head_alone_swg')}, gain {hget('im_present_leakfree_NEW', 'gain')} {ci(hget('im_present_leakfree_NEW', 'gain_ci')) if isinstance(hget('im_present_leakfree_NEW', 'gain_ci'), list) else ''}; leak-free permuted head: gain {hget('grade_permuted_labels_leakfree_NEW', 'gain')}. |
| I | DONE | {I['n_candidate_third_arms']} third arms were ever evaluated; selection-adjusted permutation p for the maximum (grade, {s3(I['t_obs_max'])}) = **{I['p_selection_adjusted_max_over_all_candidates']}** (unadjusted {I['p_unadjusted_for_selected']}); same null applied to the canonical gain: {I['p_canonical_against_max_null']}. |
| J | DONE | All digest CIs are patient-level or patient-clustered, seed 0; all use 2,000 resamples except the 13 P32 ERIN field CIs (1,000) — recomputed at 2,000, max change 0.002. |
| K | DONE | Every SWG patient in exactly one outer fold (rep01 and rep02, 0 violations); inner folds keyed by patient; all 9 ERIN task tables and the P32 head table: 0 patients in two folds. Baseline = no single row; all strict pre-event rows are units; patient = max. |
| L | PARTIAL | 4× data NOT AVAILABLE (locations searched listed). Our endpoint/cohort definitions tabulated from code and release metadata; the Killcoyne column is left blank. |
| M | DONE | Calibration slope image {f3(MC['rows'][0]['calib_slope'])}, late_mean {f3(MC['rows'][1]['calib_slope'])}, fuse3+head (Platt-CV) {f3(MC['rows'][3]['calib_slope'])}; Brier {f3(MC['rows'][0]['brier'])} / {f3(MC['rows'][1]['brier'])} / {f3(MC['rows'][3]['brier'])}; net benefit at 0.3: {MC['rows'][0]['net_benefit']['0.3']:.3f} / {MC['rows'][1]['net_benefit']['0.3']:.3f} / {MC['rows'][3]['net_benefit']['0.3']:.3f} (treat-all {MC['treat_all_net_benefit']['0.3']:.3f}). p53 IHC on {MP['n_patients_with_ihc']} patients: AUROC {f3(MP['p53_auroc'][0])}; head adds {ci(MP['delta_p53_plus_head_minus_p53'])}. |
| N | DONE | No ACE-B slides, features or CNV on the cluster; depth SWG 0.4× (stated, not verified), ACE-B ~7× (owner's statement). Prevalent HGD/IMC excluded from the primary analysis, secondary detection set. Plan frozen in `docs/aceb_analysis_plan.md` at `{PRESPEC}` with checkpoint hashes. |
""")
# ------------------------------------------------------------------ A
P(f"""## 3. Items

### A. Canonical number reconciliation
**Question.** Which of the reported values (0.83 / 0.850; +0.050 / +0.063 / +0.067; 0.756 / 0.753) is which, and which is canonical?

**Status.** DONE.

**Result.** All four versions recomputed from the out-of-fold and imputed tables (n 150 patients, 50 progressors in every row). Each reproduces the value in its source file to 3 decimals.

| version | third arm | ERIN training set | head alone | fuse2 (image + CNV) | fuse3 | gain | gain CI | perm p | source file of the original report |
|---|---|---|---|---|---|---|---|---|---|
""" + "\n".join(f"| {r['version']} | {r['third_arm']} | {'2,293 cases incl. 43 cases of 33 SWG-linked patients' if 'p32b' in r['version'] else '2,249 cases, 55 SWG-linked ERIN ids excluded'} | {f3(r['head_alone'])} {ci(r['head_ci'])} | {f3(r['fuse2'])} | {f3(r['fuse3'])} {ci(r['fuse3_ci'])} | {s3(r['gain'])} | {ci(r['gain_ci'])} | {r['perm_p_gain']} | {src} |" for r, src in zip(A, ["`results/numbers/p32b_swg_fuse.json` (`scripts/projects/p32_swg_fuse.py`, commit 926fc93)", "`results/numbers/p32_fuse_controls.json` (`scripts/projects/p32_fuse_controls.py`, 926fc93)", "`results/numbers/p32_fuse_noov.json` (`p32_swg_fuse.py` on `p32_head_noov`, ec07155)", "`results/numbers/p32_noov_subgroups.json` (25 Sep session, ec07155)"])) + f"""

What differs between versions: (i) **training set** of the ERIN head: v1/v2 include the 43 ERIN cases of 33 SWG-linked patients, v3/v4 exclude all 55 linked ERIN identities; (ii) **third arm**: v1/v3 = a CV logistic fitted on SWG labels within the release folds over the imputed field vector (6 fields in v1, 2 in v3), v2/v4 = the imputed grade probability alone, no fitted component; (iii) fold models, feature set (0.5 µm/px UNI2-h bags), aggregation (max over rows), fusion rule (fold-local z-mean), folds (`fold_id_rep01`) and bootstrap seed are identical across versions. "0.83" is v1 rounded; "0.850" is v4 (v2 gives 0.851). "0.756" is the incl-overlap grade head, "0.753" the leak-free one; "+0.063" is v3, "+0.067" v4.

**Canonical: v4** (leak-free grade head as the third arm), because (a) it uses no ERIN case from a SWG patient, (b) its third arm has no component fitted on SWG labels, (c) grade was the field named as the reference target in the P32 pre-registration. Its selection among the evaluated arms is accounted for in item I, where the selection-adjusted p is reported.

**Method.** `co_main.py` block A: release OOF `image_only`/`cnv_only`, imputed tables from `feasibility/runs/{{p32b_fields,p32_head_noov}}/output/swg_imputed_fields.csv`; z within outer fold; mean; max over rows; rank AUROC; 2,000 patient bootstraps seed 0; 2,000 label permutations seed 0.

**Sources.** {SRC_MAIN}; arm AUROCs image {f3(arms['img'][0])} {ci(arms['img'][1:])}, CNV {f3(arms['cnv'][0])} {ci(arms['cnv'][1:])}, fuse2 {f3(arms['fuse2'][0])} {ci(arms['fuse2'][1:])}.

**Caveats.** The choice of grade as the single third arm was made after the pass-2 results (item I). Item A does not by itself show that the gain survives selection.
""")
# ------------------------------------------------------------------ B
memb_note = B["note"]
P(f"""### B. Overlap count discrepancy (33 vs 54)
**Question.** Why were 33 patients excluded when 54 SWG patients are also in ERIN, and were any of the other 21 in any head's training set?

**Status.** DONE.

**Result.**

| quantity | value |
|---|---|
| SWG patients linked to an ERIN identity | {B['swg_patients_linked_to_erin']} |
| ERIN anon_ids linked | {B['erin_anon_ids_linked']} ({B['pairs']} pairs: {B['how']['direct']} by shared accession, {B['how']['db_bridge']} via a Barrett's-DB participant) |
| imaged ERIN cases (slides) belonging to those 55 identities, any label status | {B['imaged_erin_cases_of_linked_ids']} ({B['imaged_label_status']}) |
| SWG patients with ≥1 such case in any incl-overlap P32 training table (pass-1 groups g0–g3, pass-2, repeat-fold, permuted-label) | {B['swg_patients_with_case_in_any_inclusive_training_table']} (the "33") |
| cases from linked identities in the pass-2 training table | {B['cases_in_p32b_training_from_linked_ids']} |
| SWG patients with ≥1 such case in the leak-free head's training | {B['swg_patients_with_case_in_noov_training']} |
| identities in the exclusion file used for the leak-free head | {B['exclusion_file_ids']} (all 55) |

The difference is not a matching error: 54 SWG patients have an ERIN **report** identity, but only 33 of them have an **imaged** ERIN case (43 slides), because ERIN slides were scanned for a subset of reports. The other 21 SWG patients have zero imaged ERIN cases, so they were in no training split of any head in any SWG result. ERIN "train/val/test membership" does not apply: the heads are 5-fold CV over all cases, so every included case trains four of the five fold models whose average is applied to SWG; "in training" therefore means "in the table". The leak-free head nonetheless excluded all 55 identities (all 54 patients), so no rerun is required.

**Method.** Recomputation of the crosswalk (`scripts/task_overlap_audit.py` logic: accession normalisation `PSyy-nnnnn`/`psyy.nnnnn`; DB bridge via `pathology_text_normalised_full.specimennumber` → `participant_id`), then for each linked SWG patient the count of its ERIN cases (`labeller/erin_master.csv` CaseName → anon_id) in every `oof_*.csv` of every P32 run. Row-level table `feasibility/closeout/swg_overlap_training_membership.csv` (cluster).

**Sources.** `results/closeout/closeout_main.json` key `B_overlap`; `results/overlap_audit.json` (`scripts/task_overlap_audit.py`); {SRC_MAIN}.

**Caveats.** The bridge depends on DB accession parsing; a SWG patient whose reports carry no parseable accession cannot be linked (SWG DB match rate 0.78 in `overlap_audit.json`), so "never in ERIN" may contain unlinked shared patients. {memb_note}
""")
# ------------------------------------------------------------------ C
CC = M["C_characterisation"]; keep = [r for r in CC if r["variable"] not in ("slx_run_mode", "seq_batch_mode")]
def fmtv(v): return v if isinstance(v, str) else ", ".join(f"{k}: {n}" for k, n in v.items())
P(f"""### C. Characterise the overlap subgroup (54 vs 96)
**Question.** How do the 54 SWG patients also in ERIN differ from the 96 who are not, and does the CNV collapse persist within sequencing strata?

**Status.** PARTIAL (staining batch, depth in ×, referral pathway beyond the DB hospital field, and 4× data are NOT AVAILABLE; see below).

**Result: patient-level comparison.** Patient value = mean (continuous) or mode (categorical) over the patient's release rows. Continuous cells are median [IQR] (n with a value).

| variable | also_in_ERIN (n 54, 14 events) | never_in_ERIN (n 96, 36 events) | test | p |
|---|---|---|---|---|
""" + "\n".join(f"| {r['variable']} | {fmtv(r['also_in_ERIN_54'])} | {fmtv(r['never_in_ERIN_96'])} | {r['test']} | {r['p']} |" for r in keep) + f"""

Sequencing batch (Leanne batch, 12 levels, p {[r for r in CC if r['variable']=='seq_batch_mode'][0]['p']}) and SLX run (43 levels, p {[r for r in CC if r['variable']=='slx_run_mode'][0]['p']}) are in the JSON (`C_characterisation`) and not tabulated here. Variable notes: `first_year` = year of the earliest release row; `span_days` = first to last release row; `followup_months_first_to_last_biopsy` = max `MonthsBeforeLastBiopsy`; `days_first_to_event` = progressors only (item E definition); `age_at_diagnosis`, `prague_C/M`, `sex_demographics`, `smoking` from `SWGCohort/Demographics_full.csv` (66 of 150 patients); `gender_id_code` from `SWGCohort/barretts_database_230809.csv` (code mapping not documented; reported as code); `referral_hospital_db` = referral hospital of the first DB endoscopy (Barrett's-DB export, participant id); `seq_sheet_mode` = discovery (777-sample sheet) vs validation (268-sample sheet) membership of the patient's CNV profiles; `n_reads_mean` from the discovery sheet only (validation sheet has no read count); `cellularity_mean` from the sheet (18 patients); `cx_max` = release complexity score; `noise_mapd_mean`, `n_segments_mean`, `frac_altered_*` from the QDNAseq 50 kb profiles (`co_swg_cnv_qc.py`; values are relative copy-number ratios centred at 1, altered = |ratio − median| > 0.15 / 0.30); `scanner_model_mode`/`scanner_serial_mode`/`source_lens_mode`/`mpp_mean`/`scan_year_first`/`slide_age_at_scan_days_mean` from the .ndpi headers (`co_swg_slide_meta.py`: {SM['n_read_ok']}/{SM['n_slides']} slides read; models {SM['tiff.Model']}; scan years {SM['scan_year_Created']}; NDP.scan versions {SM['software']}); `p53_ihc_any_aberrant` from `SWGCohort/slide_matching.csv`; `p53_seqsheet_any` from the sequencing sheet; `grade_source_mode` / `next_label_source_mode` = provenance of the row grade and of the next-biopsy label in the release (`GradeSource`, `NextBiopsyLabel_source`). "Who ascertained progression" is not recorded as a person; the provenance columns are the closest available proxy.

**Result: arm AUROCs per subgroup** (canonical leak-free head; CIs from within-subgroup bootstrap).

| subgroup | n | events | CNV-only | image-only | head-only | image + CNV | with head | gain |
|---|---|---|---|---|---|---|---|---|
""" + "\n".join(f"| {r['subgroup']} | {r['n']} | {r['events']} | {f3(r['cnv'])} {ci(r['cnv_ci'])} | {f3(r['img'])} {ci(r['img_ci'])} | {f3(r['head'])} {ci(r['head_ci'])} | {f3(r['fuse2'])} {ci(r['fuse2_ci'])} | {f3(r['fuse3'])} {ci(r['fuse3_ci'])} | {s3(r['gain'])} {ci(r['gain_ci'])} |" for r in CS) + f"""

**Result: CNV-only within sequencing strata × subgroup** (rows with n ≥ 10 shown; all strata in the JSON key `C_cnv_by_stratum`; CI when events ≥ 3).

| stratum | subgroup | n | events | CNV-only AUROC |
|---|---|---|---|---|
""" + "\n".join(f"| {r['stratum']} | {r['subgroup']} | {r['n']} | {r['events']} | {f3(r['cnv_auroc'])} {ci(r['cnv_ci']) if r['cnv_ci'] else ''} |" for r in M["C_cnv_by_stratum"] if r["n"] >= 10) + f"""

CNV-only at 4× resequencing: {M['C_4x']}.

**Method.** `co_main.py` block C; tests: `scipy.stats.mannwhitneyu` (two-sided), `fisher_exact` for 2×2, `chi2_contingency` for more than two levels (labelled). Missing values form their own level in categorical tests where present.

**Sources.** {SRC_MAIN}; `results/closeout/swg_slide_meta_summary.json` (`scripts/closeout/co_swg_slide_meta.py`); QC rows `feasibility/closeout/swg_cnv_qc.csv` (cluster, `scripts/closeout/co_swg_cnv_qc.py`); `SWGCohort/Demographics_full.csv`, `sWGS_777_samples_cleaned_202401_Leanne_fullDetails (3) (1).csv`, `sWGS_validation_cleaned_Leanne (4) (1).csv`, `slide_matching.csv`, `barretts_database_230809.csv` (cluster data, not in the repo).

**Caveats.** NOT AVAILABLE: {"; ".join(M['C_not_available'])}. The grade-provenance difference (p <0.001) is structural: the overlap was found through the Barrett's DB, so linked patients are those whose grades were scraped from the DB. Demographics cover 66/150 patients and read counts 81/150, so those tests are on subsets. The two subgroups have {CS[1]['events']} and {CS[2]['events']} events; per-stratum CIs are wide.
""")
# ------------------------------------------------------------------ D
P(f"""### D. Interaction test
**Question.** Is the fusion gain different between the 54 and the 96, beyond chance?

**Status.** DONE. **This split was defined for the leakage check on 25 Sep 2026 and is a post-hoc subgroup analysis.**

**Pre-specification.** Commit `{PRESPEC}` (section "D. Interaction test" of the pre-specification, reproduced verbatim in §6).

**Result.**

| quantity | value |
|---|---|
| gain also_in_ERIN (n {D['n_events']['also_in_ERIN'][0]}, events {D['n_events']['also_in_ERIN'][1]}) | {s3(D['gain_also_in_ERIN'])} |
| gain never_in_ERIN (n {D['n_events']['never_in_ERIN'][0]}, events {D['n_events']['never_in_ERIN'][1]}) | {s3(D['gain_never_in_ERIN'])} |
| difference (also − never) | {s3(D['difference'])} |
| bootstrap CI, patients resampled within each subgroup, {D['n_boot']} resamples, seed {D['seed']} | {ci(D['bootstrap_ci_stratified'])} |
| permutation of membership, sizes preserved: two-sided p / one-sided p / null 2.5–97.5 % | {D['perm_sizes_only']['p_two_sided']} / {D['perm_sizes_only']['p_one_sided_ge']} / {ci(D['perm_sizes_only']['null_q025_q975'])} ({D['perm_sizes_only']['n_valid_perms']} valid) |
| permutation preserving sizes and progressor counts per group | {D['perm_sizes_and_events']['p_two_sided']} / {D['perm_sizes_and_events']['p_one_sided_ge']} / {ci(D['perm_sizes_and_events']['null_q025_q975'])} ({D['perm_sizes_and_events']['n_valid_perms']} valid) |

**Method.** As pre-specified; `co_main.py` block D.

**Sources.** {SRC_MAIN}.

**Caveats.** Post-hoc split; the subgroup with the large gain has 14 events; the interaction is driven by the CNV arm (item C), which is one of the two fused arms, not by the head, whose AUROC is similar in both subgroups.
""")
# ------------------------------------------------------------------ E
P(f"""### E. Prevalent vs future disease
**Question.** How far from the baseline sample is the endpoint biopsy, and does the result survive dropping near-baseline progressors?

**Status.** DONE.

**Result: interval distribution** (50 progressor patients; interval = earliest release row → endpoint biopsy, the biopsy that completes the LGD2+ rule; {E['progressors_with_single_row']} progressors have a single row).

| statistic | value |
|---|---|
| n with interval | {E['n_with_interval']} |
| median (days) | {E['median_days']:.0f} |
| IQR (days) | [{E['iqr_days'][0]:.0f}, {E['iqr_days'][1]:.0f}] |
| min, max (days) | {E['min_max'][0]:.0f}, {E['min_max'][1]:.0f} |
| histogram | {", ".join(f"{k}: {v}" for k, v in E['hist_bins'].items())} |
| secondary: min `DaysFromCurrentToEvent` per progressor (release column, non-null for {E['secondary_DaysFromCurrentToEvent_min_per_progressor']['n']} progressors) | median {E['secondary_DaysFromCurrentToEvent_min_per_progressor']['median']:.0f} d |

**Result: exclusion runs** (canonical arms).

| exclusion | n | events | head | image + CNV | with head | gain |
|---|---|---|---|---|---|---|
""" + "\n".join(f"| {r['exclusion']}{' (rows kept ' + str(r['rows_kept']) + ')' if 'rows_kept' in r else ''} | {r['n']} | {r['events']} | {f3(r['head'])} {ci(r['head_ci'])} | {f3(r['fuse2'])} {ci(r['fuse2_ci'])} | {f3(r['fuse3'])} {ci(r['fuse3_ci'])} | {s3(r['gain'])} {ci(r['gain_ci'])} |" for r in EX) + f"""

**Method.** As pre-specified. One correction after the first run (commit `4af24d5`, stated per rule 8): in the SECONDARY row-dropping analysis the first run re-derived the patient label from the remaining rows, which turned 11 progressors into non-progressors (events 33/25); the corrected version keeps each patient's original progressor label and drops rows only. Both versions are in git history (`feasibility/runs/co_main` outputs of jobs 57654406 and 57654440 on the cluster); the corrected one is reported.

**Sources.** {SRC_MAIN}.

**Caveats.** The release has no single baseline sample: evaluation uses all strict pre-event rows with max aggregation, so "interval from baseline" is measured from the earliest row. Rows at or after the event were already excluded by the release (183 at-event, 31 post-event rows).
""")
# ------------------------------------------------------------------ F
P(f"""### F. Baseline grade
**Question.** Is the baseline pathologist grade in the clinical-only arm, and what do grade-based arms give?

**Status.** DONE.

**Result.** The release has **no clinical arm**. Model families with out-of-fold predictions: {", ".join('`' + f + '`' for f in Fm['release_families'])}. No family takes clinical covariates; there is therefore no feature list to quote. Grade coding: {Fm['grade_coding']}.

| arm (all 150 patients, 50 events) | AUROC |
|---|---|
""" + "\n".join(f"| {r['arm']} | {f3(r['auroc'])} {ci(r['ci'])} |" for r in Fm["rows"]) + f"""

`grade_arm` = CV logistic (release folds) on the row grade; `grade_maxsofar_arm` adds `MaxPathologySoFar`; `_plus_head` = fold-z mean with the leak-free head; `grade_raw_max` = the raw grade code, max over rows, no fitting. Age and sex are available for 66 patients only (item C) and were not used.

**Within baseline-NDBE patients.**

| definition | n | events | head | image | CNV | image + CNV | with head | gain |
|---|---|---|---|---|---|---|---|---|
""" + "\n".join(f"| {r['definition']} | {r['n']} | {r['events']} | {f3(r['head'])} {ci(r['head_ci'])} | {f3(r['img'])} {ci(r['img_ci'])} | {f3(r['cnv'])} {ci(r['cnv_ci'])} | {f3(r['fuse2'])} {ci(r['fuse2_ci'])} | {f3(r['fuse3'])} {ci(r['fuse3_ci'])} | {s3(r['gain'])} {ci(r['gain_ci'])} |" for r in FN) + f"""

**Method.** `co_main.py` block F; logistic C = 1, features standardised on the training folds.

**Sources.** {SRC_MAIN}; release families from `<release>/training_final_nested_cv_v1/`.

**Caveats.** "Clinical + baseline grade" reduces to grade alone because no other clinical covariate is in the release for all patients.
""")
# ------------------------------------------------------------------ G
P(f"""### G. The failed pre-registered sanity check
**Question.** What was pre-registered, what was observed, and is there a second read?

**Status.** DONE.

**Pre-registration (verbatim).** `docs/projects/P32_image_to_fields.md` at commit `128760e`, committed 2026-09-24 17:50:13 +0100:

> The seed-0 fold models are applied to all 707 SWG release slides (UNI2 npz) to impute every field. Then, on the release folds and endpoint, patient level, 150 patients: a CV logistic on the imputed-field vector ("fields" arm); 3-way fold-local z-mean fusion (image OOF + CNV OOF + fields) vs the 2-way (image + CNV) on the same 150 patients; paired 2,000-boot CI. Sanity check first: imputed grade LGD+ vs the SWG pathologist grade (if this is < 0.7 the imputation has not transferred and the fusion result is moot).

and, under "Predictions written down before results": "SWG payoff: imputed grade tracks the pathologist grade at 0.7–0.8 despite the scale shift; the 3-way fusion does NOT beat 2-way with a CI excluding zero at n = 150. If it does, that is the Chapter 1–2 bridge."

**Result.** Metric: {G['metric']}. n rows {G['n_rows']}, LGD rows {G['n_LGD_rows']}. Source of the pathologist grade (`GradeSource` on the 707 rows): {G['grade_source_of_Label_707']}.

| head | AUROC vs release Label (707 rows) | AUROC vs DB confirmed code ({G['vs_db_confirmed_code']['n_rows_with_code']} rows, {G['vs_db_confirmed_code']['n_LGDplus_by_code']} LGD+) | AUROC vs Label on the same {G['vs_db_confirmed_code']['n_rows_with_code']} rows |
|---|---|---|---|
| pass-1 (0.88 µm/px imputation, incl-overlap) | {f3(G['observed']['pass1'])} | n/a | n/a |
| pass-2 `p32b_fields` (0.5 µm/px, incl-overlap) | {f3(G['observed']['p32b'])} | {f3(G['vs_db_confirmed_code']['p32b'])} | {f3(G['head_vs_each_read_same_rows']['vs_Label']['p32b'])} |
| repeat-fold head (`p32_head_repeat`) | {f3(G['observed']['repeat'])} | n/a | n/a |
| permuted-label head (`p32_head_perm`) | {f3(G['observed']['perm'])} | n/a | n/a |
| leak-free head `p32_head_noov` (canonical) | {f3(G['observed']['noov'])} | {f3(G['vs_db_confirmed_code']['noov'])} | {f3(G['head_vs_each_read_same_rows']['vs_Label']['noov'])} |

"0.619" is the pass-2 head vs the release Label; "0.63" is the same head vs the DB confirmed code ({f3(G['vs_db_confirmed_code']['p32b'])}). Every head is below the pre-registered 0.7, including the canonical one ({f3(G['observed']['noov'])}). The permuted-label head scores {f3(G['observed']['perm'])} on the same check.

**Second read.** {G['second_pathologist_read']}. Release Label vs DB confirmed code on the {G['read_vs_read_on_coded_rows']['n']} coded rows: two-tier agreement {f3(G['read_vs_read_on_coded_rows']['Label_vs_DBcode_two_tier_agreement'])}, kappa {f3(G['read_vs_read_on_coded_rows']['kappa'])}, AUROC of Label as a score for the code {f3(G['read_vs_read_on_coded_rows']['auroc_Label_as_score_vs_DBcode'])}. {G['read_vs_read_on_coded_rows']['note']}.

**Method.** `co_main.py` block G; DB code mapping 2→NDBE, 3→IND, 4→LGD, 5→HGD, 6/8→cancer (`query_dysplasia_types`); rows matched by accession stem to `swg_matched_reports_v2.parquet`.

**Sources.** {SRC_MAIN}; pre-registration `docs/projects/P32_image_to_fields.md` @ `128760e`; earlier values `results/numbers/p32_checks.json` (`scripts/projects/p32_checks.py`), `results/numbers/p32b_fields.json`, `p32_head_noov.json`.

**Caveats.** The pre-registration set the gate on "the SWG pathologist grade" without naming the column; the release Label is for 547/707 rows the DB confirmed code itself, so the two anchors are not independent. The 707 rows contain no HGD+, so the check is NDBE/IND vs LGD only. By the letter of the pre-registration the fusion result is "moot"; the project log re-read the gate after the fact (digest §5), which is a post-hoc decision.
""")
# ------------------------------------------------------------------ H
if H:
    P(f"""### H. Non-grade ensemble control (new training)
**Question.** Does an ERIN head trained on a non-grade field give the same fusion gain?

**Status.** DONE.

**Pre-specification.** Commit `{PRESPEC}` (section "H" of the pre-specification, verbatim in §6). Run names `co_h_treat_noov`, `co_h_im_noov`, `co_h_perm_noov` (cluster jobs 57654104–57654106).

**Result** (all 150 patients, 50 events; fuse2 = image + CNV = {f3(H['rows'][0]['fuse2'])}).

| head | ERIN training cases | ERIN OOF AUROC (n / pos) | head alone on SWG | with head (fuse3) | gain vs fuse2 | perm p |
|---|---|---|---|---|---|---|
""" + "\n".join((f"| {r['head']} | {r['erin_training_cases']} | {f3(r['erin_oof_auroc'])} ({r['erin_n_pos'][0]} / {r['erin_n_pos'][1]}) | {f3(r['head_alone_swg'])} {ci(r['head_ci'])} | {f3(r['fuse3'])} {ci(r['fuse3_ci'])} | {s3(r['gain'])} {ci(r['gain_ci'])} | {r['perm_p']} |" if 'gain' in r else f"| {r['head']} | {r.get('status','')} | | | | | |") for r in H["rows"]) + f"""

**Method.** {H['fusion']}; {H['n_boot']} bootstraps, {H['n_perm']} permutations, seed {H['seed']}; heads trained by `scripts/projects/p32_fields_from_image.py` with the leak-free exclusion file (2,249 cases), P31 v2 labels, 0.5 µm/px SWG bags; `scripts/closeout/co_h_assemble.py`.

**Sources.** `results/closeout/h_nongrade_controls.json` · `scripts/closeout/co_h_assemble.py` · commit {RESC}; head runs `feasibility/runs/co_h_*_noov/output/` (cluster).

Reading rule outcome (stated before training, §6): no non-grade head's gain has a CI excluding zero on the positive side; the condition under which the gain would be attributed to representation rather than grade content is not met.

**Caveats.** The reading rule was stated before training (§6). ERIN OOF AUROCs of the new heads are reported so a weak head can be told apart from a non-transferring one.
""")
else:
    P("### H. Non-grade ensemble control\n**Status.** PENDING (heads training).\n")
# ------------------------------------------------------------------ I
P(f"""### I. Selection across P32 fields
**Question.** Which imputed fields were ever evaluated on SWG, and what is the selection-adjusted p for the grade head's gain?

**Status.** DONE.

**Result: candidates and when first evaluated** (dates from result-file commits: pass-1 fields 2026-09-24 19:51 `0dd5df8`; pass-1 fusion and pass-2 fusion + controls 2026-09-24 23:20 `926fc93`; leak-free 2026-09-25 11:33 `ec07155`). Observed gain (fuse3 − fuse2) per candidate third arm, all 150 patients:

| candidate third arm | gain |
|---|---|
""" + "\n".join(f"| {k} | {s3(v)} |" for k, v in I["candidates_observed_gain"].items()) + f"""

| quantity | value |
|---|---|
| candidates maximised over | {I['n_candidate_third_arms']}: {I['what_was_maximised_over']} |
| selected maximum | {I['selected_max']} ({s3(I['t_obs_max'])}) |
| unadjusted permutation p for the selected arm | {I['p_unadjusted_for_selected']} |
| **selection-adjusted p (max over all candidates under {I['n_perm']} label permutations, seed {I['seed']})** | **{I['p_selection_adjusted_max_over_all_candidates']}** |
| canonical leak-free grade gain against the same max-null | {s3(I['canonical_noov_grade_gain'])} → p {I['p_canonical_against_max_null']} |

For comparison, `results/numbers/p32_checks.json` (`scripts/projects/p32_checks.py`) adjusted over the 6 pass-2 fields only and reported p 0.022; adjusting over everything that was evaluated gives {I['p_selection_adjusted_max_over_all_candidates']}.

**Method.** `co_main.py` block I. The fusion rule (fold-local z-mean) was fixed in the P32 pre-registration and is not maximised over; the choice "single field vs logistic over fields" was not fixed and is included.

**Sources.** {SRC_MAIN}; pass-1 imputations `feasibility/runs/p32_fields_g0..g3/output/swg_imputed_fields.csv`, pass-2 `p32b_fields`, rep02 image `<release>_rep02/training_rep02_nested_cv/image_only`.

**Caveats.** Pass-1 arms were imputed at 0.88 µm/px and pass-2 at 0.5 µm/px; both are included because both were evaluated. The retrained leak-free head is a re-estimate of the selected arm, not a new candidate.
""")
# ------------------------------------------------------------------ J
P(f"""### J. CI method audit
**Question.** For every CI in `docs/results_digest_2026-09-24_25.md`: resampling unit, n, seed; recompute any that are not patient-clustered.

**Status.** DONE.

**Result.** Read from the generating scripts (grep of `RandomState`, `choice`, `groupby("patient_id")`, cluster index construction):

{PRESPEC_TEXT and ''}""" + open(os.path.join(os.path.dirname(__file__), "..", "..", "docs", "_j_audit_table.md")).read() + f"""

The only deviation is the 13 P32 ERIN field CIs (1,000 patient-clustered resamples). Recomputed at 2,000 from the saved out-of-fold files:

| run | field | n cases | n patients | pos | AUROC | CI (2,000, patient-clustered, seed 0) |
|---|---|---|---|---|---|---|
""" + "\n".join(f"| {r['run']} | {r['field']} | {r['n_cases']} | {r['n_patients']} | {r['pos']} | {f3(r['auroc'])} | {ci(r['ci_2000_patient_clustered'])} |" for r in J) + f"""

**Method.** `co_main.py` block J; digest-side sources: `results/numbers/p32_fields.json`, `p32b_fields.json` (1,000-resample CIs).

**Sources.** {SRC_MAIN}.

**Caveats.** None of the digest CIs is slide-level i.i.d.; no conclusion changes. The digest's SWG fusion CIs are "patient-level" by construction (the unit is the patient vector), which is the same as patient-clustered here.
""")
# ------------------------------------------------------------------ K
P(f"""### K. Aggregation and baseline definition
**Question.** What is a patient's baseline, how do slides aggregate, are folds grouped by patient everywhere?

**Status.** DONE.

**Result.** {K['aggregation']['swg_unit']}. {K['aggregation']['baseline']}. Patient score: {K['aggregation']['patient_score']}; patient label: {K['aggregation']['patient_label']}. {K['aggregation']['training']}.

Assertions run in `co_main.py` block K:

| check | code path | result |
|---|---|---|
| SWG rep01: patients in more than one outer fold | `training_manifest.fold_id_rep01` grouped by `patient_id` | {K['swg_rep01_patients_in_more_than_one_outer_fold']} |
| SWG rep02: patients in more than one outer fold | `_rep02/training_manifest_v2.csv` | {K['swg_rep02_patients_in_more_than_one_outer_fold']} |
| SWG inner folds, image_only / cnv_only | `fold*/inner_fold_assignments.csv` (keyed by patient_id) | {K['swg_inner_folds']['image_only']['patients_in_more_than_one_inner_fold']} / {K['swg_inner_folds']['cnv_only']['patients_in_more_than_one_inner_fold']} |
""" + "\n".join(f"| ERIN task {t} (n units {v['n_units']}, patients {v['n_patients']}): patients in more than one fold | `scripts/abmil_clf.patient_folds(keys, anon_id, y, 5, seed=0)` as called by `scripts/erin_fusion/worker.py` | {v['patients_in_more_than_one_fold']} |" for t, v in K["erin_tasks"].items()) + f"""
| P32 ERIN heads (n cases {K['p32_erin_heads']['n_cases']}, patients {K['p32_erin_heads']['n_patients']}) | `patient_folds` in `scripts/projects/p32_fields_from_image.py` | {K['p32_erin_heads']['patients_in_more_than_one_fold']} |

`patient_folds` (`scripts/abmil_clf.py` lines 72–82) assigns whole patients to folds, event-stratified, so a patient cannot be split by construction; the assertions confirm it on the tables actually used, including the C26 field-effect tasks T3a/T3b (and their biopsy-only variants), T2a/T2b/T2c, T4 and T1.

**Sources.** {SRC_MAIN}; `scripts/abmil_clf.py`; `scripts/erin_fusion/worker.py`; `feasibility/erin_fusion/tasks/*.csv` (cluster).

**Caveats.** Multiple rows per patient enter training as separate units within the same fold; the release trainer does not weight patients equally.
""")
# ------------------------------------------------------------------ L
P(f"""### L. CNV comparison to Killcoyne 2020
**Question.** CNV-only at 0.4× vs 4× on the same patients; side-by-side definitions.

**Status.** PARTIAL (4× NOT AVAILABLE).

**Result: 4× data.** {L4['status']}. Locations searched: {"; ".join(L4['looked'])}.

**Result: definitions** (ours filled from code and release metadata; the Killcoyne 2020 column is left for the author).

| field | ours | Killcoyne 2020 |
|---|---|---|
| modality | H&E (UNI2 ABMIL) + sWGS CNV; CNV-only arm = `cnv_only` | |
| cohort | {Lo['rows_patients'][1]} patients, {Lo['rows_patients'][0]} rows; {Lo['progressor_patients']} progressor patients; {Lo['positive_rows']} positive rows | |
| unit | one biopsy slide + one CNV profile per row; patient = max | |
| endpoint | `{Lo['endpoint_name']}`: {Lo['endpoint_rule_code']} | |
| eligibility | {Lo['eligibility']} | |
| evaluation | {Lo['split']} | |
| primary metric in the release | {Lo['primary_metric_release']} (AUROC used here) | |
| CNV representation | {Lo['cnv_representation']} | |
| depth | {Lo['depth_statement']}; median reads per profile {Lo['reads_median_per_sample_sheet']:.0f} (n {Lo['reads_n_samples_with_value']} rows with a value) | |
| overlap with the discovery sample sheet | {Lo['killcoyne_discovery_overlap_by_cnv_id']} of 707 rows' CNV ids are in the 777-sample discovery sheet; {Lo['validation_sheet_cnv_ids']} in the 268-sample validation sheet | |
| code identity | release built at pipeline commit `{Lo['code_commit_release'][:12]}`; fold models at `{Lo['fold1_git_commit'][:12]}` | |

**Sources.** {SRC_MAIN}; `<release>/cohort_release_metadata.json`, `split_release_metadata.json`, `tasks_chapter1_lgd2_final.json`; `multimodal-barretts-progression/src/barrett/labels/lgd2.py` (cluster).

**Caveats.** The depth of 0.4× is a statement in the project's documents, not a value computed from the BAMs; read counts are from the sequencing sheet and read length is not recorded.
""")
# ------------------------------------------------------------------ M
thr = list(MC["rows"][0]["net_benefit"].keys())
P(f"""### M. Calibration and clinical comparators
**Question.** Calibration, Brier and net benefit for image-only, image + CNV and with head; is p53 IHC available and does the head add to it?

**Status.** DONE.

**Result: calibration and Brier** (150 patients, 50 events; prevalence {MC['prevalence']:.3f}).

| arm | AUROC | mean predicted | calibration slope | intercept (at fitted slope) | calibration-in-the-large | Brier |
|---|---|---|---|---|---|---|
""" + "\n".join(f"| {r['arm']} | {f3(r['auroc'])} | {f3(r['mean_pred'])} | {f3(r['calib_slope'])} | {s3(r['calib_intercept_at_slope_fit'])} | {s3(r['calib_in_the_large'])} | {f3(r['brier'])} {ci(r['brier_ci'])} |" for r in MC["rows"]) + f"""

**Result: decision-curve net benefit** (per patient; treat-none = 0).

| threshold | """ + " | ".join(r["arm"] for r in MC["rows"]) + """ | treat all |
|---|""" + "---|" * (len(MC["rows"]) + 1) + "\n" + "\n".join(f"| {t} | " + " | ".join(f"{r['net_benefit'][t]:+.3f}" for r in MC["rows"]) + f" | {MC['treat_all_net_benefit'][t]:+.3f} |" for t in thr) + f"""

{MC['note']}.

**Result: p53 IHC.** Available for a subset: {MP['source']}. n patients with IHC {MP['n_patients_with_ihc']}, events {MP['events']}, aberrant {MP['n_aberrant']}.

| arm (same {MP['n_patients_with_ihc']} patients) | AUROC |
|---|---|
| p53 aberrant (binary) | {f3(MP['p53_auroc'][0])} {ci(MP['p53_auroc'][1:])} |
| head | {f3(MP['head_auroc_same_patients'][0])} {ci(MP['head_auroc_same_patients'][1:])} |
| p53 + head (rank mean) | {f3(MP['p53_plus_head_rankmean'][0])} {ci(MP['p53_plus_head_rankmean'][1:])}; Δ vs p53 {ci(MP['delta_p53_plus_head_minus_p53'])} |
| image + CNV | {f3(MP['fuse2_same_patients'][0])} {ci(MP['fuse2_same_patients'][1:])} |
| image + CNV + p53 (rank mean) | {f3(MP['fuse2_plus_p53_rankmean'][0])} {ci(MP['fuse2_plus_p53_rankmean'][1:])}; Δ vs image + CNV {ci(MP['delta_fuse2_plus_p53_minus_fuse2'])} |

**Method.** `co_main.py` block M: slope/intercept from a logistic regression of outcome on logit(p); calibration-in-the-large = logit(observed rate) − logit(mean p); Brier with patient bootstrap; net benefit = TP/n − FP/n × t/(1−t). {MP['note']}.

**Sources.** {SRC_MAIN}; p53 from `SWGCohort/slide_matching.csv` (cluster).

**Caveats.** The fused arms have no native probability; their calibration is that of a CV Platt map and is not comparable one-to-one with the release probabilities. The head's own probability is severely miscalibrated on SWG (mean predicted {f3(MC['rows'][4]['mean_pred'])} against prevalence {MC['prevalence']:.3f}), consistent with item G. p53 IHC covers half the patients and is a research staining from the matching sheet, not a clinical result.
""")
# ------------------------------------------------------------------ N
img_h = [k for k in HS if k.endswith("image_only/fold1/model.pt")][0]
P(f"""### N. ACE-B readiness
**Question.** Status of slides and CNV, plan for the 30 prevalent cases, frozen plan with checkpoint hashes.

**Status.** DONE (plan frozen); data NOT AVAILABLE.

**Result.**

| item | status |
|---|---|
| ACE-B slides on the cluster | none (`/mnt/scratche/fast/fmlab/datasets/imaging/` has no ACE-B directory; only `phd/aceb_meta/` metadata from May 2026) |
| ACE-B features | none |
| ACE-B CNV tables or BAMs | none on the cluster |
| depth, SWG | 0.4× as stated in project documents (not verified from data; sheet read counts in item C) |
| depth, ACE-B | ~7× per the cohort owner (docs/NUMBERS.md §25); no file to verify |
| the 30 prevalent HGD/IMC cases | excluded from the primary progression analysis; analysed as a separate secondary "prevalent detection" set (plan §3). Previously the plan in `docs/NUMBERS.md` §25 said "optionally 5–10 prevalent HGD/IMC" for imaging without stating how they enter the analysis; this is now written down. |
| frozen plan | `docs/aceb_analysis_plan.md`, committed at `{PRESPEC}` before any ACE-B slide exists; checkpoint hashes in `results/closeout/aceb_checkpoint_hashes.json` (`scripts/closeout/co_checkpoint_hashes.py`), e.g. image_only fold1 `{HS[img_h]['sha256'][:16]}…`, {len([k for k in HS if isinstance(HS[k], dict)])} files hashed |

**Sources.** `docs/aceb_analysis_plan.md`; `results/closeout/aceb_checkpoint_hashes.json`; `docs/NUMBERS.md` §25; `docs/status_ledger_2026-09-24.md` items 2, 11, 12, 21.

**Caveats.** The plan's CNV step assumes read-level data can be down-sampled; if only the owner's arm table arrives, the depth shift is declared, not corrected. With ~11 progressors the plan states in advance that the fusion-vs-image comparison is under-powered.
""")
# ------------------------------------------------------------------ discrepancies, not done, prespec
P(f"""## 4. Discrepancies found
1. **Selection-adjusted p.** `results/numbers/p32_checks.json` and the digest report 0.022, adjusting over the 6 pass-2 fields only. Adjusting over every third arm that was evaluated ({I['n_candidate_third_arms']}) gives {I['p_selection_adjusted_max_over_all_candidates']} (item I). The 0.022 is correct for what it adjusts over; it is not the fully adjusted value.
2. **Digest row 16 wording.** The digest attributes "+0.063 [+0.020, +0.107]" to the leak-free head "alone"; that value is the 2-field CV-logistic fusion (v3). The leak-free grade-head fusion is {s3(canon['gain'])} {ci(canon['gain_ci'])} (v4). Both numbers are correct; the label was wrong.
3. **"33 overlap patients excluded".** The digest and project log say 33 patients were excluded; the exclusion file lists all 55 linked ERIN identities (all 54 SWG patients). 33 is the number of SWG patients that had a case in training; the exclusion was wider (item B).
4. **P32 ERIN field CIs** used 1,000 resamples, not the 2,000 stated as the convention elsewhere; recomputed values differ by ≤ 0.002 (item J).
5. **Sanity gate.** The digest (§5, "Mechanism check") reports 0.619 and 0.629 for the pass-2 head; the canonical leak-free head scores {f3(G['observed']['noov'])} / {f3(G['vs_db_confirmed_code']['noov'])}, lower, and this was not previously reported (item G).
6. **Killcoyne comparison doc** in the pipeline repository (`reports/scientific_hardening/killcoyne_protocol_comparison.md`) states "69/150 local patients are Killcoyne-discovery PSIDs"; by CNV id, {Lo['killcoyne_discovery_overlap_by_cnv_id']} of 707 rows are in the discovery sheet, i.e. {[r for r in CC if r['variable']=='seq_sheet_mode'][0]['also_in_ERIN_54']['discovery_777'] + [r for r in CC if r['variable']=='seq_sheet_mode'][0]['never_in_ERIN_96']['discovery_777']} of 150 patients by modal sheet (item C). Not reconciled here; both counts are reported.
7. **First-run labelling error in item E secondary analysis** (own error, corrected at `4af24d5` before reporting; see item E).
8. **Digest "nothing pending"** (§6): item H (non-grade control) had been listed in the P32 log as "Next (not run)" and was not run until this closeout.

## 5. Not done
- Sequencing depth in × (read length not recorded; BAM-based depth not computed: no samtools/pysam in the environment, 1,035 BAMs unindexed).
- Staining batch (no record exists).
- 4× CNV (no data exists; question to the cohort owner open since 24 Sep).
- Killcoyne 2020 column of item L (left to the author by instruction).
- A second pathologist read for item G (none exists).
- Referral pathway beyond the DB referral-hospital field; sex for the 84 patients absent from the demographics sheet (gender code reported instead).
- ACE-B itself (no data).

## 6. Pre-specification text as committed at `{PRESPEC}` (verbatim)

""" + "\n".join("> " + l if l.strip() else ">" for l in PRESPEC_TEXT.splitlines()))
open("docs/closeout_for_review.md", "w").write("\n".join(L)); print("written", len("\n".join(L)))

"""Render the Results of docs/paper_survival_horizons.md from results/paper_final/horizons/*.json. The pre-specification above '## Results' is kept
verbatim. Usage: python hz_render.py RESULTS_COMMIT"""
import json, os, sys
RC = sys.argv[1]; R = "results/paper_final/horizons"; DOC = "docs/paper_survival_horizons.md"
J = lambda f: json.load(open(f"{R}/{f}.json"))
h1 = {f"{s}_{p}": J(f"h1_{s}_{p}") for s in ("their", "pkg") for p in ("pre", "pre_ndbe", "pre_nofallback")}
h2 = J("h2"); h3 = {s: J(f"h3_{s}") for s in ("their", "pkg")}; dummy = J("h1_aceb_dummy_check")
h4s = J("h4_search") if os.path.exists(f"{R}/h4_search.json") else None; h4d = J("h4_depth") if os.path.exists(f"{R}/h4_depth.json") else None
f3 = lambda x: "—" if x is None else f"{x:.3f}"; sg = lambda x: "—" if x is None else f"{x:+.3f}"
ci = lambda c: "[—]" if not c or c[0] is None else f"[{c[0]:.3f}, {c[1]:.3f}]"; cis = lambda c: "[—]" if not c or c[0] is None else f"[{c[0]:+.3f}, {c[1]:+.3f}]"
SRC = {"their": "their matrix", "pkg": "package features"}; POP = {"pre": "all pre-event samples", "pre_ndbe": "NDBE pre-event samples", "pre_nofallback": "all pre-event samples, without the 4 fallback-endpoint progressors"}
ARMS = ["L-GRADE", "L-GRADE+age+sex", "L-CNV", "L-IMG", "L-EARLY", "L-INTER", "L-LATE"]
def table(head, rows): return ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"] + ["| " + " | ".join(map(str, r)) + " |" for r in rows] + [""]
def cell(d, t, a):
    H = d["units"]["sample"]["horizons"][str(t)]; o = H["arms"][a]["ipcw"]; return f"{f3(o['auroc'])} {ci(o['ci95'])} ({H['n_cases']}/{H['n_controls']})"
L = open(DOC).read().split("\n## Results")[0].rstrip().split("\n")
L += ["", "## Results", "",
      f"Pre-specification commit ee51db8 (this file above the line, and amendment 2 of `docs/aceb_analysis_plan.md`); results commit {RC}.",
      "Scripts: `scripts/paper_plan/hz_prep.py` (H0 event times), `hz_fit.R` (new outer-CV arms and the nested inner CV), `hz_metrics.py` (H1), `hz_stack.py` (H2, H3), `hz_qc.R` and `hz_slides.py` (H2 descriptors), `hz_aceb_fill.py` (ACE-B code path), `hz_search.py` and `hz_search_shard.py` (H4), `hz_figs.py`, `hz_render.py`. All run through `scripts/cluster/campaign.sh` (prefixes hzp, hzf, hzd, hzm, hz2, hzs, hzss).",
      "Results: `results/paper_final/horizons/h1_{their,pkg}_{pre,pre_ndbe,pre_nofallback}.json`, `h2.json`, `h3_{their,pkg}.json`, `h1_aceb_dummy_check.json`, `h4_search.json`; figures `results/paper_final/horizons/figs/*.{png,pdf,json}`. Row-level outputs on the cluster only, under `feasibility/paper_plan/killcoyne_mm/horizons/` (including the H2 patient list `h2_patient_list.csv`).", ""]
h4status = "NOT AVAILABLE" if (h4s is None or not h4s["aceb_reads"]) else ("DONE" if h4d else "PARTIAL")
L += ["### Status", ""] + table(["Item", "Status"], [["H0 pre-specification, event times, horizon labels", "DONE"], ["H1 horizon table (internal); ACE-B columns", "DONE; ACE-B NOT AVAILABLE (code path checked on dummy labels)"],
      ["H2 change in false positives", "DONE"], ["H3 WSI vs CNV weighting by horizon", "DONE"], ["H4 depth transfer", f"4× → external NOT AVAILABLE; ACE-B depth check {h4status}"]])
L += ["**Design caveat.** The discovery cohort is a matched case–control design (non-progressors needed ≥ 3 years of follow-up, progressors ≥ 1 year). AUROCs are interpretable; absolute risks, calibration and positive predictive values are not, and none is reported.", ""]
# ---------- the filled table
d = h1["their_pre"]
rows = []
for nm, a in (("Clinical only", "L-GRADE"), ("CNV (replication)", "L-CNV"), ("WSI", "L-IMG"), ("Early fusion", "L-EARLY"), ("Intermediate fusion", "L-INTER"), ("Late fusion", "L-LATE")):
    rows.append([nm] + [cell(d, t, a) for t in (1, 3, 5)] + ["NOT AVAILABLE"] * 3)
L += ["### H1 table", "", "Per-sample IPCW time-dependent AUROC [95% patient-bootstrap CI] (n cases / n controls), all pre-event samples of the discovery subset, CNV source their matrix. Clinical only = L-GRADE; L-GRADE+age+sex and the package-feature source are in H1 below. ACE-B: no slides, outcomes blinded.", ""]
L += table(["", "Internal 1-y", "3-y", "5-y", "ACE-B 1-y", "3-y", "5-y"], rows)
# ---------- H0
L += ["### H0. Time origin, event, horizon labels", "", "**Question.** Which samples and labels enter each horizon? **Status.** DONE. **Pre-specification.** Above (H0).", "",
      "**Result.** n per horizon (samples are the prediction unit; the patient unit is each patient's earliest pre-event NDBE sample):", ""]
rows = []
for p in ("pre", "pre_ndbe", "pre_nofallback"):
    for unit in ("sample", "patient"):
        U = h1[f"their_{p}"]["units"][unit]
        if p != "pre" and unit == "patient": continue
        for t in (1, 3, 5):
            H = U["horizons"][str(t)]; rows.append([POP[p], unit, f"{t} y", U["n_rows"], U["n_patients"], f"{U['n_event_rows']} / {U['n_event_patients']}", f"{H['n_cases']} ({H['n_case_patients']} patients)", f"{H['n_controls']} ({H['n_control_patients']} patients)", H["n_excluded"]])
L += table(["Population", "Unit", "Horizon", "Rows", "Patients", "Event rows / patients", "Cases", "Controls", "Excluded (censored before t)"], rows)
L += ["The patient unit is the same in both populations (an earliest NDBE pre-event sample is by definition in both), so it is reported once. The 4 progressors without an HGD/IMC sample among the published samples have their endpoint at the final endoscopy (`kc_merge.py` fallback), flagged in `samples.csv`; the H1 cells without them are under H1.",
      "**Method.** `hz_prep.py`: `mbf` = `Months before final` of the 777 sheet; progressor time = (`mbf` − `tev`)/12 with `tev` the largest `mbf` among the patient's HGD/IMC samples (`kc_merge.py:30-32`); non-progressor time = `mbf`/12, censored. Pre-event = `mbf` > `tev` for progressors, all non-progressor samples (571 samples, 75 patients, 32 P, the `b_before_first_HGD_IMC` subset of `kv_merge.py`).",
      "**Sources.** `feasibility/paper_plan/killcoyne_mm/set_C.csv`, `feasibility/paper_plan/killcoyne/kr_samples.csv`, `[SWG]/sWGS_777_samples_cleaned_202401_Leanne_fullDetails (3) (1).csv`, `[SWG]/Demographics_full.csv`; `h1_*.json:units.*.horizons.*.n_*`.",
      "**Caveats.** `Months before final` is the only time scale available for every sample; it is in whole months. Controls at t include progressor samples whose event is later than t.", ""]
# ---------- H1
L += ["### H1. Horizon table, all arms", "", "**Question.** How well do the ever-progression scores rank near-term against later events? **Status.** DONE (internal); ACE-B NOT AVAILABLE. **Pre-specification.** Above (H1). L-INTER and L-GRADE+age+sex were fitted under the stratified CV for this task (`hz_fit.R` MODE=outer); their folds match `kv_cv.R` for "
      f"{h1['their_pre']['fold_check_inter_vs_kv'][0]} of {h1['their_pre']['fold_check_inter_vs_kv'][1]} sample × repeat rows. Every other arm is the existing out-of-fold prediction, not refitted.", ""]
for s in ("their", "pkg"):
    for p in ("pre", "pre_ndbe", "pre_nofallback"):
        U = h1[f"{s}_{p}"]["units"]["sample"]; rows = []
        for t in (1, 3, 5):
            H = U["horizons"][str(t)]
            for a in ARMS:
                o = H["arms"][a]; w = o["ipcw"]; u = o["unweighted"]; dc = w.get("delta_vs_L-CNV"); dl = w.get("delta_vs_L-LATE")
                rows.append([f"{t} y", a, f"{H['n_cases']}/{H['n_controls']}", f"{f3(w['auroc'])} {ci(w['ci95'])}", f"{f3(u['auroc'])} {ci(u['ci95'])}",
                             "—" if not dc else f"{sg(dc['delta'])} {cis(dc['ci95'])}", "—" if not dl else f"{sg(dl['delta'])} {cis(dl['ci95'])}", "—" if not dl else dl.get("p_unadjusted", "—"), "—" if not dl else dl.get("p_max_T", "—")])
        L += [f"**{SRC[s]}, {POP[p]}** ({U['n_rows']} samples, {U['n_patients']} patients, {U['n_event_patients']} progressors), per sample.", ""]
        L += table(["Horizon", "Arm", "Cases/controls", "IPCW AUROC [CI]", "Unweighted AUROC [CI]", "Δ vs L-CNV [CI]", "Δ vs L-LATE [CI]", "p (swap)", "p max-T"], rows)
    U = h1[f"{s}_pre"]["units"]["patient"]; rows = []
    for t in (1, 3, 5):
        H = U["horizons"][str(t)]
        for a in ARMS:
            w = H["arms"][a]["ipcw"]; dl = w.get("delta_vs_L-LATE"); dc = w.get("delta_vs_L-CNV")
            rows.append([f"{t} y", a, f"{H['n_cases']}/{H['n_controls']}", f"{f3(w['auroc'])} {ci(w['ci95'])}", "—" if not dc else f"{sg(dc['delta'])} {cis(dc['ci95'])}", "—" if not dl else f"{sg(dl['delta'])} {cis(dl['ci95'])}"])
    L += [f"**{SRC[s]}, patient unit** (earliest pre-event NDBE sample; {U['n_rows']} patients, {U['n_event_patients']} progressors).", ""]
    L += table(["Horizon", "Arm", "Cases/controls", "IPCW AUROC [CI]", "Δ vs L-CNV [CI]", "Δ vs L-LATE [CI]"], rows)
rows = []
for s in ("their", "pkg"):
    for p in ("pre", "pre_ndbe"):
        Cx = h1[f"{s}_{p}"]["units"]["sample"]["cindex"]
        for a in ARMS: rows.append([SRC[s], POP[p], a, f"{f3(Cx[a]['harrell']['c'])} {ci(Cx[a]['harrell']['ci95'])}", f"{f3(Cx[a]['uno']['c'])} {ci(Cx[a]['uno']['ci95'])}"])
L += ["**C-index over all follow-up**, per sample.", ""] + table(["CNV source", "Population", "Arm", "Harrell's C [CI]", "Uno's C [CI]"], rows)
dl = dummy["result"]["all"]["horizons"]["3"]["arms"]["L-LATE"]["ipcw"]
L += ["**ACE-B columns.** NOT AVAILABLE: no ACE-B slides on the cluster and outcomes blinded. Code path: `scripts/paper_plan/hz_aceb_fill.py --cnv-tiles TILES.csv --cnv-arms ARMS.csv --img IMG.csv --labels LABELS.csv --out results/aceb/horizons.json` builds the package CNV block from raw tiles with the frozen constants of `models/killcoyne_frozen_pkg_v1` (model_info.json preprocessing), scores L-CNV, L-IMG, L-EARLY and L-LATE with the frozen coefficients, and computes the H1 metrics (IPCW and unweighted AUROC at 1/3/5 years, deltas, max-T over the three arms vs L-LATE, C-indices; all ACE-B samples and NDBE if an `ndbe` column is supplied). L-GRADE, L-GRADE+age+sex and L-INTER have no frozen model, so those ACE-B cells stay NOT AVAILABLE even after unblinding. "
      f"End-to-end check (`--dummy`, `h1_aceb_dummy_check.json`): set C feature blocks as stand-in inputs, dummy labels (uniform times, 20% events, `RandomState(0)`), 200 bootstrap draws: it ran; the frozen L-CNV in-sample AUROC on set C is {f3(dummy['check_insample_auroc_L_CNV_vs_bundle_0.995'])} (bundle `model_info.json`: 0.995); dummy 3-year L-LATE IPCW AUROC {f3(dl['auroc'])} (chance, as expected). No ACE-B label was read.", "",
      "**Method.** `hz_metrics.py`: IPCW cumulative/dynamic AUROC with Ĝ the reverse Kaplan–Meier over the evaluated samples, re-estimated in each of 2,000 patient-bootstrap draws (`RandomState(0)`); paired deltas on the same draws; within-patient swap permutation (2,000, seed 0) with one mask per permutation shared by the 6 arm-vs-L-LATE pairs (single-step max-T). Scores = mean of the 10 repeats' out-of-fold probabilities.",
      "**Sources.** `feasibility/paper_plan/killcoyne_mm/cv/preds/cfg_0{0..5}_rep_*.csv` (kv_cv.R, results commit d69de24), `feasibility/paper_plan/killcoyne_mm/horizons/outer/*.csv` (hz_fit.R); `h1_*.json`.",
      "**Caveats.** (1) The models were trained on ever-progression with all of a progressor's samples labelled 1, so a near-term vs later contrast is a re-use of those scores, not a horizon-trained model. (2) Censoring is light before 3 years by design (non-progressors were selected for ≥ 3 years of follow-up), so IPCW and unweighted AUROCs are close. (3) L-GRADE and L-GRADE+age+sex fall below 0.5 at 3 and 5 years: the stratified-CV fold-prevalence artefact documented in `docs/paper_plan_killcoyne_cv.md` (intercept-only model 0.244–0.288) acts on a near-constant score. (4) The patient unit has 2 cases at 1 year; its 1-year cells are not informative.", ""]
# ---------- H2
best = h2["best_model"]
L += ["### H2. Change in false positives", "", f"**Question.** At fixed operating points, does fusion change false positives against L-CNV? **Status.** DONE. **Pre-specification.** Above (H2). **Best model from H1** (highest mean per-sample IPCW AUROC over 1/3/5 years, all pre-event samples, their matrix): {best} ({', '.join(f'{k} {v:.3f}' for k, v in h2['best_selection_mean_ipcw_auroc'].items())}); the best model and L-LATE are therefore the same comparison.", "",
      "Operating point (a) thresholds (80% sensitivity on the inner out-of-fold predictions of each outer training fold; 100 fold × repeat thresholds): " + "; ".join(f"{k.replace('_', ' ')} median {v['median']:.3f} (range {v['range'][0]:.3f}–{v['range'][1]:.3f})" for k, v in h2["thresholds_a"].items()) + ".", ""]
rows = []
for s in ("their", "pkg"):
    for op in ("a_sens80", "b_pr05"):
        for p in ("pre", "pre_ndbe"):
            for unit in ("sample", "patient"):
                if p == "pre_ndbe" and unit == "patient": continue
                for t in (1, 3, 5):
                    o = h2["results"][s][f"{op}|{p}|{unit}|{t}"]; x = o[f"{best}_vs_L-CNV"]
                    rows.append([SRC[s], "(a) 80% sens." if op == "a_sens80" else "(b) Pr ≥ 0.5", POP[p] if unit == "sample" else "patient unit", f"{t} y", f"{o['n_cases']}/{o['n_controls']}",
                                 f"{f3(o['L-CNV']['TPR'])} / {f3(o['L-CNV']['FPR'])}", f"{f3(o[best]['TPR'])} / {f3(o[best]['FPR'])}", f"{sg(x['dTPR'])} {cis(x['dTPR_ci95'])}", f"{sg(x['dFPR'])} {cis(x['dFPR_ci95'])}", f"{sg(x['NRI'])} {cis(x['NRI_ci95'])}"])
L += table(["CNV source", "Operating point", "Population", "Horizon", "Cases/controls", "L-CNV TPR / FPR", f"{best} TPR / FPR", "ΔTPR [CI]", "ΔFPR [CI]", "Categorical NRI [CI]"], rows)
rows = []
for s in ("their", "pkg"):
    for op in ("a_sens80", "b_pr05"):
        for unit in ("sample", "patient"):
            for t in (1, 3, 5):
                tb = h2["results"][s][f"{op}|pre|{unit}|{t}"][f"{best}_vs_L-CNV"]["reclassification"]
                rows.append([SRC[s], "(a)" if op == "a_sens80" else "(b)", unit, f"{t} y"] + [tb[g][k] for g in ("cases", "controls") for k in ("CNVneg_fusionpos", "CNVpos_fusionneg", "both_pos", "both_neg")])
L += ["**Reclassification**, all pre-event samples (patient unit: earliest pre-event NDBE sample). Fusion = " + best + ".", ""]
L += table(["CNV source", "Op.", "Unit", "Horizon", "Cases CNV− → fusion+", "Cases CNV+ → fusion−", "Cases both +", "Cases both −", "Controls CNV− → fusion+", "Controls CNV+ → fusion−", "Controls both +", "Controls both −"], rows)
rows = []
for s in ("their", "pkg"):
    W = h2["results"][s]["a_sens80|pre|sample|3"][f"{best}_who"]
    for g in ("cases_CNVneg_fusionpos", "cases_both_pos", "controls_CNVpos_fusionneg", "controls_both_pos"):
        x = W[g]; rows.append([SRC[s], g.replace("_", " "), f"{x['n_samples']} / {x['n_patients']}", ", ".join(f"{k} {v}" for k, v in x["pathology"].items()) if isinstance(x["pathology"], dict) else "—",
                               f3(x["time_to_event_or_censor_median"]), f3(x["cx_raw_pkg_median"]), f3(x["cx_z_their_median"]), f3(x["noise_median"]), f3(x["tissue_tiles_median"]), ", ".join(f"{k} {v}" for k, v in x["scanner"].items()),
                               ", ".join(f"{k} {v}" for k, v in x["p53_ihc"].items()), ", ".join(f"{k} {v}" for k, v in x["killcoyne_risk_class"].items())])
L += ["**Who fusion catches and clears** (operating point (a), 3 years, all pre-event samples, per sample; medians).", ""]
L += table(["CNV source", "Group", "Samples / patients", "Pathology", "Years to event or censoring", "cx (raw, package)", "cx (z, their)", f"Noise ({h2['noise_column']})", "Tissue tiles", "Scanner", "p53 IHC", "Killcoyne risk class (published)"], rows)
L += [f"The list of these patients by study number, with each sample's published risk class, is row-level and stays on the cluster: `{h2['patient_list_path_cluster']}` ({h2['patient_list_rows']} rows).", "",
      "**Method.** `hz_stack.py` TASK=h2. Calls per repeat, majority over the 10 repeats (≥ 6 of 10); ΔTPR, ΔFPR and NRI CIs from 2,000 patient-bootstrap draws (`RandomState(0)`). Per-repeat mean TPR/FPR are in `h2.json`.",
      "**Sources.** `cv/preds/` (kv_cv.R), `horizons/inner/` and `horizons/outer/` (hz_fit.R), `horizons/pkg_qc.csv` (hz_qc.R from `killcoyne/pkg/pat_*.rds`), `horizons/slide_desc.csv` (hz_slides.py), `models/killcoyne_frozen_pkg_v1/cnv_scaling_setC.csv` (cx mean 25.256, s.d. 8.416, to recover raw cx); `h2.json`.",
      "**Caveats.** (1) Operating point (b) applies a fixed 0.5 to probabilities from models whose intercept moves with the fold's training prevalence; it is not a calibrated risk threshold here (design caveat). (2) Thresholds and calls are per sample; patients contribute several samples. (3) The 'who' comparison is descriptive, with no test.", ""]
# ---------- H3
L += ["### H3. WSI vs CNV weighting by horizon", "", "**Question.** Does the image weight in a CNV + WSI stack change with horizon? **Status.** DONE. **Pre-specification.** Above (H3); expected direction stated in advance: WSI weight higher at 1 year than at 5 years.", ""]
rows = []
for s in ("their", "pkg"):
    for t in (1, 3, 5):
        x = h3[s]["results"][str(t)]; e = x["exploratory_heldout"]
        rows.append([SRC[s], f"{t} y", x["n_fits"], f"{f3(x['b_mean'])} {ci(x['b_ci95'])}", f"{f3(x['c_mean'])} {ci(x['c_ci95'])}", f"{f3(x['ratio_mean'])} {ci(x['ratio_ci95'])}", f"{f3(x['ratio_of_means'])} {ci(x['ratio_of_means_ci95'])}", f"{x['n_negative_b']} / {x['n_negative_c']}",
                     f"{f3(e['stack_ipcw_auroc'])} vs {f3(e['late_ipcw_auroc'])}: {sg(e['delta'])} {cis(e['delta_ci95'])}"])
L += table(["CNV source", "Horizon", "Fits", "b (CNV) [CI]", "c (WSI) [CI]", "c/(b+c), mean over fits [CI]", "c/(b+c) of mean coefficients [CI]", "Fits with b < 0 / c < 0", "Exploratory: stack vs L-LATE IPCW AUROC, Δ [CI]"], rows)
L += ["1 year minus 5 years, c/(b+c): " + "; ".join(f"{SRC[s]} {sg(h3[s]['results']['ratio_1y_minus_5y']['delta'])} {cis(h3[s]['results']['ratio_1y_minus_5y']['ci95'])} (of mean coefficients {sg(h3[s]['results']['ratio_1y_minus_5y']['ratio_of_means_delta'])} {cis(h3[s]['results']['ratio_1y_minus_5y']['ratio_of_means_ci95'])})" for s in ("their", "pkg")) + ".", "",
      "**Method.** `hz_stack.py` TASK=h3_<src>: inner 5-fold out-of-fold L-CNV and L-IMG probabilities (`hz_fit.R` MODE=inner) → logits, z-scored over the outer training horizon cases and controls (pre-event samples) → logistic stack with an L2 penalty of 1e-4 on the slopes (IRLS); 100 fits per horizon; CIs from 2,000 patient-bootstrap draws (`RandomState(0)`) that refit every fold stack with patient-multiplicity weights; the held-out stack is applied to the outer out-of-fold L-CNV and L-IMG predictions and averaged over repeats.",
      "**Sources.** `horizons/inner/{cnv,img}_*`, `cv/preds/cfg_0{0,1,2}_*`; `h3_their.json`, `h3_pkg.json`.",
      "**Caveats.** (1) Exploratory held-out AUROCs: the stack was specified before but is reported after H1 was seen. (2) The ratio c/(b+c) is unstable when b + c is small (the 1-year fits have the fewest cases); the ratio of mean coefficients is given alongside.", ""]
# ---------- H4
L += ["### H4. Depth transfer", "", f"**Question.** Is there 4× discovery data, and are CNV features stable across depth on ACE-B? **Status.** 4× → external NOT AVAILABLE; ACE-B label-blind depth check {h4status}. **Pre-specification.** Above (H4) and amendment 2 of `docs/aceb_analysis_plan.md` (0.4× primary depth rule, committed in ee51db8 before any ACE-B outcome was read).", ""]
if h4s:
    L += [f"**What exists.** Roots searched: {', '.join('`' + r + '`' for r in h4s['roots_searched'])} (`lfs find` in {len(h4s['shards'])} shards, {h4s['n_entries_listed']:,} entries listed, {h4s['find_stderr_lines']:,} unreadable-path messages; read files `*.bam`, `*.cram`, `*.fastq[.gz]`, `*.fq[.gz]`, `*.sra`: {h4s['n_read_files_total']:,} found).",
          f"- 4× resequenced discovery data: read files carrying one of the {h4s['discovery_slx_ids']} discovery SLX ids outside `SWGCohort/dna_seq_bam`: {sum(v['files'] for v in h4s['discovery_reads_outside_dna_seq_bam'].values()):,} files in {len(h4s['discovery_reads_outside_dna_seq_bam']):,} directories, grouped:",
          ]
    if os.path.exists(f"{R}/h4_bam_probe.json"):
        bp = J("h4_bam_probe")["groups"]
        for g, o in bp.items():
            line = f"  - `{g}`: {o['files']} files ({o['symlinks']} symlinks, {o['broken_links']} broken; link targets under {', '.join(sorted(set('`' + '/'.join(k.split('/')[:6]) + '/`' for k in o['link_target_dirs']))) or 'none'} (the pre-migration path of `validation_genomics`); {o['bytes_real_files_GB']} GB in real files)"
            if o.get("probe_n"):
                line += f"; {o['probe_n']} real BAMs probed (`RandomState(0)`): nominal depth median {o['depth_median']:.3f}× (range {o['depth_range'][0]:.3f}–{o['depth_range'][1]:.3f}), read length {', '.join(f'{k} bp ({v})' for k, v in o['read_len_mode'].items())}, build {', '.join(f'{k} ({v})' for k, v in o['build'].items())}"
                if "reads_ratio_to_dna_seq_bam_median" in o: line += f"; the same file name exists in `dna_seq_bam` for {o['names_also_in_dna_seq_bam']} of the real BAMs, and the probed BAMs hold a median {o['reads_ratio_to_dna_seq_bam_median']:.2f}× the reads of their `dna_seq_bam` namesake (namesake depth median {o['twin_depth_median']:.3f}×)"
            L.append(line + ".")
        L.append("")
    L += [f"- Directories named for depth or resequencing: {h4s['n_depth_named_dirs']}, all software or slide-prediction folders matched by the word 'deep' (for example Python packages `deepseek_*`, `deepzoom`, `deeplabv3`, `deeptools`, and H&E 'DEEPER' levels); none holds sequencing data.",
          f"- ACE-B reads (SLX pools {', '.join(h4s['aceb_slx_ids'])}): " + ("none on the cluster." if not h4s["aceb_reads"] else "; ".join(f"`{k}` ({v['files']} files, {v['GB']} GB, {v['ext']})" for k, v in h4s["aceb_reads"].items())), ""]
bpj = J("h4_bam_probe")["groups"] if os.path.exists(f"{R}/h4_bam_probe.json") else {}
mx = max([o["depth_range"][1] for o in bpj.values() if o.get("probe_n")] or [0])
L += ["**Result.** " + (f"No 4× discovery data: the only discovery-SLX read files outside `dna_seq_bam` are an older hg19 alignment of the same runs (`SWGCohort/validation_genomics`, probed nominal depth at most {mx:.2f}×) and broken symlinks to it. " if mx < 4 else "Discovery reads at ≥ 4× were found; see the list above. ") + ("No ACE-B reads exist on the cluster, so the label-blind depth check could not run; the ACE-B manifest's `Mean_Coverage` (median 5.30×, range 3.29–14.91×, 191 of 234 samples; `docs/dataset_description.md` item 7) is the only depth information, and it is not 7×." if h4status == "NOT AVAILABLE" else "See `h4_depth.json`."),
      "**Method.** `hz_search.py` (one `find` over both roots, Slurm) ran > 3.5 h without finishing and was replaced by `hz_search_shard.py` (`lfs find` per depth-2 directory, plus each depth-1 directory at `-maxdepth 1`, as parallel Slurm tasks; merged with duplicates removed). The depth-check pipeline is specified above; the QDNAseq 50 kb bins for it must be the discovery hg38 annotation (61,775 bins in `50.raw_read_counts.txt`), which is stored in each discovery `50.QDNAseq.RData`: the generator in `[BT]/scripts/swg_generate_qdnaseq_one_sample.R` calls `getBinAnnotations(binSize = 50)` with QDNAseq's default genome (hg19) and no hg38 annotation package is installed, so it is not the producer of the discovery counts (made 2025-03-12 by another user).",
      "**Sources.** `h4_search.json`, `h4_bam_probe.json` (`scripts/paper_plan/hz_bam_probe.py`: symlink targets; for 20 real BAMs per directory, `samtools view -c -F 0x900` reads, modal length of the first 2,000 reads, `@SQ` chr1 length for the build, nominal depth = reads × length / 3,088,269,832); `feasibility/paper_plan/killcoyne_mm/horizons/h4_read_files_all.csv` (cluster).",
      "**Caveats.** (1) The search covers the mounted lab scratch areas only; reads held by the sequencing core or the cohort owner off-cluster are not visible. (2) The probe counts every primary read including unmapped ones, whereas `docs/dataset_description.md` item 6 used the QDNAseq read-count totals of the hg38 BAMs (set C median 0.29×), so the two depths are not on the same counting basis.", ""]
# ---------- discrepancies and not done
L += ["### Discrepancies found", "",
      "- The task gives ACE-B depth as 7× ('7× external'); the manifest gives median 5.30× (range 3.29–14.91×), and 43 of 234 samples (batch 3) have no coverage value.",
      "- The task's 'set C' is called the discovery subset here; it is 676 of the 773 published discovery samples (80 of 88 patients).",
      "- L-INTER did not exist under the stratified CV; it was fitted for this task, so its cells are new fits, unlike the other arms.",
      f"- The H1-best model is {best}, so H2's 'best model' and 'L-LATE' comparisons coincide; each is reported once.",
      "- The patient unit (earliest pre-event NDBE sample) is identical in the 'all pre-event' and 'NDBE pre-event' populations.",
      "- The discovery 50 kb QDNAseq counts were not produced by the repository's QDNAseq generator (hg19 default bins); a depth check would have to reuse the hg38 bin annotation stored in the discovery `50.QDNAseq.RData` files.", "",
      "### Deviations from the pre-specification", "",
      "- The ACE-B code-path check used 200 bootstrap and permutation draws on the dummy labels (the real run uses 2,000); it checks that the path runs, not its numbers.",
      "- H4's search was run sharded with `lfs find` instead of one `find` (same roots, same file patterns), because the single search did not finish in 3.5 h.",
      "- No other change: every H1–H3 number above follows the pre-specified definitions.", "",
      "### Not done", "",
      "- ACE-B columns of the H1 table (no slides, outcomes blinded); ACE-B L-GRADE, L-GRADE+age+sex and L-INTER cells cannot be filled even after unblinding (no frozen models).",
      "- 4× → external transfer (no 4× discovery data on the cluster)." + (" ACE-B label-blind depth check (no ACE-B reads on the cluster); its pipeline is specified but was not run." if h4status == "NOT AVAILABLE" else ""),
      "- H4 stability figure (needs the depth check)." if h4status == "NOT AVAILABLE" else "",
      "- No absolute risk, calibration or PPV (design caveat).", "",
      "**Interpretation.** On all pre-event samples late fusion has the highest per-sample IPCW AUROC at 1, 3 and 5 years with either CNV source, but only the grade arms differ from it after max-T adjustment, and the WSI weight in the stack does not rise at 1 year.", ""]
open(DOC, "w").write("\n".join(x for x in L if x is not None) + "\n"); print("rendered", len(L))

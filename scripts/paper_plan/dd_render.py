"""Render docs/dataset_description.md from results/paper_final/dataset_description/*.json (written by dd_describe.py). Usage: python dd_render.py COMMIT"""
import json, sys
C = sys.argv[1]
R = "results/paper_final/dataset_description"
J = {p: json.load(open(f"{R}/{p}.json")) for p in ("core", "wsi_release", "wsi_setc", "swgs")}
core, wr, wc, sw = J["core"], J["wsi_release"], J["wsi_setc"], J["swgs"]

def f(x, nd=2):
    if x is None: return "—"
    if isinstance(x, float) and x.is_integer() and nd <= 2 and abs(x) >= 100: return f"{int(x):,}"
    return f"{x:,.{nd}f}" if isinstance(x, float) else f"{x:,}" if isinstance(x, int) else str(x)
def stat_cells(d, nd=2, rng=True):
    if not d or d.get("n", 0) == 0: return ["0", "—", "—", "—"] + (["—"] if rng else [])
    c = [str(d["n"]), f(d["mean"], nd), f(d["median"], nd), f"{f(d['q1'], nd)}–{f(d['q3'], nd)}"]
    return c + ([f"{f(d['min'], nd)}–{f(d['max'], nd)}"] if rng else [])
def counts(d): return ", ".join(f"{k}: {v}" for k, v in sorted(d.items(), key=lambda kv: -kv[1])) if d else "—"
def table(head, rows):
    return ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"] + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows] + [""]
SH = {"all": "both sheets", "777": "777-sheet", "268": "268-sheet"}; LB = {"all": "all", "P": "progressors", "NP": "non-progressors"}
L = ["# Dataset description: frozen SWG release, Killcoyne-protocol set C, ACE-B manifest", "",
     f"Report only: nothing is fitted or retrained. All numbers are computed by `scripts/paper_plan/dd_describe.py` (commit {C}), run through Slurm (`scripts/cluster/campaign.sh`, prefixes dd1, dd2, dd3), and written as aggregates to `{R}/{{core,wsi_release,wsi_setc,swgs}}.json`; this file is rendered by `scripts/paper_plan/dd_render.py`. Every row names its input file and the JSON key that holds the number. No row-level data are in this repository.", "",
     "Path aliases (cluster):", "",
     "- `[REL]` = `/mnt/scratche/slow/fmlab/zuberi01/phd/barretts_retraining/barretts_training/analysis/chapter1_lgd2_final_pre_event_20260713_final` (frozen release)",
     "- `[BT]` = `/mnt/scratche/slow/fmlab/zuberi01/phd/barretts_retraining/barretts_training`",
     "- `[SWG]` = `/mnt/scratche/fast/fmlab/datasets/imaging/SWGCohort`",
     "- `[UNI]` = `/mnt/scratche/slow/fmlab/zuberi01/phd/barretts_retraining/data/foundation_outputs/uni2_tile224_lvl2`",
     "- `[KC]` = `/mnt/scratche/slow/fmlab/zuberi01/phd/thesis/feasibility/paper_plan/killcoyne` (and `[KC]_mm`, set C)",
     "- `[DB]` = `/mnt/scratche/slow/fmlab/zuberi01/barretts_db_export` (Barrett's database export)",
     "- `[ACEB]` = `/mnt/scratche/slow/fmlab/zuberi01/phd/aceb_meta/ACEB_samples_for Rehan.csv`",
     "- `[777]` = `[SWG]/sWGS_777_samples_cleaned_202401_Leanne_fullDetails (3) (1).csv`; `[268]` = `[SWG]/sWGS_validation_cleaned_Leanne (4) (1).csv`",
     "",
     "Definitions used throughout:", "",
     "- Release rows = the 707 strict pre-event rows of `[REL]/training_manifest.csv`, joined to `[REL]/pre_event_cohort.csv` (SampleID) and `[REL]/matched_manifest.csv` (biopsy_id, slide_id, cnv_id). Progressor = patient max of `training_manifest.y_progressor` (LGD2+ endpoint of the release).",
     "- Sheet of a release sequencing sample: 777-sheet if its cnv_id is a `combined_name` in `[777]`, 268-sheet if it is in `[SWG]/copy_number_hg38/val/`; a patient takes the sheet of its samples (no patient is mixed).",
     "- Set C = `[KC]_mm/set_C.csv`: the published Killcoyne samples with a UNI2 bag; progressor = sheet `Status` P.",
     "- Statistics are over patients unless the row says samples or slides; IQR = 25th–75th percentile; n = patients (or samples/slides) with a value.", "",
     "Status: items 1–4 DONE; item 5 DONE; item 6 DONE; item 7 PARTIAL (the manifest has coverage, batch and pool only; WSI, read length, build, bins and pipeline NOT AVAILABLE).", ""]

# ---------- item 1
i1 = core["item1"]
L += ["## 1. Patients", ""]
rows = [["release", SH[s], i1["release"][s]["patients"], i1["release"][s]["P"], i1["release"][s]["NP"], "`[REL]/training_manifest.csv`, `[777]`, `[SWG]/copy_number_hg38/val/`; `core.json:item1.release`"] for s in ("all", "777", "268")]
rows += [["set C", "777-sheet (all samples)", i1["setC"]["all"]["patients"], i1["setC"]["all"]["P"], i1["setC"]["all"]["NP"], "`[KC]_mm/set_C.csv` (Status); `core.json:item1.setC`"]]
L += table(["Cohort", "Sheet", "Patients (n)", "Progressors (n)", "Non-progressors (n)", "Source"], rows)
L += [f"Release samples by sheet: {counts(i1['release_samples_by_sheet'])} (n = 707; `core.json:item1.release_samples_by_sheet`). The sheet assignment agrees with `feasibility/paper_plan/f2_strata.csv` for {i1['release_sheet_vs_f2_strata']['agree']} of {i1['release_sheet_vs_f2_strata']['n']} patients; matching on patient ID instead would give {counts(i1['release_sheet_vs_patient_id_membership']['id_membership_counts'])}. "
      f"The master column `Progressor_label` gives {i1['release_Progressor_label_patients']['P']} progressors of {i1['release_Progressor_label_patients']['n']} (the 35 in `[REL]/table1_cohort.json`); the release label used by the models is `y_progressor` (50). "
      f"All {i1['setC_samples_in_777_sheet']['in']} of {i1['setC_samples_in_777_sheet']['n']} set C samples are 777-sheet rows; {i1['setC_patients_with_release_rows']} of the 80 set C patients also have release rows.", ""]

# ---------- item 2
i2 = core["item2"]
L += ["## 2. Timepoints per patient", ""]
U = {"rows": "release rows (slide + sWGS pairs)", "biopsies": "distinct biopsies", "slides": "distinct slides", "swgs": "distinct sWGS profiles"}
rows = []
for u in ("rows", "biopsies", "slides", "swgs"):
    for s in ("all", "777", "268"):
        for lab in ("all", "P", "NP"):
            d = i2["release"][u][f"{s}|{lab}"]
            rows.append(["release", U[u], SH[s], LB[lab]] + stat_cells(d, 1) + [d.get("single", 0), f"`[REL]/matched_manifest.csv`; `core.json:item2.release.{u}`"])
for u, nm in (("swgs", "sWGS samples"), ("slides", "distinct slides")):
    for lab in ("all", "P", "NP"):
        d = i2["setC"][u][f"all|{lab}"]
        rows.append(["set C", nm, "777-sheet", LB[lab]] + stat_cells(d, 1) + [d.get("single", 0), ("`[KC]_mm/set_C.csv`" if u == "swgs" else "`[KC]/uni2_npz_map.csv`") + f"; `core.json:item2.setC.{u}`"])
L += table(["Cohort", "Unit", "Sheet", "Label", "Patients (n)", "Mean", "Median", "IQR", "Range", "Patients with one", "Source"], rows)
L += ["In the release each row is one slide paired with one sWGS profile; 707 rows hold 707 distinct slides and 693 distinct sWGS profiles (`[REL]/feature_views/feature_view_metadata.json`). The field `biopsies_per_patient` of `[REL]/table1_cohort.json` (median 3, IQR 2–7, range 1–19) counts rows, not distinct biopsies.", ""]

# ---------- item 3
i3 = core["item3"]
L += ["## 3. Follow-up (years)", ""]
rows = []
for m, nm in (("first_to_last", "first to last sample"), ("first_to_endpoint", "first sample to endpoint")):
    for s in ("all", "777", "268"):
        for lab in (("all", "P", "NP") if m == "first_to_last" else ("P",)):
            d = i3["release"][m][f"{s}|{lab}"]
            rows.append(["release", nm, SH[s], LB[lab]] + stat_cells(d, rng=False) + [f"`[REL]/pre_event_cohort.csv` (Date{', is_lgd2_event_at_current, NextBiopsyDate, NextBiopsyProgression_LGD2plus' if m != 'first_to_last' else ''}); `core.json:item3.release.{m}`"])
for m, nm in (("first_to_last", "first to last sample"), ("first_to_endpoint", "first sample to endpoint")):
    for lab in (("all", "P", "NP") if m == "first_to_last" else ("P",)):
        d = i3["setC"][m][f"all|{lab}"]
        rows.append(["set C", nm, "777-sheet", LB[lab]] + stat_cells(d, rng=False) + [f"`[777]` (Months before final), `[KC]/kr_samples.csv` (Pathology); `core.json:item3.setC.{m}`"])
L += table(["Cohort", "Measure", "Sheet", "Label", "Patients (n)", "Mean", "Median", "IQR", "Source"], rows)
L += [f"Release: sample dates are `pre_event_cohort.Date` for the release rows ({i3['release_rows_with_date']} of 707 dated). The endpoint date is the earliest of the date of a row flagged `is_lgd2_event_at_current` and the `NextBiopsyDate` of a row with `NextBiopsyProgression_LGD2plus` = 1, over all 959 cohort rows of the patient; available for {i3['release_P_with_event_date']['n']} of {i3['release_P_with_event_date']['of']} progressors. "
      f"Set C: times are differences of the sheet's `Months before final` ({i3['setC_samples_with_mbf']['n']} of {i3['setC_samples_with_mbf']['of']} samples have it) divided by 12. The endpoint is the first HGD/IMC sample of the patient among the published samples (`kr_samples.csv` Pathology or sheet `Path_class_per_OGD`), as in `scripts/paper_plan/kc_merge.py:31-32`, for {i3['setC_P_with_hgd_imc_sample']['n']} progressors; for the other {i3['setC_P_with_hgd_imc_sample']['fallback_final_endoscopy']} the final endoscopy (same fallback). "
      f"{i3['setC_P_endpoint_before_first_setC_sample']} progressor has its endpoint before its first set C sample (negative time; set C contains post-event samples).", ""]

# ---------- item 4
i4 = core["item4"]
L += ["## 4. Demographics at first sample", ""]
L += ["Release-only = the cohort's demographic sheet `[SWG]/Demographics_full.csv` (Study Number or Alternate Study Number = release patient ID); the release itself carries no demographic columns. Database-completed = release-only value, else the Barrett's database through `pre_event_cohort.participant_id`: date of birth from `[DB]/initial_history.parquet`; Prague C/M from `[DB]/endoscopy.parquet` (`barretts_circumference`, `barretts_maximum`, the endoscopy closest in date to the first sample), else `initial_history` `praguec`/`praguem`. The database has no sex field outside the OCCAMS tables. Its smoking field is coded 0/1/2/3 with no codebook in the export, so it cannot be merged with the Y/N sheet; the codes are listed for the patients the sheet lacks. Baseline grade = grade of the patient's first release row (`pre_event_cohort.Label`: 0 NDBE, 1 IND, 2 LGD); for set C, the `Pathology` of the earliest set C sample.", ""]
rows = []
for coh, key in (("release", "release"), ("set C", "setC")):
    T = i4[key]
    for lab in ("all", "P", "NP"):
        o = T[lab]; N = o["n"]
        for v, nm in (("age", "age at first sample (years)"), ("C", "Prague C (cm)"), ("M", "Prague M (cm)")):
            for ver in ("release", "completed"):
                d = o[f"{v}_{ver}"]
                summ = "—" if d.get("n", 0) == 0 else f"mean {f(d['mean'])}; median {f(d['median'])} [{f(d['q1'])}–{f(d['q3'])}]; range {f(d['min'])}–{f(d['max'])}"
                srcs = counts(o.get(f"{v}_completed_sources", {})) if ver == "completed" else "demographics"
                extra = ""
                if ver == "completed" and v == "C" and o["C_endoscopy_gap_days"].get("n", 0):
                    g = o["C_endoscopy_gap_days"]; extra = f"; endoscopy-to-first-sample gap median {f(g['median'], 0)} days (range {f(g['min'], 0)}–{f(g['max'], 0)})"
                rows.append([coh, LB[lab], nm, "release-only" if ver == "release" else "database-completed", f"{d.get('n', 0)} / {N}", summ, ("`Demographics_full.csv` (Date of birth / Circumference / Maximal), `pre_event_cohort.Date`" if ver == "release" else f"by source: {srcs}{extra}; `Demographics_full.csv`, `[DB]/initial_history.parquet`, `[DB]/endoscopy.parquet`") + f"; `core.json:item4.{key}.{lab}.{v}_{ver}`"])
        rows.append([coh, LB[lab], "sex", "release-only (database: NOT AVAILABLE)", f"{o['sex_n']} / {N}", counts(o["sex_release"]), f"`Demographics_full.csv` (Sex); `core.json:item4.{key}.{lab}.sex_release`"])
        rows.append([coh, LB[lab], "smoking", "release-only", f"{o['smoking_n']} / {N}", counts(o["smoking_release"]), f"`Demographics_full.csv` (Smoking Status); `core.json:item4.{key}.{lab}.smoking_release`"])
        nraw = sum(o["smoking_db_raw_codes_for_missing"].values())
        rows.append([coh, LB[lab], "smoking", "database code, patients missing above (not harmonised)", f"{nraw} / {N - o['smoking_n']}", counts({f"code {k}": v for k, v in o["smoking_db_raw_codes_for_missing"].items()}), f"`[DB]/initial_history.parquet` (smoking); `core.json:item4.{key}.{lab}.smoking_db_raw_codes_for_missing`"])
        rows.append([coh, LB[lab], "baseline grade", "release", f"{o['baseline_grade_n']} / {N}", counts(o["baseline_grade"]), ("`[REL]/pre_event_cohort.csv` (Label, Date)" if key == "release" else "`[KC]_mm/set_C.csv` (Pathology), `[777]` (Months before final)") + f"; `core.json:item4.{key}.{lab}.baseline_grade`"])
L += table(["Cohort", "Label", "Variable", "Version", "n available / N", "Summary", "Source"], rows)
L += [f"Coverage: release {i4['release']['in_demographics_sheet']} of 150 patients are in the demographic sheet and {i4['release']['with_participant_id']} have a database participant_id; set C {i4['setC']['in_demographics_sheet']} of 80 and {i4['setC']['with_participant_id']} (`core.json:item4`). Set C first-sample dates: {', '.join(f'{k} for {v} patients' for k, v in json.loads(i4['setC']['first_sample_date_source'].split(': ', 1)[1]).items())} (the cohort date of the same sequencing sample).", ""]

# ---------- item 5
def wsi_rows(coh, w, key, extra_src):
    rows = [[coh, "slides", w["n_slides"], f"{w['n_slides']} (read errors {w['errors']})", f"{extra_src}; `{key}.json:n_slides`"]]
    for p, n in w["scanners"].items():
        mp = w["mpp_native_by_scanner"][p]; lp = w["level_mpp_by_scanner"][p]
        rows.append([coh, f"scanner {p} (Hamamatsu, `hamamatsu.Product`)", n, f"{n} slides" + (f", {w['patients_by_scanner'][p]} patients" if "patients_by_scanner" in w else "") + f"; native {f(mp['median'], 4)} µm/px (range {f(mp['min'], 4)}–{f(mp['max'], 4)}); at the tiling level {f(lp['median'], 4)} µm/px", f"slide properties (openslide); `{key}.json:scanners, mpp_native_by_scanner, level_mpp_by_scanner`"])
    rows.append([coh, "objective power (openslide)", sum(w["objective"].values()), ", ".join(f"{k}× ({v} slides)" for k, v in w["objective"].items()), f"slide properties; `{key}.json:objective`"])
    rows.append([coh, "native resolution, all slides (µm/px, level 0)", w["mpp_native"]["n"], f"median {f(w['mpp_native']['median'], 4)}; range {f(w['mpp_native']['min'], 4)}–{f(w['mpp_native']['max'], 4)}", f"`openslide.mpp-x`; `{key}.json:mpp_native`"])
    rows.append([coh, "tiling level / tile size (px)", sum(w["level"].values()), f"level {'/'.join(w['level'])} / {'/'.join(w['tile_size'])} px (all slides)", f"npz keys `level`, `tile_size`; `{key}.json:level, tile_size`"])
    lm = w["level_mpp"]; rows.append([coh, "resolution at tiling level (µm/px) and tile field (µm)", lm["n"], f"median {f(lm['median'], 4)} µm/px (range {f(lm['min'], 4)}–{f(lm['max'], 4)}); 224 px = {f(224 * lm['median'], 1)} µm", f"mpp × `level_downsamples[2]`; `{key}.json:level_mpp`"])
    for k, nm in (("tissue_grid_tiles", "tissue tiles available (kept: non-overlapping level-2 224-px cells whose centre is in the tissue mask)"), ("tiles_requested", "tiles sampled (requested)"), ("tiles_ok", "tiles embedded"), ("distinct_sampled", "distinct sampled tile positions")):
        d = w[k]; rows.append([coh, nm, d.get("n", 0), "—" if not d.get("n") else f"mean {f(d['mean'], 1)}; median {f(d['median'], 1)} [{f(d['q1'], 1)}–{f(d['q3'], 1)}]; range {f(d['min'], 1)}–{f(d['max'], 1)}", ("`[SWG]/unannotated/masks/*_tissuetector.png`" if k == "tissue_grid_tiles" else "npz `tiles_requested`/`tiles_ok`/`coords_level`") + f"; `{key}.json:{k}`"])
    rows.append([coh, "slides with fewer tissue tiles than the 256 sampled (sampled tiles overlap)", w["n_slides"], w.get("slides_tissue_tiles_lt_requested", "PENDING"), f"`{key}.json:slides_tissue_tiles_lt_requested`"])
    rows.append([coh, "slides sampled with replacement (mask pixels < 256)", w["n_slides"], w["with_replacement"], f"`{key}.json:with_replacement`"])
    rows.append([coh, "feature extractor", w["n_slides"], f"UNI2-h (`timm` `hf-hub:MahmoodLab/UNI2-h`), embedding dim {'/'.join(w['emb_dim'])} (all slides); 256 tile centres drawn uniformly at random from tissue-mask pixels (seed 20260226 + chunk)", f"`[BT]/scripts/extract_foundation_from_masks.py` (lines 66-72, 156, 234-238); `{key}.json:emb_dim`"])
    return rows
L += ["## 5. Whole-slide images", ""]
rows = wsi_rows("release", wr, "wsi_release", "`[REL]/pre_event_cohort.csv` ImageAbsPath, `[REL]/feature_views/uni2/uni2_index.csv`")
rows += wsi_rows("set C", wc, "wsi_setc", "`[KC]/uni2_npz_map.csv`")
rows.append(["set C", "tiles per sample bag (union of the sample's slides)", wc["bag_samples"], f"median {f(wc['bag_tiles_per_sample']['median'], 0)}; range {f(wc['bag_tiles_per_sample']['min'], 0)}–{f(wc['bag_tiles_per_sample']['max'], 0)}; slides per sample {counts(wc['bag_slides_per_sample'])}", "`[KC]_mm/bag_index.csv`; `wsi_setc.json:bag_tiles_per_sample`"])
fv = wr["feature_views"]
rows.append(["release", "frozen feature views in the release (the image model uses uni2)", fv["uni2"]["rows"], "; ".join(f"{v}: {fv[v]['rows']} rows, dim {'/'.join(fv[v]['feat_dim'])}" for v in fv), "`[REL]/feature_views/*/*_index.csv`; `wsi_release.json:feature_views`"])
L += table(["Cohort", "Quantity", "n", "Value", "Source"], rows)
L += ["Tiles come from the UNI2 extraction of 2026-02-26 (`[UNI]/manifest/manifest_dedup.csv`), which the release indexes. The tissue mask is the `tissuetector` PNG of each slide.", ""]

# ---------- item 6
def sw_rows(coh, d, key):
    r = []
    r.append([coh, "sWGS samples (with read counts / with BAM)", d["samples"], f"{d['with_readcount']} / {d['with_bam']}", f"`{key}.samples`"])
    for k, nm, nd in (("total_reads", "total reads per sample", 0), ("used_reads", "reads used by QDNAseq", 0), ("depth_total", "depth per sample (× = total reads × read length / 3,088,269,832)", 4)):
        x = d[k]; r.append([coh, nm, x["n"], f"mean {f(x['mean'], nd)}; median {f(x['median'], nd)}; range {f(x['min'], nd)}–{f(x['max'], nd)}", f"`{{kb}}.readCountSummary.txt`; `{key}.{k}`"])
    r.append([coh, "read length (bp; first 2,000 reads per BAM)", d["with_bam"], f"modal length {', '.join(f'{k} bp ({v} samples)' for k, v in d['read_length_mode'].items())}; range {d['read_length_range'][0]}–{d['read_length_range'][1]} (trimmed)", f"`[SWG]/dna_seq_bam/*.bam`; `{key}.read_length_mode`"])
    r.append([coh, "alignment", d["with_bam"], f"bwa {'/'.join(d['bwa'])} `{'/'.join(d['aligner_cmd'])}` (all {d['with_bam']} BAMs) to hg38 ({counts(d['reference'])}); chr1–22, X, Y length {', '.join(f'{int(k):,} bp ({v} BAMs)' for k, v in d['genome_len_chr1_22_X_Y'].items())}", f"BAM `@PG`/`@SQ` headers; `{key}.bwa, reference`"])
    return r
L += ["## 6. Shallow whole-genome sequencing", ""]
rel_sw = sw["release"]; rows = []
for s in ("all", "777", "268"):
    rows += [[f"release, {SH[s]}"] + x[1:] for x in [[None] + y[1:] for y in sw_rows("", rel_sw[s], f"swgs.json:release.{s}")]]
rows = [[r[0]] + r[1:4] + [r[4].replace("{kb}", "500").replace("`500.readCountSummary.txt`", "`[SWG]/copy_number_hg38/{train/perPatient,val}/500kb/<id>/500.readCountSummary.txt`")] for r in rows]
rc = [[ "set C"] + y[1:] for y in sw_rows("", sw["setC"], "swgs.json:setC")]
rc = [r[:4] + [r[4].replace("`{kb}.readCountSummary.txt`", "`[SWG]/copy_number_hg38/train/perPatient/50kb/<id>/50.readCountSummary.txt`")] for r in rc]
rows += rc
x = sw["setC_sheet_number_of_reads"]
rows.append(["set C", "reads per sample, sheet column `Number of reads`", x["n"], f"mean {f(x['mean'], 0)}; median {f(x['median'], 0)}; range {f(x['min'], 0)}–{f(x['max'], 0)}", "`[777]`; `swgs.json:setC_sheet_number_of_reads`"])
a = sw["release_cnv_build"]
rows += [
    ["release", "CNV source: build / bins", 693, "hg38 / QDNAseq 500 kb (`500.copy_number.txt`)", "`[BT]/scripts/killcoyne_reproduce_from_500kb.R` (lines 28, 294-332: `bin_kb` default 500, maps the 50 kb path to the 500 kb folder); `[BT]/slurm/submit_stage1_segment.sh` line 45 (no `--bin_kb`, so 500)"],
    ["release", "CNV pipeline", 707, f"QDNAseq 500 kb copy number → {a['segmentation_method']} (copynumber {a['copynumber_version']}), gamma 40, kmin 5, QC none → 587 5-Mb windows (arm-adjusted) + 44 arm means + cx = 632 features", "`[BT]/data/killcoyne_repro_strict_500kb_slurm_v2/run_config.json` (args); `[REL]/feature_views/cnv/*.csv` (column counts); `swgs.json:release_cnv_build`"],
    ["release", "producer of the 500 kb QDNAseq folders", 693, "NOT AVAILABLE: no generating script in `[BT]` (its QDNAseq generator `scripts/swg_generate_qdnaseq_one_sample.R` accepts only 10 or 50 kb)", "`[BT]/scripts/swg_generate_qdnaseq_one_sample.R` line 165"],
    ["set C", "CNV source 'pkg' (ours): build / bins / pipeline", 676, "hg38 / QDNAseq 50 kb raw + fitted counts / BarrettsProgressionRisk `segmentRawData` (gamma2 250, cutoff 0.008) with hg38 adaptations → 632 features", "`[KC]_mm/cnv_pkg_C.csv` (632 feature columns); `docs/paper_plan_killcoyne_reconcile.md` lines 16-17, 368"],
    ["set C", "CNV source 'their' (Killcoyne shipped matrix): build / bins / pipeline", 676, "hg19 / 50 kb / the package's shipped training matrix (773 × 634), rows aligned to our samples → 634 features", "`[KC]_mm/cnv_their_C.csv` (634 feature columns); `docs/paper_plan_killcoyne_reconcile.md` lines 11, 366, 368"],
]
L += table(["Cohort", "Quantity", "n", "Value", "Source"], rows)
L += ["Depth is nominal: reads × modal read length over the hg38 chr1–22, X, Y length, with no correction for duplicates or mapping quality. The 'their' source has no read-level data of its own; it comes from the same sequencing runs.", ""]

# ---------- item 7
i7 = core["item7"]; mc = i7["mean_coverage"]
L += ["## 7. ACE-B (manifest only; no outcome data read)", ""]
rows = [
    ["samples / patients", i7["samples"], f"{i7['samples']} samples ({i7['sample_ids_unique']} distinct SampleID), {i7['patients']} PatientID", "`[ACEB]`; `core.json:item7.samples, patients`"],
    ["samples per patient", i7["samples_per_patient"]["n"], f"mean {f(i7['samples_per_patient']['mean'])}; median {f(i7['samples_per_patient']['median'], 0)}; range {f(i7['samples_per_patient']['min'], 0)}–{f(i7['samples_per_patient']['max'], 0)}", "`[ACEB]`; `core.json:item7.samples_per_patient`"],
    ["WSI (slides, scanner, resolution, tiling, extractor)", 0, "NOT AVAILABLE: the manifest has no slide fields and no ACE-B slides are on the cluster", f"`[ACEB]` columns {', '.join(c for c in i7['columns'] if c != 'Pathology')} (+ Pathology, not read); `/mnt/scratche/fast/fmlab/datasets/imaging` has no ACE-B directory; `core.json:item7.imaging_dirs_matching_ace`"],
    ["sequencing batches", i7["samples"], counts(i7["batches"]), "`[ACEB]` Batch; `core.json:item7.batches`"],
    ["sequencing pools (SLX)", i7["samples"], counts(i7["slx_pools"]), "`[ACEB]` SLX; `core.json:item7.slx_pools`"],
    ["depth per sample (`Mean_Coverage`, ×)", mc["n"], f"mean {f(mc['mean'])}; median {f(mc['median'])}; range {f(mc['min'])}–{f(mc['max'])}", "`[ACEB]` Mean_Coverage; `core.json:item7.mean_coverage`"],
]
for b, d in sorted(i7["mean_coverage_by_batch"].items()):
    rows.append([f"depth, {b}", d.get("n", 0), "—" if not d.get("n") else f"mean {f(d['mean'])}; median {f(d['median'])}; range {f(d['min'])}–{f(d['max'])}", f"`[ACEB]`; `core.json:item7.mean_coverage_by_batch`"])
rows += [["read length, genome build, bin size, CNV pipeline", 0, "NOT AVAILABLE in the manifest", "`[ACEB]` columns"]]
L += table(["Quantity", "n", "Value", "Source"], rows)
L += ["The ACE-B cases file and the cohort owner's outcome split were not read. The manifest's `Pathology` column was not read.", ""]
open("docs/dataset_description.md", "w").write("\n".join(L))
print("rendered", len(L))

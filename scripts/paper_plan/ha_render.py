"""Render the Results of docs/paper_horizon_answers.md from results/paper_final/horizon_answers/*.json and results/paper_final/horizons/h3_*.json;
the pre-specification above '## Results' is kept verbatim. Usage: python ha_render.py RESULTS_COMMIT"""
import json, os, sys
RC = sys.argv[1]; R = "results/paper_final/horizon_answers"; DOC = "docs/paper_horizon_answers.md"
J = lambda f: json.load(open(f"{R}/{f}.json"))
q0 = {f"{s}_{p}": J(f"q0_{s}_{p}") for s in ("their", "pkg") for p in ("pre", "pre_ndbe", "pre_nofallback")}; chk = J("q0_check"); ic = J("q0_intercept_only"); q2 = J("q2")
h3 = {s: json.load(open(f"results/paper_final/horizons/h3_{s}.json"))["results"] for s in ("their", "pkg")}
q4 = J("q4_search") if os.path.exists(f"{R}/q4_search.json") else None
f3 = lambda x: "—" if x is None else f"{x:.3f}"; sg = lambda x: "—" if x is None else f"{x:+.3f}"
ci = lambda c: "[—]" if not c or c[0] is None else f"[{c[0]:.3f}, {c[1]:.3f}]"; cis = lambda c: "[—]" if not c or c[0] is None else f"[{c[0]:+.3f}, {c[1]:+.3f}]"
def table(h, rows): return ["| " + " | ".join(h) + " |", "|" + "|".join("---" for _ in h) + "|"] + ["| " + " | ".join(map(str, r)) + " |" for r in rows] + [""]
ROWS = [("Clinical Only (baseline)", "L-CLIN"), ("CNV (replication)", "L-CNV"), ("WSI", "L-IMG"), ("Early fusion", "L-EARLY"), ("Inter fusion", "L-INTER"), ("Late fusion", "L-LATE")]
def main_table(key):
    H = q0[key]["units"]["sample"]["horizons"]; rows = [[nm] + [f"{f3(H[str(t)]['arms'][a]['ipcw']['auroc'])} {ci(H[str(t)]['arms'][a]['ipcw']['ci95'])}" for t in (1, 3, 5)] + ["pending"] * 3 for nm, a in ROWS]
    rows.append(["n cases / n controls (samples; patients)"] + [f"{H[str(t)]['n_cases']} / {H[str(t)]['n_controls']}; {H[str(t)]['n_case_patients']} / {H[str(t)]['n_control_patients']}" for t in (1, 3, 5)] + ["pending (slides being scanned)"] * 3)
    return table(["", "Internal 1-year", "Internal 3-year", "Internal 5-year", "ACE-B 1", "ACE-B 3", "ACE-B 5"], rows)
L = open(DOC).read().split("\n## Results")[0].rstrip().split("\n")
st4 = "NOT AVAILABLE" if (q4 is None or not q4["aceb_reads"]) else "PARTIAL"
L += ["", "## Results", "",
      f"Pre-specification commit 81473df; results commit {RC}. Scripts `scripts/paper_plan/ha_clin.py` (L-CLIN), `ha_metrics.py` (Q0/Q1), `ha_intercept.py` (added sanity check), `ha_q2.py` (Q2), `ha_search_shard.py` (Q4), `ha_figs.py`, `ha_render.py`; reused: `kv_cv.R` (d69de24), `hz_fit.R`, `hz_prep.py`, `hz_stack.py` (4d7efa9). All run through `scripts/cluster/campaign.sh` (prefixes hac, ham, hai, haq2, has, has2).",
      "Results: `results/paper_final/horizon_answers/q0_{their,pkg}_{pre,pre_ndbe,pre_nofallback}.json`, `q0_check.json`, `q0_intercept_only.json`, `q2.json`, `q4_search.json`; Q3 from `results/paper_final/horizons/h3_{their,pkg}.json` (4d7efa9). Figures `results/paper_final/horizon_answers/figs/*.{png,pdf,json}`. Row-level files on the cluster only, under `feasibility/paper_plan/killcoyne_mm/horizon_answers/` (L-CLIN predictions, Q2 patient list).", ""]
L += ["### Status", ""] + table(["Item", "Status"], [["Q0 table (internal)", "DONE; ACE-B pending (slides being scanned)"], ["Q1 best model by horizon", "DONE"], ["Q2 false positives and newly captured patients", "DONE (patient list by study number on the cluster only)"],
                                                     ["Q3 WSI vs CNV weighting (exploratory)", "DONE (outputs of the identical H3 specification reused)"], ["Q4 4× CNV and depth transfer", f"4× → external: NOT AVAILABLE (no 4× training data); ACE-B depth check {st4}"]])
L += ["### Q0. The table", "", "Discovery subset, all pre-event samples, CNV source their matrix. IPCW time-dependent AUROC [95% patient-bootstrap CI], per sample. Clinical Only = L-CLIN.", ""] + main_table("their_pre")
H = q0["their_pre"]["units"]["sample"]["horizons"]
L += [f"L-CLIN without grade (demographics only, sanity check): {', '.join(f'{t} y {f3(H[str(t)]['arms']['L-CLIN-nograde']['ipcw']['auroc'])} {ci(H[str(t)]['arms']['L-CLIN-nograde']['ipcw']['ci95'])}' for t in (1, 3, 5))}. It sits below 0.5, not near it; the intercept-only prediction (each sample's training-fold prevalence; check added after the pre-specification) scores {', '.join(f'{t} y {f3(ic['pre'][str(t)]['auroc'])} {ci(ic['pre'][str(t)]['ci95'])}' for t in (1, 3, 5))}, so the shortfall is the stratified-CV fold-prevalence artefact of `docs/paper_plan_killcoyne_cv.md` (folds of 3–4 P patients in 8), which a near-constant score inherits, not a reversed demographic signal.", "",
      "**Supplement: package features** (all pre-event samples).", ""] + main_table("pkg_pre") + ["**Supplement: NDBE pre-event samples only**, their matrix.", ""] + main_table("their_pre_ndbe") + ["**Supplement: NDBE pre-event samples only**, package features.", ""] + main_table("pkg_pre_ndbe") + \
     ["**Supplement: without the 4 fallback-endpoint progressors**, their matrix.", ""] + main_table("their_pre_nofallback")
rows = []
for s in ("their", "pkg"):
    U = q0[f"{s}_pre"]["units"]
    rows.append([s, "per sample, Harrell's C"] + [f"{f3(U['sample']['harrell'][a]['c'])} {ci(U['sample']['harrell'][a]['ci95'])}" for _, a in ROWS])
    for t in (1, 3, 5):
        Hp = U["patient"]["horizons"][str(t)]; rows.append([s, f"per patient {t} y ({Hp['n_cases']}/{Hp['n_controls']})"] + [f"{f3(Hp['arms'][a]['ipcw']['auroc'])} {ci(Hp['arms'][a]['ipcw']['ci95'])}" for _, a in ROWS])
    for t in (1, 3, 5):
        Hs = U["sample"]["horizons"][str(t)]; rows.append([s, f"unweighted {t} y"] + [f"{f3(Hs['arms'][a]['unweighted']['auroc'])} {ci(Hs['arms'][a]['unweighted']['ci95'])}" for _, a in ROWS])
L += ["**Secondary metrics** (all pre-event samples; patient unit = earliest pre-event NDBE sample).", ""] + table(["CNV source", "Metric"] + [a for _, a in ROWS], rows)
L += [f"Time check: the 777 sheet's `Months before final` (used) agrees with the Source Data for every patient (MOESM4 'Supporting data for Figure 2d': each patient's set of sample months is contained in the published set for {chk['41591_2020_1033_MOESM4_ESM.xlsx']['patients_months_subset_of_published']}/{chk['41591_2020_1033_MOESM4_ESM.xlsx']['patients_compared']} patients; MOESM11: {chk['41591_2020_1033_MOESM11_ESM.xlsx']['patients_months_subset_of_published']}/{chk['41591_2020_1033_MOESM11_ESM.xlsx']['patients_compared']}; the Source Data have no sample identifiers, so the check is per patient; `q0_check.json`).", ""]
# ---------- Q1
def best_rows(key):
    H = q0[key]["units"]["sample"]["horizons"]; rows = []
    for t in (1, 3, 5):
        h = H[str(t)]; b = h["best"]; o = h["arms"][b]["ipcw"]; dc = o.get("delta_vs_L-CNV"); dl = o.get("delta_vs_L-LATE")
        rows.append([f"{t} y", b, f"{f3(o['auroc'])} {ci(o['ci95'])}", "—" if not dc else f"{sg(dc['delta'])} {cis(dc['ci95'])} (p {dc.get('p_unadjusted', '—')}, max-T {dc.get('p_max_T', '—')})", "—" if not dl else f"{sg(dl['delta'])} {cis(dl['ci95'])} (p {dl.get('p_unadjusted', '—')}, max-T {dl.get('p_max_T', '—')})", " > ".join(h["ranking_ipcw"])])
    return rows
def delta_rows(key):
    H = q0[key]["units"]["sample"]["horizons"]; rows = []
    for t in (1, 3, 5):
        for _, a in ROWS:
            o = H[str(t)]["arms"][a]["ipcw"]; dl = o.get("delta_vs_L-LATE"); dc = o.get("delta_vs_L-CNV")
            fmt = lambda d: "—" if not d else f"{sg(d['delta'])} {cis(d['ci95'])}" + (f" (p {d['p_unadjusted']}, max-T {d['p_max_T']})" if "p_max_T" in d else "")
            rows.append([f"{t} y", a, f"{f3(o['auroc'])}", fmt(dl), fmt(dc)])
    return rows
Hb = q0["their_pre"]["units"]["sample"]["horizons"]; bests = {t: Hb[str(t)]["best"] for t in (1, 3, 5)}
lt = {t: Hb[str(t)]["arms"]["L-LATE"]["ipcw"] for t in (1, 3, 5)}
L += ["## Which is best model? 1/3/5 year?", "",
      f"**Answer.** On their CNV matrix the best arm is {(bests[1] + ' at all three horizons') if len(set(bests.values())) == 1 else ', '.join(f'{bests[t]} at {t} year{'s' if t > 1 else ''}' for t in (1, 3, 5))}: AUROC {', '.join(f3(lt[t]['auroc']) for t in (1, 3, 5))}, with L-EARLY second at every horizon. "
      f"Its gain over L-CNV is {', '.join(sg(lt[t]['delta_vs_L-CNV']['delta']) for t in (1, 3, 5))} (max-T p {', '.join(str(lt[t]['delta_vs_L-CNV']['p_max_T']) for t in (1, 3, 5))}); after selection adjustment no arm differs from L-LATE except L-CNV at 3 years (max-T p {Hb['3']['arms']['L-CNV']['ipcw']['delta_vs_L-LATE']['p_max_T']}). "
      "The order of the middle arms changes across horizons (rank orders below), but the top two do not; on NDBE samples only, L-EARLY is first at 1 and 3 years and L-LATE at 5 years. One line: late fusion is the best arm at every horizon, by margins that adjustment mostly cannot separate from early or intermediate fusion.", ""]
L += ["**Best arm per horizon**, their matrix, all pre-event samples.", ""] + table(["Horizon", "Best", "AUROC [CI]", "Δ vs L-CNV [CI] (p, max-T)", "Δ vs L-LATE [CI] (p, max-T)", "Ranking (IPCW AUROC)"], best_rows("their_pre"))
L += ["Package features:", ""] + table(["Horizon", "Best", "AUROC [CI]", "Δ vs L-CNV [CI] (p, max-T)", "Δ vs L-LATE [CI] (p, max-T)", "Ranking (IPCW AUROC)"], best_rows("pkg_pre"))
L += ["NDBE pre-event samples, their matrix:", ""] + table(["Horizon", "Best", "AUROC [CI]", "Δ vs L-CNV [CI] (p, max-T)", "Δ vs L-LATE [CI] (p, max-T)", "Ranking (IPCW AUROC)"], best_rows("their_pre_ndbe"))
L += ["**All arms, paired deltas**, their matrix, all pre-event samples (max-T over the 5 non-clinical arms; clinical arms have unadjusted CIs only).", ""] + table(["Horizon", "Arm", "AUROC", "Δ vs L-LATE [CI]", "Δ vs L-CNV [CI]"], delta_rows("their_pre"))
L += ["Package features, all pre-event samples:", ""] + table(["Horizon", "Arm", "AUROC", "Δ vs L-LATE [CI]", "Δ vs L-CNV [CI]"], delta_rows("pkg_pre"))
L += ["**Method.** `ha_metrics.py`: IPCW cumulative/dynamic AUROC (reverse Kaplan–Meier weights re-estimated per draw), 2,000 patient-bootstrap draws `RandomState(0)`, paired deltas on the same draws, within-patient swap permutation (2,000, seed 0) with single-step max-T (families in the pre-specification). Scores: mean over 10 repeats of out-of-fold probabilities. **Sources.** `cv/preds/` (kv_cv.R), `horizons/outer/inter_*` (hz_fit.R), `horizon_answers/clin_outer.csv` (ha_clin.py), `horizons/samples.csv` (hz_prep.py); `q0_*.json`.",
      "**Caveats.** The models were trained on ever-progression, so these are ever-progression scores ranked against near-term and later events. The 1-year cells rest on 29 case samples from 14 patients.", ""]
# ---------- Q2
def q2rows(src):
    rows = []
    for t in (1, 3, 5):
        for u in ("sample", "patient"):
            o = q2["results"][src][f"{t}|{u}"]; b = q2["best_by_horizon_their_pre"][str(t)]; f_ = b if b != "L-CNV" else "L-LATE"; x = o[f"{f_}_vs_L-CNV"]
            rows.append([f"{t} y", u, f"{o['n_cases']}/{o['n_controls']}", f_, f"{f3(o['L-CNV']['TPR'])} / {f3(o['L-CNV']['FPR'])}", f"{f3(o[f_]['TPR'])} / {f3(o[f_]['FPR'])}", f"{sg(x['dFPR'])} {cis(x['dFPR_ci95'])}", f"{sg(x['dTPR'])} {cis(x['dTPR_ci95'])}", f"{sg(x['NRI'])} {cis(x['NRI_ci95'])}"])
    return rows
def reclrows(src):
    rows = []
    for t in (1, 3, 5):
        b = q2["best_by_horizon_their_pre"][str(t)]; f_ = b if b != "L-CNV" else "L-LATE"; tab = q2["results"][src][f"{t}|sample"][f"{f_}_vs_L-CNV"]["reclassification"]; tp = q2["results"][src][f"{t}|patient"][f"{f_}_vs_L-CNV"]["reclassification"]
        for g in ("cases", "controls"):
            rows.append([f"{t} y", g] + [f"{tab[g][k]} ({tab[g + '_patients'][k]}) / {tp[g][k]}" for k in ("CNVneg_bestpos", "CNVpos_bestneg", "both_pos", "both_neg")])
    return rows
def grows(src):
    rows = []
    for t in (1, 3, 5):
        b = q2["best_by_horizon_their_pre"][str(t)]; f_ = b if b != "L-CNV" else "L-LATE"; G = q2["results"][src][f"{t}|sample"][f"{f_}_groups"]
        for g, x in G.items():
            rows.append([f"{t} y", g.replace("_", " "), f"{x['n_samples']} / {x['n_patients']}", ", ".join(f"{k} {v}" for k, v in x["pathology"].items()), "—" if x["time_median_years"] is None else f"{x['time_median_years']:.1f}", "—" if x["cx_raw_pkg_median"] is None else f"{x['cx_raw_pkg_median']:.0f}", f3(x["noise_varMAD_median"]), "—" if x["tissue_tiles_median"] is None else f"{x['tissue_tiles_median']:.0f}",
                         ", ".join(f"{k} {v}" for k, v in x["scanner"].items()), ", ".join(f"{k} {v}" for k, v in x["p53_ihc"].items()), ", ".join(f"{k} {v}" for k, v in x["killcoyne_risk_class_MOESM4"].items())])
    return rows
x3 = q2["results"]["their"]["3|sample"]["L-LATE_vs_L-CNV"]; x1 = q2["results"]["their"]["1|sample"]["L-LATE_vs_L-CNV"]; x5 = q2["results"]["their"]["5|sample"]["L-LATE_vs_L-CNV"]; g3 = q2["results"]["their"]["3|sample"]["L-LATE_groups"]
L += ["## Change in false positive rate: what patients does the best model capture that was previously missing?", "",
      f"**Answer.** The best arm is L-LATE at every horizon, so the comparison is L-LATE against L-CNV (their matrix). At each model's 80%-sensitivity threshold for that horizon, L-LATE has fewer false positives: ΔFPR {sg(x1['dFPR'])} {cis(x1['dFPR_ci95'])} at 1 year, {sg(x3['dFPR'])} {cis(x3['dFPR_ci95'])} at 3 years and {sg(x5['dFPR'])} {cis(x5['dFPR_ci95'])} at 5 years, with NRI {sg(x1['NRI'])}, {sg(x3['NRI'])} and {sg(x5['NRI'])}. "
      f"At 3 years it newly captures {g3['captured_cases_CNVneg_bestpos']['n_samples']} case samples from {g3['captured_cases_CNVneg_bestpos']['n_patients']} patients that L-CNV missed; most were low risk in Killcoyne's published classes ({', '.join(f'{k} {v}' for k, v in g3['captured_cases_CNVneg_bestpos']['killcoyne_risk_class_MOESM4'].items())}), with lower CNV complexity (median cx {g3['captured_cases_CNVneg_bestpos']['cx_raw_pkg_median']:.0f} vs {g3['cases_both_pos']['cx_raw_pkg_median']:.0f}) and fewer tissue tiles ({g3['captured_cases_CNVneg_bestpos']['tissue_tiles_median']:.0f} vs {g3['cases_both_pos']['tissue_tiles_median']:.0f}) than cases positive under both. "
      f"It clears {g3['cleared_controls_CNVpos_bestneg']['n_samples']} control samples from {g3['cleared_controls_CNVpos_bestneg']['n_patients']} patients. One line: late fusion mainly removes false positives; the cases it adds are CNV-quiet ones that the published model also called low risk.", ""]
L += ["**FPR, TPR and NRI**, their matrix (patient unit: earliest pre-event NDBE sample).", ""] + table(["Horizon", "Unit", "Cases/controls", "Best", "L-CNV TPR / FPR", "Best TPR / FPR", "ΔFPR [CI]", "ΔTPR [CI]", "NRI [CI]"], q2rows("their"))
L += ["Package features:", ""] + table(["Horizon", "Unit", "Cases/controls", "Best", "L-CNV TPR / FPR", "Best TPR / FPR", "ΔFPR [CI]", "ΔTPR [CI]", "NRI [CI]"], q2rows("pkg"))
L += ["**Reclassification**, their matrix: samples (patients with ≥ 1 such sample) / patient unit.", ""] + table(["Horizon", "Group", "CNV− / best+", "CNV+ / best−", "both +", "both −"], reclrows("their"))
L += ["Package features:", ""] + table(["Horizon", "Group", "CNV− / best+", "CNV+ / best−", "both +", "both −"], reclrows("pkg"))
L += ["**Newly captured and newly cleared**, their matrix, per sample (medians; counts).", ""] + table(["Horizon", "Group", "Samples / patients", "Pathology", "Years to event or censoring", "cx (raw)", "Noise (varMAD)", "Tissue tiles", "Scanner", "p53 IHC", "Killcoyne risk class (MOESM4)"], grows("their"))
L += [f"Study numbers: the per-patient list ({q2['patient_list_rows']} rows: study number, sample, pathology, time, cx, noise, tissue tiles, scanner, p53, published risk class, for every group, horizon and CNV source) is row-level patient data and stays on the cluster: `{q2['patient_list_path_cluster']}`. Published risk class matched for {q2['risk_class_matched']}/676 samples.", "",
      "Thresholds (median over the 100 fold × repeat fits): " + "; ".join(f"{t} y L-CNV {q2['results']['their'][f'{t}|sample']['thresholds']['L-CNV']['median']}, L-LATE {q2['results']['their'][f'{t}|sample']['thresholds']['L-LATE']['median']}" for t in (1, 3, 5)) + " (their matrix).", "",
      "**Method.** `ha_q2.py`: per model, horizon, repeat and outer fold, the threshold is the largest value with ≥ 80% sensitivity among that horizon's pre-event training cases on the inner out-of-fold predictions (`hz_fit.R` inner; L-CLIN inner from `ha_clin.py`); held-out calls, positive in ≥ 6 of 10 repeats; ΔFPR, ΔTPR and categorical NRI with 2,000 patient-bootstrap draws. **Sources.** As above, plus `horizons/slide_desc.csv`, `horizons/pkg_qc.csv`, `models/killcoyne_frozen_pkg_v1/cnv_scaling_setC.csv` (raw cx), `killcoyne_data_from_paper/41591_2020_1033_MOESM4_ESM.xlsx` (risk class); `q2.json`.",
      "**Caveats.** FPR and TPR describe this matched case–control sample, not a screening population (design caveat). The patient unit has 2 cases at 1 year. The group comparison is descriptive, with no test.", ""]
# ---------- Q3
r = {s: h3[s] for s in ("their", "pkg")}
L += ["## What's the weighting between WSI and CNV? Does this change between 1-year and 5-year?", "",
      f"**Answer (exploratory).** The WSI share c/(b+c) of the horizon-specific stack is about one half at every horizon: {', '.join(f'{f3(r['their'][str(t)]['ratio_mean'])} {ci(r['their'][str(t)]['ratio_ci95'])}' for t in (1, 3, 5))} at 1, 3 and 5 years on their matrix (package features {', '.join(f3(r['pkg'][str(t)]['ratio_mean']) for t in (1, 3, 5))}). "
      f"The 1-year minus 5-year difference is {sg(r['their']['ratio_1y_minus_5y']['delta'])} {cis(r['their']['ratio_1y_minus_5y']['ci95'])} (package {sg(r['pkg']['ratio_1y_minus_5y']['delta'])} {cis(r['pkg']['ratio_1y_minus_5y']['ci95'])}), so the pre-stated expectation of a higher WSI share at 1 year is not supported. "
      f"Held out, no horizon-specific stack beats the fixed 50/50 L-LATE (Δ {', '.join(sg(r['their'][str(t)]['exploratory_heldout']['delta']) for t in (1, 3, 5))}). One line: equal weighting is as good as fitted weighting at every horizon.", ""]
rows = [[("their matrix" if s == "their" else "package features"), f"{t} y", r[s][str(t)]["n_fits"], f"{f3(r[s][str(t)]['b_mean'])} {ci(r[s][str(t)]['b_ci95'])}", f"{f3(r[s][str(t)]['c_mean'])} {ci(r[s][str(t)]['c_ci95'])}", f"{f3(r[s][str(t)]['ratio_mean'])} {ci(r[s][str(t)]['ratio_ci95'])}",
         f"{f3(r[s][str(t)]['exploratory_heldout']['stack_ipcw_auroc'])} vs {f3(r[s][str(t)]['exploratory_heldout']['late_ipcw_auroc'])}: {sg(r[s][str(t)]['exploratory_heldout']['delta'])} {cis(r[s][str(t)]['exploratory_heldout']['delta_ci95'])}"] for s in ("their", "pkg") for t in (1, 3, 5)]
L += table(["CNV source", "Horizon", "Fits", "b (CNV) [CI]", "c (WSI) [CI]", "WSI share c/(b+c) [CI]", "Held-out stack vs L-LATE IPCW AUROC, Δ [CI]"], rows)
L += ["**Method.** Identical specification to `docs/paper_survival_horizons.md` H3 (pre-specified ee51db8, run 4d7efa9 with `hz_stack.py` TASK=h3_<src>); outputs reused, not recomputed. **Sources.** `results/paper_final/horizons/h3_their.json`, `h3_pkg.json`.",
      "**Caveats.** Exploratory; the 1-year stacks rest on few cases, which is why the 1-year interval is wide (0.02–0.81).", ""]
# ---------- Q4
L += ["## Does 4x CNV transfer to 7x external downsampled?", ""]
if q4:
    dd = q4["discovery_reads_outside_dna_seq_bam"]; grp = {}
    for k, v in dd.items():
        g = "/".join(k.split("/")[:9]); grp[g] = grp.get(g, 0) + v["files"]
    L += [f"**Answer.** 4× → external: NOT AVAILABLE (no 4× training data). The re-run search ({len(q4['shards'])} `lfs find` shards over `/mnt/scratche/fast/fmlab` and `/mnt/scratche/slow/fmlab`, {q4['n_entries_listed']:,} entries, {q4['n_read_files_total']:,} read files) finds no resequenced discovery data: the discovery-SLX read files outside `dna_seq_bam` are " + ", ".join(f"`{g}` ({c})" for g, c in grp.items()) + ", i.e. the older hg19 alignment of the same runs (probed nominal depth at most 1.02×, `results/paper_final/horizons/h4_bam_probe.json`) and broken symlinks to it. "
          + ("No ACE-B reads (SLX pools " + ", ".join(q4["aceb_slx_ids"]) + ") are on the cluster, so the label-blind depth check is NOT AVAILABLE." if not q4["aceb_reads"] else "ACE-B reads were found: " + "; ".join(f"`{k}` ({v['files']} files)" for k, v in q4["aceb_reads"].items()) + "; the depth check was not run in this task (PARTIAL)."), "",
          f"**Paths checked.** Roots `/mnt/scratche/fast/fmlab`, `/mnt/scratche/slow/fmlab`, every depth-2 directory as its own shard plus each depth-1 directory at `-maxdepth 1` (shard list `feasibility/closeout/ha_search.tsv`, `ha_search2.tsv`); patterns: read-file extensions `bam`, `cram`, `fastq[.gz]`, `fq[.gz]`, `sra`; discovery SLX ids from the 777 sheet ({q4['discovery_slx_ids']}); ACE-B SLX pools from the manifest; directory names with 4x/deep/reseq/high-depth ({q4['n_depth_named_dirs']} hits, all software or slide folders). Unreadable-path messages: {q4['find_stderr_lines']:,}.", ""]
else:
    L += ["**Answer.** NOT AVAILABLE: the search did not complete.", ""]
L += ["**Method.** `ha_search_shard.py` (same code as `hz_search_shard.py`, fresh output folder). **Sources.** `q4_search.json`.", "**Caveats.** Only the mounted lab scratch is visible; data held by the sequencing core or the cohort owner off-cluster cannot be seen.", "",
      "### Design caveat", "", "The discovery cohort is a matched case–control design: non-progressors had at least 3 years of follow-up and progressors at least 1. AUROCs are valid; absolute risks, calibration and positive predictive values are not, and none is reported.", "",
      "### Discrepancies found", "",
      "- L-CLIN without grade is not near 0.5 (0.29–0.35); the intercept-only prediction scores the same, so the stratified-CV fold-prevalence artefact, not the demographics, sets it. The same artefact pulls L-CLIN below 0.5 at 3 and 5 years.",
      "- The question heading says '7x external'; the ACE-B manifest gives median 5.3× (range 3.3–14.9×), as found in `docs/dataset_description.md` item 7.",
      "- The Source Data carry no sample identifiers for `Months before final`, so the time check is per patient (all 80 agree).", "",
      "### Not done", "",
      "- ACE-B columns: pending (slides being scanned); L-CLIN and L-INTER have no frozen model for ACE-B.",
      "- Q4 depth check (no ACE-B reads on the cluster)." if st4 == "NOT AVAILABLE" else "- Q4 depth check not run in this task.",
      "- The per-patient study-number list is not in this document (cluster-only, as above).",
      "- Q4 figure (probability stability by depth): not produced, no depth check.", ""]
open(DOC, "w").write("\n".join(L) + "\n"); print("rendered", len(L))

"""Append results to docs/paper_triage.md below the pre-specification (f5944d2); text above the closing '---' line is not changed.
Usage: python tr_render.py RESULTS_COMMIT. Inputs: results/paper_final/triage/{their,pkg}_{pre,pre_ndbe}.json and answers.json."""
import json, sys
RC = sys.argv[1]; R = "results/paper_final/triage"; DOC = "docs/paper_triage.md"
J = {t: json.load(open(f"{R}/{t}.json")) for t in ("their_pre", "their_pre_ndbe", "pkg_pre", "pkg_pre_ndbe")}; ANS = json.load(open(f"{R}/answers.json"))
f3 = lambda x: "—" if x is None else f"{x:.3f}"; ci = lambda c: f" [{c[0]:.3f}, {c[1]:.3f}]"; cis = lambda c: f" [{c[0]:+.3f}, {c[1]:+.3f}]"; sg = lambda x: f"{x:+.3f}"; pc = lambda x: f"{100 * x:.1f}%"
bp = lambda p: "< 0.0005" if p == 0 else f"{p}"
LAB = {"their_pre": "their, all pre-event", "their_pre_ndbe": "their, NDBE pre-event", "pkg_pre": "package, all pre-event", "pkg_pre_ndbe": "package, NDBE pre-event"}
SN = {"triage_LATE": "Triage, L-LATE in middle band", "triage_CNV": "Triage, L-CNV in middle band", "A": "(A) sequence everyone, L-CNV", "B": "(B) sequence everyone, L-LATE", "C": "(C) H&E only, L-IMG"}
doc = open(DOC).read(); head = doc[: doc.rindex("\n---")].rstrip("\n") + "\n\n---\n"
L = ["", "## Results", "", f"Pre-specification commit f5944d2; results commit {RC}. Scripts `scripts/paper_plan/tr_triage.py` (Slurm via `scripts/cluster/campaign.sh`, prefix tr), `tr_fig.py`, `tr_render.py`. Aggregates `results/paper_final/triage/{{their,pkg}}_{{pre,pre_ndbe}}.json`; per-sample bands, calls and missed-patient rows on the cluster only (`feasibility/paper_plan/killcoyne_mm/triage/`). Figure `~/Downloads/be_paper_figs/v3/13_F_triage.{{pdf,png}}`. All CIs: 2,000 patient-bootstrap draws; p unadjusted unless stated.", ""]
pcm = J["their_pre"]["primary_criterion"]
L += ["### Status", "", "| Item | Status |", "|---|---|", "| 1 Non-inferiority of histology; fusion vs histology | DONE |", "| 2 Triage strategy and comparators | DONE |", "| 3 Strategy metrics, paired differences, middle band | DONE |", "| 4 Prevalence projection | DONE |",
      f"| Primary success criterion | {'MET' if pcm['met'] else 'NOT MET'}: lower bound Δ sensitivity vs (B) {f3(pcm['one_sided_95_lower'])} (> −0.05: {'yes' if pcm['noninferior_sens'] else 'no'}); share sequenced {pc(pcm['share_sequenced'])} (< 70%: {'yes' if pcm['share_below_0.70'] else 'no'}) |", ""]
L += ["### Answers", ""] + [f"{i}. {ANS[str(i)]}" for i in (1, 2, 3, 4)] + [""]
L += ["### 1. Histology vs CNV (fold-stratified per-sample AUROC, mean over 10 repeats)", "", "| CNV source, population | L-IMG | L-CNV | Δ (L-IMG − L-CNV) [95% CI] | One-sided 95% lower bound | Non-inferior at −0.05 |", "|---|---|---|---|---|---|"]
for t, j in J.items():
    a = j["section1"]["auroc"]; d = j["section1"]["img_vs_cnv"]
    L.append(f"| {LAB[t]} | {f3(a['L-IMG']['value'])}{ci(a['L-IMG']['ci95'])} | {f3(a['L-CNV']['value'])}{ci(a['L-CNV']['ci95'])} | {sg(d['delta'])}{cis(d['ci95'])} | {f3(d['one_sided_95_lower'])} | {'yes' if d['non_inferior'] else 'no'} |")
L += ["", "**Fusion vs histology alone** (Δ AUROC vs L-IMG; max-T adjusted over the three).", "", "| CNV source, population | Model | AUROC | Δ vs L-IMG [95% CI] | p unadjusted | p max-T adjusted |", "|---|---|---|---|---|---|"]
for t, j in J.items():
    for m, v in j["section1"]["fusion_vs_img"].items(): L.append(f"| {LAB[t]} | {m} | {f3(j['section1']['auroc'][m]['value'])} | {sg(v['delta'])}{cis(v['ci95'])} | {bp(v['p_unadjusted'])} | {bp(v['p_maxT_adjusted'])} |")
L += ["", "### 2–3. Strategies", ""]
for t, j in J.items():
    L += [f"**{LAB[t]}** ({j['n_samples']} samples, {j['n_patients']} patients; {j['n_progressor_samples']} progressor samples from {j['n_progressor_patients']} patients).", "",
          "| Strategy | Sensitivity [95% CI] | Specificity [95% CI] | Progressor samples missed | Progressor patients with ≥ 1 / all samples missed | Δ sens vs A | Δ spec vs A | Δ sens vs B (one-sided 95% lower) | Δ spec vs B |", "|---|---|---|---|---|---|---|---|---|"]
    for s, o in j["strategies"].items():
        va, vb = o.get("vs_A"), o.get("vs_B")
        L.append(f"| {SN[s]} | {f3(o['sensitivity']['value'])}{ci(o['sensitivity']['ci95'])} | {f3(o['specificity']['value'])}{ci(o['specificity']['ci95'])} | {o['missed_progressor_samples']} | {o['progressor_patients_with_any_missed']} / {o['progressor_patients_all_missed']} | "
                 + (f"{sg(va['sens']['delta'])}{cis(va['sens']['ci95'])}" if va else "—") + " | " + (f"{sg(va['spec']['delta'])}{cis(va['spec']['ci95'])}" if va else "—") + " | "
                 + (f"{sg(vb['sens']['delta'])}{cis(vb['sens']['ci95'])} ({f3(vb['sens']['one_sided_95_lower'])})" if vb else "—") + " | " + (f"{sg(vb['spec']['delta'])}{cis(vb['spec']['ci95'])}" if vb else "—") + " |")
    b = j["bands"]; fo = j["folds"]
    L += ["", "| Band | All samples [95% CI] | Progressor samples | Non-progressor samples |", "|---|---|---|---|"] + [f"| {nm} | {pc(b[k]['all']['value'])} [{pc(b[k]['all']['ci95'][0])}, {pc(b[k]['all']['ci95'][1])}] | {pc(b[k]['progressor']['value'])} | {pc(b[k]['non_progressor']['value'])} |" for k, nm in (("low", "Low (cleared)"), ("mid", "Middle (sequenced)"), ("high", "High (flagged)"))]
    cu = fo["cutoffs"]; L += ["", f"Folds without a middle band: {fo['no_middle_band']}/100; t_seq fallbacks (< 5 training middle-band cases): L-LATE {fo['t_seq_fallback']['L-LATE']}/100, L-CNV {fo['t_seq_fallback']['L-CNV']}/100. Cut-offs, median (range): t_low {f3(cu['t_low']['median'])} ({f3(cu['t_low']['range'][0])}–{f3(cu['t_low']['range'][1])}), c* {f3(cu['c_star']['median'])} ({f3(cu['c_star']['range'][0])}–{f3(cu['c_star']['range'][1])}), t_seq L-LATE {f3(cu['t_seq_L-LATE']['median'])}, t_seq L-CNV {f3(cu['t_seq_L-CNV']['median'])}.", ""]
L += ["**Middle band only: does CNV discriminate where histology is uncertain?** (fold-stratified AUROC on held-out middle-band samples, mean over repeats).", "", "| CNV source, population | L-IMG | L-CNV | L-LATE | Δ L-CNV − L-IMG [95% CI], p | Δ L-LATE − L-IMG [95% CI], p |", "|---|---|---|---|---|---|"]
for t, j in J.items():
    m = j["middle_band_auroc"]; L.append(f"| {LAB[t]} | {f3(m['L-IMG']['value'])}{ci(m['L-IMG']['ci95'])} | {f3(m['L-CNV']['value'])}{ci(m['L-CNV']['ci95'])} | {f3(m['L-LATE']['value'])} | {sg(m['CNV_minus_IMG']['delta'])}{cis(m['CNV_minus_IMG']['ci95'])}, {bp(m['CNV_minus_IMG']['p_unadjusted'])} | {sg(m['LATE_minus_IMG']['delta'])}{cis(m['LATE_minus_IMG']['ci95'])}, {bp(m['LATE_minus_IMG']['p_unadjusted'])} |")
L += ["", "**Band characteristics** (pooled over sample × repeat assignments, each 1/10; their matrix, all pre-event samples; bands depend on L-IMG only).", "", "| Band | Samples (sum of 1/10) | Progressor share | Raw cx, mean (SD) | Tissue tiles, mean (SD); median | NDBE / ID / LGD | Scanner C13210 share |", "|---|---|---|---|---|---|---|"]
for t in ("their_pre", "their_pre_ndbe"):
    for k, nm in (("low", "Low"), ("mid", "Middle"), ("high", "High")):
        c = J[t]["band_characteristics"][k]
        if c is None: continue
        g = c["grade_share"]; L.append(f"| {nm} ({LAB[t].split(', ')[1]}) | {c['sample_repeats_div10']:.1f} | {f3(c['progressor_share'])} | {c['cx_raw_mean']:.1f} ({c['cx_raw_sd']:.1f}) | {c['tiles_mean']:.0f} ({c['tiles_sd']:.0f}); {c['tiles_median']:.0f} | {f3(g['NDBE'])} / {f3(g['ID'])} / {f3(g['LGD'])} | {f3(c['scanner_C13210_share'])} |")
L += ["", "### 4. Prevalence projection (not observed)", "", "Share of samples in each band if progressor samples made up 2%, 5% or 10% of samples, holding each class's band rates fixed (triage bands depend on L-IMG only).", "", "| CNV source, population | Prevalence | Sequenced (middle) [95% CI] | Cleared (low) | Flagged (high) |", "|---|---|---|---|---|"]
for t in ("their_pre", "their_pre_ndbe"):
    for k, q in J[t]["prevalence_projection"].items():
        L.append(f"| {LAB[t]} | {pc(q['prevalence'])}{' (observed)' if k == 'observed' else ''} | {pc(q['mid']['value'])} [{pc(q['mid']['ci95'][0])}, {pc(q['mid']['ci95'][1])}] | {pc(q['low']['value'])} | {pc(q['high']['value'])} |")
L += ["", "### Figure", "", f"`13_F_triage`: (a) sample flow for their matrix, all pre-event samples, triage with L-LATE; a sample's band is its most frequent band over the 10 repeats (ties → middle), which equals its band in all 10 repeats for {pc(J['their_pre']['flow']['modal_band_equals_all_repeats_share'])} of samples; (b) projected share sequenced against prevalence.", "",
      "### Deviations and caveats", ""] + [f"- {d}" for d in ANS.get("deviations", [])] + [
      "- Matched case–control cohort (about 28% progressor samples); sensitivities and specificities are per sample, the label is patient status shared by all of a patient's samples, and absolute numbers sequenced at other prevalences are projections.",
      "- The ≥ 6-of-10 vote and the band shares use the same stored predictions; bootstrap CIs hold calls fixed and resample patients (thresholds are not re-chosen per draw), so they do not include threshold-selection variability.",
      "- Triage bands are set by L-IMG only, so they are the same for both CNV sources; only the middle-band call changes.", ""]
open(DOC, "w").write(head + "\n".join(L) + "\n"); print("rendered")

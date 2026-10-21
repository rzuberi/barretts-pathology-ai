"""Exploration E1: render results/paper_final/explore_E1/*.json into docs/paper_explore_E1.md (everything below '## Results' is regenerated)."""
import json, os, sys
T = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); RD = T + "/results/paper_final/explore_E1"; DOC = T + "/docs/paper_explore_E1.md"
J = lambda f: json.load(open(f"{RD}/{f}.json")) if os.path.exists(f"{RD}/{f}.json") else None
L = []; A = L.append
def f3(x): return "–" if x is None else (f"{x:+.3f}" if isinstance(x, (int, float)) else str(x))
def v3(x): return "–" if x is None else (f"{x:.3f}" if isinstance(x, (int, float)) else str(x))
def ci(c, fmt="{:+.3f}"): return "–" if not c or c[0] is None else f"[{fmt.format(c[0])}, {fmt.format(c[1])}]"
def d1(d, k="delta"): return f"{d[k]:+.1f} {ci(d['ci95'], '{:+.1f}')}" if d and d.get(k) is not None else "–"
def d3(d): return f"{d['delta']:+.3f} {ci(d['ci95'])}" if d and d.get("delta") is not None else "–"
SUMMARY_TEXT = """### Summary

**Why non-progressors get flagged.**
- **False flags sit with patients, not single endoscopies.** Under R49, an ever-flagged non-progressor (NP) patient has 2.8 flagged endoscopies on average. Only 21% of these patients are flagged exactly once, and 57% of their endoscopies are flagged. The 10 most-flagged NP patients account for 46% of all false-flagged NP endoscopies; the 20 most-flagged account for 79%. Between patients, the endoscopy H&E score of non-progressors has an ICC of 0.21. A rule that clears single endoscopies therefore rarely clears a patient: patient-level false flags fall slowly even when endoscopy specificity rises.
- **More looks, more flags.** NP endoscopies flagged by R49 carry more samples than unflagged ones (2.27 against 1.88; +0.39 [+0.05, +0.74]). Flagged NP patients have more endoscopies (4.9 against 4.0) and more samples (10.2 against 7.7) than NP patients who are never flagged. The maximum over more samples and more endoscopies inflates the chance of a false flag.
- **Biology that looks like risk.** Flagged NP patients have longer Barrett's segments (Prague M 5.1 against 3.6 cm; +1.6 [+0.5, +2.7]) and a higher highest sheet grade (+0.54 [+0.07, +0.97]; more ID/LGD). What separates false-positive NP endoscopies from true-positive progressor endoscopies is mostly what the model cannot see in this cohort: aberrant p53 (0.7% against 22%), a later HGD+ report, and the H&E and copy-number scores themselves (both lower in false positives).
- **Slide and study artefacts.** Flagged NP patients more often have slides from the C13210 scanner (25% against 9%) and a higher background-tile share (0.060 against 0.019). They are less often sequenced in batches holding only one class (41% against 82%; −0.41 [−0.62, −0.18]). The "marker on slide" note is on 7.5% of false-positive NP endoscopies and 0.3% of true negatives. Only 11 NP patients are never flagged under R49, so these patient contrasts rest on a small group.
- **Inside the H&E score, false positives are a diffuse, moderate shift, not a few odd tiles.** The logit decomposes exactly into tile contributions. In false-positive slides, 53% of tiles push the score up (true negatives 40%, true positives 63%). The positive evidence is *less* concentrated than in true negatives (top-16 tiles hold 29% of the positive contribution against 38%). Background, edge, blurred and ink tiles add almost nothing to the score (background contribution +0.05 logit). Removing them, trimming, or using the median moves false-positive and true-positive scores alike (all differences in score change cross 0).

**Fixes (nested, against R49 and P1 90%).**
- **Robust tile aggregates do not help.** Means over non-background or core tiles, trimmed means and medians leave sample AUROC unchanged or lower (0.846 / 0.804 / 0.846 / 0.837 against 0.847). Within R49 they change NP patients flagged by −0.1 to +0.4.
- **Single-feature vetoes and second-stage filters on flagged endoscopies** raise endoscopy specificity, up to about 0.64 from 0.56. But among variants losing at most 0.5 progressor patients they remove at most about 1 NP patient (typically at a cost of 0.2–0.4 progressor patients); larger reductions cost more progressors. Their held-out AUROC among flagged endoscopies is 0.68–0.75. The apparent (non-nested) versions look much better (down to 25 NP patients with clinical features), which shows how easily these filters overfit.
- **Most promising deployable change: penalise the endoscopy maximum for the number of samples** (score = max logit − c·log(samples), c chosen on training patients).
  - Within R49 (keeping training sensitivity within 0.02): NP patients flagged 30.8 against 31.8 (−1.0 [−1.8, −0.3]); progressor patients caught 30.7 against 31.1 (−0.4 [−1.3, 0.0]). It also sequences fewer samples (1.75 against 1.94 per patient).
  - Applied to H&E alone, without sequencing: caught 30.8, NP flagged 32.6, against P1's 31.1 / 36.0. That is −3.4 [−6.6, −0.4] NP against P1 at 90%, but only −2.3 [−5.7, +1.4] against P1 matched to the same endoscopy sensitivity. Endoscopy AUROC rises from 0.837 to 0.850 (+0.013 [−0.011, +0.036]).
  - None of these clears the bar of fewer NP patients with no progressor loss and an interval excluding zero.
- **Not deployable, but informative about confounding.** A filter that knows only the progressor share of the sample's sequencing batch, computed from training patients only, removes 2.9 [1.4, 4.8] NP patients from R49 with no progressor loss. Using batch shares computed from all labels, a leak, removes 12 NP patients. Slide metadata (scanner, slide age, biopsy year) removes none. Sequencing-batch structure carries label information that the H&E-driven flags partly track. This is a confounding risk for transport to ACE-B, not a fix.

"""
NOTES_TEXT = """### Implementation notes

- **Population and rules.** These are exactly round 8's: `rv8b_step3.py` and `rv8_landscape.py` are executed unchanged up to their dispatch. R49 = P6 t_low 95% / t_high 90%; P1 = max L-IMG at 90% training endoscopy sensitivity. The check reproduces round 8 (31.1 / 31.8 and 31.1 / 36.0).
- **Endoscopy classes and covariates.** FP-NP / TN-NP / TP-P / FN-P are counted per repeat. Means are weighted by the share of repeats in each class. Sample covariates come from the endoscopy's top-H&E sample (repeat-mean outer L-IMG). Database grade at endoscopy = highest confirmed grade in Barrett's database reports within 14 days of the sample's exact date (rv7_round7.py linkage). Later DB group = highest grade more than 14 days after the last sequenced endoscopy (R54 logic). Same-block labels come from round-7 lookup Q1. Slide notes come from `slide_matching` (`Slide Notes`). Batch composition uses the sheet `Batch`; the batch progressor share uses all labels, so it is descriptive only.
- **Tile scores.** u_t = w·(e_t − μ)/σ, with w, μ, σ from the exact refit of each outer or inner fit, so the sample logit = b0 + mean_t u_t. Tile flags come from per-tile pixel statistics at the stored positions (`ex_e1_px.py`; no pixels saved):
  - background: > 50% white pixels (R55);
  - edge: non-background with tissue-mask share < 0.8 in the 3 × 3-tile neighbourhood, or > 25% white;
  - blur: non-background below the 10th percentile of Laplacian variance;
  - ink: > 2% saturated green-blue pixels (almost none found).
- **Nesting.** Training-side quantities for each (repeat, outer fold) come from training patients only:
  - R49 / P1 flags of training endoscopies: from inner out-of-fold L-IMG / L-LATE and the fold's own cut-offs;
  - tile summaries of training samples: under the inner fits whose held-out fold contains them.
  - Vetoes choose the feature direction and threshold so that at most eps of training flagged progressor endoscopies are vetoed. The "auto" veto picks the feature per fold too.
  - Filters are L2 logistic regressions (C = 1) on standardised features plus the gate score, trained on training flagged endoscopies. Their threshold comes from patient-grouped 5-fold out-of-fold probabilities within training.
  - Score substitutions recompute P1 thresholds and the R49 cut-offs from the substituted inner scores.
  - The sample-count penalty picks c (grid 0–3) by training specificity at 90% training sensitivity (P1), or by the largest NP reduction with training R49 sensitivity within eps (R49).
- **Vetoed endoscopies keep their sequencing.** Δ for vetoes and filters therefore leaves samples sequenced unchanged.
- **Apparent (optimistic) values.** These set the threshold or fit the model on the evaluated endoscopies of each repeat.
- **Matched comparison.** P1 matched = P1's threshold re-matched in every repeat and draw to the rule's realised endoscopy sensitivity (rv8b S1 (i)).
- **Multiplicity.** 226 nested variants were scored. The selected rows in the matched table are the best of many and are optimistic as a group.
- **Privacy.** Only aggregates are committed. Row-level tile scores, pixel statistics, coefficients and calls stay on the cluster (`killcoyne_mm/explore_E1/`).
- **Figures.** Local only, in `paper/figures_explore_E1/`: e1_char_r49.png, e1_char_p1.png, e1_tiles_r49.png, e1_fixes.png.
"""
A("## Results\n")
A("Scripts `scripts/paper_plan/ex_e1_coef.R`, `ex_e1_px.py`, `ex_e1_main.py`, `ex_e1_analysis.py`; this section is rendered by `ex_e1_render.py` from `results/paper_final/explore_E1/`. "
  "Exploratory: many comparisons, nothing adjusted, nothing confirmatory. Single model = per-repeat values averaged over the 10 repeats. Intervals: paired patient bootstrap, 2,000 draws (`RandomState(0)`), thresholds and calls fixed.\n")
P = J("prep"); U = J("u_check")
A(SUMMARY_TEXT)
if U: A(f"Checks: refitted H&E model coefficients reproduce every stored outer and inner prediction (max |Δ prediction| 1e-16, λ identical; `ex_e1_coef.R`); the tile decomposition b0 + mean_t u_t reproduces the stored logits (max |Δ logit| outer {U['check']['max_abs_logit_diff_vs_stored']['outer']:.1e} over {U['check']['n_checked']['outer']} sample x repeat cells, inner {U['check']['max_abs_logit_diff_vs_stored']['inner']:.1e} over {U['check']['n_checked']['inner']}). Tiles: {P['prep']['tiles'] if P else '?'} stored UNI2-h tile positions re-read for pixel statistics; tile flags (share of all tiles): " + (", ".join(f"{k} {v}" for k, v in P['prep']['tile_flags_share_all_tiles'].items()) if P else "") + ".\n")
c1 = J("char")
if c1:
    R = c1["E1_1"]; A("### E1.1 Who gets flagged\n")
    ch = R["check_round8"]; A(f"Check against round 8: R49 caught {ch['R49']['caught']} / NP flagged {ch['R49']['np_flagged']} (31.1 / 31.8); P1 90% {ch['P1 90%']['caught']} / {ch['P1 90%']['np_flagged']} (31.1 / 36.0).\n")
    for rn in ("R49", "P1 90%"):
        r = R["rules"][rn]; cnt = r["endoscopy_class_counts_mean_over_repeats"]
        A(f"**{rn}.** Endoscopies per repeat (mean): false-positive NP {cnt['FP-NP']}, true-negative NP {cnt['TN-NP']}, true-positive P {cnt['TP-P']}, false-negative P {cnt['FN-P']}.\n")
        A("| Covariate | FP-NP | TN-NP | TP-P | Δ FP−TN [95% CI] (std.) | Δ FP−TP [95% CI] (std.) | AUROC TP vs FP |"); A("|---|---|---|---|---|---|---|")
        items = sorted(r["endoscopy"].items(), key=lambda kv: -abs(kv[1]["contrasts"]["FP-NP - TP-P"].get("standardised") or 0))
        for nm, o in items:
            g = o["groups"]; a_ = o["contrasts"]["FP-NP - TN-NP"]; b_ = o["contrasts"]["FP-NP - TP-P"]
            A(f"| {nm} | {v3(g['FP-NP']['mean'])} | {v3(g['TN-NP']['mean'])} | {v3(g['TP-P']['mean'])} | {f3(a_['delta'])} {ci(a_['ci95'])} ({f3(a_['standardised'])}) | {f3(b_['delta'])} {ci(b_['ci95'])} ({f3(b_['standardised'])}) | {v3(o['auroc_TP-P_vs_FP-NP'])} |")
        A("")
        pc = r["patient_class_counts"]; A(f"Patient level ({rn}; weighted by share of repeats flagged): NP flagged {pc['NP flagged']:.1f}, NP never flagged {pc['NP never flagged']:.1f}, P caught {pc['P caught']:.1f}, P missed {pc['P missed']:.1f}.\n")
        A("| Patient covariate | NP flagged | NP never | P caught | Δ NP flagged − never [95% CI] (std.) |"); A("|---|---|---|---|---|")
        for nm, o in r["patient"].items():
            g = o["groups"]; a_ = o["contrasts"]["NP flagged - NP never flagged"]
            A(f"| {nm} | {v3(g['NP flagged']['mean'])} | {v3(g['NP never flagged']['mean'])} | {v3(g['P caught']['mean'])} | {f3(a_['delta'])} {ci(a_['ci95'])} ({f3(a_['standardised'])}) |")
        A("")
    pe = R.get("persistence")
    if pe:
        A("**Persistence of false flags.**\n"); A("| Quantity | R49 | P1 90% |"); A("|---|---|---|")
        fr = {rn: R["rules"][rn]["flag_repetition"]["non_progressor"] for rn in ("R49", "P1 90%")}; frp = {rn: R["rules"][rn]["flag_repetition"]["progressor"] for rn in ("R49", "P1 90%")}
        A(f"| Flagged endoscopies per ever-flagged NP patient | {fr['R49']['flagged_endoscopies_per_ever_flagged_patient']} | {fr['P1 90%']['flagged_endoscopies_per_ever_flagged_patient']} |")
        A(f"| Share of ever-flagged NP patients flagged exactly once | {fr['R49']['share_of_ever_flagged_with_exactly_one_flag']} | {fr['P1 90%']['share_of_ever_flagged_with_exactly_one_flag']} |")
        A(f"| Share of an ever-flagged NP patient's endoscopies that are flagged | {fr['R49']['share_of_endoscopies_flagged_among_ever_flagged']} | {fr['P1 90%']['share_of_endoscopies_flagged_among_ever_flagged']} |")
        A(f"| (Progressors: share of a caught patient's endoscopies flagged) | {frp['R49']['share_of_endoscopies_flagged_among_ever_flagged']} | {frp['P1 90%']['share_of_endoscopies_flagged_among_ever_flagged']} |")
        for rn in ("R49", "P1 90%"): pass
        A(f"| Share of all false-flagged NP endoscopies from the 10 / 20 most-flagged NP patients | {pe['R49_false_flag_concentration']['share_of_false_flags_from_top_10_np_patients']} / {pe['R49_false_flag_concentration']['share_from_top_20']} | {pe['P1 90%_false_flag_concentration']['share_of_false_flags_from_top_10_np_patients']} / {pe['P1 90%_false_flag_concentration']['share_from_top_20']} |")
        A("")
        a_, b_ = pe["icc_endoscopy_hne_score_non_progressor"], pe["icc_endoscopy_hne_score_progressor"]
        A(f"Between-patient share of variance (ICC) of the endoscopy H&E score: non-progressors {a_['value']} {ci(a_['ci95_200_draws'], '{:.3f}')}, progressors {b_['value']} {ci(b_['ci95_200_draws'], '{:.3f}')} (patients with ≥ 2 pre-event endoscopies; interval from 200 patient-bootstrap draws).\n")
    q = R["np_flags_P1_vs_R49"]; A(f"NP endoscopy flags per repeat: flagged by both P1 90% and R49 {q['both']}, by P1 only (cleared by sequencing) {q['P1 only (cleared by sequencing)']}, by R49 only {q['R49 only']}.\n")
t2 = J("tiles")
if t2:
    R = t2["E1_2"]; A("### E1.2 Inside the H&E score\n")
    A(f"Logit decomposition check (b0 + mean tile score = stored outer logit), max |difference| × 1000: {R['check_logit_decomposition_max_abs_diff']}.\n")
    for rn in ("R49", "P1 90%"):
        A(f"**{rn}** (top-H&E sample of each endoscopy, outer model; per-repeat class means averaged):\n"); A("| Tile summary | FP-NP | TN-NP | TP-P | Δ FP−TP [95% CI] (std.) | AUROC TP vs FP |"); A("|---|---|---|---|---|---|")
        for nm, o in R["rules"][rn].items():
            g = o["groups"]; b_ = o["contrasts"]["FP-NP - TP-P"]
            A(f"| {nm} | {v3(g['FP-NP']['mean'])} | {v3(g['TN-NP']['mean'])} | {v3(g['TP-P']['mean'])} | {f3(b_['delta'])} {ci(b_['ci95'])} ({f3(b_['standardised'])}) | {v3(o['auroc_TP-P_vs_FP-NP_per_repeat_mean'])} |")
        A("")
sm = J("summary")
if sm:
    R = sm["E1_3_summary"]; A("### E1.3 Fixes\n")
    A(f"{R['n_fixes_evaluated']} nested rule variants were scored. Deployable variants (no batch, slide-metadata or label information) ranked by Δ non-progressor patients flagged against R49, among those with Δ progressor patients caught ≥ −0.5 (top 25; all rows in `summary.json`):\n")
    A("| Rank | Rule | Caught | NP flagged | Endo sens / spec | Seq / patient | Δ NP flagged vs R49 [CI] | Δ caught vs R49 [CI] | Δ NP flagged vs P1 [CI] | Δ caught vs P1 [CI] | Apparent (optimistic) caught / NP flagged |"); A("|---|---|---|---|---|---|---|---|---|---|---|")
    rows = sorted([r for r in R["all"] if r["d_caught_vs_R49"] is not None and r["d_caught_vs_R49"] >= -0.5 and r.get("deployable", True)], key=lambda r: r["d_np_vs_R49"])
    for i, r in enumerate(rows[:25]):
        ap = r.get("apparent_optimistic"); aps = f"{ap['progressor_patients_caught']:.1f} / {ap['non_progressor_patients_flagged']:.1f}" if ap else "–"
        A(f"| {i + 1} | {r['label']} | {r['caught']:.1f} | {r['np_flagged']:.1f} | {r['endo_sens']:.3f} / {r['endo_spec']:.3f} | {r['seq_per_patient']:.2f} | {r['d_np_vs_R49']:+.1f} {ci(r['d_np_vs_R49_ci'], '{:+.1f}')} | {r['d_caught_vs_R49']:+.1f} {ci(r['d_caught_vs_R49_ci'], '{:+.1f}')} | {r['d_np_vs_P1']:+.1f} {ci(r['d_np_vs_P1_ci'], '{:+.1f}')} | {r['d_caught_vs_P1']:+.1f} {ci(r['d_caught_vs_P1_ci'], '{:+.1f}')} | {aps} |")
    A("")
    nd = [r for r in R["all"] if not r.get("deployable", True)]
    A("Not deployable (they use sequencing-batch composition or slide metadata; shown to measure confounding, not as fixes):\n")
    A("| Rule | Caught | NP flagged | Δ NP flagged vs R49 [CI] | Δ caught vs R49 [CI] | Apparent caught / NP flagged |"); A("|---|---|---|---|---|---|")
    for r in sorted(nd, key=lambda r: r["d_np_vs_R49"]):
        ap = r.get("apparent_optimistic"); aps = f"{ap['progressor_patients_caught']:.1f} / {ap['non_progressor_patients_flagged']:.1f}" if ap else "–"
        A(f"| {r['label']} | {r['caught']:.1f} | {r['np_flagged']:.1f} | {r['d_np_vs_R49']:+.1f} {ci(r['d_np_vs_R49_ci'], '{:+.1f}')} | {r['d_caught_vs_R49']:+.1f} {ci(r['d_caught_vs_R49_ci'], '{:+.1f}')} | {aps} |")
    A("")
    A("Selected variants with P1 re-matched to their realised endoscopy sensitivity (Δ = rule − matched P1):\n")
    A("| Rule | Endo sens | Endo spec | Δ endo spec vs matched P1 [CI] | Δ NP flagged vs matched P1 [CI] | Δ caught vs matched P1 [CI] |"); A("|---|---|---|---|---|---|")
    for o in list(R["base"].values()) + R["selected_matched"]:
        m = o["delta_vs_P1_matched_endoscopy_sensitivity"]
        A(f"| {o['label']} | {o['value']['endoscopy_sensitivity']:.3f} | {o['value']['endoscopy_specificity']:.3f} | {d3(m['endoscopy_specificity'])} | {d1(m['non_progressor_patients_flagged'])} | {d1(m['progressor_patients_caught'])} |")
    A("")
for w in ("score", "nsamp", "veto", "filter"):
    j = J(f"fix_{w}")
    if not j: continue
    if w == "score":
        A("**Score substitutions** (sample H&E score replaced by a robust aggregate of tile scores; thresholds from the new inner out-of-fold scores):\n")
        A("| Aggregate | Sample AUROC new / original | Δ AUROC [CI] | P1 90%: caught / NP flagged | R49: caught / NP flagged / seq |"); A("|---|---|---|---|---|")
        for fx in j["E1_3"]["fixes"]:
            au = fx["sample_auroc_fold_stratified_pre_event"]; p1 = fx["P1_new_score_90"]["value"]; r4 = fx["R49_new_score"]["value"]
            A(f"| {fx['agg']} | {v3(au['value_new'])} / {v3(au['value_original'])} | {d3(au['delta'])} | {p1['progressor_patients_caught']:.1f} / {p1['non_progressor_patients_flagged']:.1f} | {r4['progressor_patients_caught']:.1f} / {r4['non_progressor_patients_flagged']:.1f} / {r4['samples_sequenced_per_patient']:.2f} |")
        A("")
    if w == "nsamp":
        A("**Sample-count penalty** (endoscopy score = max logit L-IMG − c·log(samples); c from the training patients of each fold):\n")
        for fx in j["E1_3"]["fixes"]: A(f"- {fx['nested']['label']}: c median {fx['c_chosen_median']}; caught {fx['nested']['value']['progressor_patients_caught']:.1f}, NP flagged {fx['nested']['value']['non_progressor_patients_flagged']:.1f}, endoscopy sens / spec {fx['nested']['value']['endoscopy_sensitivity']:.3f} / {fx['nested']['value']['endoscopy_specificity']:.3f}, samples sequenced / patient {fx['nested']['value']['samples_sequenced_per_patient']:.2f}")
        for nm, a in j["E1_3"].get("endoscopy_auroc_fold_stratified", {}).items(): A(f"- Endoscopy-level fold-stratified AUROC, {nm}: {a['auroc_new']} vs max {a['auroc_max']}, Δ {d3(a['delta'])}")
        A("")
    if w == "veto":
        au = [fx for fx in j["E1_3"]["fixes"] if fx["feature"] == "auto"]
        if au:
            A("Automatic veto (feature and direction chosen per fold on training patients) — features chosen (fold count over 100 splits):\n")
            for fx in au: A(f"- {fx['rule']} eps {fx['eps']}: " + ", ".join(f"{k} {v}" for k, v in sorted(fx["features_chosen"].items(), key=lambda kv: -kv[1])))
            A("")
    if w == "filter":
        A("**Second-stage filter**, held-out AUROC of the filter among flagged endoscopies (P vs NP, per-repeat mean):\n")
        A("| Rule | Feature set | AUROC | eps 0: caught / NP flagged | eps 0.05: caught / NP flagged |"); A("|---|---|---|---|---|")
        fx_ = j["E1_3"]["fixes"]
        for fx in fx_:
            if fx["eps"] != 0.0: continue
            o5 = [g for g in fx_ if g["rule"] == fx["rule"] and g["feature_set"] == fx["feature_set"] and g["eps"] == 0.05][0]
            A(f"| {fx['rule']} | {fx['feature_set']} | {v3(fx['auroc_heldout_flagged_P_vs_NP_per_repeat_mean'])} | {fx['nested']['value']['progressor_patients_caught']:.1f} / {fx['nested']['value']['non_progressor_patients_flagged']:.1f} | {o5['nested']['value']['progressor_patients_caught']:.1f} / {o5['nested']['value']['non_progressor_patients_flagged']:.1f} |")
        A("")
A(NOTES_TEXT)
s = open(DOC).read(); s = s[:s.index("## Results")] + "\n".join(L) + "\n"; open(DOC, "w").write(s); print("rendered", len(L), "lines")

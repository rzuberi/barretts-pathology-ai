"""Render the round-9 results (appended below the final '---' of docs/paper_review_round9_false_flags.md) from results/paper_final/review_round9/*.json.
Idempotent: everything after the plan's last '---' line is replaced. Usage: python3 scripts/paper_plan/rv9_render.py"""
import os, json, subprocess, re
T = "/Users/zuberi01/Documents/thesis"; RES = T + "/results/paper_final/review_round9"; DOC = T + "/docs/paper_review_round9_false_flags.md"
J = lambda n: json.loads(open(f"{RES}/{n}.json").read().replace(" | ", " · ")) if os.path.exists(f"{RES}/{n}.json") else None   # method names contain ' | ' (table separator)
def sg(x, d=1): return "n/a" if x is None else (x if isinstance(x, str) else f"{x:+.{d}f}")
def f1(x, d=1): return "n/a" if x is None else (x if isinstance(x, str) else f"{x:.{d}f}")
def ci(c, d=1): return "n/a" if not c or c[0] is None else f"[{c[0]:+.{d}f}, {c[1]:+.{d}f}]"
def dl(d, k, dd=1): x = d[k]; return f"{sg(x['delta'], dd)} {ci(x['ci95'], dd)}"
def row_main(i, m):
    v = m["values"]; r = m["vs_R49"]; p = m["vs_P1_90"]; c99 = r["np_flagged"]["ci99_bonferroni"]
    return (f"| {i} | {m['name'].replace('[nested]', '').replace('[nested, pAUC selection]', '(pAUC selection)').strip()} | {f1(v['caught'])} | {f1(v['np_flagged'])} | **{sg(r['np_flagged']['delta'])}** {ci(r['np_flagged']['ci95'])} | {ci(c99)} | "
            f"{sg(r['caught']['delta'])} {ci(r['caught']['ci95'])} | {f1(v['flags_py'], 3)} ({sg(r['flags_py']['delta'], 3)}) | {sg(r['endo_spec']['delta'], 3)} | {f1(v['seq'], 2)} ({sg(r['seq']['delta'], 2)}) | "
            f"{sg(p['np_flagged']['delta'])} {ci(p['np_flagged']['ci95'])} | {sg(p['caught']['delta'])} |")
HDR = ("| # | Method | Caught | NP flagged | Δ NP flagged vs R49 [95% CI] | 99.2% CI | Δ caught [95% CI] | Flags / NP py (Δ) | Δ endoscopy spec. | Samples seq. / patient (Δ) | Δ NP flagged vs P1 90% [95% CI] | Δ caught vs P1 |\n"
       "|---|---|---|---|---|---|---|---|---|---|---|---|")
def rank_key(m): d = m["vs_R49"]; return (0 if d["caught"]["delta"] >= -0.5 else 1, d["np_flagged"]["delta"], -d["caught"]["delta"])

NOTES_TMPL = """### What the round shows (exploratory; every statement post hoc)

{summary}

### Implementation notes

**Conventions reused.** Setup, endoscopy units (571 pre-event samples, 288 endoscopies, 75 patients, 32 progressors), exact endoscopy dates, patient outcomes (`patient_arrays`), bootstrap draws (`pdraws`, `RandomState(0)`, 2,000, one-class draws rejected), weighted-sum metrics (`parts`, `per_rep`, `met`) and `dstat` are executed unchanged from `rv8_landscape.py` / `rv8b_step3.py`; R49 is `r49_calls`, P1 90% is `thr_arr(img, 0.90)`. Counts are shares × 32 or × 43 in every draw. Check run first in a smoke run (50 draws) and repeated in the full run: R49 31.1 / 31.8, P1 90% 31.1 / 36.0; the engine's per-fold R49 equals `r49_calls` on every held-out endoscopy and its sample-level cut-off re-derivation equals `tb_common.triage` in all 100 folds.

**Nested evaluation (the non-negotiable).** For each outer (repeat, fold) every method is built from the training patients only: training endoscopies carry the stored inner out-of-fold predictions of that fold, held-out endoscopies the stored outer predictions. A patient's memory score uses only that patient's own endoscopies up to the current one (time order by exact date), so no held-out label or other held-out patient enters. The operating threshold is the D-th largest training progressor-patient maximum score, D = the number of training progressor patients the R49 gate itself detects in that fold (28.33 of 28.33 on average; mean reached by the method is in the selection-frequency table); gate-structured methods choose among their candidate cut-off cells the one with the lowest training non-progressor patients flagged subject to detection ≥ D (ties: fewer flagged endoscopies, fewer samples sequenced, grid order). If no candidate reaches D (the pure k-of-n rules, 77 to 100 folds) the candidate with the highest training detection is taken and the fold is counted as infeasible; that is where their progressor losses come from. Hyper-parameters within a family are chosen by the same rule (rows 'nested'); '(pAUC selection)' picks the score-type candidate with the highest patient-level partial AUC (FPR ≤ 0.2, normalised) of the training patients' maximum scores and then applies the same threshold rule. Logistic stackers (B1 family) are 5-fold patient-grouped cross-validated inside the training patients: cut-offs and thresholds use the out-of-fold stacked scores, held-out endoscopies the all-training fit.

**Apparent results (optimistic).** 'Grid point' rows fix one hyper-parameter value for all folds (threshold still training-chosen) and are picked after seeing the evaluation; 'evaluation-matched' rows set the threshold on the evaluation patients, per repeat, so that progressor patients caught equals R49's (re-matched in every bootstrap draw; candidates are the progressor patients' own maximum scores). Both are labelled optimistic in the tables; evaluation-matched is available for score-type methods (a continuous dial) only.

**Interpretation choices where the plan was open.**
- A1(a) running mean, A1(b) exponentially weighted (half-life 1, 2, 4 years, weights 0.5^(Δt/h), Δt from exact dates), A1(c) persistence = 'consecutive-2 or score ≥ t_high' (first endoscopy: t_high only; t_high at the 90, 95 or 97.5th percentile of training non-progressor endoscopy scores, t from the detection rule). Scores are logits of the endoscopy-maximum L-IMG; the gate score is the literal reading of the plan (L-IMG logit, replaced in the middle band by the sequenced top sample's L-LATE logit) and, as an addition, a R49-aligned version in which the middle-band logit is shifted so that the gate's own decision (L-LATE ≥ t_seq) is 'score ≥ logit c*' (R49 is then exactly 'score ≥ logit c*', verified on every held-out endoscopy). Sequencing for the gate-score memory rules is R49's (1.94 samples per patient).
- A2: certificate issued after a sequenced (middle-band) endoscopy whose top sample's L-CNV logit is below t_quiet (training non-progressor sequenced endoscopies, 50th or 75th percentile); for Y = 2 or 4 years afterwards both gate cut-offs are raised to at least the training H&E threshold at 85% or 95% non-progressor specificity (so the low band widens, and at 95% the certified patient is judged by H&E alone). 'No certificate' (= R49) is a candidate.
- A3: second-highest sample's L-IMG against t_mid (80% or 50% sensitivity among training progressor endoscopies with ≥ 2 samples); otherwise the top sample is sequenced and the flag stands only if its L-CNV ≥ the copy-number threshold (80% or 90% sensitivity among training progressor endoscopies, top sample). Both the gate and an H&E-alone version (threshold by the detection rule) are candidates; single-sample endoscopies follow the base rule.
- B1: logistic regression (L2, C = 1, standardised inputs) on endoscopy-maximum L-IMG logit, ID/LGD indicator, age and sex (age at the endoscopy from the stored age at sample), fitted on endoscopy labels; middle band: second regression that adds the top sample's L-CNV logit; endoscopy-level cut-offs (t_low at 90, 95 or 98% training progressor-endoscopy sensitivity, c* at 85, 90 or 95% training non-progressor-endoscopy specificity, t_seq at 70, 80 or 90% sensitivity) form the 27-cell candidate grid. B2 = EW memory over the B1 gate score aligned as above.
- B3: indeterminate when the sequenced call's L-LATE logit is within ±ε of logit t_seq; ε = the 10% or 20% quantile of training middle-band |distance|; indeterminate endoscopies are not flagged (progressor ones count as missed).
- B4: `rv9_refit.R` refits the H&E elastic net with the stored folds and λ rule (outer 10 repeats × 10 folds, inner 10 × 10 × 5), only the observation weights (non-progressor weight 2) or the training rows (NDBE samples only, image block scaled on them; predictions for all held-out samples) changed; L-CNV is the stored one. Reproduction check (weight 1, all rows): exactly the stored predictions. The gate is rebuilt from the refitted L-IMG with the same cut-off rules ('R49 rules rebuilt') and, additionally, with the round-9 grid ('cut-offs re-tuned'); for these rows the detection target is the refitted R49's own training detection.

**Additions (not in the plan).**
- Controls: C1 (R49 with its 27 cut-off cells re-tuned by the round-9 rule), threshold-only dials ('threshold only' rows) on H&E, the gate score, the stacked score and max L-LATE (sequence all samples), and B1-img (stacked gate on L-IMG alone): they show what the training-side tuning rule gives without any memory.
- R49-aligned gate score (above); max L-LATE over all samples as a sequence-everything reference; H&E sample aggregation variants (mean over samples, mean of the top 2) with threshold-only, running-mean and EW rules.
- More memory functions: mean of the last 2; k-of-n variants (coordinator addition: 2-of-n anywhere, 2-of-last-3, consecutive-2, each pure and with a t_high override); 'change from own mean' (per-patient recalibration: score minus λ × the mean of the patient's earlier scores, λ = 0.5 or 1, first endoscopy minus the training non-progressor mean); EW with longer half-lives (0.5, 8, 16 years).
- p53 where available: B1 + p53 (aberrant at the endoscopy, result missing) and A2' (a p53-normal result certifies, same raise as A2); A1+A2' (EW memory with the p53 certificate).
- Combinations: A1+A2 (EW memory of the aligned gate score with the copy-number certificate lowering the score by 0.5 or 1 logit for Y years), B1+A1+A2 (same on the stacked gate score), A1+A2'.
- A4 (literature idea 3): the high band is no longer flagged on the slide alone; every endoscopy at or above t_low has its top sample sequenced and is flagged if L-LATE ≥ t_seq (t_low 90, 95, 98%; t_seq 70, 80, 90%); and memory rules on its aligned score. It sequences about 3 samples per patient (R49 1.94).
- Coordinator items: (1) per-patient ICC of non-progressor (and progressor) endoscopy scores before the memory rules, with the persistence ceiling (non-progressor patients flagged once only, twice, ≥ 3 times); (2) k-of-n variants with the loss from progressors with a single pre-event endoscopy (10 of 32); (3) cumulative false-flag curves by endoscopy index (first-k windows, all patients and patients with ≥ k endoscopies) for R49, P1 90% and the best memory rule; (4) pAUC (FPR ≤ 0.2) selection of the hyper-parameters as the alternative criterion.

**Robustness rows.** NDBE and ≥ 2 samples re-run the whole nested procedure in the row's population (the round-7c definitions, with `PRE` restricted before `setup`); R52(a) uses the round-7 refit predictions of the 693-sample set (4 added progressor patients, 17 samples; their age and sex are cohort means, used only by stacked rows, and their event is the endoscopy at 0 months before final) with R49 and P1 re-derived in that set; tissue strata (< / ≥ 256 tiles, endoscopy by its top-L-IMG sample or all its samples) restrict the evaluation of the main run's held-out calls. Between-repeat variance is read from the 10 per-repeat results. **Refit bootstrap (F3): not re-scored.** The 200 stored draws carry one prediction set per resampled population with copy identifiers and no per-copy patient timelines or repeats, so the longitudinal rules and the nested fold structure cannot be rebuilt from them without a new pipeline; the top-3 methods have no F3 row.

**Not done.** No ACE-B file was read. No deeper (4×) copy number (R58). The round-8 policy P7 to P9 (sequence once, then H&E) were not combined with memory.

**Privacy.** Only aggregates are committed (`results/paper_final/review_round9/`); row-level arrays, the later-grade table, refit predictions and per-fold candidate files stay on the cluster (`killcoyne_mm/review_round9/`). No identifiers or dates; patient cells below 5 are shown as '<5' in F0, in the curves and in the progressor-with-one-endoscopy columns where applicable.
"""
def summary(M):
    by = {(m["family"], m["kind"]): m for m in M["methods"]}; g = lambda f, k="nested": by[(f, k)]; d = lambda m, k="np_flagged": m["vs_R49"][k]["delta"]
    def line(f, label, k="nested"):
        m = g(f, k); v = m["values"]; r = m["vs_R49"]; return f"{label}: {v['np_flagged']:.1f} NP flagged ({d(m):+.1f} [{r['np_flagged']['ci95'][0]:+.1f}, {r['np_flagged']['ci95'][1]:+.1f}]), {v['caught']:.1f} caught ({d(m, 'caught'):+.1f}), {v['seq']:.2f} samples per patient"
    L = ["- **Memory over the gate score is the main lever.** " + line("A1:gate_al:ew", "EW memory of the R49-aligned gate score (hyper-parameter chosen in training)") + "; " + line("A1:gate_lit:mean", "running mean of the literal gate score") + ". The same pattern appears with L-LATE over all samples and, without copy number, with sample-mean H&E (smaller). Against R49: 31.8 NP flagged, 31.1 caught, 1.94 samples per patient.",
         "- **How much is just re-tuning.** " + line("C1", "R49 with cut-offs re-tuned by the round-9 rule") + "; " + line("A1:gate_al:none", "the gate score with a single re-tuned threshold") + ". Memory adds about " + f"{d(g('C1')) - d(g('A1:gate_al:ew')):.1f}" + " fewer flagged non-progressors beyond that.",
         "- **A4 (confirm every above-low endoscopy by sequencing)** with memory is nominally first: " + line("A4:conf_al:ew", "A4 + EW") + ", at the price of about one more sample per patient; without memory " + line("A4", "A4 alone") + ".",
         "- **Pure k-of-n persistence loses progressors:** a progressor with one pre-event endoscopy (10 of 32) cannot be flagged by 2-of-n or consecutive-2; they catch about 21 of 32. With a t_high override the loss is " + f"{-d(g('A1:gate_al:cons2'.replace('cons2', 'pers')), 'caught'):.1f}" + " progressors (consecutive-2 or t_high on the aligned gate score) for " + f"{d(g('A1:gate_al:pers')):+.1f}" + " non-progressors flagged, far less than EW memory gives at a fraction of that loss.",
         "- **Other levers are small:** " + line("A2:cn", "A2 copy-number certificate") + "; " + line("A2:p53", "A2' p53 certificate") + "; " + line("A3", "A3") + "; " + line("B3", "B3 abstention") + "; " + line("B1:full", "B1 stacked gate") + " (no gain from grade, age and sex); B4 refits: " + line("B4a:C1", "weight 2, re-tuned") + " and " + line("B4b:C1", "NDBE only, re-tuned") + f"; with R49's own cut-off rules rebuilt they change {d(g('B4a:R49x')):+.1f} and {d(g('B4b:R49x')):+.1f}" + ". 'Change from own mean' recalibration is worse than no memory.",
         "- **Combining** the EW memory with a copy-number or p53 certificate lowers flags further (" + line("A12", "A1+A2") + ") but costs a little progressor detection (the point estimate falls just below the −0.5 line).",
         "- **Why memory works here:** non-progressors' scores are only weakly correlated within a patient (ICC about 0.2 to 0.3) while progressors' are strongly correlated (about 0.6); averaging shrinks the non-progressors' noise and leaves progressors' consistent signal. Almost all of the reduction appears from the second endoscopy on (first-endoscopy false-flag share unchanged).",
         "- **F0:** flagged non-progressors were not more likely than unflagged ones to show a later IND, LGD or HGD record (counts 5 or 6 of 31 or 36 flagged against fewer than 5 unflagged; all Fisher p ≥ 0.4), but only 7 non-progressors have any later IND or worse, so the comparison cannot detect a moderate difference.",
         "- **Caveats.** About 70 families and several hundred grid points were tried on 75 patients; the top rows are the winners of that search and their intervals do not account for it (the 99.2% intervals cover 6 families only). The nested design protects against tuning on the evaluation patients, not against the choice of which families to report. Results are post hoc on a cohort with repeat endoscopies; ACE-B (mostly one endoscopy per patient) cannot test them. A method worth registering must be fixed (family, half-life rule, cut-offs) before any external look."]
    return "\n".join(L)

def render():
    M = J("merge")["merge"]; F0 = J("f0"); ICC = J("icc"); CU = J("curves"); RC = J("refit_check"); L = []; A = L.append
    h = subprocess.run(["git", "-C", T, "log", "-1", "--format=%h", "--", "results/paper_final/review_round9"], capture_output=True, text=True).stdout.strip() or "pending"
    meths = M["methods"]; nest = sorted([m for m in meths if m["kind"] in ("nested", "nested_pauc")], key=rank_key); byname = {m["name"]: m for m in meths}
    A("## Results\n")
    A(f"Rendered by `scripts/paper_plan/rv9_render.py` from `results/paper_final/review_round9/` (aggregate results {h}); scripts `rv9_falseflags.py` (everything except B4 fitting), `rv9_later.py` (F0 input), `rv9_refit.R` (B4 refits); figures by `rv9_figs.py` (local, not committed). "
      "**This round is exploratory: every number is a post hoc finding, nothing is adjusted for the many methods tried, and no method replaces R49.** Single model = per-repeat values averaged over the 10 repeats. "
      f"Intervals: paired patient bootstrap, {M['draws']:,} draws (`RandomState(0)`); the 99.2% interval (Bonferroni over 6 method families) is shown for reference. No ACE-B file was read; frozen bundles and stored predictions are unchanged; only B4 refits the H&E model (`rv9_refit.R`).\n")
    chk = M["check_round8"]; A(f"Check (round-8 settings, observed): R49 {chk['R49']['caught']:.1f} caught / {chk['R49']['np_flagged']:.1f} NP flagged; P1 90% {chk['P1_90']['caught']:.1f} / {chk['P1_90']['np_flagged']:.1f} (expected 31.1 / 31.8 and 31.1 / 36.0); reproduced: {'yes' if chk['reproduced'] else 'NO'}. "
      f"Refit reproduction (`rv9_refit.R` with weight 1 and all rows against the stored predictions): outer repeat 1 max |difference| {RC['refit_check']['outer_rep_01']['max_abs_pred_diff']:.1e}, inner repeat 1 fold 1 {RC['refit_check']['inner_rep_01_fold_01']['max_abs_pred_diff']:.1e}; passed: {'yes' if RC['refit_check']['passed'] else 'NO'}.\n" if RC else f"Check (round-8 settings, observed): R49 {chk['R49']['caught']:.1f} / {chk['R49']['np_flagged']:.1f}; P1 90% {chk['P1_90']['caught']:.1f} / {chk['P1_90']['np_flagged']:.1f}; reproduced: {'yes' if chk['reproduced'] else 'NO'}.\n")
    # ---------------------------------------------------------------- F0
    if F0:
        F = F0["F0"]; A("### F0. Are flagged non-progressors more likely to show later dysplasia? (descriptive)\n")
        A(f"{F['definition'][0].upper() + F['definition'][1:]}. Non-progressor patients: {F['n_non_progressor_patients']}. Group sizes: " + "; ".join(f"{k} {v}" for k, v in F["group_sizes"].items()) + ".\n")
        A("| Rule | NP ever flagged (≥ 5 of 10 repeats) | Population | Outcome | Flagged: later / not | Not flagged: later / not | Share later: flagged vs not flagged | Fisher p |\n|---|---|---|---|---|---|---|---|")
        for r, o in F["rules"].items():
            for tn, t in o["tables"].items():
                pop, out = tn.split(":"); A(f"| {r} | {o['ever_flagged_in_ge5_of_10_repeats']} | {pop.replace('_', ' ')} | {out.replace('_', ' ')} | {t['flagged_later']} / {t['flagged_not_later']} | {t['not_flagged_later']} / {t['not_flagged_not_later']} | {t['share_later_among_flagged']} vs {t['share_later_among_not_flagged']} | {t['fisher_p_two_sided']} |")
            a = o["any_later_record"]; A(f"| {r} | {o['ever_flagged_in_ge5_of_10_repeats']} | all 43 | any later database record | {a['flagged_with_record']} / {a['flagged_without']} | {a['not_flagged_with']} / {a['not_flagged_without']} | {a['share_flagged']} vs {a['share_not_flagged']} | {a['fisher_p_two_sided']} |")
        A("\nFlagged non-progressor endoscopies per patient-year by later-grade group (groups below 5 patients: '<5'):\n")
        gs = list(next(iter(F["rules"].values()))["flags_per_patient_year_by_later_group"]); A("| Rule | " + " | ".join(gs) + " |\n|---|" + "---|" * len(gs))
        for r, o in F["rules"].items(): A(f"| {r} | " + " | ".join(f"{o['flags_per_patient_year_by_later_group'][g]['flagged_endoscopies_per_patient_year']} (n = {o['flags_per_patient_year_by_later_group'][g]['patients']})" for g in gs) + " |")
        A("")
    # ---------------------------------------------------------------- ICC
    if ICC:
        I = ICC["ICC"]; A("### Per-patient intraclass correlation of endoscopy scores (before the memory rules)\n")
        A("| Score | Class | ICC(1) [95% CI] | Between-patient MS | Within-patient MS | Patients / endoscopies |\n|---|---|---|---|---|---|")
        for sc_, v in I["scores"].items():
            for cl, x in v.items(): A(f"| {sc_} | {cl.replace('_', '-')} | {x['icc1_mean_over_repeats']:.2f} [{x['ci95'][0]:.2f}, {x['ci95'][1]:.2f}] | {x['between_patient_mean_square']:.2f} | {x['within_patient_mean_square']:.2f} | {x['patients']} / {x['endoscopies']} |")
        A(f"\n{I['estimator']}. {I['reading']}.\n")
        A("Persistence ceiling: non-progressor patients (mean over repeats) flagged at exactly 1, 2 and ≥ 3 endoscopies. A patient flagged once only is the most a persistence rule can remove.\n")
        A("| Rule | NP ever flagged | Flagged once only | Twice | ≥ 3 times | Flagged ≥ 2 times | Share of flagged NP with one flag only |\n|---|---|---|---|---|---|---|")
        for r, c in I["persistence_ceiling_np_patients"].items(): A(f"| {r} | {c['np_patients_ever_flagged']} | {c['np_flagged_once_only']} | {c['np_flagged_twice']} | {c['np_flagged_3_or_more']} | {c['np_flagged_at_least_twice']} | {c['share_of_flagged_np_with_only_one_flagged_endoscopy']} |")
        A("")
    # ---------------------------------------------------------------- ranking
    A("### Ranking: nested (held-out) results, all families\n")
    A("Every row: operating threshold (and, within a family, the hyper-parameter) chosen on the training patients' inner out-of-fold predictions of each outer fold by the plan's rule; performance on the held-out patients. "
      "Δ against R49 and against P1 at 90% on the same bootstrap draws. Ranked by Δ NP flagged among rows with Δ caught ≥ -0.5 (point estimate); the other rows follow. 'Control' rows are threshold-only dials and re-tuned R49 (they show what the training-side tuning rule alone does). "
      "'(pAUC selection)' rows choose the hyper-parameter by patient-level partial AUC (FPR ≤ 0.2) instead.\n")
    A(HDR)
    for i, m in enumerate(nest, 1): A(row_main(i, m))
    A("")
    top3 = M["top3_names"]; A("**Top 3 (excluding controls and pAUC variants): " + "; ".join(f"{i}. {n.replace(' [nested]', '')}" for i, n in enumerate(top3, 1)) + ".**\n")
    # ---------------------------------------------------------------- progressors with one endoscopy / k-of-n
    A("### k-of-n persistence and progressors with a single pre-event endoscopy\n")
    A("A 2-of-n rule cannot flag a progressor with one pre-event endoscopy unless an override threshold (t_high) flags it outright.\n")
    A("| Method (nested) | Progressors with 1 endoscopy: caught (of n) | R49 | Δ caught overall | Δ NP flagged [95% CI] |\n|---|---|---|---|---|")
    for m in nest:
        if m["kind"] == "nested" and m["family"].split(":")[-1] in ("k2any", "k2of3", "cons2", "k2any_o", "k2of3_o", "pers") and "caught_progressors_with_one_endoscopy" in m:
            q = m["caught_progressors_with_one_endoscopy"]; A(f"| {m['name'].replace(' [nested]', '')} | {q['caught']} (of {q['n']}) | {q['R49']} | {sg(m['vs_R49']['caught']['delta'])} | {dl(m['vs_R49'], 'np_flagged')} |")
    A("")
    # ---------------------------------------------------------------- group tables: grid points and extras
    def grid_table(title, fam_pred, maxrows=40):
        rows = [m for m in meths if m["kind"] == "grid_point" and fam_pred(m["family"])]; rows = sorted(rows, key=rank_key)[:maxrows]
        if not rows: return
        A(f"#### {title}\n"); A("Fixed hyper-parameter picked after seeing the evaluation (optimistic); the threshold is still the training-chosen one.\n"); A(HDR)
        for i, m in enumerate(rows, 1): A(row_main(i, m))
        A("")
    A("### Grid points (apparent, labelled optimistic)\n")
    grid_table("A1 memory rules", lambda f: f.startswith("A1:") and not f.endswith(":none"), 40); grid_table("A2 / A2' certificates", lambda f: f.startswith("A2:")); grid_table("A3 agreement", lambda f: f == "A3"); grid_table("B3 abstention", lambda f: f == "B3")
    grid_table("A1+A2 and B1+A1+A2 combined", lambda f: f in ("A12", "B2A2"), 20); grid_table("B1 stacked gate and controls (cut-off cells)", lambda f: f.startswith("B1:") or f == "C1", 30); grid_table("B4 refits (re-tuned cut-off cells)", lambda f: f.startswith("B4"), 15)
    ev = sorted([m for m in meths if m["kind"] == "matched"], key=rank_key)[:30]
    A("### Evaluation-matched versions (optimistic)\n"); A("Threshold set on the evaluation patients, per repeat, so that progressor patients caught equals R49's (re-matched in every bootstrap draw). Score-type methods only (candidates: the progressor patients' own scores); the number shown is the 30 best.\n"); A(HDR)
    for i, m in enumerate(ev, 1): A(row_main(i, m))
    A("")
    # ---------------------------------------------------------------- B3, A2 details, selection frequencies
    b3 = [m for m in nest if m["family"] == "B3" and m["kind"] == "nested"]
    if b3: m = b3[0]; A(f"### B3. Abstention: indeterminate endoscopies\n\nNested row: mean per repeat {m['indeterminate_endoscopies_per_repeat_mean']} indeterminate endoscopies, of which {m['indeterminate_progressor_endoscopies_per_repeat_mean']} progressor endoscopies (counted as missed, 288 pre-event endoscopies in all). Selected share over the 100 folds: {m['selected_candidate_counts_over_100_folds']}.\n")
    A("### Choices made inside training (selection frequencies over the 100 outer folds)\n"); A("| Family | Chosen candidate: folds | Infeasible folds (training detection target unreachable) | Mean training detection target / reached |\n|---|---|---|---|")
    for m in nest:
        if m["kind"] == "nested" and "selected_candidate_counts_over_100_folds" in m: A(f"| {m['name'].replace(' [nested]', '')} | {'; '.join(f'{k}: {v}' for k, v in m['selected_candidate_counts_over_100_folds'].items() if v)} | {m['infeasible_folds_np']} | {m['mean_training_detection_target']} / {m['mean_training_detection_method']} |")
    A("")
    # ---------------------------------------------------------------- curves
    if CU:
        C = CU["curves"]; A("### Cumulative false-flag curves by endoscopy index\n"); A(f"Best memory rule in the ranking: **{C['best_memory_rule'].replace(' [nested]', '')}**. {C['note']}.\n")
        A("| k | Rule | NP flagged: all patients [95% CI] | NP flagged: patients with ≥ k endoscopies (n) | Progressors caught: all | Progressors caught: ≥ k (n) | Δ NP vs R49 [95% CI] (all) |\n|---|---|---|---|---|---|---|")
        for k in range(1, 7):
            for r, o in C["curves"].items():
                a = o[f"k={k}"]["all_patients_first_k_endoscopies"]; b = o[f"k={k}"]["restricted_to_patients_with_ge_k_endoscopies"]; d = C["delta_np_share_vs_R49_all_patients"].get(r, {}).get(f"k={k}")
                A(f"| {k} | {r.replace(' [nested]', '')} | {a['np']['share']} {a['np']['ci95'] if isinstance(a['np']['ci95'], str) else ci(a['np']['ci95'], 2).replace('+', '')} | {b['np']['share']} (n = {b['np']['patients']}) | {a['prog']['share']} | {b['prog']['share']} (n = {b['prog']['patients']}) | "
                  f"{'' if d is None else sg(d['delta'], 3) + ' ' + ci(d['ci95'], 3)} |")
        A("")
    # ---------------------------------------------------------------- robustness
    rob = {}
    for r in ("ndbe", "ge2", "tis_lt256", "tis_ge256", "tis_lt256pure", "tis_ge256pure", "r52a"):
        x = J("rob_" + r)
        if x: rob[r] = x["ROB"]
    if rob:
        A("### Robustness of the top 3 methods\n"); A("Rows as the R49 robustness rows (`docs/paper_review_round7c_gate_robustness.md`) that can be re-scored: the same nested procedure is repeated inside the row's population (NDBE-only, ≥ 2 samples, R52(a) refit set: every training-side choice is redone; tissue strata: held-out calls of the main run, evaluation restricted). "
          "Between-repeat variance and the refit bootstrap are below.\n")
        A("| Row | Population (endoscopies; progressor / non-progressor patients) | Rule | Caught | NP flagged | Δ NP flagged vs R49 [95% CI] | Δ caught vs R49 [95% CI] | Δ NP flagged vs P1 90% [95% CI] |\n|---|---|---|---|---|---|---|---|")
        for r, o in rob.items():
            p = o["population"]; popt = f"{p['endoscopies']}; {p['progressor_patients']} / {p['non_progressor_patients']}"
            for nm, v in o["results"].items():
                if nm in ("R49", "P1 90%"): A(f"| {o['row']} | {popt} | {nm} | {f1(v['caught'])} | {f1(v['np_flagged'])} | | | |")
                else: A(f"| {o['row']} | {popt} | {nm} | {f1(v['values']['caught'])} | {f1(v['values']['np_flagged'])} | {dl(v['vs_R49'], 'np_flagged')} | {dl(v['vs_R49'], 'caught')} | {dl(v['vs_P1_90'], 'np_flagged')} |")
        A("\nRow definitions: " + "; ".join(f"**{r}**: {o['definition']}" for r, o in rob.items()) + ".\n")
        A("Between-repeat variance of the main-run result (Δ against R49 per repeat, 10 repeats):\n"); A("| Method | Δ NP flagged: mean (SD; min, max) | Δ caught: mean (SD; min, max) |\n|---|---|---|")
        for n in top3:
            m = byname[n]; a = m["between_repeat_delta_np_flagged_vs_R49"]; b = m["between_repeat_delta_caught_vs_R49"]; A(f"| {n.replace(' [nested]', '')} | {a['mean']:+.2f} ({a['sd']:.2f}; {a['min']:+.1f}, {a['max']:+.1f}) | {b['mean']:+.2f} ({b['sd']:.2f}; {b['min']:+.1f}, {b['max']:+.1f}) |")
        A("\nRefit bootstrap (F3, 200 draws): not re-scored; see Implementation notes.\n")
    A("### Figures (local, not committed)\n"); A("`paper/figures_r9/`: `r9_forest.png` (ranked Δ NP patients flagged against R49, Δ caught annotated), `r9_scatter.png` (caught against NP flagged for every method with R49, P1 90%, P3 90% and grade marked), `r9_f0.png` (F0), `r9_icc.png` (ICC), `r9_curves.png` (cumulative false-flag curves).\n")
    return "\n".join(L)
if __name__ == "__main__":
    body = render(); notes = NOTES_TMPL.format(summary=summary(J("merge")["merge"]))
    doc = open(DOC).read(); i = doc.rindex("\n---\n"); head = doc[:i + 5]; open(DOC, "w").write(head + "\n" + body + "\n" + notes); print("rendered", len(body), "chars")

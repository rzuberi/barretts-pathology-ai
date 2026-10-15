"""Append results to docs/paper_triage_robustness.md below the pre-specification (32653f1); text above the closing '---' line is not changed.
Usage: python tb_render.py RESULTS_COMMIT. Inputs: results/paper_final/triage_robustness/*.json (answers.json holds the one-line answers and deviations)."""
import json, sys, os
RC = sys.argv[1]; R = "results/paper_final/triage_robustness"; DOC = "docs/paper_triage_robustness.md"; L_ = lambda f: json.load(open(f"{R}/{f}.json"))
TASKS = ["their_pre", "their_pre_ndbe", "pkg_pre", "pkg_pre_ndbe"]; LAB = {"their_pre": "their, all pre-event", "their_pre_ndbe": "their, NDBE pre-event", "pkg_pre": "package, all pre-event", "pkg_pre_ndbe": "package, NDBE pre-event"}
BU = {t: L_(f"budget_{t}") for t in TASKS}; GR = {t: L_(f"grid_{t}") for t in TASKS}; LV = {t: L_(f"levels_{t}") for t in TASKS}; SC = {t: L_(f"selci_{t}") for t in ("their_pre", "their_pre_ndbe", "pkg_pre")}
SCN = L_("scanner"); FZ = {s: L_(f"frozen_cutoffs_{s}") for s in ("their", "pkg")}; PW = {t: L_(f"power_{t}") for t in ("S2", "owner")}; ANS = L_("answers")
f3 = lambda x: "—" if x is None else f"{x:.3f}"; ci = lambda c: "" if not c or c[0] is None else f" [{c[0]:.3f}, {c[1]:.3f}]"; cis = lambda c: f" [{c[0]:+.3f}, {c[1]:+.3f}]"; sg = lambda x: f"{x:+.3f}"; pc = lambda x: "not reached" if x is None or x != x else f"{100 * x:.0f}%"
SN = {"triage_LATE": "Triage (L-LATE middle band)", "A": "(A) sequence everyone, L-CNV", "B": "(B) sequence everyone, L-LATE", "C": "(C) H&E only, L-IMG"}
doc = open(DOC).read(); head = doc[: doc.rindex("\n---")].rstrip("\n") + "\n\n---\n"
L = ["", "## Results", "", f"Pre-specification commit 32653f1; results commit {RC}. Scripts `scripts/paper_plan/tb_*.py`, `tb_scanner.R` (Slurm via `scripts/cluster/campaign.sh`, prefixes tbs smoke, tb1, tb2), `tb_fig.py`, `tb_render.py`. Aggregates `results/paper_final/triage_robustness/`; row-level outputs on the cluster only. Figure `~/Downloads/be_paper_figs/v3/14_F_budget_curve.{{pdf,png}}`. Frozen cut-offs `models/killcoyne_frozen_v1/triage_cutoffs.json`.", ""]
L += ["### Status", "", "| Item | Status |", "|---|---|"] + [f"| {k} | {v} |" for k, v in ANS["status"].items()] + [""]
L += ["### Answers", ""] + [f"{i}. {ANS[str(i)]}" for i in range(1, 7)] + [""]
# 1
L += ["### 1. Sequencing-budget curve", ""]
for t in TASKS:
    b = BU[t]; L += [f"**{LAB[t]}.** (A): sensitivity {f3(b['A']['sensitivity'])}, specificity {f3(b['A']['specificity'])}; (B): {f3(b['B']['sensitivity'])}, {f3(b['B']['specificity'])}. Smallest share matching (A): {pc(b['smallest_s_vs_A']['value'])} (95% CI {pc(b['smallest_s_vs_A']['ci95'][0])}–{pc(b['smallest_s_vs_A']['ci95'][1])}; not reached in {b['smallest_s_vs_A']['draws_not_reached']} of 2,000 draws); matching (B): {pc(b['smallest_s_vs_B']['value'])} (95% CI {pc(b['smallest_s_vs_B']['ci95'][0])}–{pc(b['smallest_s_vs_B']['ci95'][1])}; not reached in {b['smallest_s_vs_B']['draws_not_reached']}).", ""]
    if t == "their_pre":
        L += ["| Target share | Realised share | Sensitivity [95% CI] | Specificity [95% CI] |", "|---|---|---|---|"] + [f"| {pc(g['target_share'])} | {pc(g['realised_share'])} | {f3(g['sensitivity'])}{ci(g['sens_ci95'])} | {f3(g['specificity'])}{ci(g['spec_ci95'])} |" for g in b["grid"]] + [""]
# 2
L += ["### 2. Cut-off grid (L-LATE in the middle band; the primary is 95/90; no cell selected)", ""]
for t in TASKS:
    g = GR[t]; L += [f"**{LAB[t]}** (95/90 reproduces `paper_triage.md`: {'yes' if g['reproduces_paper_triage_95_90'] else 'NO'}).", "", "| t_low sens / t_high spec | Sensitivity | Specificity | Share sequenced | Progressor patients all missed | Δ sens vs A | Δ spec vs A | Δ sens vs B | Δ spec vs B |", "|---|---|---|---|---|---|---|---|---|"]
    for c in g["cells"]:
        L.append(f"| {int(100 * c['t_low_sens'])} / {int(100 * c['t_high_spec'])} | {f3(c['sensitivity'])} | {f3(c['specificity'])} | {pc(c['share_sequenced'])} | {c['progressor_patients_all_missed']} | " + " | ".join(f"{sg(c[k]['delta'])}{cis(c[k]['ci95'])}" for k in ("d_sens_vs_A", "d_spec_vs_A", "d_sens_vs_B", "d_spec_vs_B")) + " |")
    L += [""]
# 3
L += ["### 3. CIs including threshold selection (2,000 draws; cut-offs re-chosen in every draw)", "", "| Population | Strategy | Sensitivity | CI, selection-inclusive | CI, fixed thresholds | Specificity | CI, selection-inclusive | CI, fixed thresholds |", "|---|---|---|---|---|---|---|---|"]
for t, s in SC.items():
    for k, v in s["selection_inclusive"].items(): L.append(f"| {LAB[t]} | {SN[k]} | {f3(v['point_sensitivity'])} | {ci(v['sensitivity_ci95']).strip()} | {ci(v['fixed_threshold_sens_ci95']).strip()} | {f3(v['point_specificity'])} | {ci(v['specificity_ci95']).strip()} | {ci(v['fixed_threshold_spec_ci95']).strip()} |")
L += ["", "| Population | Δ sensitivity triage − B [selection-inclusive CI]; one-sided 95% lower | Fixed-threshold CI; lower | Criterion (> −0.05) | Δ specificity triage − A [selection-inclusive CI] | Fixed-threshold CI | Share sequenced [selection-inclusive CI] |", "|---|---|---|---|---|---|---|"]
for t, s in SC.items():
    d = s["d_sens_vs_B"]; a = s["d_spec_vs_A"]; sh = s["share_sequenced"]
    L.append(f"| {LAB[t]} | {sg(d['delta'])}{cis(d['ci95'])}; {f3(d['one_sided_95_lower'])} | {cis(d['fixed_threshold_ci95']).strip()}; {f3(d['fixed_threshold_one_sided_95_lower'])} | {'holds' if d['criterion_holds'] else 'fails'} | {sg(a['delta'])}{cis(a['ci95'])} | {cis(a['fixed_threshold_ci95']).strip()} | {pc(sh['point'])} [{pc(sh['ci95'][0])}, {pc(sh['ci95'][1])}] |")
# 4
L += ["", "### 4. Endoscopy and patient level", ""]
for t in TASKS:
    v = LV[t]; st = v["structure"]
    L += [f"**{LAB[t]}** ({v['n_samples']} samples, {v['n_endoscopies']} endoscopies of which {v['n_progressor_endoscopies']} from progressors, {st['n_patients']} patients; endoscopies per patient mean {st['endoscopies_per_patient']['mean']}, median {st['endoscopies_per_patient']['median']:.0f}, range {st['endoscopies_per_patient']['range'][0]}–{st['endoscopies_per_patient']['range'][1]}; samples per endoscopy mean {st['samples_per_endoscopy']['mean']}, median {st['samples_per_endoscopy']['median']:.0f}, range {st['samples_per_endoscopy']['range'][0]}–{st['samples_per_endoscopy']['range'][1]}). Triage sequences {pc(v['sequenced']['share_samples_sequenced'])} of samples and touches {pc(v['sequenced']['share_endoscopies_with_any_sample_sequenced'])} of endoscopies (≥ 1 sample sequenced).", "",
          "| Strategy | Endoscopy sensitivity [95% CI] | Endoscopy specificity [95% CI] | Progressor patients caught / missed | Non-progressor patients false positive / true negative |", "|---|---|---|---|---|"]
    for s, e in v["endoscopy_level"].items():
        p = v["patient_level"][s]; L.append(f"| {SN[s]} | {f3(e['sensitivity'])}{ci(e['sens_ci95'])} | {f3(e['specificity'])}{ci(e['spec_ci95'])} | {p['progressor_caught']} / {p['progressor_missed']} | {p['nonprogressor_false_positive']} / {p['nonprogressor_true_negative']} |")
    L += [""]
# 5
L += ["### 5. Scanner transfer (L-IMG refitted; C13239-01 = s39, C13210 = s10)", "", "| Within-scanner CV | AUROC, fold-stratified [95% CI] | Samples | Patients (progressor) |", "|---|---|---|---|"]
for sc, w in SCN["within"].items(): L.append(f"| {sc} | {f3(w['auroc_fold_stratified'])}{ci(w['ci95'])} | {w['n_samples']} | {w['n_patients']} ({w['n_P_patients']}) |")
L += ["", "| Design, training → test, harmonisation | Estimable | AUROC on test scanner [95% CI] | Test samples, patients (progressor) | Training patients (progressor) | Test bands low / middle / high (training-scanner inner OOF) |", "|---|---|---|---|---|---|"]
for job, c in sorted(SCN["cross"].items()):
    _, d, s_, h = job.split("_"); nm = f"({d}) {s_} → {'s10' if s_ == 's39' else 's39'}, {'per-scanner z' if h == 'perscan' else 'none'}"
    if not c["estimable"]: L.append(f"| {nm} | no | — | {c['n_test']}, {c['n_test_patients']} ({c['n_test_P_patients']}) | {c['n_train_patients']} ({c['n_train_P_patients']}) | — |"); continue
    bt, bi = c["bands_test_scanner"], c["bands_training_scanner_inner_oof"]
    L.append(f"| {nm} | yes | {f3(c['auroc'])}{ci(c['ci95'])} | {c['n_test']}, {c['n_test_patients']} ({c['n_test_P_patients']}) | {c['n_train_patients']} ({c['n_train_P_patients']}) | {pc(bt['low']['all'])} / {pc(bt['mid']['all'])} / {pc(bt['high']['all'])} ({pc(bi['low']['all'])} / {pc(bi['mid']['all'])} / {pc(bi['high']['all'])}) |")
# 6
f = FZ["their"]; fp = FZ["pkg"]
L += ["", "### 6. Frozen cut-offs and ACE-B power", "", "| Cut-off | Their matrix: median (range) — written to `models/killcoyne_frozen_v1/triage_cutoffs.json` | Package features: median (range) — reported only |", "|---|---|---|"]
for k in ("t_low", "t_high", "t_seq", "t_B"): L.append(f"| {k} | {f['cutoffs'][k]:.4f} ({f['ranges'][k][0]:.4f}–{f['ranges'][k][1]:.4f}) | {fp['cutoffs'][k]:.4f} ({fp['ranges'][k][0]:.4f}–{fp['ranges'][k][1]:.4f}) |")
p0 = PW["S2"]
L += ["", f"Frozen cut-offs on the internal repeat-mean held-out predictions (all pre-event samples): triage sensitivity {f3(p0['internal_sens_triage'])} vs (B) {f3(p0['internal_sens_B'])} (E1 Δ {sg(p0['internal_delta_E1'])}; one-sided 95% lower bound E2 {sg(p0['internal_lower_E2'])}); specificity {f3(p0['internal_spec_triage'])} vs {f3(p0['internal_spec_B'])}; share sequenced {pc(p0['internal_share_sequenced'])}.", "",
      "| ACE-B size | Effect | Target Δ | Power [95% CI] | Median simulated Δ | Median lower bound |", "|---|---|---|---|---|---|"]
for t, p in PW.items():
    for e, v in p["effects"].items(): L.append(f"| {'S2 (database, prevalent excluded)' if t == 'S2' else 'owner split, prevalent excluded'}: {p['n_P']} P / {p['n_NP']} NP patients | {e} | {sg(v['target_delta'])} | {f3(v['power'])} [{f3(v['power_ci95_wilson'][0])}, {f3(v['power_ci95_wilson'][1])}] | {sg(v['median_simulated_delta'])} | {sg(v['median_lower_bound'])} |")
L += ["", "### Deviations and caveats", ""] + [f"- {d}" for d in ANS["deviations"]] + [""]
open(DOC, "w").write(head + "\n".join(L) + "\n"); print("rendered")

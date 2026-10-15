"""Render Results of docs/paper_risk_strata.md from results/paper_final/risk_strata/*.json. Usage: python rs_render.py RESULTS_COMMIT"""
import json, sys
RC = sys.argv[1]; R = "results/paper_final/risk_strata"; DOC = "docs/paper_risk_strata.md"; J = lambda f: json.load(open(f"{R}/{f}.json"))
SA, SP, SN, DP, DN, KM = J("sep_all676"), J("sep_pre"), J("sep_pre_ndbe"), J("disc_pre"), J("disc_pre_ndbe"), J("figs/10_F_risk_KM")
f3 = lambda x: "—" if x is None else f"{x:.3f}"; f2 = lambda x: "—" if x is None else f"{x:.2f}"; bp = lambda p: "< 0.0005" if p == 0 else f"{p}"; pf = lambda p: "< 1e-15" if p < 1e-15 else f"{p:.1e}"; sg = lambda x: "—" if x is None else f"{x:+.3f}"
ci = lambda c: "" if not c or c[0] is None else f" [{c[0]:.3f}, {c[1]:.3f}]"; ci2 = lambda c: "" if not c or c[0] is None else f" [{c[0]:.2f}, {c[1]:.2f}]"; cis = lambda c: "" if not c or c[0] is None else f" [{c[0]:+.3f}, {c[1]:+.3f}]"
def table(h, rows): return ["| " + " | ".join(h) + " |", "|" + "|".join("---" for _ in h) + "|"] + ["| " + " | ".join(map(str, r)) + " |" for r in rows] + [""]
NM = {"P": "Killcoyne published (P)", "C": "CNV, their matrix (C)", "L": "Late fusion, their matrix (L)", "C_pkg": "CNV, package features", "L_pkg": "Late fusion, package features", "WSI": "WSI (L-IMG)", "EARLY": "Early fusion", "INTER": "Inter fusion",
      "EARLY_pkg": "Early fusion, package", "INTER_pkg": "Inter fusion, package", "GRADE": "Pathology grade (raw score)", "INTERCEPT": "Intercept-only (reference)"}
SEP = ["P", "C", "L", "C_pkg", "L_pkg"]
L = open(DOC).read().split("\n## Results")[0].rstrip().split("\n")
L += ["", "## Results", "", f"Pre-specification commit b1f788a; results commit {RC}. Scripts `scripts/paper_plan/rs_strata.py` (Slurm via `scripts/cluster/campaign.sh`, prefixes rs, rs2), `rs_km_fig.py`, `rs_render.py`. Results `results/paper_final/risk_strata/{{sep_all676,sep_pre,sep_pre_ndbe,disc_pre,disc_pre_ndbe,km}}.json`; figure numbers `results/paper_final/risk_strata/figs/10_F_risk_KM.json`; figures `~/Downloads/be_paper_figs/v3/10_F_risk_KM.{{pdf,png}}`, `10_F_risk_KM_patient_supplementary.{{pdf,png}}`.", "",
      "Population: the 676 samples from 80 patients of the Killcoyne discovery cohort with a matched H&E slide; primary analyses on its pre-event samples (571 samples, 75 patients, 161 samples with an observed event, 32 progressor patients).", ""]
L += ["### Status", ""] + table(["Item", "Status"], [["Step 0 (Killcoyne's definitions and metrics)", "DONE (above; definitions match, no change)"], ["1 Risk classes (primary, class-size matched)", "DONE"], ["2 Separation: Killcoyne metrics, Cox, trend, paired Δ", "DONE (Cox ties deviate, see Deviations)"],
      ["3 Kaplan–Meier figure (sample level; patient level supplementary)", "DONE"], ["4 Discrimination and accuracy", "DONE (Brier not reported for grade, which is not a probability)"]])
# ---------- answers
lp = SP["models"]["L"]["primary"]; cx = lp["cox"]; pc = SP["paired"]; d = DP["paired"]["L_vs_C"]
lo, hi = lp["classes"][0], lp["classes"][2]
L += ["### Answers", "",
      f"1. **Yes, for late fusion the classes separate clearly:** on pre-event samples the high class holds {hi['event_samples']}/{hi['samples']} samples with an observed progression against {lo['event_samples']}/{lo['samples']} in the low class, HR high vs low {f2(cx['HR_high_vs_low'])}{ci2(cx['HR_high_ci95'])} (patient-clustered), and the same holds on NDBE samples (HR {f2(SN['models']['L']['primary']['cox']['HR_high_vs_low'])}{ci2(SN['models']['L']['primary']['cox']['HR_high_ci95'])}); the moderate class is not separable from low on NDBE samples (HR {f2(SN['models']['L']['primary']['cox']['HR_moderate_vs_low'])}{ci2(SN['models']['L']['primary']['cox']['HR_moderate_ci95'])}).",
      f"2. **Better than CNV, not clearly better than Killcoyne's published model:** L vs C, Δ log HR (high vs low) {sg(pc['L_vs_C|primary']['delta_logHR_high_vs_low']['delta'])}{cis(pc['L_vs_C|primary']['delta_logHR_high_vs_low']['ci95'])} and Δ share of events in the high class {sg(pc['L_vs_C|primary']['delta_share_events_in_high']['delta'])}{cis(pc['L_vs_C|primary']['delta_share_events_in_high']['ci95'])} (primary thresholds; class-size matched {sg(pc['L_vs_C|matched']['delta_logHR_high_vs_low']['delta'])}{cis(pc['L_vs_C|matched']['delta_logHR_high_vs_low']['ci95'])} and {sg(pc['L_vs_C|matched']['delta_share_events_in_high']['delta'])}{cis(pc['L_vs_C|matched']['delta_share_events_in_high']['ci95'])}); L vs P, Δ log HR {sg(pc['L_vs_P|primary']['delta_logHR_high_vs_low']['delta'])}{cis(pc['L_vs_P|primary']['delta_logHR_high_vs_low']['ci95'])} (primary) and {sg(pc['L_vs_P|matched']['delta_logHR_high_vs_low']['delta'])}{cis(pc['L_vs_P|matched']['delta_logHR_high_vs_low']['ci95'])} (matched), with only the matched-scheme event share favouring L ({sg(pc['L_vs_P|matched']['delta_share_events_in_high']['delta'])}{cis(pc['L_vs_P|matched']['delta_share_events_in_high']['ci95'])}); all unadjusted.",
      f"3. **Yes on all pre-event samples, L beats C on C-index and Brier:** Harrell's C {f3(DP['models']['L']['harrell']['value'])} vs {f3(DP['models']['C']['harrell']['value'])} (Δ {sg(d['harrell']['delta'])}{cis(d['harrell']['ci95'])}), Uno's C Δ {sg(d['uno']['delta'])}{cis(d['uno']['ci95'])}, integrated Brier 0–5 years {f3(DP['models']['L']['ibs5']['value'])} vs {f3(DP['models']['C']['ibs5']['value'])} (Δ {sg(d['ibs5']['delta'])}{cis(d['ibs5']['ci95'])}); but no model's integrated Brier beats the Kaplan–Meier null ({f3(DP['km_null']['ibs5']['value'])}), because ever-progression probabilities are used as time-specific risks; on NDBE samples the C-index differences are not significant ({sg(DN['paired']['L_vs_C']['harrell']['delta'])}{cis(DN['paired']['L_vs_C']['harrell']['ci95'])}).", ""]
# ---------- 1
def cls_rows(S):
    rows = []
    for m in SEP:
        for sch in ("primary", "matched"):
            if m == "P" and sch == "matched": continue
            o = S["models"][m][sch]; rows.append([NM[m], sch] + [f"{c['samples']} / {c['patients']} / {c['event_samples'] if c['event_samples'] is not None else c['progressor_samples']}" for c in o["classes"]] + [", ".join(f2(x) if x is not None else "—" for x in S["matched_cutpoints"][m]) if sch == "matched" else "0.30, 0.50"])
    return rows
L += ["### 1. Risk classes", "", f"Samples / patients / samples with an observed event, per class. Class-size matched cut-points reproduce (P)'s proportions: pre-event {', '.join(f2(x) for x in SP['P_class_proportions'])} (low, moderate, high); NDBE {', '.join(f2(x) for x in SN['P_class_proportions'])}.", "",
      "**All pre-event samples (571 / 75)**", ""] + table(["Model", "Scheme", "Low", "Moderate", "High", "Cut-points (low/moderate, moderate/high)"], cls_rows(SP))
L += ["**NDBE pre-event samples (438 / 71)**", ""] + table(["Model", "Scheme", "Low", "Moderate", "High", "Cut-points"], cls_rows(SN))
# ---------- 2
def kill_rows(S, label):
    rows = []
    for m in ("P", "C", "L", "C_pkg", "L_pkg"):
        o = S["models"][m]["primary"]; ss = o["sens_spec"]; cs = o["class_share_by_status"]
        rows.append([label, NM[m], f"{f3(ss['low_vs_above']['sensitivity'])}{ci(ss['low_vs_above'].get('sensitivity_ci95'))} / {f3(ss['low_vs_above']['specificity'])}{ci(ss['low_vs_above'].get('specificity_ci95'))}",
                     f"{f3(ss['high_vs_below']['sensitivity'])}{ci(ss['high_vs_below'].get('sensitivity_ci95'))} / {f3(ss['high_vs_below']['specificity'])}{ci(ss['high_vs_below'].get('specificity_ci95'))}",
                     f"{cs['progressor']['NDBE'][2]:.3f} (n {cs['progressor']['NDBE'][3]})" if cs['progressor']['NDBE'][2] is not None else "—", f"{cs['nonprogressor']['NDBE'][0]:.3f} (n {cs['nonprogressor']['NDBE'][3]})" if cs['nonprogressor']['NDBE'][0] is not None else "—"])
    return rows
L += ["### 2. Separation", "", "**Killcoyne's metrics, recomputed** (per sample, label = progressor status; patient-bootstrap 95% CIs). Killcoyne reported, on 773 samples: sensitivity / specificity 0.87 / 0.65 at Pr ≤ 0.3 and 0.72 / 0.82 at Pr ≥ 0.5; 60.5% of progressor NDBE samples high and 64.7% of non-progressor NDBE samples low.", ""]
L += table(["Population", "Model", "Sens / spec, low vs above (Pr > 0.3)", "Sens / spec, high (Pr ≥ 0.5) vs below", "Progressor NDBE samples high", "Non-progressor NDBE samples low"], kill_rows(SA, "all 676 samples") + kill_rows(SP, "pre-event"))
rows = []
for m in ("P", "C", "L"):
    E = SA["models"][m]["primary"]["endoscopy_sensitivity"]; rows.append([NM[m]] + [f"{f2(e['share_high'])} ({e['endoscopies']})" for e in E])
L += ["**Endoscopy-level sensitivity** (Fig. 3b of Killcoyne): share of progressor endoscopies whose highest-scoring sample is high risk, by months before the endpoint, all 676 samples (endoscopies in parentheses). Killcoyne (p. 1727): 50% of endoscopies at least 8 years before HGD/IMC had at least one high-risk sample; their Fig. 3b labels the fourth bin 45–72 months, the pre-specification uses 48–72.", ""]
L += table(["Model"] + [e["months_before_endpoint"] for e in SA["models"]["P"]["primary"]["endoscopy_sensitivity"]], rows)
rows = []
for m in ("P", "C", "L"):
    for dd in SA["models"][m]["primary"]["deciles"]: pass
    D = SA["models"][m]["primary"]["deciles"]; rows.append([NM[m]] + [f"{x['P']}:{x['NP']} ({f2(x['mean_pred'])})" for x in D])
L += ["**Decile calibration** (Fig. 2a inset of Killcoyne): progressor:non-progressor samples (mean predicted probability) per decile of predicted probability, all 676 samples.", ""] + table(["Model"] + [f"D{i}" for i in range(1, 11)], rows)
def cox_rows(S, label):
    rows = []
    for m in SEP:
        for sch in ("primary", "matched"):
            if m == "P" and sch == "matched": continue
            o = S["models"][m][sch]; c = o["cox"]; t = o["logrank_trend"]; oc = o["ordinal_cox_robust"]
            if isinstance(c, str): rows.append([label, NM[m], sch, c, "", "", ""]); continue
            rows.append([label, NM[m], sch, f"{f2(c['HR_moderate_vs_low'])}{ci2(c['HR_moderate_ci95'])}", f"{f2(c['HR_high_vs_low'])}{ci2(c['HR_high_ci95'])}", f"z {f2(t['z'])}, p {pf(t['p'])}", f"{f2(oc['HR_per_class'])}{ci2(oc.get('ci95'))}, p {oc['p']:.1e}" if isinstance(oc, dict) else oc])
    return rows
L += ["**Cox model on class** (low = reference; sample level; patient-clustered robust variance) and **log-rank trend test** (samples treated as independent) with the cluster-robust ordinal-class Cox as the clustered counterpart.", ""]
L += table(["Population", "Model", "Scheme", "HR moderate vs low [95% CI]", "HR high vs low [95% CI]", "Log-rank trend", "Ordinal class HR per step (robust) [CI]"], cox_rows(SP, "pre-event") + cox_rows(SN, "NDBE pre-event"))
rows = []
for S, lab in ((SP, "pre-event"), (SN, "NDBE pre-event")):
    for k, v in S["paired"].items():
        a, sch = k.split("|"); rows.append([lab, a.replace("_vs_", " vs ").replace("_pkg", " (package)"), sch, f"{sg(v['delta_logHR_high_vs_low']['delta'])}{cis(v['delta_logHR_high_vs_low']['ci95'])}, p {bp(v['delta_logHR_high_vs_low']['p_bootstrap_unadjusted'])} ({v['delta_logHR_high_vs_low']['valid_draws']} draws)",
                                            f"{sg(v['delta_share_events_in_high']['delta'])}{cis(v['delta_share_events_in_high']['ci95'])}, p {bp(v['delta_share_events_in_high']['p_bootstrap_unadjusted'])}"])
L += ["**Paired comparisons** (2,000 patient-bootstrap draws, Cox refitted per draw with Breslow ties; draws with no event in the low or high class dropped; unadjusted).", ""] + table(["Population", "Comparison", "Scheme", "Δ log HR high vs low [CI], p", "Δ share of events in the high class [CI], p"], rows)
# ---------- 3
rows = []
for lev in ("sample", "patient"):
    for m in ("P", "C", "L"):
        for c in ("low", "moderate", "high"):
            x = KM[lev][m].get(c)
            if x: rows.append([lev, NM[m], c, f"{x['n']} / {x['events']}", " / ".join(str(x["at_risk"][str(a)] if str(a) in x["at_risk"] else x["at_risk"][a]) for a in (0, 2, 4, 6, 8)), " / ".join(f2(x["surv_at"][str(a)]) for a in (2, 4, 6, 8))])
L += ["### 3. Kaplan–Meier", "", "Figure `10_F_risk_KM` (sample level, primary) and `10_F_risk_KM_patient_supplementary` (patient level: each patient's earliest pre-event sample). Primary thresholds; 95% patient-bootstrap bands.", ""] + table(["Level", "Model", "Class", "n / events", "At risk at 0 / 2 / 4 / 6 / 8 years", "Progression-free at 2 / 4 / 6 / 8 years"], rows)
# ---------- 4
def disc_rows(D):
    rows = []
    for m in ["P", "C", "L", "WSI", "EARLY", "INTER", "GRADE", "C_pkg", "L_pkg", "EARLY_pkg", "INTER_pkg", "INTERCEPT"]:
        o = D["models"][m]; b = (lambda k: "n/a" if m == "GRADE" else f"{f3(o[k]['value'])}{ci(o[k]['ci95'])}")
        if m == "GRADE" and D is DN: rows.append([NM[m], "not estimable: grade is constant (NDBE) in this population", "—", "—", "n/a", "n/a", "n/a", "n/a"]); continue
        rows.append([NM[m], f"{f3(o['auroc']['value'])}{ci(o['auroc']['ci95'])} ({D['auroc_type'][m]})", f"{f3(o['harrell']['value'])}{ci(o['harrell']['ci95'])}", f"{f3(o['uno']['value'])}{ci(o['uno']['ci95'])}", b("brier_1"), b("brier_3"), b("brier_5"), b("ibs5")])
    k = D["km_null"]; rows.append(["Kaplan–Meier null", "—", "—", "—", f"{f3(k['brier_1']['value'])}{ci(k['brier_1']['ci95'])}", f"{f3(k['brier_3']['value'])}{ci(k['brier_3']['ci95'])}", f"{f3(k['brier_5']['value'])}{ci(k['brier_5']['ci95'])}", f"{f3(k['ibs5']['value'])}{ci(k['ibs5']['ci95'])}"])
    return rows
H = ["Model", "AUROC ever-progression [CI] (type)", "Harrell's C (τ 8 y)", "Uno's C (τ 8 y)", "Brier 1 y", "Brier 3 y", "Brier 5 y", "Integrated Brier 0–5 y"]
L += ["### 4. Discrimination and accuracy", "", "**Caveat:** matched case–control cohort, so Brier values compare models with each other and do not measure real-world calibration; each model's ever-progression probability is used as its predicted risk at every horizon. Lower Brier is better.", "",
      f"**All pre-event samples** ({DP['n_samples']} samples, {DP['n_patients']} patients, {DP['n_events']} samples with an event).", ""] + table(H, disc_rows(DP))
L += [f"**NDBE pre-event samples** ({DN['n_samples']} samples, {DN['n_patients']} patients, {DN['n_events']} with an event).", ""] + table(H, disc_rows(DN))
rows = []
for D, lab in ((DP, "pre-event"), (DN, "NDBE pre-event")):
    for k, v in D["paired"].items():
        rows.append([lab, k.replace("_vs_", " vs ").replace("_pkg", " (package)")] + [f"{sg(v[x]['delta'])}{cis(v[x]['ci95'])}, p {bp(v[x]['p_bootstrap_unadjusted'])}" for x in ("auroc", "harrell", "uno", "ibs5")])
L += ["**Paired Δ** (2,000 patient-bootstrap draws, unadjusted; for Brier, negative favours the first model).", ""] + table(["Population", "Comparison", "Δ AUROC", "Δ Harrell's C", "Δ Uno's C", "Δ integrated Brier"], rows)
L += [f"The intercept-only reference scores Harrell's C {f3(DP['models']['INTERCEPT']['harrell']['value'])} on pre-event samples: the pooled C-indices of the cross-validated models carry the stratified-CV fold-prevalence artefact (`docs/paper_horizon_answers.md`), which (P) and the raw grade do not; the AUROC column is fold-stratified for the cross-validated models and is free of it (intercept-only 0.500).", "",
      "### Deviations from the pre-specification", "",
      "- Cox HRs and their CIs: the pre-specified statsmodels PHReg with Breslow ties and patient-clustered robust variance returned non-finite robust standard errors (string and integer group labels), so every reported HR, CI and robust p uses the fallback built into the script: lifelines `CoxPHFitter`, Efron ties, patient-clustered sandwich variance (`cox.source` in the JSON). The paired bootstrap comparisons use statsmodels PHReg with Breslow ties as specified (point estimates only, no robust variance needed), so their point Δ log HR can differ slightly from the difference of the tabulated Efron HRs.",
      "- The first run (prefix rs) of the two pre-event separation tasks returned NaN robust CIs and p-values; they were rerun with the fallback (prefix rs2). Class counts, Killcoyne metrics, trend tests and the paired bootstrap are identical between the two runs; the tabulated HR point estimates differ from the first run because of the Efron ties (e.g. L, high vs low, 10.85 with Breslow vs 11.64 with Efron). The all-676-samples task was not rerun: only its Killcoyne metrics are reported, and its Cox fields in `sep_all676.json` are the NaN-CI first-run values and are not used.",
      "- Sensitivity and specificity CIs are given for the primary thresholds only; for the class-size matched scheme the cut-points are data-derived and no CI is given.", "",
      "### Caveats", "",
      "- (P) are Killcoyne's leave-one-patient-out predictions from a model trained on all 773 samples (their population, their features, hg19); they are not cross-validated in our folds and are not a like-for-like comparison with C and L.",
      "- Samples are clustered within patients; sample-level KM and the log-rank trend test treat samples as independent (the bands and the ordinal Cox use patient clustering).",
      "- Matched case–control design: event rates, KM curves and Brier values are not population risks; HRs compare classes within this cohort.", ""]
open(DOC, "w").write("\n".join(L) + "\n"); print("rendered", len(L))

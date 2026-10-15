"""Append Addendum A results to docs/paper_risk_strata.md below its pre-specification (f29bd1d); nothing above '### Addendum A results' is changed. Usage: python rs_fsc_render.py RESULTS_COMMIT"""
import json, sys
RC = sys.argv[1]; R = "results/paper_final/risk_strata"; DOC = "docs/paper_risk_strata.md"; MK = "\n### Addendum A results"
F = {p: json.load(open(f"{R}/fsc_{p}.json")) for p in ("pre", "pre_ndbe")}; D = {p: json.load(open(f"{R}/disc_{p}.json")) for p in ("pre", "pre_ndbe")}
f3 = lambda x: "—" if x is None else f"{x:.3f}"; sg = lambda x: f"{x:+.3f}"; ci = lambda c: f" [{c[0]:.3f}, {c[1]:.3f}]"; cis = lambda c: f" [{c[0]:+.3f}, {c[1]:+.3f}]"; bp = lambda p: "< 0.0005" if p == 0 else f"{p}"
NM = {"P": "Killcoyne published (P) †", "C": "CNV, their matrix (C)", "L": "Late fusion, their matrix (L)", "WSI": "WSI (L-IMG)", "EARLY": "Early fusion", "INTER": "Inter fusion", "GRADE": "Pathology grade (raw score) †",
      "C_pkg": "CNV, package features", "L_pkg": "Late fusion, package features", "EARLY_pkg": "Early fusion, package", "INTER_pkg": "Inter fusion, package", "INTERCEPT": "Intercept-only (reference)"}
LAB = {"pre": "All pre-event samples", "pre_ndbe": "NDBE pre-event samples"}
doc = open(DOC).read(); head = doc.split(MK)[0].rstrip("\n"); assert "## Addendum A: fold-stratified C-index (pre-specification)" in head
L = ["", "### Addendum A results", "", f"Pre-specification commit f29bd1d; results commit {RC}. Script `scripts/paper_plan/rs_fsc.py` (Slurm via `scripts/cluster/campaign.sh`, prefix fsc), `scripts/paper_plan/rs_fsc_render.py`. Aggregates `results/paper_final/risk_strata/fsc_pre.json`, `fsc_pre_ndbe.json`.", ""]
fm = F["pre"]["fold_match_vs_cfg0"]; ic = {p: F[p]["intercept_check"] for p in F}
pc = {p: all(abs(F[p]["pooled_crosscheck"][m]["harrell"] - D[p]["models"][m]["harrell"]["value"]) < 0.0015 and abs(F[p]["pooled_crosscheck"][m]["uno"] - D[p]["models"][m]["uno"]["value"]) < 0.0015 for m in D[p]["models"] if m in F[p]["pooled_crosscheck"]) for p in F}
L += ["| Check | Result |", "|---|---|",
      f"| Fold columns of every cross-validated model equal the cfg-0 folds, all 10 repeats | {'yes' if all(fm.values()) else 'no: ' + ', '.join(k for k, v in fm.items() if not v) + ' (scored on own folds)'} |",
      f"| Intercept-only, fold-stratified Harrell's / Uno's C (must be 0.500) | pre-event {f3(ic['pre']['harrell'])} / {f3(ic['pre']['uno'])}; NDBE {f3(ic['pre_ndbe']['harrell'])} / {f3(ic['pre_ndbe']['uno'])}: {'PASS' if ic['pre']['pass'] and ic['pre_ndbe']['pass'] else 'FAIL'} |",
      f"| Pooled C recomputed by this script equals Section 4 (±0.001) | {'yes, both populations' if all(pc.values()) else 'NO: ' + str(pc)} |", ""]
def tbl(p):
    out = [f"**{LAB[p]}** ({F[p]['n_samples']} samples, {F[p]['n_patients']} patients, {F[p]['n_events']} samples with an event; {F[p]['draws']} patient-bootstrap draws).", "",
           "| Model | Harrell's C, fold-stratified [95% CI] | Uno's C, fold-stratified [95% CI] | Harrell's C, pooled (Section 4) | Uno's C, pooled (Section 4) |", "|---|---|---|---|---|"]
    for m in NM:
        o = F[p]["models"][m]; d = D[p]["models"][m]
        if isinstance(o, str): out.append(f"| {NM[m]} | not estimable (grade constant) | — | — | — |"); continue
        out.append(f"| {NM[m]} | {f3(o['harrell']['value'])}{ci(o['harrell']['ci95'])} | {f3(o['uno']['value'])}{ci(o['uno']['ci95'])} | {f3(d['harrell']['value'])} | {f3(d['uno']['value'])} |")
    return out + [""]
L += tbl("pre") + tbl("pre_ndbe") + ["† Not cross-validated: scored on the same within-fold pairs as the cross-validated models (cfg-0 folds), averaged over the 10 repeats; only the pair set varies across repeats.", ""]
L += ["**Paired Δ, fold-stratified** (same 2,000 patient-bootstrap draws; unadjusted two-sided bootstrap p).", "", "| Population | Comparison | Δ Harrell's C [95% CI], p | Δ Uno's C [95% CI], p |", "|---|---|---|---|"]
for p in F:
    for k, v in F[p]["paired"].items():
        L.append(f"| {LAB[p]} | {k.replace('_vs_', ' vs ').replace('_pkg', ' (package)')} | {sg(v['harrell']['delta'])}{cis(v['harrell']['ci95'])}, p {bp(v['harrell']['p_bootstrap_unadjusted'])} | {sg(v['uno']['delta'])}{cis(v['uno']['ci95'])}, p {bp(v['uno']['p_bootstrap_unadjusted'])} |")
a, b = F["pre"]["paired"]["L_vs_C"], F["pre"]["paired"]["L_vs_P"]; an = F["pre_ndbe"]["paired"]["L_vs_C"]
L += ["", f"**Answer (one line).** The Section 4 answer to question 3 holds on all pre-event samples under fold-stratified scoring: L beats C on Harrell's C ({sg(a['harrell']['delta'])}{cis(a['harrell']['ci95'])}) and Uno's C ({sg(a['uno']['delta'])}{cis(a['uno']['ci95'])}), with every model's C-index lower than its pooled value; L vs P is not significant ({sg(b['harrell']['delta'])}{cis(b['harrell']['ci95'])}), and on NDBE samples neither difference is significant (L vs C {sg(an['harrell']['delta'])}{cis(an['harrell']['ci95'])}).", "",
      "**Caveat (observed after running, not pre-specified).** For every model the fold-stratified point estimate sits near the upper end of its percentile CI (e.g. L, pre-event, " + f"{f3(F['pre']['models']['L']['harrell']['value'])}{ci(F['pre']['models']['L']['harrell']['ci95'])}). A likely cause: a patient drawn more than once keeps its fold, so its copies are compared with each other inside one fold, and these within-patient pairs make up a much larger share of the within-fold pairs than of the pooled pairs; within-patient ordering is close to uninformative, which pulls the bootstrap distribution towards 0.5. The CIs of single models are therefore conservative on the low side; the paired Δ, which uses the same pairs for both models, is less affected. The method was not changed after seeing this.", ""]
open(DOC, "w").write(head + "\n" + "\n".join(L)); print("rendered addendum")

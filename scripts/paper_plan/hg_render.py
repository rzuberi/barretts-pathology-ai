"""Render Results of docs/paper_horizon_grade_row.md. Grade row from results/paper_final/horizon_grade_row/hg_*.json; other rows copied from
horizon_answers/q0_*.json (pooled) and horizon_foldstrat/fs_*.json (fold-stratified). Usage: python hg_render.py RESULTS_COMMIT"""
import json, sys
RC = sys.argv[1]; DOC = "docs/paper_horizon_grade_row.md"; G = "results/paper_final/horizon_grade_row"
HG = {(s, t): json.load(open(f"{G}/hg_{s}_{t}.json")) for s in ("their", "pkg") for t in (1, 3, 5)}; NC = json.load(open(f"{G}/hg_ndbecheck.json"))
Q = {(s, p): json.load(open(f"results/paper_final/horizon_answers/q0_{s}_{p}.json"))["units"]["sample"]["horizons"] for s in ("their", "pkg") for p in ("pre", "pre_ndbe")}
FS = {(s, p, t): json.load(open(f"results/paper_final/horizon_foldstrat/fs_{s}_{p}_{t}.json")) for s in ("their", "pkg") for p in ("pre", "pre_ndbe") for t in (1, 3, 5)}
f3 = lambda x: "—" if x is None else f"{x:.3f}"; sg = lambda x: "—" if x is None else f"{x:+.3f}"
ci = lambda c: "[—]" if not c or c[0] is None else f"[{c[0]:.3f}, {c[1]:.3f}]"; cis = lambda c: "[—]" if not c or c[0] is None else f"[{c[0]:+.3f}, {c[1]:+.3f}]"
def table(h, rows): return ["| " + " | ".join(h) + " |", "|" + "|".join("---" for _ in h) + "|"] + ["| " + " | ".join(map(str, r)) + " |" for r in rows] + [""]
SRC = {"their": "their matrix", "pkg": "package features"}; POP = {"pre": "all pre-event samples", "pre_ndbe": "NDBE pre-event samples"}
ARMS = [("CNV (replication)", "L-CNV"), ("WSI", "L-IMG"), ("Early fusion", "L-EARLY"), ("Inter fusion", "L-INTER"), ("Late fusion", "L-LATE")]
def cell(fs_v, fs_c, po_v, po_c): return f"{f3(fs_v)} {ci(fs_c)}; pooled {f3(po_v)} {ci(po_c)}"
def main(s, p, arms, grade_row=True):
    rows = []
    if grade_row:
        if p == "pre": rows.append(["Pathology grade (raw score)"] + [cell(HG[(s, t)]["arms"]["GRADE"]["foldstrat"]["auroc"], HG[(s, t)]["arms"]["GRADE"]["foldstrat"]["ci95"], HG[(s, t)]["arms"]["GRADE"]["pooled"]["auroc"], HG[(s, t)]["arms"]["GRADE"]["pooled"]["ci95"]) for t in (1, 3, 5)] + ["pending"] * 3)
        else: rows.append(["Pathology grade (raw score)"] + ["not estimable (grade constant: all NDBE)"] * 3 + ["pending"] * 3)
    for nm, a in arms:
        rows.append([nm] + [cell(FS[(s, p, t)]["arms"][a]["ipcw"], FS[(s, p, t)]["arms"][a]["ipcw_ci95"], Q[(s, p)][str(t)]["arms"][a]["ipcw"]["auroc"], Q[(s, p)][str(t)]["arms"][a]["ipcw"]["ci95"]) for t in (1, 3, 5)] + ["pending"] * 3)
    rows.append(["n cases / n controls (samples; patients)"] + [f"{FS[(s, p, t)]['n_cases']} / {FS[(s, p, t)]['n_controls']}; {FS[(s, p, t)]['n_case_patients']} / {FS[(s, p, t)]['n_control_patients']}" for t in (1, 3, 5)] + ["pending (slides being scanned)"] * 3)
    return table(["", "Internal 1-year", "Internal 3-year", "Internal 5-year", "ACE-B 1", "ACE-B 3", "ACE-B 5"], rows)
L = open(DOC).read().split("\n## Results")[0].rstrip().split("\n")
chk = all(HG[k]["check_against_existing"][a]["pooled_here"] == HG[k]["check_against_existing"][a]["pooled_q0"] and HG[k]["check_against_existing"][a]["foldstrat_here"] == HG[k]["check_against_existing"][a]["foldstrat_fs"] for k in HG for a in ("L-CNV", "L-LATE"))
L += ["", "## Results", "", f"Pre-specification commit 487b2a7; results commit {RC}. Script `scripts/paper_plan/hg_grade.py` (Slurm via `scripts/cluster/campaign.sh`, prefix hg), `scripts/paper_plan/hg_render.py`. Results `results/paper_final/horizon_grade_row/hg_{{their,pkg}}_{{1,3,5}}.json`, `hg_ndbecheck.json`. Other rows: pooled from `results/paper_final/horizon_answers/q0_*.json` (afa0278), fold-stratified from `results/paper_final/horizon_foldstrat/fs_*.json` (d6b43ff).", "",
      "### Status", ""] + table(["Item", "Status"], [["Pathology grade (raw score) row, pooled and fold-stratified, all pre-event samples", "DONE"], ["NDBE pre-event samples only", f"NOT ESTIMABLE (distinct grades: {NC['distinct_grades']}, n = {NC['n']})"],
      ["Paired Δ vs L-CNV and vs L-LATE", "DONE"], ["Supplement: L-CLIN and demographics only", "DONE (copied, no new computation)"]]) + \
     [f"Check: L-CNV and L-LATE recomputed on the same draws reproduce the stored pooled and fold-stratified AUROCs exactly ({'yes' if chk else 'no'}). Pre-event grades: " + ", ".join(f"{k} {v}" for k, v in NC["pre_event_grades"].items()) + " samples.", ""]
g = {t: HG[("their", t)]["arms"]["GRADE"] for t in (1, 3, 5)}
L += ["### Answer", "", f"Raw pathology grade ranks samples above chance only at 1 year (fold-stratified {f3(g[1]['foldstrat']['auroc'])} {ci(g[1]['foldstrat']['ci95'])}, pooled {f3(g[1]['pooled']['auroc'])}) and close to chance at 3 and 5 years ({f3(g[3]['foldstrat']['auroc'])}, {f3(g[5]['foldstrat']['auroc'])}). "
      f"It is below L-CNV at every horizon (fold-stratified Δ {', '.join(sg(g[t]['foldstrat']['delta_vs_L-CNV']['delta']) for t in (1, 3, 5))}) and below L-LATE ({', '.join(sg(g[t]['foldstrat']['delta_vs_L-LATE']['delta']) for t in (1, 3, 5))}); the 1-year Δ vs L-CNV has a CI crossing zero. On NDBE samples the grade is constant and the row is not estimable. One line: grade carries near-term information only, which the CNV and image models exceed.", ""]
for s in ("their", "pkg"):
    for p in ("pre", "pre_ndbe"):
        L += [f"### Table: {SRC[s]}, {POP[p]}", "", "Each cell: fold-stratified IPCW AUROC [95% patient-bootstrap CI]; pooled IPCW AUROC [CI]. The Pathology grade row is identical for both CNV sources.", ""] + main(s, p, ARMS)
rows = []
for s in ("their", "pkg"):
    for t in (1, 3, 5):
        for m in ("foldstrat", "pooled"):
            o = HG[(s, t)]["arms"]["GRADE"][m]
            rows.append([SRC[s], f"{t} y", "fold-stratified" if m == "foldstrat" else "pooled", f"{f3(o['auroc'])} {ci(o['ci95'])}", f"{sg(o['delta_vs_L-CNV']['delta'])} {cis(o['delta_vs_L-CNV']['ci95'])} (p {o['delta_vs_L-CNV']['p_unadjusted_swap']})", f"{sg(o['delta_vs_L-LATE']['delta'])} {cis(o['delta_vs_L-LATE']['ci95'])} (p {o['delta_vs_L-LATE']['p_unadjusted_swap']})"])
L += ["### Paired Δ of the Pathology grade row", "", "All pre-event samples; same bootstrap draws; unadjusted swap-permutation p (the row is outside the max-T family). NDBE only: not estimable.", ""] + table(["CNV source", "Horizon", "Metric", "Grade AUROC [CI]", "Δ vs L-CNV [CI] (p)", "Δ vs L-LATE [CI] (p)"], rows)
L += ["### Supplement: L-CLIN and demographics only", "", "**Note.** The discovery cohort was matched on age, sex and Barrett's segment length (Killcoyne 2020, Supplementary Table 1), so demographic predictors are not expected to discriminate; the values below sit at or under 0.5, and under the pooled metric they also carry the stratified-CV fold-prevalence artefact (`docs/paper_horizon_answers.md`). L-CLIN = L2 logistic on grade, age at BE diagnosis, sex and Prague M; demographics only = the same without grade (`scripts/paper_plan/ha_clin.py`).", ""]
for s in ("their",):
    for p in ("pre", "pre_ndbe"):
        L += [f"{POP[p]} (identical for both CNV sources):", ""] + main(s, p, [("L-CLIN", "L-CLIN"), ("Demographics only", "L-CLIN-nograde")], grade_row=False)
L += ["**Method.** Score = ordinal grade (NDBE 0, ID 1, LGD 2), constant across repeats; pooled metric over all case–control pairs; fold-stratified metric over same-fold pairs in each repeat's outer folds, averaged over 10 repeats; IPCW weights from the reverse Kaplan–Meier, re-estimated per bootstrap draw. **Sources.** `feasibility/paper_plan/killcoyne_mm/set_C.csv` (grade), `horizons/samples.csv` (times, events), `cv/preds/` (folds, L-CNV, L-IMG).",
      "**Caveats.** (1) Ties: with three grade levels most case–control pairs are tied and count ½, which pulls the row toward 0.5. (2) Design caveat: AUROCs only; no absolute risk, calibration or PPV. (3) Pre-event samples contain no HGD/IMC (they are excluded by the endpoint definition), so the grade spans only NDBE, ID and LGD.", ""]
open(DOC, "w").write("\n".join(L) + "\n"); print("rendered", len(L))

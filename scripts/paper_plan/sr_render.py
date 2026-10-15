"""Append results to docs/paper_scanner_robustness.md below the pre-specification (abe96e9); text above the closing '---' line is not changed.
Usage: python sr_render.py RESULTS_COMMIT. Inputs: results/paper_final/scanner_robustness/{index,smoke_validation,answers}.json (+ downstream.json if item 4 ran)."""
import json, sys, os
RC = sys.argv[1]; R = "results/paper_final/scanner_robustness"; DOC = "docs/paper_scanner_robustness.md"; L_ = lambda f: json.load(open(f"{R}/{f}.json"))
IX = L_("index"); SV = L_("smoke_validation"); ANS = L_("answers"); DS = L_("downstream") if os.path.exists(f"{R}/downstream.json") else None
f3 = lambda x: "—" if x is None else f"{x:.3f}"; ci = lambda c: "" if not c or c[0] is None else f" [{c[0]:.3f}, {c[1]:.3f}]"; sg = lambda x: f"{x:+.3f}"; cis = lambda c: f" [{c[0]:+.3f}, {c[1]:+.3f}]"
doc = open(DOC).read(); head = doc[: doc.rindex("\n---")].rstrip("\n") + "\n\n---\n"; EX = IX["extractors"]; P = IX["primary"]; po = IX["pool"]; ck = IX["checks"]
L = ["", "## Results", "", f"Pre-specification commit abe96e9; results commit {RC}. Scripts `scripts/paper_plan/sr_*.py` (Slurm via `scripts/cluster/campaign.sh`, prefixes {ANS['prefixes']}). {ANS['compute']} Aggregates `results/paper_final/scanner_robustness/`; row-level outputs (tiles, embeddings, neighbour lists) on the cluster only. Figure `~/Downloads/be_paper_figs/v3/15_F_scanner_robustness.{{pdf,png}}`.", ""]
L += ["### Status", "", "| Item | Status |", "|---|---|"] + [f"| {k} | {v} |" for k, v in ANS["status"].items()] + [""]
L += ["### Answers", ""] + [f"{i}. {ANS[str(i)]}" for i in range(1, 6)] + [""]
L += ["### Checks", "",
      f"- Smoke validation: Prov-GigaPath re-embedded from the saved tiles of 2 slides vs the stored embeddings, per-tile cosine median {SV['cos_median']:.5f}, minimum {SV['cos_min']:.5f} (required ≥ 0.99 / ≥ 0.95): {'passed' if SV['passed'] else 'FAILED'}.",
      f"- Prov-GigaPath `coords_level` equal to UNI2-h's on {ck['gigapath_slides_coords_equal']} of {ck['slides']} slides: {'included as reference' if ck['gigapath_coords_equal_all_slides'] else 'excluded'}.",
      f"- UNI2-h sample representation vs `img_mean.csv`: max |difference| {ck['uni2_img_mean_max_abs_diff']:.2e} (≤ 1e-4 required).",
      f"- Pool: {po['samples']} samples from {po['patients']} patients ({', '.join(f'{k} {v}' for k, v in po['samples_by_scanner'].items())}); {po['endoscopies']} endoscopies, {po['endoscopies_on_both_scanners']} with samples on both scanners; {po['tiles_S4']:,} tiles for S4.",
      f"- Feature dimensions: {', '.join(f'{k} {v}' for k, v in IX['dims'].items())}.", ""]
L += ["### 2–3. Primary robustness index (pool of dual-scanner patients, cosine, k = 10) and decision", "",
      "| Extractor | R_10 [95% CI] | Chance | SO / OS / SS neighbours | ΔR vs UNI2-h [95% CI] | Ratio to UNI2-h | CI above 0 | Ratio ≥ 1.25 | Qualifies |", "|---|---|---|---|---|---|---|---|---|"]
for e in EX:
    p = P[e]; nc = p["neighbour_classes_at_k"]; d = IX["decision"]["candidates"].get(e)
    tail = "— | — | — | — | — (incumbent)" if d is None else f"{sg(d['dR'])}{cis(d['dR_ci95'])} | {d['ratio_to_uni2']:.2f} | {'yes' if d['crit_ci_above_0'] else 'no'} | {'yes' if d['crit_ratio_ge_1.25'] else 'no'} | {('yes' if d['qualifies'] else 'no') if d['eligible'] else 'not eligible (reference)'}"
    L.append(f"| {e} | {f3(p['R'])}{ci(p['ci95'])} | {f3(p['chance'])} | {nc['SO']} / {nc['OS']} / {nc['SS']} | {tail} |")
L += ["", f"**Decision: {IX['decision']['decision']}.**", ""]
S = IX["secondary"]
L += ["### Secondary analyses (report only)", "", "**S1. R_k at selected k** (pool of dual-scanner patients).", "", "| Extractor | " + " | ".join(f"k = {k}" for k in (1, 5, 10, 20, 50)) + " |", "|---|" + "---|" * 5]
for e in EX: cv = {c["k"]: c for c in S["S1_curve"][e]}; L.append(f"| {e} | " + " | ".join(f"{f3(cv[k]['R'])}{ci(cv[k]['ci95'])}" for k in (1, 5, 10, 20, 50)) + " |")
L += ["", "| Analysis | " + " | ".join(EX) + " | Chance |", "|---|" + "---|" * (len(EX) + 1)]
LAB = {"S2_all_sample_pool": "S2 all 676 samples as neighbours, k = 10", "S3_zscored": "S3 z-scored features, k = 10", "S4_tile_level": "S4 tile level, k = 50", "S5_deJong_form": "S5 de Jong form (same patient / same scanner), k = 10", "S6_other_endoscopy": "S6 other-endoscopy neighbours only, k = 10"}
for s, lab in LAB.items(): L.append(f"| {lab} | " + " | ".join(f"{f3(S[s][e]['R'])}{ci(S[s][e]['ci95'])}" for e in EX) + f" | {f3(S[s]['UNI2-h']['chance'])} |")
L += ["", "| Analysis | " + " | ".join(f"Δ {e} − UNI2-h [95% CI]" for e in ("Virchow2", "H-optimus-0")) + " | Same direction as primary |", "|---|---|---|---|"]
for s, lab in LAB.items():
    A = S["agreement"][s]; L.append(f"| {lab} | " + " | ".join(f"{sg(A[e]['dR'])}{cis(A[e]['dR_ci95'])}" for e in ("Virchow2", "H-optimus-0")) + " | " + ", ".join(f"{e}: {'yes' if A[e]['same_sign_as_primary'] else 'no'}" for e in ("Virchow2", "H-optimus-0")) + " |")
L += ["", "**S7. Scanner probe** (all 676 samples, 5-fold patient-grouped, pooled out-of-fold AUROC): " + "; ".join(f"{e} {S['S7_scanner_probe'][e]['auroc_oof']:.4f}" for e in EX) + ".", ""]
L += ["### 4. Downstream run", ""] + (DS["lines"] if DS else ["Not run: the pre-specified rule kept UNI2-h."]) + [""]
L += ["### Deviations and caveats", ""] + [f"- {d}" for d in ANS["deviations"]] + [""]
open(DOC, "w").write(head + "\n".join(L)); print("SR RENDER DONE")

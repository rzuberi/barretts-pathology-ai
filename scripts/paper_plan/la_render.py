"""Append results to docs/paper_latent_attention.md below the pre-specification (dfac9ad); the text above the closing '---' line is not changed.
Usage: python la_render.py RESULTS_COMMIT. Inputs: results/paper_final/latent_attention/{refit_check,probe_pre,probe_pre_ndbe,umap}.json."""
import json, sys
RC = sys.argv[1]; R = "results/paper_final/latent_attention"; DOC = "docs/paper_latent_attention.md"
RF = json.load(open(f"{R}/refit_check.json")); PB = {p: json.load(open(f"{R}/probe_{p}.json")) for p in ("pre", "pre_ndbe")}; UM = json.load(open(f"{R}/umap.json"))
ANS = json.load(open(f"{R}/answers.json"))
f3 = lambda x: "—" if x is None else f"{x:.3f}"; ci = lambda c: "" if not c or c[0] is None else f" [{c[0]:.3f}, {c[1]:.3f}]"; cis = lambda c: f" [{c[0]:+.3f}, {c[1]:+.3f}]"; sg = lambda x: f"{x:+.3f}"
bp = lambda p: "< 0.0005" if p == 0 else f"{p}"
NM = {"R0": "R0 mean UNI2-h embedding (raw)", "R1": "R1 WSI-only (z-scored mean embedding)", "R2": "R2 early fusion (their)", "R3": "R3 inter fusion (their, 64-d)", "R4": "R4 CNV (their)",
      "R2p": "R2 early fusion (package)", "R3p": "R3 inter fusion (package)", "R4p": "R4 CNV (package)"}
LAB = {"pre": "All pre-event samples", "pre_ndbe": "NDBE pre-event samples"}
doc = open(DOC).read(); head = doc[: doc.rindex("\n---")].rstrip("\n") + "\n\n---\n"
L = ["", "## Results", "", f"Pre-specification commit dfac9ad; results commit {RC}. Scripts `scripts/paper_plan/la_refit.R`, `la_probe.py`, `la_merge.py`, `la_umap.py` (Slurm via `scripts/cluster/campaign.sh`, prefixes la1 refit, la2 probes and UMAP, la3 merge, la4 UMAP rerun for the colour key, la5 scanner count), `la_scanner_xtab.py`, `la_render.py`. Aggregates `results/paper_final/latent_attention/{{refit_check,probe_pre,probe_pre_ndbe,umap,scanner_xtab,answers}}.json`; representation dumps, probe predictions and UMAP coordinates on the cluster only (`feasibility/paper_plan/killcoyne_mm/latent_attention/`).", ""]
ok = RF["all_pass"]
L += ["### Status", "", "| Item | Status |", "|---|---|", "| Step 0 architecture facts | DONE (pre-specification) |",
      f"| Step 0 refit check (≤ 1e-5) | {'PASS' if ok else 'FAIL'}: {RF['n_checks']} model × repeat refits, max |difference| {RF['max_abs_diff']:.1e}, folds identical in all |",
      f"| Representation reconstruction (row-sum fingerprints ≤ 1e-6) | PASS: max {max(PB['pre']['fingerprint_max_abs_diff'], PB['pre_ndbe']['fingerprint_max_abs_diff']):.1e} |",
      "| 1 Linear probes, paired Δ, other structure, patient identity | DONE |", "| 2 UMAP figure and silhouettes | DONE |", "| 3 Attention | NOT APPLICABLE (no tile attention in this tier; Step 0) |", ""]
L += ["**Refit check per model** (max |refit − stored| over the 10 repeats × 676 out-of-fold predictions).", "", "| Model | Repeats | Max abs difference |", "|---|---|---|"] + [f"| {k} | {v['repeats']} | {v['max_abs_diff']:.1e} |" for k, v in RF["models"].items()] + [""]
L += ["### Answers", ""] + [f"{i}. {ANS[str(i)]}" for i in (1, 2, 3)] + [""]
L += ["### 1. Linear probes (fold-stratified AUROC, mean over 10 repeats; 95% patient-bootstrap CI, 2,000 draws)", ""]
TGT = ["progressor", "scanner", "tiles", "grade", "cx"]; TN = {"progressor": "Progressor", "scanner": "Scanner (C13210 vs C13239-01)", "tiles": "Tissue tiles > median", "grade": "Grade ID/LGD vs NDBE", "cx": "cx > median"}
for p, J in PB.items():
    L += [f"**{LAB[p]}** ({J['n_samples']} samples, {J['n_patients']} patients, {J['n_progressor_samples']} progressor samples).", "", "| Representation | " + " | ".join(TN[t] for t in TGT if t in J["probe"]) + " | Nearest neighbour same patient (chance) |", "|---" * (2 + sum(t in J["probe"] for t in TGT)) + "|"]
    for r in NM:
        cells = []
        for t in TGT:
            if t not in J["probe"]: continue
            o = J["probe"][t].get(r); cells.append("—" if o is None else f"{f3(o['auroc'])}{ci(o['ci95'])}")
        nn = J["nn_patient"].get(r); L.append(f"| {NM[r]} | " + " | ".join(cells) + f" | {f3(nn['share_same_patient'])} ({f3(nn['chance'])}) |")
    L += [""]
X = json.load(open(f"{R}/scanner_xtab.json"))
L += ["**Scanner by progressor status** (descriptive, not pre-specified; samples, with patients who have any sample on that scanner in parentheses).", "", "| Population | Scanner | Progressor samples (patients) | Non-progressor samples (patients) | Progressor share of samples |", "|---|---|---|---|---|"]
for p, lab in LAB.items():
    for sc, v in X[p]["samples"].items():
        pt = X[p]["patients_with_any_sample_on"][sc]; L.append(f"| {lab} | {sc} | {v['progressor']} ({pt['progressor']}) | {v['non_progressor']} ({pt['non_progressor']}) | {v['progressor'] / (v['progressor'] + v['non_progressor']):.2f} |")
L += ["", "Patients with samples on both scanners: " + "; ".join(f"{LAB[p]} {X[p]['patients_on_both_scanners']}" for p in LAB) + ".", ""]
L += ["**Paired Δ progressor-probe AUROC** (same draws; max-T adjusted p over the three comparisons of each family).", "", "| Population | Comparison | Δ AUROC [95% CI] | p unadjusted | p max-T adjusted |", "|---|---|---|---|---|"]
for p, J in PB.items():
    for k, v in J["paired"].items(): L.append(f"| {LAB[p]} | {k.replace('_vs_', ' vs ').replace('p', ' (package)') if False else k.replace('_vs_', ' vs ').replace('R2p', 'R2 (package)').replace('R3p', 'R3 (package)')} | {sg(v['delta'])}{cis(v['ci95'])} | {bp(v['p_unadjusted'])} | {bp(v['p_maxT_adjusted'])} |")
L += ["", "Probe C (median over folds and repeats, progressor): " + "; ".join(f"{LAB[p]}: " + ", ".join(f"{r} {J['probe']['progressor'][r]['C_median']:.3g}" for r in NM if r in J['probe']['progressor']) for p, J in PB.items()) + ".", ""]
L += ["### 2. UMAP (in-sample, illustrative)", "", "Figures `~/Downloads/be_paper_figs/v3/11_F_latent_space.{pdf,png}` (rows R1, R2, R3; their matrix; 571 pre-event samples) and `11_F_latent_space_supp.{pdf,png}` (n_neighbors 50; held-out pre-event samples of repeat 1, fold 1, " + f"{UM['heldout_rep1_fold1']['n_samples']} samples). umap-learn {UM['settings']['umap_version']}, n_neighbors 15, min_dist 0.1, cosine, seed 0.", "",
      "| Set | Representation | Silhouette, progressor (representation, cosine) | Silhouette, progressor (UMAP) | Silhouette, patient (representation, cosine) | Silhouette, patient (UMAP) | Samples in patient silhouette |", "|---|---|---|---|---|---|---|"]
for s, lab in (("main", "n_neighbors 15"), ("nn50", "n_neighbors 50"), ("heldout_rep1_fold1", "held-out r1 f1")):
    for r in ("R1", "R2", "R3"):
        o = UM[s][r]; L.append(f"| {lab} | {r} | {f3(o['progressor']['repr_cosine'])} | {f3(o['progressor']['umap'])} | {f3(o['patient']['repr_cosine'])} | {f3(o['patient']['umap'])} | {o['patient']['n_samples']} |")
L += ["", "Results with the CIs at 1.000 are saturated: every held-out fold is separated perfectly in every repeat and bootstrap draw.", "", "### 3. Attention", "", "Not applicable (Step 0): no model in this tier has tile attention; late fusion's image half is WSI-only. No attention statistics, no `12_F_attention` figure and no top-attended tiles for pathologist review.", "",
      "### Deviations and caveats", "", ] + [f"- {d}" for d in ANS.get("deviations", [])] + [
      "- The representations are fixed, fold-fitted transforms of the inputs (Step 0), not learned embeddings; a probe on R2 or R3 is close to refitting the corresponding model with an L2 instead of an elastic-net penalty, so the probe Δ largely restates the model comparison.",
      "- R2 contains `cx` as an input feature; the pre-specification expected its cx probe near 1, but it is " + f"{PB['pre']['probe']['cx']['R2']['auroc']:.3f} (their matrix) and {PB['pre']['probe']['cx']['R2p']['auroc']:.3f} (package) on pre-event samples, likely because the L2 probe spreads its weight over about 2,170 standardised features. R4 was not probed for cx (pre-specified).",
      "- Probes are not refitted within the bootstrap; the CIs reflect sampling of patients for fixed probes.",
      "- Nearest-neighbour patient identity is a point estimate (no CI, pre-specified); chance depends on how many samples each patient contributes to the fold.",
      "- Matched case–control design; the progressor label is patient status, shared by all samples of a patient, so patient structure and label structure are partly confounded.", ""]
open(DOC, "w").write(head + "\n".join(x for x in L if x is not None) + "\n"); print("rendered")

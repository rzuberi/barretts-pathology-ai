"""Append results to docs/paper_plan_killcoyne_checks.md (the pre-specification above is kept as is)."""
import json, sys
R = json.load(open("results/paper_final/killcoyne_checks.json")); FZ = json.load(open("models/killcoyne_frozen_v1/MANIFEST.json")); MI = json.load(open("models/killcoyne_frozen_v1/model_info.json"))
commit = sys.argv[1] if len(sys.argv) > 1 else "<results commit>"
f = lambda x: "—" if x is None else f"{x:.3f}"; ci = lambda c: "" if not c else f" [{f(c[0])}, {f(c[1])}]"
dl = lambda d: "—" if not d else f"{d['delta']:+.3f}{ci(d['ci95'])} (p {d['perm_p']})"
SRC = {"their": "their shipped CNV matrix", "pkg": "package features from our counts"}
SUBN = {"all_C": "All of set C", "a_NDBE_only": "(a) NDBE samples only", "b_before_first_HGD_IMC": "(b) before the first HGD/IMC", "c_NDBE_and_before": "(c) NDBE and before the first HGD/IMC"}
L = ["", "---", "", "## Results", "", f"Sources: `results/paper_final/killcoyne_checks.json` · scripts `scripts/paper_plan/kc_*.{{py,R}}` · frozen models `models/killcoyne_frozen_v1/` · results commit {commit}. Row-level outputs on the cluster under `feasibility/paper_plan/killcoyne_mm/checks/`.", "",
     "### Status", "", "| Item | Status |", "|---|---|", "| 1 Selection-adjusted p | DONE |", "| 2 Fold-honest image scaling | DONE |", "| 3 Sample subsets | DONE |", "| 4 Pathology grade | DONE |", "| 5 Frozen models | DONE |", ""]
ev = R["event_definition"]; L += ["Subsets: " + "; ".join(f"{SUBN[k]} {v['n_samples']} samples, {v['n_patients']} patients ({v['n_P_patients']} P)" for k, v in R["subsets"].items()) + f". First HGD/IMC dated from an HGD/IMC record for {ev['dated_from_HGD_IMC_record']} of {ev['P_patients_in_C']} P patients; {ev['fallback_final_endoscopy']} use the final endoscopy.", ""]
L += ["### 1. Selection-adjusted permutation p (max-T over the 8 non-CNV arms, all of set C)", ""]
for src in ["their", "pkg"]:
    L += [f"CNV source: {SRC[src]}.", "", "| Arm | Δ per sample | p unadjusted | p max-T | Δ patient max | p unadjusted | p max-T |", "|---|---|---|---|---|---|---|"]
    for a, v in R["item1_max_T"][src].items():
        s, m = v["per_sample_auroc"], v["patient_max_auroc"]; L.append(f"| {a} | {s['delta']:+.3f} | {s['p_unadjusted']} | {s['p_max_T_adjusted']} | {m['delta']:+.3f} | {m['p_unadjusted']} | {m['p_max_T_adjusted']} |")
    L.append("")
L += ["### 2. Whole-set vs fold-honest image scaling (all of set C)", "", "| CNV source | Arm | Scaling | Per-sample AUROC | Patient-max AUROC | Δ per sample vs L-CNV | Δ patient max vs L-CNV |", "|---|---|---|---|---|---|---|"]
for src in ["their", "pkg"]:
    X = R["items2_3"][src]["all_C"]
    for a in ["L-IMG", "L-EARLY", "L-LATE"]:
        for lab, key in [("whole set", a), ("fold-honest", a + " (fold-honest)")]:
            v = X[key]; L.append(f"| {src} | {a} | {lab} | {f(v['per_sample_auroc'])}{ci(v['ci95']['per_sample_auroc'])} | {f(v['patient_max_auroc'])}{ci(v['ci95']['patient_max_auroc'])} | {dl(v['delta_vs_L_CNV']['per_sample_auroc'])} | {dl(v['delta_vs_L_CNV']['patient_max_auroc'])} |")
L += ["", "### 3. Existing predictions on sample subsets (no refit)", ""]
for src in ["their", "pkg"]:
    for sub in ["a_NDBE_only", "b_before_first_HGD_IMC", "c_NDBE_and_before", "all_C"]:
        X = R["items2_3"][src][sub]; s0 = R["subsets"][sub]
        L += [f"**{SUBN[sub]}**, CNV source {src} ({s0['n_samples']} samples, {s0['n_patients']} patients).", "", "| Arm | Per-sample AUROC | Δ vs L-CNV | Patient-max AUROC | Δ vs L-CNV |", "|---|---|---|---|---|"]
        for a, v in X.items():
            d = v.get("delta_vs_L_CNV", {}); L.append(f"| {a} | {f(v['per_sample_auroc'])}{ci(v['ci95']['per_sample_auroc'])} | {dl(d.get('per_sample_auroc'))} | {f(v['patient_max_auroc'])}{ci(v['ci95']['patient_max_auroc'])} | {dl(d.get('patient_max_auroc'))} |")
        L.append("")
L += ["### 4. Pathology grade", ""]
for src in ["their", "pkg"]:
    for sub in ["all_C", "a_NDBE_only"]:
        X = R["item4"][src][sub]; L += [f"**{SUBN[sub]}**, CNV source {src}.", "", "| Arm | Per-sample AUROC | Patient-max AUROC | Δ per sample | Δ patient max |", "|---|---|---|---|---|"]
        for a, v in X.items():
            d = v.get("delta_vs_L_CNV+GRADE") or v.get("delta_vs_L_CNV"); ref = " vs L-CNV+GRADE" if "delta_vs_L_CNV+GRADE" in v else (" vs L-CNV" if d else "")
            L.append(f"| {a} | {f(v['per_sample_auroc'])}{ci(v['ci95']['per_sample_auroc'])} | {f(v['patient_max_auroc'])}{ci(v['ci95']['patient_max_auroc'])} | {dl(d['per_sample_auroc']) + ref if d else '—'} | {dl(d['patient_max_auroc']) + ref if d else '—'} |")
        L.append("")
L += ["### 5. Frozen models (their CNV matrix, set-C scaling)", "", f"Bundle SHA-256 `{FZ['bundle_sha256']}` (MANIFEST.json lists each file's SHA-256).", "", "| Model | λ | Features | Non-zero coefficients | In-sample AUROC (not a performance estimate) |", "|---|---|---|---|---|"]
for nm in ["L_CNV", "L_IMG", "L_EARLY"]:
    v = MI[nm]; L.append(f"| {nm.replace('_', '-')} | {v['lambda']:.5f} | {v['n_features']} | {v['nonzero']} | {MI['insample_check']['auroc_' + nm]:.3f} |")
L += [f"| L-LATE | mean of L-CNV and L-IMG | — | — | {MI['insample_check']['auroc_L_LATE']:.3f} |", "", "Files: " + ", ".join(f"`{k}`" for k in FZ["files"]) + ".", "", "INTERPRETATION_LINE", ""]
open("docs/paper_plan_killcoyne_checks.md", "a").write("\n".join(L)); print("rendered")

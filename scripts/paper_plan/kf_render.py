"""Append results to docs/paper_plan_killcoyne_final.md (the pre-specification above is kept as is)."""
import json, sys
R = json.load(open("results/paper_final/killcoyne_final.json")); commit = sys.argv[1] if len(sys.argv) > 1 else "<results commit>"
f = lambda x: "—" if x is None else f"{x:.3f}"; ci = lambda c: "" if not c else f" [{f(c[0])}, {f(c[1])}]"
dl = lambda d: "—" if not d else f"{d['delta']:+.3f}{ci(d['ci95'])} (p {d['perm_p']})"
SUBN = {"all_C": "All of set C", "a_NDBE_only": "(a) NDBE samples only", "b_before_first_HGD_IMC": "(b) before the first HGD/IMC", "c_NDBE_and_before": "(c) NDBE and before the first HGD/IMC"}
L = ["", "---", "", "## Results", "", f"Sources: `results/paper_final/killcoyne_final.json` · scripts `scripts/paper_plan/kf_*.py` · results commit {commit}. Row-level outputs on the cluster under `feasibility/paper_plan/killcoyne_mm/final/`.", "",
     "### Status", "", "| Item | Status |", "|---|---|", "| 1 Fold-stratified AUROC | DONE |", "| 2 Selection-adjusted p | PARTIAL (3 of the 8 arms exist under the stratified CV; no refit) |", "| 3 ACE-B power | DONE (patient status from the database timeline, not the owner's per-patient labels) |", "",
     "### 1. Fold-stratified AUROC (within-fold P–NP pairs, mean over 10 repeats)", "", "| CNV source | Subset | Samples / patients (P) | L-CNV | L-IMG (Δ) | L-EARLY (Δ) | L-LATE (Δ) | Intercept-only |", "|---|---|---|---|---|---|---|---|"]
for src in ["their", "pkg"]:
    for sub in ["all_C", "a_NDBE_only", "b_before_first_HGD_IMC", "c_NDBE_and_before"]:
        v = R["item1_fold_stratified"][src][sub]; cell = lambda a: f"{f(v[a]['fold_stratified_auroc'])}{ci(v[a]['ci95'])}" + (f"; Δ {dl(v[a]['delta_vs_L_CNV'])}" if "delta_vs_L_CNV" in v[a] else "")
        L.append(f"| {src} | {SUBN[sub]} | {v['n_samples']} / {v['n_patients']} ({v['n_P_patients']}) | {cell('L-CNV')} | {cell('L-IMG')} | {cell('L-EARLY')} | {cell('L-LATE')} | {f(v['intercept_only_fold_stratified_auroc'])} |")
L += ["", "### 2. Max-T adjusted p over the 3 non-CNV arms available under the stratified CV", "", "Adjusted p over 3 arms is a lower bound on the adjusted p over the checks' 8-arm family.", "",
      "| CNV source / subset | Arm | Metric | Δ vs L-CNV | p unadjusted | p max-T |", "|---|---|---|---|---|---|"]
for key, arms in R["item2_max_T"].items():
    for a, m in arms.items():
        for k in ["per_sample_auroc", "patient_max_auroc", "fold_stratified_auroc"]:
            L.append(f"| {key} | {a} | {k.replace('_', ' ')} | {m[k]['delta']:+.3f} | {m[k]['p_unadjusted']} | {m[k]['p_max_T_adjusted']} |")
Z = R["aceb_size"]
L += ["", "### 3. ACE-B power for the primary contrast (L-LATE pkg − L-CNV pkg, per-sample AUROC, NDBE samples, one-sided)", "",
      f"Manifest: {Z['manifest_samples']} sWGS samples from {Z['manifest_cases']} cases; {Z['IM_samples']} NDBE (IM) samples from {Z['cases_with_IM']} cases. NDBE cases by database status: {Z['IM_cases_by_status']}; NDBE samples by status: {Z['IM_samples_by_status']}. Owner's aggregate split: {Z['owner_split']['progressors']} progressors, {Z['owner_split']['non_progressors']} non-progressors, {Z['owner_split']['prevalent_HGD_IMC']} prevalent HGD/IMC ({Z['owner_split']['patients']} patients, {Z['owner_split']['samples']} samples).", "",
      "Rows marked [post hoc] were added after the pre-specified runs showed two problems: (1) simulated patients bring set C's NDBE samples (about 6 per patient) whereas ACE-B's cases have 1–3, so the pre-specified simulation has 4–5 times ACE-B's sample count; the S1m/S2m rows keep, for each drawn patient, a number of its NDBE samples drawn from ACE-B's per-case distribution for its class. (2) The pre-specified +0.021 scenario blends 20% L-LATE into L-CNV, which makes the two arms nearly identical and the difference nearly noise-free, so its power is not a conservative figure; the last column instead shifts each full-effect simulation's lower bound down by 0.038 (the gap between +0.059 and +0.021), keeping the full-effect variance.", "",
      "| Size scenario | Effect | P / NP patients | ACE-B NDBE samples | Median simulated samples | Set-C Δ (blend w) | Median simulated Δ | Median lower bound | Power (lower bound > 0) | Power at +0.021 by shift [post hoc] |", "|---|---|---|---|---|---|---|---|---|---|"]
for k, v in R["item3_power"].items():
    S, E = k.split("_"); lab = Z["scenarios"][S.rstrip("m")]["label"] + (" [post hoc: ACE-B sample count per patient]" if S.endswith("m") else "")
    sh = v.get("power_E2_shift_post_hoc"); nA = Z["scenarios"][S.rstrip("m")]["n_samples"]
    L.append(f"| {S}: {lab} | {'development (+0.059)' if E == 'E1' else 'lower bound (+0.021, blend)'} | {v['n_P']} / {v['n_NP']} | {nA if nA is not None else '—'} | {v['median_simulated_samples']} | {v['set_C_NDBE_delta']:+.3f} ({v['blend_w']}) | {v['median_simulated_delta']:+.3f} | {v['median_lower95']:+.3f} | {v['power_lower95_above_0']:.3f}{ci(v['power_95ci'])} | {'—' if sh is None else f'{sh:.3f}'} |")
L += ["", "INTERPRETATION_LINE", ""]
open("docs/paper_plan_killcoyne_final.md", "a").write("\n".join(L)); print("rendered")

"""Renders docs/paper_final_inputs.md from results/paper_final/*.json and figs/*.json. Args: PRESPEC RESULTS_COMMIT PRESPEC_TEXT_FILE."""
import json, os, sys
R = "results/paper_final"; PRESPEC, RESC, PRETXT = (sys.argv + ["e6e00c0", "pending", ""])[1:4]; PRE = open(PRETXT).read() if PRETXT and os.path.exists(PRETXT) else ""
M = json.load(open(f"{R}/final_checks.json")); FJ = {n: json.load(open(f"{R}/figs/{n}.json")) for n in ["F_intro_forest", "F_table_forest", "F_D1_risk_groups", "F_D2_latent", "F_D3_attention", "F_D4_cnv_change", "F_D5_false_positives", "F_D6_false_negatives"]}
def f3(x): return "n/a" if x is None else (f"{x:.3f}" if isinstance(x, (int, float)) and not isinstance(x, bool) else str(x))
def s3(x): return "n/a" if x is None else f"{x:+.3f}"
def ci(c): return "n/a" if not c or c[0] is None else (f"[{c[0]:+.3f}, {c[1]:+.3f}]" if (c[0] < 0 or c[1] < 0) else f"[{c[0]:.3f}, {c[1]:.3f}]")
def e3(v): return f"{f3(v[0])} {ci(v[1:])}"
def tb(hdr, rows): return "| " + " | ".join(hdr) + " |\n|" + "---|" * len(hdr) + "\n" + "\n".join("| " + " | ".join(str(v) for v in r) + " |" for r in rows)
NAME = {"C2_grade_maxsofar": "Clinical (C2)", "cnv_only": "CNV (release RF)", "cnv_km": "CNV (Killcoyne method)", "image_only": "WSI", "early_fusion": "Early fusion", "intermediate_fusion": "Intermediate fusion", "late_mean": "Late fusion (mean)", "coattention_fusion": "Co-attention (suppl.)", "late_stack_logit": "Late stack (suppl.)"}
K1, K2, K3, K4, K5 = M["K1"], M["K2"], M["K3"], M["K4"], M["K5"]; SRC = lambda f, s="pk_main.py": f"`results/paper_final/{f}` · `scripts/paper_plan/{s}` · commit {RESC}"; L = []; P = L.append
FT = FJ["F_table_forest"]; FI = FJ["F_intro_forest"]; ne = FT["n_events"]
P(f"""# BE paper: final checks and figures (K1–K5, F-intro, F-table, F-D1–F-D6)

## 1. Header
- Date: 26 September 2026. Commit at start: `4e0cc2d`. Pre-specification commit: `{PRESPEC}` (verbatim in §6). Results commit: `{RESC}`; this text is the next commit. Frozen release only; nothing retrained.
- Scripts: `scripts/paper_plan/pk_gpu.py` (K4 ablation), `pk_main.py` (K1–K5, all figures), `pk_render.py`, `pk_check_report.py`. Results: `results/paper_final/final_checks.json`, `results/paper_final/k4_ablation.json`; figures `results/paper_final/figs/<name>.png|pdf` with `<name>.json` holding the plotted numbers. Row-level ablation scores: `feasibility/paper_plan/pk_ablation_rows.csv` (cluster). No tissue images are committed; montages remain at `feasibility/paper_plan/figs/attention/` (cluster).
- Analysis decisions applied: primary population = 82 discovery patients ({ne['discovery']['LGD2plus'][1]} LGD2+ events, {ne['discovery']['E_HGD'][1]} HGD/IMC events); secondary = stratified all-150, then pooled (confounded); primary endpoint LGD2+; E-HGD labelled as added after seeing the CNV results; tissue amount an unresolved confound; arms as listed; ERIN grade head excluded.

## 2. Status table
| item | status | key number / path |
|---|---|---|
| K1 FP by stratum | DONE | late fusion later HGD+: discovery FP {K1['per_stratum']['late_mean']['discovery']['laterHGD_FP']}/{K1['per_stratum']['late_mean']['discovery']['FP']} vs TN {K1['per_stratum']['late_mean']['discovery']['laterHGD_TN']}/{K1['per_stratum']['late_mean']['discovery']['TN']} (p {K1['per_stratum']['late_mean']['discovery']['fisher_p']}); validation FP {K1['per_stratum']['late_mean']['validation']['laterHGD_FP']}/{K1['per_stratum']['late_mean']['validation']['FP']} vs TN {K1['per_stratum']['late_mean']['validation']['laterHGD_TN']}/{K1['per_stratum']['late_mean']['validation']['TN']} (p {K1['per_stratum']['late_mean']['validation']['fisher_p']}) |
| K2 FN within discovery | DONE | late fusion FN {K2['late_mean']['n_FN']} vs TP {K2['late_mean']['n_TP']}; kept_tiles MW p {K2['late_mean']['comparison']['kept_tiles']['mannwhitney_p']}, cx p {K2['late_mean']['comparison']['cx']['mannwhitney_p']}, reads p {K2['late_mean']['comparison']['reads']['mannwhitney_p']} |
| K3 Latent within discovery | DONE | probe AUROC discovery: image {f3(K3['image_only']['probe_auroc_discovery'])}, intermediate {f3(K3['intermediate_fusion']['probe_auroc_discovery'])}, early {f3(K3['early_fusion']['probe_auroc_discovery'])} (pooled {f3(K3['image_only']['probe_auroc_pooled_item6'])} / {f3(K3['intermediate_fusion']['probe_auroc_pooled_item6'])} / {f3(K3['early_fusion']['probe_auroc_pooled_item6'])}) |
| K4 CNV use and rank shift | {'DONE' if isinstance(K4['ablation'], dict) else 'PARTIAL'} | {('discovery CNV-destroyed Δ: early ' + s3(K4['ablation']['early_fusion']['discovery_82']['cnv_permuted_jointly_mean_of_50']['delta_auroc']) + ', intermediate ' + s3(K4['ablation']['intermediate_fusion']['discovery_82']['cnv_permuted_jointly_mean_of_50']['delta_auroc']) + ', co-attention ' + s3(K4['ablation']['coattention_fusion']['discovery_82']['cnv_permuted_jointly_mean_of_50']['delta_auroc'])) if isinstance(K4['ablation'], dict) else 'ablation pending'}; Spearman CNV vs late (discovery) {f3(K4['rank_shift_discovery']['spearman_cnv_vs_late'])}, NRI {K4['rank_shift_discovery']['categorical_NRI']} |
| K5 Risk groups within discovery | DONE | late fusion LGD2+ tertile rates {', '.join(f3(K5['LGD2plus']['late_mean']['groups'][g]['rate']) for g in ['low', 'moderate', 'high'])}; OR high vs low {K5['LGD2plus']['late_mean']['OR_high_vs_low_haldane']}; trend p {K5['LGD2plus']['late_mean']['cochran_armitage_z_p'][1]} |
| F-intro | DONE | `results/paper_final/figs/F_intro_forest.png` |
| F-table | DONE | `results/paper_final/figs/F_table_forest.png` |
| F-D1 | DONE | `results/paper_final/figs/F_D1_risk_groups.png` |
| F-D2 | DONE | `results/paper_final/figs/F_D2_latent.png` |
| F-D3 | DONE | `results/paper_final/figs/F_D3_attention.png` |
| F-D4 | {'DONE' if isinstance(K4['ablation'], dict) else 'PARTIAL'} | `results/paper_final/figs/F_D4_cnv_change.png` |
| F-D5 | DONE | `results/paper_final/figs/F_D5_false_positives.png` |
| F-D6 | DONE | `results/paper_final/figs/F_D6_false_negatives.png` |

## 3. Checks

### K1. False positives by stratum
**Question.** Where does the pooled FP excess of later HGD+ live?

**Status.** DONE. **Pre-specification.** `{PRESPEC}` §K1.

""" + tb(["model", "stratum", "FP", "TN", "later HGD+ FP (rate [Wilson])", "later HGD+ TN (rate [Wilson])", "Fisher p"], [[NAME[a], l, v["FP"], v["TN"], f"{v['laterHGD_FP']} ({f3(v['rate_FP'])} {ci(v['wilson_FP'])})", f"{v['laterHGD_TN']} ({f3(v['rate_TN'])} {ci(v['wilson_TN'])})", v["fisher_p"]] for a, d in K1["per_stratum"].items() for l, v in d.items()]) + "\n\nValidation non-progressors, logistic later HGD+ ~ FP + baseline grade + log kept_tiles:\n\n" + tb(["model", "OR FP [CI]", "p", "OR log tiles", "OR baseline grade", "n / events", "separated"], [[NAME[a], f"{f3(v.get('OR_FP'))} {ci(v.get('ci'))}", v.get("p"), f3(v.get("OR_log_tiles")), f3(v.get("OR_baseline_grade")), f"{v.get('n')} / {v.get('events')}", v.get("separated", v.get("error"))] for a, v in K1["validation_logistic"].items()]) + f"""

Spearman of kept_tiles with later HGD+ among validation non-progressors: {K1['spearman_tiles_vs_laterHGD_validation_nonprogressors']}. Discovery non-progressors: {K1['discovery_nonprogressors']}.

**Method.** Pooled operating point (follow-up predictions). **Sources.** {SRC('final_checks.json')} (`K1`). **Caveats.** Validation non-progressors number {K1['validation_logistic']['late_mean'].get('n')} with {K1['validation_logistic']['late_mean'].get('events')} later-HGD+ events; logistic fits with three covariates are near separation for some models.

### K2. False negatives within discovery
**Question.** Does the FN vs TP difference survive when both groups are discovery progressors?

**Status.** DONE. **Pre-specification.** `{PRESPEC}` §K2.

""" + "\n\n".join(f"**{NAME[a]}** (FN {v['n_FN']}, TP {v['n_TP']}, discovery progressors only)\n\n" + tb(["variable", "FN median (n)", "TP median (n)", "test", "p"], [[c, f"{w['FN'][0]} ({w['FN'][1]})", f"{w['TP'][0]} ({w['TP'][1]})", "Mann-Whitney", w["mannwhitney_p"]] if "mannwhitney_p" in w else [c, str(w["table"]), "", "Fisher (2×2 only)", w["fisher_p"]] for c, w in v["comparison"].items()]) for a, v in K2.items()) + f"""

**Sources.** {SRC('final_checks.json')} (`K2`). **Caveats.** Reads exist only for discovery-sheet profiles; small FN groups.

### K3. Latent space within discovery
**Question.** How much of the pooled probe performance was stratum?

**Status.** DONE. **Pre-specification.** `{PRESPEC}` §K3.

""" + tb(["representation", "probe AUROC pooled (item 6)", "probe AUROC discovery [CI]", "kNN-10 discovery [CI]", "Δ probe vs image (discovery)", "Δ probe vs CNV (discovery)"], [[fam, f3(v["probe_auroc_pooled_item6"]), f"{f3(v['probe_auroc_discovery'])} {ci(v['probe_ci'])}", f"{f3(v['knn10_auroc_discovery'])} {ci(v['knn_ci'])}", e3(v["probe_delta_vs_image_only_discovery"]) if "probe_delta_vs_image_only_discovery" in v else "—", e3(v["probe_delta_vs_cnv_only_discovery"]) if "probe_delta_vs_cnv_only_discovery" in v else "—"] for fam, v in K3.items()]) + f"""

n {K3['image_only']['n']} discovery patients, {K3['image_only']['events']} events. **Sources.** {SRC('final_checks.json')} (`K3`). **Caveats.** Probes fitted on all training-fold patients, evaluated on discovery held-out patients.

### K4. CNV use and rank shift within discovery
**Status.** {'DONE' if isinstance(K4['ablation'], dict) else 'PARTIAL'}. **Pre-specification.** `{PRESPEC}` §K4.
""")
if isinstance(K4["ablation"], dict):
    P(tb(["model", "population", "n / events", "CNV permuted jointly Δ [CI]", "CNV → training mean Δ [CI]", "image bags permuted Δ [CI]", "image → mean tile Δ [CI]", "baseline AUROC"], [[fam, pop, f"{v['cnv_permuted_jointly_mean_of_50']['n']} / {v['cnv_permuted_jointly_mean_of_50']['events']}", f"{s3(v['cnv_permuted_jointly_mean_of_50']['delta_auroc'])} {ci(v['cnv_permuted_jointly_mean_of_50']['ci'])}", f"{s3(v['cnv_replaced_by_training_mean']['delta_auroc'])} {ci(v['cnv_replaced_by_training_mean']['ci'])}", f"{s3(v['image_bags_permuted_mean_of_50']['delta_auroc'])} {ci(v['image_bags_permuted_mean_of_50']['ci'])}", f"{s3(v['image_replaced_by_mean_tile']['delta_auroc'])} {ci(v['image_replaced_by_mean_tile']['ci'])}", f3(v["cnv_permuted_jointly_mean_of_50"]["baseline_auroc"])] for fam, d in K4["ablation"].items() for pop, v in d.items()]))
rs = K4["rank_shift_discovery"]
P(f"""
8a within discovery (n {rs['n']}, events {rs['events']}): Spearman(CNV, late fusion) {f3(rs['spearman_cnv_vs_late'])}; progressors up > 20 points {rs['progressors_up_gt20']}, down {rs['progressors_down_gt20']}; non-progressors up {rs['nonprogressors_up_gt20']}, down {rs['nonprogressors_down_gt20']}; categorical NRI {rs['categorical_NRI'][0]} [{rs['categorical_NRI'][1]}, {rs['categorical_NRI'][2]}].

**Sources.** {SRC('k4_ablation.json', 'pk_gpu.py')}; {SRC('final_checks.json')} (`K4`). **Caveats.** Ablated score per row = mean over 50 permutations; deltas on 82 patients have wide CIs.

### K5. Risk groups within discovery
**Status.** DONE. **Pre-specification.** `{PRESPEC}` §K5.

""" + tb(["endpoint", "arm", "low n / events / rate [Wilson]", "moderate", "high", "OR high vs low [CI]", "Cochran–Armitage z, p"], [[ep, NAME[a], *(f"{v['groups'][g]['n']} / {v['groups'][g]['events']} / {f3(v['groups'][g]['rate'])} {ci(v['groups'][g]['wilson'])}" for g in ["low", "moderate", "high"]), e3(v["OR_high_vs_low_haldane"]), v["cochran_armitage_z_p"]] for ep, d in K5.items() for a, v in d.items()]) + f"""

**Sources.** {SRC('final_checks.json')} (`K5`). **Caveats.** Tertile cut-points from all training-fold patients per fold, then restricted to discovery; tertiles are therefore unequal in size within discovery.

## 4. Figures
""")
def figsec(name, title, panels, table):
    P(f"### {title}\n**File.** `results/paper_final/figs/{name}.png`, `.pdf`; numbers in `results/paper_final/figs/{name}.json`.\n\n{panels}\n\n{table}\n")
rows = FI["rows"]
figsec("F_intro_forest", "F-intro. Forest plot: WSI vs CNV arms by population and endpoint", "Left: AUROC [CI] for WSI, CNV (release RF), CNV (Killcoyne method) for discovery / stratified / pooled × LGD2+ / E-HGD (post hoc). Right: WSI − CNV difference [CI] for both CNV arms; dashed line = pre-specified non-inferiority margin −0.05.", tb(["population", "endpoint", "n / events", "WSI", "CNV (RF)", "CNV (KM)", "WSI − CNV(RF)", "WSI − CNV(KM)"], [[r["population"], r["endpoint"], f"{r['n_events'][0]} / {r['n_events'][1]}", e3(r["image_only"]), e3(r["cnv_only"]), e3(r["cnv_km"]), e3(r["WSI_minus_cnv_only"]), e3(r["WSI_minus_cnv_km"])] for r in rows]))
figsec("F_table_forest", "F-table. Forest plots of all whiteboard arms", f"A: discovery LGD2+ (n {ne['discovery']['LGD2plus'][0]}, events {ne['discovery']['LGD2plus'][1]}). B: discovery E-HGD post hoc (events {ne['discovery']['E_HGD'][1]}). C: pooled (confounded) vs stratified vs discovery, LGD2+. Grey rows = supplementary arms.", tb(["arm", "A discovery LGD2+", "B discovery E-HGD", "C pooled LGD2+", "C stratified LGD2+", "C discovery LGD2+"], [[NAME[a], e3(FT["A_discovery_LGD2plus"][a]), e3(FT["B_discovery_EHGD"][a]), e3(FT["C_pooled_stratified_discovery_LGD2plus"][a]["pooled"]), e3(FT["C_pooled_stratified_discovery_LGD2plus"][a]["stratified"]), e3(FT["C_pooled_stratified_discovery_LGD2plus"][a]["discovery"])] for a in NAME]))
D1 = FJ["F_D1_risk_groups"]
figsec("F_D1_risk_groups", "F-D1. Risk groups", "Left: progression rate per training-fold tertile with Wilson CIs (discovery, LGD2+); counts events/n printed on bars. Right: Mantel–Haenszel OR high vs low across the two strata (all 150), from round 3.", tb(["arm", "low", "moderate", "high", "MH OR two strata [CI]"], [[NAME[a], *(f"{D1['tertile_rates_discovery_LGD2plus'][a]['groups'][g]['events']}/{D1['tertile_rates_discovery_LGD2plus'][a]['groups'][g]['n']} = {f3(D1['tertile_rates_discovery_LGD2plus'][a]['groups'][g]['rate'])} {ci(D1['tertile_rates_discovery_LGD2plus'][a]['groups'][g]['wilson'])}" for g in ["low", "moderate", "high"]), f"{D1['MH_OR_two_strata'][a]['OR']} {ci(D1['MH_OR_two_strata'][a]['ci'])}"] for a in ["C2_grade_maxsofar", "cnv_only", "image_only", "late_mean"]]))
D2 = FJ["F_D2_latent"]
figsec("F_D2_latent", "F-D2. Latent space", f"A: linear-probe AUROC per representation, pooled (item 6) vs discovery-only evaluation (K3). B: UMAP of image-only and intermediate-fusion held-out patient embeddings, five folds (projection fitted on each fold's training patients), coloured by stratum. C (insets): the same projections, discovery patients only, coloured by label. Held-out counts per panel: {D2['B_C_umap']['n_heldout_per_fold']}.", tb(["representation", "pooled probe [CI]", "discovery probe [CI]"], [[f, e3([v["pooled_item6"]] + v["pooled_ci"]), e3([v["discovery"]] + v["discovery_ci"])] for f, v in D2["A_probe_auroc"].items()]))
D3 = FJ["F_D3_attention"]
figsec("F_D3_attention", "F-D3. Attention", f"A: per-slide Spearman correlation of tile attention, image-only vs intermediate and vs co-attention (707 held-out rows each). B: attention mass on the top 5 % of tiles per model, uniform = 0.05. Tile montages (tissue) stay on the cluster: {D3['montage_paths_cluster']}.", tb(["quantity", "model", "n rows", "median [IQR]"], [["Spearman vs image-only", f, v["n_rows"], f"{v['median']} [{v['iqr'][0]}, {v['iqr'][1]}]"] for f, v in D3["A_spearman_rows"].items()] + [["top-5 % mass", f, v["n_rows"], f"{v['median']} [{v['iqr'][0]}, {v['iqr'][1]}]"] for f, v in D3["B_top5pct_mass"].items()]))
D4 = FJ["F_D4_cnv_change"]
figsec("F_D4_cnv_change", "F-D4. Does CNV prediction change?", f"A: CNV-only vs late-fusion patient percentile within discovery (n {D4['A_scatter_discovery']['n']}, events {D4['A_scatter_discovery']['events']}), coloured by label, with the diagonal; Spearman {D4['A_scatter_discovery']['spearman']}. B: ΔAUROC when CNV or image input is destroyed (jointly permuted, mean of 50), per learned fusion model, discovery and pooled (K4).", "Panel B numbers: see the K4 table above (`k4_ablation.json`).")
D5 = FJ["F_D5_false_positives"]
figsec("F_D5_false_positives", "F-D5. False positives", "Later-HGD+ rate for FP vs TN, per stratum, with Wilson CIs and counts on bars, for late fusion, WSI, CNV (RF), C2 (pooled operating point).", tb(["model", "stratum", "FP later HGD+ / FP", "TN later HGD+ / TN", "Fisher p"], [[NAME[a], l, f"{v['laterHGD_FP']} / {v['FP']}", f"{v['laterHGD_TN']} / {v['TN']}", v["fisher_p"]] for a, d in D5.items() for l, v in d.items()]))
D6 = FJ["F_D6_false_negatives"]
figsec("F_D6_false_negatives", "F-D6. False negatives", f"Discovery progressors only; late-fusion FN (n {D6['n_FN']}) vs TP (n {D6['n_TP']}): boxplots of kept tiles, cx, interval first row → endpoint, max grade so far; bar panel of endpoint type.", tb(["variable", "FN median (n)", "TP median (n)", "Mann-Whitney p"], [[c, f"{sorted(v['FN_values'])[len(v['FN_values']) // 2] if v['FN_values'] else 'n/a'} ({len(v['FN_values'])})", f"{sorted(v['TP_values'])[len(v['TP_values']) // 2] if v['TP_values'] else 'n/a'} ({len(v['TP_values'])})", v["mannwhitney_p"]] for c, v in D6["boxes"].items()]) + "\n\nEndpoint type (FN / TP): " + str(D6["endpoint_type"]))
P(f"""## 5. Discrepancies found
1. The pooled late-fusion FP excess of later HGD+ (OR 5.4 in F5/R5) is confined to the validation stratum: discovery FP {K1['per_stratum']['late_mean']['discovery']['laterHGD_FP']}/{K1['per_stratum']['late_mean']['discovery']['FP']} vs TN {K1['per_stratum']['late_mean']['discovery']['laterHGD_TN']}/{K1['per_stratum']['late_mean']['discovery']['TN']}, validation {K1['per_stratum']['late_mean']['validation']['laterHGD_FP']}/{K1['per_stratum']['late_mean']['validation']['FP']} vs {K1['per_stratum']['late_mean']['validation']['laterHGD_TN']}/{K1['per_stratum']['late_mean']['validation']['TN']} (K1).
2. Probe AUROCs fall from pooled to discovery-only: image {f3(K3['image_only']['probe_auroc_pooled_item6'])} → {f3(K3['image_only']['probe_auroc_discovery'])}, early {f3(K3['early_fusion']['probe_auroc_pooled_item6'])} → {f3(K3['early_fusion']['probe_auroc_discovery'])}, intermediate {f3(K3['intermediate_fusion']['probe_auroc_pooled_item6'])} → {f3(K3['intermediate_fusion']['probe_auroc_discovery'])} (K3).
3. Within discovery the FN vs TP tile-count difference (pooled p 0.002) becomes p {K2['late_mean']['comparison']['kept_tiles']['mannwhitney_p']} (K2).

## 6. Not done
- ACE-B (no data). Tissue montages and review pack not committed (patient tissue).

## 7. Pre-specification text as committed at `{PRESPEC}` (verbatim)

""" + "\n".join("> " + l if l.strip() else ">" for l in PRE.splitlines()))
open("docs/paper_final_inputs.md", "w").write("\n".join(L)); print("written")

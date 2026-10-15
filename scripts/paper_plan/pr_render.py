import json, os, sys
R = "results/paper_final"; PRESPEC, RESC, PRETXT = (sys.argv + ["9ac8165", "pending", ""])[1:4]; PRE = open(PRETXT).read() if PRETXT and os.path.exists(PRETXT) else ""; M = json.load(open(f"{R}/rowcount.json")); FI = json.load(open(f"{R}/figs/F_intro_forest_v2.json")); FT = json.load(open(f"{R}/figs/F_table_forest_v2.json"))
def f3(x): return "n/a" if x is None else f"{x:.3f}"
def s3(x): return "n/a" if x is None else f"{x:+.3f}"
def ci(c): return "n/a" if not c or c[0] is None else (f"[{c[0]:+.3f}, {c[1]:+.3f}]" if (c[0] < 0 or c[1] < 0) else f"[{c[0]:.3f}, {c[1]:.3f}]")
def e3(v): return f"{f3(v[0])} {ci(v[1:])}"
def tb(hdr, rows): return "| " + " | ".join(hdr) + " |\n|" + "---|" * len(hdr) + "\n" + "\n".join("| " + " | ".join(str(v) for v in r) + " |" for r in rows)
NAME = {"C2_grade_maxsofar": "Clinical (C2)", "cnv_only": "CNV (release RF)", "cnv_km": "CNV (Killcoyne method)", "image_only": "WSI", "early_fusion": "Early fusion", "intermediate_fusion": "Intermediate fusion", "late_mean": "Late fusion (mean)", "coattention_fusion": "Co-attention (suppl.)", "late_stack_logit": "Late stack (suppl.)", "C2+image": "C2 + WSI", "C2+cnv_only": "C2 + CNV(RF)", "C2+cnv_km": "C2 + CNV(KM)", "C2+late_mean": "C2 + late fusion", "C2+image+cnv_only": "C2 + WSI + CNV(RF)"}
I1, I2, I3 = M["item1_rows_per_patient"], M["item2_aggregation_by_rowcount"], M["item3_mean_aggregation_differences"]; L = []; P = L.append; SRC = f"`results/paper_final/rowcount.json` · `scripts/paper_plan/pr_rowcount.py` · commit {RESC}"
P(f"""# BE paper plan: row count, aggregation and label (report only)

## 1. Header
- Date: 27 September 2026. Commit at start: `6cafa6c`. Pre-specification commit: `{PRESPEC}` (verbatim in §6). Results commit: `{RESC}`; this text is the next commit. Frozen release; nothing retrained; the max-over-rows convention is unchanged (report only).
- Outputs: `results/paper_final/rowcount.json`; figures `results/paper_final/figs/F_intro_forest_v2.{{png,pdf,json}}`, `F_table_forest_v2.{{png,pdf,json}}`; copies of all paper figures in `~/Downloads/be_paper_figs/` with `README.md` (laptop only, not committed).

## 2. Status table
| item | status | key number |
|---|---|---|
| 1 Rows per patient by label | DONE | discovery LGD2+: progressors median {I1['discovery']['LGD2plus']['rows_progressors_median_iqr'][0]} vs non-progressors {I1['discovery']['LGD2plus']['rows_nonprogressors_median_iqr'][0]} rows, MW p {I1['discovery']['LGD2plus']['mannwhitney_p']}; row count alone AUROC {f3(I1['discovery']['LGD2plus']['auroc_rowcount_more_rows_higher'])} {ci(I1['discovery']['LGD2plus']['ci'])} |
| 2 Aggregation by row-count group | DONE | late fusion (1–2 rows, n {I2['row_count_groups']['rows_1_2']['n']}/{I2['row_count_groups']['rows_1_2']['events']}): max {f3(I2['by_arm']['late_mean']['rows_1_2']['max'])}, mean {f3(I2['by_arm']['late_mean']['rows_1_2']['mean'])}; (≥3 rows, n {I2['row_count_groups']['rows_ge3']['n']}/{I2['row_count_groups']['rows_ge3']['events']}): max {f3(I2['by_arm']['late_mean']['rows_ge3']['max'])}, mean {f3(I2['by_arm']['late_mean']['rows_ge3']['mean'])}, last {f3(I2['by_arm']['late_mean']['rows_ge3']['last'])} |
| 3 Mean-aggregation differences | DONE | discovery LGD2+: WSI − CNV(RF) {s3(I3['discovery__LGD2plus']['differences']['image_only_minus_cnv_only']['delta'])} {ci(I3['discovery__LGD2plus']['differences']['image_only_minus_cnv_only']['ci'])}; late − WSI {s3(I3['discovery__LGD2plus']['differences']['late_mean_minus_image_only']['delta'])} {ci(I3['discovery__LGD2plus']['differences']['late_mean_minus_image_only']['ci'])}, selection-adjusted p {I3['discovery__LGD2plus']['differences']['late_mean_minus_image_only']['perm_p_selection_adjusted_max_over_5_fusion_arms']} |
| 4 F-intro v2, F-table v2 | DONE | `results/paper_final/figs/F_intro_forest_v2.png`, `F_table_forest_v2.png` |
| 5 Figure copies | DONE | `~/Downloads/be_paper_figs/` (README lists whiteboard point per file) |

## 3. Items

### 1. Rows per patient by label
**Status.** DONE. **Pre-specification.** `{PRESPEC}`.

""" + tb(["population", "endpoint", "n / events", "rows, progressors median [IQR]", "rows, non-progressors median [IQR]", "Mann-Whitney p", "AUROC of row count alone (more rows = higher) [CI]"], [[pop, ep, f"{v['n']} / {v['events']}", f"{v['rows_progressors_median_iqr'][0]} [{v['rows_progressors_median_iqr'][1]}, {v['rows_progressors_median_iqr'][2]}]", f"{v['rows_nonprogressors_median_iqr'][0]} [{v['rows_nonprogressors_median_iqr'][1]}, {v['rows_nonprogressors_median_iqr'][2]}]", v["mannwhitney_p"], e3([v["auroc_rowcount_more_rows_higher"]] + v["ci"])] for pop, d in I1.items() for ep, v in d.items()]) + f"""

**Sources.** {SRC} (`item1_rows_per_patient`). **Caveats.** Row count is the number of strict pre-event release rows, which depends on surveillance history and on the event time (progressors accrue rows until the endpoint).

### 2. Aggregation rule by row-count group (discovery, LGD2+)
**Status.** DONE. **Pre-specification.** `{PRESPEC}`. Groups: 1–2 rows n {I2['row_count_groups']['rows_1_2']['n']} / events {I2['row_count_groups']['rows_1_2']['events']}; ≥ 3 rows n {I2['row_count_groups']['rows_ge3']['n']} / events {I2['row_count_groups']['rows_ge3']['events']}.

""" + tb(["arm", "all 82: max / mean / last", "1–2 rows: max / mean / last", "≥3 rows: max / mean / last", "Spearman(row count, score) non-progressors: max / mean"], [[NAME[a], " / ".join(f3(v["all_82"][h]) for h in ["max", "mean", "last"]), " / ".join(f3(v["rows_1_2"][h]) for h in ["max", "mean", "last"]), " / ".join(f3(v["rows_ge3"][h]) for h in ["max", "mean", "last"]), f"{s3(I2['spearman_rowcount_vs_score_discovery_nonprogressors'][a]['max'])} / {s3(I2['spearman_rowcount_vs_score_discovery_nonprogressors'][a]['mean'])}"] for a, v in I2["by_arm"].items()]) + f"""

**Sources.** {SRC} (`item2_aggregation_by_rowcount`). **Caveats.** Within-group AUROCs rest on few events; "last" = the row with the latest date.

### 3. Paired differences under mean aggregation
**Status.** DONE. **Pre-specification.** `{PRESPEC}`. Permutations within stratum for the stratified population.

""" + "\n\n".join(f"**{k.replace('__', ', ').replace('LGD2plus', 'LGD2+').replace('E_HGD', 'E-HGD (post hoc)')}** (n {v['n']}, events {v['events']}); mean-aggregated AUROCs: " + ", ".join(f"{NAME[a]} {f3(x)}" for a, x in v["arms_mean_auroc"].items()) + "\n\n" + tb(["difference", "Δ [CI]", "perm p", "selection-adjusted p (fusion − WSI)"], [[d.replace("_minus_", " − "), f"{s3(w['delta'])} {ci(w['ci'])}", w["perm_p"], w.get("perm_p_selection_adjusted_max_over_5_fusion_arms", "")] for d, w in v["differences"].items()]) for k, v in I3.items()) + f"""

**Sources.** {SRC} (`item3_mean_aggregation_differences`). **Caveats.** C2 + modality combinations use the fold-z row combination of the follow-up with mean aggregation; the arms were trained with max-over-rows evaluation in the release.

### 4. Figures v2
**Status.** DONE. `F_intro_forest_v2`: the F-intro panels (max over rows, top row) with the same panels under mean over rows (bottom row). `F_table_forest_v2`: F-table panels A–C (max, top row) and A–C under mean (bottom row). Numbers in the figure JSONs.

""" + tb(["population", "endpoint", "n / events", "WSI max", "WSI mean", "CNV(RF) max", "CNV(RF) mean", "WSI − CNV(RF) max", "WSI − CNV(RF) mean"], [[a["population"], a["endpoint"], f"{a['n_events'][0]} / {a['n_events'][1]}", e3(a["image_only"]), e3(b["image_only"]), e3(a["cnv_only"]), e3(b["cnv_only"]), e3(a["WSI_minus_cnv_only"]), e3(b["WSI_minus_cnv_only"])] for a, b in zip(FI["rows"]["max"], FI["rows"]["mean"])]) + "\n\n" + tb(["arm", "A discovery LGD2+ max", "A mean", "B discovery E-HGD max", "B mean", "C stratified LGD2+ max", "C mean"], [[NAME[a], e3(FT["max"]["A"][a]), e3(FT["mean"]["A"][a]), e3(FT["max"]["B"][a]), e3(FT["mean"]["B"][a]), e3(FT["max"]["C"][a]["stratified"]), e3(FT["mean"]["C"][a]["stratified"])] for a in NAME if a in FT["max"]["A"]]) + f"""

**Sources.** `results/paper_final/figs/F_intro_forest_v2.json`, `F_table_forest_v2.json` · `scripts/paper_plan/pr_rowcount.py` · commit {RESC}.

### 5. Figure copies
**Status.** DONE. `~/Downloads/be_paper_figs/` holds the PNGs of F_intro (v1 and v2), F_table (v1 and v2) and F_D1–F_D6 with `README.md`; no tissue images.

## 4. Discrepancies found
1. Row count differs by label in every population (item 1) and predicts the label by itself; the max-over-rows aggregation used throughout is therefore entangled with row count, and mean aggregation gives higher AUROCs for every arm (item 3; `docs/paper_plan_aggregation.md`). The aggregation rule was fixed in the release before any of this work and is left unchanged here.

## 5. Not done
- No change to the aggregation convention; no retraining.

## 6. Pre-specification text as committed at `{PRESPEC}` (verbatim)

""" + "\n".join("> " + l if l.strip() else ">" for l in PRE.splitlines()))
open("docs/paper_plan_rowcount.md", "w").write("\n".join(L)); print("written")

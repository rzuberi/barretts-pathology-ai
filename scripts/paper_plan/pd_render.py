import json, os, sys
R = "results/paper_plan"; PRESPEC, RESC, PRETXT = (sys.argv + ["30b2317", "pending", ""])[1:4]; PRE = open(PRETXT).read() if PRETXT and os.path.exists(PRETXT) else ""; M = json.load(open(f"{R}/discovery_main.json"))
def f3(x): return "n/a" if x is None else (f"{x:.3f}" if isinstance(x, (int, float)) and not isinstance(x, bool) else str(x))
def s3(x): return "n/a" if x is None else f"{x:+.3f}"
def ci(c): return "n/a" if not c or c[0] is None else (f"[{c[0]:+.3f}, {c[1]:+.3f}]" if (c[0] < 0 or c[1] < 0) else f"[{c[0]:.3f}, {c[1]:.3f}]")
def tb(hdr, rows): return "| " + " | ".join(hdr) + " |\n|" + "---|" * len(hdr) + "\n" + "\n".join("| " + " | ".join(str(v) for v in r) + " |" for r in rows)
NAME = {"C1_grade": "Clinical C1", "C2_grade_maxsofar": "Clinical C2 (primary)", "C3_plus_streak": "Clinical C3", "C4_plus_surveillance_3a": "Clinical C4 (=3a)", "cnv_only": "CNV (release RF)", "cnv_km": "CNV (Killcoyne method)", "image_only": "WSI", "early_fusion": "Early fusion", "intermediate_fusion": "Intermediate fusion", "late_mean": "Late fusion (image + RF CNV)", "late_mean_km": "Late fusion (image + KM CNV)", "coattention_fusion": "Co-attention (extra)", "late_stack_logit": "Late stack (extra)", "C2+image": "C2 + WSI", "C2+cnv_only": "C2 + CNV(RF)", "C2+cnv_km": "C2 + CNV(KM)", "C2+late_mean": "C2 + late fusion", "C2+image+cnv_only": "C2 + WSI + CNV(RF)"}
E = M["endpoints"]; FP = M["fp_analysis_discovery_nonprogressors"]; L = []; P = L.append; SRC = f"`results/paper_plan/discovery_main.json` · `scripts/paper_plan/pd_discovery.py` · commit {RESC}"
P(f"""# BE paper plan: discovery-stratum analysis (82 Killcoyne-discovery patients)

## 1. Header
- Date: 26 September 2026. Commit at start: `c93ca01`. Pre-specification commit: `{PRESPEC}` (verbatim in §5). Results commit: `{RESC}`; this text is the next commit. Frozen release only. Script `scripts/paper_plan/pd_discovery.py`; result `results/paper_plan/discovery_main.json`.
- Cohort: the {M['n_patients']} discovery-stratum patients; LGD2+ {E['LGD2plus']['events']} events; E-HGD {E['E_HGD']['events']} events. Patient scores as computed in earlier rounds (max over rows, fold-honest); nothing refitted except the discovery-refit operating thresholds in the FP analysis.

## 2. Status table
| item | status | key number |
|---|---|---|
| D1 Arms within discovery, LGD2+ | DONE | WSI {f3(E['LGD2plus']['arms']['image_only']['auroc'])} {ci(E['LGD2plus']['arms']['image_only']['auroc_ci'])}; late fusion {f3(E['LGD2plus']['arms']['late_mean']['auroc'])}; CNV(RF) {f3(E['LGD2plus']['arms']['cnv_only']['auroc'])}; C2 {f3(E['LGD2plus']['arms']['C2_grade_maxsofar']['auroc'])} |
| D2 Arms within discovery, E-HGD | DONE | WSI {f3(E['E_HGD']['arms']['image_only']['auroc'])}; late fusion {f3(E['E_HGD']['arms']['late_mean']['auroc'])}; CNV(RF) {f3(E['E_HGD']['arms']['cnv_only']['auroc'])}; CNV(KM) {f3(E['E_HGD']['arms']['cnv_km']['auroc'])} |
| D3 Paired differences and selection-adjusted p | DONE | LGD2+: late − WSI {s3(E['LGD2plus']['differences']['late_mean_minus_image_only']['delta'])} {ci(E['LGD2plus']['differences']['late_mean_minus_image_only']['ci'])}, selection-adjusted p {E['LGD2plus']['differences']['late_mean_minus_image_only']['perm_p_selection_adjusted_max_over_5_fusion_arms']}; WSI − CNV(RF) {s3(E['LGD2plus']['differences']['image_only_minus_cnv_only']['delta'])} {ci(E['LGD2plus']['differences']['image_only_minus_cnv_only']['ci'])} |
| D4 FP analysis within discovery non-progressors | DONE | {FP['n_nonprogressors']} non-progressors, {FP['later_HGDplus_total']} with later HGD+; late fusion (pooled thresholds) FP/TN {FP['models']['late_mean']['pooled_thresholds']['n_FP']}/{FP['models']['late_mean']['pooled_thresholds']['n_TN']}, later HGD+ {FP['models']['late_mean']['pooled_thresholds']['later_HGDplus']['FP']} vs {FP['models']['late_mean']['pooled_thresholds']['later_HGDplus']['TN']}, Fisher p {FP['models']['late_mean']['pooled_thresholds']['later_HGDplus']['fisher_p']} |

## 3. Items
""")
for ep, title in [("LGD2plus", "D1. LGD2+ endpoint within discovery"), ("E_HGD", "D2. E-HGD endpoint within discovery")]:
    e = E[ep]
    P(f"""### {title}
**Question.** How do the arms perform when progressors and non-progressors come from the same design stratum?

**Status.** DONE. **Pre-specification.** `{PRESPEC}`.

n {e['n']}, events {e['events']}.

""" + tb(["arm", "AUROC [CI]", "AUPRC [CI]"], [[NAME.get(a, a), f"{f3(v['auroc'])} {ci(v['auroc_ci'])}", f"{f3(v['auprc'])} {ci(v['auprc_ci'])}"] for a, v in e["arms"].items()]) + "\n\n**Paired differences** (one-sided permutation p that the first arm is higher):\n\n" + tb(["difference", "Δ AUROC [CI]", "perm p", "selection-adjusted p (max over 5 fusion arms, fusion − WSI only)"], [[k.replace("_minus_", " − "), f"{s3(v['delta'])} {ci(v['ci'])}", v["perm_p"], v.get("perm_p_selection_adjusted_max_over_5_fusion_arms", "")] for k, v in e["differences"].items()]) + f"""

**Method.** Patient bootstrap (2,000, seed 0) over the 82; label permutations (2,000, seed 0). **Sources.** {SRC} (`endpoints.{ep}`). **Caveats.** The discovery stratum is a matched case–control design with our LGD2+ label disagreeing with the sheet status for 11 patients; E-HGD has {E['E_HGD']['events']} events.
""")
P("### D3. Paired differences and selection adjustment\n**Status.** DONE. Reported inside D1 and D2 (both endpoints). **Sources.** " + SRC + " (`endpoints.*.differences`).\n")
rows = []
for a, v in FP["models"].items():
    for ver in ["pooled_thresholds", "discovery_refit_thresholds"]:
        w = v[ver]; rows.append([NAME.get(a, a), ver.replace("_", " "), f"{w['n_FP']} / {w['n_TN']}", f"{w['later_HGDplus']['FP']} vs {w['later_HGDplus']['TN']}, Fisher p {w['later_HGDplus']['fisher_p']}", f"{f3(w['later_HGDplus'].get('OR_FP_adjusted'))} {ci(w['later_HGDplus'].get('ci'))}, p {w['later_HGDplus'].get('p')}{' (' + w['later_HGDplus']['note'] + ')' if 'note' in w['later_HGDplus'] else ''}{w['later_HGDplus'].get('logit_error', '')}", f"{w['later_LGDplus']['FP']} vs {w['later_LGDplus']['TN']}, Fisher p {w['later_LGDplus']['fisher_p']}", f"{f3(w['later_LGDplus'].get('OR_FP_adjusted'))} {ci(w['later_LGDplus'].get('ci'))}, p {w['later_LGDplus'].get('p')}{w['later_LGDplus'].get('logit_error', '')}"])
P(f"""### D4. False positives within discovery non-progressors
**Question.** Does the FP excess of later disease hold when only discovery non-progressors are compared?

**Status.** DONE. **Pre-specification.** `{PRESPEC}`. {FP['n_nonprogressors']} discovery non-progressors; later HGD+ in {FP['later_HGDplus_total']}, later LGD+ in {FP['later_LGDplus_total']}.

""" + tb(["model", "thresholds", "FP / TN", "later HGD+ FP vs TN [k/n]", "OR FP adjusted for baseline and max grade [CI], p", "later LGD+ FP vs TN", "OR adjusted"], rows) + f"""

**Method.** F5 model without a stratum term; two threshold versions as pre-specified. **Sources.** {SRC} (`fp_analysis_discovery_nonprogressors`). **Caveats.** {FP['n_nonprogressors']} non-progressors and {FP['later_HGDplus_total']} later-HGD+ events: Wald CIs are wide and several fits are near separation.

## 4. Discrepancies found
1. Compare with the pooled figures in `docs/paper_plan_round3.md` R6: within discovery, WSI {f3(E['LGD2plus']['arms']['image_only']['auroc'])} (pooled 0.731), late fusion {f3(E['LGD2plus']['arms']['late_mean']['auroc'])} (0.774), CNV(RF) {f3(E['LGD2plus']['arms']['cnv_only']['auroc'])} (0.663), C2 {f3(E['LGD2plus']['arms']['C2_grade_maxsofar']['auroc'])} (0.681).

## 5. Pre-specification text as committed at `{PRESPEC}` (verbatim)

""" + "\n".join("> " + l if l.strip() else ">" for l in PRE.splitlines()))
open("docs/paper_plan_discovery.md", "w").write("\n".join(L)); print("written")

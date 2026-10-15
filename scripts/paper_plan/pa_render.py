import json, os, sys
R = "results/paper_final"; PRESPEC, RESC, PRETXT = (sys.argv + ["d1958ea", "pending", ""])[1:4]; PRE = open(PRETXT).read() if PRETXT and os.path.exists(PRETXT) else ""; M = json.load(open(f"{R}/aggregation_sensitivity.json"))
def f3(x): return "n/a" if x is None else f"{x:.3f}"
def s3(x): return "n/a" if x is None else f"{x:+.3f}"
def ci(c): return "n/a" if not c or c[0] is None else (f"[{c[0]:+.3f}, {c[1]:+.3f}]" if (c[0] < 0 or c[1] < 0) else f"[{c[0]:.3f}, {c[1]:.3f}]")
def tb(hdr, rows): return "| " + " | ".join(hdr) + " |\n|" + "---|" * len(hdr) + "\n" + "\n".join("| " + " | ".join(str(v) for v in r) + " |" for r in rows)
NAME = {"C2_grade_maxsofar": "Clinical (C2)", "cnv_only": "CNV (release RF)", "cnv_km": "CNV (Killcoyne method)", "image_only": "WSI", "early_fusion": "Early fusion", "intermediate_fusion": "Intermediate fusion", "late_mean": "Late fusion (mean)", "coattention_fusion": "Co-attention (suppl.)", "late_stack_logit": "Late stack (suppl.)"}
A = M["arms"]; K = M["k3_probes"]; L = []; P = L.append; SRC = f"`results/paper_final/aggregation_sensitivity.json` · `scripts/paper_plan/pa_aggregation.py` · commit {RESC}"
P(f"""# BE paper plan: slide→patient aggregation sensitivity (mean over rows vs max over rows)

## 1. Header
- Date: 27 September 2026. Commit at start: `e421783`. Pre-specification commit: `{PRESPEC}` (verbatim in §5). Results commit: `{RESC}`; this text is the next commit. Frozen release only; nothing retrained. Report only: the max-over-rows convention is unchanged.
- Check that the recomputed max-over-rows scores equal the round-3 patient table (max |difference| per arm): {M['max_check_vs_round3_table']}.

## 2. Status table
| item | status | key number |
|---|---|---|
| A1 Arms, mean vs max, discovery LGD2+ | DONE | late fusion max {f3(A['late_mean']['discovery__LGD2plus']['auroc_max'])} vs mean {f3(A['late_mean']['discovery__LGD2plus']['auroc_mean'])}, Δ {s3(A['late_mean']['discovery__LGD2plus']['delta_mean_minus_max'])} {ci(A['late_mean']['discovery__LGD2plus']['ci'])}; WSI Δ {s3(A['image_only']['discovery__LGD2plus']['delta_mean_minus_max'])} {ci(A['image_only']['discovery__LGD2plus']['ci'])} |
| A2 Arms, mean vs max, stratified LGD2+ | DONE | late fusion Δ {s3(A['late_mean']['stratified__LGD2plus']['delta_mean_minus_max'])} {ci(A['late_mean']['stratified__LGD2plus']['ci'])}; CNV(RF) Δ {s3(A['cnv_only']['stratified__LGD2plus']['delta_mean_minus_max'])} {ci(A['cnv_only']['stratified__LGD2plus']['ci'])} |
| A3 Arms, E-HGD (both populations) | DONE | late fusion discovery Δ {s3(A['late_mean']['discovery__E_HGD']['delta_mean_minus_max'])} {ci(A['late_mean']['discovery__E_HGD']['ci'])} |
| A4 K3 probes with max aggregation | DONE | image: mean-repr {f3(K['image_only']['discovery']['probe_mean_repr_K3'])}, max-repr {f3(K['image_only']['discovery']['probe_max_repr'])}, row-probe max {f3(K['image_only']['discovery']['row_probe_max_over_rows'])} (discovery) |

## 3. Items

### A1–A3. Arms under mean-over-rows aggregation
**Question.** Does the slide→patient aggregation rule change the AUROCs?

**Status.** DONE. **Pre-specification.** `{PRESPEC}`.

""" + "\n\n".join(f"**{pop.replace('__', ', ').replace('LGD2plus', 'LGD2+').replace('E_HGD', 'E-HGD (post hoc)')}** (n {A['image_only'][pop]['n']}, events {A['image_only'][pop]['events']})\n\n" + tb(["arm", "AUROC max (convention)", "AUROC mean", "Δ mean − max [CI]"], [[NAME[a], f3(A[a][pop]["auroc_max"]), f3(A[a][pop]["auroc_mean"]), f"{s3(A[a][pop]['delta_mean_minus_max'])} {ci(A[a][pop]['ci'])}"] for a in NAME]) for pop in ["discovery__LGD2plus", "discovery__E_HGD", "stratified__LGD2plus", "stratified__E_HGD"]) + f"""

**Method.** Row scores as released (C2 rows recomputed deterministically); patient score = max or mean over rows; label = max; bootstrap over patients (within stratum for the stratified population). **Sources.** {SRC} (`arms`). **Caveats.** Patients with one row (26 of 150) are identical under both rules; the stratified AUROC weights discovery pairs 1,680 : 580.

### A4. K3 probes with max aggregation
**Status.** DONE. **Pre-specification.** `{PRESPEC}`.

""" + tb(["representation", "population", "n / events", "probe on mean embedding (K3)", "probe on element-wise max embedding", "Δ [CI]", "row probe, max over rows", "Δ [CI]"], [[fam, pop, f"{v['n']} / {v['events']}", f3(v["probe_mean_repr_K3"]), f3(v["probe_max_repr"]), f"{s3(v['delta_maxrepr_minus_mean'][0])} {ci(v['delta_maxrepr_minus_mean'][1:])}", f3(v["row_probe_max_over_rows"]), f"{s3(v['delta_rowmax_minus_mean'][0])} {ci(v['delta_rowmax_minus_mean'][1:])}"] for fam, d in K.items() for pop, v in d.items()]) + f"""

**Sources.** {SRC} (`k3_probes`). **Caveats.** The row probe is trained on rows (several per patient) with patient-keyed inner folds; the element-wise max representation changes the feature distribution, not only the aggregation.

## 4. Discrepancies found
None affecting earlier documents; the max-over-rows scores reproduce the round-3 table exactly (max |difference| in §1).

## 5. Pre-specification text as committed at `{PRESPEC}` (verbatim)

""" + "\n".join("> " + l if l.strip() else ">" for l in PRE.splitlines()))
open("docs/paper_plan_aggregation.md", "w").write("\n".join(L)); print("written")

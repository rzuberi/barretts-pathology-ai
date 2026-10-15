"""Render reports/erin_imminent_tasks.md from results/erin_progression_fusion/results.json (assembler output)."""
import json, os
T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis" if os.path.exists("/mnt/scratche") else os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
R = json.load(open(T + "/results/erin_progression_fusion/results.json")); C = json.load(open(T + "/feasibility/erin_fusion/tasks/task_counts.json")) if os.path.exists(T + "/feasibility/erin_fusion/tasks/task_counts.json") else {}
DESC = {"T1": "first NDBE/IND report → HGD/cancer ≤ 1 y (FEASIBILITY ONLY)", "T2a": "benign report (landmark) → LGD+ ≤ 1 y — PRIMARY", "T2b": "benign report → HGD+ ≤ 1 y", "T2c": "benign report → next report LGD+",
        "T3a": "benign-section slide → case has LGD+ elsewhere (field effect)", "T3b": "benign-section slide → case has HGD+ elsewhere", "T4": "benign report → prior LGD+ in history"}
ARM = {"a": "baseline (grade+age)", "b": "text (nomic embedding)", "c": "image (ABMIL, UNI2)", "d": "late-mean fusion b+c"}
L = ["# ERIN imminent-dysplasia task set — results (assembled %s)\n" % __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M"),
     "Pre-registration: `docs/erin_imminent_tasks_preregistration.md` (committed before the first job). Machinery: `scripts/abmil_clf.py` trainer, `patient_folds` (seed 0), 3 seeds, patient-clustered 2,000-replicate bootstraps, fold-local z-scored late fusion. No hyper-parameter tuning. Numbers only; the pre-registered rules say what they mean.\n",
     "## Why these tasks\nThe 3-year progression experiment was infeasible (all scanned ERIN slides are 2022–2025; 0 negatives at 3 years — `reports/erin_progression_fusion.md`). These are the tasks the same window CAN support: imminent dysplasia within a year, next-report upgrade, synchronous dysplasia elsewhere in the case (field effect), and prior dysplasia in the patient's history.\n",
     "## Cohort counts\n| task | definition | n | pos | neg | patients | slides | status |\n|---|---|---|---|---|---|---|---|"]
for t, c in C.items(): L.append(f"| {t} | {DESC.get(t, '')} | {c['n']} | {c['pos']} | {c['neg']} | {c['patients']} | {c['slides']} | {'feasibility only' if c['feasibility_only'] else 'interpretable'} |")
L.append("\n## Per-arm metrics (patient-level where one sample per patient; landmark tasks have several samples per patient — bootstraps resample patients)\n| task | arm | AUROC [95% CI] | AUPRC [95% CI] | Brier | spec@sens0.95 | spec@sens1.0 | sens@spec0.80 | image units |\n|---|---|---|---|---|---|---|---|---|")
for t, r in R["tasks"].items():
    for a, v in r["arms"].items(): L.append(f"| {t} | {ARM.get(a, a)} | {v['auroc']} {v['auroc_ci']} | {v['auprc']} {v['auprc_ci']} | {v['brier']} | {v['spec_at_sens_0.95']} | {v['spec_at_sens_1.0']} | {v['sens_at_spec_0.80']} | {r['img_units_done']} |")
L.append("\n## Paired deltas (AUROC, patient-clustered 95% CI)\n| task | contrast | delta | CI | rule |\n|---|---|---|---|---|")
for t, r in R["tasks"].items():
    for k, v in r["deltas"].items():
        lo, hi = v["ci"]; verdict = "CI excludes 0 (positive)" if (lo is not None and lo > 0) else ("CI excludes 0 (negative)" if (hi is not None and hi < 0) else "CI includes 0")
        if t == "T1": verdict = "feasibility only — not interpreted"
        L.append(f"| {t} | {k} | {v['mean']} | [{lo}, {hi}] | {verdict} |")
L.append("\n## Permutation null (T2a)")
for k in ("perm_null_c", "perm_null_d"):
    v = R["tasks"].get("T2a", {}).get(k)
    if v: L.append(f"- {k}: {v['n']} permutations, null mean {v['mean']}, 95th pct {v['p95']}, empirical p = {v['p_empirical']}")
    else: L.append(f"- {k}: not available (permutation shards incomplete)")
L.append("\n## Tile-count confound (AUROC of tile count alone for the image task label)")
for t, r in R["tasks"].items():
    if "tilecount_confound_auroc" in r: L.append(f"- {t}: {r['tilecount_confound_auroc']}")
L.append("\n## What the pre-registered rules say\n")
t2 = R["tasks"].get("T2a", {}); d = t2.get("deltas", {}); fus = next((v for k, v in d.items() if k.startswith("late_mean")), None)
if fus:
    L.append(f"- **T2a primary (fusion vs best single modality):** delta {fus['mean']} [{fus['ci'][0]}, {fus['ci'][1]}] → " + ("CI excludes 0 — the selection-adjusted permutation over {b, c, d} is REQUIRED before any statement (not yet run)." if fus["ci"][0] is not None and fus["ci"][0] > 0 else "no demonstrated fusion gain (rule: CI must exclude 0)."))
for t, r in R["tasks"].items():
    ca = r["deltas"].get("image - baseline")
    if ca and t != "T1": L.append(f"- {t} image − baseline: {ca['mean']} [{ca['ci'][0]}, {ca['ci'][1]}] → " + ("the image carries information beyond grade + age" if ca["ci"][0] is not None and ca["ci"][0] > 0 else "not demonstrated") + f" ({DESC[t]}).")
L.append("- T1 is a feasibility run (18 positives): reported, not interpreted.\n- All tasks live in the 2022–2025 imaging window with ≤ 3 years follow-up; none is 'progression prediction' in the SWG sense.")
L.append("\n## Not finished / caveats\n- Image units per task are listed above; any task below 15/15 has an incomplete image arm.\n- The text arm depends on the `nomic-embed-text` embedding task; if absent, arms b and d are missing.\n- T2 landmark samples share patients (several benign reports per patient); folds are patient-disjoint but the effective n is the patient count.\n")
L.append("## Decisions not pre-specified (to defend or change)\n1. Slides capped at 1,500 tiles each when pooling a case bag (memory); ABMIL's own MAX_TILES=800 per-epoch subsample then applies.\n2. Landmark tasks allow multiple reports per patient; a one-per-patient variant was not run.\n3. Negative definition for the 1-year label: no LGD+/HGD+ event AND last report ≥ 365 days after the landmark.\n4. Text arm embeds FinalDiagnosis + MicroscopicDescription only (no addenda), truncated at 6,000 characters.\n5. Late fusion uses fold-local z-scoring (per 2.46), not the SWG release's pooled z-scoring.\n6. Baseline for T3 uses section grade (NORMAL_OTHER vs NDBE) + age; for others index grade (NDBE vs IND) + age.\n")
cp = R.get("compute", {}); L.append(f"## Compute\nGPU-hours by partition for the worker jobs (allocated GPUs × wall time, idle included): {cp.get('gpu_hours_by_partition')}. Task→job log in `results/erin_progression_fusion/results.json` (`task_partition_log`).\n")
L.append("## Figures\n`results/erin_progression_fusion/figures/`: fig1_roc_T2a.png (+ roc_T2a_*.csv), fig2_forest_deltas.png (+ .csv), fig3_reliability_T2a_d.png (+ .csv).")
os.makedirs(T + "/reports", exist_ok=True); open(T + "/reports/erin_imminent_tasks.md", "w").write("\n".join(L) + "\n"); print("wrote reports/erin_imminent_tasks.md")

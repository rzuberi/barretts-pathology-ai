"""Exploration E3: markdown tables for docs/paper_explore_E3.md from results/paper_final/explore_E3/*.json (prints to stdout)."""
import json, glob, os, re
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); RES = ROOT + "/results/paper_final/explore_E3"
NAMES = {"stored": "current L-IMG (stored)", "base": "current L-IMG, refitted (check)", "base_tpall": "tile-score mean (check)"}
def nm(v):
    if v in NAMES: return NAMES[v]
    for a, b in (("cat_u2l12", "concat UNI2-h L1+L2"), ("cat_gp123", "concat GigaPath L1+L2+L3"), ("cat_u2gp2", "concat UNI2-h+GigaPath L2")): v = v.replace(a, b)
    v = v.replace("base_tp", "tile-score pooling of the current model: ").replace("base", "UNI2-h L2"); v = re.sub(r"_mean(?=$|\+)", "", v); v = v.replace("u2l2_", "UNI2-h L2 pooled ").replace("u2l1", "UNI2-h L1")
    for a, b in (("gpl", "GigaPath L"), ("v2l2", "Virchow2 L2"), ("ho20", "H-optimus-0 20x"), ("v220", "Virchow2 20x"), ("co20", "CONCH 20x"), ("ct20", "CTransPath 20x"), ("mu20", "MUSK 20x"),
                 ("cat_u2l12", "concat UNI2-h L1+L2"), ("cat_gp123", "concat GigaPath L1+L2+L3"), ("cat_u2gp2", "concat UNI2-h+GigaPath L2"), ("mlogit:", "[mean logit] "), ("medlogit:", "[median logit] "),
                 ("mprob:", "[mean prob.] "), ("top2logit:", "[top-2 logit] "), ("alt:", "[pooled-embedding endoscopy] "), ("ens:", "ensemble "), ("sel:", "nested choice among "), ("+", " + ")):
        v = v.replace(a, b)
    return v
f2 = lambda d: f"{d['delta']:+.1f} [{d['ci95'][0]:+.1f}, {d['ci95'][1]:+.1f}]"
f3 = lambda d: f"{d['delta']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]"
R = [json.load(open(f)) for f in glob.glob(RES + "/v_*.json")]; R = [r for r in R if r["variant"] not in ("base", "base_tpall")]
R.sort(key=lambda r: (r["contrasts"]["P1_90_new_vs_P1_90"]["delta_new_minus_ref"]["non_progressor_patients_flagged"]["delta"]))
print("| Variant | AUROC sample / endoscopy / patient | Δ endoscopy AUROC [95% CI] | P1 90%: caught / NP flagged | Δ NP vs P1 (current) | P1 matched to R49 sens.: Δ NP vs R49 | Δ spec. vs R49 (matched) | R49 rebuilt: caught / NP / seq. per patient | Δ NP vs R49 | Δ caught vs R49 | Apparent P1 NP (optimistic) |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for r in R:
    a = r["auroc"]; c = r["contrasts"]; p1 = r["cells"]["P1 90%"]; rr = r["cells"]["R49"]; g = lambda k, m: c[k]["delta_new_minus_ref"][m]
    print(f"| {nm(r['variant'])} | {a['sample']['new']:.3f} / {a['endoscopy']['new']:.3f} / {a['patient']['new']:.3f} | {f3(a['endoscopy']['delta'])} | {p1['progressor_patients_caught']:.1f} / {p1['non_progressor_patients_flagged']:.1f} | "
          f"{f2(g('P1_90_new_vs_P1_90', 'non_progressor_patients_flagged'))} | {f2(g('P1_new_matched_to_R49', 'non_progressor_patients_flagged'))} | {f3(g('P1_new_matched_to_R49', 'endoscopy_specificity'))} | "
          f"{rr['progressor_patients_caught']:.1f} / {rr['non_progressor_patients_flagged']:.1f} / {rr['samples_sequenced_per_patient']:.2f} | {f2(g('R49_new_vs_R49', 'non_progressor_patients_flagged'))} | {f2(g('R49_new_vs_R49', 'progressor_patients_caught'))} | "
          f"{r['apparent_optimistic_P1_at_90_eval_endoscopy_sensitivity']['new']['non_progressor_patients_flagged']:.1f} |")
print("\n\n| Variant | Memory rule | Caught | NP flagged | Seq. / patient | Δ NP vs R49 | Δ caught vs R49 | Δ NP vs same rule, current L-IMG | Δ caught vs same rule | Δ endoscopy spec. vs R49 |")
print("|---|---|---|---|---|---|---|---|---|---|")
FN = {"A1:img:none": "H&E alone, no memory", "A1:img:ew": "H&E alone, EW memory", "A1:gate_al:none": "gate score, no memory", "A1:gate_al:ew": "gate score (R49-aligned), EW memory"}
M = [json.load(open(f)) for f in glob.glob(RES + "/mem_*.json")]
rows = [(r, m) for r in M for m in r["methods"]]; rows.sort(key=lambda x: (x[1]["family"], x[1]["vs_R49"]["np_flagged"]["delta"]))
for r, m in rows:
    print(f"| {nm(r['variant'])} | {FN[m['family']]} | {m['values']['caught']:.1f} | {m['values']['np_flagged']:.1f} | {m['values']['seq']:.2f} | {f2(m['vs_R49']['np_flagged'])} | {f2(m['vs_R49']['caught'])} | "
          f"{f2(m['vs_stored_same_rule']['np_flagged'])} | {f2(m['vs_stored_same_rule']['caught'])} | {f3(m['vs_R49']['endo_spec'])} |")

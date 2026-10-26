"""Two papers, agent P2N (docs/two_papers_P2N.md): supplementary variant log of the Paper 2 rule search, built only from the committed aggregate JSONs
(results/paper_final/review_round8, review_round9, explore_E1..E9). Counts every held-out (nested) rule cell that was scored, by stage and family, and the
best cell of each family under the selection criterion (fewest non-progressor patients flagged with progressor patients caught >= R49's 31.1 - 1.0).
Optimistic cells (thresholds set on the evaluation patients: 'matched', 'apparent', 'APPARENT') are counted separately and never ranked. Runs locally (aggregates only).
-> results/paper_final/two_papers_P2N/variant_log.json"""
import json, glob, os, re
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "paper_final"); OUT = R + "/two_papers_P2N"; os.makedirs(OUT, exist_ok=True)
C49, N49, TOL = 31.1, 31.8, 1.0
J = lambda p: json.load(open(f"{R}/{p}"))
ROWS = []
def fam(stage, family, varied, cells, n_opt=0, note="", sel=None):
    """cells: list of (label, caught, np_flagged, seq)"""
    ok = [c for c in cells if c[1] is not None and c[2] is not None and c[1] >= C49 - TOL - 1e-9 and not re.search(r"LEAKS|not deployable", str(c[0]))]
    best = min(ok, key=lambda c: (c[2], -c[1])) if ok else None   # label-leaking / non-deployable demonstrations are counted, never ranked
    ROWS.append({"stage": stage, "family": family, "varied": varied, "nested_cells": len(cells), "optimistic_cells_not_ranked": n_opt, "best_within_1_progressor": None if best is None else
                 {"label": best[0], "caught": best[1], "np_flagged": best[2], "seq_per_patient": best[3]}, "note": note})
def gv(d, *keys):
    for k in keys:
        if isinstance(d, dict) and k in d and d[k] is not None: return d[k]
    return None
# ---------------------------------------------------------------- before the false-flag search (R49 fixed before these were seen)
P8 = []
for f in sorted(glob.glob(f"{R}/review_round8/pol_*.json")):
    for c in json.load(open(f))["L1"]["cells"]: P8.append((f"{c.get('family')} {c.get('label')}", gv(c, "progressor_patients_caught"), gv(c, "non_progressor_patients_ever_flagged"), gv(c, "samples_sequenced_per_patient")))
fam("0 Round 8 strategy map (descriptive)", "Policies P1-P10 (H&E alone, CN alone, sequence all, gates, first-look, interval sequencing, grade/p53)", "policy x thresholds", P8,
    note="R49 (round 7, pre-specified) fixed as the main rule before the map was read; round 7b's 36 gate designs (apparent, not nested) are not counted")
# ---------------------------------------------------------------- round 9
m9 = J("review_round9/merge.json")["merge"]["methods"]; G9 = {}
for m in m9:
    if m["kind"] == "reference": continue
    key = m["family"].split(":")[0] if not m["family"].startswith("A1") else "A1:" + m["family"].split(":")[1] if ":" in m["family"] else m["family"]
    G9.setdefault(key, []).append(m)
DESC = {"A1:img": "H&E alone (max L-IMG): memory functions", "A1:gate_lit": "Gate score (literal): memory functions", "A1:gate_al": "Gate score (R49-aligned): memory functions incl. longer half-lives",
        "A1:b1_al": "Stacked B1 gate score: memory functions", "G": "Sample aggregation (mean, top-2, max L-LATE) x memory", "A2": "Certificates (copy-number A2, p53 A2')", "A3": "Agreement within endoscopy",
        "A4": "Confirm-by-sequencing (high band also sequenced) +- memory", "B1": "Stacked gate (grade, age, sex, p53)", "B3": "Abstention in the middle band", "C1": "Control: R49 cut-offs re-tuned",
        "A12": "Memory + copy-number certificate (A1+A2)", "A12p": "Memory + p53 certificate", "B2A2": "Stacked gate + memory + certificate", "B4a": "Refit: non-progressor weight 2", "B4b": "Refit: NDBE samples only"}
for key, L in sorted(G9.items()):
    cells = [(m["name"], m["values"]["caught"], m["values"]["np_flagged"], m["values"]["seq"]) for m in L if m["kind"] in ("nested", "nested_pauc", "grid_point")]
    fam("1 Round 9 (false-flag methods)", DESC.get(key, key), "memory function / half-life / certificate / cut-off cells", cells, sum(m["kind"] == "matched" for m in L))
# ---------------------------------------------------------------- E1
E1 = []; nopt = 0
for f in ("fix_veto", "fix_filter", "fix_nsamp", "fix_score"):
    d = J(f"explore_E1/{f}.json")["E1_3"]
    for x in d["fixes"]:
        v = x["nested"]["value"] if isinstance(x.get("nested"), dict) and isinstance(x["nested"].get("value"), dict) else (x.get("nested") or {})
        E1.append((f"{x.get('family')} {x.get('rule')} {x.get('feature')} eps {x.get('eps')} {v.get('label', '') if isinstance(v, dict) else ''} {x['nested'].get('label', '') if isinstance(x.get('nested'), dict) else ''}", gv(v, "caught", "progressor_patients_caught"), gv(v, "np_flagged", "non_progressor_patients_flagged", "non_progressor_patients_ever_flagged"), gv(v, "seq", "samples_sequenced_per_patient")))
        nopt += int(x.get("apparent_optimistic") is not None)
fam("2 E1 (error analysis)", "Tile-score substitutions, single-feature vetoes, second-stage filters, sample-count penalty (incl. non-deployable batch/metadata controls)", "feature x eps x base (R49, P1)", E1, nopt)
# ---------------------------------------------------------------- E2
def e2cells(f): return [(c["label"], c["progressor_patients_caught"], c["non_progressor_patients_ever_flagged"], c["samples_sequenced_per_patient"]) for c in J(f"explore_E2/{f}.json")["cells"]]
fam("3 E2 (copy-number veto)", "Fixed copy-number / trained-score vetoes on P1, R49 high band, R49 all flags", "25 quietness features + L-CNV + L-LATE x k 1/2 x 4 threshold rules x 3 bases", sum((e2cells(f"fixed_{b}") for b in ("P1", "R49h", "R49hm")), []),
    sum(1 for b in ("P1", "R49h", "R49hm") for c in J(f"explore_E2/fixed_{b}.json")["cells"] if c.get("apparent")))
fam("3 E2 (copy-number veto)", "Elastic-net vetoes trained on H&E-flagged samples", "block x population x k x rule x base", e2cells("en_eval"))
fam("3 E2 (copy-number veto)", "Double vetoes (trained score low AND quiet)", "score x quietness feature x k x bar x rule x base", e2cells("combo"))
fam("3 E2 (copy-number veto)", "Vetoes allowed only at NDBE / NDBE-ID endoscopies", "gate x feature x rule x base", e2cells("grade"))
fam("3 E2 (copy-number veto)", "H&E AND copy number (all samples sequenced)", "H&E x CN thresholds", e2cells("and"))
fam("3 E2 (copy-number veto)", "Batch- and noise-adjusted vetoes", "feature x adjustment x rule x base", e2cells("drift"))
fam("3 E2 (copy-number veto)", "Within-patient stability vetoes", "score x rule x base", e2cells("stab"))
fam("3 E2 (copy-number veto)", "Veto chosen per outer fold on training patients", "candidate family x slack x base", sum((e2cells(f"sel_{b}") for b in ("P1", "R49h", "R49hm")), []))
fam("3 E2 (copy-number veto)", "Vetoes on round-9 memory bases (A1, A1 longer half-lives, A12): the final rule's origin", "17 vetoes x 3 memory bases (+ the bases)", e2cells("mem"),
    note="the quiet-genome check (ps_oac, top-2 samples, cn90 bar) was picked here after seeing the R49-base cells")
# ---------------------------------------------------------------- E3
E3v = []; nopt = 0
for f in sorted(glob.glob(f"{R}/explore_E3/v_*.json")):
    d = json.load(open(f)); nopt += 1
    for k, c in d["cells"].items(): E3v.append((f"{d['variant']} {k}", c["progressor_patients_caught"], c["non_progressor_patients_flagged"], c["samples_sequenced_per_patient"]))
fam("4 E3 (H&E model / aggregation)", "Extractors, scales, poolings, ensembles, endoscopy aggregations under P1 80/90/95% and R49 rebuilt", "57 score variants x 4 rules", E3v, nopt)
E3m = []
for f in sorted(glob.glob(f"{R}/explore_E3/mem_*.json")):
    d = json.load(open(f))
    for m in d["methods"]: E3m.append((f"{d['variant']} {m['family']}", m["values"]["caught"], m["values"]["np_flagged"], m["values"]["seq"]))
fam("4 E3 (H&E model / aggregation)", "Score variants inside the round-9 memory engine (mean-logit + memory found here)", "14 score variants x 4 families", E3m)
# ---------------------------------------------------------------- E4
E4 = []
for f in sorted(glob.glob(f"{R}/explore_E4/eval_*.json")):
    d = json.load(open(f))
    for k, c in d["rules"].items(): r = c["rule"]; E4.append((f"{d['scheme']} {k}", r["progressor_patients_caught"], r["non_progressor_patients_flagged"], r["samples_sequenced_per_patient"]))
fam("5 E4 (time-aware models)", "Cox / horizon / certainty-weighted refits under P1, P3 and four R49 cells", "20 score schemes x 8 rules", E4)
E4m = []
for f in sorted(glob.glob(f"{R}/explore_E4/mem_*.json")):
    d = json.load(open(f))
    for k, c in d["families"].items():
        v = c.get("values") or c.get("rule") or c; E4m.append((f"{d['scheme']} {k}", gv(v, "caught", "progressor_patients_caught"), gv(v, "np_flagged", "non_progressor_patients_flagged"), gv(v, "seq", "samples_sequenced_per_patient")))
fam("5 E4 (time-aware models)", "Time-aware scores inside the memory engine", "6 schemes x 3 families", E4m)
# ---------------------------------------------------------------- E5
E5 = []; nopt = 0
for f, key in (("e5_a", "A"), ("e5_b", "B"), ("e5_c", "C"), ("e5_mem", "MEM")):
    for k, c in J(f"explore_E5/{f}.json")[key].items():
        if "APPARENT" in k.upper(): nopt += 1; continue
        o = c.get("observed", c); E5.append((k, gv(o, "progressor_patients_caught", "caught"), gv(o, "non_progressor_patients_flagged", "np_flagged"), gv(o, "samples_sequenced_per_patient", "seq")))
fam("6 E5 (ERIN dysplasia model)", "ERIN score alone, stacked with L-IMG, as veto / rescue inside R49, inside memory", "score x use x quantile", E5, nopt)
# ---------------------------------------------------------------- E7 (stress tests; only the check-bar cells are rule candidates)
e7 = J("explore_E7/base.json"); b5 = e7["test5_bar_sweep"]
fam("7 E7 (artefact tests)", "Quiet-genome check bar sweep 70-97.5% and in-fold bar on A12", "bar", [(k, v["values"]["caught"], v["values"]["np_flagged"], v["values"]["seq_per_patient"]) for k, v in b5.items()],
    note="the other E7 tests (exposure, order, prevalent disease, one look, missing records, batch, permutation) are robustness checks, not candidates")
# ---------------------------------------------------------------- E8
E8 = []; nopt = 0
for f in ("arms", "qc", "noise", "conc", "flat", "soft"):
    d = J(f"explore_E8/{f}.json"); cells = d[f]["cells"] if f != "soft" else d["cells"]
    for c in cells: E8.append((c["label"], c["progressor_patients_caught"], c["non_progressor_patients_ever_flagged"], c["samples_sequenced_per_patient"])); nopt += int(c.get("apparent") is not None)
fam("8 E8 (check biology and variants)", "Check variants on A12: arm subsets, QC gates, noise scaling, two-sample agreement, flattening, soft veto (incl. GC-rich / fragile-site controls)", "score / bar / step", E8, nopt,
    note="1,000 random arm sets and random sample pairs (controls, not candidates) are not counted")
# ---------------------------------------------------------------- E9
OP = J("explore_E9/frontier.json")["operating_points_target_D"]; NEW9 = [k for k in OP if k in ("ml_memcert", "ml_final", "ml_memcheck", "ml_seq0", "ml_seq", "st_seq0", "st_seq", "memcheck", "ml_gate", "ml_gatemem")]
fam("9 E9 (curves and tiers)", "Mean-logit full stack, check-in-the-loop, tier combinations", "combination", [(OP[k]["label"], OP[k]["caught"], OP[k]["np_flagged"], OP[k]["seq_per_patient"]) for k in NEW9],
    note="operating curves sweep the training detection target of existing families; the swept points are not counted as variants")
tot = sum(r["nested_cells"] for r in ROWS); tot9 = sum(r["nested_cells"] for r in ROWS if not r["stage"].startswith("0"))
json.dump({"rows": ROWS, "total_nested_cells": tot, "total_nested_cells_from_round9_on": tot9, "total_optimistic_cells_not_ranked": sum(r["optimistic_cells_not_ranked"] for r in ROWS),
           "criterion": f"best = fewest NP patients flagged among cells with caught >= {C49} - {TOL}; R49 reference {C49} / {N49}",
           "note": "counts are held-out (nested) rule cells as stored in the committed aggregates; a 'family' row may contain per-fold selected rows and fixed grid points of the same candidates"},
          open(f"{OUT}/variant_log.json", "w"), indent=1)
for r in ROWS: print(f"{r['stage'][:34]:34s} | {r['family'][:70]:70s} | {r['nested_cells']:5d} | {r['best_within_1_progressor'] and (r['best_within_1_progressor']['caught'], r['best_within_1_progressor']['np_flagged'])}")
print("total", tot, "from round 9 on", tot9)

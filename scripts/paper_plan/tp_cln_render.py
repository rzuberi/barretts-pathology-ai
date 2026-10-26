"""Two papers, agent CLN: markdown tables for docs/two_papers_CLN.md from results/paper_final/two_papers_CLN/ (local, aggregates only). Prints to stdout."""
import json, os, sys
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../results/paper_final/two_papers_CLN")
J = lambda p: json.load(open(os.path.join(R, p)))
C = J("compare.json"); MOD = ("stored", "clean", "rand"); D = {m: J(f"{m}/design.json")["design"] for m in MOD}; P = {m: J(f"{m}/per1000.json")["per1000"] for m in MOD}; L = {m: J(f"{m}/landmark.json")["landmark"] for m in MOD}
NM = {"R49": "Per-endoscopy gate R49", "P1": "H&E alone (P1 90%)", "A1": "Memory (gate score)", "A12": "Memory + certificate (light, R49 gate)", "final": "Confirmatory: final rule, cn90 bar",
      "final_infold": "Confirmatory: final rule, fold-internal bar", "ml_mem": "Tier 0: mean-logit + memory", "ml_A12": "Mean-logit gate + memory + certificate (light)", "ml_final": "Mean-logit full stack, cn90 bar",
      "ml_final_infold": "Mean-logit full stack, fold-internal bar", "ml_single": "Mean-logit, single endoscopy", "gate_single": "Gate score, single endoscopy", "P3": "Sequence everything (P3 90%)", "g53": "Grade IND/LGD or p53 aberrant"}
ORDER = ["R49", "P1", "A1", "A12", "final", "final_infold", "ml_mem", "ml_A12", "ml_final", "ml_final_infold"]
def f1(x): return f"{x:.1f}" if isinstance(x, (int, float)) else str(x)
def ci(v, nd=1): return f"[{v['ci95'][0]:.{nd}f}, {v['ci95'][1]:.{nd}f}]"
def dl(v, nd=1, bold=True):
    s = f"{v['delta']:+.{nd}f} {ci(v, nd)}".replace("-", "−"); lo, hi = v["ci95"]
    return f"**{s}**" if bold and lo is not None and hi is not None and (lo > 0 or hi < 0) else s
def cell(b): return f"{b['caught']['value']:.1f} / {b['np_flagged']['value']:.1f} / {b['seq_per_patient']:.2f}"
out = []
out.append("### 1. Every rule held out, three L-IMG versions\n")
out.append("Caught of 32 / non-progressor patients flagged of 43 / samples sequenced per patient (repeat mean); Δ = paired patient-bootstrap difference in non-progressors flagged [95% CI] (caught Δ in the next column); bold = CI excludes 0.\n")
out.append("| Rule | Published L-IMG | Clean-tile L-IMG | Random-removal control | Clean − published, NP | Clean − published, caught | Clean − random, NP | Clean: rule − R49, NP | Clean: rule − R49, caught |")
out.append("|---|---|---|---|---|---|---|---|---|")
for k in ORDER:
    r = C["raw"][k]; vs = r.get("clean_vs_R49_same_model")
    out.append(f"| {NM[k]} | {cell(r['stored'])} | {cell(r['clean'])} | {cell(r['rand'])} | {dl(r['clean_minus_stored']['np_flagged'])} | {dl(r['clean_minus_stored']['caught'])} | {dl(r['clean_minus_rand']['np_flagged'])} | "
               f"{dl(vs['np_flagged']) if vs else '—'} | {dl(vs['caught']) if vs else '—'} |")
out.append("\nCIs of the clean-model counts: " + "; ".join(f"{NM[k]} caught {ci(C['raw'][k]['clean']['caught'])}, NP {ci(C['raw'][k]['clean']['np_flagged'])}" for k in ("R49", "ml_mem", "ml_A12", "final", "ml_final")) + ".\n")
out.append("Saving against the same model's R49 (NP flagged by the rule minus by R49), clean minus published / clean minus random: " + "; ".join(f"{NM[k]} {dl(C['raw'][k]['saving_vs_R49_clean_minus_stored'])} / {dl(C['raw'][k]['saving_vs_R49_clean_minus_rand'])}" for k in ("ml_mem", "A1", "A12", "final", "ml_A12", "ml_final")) + ".\n")
if "mean_logit_minus_gate_stack" in C:
    out.append("\n**Mean-logit stack minus R49-gate stack, same model** (NP flagged Δ; caught Δ; samples per patient Δ):\n")
    out.append("| Contrast | Model | Raw: NP | Raw: caught | Raw: samples / patient | Equal exposure: NP |"); out.append("|---|---|---|---|---|---|")
    for c_, v in C["mean_logit_minus_gate_stack"].items():
        for m in MOD: out.append(f"| {c_} | {m} | {dl(v[m]['raw']['np_flagged'])} | {dl(v[m]['raw']['caught'])} | {dl(v[m]['raw']['seq_per_patient'], 2)} | {dl(v[m]['equal_exposure']['np_flagged'])} |")
out.append("\n### 2a. Equal exposure (E7 truncation, 200 draws; frozen per-fold rules)\n")
out.append("| Rule | NP flagged: published | clean | random | Rule − R49, published | Rule − R49, clean | Clean − published | Clean − random | Caught, clean |"); out.append("|---|---|---|---|---|---|---|---|---|")
E = C["equal_exposure"]
for k in ORDER:
    e = E[k]; out.append(f"| {NM[k]} | {e['stored']['np_flagged']['value']:.1f} | {e['clean']['np_flagged']['value']:.1f} {ci(e['clean']['np_flagged'])} | {e['rand']['np_flagged']['value']:.1f} | "
                         f"{dl(e['stored_vs_R49_same_model']['np_flagged']) if k != 'R49' else '—'} | {dl(e['clean_vs_R49_same_model']['np_flagged']) if k != 'R49' else '—'} | {dl(e['clean_minus_stored']['np_flagged'])} | {dl(e['clean_minus_rand']['np_flagged'])} | {e['clean']['caught']['value']:.1f} |")
out.append("\nMemory gain at equal exposure (memory minus the same base without memory, NP flagged): " + "; ".join(f"{k.split(' (')[0]}: published {dl(v['stored']['np_flagged'])}, clean {dl(v['clean']['np_flagged'])}, random {dl(v['rand']['np_flagged'])}" for k, v in E.items() if "memory gain" in k) + ".\n")
out.append("\n### 2b. Landmark AUROC gain of memory (history to date; margin AUROC, memory minus single endoscopy)\n")
out.append("| Pair | Horizon | Published, all landmarks | Clean, all landmarks | Clean, equal exposure | Clean, patient-weighted | Random, all |"); out.append("|---|---|---|---|---|---|---|")
for pr in ("ml_mem - ml_single", "A1 - gate_single", "A12 - gate_single"):
    for h in ("3y", "5y", "ever"):
        g = lambda m, blk: L[m][blk][h]["auroc_memory_minus_single"][pr]
        out.append(f"| {pr} | {h} | {dl(g('stored', 'all_landmarks'), 3)} | {dl(g('clean', 'all_landmarks'), 3)} | {dl(g('clean', 'equal_exposure_landmarks'), 3)} | {dl(g('clean', 'patient_weighted'), 3)} | {dl(g('rand', 'all_landmarks'), 3)} |")
out.append("\nLandmark sensitivity / specificity at the operating points, ≤ 5 y, clean (published): " + "; ".join(f"{NM[k]} {L['clean']['all_landmarks']['5y']['binary'][k]['sens']['value']:.3f} / {L['clean']['all_landmarks']['5y']['binary'][k]['spec']['value']:.3f} ({L['stored']['all_landmarks']['5y']['binary'][k]['sens']['value']:.3f} / {L['stored']['all_landmarks']['5y']['binary'][k]['spec']['value']:.3f})" for k in ("ml_mem", "final", "ml_final", "R49") if k in L['clean']['all_landmarks']['5y']['binary']) + ".\n")
out.append("\n### 2c. Safety (never flagged to date + two consecutive lows) and per 1,000 patient-years (λ = 0.45%/yr, model-based)\n")
out.append("| Rule | Model | Progressors de-escalated at the last pre-event endoscopy (integer range) | Progressors ever de-escalated before the event | NP de-escalated at last endoscopy | Missed (never flagged), mean (range) | Per 1,000 py in de-escalated [CI] | Exact upper |")
out.append("|---|---|---|---|---|---|---|---|")
for k in ["ml_mem", "A12", "final", "final_infold", "ml_A12", "ml_final", "ml_final_infold", "R49", "P1", "g53"]:
    for m in ("stored", "clean", "rand"):
        if k not in D[m]["two_lows"]: continue
        t = D[m]["two_lows"][k]["never_flagged_to_date_two_consecutive_lows"]; s = D[m]["safety"][k]["missed_never_flagged"]; ir = t["progressors_deescalated_at_last_integer_per_repeat"]; p = P[m][k]["incidence_0.45pct_per_year"]
        pr_ = p["progressions_per_1000_py_in_deescalated"]; mr = s["per_repeat_integer_counts"]
        out.append(f"| {NM[k]} | {m} | {t['progressors_deescalated_at_last_pre_event_endoscopy']['value']:.1f} {ci(t['progressors_deescalated_at_last_pre_event_endoscopy'])} ({min(ir)}–{max(ir)}) | {t['progressors_ever_deescalated_before_event']['value']:.1f} | "
                   f"{t['np_deescalated_at_last_endoscopy']['value']:.1f} | {s['mean']:.1f} ({min(mr)}–{max(mr)}) | {pr_['value']:.2f} [{pr_['ci95'][0]:.2f}, {pr_['ci95'][1]:.2f}] | {p['upper_95_from_exact_binomial_on_mean_misses_rounded_up']:.2f} |")
out.append("\n### 2d. Lead time\n")
out.append("| Window | Rule | Published caught / NP | Clean caught / NP | Clean: Δ caught vs R49 | Clean: Δ NP vs R49 |"); out.append("|---|---|---|---|---|---|")
for w, lab in (("gt_2y_before_event", "> 2 y (NP: all)"), ("gt_4y_before_event", "> 4 y (NP: all)"), ("gt_4y_before_event_symmetric", "> 4 y (NP: > 4 y before last)")):
    for k in ("ml_mem", "final", "final_infold", "ml_final", "ml_final_infold", "R49", "g53"):
        a, b = D["stored"]["lead_time"][w].get(k), D["clean"]["lead_time"][w].get(k)
        if not b: continue
        n_ = D["clean"]["lead_time"][w]["progressors_with_an_endoscopy_in_window"]
        out.append(f"| {lab} ({n_} P) | {NM[k]} | {a['values']['caught']['value']:.1f} / {a['values']['np_flagged']['value']:.1f} | {b['values']['caught']['value']:.1f} / {b['values']['np_flagged']['value']:.1f} | "
                   f"{dl(b['vs_R49']['caught']) if k != 'R49' else '—'} | {dl(b['vs_R49']['np_flagged']) if k != 'R49' else '—'} |")
print("\n".join(out))

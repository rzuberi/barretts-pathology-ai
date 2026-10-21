"""Exploration E4: render results/paper_final/explore_E4/*.json into docs/paper_explore_E4.md (everything after '## Results' is regenerated).
Usage: python ex_e4_render.py results/paper_final/explore_E4 scripts/paper_plan/ex_e4_notes.md  (notes_md: hand-written interpretation inserted after the tables)"""
import json, glob, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
JD = sys.argv[1] if len(sys.argv) > 1 else ROOT + "/results/paper_final/explore_E4"; DOC = ROOT + "/docs/paper_explore_E4.md"
NOTES = open(sys.argv[2]).read() if len(sys.argv) > 2 else ""
J = {os.path.basename(f)[5:-5]: json.load(open(f)) for f in sorted(glob.glob(JD + "/eval_*.json"))}; PREP = json.load(open(JD + "/prep.json")) if os.path.exists(JD + "/prep.json") else None
ORDER = ["stored", "coxx", "coxs", "coxp", "h3", "h5", "wt", "wt2", "coxx_img+stored_cnv", "coxp_img+stored_cnv", "stored_img+coxx_cnv", "wt_img+stored_cnv", "h5_img+stored_cnv",
         "ens_coxx", "ens_h3", "ens_h5", "ens_wt2", "stored+h3late", "stored+h5late", "stored+ensh3late"]
ORDER = [s for s in ORDER if s in J] + [s for s in J if s not in ORDER]
NAME = {"stored": "stored (control)", "coxx": "Cox, extended follow-up", "coxs": "Cox, study follow-up", "h3": "3-year horizon binary", "h5": "5-year horizon binary", "wt": "certainty-weighted binary",
        "coxx_img+stored_cnv": "Cox H&E + stored copy number", "stored_img+coxx_cnv": "stored H&E + Cox copy number", "wt_img+stored_cnv": "weighted H&E + stored copy number", "h5_img+stored_cnv": "5-year H&E + stored copy number",
        "coxp": "Cox, pre-event samples only", "wt2": "strongly weighted binary", "coxp_img+stored_cnv": "Cox (pre-event) H&E + stored copy number", "ens_coxx": "stored + Cox ensemble",
        "ens_h3": "stored + 3-year ensemble", "ens_h5": "stored + 5-year ensemble", "ens_wt2": "stored + strongly weighted ensemble", "stored+h3late": "stored bands, 3-year fusion",
        "stored+h5late": "stored bands, 5-year fusion", "stored+ensh3late": "stored bands, stored + 3-year ensemble fusion"}
f1 = lambda x: "n/a" if x is None else f"{x:.1f}"; f3 = lambda x: "n/a" if x is None else f"{x:.3f}"
def ci(d, f=f1): return f"{f(d['delta']):s} [{f(d['ci95'][0])}, {f(d['ci95'][1])}]" if d and d.get("ci95") and d["ci95"][0] is not None else (f(d["delta"]) if d else "n/a")
def sg(x, f=f1): return ("+" if x is not None and x > 0 else "") + f(x)
def cis(d, f=f1): return f"{sg(d['delta'], f)} [{sg(d['ci95'][0], f)}, {sg(d['ci95'][1], f)}]"
L = []
if PREP:
    p = PREP; L += ["### Time axis", "",
        f"- Progressor pre-event samples: years to event median {p['progressor_pre_event_years_to_event']['median']:.1f} (IQR {p['progressor_pre_event_years_to_event']['q1']:.1f}–{p['progressor_pre_event_years_to_event']['q3']:.1f}); {p['progressor_pre_event_gt8y']} samples > 8 years before the event (down-weighted in the weighted model). The {p['progressor_samples_not_pre_event_set_to_eps']} at/after-event progressor samples of the training set get 0.01 years.",
        f"- Non-progressor samples, study follow-up (to the last sequenced endoscopy): median {p['np_years_study_followup']['median']:.1f} years (IQR {p['np_years_study_followup']['q1']:.1f}–{p['np_years_study_followup']['q3']:.1f}); extended with the Barrett's database: median {p['np_years_extended_followup']['median']:.1f} (IQR {p['np_years_extended_followup']['q1']:.1f}–{p['np_years_extended_followup']['q3']:.1f}).",
        f"- Non-progressor patients linked {p['np_linked']} of {p['np_patients']}; follow-up extended for {p['np_extended']} (median {p['np_years_added_per_patient']['median']:.1f} years added per patient, as data2.md B2); stopped before a later HGD+ record: {p['np_later_hgdplus_stopped']}.",
        f"- Horizon training sets: non-progressor samples with ≥ 3 / ≥ 5 years extended follow-up {p['np_samples_ext_ge3']} / {p['np_samples_ext_ge5']} (study follow-up only: {p['np_samples_study_ge3']} / {p['np_samples_study_ge5']}); progressor samples within 3 / 5 years of the event (incl. at/after-event) {p['progressor_samples_le3y']} / {p['progressor_samples_le5y']}.", ""]
chk = J.get("stored", {}).get("check_reference")
if chk: L += [f"Control: the evaluation code run on the stored scores reproduces R49 ({chk['got']['R49']['caught']} caught / {chk['got']['R49']['np_flagged']} non-progressors flagged) and P1 90% ({chk['got']['P1 90%']['caught']} / {chk['got']['P1 90%']['np_flagged']}); reproduced: {'yes' if chk['reproduced'] else 'NO'}. All stored-control deltas are 0.", ""]
# ---- ranked table
rows = []
for s in ORDER:
    for k, v in J[s]["rules"].items():
        if s == "stored" and "new" in k and not k.startswith("R49-new t_low 95/t_high 90") and not k.startswith("P1-new 90"): pass
        if s == "stored": continue
        rows.append((s, k, v))
key = lambda r: (r[2]["delta_vs_R49"]["progressor_patients_caught"]["delta"] < -0.5, r[2]["delta_vs_R49"]["non_progressor_patients_flagged"]["delta"])
rows.sort(key=key)
L += ["### Ranked results against R49 and P1 (held-out patients, thresholds from training patients)", "",
      "Every rule, every scheme. Caught = progressor patients flagged at ≥ 1 pre-event endoscopy (of 32); NP flagged = non-progressor patients ever flagged (of 43). Δ = rule − reference with paired patient-bootstrap 95% CI (2,000 draws, `RandomState(0)`). Sorted: rules losing > 0.5 progressor patients against R49 last, then by Δ NP flagged against R49. R49 = 31.1 caught / 31.8 flagged, endoscopy 0.907 / 0.560, 1.94 samples sequenced per patient; P1 90% = 31.1 / 36.0, 0.898 / 0.524.", "",
      "| # | Scheme | Rule | Caught | NP flagged | Endo sens / spec | Seq / patient | Δ NP flagged vs R49 | Δ caught vs R49 | Δ NP flagged vs P1 | Δ caught vs P1 | Δ NP flagged vs same rule, stored scores | Δ caught vs same rule, stored scores |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for i, (s, k, v) in enumerate(rows, 1):
    r = v["rule"]; a = v["delta_vs_R49"]; b = v["delta_vs_P1 90%"]
    L.append(f"| {i} | {NAME.get(s, s)} | {k} | {f1(r['progressor_patients_caught'])} | {f1(r['non_progressor_patients_flagged'])} | {f3(r['endoscopy_sensitivity'])} / {f3(r['endoscopy_specificity'])} | {r['samples_sequenced_per_patient']:.2f} | "
             f"{cis(a['non_progressor_patients_flagged'])} | {cis(a['progressor_patients_caught'])} | {cis(b['non_progressor_patients_flagged'])} | {cis(b['progressor_patients_caught'])} | "
             + (f"{cis(v['delta_vs_stored_same_rule']['non_progressor_patients_flagged'])} | {cis(v['delta_vs_stored_same_rule']['progressor_patients_caught'])} |" if "delta_vs_stored_same_rule" in v else "n/a | n/a |"))
if "stored" in J:
    L += ["", "Reference: the same rules built on the stored scores (the 'same rule, stored scores' comparator; R49 is the t_low 95 / t_high 90 gate, P1 is P1-new 90% here):", "", "| Rule | Caught | NP flagged | Endo sens / spec | Seq / patient |", "|---|---|---|---|---|"]
    for k, v in J["stored"]["rules"].items(): r = v["rule"]; L.append(f"| {k} | {f1(r['progressor_patients_caught'])} | {f1(r['non_progressor_patients_flagged'])} | {f3(r['endoscopy_sensitivity'])} / {f3(r['endoscopy_specificity'])} | {r['samples_sequenced_per_patient']:.2f} |")
L += [""]
# ---- matched sensitivity
L += ["### Endoscopy specificity at matched endoscopy sensitivity", "",
      "Each endoscopy score is thresholded per repeat to the reference rule's realised endoscopy sensitivity (re-matched in every bootstrap draw; evaluation-matched, so this isolates ranking quality, not a deployable threshold). 'vs stored score' = the same matching applied to the stored score (max L-IMG for H&E, max L-LATE for the fusion), i.e. the gain from the time-aware training alone.", "",
      "| Scheme | Score | Matched to | Spec at matched sens | Δ spec vs reference [CI] | Δ spec vs stored score [CI] | NP flagged | Δ NP flagged vs reference [CI] | Δ caught vs reference |", "|---|---|---|---|---|---|---|---|---|"]
for s in ORDER:
    for ref, R in J[s]["matched_sensitivity"].items():
        for nm in ("new H&E max (P1-new score)", "new fusion max (P3-new score, sequence all)"):
            x = R[nm]; d = x["delta_vs_reference"]; ds = x["delta_vs_stored_score_at_same_sensitivity"]
            L.append(f"| {NAME.get(s, s)} | {'H&E max' if 'H&E' in nm else 'fusion max (sequence all)'} | {ref} ({f3(R['reference']['endoscopy_sensitivity'])}) | {f3(x['at_matched']['endoscopy_specificity'])} | {cis(d['endoscopy_specificity'], f3)} | {cis(ds['endoscopy_specificity'], f3)} | "
                     f"{f1(x['at_matched']['non_progressor_patients_flagged'])} | {cis(d['non_progressor_patients_flagged'])} | {sg(d['progressor_patients_caught']['delta'])} |")
L += ["", "New gates against H&E alone matched to the gate's own realised endoscopy sensitivity (round 8b S1 (a)(i) form; Δ = gate − P1):", "", "| Scheme | Gate | Δ spec [CI] | Δ NP flagged [CI] | Δ caught [CI] |", "|---|---|---|---|---|"]
for s in ORDER:
    for k, g in J[s]["gate_vs_P1_matched_to_gate"].items():
        L.append(f"| {NAME.get(s, s)} | {k} | {cis(g['endoscopy_specificity'], f3)} | {cis(g['non_progressor_patients_flagged'])} | {cis(g['progressor_patients_caught'])} |")
# ---- discrimination
L += ["", "### Discrimination", "", "Fold-stratified (pairs within the same outer fold), mean over 10 repeats, pre-event samples; Δ vs the stored score of the same modality with paired patient-bootstrap CI. Time-dependent AUROC: IPCW, sheet times, as `hf_foldstrat.py`. C-index: Harrell, fold-stratified, study or extended follow-up for non-progressors.", ""]
MS = [("auroc_sample", "AUROC sample"), ("auroc_endoscopy_max", "AUROC endoscopy (max)"), ("td_auroc_1y", "td-AUROC 1 y"), ("td_auroc_3y", "td-AUROC 3 y"), ("td_auroc_5y", "td-AUROC 5 y"), ("cindex_study", "C study FU"), ("cindex_ext", "C extended FU")]
for m in ("L-IMG", "L-CNV", "L-LATE"):
    L += [f"**{ {'L-IMG': 'H&E', 'L-CNV': 'Copy number', 'L-LATE': 'Fusion'}[m] }** (stored: " + ", ".join(f"{lab} {f3(J['stored']['discrimination'][f'{k}_{m}']['scheme'])}" for k, lab in MS) + ")" if "stored" in J else f"**{m}**", "",
          "| Scheme | " + " | ".join(lab for _, lab in MS) + " |", "|---|" + "---|" * len(MS)]
    for s in ORDER:
        if s == "stored": continue
        D = J[s]["discrimination"]; L.append(f"| {NAME.get(s, s)} | " + " | ".join(f"{f3(D[f'{k}_{m}']['scheme'])} ({sg(D[f'{k}_{m}']['delta']['delta'], f3)} [{sg(D[f'{k}_{m}']['delta']['ci95'][0], f3)}, {sg(D[f'{k}_{m}']['delta']['ci95'][1], f3)}])" for k, _ in MS) + " |")
    L += [""]
# ---- round-9 memory rule on time-aware scores (coordinator request)
MEM = {os.path.basename(f)[4:-5]: json.load(open(f)) for f in sorted(glob.glob(JD + "/mem_*.json"))}
if MEM:
    FN = {"A1:gate_al:ew": "EW memory of the R49-aligned gate score", "A1:gate_al:none": "gate score, threshold only (no memory)", "A1:img:ew": "EW memory of H&E alone"}
    L += ["### Time-aware scores inside the round-9 memory rule", "", "Round 9's nested engine (`rv9_falseflags.py` `run_families`, executed unchanged) with the scheme's scores swapped in: exponentially weighted mean of the patient's endoscopy scores so far (half-life 1, 2 or 4 years chosen on training patients), operating threshold = training progressor-patient detection ≥ R49's in the fold. Only probability-scale (binary) schemes, because the engine works on logits. 'Same rule, stored scores' = the identical rule on the stored predictions (reproduces round 9: EW gate 23.6 NP flagged / 30.7 caught).", "",
          "| Scheme | Rule | Caught | NP flagged | Seq / patient | Δ NP flagged vs R49 | Δ caught vs R49 | Δ NP flagged vs same rule, stored | Δ caught vs same rule, stored |", "|---|---|---|---|---|---|---|---|---|"]
    for s_ in ["stored"] + [x for x in MEM if x != "stored"]:
        if s_ not in MEM: continue
        for fam, o in MEM[s_]["families"].items():
            v = o["values"]; a = o["vs_R49"]; b = o["vs_same_rule_stored_scores"]
            L.append(f"| {NAME.get(s_, s_)} | {FN.get(fam, fam)} | {f1(v['caught'])} | {f1(v['np_flagged'])} | {v['seq']:.2f} | {cis(a['np_flagged'])} | {cis(a['caught'])} | {cis(b['np_flagged'])} | {cis(b['caught'])} |")
    L += [""]
# ---- apparent and fit meta
L += ["### Apparent (non-nested) numbers: OPTIMISTIC", "", "P1-new with the 90% endoscopy-sensitivity threshold set on the evaluated (held-out) endoscopies themselves, per repeat. Optimistic by construction; not a deployable rule.", "", "| Scheme | Caught | NP flagged | Endo sens / spec |", "|---|---|---|---|"]
for s in ORDER:
    a = list(J[s]["apparent"].values())[0]; L.append(f"| {NAME.get(s, s)} | {f1(a['caught'])} | {f1(a['np_flagged'])} | {f3(a['endoscopy_sensitivity'])} / {f3(a['endoscopy_specificity'])} |")
L += ["", "Fitted models (outer fits, median over 100 repeat × fold): λ, non-zero coefficients, training samples used.", "", "| Fit | λ | Non-zero coefficients | Training samples |", "|---|---|---|---|"]
FM = {}
for s in ORDER: FM.update({k: v for k, v in J[s]["fit_meta"].get("fits", {}).items() if isinstance(v, dict)})
for k in sorted(FM): x = FM[k]; L.append(f"| {k} | {x['lambda_median']} | {x['nnz_median']} | {x['n_train_median']} |")
doc = open(DOC).read(); head = doc[:doc.index("## Results")]
open(DOC, "w").write(head + "## Results\n\n" + (NOTES.strip() + "\n\n" if NOTES else "") + "\n".join(L) + "\n"); print("rendered", len(J), "schemes")

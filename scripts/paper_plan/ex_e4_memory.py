"""Exploration E4 add-on (coordinator request, 20 Oct 2026): the round-9 patient-level memory rule (A1, exponentially weighted mean of the R49-aligned gate score,
half-life 1/2/4 years chosen on training patients) applied to E4's time-aware scores. rv9_falseflags.py is executed unchanged up to its merge task (TASK=none, so no task block runs) (statics, memory,
make_ctx, run_families, refs, build_eval, contrast, pdraws: round 9's nested engine and thresholds from training patients); the scheme's scores are swapped in through
`load` exactly as ex_e4_eval.py does (its builder code is executed from that file). Only schemes built from probabilities (binary models) are used, because the round-9
engine works on logits of probabilities. Aggregates -> results/paper_final/explore_E4/mem_<scheme>.json.  TASK = mem_<scheme>"""
import os, sys, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
TASK_E4 = os.environ["TASK"]; os.environ["TASK"] = "none"
_s = open(HERE + "/rv9_falseflags.py").read(); exec(compile(_s[:_s.index('\nif TASK == "merge":')], HERE + "/rv9_falseflags.py", "exec"))   # TASK=none: its task blocks before this point do nothing
TASK = TASK_E4; OUTE = T + "/results/paper_final/explore_E4"; RE = M + "/explore_E4"; FITS = RE + "/fits"
_e = open(HERE + "/ex_e4_eval.py").read(); exec(compile(_e[_e.index("STORED = None"):_e.index("def setup_with(")], HERE + "/ex_e4_eval.py", "exec"))   # read_cfg, logit_z, SCHEMES, build
FAMS_E4 = ["A1:gate_al:ew", "A1:gate_al:none", "A1:img:ew"]
_load_stored = load
def run_scheme(sch):
    global load
    O, I, FO, meta = build(sch); assert not any(str(v).startswith("cox") or "'cox" in str(v) for v in SCHEMES[sch].values()), "Cox scores are not probabilities"
    load = lambda src: (O, I, FO)
    try: S, ST, INFO = prep_state(); R = run_families(S, ST, FAMS_E4, INFO)
    finally: load = _load_stored
    return S, R, meta
sch = TASK[4:]
S0 = setup(); D = pdraws(S0); RF = refs(S0); MF, evalm, matched_eval, nPp, nNp = build_eval(S0, D)
REF = {nm: dict(zip(("A", "dr", "ob", "po", "pb"), evalm(F, NSr))) for nm, (F, NSr) in RF.items()}
S, R, meta = run_scheme(sch); Rs = run_scheme("stored")[1] if sch != "stored" else R
out = {}
for f in FAMS_E4:
    A, dr, ob, po, pb = evalm(R[f]["sel"]["np"][0], R[f]["sel"]["np"][1]); _, dr0, ob0, _, _ = evalm(Rs[f]["sel"]["np"][0], Rs[f]["sel"]["np"][1])
    out[f] = {"name": pretty(f) + " [nested]", "values": {k: r3(v) for k, v in ob.items()}, "vs_R49": contrast(ob, dr, REF["R49"]["ob"], REF["R49"]["dr"]), "vs_P1_90": contrast(ob, dr, REF["P1 90%"]["ob"], REF["P1 90%"]["dr"]),
              "vs_same_rule_stored_scores": contrast(ob, dr, ob0, dr0), "same_rule_stored_scores": {k: r3(v) for k, v in ob0.items()},
              "selected_candidate_counts_over_100_folds": {R[f]["labels"][q]: int((np.array(R[f]["chosen"]["np"]) == q).sum()) for q in range(len(R[f]["labels"]))}, "infeasible_folds": R[f]["infeasible"]["np"]}
chk = {"R49": {k: r3(v) for k, v in REF["R49"]["ob"].items()}, "P1_90": {k: r3(v) for k, v in REF["P1 90%"]["ob"].items()}}
res = {"task": TASK, "scheme": sch, "sources": {k: str(v) for k, v in SCHEMES[sch].items()}, "families": out, "references": chk, "draws": int(len(D)),
       "note": "round-9 nested engine (rv9_falseflags.py run_families, 'np' selection: threshold = training progressor-patient detection >= R49's in the fold, half-life by lowest training NP flagged); "
               "deltas paired on rv8 pdraws (RandomState(0)); 'same rule, stored scores' = the identical family run on the stored predictions (reproduces round 9)"}
json.dump(res, open(f"{OUTE}/{TASK}.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)); print("EXE4 DONE", TASK, json.dumps({f: (o["values"]["caught"], o["values"]["np_flagged"]) for f, o in out.items()}), flush=True)

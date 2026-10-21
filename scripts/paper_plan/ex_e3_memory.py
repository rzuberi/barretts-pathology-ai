"""Exploration E3 (docs/paper_explore_E3.md): the round-9 patient-memory rules with a new H&E score. The module body of rv9_falseflags.py (which runs
rv8b_step3.py and rv8_landscape.py) is executed unchanged with a task name that matches none of its branches (definitions only): statics, efeat, gate, the A1 memory builders, the nested engine
run_families (threshold = highest giving the R49 gate's own training progressor-patient detection; half-life chosen by lowest training non-progressor
patients flagged) and build_eval. The E3 variant readers come from ex_e3_score.py (helpers only). Only L-IMG changes (L-LATE = (L-CNV + sample L-IMG) / 2).
Endoscopy-aggregated variants (mlogit:...): each sample carries its endoscopy's score plus 1e-6 x its own sample logit, so the endoscopy maximum is the
aggregated score (to 1e-6 on the logit scale) and the sequenced top-H&E sample is the sample model's top sample.
Reference rules (R49, P1 90%) are the stored round-8 ones on the same bootstrap draws. VAR -> results/paper_final/explore_E3/mem_<VAR>.json."""
import os, sys, re, glob, json, numpy as np, pandas as pd
TH = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"; PP = TH + "/scripts/paper_plan"; sys.path.insert(0, PP); CODE = os.path.dirname(os.path.abspath(__file__))
VAR = os.environ["VAR"]; os.environ["TASK"] = "exe3mem"
_f = __file__; __file__ = PP + "/rv9_falseflags.py"; exec(compile(open(__file__).read(), __file__, "exec")); __file__ = _f   # whole module: TASK matches no branch, definitions only
_e = open(CODE + "/ex_e3_score.py").read(); exec(compile(_e[_e.index('E3 = M + "/explore_E3"'):_e.index('\n# ------------------------------------------------------------------ one variant')], CODE + "/ex_e3_score.py", "exec"))
FAMS_E3 = ["A1:img:none", "A1:img:ew", "A1:gate_al:none", "A1:gate_al:ew"]
S0 = setup(); D = pdraws(S0); RF = refs(S0); MF, evalm, matched_eval, nPp, nNp = build_eval(S0, D); REF = {}
for nm, (F, NSr) in RF.items(): A, dr, ob, po, pb = evalm(F, NSr); REF[nm] = dict(dr=dr, ob=ob)
ST0, INFO0 = statics(S0), TC.triage(S0["O"], S0["I"], S0["FO"], S0["popm"], 0.95, 0.90)[2]; R0 = run_families(S0, ST0, FAMS_E3, INFO0); SREF = {}
for f in FAMS_E3: A, dr, ob, po, pb = evalm(*R0[f]["sel"]["np"]); SREF[f] = dict(dr=dr, ob=ob)   # the stored L-IMG under the same memory rule (round 9)
Oi, Ii, Op, Ip, vinfo = variant(VAR)
if Op is not None:   # endoscopy score carried by every sample, sample-model tie-break for the top-H&E pick
    Oi = sg(lg_(Oi) + 1e-6 * lg_(Op)); Ii = {k: pd.Series(sg(lg_(Ii[k].values) + 1e-6 * lg_(Ip[k].reindex(Ii[k].index).values)), index=Ii[k].index) for k in Ii}
O = dict(O0); I = dict(I0); Ops = Oi if Op is None else Op; Ips = Ii if Ip is None else Ip
O["L-IMG"] = Oi; O["L-LATE"] = (O["L-CNV"] + Ops) / 2; I["L-IMG"] = Ii; I["L-LATE"] = {k: (I["L-CNV"][k] + Ips[k].reindex(I["L-CNV"][k].index)) / 2 for k in I["L-CNV"]}
load = lambda src: (O, I, FO); S, ST, INFO = prep_state(); assert (S["plab"] == S0["plab"]).all() and S["E"] == S0["E"]
R = run_families(S, ST, FAMS_E3, INFO); out = []
for f in FAMS_E3:
    Rf = R[f]; F, NS = Rf["sel"]["np"]; A, dr, ob, po, pb = evalm(F, NS)
    out.append({"family": f, "name": pretty(f), "values": {k: r3(v) for k, v in ob.items()}, "vs_R49": contrast(ob, dr, REF["R49"]["ob"], REF["R49"]["dr"]), "vs_P1_90": contrast(ob, dr, REF["P1 90%"]["ob"], REF["P1 90%"]["dr"]), "vs_stored_same_rule": contrast(ob, dr, SREF[f]["ob"], SREF[f]["dr"]), "stored_same_rule": {k: r3(v) for k, v in SREF[f]["ob"].items()},
                "selected_candidate_counts_over_100_folds": {Rf["labels"][q]: int((np.array(Rf["chosen"]["np"]) == q).sum()) for q in range(len(Rf["labels"]))}, "infeasible_folds": Rf["infeasible"]["np"]})
jdump({"variant": VAR, "info": vinfo, "methods": out, "references": {k: {m: r3(v) for m, v in REF[k]["ob"].items()} for k in REF}, "draws": int(len(D)),
       "note": "round-9 nested engine (rv9_falseflags.py run_families, 'np' selection) with the variant's L-IMG; deltas paired against the stored R49 and P1 90% on the same draws; "
               "round-9 reference for the stored model: A1 gate score, R49-aligned, exponentially weighted = 30.7 caught / 23.6 non-progressor patients flagged"}, "mem_" + fname(VAR))

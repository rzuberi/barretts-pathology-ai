"""Item 2, cut-off grid (docs/paper_triage_robustness.md @ 32653f1). TASK = <src>_<pop>. t_low at 90/95/98% training sensitivity x t_high at 85/90/95%
training specificity; L-LATE in the middle band; 95/90 must reproduce results/paper_final/triage/<TASK>.json exactly (else stop).
Aggregates -> results/paper_final/triage_robustness/grid_<TASK>.json."""
import os, json, numpy as np
from tb_common import *
TASK = os.environ["TASK"]; src, pop = TASK.split("_", 1); popm = popmask(pop); rows = np.where(popm)[0]; os.makedirs(OUT, exist_ok=True)
O, I, FO = load(src); comp, _ = comparators(O, I, FO, popm); CA, CB = vote(comp["A"]), vote(comp["B"])
CELLS = {}
for ql in (0.90, 0.95, 0.98):
    for qh in (0.85, 0.90, 0.95):
        band, call, info = triage(O, I, FO, popm, ql, qh); CELLS[(ql, qh)] = (vote(call), band, info)
ref = json.load(open(f"{T}/results/paper_final/triage/{TASK}.json"))["strategies"]["triage_LATE"]; c0 = CELLS[(0.95, 0.90)][0]; s0, p0 = sens_spec(c0, rows)
assert r3(s0) == ref["sensitivity"]["value"] and r3(p0) == ref["specificity"]["value"] and int((y[rows] == 1).sum() - c0[rows][y[rows] == 1].sum()) == ref["missed_progressor_samples"], "95/90 cell does not reproduce paper_triage.md"
boots = boots_for(rows)
def metr(b): return {**{c: sens_spec(v[0], b) for c, v in CELLS.items()}, "A": sens_spec(CA, b), "B": sens_spec(CB, b)}
obs = metr(rows); BS = [metr(b) for b in boots]; res = {"task": TASK, "prespec": "32653f1", "reproduces_paper_triage_95_90": True, "cells": []}
for (ql, qh), (c, band, info) in CELLS.items():
    yr = y[rows] == 1; mp = pd.Series(~c[rows][yr], index=pat[rows][yr]); o = {"t_low_sens": ql, "t_high_spec": qh, "sensitivity": r3(obs[(ql, qh)][0]), "specificity": r3(obs[(ql, qh)][1]),
         "share_sequenced": r3((band[rows] == 1).mean(0).mean()), "progressor_patients_all_missed": int(mp.groupby(level=0).all().sum()), "missed_progressor_samples": int((yr & ~c[rows]).sum()), "folds_no_middle": info["no_mid"]}
    for rf in ("A", "B"):
        for i, met in ((0, "sens"), (1, "spec")):
            d = np.array([x[(ql, qh)][i] - x[rf][i] for x in BS]); o[f"d_{met}_vs_{rf}"] = {"delta": r3(obs[(ql, qh)][i] - obs[rf][i]), "ci95": pct(d), "one_sided_95_lower": r3(np.percentile(d, 5))}
    res["cells"].append(o)
json.dump(res, open(f"{OUT}/grid_{TASK}.json", "w"), indent=1); print("TB GRID DONE", TASK, flush=True)

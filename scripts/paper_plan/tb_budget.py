"""Item 1, sequencing-budget curve (docs/paper_triage_robustness.md @ 32653f1). TASK = <src>_<pop>. Middle band = the s share of training samples nearest
(in rank) to the training Youden threshold of L-IMG; L-LATE inside at the 80%-sensitivity threshold of training middle-band cases (fallback < 5 cases).
Aggregates -> results/paper_final/triage_robustness/budget_<TASK>.json."""
import os, json, numpy as np
from tb_common import *
TASK = os.environ["TASK"]; src, pop = TASK.split("_", 1); popm = popmask(pop); rows = np.where(popm)[0]; os.makedirs(OUT, exist_ok=True)
O, I, FO = load(src); comp, _ = comparators(O, I, FO, popm); CA, CB = vote(comp["A"]), vote(comp["B"])
def youden(s, yc):
    u = np.unique(s); P = yc.sum(); N = (~yc).sum(); best, tb = -np.inf, u[0]
    for t in u:
        j = (s[yc] >= t).sum() / P - (s[~yc] >= t).sum() / N
        if j > best + 1e-12: best, tb = j, t   # strict improvement: ties keep the smaller threshold
    return tb
S_GRID = np.round(np.arange(0, 1.0001, 0.05), 2); CALLS, REAL, FB = {}, {}, {}
for s_ in S_GRID:
    call = np.zeros((n, 10), bool); seq = np.zeros((n, 10), bool); fb = 0
    for j, r in enumerate(REPS):
        for k in range(1, 11):
            si = I["L-IMG"][(r, k)]; tri = si.index.values[popm[si.index.values]]; st = si.reindex(tri).values; yc = y[tri] == 1
            te = FO[:, j] == k; sh = O["L-IMG"][te, j]; tj = youden(st, yc); m = int(round(s_ * len(st)))
            if s_ >= 1.0: inb = np.ones(te.sum(), bool); mid_tr = np.ones(len(st), bool)
            elif m == 0: inb = np.zeros(te.sum(), bool); mid_tr = np.zeros(len(st), bool)
            else:
                o = np.argsort(st, kind="stable"); ss = st[o]; p0 = int(np.searchsorted(ss, tj, "left")); up_n = int(np.ceil(m / 2)); dn_n = m // 2
                hi_i = min(len(ss) - 1, p0 + up_n - 1); lo_i = max(0, p0 - dn_n); got = hi_i - lo_i + 1
                if got < m: lo_i = max(0, lo_i - (m - got)); got = hi_i - lo_i + 1
                if got < m: hi_i = min(len(ss) - 1, hi_i + (m - got))
                lo, hi = ss[lo_i], ss[hi_i]; inb = (sh >= lo) & (sh <= hi); mid_tr = (st >= lo) & (st <= hi)
            ml = I["L-LATE"][(r, k)].reindex(tri).values; cm = ml[mid_tr & yc]
            if mid_tr.any():
                if len(cm) >= 5: ts = thr_sens(cm, 0.8)
                else: ts = thr_sens(ml[yc], 0.8); fb += 1
            else: ts = np.inf
            if s_ >= 1.0: c = O["L-LATE"][te, j] >= ts                       # sequence everyone (= B)
            elif m == 0: c = sh >= tj                                          # H&E only at the Youden threshold
            else: c = np.where(inb, O["L-LATE"][te, j] >= ts, sh > hi)        # below band cleared, above flagged
            call[te, j] = c; seq[te, j] = inb
    CALLS[s_] = vote(call); REAL[s_] = seq[rows].mean(0).mean(); FB[s_] = fb
def metr(b):
    o = {s_: sens_spec(CALLS[s_], b) for s_ in S_GRID}; o["A"] = sens_spec(CA, b); o["B"] = sens_spec(CB, b); return o
def smallest(o, ref):
    for s_ in S_GRID:
        if o[s_][0] >= o[ref][0] - 1e-12 and o[s_][1] >= o[ref][1] - 1e-12: return float(s_)
    return np.nan
obs = metr(rows); boots = boots_for(rows); BS = [metr(b) for b in boots]
res = {"task": TASK, "prespec": "32653f1", "n_samples": int(len(rows)), "grid": []}
for s_ in S_GRID:
    se = np.array([x[s_][0] for x in BS]); sp = np.array([x[s_][1] for x in BS])
    res["grid"].append({"target_share": float(s_), "realised_share": r3(REAL[s_]), "sensitivity": r3(obs[s_][0]), "sens_ci95": pct(se), "specificity": r3(obs[s_][1]), "spec_ci95": pct(sp), "t_seq_fallbacks": FB[s_]})
res["A"] = {"sensitivity": r3(obs["A"][0]), "specificity": r3(obs["A"][1])}; res["B"] = {"sensitivity": r3(obs["B"][0]), "specificity": r3(obs["B"][1])}
for ref in ("A", "B"):
    sb = np.array([smallest(x, ref) for x in BS]); res[f"smallest_s_vs_{ref}"] = {"value": smallest(obs, ref), "ci95": pct(sb[np.isfinite(sb)]) if np.isfinite(sb).any() else [None, None], "draws_not_reached": int((~np.isfinite(sb)).sum())}
json.dump(res, open(f"{OUT}/budget_{TASK}.json", "w"), indent=1); print("TB BUDGET DONE", TASK, res["smallest_s_vs_A"], res["smallest_s_vs_B"], flush=True)

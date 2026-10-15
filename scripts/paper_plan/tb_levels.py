"""Item 4, endoscopy and patient level (docs/paper_triage_robustness.md @ 32653f1). TASK = <src>_<pop>. Endoscopy unit = (patient, Endoscopy number);
an endoscopy is positive if any of its population samples has a positive final (>= 6 of 10) call. Aggregates -> results/paper_final/triage_robustness/levels_<TASK>.json."""
import os, json, numpy as np
from tb_common import *
TASK = os.environ["TASK"]; src, pop = TASK.split("_", 1); popm = popmask(pop); rows = np.where(popm)[0]; os.makedirs(OUT, exist_ok=True)
O, I, FO = load(src); band, call, _ = triage(O, I, FO, popm); comp, _ = comparators(O, I, FO, popm)
CALL = {"triage_LATE": vote(call), "A": vote(comp["A"]), "B": vote(comp["B"]), "C": vote(comp["C"])}
eu = np.unique(ENDO[rows]); ei = {e: i for i, e in enumerate(eu)}; erow = np.array([ei[e] for e in ENDO[rows]])
e_pat = pd.Series(pat[rows]).groupby(erow).first().values; e_y = pd.Series(y[rows]).groupby(erow).max().values
def boots_tagged(rows, seed=0, nb=NB):   # same draws as tb_common.boots_for, plus a copy index per sample (position of the patient in the draw)
    pp = pat[rows]; up = np.unique(pp); ro = {q: rows[pp == q] for q in up}; rng = np.random.RandomState(seed); out = []
    while len(out) < nb:
        dr = rng.choice(len(up), len(up)); b = np.concatenate([ro[up[i]] for i in dr]); t = np.concatenate([np.full(len(ro[up[i]]), q) for q, i in enumerate(dr)])
        if 0 < y[b].sum() < len(b): out.append((b, t))
    return out
def endo_metrics(b, t):
    key = pd.Series(ENDO[b]).astype(str).values + "#" + t.astype(str); lab = pd.Series(y[b]).groupby(key).max(); o = {}
    for s, c in CALL.items():
        pos = pd.Series(c[b]).groupby(key).max(); o[s] = (pos[lab == 1].mean(), 1 - pos[lab == 0].mean())
    return o
obs = endo_metrics(rows, np.zeros(len(rows), int)); BS = [endo_metrics(b, t) for b, t in boots_tagged(rows)]
res = {"task": TASK, "prespec": "32653f1", "n_samples": int(len(rows)), "n_endoscopies": int(len(eu)), "n_progressor_endoscopies": int(e_y.sum()), "endoscopy_level": {}, "patient_level": {}}
for s in CALL:
    res["endoscopy_level"][s] = {"sensitivity": r3(obs[s][0]), "sens_ci95": pct([x[s][0] for x in BS]), "specificity": r3(obs[s][1]), "spec_ci95": pct([x[s][1] for x in BS])}
seq_e = np.array([pd.Series(band[rows, j] == 1).groupby(erow).max().mean() for j in range(10)])
res["sequenced"] = {"share_endoscopies_with_any_sample_sequenced": r3(seq_e.mean()), "share_samples_sequenced": r3((band[rows] == 1).mean(0).mean())}
ppat = pd.Series(y[rows], index=pat[rows]).groupby(level=0).max()
for s, c in CALL.items():
    pos = pd.Series(c[rows], index=pat[rows]).groupby(level=0).max()
    res["patient_level"][s] = {"progressor_caught": int(((ppat == 1) & pos).sum()), "progressor_missed": int(((ppat == 1) & ~pos).sum()), "nonprogressor_false_positive": int(((ppat == 0) & pos).sum()), "nonprogressor_true_negative": int(((ppat == 0) & ~pos).sum())}
epp = pd.Series(eu).str.split("|").str[0].value_counts(); spe = pd.Series(erow).value_counts()
res["structure"] = {"n_patients": int(len(ppat)), "endoscopies_per_patient": {"mean": r3(epp.mean()), "median": float(epp.median()), "range": [int(epp.min()), int(epp.max())]},
                    "samples_per_endoscopy": {"mean": r3(spe.mean()), "median": float(spe.median()), "range": [int(spe.min()), int(spe.max())]}}
json.dump(res, open(f"{OUT}/levels_{TASK}.json", "w"), indent=1); print("TB LEVELS DONE", TASK, flush=True)

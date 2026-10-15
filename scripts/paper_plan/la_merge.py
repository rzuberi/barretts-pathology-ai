"""Latent space Section 1 merge (docs/paper_latent_attention.md @ dfac9ad). POP = pre | pre_ndbe. Fold-stratified probe AUROC (mean over 10 repeats),
patient bootstrap (2,000 draws, seed 0, resampled patients keep their folds, probes not refitted), paired delta with max-T, nearest-neighbour patient identity.
Aggregates -> results/paper_final/latent_attention/probe_<POP>.json."""
import os, glob, json, numpy as np, pandas as pd
from scipy.stats import rankdata
T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"; M = T + "/feasibility/paper_plan/killcoyne_mm"; LA = M + "/latent_attention"; OUT = T + "/results/paper_final/latent_attention"; os.makedirs(OUT, exist_ok=True)
POP = os.environ["POP"]; NB = 2000; REPS = range(1, 11); REPRS = ["R0", "R1", "R2", "R3", "R4", "R2p", "R3p", "R4p"]; r3 = lambda x: None if x is None or not np.isfinite(x) else round(float(x), 3)
files = sorted(glob.glob(f"{LA}/probe/{POP}_rep_*_R*.csv")); assert len(files) == 80, len(files)
fpd = max(json.load(open(f.replace(".csv", ".json")))["fingerprint_max_abs_diff"] for f in files)
D = pd.concat([pd.read_csv(f, dtype={"Sample": str}) for f in files])
C = pd.read_csv(M + "/set_C.csv", dtype=str); ids = C.Sample.values; ix = {s: i for i, s in enumerate(ids)}
SM = pd.read_csv(M + "/horizons/samples.csv", dtype={"Sample": str, "Patient": str}).set_index("Sample").reindex(ids); pat = SM.Patient.values; y = SM.y.values.astype(int)
popm = SM.pre.astype(bool).values & (True if POP == "pre" else SM.ndbe.astype(bool).values); rows = np.where(popm)[0]
D["i"] = D.Sample.map(ix); P = D[D.target != "nn_same_patient"]; NN = D[D.target == "nn_same_patient"]
FO = {r: pd.read_csv(f"{M}/cv/preds/cfg_00_rep_{r:02d}.csv", dtype={"Sample": str}).set_index("Sample").fold.reindex(ids).values.astype(int) for r in REPS}
KEYS = sorted(set(zip(P.repr, P.target)))
PR, LB = {}, {}
for (rp, t), g in P.groupby(["repr", "target"]):
    for r, h in g.groupby("rep"):
        a = np.full(len(ids), np.nan); a[h.i.values] = h.pred.values; PR[(rp, t, r)] = a
        l = np.full(len(ids), np.nan); l[h.i.values] = h.label.values; LB[(rp, t)] = l
def fs_auc(key, b):
    rp, t = key; lab = LB[key]; out = []
    for r in REPS:
        p = PR.get((rp, t, r))
        if p is None: continue
        f = FO[r][b]; x = p[b]; l = lab[b]; ok = np.isfinite(x) & np.isfinite(l); num = den = 0.0
        for k in range(1, 11):
            m = ok & (f == k); yk = l[m]; n1 = (yk == 1).sum(); n0 = (yk == 0).sum()
            if n1 == 0 or n0 == 0: continue
            rk = rankdata(x[m]); num += rk[yk == 1].sum() - n1 * (n1 + 1) / 2; den += n1 * n0
        if den: out.append(num / den)
    return float(np.mean(out)) if out else np.nan
pp = pat[rows]; up = np.unique(pp); ro = {q: rows[pp == q] for q in up}; rng = np.random.RandomState(0); boots = []
while len(boots) < NB:
    b = np.concatenate([ro[up[i]] for i in rng.choice(len(up), len(up))])
    if 0 < y[b].sum() < len(b): boots.append(b)
obs = {k: fs_auc(k, rows) for k in KEYS}; BS = {k: np.array([fs_auc(k, b) for b in boots]) for k in KEYS}
pct = lambda a: [r3(np.nanpercentile(a, 2.5)), r3(np.nanpercentile(a, 97.5))]
res = {"task": POP, "prespec": "dfac9ad", "n_samples": int(len(rows)), "n_patients": int(len(up)), "n_progressor_samples": int(y[rows].sum()), "draws": NB, "fingerprint_max_abs_diff": fpd, "probe": {}, "paired": {}, "nn_patient": {}}
for (rp, t) in KEYS:
    Cs = P[(P.repr == rp) & (P.target == t)].groupby(["rep", "fold"]).C.first()
    res["probe"].setdefault(t, {})[rp] = {"auroc": r3(obs[(rp, t)]), "ci95": pct(BS[(rp, t)]), "C_median": float(np.median(Cs)), "n_labelled": int(np.isfinite(LB[(rp, t)][rows]).sum())}
for fam, comps in (("their", [("R2", "R1"), ("R3", "R1"), ("R2", "R3")]), ("pkg", [("R2p", "R1"), ("R3p", "R1"), ("R2p", "R3p")])):
    d0 = np.array([obs[(a, "progressor")] - obs[(b, "progressor")] for a, b in comps]); Db = np.column_stack([BS[(a, "progressor")] - BS[(b, "progressor")] for a, b in comps])
    sd = np.nanstd(Db, 0, ddof=1); Tb = np.nanmax(np.abs((Db - d0) / sd), 1); Tobs = np.abs(d0 / sd)
    for j, (a, b) in enumerate(comps):
        dj = Db[:, j]; res["paired"][f"{a}_vs_{b}"] = {"family": fam, "delta": r3(d0[j]), "ci95": pct(dj), "p_unadjusted": round(float(min(1.0, 2 * min((dj <= 0).mean(), (dj >= 0).mean()))), 4),
                                                       "p_maxT_adjusted": round(float((Tb >= Tobs[j]).mean()), 4)}
for rp, g in NN.groupby("repr"):
    sh = g.groupby("rep").pred.mean(); ch = g.groupby("rep").label.mean()
    res["nn_patient"][rp] = {"share_same_patient": r3(sh.mean()), "chance": r3(ch.mean()), "ratio": r3(sh.mean() / ch.mean()), "per_repeat_range": [r3(sh.min()), r3(sh.max())]}
json.dump(res, open(f"{OUT}/probe_{POP}.json", "w"), indent=1); print("LA MERGE DONE", POP, flush=True)

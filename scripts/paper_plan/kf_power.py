"""Killcoyne final item 3 (docs/paper_plan_killcoyne_final.md @ 8342ac0): power of the ACE-B primary contrast by simulation from set C.
TASK = <S1|S2|S3>_<E1|E2>_<block 0-7>; 250 simulations per block (simulation index = block*250 + i, used as its seed).
Parts -> feasibility/paper_plan/killcoyne_mm/final/power/."""
import json, glob, os, sys, numpy as np, pandas as pd
from scipy.stats import rankdata
T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"; M = T + "/feasibility/paper_plan/killcoyne_mm"; O = M + "/final/power"; os.makedirs(O, exist_ok=True)
TASK = os.environ["TASK"]; scen, eff, blk = TASK.split("_"); blk = int(blk); outf = f"{O}/{TASK}.json"
matched = scen.endswith("m"); scen = scen.rstrip("m")   # post hoc: S1m/S2m draw each patient's NDBE sample count from ACE-B's per-case distribution
if os.path.exists(outf): sys.exit(0)
def auc(y, s):
    r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1; return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
C = pd.read_csv(M + "/set_C.csv", dtype=str).set_index("Sample"); C["y"] = (C.Status == "P").astype(int)
V = pd.concat([pd.read_csv(f, dtype={"Sample": str, "Patient": str}) for f in sorted(glob.glob(M + "/cv/preds/*.csv"))])
avg = lambda cfg: V[V.cfg == cfg].pivot_table(index="Sample", columns="rep", values="pred").reindex(C.index).mean(1).values
cnv, img = avg(1), avg(2); late = (cnv + img) / 2           # package L-CNV, L-IMG; L-LATE (repeat-averaged, as the development estimate)
nd = (C.Pathology == "NDBE").values; y = C.y.values[nd]; pat = C.Patient.values[nd]; cn, la = cnv[nd], late[nd]
d_full = auc(y, la) - auc(y, cn)
def blend(w): return (1 - w) * cn + w * la
if eff == "E1": w = 1.0
else:   # bisection: blended fusion score whose set-C NDBE delta is +0.021
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2; (lo, hi) = (mid, hi) if auc(y, blend(mid)) - auc(y, cn) < 0.021 else (lo, mid)
    w = (lo + hi) / 2
fu = blend(w); d_set = auc(y, fu) - auc(y, cn)
Z = json.load(open(M + "/final/aceb_size.json")); size = Z["scenarios"][scen]; nP, nN = size["n_P"], size["n_NP"]
H = Z["IM_per_case_hist"]; pst = ["prevalent", "progressed"] if scen == "S1" else ["progressed"]
def hist(keys):
    c = {}
    for k in keys:
        for n, m in H.get(k, {}).items(): c[int(n)] = c.get(int(n), 0) + m
    ks = np.array(sorted(c)); return ks, np.array([c[k] for k in ks], float) / sum(c.values())
HP, HN = hist(pst), hist(["non_progressor"])
rows_of = {p: np.where(pat == p)[0] for p in np.unique(pat)}; ypat = {p: y[r[0]] for p, r in rows_of.items()}
Pp = np.array(sorted(p for p in rows_of if ypat[p] == 1)); Np = np.array(sorted(p for p in rows_of if ypat[p] == 0))
out = []
for i in range(250):
    sim = blk * 250 + i; rng = np.random.RandomState(sim)
    drawn = np.concatenate([rng.choice(Pp, nP, replace=True), rng.choice(Np, nN, replace=True)])
    parts = [rows_of[p] for p in drawn]
    if matched:   # each drawn patient keeps k of its NDBE samples (k from ACE-B's per-case NDBE count for its class, capped at what it has)
        parts = [rng.choice(q, min(len(q), int(rng.choice((HP if ypat[p] == 1 else HN)[0], p=(HP if ypat[p] == 1 else HN)[1]))), replace=False) for p, q in zip(drawn, parts)]
    rr = np.concatenate(parts); ys = y[rr]; a, b = fu[rr], cn[rr]; delta = auc(ys, a) - auc(ys, b)
    brng = np.random.RandomState(sim); bd = []; k = len(parts)
    while len(bd) < 2000:
        ix = brng.choice(k, k); q = np.concatenate([parts[j] for j in ix]); yq = y[q]
        if 0 < yq.sum() < len(yq): bd.append(auc(yq, fu[q]) - auc(yq, cn[q]))
    out.append({"sim": sim, "delta": delta, "lower95": float(np.percentile(bd, 5)), "n_samples": int(len(rr))})
json.dump({"task": TASK, "w": w, "set_C_delta_at_w": d_full if eff == "E1" else d_set, "n_P": nP, "n_NP": nN, "sims": out}, open(outf, "w")); print("KF POWER DONE", TASK)

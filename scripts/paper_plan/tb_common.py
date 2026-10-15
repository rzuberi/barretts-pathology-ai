"""Shared pieces for docs/paper_triage_robustness.md (pre-specification 32653f1): stored outer and inner predictions, the paper_triage.md cut-off rules
(tr_triage.py @ 8462ef2, unchanged in substance; optional integer weights for bootstrap re-selection), comparators, vote, patient bootstrap draws."""
import glob, numpy as np, pandas as pd
T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"; M = T + "/feasibility/paper_plan/killcoyne_mm"; HZ = M + "/horizons"; REPS = list(range(1, 11)); NB = 2000
OUT = T + "/results/paper_final/triage_robustness"; ROW = M + "/triage_robustness"
r3 = lambda x: None if x is None or not np.isfinite(x) else round(float(x), 3)
SM = pd.read_csv(HZ + "/samples.csv", dtype={"Sample": str, "Patient": str}); n = len(SM); ids = SM.Sample.values; ix = pd.Series(np.arange(n), index=ids)
y = SM.y.values.astype(int); pat = SM.Patient.values; PRE = SM.pre.astype(bool).values; NDBE = SM.ndbe.astype(bool).values
ENDO = (pd.Series(pat) + "|" + pd.read_csv(M + "/set_C.csv", dtype=str).set_index("Sample").Endoscopy.reindex(ids).values).values   # Endoscopy is numbered within patient (11 distinct values): unit = (patient, endoscopy)
def popmask(pop): return PRE & (True if pop == "pre" else NDBE)
def _outer(files):
    D = pd.concat([pd.read_csv(f, dtype={"Sample": str}) for f in files])
    return D.pivot_table(index="Sample", columns="rep", values="pred").reindex(ids)[REPS].values, D.pivot_table(index="Sample", columns="rep", values="fold").reindex(ids)[REPS].values.astype(int)
def _inner(files):
    D = pd.concat([pd.read_csv(f, dtype={"Sample": str}) for f in files]); assert D.groupby(["rep", "fold"]).ngroups == 100
    return {(r, k): pd.Series(d.pred.values, index=ix[d.Sample].values) for (r, k), d in D.groupby(["rep", "fold"])}
def load(src):
    kvf = lambda c: sorted(glob.glob(f"{M}/cv/preds/cfg_{c:02d}_rep_*.csv")); O, I = {}, {}
    O["L-CNV"], FO = _outer(kvf(0 if src == "their" else 1)); O["L-IMG"], F2 = _outer(kvf(2)); assert (FO == F2).all(); O["L-LATE"] = (O["L-CNV"] + O["L-IMG"]) / 2
    I["L-CNV"] = _inner(sorted(glob.glob(f"{HZ}/inner/cnv_{src}_rep_*_fold_*.csv"))); I["L-IMG"] = _inner(sorted(glob.glob(f"{HZ}/inner/img_rep_*_fold_*.csv")))
    I["L-LATE"] = {k: (I["L-CNV"][k] + I["L-IMG"][k].reindex(I["L-CNV"][k].index)) / 2 for k in I["L-CNV"]}
    return O, I, FO
def thr_sens(cs, q, w=None):   # largest threshold keeping >= q sensitivity (score >= thr positive); with integer weights = replication
    if w is not None: cs = np.repeat(cs, w)
    cs = np.sort(cs)[::-1]; return cs[int(np.ceil(q * len(cs))) - 1] if len(cs) else np.inf
def c_star(ct, q, w=None):     # the ceil(q * n_controls)-th smallest training-control score; high band = score > c*
    if w is not None: ct = np.repeat(ct, w)
    ct = np.sort(ct); return ct[int(np.ceil(q * len(ct))) - 1] if len(ct) else -np.inf
def triage(O, I, FO, popm, ql=0.95, qh=0.90, mid="L-LATE", W=None):
    """band (n x 10: 0 low, 1 middle, 2 high), calls (n x 10), info. W = per-sample integer multiplicity for training rows (None = 1)."""
    band = np.full((n, 10), -1); call = np.zeros((n, 10), bool); info = {"no_mid": 0, "fallback": 0, "t_low": [], "c_star": [], "t_seq": []}
    for j, r in enumerate(REPS):
        for k in range(1, 11):
            si = I["L-IMG"][(r, k)]; tri = si.index.values[popm[si.index.values]]; w = None if W is None else W[tri]
            if w is not None: keep = w > 0; tri = tri[keep]; w = w[keep]
            s_tr = si.reindex(tri).values; yc = y[tri] == 1
            tl = thr_sens(s_tr[yc], ql, None if w is None else w[yc]); cs = c_star(s_tr[~yc], qh, None if w is None else w[~yc]); info["t_low"].append(tl); info["c_star"].append(cs)
            te = FO[:, j] == k; s = O["L-IMG"][:, j]
            if tl > cs: info["no_mid"] += 1; band[te, j] = np.where(s[te] > cs, 2, 0); mid_tr = np.zeros(len(tri), bool)
            else: band[te, j] = np.where(s[te] < tl, 0, np.where(s[te] > cs, 2, 1)); mid_tr = (s_tr >= tl) & (s_tr <= cs)
            mi = I[mid].get((r, k)).reindex(tri).values; sel = mid_tr & yc
            if (sel.sum() if w is None else w[sel].sum()) >= 5: ts = thr_sens(mi[sel], 0.8, None if w is None else w[sel])
            else: ts = thr_sens(mi[yc], 0.8, None if w is None else w[yc]); info["fallback"] += 1
            info["t_seq"].append(ts); call[te, j] = np.where(band[te, j] == 1, O[mid][te, j] >= ts, band[te, j] == 2)
    return band, call, info
def comparators(O, I, FO, popm, W=None):
    COMPM = {"A": "L-CNV", "B": "L-LATE", "C": "L-IMG"}; calls = {a: np.zeros((n, 10), bool) for a in COMPM}; thr = {a: [] for a in COMPM}
    for j, r in enumerate(REPS):
        for k in range(1, 11):
            tri = I["L-IMG"][(r, k)].index.values; tri = tri[popm[tri]]; w = None if W is None else W[tri]
            if w is not None: keep = w > 0; tri = tri[keep]; w = w[keep]
            yc = y[tri] == 1; te = FO[:, j] == k
            for a, m in COMPM.items():
                t = thr_sens(I[m][(r, k)].reindex(tri).values[yc], 0.8, None if w is None else w[yc]); thr[a].append(t); calls[a][te, j] = O[m][te, j] >= t
    return calls, thr
vote = lambda c: c.sum(1) >= 6
def boots_for(rows, seed=0, nb=NB):
    pp = pat[rows]; up = np.unique(pp); ro = {q: rows[pp == q] for q in up}; rng = np.random.RandomState(seed); out = []
    while len(out) < nb:
        b = np.concatenate([ro[up[i]] for i in rng.choice(len(up), len(up))])
        if 0 < y[b].sum() < len(b): out.append(b)
    return out
def sens_spec(c, b): yb = y[b]; return c[b][yb == 1].mean(), 1 - c[b][yb == 0].mean()
pct = lambda a: [r3(np.nanpercentile(a, 2.5)), r3(np.nanpercentile(a, 97.5))]
p2 = lambda a: round(float(min(1.0, 2 * min(np.nanmean(np.asarray(a) <= 0), np.nanmean(np.asarray(a) >= 0)))), 4)

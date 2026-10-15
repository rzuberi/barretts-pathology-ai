"""Horizon answers Q0 sanity check (added after the pre-specification, labelled as such in the output): IPCW td-AUROC at 1/3/5 years of the
intercept-only prediction (each sample's outer training-fold prevalence, `train_prev` in the kv_cv.R prediction files, mean over 10 repeats) on all
pre-event samples and NDBE pre-event samples, with the same patient bootstrap. Explains L-CLIN without grade < 0.5 if it matches.
-> results/paper_final/horizon_answers/q0_intercept_only.json"""
import json, glob, numpy as np, pandas as pd
T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"; M = T + "/feasibility/paper_plan/killcoyne_mm"; HZ = M + "/horizons"
SM = pd.read_csv(HZ + "/samples.csv", dtype={"Sample": str, "Patient": str}); n = len(SM)
D = pd.concat([pd.read_csv(f, dtype={"Sample": str}) for f in sorted(glob.glob(M + "/cv/preds/cfg_00_rep_*.csv"))]); P = D.pivot_table(index="Sample", columns="rep", values="train_prev").reindex(SM.Sample).values; s0 = P.mean(1)
time = SM.time.values.astype(float); ev = SM.event.values.astype(int); pat = SM.Patient.values; pre = SM.pre.astype(bool).values; ndbe = SM.ndbe.astype(bool).values
def Gm(tt, dd, at):
    u = np.unique(tt); nr = (tt[None, :] >= u[:, None]).sum(1); dc = ((tt[None, :] == u[:, None]) & (dd[None, :] == 0)).sum(1); sv = np.cumprod(1 - dc / nr); pv = np.concatenate([[1.0], sv[:-1]])
    k = np.searchsorted(u, at, side="left"); o = np.ones(len(at)); i = k < len(u); o[i] = pv[k[i]]; o[~i] = sv[-1]; return np.clip(o, 1e-12, None)
def auc(sc, tt, dd, t):
    case = (dd == 1) & (tt <= t); ctrl = ((dd == 0) & (tt >= t)) | ((dd == 1) & (tt > t))
    if not case.sum() or not ctrl.sum(): return np.nan
    w = 1 / Gm(tt, dd, tt[case]); cs = np.sort(sc[ctrl]); m = sc[case]; lo = np.searchsorted(cs, m, "left"); hi = np.searchsorted(cs, m, "right"); return float((w * (lo + 0.5 * (hi - lo))).sum() / (w.sum() * len(cs)))
out = {}
for nm, msk in (("pre", pre), ("pre_ndbe", pre & ndbe)):
    rows = np.where(msk)[0]; tt, dd, pp = time[rows], ev[rows], pat[rows]; up = np.unique(pp); ro = {q: np.where(pp == q)[0] for q in up}; rng = np.random.RandomState(0)
    boots = [np.concatenate([ro[up[i]] for i in rng.choice(len(up), len(up))]) for _ in range(2000)]; o = {}
    for t in (1.0, 3.0, 5.0):
        a = auc(s0[rows], tt, dd, t); b = np.array([auc(s0[rows][x], tt[x], dd[x], t) for x in boots]); b = b[np.isfinite(b)]
        o[str(int(t))] = {"auroc": round(a, 3), "ci95": [round(float(np.percentile(b, 2.5)), 3), round(float(np.percentile(b, 97.5)), 3)]}
    out[nm] = o
json.dump(out, open(T + "/results/paper_final/horizon_answers/q0_intercept_only.json", "w"), indent=1); print("HA INTERCEPT DONE", out)

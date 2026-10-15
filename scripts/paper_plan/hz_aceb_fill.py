"""Horizons H1, ACE-B columns (docs/paper_survival_horizons.md @ ee51db8): the exact code path that will fill the ACE-B 1/3/5-year cells from the
frozen package bundle models/killcoyne_frozen_pkg_v1 (L-CNV, L-IMG, L-EARLY, L-LATE; no frozen L-GRADE, L-GRADE+age+sex or L-INTER exists).
Usage:
  python hz_aceb_fill.py --cnv-tiles TILES.csv --cnv-arms ARMS.csv --img IMG.csv --labels LABELS.csv --out OUT.json   (real use, after unblinding)
  python hz_aceb_fill.py --dummy --out OUT.json   (end-to-end check: set C feature blocks as stand-in inputs, dummy labels from RandomState(0))
TILES/ARMS: Sample + raw (unscaled) package 5-Mb tiles / arms (kr_package.R tileSegments output); IMG: Sample + raw UNI2 slide-mean embedding;
LABELS: Sample, Patient, time (years), event (0/1), optional ndbe (0/1). Metric code copied from hz_metrics.py (IPCW/unweighted td-AUROC, C-indices,
patient bootstrap 2,000 RandomState(0), swap-permutation max-T over L-CNV, L-IMG, L-EARLY vs L-LATE)."""
import argparse, json, numpy as np, pandas as pd
B = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis/models/killcoyne_frozen_pkg_v1"; M = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis/feasibility/paper_plan/killcoyne_mm"
ap = argparse.ArgumentParser(); ap.add_argument("--cnv-tiles"); ap.add_argument("--cnv-arms"); ap.add_argument("--cnv-block"); ap.add_argument("--img"); ap.add_argument("--labels"); ap.add_argument("--out", required=True); ap.add_argument("--dummy", action="store_true"); ap.add_argument("--bundle", default=B)
a = ap.parse_args(); B = a.bundle
sc = pd.read_csv(f"{B}/cnv_scaling_setC.csv"); W = pd.read_csv(f"{B}/arm_tile_incidence.csv").set_index("arm"); imsc = pd.read_csv(f"{B}/image_scaling.csv").set_index("feature")
coef = {m: pd.read_csv(f"{B}/coef_L_{m}.csv").set_index("feature").coefficient for m in ("CNV", "IMG", "EARLY")}
def cnv_block(tiles, arms):
    """frozen preprocessing (model_info.json cnv_preprocessing): z-score tiles/arms with the set C constants, cx, arm adjustment"""
    ts = sc[sc.kind == "tile"].set_index("feature"); as_ = sc[sc.kind == "arm"].set_index("feature"); cx = sc[sc.kind == "cx"].iloc[0]
    S = (tiles[ts.index] - ts["mean"]) / ts["sd"].replace(0, 1); A = (arms[as_.index] - as_["mean"]) / as_["sd"].replace(0, 1)
    c = S.apply(lambda r: ((r >= r.mean() + 2 * r.std()) | (r <= r.mean() - 2 * r.std())).sum(), axis=1); cz = (c - cx["mean"]) / cx["sd"]
    X = pd.concat([S - A.values @ W.loc[A.columns, S.columns].values, A, cz.rename("cx")], axis=1)
    return X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
def prob(m, X):
    c = coef[m]; return 1 / (1 + np.exp(-(c["(Intercept)"] + X[c.index[1:]].values @ c.values[1:])))
def score(X_cnv, IMraw):
    Z = (IMraw[imsc.index] - imsc["mean"]) / imsc["sd"].replace(0, 1)
    out = pd.DataFrame({"L-CNV": prob("CNV", X_cnv), "L-IMG": prob("IMG", Z)}, index=X_cnv.index); out["L-EARLY"] = prob("EARLY", pd.concat([X_cnv, Z], axis=1)); out["L-LATE"] = (out["L-CNV"] + out["L-IMG"]) / 2
    return out
# ---------- metrics (copied from hz_metrics.py)
r3 = lambda x: None if x is None or not np.isfinite(x) else round(float(x), 3)
def Gminus(tt, dd, at):
    u = np.unique(tt); nrisk = (tt[None, :] >= u[:, None]).sum(1); dc = ((tt[None, :] == u[:, None]) & (dd[None, :] == 0)).sum(1)
    surv = np.cumprod(1 - dc / nrisk); prev = np.concatenate([[1.0], surv[:-1]]); k = np.searchsorted(u, at, side="left"); out = np.ones(len(at)); ins = k < len(u)
    out[ins] = prev[k[ins]]; out[~ins] = surv[-1]; return np.clip(out, 1e-12, None)
def labels(tt, dd, t): return (dd == 1) & (tt <= t), ((dd == 0) & (tt >= t)) | ((dd == 1) & (tt > t))
def tdauc(s, tt, dd, t, wtd=True):
    case, ctrl = labels(tt, dd, t)
    if case.sum() == 0 or ctrl.sum() == 0: return np.nan
    w = 1 / Gminus(tt, dd, tt[case]) if wtd else np.ones(case.sum()); cs = np.sort(s[ctrl]); m = s[case]; lo = np.searchsorted(cs, m, "left"); hi = np.searchsorted(cs, m, "right")
    return float((w * (lo + 0.5 * (hi - lo))).sum() / (w.sum() * len(cs)))
def cidx(s, tt, dd, uno):
    ii = np.where(dd == 1)[0]
    if len(ii) == 0: return np.nan
    w = (1 / Gminus(tt, dd, tt[ii]) ** 2) if uno else np.ones(len(ii)); comp = tt[ii][:, None] < tt[None, :]; conc = (s[ii][:, None] > s[None, :]) + 0.5 * (s[ii][:, None] == s[None, :])
    den = (w[:, None] * comp).sum(); return float((w[:, None] * comp * conc).sum() / den) if den > 0 else np.nan
def pctl(x): x = np.asarray(x, float); x = x[np.isfinite(x)]; return [r3(np.percentile(x, 2.5)), r3(np.percentile(x, 97.5))] if len(x) else [None, None]
def evaluate(S, L, NB=2000):
    ARMS = ["L-CNV", "L-IMG", "L-EARLY", "L-LATE"]; res = {}
    pops = {"all": np.ones(len(L), bool)}
    if "ndbe" in L: pops["ndbe"] = L.ndbe.astype(int).values == 1
    for pn, pm in pops.items():
        rows = np.where(pm)[0]; tt, dd, pp = L.time.values[rows].astype(float), L.event.values[rows].astype(int), L.Patient.values[rows]; up = np.unique(pp); ro = {q: np.where(pp == q)[0] for q in up}
        rng = np.random.RandomState(0); boots = [np.concatenate([ro[up[i]] for i in rng.choice(len(up), len(up))]) for _ in range(NB)]
        prs = np.random.RandomState(0); masks = []
        for _ in range(NB):
            sw = prs.rand(len(up)) < 0.5; mk = np.zeros(len(rows), bool)
            for i in np.where(sw)[0]: mk[ro[up[i]]] = True
            masks.append(mk)
        sc_ = {k: S[k].values[rows] for k in ARMS}; P = {"n_samples": int(len(rows)), "n_patients": int(len(up)), "horizons": {}, "cindex": {}}
        for t in (1.0, 3.0, 5.0):
            case, ctrl = labels(tt, dd, t); H = {"n_cases": int(case.sum()), "n_controls": int(ctrl.sum()), "arms": {}}
            if case.sum() and ctrl.sum():
                for wtd, key in ((True, "ipcw"), (False, "unweighted")):
                    obs = {k: tdauc(sc_[k], tt, dd, t, wtd) for k in ARMS}; bs = {k: np.array([tdauc(sc_[k][b], tt[b], dd[b], t, wtd) for b in boots]) for k in ARMS}
                    for k in ARMS:
                        o = H["arms"].setdefault(k, {}); o[key] = {"auroc": r3(obs[k]), "ci95": pctl(bs[k])}
                        for ref in ("L-CNV", "L-LATE"):
                            if k != ref: o[key][f"delta_vs_{ref}"] = {"delta": r3(obs[k] - obs[ref]), "ci95": pctl(bs[k] - bs[ref])}
                    if wtd:
                        oth = ["L-CNV", "L-IMG", "L-EARLY"]; pdl = {k: np.array([tdauc(np.where(mk, sc_["L-LATE"], sc_[k]), tt, dd, t) - tdauc(np.where(mk, sc_[k], sc_["L-LATE"]), tt, dd, t) for mk in masks]) for k in oth}
                        mx = np.max(np.abs(np.stack([pdl[k] for k in oth])), 0)
                        for k in oth:
                            d0 = obs[k] - obs["L-LATE"]; H["arms"][k]["ipcw"]["delta_vs_L-LATE"].update({"p_unadjusted": round(float((1 + (np.abs(pdl[k]) >= abs(d0) - 1e-12).sum()) / (NB + 1)), 4), "p_max_T": round(float((1 + (mx >= abs(d0) - 1e-12).sum()) / (NB + 1)), 4)})
            P["horizons"][str(int(t))] = H
        for k in ARMS: P["cindex"][k] = {nm: {"c": r3(cidx(sc_[k], tt, dd, u)), "ci95": pctl([cidx(sc_[k][b], tt[b], dd[b], u) for b in boots])} for nm, u in (("harrell", False), ("uno", True))}
        res[pn] = P
    return res
if a.dummy:
    Xc = pd.read_csv(f"{M}/cnv_pkg_C.csv", dtype={"sample_id": str}).set_index("sample_id"); IM = pd.read_csv(f"{M}/img_mean.csv", dtype={"Sample": str}).set_index("Sample").reindex(Xc.index)
    S = score(Xc, IM); C = pd.read_csv(f"{M}/set_C.csv", dtype=str).set_index("Sample").reindex(Xc.index)
    rng = np.random.RandomState(0); L = pd.DataFrame({"Sample": Xc.index, "Patient": C.Patient.values, "time": rng.uniform(0.1, 8, len(Xc)).round(2), "event": (rng.rand(len(Xc)) < 0.2).astype(int), "ndbe": (rng.rand(len(Xc)) < 0.7).astype(int)})
    from scipy.stats import rankdata
    y = (C.Status == "P").astype(int).values; r = rankdata(S["L-CNV"]); insample = float((r[y == 1].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (len(y) - y.sum())))
    out = {"mode": "dummy", "note": "set C feature blocks as stand-in inputs; dummy labels (uniform times 0.1-8 y, event 20%, ndbe 70%, RandomState(0)); no ACE-B data read",
           "check_insample_auroc_L_CNV_vs_bundle_0.995": r3(insample), "n": len(L), "result": evaluate(S, L.set_index("Sample").reindex(S.index).reset_index(), NB=200)}
else:
    tiles = pd.read_csv(a.cnv_tiles, dtype={"Sample": str}).set_index("Sample"); arms = pd.read_csv(a.cnv_arms, dtype={"Sample": str}).set_index("Sample")
    IM = pd.read_csv(a.img, dtype={"Sample": str}).set_index("Sample"); X = cnv_block(tiles, arms); S = score(X, IM.reindex(X.index))
    L = pd.read_csv(a.labels, dtype={"Sample": str, "Patient": str}).set_index("Sample").reindex(S.index).reset_index(); out = {"mode": "aceb", "n": len(L), "result": evaluate(S, L)}
json.dump(out, open(a.out, "w"), indent=1); print("HZ ACEB FILL DONE", out["mode"])

"""One-paper Fig. 3c-d, Fig. 2d and Extended Data calibration: aggregates computed on the cluster from stored held-out predictions (no new model).
Runs the set-up of tp_kcf_figs.py unchanged (everything before its smoke section: predictions, training-matched risk classes, endoscopy units,
Fig. 3b bins, patient draws), then adds:
  np_rate   share of non-progressor pre-event endoscopies with >= 1 high-risk sample, per score, at the same per-fold cut-offs used for the
            progressor bins of Killcoyne Fig. 3b (82% training-specificity high cut); grade comparator = >= 1 LGD sample. Progressor bins recomputed
            and asserted equal to two_papers_KCF/figs.json fig3b.
  ndbe_cn   Killcoyne Fig. 1d analogue: % of 5-Mb windows altered (|z| > 2) in NDBE samples (set C), H&E-high vs H&E-low (median class), copy
            number P vs NP, combined high vs low; differences with patient-bootstrap CI. Asserted equal to figs.json fig1d.
  spearman  within-class Spearman correlation of the artefact-free H&E score and the copy-number score (repeat-mean probabilities), 571 pre-event
            samples; all-tile L-IMG reproduced against review_round1/r9.json as a smoke check.
  calib     calibration on the 571 pre-event samples: per repeat, logistic recalibration of status on logit(p) (slope), averaged over repeats
            (P1A method; L-CNV must reproduce P1A's 0.565); decile curves of the repeat-mean probability (observed progressor share, Wilson CI).
CIs: patient bootstrap, 2,000 draws, RandomState(0), predictions fixed (tp_kcf_figs.pdraws over the 80 set-C patients; pre-event rows of the drawn
patients). Output: results/paper_final/one_paper/fig3_stats.json (aggregates only)."""
import os, json, numpy as np
HERE = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis/scripts/paper_plan"
os.environ["TASK"] = "none"
_s = open(HERE + "/tp_kcf_figs.py").read(); _cut = _s.index("# ================================================================== smoke")
exec(compile(_s[:_cut], HERE + "/tp_kcf_figs.py", "exec"))
from scipy.stats import spearmanr
OUT1 = T + "/results/paper_final/one_paper"; os.makedirs(OUT1, exist_ok=True)
FJ = json.load(open(T + "/results/paper_final/two_papers_KCF/figs.json"))
r4 = lambda v: None if v is None or not np.isfinite(v) else round(float(v), 4)
D = pdraws(NB, 0)
ci = lambda a: [r4(np.nanpercentile(a, 2.5)), r4(np.nanpercentile(a, 97.5))]
res = {"note": __doc__.split("Output:")[0].strip(), "n_draws": len(D)}
# ------------------------------------------------------------------ np_rate and progressor bins (per patient sums, summed over a draw)
def rate_parts(mask_e, hit):   # per patient: numerator (hits over endoscopies x repeats) and denominator
    num = np.zeros(NPAT); den = np.zeros(NPAT)
    for e in np.where(mask_e)[0]:
        h = hit[e]; num[E_pat[e]] += np.sum(h); den[E_pat[e]] += np.size(h)
    return num, den
def rate(num, den, c=None):
    if c is None: return num.sum() / den.sum()
    d = den[c].sum(); return num[c].sum() / d if d else np.nan
npm = (E_y == 0) & E_pre
out = {}
for m in MODS + ["GRADE"]:
    hit = (E_maxg >= 2)[:, None] if m == "GRADE" else (ECLS[m] == 2)
    nu, de = rate_parts(npm, hit); pt = rate(nu, de); bs = np.array([rate(nu, de, c) for c in D])
    o = {"np_endoscopies_with_high": {"share": r4(pt), "ci": ci(bs), "endoscopies": int(npm.sum()), "patients": int(len(set(E_pat[npm])))}}
    bins = {}
    for b in BIN_NAMES:
        if m == "GRADE" and b == "Endpoint": continue
        mk = E_BIN == b; nu2, de2 = rate_parts(mk, hit); p2 = rate(nu2, de2); bs2 = np.array([rate(nu2, de2, c) for c in D])
        bins[b] = {"share": r4(p2), "ci": ci(bs2), "endoscopies": int(mk.sum())}
        if m != "GRADE": assert abs(p2 - FJ["fig3b"][m][b]["share"]) < 6e-4, (m, b, p2, FJ["fig3b"][m][b]["share"])
    o["progressor_bins"] = bins; out[m] = o
res["np_rate"] = out
# ------------------------------------------------------------------ ndbe_cn: Fig. 1d analogue with CIs
ALT = (np.abs(Zw) > 2); per_s_alt = ALT.sum(1).astype(float); nwin = ALT.shape[1]; nd = PATH == "NDBE"
pidx_s = np.array([pidx[q] for q in pat])
def grp_parts(mask):
    num = np.bincount(pidx_s[mask], weights=per_s_alt[mask], minlength=NPAT); den = np.bincount(pidx_s[mask], minlength=NPAT).astype(float) * nwin
    return num, den
groups = {"CN": (nd & (y == 1), nd & (y == 0)), "HE": (nd & (MC["HE"] == 2), nd & (MC["HE"] == 0)), "COMB": (nd & (MC["COMB"] == 2), nd & (MC["COMB"] == 0))}
o = {}
for m, (hi, lo) in groups.items():
    a, b = grp_parts(hi), grp_parts(lo); ph, pl = 100 * rate(*a), 100 * rate(*b)
    bh = np.array([100 * rate(*a, c) for c in D]); bl = np.array([100 * rate(*b, c) for c in D])
    o[m] = {"high_or_P": {"n": int(hi.sum()), "pct": r4(ph), "ci": ci(bh)}, "low_or_NP": {"n": int(lo.sum()), "pct": r4(pl), "ci": ci(bl)},
            "diff": {"delta": r4(ph - pl), "ci": ci(bh - bl)}}
assert abs(o["HE"]["high_or_P"]["pct"] - FJ["fig1d"]["HE"]["H&E high-risk samples"]["pct_windows_altered"]) < 6e-3
assert abs(o["HE"]["low_or_NP"]["pct"] - FJ["fig1d"]["HE"]["H&E low-risk samples"]["pct_windows_altered"]) < 6e-3
res["ndbe_cn"] = o
# ------------------------------------------------------------------ spearman within class, pre-event
pre = np.where(PRE)[0]; S_pre = [S_of[i][PRE[S_of[i]]] for i in range(NPAT)]
def spear(a, b, rows): return spearmanr(a[rows], b[rows]).correlation if len(rows) > 2 else np.nan
o = {}
for nm, A_, B_ in (("clean", P["HE"].mean(1), P["CN"].mean(1)), ("all_tile", O["L-IMG"].mean(1), P["CN"].mean(1))):
    oo = {}
    for lab, yy in (("progressor_samples", 1), ("nonprogressor_samples", 0), ("all", None)):
        rows = pre if yy is None else pre[y[pre] == yy]; pt = spear(A_, B_, rows); bs = []
        for c in D:
            rr = np.concatenate([S_pre[i] for i in c]); rr = rr if yy is None else rr[y[rr] == yy]; bs.append(spear(A_, B_, rr))
        oo[lab] = {"spearman": r4(pt), "ci95": ci(np.array(bs)), "n": int(len(rows))}
    o[nm] = oo
r9 = json.load(open(T + "/results/paper_final/review_round1/r9.json"))["R9"]["their"]
o["smoke_all_tile_vs_r9"] = {k: [o["all_tile"][k]["spearman"], r9[k]["spearman"]] for k in ("progressor_samples", "nonprogressor_samples", "all")}
res["spearman"] = o
# ------------------------------------------------------------------ calibration, pre-event
def slope(p, yy):
    x = lg(p); b = np.zeros(2); X = np.c_[np.ones_like(x), x]
    for _ in range(50):
        mu = 1 / (1 + np.exp(-X @ b)); W = mu * (1 - mu); g = X.T @ (yy - mu); H = X.T @ (X * W[:, None])
        st = np.linalg.solve(H + 1e-9 * np.eye(2), g); b += st
        if np.abs(st).max() < 1e-10: break
    return b[1]
o = {}
for m in MODS:
    X = P[m]; pt = np.mean([slope(X[pre, j], y[pre]) for j in range(10)])
    bs = []
    for c in D:
        rr = np.concatenate([S_pre[i] for i in c]); bs.append(np.mean([slope(X[rr, j], y[rr]) for j in range(10)]))
    pm = X[pre].mean(1); yy = y[pre]; edges = np.quantile(pm, np.linspace(0, 1, 11)); dec = []
    for k in range(10):
        mk = (pm >= edges[k]) & ((pm < edges[k + 1]) if k < 9 else (pm <= edges[k + 1])); nn = int(mk.sum()); ph = yy[mk].mean(); z = 1.96
        den = 1 + z * z / nn; cen = (ph + z * z / (2 * nn)) / den; hw = z * np.sqrt(ph * (1 - ph) / nn + z * z / (4 * nn * nn)) / den
        dec.append({"pred_mean": r4(pm[mk].mean()), "obs": r4(ph), "wilson": [r4(cen - hw), r4(cen + hw)], "n": nn})
    o[m] = {"slope": r4(pt), "slope_ci95": ci(np.array(bs)), "deciles": dec, "n_samples": int(len(pre)), "progressor_share": r4(yy.mean())}
P1A = json.load(open(T + "/results/paper_final/two_papers_P1A/main.json"))["calibration"]
o["smoke_CN_vs_P1A"] = [o["CN"]["slope"], P1A["L-CNV::unweighted::slope"]["value"]]
res["calib"] = o
json.dump(res, open(OUT1 + "/fig3_stats.json", "w"), indent=1, default=lambda v: v.item() if hasattr(v, "item") else str(v))
print("OP FIG3 STATS DONE", json.dumps(res["spearman"]["smoke_all_tile_vs_r9"]), json.dumps(res["calib"]["smoke_CN_vs_P1A"]), flush=True)

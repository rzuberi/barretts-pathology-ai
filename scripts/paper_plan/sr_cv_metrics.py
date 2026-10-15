"""Scanner robustness, item 4 metrics (docs/paper_scanner_robustness.md @ abe96e9). D0: sr_cv.R repro_1 with UNI2-h vs stored cv/preds/cfg_02_rep_01
(max |diff| <= 1e-5). D1: fold-stratified per-sample AUROC (pairs within an outer fold of a repeat; mean over 10 repeats) of L-IMG and L-LATE (mean of
L-IMG and stored L-CNV, their / package) with the switch extractor vs the stored UNI2-h arms, all pre-event and NDBE pre-event; paired Δ with 2,000
patient-bootstrap draws (RandomState(0), draws without both classes redrawn, folds kept). D2: within-C13239-01 CV and design (b) C13239-01 -> C13210,
as tb_scanner_metrics.py. Aggregates -> results/paper_final/scanner_robustness/downstream.json (includes rendered table lines)."""
import os, sys, glob, json, numpy as np, pandas as pd
from scipy.stats import rankdata
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sr_common import ROW, OUT, M
EXT = json.load(open(ROW + "/decision.json"))["switch_to"]; KEY = {"H-optimus-0": "hoptimus0", "Virchow2": "virchow2"}[EXT]; REPS = list(range(1, 11)); NB = 2000
r3 = lambda x: None if x is None or not np.isfinite(x) else round(float(x), 3); pct = lambda a: [r3(np.nanpercentile(a, 2.5)), r3(np.nanpercentile(a, 97.5))]
SM = pd.read_csv(M + "/horizons/samples.csv", dtype={"Sample": str, "Patient": str}); ids = SM.Sample.values; n = len(ids); y = SM.y.values.astype(int); pat = SM.Patient.values
PRE = SM.pre.astype(bool).values; NDBE = SM.ndbe.astype(bool).values; POP = {"all pre-event": PRE, "NDBE pre-event": PRE & NDBE}
def piv(files):
    D = pd.concat([pd.read_csv(f, dtype={"Sample": str}) for f in files]); P = D.pivot_table(index="Sample", columns="rep", values="pred").reindex(ids)[REPS]
    F = D.pivot_table(index="Sample", columns="rep", values="fold").reindex(ids)[REPS]; assert P.notna().all().all(); return P.values, F.values.astype(int)
kvf = lambda c: sorted(glob.glob(f"{M}/cv/preds/cfg_{c:02d}_rep_*.csv"))
res = {"prespec": "abe96e9", "extractor": EXT}
# D0
a = pd.read_csv(f"{ROW}/cv/repro_1_uni2.csv", dtype={"Sample": str}).set_index("Sample"); b = pd.read_csv(f"{M}/cv/preds/cfg_02_rep_01.csv", dtype={"Sample": str}).set_index("Sample").reindex(a.index)
res["D0_repro_max_abs_diff"] = float(np.abs(a.pred - b.pred).max()); res["D0_folds_equal"] = bool((a.fold == b.fold).all()); res["D0_passed"] = bool(res["D0_repro_max_abs_diff"] <= 1e-5 and res["D0_folds_equal"])
# D1
A = {}; A["L-CNV their"], FO = piv(kvf(0)); A["L-CNV package"], F1 = piv(kvf(1)); A["L-IMG UNI2-h"], F2 = piv(kvf(2))
A[f"L-IMG {EXT}"], FN = piv(sorted(glob.glob(f"{ROW}/cv/cv_*_{KEY}.csv"))); assert (FO == F1).all() and (FO == F2).all() and (FO == FN).all(); res["D1_folds_equal_stored"] = True
for src in ("their", "package"):
    A[f"L-LATE {src} UNI2-h"] = (A[f"L-CNV {src}"] + A["L-IMG UNI2-h"]) / 2; A[f"L-LATE {src} {EXT}"] = (A[f"L-CNV {src}"] + A[f"L-IMG {EXT}"]) / 2
def fs(S, rows):
    yy = y[rows]; out = []
    for j in range(10):
        f = FO[rows, j]; x = S[rows, j]; num = den = 0.0
        for k in range(1, 11):
            mk = f == k; yk = yy[mk]; n1 = yk.sum(); n0 = len(yk) - n1
            if n1 == 0 or n0 == 0: continue
            rk = rankdata(x[mk]); num += rk[yk == 1].sum() - n1 * (n1 + 1) / 2; den += n1 * n0
        if den: out.append(num / den)
    return np.mean(out)
def boots(rows):
    pp = pat[rows]; up = np.unique(pp); ro = {q: rows[pp == q] for q in up}; rng = np.random.RandomState(0); out = []
    while len(out) < NB:
        bb = np.concatenate([ro[up[i]] for i in rng.choice(len(up), len(up))])
        if 0 < y[bb].sum() < len(bb): out.append(bb)
    return out
res["D1"] = {}
for pn, pm in POP.items():
    rows = np.where(pm)[0]; B = boots(rows); d = {"n_samples": int(len(rows)), "n_patients": int(len(np.unique(pat[rows]))), "arms": {}, "delta": {}}
    bs = {}
    for arm in ["L-IMG UNI2-h", f"L-IMG {EXT}", "L-LATE their UNI2-h", f"L-LATE their {EXT}", "L-LATE package UNI2-h", f"L-LATE package {EXT}"]:
        bs[arm] = np.array([fs(A[arm], bb) for bb in B]); d["arms"][arm] = {"auroc": r3(fs(A[arm], rows)), "ci95": pct(bs[arm])}
    for lab, new, old in (("L-IMG", f"L-IMG {EXT}", "L-IMG UNI2-h"), ("L-LATE their", f"L-LATE their {EXT}", "L-LATE their UNI2-h"), ("L-LATE package", f"L-LATE package {EXT}", "L-LATE package UNI2-h")):
        dd = bs[new] - bs[old]; d["delta"][lab] = {"delta": r3(d["arms"][new]["auroc"] - d["arms"][old]["auroc"]), "ci95": pct(dd), "p_two_sided": round(float(min(1.0, 2 * min(np.mean(dd <= 0), np.mean(dd >= 0)))), 4)}
    res["D1"][pn] = d; print(pn, json.dumps(d), flush=True)
# D2 (tb_scanner_metrics.py logic)
def thr_sens(cs, q): cs = np.sort(cs)[::-1]; return cs[int(np.ceil(q * len(cs))) - 1]
def c_star(ct, q): ct = np.sort(ct); return ct[int(np.ceil(q * len(ct))) - 1]
def auc(yy, s): r = rankdata(s); n1 = yy.sum(); n0 = len(yy) - n1; return (r[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if 0 < n1 < len(yy) else np.nan
def boots_df(df):
    up = np.unique(df.Patient); ro = {q: np.where(df.Patient.values == q)[0] for q in up}; rng = np.random.RandomState(0); out = []
    while len(out) < NB:
        bb = np.concatenate([ro[up[i]] for i in rng.choice(len(up), len(up))])
        if 0 < df.y.values[bb].sum() < len(bb): out.append(bb)
    return out
Wd = pd.concat([pd.read_csv(f, dtype={"Sample": str, "Patient": str}) for f in sorted(glob.glob(f"{ROW}/cv/within_s39_*_{KEY}.csv"))]); assert Wd.rep.nunique() == 10
P_ = Wd.pivot_table(index="Sample", columns="rep", values="pred"); F_ = Wd.pivot_table(index="Sample", columns="rep", values="fold").astype(int); S_ = Wd.drop_duplicates("Sample").set_index("Sample").reindex(P_.index)
def fsw(bb):
    yy = S_.y.values[bb]; out = []
    for j in range(10):
        f = F_.values[bb, j]; x = P_.values[bb, j]; num = den = 0.0
        for k in range(1, 11):
            mk = f == k; yk = yy[mk]; n1 = yk.sum(); n0 = len(yk) - n1
            if n1 == 0 or n0 == 0: continue
            rk = rankdata(x[mk]); num += rk[yk == 1].sum() - n1 * (n1 + 1) / 2; den += n1 * n0
        if den: out.append(num / den)
    return np.mean(out)
res["D2"] = {"within_s39": {"auroc_fold_stratified": r3(fsw(np.arange(len(S_)))), "ci95": pct([fsw(bb) for bb in boots_df(S_.reset_index())]), "n_samples": int(len(S_)), "n_patients": int(S_.Patient.nunique()), "reference_uni2": 0.841}}
X = pd.read_csv(f"{ROW}/cv/cross_b_s39_{KEY}.csv", dtype={"Sample": str, "Patient": str}); te = X[X.role == "test"].reset_index(drop=True); inn = X[X.role == "inner"]
yi = inn.y.values == 1; tl = thr_sens(inn.pred.values[yi], 0.95); cs = c_star(inn.pred.values[~yi], 0.90)
def bands(df): s = df.pred.values; bb = np.where(s < tl, 0, np.where(s > cs, 2, 1)) if tl <= cs else np.where(s > cs, 2, 0); return {nm: r3(np.mean(bb == g)) for g, nm in ((0, "low"), (1, "mid"), (2, "high"))}
res["D2"]["cross_b_s39_to_s10"] = {"auroc": r3(auc(te.y.values, te.pred.values)), "ci95": pct([auc(te.y.values[bb], te.pred.values[bb]) for bb in boots_df(te)]), "n_test": int(len(te)), "n_test_patients": int(te.Patient.nunique()),
                                    "n_train_patients": int(inn.Patient.nunique()), "bands_test_scanner": bands(te), "bands_training_scanner_inner_oof": bands(inn), "reference_uni2": {"auroc": 0.726, "bands_test": "50% / 30% / 21%", "bands_train_inner": "18% / 61% / 20%"}}
f3 = lambda x: f"{x:.3f}"; ci = lambda c: f" [{c[0]:.3f}, {c[1]:.3f}]"; cis = lambda c: f" [{c[0]:+.3f}, {c[1]:+.3f}]"; pc = lambda b: " / ".join(f"{100 * b[k]:.0f}%" for k in ("low", "mid", "high"))
L = [f"D0: `sr_cv.R` with the UNI2-h matrix reproduces stored `cfg_02_rep_01` predictions: max |difference| {res['D0_repro_max_abs_diff']:.1e}, folds equal: {'yes' if res['D0_folds_equal'] else 'NO'} ({'passed' if res['D0_passed'] else 'FAILED'}).", "",
     f"**D1. Fold-stratified per-sample AUROC, stratified patient-grouped 10-fold × 10 CV (stored folds), {EXT} vs UNI2-h.**", "",
     f"| Population | Arm | UNI2-h [95% CI] | {EXT} [95% CI] | Δ [95% CI] | p (two-sided) |", "|---|---|---|---|---|---|"]
for pn, d in res["D1"].items():
    for lab in ("L-IMG", "L-LATE their", "L-LATE package"):
        o = d["arms"][f"{lab} UNI2-h"]; nw = d["arms"][f"{lab} {EXT}"]; dl = d["delta"][lab]
        L.append(f"| {pn} ({d['n_samples']} samples, {d['n_patients']} patients) | {lab} | {f3(o['auroc'])}{ci(o['ci95'])} | {f3(nw['auroc'])}{ci(nw['ci95'])} | {dl['delta']:+.3f}{cis(dl['ci95'])} | {dl['p_two_sided']} |")
w = res["D2"]["within_s39"]; c = res["D2"]["cross_b_s39_to_s10"]
L += ["", f"**D2. Scanner transfer with {EXT}** (L-IMG refitted; pre-event samples).", "", "| Design | AUROC [95% CI] | UNI2-h reference | Test bands low / middle / high (training-scanner inner OOF) |", "|---|---|---|---|",
      f"| Within C13239-01, fold-stratified, {w['n_samples']} samples, {w['n_patients']} patients | {f3(w['auroc_fold_stratified'])}{ci(w['ci95'])} | 0.841 | — |",
      f"| (b) C13239-01 → C13210, {c['n_test']} test samples, {c['n_test_patients']} patients; {c['n_train_patients']} training patients | {f3(c['auroc'])}{ci(c['ci95'])} | 0.726; bands 50% / 30% / 21% (18% / 61% / 20%) | {pc(c['bands_test_scanner'])} ({pc(c['bands_training_scanner_inner_oof'])}) |"]
res["lines"] = L; json.dump(res, open(f"{OUT}/downstream.json", "w"), indent=1); print("\n".join(L)); print("SR CV METRICS DONE", flush=True)

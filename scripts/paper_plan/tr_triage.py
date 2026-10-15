"""H&E-first triage (docs/paper_triage.md @ f5944d2). TASK = <src>_<pop>, src in their | pkg, pop in pre | pre_ndbe.
Stored outer predictions (kv_cv.R d69de24; hz_fit.R outer) and stored inner out-of-fold predictions (hz_fit.R inner) only; nothing refitted.
Aggregates -> results/paper_final/triage/<TASK>.json; per-sample bands, calls and missed-patient lists -> feasibility/paper_plan/killcoyne_mm/triage/ (cluster only)."""
import os, glob, json, numpy as np, pandas as pd
from scipy.stats import rankdata
T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"; M = T + "/feasibility/paper_plan/killcoyne_mm"; HZ = M + "/horizons"; OUT = T + "/results/paper_final/triage"; ROW = M + "/triage"
os.makedirs(OUT, exist_ok=True); os.makedirs(ROW, exist_ok=True)
TASK = os.environ["TASK"]; src, pop = TASK.split("_", 1); NB = 2000; REPS = list(range(1, 11)); r3 = lambda x: None if x is None or not np.isfinite(x) else round(float(x), 3)
SM = pd.read_csv(HZ + "/samples.csv", dtype={"Sample": str, "Patient": str}); n = len(SM); ids = SM.Sample.values; ix = pd.Series(np.arange(n), index=ids)
y = SM.y.values.astype(int); pat = SM.Patient.values; popm = SM.pre.astype(bool).values & (True if pop == "pre" else SM.ndbe.astype(bool).values); rows = np.where(popm)[0]
def outer(files):
    D = pd.concat([pd.read_csv(f, dtype={"Sample": str}) for f in files])
    return D.pivot_table(index="Sample", columns="rep", values="pred").reindex(ids)[REPS].values, D.pivot_table(index="Sample", columns="rep", values="fold").reindex(ids)[REPS].values.astype(int)
def inner(files):
    D = pd.concat([pd.read_csv(f, dtype={"Sample": str}) for f in files]); assert D.groupby(["rep", "fold"]).ngroups == 100
    return {(r, k): pd.Series(d.pred.values, index=ix[d.Sample].values) for (r, k), d in D.groupby(["rep", "fold"])}
kvf = lambda c: sorted(glob.glob(f"{M}/cv/preds/cfg_{c:02d}_rep_*.csv"))
O, I = {}, {}
O["L-CNV"], FO = outer(kvf(0 if src == "their" else 1)); O["L-IMG"], F2 = outer(kvf(2)); O["L-EARLY"], F3 = outer(kvf(3 if src == "their" else 4)); O["L-INTER"], F4 = outer(sorted(glob.glob(f"{HZ}/outer/inter_{src}_rep_*.csv")))
assert (FO == F2).all() and (FO == F3).all() and (FO == F4).all(); O["L-LATE"] = (O["L-CNV"] + O["L-IMG"]) / 2
I["L-CNV"] = inner(sorted(glob.glob(f"{HZ}/inner/cnv_{src}_rep_*_fold_*.csv"))); I["L-IMG"] = inner(sorted(glob.glob(f"{HZ}/inner/img_rep_*_fold_*.csv")))
I["L-LATE"] = {k: (I["L-CNV"][k] + I["L-IMG"][k].reindex(I["L-CNV"][k].index)) / 2 for k in I["L-CNV"]}
for k in I["L-CNV"]: assert I["L-IMG"][k].index.sort_values().equals(I["L-CNV"][k].index.sort_values())
def thr_sens(cs, q):   # largest threshold keeping >= q sensitivity (score >= thr positive); ha_q2.py:49 convention
    cs = np.sort(cs)[::-1]; return cs[int(np.ceil(q * len(cs))) - 1] if len(cs) else np.inf
# ---------------------------------------------------------------- triage and comparators, per repeat x fold
band = np.full((n, 10), -1); seq_call = {m: np.zeros((n, 10), bool) for m in ("L-LATE", "L-CNV")}; comp = {a: np.zeros((n, 10), bool) for a in ("A", "B", "C")}
cut = {"t_low": [], "c_star": [], "t_seq_L-LATE": [], "t_seq_L-CNV": []}; nomid = 0; fallback = {"L-LATE": 0, "L-CNV": 0}; COMPM = {"A": "L-CNV", "B": "L-LATE", "C": "L-IMG"}
for j, r in enumerate(REPS):
    for k in range(1, 11):
        si = I["L-IMG"][(r, k)]; tri = si.index.values[popm[si.index.values]]; s_tr = si.reindex(tri).values; yc = y[tri] == 1
        t_low = thr_sens(s_tr[yc], 0.95); c_star = np.sort(s_tr[~yc])[int(np.ceil(0.9 * (~yc).sum())) - 1]; cut["t_low"].append(t_low); cut["c_star"].append(c_star)
        te = FO[:, j] == k; s = O["L-IMG"][:, j]
        if t_low > c_star: nomid += 1; band[te, j] = np.where(s[te] > c_star, 2, 0); mid_tr = np.zeros(len(tri), bool)
        else: band[te, j] = np.where(s[te] < t_low, 0, np.where(s[te] > c_star, 2, 1)); mid_tr = (s_tr >= t_low) & (s_tr <= c_star)
        for m in ("L-LATE", "L-CNV"):
            mi = I[m][(r, k)].reindex(tri).values; cm = mi[mid_tr & yc]
            if len(cm) >= 5: ts = thr_sens(cm, 0.8)
            else: ts = thr_sens(mi[yc], 0.8); fallback[m] += 1
            cut[f"t_seq_{m}"].append(ts); seq_call[m][te, j] = np.where(band[te, j] == 1, O[m][te, j] >= ts, band[te, j] == 2)
        for a, m in COMPM.items(): comp[a][te, j] = O[m][te, j] >= thr_sens(I[m][(r, k)].reindex(tri).values[yc], 0.8)
CALL = {"triage_LATE": seq_call["L-LATE"].sum(1) >= 6, "triage_CNV": seq_call["L-CNV"].sum(1) >= 6, **{a: comp[a].sum(1) >= 6 for a in comp}}
assert (band[rows] >= 0).all()
# ---------------------------------------------------------------- statistics on a (bootstrap) row set
MODELS = ["L-IMG", "L-CNV", "L-EARLY", "L-INTER", "L-LATE"]
def fs_auc(P, b, mask=None):
    out = []
    for j in range(10):
        f = FO[b, j]; x = P[b, j]; l = y[b]; ok = np.ones(len(b), bool) if mask is None else mask[b, j]; num = den = 0.0
        for k in range(1, 11):
            mm = ok & (f == k); yk = l[mm]; n1 = yk.sum(); n0 = len(yk) - n1
            if n1 == 0 or n0 == 0: continue
            rk = rankdata(x[mm]); num += rk[yk == 1].sum() - n1 * (n1 + 1) / 2; den += n1 * n0
        if den: out.append(num / den)
    return float(np.mean(out)) if out else np.nan
MID = band == 1
def stats(b):
    o = {f"auc_{m}": fs_auc(O[m], b) for m in MODELS}
    for m in ("L-CNV", "L-IMG", "L-LATE"): o[f"midauc_{m}"] = fs_auc(O[m], b, MID)
    yb = y[b]
    for s, c in CALL.items(): o[f"sens_{s}"] = c[b][yb == 1].mean(); o[f"spec_{s}"] = 1 - c[b][yb == 0].mean()
    for g, nm in ((0, "low"), (1, "mid"), (2, "high")):
        bb = band[b] == g; o[f"share_{nm}"] = bb.mean(0).mean(); o[f"share_{nm}_P"] = bb[yb == 1].mean(0).mean(); o[f"share_{nm}_NP"] = bb[yb == 0].mean(0).mean()
    return o
pp = pat[rows]; up = np.unique(pp); ro = {q: rows[pp == q] for q in up}; rng = np.random.RandomState(0); boots = []
while len(boots) < NB:
    b = np.concatenate([ro[up[i]] for i in rng.choice(len(up), len(up))])
    if 0 < y[b].sum() < len(b): boots.append(b)
obs = stats(rows); BS = [stats(b) for b in boots]; arr = lambda k: np.array([x[k] for x in BS], float)
pct = lambda a: [r3(np.nanpercentile(a, 2.5)), r3(np.nanpercentile(a, 97.5))]; p2 = lambda a: round(float(min(1.0, 2 * min(np.nanmean(a <= 0), np.nanmean(a >= 0)))), 4)
res = {"task": TASK, "prespec": "f5944d2", "n_samples": int(len(rows)), "n_patients": int(len(up)), "n_progressor_samples": int(y[rows].sum()), "n_progressor_patients": int(len(set(pp[y[rows] == 1]))), "draws": NB}
# Section 1
S1 = {"auroc": {m: {"value": r3(obs[f"auc_{m}"]), "ci95": pct(arr(f"auc_{m}"))} for m in MODELS}}
d = arr("auc_L-IMG") - arr("auc_L-CNV"); d0 = obs["auc_L-IMG"] - obs["auc_L-CNV"]
S1["img_vs_cnv"] = {"delta": r3(d0), "ci95": pct(d), "one_sided_95_lower": r3(np.nanpercentile(d, 5)), "margin": -0.05, "non_inferior": bool(np.nanpercentile(d, 5) > -0.05), "p_two_sided": p2(d)}
fam = ["L-EARLY", "L-INTER", "L-LATE"]; D0 = np.array([obs[f"auc_{m}"] - obs["auc_L-IMG"] for m in fam]); DB = np.column_stack([arr(f"auc_{m}") - arr("auc_L-IMG") for m in fam])
sd = np.nanstd(DB, 0, ddof=1); Tb = np.nanmax(np.abs((DB - D0) / sd), 1); To = np.abs(D0 / sd)
S1["fusion_vs_img"] = {m: {"delta": r3(D0[i]), "ci95": pct(DB[:, i]), "p_unadjusted": p2(DB[:, i]), "p_maxT_adjusted": round(float(np.mean(Tb >= To[i])), 4)} for i, m in enumerate(fam)}
res["section1"] = S1
# Section 2/3: strategies
ST = {}
for s, c in CALL.items():
    cr = c[rows]; yr = y[rows] == 1; missed = rows[yr & ~cr]; mp = pd.Series(~cr[yr], index=pat[rows][yr])
    ST[s] = {"sensitivity": {"value": r3(obs[f"sens_{s}"]), "ci95": pct(arr(f"sens_{s}"))}, "specificity": {"value": r3(obs[f"spec_{s}"]), "ci95": pct(arr(f"spec_{s}"))},
             "missed_progressor_samples": int(len(missed)), "progressor_patients_with_any_missed": int(mp.groupby(level=0).any().sum()), "progressor_patients_all_missed": int(mp.groupby(level=0).all().sum()),
             "positive_calls": int(cr.sum())}
    for ref in ("A", "B"):
        if s == ref: continue
        for met in ("sens", "spec"):
            dd = arr(f"{met}_{s}") - arr(f"{met}_{ref}"); ST[s].setdefault("vs_" + ref, {})[met] = {"delta": r3(obs[f"{met}_{s}"] - obs[f"{met}_{ref}"]), "ci95": pct(dd), "one_sided_95_lower": r3(np.nanpercentile(dd, 5)), "p_two_sided": p2(dd)}
res["strategies"] = ST
BAND = {nm: {"all": {"value": r3(obs[f"share_{nm}"]), "ci95": pct(arr(f"share_{nm}"))}, "progressor": {"value": r3(obs[f"share_{nm}_P"]), "ci95": pct(arr(f"share_{nm}_P"))},
             "non_progressor": {"value": r3(obs[f"share_{nm}_NP"]), "ci95": pct(arr(f"share_{nm}_NP"))}} for nm in ("low", "mid", "high")}
res["bands"] = BAND; res["folds"] = {"n": 100, "no_middle_band": nomid, "t_seq_fallback": fallback, "cutoffs": {k: {"median": r3(np.median(v)), "range": [r3(min(v)), r3(max(v))]} for k, v in cut.items()}}
dl = arr("sens_triage_LATE") - arr("sens_B"); share_mid = obs["share_mid"]
res["primary_criterion"] = {"applies_to": "their_pre", "delta_sens_vs_B": r3(obs["sens_triage_LATE"] - obs["sens_B"]), "one_sided_95_lower": r3(np.nanpercentile(dl, 5)), "noninferior_sens": bool(np.nanpercentile(dl, 5) > -0.05),
                            "share_sequenced": r3(share_mid), "share_below_0.70": bool(share_mid < 0.70), "met": bool(np.nanpercentile(dl, 5) > -0.05 and share_mid < 0.70)}
md = arr("midauc_L-CNV") - arr("midauc_L-IMG"); ml = arr("midauc_L-LATE") - arr("midauc_L-IMG")
res["middle_band_auroc"] = {m: {"value": r3(obs[f"midauc_{m}"]), "ci95": pct(arr(f"midauc_{m}"))} for m in ("L-CNV", "L-IMG", "L-LATE")}
res["middle_band_auroc"]["CNV_minus_IMG"] = {"delta": r3(obs["midauc_L-CNV"] - obs["midauc_L-IMG"]), "ci95": pct(md), "p_unadjusted": p2(md)}
res["middle_band_auroc"]["LATE_minus_IMG"] = {"delta": r3(obs["midauc_L-LATE"] - obs["midauc_L-IMG"]), "ci95": pct(ml), "p_unadjusted": p2(ml)}
# characteristics (pooled sample x repeat, weight 1/10)
cxp = pd.read_csv(M + "/cnv_pkg_C.csv", usecols=["sample_id", "cx"], dtype={"sample_id": str}).set_index("sample_id").cx.reindex(ids).values
scl = pd.read_csv(T + "/models/killcoyne_frozen_pkg_v1/cnv_scaling_setC.csv"); cr_ = scl[scl.kind == "cx"].iloc[0]; cx_raw = cxp * cr_["sd"] + cr_["mean"]
sd_ = pd.read_csv(HZ + "/slide_desc.csv", dtype={"Sample": str}).set_index("Sample").reindex(ids); tiles = sd_.tissue_tiles.values.astype(float); scan = sd_.scanner.values; grade = SM.Pathology.values
CH = {}
for g, nm in ((0, "low"), (1, "mid"), (2, "high")):
    ri, cj = np.where(band[rows] == g); ii = rows[ri]; w = len(ii) / 10
    if not len(ii): CH[nm] = None; continue
    CH[nm] = {"sample_repeats_div10": r3(w), "cx_raw_mean": r3(cx_raw[ii].mean()), "cx_raw_sd": r3(cx_raw[ii].std(ddof=1)), "tiles_mean": r3(tiles[ii].mean()), "tiles_sd": r3(tiles[ii].std(ddof=1)), "tiles_median": r3(np.median(tiles[ii])),
              "grade_share": {gname: r3(np.mean(grade[ii] == gname)) for gname in ("NDBE", "ID", "LGD")}, "scanner_C13210_share": r3(np.mean(scan[ii] == "C13210")), "progressor_share": r3(np.mean(y[ii]))}
res["band_characteristics"] = CH
# prevalence projection
PREV = {}
for pi in (0.02, 0.05, 0.10, float(y[rows].mean())):
    key = "observed" if pi > 0.2 else f"{pi:.2f}"; PREV[key] = {"prevalence": r3(pi)}
    for nm in ("low", "mid", "high"):
        a = pi * arr(f"share_{nm}_P") + (1 - pi) * arr(f"share_{nm}_NP"); PREV[key][nm] = {"value": r3(pi * obs[f"share_{nm}_P"] + (1 - pi) * obs[f"share_{nm}_NP"]), "ci95": pct(a)}
res["prevalence_projection"] = PREV
res["curve"] = {"prevalence": [r3(p) for p in np.round(np.arange(0.005, 0.3001, 0.005), 3)], "share_mid": [r3(p * obs["share_mid_P"] + (1 - p) * obs["share_mid_NP"]) for p in np.arange(0.005, 0.3001, 0.005)],
                "share_mid_lo": [r3(np.nanpercentile(p * arr("share_mid_P") + (1 - p) * arr("share_mid_NP"), 2.5)) for p in np.arange(0.005, 0.3001, 0.005)],
                "share_mid_hi": [r3(np.nanpercentile(p * arr("share_mid_P") + (1 - p) * arr("share_mid_NP"), 97.5)) for p in np.arange(0.005, 0.3001, 0.005)]}
# flow (modal band, ties -> middle; final >= 6/10 call)
cnt = np.stack([(band[rows] == g).sum(1) for g in (0, 1, 2)], 1); mx = cnt.max(1, keepdims=True); modal = np.where(cnt[:, 1] == mx[:, 0], 1, cnt.argmax(1))
FL = {"modal_band_equals_all_repeats_share": r3(np.mean(mx[:, 0] == 10))}
for s in ("triage_LATE", "triage_CNV"):
    c = CALL[s][rows]; yr = y[rows]
    FL[s] = {nm: {"progressor": int(((modal == g) & (yr == 1)).sum()), "non_progressor": int(((modal == g) & (yr == 0)).sum()),
                  "called_positive": {"progressor": int(((modal == g) & (yr == 1) & c).sum()), "non_progressor": int(((modal == g) & (yr == 0) & c).sum())}} for g, nm in ((0, "low"), (1, "mid"), (2, "high"))}
res["flow"] = FL
# row level (cluster only)
R = pd.DataFrame({"Sample": ids[rows], "Patient": pat[rows], "y": y[rows], "modal_band": modal, **{f"band_r{j + 1}": band[rows, j] for j in range(10)}, **{f"call_{s}": CALL[s][rows].astype(int) for s in CALL}})
R.to_csv(f"{ROW}/{TASK}_rows.csv", index=False)
json.dump(res, open(f"{OUT}/{TASK}.json", "w"), indent=1); print("TR DONE", TASK, json.dumps(res["primary_criterion"]), flush=True)

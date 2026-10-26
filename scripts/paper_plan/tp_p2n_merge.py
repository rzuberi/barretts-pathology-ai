"""Two papers, agent P2N (docs/two_papers_P2N.md): reproduction check and merge of the selection-aware permutation null (tp_p2n_search.py outputs).
TASK = check0 (SHUF 0: known numbers; refit s0000 equals the stored predictions) | merge (null distributions, p-values) -> results/paper_final/two_papers_P2N/<TASK>.json.
Aggregates only (no identifiers, no dates). Selection statistic per label set: among all searched cells, the one with the fewest held-out non-progressor patients flagged
subject to progressor patients caught >= the R49 gate's minus a tolerance (primary 1.0; also 0, 0.5, 2.0); statistic = its NP flagged minus R49's (same labels).
p = (1 + #{null statistic <= real statistic}) / (B + 1)."""
import os, sys, json, glob, numpy as np, pandas as pd
T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"; M = T + "/feasibility/paper_plan/killcoyne_mm"; XD = M + "/two_papers_P2N"; LIMG = os.environ.get("LIMG", "stored"); SD_ = XD + ("/search" if LIMG == "stored" else "/search_clean"); SFX = "" if LIMG == "stored" else "_clean"   # clean: ART's clean-tile L-IMG (22 Oct)
OUT = T + "/results/paper_final/two_papers_P2N"; os.makedirs(OUT, exist_ok=True); TASK = os.environ["TASK"]
r2 = lambda x: None if x is None or not np.isfinite(x) else round(float(x), 2); r3 = lambda x: None if x is None or not np.isfinite(x) else round(float(x), 3); r4 = lambda x: round(float(x), 4)
def jd(o, name): json.dump(o, open(f"{OUT}/{name}.json", "w"), indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x)); print("TPP2N DONE", name, flush=True)
def load(f):
    z = np.load(f, allow_pickle=False); v = z["vals"].astype(float).mean(2)   # (cells, 3): caught, NP flagged, samples/patient (means over the 10 repeats)
    return list(z["names"]), v, json.loads(str(z["times"]))
REF = "st|ref|R49"
NAMED = {"R49 gate": REF, "P1 90%": "st|ref|P1_90", "memory (A1 EW gate score)": "st|r9|A1:gate_al:ew|sel_np", "memory + certificate (A12)": "st|r9|A12|sel_np",
         "final rule (A12 + check, cn90)": "st|mem|A12 + ps_oac k2 cn90", "final rule, in-fold bar": "st|E7bar|A12 + check in-fold bar slack 0",
         "mean-logit + memory (no sequencing)": "ml|r9|A1:img:ew|sel_np", "mean-logit gate + memory": "ml|r9|A1:gate_al:ew|sel_np", "mean-logit full stack": "ml|mem|A12 + ps_oac k2 cn90",
         "R49 + check (E2 cell)": "st|E2fixed|R49hm veto ps_oac k2 cn90", "memory + check": "st|mem|A1 ew + ps_oac k2 cn90", "R49 re-tuned (C1)": "st|r9|C1|sel_np",
         "mean-logit R49 rebuilt": "ml|ref|R49 rebuilt", "mean-logit P1 90%": "ml|ref|P1_90", "17p-only check": "st|E8arms|A12 + check 17p k2 cn90",
         "mean-logit light (gate + memory + certificate)": "ml|r9|A12|sel_np"}
EXPECT = {"R49 gate": (31.1, 31.8, 1.94), "P1 90%": (31.1, 36.0, 0.0), "memory (A1 EW gate score)": (30.7, 23.6, 1.94), "memory + certificate (A12)": (30.5, 21.5, 1.94),
          "final rule (A12 + check, cn90)": (30.5, 17.2, 3.82), "final rule, in-fold bar": (30.4, 16.4, 3.82), "mean-logit + memory (no sequencing)": (31.0, 26.6, 0.0),
          "mean-logit gate + memory": (30.5, 22.6, 1.43), "mean-logit full stack": (30.3, 16.2, 3.12), "R49 + check (E2 cell)": (31.1, 26.6, 4.35), "memory + check": (30.7, 20.4, 4.05),
          "R49 re-tuned (C1)": (30.6, 29.4, 1.87), "mean-logit R49 rebuilt": (30.9, 31.2, 1.43), "mean-logit P1 90%": (31.2, 32.2, 0.0), "17p-only check": (30.5, 19.7, 3.82)}
if LIMG == "clean":   # agent CLN's clean-tile numbers (docs/two_papers_CLN.md, smoke and table 1)
    EXPECT = {"R49 gate": (31.3, 32.8, 2.16), "P1 90%": (31.3, 37.8, 0.0), "memory (A1 EW gate score)": (30.7, 25.1, 2.16), "memory + certificate (A12)": (30.5, 23.5, 2.16),
              "final rule (A12 + check, cn90)": (30.3, 19.8, 4.06), "final rule, in-fold bar": (30.2, 18.6, 4.06), "mean-logit + memory (no sequencing)": (31.1, 29.7, 0.0),
              "mean-logit light (gate + memory + certificate)": (30.3, 21.9, 1.70), "mean-logit full stack": (30.2, 17.9, 3.41)}
else: EXPECT["mean-logit light (gate + memory + certificate)"] = (30.3, 20.1, 1.43)
TOLS = (0.0, 0.5, 1.0, 2.0); PRIMARY = 1.0
def family_of(nm):
    p = nm.split("|"); return p[0] + "|" + p[1] + ("|" + p[2] if p[1] == "r9" else "")
def stats(names, v, core=False):
    """selection statistics for one label set (core: the E3 extractor-variant cells 'e3x:' left out, i.e. the same family set as the full-refit null)"""
    if core: keep = [i for i, x in enumerate(names) if not x.startswith("e3x:")]; names = [names[i] for i in keep]; v = v[keep]
    i0 = names.index(REF); c0, n0 = v[i0, 0], v[i0, 1]; dC = v[:, 0] - c0; dN = v[:, 1] - n0; o = {"R49": [r3(c0), r3(n0)]}
    for tol in TOLS:
        ok = np.where(dC >= -tol - 1e-9)[0]; b = ok[np.lexsort((-dC[ok], dN[ok]))[0]]   # fewest NP flagged; ties: more caught
        o[f"tol{tol}"] = {"dNP": float(dN[b]), "dC": float(dC[b]), "cell": names[b], "seq": float(v[b, 2])}
    z = np.where(v[:, 2] < 1e-9)[0]   # no-sequencing tier: cells that sequence nothing (selection restricted to them)
    for tol in (0.5, 1.0):
        ok = z[dC[z] >= -tol - 1e-9]; b = ok[np.lexsort((-dC[ok], dN[ok]))[0]]; o[f"tier0_tol{tol}"] = {"dNP": float(dN[b]), "dC": float(dC[b]), "cell": names[b], "seq": 0.0}
    ex = dN / 43 - dC / 32; b = int(np.argmin(ex)); o["excess"] = {"value": float(ex[b]), "dNP": float(dN[b]), "dC": float(dC[b]), "cell": names[b]}
    o["named"] = {k: {"dNP": float(dN[names.index(c)]), "dC": float(dC[names.index(c)]), "excess": float(ex[names.index(c)])} for k, c in NAMED.items() if c in names}
    return o
if TASK == "check0":
    res = {}
    for mode in ("b", "a"):
        f = f"{SD_}/{mode}_free_s0000.npz"
        if not os.path.exists(f): res[mode] = "missing"; continue
        names, v, times = load(f); obs = {k: [r2(v[names.index(c), 0]), r2(v[names.index(c), 1]), r2(v[names.index(c), 2])] for k, c in NAMED.items() if c in names}
        bad = {k: (obs[k], EXPECT[k]) for k in EXPECT if k in obs and (abs(obs[k][0] - EXPECT[k][0]) > 0.051 or abs(obs[k][1] - EXPECT[k][1]) > 0.051 or abs(obs[k][2] - EXPECT[k][2]) > 0.006)}
        fams = pd.Series([family_of(x) for x in names]).value_counts()
        res[mode] = {"observed": obs, "expected": {k: list(v_) for k, v_ in EXPECT.items()}, "mismatches": bad, "reproduced": not bad, "n_cells": len(names), "cells_by_source": pd.Series([x.split("|")[0] + "|" + x.split("|")[1] for x in names]).value_counts().to_dict(),
                     "n_families": int(len(fams)), "seconds": times, "real_selection": stats(names, v)}
    # refit s0000 must equal the stored predictions (outer cfg_00 / cfg_02 and horizons inner files)
    d = f"{XD}/refit/s0000"; chk = {}
    if os.path.isdir(d):
        sys.path.insert(0, T + "/scripts/paper_plan"); import tb_common as TC
        O, I, FO = TC.load("their")
        if LIMG == "clean":   # reference = ART's clean refit (refit/clean/{outer,inner}_rep_XX.csv)
            ad = M + "/two_papers_ART/refit/clean"; Da = pd.concat([pd.read_csv(f, dtype={"Sample": str}) for f in sorted(glob.glob(ad + "/outer_rep_*.csv"))])
            O["L-IMG"] = Da.pivot_table(index="Sample", columns="rep", values="pred").reindex(TC.ids)[TC.REPS].values
            Dia = pd.concat([pd.read_csv(f, dtype={"Sample": str}) for f in sorted(glob.glob(ad + "/inner_rep_*.csv"))]); I["L-IMG"] = {(r, k): pd.Series(dd.pred.values, index=TC.ix[dd.Sample].values) for (r, k), dd in Dia.groupby(["rep", "fold"])}
        for mdl, key in (("cnv", "L-CNV"), ("img" if LIMG == "stored" else "imgclean", "L-IMG")):
            fo = sorted(glob.glob(f"{d}/{mdl}_outer_rep_*.csv.gz")); fi = sorted(glob.glob(f"{d}/{mdl}_inner_rep_*.csv.gz"))
            if len(fo) < 10 or len(fi) < 10: chk[key] = f"incomplete ({len(fo)}, {len(fi)})"; continue
            D = pd.concat([pd.read_csv(f, dtype={"Sample": str}) for f in fo]); Oo = D.pivot_table(index="Sample", columns="rep", values="pred").reindex(TC.ids)[TC.REPS].values
            Fo = D.pivot_table(index="Sample", columns="rep", values="fold").reindex(TC.ids)[TC.REPS].values.astype(int)
            Di = pd.concat([pd.read_csv(f, dtype={"Sample": str}) for f in fi]); mx = 0.0
            for (r, k), dd in Di.groupby(["rep", "fold"]): s = pd.Series(dd.pred.values, index=TC.ix[dd.Sample].values); mx = max(mx, float(np.abs(s - I[key][(r, k)].reindex(s.index)).max()))
            chk[key] = {"outer_max_abs_diff": float(np.nanmax(np.abs(Oo - O[key]))), "inner_max_abs_diff": mx, "folds_equal": bool((Fo == FO).all())}
    res["refit_s0000_vs_stored"] = chk; res["limg"] = LIMG; jd(res, "check0" + SFX)
elif TASK == "merge":
    res = {"selection_criterion": "fewest held-out NP patients flagged among cells with progressor patients caught >= R49's - tol (same labels); primary tol 1.0", "tolerances": TOLS}
    real = {}
    for mode in ("b", "a"):
        f = f"{SD_}/{mode}_free_s0000.npz"
        if os.path.exists(f):
            names0, v0, _ = load(f); real[mode] = stats(names0, v0); res.setdefault("n_cells", {})[mode] = len(names0)
            if mode == "b": real["bcore"] = stats(names0, v0, True); res["n_cells"]["bcore"] = sum(not x.startswith("e3x:") for x in names0)
    res["real"] = real
    for mode, kind, core in (("a", "free", False), ("b", "free", False), ("b", "strat", False), ("b", "free", True), ("b", "strat", True)):
        fs = sorted(f for f in glob.glob(f"{SD_}/{mode}_{kind}_s*.npz") if not f.endswith("s0000.npz"))
        if not fs or mode not in real: continue
        recs = []; ncell = set()
        for f in fs:
            names, v, _ = load(f); ncell.add(len(names)); recs.append(stats(names, v, core))
        B = len(recs); rv = real["bcore" if core else mode]; o = {"B": B, "cells_per_label_set": sorted(ncell), "core_family_set_only": core}
        def summ(null, obs):
            null = np.asarray(null, float); p = (1 + int((null <= obs + 1e-9).sum())) / (B + 1)
            return {"real": r3(obs), "null_mean": r3(null.mean()), "null_p2.5_p97.5": [r3(np.percentile(null, 2.5)), r3(np.percentile(null, 97.5))], "null_min": r3(null.min()), "null_p1": r3(np.percentile(null, 1)),
                    "n_null_at_or_below_real": int((null <= obs + 1e-9).sum()), "p": r4(p), "p_floor": r4(1 / (B + 1))}
        for k in [f"tol{t}" for t in TOLS] + ["tier0_tol0.5", "tier0_tol1.0"]:
            dn = [r[k]["dNP"] for r in recs]; dc = [r[k]["dC"] for r in recs]
            o[k] = {"dNP": summ(dn, rv[k]["dNP"]), "real_dC": r3(rv[k]["dC"]), "real_cell": rv[k]["cell"], "null_dC_mean": r3(np.mean(dc)), "null_dC_p2.5_p97.5": [r3(np.percentile(dc, 2.5)), r3(np.percentile(dc, 97.5))],
                    "null_selected_source_counts": pd.Series([r[k]["cell"].split("|")[0] + "|" + r[k]["cell"].split("|")[1] for r in recs]).value_counts().head(10).to_dict(),
                    "null_dNP_values": [r3(x) for x in dn], "null_dC_values": [r3(x) for x in dc]}
        o["excess"] = {"value": summ([r["excess"]["value"] for r in recs], rv["excess"]["value"]), "real_cell": rv["excess"]["cell"], "null_values": [r4(r["excess"]["value"]) for r in recs]}
        o["named_rules"] = {nm: {"dNP": summ([r["named"][nm]["dNP"] for r in recs], rv["named"][nm]["dNP"]), "dC_null_mean": r3(np.mean([r["named"][nm]["dC"] for r in recs])), "real_dC": r3(rv["named"][nm]["dC"]),
                                 "excess": summ([r["named"][nm]["excess"] for r in recs], rv["named"][nm]["excess"])} for nm in NAMED if nm in rv["named"] and nm != "R49 gate"}
        res[f"{mode}{'core' if core else ''}_{kind}"] = o
    res["limg"] = LIMG; jd(res, "merge" + SFX)

"""Exploration E8 (docs/paper_explore_E8.md; rules paper/reviews_round7/decisions/EXPLORE_RULES.md): per-sample QC, noise, segment-structure and fragile-site
features from the QDNAseq 50-kb outputs (hg38; read-only). Row-level output only on the cluster: killcoyne_mm/explore_E8/feat8_<chunk>.csv.
Usage: CHUNK=i NCHUNK=k python ex_e8_feat.py"""
import os, numpy as np, pandas as pd
T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"; M = T + "/feasibility/paper_plan/killcoyne_mm"; XD = M + "/explore_E8"; os.makedirs(XD, exist_ok=True)
CH = int(os.environ.get("CHUNK", 0)); NCH = int(os.environ.get("NCHUNK", 1)); OUTF = f"{XD}/feat8_{CH:02d}.csv"
FRAG = {"fhit": ("3", 59747277, 61251459), "wwox": ("16", 78099400, 79212667)}   # hg38 gene bodies (FRA3B, FRA16D)
C = pd.read_csv(M + "/set_C.csv", dtype=str); C = C.iloc[CH::NCH]
def one(sample, raw):
    d = os.path.dirname(raw); o = {"Sample": sample}
    try:
        rc = pd.read_csv(d + "/50.readCountSummary.txt", sep="\t"); o["total_reads"] = float(rc["total.reads"].iloc[0]); o["used_reads"] = float(rc["used.reads"].iloc[0]); o["expvar"] = float(rc["expected.variance"].iloc[0])
    except Exception: pass
    cn = pd.read_csv(d + "/50.copy_number.txt", sep="\t"); sg = pd.read_csv(d + "/50.copy_number_segmented.txt", sep="\t"); assert len(cn) == len(sg) and (cn.feature.values == sg.feature.values).all()
    ch = cn.chromosome.astype(str).values; st = cn.start.values.astype(float); v = cn.iloc[:, 4].values.astype(float); s = sg.iloc[:, 4].values.astype(float)
    auto = np.isin(ch, [str(i) for i in range(1, 23)]); ok = auto & np.isfinite(v) & np.isfinite(s); v, s, ch, st = v[ok], s[ok], ch[ok], st[ok]
    same = ch[1:] == ch[:-1]; o["mapd"] = float(np.median(np.abs(np.diff(v)[same]))); o["segmad"] = float(1.4826 * np.median(np.abs(v - s))); o["n_bins"] = int(len(v))
    # segments = runs of identical segmented value within a chromosome
    brk = np.r_[True, ~same | (np.diff(s) != 0)]; sid = np.cumsum(brk) - 1; nb = np.bincount(sid); lev = s[brk]; gm = float(np.median(s)); amp = lev - gm; big = nb >= 20   # >= 1 Mb
    o["nseg"] = int(len(nb)); o["nseg_1mb"] = int(big.sum())
    for lo, hi, nm in ((0.03, 0.10, "small"), (0.10, np.inf, "large"), (0.02, 0.06, "tiny")):
        m = big & (np.abs(amp) > lo) & (np.abs(amp) <= hi); o[f"nseg_{nm}"] = int(m.sum()); o[f"frac_{nm}"] = float(nb[m].sum() / len(v))
    o["n_levels05"] = int(len(np.unique(np.round(amp[big & (np.abs(amp) > 0.03)] / 0.05)))); o["seg_amp_sd"] = float(np.sqrt(np.average(amp ** 2, weights=nb)))
    o["max_abs_amp_1mb"] = float(np.abs(amp[big]).max()) if big.any() else 0.0
    for nm, (c_, a, b) in FRAG.items():
        inr = (ch == c_) & (st >= a) & (st <= b); fl = (ch == c_) & (st >= a - 10e6) & (st <= b + 10e6) & ~inr
        o[f"{nm}_bins"] = int(inr.sum()); o[f"{nm}_dev"] = float(np.median(v[inr]) - np.median(v[fl])) if inr.sum() >= 5 and fl.sum() >= 20 else np.nan
        o[f"{nm}_segdev"] = float(np.min(s[inr]) - np.median(s[fl])) if inr.sum() >= 5 and fl.sum() >= 20 else np.nan
    return o
rows = []
for sm, raw in zip(C.Sample.values, C.raw.values):
    try: rows.append(one(sm, raw))
    except Exception as ex: rows.append({"Sample": sm, "error": str(ex)[:200]})
pd.DataFrame(rows).to_csv(OUTF, index=False); print("EXE8 FEAT DONE", CH, len(rows), flush=True)

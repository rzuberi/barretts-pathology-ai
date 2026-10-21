"""Explore/pert report: official PerturbQA auc_per_gene (DE, DIR) + 3-class accuracy per cell line and mean over cells for every
preds/*__test.jsonl here plus Task F's zero-shot / SFT / GRPO test preds; paired bootstrap over perturbations (NB resamples) for the
models in MAIN (env, comma list; default: all non-kNN-grid models) with differences vs Task F SFT. Copy of f_report.py's machinery.
Out: results.json, results.md (table). Val-based choice among the kNN grid: knn_select.json."""
import glob, json, os, sys
import numpy as np
sys.path.insert(0, "/mnt/scratche/slow/fmlab/zuberi01/perturbqa/PerturbQA"); from perturbqa.eval import auc_per_gene
B = "/mnt/scratche/slow/fmlab/zuberi01"; F = B + "/analysis/rl_followups/F"; O = B + "/analysis/rl_explore/pert"; NB = int(os.environ.get("NB", 1000)); RNG = np.random.RandomState(20261016)
M = {"F_zero_shot": f"{F}/preds/zs__test.jsonl", "F_sft": f"{F}/preds/sft_all__checkpoint-3000__test.jsonl", "F_grpo": f"{F}/preds/grpo_all__checkpoint-300__test.jsonl"}
for f in sorted(glob.glob(f"{O}/preds/*__test.jsonl")):
    t = os.path.basename(f)[: -len("__test.jsonl")]
    if not t.startswith("smk_"): M[t] = f
rows = [json.loads(l) for l in open(F + "/data/test.jsonl")]; key = lambda r: (r["cell"], r["pert"], r["gene"]); cells = ["k562", "rpe1", "hepg2", "jurkat"]
P = {m: {key(r): r for r in map(json.loads, open(f))} for m, f in M.items() if os.path.exists(f)}
grid = lambda m: m.split("_")[0] in ("knn", "strength", "blend")
MAIN = set(os.environ["MAIN"].split(",")) if os.environ.get("MAIN") else {m for m in P if not grid(m)}
# val-based choice inside the kNN grid: best val DE (knn_*), best val DIR, best val DE for strength_* and blend_*
vrows = [json.loads(l) for l in open(F + "/data/val.jsonl")]; vsel = {}
for fam in ("knn", "strength", "blend"):
    sc = {}
    for f in glob.glob(f"{O}/preds/{fam}_*__val.jsonl"):
        t = os.path.basename(f)[: -len("__val.jsonl")]; p = {key(r): r for r in map(json.loads, open(f))}
        de = dir_ = []; d1, d2 = [], []
        for c in cells:
            R = [r for r in vrows if r["cell"] == c and key(r) in p]; Rd = [r for r in R if r["dir"] is not None]
            if not R: continue
            d1.append(auc_per_gene([(r["pert"], r["gene"]) for r in R], [p[key(r)]["p_up"] + p[key(r)]["p_down"] for r in R], [r["de"] for r in R]))
            d2.append(auc_per_gene([(r["pert"], r["gene"]) for r in Rd], [p[key(r)]["p_up"] / max(p[key(r)]["p_up"] + p[key(r)]["p_down"], 1e-12) for r in Rd], [r["dir"] for r in Rd]))
        sc[t] = {"val_de": round(float(np.mean(d1)), 4), "val_dir": round(float(np.mean(d2)), 4)}
    if sc:
        vsel[fam] = {"scores": sc, "best_de": max(sc, key=lambda t: sc[t]["val_de"]), "best_dir": max(sc, key=lambda t: sc[t]["val_dir"])}; MAIN |= {vsel[fam]["best_de"], vsel[fam]["best_dir"]}
json.dump(vsel, open(O + "/knn_select.json", "w"), indent=1); print("val-selected", {k: (v["best_de"], v["best_dir"]) for k, v in vsel.items()})
class Gene:
    def __init__(s, genes, score, y):
        o = np.lexsort((score, genes)); s.o = o; g, sc, s.y = genes[o], score[o], y[o].astype(bool)
        s.gid = np.unique(g, return_inverse=True)[1]; s.ng = s.gid.max() + 1
        tie = np.r_[True, (g[1:] != g[:-1]) | (sc[1:] != sc[:-1])]; s.tid = np.cumsum(tie) - 1; s.gstart = np.r_[True, g[1:] != g[:-1]]
    def auc(s, w):
        w = w[s.o]; wn = w * ~s.y; wp = w * s.y
        cn = np.cumsum(wn); first = np.flatnonzero(s.gstart); base = np.repeat(cn[first] - wn[first], np.diff(np.r_[first, len(w)]))
        tn = np.bincount(s.tid, wn); tstart = np.r_[0, np.flatnonzero(np.diff(s.tid)) + 1]
        tend = np.r_[tstart[1:] - 1, len(w) - 1]; below = (cn - base)[tend][s.tid] - tn[s.tid]
        num = np.bincount(s.gid, wp * (below + 0.5 * tn[s.tid]), s.ng); Pg = np.bincount(s.gid, wp, s.ng); Ng = np.bincount(s.gid, wn, s.ng)
        ok = (Pg > 0) & (Ng > 0); return float((num[ok] / (Pg[ok] * Ng[ok])).mean())
ci = lambda v: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
res = {"models": M, "n_test_pairs": len(rows), "bootstrap": NB, "main": sorted(MAIN), "per_cell": {}}; ALLB = {}
for c in cells:
    R = [r for r in rows if r["cell"] == c]; Rd = [r for r in R if r["dir"] is not None]
    perts = np.unique([r["pert"] for r in R]); pidx = {p: i for i, p in enumerate(perts)}
    pi = np.array([pidx[r["pert"]] for r in R]); pid = np.array([pidx[r["pert"]] for r in Rd])
    ws = [np.bincount(RNG.randint(0, len(perts), len(perts)), minlength=len(perts)).astype(float) for _ in range(NB)]
    out = {"n_pairs": len(R), "n_dir_pairs": len(Rd), "n_perturbations": len(perts)}; boots = {}
    for m, p in P.items():
        if any(key(r) not in p for r in R): continue
        q = [p[key(r)] for r in R]; qd = [p[key(r)] for r in Rd]
        de_s = np.array([x["p_up"] + x["p_down"] for x in q]); di_s = np.array([x["p_up"] / max(x["p_up"] + x["p_down"], 1e-12) for x in qd])
        de_y = np.array([r["de"] for r in R]); di_y = np.array([r["dir"] for r in Rd])
        de = auc_per_gene([(r["pert"], r["gene"]) for r in R], de_s, de_y); di = auc_per_gene([(r["pert"], r["gene"]) for r in Rd], di_s, di_y)
        am = np.array([["up", "down", "none"][int(np.argmax([x["p_up"], x["p_down"], x["p_none"]]))] for x in q]); corr = am == np.array([r["label"] for r in R])
        out[m] = {"de_auroc": round(de, 4), "dir_auroc": round(di, 4), "acc3": round(float(corr.mean()), 4)}
        if m in MAIN:
            gd, gi = Gene(np.array([r["gene"] for r in R]), de_s, de_y), Gene(np.array([r["gene"] for r in Rd]), di_s, di_y)
            assert abs(gd.auc(np.ones(len(R))) - de) < 1e-6
            b = np.array([[gd.auc(w[pi]), gi.auc(w[pid])] for w in ws]); boots[m] = b; out[m].update({"de_ci": ci(b[:, 0]), "dir_ci": ci(b[:, 1])})
    for a in boots:
        for z in ("F_sft", "F_zero_shot") + ((("sft_" + a[5:]),) if a.startswith("grpo_") else ()):
            if z in boots and a != z:
                d = boots[a] - boots[z]; pt = [out[a]["de_auroc"] - out[z]["de_auroc"], out[a]["dir_auroc"] - out[z]["dir_auroc"]]
                out[f"{a}_minus_{z}"] = {"de": round(pt[0], 4), "de_ci": ci(d[:, 0]), "dir": round(pt[1], 4), "dir_ci": ci(d[:, 1]), "mean_de_dir": round(float(np.mean(pt)), 4), "mean_ci": ci(d[:, :2].mean(1))}
    ALLB[c] = boots; res["per_cell"][c] = out; print(c, {m: (out[m]["de_auroc"], out[m]["dir_auroc"]) for m in out if isinstance(out[m], dict) and "de_auroc" in out[m] and m in MAIN}, flush=True)
mean = {}
for m in P:
    v = [res["per_cell"][c][m] for c in cells if m in res["per_cell"][c]]
    if len(v) == len(cells):
        mean[m] = {k: round(float(np.mean([x[k] for x in v])), 4) for k in ("de_auroc", "dir_auroc", "acc3")}
        if all(m in ALLB[c] for c in cells):
            b = np.mean([ALLB[c][m] for c in cells], 0); mean[m].update({"de_ci": ci(b[:, 0]), "dir_ci": ci(b[:, 1])})
            for z in ("F_sft", "F_zero_shot") + ((("sft_" + m[5:]),) if m.startswith("grpo_") else ()):
                if z in mean and all(z in ALLB[c] for c in cells) and z != m:
                    d = np.mean([ALLB[c][m] - ALLB[c][z] for c in cells], 0); pt = [mean[m]["de_auroc"] - mean[z]["de_auroc"], mean[m]["dir_auroc"] - mean[z]["dir_auroc"]]
                    mean[m][f"minus_{z}"] = {"de": round(pt[0], 4), "de_ci": ci(d[:, 0]), "dir": round(pt[1], 4), "dir_ci": ci(d[:, 1]), "mean_de_dir": round(float(np.mean(pt)), 4), "mean_ci": ci(d.mean(1))}
res["mean_over_cells"] = mean; json.dump(res, open(O + "/results.json", "w"), indent=1)
L = ["| model | " + " | ".join(f"{c} DE/DIR" for c in cells) + " | mean DE | mean DIR | acc3 |", "|---" * (len(cells) + 4) + "|"]
for m in sorted(mean, key=lambda m: -mean[m]["de_auroc"]):
    f = lambda c: f"{res['per_cell'][c][m]['de_auroc']:.3f}/{res['per_cell'][c][m]['dir_auroc']:.3f}"
    L.append(f"| {m} | " + " | ".join(f(c) for c in cells) + f" | {mean[m]['de_auroc']:.3f}" + (f" {mean[m]['de_ci']}" if "de_ci" in mean[m] else "") + f" | {mean[m]['dir_auroc']:.3f}" + (f" {mean[m]['dir_ci']}" if "dir_ci" in mean[m] else "") + f" | {mean[m]['acc3']:.3f} |")
open(O + "/results.md", "w").write("\n".join(L) + "\n"); print("\n".join(L))
for m in sorted(mean):
    for k, v in mean[m].items():
        if k.startswith("minus_"): print(m, k, json.dumps(v))

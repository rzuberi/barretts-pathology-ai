"""Follow-up F report: DE / DIR per-gene AUROC (PerturbQA auc_per_gene) and 3-class accuracy per cell line, for zero-shot, SFT and
SFT->GRPO (all-cells and the no-RPE1 cross-cell variant), with a 2,000-resample bootstrap over perturbations (paired across models).
Out: rl_followups/F/results.json"""
import json, os, sys
import numpy as np
sys.path.insert(0, "/mnt/scratche/slow/fmlab/zuberi01/perturbqa/PerturbQA"); from perturbqa.eval import auc_per_gene
F = "/mnt/scratche/slow/fmlab/zuberi01/analysis/rl_followups/F"; NB = int(os.environ.get("NB", 2000)); RNG = np.random.RandomState(20261016)
def sel(p):
    f = f"{F}/preds/selected_{p}.json"; return json.load(open(f))["selected"] if os.path.exists(f) else None
M = {"zero_shot": "zs"}
for v in ("all", "nor"):
    for k in ("sft", "grpo"):
        s = sel(f"{k}_{v}")
        if s: M[f"{k}_{v}"] = f"{k}_{v}__{s}"
rows = [json.loads(l) for l in open(F + "/data/test.jsonl")]; key = lambda r: (r["cell"], r["pert"], r["gene"])
P = {}
for m, t in M.items():
    f = f"{F}/preds/{t}__test.jsonl"
    if os.path.exists(f): P[m] = {key(r): r for r in map(json.loads, open(f))}
print("models", {m: len(p) for m, p in P.items()})
cells = sorted({r["cell"] for r in rows})
class Gene:
    """per-gene weighted AUROC, vectorised over genes; weights come from perturbation multiplicities in each bootstrap resample"""
    def __init__(s, genes, score, y):
        o = np.lexsort((score, genes)); s.o = o; g, sc, s.y = genes[o], score[o], y[o].astype(bool)
        s.gid = np.unique(g, return_inverse=True)[1]; s.ng = s.gid.max() + 1
        tie = np.r_[True, (g[1:] != g[:-1]) | (sc[1:] != sc[:-1])]; s.tid = np.cumsum(tie) - 1
        s.gstart = np.r_[True, g[1:] != g[:-1]]
    def auc(s, w):
        w = w[s.o]; wn = w * ~s.y; wp = w * s.y
        cn = np.cumsum(wn); first = np.flatnonzero(s.gstart); base = np.repeat(cn[first] - wn[first], np.diff(np.r_[first, len(w)]))
        tn = np.bincount(s.tid, wn); tstart = np.r_[0, np.flatnonzero(np.diff(s.tid)) + 1]
        tend = np.r_[tstart[1:] - 1, len(w) - 1]; below = (cn - base)[tend][s.tid] - tn[s.tid]  # neg weight strictly below each tie group, within gene
        num = np.bincount(s.gid, wp * (below + 0.5 * tn[s.tid]), s.ng); Pg = np.bincount(s.gid, wp, s.ng); Ng = np.bincount(s.gid, wn, s.ng)
        ok = (Pg > 0) & (Ng > 0); return float((num[ok] / (Pg[ok] * Ng[ok])).mean())
ALLB = {}
res = {"models": M, "n_test_pairs": len(rows), "per_cell": {}, "bootstrap": NB}
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
        gd, gi = Gene(np.array([r["gene"] for r in R]), de_s, de_y), Gene(np.array([r["gene"] for r in Rd]), di_s, di_y)
        chk = gd.auc(np.ones(len(R))); assert abs(chk - de) < 1e-6, (chk, de)
        b = np.array([[gd.auc(w[pi]), gi.auc(w[pid]), (corr * w[pi]).sum() / w[pi].sum()] for w in ws]); boots[m] = b
        ci = lambda v: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
        out[m] = {"de_auroc": round(de, 4), "de_ci": ci(b[:, 0]), "dir_auroc": round(di, 4), "dir_ci": ci(b[:, 1]), "acc3": round(float(corr.mean()), 4), "acc3_ci": ci(b[:, 2]),
                  "mean_prob_mass": round(float(np.mean([x["mass"] for x in q])), 4)}
    for a, z in (("grpo_all", "sft_all"), ("sft_all", "zero_shot"), ("grpo_nor", "sft_nor"), ("sft_nor", "zero_shot")):
        if a in boots and z in boots:
            d = boots[a] - boots[z]; pt = [out[a]["de_auroc"] - out[z]["de_auroc"], out[a]["dir_auroc"] - out[z]["dir_auroc"]]
            out[f"{a}_minus_{z}"] = {"de": round(pt[0], 4), "de_ci": ci(d[:, 0]), "dir": round(pt[1], 4), "dir_ci": ci(d[:, 1]),
                                     "mean_de_dir": round(float(np.mean(pt)), 4), "mean_ci": ci(d[:, :2].mean(1))}
    ALLB[c] = boots; out["frac_none"] = round(float(np.mean([r["label"] == "none" for r in R])), 4)
    res["per_cell"][c] = out; print(c, json.dumps({k: v for k, v in out.items() if isinstance(v, dict)})[:900], flush=True)
for m in P:
    v = [res["per_cell"][c][m] for c in cells if m in res["per_cell"][c]]
    if len(v) == len(cells): res[f"{m}_mean_over_cells"] = {k: round(float(np.mean([x[k] for x in v])), 4) for k in ("de_auroc", "dir_auroc", "acc3")}
ci = lambda v: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
for a, z in (("grpo_all", "sft_all"), ("sft_all", "zero_shot")):
    if all(a in ALLB[c] and z in ALLB[c] for c in cells):
        d = np.mean([ALLB[c][a][:, :2] - ALLB[c][z][:, :2] for c in cells], 0)  # (NB, 2): mean over cells, cells resampled independently
        pt = [np.mean([res["per_cell"][c][a][k] - res["per_cell"][c][z][k] for c in cells]) for k in ("de_auroc", "dir_auroc")]
        res[f"PRIMARY_{a}_minus_{z}_mean_over_cells"] = {"de": round(float(pt[0]), 4), "de_ci": ci(d[:, 0]), "dir": round(float(pt[1]), 4), "dir_ci": ci(d[:, 1]),
                                                       "mean_de_dir": round(float(np.mean(pt)), 4), "mean_ci": ci(d.mean(1))}
json.dump(res, open(F + "/results.json", "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k.startswith("PRIMARY") or k.endswith("over_cells")}, indent=1))
print("rpe1 nor", json.dumps({k: v for k, v in res["per_cell"]["rpe1"].items() if "nor" in k}))
print("frac_none", {c: res["per_cell"][c]["frac_none"] for c in cells})

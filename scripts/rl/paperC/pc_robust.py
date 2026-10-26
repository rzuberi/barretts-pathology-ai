"""Paper C item 2: robustness of the two-way corrected metric, plus the data for the "LLM copies retrieval" figure.

Three questions.
 1. How are the knockdown and gene effects estimated? Variants of the residualisation:
      own      : alternating centring on the method's own test scores (the primary metric, as in pc_common.twoway_residual)
      cf2      : cross-fitted - effects estimated on one random half of each knockdown's pairs and applied to the other half
      shrunk-L : group means shrunk towards the global mean by n/(n + L) for L in SHRINK (L = 0 is `own`)
      genetrain: gene effect taken from the *training* label rate for that gene (not from the score), knockdown effect from the score
 2. Per cell line for every variant (already have the primary metric per line; this adds the variants).
 3. A simulation: a pure-shortcut predictor (knockdown effect + gene effect, no interaction) must score 0.5 under the
    corrected metric at any effect size, and a predictor carrying a known gene-by-knockdown signal must recover it.

Out: $O/robust.json, $O/copy_scatter.json (per-knockdown mean scores of sft_r vs its k = 10 retrieval, and per-pair agreement).
CPU only; public PerturbQA data.
"""
import json, sys
import numpy as np, pandas as pd
from multiprocessing import Pool
from pc_common import *

SHRINK = [0, 1, 5, 20, 100]
KEY = ["knn_mean_k20", "knn_mean_k10", "F_sft", "sft_r", "corevote_meansim", "shiftknn_k20", "listsize_only"]
PATHS = {"F_sft": F + "/preds/sft_all__checkpoint-3000__test.jsonl", "sft_r": X + "/preds/sft_r__checkpoint-2000__test.jsonl",
         "knn_mean_k10": X + "/preds/knn_mean_k10__test.jsonl", "knn_mean_k20": X + "/preds/knn_mean_k20__test.jsonl",
         "corevote_meansim": O + "/preds/corevote_meansim__test.jsonl", "shiftknn_k20": O + "/preds/shiftknn_k20__test.jsonl",
         "listsize_only": O + "/preds/listsize_only__test.jsonl"}


def centre_shrunk(s, grp, lam):
    """Subtract a shrunk group mean: mean_g pulled towards the global mean by n_g / (n_g + lam)."""
    s = np.asarray(s, float); g = pd.Series(s).groupby(np.asarray(grp))
    m = g.transform("mean").values; n = g.transform("size").values; gm = s.mean()
    return np.round(s - (gm + (m - gm) * n / (n + lam)), 9)


def twoway(s, perts, genes, lam=0.0, n_iter=20, scale="rank"):
    """scale: 'rank' = the primary metric (alternating centring of the global percentile rank); 'score' = centre the score
    itself; 'logitrank' = centre the logit of the percentile rank. A shortcut that is additive in knockdown and gene on the
    *score* scale is removed exactly only if the centring is done on that same scale, so the scale is part of the metric."""
    r = {"rank": lambda: global_rank(s), "score": lambda: np.asarray(s, float),
         "logitrank": lambda: np.log(global_rank(s) / (1 - global_rank(s)))}[scale]()
    for _ in range(n_iter):
        r = centre_shrunk(r, perts, lam); r = centre_shrunk(r, genes, lam)
    return r


def twoway_cf(s, perts, genes, seed=0, n_iter=20):
    """Cross-fitted: effects estimated on the complementary half of the pairs, applied to this half."""
    r = global_rank(s); rng = np.random.default_rng(seed); half = rng.integers(0, 2, len(r))
    out = r.copy()
    for _ in range(n_iter):
        for grp in (perts, genes):
            new = out.copy()
            for h in (0, 1):
                fit, app = half == 1 - h, half == h
                m = pd.Series(out[fit]).groupby(np.asarray(grp)[fit]).mean()
                new[app] = out[app] - pd.Series(np.asarray(grp)[app]).map(m).fillna(out[fit].mean()).values
            out = np.round(new, 9)
    return out


def var_shares(s, perts, genes, n_iter=20):
    """Share of the score's (percentile-rank) variance taken by the additive knockdown part, the additive gene part and the
    gene-by-knockdown remainder. The additive parts are the alternating-projection fit, so they are the exact components the
    corrected metric removes. Shares are normalised to sum to 1 (the two additive parts are near-orthogonal but not exactly)."""
    r = global_rank(s); resid = r.copy(); pc = np.zeros_like(r); gc = np.zeros_like(r)
    for _ in range(n_iter):
        m = pd.Series(resid).groupby(np.asarray(perts)).transform("mean").values; pc += m; resid = resid - m
        m = pd.Series(resid).groupby(np.asarray(genes)).transform("mean").values; gc += m; resid = resid - m
    v = np.array([pc.var(), gc.var(), resid.var()]); v = v / v.sum()
    return {"knockdown": float(v[0]), "gene": float(v[1]), "interaction": float(v[2])}


def gene_train_rate(c, kind):
    """Training label rate per gene: DE rate, or up-fraction among changed (label-free on test: training labels only)."""
    de, dr, _ = load_cell(c)
    tr = de[de.split == "train"] if "split" in de.columns else de
    if kind == "de": return tr.groupby("gene").label.mean()
    trd = dr[dr.split == "train"] if "split" in dr.columns else dr
    return trd.groupby("gene").label.mean()


def run_cell(c):
    df = test_frame(c); perts = df.pert.values; genes = df.gene.values
    dmask = (df.de.values == 1) & (df["dir"].values >= 0); ydir = (df["dir"].values[dmask] == 1).astype(int)
    Gde = GeneAUC(genes, df.de.values); Gdir = GeneAUC(genes[dmask], ydir)
    gt = {k: gene_train_rate(c, k) for k in ("de", "dir")}
    out = {}
    for tag in KEY:
        sde, sdir = scores_from_preds(PATHS[tag], df, c)
        sd = sdir[dmask]; pdm = perts[dmask]; gdm = genes[dmask]
        v = {}
        for lam in SHRINK:
            v[f"DE_2w_shrink{lam}"] = Gde.auc(Gde.build(twoway(sde, perts, genes, lam)), np.ones(len(sde)))
            v[f"DIR_2w_shrink{lam}"] = Gdir.auc(Gdir.build(twoway(sd, pdm, gdm, lam)), np.ones(len(sd)))
        v["DE_2w_cf2"] = Gde.auc(Gde.build(twoway_cf(sde, perts, genes)), np.ones(len(sde)))
        v["DIR_2w_cf2"] = Gdir.auc(Gdir.build(twoway_cf(sd, pdm, gdm)), np.ones(len(sd)))
        for sc in ("score", "logitrank"):
            v[f"DE_2w_{sc}"] = Gde.auc(Gde.build(twoway(sde, perts, genes, scale=sc)), np.ones(len(sde)))
            v[f"DIR_2w_{sc}"] = Gdir.auc(Gdir.build(twoway(sd, pdm, gdm, scale=sc)), np.ones(len(sd)))
        # gene effect from the training label rate, knockdown effect from the score
        for m, (G, s, pp, gg, kind, y) in {"DE": (Gde, sde, perts, genes, "de", df.de.values), "DIR": (Gdir, sd, pdm, gdm, "dir", ydir)}.items():
            gr = pd.Series(gg).map(gt[kind]).fillna(gt[kind].mean()).values
            r = global_rank(s)
            for _ in range(20):
                r = centre_by_pert(r, pp); r = partial_out(r, gr)
            v[f"{m}_2w_genetrain"] = G.auc(G.build(r), np.ones(len(s)))
        out[tag] = {k: float(x) for k, x in v.items()}
        print(c, tag, {k: round(x, 3) for k, x in out[tag].items()}, flush=True)
    # variance decomposition of each method's score: additive knockdown part, additive gene part, interaction (residual)
    vd = {}
    for tag in KEY:
        sde, sdir = scores_from_preds(PATHS[tag], df, c)
        vd[tag] = {m: var_shares(s, pp, gg) for m, (s, pp, gg) in
                   {"DE": (sde, perts, genes), "DIR": (sdir[dmask], perts[dmask], genes[dmask])}.items()}
    # "LLM copies retrieval": per-pair and per-knockdown agreement of sft_r with its k = 10 retrieval
    rde, rdir = scores_from_preds(PATHS["sft_r"], df, c); kde, kdir = scores_from_preds(PATHS["knn_mean_k10"], df, c)
    from scipy.stats import spearmanr
    cp = {"pair_de_spearman": float(spearmanr(rde, kde).statistic), "pair_dir_spearman": float(spearmanr(rdir[dmask], kdir[dmask]).statistic),
          "pert_de": [list(map(float, pd.Series(x).groupby(perts).mean().values)) for x in (rde, kde)],
          "pert_dir": [list(map(float, pd.Series(x).groupby(pdm).mean().values)) for x in (rdir[dmask], kdir[dmask])]}
    return c, out, cp, vd


def simulate(seed=0, n_pert=300, n_gene=400, listsize=(40, 120)):
    """Pairs with an additive knockdown + gene logit and an interaction term. A predictor built only from the additive part
    must score 0.5 under the corrected metric; one that sees the interaction must score above it."""
    rng = np.random.default_rng(seed)
    a = rng.normal(0, 1.0, n_pert); b = rng.normal(0, 1.0, n_gene)          # knockdown and gene effects
    U = rng.normal(0, 1.0, (n_pert, 4)); V = rng.normal(0, 1.0, (n_gene, 4))  # rank-4 interaction
    rows = []
    for p in range(n_pert):
        gs = rng.choice(n_gene, rng.integers(*listsize), replace=False)
        inter = (U[p] @ V[gs].T) / 2
        logit = a[p] + b[gs] + inter
        y = rng.random(len(gs)) < 1 / (1 + np.exp(-logit))
        for g, i, yy in zip(gs, inter, y): rows.append((p, g, i, yy, a[p] + b[g]))
    d = pd.DataFrame(rows, columns=["pert", "gene", "inter", "y", "addit"]); d["y"] = d.y.astype(int)
    G = GeneAUC(d.gene.values, d.y.values); w = np.ones(len(d))
    noise = rng.normal(0, 1, len(d))
    out = {}
    for name, s in {"shortcut_only (additive)": d.addit.values, "oracle_logit (additive + interaction)": d.addit.values + d.inter.values,
                    "interaction_only": d.inter.values, "knockdown_effect_only": d.groupby("pert").addit.transform("mean").values,
                    "noise": noise}.items():
        out[name] = {"raw": float(G.auc(G.build(s), w)),
                     **{f"corrected_{sc}": float(G.auc(G.build(twoway(s, d.pert.values, d.gene.values, scale=sc)), w))
                        for sc in ("rank", "score", "logitrank")}}
    return out


if __name__ == "__main__":
    with Pool(4) as pool: res = pool.map(run_cell, CELLS)
    rob = {c: r for c, r, _, _ in res}
    mean = {t: {m: float(np.mean([rob[c][t][m] for c in CELLS])) for m in rob[CELLS[0]][t]} for t in KEY}
    sim = {f"seed{s}": simulate(s) for s in (0, 1, 2)}
    json.dump({"per_cell": rob, "mean": mean, "simulation": sim}, open(O + "/robust.json", "w"), indent=1)
    json.dump({c: cp for c, _, cp, _ in res}, open(O + "/copy_scatter.json", "w"), indent=1)
    v = {c: x for c, _, _, x in res}
    vdm = {t_: {m: {k: float(np.mean([v[c][t_][m][k] for c in CELLS])) for k in ("knockdown", "gene", "interaction")}
                for m in ("DE", "DIR")} for t_ in KEY}
    json.dump(vdm, open(O + "/vardecomp.json", "w"), indent=1)
    print(json.dumps(mean, indent=1)); print(json.dumps(sim["seed0"], indent=1)); print("ROBUST DONE")

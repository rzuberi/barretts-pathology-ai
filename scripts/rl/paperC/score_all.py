"""Paper C scoring: every method under the raw and the shift-/list-size-corrected metrics, with a paired bootstrap over
perturbations (2,000 resamples, seed 0, the same resamples for every method and metric within a cell line).
Writes $O/results.json (point estimates + 95% CIs per cell and mean over cells), $O/boot.npz (bootstrap draws for any later
paired comparison), $O/diffs.json (paired differences against three references) and $O/results.md (tables).

Metrics (see pc_common for the transforms). Primary corrected metrics remove both the knockdown and the gene effect:
  DE_raw;  DE_2w (two-way knockdown + gene residual);  DE_lsg (log list size and the gene effect partialled out, alternating)
  DIR_raw; DIR_2w (two-way residual);                  DIR_resg (predicted up-fraction shiftknn_k20 and the gene effect partialled out, alternating)
Naive one-way corrections kept for the record (they re-introduce the gene prior and push prior-driven scores below 0.5):
  DE_wp, DIR_wp (within-knockdown rank), DIR_cen (own per-knockdown mean removed)
"""
import json, sys, os
import numpy as np, pandas as pd
from multiprocessing import Pool
from pc_common import *

NB, SEED = 2000, 0
METHODS = {  # tag -> preds file
    "F_zero_shot": F + "/preds/zs__test.jsonl", "F_sft": F + "/preds/sft_all__checkpoint-3000__test.jsonl", "F_grpo": F + "/preds/grpo_all__checkpoint-300__test.jsonl",
    "sft_r": X + "/preds/sft_r__checkpoint-2000__test.jsonl", "sft_kr": X + "/preds/sft_kr__checkpoint-666__test.jsonl", "grpo_kr": X + "/preds/grpo_kr__final__test.jsonl",
    "knn_mean_k10": X + "/preds/knn_mean_k10__test.jsonl", "knn_mean_k20": X + "/preds/knn_mean_k20__test.jsonl", "knn_kg_k20": X + "/preds/knn_kg_k20__test.jsonl",
    "oracle_self": X + "/preds/oracle_strength__test.jsonl",
}
for t in ["shiftknn_k10", "shiftknn_k20", "shiftknn_k50", "shift_ridge", "shift_kridge", "listsize_only", "oracle_loo",
          "corevote_full", "corevote_contrast", "corevote_anchor", "corevote_meansim"]:
    METHODS[t] = O + f"/preds/{t}__test.jsonl"
METHODS = {k: v for k, v in METHODS.items() if os.path.exists(v)}
REFS = ["knn_mean_k20", "shiftknn_k20", "F_sft"]


def run_cell(c):
    df = test_frame(c); n = len(df); perts = df.pert.values; genes = df.gene.values
    pid, pidx = np.unique(perts, return_inverse=True); rng = np.random.default_rng(SEED + CELLS.index(c))
    cnt = np.stack([np.bincount(rng.integers(0, len(pid), len(pid)), minlength=len(pid)) for _ in range(NB)])   # NB x perts
    Wall = cnt[:, pidx].astype(np.float32)                             # NB x pairs
    dmask = (df.de.values == 1) & (df["dir"].values >= 0); ydir = (df["dir"].values[dmask] == 1).astype(int); Wdir = Wall[:, dmask]
    sz = df.groupby("pert").size(); loglist = np.log(sz.loc[perts].values)
    shift = json.load(open(O + "/shift_pred.json"))[c]; up_hat = np.array([shift[p]["shiftknn_k20"][1] for p in perts])
    Gde = GeneAUC(genes, df.de.values); Gdir = GeneAUC(genes[dmask], ydir)
    keys_de = list(zip(df.pert, df.gene)); keys_dir = [k for k, m in zip(keys_de, dmask) if m]
    S = {}
    for tag, path in METHODS.items():
        S[tag] = scores_from_preds(path, df, c)
    # label-free shift-only version of the kNN (per-knockdown mean of its own scores; R3's decomposition)
    de20, dir20 = S["knn_mean_k20"]
    S["knn_k20_shiftpart"] = (pd.Series(de20).groupby(perts).transform("mean").values,
                              pd.Series(np.where(dmask, dir20, np.nan)).groupby(perts).transform("mean").fillna(0.5).values)
    res = {}; boot = {}
    for tag, (sde, sdir) in S.items():
        variants = {"DE_raw": (Gde, sde, Wall), "DE_2w": (Gde, twoway_residual(sde, perts, genes), Wall),
                    "DE_lsg": (Gde, covariate_residual(sde, loglist, genes), Wall), "DE_wp": (Gde, within_pert_rank(sde, perts), Wall)}
        sd = sdir[dmask]; pdm = perts[dmask]
        gdm = genes[dmask]
        variants.update({"DIR_raw": (Gdir, sd, Wdir), "DIR_2w": (Gdir, twoway_residual(sd, pdm, gdm), Wdir),
                         "DIR_resg": (Gdir, covariate_residual(sd, up_hat[dmask], gdm), Wdir),
                         "DIR_wp": (Gdir, within_pert_rank(sd, pdm), Wdir), "DIR_cen": (Gdir, centre_by_pert(sd, pdm), Wdir)})
        res[tag] = {}; boot[tag] = {}
        for m, (G, s, W) in variants.items():
            C = G.build(s); point = G.auc(C, np.ones(len(s)))
            chk = auc_per_gene_official(keys_de if m.startswith("DE") else keys_dir, s, df.de.values if m.startswith("DE") else ydir)
            assert abs(point - chk) < 1e-6, (tag, m, point, chk)
            b = np.array([G.auc(C, W[i]) for i in range(NB)])
            res[tag][m] = [point, float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]; boot[tag][m] = b.astype(np.float32)
        print(c, tag, {m: round(v[0], 3) for m, v in res[tag].items()}, flush=True)
    return c, res, boot


if __name__ == "__main__":
    with Pool(4) as pool: out = pool.map(run_cell, CELLS)
    res = {c: r for c, r, _ in out}; boot = {c: b for c, _, b in out}
    tags = list(res[CELLS[0]]); metrics = list(res[CELLS[0]][tags[0]])
    final = {}
    for t in tags:
        final[t] = {}
        for m in metrics:
            bm = np.mean([boot[c][t][m] for c in CELLS], 0)
            final[t][m] = {"mean": [float(np.mean([res[c][t][m][0] for c in CELLS])), float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))],
                           **{c: res[c][t][m] for c in CELLS}}
    diffs = {}
    for ref in REFS:
        for t in tags:
            if t == ref: continue
            diffs[f"{t} - {ref}"] = {}
            for m in metrics:
                d = np.mean([boot[c][t][m] - boot[c][ref][m] for c in CELLS], 0)
                diffs[f"{t} - {ref}"][m] = [final[t][m]["mean"][0] - final[ref][m]["mean"][0], float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
    json.dump(final, open(O + "/results.json", "w"), indent=1); json.dump(diffs, open(O + "/diffs.json", "w"), indent=1)
    np.savez_compressed(O + "/boot.npz", **{f"{c}__{t}__{m}": boot[c][t][m] for c in CELLS for t in tags for m in metrics})
    order = sorted(tags, key=lambda t: -final[t]["DE_raw"]["mean"][0])
    L = ["# Paper C: all methods, raw vs corrected metrics (mean over K562/RPE1/HepG2/Jurkat; official `auc_per_gene`; 95% paired bootstrap over perturbations, 2,000 resamples)", "",
         "| method | " + " | ".join(metrics) + " |", "|---|" + "---|" * len(metrics)]
    for t in order: L.append(f"| {t} | " + " | ".join(f"{final[t][m]['mean'][0]:.3f} [{final[t][m]['mean'][1]:.3f}, {final[t][m]['mean'][2]:.3f}]" for m in metrics) + " |")
    L += ["", "## Per cell line (point estimates, DE_raw / DE_2w / DIR_raw / DIR_2w)", "", "| method | " + " | ".join(CELLS) + " |", "|---|" + "---|" * 4]
    for t in order: L.append(f"| {t} | " + " | ".join(" / ".join(f"{final[t][m][c][0]:.3f}" for m in ("DE_raw", "DE_2w", "DIR_raw", "DIR_2w")) for c in CELLS) + " |")
    for ref in REFS:
        L += ["", f"## Paired differences vs {ref} (mean over cell lines)", "", "| comparison | " + " | ".join(metrics) + " |", "|---|" + "---|" * len(metrics)]
        for k, v in diffs.items():
            if k.endswith(" - " + ref): L.append(f"| {k} | " + " | ".join(f"{v[m][0]:+.3f} [{v[m][1]:+.3f}, {v[m][2]:+.3f}]" for m in metrics) + " |")
    open(O + "/results.md", "w").write("\n".join(L) + "\n"); print("\n".join(L)); print("SCORE DONE")

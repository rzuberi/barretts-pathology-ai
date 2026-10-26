"""Paper C item 5 scoring: the thinking-mode runs and their comparators on the *same* subsample of test perturbations
(one in four per cell line, seed 0; $O/test_sub_perts.json). Raw and two-way corrected auc_per_gene, 2,000 paired bootstrap
resamples of perturbations, plus paired differences against the retrieval sentence the LLM was shown (kNN k10) and against
the non-thinking SFT arms. Out: $O/results_sub.json, $O/results_sub.md.
"""
import json, os
import numpy as np, pandas as pd
from multiprocessing import Pool
from pc_common import *

NB, SEED = 2000, 0
TAGS = {"think_r": X + "/preds/pc2_think_r__test_sub.jsonl", "think_plain": X + "/preds/pc2_think_plain__test_sub.jsonl",
        "zs_nothink": F + "/preds/zs__test.jsonl", "sft_r": X + "/preds/sft_r__checkpoint-2000__test.jsonl",
        "F_sft": F + "/preds/sft_all__checkpoint-3000__test.jsonl",
        "knn_mean_k10": X + "/preds/knn_mean_k10__test.jsonl", "knn_mean_k20": X + "/preds/knn_mean_k20__test.jsonl",
        "shiftknn_k20": O + "/preds/shiftknn_k20__test.jsonl", "listsize_only": O + "/preds/listsize_only__test.jsonl"}
REFS = ["knn_mean_k10", "think_r", "sft_r"]
SUB = json.load(open(O + "/test_sub_perts.json"))


def run_cell(c):
    df = test_frame(c); keep = set(SUB[c]); m = df.pert.isin(keep).values
    df = df[m].reset_index(drop=True); perts = df.pert.values; genes = df.gene.values
    pid = np.unique(perts); rng = np.random.default_rng(SEED + CELLS.index(c))
    _, pidx = np.unique(perts, return_inverse=True)
    cnt = np.stack([np.bincount(rng.integers(0, len(pid), len(pid)), minlength=len(pid)) for _ in range(NB)])
    Wall = cnt[:, pidx].astype(np.float32)
    dmask = (df.de.values == 1) & (df["dir"].values >= 0); ydir = (df["dir"].values[dmask] == 1).astype(int); Wdir = Wall[:, dmask]
    Gde = GeneAUC(genes, df.de.values); Gdir = GeneAUC(genes[dmask], ydir)
    res, boot = {}, {}
    for tag, path in TAGS.items():
        if not os.path.exists(path): print("missing", tag, path, flush=True); continue
        sde, sdir = scores_from_preds(path, df, c); sd = sdir[dmask]; pdm = perts[dmask]; gdm = genes[dmask]
        variants = {"DE_raw": (Gde, sde, Wall), "DE_2w": (Gde, twoway_residual(sde, perts, genes), Wall),
                    "DIR_raw": (Gdir, sd, Wdir), "DIR_2w": (Gdir, twoway_residual(sd, pdm, gdm), Wdir)}
        res[tag] = {}; boot[tag] = {}
        for k, (G, s, W) in variants.items():
            C = G.build(s); point = G.auc(C, np.ones(len(s)))
            b = np.array([G.auc(C, W[i]) for i in range(NB)])
            res[tag][k] = [point, float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]; boot[tag][k] = b.astype(np.float32)
        print(c, tag, {k: round(v[0], 3) for k, v in res[tag].items()}, flush=True)
    return c, res, boot


if __name__ == "__main__":
    with Pool(4) as pool: out = pool.map(run_cell, CELLS)
    res = {c: r for c, r, _ in out}; boot = {c: b for c, _, b in out}
    tags = [t for t in TAGS if all(t in res[c] for c in CELLS)]; metrics = ["DE_raw", "DE_2w", "DIR_raw", "DIR_2w"]
    final = {t: {m: {"mean": [float(np.mean([res[c][t][m][0] for c in CELLS]),),
                              *np.percentile(np.mean([boot[c][t][m] for c in CELLS], 0), [2.5, 97.5]).tolist()],
                     **{c: res[c][t][m] for c in CELLS}} for m in metrics} for t in tags}
    diffs = {}
    for ref in REFS:
        if ref not in tags: continue
        for t in tags:
            if t == ref: continue
            diffs[f"{t} - {ref}"] = {m: [final[t][m]["mean"][0] - final[ref][m]["mean"][0],
                                         *np.percentile(np.mean([boot[c][t][m] - boot[c][ref][m] for c in CELLS], 0), [2.5, 97.5]).tolist()]
                                     for m in metrics}
    json.dump({"results": final, "diffs": diffs, "n_perts": {c: len(SUB[c]) for c in CELLS}}, open(O + "/results_sub.json", "w"), indent=1)
    L = ["# Paper C item 5: thinking-mode Qwen3-8B on the 1-in-4 perturbation subsample (same pairs for every row)", "",
         "| method | " + " | ".join(metrics) + " |", "|---|" + "---|" * len(metrics)]
    for t in sorted(tags, key=lambda t: -final[t]["DE_raw"]["mean"][0]):
        L.append(f"| {t} | " + " | ".join(f"{final[t][m]['mean'][0]:.3f} [{final[t][m]['mean'][1]:.3f}, {final[t][m]['mean'][2]:.3f}]" for m in metrics) + " |")
    L += ["", "## Paired differences", "", "| comparison | " + " | ".join(metrics) + " |", "|---|" + "---|" * len(metrics)]
    for k, v in diffs.items():
        L.append(f"| {k} | " + " | ".join(f"{v[m][0]:+.3f} [{v[m][1]:+.3f}, {v[m][2]:+.3f}]" for m in metrics) + " |")
    open(O + "/results_sub.md", "w").write("\n".join(L) + "\n"); print("\n".join(L)); print("SCORE_SUB DONE")

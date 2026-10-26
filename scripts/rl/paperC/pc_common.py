"""Paper C shared code: paths, loaders, fast weighted per-gene AUROC (for a paired bootstrap over perturbations), and the
shift-/list-size-corrected transforms of a score. Public PerturbQA data only.

Metric notes
- Official metric: PerturbQA `auc_per_gene` = AUROC within each target gene, mean over genes with both classes. Our fast
  version reproduces it exactly at unit weights and adds per-perturbation multiplicity weights so that a bootstrap over
  perturbations can be scored without rebuilding the groups.
- Corrections of a score s(p, g):
  * `wp`  : within-perturbation rank, s -> rank of s among p's scored test pairs, scaled to (0, 1). Removes any per-knockdown
            shift (DIR) or strength / list-size effect (DE) whatever its scale.
  * `res` : s (as a global percentile rank) with a *fixed covariate* partialled out by OLS: a predicted per-knockdown
            up-fraction for DIR, log list size for DE. The covariate is label-free on test.
  * `cen` : s minus its own per-perturbation mean (the decomposition of R3's check).
"""
import json, os
import numpy as np, pandas as pd
from collections import defaultdict
from scipy import sparse
from scipy.stats import rankdata

B = "/mnt/scratche/slow/fmlab/zuberi01"
Q = B + "/perturbqa/PerturbQA/perturbqa/datasets"
X = B + "/analysis/rl_explore/pert"          # exploration outputs (kNN preds, downloads)
F = B + "/analysis/rl_followups/F"           # Task F outputs (LLM preds)
O = B + "/analysis/paperC"                    # this paper's outputs
D = X + "/downloads"; GS = D + "/gene_summary/gene_summary"; KG = D + "/kg/perturbqa/datasets/kg"
CELLS = ["k562", "rpe1", "hepg2", "jurkat"]
os.makedirs(O + "/preds", exist_ok=True)


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f]


def write_preds(tag, rows, split="test"):
    """rows: dicts with cell, pert, gene, label, de, dir, pde (P(change)), pdir (P(up | change)). Task F pred format."""
    with open(f"{O}/preds/{tag}__{split}.jsonl.part", "w") as f:
        for r in rows:
            pde = min(max(float(r["pde"]), 1e-3), 1 - 1e-3); pdir = min(max(float(r["pdir"]), 0.0), 1.0)
            f.write(json.dumps({"cell": r["cell"], "pert": r["pert"], "gene": r["gene"], "label": r["label"], "de": r["de"], "dir": r["dir"],
                                "p_up": pde * pdir, "p_down": pde * (1 - pdir), "p_none": 1 - pde, "mass": 1.0, "top": ""}) + "\n")
    os.replace(f"{O}/preds/{tag}__{split}.jsonl.part", f"{O}/preds/{tag}__{split}.jsonl")


def load_cell(c):
    de = pd.read_csv(f"{Q}/{c}-de.csv"); dr = pd.read_csv(f"{Q}/{c}-dir.csv")
    dmap = {(a, b): int(l) for a, b, l in zip(dr.pert, dr.gene, dr.label)}   # 1 = up, 0 = down
    return de, dr, dmap


def test_frame(c):
    """One row per test pair with keys and labels, in the order of the exploration kNN preds (full test set)."""
    rows = [r for r in read_jsonl(f"{X}/preds/knn_mean_k20__test.jsonl") if r["cell"] == c]
    df = pd.DataFrame({"pert": [r["pert"] for r in rows], "gene": [r["gene"] for r in rows], "de": [int(r["de"]) for r in rows],
                       "dir": [(-1 if r["dir"] is None else int(r["dir"])) for r in rows]})
    return df


def scores_from_preds(path, df, c):
    """Align a preds file to df's (pert, gene) order; returns (de_score, dir_score)."""
    m = {(r["pert"], r["gene"]): (r["p_up"] + r["p_down"], r["p_up"] / max(r["p_up"] + r["p_down"], 1e-12))
         for r in read_jsonl(path) if r["cell"] == c}
    de = np.array([m[(p, g)][0] for p, g in zip(df.pert, df.gene)]); dr = np.array([m[(p, g)][1] for p, g in zip(df.pert, df.gene)])
    return de, dr


def auc_per_gene_official(keys, pred, true):
    """Verbatim logic of perturbqa.eval.auc_per_gene (kept local so scoring does not import the repo)."""
    from sklearn.metrics import roc_auc_score
    pred = np.asarray(pred); true = np.asarray(true); groups = defaultdict(list)
    for i, k in enumerate(keys): groups[k[1]].append(i)
    out = []
    for idx in groups.values():
        ts = true[idx]
        if ts.sum() == 0 or ts.sum() == len(ts): continue
        out.append(roc_auc_score(ts, pred[idx]))
    return float(np.mean(out))


class GeneAUC:
    """Per-gene AUROC with per-pair weights (multiplicity of the pair's perturbation in a bootstrap resample).
    auc = mean over genes with weighted positives and negatives of  sum_{i+, j-} w_i w_j [s_i > s_j] (+0.5 ties) / (W+ W-)."""
    def __init__(self, genes, y):
        self.y = np.asarray(y).astype(int); self.gid, self.gidx = np.unique(np.asarray(genes), return_inverse=True)
        self.ng = len(self.gid); self.n = len(self.y)
        self.groups = defaultdict(list)
        for i, g in enumerate(self.gidx): self.groups[g].append(i)

    def build(self, s):
        s = np.asarray(s, dtype=float); rows, cols, vals = [], [], []
        for g, idx in self.groups.items():
            idx = np.asarray(idx); P = idx[self.y[idx] == 1]; N = idx[self.y[idx] == 0]
            if len(P) == 0 or len(N) == 0: continue
            c = (s[P][:, None] > s[N][None, :]).astype(float) + 0.5 * (s[P][:, None] == s[N][None, :])
            r, k = np.nonzero(c); rows.append(P[r]); cols.append(N[k]); vals.append(c[r, k])
        if rows:
            rows = np.concatenate(rows); cols = np.concatenate(cols); vals = np.concatenate(vals)
        C = sparse.csr_matrix((vals, (rows, cols)), shape=(self.n, self.n))
        return C

    def auc(self, C, w):
        """w: per-pair weight (float array, length n)."""
        v = C @ w; num = np.bincount(self.gidx, weights=w * v * self.y, minlength=self.ng)
        wp = np.bincount(self.gidx, weights=w * self.y, minlength=self.ng); wn = np.bincount(self.gidx, weights=w * (1 - self.y), minlength=self.ng)
        den = wp * wn; ok = den > 0
        return float(np.mean(num[ok] / den[ok]))


def boot_weights(perts, n_boot, seed):
    """Multiplicity of each pair's perturbation under n_boot resamples of perturbations with replacement. Returns (n_boot, n_pairs) int16."""
    pid, pidx = np.unique(np.asarray(perts), return_inverse=True); rng = np.random.default_rng(seed); P = len(pid)
    W = np.zeros((n_boot, len(perts)), dtype=np.int16)
    for b in range(n_boot):
        cnt = np.bincount(rng.integers(0, P, P), minlength=P); W[b] = cnt[pidx]
    return W


# ---------- corrections
def within_pert_rank(s, perts):
    s = np.asarray(s, dtype=float); out = np.zeros_like(s); perts = np.asarray(perts)
    for p in np.unique(perts):
        idx = np.flatnonzero(perts == p); out[idx] = rankdata(s[idx], method="average") / (len(idx) + 1)
    return out


def global_rank(s):
    s = np.asarray(s, dtype=float); return rankdata(s, method="average") / (len(s) + 1)


def partial_out(s, cov):
    """Residual of the global percentile rank of s after OLS on the global rank of cov (with intercept); rank-invariant on both sides."""
    r = global_rank(s); A = np.column_stack([np.ones(len(r)), global_rank(cov)])
    beta, *_ = np.linalg.lstsq(A, r, rcond=None); return np.round(r - A @ beta, 9)   # rounded: a constant score must residualise to exact ties


def centre_by_pert(s, perts):
    s = np.asarray(s, dtype=float); perts = np.asarray(perts); out = s.copy()
    for p in np.unique(perts):
        idx = np.flatnonzero(perts == p); out[idx] = s[idx] - s[idx].mean()
    return np.round(out, 9)


def centre_by_gene(s, genes):
    return centre_by_pert(s, genes)


def twoway_residual(s, perts, genes, n_iter=20):
    """Residual of the global percentile rank of s after an additive knockdown + gene model (alternating projections).
    A score that is constant within knockdowns (a shift-only predictor) or within genes (a gene prior) residualises to 0."""
    r = global_rank(s)
    for _ in range(n_iter):
        r = centre_by_pert(r, perts); r = centre_by_pert(r, genes)
    return r


def covariate_residual(s, cov, genes, n_iter=20):
    """Residual of the global rank of s after partialling out a fixed per-knockdown covariate (rank) and the gene effect,
    alternating until both are removed. Unlike twoway_residual, only the covariate's share of the knockdown effect is removed."""
    r = global_rank(s)
    for _ in range(n_iter):
        r = partial_out(r, cov); r = centre_by_pert(r, genes)
    return r

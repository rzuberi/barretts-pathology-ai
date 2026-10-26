"""Paper C, item 2: re-implementation of CORE-Voting (Yuan et al., arXiv 2606.01042, Sec. 3 and App. C) on PerturbQA.
No code was released, so this follows the paper's formulas:
  signature  phi(p): over KG-grounded genes h, distance-identity features 1[dist:h:d(p,h)] (d <= 6) and a proximity-rate
             feature rho(d(p,h)) where rho(d) = positive rate at graph distance d in the support (training) labels;
  similarity sim(p,p') = max(0, cos(phi(p), phi(p'))), vote weight w = sim ** eta, eta = 0.5; all support perturbations with an
             observed label on the gene vote (App. C.3);
  contrast   s_c(p,g) = V+ / (V+ + V-), fallback = support positive rate of g (then the global rate);
  anchor     r_g(p') = 2y-1 over support labels; top H = 10 genes by cos(r_g, r_h) (self included, sim 1);
             s_a(p,g) = sum_h sim(g,h) s_c(p,h) / sum_h sim(g,h);
  final      s = (1-alpha) s_c + alpha s_a, alpha = 0.5.
Differences from the paper that cannot be avoided: their graph is ReasonKG (VCWorld + PrimeKG + HetioNet + PertKGE + BioKG);
ours is PerturbQA's own KG (STRING, BioPlex, CORUM, GO, Reactome; hubs > 1000 dropped as in PerturbQA), with complexes, GO
terms and pathways as intermediate nodes. Direction (DIR) is scored with the same machinery on the direction labels, which the
paper does not do. Variants written: corevote_full, corevote_contrast (alpha = 0), corevote_anchor (alpha = 1) and
corevote_meansim (contrast + anchor with our kNN's mean similarity in place of the path signature).
"""
import json, sys, time
import numpy as np, pandas as pd
from collections import defaultdict
from scipy import sparse
from scipy.sparse.csgraph import dijkstra
from pc_common import *
from pc_kg import feat, features, sims   # same KG neighbour sets / similarity as the kNN

ETA, ALPHA, H, DMAX = 0.5, 0.5, 10, 6
t0 = time.time()
# ---- heterogeneous graph: genes + complexes + GO terms + pathways; undirected; hubs dropped as in PerturbQA's notebook
node = {}; edges = set()
def nid(x): return node.setdefault(x, len(node))
string = json.load(open(KG + "/string.json"))
for g, rels in string.items():
    if len(rels) <= 1000:
        for r in rels: edges.add((nid("g:" + g), nid("g:" + r[0])))
for g, rels in json.load(open(KG + "/bioplex.json")).items():
    for r in rels: edges.add((nid("g:" + g), nid("g:" + r[0])))
g2c, _ = json.load(open(KG + "/corum.json"))
for g, rels in g2c.items():
    for r in rels: edges.add((nid("g:" + g), nid("c:" + r[0])))
g2go, go2g, _ = json.load(open(KG + "/go.json")); big = {t for t, gs in go2g.items() if len(gs) > 1000}
for g, rels in g2go.items():
    for r in rels:
        if r[0] not in big: edges.add((nid("g:" + g), nid("t:" + r[0])))
g2r, _ = json.load(open(KG + "/reactome.json"))
for g, rels in g2r.items():
    for r in rels: edges.add((nid("g:" + g), nid("p:" + r[0])))
E = np.array(list(edges)); N = len(node)
A = sparse.csr_matrix((np.ones(len(E)), (E[:, 0], E[:, 1])), shape=(N, N)); A = ((A + A.T) > 0).astype(np.float32)
print("graph nodes", N, "edges", A.nnz // 2, f"{time.time() - t0:.0f}s", flush=True)

PRED = defaultdict(list); INFO = {}
for c in CELLS:
    de, dr, dmap = load_cell(c); df = test_frame(c)
    tr = de[de.split == "train"]; te = de[de.split == "test"]
    perts = sorted(set(de.pert)); pidx = {p: i for i, p in enumerate(perts)}
    genes = sorted(set(de.gene) | set(de.pert)); gidx = {g: i for i, g in enumerate(genes)}
    trainp = sorted(set(tr.pert)); testp = sorted(set(te.pert)); itr = np.array([pidx[p] for p in trainp]); ite = np.array([pidx[p] for p in testp])
    # ---- distances from every perturbation gene to every gene, capped at DMAX
    grounded_p = np.array([("g:" + p) in node for p in perts]); grounded_g = np.array([("g:" + g) in node for g in genes])
    src = [node["g:" + p] for p in perts if ("g:" + p) in node]
    Dfull = dijkstra(A, directed=False, unweighted=True, indices=src, limit=DMAX)   # inf beyond DMAX
    gcol = np.array([node.get("g:" + g, -1) for g in genes])
    Dist = np.full((len(perts), len(genes)), np.inf, dtype=np.float32)
    Dist[np.ix_(grounded_p, grounded_g)] = Dfull[:, gcol[grounded_g]]
    Dist[~grounded_p] = np.inf
    for p in perts:   # self distance 0 -> treat as 1 (the pert gene itself)
        if p in gidx: Dist[pidx[p], gidx[p]] = max(Dist[pidx[p], gidx[p]], 1.0)
    Db = np.where(np.isinf(Dist), DMAX + 1, Dist).astype(np.int8)   # buckets 1..6, 7 = unreachable
    # ---- rho(d): support positive rate by distance bucket
    dtr = Db[[pidx[p] for p in tr.pert], [gidx[g] for g in tr.gene]]; ytr = tr.label.values
    glob = float(ytr.mean()); rho = np.array([glob] + [float(ytr[dtr == d].mean()) if (dtr == d).sum() >= 20 else glob for d in range(1, DMAX + 2)], dtype=np.float32)
    INFO[c] = {"rho_by_distance_1..6_unreach": rho[1:].round(3).tolist(), "n_pairs_by_distance": [int((dtr == d).sum()) for d in range(1, DMAX + 2)],
               "perts_in_graph": int(grounded_p.sum()), "perts": len(perts), "test_perts_in_graph": int(grounded_p[ite].sum())}
    # ---- signature cosine: identity block (sum over d of (D==d)(D==d)^T) + proximity block (rho(D) rho(D)^T)
    sub = np.concatenate([ite, itr]); Dsub = Db[sub]
    dot = np.zeros((len(sub), len(sub)), dtype=np.float32)
    for d in range(1, DMAX + 1):
        M = (Dsub == d).astype(np.float32); dot += M @ M.T
    R = rho[Dsub.astype(int)]; R[Dsub == DMAX + 1] = rho[DMAX + 1]; R[~grounded_p[sub]] = 0.0; dot += R @ R.T
    nrm = np.sqrt(np.maximum(np.diag(dot), 1e-12)); cos = dot / nrm[:, None] / nrm[None, :]
    Wsig = np.maximum(cos[: len(ite), len(ite):], 0.0) ** ETA; Wsig[:, ~grounded_p[itr]] = 0.0; Wsig[~grounded_p[ite], :] = 0.0
    # mean-similarity variant (our kNN's similarity, all supports weighted)
    Xkg, T = features(perts); S = sims(Xkg, T); Wms = np.maximum(S[np.ix_(ite, itr)], 0.0) ** ETA
    print(c, "signature done", f"{time.time() - t0:.0f}s", "median #supports with w>0 per test pert", np.median((Wsig > 0).sum(1)), flush=True)
    # ---- support label matrices (train perts x genes)
    tpi = {p: i for i, p in enumerate(trainp)}
    r_ = np.array([tpi[p] for p in tr.pert]); c_ = np.array([gidx[g] for g in tr.gene]); y_ = tr.label.values.astype(np.float32)
    Lpos = sparse.csr_matrix((y_, (r_, c_)), shape=(len(trainp), len(genes))); Lneg = sparse.csr_matrix((1 - y_, (r_, c_)), shape=(len(trainp), len(genes)))
    up = np.array([dmap.get((p, g), -1) for p, g in zip(tr.pert, tr.gene)]); isde = (y_ == 1) & (up >= 0)
    Lup = sparse.csr_matrix(((up == 1)[isde].astype(np.float32), (r_[isde], c_[isde])), shape=Lpos.shape)
    Ldn = sparse.csr_matrix(((up == 0)[isde].astype(np.float32), (r_[isde], c_[isde])), shape=Lpos.shape)
    prior_de = np.asarray(Lpos.sum(0)).ravel() / np.maximum(np.asarray((Lpos + Lneg).sum(0)).ravel(), 1); cnt_de = np.asarray((Lpos + Lneg).sum(0)).ravel()
    prior_de = np.where(cnt_de > 0, prior_de, glob)
    gup = float(Lup.sum() / max(Lup.sum() + Ldn.sum(), 1)); cnt_dir = np.asarray((Lup + Ldn).sum(0)).ravel()
    prior_up = np.where(cnt_dir > 0, np.asarray(Lup.sum(0)).ravel() / np.maximum(cnt_dir, 1), gup)
    # ---- anchors from support response profiles
    def anchors(Lp, Ln):
        Rg = (Lp - Ln).T.tocsr()   # genes x supports, +1 / -1 / 0
        n = np.sqrt(np.asarray(Rg.multiply(Rg).sum(1)).ravel()); n[n == 0] = 1
        G = (sparse.diags(1 / n) @ Rg @ Rg.T @ sparse.diags(1 / n)).toarray(); G = np.maximum(G, 0.0); np.fill_diagonal(G, 1.0)
        G[n == 1, :] = 0  # genes with no profile: no anchors (fallback to contrast)
        top = np.argpartition(-G, H, axis=1)[:, :H]; Aw = np.zeros_like(G)
        rows = np.repeat(np.arange(len(G)), H); Aw[rows, top.ravel()] = G[rows, top.ravel()]
        s = Aw.sum(1); Aw[s > 0] /= s[s > 0][:, None]
        return Aw   # genes x genes, row = weights over anchors
    Ade = anchors(Lpos, Lneg); Adir = anchors(Lup, Ldn)
    def contrast(W, Lp, Ln, prior):
        Vp = np.asarray(W @ Lp.toarray() if sparse.issparse(Lp) else W @ Lp); Vn = np.asarray(W @ Ln.toarray()); tot = Vp + Vn
        return np.where(tot > 0, Vp / np.maximum(tot, 1e-12), prior[None, :])
    out = {}
    for nm, W in (("sig", Wsig), ("ms", Wms)):
        Sc = contrast(W, Lpos, Lneg, prior_de); Sa = Sc @ Ade.T; noanc = Ade.sum(1) == 0; Sa[:, noanc] = Sc[:, noanc]
        Dc = contrast(W, Lup, Ldn, prior_up); Da = Dc @ Adir.T; noanc = Adir.sum(1) == 0; Da[:, noanc] = Dc[:, noanc]
        out[nm] = {"contrast": (Sc, Dc), "anchor": (Sa, Da), "full": ((1 - ALPHA) * Sc + ALPHA * Sa, (1 - ALPHA) * Dc + ALPHA * Da)}
    tp = {p: i for i, p in enumerate(testp)}; ri = np.array([tp[p] for p in df.pert]); ci = np.array([gidx[g] for g in df.gene])
    for nm, var, tag in (("sig", "full", "corevote_full"), ("sig", "contrast", "corevote_contrast"), ("sig", "anchor", "corevote_anchor"), ("ms", "full", "corevote_meansim")):
        Sde, Sdir = out[nm][var]
        PRED[tag] += [{"cell": c, "pert": p, "gene": g, "label": "x", "de": int(d), "dir": (None if di < 0 else int(di)), "pde": Sde[a, b], "pdir": Sdir[a, b]}
                      for p, g, d, di, a, b in zip(df.pert, df.gene, df.de, df["dir"], ri, ci)]
    # quick point estimate
    Sde, Sdir = out["sig"]["full"]; keys = list(zip(df.pert, df.gene))
    INFO[c]["DE_full"] = auc_per_gene_official(keys, Sde[ri, ci], df.de.values)
    Sc, _ = out["sig"]["contrast"]; INFO[c]["DE_contrast"] = auc_per_gene_official(keys, Sc[ri, ci], df.de.values)
    print(c, INFO[c], f"{time.time() - t0:.0f}s", flush=True)

for tag, rows in PRED.items(): write_preds(tag, rows)
json.dump(INFO, open(O + "/corevote_info.json", "w"), indent=1); print("COREVOTE DONE")

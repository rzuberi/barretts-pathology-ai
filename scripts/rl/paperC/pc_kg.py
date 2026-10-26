"""KG neighbour-set features, gene-summary TF-IDF and the mean similarity: identical construction to explore/pert/knn_baseline.py."""
import json
import numpy as np
from collections import defaultdict
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from pc_common import KG, GS

# ---- KG neighbour sets and summaries: identical construction to explore/pert/knn_baseline.py
feat = defaultdict(set)
string = json.load(open(KG + "/string.json"))
for g, rels in string.items():
    if len(rels) <= 1000: feat[g] |= {"S:" + r[0] for r in rels}
for g, rels in json.load(open(KG + "/bioplex.json")).items(): feat[g] |= {"B:" + r[0] for r in rels}
g2c, _ = json.load(open(KG + "/corum.json"))
for g, rels in g2c.items(): feat[g] |= {"C:" + r[0] for r in rels}
g2go, go2g, _ = json.load(open(KG + "/go.json")); big = {t for t, gs in go2g.items() if len(gs) > 1000}
for g, rels in g2go.items(): feat[g] |= {"G:" + r[0] for r in rels if r[0] not in big}
g2r, _ = json.load(open(KG + "/reactome.json"))
for g, rels in g2r.items(): feat[g] |= {"R:" + r[0] for r in rels}
desc_pert = json.load(open(GS + "/desc_pert.json"))


def features(perts):
    vocab = {}; rows, cols = [], []
    for i, p in enumerate(perts):
        for f in feat.get(p, ()): rows.append(i); cols.append(vocab.setdefault(f, len(vocab)))
    Xkg = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(perts), max(1, len(vocab))))
    T = TfidfVectorizer(stop_words="english", min_df=2, sublinear_tf=True).fit_transform([desc_pert.get(p, "") for p in perts])
    return Xkg, T


def sims(Xkg, T):
    inter = (Xkg @ Xkg.T).toarray(); n = np.asarray(Xkg.sum(1)).ravel(); union = n[:, None] + n[None, :] - inter
    kg = np.where(union > 0, inter / np.maximum(union, 1), 0.0); tx = (T @ T.T).toarray()
    return (kg + tx) / 2   # diagonal kept (=1) for the kernel; zeroed where needed



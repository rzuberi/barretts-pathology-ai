"""Explore/pert CPU baselines + retrieval sentences. pathology env.
Per cell line: similarity between perturbed genes from PerturbQA's KG (Jaccard over STRING/BioPlex/CORUM/GO/Reactome neighbour sets),
TF-IDF cosine over PerturbQA gene summaries, and their mean. For a (pert p, gene g) query, the k most similar *training* perturbations
that measured g give the DE / direction score (PerturbQA retrieval option "same gene, close perturbation"); 'strength' = mean training
DE rate of the k most similar perturbations overall; oracle = p's true DE rate on test (diagnostic only).
Writes preds/knn_*__{val,test}.jsonl in Task F format and data/{train_all,grpo_all,val,test}.jsonl with a 'ret' sentence (k=10, mean sim)."""
import json, os, random, sys
import numpy as np, pandas as pd
from collections import defaultdict
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
B = "/mnt/scratche/slow/fmlab/zuberi01"; Q = B + "/perturbqa/PerturbQA/perturbqa/datasets"; F = B + "/analysis/rl_followups/F"; O = B + "/analysis/rl_explore/pert"
D = O + "/downloads"; GS = D + "/gene_summary/gene_summary"; KG = D + "/kg/perturbqa/datasets/kg"; CELLS = ["k562", "rpe1", "hepg2", "jurkat"]
os.makedirs(O + "/preds", exist_ok=True); os.makedirs(O + "/data", exist_ok=True)
# ---- knowledge-graph neighbour sets (hubs with > 1000 neighbours dropped, as in PerturbQA's notebook)
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
desc_pert = json.load(open(GS + "/desc_pert.json")); print("kg genes", len(feat), "summaries", len(desc_pert), flush=True)
def sims(perts):
    """(n, n) similarity matrices: kg Jaccard, text cosine, mean"""
    vocab = {}; rows, cols = [], []
    for i, p in enumerate(perts):
        for f in feat.get(p, ()): rows.append(i); cols.append(vocab.setdefault(f, len(vocab)))
    X = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(perts), max(1, len(vocab))))
    inter = (X @ X.T).toarray(); n = np.asarray(X.sum(1)).ravel(); union = n[:, None] + n[None, :] - inter
    kg = np.where(union > 0, inter / np.maximum(union, 1), 0.0)
    T = TfidfVectorizer(stop_words="english", min_df=2, sublinear_tf=True).fit_transform([desc_pert.get(p, "") for p in perts]); tx = (T @ T.T).toarray()
    for m in (kg, tx): np.fill_diagonal(m, 0.0)
    return {"kg": kg, "text": tx, "mean": (kg + tx) / 2}
rnd = random.Random(20261016); SETS = {s: [json.loads(l) for l in open(f"{F}/data/{s}.jsonl")] for s in ("train_all", "grpo_all", "val", "test")}
RET = {}; PRED = defaultdict(list); KS = (5, 10, 20)
for c in CELLS:
    de = pd.read_csv(f"{Q}/{c}-de.csv"); dr = pd.read_csv(f"{Q}/{c}-dir.csv"); dmap = {(a, b): l for a, b, l in zip(dr.pert, dr.gene, dr.label)}
    trp = sorted(de[de.split == "train"].pert.unique()); rnd.shuffle(trp); valp = set(trp[: max(1, len(trp) // 10)])  # Task F's val perturbations (same RNG stream)
    assert {r["pert"] for r in SETS["val"] if r["cell"] == c} <= valp, c
    lab = {}; measured = defaultdict(list)
    for a, b, l, s in zip(de.pert, de.gene, de.label, de.split):
        if s != "train": continue
        if l == 1 and (a, b) not in dmap: continue
        lab[(a, b)] = 0 if l == 0 else (1 if dmap[(a, b)] == 1 else -1); measured[b].append(a)
    derate = {p: v for p, v in de[de.split == "train"].groupby("pert").label.mean().items()}
    perts = sorted(set(de.pert)); pidx = {p: i for i, p in enumerate(perts)}; S = sims(perts)
    trainp = np.array([p in derate and p not in valp for p in perts]); allp = np.array([p in derate for p in perts])
    te = de[de.split == "test"]; orac_de = te.groupby("pert").label.mean().to_dict()
    orac_dir = {p: np.mean([dmap.get((p, g), 0) == 1 for g, l in zip(gg, ll) if l == 1] or [0.5]) for p, (gg, ll) in te.groupby("pert").agg({"gene": list, "label": list}).iterrows()} if False else {}
    for p, grp in te.groupby("pert"):
        ups = [dmap[(p, g)] == 1 for g, l in zip(grp.gene, grp.label) if l == 1 and (p, g) in dmap]; orac_dir[p] = float(np.mean(ups)) if ups else 0.5
    print(c, "perts", len(perts), "train-part", trainp.sum(), "all-train", allp.sum(), flush=True)
    for s, rows in SETS.items():
        pool = allp if s == "test" else trainp
        for r in rows:
            if r["cell"] != c: continue
            i = pidx[r["pert"]]; g = r["gene"]; out = {}
            cand = np.array([pidx[q] for q in measured.get(g, ()) if pool[pidx[q]] and q != r["pert"]], dtype=int)
            for sn, M in S.items():
                srow = M[i] * pool; srow[i] = 0; poolidx = np.flatnonzero(pool); poolidx = poolidx[poolidx != i]
                top_all = poolidx[np.argsort(-srow[poolidx])[:max(KS)]]
                cs = M[i, cand] if len(cand) else np.array([]); order = np.argsort(-cs) if len(cand) else []
                for k in KS:
                    sel = cand[order[:k]]; w = np.maximum(M[i, sel], 1e-6) if len(sel) else np.array([])
                    ls = np.array([lab[(perts[j], g)] for j in sel])
                    strength = float(np.average([derate[perts[j]] for j in top_all[:k]], weights=np.maximum(srow[top_all[:k]], 1e-6)))
                    pair = float(np.average(ls != 0, weights=w)) if len(sel) else strength
                    dirs = float(np.average(ls[ls != 0] == 1, weights=w[ls != 0])) if (ls != 0).sum() else 0.5
                    out[f"{sn}_k{k}"] = (pair, dirs, strength)
                    if sn == "mean" and k == 10:
                        top5 = ", ".join(perts[j] for j in sel[:5]); n = len(sel); a = int((ls == 1).sum()); b = int((ls == -1).sum()); z = int((ls == 0).sum())
                        RET[(c, r["pert"], g)] = (f"Of the {k} knockdowns in the training data most similar to {r['pert']} that also measured {g}" + (f" (closest: {top5}), {g} went up after {a}, down after {b} and was unchanged after {z} of the {n}." if n else f", there were none.") +
                                                 f" The 10 most similar training knockdowns overall changed {100 * strength:.0f}% of the genes measured.")
            if s in ("val", "test"):
                base = {k: r[k] for k in ("cell", "pert", "gene", "label", "de", "dir")}
                def emit(tag, pde, pdir):
                    pde = min(max(pde, 1e-3), 1 - 1e-3); PRED[(tag, s)].append({**base, "p_up": pde * pdir, "p_down": pde * (1 - pdir), "p_none": 1 - pde, "mass": 1.0, "top": ""})
                for key, (pair, dirs, strength) in out.items():
                    emit(f"knn_{key}", pair, dirs); emit(f"strength_{key}", strength, dirs); emit(f"blend_{key}", 0.5 * pair + 0.5 * strength, dirs)
                if s == "test": emit("oracle_strength", orac_de[r["pert"]], orac_dir[r["pert"]])
for (tag, s), rows in PRED.items():
    with open(f"{O}/preds/{tag}__{s}.jsonl", "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
for s, rows in SETS.items():
    with open(f"{O}/data/{s}.jsonl.part", "w") as f:
        for r in rows: f.write(json.dumps({**r, "ret": RET[(r["cell"], r["pert"], r["gene"])]}) + "\n")
    os.replace(f"{O}/data/{s}.jsonl.part", f"{O}/data/{s}.jsonl")
print("example ret:", RET[(SETS["test"][0]["cell"], SETS["test"][0]["pert"], SETS["test"][0]["gene"])]); print("KNN DONE", len(PRED), "pred files")

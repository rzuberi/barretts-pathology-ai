"""Standalone: rank-average combo of p1 token probability and the final-layer err probe (ranks on the population, no labels), pred files + AUROC."""
import json, os, sys, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")); from common import *
O = B + "/analysis/rl_explore/probe"; A0 = B + "/analysis/rl_abstain"; R = B + "/analysis/rl_grader"; RNG = np.random.RandomState(20261016)
def load(f): return {json.loads(l)["id"]: json.loads(l) for l in open(f)}
out = {}
for L in (17, 34):
    for s in ("pop", "test485"):
        a = load(f"{A0}/preds/p1best_conf__{s}.jsonl"); b = load(f"{O}/preds/probe_err_L{L}_lr__{s}.jsonl"); ids = sorted(a)
        ra = np.argsort(np.argsort([a[i].get("p_grade") or 0 for i in ids])); rb = np.argsort(np.argsort([b[i]["p_grade"] for i in ids])); sc = (ra + rb) / 2 / len(ids)
        with open(f"{O}/preds/combo_err_L{L}__{s}.jsonl", "w") as f:
            for i, v in zip(ids, sc): d = {k: a[i][k] for k in a[i] if k not in ("raw", "samples")}; d["p_grade"] = float(v); f.write(json.dumps(d) + "\n")
        if s == "test485": out[L] = {i: float(v) for i, v in zip(ids, sc)}
lab = load(R + "/data/test485.jsonl"); ids = sorted(lab); hum = np.array([lab[i]["human"] for i in ids]); w = np.array([lab[i]["w"] for i in ids]); pats = np.array([lab[i]["anon_id"] for i in ids])
p1 = load(f"{A0}/preds/p1best_conf__test485.jsonl"); wrong = np.array([p1[i]["pred"] or "X" for i in ids]) != hum; tp = np.array([p1[i].get("p_grade") or 0 for i in ids])
def wauc(score, wrong, ww):
    pos, neg = ~wrong, wrong; o = np.argsort(score, kind="stable"); s, p_, n_, w_ = score[o], pos[o], neg[o], ww[o]; tot = 0.0; cum = 0.0; i = 0
    while i < len(s):
        j = i
        while j < len(s) and s[j] == s[i]: j += 1
        wp = (w_[i:j] * p_[i:j]).sum(); wn = (w_[i:j] * n_[i:j]).sum(); tot += wp * (cum + 0.5 * wn); cum += wn; i = j
    return tot / (ww[pos].sum() * ww[neg].sum())
up = np.unique(pats); grp = {p: np.where(pats == p)[0] for p in up}; BS = [np.concatenate([grp[p] for p in RNG.choice(up, len(up))]) for _ in range(2000)]
res = {}
for L, d in out.items():
    sc = np.array([d[i] for i in ids]); dd = [wauc(sc[ix], wrong[ix], w[ix]) - wauc(tp[ix], wrong[ix], w[ix]) for ix in BS]
    res[f"combo_err_L{L}"] = {"auroc": round(float(wauc(sc, wrong, w)), 4), "ci": [round(float(x), 4) for x in np.percentile([wauc(sc[ix], wrong[ix], w[ix]) for ix in BS], [2.5, 97.5])], "minus_tokenprob": {"point": round(float(wauc(sc, wrong, w) - wauc(tp, wrong, w)), 4), "ci": [round(float(x), 4) for x in np.percentile(dd, [2.5, 97.5])]}}
json.dump(res, open(O + "/combo_err_auroc.json", "w"), indent=1); print(res)

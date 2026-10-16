"""Follow-up A: calibration and error detection of existing confidence scores on the 479 human sections (weighted) and the n=50 slice.
Scores (system judged): jury vote fraction (jury consensus); RL grade-token prob (RL grade); RL 8-sample agreement (RL grade);
alpha=0.2 grade-token prob (its grade). Out: rl_followups/A/{results.json, reliability.png, risk_coverage.png}."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__)); from common import *
R = B + "/analysis/rl_grader"; A0 = B + "/analysis/rl_abstain"; O = B + "/analysis/rl_followups/A"; NB = 2000; RNG = np.random.RandomState(20261016)
lab = {json.loads(l)["id"]: json.loads(l) for l in open(R + "/data/test485.jsonl")}; ids = sorted(lab)
def load(f): return {json.loads(l)["id"]: json.loads(l) for l in open(f)}
p1 = load(A0 + "/preds/p1best_conf__test485.jsonl"); a2 = load(A0 + "/preds/alpha0.2__checkpoint-150__test485.jsonl")
hum = np.array([lab[i]["human"] for i in ids]); w = np.array([lab[i]["w"] for i in ids]); pats = np.array([lab[i]["anon_id"] for i in ids]); rnd = np.array([lab[i]["stratum"] == "random" for i in ids])
S_ = {"jury vote fraction": (np.array([lab[i]["label"] for i in ids]), np.array([lab[i]["jury_vf"] for i in ids])),
      "RL grade probability": (np.array([p1[i]["pred"] or "X" for i in ids]), np.array([p1[i].get("p_grade") or 0 for i in ids])),
      "RL 8-sample agreement": (np.array([p1[i]["pred"] or "X" for i in ids]), np.array([np.mean([s == p1[i]["pred"] for s in p1[i]["samples"]]) for i in ids])),
      "abstention model (α=0.2) grade probability": (np.array([a2[i]["pred"] or "X" for i in ids]), np.array([a2[i].get("p_grade") or 0 for i in ids]))}
def wauc(score, wrong, ww):
    """weighted P(score of a correct section > score of a wrong one), ties 1/2: high score should mean correct."""
    pos, neg = ~wrong, wrong
    if not pos.any() or not neg.any(): return np.nan
    o = np.argsort(score, kind="stable"); s, p_, n_, w_ = score[o], pos[o], neg[o], ww[o]
    tot = 0.0; cum_neg = 0.0; i = 0
    while i < len(s):
        j = i
        while j < len(s) and s[j] == s[i]: j += 1
        wp = (w_[i:j] * p_[i:j]).sum(); wn = (w_[i:j] * n_[i:j]).sum(); tot += wp * (cum_neg + 0.5 * wn); cum_neg += wn; i = j
    return tot / (ww[pos].sum() * ww[neg].sum())
def ece(score, correct, ww, nb=10):
    bins = np.minimum((score * nb).astype(int), nb - 1); e = 0.0; rows = []
    for b in range(nb):
        m = bins == b
        if not m.any(): continue
        conf = (score[m] * ww[m]).sum() / ww[m].sum(); acc = (correct[m] * ww[m]).sum() / ww[m].sum(); e += ww[m].sum() / ww.sum() * abs(acc - conf)
        rows.append({"bin": b, "n": int(m.sum()), "mean_conf": round(float(conf), 4), "accuracy": round(float(acc), 4)})
    return e, rows
up = np.unique(pats); grp = {p: np.where(pats == p)[0] for p in up}; BS = [np.concatenate([grp[p] for p in RNG.choice(up, len(up))]) for _ in range(NB)]
ci = lambda v: [round(float(np.nanpercentile(v, 2.5)), 4), round(float(np.nanpercentile(v, 97.5)), 4)]
res = {"n": len(ids), "n_random": int(rnd.sum()), "scores": {}}
for k, (g, sc) in S_.items():
    wrong = g != hum; r = {"error_rate_weighted": round(float((wrong * w).sum() / w.sum()), 4), "n_errors": int(wrong.sum())}
    r["auroc_weighted"] = round(float(wauc(sc, wrong, w)), 4); r["auroc_ci"] = ci([wauc(sc[ix], wrong[ix], w[ix]) for ix in BS])
    r["auroc_random50"] = round(float(wauc(sc[rnd], wrong[rnd], np.ones(rnd.sum()))), 4) if 0 < wrong[rnd].sum() < rnd.sum() else None
    r["errors_random50"] = int(wrong[rnd].sum()); r["frac_score_eq_1"] = round(float((sc >= 0.9999).mean()), 3)
    if k in ("jury vote fraction", "RL grade probability"):
        e, rows = ece(sc, (~wrong).astype(float), w); r["ece_weighted"] = round(float(e), 4); r["ece_ci"] = ci([ece(sc[ix], (~wrong[ix]).astype(float), w[ix])[0] for ix in BS]); r["reliability_bins"] = rows
    res["scores"][k] = r
gj, sj = S_["jury vote fraction"]; gr, sr = S_["RL grade probability"]; wj, wr = gj != hum, gr != hum
d = [wauc(sr[ix], wr[ix], w[ix]) - wauc(sj[ix], wj[ix], w[ix]) for ix in BS]
res["primary_auroc_RLprob_minus_juryvf"] = {"point": round(float(wauc(sr, wr, w) - wauc(sj, wj, w)), 4), "ci": ci(d)}
json.dump(res, open(O + "/results.json", "w"), indent=1)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(4.6, 4.4))
for k, col in (("jury vote fraction", "#2563eb"), ("RL grade probability", "#dc2626")):
    rb = res["scores"][k]["reliability_bins"]; ax.plot([x["mean_conf"] for x in rb], [x["accuracy"] for x in rb], "o-", color=col, label=f"{k} (ECE {res['scores'][k]['ece_weighted']:.3f})")
ax.plot([0, 1], [0, 1], "k:", lw=0.8); ax.set_xlabel("confidence (bin mean)"); ax.set_ylabel("accuracy vs human (weighted)"); ax.legend(fontsize=7); ax.set_title("Reliability, 479 human-labelled sections", fontsize=9)
plt.tight_layout(); plt.savefig(O + "/reliability.png", dpi=160); plt.close()
fig, ax = plt.subplots(figsize=(6, 4))
for k, (g, sc) in S_.items():
    o = np.lexsort((RNG.rand(len(sc)), -sc)); wrong = (g != hum)[o]; ww = w[o]; cov = np.cumsum(ww) / ww.sum(); risk = np.cumsum(ww * wrong) / np.cumsum(ww)
    ax.plot(cov, risk, label=f"{k} (AUROC {res['scores'][k]['auroc_weighted']:.2f})")
ax.axvline(0.87, color="k", ls="--", lw=0.8); ax.text(0.872, 0.005, "jury triage at 20% of reports", fontsize=7)
ax.set_xlabel("coverage (share of sections kept, most confident first)"); ax.set_ylabel("error rate among kept (weighted)"); ax.legend(fontsize=7); ax.set_title("Risk–coverage", fontsize=9)
plt.tight_layout(); plt.savefig(O + "/risk_coverage.png", dpi=160); plt.close()
for k, r in res["scores"].items(): print(k, {x: r[x] for x in r if x != "reliability_bins"})
print("primary", res["primary_auroc_RLprob_minus_juryvf"])

"""clf strand metrics (CPU): project 1's metrics.py logic re-implemented unchanged (same bootstrap seed, same patient resampling)
plus paired deltas vs RL seed 0, SFT seed 0, jury, MedGemma-27B, and error AUROC (triage score = max softmax prob).
Env: TAGS (comma list of our tags). Out: O/metrics.json"""
import json, os, sys
import numpy as np
from clf_common import *
from sklearn.metrics import roc_auc_score
TAGS = os.environ["TAGS"].split(","); NB = 2000
REFS = {"rl_s0": B + "/analysis/rl_grader/preds/grpo_sft__checkpoint-300", "sft_s0": B + "/analysis/rl_grader/preds/sft__final",
        "medgemma27b_zs": B + "/analysis/rl_grader/preds/medgemma27b_zs", "rl_s1": B + "/analysis/rl_followups/C/preds/grpo_sft_s1__checkpoint-300",
        "rl_s2": B + "/analysis/rl_followups/C/preds/grpo_sft_s2__checkpoint-250", "rl_s0_conf": B + "/analysis/rl_abstain/preds/p1best_conf"}
def load(path, s):
    f = f"{path}__{s}.jsonl"; return {json.loads(l)["id"]: json.loads(l) for l in open(f)} if os.path.exists(f) else None
h = load(B + "/analysis/rl_grader/preds/grpo_sft__checkpoint-300", "test485"); ids = sorted(h); w = np.array([h[i]["w"] for i in ids]); pats = np.array([h[i]["anon_id"] for i in ids])
hum = np.array([h[i]["human"] for i in ids]); rand = np.array([h[i]["stratum"] == "random" for i in ids])
preds = {"jury": np.array([h[i]["label"] for i in ids])}; conf = {}
for t, p in REFS.items():
    d = load(p, "test485")
    if d: preds[t] = np.array([d[i]["pred"] or "INVALID" for i in ids]); conf[t] = np.array([d[i].get("p_grade") or 0.0 for i in ids])
for t in TAGS:
    d = load(f"{O}/preds/{t}", "test485"); preds[t] = np.array([d[i]["pred"] for i in ids]); conf[t] = np.array([d[i]["p_grade"] for i in ids])
up = np.unique(pats); grp = {p: np.where(pats == p)[0] for p in up}; rng = np.random.RandomState(0)
BS = [np.concatenate([grp[p] for p in rng.choice(up, len(up))]) for _ in range(NB)]
def stats(pr, ix=None):
    ix = np.arange(len(ids)) if ix is None else ix; ex = (pr[ix] == hum[ix]).astype(float); dy = (np.isin(pr[ix], list(POS)) == np.isin(hum[ix], list(POS))).astype(float)
    return {"exact_w": float((ex * w[ix]).sum() / w[ix].sum()), "dysp_w": float((dy * w[ix]).sum() / w[ix].sum())}
ci = lambda v: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
R = {"tags": TAGS, "test485": {}, "val": {}, "test": {}}
for t, pr in preds.items():
    s0 = stats(pr); bs = [stats(pr, ix) for ix in BS]
    r = {k: round(v, 4) for k, v in s0.items()}; r.update({k + "_ci": ci([b[k] for b in bs]) for k in s0})
    r["exact_unweighted"] = round(float((pr == hum).mean()), 4); r["exact_random50"] = round(float((pr[rand] == hum[rand]).mean()), 4); r["dysp_random50"] = round(float((np.isin(pr[rand], list(POS)) == np.isin(hum[rand], list(POS))).mean()), 4)
    r["ind_overcall"] = int(((pr == "IND") & (hum != "IND")).sum()); r["confusion"] = {g: {p: int(((hum == g) & (pr == p)).sum()) for p in GRADES + ["INVALID"]} for g in GRADES}
    if t in conf and conf[t].max() > 0:
        err = (pr != hum).astype(int); sc = 1 - conf[t]
        r["error_auroc"] = round(float(roc_auc_score(err, sc)), 4) if 0 < err.sum() < len(err) else None
        r["error_auroc_weighted"] = round(float(roc_auc_score(err, sc, sample_weight=w)), 4) if 0 < err.sum() < len(err) else None
        r["error_auroc_ci"] = ci([roc_auc_score(err[ix], sc[ix]) for ix in BS if 0 < err[ix].sum() < len(ix)])
    if t in TAGS:
        for other in ["rl_s0", "sft_s0", "jury", "medgemma27b_zs", "rl_s1", "rl_s2"]:
            if other not in preds: continue
            po = preds[other]; d0 = s0["exact_w"] - stats(po)["exact_w"]; db = [stats(pr, ix)["exact_w"] - stats(po, ix)["exact_w"] for ix in BS]
            r[f"exact_w_minus_{other}"] = {"point": round(d0, 4), "ci": ci(db)}
            d0 = s0["dysp_w"] - stats(po)["dysp_w"]; db = [stats(pr, ix)["dysp_w"] - stats(po, ix)["dysp_w"] for ix in BS]; r[f"dysp_w_minus_{other}"] = {"point": round(d0, 4), "ci": ci(db)}
        f = f"{O}/preds/{t}__test485.timing.json"; tm = json.load(open(f)); r["s_per_report"] = round(tm["seconds"] / tm["reports"], 4); r["params"] = tm.get("params")
        lg = f"{O}/runs/{t}/log.json"
        if os.path.exists(lg): r["train_log"] = {k: v for k, v in json.load(open(lg)).items() if k != "evals"}
    R["test485"][t] = r
for s in ("val", "test"):
    for t in TAGS + ["rl_s0", "sft_s0"]:
        d = load(f"{O}/preds/{t}" if t in TAGS else REFS[t], s)
        if d: R[s][t] = {"n": len(d), "exact_vs_jury": round(float(np.mean([v["pred"] == v["label"] for v in d.values()])), 4)}
json.dump(R, open(f"{O}/metrics.json", "w"), indent=1)
for t, r in R["test485"].items(): print(f"{t:18s} exact_w {r['exact_w']:.3f} {r['exact_w_ci']} dysp_w {r['dysp_w']:.3f} unw {r['exact_unweighted']:.3f} rand50 {r['exact_random50']:.3f} errAUC {r.get('error_auroc')}", {k: v for k, v in r.items() if k.startswith("exact_w_minus")}, r.get("s_per_report"), r.get("params"))
for s in ("val", "test"): print(s, R[s])

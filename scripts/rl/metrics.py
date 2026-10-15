"""Project 1 metrics (CPU). Env: TAGS (comma list of prediction tags), REF (tag compared against, default first), OUT (json path).
test485: population-weighted (w), unweighted and random-slice exact/dysplasia agreement with Rehan's worst grade; the jury is added as tag 'jury'.
val/test: exact agreement with the jury label. Paired patient-clustered bootstrap (2,000) for deltas vs REF and vs jury."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__)); from common import *
P = os.environ.get("PREDS", B + "/analysis/rl_grader/preds"); TAGS = os.environ["TAGS"].split(","); REF = os.environ.get("REF", TAGS[0]); NB = 2000
def load(tag, s):
    f = f"{P}/{tag}__{s}.jsonl"; return {json.loads(l)["id"]: json.loads(l) for l in open(f)} if os.path.exists(f) else None
R = {"tags": TAGS, "ref": REF, "test485": {}, "val": {}, "test": {}}
h = load(TAGS[0], "test485"); ids = sorted(h); w = np.array([h[i]["w"] for i in ids]); pats = np.array([h[i]["anon_id"] for i in ids])
hum = np.array([h[i]["human"] for i in ids]); rand = np.array([h[i]["stratum"] == "random" for i in ids])
preds = {"jury": np.array([h[i]["label"] for i in ids])}
for t in TAGS:
    d = load(t, "test485")
    if d: preds[t] = np.array([d[i]["pred"] or "INVALID" for i in ids])
up = np.unique(pats); grp = {p: np.where(pats == p)[0] for p in up}; rng = np.random.RandomState(0)
BS = [np.concatenate([grp[p] for p in rng.choice(up, len(up))]) for _ in range(NB)]
def stats(pr, ix=None):
    ix = np.arange(len(ids)) if ix is None else ix; ex = (pr[ix] == hum[ix]).astype(float); dy = (np.isin(pr[ix], list(POS)) == np.isin(hum[ix], list(POS))).astype(float)
    return {"exact_w": float((ex * w[ix]).sum() / w[ix].sum()), "dysp_w": float((dy * w[ix]).sum() / w[ix].sum())}
ci = lambda v: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
for t, pr in preds.items():
    s0 = stats(pr); bs = [stats(pr, ix) for ix in BS]
    r = {k: round(v, 4) for k, v in s0.items()}; r.update({k + "_ci": ci([b[k] for b in bs]) for k in s0})
    r["exact_unweighted"] = round(float((pr == hum).mean()), 4); r["exact_random50"] = round(float((pr[rand] == hum[rand]).mean()), 4); r["dysp_random50"] = round(float((np.isin(pr[rand], list(POS)) == np.isin(hum[rand], list(POS))).mean()), 4)
    r["parse_rate"] = round(float((pr != "INVALID").mean()), 4)
    r["ind_overcall"] = int(((pr == "IND") & (hum != "IND")).sum()); r["ind_true"] = int(((pr == "IND") & (hum == "IND")).sum())
    r["confusion"] = {g: {p: int(((hum == g) & (pr == p)).sum()) for p in GRADES + ["INVALID"]} for g in GRADES}
    for other in [REF, "jury"]:
        if other == t or other not in preds: continue
        po = preds[other]; d0 = s0["exact_w"] - stats(po)["exact_w"]; db = [stats(pr, ix)["exact_w"] - stats(po, ix)["exact_w"] for ix in BS]
        r[f"exact_w_minus_{other}"] = {"point": round(d0, 4), "ci": ci(db)}
    R["test485"][t] = r
for s in ("val", "test"):
    for t in TAGS:
        d = load(t, s)
        if d: R[s][t] = {"n": len(d), "exact_vs_jury": round(float(np.mean([v["pred"] == v["label"] for v in d.values()])), 4), "parse_rate": round(float(np.mean([v["pred"] is not None for v in d.values()])), 4)}
for t in TAGS:
    f = f"{P}/{t}__test485.timing.json"
    if os.path.exists(f): R["test485"][t]["s_per_report"] = round(json.load(open(f))["seconds"] / max(json.load(open(f))["reports"], 1), 3)
out = os.environ.get("OUT", B + "/analysis/rl_grader/metrics.json"); json.dump(R, open(out, "w"), indent=1)
for t, r in R["test485"].items(): print(f"{t:28s} exact_w {r['exact_w']:.3f} {r['exact_w_ci']}  dysp_w {r['dysp_w']:.3f}  parse {r['parse_rate']:.3f}  rand50 {r['exact_random50']:.3f}", {k: v for k, v in r.items() if k.startswith('exact_w_minus')})
for s in ("val", "test"): print(s, R[s])

"""Project 2 metrics (CPU). For each method: a labelling system (its own grades) + an ordering of the 4,929 population sections
(least confident first). Reviewing the first k sections opens their reports; reviewed sections count as correct. Population accuracy at
each budget = Horvitz-Thompson weighted mean over the 479 human-labelled sections of [reviewed OR system grade == human grade]
(same estimator as Paper B triage v2). Primary: accuracy at 17% of reports opened, RL-abstention (best alpha on validation) minus
jury vote-fraction triage, paired patient bootstrap (2,000). Env: ALPHA_TAGS (comma list of alpha<a>__<ckpt> tags), BEST (tag), OUT."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__)); from common import *
R = B + "/analysis/rl_grader"; A = B + "/analysis/rl_abstain"; NB = 2000; RNG = np.random.RandomState(20261016)
pop = [json.loads(l) for l in open(R + "/data/pop.jsonl")]; pid = [p["id"] for p in pop]; prep = np.array([p["report_id"] for p in pop]); nrep = len(set(prep))
lab = {json.loads(l)["id"]: json.loads(l) for l in open(R + "/data/test485.jsonl")}
L = sorted(set(lab) & set(pid)); pos = {i: k for k, i in enumerate(pid)}; Li = np.array([pos[i] for i in L])
w = np.array([lab[i]["w"] for i in L]); hum = np.array([lab[i]["human"] for i in L]); pats = np.array([lab[i]["anon_id"] for i in L]); amb = np.array([lab[i]["ambiguous"] for i in L])
def load(tag, s="pop"):
    f = f"{A}/preds/{tag}__{s}.jsonl"; return {json.loads(l)["id"]: json.loads(l) for l in open(f)} if os.path.exists(f) else None
M = {}   # method -> (grade per pop section, score per pop section: LOWER = review first)
M["jury_votefrac"] = (np.array([p["label"] for p in pop]), np.array([p["jury_vf"] if p["jury_vf"] is not None else 0 for p in pop]))
c = load("p1best_conf")
if c:
    g = np.array([c[i]["pred"] or "INVALID" for i in pid])
    M["p1_tokenprob"] = (g, np.array([c[i].get("p_grade") if c[i].get("p_grade") is not None else 0 for i in pid]))
    M["p1_agree8"] = (g, np.array([np.mean([s == c[i]["pred"] for s in c[i].get("samples") or []]) if c[i].get("samples") else 0 for i in pid]))
    M["p1_random"] = (g, RNG.rand(len(pid)))
for tag in [t for t in os.environ.get("ALPHA_TAGS", "").split(",") if t]:
    d = load(tag)
    if not d: continue
    g = np.array([d[i]["pred"] or "INVALID" for i in pid]); pg = np.array([d[i].get("p_grade") or 0 for i in pid])
    M[tag] = (g, np.where(g == "UNSURE", -1.0, pg))
BUD = np.arange(0, 501) / 1000.0   # 0-50% of sections in 0.1% steps
def curve(g, sc, ix=None):
    order = np.lexsort((RNG.rand(len(sc)), sc)); ss = sc[order]; reps = prep[order]
    opened = np.zeros(len(sc) + 1, int); seen = set()
    for j, r in enumerate(reps): seen.add(r); opened[j + 1] = len(seen)
    n = np.round(BUD * len(sc)).astype(int); thr = np.array([-np.inf if k == 0 else ss[k - 1] for k in n]); fr = opened[n] / nrep
    rev = sc[Li][None, :] <= thr[:, None]; ok = (g[Li] == hum)[None, :]; Z = (rev | ok).astype(float)
    return fr, Z
def at(fr, Z, frac, ix):
    k = max(np.searchsorted(fr, frac, side="right") - 1, 0); ww = w[ix]; return float((Z[k, ix] * ww).sum() / ww.sum())
up = np.unique(pats); grp = {p: np.where(pats == p)[0] for p in up}; BS = [np.concatenate([grp[p] for p in RNG.choice(up, len(up))]) for _ in range(NB)]
ALL = np.arange(len(L)); out = {"n_population": len(pid), "n_reports": nrep, "n_labelled": len(L), "methods": {}}
curves = {}
for m, (g, sc) in M.items():
    fr, Z = curve(g, sc); curves[m] = (fr, Z)
    r = {"acc_no_review": round(at(fr, Z, 0.0, ALL), 4)}
    for f in (0.05, 0.10, 0.17, 0.20, 0.30): r[f"acc@{int(f*100)}%reports"] = round(at(fr, Z, f, ALL), 4); r[f"acc@{int(f*100)}%reports_ci"] = [round(float(x), 4) for x in np.percentile([at(fr, Z, f, ix) for ix in BS], [2.5, 97.5])]
    if m.startswith("alpha"):
        abst = g == "UNSURE"; r["abstention_rate_sections"] = round(float(abst.mean()), 4); r["reports_opened_by_abstentions"] = round(len(set(prep[abst])) / nrep, 4)
        ans = ~abst[Li]; r["selective_accuracy_weighted"] = round(float(((g[Li] == hum) * w)[ans].sum() / w[ans].sum()), 4) if ans.any() else None
        r["abstained_labelled"] = int(abst[Li].sum()); r["abstained_and_ambiguous"] = int((abst[Li] & (amb == 1)).sum()); r["ambiguous_total"] = int(amb.sum())
    out["methods"][m] = r
best = os.environ.get("BEST")
if best and best in curves:
    fb, Zb = curves[best]; fj, Zj = curves["jury_votefrac"]; d0 = at(fb, Zb, 0.17, ALL) - at(fj, Zj, 0.17, ALL)
    db = [at(fb, Zb, 0.17, ix) - at(fj, Zj, 0.17, ix) for ix in BS]
    out["primary_best_minus_jury_at_17pct_reports"] = {"best": best, "point": round(d0, 4), "ci": [round(float(x), 4) for x in np.percentile(db, [2.5, 97.5])]}
out["curves"] = {m: {"frac_reports": [round(float(x), 4) for x in fr[::10]], "acc": [round(at(fr, Z, float(x), ALL), 4) for x in fr[::10]]} for m, (fr, Z) in curves.items()}
json.dump(out, open(os.environ.get("OUT", A + "/metrics.json"), "w"), indent=1)
for m, r in out["methods"].items(): print(f"{m:28s}", {k: v for k, v in r.items() if not k.endswith("_ci")})
print(out.get("primary_best_minus_jury_at_17pct_reports"))

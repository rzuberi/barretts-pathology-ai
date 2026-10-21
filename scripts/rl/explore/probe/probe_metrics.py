"""Copy of scripts/rl/p2_metrics.py (v2.1 tie-aware triage estimator) for the probe strand, with paired comparisons of each tag in
COMPARE (comma list) against jury vote-fraction triage AND the p1 token-probability triage at 10/17/20/30% of reports.
Env as p2_metrics: PDIR (preds in PDIR/preds/<tag>__pop.jsonl), ALPHA_TAGS, BEST, PRIMARY_FRAC, OUT, plus COMPARE."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")); from common import *
R = B + "/analysis/rl_grader"; A = os.environ.get("PDIR", B + "/analysis/rl_abstain"); PFRAC = float(os.environ.get("PRIMARY_FRAC", 0.17)); A0 = B + "/analysis/rl_abstain"; NB = 2000; RNG = np.random.RandomState(20261016)
pop = [json.loads(l) for l in open(R + "/data/pop.jsonl")]; pid = [p["id"] for p in pop]; prep = np.array([p["report_id"] for p in pop]); nrep = len(set(prep))
lab = {json.loads(l)["id"]: json.loads(l) for l in open(R + "/data/test485.jsonl")}
L = sorted(set(lab) & set(pid)); pos = {i: k for k, i in enumerate(pid)}; Li = np.array([pos[i] for i in L])
w = np.array([lab[i]["w"] for i in L]); hum = np.array([lab[i]["human"] for i in L]); pats = np.array([lab[i]["anon_id"] for i in L]); amb = np.array([lab[i]["ambiguous"] for i in L])
def load(tag, s="pop"):
    f = f"{A}/preds/{tag}__{s}.jsonl"; return {json.loads(l)["id"]: json.loads(l) for l in open(f)} if os.path.exists(f) else None
M = {}   # method -> (grade per pop section, score per pop section: LOWER = review first)
M["jury_votefrac"] = (np.array([p["label"] for p in pop]), np.array([p["jury_vf"] if p["jury_vf"] is not None else 0 for p in pop]))
c = None
f0 = f"{A0}/preds/p1best_conf__pop.jsonl"
if os.path.exists(f0): c = {json.loads(l)["id"]: json.loads(l) for l in open(f0)}
if c:
    g = np.array([c[i]["pred"] or "INVALID" for i in pid])
    M["p1_tokenprob"] = (g, np.array([c[i].get("p_grade") if c[i].get("p_grade") is not None else 0 for i in pid]))
    M["p1_agree8"] = (g, np.array([np.mean([s == c[i]["pred"] for s in c[i].get("samples") or []]) if c[i].get("samples") else 0 for i in pid]))
    M["p1_random"] = (g, RNG.rand(len(pid)))
for tag in [t for t in os.environ.get("ALPHA_TAGS", "").split(",") if t]:
    d = load(tag)
    if not d: continue
    g = np.array([d[i]["pred"] or "INVALID" for i in pid])
    if any("conf" in d[i] for i in pid[:50]): sc = np.array([d[i].get("conf") if d[i].get("conf") is not None else 0.0 for i in pid])   # D2: stated confidence
    else: sc = np.where(g == "UNSURE", -1.0, np.array([d[i].get("p_grade") or 0 for i in pid]))
    M[tag] = (g, sc)
BUD = np.arange(0, 501) / 1000.0   # 0-50% of sections in 0.1% steps
def curve(g, sc, ix=None):
    order = np.lexsort((RNG.rand(len(sc)), sc)); ss = sc[order]; reps = prep[order]
    opened = np.zeros(len(sc) + 1, int); seen = set()
    for j, r in enumerate(reps): seen.add(r); opened[j + 1] = len(seen)
    n = np.round(BUD * len(sc)).astype(int); fr = opened[n] / nrep
    # tie-aware review probability: below the k-th score -> reviewed; tied at it -> reviewed with prob (#tied within budget)/(#tied)
    q = np.zeros((len(n), len(Li)))
    for b, k in enumerate(n):
        if k == 0: continue
        thr = ss[k - 1]; below = int((ss < thr).sum()); tied = int((ss == thr).sum()); frac = (k - below) / tied
        s_l = sc[Li]; q[b] = np.where(s_l < thr, 1.0, np.where(s_l == thr, frac, 0.0))
    ok = (g[Li] == hum).astype(float)[None, :]; Z = q + (1 - q) * ok
    return fr, Z
def at(fr, Z, frac, ix):
    k = max(np.searchsorted(fr, frac, side="right") - 1, 0); ww = w[ix]; return float((Z[k, ix] * ww).sum() / ww.sum())
up = np.unique(pats); grp = {p: np.where(pats == p)[0] for p in up}; BS = [np.concatenate([grp[p] for p in RNG.choice(up, len(up))]) for _ in range(NB)]
ALL = np.arange(len(L)); out = {"n_population": len(pid), "n_reports": nrep, "n_labelled": len(L), "methods": {}}
curves = {}
for m, (g, sc) in M.items():
    fr, Z = curve(g, sc); curves[m] = (fr, Z)
    r = {"acc_no_review": round(at(fr, Z, 0.0, ALL), 4)}
    for f in sorted({0.05, 0.10, 0.17, 0.20, 0.30, PFRAC}): r[f"acc@{int(f*100)}%reports"] = round(at(fr, Z, f, ALL), 4); r[f"acc@{int(f*100)}%reports_ci"] = [round(float(x), 4) for x in np.percentile([at(fr, Z, f, ix) for ix in BS], [2.5, 97.5])]
    if m not in ("jury_votefrac", "p1_tokenprob", "p1_agree8", "p1_random"):
        abst = g == "UNSURE"; r["abstention_rate_sections"] = round(float(abst.mean()), 4); r["reports_opened_by_abstentions"] = round(len(set(prep[abst])) / nrep, 4)
        ans = ~abst[Li]; r["selective_accuracy_weighted"] = round(float(((g[Li] == hum) * w)[ans].sum() / w[ans].sum()), 4) if ans.any() else None
        r["abstained_labelled"] = int(abst[Li].sum()); r["abstained_and_ambiguous"] = int((abst[Li] & (amb == 1)).sum()); r["ambiguous_total"] = int(amb.sum())
    out["methods"][m] = r
best = os.environ.get("BEST")
if best and best in curves:
    fb, Zb = curves[best]; fj, Zj = curves["jury_votefrac"]; d0 = at(fb, Zb, PFRAC, ALL) - at(fj, Zj, PFRAC, ALL)
    db = [at(fb, Zb, PFRAC, ix) - at(fj, Zj, PFRAC, ix) for ix in BS]
    out[f"primary_best_minus_jury_at_{int(PFRAC*100)}pct_reports"] = {"best": best, "point": round(d0, 4), "ci": [round(float(x), 4) for x in np.percentile(db, [2.5, 97.5])]}
# probe strand addition: paired differences of each COMPARE tag vs jury and vs token-prob triage at several budgets
out["paired"] = {}
for tag in [t for t in os.environ.get("COMPARE", "").split(",") if t and t in curves]:
    ft, Zt = curves[tag]; out["paired"][tag] = {}
    for ref in ("jury_votefrac", "p1_tokenprob"):
        if ref not in curves: continue
        fr_, Zr = curves[ref]
        for f in (0.10, 0.17, 0.20, 0.30):
            d0 = at(ft, Zt, f, ALL) - at(fr_, Zr, f, ALL); db = [at(ft, Zt, f, ix) - at(fr_, Zr, f, ix) for ix in BS]
            out["paired"][tag][f"minus_{ref}@{int(f*100)}%"] = {"point": round(d0, 4), "ci": [round(float(x), 4) for x in np.percentile(db, [2.5, 97.5])]}
out["curves"] = {m: {"frac_reports": [round(float(x), 4) for x in fr[::10]], "acc": [round(at(fr, Z, float(x), ALL), 4) for x in fr[::10]]} for m, (fr, Z) in curves.items()}
json.dump(out, open(os.environ.get("OUT", A + "/metrics.json"), "w"), indent=1)
for m, r in out["methods"].items(): print(f"{m:28s}", {k: v for k, v in r.items() if not k.endswith("_ci")})
print({k: v for k, v in out.items() if k.startswith("primary")})
for t, r in out["paired"].items(): print("paired", t, r)

"""Item 2 (CPU): scripts/rl/p2_metrics.py's tie-aware Horvitz-Thompson triage estimator (unchanged logic) generalised to
  METHODS  name=/abs/prefix,...   (prefix__pop.jsonl with pred + p_grade, or conf for the D2 stated-confidence model; UNSURE -> first)
  EXCLUDE  none | train_patients | train_rows   (drop population sections of training patients / literal train.jsonl rows; the 479 stay)
Always includes jury vote fraction, the project-1 token probability and random order. Budgets 5/10/17/20/30% of reports; paired
patient bootstrap (2,000) deltas vs the jury at every budget. Out: OUT json."""
import json, os
from pa_common import *
EXC = os.environ.get("EXCLUDE", "none"); RNG = np.random.RandomState(20261016)
pop = rows("pop")
if EXC == "train_patients": tp = {r["anon_id"] for r in rows("train")}; pop = [p for p in pop if p["anon_id"] not in tp]
elif EXC == "train_rows": ti = {r["id"] for r in rows("train")}; pop = [p for p in pop if p["id"] not in ti]
pid = [p["id"] for p in pop]; prep = np.array([p["report_id"] for p in pop]); nrep = len(set(prep))
lab = {r["id"]: r for r in rows("test485")}; L = sorted(set(lab) & set(pid)); assert len(L) == 479, len(L)
pos = {i: k for k, i in enumerate(pid)}; Li = np.array([pos[i] for i in L])
w = np.array([lab[i]["w"] for i in L]); hum = np.array([lab[i]["human"] for i in L]); pats = np.array([lab[i]["anon_id"] for i in L])
M = {"jury_votefrac": (np.array([p["label"] for p in pop]), np.array([p["jury_vf"] if p["jury_vf"] is not None else 0 for p in pop]))}
def add(name, prefix):
    d = load_preds(prefix, "pop")
    if not d: print("missing", prefix); return
    g = np.array([d[i]["pred"] or "INVALID" for i in pid])
    if any("conf" in d[i] for i in pid[:50]): sc = np.array([d[i].get("conf") if d[i].get("conf") is not None else 0.0 for i in pid])
    else: sc = np.where(g == "UNSURE", -1.0, np.array([d[i].get("p_grade") or 0 for i in pid]))
    M[name] = (g, sc)
add("p1_tokenprob", B + "/analysis/rl_abstain/preds/p1best_conf"); M["p1_random"] = (M["p1_tokenprob"][0], RNG.rand(len(pid)))
for spec in [x for x in os.environ.get("METHODS", "").split(",") if x]: n, p = spec.split("="); add(n, p)
BUD = np.arange(0, 501) / 1000.0
def curve(g, sc):
    order = np.lexsort((RNG.rand(len(sc)), sc)); ss = sc[order]; reps = prep[order]
    opened = np.zeros(len(sc) + 1, int); seen = set()
    for j, r in enumerate(reps): seen.add(r); opened[j + 1] = len(seen)
    n = np.round(BUD * len(sc)).astype(int); fr = opened[n] / nrep; q = np.zeros((len(n), len(Li)))
    for b, k in enumerate(n):
        if k == 0: continue
        thr = ss[k - 1]; below = int((ss < thr).sum()); tied = int((ss == thr).sum()); frac = (k - below) / tied
        s_l = sc[Li]; q[b] = np.where(s_l < thr, 1.0, np.where(s_l == thr, frac, 0.0))
    ok = (g[Li] == hum).astype(float)[None, :]; return fr, q + (1 - q) * ok
def at(fr, Z, frac, ix):
    k = max(np.searchsorted(fr, frac, side="right") - 1, 0); ww = w[ix]; return float((Z[k, ix] * ww).sum() / ww.sum())
up = np.unique(pats); grp = {p: np.where(pats == p)[0] for p in up}; BS = [np.concatenate([grp[p] for p in RNG.choice(up, len(up))]) for _ in range(NB)]
ALL = np.arange(len(L)); FR = [0.05, 0.10, 0.17, 0.20, 0.30]
out = {"exclude": EXC, "n_population": len(pid), "n_reports": nrep, "n_labelled": len(L), "methods": {}}; curves = {m: curve(*gs) for m, gs in M.items()}
fj, Zj = curves["jury_votefrac"]
for m, (fr, Z) in curves.items():
    r = {"acc_no_review": round(at(fr, Z, 0.0, ALL), 4)}
    for f in FR:
        r[f"acc@{int(f*100)}%"] = round(at(fr, Z, f, ALL), 4); r[f"acc@{int(f*100)}%_ci"] = ci([at(fr, Z, f, ix) for ix in BS])
        if m != "jury_votefrac": r[f"minus_jury@{int(f*100)}%"] = {"point": round(at(fr, Z, f, ALL) - at(fj, Zj, f, ALL), 4), "ci": ci([at(fr, Z, f, ix) - at(fj, Zj, f, ix) for ix in BS])}
    out["methods"][m] = r
out["curves"] = {m: {"frac_reports": [round(float(x), 4) for x in fr[::10]], "acc": [round(at(fr, Z, float(x), ALL), 4) for x in fr[::10]]} for m, (fr, Z) in curves.items()}
f = os.environ.get("OUT", O + f"/triage_{EXC}.json"); json.dump(out, open(f, "w"), indent=1); os.chmod(f, 0o600)
print(f"EXCLUDE={EXC}: population {len(pid)} sections / {nrep} reports")
for m, r in out["methods"].items(): print(f"{m:22s} no-review {100*r['acc_no_review']:.1f} | " + " | ".join(f"{int(x*100)}%: {100*r[f'acc@{int(x*100)}%']:.1f}" + (f" ({100*r[f'minus_jury@{int(x*100)}%']['point']:+.1f} [{100*r[f'minus_jury@{int(x*100)}%']['ci'][0]:+.1f},{100*r[f'minus_jury@{int(x*100)}%']['ci'][1]:+.1f}])" if m != "jury_votefrac" else "") for x in FR))

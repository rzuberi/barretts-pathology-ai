"""Follow-up: does adding the project-1 RL model (MedGemma-4B SFT->GRPO) to the v3 jury improve agreement with the human labels?
Fixed before looking: primary = population-weighted exact worst-grade agreement on the 479 human sections, J11 - J10, paired patient
bootstrap (2,000). Variants: J10 (original 10 jurors, rebuilt from per-juror outputs), J11 (+ RL model), J10swap (RL replaces granite3.3:8b).
Rule as build_sections_v3_consensus.py: a grade is kept if listed by >= ceil(n/2) of the jurors that returned the section; consensus = worst kept.
RL juror votes for its single (worst) grade. Secondary: triage with the new vote fraction (tie-aware HT estimator, % of reports opened)."""
import glob, json, os, sys
from collections import Counter, defaultdict
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__)); from common import *
T = B + "/phd/thesis"; R = B + "/analysis/rl_grader"; A = B + "/analysis/rl_abstain"; OUT = B + "/analysis/rl_grader/eleventh_juror"; os.makedirs(OUT, exist_ok=True)
NB = 2000; RNG = np.random.RandomState(20261016)
pop = [json.loads(l) for l in open(R + "/data/pop.jsonl")]; lab = {json.loads(l)["id"]: json.loads(l) for l in open(R + "/data/test485.jsonl")}
need = {(p["report_id"], p["section"]) for p in pop}; cases = {c for c, _ in need}
votes = defaultdict(dict)   # (case, section) -> juror -> grade set
for f in glob.glob(T + "/feasibility/runs/sec3full_*/output/sections_v3_*.csv"):
    juror = os.path.basename(f).replace("sections_v3_", "").rsplit("_shard", 1)[0]
    d = pd.read_csv(f, dtype=str).fillna("").drop_duplicates("CaseName"); d = d[d.CaseName.isin(cases)]
    for r in d.itertuples():
        if r.sections_json.startswith("PARSE_FAIL"): continue
        seen = {}
        for s in json.loads(r.sections_json):
            key = s["section"].upper().strip(); key = "WHOLE" if key in ("WHOLE", "WHOLE_REPORT", "") else key[:2]
            seen.setdefault(key, set()).update(s["grades"])
        for key, gs in seen.items():
            if (r.CaseName, key) in need: votes[(r.CaseName, key)][juror] = gs
rl_pop = {json.loads(l)["id"]: json.loads(l)["pred"] for l in open(A + "/preds/p1best_conf__pop.jsonl")}
rl_485 = {json.loads(l)["id"]: json.loads(l)["pred"] for l in open(R + "/preds/grpo_sft__checkpoint-300__test485.jsonl")}
same = sum(rl_pop.get(i) == g for i, g in rl_485.items()); print(f"RL predictions identical between the two eval runs on {same}/{len(rl_485)} human sections")
def consensus(jv):
    n = len(jv)
    if n == 0: return None, np.nan
    thr = (n + 1) // 2; cnt = Counter(g for gs in jv.values() for g in gs if g in ORD)
    kept = [g for g, k in cnt.items() if k >= thr]
    if not kept: return None, np.nan
    w = max(kept, key=lambda g: ORD[g]); return w, cnt[w] / n
VAR = {}
for p in pop:
    jv = votes.get((p["report_id"], p["section"]), {}); rl = rl_pop.get(p["id"])
    j11 = dict(jv); j11["rl_medgemma4b"] = {rl} if rl else set()
    js = {k: v for k, v in jv.items() if k != "granite3_3_8b"}; js["rl_medgemma4b"] = {rl} if rl else set()
    for name, d in (("J10", jv), ("J11", j11), ("J10swap", js)): VAR.setdefault(name, {})[p["id"]] = consensus(d)
ids = sorted(lab); w = np.array([lab[i]["w"] for i in ids]); hum = np.array([lab[i]["human"] for i in ids]); pats = np.array([lab[i]["anon_id"] for i in ids])
rand = np.array([lab[i]["stratum"] == "random" for i in ids]); jury_label = np.array([lab[i]["label"] for i in ids])
repro = float(np.mean([VAR["J10"][i][0] == lab[i]["label"] for i in ids])); print(f"J10 rebuilt consensus == stored jury label on {repro:.3f} of human sections")
up = np.unique(pats); grp = {p: np.where(pats == p)[0] for p in up}; BS = [np.concatenate([grp[p] for p in RNG.choice(up, len(up))]) for _ in range(NB)]
def ex(g, ix): return float(((g[ix] == hum[ix]) * w[ix]).sum() / w[ix].sum())
def dy(g, ix): return float(((np.isin(g[ix], list(POS)) == np.isin(hum[ix], list(POS))) * w[ix]).sum() / w[ix].sum())
ci = lambda v: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
ALL = np.arange(len(ids)); res = {"reproduction_J10_vs_stored": round(repro, 4), "rl_eval_runs_identical": f"{same}/{len(rl_485)}", "variants": {}}
G = {k: np.array([VAR[k][i][0] or "NONE" for i in ids]) for k in VAR}; G["RL alone"] = np.array([rl_485[i] or "NONE" for i in ids])
for k, g in G.items():
    r = {"exact_w": round(ex(g, ALL), 4), "exact_w_ci": ci([ex(g, ix) for ix in BS]), "dysp_w": round(dy(g, ALL), 4), "exact_unweighted": round(float((g == hum).mean()), 4),
         "exact_random50": round(float((g[rand] == hum[rand]).mean()), 4), "changed_vs_J10": int((g != G["J10"]).sum()),
         "fixed_vs_J10": int(((g == hum) & (G["J10"] != hum)).sum()), "broken_vs_J10": int(((g != hum) & (G["J10"] == hum)).sum())}
    if k != "J10":
        r["exact_w_minus_J10"] = {"point": round(ex(g, ALL) - ex(G["J10"], ALL), 4), "ci": ci([ex(g, ix) - ex(G["J10"], ix) for ix in BS])}
        r["dysp_w_minus_J10"] = {"point": round(dy(g, ALL) - dy(G["J10"], ALL), 4), "ci": ci([dy(g, ix) - dy(G["J10"], ix) for ix in BS])}
    res["variants"][k] = r
# secondary: triage with each variant's own vote fraction over the population (tie-aware)
pid = [p["id"] for p in pop]; prep = np.array([p["report_id"] for p in pop]); nrep = len(set(prep)); pos = {i: n for n, i in enumerate(pid)}; Li = np.array([pos[i] for i in ids])
def tri(k, frac):
    g = np.array([VAR[k][i][0] or "NONE" for i in pid]); sc = np.array([VAR[k][i][1] if not np.isnan(VAR[k][i][1]) else 0 for i in pid])
    order = np.lexsort((RNG.rand(len(sc)), sc)); ss = sc[order]; seen = set(); opened = np.zeros(len(sc) + 1, int)
    for j, r_ in enumerate(prep[order]): seen.add(r_); opened[j + 1] = len(seen)
    kk = int(np.searchsorted(opened / nrep, frac, side="right") - 1)
    thr = ss[kk - 1] if kk > 0 else -np.inf; below = (ss < thr).sum(); tied = (ss == thr).sum(); q_t = (kk - below) / tied if kk > 0 else 0
    s_l = sc[Li]; q = np.where(s_l < thr, 1.0, np.where(s_l == thr, q_t, 0.0)) if kk > 0 else np.zeros(len(Li)); ok = (g[Li] == hum).astype(float); Z = q + (1 - q) * ok
    return lambda ix: float((Z[ix] * w[ix]).sum() / w[ix].sum())
res["triage"] = {}
for frac in (0.10, 0.17, 0.20):
    f10 = tri("J10", frac); f11 = tri("J11", frac)
    res["triage"][f"{int(frac*100)}%reports"] = {"J10": round(f10(ALL), 4), "J11": round(f11(ALL), 4), "J11_minus_J10": {"point": round(f11(ALL) - f10(ALL), 4), "ci": ci([f11(ix) - f10(ix) for ix in BS])}}
json.dump(res, open(OUT + "/results.json", "w"), indent=1); os.chmod(OUT + "/results.json", 0o600)
for k, r in res["variants"].items(): print(f"{k:9s}", {x: r[x] for x in r if x not in ("exact_w_ci",)})
print("triage:", res["triage"])

"""Explore/grader report (CPU). Scores every single model and ensemble on the 479 human sections exactly as project 1 (population-weighted
exact and dysplasia yes/no, paired patient bootstrap 2,000 vs the seed-0 RL model and vs the jury), plus jury-test-fold agreement.
Checkpoints of the new runs (a1, a2, a3) are selected on the jury-labelled VALIDATION fold only (sel.py rule: best accuracy, earliest on ties).
Ensemble rule fixed in PLAN.md: majority of valid votes, ties -> seed-0 RL model's prediction. Out: analysis/rl_explore/grader/results.json"""
import glob, json, os, sys
from collections import Counter
import numpy as np
sys.path.insert(0, "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis/scripts/rl"); from common import *
R = B + "/analysis/rl_grader"; FU = B + "/analysis/rl_followups"; O = B + "/analysis/rl_explore/grader"; P = O + "/preds"; NB = 2000; RNG = np.random.RandomState(20261016)
lab = {json.loads(l)["id"]: json.loads(l) for l in open(R + "/data/test485.jsonl")}; ids = sorted(lab)
w = np.array([lab[i]["w"] for i in ids]); hum = np.array([lab[i]["human"] for i in ids]); pats = np.array([lab[i]["anon_id"] for i in ids]); rand = np.array([lab[i]["stratum"] == "random" for i in ids])
def load(f): d = {json.loads(l)["id"]: json.loads(l)["pred"] for l in open(f)}; return np.array([d.get(i) or "INVALID" for i in ids])
def jt(f): r = [json.loads(l) for l in open(f)] if os.path.exists(f) else []; return round(float(np.mean([x["pred"] == x["label"] for x in r])), 4) if r else None
def jt_preds(f): return {json.loads(l)["id"]: (json.loads(l)["pred"], json.loads(l)["label"]) for l in open(f)} if os.path.exists(f) else None
sel = {s: json.load(open(f"{FU}/C/preds/selected_grpo_sft_s{s}.json"))["selected"] for s in (1, 2)}
src = {"rl_s0": (R + "/preds/grpo_sft__checkpoint-300", "seed-0 RL model (project 1)"), "rl_s1": (f"{FU}/C/preds/grpo_sft_s1__{sel[1]}", "seed-1 RL"), "rl_s2": (f"{FU}/C/preds/grpo_sft_s2__{sel[2]}", "seed-2 RL"),
       "sft_s0": (R + "/preds/sft__final", "seed-0 SFT"), "medgemma27b": (R + "/preds/medgemma27b_zs", "MedGemma-27B zero-shot")}
# new runs: pick checkpoint on validation
selected = {}
for run in ("a1", "a2", "a3", "short"):
    sc = {}
    for f in glob.glob(f"{P}/{run}__checkpoint-*__val.jsonl"):
        ck = os.path.basename(f).split("__")[1]; r = [json.loads(l) for l in open(f)]; sc[ck] = round(sum(x["pred"] == x["label"] for x in r) / len(r), 4)
    if run == "a3":   # continuation: checkpoint-300 (the published model) competes too
        r = [json.loads(l) for l in open(R + "/preds/grpo_sft__checkpoint-300__val.jsonl")]; sc["checkpoint-300"] = round(sum(x["pred"] == x["label"] for x in r) / len(r), 4)
    if not sc: continue
    best = max(sorted(sc), key=lambda k: (sc[k], -int(k.split("-")[1])))
    selected[run] = {"scores": sc, "selected": best, "test485_available": os.path.exists(f"{P}/{run}__{best}__test485.jsonl")}
    if selected[run]["test485_available"]: src[run] = (f"{P}/{run}__{best}", {"a1": "soft jury reward + low-agreement sections", "a2": "soft jury reward, P1 data", "a3": "P1 reward, 600 steps", "short": "P1 RL model + 100 GRPO steps, soft jury reward + low-agreement sections"}[run])
    elif run == "a3" and best == "checkpoint-300": src[run] = src["rl_s0"]; selected[run]["note"] = "validation preferred the original checkpoint-300"
M = {k: load(v[0] + "__test485.jsonl") for k, v in src.items()}; M["jury"] = np.array([lab[i]["label"] for i in ids])
JT = {k: jt(v[0] + "__test.jsonl") for k, v in src.items()}
def vote(keys, tie="rl_s0"):
    out = []
    for n in range(len(ids)):
        c = Counter(M[k][n] for k in keys if M[k][n] != "INVALID")
        if not c: out.append("INVALID"); continue
        top = max(c.values()); cand = [g for g, v in c.items() if v == top]
        out.append(cand[0] if len(cand) == 1 else (M[tie][n] if M[tie][n] in cand else max(cand, key=lambda g: ORD[g])))
    return np.array(out)
def vote_test(keys, tie="rl_s0"):
    T = {k: jt_preds(src[k][0] + "__test.jsonl") for k in keys}
    if any(v is None for v in T.values()): return None
    ok = 0; n = 0
    for i in T[keys[0]]:
        c = Counter(T[k][i][0] for k in keys if T[k][i][0]); lab_ = T[keys[0]][i][1]
        if not c: n += 1; continue
        top = max(c.values()); cand = [g for g, v in c.items() if v == top]
        p = cand[0] if len(cand) == 1 else (T[tie][i][0] if T[tie][i][0] in cand else max(cand, key=lambda g: ORD[g]))
        ok += p == lab_; n += 1
    return round(ok / n, 4)
ENS = {"ens_3seeds": ["rl_s0", "rl_s1", "rl_s2"], "ens_3seeds+27b": ["rl_s0", "rl_s1", "rl_s2", "medgemma27b"]}
new = [k for k in ("a1", "a2", "a3", "short") if k in M and src[k] is not src["rl_s0"]]
if new: ENS["ens_3seeds+new"] = ["rl_s0", "rl_s1", "rl_s2"] + new; ENS["ens_all+27b"] = ["rl_s0", "rl_s1", "rl_s2"] + new + ["medgemma27b"]
for k, keys in ENS.items(): M[k] = vote(keys); JT[k] = vote_test(keys)
up = np.unique(pats); grp = {p: np.where(pats == p)[0] for p in up}; BS = [np.concatenate([grp[p] for p in RNG.choice(up, len(up))]) for _ in range(NB)]; ALL = np.arange(len(ids))
ex = lambda g, ix: float(((g[ix] == hum[ix]) * w[ix]).sum() / w[ix].sum()); dy = lambda g, ix: float(((np.isin(g[ix], list(POS)) == np.isin(hum[ix], list(POS))) * w[ix]).sum() / w[ix].sum())
ci = lambda v: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
res = {"selected_checkpoints": selected, "ensembles": ENS, "models": {}}
for k, g in M.items():
    r = {"desc": src[k][1] if k in src else ("10-model jury" if k == "jury" else "majority vote: " + ", ".join(ENS[k])),
         "exact_w": round(ex(g, ALL), 4), "exact_w_ci": ci([ex(g, ix) for ix in BS]), "dysp_w": round(dy(g, ALL), 4), "dysp_w_ci": ci([dy(g, ix) for ix in BS]),
         "exact_unweighted": round(float((g == hum).mean()), 4), "exact_random50": round(float((g[rand] == hum[rand]).mean()), 4), "parse_rate": round(float((g != "INVALID").mean()), 4),
         "ind_overcall": int(((g == "IND") & (hum != "IND")).sum()), "jury_test_fold_agreement": JT.get(k),
         "confusion": {h: {p: int(((hum == h) & (g == p)).sum()) for p in GRADES + ["INVALID"]} for h in GRADES}}
    for other in ("rl_s0", "jury"):
        if other != k: r[f"exact_w_minus_{other}"] = {"point": round(ex(g, ALL) - ex(M[other], ALL), 4), "ci": ci([ex(g, ix) - ex(M[other], ix) for ix in BS])}
    if k != "rl_s0": r["changed_vs_rl_s0"] = {"n": int((g != M["rl_s0"]).sum()), "fixed": int(((g == hum) & (M["rl_s0"] != hum)).sum()), "broken": int(((g != hum) & (M["rl_s0"] == hum)).sum())}
    res["models"][k] = r
json.dump(res, open(O + "/results.json", "w"), indent=1); os.chmod(O + "/results.json", 0o600)
print("selected:", {k: v["selected"] for k, v in selected.items()})
for k, r in res["models"].items():
    print(f"{k:18s} exact_w {r['exact_w']:.3f} {r['exact_w_ci']} dysp {r['dysp_w']:.3f} unw {r['exact_unweighted']:.3f} r50 {r['exact_random50']:.3f} jt {r['jury_test_fold_agreement']}",
          "vs_s0", r.get("exact_w_minus_rl_s0"), "vs_jury", r.get("exact_w_minus_jury"), r.get("changed_vs_rl_s0"))

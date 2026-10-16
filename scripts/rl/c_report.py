"""Follow-up C report: 3 seeds of SFT and SFT->GRPO on the 479 human sections (weighted), RL - SFT per seed (paired patient bootstrap),
mean and range, and RL vs MedGemma-27B (best seed and per seed). Out: rl_followups/C/results.json"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__)); from common import *
R = B + "/analysis/rl_grader"; FU = B + "/analysis/rl_followups"; NB = 2000; RNG = np.random.RandomState(20261016)
lab = {json.loads(l)["id"]: json.loads(l) for l in open(R + "/data/test485.jsonl")}; ids = sorted(lab)
w = np.array([lab[i]["w"] for i in ids]); hum = np.array([lab[i]["human"] for i in ids]); pats = np.array([lab[i]["anon_id"] for i in ids])
def load(f): d = {json.loads(l)["id"]: json.loads(l)["pred"] for l in open(f)}; return np.array([d.get(i) or "X" for i in ids])
def loadset(f): r = [json.loads(l) for l in open(f)]; return float(np.mean([x["pred"] == x["label"] for x in r]))
sel = {s: json.load(open(f"{FU}/C/preds/selected_grpo_sft_s{s}.json"))["selected"] for s in (1, 2)}
M = {"sft": {0: load(R + "/preds/sft__final__test485.jsonl"), 1: load(FU + "/C/preds/sft_s1__final__test485.jsonl"), 2: load(FU + "/C/preds/sft_s2__final__test485.jsonl")},
     "rl": {0: load(R + "/preds/grpo_sft__checkpoint-300__test485.jsonl"), 1: load(f"{FU}/C/preds/grpo_sft_s1__{sel[1]}__test485.jsonl"), 2: load(f"{FU}/C/preds/grpo_sft_s2__{sel[2]}__test485.jsonl")}}
mg = load(R + "/preds/medgemma27b_zs__test485.jsonl")
jt = {"sft": {0: loadset(R + "/preds/sft__final__test.jsonl"), 1: loadset(FU + "/C/preds/sft_s1__final__test.jsonl"), 2: loadset(FU + "/C/preds/sft_s2__final__test.jsonl")},
      "rl": {0: loadset(R + "/preds/grpo_sft__checkpoint-300__test.jsonl"), 1: loadset(f"{FU}/C/preds/grpo_sft_s1__{sel[1]}__test.jsonl"), 2: loadset(f"{FU}/C/preds/grpo_sft_s2__{sel[2]}__test.jsonl")}}
up = np.unique(pats); grp = {p: np.where(pats == p)[0] for p in up}; BS = [np.concatenate([grp[p] for p in RNG.choice(up, len(up))]) for _ in range(NB)]
ex = lambda g, ix: float(((g[ix] == hum[ix]) * w[ix]).sum() / w[ix].sum()); dy = lambda g, ix: float(((np.isin(g[ix], list(POS)) == np.isin(hum[ix], list(POS))) * w[ix]).sum() / w[ix].sum())
ALL = np.arange(len(ids)); ci = lambda v: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
res = {"selected_checkpoints": {0: "checkpoint-300 (project 1)", **sel}, "per_seed": {}}
for s in (0, 1, 2):
    a, b = M["sft"][s], M["rl"][s]
    res["per_seed"][s] = {"sft_exact": round(ex(a, ALL), 4), "rl_exact": round(ex(b, ALL), 4), "sft_dysp": round(dy(a, ALL), 4), "rl_dysp": round(dy(b, ALL), 4),
        "rl_minus_sft": {"point": round(ex(b, ALL) - ex(a, ALL), 4), "ci": ci([ex(b, ix) - ex(a, ix) for ix in BS])},
        "rl_minus_medgemma27b": {"point": round(ex(b, ALL) - ex(mg, ALL), 4), "ci": ci([ex(b, ix) - ex(mg, ix) for ix in BS])},
        "jury_test_fold_agreement": {"sft": round(jt["sft"][s], 4), "rl": round(jt["rl"][s], 4)}}
d = [res["per_seed"][s]["rl_minus_sft"]["point"] for s in (0, 1, 2)]
res["summary"] = {k: {"mean": round(float(np.mean([res["per_seed"][s][k] for s in (0, 1, 2)])), 4), "min": round(float(min(res["per_seed"][s][k] for s in (0, 1, 2))), 4), "max": round(float(max(res["per_seed"][s][k] for s in (0, 1, 2))), 4)} for k in ("sft_exact", "rl_exact", "sft_dysp", "rl_dysp")}
res["summary"]["rl_minus_sft"] = {"mean": round(float(np.mean(d)), 4), "min": round(float(min(d)), 4), "max": round(float(max(d)), 4)}
res["preregistered_rl_minus_sft_positive_in_all_3"] = bool(all(x > 0 for x in d)); res["medgemma27b_exact"] = round(ex(mg, ALL), 4)
best = max((0, 1, 2), key=lambda s: res["per_seed"][s]["rl_exact"]); res["best_seed"] = best
os.makedirs(FU + "/C", exist_ok=True); json.dump(res, open(FU + "/C/results.json", "w"), indent=1); print(json.dumps(res, indent=1)[:3000])

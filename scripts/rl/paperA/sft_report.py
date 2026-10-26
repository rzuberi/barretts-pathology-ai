"""Item 3 (CPU): completion-masked SFT (seeds 0-2) vs the original whole-sequence SFT and the RL model, per seed, on the 479 human
sections (weighted, paired patient bootstrap 2,000 as metrics.py) + SFT maj@8 (majority of 8 samples at T=1; ties -> greedy).
Out: O/sft/results.json"""
import json, os
from pa_common import *
RG = B + "/analysis/rl_grader/preds"; C = B + "/analysis/rl_followups/C/preds"; S = O + "/sft/preds"
SRC = {0: {"rl": RG + "/grpo_sft__checkpoint-300", "sft": RG + "/sft__final"}, 1: {"rl": C + "/grpo_sft_s1__checkpoint-300", "sft": C + "/sft_s1__final"}, 2: {"rl": C + "/grpo_sft_s2__checkpoint-250", "sft": C + "/sft_s2__final"}}
lab = {r["id"]: r for r in rows("test485")}; ids = sorted(lab); w = np.array([lab[i]["w"] for i in ids]); pats = np.array([lab[i]["anon_id"] for i in ids]); hum = np.array([lab[i]["human"] for i in ids])
BS = patient_bootstrap(pats); ALL = np.arange(len(ids))
def arr(prefix, s="test485", key="pred"): d = load_preds(prefix, s); return None if d is None else np.array([d[i].get(key) or "INVALID" for i in ids])
def maj8(prefix):
    d = load_preds(prefix, "test485")
    if d is None: return None
    out = []
    for i in ids:
        c = Counter(x for x in (d[i].get("samples") or []) if x); g = d[i].get("pred") or "INVALID"
        if c: top = max(c.values()); cands = [k for k, v in c.items() if v == top]; g = g if g in cands else cands[0]
        out.append(g)
    return np.array(out)
def ex(pr, ix): return float(((pr[ix] == hum[ix]) * w[ix]).sum() / w[ix].sum())
def dy(pr, ix): return float(((np.isin(pr[ix], list(POS)) == np.isin(hum[ix], list(POS))) * w[ix]).sum() / w[ix].sum())
def valtest(prefix): return {s: (round(float(np.mean([v["pred"] == v["label"] for v in d.values()])), 4) if (d := load_preds(prefix, s)) else None) for s in ("val", "test")}
R = {"seeds": {}}
for k in (0, 1, 2):
    P = {"rl": arr(SRC[k]["rl"]), "sft_orig": arr(SRC[k]["sft"]), "sft_masked": arr(f"{S}/sftm_s{k}"), "sft_masked_maj8": maj8(f"{S}/sftm_s{k}_maj8")}
    r = {}
    for m, pr in P.items():
        if pr is None: r[m] = None; continue
        r[m] = {"exact_w": round(ex(pr, ALL), 4), "exact_w_ci": ci([ex(pr, ix) for ix in BS]), "dysp_w": round(dy(pr, ALL), 4), "exact_unw": round(float((pr == hum).mean()), 4), "parse_rate": round(float((pr != "INVALID").mean()), 4)}
    r["sft_masked_valtest"] = valtest(f"{S}/sftm_s{k}"); r["sft_orig_valtest"] = valtest(SRC[k]["sft"])
    for a, b_ in (("rl", "sft_masked"), ("rl", "sft_masked_maj8"), ("sft_masked", "sft_orig"), ("rl", "sft_orig")):
        if P[a] is not None and P[b_] is not None: r[f"{a}_minus_{b_}"] = {"point": round(ex(P[a], ALL) - ex(P[b_], ALL), 4), "ci": ci([ex(P[a], ix) - ex(P[b_], ix) for ix in BS])}
    R["seeds"][k] = r
have = [k for k in (0, 1, 2) if R["seeds"][k]["sft_masked"]]
if have:
    for a, b_ in (("rl", "sft_masked"), ("rl", "sft_masked_maj8")):
        arrs = [(arr(SRC[k]["rl"]), (arr(f"{S}/sftm_s{k}") if b_ == "sft_masked" else maj8(f"{S}/sftm_s{k}_maj8"))) for k in have]
        arrs = [(x, y) for x, y in arrs if y is not None]
        if arrs: R[f"mean_of_seeds_{a}_minus_{b_}"] = {"n_seeds": len(arrs), "point": round(float(np.mean([ex(x, ALL) - ex(y, ALL) for x, y in arrs])), 4), "ci": ci([np.mean([ex(x, ix) - ex(y, ix) for x, y in arrs]) for ix in BS])}
mkdir(O + "/sft"); json.dump(R, open(O + "/sft/results.json", "w"), indent=1); os.chmod(O + "/sft/results.json", 0o600)
for k, r in R["seeds"].items():
    print(f"seed {k}: " + " | ".join(f"{m} {100*v['exact_w']:.1f} {[round(100*x,1) for x in v['exact_w_ci']]} (unw {100*v['exact_unw']:.1f}, dysp {100*v['dysp_w']:.1f}, parse {v['parse_rate']})" for m, v in r.items() if isinstance(v, dict) and "exact_w" in v))
    print("   ", {kk: (round(100*v["point"], 1), [round(100*x, 1) for x in v["ci"]]) for kk, v in r.items() if "minus" in kk}, "val/test masked", r["sft_masked_valtest"], "orig", r["sft_orig_valtest"])
print({k: v for k, v in R.items() if k.startswith("mean")})

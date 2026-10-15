"""Leave-one-family-out jury (GLM proposal; answers the correlated-LLM-errors
criticism with a number). Rebuild ERIN corpus majorities dropping each model
FAMILY (both qwen3 sizes together, both gemma3 sizes together, etc.); readout:
label flip rate vs the full jury, train-eligible set churn, and whether any
single family is load-bearing.
"""
import glob, json, os
import pandas as pd

T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"
OUT = os.environ.get("OUTDIR", ".")
GRADES = {"NDBE", "IND", "LGD", "HGD", "CANCER"}
FAMILIES = {"qwen3": ["qwen3_14b", "qwen3_32b"], "gemma3": ["gemma3_12b", "gemma3_27b"],
            "phi4": ["phi4_14b"], "mistral": ["mistral-small3.2"],
            "deepseek": ["deepseek-r1_14b"], "llama": ["llama3.1_8b"]}

votes = {}
for f in glob.glob(T + "/labeller/llm_full/llm_grades_*.csv"):
    model = os.path.basename(f).replace("llm_grades_", "").rsplit("_shard", 1)[0]
    for line in open(f).read().splitlines()[1:]:
        cid, _, g = line.rpartition(",")
        if g in GRADES:
            votes.setdefault(model, {})[cid] = g
print("models:", {k: len(v) for k, v in votes.items()}, flush=True)

def majority(models):
    out = {}
    ids = set.union(*(set(votes[m]) for m in models if m in votes))
    for cid in ids:
        vs = [votes[m][cid] for m in models if m in votes and cid in votes[m]]
        if len(vs) < max(4, len(models) - 2): continue
        top = max(set(vs), key=vs.count)
        out[cid] = (top, vs.count(top) / len(vs))
    return out

all_models = sorted(votes)
full = majority(all_models)
full_elig = {c for c, (g, fr) in full.items() if fr >= 0.75}
res = {"_meta": {"models": all_models, "n_reports": len(full),
                 "full_train_eligible": len(full_elig)}}
for fam, members in FAMILIES.items():
    kept = [m for m in all_models if m not in members]
    lofo = majority(kept)
    common = set(full) & set(lofo)
    flips = sum(full[c][0] != lofo[c][0] for c in common)
    lofo_elig = {c for c, (g, fr) in lofo.items() if fr >= 0.75}
    res[f"drop_{fam}"] = {
        "n_jurors_left": len(kept), "n_common": len(common),
        "label_flip_rate": round(flips / len(common), 4),
        "eligible_change": len(lofo_elig & set(common)) - len(full_elig & common),
        "eligible_jaccard": round(len(lofo_elig & full_elig) /
                                  max(len(lofo_elig | full_elig), 1), 4)}
    print(fam, res[f"drop_{fam}"], flush=True)
worst = max((res[f"drop_{f}"]["label_flip_rate"], f) for f in FAMILIES)
res["verdict"] = {"max_flip_rate": worst[0], "most_load_bearing_family": worst[1]}
json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=2)
print("wrote results.json")

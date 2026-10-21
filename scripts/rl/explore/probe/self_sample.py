"""Probe strand part 2: the p1 model's own 8-sample accuracy (T=1, project-1 prompt) on exactly the GRPO training items that
train_selfk.py will use (same class-capped selection, SEED=0, CAP=1200, + all_lowconf uncapped). Out: OUTD/ownacc.json (id -> own
accuracy vs jury consensus) and OUTD/ownacc_summary.json. Env: OUTD, N (smoke cap), SETS=val (smoke: sample val rows instead)."""
import json, os, random, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")); from common import *
P1 = B + "/analysis/rl_grader/runs/p1_best_merged"; D = B + "/analysis/rl_grader/data"; OUTD = os.environ["OUTD"]; os.makedirs(OUTD, exist_ok=True); N = int(os.environ.get("N", 0)); K = 8
if os.environ.get("SETS", "train") == "val": rows = [json.loads(l) for l in open(D + "/val.jsonl")]
else:
    rows = [json.loads(l) for l in open(D + "/train.jsonl")]; extra = [json.loads(l) for l in open(D + "/all_lowconf.jsonl")]
    rnd = random.Random(0); by = {}
    for r in rows: by.setdefault(r["label"], []).append(r)
    rows = [x for g in by.values() for x in rnd.sample(g, min(1200, len(g)))]; rows += extra; rnd.shuffle(rows)
if N: rows = rows[:N]
print(f"sampling {K} completions for {len(rows)} items", flush=True)
from vllm import LLM, SamplingParams
llm = LLM(model=P1, max_model_len=3584, gpu_memory_utilization=0.85, dtype="bfloat16", seed=0, enforce_eager=True, limit_mm_per_prompt={"image": 0})
msgs = [make_messages(r["report"], r["section"]) for r in rows]; t0 = time.time()
res = llm.chat(msgs, SamplingParams(temperature=1.0, max_tokens=384, n=K, seed=0), use_tqdm=False)
own = {}; rowsum = []
for r, o in zip(rows, res):
    g = [parse_grade(x.text) for x in o.outputs]; a = float(np.mean([x == r["label"] for x in g])); own[r["id"]] = a
    rowsum.append({"id": r["id"], "label": r["label"], "vf": r.get("vf"), "own_acc": a, "n_invalid": sum(x is None for x in g), "n_distinct": len(set(g))})
json.dump(own, open(OUTD + "/ownacc.json", "w"))
with open(OUTD + "/ownacc_rows.jsonl", "w") as f:
    for d in rowsum: f.write(json.dumps(d) + "\n")
a = np.array([d["own_acc"] for d in rowsum]); vf = np.array([d["vf"] if d["vf"] is not None else 1.0 for d in rowsum])
summ = {"n": len(rows), "seconds": round(time.time() - t0), "mean_own_acc": round(float(a.mean()), 4), "frac_own_lt_0.5": round(float((a < 0.5).mean()), 4), "frac_own_eq_1": round(float((a == 1).mean()), 4),
        "frac_own_lt_0.5_among_vf_lt_0.75": round(float((a[vf < 0.75] < 0.5).mean()), 4) if (vf < 0.75).any() else None, "frac_own_lt_0.5_among_vf_ge_0.75": round(float((a[vf >= 0.75] < 0.5).mean()), 4),
        "by_label": {g: {"n": int((np.array([d["label"] for d in rowsum]) == g).sum()), "frac_own_lt_0.5": round(float((a[np.array([d["label"] for d in rowsum]) == g] < 0.5).mean()), 4)} for g in GRADES if any(d["label"] == g for d in rowsum)}}
json.dump(summ, open(OUTD + "/ownacc_summary.json", "w"), indent=1); print("OWNACC", summ, flush=True); print("SAMPLE DONE", flush=True)

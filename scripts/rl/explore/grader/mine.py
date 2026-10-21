"""Explore/grader: mine 'mixed' prompts. Samples 8 completions at T=1 (as GRPO does) from the SFT model for every GRPO training prompt
(same class-capped seed-0 sample as project 1, plus the low-agreement sections) and records the 8-sample accuracy against the jury label.
GRPO groups where all 8 samples agree have zero reward variance and no gradient; run a4 trains only on prompts with 0 < acc < 1.
Env: BASE (sft_merged), OUT (json path). Out: {id: {"acc": k/8, "n_valid": v, "grades": {grade: count}}} - no report text."""
import json, os, random, sys
from collections import Counter
sys.path.insert(0, "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis/scripts/rl"); from common import *
BASE = os.environ["BASE"]; OUT = os.environ["OUT"]; D = B + "/analysis/rl_grader/data"; CAP = 1200
rows = [json.loads(l) for l in open(D + "/train.jsonl")]; rnd = random.Random(0)
by = {}
for r in rows: by.setdefault(r["label"], []).append(r)
rows = [x for g in by.values() for x in rnd.sample(g, min(CAP, len(g)))]
rows += [json.loads(l) for l in open(D + "/all_lowconf.jsonl")]
print(f"mining {len(rows)} prompts", flush=True)
from vllm import LLM, SamplingParams
llm = LLM(model=BASE, max_model_len=3584, gpu_memory_utilization=0.85, dtype="bfloat16", seed=0, enforce_eager=True, limit_mm_per_prompt={"image": 0})
sp = SamplingParams(temperature=1.0, max_tokens=256, n=8, seed=0)
res = llm.chat([make_messages(r["report"], r["section"]) for r in rows], sp, use_tqdm=False)
out = {}
for r, o in zip(rows, res):
    gs = [parse_grade(x.text) for x in o.outputs]; c = Counter(g or "INVALID" for g in gs)
    out[r["id"]] = {"acc": sum(g == r["label"] for g in gs) / len(gs), "n_valid": sum(g is not None for g in gs), "grades": dict(c), "label": r["label"], "vf": r.get("vf")}
json.dump(out, open(OUT, "w")); os.chmod(OUT, 0o600)
acc = [v["acc"] for v in out.values()]; mixed = sum(0 < a < 1 for a in acc)
print(f"MINED {len(out)} prompts: all-right {sum(a == 1 for a in acc)}, all-wrong {sum(a == 0 for a in acc)}, mixed {mixed} ({mixed / len(acc):.3f})", flush=True)
hi = [v for v in out.values() if (v['vf'] or 1) >= 0.75]; lo = [v for v in out.values() if (v['vf'] or 1) < 0.75]
print(f"  mixed among vf>=0.75: {sum(0 < v['acc'] < 1 for v in hi)}/{len(hi)}; among low-agreement: {sum(0 < v['acc'] < 1 for v in lo)}/{len(lo)}", flush=True)
print("MINE DONE", flush=True)

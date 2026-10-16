"""Project 3: 2 paraphrases per base question with a local LLM (vLLM, Qwen3-4B). A paraphrase is kept only if it preserves every
number in the original and is not identical to it. Out: analysis/rl_sql/questions.jsonl (originals + paraphrases, variant 0..2)."""
import json, os, re, sys
O = "/mnt/scratche/slow/fmlab/zuberi01/analysis/rl_sql"; HF = os.environ["HF_HOME"]
base = [json.loads(l) for l in open(O + "/questions_base.jsonl")]
from vllm import LLM, SamplingParams
mp = [d for d in os.popen(f"ls -d {HF}/hub/models--Qwen--Qwen3-4B/snapshots/*/").read().split()][0]
llm = LLM(model=mp, max_model_len=2048, gpu_memory_utilization=0.85, seed=0)
P = ("Rewrite this database question in two different ways a researcher might ask it. Keep exactly the same meaning and every number, year "
     "and grade name. Output two lines, one paraphrase per line, nothing else.\n\nQuestion: {q} /no_think")
outs = llm.chat([[{"role": "user", "content": P.format(q=b["question"])}] for b in base], SamplingParams(temperature=0.7, max_tokens=160, seed=0), use_tqdm=False)
nums = lambda s: sorted(re.findall(r"\d+(?:\.\d+)?", s))
rows, kept = [], 0
for b, o in zip(base, outs):
    rows.append({**b, "variant": 0})
    lines = [re.sub(r"^\s*(\d+[.)]|[-*])\s*", "", l).strip() for l in re.sub(r"<think>.*?</think>", "", o.outputs[0].text, flags=re.S).splitlines() if l.strip()]
    v = 1
    for l in lines[:3]:
        if v > 2: break
        if l.lower() == b["question"].lower() or nums(l) != nums(b["question"]) or len(l) < 10: continue
        rows.append({**b, "question": l, "variant": v}); v += 1; kept += 1
with open(O + "/questions.jsonl", "w") as f:
    for r in rows: f.write(json.dumps(r) + "\n")
os.chmod(O + "/questions.jsonl", 0o600)
from collections import Counter
print(f"{len(base)} base -> {len(rows)} questions ({kept} paraphrases kept);", Counter((r["split"], r["answerable"]) for r in rows))

"""Explore/agent, training-diversity step 2 (GPU, Qwen3-4B as in sql_paraphrase.py): 2 fresh paraphrases (seed 1) of every
answerable TRAINING base question (original variants 0 -> new variants 3,4) and of each new augmented base question (variants 1,2).
Kept only if every number survives and the text is new. Out: $O2/questions_aug.jsonl = questions_agent.jsonl + new training rows. Env: HF_HOME, N (smoke)."""
import json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from agent2 import O, O2
HF = os.environ["HF_HOME"]; N = int(os.environ.get("N", 0))
qs = [json.loads(l) for l in open(O + "/questions_agent.jsonl")]; aug = [json.loads(l) for l in open(O2 + "/aug_base.jsonl")]
orig = [q for q in qs if q["split"] == "train" and q["answerable"] and q["variant"] == 0]
work = [(q, 3) for q in orig] + [(q, 1) for q in aug]; work = work[:N] if N else work
from vllm import LLM, SamplingParams
mp = os.popen(f"ls -d {HF}/hub/models--Qwen--Qwen3-4B/snapshots/*/").read().split()[0]
llm = LLM(model=mp, max_model_len=2048, gpu_memory_utilization=0.85, seed=1)
P = ("Rewrite this database question in two different ways a researcher might ask it. Keep exactly the same meaning and every number, year "
     "and grade name. Output two lines, one paraphrase per line, nothing else.\n\nQuestion: {q} /no_think")
outs = llm.chat([[{"role": "user", "content": P.format(q=b["question"])}] for b, _ in work], SamplingParams(temperature=0.9, max_tokens=160, seed=1), use_tqdm=False)
nums = lambda s: sorted(re.findall(r"\d+(?:\.\d+)?", s)); existing = {q["question"].lower() for q in qs}
new = [dict(q, aug=1) for q in aug[:N] if True] if N else [dict(q, aug=1) for q in aug]; kept = 0
for (b, v0), o in zip(work, outs):
    lines = [re.sub(r"^\s*(\d+[.)]|[-*])\s*", "", l).strip() for l in re.sub(r"<think>.*?</think>", "", o.outputs[0].text, flags=re.S).splitlines() if l.strip()]
    v = v0
    for l in lines[:3]:
        if v > v0 + 1: break
        if l.lower() in existing or nums(l) != nums(b["question"]) or len(l) < 10: continue
        existing.add(l.lower()); new.append({**{k: b[k] for k in ("template", "type", "answerable", "gold_sql", "answer", "split")}, "question": l, "variant": v, "aug": 1}); v += 1; kept += 1
OUTQ = os.environ.get("OUTQ", O2 + "/questions_aug.jsonl")
with open(OUTQ, "w") as f:
    for r in qs + new: f.write(json.dumps(r) + "\n")
os.chmod(OUTQ, 0o600)
from collections import Counter
print(f"{len(work)} inputs -> {kept} paraphrases kept; {len(new)} new training rows; train answerable now {sum(1 for r in qs + new if r['split'] == 'train' and r['answerable'])} (was {sum(1 for r in qs if r['split'] == 'train' and r['answerable'])});",
      dict(Counter(r["template"] for r in new)))

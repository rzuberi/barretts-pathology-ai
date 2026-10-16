"""Project 3 evaluation with vLLM (greedy): zero-shot, 5-shot (fixed 5 train-template examples, seed 0) or a LoRA checkpoint (merged).
Env: BASE, ADAPTER, TAG, SHOTS (0|5), SETS (val,test), N. Out: analysis/rl_sql/preds/<TAG>__<set>.jsonl + summary line."""
import json, os, random, shutil, sys, tempfile, time
sys.path.insert(0, os.path.dirname(__file__)); from sql_common import *
BASE = os.environ["BASE"]; AD = os.environ.get("ADAPTER", ""); TAG = os.environ["TAG"]; SH = int(os.environ.get("SHOTS", 0)); N = int(os.environ.get("N", 0))
SETS = os.environ.get("SETS", "val,test").split(","); PD = os.environ.get("PREDS", O + "/preds"); os.makedirs(PD, exist_ok=True)
qs = [json.loads(l) for l in open(os.environ.get("QFILE", O + "/questions.jsonl"))]
tr = [q for q in qs if q["split"] == "train" and q["variant"] == 0]; rnd = random.Random(0)
shots = rnd.sample([q for q in tr if q["answerable"]], 4) + rnd.sample([q for q in tr if not q["answerable"]], 1) if SH else None
model_dir = BASE; tmp = None
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
LORA = LoRARequest('adapter', 1, AD) if AD else None
llm = LLM(model=model_dir, max_model_len=4096, gpu_memory_utilization=0.85, seed=0, enable_lora=bool(AD), max_lora_rank=16)
for s in SETS:
    items = [q for q in qs if q["split"] == s]; items = items[:N] if N else items
    t0 = time.time(); outs = llm.chat([messages(q["question"], shots) for q in items], SamplingParams(temperature=0, max_tokens=384), use_tqdm=False, lora_request=LORA); dt = time.time() - t0
    rows = []
    for q, o in zip(items, outs):
        r, ok, kind = score(o.outputs[0].text, q); rows.append({"template": q["template"], "type": q["type"], "answerable": q["answerable"], "variant": q["variant"], "correct": ok, "kind": kind, "reward": r, "raw": o.outputs[0].text[:400]})
    with open(f"{PD}/{TAG}__{s}.jsonl", "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    a = [r for r in rows if r["answerable"]]; u = [r for r in rows if not r["answerable"]]
    print(f"{TAG} {s}: n={len(rows)} exec_acc(answerable)={sum(r['correct'] for r in a) / max(len(a), 1):.3f} refusal(unanswerable)={sum(r['correct'] for r in u) / max(len(u), 1):.3f} {dt:.0f}s", flush=True)
if tmp: shutil.rmtree(tmp, ignore_errors=True)
print("EVAL DONE")

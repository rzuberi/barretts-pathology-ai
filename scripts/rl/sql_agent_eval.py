"""Task E: evaluate the agent (greedy) on val/test of questions_agent.jsonl. Env: BASE, ADAPTER, TAG, SETS, N."""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(__file__)); from sql_agent import *
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
BASE = os.environ["BASE"]; AD = os.environ.get("ADAPTER", ""); TAG = os.environ["TAG"]; SETS = os.environ.get("SETS", "val,test").split(","); N = int(os.environ.get("N", 0))
PD = O + "/agent_preds"; os.makedirs(PD, exist_ok=True)
qs = [json.loads(l) for l in open(O + "/questions_agent.jsonl")]
llm = LLM(model=BASE, max_model_len=8192, gpu_memory_utilization=0.85, seed=0, enable_lora=bool(AD), max_lora_rank=16)
lora = LoRARequest("a", 1, AD) if AD else None; sp = SamplingParams(temperature=0, max_tokens=320)
for s in SETS:
    items = [q for q in qs if q["split"] == s]; items = items[:N] if N else items; t0 = time.time()
    eps = run_episodes(llm, items, sp, lora)
    with open(f"{PD}/{TAG}__{s}.jsonl", "w") as f:
        for e in eps: f.write(json.dumps({"template": e["item"]["template"], "type": e["item"]["type"], "answerable": e["item"]["answerable"], "variant": e["item"]["variant"], "correct": e["correct"], "kind": e["kind"], "n_calls": e["n_calls"], "turns": [m["content"][:300] for m in e["messages"][2:]]}) + "\n")
    a = [e for e in eps if e["item"]["answerable"]]; u = [e for e in eps if not e["item"]["answerable"]]
    print(f"{TAG} {s}: n={len(eps)} exec_acc={sum(e['correct'] for e in a) / max(len(a), 1):.3f} refusal={sum(e['correct'] for e in u) / max(len(u), 1):.3f} mean_calls={sum(e['n_calls'] for e in eps) / len(eps):.2f} {time.time() - t0:.0f}s", flush=True)
print("EVAL DONE")

"""Explore/agent: greedy agent evaluation of one base model with 0..n LoRA adapters in a single vLLM session.
Env: BASE, ADAPTERS ("tag=path,tag=path" or "" -> tag NOADAPTER via TAG), TAG (prefix), SETS, N, QFILE, PREDS."""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from agent2 import *
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
BASE = os.environ["BASE"]; ADS = [a for a in os.environ.get("ADAPTERS", "").split(",") if a]; TAG = os.environ["TAG"]
SETS = os.environ.get("SETS", "val").split(","); N = int(os.environ.get("N", 0)); PD = os.environ.get("PREDS", O2 + "/preds"); os.makedirs(PD, exist_ok=True)
qs = [json.loads(l) for l in open(os.environ.get("QFILE", O + "/questions_agent.jsonl"))]
llm = LLM(model=BASE, max_model_len=8192, gpu_memory_utilization=0.85, seed=0, enable_lora=bool(ADS), max_lora_rank=16)
sp = SamplingParams(temperature=0, max_tokens=320)
runs = [(f"{TAG}__{t}", LoRARequest(t, i + 1, p)) for i, (t, p) in enumerate(a.split("=", 1) for a in ADS)] or [(TAG, None)]
for tag, lora in runs:
    for s in SETS:
        items = [q for q in qs if q["split"] == s]; items = items[:N] if N else items; t0 = time.time()
        eps = run_episodes(llm, items, sp, lora)
        with open(f"{PD}/{tag}__{s}.jsonl", "w") as f:
            for e in eps: f.write(json.dumps({"template": e["item"]["template"], "type": e["item"]["type"], "answerable": e["item"]["answerable"], "variant": e["item"]["variant"], "correct": e["correct"], "kind": e["kind"], "n_calls": e["n_calls"], "n_probe_ok": e["n_probe_ok"], "turns": [m["content"][:300] for m in e["messages"][2:]]}) + "\n")
        os.chmod(f"{PD}/{tag}__{s}.jsonl", 0o600)
        a = [e for e in eps if e["item"]["answerable"]]; u = [e for e in eps if not e["item"]["answerable"]]
        print(f"{tag} {s}: n={len(eps)} exec_acc={sum(e['correct'] for e in a) / max(len(a), 1):.3f} refusal={sum(e['correct'] for e in u) / max(len(u), 1):.3f} false_ref={sum(e['kind'] == 'refused' for e in a) / max(len(a), 1):.3f} mean_calls={sum(e['n_calls'] for e in eps) / len(eps):.2f} {time.time() - t0:.0f}s", flush=True)
print("EVAL DONE")

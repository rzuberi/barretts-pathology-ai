"""Explore/agent, expert iteration step 1: sample K episodes per TRAINING question from each adapter (and the bare base when
'none=' is given) at temperature TEMP; write every episode with its outcome. Env: BASE, ADAPTERS ("tag=path,none="), K, TEMP, QFILE, OUT, N."""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from agent2 import *
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
BASE = os.environ["BASE"]; ADS = [a.split("=", 1) for a in os.environ["ADAPTERS"].split(",") if a]; K = int(os.environ.get("K", 8)); N = int(os.environ.get("N", 0))
qs = [json.loads(l) for l in open(os.environ.get("QFILE", O + "/questions_agent.jsonl"))]; tr = [q for q in qs if q["split"] == "train"]; tr = tr[:N] if N else tr
llm = LLM(model=BASE, max_model_len=8192, gpu_memory_utilization=0.85, seed=0, enable_lora=True, max_lora_rank=16)
sp = SamplingParams(temperature=float(os.environ.get("TEMP", 1.0)), max_tokens=320, seed=None)
out = os.environ["OUT"]; os.makedirs(os.path.dirname(out), exist_ok=True); n_ok = 0; n = 0
with open(out, "w") as f:
    for i, (tag, path) in enumerate(ADS):
        lora = LoRARequest(tag, i + 1, path) if path else None; t0 = time.time()
        eps = run_episodes(llm, [q for q in tr for _ in range(K)], sp, lora)
        for e in eps:
            f.write(json.dumps({"source": tag, "template": e["item"]["template"], "answerable": e["item"]["answerable"], "variant": e["item"]["variant"], "question": e["item"]["question"], "correct": e["correct"], "kind": e["kind"], "n_calls": e["n_calls"], "messages": e["messages"]}) + "\n")
        k = sum(e["correct"] for e in eps); n_ok += k; n += len(eps)
        print(f"{tag}: {len(eps)} episodes, {k} correct ({k / len(eps):.3f}), {time.time() - t0:.0f}s", flush=True)
os.chmod(out, 0o600); print(f"GEN DONE {n_ok}/{n} correct episodes -> {out}")

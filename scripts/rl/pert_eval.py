"""Task F evaluation: next-token probabilities of up/down/none (vLLM, greedy, 1 token, top-20 logprobs, thinking disabled).
Env: BASE, ADAPTER, TAG, SETS (val,val_nor,test), CELLS (filter), N. Out: F/preds/<TAG>__<set>.jsonl with p_up, p_down, p_none."""
import json, math, os, sys, time
sys.path.insert(0, os.path.dirname(__file__)); from pert_common import *
BASE = os.environ["BASE"]; AD = os.environ.get("ADAPTER", ""); TAG = os.environ["TAG"]; SETS = os.environ.get("SETS", "val,test").split(","); N = int(os.environ.get("N", 0))
CF = set(os.environ.get("CELLS", ",".join(CELLS)).split(",")); PD = O + "/preds"; os.makedirs(PD, exist_ok=True)
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
llm = LLM(model=BASE, max_model_len=512, gpu_memory_utilization=0.85, seed=0, enable_lora=bool(AD), max_lora_rank=16, enforce_eager=True)
LORA = LoRARequest("a", 1, AD) if AD else None; sp = SamplingParams(temperature=0, max_tokens=1, logprobs=20)
VAR = {w: {w, w.capitalize(), w.upper(), " " + w, " " + w.capitalize()} for w in WORDS}
for s in SETS:
    out = f"{PD}/{TAG}__{s}.jsonl"
    if os.path.exists(out): continue
    rows = [json.loads(l) for l in open(f"{O}/data/{s}.jsonl")]; rows = [r for r in rows if r["cell"] in CF]; rows = rows[:N] if N else rows
    t0 = time.time(); res = llm.chat([messages(r["cell"], r["pert"], r["gene"]) for r in rows], sp, use_tqdm=False, lora_request=LORA, chat_template_kwargs={"enable_thinking": False})
    with open(out + ".part", "w") as f:
        for r, o in zip(rows, res):
            lp = o.outputs[0].logprobs[0] if o.outputs[0].logprobs else {}; p = {w: 0.0 for w in WORDS}
            for tid, v in lp.items():
                for w in WORDS:
                    if (v.decoded_token or "") in VAR[w]: p[w] += math.exp(v.logprob)
            z = sum(p.values()) or 1e-9
            f.write(json.dumps({**{k: r[k] for k in ("cell", "pert", "gene", "label", "de", "dir")}, "p_up": p["up"] / z, "p_down": p["down"] / z, "p_none": p["none"] / z, "mass": z, "top": o.outputs[0].text}) + "\n")
    os.replace(out + ".part", out); print(f"{TAG} {s}: {len(rows)} items in {time.time() - t0:.0f}s", flush=True)
print("EVAL DONE")

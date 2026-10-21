"""Explore/pert evaluation: next-token probabilities of up/down/none (vLLM, greedy, 1 token, top-20 logprobs, thinking off).
Copy of scripts/rl/pert_eval.py; one vLLM init, several adapters. Env: BASE, ADAPTERS (comma list of adapter dirs; '' = base),
TAG (prefix; tag = TAG__<adapter basename>), SETS, CELLS, N, PROMPT, THINK (1 = generate a capped thinking trace first, then read the
answer-word probabilities in a second pass). Out: O/preds/<tag>__<set>.jsonl"""
import json, math, os, sys, time
sys.path.insert(0, os.path.dirname(__file__)); from xpert_common import *
BASE = os.environ["BASE"]; ADS = [a for a in os.environ.get("ADAPTERS", "").split(",")]; TAG = os.environ["TAG"]; SETS = os.environ.get("SETS", "val,test").split(","); N = int(os.environ.get("N", 0))
CF = set(os.environ.get("CELLS", ",".join(CELLS)).split(",")); PD = O + "/preds"; os.makedirs(PD, exist_ok=True); THINK = os.environ.get("THINK", "0") == "1"; MAXTHINK = int(os.environ.get("MAXTHINK", 300))
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
llm = LLM(model=BASE, max_model_len=int(os.environ.get("MAXMODEL", 1024)), gpu_memory_utilization=0.85, seed=0, enable_lora=any(ADS), max_lora_rank=16, enforce_eager=True)
sp = SamplingParams(temperature=0, max_tokens=1, logprobs=20); spt = SamplingParams(temperature=0, max_tokens=MAXTHINK, stop=["</think>"], include_stop_str_in_output=True)
VAR = {w: {w, w.capitalize(), w.upper(), " " + w, " " + w.capitalize()} for w in WORDS}
for i, AD in enumerate(ADS):
    LORA = LoRARequest(f"a{i}", i + 1, AD) if AD else None; tag = f"{TAG}__{os.path.basename(AD.rstrip('/'))}" if AD else TAG
    for s in SETS:
        out = f"{PD}/{tag}__{s}.jsonl"
        if os.path.exists(out): continue
        rows = [json.loads(l) for l in open(data_path(s))]; rows = [r for r in rows if r["cell"] in CF]; rows = rows[:N] if N else rows
        t0 = time.time(); msgs = [messages(r["cell"], r["pert"], r["gene"], r.get("ret"), think=THINK) for r in rows]; thinks = None
        if THINK:  # pass 1: thinking trace (capped); pass 2: answer-word probabilities after the closed trace
            g = llm.chat(msgs, spt, use_tqdm=False, lora_request=LORA, chat_template_kwargs={"enable_thinking": True})
            thinks = [o.outputs[0].text if o.outputs[0].text.rstrip().endswith("</think>") else o.outputs[0].text + "\n</think>\n" for o in g]
            if not thinks[0].lstrip().startswith("<think>"): thinks = ["<think>\n" + t for t in thinks]
            pr = [llm.get_tokenizer().apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=True) + t for m, t in zip(msgs, thinks)]
            res = llm.generate(pr, sp, use_tqdm=False, lora_request=LORA)
        else: res = llm.chat(msgs, sp, use_tqdm=False, lora_request=LORA, chat_template_kwargs={"enable_thinking": False})
        with open(out + ".part", "w") as f:
            for j, (r, o) in enumerate(zip(rows, res)):
                lp = o.outputs[0].logprobs[0] if o.outputs[0].logprobs else {}; p = {w: 0.0 for w in WORDS}
                for tid, v in lp.items():
                    for w in WORDS:
                        if (v.decoded_token or "") in VAR[w]: p[w] += math.exp(v.logprob)
                z = sum(p.values()) or 1e-9
                f.write(json.dumps({**{k: r[k] for k in ("cell", "pert", "gene", "label", "de", "dir")}, "p_up": p["up"] / z, "p_down": p["down"] / z, "p_none": p["none"] / z, "mass": z, "top": o.outputs[0].text,
                                    **({"think": thinks[j][:2000]} if thinks else {})}) + "\n")
        os.replace(out + ".part", out); print(f"{tag} {s}: {len(rows)} items in {time.time() - t0:.0f}s", flush=True)
print("EVAL DONE")

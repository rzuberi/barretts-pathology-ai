"""Fallback evaluator with plain Transformers batched generation (greedy), for models vLLM runs incorrectly (MedGemma/Gemma 3 on this
cluster's driver). Same env, outputs and file format as evaluate.py. Env: BASE, ADAPTER, TAG, SETS, N, UNSURE, LOGPROBS, BS (batch size)."""
import json, math, os, sys, time
import torch
sys.path.insert(0, os.path.dirname(__file__)); from common import *
from transformers import AutoProcessor, AutoModelForImageTextToText, AutoModelForCausalLM
BASE = os.environ.get("BASE"); AD = os.environ.get("ADAPTER", ""); TAG = os.environ["TAG"]; N = int(os.environ.get("N", 0)); BS = int(os.environ.get("BS", 16))
SETS = os.environ.get("SETS", "val,test,test485").split(","); UNS = os.environ.get("UNSURE") == "1"; LP = os.environ.get("LOGPROBS") == "1"
D = B + "/analysis/rl_grader/data"; P = os.environ.get("PREDS", B + "/analysis/rl_grader/preds"); os.makedirs(P, exist_ok=True)
if all(os.path.exists(f"{P}/{TAG}__{s}.jsonl") for s in SETS): print("done already"); sys.exit(0)
proc = AutoProcessor.from_pretrained(BASE); tok = proc.tokenizer; tok.padding_side = "left"
try: m = AutoModelForImageTextToText.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map="cuda")
except Exception: m = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map="cuda")
if AD:
    from peft import PeftModel
    try: m = PeftModel.from_pretrained(m, AD).merge_and_unload()
    except Exception as e:   # adapters trained on the CausalLM wrapper: load into the same class
        print("adapter into ImageTextToText failed, using CausalLM:", repr(e)[:120], flush=True)
        m = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map="cuda"); m = PeftModel.from_pretrained(m, AD).merge_and_unload()
m.eval()
def enc(msgs):
    mm = [{"role": x["role"], "content": [{"type": "text", "text": x["content"]}]} for x in msgs]
    return proc.apply_chat_template(mm, add_generation_prompt=True, tokenize=False)
for s in SETS:
    out = f"{P}/{TAG}__{s}.jsonl"
    if os.path.exists(out): continue
    rows = [json.loads(l) for l in open(f"{D}/{s}.jsonl")]; rows = rows[:N] if N else rows
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]["report"]))   # length-sorted batches
    texts = [enc(make_messages(r["report"], r["section"], allow_unsure=UNS)) for r in rows]; res = [None] * len(rows); pg = [None] * len(rows)
    t0 = time.time()
    for i in range(0, len(order), BS):
        idx = order[i:i + BS]; b = tok([texts[j] for j in idx], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.inference_mode(): g = m.generate(**b, max_new_tokens=384, do_sample=False, output_scores=LP, return_dict_in_generate=True)
        seqs = g.sequences[:, b["input_ids"].shape[1]:]
        for k, j in enumerate(idx):
            res[j] = tok.decode(seqs[k], skip_special_tokens=True)
            if LP:   # probability of the first token after '"grade": "'
                toks = [tok.decode([t]) for t in seqs[k]]; acc = ""
                for step, t in enumerate(toks):
                    if acc.rstrip().endswith('"grade": "') or acc.rstrip().endswith('"grade":"'):
                        pg[j] = float(torch.softmax(g.scores[step][k].float(), -1).max()); break
                    acc += t
    dt = time.time() - t0
    with open(out + ".part", "w") as f:
        for r, t, p_ in zip(rows, res, pg):
            d = {k: r[k] for k in r if k != "report"}; d.update({"raw": t[:600], "pred": parse_grade(t, allow_unsure=UNS)})
            if LP: d["p_grade"] = p_
            f.write(json.dumps(d) + "\n")
    os.replace(out + ".part", out); nrep = len({r["report_id"] for r in rows})
    json.dump({"tag": TAG, "set": s, "n": len(rows), "reports": nrep, "seconds": dt, "engine": "transformers", "batch": BS}, open(f"{P}/{TAG}__{s}.timing.json", "w"))
    print(f"{TAG} {s}: {len(rows)} sections, {nrep} reports, {dt:.0f}s ({dt / max(nrep, 1):.2f} s/report), parse rate {sum(1 for t in res if parse_grade(t, allow_unsure=UNS)) / len(rows):.3f}", flush=True)
print("EVAL DONE", flush=True)

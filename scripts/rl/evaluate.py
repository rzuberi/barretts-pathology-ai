"""Evaluate a base model or a LoRA checkpoint with vLLM (greedy) on val / test / test485. Merges the adapter into a temp dir first.
Env: BASE, ADAPTER (dir or empty = zero-shot), TAG, SETS (comma list: val,test,test485), N (cap per set, smoke), UNSURE=1 (project 2 prompt),
     LOGPROBS=1 (record grade-token probability), SAMPLES (k samples at T=1 for agreement, project 2). Out: analysis/rl_grader/preds/<TAG>__<set>.jsonl"""
import json, os, shutil, sys, tempfile, time
sys.path.insert(0, os.path.dirname(__file__)); from common import *
BASE = os.environ.get("BASE", "google/medgemma-4b-it"); AD = os.environ.get("ADAPTER", ""); TAG = os.environ["TAG"]; N = int(os.environ.get("N", 0))
SETS = os.environ.get("SETS", "val,test,test485").split(","); UNS = os.environ.get("UNSURE") == "1"; LP = os.environ.get("LOGPROBS") == "1"; K = int(os.environ.get("SAMPLES", 0))
D = B + "/analysis/rl_grader/data"; P = os.environ.get("PREDS", B + "/analysis/rl_grader/preds"); os.makedirs(P, exist_ok=True)
if all(os.path.exists(f"{P}/{TAG}__{s}.jsonl") for s in SETS): print("done already"); sys.exit(0)
model_dir = BASE; tmp = None
if AD:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
    from peft import PeftModel
    tmp = tempfile.mkdtemp(dir=B + "/analysis/rl_grader/tmp_merged") if os.path.isdir(B + "/analysis/rl_grader/tmp_merged") else tempfile.mkdtemp(dir=os.environ.get("TMPDIR", "/tmp"))
    m = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16); m = PeftModel.from_pretrained(m, AD).merge_and_unload(); m.save_pretrained(tmp, safe_serialization=True)
    AutoTokenizer.from_pretrained(BASE).save_pretrained(tmp)
    try: AutoProcessor.from_pretrained(BASE).save_pretrained(tmp)
    except Exception: pass
    for f in ("preprocessor_config.json", "processor_config.json", "chat_template.json", "chat_template.jinja"):
        src = os.path.join(BASE if os.path.isdir(BASE) else "", f)
        if os.path.isdir(BASE) and os.path.exists(src) and not os.path.exists(os.path.join(tmp, f)): shutil.copy(src, tmp)
    del m; torch.cuda.empty_cache(); model_dir = tmp
from vllm import LLM, SamplingParams
cfgtxt = open(os.path.join(model_dir, "config.json")).read().lower() if os.path.isdir(model_dir) else model_dir.lower()
kw = {"limit_mm_per_prompt": {"image": 0}} if "gemma3" in cfgtxt and "vision" in cfgtxt else {}
llm = LLM(model=model_dir, max_model_len=3584, gpu_memory_utilization=float(os.environ.get("VLLM_MEM", 0.85)), dtype="bfloat16", seed=0, **kw)
greedy = SamplingParams(temperature=0.0, max_tokens=128, logprobs=5 if LP else None)
for s in SETS:
    out = f"{P}/{TAG}__{s}.jsonl"
    if os.path.exists(out): continue
    rows = [json.loads(l) for l in open(f"{D}/{s}.jsonl")]
    if N: rows = rows[:N]
    msgs = [make_messages(r["report"], r["section"], allow_unsure=UNS) for r in rows]
    t0 = time.time(); res = llm.chat(msgs, greedy, use_tqdm=False); dt = time.time() - t0
    samp = llm.chat(msgs, SamplingParams(temperature=1.0, max_tokens=128, n=K, seed=0), use_tqdm=False) if K else None
    with open(out + ".part", "w") as f:
        for i, (r, o) in enumerate(zip(rows, res)):
            txt = o.outputs[0].text; g = parse_grade(txt, allow_unsure=UNS); d = {k: r[k] for k in r if k != "report"}; d.update({"raw": txt[:300], "pred": g})
            if LP and o.outputs[0].logprobs:
                # probability of the first token of the grade value: find the token right after '"grade": "'
                toks = o.outputs[0].logprobs; text_so_far = ""; pgr = None
                for step in toks:
                    best = max(step.values(), key=lambda v: v.logprob); 
                    if text_so_far.rstrip().endswith('"grade": "') or text_so_far.rstrip().endswith('"grade":"'):
                        import math; pgr = math.exp(best.logprob); break
                    text_so_far += best.decoded_token or ""
                d["p_grade"] = pgr
            if samp: d["samples"] = [parse_grade(x.text, allow_unsure=UNS) for x in samp[i].outputs]
            f.write(json.dumps(d) + "\n")
    os.replace(out + ".part", out)
    nrep = len({r["report_id"] for r in rows}); print(f"{TAG} {s}: {len(rows)} sections, {nrep} reports, {dt:.0f}s ({dt / max(nrep, 1):.2f} s/report), parse rate {sum(1 for l in open(out) if json.loads(l)['pred']) / len(rows):.3f}", flush=True)
    json.dump({"tag": TAG, "set": s, "n": len(rows), "reports": nrep, "seconds": dt}, open(f"{P}/{TAG}__{s}.timing.json", "w"))
if tmp: shutil.rmtree(tmp, ignore_errors=True)
print("EVAL DONE", flush=True)

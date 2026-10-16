"""Diagnose zero-shot MedGemma-4B: HF transformers generate vs vLLM on the same 12 validation prompts; also a grade-only answer format."""
import json, os, sys, torch
sys.path.insert(0, os.path.dirname(__file__)); from common import *
HF = os.environ["HF_HOME"]; BASE = os.popen(f"ls -d {HF}/hub/models--google--medgemma-4b-it/snapshots/*/").read().split()[0]
rows = [json.loads(l) for l in open(B + "/analysis/rl_grader/data/val.jsonl")][:12]
from transformers import AutoProcessor, AutoModelForImageTextToText, AutoTokenizer
proc = AutoProcessor.from_pretrained(BASE); m = AutoModelForImageTextToText.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map="cuda").eval()
def hf(msgs):
    mm = [{"role": x["role"], "content": [{"type": "text", "text": x["content"]}]} for x in msgs]
    inp = proc.apply_chat_template(mm, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to("cuda")
    with torch.inference_mode(): out = m.generate(**inp, max_new_tokens=200, do_sample=False)
    return proc.decode(out[0][inp["input_ids"].shape[-1]:], skip_special_tokens=True)
res = {"hf": [hf(make_messages(r["report"], r["section"])) for r in rows]}
def report(k, v):
    ok = sum(1 for t in v if parse_grade(t)); acc = sum(1 for t, r in zip(v, rows) if parse_grade(t) == r["label"])
    print(f"{k:16s} parse {ok}/12 correct {acc}/12 | first: {v[0][:160]!r}", flush=True)
report("hf", res["hf"]); del m; import gc; gc.collect(); torch.cuda.empty_cache()

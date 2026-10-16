"""vLLM half of the zero-shot diagnostic (own process). Variants: as-is chat, system folded into user, and text-only architecture override."""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__)); from common import *
def main():
    HF = os.environ["HF_HOME"]; BASE = os.popen(f"ls -d {HF}/hub/models--google--medgemma-4b-it/snapshots/*/").read().split()[0]
    rows = [json.loads(l) for l in open(B + "/analysis/rl_grader/data/val.jsonl")][:12]
    from vllm import LLM, SamplingParams
    llm = LLM(model=BASE, max_model_len=3584, gpu_memory_utilization=0.6, dtype="bfloat16", limit_mm_per_prompt={"image": 0}, enforce_eager=os.environ.get("EAGER") == "1")
    print("backend", os.environ.get("VLLM_ATTENTION_BACKEND"), "eager", os.environ.get("EAGER"), flush=True)
    sp = SamplingParams(temperature=0, max_tokens=200)
    fold = lambda ms: [{"role": "user", "content": ms[0]["content"] + "\n\n" + ms[1]["content"]}]
    tok = llm.get_tokenizer()
    res = {"vllm_chat": [o.outputs[0].text for o in llm.chat([make_messages(r["report"], r["section"]) for r in rows], sp, use_tqdm=False)],
           "vllm_userfold": [o.outputs[0].text for o in llm.chat([fold(make_messages(r["report"], r["section"])) for r in rows], sp, use_tqdm=False)],
           "vllm_rawtemplate": [o.outputs[0].text for o in llm.generate([tok.apply_chat_template(make_messages(r["report"], r["section"]), tokenize=False, add_generation_prompt=True) for r in rows], sp, use_tqdm=False)]}
    for k, v in res.items():
        ok = sum(1 for t in v if parse_grade(t)); acc = sum(1 for t, r in zip(v, rows) if parse_grade(t) == r["label"])
        print(f"{k:16s} parse {ok}/12 correct {acc}/12 | first: {v[0][:160]!r}", flush=True)
if __name__ == "__main__": main()

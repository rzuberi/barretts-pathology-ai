"""calib strand: full six-grade distribution for a grader model (vLLM, one GPU).
Pass 1: greedy generation (as evaluate.py; logprobs=20 kept at the grade position -> first-token probs of each grade).
Pass 2: for each section, score each GRADE string: log P(GRADE + '"' | chat prompt + generated text up to '"grade": "').
Env: BASE, ADAPTER (LoRA dir or empty), TAG, SETS (comma), N (cap per set, smoke), CONF=1 (D2 prompt), OUT (preds dir).
Out: OUT/<TAG>__<set>.jsonl with pred, p_grade (old first-token prob), lp6 (six log-scores), first_lp (first-token logprobs
from top-20, None if absent), report_chars, scored_fallback. Report text is never written out."""
import json, math, os, re, sys, time
RL = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis/scripts/rl"; sys.path.insert(0, RL); from common import *
BASE = os.environ["BASE"]; AD = os.environ.get("ADAPTER", ""); TAG = os.environ["TAG"]; N = int(os.environ.get("N", 0))
SETS = os.environ.get("SETS", "val,test,test485,pop,all_lowconf").split(","); CONF = os.environ.get("CONF") == "1"
D = B + "/analysis/rl_grader/data"; P = os.environ.get("OUT", B + "/analysis/rl_explore/calib/preds"); os.makedirs(P, exist_ok=True)
_GR = re.compile(r'"grade"\s*:\s*"')

def main():
    if all(os.path.exists(f"{P}/{TAG}__{s}.jsonl") for s in SETS): print("done already"); return
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    LORA = LoRARequest("adapter", 1, AD) if AD else None
    cfgtxt = open(os.path.join(BASE, "config.json")).read().lower()
    kw = {"limit_mm_per_prompt": {"image": 0}} if "gemma3" in cfgtxt and "vision" in cfgtxt else {}
    llm = LLM(model=BASE, max_model_len=3584, gpu_memory_utilization=float(os.environ.get("VLLM_MEM", 0.85)), dtype="bfloat16", seed=0,
              enable_lora=bool(AD), max_lora_rank=16, enforce_eager=True, enable_prefix_caching=True, max_logprobs=20, **kw)
    tok = llm.get_tokenizer()
    # first token id of each grade when it follows '"grade": "'
    probe = '{"reasoning": "x.", "grade": "'; pids = tok(probe, add_special_tokens=False)["input_ids"]
    first_id = {}
    for g in GRADES:
        ids = tok(probe + g + '"', add_special_tokens=False)["input_ids"]; assert ids[:len(pids)] == pids, g; first_id[g] = ids[len(pids)]
    assert len(set(first_id.values())) == 6, first_id
    print("first token per grade:", {g: tok.convert_ids_to_tokens([i])[0] for g, i in first_id.items()}, flush=True)
    greedy = SamplingParams(temperature=0.0, max_tokens=384, logprobs=20)
    score_sp = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0)
    for s in SETS:
        out = f"{P}/{TAG}__{s}.jsonl"
        if os.path.exists(out): continue
        rows = [json.loads(l) for l in open(f"{D}/{s}.jsonl")]
        if N: rows = rows[:N]
        msgs = [make_messages(r["report"], r["section"], conf=CONF) for r in rows]
        t0 = time.time(); res = llm.chat(msgs, greedy, use_tqdm=False, lora_request=LORA); t1 = time.time()
        recs = []; prompts = []; meta = []   # meta: (row index, grade, n prefix tokens)
        for i, (r, o) in enumerate(zip(rows, res)):
            txt = o.outputs[0].text; g = parse_grade(txt); d = {k: r[k] for k in r if k != "report"}
            d.update({"raw": txt[:600], "pred": g, "report_chars": len(r["report"]), "p_grade": None, "first_lp": None, "scored_fallback": False})
            # first-token logprobs at the grade position (top-20 of the generated step)
            if o.outputs[0].logprobs:
                tids = list(o.outputs[0].token_ids); text_so_far = ""
                for step, (tid, lp) in enumerate(zip(tids, o.outputs[0].logprobs)):
                    mm = _GR.search(text_so_far)
                    if mm and mm.end() == len(text_so_far):
                        d["p_grade"] = math.exp(lp[tid].logprob) if tid in lp else None
                        d["first_lp"] = {gg: (lp[fi].logprob if fi in lp else None) for gg, fi in first_id.items()}; break
                    text_so_far += (lp[tid].decoded_token if tid in lp else tok.decode([tid])) or ""
            m = _GR.search(txt)
            chat_prompt = tok.apply_chat_template(msgs[i], add_generation_prompt=True, tokenize=False)
            if m: prefix = chat_prompt + txt[:m.end()]
            else: prefix = chat_prompt + '{"reasoning": "", "grade": "'; d["scored_fallback"] = True
            pre_ids = tok(prefix, add_special_tokens=False)["input_ids"]
            for gg in GRADES:
                ids = tok(prefix + gg + '"', add_special_tokens=False)["input_ids"]
                if ids[:len(pre_ids)] != pre_ids: ids = None
                prompts.append(ids); meta.append((i, gg, len(pre_ids)))
            recs.append(d)
        ok = [j for j, p_ in enumerate(prompts) if p_ is not None] if os.environ.get("SCORE_STRINGS") == "1" else []   # pass 2 off by default (smoke 20 Oct: string scores disagreed with generation; first_lp is used instead)
        outs = llm.generate([{"prompt_token_ids": prompts[j]} for j in ok], score_sp, use_tqdm=False, lora_request=LORA) if ok else []
        lp6 = [[None] * 6 for _ in rows]
        for j, o in zip(ok, outs):
            i, gg, npre = meta[j]; plp = o.prompt_logprobs; tot = 0.0
            for pos in range(npre, len(prompts[j])):
                tid = prompts[j][pos]; tot += plp[pos][tid].logprob
            lp6[i][ORD[gg]] = tot
        t2 = time.time()
        with open(out + ".part", "w") as f:
            for d, l6 in zip(recs, lp6): d["lp6"] = l6; f.write(json.dumps(d) + "\n")
        os.replace(out + ".part", out)
        nrep = len({r["report_id"] for r in rows}); nbad = sum(any(x is None for x in l6) for l6 in lp6)
        agree = sum(1 for d, l6 in zip(recs, lp6) if d["pred"] and all(x is not None for x in l6) and GRADES[max(range(6), key=lambda k: l6[k])] == d["pred"])
        print(f"{TAG} {s}: {len(rows)} sections, {nrep} reports; gen {t1 - t0:.0f}s, score {t2 - t1:.0f}s; parse {sum(1 for d in recs if d['pred']) / len(rows):.3f}; "
              f"unscored {nbad}; argmax(lp6)==pred {agree}/{len(rows)}; fallback {sum(d['scored_fallback'] for d in recs)}", flush=True)
        json.dump({"tag": TAG, "set": s, "n": len(rows), "reports": nrep, "gen_s": t1 - t0, "score_s": t2 - t1}, open(f"{P}/{TAG}__{s}.timing.json", "w"))
    print("SCORE6 DONE", flush=True)

if __name__ == "__main__": main()

"""Explore/agent: custom GRPO for the agent (copy of sql_agent_grpo.py with agent2's prompt/reward and a QFILE option).
Env: BASE (merged model), OUT, STEPS, B, G, LR, BETA, N_TRAIN, SEED, QFILE, VLLM_MEM."""
import json, os, random, sys, time
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from agent2 import *
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, PeftModel
def main():
    BASE = os.environ["BASE"]; OUT = os.environ["OUT"]; STEPS = int(os.environ.get("STEPS", 150)); Bq = int(os.environ.get("B", 8)); G = int(os.environ.get("G", 8))
    LR = float(os.environ.get("LR", 1e-5)); BETA = float(os.environ.get("BETA", 0.04)); SEED = int(os.environ.get("SEED", 0)); NTR = int(os.environ.get("N_TRAIN", 0))
    os.makedirs(OUT, exist_ok=True); torch.manual_seed(SEED); rnd = random.Random(SEED)
    qs = [json.loads(l) for l in open(os.environ.get("QFILE", O + "/questions_agent.jsonl"))]; tr = [q for q in qs if q["split"] == "train"]; tr = tr[:NTR] if NTR else tr
    print(len(tr), "training questions", flush=True)
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    llm = LLM(model=BASE, max_model_len=8192, gpu_memory_utilization=float(os.environ.get("VLLM_MEM", 0.4)), seed=SEED, enable_lora=True, max_lora_rank=16)
    sp = SamplingParams(temperature=1.0, max_tokens=320, seed=None)
    tok = AutoTokenizer.from_pretrained(BASE); dev = "cuda"
    model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(dev); model.gradient_checkpointing_enable(); model.config.use_cache = False
    start = 0; ck = sorted([d for d in os.listdir(OUT) if d.startswith("checkpoint-")], key=lambda d: int(d.split("-")[1]))
    if ck:
        model = PeftModel.from_pretrained(model, os.path.join(OUT, ck[-1]), is_trainable=True); start = int(ck[-1].split("-")[1]); print("resumed", ck[-1], flush=True)
    else:
        model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM", target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.enable_input_require_grads()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    def spans(messages):
        ids = tok.apply_chat_template(messages, tokenize=True); mask = [0] * len(ids)
        for k, m in enumerate(messages):
            if m["role"] != "assistant": continue
            a = len(tok.apply_chat_template(messages[:k], tokenize=True, add_generation_prompt=True)); b = len(tok.apply_chat_template(messages[:k + 1], tokenize=True))
            for j in range(a, min(b, len(ids))): mask[j] = 1
        return ids, mask
    def token_logps(ids, pos):
        x = torch.tensor([ids], device=dev); logits = model(input_ids=x, logits_to_keep=pos - 1).logits[0].float()
        return torch.log_softmax(logits, -1).gather(1, x[0, pos].unsqueeze(1)).squeeze(1)
    log = []
    for step in range(start, STEPS):
        t0 = time.time(); adir = os.path.join(OUT, "_current"); model.save_pretrained(adir)
        lora = LoRARequest(f"s{step}", step + 1, adir)
        batch = rnd.sample(tr, min(Bq, len(tr))); items = [q for q in batch for _ in range(G)]
        eps = run_episodes(llm, items, sp, lora)
        loss_sum, n_used = 0.0, 0; opt.zero_grad(set_to_none=True)
        for gi in range(len(batch)):
            grp = eps[gi * G:(gi + 1) * G]; r = torch.tensor([e["reward"] for e in grp]); sd = r.std()
            if sd < 1e-6: continue
            adv = (r - r.mean()) / (sd + 1e-4)
            for e, a in zip(grp, adv):
                ids, mask = spans(e["messages"]); ids = ids[:6144]; mask = mask[:6144]
                pos = torch.tensor([j for j in range(1, len(ids)) if mask[j]], device=dev, dtype=torch.long)
                if len(pos) == 0: continue
                lp = token_logps(ids, pos)
                with torch.no_grad(), model.disable_adapter(): ref = token_logps(ids, pos)
                d = ref - lp; kl = torch.exp(d) - d - 1
                loss = (-a.to(dev) * lp + BETA * kl).mean() / (len(batch) * G)
                loss.backward(); loss_sum += float(loss); n_used += 1
        if n_used: torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0); opt.step()
        an = [e for e in eps if e["item"]["answerable"]]
        rec = {"step": step + 1, "reward": round(sum(e["reward"] for e in eps) / len(eps), 4), "acc": round(sum(e["correct"] for e in eps) / len(eps), 4),
               "false_ref": round(sum(e["kind"] == "refused" for e in an) / max(len(an), 1), 3), "calls": round(sum(e["n_calls"] for e in eps) / len(eps), 2), "episodes_used": n_used, "loss": round(loss_sum, 5), "sec": round(time.time() - t0, 1)}
        log.append(rec); print(rec, flush=True)
        if (step + 1) % 25 == 0 or step + 1 == STEPS: model.save_pretrained(os.path.join(OUT, f"checkpoint-{step + 1}"))
    model.save_pretrained(os.path.join(OUT, "final")); json.dump({"steps": STEPS, "B": Bq, "G": G, "lr": LR, "beta": BETA, "base": BASE, "qfile": os.environ.get("QFILE", ""), "log": log}, open(os.path.join(OUT, "train_summary.json"), "a"), indent=1)
    print("TRAIN DONE", flush=True)
if __name__ == "__main__":
    main()

"""Explore/agent, expert iteration step 3: supervised fine-tuning of the merged SFT model on kept agent trajectories; loss on
assistant tokens only (tool results and prompts masked). LoRA r=16. Env: BASE, DATA, OUT, EPOCHS (2), LR (1e-4), ACC (8), MAXLEN (6144)."""
import json, math, os, random, sys, time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
BASE = os.environ["BASE"]; DATA = os.environ["DATA"]; OUT = os.environ["OUT"]; EP = int(os.environ.get("EPOCHS", 2)); LR = float(os.environ.get("LR", 1e-4)); ACC = int(os.environ.get("ACC", 8)); ML = int(os.environ.get("MAXLEN", 6144))
os.makedirs(OUT, exist_ok=True); torch.manual_seed(0); rows = [json.loads(l) for l in open(DATA)]; dev = "cuda"
tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(dev); model.gradient_checkpointing_enable(); model.config.use_cache = False
model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM", target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])); model.enable_input_require_grads()
def spans(messages):
    ids = tok.apply_chat_template(messages, tokenize=True); mask = [0] * len(ids)
    for k, m in enumerate(messages):
        if m["role"] != "assistant": continue
        a = len(tok.apply_chat_template(messages[:k], tokenize=True, add_generation_prompt=True)); b = len(tok.apply_chat_template(messages[:k + 1], tokenize=True))
        for j in range(a, min(b, len(ids))): mask[j] = 1
    return ids[:ML], mask[:ML]
data = [spans(r["messages"]) for r in rows]; print(len(data), "trajectories; mean tokens", round(sum(len(d[0]) for d in data) / len(data)), "mean assistant tokens", round(sum(sum(d[1]) for d in data) / len(data)), flush=True)
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR); total = math.ceil(len(data) * EP / ACC); sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, int(0.03 * total))) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
rnd = random.Random(0); step = 0; log = []; t0 = time.time()
for ep in range(EP):
    order = list(range(len(data))); rnd.shuffle(order); acc_loss = 0.0
    for i, ix in enumerate(order):
        ids, mask = data[ix]; x = torch.tensor([ids], device=dev); lab = torch.tensor([[t if m else -100 for t, m in zip(ids, mask)]], device=dev)
        loss = model(input_ids=x, labels=lab).loss / ACC; loss.backward(); acc_loss += float(loss)
        if (i + 1) % ACC == 0 or i + 1 == len(order):
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0); opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
            if step % 5 == 0 or i + 1 == len(order): rec = {"epoch": ep + 1, "step": step, "loss": round(acc_loss, 4), "sec": round(time.time() - t0)}; log.append(rec); print(rec, flush=True)
            acc_loss = 0.0
    model.save_pretrained(os.path.join(OUT, f"epoch-{ep + 1}"))
model.save_pretrained(os.path.join(OUT, "final")); json.dump({"n": len(data), "epochs": EP, "lr": LR, "acc": ACC, "log": log}, open(os.path.join(OUT, "train_summary.json"), "w"), indent=1); print("TRAIN DONE", flush=True)

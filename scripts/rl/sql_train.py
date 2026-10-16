"""Project 3 training: SFT on gold SQL or GRPO with execution reward. LoRA r=16 on Qwen2.5-Coder-7B-Instruct. Resume-capable.
Env: MODE=sft|grpo, BASE, OUT, STEPS, N_TRAIN (smoke cap)."""
import json, os, random, sys
import torch
sys.path.insert(0, os.path.dirname(__file__)); from sql_common import *
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
MODE = os.environ["MODE"]; BASE = os.environ["BASE"]; OUT = os.environ["OUT"]; STEPS = int(os.environ.get("STEPS", 300)); NTR = int(os.environ.get("N_TRAIN", 0))
os.makedirs(OUT, exist_ok=True)
qs = [json.loads(l) for l in open(O + "/questions.jsonl")]; tr = [q for q in qs if q["split"] == "train"]; random.Random(0).shuffle(tr)
if NTR: tr = tr[:NTR]
print(MODE, len(tr), "training questions", flush=True)
lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM", target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
tok = AutoTokenizer.from_pretrained(BASE); model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
ck = sorted([d for d in os.listdir(OUT) if d.startswith("checkpoint-")], key=lambda d: int(d.split("-")[1])); last = os.path.join(OUT, ck[-1]) if ck else None
common = dict(output_dir=OUT, bf16=True, logging_steps=5, save_steps=50 if STEPS >= 100 else max(5, STEPS // 2), report_to="none", gradient_checkpointing=True, max_steps=STEPS, seed=0)
if MODE == "sft":
    from trl import SFTTrainer, SFTConfig
    ds = Dataset.from_list([{"messages": messages(q["question"]) + [{"role": "assistant", "content": "CANNOT_ANSWER" if not q["answerable"] else f"```sql\n{q['gold_sql']}\n```"}]} for q in tr])
    cfg = SFTConfig(**common, learning_rate=1e-4, per_device_train_batch_size=4, gradient_accumulation_steps=4, max_length=2048, lr_scheduler_type="cosine", warmup_ratio=0.03)
    t = SFTTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok, peft_config=lora)
else:
    from trl import GRPOTrainer, GRPOConfig
    ds = Dataset.from_list([{"prompt": messages(q["question"]), "item": json.dumps({"answerable": q["answerable"], "answer": q["answer"]})} for q in tr])
    def reward(prompts, completions, item, **kw):
        return [score(c[-1]["content"] if isinstance(c, list) else c, json.loads(i))[0] for c, i in zip(completions, item)]
    cfg = GRPOConfig(**common, learning_rate=1e-5, per_device_train_batch_size=16, gradient_accumulation_steps=8, num_generations=8, max_prompt_length=1536,
                     max_completion_length=384, beta=0.04, temperature=1.0, lr_scheduler_type="constant_with_warmup", warmup_ratio=0.02,
                     use_vllm=True, vllm_mode="colocate", vllm_gpu_memory_utilization=float(os.environ.get("VLLM_MEM", 0.35)))
    t = GRPOTrainer(model=model, args=cfg, train_dataset=ds, reward_funcs=[reward], processing_class=tok, peft_config=lora)
t.train(resume_from_checkpoint=last); t.save_model(os.path.join(OUT, "final"))
json.dump({"mode": MODE, "n_train": len(tr), "steps": STEPS, "log": t.state.log_history}, open(os.path.join(OUT, "train_summary.json"), "w"), indent=1); print("TRAIN DONE", flush=True)

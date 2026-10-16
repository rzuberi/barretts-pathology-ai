"""Task F training: SFT or GRPO (LoRA r=16) on the PerturbQA three-class task. Env: MODE, BASE, OUT, DATA (jsonl), STEPS, N, SEED."""
import json, os, sys
import torch
sys.path.insert(0, os.path.dirname(__file__)); from pert_common import *
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
try:
    import vllm; _o = vllm.LLM.__init__
    def _e(self, *a, **k): k["enforce_eager"] = True; _o(self, *a, **k)
    vllm.LLM.__init__ = _e
except Exception: pass
MODE = os.environ["MODE"]; BASE = os.environ["BASE"]; OUT = os.environ["OUT"]; STEPS = int(os.environ.get("STEPS", 3000)); N = int(os.environ.get("N", 0)); SEED = int(os.environ.get("SEED", 0))
rows = [json.loads(l) for l in open(os.environ["DATA"])]; rows = rows[:N] if N else rows; os.makedirs(OUT, exist_ok=True)
print(MODE, len(rows), "items", flush=True)
lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM", target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
tok = AutoTokenizer.from_pretrained(BASE); model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
ck = sorted([d for d in os.listdir(OUT) if d.startswith("checkpoint-")], key=lambda d: int(d.split("-")[1])); last = os.path.join(OUT, ck[-1]) if ck else None
common = dict(output_dir=OUT, bf16=True, logging_steps=10, save_steps=max(50, STEPS // 6), report_to="none", gradient_checkpointing=True, max_steps=STEPS, seed=SEED)
if MODE == "sft":
    from trl import SFTTrainer, SFTConfig
    ds = Dataset.from_list([{"messages": messages(r["cell"], r["pert"], r["gene"]) + [{"role": "assistant", "content": r["label"]}]} for r in rows])
    cfg = SFTConfig(**common, learning_rate=1e-4, per_device_train_batch_size=16, gradient_accumulation_steps=1, max_length=256, lr_scheduler_type="cosine", warmup_ratio=0.03)
    t = SFTTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok, peft_config=lora)
else:
    from trl import GRPOTrainer, GRPOConfig
    ds = Dataset.from_list([{"prompt": messages(r["cell"], r["pert"], r["gene"]), "label": r["label"]} for r in rows])
    def reward(prompts, completions, label, **kw):
        out = []
        for c, y in zip(completions, label):
            g = parse(c[-1]["content"] if isinstance(c, list) else c); out.append(-0.5 if g is None else 1.0 if g == y else 0.0)
        return out
    cfg = GRPOConfig(**common, learning_rate=1e-5, per_device_train_batch_size=16, gradient_accumulation_steps=8, num_generations=8, max_prompt_length=256,
                     max_completion_length=16, beta=0.04, temperature=1.0, lr_scheduler_type="constant_with_warmup", warmup_ratio=0.02,
                     use_vllm=True, vllm_mode="colocate", vllm_gpu_memory_utilization=0.35)
    t = GRPOTrainer(model=model, args=cfg, train_dataset=ds, reward_funcs=[reward], processing_class=tok, peft_config=lora)
t.train(resume_from_checkpoint=last); t.save_model(os.path.join(OUT, "final"))
json.dump({"mode": MODE, "n": len(rows), "steps": STEPS, "log": t.state.log_history}, open(os.path.join(OUT, "train_summary.json"), "w"), indent=1); print("TRAIN DONE", flush=True)

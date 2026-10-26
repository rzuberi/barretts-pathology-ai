"""Completion-masked SFT control for project 1 / follow-up C: scripts/rl/train.py's SFT branch with the dataset in TRL's
prompt-completion conversational format, so the loss covers only the assistant JSON (completion_only_loss=True) instead of the
whole prompt+report (the original used 'messages' with assistant_only_loss=False under TRL 0.24). Everything else identical:
MedGemma-4B-it, LoRA r16, lr 1e-4, bs 4x4, 900 steps, cosine, max_length 3072, same seed semantics. Env: SEED, OUT, STEPS, N_TRAIN, BASE."""
import json, os, random, sys
import torch
sys.path.insert(0, "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis/scripts/rl"); from common import *
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
BASE = os.environ.get("BASE", "google/medgemma-4b-it"); OUT = os.environ["OUT"]; os.makedirs(OUT, exist_ok=True)
STEPS = int(os.environ.get("STEPS", 900)); NTR = int(os.environ.get("N_TRAIN", 0)); SEED = int(os.environ.get("SEED", 0))
rows = [json.loads(l) for l in open(B + "/analysis/rl_grader/data/train.jsonl")]
rnd = random.Random(SEED); rnd.shuffle(rows)
if NTR: rows = rows[:NTR]
print(f"sft-masked: {len(rows)} training items; base {BASE}; seed {SEED}", flush=True)
lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                  target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)|^(?!.*vision).*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)")
tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, attn_implementation="eager")
last = None
ck = sorted([d for d in os.listdir(OUT) if d.startswith("checkpoint-")], key=lambda d: int(d.split("-")[1])); last = os.path.join(OUT, ck[-1]) if ck else None
if last: print("resuming from", last, flush=True)
ds = Dataset.from_list([{"prompt": make_messages(r["report"], r["section"]), "completion": [{"role": "assistant", "content": json.dumps({"grade": r["label"]})}]} for r in rows])
cfg = SFTConfig(output_dir=OUT, bf16=True, logging_steps=5, save_steps=50, save_total_limit=None, report_to="none", gradient_checkpointing=True, max_steps=STEPS, seed=SEED,
                learning_rate=1e-4, per_device_train_batch_size=4, gradient_accumulation_steps=4, max_length=3072,
                completion_only_loss=True, assistant_only_loss=False, lr_scheduler_type="cosine", warmup_ratio=0.03)
tr = SFTTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok, peft_config=lora)
# sanity: the collator must mask the prompt -> count supervised tokens in the first example
ex = tr.train_dataset[0]; nlab = sum(1 for m in ex.get("completion_mask", []) if m)
print("first example: total tokens", len(ex["input_ids"]), "completion tokens", nlab, flush=True)
assert 0 < nlab < 40, f"completion mask looks wrong: {nlab}"
tr.train(resume_from_checkpoint=last)
tr.save_model(os.path.join(OUT, "final"))
json.dump({"mode": "sft_masked", "base": BASE, "n_train": len(rows), "steps": STEPS, "seed": SEED, "completion_tokens_example0": nlab, "log": tr.state.log_history}, open(os.path.join(OUT, "train_summary.json"), "w"), indent=1)
print("TRAIN DONE", flush=True)

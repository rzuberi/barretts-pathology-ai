"""Project 1/2 training: SFT or GRPO with LoRA on the per-section grading task. Resume-capable (h200 preemption).
Env: MODE=sft|grpo, BASE (HF id or local dir), OUT (run dir), STEPS, N_TRAIN (cap, smoke), ALPHA (project 2 abstention reward; unset = project 1),
     DATA (train jsonl, default rl_grader/data/train.jsonl), EXTRA (optional extra train jsonl, e.g. all_lowconf), CAP_PER_CLASS (grpo, default 1200)."""
import json, os, random, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__)); from common import *
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
try:   # CUDA-graph capture corrupts Gemma 3 generations in vLLM 0.11 on this cluster's driver: force eager mode for TRL's colocated vLLM too
    import vllm
    _orig_init = vllm.LLM.__init__
    def _eager_init(self, *a, **k): k["enforce_eager"] = True; _orig_init(self, *a, **k)
    vllm.LLM.__init__ = _eager_init
except Exception as _e: print("vllm patch skipped:", _e)
MODE = os.environ["MODE"]; BASE = os.environ.get("BASE", "google/medgemma-4b-it"); OUT = os.environ["OUT"]; os.makedirs(OUT, exist_ok=True)
STEPS = int(os.environ.get("STEPS", 300)); NTR = int(os.environ.get("N_TRAIN", 0)); ALPHA = os.environ.get("ALPHA"); ALPHA = float(ALPHA) if ALPHA else None
D = B + "/analysis/rl_grader/data"; DATA = os.environ.get("DATA", D + "/train.jsonl"); EXTRA = os.environ.get("EXTRA", ""); CAP = int(os.environ.get("CAP_PER_CLASS", 1200))
rows = [json.loads(l) for l in open(DATA)] + ([json.loads(l) for l in open(EXTRA)] if EXTRA else [])
rnd = random.Random(0)
if MODE == "grpo":   # class-capped sample (rare grades carry the signal)
    by = {}
    for r in rows: by.setdefault(r["label"], []).append(r)
    rows = [x for g in by.values() for x in rnd.sample(g, min(CAP, len(g)))]
rnd.shuffle(rows)
if NTR: rows = rows[:NTR]
print(f"{MODE}: {len(rows)} training items from {DATA}{' + ' + EXTRA if EXTRA else ''}; base {BASE}; alpha {ALPHA}", flush=True)
lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                  target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)|^(?!.*vision).*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)")
tok = AutoTokenizer.from_pretrained(BASE)
def is_gemma(p):
    try: return "gemma" in (open(os.path.join(p, "config.json")).read().lower() if os.path.isdir(p) else p.lower())
    except Exception: return "gemma" in p.lower()
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, attn_implementation="eager" if is_gemma(BASE) else "sdpa")
last = None
if os.path.isdir(OUT):
    ck = sorted([d for d in os.listdir(OUT) if d.startswith("checkpoint-")], key=lambda d: int(d.split("-")[1]))
    last = os.path.join(OUT, ck[-1]) if ck else None
if last: print("resuming from", last, flush=True)
common_args = dict(output_dir=OUT, bf16=True, logging_steps=5, save_steps=50 if STEPS >= 100 else max(5, STEPS // 2), save_total_limit=None, report_to="none",
                   gradient_checkpointing=True, max_steps=STEPS, seed=0)
if MODE == "sft":
    from trl import SFTTrainer, SFTConfig
    ds = Dataset.from_list([{"messages": make_messages(r["report"], r["section"]) + [{"role": "assistant", "content": json.dumps({"grade": r["label"]})}]} for r in rows])
    cfg = SFTConfig(**common_args, learning_rate=1e-4, per_device_train_batch_size=4, gradient_accumulation_steps=4, max_length=3072,
                    assistant_only_loss=False, completion_only_loss=None, lr_scheduler_type="cosine", warmup_ratio=0.03)
    tr = SFTTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok, peft_config=lora)
else:
    from trl import GRPOTrainer, GRPOConfig
    ds = Dataset.from_list([{"prompt": make_messages(r["report"], r["section"], allow_unsure=ALPHA is not None), "label": r["label"]} for r in rows])
    def reward(prompts, completions, label, **kw):
        out = []
        for c, y in zip(completions, label):
            g = parse_grade(completion_text(c), allow_unsure=ALPHA is not None)
            if ALPHA is None:
                out.append(-0.5 if g is None else 1.0 if g == y else 0.3 if ((g in POS) == (y in POS)) else 0.0)
            else:
                out.append(-1.0 if g is None else ALPHA if g == "UNSURE" else 1.0 if g == y else -1.0)
        return out
    use_vllm = os.environ.get("USE_VLLM", "1") == "1"
    cfg = GRPOConfig(**common_args, learning_rate=2e-5, per_device_train_batch_size=16, gradient_accumulation_steps=8, num_generations=8,
                     max_prompt_length=3072, max_completion_length=256, beta=0.04, temperature=1.0, lr_scheduler_type="constant_with_warmup", warmup_ratio=0.02,
                     use_vllm=use_vllm, vllm_mode="colocate", vllm_gpu_memory_utilization=float(os.environ.get("VLLM_MEM", 0.35)),
                     log_completions=False)
    tr = GRPOTrainer(model=model, args=cfg, train_dataset=ds, reward_funcs=[reward], processing_class=tok, peft_config=lora)
tr.train(resume_from_checkpoint=last)
tr.save_model(os.path.join(OUT, "final"))
json.dump({"mode": MODE, "base": BASE, "n_train": len(rows), "steps": STEPS, "alpha": ALPHA, "log": tr.state.log_history}, open(os.path.join(OUT, "train_summary.json"), "w"), indent=1)
print("TRAIN DONE", flush=True)

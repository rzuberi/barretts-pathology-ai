"""Explore/grader: GRPO with LoRA on the per-section grading task, project-1 recipe plus a SOFT jury-vote reward. Copy of scripts/rl/train.py
(GRPO branch only; the original is untouched). Resume-capable (h200 preemption).
Env: BASE (merged SFT model dir), OUT (run dir), STEPS, N_TRAIN (cap, smoke), SEED, REWARD=p1|soft, VOTES (votes.json for soft),
     DATA (train jsonl), EXTRA (optional extra train jsonl, e.g. all_lowconf), KEEP_EXTRA=1 (extra uncapped), CAP_PER_CLASS (default 1200).
soft reward: r = max(share(pred), 0.3 if pred on the same side of the dysplasia line as the consensus label), -0.5 invalid JSON;
share(pred) = fraction of jurors who listed the predicted grade. Unanimous sections -> identical to the project-1 reward."""
import json, os, random, sys
import numpy as np, torch
sys.path.insert(0, "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis/scripts/rl"); from common import *
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
try:   # CUDA-graph capture corrupts Gemma 3 generations in vLLM 0.11 on this cluster's driver: force eager mode for TRL's colocated vLLM too
    import vllm
    _orig_init = vllm.LLM.__init__
    def _eager_init(self, *a, **k): k["enforce_eager"] = True; _orig_init(self, *a, **k)
    vllm.LLM.__init__ = _eager_init
except Exception as _e: print("vllm patch skipped:", _e)
BASE = os.environ["BASE"]; OUT = os.environ["OUT"]; os.makedirs(OUT, exist_ok=True)
STEPS = int(os.environ.get("STEPS", 300)); NTR = int(os.environ.get("N_TRAIN", 0)); SEED = int(os.environ.get("SEED", 0))
REWARD = os.environ.get("REWARD", "soft"); KEEP_EXTRA = os.environ.get("KEEP_EXTRA") == "1"
D = B + "/analysis/rl_grader/data"; DATA = os.environ.get("DATA", D + "/train.jsonl"); EXTRA = os.environ.get("EXTRA", ""); CAP = int(os.environ.get("CAP_PER_CLASS", 1200))
VOTES = json.load(open(os.environ["VOTES"])) if REWARD == "soft" else {}
rows = [json.loads(l) for l in open(DATA)]; extra = [json.loads(l) for l in open(EXTRA)] if EXTRA else []
if not KEEP_EXTRA: rows += extra; extra = []
rnd = random.Random(SEED)   # seed 0 == project 1 sampling
by = {}
for r in rows: by.setdefault(r["label"], []).append(r)
rows = [x for g in by.values() for x in rnd.sample(g, min(CAP, len(g)))]   # class-capped sample (rare grades carry the signal)
rows += extra   # KEEP_EXTRA: low-agreement sections added uncapped
if os.environ.get("MINED"):   # a4: keep only prompts whose 8 SFT samples were mixed (0 < acc < 1); unanimous groups give GRPO no gradient
    MN = json.load(open(os.environ["MINED"])); n0 = len(rows); rows = [r for r in rows if r["id"] in MN and 0 < MN[r["id"]]["acc"] < 1]
    print(f"mined filter: {len(rows)} of {n0} prompts kept (mixed 8-sample outcomes)", flush=True)
rnd.shuffle(rows)
if NTR: rows = rows[:NTR]
if REWARD == "soft":
    miss = sum(r["id"] not in VOTES for r in rows); print(f"soft reward: {miss} of {len(rows)} items without vote shares (fall back to hard label)", flush=True)
print(f"grpo: {len(rows)} training items from {DATA}{' + ' + EXTRA if EXTRA else ''}; base {BASE}; reward {REWARD}; seed {SEED}; uncapped extra {len(extra)}", flush=True)
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
                   gradient_checkpointing=True, max_steps=STEPS, seed=SEED)
from trl import GRPOTrainer, GRPOConfig
ds = Dataset.from_list([{"prompt": make_messages(r["report"], r["section"]), "label": r["label"], "votes_json": json.dumps(VOTES.get(r["id"], {r["label"]: 1.0}))} for r in rows])
def reward(prompts, completions, label, votes_json, **kw):
    out = []
    for c, y, vj in zip(completions, label, votes_json):
        g = parse_grade(completion_text(c))
        if g is None: out.append(-0.5); continue
        side = 0.3 if ((g in POS) == (y in POS)) else 0.0
        if REWARD == "p1": out.append(1.0 if g == y else side)
        else: out.append(max(float(json.loads(vj).get(g, 0.0)), side))
    return out
cfg = GRPOConfig(**common_args, learning_rate=2e-5, per_device_train_batch_size=16, gradient_accumulation_steps=8, num_generations=8,
                 max_prompt_length=3072, max_completion_length=256, beta=0.04, temperature=1.0, lr_scheduler_type="constant_with_warmup", warmup_ratio=0.02,
                 use_vllm=True, vllm_mode="colocate", vllm_gpu_memory_utilization=float(os.environ.get("VLLM_MEM", 0.35)), log_completions=False)
tr = GRPOTrainer(model=model, args=cfg, train_dataset=ds, reward_funcs=[reward], processing_class=tok, peft_config=lora)
tr.train(resume_from_checkpoint=last)
tr.save_model(os.path.join(OUT, "final"))
json.dump({"mode": "grpo", "base": BASE, "n_train": len(rows), "steps": STEPS, "reward": REWARD, "seed": SEED, "extra": EXTRA, "log": tr.state.log_history}, open(os.path.join(OUT, "train_summary.json"), "w"), indent=1)
print("TRAIN DONE", flush=True)

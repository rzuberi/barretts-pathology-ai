"""Merge a LoRA adapter into its base and save a full model (used once: SFT -> base for GRPO-from-SFT). argv: base adapter out"""
import os, shutil, sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from peft import PeftModel
base, ad, out = sys.argv[1:4]
if os.path.exists(os.path.join(out, "config.json")): print("exists"); sys.exit(0)
m = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16); m = PeftModel.from_pretrained(m, ad).merge_and_unload()
m.save_pretrained(out, safe_serialization=True); AutoTokenizer.from_pretrained(base).save_pretrained(out)
try: AutoProcessor.from_pretrained(base).save_pretrained(out)
except Exception as e: print("no processor", e)
for f in os.listdir(base):
    if f.endswith((".json", ".jinja", ".model")) and not os.path.exists(os.path.join(out, f)): shutil.copy(os.path.join(base, f), out)
print("merged ->", out)

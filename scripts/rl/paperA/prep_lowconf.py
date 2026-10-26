"""Tokenise all_lowconf.jsonl for the encoder exactly as explore/clf/prep.py did (same text_for, same cap). Out: O/tok/mbb__all_lowconf.pt"""
import os, sys, torch
os.environ["HF_HOME"] = "/mnt/scratche/slow/fmlab/zuberi01/hf_cache"; os.environ["HF_HUB_OFFLINE"] = "1"
from pa_common import *
sys.path.insert(0, T + "/scripts/rl/explore/clf"); from clf_common import text_for
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL); mkdir(O + "/tok"); out = O + "/tok/mbb__all_lowconf.pt"
if os.path.exists(out): print("exists"); sys.exit(0)
rs = rows("all_lowconf"); enc = tok([text_for(r) for r in rs], truncation=True, max_length=MAXLEN, add_special_tokens=True)["input_ids"]
torch.save({"ids": [r["id"] for r in rs], "input_ids": enc, "label": [ORD.get(r["label"], -1) for r in rs]}, out)
print("all_lowconf", len(rs), "max tokens", max(map(len, enc)), "at cap", sum(len(e) >= MAXLEN for e in enc))

"""Pre-tokenise all splits once for one model key (mbb / mbl / cbert). Out: O/tok/<key>__<set>.pt  (list of id lists + labels)."""
import os, sys, time, torch
os.environ["HF_HOME"] = "/mnt/scratche/slow/fmlab/zuberi01/hf_cache"; os.environ["HF_HUB_OFFLINE"] = "1"
from clf_common import *
from transformers import AutoTokenizer
key = sys.argv[1]; name, maxlen = MODELS[key]; tok = AutoTokenizer.from_pretrained(name); os.makedirs(O + "/tok", exist_ok=True)
ntok = (lambda t: len(tok(t, add_special_tokens=False)["input_ids"])) if maxlen <= 512 else None
for s in SETS:
    out = f"{O}/tok/{key}__{s}.pt"
    if os.path.exists(out): continue
    rs = rows(s); t0 = time.time(); texts = [text_for(r, maxlen - 2 if ntok else None, ntok) for r in rs]
    enc = tok(texts, truncation=True, max_length=maxlen, add_special_tokens=True)["input_ids"]
    lab = [ORD.get(r.get("human") if s == "test485" else r["label"], -1) for r in rs]
    trunc = sum(1 for e in enc if len(e) >= maxlen); L = sorted(len(e) for e in enc)
    torch.save({"ids": [r["id"] for r in rs], "input_ids": enc, "label": lab}, out + f".{os.getpid()}.tmp"); os.replace(out + f".{os.getpid()}.tmp", out)
    print(f"{key} {s}: n={len(rs)} tokens p50={L[len(L)//2]} p99={L[int(.99*len(L))]} max={L[-1]} at_cap={trunc} ({time.time()-t0:.0f}s)", flush=True)
print("PREP DONE")

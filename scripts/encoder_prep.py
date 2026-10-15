"""Encoder-sweep prep: check HF auth, download Virchow2 + GigaPath to shared cache."""
import os
os.environ.setdefault("HF_HOME", "/mnt/scratche/fast/fmlab/zuberi01/hf_cache")
from huggingface_hub import whoami, snapshot_download

try:
    print("token user:", whoami()["name"])
except Exception as e:
    print("NO TOKEN:", type(e).__name__, str(e)[:200])
    print("-> gated encoders need `huggingface-cli login` with Rehan's account once")
    raise SystemExit(0)

for m in ["paige-ai/Virchow2", "prov-gigapath/prov-gigapath"]:
    try:
        p = snapshot_download(m)
        print("ok", m, "->", p)
    except Exception as e:
        print("FAIL", m, type(e).__name__, str(e)[:200])

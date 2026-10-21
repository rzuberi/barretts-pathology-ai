import os; os.environ["HF_HOME"]="/mnt/scratche/slow/fmlab/zuberi01/hf_cache"
from huggingface_hub import snapshot_download
for m in ["thomas-sounack/BioClinical-ModernBERT-base"]:
    p=snapshot_download(m, allow_patterns=["*.json","*.txt","model.safetensors"] if "ModernBERT" in m else ["*.json","*.txt","pytorch_model.bin"]); print("DL OK", m, p, flush=True)

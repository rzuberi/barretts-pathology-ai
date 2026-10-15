import os
os.environ.setdefault("HF_HOME", "/mnt/scratche/fast/fmlab/zuberi01/hf_cache")
from huggingface_hub import snapshot_download as s
print(s("bioptimus/H-optimus-0"))
print(s("owkin/phikon-v2"))

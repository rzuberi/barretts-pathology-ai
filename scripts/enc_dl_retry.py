import os, time, sys
os.environ.setdefault("HF_HOME", "/mnt/scratche/fast/fmlab/zuberi01/hf_cache")
from huggingface_hub import snapshot_download as s
name = sys.argv[1]
for attempt in range(30):  # ~5 days at 4h
    try:
        print(s(name)); sys.exit(0)
    except Exception as e:
        print(f"attempt {attempt}: {type(e).__name__}: {str(e)[:120]}", flush=True)
        time.sleep(4 * 3600)
sys.exit(1)

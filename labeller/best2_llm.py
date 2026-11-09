"""Local-LLM extraction over a CSV shard (text never leaves the cluster). Idempotent: skips if OUTFILE exists.
Env: MODEL, INPUT (csv with id,text), PROMPT (structure|crf), SHARD, NSHARDS, OUTFILE."""
import json, os, re, subprocess, sys, time, urllib.request, concurrent.futures as cf
import pandas as pd
MODEL = os.environ["MODEL"]; INPUT = os.environ["INPUT"]; PK = os.environ["PROMPT"]; OUTF = os.environ["OUTFILE"]
SHARD = int(os.environ.get("SHARD", "0")); NS = int(os.environ.get("NSHARDS", "1"))
if os.path.exists(OUTF): print("exists, skip", OUTF); sys.exit(0)
HERE = os.path.dirname(os.path.abspath(__file__)); PROMPT = open(f"{HERE}/best2_prompt_{PK}.txt").read()
os.environ.setdefault("OLLAMA_MODELS", "/mnt/scratche/slow/fmlab/zuberi01/ollama-models"); os.environ.setdefault("OLLAMA_NUM_PARALLEL", "6")
PORT = 20000 + int(os.environ.get("SLURM_JOB_ID", "0")) % 20000; os.environ["OLLAMA_HOST"] = f"127.0.0.1:{PORT}"; BASE = f"http://127.0.0.1:{PORT}"
os.makedirs(os.path.dirname(OUTF), exist_ok=True)
srv = subprocess.Popen([os.path.expanduser("~/.local/bin/ollama"), "serve"], stdout=open(OUTF + f".ollama_{os.environ.get('SLURM_JOB_ID','x')}.log", "w"), stderr=subprocess.STDOUT)
for _ in range(60):
    try: urllib.request.urlopen(BASE + "/api/tags", timeout=3); break
    except Exception: time.sleep(2)
def ask(text):
    body = json.dumps({"model": MODEL, "prompt": PROMPT + text[:8000] + "\n/no_think", "stream": False, "format": "json",
                       "options": {"temperature": 0, "num_ctx": 8192}}).encode(); err = ""
    for _ in range(3):
        try:
            r = json.loads(urllib.request.urlopen(urllib.request.Request(BASE + "/api/generate", body, {"Content-Type": "application/json"}), timeout=600).read())["response"]
            r = re.sub(r"<think>.*?</think>", "", r, flags=re.S).strip(); return json.loads(r[r.find("{"): r.rfind("}") + 1]), ""
        except Exception as e: err = repr(e)[:200]; time.sleep(3)
    return None, err
d = pd.read_csv(INPUT, dtype=str).fillna(""); d = d[d.index % NS == SHARD]
print(f"model={MODEL} prompt={PK} shard={SHARD}/{NS} n={len(d)}", flush=True)
with cf.ThreadPoolExecutor(6) as ex: out = list(ex.map(ask, d.text))
res = pd.DataFrame({"id": d.id.values, "json": [json.dumps(j) if j else "" for j, _ in out], "err": [e for _, e in out]})
ok = int((res.json != "").sum()); print("parsed ok:", ok, "of", len(res), flush=True); srv.terminate()
if ok < 0.9 * len(res): raise SystemExit(f"only {ok}/{len(res)} parsed - not writing")
res.to_csv(OUTF + ".tmp", index=False); os.replace(OUTF + ".tmp", OUTF)

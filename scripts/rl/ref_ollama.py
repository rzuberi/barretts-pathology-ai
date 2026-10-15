"""Reference: an ollama model (e.g. medgemma:27b) zero-shot on the same single-section prompt. Env: MODEL, SETS, TAG, N, CONC.
Writes preds in the same format as evaluate.py (rl_grader/preds/<TAG>__<set>.jsonl + timing)."""
import json, os, subprocess, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(__file__)); from common import *
MODEL = os.environ["MODEL"]; TAG = os.environ["TAG"]; SETS = os.environ.get("SETS", "test485").split(","); N = int(os.environ.get("N", 0)); CONC = int(os.environ.get("CONC", 4))
D = B + "/analysis/rl_grader/data"; P = B + "/analysis/rl_grader/preds"; os.makedirs(P, exist_ok=True)
os.environ.setdefault("OLLAMA_MODELS", B + "/ollama-models"); PORT = 23000 + int(os.environ.get("SLURM_JOB_ID", "0")) % 20000
os.environ["OLLAMA_HOST"] = f"127.0.0.1:{PORT}"; BASE = f"http://127.0.0.1:{PORT}"
srv = subprocess.Popen([os.path.expanduser("~/.local/bin/ollama"), "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
for _ in range(60):
    try: urllib.request.urlopen(BASE + "/api/tags", timeout=3); break
    except Exception: time.sleep(2)
def one(r):
    m = make_messages(r["report"], r["section"])
    body = {"model": MODEL, "messages": m, "stream": False, "format": "json", "options": {"temperature": 0.0, "num_predict": 160, "num_ctx": 6144}}
    req = urllib.request.Request(BASE + "/api/chat", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try: txt = json.loads(urllib.request.urlopen(req, timeout=600).read())["message"]["content"]
    except Exception as e: txt = f"ERR {type(e).__name__}"
    return txt
for s in SETS:
    out = f"{P}/{TAG}__{s}.jsonl"
    if os.path.exists(out): continue
    rows = [json.loads(l) for l in open(f"{D}/{s}.jsonl")]; rows = rows[:N] if N else rows
    t0 = time.time()
    with ThreadPoolExecutor(CONC) as ex: txts = list(ex.map(one, rows))
    dt = time.time() - t0
    with open(out + ".part", "w") as f:
        for r, t in zip(rows, txts):
            d = {k: r[k] for k in r if k != "report"}; d.update({"raw": t[:300], "pred": parse_grade(t)}); f.write(json.dumps(d) + "\n")
    os.replace(out + ".part", out); nrep = len({r["report_id"] for r in rows})
    json.dump({"tag": TAG, "set": s, "n": len(rows), "reports": nrep, "seconds": dt, "conc": CONC}, open(f"{P}/{TAG}__{s}.timing.json", "w"))
    print(TAG, s, len(rows), f"{dt:.0f}s", flush=True)
srv.terminate(); print("REF DONE")

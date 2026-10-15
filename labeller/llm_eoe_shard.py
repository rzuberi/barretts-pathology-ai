"""EoE side-quest (Rehan 2026-09-02): adjudicate eosinophilic oesophagitis in
ERIN reports. Input = keyword-positive reports + random keyword-negative
controls. Output per report: diagnosed / suspected / negated_or_incidental /
absent, plus eos-per-HPF count when stated. Same ollama mechanics as the jury.
"""
import csv, json, os, subprocess, time, urllib.request
import pandas as pd

MODEL = os.environ["MODEL"]
INPUT = os.environ["INPUT"]
OUT = os.environ.get("OUTDIR", ".")
SHARD, N_SHARDS = int(os.environ.get("SHARD", 0)), int(os.environ.get("N_SHARDS", 1))
CONC = int(os.environ.get("CONC", 2))
REQ_TIMEOUT = int(os.environ.get("REQ_TIMEOUT", 900))
VALID = {"DIAGNOSED", "SUSPECTED", "NEGATED_OR_INCIDENTAL", "ABSENT"}

os.environ.setdefault("OLLAMA_MODELS", "/mnt/scratche/slow/fmlab/zuberi01/ollama-models")
PORT = 20000 + int(os.environ.get("SLURM_JOB_ID", "0")) % 20000
os.environ["OLLAMA_HOST"] = f"127.0.0.1:{PORT}"
BASE = f"http://127.0.0.1:{PORT}"
srv = subprocess.Popen([os.path.expanduser("~/.local/bin/ollama"), "serve"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
for _ in range(60):
    try:
        urllib.request.urlopen(BASE + "/api/tags", timeout=3); break
    except Exception:
        time.sleep(2)

PROMPT = """You are a GI pathologist. Read this oesophageal pathology report and decide
the status of EOSINOPHILIC OESOPHAGITIS (EoE) in it. Output ONLY JSON:
{"eoe": "<DIAGNOSED|SUSPECTED|NEGATED_OR_INCIDENTAL|ABSENT>", "eos_per_hpf": <number or null>}
DIAGNOSED: report states eosinophilic oesophagitis as a diagnosis, or describes
intraepithelial eosinophil counts/density meeting or clearly consistent with EoE
(e.g. >=15 per HPF) as the conclusion.
SUSPECTED: eosinophilia raised as possible/query EoE, or counts given with
differential including EoE but not concluded.
NEGATED_OR_INCIDENTAL: eosinophils only mentioned as absent/not increased, or a
few eosinophils as part of nonspecific inflammation with no EoE consideration.
ABSENT: no mention of eosinophils at all.

REPORT:
"""

rep = pd.read_csv(INPUT, dtype=str).fillna("")
rep = rep.iloc[SHARD::N_SHARDS]
tag = MODEL.replace(":", "_").replace(".", "_")
out = os.path.join(OUT, f"eoe_{tag}_shard{SHARD}.csv")
done = set()
if os.path.exists(out):
    done = set(pd.read_csv(out)["CaseName"])
todo = [(r["CaseName"], r["_text"]) for _, r in rep.iterrows() if r["CaseName"] not in done]
print(f"{MODEL} shard {SHARD}: {len(todo)} to do", flush=True)
if not os.path.exists(out):
    with open(out, "w", newline="") as f:
        csv.writer(f).writerow(["CaseName", "eoe", "eos_per_hpf"])

def one(cid, text):
    body = json.dumps({"model": MODEL, "prompt": PROMPT + text[:8000] + "\n/no_think",
                       "stream": False, "format": "json", "think": False,
                       "options": {"temperature": 0.0, "num_predict": 200, "num_ctx": 8192}}).encode()
    req = urllib.request.Request(BASE + "/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        r = json.loads(json.loads(urllib.request.urlopen(req, timeout=REQ_TIMEOUT).read())["response"])
        g = str(r.get("eoe", "")).upper().strip()
        n = r.get("eos_per_hpf")
        return (g if g in VALID else "PARSE_FAIL", n if isinstance(n, (int, float)) else "")
    except Exception:
        return ("PARSE_FAIL", "")

from concurrent.futures import ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=CONC) as ex:
    for i in range(0, len(todo), CONC * 4):
        chunk = todo[i:i + CONC * 4]
        res = list(ex.map(lambda t: one(*t), chunk))
        with open(out, "a", newline="") as f:
            w = csv.writer(f)
            for (cid, _), (g, n) in zip(chunk, res):
                w.writerow([cid, g, n])
        if i % 100 == 0: print(f"{i}/{len(todo)}", flush=True)
print("shard complete", flush=True)
srv.terminate()

"""C1: zero-shot prognosis from the patient's report history up to the index date (ERIN cohort v3).
Env: MODEL, SHARD, N_SHARDS, OUTDIR. Reports strictly after the index date are never shown."""
import json, os, re, subprocess, time, urllib.request, threading
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"; ERIN = "/mnt/scratche/fast/fmlab/datasets/imaging/ERIN/data/PathologyReport_AnonIds.csv"
OUT = os.environ.get("OUTDIR", "."); MODEL = os.environ.get("MODEL", "medgemma:27b"); CONC = int(os.environ.get("CONC", "4"))
SHARD = int(os.environ.get("SHARD", "0")); N = int(os.environ.get("N_SHARDS", "1"))
os.environ.setdefault("OLLAMA_MODELS", "/mnt/scratche/slow/fmlab/zuberi01/ollama-models"); os.environ.setdefault("OLLAMA_NUM_PARALLEL", str(CONC))
PORT = 20000 + int(os.environ.get("SLURM_JOB_ID", "0")) % 20000; os.environ["OLLAMA_HOST"] = f"127.0.0.1:{PORT}"; BASE = f"http://127.0.0.1:{PORT}"
slog = open(os.path.join(OUT, "ollama_server.log"), "w"); srv = subprocess.Popen([os.path.expanduser("~/.local/bin/ollama"), "serve"], stdout=slog, stderr=slog)
for _ in range(60):
    try: urllib.request.urlopen(BASE + "/api/tags", timeout=3); break
    except Exception: time.sleep(2)
coh = pd.read_csv(T + "/labeller/erin_progression_cohort_v3.csv", dtype=str); coh["index_date"] = pd.to_datetime(coh.index_date, errors="coerce")
rep = pd.read_csv(ERIN, dtype=str).fillna(""); rep["d"] = pd.to_datetime(rep.CollectedOrOrdered, dayfirst=True, errors="coerce")
TEXT = ["FinalDiagnosis_redacted", "MicroscopicDescription_redacted"]
coh = coh.reset_index(drop=True); coh = coh[coh.index % N == SHARD]
PROMPT = """You are an expert gastrointestinal pathologist and Barrett's surveillance clinician.
Below is one patient's complete oesophageal pathology report history up to and including today's report, oldest first, with dates. Nothing after today is known.
Estimate the probability (0-100) that this patient will be diagnosed with high-grade dysplasia or oesophageal adenocarcinoma within the next 5 years of surveillance.
Consider: current and prior grades, persistence or regression of dysplasia, segment length and extent if described, intestinal metaplasia, inflammation, p53 if mentioned, and the interval pattern.
Reply with ONLY a JSON object: {"risk": <integer 0-100>}
HISTORY:
"""
def history(pid, idx):
    r = rep[(rep.anon_id == pid) & (rep.d <= idx)].sort_values("d")
    parts = [f"[{x.d.date()}] " + " ".join(x[c] for c in TEXT).strip()[:2500] for _, x in r.iterrows()]
    return "\n\n".join(parts[-8:]), len(r)
def ask(text):
    body = json.dumps({"model": MODEL, "prompt": PROMPT + text, "stream": False, "format": "json", "think": False, "options": {"temperature": 0, "num_predict": 60, "num_ctx": 16384}}).encode()
    try:
        resp = json.loads(urllib.request.urlopen(urllib.request.Request(BASE + "/api/generate", data=body, headers={"Content-Type": "application/json"}), timeout=600).read())["response"]
        m = re.search(r'"risk"\s*:\s*"?(\d+)', resp); return int(m.group(1)) if m else None
    except Exception: return None
out = os.path.join(OUT, f"prognosis_{MODEL.replace(':', '_')}_shard{SHARD}.csv"); done = set(pd.read_csv(out, dtype=str).anon_id) if os.path.exists(out) else set()
if not os.path.exists(out): open(out, "w").write("anon_id,n_reports_shown,risk\n")
lock = threading.Lock()
def work(row):
    h, n = history(row.anon_id, row.index_date); r = ask(h) if h else None
    with lock: open(out, "a").write(f"{row.anon_id},{n},{'' if r is None else r}\n")
with ThreadPoolExecutor(CONC) as ex: list(ex.map(work, [r for r in coh.itertuples() if r.anon_id not in done]))
json.dump({"model": MODEL, "shard": SHARD, "n": len(coh)}, open(os.path.join(OUT, "results.json"), "w")); print("done", MODEL, SHARD, len(coh)); srv.terminate()

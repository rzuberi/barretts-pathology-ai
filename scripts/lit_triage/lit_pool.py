"""GPU pool worker for abstract triage. Tasks = (model, chunk of papers) under RUN/queue, ordered model-major
(fastest model first). A worker keeps its loaded model while that model has chunks left, so the whole pool
finishes one model before the next. Chunks are claimed with atomic mkdir; a lock whose Slurm job is gone is reclaimed.
Env: RUN (dir with context.md, abstracts.json), MODELS (comma list, fastest first), CHUNK (default 16)."""
import json, os, re, subprocess, time, urllib.request, shutil
RUN = os.environ["RUN"]; MODELS = os.environ.get("MODELS", "mistral-small3.2:latest,qwen3:32b,gemma3:27b,medgemma:27b,gpt-oss:120b").split(",")
CHUNK = int(os.environ.get("CHUNK", "16")); JOB = os.environ.get("SLURM_JOB_ID", "0")
NGPU = len([g for g in os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",") if g])
BIG = {"gpt-oss:120b", "llama3.3:70b"}  # need 2 L40S (or 1 H200)
GPU_MEM = "h200" if "h200" in os.environ.get("SLURM_JOB_PARTITION", "") else "l40s"
def can_run(m): return m not in BIG or NGPU >= 2 or GPU_MEM == "h200"
os.environ.setdefault("OLLAMA_MODELS", "/mnt/scratche/slow/fmlab/zuberi01/ollama-models"); os.environ["OLLAMA_MAX_LOADED_MODELS"] = "1"
PORT = 20000 + int(JOB) % 20000; os.environ["OLLAMA_HOST"] = f"127.0.0.1:{PORT}"; URL = f"http://127.0.0.1:{PORT}"
Q = f"{RUN}/queue"; OUT = f"{RUN}/out"; os.makedirs(OUT, exist_ok=True)
papers = sorted(json.load(open(f"{RUN}/abstracts.json")).values(), key=lambda p: p["idx"])
chunks = [papers[i:i + CHUNK] for i in range(0, len(papers), CHUNK)]
os.makedirs(Q, exist_ok=True)
slog = open(f"{OUT}/server_{JOB}.log", "w"); srv = subprocess.Popen([os.path.expanduser("~/.local/bin/ollama"), "serve"], stdout=slog, stderr=slog)
for _ in range(120):
    try: urllib.request.urlopen(URL, timeout=2); break
    except Exception: time.sleep(1)
CONTEXT = open(f"{RUN}/context.md").read(); TEMPLATE = open(f"{RUN}/prompt.txt").read()
def tag(m): return m.replace(":", "_").replace("/", "_")
def live_jobs():
    r = subprocess.run(["squeue", "-u", os.environ.get("USER", "zuberi01"), "-h", "-o", "%i"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, universal_newlines=True)
    return set(r.stdout.split()) if r.returncode == 0 else None   # None = Slurm unreachable: never reclaim
def claim(m, k):
    d = f"{Q}/{tag(m)}__{k:03d}"
    if os.path.exists(d + ".done"): return False
    try: os.mkdir(d); open(f"{d}/owner", "w").write(JOB); return True
    except FileExistsError:
        try: owner = open(f"{d}/owner").read().strip()
        except Exception: return False
        live = live_jobs()
        if owner and live is not None and owner not in live:  # stale lock from a killed job
            shutil.rmtree(d, ignore_errors=True)
            try: os.mkdir(d); open(f"{d}/owner", "w").write(JOB); return True
            except FileExistsError: return False
        return False
def score(m, p):
    text = TEMPLATE.replace("{context}", CONTEXT).replace("{title}", p["title"]).replace("{year}", str(p["year"])).replace("{venue}", p["venue"]).replace("{abstract}", p["abstract"][:4000])
    big = m.startswith("gpt-oss")
    body = {"model": m, "prompt": text, "stream": False, "keep_alive": "30m", "options": {"temperature": 0, "seed": 0, "num_predict": 2000 if big else 300, "num_ctx": int(os.environ.get("NUM_CTX", "8192"))}}
    if big: body["think"] = "low"
    else: body["format"] = "json"; body["think"] = False
    raw, parsed = "", None
    for _ in range(2):
        try:
            r = json.load(urllib.request.urlopen(urllib.request.Request(f"{URL}/api/generate", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=600))
            raw = r.get("response", ""); mm = re.search(r"\{.*\}", raw, re.S); parsed = json.loads(mm.group(0)) if mm else None
            if parsed is not None and "score" in parsed: break
        except Exception as e: raw = f"ERROR {e}"
    return parsed, raw
current = None; idle_passes = 0
while not os.path.exists(f"{RUN}/STOP"):
    order = ([current] if current else []) + [m for m in MODELS if m != current]   # model affinity
    if NGPU >= 2 or GPU_MEM == "h200": order = [m for m in order if m in BIG] + [m for m in order if m not in BIG]  # big-GPU workers take the big models first
    got = None
    for m in order:
        if not can_run(m): continue
        for k in range(len(chunks)):
            if claim(m, k): got = (m, k); break
        if got: break
    if not got:
        idle_passes += 1
        if idle_passes >= 2: break   # nothing claimable twice in a row: done (or all held by live workers)
        time.sleep(30); continue
    idle_passes = 0; m, k = got; current = m; d = f"{Q}/{tag(m)}__{k:03d}"
    rows = []
    for p in chunks[k]:
        t0 = time.time(); parsed, raw = score(m, p)
        rows.append({"idx": p["idx"], "title": p["title"], "model": m, "ok": parsed is not None and "score" in parsed, "result": parsed, "raw": raw[:1500], "sec": round(time.time() - t0, 1), "job": JOB})
    tmp = f"{OUT}/{tag(m)}__{k:03d}.jsonl.tmp"
    with open(tmp, "w") as fh: fh.write("".join(json.dumps(r) + "\n" for r in rows))
    os.replace(tmp, f"{OUT}/{tag(m)}__{k:03d}.jsonl"); open(d + ".done", "w").write(JOB); shutil.rmtree(d, ignore_errors=True)
srv.terminate()

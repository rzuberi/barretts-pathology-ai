"""Abstract-level usefulness triage of the histo x CNV literature set with a local ollama model.
Env: MODEL (ollama tag), BASE_DIR (holds abstracts.json; writes out/<model>.jsonl). Resume-capable."""
import json, os, re, subprocess, time, urllib.request
MODEL = os.environ["MODEL"]; BASE = os.environ.get("BASE_DIR", "/mnt/scratche/slow/fmlab/zuberi01/phd/lit_triage")
os.environ.setdefault("OLLAMA_MODELS", "/mnt/scratche/slow/fmlab/zuberi01/ollama-models"); os.environ.setdefault("OLLAMA_NUM_PARALLEL", "1")
PORT = 20000 + int(os.environ.get("SLURM_JOB_ID", "0")) % 20000; os.environ["OLLAMA_HOST"] = f"127.0.0.1:{PORT}"; URL = f"http://127.0.0.1:{PORT}"
os.makedirs(f"{BASE}/out", exist_ok=True); tag = MODEL.replace(":", "_").replace("/", "_")
slog = open(f"{BASE}/out/{tag}.server.log", "w"); srv = subprocess.Popen([os.path.expanduser("~/.local/bin/ollama"), "serve"], stdout=slog, stderr=slog)
for _ in range(120):
    try: urllib.request.urlopen(URL, timeout=2); break
    except Exception: time.sleep(1)
CONTEXT = open(f"{BASE}/context.md").read()
PROMPT = CONTEXT + """

Read the paper below and judge what it could bring to OUR project. Answer with ONE JSON object only:
{"score": 0-3, "idea_types": [subset of "pretrained_model","method","cnv_feature","histology_feature","analysis","label_or_data_source","evaluation","none"], "what_we_could_try": "<=30 words, concrete, specific to our data", "needs": "<=15 words: data/code/weights needed"}
score 3 = gives something concrete we could test on our own H&E + sWGS copy-number Barrett's data now (model weights, method, feature, analysis).
score 2 = transferable idea that needs adaptation or extra data.
score 1 = background or context only.
score 0 = not useful to us.
Be strict: most papers are 1 or 2.

PAPER
Title: {title}
Year/venue: {year}, {venue}
Abstract: {abstract}
"""
papers = json.load(open(f"{BASE}/abstracts.json")); papers = sorted(papers.values(), key=lambda p: p["idx"])
REV = bool(os.environ.get("REVERSE"))  # reverse shard: takes remaining papers from the end, stops where the forward job has reached
fwd = f"{BASE}/out/{tag}.jsonl"; out = f"{BASE}/out/{tag}.rev.jsonl" if REV else fwd
def seen(path):
    s = set()
    if os.path.exists(path):
        for l in open(path): s.add(json.loads(l)["idx"])
    return s
done = seen(out)
if REV: papers = papers[::-1]
gptoss = MODEL.startswith("gpt-oss")
with open(out, "a") as fh:
    for p in papers:
        if p["idx"] in done: continue
        if REV and p["idx"] in seen(fwd): break
        text = PROMPT.replace("{title}", p["title"]).replace("{year}", str(p["year"])).replace("{venue}", p["venue"]).replace("{abstract}", p["abstract"][:4000])
        body = {"model": MODEL, "prompt": text, "stream": False, "options": {"temperature": 0, "seed": 0, "num_predict": 2000 if gptoss else 300, "num_ctx": 8192}}
        if gptoss: body["think"] = "low"
        else: body["format"] = "json"; body["think"] = False
        raw, parsed, t0 = "", None, time.time()
        for attempt in range(2):
            try:
                r = json.load(urllib.request.urlopen(urllib.request.Request(f"{URL}/api/generate", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=600))
                raw = r.get("response", "")
                m = re.search(r"\{.*\}", raw, re.S); parsed = json.loads(m.group(0)) if m else None
                if parsed is not None and "score" in parsed: break
            except Exception as e: raw = f"ERROR {e}"
        fh.write(json.dumps({"idx": p["idx"], "title": p["title"], "model": MODEL, "ok": parsed is not None and "score" in parsed, "result": parsed, "raw": raw[:1500], "sec": round(time.time() - t0, 1)}) + "\n"); fh.flush()
n = len(seen(fwd) | seen(f"{BASE}/out/{tag}.rev.jsonl")); assert n >= len(papers) or REV, f"only {n}/{len(papers)} rows"
json.dump({"model": MODEL, "rows": n}, open(f"{BASE}/out/{tag}{'.rev' if REV else ''}.done.json", "w")); srv.terminate()

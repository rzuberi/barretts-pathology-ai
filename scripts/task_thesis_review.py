"""Thesis red-team review (2.21): local models read the thesis pack and attack it.

Builds a review pack (plan + pre-registrations + drafts + results digest — no
patient data), then asks MODEL for structured criticisms, each tied to a quoted
claim. Output: one JSON list per model; aggregation happens across models later.
"""
import glob, json, os, subprocess, time, urllib.request

T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"
OUT = os.environ.get("OUTDIR", ".")
MODEL = os.environ.get("MODEL", "qwen3:32b")
os.environ.setdefault("OLLAMA_MODELS", "/mnt/scratche/slow/fmlab/zuberi01/ollama-models")
PORT = 21000 + int(os.environ.get("SLURM_JOB_ID", "0")) % 20000
os.environ["OLLAMA_HOST"] = f"127.0.0.1:{PORT}"
BASE = f"http://127.0.0.1:{PORT}"

# --- build pack ---
sections = []
for f in ([T + "/EXECUTION_PLAN.md"] + sorted(glob.glob(T + "/docs/*.md"))
          + sorted(glob.glob(T + "/chapters/*/*.md"))):
    sections.append(f"\n===== FILE: {os.path.basename(f)} =====\n" + open(f).read())
digest = []
for f in sorted(glob.glob(T + "/results/*.json")):
    try:
        digest.append(f"{os.path.basename(f)}: {json.dumps(json.load(open(f)))[:1200]}")
    except Exception:
        pass
pack = "\n".join(sections) + "\n===== RESULTS DIGEST =====\n" + "\n".join(digest)
print(f"pack chars: {len(pack)}", flush=True)

srv = subprocess.Popen([os.path.expanduser("~/.local/bin/ollama"), "serve"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
for _ in range(60):
    try:
        urllib.request.urlopen(BASE + "/api/tags", timeout=3); break
    except Exception: time.sleep(2)
subprocess.run([os.path.expanduser("~/.local/bin/ollama"), "pull", MODEL])

PROMPT = """You are a sceptical PhD examiner in computational pathology examining a
thesis-in-progress. Below are its plan, pre-registrations, chapter drafts, and a
digest of all experimental results. Identify the 10 most serious weaknesses.

Rules: every criticism MUST (a) quote or precisely name the specific claim, number,
or design choice it attacks, (b) explain the flaw in 1-3 sentences, (c) say what
would fix or test it. No generic advice. Rank by severity.

Reply with ONLY a JSON array: [{"target": "...", "flaw": "...", "fix": "...",
"severity": 1-5}, ...]

MATERIALS:
"""
CHUNK = 90000  # chars per pass; review in overlapping windows, merge
crits = []
for i in range(0, len(pack), CHUNK):
    body = json.dumps({"model": MODEL, "prompt": PROMPT + pack[i:i + CHUNK] + "\n/no_think",
                       "stream": False, "format": "json", "think": False,
                       "options": {"temperature": 0.3, "num_predict": 3000, "num_ctx": 32768}}).encode()
    req = urllib.request.Request(BASE + "/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    resp = json.loads(urllib.request.urlopen(req, timeout=1800).read())["response"]
    try:
        got = json.loads(resp)
        if isinstance(got, dict): got = next(iter(got.values()))
        crits.extend([c for c in got if isinstance(c, dict) and "flaw" in c])
        print(f"window {i // CHUNK}: +{len(got)} criticisms", flush=True)
    except Exception as e:
        print(f"window {i // CHUNK}: parse fail {e}: {resp[:200]!r}", flush=True)

tag = MODEL.replace(":", "_").replace("/", "_")
json.dump(crits, open(os.path.join(OUT, f"review_{tag}.json"), "w"), indent=2)
print(f"wrote {len(crits)} criticisms for {MODEL}")
srv.terminate()

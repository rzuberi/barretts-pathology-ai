"""Wave-3 GAP REVIEW (the pre-writing gate, Rehan 2026-08-26): the compute
campaign is believed complete — six frontier families read the ENTIRE results
corpus and answer one question: what compute is missing? Not writing critiques.
An empty/minor list from independent families = gate passes, writing begins.
"""
import glob, json, os, urllib.request
from concurrent.futures import ThreadPoolExecutor

T = os.environ.get("THESIS", "/Users/zuberi01/Documents/thesis")
OUT = os.path.join(T, "review")
KEY = os.environ.get("OPENROUTER_KEY") or open(os.path.expanduser("~/.openrouter_key")).read().strip()
MODELS = ["openai/gpt-5.6-terra-pro", "x-ai/grok-4.6", "moonshotai/kimi-k3",
          "deepseek/deepseek-v4-pro-0813", "z-ai/glm-5.3", "qwen/qwen3.8-max"]

sections = []
for f in ([T + "/EXECUTION_PLAN.md"] + sorted(glob.glob(T + "/docs/*.md"))
          + sorted(glob.glob(T + "/chapters/*/*.md"))):
    sections.append(f"\n===== FILE: {os.path.basename(f)} =====\n" + open(f).read())
digest = []
for f in sorted(glob.glob(T + "/results/*.json")):
    try:
        digest.append(f"{os.path.basename(f)}: {json.dumps(json.load(open(f)))[:2000]}")
    except Exception:
        pass
pack = "\n".join(sections) + "\n===== FULL RESULTS DIGEST =====\n" + "\n".join(digest)
print(f"pack chars: {len(pack)} (~{len(pack)//4} tokens)")

PROMPT = """You are a senior computational-pathology examiner. The PhD student below
believes their COMPUTE campaign is complete and is about to start writing. Your ONLY
job: identify MISSING COMPUTE — experiments, controls, ablations, or statistical
analyses that must run BEFORE writing because a chapter claim depends on them.

Explicitly NOT wanted: writing/presentation advice, figure suggestions, restating
known human-dependent items (pathologist sample R.4, database code map R.5), or
nice-to-have extensions that don't underpin an existing claim. A short or empty
list is a valid and welcome answer if the corpus genuinely supports the claims.

Reply with ONLY a JSON array (empty [] if nothing material is missing):
[{"gap": "...", "claim_at_risk": "which thesis claim depends on it",
  "what_to_run": "concrete experiment/analysis", "blocking": true|false,
  "severity": 1-5}, ...]

MATERIALS:
""" + pack

def run_one(m):
    tag = m.replace("/", "_").replace(".", "_").replace(":", "_")
    path = os.path.join(OUT, f"gapreview_{tag}.json")
    if os.path.exists(path):
        print(f"{m}: already done", flush=True); return 1
    body = json.dumps({"model": m, "max_tokens": 16000, "temperature": 0.3,
                       "reasoning": {"effort": "low"},
                       "messages": [{"role": "user", "content": PROMPT}]}).encode()
    req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions",
                                 data=body, headers={"Authorization": f"Bearer {KEY}",
                                                     "Content-Type": "application/json"})
    try:
        r = json.loads(urllib.request.urlopen(req, timeout=900).read())
        msg = r["choices"][0]["message"]
        txt = msg.get("content") or msg.get("reasoning_content") or ""
        usage = r.get("usage", {})
    except Exception as e:
        print(f"{m}: SKIP ({type(e).__name__}: {str(e)[:120]})", flush=True); return 0
    if not txt.strip():
        print(f"{m}: SKIP (empty)", flush=True); return 0
    start, end = txt.find("["), txt.rfind("]") + 1
    try:
        gaps = json.loads(txt[start:end])
    except Exception:
        cut = txt.rfind("}")
        try:
            gaps = json.loads(txt[start:cut + 1] + "]")
        except Exception:
            gaps = [{"raw": txt}]
    json.dump(gaps, open(path, "w"), indent=2)
    print(f"{m}: {len(gaps)} gaps | tokens {usage.get('prompt_tokens')}+{usage.get('completion_tokens')}", flush=True)
    return 1

with ThreadPoolExecutor(max_workers=6) as ex:
    done = sum(ex.map(run_one, MODELS))
print(f"models completed: {done}/{len(MODELS)}", flush=True)

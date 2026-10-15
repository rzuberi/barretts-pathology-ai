"""Frontier-model thesis review via OpenRouter (2.21 extension, Rehan 2026-08-25).

Single-pass review of the full pack (no chunking — frontier context). Key read
from OPENROUTER_KEY env or ~/.openrouter_key. Pack contains NO patient data
(plan, pre-registrations, drafts, results digest only). Skips unavailable models.
"""
import glob, json, os, urllib.request

T = os.environ.get("THESIS", "/Users/zuberi01/Documents/thesis")
OUT = os.path.join(T, "review")
os.makedirs(OUT, exist_ok=True)
KEY = os.environ.get("OPENROUTER_KEY") or open(os.path.expanduser("~/.openrouter_key")).read().strip()
# wave 2 (2026-08-25): five families unused in wave 1; wave-1 findings doc is
# excluded from the pack so this wave is blind to prior conclusions.
MODELS = ["deepseek/deepseek-v4-pro-0813", "moonshotai/kimi-k3", "z-ai/glm-5.3",
          "qwen/qwen3.8-max", "google/gemini-3.7-flash"]
MAX_REVIEWERS = 5
BLIND = ("review_findings",)

sections = []
for f in ([T + "/EXECUTION_PLAN.md"] + sorted(glob.glob(T + "/docs/*.md"))
          + sorted(glob.glob(T + "/chapters/*/*.md"))):
    if any(b in f for b in BLIND):
        continue
    sections.append(f"\n===== FILE: {os.path.basename(f)} =====\n" + open(f).read())
digest = [f"{os.path.basename(f)}: {json.dumps(json.load(open(f)))[:1500]}"
          for f in sorted(glob.glob(T + "/results/*.json"))]
pack = "\n".join(sections) + "\n===== RESULTS DIGEST =====\n" + "\n".join(digest)
print(f"pack chars: {len(pack)} (~{len(pack)//4} tokens)")

PROMPT = """You are a sceptical PhD examiner in computational pathology examining a
thesis-in-progress on multimodal deep learning for oesophageal cancer. Below are its
execution plan, pre-registrations, chapter drafts, and a digest of every experimental
result. This work was largely designed and executed with the help of a frontier LLM
from a different lab; hunt specifically for the failure modes such work exhibits:
circular reasoning, self-validation loops, claims outrunning evidence, missing
baselines/corrections, endpoint choices that flatter the hypothesis, and
inconsistencies between plan and results.

Identify the 12 most serious weaknesses. Every criticism MUST (a) quote or precisely
name the claim/number/design it attacks, (b) explain the flaw in 1-3 sentences,
(c) give the concrete fix or test. Rank by severity (5 = thesis-threatening).

Reply with ONLY a JSON array: [{"target": "...", "flaw": "...", "fix": "...",
"severity": 1-5}, ...]

MATERIALS:
""" + pack

done = 0
for m in MODELS:
    if done >= MAX_REVIEWERS: break
    body = json.dumps({"model": m, "max_tokens": 16000, "temperature": 0.3,
                       "reasoning": {"effort": "low"},
                       "messages": [{"role": "user", "content": PROMPT}]}).encode()
    req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions",
                                 data=body, headers={"Authorization": f"Bearer {KEY}",
                                                     "Content-Type": "application/json"})
    tag = m.replace("/", "_").replace(".", "_").replace(":", "_")
    if glob.glob(os.path.join(OUT, f"review_frontier_w2_{tag}.json")):
        print(f"{m}: already done"); done += 1; continue
    try:
        r = json.loads(urllib.request.urlopen(req, timeout=900).read())
        msg = r["choices"][0]["message"]
        txt = msg.get("content") or msg.get("reasoning_content") or msg.get("reasoning") or ""
        usage = r.get("usage", {})
    except Exception as e:
        print(f"{m}: SKIP ({type(e).__name__}: {str(e)[:120]})")
        continue
    if not txt.strip():
        print(f"{m}: SKIP (empty content; keys={list(msg)})"); continue
    start, end = txt.find("["), txt.rfind("]") + 1
    try:
        crits = json.loads(txt[start:end])
    except Exception:
        # salvage a truncated array: keep up to the last complete object
        cut = txt.rfind("}")
        try:
            crits = json.loads(txt[start:cut + 1] + "]")
        except Exception:
            crits = [{"raw": txt}]
    json.dump(crits, open(os.path.join(OUT, f"review_frontier_w2_{tag}.json"), "w"), indent=2)
    print(f"{m}: {len(crits)} criticisms | tokens {usage.get('prompt_tokens')}+{usage.get('completion_tokens')}")
    done += 1
print(f"reviewers completed: {done}")

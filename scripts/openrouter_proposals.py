"""Constructive consultation (Rehan 2026-08-25): ask frontier + small models for
proposals to improve results, strengthen the thesis, and suggest campaigns —
additions to existing chapters or new chapters. Unlike the red-team review, the
pack INCLUDES review_findings.md: proposers should know the weaknesses.
Raw answers logged one JSON per model under review/proposals_<tag>.json.
"""
import glob, json, os, urllib.request

T = os.environ.get("THESIS", "/Users/zuberi01/Documents/thesis")
OUT = os.path.join(T, "review")
os.makedirs(OUT, exist_ok=True)
KEY = os.environ.get("OPENROUTER_KEY") or open(os.path.expanduser("~/.openrouter_key")).read().strip()

BIG = ["openai/gpt-5.6-terra-pro", "x-ai/grok-4.6", "moonshotai/kimi-k3",
       "deepseek/deepseek-v4-pro-0813", "z-ai/glm-5.3", "qwen/qwen3.8-max"]
SMALL = ["openai/gpt-5.6-luna", "google/gemini-3.7-flash",
         "deepseek/deepseek-v4-flash-0731", "qwen/qwen3.7-flash"]
MODELS = BIG + SMALL

sections = []
for f in ([T + "/EXECUTION_PLAN.md"] + sorted(glob.glob(T + "/docs/*.md"))
          + sorted(glob.glob(T + "/chapters/*/*.md"))):
    sections.append(f"\n===== FILE: {os.path.basename(f)} =====\n" + open(f).read())
digest = [f"{os.path.basename(f)}: {json.dumps(json.load(open(f)))[:1500]}"
          for f in sorted(glob.glob(T + "/results/*.json"))]
pack = "\n".join(sections) + "\n===== RESULTS DIGEST =====\n" + "\n".join(digest)
print(f"pack chars: {len(pack)} (~{len(pack)//4} tokens)")

PROMPT = """You are a senior computational-pathology researcher advising a PhD student
whose thesis (materials below) is 'histology-anchored multimodal deep learning in
oesophageal cancer: where and why fusion fails to replicate, and how to validate
report-derived supervision'. The red-team findings are included — you know the
weaknesses. Available resources: SWGCohort (150pt Barrett's, H&E+sWGS), ERIN
(2,280 imaged cases + free-text reports), OCCAMS (87-141pt H&E+WGS+clinical),
TCGA-ESCA/STAD (446 slides+omics+reports), a Slurm cluster with 16xH200 + 24xL40S,
five extracted foundation-model encoders, an 8-LLM labelling jury, and a working
ABMIL/Cox/fusion codebase.

Propose the 8 most valuable additions: experiments/campaigns that IMPROVE existing
results, STRENGTHEN weak claims, or open NEW chapters. Favour high
insight-per-GPU-hour. Both quick wins and one or two ambitious ideas welcome.
Be concrete: what to run, on what data, what the readout is, and what result would
change the thesis. No generic advice (no "get more data", "consult a statistician").

Reply with ONLY a JSON array:
[{"title": "...", "type": "improve_existing|new_experiment|new_chapter",
  "chapter": "Ch1-5|new", "design": "what to run, data, readout (2-4 sentences)",
  "decisive_result": "what outcome would change/strengthen the thesis",
  "compute": "laptop|CPU-hours|GPU-hours estimate", "impact": 1-5}, ...]

MATERIALS:
""" + pack

done = 0
for m in MODELS:
    tag = m.replace("/", "_").replace(".", "_").replace(":", "_")
    path = os.path.join(OUT, f"proposals_{tag}.json")
    if os.path.exists(path):
        print(f"{m}: already done"); done += 1; continue
    body = json.dumps({"model": m, "max_tokens": 16000, "temperature": 0.7,
                       "reasoning": {"effort": "low"},
                       "messages": [{"role": "user", "content": PROMPT}]}).encode()
    req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions",
                                 data=body, headers={"Authorization": f"Bearer {KEY}",
                                                     "Content-Type": "application/json"})
    try:
        r = json.loads(urllib.request.urlopen(req, timeout=900).read())
        msg = r["choices"][0]["message"]
        txt = msg.get("content") or msg.get("reasoning_content") or msg.get("reasoning") or ""
        usage = r.get("usage", {})
    except Exception as e:
        print(f"{m}: SKIP ({type(e).__name__}: {str(e)[:120]})")
        continue
    if not txt.strip():
        print(f"{m}: SKIP (empty content)"); continue
    start, end = txt.find("["), txt.rfind("]") + 1
    try:
        props = json.loads(txt[start:end])
    except Exception:
        cut = txt.rfind("}")
        try:
            props = json.loads(txt[start:cut + 1] + "]")
        except Exception:
            props = [{"raw": txt}]
    json.dump(props, open(path, "w"), indent=2)
    print(f"{m}: {len(props)} proposals | tokens {usage.get('prompt_tokens')}+{usage.get('completion_tokens')}")
    done += 1
print(f"models completed: {done}/{len(MODELS)}")

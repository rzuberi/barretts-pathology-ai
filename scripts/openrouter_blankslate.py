"""Blank-slate ideation (Rehan 2026-08-25): models see ONLY the data + compute
inventory — no thesis framing, no hypothesis, no results — and propose research
programmes. Contrast with the primed consultation shows which of our directions
are 'obvious given the data' vs which are choices, and surfaces directions we
never considered. Log-only: nothing is promoted from here without joint decision.
"""
import json, os, urllib.request
from concurrent.futures import ThreadPoolExecutor

T = os.environ.get("THESIS", "/Users/zuberi01/Documents/thesis")
OUT = os.path.join(T, "review")
KEY = os.environ.get("OPENROUTER_KEY") or open(os.path.expanduser("~/.openrouter_key")).read().strip()

MODELS = ["openai/gpt-5.6-terra-pro", "x-ai/grok-4.6", "moonshotai/kimi-k3",
          "deepseek/deepseek-v4-pro-0813", "z-ai/glm-5.3", "qwen/qwen3.8-max",
          "openai/gpt-5.6-luna", "google/gemini-3.7-flash",
          "deepseek/deepseek-v4-flash-0731", "qwen/qwen3.7-flash"]

INVENTORY = """DATA (all with appropriate approvals, oesophageal/gastric focus):
1. SWGCohort: 150 Barrett's oesophagus surveillance patients, 707 biopsy samples;
   H&E whole-slide images + shallow whole-genome sequencing (copy number) per
   sample; longitudinal follow-up with progression outcomes (HGD/cancer);
   pathologist grade per biopsy.
2. ERIN: 2,280 imaged oesophageal cases (H&E WSIs) from ~1,600 patients, each
   linked to its free-text pathology report (7,149 reports across the wider
   corpus); longitudinal (median span ~3.75 years, many patients with multiple
   timepoints); no structured labels beyond what can be extracted from text.
3. OCCAMS: 87-225 oesophageal adenocarcinoma resection cases; H&E WSIs + full
   WGS (TP53 status, ploidy, whole-genome doubling) + clinical outcome data
   (survival, stage, treatment).
4. TCGA-ESCA + TCGA-STAD: 446 cases with diagnostic WSIs, full multi-omics
   (mutations, CNV, expression), machine-readable pathology reports, and
   survival endpoints; 9,517 pan-cancer report+slide pairs available beyond GI.
5. Barrett's specialist database: 13,645 free-text pathology reports (no images
   for most), longitudinal per patient.

FEATURES ALREADY EXTRACTED: tile-level embeddings for all WSIs from five
pathology foundation models (UNI2-h, Virchow2, GigaPath, Phikon-v2,
H-Optimus-0).

COMPUTE: Slurm cluster with 16x NVIDIA H200 (141GB), 24x L40S (48GB), large
CPU partition; can run local LLMs up to ~120B (ollama) on-cluster; no
patient data may leave the cluster.

TOOLING THAT EXISTS: attention-based MIL training code (classification + Cox
survival), an 8-model local-LLM jury pipeline for extracting labels from
free-text reports, a rule-based report labeller."""

PROMPT = """You are a creative senior researcher in computational pathology. Below is
an inventory of datasets, precomputed features, and compute available to a PhD
student. Nothing else is decided — no thesis question, no prior results. Propose
the 8 most interesting, feasible research ideas this specific combination of
resources enables. Favour ideas where this data combination is unusual or uniquely
suited; both safe and bold ideas welcome. Be concrete about design and readout.

Reply with ONLY a JSON array:
[{"title": "...", "idea": "2-3 sentence pitch", "data_used": ["..."],
  "design": "what to run and what the readout is (2-4 sentences)",
  "why_this_data": "what makes this combination suited/unique for it",
  "impact": 1-5, "compute": "laptop|CPU|GPU-hours estimate"}, ...]

INVENTORY:
""" + INVENTORY

def run_one(m):
    tag = m.replace("/", "_").replace(".", "_").replace(":", "_")
    path = os.path.join(OUT, f"blankslate_{tag}.json")
    if os.path.exists(path):
        print(f"{m}: already done", flush=True); return 1
    body = json.dumps({"model": m, "max_tokens": 16000, "temperature": 0.9,
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
        print(f"{m}: SKIP ({type(e).__name__}: {str(e)[:120]})", flush=True); return 0
    if not txt.strip():
        print(f"{m}: SKIP (empty content)", flush=True); return 0
    start, end = txt.find("["), txt.rfind("]") + 1
    try:
        ideas = json.loads(txt[start:end])
    except Exception:
        cut = txt.rfind("}")
        try:
            ideas = json.loads(txt[start:cut + 1] + "]")
        except Exception:
            ideas = [{"raw": txt}]
    json.dump(ideas, open(path, "w"), indent=2)
    print(f"{m}: {len(ideas)} ideas | tokens {usage.get('prompt_tokens')}+{usage.get('completion_tokens')}", flush=True)
    return 1

with ThreadPoolExecutor(max_workers=10) as ex:
    done = sum(ex.map(run_one, MODELS))
print(f"models completed: {done}/{len(MODELS)}", flush=True)

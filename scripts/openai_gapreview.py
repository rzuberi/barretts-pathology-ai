"""Astra gap review: 3 adversarial passes + synthesis over the thesis dossier.

Dossier = claims register + project summaries + EXECUTION_PLAN + all results
JSONs (minified). Aggregate numbers only — NO report text or patient data.
Passes run in parallel; synthesis merges/dedupes/ranks. Outputs to
review/astra/. Key: ~/.openai_key. Cost guard: max_output_tokens per call.
"""
import glob, json, os, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

T = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEY = open(os.path.expanduser("~/.openai_key")).read().strip()
MODEL = "gpt-6-astra"
OUTD = os.path.join(T, "review", "astra")
os.makedirs(OUTD, exist_ok=True)

# ---- dossier ----
parts = ["# THESIS DOSSIER (aggregate statistics only)\n"]
for f in ("docs/claims_register.md", "docs/project_summaries.md", "EXECUTION_PLAN.md"):
    parts.append("\n\n===== FILE: %s =====\n" % f)
    parts.append(open(os.path.join(T, f)).read())
parts.append("\n\n===== RESULTS JSONS (ground truth) =====\n")
for f in sorted(glob.glob(os.path.join(T, "results", "*.json"))):
    try:
        parts.append("\n--- results/%s ---\n" % os.path.basename(f))
        parts.append(json.dumps(json.load(open(f)), separators=(",", ":")))
    except Exception as e:
        parts.append("(unreadable: %s)" % e)
DOSSIER = "".join(parts)
print("dossier chars:", len(DOSSIER))

SCHEMA = """Respond with ONLY a JSON array. Each finding:
{"id": "A1", "category": "missing_computation|methodological_flaw|statistical_issue|claim_overreach|missing_baseline",
 "severity": "blocker|major|minor", "target_claim": "C7 or file/section",
 "the_gap": "...", "why_it_matters": "...", "concrete_fix": "...",
 "addressable_with_existing_data": true, "est_compute": "none|cpu-hours|gpu-hours"}
Findings that merely restate the dossier's Known Limitations list score zero —
omit them unless you add something genuinely new about them. Max 12 findings,
fewer is better. If a category is clean, return fewer findings, not filler."""

PASSES = {
    "stats": "You are a hostile statistical reviewer for a top methods journal. "
        "Attack ONLY the statistics: fold hygiene and leakage (incl. subtle "
        "leakage via label provenance, patient overlap, or arm selection), "
        "bootstrap and CI validity, multiplicity handling, power claims, "
        "calibration, metric choices, and whether any number in the results "
        "JSONs contradicts the procedure described. Check arithmetic where "
        "possible.",
    "compute": "You are the examiner deciding whether this PhD's experimental "
        "programme is COMPLETE. Identify missing computations: controls, "
        "ablations, baselines, sensitivity analyses, or replications a "
        "sceptical examiner would demand before accepting the claims — "
        "prioritise ones that could OVERTURN a headline claim, and say for "
        "each whether existing saved artefacts (OOF predictions, folds, "
        "labels) suffice to run it cheaply.",
    "claims": "You are auditing claim-evidence alignment. For EACH claim "
        "C1-C25 decide: does the cited JSON actually license the claim as "
        "worded? Flag overreach, wrong directionality, confounded "
        "comparisons, numbers that don't match, and claims whose evidence "
        "is exploratory but worded as confirmatory. Report only claims with "
        "problems.",
}

def call(name, system, user, max_out=30000):
    body = {"model": MODEL, "max_output_tokens": max_out,
            "reasoning": {"effort": "medium"},
            "input": [{"role": "system", "content": system},
                      {"role": "user", "content": user}]}
    req = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(body).encode(),
        headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
    t0 = time.time()
    raw = json.load(urllib.request.urlopen(req, timeout=1800))
    txt = "".join(c.get("text", "") for item in raw.get("output", [])
                  if item.get("type") == "message"
                  for c in item.get("content", []))
    json.dump(raw, open(os.path.join(OUTD, name + "_raw.json"), "w"))
    open(os.path.join(OUTD, name + ".txt"), "w").write(txt)
    u = raw.get("usage", {})
    print("%s done %.0fs in=%s out=%s" % (name, time.time() - t0,
          u.get("input_tokens"), u.get("output_tokens")), flush=True)
    return txt, u

if __name__ == "__main__":
    usage = []
    with ThreadPoolExecutor(3) as ex:
        futs = {k: ex.submit(call, k, brief + "\n\n" + SCHEMA, DOSSIER)
                for k, brief in PASSES.items()}
        outs = {k: f.result() for k, f in futs.items()}
    usage += [outs[k][1] for k in outs]
    syn_in = "\n\n".join("### PASS %s findings\n%s" % (k, outs[k][0]) for k in outs)
    syn, u = call("synthesis",
        "Three adversarial review passes over one thesis are attached. Merge "
        "them: dedupe overlapping findings, drop any that merely restate the "
        "known-limitations list, rank the rest by (severity, how cheaply "
        "addressable). " + SCHEMA.replace("Max 12", "Max 15"),
        "Known limitations (score zero):\n" +
        open(os.path.join(T, "docs/claims_register.md")).read().split("## Known limitations")[1]
        + "\n\n" + syn_in, 30000)
    usage.append(u)
    ti = sum(x.get("input_tokens", 0) for x in usage)
    to = sum(x.get("output_tokens", 0) for x in usage)
    print("TOTAL in=%d out=%d est_cost=$%.2f" % (ti, to, ti * 10 / 1e6 + to * 50 / 1e6))

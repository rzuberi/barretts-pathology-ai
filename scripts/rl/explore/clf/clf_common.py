"""clf strand shared bits. Reads the RL project's data read-only; writes only under analysis/rl_explore/clf."""
import json, os, re, sys
B = "/mnt/scratche/slow/fmlab/zuberi01"
sys.path.insert(0, B + "/phd/thesis/scripts/rl"); from common import GRADES, ORD, POS   # label constants only
D = B + "/analysis/rl_grader/data"; O = B + "/analysis/rl_explore/clf"; HF = B + "/hf_cache"
MODELS = {"mbb": ("thomas-sounack/BioClinical-ModernBERT-base", 2048), "mbl": ("thomas-sounack/BioClinical-ModernBERT-large", 2048),
          "cbert": ("emilyalsentzer/Bio_ClinicalBERT", 512)}
SETS = ["train", "val", "test", "test485", "pop"]
def rows(s): return [json.loads(l) for l in open(f"{D}/{s}.jsonl")]
def sec_re(s): return re.compile(rf"(?<![A-Za-z]){re.escape(s)}(?![A-Za-z])")
def text_for(r, budget_tokens=None, ntok=None):
    """'[SECTION X]\\n' + report. With a token budget (512-ctx models): keep lines naming the section first, then the rest in order."""
    head = f"[SECTION {r['section']}]\n"; rep = r["report"]
    if budget_tokens is None or ntok is None or ntok(head + rep) <= budget_tokens: return head + rep
    lines = rep.split("\n"); rx = sec_re(r["section"]) if r["section"] != "WHOLE" else None
    pri = [i for i, l in enumerate(lines) if rx and rx.search(l)]; keep = set(); used = ntok(head)
    for i in pri + [i for i in range(len(lines)) if i not in pri]:
        c = ntok(lines[i]) + 1
        if used + c > budget_tokens:
            if i in pri: keep.add(i)   # section lines always kept (tokenizer truncation handles the tail)
            continue
        keep.add(i); used += c
    return head + "\n".join(lines[i] for i in sorted(keep))

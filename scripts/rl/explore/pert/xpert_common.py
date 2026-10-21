"""Explore/pert shared: PerturbQA three-class prompt with optional knowledge (PerturbQA gene summaries) and retrieval sentence.
Copy of scripts/rl/pert_common.py with PROMPT modes: plain (Task F), k (summaries), r (retrieved training outcomes), kr (both)."""
import json, os, re
B = "/mnt/scratche/slow/fmlab/zuberi01"; P = B + "/perturbqa"; Q = P + "/PerturbQA/perturbqa/datasets"
F = B + "/analysis/rl_followups/F"                      # Task F data (items) and preds, read only
O = B + "/analysis/rl_explore/pert"                     # our outputs
D = O + "/downloads"; GS = D + "/gene_summary/gene_summary"; KG = D + "/kg/perturbqa/datasets/kg"
CELLS = ["k562", "rpe1", "hepg2", "jurkat"]
CELL_NAME = {"k562": "K562 (chronic myelogenous leukaemia)", "rpe1": "RPE1 (retinal pigment epithelium)", "hepg2": "HepG2 (hepatocellular carcinoma)", "jurkat": "Jurkat (T-cell leukaemia)"}
WORDS = ["up", "down", "none"]
PROMPT = os.environ.get("PROMPT", "plain")
_desc = {}
def desc(kind, g):
    if kind not in _desc: _desc[kind] = json.load(open(f"{GS}/desc_{kind}.json"))
    return _desc[kind].get(g, "").strip()
def messages(cell, pert, gene, ret=None, think=False):
    q = (f"In {CELL_NAME[cell]} cells, the gene {pert} is knocked down with CRISPR interference. "
         f"Compared with non-targeting control cells, does the expression of {gene} go up, go down, or not change significantly?")
    if PROMPT in ("k", "kr"):
        q += f"\n\nAbout the knocked-down gene {pert}: {desc('pert', pert)}\n\nAbout the measured gene {gene}: {desc('gene', gene)}"
    if PROMPT in ("kr", "r") and ret: q += f"\n\nFrom the training data: {ret}"
    q += "\n\nAnswer with exactly one word: up, down or none." + ("" if think else " /no_think")
    return [{"role": "user", "content": q}]
def parse(text):
    t = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip().lower()
    m = re.match(r"[^a-z]*([a-z]+)", t)
    return m.group(1) if m and m.group(1) in WORDS else None
def data_path(s):
    """our data copies carry the retrieval sentence ('ret'); fall back to Task F's items"""
    f = f"{O}/data/{s}.jsonl"; return f if os.path.exists(f) else f"{F}/data/{s}.jsonl"

"""Task F shared: PerturbQA three-class prompt and answer parsing."""
import re
P = "/mnt/scratche/slow/fmlab/zuberi01/perturbqa"; Q = P + "/PerturbQA/perturbqa/datasets"; O = "/mnt/scratche/slow/fmlab/zuberi01/analysis/rl_followups/F"
CELLS = ["k562", "rpe1", "hepg2", "jurkat"]
CELL_NAME = {"k562": "K562 (chronic myelogenous leukaemia)", "rpe1": "RPE1 (retinal pigment epithelium)", "hepg2": "HepG2 (hepatocellular carcinoma)", "jurkat": "Jurkat (T-cell leukaemia)"}
WORDS = ["up", "down", "none"]
def messages(cell, pert, gene):
    return [{"role": "user", "content": (f"In {CELL_NAME[cell]} cells, the gene {pert} is knocked down with CRISPR interference. "
            f"Compared with non-targeting control cells, does the expression of {gene} go up, go down, or not change significantly? "
            "Answer with exactly one word: up, down or none. /no_think")}]
def parse(text):
    t = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip().lower()
    m = re.match(r"[^a-z]*([a-z]+)", t)
    return m.group(1) if m and m.group(1) in WORDS else None

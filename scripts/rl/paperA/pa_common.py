"""Paper A (jury -> 150M encoder) shared bits. Reads project-1 / clf-strand data READ-ONLY; writes only under analysis/paperA."""
import glob, json, os, sys
from collections import Counter
import numpy as np
B = "/mnt/scratche/slow/fmlab/zuberi01"
sys.path.insert(0, B + "/phd/thesis/scripts/rl"); from common import GRADES, ORD, POS   # label constants only
T = B + "/phd/thesis"; D = B + "/analysis/rl_grader/data"; CLF = B + "/analysis/rl_explore/clf"; O = B + "/analysis/paperA"; HF = B + "/hf_cache"
MODEL = "thomas-sounack/BioClinical-ModernBERT-base"; MAXLEN = 2048; NB = 2000
TIERS = {"NORMAL_OTHER": "non-dysplastic", "NDBE": "non-dysplastic", "IND": "IND", "LGD": "LGD", "HGD": "HGD+cancer", "CANCER": "HGD+cancer"}
def rows(s, d=D): return [json.loads(l) for l in open(f"{d}/{s}.jsonl")]
def load_preds(prefix, s):
    f = f"{prefix}__{s}.jsonl"; return {json.loads(l)["id"]: json.loads(l) for l in open(f)} if os.path.exists(f) else None
def mkdir(p): os.makedirs(p, exist_ok=True); os.chmod(p, 0o700); return p
def patient_bootstrap(pats, nb=NB, seed=0):
    up = np.unique(pats); grp = {p: np.where(pats == p)[0] for p in up}; rng = np.random.RandomState(seed)
    return [np.concatenate([grp[p] for p in rng.choice(up, len(up))]) for _ in range(nb)]
ci = lambda v: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
def juror_votes(cases):
    """(CaseName, section) -> juror -> set of grades, from the v3 per-juror outputs (rule as eleventh_juror.py)."""
    import pandas as pd
    votes = {}
    for f in glob.glob(T + "/feasibility/runs/sec3full_*/output/sections_v3_*.csv"):
        juror = os.path.basename(f).replace("sections_v3_", "").rsplit("_shard", 1)[0]
        d = pd.read_csv(f, dtype=str).fillna("").drop_duplicates("CaseName"); d = d[d.CaseName.isin(cases)]
        for r in d.itertuples():
            if r.sections_json.startswith("PARSE_FAIL"): continue
            seen = {}
            for s in json.loads(r.sections_json):
                key = s["section"].upper().strip(); key = "WHOLE" if key in ("WHOLE", "WHOLE_REPORT", "") else key[:2]
                seen.setdefault(key, set()).update(g for g in s["grades"] if g in ORD)
            for key, gs in seen.items():
                if gs: votes.setdefault((r.CaseName, key), {})[juror] = gs
    return votes
def consensus(jv):
    """build_sections_v3_consensus rule: grade kept if listed by >= ceil(n/2) of the returning jurors; consensus = worst kept. -> (grade, vote fraction)"""
    n = len(jv)
    if n == 0: return None, float("nan")
    thr = (n + 1) // 2; cnt = Counter(g for gs in jv.values() for g in gs); kept = [g for g, k in cnt.items() if k >= thr]
    if not kept: return None, float("nan")
    w = max(kept, key=lambda g: ORD[g]); return w, cnt[w] / n
def worst(gs): return max(gs, key=lambda g: ORD[g]) if gs else None

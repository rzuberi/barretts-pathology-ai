"""Item 6a, frozen triage cut-offs (docs/paper_triage_robustness.md @ 32653f1): medians (and ranges) over the 100 repeat x fold cut-offs of paper_triage.md,
all pre-event samples, recomputed with the same rules; must reproduce results/paper_final/triage/<src>_pre.json cut-off medians and ranges.
Writes results/paper_final/triage_robustness/frozen_cutoffs_<src>.json (their -> copied into models/killcoyne_frozen_v1/triage_cutoffs.json locally)."""
import os, json, numpy as np
from tb_common import *
os.makedirs(OUT, exist_ok=True)
for src in ("their", "pkg"):
    popm = popmask("pre"); O, I, FO = load(src); band, call, info = triage(O, I, FO, popm); _, thr = comparators(O, I, FO, popm)
    ref = json.load(open(f"{T}/results/paper_final/triage/{src}_pre.json"))["folds"]["cutoffs"]
    for k_new, k_ref in (("t_low", "t_low"), ("c_star", "c_star"), ("t_seq", "t_seq_L-LATE")):
        assert r3(np.median(info[k_new])) == ref[k_ref]["median"] and [r3(min(info[k_new])), r3(max(info[k_new]))] == ref[k_ref]["range"], (src, k_new)
    med = lambda v: float(np.median(v)); rng_ = lambda v: [float(min(v)), float(max(v))]
    out = {"model_bundle": "models/killcoyne_frozen_v1" if src == "their" else "models/killcoyne_frozen_pkg_v1 (reported only; not written into that bundle)",
           "source": "docs/paper_triage.md (pre-spec f5944d2, results 650701b); recomputed by scripts/paper_plan/tb_frozen.py (docs/paper_triage_robustness.md @ 32653f1)",
           "population": "all pre-event samples of the 676 samples from 80 patients of the Killcoyne discovery cohort with a matched H&E slide (571 samples, 75 patients)",
           "statistic": "median (and range) over the 100 repeat x fold cut-offs chosen on inner out-of-fold training predictions",
           "rules": {"t_low": "L-IMG < t_low -> cleared (95% training sensitivity)", "t_high": "L-IMG > t_high -> flagged high risk (t_high = c*, 90% training specificity)",
                     "t_seq": "t_low <= L-IMG <= t_high -> sequenced; positive if L-LATE >= t_seq (80% sensitivity among training middle-band cases)",
                     "t_B": "comparator (B): positive if L-LATE >= t_B (80% training sensitivity), needed for the triage criterion"},
           "cutoffs": {"t_low": med(info["t_low"]), "t_high": med(info["c_star"]), "t_seq": med(info["t_seq"]), "t_B": med(thr["B"])},
           "ranges": {"t_low": rng_(info["t_low"]), "t_high": rng_(info["c_star"]), "t_seq": rng_(info["t_seq"]), "t_B": rng_(thr["B"])},
           "caveat": "chosen on fold-model predictions; applying them to the frozen models (fitted on all 676 samples) assumes similar score distributions on new data"}
    json.dump(out, open(f"{OUT}/frozen_cutoffs_{src}.json", "w"), indent=1, sort_keys=True); print("TB FROZEN DONE", src, out["cutoffs"], flush=True)

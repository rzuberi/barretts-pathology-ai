"""Item 3, CIs including threshold selection (docs/paper_triage_robustness.md @ 32653f1). TASK = <src>_<pop>, SHARD = 0..9 (200 draws each of the 2,000
patient-bootstrap draws, same seed and redraw rule). In each draw every cut-off of every repeat x fold is re-chosen on the resampled inner training predictions
(integer patient multiplicities = replication), calls and vote recomputed, metrics on the resampled rows. MERGE=1 combines shards -> selci_<TASK>.json."""
import os, json, glob, numpy as np
from tb_common import *
TASK = os.environ["TASK"]; src, pop = TASK.split("_", 1); popm = popmask(pop); rows = np.where(popm)[0]; D = f"{ROW}/selci"; os.makedirs(D, exist_ok=True); os.makedirs(OUT, exist_ok=True)
KEYS = ["sens_T", "spec_T", "sens_A", "spec_A", "sens_B", "spec_B", "sens_C", "spec_C", "share_mid"]
if os.environ.get("MERGE") == "1":
    A = np.vstack([np.load(f) for f in sorted(glob.glob(f"{D}/{TASK}_shard_*.npy"))]); assert A.shape[0] == NB, A.shape
    fx = json.load(open(f"{T}/results/paper_final/triage/{TASK}.json")); col = {k: A[:, i] for i, k in enumerate(KEYS)}
    res = {"task": TASK, "prespec": "32653f1", "draws": int(A.shape[0]), "selection_inclusive": {}}
    for s, nm in (("T", "triage_LATE"), ("A", "A"), ("B", "B"), ("C", "C")):
        res["selection_inclusive"][nm] = {"sensitivity_ci95": pct(col[f"sens_{s}"]), "specificity_ci95": pct(col[f"spec_{s}"]),
                                          "fixed_threshold_sens_ci95": fx["strategies"][nm]["sensitivity"]["ci95"], "fixed_threshold_spec_ci95": fx["strategies"][nm]["specificity"]["ci95"],
                                          "point_sensitivity": fx["strategies"][nm]["sensitivity"]["value"], "point_specificity": fx["strategies"][nm]["specificity"]["value"]}
    dsB = col["sens_T"] - col["sens_B"]; dspA = col["spec_T"] - col["spec_A"]; fv = fx["strategies"]["triage_LATE"]
    res["d_sens_vs_B"] = {"delta": fv["vs_B"]["sens"]["delta"], "ci95": pct(dsB), "one_sided_95_lower": r3(np.percentile(dsB, 5)), "criterion_holds": bool(np.percentile(dsB, 5) > -0.05),
                          "fixed_threshold_ci95": fv["vs_B"]["sens"]["ci95"], "fixed_threshold_one_sided_95_lower": fv["vs_B"]["sens"]["one_sided_95_lower"]}
    res["d_spec_vs_A"] = {"delta": fv["vs_A"]["spec"]["delta"], "ci95": pct(dspA), "fixed_threshold_ci95": fv["vs_A"]["spec"]["ci95"]}
    res["share_sequenced"] = {"point": fx["bands"]["mid"]["all"]["value"], "ci95": pct(col["share_mid"]), "fixed_threshold_ci95": fx["bands"]["mid"]["all"]["ci95"]}
    json.dump(res, open(f"{OUT}/selci_{TASK}.json", "w"), indent=1); print("TB SELCI MERGE DONE", TASK, flush=True); raise SystemExit(0)
SH = int(os.environ["SHARD"]); outf = f"{D}/{TASK}_shard_{SH:02d}.npy"
if os.path.exists(outf): raise SystemExit(0)
O, I, FO = load(src); boots = boots_for(rows); out = []
for b in boots[SH * 200:(SH + 1) * 200]:
    W = np.bincount(b, minlength=n)   # sample multiplicity = patient multiplicity (patients bring all their population samples)
    band, call, _ = triage(O, I, FO, popm, W=W); comp, _ = comparators(O, I, FO, popm, W=W); CT = vote(call)
    v = [*sens_spec(CT, b), *sens_spec(vote(comp["A"]), b), *sens_spec(vote(comp["B"]), b), *sens_spec(vote(comp["C"]), b), (band[b] == 1).mean(0).mean()]; out.append(v)
np.save(outf, np.array(out)); print("TB SELCI SHARD DONE", TASK, SH, flush=True)

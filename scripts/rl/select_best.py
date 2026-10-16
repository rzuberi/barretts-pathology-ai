"""Pick a run's best checkpoint on VALIDATION only. argv: project(p1|p3) run. Prints the checkpoint dir name and writes selected_<run>.json."""
import glob, json, os, sys
proj, run = sys.argv[1:3]
if proj == "p1":
    P = os.environ.get("SEL_DIR", "/mnt/scratche/slow/fmlab/zuberi01/analysis/rl_grader"); sc = {}
    for f in glob.glob(f"{P}/preds/{run}__checkpoint-*__val.jsonl"):
        r = [json.loads(l) for l in open(f)]; sc[f.split("__")[1]] = sum(x["pred"] == x["label"] for x in r) / len(r)
elif proj == "p2":   # area under the selective-accuracy curve on validation (jury label), ordering = answered by grade prob, UNSURE last
    import numpy as np
    P = os.environ.get("SEL_DIR", "/mnt/scratche/slow/fmlab/zuberi01/analysis/rl_abstain"); sc = {}
    for f in glob.glob(f"{P}/preds/{run}__checkpoint-*__val.jsonl"):
        r = [json.loads(l) for l in open(f)]
        conf = np.array([-1.0 if x["pred"] in (None, "UNSURE") else (x["conf"] if x.get("conf") is not None else (0.0 if "conf" in x else (x.get("p_grade") or 0))) for x in r]); ok = np.array([x["pred"] == x["label"] for x in r], float)
        o = np.argsort(-conf); cum = np.cumsum(ok[o]) / np.arange(1, len(o) + 1); sc[f.split("__")[1]] = float(cum.mean())
else:
    P = "/mnt/scratche/slow/fmlab/zuberi01/analysis/rl_sql"; sc = {}
    for f in glob.glob(f"{P}/preds/{run}__checkpoint-*__val.jsonl"):
        r = [json.loads(l) for l in open(f)]; sc[f.split("__")[1]] = sum(x["correct"] for x in r) / len(r)
best = max(sorted(sc), key=lambda k: (sc[k], int(k.split("-")[1])))
json.dump({"run": run, "val_scores": sc, "selected": best, "rule": "highest validation accuracy (p1: exact vs jury; p3: overall correctness incl. refusals); ties -> later step"}, open(f"{P}/selected_{run}.json", "w"), indent=1)
print(best)

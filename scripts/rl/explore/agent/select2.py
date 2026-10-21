"""Pick a run's best checkpoint on VALIDATION only: mean over the 7 validation templates of per-template correctness (accuracy for
answerable, correct-refusal rate for unanswerable); ties -> later step. argv: preds_dir prefix. Writes selected_<prefix>.json."""
import collections, glob, json, os, sys
P, run = sys.argv[1:3]; sc = {}
for f in glob.glob(f"{P}/{run}__checkpoint-*__val.jsonl"):
    d = collections.defaultdict(list)
    for l in open(f): r = json.loads(l); d[r["template"]].append(r["correct"])
    sc[f.split("__")[-2]] = sum(sum(v) / len(v) for v in d.values()) / len(d)
best = max(sorted(sc), key=lambda k: (sc[k], int(k.split("-")[1])))
json.dump({"prefix": run, "set": "val", "rule": "mean per-template correctness over the 7 validation templates; ties -> later step", "scores": sc, "selected": best}, open(f"{P}/selected_{run}.json", "w"), indent=1); print(best)

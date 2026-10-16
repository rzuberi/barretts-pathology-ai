"""Pick a checkpoint on VALIDATION only. argv: kind(acc|sql|pert) preds_dir tag_prefix set. Prints the best checkpoint name."""
import glob, json, os, sys
kind, pdir, pre, s = sys.argv[1:5]; sc = {}
for f in glob.glob(f"{pdir}/{pre}__*__{s}.jsonl"):
    ck = os.path.basename(f)[len(pre) + 2:-len(f"__{s}.jsonl")]; r = [json.loads(l) for l in open(f)]
    if kind == "acc": v = sum(x["pred"] == x["label"] for x in r) / len(r)
    elif kind == "sql": v = sum(x["correct"] for x in r) / len(r)
    else: v = sum(max(("up", "down", "none"), key=lambda w: x["p_" + w]) == x["label"] for x in r) / len(r)
    sc[ck] = v
best = max(sorted(sc), key=lambda k: (sc[k], int(k.split("-")[1]) if "-" in k else 10**9))
json.dump({"prefix": pre, "set": s, "scores": sc, "selected": best}, open(f"{pdir}/selected_{pre}.json", "w"), indent=1); print(best)

"""Paper C item 5: fixed subsample of test perturbations (one in four per cell line, seed 0) for the thinking-LLM runs.
Writes $X/data/test_sub.jsonl (same rows and 'ret' sentence as test.jsonl, perturbation-filtered) and $O/test_sub_perts.json."""
import json, random
B = "/mnt/scratche/slow/fmlab/zuberi01"; X = B + "/analysis/rl_explore/pert"; O = B + "/analysis/paperC"
rows = [json.loads(l) for l in open(X + "/data/test.jsonl")]
perts = {}
for r in rows: perts.setdefault(r["cell"], set()).add(r["pert"])
rng = random.Random(0); keep = {c: sorted(rng.sample(sorted(p), len(p) // 4)) for c, p in perts.items()}
sub = [r for r in rows if r["pert"] in set(keep[r["cell"]])]
with open(X + "/data/test_sub.jsonl", "w") as f:
    for r in sub: f.write(json.dumps(r) + "\n")
json.dump(keep, open(O + "/test_sub_perts.json", "w"), indent=1)
print({c: (len(keep[c]), len(perts[c])) for c in keep}, "pairs", len(sub), "of", len(rows))

"""Task E re-split: >= 20 answerable test templates (moved from training), seed 20261016. Out: rl_sql/questions_agent.jsonl + templates_agent.json"""
import json, random
O = "/mnt/scratche/slow/fmlab/zuberi01/analysis/rl_sql"
qs = [json.loads(l) for l in open(O + "/questions.jsonl")]
ans = sorted({q["template"] for q in qs if q["answerable"]}); una = sorted({q["template"] for q in qs if not q["answerable"]})
r = random.Random(20261016); r.shuffle(ans); r.shuffle(una)
split = {**{t: "test" for t in ans[:20]}, **{t: "val" for t in ans[20:24]}, **{t: "train" for t in ans[24:]},
         **{t: "test" for t in una[:8]}, **{t: "val" for t in una[8:11]}, **{t: "train" for t in una[11:]}}
for q in qs: q["split"] = split[q["template"]]
with open(O + "/questions_agent.jsonl", "w") as f:
    for q in qs: f.write(json.dumps(q) + "\n")
from collections import Counter
c = Counter((q["split"], q["answerable"]) for q in qs); t = Counter((split[k], k in ans) for k in split)
json.dump({"split": split, "templates_by_split": {f"{a}|{'ans' if b else 'unans'}": n for (a, b), n in t.items()}, "questions": {f"{a}|{'ans' if b else 'unans'}": n for (a, b), n in c.items()}}, open(O + "/templates_agent.json", "w"), indent=1)
print(dict(t), dict(c))

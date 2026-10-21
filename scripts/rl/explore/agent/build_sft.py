"""Explore/agent, expert iteration step 2 (CPU): keep correct episodes (correct final answer, or a correct refusal), drop duplicates
(same assistant turns), keep at most CAP per answerable question (fewest tool calls first) and CAP_U per unanswerable question.
argv: traj.jsonl out.jsonl [CAP=2] [CAP_U=1]"""
import json, sys, collections, random
src, dst = sys.argv[1:3]; CAP = int(sys.argv[3]) if len(sys.argv) > 3 else 2; CAPU = int(sys.argv[4]) if len(sys.argv) > 4 else 1
eps = [json.loads(l) for l in open(src)]; by = collections.defaultdict(list); seen = set()
for e in eps:
    if not e["correct"]: continue
    key = (e["question"], tuple(m["content"] for m in e["messages"] if m["role"] == "assistant"))
    if key in seen: continue
    seen.add(key); by[e["question"]].append(e)
rnd = random.Random(0); rows = []
for q, lst in by.items():
    rnd.shuffle(lst); lst.sort(key=lambda e: e["n_calls"]); rows += lst[:(CAP if lst[0]["answerable"] else CAPU)]
rnd.shuffle(rows)
with open(dst, "w") as f:
    for r in rows: f.write(json.dumps(r) + "\n")
import os; os.chmod(dst, 0o600)
qa = {e["question"] for e in eps if e["answerable"]}; qu = {e["question"] for e in eps if not e["answerable"]}
ka = {q for q in by if by[q][0]["answerable"]}; ku = {q for q in by if not by[q][0]["answerable"]}
tpl = collections.Counter(r["template"] for r in rows)
print(json.dumps({"episodes": len(eps), "correct_episodes": sum(e["correct"] for e in eps), "answerable_questions": len(qa), "solved_answerable": len(ka), "unanswerable_questions": len(qu), "refused_unanswerable": len(ku),
                  "kept_trajectories": len(rows), "kept_answerable": sum(r["answerable"] for r in rows), "mean_calls_kept": round(sum(r["n_calls"] for r in rows) / max(len(rows), 1), 2), "per_template": dict(tpl)}, indent=1))

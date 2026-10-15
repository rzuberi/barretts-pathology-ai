"""Build the task queue (JSONL) for the ERIN imminent task set."""
import json, os
Q = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis/feasibility/erin_fusion/queue"; os.makedirs(Q + "/claims", exist_ok=True); os.makedirs(Q + "/results", exist_ok=True)
TEXT_TASKS = ["T1", "T2a", "T2b", "T2c"]; ALL = ["T1", "T2a", "T2b", "T2c", "T3a", "T3b", "T4"]
tasks = [{"id": "embed", "type": "embed", "gpu": False}]
for t in ALL:
    tasks.append({"id": f"tab_{t}", "type": "tab", "task": t, "gpu": False, "needs": ["embed"] if t in TEXT_TASKS else []})
    for f in range(5):
        for s in range(3): tasks.append({"id": f"img_{t}_f{f}_s{s}", "type": "img", "task": t, "fold": f, "seed": s, "gpu": True})
for p in range(1, 51):
    for f in range(5): tasks.append({"id": f"perm_T2a_p{p}_f{f}", "type": "img", "task": "T2a", "fold": f, "seed": 0, "perm": p, "gpu": True})
tasks.append({"id": "tabperm_T2a", "type": "tabperm", "task": "T2a", "gpu": False, "needs": ["embed"]})
# order: cheap+critical first so early numbers appear
prio = {"embed": 0, "tab": 1, "img": 2, "tabperm": 3}
tasks.sort(key=lambda t: (prio[t["type"]], 1 if t.get("perm") else 0, t.get("task") != "T2a", t.get("fold", 0), t.get("seed", 0)))
with open(Q + "/tasks.jsonl", "w") as fh:
    for t in tasks: fh.write(json.dumps(t) + "\n")
print(len(tasks), "tasks;", sum(t["gpu"] for t in tasks), "gpu")

"""Project 1 report (CPU): metrics for all models on the 479 human sections (+ val/test), jury cost per report, figures, results.json.
Models: zs, sft final, grpo_base and grpo_sft at their validation-selected checkpoints, MedGemma-27B zero-shot; the jury is added by metrics.py."""
import glob, json, os, subprocess, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__)); from common import *
R = B + "/analysis/rl_grader"; T = B + "/phd/thesis"
sel = {r: json.load(open(f"{R}/selected_{r}.json"))["selected"] for r in ("grpo_base", "grpo_sft") if os.path.exists(f"{R}/selected_{r}.json")}
tags = ["zs", "sft__final"] + [f"{r}__{c}" for r, c in sel.items()] + ["medgemma27b_zs"]
env = dict(os.environ, TAGS=",".join(tags), REF="zs", OUT=R + "/metrics.json")
print(subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "metrics.py")], env=env, capture_output=True, text=True).stdout)
M = json.load(open(R + "/metrics.json"))
# jury cost: sum over the 10 jurors of seconds per report (v3 outputs, 'secs' column), for the 143 human-labelled reports
reps = {json.loads(l)["report_id"] for l in open(R + "/data/test485.jsonl")}; secs = {}
for f in glob.glob(T + "/feasibility/runs/sec3full_*/output/sections_v3_*.csv"):
    d = pd.read_csv(f, dtype={"CaseName": str}, usecols=["CaseName", "secs"]); d = d[d.CaseName.isin(reps)]
    for c, s in zip(d.CaseName, d.secs): secs[c] = secs.get(c, 0) + float(s or 0)
jury_spr = float(np.median(list(secs.values()))) if secs else None
names = {"jury": "10-model jury", "zs": "MedGemma-4B zero-shot", "sft__final": "MedGemma-4B SFT", "medgemma27b_zs": "MedGemma-27B zero-shot"}
for r, c in sel.items(): names[f"{r}__{c}"] = f"MedGemma-4B GRPO ({'from base' if r == 'grpo_base' else 'from SFT'}, {c})"
out = {"selected_checkpoints": sel, "val_selection": {r: json.load(open(f"{R}/selected_{r}.json"))["val_scores"] for r in sel},
       "jury_seconds_per_report_median_sum_of_10": jury_spr, "metrics": M, "names": names}
json.dump(out, open(R + "/results.json", "w"), indent=1)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
order = ["zs", "sft__final"] + [f"{r}__{c}" for r, c in sel.items()] + ["medgemma27b_zs", "jury"]
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for k, (m, title) in enumerate((("exact_w", "exact grade agreement"), ("dysp_w", "dysplasia yes/no agreement"))):
    v = [M["test485"][t][m] for t in order]; ci = [M["test485"][t][m + "_ci"] for t in order]
    ax[k].barh(range(len(order)), v, xerr=[[a - c[0] for a, c in zip(v, ci)], [c[1] - a for a, c in zip(v, ci)]], color=["#9ca3af", "#2563eb", "#dc2626", "#f59e0b", "#7c3aed", "#059669"][:len(order)], capsize=3)
    ax[k].set_yticks(range(len(order))); ax[k].set_yticklabels([names[t] for t in order] if k == 0 else [], fontsize=8); ax[k].set_xlim(0.3 if m == "exact_w" else 0.7, 1.0)
    ax[k].set_title(f"{title}, 479 human-labelled sections (weighted)", fontsize=9)
plt.tight_layout(); plt.savefig(R + "/agreement.png", dpi=160); plt.close()
fig, ax = plt.subplots(figsize=(6, 3.6))
for r in sel:
    lg = json.load(open(f"{R}/runs/{r}/train_summary.json"))["log"]; st = [x["step"] for x in lg if "reward" in x]; rw = [x["reward"] for x in lg if "reward" in x]
    ax.plot(st, rw, label=names[f"{r}__{sel[r]}"].split(" (")[1].split(",")[0])
ax.set_xlabel("GRPO step"); ax.set_ylabel("mean reward (training batch)"); ax.legend(fontsize=8); ax.set_title("GRPO training reward", fontsize=9)
plt.tight_layout(); plt.savefig(R + "/grpo_reward.png", dpi=160); plt.close()
print("jury s/report (median, sum of 10 jurors):", jury_spr)

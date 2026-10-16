import json, glob, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
A = "/mnt/scratche/slow/fmlab/zuberi01/analysis/rl_abstain"; M = json.load(open(A + "/metrics.json"))
lab = {"jury_votefrac": "10-model jury, vote-fraction triage", "p1_tokenprob": "RL model, grade-token probability", "p1_agree8": "RL model, agreement of 8 samples",
       "p1_random": "RL model, random review", "alpha0.2__checkpoint-150": "abstention RL α=0.2 (selected)", "alpha0.4__checkpoint-50": "abstention RL α=0.4", "alpha0.6__checkpoint-100": "abstention RL α=0.6"}
fig, ax = plt.subplots(figsize=(7, 4.3))
for m, c in M["curves"].items(): ax.plot([100 * x for x in c["frac_reports"]], [100 * y for y in c["acc"]], label=lab.get(m, m), lw=2 if m in ("jury_votefrac", "alpha0.2__checkpoint-150") else 1)
ax.axvline(17, color="k", ls="--", lw=0.8); ax.set_xlabel("% of reports sent to a human"); ax.set_ylabel("estimated population exact accuracy (%)")
ax.set_xlim(0, 60); ax.set_ylim(88, 100.5); ax.legend(fontsize=7, loc="lower right"); ax.set_title("Human review budget vs label accuracy (479 human-labelled sections, weighted)", fontsize=9)
plt.tight_layout(); plt.savefig(A + "/triage_comparison.png", dpi=160); plt.close()
for a in ("0.2", "0.4", "0.6"):
    s = json.load(open(f"{A}/selected_alpha{a}.json")); lg = json.load(open(f"{A}/runs/alpha{a}/train_summary.json"))["log"]
    rw = [x["reward"] for x in lg if "reward" in x]; ln = [x.get("completions/mean_length") for x in lg if "reward" in x]
    v = [json.loads(l) for l in open(f"{A}/preds/alpha{a}__{s['selected']}__val.jsonl")]
    print(a, "selected", s["selected"], "val AUSC", round(s["val_scores"][s["selected"]], 4), "| reward first/last", round(rw[0], 3), round(rw[-1], 3), "| len first/last", round(ln[0], 1), round(ln[-1], 1),
          "| val UNSURE", sum(x["pred"] == "UNSURE" for x in v), "/", len(v), "| val acc", round(sum(x["pred"] == x["label"] for x in v) / len(v), 4))
c = [json.loads(l) for l in open(A + "/preds/p1best_conf__val.jsonl")]; print("p1 model val acc vs jury", round(sum(x["pred"] == x["label"] for x in c) / len(c), 4))
print({m: (r["acc@17%reports"], r["acc@17%reports_ci"]) for m, r in M["methods"].items()})

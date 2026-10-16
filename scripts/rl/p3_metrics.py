"""Project 3 metrics (CPU): execution accuracy on held-out TEST templates (answerable), refusal rate (unanswerable), by question type;
bootstrap over templates (2,000) for every CI and for paired differences (RL - SFT primary). Writes rl_sql/results.json + figure."""
import json, os
import numpy as np
O = "/mnt/scratche/slow/fmlab/zuberi01/analysis/rl_sql"; NB = 2000; RNG = np.random.RandomState(20261016)
sel = {"sft": json.load(open(O + "/selected_sft.json"))["selected"] if os.path.exists(O + "/selected_sft.json") else "checkpoint-150",
       "grpo": json.load(open(O + "/selected_grpo.json"))["selected"]}
TAGS = {"zero-shot": "zs", "5-shot": "fs5", "SFT": f"sft__{sel['sft']}", "GRPO": f"grpo__{sel['grpo']}"}
def load(tag, s): return [json.loads(l) for l in open(f"{O}/preds/{tag}__{s}.jsonl")]
R = {"selected_checkpoints": sel, "methods": {}, "by_type": {}}
for split in ("val", "test"):
    D = {m: load(t, split) for m, t in TAGS.items()}
    tmpl = sorted({r["template"] for r in D["SFT"]}); tix = {t: i for i, t in enumerate(tmpl)}
    BS = [RNG.choice(len(tmpl), len(tmpl)) for _ in range(NB)]
    def acc(rows, answerable, ts=None):
        rr = [r for r in rows if r["answerable"] == answerable and (ts is None or r["template"] in ts)]; return np.mean([r["correct"] for r in rr]) if rr else np.nan
    def boot(rows, answerable):
        per = {}
        for r in rows:
            if r["answerable"] == answerable: per.setdefault(r["template"], []).append(r["correct"])
        keys = sorted(per); vals = []
        for b in range(NB):
            pick = RNG.choice(len(keys), len(keys)); v = [x for k in pick for x in per[keys[k]]]; vals.append(np.mean(v))
        return [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)]
    for m, rows in D.items():
        a = [r for r in rows if r["answerable"]]; u = [r for r in rows if not r["answerable"]]
        R["methods"].setdefault(m, {})[split] = {"n_answerable": len(a), "n_unanswerable": len(u), "n_templates": len({r["template"] for r in rows}),
            "exec_acc": round(float(acc(rows, True)), 4), "exec_acc_ci": boot(rows, True), "refusal_rate": round(float(acc(rows, False)), 4) if u else None,
            "sql_error_rate": round(float(np.mean([r["kind"] == "sql_error" for r in a])), 4), "false_refusal_rate": round(float(np.mean([r["kind"] == "refused_or_empty" for r in a])), 4)}
    # paired template bootstrap: GRPO - SFT on answerable exec accuracy
    per = {}
    for m in ("GRPO", "SFT", "zero-shot"):
        for r in D[m]:
            if r["answerable"]: per.setdefault(r["template"], {}).setdefault(m, []).append(r["correct"])
    keys = sorted(per)
    def diff(a, b, pick): xa = [x for k in pick for x in per[keys[k]][a]]; xb = [x for k in pick for x in per[keys[k]][b]]; return np.mean(xa) - np.mean(xb)
    allk = range(len(keys))
    for a, b in (("GRPO", "SFT"), ("GRPO", "zero-shot"), ("SFT", "zero-shot")):
        d = [diff(a, b, RNG.choice(len(keys), len(keys))) for _ in range(NB)]
        R.setdefault("paired", {}).setdefault(split, {})[f"{a} - {b}"] = {"point": round(float(diff(a, b, allk)), 4), "ci": [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)]}
    if split == "test":
        types = sorted({r["type"] for r in D["SFT"] if r["answerable"]})
        for m, rows in D.items(): R["by_type"][m] = {t: round(float(np.mean([r["correct"] for r in rows if r["answerable"] and r["type"] == t])), 3) for t in types}
        R["by_type_n"] = {t: sum(1 for r in D["SFT"] if r["answerable"] and r["type"] == t) for t in types}
        R["test_templates"] = sorted({r["template"] for r in D["SFT"]})
json.dump(R, open(O + "/results.json", "w"), indent=1)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8)); ms = list(TAGS)
for k, (key, title) in enumerate((("exec_acc", "execution accuracy, answerable"), ("refusal_rate", "correct refusals, unanswerable"))):
    v = [R["methods"][m]["test"][key] or 0 for m in ms]; ax[k].bar(ms, v, color=["#9ca3af", "#7c3aed", "#2563eb", "#dc2626"])
    if key == "exec_acc":
        lo = [v[i] - R["methods"][m]["test"]["exec_acc_ci"][0] for i, m in enumerate(ms)]; hi = [R["methods"][m]["test"]["exec_acc_ci"][1] - v[i] for i, m in enumerate(ms)]; ax[k].errorbar(ms, v, yerr=[lo, hi], fmt="none", ecolor="k", capsize=3)
    ax[k].set_ylim(0, 1); ax[k].set_title(f"held-out test templates: {title}", fontsize=9)
plt.tight_layout(); plt.savefig(O + "/test_accuracy.png", dpi=160); plt.close()
print(json.dumps({m: R["methods"][m]["test"] for m in ms}, indent=0)); print(R["paired"]["test"]); print(R["by_type"], R["by_type_n"])

"""Follow-up E report: (a) project-3-recipe SFT single-shot, (b) same SFT model as a tool-using agent, (c) agent after GRPO, on the
re-split held-out test (20 answerable + 8 unanswerable templates). Execution accuracy (answerable), correct-refusal rate (unanswerable),
false refusals, mean tool calls, accuracy by type; 2,000-resample bootstrap over templates, paired. Out: rl_sql/agent_results.json"""
import json, os
import numpy as np
O = "/mnt/scratche/slow/fmlab/zuberi01/analysis/rl_sql"; RNG = np.random.RandomState(20261016); NB = 2000
def sel(d, p): return json.load(open(f"{O}/{d}/selected_{p}.json"))["selected"]
F = {"a_sft_single_shot": f"{O}/agsplit_preds/agsplit_sft__{sel('agsplit_preds', 'agsplit_sft')}__test.jsonl",
     "b_sft_agent": f"{O}/agent_preds/agent_untrained__test.jsonl",
     "c_grpo_agent": f"{O}/agent_preds/agent_grpo__{sel('agent_preds', 'agent_grpo')}__test.jsonl"}
key = lambda r: (r["template"], r["variant"], r.get("question_id", r.get("qid", "")))
P = {m: [json.loads(l) for l in open(f)] for m, f in F.items()}
for m, rows in P.items(): rows.sort(key=lambda r: (r["template"], r["variant"]))
n0 = len(P["a_sft_single_shot"]); assert all(len(v) == n0 for v in P.values()), {m: len(v) for m, v in P.items()}
ref = P["a_sft_single_shot"]; ans = np.array([bool(r["answerable"]) for r in ref]); tpl = np.array([r["template"] for r in ref]); typ = np.array([r["type"] for r in ref])
for m in P: assert [r["template"] for r in P[m]] == list(tpl), m
C = {m: np.array([bool(r["correct"]) for r in P[m]]) for m in P}; K = {m: np.array([r["kind"] for r in P[m]]) for m in P}
calls = {m: np.array([r.get("n_calls", 0) for r in P[m]], float) for m in P}
ta, tu = np.unique(tpl[ans]), np.unique(tpl[~ans]); gi = {t: np.where(tpl == t)[0] for t in np.unique(tpl)}
BA = [np.concatenate([gi[t] for t in RNG.choice(ta, len(ta))]) for _ in range(NB)]; BU = [np.concatenate([gi[t] for t in RNG.choice(tu, len(tu))]) for _ in range(NB)]
ci = lambda v: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
IA, IU = np.where(ans)[0], np.where(~ans)[0]
res = {"files": F, "n_questions": n0, "n_answerable": int(ans.sum()), "n_unanswerable": int((~ans).sum()), "answerable_templates": len(ta), "unanswerable_templates": len(tu), "models": {}}
for m in P:
    c = C[m]; res["models"][m] = {"exec_acc": round(float(c[IA].mean()), 4), "exec_acc_ci": ci([c[ix].mean() for ix in BA]),
        "correct_refusal": round(float(c[IU].mean()), 4), "correct_refusal_ci": ci([c[ix].mean() for ix in BU]),
        "false_refusal_on_answerable": round(float((K[m][IA] == "refused").mean()), 4), "no_final_answer": round(float((K[m] == "no_final").mean()), 4),
        "mean_tool_calls": round(float(calls[m].mean()), 2), "mean_tool_calls_answerable": round(float(calls[m][IA].mean()), 2),
        "by_type": {t: {"n": int(((typ == t) & ans).sum()), "acc": round(float(c[(typ == t) & ans].mean()), 4)} for t in sorted(set(typ[ans]))},
        "by_template": {t: round(float(c[gi[t]].mean()), 3) for t in sorted(gi)}}
for a, b in (("b_sft_agent", "a_sft_single_shot"), ("c_grpo_agent", "b_sft_agent"), ("c_grpo_agent", "a_sft_single_shot")):
    d = C[a].astype(float) - C[b].astype(float)
    res[f"{a}_minus_{b}"] = {"exec_acc": round(float(d[IA].mean()), 4), "exec_acc_ci": ci([d[ix].mean() for ix in BA]),
                             "correct_refusal": round(float(d[IU].mean()), 4), "correct_refusal_ci": ci([d[ix].mean() for ix in BU])}
json.dump(res, open(O + "/agent_results.json", "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k not in ("models",)}, indent=1)[:2500])
for m, v in res["models"].items(): print(m, {k: v[k] for k in ("exec_acc", "exec_acc_ci", "correct_refusal", "false_refusal_on_answerable", "no_final_answer", "mean_tool_calls")})

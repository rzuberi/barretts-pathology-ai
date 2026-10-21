"""Explore/agent report: e_report-style metrics for every method in METHODS (name -> test preds file), paired 2,000-resample
bootstrap over templates vs the two baselines (single-shot SFT = agsplit final; Task-E GRPO agent = checkpoint-150).
Question-level and per-template (primary) views. argv: out.json; env METHODS (json)."""
import collections, json, os, sys
import numpy as np
O = "/mnt/scratche/slow/fmlab/zuberi01/analysis/rl_sql"; RNG = np.random.RandomState(20261016); NB = 2000
F = {"a_sft_single_shot": f"{O}/agsplit_preds/agsplit_sft__final__test.jsonl", "b_sft_agent": f"{O}/agent_preds/agent_untrained__test.jsonl", "c_grpo_agent_taskE": f"{O}/agent_preds/agent_grpo__checkpoint-150__test.jsonl"}
F.update(json.loads(os.environ["METHODS"]))
P = {m: sorted([json.loads(l) for l in open(f)], key=lambda r: (r["template"], r["variant"])) for m, f in F.items()}
ref = P["a_sft_single_shot"]; n0 = len(ref); assert all(len(v) == n0 for v in P.values()), {m: len(v) for m, v in P.items()}
ans = np.array([bool(r["answerable"]) for r in ref]); tpl = np.array([r["template"] for r in ref]); typ = np.array([r["type"] for r in ref])
for m in P: assert [r["template"] for r in P[m]] == list(tpl), m
C = {m: np.array([bool(r["correct"]) for r in P[m]]) for m in P}; K = {m: np.array([r["kind"] for r in P[m]]) for m in P}; calls = {m: np.array([r.get("n_calls", 0) for r in P[m]], float) for m in P}
ta, tu = np.unique(tpl[ans]), np.unique(tpl[~ans]); gi = {t: np.where(tpl == t)[0] for t in np.unique(tpl)}; IA, IU = np.where(ans)[0], np.where(~ans)[0]
BA = [RNG.choice(ta, len(ta)) for _ in range(NB)]; BU = [RNG.choice(tu, len(tu)) for _ in range(NB)]
ci = lambda v: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
def tmean(c, ts): return float(np.mean([c[gi[t]].mean() for t in ts]))
def qmean(c, ts): ix = np.concatenate([gi[t] for t in ts]); return float(c[ix].mean())
res = {"files": F, "n_questions": n0, "n_answerable": int(ans.sum()), "n_unanswerable": int((~ans).sum()), "answerable_templates": len(ta), "unanswerable_templates": len(tu), "models": {}, "diffs": {}}
for m in P:
    c = C[m]; res["models"][m] = {"exec_acc": round(float(c[IA].mean()), 4), "exec_acc_ci": ci([qmean(c, b) for b in BA]), "tpl_acc": round(tmean(c, ta), 4), "tpl_acc_ci": ci([tmean(c, b) for b in BA]),
        "correct_refusal": round(float(c[IU].mean()), 4), "correct_refusal_ci": ci([qmean(c, b) for b in BU]), "tpl_refusal": round(tmean(c, tu), 4),
        "false_refusal_on_answerable": round(float((K[m][IA] == "refused").mean()), 4), "tpl_false_refusal": round(float(np.mean([(K[m][gi[t]] == "refused").mean() for t in ta])), 4),
        "no_final_answer": round(float((K[m] == "no_final").mean()), 4), "sql_error": round(float((K[m][IA] == "sql_error").mean()), 4), "mean_tool_calls": round(float(calls[m].mean()), 2), "mean_tool_calls_answerable": round(float(calls[m][IA].mean()), 2),
        "by_type": {t: {"n": int(((typ == t) & ans).sum()), "templates": int(len(set(tpl[(typ == t) & ans]))), "acc": round(float(c[(typ == t) & ans].mean()), 4)} for t in sorted(set(typ[ans]))},
        "non_per_year_acc": round(float(c[IA][typ[IA] != "per_year"].mean()), 4), "by_template": {t: round(float(c[gi[t]].mean()), 3) for t in sorted(gi)}}
for a in P:
    for b in ("a_sft_single_shot", "c_grpo_agent_taskE"):
        if a == b: continue
        d = C[a].astype(float) - C[b].astype(float); tb = [C[a][gi[t]].mean() - C[b][gi[t]].mean() for t in ta]
        res["diffs"][f"{a}_minus_{b}"] = {"exec_acc": round(float(d[IA].mean()), 4), "exec_acc_ci": ci([qmean(d, s) for s in BA]), "tpl_acc": round(float(np.mean(tb)), 4), "tpl_acc_ci": ci([tmean(d, s) for s in BA]),
            "templates_better": int(sum(x > 0 for x in tb)), "templates_worse": int(sum(x < 0 for x in tb)), "correct_refusal": round(float(d[IU].mean()), 4), "correct_refusal_ci": ci([qmean(d, s) for s in BU])}
json.dump(res, open(sys.argv[1], "w"), indent=1); os.chmod(sys.argv[1], 0o600)
for m, v in res["models"].items(): print(m, {k: v[k] for k in ("exec_acc", "exec_acc_ci", "tpl_acc", "tpl_acc_ci", "correct_refusal", "false_refusal_on_answerable", "non_per_year_acc", "mean_tool_calls")})
for k, v in res["diffs"].items(): print(k, v)

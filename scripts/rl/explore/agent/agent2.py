"""Explore/agent: Task-E agent with (i) the schema doc in the system prompt and (ii) a shaped reward. Copy of sql_agent.py's
run_episodes/score_final; everything else is imported. Env: SCHEMA_PROMPT (1), SHAPED (1)."""
import json, os, sys
RL = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, RL)
from sql_common import run, norm, O, DB, SCHEMA
from sql_agent import TABLES, describe, parse_turn, tool_result, MAX_CALLS, SYS_AGENT as SYS_AGENT0
O2 = "/mnt/scratche/slow/fmlab/zuberi01/analysis/rl_explore/agent"
SYS_AGENT = SYS_AGENT0 + ("\n\nDatabase schema (so you need not call describe unless you want to check values):\n" + SCHEMA.strip() if os.environ.get("SCHEMA_PROMPT", "1") == "1" else "")
SHAPED = os.environ.get("SHAPED", "1") == "1"
def score_final(final, item, n_probe_ok=0):
    """final: ('sql', query) | ('refuse', None) | (None, None). Returns (reward, correct, kind). Shaped: refusing an answerable
    question costs -0.5, an executed-but-wrong final query earns 0.1, and any successfully executed run_sql probe adds 0.05 (answerable only)."""
    kind, q = final
    if not item["answerable"]:
        return (1.0, True, "refused") if kind == "refuse" else ((-0.5, False, "answered_unanswerable") if kind == "sql" else (0.0, False, "no_final"))
    bonus = 0.05 if (SHAPED and n_probe_ok > 0) else 0.0
    if kind == "refuse": return (-0.5 if SHAPED else 0.0) + bonus, False, "refused"
    if kind != "sql": return 0.0 + bonus, False, "no_final"
    rows, ok = run(q)
    if not ok: return 0.0 + bonus, False, "sql_error"
    if norm(rows) == norm(item["answer"]): return 1.0 + bonus, True, "correct"
    return (0.1 if SHAPED else 0.0) + bonus, False, "wrong_result"
def run_episodes(llm, items, sp, lora=None, chat_kw=None):
    """Turn-synchronous episodes. Returns dicts: messages, n_calls, n_probe_ok, final, reward, correct, kind, item."""
    eps = [{"messages": [{"role": "system", "content": SYS_AGENT}, {"role": "user", "content": it["question"]}], "n_calls": 0, "n_probe_ok": 0, "final": (None, None), "done": False, "item": it} for it in items]
    for turn in range(MAX_CALLS + 1):
        live = [e for e in eps if not e["done"]]
        if not live: break
        outs = llm.chat([e["messages"] for e in live], sp, use_tqdm=False, lora_request=lora, **(chat_kw or {}))
        for e, o in zip(live, outs):
            txt = o.outputs[0].text; e["messages"].append({"role": "assistant", "content": txt}); d = parse_turn(txt)
            if d is not None and "final_sql" in d: e["final"] = ("sql", str(d["final_sql"])); e["done"] = True
            elif d is not None and str(d.get("final", "")).upper().startswith("CANNOT"): e["final"] = ("refuse", None); e["done"] = True
            elif turn == MAX_CALLS: e["done"] = True
            else:
                e["n_calls"] += 1; res = tool_result(d) if d else "error: reply with one JSON object"
                if d and d.get("tool") == "run_sql" and not res.startswith("error"): e["n_probe_ok"] += 1
                e["messages"].append({"role": "user", "content": "TOOL RESULT:\n" + res})
    for e in eps:
        r, ok, kind = score_final(e["final"], e["item"], e["n_probe_ok"]); e.update(reward=r, correct=ok, kind=kind)
    return eps

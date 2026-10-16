"""Task E agent: multi-turn tool use over erin_labels.sqlite. Tools: list_tables, describe(table), run_sql(query); <= 6 calls,
then a final SQL answer or a refusal. Used by sql_agent_eval.py (greedy) and sql_agent_grpo.py (sampled)."""
import json, os, re, sqlite3, sys
sys.path.insert(0, os.path.dirname(__file__)); from sql_common import run, norm, O, DB
MAX_CALLS = 6
SYS_AGENT = """You answer questions about a pathology label database (SQLite) by using tools. Reply each turn with exactly ONE JSON object and nothing else:
{"tool": "list_tables"}
{"tool": "describe", "table": "<table name>"}
{"tool": "run_sql", "query": "<SQLite SELECT query>"}
{"final_sql": "<SQLite SELECT query whose result answers the question>"}
{"final": "CANNOT_ANSWER"}   (when the database cannot answer the question, or it is too vague to answer)
You may use at most 6 tool calls before your final reply. Grades in order: NORMAL_OTHER < NDBE < IND < LGD < HGD < CANCER."""
_SCHEMA = open(O + "/SCHEMA.md").read() if os.path.exists(O + "/SCHEMA.md") else ""
def _tables():
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True); r = [x[0] for x in con.execute("select name from sqlite_master where type='table'")]; con.close(); return r
TABLES = _tables()
def describe(t):
    if t not in TABLES: return f"error: no table named {t!r}; tables are {', '.join(TABLES)}"
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True); ddl = con.execute("select sql from sqlite_master where name=?", (t,)).fetchone()[0]; con.close()
    note = next((l for l in _SCHEMA.splitlines() if l.startswith(f"- **{t}**")), "")
    return ddl + ("\n" + re.sub(r"\*\*", "", note) if note else "")
def parse_turn(text):
    t = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S); m = re.search(r"\{.*\}", t, re.S)
    if not m: return None
    try: return json.loads(m.group(0))
    except Exception: return None
def tool_result(d):
    if d.get("tool") == "list_tables": return ", ".join(TABLES)
    if d.get("tool") == "describe": return describe(str(d.get("table", "")))
    if d.get("tool") == "run_sql":
        rows, ok = run(str(d.get("query", "")))
        return ("error: query failed" if not ok else json.dumps(rows[:20]) + (f" ... ({len(rows)} rows)" if len(rows) > 20 else ""))
    return "error: unknown tool; reply with one JSON object as instructed"
def score_final(final, item):
    """final: ('sql', query) | ('refuse', None) | (None, None). Returns (reward, correct, kind)."""
    kind, q = final
    if not item["answerable"]:
        return (1.0, True, "refused") if kind == "refuse" else ((-0.5, False, "answered_unanswerable") if kind == "sql" else (0.0, False, "no_final"))
    if kind != "sql": return 0.0, False, ("refused" if kind == "refuse" else "no_final")
    rows, ok = run(q)
    if not ok: return 0.0, False, "sql_error"
    return (1.0, True, "correct") if norm(rows) == norm(item["answer"]) else (0.0, False, "wrong_result")
def run_episodes(llm, items, sp, lora=None, chat_kw=None):
    """Runs all episodes turn-synchronously. Returns list of dicts: messages, n_calls, final, reward, correct, kind."""
    eps = [{"messages": [{"role": "system", "content": SYS_AGENT}, {"role": "user", "content": it["question"]}], "n_calls": 0, "final": (None, None), "done": False, "item": it} for it in items]
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
                e["n_calls"] += 1; e["messages"].append({"role": "user", "content": "TOOL RESULT:\n" + (tool_result(d) if d else "error: reply with one JSON object")})
    for e in eps:
        r, ok, kind = score_final(e["final"], e["item"]); e.update(reward=r, correct=ok, kind=kind)
    return eps

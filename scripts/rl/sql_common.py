"""Project 3 shared: prompt, SQL extraction, sandboxed execution and answer comparison. No data in this file."""
import json, os, re, sqlite3, threading
O = "/mnt/scratche/slow/fmlab/zuberi01/analysis/rl_sql"; DB = O + "/erin_labels.sqlite"
SCHEMA = open(O + "/SCHEMA.md").read() if os.path.exists(O + "/SCHEMA.md") else ""
SYS = "You write SQLite queries over a pathology label database. Reply with one SQL query in a ```sql block, or with exactly CANNOT_ANSWER if the database cannot answer the question or the question is too vague."
def messages(q, shots=None):
    m = [{"role": "system", "content": SYS + "\n\n" + SCHEMA}]
    for s in shots or []:
        m += [{"role": "user", "content": s["question"]}, {"role": "assistant", "content": "CANNOT_ANSWER" if not s["answerable"] else f"```sql\n{s['gold_sql']}\n```"}]
    return m + [{"role": "user", "content": q}]
def extract(text):
    if not isinstance(text, str): return None, False
    t = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    m = re.search(r"```sql\s*(.*?)```", t, re.S) or re.search(r"```\s*(SELECT.*?|WITH.*?)```", t, re.S | re.I)
    if m: return m.group(1).strip().rstrip(";"), False
    if "CANNOT_ANSWER" in t: return None, True
    m = re.search(r"\b(WITH|SELECT)\b.*", t, re.S | re.I)
    return (m.group(0).strip().rstrip(";"), False) if m else (None, False)
_local = threading.local()
def run(sql, timeout=5.0):
    """read-only execution with a wall-clock limit; returns (rows or None, ok)"""
    if not sql or re.search(r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|ATTACH|PRAGMA)\b", sql, re.I): return None, False
    if not hasattr(_local, "con"): _local.con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True, check_same_thread=False)
    con = _local.con; import time; t0 = time.time()
    con.set_progress_handler(lambda: 1 if time.time() - t0 > timeout else 0, 10000)
    try: return [list(r) for r in con.execute(sql).fetchall()], True
    except Exception: return None, False
    finally: con.set_progress_handler(None, 0)
def norm(rows):
    def v(x): return round(x, 2) if isinstance(x, float) else x
    return sorted(json.dumps([v(x) for x in r]) for r in (rows or []))
def score(text, item):
    """returns (reward, correct, kind) with the brief's reward."""
    sql, refused = extract(text)
    if not item["answerable"]:
        return (1.0, True, "refused") if refused else (-0.5, False, "answered_unanswerable")
    if refused or not sql: return 0.0, False, "refused_or_empty"
    rows, ok = run(sql)
    if not ok: return 0.0, False, "sql_error"
    if norm(rows) == norm(item["answer"]): return 1.0, True, "correct"
    return 0.1, False, "wrong_result"

"""Load analysis/erin_handlabel/app_cases.json into the grading app for grader 'rehan',
replacing the earlier assignments (which have no labels). Backs up cases.json and grading.db.
The app must be restarted afterwards (it reads cases.json at start)."""
import json, os, shutil, sqlite3, datetime
H = "/mnt/scratche/slow/fmlab/zuberi01/hand_grading"
A = "/mnt/scratche/slow/fmlab/zuberi01/analysis/erin_handlabel"
ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")
shutil.copy2(H + "/cases.json", f"{H}/cases.json.bak_{ts}"); shutil.copy2(H + "/grading.db", f"{H}/grading.db.bak_{ts}")
new = json.load(open(A + "/app_cases.json"))
allc = [c for c in json.load(open(H + "/cases.json")) if c.get("block") not in ("erin150", "hl2026", "hl2026_repeat")] + new
json.dump(allc, open(H + "/cases.json", "w")); os.chmod(H + "/cases.json", 0o600)
db = sqlite3.connect(H + "/grading.db")
n_lab = db.execute("SELECT COUNT(*) FROM labels WHERE grader='rehan'").fetchone()[0]
old = db.execute("SELECT COUNT(*) FROM assignments WHERE grader='rehan'").fetchone()[0]
if n_lab:
    raise SystemExit(f"rehan already has {n_lab} labels — not replacing assignments; stop and decide")
db.execute("DELETE FROM assignments WHERE grader='rehan'")
db.executemany("INSERT INTO assignments VALUES (?,?,?)", [("rehan", c["case"], i) for i, c in enumerate(new)])
db.commit()
now = db.execute("SELECT COUNT(*) FROM assignments WHERE grader='rehan'").fetchone()[0]
print(f"cases.json: {len(allc)} total ({len(new)} new) | rehan assignments {old} -> {now} (labels {n_lab}) | backups *_{ts}")

"""Pull-worker for the ERIN imminent task set. Claims one task at a time (atomic mkdir), runs it, writes
results/<id>.json atomically, claims the next; exits when nothing runnable remains. Env: ONLY (comma ids), GPU_ONLY."""
import glob, json, os, sys, time, subprocess, urllib.request
import numpy as np, pandas as pd
sys.path.insert(0, "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis/scripts")
T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"; Q = T + "/feasibility/erin_fusion/queue"; TK = T + "/feasibility/erin_fusion/tasks"
ERIN = "/mnt/scratche/fast/fmlab/datasets/imaging/ERIN/data/PathologyReport_AnonIds.csv"
JOB = os.environ.get("SLURM_JOB_ID", "local"); PART = os.environ.get("SLURM_JOB_PARTITION", "?"); ONLY = set(filter(None, os.environ.get("ONLY", "").split(",")))
import torch; HAS_GPU = torch.cuda.is_available()
def write_json(path, obj):
    tmp = path + f".tmp{JOB}"; json.dump(obj, open(tmp, "w"), default=str); os.replace(tmp, path)
def claim(t):
    d = f"{Q}/claims/{t['id']}"
    try: os.mkdir(d); open(d + "/job", "w").write(f"{JOB}@{PART}"); return True
    except FileExistsError:
        try:  # stale-claim reclaim: owner job no longer in squeue
            owner = open(d + "/job").read().split("@")[0]
            if owner != "local" and subprocess.run(["squeue", "-h", "-j", owner], capture_output=True, text=True).stdout.strip() == "":
                os.rename(d, d + f".stale{int(time.time())}"); os.mkdir(d); open(d + "/job", "w").write(f"{JOB}@{PART}"); return True
        except Exception: pass
        return False
def done(tid): return os.path.exists(f"{Q}/results/{tid}.json")
_cache = {}
def table(t):
    if t not in _cache: _cache[t] = pd.read_csv(f"{TK}/{t}.csv", dtype={"sample_id": str, "anon_id": str, "CaseName": str})
    return _cache[t]
def folds_for(t):
    from abmil_clf import patient_folds
    d = table(t); keys = list(d.sample_id); pat = dict(zip(d.sample_id, d.anon_id)); y = dict(zip(d.sample_id, d.y.astype(int)))
    return keys, pat, y, patient_folds(keys, pat, y, 5, seed=0)
def load_bags(d, keys):
    import h5py; bags = {}; ntiles = {}
    for sid, h5s in zip(d.sample_id, d.h5_list):
        if sid not in keys: continue
        parts = []
        for p in str(h5s).split("|"):
            with h5py.File(p) as h: X = np.asarray(h["features"], np.float16)
            if len(X) > 1500: X = X[np.random.RandomState(0).choice(len(X), 1500, replace=False)]
            parts.append(X)
        bags[sid] = np.concatenate(parts).astype(np.float32); ntiles[sid] = int(sum(len(x) for x in parts))
    return bags, ntiles
def run_embed(t):
    os.environ.setdefault("OLLAMA_MODELS", "/mnt/scratche/slow/fmlab/zuberi01/ollama-models"); port = 20000 + int(JOB if JOB.isdigit() else 0) % 20000
    os.environ["OLLAMA_HOST"] = f"127.0.0.1:{port}"; base = f"http://127.0.0.1:{port}"; log = open(f"{Q}/ollama_embed_{JOB}.log", "w")
    srv = subprocess.Popen([os.path.expanduser("~/.local/bin/ollama"), "serve"], stdout=log, stderr=log)
    for _ in range(60):
        try: urllib.request.urlopen(base + "/api/tags", timeout=3); break
        except Exception: time.sleep(2)
    cases = set()
    for tt in ("T1", "T2a", "T2b", "T2c"): cases |= set(table(tt).CaseName)
    rep = pd.read_csv(ERIN, dtype=str).fillna(""); rep = rep[rep.CaseName.isin(cases)].drop_duplicates("CaseName")
    E = {}
    for c, a, b in zip(rep.CaseName, rep.FinalDiagnosis_redacted, rep.MicroscopicDescription_redacted):
        body = json.dumps({"model": "nomic-embed-text", "prompt": (a + " " + b)[:6000]}).encode()
        for _ in range(3):
            try: E[c] = np.asarray(json.loads(urllib.request.urlopen(urllib.request.Request(base + "/api/embeddings", data=body, headers={"Content-Type": "application/json"}), timeout=120).read())["embedding"], np.float32); break
            except Exception: time.sleep(2)
    np.savez(f"{Q}/results/embed.npz", keys=np.array(list(E)), X=np.stack(list(E.values()))); srv.terminate()
    return {"n_reports": len(E), "dim": int(next(iter(E.values())).shape[0]), "model": "nomic-embed-text"}
def tab_design(t, arm, d):
    if arm == "a":
        g = (d.grade.isin(["IND"]) if not t.startswith("T3") else d.grade.eq("NDBE")).astype(float).values; age = d.age.fillna(d.age.median()).values
        return np.column_stack([g, age])
    z = np.load(f"{Q}/results/embed.npz", allow_pickle=True); E = dict(zip(z["keys"], z["X"])); return np.stack([E.get(c, np.zeros(768, np.float32)) for c in d.CaseName])
def fit_lr(X, y, tr, te, seed):
    from sklearn.linear_model import LogisticRegression; from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(X[tr]); m = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced", random_state=seed).fit(sc.transform(X[tr]), y[tr]); return m.predict_proba(sc.transform(X[te]))[:, 1]
def run_tab(t, perm=None):
    d = table(t); keys, pat, y, folds = folds_for(t); idx = {k: i for i, k in enumerate(keys)}; yv = np.array([y[k] for k in keys])
    if perm is not None: yv = np.random.RandomState(1000 + perm).permutation(yv)
    arms = ["a"] + (["b"] if t in ("T1", "T2a", "T2b", "T2c") else []); out = {}
    for arm in arms:
        X = tab_design(t, arm, d); oof = np.zeros((3, len(keys)))
        for s in range(3):
            for f in range(5):
                te = [idx[k] for k in folds[f]]; tr = [i for i in range(len(keys)) if i not in set(te)]; oof[s, te] = fit_lr(X, yv, tr, te, s)
        out[arm] = dict(zip(keys, oof.mean(0).round(6).tolist()))
    return {"task": t, "arms": out, "perm": perm, "y_used": dict(zip(keys, yv.astype(int).tolist()))}
def run_img(t, fold, seed, perm=None):
    from abmil_clf import train_abmil_clf_fold
    d = table(t); keys, pat, y, folds = folds_for(t); te = folds[fold]; tr = [k for j, fl in enumerate(folds) if j != fold for k in fl]
    if perm is not None:
        yv = np.random.RandomState(1000 + perm).permutation([y[k] for k in keys]); y = dict(zip(keys, [int(v) for v in yv]))
    bags, ntiles = load_bags(d, set(keys)); t0 = time.time()
    o = train_abmil_clf_fold(bags, tr, te, y, seed)
    return {"task": t, "fold": fold, "seed": seed, "perm": perm, "preds": {k: round(float(v), 6) for k, v in o.items()}, "ntiles": {k: ntiles[k] for k in te}, "train_s": round(time.time() - t0, 1), "partition": PART}
def main():
    tasks = [json.loads(l) for l in open(Q + "/tasks.jsonl")]
    if ONLY: tasks = [t for t in tasks if t["id"] in ONLY]
    idle = 0
    while True:
        progressed = False
        for t in tasks:
            if done(t["id"]): continue
            if t["gpu"] and not HAS_GPU: continue
            if any(not done(n) for n in t.get("needs", [])): continue
            if not claim(t): continue
            t0 = time.time(); print("RUN", t["id"], flush=True)
            try:
                if t["type"] == "embed": r = run_embed(t)
                elif t["type"] == "tab": r = run_tab(t["task"])
                elif t["type"] == "tabperm": r = {"perms": [run_tab(t["task"], perm=p) for p in range(1, 51)]}
                else: r = run_img(t["task"], t["fold"], t["seed"], t.get("perm"))
                r["_wall_s"] = round(time.time() - t0, 1); r["_job"] = f"{JOB}@{PART}"; write_json(f"{Q}/results/{t['id']}.json", r); print("DONE", t["id"], r["_wall_s"], "s", flush=True)
            except Exception as e:
                print("FAIL", t["id"], repr(e)[:300], flush=True); os.rename(f"{Q}/claims/{t['id']}", f"{Q}/claims/{t['id']}.failed{int(time.time())}")
            progressed = True
        remaining = [t for t in tasks if not done(t["id"]) and (HAS_GPU or not t["gpu"])]
        if not remaining: print("queue empty for this worker", flush=True); return
        if not progressed:
            idle += 1
            if idle > 20 or (ONLY and idle > 2): print("nothing claimable; exiting", flush=True); return
            time.sleep(60)
        else: idle = 0
if __name__ == "__main__": main()

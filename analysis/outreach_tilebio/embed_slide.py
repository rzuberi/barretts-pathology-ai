"""Phase 2 (GPU): embed every tile of data/tiles/{ds}/{slide}.npy with UNI2-h and Virchow2 (fp16 autocast).
Per slide: data/feats/{model}/{ds}/{slide}.npy (fp16) + data/feats/cost/{ds}/{slide}.json with CUDA-synchronised forward seconds for all tiles
and for 16 / 256 random tiles (incl. host->device copy + normalisation; excl. disk read and model load).
Usage: embed_slide.py DS SLIDE                 one slide (smoke)
       embed_slide.py --pull LIST [--wall S]   pull worker: models loaded once, claims slides (lines "ds slide") one by one via per-slide
                                              lock dirs (stale if the owner job left squeue), prefetches the next slide's tiles while the GPU works.
Idempotent per slide and model."""
import argparse, json, os, random, subprocess, sys, threading, time
import numpy as np
A = "/mnt/scratche/slow/fmlab/zuberi01/analysis/outreach_tilebio"; HF = "/mnt/scratche/slow/fmlab/zuberi01/hf_cache"
os.environ.setdefault("HF_HOME", HF); os.environ["HF_HUB_OFFLINE"] = "1"
ap = argparse.ArgumentParser(); ap.add_argument("ds", nargs="?"); ap.add_argument("slide", nargs="?"); ap.add_argument("--pull"); ap.add_argument("--wall", type=int, default=3 * 3600)
ap.add_argument("--models", default="uni2h,virchow2"); ap.add_argument("--bs", type=int, default=256)
a = ap.parse_args(); MODELS = a.models.split(","); T0 = time.time(); JOB = os.environ.get("SLURM_JOB_ID", f"nojob{os.getpid()}")
def paths(ds, sl): return {m: f"{A}/data/feats/{m}/{ds}/{sl}.npy" for m in MODELS}, f"{A}/data/feats/cost/{ds}/{sl}.json"
def is_done(ds, sl): o, c = paths(ds, sl); return all(os.path.exists(p) for p in o.values()) and os.path.exists(c)
items = [tuple(l.split()[-2:]) for l in open(a.pull) if l.strip()] if a.pull else [(a.ds, a.slide)]
if all(is_done(*it) for it in items): print("done already"); sys.exit(0)
import torch, timm
dev = "cuda"; gpu = torch.cuda.get_device_name(0)
def read_tiles(path):
    """sequential chunked read of the uint8 npy (mmap over Lustre is slow; see tile-budget smoke lessons)"""
    with open(path, "rb") as f:
        ver = np.lib.format.read_magic(f); shape, fo, dt = np.lib.format._read_array_header(f, ver)
        X = np.empty(shape, dt); buf = X.reshape(-1).view(np.uint8); step = 1 << 28; o = 0   # bytes (uint8 tiles: identical)
        while o < buf.size: n = f.readinto(memoryview(buf[o:o + step])); assert n > 0; o += n
    return X
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
def build(name):
    from huggingface_hub import snapshot_download
    if name == "uni2h":
        m = timm.create_model("vit_giant_patch14_224", pretrained=False, img_size=224, patch_size=14, depth=24, num_heads=24, init_values=1e-5,
                              embed_dim=1536, mlp_ratio=2.66667 * 2, num_classes=0, no_embed_class=True, mlp_layer=timm.layers.SwiGLUPacked,
                              act_layer=torch.nn.SiLU, reg_tokens=8, dynamic_img_size=True)
        m.load_state_dict(torch.load(snapshot_download("MahmoodLab/UNI2-h") + "/pytorch_model.bin", map_location="cpu"), strict=True)
        f = lambda x: m(x)
    else:
        from safetensors.torch import load_file   # explicit config.json args + direct safetensors load (hf-hub: path took ~3 min cold)
        m = timm.create_model("vit_huge_patch14_224", pretrained=False, img_size=224, init_values=1e-5, num_classes=0, reg_tokens=4, mlp_ratio=5.3375,
                              global_pool="", dynamic_img_size=True, mlp_layer=timm.layers.SwiGLUPacked, act_layer=torch.nn.SiLU)
        m.load_state_dict(load_file(snapshot_download("paige-ai/Virchow2") + "/model.safetensors"), strict=True)
        def f(x):
            o = m(x); return torch.cat([o[:, 0], o[:, 5:].mean(1)], 1)   # CLS + mean of patch tokens (4 register tokens skipped), 2,560-d
    m.eval().to(dev); return f
@torch.inference_mode()
def embed(f, Xs, bs):
    out = []
    for i in range(0, len(Xs), bs):
        x = torch.from_numpy(Xs[i:i + bs]).to(dev, non_blocking=True).permute(0, 3, 1, 2).float().div_(255).sub_(MEAN).div_(STD)
        with torch.autocast("cuda", dtype=torch.float16): out.append(f(x).half())
    return torch.cat(out).cpu().numpy() if out else np.zeros((0, 1), np.float16)
def timed(f, Xs, bs):
    torch.cuda.synchronize(); t = time.time(); E = embed(f, Xs, bs); torch.cuda.synchronize(); return E, time.time() - t
t = time.time(); FN = {}; LOAD = {}
for name in MODELS:
    t1 = time.time(); FN[name] = build(name); LOAD[name] = round(time.time() - t1, 1)
_w = np.zeros((32, 224, 224, 3), np.uint8)
for name in MODELS: embed(FN[name], _w, 32)   # warm-up
print(f"job {JOB} on {gpu}: models loaded {LOAD}", flush=True)
def do_slide(ds, sl, X, t_disk):
    outs, cost_js = paths(ds, sl); N = len(X); assert N > 0, "slide has no tissue tiles"
    for p in list(outs.values()) + [cost_js]: os.makedirs(os.path.dirname(p), exist_ok=True)
    cost = json.load(open(cost_js)) if os.path.exists(cost_js) else {"ds": ds, "slide": sl, "n_tiles": N}
    rng = np.random.RandomState(7); sub = {k: np.sort(rng.choice(N, min(k, N), replace=False)) for k in (16, 256)}
    for name in MODELS:
        f = FN[name]; c = {"gpu": gpu, "t_model_load_s": LOAD[name], "t_disk_read_s": round(t_disk, 1)}
        for k, ix in sub.items(): _, c[f"gpu_s_{k}"] = timed(f, X[ix], min(a.bs, max(1, len(ix))))
        if not os.path.exists(outs[name]):
            E, c["gpu_s_all"] = timed(f, X, a.bs)
            assert E.shape[0] == N and np.isfinite(E.astype(np.float32)).all(), "bad embedding"
            np.save(outs[name] + ".part.npy", E); os.replace(outs[name] + ".part.npy", outs[name]); c["dim"] = int(E.shape[1])
        else: c.update({k: v for k, v in cost.get(name, {}).items() if k in ("gpu_s_all", "dim")})
        cost[name] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in c.items()}
    json.dump(cost, open(cost_js + ".part", "w")); os.replace(cost_js + ".part", cost_js)
    return {m: cost[m].get("gpu_s_all") for m in MODELS}
# ---- claims (pull mode) ----
LK = f"{A}/data/feats/locks"; os.makedirs(LK, exist_ok=True)
def owner_alive(lock):
    try: jid = open(lock + "/job_id").read().strip()
    except OSError: return time.time() - os.path.getmtime(lock) < 1200
    if jid.startswith("nojob"): return time.time() - os.path.getmtime(lock) < 1200
    return bool(subprocess.run(["squeue", "-h", "-j", jid], capture_output=True, text=True).stdout.strip())
def claim(ds, sl):
    if is_done(ds, sl): return None
    lock = f"{LK}/{ds}__{sl}"
    try: os.mkdir(lock)
    except FileExistsError:
        if owner_alive(lock): return None
        try: [os.remove(f"{lock}/{x}") for x in os.listdir(lock)]; os.rmdir(lock); os.mkdir(lock)
        except OSError: return None
    open(lock + "/job_id", "w").write(JOB); return lock
def release(lock):
    try: [os.remove(f"{lock}/{x}") for x in os.listdir(lock)]; os.rmdir(lock)
    except OSError: pass
order = items[:]
if a.pull:   # CAMELYON16 (focal task) before TCGA, each shuffled per job to spread lock contention
    rr = random.Random(JOB); c16 = [it for it in items if it[0] == "c16"]; oth = [it for it in items if it[0] != "c16"]; rr.shuffle(c16); rr.shuffle(oth); order = c16 + oth
done = failed = 0; pre = {}   # prefetch: one background read of the next claimed slide
def fetch(ds, sl, box):
    t = time.time()
    try: box["X"] = read_tiles(f"{A}/data/tiles/{ds}/{sl}.npy")
    except Exception as e: box["err"] = e
    box["t"] = time.time() - t
def next_claim(i):
    while i < len(order):
        ds, sl = order[i]; i += 1
        if os.path.exists(f"{A}/data/tiles/{ds}/{sl}.json") and (lk := claim(ds, sl)): return i, (ds, sl, lk)
        if not a.pull and not os.path.exists(f"{A}/data/tiles/{ds}/{sl}.json"): sys.exit(f"no tiles for {ds}/{sl}")
    return i, None
def run_pass():
    global done, failed
    i, cur = next_claim(0); box = {}; th = None
    if cur: th = threading.Thread(target=fetch, args=(cur[0], cur[1], box)); th.start()
    while cur:
        th.join(); ds, sl, lk = cur; X, t_disk, err = box.get("X"), box.get("t", 0), box.get("err")
        nxt = None
        if time.time() - T0 < a.wall - 900: i, nxt = next_claim(i)   # claim + prefetch the next slide before computing this one
        if nxt: box = {}; th = threading.Thread(target=fetch, args=(nxt[0], nxt[1], box)); th.start()
        try:
            if err: raise err
            g = do_slide(ds, sl, X, t_disk); done += 1; print(f"[done] {ds}/{sl[:40]} n={len(X)} disk={t_disk:.0f}s gpu={g}", flush=True)
        except Exception as e:
            failed += 1; print(f"[fail] {ds}/{sl}: {type(e).__name__}: {e}", flush=True)
            if not a.pull: release(lk); raise
        finally: release(lk); X = None
        cur = nxt
while True:
    run_pass()
    waiting = [it for it in items if not os.path.exists(f"{A}/data/tiles/{it[0]}/{it[1]}.json")]   # tiling still in progress
    if not a.pull or not waiting or time.time() - T0 > a.wall - 1200: break
    print(f"[wait] {len(waiting)} slides not tiled yet; rescanning in 120 s", flush=True); time.sleep(120)
left = sum(not is_done(*it) for it in items)
print(f"[exit] job={JOB} done={done} failed={failed} left_in_list={left} elapsed={time.time()-T0:.0f}s", flush=True)
sys.exit(1 if failed else 0)   # slides left by other live workers or by walltime are picked up by later workers; counts verified separately

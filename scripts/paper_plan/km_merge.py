"""Killcoyne protocol with imaging: merge (docs/paper_plan_killcoyne_multimodal.md @ 3904c44). Aggregates only ->
results/paper_final/killcoyne_multimodal.json."""
import json, glob, os, numpy as np, pandas as pd
from scipy.stats import rankdata
T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"; K = T + "/feasibility/paper_plan/killcoyne"; M = K + "_mm"; OUT = os.environ.get("OUTDIR", T + "/results/paper_final")
def auc(y, s):
    y = np.asarray(y).astype(int); r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1; return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if 0 < n1 < len(y) else float("nan")
def ap(y, s):
    y = np.asarray(y).astype(int); o = np.argsort(-np.asarray(s), kind="mergesort"); y = y[o]; tp = np.cumsum(y); return float(np.sum((tp / np.arange(1, len(y) + 1))[y == 1]) / max(y.sum(), 1))
r3 = lambda x: None if x is None or not np.isfinite(x) else round(float(x), 3)
C = pd.read_csv(M + "/set_C.csv", dtype=str).set_index("Sample"); C["y"] = (C.Status == "P").astype(int); rel = set(pd.read_csv(K + "/rel_rows.csv", dtype=str).Sample)
pats = np.array(sorted(C.Patient.unique())); rows_of = {p: np.where(C.Patient.values == p)[0] for p in pats}; yv = C.y.values; ypat = C.groupby("Patient").y.max().reindex(pats).values
def stats(s, idx_rows=None):   # s aligned to C rows
    y, sc = (yv, s) if idx_rows is None else (yv[idx_rows], s[idx_rows]); P = C.Patient.values if idx_rows is None else C.Patient.values[idx_rows]
    d = pd.DataFrame({"p": P, "y": y, "s": sc}); g = d.groupby("p")
    return {"per_sample_auroc": auc(y, sc), "patient_mean_auroc": auc(g.y.max(), g.s.mean()), "patient_max_auroc": auc(g.y.max(), g.s.max()), "per_sample_auprc": ap(y, sc), "patient_max_auprc": ap(g.y.max(), g.s.max())}
KEYS = ["per_sample_auroc", "patient_mean_auroc", "patient_max_auroc", "per_sample_auprc", "patient_max_auprc"]
rng = np.random.RandomState(0); BOOT = []
while len(BOOT) < 2000:
    ix = rng.choice(len(pats), len(pats))
    if 0 < ypat[ix].sum() < len(ix): BOOT.append(ix)
def boot_rows(ix):
    parts = [rows_of[pats[i]] for i in ix]; rr = np.concatenate(parts); pid = np.concatenate([np.full(len(q), j) for j, q in enumerate(parts)]); return rr, pid
BR = [boot_rows(ix) for ix in BOOT]
def bstat(s, rr, pid):
    y, sc = yv[rr], s[rr]; d = pd.DataFrame({"p": pid, "y": y, "s": sc}); g = d.groupby("p"); return auc(y, sc), auc(g.y.max(), g.s.max())
def summarise(s, ci=True):
    out = {k: r3(v) for k, v in stats(s).items()}; relidx = np.where(C.index.isin(rel))[0]; out["release_rows"] = {k: r3(v) for k, v in stats(s, relidx).items()}; out["release_rows"]["n_samples"] = int(len(relidx))
    exidx = np.where(~C.Pathology.isin(["HGD", "IMC"]).values)[0]; out["excl_HGD_IMC_post_hoc"] = {k: r3(v) for k, v in stats(s, exidx).items()}; out["excl_HGD_IMC_post_hoc"]["n_samples"] = int(len(exidx))   # added post hoc: same predictions, HGD/IMC samples not scored
    if ci:
        b = np.array([bstat(s, rr, pid) for rr, pid in BR]); out["ci95"] = {"per_sample_auroc": [r3(np.percentile(b[:, 0], 2.5)), r3(np.percentile(b[:, 0], 97.5))], "patient_max_auroc": [r3(np.percentile(b[:, 1], 2.5)), r3(np.percentile(b[:, 1], 97.5))]}
    return out
def delta(a, b):   # arm a minus reference b: paired bootstrap CI + within-patient swap permutation
    obs = np.array(stats(a)["per_sample_auroc"] - stats(b)["per_sample_auroc"]), stats(a)["patient_max_auroc"] - stats(b)["patient_max_auroc"]
    bs = np.array([np.subtract(bstat(a, rr, pid), bstat(b, rr, pid)) for rr, pid in BR]); rng = np.random.RandomState(0); pm = []
    for _ in range(2000):
        sw = rng.rand(len(pats)) < 0.5; mask = np.zeros(len(yv), bool)
        for i in np.where(sw)[0]: mask[rows_of[pats[i]]] = True
        a2 = np.where(mask, b, a); b2 = np.where(mask, a, b); pm.append((stats(a2)["per_sample_auroc"] - stats(b2)["per_sample_auroc"], stats(a2)["patient_max_auroc"] - stats(b2)["patient_max_auroc"]))
    pm = np.array(pm); o = [float(obs[0]), float(obs[1])]
    return {k: {"delta": r3(o[j]), "ci95": [r3(np.percentile(bs[:, j], 2.5)), r3(np.percentile(bs[:, j], 97.5))], "perm_p": round(float((1 + (np.abs(pm[:, j]) >= abs(o[j]) - 1e-12).sum()) / 2001), 4)} for j, k in enumerate(["per_sample_auroc", "patient_max_auroc"])}
res = {"_spec": "docs/paper_plan_killcoyne_multimodal.md @ 3904c44", "prep": json.load(open(M + "/prep_info.json"))}
Lf = sorted(glob.glob(M + "/linear/*.csv")); L = pd.concat([pd.read_csv(f, dtype={"Sample": str, "Patient": str}) for f in Lf])
Nf = sorted(glob.glob(M + "/neural/*.csv")); N = pd.concat([pd.read_csv(f, dtype={"sample_id": str, "patient_id": str}) for f in Nf]) if Nf else pd.DataFrame()
res["files"] = {"linear_chunks": len(Lf), "neural_chunks": len(Nf)}
def lin(cfg, rule):
    d = L[(L.cfg == cfg) & (L.rule == rule)].drop_duplicates("Sample").set_index("Sample").pred.reindex(C.index); return d.values if d.notna().all() else None
def neu(m):
    if N.empty: return None, None
    d = N[N.model == m]; per_seed = {}
    for s_, q in d.groupby("seed"):
        v = q.drop_duplicates("sample_id").set_index("sample_id").y_prob.reindex(C.index)
        if v.notna().all(): per_seed[int(s_)] = v.values
    if len(per_seed) < 3: return None, per_seed
    return np.mean([per_seed[k] for k in sorted(per_seed)], 0), per_seed
arms, status = {}, {}
for src, cc, ce, ci_ in [("their", 0, 2, 3), ("pkg", 4, 5, 6)]:
    for rule in ["class_min", "global", "dev_min"]:
        tag = "" if rule == "class_min" else f" [{rule}]"
        cn, im, ea, it = lin(cc, rule), lin(1, rule), lin(ce, rule), lin(ci_, rule)
        for nm, v in [(f"L-CNV ({src}){tag}", cn), (f"L-IMG{tag}", im), (f"L-EARLY ({src}){tag}", ea), (f"L-INTER ({src}){tag}", it), (f"L-LATE ({src}){tag}", None if cn is None or im is None else (cn + im) / 2)]:
            arms[nm] = v
    nimg, _ = neu("img"); arms["N-IMG"] = nimg
    for kind in ["early", "inter"]:
        v, ps = neu(f"{kind}_{src}"); arms[f"N-{kind.upper()} ({src})"] = v
    arms[f"N-LATE ({src})"] = None if nimg is None or arms[f"L-CNV ({src})"] is None else (nimg + arms[f"L-CNV ({src})"]) / 2
res["arms"] = {}; res["deltas_vs_L_CNV"] = {}
for nm, v in arms.items():
    status[nm] = "DONE" if v is not None else "NOT AVAILABLE"
    if v is None: continue
    primary = "[" not in nm; res["arms"][nm] = summarise(v, ci=primary)
    src = "pkg" if "(pkg" in nm else "their"; ref = arms.get(f"L-CNV ({src})" + (nm[nm.index(" ["):] if "[" in nm else ""))
    if primary and not nm.startswith("L-CNV") and ref is not None: res["deltas_vs_L_CNV"][nm] = delta(v, ref)
res["status"] = status
res["neural_seed_auroc"] = {m: {int(s_): r3(auc(C.y.values, v)) for s_, v in (neu(m)[1] or {}).items()} for m in ["img", "early_their", "inter_their", "early_pkg", "inter_pkg"]}
if not N.empty: res["neural_best_epoch"] = N.groupby("model").best_epoch.describe()[["mean", "min", "max"]].round(2).to_dict("index")
rec = json.load(open(OUT + "/killcoyne_reconcile.json")); g = {(r["cfg"], r["rule"]): r for r in rec["grid"]}
res["anchors_773"] = {"their matrix, class min (cfg 32)": g[(32, "class_min")]["own"]["per_sample"], "their matrix, global (cfg 32)": g[(32, "global")]["own"]["per_sample"],
                      "package features, plan B": rec["B"]["own"]["per_sample"], "published": rec["A0"]["all_773"]["per_sample"]["auroc"]}
os.makedirs(OUT, exist_ok=True); json.dump(res, open(OUT + "/killcoyne_multimodal.json", "w"), indent=1)
for nm, v in res["arms"].items(): print(nm.ljust(34), v["per_sample_auroc"], v["patient_max_auroc"], v.get("ci95", ""))
print(json.dumps(res["deltas_vs_L_CNV"], indent=0)[:3000]); print("MERGE DONE")

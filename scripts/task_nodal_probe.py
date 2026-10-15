"""NODAL STATUS from the PRE-TREATMENT OGD biopsy (Rehan, 21 Sep 2026; derived from task_oac_response.py).
LABEL=ypN (primary): pathological node status at resection, ypN1-3 vs ypN0 (nx/missing excluded) --
the ground truth, but measured after neoadjuvant therapy. LABEL=cN (secondary): the clinician's
pre-treatment imaging estimate, cN1-3 vs cN0. Clinical baseline for ypN INCLUDES cN (what the
clinician already knows before surgery); for cN the nstage column is removed from the baseline
(it would be the label itself). Everything else -- OGD-only bags, leakage assertions, folds,
seeds, permutation shards, tile-count confound -- is unchanged from the response probe.

[original header follows]
OAC neoadjuvant response from the PRE-TREATMENT biopsy (docs/oac_response_preregistration.md).

Predict tumour regression grade -- scored later on the resection -- from the
diagnostic OGD slide alone. ABMIL classification head, patient-disjoint folds.

LEAKAGE RULE: the resection slide *is* the answer (fibrosis, residual tumour).
No `_RES` slide may enter a bag. Enforced as an assertion, not a filter.

Env: ENCODER (default features_uni_v2) | PERM=<i> runs permutation shard i
     (label-permutation null, 1 seed) instead of the real run | OUTDIR
     PREP=1 builds the pooled-bag cache and exits (run once per encoder before
     the fan-out: the campaign is 50+ jobs and the features live on /slow).
     REGIMEN=ecf restricts to neoadjuvant_grouped == ECF/ECX/EOF/EOX (the
     pre-registered single-regimen sensitivity arm; 81% of the cohort).
"""
import glob, json, os, re, sys
import numpy as np, pandas as pd, h5py, torch, torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from abmil_cox import ABMIL, _bag, RNG

BASE = "/mnt/scratche/slow/fmlab/datasets/imaging/occams/wsi_data"
FEAT_ROOT = BASE + "/slides/features/20x_224px"
MASTER = "/home/zuberi01/occams_work/occams_master_20260511.csv"
OUT = os.environ.get("OUTDIR", ".")
ENC = os.environ.get("ENCODER", "features_uni_v2")
PERM = os.environ.get("PERM", "")
REGIMEN = os.environ.get("REGIMEN", "all")
LABEL = os.environ.get("LABEL", "ypN")
LABEL_COL = {"ypN": "resection_path_nstage_rp_tnm7", "cN": "nstage_primary_tumour_final_pretreatment_staging_tnm7"}[LABEL]
SEEDS = [0] if PERM else [0, 1, 2]
EPOCHS = int(os.environ.get("EPOCHS", "30"))   # lowered only for smoke tests
MAX_LOAD = 4000                      # per-case tile cap at load (memory bound)
PREP = os.environ.get("PREP", "")
CACHE_DIR = os.environ.get("OAC_CACHE", "/mnt/scratche/fast/fmlab/zuberi01/oac_cache")
CACHE = os.path.join(CACHE_DIR, f"ogd_bags_{ENC}.npz")


def norm_id(s):
    s = str(s).strip().upper().replace("/", "-")
    m = re.search(r"(?:OCCAMS|OC)[-_ ]?([A-Z]{2})[-_ ]?0*([0-9]+)", s)
    return f"{m.group(1)}{int(m.group(2)):04d}" if m else s


# ---------- bags: OGD tiles only, pooled per case ----------
files = sorted(glob.glob(os.path.join(FEAT_ROOT, ENC, "*.h5")))
# 25 OGD slides are named *_NOT-HE* -- IHC / special stains, not H&E. They must be
# excluded: their tile features are a different modality, and the *number* of extra
# stains a case received reflects diagnostic uncertainty, which is a leak vector.
# (Checked 2026-09-09: none of them win a bag under task_occams_v3's rule, so the
# pre-registered Ch2 cohort is unaffected -- this filter matters only for pooling.)
ogd = [f for f in files if "_OGD" in os.path.basename(f).upper()
       and "NOT-HE" not in os.path.basename(f).upper()]
n_nothe = sum(1 for f in files if "NOT-HE" in os.path.basename(f).upper())
assert ogd, f"no OGD slides under {os.path.join(FEAT_ROOT, ENC)}"
for f in ogd:                        # LEAKAGE ASSERTIONS
    b = os.path.basename(f).upper()
    assert "_RES" not in b, f"resection slide in bag set: {f}"
    assert "NOT-HE" not in b, f"non-H&E slide in bag set: {f}"
print(f"[{ENC}] OGD H&E slides={len(ogd)} (excluded {n_nothe} NOT-HE)", flush=True)

def build_cache():
    per_case, ns = {}, {}
    for i, f in enumerate(ogd):
        c = norm_id(os.path.basename(f).split("_")[0])
        with h5py.File(f) as h:
            per_case.setdefault(c, []).append(np.asarray(h["features"]))
        ns[c] = ns.get(c, 0) + 1
        if (i + 1) % 50 == 0:
            print(f"  read {i+1}/{len(ogd)} slides", flush=True)
    ks = sorted(per_case)
    mats = []
    for c in ks:
        F = np.vstack(per_case[c])
        if len(F) > MAX_LOAD:
            F = F[np.sort(RNG(0).choice(len(F), MAX_LOAD, replace=False))]
        mats.append(F.astype(np.float32))
    off = np.cumsum([0] + [len(m) for m in mats]).astype(np.int64)
    os.makedirs(CACHE_DIR, exist_ok=True)
    tmp = f"{CACHE}.{os.getpid()}.tmp.npz"         # pid-unique: concurrent builders cannot collide
    np.savez(tmp, X=np.vstack(mats), offsets=off,
             keys=np.array(ks), n_slides=np.array([ns[c] for c in ks]))
    os.replace(tmp, CACHE)
    print(f"cached {CACHE}  cases={len(ks)} tiles={off[-1]}", flush=True)

if PREP:
    build_cache(); sys.exit(0)
if not os.path.exists(CACHE):
    print(f"[warn] no cache at {CACHE}; building inline", flush=True)
    build_cache()
z = np.load(CACHE, allow_pickle=False)
X_all, off, ks = z["X"], z["offsets"], [str(k) for k in z["keys"]]
bags = {k: X_all[off[i]:off[i + 1]] for i, k in enumerate(ks)}
tile_n = {k: int(off[i + 1] - off[i]) for i, k in enumerate(ks)}
n_slides = {k: int(v) for k, v in zip(ks, z["n_slides"])}
print(f"[{ENC}] OGD slides={len(ogd)} cases={len(bags)} tiles={off[-1]}", flush=True)

# ---------- labels: TRG1-3 responder vs TRG4-5 ----------
m = pd.read_csv(MASTER, dtype=str, low_memory=False)
m["cid"] = m["occams_id"].map(norm_id)
m = m.drop_duplicates("cid").set_index("cid")
nst = m[LABEL_COL].astype(str).str.strip().str.lower()
y = {c: (1.0 if re.match(r"^n[1-3]", nst[c]) else 0.0) for c in m.index
     if c in bags and (re.match(r"^n[1-3]", nst[c]) or nst[c] == "n0")}
trg = pd.Series(dtype=float)
keys = sorted(y)
if REGIMEN == "ecf":                 # pre-registered sensitivity: one regimen family
    grp = m["neoadjuvant_grouped"].astype(str).str.strip()
    keys = [k for k in keys if grp.get(k) == "ECF/ECX/EOF/EOX"]
elif REGIMEN != "all":
    sys.exit(f"unknown REGIMEN {REGIMEN}")
lab = np.array([y[k] for k in keys])
print(f"[label={LABEL} regimen={REGIMEN}] n={len(keys)} node_positive={int(lab.sum())} node_negative={int((1-lab).sum())}", flush=True)

# ---------- clinical baseline arm ----------
# Columns are digit-extracted, so free-text fields yield all-NaN. One such column
# (ps7_mstage_location_of_distant_metastasis..., 100% missing here) NaN-poisoned the
# linear arm end to end. Drop columns that are mostly missing or constant.
cand = [c for c in m.columns if re.search(
    r"age_at_diagnosis|pre_treatment_performance_status|"
    r"(ps_tstage|nstage|mstage).*pretreatment", c, re.I)]
if LABEL == "cN":                      # the pre-treatment N column IS the label: keep it out of the baseline
    cand = [c for c in cand if "nstage" not in c.lower()]
clin_cols, dropped = [], {}
for c in cand:
    v = pd.to_numeric(m.loc[keys, c].astype(str).str.extract(r"(\d+)", expand=False),
                      errors="coerce")
    if v.isna().mean() > 0.5:
        dropped[c] = f"missing {100*v.isna().mean():.0f}%"
    elif v.nunique(dropna=True) < 2:
        dropped[c] = "constant"
    else:
        clin_cols.append(c)
    if len(clin_cols) == 6:
        break
assert clin_cols, "no usable clinical columns"
print("clinical columns:", clin_cols, flush=True)
print("dropped:", dropped, flush=True)
Xc = np.column_stack([pd.to_numeric(
    m.loc[keys, c].astype(str).str.extract(r"(\d+)", expand=False),
    errors="coerce").values for c in clin_cols]).astype(float)
Xc = np.where(np.isnan(Xc), np.nanmean(Xc, axis=0), Xc)
Xc = (Xc - Xc.mean(0)) / (Xc.std(0) + 1e-8)
assert np.isfinite(Xc).all(), "non-finite clinical design matrix"


def auc(y_true, score):
    """Mann-Whitney AUC with MID-RANKS for ties.

    argsort(argsort(x)) gives ordinal ranks, which for tied scores silently falls
    back to input order -- on tied clinical staging values that manufactured an
    AUC of 0.766 out of alphabetical case-ID order (caught in smoke, 2026-09-09).
    """
    y_true = np.asarray(y_true, dtype=float)
    score = np.asarray(score, dtype=float)
    ok = ~np.isnan(score)
    y_true, score = y_true[ok], score[ok]
    n1, n0 = int((y_true == 1).sum()), int((y_true == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    srt = score[order]
    mid = np.empty(len(srt), dtype=float)
    i = 0
    while i < len(srt):
        j = i
        while j + 1 < len(srt) and srt[j + 1] == srt[i]:
            j += 1
        mid[i:j + 1] = (i + j) / 2.0 + 1.0
        i = j + 1
    r = np.empty(len(srt), dtype=float); r[order] = mid
    return (r[y_true == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def boot_auc(y_true, score, n_boot=1000, seed=0):
    rng = RNG(seed); y_true, score = np.asarray(y_true), np.asarray(score)
    vals = []
    for _ in range(n_boot):
        i = rng.choice(len(y_true), len(y_true), replace=True)
        if len(set(y_true[i])) < 2:
            continue
        vals.append(auc(y_true[i], score[i]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def folds(keys, lab, n_folds=5, seed=0):
    rng = RNG(seed); out = [[] for _ in range(n_folds)]
    for cls in (0.0, 1.0):
        idx = [k for k, l in zip(keys, lab) if l == cls]
        rng.shuffle(idx)
        for j, k in enumerate(idx):
            out[j % n_folds].append(k)
    return out


def train_cls_fold(bags, tr, te, y, seed, epochs=30, lr=1e-4, mb=16):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = RNG(seed); torch.manual_seed(seed)
    model = ABMIL(d_in=next(iter(bags.values())).shape[1]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    npos = sum(y[k] for k in tr); nneg = len(tr) - npos
    pw = torch.tensor([nneg / max(npos, 1)], device=dev)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    for _ in range(epochs):
        order = list(rng.permutation(tr))
        for i in range(0, len(order), mb):
            chunk = order[i:i + mb]
            logits = torch.stack([model(_bag(bags, k, rng).to(dev))[0] for k in chunk])
            loss = lossf(logits, torch.tensor([y[k] for k in chunk],
                                              dtype=torch.float32, device=dev))
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.inference_mode():
        return {k: float(model(_bag(bags, k, RNG(0)).to(dev))[0]) for k in te}


def linear_fold(X, keys, tr, te, y, seed, epochs=400, lr=5e-3):
    dev = "cpu"; torch.manual_seed(seed)
    ix = {k: i for i, k in enumerate(keys)}
    lin = nn.Linear(X.shape[1], 1)
    opt = torch.optim.Adam(lin.parameters(), lr=lr, weight_decay=1e-3)
    Xtr = torch.tensor(X[[ix[k] for k in tr]], dtype=torch.float32)
    ytr = torch.tensor([y[k] for k in tr], dtype=torch.float32)
    npos = float(ytr.sum()); pw = torch.tensor([(len(ytr) - npos) / max(npos, 1)])
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    for _ in range(epochs):
        loss = lossf(lin(Xtr).squeeze(-1), ytr)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.inference_mode():
        p = lin(torch.tensor(X[[ix[k] for k in te]], dtype=torch.float32)).squeeze(-1)
    return {k: float(v) for k, v in zip(te, p)}


# ---------- run ----------
y_run = dict(y)
if PERM:                             # label-permutation null, cases fixed
    rng = RNG(1000 + int(PERM))
    shuf = rng.permutation([y[k] for k in keys])
    y_run = {k: float(v) for k, v in zip(keys, shuf)}

res = {}
ytrue = [y_run[k] for k in keys]
for arm in ("hist", "clin"):
    per_seed, aucs = [], []
    for seed in SEEDS:
        oof = {}
        for fi, te in enumerate(folds(keys, ytrue, seed=seed)):
            te_set = set(te)
            tr = [k for k in keys if k not in te_set]
            oof.update(train_cls_fold(bags, tr, te, y_run, seed, epochs=EPOCHS) if arm == "hist"
                       else linear_fold(Xc, keys, tr, te, y_run, seed))
            print(f"  {arm} seed{seed} fold{fi} done", flush=True)
        per_seed.append(oof)
        aucs.append(auc(ytrue, [oof[k] for k in keys]))
    # seed-averaged OOF scores, then metric once -- matches task_occams_v3 convention
    mean_oof = {k: float(np.mean([p[k] for p in per_seed])) for k in keys}
    score = [mean_oof[k] for k in keys]
    res[arm] = {"auc": round(auc(ytrue, score), 4),
                "auc_per_seed": [round(a, 4) for a in aucs]}
    if not PERM:
        res[arm]["auc_ci"] = [round(v, 4) for v in boot_auc(ytrue, score)]

if not PERM:                         # pre-specified confound: biopsy adequacy
    tn = [tile_n[k] for k in keys]
    res["confound_tilecount"] = {
        "auc_of_tilecount_alone": round(auc([y[k] for k in keys], tn), 4),
        "mean_tiles_node_positive": round(float(np.mean([tile_n[k] for k in keys if y[k] == 1])), 1),
        "mean_tiles_node_negative": round(float(np.mean([tile_n[k] for k in keys if y[k] == 0])), 1),
        "mean_ogd_slides_per_case": round(float(np.mean([n_slides[k] for k in keys])), 2)}
    res["_meta"] = {"encoder": ENC, "regimen": REGIMEN, "n": len(keys),
                    "label": LABEL, "label_column": LABEL_COL,
                    "n_node_positive": int(lab.sum()), "n_node_negative": int((1 - lab).sum()),
                    "seeds": SEEDS, "epochs": EPOCHS, "max_load_tiles": MAX_LOAD,
                    "clinical_columns": clin_cols,
                    "clinical_dropped": dropped,
                    "leakage_assertion": "no _RES and no NOT-HE in any bag (asserted)",
                    "n_not_he_excluded": n_nothe}

tag = (f"_{REGIMEN}" if REGIMEN != "all" else "") + (f"_perm{PERM}" if PERM else "")
path = os.path.join(OUT, f"oac_nodal_{LABEL}_{ENC.replace('features_', '')}{tag}.json")
json.dump(res, open(path, "w"), indent=1)
print("wrote", path); print(json.dumps(res, indent=1))

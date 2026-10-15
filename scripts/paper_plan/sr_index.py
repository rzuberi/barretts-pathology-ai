"""Scanner robustness, items 1-3 and secondaries S1-S7 (docs/paper_scanner_robustness.md @ abe96e9). Label-free: reads sample, patient, endoscopy
and scanner only (no progression label is read). Sample representation = mean over the concatenated tile bags of a sample's slides (km_prep.py
order). Robustness index R_k = sum|SO| / (sum|SO| + sum|OS|) over the k cosine nearest neighbours (Komen 2025 Methods 4.4), patient = biology,
scanner = confounder; pool P = the samples of patients with samples on both scanners. Patient bootstrap 2,000 draws RandomState(0), neighbour
lists fixed, paired across extractors. Row-level outputs -> killcoyne_mm/scanner_robustness/; aggregates -> results/paper_final/scanner_robustness/index.json."""
import os, sys, glob, json, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sr_common import ROW, OUT, M, slide_map, gp_of, sha_of
os.makedirs(OUT, exist_ok=True); NB = 2000; K0 = 10; KMAX = 50; KT = 50; NT = 32; r4 = lambda x: None if x is None or not np.isfinite(x) else round(float(x), 4)
pct = lambda a: [r4(np.percentile(a, 2.5)), r4(np.percentile(a, 97.5))]
mp = slide_map(); C = pd.read_csv(M + "/set_C.csv", dtype=str); ids = C.Sample.values; n = len(ids)
pat = C.Patient.values; endo = (C.Patient + "|" + C.Endoscopy).values; scan = mp.drop_duplicates("cnv").set_index("cnv").scanner.reindex(ids).values
assert pd.notna(scan).all()
# ---- checks: Prov-GigaPath coords on every slide (sr_tiles.py), extractor availability
TC = pd.concat([pd.read_csv(f) for f in glob.glob(ROW + "/tiles_check/c*of*.csv")]).drop_duplicates("sha"); assert len(TC) == 683, len(TC)
gp_ok = bool(TC.gp_coords_equal.all() and (TC.n_emb_gp == TC.n_coords).all())
EXT = ["uni2", "virchow2", "hoptimus0"] + (["gigapath"] if gp_ok else []); NAME = {"uni2": "UNI2-h", "virchow2": "Virchow2", "hoptimus0": "H-optimus-0", "gigapath": "Prov-GigaPath"}
def tiles(m, f):
    if m == "uni2": return np.load(f, allow_pickle=True)["embeddings"]
    if m == "gigapath": return np.load(gp_of(f), allow_pickle=True)["embeddings"]
    return np.load(f"{ROW}/emb/{m}/{sha_of(f)}.npz")["embeddings"]
# ---- pool P and tile subset
nsc = pd.Series(scan).groupby(pat).nunique(); dual = set(nsc[nsc > 1].index); inP = np.array([p in dual for p in pat]); Pi = np.where(inP)[0]
upP = np.array(sorted(dual)); pidx = {p: i for i, p in enumerate(upP)}
SX, TX, TS = {}, {}, None   # sample means, tile subsets, tile -> sample index
setP = set(ids[Pi]); row_of = {s: i for i, s in enumerate(ids)}
for m in EXT:
    mean_, tl, ts = {}, [], []
    for s, g in mp.groupby("cnv", sort=True):
        B = [np.asarray(tiles(m, f), np.float32) for f in g.npz]; mean_[s] = np.concatenate(B, 0).mean(0)
        if s in setP: tl += [b[:NT] for b in B]; ts += [np.full(min(NT, len(b)), row_of[s]) for b in B]
    SX[m] = np.vstack([mean_[s] for s in ids]); TX[m] = np.vstack(tl); TS = np.concatenate(ts); np.save(f"{ROW}/sample_mean_{m}.npy", SX[m])
    print("loaded", m, SX[m].shape, TX[m].shape, flush=True)
IM = pd.read_csv(M + "/img_mean.csv").set_index("Sample").reindex(ids).values; uni2_maxdiff = float(np.abs(IM - SX["uni2"]).max()); assert uni2_maxdiff <= 1e-4, uni2_maxdiff
# ---- neighbour counts
def knn(Q, Y, excl, kmax):
    Qn = Q / np.linalg.norm(Q, axis=1, keepdims=True); Yn = Y / np.linalg.norm(Y, axis=1, keepdims=True); S = Qn @ Yn.T; S[excl] = -np.inf
    return np.argsort(-S, axis=1, kind="stable")[:, :kmax]
def counts(nb, qp, qs, yp, ys):
    """cumulative per-query counts over k = 1..kmax of SS, SO, OS (patient = qp/yp, scanner = qs/ys)."""
    sp = yp[nb] == qp[:, None]; ss_ = ys[nb] == qs[:, None]
    return {"SS": np.cumsum(sp & ss_, 1), "SO": np.cumsum(sp & ~ss_, 1), "OS": np.cumsum(~sp & ss_, 1)}
def chance(excl, qp, qs, yp, ys):
    sp = yp[None, :] == qp[:, None]; ss_ = ys[None, :] == qs[:, None]; ok = ~excl
    return {"SS": float((sp & ss_ & ok).sum()), "SO": float((sp & ~ss_ & ok).sum()), "OS": float((~sp & ss_ & ok).sum())}
rng = np.random.RandomState(0); W = np.stack([np.bincount(rng.choice(len(upP), len(upP)), minlength=len(upP)) for _ in range(NB)])   # draws x patients
def per_patient(c, qp, k):   # sum of counts at k per pool patient
    a = np.array([pidx[p] for p in qp]); return {t: np.bincount(a, weights=c[t][:, k - 1], minlength=len(upP)) for t in c}
def R_of(pp, form="robo"):
    num, den = (pp["SO"], pp["SO"] + pp["OS"]) if form == "robo" else (pp["SS"] + pp["SO"], pp["SS"] + pp["OS"])
    return num.sum() / den.sum(), (W @ num) / (W @ den)
def analysis(m, Q, qp, qs, Y, yp, ys, excl, kmax, kpt, form="robo", curve=False):
    nb = knn(Q, Y, excl, kmax); c = counts(nb, qp, qs, yp, ys); ch = chance(excl, qp, qs, yp, ys)
    pt, bs = R_of(per_patient(c, qp, kpt), form)
    rc = ch["SO"] / (ch["SO"] + ch["OS"]) if form == "robo" else (ch["SS"] + ch["SO"]) / (ch["SS"] + ch["OS"])
    out = {"R": r4(pt), "ci95": pct(bs), "chance": r4(rc), "k": kpt, "neighbour_classes_at_k": {t: int(c[t][:, kpt - 1].sum()) for t in c}}
    if curve:
        out["curve"] = []
        for k in range(1, kmax + 1):
            p_, b_ = R_of(per_patient(c, qp, k), form); out["curve"].append({"k": k, "R": r4(p_), "ci95": pct(b_)})
    return out, bs, nb
res = {"prespec": "abe96e9", "extractors": [NAME[m] for m in EXT], "dims": {NAME[m]: int(SX[m].shape[1]) for m in EXT},
       "checks": {"uni2_img_mean_max_abs_diff": uni2_maxdiff, "gigapath_coords_equal_all_slides": gp_ok, "gigapath_slides_coords_equal": int(TC.gp_coords_equal.sum()), "slides": int(len(TC))},
       "pool": {"patients": int(len(upP)), "samples": int(len(Pi)), "samples_by_scanner": pd.Series(scan[Pi]).value_counts().to_dict(),
                "endoscopies": int(len(set(endo[Pi]))), "endoscopies_on_both_scanners": int((pd.Series(scan[Pi]).groupby(endo[Pi]).nunique() > 1).sum()), "tiles_S4": int(len(TS))},
       "primary": {}, "secondary": {"S1_curve": {}, "S2_all_sample_pool": {}, "S3_zscored": {}, "S4_tile_level": {}, "S5_deJong_form": {}, "S6_other_endoscopy": {}, "S7_scanner_probe": {}}}
BS = {}; qp, qs = pat[Pi], scan[Pi]; selfP = np.eye(len(Pi), dtype=bool); sameendo = endo[Pi][:, None] == endo[Pi][None, :]
selfAll = np.zeros((len(Pi), n), bool); selfAll[np.arange(len(Pi)), Pi] = True
mu, sd = None, None
for m in EXT:
    X = SX[m]; XP = X[Pi]
    o, BS[("primary", m)], nb = analysis(m, XP, qp, qs, XP, qp, qs, selfP, KMAX, K0, curve=True); np.save(f"{ROW}/neighbours_primary_{m}.npy", Pi[nb])
    res["secondary"]["S1_curve"][NAME[m]] = o.pop("curve"); res["primary"][NAME[m]] = o
    o, BS[("S2", m)], _ = analysis(m, XP, qp, qs, X, pat, scan, selfAll, K0, K0); res["secondary"]["S2_all_sample_pool"][NAME[m]] = o
    Z = (X - X.mean(0)) / np.where(X.std(0, ddof=1) > 0, X.std(0, ddof=1), 1); Z[:, X.std(0, ddof=1) == 0] = 0
    o, BS[("S3", m)], _ = analysis(m, Z[Pi], qp, qs, Z[Pi], qp, qs, selfP, K0, K0); res["secondary"]["S3_zscored"][NAME[m]] = o
    tp, tsc = pat[TS], scan[TS]; exT = TS[:, None] == TS[None, :]
    o, BS[("S4", m)], _ = analysis(m, TX[m], tp, tsc, TX[m], tp, tsc, exT, KT, KT); res["secondary"]["S4_tile_level"][NAME[m]] = o
    o, BS[("S5", m)], _ = analysis(m, XP, qp, qs, XP, qp, qs, selfP, K0, K0, form="dejong"); res["secondary"]["S5_deJong_form"][NAME[m]] = o
    o, BS[("S6", m)], _ = analysis(m, XP, qp, qs, XP, qp, qs, sameendo, K0, K0); res["secondary"]["S6_other_endoscopy"][NAME[m]] = o
    oof = np.zeros(n); yv = (scan == "C13210").astype(int)
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X, yv, pat):
        oof[te] = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)).fit(X[tr], yv[tr]).predict_proba(X[te])[:, 1]
    res["secondary"]["S7_scanner_probe"][NAME[m]] = {"auroc_oof": r4(roc_auc_score(yv, oof))}
    print("done", m, res["primary"][NAME[m]], flush=True)
# ---- decision rule (item 3) and deltas
U = res["primary"]["UNI2-h"]["R"]; dec = {"rule": "eligible E in {Virchow2, H-optimus-0}; switch if lower 95% CI of dR_10 > 0 AND R_10(E) >= 1.25 x R_10(UNI2-h); if both, larger R_10",
                                          "candidates": {}}
for m in [x for x in EXT if x != "uni2"]:
    d = BS[("primary", m)] - BS[("primary", "uni2")]; Rm = res["primary"][NAME[m]]["R"]
    e = {"R_10": Rm, "dR": r4(Rm - U), "dR_ci95": pct(d), "ratio_to_uni2": r4(Rm / U), "eligible": m in ("virchow2", "hoptimus0")}
    e["crit_ci_above_0"] = bool(np.percentile(d, 2.5) > 0); e["crit_ratio_ge_1.25"] = bool(Rm >= 1.25 * U); e["qualifies"] = bool(e["eligible"] and e["crit_ci_above_0"] and e["crit_ratio_ge_1.25"])
    dec["candidates"][NAME[m]] = e
q = [k for k, v in dec["candidates"].items() if v["qualifies"]]; dec["switch_to"] = max(q, key=lambda k: dec["candidates"][k]["R_10"]) if q else None
dec["decision"] = f"switch to {dec['switch_to']}" if q else "keep UNI2-h"; res["decision"] = dec
agree = {}
for sname, key in (("S2_all_sample_pool", "S2"), ("S3_zscored", "S3"), ("S4_tile_level", "S4"), ("S5_deJong_form", "S5"), ("S6_other_endoscopy", "S6")):
    agree[sname] = {}
    for m in ("virchow2", "hoptimus0"):
        d = BS[(key, m)] - BS[(key, "uni2")]; dp = res["secondary"][sname][NAME[m]]["R"] - res["secondary"][sname]["UNI2-h"]["R"]
        agree[sname][NAME[m]] = {"dR": r4(dp), "dR_ci95": pct(d), "same_sign_as_primary": bool(np.sign(dp) == np.sign(res["decision"]["candidates"][NAME[m]]["dR"]))}
res["secondary"]["agreement"] = agree
json.dump(res, open(f"{OUT}/index.json", "w"), indent=1); json.dump(dec, open(f"{ROW}/decision.json", "w"), indent=1)
print(json.dumps(res["primary"], indent=1)); print(json.dumps(dec, indent=1)); print("SR INDEX DONE", flush=True)

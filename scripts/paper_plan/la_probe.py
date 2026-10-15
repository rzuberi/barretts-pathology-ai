"""Latent space Section 1 (docs/paper_latent_attention.md @ dfac9ad): linear probes per population x repeat x representation. POP = pre | pre_ndbe, REP = 1..10,
REPR = R0 | R1 | R2 | R3 | R4 | R2p | R3p | R4p. Representations rebuilt per outer fold on the fold's training rows (all 676-sample training rows, as the model);
R1/R2 checked against the la_refit.R row-sum fingerprints (<= 1e-6), R3 read from the la_refit.R dumps. Row-level output stays on the cluster."""
import os, json, numpy as np, pandas as pd, warnings
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"; M = T + "/feasibility/paper_plan/killcoyne_mm"; LA = M + "/latent_attention"; RF = LA + "/refit"; OD = LA + "/probe"; os.makedirs(OD, exist_ok=True)
POP, REP, REPR = os.environ["POP"], int(os.environ["REP"]), os.environ["REPR"]; outf = f"{OD}/{POP}_rep_{REP:02d}_{REPR}.csv"
if os.path.exists(outf): raise SystemExit(0)
C = pd.read_csv(M + "/set_C.csv", dtype=str); ids = C.Sample.values; n = len(ids)
SM = pd.read_csv(M + "/horizons/samples.csv", dtype={"Sample": str, "Patient": str}).set_index("Sample").reindex(ids); assert SM.Patient.notna().all()
SD = pd.read_csv(M + "/horizons/slide_desc.csv", dtype={"Sample": str}).set_index("Sample").reindex(ids); assert SD.scanner.notna().all()
pat = SM.Patient.values; y = SM.y.values.astype(int); pre = SM.pre.astype(bool).values; ndbe = SM.ndbe.astype(bool).values
popm = pre if POP == "pre" else pre & ndbe
def block(f, key):
    R = pd.read_csv(f"{M}/{f}", dtype={key: str}).set_index(key).reindex(ids); assert R.notna().all().all(); return R
IM = block("img_mean.csv", "Sample").values.astype(float); CNt = block("cnv_their_C.csv", "sample_id"); CNp = block("cnv_pkg_C.csv", "sample_id")
cx = CNp["cx"].values.astype(float)   # package cx (monotone in raw cx: frozen scaling, models/killcoyne_frozen_pkg_v1/cnv_scaling_setC.csv)
CNt, CNp = CNt.values.astype(float), CNp.values.astype(float)
FO = pd.read_csv(f"{M}/cv/preds/cfg_00_rep_{REP:02d}.csv", dtype={"Sample": str}).set_index("Sample").fold.reindex(ids).values.astype(int)
def zs(X, tr):   # kv_common.R:5 (R sd = ddof 1; zero-sd columns -> 0)
    mu = X[tr].mean(0); sd = X[tr].std(0, ddof=1); Z = (X - mu) / np.where(sd > 0, sd, 1); Z[:, sd == 0] = 0; return Z
def rep_matrix(k, tr):
    if REPR == "R0": return IM
    if REPR == "R1": return zs(IM, tr)
    if REPR in ("R2", "R2p"): return np.hstack([CNt if REPR == "R2" else CNp, zs(IM, tr)])
    if REPR in ("R4", "R4p"): return CNt if REPR == "R4" else CNp
    D = pd.read_csv(f"{RF}/repr_inter_{'their' if REPR == 'R3' else 'pkg'}_rep_{REP:02d}_fold_{k:02d}.csv.gz", dtype={"Sample": str}).set_index("Sample").reindex(ids); return D.values.astype(float)
FPK = {"R1": "img", "R2": "early_their", "R2p": "early_pkg"}
fpr = pd.read_csv(f"{RF}/rowsum_{FPK[REPR]}_rep_{REP:02d}.csv", dtype={"Sample": str}) if REPR in FPK else None
med = lambda v: np.median(v[popm])
TG = {"progressor": y.astype(float), "scanner": (SD.scanner.values == "C13210").astype(float), "tiles": (SD.tissue_tiles.values.astype(float) > med(SD.tissue_tiles.values.astype(float))).astype(float)}
if POP == "pre": TG["grade"] = np.where(SM.Pathology.values == "NDBE", 0.0, np.where(np.isin(SM.Pathology.values, ["ID", "LGD"]), 1.0, np.nan))
if REPR in ("R0", "R1", "R2", "R3", "R2p", "R3p"): TG["cx"] = (cx > med(cx)).astype(float)
CS = 10.0 ** np.arange(-4, 2.0001, 0.5)
def probe(X, lab, tr, te, seed):
    ok = tr[np.isfinite(lab[tr])]; yy = lab[ok].astype(int)
    if yy.min() == yy.max(): return None, None
    cvs = StratifiedGroupKFold(5, shuffle=True, random_state=seed); sc = np.zeros(len(CS)); cnt = np.zeros(len(CS))
    for a, b in cvs.split(X[ok], yy, pat[ok]):
        if yy[a].min() == yy[a].max() or yy[b].min() == yy[b].max(): continue
        for j, c in enumerate(CS):
            m = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=5000)).fit(X[ok][a], yy[a]); sc[j] += roc_auc_score(yy[b], m.predict_proba(X[ok][b])[:, 1]); cnt[j] += 1
    cbest = CS[int(np.argmax(np.where(cnt > 0, sc / np.maximum(cnt, 1), -1)))]   # argmax returns the first (smallest C) on ties
    m = make_pipeline(StandardScaler(), LogisticRegression(C=cbest, max_iter=5000)).fit(X[ok], yy); return m.predict_proba(X[te])[:, 1], cbest
rows, fpdiff = [], 0.0
for k in range(1, 11):
    te_all = FO == k; tr = np.where(~te_all)[0]; X = rep_matrix(k, tr)
    if fpr is not None: f = fpr[fpr.fold == k].set_index("Sample").reindex(ids); fpdiff = max(fpdiff, float(np.abs(X.sum(1) - f.rowsum.values).max())); assert int(f.ncol.iloc[0]) == X.shape[1]
    ptr = np.intersect1d(tr, np.where(popm)[0]); pte = np.where(te_all & popm)[0]
    for t, lab in TG.items():
        pr, cb = probe(X, lab, ptr, pte, 1000 * REP + k)
        if pr is None: continue
        rows += [{"Sample": ids[i], "rep": REP, "fold": k, "repr": REPR, "target": t, "label": lab[i], "pred": p, "C": cb} for i, p in zip(pte, pr)]
    Xn = X[pte] / np.clip(np.linalg.norm(X[pte], axis=1, keepdims=True), 1e-12, None); S = Xn @ Xn.T; np.fill_diagonal(S, -np.inf); nn = S.argmax(1); pp = pat[pte]
    same_n = np.array([(pp == q).sum() - 1 for q in pp])
    rows += [{"Sample": ids[i], "rep": REP, "fold": k, "repr": REPR, "target": "nn_same_patient", "label": same_n[j] / (len(pte) - 1), "pred": float(pp[nn[j]] == pp[j]), "C": np.nan} for j, i in enumerate(pte)]
assert fpdiff <= 1e-6, f"representation fingerprint mismatch {fpdiff}"
pd.DataFrame(rows).to_csv(outf, index=False); json.dump({"fingerprint_max_abs_diff": fpdiff}, open(outf.replace(".csv", ".json"), "w")); print("LA PROBE DONE", POP, REP, REPR, fpdiff, flush=True)

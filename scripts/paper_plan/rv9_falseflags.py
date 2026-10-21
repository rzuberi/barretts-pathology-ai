"""Review round 9: methods to reduce false flags in non-progressors (docs/paper_review_round9_false_flags.md @ 7db8e14). EXPLORATORY.
Stored patient-grouped outer and inner out-of-fold predictions are re-scored (tb_common.py loaders) except TASK=m_B4* (refits by rv9_refit.R); no ACE-B file is read,
frozen bundles and stored predictions are not touched. Setup, endoscopy units, patient outcomes, bootstrap draws and the weighted-sum metric helpers are those of
rv8_landscape.py / rv8b_step3.py, whose module bodies are executed here unchanged up to their task dispatch (setup, thr_arr, r49_calls, patient_arrays, pdraws, parts, met, dstat, ...).
Nested evaluation: for every outer (repeat, fold) each method is built on the fold's TRAINING patients only (inner out-of-fold predictions; logistic stackers are
cross-validated inside the training patients) and applied to the fold's held-out patients (outer predictions). Operating threshold: the highest threshold at which the
training progressor-patient detection is >= the R49 gate's own training-patient detection in that fold; hyper-parameters: lowest training non-progressor patients
flagged subject to that constraint (alternative: patient-level partial AUC at FPR <= 0.2, then the same threshold rule).
Row-level files only under killcoyne_mm/review_round9/ on the cluster; aggregates (no identifiers, no dates, cells < 5 shown as '<5') -> results/paper_final/review_round9/<TASK>.json.
TASK = prep_later | f0 | icc | m_<GROUP> | m_B4a | m_B4b | merge | rob_<row> | curves"""
import os, sys, json, time, numpy as np
from scipy.stats import fisher_exact
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
assert not os.environ.get("RV8_NB") and not os.environ.get("RV8B_NB"), "round-8 smoke switches must not be set"
_s = open(HERE + "/rv8b_step3.py").read(); exec(compile(_s[:_s.index('\nres = {"task": TASK}')], HERE + "/rv8b_step3.py", "exec"))
from sklearn.linear_model import LogisticRegression
PRESPEC = "7db8e14"; OUT = T + "/results/paper_final/review_round9"; R9 = M + "/review_round9"; SMOKE = bool(os.environ.get("RV9_NB"))
if SMOKE:   # smoke mode: fewer bootstrap draws, aggregates written aside (never to results/)
    NB = int(os.environ["RV9_NB"]); TC.NB = NB; OUT = R9 + "/smoke_json"; R9 = R9 + "/smoke"
for d in (OUT, R9): os.makedirs(d, exist_ok=True)
NEG, BIG = -1e6, 1e6
GRID27 = [(ql, qh, qs) for ql in (0.90, 0.95, 0.98) for qh in (0.85, 0.90, 0.95) for qs in (0.70, 0.80, 0.90)]; NOMINAL = (0.95, 0.90, 0.80)
def dump9(res, name=None):
    res["prespec"] = PRESPEC; json.dump(res, open(f"{OUT}/{name or TASK}.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)); print("RV9 DONE", name or TASK, flush=True)
# ================================================================== static structure (patient-ordered matrices)
def statics(S):
    E, P_, es, ORD = S["E"], S["P_"], S["es"], S["ORD"]; mx = max(len(s) for s in es); IDX = np.full((E, mx), -1)
    for e, s in enumerate(es): IDX[e, :len(s)] = s
    W = max(len(o) for o in ORD); PADI = np.full((P_, W), -1); TY = np.zeros((P_, W))
    for i, o in enumerate(ORD): PADI[i, :len(o)] = o; TY[i, :len(o)] = np.asarray((S["ED"][o] - S["ED"][o[0]]) / DAY, float) / 365.25
    VAL = PADI >= 0; GRv = SM.Pathology.map({"NDBE": 0, "ID": 1, "LGD": 2, "HGD": 3, "IMC": 4}).values.astype(float); dys = (GRv == 1) | (GRv == 2)
    P53 = pd.to_numeric(SM["P53 IHC"], errors="coerce").values; hasp = np.isfinite(P53); ab = np.where(hasp, P53, 0) == 1
    ST = dict(IDX=IDX, MSK=IDX >= 0, PADI=PADI, PADS=np.where(VAL, PADI, 0), VAL=VAL, TY=TY, W=W,
              DYS=np.array([dys[s].any() for s in es]).astype(float), AGE=np.array([SM.age.values[s].mean() for s in es]), SEX=np.array([SM.sex_M.values[s].mean() for s in es]),
              P53AV=np.array([hasp[s].any() for s in es]), P53AB=np.array([ab[s].any() for s in es]), PF=np.stack([S["EF"][o[0]] for o in ORD]), EW={})
    for h in (0.5, 1, 2, 4, 8, 16):
        d = TY[:, :, None] - TY[:, None, :]; low = np.tril(np.ones((W, W), bool))[None] & VAL[:, None, :] & VAL[:, :, None]; ST["EW"][h] = np.where(low, 0.5 ** (np.maximum(d, 0) / h), 0.0)
    return ST
def to_mat(ST, s): return np.where(ST["VAL"], s[ST["PADS"]], np.nan)
def from_mat(ST, Mx, E): u = np.zeros(E); u[ST["PADI"][ST["VAL"]]] = Mx[ST["VAL"]]; return u
def memory(ST, s, kind, par=None, mu0=0.0):
    """memory score of every endoscopy from the patient's endoscopies so far (time order); every kind returns a score with flag = score >= t"""
    Sm = to_mat(ST, s); V = ST["VAL"]; S0 = np.where(V, Sm, 0.0); W = ST["W"]; first = np.zeros_like(V); first[:, 0] = True
    prev = np.concatenate([np.full((S0.shape[0], 1), NEG), S0[:, :-1]], 1); prev2 = np.concatenate([np.full((S0.shape[0], 2), NEG), S0[:, :-2]], 1)
    if kind == "none": U = Sm
    elif kind == "mean": U = np.cumsum(S0, 1) / np.maximum(np.cumsum(V, 1), 1)
    elif kind == "ew": Wt = ST["EW"][par]; U = (Wt * S0[:, None, :]).sum(2) / np.maximum(Wt.sum(2), 1e-12)
    elif kind == "last2": U = np.where(first, S0, (S0 + prev) / 2)
    elif kind == "cons2": U = np.where(first, NEG, np.minimum(S0, prev))
    elif kind == "k2of3": st = np.sort(np.stack([S0, prev, prev2], 0), 0)[::-1]; U = st[1]
    elif kind == "k2any":
        m1 = np.full(S0.shape[0], NEG); m2 = np.full(S0.shape[0], NEG); U = np.full(S0.shape, NEG)
        for e in range(W):
            x = np.where(V[:, e], S0[:, e], NEG); m2 = np.where(V[:, e], np.maximum(m2, np.minimum(m1, x)), m2); m1 = np.where(V[:, e], np.maximum(m1, x), m1); U[:, e] = m2
    elif kind.startswith("pers_"):   # override: score >= par (t_high) flags outright; else the k-of-n rule `kind[5:]`
        base = {"cons2": memory(ST, s, "cons2"), "k2of3": memory(ST, s, "k2of3"), "k2any": memory(ST, s, "k2any")}[kind[5:]]; return np.where(s >= par, BIG, base)
    elif kind == "recal":
        cm = np.cumsum(S0, 1) / np.maximum(np.cumsum(V, 1), 1); pm = np.concatenate([np.full((S0.shape[0], 1), mu0), cm[:, :-1]], 1); U = Sm - par * pm
    else: raise KeyError(kind)
    return from_mat(ST, U, len(s))
def cert_flags(ST, qual, Y):
    """endoscopy is certified if an EARLIER endoscopy of the patient (strictly before in time order) with qual True lies <= Y years before it"""
    Q = np.where(ST["VAL"], qual[ST["PADS"]], False); W = ST["W"]; TY = ST["TY"]; d = TY[:, :, None] - TY[:, None, :]
    strict = np.tril(np.ones((W, W), bool), -1)[None] & ST["VAL"][:, None, :] & ST["VAL"][:, :, None]
    return from_mat(ST, (strict & Q[:, None, :] & (d <= Y + 1e-9)).any(2).astype(float), len(qual)) > 0
# ================================================================== per-endoscopy features of one outer fold (training: inner out-of-fold, test: outer)
def efeat(ST, xi, xc, xl):
    IDX, MSK = ST["IDX"], ST["MSK"]; E = IDX.shape[0]; R = np.arange(E); Pi = np.where(MSK, xi[np.where(MSK, IDX, 0)], np.nan); Pc = np.where(MSK, xc[np.where(MSK, IDX, 0)], np.nan); Pl = np.where(MSK, xl[np.where(MSK, IDX, 0)], np.nan)
    srt = np.sort(np.where(MSK, Pi, -1.0), 1)[:, ::-1]; top = np.argmax(np.where(MSK, Pi, -1.0), 1); z = lg(Pi)
    ef = {"pi": srt[:, 0], "zi": lg(srt[:, 0]), "pi2": np.where(srt[:, 1] >= 0, srt[:, 1], np.nan), "zmean": np.nanmean(z, 1), "ztop2": np.where(srt[:, 1] >= 0, (lg(srt[:, 0]) + lg(np.maximum(srt[:, 1], 1e-6))) / 2, lg(srt[:, 0])),
          "pc_top": Pc[R, top], "zc_top": lg(Pc[R, top]), "pl_top": Pl[R, top], "zl_top": lg(Pl[R, top]), "zl_max": lg(np.nanmax(Pl, 1)), "zc_max": lg(np.nanmax(Pc, 1))}
    ef["zi2"] = np.where(np.isfinite(ef["pi2"]), lg(np.where(np.isfinite(ef["pi2"]), ef["pi2"], 0.5)), ef["zi"]); return ef
class Ctx: pass
def gate(s1, s2, cuts):
    """three-band gate: low (s1 < t_low), high (s1 > c*), middle: top sample sequenced and flagged if its score s2 >= t_seq (rv7 r49 rule; cuts may be arrays)"""
    tl, cs, ts = cuts; band = np.where(tl > cs, np.where(s1 > cs, 2, 0), np.where(s1 < tl, 0, np.where(s1 > cs, 2, 1))); flag = np.where(band == 1, s2 >= ts, band == 2); return flag, (band == 1).astype(int), band
def cuts_sample(c, ql, qh, qs):
    """tb_common.triage cut-offs at (t_low sensitivity, t_high specificity, t_seq sensitivity) re-derived on the fold's training samples (inner predictions)"""
    s, yc, li = c.s_img_tr, c.yc_tr, c.s_late_tr; tl = thr_sens(s[yc], ql); cs = c_star(s[~yc], qh); mid = (s >= tl) & (s <= cs) if tl <= cs else np.zeros(len(s), bool); sel = mid & yc
    ts = thr_sens(li[sel], qs) if sel.sum() >= 5 else thr_sens(li[yc], qs); return tl, cs, ts
def cuts_endo(c, e1, e2, ql, qh, qs):
    trm = c.tr_e; P = c.elab == 1; tl = thr_sens(e1[trm & P], ql); cs = c_star(e1[trm & ~P], qh); mid = (e1 >= tl) & (e1 <= cs) if tl <= cs else np.zeros(len(e1), bool); sel = trm & P & mid
    ts = thr_sens(e2[sel], qs) if sel.sum() >= 5 else thr_sens(e2[trm & P], qs); return tl, cs, ts
def make_ctx(S, ST, j, k, INFO):
    c = Ctx(); c.S, c.ST, c.j, c.k = S, ST, j, k; r = REPS[j]; te_s = S["FO"][:, j] == k; xs = {}
    for m in MODELS:
        x = np.full(n, np.nan); s = S["I"][m][(r, k)]; x[s.index.values] = s.values; x[te_s] = S["O"][m][te_s, j]; xs[m] = x
    c.ef = efeat(ST, xs["L-IMG"], xs["L-CNV"], xs["L-LATE"]); c.te_e = S["EF"][:, j] == k; c.tr_e = ~c.te_e; c.trP = ST["PF"][:, j] != k; c.isP = S["plab"] == 1; c.elab = S["elab"]; c.nsz = S["nsz"]
    si = S["I"]["L-IMG"][(r, k)]; tri = si.index.values[S["popm"][si.index.values]]; c.s_img_tr = si.reindex(tri).values; c.yc_tr = y[tri] == 1; c.s_late_tr = S["I"]["L-LATE"][(r, k)].reindex(tri).values
    i = j * 10 + (k - 1); c.cuts = (INFO["t_low"][i], INFO["c_star"][i], INFO["t_seq"][i]); ef = c.ef
    c.r49_flag, c.r49_ns, c.r49_band = gate(ef["pi"], ef["pl_top"], c.cuts); c.D = c.patient_det(c.r49_flag); c.cache = {}; return c
def _pdet(self, f): return int((((self.S["INC"] @ f.astype(float)) > 0)[self.trP & self.isP]).sum())
Ctx.patient_det = _pdet
def pat_scores(c, u): return np.where(c.ST["VAL"], u[c.ST["PADS"]], -np.inf).max(1)
def thr_for(c, pm):
    pp = np.sort(pm[c.trP & c.isP])[::-1]; pp = pp[pp > NEG / 2]
    if len(pp) == 0: return np.inf
    d = min(c.D, len(pp)); return pp[d - 1] if d >= 1 else np.inf
def pauc(pos, neg, fmax=0.2):
    thr = np.unique(np.concatenate([pos, neg]))[::-1]; tpr = np.r_[0.0, [(pos >= t).mean() for t in thr]]; fpr = np.r_[0.0, [(neg >= t).mean() for t in thr]]; o = np.argsort(fpr, kind="stable"); fpr, tpr = fpr[o], tpr[o]
    m = fpr <= fmax; x, v = list(fpr[m]), list(tpr[m])
    if x[-1] < fmax: nx = np.argmax(fpr > fmax); f0, f1, t0, t1 = fpr[nx - 1], fpr[nx], tpr[nx - 1], tpr[nx]; x.append(fmax); v.append(t0 + (t1 - t0) * (fmax - f0) / (f1 - f0))
    return float(np.trapz(v, x) / fmax)
# ================================================================== logistic stacker (cross-validated inside the training patients)
def lr_scores(c, cols1, extra2, tag):
    """B1 stacked scores: LR1 on cols1, LR2 on cols1 + extra2; training endoscopies get 5-fold patient-grouped out-of-fold linear predictors, held-out ones the all-training fit"""
    if tag in c.cache: return c.cache[tag]
    X1 = np.column_stack(cols1); X2 = np.column_stack(cols1 + extra2); E = len(X1); trm = c.tr_e; lab = c.elab; epat = c.S["epat"]; out = []
    up = np.unique(epat[trm]); pl = np.array([lab[trm & (epat == p)].max() for p in up]); rng = np.random.RandomState(1000 * c.j + c.k); ordp = np.r_[rng.permutation(up[pl == 1]), rng.permutation(up[pl == 0])]
    fo = {p: i % 5 for i, p in enumerate(ordp)}; ef = np.array([fo.get(p, -1) for p in epat])
    def fit(X, tr):
        mu = X[tr].mean(0); sd = X[tr].std(0); sd[sd == 0] = 1; m = LogisticRegression(C=1.0, max_iter=500).fit((X[tr] - mu) / sd, lab[tr]); return lambda Z: m.decision_function((Z - mu) / sd)
    for X in (X1, X2):
        e = np.zeros(E); f = fit(X, trm); e[~trm] = f(X[~trm])
        for q in range(5): tr = trm & (ef != q); te = trm & (ef == q); e[te] = fit(X, tr)(X[te])
        out.append(e)
    c.cache[tag] = tuple(out); return c.cache[tag]
def b1_variant(c, variant):
    ef, ST = c.ef, c.ST; z = (ef["zi"],); c1 = [ef["zi"]]
    if variant in ("full", "p53"): c1 += [ST["DYS"], ST["AGE"], ST["SEX"]]
    if variant == "p53": c1 += [ST["P53AB"].astype(float), (~ST["P53AV"]).astype(float)]
    return lr_scores(c, c1, [ef["zc_top"]], "b1_" + variant)
# ================================================================== bases, candidate builders
def sc(label, params, u, ns): return {"kind": "score", "label": label, "params": params, "u": u, "ns": ns}
def gc(label, params, flag, ns, extra=None): return {"kind": "gate", "label": label, "params": params, "flag": flag, "ns": ns, "extra": extra or {}}
def gate_scores(c):
    """literal and R49-aligned gate scores: literal = L-IMG logit, replaced in the middle band by the sequenced top sample's L-LATE logit; aligned shifts that logit so that
    the gate's own decision (L-LATE >= t_seq) is the score >= logit c* (R49 == score >= logit c*)"""
    ef = c.ef; tl, cs, ts = c.cuts; mid = c.r49_band == 1; return np.where(mid, ef["zl_top"], ef["zi"]), np.where(mid, lg(cs) + ef["zl_top"] - lg(ts), ef["zi"])
def b1_al(c):
    e1, e2 = b1_variant(c, "full"); cu = cuts_endo(c, e1, e2, *NOMINAL); fl, ns, band = gate(e1, e2, cu); return np.where(band == 1, cu[1] + e2 - cu[2], e1), ns, cu
def base(c, name):
    ef = c.ef; z = np.zeros(len(ef["zi"]), int)
    if name == "img": return ef["zi"], z
    if name == "img_mean": return ef["zmean"], z
    if name == "img_top2": return ef["ztop2"], z
    if name == "conf_al":   # sequence every endoscopy at or above t_low; score 0 = the gate's own flag boundary (L-LATE >= t_seq), below t_low: negative, ordered by L-IMG
        tl, cs, ts = c.cuts; mid = ef["pi"] >= tl; return np.where(mid, ef["zl_top"] - lg(ts), -1.0 - (lg(tl) - ef["zi"])), mid.astype(int)
    if name == "late_max": return ef["zl_max"], c.nsz.copy()
    if name == "gate_lit": return gate_scores(c)[0], c.r49_ns
    if name == "gate_al": return gate_scores(c)[1], c.r49_ns
    if name == "b1_al": s, ns, _ = b1_al(c); return s, ns
    raise KeyError(name)
MEMFAMS = ["none", "mean", "ew", "last2", "pers", "k2any", "k2of3", "cons2", "k2any_o", "k2of3_o", "recal"]
def mem_cands(c, s, ns, mf, bname):
    npq = s[c.tr_e & (c.elab == 0)]; ths = [("q90", np.quantile(npq, 0.90)), ("q95", np.quantile(npq, 0.95)), ("q975", np.quantile(npq, 0.975))]; L = []
    if mf == "none": L.append(sc("none", {}, s, ns))
    elif mf == "mean": L.append(sc("mean", {}, memory(c.ST, s, "mean"), ns))
    elif mf == "ew":
        for h in (1, 2, 4): L.append(sc(f"ew h={h}y", {"h": h}, memory(c.ST, s, "ew", h), ns))
    elif mf == "ew_ext":
        for h in (0.5, 8, 16): L.append(sc(f"ew h={h}y", {"h": h}, memory(c.ST, s, "ew", h), ns))
    elif mf == "last2": L.append(sc("mean of last 2", {}, memory(c.ST, s, "last2"), ns))
    elif mf == "pers":
        for nm, th in ths: L.append(sc(f"consecutive-2 or t_high={nm}", {"t_high": nm}, memory(c.ST, s, "pers_cons2", th), ns))
    elif mf in ("k2any", "k2of3", "cons2"): L.append(sc({"k2any": "2-of-n anywhere", "k2of3": "2-of-last-3", "cons2": "consecutive-2"}[mf], {}, memory(c.ST, s, mf), ns))
    elif mf in ("k2any_o", "k2of3_o"):
        for nm, th in ths: L.append(sc(f"{'2-of-n anywhere' if mf == 'k2any_o' else '2-of-last-3'} or t_high={nm}", {"t_high": nm}, memory(c.ST, s, "pers_" + mf[:-2], th), ns))
    elif mf == "recal":
        mu0 = float(npq.mean())
        for lam in (0.5, 1.0): L.append(sc(f"change from own mean, lambda={lam}", {"lambda": lam}, memory(c.ST, s, "recal", lam, mu0), ns))
    return L
def f_score(base_name, mf):
    def b(c): s, ns = base(c, base_name); return mem_cands(c, s, ns, mf, base_name)
    return b
def cert_candidates(c, kind):
    """A2: after a sequenced endoscopy with CN-quiet top sample (kind 'cn') or after a p53-normal endoscopy (kind 'p53'), the gate's H&E cut-offs are raised to at least the
    training-specificity threshold for Y years"""
    ef, ST = c.ef, c.ST; tl, cs, ts = c.cuts; L = [gc("no certificate (= R49)", {}, c.r49_flag, c.r49_ns)]; seq = c.r49_band == 1
    tqs = {50: np.percentile(ef["zc_top"][c.tr_e & (c.elab == 0) & seq], 50), 75: np.percentile(ef["zc_top"][c.tr_e & (c.elab == 0) & seq], 75)} if kind == "cn" else {0: 0}
    thD = {d: c_star(c.s_img_tr[~c.yc_tr], d) for d in (0.85, 0.95)}
    for tq, tv in tqs.items():
        qual = (seq & (ef["zc_top"] < tv)) if kind == "cn" else (ST["P53AV"] & ~ST["P53AB"])
        for d, thd in thD.items():
            for Y in (2, 4):
                ce = cert_flags(ST, qual, Y); fl, ns, _ = gate(ef["pi"], ef["pl_top"], (np.where(ce, np.maximum(tl, thd), tl), np.where(ce, np.maximum(cs, thd), cs), ts))
                L.append(gc(f"{'t_quiet=p' + str(tq) + ', ' if kind == 'cn' else ''}raise to {int(d * 100)}%-spec, Y={Y}y", {"t_quiet_pct": tq, "delta_spec": d, "Y": Y}, fl, ns, {"share_endoscopies_certified": float(ce.mean())}))
    return L
def f_A2(kind): return lambda c: cert_candidates(c, kind)
def f_A3(c):
    ef = c.ef; tl, cs, ts = c.cuts; trm = c.tr_e; P = c.elab == 1; two = c.nsz >= 2; L = [gc("no agreement rule (= R49)", {}, c.r49_flag, c.r49_ns)]; hi = c.r49_band == 2
    for qm in (0.5, 0.8):
        tmid = thr_sens(ef["pi2"][trm & P & two], qm)
        for qc in (0.8, 0.9):
            tc = thr_sens(ef["pc_top"][trm & P], qc); need = hi & two & ~(ef["pi2"] >= tmid); fl = np.where(need, ef["pc_top"] >= tc, c.r49_flag); ns = np.where(need, 1, c.r49_ns)
            L.append(gc(f"gate: t_mid={int(qm * 100)}% sens, t_cnv={int(qc * 100)}% sens", {"base": "gate", "q_mid": qm, "q_cnv": qc}, fl, ns))
    cand_t = np.unique(np.quantile(ef["pi"][trm], np.linspace(0.3, 0.999, 120)))
    for qm in (0.5, 0.8):
        tmid = thr_sens(ef["pi2"][trm & P & two], qm)
        for qc in (0.8, 0.9):
            tc = thr_sens(ef["pc_top"][trm & P], qc); best = None
            for t in cand_t[::-1]:
                f0 = ef["pi"] >= t; need = f0 & two & ~(ef["pi2"] >= tmid); fl = np.where(need, ef["pc_top"] >= tc, f0)
                if c.patient_det(fl) >= c.D: best = (t, fl, need.astype(int)); break
            if best is None: t = cand_t[0]; f0 = ef["pi"] >= t; need = f0 & two & ~(ef["pi2"] >= tmid); best = (t, np.where(need, ef["pc_top"] >= tc, f0), need.astype(int))
            L.append(gc(f"H&E alone: t_mid={int(qm * 100)}% sens, t_cnv={int(qc * 100)}% sens", {"base": "img", "q_mid": qm, "q_cnv": qc}, best[1], best[2]))
    return L
def f_B3(c):
    ef = c.ef; tl, cs, ts = c.cuts; mid = c.r49_band == 1; dist = np.abs(ef["zl_top"] - lg(ts)); L = [gc("no abstention (= R49)", {}, c.r49_flag, c.r49_ns, {"indeterminate_endoscopies": 0})]
    for sh in (0.10, 0.20):
        eps = np.quantile(dist[c.tr_e & mid], sh) if (c.tr_e & mid).any() else 0.0; ind = mid & (dist <= eps); L.append(gc(f"indeterminate share {int(sh * 100)}%", {"share": sh}, c.r49_flag & ~ind, c.r49_ns, {"ind": ind}))
    return L
def f_C1(c):
    L = []
    for g in GRID27: fl, ns, _ = gate(c.ef["pi"], c.ef["pl_top"], cuts_sample(c, *g)); L.append(gc(f"cut-offs {int(g[0] * 100)}/{int(g[1] * 100)}/{int(g[2] * 100)}", {"ql": g[0], "qh": g[1], "qs": g[2]}, fl, ns))
    return L
def f_B1(variant):
    def b(c):
        e1, e2 = b1_variant(c, variant); L = []
        for g in GRID27: fl, ns, _ = gate(e1, e2, cuts_endo(c, e1, e2, *g)); L.append(gc(f"cut-offs {int(g[0] * 100)}/{int(g[1] * 100)}/{int(g[2] * 100)}", {"ql": g[0], "qh": g[1], "qs": g[2]}, fl, ns))
        return L
    return b
def f_A12(c):
    """A1 + A2: EW memory of the gate-aligned score, with a CN-quiet certificate that raises the score threshold (shifts the score down by delta) for Y years"""
    ef, ST = c.ef, c.ST; s, ns = base(c, "gate_al"); seq = c.r49_band == 1; zq = ef["zc_top"][c.tr_e & (c.elab == 0) & seq]; L = []
    for h in (1, 2, 4):
        u = memory(ST, s, "ew", h)
        for tq in (50, 75):
            qual = seq & (ef["zc_top"] < np.percentile(zq, tq))
            for dl in (0.5, 1.0):
                for Y in (2, 4): L.append(sc(f"ew h={h}y, t_quiet=p{tq}, +{dl} logit, Y={Y}y", {"h": h, "t_quiet_pct": tq, "delta": dl, "Y": Y}, u - dl * cert_flags(ST, qual, Y), ns))
    return L
def f_B2A2(c):
    ef, ST = c.ef, c.ST; s, ns, cu = b1_al(c); e1, e2 = b1_variant(c, "full"); fl, _, band = gate(e1, e2, cu); seq = band == 1; zq = ef["zc_top"][c.tr_e & (c.elab == 0) & seq]; L = []
    for h in (1, 2, 4):
        u = memory(ST, s, "ew", h)
        for tq in (50, 75):
            qual = seq & (ef["zc_top"] < np.percentile(zq, tq))
            for dl in (0.5, 1.0):
                for Y in (2, 4): L.append(sc(f"B1 + ew h={h}y, t_quiet=p{tq}, +{dl} logit, Y={Y}y", {"h": h, "t_quiet_pct": tq, "delta": dl, "Y": Y}, u - dl * cert_flags(ST, qual, Y), ns))
    return L
def f_A12p(c):
    """A1 + p53: EW memory of the gate-aligned score with a p53-normal certificate (score shifted down by delta for Y years after an endoscopy with a normal p53 result)"""
    ef, ST = c.ef, c.ST; s, ns = base(c, "gate_al"); qual = ST["P53AV"] & ~ST["P53AB"]; L = []
    for h in (1, 2, 4):
        u = memory(ST, s, "ew", h)
        for dl in (0.5, 1.0):
            for Y in (2, 4): L.append(sc(f"ew h={h}y, p53-normal certificate, +{dl} logit, Y={Y}y", {"h": h, "delta": dl, "Y": Y}, u - dl * cert_flags(ST, qual, Y), ns))
    return L
def f_A4(c):
    """A4: confirm the high band by sequencing (literature idea 3): no outright slide-only flag; every endoscopy at or above t_low has its top sample sequenced and is flagged if L-LATE >= t_seq"""
    L = []
    for ql in (0.90, 0.95, 0.98):
        for qs in (0.70, 0.80, 0.90):
            tl, cs, ts = cuts_sample(c, ql, 0.90, qs); fl, ns, _ = gate(c.ef["pi"], c.ef["pl_top"], (tl, np.inf, ts)); L.append(gc(f"t_low {int(ql * 100)}%, t_seq {int(qs * 100)}%", {"ql": ql, "qs": qs}, fl, ns))
    return L
def f_R49x(c):
    return [gc("R49 gate rebuilt on this L-IMG (nominal cut-off rules)", {}, c.r49_flag, c.r49_ns)]
def load_refit_var(var):
    """stored L-CNV with the refitted L-IMG of rv9_refit.R (outer and inner), same folds; L-LATE = mean of the two"""
    O, I, FO = _load0("their"); D_ = M + f"/review_round9/refit/{var}"; need = lambda fs, k: fs if len(fs) == k else (_ for _ in ()).throw(AssertionError((len(fs), k)))
    oo, fo = TC._outer(need(sorted(glob.glob(D_ + "/outer_img_rep_*.csv")), 10)); ii = TC._inner(need(sorted(glob.glob(D_ + "/inner_img_rep_*_fold_*.csv")), 100)); assert (fo == FO).all()
    O["L-IMG"] = oo; O["L-LATE"] = (O["L-CNV"] + O["L-IMG"]) / 2; I["L-IMG"] = ii; I["L-LATE"] = {k: (I["L-CNV"][k] + ii[k].reindex(I["L-CNV"][k].index)) / 2 for k in I["L-CNV"]}
    for k in I["L-CNV"]: assert ii[k].index.sort_values().equals(I["L-CNV"][k].index.sort_values()), k
    return O, I, FO
import glob
_load0 = load
# family registry: name -> builder; groups run as one cluster job each
FAMS = {}
for _b in ("img", "gate_lit", "gate_al", "b1_al"):
    for _m in MEMFAMS: FAMS[f"A1:{_b}:{_m}"] = f_score(_b, _m)
for _b in ("img_mean", "img_top2"):
    for _m in ("none", "mean", "ew"): FAMS[f"G:{_b}:{_m}"] = f_score(_b, _m)
for _m in ("none", "mean", "ew"): FAMS[f"G:late_max:{_m}"] = f_score("late_max", _m)
for _m in ("none", "mean", "ew"): FAMS[f"A4:conf_al:{_m}"] = f_score("conf_al", _m)
FAMS["A1:gate_al:ew_ext"] = f_score("gate_al", "ew_ext")
FAMS.update({"A12p": f_A12p, "A4": f_A4})
FAMS.update({"A2:cn": f_A2("cn"), "A2:p53": f_A2("p53"), "A3": f_A3, "B3": f_B3, "C1": f_C1, "B1:full": f_B1("full"), "B1:img": f_B1("img"), "B1:p53": f_B1("p53"), "A12": f_A12, "B2A2": f_B2A2})
GROUPS = {"A1img": [f for f in FAMS if f.startswith("A1:img:")], "A1gl": [f for f in FAMS if f.startswith("A1:gate_lit:")], "A1ga": [f for f in FAMS if f.startswith("A1:gate_al:")],
          "B2": [f for f in FAMS if f.startswith("A1:b1_al:")], "G": [f for f in FAMS if f.startswith("G:")], "A2": ["A2:cn", "A2:p53"], "A3": ["A3"], "B3": ["B3"], "C1": ["C1"],
          "B1": ["B1:full", "B1:img", "B1:p53"], "A12": ["A12", "B2A2"], "X": ["A1:gate_al:ew_ext", "A12p", "A4", "A4:conf_al:none", "A4:conf_al:mean", "A4:conf_al:ew"]}
# ================================================================== the nested engine
def run_families(S, ST, fams, INFO, builders=None, cols=None):
    """for every outer (repeat, fold): build every family's candidates on the fold, choose the operating threshold / candidate on the training patients, apply to the held-out patients"""
    builders = builders or FAMS; E = S["E"]; R = {}
    for f in fams: R[f] = dict(sel={cr: (np.zeros((E, 10), bool), np.zeros((E, 10), int)) for cr in ("np", "pauc")}, chosen={"np": [], "pauc": []}, labels=None, kinds=None, params=None, F=None, NS=None, U=None, infeasible={"np": 0, "pauc": 0}, trainstats=[], extra={})
    for j in range(10):
        for k in range(1, 11):
            c = make_ctx(S, ST, j, k, INFO); te = c.te_e
            for f in fams:
                cands = builders[f](c); Rf = R[f]; nC = len(cands)
                if Rf["F"] is None:
                    Rf["labels"] = [x["label"] for x in cands]; Rf["kinds"] = [x["kind"] for x in cands]; Rf["params"] = [x["params"] for x in cands]
                    Rf["F"] = np.zeros((nC, E, 10), bool); Rf["NS"] = np.zeros((nC, E, 10), np.int8); Rf["U"] = np.full((nC, E, 10), np.nan, np.float32); Rf["IND"] = np.zeros((nC, E, 10), bool)
                tr_stats = []
                for q, x in enumerate(cands):
                    if x["kind"] == "score":
                        pm = pat_scores(c, x["u"]); t = thr_for(c, pm); flag = x["u"] >= t; Rf["U"][q][:, j][te] = x["u"][te]; pa = pauc(pm[c.trP & c.isP], pm[c.trP & ~c.isP]) if np.isfinite(t) else -1
                    else: flag = x["flag"]; pa = -1
                    det = c.patient_det(flag); inc = c.S["INC"] @ flag.astype(float) > 0; npf = int((inc & ~c.isP & c.trP).sum()); nfe = int((flag & c.tr_e & (c.elab == 0)).sum()); nsm = float(x["ns"][c.tr_e].mean())
                    tr_stats.append((det, npf, nfe, nsm, pa)); Rf["F"][q][:, j][te] = flag[te]; Rf["NS"][q][:, j][te] = x["ns"][te]
                    if "ind" in x.get("extra", {}): Rf["IND"][q][:, j][te] = x["extra"]["ind"][te]
                ts_ = np.array(tr_stats); feas = ts_[:, 0] >= c.D
                all_score = all(x["kind"] == "score" for x in cands)
                for cr in ("np", "pauc"):
                    if cr == "pauc" and nC > 1 and all_score: pick = max(range(nC), key=lambda q: (ts_[q, 4], -q))
                    else:
                        if feas.any(): order = sorted([q for q in range(nC) if feas[q]], key=lambda q: (ts_[q, 1], ts_[q, 2], ts_[q, 3], q))
                        else: Rf["infeasible"][cr] += 1; order = sorted(range(nC), key=lambda q: (-ts_[q, 0], ts_[q, 1], ts_[q, 2], q))
                        pick = order[0]
                    Rf["chosen"][cr].append(int(pick)); Rf["sel"][cr][0][:, j][te] = Rf["F"][pick][:, j][te]; Rf["sel"][cr][1][:, j][te] = Rf["NS"][pick][:, j][te]
                Rf["trainstats"].append(ts_[Rf["chosen"]["np"][-1]][:4].tolist() + [c.D])
    return R
def save_group(R, name):
    arrs = {}; meta = {}
    for f, Rf in R.items():
        key = f.replace(":", "_")
        arrs[key + "__F"] = Rf["F"]; arrs[key + "__NS"] = Rf["NS"]; arrs[key + "__U"] = Rf["U"]; arrs[key + "__IND"] = Rf["IND"]
        for cr in ("np", "pauc"): arrs[f"{key}__sel_{cr}_F"] = Rf["sel"][cr][0]; arrs[f"{key}__sel_{cr}_NS"] = Rf["sel"][cr][1]; arrs[f"{key}__chosen_{cr}"] = np.array(Rf["chosen"][cr])
        arrs[key + "__train"] = np.array(Rf["trainstats"])
        meta[f] = {"key": key, "labels": Rf["labels"], "kinds": Rf["kinds"], "params": Rf["params"], "infeasible_folds": Rf["infeasible"]}
    np.savez_compressed(f"{R9}/grp_{name}.npz", **arrs); json.dump(meta, open(f"{R9}/grp_{name}.json", "w"), default=str); print("RV9 DONE", name, flush=True)
def prep_state(S=None):
    S = S or setup(); ST = statics(S); _, _, INFO = TC.triage(S["O"], S["I"], S["FO"], S["popm"], 0.95, 0.90); return S, ST, INFO
def refs(S):
    """round-8 comparators: R49 (the stored three-band gate) and P1 at its 90% training cell"""
    INFO = TC.triage(S["O"], S["I"], S["FO"], S["popm"], 0.95, 0.90)[2]; F, NSr, _ = r49_calls(S["O"], INFO, S["FO"], S["es"]); X = S["OUTS"]["img"]; Z = np.zeros((S["E"], 10), int)
    return {"R49": (F, NSr), "P1 90%": (X >= thr_arr(S, "img", 0.90)[0], Z)}
# ================================================================== tasks
res = {"task": TASK}
if TASK in ("m_B4a", "m_B4b"):
    var = "b4" + TASK[-1]; load = lambda src: load_refit_var(var); S, ST, INFO = prep_state(); bl = {f"B4{TASK[-1]}:R49x": f_R49x, f"B4{TASK[-1]}:C1": f_C1}; R = run_families(S, ST, list(bl), INFO, builders=bl); save_group(R, "B4" + TASK[-1])
elif TASK.startswith("m_") and TASK[2:] in GROUPS:
    S, ST, INFO = prep_state(); t0 = time.time(); R = run_families(S, ST, GROUPS[TASK[2:]], INFO); save_group(R, TASK[2:]); print("seconds", round(time.time() - t0, 1))
elif TASK == "check_r49":   # smoke: engine R49 == r49_calls on held-out endoscopies, sample-level cut-off re-derivation == tb_common.triage
    S, ST, INFO = prep_state(); RF = refs(S); bad = 0; badc = 0; badg = 0; gv = 0
    for j in range(10):
        for k in range(1, 11):
            c = make_ctx(S, ST, j, k, INFO); te = c.te_e; bad += int((c.r49_flag[te] != RF["R49"][0][te, j]).sum()); badc += int(np.max(np.abs(np.array(cuts_sample(c, *NOMINAL)) - np.array(c.cuts))) > 1e-12)
            _, ga = gate_scores(c); fl = ga >= lg(c.cuts[1]); badg += int((fl[te] != RF["R49"][0][te, j]).sum())
    D = pdraws(S); out = {}
    for nm, (F, NSr) in RF.items():
        A = patient_arrays(S, F, NSr); p = parts(S, A, np.ones((1, S["P_"]))); out[nm] = {"caught": r2(32 * met(p, "caught_share")[0]), "np_flagged": r2(43 * met(p, "np_flagged_share")[0]), "seq": r3(met(p, "seq_per_patient")[0])}
    res["check"] = {"engine_vs_r49_calls_mismatches": bad, "cuts_sample_mismatching_folds": badc, "aligned_gate_threshold_mismatches": badg, "observed": out, "expected": {"R49": [31.1, 31.8], "P1 90%": [31.1, 36.0]},
                    "reproduced": bool(bad == 0 and badc == 0 and badg == 0 and abs(out["R49"]["caught"] - 31.1) < 0.051 and abs(out["R49"]["np_flagged"] - 31.8) < 0.051 and abs(out["P1 90%"]["caught"] - 31.1) < 0.051 and abs(out["P1 90%"]["np_flagged"] - 36.0) < 0.051)}
    print(json.dumps(res["check"], indent=1)); dump9(res, "check_r49" + ("_smoke" if SMOKE else ""))
# ------------------------------------------------------------------ merge helpers
import glob
BN = {"conf_al": "gate with the high band also sequenced (score 0 = flag boundary)", "img": "H&E alone (max L-IMG)", "gate_lit": "gate score, literal", "gate_al": "gate score, R49-aligned", "b1_al": "stacked B1 gate score, aligned", "img_mean": "H&E mean over samples", "img_top2": "H&E mean of top-2 samples", "late_max": "max L-LATE (sequence all samples)"}
MN = {"none": "threshold only", "mean": "running mean", "ew": "exponentially weighted mean", "last2": "mean of last 2", "pers": "consecutive-2 or t_high", "k2any": "2-of-n anywhere", "k2of3": "2-of-last-3",
      "cons2": "consecutive-2", "k2any_o": "2-of-n anywhere or t_high", "k2of3_o": "2-of-last-3 or t_high", "recal": "change from own mean"}
PRET = {"A2:cn": "A2 copy-number certificate (gate)", "A2:p53": "A2' p53-normal certificate (gate)", "A3": "A3 agreement within endoscopy", "B3": "B3 abstention in middle band", "C1": "control: R49 with cut-offs re-tuned by the round-9 rule",
        "B1:full": "B1 stacked gate (L-IMG, grade, age, sex; + L-CNV in middle band)", "B1:img": "B1-img control: stacked gate on L-IMG only (endoscopy-level cut-offs)", "B1:p53": "B1 + p53 (aberrant, missing)",
        "A12p": "A1+A2': EW memory of gate score + p53-normal certificate", "A4": "A4 confirm the high band by sequencing (no outright slide-only flag)", "A12": "A1+A2: EW memory of gate score + copy-number certificate", "B2A2": "B1+A1+A2: stacked gate + EW memory + copy-number certificate", "R49x": "R49 rebuilt on refitted L-IMG"}
def pretty(f):
    if f in PRET: return PRET[f]
    if f.startswith("B4"): return {"B4a": "B4(a) refit, non-progressor weight 2", "B4b": "B4(b) refit, NDBE samples only"}[f[:3]] + (", R49 rules rebuilt" if f.endswith("R49x") else ", cut-offs re-tuned by the round-9 rule")
    p = f.split(":"); return f"{ {'A1': 'A1', 'A4': 'A4 confirm-by-sequencing'}.get(p[0], 'Sample aggregation') } | {BN[p[1]]} | {MN.get(p[2], 'exponentially weighted, longer half-lives (0.5, 8, 16 y)')}"
def is_control(f): return f in ("C1", "B1:img", "R49x") or f.endswith(":none")
def load_groups(names=None):
    G = {}
    for p in sorted(glob.glob(R9 + "/grp_*.json")):
        nm = os.path.basename(p)[4:-5]
        if names and nm not in names: continue
        meta = json.load(open(p)); z = np.load(p[:-5] + ".npz")
        for f, m in meta.items(): G[f] = dict(meta=m, z=z, group=nm)
    return G
def build_eval(S, D):
    one = np.ones((1, S["P_"])); nPp = int((S["plab"] == 1).sum()); nNp = int((S["plab"] == 0).sum())
    MF = {"caught": lambda p: nPp * met(p, "caught_share"), "np_flagged": lambda p: nNp * met(p, "np_flagged_share"), "flags_py": lambda p: met(p, "flags_per_np_py"), "endo_spec": lambda p: met(p, "endo_spec"),
          "seq": lambda p: met(p, "seq_per_patient"), "endo_sens": lambda p: met(p, "endo_sens")}
    def evalm(F, NS, em=None):
        A = patient_arrays(S, F, NS, em); pb = parts(S, A, D); po = parts(S, A, one); return A, {k: f(pb) for k, f in MF.items()}, {k: float(f(po)[0]) for k, f in MF.items()}, po, pb
    def matched_eval(U, NS, Dm, refc):
        isP = S["plab"] == 1; INC = S["INC"]; B = Dm.shape[0]; Wp = Dm * isP; Wn = Dm * ~isP; PYn = Wn @ S["PY"]; NEv = S["NE"]; r = np.arange(B); out = {k: np.zeros((B, 10)) for k in ("caught", "np_flagged", "flags_py", "endo_spec", "seq")}
        for j in range(10):
            Pm = np.where(INC > 0, U[None, :, j], -np.inf).max(1); cth = np.unique(Pm[isP])[::-1]; Fc = U[:, j][:, None] >= cth[None, :]; NFc = INC @ Fc.astype(float); FLc = (NFc > 0).astype(float)
            cau = Wp @ FLc; ok = cau >= refc[:, j][:, None] - 1e-9; idx = np.where(ok.any(1), ok.argmax(1), len(cth) - 1); nsp = INC @ NS[:, j].astype(float)
            out["caught"][:, j] = nPp * cau[r, idx] / Wp.sum(1); out["np_flagged"][:, j] = nNp * (Wn @ FLc)[r, idx] / Wn.sum(1); nfn = (Wn @ NFc)[r, idx]
            out["flags_py"][:, j] = nfn / PYn; out["endo_spec"][:, j] = 1 - nfn / (Wn @ NEv); out["seq"][:, j] = (Dm @ nsp) / Dm.sum(1)
        return {k: v.mean(1) for k, v in out.items()}
    return MF, evalm, matched_eval, nPp, nNp
def contrast(ob, dr, rob, rdr):
    o = {k: dstat(ob[k] - rob[k], dr[k] - rdr[k]) for k in ("np_flagged", "caught", "flags_py", "endo_spec", "seq")}; o["np_flagged"]["ci99_bonferroni"] = [r3(x) for x in np.percentile(dr["np_flagged"] - rdr["np_flagged"], [0.4, 99.6])]; return o
if TASK == "merge":
    S = setup(); ST = statics(S); D = pdraws(S); RF = refs(S); MF, evalm, matched_eval, nPp, nNp = build_eval(S, D); G = load_groups(); isP = S["plab"] == 1
    REF = {}
    for nm, (F, NSr) in RF.items(): A, dr, ob, po, pb = evalm(F, NSr); REF[nm] = dict(A=A, dr=dr, ob=ob, po=po, pb=pb, F=F, NS=NSr)
    def record(name, fam, kind, F, NS, matched=None, extra=None):
        if matched is None: A, dr, ob, po, pb = evalm(F, NS)
        else: ob = {k: float(v[-1]) for k, v in matched.items()}; dr = {k: v[:-1] for k, v in matched.items()}; A = None
        rec = {"name": name, "family": fam, "kind": kind, "values": {k: r3(v) for k, v in ob.items()}, "vs_R49": contrast(ob, dr, REF["R49"]["ob"], REF["R49"]["dr"]), "vs_P1_90": contrast(ob, dr, REF["P1 90%"]["ob"], REF["P1 90%"]["dr"])}
        if matched is None:
            one = (S["NE"] == 1) & isP; ns_ = int(one.sum()); rec["caught_progressors_with_one_endoscopy"] = {"n": m5(ns_), "caught": r2(float(A["FL"][one].mean(0).mean() * ns_)) if ns_ else None, "R49": r2(float(REF["R49"]["A"]["FL"][one].mean(0).mean() * ns_)) if ns_ else None}
            dn = nNp * per_rep(po, "np_flagged_share")[0] - nNp * per_rep(REF["R49"]["po"], "np_flagged_share")[0]; rec["between_repeat_delta_np_flagged_vs_R49"] = {"mean": r3(dn.mean()), "sd": r3(dn.std(ddof=1)), "min": r3(dn.min()), "max": r3(dn.max())}
            dc = nPp * per_rep(po, "caught_share")[0] - nPp * per_rep(REF["R49"]["po"], "caught_share")[0]; rec["between_repeat_delta_caught_vs_R49"] = {"mean": r3(dc.mean()), "sd": r3(dc.std(ddof=1)), "min": r3(dc.min()), "max": r3(dc.max())}
        if extra: rec.update(extra)
        return rec
    methods = []; thcount = {}; infeas = {}
    for f, g in G.items():
        key = g["meta"]["key"]; z = g["z"]; labs = g["meta"]["labels"]; kinds = g["meta"]["kinds"]; chn = z[f"{key}__chosen_np"]; chp = z[f"{key}__chosen_pauc"]
        thcount[f] = {labs[q]: int((chn == q).sum()) for q in range(len(labs))}; infeas[f] = g["meta"]["infeasible_folds"]; trs = z[f"{key}__train"]
        ex = {"selected_candidate_counts_over_100_folds": thcount[f], "infeasible_folds_np": infeas[f]["np"], "mean_training_detection_target": r3(trs[:, 4].mean()), "mean_training_detection_method": r3(trs[:, 0].mean())}
        if f == "B3":
            IND = z[f"{key}__IND"]; tot = np.zeros(10); totP = np.zeros(10)
            for j in range(10):
                for k in range(1, 11): q = chn[j * 10 + k - 1]; te = S["EF"][:, j] == k; tot[j] += IND[q][te, j].sum(); totP[j] += (IND[q][te, j] & (S["elab"][te] == 1)).sum()
            ex["indeterminate_endoscopies_per_repeat_mean"] = r2(tot.mean()); ex["indeterminate_progressor_endoscopies_per_repeat_mean"] = r2(totP.mean())
        methods.append(record(pretty(f) + " [nested]", f, "nested", z[f"{key}__sel_np_F"], z[f"{key}__sel_np_NS"], extra=ex))
        if len(labs) > 1 and all(k_ == "score" for k_ in kinds):
            methods.append(record(pretty(f) + " [nested, pAUC selection]", f, "nested_pauc", z[f"{key}__sel_pauc_F"], z[f"{key}__sel_pauc_NS"], extra={"selected_candidate_counts_over_100_folds": {labs[q]: int((chp == q).sum()) for q in range(len(labs))}}))
        Fg = z[f"{key}__F"]; NSg = z[f"{key}__NS"].astype(int)
        for q in range(len(labs)):
            if len(labs) > 1: methods.append(record(f"{pretty(f)} | {labs[q]}", f, "grid_point", Fg[q], NSg[q], extra={"optimistic": "fixed hyper-parameter, picked after seeing the evaluation (threshold still from training)"}))
        U = z[f"{key}__U"]
        for q in ({int(np.bincount(chn).argmax())} | (set(range(len(labs))) if len(labs) > 1 else set())):
            if kinds[q] != "score": continue
            mt = matched_eval(U[q].astype(float), NSg[q], np.vstack([D, np.ones((1, S["P_"]))]), np.vstack([REF["R49"]["pb"]["caught"], REF["R49"]["po"]["caught"]]))
            methods.append(record(f"{pretty(f)} | {labs[q]} [evaluation-matched]" if len(labs) > 1 else f"{pretty(f)} [evaluation-matched]", f, "matched", None, None, matched=mt, extra={"optimistic": "threshold set on the evaluation patients so that progressor patients caught equals R49's in each repeat",
                                                                                                                                                                                   "modal_candidate_in_training": labs[int(np.bincount(chn).argmax())] == labs[q]}))
    # R49 re-tuned with the round-9 rule and the references themselves
    for nm in REF: methods.append({"name": nm + " (reference)", "family": nm, "kind": "reference", "values": {k: r3(v) for k, v in REF[nm]["ob"].items()}})
    nest = [m for m in methods if m["kind"] in ("nested", "nested_pauc")]
    def rk(m): d = m["vs_R49"]; return (0 if d["caught"]["delta"] >= -0.5 else 1, d["np_flagged"]["delta"], -d["caught"]["delta"])
    ranking = sorted(nest, key=rk); top3 = [m["family"] for m in ranking if not is_control(m["family"]) and m["kind"] == "nested"][:3]; top3k = [m["name"] for m in ranking if not is_control(m["family"]) and m["kind"] == "nested"][:3]
    chk = {"R49": REF["R49"]["ob"], "P1_90": REF["P1 90%"]["ob"], "reproduced": bool(abs(REF["R49"]["ob"]["caught"] - 31.1) < 0.051 and abs(REF["R49"]["ob"]["np_flagged"] - 31.8) < 0.051 and abs(REF["P1 90%"]["ob"]["np_flagged"] - 36.0) < 0.051 and abs(REF["P1 90%"]["ob"]["caught"] - 31.1) < 0.051)}
    res["merge"] = {"methods": methods, "ranking": [m["name"] for m in ranking], "top3_families": top3, "top3_names": top3k, "check_round8": chk, "draws": int(len(D)), "n_progressor_patients": nPp, "n_non_progressor_patients": nNp,
                    "note": "every row is paired with R49 and P1 at 90% on the same patient-bootstrap draws (rv8 pdraws, RandomState(0)); 'nested' rows: operating threshold and hyper-parameters chosen on training patients' inner out-of-fold predictions, "
                            "performance on held-out patients; grid_point and matched rows are optimistic (apparent); 99.2% = Bonferroni (6 families) percentile interval"}
    dump9(res, "merge")
# ================================================================== F0: are flagged non-progressors more likely to show later dysplasia? (descriptive)
if TASK == "f0":
    S = setup(); L = pd.read_csv(R9 + "/later.csv", dtype={"Patient": str}).set_index("Patient"); RF = refs(S); X = S["OUTS"]; nsz = np.repeat(S["nsz"][:, None], 10, 1)
    RF["P3 90%"] = (X["late"] >= thr_arr(S, "late", 0.90)[0], nsz); isN = S["plab"] == 0; up = S["up"]; Lr = L.reindex(up); assert Lr.cls.notna().all()
    anyr = Lr.any_after.fillna(False).astype(bool).values; mx = pd.to_numeric(Lr.max_after, errors="coerce").values
    grp = np.where(~anyr, "no later database record", np.where(np.isnan(mx), "records after, none with a confirmed grade", np.where(mx == 0, "later NDBE only", np.where(mx == 1, "later IND", np.where(mx == 2, "later LGD", "later HGD or worse")))))
    nps = np.where(isN)[0]; ind = (np.nan_to_num(mx, nan=-1) >= 1); lgd = (np.nan_to_num(mx, nan=-1) >= 2); R = {"n_non_progressor_patients": int(isN.sum()), "group_sizes": {g: m5(int((grp[nps] == g).sum())) for g in sorted(set(grp[nps]))}}
    out = {}
    for nm, (F, NSr) in RF.items():
        A = patient_arrays(S, F, NSr); flag = A["FL"].sum(1) >= 5; o = {"ever_flagged_in_ge5_of_10_repeats": int(flag[nps].sum()), "tables": {}}
        for tn, sub in (("all_43", np.ones(len(up), bool)), ("with_any_later_record", anyr)):
            for lab, later in (("later_IND_or_worse", ind), ("later_LGD_or_worse", lgd)):
                m = nps[sub[nps]]; a = int((flag[m] & later[m]).sum()); b = int((flag[m] & ~later[m]).sum()); c_ = int((~flag[m] & later[m]).sum()); d_ = int((~flag[m] & ~later[m]).sum())
                orr, p = fisher_exact([[a, b], [c_, d_]]); o["tables"][f"{tn}:{lab}"] = {"flagged_later": m5(a), "flagged_not_later": m5(b), "not_flagged_later": m5(c_), "not_flagged_not_later": m5(d_), "fisher_p_two_sided": r3(p),
                                                                                        "odds_ratio": r3(orr) if np.isfinite(orr) and not any(0 < x < 5 for x in (a, b, c_, d_)) else "<5 cell", "share_later_among_flagged": r3(a / max(a + b, 1)) if a >= 5 or a == 0 else "<5", "share_later_among_not_flagged": r3(c_ / max(c_ + d_, 1)) if c_ >= 5 or c_ == 0 else "<5"}
        fa = nps[flag[nps]]; nf = nps[~flag[nps]]; a = int(anyr[fa].sum()); c_ = int(anyr[nf].sum())
        orr, p = fisher_exact([[a, len(fa) - a], [c_, len(nf) - c_]]); o["any_later_record"] = {"flagged_with_record": m5(a), "flagged_without": m5(len(fa) - a), "not_flagged_with": m5(c_), "not_flagged_without": m5(len(nf) - c_), "fisher_p_two_sided": r3(p),
                                                                                                "share_flagged": r3(a / max(len(fa), 1)) if a >= 5 or a == 0 else "<5", "share_not_flagged": r3(c_ / max(len(nf), 1)) if c_ >= 5 or c_ == 0 else "<5"}
        o["flags_per_patient_year_by_later_group"] = {}
        for g in ["no later database record", "records after, none with a confirmed grade", "later NDBE only", "later IND or worse", "later LGD or worse"]:
            m = nps[(ind[nps] if g == "later IND or worse" else lgd[nps] if g == "later LGD or worse" else grp[nps] == g)]
            o["flags_per_patient_year_by_later_group"][g] = {"patients": m5(len(m)), "flagged_endoscopies_per_patient_year": r3(float(A["NF"][m].sum(0).mean() / S["PY"][m].sum())) if len(m) >= 5 else "<5",
                                                             "patients_flagged_in_ge5_repeats": m5(int(flag[m].sum())) if len(m) >= 5 else "<5"}
        out[nm] = o
    R["rules"] = out; R["definition"] = ("later disease = highest Barrett's database confirmed grade after the last sequenced endoscopy (round-7 lookup B2: records > 14 days after the latest exact sample date); IND, LGD, HGD+. "
                                         "'ever flagged' = flagged at >= 1 pre-event endoscopy in >= 5 of 10 repeats (stored held-out predictions); rates = flagged non-progressor endoscopies (mean over repeats) per patient-year (first sequenced endoscopy to last sequenced endoscopy); "
                                         "groups < 5 patients shown as '<5'"); res["F0"] = R; dump9(res)
# ================================================================== ICC of non-progressor scores
elif TASK == "icc":
    S = setup(); RF = refs(S); X = S["OUTS"]; es = S["es"]; E = S["E"]; INFO = TC.triage(S["O"], S["I"], S["FO"], S["popm"], 0.95, 0.90)[2]; TL, CSt, TS = cutoff_arrays(INFO, S["FO"]); s0 = np.array([s[0] for s in es])
    _, _, GB = r49_calls(S["O"], INFO, S["FO"], es); zi = lg(X["img"]); zl = lg(X["top"]); cs = CSt[s0]; ts = TS[s0]; SC = {"logit max L-IMG": zi, "gate score, literal": np.where(GB == 1, zl, zi), "gate score, R49-aligned (0 = R49 flag boundary)": np.where(GB == 1, lg(cs) + zl - lg(ts), zi) - lg(cs)}
    pid = S["pidx"]; isP = S["plab"] == 1
    def icc1(vals):
        k = len(vals); ni = np.array([len(v) for v in vals]); N = ni.sum(); m = np.concatenate(vals).mean(); mi = np.array([v.mean() for v in vals]); msb = (ni * (mi - m) ** 2).sum() / (k - 1)
        msw = sum(((v - v.mean()) ** 2).sum() for v in vals) / (N - k); n0 = (N - (ni ** 2).sum() / N) / (k - 1); return (msb - msw) / (msb + (n0 - 1) * msw), msb, msw
    rng = np.random.RandomState(0); out = {}; BN_ = 2000
    for nm, V in SC.items():
        out[nm] = {}
        for cl, pm in (("non_progressor", ~isP), ("progressor", isP)):
            pats = np.where(pm)[0]; per = [[V[pid == i, j] for i in pats] for j in range(10)]; icc_r = [icc1(p)[0] for p in per]; comps = np.array([icc1(p)[1:] for p in per]).mean(0)
            bs = [np.mean([icc1([per[j][i] for i in rng.choice(len(pats), len(pats))])[0] for j in range(10)]) for _ in range(BN_ if not SMOKE else 200)]
            out[nm][cl] = {"icc1_mean_over_repeats": r3(np.mean(icc_r)), "ci95": pct(bs), "icc1_range_over_repeats": [r3(min(icc_r)), r3(max(icc_r))], "between_patient_mean_square": r3(comps[0]), "within_patient_mean_square": r3(comps[1]), "patients": len(pats),
                           "endoscopies": int(np.isin(pid, pats).sum())}
    ceil = {}
    for nm, (F, NSr) in RF.items():
        A = patient_arrays(S, F, NSr); N_ = ~isP; nf = A["NF"][N_]; fl = nf > 0
        ceil[nm] = {"np_patients_ever_flagged": r2(fl.sum(0).mean()), "np_flagged_once_only": r2(((nf == 1)).sum(0).mean()), "np_flagged_twice": r2((nf == 2).sum(0).mean()), "np_flagged_3_or_more": r2((nf >= 3).sum(0).mean()),
                    "np_flagged_at_least_twice": r2((nf >= 2).sum(0).mean()), "share_of_flagged_np_with_only_one_flagged_endoscopy": r3(float(((nf == 1).sum(0) / np.maximum(fl.sum(0), 1)).mean()))}
    res["ICC"] = {"scores": out, "persistence_ceiling_np_patients": ceil, "estimator": "one-way random-effects ICC(1) for unequal group sizes (ANOVA), per repeat on the held-out endoscopy scores, averaged over the 10 repeats; patient bootstrap of the repeat-mean ICC (2,000 draws, RandomState(0), resampled patients as separate clusters)",
                  "reading": "ICC near 1: a patient's score is nearly fixed, so averaging or persistence rules cannot separate non-progressors; near 0: independent errors, so they can"}; dump9(res)
# ================================================================== cumulative false-flag curves by endoscopy index
def curve(S, F, D, ks=(1, 2, 3, 4, 5, 6)):
    POS = S["POS"]; INC = S["INC"]; isP = S["plab"] == 1; NEp = S["NE"]; o = {}
    for kk in ks:
        FL = ((INC @ (F & (POS < kk)[:, None]).astype(float)) > 0).astype(float)
        for rst in (False, True):
            m = (NEp >= kk) if rst else np.ones(S["P_"], bool); r = {}
            for cl, pm in (("np", ~isP), ("prog", isP)):
                W1 = np.ones((1, S["P_"])) * pm * m; Wd = D * pm * m; sh = lambda W: np.nanmean(np.where(W.sum(1)[:, None] > 0, (W @ FL) / np.maximum(W.sum(1), 1e-9)[:, None], np.nan), 1)
                r[cl] = (sh(W1)[0], sh(Wd), int((pm * m).sum()))
            o[(kk, rst)] = r
    return o
if TASK == "curves":
    S = setup(); D = pdraws(S); RF = refs(S); M_ = json.load(open(f"{OUT}/merge.json"))["merge"]; G = load_groups(); mem = [m for m in M_["ranking"] if m.endswith("[nested]") and not any(m.startswith(x) for x in ("control", "B1-img"))]
    cand = [m for m in M_["methods"] if m["kind"] == "nested" and not is_control(m["family"]) and m["family"].split(":")[-1] not in ("none",) and (m["family"].startswith("A1:") or m["family"] in ("A12", "B2A2"))]
    cand = sorted(cand, key=lambda m: (0 if m["vs_R49"]["caught"]["delta"] >= -0.5 else 1, m["vs_R49"]["np_flagged"]["delta"])); best = cand[0]; g = G[best["family"]]; key = g["meta"]["key"]
    RF[best["name"]] = (g["z"][f"{key}__sel_np_F"], g["z"][f"{key}__sel_np_NS"]); out = {}
    for nm, (F, NSr) in RF.items():
        cv = curve(S, F, D); out[nm] = {}
        for (kk, rst), r in cv.items():
            e = {}
            for cl in ("np", "prog"):
                ob, dr, npat = r[cl]; ok = npat >= 5
                e[cl] = {"patients": m5(npat), "share": r3(ob) if ok else "<5", "ci95": pct(dr[np.isfinite(dr)]) if ok else "<5"}
            out[nm].setdefault(f"k={kk}", {})["restricted_to_patients_with_ge_k_endoscopies" if rst else "all_patients_first_k_endoscopies"] = e
    dlt = {}
    for nm in out:
        if nm == "R49": continue
        dlt[nm] = {}
        for kk in (1, 2, 3, 4, 5, 6):
            c1 = curve(S, RF[nm][0], D, (kk,)); c0 = curve(S, RF["R49"][0], D, (kk,)); a = c1[(kk, False)]["np"]; b = c0[(kk, False)]["np"]; dlt[nm][f"k={kk}"] = dstat(a[0] - b[0], a[1] - b[1])
    res["curves"] = {"best_memory_rule": best["name"], "curves": out, "delta_np_share_vs_R49_all_patients": dlt, "note": "share of patients flagged at >= 1 of their first k pre-event endoscopies (patients with fewer than k endoscopies contribute all of theirs; "
                     "'restricted' keeps only patients with >= k endoscopies, the matched first-k window); held-out predictions, 10-repeat mean; patient bootstrap CIs; groups < 5 patients '<5'"}; dump9(res)
# ================================================================== robustness rows of the top-3 methods
def use_set9(V):
    """rv7_round7.py use_set (unchanged in substance): rebind the sample-level globals of tb_common and of this module to a variant set"""
    g = {"n": len(V), "ids": V.Sample.values.astype(str), "y": V.y.values.astype(int), "pat": V.Patient.values.astype(str), "PRE": V.pre.astype(str).eq("True").values, "NDBE": V.ndbe.astype(str).eq("True").values}
    g["ix"] = pd.Series(np.arange(g["n"]), index=g["ids"]); g["ENDO"] = (pd.Series(g["pat"]) + "|" + V.Endoscopy.astype(str).values).values
    for k_, v_ in g.items(): setattr(TC, k_, v_); globals()[k_] = v_
def load_r52a():
    """rv7c_gate_rob.py load_var for the round-7 refit r52a (+ same-block samples): refit outer and inner predictions of both models"""
    D_ = M + "/review_round7/refit/r52a"; f = lambda p_, k_: (sorted(glob.glob(f"{D_}/{p_}")), k_)
    def need_(t): assert len(t[0]) == t[1], (len(t[0]), t[1]); return t[0]
    oc, oi = need_(f("outer_cnv_their_rep_*.csv", 10)), need_(f("outer_img_rep_*.csv", 10)); ic, ii = need_(f("inner_cnv_their_rep_*_fold_*.csv", 100)), need_(f("inner_img_rep_*_fold_*.csv", 100)); O, I = {}, {}
    O["L-CNV"], FO = TC._outer(oc); O["L-IMG"], F2 = TC._outer(oi); assert (FO == F2).all(); O["L-LATE"] = (O["L-CNV"] + O["L-IMG"]) / 2
    I["L-CNV"] = TC._inner(ic); I["L-IMG"] = TC._inner(ii); I["L-LATE"] = {k_: (I["L-CNV"][k_] + I["L-IMG"][k_].reindex(I["L-CNV"][k_].index)) / 2 for k_ in I["L-CNV"]}; return O, I, FO
def sm_variant(V):
    """samples table of the variant set: stored rows unchanged, added rows get their patient's event, age (via months before final) and sex; p53 missing"""
    base = SM.set_index("Sample"); assert (V.Sample.values[:len(SM)] == SM.Sample.values).all(); cols = list(SM.columns); out = []
    for r_ in V.itertuples():
        if r_.Sample in base.index: out.append(base.loc[r_.Sample].to_dict() | {"Sample": r_.Sample}); continue
        pr = SM[SM.Patient == r_.Patient]; d = {c_: np.nan for c_ in cols}
        if len(pr): afin = (pr.age + pr.mbf / 12).mean(); extra = {"study_number": pr.study_number.iloc[0], "Status": pr.Status.iloc[0], "tev": pr.tev.iloc[0], "fallback_endpoint": pr.fallback_endpoint.iloc[0], "event": pr.event.iloc[0], "age": afin - float(r_.mbf) / 12, "sex_M": pr.sex_M.iloc[0]}
        else: extra = {"study_number": "new_" + r_.Patient, "Status": "P" if int(r_.y) else "NP", "tev": 0.0, "fallback_endpoint": False, "event": 1, "age": float(SM.age.mean()), "sex_M": float(SM.sex_M.mean())}   # patients added by R52: event endoscopy (months before final 0) as event; age / sex are cohort means (used only by stacked B1 rows, not run here)
        d.update({"Sample": r_.Sample, "Patient": r_.Patient, "y": int(r_.y), "Pathology": r_.Pathology, "mbf": float(r_.mbf), "pre": r_.pre == "True", "ndbe": r_.ndbe == "True", "time": float(r_.time), **extra}); out.append(d)
    return pd.DataFrame(out)[cols]
ROWS = {"ndbe": "pre-event NDBE samples only (cut-offs and comparators re-chosen on NDBE training samples; round-7c row)", "ge2": "endoscopies with >= 2 pre-event samples (cut-offs re-chosen on them; round-7c row)",
        "tis_lt256": "< 256 tissue tiles (endoscopy by its top-L-IMG sample)", "tis_ge256": ">= 256 tissue tiles (endoscopy by its top-L-IMG sample)",
        "r52a": "R52(a): + 17 same-block samples, refit (round-7 refit predictions, all cut-offs and rules re-derived)", "tis_lt256pure": "< 256 tissue tiles (all samples of the endoscopy in the stratum)", "tis_ge256pure": ">= 256 tissue tiles (all samples of the endoscopy in the stratum)"}
if TASK.startswith("rob_"):
    row = TASK[4:]; top = json.load(open(f"{OUT}/merge.json"))["merge"]["top3_families"]; top_extra = [x for x in os.environ.get("RV9_EXTRA", "").split(",") if x]; fams = list(dict.fromkeys(top + top_extra)); em = None; Gm = None
    if row in ("ndbe", "ge2"):
        if row == "ndbe": PRE = PRE & NDBE
        else: cnt = pd.Series(ENDO[PRE]).value_counts(); PRE = PRE & (pd.Series(ENDO).map(cnt).fillna(0).values >= 2)
        S, ST, INFO = prep_state(); R = run_families(S, ST, fams, INFO); Gm = {f: (R[f]["sel"]["np"][0], R[f]["sel"]["np"][1]) for f in fams}; RF = refs(S)
    elif row == "r52a":
        V = pd.read_csv(M + "/review_round7/sets/r52a.csv", dtype={"Sample": str, "Patient": str, "Endoscopy": str}); SM = sm_variant(V); use_set9(V); load = lambda src: load_r52a(); S, ST, INFO = prep_state(); R = run_families(S, ST, fams, INFO)
        Gm = {f: (R[f]["sel"]["np"][0], R[f]["sel"]["np"][1]) for f in fams}; RF = refs(S)
    else:
        S, ST, INFO = prep_state(); RF = refs(S); G = load_groups(); Gm = {f: (G[f]["z"][G[f]["meta"]["key"] + "__sel_np_F"], G[f]["z"][G[f]["meta"]["key"] + "__sel_np_NS"]) for f in fams}
        TILES = pd.read_csv(HZ + "/slide_desc.csv", dtype={"Sample": str}).set_index("Sample").reindex(ids).tissue_tiles.values.astype(float); lt = "lt" in row; st = (TILES < 256) if lt else (TILES >= 256); mimg = S["O"]["L-IMG"].mean(1)
        top_s = np.array([s[np.argmax(mimg[s])] for s in S["es"]]); em = np.array([st[s].all() for s in S["es"]]) if row.endswith("pure") else st[top_s]
    D = pdraws(S); MF, evalm, matched_eval, nPp, nNp = build_eval(S, D); REF = {nm: evalm(F, NSr, em) for nm, (F, NSr) in RF.items()}; out = {"R49": {k: r3(v) for k, v in REF["R49"][2].items()}, "P1 90%": {k: r3(v) for k, v in REF["P1 90%"][2].items()}}
    for f in fams:
        F, NSr = Gm[f]; A, dr, ob, po, pb = evalm(F, NSr, em); out[pretty(f)] = {"family": f, "values": {k: r3(v) for k, v in ob.items()}, "vs_R49": contrast(ob, dr, REF["R49"][2], REF["R49"][1]), "vs_P1_90": contrast(ob, dr, REF["P1 90%"][2], REF["P1 90%"][1])}
    pop = {"endoscopies": int(S["E"] if em is None else em.sum()), "progressor_patients": nPp, "non_progressor_patients": nNp}
    res["ROB"] = {"row": row, "definition": ROWS[row], "population": pop, "families": fams, "results": out, "note": "same nested procedure in the row's own population (ndbe, ge2: all training-patient choices redone; tissue rows: held-out calls of the main run, evaluation restricted); R49 and P1 90% re-derived in the row"}; dump9(res)
# ================================================================== refit reproduction check (rv9_refit.R, weights 1, all rows, against the stored predictions)
if TASK == "refit_check":
    D_ = M + "/review_round9/refit/b4w1"; o = pd.read_csv(D_ + "/outer_img_rep_01.csv", dtype={"Sample": str}); so = pd.read_csv(M + "/cv/preds/cfg_02_rep_01.csv", dtype={"Sample": str}); mo = o.merge(so, on="Sample"); assert len(mo) == len(o) == len(so)
    i = pd.read_csv(D_ + "/inner_img_rep_01_fold_01.csv", dtype={"Sample": str}); si = pd.read_csv(HZ + "/inner/img_rep_01_fold_01.csv", dtype={"Sample": str}); mi = i.merge(si, on=["Sample", "inner_fold"]); assert len(mi) == len(i) == len(si)
    res["refit_check"] = {"outer_rep_01": {"n": int(len(mo)), "max_abs_pred_diff": float(np.abs(mo.pred_x - mo.pred_y).max()), "folds_equal": bool((mo.fold_x == mo.fold_y).all())},
                          "inner_rep_01_fold_01": {"n": int(len(mi)), "max_abs_pred_diff": float(np.abs(mi.pred_x - mi.pred_y).max()), "max_abs_lambda_diff": float(np.abs(mi.lambda_x - mi.lambda_y).max())}}
    res["refit_check"]["passed"] = bool(res["refit_check"]["outer_rep_01"]["max_abs_pred_diff"] <= 1e-5 and res["refit_check"]["outer_rep_01"]["folds_equal"] and res["refit_check"]["inner_rep_01_fold_01"]["max_abs_pred_diff"] <= 1e-5)
    dump9(res)

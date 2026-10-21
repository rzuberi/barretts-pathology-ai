"""Exploration E2 (docs/paper_explore_E2.md; rules paper/reviews_round7/decisions/EXPLORE_RULES.md): copy number as a VETO on H&E flags.
Exploratory, not confirmatory. Stored outer / inner out-of-fold predictions are re-scored with the round-8 helpers (rv8_landscape.py, executed into a
namespace, unchanged); copy-number features are built from stored blocks / tiles and QDNAseq 50-kb outputs (read-only). Every threshold and model is set on
the outer fold's training patients (inner out-of-fold H&E scores decide which training endoscopies H&E flags); evaluation is on held-out patients.
Row-level files only under killcoyne_mm/explore_E2/ on the cluster; aggregates (no identifiers, no dates) -> results/paper_final/explore_E2/<TASK>.json.
TASK = feat | desc | fixed_<P1|R49h|R49hm> | en_<fs>_<pop>_<rep> | en_eval | drift | stab | best"""
import os, sys, json, glob, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); TASK = os.environ["TASK"]
os.environ["TASK"] = "__ex_e2_import__"; NS8 = {"__file__": os.path.join(HERE, "rv8_landscape.py"), "__name__": "rv8_landscape"}
try: exec(compile(open(NS8["__file__"]).read(), NS8["__file__"], "exec"), NS8)
except SystemExit: pass
os.environ["TASK"] = TASK
g8 = NS8; TC = g8["TC"]; setup = g8["setup"]; patient_arrays = g8["patient_arrays"]; pdraws = g8["pdraws"]; lstats = g8["lstats"]; cell_result = g8["cell_result"]
r49_calls = g8["r49_calls"]; ci2 = g8["ci2"]; dstat = g8["dstat"]; lg = g8["lg"]; m5 = g8["m5"]; r2 = g8["r2"]; r3 = g8["r3"]; sheet = g8["sheet"]; thr_sens = g8["thr_sens"]
T = g8["T"]; M = g8["M"]; KD = g8["KD"]; ids = g8["ids"]; n = g8["n"]; y = g8["y"]; pat = g8["pat"]; SM = g8["SM"]; REPS = g8["REPS"]; NB = g8["NB"]
OUT = T + "/results/paper_final/explore_E2"; XD = M + "/explore_E2"
if os.environ.get("EX_NB"): NB = int(os.environ["EX_NB"]); OUT = XD + "/smoke_json"
for d in (OUT, XD): os.makedirs(d, exist_ok=True)
SWG = "/mnt/scratche/fast/fmlab/datasets/imaging/SWGCohort"
def dump(res, name=None):
    json.dump(res, open(f"{OUT}/{name or TASK}.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)); print("EXE2 DONE", name or TASK, flush=True)
def J(p):
    try: return json.load(open(p))
    except FileNotFoundError: return None
LOSS = ["17p", "9p", "18q", "5q", "4q", "13q", "3p"]; GAIN = ["8q", "7p", "20q"]
# ------------------------------------------------------------------ copy-number features (row-level, cluster only)
def arm_names(cols):
    out = []
    for c in cols:
        ch, rg = c.split(":"); s = int(rg.split("-")[0]); out.append(f"{ch}{'p' if s == 1 else 'q'}")
    return out
def tile_feats(F, tag):
    """F: tiles table (Sample, qc_pass, cx_raw, 587 tiles, 44 arms; relative copy number ~1). Deviations relative to the sample's median tile."""
    cols = [c for c in F.columns if c not in ("Sample", "qc_pass", "cx_raw")]; W = np.array([int(c.split(":")[1].split("-")[1]) - int(c.split(":")[1].split("-")[0]) for c in cols])
    tc = [c for c, w in zip(cols, W) if w < 6e6]; ac = [c for c, w in zip(cols, W) if w >= 6e6]; assert len(tc) == 587 and len(ac) == 44, (len(tc), len(ac))
    X = F[tc].values.astype(float); A = F[ac].values.astype(float); med = np.nanmedian(X, 1, keepdims=True); D = X / med - 1; DA = A / med - 1
    an = arm_names(ac); tchr = np.array([c.split(":")[0] for c in tc]); ts = np.array([int(c.split(":")[1].split("-")[0]) for c in tc])
    # tile -> arm index (arm columns give arm ranges)
    ar = [(c.split(":")[0], int(c.split(":")[1].split("-")[0]), int(c.split(":")[1].split("-")[1])) for c in ac]; t2a = np.full(len(tc), -1)
    for i, (ch, s, e) in enumerate(ar): t2a[(tchr == ch) & (ts >= s) & (ts <= e)] = i
    DT = np.where(t2a[None, :] >= 0, X / np.where(t2a >= 0, A[:, np.maximum(t2a, 0)], np.nan) - 1, np.nan)   # tile relative to its arm (focal)
    o = pd.DataFrame({"Sample": F.Sample.values}); ok = np.isfinite(D)
    for t in (0.05, 0.10, 0.15): o[f"{tag}fga{int(t * 100):02d}"] = np.where(ok, np.abs(D) > t, False).sum(1) / ok.sum(1)
    for t in (0.05, 0.10): o[f"{tag}narm{int(t * 100):02d}"] = (np.abs(np.nan_to_num(DA)) > t).sum(1)
    o[f"{tag}armabs"] = np.nansum(np.abs(DA), 1); o[f"{tag}sd"] = np.nanstd(D, 1); o[f"{tag}maxdev"] = np.nanmax(np.abs(D), 1)
    o[f"{tag}focal10"] = (np.abs(np.nan_to_num(DT)) > 0.10).sum(1); o[f"{tag}cx"] = F.cx_raw.values.astype(float); o[f"{tag}qc"] = F.qc_pass.astype(str).str.upper().eq("TRUE").values
    ai = {a: i for i, a in enumerate(an)}; oac = np.zeros(len(F))
    for a in LOSS: o[f"{tag}arm_{a}"] = DA[:, ai[a]]; oac += np.maximum(0, -np.nan_to_num(DA[:, ai[a]]))
    for a in GAIN: o[f"{tag}arm_{a}"] = DA[:, ai[a]]; oac += np.maximum(0, np.nan_to_num(DA[:, ai[a]]))
    o[f"{tag}oac"] = oac
    ARM = pd.DataFrame(DA, columns=[f"{tag}A_{a}" for a in an]); ARM.insert(0, "Sample", F.Sample.values)
    return o, ARM, D
def mapd_of(path):
    try:
        d = pd.read_csv(path, sep="\t"); v = d.iloc[:, 4].values.astype(float); ch = d.iloc[:, 1].astype(str).values; ok = np.isfinite(v)
        v, ch = v[ok], ch[ok]; df = np.diff(v)[ch[1:] == ch[:-1]]; return float(np.median(np.abs(df))), float(np.std(v))
    except Exception: return np.nan, np.nan
def load_feat():
    return pd.read_csv(XD + "/features.csv", dtype={"Sample": str}).set_index("Sample").reindex(ids)
# ------------------------------------------------------------------ evaluation scaffolding
def build():
    """round-8 setup plus: R49 (test + training analogue), P1 90% (test + training), top-H&E samples per endoscopy (outer per repeat; inner per fold)"""
    S = setup(); O, I, FO, es, E = S["O"], S["I"], S["FO"], S["es"], S["E"]
    band, call, info = TC.triage(O, I, FO, S["popm"], 0.95, 0.90); GC, NS49, GB = r49_calls(O, info, FO, es)
    TOPO = np.full((E, 10, 2), -1)
    for e, s in enumerate(es):
        for j in range(10): od = s[np.argsort(-O["L-IMG"][s, j], kind="mergesort")]; TOPO[e, j, 0] = od[0]; TOPO[e, j, 1] = od[1] if len(od) > 1 else -1
    TR = S["TR"]; TRX = {}
    t90 = np.zeros((E, 10))
    for j, r in enumerate(REPS):
        for k in range(1, 11):
            tr = TR[(j, k)]; tidx = tr["idx"]; li = I["L-IMG"][(r, k)]; t = thr_sens(tr["img"][tr["lab"] == 1], 0.90); t90[S["EF"][:, j] == k, j] = t
            i0 = j * 10 + (k - 1); tl, cs, ts = info["t_low"][i0], info["c_star"][i0], info["t_seq"][i0]; im = tr["img"]
            gb = np.where(im > cs, 2, 0) if tl > cs else np.where(im < tl, 0, np.where(im > cs, 2, 1)); gc = (gb == 2) | ((gb == 1) & (tr["top"] >= ts))
            top = np.full((len(tidx), 2), -1)
            for q, e in enumerate(tidx):
                s = es[e]; od = s[np.argsort(-li.reindex(s).values, kind="mergesort")]; top[q, 0] = od[0]; top[q, 1] = od[1] if len(od) > 1 else -1
            TRX[(j, k)] = {"p1": im >= t, "gb": gb, "gc": gc, "top": top, "pid": S["pidx"][tidx], "lab": tr["lab"], "idx": tidx, "t90": t}
    P1F = S["OUTS"]["img"] >= t90
    S.update(GC=GC, NS49=NS49, GB=GB, TOPO=TOPO, TRX=TRX, P1F=P1F, info=info, band=band)
    S["D"] = pdraws(S); S["A_R49"] = patient_arrays(S, GC, NS49); S["A_P1"] = patient_arrays(S, P1F, np.zeros((E, 10), int))
    return S
def base_of(S, b):
    """test (BF, BNS, VE high-band-like, VEmid) and per-fold training (bf, ve) for base b"""
    E = S["E"]; Z = np.zeros((E, 10), bool)
    if b == "P1": return S["P1F"], np.zeros((E, 10), int), S["P1F"], Z, {kk: (v["p1"], v["p1"], np.zeros(len(v["p1"]), bool)) for kk, v in S["TRX"].items()}
    if b == "R49h": return S["GC"], S["NS49"], S["GB"] == 2, Z, {kk: (v["gc"], v["gb"] == 2, np.zeros(len(v["gc"]), bool)) for kk, v in S["TRX"].items()}
    if b == "R49hm": return S["GC"], S["NS49"], S["GB"] == 2, S["GC"] & (S["GB"] == 1), {kk: (v["gc"], v["gc"], v["gc"] & (v["gb"] == 1)) for kk, v in S["TRX"].items()}
    raise ValueError(b)
def vs_test(S, f, k):
    """veto score per endoscopy x repeat from sample scores f: (n,) fixed or (n, 10) per repeat; k = samples sequenced (top-H&E first)"""
    TOP = S["TOPO"]; V = np.full((S["E"], 10), np.nan)
    for j in range(10):
        fj = f if f.ndim == 1 else f[:, j]; v = fj[TOP[:, j, 0]]
        if k == 2: v2 = np.where(TOP[:, j, 1] >= 0, fj[np.maximum(TOP[:, j, 1], 0)], -np.inf); v = np.fmax(v, v2)
        V[:, j] = v
    return np.where(np.isfinite(V) | (V == -np.inf), V, np.inf)   # missing feature -> never vetoes
def vs_train(S, ftr, top, k):
    v = ftr[top[:, 0]]
    if k == 2: v = np.fmax(v, np.where(top[:, 1] >= 0, ftr[np.maximum(top[:, 1], 0)], -np.inf))
    return np.where(np.isfinite(v) | (v == -np.inf), v, np.inf)
RULES = ("strict", "loss1", "endo95", "cn90")
def pick_t(bf, ve, v, pid, lab, rule):
    """highest veto threshold on training endoscopies: strict = no training progressor patient lost; loss1 = at most 1 lost; endo95 = >= 95% of vetoable
    training progressor endoscopies kept; cn90 = the copy-number-alone threshold (90% of all training progressor endoscopies, any H&E score)"""
    if rule == "cn90":
        x = v[(lab == 1) & np.isfinite(v)]; return thr_sens(x, 0.90) if len(x) else -np.inf
    if rule == "endo95":
        x = v[ve & bf & (lab == 1)]; return thr_sens(x, 0.95) if len(x) else -np.inf
    sv = np.where(bf, np.where(ve, v, np.inf), -np.inf); P = np.unique(pid[lab == 1]); ps = np.array([sv[pid == p].max() for p in P]); det = np.sort(ps[ps > -np.inf])[::-1]
    need = len(det) - (0 if rule == "strict" else 1)
    if need <= 0: return np.inf
    return det[need - 1]
def veto_V(S, b, V, vtr, rule, k=1, ve_test=None, ve_train=None):
    """core: V (E, 10) held-out veto scores, vtr[(j, k)] training-endoscopy veto scores (aligned with TRX idx); optional extra vetoability masks.
    Returns F, NS (sequencing of the k top samples at every vetoable flagged endoscopy; the middle band's top sample is already sequenced), TH"""
    BF, BNS, VEh, VEm, TRB = base_of(S, b); TH = np.full((S["E"], 10), np.nan); EF = S["EF"]
    if ve_test is not None: VEh = VEh & ve_test; VEm = VEm & ve_test
    VE = VEh | VEm
    for (j, kk), tx in S["TRX"].items():
        bf, ve, vem = TRB[(j, kk)]; vv = ve | vem
        if ve_train is not None: vv = vv & ve_train[(j, kk)]
        TH[EF[:, j] == kk, j] = pick_t(bf, vv, vtr[(j, kk)], tx["pid"], tx["lab"], rule)
    veto = VE & BF & (V < TH); F = BF & ~veto
    k2 = np.minimum(k, S["nsz"])[:, None]; NS = BNS + np.where(VEh & BF, k2, 0) + np.where(VEm & BF, k2 - 1, 0)
    if TRAINSTATS is not None:   # training-patient outcome of this rule per fold (resubstitution: the threshold was set on the same patients)
        for (j, kk), tx in S["TRX"].items():
            bf, ve, vem = TRB[(j, kk)]; vv = ve | vem
            if ve_train is not None: vv = vv & ve_train[(j, kk)]
            t = TH[EF[:, j] == kk, j][0]; keep = bf & ~(vv & (vtr[(j, kk)] < t)); pid = tx["pid"]; lab = tx["lab"]
            TRAINSTATS[(j, kk)] = (len(np.unique(pid[keep & (lab == 0)])), len(np.unique(pid[keep & (lab == 1)])), len(np.unique(pid[bf & (lab == 1)])), int(vv.sum() * k))
    return F, NS, TH
TRAINSTATS = None
def run_veto(S, b, k, ftest, ftrain, rule, direction=1.0):
    """ftest: sample scores (n,) or (n, 10); ftrain: callable (j, k) -> (n,) sample score for training rows (inner predictions, or the fixed vector)"""
    V = direction * vs_test(S, ftest, k); vtr = {(j, kk): direction * vs_train(S, ftrain(j, kk), tx["top"], k) for (j, kk), tx in S["TRX"].items()}
    return veto_V(S, b, V, vtr, rule, k)
def matched_spec_p1(S, A, F):
    """Δ endoscopy specificity of F against P1 (max L-IMG) at F's endoscopy sensitivity, per repeat (pooled folds), observed and over the draws"""
    D = S["D"]; pidx = S["pidx"]; P = S["elab"] == 1; img = S["OUTS"]["img"]; out_o, out_b = [], []
    for Wm, store in ((np.ones((1, S["P_"])), out_o), (D, out_b)):
        w = Wm[:, pidx]; res = []
        for j in range(10):
            wp = w * P; wn = w * ~P; sens_m = (wp * F[:, j]).sum(1) / wp.sum(1); spec_m = 1 - (wn * F[:, j]).sum(1) / wn.sum(1)
            od = np.argsort(-img[:, j], kind="mergesort"); s = img[od, j]; cp = np.cumsum(wp[:, od], 1); cn = np.cumsum(wn[:, od], 1)
            last = np.searchsorted(-s, -s, side="right") - 1   # last position with an equal score
            idx = np.argmax(cp >= (sens_m * wp.sum(1))[:, None] - 1e-9, 1); li = last[idx]; spec_p1 = 1 - cn[np.arange(len(li)), li] / wn.sum(1)
            res.append(spec_m - spec_p1)
        store.append(np.mean(res, 0))
    return dstat(float(out_o[0][0]), out_b[0])
def contrasts(S, A, F=None):
    D = S["D"]; one = np.ones((1, S["P_"])); nP = int((S["plab"] == 1).sum()); nN = int((S["plab"] == 0).sum()); o = {}
    for ref in ("R49", "P1"):
        Ar = S[f"A_{ref}"]; a0, r0 = lstats(S, A, one), lstats(S, Ar, one); ab, rb = lstats(S, A, D), lstats(S, Ar, D)
        d = lambda key, mult=1.0: dstat(mult * float(a0[key][0] - r0[key][0]), mult * (ab[key] - rb[key]))
        o[ref] = {"d_np_patients_flagged": d("np_flagged_share", nN), "d_progressor_patients_caught": d("caught_share", nP), "d_endo_spec": d("endo_spec"), "d_endo_sens": d("endo_sens"),
                  "d_seq_per_patient": d("seq_per_patient"), "d_flags_per_np_py": d("flags_per_np_py"), "d_cost_per_patient": d("cost_per_patient")}
    if F is not None: o["P1_matched_sens_d_endo_spec"] = matched_spec_p1(S, A, F)
    return o
def summarize(S, F, NS, label, params, extra=None):
    A = patient_arrays(S, F, NS); c = cell_result(S, A, S["D"], "E2", label, params, 0, extra); c["contrasts"] = contrasts(S, A, F); return c
def short(c):
    k = ("progressor_patients_caught", "non_progressor_patients_ever_flagged", "samples_sequenced_per_patient", "endoscopy_sensitivity", "endoscopy_specificity")
    return {**{x: c[x] for x in k}, "dNP_vs_R49": c["contrasts"]["R49"]["d_np_patients_flagged"], "dP_vs_R49": c["contrasts"]["R49"]["d_progressor_patients_caught"],
            "dNP_vs_P1": c["contrasts"]["P1"]["d_np_patients_flagged"], "dP_vs_P1": c["contrasts"]["P1"]["d_progressor_patients_caught"]}
def apparent(S, b, k, ftest, direction=1.0):
    """OPTIMISTIC: one veto threshold per repeat chosen on the evaluation endoscopies = the highest keeping every progressor patient the base catches"""
    BF, BNS, VEh, VEm, _ = base_of(S, b); VE = VEh | VEm; V = direction * vs_test(S, ftest, k); F = np.zeros_like(BF)
    for j in range(10):
        t = pick_t(BF[:, j], VE[:, j], V[:, j], S["pidx"], S["elab"], "strict"); F[:, j] = BF[:, j] & ~(VE[:, j] & (V[:, j] < t))
    k2 = np.minimum(k, S["nsz"])[:, None]; NS = BNS + np.where(VEh & BF, k2, 0) + np.where(VEm & BF, k2 - 1, 0)
    A = patient_arrays(S, F, NS); one = np.ones((1, S["P_"])); a = lstats(S, A, one); nP = int((S["plab"] == 1).sum()); nN = int((S["plab"] == 0).sum())
    return {"caught": r2(float(a["caught_share"][0]) * nP), "np_flagged": r2(float(a["np_flagged_share"][0]) * nN), "seq_per_patient": r3(float(a["seq_per_patient"][0])), "label": "OPTIMISTIC (threshold on evaluation patients)"}
def wauc(x, lab, w):
    """weighted AUROC (higher x = progressor), multiplicity weights"""
    xp, xn, wp, wn = x[lab == 1], x[lab == 0], w[lab == 1], w[lab == 0]
    if wp.sum() == 0 or wn.sum() == 0: return np.nan
    Wt = wp[:, None] * wn[None, :]; return float((((xp[:, None] > xn[None, :]) + 0.5 * (xp[:, None] == xn[None, :])) * Wt).sum() / Wt.sum())
# ------------------------------------------------------------------ sample score dictionary for the fixed vetoes
FIXED = ["ps_fga05", "ps_fga10", "ps_fga15", "ps_narm05", "ps_narm10", "ps_armabs", "ps_oac", "ps_focal10", "ps_maxdev", "ps_sd", "ps_cx",
         "jt_fga05", "jt_fga10", "jt_fga15", "jt_narm05", "jt_narm10", "jt_armabs", "jt_oac", "jt_focal10", "jt_cx",
         "th_cx", "th_narmz2", "th_armzabs", "th_ntilez2", "th_oacz"]
def double_V(S, q, Om, Im, kq, qq):
    """double veto scores: inf (never vetoed) when the k_q-sample quietness score q reaches its copy-number-alone training threshold (share qq of training
    progressor endoscopies), else the top sample's trained score (Om (n,10) held-out; Im(j, k) -> (n,) inner)"""
    Vq = vs_test(S, q, kq); Vm = vs_test(S, Om, 1); V = np.full((S["E"], 10), np.nan); vtr = {}; EF = S["EF"]
    for (j, kk), tx in S["TRX"].items():
        vq = vs_train(S, q, tx["top"], kq); t2 = thr_sens(vq[(tx["lab"] == 1) & np.isfinite(vq)], qq); vm = vs_train(S, Im(j, kk), tx["top"], 1)
        vtr[(j, kk)] = np.where(vq >= t2, np.inf, vm); te = EF[:, j] == kk; V[te, j] = np.where(Vq[te, j] >= t2, np.inf, Vm[te, j])
    return V, vtr
res = {"task": TASK}
# ================================================================== feat: per-sample copy-number features (row-level, cluster only)
if TASK == "feat":
    RS = M + "/review_round4/reseg"; out = None; info = {}
    for tag, f in (("jt_", "tiles_joint_stored_analysed.csv"), ("ps_", "tiles_persample_analysed.csv")):
        F = pd.read_csv(f"{RS}/{f}", dtype={"Sample": str}); assert set(F.Sample) >= set(ids), tag; F = F.set_index("Sample").reindex(ids).reset_index()
        o, ARM, _ = tile_feats(F, tag); info[f"{tag}qc_pass"] = int(o[f"{tag}qc"].sum()); o = o.merge(ARM, on="Sample"); out = o if out is None else out.merge(o, on="Sample")
    TH = pd.read_csv(M + "/cnv_their_C.csv", dtype={"sample_id": str}).set_index("sample_id").reindex(ids); cols = list(TH.columns); W = {c: (int(c.split(":")[1].split("-")[1]) - int(c.split(":")[1].split("-")[0])) for c in cols if c != "cx"}
    tcs = [c for c, w in W.items() if w < 6e6]; acs = [c for c, w in W.items() if w >= 6e6]; an = arm_names(acs); AZ = TH[acs].values
    th = pd.DataFrame({"Sample": ids, "th_cx": TH.cx.values, "th_narmz2": (np.abs(AZ) > 2).sum(1), "th_armzabs": np.abs(AZ).sum(1), "th_ntilez2": (np.abs(TH[tcs].values) > 2).sum(1)})
    ai = {a: i for i, a in enumerate(an)}; th["th_oacz"] = sum(np.maximum(0, -AZ[:, ai[a]]) for a in LOSS) + sum(np.maximum(0, AZ[:, ai[a]]) for a in GAIN)
    out = out.merge(th, on="Sample")
    C = pd.read_csv(M + "/set_C.csv", dtype=str).set_index("Sample").reindex(ids); mp = [mapd_of(os.path.join(os.path.dirname(p), "50.copy_number.txt")) for p in C.raw.values]
    out["mapd"] = [a for a, b in mp]; out["sd50"] = [b for a, b in mp]; info["mapd_missing"] = int(np.isnan(out.mapd).sum())
    fd = sheet(); fd = fd[fd.combined_name.notna()].drop_duplicates("combined_name").set_index("combined_name")
    out["batch"] = pd.Series(ids).map(fd.Batch).values; out["slx"] = pd.Series(ids).str.extract(r"SLX-(\d+)")[0].astype(float).values
    out["reads"] = pd.to_numeric(pd.Series(ids).map(fd["Number of reads"]), errors="coerce").values; out["cellularity"] = pd.to_numeric(pd.Series(ids).map(fd["% Barrett's cellularity"]).astype(str).str.replace("%", ""), errors="coerce").values
    info.update({"batch_missing": int(pd.isna(out.batch).sum()), "reads_missing": int(np.isnan(out.reads).sum()), "cellularity_missing": int(np.isnan(out.cellularity).sum()), "n": int(len(out)), "n_features": int(out.shape[1] - 1)})
    out.to_csv(XD + "/features.csv", index=False); res["feat"] = info; dump(res)
# ================================================================== desc: features among H&E-flagged endoscopies; H&E vs copy-number false-positive independence
elif TASK == "desc":
    S = build(); FT = load_feat(); D = S["D"]; pidx = S["pidx"]; elab = S["elab"]; es = S["es"]; O = S["O"]; R = {}
    R["check"] = {"R49": short(summarize(S, S["GC"], S["NS49"], "R49", {})) , "P1_90": short(summarize(S, S["P1F"], np.zeros((S["E"], 10), int), "P1 90%", {}))}
    sc = {f: FT[f].values.astype(float) for f in FIXED + [c for c in FT.columns if c.startswith("ps_arm_") or c.startswith("jt_arm_")] + ["mapd", "reads", "cellularity"]}
    for m in ("L-CNV", "L-IMG", "L-LATE"): sc[m] = lg(O[m])
    flag = S["P1F"]; feats = {}
    for f, x in sc.items():
        V = vs_test(S, x, 1); V = np.where(np.isfinite(V), V, np.nan); obs = []; bs = np.zeros((len(D), 10)); med = []
        for j in range(10):
            mk = flag[:, j] & np.isfinite(V[:, j]); obs.append(wauc(V[mk, j], elab[mk], np.ones(mk.sum())))
            for bi in range(len(D)): bs[bi, j] = wauc(V[mk, j], elab[mk], D[bi][pidx[mk]])
            med.append((np.nanmedian(V[mk & (elab == 1), j]), np.nanmedian(V[mk & (elab == 0), j])))
        # training-patient view: AUROC among each fold's H&E-flagged training endoscopies (inner flags, inner top sample)
        trv = []
        for (j, kk), tx in S["TRX"].items():
            if f in ("L-CNV", "L-IMG", "L-LATE"): xi = np.full(n, np.nan); s_ = S["I"][f][(REPS[j], kk)]; xi[s_.index.values] = lg(s_.values)
            else: xi = x
            v = xi[tx["top"][:, 0]]; mk = tx["p1"] & np.isfinite(v); trv.append(wauc(v[mk], tx["lab"][mk], np.ones(mk.sum())))
        feats[f] = {"auroc_flagged_heldout": r3(np.nanmean(obs)), "ci95": ci2(np.nanmean(bs, 1)), "auroc_flagged_training_folds_mean": r3(np.nanmean(trv)),
                    "median_progressor": r3(np.mean([a for a, b in med])), "median_non_progressor": r3(np.mean([b for a, b in med]))}
    R["features_top_sample_P1_flagged"] = feats; R["n_flagged_endoscopies_mean"] = {"progressor": r2(float((flag & (elab == 1)[:, None]).sum(0).mean())), "non_progressor": r2(float((flag & (elab == 0)[:, None]).sum(0).mean()))}
    # same, R49 high band (the flags R49 makes without sequencing)
    hb = {}
    for f in ("ps_fga10", "ps_narm05", "ps_oac", "jt_fga10", "th_cx", "L-CNV"):
        V = vs_test(S, sc[f], 1); obs = [wauc(V[S["GB"][:, j] == 2, j], elab[S["GB"][:, j] == 2], np.ones((S["GB"][:, j] == 2).sum())) for j in range(10)]; hb[f] = r3(np.nanmean(obs))
    R["auroc_R49_high_band_top_sample"] = hb; R["n_R49_high_band_mean"] = {"progressor": r2(float(((S["GB"] == 2) & (elab == 1)[:, None]).sum(0).mean())), "non_progressor": r2(float(((S["GB"] == 2) & (elab == 0)[:, None]).sum(0).mean()))}
    # independence of H&E and copy-number false positives among non-progressor endoscopies (P1 90% vs P2 90%) and patients
    T2, _ = g8["thr_arr"](S, "cnv", 0.90); P2F = S["OUTS"]["cnv"] >= T2; N = elab == 0; tab = {}
    for nm, mk in (("non_progressor_endoscopies", N), ("progressor_endoscopies", ~N)):
        a = (S["P1F"] & P2F)[mk].sum(0).mean(); b_ = (S["P1F"] & ~P2F)[mk].sum(0).mean(); c_ = (~S["P1F"] & P2F)[mk].sum(0).mean(); d_ = (~S["P1F"] & ~P2F)[mk].sum(0).mean(); tot = a + b_ + c_ + d_
        tab[nm] = {"both": r2(a), "H&E only": r2(b_), "CN only": r2(c_), "neither": r2(d_), "expected_both_if_independent": r2((a + b_) * (a + c_) / tot),
                   "phi": r3(float((a * d_ - b_ * c_) / np.sqrt((a + b_) * (c_ + d_) * (a + c_) * (b_ + d_))))}
    gN = np.mean([np.corrcoef(lg(S["OUTS"]["img"][N, j]), lg(S["OUTS"]["cnv"][N, j]))[0, 1] for j in range(10)]); R["fp_independence"] = tab; R["corr_max_logit_img_cnv_np_endoscopies"] = r3(gN)
    A1 = S["A_P1"]; A2 = patient_arrays(S, P2F, np.zeros((S["E"], 10), int)); isN = S["plab"] == 0
    R["fp_independence_patients"] = {"both": r2((A1["FL"] * A2["FL"])[isN].sum(0).mean()), "H&E only": r2((A1["FL"] * (1 - A2["FL"]))[isN].sum(0).mean()), "CN only": r2(((1 - A1["FL"]) * A2["FL"])[isN].sum(0).mean()), "neither": r2(((1 - A1["FL"]) * (1 - A2["FL"]))[isN].sum(0).mean())}
    R["note"] = ("held-out P1 (90% training endoscopy sensitivity) flagged endoscopies; feature of the top-H&E sample (outer L-IMG); AUROC progressor vs non-progressor endoscopy pooled over folds within a repeat, mean over repeats; "
                 "CI from patient-bootstrap multiplicity weights (RandomState(0)); training view = mean over the 100 outer folds of the AUROC among the fold's training endoscopies H&E flags by inner out-of-fold L-IMG; predictions as logits; "
                 "ps_ = per-sample segmentation (R36), jt_ = joint per-patient segmentation as stored, th_ = shipped block (cnv_their_C.csv); deviations relative to the sample's median 5-Mb tile")
    res["desc"] = R; dump(res)
# ================================================================== fixed-score vetoes on one base
elif TASK.startswith("fixed_"):
    b = TASK.split("_")[1]; S = build(); FT = load_feat(); cells = []
    sc = {f: FT[f].values.astype(float) for f in FIXED}
    for f, x in sc.items():
        for k in (1, 2):
            for rule in RULES:
                F, NS, TH = run_veto(S, b, k, x, lambda j, kk: x, rule); c = summarize(S, F, NS, f"{b} veto {f} k{k} {rule}", {"base": b, "feature": f, "k": k, "rule": rule}); c["apparent"] = apparent(S, b, k, x) if rule == "strict" else None; cells.append(c)
    # stored L-CNV / L-LATE of the sequenced sample(s) (outer on held-out, inner on training): the trained-model veto ("AND" rule)
    O, I = S["O"], S["I"]
    for m in ("L-CNV", "L-LATE"):
        def ftr(j, kk, m=m): x = np.full(n, np.nan); s_ = I[m][(REPS[j], kk)]; x[s_.index.values] = s_.values; return x
        for k in (1, 2):
            for rule in RULES:
                F, NS, TH = run_veto(S, b, k, O[m], ftr, rule); c = summarize(S, F, NS, f"{b} veto {m} k{k} {rule}", {"base": b, "feature": m, "k": k, "rule": rule}); c["apparent"] = apparent(S, b, k, O[m]) if rule == "strict" else None; cells.append(c)
    res["cells"] = cells; res["base"] = b; dump(res)
# ================================================================== en_<fs>_<pop>_<rep>: elastic-net veto trained on H&E-flagged training endoscopies (row-level predictions)
elif TASK.startswith("en_") and TASK != "en_eval":
    import warnings; from sklearn.linear_model import LogisticRegression; from sklearn.exceptions import ConvergenceWarning; warnings.simplefilter("ignore", ConvergenceWarning)
    _, fs, pop, rep = TASK.split("_"); rep = int(rep); j = rep - 1; S = setup(); es = S["es"]; I = S["I"]; FO = S["FO"]; TR = S["TR"]
    if fs == "their": X = pd.read_csv(M + "/cnv_their_C.csv", dtype={"sample_id": str}).set_index("sample_id").reindex(ids).values.astype(float)
    else:
        FT = load_feat(); cols = [c for c in FT.columns if c.startswith("ps_A_")] + ["ps_fga05", "ps_fga10", "ps_fga15", "ps_narm05", "ps_narm10", "ps_armabs", "ps_sd", "ps_maxdev", "ps_focal10", "ps_cx", "th_cx"]
        X = FT[cols].values.astype(float); X = np.where(np.isfinite(X), X, np.nanmedian(X, 0))
    CG = np.logspace(-2.5, 1, 8)
    def fit_pred(tr, te, grp):
        """standardise on tr; C by patient-grouped CV on tr (mean binomial deviance; groups = grp labels); fit; predict te"""
        mu = X[tr].mean(0); sd = X[tr].std(0); sd[sd == 0] = 1; Z = (X - mu) / sd; yt = y[tr]; ug = np.unique(grp); dev = np.zeros(len(CG))
        if len(np.unique(yt)) < 2: return np.full(len(te), yt.mean()), np.nan, 0
        for g in ug:
            a = tr[grp != g]; b_ = tr[grp == g]
            if len(np.unique(y[a])) < 2: continue
            for ci, C in enumerate(CG):
                m = LogisticRegression(penalty="elasticnet", l1_ratio=0.9, solver="saga", C=C, max_iter=3000, tol=1e-3).fit(Z[a], y[a]); pr = np.clip(m.predict_proba(Z[b_])[:, 1], 1e-5, 1 - 1e-5)
                dev[ci] += -2 * (y[b_] * np.log(pr) + (1 - y[b_]) * np.log(1 - pr)).sum()
        C = CG[int(np.argmin(dev))]; m = LogisticRegression(penalty="elasticnet", l1_ratio=0.9, solver="saga", C=C, max_iter=5000, tol=1e-4).fit(Z[tr], yt)
        return m.predict_proba(Z[te])[:, 1], C, int((m.coef_ != 0).sum())
    q = {"flag": 0.90, "wide": 0.95, "all": None}[pop]; rows = []; info = []
    for k in range(1, 11):
        tr = TR[(j, k)]; tidx = tr["idx"]; t = thr_sens(tr["img"][tr["lab"] == 1], q) if q else -np.inf; popE = tidx[tr["img"] >= t]
        trs = np.concatenate([es[e] for e in tidx]); pops = np.concatenate([es[e] for e in popE])
        inn = pd.read_csv(f"{M}/horizons/inner/img_rep_{rep:02d}_fold_{k:02d}.csv", dtype={"Sample": str}); ifo = pd.Series(inn.inner_fold.values, index=g8["ix"][inn.Sample].values).reindex(trs).values
        assert np.isfinite(ifo).all(); ifo_of = pd.Series(ifo, index=trs)
        for i in range(1, 6):   # inner out-of-fold veto scores for every training sample of inner fold i
            a = pops[ifo_of.reindex(pops).values != i]; te = trs[ifo == i]; pr, C, nz = fit_pred(a, te, ifo_of.reindex(a).values)
            rows.append(pd.DataFrame({"i": te, "rep": rep, "fold": k, "kind": "inner", "pred": pr})); info.append({"fold": k, "inner": i, "C": C, "nnz": nz, "n_train": len(a)})
        te = np.where(FO[:, j] == k)[0]; te = te[S["popm"][te]]; pr, C, nz = fit_pred(pops, te, ifo_of.reindex(pops).values)
        rows.append(pd.DataFrame({"i": te, "rep": rep, "fold": k, "kind": "outer", "pred": pr})); info.append({"fold": k, "inner": 0, "C": C, "nnz": nz, "n_train": len(pops)})
    os.makedirs(XD + "/en", exist_ok=True); pd.concat(rows).to_csv(f"{XD}/en/{fs}_{pop}_rep_{rep:02d}.csv", index=False); pd.DataFrame(info).to_csv(f"{XD}/en/{fs}_{pop}_rep_{rep:02d}_info.csv", index=False)
    print("EXE2 DONE", TASK, flush=True)
# ================================================================== en_eval: the elastic-net vetoes on the three bases
elif TASK == "en_eval":
    S = build(); cells = []; meta = {}
    for fs in ("their", "compact"):
        for pop in ("flag", "wide", "all"):
            fl = sorted(glob.glob(f"{XD}/en/{fs}_{pop}_rep_*[0-9].csv"))
            if len(fl) < 10: meta[f"{fs}_{pop}"] = f"missing ({len(fl)} of 10 repeats)"; continue
            Dd = pd.concat([pd.read_csv(f) for f in fl]); Oo = np.full((n, 10), np.nan); INN = {}
            for (r, k, kind), d in Dd.groupby(["rep", "fold", "kind"]):
                if kind == "outer": Oo[d.i.values, r - 1] = d.pred.values
                else: x = np.full(n, np.nan); x[d.i.values] = d.pred.values; INN[(r - 1, k)] = x
            inf = pd.concat([pd.read_csv(f.replace(".csv", "_info.csv")) for f in fl]); meta[f"{fs}_{pop}"] = {"median_nonzero_coefficients_outer": float(inf[inf.inner == 0].nnz.median()), "median_C_outer": float(inf[inf.inner == 0].C.median())}
            Vauc = vs_test(S, Oo, 1); fl_ = S["P1F"]; meta[f"{fs}_{pop}"]["auroc_top_sample_P1_flagged_heldout"] = r3(np.nanmean([wauc(Vauc[fl_[:, j], j], S["elab"][fl_[:, j]], np.ones(fl_[:, j].sum())) for j in range(10)]))
            for b in ("P1", "R49h", "R49hm"):
                for k in (1, 2):
                    for rule in RULES:
                        F, NS, TH = run_veto(S, b, k, Oo, lambda j, kk: INN[(j, kk)], rule); c = summarize(S, F, NS, f"{b} veto EN[{fs},{pop}] k{k} {rule}", {"base": b, "feature": f"EN_{fs}_{pop}", "k": k, "rule": rule})
                        c["apparent"] = apparent(S, b, k, Oo) if rule == "strict" else None; cells.append(c)
    res["cells"] = cells; res["meta"] = meta; dump(res)
# ================================================================== drift: copy-number change over time within non-progressors, batch / noise; batch- and noise-aware vetoes
elif TASK == "drift":
    import statsmodels.formula.api as smf; from scipy.stats import spearmanr
    S = build(); FT = load_feat(); O = S["O"]; es = S["es"]; pidx = S["pidx"]; plab = S["plab"]; elab = S["elab"]; R = {}
    SC = np.array([lg(O["L-CNV"][s]).max(0).mean() for s in es]); SCI = np.array([lg(O["L-IMG"][s]).max(0).mean() for s in es])
    bt = FT.batch.astype(str).values; lmapd = np.log(FT.mapd.values); lreads = np.log(FT.reads.values); fga = FT.ps_fga10.values; fgaj = FT.jt_fga10.values
    eb = np.array([pd.Series(bt[s]).mode().iloc[0] for s in es]); emapd = np.array([np.nanmean(lmapd[s]) for s in es]); ereads = np.array([np.nanmean(lreads[s]) for s in es])
    efga = np.array([np.nanmax(fga[s]) for s in es]); efgaj = np.array([np.nanmax(fgaj[s]) for s in es]); slx = FT.slx.values; eslx = np.array([np.nanmedian(slx[s]) for s in es])
    # first vs last pre-event endoscopy, patients with >= 2
    out = {}
    for cl, v in (("non_progressor", 0), ("progressor", 1)):
        P = [i for i in range(S["P_"]) if plab[i] == v and len(S["ORD"][i]) >= 2]; fl = [(S["ORD"][i][0], S["ORD"][i][-1]) for i in P]
        same = np.array([eb[a] == eb[b_] for a, b_ in fl]); d = {nm: np.array([arr[b_] - arr[a] for a, b_ in fl]) for nm, arr in (("L-CNV", SC), ("L-IMG", SCI), ("log_mapd", emapd), ("log_reads", ereads), ("ps_fga10_max", efga), ("jt_fga10_max", efgaj), ("slx_run_number", eslx))}
        o = {"n_patients": len(P), "first_last_same_batch": m5(int(same.sum())), "first_last_different_batch": m5(int((~same).sum()))}
        for nm, dd in d.items():
            o[nm] = {"mean_last_minus_first": r3(np.nanmean(dd)), "same_batch_mean": r3(np.nanmean(dd[same])) if same.sum() >= 5 else None, "different_batch_mean": r3(np.nanmean(dd[~same])) if (~same).sum() >= 5 else None}
        o["spearman_dLCNV_vs_dlogmapd"] = r3(spearmanr(d["L-CNV"], d["log_mapd"], nan_policy="omit")[0]); o["spearman_dLCNV_vs_dslx"] = r3(spearmanr(d["L-CNV"], d["slx_run_number"], nan_policy="omit")[0])
        o["spearman_dLCNV_vs_dfga"] = r3(spearmanr(d["L-CNV"], d["ps_fga10_max"], nan_policy="omit")[0]); out[cl] = o
    R["first_vs_last"] = out
    # mixed models among non-progressor endoscopies (patients >= 2 endoscopies): time slope with and without batch / noise adjustment
    keep = np.isin(pidx, [i for i in range(S["P_"]) if len(S["ORD"][i]) >= 2]); mm = {}
    df = pd.DataFrame({"sc": SC, "tte": S["TTE"], "batch": eb, "lmapd": emapd, "lreads": ereads, "fga": efga, "pid": pidx, "cls": elab})[keep]
    for cl, v in (("non_progressor", 0), ("progressor", 1)):
        d_ = df[df.cls == v].dropna(); o = {}
        for nm, fml in (("time only", "sc ~ tte"), ("+ batch", "sc ~ tte + C(batch)"), ("+ log MAPD + log reads", "sc ~ tte + lmapd + lreads"), ("+ batch + log MAPD + log reads", "sc ~ tte + C(batch) + lmapd + lreads"),
                        ("fga ~ time", "fga ~ tte"), ("fga ~ time + batch", "fga ~ tte + C(batch)"), ("log MAPD ~ time", "lmapd ~ tte")):
            try:
                f_ = smf.mixedlm(fml, d_, groups=d_["pid"]).fit(reml=True); ci = f_.conf_int(); o[nm] = {"tte": r3(f_.params["tte"]), "ci95": [r3(ci.loc["tte", 0]), r3(ci.loc["tte", 1])], "p": round(float(f_.pvalues["tte"]), 4), "n": int(len(d_))}
                if "lmapd" in fml and nm.startswith("+"): o[nm]["lmapd"] = r3(f_.params["lmapd"]); o[nm]["lmapd_p"] = round(float(f_.pvalues["lmapd"]), 4)
            except Exception as ex: o[nm] = f"failed: {str(ex)[:120]}"
        mm[cl] = o
    R["mixed_models_negative_tte_slope_means_rising_towards_reference"] = mm
    # batch aggregates among pre-event samples
    rows = S["rows"]; mL = lg(O["L-CNV"]).mean(1); ba = {}
    for bname in sorted(set(bt[rows])):
        s = rows[bt[rows] == bname]; o = {"n_samples_np": m5(int((y[s] == 0).sum())), "n_samples_p": m5(int((y[s] == 1).sum()))}
        for cl, v in (("np", 0), ("p", 1)):
            ss = s[y[s] == v]
            if len(ss) >= 5: o[f"mean_LCNV_logit_{cl}"] = r3(mL[ss].mean()); o[f"median_mapd_{cl}"] = r3(np.nanmedian(FT.mapd.values[ss])); o[f"median_ps_fga10_{cl}"] = r3(np.nanmedian(fga[ss])); o[f"median_years_to_reference_{cl}"] = r2(np.median(S["TTE"][S["SEi"][ss]]))
        ba[bname] = o
    R["by_batch"] = ba
    npre = rows[y[rows] == 0]; ppre = rows[y[rows] == 1]
    R["sample_spearman"] = {cl: {"LCNV_vs_log_mapd": r3(spearmanr(mL[s], lmapd[s], nan_policy="omit")[0]), "LCNV_vs_log_reads": r3(spearmanr(mL[s], lreads[s], nan_policy="omit")[0]), "ps_fga10_vs_log_mapd": r3(spearmanr(fga[s], lmapd[s], nan_policy="omit")[0]),
                                 "jt_fga10_vs_log_mapd": r3(spearmanr(fgaj[s], lmapd[s], nan_policy="omit")[0]), "th_cx_vs_log_mapd": r3(spearmanr(FT.th_cx.values[s], lmapd[s], nan_policy="omit")[0])} for cl, s in (("non_progressor", npre), ("progressor", ppre))}
    # batch-centred and noise-adjusted vetoes (adjustments fitted on each outer fold's training non-progressor samples)
    cells = []; trs_of = {kk: np.concatenate([es[e] for e in tx["idx"]]) for kk, tx in S["TRX"].items()}
    for f in ("ps_fga10", "ps_narm05", "jt_fga10", "th_cx", "L-CNV"):
        base = (lg(O["L-CNV"]) if f == "L-CNV" else FT[f].values.astype(float))
        for adj in ("batch", "noise"):
            ftest = np.full((n, 10), np.nan); ftr = {}
            for (j, kk), tx in S["TRX"].items():
                tri = trs_of[(j, kk)]; te = np.where(S["FO"][:, j] == kk)[0]
                if f == "L-CNV": xs = np.full(n, np.nan); s_ = S["I"]["L-CNV"][(REPS[j], kk)]; xs[s_.index.values] = lg(s_.values); xs[te] = base[te, j]
                else: xs = base.copy()
                nps = tri[y[tri] == 0]
                if adj == "batch":
                    gm = np.nanmedian(xs[nps]); bm = pd.Series(xs[nps]).groupby(bt[nps]).agg(lambda v: np.nanmedian(v) if np.isfinite(v).sum() >= 5 else np.nan); off = pd.Series(bt).map(bm).fillna(gm).values; xa = xs - off
                else:
                    ok = np.isfinite(xs[nps]) & np.isfinite(lmapd[nps]); cf = np.polyfit(lmapd[nps][ok], xs[nps][ok], 1); xa = xs - np.polyval(cf, np.where(np.isfinite(lmapd), lmapd, np.nanmedian(lmapd)))
                ftr[(j, kk)] = xa; ftest[te, j] = xa[te]
            for b in ("P1", "R49hm"):
                for rule in ("strict", "loss1", "endo95"):
                    F, NS, TH = run_veto(S, b, 1, ftest, lambda j, kk: ftr[(j, kk)], rule); cells.append(summarize(S, F, NS, f"{b} veto {f} {adj}-adjusted k1 {rule}", {"base": b, "feature": f, "adjust": adj, "k": 1, "rule": rule}))
    R["note"] = ("endoscopy L-CNV score = mean over repeats of the maximum logit (as round 8 L3); time = years before the reference date (event / last sequenced endoscopy); endoscopy batch = the commonest sheet Batch of its samples; "
                 "MAPD = median absolute difference of consecutive 50-kb log2 copy-number bins (QDNAseq copy_number.txt) within chromosomes; batch adjustment = minus the training non-progressor median of the sample's batch (>= 5 samples, else the global training non-progressor median); "
                 "noise adjustment = residual on log MAPD, line fitted on training non-progressor samples; both refitted per outer fold; the held-out sample is adjusted with its fold's training constants")
    res["drift"] = R; res["cells"] = cells; dump(res)
# ================================================================== stab: within-patient copy-number stability veto (no history at the first endoscopy)
elif TASK == "stab":
    S = build(); FT = load_feat(); es = S["es"]; I = S["I"]; O = S["O"]; ARM = FT[[c for c in FT.columns if c.startswith("ps_A_")]].values.astype(float); ARM = np.where(np.isfinite(ARM), ARM, 0.0)
    FE = np.array([S["ORD"][S["pidx"][e]][0] for e in range(S["E"])]); FIRST = S["FIRST"]; cur = FT.ps_armabs.values.astype(float)
    def pairscore(a, b_, kind):
        da, db = ARM[a], ARM[b_]
        if kind == "change": return np.abs(da - db).mean(-1)
        if kind == "new": return ((np.abs(da) > 0.05) & (np.abs(db) <= 0.05)).sum(-1).astype(float)
        if kind == "cur_plus_change": return cur[a] + np.abs(da - db).sum(-1)
    cells = []
    for kind in ("change", "new", "cur_plus_change"):
        V = np.full((S["E"], 10), np.nan)
        for j in range(10): V[:, j] = pairscore(S["TOPO"][:, j, 0], S["TOPO"][FE, j, 0], kind)
        vtr = {}; vet = {}
        for (j, kk), tx in S["TRX"].items():
            pos = {e: q for q, e in enumerate(tx["idx"])}; top = tx["top"][:, 0]; base_s = np.array([top[pos[FE[e]]] for e in tx["idx"]]); vtr[(j, kk)] = pairscore(top, base_s, kind); vet[(j, kk)] = ~FIRST[tx["idx"]]
        for b in ("P1", "R49hm"):
            for rule in ("strict", "loss1", "endo95"):
                F, NS, TH = veto_V(S, b, V, vtr, rule, 1, ve_test=~FIRST[:, None], ve_train=vet)
                BF, _, VEh, VEm, _ = base_of(S, b); used = (VEh | VEm) & BF & ~FIRST[:, None]; seqd = (VEh | VEm) & BF & FIRST[:, None]
                extra = np.zeros((S["E"], 10), int)   # the first endoscopy's top sample is sequenced once per patient if a later veto needs it and it was not sequenced
                for i in range(S["P_"]):
                    o = S["ORD"][i]; need = used[o].any(0) & ~seqd[o[0]] & ~((S["GB"][o[0]] == 1) if b != "P1" else np.zeros(10, bool)); extra[o[0]] += need
                cells.append(summarize(S, F, NS + extra, f"{b} stability veto {kind} {rule}", {"base": b, "feature": f"stability_{kind}", "k": 1, "rule": rule, "first_endoscopy": "no veto (no history)"}))
    res["cells"] = cells
    res["note"] = ("stability between the endoscopy's top-H&E sample and the top-H&E sample of the patient's first pre-event endoscopy (per-sample segmentation arm deviations): change = mean |Δ arm|; new = arms altered (> 5%) now but not at baseline; "
                   "cur_plus_change = current arm burden + summed |Δ arm|; the first endoscopy keeps its flag; extra sequencing = the baseline sample when it was not already sequenced")
    dump(res)
# ================================================================== sel_<base>: the veto chosen per outer fold on training patients from a candidate family (handles the multiplicity of cells)
elif TASK.startswith("sel_"):
    b = TASK.split("_")[1]; S = build(); FT = load_feat(); O, I = S["O"], S["I"]; CAND = []
    def ftr_m(m):
        def f(j, kk): x = np.full(n, np.nan); s_ = I[m][(REPS[j], kk)]; x[s_.index.values] = s_.values; return x
        return f
    feats = [(f, FT[f].values.astype(float), (lambda x: (lambda j, kk: x))(FT[f].values.astype(float)), "quiet") for f in FIXED] + [(m, O[m], ftr_m(m), "model") for m in ("L-CNV", "L-LATE")]
    for fs in ("their", "compact"):
        for pop in ("flag", "wide"):
            fl = sorted(glob.glob(f"{XD}/en/{fs}_{pop}_rep_*[0-9].csv"))
            if len(fl) < 10: continue
            Dd = pd.concat([pd.read_csv(f) for f in fl]); Oo = np.full((n, 10), np.nan); INN = {}
            for (r, k, kind), d in Dd.groupby(["rep", "fold", "kind"]):
                if kind == "outer": Oo[d.i.values, r - 1] = d.pred.values
                else: x = np.full(n, np.nan); x[d.i.values] = d.pred.values; INN[(r - 1, k)] = x
            feats.append((f"EN_{fs}_{pop}", Oo, (lambda D_: (lambda j, kk: D_[(j, kk)]))(INN), "en"))
    for f, ft, fr, grp in feats:
        for k in (1, 2):
            for rule in RULES:
                TRAINSTATS = {}; F, NS, TH = run_veto(S, b, k, ft, fr, rule); CAND.append({"name": f"{f} k{k} {rule}", "grp": grp, "F": F, "NS": NS, "tr": TRAINSTATS})
    GR = SM.Pathology.map({"NDBE": 0, "ID": 1, "LGD": 2, "HGD": 3, "IMC": 4}).values.astype(float); EG = np.array([GR[s_].max() for s_ in S["es"]])
    vt = (EG <= 1)[:, None]; vtr_m = {kk: EG[tx["idx"]] <= 1 for kk, tx in S["TRX"].items()}
    for m in ("L-CNV", "L-LATE"):
        for k in (1, 2):
            V = vs_test(S, O[m], k); vtr = {kk: vs_train(S, ftr_m(m)(*kk), tx["top"], k) for kk, tx in S["TRX"].items()}
            for rule in RULES:
                TRAINSTATS = {}; F, NS, TH = veto_V(S, b, V, vtr, rule, k, ve_test=vt, ve_train=vtr_m); CAND.append({"name": f"{m} k{k} {rule} (NDBE/ID only)", "grp": "model_gated", "F": F, "NS": NS, "tr": TRAINSTATS})
    for m in ("L-LATE", "L-CNV"):
        for qf in ("ps_oac", "jt_oac", "th_ntilez2", "jt_narm05"):
            for qq in (0.90, 0.80):
                V, vtr = double_V(S, FT[qf].values.astype(float), O[m], ftr_m(m), 2, qq)
                for rule in ("strict", "loss1", "endo95"):
                    TRAINSTATS = {}; F, NS, TH = veto_V(S, b, V, vtr, rule, 2); CAND.append({"name": f"{m} low AND {qf} quiet<{int(qq * 100)} kq2 {rule}", "grp": "double", "F": F, "NS": NS, "tr": TRAINSTATS})
    cells = []; E = S["E"]; EF = S["EF"]
    for fam, grps in (("double vetoes (trained score low AND quiet)", ("double",)), ("quiet features", ("quiet",)), ("trained models (L-CNV, L-LATE)", ("model",)), ("trained models, veto only at NDBE/ID endoscopies", ("model_gated",)),
                      ("trained models, gated or not", ("model", "model_gated")), ("elastic-net vetoes", ("en",)), ("all", ("quiet", "model", "model_gated", "en", "double"))):
        cs = [c for c in CAND if c["grp"] in grps]
        if not cs: continue
        for slack in (0, 1):
            F = np.zeros((E, 10), bool); NS = np.zeros((E, 10), int); pick = {}
            for (j, kk) in S["TRX"]:
                ok = [c for c in cs if c["tr"][(j, kk)][1] >= c["tr"][(j, kk)][2] - slack]
                best = min(ok, key=lambda c: (c["tr"][(j, kk)][0], c["tr"][(j, kk)][3])) if ok else None; te = EF[:, j] == kk
                if best is None: continue
                F[te, j] = best["F"][te, j]; NS[te, j] = best["NS"][te, j]; pick[best["name"]] = pick.get(best["name"], 0) + 1
            c = summarize(S, F, NS, f"{b} veto selected per fold from {fam}, training loss <= {slack}", {"base": b, "family": fam, "slack": slack}); c["picks_top"] = dict(sorted(pick.items(), key=lambda x: -x[1])[:8]); cells.append(c)
    res["cells"] = cells; res["n_candidates"] = len(CAND)
    res["note"] = ("per outer fold, among the candidate vetoes of the family (feature x k x threshold rule), the one with the fewest training non-progressor patients flagged (ties: less sequencing) among those whose "
                   "training progressor-patient detection is >= the base's training detection minus the slack; training outcomes are resubstitution (thresholds set on the same training patients)")
    dump(res)
# ================================================================== decomp: where R49's false flags come from, and what the leading vetoes remove
elif TASK == "decomp":
    S = build(); FT = load_feat(); O, I = S["O"], S["I"]; INC = S["INC"]; isN = S["plab"] == 0; isP = ~isN; GB = S["GB"]; GC = S["GC"]; R = {}
    hi = (INC @ (GC & (GB == 2))) > 0; mi = (INC @ (GC & (GB == 1))) > 0
    R["R49_np_patients_flag_source"] = {"high band only": r2((hi & ~mi)[isN].sum(0).mean()), "middle band only": r2((~hi & mi)[isN].sum(0).mean()), "both": r2((hi & mi)[isN].sum(0).mean())}
    R["R49_progressor_patients_flag_source"] = {"high band only": r2((hi & ~mi)[isP].sum(0).mean()), "middle band only": r2((~hi & mi)[isP].sum(0).mean()), "both": r2((hi & mi)[isP].sum(0).mean())}
    eN = S["elab"] == 0; R["R49_flagged_np_endoscopies"] = {"high band": r2((GC & (GB == 2))[eN].sum(0).mean()), "middle band called": r2((GC & (GB == 1))[eN].sum(0).mean()), "middle band sequenced": r2((GB == 1)[eN].sum(0).mean())}
    GR = SM.Pathology.map({"NDBE": 0, "ID": 1, "LGD": 2, "HGD": 3, "IMC": 4}).values.astype(float); pg = np.array([GR[np.concatenate([S["es"][e] for e in np.where(S["pidx"] == i)[0]])].max() for i in range(S["P_"])])
    def ftr_m(m):
        def f(j, kk): x = np.full(n, np.nan); s_ = I[m][(REPS[j], kk)]; x[s_.index.values] = s_.values; return x
        return f
    for nm, b, k, ft, fr, rule in (("R49hm veto L-LATE k1 strict", "R49hm", 1, O["L-LATE"], ftr_m("L-LATE"), "strict"), ("R49hm veto L-LATE k1 loss1", "R49hm", 1, O["L-LATE"], ftr_m("L-LATE"), "loss1"),
                                   ("R49hm veto ps_oac k2 cn90", "R49hm", 2, FT.ps_oac.values, lambda j, kk: FT.ps_oac.values, "cn90"), ("R49hm veto L-CNV k1 strict", "R49hm", 1, O["L-CNV"], ftr_m("L-CNV"), "strict")):
        F, NS, TH = run_veto(S, b, k, ft, fr, rule); FL = (INC @ F) > 0; F49 = (INC @ GC) > 0; cleared = F49 & ~FL
        o = {"np_patients_cleared_mean": r2(cleared[isN].sum(0).mean()), "progressor_patients_lost_mean": r2(cleared[isP].sum(0).mean())}
        for g, gn in ((0, "NDBE"), (1, "ID"), (2, "LGD")): o[f"np_cleared_highest_grade_{gn}"] = m5(round(float(cleared[isN & (pg == g)].sum(0).mean()), 1)); o[f"np_flagged_by_R49_highest_grade_{gn}"] = m5(round(float(F49[isN & (pg == g)].sum(0).mean()), 1))
        vet = GC & ~F; tte = S["TTE"]; vp = vet & (S["elab"] == 1)[:, None]
        o["vetoed_progressor_endoscopies_mean"] = r2(vp.sum(0).mean()); o["vetoed_progressor_endoscopy_years_to_event_median"] = r2(float(np.median(np.repeat(tte[:, None], 10, 1)[vp]))) if vp.any() else None
        o["kept_progressor_endoscopy_years_to_event_median"] = r2(float(np.median(np.repeat(tte[:, None], 10, 1)[F & (S["elab"] == 1)[:, None]])))
        o["vetoed_np_endoscopies_mean"] = r2((vet & eN[:, None]).sum(0).mean()); R[nm] = o
    res["decomp"] = R; dump(res)
# ================================================================== combo: double veto = vetoed only if the trained score is low AND the profile is quiet
elif TASK == "combo":
    S = build(); FT = load_feat(); O, I = S["O"], S["I"]; cells = []; EF = S["EF"]
    def inner_of(m, j, kk): x = np.full(n, np.nan); s_ = I[m][(REPS[j], kk)]; x[s_.index.values] = s_.values; return x
    for m in ("L-LATE", "L-CNV"):
        for qf in ("ps_oac", "th_ntilez2", "jt_narm05", "ps_fga10"):
            q = FT[qf].values.astype(float)
            for kq in (1, 2):
                for qq in (0.90, 0.80):
                    Vq = vs_test(S, q, kq); Vm = vs_test(S, O[m], 1); V = np.full((S["E"], 10), np.nan); vtr = {}
                    for (j, kk), tx in S["TRX"].items():
                        vq = vs_train(S, q, tx["top"], kq); t2 = thr_sens(vq[(tx["lab"] == 1) & np.isfinite(vq)], qq); vm = vs_train(S, inner_of(m, j, kk), tx["top"], 1)
                        vtr[(j, kk)] = np.where(vq >= t2, np.inf, vm); te = EF[:, j] == kk; V[te, j] = np.where(Vq[te, j] >= t2, np.inf, Vm[te, j])
                    for b in ("P1", "R49hm"):
                        for rule in ("strict", "loss1", "endo95"):
                            F, NS, TH = veto_V(S, b, V, vtr, rule, max(kq, 1)); cells.append(summarize(S, F, NS, f"{b} double veto {m} low AND {qf} quiet (< {int(qq * 100)}% CN-alone) kq{kq} {rule}", {"base": b, "feature": f"{m}+{qf}", "k": kq, "q_quiet": qq, "rule": rule}))
    res["cells"] = cells; res["note"] = "an H&E flag is vetoed only if the quietness score of the k_q top-H&E samples is below its copy-number-alone training threshold (90% or 80% of training progressor endoscopies) AND the top sample's trained score is below the rule's threshold"; dump(res)
# ================================================================== and: H&E AND copy number, both above their own training thresholds (sequence every sample of an H&E-flagged endoscopy)
elif TASK == "and":
    S = build(); X = S["OUTS"]; cells = []; thr_arr = g8["thr_arr"]
    for qi in (0.80, 0.90, 0.95):
        Ti, _ = thr_arr(S, "img", qi); Fi = X["img"] >= Ti
        for qc in (0.80, 0.90, 0.95):
            Tc, _ = thr_arr(S, "cnv", qc); F = Fi & (X["cnv"] >= Tc); NS = np.where(Fi, S["nsz"][:, None], 0)
            cells.append(summarize(S, F, NS, f"AND: max L-IMG >= {int(qi * 100)}% and max L-CNV (all samples) >= {int(qc * 100)}%", {"base": "AND", "q_img": qi, "q_cnv": qc}))
    res["cells"] = cells; res["note"] = "endoscopy flagged if its max L-IMG reaches the P1 threshold at q_img AND its max L-CNV over all samples reaches the P2 threshold at q_cnv (both training endoscopy-sensitivity thresholds); every sample of an H&E-flagged endoscopy is sequenced"; dump(res)
# ================================================================== grade: veto allowed only at endoscopies of low grade (dysplastic endoscopies keep their flag)
elif TASK == "grade":
    S = build(); FT = load_feat(); O, I = S["O"], S["I"]; cells = []
    GR = SM.Pathology.map({"NDBE": 0, "ID": 1, "LGD": 2, "HGD": 3, "IMC": 4}).values.astype(float); EG = np.array([GR[s].max() for s in S["es"]])
    def inner_of(m, j, kk): x = np.full(n, np.nan); s_ = I[m][(REPS[j], kk)]; x[s_.index.values] = s_.values; return x
    for gname, gmax in (("NDBE only", 0), ("NDBE or ID", 1)):
        vt = (EG <= gmax)[:, None]; vtr_m = {kk: EG[tx["idx"]] <= gmax for kk, tx in S["TRX"].items()}
        for f, k in (("L-LATE", 1), ("L-CNV", 1), ("ps_oac", 2), ("jt_oac", 2)):
            if f.startswith("L-"): V = vs_test(S, O[f], k); vtr = {kk: vs_train(S, inner_of(f, *kk), tx["top"], k) for kk, tx in S["TRX"].items()}
            else: x = FT[f].values.astype(float); V = vs_test(S, x, k); vtr = {kk: vs_train(S, x, tx["top"], k) for kk, tx in S["TRX"].items()}
            for b in ("P1", "R49hm"):
                for rule in RULES:
                    F, NS, TH = veto_V(S, b, V, vtr, rule, k, ve_test=vt, ve_train=vtr_m)
                    cells.append(summarize(S, F, NS, f"{b} veto {f} k{k} {rule}, vetoable only at {gname} endoscopies", {"base": b, "feature": f, "k": k, "rule": rule, "grade_gate": gname}))
    res["cells"] = cells; res["note"] = "as the fixed / trained vetoes, but only endoscopies whose highest sample grade is NDBE (or NDBE/ID) can be vetoed and sequenced for the veto; dysplastic endoscopies keep the base call"; dump(res)
# ================================================================== mem: the best vetoes on top of the round-9 patient-memory rule (exponentially weighted R49-aligned gate score)
elif TASK == "mem":
    S = build(); FT = load_feat(); O, I = S["O"], S["I"]; cells = []; R9 = M + "/review_round9"
    def inner_of(m, j, kk): x = np.full(n, np.nan); s_ = I[m][(REPS[j], kk)]; x[s_.index.values] = s_.values; return x
    GR = SM.Pathology.map({"NDBE": 0, "ID": 1, "LGD": 2, "HGD": 3, "IMC": 4}).values.astype(float); EG = np.array([GR[s_].max() for s_ in S["es"]])
    gate_t = (EG <= 1)[:, None]; gate_tr = {kk: EG[tx["idx"]] <= 1 for kk, tx in S["TRX"].items()}
    VETOES = []
    for m, k in (("L-LATE", 1), ("L-CNV", 1)):
        V = vs_test(S, O[m], k); vtr = {kk: vs_train(S, inner_of(m, *kk), tx["top"], k) for kk, tx in S["TRX"].items()}
        for rule in ("strict", "loss1", "endo95"):
            VETOES.append((f"{m} k{k} {rule}", V, vtr, rule, k, None)); VETOES.append((f"{m} k{k} {rule}, NDBE/ID only", V, vtr, rule, k, "gate"))
    q = FT.ps_oac.values.astype(float); V = vs_test(S, q, 2); vtr = {kk: vs_train(S, q, tx["top"], 2) for kk, tx in S["TRX"].items()}; VETOES.append(("ps_oac k2 cn90", V, vtr, "cn90", 2, None))
    for qq in (0.80, 0.90):
        V, vtr = double_V(S, q, O["L-LATE"], lambda j, kk: inner_of("L-LATE", j, kk), 2, qq)
        for rule in ("strict", "endo95"): VETOES.append((f"L-LATE low AND ps_oac quiet<{int(qq * 100)} kq2 {rule}", V, vtr, rule, 2, None))
    out = {}
    for grp, key in (("X", "A1_gate_al_ew_ext"), ("A1ga", "A1_gate_al_ew"), ("A12", "A12")):
        f = f"{R9}/grp_{grp}.npz"
        if not os.path.exists(f): out[key] = "missing"; continue
        Z = np.load(f)
        if f"{key}__sel_np_F" not in Z.files: out[key] = "key missing"; continue
        FM = Z[f"{key}__sel_np_F"].astype(bool); NM = Z[f"{key}__sel_np_NS"].astype(int); assert FM.shape == (S["E"], 10)
        c0 = summarize(S, FM, NM, f"round-9 memory {key} (selected per fold, as stored)", {"base": f"mem:{key}"}); cells.append(c0); out[key] = short(c0)
        for nm, V, vtr, rule, k, g in VETOES:
            _, _, TH = veto_V(S, "R49hm", V, vtr, rule, k, ve_test=gate_t if g else None, ve_train=gate_tr if g else None)
            ve = FM & (gate_t if g else True); veto = ve & (V < TH); F = FM & ~veto
            k2 = np.minimum(k, S["nsz"])[:, None]; NS = NM + np.where(ve, np.where(S["GB"] == 1, k2 - 1, k2), 0)
            c = summarize(S, F, NS, f"memory {key} + veto {nm}", {"base": f"mem:{key}", "feature": nm, "k": k, "rule": rule})
            A = patient_arrays(S, F, NS); Am = patient_arrays(S, FM, NM); one = np.ones((1, S["P_"])); D = S["D"]; a0, r0, ab, rb = lstats(S, A, one), lstats(S, Am, one), lstats(S, A, D), lstats(S, Am, D)
            nP = int((S["plab"] == 1).sum()); nN = int((S["plab"] == 0).sum()); dd = lambda kk_, mult=1.0: dstat(mult * float(a0[kk_][0] - r0[kk_][0]), mult * (ab[kk_] - rb[kk_]))
            c["contrasts"]["memory_rule"] = {"d_np_patients_flagged": dd("np_flagged_share", nN), "d_progressor_patients_caught": dd("caught_share", nP), "d_seq_per_patient": dd("seq_per_patient"), "d_endo_spec": dd("endo_spec"), "d_cost_per_patient": dd("cost_per_patient")}
            cells.append(c)
    res["cells"] = cells; res["memory_alone"] = out
    res["note"] = ("round-9 memory rule flags/sequencing as stored (killcoyne_mm/review_round9/grp_*.npz, per-fold selected 'np' criterion); the veto is applied to every memory flag "
                   "(or only NDBE/ID endoscopies), with the veto threshold set on training patients against the R49hm base (the memory rule's own training flags are not stored), so this is a cheap approximation; "
                   "sequencing = stored memory-rule sequencing + k top samples at every vetoable memory flag (k - 1 in R49's middle band)")
    dump(res)
else:
    raise SystemExit(f"unknown TASK {TASK}")

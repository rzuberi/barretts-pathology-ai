"""Two papers, agent REG (docs/two_papers_REG.md): freeze one rule per Paper 2 tier plus the R49 gate, chosen on all 75 patients with the existing
selection code, and verify the frozen objects. No rule is re-implemented: the module body of ex_e9_frontier.py is executed unchanged (which executes
rv9_falseflags.py, rv8b_step3.py, rv8_landscape.py and the helper section of ex_e3_score.py; check_arrays() runs ex_e2_veto.py up to its dispatch).
New pieces: (1) full_ctx = the round-9 fold context with ALL 75 patients as training patients and their outer out-of-fold predictions of one repeat in
the role that inner out-of-fold predictions play inside an outer fold; (2) sequential freezing over the 10 repeats (continuous values -> median,
discrete choices -> mode, ties to the first candidate; every later step re-run with the earlier frozen values); (3) a comparison against the
standalone scorer tp_reg_score.py. Tiers: no sequencing = mean-logit + EW memory (E9 'ml_mem'); light = EW memory of the R49-aligned gate score +
copy-number certificate (round-9 A12, E9 'memcert'); confirmatory = light + quiet-genome check, ps_oac of the two top-H&E samples at the cn90 bar
(E9 'final'); plus the R49 gate (their matrix) and its package-feature flavour (t_seq on package-feature L-LATE; for ACE-B and the 4x retest).
TASK = nested (smoke: the selection code on the 100 stored folds must reproduce E9's operating points) | freeze | verify (after freeze; models + scorer).
Row-level outputs stay under killcoyne_mm/two_papers_REG/ on the cluster; aggregates (no identifiers) -> results/paper_final/two_papers_REG/."""
import os, sys, json, glob, shutil, hashlib, subprocess, collections, numpy as np, pandas as pd
THESIS = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"; PP = THESIS + "/scripts/paper_plan"; sys.path.insert(0, PP)
RTASK = os.environ["TASK"]; os.environ["TASK"] = "tpreg_import"
_fself = __file__; __file__ = PP + "/ex_e9_frontier.py"; exec(compile(open(__file__).read(), __file__, "exec")); __file__ = _fself   # definitions only
os.environ["TASK"] = RTASK
import tp_reg_score as SC
REG = M + "/two_papers_REG"; FZ = REG + "/frozen"; OUTR = T + "/results/paper_final/two_papers_REG"
for d_ in (REG, FZ, OUTR): os.makedirs(d_, exist_ok=True)
def jd(o, path): json.dump(o, open(path, "w"), indent=1, sort_keys=True, default=lambda x: x.item() if hasattr(x, "item") else str(x))
# ------------------------------------------------------------------ the round-9 selection rule (rv9 run_families 'np' criterion == E9 sweep at target D)
def pick(c, cands):
    """per candidate: threshold = the D-th highest training progressor-patient maximum score; pick the feasible candidate (training detection >= D) with the
    fewest training non-progressor patients flagged (ties: flagged training NP endoscopies, sequencing, index); else the most detecting one"""
    INC = c.S["INC"]; trN = c.trP & ~c.isP; st = []; fl = []; th = []
    for x in cands:
        pm = pat_scores(c, x["u"]); t = thr_for(c, pm); f = x["u"] >= t; inc = INC @ f.astype(float) > 0
        st.append((c.patient_det(f), int((inc & trN).sum()), int((f & c.tr_e & (c.elab == 0)).sum()), float(x["ns"][c.tr_e].mean()))); fl.append(f); th.append(t)
    st = np.array(st); feas = st[:, 0] >= c.D; nC = len(cands)
    q = sorted([q for q in range(nC) if feas[q]], key=lambda q: (st[q, 1], st[q, 2], st[q, 3], q))[0] if feas.any() else sorted(range(nC), key=lambda q: (-st[q, 0], st[q, 1], st[q, 2], q))[0]
    return q, th[q], fl[q], st[q]
def ev(S, F, NS):
    A = patient_arrays(S, F, NS); p = parts(S, A, np.ones((1, S["P_"]))); nP = int((S["plab"] == 1).sum()); nN = int((S["plab"] == 0).sum())
    return {"caught": r2(nP * met(p, "caught_share")[0]), "np_flagged": r2(nN * met(p, "np_flagged_share")[0]), "seq_per_patient": r3(met(p, "seq_per_patient")[0])}
def with_check(F, NS, band, V, TH, k2):
    veto = F & (V < TH); return F & ~veto, NS + np.where(F, np.where(band == 1, k2 - 1, k2), 0)
# ------------------------------------------------------------------ full context: all 75 patients are training patients; outer OOF predictions of repeat j
def full_ctx(S, ST, j, cuts=None, late=None):
    c = Ctx(); c.S, c.ST, c.j, c.k = S, ST, j, 0; xs = {m: S["O"][m][:, j].copy() for m in MODELS}
    if late is not None: xs["L-LATE"] = late[:, j].copy()
    c.ef = efeat(ST, xs["L-IMG"], xs["L-CNV"], xs["L-LATE"]); E = S["E"]; c.te_e = np.ones(E, bool); c.tr_e = np.ones(E, bool); c.trP = np.ones(S["P_"], bool)
    c.isP = S["plab"] == 1; c.elab = S["elab"]; c.nsz = S["nsz"]; rows = S["rows"]; c.s_img_tr = xs["L-IMG"][rows]; c.yc_tr = y[rows] == 1; c.s_late_tr = xs["L-LATE"][rows]
    c.cuts = tuple(cuts) if cuts is not None else cuts_sample(c, *NOMINAL); ef = c.ef
    c.r49_flag, c.r49_ns, c.r49_band = gate(ef["pi"], ef["pl_top"], c.cuts); c.D = c.patient_det(c.r49_flag); c.cache = {}; return c
def a12_cand(c, h, tq, dl, Y, qcut=None):
    """one A12 candidate (f_A12 construction) with optional frozen quiet cut (logit L-CNV of the top sample)"""
    ef, ST_ = c.ef, c.ST; s, ns = base(c, "gate_al"); seq = c.r49_band == 1; zq = ef["zc_top"][c.tr_e & (c.elab == 0) & seq]; v = np.percentile(zq, tq) if qcut is None else qcut
    u = memory(ST_, s, "ew", h) - dl * cert_flags(ST_, seq & (ef["zc_top"] < v), Y); return {"kind": "score", "u": u, "ns": ns, "qcut": float(np.percentile(zq, tq))}
def mode_first(choices):
    cnt = collections.Counter(choices); best = max(cnt.values()); return [x for x in choices if cnt[x] == best][0], dict(cnt)
med = lambda v: float(np.median(np.asarray(v, float)))
# ================================================================== nested smoke
if RTASK == "nested":
    out = {}
    for var in ("st", "ml"):
        S, ST, INFO = state(var); E = S["E"]; F = {k: np.zeros((E, 10), bool) for k in ("R49", "A12", "mem")}; NS = {k: np.zeros((E, 10), int) for k in F}; BAND = np.zeros((E, 10), int); ch = {"A12": [], "mem": []}
        for j in range(10):
            for k in range(1, 11):
                c = make_ctx(S, ST, j, k, INFO); te = c.te_e; BAND[te, j] = c.r49_band[te]; F["R49"][te, j] = c.r49_flag[te]; NS["R49"][te, j] = c.r49_ns[te]
                fams = {"A12": f_A12(c), "mem": f_score("img" if var == "ml" else "gate_al", "ew")(c)}
                for nm, cands in fams.items(): q, t, f, _ = pick(c, cands); ch[nm].append(q); F[nm][te, j] = f[te]; NS[nm][te, j] = cands[q]["ns"][te]
        o = {nm: ev(S, F[nm], NS[nm]) for nm in F}
        if var == "st":
            CK = check_arrays(); assert (CK["GB"] == BAND).all(); Ff, NSf = with_check(F["A12"], NS["A12"], BAND, CK["V"], CK["TH"], CK["k2"][:, None]); o["final"] = ev(S, Ff, NSf)
        Z = np.load(f"{XD9}/sweep_{var}_{'core'}.npz"); ZA = np.load(f"{XD9}/sweep_{var}_A12.npz")
        o["chosen_equal_to_E9"] = {"A12": bool((np.array(ch["A12"]) == ZA["A12__chosen"][0]).all()), "mem": bool((np.array(ch["mem"]) == Z[("img_ew" if var == "ml" else "gate_al_ew") + "__chosen"][0]).all())}
        out[var] = o
    exp = {"st": {"R49": [31.1, 31.8, 1.94], "A12": [30.5, 21.5, 1.94], "mem": [30.7, 23.6, 1.94], "final": [30.5, 17.2, 3.82]}, "ml": {"mem": [31.0, 26.6, 0.0], "A12": [30.3, 20.1, 1.43]}}
    ok = all(abs(out[v][k]["caught"] - e[0]) < 0.051 and abs(out[v][k]["np_flagged"] - e[1]) < 0.051 and abs(out[v][k]["seq_per_patient"] - e[2]) < 0.006 for v in exp for k, e in exp[v].items())
    out["expected_E9"] = exp; out["reproduced"] = bool(ok and all(all(out[v]["chosen_equal_to_E9"].values()) for v in ("st", "ml")))
    print(json.dumps(out, indent=1)); jd(out, f"{OUTR}/nested_smoke.json"); print("TPREG DONE nested", out["reproduced"], flush=True)
# ================================================================== freeze
if RTASK == "freeze":
    S, ST, INFO = state("st"); Sm, STm, INFOm = state("ml"); E = S["E"]; Op, _, _ = TC.load("pkg"); assert np.allclose(Op["L-IMG"], S["O"]["L-IMG"], equal_nan=True)
    CK = check_arrays(); assert CK["S2"]["E"] == E and (CK["S2"]["plab"] == S["plab"]).all(); V = CK["V"]; k2 = CK["k2"]
    tr = {"st_cuts": [], "pkg_cuts": [], "ml_cuts": []}
    # ---- stage 1: gate cut-offs (sample level, 95% sens / 90% spec / 80% middle-band sens), per repeat on all patients' out-of-fold predictions
    for j in range(10):
        tr["st_cuts"].append(full_ctx(S, ST, j).cuts); tr["pkg_cuts"].append(full_ctx(S, ST, j, late=Op["L-LATE"]).cuts); tr["ml_cuts"].append(full_ctx(Sm, STm, j).cuts)
    fc = {k: tuple(med([c_[i] for c_ in v]) for i in range(3)) for k, v in tr.items()}; assert fc["st_cuts"][:2] == fc["pkg_cuts"][:2]
    # ---- stage 2: discrete hyper-parameters with the frozen cut-offs (A12 tuple; mean-logit memory half-life)
    a12c, mlc, Dst, Dml = [], [], [], []
    for j in range(10):
        c = full_ctx(S, ST, j, fc["st_cuts"]); Dst.append(c.D); cands = f_A12(c); q, _, _, _ = pick(c, cands); p_ = cands[q]["params"]; a12c.append((p_["h"], p_["t_quiet_pct"], p_["delta"], p_["Y"]))
        cm = full_ctx(Sm, STm, j, fc["ml_cuts"]); Dml.append(cm.D); cands = f_score("img", "ew")(cm); q, _, _, _ = pick(cm, cands); mlc.append(cands[q]["params"]["h"])
    a12, a12cnt = mode_first(a12c); hml, mlcnt = mode_first(mlc)
    # ---- stage 3: continuous values (quiet cut, then thresholds) with everything earlier frozen; check bar (cn90) per repeat
    qc = [a12_cand(full_ctx(S, ST, j, fc["st_cuts"]), *a12)["qcut"] for j in range(10)]; qcf = med(qc)
    ta, tm, bars = [], [], []
    for j in range(10):
        c = full_ctx(S, ST, j, fc["st_cuts"]); x = a12_cand(c, *a12, qcut=qcf); ta.append(thr_for(c, pat_scores(c, x["u"])))
        cm = full_ctx(Sm, STm, j, fc["ml_cuts"]); s_, _ = base(cm, "img"); u = memory(STm, s_, "ew", hml); tm.append(thr_for(cm, pat_scores(cm, u)))
        v = V[:, j][(S["elab"] == 1) & np.isfinite(V[:, j])]; bars.append(thr_sens(v, 0.90))
    taf, tmf, barf = med(ta), med(tm), med(bars)
    ck = lambda t: {"t_low": t[0], "c_star": t[1], "t_seq": t[2]}
    rules = {"rules": {
        "R49": {"cuts": ck(fc["st_cuts"]), "definition": "endoscopy score = max sample L-IMG; low if < t_low, high (flag) if > c_star, middle: the top-L-IMG sample is sequenced and the endoscopy is flagged if its L-LATE (their matrix) >= t_seq"},
        "R49_pkg": {"cuts": ck(fc["pkg_cuts"]), "definition": "as R49 with L-LATE from package-feature L-CNV (models_pkg); t_low and c_star identical; used where copy number comes from reads (ACE-B, 4x)"},
        "tier0_no_sequencing": {"half_life_years": hml, "threshold": tmf, "mean_logit_gate_cuts_for_D": ck(fc["ml_cuts"]),
                                "definition": "endoscopy score z = mean of the sample L-IMG logits (probabilities clipped to [1e-6, 1-1e-6]; 1e-6 x top-sample logit tie-break); patient memory at endoscopy k = sum_{i<=k} w_i z_i / sum w_i, w_i = 0.5 ** ((t_k - t_i) / h); flag if memory >= threshold. No sequencing."},
        "tier_light": {"half_life_years": a12[0], "quiet_percentile": a12[1], "quiet_cut_logit_cnv": qcf, "delta_logit": a12[2], "years": a12[3], "threshold": taf,
                       "definition": "gate-aligned score g = logit(max L-IMG), or in the R49 middle band logit(c_star) + logit(L-LATE of the sequenced top sample) - logit(t_seq); EW memory of g with half-life h; minus delta while certified (an earlier middle-band endoscopy <= Y years before whose sequenced top sample had logit L-CNV < quiet cut); flag if >= threshold. Sequences the R49 middle band's top sample."},
        "tier_confirmatory": {"base": "tier_light", "check_bar_ps_oac": barf, "check_samples": 2,
                              "definition": "tier_light flags; at every flag the two top-L-IMG samples are sequenced (one already sequenced in the middle band) and the flag is cancelled if max ps_oac of the two < check bar (cn90: 90% of progressor endoscopies exceed it). ps_oac = sum over 17p 9p 18q 5q 4q 13q 3p of max(0, -arm deviation) + over 8q 7p 20q of max(0, arm deviation); arm deviation = arm copy number / the sample's median 5-Mb tile - 1 (per-sample package segmentation, ex_e2_veto.py tile_feats)"}},
        "selection": {"population": "571 pre-event samples, 288 endoscopies, 75 patients (32 progressors) of the 676 samples from 80 patients with a matched H&E slide",
                      "training_predictions": "outer out-of-fold predictions of the stored 10 x 10 patient-grouped CV (killcoyne_mm/cv/preds, cfg 00 / 01 / 02), one repeat at a time, in the role of inner out-of-fold predictions",
                      "rule": "rv9 operating rule: threshold = the D-th highest progressor-patient maximum score, D = progressor patients the R49 gate detects; candidate with the fewest non-progressor patients flagged among those detecting >= D",
                      "freezing": "per repeat; continuous values = median over the 10 repeats, discrete = mode (ties: first in candidate order); sequential: gate cut-offs -> hyper-parameters -> quiet cut -> thresholds; check bar = 90%-sensitivity bar of max ps_oac over the two top-H&E samples of progressor endoscopies",
                      "caveat": "chosen on fold-model (out-of-fold) predictions; applying them to the frozen models fitted on all of set C assumes similar score distributions on new data (tb_frozen.py caveat)"}}
    trace = {"cuts_per_repeat": {k: [list(map(float, c_)) for c_ in v] for k, v in tr.items()}, "D_per_repeat": {"st": Dst, "ml": Dml}, "a12_choice_per_repeat": [list(x) for x in a12c], "a12_choice_counts": {str(k): v for k, v in a12cnt.items()},
             "ml_h_per_repeat": mlc, "ml_h_counts": {str(k): v for k, v in mlcnt.items()}, "quiet_cut_per_repeat": qc, "a12_threshold_per_repeat": ta, "ml_threshold_per_repeat": tm, "check_bar_per_repeat": bars,
             "tb_frozen_reference": json.load(open(T + "/results/paper_final/triage_robustness/frozen_cutoffs_their.json"))["cutoffs"]}
    # ---- apparent (OPTIMISTIC) numbers: frozen rules on the same out-of-fold predictions, all patients, per repeat; engine construction
    F = {k: np.zeros((E, 10), bool) for k in ("R49", "R49_pkg", "tier0", "light", "confirm")}; NS = {k: np.zeros((E, 10), int) for k in F}; U = {k: np.zeros((E, 10)) for k in ("tier0", "light")}
    for j in range(10):
        c = full_ctx(S, ST, j, fc["st_cuts"]); F["R49"][:, j] = c.r49_flag; NS["R49"][:, j] = c.r49_ns
        cp = full_ctx(S, ST, j, fc["pkg_cuts"], late=Op["L-LATE"]); F["R49_pkg"][:, j] = cp.r49_flag; NS["R49_pkg"][:, j] = cp.r49_ns
        x = a12_cand(c, *a12, qcut=qcf); U["light"][:, j] = x["u"]; F["light"][:, j] = x["u"] >= taf; NS["light"][:, j] = x["ns"]
        F["confirm"][:, j], NS["confirm"][:, j] = with_check(F["light"][:, j], NS["light"][:, j], c.r49_band, V[:, j], barf, k2)
        cm = full_ctx(Sm, STm, j, fc["ml_cuts"]); s_, _ = base(cm, "img"); u = memory(STm, s_, "ew", hml); U["tier0"][:, j] = u; F["tier0"][:, j] = u >= tmf
    app = {k: ev(S, F[k], NS[k]) for k in F}
    nested_ref = {"R49": [31.1, 31.8, 1.94], "R49_pkg": None, "tier0": [31.0, 26.6, 0.0], "light": [30.5, 21.5, 1.94], "confirm": [30.5, 17.2, 3.82]}
    # ---- scanner reset reference: class-wise quantiles of the out-of-fold L-IMG probability, pre-event samples, pooled over repeats
    rows = S["rows"]; qs = np.linspace(0, 1, 1001); X = S["O"]["L-IMG"][rows]; yy = y[rows]
    reset = {"probs": qs.tolist(), "quantiles_progressor": np.quantile(X[yy == 1].ravel(), qs).tolist(), "quantiles_non_progressor": np.quantile(X[yy == 0].ravel(), qs).tolist(),
             "pi_registered": 15 / 117, "pi_discovery_samples": float(yy.mean()),
             "definition": "label-free quantile mapping of new L-IMG probabilities onto the discovery out-of-fold L-IMG distribution mixed at progressor share pi (registered constant 15/117, ACE-B's expected progressor share from the cohort description, fixed before data access); applied to every sample L-IMG before any rule; L-LATE uses the reset L-IMG; L-CNV untouched; monotone, AUROC unchanged",
             "discovery_band_shares_at_frozen_cuts": {"sample_low": float((X < fc["st_cuts"][0]).mean()), "sample_high": float((X > fc["st_cuts"][1]).mean())}}
    agg = {"paper1_patient_score": "per model, sample probability -> logit (clip 1e-6) -> mean over the endoscopy's samples -> max over the patient's included endoscopies (equal to the endoscopy mean logit when a patient has one endoscopy)",
           "sensitivity_aggregation": "max over all the patient's sample probabilities", "late": "L-LATE sample probability = (L-CNV + L-IMG) / 2, then the same aggregation"}
    # ---- write the rule files and the row-level verification inputs (cluster only)
    rules["frozen_on"] = pd.Timestamp.now().isoformat(timespec="seconds"); rules["patient_aggregation"] = agg
    rules["code"] = {"scripts": ["scripts/paper_plan/tp_reg_freeze.py", "scripts/paper_plan/tp_reg_score.py", "scripts/paper_plan/kc_freeze.R", "scripts/paper_plan/kv_freeze.R", "scripts/paper_plan/kv_common.R", "scripts/paper_plan/ex_e9_frontier.py", "scripts/paper_plan/rv9_falseflags.py", "scripts/paper_plan/ex_e2_veto.py", "scripts/paper_plan/ex_e3_score.py"],
                     "commit": subprocess.run(["git", "-C", T, "log", "-1", "--format=%H", "--", "scripts/paper_plan/tp_reg_freeze.py"], capture_output=True, text=True).stdout.strip()}
    jd(rules, FZ + "/rules.json"); jd(reset, FZ + "/scanner_reset.json"); jd(trace, REG + "/freeze_trace.json")
    np.savez_compressed(REG + "/apparent_flags.npz", **{f"F_{k}": v for k, v in F.items()}, **{f"NS_{k}": v for k, v in NS.items()}, **{f"U_{k}": v for k, v in U.items()})
    res = {"frozen": rules["rules"], "selection": rules["selection"], "trace": {k: v for k, v in trace.items() if k != "cuts_per_repeat"}, "cuts_per_repeat": trace["cuts_per_repeat"],
           "apparent_OPTIMISTIC_frozen_rules_on_oof": app, "nested_reference_E9": nested_ref, "patient_aggregation": agg, "code_commit": rules["code"]["commit"]}
    jd(res, OUTR + "/freeze.json"); print(json.dumps({"cuts": fc, "a12": a12, "hml": hml, "qcut": qcf, "t_a12": taf, "t_ml": tmf, "bar": barf, "apparent": app}, indent=1)); print("TPREG DONE freeze", flush=True)
# ================================================================== verify
if RTASK == "verify":
    rules = json.load(open(FZ + "/rules.json")); reset = json.load(open(FZ + "/scanner_reset.json")); S, ST, INFO = state("st"); E = S["E"]; Z = np.load(REG + "/apparent_flags.npz")
    Op, _, _ = TC.load("pkg"); FT = pd.read_csv(M + "/explore_E2/features.csv", dtype={"Sample": str}).set_index("Sample").reindex(ids); q = FT.ps_oac.values.astype(float)
    # (a) standalone scorer == engine construction, flag for flag, every repeat (their matrix: R49, tiers; package: R49_pkg)
    epos = np.full(n, -1); eord = np.zeros(n, int); ety = np.zeros(n)
    for i, o in enumerate(S["ORD"]):
        for w, e in enumerate(o): epos[S["es"][e]] = e; eord[S["es"][e]] = w; ety[S["es"][e]] = ST["TY"][i, w]
    rows = S["rows"]; mm = collections.Counter(); keymap = {"R49": "R49_flag", "R49_pkg": "R49_pkg_flag", "tier0": "tier0_flag", "light": "light_flag", "confirm": "confirm_flag"}; seqmap = {"light": "light_seq", "confirm": "confirm_seq", "R49": "R49_seq"}
    for j in range(10):
        df = pd.DataFrame({"patient": pat[rows], "endo": epos[rows], "order": eord[rows], "t_years": ety[rows], "p_img": S["O"]["L-IMG"][rows, j], "p_cnv": S["O"]["L-CNV"][rows, j], "ps_oac": q[rows]})
        Es, _ = SC.score(df, rules); idx = Es.endo.values.astype(int)
        dfp = df.assign(p_cnv=Op["L-CNV"][rows, j]); Ep, _ = SC.score(dfp, {"rules": {"R49": rules["rules"]["R49_pkg"], "R49_pkg": rules["rules"]["R49_pkg"], **{k: rules["rules"][k] for k in ("tier0_no_sequencing", "tier_light", "tier_confirmatory")}}})
        for k, col in keymap.items():
            src = Ep if k == "R49_pkg" else Es; mm[f"flag_{k}"] += int((src[col].values.astype(bool) != Z[f"F_{k}"][idx, j]).sum())
        for k, col in seqmap.items(): mm[f"seq_{k}"] += int((Es[col].values != Z[f"NS_{k}"][idx, j]).sum())
        mm["score_tier0_maxabs_x1e9"] = max(mm["score_tier0_maxabs_x1e9"], int(1e9 * np.nanmax(np.abs(Es.tier0_score.values - Z["U_tier0"][idx, j]))))
        mm["score_light_maxabs_x1e9"] = max(mm["score_light_maxabs_x1e9"], int(1e9 * np.nanmax(np.abs(Es.light_score.values - Z["U_light"][idx, j]))))
    # (b) frozen models: refits byte-identical to the committed bundles; in-sample (apparent) AUROC recomputed from the frozen coefficients
    sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest(); ident = {}
    for sub, ref in (("models_their", "killcoyne_frozen_v1"), ("models_pkg", "killcoyne_frozen_pkg_v1")):
        man = json.load(open(f"{T}/models/{ref}/MANIFEST.json"))["files"]
        for f_, h_ in man.items():
            if f_ in ("model_info.json", "extraction_settings.json") or not os.path.exists(f"{FZ}/{sub}/{f_}"): continue
            ident[f"{sub}/{f_}"] = sha(f"{FZ}/{sub}/{f_}") == h_
    C = pd.read_csv(M + "/set_C.csv", dtype=str); yC = (C.Status == "P").astype(int).values; IM = pd.read_csv(M + "/img_mean.csv", dtype={"Sample": str}).set_index("Sample").reindex(C.Sample)
    sc_ = pd.read_csv(FZ + "/models_their/image_scaling.csv"); Xi = (IM[sc_.feature].values - sc_["mean"].values) / np.where(sc_["sd"].values > 0, sc_["sd"].values, 1)
    def prob(coef, X): cf = pd.read_csv(coef); b0 = cf.coefficient.values[0]; return sg(b0 + X @ cf.coefficient.values[1:])
    CN = pd.read_csv(M + "/cnv_their_C.csv", dtype={"sample_id": str}).set_index("sample_id").reindex(C.Sample)
    pi_ = prob(FZ + "/models_their/coef_L_IMG.csv", Xi); pc_ = prob(FZ + "/models_their/coef_L_CNV.csv", CN[pd.read_csv(FZ + "/models_their/coef_L_CNV.csv").feature.values[1:]].values)
    from sklearn.metrics import roc_auc_score
    mi = json.load(open(FZ + "/models_their/model_info.json"))["insample_check"]; pre = C.Sample.isin(ids[rows]).values
    insample = {"L_IMG": [r3(roc_auc_score(yC, pi_)), r3(mi["auroc_L_IMG"])], "L_CNV": [r3(roc_auc_score(yC, pc_)), r3(mi["auroc_L_CNV"])], "L_LATE": [r3(roc_auc_score(yC, (pi_ + pc_) / 2)), r3(mi["auroc_L_LATE"])],
                "L_IMG_pre_event_571": r3(roc_auc_score(yC[pre], pi_[pre]))}
    # (c) development estimates of the registered Paper 1 patient score (nested, out-of-fold, fold-stratified within repeat, repeat mean) and of sample-level AUROC
    def pauc_fs(col_src):
        vals = []
        for j in range(10):
            d = pd.DataFrame({"patient": pat[rows], "endo": epos[rows], "z": lg(col_src[rows, j]), "fold": S["FO"][rows, j], "y": y[rows]})
            e = d.groupby(["patient", "endo"]).agg(z=("z", "mean"), fold=("fold", "first"), y=("y", "max")).groupby(level=0).agg(z=("z", "max"), fold=("fold", "first"), y=("y", "max"))
            vals.append(wauc(e.z.values, e.y.values, e.fold.values))
        return r3(np.mean(vals))
    dev = {"L_IMG": pauc_fs(S["O"]["L-IMG"]), "L_CNV_their": pauc_fs(S["O"]["L-CNV"]), "L_CNV_pkg": pauc_fs(Op["L-CNV"]), "L_LATE_their": pauc_fs(S["O"]["L-LATE"]), "L_LATE_pkg": pauc_fs(Op["L-LATE"])}
    # (d) scanner reset is monotone and maps the discovery distribution onto itself at the discovery share
    xr = S["O"]["L-IMG"][rows, 0]; xm = SC.scanner_reset(xr, reset, pi=reset["pi_discovery_samples"]); mono = bool((np.diff(xm[np.argsort(xr, kind="mergesort")]) >= -1e-12).all())
    out = {"scorer_vs_engine_mismatches": dict(mm), "scorer_reproduces_engine": all(v == 0 for k, v in mm.items() if not k.startswith("score_")) and mm["score_tier0_maxabs_x1e9"] < 10 and mm["score_light_maxabs_x1e9"] < 10,
           "refit_files_identical_to_committed_bundles": ident, "insample_auroc_recomputed_vs_model_info": insample,
           "development_patient_auroc_registered_aggregation_nested": dev, "scanner_reset_monotone": mono, "scanner_reset_selfmap_maxabs": r3(np.abs(xm - xr).max())}
    print(json.dumps(out, indent=1)); jd(out, OUTR + "/verify.json"); jd(out, FZ + "/verify.json"); print("TPREG DONE verify", flush=True)
# ================================================================== manifest (after verify): sha256 of every bundle file
if RTASK == "manifest":
    for sub in ("models_their", "models_pkg"):
        if not os.path.exists(f"{FZ}/{sub}/extraction_settings.json"): subprocess.run(["python", PP + "/kc_freeze_manifest.py", f"{FZ}/{sub}", f"two_papers_REG_{sub}", f"scripts/paper_plan/{'kc' if sub == 'models_their' else 'kv'}_freeze.R refit"], check=True)
    os.makedirs(FZ + "/code", exist_ok=True)
    for s_ in json.load(open(FZ + "/rules.json"))["code"]["scripts"]: shutil.copy2(f"{T}/{s_}", f"{FZ}/code/{os.path.basename(s_)}")
    files = sorted(f for f in (os.path.relpath(p, FZ) for p in glob.glob(FZ + "/**/*", recursive=True) if os.path.isfile(p)) if f != "MANIFEST.json")
    man = {"bundle": "killcoyne_mm/two_papers_REG/frozen", "spec": "docs/two_papers_REG.md; paper/two_papers/REGISTRATION_draft.md", "frozen_on": json.load(open(FZ + "/rules.json"))["frozen_on"],
           "code_commit": json.load(open(FZ + "/rules.json"))["code"]["commit"], "code_copy_commit": subprocess.run(["git", "-C", T, "log", "-1", "--format=%H", "--", "scripts/paper_plan/tp_reg_freeze.py", "scripts/paper_plan/tp_reg_score.py"], capture_output=True, text=True).stdout.strip(), "files": {f: hashlib.sha256(open(f"{FZ}/{f}", "rb").read()).hexdigest() for f in files}}
    man["bundle_sha256"] = hashlib.sha256("".join(f"{k}:{v}\n" for k, v in sorted(man["files"].items())).encode()).hexdigest()
    jd(man, FZ + "/MANIFEST.json"); jd(man, OUTR + "/frozen_MANIFEST.json"); print(json.dumps(man, indent=1)); print("TPREG DONE manifest", flush=True)

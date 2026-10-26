"""Two papers, agent CLN (docs/two_papers_CLN.md): frozen bundle v2 (clean-tile L-IMG) with REG's freeze procedure.
tp_reg_freeze.py is executed with three textual substitutions only (each asserted to occur once): (1) output paths -> bundle killcoyne_mm/two_papers_REG/
frozen_v2_clean (v1 untouched), row-level killcoyne_mm/two_papers_CLN/freeze, aggregates results/paper_final/two_papers_CLN/freeze_v2; (2) the model swap
below is run right after its E9 import; (3) the scorer import -> tp_cln_score (REG's scorer + the mean-logit light / confirmatory tiers); in verify the
in-sample check reads the clean-tile features instead of img_mean.csv; in manifest code copies fall back to the CLN / ART code folders (the cluster
checkout cannot be pulled) and the bundle name / spec strings name v2.
Model swap: L-IMG = ART refit <CLN_MODEL> (outer + inner out-of-fold, stored folds), L-LATE = (L-CNV + L-IMG) / 2 for both copy-number flavours: the E9
namespace's load_stored / state() (as tp_art_gate.py state_v) and tb_common.load (picked up by every later 'from tb_common import *', i.e. the
quiet-genome check arrays built by check_arrays() -> ex_e2_veto.build, and TC.load('pkg')).
v2 tiers (one rule per tier; chosen on the CLN held-out results, docs/two_papers_CLN.md): R49 gate (+ package flavour for ACE-B / 4x), tier 0 = mean-logit
+ EW memory (REG), light = mean-logit gate + EW memory + copy-number certificate (E9 'ml_memcert'), confirmatory = light + quiet-genome check at the cn90 bar
(E9 'ml_final'); frozen with REG's sequence (gate cut-offs -> modal hyper-parameters -> median quiet cut -> median threshold; median cn90 bar) on the
mean-logit full context. REG's R49-gate light / confirmatory values are kept in freeze.json as the not-chosen alternative.
Tile QC (ART's flag, thresholds fixed; blur cut = 5th percentile of tissue-tile Laplacian variance in set C, stored as a constant) -> qc/qc_spec.json.
TASK = freeze | verify | manifest (REG's) ; CLN_MODEL = clean (bundle) | stored (smoke: the gate-tier values must equal bundle v1 exactly); CLN_FZ = bundle path."""
import os, sys, json, numpy as np, pandas as pd
THESIS = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"; PP = THESIS + "/scripts/paper_plan"; HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, PP)
RTASK = os.environ["TASK"]; MODEL = os.environ.get("CLN_MODEL", "clean"); M_ = THESIS + "/feasibility/paper_plan/killcoyne_mm"
FZ_ = os.environ.get("CLN_FZ", M_ + "/two_papers_REG/frozen_v2_clean"); assert FZ_ != M_ + "/two_papers_REG/frozen", "v1 bundle is read-only"
CODE_DIRS = [M_ + "/two_papers_CLN/code", M_ + "/two_papers_ART/code"]; CLN_COMMIT = os.environ.get("CLN_COMMIT", "")
IMGF = M_ + ("/img_mean.csv" if MODEL == "stored" else f"/two_papers_ART/feats/{MODEL}.csv.gz")
def rep1(s, a, b): assert s.count(a) == 1, a[:80]; return s.replace(a, b)
src = open(PP + "/tp_reg_freeze.py").read()
src = rep1(src, 'REG = M + "/two_papers_REG"; FZ = REG + "/frozen"; OUTR = T + "/results/paper_final/two_papers_REG"',
           'REG = M + "/two_papers_CLN/freeze" + ("" if CLN_MODEL == "clean" else "_" + CLN_MODEL); FZ = CLN_FZ; OUTR = T + "/results/paper_final/two_papers_CLN/freeze_v2" + ("" if CLN_MODEL == "clean" else "_smoke_" + CLN_MODEL)')
src = rep1(src, 'os.environ["TASK"] = RTASK\nimport tp_reg_score as SC', 'os.environ["TASK"] = RTASK\n_cln_swap(globals())\nimport tp_cln_score as SC')
src = rep1(src, 'IM = pd.read_csv(M + "/img_mean.csv", dtype={"Sample": str})', 'IM = pd.read_csv(CLN_IMGF, dtype={"Sample": str})')
src = rep1(src, 'shutil.copy2(f"{T}/{s_}", f"{FZ}/code/{os.path.basename(s_)}")', 'shutil.copy2(_cln_src(s_), f"{FZ}/code/{os.path.basename(s_)}")')
src = rep1(src, '"bundle": "killcoyne_mm/two_papers_REG/frozen", "spec": "docs/two_papers_REG.md; paper/two_papers/REGISTRATION_draft.md"',
           '"bundle": "killcoyne_mm/two_papers_REG/frozen_v2_clean", "spec": "docs/two_papers_CLN.md (v2: clean-tile L-IMG, tile QC); procedure docs/two_papers_REG.md", "cln_commit": CLN_COMMIT')
_cut = src.index('\nif RTASK == "nested":'); assert RTASK in ("freeze", "verify", "manifest")
def _cln_src(s_):
    for d in [THESIS] + CODE_DIRS:
        p = f"{d}/{s_}" if d == THESIS else f"{d}/{os.path.basename(s_)}"
        if os.path.exists(p): return p
    raise FileNotFoundError(s_)
def _cln_swap(g):
    """L-IMG -> ART refit MODEL everywhere (sample-level state, mean-logit state, check arrays, package flavour)"""
    TC = g["TC"]; g["RD"] = M_ + "/two_papers_ART/refit"
    if MODEL == "stored": Oi, Ii = g["O0"]["L-IMG"], g["I0"]["L-IMG"]
    else: Oi, Ii = g["variant"](MODEL)[:2]
    def sw(O, I):
        O = dict(O); I = dict(I); O["L-IMG"] = Oi; O["L-LATE"] = (O["L-CNV"] + Oi) / 2; I["L-IMG"] = Ii
        I["L-LATE"] = {k: (I["L-CNV"][k] + Ii[k].reindex(I["L-CNV"][k].index)) / 2 for k in I["L-CNV"]}; return O, I
    load0 = TC.load
    def load_cln(srcname): O, I, FO = load0(srcname); O, I = sw(O, I); return O, I, FO
    TC.load = load_cln; Os, Is = sw(g["O0"], g["I0"]); FO = g["FO"]
    g["load_stored"] = lambda s: (Os, Is, FO) if s == "their" else load_cln(s); g["load"] = g["load_stored"]
    def state(var):
        if var == "st": g["load"] = g["load_stored"]; return g["prep_state"]()
        Om, Im, Op, Ip, _ = g["variant"]("mlogit:" + MODEL); sg_, lg_ = g["sg"], g["lg_"]
        Om = sg_(lg_(Om) + 1e-6 * lg_(Op)); Im = {k: pd.Series(sg_(lg_(Im[k].values) + 1e-6 * lg_(Ip[k].reindex(Im[k].index).values)), index=Im[k].index) for k in Im}
        O = dict(g["O0"]); I = dict(g["I0"]); O["L-IMG"] = Om; O["L-LATE"] = (O["L-CNV"] + Op) / 2; I["L-IMG"] = Im; I["L-LATE"] = {k: (I["L-CNV"][k] + Ip[k].reindex(I["L-CNV"][k].index)) / 2 for k in I["L-CNV"]}
        g["load"] = lambda s: (O, I, FO)
        try: return g["prep_state"]()
        finally: g["load"] = g["load_stored"]
    g["state"] = state
NSF = {"__file__": PP + "/tp_reg_freeze.py", "__name__": "tp_reg_freeze_cln", "CLN_MODEL": MODEL, "CLN_FZ": FZ_, "CLN_IMGF": IMGF, "CLN_COMMIT": CLN_COMMIT, "_cln_swap": _cln_swap, "_cln_src": _cln_src}
exec(compile(src, PP + "/tp_reg_freeze.py", "exec"), NSF)   # REG's own task branch runs here (freeze / verify / manifest)
g = NSF; jd = g["jd"]; FZ = g["FZ"]; REG = g["REG"]; OUTR = g["OUTR"]
# ================================================================== freeze (after REG's): mean-logit light / confirmatory tiers, tile QC spec, code list
if RTASK == "freeze":
    S, ST, Sm, STm, fc, V, k2, barf = (g[k] for k in ("S", "ST", "Sm", "STm", "fc", "V", "k2", "barf")); full_ctx, pick, f_A12, a12_cand, thr_for, pat_scores, mode_first, med, ev, with_check = (
        g[k] for k in ("full_ctx", "pick", "f_A12", "a12_cand", "thr_for", "pat_scores", "mode_first", "med", "ev", "with_check"))
    E = S["E"]; rules = json.load(open(FZ + "/rules.json")); fr = json.load(open(OUTR + "/freeze.json"))
    # REG sequence on the mean-logit full context: modal A12 tuple -> median quiet cut -> median threshold (gate cut-offs = REG's frozen ml_cuts)
    ch = []
    for j in range(10): cm = full_ctx(Sm, STm, j, fc["ml_cuts"]); cands = f_A12(cm); q, _, _, _ = pick(cm, cands); p_ = cands[q]["params"]; ch.append((p_["h"], p_["t_quiet_pct"], p_["delta"], p_["Y"]))
    a12m, cnt = mode_first(ch); qcm = [a12_cand(full_ctx(Sm, STm, j, fc["ml_cuts"]), *a12m)["qcut"] for j in range(10)]; qcmf = med(qcm)
    tml = [thr_for(cm, pat_scores(cm, a12_cand(cm, *a12m, qcut=qcmf)["u"])) for cm in (full_ctx(Sm, STm, j, fc["ml_cuts"]) for j in range(10))]; tmlf = med(tml)
    Z = dict(np.load(REG + "/apparent_flags.npz")); FL = np.zeros((E, 10), bool); NL = np.zeros((E, 10), int); UL = np.zeros((E, 10)); FC = np.zeros((E, 10), bool); NC = np.zeros((E, 10), int)
    for j in range(10):
        cm = full_ctx(Sm, STm, j, fc["ml_cuts"]); x = a12_cand(cm, *a12m, qcut=qcmf); UL[:, j] = x["u"]; FL[:, j] = x["u"] >= tmlf; NL[:, j] = x["ns"]
        FC[:, j], NC[:, j] = with_check(FL[:, j], NL[:, j], cm.r49_band, V[:, j], barf, k2)
    for k in ("light", "confirm"): Z[f"F_{k}_r49gate"] = Z[f"F_{k}"]; Z[f"NS_{k}_r49gate"] = Z[f"NS_{k}"]
    Z["U_light_r49gate"] = Z["U_light"]; Z.update(F_light=FL, NS_light=NL, U_light=UL, F_confirm=FC, NS_confirm=NC); np.savez_compressed(REG + "/apparent_flags.npz", **Z)
    ck = lambda t: {"t_low": t[0], "c_star": t[1], "t_seq": t[2]}; alt = {k: rules["rules"][k] for k in ("tier_light", "tier_confirmatory")}
    rules["rules"]["tier_light"] = {"gate": "mean_logit", "gate_cuts": ck(fc["ml_cuts"]), "half_life_years": a12m[0], "quiet_percentile": a12m[1], "quiet_cut_logit_cnv": qcmf, "delta_logit": a12m[2], "years": a12m[3], "threshold": tmlf,
        "definition": "mean-logit gate: endoscopy score z = mean of the sample L-IMG logits (1e-6 x top-sample logit tie-break); low if sigmoid(z) < t_low, high if > c_star, middle: the top-L-IMG sample is sequenced; gate-aligned score g = z, or in the middle band logit(c_star) + logit(L-LATE of the sequenced top sample) - logit(t_seq); EW memory of g with half-life h; minus delta while certified (an earlier middle-band endoscopy <= Y years before whose sequenced top sample had logit L-CNV < quiet cut); flag if >= threshold. Sequences the mean-logit gate's middle-band top sample."}
    rules["rules"]["tier_confirmatory"] = {"base": "tier_light", "check_bar_ps_oac": barf, "check_samples": 2, "definition": alt["tier_confirmatory"]["definition"].replace("(one already sequenced in the middle band)", "(one already sequenced in the mean-logit gate's middle band)")}
    spec_q = None
    if MODEL != "stored":
        Q = pd.read_parquet(M_ + "/two_papers_ART/qc_all.parquet"); bg = Q.tissue.values < 0.15; okq = ~bg & np.isfinite(Q.lapvar_t.values); lap = float(np.nanquantile(Q.lapvar_t.values[okq], 0.05))
        assert abs(lap - json.load(open(THESIS + "/results/paper_final/two_papers_ART/feats.json"))["feats"]["lap_thr"]) < 1e-3
        spec_q = {"thresholds": {"tissue_min": 0.15, "line_d_min": 0.5, "ink_max": 0.02, "lapvar_t_min": lap, "min_tiles": 32},
                  "tile_geometry": "UNI2-h tiles of the L-IMG bags: 224 px at level 2 (about 0.88 um/px), stored coordinates are tile CENTRES (top-left = centre - 112, clamped to the level; extract_foundation_from_masks.py); 256 tiles per slide",
                  "pixel_statistics": "tp_art_qc.py pixel_stats (code/tp_art_qc.py): tissue = share of pixels with HSV saturation > 0.15 and not white (all RGB > 220); line_d = max over rows and columns (3-line running mean) of the share of dark unstained pixels (s < 0.12, grey < 160); ink = share with hue 80-250 deg, s > 0.25, not white; lapvar_t = variance of the Laplacian of grey over tissue pixels (1-px border dropped), missing if < 500 tissue pixels",
                  "flag": "artefact = (tissue < tissue_min) OR (not glass AND line_d >= line_d_min) OR (ink > ink_max) OR (not glass AND lapvar_t < lapvar_t_min, missing counted as below)",
                  "embedding": "sample embedding = mean of the clean tiles' UNI2-h embeddings over all the sample's slides; fewer than min_tiles clean: top up to min_tiles with flagged tiles by highest tissue share, then highest lapvar_t, then bag order (code/tp_cln_pipeline.py)",
                  "lapvar_t_min_origin": "5th percentile of lapvar_t over non-glass tiles with a value, all set-C tiles (ART, docs/two_papers_ART.md); fixed here as a constant", "source": "agent ART flag, docs/two_papers_ART.md section 1"}
        os.makedirs(FZ + "/qc", exist_ok=True); jd(spec_q, FZ + "/qc/qc_spec.json")
    rules["pipeline"] = ["tile QC (qc/qc_spec.json; code/tp_art_qc.py statistics, code/tp_cln_pipeline.py flag and embedding)", "clean-tile sample embedding -> L-IMG (models_*/coef_L_IMG.csv, image_scaling.csv)",
                         "scanner reset of L-IMG (scanner_reset.json) when the scanner differs", "L-CNV from copy number (models_their: their matrix; models_pkg: package features from reads)", "L-LATE = (L-CNV + L-IMG) / 2", "rules (this file); scorer code/tp_cln_score.py"]
    rules["version"] = "v2_clean"; rules["selection"]["training_predictions"] = ("outer out-of-fold predictions of the stored 10 x 10 patient-grouped CV with L-IMG refitted on the clean-tile mean embedding (agent ART, "
                                                                                 "killcoyne_mm/two_papers_ART/refit/clean; same folds, recipe and nested lambda; L-CNV their cfg 00 / package cfg 01), one repeat at a time, in the role of inner out-of-fold predictions")
    rules["code"] = {"scripts": rules["code"]["scripts"] + ["scripts/paper_plan/tp_cln_freeze.py", "scripts/paper_plan/tp_cln_freeze.R", "scripts/paper_plan/tp_cln_score.py", "scripts/paper_plan/tp_cln_pipeline.py", "scripts/paper_plan/tp_art_qc.py", "scripts/paper_plan/tp_art_refit.R", "scripts/paper_plan/tb_common.py", "scripts/paper_plan/rv8_landscape.py", "scripts/paper_plan/rv8b_step3.py"],
                     "commit": rules["code"]["commit"], "cln_commit": CLN_COMMIT}
    jd(rules, FZ + "/rules.json")
    app = {k: ev(S, *v) for k, v in {"light": (FL, NL), "confirm": (FC, NC)}.items()}
    fr["frozen"] = rules["rules"]; fr["not_chosen_r49gate_light_and_confirmatory"] = alt; fr["apparent_OPTIMISTIC_frozen_rules_on_oof"].update({f"{k}_r49gate": fr["apparent_OPTIMISTIC_frozen_rules_on_oof"][k] for k in ("light", "confirm")}); fr["apparent_OPTIMISTIC_frozen_rules_on_oof"].update(app)
    fr["trace"]["ml_a12_choice_per_repeat"] = [list(x) for x in ch]; fr["trace"]["ml_a12_choice_counts"] = {str(k): v for k, v in cnt.items()}; fr["trace"]["ml_quiet_cut_per_repeat"] = qcm; fr["trace"]["ml_a12_threshold_per_repeat"] = tml
    fr["model"] = MODEL; fr["qc_spec"] = spec_q; jd(fr, OUTR + "/freeze.json"); print(json.dumps({"ml_a12": a12m, "counts": {str(k): v for k, v in cnt.items()}, "qcut": qcmf, "thr": tmlf, "apparent": fr["apparent_OPTIMISTIC_frozen_rules_on_oof"]}, indent=1)); print("TPCLN DONE freeze", flush=True)
# ================================================================== verify (after REG's): v1 reproduction of the R scripts, flag-level scorer result
if RTASK == "verify":
    import hashlib; vf = json.load(open(OUTR + "/verify.json")); sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest(); sm = {}
    for sub, ref in (("models_their", "killcoyne_frozen_v1"), ("models_pkg", "killcoyne_frozen_pkg_v1")):
        for f_, h_ in json.load(open(f"{THESIS}/models/{ref}/MANIFEST.json"))["files"].items():
            p = f"{M_}/two_papers_CLN/freeze_smoke/{sub}/{f_}"
            if os.path.exists(p): sm[f"{sub}/{f_}"] = sha(p) == h_
    vf["smoke_tp_cln_freeze_R_on_v1_input_byte_identical_to_v1"] = sm
    vf["note_refit_files_identical_to_committed_bundles"] = "v2 models are refitted on the clean-tile features: L-CNV coefficients and copy-number scaling must equal v1 (True); image-dependent files (L-IMG, L-EARLY, image_scaling) differ by design"
    pv = M_ + "/two_papers_CLN/freeze/verify_pipeline.json"; vf["pipeline_verify"] = json.load(open(pv)) if os.path.exists(pv) else "missing"
    vf["model"] = MODEL; jd(vf, OUTR + "/verify.json"); jd(vf, FZ + "/verify.json"); print("TPCLN DONE verify", vf["scorer_reproduces_engine"], flush=True)

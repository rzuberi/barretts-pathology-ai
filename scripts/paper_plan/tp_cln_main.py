"""Two papers, agent CLN (docs/two_papers_CLN.md): Paper 2 tiers re-established on the clean-tile L-IMG (agent ART), with the published L-IMG and ART's
random-tile-removal control run through the same code. EXPLORATORY.
No rule is re-implemented: the module body of tp_p2s_main.py is executed up to its dispatch (it executes ex_e7_stress.py, ex_e2_veto.py, rv9_falseflags.py
and the helper section of ex_e3_score.py); the model is swapped at the loader only and P2S's own task branches (rules | landmark | design | per1000) are
then executed unchanged with the row-level and aggregate paths redirected to the CLN space.
  model swap, sample level: round-8 setup's load("their") inside ex_e2_veto.build (namespace g8) returns L-IMG = ART refit <model> (outer and inner
      out-of-fold predictions, stored folds) and L-LATE = (stored L-CNV + new L-IMG) / 2, so the R49 gate cut-offs, middle band, top-H&E samples, certificate
      quiet cut, check bars and every threshold are re-derived on the new model inside the training folds (as tp_art_gate.py via prep_state);
  model swap, mean-logit tiers: E3 variant("mlogit:<model>") exactly as P2S state_ml / E9 state('ml') / tp_art_gate.py state_v.
Additions: (1) the mean-logit full stack (E9 'ml_final': A12 on the mean-logit gate + quiet-genome check at the cn90 bar, k2 - 1 samples in the mean-logit
gate's middle band) and its fold-internal bar (E7 infold_bar), appended to rules.npz; (2) TASK=compare: cross-model paired contrasts (patient bootstrap,
rv8 pdraws RandomState(0); equal exposure with P2S/E7's truncation draws RandomState(7001), 200 draws).
Models (CLN_MODEL): stored (published L-IMG; must reproduce P2S / E9) | clean (ART clean-tile refit) | rand (ART random-removal control, same tile count).
Row-level files only under killcoyne_mm/two_papers_CLN/<model>/ on the cluster; aggregates (no identifiers, no dates, patient cells < 5 as '<5') ->
results/paper_final/two_papers_CLN/<model>/<task>.json and results/paper_final/two_papers_CLN/compare.json.
TASK = rules | landmark | design | per1000 (CLN_MODEL=...) | compare.   CLN_NB=k: smoke (k bootstrap draws, aggregates written aside)."""
import os, sys, json, shutil, time, numpy as np, pandas as pd
THESIS = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"; PP = THESIS + "/scripts/paper_plan"; sys.path.insert(0, PP)
CTASK = os.environ["TASK"]; MODEL = os.environ.get("CLN_MODEL", "stored"); assert MODEL in ("stored", "clean", "rand"), MODEL
_src = open(PP + "/tp_p2s_main.py").read(); _cut = _src.index('\nres = {"task": TASK')
G = {"__file__": PP + "/tp_p2s_main.py", "__name__": "p2s_cln"}
exec(compile(_src[:_cut], PP + "/tp_p2s_main.py", "exec"), G)   # P2S definitions only (its dispatch is executed below for the P2S tasks)
os.environ["TASK"] = CTASK
G7, G9, G2, g8 = G["G7"], G["G9"], G["G2"], G["g8"]; M = G["M"]; T = G["T"]; r2 = G["r2"]; r3 = G["r3"]; m5 = G["m5"]
CX = M + "/two_papers_CLN"; OUTC = T + "/results/paper_final/two_papers_CLN"; SMOKE = bool(os.environ.get("CLN_NB"))
if SMOKE: G["NB"] = int(os.environ["CLN_NB"]); OUTC = CX + "/smoke_json"
G["SMOKE"] = SMOKE; XDm = f"{CX}/{MODEL}"; OUTm = f"{OUTC}/{MODEL}"
for d_ in (CX, XDm, OUTC, OUTm): os.makedirs(d_, exist_ok=True)
G["XD"] = XDm; G["OUT"] = OUTm
if not os.path.exists(XDm + "/clin.npz"): shutil.copy2(M + "/two_papers_P2S/clin.npz", XDm + "/clin.npz")   # P2S clinical table (model-free), read-only copy
def jd(o, path): json.dump(o, open(path, "w"), indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x)); print("TPCLN DONE", path, flush=True)
# ================================================================== model swap
ART_RD = M + "/two_papers_ART/refit"; LOAD8 = g8["load"]
def swap(model):
    """sample-level L-IMG of the round-8 setup inside ex_e2_veto.build (g8 namespace): stored or an ART refit; L-LATE = (stored L-CNV + L-IMG) / 2"""
    if model == "stored": g8["load"] = LOAD8; return
    G9["RD"] = ART_RD; Oi, Ii = G9["variant"](model)[:2]; O0, I0, FO = G9["O0"], G9["I0"], G9["FO"]
    O = dict(O0); I = dict(I0); O["L-IMG"] = Oi; O["L-LATE"] = (O["L-CNV"] + Oi) / 2; I["L-IMG"] = Ii; I["L-LATE"] = {k: (I["L-CNV"][k] + Ii[k].reindex(I["L-CNV"][k].index)) / 2 for k in I["L-CNV"]}
    def ld(src): assert src == "their", src; return (O, I, FO)
    g8["load"] = ld
def state_ml():
    """P2S state_ml with the model's mean-logit variant (E3 'mlogit:<model>')"""
    g = G9; g["RD"] = ART_RD; Oi, Ii, Op, Ip, _ = g["variant"]("mlogit:" + MODEL); sg, lg_ = g["sg"], g["lg_"]
    Oi = sg(lg_(Oi) + 1e-6 * lg_(Op)); Ii = {k: pd.Series(sg(lg_(Ii[k].values) + 1e-6 * lg_(Ip[k].reindex(Ii[k].index).values)), index=Ii[k].index) for k in Ii}
    O = dict(g["O0"]); I = dict(g["I0"]); O["L-IMG"] = Oi; O["L-LATE"] = (O["L-CNV"] + Op) / 2; I["L-IMG"] = Ii; I["L-LATE"] = {k: (I["L-CNV"][k] + Ip[k].reindex(I["L-CNV"][k].index)) / 2 for k in I["L-CNV"]}
    keep = g["load"]; g["load"] = lambda src: (O, I, g["FO"])
    try: return g["prep_state"]()
    finally: g["load"] = keep
swap(MODEL); G["state_ml"] = state_ml
G["MAIN"] = ["ml_mem", "final", "final_infold", "ml_final", "ml_final_infold"]
G["NAMES"].update({"ml_final": "Mean-logit full stack (mean-logit gate + memory + certificate + check, cn90 bar)", "ml_final_infold": "Mean-logit full stack, fold-internal bar",
                   "ml_A12": "Mean-logit gate + memory + certificate (no check)"})
# ================================================================== P2S tasks on this model
if CTASK in ("rules", "landmark", "design", "per1000"):
    t0 = time.time(); exec(compile(_src[_cut:], PP + "/tp_p2s_main.py", "exec"), G)   # P2S dispatch, unchanged (writes OUTm/<task>.json, rules -> XDm/rules.npz)
    if CTASK == "rules":   # mean-logit full stack (E9 ml_final) and its fold-internal bar, appended to rules.npz
        S, FT, ST, INFO, V, VTR, TH, TV = G["state_st"](); Sm, STm, INFOm = state_ml(); assert (Sm["plab"] == S["plab"]).all() and all((a == b).all() for a, b in zip(Sm["es"], S["es"]))
        GBml = g8["r49_calls"](Sm["O"], INFOm, Sm["FO"], Sm["es"])[2]; e = G["engine2"](Sm, STm, INFOm, G9["FAMS"]["A12"])
        Ra = {"ml_A12": (e["F"], e["NS"])}; Ra["ml_final"] = G7["apply_check"](S, e["F"], e["NS"], GBml, V, TH[0.90]); THi, pk = G7["infold_bar"](S, e["TRF"], VTR, TV)
        Ra["ml_final_infold"] = G7["apply_check"](S, e["F"], e["NS"], GBml, V, THi)
        z = dict(np.load(XDm + "/rules.npz")); z.update({f"{k}__F": v[0] for k, v in Ra.items()}); z.update({f"{k}__NS": v[1].astype(np.int16) for k, v in Ra.items()}); z["ml_A12__MG"] = e["MG"]
        np.savez_compressed(XDm + "/rules.npz", **z); np.savez_compressed(XDm + "/bands.npz", GB=S["GB"], GBml=GBml)
        r = json.load(open(OUTm + "/rules.json")); obs = r["reproduction"]["observed"]
        for k, v in Ra.items(): p_ = G7["pt"](S, *v); obs[k] = [p_["caught"], p_["np_flagged"], p_["seq_per_patient"]]
        r["reproduction"]["selection"]["ml_A12"] = {"chosen": {e["labels"][q]: int(np.sum(np.array(e["chosen"]) == q)) for q in set(e["chosen"])}}; r["reproduction"]["selection"]["ml_infold_bar_picks"] = pk
        exp = {"stored": {**{k: list(v) for k, v in G["EXPECT"].items()}, "ml_A12": [30.3, 20.1], "ml_final": [30.3, 16.2], "P1": [31.1, 36.0]},   # P2S / E9 / REG nested smoke
               "clean": {"R49": [31.3, 32.8], "A1": [30.7, 25.1], "A12": [30.5, 23.5], "ml_mem": [31.1, 29.7]}}.get(MODEL, {})   # ART gate.json
        r["model"] = MODEL; r["smoke_expected"] = exp; r["smoke_reproduced"] = {k: bool(abs(obs[k][0] - e_[0]) < 0.051 and abs(obs[k][1] - e_[1]) < 0.051) for k, e_ in exp.items()}
        r["gate_band_shares"] = {"st_middle": r3((S["GB"] == 1).mean()), "ml_middle": r3((GBml == 1).mean())}; r["seconds_total"] = round(time.time() - t0)
        jd(r, OUTm + "/rules.json"); print(json.dumps({k: obs[k] for k in obs if not k.startswith(("prague", "parasa"))}), json.dumps(r["smoke_reproduced"]), flush=True)
# ================================================================== compare: cross-model paired contrasts (raw counts, equal exposure)
if CTASK == "compare":
    assert MODEL == "stored"
    S, FT, ST, INFO, V, VTR, TH, TV = G["state_st"](); D = S["D"]; isP = S["plab"] == 1; patient_arrays = G["patient_arrays"]
    MODELS_ = ("stored", "clean", "rand"); RR = {}
    for m in MODELS_:
        z = np.load(f"{CX}/{m}/rules.npz"); RR[m] = {k[:-3]: (z[k].astype(bool), z[k[:-3] + "__NS"].astype(int)) for k in z.files if k.endswith("__F")}
    TIERS = ["R49", "P1", "A1", "A12", "final", "final_infold", "ml_mem", "ml_single", "gate_single", "ml_A12", "ml_final", "ml_final_infold", "P3", "g53"]
    A = {m: {k: patient_arrays(S, *RR[m][k]) for k in TIERS} for m in MODELS_}; out = {"tiers": TIERS, "names": {k: G["NAMES"].get(k, k) for k in TIERS}}
    raw = {}
    for k in TIERS:
        o = {m: G["tier_block"](S, A[m][k], D) for m in MODELS_}
        for m in MODELS_:
            if k != "R49": o[f"{m}_vs_R49_same_model"] = G7["delta"](S, A[m][k], A[m]["R49"], D)
            if k != "ml_mem": o[f"{m}_vs_tier0_same_model"] = G7["delta"](S, A[m][k], A[m]["ml_mem"], D)
        o["clean_minus_stored"] = G7["delta"](S, A["clean"][k], A["stored"][k], D); o["rand_minus_stored"] = G7["delta"](S, A["rand"][k], A["stored"][k], D)
        o["clean_minus_rand"] = G7["delta"](S, A["clean"][k], A["rand"][k], D)
        o["saving_vs_R49_clean_minus_stored"] = None
        raw[k] = o
    # saving against the same model's R49 (tier - R49), clean minus stored: does the tier's advantage over the gate shrink?
    for k in TIERS:
        if k == "R49": continue
        one = np.ones((1, S["P_"])); ls = G["lstats"]; nN = int((~isP).sum())
        sv = {m: (ls(S, A[m][k], one)["np_flagged_share"][0] - ls(S, A[m]["R49"], one)["np_flagged_share"][0], ls(S, A[m][k], D)["np_flagged_share"] - ls(S, A[m]["R49"], D)["np_flagged_share"]) for m in ("stored", "clean", "rand")}
        raw[k]["saving_vs_R49_clean_minus_stored"] = G["dstat"](nN * (sv["clean"][0] - sv["stored"][0]), nN * (sv["clean"][1] - sv["stored"][1]))
        raw[k]["saving_vs_R49_clean_minus_rand"] = G["dstat"](nN * (sv["clean"][0] - sv["rand"][0]), nN * (sv["clean"][1] - sv["rand"][1]))
    out["raw"] = raw
    # equal exposure (E7 test 1(a) as P2S landmark: NP histories truncated to m endoscopies, m ~ progressors' counts; 200 draws, RandomState(7001)); frozen per-fold rules
    rng = np.random.RandomState(7001); NE = S["NE"].astype(int); NEP = NE[isP]; POS = S["POS"]; masks = []
    for d in range(200): mm = np.where(isP, NE, np.minimum(NE, rng.choice(NEP, S["P_"]))); masks.append(POS < mm[S["pidx"]])
    PYs = [G7["trunc_py"](S, ST, kp) for kp in masks]; Se = dict(S, PY=np.mean(PYs, 0)); AE = {}
    for m in MODELS_:
        AE[m] = {}
        for k in TIERS:
            L = [patient_arrays(dict(S, PY=PYs[i]), RR[m][k][0], RR[m][k][1], kp) for i, kp in enumerate(masks)]; AE[m][k] = G7["avgA"](L)
    ee = {}
    for k in TIERS:
        o = {m: G7["full"](Se, AE[m][k], D) for m in MODELS_}
        for m in MODELS_:
            if k != "R49": o[f"{m}_vs_R49_same_model"] = G7["delta"](Se, AE[m][k], AE[m]["R49"], D)
        o["clean_minus_stored"] = G7["delta"](Se, AE["clean"][k], AE["stored"][k], D); o["clean_minus_rand"] = G7["delta"](Se, AE["clean"][k], AE["rand"][k], D)
        ee[k] = o
    for a, s_ in (("ml_mem", "ml_single"), ("A1", "gate_single"), ("A12", "gate_single")):
        ee[f"{a} - {s_} (memory gain)"] = {m: G7["delta"](Se, AE[m][a], AE[m][s_], D) for m in MODELS_}
    ee["check_stored_equals_P2S"] = {"ml_mem_vs_R49_np": ee["ml_mem"]["stored_vs_R49_same_model"]["np_flagged"]["delta"], "expected_P2S": -3.6,
                                     "final_vs_R49_np": ee["final"]["stored_vs_R49_same_model"]["np_flagged"]["delta"], "expected_P2S_final": -11.3}
    out["equal_exposure"] = ee
    # within-model: mean-logit stack minus R49-gate stack (choice of the light / confirmatory tier for bundle v2), raw and at equal exposure
    out["mean_logit_minus_gate_stack"] = {f"{a} - {b}": {m: {"raw": G7["delta"](S, A[m][a], A[m][b], D), "equal_exposure": G7["delta"](Se, AE[m][a], AE[m][b], D)} for m in MODELS_}
                                          for a, b in (("ml_A12", "A12"), ("ml_final", "final"), ("ml_final_infold", "final_infold"), ("ml_final", "final_infold"))}
    out["note"] = ("held-out; every threshold / hyper-parameter / bar chosen on training patients of each outer fold (inner out-of-fold predictions of the model in question); "
                   "paired patient bootstrap on the same draws for all three models (rv8 pdraws, RandomState(0)); stored = published L-IMG; clean = ART clean-tile L-IMG; "
                   "rand = ART random-tile-removal control (same number of tiles per sample); L-LATE = (stored L-CNV + L-IMG) / 2 in every model")
    jd(out, OUTC + "/compare.json")

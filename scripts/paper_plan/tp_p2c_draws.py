"""Two papers, agent P2C (docs/two_papers_P2C.md): paired patient-bootstrap draws of the class-specific per-patient rates that feed Paper 2's
early cost-consequence model. No rule is re-implemented: ex_e9_frontier.py is executed unchanged with a task name matching none of its branches
(state, sweep, rates, check_arrays, path_scores, refs, pdraws and the round-9 / round-8 helpers), and its stored sweeps
(killcoyne_mm/explore_E9/sweep_*.npz) are read. with_check() below is copied verbatim from ex_e9_frontier.py's merge block.
The only new rule is the sequence-everything comparator (max L-LATE over all samples of an endoscopy, round-9 base 'late_max'), swept on the same
E9 engine (TASK=p2c_late). Everything chosen on training patients; rates on held-out patients at the operating target D.
Row-level arrays only under killcoyne_mm/two_papers_P2C/; aggregates (bootstrap draws of cohort-level rates, no identifiers) ->
results/paper_final/two_papers_P2C/rate_draws.json.  TASK = p2c_late | p2c_draws.  P2C_NB=k: smoke (k draws, written aside)."""
import os, sys, json, glob, time, numpy as np
THESIS = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"; PP = THESIS + "/scripts/paper_plan"; sys.path.insert(0, PP)
P2TASK = os.environ["TASK"]; os.environ["TASK"] = "p2c_defs"
_f = __file__; __file__ = PP + "/ex_e9_frontier.py"; exec(compile(open(__file__).read(), __file__, "exec")); __file__ = _f   # definitions only
os.environ["TASK"] = P2TASK
XP = M + "/two_papers_P2C"; OUTP = T + "/results/paper_final/two_papers_P2C"; NBP = 2000
if os.environ.get("P2C_NB"): NBP = int(os.environ["P2C_NB"]); OUTP = XP + "/smoke_json"
for d_ in (XP, OUTP): os.makedirs(d_, exist_ok=True)
t0 = time.time()
# ------------------------------------------------------------------ sequence-everything comparator on the E9 sweep engine
if P2TASK == "p2c_late":
    S, ST, INFO = state("st"); b = {"late_max:none": f_score("late_max", "none"), "late_max:ew": f_score("late_max", "ew")}
    R, BAND, DD, NTP = sweep(S, ST, INFO, list(b), b); arrs = {"BAND": BAND, "D": DD, "NTP": NTP}; meta = {}
    for f, Rf in R.items():
        key = f.replace(":", "_"); arrs.update({f"{key}__F": Rf["F"], f"{key}__NS": Rf["NS"], f"{key}__chosen": Rf["chosen"], f"{key}__infeasible": Rf["infeasible"]}); meta[f] = {"key": key, "labels": Rf["labels"]}
    np.savez_compressed(f"{XP}/sweep_st_late.npz", **arrs); json.dump({"meta": meta, "targets": TNAMES, "seconds": round(time.time() - t0, 1)}, open(f"{XP}/sweep_st_late.json", "w"))
    print("P2C DONE late", round(time.time() - t0, 1), flush=True)
# ------------------------------------------------------------------ paired draws at target D
if P2TASK == "p2c_draws":
    S, ST, INFO = state("st"); D = pdraws(S, NBP); MB = np.vstack([np.ones((1, S["P_"])), D]); E = S["E"]
    CK = check_arrays(); assert CK["S2"]["E"] == E and (CK["S2"]["plab"] == S["plab"]).all(); V = CK["V"]; THc = CK["TH"]; k2 = CK["k2"][:, None]
    Z = {os.path.basename(f)[6:-4]: np.load(f) for f in sorted(glob.glob(XD9 + "/sweep_*.npz"))}; Z["st_late"] = np.load(XP + "/sweep_st_late.npz")
    assert (Z["st_core"]["BAND"] == CK["GB"]).all(), "engine gate band differs from r49_calls"
    def arr(grp, fam, ti=0):
        key = fam.replace(":", "_"); return Z[grp][f"{key}__F"][ti], Z[grp][f"{key}__NS"][ti].astype(int)
    def with_check(FN, band):   # verbatim from ex_e9_frontier.py (merge), target-D slices
        F, NS = FN; veto = F & (V < THc); kk = k2 if band is None else np.where(band == 1, k2 - 1, k2)
        return F & ~veto, NS + np.where(F, kk, 0)
    Bst = Z["st_core"]["BAND"]; Bml = Z["ml_core"]["BAND"]; PS = path_scores(S); rep = lambda b_: np.repeat(b_[:, None], 10, 1); Zi = np.zeros((E, 10), int); RF = refs(S)
    G = rep(PS["g53"] >= 1); L = rep(PS["g53"] >= 2)   # L: LGD, or p53 aberrant (E9's g53 score >= 2) - the override in Paper 2's eligibility statement
    AI = {"ml_mem": arr("ml_core", "img:ew"), "ml_gatemem": arr("ml_core", "gate_al:ew"), "ml_memcert": arr("ml_A12", "A12"), "memcert": arr("st_A12", "A12"),
          "R49": RF["R49"], "seq_all": arr("st_late", "late_max:none"), "seq_all_mem": arr("st_late", "late_max:ew")}
    AI["ml_final"] = with_check(AI["ml_memcert"], Bml); AI["final"] = with_check(AI["memcert"], Bst)
    RULES = {"g53": (G, Zi), "grade": (rep(PS["grade"] >= 1), Zi), "l53": (L, Zi)}
    for nm, (F, NS) in AI.items(): RULES[nm] = (F, NS); RULES[nm + "_or_g53"] = (F | G, NS); RULES[nm + "_not_g53"] = (F & ~G, NS); RULES[nm + "_or_l53"] = (F | L, NS)
    out = {"note": ("P2C: class-specific per-patient rates at operating target D (held-out patients; thresholds and hyper-parameters chosen on training patients), "
                    f"row 0 = observed, rows 1.. = {NBP} paired patient-bootstrap draws (rv8 pdraws, RandomState(0)); repeat-averaged as ex_e9_frontier.rates. "
                    "sens / fpr = share of progressor / non-progressor patients ever flagged; nsp, nsn = samples sequenced per progressor / non-progressor patient; "
                    "nfp, nfn = flagged endoscopies per patient; nep, nen = pre-event endoscopies per patient; pyn = non-progressor person-years per patient. "
                    "<rule>_or_g53 = rule OR pathologist grade IND/LGD or p53 aberrant (union flags; never-flagged = de-escalation eligible); <rule>_not_g53 = endoscopies flagged "
                    "by the rule but not by grade/p53 (incremental work-ups); l53 = LGD or p53 aberrant (g53 score >= 2); <rule>_or_l53 = rule OR l53; ml_final / final = check applied to ml_memcert / memcert (E9 construction); seq_all = max L-LATE of every sample, "
                    "threshold only; seq_all_mem = same with EW memory"),
           "n_progressor_patients": int((S["plab"] == 1).sum()), "n_non_progressor_patients": int((S["plab"] == 0).sum()), "n_draws": NBP, "rules": {}}
    KEYS = ("sens", "fpr", "nsp", "nsn", "nfp", "nfn", "nep", "nen", "pyn")
    for nm, (F, NS) in RULES.items():
        r = rates(S, patient_arrays(S, F, NS)); out["rules"][nm] = {k: [round(float(x), 5) for x in r[k]] for k in KEYS}
        print(nm, {k: round(float(r[k][0]), 3) for k in ("sens", "fpr", "nsp", "nsn", "nfp", "nfn")}, flush=True)
    for nm in ("seq_all", "seq_all_mem"):
        out[f"{nm}_selected"] = {str(k): int(v) for k, v in zip(*np.unique(Z["st_late"][nm.replace("seq_all_mem", "late_max_ew").replace("seq_all", "late_max_none") + "__chosen"][0], return_counts=True))}
    json.dump(out, open(f"{OUTP}/rate_draws.json", "w")); print("P2C DONE draws", round(time.time() - t0, 1), flush=True)

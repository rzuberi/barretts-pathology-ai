"""Two-paper agent P2S (docs/two_papers_P2S.md): Paper 2 safety and design analyses (paper/two_papers/PAPER2_plan_v2.md items 2-8) for the two main tiers,
  no sequencing  = mean-logit endoscopy score + EW memory (E3 / E9 'A1:img:ew' on the mean-logit variant; 31.0 caught / 26.6 NP flagged),
  confirmatory   = final rule (A12 memory + certificate + quiet-genome check, cn90 bar; 30.5 / 17.2) and its fold-internal bar (30.4 / 16.4),
with the R49 gate, H&E alone (P1 90%), sequencing everything (P3 90%), grade and grade + p53 as comparators. EXPLORATORY.
No rule is re-implemented: the module body of ex_e7_stress.py is executed up to its dispatch (it executes ex_e2_veto.py and rv9_falseflags.py: build, statics,
memory, cert_flags, make_ctx, the FAMS builders, check_parts, bars, apply_check, infold_bar, patient_arrays, lstats, full, delta), and the helper section of
ex_e3_score.py (mean-logit variant, exactly as ex_e9_frontier.py state('ml')). The only new engine piece is engine2 = ex_e7_stress.engine plus the chosen
candidate's held-out margin (score - fold threshold) for the landmark AUROCs. Every threshold / hyper-parameter / bar is chosen on training patients (inner
out-of-fold predictions of each outer fold); evaluation on held-out patients. The clinical comparators (Prague M, Parasa-style score) are scored with the same
engine (training-chosen threshold at the R49 detection target; in-fold imputation of missing values from training endoscopies only).
Row-level files only under killcoyne_mm/two_papers_P2S/ on the cluster; aggregates (no identifiers, no dates, patient cells < 5 shown as '<5') ->
results/paper_final/two_papers_P2S/<TASK>.json.   TASK = prep | rules | audit | landmark | design | comp | per1000      P2S_NB=k: smoke (k bootstrap draws)"""
import os, sys, json, time, glob, numpy as np, pandas as pd
from scipy.stats import beta as beta_dist
P = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, P); TASK = os.environ["TASK"]
os.environ["TASK"] = "__tp_p2s_e7__"; G7 = {"__file__": P + "/ex_e7_stress.py", "__name__": "e7_p2s"}
_s = open(P + "/ex_e7_stress.py").read(); exec(compile(_s[:_s.index('\nres = {"task": TASK}')], P + "/ex_e7_stress.py", "exec"), G7)
G2, G9 = G7["G2"], G7["G9"]; g8 = G2["g8"]
os.environ["TASK"] = "__tp_p2s_e3__"
_e = open(P + "/ex_e3_score.py").read(); exec(compile(_e[_e.index('E3 = M + "/explore_E3"'):_e.index('\n# ------------------------------------------------------------------ one variant')], P + "/ex_e3_score.py", "exec"), G9)
os.environ["TASK"] = TASK
TC = G7["TC"]; M = G7["M"]; T = G7["T"]; SM = G7["SM"]; lg = G7["lg"]; m5 = G7["m5"]; dstat = G7["dstat"]; ci2 = G7["ci2"]; patient_arrays = G7["patient_arrays"]; lstats = G7["lstats"]
r2 = G7["r2"]; r3 = G7["r3"]; DAY = np.timedelta64(1, "D")
OUT = T + "/results/paper_final/two_papers_P2S"; XD = M + "/two_papers_P2S"; NB = 2000; SMOKE = bool(os.environ.get("P2S_NB"))
if SMOKE: NB = int(os.environ["P2S_NB"]); OUT = XD + "/smoke_json"
for d in (OUT, XD): os.makedirs(d, exist_ok=True)
def dump(res, name=None):
    json.dump(res, open(f"{OUT}/{name or TASK}.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)); print("TPP2S DONE", name or TASK, flush=True)
pct = lambda a: [r3(np.nanpercentile(a, 2.5)), r3(np.nanpercentile(a, 97.5))] if np.isfinite(np.asarray(a, float)).any() else [None, None]
EXPECT = {"R49": (31.1, 31.8), "P1": (31.1, 36.0), "grade": (17.0, 25.0), "A1": (30.7, 23.6), "A12": (30.5, 21.5), "final": (30.5, 17.2), "final_infold": (30.4, 16.4), "ml_mem": (31.0, 26.6)}
# ================================================================== states (stored L-IMG as E7; mean-logit as E9 state('ml'))
def state_st():
    S, FT, ST, INFO, V, VTR, TH, TV = G7["base_setup"](); S["D"] = S["D"][:NB]; return S, FT, ST, INFO, V, VTR, TH, TV
def state_ml():
    g = G9; Oi, Ii, Op, Ip, _ = g["variant"]("mlogit:stored"); sg, lg_ = g["sg"], g["lg_"]
    Oi = sg(lg_(Oi) + 1e-6 * lg_(Op)); Ii = {k: pd.Series(sg(lg_(Ii[k].values) + 1e-6 * lg_(Ip[k].reindex(Ii[k].index).values)), index=Ii[k].index) for k in Ii}
    O = dict(g["O0"]); I = dict(g["I0"]); O["L-IMG"] = Oi; O["L-LATE"] = (O["L-CNV"] + Op) / 2; I["L-IMG"] = Ii; I["L-LATE"] = {k: (I["L-CNV"][k] + Ip[k].reindex(I["L-CNV"][k].index)) / 2 for k in I["L-CNV"]}
    keep = g["load"]; g["load"] = lambda src: (O, I, g["FO"])
    try: return g["prep_state"]()
    finally: g["load"] = keep
# ================================================================== engine2 = ex_e7_stress.engine (criterion 'np') + held-out margin of the chosen candidate
def engine2(S, ST, INFO, builder, extra=None):
    mk = G9["make_ctx"]; pat_scores = G9["pat_scores"]; thr_for = G9["thr_for"]; E = S["E"]; elab = S["elab"]
    F = np.zeros((E, 10), bool); NS = np.zeros((E, 10), int); MG = np.full((E, 10), np.nan); TRF = {}; CH = []; labels = None
    for j in range(10):
        for k in range(1, 11):
            c = mk(S, ST, j, k, INFO)
            if extra: extra(c)
            te = c.te_e; cands = builder(c); labels = labels or [x["label"] for x in cands]; ts_ = []; fl = []; tt = []
            for x in cands:
                assert x["kind"] == "score"
                pm = pat_scores(c, x["u"]); t = thr_for(c, pm); flag = x["u"] >= t
                det = c.patient_det(flag); inc = (S["INC"] @ flag.astype(float)) > 0; npf = int((inc & ~c.isP & c.trP).sum()); nfe = int((flag & c.tr_e & (elab == 0)).sum()); nsm = float(x["ns"][c.tr_e].mean())
                ts_.append((det, npf, nfe, nsm)); fl.append(flag); tt.append(t)
            ts_ = np.array(ts_); feas = ts_[:, 0] >= c.D; nC = len(cands)
            pick = (sorted([q for q in range(nC) if feas[q]], key=lambda q: (ts_[q, 1], ts_[q, 2], ts_[q, 3], q)) if feas.any() else sorted(range(nC), key=lambda q: (-ts_[q, 0], ts_[q, 1], ts_[q, 2], q)))[0]
            CH.append(int(pick)); F[te, j] = fl[pick][te]; NS[te, j] = cands[pick]["ns"][te]; MG[te, j] = cands[pick]["u"][te] - tt[pick]; TRF[(j, k)] = fl[pick]
    return dict(F=F, NS=NS, MG=MG, TRF=TRF, chosen=CH, labels=labels)
def fixed_path(S):
    """pathology comparators per endoscopy (as ex_e9_frontier.path_scores / ex_e7 'grade'): grade IND/LGD at the endoscopy; grade IND/LGD or p53 aberrant (p53 missing = normal)"""
    GRv = SM.Pathology.map({"NDBE": 0, "ID": 1, "LGD": 2, "HGD": 3, "IMC": 4}).values.astype(float); P53v = pd.to_numeric(SM["P53 IHC"], errors="coerce").values; hasp = np.isfinite(P53v)
    es = S["es"]; g = np.array([GRv[s].max() for s in es]); ab = np.array([(np.where(hasp, P53v, 0)[s] == 1).any() for s in es]); return {"grade": g >= 1, "g53": np.maximum(g, 2.0 * ab) >= 1, "lgd": g >= 2}
# ================================================================== clinical table (row-level, cluster only)
def prep_clin(S):
    G7r = {"__file__": P + "/rv7_round7.py", "__name__": "rv7_p2s"}; os.environ["TASK"] = "__tp_p2s_rv7__"
    _r = open(P + "/rv7_round7.py").read(); exec(compile(_r[:_r.index('\nres = {"task": TASK}')], P + "/rv7_round7.py", "exec"), G7r); os.environ["TASK"] = TASK
    L = G7r["db_link"](); DBX = G7r["DBX"]; SWG = G7r["SWG"]; KD = G7r["KD"]; up = S["up"]; es = S["es"]; E = S["E"]; pidx = S["pidx"]; ED = pd.to_datetime(S["ED"]); info = {}
    REF = ED + pd.to_timedelta(S["TTE"] * 365.25, unit="D")   # reference date per endoscopy row (patient event / last sequenced endoscopy)
    # demographics (patient level) via study number, as ex_e1_main.py
    dm = pd.read_csv(SWG + "/Demographics_full.csv", dtype=str); dmi = {}
    for _, r_ in dm.iterrows():
        for c_ in ("Study Number", "Alternate Study Number"):
            v_ = r_[c_].strip() if isinstance(r_[c_], str) else ""
            if v_ and v_ not in dmi: dmi[v_] = r_
    stud = SM.groupby("Patient").study_number.first().astype(str)
    num = lambda v: pd.to_numeric(pd.Series([v]).astype(str).str.strip().replace({"": np.nan, "nan": np.nan}), errors="coerce").iloc[0]
    demM = np.array([num(dmi[stud[p]]["Maximal"]) if stud.get(p) in dmi else np.nan for p in up], float); demC = np.array([num(dmi[stud[p]]["Circumference "]) if stud.get(p) in dmi else np.nan for p in up], float)
    smk_raw = [str(dmi[stud[p]]["Smoking Status"]).strip().upper() if stud.get(p) in dmi and isinstance(dmi[stud[p]]["Smoking Status"], str) else "" for p in up]
    smk = np.array([1.0 if s_.startswith("Y") else (0.0 if s_.startswith("N") else np.nan) for s_ in smk_raw])
    info["patients_linked_to_demographics"] = int(sum(stud.get(p) in dmi for p in up)); info["smoking_known_patients"] = int(np.isfinite(smk).sum()); info["prague_M_demographics_patients"] = int(np.isfinite(demM).sum())
    # Barrett's database endoscopies (Prague per endoscopy) and last record date
    en = pd.concat([pd.read_parquet(DBX + f"/{f}", columns=["participant_id", "endoscopydate", "barretts_circumference", "barretts_maximum"]) for f in ("endoscopy.parquet", "mat_endoscopy_full.parquet")])
    en["pid"] = en.participant_id.astype(str).str.replace(r"\.0$", "", regex=True); en["d"] = pd.to_datetime(en.endoscopydate, dayfirst=True, errors="coerce"); en["Mx"] = pd.to_numeric(en.barretts_maximum, errors="coerce"); en["Cx"] = pd.to_numeric(en.barretts_circumference, errors="coerce")
    en = en.dropna(subset=["d"]); en.loc[(en.Mx < 0) | (en.Mx > 25), "Mx"] = np.nan; en.loc[(en.Cx < 0) | (en.Cx > 25), "Cx"] = np.nan
    pt = L["pt"]; link = L["link"]; Me = np.full(E, np.nan); Ce = np.full(E, np.nan); msrc = np.zeros(E, int); last = np.full(len(up), np.nan)   # msrc 1 = DB at the endoscopy (+-60 d), 2 = demographics, 3 = DB patient median, 0 = missing
    for i, p in enumerate(up):
        pid = link.get(p); e_ = en[en.pid == pid] if pid is not None else en.iloc[:0]; ds = []
        if pid is not None:
            ds = list(pt[pt.pid == pid].d.dropna()) + list(e_.d)
        if ds: last[i] = (max(ds) - REF[np.where(pidx == i)[0][0]]) / DAY / 365.25
        for e in np.where(pidx == i)[0]:
            cand = e_[(e_.d - ED[e]).abs() <= pd.Timedelta(days=60)].dropna(subset=["Mx"])
            if len(cand): r_ = cand.iloc[np.argmin((cand.d - ED[e]).abs().values)]; Me[e] = r_.Mx; Ce[e] = r_.Cx; msrc[e] = 1
            elif np.isfinite(demM[i]): Me[e] = demM[i]; Ce[e] = demC[i]; msrc[e] = 2
            elif e_.Mx.notna().any(): Me[e] = float(e_.Mx.median()); Ce[e] = float(e_.Cx.median()) if e_.Cx.notna().any() else np.nan; msrc[e] = 3
    info["prague_M_source_endoscopies"] = {"db_at_endoscopy_60d": int((msrc == 1).sum()), "demographics": int((msrc == 2).sum()), "db_patient_median": int((msrc == 3).sum()), "missing": int((msrc == 0).sum()), "of": int(E)}
    info["prague_M_missing_patients"] = int(sum(not np.isfinite(Me[pidx == i]).any() for i in range(len(up))))
    info["db_last_record_after_reference_years"] = {"non_progressor_median": r2(np.nanmedian(last[S["plab"] == 0])), "non_progressor_linked": int(np.isfinite(last[S["plab"] == 0]).sum())}
    # event pathology and focal annotation at the event endoscopy (sheet; hz_prep.py / rv8 dates() event logic)
    fd = g8["sheet"](); fd = fd[fd.combined_name.notna()].copy(); fdu = fd.drop_duplicates("combined_name").set_index("combined_name")
    A = pd.read_csv(KD + "/kr_samples.csv", dtype=str).set_index("Sample"); A["Patient"] = A.Patient.astype(str).str.replace(r"\.0$", "", regex=True)
    A["mbf"] = pd.to_numeric(A.index.map(fdu["Months before final"]), errors="coerce"); A["ogd"] = A.index.map(fdu["Path_class_per_OGD"]).astype(str)
    A["note"] = (A.index.map(fdu["notes on label review"]).fillna("").astype(str) + " | " + A.index.map(fdu["Pathology notes"]).fillna("").astype(str)).str.lower()
    tev = SM.groupby("Patient").tev.first(); fb = SM.groupby("Patient").fallback_endpoint.first().astype(str).str.lower().isin(["true", "1"])
    ev = np.full(len(up), "", object); focal = np.zeros(len(up), bool)
    for i, p in enumerate(up):
        if S["plab"][i] != 1: continue
        if fb.get(p, False): ev[i] = "none"; continue
        rows = A[(A.Patient == p) & ((A.mbf - float(tev[p])).abs() < 1e-9)]; fr = fd[(fd.Patient.astype(str).str.replace(r"\.0$", "", regex=True) == p) & ((pd.to_numeric(fd["Months before final"], errors="coerce") - float(tev[p])).abs() < 1e-9)]
        grades = pd.concat([rows.Pathology.astype(str), rows.ogd.astype(str), fr.Pathology.astype(str), fr.Path_class_per_OGD.astype(str)]).str.upper()
        notes = pd.concat([rows.note, (fr["notes on label review"].fillna("").astype(str) + " | " + fr["Pathology notes"].fillna("").astype(str)).str.lower()])
        ev[i] = "IMC" if grades.str.contains("IMC").any() else ("HGD" if grades.str.contains("HGD").any() else "none"); focal[i] = bool(notes.str.contains("focal").any())   # HGD/IMC OGD class counted as IMC (conservative)
    info["event_type_pre_event_progressors"] = {k: int((ev == k).sum()) for k in ("HGD", "IMC", "none")}; info["focal_hgd_noted"] = int((focal & (ev == "HGD")).sum()); info["expected_data_md_Q8"] = {"HGD": 21, "IMC": 7, "none": 4, "focal_of_HGD": 7}
    GRv = SM.Pathology.map({"NDBE": 0, "ID": 1, "LGD": 2, "HGD": 3, "IMC": 4}).values.astype(float); lgd = np.array([(GRv[s] == 2).any() for s in es])
    age_first = np.array([SM.age.values[S["es"][S["ORD"][i][0]]].mean() for i in range(len(up))]); sexM = np.array([SM.sex_M.values[S["es"][S["ORD"][i][0]]].mean() for i in range(len(up))])
    np.savez(XD + "/clin.npz", Me=Me, Ce=Ce, msrc=msrc, smk=smk, ev=ev.astype(str), focal=focal, last_after_ref=last, lgd=lgd, age_first=age_first, sexM=sexM)
    return info
def load_clin(): z = np.load(XD + "/clin.npz", allow_pickle=False); return {k: z[k] for k in z.files}
# ------------------------------------------------------------------ clinical score builders (in-fold imputation from training endoscopies only)
def clin_builder(kind, mf, CL, S):
    sexM_e = CL["sexM"][S["pidx"]]; smk_e = CL["smk"][S["pidx"]]; lgd = CL["lgd"].astype(float)
    def b(c):
        tr = c.tr_e; Me = CL["Me"].copy(); med = np.nanmedian(Me[tr]); Me = np.where(np.isfinite(Me), Me, med)
        if kind == "prague": s = Me
        else: ps = np.nanmean(smk_e[tr]); s = 9.0 * sexM_e + 5.0 * np.where(np.isfinite(smk_e), smk_e, ps) + Me + 11.0 * lgd   # Parasa 2018 points: male 9, smoking 5, 1 per cm, LGD 11
        z = np.zeros(len(s), int); sc = G9["sc"]
        if mf == "none": return [sc("none", {}, s, z)]
        return [sc(f"ew h={h}y", {"h": h}, G9["memory"](c.ST, s, "ew", h), z) for h in (1, 2, 4)]
    return b
def clin_fixed(CL, S, ST):
    sexM_e = CL["sexM"][S["pidx"]]; smk_e = np.nan_to_num(CL["smk"][S["pidx"]], nan=0.0); Me = np.nan_to_num(CL["Me"], nan=0.0); par = 9.0 * sexM_e + 5.0 * smk_e + Me + 11.0 * CL["lgd"]
    o = {"prague_M_ge3": Me >= 3, "parasa_gt10": par > 10, "parasa_gt20": par > 20}
    mem = G9["memory"](ST, par, "ew", 2); memM = G9["memory"](ST, Me, "ew", 2); o["prague_M_ge3_ew2"] = memM >= 3; o["parasa_gt10_ew2"] = mem > 10; o["parasa_gt20_ew2"] = mem > 20
    return o
# ================================================================== rules: every tier and comparator, row-level arrays
def build_rules():
    t0 = time.time(); S, FT, ST, INFO, V, VTR, TH, TV = state_st(); GB = S["GB"]; R = {}; MG = {}; info = {}
    for nm, fam in (("gate_single", "A1:gate_al:none"), ("A1", "A1:gate_al:ew"), ("A12", "A12")):
        e = engine2(S, ST, INFO, G9["FAMS"][fam]); R[nm] = (e["F"], e["NS"]); MG[nm] = e["MG"]; info[nm] = {"chosen": {e["labels"][q]: int(np.sum(np.array(e["chosen"]) == q)) for q in set(e["chosen"])}}
        if nm == "A12": TRF12 = e["TRF"]
    FM, NM = R["A12"]; R["final"] = G7["apply_check"](S, FM, NM, GB, V, TH[0.90]); THi, pk = G7["infold_bar"](S, TRF12, VTR, TV); R["final_infold"] = G7["apply_check"](S, FM, NM, GB, V, THi); info["infold_bar_picks"] = pk
    Z = np.zeros((S["E"], 10), int); PA = fixed_path(S); rep = lambda a: np.repeat(a[:, None], 10, 1)
    R["R49"] = (S["GC"], S["NS49"]); R["P1"] = (S["P1F"], Z); R["grade"] = (rep(PA["grade"]), Z); R["g53"] = (rep(PA["g53"]), Z)
    TH3, _ = g8["thr_arr"](S, "late", 0.90); R["P3"] = (S["OUTS"]["late"] >= TH3, np.repeat(S["nsz"][:, None], 10, 1))
    Sm, STm, INFOm = state_ml(); assert (Sm["plab"] == S["plab"]).all() and Sm["E"] == S["E"] and all((a == b).all() for a, b in zip(Sm["es"], S["es"]))
    for nm, fam in (("ml_single", "A1:img:none"), ("ml_mem", "A1:img:ew")):
        e = engine2(Sm, STm, INFOm, G9["FAMS"][fam]); R[nm] = (e["F"], e["NS"]); MG[nm] = e["MG"]; info[nm] = {"chosen": {e["labels"][q]: int(np.sum(np.array(e["chosen"]) == q)) for q in set(e["chosen"])}}
    if os.path.exists(XD + "/clin.npz"):
        CL = load_clin()
        for kind in ("prague", "parasa"):
            for mf in ("none", "ew"):
                e = engine2(S, ST, INFO, clin_builder(kind, mf, CL, S)); nm = f"{kind}_{'single' if mf == 'none' else 'mem'}"; R[nm] = (e["F"], e["NS"]); MG[nm] = e["MG"]
                info[nm] = {"chosen": {e["labels"][q]: int(np.sum(np.array(e["chosen"]) == q)) for q in set(e["chosen"])}}
        for nm, f in clin_fixed(CL, S, ST).items(): R[nm] = (rep(f), Z)
    np.savez_compressed(XD + "/rules.npz", **{f"{k}__F": v[0] for k, v in R.items()}, **{f"{k}__NS": v[1].astype(np.int16) for k, v in R.items()}, **{f"{k}__MG": v for k, v in MG.items()})
    json.dump(info, open(XD + "/rules_info.json", "w"), default=str)
    # reproduction against the stored E7 / E9 flags
    rep_ = {}; z7 = np.load(M + "/explore_E7/base_flags.npz")
    for k in ("R49", "P1", "A1", "A12", "final", "final_infold", "grade"): rep_[f"{k}_flag_mismatches_vs_E7_base"] = int((R[k][0] != z7[f"{k}__F"].astype(bool)).sum()) if f"{k}__F" in z7.files else "absent"
    try:
        z9 = np.load(M + "/explore_E9/sweep_ml_core.npz"); kk = [x for x in z9.files if "img" in x and "ew" in x and x.endswith("F")]
        if kk: rep_["ml_mem_flag_mismatches_vs_E9_sweep_target_D"] = int((R["ml_mem"][0] != z9[kk[0]][0].astype(bool)).sum()); rep_["E9_key"] = kk[0]
        else: rep_["E9_keys"] = z9.files[:20]
    except Exception as ex: rep_["E9_error"] = str(ex)[:200]
    obs = {k: G7["pt"](S, *R[k]) for k in R}; rep_["observed"] = {k: [obs[k]["caught"], obs[k]["np_flagged"], obs[k]["seq_per_patient"]] for k in obs}
    rep_["expected"] = EXPECT; rep_["reproduced"] = bool(all(abs(obs[k]["caught"] - e[0]) < 0.051 and abs(obs[k]["np_flagged"] - e[1]) < 0.051 for k, e in EXPECT.items()))
    rep_["seconds"] = round(time.time() - t0); rep_["selection"] = info; return rep_
def load_rules():
    z = np.load(XD + "/rules.npz"); R = {}; MG = {}
    for k in z.files:
        nm, part = k.rsplit("__", 1)
        if part == "F": R[nm] = (z[k].astype(bool), z[nm + "__NS"].astype(int))
        if part == "MG": MG[nm] = z[k].astype(float)
    return R, MG
NAMES = {"ml_mem": "No sequencing: mean-logit + memory", "ml_single": "Mean-logit, single endoscopy", "final": "Confirmatory: final rule (cn90 bar)", "final_infold": "Confirmatory: final rule, fold-internal bar",
         "A12": "Memory + certificate (gate)", "A1": "Memory (gate score)", "gate_single": "Gate score, single endoscopy (re-tuned)", "R49": "Per-endoscopy gate R49 (this study)", "P1": "H&E alone (P1 90%)",
         "P3": "Sequence everything (max L-LATE, 90%)", "grade": "Grade IND/LGD", "g53": "Grade IND/LGD or p53 aberrant", "prague_single": "Prague M, single endoscopy (training threshold)", "prague_mem": "Prague M + memory (training threshold)",
         "parasa_single": "Parasa-style score, single endoscopy (training threshold)", "parasa_mem": "Parasa-style score + memory (training threshold)", "prague_M_ge3": "Prague M >= 3 cm (ever)", "prague_M_ge3_ew2": "EW-mean Prague M >= 3 cm",
         "parasa_gt10": "Parasa > 10 (intermediate/high, ever)", "parasa_gt20": "Parasa > 20 (high, ever)", "parasa_gt10_ew2": "EW-mean Parasa > 10", "parasa_gt20_ew2": "EW-mean Parasa > 20"}
MAIN = ["ml_mem", "final", "final_infold"]; COMPS = ["R49", "P1", "P3", "grade", "g53"]; PAIRS = {"ml_mem": "ml_single", "final": "R49", "final_infold": "R49", "A1": "gate_single", "A12": "gate_single"}
# ================================================================== helpers
def ord_pos(S):
    """per endoscopy: patient index, position in time order, previous endoscopy (or -1)"""
    prev = np.full(S["E"], -1)
    for o in S["ORD"]: prev[o[1:]] = o[:-1]
    return prev
def counts_dist(S, A, cls):
    """integer patient counts per repeat (flagged) -> distribution over the 10 repeats; majority-vote count (flagged in >= 5 of 10 repeats)"""
    m = (S["plab"] == cls) & A["pinc"]; per = A["FL"][m].sum(0).astype(int); mv = int((A["FL"][m].sum(1) >= 5).sum())
    return {"per_repeat": [int(x) for x in per], "min": int(per.min()), "median": float(np.median(per)), "max": int(per.max()), "majority_vote": mv, "n": int(m.sum())}
def cp(k, n):
    lo = 0.0 if k == 0 else beta_dist.ppf(0.025, k, n - k + 1); hi = 1.0 if k == n else beta_dist.ppf(0.975, k + 1, n - k); return [r3(lo), r3(hi)]
def tier_block(S, A, D, nP=None, nN=None):
    one = np.ones((1, S["P_"])); ob = lstats(S, A, one); bs = lstats(S, A, D); pi = A["pinc"]; nP = nP or int(((S["plab"] == 1) & pi).sum()); nN = nN or int(((S["plab"] == 0) & pi).sum())
    return {"caught": {"value": r2(nP * ob["caught_share"][0]), "ci95": ci2(nP * bs["caught_share"]), "of": nP}, "np_flagged": {"value": r2(nN * ob["np_flagged_share"][0]), "ci95": ci2(nN * bs["np_flagged_share"]), "of": nN},
            "missed": {"value": r2(nP * (1 - ob["caught_share"][0])), "ci95": ci2(nP * (1 - bs["caught_share"]))}, "np_never_flagged": {"value": r2(nN * (1 - ob["np_flagged_share"][0])), "ci95": ci2(nN * (1 - bs["np_flagged_share"]))},
            "seq_per_patient": r3(ob["seq_per_patient"][0]), "flags_per_np_py": {"value": r3(ob["flags_per_np_py"][0]), "ci95": ci2(bs["flags_per_np_py"])}}
res = {"task": TASK, "smoke_draws": NB if SMOKE else None}
# ================================================================== tasks
if TASK == "prep":
    S = state_st()[0]; res["prep"] = prep_clin(S); print(json.dumps(res["prep"], indent=1)); dump(res)
elif TASK == "rules":
    res["reproduction"] = build_rules(); print(json.dumps({k: v for k, v in res["reproduction"].items() if k != "selection"}, indent=1)); dump(res)
# ------------------------------------------------------------------ item 1: memory uses visits <= k only (code audit + tests)
elif TASK == "audit":
    S, FT, ST, INFO, V, VTR, TH, TV = state_st(); GB = S["GB"]; R, MG = load_rules(); out = {}; rng = np.random.RandomState(2026)
    ED = pd.to_datetime(S["ED"]); same = 0; nondec = 0
    for o in S["ORD"]:
        d = np.diff(np.asarray(ED[o].values, "datetime64[D]").astype(np.int64)); nondec += int((d < 0).sum()); same += int((d == 0).sum())
    out["time_order"] = {"patients_with_order_not_by_date": nondec, "consecutive_endoscopy_pairs_on_the_same_date": same, "endoscopies_with_samples_on_more_than_one_date": S["dinfo"].get("endoscopies_with_more_than_one_sample_date")}
    # (a) unit test of the memory functions and the certificate: future scores replaced by noise, past memory unchanged
    cut = np.array([rng.randint(len(o)) for o in S["ORD"]]); pos = S["POS"]; fut = pos > cut[S["pidx"]]; s0 = rng.randn(S["E"]); s1 = np.where(fut, rng.randn(S["E"]) * 10, s0); q0 = rng.rand(S["E"]) < 0.4; q1 = np.where(fut, ~q0, q0)
    ut = {}
    for kind, par in (("mean", None), ("ew", 0.5), ("ew", 1), ("ew", 2), ("ew", 4), ("ew", 16), ("last2", None), ("cons2", None), ("k2any", None), ("k2of3", None)):
        u0 = G9["memory"](ST, s0, kind, par); u1 = G9["memory"](ST, s1, kind, par); ut[f"{kind}{'' if par is None else ' h=' + str(par)}"] = int((~np.isclose(u0[~fut], u1[~fut], rtol=0, atol=1e-12)).sum())
    for Y in (2, 4): ut[f"certificate Y={Y}"] = int((G9["cert_flags"](ST, q0, Y)[~fut] != G9["cert_flags"](ST, q1, Y)[~fut]).sum())
    # the current endoscopy must enter its own memory (sanity: changing only position cut changes the memory there)
    s2 = s0.copy(); atc = pos == cut[S["pidx"]]; s2[atc] += 5.0; ut["current_endoscopy_enters_own_memory_ew2"] = bool(np.all(G9["memory"](ST, s2, "ew", 2)[atc] > G9["memory"](ST, s0, "ew", 2)[atc]))
    out["unit_test_mismatches_at_or_before_cut"] = ut
    # (b) end-to-end: held-out (outer) predictions of every sample in a later endoscopy replaced by noise; the full nested pipeline rerun; flags up to the cut compared
    def perturb(Sx):
        S2 = dict(Sx); O2 = {m: Sx["O"][m].copy() for m in Sx["O"]}; samp = np.concatenate([Sx["es"][e] for e in np.where(fut)[0]])
        for m in ("L-IMG", "L-CNV"): O2[m][samp] = np.clip(rng.rand(len(samp), 10), 1e-4, 1 - 1e-4)
        O2["L-LATE"] = (O2["L-CNV"] + O2["L-IMG"]) / 2; S2["O"] = O2; return S2
    e2e = {}; S2 = perturb(S)
    for nm, fam in (("A1", "A1:gate_al:ew"), ("A12", "A12")):
        e = engine2(S2, ST, INFO, G9["FAMS"][fam]); e2e[nm] = int((e["F"][~fut] != R[nm][0][~fut]).sum())
        if nm == "A12":
            Fp, Np = G7["apply_check"](S, e["F"], e["NS"], GB, V, TH[0.90]); e2e["final"] = int((Fp[~fut] != R["final"][0][~fut]).sum())
            THi, _ = G7["infold_bar"](S, e["TRF"], VTR, TV); Fi, _ = G7["apply_check"](S, e["F"], e["NS"], GB, V, THi); e2e["final_infold"] = int((Fi[~fut] != R["final_infold"][0][~fut]).sum())
            e2e["final_flags_changed_after_cut_(sanity, expected > 0)"] = int((Fp[fut] != R["final"][0][fut]).sum())
    Sm, STm, INFOm = state_ml(); Sm2 = perturb(Sm); e = engine2(Sm2, STm, INFOm, G9["FAMS"]["A1:img:ew"]); e2e["ml_mem"] = int((e["F"][~fut] != R["ml_mem"][0][~fut]).sum())
    e2e["ml_mem_flags_changed_after_cut_(sanity, expected > 0)"] = int((e["F"][fut] != R["ml_mem"][0][fut]).sum())
    out["end_to_end_flag_mismatches_at_or_before_cut"] = e2e
    # (c) other held-out patients: the whole history of a random half of patients replaced by noise; the other half's flags unchanged
    half = rng.rand(S["P_"]) < 0.5; fut_save = fut.copy(); fut = half[S["pidx"]]; S3 = perturb(S); e = engine2(S3, ST, INFO, G9["FAMS"]["A12"]); Fp, _ = G7["apply_check"](S, e["F"], e["NS"], GB, V, TH[0.90])
    out["other_patients_test_final_flag_mismatches_in_untouched_half"] = int((Fp[~fut] != R["final"][0][~fut]).sum()); fut = fut_save
    out["code_audit"] = ["rv9 memory(): EW weights ST['EW'][h] are lower-triangular in the patient's time order (np.tril, column <= row) and masked to valid slots; 'mean' is a cumulative sum; so the score at slot k uses slots <= k",
                         "time order ORD (rv8 setup): exact endoscopy date (pre_event_cohort.Date), ties broken by months-before-final; weights use max(dt, 0) only for the diagonal",
                         "rv9 cert_flags(): strictly earlier endoscopies (np.tril(..., -1)) within Y years",
                         "quiet-genome check (ex_e2 vs_test / ex_e7 apply_check): the current endoscopy's own top-2 samples only; a cancelled flag is not fed back into the memory",
                         "thresholds, half-life, certificate and check bar: training patients' inner out-of-fold predictions only (make_ctx / thr_for / infold_bar); held-out patients' outer predictions enter only their own memory",
                         "patient-level outcomes use the maximum over a patient's endoscopies (ever flagged); this is an evaluation summary, not an input to any score"]
    out["verdict"] = "pass" if all(v == 0 for k, v in ut.items() if isinstance(v, int) and not isinstance(v, bool)) and ut["current_endoscopy_enters_own_memory_ew2"] and all(v == 0 for k, v in e2e.items() if "sanity" not in k) and out["other_patients_test_final_flag_mismatches_in_untouched_half"] == 0 else "FAIL"
    res["audit"] = out; print(json.dumps(out, indent=1)); dump(res)
# ------------------------------------------------------------------ item 2: landmark analysis (+ equal exposure, patient level and landmark level)
elif TASK == "landmark":
    S, FT, ST, INFO, V, VTR, TH, TV = state_st(); D = S["D"]; R, MG = load_rules(); CL = load_clin(); isP = S["plab"] == 1; pidx = S["pidx"]; TTE = S["TTE"]; E = S["E"]; isPe = isP[pidx]
    fu_np = TTE + np.nan_to_num(np.maximum(CL["last_after_ref"][pidx], 0), nan=0.0)   # NP known event-free follow-up after the landmark (to the reference date, extended by later database records)
    HZ = {"1y": 1.0, "3y": 3.0, "5y": 5.0, "ever": np.inf}
    def outcome(h):
        pos = isPe & (TTE <= h); neg = (~isPe & (fu_np >= h if np.isfinite(h) else True)) | (isPe & (TTE > h)); return pos, neg
    Wd = np.vstack([np.ones((1, S["P_"])), D])[:, pidx]   # landmark weights = patient multiplicity (row 0 observed)
    def bin_metrics(F, pos, neg, w):
        """per weight row: sensitivity, specificity, PPV, NPV at landmark level (mean over repeats)"""
        Ff = F.astype(float); wp = w * pos; wn = w * neg
        tp = wp @ Ff; fn = wp.sum(1)[:, None] - tp; fp = wn @ Ff; tn = wn.sum(1)[:, None] - fp
        with np.errstate(invalid="ignore", divide="ignore"): return {"sens": np.nanmean(tp / (tp + fn), 1), "spec": np.nanmean(tn / (tn + fp), 1), "ppv": np.nanmean(tp / (tp + fp), 1), "npv": np.nanmean(tn / (tn + fn), 1)}
    def auc_rows(X, pos, neg, w):
        ip = np.where(pos)[0]; ineg = np.where(neg)[0]; out = np.zeros((w.shape[0], 10))
        for j in range(10):
            xp = X[ip, j][:, None]; xn = X[ineg, j][None, :]; C = (xp > xn) + 0.5 * (xp == xn); wp = w[:, ip]; wn = w[:, ineg]
            out[:, j] = np.einsum("bi,ij,bj->b", wp, C, wn) / (wp.sum(1) * wn.sum(1))
        return out.mean(1)
    def summ(o, d): return {"value": r3(o), "ci95": pct(d)}
    def landmark_block(w, label):
        blk = {}
        for hn, h in HZ.items():
            pos, neg = outcome(h); b = {"positive_landmarks": m5(int(pos.sum())), "negative_landmarks": m5(int(neg.sum())), "positive_patients": m5(int(len(set(pidx[pos])))), "censored_np_landmarks": m5(int((~isPe & ~neg & ~pos).sum()))}
            ww = w if label != "patient_weighted" else w / np.maximum(np.bincount(pidx, minlength=S["P_"])[pidx], 1)
            Mt = {k: bin_metrics(R[k][0], pos, neg, ww) for k in MAIN + list(PAIRS.values()) + ["A1", "A12"] + COMPS + ["prague_mem", "parasa_mem"] if k in R}
            b["binary"] = {k: {m: summ(v[m][0], v[m][1:]) for m in ("sens", "spec", "ppv", "npv")} for k, v in Mt.items()}
            b["memory_minus_single"] = {}
            for a, s_ in PAIRS.items():
                if a in Mt and s_ in Mt: b["memory_minus_single"][f"{a} - {s_}"] = {m: dstat(Mt[a][m][0] - Mt[s_][m][0], Mt[a][m][1:] - Mt[s_][m][1:]) for m in ("sens", "spec")}
            if int(pos.sum()) >= 5:
                Au = {k: auc_rows(MG[k], pos, neg, ww) for k in ("ml_mem", "ml_single", "A1", "gate_single", "A12") if k in MG}
                b["auroc_margin"] = {k: summ(v[0], v[1:]) for k, v in Au.items()}
                b["auroc_memory_minus_single"] = {f"{a} - {s_}": dstat(Au[a][0] - Au[s_][0], Au[a][1:] - Au[s_][1:]) for a, s_ in (("ml_mem", "ml_single"), ("A1", "gate_single"), ("A12", "gate_single"))}
            blk[hn] = b
        return blk
    out = {"all_landmarks": landmark_block(Wd, "landmark"), "patient_weighted": landmark_block(Wd, "patient_weighted")}
    # equal exposure at landmark level: NP landmarks kept only within the first m endoscopies, m ~ progressors' endoscopy counts (E7 test 1 draws, RandomState(7001), 200 draws)
    rng = np.random.RandomState(7001); NE = S["NE"].astype(int); NEP = NE[isP]; POS = S["POS"]; KEEP = np.zeros(E)
    masks = []
    for d in range(200):
        m = np.where(isP, NE, np.minimum(NE, rng.choice(NEP, S["P_"]))); kp = POS < m[pidx]; KEEP += kp; masks.append(kp)
    KEEP /= 200; out["equal_exposure_landmarks"] = landmark_block(Wd * KEEP[None, :], "landmark"); out["equal_exposure_landmarks"]["note"] = "landmark weight = share of the 200 E7 truncation draws in which the NP endoscopy is kept"
    # equal exposure at patient level (E7 test 1(a), frozen rules) for the tiers, comparators and single-endoscopy counterparts
    PYs = []; acc = {k: [] for k in R}
    for kp in masks:
        Se = dict(S, PY=G7["trunc_py"](S, ST, kp)); PYs.append(Se["PY"])
        for k in R: acc[k].append(patient_arrays(Se, R[k][0], R[k][1], kp))
    Se = dict(S, PY=np.mean(PYs, 0)); AA = {k: G7["avgA"](L) for k, L in acc.items()}; ee = {}
    for k in R: ee[k] = {"values": G7["full"](Se, AA[k], D), "vs_R49": G7["delta"](Se, AA[k], AA["R49"], D), "vs_g53": G7["delta"](Se, AA[k], AA["g53"], D)}
    for a, s_ in PAIRS.items():
        if a in AA and s_ in AA: ee[f"{a} - {s_} (memory gain)"] = G7["delta"](Se, AA[a], AA[s_], D)
    ee["check_E7_A1_vs_R49_np_delta_expected_-6.1"] = ee["A1"]["vs_R49"]["np_flagged"]["delta"]
    out["equal_exposure_patients"] = ee
    out["definitions"] = {"landmark": "every pre-event endoscopy (288), scored with the patient's history up to and including it (memory is causal: audit task)",
                          "positive at horizon h": "progressor endoscopy with years to the event <= h (ever: every progressor endoscopy)",
                          "negative at horizon h": "non-progressor endoscopy with known event-free follow-up >= h (to the last sequenced endoscopy, extended by later Barrett's database records), or progressor endoscopy with years to the event > h",
                          "weights": "patient-bootstrap multiplicities (2,000 draws, RandomState(0)); patient_weighted = each patient's landmarks share weight 1",
                          "auroc_margin": "landmark AUROC of the held-out score minus its fold threshold (pooled over folds), repeat mean; memory vs the same base without memory, each with the round-9 training threshold rule"}
    res["landmark"] = out; dump(res)
# ------------------------------------------------------------------ items 3, 4, 5, 7: two-lows rule, safety block, lead time, subgroups
elif TASK == "design":
    S, FT, ST, INFO, V, VTR, TH, TV = state_st(); D = S["D"]; R, MG = load_rules(); CL = load_clin(); isP = S["plab"] == 1; pidx = S["pidx"]; TTE = S["TTE"]; E = S["E"]; prev = ord_pos(S)
    RULES = [k for k in MAIN + ["A1", "A12", "ml_single", "gate_single"] + COMPS + ["prague_mem", "parasa_mem", "prague_single", "parasa_single"] if k in R]; A = {k: patient_arrays(S, *R[k]) for k in RULES}; out = {}
    out["tiers"] = {k: {"name": NAMES[k], **tier_block(S, A[k], D), "integer_counts": {"progressors_caught": counts_dist(S, A[k], 1), "np_flagged": counts_dist(S, A[k], 0)}} for k in RULES}
    for k in RULES:
        o = out["tiers"][k]; mvP = o["integer_counts"]["progressors_caught"]["majority_vote"]; mvN = o["integer_counts"]["np_flagged"]["majority_vote"]
        o["majority_vote"] = {"missed": m5(32 - mvP), "missed_cp95_of_32": cp(32 - mvP, 32), "np_never_flagged": 43 - mvN, "np_never_flagged_cp95_share": cp(43 - mvN, 43)}
        if k != "R49": o["vs_R49"] = G7["delta"](S, A[k], A["R49"], D)
        if k != "g53": o["vs_g53"] = G7["delta"](S, A[k], A["g53"], D)
    # ---- NPV among cleared patients (cleared = never flagged), cohort (case-control) and per repeat integers
    one = np.ones((1, S["P_"])); npv = {}
    for k in RULES:
        Ak = A[k]; Mb = np.vstack([one, D]); sP = Mb * isP; sN = Mb * ~isP; cl = 1 - Ak["FL"]; mP = sP @ cl; cN = sN @ cl
        with np.errstate(invalid="ignore", divide="ignore"): v = np.nanmean(cN / (cN + mP), 1)
        pr = [(int(((~isP) & (Ak["FL"][:, j] == 0)).sum()), int((isP & (Ak["FL"][:, j] == 0)).sum())) for j in range(10)]
        npv[k] = {"npv_cohort_case_control": {"value": r3(v[0]), "ci95": pct(v[1:])}, "per_repeat_cleared_np_and_missed_p": [list(x) for x in pr]}
    out["npv_cleared"] = npv
    # ---- item 3: two-consecutive-lows rule (de-escalate only after >= 2 consecutive low endoscopies)
    TWO = {}
    for k in RULES:
        F = R[k][0]; low = ~F; low_prev = np.where(prev[:, None] >= 0, low[np.maximum(prev, 0)], False); deesc1 = low; deesc2 = low & low_prev
        lastE = np.array([o[-1] for o in S["ORD"]]); blk = {}; nfd = np.zeros_like(F)
        for o in S["ORD"]: nfd[o] = ~np.maximum.accumulate(F[o], axis=0)   # never flagged up to and including this endoscopy
        for nm, DE in (("single_look", deesc1), ("two_consecutive_lows", deesc2), ("never_flagged_to_date_single_look", deesc1 & nfd), ("never_flagged_to_date_two_consecutive_lows", deesc2 & nfd)):
            ever = (S["INC"] @ DE.astype(float)) > 0; atlast = DE[lastE]; neverfl = A[k]["FL"] == 0; Mb = np.vstack([one, D])
            share_np_endo = (S["INC"][~isP] @ DE.astype(float)).sum(0) / S["NE"][~isP].sum()
            def cnt(X, cls):
                msk = isP if cls else ~isP; n = int(msk.sum()); Wm = Mb * msk; v = (Wm @ X.astype(float)).mean(1) / Wm.sum(1) * n; return {"value": r2(v[0]), "ci95": ci2(v[1:])}
            blk[nm] = {"np_ever_deescalated": cnt(ever, 0), "np_deescalated_at_last_endoscopy": cnt(atlast, 0), "np_never_flagged_and_deescalated": cnt(ever & neverfl, 0),
                       "share_of_np_endoscopies_followed_by_a_longer_interval": r3(share_np_endo.mean()),
                       "progressors_ever_deescalated_before_event": cnt(ever, 1), "progressors_deescalated_at_last_pre_event_endoscopy": cnt(atlast, 1), "missed_progressors_deescalated": cnt(ever & neverfl, 1),
                       "progressors_deescalated_at_last_integer_per_repeat": [int((isP & atlast[:, j]).sum()) for j in range(10)]}
        dd = {}
        for key in ("np_ever_deescalated", "np_deescalated_at_last_endoscopy", "np_never_flagged_and_deescalated", "progressors_deescalated_at_last_pre_event_endoscopy", "progressors_ever_deescalated_before_event"):
            dd[key] = r2(blk["two_consecutive_lows"][key]["value"] - blk["single_look"][key]["value"])
        blk["two_lows_minus_single_look"] = dd; dd2 = {}
        for key in dd: dd2[key] = r2(blk["never_flagged_to_date_two_consecutive_lows"][key]["value"] - blk["never_flagged_to_date_single_look"][key]["value"])
        blk["two_lows_minus_single_look_never_flagged_to_date"] = dd2; TWO[k] = blk
    out["two_lows"] = TWO
    # ---- item 4: safety block (missed = never flagged; 'cleared progressor' = not flagged at the last pre-event endoscopy)
    lastE = np.array([o[-1] for o in S["ORD"]]); ev = CL["ev"]; foc = CL["focal"]; Mlast = CL["Me"][lastE]; tlast = TTE[lastE]; ne1 = S["NE"] == 1
    SAFE = {}
    for k in RULES:
        FL = A[k]["FL"]; miss = (FL == 0) & isP[:, None]; clr = (~R[k][0][lastE]) & isP[:, None]
        def desc(X):
            per = X.sum(0); pr = X.mean(1); anyp = X.any(1); o = {"per_repeat_integer_counts": [int(x) for x in per], "mean": r2(per.mean()), "patients_ever_in_any_repeat": m5(int(anyp.sum())), "patients_in_ge5_of_10_repeats": m5(int((pr >= 0.5).sum())),
                                                                   "progressor_repeats_total": int(X.sum())}
            W = pr; tot = W.sum()
            def ec(m):   # expected count (mean over repeats) in a category; suppressed when fewer than 5 distinct patients contribute
                k_ = int(((W > 0) & m).sum()); return 0 if k_ == 0 else (r2((W * m).sum()) if k_ >= 5 else "<5 patients")
            if tot > 0:
                o["expected_count_by_event"] = {e_: ec(ev == e_) for e_ in ("HGD", "IMC", "none")}
                o["expected_count_focal_hgd"] = ec(foc & (ev == "HGD"))
                o["expected_count_by_years_last_cleared_to_event"] = {"<1": ec(tlast < 1), "1-3": ec((tlast >= 1) & (tlast < 3)), "3-5": ec((tlast >= 3) & (tlast < 5)), ">=5": ec(tlast >= 5)}
                o["expected_count_5y_interval_postpones_detection_past_event"] = ec(tlast < 5); o["expected_count_3y_interval_postpones_past_event"] = ec(tlast < 3)
                o["expected_count_by_prague_M"] = {"<3 cm": ec((np.nan_to_num(Mlast, nan=-1) >= 0) & (Mlast < 3)), ">=3 cm": ec(Mlast >= 3), "unknown": ec(~np.isfinite(Mlast))}
                o["expected_count_single_pre_event_endoscopy"] = ec(ne1)
                o["most_expected_misses_have_one_pre_event_endoscopy"] = bool((W * ne1).sum() / tot >= 0.5); o["most_expected_misses_postponed_past_event_by_5y_interval"] = bool((W * (tlast < 5)).sum() / tot >= 0.5)
                o["any_IMC"] = bool(((W > 0) & (ev == "IMC")).any()); o["all_postponed_past_event_by_5y_interval"] = bool(np.all(tlast[W > 0] < 5))
                o["years_last_cleared_to_event_weighted_median"] = r2(float(np.median(np.repeat(tlast[W > 0], np.round(W[W > 0] * 10).astype(int))))) if (W > 0).sum() >= 5 else "<5 patients"
            return o
        SAFE[k] = {"missed_never_flagged": desc(miss), "cleared_at_last_pre_event_endoscopy": desc(clr)}
    out["safety"] = SAFE; out["safety_reference"] = {"progressors": int(isP.sum()), "event_type": {e_: m5(int(((ev == e_) & isP).sum())) for e_ in ("HGD", "IMC", "none")}, "focal_hgd": m5(int((foc & (ev == "HGD")).sum())),
                                                     "progressors_with_one_pre_event_endoscopy": m5(int((ne1 & isP).sum())), "years_last_pre_event_endoscopy_to_event_median_all_progressors": r2(float(np.median(tlast[isP])))}
    # ---- item 5: lead time (> 2 y and > 4 y before the event; NP all endoscopies, or symmetric: > 2 / > 4 y before the last sequenced endoscopy)
    LT = {}
    for yr in (2, 4):
        for sym in (False, True):
            keep = (TTE > yr) | (~isP[pidx] if not sym else np.zeros(E, bool)); AA = {k: patient_arrays(S, R[k][0], R[k][1], keep) for k in RULES}; nPe = int((AA["R49"]["pinc"] & isP).sum()); nNe = int((AA["R49"]["pinc"] & ~isP).sum())
            blk = {"progressors_with_an_endoscopy_in_window": m5(nPe), "non_progressors_with_an_endoscopy_in_window": m5(nNe)}
            for k in RULES:
                blk[k] = {"values": tier_block(S, AA[k], D), "vs_R49": G7["delta"](S, AA[k], AA["R49"], D), "vs_g53": G7["delta"](S, AA[k], AA["g53"], D)}
                if k in PAIRS: blk[k]["vs_single_endoscopy_counterpart"] = G7["delta"](S, AA[k], AA[PAIRS[k]], D)
            LT[f"gt_{yr}y_before_event" + ("_symmetric" if sym else "")] = blk
    out["lead_time"] = LT
    # ---- item 7: subgroups
    Mp = np.array([np.nanmedian(CL["Me"][pidx == i]) if np.isfinite(CL["Me"][pidx == i]).any() else np.nan for i in range(S["P_"])]); age = CL["age_first"]; sexM = CL["sexM"] >= 0.5
    spe = np.array([S["nsz"][pidx == i].mean() for i in range(S["P_"])])
    SG = {"segment <3 cm": Mp < 3, "segment >=3 cm": Mp >= 3, "segment unknown": ~np.isfinite(Mp), "age <65 at first endoscopy": age < 65, "age >=65": age >= 65, "male": sexM, "female": ~sexM,
          "mean samples per endoscopy <2": spe < 2, "mean samples per endoscopy >=2": spe >= 2}
    SUB = {}
    for nm, pm in SG.items():
        nPs = int((pm & isP).sum()); nNs = int((pm & ~isP).sum()); b = {"progressors": m5(nPs), "non_progressors": m5(nNs)}
        if nPs >= 5 and nNs >= 5:
            Dm = D * pm[None, :]; Om = pm[None, :].astype(float)
            for k in RULES:
                Ak = A[k]; ob = lstats(S, Ak, Om); bs = lstats(S, Ak, Dm); ob0 = lstats(S, A["R49"], Om); bs0 = lstats(S, A["R49"], Dm); ob1 = lstats(S, A["g53"], Om); bs1 = lstats(S, A["g53"], Dm)
                b[k] = {"caught": {"value": r2(nPs * ob["caught_share"][0]), "ci95": ci2(nPs * bs["caught_share"])}, "np_flagged": {"value": r2(nNs * ob["np_flagged_share"][0]), "ci95": ci2(nNs * bs["np_flagged_share"])},
                        "vs_R49_np": dstat(nNs * (ob["np_flagged_share"][0] - ob0["np_flagged_share"][0]), nNs * (bs["np_flagged_share"] - bs0["np_flagged_share"])), "vs_R49_caught": dstat(nPs * (ob["caught_share"][0] - ob0["caught_share"][0]), nPs * (bs["caught_share"] - bs0["caught_share"])),
                        "vs_g53_np": dstat(nNs * (ob["np_flagged_share"][0] - ob1["np_flagged_share"][0]), nNs * (bs["np_flagged_share"] - bs1["np_flagged_share"]))}
        SUB[nm] = b
    for nm, em in (("endoscopies with 1 sample", S["nsz"] == 1), ("endoscopies with >=2 samples", S["nsz"] >= 2)):
        AA = {k: patient_arrays(S, R[k][0], R[k][1], em) for k in RULES}; b = {"endoscopies": int(em.sum()), "progressors_with_one": m5(int((AA["R49"]["pinc"] & isP).sum())), "np_with_one": m5(int((AA["R49"]["pinc"] & ~isP).sum()))}
        for k in RULES: b[k] = {"values": tier_block(S, AA[k], D), "vs_R49": G7["delta"](S, AA[k], AA["R49"], D)}
        SUB[nm] = b
    out["subgroups"] = SUB; out["subgroup_note"] = "patient-level subgroups: frozen held-out calls, bootstrap restricted to the subgroup's patients; segment = median Prague M over the patient's endoscopies (database at the endoscopy, else Demographics_full); endoscopy-level rows restrict the evaluated endoscopies (memory still from the full history)"
    res["design"] = out; dump(res)
# ------------------------------------------------------------------ item 6: clinical comparators (Prague M, Parasa-style score), with memory
elif TASK == "comp":
    S, FT, ST, INFO, V, VTR, TH, TV = state_st(); D = S["D"]; R, MG = load_rules(); CL = load_clin(); isP = S["plab"] == 1; pidx = S["pidx"]
    info = json.load(open(OUT.replace("/smoke_json", "") + "/prep.json"))["prep"] if os.path.exists(OUT.replace("/smoke_json", "") + "/prep.json") else None
    keys = [k for k in ("prague_single", "prague_mem", "parasa_single", "parasa_mem", "prague_M_ge3", "prague_M_ge3_ew2", "parasa_gt10", "parasa_gt20", "parasa_gt10_ew2", "parasa_gt20_ew2") if k in R]
    A = {k: patient_arrays(S, *R[k]) for k in keys + MAIN + ["R49", "g53", "grade", "P1"]}; out = {"missingness": info}
    smk = CL["smk"]; out["smoking_known"] = {"progressors": m5(int(np.isfinite(smk[isP]).sum())), "non_progressors": m5(int(np.isfinite(smk[~isP]).sum())), "of": [int(isP.sum()), int((~isP).sum())]}
    sel = json.load(open(XD + "/rules_info.json"))
    for k in keys:
        out[k] = {"name": NAMES[k], **tier_block(S, A[k], D), "vs_R49": G7["delta"](S, A[k], A["R49"], D), "vs_g53": G7["delta"](S, A[k], A["g53"], D), "vs_no_sequencing_tier": G7["delta"](S, A[k], A["ml_mem"], D),
                  "selection": sel.get(k)}
    for a, s_ in (("prague_mem", "prague_single"), ("parasa_mem", "parasa_single"), ("prague_M_ge3_ew2", "prague_M_ge3"), ("parasa_gt10_ew2", "parasa_gt10"), ("parasa_gt20_ew2", "parasa_gt20")):
        if a in A and s_ in A: out[f"{a} - {s_} (memory gain)"] = G7["delta"](S, A[a], A[s_], D)
    for k in MAIN: out[k] = {"name": NAMES[k], **tier_block(S, A[k], D)}
    # evaluation-matched clinical scores (OPTIMISTIC for the comparator: its threshold is set on the evaluated patients, per repeat)
    def env(MGk, tc, tn):
        a, b = [], []
        for j in range(10):
            pm = np.array([MGk[S["pidx"] == i, j].max() for i in range(S["P_"])]); th = np.unique(pm); cau = np.array([(isP & (pm >= t)).sum() for t in th]); npf = np.array([(~isP & (pm >= t)).sum() for t in th])
            ok = cau >= np.floor(tc + 1e-9); a.append(npf[ok].min() if ok.any() else np.nan); ok2 = npf <= tn + 1e-9; b.append(cau[ok2].max() if ok2.any() else 0)
        return r2(np.nanmean(a)), r2(np.mean(b))
    mt = {}
    for k in ("prague_single", "prague_mem", "parasa_single", "parasa_mem"):
        if k in MG:
            for t in MAIN:
                tc = out[t]["caught"]["value"]; tn = out[t]["np_flagged"]["value"]; npf, cau = env(MG[k], tc, tn)
                mt[f"{k} at {t}"] = {"tier_caught": tc, "tier_np_flagged": tn, "comparator_np_flagged_at_caught_ge_floor_of_tier": npf, "comparator_caught_at_np_flagged_le_tier": cau}
    out["evaluation_matched_optimistic"] = mt
    # matched: NP flagged by each tier minus the clinical score at the tier's own caught count (evaluation-matched, optimistic for the comparator) via the margin dial
    out["note"] = ("Parasa-style score per endoscopy = 9 x male + 5 x smoker + Prague M (cm) + 11 x LGD at the endoscopy (Parasa et al., Gastroenterology 2018); missing Prague M = training-endoscopy median, missing smoking = training share of smokers x 5 "
                   "(training-chosen rows) or 0 (fixed cut-off rows); 'training threshold' rows use the round-9 rule (highest threshold reaching the R49 gate's own training progressor-patient detection; half-life 1/2/4 y by fewest training NP flagged)")
    res["comp"] = out; dump(res)
# ------------------------------------------------------------------ item 9: per 1,000 patient-years in the de-escalated (cleared) group, model-based
elif TASK == "per1000":
    S, FT, ST, INFO, V, VTR, TH, TV = state_st(); D = S["D"]; R, MG = load_rules(); isP = S["plab"] == 1
    RULES = [k for k in MAIN + ["A1", "A12", "ml_single"] + COMPS + ["prague_mem", "parasa_mem", "parasa_gt10"] if k in R]; one = np.ones((1, S["P_"])); out = {}
    for k in RULES:
        Ak = patient_arrays(S, *R[k]); ob = lstats(S, Ak, one); bs = lstats(S, Ak, D); sens = np.r_[ob["caught_share"], bs["caught_share"]]; spec = 1 - np.r_[ob["np_flagged_share"], bs["np_flagged_share"]]
        mvP = int((Ak["FL"][isP].sum(1) >= 5).sum()); miss_cp = cp(32 - mvP, 32); spec_mv = 1 - (Ak["FL"][~isP].sum(1) >= 5).mean(); o = {"name": NAMES[k]}
        for lam in (0.003, 0.0045, 0.006):
            rate = 1000 * lam * (1 - sens) / (lam * (1 - sens) + (1 - lam) * spec); flagged_rate = 1000 * lam * sens / (lam * sens + (1 - lam) * (1 - spec)); cleared = lam * (1 - sens) + (1 - lam) * spec
            ub = 1000 * lam * miss_cp[1] / (lam * miss_cp[1] + (1 - lam) * spec_mv); kr = int(np.ceil(32 * (1 - sens[0]) - 1e-9)); cr = cp(kr, 32)[1]; ub2 = 1000 * lam * cr / (lam * cr + (1 - lam) * spec[0])
            o[f"incidence_{lam * 100:.2f}pct_per_year"] = {"progressions_per_1000_py_in_deescalated": {"value": r3(rate[0]), "ci95": pct(rate[1:])}, "upper_95_from_exact_binomial_on_majority_vote_misses": r3(ub), "upper_95_from_exact_binomial_on_mean_misses_rounded_up": r3(ub2), "misses_used_rounded_up": kr,
                                                         "progressions_per_1000_py_in_flagged": {"value": r2(flagged_rate[0]), "ci95": pct(flagged_rate[1:])}, "share_deescalated_per_1000_patients": {"value": r2(1000 * cleared[0]), "ci95": pct(1000 * cleared[1:])},
                                                         "relative_rate_vs_no_triage": {"value": r3(rate[0] / (1000 * lam)), "ci95": pct(rate[1:] / (1000 * lam))}, "meets_0.2pct_per_year_bar_upper95": bool(np.nanpercentile(rate[1:], 97.5) <= 2.0)}
        out[k] = o
    out["model"] = ("MODEL-BASED (case-control cohort; no absolute risk is estimated directly): annual progression incidence lambda among NDBE patients; the rule's patient-level sensitivity (progressors caught at >= 1 pre-event endoscopy) "
                    "and specificity (non-progressors never flagged) are assumed to transport. Progressions per 1,000 patient-years in the de-escalated group = 1000 x lambda (1 - sens) / (lambda (1 - sens) + (1 - lambda) spec). "
                    "CIs: patient bootstrap (2,000 draws, RandomState(0)) of sens and spec; the bootstrap cannot exceed the cohort's observed misses, so the exact Clopper-Pearson upper limit on the majority-vote miss count is given alongside. "
                    "Guideline bar (gastro review): <= 2 per 1,000 patient-years (short-segment NDBE, about 0.2%/yr).")
    res["per1000"] = out; dump(res)
else:
    raise SystemExit(f"unknown TASK {TASK}")

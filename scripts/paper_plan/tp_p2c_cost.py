"""Two papers, agent P2C (docs/two_papers_P2C.md): early cost-consequence analysis for Paper 2 (NICE early value assessment style, CHEERS 2022).
Runs locally on aggregates only: results/paper_final/two_papers_P2C/rate_draws.json (paired patient-bootstrap draws of class-specific per-patient rates,
from tp_p2c_draws.py on the cluster). Writes results/paper_final/two_papers_P2C/cost_consequence.json and figures in paper/two_papers/figs_P2C/.

Model (per 1,000 non-dysplastic patients on expert read, 10-year horizon, NHS England perspective, 2023/24 GBP, 3.5% discounting of costs and QALYs):
- a share pi progresses (HGD/OAC) within the horizon (1.5%, 3%, 5%, 10%); class-specific held-out rates of the cohort are reweighted to pi;
- guideline surveillance every I_g years (base 3; endoscopies at 0, 3, 6, 9); a de-escalated patient moves one step to I_x (base 5; endoscopies at 0, 5);
- de-escalated = never flagged by the strategy's pre-check AI rule nor overridden by the pathologist: base override = LGD or p53 aberrant ever
  (the plan's eligibility: non-dysplastic on expert read, LGD/HGD overrides); scenario = any IND/LGD or p53 aberrant;
  a flag cancelled by the quiet-genome check keeps the guideline interval (so the confirmatory tier de-escalates exactly as the light tier);
- every AI flag on an endoscopy not already flagged by grade/p53 triggers the band action: expert pathology review with p53, specialist outpatient review and
  an expert endoscopy (premium over a routine list) at the guideline interval; grade/p53 flags trigger current care (expert review + 6-month repeat endoscopy);
- de-escalated progressors carry an excess probability of late-stage diagnosis (two-step HGD -> early OAC -> late OAC chain, HTW EAR069 / Swart 2021 rates,
  onset uniform within the interval), valued at the cost difference late vs early plus lambda x QALY loss.
"""
import json, os, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
T = "/Users/zuberi01/Documents/thesis"; OUT = os.environ.get("P2C_OUT", T + "/results/paper_final/two_papers_P2C"); FIG = T + "/paper/two_papers/figs_P2C"
os.makedirs(OUT, exist_ok=True); os.makedirs(FIG, exist_ok=True)
RD = json.load(open(os.environ.get("P2C_RATES", OUT + "/rate_draws.json"))); R = {k: {q: np.array(v, float) for q, v in d.items()} for k, d in RD["rules"].items()}; NB = len(R["g53"]["sens"]) - 1
rng = np.random.default_rng(20261021)
# ------------------------------------------------------------------ parameters: (base, PSA sampler)
def gam(m, se): k = (m / se) ** 2; return lambda n: rng.gamma(k, m / k, n)
def lnr(m, lo, hi): s = (np.log(hi) - np.log(lo)) / (2 * 1.96); return lambda n: m * np.exp(rng.normal(0, s, n) - s * s / 2)
def bet(m, se): a = m * (m * (1 - m) / se ** 2 - 1); b = (1 - m) * (m * (1 - m) / se ** 2 - 1); return lambda n: rng.beta(a, b, n)
def unif(lo, hi): return lambda n: rng.uniform(lo, hi, n)
P = {  # name: (base, sampler, source)
    "c_endo": (745.0, gam(745.0, 74.5), "FE21Z diagnostic upper GI endoscopy with biopsy, day case, National Cost Collection 2023/24 (NHS England 2024; as used in HTW EAR069, Oct 2025)"),
    "c_op": (181.0, gam(181.0, 18.1), "WF01A consultant-led follow-up attendance, upper GI service, NCC 2023/24 (HTW EAR069)"),
    "c_emr": (1219.31, gam(1219.31, 121.93), "FE02A-C major therapeutic endoscopy, day case, NCC 2023/24 (HTW EAR069)"),
    "c_rfa": (1492.51, gam(1492.51, 149.25), "FE01Z complex therapeutic endoscopy, day case, NCC 2023/24 (HTW EAR069)"),
    "c_oes": (6362.86, gam(6362.86, 636.29), "FF02A-C oesophagectomy, elective, NCC 2023/24 (HTW EAR069 table A10)"),
    "c_chemo": (3965.78, gam(3965.78, 396.58), "chemotherapy 1 year, Swart 2021 / NICE NG83, inflated to 2023/24 (HTW EAR069)"),
    "c_palrfa": (4505.53, gam(4505.53, 450.55), "palliative RFA 1 year (HTW EAR069)"),
    "c_stent": (4807.78, gam(4807.78, 480.78), "FE10A-D luminal stent, NCC 2023/24 (HTW EAR069)"),
    "c_pall": (8759.82, gam(8759.82, 875.98), "palliative care, final year of life (HTW EAR069)"),
    "c_review": (100.0, lnr(100.0, 44.0, 200.0), "expert (second) pathology review; assumption (range 44-200; 44 = round-7d NHS histopathology proxy)"),
    "c_p53": (25.0, lnr(25.0, 15.0, 50.0), "p53 immunohistochemistry per block; assumption"),
    "prem_expert": (0.25, unif(0.0, 0.5), "expert-centre endoscopy premium as a share of FE21Z (longer slot, high-definition, referral); assumption"),
    "c_slide": (5.0, lnr(5.0, 1.0, 15.0), "scan + storage + QA per slide; Ardon 2023 J Pathol Inform (USD 0.55-19.53 per scan, 0.79-0.92 staff per slide), ~1-1.2 GB/slide storage"),
    "c_ai": (10.0, lnr(10.0, 2.0, 30.0), "AI licence and sign-off per endoscopy scored; assumption"),
    "c_backscan": (10.0, lnr(10.0, 5.0, 20.0), "retrieve + scan one archived slide (memory needs the prior endoscopy); assumption"),
    "seq_fail": (0.05, unif(0.02, 0.15), "sWGS failure -> repeat assay; assumption (FFPE sWGS library success ~98% in one series)"),
    "slides_per_endo": (2.05, None, "cohort: 9.54 samples per non-progressor over 4.65 endoscopies"),
    "p_hgd_oac": (0.119, bet(0.119, 0.012), "annual HGD -> early OAC (Swart 2021 via HTW EAR069)"),
    "p_early_late": (0.80, bet(0.80, 0.08), "annual early -> late OAC (Swart 2021 expert opinion, via HTW EAR069)"),
    "p_surg": (0.50, bet(0.50, 0.05), "late OAC suitable for surgery (Sami 2021, HTW EAR069)"),
    "p_death_postsurg": (0.24, bet(0.24, 0.024), "annual death after oesophagectomy (5-y survival 25%, Ovrebo 2012, HTW EAR069)"),
    "u_ndbe": (0.91, bet(0.91, 0.05), "NDBE / treated utility (Benaglia 2013, Sami 2021)"),
    "u_late": (0.675, bet(0.675, 0.032), "late OAC utility (Benaglia 2013, Sami 2021)"),
    "u_surg_y1": (0.731, None, "late OAC, year of surgery (HTW EAR069)"),
    "u_postsurg": (0.836, bet(0.836, 0.016), "after surgery (Benaglia 2013, Sami 2021)"),
    "age_mult": (0.80, unif(0.75, 0.85), "general-population utility multiplier around age 70-80 (Hernandez Alava 2022 NICE DSU)"),
    "le_70": (15.5, unif(13.0, 18.0), "remaining life expectancy at about age 70, ONS national life tables (approx.)"),
    "lam": (20000.0, None, "NICE threshold GBP per QALY (lower end)"),
}
DISC = 0.035; H = 10
def base(): return {k: v[0] for k, v in P.items()}
def draw(n):
    o = {}
    for k, (b, s, _) in P.items(): o[k] = s(n) if s is not None else np.full(n, b)
    return o
def sched(I, H=H, r=DISC):
    t = np.arange(0, H, I); return len(t), float((1 / (1 + r) ** t).sum())
def plate(I, h1, h2):
    """P(late OAC at detection) for HGD onset uniform within an interval of I years; h1, h2 hazards (HGD->early, early->late)"""
    t = np.linspace(0, I, 401)[:, None]; h1 = np.atleast_1d(h1)[None]; h2 = np.atleast_1d(h2)[None]; S = (h2 * np.exp(-h1 * t) - h1 * np.exp(-h2 * t)) / (h2 - h1); v = 1 - S.mean(0); return v if v.size > 1 else float(v[0])
def harm(p, Ig, Ix):
    """per de-escalated progressor: (excess P(late), delta cost late - early, QALY loss late vs early-cured), arrays"""
    h1 = -np.log(1 - p["p_hgd_oac"]); h2 = -np.log(1 - p["p_early_late"]); dpl = plate(Ix, h1, h2) - plate(Ig, h1, h2)
    c_early = p["c_emr"] + 3.5 * p["c_rfa"] + (6 - 2 / Ig) * p["c_endo"]
    q = 1 - p["p_death_postsurg"]; ys = 1 / (1 - q / (1 + DISC))   # discounted expected years alive after surgery
    c_surg = p["c_oes"] + p["c_chemo"] + 2 * ys * p["c_op"] + 0.75 * p["c_pall"]
    c_palv = 0.25 * p["c_chemo"] + 0.75 * (p["c_palrfa"] + p["c_stent"]) + p["c_pall"]
    c_late = p["p_surg"] * c_surg + (1 - p["p_surg"]) * c_palv
    le = (1 - (1 + DISC) ** -p["le_70"]) / DISC
    q_early = le * p["u_ndbe"] * p["age_mult"]; q_pal = 0.5 * p["u_late"] * p["age_mult"]; q_sur = (p["u_surg_y1"] + (ys - 1) * p["u_postsurg"]) * p["age_mult"]
    q_late = p["p_surg"] * q_sur + (1 - p["p_surg"]) * q_pal
    return dpl, c_late - c_early, q_early - q_late, c_early, c_late
def flag_cost(p): return p["c_review"] + p["c_p53"] + p["c_op"] + p["prem_expert"] * p["c_endo"]
DPROG = (1 - (1 + DISC) ** -H) / (H * np.log(1 + DISC))   # mean discount factor for an event uniform over the horizon
# ------------------------------------------------------------------ strategies
STRATS = {  # name: (label, flag rule, de-escalation rule or None, sequencing rule, sequencing scales with guideline endoscopies)
    "CC": ("Current care (grade +/- p53, guideline intervals)", None, None, None),
    "T0": ("No-sequencing tier (mean-logit + memory)", "ml_mem", "ml_mem", "ml_mem"),
    "T1": ("Light tier (mean-logit gate + memory + certificate)", "ml_memcert", "ml_memcert", "ml_memcert"),
    "T2": ("Confirmatory tier (mean-logit full stack)", "ml_final", "ml_memcert", "ml_final"),
    "SE": ("Sequence everything (max L-LATE, every sample)", "seq_all", "seq_all", "seq_all"),
    "T1s": ("Light tier, stored-score variant (memory + certificate)", "memcert", "memcert", "memcert"),
    "T2s": ("Confirmatory tier, stored-score variant (final rule)", "final", "memcert", "final"),
    "SEm": ("Sequence everything with memory", "seq_all_mem", "seq_all_mem", "seq_all_mem"),
    "GD": ("Grade/p53-guided de-escalation, no AI (not guideline)", None, "GD", None),
}
MAIN = ["CC", "T0", "T1", "T2", "SE"]
def evaluate(pi, p, rows, Ig=3, Ix=5, backscan=True, flag_once=False, c_seq=250.0, deesc_delay=False, c_flag=None, override="l53", flag_benefit=0.0):
    """per 1,000 patients; p = parameter dict (scalars or arrays); rows = index into the rate draws (0 = observed); returns dict of arrays"""
    nG, dG = sched(Ig); nX, dX = sched(Ix)
    if deesc_delay:   # no archived slide: de-escalation only after a second prospective low endoscopy (0, Ig, then Ig + Ix)
        t = np.r_[0, Ig, np.arange(Ig + Ix, H, Ix)]; nX, dX = len(t), float((1 / (1 + DISC) ** t).sum())
    dpl, dcl, dql, _, _ = harm(p, Ig, Ix)
    nP, nN = 1000 * pi, 1000 * (1 - pi); g = R["g53"]
    c_g53 = p["c_review"] + p["c_endo"]   # current care work-up of IND/LGD/p53-aberrant: expert review + 6-month repeat endoscopy
    if c_flag is None: c_flag = flag_cost(p)
    out = {}
    for s, (lab, fr, dr, sr) in STRATS.items():
        o = {k: 0.0 for k in ("cost", "cost_noharm", "qaly", "endos", "endos_disc", "slides", "samples", "samples_disc", "np_flag_rule", "np_flag_any", "prog_flag_any", "prog_deesc", "np_deesc", "ai_flags", "ai_flags_disc", "late_excess", "c_seq_part", "c_flag_part")}
        for cls, n, sk, fk, nfk, nek, nsk in (("P", nP, "sens", "sens", "nfp", "nep", "nsp"), ("N", nN, "fpr", "fpr", "nfn", "nen", "nsn")):
            ne = g[nek][rows]
            if dr is None: d = 0.0
            elif dr == "GD": d = 1 - R[override][sk][rows]
            else: d = 1 - R[dr + "_or_" + override][sk][rows]
            e_n = d * nX + (1 - d) * nG; e_d = d * dX + (1 - d) * dG
            gfl = g[nfk][rows] / ne; cost = n * (p["c_endo"] * e_d + c_g53 * gfl * dG)
            o["endos"] = o["endos"] + n * e_n; o["endos_disc"] = o["endos_disc"] + n * e_d
            if fr is not None:
                slides = p["slides_per_endo"] * (e_n + (1 if backscan else 0)); slides_d = p["slides_per_endo"] * (e_d + (1 if backscan else 0))
                cost = cost + n * (p["c_slide"] * slides_d + p["c_ai"] * e_d + (p["c_backscan"] * p["slides_per_endo"] if backscan else 0))
                o["slides"] = o["slides"] + n * slides
                af = R[fr + "_not_g53"]; afl = R[fr + "_not_g53"][nfk][rows] / R[fr + "_not_g53"][nek][rows]
                if flag_once: nflag_n = R[fr + "_or_g53"][sk][rows] - g[sk][rows]; nflag_d = nflag_n   # once per patient newly flagged by the rule
                else: nflag_n = afl * nG; nflag_d = afl * dG
                o["ai_flags"] = o["ai_flags"] + n * nflag_n; o["ai_flags_disc"] = o["ai_flags_disc"] + n * nflag_d
                cf = n * c_flag * nflag_d; cost = cost + cf; o["c_flag_part"] = o["c_flag_part"] + cf
                sq = R[sr][nsk][rows] / R[sr][nek][rows]; sn = sq * (e_n if sr.startswith("seq_all") else nG); sd = sq * (e_d if sr.startswith("seq_all") else dG)
                o["samples"] = o["samples"] + n * sn; o["samples_disc"] = o["samples_disc"] + n * sd
                cs = n * c_seq * (1 + p["seq_fail"]) * sd; cost = cost + cs; o["c_seq_part"] = o["c_seq_part"] + cs
                fl_rule = R[fr][sk][rows]; fl_any = R[fr + "_or_g53"][sk][rows]
            else:
                fl_rule = g[sk][rows]; fl_any = g[sk][rows]
            o["cost_noharm"] = o["cost_noharm"] + cost
            if cls == "P":
                o["prog_flag_any"] = n * fl_any; o["prog_deesc"] = n * d; le = n * d * dpl
                if fr is not None and flag_benefit > 0:   # scenario: the expert work-up lowers late-stage risk in progressors flagged by AI but not by grade/p53
                    h1_ = -np.log(1 - p["p_hgd_oac"]); h2_ = -np.log(1 - p["p_early_late"]); le = le - n * (fl_any - g[sk][rows]) * flag_benefit * plate(Ig, h1_, h2_)
                o["late_excess"] = le
                cost = cost + le * dcl * DPROG; o["qaly"] = -le * dql * DPROG
            else:
                o["np_flag_rule"] = n * fl_rule; o["np_flag_any"] = n * fl_any; o["np_deesc"] = n * d
            o["cost"] = o["cost"] + cost
        out[s] = o
    for s in out:
        out[s]["d_cost"] = out[s]["cost"] - out["CC"]["cost"]; out[s]["d_qaly"] = out[s]["qaly"] - out["CC"]["qaly"]
        out[s]["nmb20"] = -out[s]["d_cost"] + p["lam"] * out[s]["d_qaly"]; out[s]["nmb30"] = -out[s]["d_cost"] + 30000 * out[s]["d_qaly"]
        out[s]["d_cost_noharm"] = out[s]["cost_noharm"] - out["CC"]["cost_noharm"]
    out["_harm"] = {"d_p_late": dpl, "d_cost_late": dcl, "qaly_loss_late": dql, "value_per_deesc_progressor": dpl * (dcl + p["lam"] * dql) * DPROG}
    return out
# ------------------------------------------------------------------ run: base case (observed rates, base costs) and PSA (paired bootstrap draws x parameter draws)
PIS = {"1.5%": 0.015, "3%": 0.03, "5%": 0.05, "10%": 0.10}; SEQP = (85.0, 250.0, 400.0)
pb = base(); pd_ = draw(NB); IDX = np.arange(1, NB + 1)
r1 = lambda x: None if x is None or not np.isfinite(x) else round(float(x), 1)
def summ(b, d): d = np.asarray(d, float); return {"value": r1(b), "ci95": [r1(np.nanpercentile(d, 2.5)), r1(np.nanpercentile(d, 97.5))]}
RES = {"note": ("P2C early cost-consequence model; per 1,000 non-dysplastic patients, 10-year horizon, 3.5% discount, NHS England 2023/24 GBP. value = observed held-out rates with "
                "base-case costs; ci95 = 2.5/97.5 percentiles over 2,000 PSA iterations, each pairing one patient-bootstrap draw of all rules' rates with one draw of every cost, "
                "transition and utility parameter. 'endos' and 'samples' undiscounted counts; costs and QALYs discounted."), "parameters": {k: {"base": v[0], "source": v[2]} for k, v in P.items()},
       "strategies": {k: v[0] for k, v in STRATS.items()}, "per_1000": {}}
KEYS = ("slides", "samples", "endos", "np_flag_rule", "np_flag_any", "np_deesc", "prog_flag_any", "prog_deesc", "late_excess", "ai_flags", "cost", "d_cost", "d_cost_noharm", "d_qaly", "nmb20")
for pn, pi in PIS.items():
    RES["per_1000"][pn] = {}
    for cs in SEQP:
        B = evaluate(pi, pb, 0, c_seq=cs); D = evaluate(pi, pd_, IDX, c_seq=cs); blk = {}
        for s in STRATS: blk[s] = {k: summ(B[s][k], D[s][k]) for k in KEYS}; blk[s]["p_saves"] = round(float((D[s]["d_cost"] < 0).mean()), 3); blk[s]["p_nmb20_pos"] = round(float((D[s]["nmb20"] > 0).mean()), 3)
        blk["_harm"] = {k: summ(B["_harm"][k], D["_harm"][k]) for k in B["_harm"]}
        RES["per_1000"][pn][f"seq_{int(cs)}"] = blk
# ------------------------------------------------------------------ break-even analyses (3% unless stated): solve the incremental cost (or NMB) for a single input
def breakeven(pi, p, rows, ref="CC", what="seq", nmb=False, **kw):
    """value of one input (GBP per sample, or GBP per AI flag) at which strategy minus ref is cost-neutral (nmb=False) or NMB-neutral at lambda (nmb=True);
    the model is linear in each input, so two evaluations give intercept and slope. Returns {strategy: (break-even value, slope)}"""
    def net(E, s): return (E[s]["cost"] - nmb * p["lam"] * E[s]["qaly"]) - (E[ref]["cost"] - nmb * p["lam"] * E[ref]["qaly"])
    key = "c_seq" if what == "seq" else "c_flag"; kw = dict(kw); kw.pop(key, None)
    E0 = evaluate(pi, p, rows, **{key: 0.0}, **kw); E1 = evaluate(pi, p, rows, **{key: 1000.0}, **kw); o = {}
    for s in STRATS:
        if s == ref: continue
        a0 = net(E0, s); sl = (net(E1, s) - a0) / 1000.0
        with np.errstate(divide="ignore", invalid="ignore"): o[s] = (np.where(np.abs(sl) > 1e-6, -a0 / sl, np.nan), sl)
    return o
def miss_threshold(pi, p, rows, ref="CC", **kw):
    """value V of one de-escalated progressor (cost + lambda x QALY, discounted) above which the strategy stops being preferred to ref (NMB = 0)"""
    E = evaluate(pi, p, rows, **kw); o = {}
    for s in STRATS:
        if s == ref: continue
        dc = E[s]["cost_noharm"] - E[ref]["cost_noharm"]; dm = E[s]["prog_deesc"] - E[ref]["prog_deesc"]
        with np.errstate(divide="ignore", invalid="ignore"): o[s] = np.where(dm > 1e-9, -dc / dm, np.where(dc < 0, np.inf, -np.inf)) if np.ndim(dm) else (-dc / dm if dm > 1e-9 else (np.inf if dc < 0 else -np.inf))
    return o, E["_harm"]["value_per_deesc_progressor"]
BE = {}
for pn, pi in PIS.items():
    BE[pn] = {}
    for ref in ("CC", "T0"):
        b_seq = breakeven(pi, pb, 0, ref, "seq"); d_seq = breakeven(pi, pd_, IDX, ref, "seq")
        b_sqn = breakeven(pi, pb, 0, ref, "seq", nmb=True); d_sqn = breakeven(pi, pd_, IDX, ref, "seq", nmb=True)
        b_fl = breakeven(pi, pb, 0, ref, "flag", c_seq=250.0); d_fl = breakeven(pi, pd_, IDX, ref, "flag", c_seq=250.0)
        b_m, vb = miss_threshold(pi, pb, 0, ref); d_m, vd = miss_threshold(pi, pd_, IDX, ref)
        b_f85 = breakeven(pi, pb, 0, ref, "flag", c_seq=85.0); d_f85 = breakeven(pi, pd_, IDX, ref, "flag", c_seq=85.0)
        b_f400 = breakeven(pi, pb, 0, ref, "flag", c_seq=400.0); d_f400 = breakeven(pi, pd_, IDX, ref, "flag", c_seq=400.0)
        b_so = breakeven(pi, pb, 0, ref, "seq", flag_once=True); d_so = breakeven(pi, pd_, IDX, ref, "seq", flag_once=True)
        blk = {}
        for s in b_seq:
            blk[s] = {"breakeven_seq_price_costonly": summ(b_seq[s][0], d_seq[s][0]), "seq_slope_gbp_per_gbp": r1(b_seq[s][1]),
                      "breakeven_seq_price_nmb20": summ(b_sqn[s][0], d_sqn[s][0]),
                      "breakeven_flag_cost_at_seq250": summ(b_fl[s][0], d_fl[s][0]), "flag_slope": r1(b_fl[s][1]),
                      "breakeven_flag_cost_at_seq85": summ(b_f85[s][0], d_f85[s][0]), "breakeven_flag_cost_at_seq400": summ(b_f400[s][0], d_f400[s][0]),
                      "breakeven_seq_price_flag_once_per_patient": summ(b_so[s][0], d_so[s][0]), "seq_slope_flag_once": r1(b_so[s][1]),
                      "miss_value_threshold_gbp": summ(b_m[s], d_m[s]) if np.isfinite(b_m[s]) else ("prefers at any miss value" if b_m[s] > 0 else "never preferred"),
                      "p_preferred_at_modelled_miss_value": round(float(np.mean(d_m[s] > vd)), 3)}
        BE[pn][f"vs_{ref}"] = blk
    BE[pn]["modelled_value_per_deesc_progressor"] = summ(vb, vd)
RES["break_even"] = BE
RES["break_even_note"] = ("breakeven_seq_price: GBP per sample at which the strategy's incremental cost vs the reference is 0 (cost only) or its NMB at GBP 20k/QALY is 0 (the strategy is preferred BELOW it "
                          "when seq_slope > 0; a negative value means it is cheaper at any price only if the slope is negative, i.e. it sequences less than the reference); "
                          "breakeven_flag_cost: GBP per AI flag at which incremental cost is 0 (preferred BELOW it when flag_slope > 0); miss_value_threshold: value of one de-escalated "
                          "progressor (discounted cost + GBP 20k x QALY) above which the strategy stops being preferred (NMB < 0) at GBP 250 per sample")
# ------------------------------------------------------------------ scenarios (3%, GBP 250 per sample): one-way changes, report d_cost and NMB for the main strategies
SCN = {"base": {}, "interval 2 -> 3 y": {"Ig": 2, "Ix": 3}, "interval 3 -> 6 y (two steps)": {"Ig": 3, "Ix": 6}, "interval 5 -> 10 y": {"Ig": 5, "Ix": 10},
       "no archive back-scan (de-escalate after 2nd prospective low)": {"backscan": False, "deesc_delay": True}, "flag costed once per patient": {"flag_once": True}, "override = any IND/LGD or p53 aberrant": {"override": "g53"},
       "sequencing GBP 85": {"c_seq": 85.0}, "AI flag work-up halves late-stage risk in AI-flagged progressors": {"flag_benefit": 0.5}, "sequencing GBP 400": {"c_seq": 400.0}}
PSC = {"AI GBP 30 per endoscopy, slide GBP 15": {"c_ai": 30.0, "c_slide": 15.0}, "flag = extra endoscopy (prem 1.0)": {"prem_expert": 1.0}, "flag = review + p53 only": {"prem_expert": 0.0, "c_op": 0.0},
       "endoscopy GBP 713 (FE21Z/FE22Z weighted)": {"c_endo": 713.0}, "endoscopy GBP 500": {"c_endo": 500.0}, "early->late 0.5/y": {"p_early_late": 0.5}}
SC = {}
for nm, kw in list(SCN.items()) + [(k, {"pp": v}) for k, v in PSC.items()]:
    kw = dict(kw); pp = kw.pop("pp", {}); cs = kw.pop("c_seq", 250.0); B = evaluate(0.03, dict(pb, **pp), 0, c_seq=cs, **kw); D = evaluate(0.03, dict(pd_, **{k: np.full(NB, v) for k, v in pp.items()}), IDX, c_seq=cs, **kw)
    SC[nm] = {s: {"d_cost": summ(B[s]["d_cost"], D[s]["d_cost"]), "nmb20": summ(B[s]["nmb20"], D[s]["nmb20"]), "prog_deesc": summ(B[s]["prog_deesc"], D[s]["prog_deesc"]), "endos": summ(B[s]["endos"], D[s]["endos"])} for s in MAIN + ["GD"]}
RES["scenarios_3pct_seq250"] = SC
# ------------------------------------------------------------------ budget impact: typical ICB and Cancer Alliance
BI = {"assumptions": {"england_population_mid2023": 57.7e6, "n_icb": 42, "n_cancer_alliances": 21,
                      "surveilled_per_million_base": 2500, "surveilled_per_million_range": [1500, 4000],
                      "derivation": ("UK primary-care incidence of diagnosed Barrett's 15.6-85.6 per 100,000 person-years at ages 40-74 (Masclee et al. 2014, about 22 per 100,000 per year "
                                     "across all ages) x about 11-12 years under surveillance = about 0.25% of the population; uptake 100% of eligible NDBE on expert read"),
                      "uptake_profile_years_1_5": [0.2, 0.4, 0.6, 0.8, 1.0], "annualisation": "discounted 10-year incremental cost per 1,000 / 10 x patients / 1,000 (approximate; undiscounted spend is about 15% higher)"}}
icb = 57.7e6 / 42; ca = 57.7e6 / 21
for unit, popn in (("ICB", icb), ("Cancer Alliance", ca)):
    n = popn / 1e6 * 2500; BI[unit] = {"population": round(popn), "patients_in_surveillance": round(n), "range": [round(popn / 1e6 * 1500), round(popn / 1e6 * 4000)]}
    for s in MAIN:
        # annual steady-state incremental spend (undiscounted 10-year per-1,000 cost / 10), 3% progression, GBP 85 / 250 / 400
        vals = {}
        for cs in SEQP:
            B = evaluate(0.03, dict(pb, lam=pb["lam"]), 0, c_seq=cs); D = evaluate(0.03, pd_, IDX, c_seq=cs)
            vals[f"seq_{int(cs)}"] = summ(B[s]["d_cost"] * n / 1000 / 10, D[s]["d_cost"] * n / 1000 / 10)
        BI[unit][s] = {"annual_incremental_spend_gbp_steady_state": vals}
    # 5-year rollout, base seq 250
    B = evaluate(0.03, pb, 0, c_seq=250.0)
    BI[unit]["five_year_rollout_seq250"] = {s: r1(sum(u * B[s]["d_cost"] * n / 1000 / 10 for u in BI["assumptions"]["uptake_profile_years_1_5"])) for s in MAIN}
    BI[unit]["setup_one_off_gbp"] = {"scanner_if_none": 100000, "integration_validation": 15000, "note": "scanner capital 'upwards of GBP 100,000' (vendor guidance); LIS / integration ~GBP 15,000 (unverified vendor figure); only for services without digital pathology"}
RES["budget_impact"] = BI
json.dump(RES, open(OUT + "/cost_consequence.json", "w"), indent=1)
print("written", OUT + "/cost_consequence.json")
# ------------------------------------------------------------------ figures (PNG 200 dpi, fonts >= 9 pt)
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.spines.top": False, "axes.spines.right": False})
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
DIV = LinearSegmentedColormap.from_list("div", ["#184f95", "#6da7ec", "#f0efec", "#ec8a8a", "#a32b2b"])
COL = {"CC": "#52514e", "T0": "#2a78d6", "T1": "#1baf7a", "T2": "#eb6834", "SE": "#e87ba4", "GD": "#898781"}
SHORT = {"CC": "Current care", "T0": "No sequencing", "T1": "Light", "T2": "Confirmatory", "SE": "Sequence everything", "GD": "Pathology-guided\nde-escalation (no AI)"}
FIGPI = 0.03
# 1. two-way: sequencing price x flag cost
sp = np.linspace(0, 600, 121); fc = np.linspace(0, 1500, 151); fig, axs = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
for ax, (s, ref, ttl) in zip(axs, (("T2", "CC", "A  Confirmatory tier minus current care"), ("T2", "T0", "B  Confirmatory tier minus no-sequencing tier"))):
    Zg = np.zeros((len(fc), len(sp)))
    E00 = evaluate(FIGPI, pb, 0, c_seq=0.0, c_flag=0.0); E10 = evaluate(FIGPI, pb, 0, c_seq=1000.0, c_flag=0.0); E01 = evaluate(FIGPI, pb, 0, c_seq=0.0, c_flag=1000.0)
    d = lambda E: E[s]["cost"] - E[ref]["cost"]; a0 = d(E00); ks = (d(E10) - a0) / 1000; kf = (d(E01) - a0) / 1000
    Zg = (a0 + ks * sp[None, :] + kf * fc[:, None]) / 1000
    lim = np.abs(Zg).max(); im = ax.pcolormesh(sp, fc, Zg, cmap=DIV, norm=TwoSlopeNorm(0, -lim, lim), shading="auto", rasterized=True)
    cs = ax.contour(sp, fc, Zg, levels=[0], colors="#0b0b0b", linewidths=2); ax.clabel(cs, fmt={0: "cost-neutral"}, fontsize=9)
    ax.axvspan(85, 400, color="none", ec="#52514e", ls="--", lw=1); ax.text(242, 1450, "accredited sWGS range\n£85–£400", ha="center", va="top", fontsize=9, color="#0b0b0b")
    fb = flag_cost(pb); ax.plot(250, fb, "o", ms=8, mfc="white", mec="#0b0b0b", mew=2); ax.annotate(f"base case\n£250, £{fb:.0f}", (250, fb), (300, fb + 200), fontsize=9, arrowprops=dict(arrowstyle="-", color="#52514e"))
    ax.set_title(ttl, loc="left", fontsize=11); ax.set_xlabel("sequencing price per sample (£)"); ax.set_ylabel("cost of one AI flag work-up (£)")
    cb = fig.colorbar(im, ax=ax, shrink=0.85); cb.set_label("incremental 10-year cost per 1,000 (£ thousand)\nblue = saves, red = costs more", fontsize=9)
fig.suptitle("Two-way threshold analysis at 3% 10-year progression (base-case rates; de-escalation 3 → 5 years)", fontsize=11, x=0.01, ha="left")
fig.savefig(FIG + "/fig_p2c_twoway.png", dpi=200); plt.close(fig)
# 2. efficiency frontier
OFFA = {"CC": (10, 8), "T0": (10, 10), "T1": (8, 18), "T2": (8, 34), "SE": (10, 8)}
B = evaluate(FIGPI, pb, 0, c_seq=250.0); D = evaluate(FIGPI, pd_, IDX, c_seq=250.0)
fig, axs = plt.subplots(1, 2, figsize=(12.5, 5.6), constrained_layout=True)
for ax, (xk, xl, strs) in zip(axs, (("np_flag_rule", "non-progressors flagged by the strategy, per 1,000 (10 years)", MAIN), ("prog_deesc", "progressors de-escalated to a longer interval, per 1,000", MAIN + ["GD"]))):
    pts = []
    for s in strs:
        x = float(B[s][xk]); y = float(B[s]["d_cost"]) / 1000; lo, hi = np.percentile(D[s]["d_cost"] / 1000, [2.5, 97.5]); xlo, xhi = np.percentile(D[s][xk], [2.5, 97.5])
        ax.errorbar(x, y, yerr=[[y - lo], [hi - y]], xerr=[[x - xlo], [xhi - x]], fmt="o", ms=9, color=COL[s], mec="white", mew=2, elinewidth=1.2, capsize=0, zorder=3); pts.append((x, y, s))
        if xk == "np_flag_rule":
            off = OFFA[s]; ax.annotate(f"{SHORT[s]} · {float(B[s]['prog_deesc']):.1f} missed", (x, y), off, textcoords="offset points", fontsize=9, color="#0b0b0b", ha="right" if off[0] < 0 else "left")
        else:
            ax.plot([], [], "o", color=COL[s], ms=8, label=f"{SHORT[s].replace(chr(10), ' ')}: {float(B[s]['np_deesc']):.0f} NP de-escalated")
    pts.sort(); fr = []; best = np.inf
    for x, y, s in pts:
        if y < best: fr.append((x, y)); best = y
    if xk == "np_flag_rule":
        fr = []; best = np.inf
        for x, y, s in sorted(pts, key=lambda t: t[0]):
            if y < best - 1e-9: fr.append((x, y)); best = y
        if len(fr) > 1: ax.plot(*zip(*fr), color="#898781", lw=1.5, ls="--", zorder=1)
    ax.axhline(0, color="#c3c2b7", lw=1); ax.set_xlabel(xl); ax.set_ylabel("incremental 10-year cost vs current care\n(£ thousand per 1,000; 95% interval)")
    ax.grid(axis="y", color="#e1e0d9", lw=0.6)
axs[0].set_title("A  Cost vs non-progressors flagged ('missed' = progressors de-escalated)", loc="left", fontsize=11)
fig.legend(*axs[1].get_legend_handles_labels(), loc="outside lower center", ncol=3, fontsize=9, frameon=False)
axs[1].set_title("B  Cost vs progressors exposed to a longer interval", loc="left", fontsize=11)
fig.suptitle(f"3% 10-year progression, sequencing £250 per sample, flag work-up £{flag_cost(pb):.0f}, per flagged endoscopy", fontsize=11, x=0.01, ha="left")
fig.savefig(FIG + "/fig_p2c_frontier.png", dpi=200); plt.close(fig)
print("figures", FIG)

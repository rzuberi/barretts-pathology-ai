"""Two papers, agent CLN (docs/two_papers_CLN.md): standalone scorer for frozen bundle v2 (killcoyne_mm/two_papers_REG/frozen_v2_clean/).
REG's scorer (tp_reg_score.py, unchanged and imported) applies R49, its package flavour and tier 0; this module adds the v2 light and confirmatory tiers,
which sit on the MEAN-LOGIT gate (rules.json "gate": "mean_logit"): endoscopy score z = mean sample L-IMG logit (REG's z_meanlogit, with its 1e-6
top-sample tie-break), banded at the mean-logit gate cut-offs; in the middle band the top-L-IMG sample is sequenced and the gate-aligned score is
logit(c*) + logit(its L-LATE) - logit(t_seq); EW memory of that score, minus delta while certified (an EARLIER middle-band endoscopy <= Y years before
whose sequenced top sample had logit L-CNV < quiet cut); flag if >= threshold; confirmatory = light + quiet-genome check of the two top-L-IMG samples.
Written from the rule text in rules.json, without the round-8/9 engine; tp_cln_freeze.py checks it against the engine flag for flag. numpy / pandas only.
Input as tp_reg_score.score: one row per sample (patient, endo, order, t_years, p_img, p_cnv, ps_oac); p_img must already be the v2 (clean-tile) L-IMG."""
import numpy as np, pandas as pd
import tp_reg_score as R0
from tp_reg_score import lg, sg, band_of, endoscopy_table, ew_memory, certified, check_score, scanner_reset, patient_scores   # re-exported

def score(df, rules):
    R = rules["rules"]; L = R["tier_light"]; C = R["tier_confirmatory"]
    if L.get("gate") != "mean_logit": return R0.score(df, rules)   # v1-style rules (R49-gate light / confirmatory tiers)
    E, df = R0.score(df, rules)   # R49, R49_pkg, tier 0 (and REG's R49-gate light / confirmatory columns, replaced below)
    cu = L["gate_cuts"]; pz = sg(E.z_meanlogit.values); bd = np.array([band_of(s, cu) for s in pz]); mid = bd == 1; E["mlgate_band"] = bd
    g_al = np.where(mid, lg(cu["c_star"]) + lg(E.late_top.values) - lg(cu["t_seq"]), E.z_meanlogit.values)
    qual = mid & (lg(E.cnv_top.values) < L["quiet_cut_logit_cnv"])
    u = ew_memory(E, g_al, L["half_life_years"]) - L["delta_logit"] * certified(E, qual, L["years"])
    E["light_score"] = u; E["light_flag"] = u >= L["threshold"]; E["light_seq"] = mid.astype(int)
    V = check_score(E, df); k2 = np.minimum(2, E.n_samples.values); fl = E.light_flag.values
    E["confirm_check_score"] = V; E["confirm_flag"] = fl & ~(V < C["check_bar_ps_oac"]); E["confirm_seq"] = E.light_seq.values + np.where(fl, np.where(mid, k2 - 1, k2), 0)
    return E, df

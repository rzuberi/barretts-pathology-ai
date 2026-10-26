"""Two papers, agent REG (docs/two_papers_REG.md): standalone scorer for the frozen bundle killcoyne_mm/two_papers_REG/frozen/.
Applies the frozen models' sample probabilities and the frozen rules (rules.json) to a new cohort. Written independently of the round-8/9 engine
(rv8_landscape.py, rv9_falseflags.py, ex_e2_veto.py, ex_e3_score.py, ex_e9_frontier.py); tp_reg_freeze.py checks that it reproduces the engine
flag for flag on the discovery data. numpy / pandas only.

Input: one row per sample with columns
  patient, endo (endoscopy key, unique within patient), order (time order of the endoscopy within the patient, 0 = first), t_years (endoscopy date
  in years from any origin), p_img (frozen L-IMG probability), p_cnv (frozen L-CNV probability; NaN if not sequenced), ps_oac (quiet-genome score
  of the sample; NaN if not sequenced). Rows keep their input order, which breaks ties when two samples of an endoscopy have the same p_img.
Output: one row per endoscopy with, for every rule, the flag, the rule score and the number of samples the rule sequences.
"""
import json, numpy as np, pandas as pd

CLIP = 1e-6
def lg(p): p = np.clip(np.asarray(p, float), CLIP, 1 - CLIP); return np.log(p / (1 - p))
def sg(z): return 1 / (1 + np.exp(-np.asarray(z, float)))

def band_of(s1, cuts):
    """three-band gate on the endoscopy H&E score s1 (probability): 0 low (s1 < t_low), 2 high (s1 > c_star), 1 middle; no middle band when t_low > c_star"""
    tl, cs = cuts["t_low"], cuts["c_star"]
    if tl > cs: return 2 if s1 > cs else 0
    return 0 if s1 < tl else (2 if s1 > cs else 1)

def endoscopy_table(df):
    """per endoscopy: H&E max and its sample, the two top-H&E samples, the mean-logit score; rows in (patient, order) order"""
    df = df.reset_index(drop=True).copy(); df["p_late"] = (df.p_cnv + df.p_img) / 2; out = []
    for (p, e), g in df.groupby(["patient", "endo"], sort=False):
        pi = g.p_img.values; od = np.argsort(-pi, kind="mergesort"); top = g.index.values[od[0]]; top2 = g.index.values[od[1]] if len(g) > 1 else -1
        z = lg(pi); a = sg(z.mean()); b = sg(lg(a) + 1e-6 * z)           # mean logit; 1e-6 x sample logit breaks ties (engine construction)
        out.append({"patient": p, "endo": e, "order": int(g.order.iloc[0]), "t_years": float(g.t_years.iloc[0]), "n_samples": len(g), "p_max": pi.max(),
                    "top": top, "top2": top2, "late_top": df.p_late[top], "cnv_top": df.p_cnv[top], "z_meanlogit": float(lg(b.max()))})
    E = pd.DataFrame(out).sort_values(["patient", "order"], kind="mergesort").reset_index(drop=True)
    assert not E.duplicated(["patient", "order"]).any(), "two endoscopies share a time order within a patient"
    return E, df

def ew_memory(E, s, h):
    """exponentially weighted mean of the patient's endoscopy scores so far (time order): weight 0.5 ** ((t_k - t_i) / h) for i <= k"""
    u = np.full(len(E), np.nan)
    for _, g in E.groupby("patient", sort=False):
        ix = g.index.values; t = g.t_years.values; v = s[ix]
        for q in range(len(ix)): w = 0.5 ** (np.maximum(t[q] - t[:q + 1], 0) / h); u[ix[q]] = (w * v[:q + 1]).sum() / w.sum()
    return u

def certified(E, qual, Y):
    """True if an EARLIER endoscopy of the patient (strictly before in time order) had qual True and lies <= Y years before"""
    c = np.zeros(len(E), bool)
    for _, g in E.groupby("patient", sort=False):
        ix = g.index.values; t = g.t_years.values; q = qual[ix]
        for k in range(len(ix)): c[ix[k]] = bool((q[:k] & (t[k] - t[:k] <= Y + 1e-9)).any())
    return c

def check_score(E, df):
    """quiet-genome check score: max ps_oac over the two top-H&E samples (one if the endoscopy has one sample); missing -> +inf (never cancels)"""
    q = df.ps_oac.values.astype(float); v = q[E.top.values]; v2 = np.where(E.top2.values >= 0, q[np.maximum(E.top2.values, 0)], -np.inf); v = np.fmax(v, v2)
    return np.where(np.isfinite(v) | (v == -np.inf), v, np.inf)

def score(df, rules):
    """apply every frozen rule; returns the endoscopy table with <rule>_flag, <rule>_score, <rule>_seq columns"""
    E, df = endoscopy_table(df); R = rules["rules"]
    # R49 gate (and the package-feature flavour): only t_seq differs between flavours
    for nm in ("R49", "R49_pkg"):
        if nm not in R: continue
        cu = R[nm]["cuts"]; bd = np.array([band_of(s, cu) for s in E.p_max]); E[f"{nm}_band"] = bd
        E[f"{nm}_flag"] = np.where(bd == 1, E.late_top >= cu["t_seq"], bd == 2); E[f"{nm}_seq"] = (bd == 1).astype(int)
    # gate-aligned score: H&E max logit, replaced in the middle band by the top sample's L-LATE logit shifted so that the gate's own decision is score >= logit(c*)
    cu = R["R49"]["cuts"]; mid = E.R49_band.values == 1
    g_al = np.where(mid, lg(cu["c_star"]) + lg(E.late_top.values) - lg(cu["t_seq"]), lg(E.p_max.values))
    # tier 0, no sequencing: EW memory of the mean-logit endoscopy score
    r = R["tier0_no_sequencing"]; u = ew_memory(E, E.z_meanlogit.values, r["half_life_years"]); E["tier0_score"] = u; E["tier0_flag"] = u >= r["threshold"]; E["tier0_seq"] = 0
    # light tier: EW memory of the gate-aligned score with a copy-number certificate
    r = R["tier_light"]; qual = mid & (lg(E.cnv_top.values) < r["quiet_cut_logit_cnv"])
    u = ew_memory(E, g_al, r["half_life_years"]) - r["delta_logit"] * certified(E, qual, r["years"]); E["light_score"] = u
    E["light_flag"] = u >= r["threshold"]; E["light_seq"] = mid.astype(int)
    # confirmatory tier: light tier + quiet-genome check on its flags (two top-H&E samples sequenced; one already in the middle band)
    r = R["tier_confirmatory"]; V = check_score(E, df); k2 = np.minimum(2, E.n_samples.values); fl = E.light_flag.values
    E["confirm_check_score"] = V; E["confirm_flag"] = fl & ~(V < r["check_bar_ps_oac"]); E["confirm_seq"] = E.light_seq.values + np.where(fl, np.where(mid, k2 - 1, k2), 0)
    return E, df

def scanner_reset(p_new, reset, pi=None):
    """label-free scanner reset: map new L-IMG probabilities onto the discovery out-of-fold distribution at progressor-sample share pi
    (quantile mapping: p' = F_ref^-1(F_new(p)), F_ref = pi F1 + (1 - pi) F0 from the stored class quantiles); monotone, so AUROC is unchanged"""
    pi = reset["pi_registered"] if pi is None else pi; qs = np.asarray(reset["probs"]); x1 = np.asarray(reset["quantiles_progressor"]); x0 = np.asarray(reset["quantiles_non_progressor"])
    grid = np.unique(np.concatenate([x0, x1])); Fm = pi * np.interp(grid, x1, qs, left=0, right=1) + (1 - pi) * np.interp(grid, x0, qs, left=0, right=1)
    p_new = np.asarray(p_new, float); r_ = pd.Series(p_new).rank(method="average").values; Fn = (r_ - 0.5) / len(p_new)
    Fm_u, iu = np.unique(Fm, return_index=True); return np.interp(Fn, Fm_u, grid[iu])

def patient_scores(df, col="p_img"):
    """registered patient score (Paper 1 ACE-B): endoscopy score = mean of the sample logits; patient score = max over the patient's endoscopies (logit scale)"""
    z = pd.Series(lg(df[col].values), index=df.index); e = z.groupby([df.patient, df.endo]).mean(); return e.groupby(level=0).max()

if __name__ == "__main__":
    import sys
    rules = json.load(open(sys.argv[1])); df = pd.read_csv(sys.argv[2]); E, _ = score(df, rules); E.to_csv(sys.argv[3], index=False)

"""Round 4 analysis (PLAN.md v2): idea 1 multifocal LGD -> HGD+ in follow-up; idea 2 next endoscopy as referee for the slide model.
Prints and writes aggregates only (analysis/progression/results.json)."""
import json, sys, warnings
sys.path.insert(0, sys.path[0]); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from sklearn.metrics import roc_auc_score
from prep import *
OUT = B + "/analysis/progression/results.json"
R = {}
sec, rep = sections(); e, case2e = endoscopies(sec)
R["data"] = {"endoscopies": len(e), "patients": int(e.anon_id.nunique()), "multi_report_endoscopies": int((e.n_reports > 1).sum())}


def follow(d, j):
    """From endoscopy row j (exclusive) onward: (days, event, event_on_emr, n_endoscopies_in_fu). Censor at EMR without HGD+."""
    t0 = d.date.iloc[j]
    for k in range(j + 1, len(d)):
        if d.hgdplus.iloc[k]: return (d.date.iloc[k] - t0).days, 1, bool(d.emr.iloc[k]), k - j
        if d.emr.iloc[k]: return (d.date.iloc[k] - t0).days, 0, False, k - j
    return (d.date.iloc[-1] - t0).days, 0, False, len(d) - 1 - j


# ---------- idea 1 ----------
dt_rows = []
rows, flow = [], {"first_lgd_no_prior_hgd": 0, "dropped_one_pot": 0, "dropped_no_followup": 0}
for pid, d in e.groupby("anon_id"):
    d = d.reset_index(drop=True)
    hit = d.index[d.n_lgd > 0]
    if not len(hit): continue
    i = hit[0]
    if d.hgdplus.iloc[:i + 1].any(): continue
    flow["first_lgd_no_prior_hgd"] += 1
    if d.n_bx.iloc[i] < 2: flow["dropped_one_pot"] += 1; continue
    if i == len(d) - 1: flow["dropped_no_followup"] += 1; continue
    t, ev, ev_emr, nfu = follow(d, i)
    for k in range(i + 1, len(d)):  # discrete-time rows (added after results review)
        dt_rows.append(dict(pid=pid, multi=int(d.n_lgd.iloc[i] >= 2), y=int(d.hgdplus.iloc[k]), log_pots_here=np.log1p(d.n_bx.iloc[k]),
                            emr_here=int(d.emr.iloc[k]), log_pots_index=np.log(d.n_bx.iloc[i]), age=d.age.iloc[i]))
        if d.hgdplus.iloc[k] or d.emr.iloc[k]: break
    j1 = i + 1
    first_rep = dict(hgd=int(d.hgdplus.iloc[j1]), on_emr=int(d.hgdplus.iloc[j1] and d.emr.iloc[j1]), gap=(d.date.iloc[j1] - d.date.iloc[i]).days)
    pre = d.iloc[i + 1:]; pre = pre[(pre.date - d.date.iloc[i]).dt.days <= t]
    gaps_pre = d.loc[[i] + list(pre.index), "date"].diff().dt.days.dropna().tolist()
    after = d.iloc[i + 1:]
    early_emr = bool(d.emr.iloc[i] or (after.emr & ((after.date - d.date.iloc[i]).dt.days <= 183)).any())
    pers = None
    j = i + 1
    if not d.hgdplus.iloc[j] and not d.emr.iloc[j] and j < len(d) - 1:
        tp, evp, evp_emr, _ = follow(d, j); pers = dict(pers=int(d.n_lgd.iloc[j] > 0), t=max(tp, 1), ev=evp, ev_emr=int(evp_emr), multi=int(d.n_lgd.iloc[i] >= 2))
    rows.append(dict(pid=pid, n_lgd=int(d.n_lgd.iloc[i]), n_lgd_conf=int(d.n_lgd_conf.iloc[i]), n_bx=int(d.n_bx.iloc[i]), age=d.age.iloc[i],
                     t=max(t, 1), ev=ev, ev_emr=ev_emr, nfu=nfu, gap_next=(d.date.iloc[i + 1] - d.date.iloc[i]).days,
                     incident=bool(i > 0 and d.worst.iloc[:i].max() <= RANK["NDBE"]), early_emr=early_emr, persist=pers,
                     first_rep=first_rep, gaps_pre=gaps_pre))
c = pd.DataFrame(rows)
c["multi"] = (c.n_lgd >= 2).astype(int); c["multi_conf"] = (c.n_lgd_conf >= 2).astype(int); c["log_nbx"] = np.log(c.n_bx)
flow["cohort"] = len(c); flow["events"] = int(c.ev.sum()); flow["age_missing"] = int(c.age.isna().sum())
R["idea1_flow"] = flow


def cox(df, cols, expo):
    df = df.dropna(subset=cols)
    m = CoxPHFitter().fit(df[["t", "ev"] + cols], "t", "ev")
    s = m.summary.loc[expo]
    return {"n": len(df), "events": int(df.ev.sum()), "events_multi": int(df.loc[df[expo] == 1, "ev"].sum()), "n_multi": int((df[expo] == 1).sum()),
            "HR": round(float(s["exp(coef)"]), 2), "lo": round(float(s["exp(coef) lower 95%"]), 2), "hi": round(float(s["exp(coef) upper 95%"]), 2), "p": round(float(s["p"]), 4)}


def crude(df, expo):
    out = {}
    for v, g in df.groupby(expo):
        out[f"{expo}={v}"] = {"n": len(g), "events": int(g.ev.sum()), "person_years": round(g.t.sum() / 365.25, 1),
                              "per_100py": round(100 * g.ev.sum() / (g.t.sum() / 365.25), 1)}
    return out


I = {}
I["primary_adjusted"] = cox(c, ["multi", "log_nbx", "age"], "multi")
I["c1_unadjusted"] = cox(c, ["multi"], "multi")
I["crude_rates"] = crude(c, "multi")
for name, lo, hi in (("0_12m", 0, 365), ("after_12m", 365, None)):
    if hi: p = c.assign(ev=((c.ev == 1) & (c.t <= hi)).astype(int), t=c.t.clip(upper=hi))
    else: p = c[c.t > lo].assign(t=lambda x: x.t - lo)
    I[f"c2_{name}"] = cox(p, ["multi"], "multi") if p.ev.sum() >= 10 else {"crude": crude(p, "multi"), "note": "under 10 events, crude only"}
I["c3_conf_ge075"] = cox(c[c.n_lgd_conf >= 1], ["multi_conf", "log_nbx", "age"], "multi_conf")
inc = c[c.incident]
I["c4_incident"] = cox(inc, ["multi", "log_nbx", "age"], "multi") if inc.ev.sum() >= 10 else {"crude": crude(inc, "multi"), "note": "under 10 events, crude only"}
I["c5_no_early_emr"] = cox(c[~c.early_emr], ["multi", "log_nbx", "age"], "multi")
I["c6_surveillance"] = {str(v): {"median_days_to_next": float(g.gap_next.median()), "median_days_between_endoscopies_before_event": float(np.median(sum(g.gaps_pre, []))),
                                 "events": int(g.ev.sum()), "events_on_emr_or_resection": int(g.ev_emr.sum()), "early_emr": int(g.early_emr.sum()),
                                 "median_pots_at_index": float(g.n_bx.median())} for v, g in c.groupby("multi")}
I["c7_by_pots"] = {k: {"n": len(g), "events": int(g.ev.sum())} for k, g in c.groupby(c.n_lgd.clip(upper=3).map({1: "1", 2: "2", 3: ">=3"}))}
km = {}
for v, g in c.groupby("multi"):
    f = KaplanMeierFitter().fit(g.t, g.ev)
    km[str(v)] = {f"{y}y": round(float(1 - f.predict(365.25 * y)), 3) for y in (1, 3, 5)} | {f"at_risk_{y}y": int((g.t >= 365.25 * y).sum()) for y in (1, 3, 5)}
I["cum_incidence"] = km
pc = pd.DataFrame([dict(r.persist, age=r.age, log_nbx=r.log_nbx) for r in c.itertuples() if r.persist])
sp = cox(pc, ["pers", "log_nbx", "age"], "pers") if pc.ev.sum() >= 10 else {}
sp = {k.replace("_multi", "_persistent"): v for k, v in sp.items()}
I["secondary_persistence"] = sp | {"crude": crude(pc, "pers")}
# clinical review additions (added after results review)
fr = pd.DataFrame(list(c.first_rep)).assign(multi=c.multi.values)
I["r1_first_repeat_yield"] = {str(v): {"n": len(g), "hgdplus": int(g.hgd.sum()), "hgdplus_on_emr_or_resection": int(g.on_emr.sum()),
                                       "median_gap_days": float(g.gap.median())} for v, g in fr.groupby("multi")}
I["r2_after_clean_first_repeat"] = {str(v): {"n": len(g), "events": int(g.ev.sum()), "events_on_biopsy_only": int((g.ev - g.ev_emr).sum()),
                                             "person_years": round(g.t.sum() / 365.25, 1)} for v, g in pc.groupby("multi")}
import statsmodels.api as sm
dtr = pd.DataFrame(dt_rows)
def gee(cols):
    X = sm.add_constant(dtr[cols].astype(float))
    m = sm.GEE(dtr.y, X, groups=dtr.pid, family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable()).fit()
    ci = m.conf_int().loc["multi"]
    return {"rows": len(dtr), "events": int(dtr.y.sum()), "OR": round(float(np.exp(m.params["multi"])), 2), "lo": round(float(np.exp(ci[0])), 2),
            "hi": round(float(np.exp(ci[1])), 2), "p": round(float(m.pvalues["multi"]), 4)}
I["r3_discrete_time_with_emr"] = gee(["multi", "log_pots_here", "emr_here", "log_pots_index"])
I["r3_discrete_time_no_emr"] = gee(["multi", "log_pots_here", "log_pots_index"])
R["idea1"] = I


# ---------- idea 2 ----------
sl = slide_scores()
sl = sl[(sl.n_pred > 0) & sl.sec_site.isin(OES)].copy()
sl["eid"] = sl.CaseName.map(case2e)
R["idea2_slides"] = {"scored_oes_slides": len(sl), "unmapped": int(sl.eid.isna().sum())}
es = sl.dropna(subset=["eid"]).groupby("eid").agg(pmax=("p_dys", "max"), nsl=("p_dys", "size"))
e2 = e.merge(es, left_on="eid", right_index=True, how="left")
rng = np.random.default_rng(0)


def auroc(y, s):
    y, s = np.asarray(y), np.asarray(s)
    if len(set(y)) < 2: return None
    bs = []
    for _ in range(2000):
        k = rng.integers(0, len(y), len(y))
        if len(set(y[k])) == 2: bs.append(roc_auc_score(y[k], s[k]))
    return {"n": len(y), "events": int(y.sum()), "auroc": round(roc_auc_score(y, s), 3), "lo": round(float(np.percentile(bs, 2.5)), 3), "hi": round(float(np.percentile(bs, 97.5)), 3)}


def pairs(grade, prior_block):
    """First eligible index per patient: worst == grade, scored slides, no earlier endoscopy at or above prior_block."""
    have, none = [], []
    for pid, d in e2.groupby("anon_id"):
        d = d.reset_index(drop=True)
        prior = d.worst.cummax().shift(1, fill_value=-1)
        ok = d.index[(d.worst == RANK[grade]) & d.pmax.notna() & (prior < RANK[prior_block])]
        if not len(ok): continue
        i = ok[0]
        if i == len(d) - 1: none.append(d.pmax.iloc[i]); continue
        have.append(dict(p=d.pmax.iloc[i], nsl=d.nsl.iloc[i], y=int(d.dysplasia.iloc[i + 1]), gap=(d.date.iloc[i + 1] - d.date.iloc[i]).days))
    return pd.DataFrame(have), np.array(none)


J = {}
p, none = pairs("NDBE", "LGD")
J["primary_ndbe"] = auroc(p.y, p.p)
J["ndbe_slide_count_only"] = auroc(p.y, p.nsl)
q = p[p.gap <= 365]; J["ndbe_gap_le_365"] = auroc(q.y, q.p)
J["ndbe_selection"] = {"eligible": len(p) + len(none), "no_next_endoscopy": len(none), "median_score_analysed": round(float(p.p.median()), 3),
                       "median_score_no_next": round(float(np.median(none)), 3) if len(none) else None, "median_gap_days": float(p.gap.median())}
p, _ = pairs("LGD", "HGD"); J["secondary_lgd_reverse"] = auroc(p.y, p.p)
p, _ = pairs("IND", "LGD")
J["ind_descriptive"] = {"n": len(p), "events": int(p.y.sum()), "median_score_events": round(float(p[p.y == 1].p.median()), 3) if p.y.sum() else None,
                        "median_score_no_event": round(float(p[p.y == 0].p.median()), 3)}
R["idea2"] = J
json.dump(R, open(OUT, "w"), indent=1, default=str)
print(json.dumps(R, indent=1, default=str))

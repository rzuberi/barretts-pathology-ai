"""ERL checks: (a) smoke = reproduce the digest's ERIN 3-year funnel (docs/results_digest_2026-09-21_22.md §1.2: 836 y3-defined index cases,
115 progressors; 2014-2021: 818 cases, 97 progressors) with the build_cohort.py rule on the same label file; (b) Barrett's-DB designs with the
pre-event endoscopies restricted to recent eras (glass slides < ~10-15 years old), from the tp_erl_archive.py tables. Aggregates only."""
import json, os, numpy as np, pandas as pd
from scipy.stats import chi2
T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"; W = T + "/feasibility/paper_plan/killcoyne_mm/two_papers_ERL"; R = T + "/results/paper_final/two_papers_ERL"
def sup(n): n = int(n); return n if (n == 0 or n >= 5) else "<5"
out = {}
# (a) digest funnel, build_cohort.py logic verbatim (index = first train-eligible/adjudicated NDBE/IND report; y3 rule with H = 1,095 d)
ORD = {"NDBE": 0, "IND": 1, "LGD": 2, "HGD": 3, "CANCER": 4}; H = 1095
lab = pd.read_csv(T + "/labeller/erin_labels_jury_final.csv", dtype=str).drop_duplicates("CaseName")
lab["d"] = pd.to_datetime(lab.CollectedOrOrdered, dayfirst=True, errors="coerce"); lab = lab[lab.d.notna()]
usable = lab[lab.label_status.isin(["train_eligible", "adjudicated"]) & lab.final_label.isin(ORD)].copy(); usable["g"] = usable.final_label.map(ORD)
rows = []
for pid, g in usable.sort_values("d").groupby("anon_id"):
    idx = g[g.g <= 1]
    if idx.empty: continue
    i0 = idx.iloc[0]; later = g[g.d > i0.d]; last = lab[lab.anon_id == pid].d.max(); ev = later[later.g >= 3]
    t_ev = (ev.d.min() - i0.d).days if len(ev) else None; fu = (last - i0.d).days
    y3 = 1 if (t_ev is not None and t_ev <= H) else (0 if (t_ev is None and fu >= H) else -1); rows.append({"y3": y3, "year": i0.d.year})
c = pd.DataFrame(rows); d3 = c[c.y3 >= 0]; e = d3[d3.year <= 2021]
out["smoke_digest_funnel"] = {"index_patients": len(c), "y3_defined": len(d3), "y3_pos": int(d3.y3.sum()), "y3_defined_2014_2021": len(e), "y3_pos_2014_2021": int(e.y3.sum()),
                              "expected": {"index_patients": 2092, "y3_defined": 836, "y3_pos": 115, "y3_defined_2014_2021": 818, "y3_pos_2014_2021": 97}}
# (b) DB designs by era
DP = pd.read_csv(W + "/db_patients.csv", parse_dates=["t_ev", "last_fu"]); DE = pd.read_csv(W + "/db_endoscopies.csv", parse_dates=["d"])
DP["pre_list"] = DP.pre_dates.fillna("").map(lambda s: sorted(pd.to_datetime([x for x in s.split("|") if x])))
el = DP[DP.barretts & ~DP.excluded_overlap].copy()
CLEAR, MISS = 16.4 / 43, 1.0 / 32
def pois_up(x): return float(chi2.ppf(0.975, 2 * x + 2) / 2)
for y0 in (2005, 2010, 2014):
    el["pl"] = el.pre_list.map(lambda l: [x for x in l if x.year >= y0])
    for k, h in ((2, 3), (3, 5)):
        kth = pd.to_datetime(el.pl.map(lambda l: l[k - 1] if len(l) >= k else pd.NaT)); fu = (el.last_fu - kth).dt.days / 365.25; tte = (el.t_ev - kth).dt.days / 365.25
        Pp = el[el.prog & (el.pl.map(len) >= k)]; Pn = el[~el.prog & (el.pl.map(len) >= k) & (fu >= h)]; Nall = el[~el.prog & (el.pl.map(len) >= k)]
        nE = int(Pp.pl.map(len).sum() + sum(len([x for x in l if x <= lf - pd.Timedelta(days=int(365.25 * h))]) for l, lf in zip(Pn.pl, Pn.last_fu)))
        py = float(fu[Nall.index].clip(lower=0).sum()); xm = int(round(MISS * len(Pp)))
        out[f"DB_from{y0}_k{k}_h{h}"] = {"progressors": sup(len(Pp)), "non_progressors": sup(len(Pn)), "endoscopies": sup(nE), "slides_est_2_per_endoscopy": 2 * nE,
                                          "np_py_cohort": round(py), "upper95_if_0_missed_per_100py": round(100 * pois_up(0) / (CLEAR * py), 3) if py else None,
                                          "upper95_at_expected_missed_per_100py": round(100 * pois_up(xm) / (CLEAR * py), 3) if py else None,
                                          "crude_incidence_per_100py": round(100 * len(Pp) / max(py + float(tte[Pp.index].clip(lower=0).sum()), 1e-9), 2)}
json.dump(out, open(R + "/checks.json", "w"), indent=1, default=str); print(json.dumps(out, indent=1, default=str))

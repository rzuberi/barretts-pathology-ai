"""ERL step 3: the targeted scan request. Progressors (P) = every eligible patient with >= 2 pre-event Barrett's endoscopies (primary subgroup >= 3);
non-progressors (NP) = gastro criteria, >= 3 pre-event endoscopies and >= 5 years' follow-up after the third, sampled 2 per P (seed 0), matched on the
year of the second pre-event endoscopy (closest first, within 3 years if possible) and on the number of pre-event endoscopies (2-3 / 4+); the sampling
fraction is recorded so cohort person-years can be reweighted. Endoscopies requested: P all pre-event; NP those with >= 5 years' follow-up after them.
Two sources: (A) ERIN report history, (B) the Barrett's database (same archive, PS accessions, longer history). Lists + key stay on the cluster."""
import json, os, numpy as np, pandas as pd
T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"; W = T + "/feasibility/paper_plan/killcoyne_mm/two_papers_ERL"; R = T + "/results/paper_final/two_papers_ERL"
def sup(n): n = int(n); return n if (n == 0 or n >= 5) else "<5"
v3 = pd.read_csv(T + "/labeller/erin_sections_v3_consensus.csv", dtype=str); oes = v3[v3.site.isin(["OESOPHAGUS", "GOJ"])].groupby("CaseName").section.agg(lambda s: "".join(sorted(set(s))))
def build(Pt, Et, src, idcol):
    Pt = Pt.copy(); Pt["pl"] = Pt.pre_dates.fillna("").map(lambda s: sorted(pd.to_datetime([x for x in s.split("|") if x]))); Pt["n"] = Pt.pl.map(len)
    el = Pt[Pt.barretts & ~Pt.excluded_overlap].copy(); el["y2"] = el.pl.map(lambda l: l[1].year if len(l) >= 2 else np.nan)
    el["t3"] = pd.to_datetime(el.pl.map(lambda l: l[2] if len(l) >= 3 else pd.NaT)); el["fu3"] = (el.last_fu - el.t3).dt.days / 365.25
    P = el[el.prog & (el.n >= 2)].sample(frac=1, random_state=0); pool = el[~el.prog & (el.n >= 3) & (el.fu3 >= 5)].copy(); pool["band"] = np.where(pool.n >= 4, 1, 0)
    rng = np.random.RandomState(0); chosen = []; diffs = []
    for p in P.itertuples():
        for _ in range(2):
            c = pool[~pool[idcol].isin(chosen)]
            if c.empty: break
            b = 1 if p.n >= 4 else 0; c = c.assign(cost=(c.y2 - p.y2).abs() + 10 * (c.band != b) + rng.rand(len(c)) * 1e-3); best = c.sort_values("cost").iloc[0]; chosen.append(best[idcol]); diffs.append(abs(best.y2 - p.y2))
    NP = pool[pool[idcol].isin(chosen)]
    rows = []
    for role, S in (("P", P), ("NP", NP)):
        for i, p in enumerate(S.itertuples()):
            ds = p.pl if role == "P" else [x for x in p.pl if x <= p.last_fu - pd.Timedelta(days=int(365.25 * 5))]
            g = Et[(Et[idcol] == getattr(p, idcol)) & Et.d.isin(ds)].sort_values("d")
            for j, e in enumerate(g.itertuples()):
                acc = e.cases if src == "ERIN" else e.acc
                rows.append({"study_code": f"{src[0]}{role}{i + 1:03d}", idcol: getattr(p, idcol), "role": role, "p_ge3": bool(p.n >= 3), "order": j + 1, "date": e.d, "accessions": acc,
                             "oes_sections": "|".join(f"{c}:{oes.get(c, '?')}" for c in str(acc).split("|") if c) if src == "ERIN" else "", "already_scanned": bool(e.scanned)})
    L = pd.DataFrame(rows); L["slides_est"] = L.oes_sections.map(lambda s: sum(max(len(x.split(":")[1]), 1) if x.split(":")[1] != "?" else 2 for x in s.split("|")) if s else 2)
    L["slides_cap2"] = L.slides_est.clip(upper=2)
    need = L[~L.already_scanned]; yr = need.date.dt.year
    d = {"P": sup(len(P)), "P_with_ge3": sup((P.n >= 3).sum()), "NP_pool": sup(len(pool)), "NP_sampled": sup(len(NP)), "NP_sampling_fraction": round(len(NP) / max(len(pool), 1), 3),
         "endoscopies": sup(len(L)), "endoscopies_P": sup((L.role == "P").sum()), "endoscopies_NP": sup((L.role == "NP").sum()), "already_scanned": sup(L.already_scanned.sum()),
         "to_scan": sup(len(need)), "to_scan_before_2011": sup((yr < 2011).sum()), "to_scan_2011_2015": sup(((yr >= 2011) & (yr <= 2015)).sum()), "to_scan_2016_on": sup((yr >= 2016).sum()),
         "slides_est_all_oes_blocks": int(need.slides_est.sum()), "slides_cap2": int(need.slides_cap2.sum()), "slides_cap2_slides_before_2016": int(need[yr < 2016].slides_cap2.sum())}
    L[["study_code", "role", "p_ge3", "order", "accessions", "oes_sections", "already_scanned", "slides_cap2"]].to_csv(W + f"/scan_request_{src}_final.csv", index=False)
    L[["study_code", idcol, "role", "order", "date", "accessions"]].to_csv(W + f"/scan_request_{src}_final_KEY.csv", index=False)
    pd.DataFrame({idcol: pool[idcol], "sampled": pool[idcol].isin(chosen)}).to_csv(W + f"/np_pool_{src}.csv", index=False)
    d["match_abs_year_diff_median"] = round(float(np.median(diffs)), 1) if diffs else None; d["match_within_3y_share"] = round(float(np.mean(np.array(diffs) <= 3)), 3) if diffs else None
    return d
E = pd.read_csv(W + "/endoscopies.csv", parse_dates=["d"]); Pe = pd.read_csv(W + "/patients_primary.csv", parse_dates=["t_ev", "last_report"]); Pe["last_fu"] = Pe.last_report
DE = pd.read_csv(W + "/db_endoscopies.csv", parse_dates=["d"]); DP = pd.read_csv(W + "/db_patients.csv", parse_dates=["t_ev", "last_fu"])
out = {"ERIN": build(Pe, E, "ERIN", "anon_id"), "DB": build(DP, DE, "DB", "pid")}
os.system(f"chmod 600 {W}/scan_request_* {W}/np_pool_*")
json.dump(out, open(R + "/scan_request_counts.json", "w"), indent=1, default=str); print(json.dumps(out, indent=1, default=str))

"""Shared prep for the progression analyses (ideas 1 and 2). Builds one row per endoscopy (patient + date; same-day reports merged)
from the report CSV and the jury section consensus. Aggregates only leave this module's callers; no report text is read."""
import json
import numpy as np, pandas as pd
B = "/mnt/scratche/slow/fmlab/zuberi01"
ERIN = "/mnt/scratche/slow/fmlab/datasets/imaging/ERIN/data/PathologyReport_AnonIds.csv"
SEC = B + "/phd/thesis/labeller/erin_sections_v3_consensus.csv"
SLIDES = B + "/analysis/isbi_rerun/slides.csv"
PREDS = B + "/analysis/isbi_rerun/preds"
GRADES = ["NORMAL_OTHER", "NDBE", "IND", "LGD", "HGD", "CANCER"]
RANK = {g: i for i, g in enumerate(GRADES)}
OES = {"OESOPHAGUS", "GOJ"}


def sections():
    rep = pd.read_csv(ERIN, dtype=str, usecols=["CaseName", "anon_id", "CollectedOrOrdered", "AgeAtInvestigation"]).fillna("").drop_duplicates("CaseName")
    rep["date"] = pd.to_datetime(rep.CollectedOrOrdered, dayfirst=True, errors="coerce")
    rep["age"] = pd.to_numeric(rep.AgeAtInvestigation, errors="coerce")
    sec = pd.read_csv(SEC, dtype=str).fillna("")
    sec = sec[sec.worst_grade.isin(GRADES)].drop_duplicates(["CaseName", "section"])

    def vf(r):
        try: return json.loads(r.grade_votes).get(r.worst_grade, 0) / int(r.n_jurors)
        except Exception: return np.nan
    sec["vote_frac"] = sec.apply(vf, axis=1)
    sec = sec.merge(rep[["CaseName", "anon_id", "date", "age"]], on="CaseName", how="inner")
    sec = sec[sec.date.notna()]
    sec["rank"] = sec.worst_grade.map(RANK)
    sec["oes"] = sec.site.isin(OES)
    sec["biopsy"] = sec.specimen_type.eq("BIOPSY")
    return sec, rep


def endoscopies(sec):
    """One row per endoscopy: a patient's reports within 7 days of the first. Oesophageal/GOJ sites only. Pot counts are the max over
    the endoscopy's reports, so a pot reported twice is not double counted. Returns (endoscopies, CaseName -> endoscopy id)."""
    s = sec[sec.oes]
    bx = s[s.biopsy]; lgd = bx[bx.worst_grade.eq("LGD")]
    r = s.groupby("CaseName").agg(anon_id=("anon_id", "first"), date=("date", "first"), age=("age", "first"), worst=("rank", "max"))
    r["n_bx"] = bx.groupby("CaseName").section.nunique()
    r["n_lgd"] = lgd.groupby("CaseName").section.nunique()
    r["n_lgd_conf"] = lgd[lgd.vote_frac >= 0.75].groupby("CaseName").section.nunique()
    r["emr"] = s[s.specimen_type.isin(["EMR_ESD", "RESECTION"])].groupby("CaseName").size() > 0
    r = r.fillna({"n_bx": 0, "n_lgd": 0, "n_lgd_conf": 0}).reset_index().sort_values(["anon_id", "date", "CaseName"])
    r["emr"] = r.emr.eq(True)
    eid, prev_pid, start, k = [], None, None, -1
    for pid, dt in zip(r.anon_id, r.date):
        if pid != prev_pid: prev_pid, start, k = pid, None, -1
        if start is None or (dt - start).days > 7: start, k = dt, k + 1
        eid.append(f"{pid}#{k}")
    r["eid"] = eid
    g = r.groupby("eid")
    e = g.agg(anon_id=("anon_id", "first"), date=("date", "min"), age=("age", "max"), worst=("worst", "max"), n_bx=("n_bx", "max"),
              n_lgd=("n_lgd", "max"), n_lgd_conf=("n_lgd_conf", "max"), emr=("emr", "any"), n_reports=("CaseName", "size")).reset_index()
    e = e.sort_values(["anon_id", "date"]).reset_index(drop=True)
    e["hgdplus"] = e.worst >= RANK["HGD"]; e["dysplasia"] = e.worst >= RANK["LGD"]
    gaps = e.groupby("anon_id").date.diff().dt.days.dropna()
    assert (gaps > 7).all(), "endoscopy merging failed"  # ponytail: the one self-check
    return e, dict(zip(r.CaseName, r.eid))


def slide_scores(arm="B", task="six"):
    """Out-of-fold dysplasia score per slide = P(LGD)+P(HGD)+P(CANCER), mean over seeds (and folds for the held-out human patients)."""
    sl = pd.read_csv(SLIDES, dtype={"idx": int})
    acc = np.zeros((len(sl), 6)); n = np.zeros(len(sl))
    for seed in range(3):
        for fold in range(5):
            d = np.load(f"{PREDS}/patient_{arm}_{task}_seed{seed}_fold{fold}.npz")
            for ik, pk in (("test_idx", "test_prob"), ("holdout_idx", "holdout_prob")):
                acc[d[ik]] += d[pk]; n[d[ik]] += 1
    p = acc / n[:, None]
    sl["p_dys"] = p[:, 3:].sum(1)
    sl["pred"] = [GRADES[i] for i in p.argmax(1)]
    sl["n_pred"] = n
    return sl

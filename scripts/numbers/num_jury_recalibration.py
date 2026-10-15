#!/usr/bin/env python3
"""Item 34: can the jury's over-grading be fixed without new labels? Five-fold CV over reports on the 658 SWG specimen
pairs (pathologist grade = truth): (a) plain majority; (b) drop the k jurors with worst CV QWK; (c) vote-threshold rule
(LGD+ only if >= k of 8 jurors say LGD+); (d) multinomial logistic on one-hot juror votes. Best rule by CV two-tier
agreement is then applied unchanged to the ACE-B anchor (jury_full votes on the matched DB reports)."""
import glob, json, os, itertools, numpy as np, pandas as pd
from sklearn.metrics import cohen_kappa_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
E = "/mnt/scratche/slow/fmlab/zuberi01/barretts_db_export"; T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"; A = "/mnt/scratche/slow/fmlab/zuberi01/phd/aceb_meta"; OUT = os.environ.get("OUTDIR", ".")
ORD = ["NDBE", "IND", "LGD", "HGD", "CANCER"]; O = {g: i for i, g in enumerate(ORD)}; SWG = {"BE": "NDBE", "ID": "IND", "LGD": "LGD", "HGD": "HGD", "IMC": "CANCER"}
sp = pd.read_parquet(E + "/specimen_pairs.parquet"); sp["truth"] = sp.swg_grade_code.map(SWG).map(O)
votes = {}
for f in glob.glob(E + "/jury_specimen/llm_grades_*.csv"):
    mo = os.path.basename(f).replace("llm_grades_", "").rsplit("_shard", 1)[0]
    try: d = pd.read_csv(f, dtype=str, on_bad_lines="skip")
    except Exception: continue
    for c, g in zip(d.CaseName, d.llm_grade):
        if g in O: votes.setdefault(mo, {})[c] = O[g]
J = sorted(votes); V = pd.DataFrame({mo: pd.Series(votes[mo]) for mo in J}).reindex(sp.CaseName); V.index = sp.CaseName
ok = sp.truth.notna().values & (V.notna().sum(1).values >= 5); V = V[ok]; y = sp.truth[ok].astype(int).values; grp = sp.report_id[ok].values
print("pairs usable:", len(y), "jurors:", J, flush=True)
def majority(M, thr=None):
    out = []
    for row in M.values:
        r = row[~np.isnan(row)].astype(int)
        if thr is None: out.append(np.bincount(r, minlength=5).argmax())
        else:
            if (r >= 2).sum() >= thr * len(r): out.append(int(np.bincount(r[r >= 2], minlength=5).argmax()))
            else: out.append(int(np.bincount(r[r < 2], minlength=5).argmax()) if (r < 2).any() else int(np.bincount(r, minlength=5).argmax()))
    return np.array(out)
def metrics(t, p): return {"n": int(len(t)), "exact": round(float((t == p).mean()), 4), "two_tier": round(float(((t >= 2) == (p >= 2)).mean()), 4), "qwk": round(float(cohen_kappa_score(t, p, weights="quadratic")), 4),
                          "overcall_rate_on_NDBE": round(float((p[t == 0] >= 1).mean()), 4), "LGDplus_sensitivity": round(float((p[t >= 2] >= 2).mean()), 4), "LGDplus_specificity": round(float((p[t < 2] < 2).mean()), 4)}
res = {"_meta": {"n_pairs": int(len(y)), "jurors": J, "cv": "5-fold GroupKFold by report"}, "swg": {}}
res["swg"]["a_majority"] = metrics(y, majority(V))
gkf = GroupKFold(5); folds = list(gkf.split(V, y, grp))
# (b) per-juror CV QWK and drop-k
jq = {mo: round(float(cohen_kappa_score(y[V[mo].notna().values], V[mo].dropna().astype(int).values, weights="quadratic")), 4) for mo in J}
res["swg"]["per_juror_qwk"] = jq; order = sorted(J, key=lambda m: jq[m])
for k in (1, 2, 3):
    keep = order[k:]; res["swg"][f"b_drop_{k}_worst"] = {"dropped": order[:k], **metrics(y, majority(V[keep]))}
# (c) vote-threshold rule, chosen by CV
best = None
for thr in (0.5, 0.625, 0.75, 0.875):
    pred = np.zeros(len(y), int)
    for tr, te in folds: pred[te] = majority(V.iloc[te], thr)
    m = metrics(y, pred); res["swg"][f"c_LGDplus_needs_{thr:.3f}_of_jurors"] = m
    if best is None or m["two_tier"] > best[1]["two_tier"]: best = (thr, m)
# (d) multinomial logistic on one-hot votes (CV)
X = np.zeros((len(y), len(J) * 5))
for i, mo in enumerate(J):
    v = V[mo].values
    for r in range(len(y)):
        if not np.isnan(v[r]): X[r, i * 5 + int(v[r])] = 1
pred = np.zeros(len(y), int)
for tr, te in folds: pred[te] = LogisticRegression(max_iter=2000, C=1.0).fit(X[tr], y[tr]).predict(X[te])
res["swg"]["d_multinomial_logistic_cv"] = metrics(y, pred)
res["swg"]["chosen_rule"] = {"rule": f"LGD+ needs >= {best[0]} of jurors (CV)", "cv_two_tier": best[1]["two_tier"]}
# ---- apply to ACE-B (whole-corpus jury votes on the matched DB reports) ----
ac = pd.read_csv(A + "/aceb_official_case_presence_in_barretts_db_20260527_150857.csv", dtype=str)
pt = pd.read_csv(E + "/pathology_text_normalised_full.csv", dtype=str, usecols=["pathology_text_id", "participant_id", "receiveddatetime"]); pt["d"] = pd.to_datetime(pt.receiveddatetime, dayfirst=True, errors="coerce")
SEA = {"IM": 0, "ID": 1, "LGD": 2, "HGD": 3, "IMC": 4}
fv = {}
for f in glob.glob(T + "/feasibility/runs/jury_full_*/output/llm_grades_*.csv"):
    mo = os.path.basename(f).replace("llm_grades_", "").rsplit("_shard", 1)[0]; d = pd.read_csv(f, dtype=str, on_bad_lines="skip")
    for c, g in zip(d.CaseName, d.llm_grade):
        if g in O: fv.setdefault(str(c), {})[mo] = O[g]
rows = []
for _, r in ac.iterrows():
    pids = [x.strip() for x in str(r.participant_ids).replace(";", ",").split(",") if x.strip() and x.strip().lower() != "nan"]
    d0 = min([x for x in (pd.to_datetime(r["Date AFI"], dayfirst=True, errors="coerce"), pd.to_datetime(r["Date WLE"], dayfirst=True, errors="coerce")) if pd.notna(x)], default=pd.NaT)
    if not pids or pd.isna(d0) or r.Histology_Seattle_Protocol not in SEA: continue
    cand = pt[pt.participant_id.isin(pids) & pt.d.notna()].copy(); cand["gap"] = (cand.d - d0).abs().dt.days; cand = cand[cand.gap <= 120].sort_values("gap")
    if cand.empty: continue
    pid_ = str(cand.iloc[0].pathology_text_id); vv = fv.get(pid_, {})
    if len(vv) >= 5: rows.append({"truth": SEA[r.Histology_Seattle_Protocol], **{mo: vv.get(mo, np.nan) for mo in J}})
ad = pd.DataFrame(rows); ya = ad.truth.astype(int).values; Va = ad[J]
res["aceb"] = {"n": int(len(ya)), "a_majority": metrics(ya, majority(Va)), "c_chosen_threshold_rule": metrics(ya, majority(Va, best[0])),
               **{f"b_drop_{k}_worst": metrics(ya, majority(Va[order[k:]])) for k in (1, 2, 3)}}
os.makedirs(OUT, exist_ok=True); json.dump(res, open(os.path.join(OUT, "jury_recalibration.json"), "w"), indent=1); print(json.dumps(res, indent=None))

#!/usr/bin/env python3
"""Item 30b: T4 (benign report -> prior LGD+ in history) split by whether any EARLIER report of the patient mentions
endoscopic therapy (RFA/EMR/ESD/ablation/APC/cryo). Image OOF = mean over 3 seeds of the queue's img_T4_f*_s* preds."""
import glob, json, os, re, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
TH = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"; Q = f"{TH}/feasibility/erin_fusion/queue/results"; OUT = os.environ.get("OUTDIR", "."); NB = 2000
d = pd.read_csv(f"{TH}/feasibility/erin_fusion/tasks/T4.csv", dtype=str); d["y"] = d.y.astype(int)
preds = {}
for f in glob.glob(f"{Q}/img_T4_f*_s*.json"):
    for k, v in json.load(open(f))["preds"].items(): preds.setdefault(k, []).append(float(v))
d["p"] = d.sample_id.map(lambda k: np.mean(preds[k]) if k in preds else np.nan); d = d.dropna(subset=["p"])
rep = pd.read_csv("/mnt/scratche/fast/fmlab/datasets/imaging/ERIN/data/PathologyReport_AnonIds.csv", dtype=str).fillna("")
rep["d"] = pd.to_datetime(rep.CollectedOrOrdered, dayfirst=True, errors="coerce")
rx = re.compile(r"\bRFA\b|radiofrequency|ablat|\bEMR\b|\bESD\b|mucosal resection|neosquam|argon|\bAPC\b|cryo", re.I)
rep["ther"] = (rep.ClinicalInformation_redacted + " " + rep.GrossDescription_redacted + " " + rep.MicroscopicDescription_redacted + " " + rep.FinalDiagnosis_redacted).str.contains(rx)
date_of = dict(zip(rep.CaseName, rep.d))
d["d"] = d.CaseName.map(date_of)
flag = []
for r in d.itertuples():
    g = rep[(rep.anon_id == r.anon_id) & (rep.d < r.d)]; flag.append(int(g.ther.any()))
d["prior_therapy"] = flag
# also: therapy mentioned in the index report itself (post-ablation neosquamous epithelium etc.)
d["index_mentions_therapy"] = d.CaseName.map(dict(zip(rep.CaseName, rep.ther))).astype(int)
y, p, t, g = d.y.values, d.p.values, d.prior_therapy.values, d.anon_id.values
up = np.unique(g); idx_of = {u: np.where(g == u)[0] for u in up}; rng = np.random.RandomState(0); B = []
while len(B) < NB:
    s = np.concatenate([idx_of[u] for u in rng.choice(up, len(up))])
    if len(set(y[s])) < 2 or len(set(y[s][t[s] == 0])) < 2 or len(set(y[s][t[s] == 1])) < 2: continue
    B.append(s)
def ci(v): return [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
res = {"n": int(len(d)), "pos": int(y.sum()), "prior_therapy_units": int(t.sum()), "pos_rate_prior_therapy": round(float(y[t == 1].mean()), 4), "pos_rate_no_prior_therapy": round(float(y[t == 0].mean()), 4),
       "index_report_mentions_therapy_units": int(d.index_mentions_therapy.sum()),
       "auroc_image_all": round(float(roc_auc_score(y, p)), 4), "auroc_prior_therapy_flag_alone": round(float(roc_auc_score(y, t)), 4), "ci_flag_alone": ci([roc_auc_score(y[s], t[s]) for s in B]),
       "auroc_index_mentions_therapy_alone": round(float(roc_auc_score(y, d.index_mentions_therapy.values)), 4),
       "auroc_image_within_no_prior_therapy": round(float(roc_auc_score(y[t == 0], p[t == 0])), 4), "ci": ci([roc_auc_score(y[s][t[s] == 0], p[s][t[s] == 0]) for s in B]), "n_no_prior_therapy": int((t == 0).sum()), "pos_no_prior_therapy": int(y[t == 0].sum()),
       "auroc_image_within_prior_therapy": round(float(roc_auc_score(y[t == 1], p[t == 1])), 4), "ci_prior": ci([roc_auc_score(y[s][t[s] == 1], p[s][t[s] == 1]) for s in B]), "n_prior_therapy": int((t == 1).sum()), "pos_prior_therapy": int(y[t == 1].sum()),
       "auroc_image_within_no_therapy_anywhere": round(float(roc_auc_score(y[(t == 0) & (d.index_mentions_therapy.values == 0)], p[(t == 0) & (d.index_mentions_therapy.values == 0)])), 4),
       "n_no_therapy_anywhere": int(((t == 0) & (d.index_mentions_therapy.values == 0)).sum()), "pos_no_therapy_anywhere": int(y[(t == 0) & (d.index_mentions_therapy.values == 0)].sum())}
os.makedirs(OUT, exist_ok=True); json.dump(res, open(os.path.join(OUT, "t4_treatment_split.json"), "w"), indent=1); print(json.dumps(res, indent=None))

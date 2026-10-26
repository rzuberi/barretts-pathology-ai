"""Items 2 and 4 (CPU): project-1 metrics (explore/clf/clf_metrics.py logic, unchanged) for Paper A encoder tags in O/preds, paired
against the jury, RL seed 0, the explore encoder seeds (mbb32_s0/1/2) and each other. Env: TAGS. Out: O/enc_metrics.json"""
import json, os
from sklearn.metrics import roc_auc_score
from pa_common import *
TAGS = os.environ["TAGS"].split(",")
REFS = {"rl_s0": B + "/analysis/rl_grader/preds/grpo_sft__checkpoint-300", "medgemma27b_zs": B + "/analysis/rl_grader/preds/medgemma27b_zs",
        "mbb32_s0": CLF + "/preds/mbb32_s0", "mbb32_s1": CLF + "/preds/mbb32_s1", "mbb32_s2": CLF + "/preds/mbb32_s2"}
lab = {r["id"]: r for r in rows("test485")}; ids = sorted(lab); w = np.array([lab[i]["w"] for i in ids]); pats = np.array([lab[i]["anon_id"] for i in ids])
hum = np.array([lab[i]["human"] for i in ids]); rand = np.array([lab[i]["stratum"] == "random" for i in ids])
preds = {"jury": np.array([lab[i]["label"] for i in ids])}; conf = {}
for t, p in list(REFS.items()) + [(t, O + "/preds/" + t) for t in TAGS]:
    d = load_preds(p, "test485")
    if d: preds[t] = np.array([d[i]["pred"] or "INVALID" for i in ids]); conf[t] = np.array([d[i].get("p_grade") or 0.0 for i in ids])
BS = patient_bootstrap(pats)
def stats(pr, ix=None):
    ix = np.arange(len(ids)) if ix is None else ix; ex = (pr[ix] == hum[ix]).astype(float); dy = (np.isin(pr[ix], list(POS)) == np.isin(hum[ix], list(POS))).astype(float)
    return {"exact_w": float((ex * w[ix]).sum() / w[ix].sum()), "dysp_w": float((dy * w[ix]).sum() / w[ix].sum())}
R = {"tags": TAGS, "test485": {}, "val": {}, "test": {}}
for t in TAGS + ["jury"] + list(REFS):
    if t not in preds: continue
    pr = preds[t]; s0 = stats(pr); bs = [stats(pr, ix) for ix in BS]
    r = {k: round(v, 4) for k, v in s0.items()}; r.update({k + "_ci": ci([b[k] for b in bs]) for k in s0})
    r["exact_unweighted"] = round(float((pr == hum).mean()), 4); r["exact_random50"] = round(float((pr[rand] == hum[rand]).mean()), 4)
    r["ind_overcall"] = int(((pr == "IND") & (hum != "IND")).sum()); r["lgd_recall"] = f"{int(((pr == 'LGD') & (hum == 'LGD')).sum())}/{int((hum == 'LGD').sum())}"
    if t in conf and conf[t].max() > 0:
        err = (pr != hum).astype(int); sc = 1 - conf[t]; r["error_auroc"] = round(float(roc_auc_score(err, sc)), 4); r["error_auroc_ci"] = ci([roc_auc_score(err[ix], sc[ix]) for ix in BS if 0 < err[ix].sum() < len(ix)])
    if t in TAGS:
        for other in ["jury", "rl_s0", "mbb32_s0", "mbb32_s1", "mbb32_s2"] + [x for x in TAGS if x != t]:
            if other not in preds: continue
            po = preds[other]; r[f"exact_w_minus_{other}"] = {"point": round(s0["exact_w"] - stats(po)["exact_w"], 4), "ci": ci([stats(pr, ix)["exact_w"] - stats(po, ix)["exact_w"] for ix in BS])}
        lg = f"{O}/runs/{t}/log.json"
        if os.path.exists(lg): r["train_log"] = {k: v for k, v in json.load(open(lg)).items() if k != "evals"}
    R["test485"][t] = r
for s in ("val", "test"):
    for t in TAGS:
        d = load_preds(O + "/preds/" + t, s)
        if d: R[s][t] = {"n": len(d), "exact_vs_jury": round(float(np.mean([v["pred"] == v["label"] for v in d.values()])), 4)}
json.dump(R, open(O + "/enc_metrics.json", "w"), indent=1); os.chmod(O + "/enc_metrics.json", 0o600)
for t, r in R["test485"].items(): print(f"{t:14s} exact_w {100*r['exact_w']:.1f} {[round(100*x,1) for x in r['exact_w_ci']]} dysp_w {100*r['dysp_w']:.1f} unw {100*r['exact_unweighted']:.1f} rand50 {100*r['exact_random50']:.0f} LGD {r['lgd_recall']} errAUC {r.get('error_auroc')}", {k[len('exact_w_minus_'):]: (round(100*v['point'],1), [round(100*x,1) for x in v['ci']]) for k, v in r.items() if k.startswith("exact_w_minus")}, r.get("train_log", {}).get("n_train"), r.get("train_log", {}).get("best_val_exact"))
for s in ("val", "test"): print(s, R[s])

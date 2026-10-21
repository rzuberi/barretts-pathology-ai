"""Probe strand, CPU: fit probes on train+val hidden states, score test485/pop, error-detection AUROC vs Rehan's labels, and write
pred files (p1 generated grades + probe confidence as p_grade) for probe_metrics.py. Nothing is fitted or chosen on test485/pop.
Targets: (i) 'dis' = jury vote fraction < 0.75; (ii) 'err' = forced grade != jury consensus. Env: LAYERS (default 17,34), OUT dir, TARGETS (default err,dis).
Train/val/test hold only vote fraction >= 0.75, so for (i) the 1,028 low-agreement sections (all_lowconf feats) are added, split by patient (85/15) into train/val."""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")); from common import *
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
O = os.environ.get("OUT", B + "/analysis/rl_explore/probe"); F = O + "/feats"; A0 = B + "/analysis/rl_abstain"; R = B + "/analysis/rl_grader"
LAYERS = [int(x) for x in os.environ.get("LAYERS", "17,34").split(",")]; NB = 2000; RNG = np.random.RandomState(20261016); os.makedirs(O + "/preds", exist_ok=True)
def meta(s): return [json.loads(l) for l in open(f"{F}/{s}__meta.jsonl")]
TARGETS = os.environ.get("TARGETS", "err,dis").split(",")
M = {s: meta(s) for s in ["train", "val", "test", "test485", "pop"]}
HAS_LC = os.path.exists(f"{F}/all_lowconf__meta.jsonl")
if HAS_LC:   # low-agreement sections: patient-hashed 85/15 into train/val (never test485/pop)
    import hashlib; lc = meta("all_lowconf"); lc_tr = [r for r in lc if int(hashlib.md5(r["anon_id"].encode()).hexdigest(), 16) % 100 < 85]; lc_va = [r for r in lc if r not in lc_tr]
    print("low-agreement sections added: train", len(lc_tr), "val", len(lc_va), flush=True)
def vf(r): return r["vf"] if "vf" in r else r["jury_vf"]
Y = {s: {"dis": np.array([(vf(r) if vf(r) is not None else 0) < 0.75 for r in M[s]], int), "err": np.array([r["forced_pred"] != r["label"] for r in M[s]], int)} for s in M}
for s in M: print(s, len(M[s]), "dis rate", round(Y[s]["dis"].mean(), 4), "err rate (forced vs jury)", round(Y[s]["err"].mean(), 4), flush=True)
X = {s: {L: np.load(f"{F}/{s}__L{L}.npy").astype(np.float32) for L in LAYERS} for s in M}
if HAS_LC:
    XL = {L: np.load(f"{F}/all_lowconf__L{L}.npy").astype(np.float32) for L in LAYERS}; tr_m = np.array([r in lc_tr for r in lc])
    for L in LAYERS: X["train"][L] = np.vstack([X["train"][L], XL[L][tr_m]]); X["val"][L] = np.vstack([X["val"][L], XL[L][~tr_m]])
    for s, part in (("train", lc_tr), ("val", lc_va)):
        M[s] = M[s] + part; Y[s] = {"dis": np.array([(vf(r) if vf(r) is not None else 0) < 0.75 for r in M[s]], int), "err": np.array([r["forced_pred"] != r["label"] for r in M[s]], int)}
    for s in ("train", "val"): print(s, "with low-agreement:", len(M[s]), "dis rate", round(Y[s]["dis"].mean(), 4), "err rate", round(Y[s]["err"].mean(), 4), flush=True)
res = {"targets": {}, "probes": {}}
scores = {}   # name -> dict id -> confidence (higher = more likely correct) on pop and test485
def fit_lr(Xtr, ytr, Xva, yva):
    best = None
    for C in [1e-3, 1e-2, 1e-1, 1.0]:
        m = LogisticRegression(C=C, max_iter=2000, class_weight="balanced").fit(Xtr, ytr); a = roc_auc_score(yva, m.decision_function(Xva))
        if best is None or a > best[1]: best = (C, a)
    return best
for L in LAYERS:
    sc = StandardScaler().fit(np.vstack([X["train"][L], X["val"][L]]))
    Xs = {s: sc.transform(X[s][L]) for s in M}
    for tgt in TARGETS:
        if Y["train"][tgt].sum() == 0 or Y["val"][tgt].sum() == 0: print("skip", tgt, "no positives"); continue
        t0 = time.time(); C, aval = fit_lr(Xs["train"], Y["train"][tgt], Xs["val"], Y["val"][tgt])
        lr = LogisticRegression(C=C, max_iter=2000, class_weight="balanced").fit(np.vstack([Xs["train"], Xs["val"]]), np.concatenate([Y["train"][tgt], Y["val"][tgt]]))
        mlp = MLPClassifier(hidden_layer_sizes=(256,), alpha=1e-2, early_stopping=True, max_iter=200, random_state=0).fit(np.vstack([Xs["train"], Xs["val"]]), np.concatenate([Y["train"][tgt], Y["val"][tgt]]))
        for kind, m in (("lr", lr), ("mlp", mlp)):
            name = f"probe_{tgt}_L{L}_{kind}"; p = {s: m.predict_proba(Xs[s])[:, 1] for s in M}
            r = {"C": C if kind == "lr" else None, "auroc_val_trainonly": round(float(aval), 4) if kind == "lr" else None,
                 "auroc_test_own_target": round(float(roc_auc_score(Y["test"][tgt], p["test"])), 4),
                 "auroc_test_other_target": (round(float(roc_auc_score(Y["test"]["err" if tgt == "dis" else "dis"], p["test"])), 4) if 0 < Y["test"]["err" if tgt == "dis" else "dis"].sum() < len(p["test"]) else None),
                 "mean_p_pop": round(float(p["pop"].mean()), 4), "fit_seconds": round(time.time() - t0)}
            res["probes"][name] = r; scores[name] = {s: {M[s][i]["id"]: 1 - float(p[s][i]) for i in range(len(M[s]))} for s in ("pop", "test485")}
            print(name, r, flush=True)
# forced-choice and generated token-probability baselines on the same sections
for s in ("pop", "test485"):
    scores.setdefault("forced_maxprob", {})[s] = {r["id"]: float(max(r["forced_probs"])) for r in M[s]}
    scores.setdefault("forced_margin", {})[s] = {r["id"]: float(np.sort(r["forced_probs"])[-1] - np.sort(r["forced_probs"])[-2]) for r in M[s]}
p1 = {s: {json.loads(l)["id"]: json.loads(l) for l in open(f"{A0}/preds/p1best_conf__{s}.jsonl")} for s in ("pop", "test485")}
for s in ("pop", "test485"): scores.setdefault("p1_tokenprob", {})[s] = {i: (p1[s][i].get("p_grade") or 0.0) for i in p1[s]}
# combination: mean rank of token prob and the primary probe, ranks computed on the population (no labels used)
def rankavg(a, b, s):
    ids = sorted(a[s]); ra = np.argsort(np.argsort([a[s][i] for i in ids])); rb = np.argsort(np.argsort([b[s][i] for i in ids])); return {i: float((x + y) / 2 / len(ids)) for i, x, y in zip(ids, ra, rb)}
for L in LAYERS:
    for tgt in TARGETS:
        n = f"probe_{tgt}_L{L}_lr"
        if n not in scores: continue
        scores[f"combo_{tgt}_L{L}"] = {s: rankavg(scores["p1_tokenprob"], scores[n], s) for s in ("pop", "test485")}
# error-detection AUROC on test485 vs Rehan (weighted, patient bootstrap); error = p1 generated grade != human (jury vf judges the jury)
lab = {json.loads(l)["id"]: json.loads(l) for l in open(R + "/data/test485.jsonl")}; ids = sorted(lab)
hum = np.array([lab[i]["human"] for i in ids]); w = np.array([lab[i]["w"] for i in ids]); pats = np.array([lab[i]["anon_id"] for i in ids])
gp1 = np.array([p1["test485"][i]["pred"] or "X" for i in ids]); wrong_p1 = gp1 != hum; wrong_j = np.array([lab[i]["label"] for i in ids]) != hum
gf = np.array([next(r for r in M["test485"] if r["id"] == i)["forced_pred"] for i in ids]); wrong_f = gf != hum
def wauc(score, wrong, ww):
    pos, neg = ~wrong, wrong
    if not pos.any() or not neg.any(): return np.nan
    o = np.argsort(score, kind="stable"); s, p_, n_, w_ = score[o], pos[o], neg[o], ww[o]; tot = 0.0; cum_neg = 0.0; i = 0
    while i < len(s):
        j = i
        while j < len(s) and s[j] == s[i]: j += 1
        wp = (w_[i:j] * p_[i:j]).sum(); wn = (w_[i:j] * n_[i:j]).sum(); tot += wp * (cum_neg + 0.5 * wn); cum_neg += wn; i = j
    return tot / (ww[pos].sum() * ww[neg].sum())
up = np.unique(pats); grp = {p: np.where(pats == p)[0] for p in up}; BS = [np.concatenate([grp[p] for p in RNG.choice(up, len(up))]) for _ in range(NB)]
ci = lambda v: [round(float(np.nanpercentile(v, 2.5)), 4), round(float(np.nanpercentile(v, 97.5)), 4)]
res["auroc_test485"] = {"n": len(ids), "errors_p1": int(wrong_p1.sum()), "errors_jury": int(wrong_j.sum()), "errors_forced": int(wrong_f.sum()),
                        "acc_weighted_p1": round(float(((~wrong_p1) * w).sum() / w.sum()), 4), "acc_weighted_forced": round(float(((~wrong_f) * w).sum() / w.sum()), 4), "scores": {}}
S = {"jury_votefrac": (np.array([lab[i]["jury_vf"] for i in ids]), wrong_j)}
for n in scores: S[n] = (np.array([scores[n]["test485"][i] for i in ids]), wrong_p1)
tp = S["p1_tokenprob"][0]
for n, (sc, wr) in S.items():
    r = {"auroc": round(float(wauc(sc, wr, w)), 4), "ci": ci([wauc(sc[ix], wr[ix], w[ix]) for ix in BS]), "judges": "jury" if n == "jury_votefrac" else "p1 model"}
    if n != "p1_tokenprob" and n != "jury_votefrac": r["minus_tokenprob"] = {"point": round(float(wauc(sc, wr, w) - wauc(tp, wrong_p1, w)), 4), "ci": ci([wauc(sc[ix], wr[ix], w[ix]) - wauc(tp[ix], wrong_p1[ix], w[ix]) for ix in BS])}
    res["auroc_test485"]["scores"][n] = r; print(f"AUROC {n:24s}", r, flush=True)
# forced-grade system judged by its own probe (the probe's native labelling system)
for L in LAYERS:
    if f"probe_err_L{L}_lr" not in scores: continue
    sc = np.array([scores[f"probe_err_L{L}_lr"]["test485"][i] for i in ids]); res["auroc_test485"]["scores"][f"probe_err_L{L}_lr__judging_forced_grades"] = {"auroc": round(float(wauc(sc, wrong_f, w)), 4), "ci": ci([wauc(sc[ix], wrong_f[ix], w[ix]) for ix in BS]), "judges": "forced grades"}
# pred files for the triage estimator: p1 generated grades, p_grade = probe confidence
for n in scores:
    if n == "p1_tokenprob": continue
    for s in ("pop", "test485"):
        with open(f"{O}/preds/{n}__{s}.jsonl", "w") as f:
            for i in p1[s]: d = {k: p1[s][i][k] for k in p1[s][i] if k not in ("raw", "samples")}; d["pred"] = p1[s][i]["pred"]; d["p_grade"] = scores[n][s][i]; f.write(json.dumps(d) + "\n")
    for s in ("pop",):   # the same orderings applied to the forced-choice grades (secondary)
        with open(f"{O}/preds/{n}_fg__{s}.jsonl", "w") as f:
            for r in M[s]: d = {k: r[k] for k in ("id", "report_id", "section", "anon_id", "label", "jury_vf")}; d["pred"] = r["forced_pred"]; d["p_grade"] = scores[n][s][r["id"]]; f.write(json.dumps(d) + "\n")
json.dump(res, open(O + f"/probe_results_{'_'.join(TARGETS)}.json", "w"), indent=1); print("FIT DONE", list(scores))

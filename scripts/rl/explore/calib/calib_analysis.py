"""calib strand (CPU): recalibration, combined triage scores and conformal sets from the six-grade log-scores written by score6.py.
Fits (temperature, isotonic, logistic combiners, conformal quantile) use validation only (jury labels). The 479 human sections are test only.
Triage estimator = p2_metrics.py curve/at (copied unchanged). Env: TAGS (comma, default p1,d2), OUT dir. Prints aggregates only."""
import json, math, os, sys
import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
RL = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis/scripts/rl"; sys.path.insert(0, RL); from common import *
R = B + "/analysis/rl_grader"; O = os.environ.get("OUT", B + "/analysis/rl_explore/calib"); P = O + "/preds"
if os.environ.get("FALLBACK") == "1": O = O + "/fallback_pgrade"; os.makedirs(O, exist_ok=True)
TAGS = os.environ.get("TAGS", "p1,d2").split(","); FB = os.environ.get("FALLBACK") == "1"; NB = 2000; RNG = np.random.RandomState(20261016); PFRAC = 0.20
BUDS = [0.05, 0.10, 0.17, 0.20, 0.30]

# ---------- population + labelled sections (as p2_metrics.py) ----------
pop = [json.loads(l) for l in open(R + "/data/pop.jsonl")]; pid = [p["id"] for p in pop]; prep = np.array([p["report_id"] for p in pop]); nrep = len(set(prep))
lab = {json.loads(l)["id"]: json.loads(l) for l in open(R + "/data/test485.jsonl")}
L = sorted(set(lab) & set(pid)); pos = {i: k for k, i in enumerate(pid)}; Li = np.array([pos[i] for i in L])
w = np.array([lab[i]["w"] for i in L]); hum = np.array([lab[i]["human"] for i in L]); pats = np.array([lab[i]["anon_id"] for i in L]); amb = np.array([lab[i]["ambiguous"] for i in L])
jury_pop = np.array([p["label"] for p in pop]); vf_pop = np.array([p["jury_vf"] if p["jury_vf"] is not None else 0.0 for p in pop])
up = np.unique(pats); grp = {p: np.where(pats == p)[0] for p in up}; BS = [np.concatenate([grp[p] for p in RNG.choice(up, len(up))]) for _ in range(NB)]
ALL = np.arange(len(L)); ci = lambda v: [round(float(np.nanpercentile(v, 2.5)), 4), round(float(np.nanpercentile(v, 97.5)), 4)]
BUD = np.arange(0, 501) / 1000.0
def curve(g, sc):
    order = np.lexsort((RNG.rand(len(sc)), sc)); ss = sc[order]; reps = prep[order]
    opened = np.zeros(len(sc) + 1, int); seen = set()
    for j, r in enumerate(reps): seen.add(r); opened[j + 1] = len(seen)
    n = np.round(BUD * len(sc)).astype(int); fr = opened[n] / nrep
    q = np.zeros((len(n), len(Li)))
    for b, k in enumerate(n):
        if k == 0: continue
        thr = ss[k - 1]; below = int((ss < thr).sum()); tied = int((ss == thr).sum()); frac = (k - below) / tied
        s_l = sc[Li]; q[b] = np.where(s_l < thr, 1.0, np.where(s_l == thr, frac, 0.0))
    ok = (g[Li] == hum).astype(float)[None, :]; Z = q + (1 - q) * ok
    return fr, Z
def at(fr, Z, frac, ix):
    k = max(np.searchsorted(fr, frac, side="right") - 1, 0); ww = w[ix]; return float((Z[k, ix] * ww).sum() / ww.sum())
def wauc(score, wrong, ww):
    pos_, neg = ~wrong, wrong
    if not pos_.any() or not neg.any(): return np.nan
    o = np.argsort(score, kind="stable"); s, p_, n_, w_ = score[o], pos_[o], neg[o], ww[o]
    tot = 0.0; cum_neg = 0.0; i = 0
    while i < len(s):
        j = i
        while j < len(s) and s[j] == s[i]: j += 1
        wp = (w_[i:j] * p_[i:j]).sum(); wn = (w_[i:j] * n_[i:j]).sum(); tot += wp * (cum_neg + 0.5 * wn); cum_neg += wn; i = j
    return tot / (ww[pos_].sum() * ww[neg].sum())
def ece(score, correct, ww, nb=10):
    bins = np.minimum((score * nb).astype(int), nb - 1); e = 0.0; rows = []
    for b in range(nb):
        m = bins == b
        if not m.any(): continue
        conf = (score[m] * ww[m]).sum() / ww[m].sum(); acc = (correct[m] * ww[m]).sum() / ww[m].sum(); e += ww[m].sum() / ww.sum() * abs(acc - conf)
        rows.append({"bin": b, "n": int(m.sum()), "mean_conf": round(float(conf), 4), "accuracy": round(float(acc), 4)})
    return e, rows

# ---------- per-model scores ----------
FBF = {"p1": B + "/analysis/rl_abstain/preds/p1best_conf__{s}.jsonl", "d2": B + "/analysis/rl_followups/D/preds/d2__checkpoint-150__{s}.jsonl"}
CH = {}
def load(tag, s):
    f = FBF[tag].format(s=s) if FB else f"{P}/{tag}__{s}.jsonl"
    if not os.path.exists(f): return None
    rows = [json.loads(l) for l in open(f)]
    if FB:   # fallback: only the old first-token p_grade exists -> spread the rest uniformly over the other five grades; report length from the data file
        if s not in CH: CH[s] = {json.loads(l)["id"]: len(json.loads(l)["report"]) for l in open(f"{R}/data/{s}.jsonl")}
        for r in rows:
            pg = r.get("p_grade"); pg = min(max(pg if pg is not None else 1 / 6, 1e-6), 1 - 1e-6); r["report_chars"] = CH[s][r["id"]]
            r["first_lp"] = {g: math.log(pg) if g == r["pred"] else math.log((1 - pg) / 5) for g in GRADES} if r["pred"] in ORD else None
    return rows
def arr(rows):
    """lp6 (n,6) with None -> -inf; pred index (-1 invalid); features"""
    # six-grade log-scores = first-token logprobs at the grade position (top-20 of the greedy step; absent -> floor -30, i.e. ~1e-13)
    lp = np.array([[(r["first_lp"] or {}).get(g) if (r["first_lp"] or {}).get(g) is not None else -30.0 for g in GRADES] for r in rows], float)
    pred = np.array([ORD.get(r["pred"], -1) for r in rows]); pg = np.array([r["p_grade"] if r.get("p_grade") is not None else 0.0 for r in rows])
    sec = np.array([(-1 if r["section"] == "WHOLE" else (ord(r["section"][0].upper()) - 65)) for r in rows], float)
    whole = (sec < 0).astype(float); chars = np.log(np.array([r["report_chars"] for r in rows], float))
    return lp, pred, pg, np.stack([sec, whole, chars], 1)
def soft(lp, T): z = lp / T; z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)
def pick(p, pred): return np.where(pred >= 0, p[np.arange(len(p)), np.maximum(pred, 0)], 0.0)
def feats(p, pred, meta):
    pp = np.clip(pick(p, pred), 1e-6, 1 - 1e-6); ent = -(p * np.log(np.clip(p, 1e-12, 1))).sum(1); srt = np.sort(p, 1); marg = srt[:, -1] - srt[:, -2]
    return np.column_stack([np.log(pp / (1 - pp)), ent, marg, meta])

res = {"n_labelled": len(L), "n_population": len(pid), "n_reports": nrep, "models": {}}
M = {"jury_votefrac": (jury_pop, vf_pop)}
for tag in TAGS:
    V, TE, T4, PO = (load(tag, s) for s in ("val", "test", "test485", "pop")); LC = load(tag, "all_lowconf")
    if not (V and T4 and PO): print("missing preds for", tag); continue
    if not TE: TE = V[:0] or [dict(V[0])]   # jury test fold not scored (fallback): placeholder of one row, numbers for it are not reported
    r = {"n": {s: len(x) for s, x in (("val", V), ("test", TE), ("test485", T4), ("pop", PO))}}
    lpV, prV, pgV, mV = arr(V); yV = np.array([ORD[x["label"]] for x in V]); vfV = np.array([x["vf"] for x in V])
    lpT, prT, pgT, mT = arr(TE); yT = np.array([ORD[x["label"]] for x in TE])
    lpP, prP, pgP, mP = arr(PO); yP = np.array([ORD[x["label"]] for x in PO]); vfP = np.array([x["jury_vf"] if x["jury_vf"] is not None else 0.0 for x in PO])
    assert [x["id"] for x in PO] == pid, "pop order"
    lp4, pr4, pg4, m4 = arr([{**x} for x in [dict(y) for y in sorted(T4, key=lambda z: z["id"])]]); ids4 = sorted(x["id"] for x in T4); assert ids4 == L
    r["argmax_eq_pred_val"] = round(float((lpV.argmax(1) == prV).mean()), 4); r["invalid_pred"] = {s: int((x < 0).sum()) for s, x in (("val", prV), ("test485", pr4), ("pop", prP))}
    r["lp6_mass_val_median"] = round(float(np.median(np.exp(lpV).sum(1))), 4)   # mass of the six first tokens at the grade step (1 = all mass on the six grades)
    r["first_lp_missing_val"] = int(sum(1 for x in V if not x.get("first_lp")))
    # --- temperature on val (NLL vs jury label), isotonic on p_pred(T=1) vs correctness ---
    nll = lambda logT: -np.log(np.clip(soft(lpV, math.exp(logT))[np.arange(len(yV)), yV], 1e-12, 1)).mean()
    opt = minimize_scalar(nll, bounds=(-3, 3), method="bounded"); T_ = math.exp(opt.x); r["temperature"] = round(T_, 4); r["val_nll_T1"] = round(float(nll(0.0)), 4); r["val_nll_T"] = round(float(opt.fun), 4)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(pick(soft(lpV, 1), prV), (prV == yV).astype(float))
    # --- calibration on the 479 (vs human, weighted) and on the jury test fold ---
    pred4 = np.array([GRADES[i] if i >= 0 else "X" for i in pr4]); wrong4 = pred4 != hum
    confs = {"first_token_p_grade (Task A)": pg4, "p_pred_T1": pick(soft(lp4, 1), pr4), "p_pred_temperature": pick(soft(lp4, T_), pr4), "p_pred_isotonic": iso.predict(pick(soft(lp4, 1), pr4))}
    confsT = {"first_token_p_grade (Task A)": pgT, "p_pred_T1": pick(soft(lpT, 1), prT), "p_pred_temperature": pick(soft(lpT, T_), prT), "p_pred_isotonic": iso.predict(pick(soft(lpT, 1), prT))}
    r["calibration_479"] = {}; r["calibration_jurytest"] = {}
    for k, sc in confs.items():
        e, rows = ece(sc, (~wrong4).astype(float), w)
        r["calibration_479"][k] = {"ece": round(float(e), 4), "ece_ci": ci([ece(sc[ix], (~wrong4[ix]).astype(float), w[ix])[0] for ix in BS]),
                                   "auroc": round(float(wauc(sc, wrong4, w)), 4), "auroc_ci": ci([wauc(sc[ix], wrong4[ix], w[ix]) for ix in BS]),
                                   "mean_conf": round(float((sc * w).sum() / w.sum()), 4), "accuracy": round(float(((~wrong4) * w).sum() / w.sum()), 4), "bins": rows}
        scT = confsT[k]; wT = (prT != yT); eT, _ = ece(scT, (~wT).astype(float), np.ones(len(scT)))
        r["calibration_jurytest"][k] = {"ece": round(float(eT), 4), "auroc": round(float(wauc(scT, wT, np.ones(len(scT)))), 4), "mean_conf": round(float(scT.mean()), 4), "accuracy": round(float((~wT).mean()), 4)}
    # --- combiners fit on val: target = model grade != jury consensus ---
    pV, pP = soft(lpV, T_), soft(lpP, T_); XV, XP = feats(pV, prV, mV), feats(pP, prP, mP); tV = (prV != yV).astype(int)
    agreeV, agreeP = (prV == yV).astype(float), (prP == yP).astype(float)
    def fitlr(X, t):
        sc = StandardScaler().fit(X); m = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(X), t); return lambda Z: m.predict_proba(sc.transform(Z))[:, 1], dict(zip(["coef"], [[round(float(c), 3) for c in m.coef_[0]]]))
    lr_model, c1 = fitlr(XV, tV); lr_combo, c2 = fitlr(np.column_stack([XV, vfV, agreeV]), tV)
    r["lr_model_coef"] = c1["coef"]; r["lr_combo_coef"] = c2["coef"]; r["val_disagree_rate"] = round(float(tV.mean()), 4)
    pd_model = lr_model(XP); pd_combo = lr_combo(np.column_stack([XP, vfP, agreeP]))
    if LC and tag == "p1":   # secondary: val + low-agreement sections (jury_vf < 0.75) so the vf feature spans its range; those patients overlap training
        lpC, prC, pgC, mC = arr(LC); yC = np.array([ORD[x["label"]] for x in LC]); vfC = np.array([x["vf"] for x in LC]); pC = soft(lpC, T_)
        lr_vl, c3 = fitlr(np.column_stack([np.vstack([XV, feats(pC, prC, mC)]), np.r_[vfV, vfC], np.r_[agreeV, (prC == yC)]]), np.r_[tV, (prC != yC).astype(int)])
        r["lr_combo_vl_coef"] = c3["coef"]; r["lowconf_disagree_rate"] = round(float((prC != yC).mean()), 4); pd_vl = lr_vl(np.column_stack([XP, vfP, agreeP]))
    else: pd_vl = None
    # --- conformal on val ---
    r["conformal"] = {}
    sV = 1 - pV[np.arange(len(yV)), yV]; n = len(sV)
    p4T = soft(lp4, T_); pPT = pP; hum_i = np.array([ORD[h] for h in hum]); jury4 = np.array([ORD[lab[i]["label"]] for i in L])
    pTT = soft(lpT, T_)
    for a in (0.10, 0.05):
        qh = float(np.quantile(sV, min(1.0, math.ceil((n + 1) * (1 - a)) / n), method="higher"))
        S4 = (1 - p4T) <= qh; SP = (1 - pPT) <= qh; ST = (1 - pTT) <= qh; sz4 = S4.sum(1); szP = SP.sum(1)
        cov_h = S4[np.arange(len(L)), hum_i]; cov_j = S4[np.arange(len(L)), jury4]
        send = szP != 1; sendL = send[Li]; pred_ok = (pred4 == hum)
        r["conformal"][f"alpha={a}"] = {"qhat": round(qh, 4), "coverage_jurytest": round(float(ST[np.arange(len(yT)), yT].mean()), 4), "mean_size_jurytest": round(float(ST.sum(1).mean()), 3),
            "coverage_human_weighted": round(float((cov_h * w).sum() / w.sum()), 4), "coverage_human_ci": ci([(cov_h[ix] * w[ix]).sum() / w[ix].sum() for ix in BS]),
            "coverage_human_unweighted": round(float(cov_h.mean()), 4), "coverage_jury_weighted": round(float((cov_j * w).sum() / w.sum()), 4),
            "mean_size_weighted": round(float((sz4 * w).sum() / w.sum()), 3), "mean_size_unweighted": round(float(sz4.mean()), 3), "mean_size_pop": round(float(szP.mean()), 3),
            "size_hist_479": {str(k): int((sz4 == k).sum()) for k in range(7)},
            "rule_size_ne_1": {"sections_sent_pop": round(float(send.mean()), 4), "reports_sent_pop": round(len(set(prep[send])) / nrep, 4),
                               "acc_after_review_weighted": round(float(((sendL | pred_ok) * w).sum() / w.sum()), 4), "acc_after_review_ci": ci([((sendL[ix] | pred_ok[ix]) * w[ix]).sum() / w[ix].sum() for ix in BS]),
                               "errors_caught": int((sendL & ~pred_ok).sum()), "errors_total": int((~pred_ok).sum())}}
    # --- triage methods over the population (lower score = review first) ---
    gP = np.array([GRADES[i] if i >= 0 else "INVALID" for i in prP]); ppP = pick(pP, prP); entP = -(pP * np.log(np.clip(pP, 1e-12, 1))).sum(1); srt = np.sort(pP, 1)
    qh10 = r["conformal"]["alpha=0.1"]["qhat"]; szP10 = ((1 - pP) <= qh10).sum(1)
    M[f"{tag}_first_token_p_grade"] = (gP, pgP); M[f"{tag}_p_pred_T1"] = (gP, pick(soft(lpP, 1), prP)); M[f"{tag}_p_pred_temp"] = (gP, ppP)
    M[f"{tag}_neg_entropy"] = (gP, -entP); M[f"{tag}_margin"] = (gP, srt[:, -1] - srt[:, -2]); M[f"{tag}_lr_model_only"] = (gP, -pd_model)
    M[f"{tag}_conformal_size"] = (gP, -szP10 + ppP)
    M[f"{tag}_lr_combo_with_jury"] = (gP, -pd_combo); M[f"jury_{tag}_lr_combo"] = (jury_pop, -pd_combo)
    M[f"{tag}_disagree_first_then_vf"] = (gP, agreeP + vfP); M[f"jury_{tag}_disagree_first_then_vf"] = (jury_pop, agreeP + vfP)
    M[f"{tag}_disagree_first_then_p"] = (gP, agreeP + ppP); M[f"jury_{tag}_disagree_first_then_p"] = (jury_pop, agreeP + ppP)
    if pd_vl is not None: M[f"{tag}_lr_combo_val+lowconf"] = (gP, -pd_vl); M[f"jury_{tag}_lr_combo_val+lowconf"] = (jury_pop, -pd_vl)
    r["pop_model_jury_disagree_rate"] = round(float(1 - agreeP.mean()), 4)
    res["models"][tag] = r

curves = {m: curve(g, sc) for m, (g, sc) in M.items()}; fj, Zj = curves["jury_votefrac"]
res["triage"] = {}
for m, (fr, Z) in curves.items():
    d = {"acc_no_review": round(at(fr, Z, 0.0, ALL), 4)}
    for f in BUDS:
        d[f"acc@{int(f*100)}%"] = round(at(fr, Z, f, ALL), 4); d[f"acc@{int(f*100)}%_ci"] = ci([at(fr, Z, f, ix) for ix in BS])
        if m != "jury_votefrac": d[f"diff_vs_jury@{int(f*100)}%"] = round(at(fr, Z, f, ALL) - at(fj, Zj, f, ALL), 4); d[f"diff_vs_jury@{int(f*100)}%_ci"] = ci([at(fr, Z, f, ix) - at(fj, Zj, f, ix) for ix in BS])
    res["triage"][m] = d
res["curves"] = {m: {"frac_reports": [round(float(x), 4) for x in fr[::10]], "acc": [round(at(fr, Z, float(x), ALL), 4) for x in fr[::10]]} for m, (fr, Z) in curves.items()}
json.dump(res, open(O + "/results.json", "w"), indent=1)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, len(res["models"]), figsize=(4.6 * max(1, len(res["models"])), 4.2), squeeze=False)
for ax, (tag, r) in zip(axes[0], res["models"].items()):
    for k, col in (("first_token_p_grade (Task A)", "#9ca3af"), ("p_pred_T1", "#dc2626"), ("p_pred_temperature", "#2563eb"), ("p_pred_isotonic", "#16a34a")):
        rb = r["calibration_479"][k]["bins"]; ax.plot([x["mean_conf"] for x in rb], [x["accuracy"] for x in rb], "o-", color=col, label=f"{k} (ECE {r['calibration_479'][k]['ece']:.3f})")
    ax.plot([0, 1], [0, 1], "k:", lw=0.8); ax.set_xlabel("confidence (bin mean)"); ax.set_ylabel("accuracy vs human (weighted)"); ax.set_title(f"{tag}: reliability, 479 sections (T={r['temperature']:.2f})", fontsize=9); ax.legend(fontsize=6)
plt.tight_layout(); plt.savefig(O + "/reliability_calib.png", dpi=160); plt.close()
fig, ax = plt.subplots(figsize=(7, 4.5))
for m in ["jury_votefrac"] + [k for k in curves if k.endswith(("first_token_p_grade", "p_pred_temp", "lr_model_only", "lr_combo", "disagree_first_then_vf")) and not k.startswith("jury_d2")]:
    fr, Z = curves[m]; ax.plot(fr[::5], [at(fr, Z, float(x), ALL) for x in fr[::5]], label=m, lw=1.2)
ax.set_xlim(0, 0.5); ax.set_xlabel("fraction of reports opened"); ax.set_ylabel("population exact accuracy after review"); ax.legend(fontsize=6); ax.set_title("Triage curves (479 human-labelled sections, weighted)", fontsize=9)
plt.tight_layout(); plt.savefig(O + "/triage_calib.png", dpi=160); plt.close()
for tag, r in res["models"].items():
    print(tag, "T=", r["temperature"], "argmax==pred", r["argmax_eq_pred_val"], "mass", r["lp6_mass_val_median"], "lr_model coef", r["lr_model_coef"], "lr_combo coef", r["lr_combo_coef"])
    for k, v in r["calibration_479"].items(): print("  479", k, {x: v[x] for x in v if x != "bins"})
    for k, v in r["calibration_jurytest"].items(): print("  jurytest", k, v)
    for k, v in r["conformal"].items(): print("  conformal", k, v)
for m, d in res["triage"].items(): print(f"{m:36s}", {k: v for k, v in d.items() if not k.endswith("_ci")})

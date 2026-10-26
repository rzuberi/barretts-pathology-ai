"""Item 1 (CPU): per-class recall/precision, accuracy by jury vote-fraction band and management-tier agreement on the 479 human
sections, weighted (hand-label weights w) and unweighted, patient bootstrap (2,000, RandomState(0) as metrics.py).
Models: jury, encoder seeds 0-2 (explore/clf mbb32_s*), RL seed 0, MedGemma-27B zero-shot, plus EXTRA=tag,... from O/preds.
Out: O/perclass.json, O/perclass.md"""
import json, os
from pa_common import *
M = {"jury": None, "enc_s0": CLF + "/preds/mbb32_s0", "enc_s1": CLF + "/preds/mbb32_s1", "enc_s2": CLF + "/preds/mbb32_s2",
     "rl_s0": B + "/analysis/rl_grader/preds/grpo_sft__checkpoint-300", "medgemma27b": B + "/analysis/rl_grader/preds/medgemma27b_zs"}
for t in [x for x in os.environ.get("EXTRA", "").split(",") if x]: M[t] = O + "/preds/" + t
lab = {r["id"]: r for r in rows("test485")}; ids = sorted(lab); hum = np.array([lab[i]["human"] for i in ids]); w = np.array([lab[i]["w"] for i in ids])
pats = np.array([lab[i]["anon_id"] for i in ids]); vf = np.array([lab[i]["jury_vf"] for i in ids]); amb = np.array([lab[i]["ambiguous"] for i in ids])
band = np.where(vf >= 0.999, "unanimous", np.where(vf >= 0.75, "0.75-0.99", "<0.75"))
P = {}
for m, p in M.items():
    if m == "jury": P[m] = np.array([lab[i]["label"] for i in ids]); continue
    d = load_preds(p, "test485"); P[m] = np.array([d[i]["pred"] or "INVALID" for i in ids])
BS = patient_bootstrap(pats); ALL = np.arange(len(ids)); tier = np.vectorize(TIERS.get)
def wm(x, ix, weighted): ww = w[ix] if weighted else np.ones(len(ix)); return float((x[ix] * ww).sum() / ww.sum()) if ww.sum() else float("nan")
def recall(pr, g, ix, weighted): m = hum[ix] == g; return wm((pr == hum).astype(float), ix[m], weighted) if m.any() else float("nan")
def precision(pr, g, ix, weighted): m = pr[ix] == g; return wm((pr == hum).astype(float), ix[m], weighted) if m.any() else float("nan")
def boot(fn): v = [fn(ix) for ix in BS]; v = [x for x in v if x == x]; return ci(v) if v else None
R = {"n": len(ids), "n_patients": len(set(pats)), "class_counts_human": dict(Counter(hum)), "band_counts": dict(Counter(band)), "models": {}}
for m, pr in P.items():
    r = {"per_class": {}, "band": {}, "tier": {}}
    for g in GRADES:
        r["per_class"][g] = {"n_human": int((hum == g).sum()), "n_pred": int((pr == g).sum())}
        for wt, nm in ((False, "unw"), (True, "w")):
            r["per_class"][g][f"recall_{nm}"] = round(recall(pr, g, ALL, wt), 4); r["per_class"][g][f"recall_{nm}_ci"] = boot(lambda ix: recall(pr, g, ix, wt))
            r["per_class"][g][f"precision_{nm}"] = round(precision(pr, g, ALL, wt), 4); r["per_class"][g][f"precision_{nm}_ci"] = boot(lambda ix: precision(pr, g, ix, wt))
    ok = (pr == hum).astype(float)
    for b in ("unanimous", "0.75-0.99", "<0.75"):
        bi = np.where(band == b)[0]; r["band"][b] = {"n": len(bi)}
        for wt, nm in ((False, "unw"), (True, "w")):
            r["band"][b][f"exact_{nm}"] = round(wm(ok, bi, wt), 4); r["band"][b][f"exact_{nm}_ci"] = boot(lambda ix: wm(ok, ix[np.isin(ix, bi)], wt))
            if m != "jury":
                oj = (P["jury"] == hum).astype(float); r["band"][b][f"minus_jury_{nm}"] = {"point": round(wm(ok, bi, wt) - wm(oj, bi, wt), 4), "ci": boot(lambda ix: wm(ok, ix[np.isin(ix, bi)], wt) - wm(oj, ix[np.isin(ix, bi)], wt))}
    ai = np.where(amb == 1)[0]; r["ambiguous"] = {"n": len(ai), "exact_unw": round(wm(ok, ai, False), 4), "exact_w": round(wm(ok, ai, True), 4)}
    th, tp = tier(hum), np.array([TIERS.get(x, "INVALID") for x in pr]); okt = (th == tp).astype(float)
    r["tier"] = {"exact_unw": round(wm(okt, ALL, False), 4), "exact_w": round(wm(okt, ALL, True), 4), "exact_w_ci": boot(lambda ix: wm(okt, ix, True)),
                 "confusion_human_rows": {a: {b_: int(((th == a) & (tp == b_)).sum()) for b_ in ["non-dysplastic", "IND", "LGD", "HGD+cancer", "INVALID"]} for a in ["non-dysplastic", "IND", "LGD", "HGD+cancer"]}}
    TO = {"non-dysplastic": 0, "IND": 1, "LGD": 2, "HGD+cancer": 3, "INVALID": -1}; o = np.array([TO[x] for x in tp]); h_ = np.array([TO[x] for x in th])
    r["tier"]["over_w"] = round(wm((o > h_).astype(float), ALL, True), 4); r["tier"]["under_w"] = round(wm(((o < h_) & (o >= 0)).astype(float), ALL, True), 4)
    if m != "jury": r["tier"]["exact_w_minus_jury"] = {"point": round(wm(okt, ALL, True) - wm((tier(P["jury"]) == th).astype(float), ALL, True), 4), "ci": boot(lambda ix: wm(okt, ix, True) - wm((tier(P["jury"]) == th).astype(float), ix, True))}
    r["confusion_human_rows"] = {g: {p: int(((hum == g) & (pr == p)).sum()) for p in GRADES + ["INVALID"]} for g in GRADES}
    R["models"][m] = r
json.dump(R, open(O + "/perclass.json", "w"), indent=1); os.chmod(O + "/perclass.json", 0o600)
L = ["# Per-class, vote-fraction band and management tier on the 479 human sections", "", f"n = {len(ids)} sections, {len(set(pats))} patients; human class counts {R['class_counts_human']}; bands {R['band_counts']}", ""]
f = lambda x: "—" if x is None or x != x else f"{100*x:.1f}"
for wt in ("unw", "w"):
    L += [f"## Per-class recall ({'weighted' if wt=='w' else 'unweighted'}), % [95% CI]", "", "| model | " + " | ".join(f"{g} (n={int((hum==g).sum())})" for g in GRADES) + " |", "|---|" + "---|" * 6]
    for m, r in R["models"].items(): L.append(f"| {m} | " + " | ".join(f"{f(r['per_class'][g][f'recall_{wt}'])} {r['per_class'][g][f'recall_{wt}_ci'] and [round(100*x,1) for x in r['per_class'][g][f'recall_{wt}_ci']]}" for g in GRADES) + " |")
    L += ["", f"## Per-class precision ({'weighted' if wt=='w' else 'unweighted'}), %", "", "| model | " + " | ".join(GRADES) + " |", "|---|" + "---|" * 6]
    for m, r in R["models"].items(): L.append(f"| {m} | " + " | ".join(f"{f(r['per_class'][g][f'precision_{wt}'])} (n_pred {r['per_class'][g]['n_pred']})" for g in GRADES) + " |")
    L += ["", f"## Exact agreement by jury vote-fraction band ({'weighted' if wt=='w' else 'unweighted'}), % [CI]; Δ vs jury", "", "| model | unanimous | 0.75-0.99 | <0.75 |", "|---|---|---|---|"]
    for m, r in R["models"].items(): L.append(f"| {m} | " + " | ".join(f"{f(r['band'][b][f'exact_{wt}'])} {[round(100*x,1) for x in r['band'][b][f'exact_{wt}_ci']]}" + (f"; Δ {100*r['band'][b][f'minus_jury_{wt}']['point']:+.1f} {[round(100*x,1) for x in r['band'][b][f'minus_jury_{wt}']['ci']]}" if m != "jury" else "") for b in ("unanimous", "0.75-0.99", "<0.75")) + " |")
    L.append("")
L += ["## Management tier (non-dysplastic / IND / LGD / HGD+cancer)", "", "| model | exact unw | exact w [CI] | Δ vs jury (w) | over-tier w | under-tier w |", "|---|---|---|---|---|---|"]
for m, r in R["models"].items(): t = r["tier"]; L.append(f"| {m} | {f(t['exact_unw'])} | {f(t['exact_w'])} {[round(100*x,1) for x in t['exact_w_ci']]} | " + (f"{100*t['exact_w_minus_jury']['point']:+.1f} {[round(100*x,1) for x in t['exact_w_minus_jury']['ci']]}" if m != "jury" else "—") + f" | {f(t['over_w'])} | {f(t['under_w'])} |")
L += ["", "## Tier confusion (human rows x model columns: non-dysplastic, IND, LGD, HGD+cancer, INVALID)", ""]
for m, r in R["models"].items(): L.append(f"- {m}: " + "; ".join(f"{a}: {list(v.values())}" for a, v in r["tier"]["confusion_human_rows"].items()))
L += ["", "## Ambiguous-flagged sections (Rehan's flag)", ""] + [f"- {m}: n={r['ambiguous']['n']} exact unw {f(r['ambiguous']['exact_unw'])} w {f(r['ambiguous']['exact_w'])}" for m, r in R["models"].items()]
open(O + "/perclass.md", "w").write("\n".join(L) + "\n"); os.chmod(O + "/perclass.md", 0o600); print("\n".join(L))

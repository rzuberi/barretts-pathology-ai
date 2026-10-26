"""Error inheritance (CPU): on the 479 human sections, how many of a student's errors vs Rehan the jury also makes (same section;
same wrong label), P(student wrong | jury wrong) vs P(student wrong | jury right), and the (human, predicted) confusion-cell overlap.
Models: encoder seeds (explore mbb32_s0/1/2), full_s0, cross-fitted cf_oof_s0, masked SFT seeds, RL seed 0, MedGemma-27B; EXTRA=tag,...
Weighted versions use the hand-label weights w. Out: O/inherit.json"""
import json, os
from pa_common import *
M = {"enc_s0": CLF + "/preds/mbb32_s0", "enc_s1": CLF + "/preds/mbb32_s1", "enc_s2": CLF + "/preds/mbb32_s2", "full_s0": O + "/preds/full_s0", "cf_oof_s0": O + "/preds/cf_oof_s0",
     "sftm_s0": O + "/sft/preds/sftm_s0", "sftm_s1": O + "/sft/preds/sftm_s1", "sftm_s2": O + "/sft/preds/sftm_s2",
     "rl_s0": B + "/analysis/rl_grader/preds/grpo_sft__checkpoint-300", "medgemma27b": B + "/analysis/rl_grader/preds/medgemma27b_zs"}
for t in [x for x in os.environ.get("EXTRA", "").split(",") if x]: M[t] = O + "/preds/" + t
lab = {r["id"]: r for r in rows("test485")}; ids = sorted(lab); hum = np.array([lab[i]["human"] for i in ids]); jury = np.array([lab[i]["label"] for i in ids])
w = np.array([lab[i]["w"] for i in ids]); pats = np.array([lab[i]["anon_id"] for i in ids]); BS = patient_bootstrap(pats)
J = jury != hum
R = {"n": len(ids), "jury_errors": int(J.sum()), "jury_errors_w": round(float(w[J].sum() / w.sum()), 4), "models": {}}
for m, p in M.items():
    d = load_preds(p, "test485")
    if not d: print("missing", m); continue
    pr = np.array([d[i]["pred"] or "INVALID" for i in ids]); E = pr != hum
    shared = E & J; same = shared & (pr == jury)
    f = lambda a, b: float(a.sum() / b.sum()) if b.sum() else float("nan")
    fw = lambda a, b: float(w[a].sum() / w[b].sum()) if w[b].sum() else float("nan")
    r = {"errors": int(E.sum()), "shared_with_jury": int(shared.sum()), "same_wrong_label": int(same.sum()),
         "frac_errors_shared": round(f(shared, E), 4), "frac_errors_shared_ci": ci([f(shared[ix], E[ix]) for ix in BS]),
         "frac_errors_same_label": round(f(same, E), 4), "frac_errors_shared_w": round(fw(shared, E), 4),
         "p_wrong_given_jury_wrong": round(f(shared, J), 4), "p_wrong_given_jury_right": round(f(E & ~J, ~J), 4),
         "p_wrong_given_jury_wrong_w": round(fw(shared, J), 4), "p_wrong_given_jury_right_w": round(fw(E & ~J, ~J), 4),
         "jury_errors_fixed": int((J & ~E).sum()), "new_errors": int((E & ~J).sum()),
         "frac_jury_errors_fixed": round(f(J & ~E, J), 4), "frac_jury_errors_fixed_ci": ci([f((J & ~E)[ix], J[ix]) for ix in BS])}
    cells = {}
    for h in GRADES:
        for g in GRADES + ["INVALID"]:
            if h == g: continue
            ne = int(((hum == h) & (pr == g)).sum()); nj = int(((hum == h) & (jury == g)).sum()); nb = int(((hum == h) & (pr == g) & (jury == g)).sum())
            if ne or nj: cells[f"{h}->{g}"] = {"student": ne, "jury": nj, "both": nb}
    r["confusion_cells"] = cells; R["models"][m] = r
    print(f"{m:12s} errors {r['errors']:3d}  shared {r['shared_with_jury']:3d} ({100*r['frac_errors_shared']:.0f}% {[round(100*x) for x in r['frac_errors_shared_ci']]}, w {100*r['frac_errors_shared_w']:.0f}%)  same label {r['same_wrong_label']:3d} ({100*r['frac_errors_same_label']:.0f}%)  "
          f"P(wrong|jury wrong) {100*r['p_wrong_given_jury_wrong']:.0f}% vs P(wrong|jury right) {100*r['p_wrong_given_jury_right']:.1f}%  fixed {r['jury_errors_fixed']}/{R['jury_errors']} new {r['new_errors']}")
    top = sorted(cells.items(), key=lambda kv: -(kv[1]["student"] + kv[1]["jury"]))[:6]; print("   ", {k: (v["student"], v["jury"], v["both"]) for k, v in top})
json.dump(R, open(O + "/inherit.json", "w"), indent=1); os.chmod(O + "/inherit.json", 0o600)

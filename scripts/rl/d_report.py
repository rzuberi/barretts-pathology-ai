"""Follow-up D report: triage at 20% of reports (primary) for D1/D2 vs jury vote fraction (tie-aware, via p2_metrics), plus the new
scores' error-detection AUROC and ECE on the 479 sections. Out: rl_followups/D/{metrics_d1.json, metrics_d2.json, results.json}"""
import json, os, subprocess, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__)); from common import *
FU = B + "/analysis/rl_followups"; D = FU + "/D"; R = B + "/analysis/rl_grader"
sel = {v: json.load(open(f"{D}/selected_{v}.json"))["selected"] for v in ("d1", "d2")}; tags = {v: f"{v}__{sel[v]}" for v in sel}
out = {"selected": sel}
for v, t in tags.items():
    env = dict(os.environ, PDIR=D, PRIMARY_FRAC="0.20", ALPHA_TAGS=",".join(tags.values()), BEST=t, OUT=f"{D}/metrics_{v}.json")
    print(subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "p2_metrics.py")], env=env, capture_output=True, text=True).stdout[-1500:])
    out[v] = json.load(open(f"{D}/metrics_{v}.json"))
# Task-A style metrics for the new scores
sys.path.insert(0, os.path.dirname(__file__)); import importlib.util
spec = importlib.util.spec_from_file_location("ta", os.path.join(os.path.dirname(__file__), "task_a_calibration.py"))
lab = {json.loads(l)["id"]: json.loads(l) for l in open(R + "/data/test485.jsonl")}; ids = sorted(lab); w = np.array([lab[i]["w"] for i in ids]); hum = np.array([lab[i]["human"] for i in ids])
def wauc(score, wrong, ww):
    o = np.argsort(score, kind="stable"); s, p_, n_, w_ = score[o], ~wrong[o], wrong[o], ww[o]; tot = 0.0; cn = 0.0; i = 0
    while i < len(s):
        j = i
        while j < len(s) and s[j] == s[i]: j += 1
        wp = (w_[i:j] * p_[i:j]).sum(); wn = (w_[i:j] * n_[i:j]).sum(); tot += wp * (cn + 0.5 * wn); cn += wn; i = j
    return tot / (ww[~wrong].sum() * ww[wrong].sum())
def ece(score, correct, ww, nb=10):
    b = np.minimum((score * nb).astype(int), nb - 1); e = 0.0
    for k in range(nb):
        m = b == k
        if m.any(): e += ww[m].sum() / ww.sum() * abs((correct[m] * ww[m]).sum() / ww[m].sum() - (score[m] * ww[m]).sum() / ww[m].sum())
    return e
for v, t in tags.items():
    d = {json.loads(l)["id"]: json.loads(l) for l in open(f"{D}/preds/{t}__test485.jsonl")}
    g = np.array([d[i]["pred"] or "X" for i in ids])
    sc = np.array([(d[i].get("conf") if d[i].get("conf") is not None else 0.0) if v == "d2" else (0.0 if d[i]["pred"] == "UNSURE" else (d[i].get("p_grade") or 0)) for i in ids])
    ans = g != "UNSURE"; wrong = g != hum
    out[f"{v}_scores"] = {"abstained_labelled": int((~ans).sum()), "error_auroc_answered": round(float(wauc(sc[ans], wrong[ans], w[ans])), 4), "ece_answered": round(float(ece(sc[ans], (~wrong[ans]).astype(float), w[ans])), 4),
                          "accuracy_weighted": round(float(((~wrong) * w).sum() / w.sum()), 4)}
json.dump(out, open(D + "/results.json", "w"), indent=1); print({k: v for k, v in out.items() if k.endswith("_scores") or k == "selected"})

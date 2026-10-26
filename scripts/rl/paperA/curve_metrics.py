"""Learning curve / temporal split / encoder size (CPU). For TAGS (O/preds/TAG, or explore mbb32_s* from CLF/preds): weighted exact on
the 479 [patient bootstrap 2,000, RandomState(0)], unweighted, exact by report year (PSyy) on the 479; agreement with the jury on
test.jsonl overall and by year band (<= 2021 in-period, >= 2022 later; the 479 are all 2023-25); paired delta vs REF (default full_s0)
on the 479 and on test >= 2022. Training rows / patients / params / minutes from runs/TAG/log.json, seconds per report from the
timing file. Out: O/curve_metrics.json"""
import json, os
from pa_common import *
TAGS = os.environ["TAGS"].split(","); REF = os.environ.get("REF", "full_s0")
pre = lambda t: (CLF if t.startswith("mbb32") else O) + "/preds/" + t
year = lambda i: 2000 + int(i[2:4])
lab = {r["id"]: r for r in rows("test485")}; ids = sorted(lab); w = np.array([lab[i]["w"] for i in ids]); pats = np.array([lab[i]["anon_id"] for i in ids])
hum = np.array([lab[i]["human"] for i in ids]); yr = np.array([year(i) for i in ids])
te = rows("test"); tid = [r["id"] for r in te]; tlab = np.array([r["label"] for r in te]); tpat = np.array([r["anon_id"] for r in te]); tyr = np.array([year(i) for i in tid])
BS = patient_bootstrap(pats); TBS = patient_bootstrap(tpat)
def wacc(ok, ix): return float((ok[ix] * w[ix]).sum() / w[ix].sum())
def load(t):
    d = load_preds(pre(t), "test485"); e = load_preds(pre(t), "test")
    if not d or not e: return None
    return (np.array([d[i]["pred"] or "INVALID" for i in ids]) == hum).astype(float), (np.array([e[i]["pred"] or "INVALID" for i in tid]) == tlab).astype(float)
OK = {t: load(t) for t in TAGS + [REF]}; OK = {t: v for t, v in OK.items() if v is not None}
R = {"ref": REF, "n_test485": len(ids), "n_test": len(tid), "test_year_bands": {"<=2021": int((tyr <= 2021).sum()), ">=2022": int((tyr >= 2022).sum())}, "tags": {}}
for t in TAGS:
    if t not in OK: print("missing", t); continue
    ok, tok = OK[t]; r = {"exact_w": round(wacc(ok, np.arange(len(ids))), 4), "exact_w_ci": ci([wacc(ok, ix) for ix in BS]), "exact_unw": round(float(ok.mean()), 4),
                         "by_year_479": {int(y): {"n": int((yr == y).sum()), "exact_w": round(wacc(ok, np.where(yr == y)[0]), 4), "exact_unw": round(float(ok[yr == y].mean()), 4)} for y in sorted(set(yr))},
                         "test_agree": round(float(tok.mean()), 4), "test_agree_ci": ci([tok[ix].mean() for ix in TBS]),
                         "test_agree_<=2021": round(float(tok[tyr <= 2021].mean()), 4), "test_agree_>=2022": round(float(tok[tyr >= 2022].mean()), 4),
                         "test_agree_by_year": {int(y): round(float(tok[tyr == y].mean()), 4) for y in sorted(set(tyr))}}
    if t != REF and REF in OK:
        rok, rtok = OK[REF]; late = tyr >= 2022
        r[f"minus_{REF}_479"] = {"point": round(wacc(ok, np.arange(len(ids))) - wacc(rok, np.arange(len(ids))), 4), "ci": ci([wacc(ok, ix) - wacc(rok, ix) for ix in BS])}
        r[f"minus_{REF}_test>=2022"] = {"point": round(float(tok[late].mean() - rtok[late].mean()), 4), "ci": ci([tok[ix[late[ix]]].mean() - rtok[ix[late[ix]]].mean() for ix in TBS])}
    lg = f"{O}/runs/{t}/log.json"
    if os.path.exists(lg): r["train_log"] = {k: v for k, v in json.load(open(lg)).items() if k != "evals"}
    tm = f"{pre(t)}__test485.timing.json"
    if os.path.exists(tm): x = json.load(open(tm)); r["sec_per_report"] = round(x["seconds"] / x["reports"], 4)
    R["tags"][t] = r
json.dump(R, open(O + "/curve_metrics.json", "w"), indent=1); os.chmod(O + "/curve_metrics.json", 0o600)
for t, r in R["tags"].items():
    L = r.get("train_log", {}); d = r.get(f"minus_{REF}_479", {}); dt = r.get(f"minus_{REF}_test>=2022", {})
    print(f"{t:14s} n_train {L.get('n_train'):>6} pts {L.get('n_train_patients'):>5} | 479: {100*r['exact_w']:.1f} {[round(100*x,1) for x in r['exact_w_ci']]} unw {100*r['exact_unw']:.1f} "
          f"by-year {[round(100*v['exact_w'],1) for v in r['by_year_479'].values()]} | test {100*r['test_agree']:.1f} <=21 {100*r['test_agree_<=2021']:.1f} >=22 {100*r['test_agree_>=2022']:.1f}"
          + (f" | vs {REF}: 479 {100*d['point']:+.1f} {[round(100*x,1) for x in d['ci']]}, test>=22 {100*dt['point']:+.1f} {[round(100*x,1) for x in dt['ci']]}" if d else "")
          + f" | {L.get('train_minutes')} min, {r.get('sec_per_report')} s/report")

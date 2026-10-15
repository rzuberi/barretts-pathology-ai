"""MedGemma (27B text, 4B) as extra ERIN jurors: agreement with the 8-model jury label, with each juror,
with the 78 human adjudications (cancer yes/no), label distributions, parse-fail rate, and where MedGemma
disagrees with a confident jury."""
import glob, json, os
from collections import Counter
import pandas as pd
T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"; OUT = os.environ.get("OUTDIR", ".")
ORD = ["NDBE", "IND", "LGD", "HGD", "CANCER"]
fin = pd.read_csv(T + "/labeller/erin_labels_jury_final.csv", dtype=str).drop_duplicates("CaseName").set_index("CaseName")
adj = pd.read_csv(T + "/labeller/adjudications.csv", dtype=str).drop_duplicates("CaseName").set_index("CaseName")
jur = {}
for f in glob.glob(T + "/labeller/llm_full/llm_grades_*.csv"):
    mo = os.path.basename(f).replace("llm_grades_", "").rsplit("_shard", 1)[0]; d = pd.read_csv(f, dtype=str); jur.setdefault(mo, {}).update(dict(zip(d.CaseName, d.llm_grade)))
res = {}
for tag in ("medgemma27b", "medgemma"):
    fs = glob.glob(f"{T}/feasibility/runs/jf_{tag}_s*/output/llm_grades_*.csv")
    d = pd.concat([pd.read_csv(f, dtype=str) for f in fs]).drop_duplicates("CaseName").set_index("CaseName")
    g = d.llm_grade; ok = g.isin(ORD); r = {"rows": len(d), "parse_fail": int((g == "PARSE_FAIL").sum()), "na": int((g == "NA").sum()), "parse_rate": round(float(ok.mean()), 4), "label_dist": g[ok].value_counts().to_dict()}
    j = fin.reindex(d.index); el = ok & (j.label_status == "train_eligible")
    r["agreement_with_jury_train_eligible"] = {"n": int(el.sum()), "exact": round(float((g[el] == j.final_label[el]).mean()), 4),
        "binary_LGDplus": round(float(((g[el].isin(["LGD", "HGD", "CANCER"])) == (j.final_label[el].isin(["LGD", "HGD", "CANCER"]))).mean()), 4),
        "confusion_rows_jury_cols_medgemma": pd.crosstab(j.final_label[el], g[el]).reindex(index=ORD, columns=ORD, fill_value=0).values.tolist()}
    un = ok & (j.label_status == "unsure_held_out"); r["on_jury_unsure"] = {"n": int(un.sum()), "dist": g[un].value_counts().to_dict()}
    a = adj.reindex(d.index); ad = ok & a.decision.notna()
    truth_c = a.decision[ad].str.lower().eq("yes"); pred_c = g[ad].eq("CANCER")
    r["vs_adjudication_cancer"] = {"n": int(ad.sum()), "accuracy": round(float((truth_c == pred_c).mean()), 4), "sens": round(float(pred_c[truth_c].mean()), 4) if truth_c.any() else None, "spec": round(float((~pred_c[~truth_c]).mean()), 4) if (~truth_c).any() else None}
    r["pairwise_agreement_with_each_juror"] = {mo: round(float(pd.Series({c: v.get(c) for c in d.index[ok]}).eq(g[ok]).mean()), 4) for mo, v in sorted(jur.items())}
    # disagreements with a unanimous-ish jury (jury_frac >= 0.875 = 7/8)
    conf = el & (pd.to_numeric(j.jury_frac, errors="coerce") >= 0.875); dis = conf & (g != j.final_label)
    r["disagree_with_confident_jury"] = {"n_confident": int(conf.sum()), "n_disagree": int(dis.sum()), "pattern": Counter(f"jury={a_}|medgemma={b_}" for a_, b_ in zip(j.final_label[dis], g[dis])).most_common(8)}
    res[tag] = r; print(tag, json.dumps({k: v for k, v in r.items() if k != "pairwise_agreement_with_each_juror"}, indent=1)[:1500], flush=True)
json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=2, default=str)

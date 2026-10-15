"""Part 1: per-juror benchmark against Rehan's hand labels (485 oesophageal/GOJ sections).
READ-ONLY on sources. Outputs -> analysis/juror_benchmark/ (results.json, per_section.csv, SUMMARY.md)."""
import glob, json, os, re, sqlite3
from collections import Counter, defaultdict
import numpy as np, pandas as pd

T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"; H = "/mnt/scratche/slow/fmlab/zuberi01/hand_grading"
A = "/mnt/scratche/slow/fmlab/zuberi01/analysis/erin_handlabel"; OUT = "/mnt/scratche/slow/fmlab/zuberi01/analysis/juror_benchmark"
ERIN = "/mnt/scratche/fast/fmlab/datasets/imaging/ERIN/data/PathologyReport_AnonIds.csv"
os.makedirs(OUT, exist_ok=True); os.chmod(OUT, 0o700)
ORD = {"NORMAL_OTHER": -1, "NDBE": 0, "IND": 1, "LGD": 2, "HGD": 3, "CANCER": 4}; GR = list(ORD)
rng = np.random.RandomState(20261005)

# ---- human labels (first occurrence; repeats excluded) ----
db = sqlite3.connect(f"file:{H}/grading.db?mode=ro", uri=True, timeout=5)
L = pd.read_sql('select * from section_labels where grader="rehan"', db); L["ts"] = pd.to_datetime(L.ts)
key = pd.read_csv(A + "/key_DO_NOT_OPEN_UNTIL_DONE.csv", dtype=str).fillna(""); key["item_id"] = key.case_id + "#" + key.section
k = key.merge(L.rename(columns={"grades": "hgrades"})[["item_id", "hgrades", "ambiguous", "seconds", "ts", "session_no", "note"]], on="item_id", how="inner")
k["hset"] = k.hgrades.map(lambda s: set(json.loads(s))); k["hworst"] = k.hset.map(lambda g: max(g, key=lambda x: ORD[x]) if g else "NONE")
k["ambiguous"] = k.ambiguous.astype(int); k["weight"] = pd.to_numeric(k.weight, errors="coerce"); k["af"] = pd.to_numeric(k.agree_frac, errors="coerce")
print("sections with human label:", len(k), "| targets:", int((k.target == "target").sum()), "| random:", int((k.stratum == "random").sum()))

# ---- per-juror v3 outputs for these reports ----
reports = set(k.report_id)
juror_sec = defaultdict(dict)       # juror -> (report, section) -> grade set
juror_fail = defaultdict(set)       # juror -> reports with PARSE_FAIL
for f in glob.glob(T + "/feasibility/runs/sec3full_*/output/sections_v3_*.csv"):
    juror = os.path.basename(f).replace("sections_v3_", "").rsplit("_shard", 1)[0]
    d = pd.read_csv(f, dtype=str).fillna(""); d = d[d.CaseName.isin(reports)]
    for r in d.itertuples():
        if r.sections_json.startswith("PARSE_FAIL"): juror_fail[juror].add(r.CaseName); continue
        for s in json.loads(r.sections_json):
            sec = s["section"].upper().strip()[:2]; g = set(x for x in s["grades"] if x in ORD)
            if g: juror_sec[juror][(r.CaseName, sec)] = juror_sec[juror].get((r.CaseName, sec), set()) | g
JUR = sorted(juror_sec)
for j in JUR:
    k[f"set_{j}"] = [juror_sec[j].get((c, s)) for c, s in zip(k.report_id, k.section)]
    k[f"w_{j}"] = k[f"set_{j}"].map(lambda g: max(g, key=lambda x: ORD[x]) if g else None)
k["set_JURY"] = k.grades.map(lambda s: set(s.split("|")))
k["w_JURY"] = k.worst_grade

# ---- rule baseline: keyword ladder on the section text (final dx section -> micro section -> whole) ----
rep = pd.read_csv(ERIN, dtype=str, low_memory=False).fillna("").drop_duplicates("CaseName").set_index("CaseName")
MARK = re.compile(r"(?:(?<=^)|(?<=\s)|(?<=\n))([A-J])(?:\s*[-–]\s*([A-J]))?\s?(?:[.)\]:]|-(?=\s))\s*(?=\S)")
NEG = re.compile(r"\b(no|not|without|negative for|free of|absence of|no evidence of|insufficient for)\b", re.I)
LADDER = [("NORMAL_OTHER", [r"\bsquamous (?:mucosa|epithelium)\b", r"\bno (?:significant )?abnormalit", r"\bunremarkable\b", r"\bgastritis\b", r"\boesophagitis\b", r"\breactive\b"]),
          ("NDBE", [r"\bbarrett", r"\bintestinal metaplasia\b", r"\bnegative for dysplasia\b", r"\bno (?:evidence of )?dysplasia\b", r"\bcolumnar(?:[- ]lined)? (?:mucosa|epithelium)\b", r"\bglandular mucosa\b"]),
          ("IND", [r"\bindefinite for dysplasia\b", r"\bindefinite\b", r"\batypia[^.]{0,60}(?:indefinite|uncertain|cannot exclude)"]),
          ("LGD", [r"\blow[- ]grade (?:glandular |squamous )?dysplasia\b", r"\bLGD\b"]),
          ("HGD", [r"\bhigh[- ]grade (?:glandular |squamous )?dysplasia\b", r"\bHGD\b"]),
          ("CANCER", [r"\badenocarcinoma\b", r"\bcarcinoma\b", r"\bmalignan(?:t|cy)\b", r"\bintramucosal (?:adeno)?carcinoma\b"])]
def ladder(text):
    best = None
    for sent in re.split(r"(?<=[.;:\n])\s+", str(text)):
        for li, (g, pats) in enumerate(LADDER):
            for p in pats:
                for m in re.finditer(p, sent, re.I):
                    if any(m.start() - n.end() <= 130 for n in NEG.finditer(sent[:m.start()])): continue
                    if best is None or li > best: best = li
    return GR[best] if best is not None else None
def split_sections(text):
    ms = list(MARK.finditer(text)); out = {}
    for i, m in enumerate(ms):
        end = ms[i + 1].start() if i + 1 < len(ms) else len(text); seg = text[m.end():end]
        a, b = m.group(1), m.group(2) or m.group(1)
        for L_ in [chr(c) for c in range(ord(a), ord(b) + 1)]: out.setdefault(L_, []).append(seg)
    return {kk: " ".join(v) for kk, v in out.items()}
def rule(r):
    R = rep.loc[r.report_id]; fs, ms_ = split_sections(R.FinalDiagnosis_redacted), split_sections(R.MicroscopicDescription_redacted)
    txt = (fs.get(r.section, "") + " " + ms_.get(r.section, "")).strip() or (R.FinalDiagnosis_redacted + " " + R.MicroscopicDescription_redacted)
    return ladder(txt)
k["w_RULE"] = k.apply(rule, axis=1); k["set_RULE"] = k.w_RULE.map(lambda g: {g} if g else None)

# ---- metrics ----
def qwk(a, b):
    a = np.array([ORD.get(x, -1) for x in a]) + 1; b = np.array([ORD.get(x, -1) for x in b]) + 1; n = 6
    O = np.zeros((n, n))
    for i, j in zip(a, b): O[i, j] += 1
    W = np.array([[(i - j) ** 2 / (n - 1) ** 2 for j in range(n)] for i in range(n)]); E = np.outer(O.sum(1), O.sum(0)) / O.sum()
    return float(1 - (W * O).sum() / (W * E).sum()) if (W * E).sum() else float("nan")
def metrics(d, model, w=None):
    pred = d[f"w_{model}"]; sets = d[f"set_{model}"]; ok = pred.notna()
    dd = d[ok]; pred = pred[ok]; sets = sets[ok]
    if not len(dd): return {"n": 0, "coverage": 0.0}
    hv = dd.hworst.map(lambda g: ORD.get(g, -1)); pv = pred.map(ORD)
    av = (lambda x: float(np.average(x, weights=w[ok])) if w is not None else float(np.mean(x)))
    return {"n": int(len(dd)), "coverage": round(float(ok.mean()), 3), "exact": round(av(pred == dd.hworst), 3),
            "human_in_set": round(av([h in s for h, s in zip(dd.hworst, sets)]), 3),
            "LGDplus": round(av((hv >= 2) == (pv >= 2)), 3), "qwk": round(qwk(dd.hworst, pred), 3),
            "over": round(av(pv > hv), 3), "under": round(av(pv < hv), 3)}
MODELS = JUR + ["JURY", "RULE"]
# top-3 on a held-out half split by patient (random slice + targets kept intact per patient)
rep["anon"] = rep.anon_id; k["anon"] = k.report_id.map(rep.anon)
pats = k.anon.unique(); rng.shuffle(pats); selp = set(pats[: len(pats) // 2])
S, Tt = k[k.anon.isin(selp)], k[~k.anon.isin(selp)]
acc_S = {j: metrics(S, j)["exact"] for j in JUR}; top3 = sorted(acc_S, key=acc_S.get, reverse=True)[:3]
def maj(d, cols):
    out = []
    for _, r in d.iterrows():
        c = Counter(w for w in (r[f"w_{j}"] for j in cols) if w)
        out.append(max(c.items(), key=lambda kv: (kv[1], -ORD[kv[0]]))[0] if c else None)
    return pd.Series(out, index=d.index)
k["w_TOP3"] = maj(k, top3); k["set_TOP3"] = [set().union(*[s for s in (r[f"set_{j}"] for j in top3) if s]) or None for _, r in k.iterrows()]
S, Tt = k[k.anon.isin(selp)], k[~k.anon.isin(selp)]
res = {"n_sections": int(len(k)), "jurors": JUR, "top3_selected_on_half": top3, "top3_selection_half_accuracy": acc_S,
       "split": {"n_select": int(len(S)), "n_report": int(len(Tt)), "seed": 20261005, "by": "patient"}}
subsets = {"random_50": k[k.stratum == "random"], "targets_reweighted": k[k.target == "target"], "all_485": k,
           "non_ambiguous": k[k.ambiguous == 0], "ambiguous": k[k.ambiguous == 1], "report_half_only": Tt}
for name, d in subsets.items():
    w = d.weight.values if name == "targets_reweighted" else None
    res[name] = {m: metrics(d, m, w) for m in MODELS + ["TOP3"]}
# parse / NA failure per juror on these 143 reports and 485 sections
res["juror_failure"] = {j: {"report_parse_fail": len(juror_fail[j] & reports), "section_missing": int(k[f"set_{j}"].isna().sum()),
                            "section_missing_frac": round(float(k[f"set_{j}"].isna().mean()), 3)} for j in JUR}
# agreement vs vote fraction
k["afbin"] = pd.cut(k.af, [0, 0.5, 0.75, 0.999, 1.0], labels=["<=0.5", "0.5-0.75", "0.75-<1", "1.0"])
res["jury_by_vote_fraction"] = {str(b): metrics(g, "JURY") for b, g in k.groupby("afbin", observed=True)}
# time per report (sum of section seconds, outliers >600 s capped; and wall time first->last section)
sec_time = L.copy(); sec_time["case"] = sec_time.item_id.str.split("#").str[0]; sec_time["sec_c"] = sec_time.seconds.clip(upper=600)
per_rep = sec_time.groupby("case").agg(n_sections=("item_id", "size"), sum_seconds=("sec_c", "sum"), first=("ts", "min"), last=("ts", "max"), last_secs=("seconds", "last"))
per_rep["wall_seconds"] = ((per_rep["last"] - per_rep["first"]).dt.total_seconds() + per_rep.last_secs).clip(upper=1800)
res["time_per_report"] = {"reports": int(len(per_rep)), "median_sum_seconds": float(per_rep.sum_seconds.median()), "mean_sum_seconds": round(float(per_rep.sum_seconds.mean()), 1),
                          "p90_sum_seconds": float(per_rep.sum_seconds.quantile(.9)), "median_wall_seconds": float(per_rep.wall_seconds.median()),
                          "median_seconds_per_section": float(L.seconds.median()), "by_n_sections": per_rep.groupby("n_sections").sum_seconds.median().round(0).to_dict()}
json.dump(res, open(OUT + "/results.json", "w"), indent=1, default=str)
k.drop(columns=[c for c in k.columns if c.startswith("set_")]).to_csv(OUT + "/per_section.csv", index=False)
for f in os.listdir(OUT): os.chmod(os.path.join(OUT, f), 0o600)

# ---- SUMMARY.md ----
def table(name):
    rows = ["| model | n | cov | exact | human∈set | LGD+ | QWK | over | under |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    order = sorted(MODELS + ["TOP3"], key=lambda m: -res[name][m].get("exact", 0))
    for m in order:
        r = res[name][m]
        if r.get("n", 0) == 0: continue
        rows.append(f"| {m} | {r['n']} | {r['coverage']:.2f} | {r['exact']:.3f} | {r['human_in_set']:.3f} | {r['LGDplus']:.3f} | {r['qwk']:.3f} | {r['over']:.3f} | {r['under']:.3f} |")
    return "\n".join(rows)
md = [f"# Juror benchmark vs Rehan's labels ({len(k)} oesophageal/GOJ sections, 2026-10-05)\n",
      f"Top-3 by exact agreement on a patient-split selection half (n={len(S)}): **{', '.join(top3)}** → reported on the other half (n={len(Tt)}) and, with that caveat, on every subset.",
      "RULE = keyword ladder on the section's own text (final-diagnosis section, then microscopic section, else whole report). JURY = v3 ten-juror consensus. `cov` = fraction of sections for which the model produced a grade.\n"]
for name, title in [("random_50", "Random slice (unbiased)"), ("targets_reweighted", "Targets, reweighted to population"), ("non_ambiguous", "Non-ambiguous sections"),
                    ("ambiguous", "Ambiguous sections"), ("report_half_only", "Held-out report half (fair for TOP3)"), ("all_485", "All labelled sections")]:
    md += [f"## {title} (n={subsets[name].shape[0]})", table(name), ""]
md += ["## Juror failure on these reports", "| juror | report parse-fail | sections with no output | frac |", "|---|---:|---:|---:|"]
md += [f"| {j} | {v['report_parse_fail']} | {v['section_missing']} | {v['section_missing_frac']:.3f} |" for j, v in res["juror_failure"].items()]
md += ["", "## Jury agreement vs vote fraction", "| vote fraction | n | exact | human∈set | LGD+ | QWK | over | under |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
md += [f"| {b} | {r['n']} | {r['exact']:.3f} | {r['human_in_set']:.3f} | {r['LGDplus']:.3f} | {r['qwk']:.3f} | {r['over']:.3f} | {r['under']:.3f} |" for b, r in res["jury_by_vote_fraction"].items()]
t = res["time_per_report"]
md += ["", "## Time", f"Per section: median {t['median_seconds_per_section']:.0f} s. Per report ({t['reports']} reports incl. repeats): median {t['median_sum_seconds']:.0f} s summed over its sections (mean {t['mean_sum_seconds']:.0f}, p90 {t['p90_sum_seconds']:.0f}); median wall time first→last section {t['median_wall_seconds']:.0f} s. Median by sections per report: {t['by_n_sections']}."]
open(OUT + "/SUMMARY.md", "w").write("\n".join(md)); os.chmod(OUT + "/SUMMARY.md", 0o600)
print("\n".join(md))

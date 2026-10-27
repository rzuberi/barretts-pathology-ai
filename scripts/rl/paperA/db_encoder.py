"""External temporal test (GPU, minutes): the frozen Paper A encoders (runs/TAG/model.pt, trained on ERIN jury labels only) on the
pre-Oct-2014 Barrett's-DB reports vs the database's confirmed grade code (highestgradedysconf, mapped as in
scripts/numbers/num_db_confirmed_grade_anchor.py: 2 NDBE, 3 IND, 4 LGD, 5 HGD, 6/8 CANCER; other codes excluded), next to the
whole-report jury on the same reports (jury_full_* majority, >= 4 votes, the 24 Sep anchor). READ-ONLY on sources.
Unit: report, worst grade. Encoder input = DB headings re-laid out in the ERIN layout (swg_encoder.mapped); reports whose text numbers
>= 2 specimens ("1." / "Part 2") are re-lettered (swg_encoder.lettered) and the encoder is asked for each letter; report grade = worst
letter. Otherwise '[SECTION WHOLE]'. NORMAL_OTHER folded into NDBE.
Patient exclusion: DB accession (specimennumber, PSyy-nnnnn) matched to ERIN report_id; a DB participant is dropped when any of their
DB reports is an ERIN train/val report or belongs to an ERIN train/val anon_id. Lower bound: ERIN reports absent from the DB cannot be linked.
Env: TAGS (default full_s0,full_s1,full_s2; tags without model.pt skipped), N (smoke cap on reports), OUTSUB (output subfolder).
Out: O/external_db[/OUTSUB]/{results.json, grades.csv}. Prints aggregates only."""
import json, os, re, sys
import numpy as np, pandas as pd, torch, torch.nn.functional as F
os.environ["HF_HOME"] = "/mnt/scratche/slow/fmlab/zuberi01/hf_cache"; os.environ["HF_HUB_OFFLINE"] = "1"
from pa_common import *
# text helpers copied from swg_encoder.py (importing it would run the SWG scoring)
def body(t): return re.sub(r"^SPECIMEN OF INTEREST.*?\n\n", "", t, flags=re.S)
HEAD = ["CLINICAL DETAILS", "NATURE OF SPECIMEN", "MACROSCOPIC DESCRIPTION", "MICROSCOPY FINDINGS", "MICROSCOPY COMMENT"]
HRX = re.compile(r"(?m)^\s*(" + "|".join(HEAD) + r")\s*:?\s*$")
def split_heads(t):
    parts = {}; cur = "_pre"; parts[cur] = []
    for line in t.split("\n"):
        m = HRX.match(line)
        if m: cur = m.group(1); parts.setdefault(cur, []); continue
        parts.setdefault(cur, []).append(line)
    return {k: "\n".join(v).strip() for k, v in parts.items()}
def mapped(t):
    p = split_heads(t)
    if not any(h in p for h in HEAD[1:]): return "GROSS DESCRIPTION:\n\n\nFINAL DIAGNOSIS:\n" + t.strip() + "\n\nMICROSCOPIC DESCRIPTION:\n"
    gross = "\n".join(x for x in [p.get("CLINICAL DETAILS", ""), p.get("NATURE OF SPECIMEN", ""), p.get("MACROSCOPIC DESCRIPTION", "")] if x)
    return f"GROSS DESCRIPTION:\n{gross}\n\nFINAL DIAGNOSIS:\n{p.get('MICROSCOPY COMMENT', '')}\n\nMICROSCOPIC DESCRIPTION:\n{p.get('MICROSCOPY FINDINGS', '')}{(chr(10) + p['_pre']) if p.get('_pre') else ''}"
NUMMARK = re.compile(r"(?m)(^[ \t]*\(?)(\d{1,2})(\s?[.):\]]\s*)(?=\S)"); PARTREF = re.compile(r"(?i)\b(part|specimen|pot|biopsy)(s?\s*)(\d{1,2})\b")
L = lambda n: chr(64 + int(n)) if 1 <= int(n) <= 10 else str(n)
def lettered(t): return PARTREF.sub(lambda m: m.group(1) + m.group(2) + L(m.group(3)), NUMMARK.sub(lambda m: m.group(1) + L(m.group(2)) + m.group(3), t))
sys.path.insert(0, T + "/scripts/rl/explore/clf"); from clf_common import text_for
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import glob
from sklearn.metrics import cohen_kappa_score
TAGS = [t for t in os.environ.get("TAGS", "full_s0,full_s1,full_s2").split(",") if t]; N = int(os.environ.get("N", 0))
S = mkdir(O + "/external_db" + (("/" + os.environ["OUTSUB"]) if os.environ.get("OUTSUB") else "")); EXP = B + "/barretts_db_export"
LK = B + "/phd/barretts_retraining/barretts_training/data/barretts_scrape_20260220_172725/query_dysplasia_types.csv"
GR = ["NDBE", "IND", "LGD", "HGD", "CANCER"]; O5 = {g: i for i, g in enumerate(GR)}; MAP = {2: "NDBE", 3: "IND", 4: "LGD", 5: "HGD", 6: "CANCER", 8: "CANCER"}
fold = lambda g: "NDBE" if g == "NORMAL_OTHER" else g
# ---- database reports, confirmed code, jury (identical to the anchor script) ----
pt = pd.read_csv(EXP + "/pathology_text_normalised_full.csv", dtype=str); pt["report_id"] = pt.pathology_text_id.astype(str)
pt["conf"] = pd.to_numeric(pt.highestgradedysconf, errors="coerce"); pt["dt"] = pd.to_datetime(pt.receiveddatetime, errors="coerce")
votes = {}
for f in glob.glob(T + "/feasibility/runs/jury_full_*/output/llm_grades_*.csv"):
    d = pd.read_csv(f, dtype=str, on_bad_lines="skip"); d = d[d.llm_grade.isin(ORD)]
    for c, g in zip(d.CaseName, d.llm_grade): votes.setdefault(str(c), []).append(g)
# majority of the returning jurors; ties broken towards the worst grade (the anchor script broke ties by set order, which varies per process)
pt["jury"] = pt.report_id.map({c: max(sorted(set(v), key=lambda g: ORD[g]), key=v.count) for c, v in votes.items() if len(v) >= 4})
ties = {c for c, v in votes.items() if len(v) >= 4 and sorted((v.count(g) for g in set(v)), reverse=True)[:2].count(max(v.count(g) for g in set(v))) == 2}
pt["record"] = pt.conf.map(MAP)
anchor = json.load(open(T + "/results/numbers/db_confirmed_grade_anchor.json"))["confirmed_code_core_barretts"]["jury_all"]
rep = pt[pt.record.notna() & pt.jury.notna()]; R = {"step1_reproduce_jury_anchor": {"n": int(len(rep)), "exact": round(float((rep.jury == rep.record).mean()), 4),
                                                   "stored": {"n": anchor["n"], "exact": anchor["exact"], "qwk": anchor["qwk"]}}}
R["step1_reproduce_jury_anchor"]["tied_majority_reports"] = int(rep.report_id.isin(ties).sum())
assert R["step1_reproduce_jury_anchor"]["n"] == anchor["n"], R["step1_reproduce_jury_anchor"]
# ---- ERIN train/val patient exclusion via accession ----
def norm_acc(s):
    m = re.search(r"([A-Za-z]{2})\s*(\d{2})[.\-\s]?(\d{4,6})", str(s)); return f"{m.group(1).upper()}{m.group(2)}-{m.group(3)}" if m else None
pt["acc"] = pt.specimennumber.map(norm_acc)
erin = {}
for s in ["train", "val", "test", "test485", "pop"]:
    for r in rows(s): erin.setdefault(r["report_id"], {"anon": r["anon_id"], "sets": set()})["sets"].add(s)
tv_rep = {k for k, v in erin.items() if v["sets"] & {"train", "val"}}; tv_anon = {v["anon"] for k, v in erin.items() if k in tv_rep}
pt["anon"] = pt.acc.map(lambda a: erin[a]["anon"] if a in erin else None)
excl = set(pt[pt.acc.isin(tv_rep) | pt.anon.isin(tv_anon)].participant_id)
pre = pt[(pt.dt < "2014-10-01") & pt.reporttext.notna() & pt.record.notna()].copy(); pre["year"] = pre.dt.dt.year
pre["excluded"] = pre.participant_id.isin(excl); ev = pre[~pre.excluded].copy()
R["exclusion"] = {"db_reports_linked_to_erin_any_split": int(pt.acc.isin(erin).sum()), "db_reports_linked_to_erin_train_val": int(pt.acc.isin(tv_rep).sum()),
                  "participants_excluded": len(excl), "pre2014_core_reports": int(len(pre)), "pre2014_core_patients": int(pre.participant_id.nunique()),
                  "dropped_reports": int(pre.excluded.sum()), "dropped_patients": int(pre[pre.excluded].participant_id.nunique()),
                  "kept_reports": int(len(ev)), "kept_patients": int(ev.participant_id.nunique()), "kept_with_jury": int(ev.jury.notna().sum()),
                  "record_dist_kept": ev.record.value_counts().to_dict(), "note": "lower bound: ERIN train/val reports absent from the DB export cannot be linked"}
if N: ev = ev.head(N).copy()
# ---- encoder items: whole_mapped + lettered parts ----
def parts_of(t):
    ns = {int(m.group(2)) for m in NUMMARK.finditer(t)}; ns = sorted(n for n in ns if 1 <= n <= 10)
    return ns if len(ns) >= 2 and ns[0] == 1 and ns[1] == 2 else []
items = []; ev["n_parts"] = 0
for r in ev.itertuples():
    t = mapped(body(str(r.reporttext).replace("\r", "\n"))); items.append((f"{r.report_id}|whole", "WHOLE", t)); ps = parts_of(t)
    if ps:
        lt = lettered(t); ev.loc[r.Index, "n_parts"] = len(ps)
        for n in ps: items.append((f"{r.report_id}|part{n}", L(n), lt))
print(f"{len(ev)} reports, {len(items)} encoder items, {int((ev.n_parts > 0).sum())} multi-part reports", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL); enc = tok([text_for({"section": s, "report": t}) for _, s, t in items], truncation=True, max_length=MAXLEN, add_special_tokens=True)["input_ids"]; pad = tok.pad_token_id
def score(tag):
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=6, torch_dtype=torch.float32, attn_implementation="sdpa")
    model.load_state_dict(torch.load(f"{O}/runs/{tag}/model.pt", map_location="cpu")); model = model.cuda().eval(); pred = {}
    with torch.no_grad():
        idx = sorted(range(len(enc)), key=lambda i: len(enc[i]))
        for i in range(0, len(idx), 32):
            b = idx[i:i + 32]; Ln = max(len(enc[j]) for j in b); ids = torch.full((len(b), Ln), pad, dtype=torch.long); am = torch.zeros((len(b), Ln), dtype=torch.long)
            for k, j in enumerate(b): ids[k, :len(enc[j])] = torch.tensor(enc[j]); am[k, :len(enc[j])] = 1
            with torch.autocast("cuda", dtype=torch.bfloat16): pr = F.softmax(model(input_ids=ids.cuda(), attention_mask=am.cuda()).logits.float(), -1).cpu().numpy()
            for k, j in enumerate(b): pred[items[j][0]] = GRADES[int(pr[k].argmax())]
    del model; torch.cuda.empty_cache(); return pred
tags = [t for t in TAGS if os.path.exists(f"{O}/runs/{t}/model.pt")]; R["tags_scored"] = tags; R["tags_missing_weights"] = [t for t in TAGS if t not in tags]
for tag in tags:
    pred = score(tag); ev[f"{tag}_whole"] = ev.report_id.map(lambda r: fold(pred[f"{r}|whole"]))
    ev[f"{tag}_parts"] = [worst([fold(pred[f"{r.report_id}|part{n}"]) for n in range(1, 11) if f"{r.report_id}|part{n}" in pred]) if r.n_parts else None for r in ev.itertuples()]
    ev[tag] = ev[f"{tag}_parts"].where(ev[f"{tag}_parts"].notna(), ev[f"{tag}_whole"])   # worst lettered section, else whole report
    print(tag, "scored", flush=True)
# ---- metrics ----
TIER = {"NDBE": 0, "IND": 1, "LGD": 2, "HGD": 3, "CANCER": 3}
def qwk(p, t): return float(cohen_kappa_score([O5[x] for x in t], [O5[x] for x in p], weights="quadratic")) if len(set(t)) > 1 else float("nan")
def metrics(d, col, bs):
    p, t = d[col].values, d.record.values; ex = (p == t).astype(float); tier = (d[col].map(TIER).values == d.record.map(TIER).values).astype(float)
    rec = {g: (p[t == g] == g).astype(float) for g in GR}
    m = {"n": int(len(d)), "patients": int(d.participant_id.nunique()), "exact": round(float(ex.mean()), 4), "exact_ci": ci([ex[ix].mean() for ix in bs]), "qwk": round(qwk(p, t), 4),
         "tier_exact": round(float(tier.mean()), 4), "tier_ci": ci([tier[ix].mean() for ix in bs]), "LGDplus_sensitivity": round(float((d[col].map(O5) >= 2)[d.record.map(O5) >= 2].mean()), 4),
         "over": round(float((d[col].map(O5) > d.record.map(O5)).mean()), 4), "under": round(float((d[col].map(O5) < d.record.map(O5)).mean()), 4),
         "recall": {g: [int(len(rec[g])), round(float(rec[g].mean()), 4)] for g in GR},
         "recall_ci": {g: ci([(p[ix][t[ix] == g] == g).mean() for ix in bs]) for g in ("LGD", "HGD")},
         "confusion_truth_rows_pred_cols": pd.crosstab(d.record, d[col]).reindex(index=GR, columns=GR, fill_value=0).values.tolist()}
    return m
def paired(d, a, b, bs):
    x = (d[a].values == d.record.values).astype(float) - (d[b].values == d.record.values).astype(float); return {"n": int(len(d)), "delta": round(float(x.mean()), 4), "ci": ci([x[ix].mean() for ix in bs])}
cols = {t: t for t in tags}; cols.update({f"{t}_whole": f"{t}_whole" for t in tags}); cols["jury"] = "jury"
R["all_kept"] = {}; bs_all = patient_bootstrap(ev.participant_id.values)
for nm, c in cols.items():
    if nm != "jury": R["all_kept"][nm] = metrics(ev, c, bs_all)
wj = ev[ev.jury.notna()]; bs_j = patient_bootstrap(wj.participant_id.values); R["jury_common"] = {nm: metrics(wj, c, bs_j) for nm, c in cols.items()}
R["paired_minus_jury"] = {t: paired(wj, t, "jury", bs_j) for t in tags}; R["paired_parts_minus_whole"] = {t: paired(ev, t, f"{t}_whole", bs_all) for t in tags}
ex_rows = pre[pre.excluded]; R["dropped_patients_jury"] = {"n": int(ex_rows.jury.notna().sum()), "exact": round(float((ex_rows.jury == ex_rows.record)[ex_rows.jury.notna()].mean()), 4)} if len(ex_rows) else {}
R["by_year"] = {}
for y, g in ev.groupby("year"):
    gj = g[g.jury.notna()]; R["by_year"][int(y)] = {"n": int(len(g)), "patients": int(g.participant_id.nunique()), "n_jury": int(len(gj)),
        **{t: round(float((g[t] == g.record).mean()), 4) for t in tags}, "jury": round(float((gj.jury == gj.record).mean()), 4) if len(gj) else None}
R["multi_part_reports"] = {"n": int((ev.n_parts > 0).sum()), "exact_parts_vs_whole": {t: [round(float((ev[ev.n_parts > 0][t] == ev[ev.n_parts > 0].record).mean()), 4),
                                                                                       round(float((ev[ev.n_parts > 0][f"{t}_whole"] == ev[ev.n_parts > 0].record).mean()), 4)] for t in tags}}
json.dump(R, open(S + "/results.json", "w"), indent=1, default=lambda o: None); os.chmod(S + "/results.json", 0o600)
ev[["report_id", "participant_id", "year", "conf", "record", "jury", "n_parts"] + [c for t in tags for c in (t, f"{t}_whole", f"{t}_parts")]].to_csv(S + "/grades.csv", index=False); os.chmod(S + "/grades.csv", 0o600)
def row(nm, m): return f"{nm:16s} n={m['n']:5d} exact {100*m['exact']:.1f} {[round(100*x,1) for x in m['exact_ci']]} QWK {m['qwk']:.3f} tier {100*m['tier_exact']:.1f} LGD rec {100*m['recall']['LGD'][1]:.1f} HGD rec {100*m['recall']['HGD'][1]:.1f} over {100*m['over']:.1f} under {100*m['under']:.1f}"
print(json.dumps({k: R[k] for k in ("step1_reproduce_jury_anchor", "exclusion", "tags_scored", "tags_missing_weights", "paired_minus_jury", "paired_parts_minus_whole", "multi_part_reports", "dropped_patients_jury")}))
for blk in ("all_kept", "jury_common"):
    print("==", blk)
    for nm, m in R[blk].items(): print(row(nm, m))
print("== by year"); [print(y, v) for y, v in R["by_year"].items()]
print("DB DONE")

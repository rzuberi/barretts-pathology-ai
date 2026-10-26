"""Item 5 (GPU, minutes): the frozen Paper A encoder (full_s0 weights, trained on ERIN jury labels only) on the Barrett's-DB reports
behind the SWG sWGS blocks (analysis/erin_swg_linkage common subset, n = 427 report x block pairs), vs the SWG clinical-record block grade,
next to the 8-juror specimen-scoped jury (jury_majority_8) and the whole-report jury on the same rows. READ-ONLY on sources.
Alignment (the judge's caveat): the record grade is per BLOCK; a report-level worst grade is >= it by construction. So three encoder inputs:
  whole_raw     '[SECTION WHOLE]' + raw DB text                      (report-level worst)
  whole_mapped  DB headings re-laid out in the ERIN GROSS / FINAL DIAGNOSIS / MICROSCOPIC layout the encoder was trained on (report-level worst)
  part          for pairs whose SWG block digit matches a numbered part of a multi-part report: parts renumbered as lettered sections
                (1. -> A., 'part 2' -> 'part B') and the encoder asked for that letter (block-level, the only like-for-like comparison)
  any_part      record grade is among the encoder's per-part grades (multi-part reports)
NORMAL_OTHER (no Barrett's) is folded into NDBE for the 5-grade record ladder and counted. Out: O/swg/{results.json, grades.csv, disagreements.csv}"""
import json, os, re, sys
import pandas as pd, torch, torch.nn.functional as F
os.environ["HF_HOME"] = "/mnt/scratche/slow/fmlab/zuberi01/hf_cache"; os.environ["HF_HUB_OFFLINE"] = "1"
from pa_common import *
sys.path.insert(0, T + "/scripts/rl/explore/clf"); from clf_common import text_for
from transformers import AutoTokenizer, AutoModelForSequenceClassification
S = mkdir(O + "/swg"); LK = B + "/analysis/erin_swg_linkage"; EXP = B + "/barretts_db_export"
cm = pd.read_csv(LK + "/common_subset_with_all_grades.csv", dtype=str); cm["report_id"] = cm.report_id.astype(str)
rep = pd.read_parquet(EXP + "/swg_matched_reports_v2.parquet"); rep["report_id"] = rep.pathology_text_id.astype(float).astype(int).astype(str)
txt = dict(zip(rep.report_id, rep.reporttext.astype(str).str.replace("\r", "\n")))
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
cm["parts_list"] = cm.parts.map(lambda s: json.loads(s) if s and s.startswith("[") else [])
cm["mappable_part"] = cm.apply(lambda r: r.digit_in_parts == "True" and len(r.parts_list) >= 2 and r.suffix_num not in ("", None) and 1 <= int(float(r.suffix_num)) <= 10, axis=1)
items = []   # (key, section, text)
for r in cm.itertuples():
    t = body(txt[r.report_id]); items.append((f"{r.path_id}|whole_raw", "WHOLE", t)); items.append((f"{r.path_id}|whole_mapped", "WHOLE", mapped(t)))
    if r.mappable_part:
        lt = lettered(mapped(t)); items.append((f"{r.path_id}|part", L(int(float(r.suffix_num))), lt))
        for n in r.parts_list:
            if 1 <= int(n) <= 10: items.append((f"{r.path_id}|part{int(n)}", L(int(n)), lt))
    elif r.suffix_kind == "letters" and str(r.suffix)[-1].upper() in str(r.letters): items.append((f"{r.path_id}|part", str(r.suffix)[-1].upper(), mapped(t)))
tok = AutoTokenizer.from_pretrained(MODEL); enc = tok([text_for({"section": s, "report": t}) for _, s, t in items], truncation=True, max_length=MAXLEN, add_special_tokens=True)["input_ids"]
model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=6, torch_dtype=torch.float32, attn_implementation="sdpa")
model.load_state_dict(torch.load(O + "/runs/full_s0/model.pt", map_location="cpu")); model = model.cuda().eval(); pad = tok.pad_token_id
pred = {}
with torch.no_grad():
    idx = sorted(range(len(enc)), key=lambda i: len(enc[i]))
    for i in range(0, len(idx), 32):
        b = idx[i:i + 32]; Ln = max(len(enc[j]) for j in b); ids = torch.full((len(b), Ln), pad, dtype=torch.long); am = torch.zeros((len(b), Ln), dtype=torch.long)
        for k, j in enumerate(b): ids[k, :len(enc[j])] = torch.tensor(enc[j]); am[k, :len(enc[j])] = 1
        with torch.autocast("cuda", dtype=torch.bfloat16): pr = F.softmax(model(input_ids=ids.cuda(), attention_mask=am.cuda()).logits.float(), -1).cpu().numpy()
        for k, j in enumerate(b): pred[items[j][0]] = (GRADES[int(pr[k].argmax())], float(pr[k].max()))
GR = ["NDBE", "IND", "LGD", "HGD", "CANCER"]; O5 = {g: i for i, g in enumerate(GR)}
fold = lambda g: "NDBE" if g == "NORMAL_OTHER" else g
for v in ("whole_raw", "whole_mapped", "part"):
    cm[f"enc_{v}_raw"] = cm.path_id.map(lambda p: pred.get(f"{p}|{v}", (None, None))[0]); cm[f"enc_{v}"] = cm[f"enc_{v}_raw"].map(lambda g: fold(g) if g else None)
    cm[f"enc_{v}_p"] = cm.path_id.map(lambda p: pred.get(f"{p}|{v}", (None, None))[1])
cm["enc_aligned"] = cm.enc_part.where(cm.enc_part.notna(), cm.enc_whole_mapped)
cm["enc_any_part"] = cm.apply(lambda r: any(fold(pred[f"{r.path_id}|part{int(n)}"][0]) == r.swg for n in r.parts_list if f"{r.path_id}|part{int(n)}" in pred) if r.mappable_part else None, axis=1)
def qwk(a, b):
    a, b = np.asarray([O5[v] for v in a]), np.asarray([O5[v] for v in b]); n = 5; Oc = np.zeros((n, n))
    for i, j in zip(a, b): Oc[i, j] += 1
    W = np.array([[(i - j) ** 2 / (n - 1) ** 2 for j in range(n)] for i in range(n)]); E = np.outer(Oc.sum(1), Oc.sum(0)) / Oc.sum()
    return float(1 - (W * Oc).sum() / (W * E).sum()) if (W * E).sum() else float("nan")
def metrics(p, t, pats):
    d = pd.DataFrame({"p": p, "t": t, "pat": pats}).dropna(subset=["p", "t"]); d = d[d.p.isin(GR) & d.t.isin(GR)]
    if not len(d): return {"n": 0}
    ex = (d.p == d.t).values.astype(float); up = d.pat.unique(); rng = np.random.RandomState(0); grp = {q: np.where(d.pat.values == q)[0] for q in up}
    bs = [np.concatenate([grp[q] for q in rng.choice(up, len(up))]) for _ in range(NB)]
    return {"n": int(len(d)), "exact": round(float(ex.mean()), 4), "exact_ci": ci([ex[ix].mean() for ix in bs]), "qwk": round(qwk(d.p, d.t), 4),
            "LGDplus_agree": round(float(((d.p.map(O5) >= 2) == (d.t.map(O5) >= 2)).mean()), 4), "over": round(float((d.p.map(O5) > d.t.map(O5)).mean()), 4), "under": round(float((d.p.map(O5) < d.t.map(O5)).mean()), 4),
            "confusion_pred_rows_truth_cols": pd.crosstab(d.p, d.t).reindex(index=GR, columns=GR, fill_value=0).to_dict()}
pat = cm.PatientID.fillna(cm.path_id); R = {"n_common": len(cm), "n_mappable_part": int(cm.mappable_part.sum()), "record_dist": cm.swg.value_counts().to_dict(),
     "normal_other_folded": {v: int((cm[f"enc_{v}_raw"] == "NORMAL_OTHER").sum()) for v in ("whole_raw", "whole_mapped", "part")}, "all_427": {}, "whole_jury_rows": {}, "mappable_part_rows": {}}
for nm, col in (("encoder_whole_raw", "enc_whole_raw"), ("encoder_whole_mapped", "enc_whole_mapped"), ("encoder_aligned", "enc_aligned"), ("jury_majority_8_specimen", "jury_majority_8"), ("rule_section", "rule_section")):
    R["all_427"][nm] = metrics(cm[col], cm.swg, pat)
wj = cm[cm.whole_report_jury.isin(GR)]
for nm, col in (("encoder_whole_mapped", "enc_whole_mapped"), ("whole_report_jury_8", "whole_report_jury"), ("jury_majority_8_specimen", "jury_majority_8")): R["whole_jury_rows"][nm] = metrics(wj[col], wj.swg, pat[wj.index])
mp = cm[cm.mappable_part]
for nm, col in (("encoder_part", "enc_part"), ("encoder_whole_mapped", "enc_whole_mapped"), ("jury_majority_8_specimen", "jury_majority_8"), ("rule_section", "rule_section")): R["mappable_part_rows"][nm] = metrics(mp[col], mp.swg, pat[mp.index])
R["mappable_part_rows"]["encoder_any_part_carries_record_grade"] = round(float(mp.enc_any_part.mean()), 4)
R["mappable_part_rows"]["jury_specimen_any_part_na"] = "jury had the raw suffix only"
agree = cm.enc_aligned == cm.jury_majority_8; R["encoder_vs_jury"] = {"aligned_vs_jury_majority_8_exact": round(float(agree.mean()), 4), "both_disagree_with_record": int(((cm.enc_aligned != cm.swg) & (cm.jury_majority_8 != cm.swg)).sum()),
                                                                     "encoder_disagrees_only": int(((cm.enc_aligned != cm.swg) & (cm.jury_majority_8 == cm.swg)).sum()), "jury_disagrees_only": int(((cm.enc_aligned == cm.swg) & (cm.jury_majority_8 != cm.swg)).sum())}
dis = cm[cm.enc_aligned != cm.swg].copy(); dis["enc_over_record"] = dis.enc_aligned.map(O5) > dis.swg.map(O5)
R["encoder_disagreements"] = {"n": int(len(dis)), "encoder_higher": int(dis.enc_over_record.sum()), "overlap_with_jury_167": int((dis.jury_majority_8 != dis.swg).sum())}
json.dump(R, open(S + "/results.json", "w"), indent=1, default=lambda o: None); os.chmod(S + "/results.json", 0o600)
keep = ["path_id", "report_id", "suffix", "suffix_kind", "suffix_num", "n_parts", "digit_in_parts", "mappable_part", "swg", "jury_majority_8", "whole_report_jury", "rule_section",
        "enc_whole_raw_raw", "enc_whole_mapped_raw", "enc_part_raw", "enc_aligned", "enc_whole_mapped_p", "enc_part_p", "enc_any_part"]
cm[keep].to_csv(S + "/grades.csv", index=False); os.chmod(S + "/grades.csv", 0o600)
dis["report_text"] = dis.report_id.map(lambda r: body(txt[r])); dis["rehan_adjudication"] = ""; dis["rehan_note"] = ""
dis[keep + ["report_text", "rehan_adjudication", "rehan_note"]].sort_values(["swg", "enc_aligned"]).to_csv(S + "/disagreements.csv", index=False); os.chmod(S + "/disagreements.csv", 0o600)
def row(nm, m): return f"{nm:28s} n={m['n']:3d} exact {100*m['exact']:.1f} {[round(100*x,1) for x in m['exact_ci']]} QWK {m['qwk']:.3f} LGD+ {100*m['LGDplus_agree']:.1f} over {100*m['over']:.1f} under {100*m['under']:.1f}" if m.get("n") else f"{nm}: n=0"
print(json.dumps({k: R[k] for k in ("n_common", "n_mappable_part", "record_dist", "normal_other_folded", "encoder_vs_jury", "encoder_disagreements")}))
for blk in ("all_427", "whole_jury_rows", "mappable_part_rows"):
    print("==", blk)
    for nm, m in R[blk].items(): print(row(nm, m) if isinstance(m, dict) else f"{nm}: {m}")
print("SWG DONE")

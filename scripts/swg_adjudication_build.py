"""Blind SWG-record adjudication batches for the grading app (grader `swg`).

Batch 1 = the 103 encoder-vs-record disagreements (paperA/swg/grades.csv, enc_aligned != swg) + 25 random pairs where
encoder AND record agree; batch 2 = the jury-vs-record disagreements not already in batch 1 (<=100) + 20 more agreements.
Items carry ONLY the report text and what to grade (a numbered specimen / lettered section when the SWG block suffix maps
to one, else the worst grade anywhere in the report). Record, encoder and jury grades and the stratum go to the key file
only (chmod 600, never shown to the grader). Pairs sharing (report, target) collapse into one item; the key keeps every pair.
Read-only on sources. Usage: python swg_adjudication_build.py   (writes hand_grading/items_swg.json + items_swg_batch2.json)
Loading batch 2 later: python swg_adjudication_build.py load2   (appends batch 2 to items_swg.json; labels are keyed by item_id, nothing is lost)"""
import html, json, os, re, sys
import numpy as np, pandas as pd

B = "/mnt/scratche/slow/fmlab/zuberi01"; H = B + "/hand_grading"; OUT = B + "/analysis/erin_swg_adjudication"
if len(sys.argv) > 1 and sys.argv[1] == "load2":
    a = json.load(open(H + "/items_swg.json")); b = json.load(open(H + "/items_swg_batch2.json")); ids = {i["item_id"] for i in a}
    a += [i for i in b if i["item_id"] not in ids]; json.dump(a, open(H + "/items_swg.json", "w")); os.chmod(H + "/items_swg.json", 0o600)
    print(f"items_swg.json now {len(a)} items"); sys.exit()
os.makedirs(OUT, exist_ok=True); os.chmod(OUT, 0o700)
g = pd.read_csv(B + "/analysis/paperA/swg/grades.csv", dtype=str).fillna("")
cm = pd.read_csv(B + "/analysis/erin_swg_linkage/common_subset_with_all_grades.csv", dtype=str).fillna("")
g["key"] = g.path_id + "@" + g.report_id; cm["key"] = cm.path_id + "@" + cm.report_id
g = g.merge(cm[["key", "PatientID", "letters", "section_mappable"]], on="key", how="left")
assert len(g) == 427 and g.key.is_unique
rep = pd.read_parquet(B + "/barretts_db_export/swg_matched_reports_v2.parquet")
rep["report_id"] = rep.pathology_text_id.astype(float).astype(int).astype(str)
txt = dict(zip(rep.report_id, rep.reporttext.astype(str).str.replace("\r", "\n")))
body = lambda t: re.sub(r"^SPECIMEN OF INTEREST.*?\n\n", "", t, flags=re.S).strip()

# target = what Rehan grades: numbered specimen (digit suffix among the report's parts), lettered section, else whole report
def target(r):
    if r.mappable_part == "True": return f"P{int(float(r.suffix_num))}"
    if r.suffix_kind == "letters" and r.suffix and r.suffix[-1].upper() in r.letters: return "S" + r.suffix[-1].upper()
    return "WHOLE"
g["target"] = g.apply(target, axis=1)
g["stratum"] = np.select([g.enc_aligned != g.swg, g.jury_majority_8 != g.swg], ["enc_disagree", "jury_only_disagree"], "agree")

# segment markers at line start: "2.", "2)", "2  Oesophagus", "1&2.", "1-3:", "B." ; a range covers every member
SEG = re.compile(r"(?m)^[ \t]*\(?((?:\d{1,2}|[A-J])(?:\s*[&\-–]\s*(?:\d{1,2}|[A-J]))?)(?:\s?[.):\]]|[ \t]+(?=[A-Z]))[ \t]*(?=\S)")
PARTREF = re.compile(r"(?i)\b(?:part|specimen|pot|biopsy|biopsies)s?\s*(\d{1,2})\b")
def covers(lab, want):
    ends = re.split(r"\s*[&\-–]\s*", lab)
    if len(ends) == 1: return ends[0] == want
    a, b = ends; return (a <= want <= b) if want.isalpha() else (int(a) <= int(want) <= int(b)) if want.isdigit() and a.isdigit() and b.isdigit() else False
def mark(text, tgt):
    """Escape; wrap the segment(s) starting with the target's own marker in <mark>, and inline 'specimen N' references. Returns (html, found)."""
    if tgt == "WHOLE": return html.escape(text), None
    want = tgt[1:]; ms = list(SEG.finditer(text)); out, pos, hit = [], 0, False
    ref = lambda s: PARTREF.sub(lambda m: f"<mark>{m.group(0)}</mark>" if m.group(1) == want else m.group(0), s) if want.isdigit() else s
    for i, m in enumerate(ms):
        end = ms[i + 1].start() if i + 1 < len(ms) else len(text)
        out.append(ref(html.escape(text[pos:m.start()]))); seg = html.escape(text[m.start():end])
        if covers(m.group(1), want): out.append(f"<mark>{seg}</mark>"); hit = True
        else: out.append(ref(seg))
        pos = end
    tail = ref(html.escape(text[pos:])); out.append(tail)
    return "".join(out), hit or "<mark>" in "".join(out)

def ask(tgt):
    if tgt == "WHOLE": return "Grade the WHOLE report: the worst (highest) grade anywhere in it."
    what = f"specimen {tgt[1:]}" if tgt[0] == "P" else f"section {tgt[1:]}"
    return f"Grade {what} ONLY (highlighted). Ignore the other specimens."

def build(rows, start, batch):
    items, key = [], []; n = start
    seen = {}  # (report_id, target) -> item_id, within this batch
    for r in rows.itertuples():
        k = (r.report_id, r.target)
        if k not in seen:
            n += 1; cid = f"SWG-ADJ-{n:03d}"; seen[k] = f"{cid}#{r.target}"
            h, found = mark(body(txt[r.report_id]), r.target)
            items.append({"item_id": seen[k], "case_id": cid, "section": r.target, "sections_in_case": [r.target], "pos_in_case": 1,
                          "site_prefill": "OESOPHAGUS", "site_source": "default", "highlight_found": None,
                          "html": f"<div style='background:#fef3c7;border:1px solid #f59e0b;border-radius:8px;padding:8px 12px;margin-bottom:10px'><b>{ask(r.target)}</b></div>"
                                  f"<h4>Report</h4><pre>{h}</pre>", "_found": found})
        key.append({"item_id": seen[k], "batch": batch, "path_id": r.path_id, "report_id": r.report_id, "PatientID": r.PatientID,
                    "target": r.target, "mappable": r.target != "WHOLE", "stratum": r.stratum, "swg": r.swg, "enc_aligned": r.enc_aligned,
                    "enc_part": r.enc_part_raw, "enc_whole_mapped": r.enc_whole_mapped_raw, "jury_majority_8": r.jury_majority_8,
                    "whole_report_jury": r.whole_report_jury, "rule_section": r.rule_section})
    return items, key, n

rng = np.random.RandomState(20261026)
enc_dis = g[g.stratum == "enc_disagree"]; assert len(enc_dis) == 103
agree = g[g.stratum == "agree"]
a1 = agree.iloc[rng.choice(len(agree), 25, replace=False)]
b1 = pd.concat([enc_dis, a1]).iloc[rng.permutation(103 + 25)]
items1, key1, n = build(b1, 0, 1)
used = {(r.report_id, r.target) for r in b1.itertuples()}
rest = g[~g.key.isin(b1.key) & ~g.apply(lambda r: (r.report_id, r.target) in used, axis=1)]
jd = rest[rest.stratum == "jury_only_disagree"].iloc[:100]
a2 = rest[rest.stratum == "agree"].drop(a1.index, errors="ignore"); a2 = a2.iloc[rng.choice(len(a2), 20, replace=False)]
b2 = pd.concat([jd, a2]).iloc[rng.permutation(len(jd) + 20)]
items2, key2, _ = build(b2, n, 2)
for it in items1 + items2:
    if it.pop("_found") is False: it["html"] = it["html"].replace("ONLY (highlighted)", "ONLY (marker not found in the text — read the whole report and grade that specimen)")
json.dump(items1, open(H + "/items_swg.json", "w")); os.chmod(H + "/items_swg.json", 0o600)
json.dump(items2, open(H + "/items_swg_batch2.json", "w")); os.chmod(H + "/items_swg_batch2.json", 0o600)
kf = OUT + "/key_DO_NOT_OPEN_UNTIL_DONE.csv"; pd.DataFrame(key1 + key2).to_csv(kf, index=False); os.chmod(kf, 0o600)
k = pd.DataFrame(key1 + key2)
for b in (1, 2):
    kb = k[k.batch == b]
    print(f"batch {b}: {kb.item_id.nunique()} items from {len(kb)} pairs | strata {kb.drop_duplicates('item_id').stratum.value_counts().to_dict()} "
          f"| mappable {int(kb.drop_duplicates('item_id').mappable.sum())} | patients {kb.PatientID.nunique()}")

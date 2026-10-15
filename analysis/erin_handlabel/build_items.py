"""Build the per-section item queue for the grading app (one decision per screen).
Reads analysis/erin_handlabel/sample.csv (targets + extras, in assigned case order) and the
ERIN report table; writes hand_grading/items_<grader>.json with, per item: ids, the full report
as HTML with the current section highlighted, and a site pre-fill from slide METADATA (not jury).
Read-only on sources. No grades of any kind are written into the item file."""
import json, os, re, html, sys
import pandas as pd

T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"; H = "/mnt/scratche/slow/fmlab/zuberi01/hand_grading"
A = "/mnt/scratche/slow/fmlab/zuberi01/analysis/erin_handlabel"
ERIN = "/mnt/scratche/fast/fmlab/datasets/imaging/ERIN/data/PathologyReport_AnonIds.csv"
GRADER = sys.argv[1] if len(sys.argv) > 1 else "rehan"
OPTS = set(sys.argv[2:])                     # "nohl" = no <mark> highlighting (highlight_found=None); "shuffle=<seed>" = random case order
NOHL = "nohl" in OPTS
SHUFFLE = next((int(o.split("=")[1]) for o in OPTS if o.startswith("shuffle=")), None)

s = pd.read_csv(A + "/sample.csv", dtype=str).fillna("")
rep = pd.read_csv(ERIN, dtype=str, low_memory=False).fillna("").drop_duplicates("CaseName").set_index("CaseName")
sl = pd.read_csv(T + "/campaigns/allslides/erin_slides_all.csv", dtype=str).fillna("")
sl["section"] = sl.section.str.upper().str.strip()
META2SITE = {"OESOPHAGUS": "OESOPHAGUS", "STOMACH": "STOMACH", "DUODENUM": "DUODENUM", "COLORECTAL": "COLORECTAL", "LYMPH NODE": "LYMPH_NODE"}
meta_site = {}
for (c, sec), g in sl.groupby(["CaseName", "section"]):
    o = [META2SITE.get(x) for x in g.organ_meta if META2SITE.get(x)]
    if o: meta_site[(c, sec)] = max(set(o), key=o.count)

# marker = single letter "B." or a range "B-D." / "B–D:" at a segment start; a range covers every letter in it
MARK = re.compile(r"(?:(?<=^)|(?<=\s)|(?<=\n))([A-J])(?:\s*[-–]\s*([A-J]))?\s?(?:[.)\]:]|-(?=\s))\s*(?=\S)")
def covers(m, letter):
    a, b = m.group(1), m.group(2) or m.group(1)
    return a <= letter <= b
def highlight(text, letter):
    """Wrap the segment(s) belonging to `letter` in <mark>; escape everything."""
    text = re.sub(r"\s{3,}", "\n", text).strip()
    ms = list(MARK.finditer(text))
    if not ms: return html.escape(text), False
    out, pos, hit = [], 0, False
    for i, m in enumerate(ms):
        end = ms[i + 1].start() if i + 1 < len(ms) else len(text)
        out.append(html.escape(text[pos:m.start()]))
        seg = html.escape(text[m.start():end])
        if covers(m, letter): out.append(f"<mark>{seg}</mark>"); hit = True
        else: out.append(seg)
        pos = end
    out.append(html.escape(text[pos:]))
    return "".join(out), hit

# case order = the grader's assignment order (ERIN-HL-001..); repeats use repeat_case_id
rows = []
for _, r in s.iterrows():
    rows.append({"case_id": r.case_id, "report_id": r.report_id, "section": r.section, "is_repeat": False})
    if r.repeat_case_id: rows.append({"case_id": r.repeat_case_id, "report_id": r.report_id, "section": r.section, "is_repeat": True})
df = pd.DataFrame(rows).drop_duplicates(["case_id", "section"])
df["ord"] = df.case_id.str.extract(r"(\d+)$", expand=False).astype(int)
if SHUFFLE is not None:
    # random case order; repeat cases stay after every original case (so the blind repeat is never adjacent); sections stay grouped
    import numpy as np
    rng = np.random.RandomState(SHUFFLE)
    orig = sorted(df[~df.is_repeat].case_id.unique()); reps = sorted(df[df.is_repeat].case_id.unique())
    order = {c: i for i, c in enumerate(list(rng.permutation(orig)) + list(rng.permutation(reps)))}
    df["ord"] = df.case_id.map(order)
df = df.sort_values(["ord", "section"]).reset_index(drop=True)
items = []
for i, r in df.iterrows():
    R = rep.loc[r.report_id]
    if NOHL:
        plain = lambda x: html.escape(re.sub(r"\s{3,}", "\n", x).strip())
        g, f, m = plain(R.GrossDescription_redacted), plain(R.FinalDiagnosis_redacted), plain(R.MicroscopicDescription_redacted); hg = hf = hm = None
    else:
        g, hg = highlight(R.GrossDescription_redacted, r.section)
        f, hf = highlight(R.FinalDiagnosis_redacted, r.section)
        m, hm = highlight(R.MicroscopicDescription_redacted, r.section)
    secs_in_case = sorted(df[df.case_id == r.case_id].section.tolist())
    site = meta_site.get((r.report_id, r.section)); src = "metadata" if site else "default"
    items.append({"item_id": f"{r.case_id}#{r.section}", "case_id": r.case_id, "section": r.section,
                  "sections_in_case": secs_in_case, "pos_in_case": secs_in_case.index(r.section) + 1,
                  "site_prefill": site or "OESOPHAGUS", "site_source": src, "highlight_found": None if NOHL else bool(hg or hf or hm),
                  "html": (f"<h4>Gross description</h4><pre>{g}</pre><h4>Final diagnosis</h4><pre>{f}</pre>"
                           f"<h4>Microscopic description</h4><pre>{m}</pre>")})
json.dump(items, open(f"{H}/items_{GRADER}.json", "w")); os.chmod(f"{H}/items_{GRADER}.json", 0o600)
n_hl = sum(bool(i["highlight_found"]) for i in items)
print(f"items: {len(items)} | cases: {df.case_id.nunique()} | repeats: {int(df.is_repeat.sum())} | highlight: {'off' if NOHL else n_hl} | shuffle seed: {SHUFFLE} | site from metadata: {sum(i['site_source']=='metadata' for i in items)}")

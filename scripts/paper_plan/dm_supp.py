"""Demographics completion D2 (docs/demographics_completion.md @ 24b2972): read the local copies of the Killcoyne 2020 supplementary material found by
dm_namesearch.py. (1) PMC supplementary appendix EMS86618-supplement-Supp_Appendix.docx: every table with its preceding caption paragraph (docx parsed
as zip/XML). (2) Nature Source Data files 41591_2020_1033_MOESM3..16_ESM.xlsx: sheet names, shapes, header rows, and columns that could hold demographics.
Published material; tables are summaries. -> results/paper_final/demographics/d2_supp.json"""
import zipfile, re, json, glob, xml.etree.ElementTree as ET, pandas as pd
T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"
DOCX = "/mnt/scratche/fast/fmlab/zuberi01/personal/temp_hdd/Rehan/Downloads/Admin & Travel/EMS86618-supplement-Supp_Appendix.docx"
W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
txt = lambda el: "".join(t.text or "" for t in el.iter(W + "t")).strip()
z = zipfile.ZipFile(DOCX); body = ET.fromstring(z.read("word/document.xml")).find(W + "body")
tables, paras = [], []
for el in body:
    if el.tag == W + "p":
        s = txt(el)
        if s: paras.append(s)
    elif el.tag == W + "tbl":
        rows = [[txt(c) for c in r.findall(W + "tc")] for r in el.findall(W + "tr")]
        tables.append({"caption_before": paras[-3:], "n_rows": len(rows), "rows": rows})
caps = [p for p in paras if re.match(r"(Supplementary\s+)?Table\s*S?\d", p, re.I)]
DEMO = re.compile(r"age|sex|gender|male|female|smok|length|prague|circum|maxim|segment|follow|bmi|diagnos", re.I)
xl = {}
for f in sorted(glob.glob("/mnt/scratche/slow/fmlab/zuberi01/phd/killcoyne_data_from_paper/41591_2020_1033_MOESM*_ESM.xlsx"), key=lambda p: int(re.search(r"MOESM(\d+)", p).group(1))):
    o = {}
    for sh, d in pd.read_excel(f, sheet_name=None, header=None).items():
        hdr = next((list(map(str, r)) for _, r in d.iterrows() if r.notna().sum() >= 2), [])
        o[sh] = {"shape": list(d.shape), "first_row_with_2plus_values": hdr[:40], "demographic_like_cells_in_first_5_rows": sorted(set(str(v) for v in d.head(5).values.ravel() if isinstance(v, str) and DEMO.search(v)))[:20]}
    xl[f.split("/")[-1]] = o
json.dump({"docx": DOCX, "docx_table_captions": caps, "docx_tables": tables, "source_data_xlsx": xl}, open(T + "/results/paper_final/demographics/d2_supp.json", "w"), indent=1)
print("DM SUPP DONE", len(tables), len(xl))

"""Shared definitions for the RL projects (scripts/rl). No data in this file."""
import json, re
GRADES = ["NORMAL_OTHER", "NDBE", "IND", "LGD", "HGD", "CANCER"]
ORD = {g: i for i, g in enumerate(GRADES)}
POS = {"LGD", "HGD", "CANCER"}
B = "/mnt/scratche/slow/fmlab/zuberi01"
ERIN = "/mnt/scratche/slow/fmlab/datasets/imaging/ERIN/data/PathologyReport_AnonIds.csv"
MAXCHARS = 8000

SYSTEM = "You are a GI pathologist reading a UK upper-GI pathology report."
TASK = """The report may contain several lettered specimen sections (A, B, C, ...). The GROSS DESCRIPTION lists what each
lettered specimen is and where it was taken from; the FINAL DIAGNOSIS and MICROSCOPIC DESCRIPTION give the findings.
If the report has no lettered sections, the whole report is section "WHOLE".

Grade ONLY section {section}. Give the single WORST grade present in that section, one of:
NDBE (Barrett's / intestinal metaplasia, no dysplasia) | IND (indefinite for dysplasia) | LGD | HGD |
CANCER (invasive or intramucosal carcinoma, current) | NORMAL_OTHER (no Barrett's / dysplasia / cancer: normal,
gastritis, oesophagitis, reactive change, polyp, lymph node without tumour, etc.).
Historical mentions and negated findings do NOT count.{unsure}

Answer with ONLY a JSON object: {{"reasoning": "<optional, one short sentence>", "grade": "<GRADE>"}}

REPORT:
{report}"""
UNSURE_LINE = "\nIf you cannot tell the grade reliably, answer \"UNSURE\" as the grade."

def report_text(r):
    t = ("GROSS DESCRIPTION:\n" + r["GrossDescription_redacted"] + "\n\nFINAL DIAGNOSIS:\n" + r["FinalDiagnosis_redacted"] +
         "\n\nMICROSCOPIC DESCRIPTION:\n" + r["MicroscopicDescription_redacted"])
    return t[:MAXCHARS]

def make_messages(report, section, allow_unsure=False):
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": TASK.format(section=section, report=report, unsure=UNSURE_LINE if allow_unsure else "")}]

_JSON = re.compile(r"\{.*\}", re.S)
def parse_grade(text, allow_unsure=False):
    """returns grade string or None (invalid)."""
    if not isinstance(text, str): return None
    m = _JSON.search(text)
    if not m: return None
    try: d = json.loads(m.group(0))
    except Exception:
        g = re.search(r'"grade"\s*:\s*"([A-Z_]+)"', m.group(0)); d = {"grade": g.group(1)} if g else None
    if not isinstance(d, dict): return None
    g = str(d.get("grade", "")).strip().upper()
    if g in ORD: return g
    if allow_unsure and g == "UNSURE": return "UNSURE"
    return None

def completion_text(c):
    """TRL passes completions as strings or as [{'role':'assistant','content':...}]"""
    if isinstance(c, list): return c[-1]["content"] if c else ""
    return c

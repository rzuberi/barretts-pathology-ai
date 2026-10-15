"""ERIN hand-labelling sample (2026-10-02). READ-ONLY on source data.
Outputs -> /mnt/scratche/slow/fmlab/zuberi01/analysis/erin_handlabel/
  sample.csv, README.md, key_DO_NOT_OPEN_UNTIL_DONE.csv, app_cases.json
Population: oesophageal/GOJ lettered sections (jury v3 site) with >=1 H&E WSI mapped to that
section; patients shared with SWG excluded. Strata: random 50 / grade x agreement 70 / special 30.
Repeats: 10 targets re-shown at the end under a different opaque id. Blind: no grades in the app.
"""
import json, os, re, glob
from collections import Counter, defaultdict
import numpy as np, pandas as pd

T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"; L = T + "/labeller"; C = T + "/campaigns/allslides"
ERIN = "/mnt/scratche/fast/fmlab/datasets/imaging/ERIN/data/PathologyReport_AnonIds.csv"
REL = "/mnt/scratche/slow/fmlab/zuberi01/phd/barretts_retraining/barretts_training/analysis/chapter1_lgd2_final_pre_event_20260713_final"
EXP = "/mnt/scratche/slow/fmlab/zuberi01/barretts_db_export"
OUT = "/mnt/scratche/slow/fmlab/zuberi01/analysis/erin_handlabel"
os.makedirs(OUT, exist_ok=True); os.chmod(OUT, 0o700)
rng = np.random.RandomState(20261002)
ORD = {"NORMAL_OTHER": -1, "NDBE": 0, "IND": 1, "LGD": 2, "HGD": 3, "CANCER": 4}
GRADES = list(ORD)

# ---------- reports, sections, slides ----------
rep = pd.read_csv(ERIN, dtype=str, low_memory=False).fillna("").drop_duplicates("CaseName").set_index("CaseName")
sec = pd.read_csv(L + "/erin_sections_v3_consensus.csv", dtype=str).fillna("")
sec = sec[sec.site.isin(["OESOPHAGUS", "GOJ"]) & sec.section.str.fullmatch(r"[A-J]")].copy()
sl = pd.read_csv(C + "/erin_slides_all.csv", dtype=str).fillna(""); sl["section"] = sl.section.str.upper().str.strip()
he_secs = set(zip(sl[sl.is_he == "True"].CaseName, sl[sl.is_he == "True"].section))
sec["has_he_wsi"] = [(c, s) in he_secs for c, s in zip(sec.CaseName, sec.section)]
sec["anon_id"] = sec.CaseName.map(rep.anon_id)
sec["year"] = pd.to_datetime(sec.CaseName.map(rep.CollectedOrOrdered), dayfirst=True, errors="coerce").dt.year

# ---------- SWG-shared patients -> ERIN anon_ids (accession + DB bridge; superset of the 54) ----------
def acc_norm(s):
    s = re.sub(r"[\s_][A-Za-z]{1,2}\d{0,2}\s*(\(.*\))?\s*$", "", str(s).strip())
    m = re.match(r"^\s*([A-Za-z]{1,3})\s*(\d{2})\s*[-./]?\s*(\d{3,6})\s*$", s)
    return f"{m.group(1).upper()}{m.group(2)}-{int(m.group(3)):05d}" if m else None
coh = pd.read_csv(REL + "/pre_event_cohort.csv", dtype=str); coh["acc"] = coh.BiopsyID_real.map(acc_norm)
coh["pid"] = coh.participant_id.map(lambda v: re.sub(r"\.0$", "", str(v)) if pd.notna(v) else None)
racc = pd.Series(rep.anon_id.values, index=rep.index.map(acc_norm))
excl = set(racc.reindex(coh.acc.dropna()).dropna())
db = pd.read_csv(EXP + "/pathology_text_normalised_full.csv", dtype=str, usecols=["participant_id", "specimennumber"])
db["acc"] = db.specimennumber.map(acc_norm); db["pid"] = db.participant_id.map(lambda v: re.sub(r"\.0$", "", str(v)) if pd.notna(v) else None)
bridge = db.dropna(subset=["acc", "pid"]); bridge = bridge[bridge.pid.isin(set(coh.pid.dropna()))]
excl |= set(racc.reindex(bridge.acc).dropna())
sec["swg_shared"] = sec.anon_id.isin(excl)

pop = sec[sec.has_he_wsi & ~sec.swg_shared].copy().reset_index(drop=True)
pop["sid"] = pop.CaseName + "#" + pop.section

# ---------- per-juror worst grade -> agreement class ----------
jw = defaultdict(dict)
for f in glob.glob(T + "/feasibility/runs/sec3full_*/output/sections_v3_*.csv"):
    juror = os.path.basename(f).replace("sections_v3_", "").rsplit("_shard", 1)[0]
    d = pd.read_csv(f, dtype=str).fillna("").drop_duplicates("CaseName")
    for r in d.itertuples():
        if r.sections_json.startswith("PARSE_FAIL"): continue
        for s in json.loads(r.sections_json):
            key = s["section"].upper().strip()[:2]
            g = [x for x in s["grades"] if x in ORD]
            if g: jw[(r.CaseName, key)][juror] = max(g, key=lambda x: ORD[x])
def agreement(sid_case, section, cons):
    v = jw.get((sid_case, section), {})
    if not v: return "split", 0.0
    c = Counter(v.values()); top = c.most_common(1)[0][1] / len(v)
    agree_cons = c.get(cons, 0) / len(v)
    if agree_cons == 1.0: return "unanimous", agree_cons
    if agree_cons >= 0.5: return "majority", agree_cons
    return "split", agree_cons
ag = [agreement(c, s, g) for c, s, g in zip(pop.CaseName, pop.section, pop.worst_grade)]
pop["agreement"] = [a for a, _ in ag]; pop["agree_frac"] = [f for _, f in ag]

# ---------- special-case tags ----------
whole = pd.read_csv(L + "/erin_labels_jury_final.csv", dtype=str).fillna("").drop_duplicates("CaseName").set_index("CaseName")
MARK = re.compile(r"(?:(?<=^)|(?<=\s)|(?<=\n))([A-J])\s?(?:[.)\]:]|-(?=\s))\s*(?=\S)")
def split_sections(text):
    ms = list(MARK.finditer(text)); out = {}
    for i, m in enumerate(ms):
        end = ms[i + 1].start() if i + 1 < len(ms) else len(text)
        out.setdefault(m.group(1), []).append(text[m.end():end])
    return {k: " ".join(v) for k, v in out.items()}
GRADE_WORDS = re.compile(r"dysplasia|carcinoma|malignan|barrett|intestinal metaplasia|indefinite", re.I)
XREF = re.compile(r"\b(?:as|similar to|same as|identical to|see|like)\s+(?:in|for|with)?\s*(?:part|section|specimen|pot|site)?\s*[A-H]\b(?![a-z])", re.I)
HEDGE = re.compile(r"suspicious for|cannot (?:be )?exclude|at least|worrisome for|cannot rule out|possibl[ey]\b|query\b|\?\s*(?:dysplasia|carcinoma|hgd|lgd)", re.I)
INV = re.compile(r"invasi|submucosa|muscularis propria|\bpT[1-4]|lymphovascular|perineural", re.I)
tags = []
for r in pop.itertuples():
    R = rep.loc[r.CaseName]; fd, mi = R.FinalDiagnosis_redacted, R.MicroscopicDescription_redacted
    fsec, msec = split_sections(fd), split_sections(mi)
    stext = (fsec.get(r.section, "") + " " + msec.get(r.section, "")).strip() or (fd + " " + mi)
    t = []
    if not fsec and len(msec) >= 2 and GRADE_WORDS.search(fd): t.append("conclusion_without_part")
    if XREF.search(stext): t.append("cross_reference")
    if HEDGE.search(stext): t.append("hedged_wording")
    if r.worst_grade == "CANCER" and re.search(r"intramucosal", fd + " " + mi, re.I) and not INV.search(fd + " " + mi): t.append("imc_only")
    if re.search(r"p53", stext, re.I): t.append("p53_in_section")
    wl = whole.final_label.get(r.CaseName, "")
    if wl in ORD and wl != r.worst_grade and not (wl == "NDBE" and r.worst_grade == "NORMAL_OTHER"): t.append("section_vs_whole_differs")
    tags.append("|".join(t))
pop["special_tags"] = tags
v2 = pd.read_csv(L + "/erin_slide_labels_v2.csv", dtype=str)
in1538 = set(zip(v2.CaseName, v2.section.str.upper().str.strip()))
pop["in_1538"] = [(c, s) in in1538 for c, s in zip(pop.CaseName, pop.section)]
N = len(pop)

# ---------- sampling ----------
drawn = {}  # sid -> dict(stratum, cell, p_stratum)
def draw(cands, n, stratum, cell, prefer_1538=True):
    cands = cands[~cands.sid.isin(drawn)]
    if prefer_1538:
        a, b = cands[cands.in_1538], cands[~cands.in_1538]
        pick = a.sample(min(n, len(a)), random_state=rng) if len(a) else a
        if len(pick) < n and len(b): pick = pd.concat([pick, b.sample(min(n - len(pick), len(b)), random_state=rng)])
    else:
        pick = cands.sample(min(n, len(cands)), random_state=rng)
    for sid in pick.sid: drawn[sid] = {"stratum": stratum, "cell": cell, "p_stratum": None, "cell_N": None, "cell_n": None}
    return pick
# 1. random slice (true SRS from the whole population)
draw(pop, 50, "random", "all", prefer_1538=False)
# 2. grade x agreement: allocation oversampling IND/LGD/HGD and split cells
alloc = {}
for g in GRADES:
    base = 6 if g in ("IND", "LGD", "HGD") else 3
    alloc[(g, "unanimous")] = base - 1; alloc[(g, "majority")] = base; alloc[(g, "split")] = base + 1
# total = 3*(5+6+7) + 3*(2+3+4) = 54 + 27 = 81 -> scale to 70
scale = 70 / sum(alloc.values()); alloc = {k: max(1, int(round(v * scale))) for k, v in alloc.items()}
short = 0
for (g, a), n in alloc.items():
    cands = pop[(pop.worst_grade == g) & (pop.agreement == a)]
    got = draw(cands, n, "stratified", f"{g}|{a}")
    short += n - len(got)
fill_cells = [("LGD", "majority"), ("IND", "majority"), ("HGD", "majority"), ("LGD", "unanimous"), ("IND", "unanimous"), ("HGD", "unanimous")]
rounds = 0
while short > 0 and rounds < 10:          # spread the shortfall one section at a time across the rare-grade cells
    for (g, a) in fill_cells:
        if short <= 0: break
        got = draw(pop[(pop.worst_grade == g) & (pop.agreement == a)], 1, "stratified", f"{g}|{a}+fill"); short -= len(got)
    rounds += 1
# 3. special cases: 5 per tag
for tag in ["conclusion_without_part", "cross_reference", "hedged_wording", "imc_only", "p53_in_section", "section_vs_whole_differs"]:
    draw(pop[pop.special_tags.str.contains(tag)], 5, "special", tag)
cell_stats = {}
for sid, d in drawn.items():
    cell_stats.setdefault(d["cell"], {"n": 0}); cell_stats[d["cell"]]["n"] += 1
for cell, st in cell_stats.items():
    base = cell.replace("+fill", "")
    if cell == "all": st["N"] = N
    elif "|" in base: g, a = base.split("|"); st["N"] = int(((pop.worst_grade == g) & (pop.agreement == a)).sum())
    else: st["N"] = int(pop.special_tags.str.contains(base).sum())
for sid, d in drawn.items():
    d["cell_n"] = cell_stats[d["cell"]]["n"]; d["cell_N"] = cell_stats[d["cell"]]["N"]; d["p_stratum"] = d["cell_n"] / max(d["cell_N"], 1)
S = pop[pop.sid.isin(drawn)].copy()
S["stratum"] = S.sid.map(lambda s: drawn[s]["stratum"]); S["cell"] = S.sid.map(lambda s: drawn[s]["cell"])
S["p_stratum"] = S.sid.map(lambda s: drawn[s]["p_stratum"]); S["cell_N"] = S.sid.map(lambda s: drawn[s]["cell_N"]); S["cell_n"] = S.sid.map(lambda s: drawn[s]["cell_n"])
# inclusion probability across the three draws (approximate, treating draws as independent):
p_rand = 50 / N
strat_cells = {k: v for k, v in drawn.items()}
def p_incl(r):
    p = [p_rand]
    key = f"{r.worst_grade}|{r.agreement}"
    n_cell = sum(1 for d in drawn.values() if d["stratum"] == "stratified" and d["cell"].replace("+fill", "") == key)
    N_cell = int(((pop.worst_grade == r.worst_grade) & (pop.agreement == r.agreement)).sum())
    if n_cell: p.append(n_cell / N_cell)
    for tag in str(r.special_tags).split("|"):
        rows = [d for d in drawn.values() if d["stratum"] == "special" and d["cell"] == tag]
        if tag and rows: p.append(rows[0]["cell_n"] / rows[0]["cell_N"])
    return 1 - np.prod([1 - x for x in p])
S["p_inclusion"] = S.apply(p_incl, axis=1); S["weight"] = 1 / S.p_inclusion

# ---------- reports -> cases; extras; repeats; ordering ----------
targets = S.copy()
reports = targets.CaseName.unique()
extras = pop[pop.CaseName.isin(reports) & ~pop.sid.isin(targets.sid)].copy()
extras["stratum"] = "extra"; extras["cell"] = ""; extras["p_stratum"] = np.nan; extras["p_inclusion"] = np.nan; extras["weight"] = np.nan
allsec = pd.concat([targets, extras], ignore_index=True)
# interleave strata across sessions of 20 reports
rep_strat = targets.groupby("CaseName").stratum.agg(lambda s: "special" if "special" in set(s) else ("stratified" if "stratified" in set(s) else "random"))
lists = {k: list(rng.permutation(v.index.tolist())) for k, v in rep_strat.groupby(rep_strat)}
order = []
while any(lists.values()):
    for k in ["random", "stratified", "special"]:
        if lists.get(k): order.append(lists[k].pop())
opaque = {cn: f"ERIN-HL-{i+1:03d}" for i, cn in enumerate(order)}
repeat_targets = targets[targets.stratum != "extra"].sample(10, random_state=rng)
repeat_reports = list(repeat_targets.CaseName.unique())
rep_opaque = {cn: f"ERIN-HL-{len(order)+i+1:03d}" for i, cn in enumerate(repeat_reports)}

def case_text(cn, secs_in_report):
    R = rep.loc[cn]
    return (f"Sections to grade: {', '.join(sorted(secs_in_report))}.\n\n"
            f"GROSS DESCRIPTION:\n{R.GrossDescription_redacted}\n\nFINAL DIAGNOSIS:\n{R.FinalDiagnosis_redacted}\n\n"
            f"MICROSCOPIC DESCRIPTION:\n{R.MicroscopicDescription_redacted}")
cases = []
for cn in order:
    secs_in = sorted(allsec[allsec.CaseName == cn].section.unique())
    cases.append({"case": opaque[cn], "block": "hl2026", "text": case_text(cn, secs_in)})
for cn in repeat_reports:
    secs_in = sorted(allsec[allsec.CaseName == cn].section.unique())
    cases.append({"case": rep_opaque[cn], "block": "hl2026_repeat", "text": case_text(cn, secs_in)})

# ---------- outputs ----------
allsec["case_id"] = allsec.CaseName.map(opaque); allsec["target"] = np.where(allsec.stratum == "extra", "extra", "target")
allsec["repeat_flag"] = allsec.CaseName.isin(repeat_reports) & allsec.sid.isin(repeat_targets.sid)
allsec["repeat_case_id"] = allsec.CaseName.map(rep_opaque)
allsec["session"] = allsec.case_id.map(lambda c: (int(c.split("-")[-1]) - 1) // 20 + 1)
cols = ["sid", "CaseName", "section", "case_id", "session", "stratum", "cell", "special_tags", "p_stratum", "p_inclusion", "weight", "target", "repeat_flag", "repeat_case_id", "in_1538", "site", "year"]
allsec[cols].rename(columns={"sid": "section_id", "CaseName": "report_id"}).to_csv(OUT + "/sample.csv", index=False)
allsec[cols + ["worst_grade", "grades", "agreement", "agree_frac", "n_jurors"]].rename(columns={"sid": "section_id", "CaseName": "report_id"}).to_csv(OUT + "/key_DO_NOT_OPEN_UNTIL_DONE.csv", index=False)
json.dump(cases, open(OUT + "/app_cases.json", "w"))
for f in os.listdir(OUT):
    if os.path.isfile(os.path.join(OUT, f)): os.chmod(os.path.join(OUT, f), 0o600)

summary = {"population_sections": int(N), "population_reports": int(pop.CaseName.nunique()), "population_patients": int(pop.anon_id.nunique()),
           "excluded_swg_shared_anon_ids": len(excl), "excluded_sections_swg": int((sec.has_he_wsi & sec.swg_shared).sum()),
           "population_grade_dist": pop.worst_grade.value_counts().to_dict(), "population_agreement_dist": pop.agreement.value_counts().to_dict(),
           "population_in_1538": int(pop.in_1538.sum()),
           "targets": int(len(targets)), "targets_by_stratum": targets.stratum.value_counts().to_dict(),
           "targets_by_grade": targets.worst_grade.value_counts().to_dict(), "targets_by_agreement": targets.agreement.value_counts().to_dict(),
           "stratified_cells": {k["cell"]: (k["cell_n"], k["cell_N"]) for k in drawn.values() if k["stratum"] == "stratified"},
           "special_cells": {k["cell"]: (k["cell_n"], k["cell_N"]) for k in drawn.values() if k["stratum"] == "special"},
           "special_tag_population_counts": {t: int(pop.special_tags.str.contains(t).sum()) for t in ["conclusion_without_part", "cross_reference", "hedged_wording", "imc_only", "p53_in_section", "section_vs_whole_differs"]},
           "targets_in_1538": int(targets.in_1538.sum()), "frac_targets_in_1538": round(float(targets.in_1538.mean()), 3),
           "reports": int(len(order)), "extras": int(len(extras)), "sections_to_label": int(len(allsec)), "repeat_cases": len(repeat_reports),
           "total_cases_in_app": len(cases), "sessions_of_20": int(np.ceil(len(order) / 20))}
json.dump(summary, open(OUT + "/summary.json", "w"), indent=1)
print(json.dumps(summary, indent=1))

readme = f"""# ERIN hand-labelling sample (built 2026-10-02)

Population: {N} oesophageal/GOJ lettered sections (jury-v3 `site`) from {summary['population_reports']} ERIN reports
({summary['population_patients']} patients) with >=1 H&E WSI mapped to that section; {summary['excluded_sections_swg']} sections
from {summary['excluded_swg_shared_anon_ids']} ERIN patients shared with SWG excluded. Agreement classes are over each juror's
worst grade for the section: unanimous = all jurors agree with the consensus; majority = >=50% but not all; split = <50%.
Split is nearly empty by construction (consensus keeps a grade only when >=half the jurors list it): {summary['population_agreement_dist']}.

Targets: {summary['targets']} sections in {summary['reports']} reports; with the {summary['extras']} other oesophageal/GOJ
sections in those reports you will label {summary['sections_to_label']} sections. {summary['targets_in_1538']} targets
({summary['frac_targets_in_1538']:.0%}) are in the 1,538-slide set used for the section-vs-case-max comparison.
Repeats: {summary['repeat_cases']} reports re-shown at the end under new ids ({len(order)+1}-{len(cases)}); label them without
looking back. Reports are shuffled and strata interleaved so each block of 20 mixes random / stratified / special.

| stratum | targets | cells (n drawn / N in population) |
|---|---|---|
| random (simple random sample of the population) | {summary['targets_by_stratum'].get('random',0)} | 50 / {N} |
| stratified grade x agreement | {summary['targets_by_stratum'].get('stratified',0)} | {', '.join(f"{k}: {a}/{b}" for k,(a,b) in sorted(summary['stratified_cells'].items()))} |
| special cases | {summary['targets_by_stratum'].get('special',0)} | {', '.join(f"{k}: {a}/{b}" for k,(a,b) in summary['special_cells'].items())} |

## Reweighting
Each target's `p_inclusion` = 1 - (1-p_random)(1-p_cell)(1-p_tag1)(1-p_tag2)... where p_random = 50/{N},
p_cell = (targets drawn from its grade x agreement cell, incl. fill draws) / (population size of that cell), and p_tag =
(targets drawn for a special tag) / (population sections carrying that tag), for every tag the section carries. Draws are
treated as independent (they were sequential without replacement, so this is a close approximation). Population-level
accuracy = sum(weight_i * correct_i) / sum(weight_i) with weight_i = 1/p_inclusion over TARGET rows only; extras are
labelled for completeness and have no weight (they can be reported as an unweighted secondary set). Use the random slice
alone (n=50, equal weights) as the design-free check. Per-stratum accuracies are reported unweighted.

Blinding: the app shows report text only (no jury, model or rule grade; no stratum or tag). The key with jury grades,
strata and repeat mapping is `key_DO_NOT_OPEN_UNTIL_DONE.csv`.
"""
open(OUT + "/README.md", "w").write(readme); os.chmod(OUT + "/README.md", 0o600)

"""Cohort + labels + CV folds for the Panakeia probe (public TCGA only; see DESIGN.md).
Inputs: labels_raw/ (GDC slide lists, cBioPortal PanCanAtlas clinical JSON, ABSOLUTE table).
Outputs: cohort.csv (one row per patient: chosen slide, labels, site), folds.csv (task x split x repeat -> fold),
results/cohort_flow.json (counts for the flow diagram)."""
import collections, csv, json, os, re
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold

A = "/mnt/scratche/slow/fmlab/zuberi01/analysis/outreach_panakeia"; R = A + "/labels_raw"
SLIDES = {"stad": "/mnt/scratche/fast/fmlab/datasets/imaging/tcga_stad/slides",
          "esca": "/mnt/scratche/fast/fmlab/datasets/imaging/tcga_esca/slides"}
os.makedirs(A + "/results", exist_ok=True)


def cbio(study):
    d = collections.defaultdict(dict)
    for t in ("pat", "samp"):
        for x in json.load(open(f"{R}/{study}_tcga_pan_can_atlas_2018_{t}.json")):
            d[x["patientId"]][x["clinicalAttributeId"]] = x["value"]
    return d


flow = {}
lab = {}
for s in ("stad", "esca"):
    for p, v in cbio(s).items(): lab[p] = {**v, "cohort": s}
ab = {}
for r in csv.DictReader(open(R + "/absolute.txt"), delimiter="\t"):
    if r["array"][13:15] == "01": ab[r["array"][:12]] = r   # primary tumour sample

rows = []
for proj, s in (("TCGA-STAD", "stad"), ("TCGA-ESCA", "esca")):
    g = list(csv.DictReader(open(f"{R}/{proj}.tsv"), delimiter="\t"))
    per = collections.defaultdict(list)
    for r in g:
        if "-01Z-" in r["file_name"]: per[r["cases.0.submitter_id"]].append(r)
    flow[f"{s}_patients_with_dx_slide"] = len(per)
    flow[f"{s}_dx_slides"] = len(g)
    n_lab = n_sq = 0
    for p, rs in per.items():
        if p not in lab: continue
        n_lab += 1
        if s == "esca" and lab[p].get("CANCER_TYPE_DETAILED") != "Esophageal Adenocarcinoma":
            n_sq += 1; continue
        dxn = lambda r: int(re.search(r"-DX(\d+)", r["file_name"]).group(1))
        r = sorted(rs, key=lambda r: (dxn(r), r["file_name"]))[0]
        L = lab[p]; a = ab.get(p)
        wgd = float(a["Genome doublings"]) >= 1 if a and a["call status"] == "called" and a["Genome doublings"] != "" else None
        sub = L.get("SUBTYPE")
        mantis = float(L["MSI_SCORE_MANTIS"]) if L.get("MSI_SCORE_MANTIS") not in (None, "", "NA") else None
        rows.append({"patient": p, "cohort": s, "site": p[5:7], "file_id": r["file_id"], "file_name": r["file_name"],
                     "file_size": int(r["file_size"]), "md5": r["md5sum"], "slide_path": f"{SLIDES[s]}/{r['file_name']}",
                     "n_dx_slides": len(rs), "histology": L.get("CANCER_TYPE_DETAILED"),
                     "wgd": None if wgd is None else int(wgd),
                     "absolute_ploidy": float(a["ploidy"]) if a and a["ploidy"] not in ("", "NA") else None,
                     "aneuploidy": float(L["ANEUPLOIDY_SCORE"]) if L.get("ANEUPLOIDY_SCORE") not in (None, "", "NA") else None,
                     "subtype": sub, "cin": None if not sub else int(sub.endswith("_CIN")),
                     "mantis": mantis, "msi": None if (mantis is None or s != "stad") else int(mantis >= 0.4)})
    flow[f"{s}_with_cbioportal"] = n_lab
    flow[f"{s}_squamous_dropped"] = n_sq

c = pd.DataFrame(rows).sort_values("patient").reset_index(drop=True)
c.to_csv(A + "/cohort.csv", index=False)
for s in ("stad", "esca"):
    d = c[c.cohort == s]; flow[f"{s}_adeno_cohort"] = len(d)
    for t in ("wgd", "cin", "msi", "aneuploidy"):
        v = d[t].dropna(); flow[f"{s}_{t}_labelled"] = int(len(v))
        if t != "aneuploidy": flow[f"{s}_{t}_pos"] = int(v.sum())
flow["msi_vs_subtype_concordance"] = float(((c.msi == 1) == (c.subtype == "STAD_MSI"))[c.msi.notna() & c.subtype.notna()].mean())
json.dump(flow, open(A + "/results/cohort_flow.json", "w"), indent=1)
print(json.dumps(flow, indent=1))

# ---- folds: task x split (strat|site) x repeat -> fold id per patient (-1 = not in task) ----
TASKS = {"wgd": ("wgd", None), "cin": ("cin", None), "aneu": ("aneuploidy", None), "msi": ("msi", "stad")}
F = pd.DataFrame({"patient": c.patient})
for ti, (task, (col, only)) in enumerate(TASKS.items()):
    m = c[col].notna() & ((c.cohort == only) if only else True)
    d = c[m]
    yb = (d[col] > d[col].median()).astype(int) if task == "aneu" else d[col].astype(int)
    strata = d.cohort + "_" + yb.astype(str)
    for rep in range(3):
        f = np.full(len(c), -1)
        for k, (_, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=1000 * rep + ti).split(d, strata)):
            f[d.index[te]] = k
        F[f"{task}_strat_r{rep}"] = f
        if task in ("wgd", "msi"):
            f = np.full(len(c), -1)
            for k, (_, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=1000 * rep + ti).split(d, strata, d.site)):
                f[d.index[te]] = k
            F[f"{task}_site_r{rep}"] = f
F.to_csv(A + "/folds.csv", index=False)
print("folds", F.shape)

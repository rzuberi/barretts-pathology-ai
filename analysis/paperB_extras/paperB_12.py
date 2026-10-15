"""Paper B extras, Analyses 1 (triage) and 2 (auditor). READ-ONLY on all inputs.
Outputs -> analysis/paperB_extras/{triage,auditor}/ (CSV tables, PNG+PDF figures, results JSON).
Patient-clustered bootstrap (2,000) for every CI. S4/S5 logistic models: 5-fold patient-split CV on the labelled set,
reported out-of-fold; population scored by the mean of the five fold models. Nothing is tuned on T1/T2 or on the labelled set
outside that CV."""
import glob, json, os, re, sys
from collections import Counter, defaultdict
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GroupKFold
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

B = "/mnt/scratche/slow/fmlab/zuberi01"; T = B + "/phd/thesis"; L = T + "/labeller"; C = T + "/campaigns/allslides"
HL = B + "/analysis/erin_handlabel"; HG = B + "/hand_grading"; ISBI = B + "/analysis/isbi_rerun"
OUT = B + "/analysis/paperB_extras"; TRI = OUT + "/triage"; AUD = OUT + "/auditor"
for d in (OUT, TRI, AUD): os.makedirs(d, exist_ok=True); os.chmod(d, 0o700)
ERIN = "/mnt/scratche/fast/fmlab/datasets/imaging/ERIN/data/PathologyReport_AnonIds.csv"
REL = B + "/phd/barretts_retraining/barretts_training/analysis/chapter1_lgd2_final_pre_event_20260713_final"; EXP = B + "/barretts_db_export"
ORD = {"NORMAL_OTHER": -1, "NDBE": 0, "IND": 1, "LGD": 2, "HGD": 3, "CANCER": 4}; GR = list(ORD); CLS6 = GR; C_OF = {c: i for i, c in enumerate(GR)}
POS = {"LGD", "HGD", "CANCER"}; NB = 2000; RNG = np.random.RandomState(20261007)
BUDGETS = np.arange(0, 51) / 100.0
log = []
def say(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); log.append(s)

# ======================= population (identical to build_handlabel_sample.py) =======================
rep = pd.read_csv(ERIN, dtype=str, low_memory=False).fillna("").drop_duplicates("CaseName").set_index("CaseName")
sec = pd.read_csv(L + "/erin_sections_v3_consensus.csv", dtype=str).fillna("")
sec = sec[sec.site.isin(["OESOPHAGUS", "GOJ"]) & sec.section.str.fullmatch(r"[A-J]")].copy()
sl = pd.read_csv(C + "/erin_slides_all.csv", dtype=str).fillna(""); sl["section"] = sl.section.str.upper().str.strip()
he_secs = set(zip(sl[sl.is_he == "True"].CaseName, sl[sl.is_he == "True"].section))
sec["has_he_wsi"] = [(c, s) in he_secs for c, s in zip(sec.CaseName, sec.section)]
sec["anon_id"] = sec.CaseName.map(rep.anon_id)
def acc_norm(s):
    s = re.sub(r"[\s_][A-Za-z]{1,2}\d{0,2}\s*(\(.*\))?\s*$", "", str(s).strip())
    m = re.match(r"^\s*([A-Za-z]{1,3})\s*(\d{2})\s*[-./]?\s*(\d{3,6})\s*$", s)
    return f"{m.group(1).upper()}{m.group(2)}-{int(m.group(3)):05d}" if m else None
coh = pd.read_csv(REL + "/pre_event_cohort.csv", dtype=str); coh["acc"] = coh.BiopsyID_real.map(acc_norm)
coh["pid"] = coh.participant_id.map(lambda v: re.sub(r"\.0$", "", str(v)) if pd.notna(v) else None)
racc = pd.Series(rep.anon_id.values, index=rep.index.map(acc_norm)); excl = set(racc.reindex(coh.acc.dropna()).dropna())
dbx = pd.read_csv(EXP + "/pathology_text_normalised_full.csv", dtype=str, usecols=["participant_id", "specimennumber"])
dbx["acc"] = dbx.specimennumber.map(acc_norm); dbx["pid"] = dbx.participant_id.map(lambda v: re.sub(r"\.0$", "", str(v)) if pd.notna(v) else None)
bridge = dbx.dropna(subset=["acc", "pid"]); bridge = bridge[bridge.pid.isin(set(coh.pid.dropna()))]; excl |= set(racc.reindex(bridge.acc).dropna())
sec["swg_shared"] = sec.anon_id.isin(excl)
pop = sec[sec.has_he_wsi & ~sec.swg_shared].copy().reset_index(drop=True); pop["sid"] = pop.CaseName + "#" + pop.section
N = len(pop); summ = json.load(open(HL + "/summary.json"))
say(f"population sections {N} (summary.json says {summ['population_sections']}); reports {pop.CaseName.nunique()}; patients {pop.anon_id.nunique()}")
assert N == summ["population_sections"], "population could not be reproduced"

# per-juror worst grades over the whole population
jw = defaultdict(dict)
for f in glob.glob(T + "/feasibility/runs/sec3full_*/output/sections_v3_*.csv"):
    juror = os.path.basename(f).replace("sections_v3_", "").rsplit("_shard", 1)[0]
    d = pd.read_csv(f, dtype=str).fillna("").drop_duplicates("CaseName")
    for r in d.itertuples():
        if r.sections_json.startswith("PARSE_FAIL"): continue
        for s in json.loads(r.sections_json):
            key = s["section"].upper().strip()[:2]; g = [x for x in s["grades"] if x in ORD]
            if g: jw[(r.CaseName, key)][juror] = max(g, key=lambda x: ORD[x])
def agreement(c, s, cons):
    v = jw.get((c, s), {})
    if not v: return "split", 0.0
    cnt = Counter(v.values()); a = cnt.get(cons, 0) / len(v)
    return ("unanimous" if a == 1.0 else "majority" if a >= 0.5 else "split"), a
ag = [agreement(c, s, g) for c, s, g in zip(pop.CaseName, pop.section, pop.worst_grade)]
pop["agreement"] = [a for a, _ in ag]; pop["agree_frac"] = [f for _, f in ag]
# special tags (same regexes as the sample builder)
whole = pd.read_csv(L + "/erin_labels_jury_final.csv", dtype=str).fillna("").drop_duplicates("CaseName").set_index("CaseName")
MARK = re.compile(r"(?:(?<=^)|(?<=\s)|(?<=\n))([A-J])\s?(?:[.)\]:]|-(?=\s))\s*(?=\S)")
def split_sections(text):
    ms = list(MARK.finditer(text)); out = {}
    for i, m in enumerate(ms):
        end = ms[i + 1].start() if i + 1 < len(ms) else len(text); out.setdefault(m.group(1), []).append(text[m.end():end])
    return {k: " ".join(v) for k, v in out.items()}
GRADE_WORDS = re.compile(r"dysplasia|carcinoma|malignan|barrett|intestinal metaplasia|indefinite", re.I)
XREF = re.compile(r"\b(?:as|similar to|same as|identical to|see|like)\s+(?:in|for|with)?\s*(?:part|section|specimen|pot|site)?\s*[A-H]\b(?![a-z])", re.I)
HEDGE = re.compile(r"suspicious for|cannot (?:be )?exclude|at least|worrisome for|cannot rule out|possibl[ey]\b|query\b|\?\s*(?:dysplasia|carcinoma|hgd|lgd)", re.I)
INV = re.compile(r"invasi|submucosa|muscularis propria|\bpT[1-4]|lymphovascular|perineural", re.I)
tags = []
for r in pop.itertuples():
    R = rep.loc[r.CaseName]; fd, mi = R.FinalDiagnosis_redacted, R.MicroscopicDescription_redacted
    fsec, msec = split_sections(fd), split_sections(mi); stext = (fsec.get(r.section, "") + " " + msec.get(r.section, "")).strip() or (fd + " " + mi); t = []
    if not fsec and len(msec) >= 2 and GRADE_WORDS.search(fd): t.append("conclusion_without_part")
    if XREF.search(stext): t.append("cross_reference")
    if HEDGE.search(stext): t.append("hedged_wording")
    if r.worst_grade == "CANCER" and re.search(r"intramucosal", fd + " " + mi, re.I) and not INV.search(fd + " " + mi): t.append("imc_only")
    if re.search(r"p53", stext, re.I): t.append("p53_in_section")
    wl = whole.final_label.get(r.CaseName, "")
    if wl in ORD and wl != r.worst_grade and not (wl == "NDBE" and r.worst_grade == "NORMAL_OTHER"): t.append("section_vs_whole_differs")
    tags.append("|".join(t))
pop["special_tags"] = tags

# ======================= inclusion probability for EVERY population section =======================
cells = {**{k: tuple(v) for k, v in summ["stratified_cells"].items()}, **{k: tuple(v) for k, v in summ["special_cells"].items()}}
cell_n = defaultdict(int); cell_N = {}
for k, (n, Nc) in cells.items():
    base = k.replace("+fill", ""); cell_n[base] += n; cell_N[base] = Nc
p_rand = 50 / N
def p_incl(r):
    p = [p_rand]; key = f"{r.worst_grade}|{r.agreement}"
    if cell_n.get(key): p.append(cell_n[key] / cell_N[key])
    for tag in str(r.special_tags).split("|"):
        if tag and cell_n.get(tag): p.append(cell_n[tag] / cell_N[tag])
    return 1 - np.prod([1 - x for x in p])
pop["p_sec"] = pop.apply(p_incl, axis=1)
p_rep = 1 - pop.groupby("CaseName").p_sec.apply(lambda v: np.prod(1 - v.values)); pop["p_report"] = pop.CaseName.map(p_rep)

# ======================= signals over the population =======================
def vf(r):
    try: v = json.loads(r.grade_votes); return v.get(r.worst_grade, 0) / int(r.n_jurors)
    except Exception: return np.nan
def ent(r):
    try: v = np.array([x for x in json.loads(r.grade_votes).values() if x > 0], float); p = v / v.sum(); return float(-(p * np.log(p)).sum())
    except Exception: return np.nan
pop["S1_votefrac"] = pop.apply(vf, axis=1); pop["S2_entropy"] = pop.apply(ent, axis=1)
pop["mistral_worst"] = [jw.get((c, s), {}).get("mistral-small3_2") for c, s in zip(pop.CaseName, pop.section)]
pop["S3_mistral_disagree"] = [(1 if (m is None or m != g) else 0) for m, g in zip(pop.mistral_worst, pop.worst_grade)]   # no mistral output counts as disagreement
say("population S3 disagreement rate", round(pop.S3_mistral_disagree.mean(), 4), "| mistral missing", int(pop.mistral_worst.isna().sum()))

# ======================= labelled set =======================
key = pd.read_csv(HL + "/key_DO_NOT_OPEN_UNTIL_DONE.csv", dtype=str).fillna("")
lab = pd.read_csv(HG + "/labels_rehan.csv", dtype=str).fillna(""); lab["item_id"] = lab.case_id + "#" + lab.section
lab = lab.drop_duplicates("item_id")   # first save per item
key["item_id"] = key.case_id + "#" + key.section
k = key.merge(lab[["item_id", "grades", "ambiguous", "seconds", "timestamp"]].rename(columns={"grades": "hgrades"}), on="item_id", how="inner")
def parse_grades(s):
    s = str(s).strip()
    try: g = json.loads(s) if s.startswith("[") else s.split("|")
    except Exception: g = s.split("|")
    return [x.strip() for x in g if x.strip() in ORD]
k["hset"] = k.hgrades.map(parse_grades); k = k[k.hset.map(len) > 0].copy()
k["hworst"] = k.hset.map(lambda g: max(g, key=lambda x: ORD[x]))
k["correct_exact"] = (k.hworst == k.worst_grade).astype(int); k["correct_bin"] = (k.hworst.isin(POS) == k.worst_grade.isin(POS)).astype(int)
k = k.merge(pop[["sid", "anon_id", "p_sec", "p_report", "S1_votefrac", "S2_entropy", "S3_mistral_disagree", "mistral_worst"]].rename(columns={"sid": "section_id"}), on="section_id", how="left")
k["p_inclusion"] = pd.to_numeric(k.p_inclusion, errors="coerce")
chk = k[k.target == "target"]; mx = float((chk.p_inclusion - chk.p_sec).abs().max())
say(f"labelled sections {len(k)} (targets {int((k.target=='target').sum())}, extras {int((k.target=='extra').sum())}); max |p_inclusion(sample.csv) - p_sec(recomputed)| on targets = {mx:.2e}")
assert mx < 1e-9, "sampling weights could not be reconstructed"
k["w"] = np.where(k.target == "target", 1 / k.p_sec, 1 / k.p_report)       # targets: 1/p_section; extras: 1/p_report (report's inclusion probability)
k.to_csv(TRI + "/labelled_sections_with_signals.csv", index=False)
wacc = lambda col, d=k: float((d.w * d[col]).sum() / d.w.sum())
rand = k[k.stratum == "random"]
say(f"weighted population accuracy: exact {wacc('correct_exact'):.4f}, LGD+ {wacc('correct_bin'):.4f} | random slice (n={len(rand)}) exact {rand.correct_exact.mean():.4f}, LGD+ {rand.correct_bin.mean():.4f}")

# ======================= image predictions (Analysis 2 + S5) =======================
sld = pd.read_csv(ISBI + "/slides.csv", dtype={"idx": int}).fillna(""); qc_ex = set(pd.read_csv(ISBI + "/qc/exclusions.csv").idx); sld = sld[~sld.idx.isin(qc_ex)]
def load_preds(arm, task):
    """per-slide probability: human-holdout slides = mean over the 15 (fold x seed) models' holdout predictions;
    other slides = mean over 3 seeds of their own test-fold model (out-of-fold)."""
    te, ho = defaultdict(list), defaultdict(list)
    for f in glob.glob(ISBI + f"/preds/patient_{arm}_{task}_seed*_fold*.npz"):
        z = np.load(f)
        for i, p in zip(z["test_idx"], z["test_prob"]): te[int(i)].append(p)
        for i, p in zip(z["holdout_idx"], z["holdout_prob"]): ho[int(i)].append(p)
    out = {i: np.mean(v, 0) for i, v in te.items()}; out.update({i: np.mean(v, 0) for i, v in ho.items()}); return out
P6 = {a: load_preds(a, "six") for a in "AB"}; PB = {a: load_preds(a, "bin") for a in "AB"}
sld["sid"] = sld.CaseName + "#" + sld.section
def sec_agg(P, how):
    rows = {}
    for sid, g in sld.groupby("sid"):
        ps = [P[i] for i in g.idx if i in P]
        if ps: rows[sid] = (np.mean(ps, 0) if how == "mean" else np.max(ps, 0))
    return rows
IMG = {(a, t, how): sec_agg(P, how) for a, P in [("A6", P6["A"]), ("B6", P6["B"])] + [("Ab", PB["A"]), ("Bb", PB["B"])] for t in [0] for how in ("mean", "max")}
IMG = {(a[0], a[1], how): v for (a, t, how), v in IMG.items()}
def img_scores(d, arm="B", how="mean"):
    p6 = IMG[(arm, "6", how)]; pb = IMG[(arm, "b", how)]
    s = pd.DataFrame(index=d.index)
    s["has_img"] = [sid in p6 for sid in d.section_id]
    s["img_argmax6"] = [CLS6[int(np.argmax(p6[sid]))] if sid in p6 else None for sid in d.section_id]
    s["img_p_lgdplus"] = [float(pb[sid]) if sid in pb else np.nan for sid in d.section_id]
    s["img_dis_1mP"] = [1 - float(p6[sid][C_OF[g]]) if sid in p6 else np.nan for sid, g in zip(d.section_id, d.worst_grade)]
    s["img_dis_ord"] = [abs(float((p6[sid] * np.arange(6)).sum()) - C_OF[g]) if sid in p6 else np.nan for sid, g in zip(d.section_id, d.worst_grade)]
    return s

# ======================= helpers =======================
def boot_groups(pats):
    up = np.unique(pats); groups = {p: np.where(pats == p)[0] for p in up}
    for _ in range(NB): yield np.concatenate([groups[p] for p in RNG.choice(up, len(up), replace=True)])
def ci(v): v = np.asarray(v, float); return [round(float(np.nanpercentile(v, 2.5)), 4), round(float(np.nanpercentile(v, 97.5)), 4)]
def auc_ci(y, s, pats):
    pt = roc_auc_score(y, s); v = []
    for ix in boot_groups(pats):
        yy = y[ix]
        v.append(roc_auc_score(yy, s[ix]) if 0 < yy.sum() < len(yy) else np.nan)
    return {"point": round(float(pt), 4), "ci": ci(v)}
def oof_logit(X, y, groups, w=None, seed=0):
    """5-fold patient-split CV logistic regression; returns out-of-fold P(y=1) and the list of fitted fold models."""
    X = np.asarray(X, float); mu, sd = X.mean(0), X.std(0) + 1e-9; Xs = (X - mu) / sd
    oof = np.zeros(len(y)); models = []
    for tr, te in GroupKFold(5).split(Xs, y, groups):
        m = LogisticRegression(C=1.0, max_iter=1000).fit(Xs[tr], y[tr], sample_weight=None if w is None else w[tr]); oof[te] = m.predict_proba(Xs[te])[:, 1]; models.append(m)
    return oof, (lambda Xn: np.mean([m.predict_proba((np.asarray(Xn, float) - mu) / sd)[:, 1] for m in models], 0))

# ======================= ANALYSIS 2: auditor =======================
say("\n=== ANALYSIS 2 ===")
k2 = k.copy(); S_img = img_scores(k2, "B", "mean"); k2 = pd.concat([k2, S_img], axis=1); k2 = k2[k2.has_img].copy()
k2["jury_wrong"] = 1 - k2.correct_exact; k2["jury_wrong_bin"] = 1 - k2.correct_bin
n_err = int(k2.jury_wrong.sum()); say(f"sections with >=1 holdout slide: {len(k2)} of {len(k)} | jury errors (exact) among them: {n_err} | LGD+ errors: {int(k2.jury_wrong_bin.sum())}")
if n_err < 30: say("STOP: fewer than 30 jury errors with slides"); json.dump({"stop": "fewer than 30 jury errors with slides", "n_err": n_err}, open(AUD + "/STOP.json", "w")); sys.exit(3)
res2 = {"n_sections_with_slides": int(len(k2)), "n_sections_labelled": int(len(k)), "n_jury_errors_exact": n_err, "n_jury_errors_lgdplus": int(k2.jury_wrong_bin.sum()),
        "n_patients": int(k2.anon_id.nunique()), "slides_per_section_mean": round(float(sld[sld.sid.isin(k2.section_id)].groupby("sid").size().mean()), 2)}
# (a) who does the image side with?
def sides(d, arm, how, task):
    s = img_scores(d, arm, how)
    if task == "six": img = s.img_argmax6; h = d.hworst; j = d.worst_grade
    else: img = s.img_p_lgdplus.ge(0.5); h = d.hworst.isin(POS); j = d.worst_grade.isin(POS)
    out = {}
    for name, m in [("jury_wrong", (h != j).values), ("jury_correct", (h == j).values)]:
        dd = pd.DataFrame({"img": img.values[m], "h": h.values[m], "j": j.values[m]}); n = len(dd)
        if name == "jury_wrong": out[name] = {"n": n, "image_agrees_human": round(float((dd.img == dd.h).mean()), 4), "image_agrees_jury": round(float((dd.img == dd.j).mean()), 4), "neither": round(float(((dd.img != dd.h) & (dd.img != dd.j)).mean()), 4)}
        else: out[name] = {"n": n, "image_agrees_both": round(float((dd.img == dd.h).mean()), 4), "disagrees": round(float((dd.img != dd.h).mean()), 4)}
    return out
res2["a_sides"] = {f"{arm}_{how}_{task}": sides(k2, arm, how, task) for arm in "BA" for how in ("mean", "max") for task in ("six", "bin")}
# CI for the headline fraction (B, mean, bin, image agrees human among jury-wrong): patient bootstrap
s_b = img_scores(k2, "B", "mean"); hb = k2.hworst.isin(POS).values; jb = k2.worst_grade.isin(POS).values; ib = s_b.img_p_lgdplus.ge(0.5).values; m = hb != jb
agree = (ib == hb)[m].astype(float); pats_m = k2.anon_id.values[m]
res2["a_headline_bin_B_mean_image_agrees_human_among_jury_wrong"] = {"point": round(float(agree.mean()), 4), "ci": ci([agree[ix].mean() for ix in boot_groups(pats_m)]), "n": int(m.sum())}
s6 = img_scores(k2, "B", "mean"); m6 = (k2.hworst != k2.worst_grade).values; agree6 = (s6.img_argmax6.values == k2.hworst.values)[m6].astype(float)
res2["a_headline_six_B_mean_image_agrees_human_among_jury_wrong"] = {"point": round(float(agree6.mean()), 4), "ci": ci([agree6[ix].mean() for ix in boot_groups(k2.anon_id.values[m6])]), "n": int(m6.sum())}
pd.DataFrame([{"split": k_, **v} for kk, vv in res2["a_sides"].items() for k_, v in [(kk + "|" + a, b) for a, b in vv.items()]]).to_csv(AUD + "/a_image_sides.csv", index=False)
# (b) disagreement scores as jury-error detectors
y = k2.jury_wrong.values; pats = k2.anon_id.values; res2["b_auroc"] = {}
for arm in "BA":
    for how in ("mean", "max"):
        s = img_scores(k2, arm, how)
        for col in ("img_dis_1mP", "img_dis_ord"):
            res2["b_auroc"][f"{arm}_{how}_{col}"] = auc_ci(y, s[col].values, pats)
res2["b_auroc"]["votefrac_alone"] = auc_ci(y, (1 - k2.S1_votefrac).values, pats)
res2["b_auroc"]["entropy_alone"] = auc_ci(y, k2.S2_entropy.values, pats)
pd.DataFrame([{"score": kk, **v} for kk, v in res2["b_auroc"].items()]).to_csv(AUD + "/b_error_detection_auroc.csv", index=False)
# (c) added value over vote fraction: 5-fold patient-split CV, OOF AUROC, paired bootstrap
X1 = k2[["S1_votefrac"]].values; X2 = np.column_stack([k2.S1_votefrac.values, k2.img_dis_1mP.values])
oof1, _ = oof_logit(X1, y, pats); oof2, _ = oof_logit(X2, y, pats)
a1, a2 = roc_auc_score(y, oof1), roc_auc_score(y, oof2); d = []
for ix in boot_groups(pats):
    yy = y[ix]
    d.append(roc_auc_score(yy, oof2[ix]) - roc_auc_score(yy, oof1[ix]) if 0 < yy.sum() < len(yy) else np.nan)
res2["c_added_value"] = {"votefrac_oof_auroc": round(float(a1), 4), "votefrac_plus_image_oof_auroc": round(float(a2), 4), "delta": {"point": round(float(a2 - a1), 4), "ci": ci(d)}, "cv": "5-fold GroupKFold by patient, out-of-fold; bootstrap resamples patients over the fixed OOF scores (no refit)"}
X3 = np.column_stack([k2.S1_votefrac.values, k2.img_dis_ord.values]); oof3, _ = oof_logit(X3, y, pats); res2["c_added_value"]["votefrac_plus_image_ordinal_oof_auroc"] = round(float(roc_auc_score(y, oof3)), 4)
pd.DataFrame({"section_id": k2.section_id, "anon_id": pats, "jury_wrong": y, "votefrac": k2.S1_votefrac, "img_dis_1mP": k2.img_dis_1mP, "img_dis_ord": k2.img_dis_ord, "oof_votefrac": oof1, "oof_votefrac_plus_image": oof2}).to_csv(AUD + "/c_oof_scores.csv", index=False)
crit2 = (res2["c_added_value"]["delta"]["point"] >= 0.05 and res2["c_added_value"]["delta"]["ci"][0] > 0) or res2["a_headline_bin_B_mean_image_agrees_human_among_jury_wrong"]["point"] >= 0.5
res2["success_criterion"] = {"text": "delta AUROC >= 0.05 with CI excluding 0, OR image agrees with human in >= 50% of jury-wrong sections (LGD+)", "met": bool(crit2)}
res2["caveat"] = "The image model (arm B) was trained on jury labels of other patients, so it shares the jury's systematic biases; it is not an independent rater."
# figure auditor.png
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
sd = res2["a_sides"]; cats = [("B_mean_six", "6-class"), ("B_mean_bin", "LGD+ yes/no")]; x = np.arange(2); w = 0.38
for i, (kk, nm) in enumerate(cats):
    jw_ = sd[kk]["jury_wrong"]; jc = sd[kk]["jury_correct"]
    ax[0].bar(i - w / 2, jw_["image_agrees_human"], w, color="#2563eb", label="agrees with human" if i == 0 else None)
    ax[0].bar(i - w / 2, jw_["image_agrees_jury"], w, bottom=jw_["image_agrees_human"], color="#f59e0b", label="agrees with jury" if i == 0 else None)
    ax[0].bar(i - w / 2, jw_["neither"], w, bottom=jw_["image_agrees_human"] + jw_["image_agrees_jury"], color="#9ca3af", label="neither" if i == 0 else None)
    ax[0].bar(i + w / 2, jc["image_agrees_both"], w, color="#059669", label="agrees (jury = human)" if i == 0 else None)
    ax[0].bar(i + w / 2, jc["disagrees"], w, bottom=jc["image_agrees_both"], color="#9ca3af")
    ax[0].text(i - w / 2, 1.02, f"jury wrong\nn={jw_['n']}", ha="center", fontsize=8); ax[0].text(i + w / 2, 1.02, f"jury right\nn={jc['n']}", ha="center", fontsize=8)
ax[0].set_xticks(x); ax[0].set_xticklabels([c[1] for c in cats]); ax[0].set_ylim(0, 1.3); ax[0].set_ylabel("fraction of sections"); ax[0].legend(fontsize=7, loc="upper center", ncol=2, frameon=False); ax[0].set_title("(a) Image model (arm B) vs human and jury", fontsize=10)
for s_, nm, col in [(oof1, f"vote fraction (AUROC {a1:.3f})", "#f59e0b"), (oof2, f"vote fraction + image (AUROC {a2:.3f})", "#2563eb")]:
    fp, tp, _ = roc_curve(y, s_); ax[1].plot(fp, tp, color=col, label=nm)
ax[1].plot([0, 1], [0, 1], "k:", lw=0.8); ax[1].set_xlabel("false positive rate"); ax[1].set_ylabel("true positive rate"); ax[1].legend(fontsize=8); ax[1].set_title(f"(b) Detecting jury errors (n={n_err} of {len(k2)}), out-of-fold", fontsize=10)
plt.tight_layout(); plt.savefig(AUD + "/auditor.png", dpi=160); plt.savefig(AUD + "/auditor.pdf"); plt.close()
json.dump(res2, open(AUD + "/results.json", "w"), indent=1)
say("A2 (a) B/mean: six", res2["a_sides"]["B_mean_six"]["jury_wrong"], "| bin", res2["a_sides"]["B_mean_bin"]["jury_wrong"])
say("A2 (b)", {kk: v["point"] for kk, v in res2["b_auroc"].items()}); say("A2 (c)", res2["c_added_value"]); say("A2 criterion met:", crit2)

# ======================= ANALYSIS 1: triage =======================
say("\n=== ANALYSIS 1 ===")
# S4: logistic P(jury wrong) ~ S1 + S2 + S3 on the labelled set (weighted), 5-fold patient CV
Xl = k[["S1_votefrac", "S2_entropy", "S3_mistral_disagree"]].values; yl = (1 - k.correct_exact).values; gl = k.anon_id.values; wl = k.w.values / k.w.mean()
oof4, pred4 = oof_logit(Xl, yl, gl, wl); k["S4_pwrong_oof"] = oof4; pop["S4_pwrong"] = pred4(pop[["S1_votefrac", "S2_entropy", "S3_mistral_disagree"]].values)
say("S4 OOF AUROC for jury-wrong on labelled set:", round(roc_auc_score(yl, oof4), 4))
# S5: S4 features + image disagreement (arm B, mean) where a slide exists; fitted on labelled sections with images, patient CV; population without image -> S4
simg_lab = img_scores(k, "B", "mean"); k["img_dis_1mP"] = simg_lab.img_dis_1mP.values; k["has_img"] = simg_lab.has_img.values
simg_pop = img_scores(pop.rename(columns={"sid": "section_id"}), "B", "mean"); pop["img_dis_1mP"] = simg_pop.img_dis_1mP.values; pop["has_img"] = simg_pop.has_img.values
mi = k.has_img.values; X5 = np.column_stack([Xl[mi], k.img_dis_1mP.values[mi]]); oof5, pred5 = oof_logit(X5, yl[mi], gl[mi], wl[mi])
k["S5_pwrong_oof"] = np.nan; k.loc[k.has_img, "S5_pwrong_oof"] = oof5; k.loc[~k.has_img, "S5_pwrong_oof"] = k.loc[~k.has_img, "S4_pwrong_oof"]
pop["S5_pwrong"] = pop.S4_pwrong; pm = pop.has_img.values; pop.loc[pm, "S5_pwrong"] = pred5(np.column_stack([pop[["S1_votefrac", "S2_entropy", "S3_mistral_disagree"]].values[pm], pop.img_dis_1mP.values[pm]]))
say(f"image coverage: labelled {int(mi.sum())}/{len(k)}, population {int(pm.sum())}/{N} | S5 OOF AUROC (sections with image): {roc_auc_score(yl[mi], oof5):.4f}")
# ranking value: LOWER = review first
SIG = {"S0_random": None, "S1_votefrac": "S1_votefrac", "S2_entropy": "S2_entropy", "S3_mistral": "S3_mistral_disagree", "S4_combined": "S4_pwrong", "S5_text_image": "S5_pwrong"}
pop["S0_random"] = RNG.rand(N); k["S0_random"] = RNG.rand(len(k))
def rank_value(d, name, oof=False):
    if name == "S0_random": return d.S0_random.values
    if name == "S1_votefrac": return d.S1_votefrac.values
    if name == "S2_entropy": return -d.S2_entropy.values                                  # high entropy first
    if name == "S3_mistral": return -d.S3_mistral_disagree.values * 10 + d.S1_votefrac.values   # disagreement first, ties by vote fraction
    if name == "S4_combined": return -(d.S4_pwrong_oof.values if oof else d.S4_pwrong.values)
    if name == "S5_text_image": return -(d.S5_pwrong_oof.values if oof else d.S5_pwrong.values)
def triage_curve(name, task, d_lab, d_pop, weighted=True, oof=True):
    """ONE estimator for every signal (v2, 2026-10-08): population accuracy at budget b = weighted mean over labelled sections of
    [reviewed_i OR correct_i], reviewed_i = the section's ranking value is below the population quantile threshold for budget b
    (lowest-confidence first). At b = 0 this is exactly the direct weighted baseline for every signal. Reports opened / hours come
    from the population ordering. Bootstrap = patient resampling of the labelled set (thresholds fixed)."""
    col = "correct_exact" if task == "exact" else "correct_bin"
    vl = rank_value(d_lab, name, oof); vp = rank_value(d_pop, name)
    w = d_lab.w.values if weighted else np.ones(len(d_lab)); c = d_lab[col].values.astype(float); pats = d_lab.anon_id.values
    order = np.lexsort((RNG.rand(len(vp)), vp)); vps = vp[order]; rep_sorted = d_pop.CaseName.values[order]
    n_rev = np.round(BUDGETS * len(vp)).astype(int)
    thr = np.array([(-np.inf if n == 0 else vps[n - 1]) for n in n_rev])                 # review everything with value <= thr (ties included)
    opened = np.zeros(len(vp) + 1, int); seen = set()
    for j, r_ in enumerate(rep_sorted): seen.add(r_); opened[j + 1] = len(seen)
    frac_reports = opened[n_rev] / d_pop.CaseName.nunique()
    REV = (vl[None, :] <= thr[:, None])                                                  # (budgets, labelled)
    Z = np.maximum(REV, c[None, :])                                                      # reviewed -> correct
    def curve(ix): ww = w[ix]; return (Z[:, ix] * ww).sum(1) / ww.sum()
    pt = curve(np.arange(len(vl))); bs = np.stack([curve(ix) for ix in boot_groups(pats)])
    nrep = d_pop.CaseName.nunique()
    return {"budget_sections": BUDGETS, "frac_reports": frac_reports, "hours": frac_reports * nrep * MED_REPORT_S / 3600, "hours_mean": frac_reports * nrep * MEAN_REPORT_S / 3600, "point": pt, "lo": np.percentile(bs, 2.5, 0), "hi": np.percentile(bs, 97.5, 0), "boot": bs}
# median time per report from the app (sum of section seconds per report id, first pass incl. repeats as separate reports)
lab_all = pd.read_csv(HG + "/labels_rehan.csv", dtype=str).fillna(""); per_rep = lab_all.groupby("case_id").seconds.apply(lambda s: pd.to_numeric(s, errors="coerce").sum())
MED_REPORT_S = float(per_rep.median()); MEAN_REPORT_S = float(per_rep.mean()); say(f"median seconds per report (app, n={len(per_rep)}): {MED_REPORT_S:.1f}; mean {MEAN_REPORT_S:.1f}")
res1 = {"estimator": "v2 Horvitz-Thompson: weighted mean of [reviewed or correct], thresholds from population quantiles; anchored at the direct weighted baseline for every signal", "population": {"sections": N, "reports": int(pop.CaseName.nunique()), "patients": int(pop.anon_id.nunique())}, "median_seconds_per_report": MED_REPORT_S, "mean_seconds_per_report": MEAN_REPORT_S,
        "weights": "targets: 1/p_section where p_section = 1-(1-50/N)(1-n_cell/N_cell)prod_tags(1-n_tag/N_tag) (recomputed for every population section and verified equal to sample.csv p_inclusion); extras: 1/p_report with p_report = 1-prod over the report's population sections (1-p_section)",
        "baseline_accuracy_weighted": {"exact": wacc("correct_exact"), "lgdplus": wacc("correct_bin")}, "curves": {}, "thresholds": {}, "at_budget": {}}
long_rows = []; summary_rows = []
for task in ("exact", "lgdplus"):
    res1["curves"][task] = {}
    for name in SIG:
        cv = triage_curve(name, task, k, pop); res1["curves"][task][name] = cv
        for i, b in enumerate(BUDGETS):
            long_rows.append({"task": task, "signal": name, "budget_sections": round(b, 2), "frac_reports": round(float(cv["frac_reports"][i]), 4), "hours_median": round(float(cv["hours"][i]), 2), "hours_mean": round(float(cv["hours_mean"][i]), 2), "accuracy": round(float(cv["point"][i]), 4), "lo": round(float(cv["lo"][i]), 4), "hi": round(float(cv["hi"][i]), 4)})
        row = {"task": task, "signal": name}
        for b in (0, 5, 10, 20, 30):
            i = int(b); row[f"acc@{b}%"] = f"{cv['point'][i]:.3f} [{cv['lo'][i]:.3f}, {cv['hi'][i]:.3f}]"; row[f"reports@{b}%"] = round(float(cv["frac_reports"][i]), 3)
        for thr in (0.95, 0.97, 0.98):
            hit = np.where(cv["point"] >= thr)[0]; hit_lo = np.where(cv["lo"] >= thr)[0]
            row[f"budget_for_{int(thr*100)}%_sections"] = (round(float(BUDGETS[hit[0]]), 2) if len(hit) else ">0.50")
            row[f"budget_for_{int(thr*100)}%_reports"] = (round(float(cv["frac_reports"][hit[0]]), 3) if len(hit) else ">")
            row[f"budget_for_{int(thr*100)}%_hours_median"] = (round(float(cv["hours"][hit[0]]), 2) if len(hit) else ">")
            row[f"budget_for_{int(thr*100)}%_hours_mean"] = (round(float(cv["hours_mean"][hit[0]]), 2) if len(hit) else ">")
            row[f"budget_for_{int(thr*100)}%_sections_CIlow"] = (round(float(BUDGETS[hit_lo[0]]), 2) if len(hit_lo) else ">0.50")
        summary_rows.append(row)
pd.DataFrame(long_rows).to_csv(TRI + "/triage_curves_long.csv", index=False); pd.DataFrame(summary_rows).to_csv(TRI + "/triage_summary.csv", index=False)
# S4 vs S1 at 20% (paired, same bootstrap draws impossible across signals -> use common patient resamples): recompute paired
def paired_delta(nameA, nameB, task, idx=20):
    col = "correct_exact" if task == "exact" else "correct_bin"
    cA = triage_curve(nameA, task, k, pop); st = RNG.get_state(); RNG.set_state(st)
    # paired: regenerate both curves with identical resamples
    seed = 777; outs = []
    for nm in (nameA, nameB):
        RNG.seed(seed); outs.append(triage_curve(nm, task, k, pop))
    RNG.seed(20261007 + idx)
    dpt = outs[1]["point"][idx] - outs[0]["point"][idx]; db = outs[1]["boot"][:, idx] - outs[0]["boot"][:, idx]
    return {"point": round(float(dpt), 4), "ci": ci(db)}
res1["S4_minus_S1_at_20pct"] = {t: paired_delta("S1_votefrac", "S4_combined", t) for t in ("exact", "lgdplus")}
res1["S5_minus_S1_at_20pct"] = {t: paired_delta("S1_votefrac", "S5_text_image", t) for t in ("exact", "lgdplus")}
res1["S5_minus_S4_at_20pct"] = {t: paired_delta("S4_combined", "S5_text_image", t) for t in ("exact", "lgdplus")}
# sensitivity: S1 on the random slice only, unweighted
RNG.seed(20261007)
cvr = {t: triage_curve("S1_votefrac", t, rand, pop, weighted=False) for t in ("exact", "lgdplus")}
res1["sensitivity_random_slice_S1"] = {t: {f"{b}%": f"{cvr[t]['point'][b]:.3f} [{cvr[t]['lo'][b]:.3f}, {cvr[t]['hi'][b]:.3f}]" for b in (0, 10, 20, 30)} for t in cvr}
# dominance + criterion
acc20 = {nm: res1["curves"]["exact"][nm]["point"][20] for nm in SIG}; res1["best_signal_at_20pct_exact"] = max(acc20, key=acc20.get)
def reports_to_97(nm):
    cv = res1["curves"]["exact"][nm]; hit = np.where(cv["point"] >= 0.97)[0]; return float(cv["frac_reports"][hit[0]]) if len(hit) else None
r97 = {nm: reports_to_97(nm) for nm in ("S1_votefrac", "S4_combined")}
res1["success_criterion"] = {"text": "reviewing <=20% of reports reaches >=97% population exact accuracy with S1 or S4", "reports_fraction_to_97": r97, "met": bool(any(v is not None and v <= 0.20 for v in r97.values()))}
# figure
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.3)); cols = {"S0_random": "#9ca3af", "S1_votefrac": "#2563eb", "S2_entropy": "#7c3aed", "S3_mistral": "#059669", "S4_combined": "#f59e0b", "S5_text_image": "#dc2626"}
for a, task in zip(ax, ("exact", "lgdplus")):
    for nm in SIG:
        cv = res1["curves"][task][nm]; x = 100 * cv["frac_reports"]; a.plot(x, 100 * cv["point"], color=cols[nm], label=nm); a.fill_between(x, 100 * cv["lo"], 100 * cv["hi"], color=cols[nm], alpha=0.12)
    for thr in (95, 97, 98): a.axhline(thr, color="k", lw=0.6, ls=":"); a.text(0.5, thr + 0.15, f"{thr}%", fontsize=7)
    a.axvline(20, color="k", lw=0.6, ls="--"); a.set_xlabel("% of reports opened"); a.set_ylabel("estimated population accuracy (%)"); a.set_title(f"{'exact grade' if task=='exact' else 'LGD+ yes/no'} — human reviews lowest-confidence first", fontsize=10); a.set_xlim(0, 100 * max(res1["curves"][task][n]["frac_reports"][-1] for n in SIG))
ax[0].legend(fontsize=7, loc="lower right"); plt.tight_layout(); plt.savefig(TRI + "/triage_curve.png", dpi=160); plt.savefig(TRI + "/triage_curve.pdf"); plt.close()
for task in res1["curves"]:
    for nm in res1["curves"][task]: res1["curves"][task][nm] = {kk: (v.round(4).tolist() if kk != "boot" else None) for kk, v in res1["curves"][task][nm].items()}
json.dump(res1, open(TRI + "/results.json", "w"), indent=1)
pop.drop(columns=[c for c in pop.columns if c in ("grades", "grade_votes", "site_votes", "subtype_votes", "cancer_subtypes", "unanimous_grades", "specimen_type", "spec_frac", "site_frac")]).to_csv(TRI + "/population_signals.csv", index=False)
say("A1 summary:"); say(pd.DataFrame(summary_rows)[["task", "signal", "acc@0%", "acc@10%", "acc@20%", "budget_for_97%_sections", "budget_for_97%_reports", "budget_for_97%_hours_median", "budget_for_97%_hours_mean"]].to_string())
say("S4-S1 @20%:", res1["S4_minus_S1_at_20pct"], "| S5-S1 @20%:", res1["S5_minus_S1_at_20pct"]); say("sensitivity random slice S1:", res1["sensitivity_random_slice_S1"]); say("A1 criterion:", res1["success_criterion"])
open(OUT + "/run12_log.txt", "w").write("\n".join(log))
for root, _, files in os.walk(OUT):
    for f in files: os.chmod(os.path.join(root, f), 0o600)

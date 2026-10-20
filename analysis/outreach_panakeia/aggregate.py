"""CPU unit: read preds/*.csv -> results/main.json (every number used in RESULTS.md / one-pager), figures 1 and 3.
Bootstrap: 2,000 patient resamples as count weights; vectorised weighted-rank AUROC `wauc` copied from
analysis/tilebudget/code/aggregate_tb.py (one slide per patient here, so slide = patient). Decision rules: DESIGN.md."""
import glob, json, os
import numpy as np, pandas as pd
from scipy.stats import spearmanr, rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

A = "/mnt/scratche/slow/fmlab/zuberi01/analysis/outreach_panakeia"; PR = A + "/preds"; NB = 2000
os.makedirs(A + "/figures", exist_ok=True)
c = pd.read_csv(A + "/cohort.csv").set_index("patient"); folds = pd.read_csv(A + "/folds.csv").set_index("patient")
ix = json.load(open(A + "/pack/index.json")); ntile = dict(zip(ix["patients"], ix["n_tiles"]))
usable = [p for p in c.index if ntile.get(p, 0) > 0]


def wauc(score, y, W):
    o = np.argsort(score, kind="stable"); yo = y[o].astype(np.float32); Wo = W[:, o]; neg = Wo * (1 - yo)
    cumneg = np.cumsum(neg, axis=1) - neg; num = (Wo * yo * cumneg).sum(1); den = (Wo * yo).sum(1) * neg.sum(1)
    return np.where(den > 0, num / np.maximum(den, 1e-9), np.nan)


def weights(n, seed=20261019):
    r = np.random.RandomState(seed); return np.stack([np.bincount(r.randint(0, n, n), minlength=n) for _ in range(NB)]).astype(np.float32)


def ci(v): v = np.asarray(v, float); return [round(float(np.nanpercentile(v, 2.5)), 4), round(float(np.nanpercentile(v, 97.5)), 4)]


def wspearman(a, b, W):
    """weighted Pearson on ranks for each bootstrap row (ranks from the full sample; adequate for a CI)"""
    ra, rb = rankdata(a), rankdata(b); Wn = W / W.sum(1, keepdims=True)
    ma, mb = Wn @ ra, Wn @ rb; ca, cb = ra[None] - ma[:, None], rb[None] - mb[:, None]
    return (Wn * ca * cb).sum(1) / np.sqrt((Wn * ca ** 2).sum(1) * (Wn * cb ** 2).sum(1))


def load(task, split, model):
    fs = sorted(glob.glob(f"{PR}/{task}_{split}_{model}_r*.csv"))
    if not fs: return None, []
    d = [pd.read_csv(f).set_index("patient") for f in fs]
    avg = pd.concat([x.score for x in d], axis=1).mean(1)
    return pd.DataFrame({"score": avg, "y": d[0].y, "cohort": d[0].cohort}), d


def site_only(task, col):
    """logistic regression on one-hot tissue source site + cohort, same strat folds, OOF averaged over repeats"""
    lab = c.loc[[p for p in usable if pd.notna(c.loc[p, col])]]
    Xs = pd.get_dummies(lab.site.astype(str) + "|" + lab.cohort).astype(float); Xs["esca"] = (lab.cohort == "esca").astype(float)
    outs = []
    for r in range(3):
        fc = folds.loc[lab.index, f"{task}_strat_r{r}"]; s = pd.Series(np.nan, index=lab.index)
        for k in range(5):
            tr, te = fc != k, fc == k
            m = LogisticRegression(C=1.0, max_iter=2000).fit(Xs[tr], lab.loc[tr, col].astype(int))
            s[te] = m.predict_proba(Xs[te])[:, 1]
        outs.append(s)
    return pd.concat(outs, axis=1).mean(1), lab[col].astype(int)


R = {"design": "DESIGN.md (pre-registered 2026-10-19)", "n_bootstrap": NB, "tasks": {}}
TASKS = [("msi", "MSI-H vs MSS (STAD; positive control)", "msi"), ("wgd", "Whole-genome doubling (primary)", "wgd"),
         ("cin", "CIN vs non-CIN subtype", "cin"), ("aneu", "Aneuploidy score (Spearman)", "aneuploidy")]
for task, title, col in TASKS:
    T = {"title": title}
    for split in ("strat", "site"):
        base, _ = load(task, split, "abmil")
        if base is None: continue
        keys = base.index; y = base.y.values; W = weights(len(keys)); S = {}
        for model in ("abmil", "meanlr", "shuf"):
            d, reps = load(task, split, model)
            if d is None: continue
            d = d.loc[keys]; s = d.score.values
            if task == "aneu":
                bs = wspearman(s, y, W); pt = float(spearmanr(s, y).correlation); per = [round(float(spearmanr(x.loc[keys].score, y).correlation), 4) for x in reps]
            else:
                bs = wauc(s, y.astype(int), W); pt = float(roc_auc_score(y, s)); per = [round(float(roc_auc_score(x.loc[keys].y, x.loc[keys].score)), 4) for x in reps]
            S[model] = bs
            T[f"{split}_{model}"] = {"point": round(pt, 4), "ci": ci(bs), "per_repeat": per, "n": int(len(keys)),
                                     **({} if task == "aneu" else {"n_pos": int(y.sum())})}
            if task != "aneu" and split == "strat" and model in ("abmil", "meanlr"):
                for coh in ("stad", "esca"):
                    m = (d.cohort == coh).values
                    if m.sum() > 10 and 0 < y[m].sum() < m.sum():
                        Wm = weights(int(m.sum()), seed=7); T[f"{split}_{model}_{coh}_only"] = {
                            "point": round(float(roc_auc_score(y[m], s[m])), 4), "ci": ci(wauc(s[m], y[m].astype(int), Wm)),
                            "n": int(m.sum()), "n_pos": int(y[m].sum())}
        for a, b in (("abmil", "meanlr"), ("abmil", "shuf")):
            if a in S and b in S:
                dlt = S[a] - S[b]; T[f"{split}_delta_{a}_vs_{b}"] = {"point": round(float(T[f"{split}_{a}"]["point"] - T[f"{split}_{b}"]["point"]), 4), "ci": ci(dlt)}
        if split == "strat" and task != "aneu":
            so, ys = site_only(task, col); so = so.loc[keys]; T["strat_site_only"] = {
                "point": round(float(roc_auc_score(ys.loc[keys], so)), 4), "ci": ci(wauc(so.values, ys.loc[keys].values, W)), "n": int(len(keys))}
    R["tasks"][task] = T
# cross-cohort WGD (train STAD, test ESCA-adeno)
X = {}
for model in ("abmil", "meanlr", "shuf"):
    d, reps = load("wgdx", "x", model)
    if d is None: continue
    W = weights(len(d), seed=11); X[model] = {"point": round(float(roc_auc_score(d.y, d.score)), 4), "ci": ci(wauc(d.score.values, d.y.values.astype(int), W)),
                                              "per_seed": [round(float(roc_auc_score(x.y, x.score)), 4) for x in reps], "n": int(len(d)), "n_pos": int(d.y.sum())}
R["tasks"]["wgdx"] = {"title": "WGD: train STAD, test ESCA adenocarcinoma", **X}

# ---- pre-registered decisions ----
t = R["tasks"]; dec = {}
msi = t["msi"].get("strat_abmil")
dec["positive_control_pass"] = bool(msi and msi["point"] >= 0.75 and msi["ci"][0] > 0.5)
w = t["wgd"]; wa = w.get("strat_abmil"); ws = w.get("strat_delta_abmil_vs_shuf")
pos = bool(wa and wa["ci"][0] > 0.5 and ws and ws["ci"][0] > 0)
robust = bool(pos and w.get("site_abmil", {}).get("ci", [0])[0] > 0.5 and w.get("strat_abmil_stad_only", {}).get("ci", [0])[0] > 0.5)
cov = len(usable) / len(c)
dec["feature_coverage"] = round(cov, 4)
dec["wgd_call"] = ("failed" if (not dec["positive_control_pass"] or cov < 0.9) else
                   "positive (robust)" if robust else "positive but possibly confounded by site/organ" if pos else "null")
R["decisions"] = dec
json.dump(R, open(A + "/results/main.json", "w"), indent=1)
print(json.dumps(dec, indent=1))

# ---- Figure 1: AUROC per task ----
plt.rcParams.update({"font.size": 7, "axes.spines.top": False, "axes.spines.right": False, "font.family": "DejaVu Sans"})
INK, INK2 = "#0b0b0b", "#52514e"
MODELS = [("strat_abmil", "ABMIL", "#2a78d6", "o"), ("strat_meanlr", "Mean-pool LR", "#eb6834", "s"),
          ("strat_shuf", "Shuffled-label ABMIL", "#8a8984", "D"), ("strat_site_only", "Site-only LR", "#1baf7a", "^"),
          ("site_abmil", "ABMIL, site-held-out CV", "#4a3aa7", "v")]
rows = [("msi", "MSI-H vs MSS\n(STAD, control)"), ("wgd", "WGD\n(primary)"), ("cin", "CIN subtype"), ("wgdx", "WGD\nSTAD→ESCA")]
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(3.6, 2.6), gridspec_kw={"width_ratios": [4, 1.3], "wspace": 0.45})
for i, (task, lab_) in enumerate(rows):
    for j, (key, name, colr, mk) in enumerate(MODELS):
        k2 = {"strat_abmil": "abmil", "strat_meanlr": "meanlr", "strat_shuf": "shuf"}.get(key) if task == "wgdx" else key
        r = t[task].get(k2) if k2 else None
        if not r: continue
        yy = i + (j - 2) * 0.15
        ax.errorbar(r["point"], yy, xerr=[[r["point"] - r["ci"][0]], [r["ci"][1] - r["point"]]], fmt=mk, ms=4, color=colr, ecolor=colr, elinewidth=1.2, capsize=0,
                    label=name if i == 0 or (task == "wgd" and key == "site_abmil") else None)
ax.axvline(0.5, color=INK2, lw=0.6, ls=(0, (2, 2)))
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[1] for r in rows]); ax.invert_yaxis(); ax.set_xlim(0.3, 1.0); ax.set_xlabel("AUROC (95% CI)")
h, l = ax.get_legend_handles_labels(); seen = {}
for hh, ll in zip(h, l): seen.setdefault(ll, hh)
ax.legend(seen.values(), seen.keys(), frameon=False, fontsize=5.5, loc="lower center", bbox_to_anchor=(0.6, 1.0), ncol=2, handletextpad=0.3, columnspacing=0.8)
for j, (key, name, colr, mk) in enumerate(MODELS[:3]):
    r = t["aneu"].get(key)
    if r: ax2.errorbar(j, r["point"], yerr=[[r["point"] - r["ci"][0]], [r["ci"][1] - r["point"]]], fmt=mk, ms=4, color=colr, elinewidth=1.2)
ax2.axhline(0, color=INK2, lw=0.6, ls=(0, (2, 2))); ax2.set_xticks([0, 1, 2]); ax2.set_xticklabels(["ABMIL", "Mean", "Shuf."], rotation=45, fontsize=6)
ax2.set_ylabel("Spearman ρ (95% CI)"); ax2.set_title("Aneuploidy\nscore", fontsize=7); ax2.set_xlim(-0.6, 2.6)
for f in ("pdf", "png"): fig.savefig(f"{A}/figures/fig1_auroc.{f}", bbox_inches="tight", dpi=300)

# ---- Figure 3: cohort / label flow ----
fl = json.load(open(A + "/results/cohort_flow.json"))
ex = {s: sum(1 for p in c.index[c.cohort == s] if ntile.get(p, 0) == 0) for s in ("stad", "esca")}
lab_n = {k: {s: int(c.loc[[p for p in usable if c.loc[p, "cohort"] == s], col].notna().sum()) for s in ("stad", "esca")} for k, col in
         (("wgd", "wgd"), ("cin", "cin"), ("msi", "msi"), ("aneu", "aneuploidy"))}
pos_n = {k: {s: int(c.loc[[p for p in usable if c.loc[p, "cohort"] == s], col].sum()) for s in ("stad", "esca")} for k, col in (("wgd", "wgd"), ("cin", "cin"), ("msi", "msi"))}
R["flow"] = {**fl, "excluded_no_tiles": ex, "usable": {s: int(sum(c.loc[usable, "cohort"] == s)) for s in ("stad", "esca")}, "labelled": lab_n, "positives": pos_n}
json.dump(R, open(A + "/results/main.json", "w"), indent=1)
fig, ax = plt.subplots(figsize=(3.6, 2.6)); ax.axis("off"); ax.set_xlim(-0.1, 10.1); ax.set_ylim(-0.2, 10)
def box(x, y, w, h, txt, fc="#f0efec"):
    ax.add_patch(plt.Rectangle((x, y), w, h, fc=fc, ec=INK2, lw=0.6)); ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=4.6, color=INK, linespacing=1.3)
def arr(x1, y1, x2, y2): ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="-|>", lw=0.6, color=INK2, shrinkA=0, shrinkB=0, mutation_scale=6))
box(0.1, 8.0, 4.6, 1.8, f"TCGA-STAD: {fl['stad_patients_with_dx_slide']} patients\nwith a diagnostic slide ({fl['stad_dx_slides']} slides)")
box(5.3, 8.0, 4.6, 1.8, f"TCGA-ESCA: {fl['esca_patients_with_dx_slide']} patients\nwith a diagnostic slide ({fl['esca_dx_slides']} slides)")
box(5.3, 5.3, 4.6, 1.8, f"Adenocarcinoma: {fl['esca_adeno_cohort']}\n(squamous dropped: {fl['esca_squamous_dropped']})")
box(0.1, 5.3, 4.6, 1.8, f"Adenocarcinoma with PanCanAtlas\nrecord: {fl['stad_adeno_cohort']}")
box(0.1, 2.6, 9.8, 1.8, f"One slide per patient (DX1); UNI2-h at 20x: {len(usable)} usable\n"
    f"(STAD {R['flow']['usable']['stad']}, ESCA {R['flow']['usable']['esca']}; {ex['stad'] + ex['esca']} slides unusable)", fc="#cde2fb")
lbl = (f"WGD {lab_n['wgd']['stad'] + lab_n['wgd']['esca']} ({pos_n['wgd']['stad'] + pos_n['wgd']['esca']} +)   "
       f"CIN {lab_n['cin']['stad'] + lab_n['cin']['esca']} ({pos_n['cin']['stad'] + pos_n['cin']['esca']} +)\n"
       f"Aneuploidy {lab_n['aneu']['stad'] + lab_n['aneu']['esca']}   MSI (STAD) {lab_n['msi']['stad']} ({pos_n['msi']['stad']} MSI-H)   "
       f"STAD→ESCA WGD test {lab_n['wgd']['esca']}")
box(0.1, 0.0, 9.8, 1.9, lbl)
arr(2.4, 8.0, 2.4, 7.1); arr(7.6, 8.0, 7.6, 7.1); arr(2.4, 5.3, 2.4, 4.4); arr(7.6, 5.3, 7.6, 4.4); arr(5.0, 2.6, 5.0, 1.9)
for f in ("pdf", "png"): fig.savefig(f"{A}/figures/fig3_flow.{f}", bbox_inches="tight", dpi=300)
print("figures written")

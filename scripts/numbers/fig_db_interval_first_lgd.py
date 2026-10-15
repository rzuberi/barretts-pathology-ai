#!/usr/bin/env python3
"""Item 12 (24 Sep 2026): inter-biopsy interval before vs after a first LGD, Barrett's DB report corpus
(jury grades), two boxplots of per-patient median intervals + pooled intervals. Same rules as the 23 Sep
analysis: >=3 dated graded reports, one visit per day (worst grade), no HGD/cancer before the LGD,
after-window cut at the first HGD/cancer, >=1 interval on both sides."""
import glob, json, os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, mannwhitneyu
E = "/mnt/scratche/slow/fmlab/zuberi01/barretts_db_export"; T = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"; OUT = os.environ.get("OUTDIR", ".")
ORD = {"NDBE": 0, "IND": 1, "LGD": 2, "HGD": 3, "CANCER": 4}
pt = pd.read_csv(f"{E}/pathology_text_normalised_full.csv", dtype=str, usecols=["pathology_text_id", "participant_id", "receiveddatetime"])
pt["d"] = pd.to_datetime(pt.receiveddatetime, dayfirst=True, errors="coerce")
votes = {}
for f in glob.glob(f"{T}/feasibility/runs/jury_full_*/output/llm_grades_*.csv"):
    d = pd.read_csv(f, dtype=str, on_bad_lines="skip"); d = d[d.llm_grade.isin(ORD)]
    for c, g in zip(d.CaseName, d.llm_grade): votes.setdefault(str(c), []).append(g)
lab = {c: max(set(v), key=v.count) for c, v in votes.items() if len(v) >= 4}
pt["g"] = pt.pathology_text_id.astype(str).map(lab).map(ORD); pt = pt[pt.d.notna() & pt.g.notna()].sort_values(["participant_id", "d"])
rows = []; pb = []; pa = []
for pid, g in pt.groupby("participant_id"):
    g = g.groupby("d", as_index=False).g.max()
    if len(g) < 3: continue
    gg = g.g.values; d = g.d.values; idx = np.where(gg == 2)[0]
    if not len(idx): continue
    i = idx[0]
    if (gg[:i] >= 3).any(): continue
    gaps = np.diff(d).astype("timedelta64[D]").astype(int); before = gaps[:i]; after = gaps[i:]
    later = np.where(gg[i + 1:] >= 3)[0]
    if len(later): after = after[:later[0] + 1]
    if len(before) == 0 or len(after) == 0: continue
    pb += list(before); pa += list(after); rows.append({"mb": float(np.median(before)), "ma": float(np.median(after)), "year": int(pd.Timestamp(d[i]).year)})
r = pd.DataFrame(rows); pb = np.array(pb); pa = np.array(pa)
summ = {"source_reports": f"{E}/pathology_text_normalised_full.csv", "source_grades": f"{T}/feasibility/runs/jury_full_*/output/llm_grades_*.csv",
        "event": "first report graded exactly LGD (jury majority, >=4 votes)", "patients": len(r), "intervals_before": int(len(pb)), "intervals_after": int(len(pa)),
        "per_patient_median_before_days": {"median": float(r.mb.median()), "q25": float(r.mb.quantile(.25)), "q75": float(r.mb.quantile(.75))},
        "per_patient_median_after_days": {"median": float(r.ma.median()), "q25": float(r.ma.quantile(.25)), "q75": float(r.ma.quantile(.75))},
        "wilcoxon_paired_p": float(wilcoxon(r.mb, r.ma).pvalue), "frac_shorter_after": float((r.ma < r.mb).mean()),
        "pooled_before_percentiles_10_25_50_75_90": np.percentile(pb, [10, 25, 50, 75, 90]).round(0).tolist(),
        "pooled_after_percentiles_10_25_50_75_90": np.percentile(pa, [10, 25, 50, 75, 90]).round(0).tolist(), "mannwhitney_pooled_p": float(mannwhitneyu(pb, pa).pvalue),
        "by_era": r.groupby(pd.cut(r.year, [1990, 2005, 2010, 2015, 2020, 2026]), observed=True).agg(n=("mb", "size"), before=("mb", "median"), after=("ma", "median")).reset_index().astype(str).to_dict("records")}
fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
axes[0].boxplot([r.mb, r.ma], labels=["before first LGD", "after first LGD"], showfliers=False, widths=0.55)
axes[0].set_ylabel("per-patient median interval between biopsies (days)"); axes[0].set_title(f"Per patient (n = {len(r)}); Wilcoxon p = {summ['wilcoxon_paired_p']:.3g}", fontsize=9)
axes[0].axhline(365, ls="--", lw=0.7, color="grey"); axes[0].axhline(182, ls=":", lw=0.7, color="grey")
axes[1].boxplot([pb, pa], labels=[f"before (n={len(pb)})", f"after (n={len(pa)})"], showfliers=False, widths=0.55)
axes[1].set_ylabel("interval (days)"); axes[1].set_title(f"All intervals pooled; Mann-Whitney p = {summ['mannwhitney_pooled_p']:.2g}", fontsize=9)
axes[1].axhline(365, ls="--", lw=0.7, color="grey"); axes[1].axhline(182, ls=":", lw=0.7, color="grey")
fig.suptitle("Barrett's DB: surveillance interval before vs after a first LGD report (1995-2026, jury grades)", fontsize=10); fig.tight_layout()
os.makedirs(OUT, exist_ok=True); fig.savefig(os.path.join(OUT, "fig_db_interval_first_lgd.png"), dpi=160); json.dump(summ, open(os.path.join(OUT, "db_interval_first_lgd.json"), "w"), indent=1, default=str)
print(json.dumps(summ, indent=None, default=str))

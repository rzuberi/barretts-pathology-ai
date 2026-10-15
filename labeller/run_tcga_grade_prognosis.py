"""Do pathladder-extracted grades stratify survival like clinically recorded grades?

Uses yesterday's per-case labels (tcga_validation_per_case.csv) + TCGA-CDR overall
survival. For each grade source (pathladder vs clinical record): Kaplan-Meier by
G3 vs G1/G2, log-rank test, and univariate Cox HR. If the weak labels reproduce the
recorded labels' stratification, report-derived supervision carries clinical signal.
Outputs: tcga_grade_prognosis.json + km_grade_comparison.png.
"""
import json, os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test

HERE = os.path.dirname(os.path.abspath(__file__))
per_case = pd.read_csv(os.path.join(HERE, "tcga_validation_per_case.csv"))
cdr = pd.read_excel(os.path.join(HERE, "..", "data", "TCGA-CDR.xlsx"), sheet_name="TCGA-CDR")
cdr = cdr[cdr["type"].isin(["ESCA", "STAD"])][["bcr_patient_barcode", "OS", "OS.time"]]
cdr.columns = ["barcode", "os_event", "os_days"]
df = per_case.merge(cdr, on="barcode", how="inner").dropna(subset=["os_event", "os_days"])
df = df[df["os_days"] > 0]
print(f"cases with survival: {len(df)}")

results = {"n": int(len(df))}
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)

for ax, (col, label) in zip(axes, [("grade", "pathladder (report-derived)"),
                                   ("grade_truth", "clinical record")]):
    sub = df.dropna(subset=[col]).copy()
    sub["g3"] = sub[col] == "G3"
    hi, lo = sub[sub["g3"]], sub[~sub["g3"]]
    lr = logrank_test(hi["os_days"], lo["os_days"], hi["os_event"], lo["os_event"])
    cox = CoxPHFitter().fit(sub[["os_days", "os_event", "g3"]].astype(float),
                            "os_days", "os_event")
    hr = float(cox.hazard_ratios_["g3"])
    ci = cox.confidence_intervals_.loc["g3"].values
    results[label] = {
        "n": int(len(sub)), "n_G3": int(sub["g3"].sum()),
        "logrank_p": round(float(lr.p_value), 5),
        "cox_hr_G3_vs_G12": round(hr, 3),
        "hr_95ci": [round(float(pd.np.exp(ci[0])) if hasattr(pd, "np") else float(2.718281828**ci[0]), 3),
                    round(float(2.718281828**ci[1]), 3)],
    }
    for mask, name, color in [(~sub["g3"], "G1/G2", "#2b6cb0"), (sub["g3"], "G3", "#c53030")]:
        km = KaplanMeierFitter().fit(sub.loc[mask, "os_days"] / 365.25,
                                     sub.loc[mask, "os_event"], label=f"{name} (n={mask.sum()})")
        km.plot_survival_function(ax=ax, color=color, ci_show=True, ci_alpha=0.15)
    ax.set_title(f"{label}\nlog-rank p={results[label]['logrank_p']:.2g}, "
                 f"HR={results[label]['cox_hr_G3_vs_G12']:.2f}")
    ax.set_xlabel("years"); ax.set_xlim(0, 5)
axes[0].set_ylabel("overall survival")
fig.suptitle("G3 vs G1/G2 overall survival, TCGA ESCA+STAD", y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "km_grade_comparison.png"), dpi=160, bbox_inches="tight")

print(json.dumps(results, indent=2))
json.dump(results, open(os.path.join(HERE, "tcga_grade_prognosis.json"), "w"), indent=2)
print("wrote tcga_grade_prognosis.json + km_grade_comparison.png")

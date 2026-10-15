"""Figure 15_F_scanner_robustness (docs/paper_scanner_robustness.md @ abe96e9) from results/paper_final/scanner_robustness/index.json:
A, robustness index R_k against k per extractor (pool of dual-scanner patients; 95% patient-bootstrap bands; chance level dashed);
B, primary R_10 with 95% CIs, with the decision threshold 1.25 x R_10(UNI2-h)."""
import json, os, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
J = json.load(open("results/paper_final/scanner_robustness/index.json")); OUT = os.path.expanduser("~/Downloads/be_paper_figs/v3"); NAME = "15_F_scanner_robustness"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
COL = {"UNI2-h": "#222222", "Virchow2": "#d62728", "H-optimus-0": "#1f77b4", "Prov-GigaPath": "#999999"}; EX = J["extractors"]; P = J["primary"]; pool = J["pool"]
fig, (a, b) = plt.subplots(1, 2, figsize=(9.6, 4.2), gridspec_kw={"width_ratios": [1.6, 1]})
for e in EX:
    cv = J["secondary"]["S1_curve"][e]; k = [c["k"] for c in cv]
    a.fill_between(k, [c["ci95"][0] for c in cv], [c["ci95"][1] for c in cv], color=COL[e], alpha=0.12, lw=0)
    a.plot(k, [c["R"] for c in cv], color=COL[e], lw=1.8 if e != "Prov-GigaPath" else 1.2, ls="-" if e != "Prov-GigaPath" else "--", label=e + (" (reference)" if e == "Prov-GigaPath" else ""))
a.axhline(P["UNI2-h"]["chance"], color="#555555", ls=":", lw=1, label=f"chance ({P['UNI2-h']['chance']:.3f})"); a.axvline(10, color="#bbbbbb", lw=0.8)
a.set_xlabel("k nearest neighbours"); a.set_ylabel("robustness index R_k = SO / (SO + OS)"); a.set_ylim(0, None); a.set_xlim(1, 50)
a.set_title(f"A  Same patient on the other scanner vs other patients on the same scanner\n{pool['samples']} samples, {pool['patients']} patients with samples on both scanners", fontsize=9, loc="left")
a.legend(fontsize=7.5, frameon=False, loc="upper right")
ys = list(range(len(EX)))[::-1]
for y, e in zip(ys, EX):
    b.errorbar(P[e]["R"], y, xerr=[[P[e]["R"] - P[e]["ci95"][0]], [P[e]["ci95"][1] - P[e]["R"]]], fmt="o", color=COL[e], ms=5, capsize=3)
b.axvline(1.25 * P["UNI2-h"]["R"], color="#d62728", ls="--", lw=0.9); b.axvline(P["UNI2-h"]["R"], color="#222222", ls=":", lw=0.9); b.axvline(P["UNI2-h"]["chance"], color="#999999", ls=":", lw=0.9)
b.text(P["UNI2-h"]["chance"], -0.55, " chance", fontsize=7, color="#777777", va="bottom")
b.text(1.25 * P["UNI2-h"]["R"], len(EX) - 0.45, " 1.25 x UNI2-h", fontsize=7, color="#d62728", va="bottom")
b.set_yticks(ys); b.set_yticklabels(EX); b.set_ylim(-0.6, len(EX) - 0.2); b.set_xlabel("R_10 (95% patient-bootstrap CI)")
DS = "results/paper_final/scanner_robustness/downstream.json"; sub = ""
if os.path.exists(DS):
    D = json.load(open(DS)); d = D["D1"]["all pre-event"]["delta"]["L-IMG"]["delta"]; c = D["D2"]["cross_b_s39_to_s10"]["auroc"]; sub = f"\nthen one run: L-IMG Δ {d:+.3f}; cross-scanner\n{c:.3f} (UNI2-h 0.726)"
b.set_title(f"B  Primary index, k = 10\nrule: {J['decision']['decision']}{sub}", fontsize=8.5, loc="left")
b.set_xlim(0, 0.2); b.set_xticks([0, 0.05, 0.10, 0.15, 0.20])
fig.tight_layout()
for ext in ("pdf", "png"): fig.savefig(f"{OUT}/{NAME}.{ext}", dpi=200)
print("SR FIG DONE")

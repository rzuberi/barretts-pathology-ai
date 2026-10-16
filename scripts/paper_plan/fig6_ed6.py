"""Extended Data Figure 6, foundation models (docs/paper_figures_v6.md @ d46b9a5; scanner_robustness, F5), from results/paper_final/scanner_robustness/
index.json and downstream.json, review_round2/f5_r28.json and triage/their_pre.json. (a) scanner-robustness index R_10 with 95% patient-bootstrap CIs for
UNI2-h, Virchow2, H-optimus-0 and Prov-GigaPath, with the chance level and the switching threshold; (b) L-IMG AUROC on all and NDBE pre-event biopsies
for the two extractors run downstream; (c) cross-scanner and within-C13239-01 AUROC; (d) triage share sequenced and sensitivity (single model) for the
UNI2-h and H-optimus-0 gates. Runs locally."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fig6_style import *
ix = J("scanner_robustness/index.json"); ds = J("scanner_robustness/downstream.json"); f5 = J("review_round2/f5_r28.json")["F5_R28"]["pre"]; pr0 = J("review_round4/r33_base_primary.json")["R33_base"]["claim_T"]; tr = J("triage_robustness/scanner.json")
EX = ix["extractors"]; EC = {"UNI2-h": MODEL["L-IMG"], "Virchow2": "#8E6C8A", "H-optimus-0": "#AA4499", "Prov-GigaPath": GREY}; P = ix["primary"]
fig = plt.figure(figsize=(183 * MM, 65 * MM)); ax = fig.add_axes([0.14, 0.22, 0.20, 0.66])
for i, e in enumerate(EX): ci_point(ax, P[e]["R"], len(EX) - 1 - i, P[e]["ci95"], EC[e], horizontal=True, ms=4)
ax.axvline(P["UNI2-h"]["chance"], color=LIGHT, lw=0.7, ls=":"); ax.axvline(1.25 * P["UNI2-h"]["R"], color="black", lw=0.6, ls="--"); ax.text(1.25 * P["UNI2-h"]["R"] + 0.003, len(EX) - 0.55, "1.25 × UNI2-h", fontsize=5.4, va="top"); ax.text(P["UNI2-h"]["chance"] + 0.003, -0.45, "chance", fontsize=5.4, color=GREY)
ax.set_yticks(range(len(EX))); ax.set_yticklabels(EX[::-1], fontsize=6); ax.set_xlabel("robustness index R$_{10}$ [95% CI]"); ax.set_xlim(0, 0.2); ax.set_ylim(-0.6, len(EX) - 0.4); letter(fig, ax, "a", dx=-0.13)
D1 = ds["D1"]; bx = fig.add_axes([0.42, 0.22, 0.15, 0.66]); items = [("all pre-event", "UNI2-h", D1["all pre-event"]["arms"]["L-IMG UNI2-h"]), ("all pre-event", "H-optimus-0", D1["all pre-event"]["arms"]["L-IMG H-optimus-0"]), ("NDBE pre-event", "UNI2-h", D1["NDBE pre-event"]["arms"]["L-IMG UNI2-h"]), ("NDBE pre-event", "H-optimus-0", D1["NDBE pre-event"]["arms"]["L-IMG H-optimus-0"])]
for i, (pop, e, d) in enumerate(items): ci_point(bx, (i // 2) + (-0.17 if e == "UNI2-h" else 0.17), d["auroc"], d["ci95"], EC[e], ms=3.8, label=e if i < 2 else None)
bx.set_xticks([0, 1]); bx.set_xticklabels(["all\npre-event\n(571)", "NDBE\npre-event\n(438)"], fontsize=5.6); bx.set_ylim(0.6, 1.0); bx.set_ylabel("L-IMG AUROC, fold-stratified [95% CI]"); bx.legend(loc="lower left", fontsize=5.4); bx.set_xlim(-0.6, 1.6); letter(fig, bx, "b", dx=-0.08)
cx = fig.add_axes([0.64, 0.22, 0.15, 0.66]); D2 = ds["D2"]; items2 = [("within C13239-01", "UNI2-h", {"value": D2["within_s39"]["reference_uni2"], "ci95": tr["within"]["s39"]["ci95"]}), ("within C13239-01", "H-optimus-0", {"value": D2["within_s39"]["auroc_fold_stratified"], "ci95": D2["within_s39"]["ci95"]}),
                                                               ("C13239-01 → C13210", "UNI2-h", {"value": tr["cross"]["cross_b_s39_none"]["auroc"], "ci95": tr["cross"]["cross_b_s39_none"]["ci95"]}), ("C13239-01 → C13210", "H-optimus-0", {"value": D2["cross_b_s39_to_s10"]["auroc"], "ci95": D2["cross_b_s39_to_s10"]["ci95"]})]
assert abs(tr["within"]["s39"]["auroc_fold_stratified"] - D2["within_s39"]["reference_uni2"]) < 1e-9
for i, (pop, e, d) in enumerate(items2): ci_point(cx, (i // 2) + (-0.17 if e == "UNI2-h" else 0.17), d["value"], d["ci95"], EC[e], ms=3.8)
cx.set_xticks([0, 1]); cx.set_xticklabels(["within\nC13239-01", "C13239-01\n→ C13210"], fontsize=5.8); cx.set_ylim(0.4, 1.0); cx.set_ylabel("L-IMG AUROC [95% CI]"); cx.set_xlim(-0.6, 1.6); letter(fig, cx, "c", dx=-0.08)
dx_ = fig.add_axes([0.86, 0.22, 0.13, 0.66]); g = {"UNI2-h": (pr0["share_sequenced"]["per_repeat_mean"], pr0["share_sequenced"]["ci95"], pr0["triage_no_vote"]["sensitivity"], pr0["triage_no_vote"]["sens_ci95"]), "H-optimus-0": (f5["share_sequenced"]["per_repeat_mean"], f5["share_sequenced"]["ci95"], f5["strategies"]["no_vote"]["triage"]["sensitivity"], f5["strategies"]["no_vote"]["triage"]["sens_ci95"])}
for i, (e, (s, sc, se, sec)) in enumerate(g.items()): ci_point(dx_, 0 + (i - 0.5) * 0.34, s, sc, EC[e], ms=3.8); ci_point(dx_, 1 + (i - 0.5) * 0.34, se, sec, EC[e], ms=3.8)
dx_.set_xticks([0, 1]); dx_.set_xticklabels(["share\nsequenced", "sensitivity"], fontsize=5.8); dx_.set_ylim(0.3, 1.0); dx_.set_xlim(-0.6, 1.6); dx_.set_ylabel("triage, single model [95% CI]"); letter(fig, dx_, "d", dx=-0.08)
save(fig, "ED6"); dump("ED6", {"robustness_index_R10": P, "decision": ix["decision"], "L_IMG_auroc": {f"{p} {e}": d for p, e, d in items}, "scanner_transfer": {f"{p} {e}": d for p, e, d in items2}, "triage_gate": {e: {"share": v[0], "share_ci95": v[1], "sensitivity": v[2], "sens_ci95": v[3]} for e, v in g.items()}})

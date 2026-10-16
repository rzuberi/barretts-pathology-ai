"""Extended Data Figure 5 (manuscript v7), p53 immunohistochemistry as a comparator, from results/paper_final/review_round6/r37.json. (a) availability of
a p53 result among pre-event biopsies by class and grade, with aberrant counts; (b-d) sensitivity and specificity of p53 alone, current care (grade ID or
LGD), current care + p53, the single-model triage, triage-or-p53 and B* at biopsy, endoscopy and patient level on the 450 biopsies with a p53 result
(exact CIs for deterministic rules, patient-bootstrap CIs for model rules). Runs locally."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fig7_style import *
R = J("review_round6/r37.json")["R37"]; A = R["a_availability"]; B = R["b_c_d_samples_with_p53"]; rules = B["rules"]
fig = plt.figure(figsize=(183 * MM, 75 * MM)); ax = fig.add_axes([0.08, 0.22, 0.20, 0.62]); G = A["grade_distribution"]
for i, (cls, key) in enumerate((("P", "progressor"), ("N", "non_progressor"))):
    w = sum(G["with_p53"][key].values()); wo = sum(G["without_p53"][key].values()); ax.bar(i - 0.18, w, width=0.34, color=CLASS[cls]); ax.bar(i + 0.18, wo, width=0.34, color=CLASS[cls], alpha=0.35)
    ax.text(i - 0.18, w + 4, f"{w}\n({A['aberrant_among_available'][key]} aberrant)", ha="center", fontsize=5.4); ax.text(i + 0.18, wo + 4, str(wo), ha="center", fontsize=5.4)
ax.bar([-5], [0], color="#777777", label="p53 result available"); ax.bar([-5], [0], color="#777777", alpha=0.35, label="no p53 result"); ax.set_xticks([0, 1]); ax.set_xticklabels(["progressor", "non-progressor"]); ax.set_xlim(-0.6, 1.6); ax.set_ylim(0, 420)
ax.set_ylabel("pre-event biopsies"); ax.legend(loc="upper left", fontsize=5.4); ax.text(0.03, 0.72, f"p53 available for {100 * A['pre_event_samples']['share_available']:.0f}%\nFisher p {A['pre_event_samples']['fisher_exact_p']}", transform=ax.transAxes, fontsize=5.4, va="top"); letter(fig, ax, "a", dx=-0.07, dy=0.03)
RN = [("p53", "p53 alone", MODEL["p53"]), ("D", "current care (D)", MODEL["grade"]), ("D+p53", "(D) + p53", MODEL["p53"]), ("T_single", "triage", "black"), ("T|p53_single", "triage or p53", MODEL["L-IMG"]), ("Bstar_single", "B*", MODEL["L-LATE"])]
src = {"availability": A, "n": {k: B[k] for k in ("n_samples", "n_progressor_samples", "n_patients", "n_progressor_patients", "n_endoscopies", "n_progressor_endoscopies")}, "rules": {}}
for q, (lv, ttl) in enumerate((("sample", f"per biopsy ({B['n_samples']}; {B['n_progressor_samples']} P)"), ("endoscopy", f"per endoscopy ({B['n_endoscopies']}; {B['n_progressor_endoscopies']} P)"), ("patient", f"per patient ({B['n_patients']}; {B['n_progressor_patients']} P)"))):
    a_ = fig.add_axes([0.37 + q * 0.215, 0.22, 0.17, 0.62]); src["rules"][lv] = {}
    for i, (k, lab, c_) in enumerate(RN):
        d = rules[k][lv]; sc = d.get("sens_ci95_exact", d["sens_ci95_bootstrap"]); pc = d.get("spec_ci95_exact", d["spec_ci95_bootstrap"]); yy = len(RN) - 1 - i
        ci_point(a_, d["sensitivity"], yy + 0.17, sc, CLASS["P"], horizontal=True, ms=3.2); ci_point(a_, d["specificity"], yy - 0.17, pc, CLASS["N"], horizontal=True, ms=3.2)
        src["rules"][lv][lab] = {"sensitivity": d["sensitivity"], "sens_ci95": sc, "specificity": d["specificity"], "spec_ci95": pc, "ci": "exact" if "sens_ci95_exact" in d else "patient bootstrap"}
    a_.set_yticks(range(len(RN))); a_.set_yticklabels([r[1] for r in RN][::-1] if q == 0 else [""] * len(RN), fontsize=5.8); a_.set_xlim(0, 1.02); a_.set_ylim(-0.6, len(RN) - 0.4); a_.set_xlabel("value [95% CI]", fontsize=5.8); a_.set_title(ttl, fontsize=6)
    if q == 0: a_.plot([], [], marker="o", color=CLASS["P"], ms=3.2, lw=0, label="sensitivity"); a_.plot([], [], marker="o", color=CLASS["N"], ms=3.2, lw=0, label="specificity"); a_.legend(loc="lower left", fontsize=5.2)
    letter(fig, a_, "bcd"[q], dx=-0.03 if q else -0.11, dy=0.03)
save(fig, "ED5"); dump("ED5", src)

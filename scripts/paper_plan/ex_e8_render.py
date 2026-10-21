"""Exploration E8: print the ranked results table (markdown) from results/paper_final/explore_E8/*.json (aggregates only). Local, no patient data."""
import json, glob, os, sys
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../results/paper_final/explore_E8")
def J(n):
    p = f"{D}/{n}.json"; return json.load(open(p)) if os.path.exists(p) else None
def ci(d, f=1):
    if not d or d.get("delta") is None: return "–"
    lo, hi = d["ci95"]; return f"{d['delta']:+.{f}f} [{lo:+.{f}f}, {hi:+.{f}f}]"
rows = []
for t in ("base", "qc", "noise", "arms", "conc", "flat", "seq", "soft"):
    r = J(t)
    if not r: continue
    cells = r.get("cells") or (r.get(t) or {}).get("cells") or []
    if t == "base": cells = [r["A12_full"], r["final_full"]]
    if t == "seq": cells = [r["seq"]["sequential"]]
    for c in cells: rows.append((t, c))
def key(x):
    c = x[1]; dp = c["dP_vs_final"]["delta"] or 0; return (0 if dp >= -0.5 else 1, c["dNP_vs_final"]["delta"], -dp)
print("| # | Test | Rule | Caught | NP flagged | Samples/pt | ΔNP vs final | Δcaught vs final | ΔNP vs A12 | Δcaught vs A12 | ΔNP vs R49 | Δcaught vs R49 | ΔNP vs P1 | Δspec vs P1 (matched sens) |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for i, (t, c) in enumerate(sorted(rows, key=key), 1):
    lab = c["label"].replace("A12 + ", "")
    print(f"| {i} | {t} | {lab} | {c['progressor_patients_caught']} | {c['non_progressor_patients_ever_flagged']} | {c['samples_sequenced_per_patient']} | {ci(c['dNP_vs_final'])} | {ci(c['dP_vs_final'])} | "
          f"{ci(c['dNP_vs_A12'])} | {ci(c['dP_vs_A12'])} | {ci(c['dNP_vs_R49'])} | {ci(c['dP_vs_R49'])} | {ci(c['dNP_vs_P1'])} | {ci(c.get('dspec_matched_vs_P1'), 3)} |")

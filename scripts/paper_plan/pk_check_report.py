import re, sys, os
t = open("docs/paper_final_inputs.md").read(); ok = True; body = [l for l in t.splitlines() if not l.startswith(">")]
rows = [l for l in body if re.match(r"^\| (K\d|F-)", l)]
for l in rows:
    if not re.search(r"\| (DONE|PARTIAL|NOT AVAILABLE) \|", l): print("status missing:", l[:80]); ok = False
if len(rows) != 13: print("status rows:", len(rows)); ok = False
for i in range(1, 6):
    if f"### K{i}." not in t: print("missing K", i); ok = False
for n in ["F_intro_forest", "F_table_forest", "F_D1_risk_groups", "F_D2_latent", "F_D3_attention", "F_D4_cnv_change", "F_D5_false_positives", "F_D6_false_negatives"]:
    for ext in ["png", "pdf", "json"]:
        if not os.path.exists(f"results/paper_final/figs/{n}.{ext}"): print("missing figure file", n, ext); ok = False
if any(re.search(r"\bpending\b", l, re.I) for l in body): print("pending remains:", [l[:80] for l in body if re.search(r"\bpending\b", l, re.I)][:5]); ok = False
print("OK" if ok else "FAILED"); sys.exit(0 if ok else 1)

"""Turn results/results.json (+ fig3_slide.json, gpu_hours.json) into (a) results/tables.md (pasted into RESULTS.md) and
(b) results/numbers.tex (LaTeX macros used by onepager.tex). Every number in the write-up comes from these two files.
usage: make_report.py RESULTS_DIR"""
import json, os, sys
D = sys.argv[1]; R = json.load(open(f"{D}/results.json"))
F3 = json.load(open(f"{D}/fig3_slide.json")) if os.path.exists(f"{D}/fig3_slide.json") else {}
GH = json.load(open(f"{D}/gpu_hours.json")) if os.path.exists(f"{D}/gpu_hours.json") else {}
MS = [1, 4, 16, 64, 256]; DSN = {"tcga": "TCGA LUAD vs LUSC", "c16": "CAMELYON16"}; MON = {"uni2h": "UNI2-h", "virchow2": "Virchow2"}
def f(r): return f"{r['auroc']:.3f} [{r['ci'][0]:.3f}, {r['ci'][1]:.3f}]" if r else "–"
def fd(r): return f"{r['delta_vs_all']['point']:+.3f} [{r['delta_vs_all']['ci'][0]:+.3f}, {r['delta_vs_all']['ci'][1]:+.3f}]" if r and "delta_vs_all" in r else "–"
L = ["## Tables (generated from results/results.json by code/make_report.py)", ""]
L += ["### Primary endpoint: smallest test budget whose paired ΔAUROC vs all tiles has a 95% CI containing 0 (ABMIL trained on all tiles)", "",
      "| task | encoder | random | attention-top |", "|---|---|---|---|"]
for ds in DSN:
    for mo in MON: L.append(f"| {DSN[ds]} | {MON[mo]} | {R['primary_endpoint'].get(f'{ds}|{mo}|rand', '–')} | {R['primary_endpoint'].get(f'{ds}|{mo}|attn_top', '–')} |")
for ds in DSN:
    n = R["n"].get(ds, {}); L += ["", f"### {DSN[ds]} — AUROC [95% CI] by test budget (n = {n.get('slides')} slides, {n.get('pos')} positive)", ""]
    for arm in ("all", "r256"):
        for agg in ("abmil", "meanlr"):
            L += [f"**trained on {'all tiles' if arm == 'all' else '256 random tiles'}, {'ABMIL' if agg == 'abmil' else 'mean-pool + LR'}**", "",
                  "| encoder | rule | " + " | ".join(f"m={m}" for m in MS) + " | all |", "|---|---|" + "---|" * (len(MS) + 1)]
            for mo in MON:
                ra = R["curves"].get(f"{ds}|{mo}|{arm}|{agg}|all|all")
                for rule in ("rand", "attn_top", "attn_bot"):
                    L.append(f"| {MON[mo]} | {rule} | " + " | ".join(f(R['curves'].get(f'{ds}|{mo}|{arm}|{agg}|{rule}|{m}')) for m in MS) + f" | {f(ra)} |")
            L.append("")
    L += [f"ΔAUROC vs all tiles (ABMIL, trained on all tiles):", "", "| encoder | rule | " + " | ".join(f"m={m}" for m in MS) + " |", "|---|---|" + "---|" * len(MS)]
    for mo in MON:
        for rule in ("rand", "attn_top"): L.append(f"| {MON[mo]} | {rule} | " + " | ".join(fd(R['curves'].get(f'{ds}|{mo}|all|abmil|{rule}|{m}')) for m in MS) + " |")
    L += ["", "Training on 256 random tiles minus training on all tiles (ABMIL), same test setting:", "", "| encoder | test setting | r256 | all | Δ [95% CI] |", "|---|---|---|---|---|"]
    for mo in MON:
        for k in (("all", 0), ("rand", 16), ("rand", 256), ("attn_top", 16)):
            r = R["train_arm"].get(f"{ds}|{mo}|{k[0]}|{k[1]}")
            if r: L.append(f"| {MON[mo]} | {k[0]} {k[1] or ''} | {r['r256']:.3f} | {r['all']:.3f} | {r['delta']:+.3f} [{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}] |")
    L += ["", "Encoder (Virchow2 minus UNI2-h, ABMIL trained on all tiles):", "", "| test setting | Virchow2 | UNI2-h | Δ [95% CI] |", "|---|---|---|---|"]
    for k in (("all", 0), ("rand", 16), ("rand", 256), ("attn_top", 16)):
        r = R["encoder"].get(f"{ds}|{k[0]}|{k[1]}")
        if r: L.append(f"| {k[0]} {k[1] or ''} | {r['virchow2']:.3f} | {r['uni2h']:.3f} | {r['delta']:+.3f} [{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}] |")
L += ["", "### CAMELYON16 by lesion size (each class vs all 80 normal test slides; ABMIL trained on all tiles)", ""]
for les in ("macro", "micro"):
    n = R["n"].get(f"c16_{les}", {}); L += [f"**{les}-metastases** (n = {n.get('slides')}, {n.get('pos')} positive)", "", "| encoder | rule | " + " | ".join(f"m={m}" for m in MS) + " | all | random endpoint |", "|---|---|" + "---|" * (len(MS) + 2)]
    for mo in MON:
        for rule in ("rand", "attn_top"):
            L.append(f"| {MON[mo]} | {rule} | " + " | ".join(f(R['lesion'].get(f'{les}|{mo}|{rule}|{m}')) for m in MS) + f" | {f(R['lesion'].get(f'{les}|{mo}|all|all'))} | {R['lesion'].get(f'{les}|{mo}|endpoint_rand', '–') if rule == 'rand' else ''} |")
    L.append("")
L += ["### Attention-top-16 tiles inside the tumour annotation (CAMELYON16 test tumour slides)", "", "| encoder | lesion | n | mean fraction of top-16 in tumour | slides with ≥1 top-16 tile in tumour | median tumour fraction of tissue | P(≥1 of 16 random tiles in tumour), median |", "|---|---|---|---|---|---|---|"]
for k, r in R["hit_rate"].items():
    mo, les = k.split("|"); L.append(f"| {MON[mo]} | {les} | {r['n']} | {r['mean_hit16']:.2f} | {r['slides_with_any_tumour_tile_in_top16']}/{r['n']} | {r['median_tumour_frac']:.4f} | {r['p_random16_hits_any_median']:.2f} |")
L += ["", "### Embedding cost (CUDA-synchronised forward time incl. host→device copy; excl. disk read, model load)", "", "| encoder | GPU | slides | median tiles | median GPU-s: 16 / 256 / all | tiles/s (all) | median saving ×: 16 / 256 |", "|---|---|---|---|---|---|---|"]
for k, r in R["cost"].items():
    if k.startswith("cpu_"): continue
    mo, gpu, ds = k.split("|")
    if ds != "both": continue
    L.append(f"| {MON[mo]} | {gpu.replace('NVIDIA ', '')} | {r['n_slides']} | {r['median_tiles']:.0f} | {r['median_gpu_s_16']:.3f} / {r['median_gpu_s_256']:.2f} / {r['median_gpu_s_all']:.1f} | {r['tiles_per_s_all']:.0f} | {r['saving_x_16_median']:.0f} / {r['saving_x_256_median']:.0f} |")
for k, r in R["cost"].items():
    if k.startswith("cpu_"): L.append(f"\nTiling ({k.split('|')[1]}): {r['n_slides']} slides, median {r['median_tiles']:.0f} tissue tiles (max {r['max_tiles']}, total {r['total_tiles']:,}); median read {r['median_read_s']:.0f} s with {r['read_workers']} threads.")
if GH: L += ["", f"GPU allocation used (sacct, all GPU jobs of this study incl. smoke tests): {json.dumps(GH)}"]
open(f"{D}/tables.md", "w").write("\n".join(L) + "\n")
# ---- LaTeX macros ----
M = {}
def mac(name, v): M[name] = v
for ds, D_ in (("tcga", "T"), ("c16", "C")):
    for mo, M_ in (("uni2h", "U"), ("virchow2", "V")):
        for rule, R_ in (("rand", "R"), ("attn_top", "A")): mac(f"ep{D_}{M_}{R_}", str(R["primary_endpoint"].get(f"{ds}|{mo}|{rule}", "--")))
        c = lambda k: R["curves"].get(f"{ds}|{mo}|all|abmil|{k}")
        for tag, k in (("All", "all|all"), ("RandSixteen", "rand|16"), ("RandTwoFiftySix", "rand|256"), ("RandOne", "rand|1"), ("AttnSixteen", "attn_top|16"), ("AttnOne", "attn_top|1"), ("RandSixtyFour", "rand|64")):
            r = c(k)
            if r: mac(f"auc{D_}{M_}{tag}", f"{r['auroc']:.3f}"); mac(f"ci{D_}{M_}{tag}", f"{r['ci'][0]:.3f}--{r['ci'][1]:.3f}")
            if r and "delta_vs_all" in r: mac(f"d{D_}{M_}{tag}", f"{r['delta_vs_all']['point']:+.3f}"); mac(f"dci{D_}{M_}{tag}", f"{r['delta_vs_all']['ci'][0]:+.3f} to {r['delta_vs_all']['ci'][1]:+.3f}")
    mac(f"n{D_}", str(R["n"].get(ds, {}).get("slides", "--"))); mac(f"pos{D_}", str(R["n"].get(ds, {}).get("pos", "--")))
for les, L_ in (("macro", "Ma"), ("micro", "Mi")):
    for mo, M_ in (("uni2h", "U"), ("virchow2", "V")):
        mac(f"ep{L_}{M_}", str(R["lesion"].get(f"{les}|{mo}|endpoint_rand", "--")))
        for tag, k in (("All", "all|all"), ("RandSixteen", "rand|16"), ("AttnSixteen", "attn_top|16"), ("RandSixtyFour", "rand|64"), ("RandTwoFiftySix", "rand|256")):
            r = R["lesion"].get(f"{les}|{mo}|{k}")
            if r: mac(f"auc{L_}{M_}{tag}", f"{r['auroc']:.3f}")
for k, r in R["hit_rate"].items():
    mo, les = k.split("|"); t = ("U" if mo == "uni2h" else "V") + {"macro": "Ma", "micro": "Mi", "all": "All"}[les]
    mac(f"hit{t}", f"{r['mean_hit16']:.2f}"); mac(f"hitany{t}", f"{r['slides_with_any_tumour_tile_in_top16']}/{r['n']}"); mac(f"rnd{t}", f"{r['p_random16_hits_any_median']:.2f}")
for k, r in R["cost"].items():
    if k.startswith("cpu_") or not k.endswith("|both"): continue
    mo, gpu, _ = k.split("|"); t = ("U" if mo == "uni2h" else "V") + ("L" if "L40S" in gpu else "H")
    mac(f"cs{t}", f"{r['median_gpu_s_16']:.2f}"); mac(f"ct{t}", f"{r['median_gpu_s_256']:.2f}"); mac(f"ca{t}", f"{r['median_gpu_s_all']:.0f}")
    mac(f"sx{t}", f"{r['saving_x_16_median']:.0f}"); mac(f"tx{t}", f"{r['saving_x_256_median']:.0f}"); mac(f"tps{t}", f"{r['tiles_per_s_all']:.0f}")
for k, r in R["cost"].items():
    if k.startswith("cpu_"): t = "T" if "tcga" in k else "C"; mac(f"tiles{t}", f"{r['median_tiles']:,.0f}".replace(",", "{,}")); mac(f"totaltiles{t}", f"{r['total_tiles'] / 1e6:.1f}")
if F3: mac("figslide", F3["slide"].replace("_", "\\_")); mac("fighits", str(F3["hits_top16"])); mac("figlesion", F3["lesion"]); mac("figfrac", f"{100 * F3['tumour_frac']:.1f}")
if GH: mac("gpuh", f"{GH.get('total_gpu_hours', 0):.1f}")
import re
alpha = lambda s: re.sub(r"\d", lambda d: "ABCDEFGHIJ"[int(d.group())], s)
open(f"{D}/numbers.tex", "w").write("\n".join(f"\\newcommand{{\\{alpha(k)}}}{{{v}}}" for k, v in M.items()) + "\n"); print(len(M), "macros;", len(L), "table lines")

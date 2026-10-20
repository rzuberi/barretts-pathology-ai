"""Write campaign TSVs: op_prep (one CPU task per slide), op_embed (one GPU task per group of 8 slides: model load ~30 s
would otherwise dominate an ~8 s/slide embed), op_train (one GPU task per task x split x model x repeat)."""
import pandas as pd, os
A = "/mnt/scratche/slow/fmlab/zuberi01/analysis/outreach_panakeia"; C = A + "/code"; E = "virchow2"
c = pd.read_csv(A + "/cohort.csv"); pids = list(c.patient)
with open(C + "/op_prep.tsv", "w") as f:
    for p in pids: f.write(f"{p}\t{E}\tpython {C}/prep_slide.py {p}\n")
os.makedirs(C + "/groups", exist_ok=True)
with open(C + "/op_embed.tsv", "w") as f:
    for g in range(0, len(pids), 8):
        gf = f"{C}/groups/g{g // 8:03d}.txt"; open(gf, "w").write("\n".join(pids[g:g + 8]) + "\n")
        f.write(f"g{g // 8:03d}\t{E}\tpython {C}/embed_slides.py {gf}\n")
with open(C + "/op_train.tsv", "w") as f:
    for task in ("msi", "wgd", "cin", "aneu"):
        for split in (("strat", "site") if task in ("wgd", "msi") else ("strat",)):
            for model in ("abmil", "meanlr", "shuf"):
                for r in range(3): f.write(f"{task}_{split}_{model}_r{r}\t{E}\tpython {C}/train_unit.py {task} {split} {model} {r}\n")
    for model in ("abmil", "meanlr", "shuf"):
        for r in range(3): f.write(f"wgdx_x_{model}_r{r}\t{E}\tpython {C}/train_unit.py wgdx x {model} {r}\n")
print(len(pids), "prep;", -(-len(pids) // 8), "embed groups;", sum(1 for _ in open(C + "/op_train.tsv")), "train units")

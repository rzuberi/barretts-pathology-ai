"""Task F data: three-class items per (cell, pert, gene); validation = 10% of each cell's TRAIN perturbations (perturbation-disjoint).
Out: F/data/{train_all,train_nor,val,val_nor,test}.jsonl + summary.json. Seed 20261016."""
import json, os, random
import pandas as pd
from pert_common import *
os.makedirs(O + "/data", exist_ok=True); rnd = random.Random(20261016); rows = []
for c in CELLS:
    de = pd.read_csv(f"{Q}/{c}-de.csv"); dr = pd.read_csv(f"{Q}/{c}-dir.csv"); dmap = {(a, b): l for a, b, l in zip(dr.pert, dr.gene, dr.label)}
    trp = sorted(de[de.split == "train"].pert.unique()); rnd.shuffle(trp); valp = set(trp[: max(1, len(trp) // 10)])
    for a, b, l, s in zip(de.pert, de.gene, de.label, de.split):
        if l == 1 and (a, b) not in dmap: continue
        y = "none" if l == 0 else ("up" if dmap[(a, b)] == 1 else "down")
        part = "test" if s == "test" else ("val" if a in valp else "train")
        rows.append({"cell": c, "pert": a, "gene": b, "label": y, "de": int(l), "dir": (None if l == 0 else int(dmap[(a, b)])), "part": part})
df = pd.DataFrame(rows)
def balanced(d, n_per, seed):
    return pd.concat([g.sample(min(n_per, len(g)), random_state=seed) for _, g in d.groupby("label")]).sample(frac=1, random_state=seed)
out = {"train_all": balanced(df[df.part == "train"], 16000, 1), "train_nor": balanced(df[(df.part == "train") & (df.cell != "rpe1")], 16000, 2),
       "grpo_all": balanced(df[df.part == "train"], 1600, 3), "grpo_nor": balanced(df[(df.part == "train") & (df.cell != "rpe1")], 1600, 4),
       "val": balanced(df[df.part == "val"], 1000, 5), "val_nor": balanced(df[(df.part == "val") & (df.cell != "rpe1")], 1000, 6),
       "test": df[df.part == "test"]}
for k, d in out.items(): d.to_json(f"{O}/data/{k}.jsonl", orient="records", lines=True)
s = {k: {"n": int(len(d)), "labels": d.label.value_counts().to_dict(), "cells": d.cell.value_counts().to_dict()} for k, d in out.items()}
s["total_pairs"] = int(len(df)); s["dir_pairs"] = int(df.de.sum()); json.dump(s, open(O + "/data/summary.json", "w"), indent=1); print(json.dumps(s, indent=1))

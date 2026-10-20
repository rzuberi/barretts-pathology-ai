#!/bin/bash
# pipeline smoke (one GPU job): pack the 10 smoke slides -> train/eval 2 epochs every ds x model x arm -> aggregate. Own *_smoke folders.
A=/mnt/scratche/slow/fmlab/zuberi01/analysis/outreach_tilebio; export SMOKE=1; set -e
source ~/miniforge3/etc/profile.d/conda.sh; conda activate virchow2
L=$(cat $A/code/tasks_tile_smoke.tsv | awk '{print $(NF-1), $NF}')
for ds in tcga c16; do for mo in uni2h virchow2; do
  mkdir -p $A/data/packs_smoke
  # restrict the smoke pack to the 10 smoke slides
  python - <<PY
import pandas as pd; s = pd.read_csv("$A/manifests/slides.csv"); sm = set("""$L""".split()[1::2])
print("$ds $mo", sum(s.slide.isin(sm) & (s.ds == "$ds")))
PY
  SMOKE_SLIDES="$(echo "$L" | awk -v d=$ds '$1==d {print $2}' | tr '\n' ' ')" python $A/code/pack_feats.py $ds $mo
  for arm in all r256; do DS=$ds MODEL=$mo FOLD=0 ARM=$arm python $A/code/train_eval_pub.py; done
done; done
python $A/code/aggregate_pub.py
ls -la $A/preds_smoke $A/results_smoke $A/figures_smoke

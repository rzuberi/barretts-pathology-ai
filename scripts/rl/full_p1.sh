#!/bin/bash
# Project 1 full pipeline as race-to-run tasks. Stage 1 (parallel): zero-shot eval, SFT, GRPO-from-base.
# Stage 2 (after SFT): GRPO-from-SFT (base = SFT merged). Stage 3: eval every checkpoint on val, best on test/test485. Resume-safe.
set -eo pipefail
STAGE=$1; export HF_HOME=/mnt/scratche/slow/fmlab/zuberi01/hf_cache HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
C=/mnt/scratche/slow/fmlab/zuberi01/phd/thesis/scripts/rl; R=/mnt/scratche/slow/fmlab/zuberi01/analysis/rl_grader; RUNS=$R/runs
BASE=$(ls -d $HF_HOME/hub/models--google--medgemma-4b-it/snapshots/*/ | head -1)
case $STAGE in
  zs)        BASE=$BASE TAG=zs SETS=val,test,test485 python $C/evaluate.py ;;
  sft)       MODE=sft BASE=$BASE OUT=$RUNS/sft STEPS=${SFT_STEPS:-900} python $C/train.py ;;
  grpo_base) MODE=grpo BASE=$BASE OUT=$RUNS/grpo_base STEPS=300 python $C/train.py ;;
  grposmoke) MODE=grpo BASE=$BASE OUT=$R/smoke/grpo STEPS=10 N_TRAIN=200 python $C/train.py ;;
  merge_sft) python $C/merge.py $BASE $RUNS/sft/final $RUNS/sft_merged ;;
  grpo_sft)  MODE=grpo BASE=$RUNS/sft_merged OUT=$RUNS/grpo_sft STEPS=300 python $C/train.py ;;
  eval_ckpt) # $2 = run name, $3 = checkpoint dir name; base for adapters
             B=$BASE; [ "$2" = grpo_sft ] && B=$RUNS/sft_merged
             BASE=$B ADAPTER=$RUNS/$2/$3 TAG=$2__$3 SETS=val python $C/evaluate.py ;;
  eval_best) B=$BASE; [ "$2" = grpo_sft ] && B=$RUNS/sft_merged
             BASE=$B ADAPTER=$RUNS/$2/$3 TAG=$2__$3 SETS=val,test,test485 python $C/evaluate.py ;;
esac

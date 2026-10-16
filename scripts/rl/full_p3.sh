#!/bin/bash
# Project 3 stages (race-to-run tasks). zs/fs: baselines on val+test. sft: 3 epochs (225 steps). grpo: 300 steps from base.
# ckpt: eval one checkpoint on val. best: eval one checkpoint on val+test.
set -eo pipefail
export HF_HOME=/mnt/scratche/slow/fmlab/zuberi01/hf_cache HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
C=/mnt/scratche/slow/fmlab/zuberi01/phd/thesis/scripts/rl; O=/mnt/scratche/slow/fmlab/zuberi01/analysis/rl_sql; RUNS=$O/runs
BASE=$(ls -d $HF_HOME/hub/models--Qwen--Qwen2.5-Coder-7B-Instruct/snapshots/*/ | head -1)
case $1 in
  zs)   BASE=$BASE TAG=zs SHOTS=0 SETS=val,test python $C/sql_eval.py ;;
  fs)   BASE=$BASE TAG=fs5 SHOTS=5 SETS=val,test python $C/sql_eval.py ;;
  sft)  MODE=sft BASE=$BASE OUT=$RUNS/sft STEPS=225 python $C/sql_train.py ;;
  grpo) MODE=grpo BASE=$BASE OUT=$RUNS/grpo STEPS=300 python $C/sql_train.py ;;
  ckpt) BASE=$BASE ADAPTER=$RUNS/$2/$3 TAG=$2__$3 SETS=val python $C/sql_eval.py ;;
  best) BASE=$BASE ADAPTER=$RUNS/$2/$3 TAG=$2__$3 SETS=val,test python $C/sql_eval.py ;;
esac

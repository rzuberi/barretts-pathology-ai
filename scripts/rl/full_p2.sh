#!/bin/bash
# Project 2 stages. BASE = project-1 RL model merged (rl_grader/runs/p1_best_merged). alpha runs train on train + low-confidence sections.
set -eo pipefail
export HF_HOME=/mnt/scratche/slow/fmlab/zuberi01/hf_cache HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export MASTER_ADDR=127.0.0.1 MASTER_PORT=$((20000 + ${SLURM_JOB_ID:-$$} % 30000))   # unique per job: two GRPO runs on one node clashed on port 29500
C=/mnt/scratche/slow/fmlab/zuberi01/phd/thesis/scripts/rl; R=/mnt/scratche/slow/fmlab/zuberi01/analysis/rl_grader; A=/mnt/scratche/slow/fmlab/zuberi01/analysis/rl_abstain
M=$R/runs/p1_best_merged; D=$R/data
case $1 in
  conf)  BASE=$M TAG=p1best_conf SETS=val,pop,test485 LOGPROBS=1 SAMPLES=8 PREDS=$A/preds python $C/evaluate.py ;;   # baseline (b)
  train) MODE=grpo BASE=$M OUT=$A/runs/alpha$2 STEPS=300 ALPHA=$2 EXTRA=$D/all_lowconf.jsonl CAP_PER_CLASS=${CAP:-1200} python $C/train.py ;;
  ckpt)  BASE=$M ADAPTER=$A/runs/alpha$2/$3 TAG=alpha$2__$3 SETS=val UNSURE=1 LOGPROBS=1 PREDS=$A/preds python $C/evaluate.py ;;
  best)  BASE=$M ADAPTER=$A/runs/alpha$2/$3 TAG=alpha$2__$3 SETS=val,pop,test485 UNSURE=1 LOGPROBS=1 PREDS=$A/preds python $C/evaluate.py ;;
esac

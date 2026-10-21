#!/bin/bash
# calib strand stages: score6 <model> [N]. Models: p1 (p1_best_merged), d2 (LoRA checkpoint-150, confidence prompt).
set -eo pipefail
export HF_HOME=/mnt/scratche/slow/fmlab/zuberi01/hf_cache HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
B=/mnt/scratche/slow/fmlab/zuberi01; C=$B/phd/thesis/scripts/rl/explore/calib; M=$B/analysis/rl_grader/runs/p1_best_merged
O=$B/analysis/rl_explore/calib; mkdir -p $O/preds; chmod 700 $O
N=${3:-0}; SM=""; [ "$N" != "0" ] && SM="smoke_"
case $1 in
  score6) case $2 in
    p1) BASE=$M TAG=${SM}p1 N=$N OUT=$O/preds python $C/score6.py ;;
    d2) BASE=$M ADAPTER=$B/analysis/rl_followups/D/runs/d2/checkpoint-150 TAG=${SM}d2 CONF=1 N=$N OUT=$O/preds python $C/score6.py ;;
  esac ;;
esac

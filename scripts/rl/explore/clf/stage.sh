#!/bin/bash
# clf strand stage script for race3.sh: bash stage.sh STAGE ARGS. Stages: prep KEY | train KEY SEED [EPOCHS LR BS TAG] | smoke KEY
set -euo pipefail
B=/mnt/scratche/slow/fmlab/zuberi01; C=$B/phd/thesis/scripts/rl/explore/clf; cd $C
export HF_HOME=$B/hf_cache HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
ST=$1; shift
case $ST in
  prep)  python prep.py $1 ;;
  smoke) KEY=$1 SEED=0 N=96 EPOCHS=1 EVAL_EVERY=0.5 BS=8 TAG=smoke_$1 python train_clf.py ;;
  train) KEY=$1 SEED=$2 EPOCHS=${3:-3} LR=${4:-3e-5} BS=${5:-16} TAG=${6:-$1_s$2} python train_clf.py ;;
  smoketrain) python prep.py $1; KEY=$1 SEED=0 N=96 EPOCHS=1 EVAL_EVERY=0.5 BS=8 TAG=smoke_$1 python train_clf.py && KEY=$1 SEED=${2:-0} EPOCHS=${3:-3} LR=${4:-3e-5} BS=${5:-16} TAG=$1_s${2:-0} python train_clf.py ;;
  *) echo "unknown stage $ST"; exit 1 ;;
esac

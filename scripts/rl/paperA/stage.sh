#!/bin/bash
# Paper A stage script for race3.sh: bash stage.sh STAGE ARGS. Never edit while jobs run it.
set -eo pipefail
B=/mnt/scratche/slow/fmlab/zuberi01; C=$B/phd/thesis/scripts/rl/paperA; RL=$B/phd/thesis/scripts/rl; O=$B/analysis/paperA; cd $C
export HF_HOME=$B/hf_cache HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export MASTER_ADDR=127.0.0.1 MASTER_PORT=$((20000 + ${SLURM_JOB_ID:-$$} % 30000))
BASE=$(ls -d $HF_HOME/hub/models--google--medgemma-4b-it/snapshots/*/ | head -1)
ST=$1; shift
case $ST in
  enc)        TAG=$1 SEED=$2 FOLD=$3 TEACHER=$4 SAVE=${5:-0} python train_enc.py ;;
  chain_cf)   for k in 0 1 2 3 4; do TAG=cf_s$1_f$k SEED=$1 FOLD=$k TEACHER=jury python train_enc.py; done ;;
  chain_teach) for t in juror1 top3 soft; do TAG=${t}_s$1 SEED=$1 FOLD=-1 TEACHER=$t python train_enc.py; done
              [ "$1" = 0 ] && TAG=full_s0 SEED=0 FOLD=-1 TEACHER=jury SAVE=1 python train_enc.py; true ;;
  sft)        SEED=$1 BASE=$BASE OUT=$O/sft/runs/sftm_s$1 STEPS=900 python train_sft_masked.py
              BASE=$BASE ADAPTER=$O/sft/runs/sftm_s$1/final TAG=sftm_s$1 SETS=val,test,test485 PREDS=$O/sft/preds python $RL/evaluate.py
              BASE=$BASE ADAPTER=$O/sft/runs/sftm_s$1/final TAG=sftm_s$1_maj8 SETS=test485 SAMPLES=8 PREDS=$O/sft/preds python $RL/evaluate.py ;;
  swg)        python swg_encoder.py ;;
  smoke)      SM=$O/smoke; rm -rf $SM/runs; mkdir -p $SM
              SEED=0 OUT=$SM/runs/sftm STEPS=5 N_TRAIN=40 python train_sft_masked.py 2>&1 | grep -E "first example|TRAIN DONE|Error|error" | tail -3
              BASE=$BASE ADAPTER=$SM/runs/sftm/final TAG=smk_sftm SETS=test485 N=24 SAMPLES=8 PREDS=$SM/preds python $RL/evaluate.py 2>&1 | grep -E "parse rate|Error" | tail -2
              python -c "import json; r=[json.loads(l) for l in open('$SM/preds/smk_sftm__test485.jsonl')]; print('samples', sum(len(x.get('samples') or []) for x in r), 'preds', sum(x['pred'] is not None for x in r), '/', len(r))"
              TAG=smk_cf SEED=0 FOLD=0 TEACHER=jury N=96 EPOCHS=1 BS=8 python train_enc.py 2>&1 | grep -E "training rows|val_exact|test485|Error" | tail -4
              TAG=smk_soft SEED=0 FOLD=-1 TEACHER=soft SAVE=1 N=96 EPOCHS=1 BS=8 python train_enc.py 2>&1 | grep -E "training rows|val_exact|test485|Error" | tail -4
              echo PA_SMOKE_OK ;;
  *) echo "unknown stage $ST"; exit 1 ;;
esac

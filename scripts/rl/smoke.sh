#!/bin/bash
# Project 1 smoke: 20 SFT steps + 20 GRPO steps on 50 examples, eval (N=50 per set), metrics. Stops at the first failure.
set -eo pipefail
export HF_HOME=/mnt/scratche/slow/fmlab/zuberi01/hf_cache HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
C=/mnt/scratche/slow/fmlab/zuberi01/phd/thesis/scripts/rl; R=/mnt/scratche/slow/fmlab/zuberi01/analysis/rl_grader; SM=$R/smoke; mkdir -p $SM/preds
BASE=$(ls -d $HF_HOME/hub/models--google--medgemma-4b-it/snapshots/*/ | head -1)
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "== SFT smoke"; MODE=sft BASE=$BASE OUT=$SM/sft STEPS=20 N_TRAIN=50 python $C/train.py 2>&1 | tee $SM/sft.log | grep -E "TRAIN DONE|Error|loss" | tail -4
echo "== GRPO smoke"; MODE=grpo BASE=$BASE OUT=$SM/grpo STEPS=20 N_TRAIN=50 python $C/train.py 2>&1 | tee $SM/grpo.log | grep -E "TRAIN DONE|Error|reward" | tail -5
echo "== eval"; for t in zs sft grpo; do AD=""; [ $t = sft ] && AD=$SM/sft/final; [ $t = grpo ] && AD=$SM/grpo/final
  PREDS=$SM/preds BASE=$BASE ADAPTER=$AD TAG=smoke_$t N=50 SETS=val,test485 python $C/evaluate.py 2>&1 | tee -a $SM/eval.log | grep -E "sections|DONE|Error" | tail -4; done
echo "== metrics"; PREDS=$SM/preds TAGS=smoke_zs,smoke_sft,smoke_grpo OUT=$SM/metrics.json python $C/metrics.py 2>&1 | tail -8
echo SMOKE_OK

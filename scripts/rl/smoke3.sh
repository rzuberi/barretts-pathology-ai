#!/bin/bash
# Project 3: paraphrase (once), then smoke: SFT 20 steps + GRPO 20 steps on 50 questions, eval N=40 on val, zero-shot and 5-shot too.
set -eo pipefail
export HF_HOME=/mnt/scratche/slow/fmlab/zuberi01/hf_cache HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
C=/mnt/scratche/slow/fmlab/zuberi01/phd/thesis/scripts/rl; O=/mnt/scratche/slow/fmlab/zuberi01/analysis/rl_sql; SM=$O/smoke; mkdir -p $SM/preds
BASE=$(ls -d $HF_HOME/hub/models--Qwen--Qwen2.5-Coder-7B-Instruct/snapshots/*/ | head -1)
nvidia-smi --query-gpu=name --format=csv,noheader
[ -f $O/questions.jsonl ] || python $C/sql_paraphrase.py 2>&1 | grep -vE "Warning|warn|it/s" | tail -3
echo "== zs / 5-shot"; for sh in 0 5; do PREDS=$SM/preds BASE=$BASE TAG=smoke_shot$sh SHOTS=$sh SETS=val N=40 python $C/sql_eval.py 2>&1 | tee -a $SM/eval.log | grep -E "exec_acc|DONE|Error" | tail -2; done
echo "== SFT"; MODE=sft BASE=$BASE OUT=$SM/sft STEPS=20 N_TRAIN=50 python $C/sql_train.py 2>&1 | grep -E "TRAIN DONE|Error|loss" | tail -3
echo "== GRPO"; MODE=grpo BASE=$BASE OUT=$SM/grpo STEPS=20 N_TRAIN=50 python $C/sql_train.py 2>&1 | grep -E "TRAIN DONE|Error|reward" | tail -4
for t in sft grpo; do PREDS=$SM/preds BASE=$BASE ADAPTER=$SM/$t/final TAG=smoke_$t SETS=val N=40 python $C/sql_eval.py 2>&1 | tee -a $SM/eval.log | grep -E "exec_acc|DONE|Error" | tail -2; done
echo SMOKE3_OK

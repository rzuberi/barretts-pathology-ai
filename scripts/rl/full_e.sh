#!/bin/bash
# Task E stages (agent). Never edit while jobs run.
set -eo pipefail
export HF_HOME=/mnt/scratche/slow/fmlab/zuberi01/hf_cache HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export MASTER_ADDR=127.0.0.1 MASTER_PORT=$((20000 + ${SLURM_JOB_ID:-$$} % 30000))
C=/mnt/scratche/slow/fmlab/zuberi01/phd/thesis/scripts/rl; O=/mnt/scratche/slow/fmlab/zuberi01/analysis/rl_sql; QF=$O/questions_agent.jsonl; ER=$O/agent_runs; mkdir -p $ER
BASE=$(ls -d $HF_HOME/hub/models--Qwen--Qwen2.5-Coder-7B-Instruct/snapshots/*/ | head -1); M=$ER/sft_merged
case $1 in
  e_sft)    QFILE=$QF MODE=sft BASE=$BASE OUT=$ER/sft STEPS=75 python $C/sql_train.py ;;
  e_sftck)  QFILE=$QF BASE=$BASE ADAPTER=$ER/sft/$2 TAG=agsplit_sft__$2 SETS=val PREDS=$O/agsplit_preds python $C/sql_eval.py ;;
  e_merge)  python $C/merge.py $BASE $ER/sft/$2 $M ;;
  e_a)      QFILE=$QF BASE=$BASE ADAPTER=$ER/sft/$2 TAG=agsplit_sft__$2 SETS=val,test PREDS=$O/agsplit_preds python $C/sql_eval.py ;;
  e_b)      BASE=$M TAG=agent_untrained SETS=val,test python $C/sql_agent_eval.py ;;
  e_grpo)   BASE=$M OUT=$ER/grpo STEPS=150 python $C/sql_agent_grpo.py ;;
  e_ck)     BASE=$M ADAPTER=$ER/grpo/$2 TAG=agent_grpo__$2 SETS=val python $C/sql_agent_eval.py ;;
  e_best)   BASE=$M ADAPTER=$ER/grpo/$2 TAG=agent_grpo__$2 SETS=val,test python $C/sql_agent_eval.py ;;
  e_smoke)  SM=$O/agent_smoke; rm -rf $SM; mkdir -p $SM
            QFILE=$QF MODE=sft BASE=$BASE OUT=$SM/sft STEPS=6 N_TRAIN=40 python $C/sql_train.py 2>&1 | grep -E "TRAIN DONE|Error" | tail -2
            python $C/merge.py $BASE $SM/sft/final $SM/merged 2>&1 | tail -1
            BASE=$SM/merged TAG=smk_untrained SETS=val N=12 python $C/sql_agent_eval.py 2>&1 | grep -E "exec_acc|Error" | tail -2
            BASE=$SM/merged OUT=$SM/grpo STEPS=2 B=2 G=4 N_TRAIN=16 python $C/sql_agent_grpo.py 2>&1 | grep -E "'step'|TRAIN DONE|Error|error" | tail -4
            BASE=$SM/merged ADAPTER=$SM/grpo/final TAG=smk_grpo SETS=val N=12 python $C/sql_agent_eval.py 2>&1 | grep -E "exec_acc|Error" | tail -2
            echo E_SMOKE_OK ;;
esac
